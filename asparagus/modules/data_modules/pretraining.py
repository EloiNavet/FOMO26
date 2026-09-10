import csv
import json
import lightning as pl
import logging
import math
import os
import random
import torch
import torch.distributed as dist
from asparagus.functional.collate import collate_return
from asparagus.functional.scanner_targets import ScannerTargetConfig, ScannerTargetEncoder
from asparagus.modules.datasets.PretrainDataset import PretrainDataset
from asparagus.modules.datasets.TrainDataset import SingleSubjectPredictDataset
from asparagus.modules.lightning_modules.ssl import schedules as ssl_schedules
from asparagus.modules.transforms.presets import pretrain_CPU_train_transforms, pretrain_CPU_val_transforms
from collections import Counter, defaultdict
from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler, Sampler, Subset
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import Compose
from typing import Literal, Optional, Sequence


def _is_rank_zero_process() -> bool:
    return os.environ.get("RANK", os.environ.get("LOCAL_RANK", os.environ.get("SLURM_PROCID", "0"))) == "0"


def _distributed_sampler_context(seed: int) -> tuple[int, int, int]:
    if not (dist.is_available() and dist.is_initialized()):
        return 1, 0, int(seed)
    num_replicas = dist.get_world_size()
    rank = dist.get_rank()
    shared_seed = [int(seed) if rank == 0 else None]
    dist.broadcast_object_list(shared_seed, src=0)
    return num_replicas, rank, int(shared_seed[0])


def pretrain_collate(batch):
    metadata = [item.pop("metadata", {}) for item in batch]
    # raw_image is the untransformed, variable-size volume used as the demographic base view.
    # It must not go through default_collate (torch.stack on heterogeneous spatial shapes crashes);
    # keep it as a per-sample list. demo_cpu_transforms (a val-transform preset: pad + center-crop to
    # patch_size) resize each sample before stacking in ssl/views.apply_demo_cpu_transforms_to_image.
    has_raw_image = any("raw_image" in item for item in batch)
    raw_images = [item.pop("raw_image", None) for item in batch] if has_raw_image else None
    collated = default_collate(batch)
    collated["metadata"] = metadata
    if has_raw_image:
        collated["raw_image"] = raw_images
    return collated


class SameSessionMultimodalSampler(Sampler):
    MULTIMODAL_BATCH_MODES = {"single_session", "packed_pairs"}

    def __init__(
        self,
        dataset: PretrainDataset,
        batch_size: int,
        num_samples: int,
        multimodal_probability: float = 0.5,
        demographic_probability: float = 0.0,
        demographic_modalities: Optional[Sequence[int]] = None,
        demographic_tokens: Optional[Sequence[dict]] = None,
        demographic_age_bin_years: int = 10,
        max_modalities_per_subject: int = 4,
        stage1_multimodal_batch_mode: str = "single_session",
        stage1_multimodal_allowed_pairs: Optional[Sequence[str | Sequence[int | str]]] = None,
        registered_batch_schedule=None,
        accumulate_grad_batches: int = 1,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_samples = int(num_samples)
        self.multimodal_probability = float(multimodal_probability)
        self.demographic_probability = float(demographic_probability)
        # Demographic candidate sets are keyed by *token* (structural modality or DWI b-value
        # subtype). Tokens carry an optional b-value range so DWI subtypes never pool b-values.
        self.demographic_tokens = self._normalize_demographic_tokens(demographic_tokens, demographic_modalities)
        self.demographic_modalities = tuple(dict.fromkeys(token["modality_id"] for token in self.demographic_tokens))
        for name, probability in (
            ("multimodal_probability", self.multimodal_probability),
            ("demographic_probability", self.demographic_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {probability}.")
        total_probability = self.multimodal_probability + self.demographic_probability
        if total_probability > 1.0 + 1e-8:
            raise ValueError(
                f"multimodal_probability and demographic_probability must sum to at most 1.0, got {total_probability}."
            )
        self.demographic_age_bin_years = max(1, int(demographic_age_bin_years))
        self.max_modalities_per_subject = int(max_modalities_per_subject)
        self.stage1_multimodal_batch_mode = str(stage1_multimodal_batch_mode)
        if self.stage1_multimodal_batch_mode not in self.MULTIMODAL_BATCH_MODES:
            supported = ", ".join(sorted(self.MULTIMODAL_BATCH_MODES))
            raise ValueError(
                f"stage1_multimodal_batch_mode must be one of {{{supported}}}, got {self.stage1_multimodal_batch_mode!r}."
            )
        self.stage1_multimodal_allowed_pairs = self._normalize_modality_pairs(stage1_multimodal_allowed_pairs)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        if self.num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {self.num_replicas}.")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(f"rank must be in [0, {self.num_replicas}), got {self.rank}.")
        self.num_samples_per_rank = math.ceil(self.num_samples / self.num_replicas)
        self._iteration = 0
        self._resume_iteration = None
        self._resume_position = 0
        self._active_iteration = None
        self._position = 0
        self._consumed_position = 0
        self._track_consumption = False
        dataset_len = len(dataset) if hasattr(dataset, "__len__") else len(dataset.files)
        self._all_indices = list(range(dataset_len))
        self._groups = self._build_multimodal_groups(dataset)
        self._pair_groups = self._build_multimodal_pair_groups(
            dataset,
            allowed_pairs=self.stage1_multimodal_allowed_pairs,
        )
        self._demographic_strata_by_token = self._build_demographic_subject_strata(
            dataset,
            tokens=self.demographic_tokens,
            age_bin_years=self.demographic_age_bin_years,
        )
        self.global_batch_size = self.batch_size * self.num_replicas

        # ------------------------------------------------------------------------------------------
        # Registered-batch curriculum (Task 6). Absent by default: with no schedule the iterator
        # draws exactly the same random numbers in exactly the same order as before, so every
        # historical run — including every frozen Task-5 arm — samples identically.
        # ------------------------------------------------------------------------------------------
        self.registered_batch_schedule = registered_batch_schedule
        self.accumulate_grad_batches = max(1, int(accumulate_grad_batches))
        self.batches_per_iteration = max(1, math.ceil(self.num_samples_per_rank / self.batch_size))
        self._batch_type_counts = {"multimodal": 0, "demographic": 0, "general": 0}
        self._registered_groups: list[list[int]] = []
        self._registered_pair_groups: list = []
        self.batch_mode_counts: dict[str, int] = {}
        self.batch_mode_log: list[tuple[int, str]] = []
        self.registered_fallback_count = 0
        self.last_batch_mode: str | None = None
        self.last_registered_probability = 0.0
        if self.registered_batch_schedule is not None:
            self._registered_groups = self._build_multimodal_groups(dataset, registered_only=True)
            self._registered_pair_groups = self._build_multimodal_pair_groups(
                dataset,
                allowed_pairs=self.stage1_multimodal_allowed_pairs,
                registered_only=True,
            )
            support = (
                self._registered_pair_groups
                if self.stage1_multimodal_batch_mode == "packed_pairs"
                else self._registered_groups
            )
            # A registered curriculum with no registered support would silently degrade into a
            # general-only run that still claims to be testing Stage 2. Refuse it instead.
            if float(getattr(self.registered_batch_schedule, "weight", 0.0)) > 0.0 and not support:
                raise ValueError(
                    "registered_batch_schedule has positive weight but the split contains no registered "
                    f"same-session multimodal support in mode {self.stage1_multimodal_batch_mode!r}. "
                    "Provide a registered subset or set the schedule weight to zero."
                )
        if self.demographic_probability > 0.0:
            missing = [
                token["name"] for token in self.demographic_tokens if token["name"] not in self._demographic_strata_by_token
            ]
            if missing:
                raise ValueError(
                    f"Demographic sampling found no eligible control subjects with valid age and sex for tokens {missing}."
                )
            insufficient = {
                token["name"]: self._demographic_unique_subject_count(self._demographic_strata_by_token[token["name"]])
                for token in self.demographic_tokens
                if self._demographic_unique_subject_count(self._demographic_strata_by_token[token["name"]])
                < self.global_batch_size
            }
            if insufficient:
                raise ValueError(
                    "Each demographic token must provide at least one unique eligible subject per global batch "
                    f"(required={self.global_batch_size}, available={insufficient}). Reduce the per-rank microbatch "
                    "size or use a split with more eligible subjects; subject replacement is intentionally disabled."
                )

    def __len__(self):
        return self.num_samples_per_rank

    def set_epoch(self, epoch: int):
        if self._resume_iteration is not None and int(epoch) == self._resume_iteration:
            self._iteration = int(epoch)
            return
        self._iteration = int(epoch)
        self._resume_iteration = None
        self._resume_position = 0

    def state_dict(self) -> dict:
        iteration = self._active_iteration if self._active_iteration is not None else self._iteration
        active_position = self._consumed_position if self._track_consumption else self._position
        position = active_position if self._active_iteration is not None else 0
        return {"iteration": int(iteration), "position": int(position)}

    def enable_consumption_tracking(self) -> None:
        """Checkpoint consumed samples, not worker-prefetched samples."""
        self._track_consumption = True

    def mark_consumed(self, count: int) -> None:
        if self._active_iteration is not None:
            self._consumed_position = min(self._position, self._consumed_position + max(0, int(count)))

    def load_state_dict(self, state_dict: dict) -> None:
        self._resume_iteration = int(state_dict.get("iteration", 0))
        self._resume_position = max(0, int(state_dict.get("position", 0)))
        self._iteration = self._resume_iteration

    def optimizer_step_for(self, iteration: int, batch_ordinal: int) -> int:
        """The optimizer step a given batch belongs to.

        Derived from the sampler's own deterministic position rather than read from the trainer,
        because the sampler runs inside dataloader workers that prefetch ahead of the training loop
        and therefore cannot observe ``trainer.global_step``.  Counting the same way the trainer does
        - batches divided by the accumulation factor - gives every rank the same answer for the same
        batch, with no communication and nothing to restore on resume.
        """
        batch_index = int(iteration) * self.batches_per_iteration + int(batch_ordinal)
        return batch_index // self.accumulate_grad_batches

    def registered_probability(self, optimizer_step: int) -> float:
        """Effective registered-batch probability at ``optimizer_step``, clamped to ``[0, 1]``."""
        if self.registered_batch_schedule is None:
            return 0.0
        value = ssl_schedules.effective_weight(self.registered_batch_schedule, int(optimizer_step))
        return min(max(float(value), 0.0), 1.0)

    def batch_type_counts(self) -> dict:
        """Realized counts of multimodal / demographic / general batches produced so far."""
        return dict(getattr(self, "_batch_type_counts", {"multimodal": 0, "demographic": 0, "general": 0}))

    def realized_registered_fraction(self) -> float:
        """Fraction of the batches produced so far that were genuinely registered batches."""
        total = sum(self.batch_mode_counts.values())
        if not total:
            return 0.0
        registered = sum(
            count
            for mode, count in self.batch_mode_counts.items()
            if mode.startswith("registered_") and mode != "registered_fallback_general"
        )
        return registered / total

    def _sample_registered_batch(self, rng: random.Random) -> list[int]:
        """Draw one batch restricted to the registered subset; empty when no support exists."""
        if self.stage1_multimodal_batch_mode == "packed_pairs":
            if not self._registered_pair_groups:
                return []
            return self._sample_packed_multimodal_batch(rng, pair_groups=self._registered_pair_groups)
        if not self._registered_groups:
            return []
        group = rng.choice(self._registered_groups)
        count = min(self.batch_size, self.max_modalities_per_subject, len(group))
        return rng.sample(group, count)

    def __iter__(self):
        iteration = self._resume_iteration if self._resume_iteration is not None else self._iteration
        resume_position = self._resume_position if self._resume_iteration is not None else 0
        self._resume_iteration = None
        self._resume_position = 0
        self._active_iteration = int(iteration)
        self._position = 0
        self._consumed_position = int(resume_position)
        schedule_rng = random.Random(self.seed + iteration)
        sample_rng = random.Random(self.seed + 1_000_003 * (self.rank + 1) + iteration)
        demographic_rng = random.Random(self.seed + 2_000_003 + iteration)
        self._iteration = int(iteration) + 1
        registered_rng = random.Random(self.seed + 3_000_003 + iteration)
        self.batch_mode_counts = {}
        self.batch_mode_log = []
        self.registered_fallback_count = 0
        yielded = 0
        batch_ordinal = 0
        while yielded < self.num_samples_per_rank:
            batch_indices = []
            optimizer_step = self.optimizer_step_for(iteration, batch_ordinal)
            mode = None

            # The registered curriculum is decided first, from a dedicated RNG shared by every rank
            # and keyed only on (seed, iteration). It is consulted *only* when a schedule exists, so
            # a run without one draws the identical sequence it always did.
            if self.registered_batch_schedule is not None:
                probability = self.registered_probability(optimizer_step)
                self.last_registered_probability = probability
                if probability > 0.0 and registered_rng.random() < probability:
                    batch_indices.extend(self._sample_registered_batch(sample_rng))
                    if batch_indices:
                        mode = (
                            "registered_packed"
                            if self.stage1_multimodal_batch_mode == "packed_pairs"
                            else "registered_single_session"
                        )
                    else:
                        # No registered batch could be formed: record the fallback explicitly rather
                        # than letting a general batch masquerade as a registered one.
                        self.registered_fallback_count += 1
                        mode = "registered_fallback_general"

            if mode is None:
                # All ranks share the schedule and demographic RNGs. A demographic
                # step therefore selects one modality and one unique global subject
                # batch before each rank consumes its disjoint shard.
                roll = schedule_rng.random()
                p1 = self.multimodal_probability
                p2 = p1 + self.demographic_probability
                if roll < p1:
                    if self.stage1_multimodal_batch_mode == "packed_pairs":
                        batch_indices.extend(self._sample_packed_multimodal_batch(sample_rng))
                    elif self._groups:
                        group = sample_rng.choice(self._groups)
                        count = min(self.batch_size, self.max_modalities_per_subject, len(group))
                        batch_indices.extend(sample_rng.sample(group, count))
                    mode = "general_multimodal"
                elif self._demographic_strata_by_token and roll < p2:
                    global_batch = self._sample_demographic_global_batch(demographic_rng)
                    start = self.rank * self.batch_size
                    batch_indices.extend(global_batch[start : start + self.batch_size])
                    mode = "general_demographic"
                else:
                    mode = "general"

            self.last_batch_mode = mode
            self.batch_mode_counts[mode] = self.batch_mode_counts.get(mode, 0) + 1
            # Coarse realized batch type, counted separately from the fine-grained mode. A routing
            # decision that finds no eligible group silently degrades to a general batch, so the
            # REALIZED type is recorded rather than the intended one -- configured probabilities
            # are intentions, and a stacked recipe needs what was actually served.
            realized = "general"
            if mode == "general_multimodal" and batch_indices:
                realized = "multimodal"
            elif mode == "general_demographic" and batch_indices:
                realized = "demographic"
            self._batch_type_counts[realized] = self._batch_type_counts.get(realized, 0) + 1
            self.batch_mode_log.append((optimizer_step, mode))
            batch_ordinal += 1

            while len(batch_indices) < self.batch_size:
                batch_indices.append(sample_rng.choice(self._all_indices))

            for index in batch_indices[: self.batch_size]:
                if yielded >= self.num_samples_per_rank:
                    break
                yielded += 1
                self._position = yielded
                if yielded <= resume_position:
                    continue
                yield index
        self._active_iteration = None
        self._position = 0
        self._consumed_position = 0

    @staticmethod
    def _normalize_modality_pairs(
        pairs: Optional[Sequence[str | Sequence[int | str]]],
    ) -> tuple[tuple[int, int], ...] | None:
        if pairs is None:
            return None
        normalized = []
        for pair in pairs:
            if isinstance(pair, str):
                parts = [part.strip() for part in pair.split("-") if part.strip()]
            else:
                parts = list(pair)
            if len(parts) != 2:
                raise ValueError(
                    "Stage 1 multimodal allowed pairs must be two-modality entries such as 't1w-t2w' or ('t1w', 'flair')."
                )
            modality_ids = PretrainDataset.normalize_modality_ids(parts, default=())
            if len(modality_ids) != 2 or modality_ids[0] == modality_ids[1]:
                raise ValueError(f"Invalid same-session multimodal pair {pair!r}.")
            pair_key = tuple(sorted((int(modality_ids[0]), int(modality_ids[1]))))
            if pair_key not in normalized:
                normalized.append(pair_key)
        if not normalized:
            raise ValueError("stage1_multimodal_allowed_pairs was provided but contains no valid modality pairs.")
        return tuple(normalized)

    @staticmethod
    def _format_modality_pairs(pairs: Optional[Sequence[tuple[int, int]]]) -> str:
        if pairs is None:
            return "all known modality pairs"
        return ", ".join(
            f"{PretrainDataset.modality_name(left)}-{PretrainDataset.modality_name(right)}" for left, right in pairs
        )

    @staticmethod
    def _build_multimodal_groups(dataset: PretrainDataset, registered_only: bool = False) -> list[list[int]]:
        groups = {}
        modalities = {}
        for idx, cached in enumerate(dataset._cached_metadata):
            identity = cached["identity"]
            if registered_only and not identity["is_registered_subset"]:
                continue
            key = identity["subject_session_key"]
            modality_id = identity["modality_id"]
            if modality_id < 0:
                continue
            groups.setdefault(key, []).append(idx)
            modalities.setdefault(key, set()).add(modality_id)

        return [indices for key, indices in groups.items() if len(indices) > 1 and len(modalities[key]) > 1]

    @staticmethod
    def _build_multimodal_pair_groups(
        dataset: PretrainDataset,
        allowed_pairs: Optional[Sequence[tuple[int, int]]] = None,
        registered_only: bool = False,
    ) -> list[dict]:
        by_session = defaultdict(lambda: defaultdict(list))
        for idx, cached in enumerate(dataset._cached_metadata):
            identity = cached["identity"]
            if registered_only and not identity["is_registered_subset"]:
                continue
            modality_id = int(identity["modality_id"])
            if modality_id < 0:
                continue
            by_session[identity["subject_session_key"]][modality_id].append(idx)

        pair_groups = []
        allowed = {tuple(sorted(pair)) for pair in allowed_pairs} if allowed_pairs is not None else None
        for session_key, indices_by_modality in sorted(by_session.items()):
            modality_ids = sorted(indices_by_modality)
            if allowed is None:
                candidate_pairs = [
                    (modality_ids[left], modality_ids[right])
                    for left in range(len(modality_ids))
                    for right in range(left + 1, len(modality_ids))
                ]
            else:
                candidate_pairs = [
                    pair for pair in sorted(allowed) if pair[0] in indices_by_modality and pair[1] in indices_by_modality
                ]
            for left, right in candidate_pairs:
                pair_groups.append(
                    {
                        "subject_session_key": session_key,
                        "modalities": (left, right),
                        "indices": (
                            sorted(indices_by_modality[left], key=lambda index: dataset.files[index]),
                            sorted(indices_by_modality[right], key=lambda index: dataset.files[index]),
                        ),
                    }
                )
        return pair_groups

    def _sample_packed_multimodal_batch(self, rng: random.Random, *, pair_groups=None) -> list[int]:
        """Pack ``batch_size // 2`` same-session modality pairs.

        ``pair_groups`` restricts the support, which is how the registered curriculum draws a
        registered-only packed batch without duplicating any of this logic.
        """
        available = self._pair_groups if pair_groups is None else pair_groups
        if not available or self.batch_size < 2:
            return []

        target_pairs = self.batch_size // 2
        pair_groups = list(available)
        rng.shuffle(pair_groups)
        batch_indices = []
        selected_sessions = set()

        for group in pair_groups:
            if len(batch_indices) + 2 > self.batch_size:
                break
            session_key = group["subject_session_key"]
            if session_key in selected_sessions and len(pair_groups) >= target_pairs:
                continue
            left_indices, right_indices = group["indices"]
            batch_indices.extend([rng.choice(left_indices), rng.choice(right_indices)])
            selected_sessions.add(session_key)

        # If support is smaller than the requested 8x2 geometry, make the fallback explicit:
        # reuse eligible pairs first, then let the outer random filler complete any odd slot.
        while len(batch_indices) + 2 <= self.batch_size and available:
            group = rng.choice(available)
            left_indices, right_indices = group["indices"]
            batch_indices.extend([rng.choice(left_indices), rng.choice(right_indices)])

        rng.shuffle(batch_indices)
        return batch_indices[: self.batch_size]

    @staticmethod
    def _build_demographic_indices(
        dataset: PretrainDataset,
        modality_ids: Optional[set[int]] = None,
        tokens: Optional[Sequence[dict]] = None,
    ) -> list[int]:
        """Eligible demographic scan indices (control + finite age + known sex).

        When ``tokens`` is given, a scan is eligible if it matches any token: its modality id
        plus, for DWI b-value subtypes, a finite b-value inside the token's range. ``tokens``
        takes precedence over ``modality_ids`` (the legacy structural-only filter).
        """
        indices = []
        for idx, cached in enumerate(dataset._cached_metadata):
            identity = cached["identity"]
            common = cached["common"]
            if common["pathology"] != 0 or math.isnan(common["age"]) or common["sex"] == -1:
                continue
            if tokens is not None:
                if not any(SameSessionMultimodalSampler._scan_matches_token(identity, token) for token in tokens):
                    continue
            elif modality_ids is not None and int(identity["modality_id"]) not in modality_ids:
                continue
            indices.append(idx)
        return indices

    @staticmethod
    def _scan_matches_token(identity: dict, token: dict) -> bool:
        """Whether a scan identity belongs to a demographic token (modality + optional b-value)."""
        if int(identity["modality_id"]) != int(token["modality_id"]):
            return False
        bval_min, bval_max = token.get("bval_min"), token.get("bval_max")
        if bval_min is None or bval_max is None:
            return True
        bval = identity.get("dwi_bval")
        if bval is None or (isinstance(bval, float) and math.isnan(bval)):
            return False
        return float(bval_min) <= float(bval) <= float(bval_max)

    @staticmethod
    def _build_demographic_subject_strata(
        dataset: PretrainDataset,
        tokens: Optional[Sequence[dict]] = None,
        age_bin_years: int = 10,
    ) -> dict[str, dict[tuple, dict[str, list[int]]]]:
        # Strata keyed by token *name* so DWI subtypes (which share the dwi modality id) get
        # disjoint subject pools and are never mixed inside one demographic global batch.
        tokens = list(tokens or [])
        strata_by_token = {token["name"]: defaultdict(lambda: defaultdict(list)) for token in tokens}
        age_bin_years = max(1, int(age_bin_years))
        for idx, cached in enumerate(dataset._cached_metadata):
            identity = cached["identity"]
            common = cached["common"]
            if common["pathology"] != 0 or int(identity["modality_id"]) < 0:
                continue
            if math.isnan(common["age"]) or common["sex"] == -1:
                continue
            age_bin = int(common["age"] // age_bin_years)
            key = (int(common["sex"]), age_bin)
            for token in tokens:
                if SameSessionMultimodalSampler._scan_matches_token(identity, token):
                    strata_by_token[token["name"]][key][identity["subject_key"]].append(idx)
        return {
            name: {
                key: {
                    subject: sorted(indices, key=lambda index: dataset.files[index]) for subject, indices in subjects.items()
                }
                for key, subjects in strata.items()
            }
            for name, strata in strata_by_token.items()
            if strata
        }

    @staticmethod
    def _normalize_demographic_tokens(
        tokens: Optional[Sequence[dict]],
        modalities: Optional[Sequence[int]],
    ) -> tuple[dict, ...]:
        """Normalize demographic tokens; fall back to structural tokens from modality ids."""
        if tokens:
            return tuple(
                {
                    "name": str(token["name"]),
                    "modality_id": int(token["modality_id"]),
                    "bval_min": None if token.get("bval_min") is None else float(token["bval_min"]),
                    "bval_max": None if token.get("bval_max") is None else float(token["bval_max"]),
                }
                for token in tokens
            )
        return tuple(
            {
                "name": PretrainDataset.modality_name(modality_id),
                "modality_id": int(modality_id),
                "bval_min": None,
                "bval_max": None,
            }
            for modality_id in PretrainDataset.normalize_modality_ids(modalities)
        )

    @staticmethod
    def _demographic_unique_subject_count(strata: dict[tuple, dict[str, list[int]]]) -> int:
        return len({subject for subjects in strata.values() for subject in subjects})

    def _sample_demographic_global_batch(self, rng: random.Random) -> list[int]:
        token = rng.choice(self.demographic_tokens)
        strata = self._demographic_strata_by_token[token["name"]]
        keys = sorted(strata)
        start = rng.randrange(len(keys))
        ordered_keys = keys[start:] + keys[:start]
        batch_indices = []
        selected_subjects = set()
        key_cursor = 0
        while len(batch_indices) < self.global_batch_size:
            key = ordered_keys[key_cursor % len(ordered_keys)]
            available_subjects = sorted(set(strata[key]) - selected_subjects)
            if available_subjects:
                subject = rng.choice(available_subjects)
                selected_subjects.add(subject)
                batch_indices.append(rng.choice(strata[key][subject]))
            key_cursor += 1
            if key_cursor > len(keys) * self.global_batch_size and len(batch_indices) < self.global_batch_size:
                raise RuntimeError(f"Could not construct a unique demographic global batch for token {token['name']!r}.")
        rng.shuffle(batch_indices)
        return batch_indices


class StatefulReplacementSampler(Sampler):
    """Replacement sampler with checkpointable epoch/position and rank-local RNG."""

    def __init__(self, data_source, num_samples: int, seed: int = 0, num_replicas: int = 1, rank: int = 0):
        self.data_source = data_source
        self.num_samples = int(math.ceil(int(num_samples) / int(num_replicas)))
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self._iteration = 0
        self._resume_iteration = None
        self._resume_position = 0
        self._active_iteration = None
        self._position = 0
        self._consumed_position = 0
        self._track_consumption = False

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        if self._resume_iteration is not None and int(epoch) == self._resume_iteration:
            self._iteration = int(epoch)
            return
        self._iteration = int(epoch)
        self._resume_iteration = None
        self._resume_position = 0

    def state_dict(self) -> dict:
        iteration = self._active_iteration if self._active_iteration is not None else self._iteration
        active_position = self._consumed_position if self._track_consumption else self._position
        position = active_position if self._active_iteration is not None else 0
        return {"iteration": int(iteration), "position": int(position)}

    def enable_consumption_tracking(self) -> None:
        """Checkpoint consumed samples, not worker-prefetched samples."""
        self._track_consumption = True

    def mark_consumed(self, count: int) -> None:
        if self._active_iteration is not None:
            self._consumed_position = min(self._position, self._consumed_position + max(0, int(count)))

    def load_state_dict(self, state_dict: dict) -> None:
        self._resume_iteration = int(state_dict.get("iteration", 0))
        self._resume_position = max(0, int(state_dict.get("position", 0)))
        self._iteration = self._resume_iteration

    def __iter__(self):
        iteration = self._resume_iteration if self._resume_iteration is not None else self._iteration
        resume_position = self._resume_position if self._resume_iteration is not None else 0
        self._resume_iteration = None
        self._resume_position = 0
        self._active_iteration = int(iteration)
        self._position = 0
        self._consumed_position = int(resume_position)
        self._iteration = int(iteration) + 1
        generator = torch.Generator()
        generator.manual_seed(self.seed + 1_000_003 * (self.rank + 1) + int(iteration))
        position = 0
        while position < self.num_samples:
            chunk_size = min(4096, self.num_samples - position)
            indices = torch.randint(len(self.data_source), (chunk_size,), generator=generator).tolist()
            for index in indices:
                position += 1
                self._position = position
                if position <= resume_position:
                    continue
                yield index
        self._active_iteration = None
        self._position = 0
        self._consumed_position = 0


class PretrainDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        train_split: list,
        val_split: list,
        predict_samples: Optional[list] = [],
        train_transforms: Optional[Compose] = pretrain_CPU_train_transforms,
        val_transforms: Optional[Compose] = pretrain_CPU_val_transforms,
        predict_transforms: Optional[Compose] = None,
        num_samples: Optional[int] = None,
        metadata_paths: Optional[Sequence[str]] = None,
        registered_mapping_path: Optional[str] = None,
        registered_only: bool = False,
        registered_batch_schedule=None,
        accumulate_grad_batches: int = 1,
        same_session_multimodal_batches: bool = False,
        multimodal_batch_probability: float = 0.5,
        demographic_batch_probability: float = 0.0,
        demographic_modalities: Optional[Sequence[int]] = None,
        demographic_tokens: Optional[Sequence[dict]] = None,
        demographic_age_bin_years: int = 10,
        max_modalities_per_subject: int = 4,
        stage1_multimodal_batch_mode: str = "single_session",
        stage1_multimodal_allowed_pairs: Optional[Sequence[str | Sequence[int | str]]] = None,
        sampler_seed: int = 0,
        validate_split_disjointness: bool = True,
        complete_validation: bool = False,
        routine_scan_count: int = 128,
        monitor_seed: int = 0,
        probe_train_max_subjects: int = 512,
        demographic_probe_train_max_subjects: int = 512,
        probe_val_canonical_t1w: bool = True,
        stage1_pair_session_count: int = 64,
        require_demographic_samples: bool = False,
        require_multimodal_samples: bool = False,
        require_registered_multimodal_samples: bool = False,
        return_raw_image: bool = False,
        scanner_targets_enabled: bool = False,
        scanner_target_keys: Optional[Sequence[str]] = None,
        scanner_target_ignore_index: int = -100,
        scanner_target_spacing_bins: Optional[Sequence[float]] = None,
        scanner_target_spacing_summary: str = "max",
        scanner_target_spacing_columns: Optional[Sequence[str]] = None,
        save_scanner_target_vocab: bool = True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.num_samples = num_samples
        self.predict_transforms = predict_transforms
        self.predict_samples = predict_samples
        self.metadata_paths = list(metadata_paths) if metadata_paths is not None else None
        self.registered_mapping_path = registered_mapping_path
        self.registered_only = registered_only
        # Task-6 registered-batch curriculum. None by default, in which case the sampler draws the
        # identical random sequence it always did.
        self.registered_batch_schedule = registered_batch_schedule
        self.accumulate_grad_batches = max(1, int(accumulate_grad_batches))
        self.same_session_multimodal_batches = same_session_multimodal_batches
        self.multimodal_batch_probability = multimodal_batch_probability
        self.demographic_batch_probability = demographic_batch_probability
        self.demographic_tokens = SameSessionMultimodalSampler._normalize_demographic_tokens(
            demographic_tokens, demographic_modalities
        )
        self.demographic_modalities = tuple(dict.fromkeys(token["modality_id"] for token in self.demographic_tokens))
        self.demographic_age_bin_years = int(demographic_age_bin_years)
        self.max_modalities_per_subject = max_modalities_per_subject
        self.stage1_multimodal_batch_mode = str(stage1_multimodal_batch_mode)
        if self.stage1_multimodal_batch_mode not in SameSessionMultimodalSampler.MULTIMODAL_BATCH_MODES:
            supported = ", ".join(sorted(SameSessionMultimodalSampler.MULTIMODAL_BATCH_MODES))
            raise ValueError(
                f"stage1_multimodal_batch_mode must be one of {{{supported}}}, got {self.stage1_multimodal_batch_mode!r}."
            )
        self.stage1_multimodal_allowed_pairs = SameSessionMultimodalSampler._normalize_modality_pairs(
            stage1_multimodal_allowed_pairs
        )
        self.sampler_seed = int(sampler_seed)
        self.validate_split_disjointness = bool(validate_split_disjointness)
        self.complete_validation = bool(complete_validation)
        self.routine_scan_count = int(routine_scan_count)
        self.monitor_seed = int(monitor_seed)
        self.probe_train_max_subjects = int(probe_train_max_subjects)
        self.demographic_probe_train_max_subjects = int(demographic_probe_train_max_subjects)
        self.probe_val_canonical_t1w = bool(probe_val_canonical_t1w)
        self.stage1_pair_session_count = int(stage1_pair_session_count)
        self.require_demographic_samples = bool(require_demographic_samples)
        self.require_multimodal_samples = bool(require_multimodal_samples)
        self.require_registered_multimodal_samples = bool(require_registered_multimodal_samples)
        self.return_raw_image = bool(return_raw_image)
        self.scanner_targets_enabled = bool(scanner_targets_enabled)
        self.scanner_target_config = ScannerTargetConfig(
            keys=tuple(scanner_target_keys or ("manufacturer", "field_strength", "spacing_bin")),
            ignore_index=int(scanner_target_ignore_index),
            spacing_bins=tuple(float(value) for value in (scanner_target_spacing_bins or (1.0, 1.5, 2.0, 3.0))),
            spacing_summary=str(scanner_target_spacing_summary),
            spacing_columns=tuple(
                scanner_target_spacing_columns
                or ("original_spacing", "pixdim", "source_spacing", "native_spacing", "voxel_spacing")
            ),
        )
        self.scanner_spacing_metadata_path = (
            self._resolve_scanner_spacing_metadata_path(
                self.metadata_paths,
                self.scanner_target_config.spacing_columns,
            )
            if self.scanner_targets_enabled
            else None
        )
        self.save_scanner_target_vocab = bool(save_scanner_target_vocab)
        self._train_sampler = None
        self._pending_train_sampler_state = None
        self.scanner_target_encoder: Optional[ScannerTargetEncoder] = None
        self.scanner_target_class_counts = {target: 1 for target in self.scanner_target_config.keys}
        self.scanner_target_vocabs = {}
        for name, probability in (
            ("multimodal_batch_probability", self.multimodal_batch_probability),
            ("demographic_batch_probability", self.demographic_batch_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {probability}.")
        total_probability = self.multimodal_batch_probability + self.demographic_batch_probability
        if total_probability > 1.0 + 1e-8:
            raise ValueError(
                "multimodal_batch_probability and demographic_batch_probability "
                f"must sum to at most 1.0, got {total_probability}."
            )
        if (self.require_multimodal_samples or self.require_registered_multimodal_samples) and (
            not self.same_session_multimodal_batches or self.multimodal_batch_probability <= 0.0
        ):
            raise ValueError(
                "Multimodal SSL objectives require same_session_multimodal_batches=true and "
                "multimodal_batch_probability > 0 so objective-active batches are sampled intentionally."
            )

        if _is_rank_zero_process():
            logging.info(f"Using {self.num_workers} workers")

    @staticmethod
    def _resolve_scanner_spacing_metadata_path(
        metadata_paths: Optional[Sequence[str]],
        spacing_columns: Sequence[str],
    ) -> Optional[str]:
        wanted = {str(column).strip().lower() for column in spacing_columns}
        candidates = set()
        for value in metadata_paths or ():
            path = Path(value)
            if not path.is_file():
                continue
            with path.open(newline="") as stream:
                header = next(csv.reader(stream, delimiter="\t"), [])
            normalized_header = {str(column).strip().lower() for column in header}
            if wanted & normalized_header:
                return None
            sibling_manifest = path.with_name("manifest.tsv")
            if sibling_manifest.is_file():
                candidates.add(sibling_manifest.resolve())
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous native-spacing manifests beside metadata paths: {sorted(candidates)}")
        return str(next(iter(candidates))) if candidates else None

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]):
        if stage in {"fit", "validate"}:
            self.setup_fit()
        elif stage == "test":
            raise NotImplementedError("Test stage not supported for PretrainModule.")
        elif stage == "predict":
            self.setup_predict()

    def state_dict(self) -> dict:
        state = {}
        if self._train_sampler is not None and hasattr(self._train_sampler, "state_dict"):
            state["train_sampler"] = self._train_sampler.state_dict()
        elif self._pending_train_sampler_state is not None:
            state["train_sampler"] = dict(self._pending_train_sampler_state)
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        sampler_state = dict(state_dict.get("train_sampler", {}) or {})
        if self._train_sampler is not None and hasattr(self._train_sampler, "load_state_dict"):
            self._train_sampler.load_state_dict(sampler_state)
        else:
            self._pending_train_sampler_state = sampler_state or None

    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        sampler = self._train_sampler
        trainer = getattr(self, "trainer", None)
        if sampler is not None and getattr(trainer, "training", False) and hasattr(sampler, "mark_consumed"):
            if isinstance(batch, dict) and isinstance(batch.get("image"), torch.Tensor):
                sampler.mark_consumed(batch["image"].shape[0])
        return batch

    def setup_fit(self):
        if getattr(self, "_fit_setup_complete", False):
            return
        self.train_dataset = PretrainDataset(
            self.train_split,
            transforms=self.train_transforms,
            metadata_paths=self.metadata_paths,
            registered_mapping_path=self.registered_mapping_path,
            registered_only=self.registered_only,
            return_raw_image=self.return_raw_image,
            scanner_spacing_metadata_path=self.scanner_spacing_metadata_path,
        )
        if self.scanner_targets_enabled:
            self.scanner_target_encoder = ScannerTargetEncoder.fit(
                [row["raw"] for row in self.train_dataset._cached_metadata],
                self.scanner_target_config,
            )
            self.train_dataset.set_scanner_target_encoder(self.scanner_target_encoder)
            # A scanner target whose training vocabulary is empty has no usable label at all:
            # every sample encodes to ignore_index, so its head would train on zero targets and
            # contribute nothing while still looking active in the logs. class_counts(minimum=1)
            # would silently turn it into a degenerate 1-class head, so fail loudly instead.
            # (Known case: spacing_bin, whose configured spacing_columns are absent from
            # mri_info.tsv -- only manifest.tsv carries `pixdim`, and that is NATIVE spacing.)
            empty_targets = sorted(name for name, vocab in self.scanner_target_encoder.vocab.items() if not vocab)
            if empty_targets:
                raise ValueError(
                    f"Scanner targets {empty_targets} have no valid label in the training split; "
                    "their heads would train on zero targets. Point "
                    "data.scanner_targets.spacing_columns at a column present in the wired metadata "
                    "(e.g. `pixdim` from manifest.tsv, which is native spacing) or disable "
                    "data.scanner_targets.enabled."
                )
            self.scanner_target_class_counts = self.scanner_target_encoder.class_counts(minimum=1)
            self.scanner_target_vocabs = dict(self.scanner_target_encoder.vocab)
            self._save_scanner_target_vocab()
        self.val_dataset = PretrainDataset(
            self.val_split,
            transforms=self.val_transforms,
            metadata_paths=self.metadata_paths,
            registered_mapping_path=self.registered_mapping_path,
            registered_only=self.registered_only,
            return_raw_image=self.return_raw_image,
            scanner_target_encoder=self.scanner_target_encoder,
            scanner_spacing_metadata_path=self.scanner_spacing_metadata_path,
        )
        if len(self.train_dataset) == 0:
            raise ValueError(
                "PretrainDataModule train dataset is empty after strict metadata and registered subset filtering."
            )
        if len(self.val_dataset) == 0:
            raise ValueError(
                "PretrainDataModule validation dataset is empty after strict metadata and registered subset filtering."
            )
        if self.validate_split_disjointness:
            self._assert_disjoint_fit_splits()
        if self.require_demographic_samples:
            # Per-token eligibility: each configured token (structural modality or DWI b-value
            # subtype) must have eligible controls with valid age+sex in both train and val.
            train_counts = {
                token["name"]: len(SameSessionMultimodalSampler._build_demographic_indices(self.train_dataset, tokens=[token]))
                for token in self.demographic_tokens
            }
            val_counts = {
                token["name"]: len(SameSessionMultimodalSampler._build_demographic_indices(self.val_dataset, tokens=[token]))
                for token in self.demographic_tokens
            }
            missing_train = sorted(name for name, count in train_counts.items() if count == 0)
            missing_val = sorted(name for name, count in val_counts.items() if count == 0)
            if missing_train or missing_val:
                raise ValueError(
                    "Demographic objective requires eligible controls with valid age and sex for every configured "
                    f"token in train and validation; missing_train={missing_train}, missing_val={missing_val}, "
                    f"train_counts={train_counts}, val_counts={val_counts}."
                )
        if self.same_session_multimodal_batches and self.multimodal_batch_probability > 0.0:
            train_multimodal = self._build_stage1_multimodal_support(self.train_dataset)
            if not train_multimodal:
                raise ValueError(
                    "same_session_multimodal_batches requested multimodal sampling, but the training split contains "
                    "no same-subject/session groups with at least two known modalities for "
                    f"stage1_multimodal_batch_mode={self.stage1_multimodal_batch_mode!r} and "
                    f"allowed_pairs={SameSessionMultimodalSampler._format_modality_pairs(self.stage1_multimodal_allowed_pairs)}."
                )
            if (
                self.stage1_multimodal_batch_mode == "packed_pairs"
                and len(train_multimodal) < max(1, self.batch_size // 2)
                and _is_rank_zero_process()
            ):
                logging.warning(
                    "Packed Stage 1 multimodal batches requested %d pairs per batch but only %d eligible pair groups "
                    "were found in the training split; sampler will reuse eligible pairs and random-fill any leftover slot.",
                    max(1, self.batch_size // 2),
                    len(train_multimodal),
                )
        if self.require_multimodal_samples:
            train_multimodal = self._build_stage1_multimodal_support(self.train_dataset)
            val_multimodal = self._build_stage1_multimodal_support(self.val_dataset)
            if not train_multimodal or not val_multimodal:
                allowed_pairs = SameSessionMultimodalSampler._format_modality_pairs(self.stage1_multimodal_allowed_pairs)
                raise ValueError(
                    "Multimodal Stage 1 requires same-subject/session scans with at least two known modalities "
                    "in train and validation for "
                    f"stage1_multimodal_batch_mode={self.stage1_multimodal_batch_mode!r}, "
                    f"allowed_pairs={allowed_pairs}; found train={len(train_multimodal)}, val={len(val_multimodal)}."
                )
        if self.require_registered_multimodal_samples:
            train_registered = SameSessionMultimodalSampler._build_multimodal_groups(self.train_dataset, registered_only=True)
            val_registered = SameSessionMultimodalSampler._build_multimodal_groups(self.val_dataset, registered_only=True)
            if not train_registered or not val_registered:
                raise ValueError(
                    "Multimodal Stage 2 requires registered same-subject/session scans with at least two known modalities "
                    f"in train and validation; found train={len(train_registered)}, val={len(val_registered)}."
                )
        self._build_monitor_datasets()
        self._log_dataset_summary("train", self.train_dataset)
        self._log_dataset_summary("val", self.val_dataset)
        self._log_monitor_summary("routine_validation", self.routine_val_dataset)
        self._log_monitor_summary("probe_reference_train", self.probe_reference_dataset)
        self._log_monitor_summary("probe_reference_modality_train", self.modality_probe_reference_dataset)
        self._log_monitor_summary("probe_reference_demographic_train", self.demographic_probe_reference_dataset)
        self._log_monitor_summary("probe_query_val", self.probe_query_dataset)
        self._log_monitor_summary("stage1_pairs_val", self.stage1_monitor_dataset)
        self._fit_setup_complete = True

    def _save_scanner_target_vocab(self):
        if not (self.save_scanner_target_vocab and _is_rank_zero_process() and self.scanner_target_encoder is not None):
            return
        path = Path.cwd() / "scanner_target_vocab.json"
        try:
            path.write_text(json.dumps(self.scanner_target_encoder.to_dict(), indent=2, sort_keys=True))
            logging.info("Saved scanner target vocabularies to %s", path)
        except OSError as exc:
            logging.warning("Could not save scanner target vocabularies to %s: %s", path, exc)

    def _build_stage1_multimodal_support(self, dataset: PretrainDataset, registered_only: bool = False) -> list:
        if self.stage1_multimodal_batch_mode == "packed_pairs":
            return SameSessionMultimodalSampler._build_multimodal_pair_groups(
                dataset,
                allowed_pairs=self.stage1_multimodal_allowed_pairs,
                registered_only=registered_only,
            )
        return SameSessionMultimodalSampler._build_multimodal_groups(dataset, registered_only=registered_only)

    def _assert_disjoint_fit_splits(self):
        train_subjects = {row["identity"]["subject_key"] for row in self.train_dataset._cached_metadata}
        val_subjects = {row["identity"]["subject_key"] for row in self.val_dataset._cached_metadata}
        train_sessions = {row["identity"]["subject_session_key"] for row in self.train_dataset._cached_metadata}
        val_sessions = {row["identity"]["subject_session_key"] for row in self.val_dataset._cached_metadata}
        subject_overlap = sorted(train_subjects & val_subjects)
        session_overlap = sorted(train_sessions & val_sessions)
        if subject_overlap or session_overlap:
            raise ValueError(
                "Pretraining split leakage detected: "
                f"{len(subject_overlap)} overlapping subjects and {len(session_overlap)} overlapping sessions. "
                "Regenerate splits with --group-by subject."
            )

    def _log_dataset_summary(self, name: str, dataset: PretrainDataset):
        if not _is_rank_zero_process():
            return
        subjects = {row["identity"]["subject_key"] for row in dataset._cached_metadata}
        sessions = {row["identity"]["subject_session_key"] for row in dataset._cached_metadata}
        registered = sum(bool(row["identity"]["is_registered_subset"]) for row in dataset._cached_metadata)
        demographic = SameSessionMultimodalSampler._build_demographic_indices(dataset, tokens=self.demographic_tokens)
        multimodal = SameSessionMultimodalSampler._build_multimodal_groups(dataset)
        stage1_pair_groups = SameSessionMultimodalSampler._build_multimodal_pair_groups(
            dataset,
            allowed_pairs=self.stage1_multimodal_allowed_pairs,
        )
        registered_multimodal = SameSessionMultimodalSampler._build_multimodal_groups(dataset, registered_only=True)
        logging.info(
            "%s split: scans=%d subjects=%d sessions=%d registered_scans=%d "
            "demographic_eligible_scans=%d multimodal_sessions=%d stage1_pair_groups=%d "
            "registered_multimodal_sessions=%d",
            name,
            len(dataset),
            len(subjects),
            len(sessions),
            registered,
            len(demographic),
            len(multimodal),
            len(stage1_pair_groups),
            len(registered_multimodal),
        )
        self._log_scanner_target_summary(name, dataset)

    def _log_scanner_target_summary(self, name: str, dataset: PretrainDataset):
        if not self.scanner_targets_enabled:
            return
        valid_counts = Counter()
        class_counts = dict(self.scanner_target_class_counts)
        for row in dataset._cached_metadata:
            targets = row["common"].get("scanner_targets", {})
            for target, value in targets.items():
                if int(value) != int(self.scanner_target_config.ignore_index):
                    valid_counts[target] += 1
        logging.info(
            "%s scanner targets: valid_counts=%s class_counts=%s vocab=%s",
            name,
            dict(valid_counts),
            class_counts,
            self.scanner_target_vocabs,
        )

    def _build_monitor_datasets(self):
        routine_indices = self._stratified_indices(
            self.val_dataset,
            list(range(len(self.val_dataset))),
            None if self.complete_validation else self.routine_scan_count,
            self.monitor_seed,
        )
        self.routine_val_dataset = Subset(self.val_dataset, routine_indices)

        t1w_id = PretrainDataset.MODALITY_TO_ID["t1w"]
        train_canonical = self._canonical_modality_indices(self.train_dataset, t1w_id)
        reference_indices = self._stratified_indices(
            self.train_dataset,
            train_canonical,
            self.probe_train_max_subjects,
            self.monitor_seed,
        )
        modality_reference_indices = self._stratified_indices(
            self.train_dataset,
            self._canonical_all_modality_indices(self.train_dataset),
            self.probe_train_max_subjects,
            self.monitor_seed,
        )
        demographic_reference_indices = []
        for token in self.demographic_tokens:
            demographic_reference_indices.extend(
                self._stratified_demographic_indices(
                    self.train_dataset,
                    self._demographic_canonical_indices(self.train_dataset, token["modality_id"], token=token),
                    self.demographic_probe_train_max_subjects,
                    self.monitor_seed + int(token["modality_id"]),
                )
            )
        monitored_modalities = PretrainDataset.normalize_modality_ids(
            (PretrainDataset.MODALITY_TO_ID["t1w"], *self.demographic_modalities)
        )
        val_candidates = (
            [
                index
                for modality_id in monitored_modalities
                for index in self._canonical_modality_indices(self.val_dataset, modality_id)
            ]
            if self.probe_val_canonical_t1w
            else list(range(len(self.val_dataset)))
        )
        stage1_indices = self._stage1_pair_indices(
            self.val_dataset,
            self.stage1_pair_session_count,
            self.monitor_seed,
            allowed_pairs=self.stage1_multimodal_allowed_pairs,
        )
        self.probe_reference_dataset = self._evaluation_subset(self.train_dataset, reference_indices)
        self.modality_probe_reference_dataset = self._evaluation_subset(self.train_dataset, modality_reference_indices)
        self.demographic_probe_reference_dataset = self._evaluation_subset(self.train_dataset, demographic_reference_indices)
        self.probe_query_dataset = self._evaluation_subset(self.val_dataset, val_candidates)
        self.stage1_monitor_dataset = self._evaluation_subset(self.val_dataset, stage1_indices)

    def _evaluation_subset(self, dataset: PretrainDataset, indices: list[int]) -> Subset:
        files = [dataset.files[index] for index in indices]
        evaluation_dataset = PretrainDataset(
            files,
            transforms=self.val_transforms,
            metadata_paths=self.metadata_paths,
            registered_mapping_path=self.registered_mapping_path,
            registered_only=False,
            scanner_target_encoder=self.scanner_target_encoder,
            scanner_spacing_metadata_path=self.scanner_spacing_metadata_path,
        )
        return Subset(evaluation_dataset, list(range(len(evaluation_dataset))))

    @staticmethod
    def _canonical_modality_indices(dataset: PretrainDataset, modality_id: int) -> list[int]:
        per_subject = {}
        for index, row in sorted(enumerate(dataset._cached_metadata), key=lambda pair: dataset.files[pair[0]]):
            identity = row["identity"]
            if identity["modality_id"] == int(modality_id):
                per_subject.setdefault(identity["subject_key"], index)
        return list(per_subject.values())

    @staticmethod
    def _canonical_t1w_indices(dataset: PretrainDataset) -> list[int]:
        return PretrainDataModule._canonical_modality_indices(dataset, PretrainDataset.MODALITY_TO_ID["t1w"])

    @staticmethod
    def _canonical_all_modality_indices(dataset: PretrainDataset) -> list[int]:
        per_subject_modality = {}
        for index, row in sorted(enumerate(dataset._cached_metadata), key=lambda pair: dataset.files[pair[0]]):
            identity = row["identity"]
            key = (identity["subject_key"], identity["modality_id"])
            per_subject_modality.setdefault(key, index)
        return list(per_subject_modality.values())

    @staticmethod
    def _demographic_canonical_indices(
        dataset: PretrainDataset, modality_id: int = 0, token: Optional[dict] = None
    ) -> list[int]:
        # One canonical eligible-control index per subject. With ``token`` (DWI subtype), the
        # candidate must also fall inside the token's b-value range so the probe cohort stays
        # on a single diffusion contrast; otherwise filter by ``modality_id`` (structural).
        per_subject = {}
        for index, row in sorted(enumerate(dataset._cached_metadata), key=lambda pair: dataset.files[pair[0]]):
            identity = row["identity"]
            common = row["common"]
            if token is not None:
                if not SameSessionMultimodalSampler._scan_matches_token(identity, token):
                    continue
            elif identity["modality_id"] != int(modality_id):
                continue
            if common["pathology"] != 0 or math.isnan(common["age"]) or common["sex"] == -1:
                continue
            per_subject.setdefault(identity["subject_key"], index)
        return list(per_subject.values())

    @staticmethod
    def _stratified_demographic_indices(
        dataset: PretrainDataset, candidates: list[int], max_count: Optional[int], seed: int
    ) -> list[int]:
        if max_count is None or max_count <= 0 or len(candidates) <= max_count:
            return sorted(candidates, key=lambda index: dataset.files[index])
        grouped = defaultdict(list)
        for index in candidates:
            row = dataset._cached_metadata[index]
            age_decade = int(row["common"]["age"] // 10)
            grouped[(row["common"]["sex"], row["common"]["scanner_id"], age_decade)].append(index)
        rng = random.Random(seed)
        for group in grouped.values():
            group.sort(key=lambda index: dataset.files[index])
            rng.shuffle(group)
        selected = []
        while len(selected) < max_count:
            progressed = False
            for key in sorted(grouped):
                if grouped[key] and len(selected) < max_count:
                    selected.append(grouped[key].pop())
                    progressed = True
            if not progressed:
                break
        return selected

    @staticmethod
    def _stratified_indices(dataset: PretrainDataset, candidates: list[int], max_count: Optional[int], seed: int) -> list[int]:
        if max_count is None or max_count <= 0 or len(candidates) <= max_count:
            return sorted(candidates, key=lambda index: dataset.files[index])
        grouped = defaultdict(list)
        for index in candidates:
            row = dataset._cached_metadata[index]
            grouped[
                (
                    row["identity"]["modality_id"],
                    row["common"]["pathology"],
                    row["common"]["scanner_id"],
                )
            ].append(index)
        rng = random.Random(seed)
        for group in grouped.values():
            group.sort(key=lambda index: dataset.files[index])
            rng.shuffle(group)
        keys = sorted(grouped)
        selected = []
        while len(selected) < max_count:
            progressed = False
            for key in keys:
                if grouped[key] and len(selected) < max_count:
                    selected.append(grouped[key].pop())
                    progressed = True
            if not progressed:
                break
        return selected

    @staticmethod
    def _stage1_pair_indices(
        dataset: PretrainDataset,
        session_count: int,
        seed: int,
        allowed_pairs: Optional[Sequence[tuple[int, int]]] = None,
    ) -> list[int]:
        if allowed_pairs is not None:
            pair_groups = SameSessionMultimodalSampler._build_multimodal_pair_groups(dataset, allowed_pairs=allowed_pairs)
            rng = random.Random(seed)
            rng.shuffle(pair_groups)
            selected_pairs = []
            selected_sessions = set()
            target_count = session_count if session_count > 0 else len(pair_groups)
            for group in pair_groups:
                if len(selected_pairs) >= target_count:
                    break
                session_key = group["subject_session_key"]
                if session_key in selected_sessions and len(pair_groups) >= target_count:
                    continue
                selected_pairs.append(group)
                selected_sessions.add(session_key)
            for group in pair_groups:
                if len(selected_pairs) >= target_count:
                    break
                if group not in selected_pairs:
                    selected_pairs.append(group)
            selected_indices = []
            seen = set()
            for group in selected_pairs:
                for indices in group["indices"]:
                    for index in indices:
                        if index not in seen:
                            selected_indices.append(index)
                            seen.add(index)
            return selected_indices

        groups = defaultdict(list)
        modalities = defaultdict(set)
        for index, row in enumerate(dataset._cached_metadata):
            key = row["identity"]["subject_session_key"]
            if row["identity"]["modality_id"] < 0:
                continue
            groups[key].append(index)
            modalities[key].add(row["identity"]["modality_id"])
        eligible = sorted(key for key in groups if len(modalities[key]) > 1)
        rng = random.Random(seed)
        rng.shuffle(eligible)
        selected_sessions = eligible[:session_count] if session_count > 0 else eligible
        return [index for key in selected_sessions for index in groups[key]]

    @staticmethod
    def _log_monitor_summary(name: str, subset: Subset):
        if not _is_rank_zero_process():
            return
        dataset = subset.dataset
        indices = list(subset.indices)
        rows = [dataset._cached_metadata[index] for index in indices]
        counts = Counter(
            (
                row["identity"]["modality"],
                row["common"]["pathology"],
                row["common"]["scanner_id"],
            )
            for row in rows
        )
        registered = sum(bool(row["identity"]["is_registered_subset"]) for row in rows)
        subjects = {row["identity"]["subject_key"] for row in rows}
        sessions = {row["identity"]["subject_session_key"] for row in rows}
        logging.info(
            "%s monitor: scans=%d subjects=%d sessions=%d registered_scans=%d strata=%s",
            name,
            len(rows),
            len(subjects),
            len(sessions),
            registered,
            dict(counts),
        )
        if name == "routine_validation" and len(rows) <= 256:
            records = [
                {
                    "file": row["identity"]["file_path"],
                    "subject": row["identity"]["subject_key"],
                    "modality": row["identity"]["modality"],
                    "pathology": row["common"]["pathology"],
                    "sex": row["common"]["sex"],
                    "manufacturer": row["common"]["scanner_id"],
                    "registered": row["identity"]["is_registered_subset"],
                }
                for row in rows
            ]
            logging.info("routine_validation monitor records: %s", records)
        elif name == "routine_validation":
            logging.info("routine_validation records omitted from text log for exhaustive cohort (%d scans).", len(rows))

    def setup_predict(self):
        self.predict_dataset = SingleSubjectPredictDataset(
            self.predict_samples,
            transforms=self.predict_transforms,
        )

    def train_dataloader(self):
        num_samples = self.num_samples or 999999
        num_replicas, rank, sampler_seed = _distributed_sampler_context(self.sampler_seed)
        if self.same_session_multimodal_batches or self.demographic_batch_probability > 0.0:
            sampler = SameSessionMultimodalSampler(
                self.train_dataset,
                batch_size=self.batch_size,
                num_samples=num_samples,
                multimodal_probability=self.multimodal_batch_probability,
                demographic_probability=self.demographic_batch_probability,
                demographic_tokens=self.demographic_tokens,
                demographic_age_bin_years=self.demographic_age_bin_years,
                max_modalities_per_subject=self.max_modalities_per_subject,
                stage1_multimodal_batch_mode=self.stage1_multimodal_batch_mode,
                stage1_multimodal_allowed_pairs=self.stage1_multimodal_allowed_pairs,
                registered_batch_schedule=self.registered_batch_schedule,
                accumulate_grad_batches=self.accumulate_grad_batches,
                seed=sampler_seed,
                num_replicas=num_replicas,
                rank=rank,
            )
            if self.demographic_batch_probability > 0.0 and not sampler._demographic_strata_by_token:
                raise ValueError(
                    "demographic_batch_probability is positive but no eligible control scans with valid age and sex "
                    "are available for the configured demographic tokens."
                )
        else:
            sampler = StatefulReplacementSampler(
                self.train_dataset,
                num_samples=num_samples,
                seed=sampler_seed,
                num_replicas=num_replicas,
                rank=rank,
            )
        self._train_sampler = sampler
        sampler.enable_consumption_tracking()
        if self._pending_train_sampler_state is not None:
            sampler.load_state_dict(self._pending_train_sampler_state)
            self._pending_train_sampler_state = None

        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=False,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
            sampler=sampler,
            collate_fn=pretrain_collate,
        )

    def val_dataloader(self):
        return self._evaluation_dataloader(self.routine_val_dataset)

    def probe_reference_dataloader(self):
        return self._evaluation_dataloader(self.probe_reference_dataset)

    def modality_probe_reference_dataloader(self):
        return self._evaluation_dataloader(self.modality_probe_reference_dataset)

    def demographic_probe_reference_dataloader(self):
        return self._evaluation_dataloader(self.demographic_probe_reference_dataset)

    def probe_query_dataloader(self):
        return self._evaluation_dataloader(self.probe_query_dataset)

    def stage1_monitor_dataloader(self):
        return self._evaluation_dataloader(self.stage1_monitor_dataset)

    def _evaluation_dataloader(self, dataset):
        sampler = None
        if dist.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=False,
            sampler=sampler,
            shuffle=False,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
            collate_fn=pretrain_collate,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            num_workers=self.num_workers,
            batch_size=1,
            collate_fn=collate_return,
        )


if __name__ == "__main__":
    from gardening_tools.functional.paths.read import load_json

    splits = load_json("/Users/zcr545/Desktop/Projects/repos/asparagus_data/preprocessed_data/Task999_DummyData/splits.json")
    train_split = splits["train"]
    val_split = splits["validation"]
    data_module = PretrainDataModule(
        train_split=train_split,
        val_split=val_split,
        batch_size=2,
        num_workers=6,
    )
    data_module.setup("fit")
    print(data_module.train_dataset[0]["image"].shape)
