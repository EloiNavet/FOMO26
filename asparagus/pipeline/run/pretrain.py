import contextlib
import hashlib
import hydra
import json
import lightning as pl
import logging as py_logging
import math
import os
import random
import re
import time
import torch
from asparagus.functional.hydra import fast_instantiate
from asparagus.functional.versioning import generate_unused_run_id
from asparagus.modules.datasets.PretrainDataset import PretrainDataset
from asparagus.modules.hydra.plugins.searchpath_plugins import PretrainSearchpathPlugin
from asparagus.paths import get_config_path
from asparagus.pipeline.auto_configuration.checkpoint import (
    resolve_training_resume_checkpoint,
    resolve_training_resume_seed,
)
from asparagus.pipeline.auto_configuration.experiment_setup import prepare_ssl_plugins, prepare_standard_experiment
from asparagus.pipeline.auto_configuration.logging import logging
from asparagus.pipeline.run.evaluation_identity import clean_execution_commit, executing_git_sha
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.core.plugins import Plugins
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

if os.environ.get("FOMO26_DISABLE_DOTENV") != "1":
    load_dotenv()

OmegaConf.register_new_resolver("random", lambda min, max: random.randint(min, max), replace=True)
OmegaConf.register_new_resolver(
    "version",
    lambda resume_training, run_dir: generate_unused_run_id(resume_training=resume_training, run_dir=run_dir),
    use_cache=True,
    replace=True,
)
OmegaConf.register_new_resolver("eval", eval, replace=True)
Plugins.instance().register(PretrainSearchpathPlugin)


# Minimum per-device microbatch for a ``packed_pairs`` Stage-1 anatomy run: two same-session pairs,
# so each anchor has one cross-session negative besides its own positive. See
# ``_validate_stage1_pairing_contract``.
_STAGE1_PACKED_PAIRS_MIN_MICROBATCH = 4


def _enabled_loss_names(cfg: DictConfig, names) -> list[str]:
    """Return the subset of ``names`` whose ``losses.<name>.enabled`` is truthy."""
    top_losses = cfg.get("losses", {})
    if top_losses is None:
        return []
    enabled = []
    for name in names:
        value = top_losses.get(name, {})
        if bool(dict(value or {}).get("enabled", False)):
            enabled.append(name)
    return enabled


def _stage1_regularizer_weights(cfg: DictConfig) -> dict[str, float]:
    """Positive Stage-1 regularizer weights, keyed by their config path.

    ``SelfSupervisedModule._stage1_regularization_enabled`` builds the Stage-1 head from these
    weights alone -- ``losses.multimodal_stage1.enabled`` is not consulted, because the variance and
    covariance terms regularize ``z_anatomy`` and need no pairs. A disabled Stage-1 block whose
    promoted weights survive therefore still turns the whole contrastive path on.
    """
    stage1 = dict((cfg.get("losses", {}) or {}).get("multimodal_stage1", {}) or {})
    sigreg = dict(stage1.get("sigreg", {}) or {})
    candidates = {
        "losses.multimodal_stage1.variance_weight": float(stage1.get("variance_weight", 0.0) or 0.0),
        "losses.multimodal_stage1.covariance_weight": float(stage1.get("covariance_weight", 0.0) or 0.0),
        "losses.multimodal_stage1.same_modality_hard_negative_weight": float(
            stage1.get("same_modality_hard_negative_weight", 0.0) or 0.0
        ),
    }
    if bool(sigreg.get("enabled", False)):
        candidates["losses.multimodal_stage1.sigreg.weight"] = float(sigreg.get("weight", 0.0) or 0.0)
    return {path: weight for path, weight in candidates.items() if weight > 0.0}


def _validate_stage1_pairing_contract(cfg: DictConfig) -> None:
    """Refuse a Stage-1 anatomy run whose microbatch cannot hold a negative.

    Stage-1 anatomy is an in-batch InfoNCE over ``z_anatomy``: the candidate set is exactly the
    microbatch on one device. ``accumulate_grad_batches`` and the device count multiply the
    OPTIMIZER batch, not the candidate set, so they contribute no negatives at all.

    ``packed_pairs`` fills a microbatch with ``batch_size // 2`` same-session cross-modal pairs.
    With ``batch_size=2`` there is one pair, so each anchor's candidate set is its own positive and
    nothing else; ``logsumexp`` over a single logit yields ``log_prob=0``, i.e. a loss of exactly
    zero with exactly zero gradient, at every step and for every batch the sampler can draw. That is
    indistinguishable in the logs from "disabled", which is precisely what a stacked recipe must
    never ship -- and it is invisible until ``component_starvation_window`` trips hours in.

    Two pairs (``batch_size >= 4``) is the floor at which each anchor sees one true cross-session
    negative and the term is well-posed. This is a property of the geometry, not of the sampler
    probabilities, so no amount of batch-routing tuning can substitute for it.

    SCOPE. This fires only when the run has set ``training.component_starvation_window > 0``, i.e.
    has already declared "abort if an enabled component stays inert". This gate is the static half
    of that same contract: it decides at pre-flight what the dynamic guard would otherwise discover
    hours into a multi-GPU allocation. Runs that never opted in keep their existing behaviour
    exactly -- several screening families deliberately run microbatch 1 with the guard off, and
    silently changing what they refuse would be a scientific change to campaigns in flight.
    """
    training = cfg.get("training", {}) or {}
    if int(training.get("component_starvation_window", 0) or 0) <= 0:
        return
    losses = cfg.get("losses", {}) or {}
    stage1 = dict(losses.get("multimodal_stage1", {}) or {})
    if not bool(stage1.get("enabled", False)):
        return
    if float(stage1.get("weight_anatomy", 0.0) or 0.0) <= 0.0:
        return
    data = cfg.get("data", {}) or {}
    if not bool(data.get("same_session_multimodal_batches", False)):
        return
    if str(data.get("stage1_multimodal_batch_mode", "single_session")) != "packed_pairs":
        return
    if float(data.get("multimodal_batch_probability", 0.0) or 0.0) <= 0.0:
        return
    batch_size = training.get("batch_size")
    if batch_size is None:
        # A composable fragment, not a runnable job: nothing to judge yet.
        return
    batch_size = int(batch_size)
    if batch_size >= _STAGE1_PACKED_PAIRS_MIN_MICROBATCH:
        return
    accum = int(training.get("accumulate_grad_batches", 1) or 1)
    devices = int((cfg.get("hardware", {}) or {}).get("num_devices", 1) or 1)
    raise ValueError(
        f"losses.multimodal_stage1 is enabled with data.stage1_multimodal_batch_mode=packed_pairs but "
        f"training.batch_size={batch_size} < {_STAGE1_PACKED_PAIRS_MIN_MICROBATCH}. The anatomy InfoNCE draws its "
        f"candidates from ONE microbatch on ONE device: at batch_size={batch_size} the sampler packs "
        f"{batch_size // 2} same-session pair(s), so every anchor's only candidate is its own positive and the "
        "raw loss is exactly zero with zero gradient at every step. "
        f"accumulate_grad_batches={accum} and hardware.num_devices={devices} enlarge the optimizer batch "
        f"(global {batch_size * accum * devices}) but contribute no negatives. "
        f"Raise training.batch_size to >= {_STAGE1_PACKED_PAIRS_MIN_MICROBATCH} (lower accumulate_grad_batches to "
        "hold the global batch fixed), or disable losses.multimodal_stage1 explicitly. "
        f"training.component_starvation_window={int(training.get('component_starvation_window', 0) or 0)} would "
        "abort this run anyway, but only once the term activates -- hours into the allocation."
    )


def _ssl_objective(cfg: DictConfig) -> str:
    """Return the normalized top-level SSL objective."""
    return str(cfg.ssl.get("objective", "amaes")).strip().lower() if "ssl" in cfg else "amaes"


def _outer_masking_enabled(cfg: DictConfig) -> bool:
    """AMAES owns outer masking."""
    return bool(cfg.transforms.masking)


def _validate_stage2_cross_reconstruction_contract(cfg: DictConfig) -> None:
    """Refuse Stage-2 cross reconstruction whose pairs are not provably voxel-aligned.

    ``losses.multimodal_stage2.registered_only`` filters by membership in the registered mapping
    TSV. That is not registration: ``PretrainDataset`` applies its random crop per item, so the two
    same-session scans ``SameSessionMultimodalSampler`` co-locates in a batch do not share a voxel
    grid. ``stage2_reconstruction_loss`` then takes a per-voxel ``smooth_l1`` between the decoded
    source and the target volume, which under unaligned crops optimises noise under a
    cross-reconstruction name.

    This is the Stage-2 counterpart of a cross-modal pairing check, so the two paths
    cannot disagree about what "registered" means.
    """
    losses = cfg.get("losses", {})
    stage2 = dict((losses.get("multimodal_stage2", {}) if losses is not None else {}) or {})
    if not stage2:
        return
    weight = float(stage2.get("weight_cross_reconstruction", 0.0) or 0.0)
    if weight <= 0.0:
        return
    if not bool(stage2.get("enabled", False)):
        raise ValueError(
            "losses.multimodal_stage2.weight_cross_reconstruction > 0 requires losses.multimodal_stage2.enabled=true."
        )
    if not bool(stage2.get("decoder_enabled", False)):
        raise ValueError(
            "losses.multimodal_stage2.weight_cross_reconstruction > 0 requires "
            "losses.multimodal_stage2.decoder_enabled=true (the FiLM decoder produces the prediction)."
        )
    if not bool(stage2.get("datamodule_provides_registered_pairs", False)):
        raise ValueError(
            f"losses.multimodal_stage2.weight_cross_reconstruction={weight} but no configured "
            "datamodule produces same-subject, same-session, geometrically aligned pairs: "
            "registered_only filters by registered-mapping membership, while PretrainDataset "
            "returns one volume per item and applies its random crop per item. Training anyway "
            "would take a per-voxel loss between differently-cropped volumes. Provide a datamodule "
            "emitting same-session pairs with a shared crop and a `pair_geometry`, then set "
            "losses.multimodal_stage2.datamodule_provides_registered_pairs=true; or set "
            "weight_cross_reconstruction=0."
        )


def _validate_ssl_objective_contract(cfg: DictConfig) -> None:
    """Reject ambiguous objective combinations before transforms/models are built."""
    _validate_stage2_cross_reconstruction_contract(cfg)
    objective = _ssl_objective(cfg)
    if objective != "amaes":
        # This distribution ships the AMAES reconstruction objective only. Latent-prediction
        # objectives and their auxiliaries are not part of the public code release, so a config
        # asking for one is refused rather than silently reinterpreted as AMAES.
        raise ValueError(f"Unsupported ssl.objective={objective!r}; this distribution provides amaes only.")
    aux = dict(cfg.ssl.get("aux", {}) or {}) if "ssl" in cfg else {}
    enabled_aux = sorted(name for name, value in aux.items() if bool(dict(value or {}).get("enabled", False)))
    if enabled_aux:
        raise ValueError(f"ssl.aux terms {enabled_aux} belong to objectives this distribution does not provide.")
    # Last, and deliberately so: this asks whether an evaluated loss is well-posed. "The configured
    # path never evaluates this loss at all" is the more fundamental complaint, so the objective
    # checks above must be the ones a doubly-invalid config reports.
    _validate_stage1_pairing_contract(cfg)


def _pretrain_progress_callbacks(cfg: DictConfig):
    return [TQDMProgressBar(refresh_rate=cfg.logger.log_every_n_steps)] if cfg.logger.progress_bar else []


def _is_rank_zero_process() -> bool:
    return os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) == "0"


class PersistentExceptionCheckpoint(Callback):
    """Save an exception checkpoint without deleting it during Lightning teardown."""

    def __init__(self, dirpath: str, filename: str = "on_exception") -> None:
        super().__init__()
        if not filename:
            raise ValueError("filename cannot be empty")
        self.dirpath = dirpath
        self.filename = filename

    @property
    def ckpt_path(self) -> str:
        return os.path.join(self.dirpath, self.filename + ".ckpt")

    def on_exception(self, trainer, pl_module, exception) -> None:
        os.makedirs(self.dirpath, exist_ok=True)
        trainer.save_checkpoint(self.ckpt_path)


class ForbidOptimizerStep(Callback):
    """Fail closed if a validation-only completion ever enters optimization."""

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        raise RuntimeError("completion_validation.enabled=true forbids every optimizer step")


class StopAfterValidationStep(Callback):
    """Stop after validating a step without shortening any training schedule."""

    def __init__(self, step: int, *, pause_marker: bool = False) -> None:
        super().__init__()
        self.step = int(step)
        self.pause_marker = bool(pause_marker)
        if self.step <= 0:
            raise ValueError("step must be positive")

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.global_step >= self.step:
            if self.pause_marker and trainer.is_global_zero:
                marker = os.path.join(trainer.default_root_dir, "TRAINING_PAUSED")
                with open(marker, "w", encoding="utf-8") as stream:
                    stream.write(f"global_step={int(trainer.global_step)}\n")
            trainer.should_stop = True


class StopAfterRelativeOptimizerSteps(Callback):
    """Audit a short continuation without changing the original scheduler horizon."""

    def __init__(self, delta_steps: int) -> None:
        super().__init__()
        self.delta_steps = int(delta_steps)
        if self.delta_steps <= 0:
            raise ValueError("delta_steps must be positive")
        self.target_step = None
        self.saved = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        # Lightning invokes on_fit_start before restoring loop progress from a
        # checkpoint. Anchor the relative target only when the first resumed
        # training batch starts, after global_step has been restored.
        if self.target_step is None:
            self.target_step = int(trainer.global_step) + self.delta_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self.saved or self.target_step is None or int(trainer.global_step) < self.target_step:
            return
        checkpoint_dir = os.path.join(trainer.default_root_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"resume_audit_step_{int(trainer.global_step)}.ckpt")
        trainer.save_checkpoint(checkpoint_path)
        if trainer.is_global_zero:
            marker = os.path.join(trainer.default_root_dir, "TRAINING_PAUSED")
            with open(marker, "w", encoding="utf-8") as stream:
                stream.write(f"resume_audit_global_step={int(trainer.global_step)}\n")
        self.saved = True
        trainer.should_stop = True


class ExactOptimizerStepCheckpoint(Callback):
    """Persist campaign checkpoints at exact optimizer steps, including non-epoch boundaries."""

    def __init__(
        self,
        dirpath: str,
        steps,
        *,
        stop_at_step: int | None = None,
        validation_steps=(),
    ) -> None:
        super().__init__()
        self.dirpath = dirpath
        self.steps = tuple(sorted({int(step) for step in steps}))
        self.validation_steps = frozenset(int(step) for step in validation_steps)
        if not self.steps or self.steps[0] <= 0:
            raise ValueError("exact checkpoint steps must be positive")
        if any(step <= 0 for step in self.validation_steps):
            raise ValueError("validation checkpoint steps must be positive")
        self.stop_at_step = int(stop_at_step) if stop_at_step is not None else None
        if self.stop_at_step is not None and self.stop_at_step not in self.steps:
            raise ValueError("stop_at_step must be one of the exact checkpoint steps")
        self.saved: set[int] = set()
        self.pending_validation: set[int] = set()

    def on_fit_start(self, trainer, pl_module) -> None:
        self.saved = {step for step in self.steps if os.path.isfile(os.path.join(self.dirpath, f"step_{step}.ckpt"))}
        self.pending_validation = set()

    def _publish(self, trainer, step: int) -> None:
        atomic_save_checkpoint(trainer, os.path.join(self.dirpath, f"step_{step}.ckpt"))
        self.saved.add(step)
        self.pending_validation.discard(step)
        if self.stop_at_step == step:
            if trainer.is_global_zero:
                marker = os.path.join(trainer.default_root_dir, "TRAINING_PAUSED")
                with open(marker, "w", encoding="utf-8") as stream:
                    stream.write(f"exact_checkpoint_global_step={step}\n")
            trainer.should_stop = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = int(trainer.global_step)
        if step not in self.steps or step in self.saved:
            return
        if step in self.validation_steps:
            # Saving here would precede the validation boundary at this same optimizer step. Defer
            # publication until on_validation_end so the checkpoint identity can truthfully record
            # that the boundary completed.
            self.pending_validation.add(step)
            return
        # Permanent milestones are the checkpoints downstream selection and packaging consume, so
        # they are the ones that must never be observable in a torn state.
        self._publish(trainer, step)

    def on_validation_end(self, trainer, pl_module) -> None:
        step = int(trainer.global_step)
        if step in self.pending_validation and step not in self.saved:
            self._publish(trainer, step)


class TechnicalCanaryCheckpoint(Callback):
    """Stop one TECHNICAL_ONLY probe at an exact optimizer step and publish its checkpoint."""

    def __init__(self, dirpath: str, root_dir: str, *, step: int, filename: str) -> None:
        super().__init__()
        self.dirpath = dirpath
        self.root_dir = root_dir
        self.step = int(step)
        self.filename = str(filename)
        self.saved = False
        if self.step <= 0 or self.filename != f"step_{self.step}.ckpt":
            raise ValueError("technical canary checkpoint filename must exactly match its positive optimizer step")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = int(trainer.global_step)
        if self.saved or step < self.step:
            return
        if step != self.step:
            raise RuntimeError(f"technical canary missed exact optimizer step {self.step}; observed {step}")
        atomic_save_checkpoint(trainer, os.path.join(self.dirpath, self.filename))
        if trainer.is_global_zero:
            marker = os.path.join(self.root_dir, "TECHNICAL_CANARY_COMPLETE")
            with open(marker, "w", encoding="utf-8") as stream:
                stream.write(f"global_step={step}\n")
        self.saved = True
        trainer.should_stop = True


#: Marker the Slurm batch shell writes when it receives the pre-timeout signal. It is a *request*,
#: not a stop: training continues to the next safe optimizer-step boundary before checkpointing.
STOP_AT_NEXT_BOUNDARY_MARKER = "STOP_AT_NEXT_BOUNDARY"
#: Marker family `resubmission_decision.py` already treats as "do not resubmit this chain".
UNRECOVERABLE_FAILURE_MARKER = "UNRECOVERABLE_FAILURE"
#: Written next to a checkpoint only after it has been re-read and structurally validated.
CHECKPOINT_OK_SUFFIX = ".ok"
CHECKPOINT_SHA256_SUFFIX = ".sha256"


def _unlink_if_present(path: str) -> bool:
    """Delete ``path`` if it exists. Returns whether *this* caller removed it.

    Safe to call concurrently from every DDP rank: ``FileNotFoundError`` means another rank won the
    race and the postcondition (the file is gone) already holds. Only that one error is absorbed --
    a permission or I/O failure still raises, because those do *not* establish the postcondition.
    """
    try:
        os.remove(path)
    except FileNotFoundError:
        return False
    return True


def _sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_saved_checkpoint(path: str) -> int:
    """Re-read a just-written checkpoint. Returns its global step; raises if it is unusable.

    A torn or truncated file is the failure mode that silently costs a whole segment: the chain
    resubmits, ``find_last_ckpt`` picks the newest file, and the load fails hours later. Reading it
    back once, here, converts that into an immediate, local error.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Checkpoint {path} did not deserialize to a dict (got {type(payload).__name__}).")
    if "state_dict" not in payload:
        raise RuntimeError(f"Checkpoint {path} has no 'state_dict' key; refusing to mark it valid.")
    return int(payload.get("global_step", -1))


def _checkpoint_barrier(trainer, name: str) -> None:
    """Synchronise ranks around a checkpoint publish, when there are ranks to synchronise.

    Barriers matter only under a distributed strategy. Single-process training -- and the
    lightweight trainer doubles the campaign tests use -- have no peers to wait for, so requiring
    a strategy here would make ``atomic_save_checkpoint`` unusable outside DDP.
    """
    strategy = getattr(trainer, "strategy", None)
    barrier = getattr(strategy, "barrier", None)
    if callable(barrier):
        barrier(name)


def atomic_save_checkpoint(trainer, path: str) -> None:
    """Write a checkpoint that is either complete and validated, or absent.

    ``trainer.save_checkpoint`` writes in place, so a walltime kill or a Lustre hiccup mid-write
    leaves a truncated file with a newer mtime than the last good one -- exactly what checkpoint
    discovery would then select. Writing to ``<path>.tmp``, validating it, and only then
    ``os.replace``-ing makes the visible path atomic. The ``.ok`` sidecar is written last, so
    ``find_last_ckpt.py --require-ok-marker`` can distinguish "validated" from "merely present".
    """
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp"
    # Collective: every rank participates in the write.
    trainer.save_checkpoint(tmp_path)
    _checkpoint_barrier(trainer, "atomic_checkpoint_written")
    if not getattr(trainer, "is_global_zero", True):
        _checkpoint_barrier(trainer, "atomic_checkpoint_published")
        return
    try:
        global_step = _validate_saved_checkpoint(tmp_path)
        digest = _sha256_file(tmp_path)
        os.replace(tmp_path, path)
        with open(f"{path}{CHECKPOINT_SHA256_SUFFIX}", "w", encoding="utf-8") as stream:
            stream.write(f"{digest}  {os.path.basename(path)}\n")
        with open(f"{path}{CHECKPOINT_OK_SUFFIX}", "w", encoding="utf-8") as stream:
            json.dump({"global_step": global_step, "sha256": digest}, stream, sort_keys=True)
            stream.write("\n")
    except BaseException:
        # Never leave a half-published checkpoint or a stale .ok behind.
        for stale in (tmp_path, f"{path}{CHECKPOINT_OK_SUFFIX}"):
            with contextlib.suppress(OSError):
                os.remove(stale)
        raise
    finally:
        _checkpoint_barrier(trainer, "atomic_checkpoint_published")


def write_unrecoverable_failure(root_dir: str, reason: str, detail: str = "") -> str:
    """Record a failure the chained resubmission must not retry.

    A chained-resubmission driver is expected to refuse to resubmit when a marker of this family
    exists; until this marker was written, a NaN divergence or an OOM was resubmitted as if it were
    an ordinary walltime interruption and burned the rest of the chain.
    """
    os.makedirs(root_dir, exist_ok=True)
    marker = os.path.join(root_dir, f"{UNRECOVERABLE_FAILURE_MARKER}_{reason}")
    with open(marker, "w", encoding="utf-8") as stream:
        json.dump({"reason": reason, "detail": detail}, stream, sort_keys=True)
        stream.write("\n")
    return marker


class SafeBoundaryStopCallback(Callback):
    """Honour a pre-timeout stop request at the next completed optimizer step.

    Slurm's ``--signal`` fires at an arbitrary microbatch. Stopping there would checkpoint in the
    middle of a gradient-accumulation window, so the resumed segment either replays or skips the
    partial window. This callback waits until ``accumulate_grad_batches`` microbatches have
    completed -- i.e. ``trainer.global_step`` has just advanced -- then writes a validated
    full-state checkpoint and asks the trainer to stop.
    """

    def __init__(self, dirpath: str, root_dir: str, *, filename: str = "last.ckpt") -> None:
        super().__init__()
        self.dirpath = dirpath
        self.root_dir = root_dir
        self.filename = filename
        self.marker_path = os.path.join(root_dir, STOP_AT_NEXT_BOUNDARY_MARKER)
        self.stopped = False

    def _requested(self) -> bool:
        return os.path.exists(self.marker_path)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self.stopped or not self._requested():
            return
        accumulate = max(1, int(getattr(trainer, "accumulate_grad_batches", 1)))
        # (batch_idx + 1) % accumulate == 0 is exactly "an optimizer step just completed and no
        # partial accumulation window is pending".
        if (int(batch_idx) + 1) % accumulate:
            return
        step = int(trainer.global_step)
        py_logging.warning("Pre-timeout stop requested; checkpointing at safe optimizer step %d.", step)
        atomic_save_checkpoint(trainer, os.path.join(self.dirpath, self.filename))
        if trainer.is_global_zero:
            with open(os.path.join(self.root_dir, "TRAINING_PAUSED"), "w", encoding="utf-8") as stream:
                stream.write(f"pre_timeout_global_step={step}\n")
        self.stopped = True
        trainer.should_stop = True


class UnrecoverableFailureMarker(Callback):
    """Turn NaN/OOM/corruption into a fail-closed marker instead of a silent resubmission."""

    def __init__(self, root_dir: str) -> None:
        super().__init__()
        self.root_dir = root_dir

    def on_exception(self, trainer, pl_module, exception) -> None:
        if not trainer.is_global_zero:
            return
        reason = "exception"
        if isinstance(exception, torch.cuda.OutOfMemoryError):
            reason = "oom"
        else:
            text = f"{type(exception).__name__}: {exception}".lower()
            if "nan" in text or "inf" in text or "non-finite" in text or "nonfinite" in text:
                reason = "nonfinite"
            elif "checkpoint" in text and ("corrupt" in text or "state_dict" in text):
                reason = "corrupt_checkpoint"
            elif isinstance(exception, KeyboardInterrupt):
                # SIGINT is how a graceful pre-timeout stop reaches the process: not a failure.
                return
        write_unrecoverable_failure(self.root_dir, reason, detail=f"{type(exception).__name__}: {exception}")


class RollingRecoveryCheckpoint(Callback):
    """Keep two validated full-state checkpoints at a fixed optimizer-step cadence.

    A single rolling ``last.ckpt`` is one interrupted write away from having no rolling state at
    all. Two generations bound the cost at exactly two files while guaranteeing that a torn write
    cannot destroy the last good state.

    ``last.ckpt`` is ALWAYS the newest generation. Alternating between the two names would be
    cheaper, but then ``last_prev.ckpt`` is sometimes the newer file -- which contradicts its name
    and makes ``find_last_ckpt.py``'s default ``prefer_last`` resume from a checkpoint one cadence
    behind. The rotation below costs one rename and keeps both names meaning what they say.
    """

    CURRENT = "last.ckpt"
    PREVIOUS = "last_prev.ckpt"

    def __init__(self, dirpath: str, every_n_steps: int) -> None:
        super().__init__()
        self.dirpath = dirpath
        self.every_n_steps = int(every_n_steps)
        if self.every_n_steps <= 0:
            raise ValueError("recovery checkpoint cadence must be positive")
        self.last_saved_step = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = int(trainer.global_step)
        if step <= 0 or step % self.every_n_steps or step == self.last_saved_step:
            return
        current = os.path.join(self.dirpath, self.CURRENT)
        staging = os.path.join(self.dirpath, f"{self.CURRENT}.next")
        # Write and validate the new generation under its own name first, so the existing
        # last.ckpt stays intact and resumable for the whole duration of the write.
        atomic_save_checkpoint(trainer, staging)
        if getattr(trainer, "is_global_zero", True):
            self._rotate(staging, current)
        _checkpoint_barrier(trainer, "rolling_checkpoint_rotated")
        self.last_saved_step = step

    def _rotate(self, staging: str, current: str) -> None:
        previous = os.path.join(self.dirpath, self.PREVIOUS)
        if os.path.exists(current):
            for suffix in ("", CHECKPOINT_SHA256_SUFFIX, CHECKPOINT_OK_SUFFIX):
                source = f"{current}{suffix}"
                if os.path.exists(source):
                    os.replace(source, f"{previous}{suffix}")
                else:
                    # Never leave a previous-generation receipt describing a different file.
                    with contextlib.suppress(OSError):
                        os.remove(f"{previous}{suffix}")
        for suffix in ("", CHECKPOINT_SHA256_SUFFIX, CHECKPOINT_OK_SUFFIX):
            source = f"{staging}{suffix}"
            if os.path.exists(source):
                os.replace(source, f"{current}{suffix}")


#: Dataloader workers are seeded by Lightning from (base seed, worker id, global rank). This is a
#: scientific choice, not a performance one: it fixes the augmentation stream each worker draws, so
#: changing it -- or changing the worker count under it -- changes the trajectory. The constant is
#: the single place the policy is decided, and the checkpoint records the value it was decided to.
WORKER_SEEDING_WORKERS = True
WORKER_SEEDING_POLICY = "lightning.seed_everything(workers=True)/pl_worker_init_function"


class CampaignCheckpointMetadata(Callback):
    """Embed a small, dependency-free identity record in campaign checkpoints."""

    SCHEMA_VERSION = "fomo26-campaign-checkpoint-v1"

    def __init__(self, metadata: dict, *, worker_seeding_policy: str) -> None:
        super().__init__()
        self.metadata = dict(metadata)
        # How dataloader workers derive their RNG. It decides the augmentation stream, so a run
        # seeded differently is a different experiment even at an identical `training.seed`.
        self.worker_seeding_policy = str(worker_seeding_policy)
        self.last_validation_step: int | None = None

    def on_validation_end(self, trainer, pl_module) -> None:
        self.last_validation_step = int(trainer.global_step)

    @staticmethod
    def _runtime_flags() -> dict:
        """Backend state as observed, never as configured.

        These are read, never set: the callback must not move a single numeric knob. They are
        recorded because two runs can share a configuration and still diverge numerically when the
        backend they executed on did not agree.
        """
        return {
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        }

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        global_step = int(trainer.global_step)
        record = {
            "schema_version": self.SCHEMA_VERSION,
            **self.metadata,
            "global_step": global_step,
            "validation_boundary_complete": self.last_validation_step == global_step,
            "validation_boundary_global_step": self.last_validation_step,
            # Kept at the top level as well as inside `runtime_flags`: existing readers look for
            # them here, and this record is consumed by artifacts that predate the nested block.
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "worker_seeding_policy": self.worker_seeding_policy,
            "runtime_flags": self._runtime_flags(),
            # Attested here, at save time, from the tree the training process is importing -- not
            # copied from the campaign's `launch_git_commit`, which is the submit-time commit and
            # is recorded separately. The two diverge whenever a job is requeued, resumed, or
            # starts after the checkout moved. `None` when the tree is dirty or not a checkout:
            # a `<sha>-dirty` marker names the fact of dirt, not the dirt, so it can never serve
            # as an equality key and must not be published as one.
            "scientific_execution_commit": clean_execution_commit(),
            "executing_git_sha": executing_git_sha(),
        }
        checkpoint["fomo26_campaign"] = record


def _resolve_pretrain_modality_ids(modalities) -> list[int]:
    return list(PretrainDataset.normalize_modality_ids(modalities, default=()))


def _resolve_modality_ignore_ids(modalities) -> list[int]:
    """Resolve the Stage-1 modality ignore list (names -> ids).

    Unlike :func:`_resolve_pretrain_modality_ids`, an empty list is valid here: it means "exclude
    no modality" (keep every class in the CE/SupCon target).
    """
    items = list(modalities) if modalities is not None else []
    if len(items) == 0:
        return []
    return list(PretrainDataset.normalize_modality_ids(items, default=()))


def _validate_pretrain_execution_mode(cfg: DictConfig) -> None:
    resume_training = bool(getattr(cfg, "resume_training", False))
    resume_checkpoint_path = getattr(cfg, "resume_checkpoint_path", None)
    checkpoint_inputs = [key for key in ("checkpoint_run_id", "checkpoint_path", "hf_model_id") if getattr(cfg, key, None)]
    completion = cfg.get("completion_validation") or {}
    completion_enabled = bool(completion.get("enabled", False))
    technical_initialization = cfg.get("technical_initialization") or {}
    technical_canary = cfg.get("technical_canary") or {}
    technical_initialization_enabled = bool(technical_initialization.get("enabled", False))
    technical_canary_enabled = bool(technical_canary.get("enabled", False))
    campaign = cfg.get("campaign") or {}
    technical_authority = str(campaign.get("authority", "")) == "TECHNICAL_ONLY"

    if technical_initialization_enabled or technical_canary_enabled or technical_authority:
        if not technical_authority:
            raise ValueError("technical initialization/canary requires campaign.authority=TECHNICAL_ONLY")
        if str(technical_canary.get("authority", "")) != "TECHNICAL_ONLY" or not technical_canary_enabled:
            raise ValueError("TECHNICAL_ONLY authority requires an enabled technical_canary block")
        if int(technical_canary.get("max_optimizer_steps") or 0) != 16:
            raise ValueError("Task-6 technical canary is bounded to exactly 16 optimizer steps")
        if str(technical_canary.get("checkpoint_filename", "")) != "step_16.ckpt":
            raise ValueError("Task-6 technical canary must publish step_16.ckpt")
        if resume_training or resume_checkpoint_path or checkpoint_inputs or completion_enabled or cfg.validation_only:
            raise ValueError(
                "Task-6 technical canary is a fresh weights-only or normal-initialization run; "
                "resume, evaluation and normal checkpoint inputs are forbidden"
            )
        if int(cfg.training.steps) != 250_000 or int(cfg.training.warmup_steps) != 5_000:
            raise ValueError("Task-6 technical canary must retain the 250k/5k optimizer trajectory")
        if int(cfg.training.stop_after_steps) != 32_000:
            raise ValueError("Task-6 technical canary must retain the 32k screening cutoff")
        if technical_initialization_enabled:
            # The weights-only import this armed was qualified by the Task-6 curriculum campaign,
            # which the public code release does not ship. Refusing is the only honest answer: the
            # alternative is to accept the config and silently start from random weights.
            raise ValueError(
                "technical weights-only initialization is not available in this distribution: "
                "its checkpoint-qualification authority is not part of the public code release"
            )
        return

    if completion_enabled:
        if cfg.validation_only:
            raise ValueError("completion_validation.enabled=true cannot be combined with validation_only=true.")
        if not resume_training or not resume_checkpoint_path:
            raise ValueError(
                "completion_validation.enabled=true requires resume_training=true and an explicit resume_checkpoint_path."
            )
        if checkpoint_inputs:
            raise ValueError(
                "completion validation restores only resume_checkpoint_path from the same run; "
                f"do not also provide {', '.join(checkpoint_inputs)}."
            )
        expected_step = int(completion.get("expected_step") or 0)
        expected_epoch = int(completion.get("expected_epoch", -1))
        expected_sha256 = str(completion.get("checkpoint_sha256") or "")
        if expected_step <= 0:
            raise ValueError("completion_validation.expected_step must be positive.")
        if expected_epoch < 0:
            raise ValueError("completion_validation.expected_epoch must be non-negative.")
        if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
            raise ValueError("completion_validation.checkpoint_sha256 must be a lowercase SHA256.")
        stop_after_steps = cfg.training.get("stop_after_steps")
        if stop_after_steps is None or int(stop_after_steps) != expected_step:
            raise ValueError(
                "completion validation requires training.stop_after_steps to equal its checkpoint step exactly: "
                f"expected {expected_step}, got {stop_after_steps}."
            )
        return

    if cfg.validation_only:
        if resume_training:
            raise ValueError("validation_only=true cannot be combined with resume_training=true.")
        if resume_checkpoint_path:
            raise ValueError("validation_only=true cannot be combined with resume_checkpoint_path.")
        if not cfg.checkpoint_path:
            raise ValueError("validation_only=true requires checkpoint_path to identify the checkpoint under evaluation.")
        if not cfg.data.complete_validation or not cfg.monitoring.exhaustive_evaluation:
            raise ValueError(
                "validation_only=true requires data.complete_validation=true and monitoring.exhaustive_evaluation=true."
            )
        return

    if resume_training and checkpoint_inputs:
        raise ValueError(
            "resume_training=true restores the checkpoint in this run's own output directory; "
            f"do not also provide {', '.join(checkpoint_inputs)}. Set run_id to the interrupted run ID."
        )
    if resume_checkpoint_path and not resume_training:
        raise ValueError("resume_checkpoint_path requires resume_training=true.")

    if checkpoint_inputs:
        raise ValueError(
            "Normal pretraining does not resume from checkpoint_path/checkpoint_run_id/hf_model_id. "
            "To continue an interrupted run, repeat the original command with run_id=<original_id> "
            "and resume_training=true. To evaluate a checkpoint, use the exhaustive validation profile."
        )


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completion_collective_device() -> torch.device:
    """Choose a device supported by the active distributed backend."""

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        backend = str(torch.distributed.get_backend()).lower()
        if backend == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError("NCCL completion metrics require an available CUDA device.")
            return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")


def _validate_completion_checkpoint(cfg: DictConfig, checkpoint_path: str) -> dict:
    completion = cfg.completion_validation
    expected_step = int(completion.expected_step)
    expected_epoch = int(completion.expected_epoch)
    expected_sha256 = str(completion.checkpoint_sha256)
    path = Path(checkpoint_path)
    if path.name != f"step_{expected_step}.ckpt":
        raise ValueError(
            "completion validation requires the exact authoritative checkpoint filename "
            f"step_{expected_step}.ckpt, got {path.name}."
        )
    if path.name == "on_exception.ckpt":
        raise ValueError("completion validation must never select on_exception.ckpt.")

    # The worker validates this before torchrun. Repeat the identity check in rank
    # zero so direct entrypoint use also fails closed without making every rank hash
    # and deserialize the 1.1 GB checkpoint an extra time.
    if _is_rank_zero_process():
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"completion checkpoint SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        actual_step = int(checkpoint.get("global_step", -1))
        actual_epoch = int(checkpoint.get("epoch", -1))
        if actual_step != expected_step or actual_epoch != expected_epoch:
            raise ValueError(
                "completion checkpoint position mismatch: "
                f"expected step={expected_step}, epoch={expected_epoch}; "
                f"got step={actual_step}, epoch={actual_epoch}."
            )
        del checkpoint
    return {
        "path": str(path),
        "sha256": expected_sha256,
        "step": expected_step,
        "epoch": expected_epoch,
    }


def _run_trainer_phase(
    cfg: DictConfig,
    trainer,
    model_module,
    data_module,
    *,
    fit_ckpt_path: str | None,
    completion_contract: dict | None,
) -> str:
    """Dispatch mutually exclusive fit/evaluation modes.

    Keeping completion validation in a distinct branch is the primary zero-step
    guarantee: that mode can call only ``Trainer.validate``. The
    ``ForbidOptimizerStep`` callback remains a defense-in-depth tripwire.
    """

    if completion_contract is not None:
        trainer.validate(model=model_module, datamodule=data_module, ckpt_path=fit_ckpt_path)
        return "completion_validation"
    if cfg.validation_only:
        trainer.validate(model=model_module, datamodule=data_module, ckpt_path=cfg.checkpoint_path)
        return "validation"
    trainer.fit(model=model_module, datamodule=data_module, ckpt_path=fit_ckpt_path)
    return "fit"


def _validate_campaign_dataset_contract(cfg: DictConfig) -> None:
    """Fail before setup when a campaign declares an immutable dataset gate."""
    campaign = cfg.get("campaign")
    contract = campaign.get("dataset_contract") if campaign is not None else None
    if not contract:
        return

    required_root = Path(str(contract.get("required_root", ""))).expanduser().resolve(strict=False)
    configured_root = Path(str(cfg.data.data_path)).expanduser().resolve(strict=False).parent
    if configured_root != required_root:
        raise ValueError(
            "Campaign dataset contract rejected the configured data root: "
            f"expected {required_root}, got {configured_root}. Toy or alternate datasets are not admissible."
        )

    expected_sha256 = str(contract.get("manifest_sha256") or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError(
            "Campaign dataset contract requires a 64-character manifest_sha256. "
            "Set FOMO300K_CURATED_MANIFEST_SHA256 only after the tensorized manifest is finalized."
        )
    manifest_path = required_root / str(contract.get("manifest_relative_path", ""))
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Campaign dataset manifest is missing: {manifest_path}")
    digest = hashlib.sha256()
    with manifest_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Campaign dataset manifest hash mismatch for {manifest_path}: expected {expected_sha256}, got {actual_sha256}."
        )

    split_path = Path(str(cfg.train_split_path)).expanduser().resolve(strict=False)
    required_split = str(contract.get("split", ""))
    if required_split and str(cfg.data.train_split) != required_split:
        raise ValueError(
            f"Campaign dataset contract requires data.train_split={required_split!r}, got {str(cfg.data.train_split)!r}."
        )
    if not split_path.is_file():
        raise FileNotFoundError(f"Campaign dataset split is missing: {split_path}")


def _initialize_stage2_from_campaign_parent(cfg: DictConfig, model_module) -> dict | None:
    """Load Stage-2 model weights from the frozen A0 parent without restoring training state.

    Stage-2 is a new run: optimizer, scheduler, epoch and global-step state must start from zero.
    Only tensors in the Lightning module state dict are copied.  The embedded campaign identity is
    checked here again even though the dependency gate verifies the same checkpoint before launch.
    """
    campaign = cfg.get("campaign")
    if not campaign or str(campaign.get("phase", "")) != "stage2_refinement":
        return None
    if bool(cfg.resume_training):
        raise ValueError("Stage-2 parent initialization cannot be combined with resume_training=true.")

    parent_path = Path(str(campaign.get("parent_checkpoint", ""))).expanduser()
    if not parent_path.is_file():
        raise FileNotFoundError(f"Stage-2 parent checkpoint is missing: {parent_path}")
    checkpoint = torch.load(parent_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError(f"Stage-2 parent is not a loadable Lightning checkpoint: {parent_path}")

    expected_step = int(campaign.get("parent_checkpoint_step"))
    observed_step = int(checkpoint.get("global_step", -1))
    identity = checkpoint.get("fomo26_campaign")
    if not isinstance(identity, dict):
        raise ValueError("Stage-2 parent checkpoint has no fomo26_campaign identity.")
    parent_launch_git_commit = str(
        campaign.get("parent_launch_git_commit")
        or os.environ.get("FOMO26_PARENT_LAUNCH_GIT_COMMIT")
        or campaign.get("launch_git_commit")
    )
    expected = {
        "candidate_id": str(campaign.get("parent_candidate_id")),
        "protocol": str(campaign.get("protocol")),
        "campaign_sha256": str(campaign.get("campaign_sha256")),
        "source_git_commit": str(campaign.get("source_git_commit")),
        "launch_git_commit": parent_launch_git_commit,
        "architecture": str(campaign.get("architecture")),
    }
    errors = [
        f"{key}: parent={identity.get(key)!r}, expected={value!r}"
        for key, value in expected.items()
        if str(identity.get(key)) != value
    ]
    identity_step = int(identity.get("global_step", -1))
    if observed_step != expected_step or identity_step != expected_step:
        errors.append(f"global_step: checkpoint={observed_step}, identity={identity_step}, expected={expected_step}")
    parent_contract = identity.get("dataset_contract")
    current_contract = campaign.get("dataset_contract")
    for key in ("manifest_sha256", "tensor_manifest_sha256", "accepted_tensors"):
        parent_value = parent_contract.get(key) if isinstance(parent_contract, dict) else None
        current_value = current_contract.get(key) if current_contract is not None else None
        if str(parent_value) != str(current_value):
            errors.append(f"dataset_contract.{key}: parent={parent_value!r}, expected={current_value!r}")
    if errors:
        raise ValueError("Stage-2 parent identity mismatch: " + "; ".join(errors))

    source = checkpoint["state_dict"]
    target = model_module.state_dict()
    shared = {}
    shape_mismatches = []
    for key, value in source.items():
        if key not in target:
            continue
        if tuple(value.shape) != tuple(target[key].shape):
            shape_mismatches.append((key, tuple(value.shape), tuple(target[key].shape)))
        else:
            shared[key] = value
    if shape_mismatches:
        raise ValueError(f"Stage-2 parent has shared-key shape mismatches: {shape_mismatches[:8]}")
    source_model_keys = {key for key in source if key.startswith("model.")}
    missing_source_model = sorted(source_model_keys - set(target))
    matched_model = source_model_keys & set(shared)
    coverage = len(matched_model) / len(source_model_keys) if source_model_keys else 0.0
    if missing_source_model or not source_model_keys or coverage < 0.99:
        raise ValueError(
            "Stage-2 parent model coverage failed: "
            f"matched={len(matched_model)}/{len(source_model_keys)} coverage={coverage:.4f} "
            f"missing_source_model={missing_source_model[:8]}"
        )
    incompatible = model_module.load_state_dict(shared, strict=False)
    report = {
        "parent_checkpoint": str(parent_path.resolve()),
        "parent_global_step": observed_step,
        "matched_tensors": len(shared),
        "matched_model_tensors": len(matched_model),
        "source_model_tensors": len(source_model_keys),
        "model_coverage": coverage,
        "parent_launch_git_commit": parent_launch_git_commit,
        "technical_launch_git_commit": str(campaign.get("launch_git_commit")),
        "new_target_tensors": sorted(incompatible.missing_keys),
    }
    py_logging.info("Initialized Stage-2 weights from frozen A0 parent: %s", report)
    return report


@hydra.main(
    config_path=get_config_path(),
    config_name="default_pretrain",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    # The Jean-Zay child-runtime fingerprint check lived here. It compared the interpreter,
    # venv and pinned torch/numpy build against one specific cluster installation, and armed
    # itself only from FOMO26_RUNTIME_FINGERPRINT, which that cluster's bootstrap set. The HPC
    # orchestration rail is not part of the public code release, so the check went with it
    # rather than being kept as a stub that would always pass and guarantee nothing.
    run_started_at = time.perf_counter()
    if _is_rank_zero_process():
        print(f"{OmegaConf.to_yaml(cfg)}\n Version: {cfg.run_id}\n Run dir: {HydraConfig.get().run.dir}\n")
    _validate_pretrain_execution_mode(cfg)
    _validate_ssl_objective_contract(cfg)
    _validate_campaign_dataset_contract(cfg)
    ssl_objective = _ssl_objective(cfg)
    file_store, path_store, version_store = prepare_standard_experiment(cfg)
    fit_ckpt_path = None
    completion_contract = None
    if not cfg.validation_only:
        fit_ckpt_path = resolve_training_resume_checkpoint(
            path_store.ckpt_save_dir,
            required=bool(cfg.resume_training),
            resume_checkpoint_path=getattr(cfg, "resume_checkpoint_path", None),
        )
        if fit_ckpt_path and cfg.resume_training:
            py_logging.info("Resuming pretraining from same-run checkpoint: %s", fit_ckpt_path)
        elif fit_ckpt_path:
            py_logging.warning(
                "Found an existing same-run checkpoint and will resume for backward compatibility: %s. "
                "For scheduled restarts, set resume_training=true explicitly.",
                fit_ckpt_path,
            )
        if fit_ckpt_path:
            saved_seed = resolve_training_resume_seed(path_store.run_dir)
            if saved_seed is None:
                py_logging.warning(
                    "No saved validation_mask_seed was found in %s/hparams.yaml; "
                    "the resumed dataloader seed cannot be recovered automatically.",
                    path_store.run_dir,
                )
            else:
                cfg.training.seed = saved_seed
                py_logging.info("Reusing saved training seed for same-run continuation: %s", saved_seed)
        if bool((cfg.get("completion_validation") or {}).get("enabled", False)):
            completion_contract = _validate_completion_checkpoint(cfg, fit_ckpt_path)
    logging_safe_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    pl.seed_everything(seed=cfg.training.seed, workers=WORKER_SEEDING_WORKERS)

    plugins = prepare_ssl_plugins(cfg)

    assert cfg.task is not None, "Config file is not set up correctly."

    loggers = logging(
        ckpt_wandb_id=version_store.wandb_id,
        ckpt_mlflow_id=version_store.mlflow_id,
        log_file_name=HydraConfig.get().job.name,
        run_dir=path_store.run_dir,
        version=version_store.version,
        wandb_config=logging_safe_cfg,
        wandb_experiment=HydraConfig.get().job.config_name,
        wandb_project=cfg.logger.wandb_project,
        wandb_logging=cfg.logger.wandb_logging,
        mlflow_logging=cfg.logger.mlflow_logging,
        log_to_stdout=cfg.logger.log_to_stdout,
        # A multi-segment lane declares that all its segments belong to ONE W&B run. Without this,
        # a lost offline run directory silently starts a second run and splits the step history.
        wandb_require_run_continuity=bool(cfg.logger.get("wandb_require_run_continuity", False))
        and bool(cfg.get("resume_training", False)),
    )

    # Optional reliable offline metric capture ({run_dir}/.../metrics.csv). Off by default; handy for
    # short validation runs where the W&B offline summary is not materialised.
    if cfg.logger.get("csv_logging", False):
        from lightning.pytorch.loggers import CSVLogger

        csv_logger = CSVLogger(save_dir=str(path_store.run_dir))
        if not loggers:
            loggers = [csv_logger]
        elif isinstance(loggers, (list, tuple)):
            loggers = list(loggers) + [csv_logger]
        else:
            loggers = [loggers, csv_logger]

    callbacks = _pretrain_progress_callbacks(cfg) + plugins
    if completion_contract is None:
        callbacks += [
            ModelCheckpoint(
                dirpath=path_store.ckpt_save_dir,
                every_n_epochs=cfg.model.ckpt_every_n_epoch,
                save_top_k=cfg.model.ckpt_save_top_k,
                save_last=cfg.model.ckpt_save_last,
                filename=cfg.model.ckpt_filename,
                auto_insert_metric_name=False,
                enable_version_counter=cfg.model.ckpt_enable_version_counter,
                monitor=cfg.model.get("ckpt_monitor"),
                mode=cfg.model.get("ckpt_mode", "min"),
            ),
            PersistentExceptionCheckpoint(dirpath=path_store.ckpt_save_dir, filename="on_exception"),
            LearningRateMonitor(logging_interval="epoch", log_momentum=True),
        ]
    else:
        # Standalone completion validation must neither save nor mutate a fit-time
        # checkpoint. The optimizer callback remains a defense-in-depth tripwire;
        # the primary guarantee is that _run_trainer_phase calls only validate().
        callbacks.append(ForbidOptimizerStep())

    campaign_cfg = cfg.get("campaign")
    if campaign_cfg:
        campaign_metadata = OmegaConf.to_container(campaign_cfg, resolve=True)
        campaign_metadata.update(
            {
                "model_config": str(cfg.model.net),
                "seed": int(cfg.training.seed),
                "patch_size": [int(value) for value in cfg.training.patch_size],
                "global_batch_size": int(cfg.training.global_batch_size),
                "batch_size": int(cfg.training.batch_size),
                "accumulate_grad_batches": int(cfg.training.accumulate_grad_batches),
                "number_of_devices": int(cfg.hardware.num_devices),
                "number_of_nodes": int(cfg.hardware.num_nodes),
                "precision": str(cfg.hardware.precision),
                "ssl_objective": str(ssl_objective),
                "compile_mode": cfg.hardware.compile_mode,
                "hydra_config_name": str(HydraConfig.get().job.config_name),
            }
        )
        scientific_invariant = {
            "campaign": {
                key: campaign_metadata.get(key)
                for key in (
                    "protocol",
                    "candidate_id",
                    "objective",
                    "architecture",
                    "campaign_sha256",
                    "source_git_commit",
                    "dataset_contract",
                    "parent_candidate_id",
                    "parent_checkpoint_step",
                )
            },
            "model": OmegaConf.to_container(cfg.model, resolve=True),
            "training": {
                key: OmegaConf.to_container(cfg.training, resolve=True).get(key)
                for key in (
                    "patch_size",
                    "batch_size",
                    "accumulate_grad_batches",
                    "global_batch_size",
                    "max_samples",
                    "steps",
                    "steps_per_epoch",
                    "warmup_ratio",
                    "warmup_steps",
                    "seed",
                )
            },
            "data": OmegaConf.to_container(cfg.data, resolve=True),
            "losses": OmegaConf.to_container(cfg.get("losses", {}), resolve=True),
            "ssl": OmegaConf.to_container(cfg.get("ssl", {}), resolve=True),
            "transforms": OmegaConf.to_container(cfg.transforms, resolve=True),
            "execution_geometry": {
                "devices": int(cfg.hardware.num_devices),
                "nodes": int(cfg.hardware.num_nodes),
                "precision": str(cfg.hardware.precision),
            },
        }
        invariant_blob = json.dumps(scientific_invariant, sort_keys=True, separators=(",", ":"), default=str).encode()
        campaign_metadata["scientific_invariant"] = scientific_invariant
        campaign_metadata["scientific_invariant_sha256"] = hashlib.sha256(invariant_blob).hexdigest()
        if completion_contract is None:
            callbacks.append(
                # `campaign_metadata` already carries `launch_git_commit` -- the submit-time
                # commit -- and keeps it untouched. The execution commit is attested by the
                # callback from the tree it is running in, never passed in from the campaign.
                CampaignCheckpointMetadata(campaign_metadata, worker_seeding_policy=WORKER_SEEDING_POLICY)
            )

        checkpoint_steps = campaign_cfg.get("checkpoint_steps")
        if checkpoint_steps and completion_contract is None:
            callbacks.append(
                ExactOptimizerStepCheckpoint(
                    path_store.ckpt_save_dir,
                    checkpoint_steps,
                    stop_at_step=campaign_cfg.get("stop_at_checkpoint_step"),
                    validation_steps=campaign_cfg.get("validation_steps") or (),
                )
            )
        recovery_checkpoint_every = campaign_cfg.get("recovery_checkpoint_every_steps")
        if recovery_checkpoint_every and completion_contract is None:
            callbacks.append(RollingRecoveryCheckpoint(path_store.ckpt_save_dir, recovery_checkpoint_every))

        if completion_contract is None:
            # Pre-timeout: the Slurm batch shell writes STOP_AT_NEXT_BOUNDARY on --signal=B:USR1, and
            # this stops at the next completed optimizer step rather than mid-accumulation.
            callbacks.append(SafeBoundaryStopCallback(path_store.ckpt_save_dir, str(path_store.run_dir)))
            # NaN / OOM / corruption must not look like an ordinary walltime interruption to the
            # chained resubmission, or the rest of the chain reruns the same divergence.
            callbacks.append(UnrecoverableFailureMarker(str(path_store.run_dir)))

        technical_canary = cfg.get("technical_canary") or {}
        if bool(technical_canary.get("enabled", False)) and completion_contract is None:
            callbacks.append(
                TechnicalCanaryCheckpoint(
                    path_store.ckpt_save_dir,
                    str(path_store.run_dir),
                    step=int(technical_canary.max_optimizer_steps),
                    filename=str(technical_canary.checkpoint_filename),
                )
            )

        resume_audit_delta = campaign_cfg.get("resume_audit_delta_steps")
        # The resume audit is a short continuation *of a restored checkpoint*: it anchors its target
        # at the restored global_step and stops delta steps later. On a fresh run the anchor is 0, so
        # installing it unconditionally truncates the primary calibration at `delta` optimizer steps
        # instead of running to `stop_at_checkpoint_step`. Job 513678/513679 stopped at step 10 of an
        # 800-step calibration for exactly this reason. Only arm it when a checkpoint is restored.
        if resume_audit_delta is not None and completion_contract is None and fit_ckpt_path is not None:
            callbacks.append(StopAfterRelativeOptimizerSteps(int(resume_audit_delta)))

        # A manually submitted promotion is allowed to continue a deliberately
        # paused screening run. User-created STOP markers are never touched.
        allow_segment_resume = (
            bool(campaign_cfg.get("allow_segment_resume", False)) or str(campaign_cfg.get("phase", "")) == "full_2m"
        )
        if cfg.training.get("stop_after_steps") is None and allow_segment_resume and fit_ckpt_path is not None:
            pause_path = os.path.join(path_store.run_dir, "TRAINING_PAUSED")
            # Every DDP rank executes this module before the process group exists, so an
            # isfile()-then-remove() pair is a TOCTOU race: two ranks both observe the marker and
            # the loser dies with FileNotFoundError. That is what killed the r003 q2/q3 resumes for
            # A1, D0, D1, S00, M1 and C1 -- hours of valid training discarded by a lost race on a
            # file every rank merely wants *gone*. Deleting a marker is idempotent by nature, so
            # unlink it unconditionally and let "already absent" be success.
            removed = _unlink_if_present(pause_path)
            if removed:
                py_logging.info("Removed campaign pause marker before an explicit checkpoint continuation: %s", pause_path)

    if cfg.profiler.enabled:
        callbacks.append(fast_instantiate(cfg.profiler._callback))

    stop_after_steps = cfg.training.get("stop_after_steps")
    if stop_after_steps is not None and completion_contract is None:
        if int(stop_after_steps) >= int(cfg.training.steps):
            raise ValueError(
                "training.stop_after_steps must be smaller than training.steps; remove it when no early stop is required."
            )
        pause_marker = bool(campaign_cfg and campaign_cfg.get("pause_at_stop", False))
        callbacks.append(StopAfterValidationStep(int(stop_after_steps), pause_marker=pause_marker))

    cpu_tr_transforms = fast_instantiate(
        cfg.transforms._cpu_tr_transforms,
        patch_size=cfg.training.patch_size,
    )
    cpu_val_transforms = fast_instantiate(
        cfg.transforms._cpu_val_transforms,
        patch_size=cfg.training.patch_size,
    )
    cpu_demo_tr_transforms = fast_instantiate(
        cfg.transforms._cpu_demo_tr_transforms,
        patch_size=cfg.training.patch_size,
    )
    outer_masking = _outer_masking_enabled(cfg)
    gpu_tr_transforms = fast_instantiate(
        cfg.transforms._gpu_tr_transforms,
        outer_masking,
        ndim=len(cfg.training.patch_size),
        mask_ratio=cfg.training.mask_ratio,
        mask_policy=cfg.transforms.get("mask_policy", "random"),
        mask_token_size=cfg.transforms.get("mask_token_size", [4]),
        mask_foreground_bias=cfg.transforms.get("mask_foreground_bias", 0.5),
        mask_foreground_threshold=cfg.transforms.get("mask_foreground_threshold", 0.0),
        mask_foreground_dynamic_quantiles=cfg.transforms.get("mask_foreground_dynamic_quantiles", [0.02, 0.98]),
        mask_foreground_dynamic_scale=cfg.transforms.get("mask_foreground_dynamic_scale", 0.1),
        frepa_lite=cfg.transforms.get("frepa_lite", None),
    )
    gpu_unmasked_tr_transforms = fast_instantiate(
        cfg.transforms._gpu_unmasked_tr_transforms,
        ndim=len(cfg.training.patch_size),
        mask_ratio=cfg.training.mask_ratio,
        mask_policy=cfg.transforms.get("mask_policy", "random"),
        mask_token_size=cfg.transforms.get("mask_token_size", [4]),
        mask_foreground_bias=cfg.transforms.get("mask_foreground_bias", 0.5),
        mask_foreground_threshold=cfg.transforms.get("mask_foreground_threshold", 0.0),
        mask_foreground_dynamic_quantiles=cfg.transforms.get("mask_foreground_dynamic_quantiles", [0.02, 0.98]),
        mask_foreground_dynamic_scale=cfg.transforms.get("mask_foreground_dynamic_scale", 0.1),
    )
    gpu_val_transforms = fast_instantiate(
        cfg.transforms._gpu_val_transforms,
        outer_masking,
        mask_ratio=cfg.training.mask_ratio,
        mask_policy=cfg.transforms.get("mask_policy", "random"),
        mask_token_size=cfg.transforms.get("mask_token_size", [4]),
        mask_foreground_bias=cfg.transforms.get("mask_foreground_bias", 0.5),
        mask_foreground_threshold=cfg.transforms.get("mask_foreground_threshold", 0.0),
        mask_foreground_dynamic_quantiles=cfg.transforms.get("mask_foreground_dynamic_quantiles", [0.02, 0.98]),
        mask_foreground_dynamic_scale=cfg.transforms.get("mask_foreground_dynamic_scale", 0.1),
    )

    model = fast_instantiate(
        cfg.model._pretrain_net,
    )
    if not getattr(model, "supports_reconstruction", True):
        raise ValueError(
            f"{type(model).__name__} is encoder-only and does not support AMAES/reconstruction. Select a decoder-backed model."
        )

    same_session_batches = bool(cfg.data.same_session_multimodal_batches)
    multimodal_batch_probability = float(cfg.data.multimodal_batch_probability) if same_session_batches else 0.0
    data_module = fast_instantiate(
        cfg.lightning._data_module,
        train_split=file_store.splits["train"],
        val_split=file_store.splits["val"],
        train_transforms=cpu_tr_transforms,
        val_transforms=cpu_val_transforms,
        metadata_paths=cfg.data.metadata_paths,
        registered_mapping_path=cfg.data.registered_mapping_path,
        registered_only=cfg.data.registered_only,
        same_session_multimodal_batches=same_session_batches,
        multimodal_batch_probability=multimodal_batch_probability,
        stage1_multimodal_batch_mode=cfg.data.stage1_multimodal_batch_mode,
        stage1_multimodal_allowed_pairs=cfg.data.stage1_multimodal_allowed_pairs,
        demographic_batch_probability=cfg.data.demographic_batch_probability,
        demographic_age_bin_years=cfg.data.demographic_age_bin_years,
        max_modalities_per_subject=cfg.data.max_modalities_per_subject,
        sampler_seed=cfg.training.seed,
        validate_split_disjointness=cfg.data.validate_split_disjointness,
        complete_validation=cfg.data.complete_validation,
        routine_scan_count=cfg.monitoring.routine_scan_count,
        monitor_seed=cfg.monitoring.monitor_seed,
        probe_train_max_subjects=cfg.monitoring.probe_train_max_subjects,
        demographic_probe_train_max_subjects=cfg.monitoring.demographic_probe_train_max_subjects,
        probe_val_canonical_t1w=cfg.monitoring.probe_val_canonical_t1w,
        stage1_pair_session_count=cfg.monitoring.stage1_pair_session_count,
        require_demographic_samples=(cfg.losses.demographic.enabled or cfg.data.demographic_batch_probability > 0.0),
        require_multimodal_samples=cfg.losses.multimodal_stage1.enabled,
        require_registered_multimodal_samples=cfg.losses.multimodal_stage2.enabled,
        # Driven by the SAMPLING policy, not the loss flag: `raw_image` re-bases the augmented
        # views in views.py (center-crop instead of random-crop), so tying it to
        # `losses.demographic.enabled` would make a demographic control and its treatment see
        # different patches. Both arms set the same demographic_batch_probability, so both get it.
        return_raw_image=bool(cfg.data.demographic_batch_probability > 0.0),
        scanner_targets_enabled=cfg.data.scanner_targets.enabled,
        scanner_target_keys=cfg.data.scanner_targets["keys"],
        scanner_target_ignore_index=cfg.data.scanner_targets.ignore_index,
        scanner_target_spacing_bins=cfg.data.scanner_targets.spacing_bins,
        scanner_target_spacing_summary=cfg.data.scanner_targets.spacing_summary,
        scanner_target_spacing_columns=cfg.data.scanner_targets.spacing_columns,
        save_scanner_target_vocab=cfg.data.scanner_targets.save_vocab,
    )

    if (cfg.losses.scanner_pos.enabled or cfg.losses.inv_adv.enabled) and not cfg.data.scanner_targets.enabled:
        raise ValueError("Scanner SSL branches require data.scanner_targets.enabled=true.")
    if cfg.data.scanner_targets.enabled:
        data_module.setup("fit")

    model_module = fast_instantiate(
        cfg.lightning._lightning_module,
        model=model,
        learning_rate=cfg.model.pretrain_lr,
        warmup_epochs=cfg.training.warmup_epochs,
        warmup_steps=cfg.training.get("warmup_steps"),
        train_transforms=gpu_tr_transforms,
        unmasked_transforms=gpu_unmasked_tr_transforms,
        val_transforms=gpu_val_transforms,
        demo_cpu_transforms=cpu_demo_tr_transforms,
        rec_loss_masked_only=cfg.training.rec_loss_masked_only,
        mse_foreground_aware=cfg.losses.mse.get("foreground_aware", False),
        mse_background_weight=cfg.losses.mse.get("background_weight", 0.1),
        mse_foreground_threshold=cfg.losses.mse.get("foreground_threshold", 0.0),
        mse_foreground_dynamic_quantiles=cfg.losses.mse.get("foreground_dynamic_quantiles", [0.02, 0.98]),
        mse_foreground_dynamic_scale=cfg.losses.mse.get("foreground_dynamic_scale", 0.1),
        mse_foreground_sample_normalize=cfg.losses.mse.get("foreground_sample_normalize", True),
        mse_foreground_mode=cfg.losses.mse.get("foreground_mode", None),
        mse_foreground_bonus_weight=cfg.losses.mse.get("foreground_bonus_weight", 0.0),
        mse_foreground_bonus_patch_size=cfg.losses.mse.get("foreground_bonus_patch_size", [4]),
        mse_foreground_min_fraction=cfg.losses.mse.get("foreground_min_fraction", 0.1),
        mse_foreground_bonus_start_step=cfg.losses.mse.get("foreground_bonus_start_step", 0),
        mse_foreground_bonus_warmup_steps=cfg.losses.mse.get("foreground_bonus_warmup_steps", 0),
        matched_control=OmegaConf.to_container(cfg.ssl.matched_control, resolve=True)
        if cfg.ssl.get("matched_control", None) is not None
        else None,
        optimizer=cfg.model.pretrain_optim,
        mlflow_logging=cfg.logger.mlflow_logging,
        log_every_n_steps=cfg.logger.log_every_n_steps,
        log_images_every_n_epoch=cfg.logger.log_images_every_n_epoch,
        loss_weight_mse=cfg.losses.mse.weight,
        enable_mse_loss=bool(cfg.losses.mse.enabled),
        component_starvation_window=cfg.training.component_starvation_window,
        batch_routing_log_every_n_steps=cfg.logger.batch_routing_log_every_n_steps,
        validation_embedding_monitor_enabled=cfg.logger.validation_embeddings.enabled,
        validation_embedding_sources=cfg.logger.validation_embeddings.sources,
        validation_embedding_reducers=cfg.logger.validation_embeddings.reducers,
        validation_embedding_max_points=cfg.logger.validation_embeddings.max_points,
        validation_embedding_random_state=cfg.logger.validation_embeddings.random_state,
        validation_mask_seed=cfg.training.seed,
        loss_gradient_norm_every_n_steps=cfg.logger.loss_gradient_norm_every_n_steps,
        reconstruction_data_range=cfg.monitoring.reconstruction_data_range,
        log_raw_reconstruction_metrics=cfg.monitoring.log_raw_reconstruction_metrics,
        metric_semantics_version=cfg.monitoring.metric_semantics_version,
        probe_every_n_epoch=cfg.monitoring.probe_every_n_epoch,
        probe_detailed_metrics=cfg.monitoring.detailed_probe_metrics,
        exhaustive_evaluation=cfg.monitoring.exhaustive_evaluation,
    )
    _initialize_stage2_from_campaign_parent(cfg, model_module)

    if cfg.data.complete_validation and not cfg.validation_only:
        py_logging.warning(
            "data.complete_validation=true overrides training.val_steps_per_epoch=%s and evaluates the full "
            "validation split every validation event. Use this only for explicit exhaustive runs.",
            cfg.training.val_steps_per_epoch,
        )
    elif not cfg.validation_only:
        py_logging.info(
            "Routine validation evaluates the full fixed monitoring cohort of up to %s scans; "
            "training.val_steps_per_epoch=%s is retained only for backward-compatible configuration display.",
            cfg.monitoring.routine_scan_count,
            cfg.training.val_steps_per_epoch,
        )
    # Gradient clipping (automatic optimization). Passed only when configured so
    # gradient_clip_val=null cleanly disables it without tripping Lightning's
    # "algorithm set but val is None" guard.
    gradient_clip_kwargs = {}
    if cfg.training.get("gradient_clip_val", None) is not None:
        gradient_clip_kwargs["gradient_clip_val"] = cfg.training.gradient_clip_val
        gradient_clip_kwargs["gradient_clip_algorithm"] = cfg.training.get("gradient_clip_algorithm", "norm")

    trainer = fast_instantiate(
        cfg.lightning._trainer,
        callbacks=callbacks,
        log_every_n_steps=cfg.logger.log_every_n_steps,
        logger=loggers,
        default_root_dir=path_store.run_dir,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        max_steps=cfg.training.steps,
        limit_train_batches=cfg.training.steps_per_epoch,
        # The datamodule already supplies either the fixed monitoring cohort or the
        # explicit exhaustive cohort. Truncating it would invalidate cohort comparisons.
        limit_val_batches=1.0,
        use_distributed_sampler=False,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        **gradient_clip_kwargs,
    )

    if trainer.is_global_zero:
        optimizer_steps_per_pseudo_epoch = max(
            1,
            math.ceil(cfg.training.steps_per_epoch / cfg.training.accumulate_grad_batches),
        )
        print("Training duration configured as:")
        print(f"  - Steps: {cfg.training.steps}")
        print(f"  - Global batch size: {cfg.training.global_batch_size}")
        print(f"  - Steps per pseudo epoch: {cfg.training.steps_per_epoch}")
        validation_batches = (
            "all deterministic validation batches"
            if cfg.data.complete_validation
            else f"fixed routine cohort (up to {cfg.monitoring.routine_scan_count} scans)"
        )
        print(f"  - Validation batches per pseudo epoch: {validation_batches}")
        print("  - Pseudo Epochs: {:.1f}".format(cfg.training.steps / optimizer_steps_per_pseudo_epoch))
        print(f"  - Optimizer steps per pseudo epoch: {optimizer_steps_per_pseudo_epoch}")
        print(f"  - Warmup Pseudo Epochs: {cfg.training.warmup_epochs} (ratio {cfg.training.warmup_ratio})")

    execution_mode = _run_trainer_phase(
        cfg,
        trainer,
        model_module,
        data_module,
        fit_ckpt_path=fit_ckpt_path,
        completion_contract=completion_contract,
    )
    if execution_mode == "completion_validation":
        checkpoint_visible = trainer.strategy.broadcast(os.path.isfile(fit_ckpt_path), src=0)
        if not checkpoint_visible:
            raise FileNotFoundError(f"authoritative completion checkpoint disappeared: {fit_ckpt_path}")
        trainer.strategy.barrier("completion_validation_checkpoint_sync")

        # Lightning may move the module back to CPU when standalone validation
        # tears down, while the NCCL process group remains active. Collectives
        # must still use the rank-local CUDA device.
        memory = torch.zeros(3, dtype=torch.float64, device=_completion_collective_device())
        if torch.cuda.is_available():
            memory[0] = torch.cuda.max_memory_allocated() / 1024**3
            memory[1] = torch.cuda.max_memory_reserved() / 1024**3
            memory[2] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(memory, op=torch.distributed.ReduceOp.MAX)
        trainer.strategy.barrier("completion_validation_metrics_sync")

        if trainer.is_global_zero:
            actual_sha256 = _sha256_file(fit_ckpt_path)
            if actual_sha256 != completion_contract["sha256"]:
                raise ValueError(
                    "authoritative checkpoint changed during completion validation: "
                    f"expected {completion_contract['sha256']}, got {actual_sha256}."
                )
            from asparagus.pipeline.resource_metrics import write_completion_validation_artifacts

            write_completion_validation_artifacts(
                path_store.run_dir,
                model_module,
                run_started_at,
                checkpoint_path=fit_ckpt_path,
                checkpoint_sha256=completion_contract["sha256"],
                checkpoint_step=completion_contract["step"],
                checkpoint_epoch=completion_contract["epoch"],
                global_batch_size=int(cfg.training.global_batch_size),
                devices=int(cfg.hardware.num_devices) * int(cfg.hardware.num_nodes),
                source_git_commit=os.environ.get("FOMO_GIT_COMMIT", ""),
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                cuda_max_memory_allocated_gib=float(memory[0].item()) if torch.cuda.is_available() else None,
                cuda_max_memory_reserved_gib=float(memory[1].item()) if torch.cuda.is_available() else None,
                cuda_device_total_memory_gib=float(memory[2].item()) if torch.cuda.is_available() else None,
            )
    elif execution_mode == "fit":
        if trainer.is_global_zero:
            from asparagus.pipeline.resource_metrics import write_resource_metrics

            write_resource_metrics(
                path_store.run_dir,
                model_module,
                run_started_at,
                devices=int(cfg.hardware.num_devices) * int(cfg.hardware.num_nodes),
                samples_processed=int(trainer.global_step) * int(cfg.training.global_batch_size),
                optimizer_steps=int(trainer.global_step),
            )
        if trainer.is_global_zero and int(getattr(trainer, "global_step", 0)) >= int(cfg.training.steps):
            done_path = os.path.join(path_store.run_dir, "TRAINING_DONE")
            with open(done_path, "w", encoding="utf-8") as stream:
                stream.write(f"global_step={int(getattr(trainer, 'global_step', 0))}\n")
                stream.write(f"target_steps={int(cfg.training.steps)}\n")


if __name__ == "__main__":
    main()
