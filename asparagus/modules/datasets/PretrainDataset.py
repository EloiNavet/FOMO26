import hashlib
import logging
import os
import pandas as pd
import torch
import torchvision
from asparagus.functional.loading import load_image_file
from asparagus.functional.scanner_targets import ScannerTargetEncoder
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional, Sequence


def _is_rank_zero_process() -> bool:
    return os.environ.get("RANK", os.environ.get("LOCAL_RANK", os.environ.get("SLURM_PROCID", "0"))) == "0"


class PretrainDataset(Dataset):
    _warned_unknown_scanners = set()
    _metadata_cache = {}
    _n_scrubbed_images = 0
    _n_scrubbed_labels = 0
    KNOWN_UNMAPPED_SCANNER_MANUFACTURERS = {
        "brucker",
        "fuji film co., ltd.",
        "mediso",
        "ningbo xingaoyi",
        "visage pr",
    }
    KNOWN_NON_PATHOLOGY_GROUPS = {
        "motion artefact",
    }

    MODALITY_TO_ID = {
        "t1w": 0,
        "t2w": 1,
        "flair": 2,
        "t1c": 3,
        "dwi": 4,
        "dwi_trace": 5,
        "adc": 6,
        "swi": 7,
        "gre": 8,
        "asl": 9,
        "m0scan": 10,
        "cbf": 11,
        "pdw": 12,
        "mp2rage": 13,
        "unit1": 14,
    }

    # ---- Stage 5: FOMO26 curated diffusion vocabulary (opt-in, backward-compatible) ----
    # The curated FOMO300K_curated_v2 dataset names ADC and DWI_B1000 as *distinct*
    # modalities (never the collapsed generic ``dwi``). These are ADDITIVE ids -- existing
    # ids 0-14 are never renumbered -- so enabling the curated vocab cannot change any
    # legacy id. It is gated by ``_USE_CURATED_VOCAB`` (default False) so current training
    # is byte-for-byte unchanged until explicitly switched on via
    # ``use_curated_diffusion_vocab(True)`` or the ``ASPARAGUS_CURATED_DIFFUSION_VOCAB``
    # env var (e.g. when the dataset path points at the curated tree).
    _CURATED_MODALITY_ADDITIONS = {
        "dwi_b1000": 15,
        "dwi_b0": 16,
        "t2star": 17,
    }
    # Curated diffusion suffixes recognised (in priority order) when the curated vocab is
    # active. ``adc`` / ``dwi_trace`` already exist as legacy ids and stay distinct.
    _CURATED_DIFFUSION_SUFFIXES = ("dwi_b1000", "dwi_b0", "dwi_trace", "adc", "t2star")
    # Input aliases normalised to a canonical vocab key.
    MODALITY_ALIASES = {"t2": "t2w", "t2s": "t2star", "t2starw": "t2star"}
    # FOMO26 default pretraining modality set (curated). GRE / T1c are excluded unless
    # explicitly enabled (mirrors the curation ``--enable-gre-t1c`` flag).
    DEFAULT_PRETRAIN_STRUCTURAL = ("t1w", "t2w", "flair", "t2star", "swi")
    DEFAULT_PRETRAIN_DIFFUSION = ("dwi_b1000", "adc")
    OPTIONAL_PRETRAIN_MODALITIES = ("gre", "t1c")

    _USE_CURATED_VOCAB = str(os.environ.get("ASPARAGUS_CURATED_DIFFUSION_VOCAB", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    @classmethod
    def use_curated_diffusion_vocab(cls, enabled: bool = True) -> None:
        """Enable/disable the FOMO26 curated diffusion vocabulary (process-wide switch)."""
        cls._USE_CURATED_VOCAB = bool(enabled)

    @classmethod
    def curated_diffusion_vocab_enabled(cls) -> bool:
        return bool(cls._USE_CURATED_VOCAB)

    @classmethod
    def active_modality_vocab(cls) -> dict[str, int]:
        """Modality vocab in effect: legacy by default, curated additions when enabled."""
        vocab = dict(cls.MODALITY_TO_ID)
        if cls._USE_CURATED_VOCAB:
            vocab.update(cls._CURATED_MODALITY_ADDITIONS)
        return vocab

    @classmethod
    def _normalize_modality_key(cls, key) -> str:
        key = str(key).strip().lower()
        return cls.MODALITY_ALIASES.get(key, key)

    @classmethod
    def default_pretrain_modalities(cls, enable_gre_t1c: bool = False) -> tuple[str, ...]:
        """FOMO26 default pretraining modalities; GRE/T1c only when explicitly enabled."""
        modalities = cls.DEFAULT_PRETRAIN_STRUCTURAL + cls.DEFAULT_PRETRAIN_DIFFUSION
        if enable_gre_t1c:
            modalities = modalities + cls.OPTIONAL_PRETRAIN_MODALITIES
        return modalities

    @classmethod
    def normalize_modality_ids(
        cls,
        modalities: Optional[Sequence[int | str]],
        default: Sequence[int | str] = ("t1w",),
    ) -> tuple[int, ...]:
        values = default if modalities is None else modalities
        resolved = []
        vocab = cls.active_modality_vocab()
        supported_ids = set(vocab.values())
        for value in values:
            if isinstance(value, int):
                modality_id = int(value)
            else:
                key = cls._normalize_modality_key(value)
                modality_id = vocab.get(key, -1)
            if modality_id not in supported_ids:
                raise ValueError(f"Unsupported pretraining modality {value!r}.")
            if modality_id not in resolved:
                resolved.append(modality_id)
        if not resolved:
            raise ValueError("At least one pretraining modality must be configured.")
        return tuple(resolved)

    @classmethod
    def modality_name(cls, modality_id: int) -> str:
        inverse = {value: key for key, value in cls.active_modality_vocab().items()}
        return inverse.get(int(modality_id), f"modality_{int(modality_id)}")

    @classmethod
    def modality_vocab(cls) -> dict[str, int]:
        """Deterministic Stage-1 modality vocabulary (canonical name -> class id)."""
        return cls.active_modality_vocab()

    # ---- Demographic (Dufumier) modality taxonomy -------------------------------------
    # The demographic objective is validated on these structural modalities. DWI is NOT a
    # single demographic class: the collapsed ``dwi`` id mixes heterogeneous b-value
    # contrasts (b0 ~ T2, b1000 clinical, ultra-high-b research), so it is blocked here.
    # Only explicitly validated b-value subtypes below may be used, each as its own
    # candidate set (never compared across b-values). This taxonomy lives only in the
    # demographic path: MODALITY_TO_ID / FiLM / Stage-1 modality / scanner are untouched
    # (a DWI subtype keeps modality_id == MODALITY_TO_ID["dwi"]).
    DEMOGRAPHIC_SUPPORTED_STRUCTURAL = ("t1w", "t2w", "flair")
    # Legacy (uncurated) DWI: `dwi` is one collapsed id (4) mixing b-values, so a demographic
    # subtype is defined by a b-value band parsed from the `dwi_bval<N>` filename token.
    DEMOGRAPHIC_DWI_SUBTYPES = {
        # b ~ 1000 s/mm^2 clinical DWI band (audit: multi-site, sex-balanced, ~T2w-tier).
        "dwi_b1000": {"bval_min": 900.0, "bval_max": 1100.0},
    }
    # Curated (FOMO26 curated tree) DWI: each b-value is already its own explicit modality id
    # (`_DWI_B1000` -> id 15, `_DWI_B0` -> id 16), so the demographic subtype is a plain modality
    # match with NO b-value filter. Only these curated diffusion channels are demographic-eligible
    # (adc/dwi_trace/t2star are out of scope for the age/sex objective).
    DEMOGRAPHIC_CURATED_DWI_SUBTYPES = ("dwi_b1000", "dwi_b0")

    @classmethod
    def demographic_dwi_subtype_names(cls) -> tuple[str, ...]:
        if cls._USE_CURATED_VOCAB:
            return tuple(cls.DEMOGRAPHIC_CURATED_DWI_SUBTYPES)
        return tuple(cls.DEMOGRAPHIC_DWI_SUBTYPES)

    @classmethod
    def resolve_demographic_token(cls, name) -> dict:
        """Resolve a demographic modality token to ``{name, modality_id, bval_min, bval_max}``.

        Accepts the validated structural modalities and the explicit DWI b-value subtypes.
        With the curated diffusion vocab enabled, a DWI subtype (``dwi_b1000``/``dwi_b0``) is a
        clean explicit modality id with no b-value filter; without it, ``dwi_b1000`` is the legacy
        b-value band on the collapsed ``dwi`` id. Rejects the collapsed ``dwi`` class and any
        non-validated token. ``bval_min``/``bval_max`` are ``None`` for structural and curated
        tokens.
        """
        key = cls._normalize_modality_key(name)
        if key in cls.DEMOGRAPHIC_SUPPORTED_STRUCTURAL:
            return {"name": key, "modality_id": cls.MODALITY_TO_ID[key], "bval_min": None, "bval_max": None}
        if cls._USE_CURATED_VOCAB:
            if key in cls.DEMOGRAPHIC_CURATED_DWI_SUBTYPES:
                return {"name": key, "modality_id": cls._CURATED_MODALITY_ADDITIONS[key], "bval_min": None, "bval_max": None}
        elif key in cls.DEMOGRAPHIC_DWI_SUBTYPES:
            spec = cls.DEMOGRAPHIC_DWI_SUBTYPES[key]
            return {
                "name": key,
                "modality_id": cls.MODALITY_TO_ID["dwi"],
                "bval_min": float(spec["bval_min"]),
                "bval_max": float(spec["bval_max"]),
            }
        if key == "dwi" or key.startswith("dwi"):
            raise ValueError(
                f"Demographic DWI modality {name!r} is not an explicitly validated subtype "
                f"(curated vocab {'ON' if cls._USE_CURATED_VOCAB else 'OFF'}). The collapsed 'dwi' "
                "class mixes heterogeneous b-value contrasts and remains blocked; validated DWI "
                f"subtypes are {sorted(cls.demographic_dwi_subtype_names())}."
            )
        raise ValueError(
            "Demographic Dufumier modalities are restricted to "
            f"{list(cls.DEMOGRAPHIC_SUPPORTED_STRUCTURAL)} plus validated DWI subtypes "
            f"{sorted(cls.demographic_dwi_subtype_names())}; got {name!r}."
        )

    def __init__(
        self,
        files: list,
        transforms: Optional[torchvision.transforms.Compose] = None,
        metadata_paths: Optional[Sequence[str]] = None,
        registered_mapping_path: Optional[str] = None,
        registered_only: bool = False,
        return_raw_image: bool = False,
        scanner_target_encoder: Optional[ScannerTargetEncoder] = None,
        scanner_spacing_metadata_path: Optional[str] = None,
    ):
        super().__init__()

        self.transforms = transforms
        self.return_raw_image = bool(return_raw_image)
        self.scanner_target_encoder = scanner_target_encoder
        self._allow_synthetic_metadata = not metadata_paths
        metadata_cache_key = self._metadata_cache_key(
            metadata_paths,
            registered_mapping_path,
            scanner_spacing_metadata_path,
        )
        if metadata_cache_key not in self._metadata_cache:
            self._metadata_cache[metadata_cache_key] = self._load_metadata(
                metadata_paths,
                registered_mapping_path,
                scanner_spacing_metadata_path,
            )
        self.metadata_dict = self._metadata_cache[metadata_cache_key]
        self.files = list(files)
        if registered_only:
            self.files = [file for file in self.files if self._resolve_sample_identity(file)["is_registered_subset"]]

        # Pre-compute and cache the metadata for each file to avoid CPU bottleneck in __getitem__
        self._cached_metadata = []
        for file in self.files:
            sample_identity = self._resolve_sample_identity(file)
            sample_metadata = self._metadata_for_sample(sample_identity)
            common_fields = self._build_common_metadata_fields(sample_metadata)
            self._cached_metadata.append({"identity": sample_identity, "raw": sample_metadata, "common": common_fields})

    def set_scanner_target_encoder(self, scanner_target_encoder: Optional[ScannerTargetEncoder]) -> None:
        self.scanner_target_encoder = scanner_target_encoder
        for cached in self._cached_metadata:
            cached["common"] = self._build_common_metadata_fields(cached["raw"])

    def _build_common_metadata_fields(self, sample_metadata: dict) -> dict:
        common_fields = self._select_common_metadata_fields(sample_metadata)
        if self.scanner_target_encoder is not None:
            common_fields["scanner_targets"] = self.scanner_target_encoder.encode(sample_metadata)
        return common_fields

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = load_image_file(file)

        cached_meta = self._cached_metadata[idx]
        data_dict = {
            "file_path": file,
            "image": data,
            "transforms_applied": {},
            "metadata": cached_meta["raw"],
        }
        if self.return_raw_image:
            data_dict["raw_image"] = data.clone()
        data_dict.update(cached_meta["identity"])
        data_dict.update(cached_meta["common"])
        data_dict = self._transform(data_dict)  # CPU transforms only here

        # Defensive scrub of NaN/Inf left by upstream preprocessing. The fill
        # values sit inside the normalized/clamped intensity range used downstream
        # (see Clamp min=-2, max=4 in transforms/presets/pretrain.py). NOTE: nan->0
        # lands mid-range (i.e. fake tissue intensity) rather than at the -2
        # background floor; this is preserved for run reproducibility but is worth
        # revisiting. Occurrences are counted/logged so silent data corruption is
        # visible rather than masked.
        image, image_was_scrubbed = self._scrub_nonfinite(data_dict["image"], nan=0.0, posinf=4.0, neginf=-1.0)
        if image_was_scrubbed:
            PretrainDataset._n_scrubbed_images += 1
            logging.warning(
                f"Image contained NaN/Inf and was scrubbed (occurrence #{PretrainDataset._n_scrubbed_images}): {file}"
            )
            data_dict["image"] = image

        if "label" in data_dict.keys() and data_dict["label"] is not None:
            label, label_was_scrubbed = self._scrub_nonfinite(data_dict["label"], nan=0.0, posinf=4.0, neginf=-1.0)
            if label_was_scrubbed:
                PretrainDataset._n_scrubbed_labels += 1
                logging.warning(
                    f"Label contained NaN/Inf and was scrubbed (occurrence #{PretrainDataset._n_scrubbed_labels}): {file}"
                )
                data_dict["label"] = label

        return data_dict

    @staticmethod
    def _scrub_nonfinite(value, nan: float, posinf: float, neginf: float):
        if torch.is_tensor(value):
            if torch.isnan(value).any() or torch.isinf(value).any():
                return torch.nan_to_num(value, nan=nan, posinf=posinf, neginf=neginf), True
            return value, False
        if isinstance(value, list):
            scrubbed = [PretrainDataset._scrub_nonfinite(item, nan, posinf, neginf) for item in value]
            return [item for item, _ in scrubbed], any(changed for _, changed in scrubbed)
        if isinstance(value, tuple):
            scrubbed = [PretrainDataset._scrub_nonfinite(item, nan, posinf, neginf) for item in value]
            return tuple(item for item, _ in scrubbed), any(changed for _, changed in scrubbed)
        return value, False

    def _transform(self, data_dict):
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
        return data_dict

    def _load_metadata(
        self,
        metadata_paths: Optional[Sequence[str]],
        registered_mapping_path: Optional[str],
        scanner_spacing_metadata_path: Optional[str] = None,
    ) -> dict:
        metadata = {
            "participant": {},
            "scan": {},
            "scan_basename": {},
            "registered_scan": set(),
            "registered_scan_basename": set(),
        }
        if not metadata_paths:
            return metadata

        for metadata_path in metadata_paths:
            if metadata_path is None:
                raise ValueError("PretrainDataset received a null metadata path.")

            path = Path(metadata_path)
            if not path.exists():
                raise FileNotFoundError(f"Required FOMO metadata file does not exist: {path}")

            table = pd.read_csv(path, sep="\t", low_memory=False)

            for _, row in table.iterrows():
                row_dict = {str(column).strip().lower(): value for column, value in row.to_dict().items()}

                participant_key = self._participant_key_from_row(row_dict)
                scan_key = self._scan_key_from_row(row_dict)
                if scan_key is not None:
                    metadata["scan"].setdefault(scan_key, {}).update(row_dict)
                    dataset, participant, session, filename = scan_key
                    basename_key = (dataset, participant, session, self._filename_basename(filename))
                    metadata["scan_basename"].setdefault(basename_key, {}).update(row_dict)
                elif participant_key is not None:
                    metadata["participant"].setdefault(participant_key, {}).update(row_dict)

        if scanner_spacing_metadata_path is not None:
            self._merge_scanner_spacing_metadata(metadata, Path(scanner_spacing_metadata_path))

        if registered_mapping_path is not None:
            mapping_path = Path(registered_mapping_path)
            if not mapping_path.exists():
                raise FileNotFoundError(f"Required FOMO50K/60K mapping file does not exist: {mapping_path}")
            metadata.update(self._load_registered_mapping(mapping_path))

        return metadata

    @staticmethod
    def _metadata_cache_key(
        metadata_paths: Optional[Sequence[str]],
        registered_mapping_path: Optional[str],
        scanner_spacing_metadata_path: Optional[str] = None,
    ) -> tuple:
        paths = list(metadata_paths or [])
        if registered_mapping_path is not None:
            paths.append(registered_mapping_path)
        if scanner_spacing_metadata_path is not None:
            paths.append(scanner_spacing_metadata_path)
        key = []
        for path_value in paths:
            path = Path(path_value)
            if path.exists():
                stat = path.stat()
                key.append((str(path.resolve()), stat.st_mtime_ns, stat.st_size))
            else:
                key.append((str(path), None, None))
        return tuple(key)

    def _merge_scanner_spacing_metadata(self, metadata: dict, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Required native-spacing metadata file does not exist: {path}")
        table = pd.read_csv(path, sep="\t", low_memory=False)
        normalized_columns = {str(column).strip().lower() for column in table.columns}
        spacing_columns = tuple(
            column
            for column in ("original_spacing", "pixdim", "source_spacing", "native_spacing", "voxel_spacing")
            if column in normalized_columns
        )
        if not spacing_columns:
            raise ValueError(f"Native-spacing metadata file has no supported spacing column: {path}")

        matched_rows = 0
        for _, row in table.iterrows():
            row_dict = {str(column).strip().lower(): value for column, value in row.to_dict().items()}
            scan_key = self._scan_key_from_row(row_dict)
            if scan_key is None:
                continue
            spacing_values = {
                column: row_dict[column]
                for column in spacing_columns
                if column in row_dict and not pd.isna(row_dict[column]) and str(row_dict[column]).strip()
            }
            if not spacing_values:
                continue
            metadata["scan"].setdefault(scan_key, {}).update(spacing_values)
            dataset, participant, session, filename = scan_key
            basename_key = (dataset, participant, session, self._filename_basename(filename))
            metadata["scan_basename"].setdefault(basename_key, {}).update(spacing_values)
            matched_rows += 1
        if matched_rows == 0:
            raise ValueError(f"Native-spacing metadata file has no usable scan rows: {path}")

    def _metadata_for_sample(self, sample_identity: dict) -> dict:
        metadata = {}

        participant_key = self._make_participant_key(
            sample_identity.get("dataset"),
            sample_identity.get("participant_id"),
            sample_identity.get("session_id"),
        )
        if participant_key in self.metadata_dict["participant"]:
            metadata.update(self.metadata_dict["participant"][participant_key])

        scan_key = self._make_scan_key(
            sample_identity.get("dataset"),
            sample_identity.get("participant_id"),
            sample_identity.get("session_id"),
            sample_identity.get("filename"),
        )
        if scan_key in self.metadata_dict["scan"]:
            metadata.update(self.metadata_dict["scan"][scan_key])
        else:
            basename_key = (
                sample_identity.get("dataset"),
                sample_identity.get("participant_id"),
                sample_identity.get("session_id"),
                self._filename_basename(sample_identity.get("filename")),
            )
            if basename_key in self.metadata_dict["scan_basename"]:
                metadata.update(self.metadata_dict["scan_basename"][basename_key])

        if not metadata:
            if self._allow_synthetic_metadata:
                return {
                    "dataset": sample_identity.get("dataset"),
                    "participant_id": sample_identity.get("participant_id"),
                    "session_id": sample_identity.get("session_id"),
                    "filename": sample_identity.get("filename"),
                    "modality": sample_identity.get("modality"),
                }
            raise KeyError(
                "No FOMO metadata matched sample "
                f"{sample_identity.get('dataset')} / {sample_identity.get('participant_id')} / "
                f"{sample_identity.get('session_id')} / {sample_identity.get('filename')}"
            )

        return metadata

    def _resolve_sample_identity(self, file_path: str) -> dict:
        normalized = str(file_path).replace("\\", "/")
        parts = [part for part in normalized.split("/") if part]

        subject_idx = None
        for idx, part in enumerate(parts):
            if self._is_participant_component(part):
                subject_idx = idx
                break

        dataset = ""
        participant = ""
        session = ""
        filename = Path(normalized).name

        if subject_idx is not None:
            participant = self._canonicalize_component(parts[subject_idx])
            pt_idx = None
            for idx in range(subject_idx - 1, -1, -1):
                if parts[idx].lower().startswith("pt"):
                    pt_idx = idx
                    break
            if pt_idx is not None:
                dataset = self._canonicalize_component("/".join(parts[pt_idx:subject_idx]))

            if subject_idx + 1 < len(parts) and parts[subject_idx + 1].lower().startswith(("ses-", "ses_")):
                session = self._canonicalize_component(parts[subject_idx + 1])

            filename = "/".join(parts[subject_idx:])

        if not dataset or not participant or not session or not filename:
            if self._allow_synthetic_metadata:
                stem = self._canonicalize_component(Path(normalized).stem or "synthetic")
                dataset = "synthetic"
                participant = f"sub-{stem}"
                session = "ses-000"
                filename = Path(normalized).name
            else:
                raise ValueError(
                    "Could not resolve strict FOMO sample identity from path. Expected "
                    f".../PT*/sub-*/ses-*/.../<filename>, got: {normalized}"
                )

        if not dataset or not participant or not session or not filename:
            raise ValueError(
                "Could not resolve strict FOMO sample identity from path. Expected "
                f".../PT*/sub-*/ses-*/.../<filename>, got: {normalized}"
            )

        scan_key = self._make_scan_key(dataset, participant, session, filename)
        basename_key = (*self._make_participant_key(dataset, participant, session), self._filename_basename(filename))
        is_registered_subset = scan_key in self.metadata_dict.get(
            "registered_scan", set()
        ) or basename_key in self.metadata_dict.get("registered_scan_basename", set())

        modality = self._infer_modality_from_filename(filename)
        # Validate against the *active* vocab (curated diffusion ids 15-17 when the curated
        # vocab is enabled), not just the legacy 15-entry MODALITY_TO_ID -- otherwise a curated
        # `_DWI_B1000`/`_DWI_B0`/`_T2star` scan crashes here (or, under synthetic metadata,
        # silently falls back to t1w, mislabelling a DWI scan).
        vocab = self.active_modality_vocab()
        if self._allow_synthetic_metadata and modality not in vocab:
            modality = "t1w"
        if modality not in vocab:
            supported = ", ".join(sorted(vocab))
            raise ValueError(
                "Unsupported or unclassified MRI modality inferred from filename. "
                f"file={normalized!r}, inferred_modality={modality!r}, supported_modalities=[{supported}]. "
                "Add an explicit modality mapping (or enable ASPARAGUS_CURATED_DIFFUSION_VOCAB) "
                "before using this scan."
            )

        return {
            "file_path": normalized,
            "dataset": dataset,
            "participant_id": participant,
            "session_id": session,
            "filename": self._normalize_filename(filename),
            "subject_key": self._subject_key(dataset, participant),
            "subject_session_key": self._subject_session_key(dataset, participant, session),
            "modality": modality,
            "modality_id": vocab[modality],
            # Diffusion b-value parsed from a legacy `dwi_bval<N>` filename token (NaN otherwise,
            # including curated `_DWI_B1000` names which have no bval token -- there the b-value
            # is encoded by the modality id itself). Used only by the legacy demographic DWI
            # subtype taxonomy.
            "dwi_bval": self._infer_dwi_bval_from_filename(filename),
            # Stage-1 modality classification target. Equals modality_id here (dataset is
            # strict), but is conceptually separate: objective-level masking (e.g. excluding
            # the conflated dwi class) is applied downstream, leaving modality_id intact for
            # FiLM conditioning.
            "modality_target": vocab[modality],
            "is_registered_subset": is_registered_subset,
        }

    @staticmethod
    def _is_participant_component(value: str) -> bool:
        return str(value).lower().startswith(("sub-", "sub_"))

    @classmethod
    def _participant_key_from_row(cls, row_dict: dict) -> Optional[tuple]:
        dataset = row_dict.get("dataset")
        participant = row_dict.get("participant_id")
        session = row_dict.get("session_id")
        if cls._is_missing(dataset) or cls._is_missing(participant) or cls._is_missing(session):
            return None
        return cls._make_participant_key(dataset, participant, session)

    @classmethod
    def _scan_key_from_row(cls, row_dict: dict) -> Optional[tuple]:
        participant_key = cls._participant_key_from_row(row_dict)
        filename = row_dict.get("filename")
        if participant_key is None or cls._is_missing(filename):
            return None
        return (*participant_key, cls._normalize_filename(filename))

    @classmethod
    def _make_participant_key(cls, dataset, participant, session) -> tuple:
        return (
            cls._canonicalize_component(dataset),
            cls._canonicalize_component(participant),
            cls._canonicalize_component(session),
        )

    @classmethod
    def _make_scan_key(cls, dataset, participant, session, filename) -> tuple:
        return (*cls._make_participant_key(dataset, participant, session), cls._normalize_filename(filename))

    @classmethod
    def _load_registered_mapping(cls, mapping_path: Path) -> dict:
        table = pd.read_csv(mapping_path, sep="\t", low_memory=False)
        columns = {str(column).strip().lower() for column in table.columns}

        registered_scan = set()
        registered_scan_basename = set()

        if {"dataset_300k", "filename_300k"} <= columns:
            row_specs = (("dataset_300k", "filename_300k", None, None),)
        elif cls._is_fomo_registered_manifest_path(mapping_path) and "dataset" in columns:
            filename_column = next((key for key in ("new_filename", "new_path", "filename") if key in columns), None)
            if filename_column is None:
                raise ValueError(
                    f"Malformed FOMO50K/45K mapping {mapping_path}; expected one of new_filename, new_path, filename."
                )
            row_specs = (("dataset", filename_column, "participant_id", "session_id"),)
        else:
            raise ValueError(
                f"Malformed FOMO50K/60K mapping {mapping_path}; expected columns dataset_300K and filename_300K, "
                "or a generic FOMO50K/FOMO45K manifest path with dataset plus new_filename/new_path/filename."
            )

        for _, row in table.iterrows():
            row_dict = {str(column).strip().lower(): value for column, value in row.to_dict().items()}
            dataset_key, filename_key, participant_key, session_key = row_specs[0]
            dataset = row_dict.get(dataset_key)
            filename = row_dict.get(filename_key)
            if cls._is_missing(dataset) or cls._is_missing(filename):
                raise ValueError(f"Malformed registered mapping row without dataset/filename: {row_dict}")

            participant = row_dict.get(participant_key) if participant_key is not None else None
            session = row_dict.get(session_key) if session_key is not None else None
            if cls._is_missing(participant) or cls._is_missing(session):
                parsed = cls._participant_session_from_filename(filename)
                if parsed is None:
                    raise ValueError(f"Could not derive registered participant/session from filename={filename!r}")
                participant, session = parsed

            scan_key = cls._make_scan_key(dataset, participant, session, filename)
            basename_key = (*cls._make_participant_key(dataset, participant, session), cls._filename_basename(filename))
            registered_scan.add(scan_key)
            registered_scan_basename.add(basename_key)

        return {"registered_scan": registered_scan, "registered_scan_basename": registered_scan_basename}

    @staticmethod
    def _is_fomo_registered_manifest_path(mapping_path: Path) -> bool:
        normalized = str(mapping_path).replace("\\", "/").lower()
        return "fomo50k" in normalized or "fomo45k" in normalized

    @classmethod
    def _participant_session_from_filename(cls, filename) -> Optional[tuple[str, str]]:
        stem = cls._normalize_filename(filename)
        tokens = stem.replace("/", "_").split("_")
        participant = next((token for token in tokens if token.startswith(("sub-", "sub_"))), "")
        session = next((token for token in tokens if token.startswith(("ses-", "ses_"))), "")
        if not participant or not session:
            return None
        return cls._canonicalize_component(participant), cls._canonicalize_component(session)

    @staticmethod
    def _canonicalize_component(value) -> str:
        if PretrainDataset._is_missing(value):
            return ""
        return str(value).strip().replace("\\", "/").lower()

    @staticmethod
    def _normalize_filename(value) -> str:
        if PretrainDataset._is_missing(value):
            return ""
        text = str(value).strip().replace("\\", "/").lower()
        for suffix in (".nii.gz", ".nii", ".pt", ".pkl"):
            if text.endswith(suffix):
                text = text[: -len(suffix)]
                break
        return text

    @staticmethod
    def _filename_basename(value) -> str:
        if PretrainDataset._is_missing(value):
            return ""
        return PretrainDataset._normalize_filename(str(value).replace("\\", "/").split("/")[-1])

    @staticmethod
    def _subject_key(dataset, participant) -> str:
        return "|".join(
            [
                PretrainDataset._canonicalize_component(dataset),
                PretrainDataset._canonicalize_component(participant),
            ]
        )

    @staticmethod
    def _subject_session_key(dataset, participant, session) -> str:
        return "|".join(
            [
                PretrainDataset._canonicalize_component(dataset),
                PretrainDataset._canonicalize_component(participant),
                PretrainDataset._canonicalize_component(session),
            ]
        )

    @staticmethod
    def _stable_int_hash(value) -> int:
        digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
        # Mask to 52 bits (< 2^53) so ids round-trip exactly through a float64
        # mantissa when logged (e.g. queue/site_id metrics). Collisions across the
        # handful of datasets / hundreds of diagnosis groups are negligible.
        return int.from_bytes(digest, byteorder="little", signed=False) & 0xFFFFFFFFFFFFF

    @staticmethod
    def _infer_modality_from_filename(value) -> str:
        stem = PretrainDataset._filename_basename(value)
        tokens = [token for token in stem.split("_") if not token.startswith(("sub-", "sub_", "ses-", "ses_", "run-", "run_"))]
        # Lower-case for matching; legacy names are already lower-case, curated output
        # names (e.g. ``..._DWI_B1000``) are upper-case.
        series = "_".join(tokens).lower()

        # Curated FOMO26 diffusion names (opt-in) take priority so ADC / DWI_B1000 /
        # DWI_B0 / DWI_TRACE resolve to their own distinct ids rather than generic 'dwi'.
        if PretrainDataset._USE_CURATED_VOCAB:
            for name in PretrainDataset._CURATED_DIFFUSION_SUFFIXES:
                if series == name or series.endswith(f"_{name}"):
                    return name

        if series.startswith("dwi_bval"):
            return "dwi"
        if series.startswith("dwi_trace"):
            return "dwi_trace"
        # NOTE: ``dwi_bval*`` intentionally collapses to ``dwi`` for modality_id / FiLM /
        # Stage-1 in the legacy vocab. The b-value itself is parsed separately by
        # ``_infer_dwi_bval_from_filename`` and used only by the demographic taxonomy.

        vocab = PretrainDataset.active_modality_vocab()
        for modality in vocab:
            if series == modality:
                return modality
        for modality in vocab:
            if series.endswith(f"_{modality}") or series.endswith(modality):
                return modality
        return series or "unknown"

    @staticmethod
    def _infer_dwi_bval_from_filename(value) -> float:
        """Parse the diffusion b-value from a ``dwi_bval<NNN>`` filename token.

        Returns the leading integer b-value as a float, or NaN when the scan is not a
        b-value-tagged DWI series (e.g. structural scans or the generic ``dwi`` suffix
        with no b-value). Mirrors the series tokenization used for modality inference so
        the parse is consistent with ``_infer_modality_from_filename``.
        """
        stem = PretrainDataset._filename_basename(value)
        tokens = [token for token in stem.split("_") if not token.startswith(("sub-", "sub_", "ses-", "ses_", "run-", "run_"))]
        series = "_".join(tokens)
        if not series.startswith("dwi_bval"):
            return float("nan")
        digits = ""
        for char in series[len("dwi_bval") :]:
            if char.isdigit():
                digits += char
            else:
                break
        return float(digits) if digits else float("nan")

    @staticmethod
    def _modality_id_from_filename(value) -> int:
        modality = PretrainDataset._infer_modality_from_filename(value)
        vocab = PretrainDataset.active_modality_vocab()
        if modality not in vocab:
            supported = ", ".join(sorted(vocab))
            raise ValueError(
                f"Unsupported or unclassified MRI modality {modality!r} inferred from {value!r}; "
                f"supported modalities are [{supported}]."
            )
        return vocab[modality]

    @staticmethod
    def _is_missing(value) -> bool:
        if value is None:
            return True
        try:
            missing = pd.isna(value)
        except TypeError:
            return False
        if not hasattr(missing, "__len__"):
            return bool(missing)
        return False

    @staticmethod
    def _select_common_metadata_fields(sample_metadata: dict) -> dict:
        fields = {}

        # 1. Age Mapping
        age_raw = sample_metadata.get("age")
        try:
            if not PretrainDataset._is_missing(age_raw):
                fields["age"] = float(age_raw)
            else:
                fields["age"] = float("nan")
        except (TypeError, ValueError):
            fields["age"] = float("nan")

        # 2. Sex Mapping
        sex_raw = sample_metadata.get("sex")
        sex_str = str(sex_raw).strip().lower() if not PretrainDataset._is_missing(sex_raw) else ""

        if sex_str == "m":
            fields["sex"] = 0
        elif sex_str == "f":
            fields["sex"] = 1
        else:
            fields["sex"] = -1

        # 3. Pathology Macro-Class Mapping
        group_raw = sample_metadata.get("group")
        fields["pathology"] = -1  # Default to unknown (-1)
        fields["fine_pathology"] = -1

        if not PretrainDataset._is_missing(group_raw):
            g = str(group_raw).strip().lower()
            if g:
                fields["fine_pathology"] = PretrainDataset._stable_int_hash(g)

            # 1: NEURODEGENERATIVE (AD/PD/movement disorders)
            # Evaluated before controls because some PD labels include "normal cognition".
            if any(k in g for k in ["parkinson", "movement", "dystonia"]) or g == "pd":
                fields["pathology"] = 1

            # 0: CONTROL
            # Evaluated first to ensure "nondemented" overrides "demented"
            elif (
                (
                    any(k in g for k in ["control", "normal", "nondemented", "neurotypical", "typically developing"])
                    or g
                    in [
                        "cn",
                        "hc",
                        "nh",
                        "overweight",
                        "normalweight",
                    ]
                )
                and not any(k in g for k in ["parkinson", "movement", "dystonia"])
                and g != "pd"
            ):
                fields["pathology"] = 0

            # 1: NEURODEGENERATIVE (AD/dementia)
            # "converted" is typically MCI -> AD in ADNI
            elif any(k in g for k in ["alzheimer", "dement", "frontotemporal"]) or g in ["ad", "converted", "ftd"]:
                fields["pathology"] = 1

            # 2: PSYCHIATRIC & NEURODEVELOPMENTAL
            elif any(
                k in g
                for k in [
                    "schiz",
                    "schz",
                    "bipolar",
                    "depress",
                    "obsessive",
                    "ocd",
                    "anxiety",
                    "psychosis",
                    "psychiatric",
                    "adhd",
                    "autism",
                    "dyslex",
                    "attention deficit",
                    "hyperactivity",
                    "borderline",
                    "prosopagnos",
                    "fragile x",
                    "spelling",
                    "synaesthesia",
                    "emotional dysregulation",
                    "cocaine use disorder",
                    "substance use",
                    "addiction",
                    "lncg",
                ]
            ) or g in ["bp"]:
                fields["pathology"] = 2

            # 3: TUMOR / ONCOLOGY
            elif any(
                k in g
                for k in [
                    "tumor",
                    "gliom",
                    "astrocytom",
                    "meningiom",
                    "ependymom",
                    "medulloblastoma",
                    "lymphoma",
                    "adenoma",
                    "oncolog",
                    "metasta",
                    "neuroectoderm",
                    "gangliogliom",
                    "pylocytic",
                    "dnet",
                    "gbm",
                    "gmb",
                    "pituitary",
                    "blastoma",
                    "fasciitis",
                ]
            ):
                fields["pathology"] = 3

            # 4: VASCULAR / HEMORRHAGE
            # Placed before structural to catch compound names like "premature pvl" or "hie cerebral oedeme"
            elif any(k in g for k in ["stroke", "infarct", "hemorrhag", "aneurysm", "hie"]):
                fields["pathology"] = 4

            # 5: OTHER STRUCTURAL
            # Broad net for morphological, congenital, metabolic, or physical damage
            elif any(
                k in g
                for k in [
                    "epilepsy",
                    "seizure",
                    "hydrocephalus",
                    "cyst",
                    "dysplas",
                    "sclerosis",
                    "malformation",
                    "atrophy",
                    "heterotopia",
                    "injury",
                    "lesion",
                    "tbi",
                    "structural",
                    "developmental",
                    "encephalocele",
                    "macrocephaly",
                    "pvl",
                    "cmv",
                    "abscess",
                    "cephaly",
                    "syndrom",
                    "dandy-walker",
                    "gliotic",
                    "meningitis",
                    "meningocele",
                    "mithocondriopathy",
                    "motor neuron",
                    "neurological",
                    "olfactory",
                    "neuropathy",
                    "ataxia",
                    "hearing loss",
                    "oedem",
                    "calcification",
                    "synostosis",
                    "malasy",
                    "encephalopathy",
                    "microglia",
                    "microgyria",
                    "premature",
                    "hygrom",
                    "melanin",
                    "postoperative",
                    "injur",
                    "fibromyalgia",
                    "osteoarthritis",
                    "arthritis",
                ]
            ) or g in ["t1d", "nf 1", "nf1", "ms"]:
                fields["pathology"] = 5

            elif g not in PretrainDataset.KNOWN_NON_PATHOLOGY_GROUPS:
                raise ValueError(
                    "Unclassified pathology group encountered in PretrainDataset metadata: "
                    f"{group_raw!r}. Every non-empty disease label must map to a macro pathology class. "
                    "Update PretrainDataset._select_common_metadata_fields before using this dataset."
                )

        # 4. Scanner ID Mapping
        manufacturer_raw = sample_metadata.get("manufacturer")
        fields["scanner_id"] = -1
        if not PretrainDataset._is_missing(manufacturer_raw):
            m = str(manufacturer_raw).strip().lower()
            if "siemens" in m:
                fields["scanner_id"] = 0
            elif "philips" in m:
                fields["scanner_id"] = 1
            elif m == "ge" or m.startswith("ge ") or "general electric" in m or "general electrics" in m:
                fields["scanner_id"] = 2
            elif "toshiba" in m or "canon" in m:
                fields["scanner_id"] = 3
            elif "hitachi" in m:
                fields["scanner_id"] = 4

        # If scanner ID is still unknown raise an error and print the value for manual review
        if fields["scanner_id"] == -1 and not PretrainDataset._is_missing(manufacturer_raw):
            manufacturer_key = str(manufacturer_raw)
            manufacturer_lookup = manufacturer_key.strip().lower()
            if (
                manufacturer_lookup not in PretrainDataset.KNOWN_UNMAPPED_SCANNER_MANUFACTURERS
                and manufacturer_key not in PretrainDataset._warned_unknown_scanners
            ):
                PretrainDataset._warned_unknown_scanners.add(manufacturer_key)
                if _is_rank_zero_process():
                    logging.warning(f"Unknown scanner manufacturer '{manufacturer_raw}'; mapping to -1.")

        fields["domain_manufacturer_id"] = fields["scanner_id"]

        dataset_raw = sample_metadata.get("dataset")
        fields["dataset_id"] = -1
        if not PretrainDataset._is_missing(dataset_raw):
            fields["dataset_id"] = PretrainDataset._stable_int_hash(PretrainDataset._canonicalize_component(dataset_raw))
        return fields
