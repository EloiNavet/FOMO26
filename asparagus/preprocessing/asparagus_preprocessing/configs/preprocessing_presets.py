import os
from asparagus_preprocessing.utils.dataclasses import PreprocessingConfig, SavingConfig


def _parse_target_size(value: str) -> list[int]:
    parts = value.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError("ASPARAGUS_PREPROCESSING_TARGET_SIZE must contain exactly three integers.")
    target_size = [int(part) for part in parts]
    if any(size <= 0 for size in target_size):
        raise ValueError("ASPARAGUS_PREPROCESSING_TARGET_SIZE values must be positive.")
    return target_size


def _parse_spacing(value: str) -> list[float]:
    parts = value.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError("ASPARAGUS_PREPROCESSING_TARGET_SPACING must contain exactly three values.")
    spacing = [float(part) for part in parts]
    if any(value <= 0 for value in spacing):
        raise ValueError("ASPARAGUS_PREPROCESSING_TARGET_SPACING values must be positive.")
    return spacing


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _apply_runtime_preprocessing_overrides(config: PreprocessingConfig) -> PreprocessingConfig:
    target_size_value = os.environ.get("ASPARAGUS_PREPROCESSING_TARGET_SIZE")
    output_size_value = os.environ.get("ASPARAGUS_PREPROCESSING_OUTPUT_SIZE")
    target_spacing_value = os.environ.get("ASPARAGUS_PREPROCESSING_TARGET_SPACING")
    downsample_factor_value = os.environ.get("ASPARAGUS_PREPROCESSING_DOWNSAMPLE_FACTOR")
    if sum(value is not None for value in (target_size_value, target_spacing_value, downsample_factor_value)) > 1:
        raise ValueError("Use only one of target size, target spacing, or downsample factor.")

    if target_size_value:
        config.target_size = _parse_target_size(target_size_value)
        config.target_spacing = None
        config.downsample_factor = None
        config.keep_aspect_ratio_when_using_target_size = _parse_bool(
            os.environ.get("ASPARAGUS_PREPROCESSING_KEEP_ASPECT_RATIO", "1")
        )

    if target_spacing_value:
        config.target_spacing = _parse_spacing(target_spacing_value)
        config.target_size = None
        config.downsample_factor = None

    if output_size_value:
        config.output_size = _parse_target_size(output_size_value)

    if downsample_factor_value:
        downsample_factor = float(downsample_factor_value)
        if downsample_factor <= 0:
            raise ValueError("ASPARAGUS_PREPROCESSING_DOWNSAMPLE_FACTOR must be positive.")
        config.downsample_factor = downsample_factor
        config.target_size = None
        config.target_spacing = None

    overflow_policy = os.environ.get("ASPARAGUS_PREPROCESSING_OVERFLOW_POLICY")
    if overflow_policy is not None:
        if overflow_policy not in {"reject"}:
            raise ValueError("ASPARAGUS_PREPROCESSING_OVERFLOW_POLICY must be 'reject'.")
        config.overflow_policy = overflow_policy

    allow_legacy_value = os.environ.get("ASPARAGUS_PREPROCESSING_ALLOW_LEGACY_ANISOTROPIC_TARGET_SIZE")
    if allow_legacy_value is not None:
        config.allow_legacy_anisotropic_target_size = _parse_bool(allow_legacy_value)

    crop_to_nonzero_value = os.environ.get("ASPARAGUS_PREPROCESSING_CROP_TO_NONZERO")
    if crop_to_nonzero_value is not None:
        config.crop_to_nonzero = _parse_bool(crop_to_nonzero_value)

    return config


def get_noresampling_preprocessing_config(n_modalities: int = 1) -> PreprocessingConfig:
    return _apply_runtime_preprocessing_overrides(
        PreprocessingConfig(
            normalization_operation=["no_norm"] * n_modalities,
            target_spacing=None,  # native spacing
            target_orientation="RAS",
            crop_to_nonzero=False,
            min_slices=15,
        )
    )


def get_iso_preprocessing_config(n_modalities: int = 1) -> PreprocessingConfig:
    return _apply_runtime_preprocessing_overrides(
        PreprocessingConfig(
            normalization_operation=["no_norm"] * n_modalities,
            target_spacing=[1.0, 1.0, 1.0],
            target_orientation="RAS",
            crop_to_nonzero=False,
            min_slices=15,
        )
    )


def get_FOMO300K_saving_config(save_as_tensor: bool, save_dset_metadata: bool, bidsify: bool) -> SavingConfig:
    if save_as_tensor == True:  # noqa: E712
        save_file_metadata = True
    else:
        save_file_metadata = False
    return SavingConfig(
        bidsify=bidsify,
        save_as_tensor=save_as_tensor,
        tensor_dtype="float32",
        save_dset_metadata=save_dset_metadata,
        save_file_metadata=save_file_metadata,
    )


def get_FOMO_saving_config(save_as_tensor: bool) -> SavingConfig:
    return SavingConfig(
        bidsify=False,
        save_as_tensor=save_as_tensor,
        tensor_dtype="float32",
        save_dset_metadata=False,
        save_file_metadata=True,
    )
