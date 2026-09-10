import logging
import math
import nibabel as nib
import numpy as np
import os
from asparagus_preprocessing.utils.bbox import get_bbox_for_foreground
from asparagus_preprocessing.utils.crop import crop_to_box
from asparagus_preprocessing.utils.nifti import (
    apply_nifti_preprocessing_and_return_numpy,
)
from asparagus_preprocessing.utils.pad import get_pad_box, pad_case_to_size
from asparagus_preprocessing.utils.resample import resample_and_normalize_case, resample_case_to_spacing
from asparagus_preprocessing.utils.saving import save_data_and_metadata, save_geometry_qc
from dataclasses import asdict
from typing import List, Optional, Union

STRUCTURAL_MODALITIES = {"t1w", "t2w", "flair", "t1c", "pdw", "mp2rage", "unit1"}


class GeometryQCExclusion(ValueError):
    def __init__(self, reason: str, image_properties: dict):
        super().__init__(reason)
        self.reason = reason
        self.image_properties = image_properties


def _infer_modality(path: str) -> str:
    stem = os.path.basename(path).lower()
    for suffix in (".nii.gz", ".nii", ".pt", ".pkl"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    for modality in ("mp2rage", "unit1", "flair", "t1w", "t2w", "t1c", "pdw", "dwi", "adc", "asl", "m0scan", "cbf"):
        if modality in stem:
            return modality
    return "unknown"


def _translated_affine(affine: np.ndarray, offset: list[int] | np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = np.asarray(offset, dtype=float)
    return np.asarray(affine, dtype=float) @ transform


def _geometry_qc_base(image_properties: dict, output_size: list[int]) -> dict:
    original_size = np.asarray(image_properties["original_size"], dtype=int)
    original_spacing = np.asarray(image_properties["original_spacing"], dtype=float)
    return {
        "source_shape": original_size.tolist(),
        "source_spacing": original_spacing.tolist(),
        "source_fov_mm": (original_size * original_spacing).tolist(),
        "resampled_shape": [int(value) for value in image_properties["resampled_size_before_fit"]],
        "output_shape": [int(value) for value in image_properties["new_size"]],
        "output_spacing": [float(value) for value in image_properties["new_spacing"]],
        "output_size_limit": [int(value) for value in output_size],
        "crop_box": [int(value) for value in image_properties.get("crop_box", [])],
        "pad_box": [int(value) for value in image_properties.get("pad_box", [])],
    }


def _reject_physical_overflow_before_resampling(
    images: list[np.ndarray],
    image_properties: dict,
    working_affine: np.ndarray,
    original_size: tuple[int, ...],
    original_spacing: np.ndarray,
    target_spacing: Optional[List],
    output_size: Optional[List],
    overflow_policy: str,
) -> None:
    if target_spacing is None or output_size is None:
        return
    if overflow_policy != "reject":
        raise ValueError(f"Unsupported overflow_policy={overflow_policy!r}; only 'reject' is supported.")
    target_spacing_array = np.asarray(target_spacing, dtype=float)
    current_spacing = nib.affines.voxel_sizes(working_affine)
    resampled_size = np.maximum(
        np.round(np.asarray(images[0].shape, dtype=float) * current_spacing / target_spacing_array).astype(int),
        1,
    )
    if not any(current > maximum for current, maximum in zip(resampled_size, output_size)):
        return
    processed_affine = working_affine.copy()
    processed_affine[:3, :3] = working_affine[:3, :3] @ np.diag(target_spacing_array / current_spacing)
    image_properties.update(
        {
            "original_spacing": original_spacing.tolist(),
            "original_size": original_size,
            "resampled_size_before_fit": resampled_size.tolist(),
            "new_size": resampled_size.tolist(),
            "new_spacing": target_spacing_array.tolist(),
            "processed_affine": processed_affine,
            "pad_box": [],
        }
    )
    image_properties["geometry_qc"] = _geometry_qc_base(image_properties, output_size)
    raise GeometryQCExclusion("fov_overflow", image_properties)


def _finalize_geometry_qc(
    image_properties: dict,
    source_path: str,
    output_base_path: str,
    included: bool,
    exclusion_reason: str = "",
) -> None:
    if "geometry_qc" not in image_properties:
        return
    modality = _infer_modality(output_base_path)
    original_spacing = np.asarray(image_properties["original_spacing"], dtype=float)
    structural = modality in STRUCTURAL_MODALITIES
    spacing_eligible = bool(original_spacing.max() <= 2.5 + 1e-6)
    primary_structural_eligible = bool(included and structural and spacing_eligible)
    if included and structural and not spacing_eligible:
        primary_reason = "native_spacing_above_2p5mm"
    elif not included:
        primary_reason = exclusion_reason
    elif not structural:
        primary_reason = "non_structural_modality"
    else:
        primary_reason = ""

    image_properties["geometry_qc"].update(
        {
            "source_path": str(source_path),
            "output_base_path": str(output_base_path),
            "output_path": str(output_base_path) + ".pt",
            "modality": modality,
            "structural_modality": structural,
            "included": bool(included),
            "exclusion_reason": exclusion_reason,
            "primary_structural_eligible": primary_structural_eligible,
            "primary_structural_exclusion_reason": primary_reason,
        }
    )


def process_mri_case(path, image_save_path, preprocessing_config, saving_config):
    # TODO: if no metadata saving, do not check (look at saving_config)
    ext = ".pt" if saving_config.save_as_tensor else ".nii.gz"
    is_file = os.path.isfile(image_save_path + ext)
    is_geometry_excluded = os.path.isfile(image_save_path + ".geometry_qc.json") and not is_file

    if (
        is_geometry_excluded
        or (is_file and not saving_config.save_file_metadata)
        or (is_file and saving_config.save_file_metadata and os.path.isfile(image_save_path + ".pkl"))
    ):
        return
    try:
        image = nib.load(path)
        case, image_props = preprocess_case_without_label(images=[image], **asdict(preprocessing_config), strict=False)
        _finalize_geometry_qc(image_props, path, image_save_path, included=True)
        save_data_and_metadata(case, image_props, image_save_path, saving_config)
        del case, image, image_props
    except GeometryQCExclusion as e:
        _finalize_geometry_qc(e.image_properties, path, image_save_path, included=False, exclusion_reason=e.reason)
        save_geometry_qc(e.image_properties, image_save_path)
        logging.warning(f"Excluded by geometry QC ({e.reason}): {path}")
    except EOFError:
        logging.error(f"EOFError: {path} is corrupted.")
    except ValueError as e:
        logging.error(f"ValueError {e}: {path}")
    except Exception as e:
        logging.error(f"Unexpected error {e}: {path}")


def process_dwi_case(
    path,
    bvals_path,
    bvecs_path,
    image_save_path,
    preprocessing_config,
    saving_config,
    use_trace_computation=False,
    strict=True,
):
    try:
        image = nib.load(path)
        method_name = "trace computation" if use_trace_computation else "averaging"
        logging.debug(f"Processing {path} using {method_name} method")

        if len(image.shape) == 4 and image.shape[-1] != 1:
            if not os.path.exists(bvals_path) or not os.path.exists(bvecs_path):
                logging.error(f"SKIPPED: Missing bval or bvec for 4D DWI: {path}")
                return

            bvals = np.loadtxt(bvals_path)
            bvecs = np.loadtxt(bvecs_path)

            # Check if bvecs need transposing to (3, N) format
            if bvecs.shape[1] == 3 and bvecs.shape[0] > 3:
                # Convert from (N, 3) to (3, N) format
                bvecs = bvecs.T
            elif bvecs.shape[0] != 3:
                logging.error(f"Invalid bvecs shape: {bvecs.shape}. Expected (3, N) or (N, 3)")
                return

            images, bvals = extract_3ddwi_from_4ddwi(
                image,
                bvals,
                bvecs,
                use_trace_computation=use_trace_computation,
                strict=strict,
            )
        else:
            if len(image.shape) == 4:  # Must be shape[-1] == 1
                image = nib.Nifti1Image(np.squeeze(image.get_fdata(), axis=-1), image.affine, image.header)
            images = [image]
            bvals = [None]

        for idx, image in enumerate(images):
            # TODO: convoluted/hardcoded way. Find a nice way here
            if image_save_path.endswith(".nii.gz"):
                base = image_save_path[:-7]  # Remove .nii.gz
            elif image_save_path.endswith(".nii"):
                base = image_save_path[:-4]  # Remove .nii
            else:
                base = image_save_path
            if bvals[idx] is None:
                filename = base
            elif use_trace_computation and int(bvals[idx]) > 0:
                filename = f"{base}_bval_{bvals[idx]}_trace"
            else:
                filename = f"{base}_bval_{bvals[idx]}"
            ext = ".pt" if saving_config.save_as_tensor else ".nii.gz"

            if (
                (os.path.isfile(filename + ext) and not saving_config.save_file_metadata)
                or (os.path.isfile(filename + ext) and saving_config.save_file_metadata and os.path.isfile(filename + ".pkl"))
                or (os.path.isfile(filename + ".geometry_qc.json") and not os.path.isfile(filename + ext))
            ):
                continue

            try:
                case, image_props = preprocess_case_without_label(
                    images=[image], **asdict(preprocessing_config), strict=strict
                )
                _finalize_geometry_qc(image_props, path, filename, included=True)
                save_data_and_metadata(case, image_props, filename, saving_config)
            except GeometryQCExclusion as e:
                _finalize_geometry_qc(e.image_properties, path, filename, included=False, exclusion_reason=e.reason)
                save_geometry_qc(e.image_properties, filename)
                logging.warning(f"Excluded by geometry QC ({e.reason}): {path}")
                continue
            del case, image, image_props
    except AssertionError as e:
        logging.error(f"AssertionError {e}: {path}")
    except EOFError:
        logging.error(f"EOFError: {path} is corrupted.")
    except ValueError as e:
        logging.error(f"ValueError {e}: {path}")
    except Exception as e:
        logging.error(f"Unexpected error {e}: {path}")


def process_pet_case(path, image_save_path, preprocessing_config, saving_config, strict=True):
    ext = ".pt" if saving_config.save_as_tensor else ".nii.gz"

    if (
        (os.path.isfile(image_save_path + ".geometry_qc.json") and not os.path.isfile(image_save_path + ext))
        or (os.path.isfile(image_save_path + ext) and not saving_config.save_file_metadata)
        or (
            os.path.isfile(image_save_path + ext)
            and saving_config.save_file_metadata
            and os.path.isfile(image_save_path + ".pkl")
        )
    ):
        return
    try:
        image = nib.load(path)
        if len(image.shape) == 4:
            image = extract_3dpet_from_4dpet(image, strict=strict)
        case, image_props = preprocess_case_without_label(images=[image], **asdict(preprocessing_config), strict=False)
        _finalize_geometry_qc(image_props, path, image_save_path, included=True)
        save_data_and_metadata(case, image_props, image_save_path, saving_config)
        del case, image, image_props
    except GeometryQCExclusion as e:
        _finalize_geometry_qc(e.image_properties, path, image_save_path, included=False, exclusion_reason=e.reason)
        save_geometry_qc(e.image_properties, image_save_path)
        logging.warning(f"Excluded by geometry QC ({e.reason}): {path}")
    except AssertionError as e:
        logging.error(f"AssertionError {e}: {path}")
    except EOFError:
        logging.error(f"EOFError: {path} is corrupted.")
    except ValueError as e:
        logging.error(f"ValueError {e}: {path}")
    except Exception as e:
        logging.error(f"Unexpected error {e}: {path}")


def process_perf_case(
    path,
    image_save_path,
    m0scan_patterns,
    preprocessing_config,
    saving_config,
    strict=True,
):
    ext = ".pt" if saving_config.save_as_tensor else ".nii.gz"

    if (
        (os.path.isfile(image_save_path + ".geometry_qc.json") and not os.path.isfile(image_save_path + ext))
        or (os.path.isfile(image_save_path + ext) and not saving_config.save_file_metadata)
        or (
            os.path.isfile(image_save_path + ext)
            and saving_config.save_file_metadata
            and os.path.isfile(image_save_path + ".pkl")
        )
    ):
        return
    try:
        image = nib.load(path)

        # Determine if this is an M0 scan based on filename patterns
        filename = os.path.basename(path).lower()
        is_m0scan = any(pattern.lower() in filename for pattern in m0scan_patterns)

        if len(image.shape) == 4:
            image = extract_3dperfusion_from_4dperfusion(image, is_m0scan=is_m0scan, strict=strict)
        case, image_props = preprocess_case_without_label(images=[image], **asdict(preprocessing_config), strict=strict)
        _finalize_geometry_qc(image_props, path, image_save_path, included=True)
        save_data_and_metadata(case, image_props, image_save_path, saving_config)
        del case, image_props
    except GeometryQCExclusion as e:
        _finalize_geometry_qc(e.image_properties, path, image_save_path, included=False, exclusion_reason=e.reason)
        save_geometry_qc(e.image_properties, image_save_path)
        logging.warning(f"Excluded by geometry QC ({e.reason}): {path}")
    except AssertionError as e:
        logging.error(f"AssertionError {e}: {path}")
    except EOFError:
        logging.error(f"EOFError: {path} is corrupted.")
    except ValueError as e:
        logging.error(f"ValueError {e}: {path}")
    except Exception as e:
        logging.error(f"Unexpected error {e}: {path}")


def preprocess_case_without_label(
    images: List[Union[np.ndarray, nib.Nifti1Image]],
    normalization_operation: list,
    background_pixel_value: int = 0,
    crop_to_nonzero: bool = True,
    keep_aspect_ratio_when_using_target_size: bool = False,
    image_properties: Optional[dict] = {},
    intensities: Optional[List] = None,
    target_orientation: Optional[str] = "RAS",
    target_size: Optional[List] = None,
    output_size: Optional[List] = None,
    target_spacing: Optional[List] = None,
    downsample_factor: Optional[float] = None,
    overflow_policy: str = "reject",
    allow_legacy_anisotropic_target_size: bool = False,
    strict: bool = True,
    supposed_to_be_3D: bool = True,
    remove_nans=True,
    min_slices=15,
):
    images, label, image_properties["nifti_metadata"] = apply_nifti_preprocessing_and_return_numpy(
        images=images,
        original_size=np.array(images[0].shape),
        target_orientation=target_orientation,
        label=None,
        include_header=True,
        strict=strict,
    )

    images = [safe_squeeze(image) for image in images]
    verify_3D_image_is_valid(
        images,
        supposed_to_be_3D=supposed_to_be_3D,
        remove_nans=remove_nans,
        min_slices=min_slices,
    )
    original_size = images[0].shape
    original_spacing = np.asarray(image_properties["nifti_metadata"]["original_spacing"], dtype=float)
    working_affine = np.asarray(image_properties["nifti_metadata"]["affine"], dtype=float)
    if working_affine.shape != (4, 4):
        working_affine = np.diag([*original_spacing, 1.0])

    # Cropping is performed to save computational resources. We are only removing background.
    if crop_to_nonzero:
        nonzero_box = get_bbox_for_foreground(images[0], background_label=background_pixel_value)
        image_properties["crop_to_nonzero"] = nonzero_box
        image_properties["crop_box"] = [int(value) for value in nonzero_box]
        for i in range(len(images)):
            images[i] = crop_to_box(images[i], nonzero_box)
        working_affine = _translated_affine(working_affine, nonzero_box[::2])
    else:
        image_properties["crop_to_nonzero"] = crop_to_nonzero
        image_properties["crop_box"] = []

    _reject_physical_overflow_before_resampling(
        images,
        image_properties,
        working_affine,
        original_size,
        original_spacing,
        target_spacing,
        output_size,
        overflow_policy,
    )

    if target_size is not None and not allow_legacy_anisotropic_target_size:
        spacing_ratio = float(original_spacing.max() / original_spacing.min())
        if spacing_ratio > 1.1:
            raise ValueError(
                "Legacy target_size resizing is not physically valid for anisotropic medical images "
                f"(spacing ratio={spacing_ratio:.3f}). Use target_spacing plus output_size, or explicitly "
                "enable allow_legacy_anisotropic_target_size."
            )

    if target_spacing is not None:
        if target_size is not None or downsample_factor is not None:
            raise ValueError("target_size, target_spacing, and downsample_factor are mutually exclusive.")
        images, _, processed_affine = resample_case_to_spacing(
            case=images,
            source_affine=working_affine,
            target_spacing=target_spacing,
            norm_op=normalization_operation,
            intensities=intensities,
        )
        final_target_size = None
        new_spacing = [float(value) for value in target_spacing]
    else:
        resample_target_size, final_target_size, new_spacing = determine_target_size(
            images=images,
            original_spacing=original_spacing,
            target_size=target_size,
            target_spacing=target_spacing,
            downsample_factor=downsample_factor,
            keep_aspect_ratio=keep_aspect_ratio_when_using_target_size,
        )
        size_before_resample = np.asarray(images[0].shape, dtype=float)
        images = resample_and_normalize_case(
            case=images,
            target_size=resample_target_size,
            norm_op=normalization_operation,
            intensities=intensities,
            resample_method="yucca",
        )
        processed_affine = working_affine @ np.diag([*(size_before_resample / np.asarray(images[0].shape)), 1.0])

    if final_target_size is not None:
        legacy_pad_box = get_pad_box(images[0], final_target_size)
        images = pad_case_to_size(case=images, size=final_target_size, label=None)
        processed_affine = _translated_affine(
            processed_affine,
            [-legacy_pad_box[0], -legacy_pad_box[2], -legacy_pad_box[4]],
        )
        image_properties["pad_box"] = [int(value) for value in legacy_pad_box]

    image_properties["resampled_size_before_fit"] = list(images[0].shape)
    image_properties.setdefault("pad_box", [])
    if output_size is not None:
        if target_spacing is None:
            raise ValueError("output_size requires target_spacing so the output envelope has a physical interpretation.")
        output_size = [int(value) for value in output_size]
        if overflow_policy != "reject":
            raise ValueError(f"Unsupported overflow_policy={overflow_policy!r}; only 'reject' is supported.")
        if any(current > maximum for current, maximum in zip(images[0].shape, output_size)):
            image_properties["new_size"] = list(images[0].shape)
            image_properties["processed_affine"] = processed_affine
            image_properties["original_spacing"] = original_spacing.tolist()
            image_properties["original_size"] = original_size
            image_properties["new_spacing"] = new_spacing
            image_properties["geometry_qc"] = _geometry_qc_base(image_properties, output_size)
            raise GeometryQCExclusion("fov_overflow", image_properties)
        pad_box = get_pad_box(images[0], output_size)
        images = pad_case_to_size(case=images, size=output_size, label=None)
        processed_affine = _translated_affine(processed_affine, [-pad_box[0], -pad_box[2], -pad_box[4]])
        image_properties["pad_box"] = [int(value) for value in pad_box]

    image_properties["new_size"] = list(images[0].shape)
    image_properties["foreground_locations"] = []
    image_properties["original_spacing"] = original_spacing.tolist()
    image_properties["original_size"] = original_size
    image_properties["original_orientation"] = image_properties["nifti_metadata"]["original_orientation"]
    image_properties["new_spacing"] = new_spacing
    image_properties["new_direction"] = image_properties["nifti_metadata"]["final_direction"]
    image_properties["processed_affine"] = processed_affine
    if target_spacing is not None and output_size is not None:
        image_properties["geometry_qc"] = _geometry_qc_base(image_properties, output_size)
    return images, image_properties


def preprocess_case_with_label(
    images: List[Union[np.ndarray, nib.Nifti1Image]],
    label: List[Union[np.ndarray, nib.Nifti1Image]],
    normalization_operation: list,
    background_pixel_value: int = 0,
    crop_to_nonzero: bool = True,
    keep_aspect_ratio_when_using_target_size: bool = False,
    image_properties: Optional[dict] = {},
    intensities: Optional[List] = None,
    target_orientation: Optional[str] = "RAS",
    target_size: Optional[List] = None,
    output_size: Optional[List] = None,
    target_spacing: Optional[List] = None,
    downsample_factor: Optional[float] = None,
    overflow_policy: str = "reject",
    allow_legacy_anisotropic_target_size: bool = False,
    strict: bool = True,
    supposed_to_be_3D: bool = True,
    remove_nans=True,
    min_slices=15,
):
    image_properties["pad_box"] = []
    image_properties["crop_box"] = []

    images, label, image_properties["nifti_metadata"] = apply_nifti_preprocessing_and_return_numpy(
        images=images,
        original_size=np.array(images[0].shape),
        target_orientation=target_orientation,
        label=label,
        include_header=True,
        strict=strict,
    )

    images = [safe_squeeze(image) for image in images]
    verify_3D_image_is_valid(
        images,
        supposed_to_be_3D=supposed_to_be_3D,
        remove_nans=remove_nans,
        min_slices=min_slices,
    )
    original_size = images[0].shape
    original_spacing = np.asarray(image_properties["nifti_metadata"]["original_spacing"], dtype=float)
    working_affine = np.asarray(image_properties["nifti_metadata"]["affine"], dtype=float)
    if working_affine.shape != (4, 4):
        working_affine = np.diag([*original_spacing, 1.0])

    # Cropping is performed to save computational resources. We are only removing background.
    if crop_to_nonzero:
        nonzero_box = get_bbox_for_foreground(images[0], background_label=background_pixel_value)
        for i in range(len(images)):
            images[i] = crop_to_box(images[i], nonzero_box)
        label = crop_to_box(label, nonzero_box)
        image_properties["crop_box"] = [int(i) for i in nonzero_box]
        working_affine = _translated_affine(working_affine, nonzero_box[::2])

    _reject_physical_overflow_before_resampling(
        images,
        image_properties,
        working_affine,
        original_size,
        original_spacing,
        target_spacing,
        output_size,
        overflow_policy,
    )

    image_properties["size_before_resample"] = images[0].shape
    if target_size is not None and not allow_legacy_anisotropic_target_size:
        spacing_ratio = float(original_spacing.max() / original_spacing.min())
        if spacing_ratio > 1.1:
            raise ValueError(
                "Legacy target_size resizing is not physically valid for anisotropic medical images "
                f"(spacing ratio={spacing_ratio:.3f}). Use target_spacing plus output_size."
            )

    if target_spacing is not None:
        if target_size is not None or downsample_factor is not None:
            raise ValueError("target_size, target_spacing, and downsample_factor are mutually exclusive.")
        images, label, processed_affine = resample_case_to_spacing(
            case=images,
            label=label,
            source_affine=working_affine,
            target_spacing=target_spacing,
            norm_op=normalization_operation,
            intensities=intensities,
        )
        final_target_size = None
        new_spacing = [float(value) for value in target_spacing]
    else:
        resample_target_size, final_target_size, new_spacing = determine_target_size(
            images=images,
            original_spacing=original_spacing,
            target_size=target_size,
            target_spacing=target_spacing,
            downsample_factor=downsample_factor,
            keep_aspect_ratio=keep_aspect_ratio_when_using_target_size,
        )
        size_before_resample = np.asarray(images[0].shape, dtype=float)
        images, label = resample_and_normalize_case(
            case=images,
            label=label,
            target_size=resample_target_size,
            norm_op=normalization_operation,
            intensities=intensities,
            resample_method="yucca",
        )
        processed_affine = working_affine @ np.diag([*(size_before_resample / np.asarray(images[0].shape)), 1.0])

    if final_target_size is not None:
        image_properties["shape_before_pad"] = images[0].shape
        pad_box = get_pad_box(images[0], final_target_size)
        images, label = pad_case_to_size(case=images, size=final_target_size, label=label)
        processed_affine = _translated_affine(processed_affine, [-pad_box[0], -pad_box[2], -pad_box[4]])
        image_properties["pad_box"] = [int(value) for value in pad_box]

    image_properties["resampled_size_before_fit"] = list(images[0].shape)
    if output_size is not None:
        if target_spacing is None:
            raise ValueError("output_size requires target_spacing so the output envelope has a physical interpretation.")
        output_size = [int(value) for value in output_size]
        if overflow_policy != "reject":
            raise ValueError(f"Unsupported overflow_policy={overflow_policy!r}; only 'reject' is supported.")
        if any(current > maximum for current, maximum in zip(images[0].shape, output_size)):
            image_properties["new_size"] = list(images[0].shape)
            image_properties["processed_affine"] = processed_affine
            image_properties["original_spacing"] = original_spacing.tolist()
            image_properties["original_size"] = original_size
            image_properties["new_spacing"] = new_spacing
            image_properties["geometry_qc"] = _geometry_qc_base(image_properties, output_size)
            raise GeometryQCExclusion("fov_overflow", image_properties)
        pad_box = get_pad_box(images[0], output_size)
        images, label = pad_case_to_size(case=images, size=output_size, label=label)
        processed_affine = _translated_affine(processed_affine, [-pad_box[0], -pad_box[2], -pad_box[4]])
        image_properties["pad_box"] = [int(value) for value in pad_box]

    image_properties["foreground_locations"] = get_foreground_locations(label=label, per_class=True)
    image_properties["new_size"] = list(images[0].shape)
    image_properties["original_spacing"] = original_spacing.tolist()
    image_properties["original_size"] = original_size
    image_properties["original_orientation"] = image_properties["nifti_metadata"]["original_orientation"]
    image_properties["new_spacing"] = new_spacing
    image_properties["new_direction"] = image_properties["nifti_metadata"]["final_direction"]
    image_properties["processed_affine"] = processed_affine
    if target_spacing is not None and output_size is not None:
        image_properties["geometry_qc"] = _geometry_qc_base(image_properties, output_size)
    return images, label, image_properties


def verify_3D_image_is_valid(
    images: list,
    supposed_to_be_3D: bool = True,
    remove_nans: bool = True,
    min_slices: int = 15,
):
    for image in images:
        if supposed_to_be_3D and len(image.shape) != 3:
            raise ValueError(f"image is not 3D. Shape: {image.shape}.  ")
        if np.min(image.shape) < min_slices:
            raise ValueError(f"image is too small. Shape: {image.shape}. ")
        if np.count_nonzero(image) < 1:
            raise ValueError("image is all zeros. ")
        if remove_nans:
            if np.any(np.isnan(image)):
                raise ValueError("image contains NaN values. ")


def extract_3dpet_from_4dpet(image, strict=True):
    if strict:
        assert np.min(image.shape) == image.shape[-1], (
            f"Min shape not last dimension for PET. Set strict to False to allow this. Found shape {image.shape}"
        )
    image_arr = np.mean(image.get_fdata(), axis=-1)
    header = image.header.copy()
    header.set_data_shape(image_arr.shape)
    image = nib.Nifti1Image(image_arr, image.affine, header=header)
    return image


def extract_3ddwi_from_4ddwi(image, bvals, bvecs, bval_tolerance=50, strict=True, use_trace_computation=False):
    """
    Extract 3D DWI from 4D DWI data.
    The use_trace_computation parameter determines the processing method for the entire image.
    """
    if strict:
        assert np.min(image.shape) == image.shape[-1], (
            f"Min shape not last dimension for DWI. Set strict to False to allow this. Found shape {image.shape}"
        )

    old_header = image.header.copy()

    # Group similar b-values
    bval_groups = group_bvalues(bvals, tolerance=bval_tolerance)

    dwis = []
    group_bvals = []

    for group_bval, indices in bval_groups:
        if group_bval == 0:
            # B0 images are always processed the same way
            dwi = get_data_for_bval_group(group_bval, indices, bvals, bvecs, image.get_fdata(), use_trace=False)
        else:
            dwi = get_data_for_bval_group(
                group_bval,
                indices,
                bvals,
                bvecs,
                image.get_fdata(),
                use_trace=use_trace_computation,
                bval_groups=bval_groups,
            )

        if dwi is not None:  # Only add if we got data (not skipped)
            new_header = old_header.copy()
            new_header.set_data_shape(dwi.shape)
            dwi = nib.Nifti1Image(dwi, image.affine, header=new_header)
            dwis.append(dwi)
            group_bvals.append(str(int(round(group_bval))))
        else:
            logging.error(f"Skipped b-value group: {group_bval} (processing failed)")

    return dwis, group_bvals


def get_best_basis(bvecs):
    """Finds the indices of the three bvecs closest to X, Y, and Z directions.

    Diffusion encoding is antipodally symmetric: a gradient at -X produces the
    same signal as +X. We therefore rank by |cos| so a basis is found regardless
    of the bvec sign convention (raw cosine would reject a -X gradient at cos=-1).
    """
    standard_basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    best_match = []
    max_cosines = []

    for std_vec in standard_basis:
        best_idx = None
        max_cosine = -1
        for i in range(bvecs.shape[1]):
            norm_bvec = np.linalg.norm(bvecs[:, i])
            if norm_bvec == 0:
                continue
            cos_sim = abs(np.dot(bvecs[:, i], std_vec) / norm_bvec)
            if cos_sim > max_cosine:
                max_cosine = cos_sim
                best_idx = i
        best_match.append(best_idx)
        max_cosines.append(max_cosine)

    if None in best_match:
        logging.warning(f"Could not find basis vector for some directions. Found indices: {best_match}")

    assert len(best_match) == 3
    return best_match


def group_bvalues(bvals, tolerance=50, b0_threshold=5):
    """
    Group similar b-values together. First groups b0s (including near-zero values), then remaining values.

    Args:
        bvals: Array of b-values
        tolerance: Tolerance for grouping similar non-zero b-values
        b0_threshold: B-values <= this threshold are treated as b0
    """
    groups = []

    # Find b0 and near-b0 indices (b-values <= b0_threshold)
    b0_indices = np.where(bvals <= b0_threshold)[0]
    if len(b0_indices) > 0:
        # Keep all b0/near-b0 volumes so the reference can be averaged for SNR.
        groups.append((0, b0_indices))
        logging.debug(f"Found {len(b0_indices)} b0/near-b0 volumes (b<={b0_threshold}), averaging them as reference")

    # Process non-b0 values
    non_b0_mask = bvals > b0_threshold
    non_b0_bvals = bvals[non_b0_mask]

    if len(non_b0_bvals) > 0:
        unique_non_b0 = np.unique(non_b0_bvals)
        used_indices = set(b0_indices.tolist()) if len(b0_indices) > 0 else set()

        for bval in sorted(unique_non_b0):
            similar_indices = np.where((bvals >= bval - tolerance) & (bvals <= bval + tolerance) & non_b0_mask)[0]
            available_indices = [idx for idx in similar_indices if idx not in used_indices]

            if len(available_indices) >= 3:
                group_representative = np.mean(bvals[available_indices])
                groups.append((group_representative, np.array(available_indices)))
                used_indices.update(available_indices)

    if len(groups) == 0:
        logging.warning("No valid b-value groups found")
    elif len(groups) == 1 and groups[0][0] == 0:
        logging.debug("Only b0/near-b0 volumes found, no orthogonal diffusion directions")

    return groups


def get_data_for_bval_group(group_bval, indices, bvals, bvecs, data, use_trace=False, bval_groups=None):
    """Extract data for a group of similar b-values."""
    if group_bval == 0:
        # Average all b0/near-b0 volumes for a higher-SNR reference.
        return np.mean(data[..., np.asarray(indices)], axis=-1)

    if len(indices) < 3:
        logging.warning(f"Insufficient volumes for b-value group {group_bval}: {len(indices)} < 3")
        return None

    try:
        basis_indices = get_best_basis(bvecs[:, indices])
        if None in basis_indices:
            logging.warning(f"Failed to find complete basis for b-value group {group_bval}")
            return None

        selected_volumes = data[..., [indices[i] for i in basis_indices]]

        if not use_trace:
            # Original method: simple averaging
            logging.debug(f"Using averaging method for b-value {group_bval}")
            return np.mean(selected_volumes, axis=-1)
        else:
            # Trace computation method
            logging.debug(f"Using trace computation method for b-value {group_bval}")
            # Pass the individual b-values for each selected volume
            individual_bvals = [bvals[indices[i]] for i in basis_indices]
            return compute_trace_adc(selected_volumes, individual_bvals, bval_groups, data)

    except Exception as e:
        logging.warning(f"Error processing b-value group {group_bval}: {e}")
        return None


def extract_3dperfusion_from_4dperfusion(image, is_m0scan=False, strict=True):
    if strict:
        assert np.min(image.shape) == image.shape[-1], (
            f"Min shape not last dimension for DWI. Set strict to False to allow this. Found shape {image.shape}"
        )

    data = image.get_fdata()
    tp = data.shape[-1]  # number of time points

    if is_m0scan:
        # For M0 scans, take the average across all the time points
        logging.debug(f"Processing M0 scan with {tp} time points - taking average")
        processed_data = np.mean(data, axis=-1)
    else:
        if tp <= 3:
            # For tp <= 3, take the difference between first and last
            logging.debug(f"Processing ASL scan with {tp} time points - taking difference between first and last")
            processed_data = data[..., -1] - data[..., 0]
        else:
            # For tp > 3, take the average
            logging.debug(f"Processing ASL scan with {tp} time points - taking average")
            processed_data = np.mean(data, axis=-1)

    # Create new 3D image with updated header
    header = image.header.copy()
    header.set_data_shape(processed_data.shape)
    processed_image = nib.Nifti1Image(processed_data, image.affine, header=header)

    return processed_image


def compute_trace_adc(selected_volumes, individual_bvals, bval_groups, full_data):
    """
    Compute trace ADC using the formula: Trace = ADCx + ADCy + ADCz
    where ADC = -(1/b) * ln(S/S0)

    Args:
        selected_volumes: 4D array with shape (..., 3) containing the 3 basis directions
        individual_bvals: List of 3 individual b-values for each direction
        bval_groups: List of (bval, indices) tuples from grouping
        full_data: Full 4D data array
    """
    try:
        # Find b0/near-b0 data (b-value group with value 0, which includes near-zero values)
        b0_data = None
        if bval_groups is not None:
            for bval, indices in bval_groups:
                if bval == 0:
                    # Average the b0/near-b0 reference to match get_data_for_bval_group.
                    b0_data = np.mean(full_data[..., np.asarray(indices)], axis=-1)
                    break

        if b0_data is None:
            logging.warning("No b0/near-b0 found for trace computation, reverting to averaging")
            return np.mean(selected_volumes, axis=-1)

        # Compute ADC for each direction (x, y, z)
        adc_components = []

        for i in range(selected_volumes.shape[-1]):  # For each direction
            S = selected_volumes[..., i]  # Signal for this direction
            S0 = b0_data  # Reference b0/near-b0 signal
            bval = individual_bvals[i]  # Individual b-value for this direction

            # Valid = in-tissue voxels (positive signal). Mildly-noisy voxels with
            # S > S0 are kept and clamped to ADC=0 below rather than being dumped
            # into the background bucket, so real tissue is not lost. Only true
            # background (non-positive signal) stays at 0.
            valid_mask = (S0 > 0) & (S > 0)

            # Initialize ADC array (background defaults to 0)
            adc = np.zeros_like(S, dtype=np.float32)

            # Compute ADC only for valid voxels
            if np.any(valid_mask):
                ratio = S[valid_mask] / S0[valid_mask]
                # Ensure ratio is positive and <= 1 for log (clamps S>S0 noise to ADC=0)
                ratio = np.clip(ratio, 1e-10, 1.0)
                adc[valid_mask] = -(1.0 / bval) * np.log(ratio)  # Use individual bval

            invalid_mask = ~valid_mask
            if np.any(invalid_mask):
                logging.debug(f"Found {np.sum(invalid_mask)} background voxels for direction {i} with b-value {bval}")

            adc_components.append(adc)

        # Compute trace as sum of ADC components
        trace_adc = np.sum(adc_components, axis=0)

        logging.debug(
            f"Computed trace ADC using individual b-values {individual_bvals}, "
            f"mean trace: {np.mean(trace_adc[trace_adc > 0]):.6f}"
        )

        return trace_adc

    except Exception as e:
        logging.error(f"Error in trace computation: {e}")
        logging.info("Reverting to averaging method")
        return np.mean(selected_volumes, axis=-1)


def determine_target_size(
    images: list,
    original_spacing,
    target_size,
    target_spacing,
    downsample_factor,
    keep_aspect_ratio,
):
    image_shape = np.array(images[0].shape)
    requested_resampling_ops = sum(value is not None for value in [target_size, target_spacing, downsample_factor])
    if requested_resampling_ops > 1:
        raise ValueError("target_size, target_spacing, and downsample_factor are mutually exclusive.")

    # We do not want to change the aspect ratio so we resample using the minimum alpha required
    # to attain 1 correct dimension, and then the rest will be padded.
    # Additionally we make sure each dimension is divisible by 16 to avoid issues with standard pooling/stride settings
    if target_size is not None:
        resample_target_size, final_target_size, new_spacing = determine_resample_size_from_target_size(
            current_size=image_shape,
            current_spacing=original_spacing,
            target_size=target_size,
            keep_aspect_ratio=keep_aspect_ratio,
        )

    elif downsample_factor is not None:
        resample_target_size, final_target_size, new_spacing = determine_resample_size_from_downsample_factor(
            current_size=image_shape,
            current_spacing=original_spacing,
            downsample_factor=downsample_factor,
        )

    # Otherwise we need to calculate a new target shape, and we need to factor in that
    # the images will first be transposed and THEN resampled.
    # Find new shape based on the target spacing
    elif target_spacing is not None:
        target_spacing = np.array(target_spacing, dtype=float)
        resample_target_size, final_target_size, new_spacing = determine_resample_size_from_target_spacing(
            current_size=image_shape,
            current_spacing=original_spacing,
            target_spacing=target_spacing,
        )
    else:
        resample_target_size = image_shape
        final_target_size = None
        new_spacing = original_spacing.tolist()
    return resample_target_size, final_target_size, new_spacing


def determine_resample_size_from_target_size(current_size, current_spacing, target_size, keep_aspect_ratio: bool = False):
    current_size = np.array(current_size)
    current_spacing = np.array(current_spacing, dtype=float)
    target_size = np.array(target_size, dtype=float)
    if keep_aspect_ratio:
        resample_target_size = np.maximum(np.array(current_size * np.min(target_size / current_size)).astype(int), 1)
        final_target_size = target_size.astype(int).tolist()
        final_target_size = [math.ceil(i / 16) * 16 for i in final_target_size]
    else:
        resample_target_size = target_size.astype(int).tolist()
        resample_target_size = [math.ceil(i / 16) * 16 for i in resample_target_size]
        final_target_size = None
    new_spacing = (
        (current_size.astype(float) / np.array(resample_target_size).astype(float)) * current_spacing.astype(float)
    ).tolist()
    return resample_target_size, final_target_size, new_spacing


def determine_resample_size_from_downsample_factor(
    current_size: np.ndarray,
    current_spacing: np.ndarray,
    downsample_factor: float,
) -> tuple[np.ndarray, None, list[float]]:
    downsample_factor = float(downsample_factor)
    if downsample_factor <= 0:
        raise ValueError("downsample_factor must be positive.")
    resample_target_size = np.maximum(np.round(current_size.astype(float) / downsample_factor).astype(int), 1)
    new_spacing = (current_spacing.astype(float) * downsample_factor).tolist()
    return resample_target_size, None, new_spacing


def determine_resample_size_from_target_spacing(
    current_size: np.ndarray,
    current_spacing: np.ndarray,
    target_spacing: np.ndarray,
) -> tuple[np.ndarray, None, list[float]]:
    resample_target_size = np.round((current_spacing / target_spacing) * current_size).astype(int)
    return resample_target_size, None, target_spacing.tolist()


def get_foreground_locations(
    label: np.ndarray,
    per_class: bool = False,
    max_locs_total: int = 100_000,
) -> dict[str, list]:
    foreground_locations: dict[str, list] = {}

    if not per_class:
        locs = np.array(np.nonzero(label)).T[::10].tolist()
        if locs:
            if len(locs) > max_locs_total:
                step = round(len(locs) / max_locs_total)
                locs = locs[::step]
            foreground_locations["1"] = locs
    else:
        classes = np.unique(label)[1:]
        if len(classes) == 0:
            return foreground_locations
        max_per_class = max_locs_total // len(classes)
        for c in classes:
            locs = np.array(np.where(label == int(c))).T[::10]
            if len(locs) > 0:
                if len(locs) > max_per_class:
                    locs = locs[:: round(len(locs) / max_per_class)]
                foreground_locations[str(int(c))] = locs
    return foreground_locations


def safe_squeeze_leading(array):
    if array.shape[0] == 1:
        return array.squeeze(axis=0)
    return array


def safe_squeeze_trailing(array):
    if array.shape[-1] == 1:
        return array.squeeze(axis=-1)
    return array


def safe_squeeze(array):
    array = safe_squeeze_leading(array)
    array = safe_squeeze_trailing(array)
    return array
