import logging
import nibabel as nib
import numpy as np
import pandas as pd
from asparagus_preprocessing.utils.normalize import normalizer
from nibabel.processing import resample_from_to
from numpy.typing import NDArray
from skimage.transform import resize


def resample_and_normalize_case(
    case: list,
    target_size,
    norm_op: str,
    intensities: list = None,
    label: np.ndarray = None,
    allow_missing_modalities: bool = False,
    resample_method: str = "yucca",
):
    assert resample_method in ["nnunet", "yucca"], "resample_method must be either 'nnunet' or 'yucca'"

    # Normalize and Transpose images to target view.
    # Transpose labels to target view.
    assert len(case) == len(norm_op), (
        "number of images and "
        "normalization  operations does not match. \n"
        f"len(images) == {len(case)} \n"
        f"len(norm_op) == {len(norm_op)} \n"
    )

    for i in range(len(case)):
        image = case[i]
        assert image is not None
        if image.size == 0:
            assert allow_missing_modalities is True, "missing modality and allow_missing_modalities is not enabled"
        else:
            # Normalize
            if intensities is not None:
                case[i] = normalizer(image, scheme=norm_op[i], intensities=intensities[i])
            else:
                case[i] = normalizer(image, scheme=norm_op[i])

            # Resample to target shape and spacing
            try:
                if resample_method == "nnunet":
                    case[i] = resize(case[i], output_shape=target_size, order=3, mode="edge")
                elif resample_method == "yucca":
                    case[i] = resize(case[i], output_shape=target_size, order=3)
            except OverflowError:
                logging.error("Unexpected values in either shape or image for resize")

    if label is not None:
        try:
            if resample_method == "nnunet":
                label = resize_segmentation(label, new_shape=target_size, order=3)
            elif resample_method == "yucca":
                label = resize(label, output_shape=target_size, order=0, anti_aliasing=False)

        except OverflowError:
            logging.error("Unexpected values in either shape or label for resize")
        return case, label

    return case


def resample_case_to_spacing(
    case: list[np.ndarray],
    source_affine: np.ndarray,
    target_spacing: list[float],
    norm_op: list[str],
    intensities: list = None,
    label: np.ndarray = None,
) -> tuple[list[np.ndarray], np.ndarray | None, np.ndarray]:
    """Resample volumes to physical voxel spacing while preserving source coverage."""
    source_affine = np.asarray(source_affine, dtype=float)
    current_spacing = nib.affines.voxel_sizes(source_affine)
    target_spacing_array = np.asarray(target_spacing, dtype=float)
    if target_spacing_array.shape != (3,) or np.any(target_spacing_array <= 0):
        raise ValueError(f"target_spacing must contain three positive values, got {target_spacing}.")

    target_shape = np.maximum(
        np.round(np.asarray(case[0].shape, dtype=float) * current_spacing / target_spacing_array).astype(int),
        1,
    )
    target_affine = source_affine.copy()
    target_affine[:3, :3] = source_affine[:3, :3] @ np.diag(target_spacing_array / current_spacing)
    target = (tuple(int(size) for size in target_shape), target_affine)

    output = []
    for index, image in enumerate(case):
        normalized = normalizer(image, scheme=norm_op[index], intensities=None if intensities is None else intensities[index])
        resampled = resample_from_to(
            nib.Nifti1Image(normalized, source_affine),
            target,
            order=3,
            mode="constant",
            cval=float(normalized.min()),
        )
        output.append(resampled.get_fdata(dtype=np.float32))

    output_label = None
    if label is not None:
        output_label = resample_from_to(
            nib.Nifti1Image(label, source_affine),
            target,
            order=0,
            mode="constant",
            cval=0.0,
        ).get_fdata(dtype=np.float32)
        output_label = output_label.astype(label.dtype, copy=False)
    return output, output_label, target_affine


def resize_segmentation(
    segmentation: NDArray,
    new_shape: tuple[int, ...],
    order: int = 3,
) -> NDArray:
    """
    Resize a segmentation map while preserving label integrity.

    Converts segmentation to one-hot encoding, resizes each channel, and converts
    back to segmentation map. This prevents interpolation artifacts (e.g., [0, 0, 2]
    becoming [0, 1, 2]).

    Args:
        segmentation: Input segmentation map (multi-dimensional array).
        new_shape: Target shape for resizing.
        order: Interpolation order. 0 for nearest neighbor, 1-5 for higher orders.
            See skimage documentation for details. Defaults to 3 (cubic).

    Returns:
        Resized segmentation map with same dtype as input.

    Raises:
        AssertionError: If new_shape dimensionality doesn't match segmentation.
    """
    tpe = segmentation.dtype
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(
            segmentation.astype(float),
            new_shape,
            order,
            mode="edge",
            clip=True,
            anti_aliasing=False,
        ).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        unique_labels = np.sort(pd.unique(segmentation.ravel()))
        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(
                mask.astype(float),
                new_shape,
                order,
                mode="edge",
                clip=True,
                anti_aliasing=False,
            )
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped
