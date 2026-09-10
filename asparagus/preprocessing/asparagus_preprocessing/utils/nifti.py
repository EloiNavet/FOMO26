import nibabel as nib
import nibabel.orientations as nio
import numpy as np
from typing import Union


def apply_nifti_preprocessing_and_return_numpy(
    images,
    original_size,
    target_orientation,
    label=None,
    include_header=False,
    strict=True,
) -> tuple[list[np.ndarray], np.ndarray, dict]:
    # If qform and sform are both missing the header is corrupt and we do not trust the
    # direction from the affine
    # Make sure you know what you're doing
    metadata = {
        "original_spacing": np.array([1.0] * len(original_size)).tolist(),
        "original_orientation": None,
        "final_direction": None,
        "header": None,
        "affine": None,
        "reoriented": False,
    }

    if isinstance(images[0], nib.Nifti1Image):
        if verify_nifti_header_is_valid(images[0]) is True:
            if strict:
                assert verify_orientation_is_LR_PA_IS(images[0]), (
                    "unexpected NIFTI axes. Consider RAS-conversion during task conversion"
                )
            metadata["reoriented"] = True
            metadata["original_orientation"] = get_nib_orientation(images[0])
            metadata["final_direction"] = target_orientation
            images = [
                reorient_nib_image(image, metadata["original_orientation"], metadata["final_direction"]) for image in images
            ]
            if label is not None and isinstance(label, nib.Nifti1Image):
                label = reorient_nib_image(label, metadata["original_orientation"], metadata["final_direction"])
        if include_header:
            metadata["header"] = images[0].header.binaryblock
        metadata["original_spacing"] = get_nib_spacing(images[0]).tolist()
        metadata["affine"] = images[0].affine

    images = [nifti_or_np_to_np(image) for image in images]
    if label is not None:
        label = nifti_or_np_to_np(label)
    return images, label, metadata


def nifti_or_np_to_np(array: Union[np.ndarray, nib.Nifti1Image]) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    if isinstance(array, (nib.Nifti1Image, nib.minc1.Minc1Image)):
        return array.get_fdata().astype(np.float32)
    else:
        raise TypeError(f"File data type invalid. Found: {type(array)} and expected nib.Nifti1Image or np.ndarray")


def reorient_nib_image(nib_image, original_orientation: str, target_orientation: str) -> np.ndarray:
    # The reason we don't use the affine information to get original_orientation is that it can be
    # incorrect. Therefore it can be manually specified. In the cases where header can be trusted,
    # Just use get_nib_orientation to get the original_orientation.
    if original_orientation == target_orientation:
        return nib_image
    start = nio.axcodes2ornt(original_orientation)
    end = nio.axcodes2ornt(target_orientation)
    orientation = nio.ornt_transform(start, end)
    return nib_image.as_reoriented(orientation)


def get_nib_spacing(nib_image: nib.Nifti1Image) -> np.ndarray:
    return np.array(nib_image.header.get_zooms())


def get_nib_orientation(nib_image: nib.Nifti1Image) -> str:
    affine = nib_image.affine
    return "".join(nio.aff2axcodes(affine))


def verify_nifti_header_is_valid(image: nib.Nifti1Image):
    if image.get_qform(coded=True)[1] or image.get_sform(coded=True)[1]:
        return True
    else:
        return False


def verify_orientation_is_LR_PA_IS(image: nib.Nifti1Image):
    """
    Checks whether images are in the RAS/LPI domain, which corresponds to:
    X = Left/Right (left/right)
    Y = Posterior/Anterior (backwards/forwards)
    Z = Inferior/Superior (down/up)

    If this is not the case, then images should be converted to RAS (or at least some combination of LR-PA-IS)
    during task conversion as features such as resampling to a target spacing may become unreliably or incorrect
    """
    expected_orientation_code = np.array([0.0, 1.0, 2.0])  # This means LR-PA-IS
    orientation = get_nib_orientation(image)
    if np.all(nio.axcodes2ornt(orientation)[:, 0] == expected_orientation_code):
        return True
    else:
        return False
