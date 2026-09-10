import nibabel as nib
import numpy as np
import torch
from PIL import Image


def _as_float32(tensor: torch.Tensor) -> torch.Tensor:
    """Decode a stored tensor to float32 before any arithmetic touches it.

    Volumes may be stored in a reduced-precision dtype to halve disk usage (bfloat16 is the
    candidate: unlike float16 it keeps float32's dynamic range, and real intensities in this corpus
    overflow float16). That is a **storage** choice only — everything downstream must compute in
    float32. This matters concretely: `Torch_Normalize` z-scores in place
    (``data[c] = fn(data[c])``), so a bfloat16 volume would have its mean/std computed *and stored
    back* in bfloat16, compounding the quantisation error into the normalized image the losses see.
    Casting here keeps the round-trip error to the single write, and leaves float32 data untouched.
    """
    return tensor if tensor.dtype == torch.float32 else tensor.to(torch.float32)


def load_image_file(file: str) -> torch.Tensor:
    if file.endswith(".pt"):
        return _as_float32(torch.load(file, weights_only=True))
    elif file.endswith(".nii.gz") or file.endswith(".nii"):
        nii = nib.load(file)
        data = nii.get_fdata(dtype=np.float32)
        tensor = torch.from_numpy(data)
        return tensor.unsqueeze(0)  # (H,W,D) -> (1,H,W,D) to match .pt channel convention
    elif file.endswith(".png"):
        image = Image.open(file)
        if image.mode == "RGB":
            image = image.convert("L")
        image = np.array(image)
        image = torch.tensor(image)
        return image.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported file format: {file}. Expected .pt, .png, .nii, or .nii.gz and found {file}")
