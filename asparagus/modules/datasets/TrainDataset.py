import nibabel as nib
import numpy as np
import torch
import torchvision
from asparagus.paths import get_data_path, get_source_labels_path
from gardening_tools.functional.nibabel_utils import reorient_nib_image
from gardening_tools.functional.paths.read import load_pickle, read_file_to_nifti_or_np
from gardening_tools.functional.type_conversions import nifti_or_np_to_np
from torch.utils.data import Dataset
from typing import Optional


class SegDataset(Dataset):
    def __init__(
        self,
        files: list,
        transforms: Optional[torchvision.transforms.Compose] = None,
    ):
        super().__init__()

        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = torch.load(file)
        properties = load_pickle(file.replace(".pt", ".pkl"))
        data_dict = {
            "file_path": file,
            "image": data[:-1],
            "label": data[-1:],
            "foreground_locations": properties["foreground_locations"],
            # The spacing of the tensor on disk, for transforms that reason in millimetres.
            # Training needs no restoration provenance, so only the spacing is carried, and it is
            # popped again below so the collated batch keeps exactly the keys it always had.
            "source_spacing": properties.get("new_spacing"),
            "transforms_applied": {},
        }

        return self._transform(data_dict)

    def _transform(self, data_dict):
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
        data_dict.pop("foreground_locations")
        data_dict.pop("source_spacing", None)
        return data_dict


class ClsRegDataset(Dataset):
    def __init__(
        self,
        files: list,
        transforms: Optional[torchvision.transforms.Compose] = None,
    ):
        super().__init__()

        self.files = files
        self.composed_transforms = transforms
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = torch.load(file)
        data_dict = {
            "file_path": file,
            "image": data[0],
            "CLSREG_label": data[1],
            "transforms_applied": {},
        }

        return self._transform(data_dict)

    def _transform(self, data_dict):
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
        return data_dict


class SegTestDataset(Dataset):
    def __init__(
        self,
        files: list,
        transforms: Optional[torchvision.transforms.Compose] = None,
    ):
        super().__init__()

        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = torch.load(file)
        properties = load_pickle(file.replace(".pt", ".pkl"))
        src_label = self._get_src_label(file, properties)

        id = "_".join(file.split("/")[-3:]).replace(".pt", "")
        data_dict = {
            "file_path": file,
            "image": data[:-1],
            "label": data[-1:],
            "src_label": src_label,
            "properties": properties,
            "id": id,
        }

        return self._transform(data_dict)

    def _transform(self, data_dict):
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
        return data_dict

    def _get_src_label(self, file, properties):
        # source label is the label from the original dataset without any preprocessing
        src_label_path = file.replace(get_data_path(), get_source_labels_path()).replace(".pt", "_label.nii.gz")
        src_label_nii = read_file_to_nifti_or_np(src_label_path)
        src_label_nii = reorient_nib_image(
            src_label_nii,
            original_orientation=properties["original_orientation"],
            target_orientation=properties["new_direction"],
        )
        src_label_npy = nifti_or_np_to_np(src_label_nii)
        return torch.from_numpy(src_label_npy).float().unsqueeze(0).unsqueeze(0)


class ClsRegTestDataset(Dataset):
    def __init__(
        self,
        files: list,
        transforms: Optional[torchvision.transforms.Compose] = None,
    ):
        super().__init__()

        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = torch.load(file)
        data_dict = {
            "file_path": file,
            "image": data[0],
            "CLSREG_label": data[1],
        }

        return self._transform(data_dict)

    def _transform(self, data_dict):
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
        return data_dict


def _nifti_geometry(image: "nib.Nifti1Image", file) -> dict:
    """Physical geometry of a NIfTI, in the keys the preprocessed corpora already use.

    ``prepare_fomo26_asparagus.py`` writes ``original_spacing``/``new_spacing`` from
    ``header.get_zooms()[:3]`` and ``original_orientation``/``new_direction`` from
    ``nib.aff2axcodes(affine)``; this reads them the same way so a raw NIfTI and its preprocessed
    ``.pt`` describe the same volume identically. ``new_spacing`` equals ``original_spacing``
    because loading resamples nothing -- which is exactly the invariant
    ``Torch_ResampleToSpacing._source_spacing`` relies on.

    Values are recorded, not judged. A malformed spacing is refused by
    ``spacing.resolve_target_spacing`` at the moment a resample is actually requested, which keeps
    the refusal scoped to the tasks that opt into canonicalization instead of introducing a new
    failure mode for every task that does not.
    """
    spacing = [float(value) for value in image.header.get_zooms()[:3]]
    orientation = "".join(nib.aff2axcodes(image.affine))
    return {
        "original_spacing": spacing,
        "new_spacing": list(spacing),
        "original_orientation": orientation,
        "new_direction": orientation,
    }


def _image_from_sample(loaded, file):
    """Return the image of a stored sample, whatever container it was saved in.

    A preprocessed asparagus sample on disk is ``[image, label]``, not a bare image tensor.
    Prediction consumes only the image -- taking element 0 both makes the stored format loadable
    and guarantees the label can never reach an inference path that must not see it.
    """
    if isinstance(loaded, (list, tuple)):
        if not loaded:
            raise ValueError(f"Sample contains no image: {file}")
        return loaded[0]
    if isinstance(loaded, dict):
        for key in ("image", "data"):
            if key in loaded:
                return loaded[key]
        raise ValueError(f"Sample mapping has no 'image'/'data' key: {file} (keys={sorted(loaded)})")
    return loaded


class SingleSubjectPredictDataset(Dataset):
    def __init__(
        self,
        files: list,
        transforms: Optional[torchvision.transforms.Compose] = None,
    ):
        super().__init__()

        self.files = files
        self.transforms = transforms

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        properties = {}
        all_channels = []
        for file in self.files:
            if file.endswith(".pt"):
                data = torch.load(file)
            elif file.endswith(".npy"):
                data = torch.from_numpy(np.load(file))
            elif file.endswith(".nii") or file.endswith(".nii.gz"):
                data = nib.load(file)
                properties["nifti_metadata"] = {
                    "affine": data.affine,
                    "header": data.header,
                    "reoriented": False,
                }
                # Physical geometry, read the same way prepare_fomo26_asparagus.py reads it
                # (header zooms for spacing, affine axis codes for orientation) so a prediction
                # path and the preprocessing path describe a volume identically. Without these
                # keys a runtime spacing stage has no geometry to resample from and fails closed;
                # they are metadata only, and no transform consumes them unless one is spliced in.
                properties.update(_nifti_geometry(data, file))
                data = torch.from_numpy(data.get_fdata()[np.newaxis])
            else:
                raise ValueError(f"Unsupported file type: {file}")
            data = _image_from_sample(data, file)
            data = data.float()
            all_channels.append(data)

        data = torch.vstack(all_channels)
        properties["original_size"] = data.shape[1:]  # Exclude channel dimension
        data_dict = {
            "file_path": file,
            "image": data,
            "properties": properties,
        }

        return self._transform(data_dict)

    def _transform(self, data_dict):
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
        return data_dict
