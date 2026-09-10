import csv
import dataclasses
import json
import nibabel as nib
import numpy as np
import os
import pickle
import shutil
import torch
from asparagus_preprocessing.paths import get_data_path, get_raw_labels_path
from nibabel import Nifti1Header, Nifti1Image


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o: object) -> object:
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def enhanced_save_json(
    obj,
    file: str,
    indent: int = 4,
    sort_keys: bool = True,
    cls: json.JSONEncoder = EnhancedJSONEncoder,
    permissions: int = None,
) -> None:
    with open(file, "w") as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent, cls=cls)
    if permissions is not None:
        os.chmod(file, permissions)


def save_pickle(obj, file: str, mode: str = "wb") -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)


def save_nifti(data, path, affine, header) -> None:
    tmp = Nifti1Image(data, affine=affine, header=header)
    tmp.set_qform(affine, code=1)
    tmp.set_sform(affine, code=1)
    nib.save(tmp, path)


def save_torch(data: list, path: str, dtype: str) -> None:
    tmp = np.array(data, dtype=getattr(np, dtype))
    tmp = torch.from_numpy(tmp)
    torch.save(tmp, path)


def save_torch_clsreg(data: list, label: int, path: str, dtype: str) -> None:
    tmp_data = np.array(data, dtype=getattr(np, dtype))
    tmp_data = torch.from_numpy(tmp_data)
    tmp_label = torch.tensor([label], dtype=getattr(torch, dtype))
    torch.save([tmp_data, tmp_label], path)


def save_data_and_metadata(data, metadata, data_path, saving_config) -> None:
    os.makedirs(os.path.split(data_path)[0], exist_ok=True)
    if saving_config.save_as_tensor:
        save_torch(
            data=data,
            path=data_path + ".pt",
            dtype=saving_config.tensor_dtype,
        )
    else:
        processed_affine = metadata.get("processed_affine", metadata["nifti_metadata"]["affine"])
        header = Nifti1Header(metadata["nifti_metadata"]["header"])
        header.set_data_shape(data[0].shape)
        header.set_zooms(tuple(float(value) for value in metadata["new_spacing"]))
        save_nifti(
            data[0],
            data_path + ".nii.gz",  # This will not work with n_modalities > 1
            affine=processed_affine,
            header=header,
        )
    if saving_config.save_file_metadata:
        save_pickle(metadata, data_path + ".pkl")
    if "geometry_qc" in metadata:
        save_geometry_qc(metadata, data_path)


def save_geometry_qc(metadata: dict, data_path: str) -> None:
    if "geometry_qc" not in metadata:
        return
    os.makedirs(os.path.split(data_path)[0], exist_ok=True)
    enhanced_save_json(metadata["geometry_qc"], data_path + ".geometry_qc.json")


def save_raw_label(img_path, label_path) -> None:
    source_label_dest = img_path.replace(get_data_path(), get_raw_labels_path())
    source_label_dest += "_label.nii.gz"
    os.makedirs(os.path.dirname(source_label_dest), exist_ok=True)
    shutil.copy2(label_path, source_label_dest)


def save_modified_label(img_path: str, modified_label: Nifti1Image) -> None:
    source_label_dest = img_path.replace(get_data_path(), get_raw_labels_path())
    source_label_dest += "_label.nii.gz"
    os.makedirs(os.path.dirname(source_label_dest), exist_ok=True)
    nib.save(modified_label, source_label_dest)


def save_clsreg_data_and_metadata(data, label, metadata, data_path, saving_config, table_path) -> None:
    os.makedirs(os.path.split(data_path)[0], exist_ok=True)
    if saving_config.save_as_tensor:
        save_torch_clsreg(
            data=data,
            label=label,
            path=data_path + ".pt",
            dtype=saving_config.tensor_dtype,
        )
    else:
        raise NotImplementedError("Only saving as tensor is implemented for classification/regression tasks.")

    if saving_config.save_file_metadata:
        save_pickle(metadata, data_path + ".pkl")
    save_clsreg_label_to_table(label, data_path, table_path)


def save_clsreg_label_to_table(label, data_path, table_path):
    os.makedirs(os.path.split(table_path)[0], exist_ok=True)
    with open(table_path, "a+") as f:
        writer = csv.writer(f)
        writer.writerow([data_path, label])
