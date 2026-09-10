#!/usr/bin/env python3
"""Convert local FOMO26 finetuning tasks into Asparagus tensor datasets."""

from __future__ import annotations

import argparse
import json
import nibabel as nib
import numpy as np
import os
import pickle
import random
import shutil
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class TaskSpec:
    source_name: str
    output_name: str
    kind: str
    modalities: tuple[str, ...]
    n_classes: int
    label_name: str | None = None
    segmentation_name: str | None = None
    allow_empty_masks: bool = False


TASK_SPECS: dict[str, TaskSpec] = {
    "task1_presence": TaskSpec(
        source_name="Task_1",
        output_name="CLS901_FOMO26_Task1_presence",
        kind="classification",
        modalities=("adc", "dwi_b1000", "flair", "susceptibility"),
        n_classes=2,
        label_name="label.txt",
    ),
    "task1_lesion": TaskSpec(
        source_name="Task_1",
        output_name="SEG901_FOMO26_Task1_lesion",
        kind="segmentation",
        modalities=("adc", "dwi_b1000", "flair", "susceptibility"),
        n_classes=2,
        label_name="label.txt",
        segmentation_name="seg.nii.gz",
        allow_empty_masks=True,
    ),
    "task2_lesion": TaskSpec(
        source_name="Task_2",
        output_name="SEG902_FOMO26_Task2_lesion",
        kind="segmentation",
        modalities=("dwi_b1000", "flair", "susceptibility"),
        n_classes=2,
        segmentation_name="seg.nii.gz",
    ),
    "task3_age": TaskSpec(
        source_name="Task_3",
        output_name="REGR903_FOMO26_Task3_age",
        kind="regression",
        modalities=("t1w",),
        n_classes=1,
        label_name="labels.txt",
    ),
    "task4_multiclass": TaskSpec(
        source_name="Task_4",
        output_name="SEG904_FOMO26_Task4_multiclass",
        kind="segmentation",
        modalities=("t2w",),
        n_classes=3,
        segmentation_name="seg.nii.gz",
    ),
    "task5_ppmr": TaskSpec(
        source_name="Task_5",
        output_name="CLS905_FOMO26_Task5_ppmr",
        kind="classification",
        modalities=("t1",),
        n_classes=2,
        label_name="labels.txt",
    ),
}


def _repo_default(path: str) -> Path:
    return Path.cwd() / path


def _default_env_path(name: str, fallback: str) -> Path:
    return Path(os.environ.get(name, _repo_default(fallback))).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("finetuning"), help="Folder containing Task_1 ... Task_5.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_default_env_path("ASPARAGUS_DATA", "data/fomo26_finetune_asparagus"),
        help="Asparagus output data root. Defaults to $ASPARAGUS_DATA or data/fomo26_finetune_asparagus.",
    )
    parser.add_argument(
        "--raw-labels-root",
        type=Path,
        default=_default_env_path("ASPARAGUS_RAW_LABELS", "data/fomo26_finetune_raw_labels"),
        help="Asparagus raw-label root. Defaults to $ASPARAGUS_RAW_LABELS or data/fomo26_finetune_raw_labels.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        help="Task keys to convert, or 'all'. Keys: " + ", ".join(TASK_SPECS),
    )
    parser.add_argument("--split", nargs=3, type=int, default=[70, 15, 15], metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument(
        "--extra-split",
        nargs=3,
        type=int,
        default=[85, 15, 0],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Additional split to write for final training. Use negative values to disable.",
    )
    parser.add_argument("--seed", type=int, default=2606, help="Deterministic split seed.")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing converted task folders first.")
    return parser.parse_args()


def selected_specs(task_args: list[str]) -> list[TaskSpec]:
    keys = list(TASK_SPECS) if task_args == ["all"] else task_args
    unknown = [key for key in keys if key not in TASK_SPECS]
    if unknown:
        raise SystemExit(f"Unknown task key(s): {unknown}. Available: {sorted(TASK_SPECS)}")
    return [TASK_SPECS[key] for key in keys]


def subject_dirs(task_dir: Path) -> list[Path]:
    preprocessed = task_dir / "preprocessed"
    if not preprocessed.is_dir():
        raise FileNotFoundError(f"Missing {preprocessed}. If this is Task_5, run finetuning/Task_5_extract.py first.")
    subjects = sorted(path for path in preprocessed.iterdir() if path.is_dir())
    if not subjects:
        raise FileNotFoundError(f"No subject folders found in {preprocessed}.")
    return subjects


def modality_path(session_dir: Path, modality: str) -> Path:
    if modality == "susceptibility":
        for candidate in ("swi.nii.gz", "t2s.nii.gz"):
            path = session_dir / candidate
            if path.is_file():
                return path
        raise FileNotFoundError(f"Missing susceptibility channel in {session_dir}; expected swi.nii.gz or t2s.nii.gz.")
    path = session_dir / f"{modality}.nii.gz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing modality {path}.")
    return path


def load_modalities(session_dir: Path, modalities: tuple[str, ...]) -> tuple[list[np.ndarray], nib.Nifti1Image]:
    arrays: list[np.ndarray] = []
    reference_img: nib.Nifti1Image | None = None
    reference_shape: tuple[int, ...] | None = None
    for modality in modalities:
        img = nib.load(str(modality_path(session_dir, modality)))
        arr = np.asanyarray(img.dataobj, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D image for {modality_path(session_dir, modality)}, got shape {arr.shape}.")
        if reference_shape is None:
            reference_shape = arr.shape
            reference_img = img
        elif arr.shape != reference_shape:
            raise ValueError(f"Shape mismatch in {session_dir}: expected {reference_shape}, got {arr.shape}.")
        arrays.append(arr)
    assert reference_img is not None
    return arrays, reference_img


def read_scalar_label(label_path: Path, cast: Callable[[float], float | int]) -> float | int:
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label file {label_path}.")
    value = float(label_path.read_text(encoding="utf-8").strip())
    return cast(value)


def label_path_for(task_dir: Path, subject: str, label_name: str) -> Path:
    return task_dir / "labels" / subject / "ses-01" / label_name


def segmentation_path_for(task_dir: Path, subject: str, segmentation_name: str) -> Path:
    return task_dir / "labels" / subject / "ses-01" / segmentation_name


def empty_label_like(reference: nib.Nifti1Image) -> nib.Nifti1Image:
    data = np.zeros(reference.shape, dtype=np.uint8)
    return nib.Nifti1Image(data, affine=reference.affine, header=reference.header)


def load_or_create_segmentation(task_dir: Path, subject: str, spec: TaskSpec, reference: nib.Nifti1Image) -> nib.Nifti1Image:
    assert spec.segmentation_name is not None
    label_path = segmentation_path_for(task_dir, subject, spec.segmentation_name)
    if label_path.is_file():
        return nib.load(str(label_path))
    if not spec.allow_empty_masks:
        raise FileNotFoundError(f"Missing segmentation mask {label_path}.")
    if spec.label_name is not None:
        label_value = int(read_scalar_label(label_path_for(task_dir, subject, spec.label_name), int))
        if label_value != 0:
            raise FileNotFoundError(f"Positive subject {subject} is missing required segmentation mask {label_path}.")
    return empty_label_like(reference)


def foreground_locations(label: np.ndarray, max_locs_total: int = 100_000) -> dict[str, list[list[int]]]:
    locations: dict[str, list[list[int]]] = {}
    classes = [int(value) for value in np.unique(label) if int(value) != 0]
    if not classes:
        return locations
    max_per_class = max(1, max_locs_total // len(classes))
    for cls in classes:
        locs = np.array(np.where(label == cls)).T[::10]
        if len(locs) > max_per_class:
            locs = locs[:: round(len(locs) / max_per_class)]
        locations[str(cls)] = locs.astype(int).tolist()
    return locations


def image_properties(reference: nib.Nifti1Image, label: np.ndarray | None = None) -> dict:
    shape = [int(value) for value in reference.shape]
    spacing = [float(value) for value in reference.header.get_zooms()[:3]]
    orientation = "".join(nib.aff2axcodes(reference.affine))
    properties = {
        "foreground_locations": foreground_locations(label) if label is not None else [],
        "original_size": shape,
        "size_before_resample": shape,
        "new_size": shape,
        "original_spacing": spacing,
        "new_spacing": spacing,
        "original_orientation": orientation,
        "new_direction": orientation,
        "crop_box": [],
        "pad_box": [],
        "nifti_metadata": {
            "affine": reference.affine,
            "header": reference.header,
            "original_spacing": spacing,
            "original_orientation": orientation,
            "final_direction": orientation,
        },
    }
    return properties


def save_pickle(obj: object, path: Path) -> None:
    with path.open("wb") as file:
        pickle.dump(obj, file)


def save_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(obj, file, indent=4, sort_keys=True)


def save_dataset_json(target_dir: Path, spec: TaskSpec, paths: list[str]) -> None:
    config = {
        "name": spec.output_name,
        "metadata": {
            "files_target_directory_total": len(paths),
            "files_target_directory_standard": len(paths),
            "n_classes": spec.n_classes,
            "n_modalities": len(spec.modalities),
            "task_kind": spec.kind,
        },
        "dataset_config": {
            "task_name": spec.output_name,
            "n_classes": spec.n_classes,
            "n_modalities": len(spec.modalities),
            "in_extensions": [".pt"],
            "split": None,
            "modalities": list(spec.modalities),
        },
        "preprocessing_config": {
            "normalization_operation": ["no_norm"] * len(spec.modalities),
            "target_spacing": None,
            "target_orientation": "native",
            "crop_to_nonzero": False,
            "min_slices": 0,
        },
        "saving_config": {
            "save_as_tensor": True,
            "tensor_dtype": "float32",
            "save_file_metadata": True,
            "save_dset_metadata": False,
            "bidsify": False,
        },
    }
    save_json(config, target_dir / "dataset.json")


def split_paths(paths: list[str], vals: list[int], seed: int) -> tuple[list[dict[str, list[str]]], list[str]]:
    if len(vals) != 3 or sum(vals) != 100 or any(value < 0 for value in vals):
        raise ValueError(f"Split values must be three non-negative integers summing to 100, got {vals}.")
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    n_total = len(shuffled)
    n_train = round(n_total * vals[0] / 100)
    n_val = round(n_total * vals[1] / 100)
    if vals[2] == 0:
        n_test = 0
        n_val = n_total - n_train
    else:
        n_test = n_total - n_train - n_val
    if n_train == 0 and n_total > 0:
        n_train = 1
    if vals[1] > 0 and n_val == 0 and n_total - n_train - n_test > 0:
        n_val = 1
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val : n_train + n_val + n_test]
    return [{"train": train, "val": val}], test


def write_split_files(target_dir: Path, paths: list[str], vals: list[int], seed: int) -> None:
    split, test = split_paths(paths, vals, seed)
    suffix = f"{vals[0]:02d}_{vals[1]:02d}_{vals[2]:02d}"
    save_json(split, target_dir / f"split_{suffix}.json")
    save_json(test, target_dir / f"TEST_{suffix}.json")


def output_base(target_dir: Path, subject: str) -> Path:
    return target_dir / subject / "ses-01" / "scan"


def save_classification_or_regression(task_dir: Path, target_dir: Path, subject_dir: Path, spec: TaskSpec) -> str:
    session_dir = subject_dir / "ses-01"
    arrays, reference = load_modalities(session_dir, spec.modalities)
    assert spec.label_name is not None
    cast = int if spec.kind == "classification" else float
    label = read_scalar_label(label_path_for(task_dir, subject_dir.name, spec.label_name), cast)
    image = torch.from_numpy(np.stack(arrays).astype(np.float32))
    label_tensor = torch.tensor([label], dtype=torch.float32)
    base = output_base(target_dir, subject_dir.name)
    base.parent.mkdir(parents=True, exist_ok=True)
    torch.save([image, label_tensor], base.with_suffix(".pt"))
    save_pickle(image_properties(reference), base.with_suffix(".pkl"))
    return str(base.with_suffix(".pt").resolve())


def save_segmentation(task_dir: Path, target_dir: Path, raw_labels_dir: Path, subject_dir: Path, spec: TaskSpec) -> str:
    session_dir = subject_dir / "ses-01"
    arrays, reference = load_modalities(session_dir, spec.modalities)
    label_img = load_or_create_segmentation(task_dir, subject_dir.name, spec, reference)
    label = np.asanyarray(label_img.dataobj)
    if label.shape != arrays[0].shape:
        raise ValueError(f"Label shape mismatch for {subject_dir.name}: image {arrays[0].shape}, label {label.shape}.")
    if int(np.max(label)) >= spec.n_classes:
        raise ValueError(
            f"{subject_dir.name} has label value {int(np.max(label))}, but {spec.output_name} has {spec.n_classes} classes."
        )

    base = output_base(target_dir, subject_dir.name)
    base.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.from_numpy(np.stack([*arrays, label.astype(np.float32)]).astype(np.float32))
    torch.save(tensor, base.with_suffix(".pt"))
    save_pickle(image_properties(reference, label=label), base.with_suffix(".pkl"))

    raw_label_path = raw_labels_dir / spec.output_name / subject_dir.name / "ses-01" / "scan_label.nii.gz"
    raw_label_path.parent.mkdir(parents=True, exist_ok=True)
    if segmentation_path_for(task_dir, subject_dir.name, spec.segmentation_name or "").is_file():
        shutil.copy2(segmentation_path_for(task_dir, subject_dir.name, spec.segmentation_name or ""), raw_label_path)
    else:
        nib.save(label_img, raw_label_path)
    return str(base.with_suffix(".pt").resolve())


def convert_task(
    source_root: Path,
    data_root: Path,
    raw_labels_root: Path,
    spec: TaskSpec,
    overwrite: bool,
    seed: int,
    splits: list[list[int]],
) -> None:
    task_dir = source_root / spec.source_name
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Missing {task_dir}. If this is Task_5, run finetuning/Task_5_extract.py first.")
    target_dir = data_root / spec.output_name
    if overwrite and target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []
    for subject_dir in subject_dirs(task_dir):
        if spec.kind in {"classification", "regression"}:
            paths.append(save_classification_or_regression(task_dir, target_dir, subject_dir, spec))
        elif spec.kind == "segmentation":
            paths.append(save_segmentation(task_dir, target_dir, raw_labels_root, subject_dir, spec))
        else:
            raise ValueError(f"Unsupported task kind {spec.kind}.")

    save_dataset_json(target_dir, spec, paths)
    save_json(paths, target_dir / "paths.json")
    for vals in splits:
        write_split_files(target_dir, paths, vals, seed)
    print(f"{spec.output_name}: wrote {len(paths)} samples to {target_dir}")


def main() -> None:
    args = parse_args()
    source_root = args.source.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    raw_labels_root = args.raw_labels_root.expanduser().resolve()
    splits = [args.split]
    if args.extra_split and all(value >= 0 for value in args.extra_split):
        splits.append(args.extra_split)
    for spec in selected_specs(args.tasks):
        convert_task(source_root, data_root, raw_labels_root, spec, args.overwrite, args.seed, splits)


if __name__ == "__main__":
    main()
