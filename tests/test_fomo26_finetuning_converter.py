import nibabel as nib
import numpy as np
import torch
from finetuning.prepare_fomo26_asparagus import TASK_SPECS, convert_task
from pathlib import Path


def _save_nii(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine=np.eye(4)), path)


def _write_label(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _make_subject(task_dir: Path, subject: str, modalities: list[str], shape=(8, 9, 10)) -> None:
    session = task_dir / "preprocessed" / subject / "ses-01"
    for index, modality in enumerate(modalities):
        _save_nii(session / f"{modality}.nii.gz", np.full(shape, index + 1, dtype=np.float32))


def test_task1_converter_writes_classification_and_empty_negative_segmentation(tmp_path):
    source = tmp_path / "source"
    data_root = tmp_path / "data"
    raw_root = tmp_path / "raw"
    task_dir = source / "Task_1"

    _make_subject(task_dir, "sub-01", ["adc", "dwi_b1000", "flair", "swi"])
    _write_label(task_dir / "labels" / "sub-01" / "ses-01" / "label.txt", "0")

    convert_task(source, data_root, raw_root, TASK_SPECS["task1_presence"], overwrite=False, seed=1, splits=[[70, 15, 15]])
    cls_path = data_root / "CLS901_FOMO26_Task1_presence" / "sub-01" / "ses-01" / "scan.pt"
    image, label = torch.load(cls_path, weights_only=False)
    assert image.shape == (4, 8, 9, 10)
    assert label.item() == 0

    convert_task(source, data_root, raw_root, TASK_SPECS["task1_lesion"], overwrite=False, seed=1, splits=[[70, 15, 15]])
    seg_path = data_root / "SEG901_FOMO26_Task1_lesion" / "sub-01" / "ses-01" / "scan.pt"
    tensor = torch.load(seg_path, weights_only=False)
    assert tensor.shape == (5, 8, 9, 10)
    assert torch.count_nonzero(tensor[-1]).item() == 0
    assert (raw_root / "SEG901_FOMO26_Task1_lesion" / "sub-01" / "ses-01" / "scan_label.nii.gz").is_file()


def test_task4_converter_preserves_multiclass_labels_and_split_files(tmp_path):
    source = tmp_path / "source"
    data_root = tmp_path / "data"
    raw_root = tmp_path / "raw"
    task_dir = source / "Task_4"

    for idx in range(4):
        subject = f"sub-{idx + 1:02d}"
        _make_subject(task_dir, subject, ["t2w"], shape=(7, 8, 9))
        label = np.zeros((7, 8, 9), dtype=np.uint8)
        label[1:3] = 1
        label[4:6] = 2
        _save_nii(task_dir / "labels" / subject / "ses-01" / "seg.nii.gz", label)

    convert_task(
        source,
        data_root,
        raw_root,
        TASK_SPECS["task4_multiclass"],
        overwrite=False,
        seed=1,
        splits=[[70, 15, 15], [85, 15, 0]],
    )

    task_out = data_root / "SEG904_FOMO26_Task4_multiclass"
    tensor = torch.load(task_out / "sub-01" / "ses-01" / "scan.pt", weights_only=False)
    assert tensor.shape == (2, 7, 8, 9)
    assert set(torch.unique(tensor[-1]).tolist()) == {0.0, 1.0, 2.0}
    assert (task_out / "dataset.json").is_file()
    assert (task_out / "paths.json").is_file()
    assert (task_out / "split_70_15_15.json").is_file()
    assert (task_out / "TEST_70_15_15.json").is_file()
    assert (task_out / "split_85_15_00.json").is_file()
    assert (task_out / "TEST_85_15_00.json").is_file()
