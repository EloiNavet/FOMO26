"""Inference-time geometry canonicalization for FOMO26 Tasks 3 and 4.

The intervention: resample an incoming volume back to the physical geometry its task was actually
fitted at, before pad/crop. These tests pin the four properties the intervention has to have to be
safe to ship:

* it is **off by default** and composes exactly as before (A);
* at the native geometry of Task 3 and Task 4 it is a **no-op** (B, C);
* off-native it **canonicalizes** to the requested spacing, and a segmentation prediction still
  lands in the original input geometry (D, E);
* it is **task-scoped** -- Tasks 1/2/5/6/7 cannot acquire it, including by env override (F);
* malformed or contradictory geometry **fails loudly** rather than resampling the wrong axes (G).
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
import torch
from asparagus.modules.transforms.presets.train import (
    CPU_clsreg_val_test_transforms_crop,
    CPU_seg_test_transforms,
)
from asparagus.modules.transforms.spacing import RuntimeSpacingError, Torch_ResampleToSpacing
from finetuning.fomo26_inference import runtime_geometry as RG

TASK3_SPACING = [1.0, 1.0, 1.0]
TASK4_SPACING = [0.5, 0.488, 0.488]


def _case(shape, spacing, orientation="RAS", seed=0):
    """A loaded case shaped like what SingleSubjectPredictDataset/ClsRegTestDataset yield."""
    rng = np.random.default_rng(seed)
    image = torch.from_numpy(rng.normal(size=(1, *shape)).astype(np.float32))
    return {
        "file_path": "synthetic.nii.gz",
        "image": image,
        "properties": {
            "original_size": list(shape),
            "original_spacing": list(spacing),
            "new_spacing": list(spacing),
            "original_orientation": orientation,
            "new_direction": orientation,
        },
    }


# ---------------------------------------------------------------- A: default is off


def test_clsreg_preset_default_composes_without_a_spacing_stage():
    composed = CPU_clsreg_val_test_transforms_crop(target_size=[16, 16, 16])
    assert not any(isinstance(stage, Torch_ResampleToSpacing) for stage in composed.transforms)
    assert len(composed.transforms) == 3


def test_seg_preset_default_composes_without_a_spacing_stage():
    composed = CPU_seg_test_transforms(patch_size=[16, 16, 16])
    assert not any(isinstance(stage, Torch_ResampleToSpacing) for stage in composed.transforms)


def test_default_clsreg_output_is_array_identical_to_the_pre_change_pipeline():
    """Disabled canonicalization must reproduce the historical pipeline exactly, not approximately."""
    target = [16, 16, 16]
    without = CPU_clsreg_val_test_transforms_crop(target_size=target)(_case((20, 22, 18), [1.0, 1.0, 1.0]))
    explicit_none = CPU_clsreg_val_test_transforms_crop(target_size=target, runtime_target_spacing=None)(
        _case((20, 22, 18), [1.0, 1.0, 1.0])
    )
    assert torch.equal(without["image"], explicit_none["image"])


# ------------------------------------------------- B / C: native geometry is a no-op


@pytest.mark.parametrize(
    "shape,spacing",
    [((32, 30, 28), TASK3_SPACING), ((32, 30, 28), TASK4_SPACING)],
    ids=["task3-native-1.0mm", "task4-native-0.5x0.488x0.488mm"],
)
def test_canonicalization_at_native_spacing_is_a_no_op(shape, spacing):
    target = [16, 16, 16]
    baseline = CPU_clsreg_val_test_transforms_crop(target_size=target)(_case(shape, spacing))
    canonical = CPU_clsreg_val_test_transforms_crop(target_size=target, runtime_target_spacing=spacing)(_case(shape, spacing))
    assert torch.equal(baseline["image"], canonical["image"])


def test_seg_canonicalization_at_task4_native_spacing_is_a_no_op():
    patch = [16, 16, 16]
    baseline = CPU_seg_test_transforms(patch_size=patch)(_case((32, 30, 28), TASK4_SPACING))
    canonical = CPU_seg_test_transforms(patch_size=patch, runtime_target_spacing=TASK4_SPACING)(
        _case((32, 30, 28), TASK4_SPACING)
    )
    assert torch.equal(baseline["image"], canonical["image"])


# ------------------------------------------------------- D: off-native canonicalizes


def test_off_native_input_is_resampled_to_the_declared_training_spacing():
    """A volume acquired at 1.25 mm must reach the network at the 1.0 mm scale it was fitted on."""
    shape = (40, 40, 40)
    acquired = _case(shape, [1.25, 1.25, 1.25])
    stage = Torch_ResampleToSpacing(target_spacing=TASK3_SPACING)
    out = stage(acquired)
    # Physical extent is preserved: 40 voxels x 1.25 mm = 50 mm -> 50 voxels at 1.0 mm.
    assert list(out["image"].shape[1:]) == [50, 50, 50]
    assert out["properties"]["new_spacing"] == TASK3_SPACING
    # And the inverse is recorded for reverse_preprocessing.
    assert out["properties"]["size_before_resample"] == list(shape)


def test_off_native_changes_the_network_input_while_native_does_not():
    target = [16, 16, 16]
    native = CPU_clsreg_val_test_transforms_crop(target_size=target, runtime_target_spacing=TASK3_SPACING)(
        _case((32, 30, 28), TASK3_SPACING)
    )
    off_native = CPU_clsreg_val_test_transforms_crop(target_size=target, runtime_target_spacing=TASK3_SPACING)(
        _case((32, 30, 28), [1.4, 1.4, 1.4])
    )
    assert not torch.equal(native["image"], off_native["image"])


# ------------------------------- E: segmentation is restored into the input geometry


def test_segmentation_prediction_is_restored_to_the_original_input_shape():
    from asparagus.functional.reverse_preprocessing import reverse_preprocessing

    shape = (40, 38, 36)
    patch = [24, 24, 24]
    case = CPU_seg_test_transforms(patch_size=patch, runtime_target_spacing=TASK4_SPACING)(_case(shape, [0.75, 0.732, 0.732]))
    props = case["properties"]
    # A prediction defined on the padded, resampled grid the network actually saw.
    logits = torch.zeros((1, 3, *case["image"].shape[1:]), dtype=torch.float32)
    restored = reverse_preprocessing(logits, props)
    assert list(restored.shape[2:]) == list(shape), "the mask must land in the input volume's own grid, not the resampled one"


def _writer_props(shape, affine):
    return {
        "original_size": list(shape),
        "nifti_metadata": {"affine": affine, "header": None, "reoriented": False},
    }


def test_written_mask_keeps_the_input_affine(tmp_path):
    """The repository's writer must put the mask at exactly the path it was given.

    This used to call ``gardening_tools.save_prediction_from_logits`` directly, which tests the
    dependency rather than our contract -- and the dependency's 0.3.2 API takes a *basename*, so
    the assertion could never hold. The shipped callback is what the release runs, so that is what
    is asserted here.
    """
    from asparagus.modules.callbacks.prediction_writer import _write_prediction

    affine = np.diag([0.5, 0.488, 0.488, 1.0])
    shape = (8, 9, 10)
    logits = np.zeros((1, 3, *shape), dtype=np.float32)
    out = tmp_path / "mask.nii.gz"
    _write_prediction(logits, str(out), _writer_props(shape, affine))
    written = nib.load(str(out))
    assert written.shape == shape
    np.testing.assert_allclose(written.affine, affine)


def test_writer_does_not_produce_a_doubled_nifti_suffix(tmp_path):
    """Regression: gardening_tools 0.3.2 appends '.nii.gz', so a naive call wrote mask.nii.gz.nii.gz.

    Nothing raised when that happened -- the callback returned normally and the requested file
    simply did not exist -- so the only way to catch a reintroduction is to assert on the directory
    listing, not on the absence of an exception.
    """
    from asparagus.modules.callbacks.prediction_writer import _write_prediction

    shape = (4, 5, 6)
    out = tmp_path / "case_042.nii.gz"
    _write_prediction(
        np.zeros((1, 2, *shape), dtype=np.float32),
        str(out),
        _writer_props(shape, np.eye(4)),
    )
    produced = sorted(p.name for p in tmp_path.iterdir())
    assert produced == ["case_042.nii.gz"], produced
    assert not (tmp_path / "case_042.nii.gz.nii.gz").exists()


def test_writer_rejects_an_output_path_without_the_nifti_suffix(tmp_path):
    """Fail loudly rather than guessing where the caller wanted the file."""
    import pytest as _pytest
    from asparagus.modules.callbacks.prediction_writer import _write_prediction

    shape = (4, 5, 6)
    with _pytest.raises(ValueError, match=r"\.nii\.gz output path"):
        _write_prediction(
            np.zeros((1, 2, *shape), dtype=np.float32),
            str(tmp_path / "mask"),
            _writer_props(shape, np.eye(4)),
        )


def test_prediction_writer_callback_writes_the_requested_case_file(tmp_path):
    """End-to-end through the callback the release actually installs."""
    from asparagus.modules.callbacks.prediction_writer import WritePredictionFromLogits

    shape = (4, 5, 6)
    writer = WritePredictionFromLogits(output_dir=str(tmp_path))
    writer.write_on_batch_end(
        None,
        None,
        {
            "logits": np.zeros((1, 2, *shape), dtype=np.float32),
            "properties": _writer_props(shape, np.eye(4)),
            "id": "sub_007",
        },
        None,
        None,
        0,
        0,
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sub_007.nii.gz"]


# ------------------------------------------------------------- F: task isolation


@pytest.mark.parametrize("task", ["1", "2", "5", "6", "7"])
def test_tasks_outside_scope_declare_no_runtime_geometry(task):
    assert RG.task_runtime_geometry(task) is None
    assert RG.task_runtime_target_spacing(task) is None
    spacing, _note = RG.resolve_for_inputs(task, ["case.nii.gz"])
    assert spacing is None


@pytest.mark.parametrize("task,expected", [("3", TASK3_SPACING), ("4", TASK4_SPACING)])
def test_in_scope_tasks_declare_the_audited_geometry(task, expected):
    assert RG.task_runtime_target_spacing(task) == expected


def test_env_override_cannot_switch_canonicalization_on_for_an_out_of_scope_task(monkeypatch):
    """The override retargets an opted-in task; it must never opt a task in."""
    monkeypatch.setenv(RG.TARGET_SPACING_ENV, "1.0,1.0,1.0")
    for task in ("1", "2", "5", "6", "7"):
        spacing, _ = RG.resolve_for_inputs(task, ["case.nii.gz"])
        assert spacing is None, f"task {task} must not acquire canonicalization by env"


def test_env_override_can_retarget_or_disable_an_in_scope_task(monkeypatch):
    monkeypatch.setenv(RG.TARGET_SPACING_ENV, "none")
    assert RG.resolve_for_inputs("3", ["case.nii.gz"])[0] is None
    monkeypatch.setenv(RG.TARGET_SPACING_ENV, "0.9,0.9,0.9")
    assert RG.resolve_for_inputs("3", ["case.nii.gz"])[0] == [0.9, 0.9, 0.9]


def test_tasks_6_7_embedding_preset_has_no_spacing_stage():
    """Tasks 6/7 run a different preset and must not inherit this behaviour by code sharing."""
    from asparagus.modules.transforms.presets.pretrain import CPU_val_transforms

    composed = CPU_val_transforms([16, 16, 16])
    assert not any(isinstance(stage, Torch_ResampleToSpacing) for stage in composed.transforms)


def test_a_caller_that_names_no_task_gets_no_canonicalization():
    assert RG.resolve_for_inputs(None, ["case.nii.gz"])[0] is None


# ----------------------------------------------------- G: loud failure on bad geometry


def test_missing_geometry_is_refused_rather_than_guessed():
    stage = Torch_ResampleToSpacing(target_spacing=TASK3_SPACING)
    with pytest.raises(RuntimeSpacingError, match="no geometry"):
        stage({"image": torch.zeros((1, 8, 8, 8)), "properties": {}})


@pytest.mark.parametrize("bad", [[0.0, 1.0, 1.0], [-1.0, 1.0, 1.0], [float("nan"), 1.0, 1.0]])
def test_malformed_source_spacing_is_refused(bad):
    stage = Torch_ResampleToSpacing(target_spacing=TASK3_SPACING)
    case = _case((8, 8, 8), bad)
    with pytest.raises(RuntimeSpacingError):
        stage(case)


def test_orientation_mismatch_is_refused_by_default():
    """A per-axis spacing target under a different axis order resamples the wrong axes."""
    with pytest.raises(RG.RuntimeGeometryError, match="fitted on RAS"):
        RG.resolve_target_spacing_for_case("3", "LPS", policy="error")


def test_orientation_mismatch_can_fall_back_to_historical_behaviour():
    spacing, note = RG.resolve_target_spacing_for_case("3", "LPS", policy="skip")
    assert spacing is None
    assert "native geometry" in note


def test_matching_orientation_canonicalizes():
    spacing, _ = RG.resolve_target_spacing_for_case("3", "RAS", policy="error")
    assert spacing == TASK3_SPACING


def test_unknown_task_is_refused():
    with pytest.raises(RG.RuntimeGeometryError, match="not declared"):
        RG.task_runtime_geometry("99")


def test_bad_env_override_is_refused(monkeypatch):
    monkeypatch.setenv(RG.TARGET_SPACING_ENV, "1.0,2.0")
    with pytest.raises(RG.RuntimeGeometryError):
        RG.resolve_for_inputs("3", ["case.nii.gz"])
