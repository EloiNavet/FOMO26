"""Geometry proofs for runtime physical-spacing normalization.

Task 2 is acquired between 0.43 mm and 0.90 mm in plane with 5.2-7.5 mm slices, so every property
these tests pin down -- shared grid, native axes, label integrity, and above all an inverse that
lands a prediction back on the source grid -- is the difference between a scale-normalized cohort
and a silently corrupted one.
"""

import numpy as np
import pytest
import torch
from asparagus.functional.reverse_preprocessing import reverse_preprocessing
from asparagus.modules.transforms.presets.train import (
    CPU_seg_test_transforms,
    CPU_seg_train_transforms,
    CPU_seg_val_transforms,
)
from asparagus.modules.transforms.spacing import (
    NATIVE_SPACING,
    RuntimeSpacingError,
    Torch_ResampleToSpacing,
    resolve_target_spacing,
    target_shape_for_spacing,
)


def make_case(shape, spacing, n_channels=3, label_value=1, with_properties=True):
    image = torch.rand((n_channels, *shape), dtype=torch.float32)
    label = torch.zeros((1, *shape), dtype=torch.float32)
    data_dict = {"image": image, "label": label, "file_path": "case.pt"}
    if with_properties:
        data_dict["properties"] = {
            "original_size": list(shape),
            "size_before_resample": list(shape),
            "new_size": list(shape),
            "original_spacing": list(spacing),
            "new_spacing": list(spacing),
            "crop_box": [],
            "pad_box": [],
        }
    else:
        data_dict["source_spacing"] = list(spacing)
    return data_dict, label_value


# --------------------------------------------------------------------------------------------
# 1. NO-OP
# --------------------------------------------------------------------------------------------


def test_noop_when_source_spacing_already_equals_request():
    data_dict, _ = make_case((32, 32, 8), (0.9, 0.9, 6.0))
    before_image = data_dict["image"].clone()
    before_label = data_dict["label"].clone()

    out = Torch_ResampleToSpacing(target_spacing=[0.9, 0.9, NATIVE_SPACING])(data_dict)

    assert list(out["image"].shape[1:]) == [32, 32, 8]
    assert torch.equal(out["image"], before_image)
    assert torch.equal(out["label"], before_label)
    # The inverse is the identity: nothing to interpolate back to.
    assert out["properties"]["size_before_resample"] == [32, 32, 8]


def test_disabled_transform_returns_input_untouched():
    data_dict, _ = make_case((17, 19, 5), (0.4492, 0.4492, 6.5))
    before = data_dict["image"].clone()
    out = Torch_ResampleToSpacing(target_spacing=None)(data_dict)
    assert torch.equal(out["image"], before)
    assert out["properties"]["new_spacing"] == [0.4492, 0.4492, 6.5]
    assert "runtime_resample" not in out["properties"]


# --------------------------------------------------------------------------------------------
# 2. 0.45 -> 0.9 IN-PLANE, physical coverage preserved
# --------------------------------------------------------------------------------------------


def test_fine_inplane_halves_and_preserves_physical_coverage():
    shape, spacing = (512, 512, 30), (0.45, 0.45, 6.0)
    data_dict, _ = make_case(shape, spacing, n_channels=1)

    out = Torch_ResampleToSpacing(target_spacing=[0.9, 0.9, NATIVE_SPACING])(data_dict)

    assert list(out["image"].shape[1:]) == [256, 256, 30]
    assert out["properties"]["new_spacing"] == [0.9, 0.9, 6.0]

    fov_before = np.asarray(shape) * np.asarray(spacing)
    fov_after = np.asarray(out["image"].shape[1:]) * np.asarray(out["properties"]["new_spacing"])
    assert np.allclose(fov_before, fov_after, rtol=0, atol=1.0)


@pytest.mark.parametrize(
    "shape,spacing,expected",
    [
        ((384, 512, 30), (0.4296875, 0.4296875, 5.6), [183, 244, 30]),
        ((512, 512, 21), (0.4492188, 0.4492188, 6.5), [256, 256, 21]),
        ((232, 256, 29), (0.8984375, 0.8984375, 5.2), [232, 256, 29]),
        ((270, 320, 21), (0.71875, 0.71875, 6.75), [216, 256, 21]),
    ],
)
def test_real_task2_geometries_resolve_to_expected_grids(shape, spacing, expected):
    """Every distinct in-plane geometry actually present in the Task-2 corpus."""
    assert target_shape_for_spacing(shape, spacing, [0.9, 0.9, spacing[2]]) == expected


# --------------------------------------------------------------------------------------------
# 3. NATIVE Z
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("z_spacing", [5.2, 6.0, 6.5, 7.5])
def test_native_axis_is_left_exactly_as_acquired(z_spacing):
    data_dict, _ = make_case((512, 512, 21), (0.45, 0.45, z_spacing), n_channels=1)
    out = Torch_ResampleToSpacing(target_spacing=[0.9, 0.9, NATIVE_SPACING])(data_dict)
    assert out["image"].shape[3] == 21
    assert out["properties"]["new_spacing"][2] == pytest.approx(z_spacing)


def test_native_sentinel_resolves_per_axis():
    assert resolve_target_spacing([0.9, 0.9, NATIVE_SPACING], [0.45, 0.45, 6.5]) == [0.9, 0.9, 6.5]


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), None, True, "isotropic"])
def test_invalid_spacing_requests_are_refused(bad):
    with pytest.raises(RuntimeSpacingError):
        resolve_target_spacing([0.9, 0.9, bad], [0.45, 0.45, 6.5])


def test_wrong_axis_count_is_refused():
    with pytest.raises(RuntimeSpacingError):
        resolve_target_spacing([0.9, 0.9], [0.45, 0.45, 6.5])


@pytest.mark.parametrize("bad_source", [[0.0, 0.9, 6.0], [0.9, -0.9, 6.0], [0.9, 0.9, float("nan")]])
def test_invalid_source_spacing_is_refused(bad_source):
    with pytest.raises(RuntimeSpacingError):
        resolve_target_spacing([0.9, 0.9, NATIVE_SPACING], bad_source)


def test_missing_geometry_fails_closed_rather_than_guessing():
    data_dict = {"image": torch.rand(1, 8, 8, 4), "label": torch.zeros(1, 8, 8, 4)}
    with pytest.raises(RuntimeSpacingError, match="carries no geometry"):
        Torch_ResampleToSpacing(target_spacing=[0.9, 0.9, NATIVE_SPACING])(data_dict)


# --------------------------------------------------------------------------------------------
# 4. MULTI-MODAL ALIGNMENT
# --------------------------------------------------------------------------------------------


def test_modalities_stay_physically_aligned_on_one_shared_grid():
    shape, spacing = (64, 64, 10), (0.45, 0.45, 6.0)
    image = torch.zeros((3, *shape), dtype=torch.float32)
    # An identical landmark in every modality.
    image[:, 40:48, 20:28, 4:6] = 1.0
    data_dict, _ = make_case(shape, spacing, n_channels=3)
    data_dict["image"] = image

    out = Torch_ResampleToSpacing(target_spacing=[0.9, 0.9, NATIVE_SPACING])(data_dict)

    resampled = out["image"]
    assert list(resampled.shape) == [3, 32, 32, 10]
    centroids = []
    for channel in range(3):
        idx = (resampled[channel] > 0.5).nonzero(as_tuple=False).float()
        assert idx.numel() > 0, f"landmark vanished from modality {channel}"
        centroids.append(idx.mean(dim=0))
    for channel in range(1, 3):
        assert torch.allclose(centroids[0], centroids[channel], atol=1e-4)


# --------------------------------------------------------------------------------------------
# 5. LABEL INTEGRITY
# --------------------------------------------------------------------------------------------


def test_nearest_neighbour_labels_introduce_no_new_ids():
    shape, spacing = (64, 64, 10), (0.45, 0.45, 6.0)
    data_dict, _ = make_case(shape, spacing)
    label = torch.zeros((1, *shape), dtype=torch.float32)
    label[0, 10:20, 10:20, 2:5] = 1.0
    label[0, 40:50, 40:50, 5:8] = 2.0
    data_dict["label"] = label
    before = set(int(v) for v in torch.unique(label))

    out = Torch_ResampleToSpacing(target_spacing=[0.9, 0.9, NATIVE_SPACING])(data_dict)

    after = set(int(v) for v in torch.unique(out["label"]))
    assert after <= before, f"resampling invented labels {after - before}"
    assert after == {0, 1, 2}
    assert out["label"].dtype == label.dtype
    # Discrete, not blended.
    assert torch.equal(out["label"], out["label"].round())


def test_label_dtype_is_preserved_for_integer_labels():
    shape, spacing = (32, 32, 8), (0.45, 0.45, 6.0)
    data_dict, _ = make_case(shape, spacing)
    label = torch.zeros((1, *shape), dtype=torch.uint8)
    label[0, 4:12, 4:12, 2:4] = 3
    data_dict["label"] = label
    out = Torch_ResampleToSpacing(target_spacing=[0.9, 0.9, NATIVE_SPACING])(data_dict)
    assert out["label"].dtype == torch.uint8
    assert set(int(v) for v in torch.unique(out["label"])) <= {0, 3}


# --------------------------------------------------------------------------------------------
# 6. FOREGROUND LOCATION
# --------------------------------------------------------------------------------------------


def test_foreground_object_stays_at_the_same_physical_location():
    shape, spacing = (64, 64, 10), (0.45, 0.45, 6.0)
    data_dict, _ = make_case(shape, spacing, n_channels=1)
    label = torch.zeros((1, *shape), dtype=torch.float32)
    label[0, 32:40, 16:24, 4:6] = 1.0
    data_dict["label"] = label

    out = Torch_ResampleToSpacing(target_spacing=[0.9, 0.9, NATIVE_SPACING])(data_dict)

    src_idx = (label[0] > 0.5).nonzero(as_tuple=False).float().mean(dim=0)
    dst_idx = (out["label"][0] > 0.5).nonzero(as_tuple=False).float().mean(dim=0)
    src_mm = src_idx.numpy() * np.asarray(spacing)
    dst_mm = dst_idx.numpy() * np.asarray(out["properties"]["new_spacing"])
    # Within one target voxel in plane, exact through plane.
    assert np.allclose(src_mm, dst_mm, atol=1.0)


def test_cached_foreground_locations_are_remapped_onto_the_new_grid():
    shape, spacing = (512, 512, 20), (0.45, 0.45, 6.0)
    data_dict, _ = make_case(shape, spacing, n_channels=1)
    data_dict["foreground_locations"] = {"1": [[400, 300, 10], [0, 0, 0], [511, 511, 19]]}

    out = Torch_ResampleToSpacing(target_spacing=[0.9, 0.9, NATIVE_SPACING])(data_dict)

    remapped = out["foreground_locations"]["1"]
    assert remapped[0] == [200, 150, 10]
    assert remapped[1] == [0, 0, 0]
    # Never allowed to address a voxel outside the resampled grid: torch_crop turns an
    # out-of-range location into np.random.randint(low > high), which raises.
    target = list(out["image"].shape[1:])
    for location in remapped:
        assert all(0 <= coordinate < size for coordinate, size in zip(location, target))


def test_remapped_locations_keep_torch_crop_sampling_usable():
    """The stale-coordinate failure mode, reproduced end to end through the real crop."""
    from asparagus.modules.transforms.crop import Torch_Crop

    shape, spacing = (512, 512, 20), (0.45, 0.45, 6.0)
    data_dict, _ = make_case(shape, spacing, n_channels=1)
    data_dict["foreground_locations"] = {"1": [[500, 500, 18]]}

    resampled = Torch_ResampleToSpacing(target_spacing=[0.9, 0.9, NATIVE_SPACING])(data_dict)
    cropped = Torch_Crop(patch_size=[64, 64, 16], p_oversample_foreground=1.0)(resampled)
    assert list(cropped["image"].shape[1:]) == [64, 64, 16]


# --------------------------------------------------------------------------------------------
# 7. ROUND TRIP through the real reverse_preprocessing
# --------------------------------------------------------------------------------------------


def test_round_trip_restores_prediction_to_source_grid_with_object_in_place():
    shape, spacing = (512, 512, 21), (0.45, 0.45, 6.5)
    patch = [160, 160, 96]
    data_dict, _ = make_case(shape, spacing, n_channels=3)
    label = torch.zeros((1, *shape), dtype=torch.float32)
    label[0, 300:340, 200:240, 8:12] = 1.0
    data_dict["label"] = label

    composed = CPU_seg_test_transforms(patch_size=patch, runtime_target_spacing=[0.9, 0.9, NATIVE_SPACING])
    out = composed(data_dict)
    properties = out["properties"]

    # Resampled to 0.9 mm in plane, then padded up to the patch depth.
    assert properties["size_before_resample"] == [512, 512, 21]
    assert list(properties["shape_before_pad"]) == [256, 256, 21]
    assert properties["pad_box"] == [0, 0, 0, 0, 37, 38]

    # A two-class prediction on the padded, resampled grid: class 1 exactly where the label is.
    padded_shape = list(out["image"].shape[1:])
    logits = torch.zeros((1, 2, *padded_shape))
    logits[:, 0] = 1.0
    logits[:, 1, 150:170, 100:120, 37 + 8 : 37 + 12] = 5.0

    restored = reverse_preprocessing(logits, properties)

    assert list(restored.shape) == [1, 2, *shape], "prediction did not land on the source grid"
    predicted = restored.argmax(dim=1)[0]
    assert predicted.sum() > 0, "foreground did not survive restoration"
    src_centroid = (label[0] > 0.5).nonzero(as_tuple=False).float().mean(dim=0)
    pred_centroid = (predicted > 0).nonzero(as_tuple=False).float().mean(dim=0)
    assert torch.allclose(src_centroid, pred_centroid, atol=6.0)


def test_round_trip_is_identity_shaped_when_resampling_is_disabled():
    shape, spacing = (232, 256, 29), (0.8984375, 0.8984375, 5.2)
    patch = [160, 160, 96]
    data_dict, _ = make_case(shape, spacing, n_channels=3)

    out = CPU_seg_test_transforms(patch_size=patch)(data_dict)
    properties = out["properties"]
    assert properties["size_before_resample"] == [232, 256, 29]
    assert "runtime_resample" not in properties

    padded_shape = list(out["image"].shape[1:])
    logits = torch.zeros((1, 2, *padded_shape))
    restored = reverse_preprocessing(logits, properties)
    assert list(restored.shape) == [1, 2, *shape]


# --------------------------------------------------------------------------------------------
# 8. PROPERTY CONTRACT
# --------------------------------------------------------------------------------------------


def test_provenance_fields_are_internally_consistent():
    shape, spacing = (512, 512, 21), (0.4492188, 0.4492188, 6.5)
    data_dict, _ = make_case(shape, spacing)
    out = Torch_ResampleToSpacing(target_spacing=[0.9, 0.9, NATIVE_SPACING])(data_dict)
    properties = out["properties"]

    assert properties["size_before_resample"] == list(shape)
    assert properties["runtime_resample"]["source_shape"] == list(shape)
    assert properties["runtime_resample"]["source_spacing"] == pytest.approx(list(spacing))
    assert properties["runtime_resample"]["target_spacing"] == [0.9, 0.9, 6.5]
    assert properties["new_spacing"] == [0.9, 0.9, 6.5]
    # Source-space geometry is what DSC/NSD are measured against; it must not move.
    assert properties["original_spacing"] == list(spacing)
    assert properties["original_size"] == list(shape)


def test_source_space_spacing_is_what_the_metric_path_resolves():
    """The metric path reads ("spacing", "itk_spacing", "original_spacing", ...) in that order."""
    shape, spacing = (512, 512, 21), (0.4492188, 0.4492188, 6.5)
    data_dict, _ = make_case(shape, spacing)
    out = Torch_ResampleToSpacing(target_spacing=[0.9, 0.9, NATIVE_SPACING])(data_dict)
    properties = out["properties"]

    resolved = None
    for key in ("spacing", "itk_spacing", "original_spacing", "spacing_after_resampling"):
        if properties.get(key) is not None:
            resolved = tuple(float(value) for value in properties[key])
            break
    assert resolved == pytest.approx(tuple(spacing)), "NSD would be measured at the wrong spacing"


def test_refuses_a_corpus_that_was_already_resampled_during_preprocessing():
    shape, spacing = (256, 256, 21), (0.9, 0.9, 6.5)
    data_dict, _ = make_case(shape, spacing)
    # An earlier preprocessing resample already owns the inverse field.
    data_dict["properties"]["size_before_resample"] = [512, 512, 21]

    with pytest.raises(RuntimeSpacingError, match="already describes an earlier resample"):
        Torch_ResampleToSpacing(target_spacing=[0.45, 0.45, NATIVE_SPACING])(data_dict)


def test_train_path_without_properties_uses_source_spacing_key():
    shape, spacing = (512, 512, 20), (0.45, 0.45, 6.0)
    data_dict, _ = make_case(shape, spacing, n_channels=3, with_properties=False)
    out = Torch_ResampleToSpacing(target_spacing=[0.9, 0.9, NATIVE_SPACING])(data_dict)
    assert list(out["image"].shape[1:]) == [256, 256, 20]
    assert "properties" not in out


# --------------------------------------------------------------------------------------------
# 9 + 10. NON-RESAMPLED COMPOSITION AND TASK-4 REGRESSION
# --------------------------------------------------------------------------------------------


def _stage_names(composed):
    return [type(stage).__name__ for stage in composed.transforms]


def test_default_seg_transform_composition_is_unchanged():
    """Disabled runtime resampling must leave every historical pipeline byte-identical."""
    assert _stage_names(CPU_seg_train_transforms(patch_size=[128, 128, 128])) == [
        "Torch_Normalize",
        "Torch_Pad",
        "Torch_Crop",
        "Torch_Spatial",
        "Torch_Mirror",
    ]
    assert _stage_names(CPU_seg_val_transforms(patch_size=[128, 128, 128])) == [
        "Torch_Normalize",
        "Torch_Pad",
        "Torch_Crop",
    ]
    assert _stage_names(CPU_seg_test_transforms(patch_size=[128, 128, 128])) == [
        "Torch_Normalize",
        "Torch_Pad",
    ]


def test_task4_final_protocol_composes_exactly_as_frozen():
    """Task 4 is frozen at the five-fold BUDGET recipe; this feature must not perturb it."""
    composed = CPU_seg_train_transforms(patch_size=[128, 128, 128], p_oversample_foreground=0.33)
    assert _stage_names(composed) == [
        "Torch_Normalize",
        "Torch_Pad",
        "Torch_Crop",
        "Torch_Spatial",
        "Torch_Mirror",
    ]
    crop = composed.transforms[2]
    assert crop.p_oversample_foreground == 0.33
    assert list(crop.patch_size) == list(composed.transforms[1].patch_size)


def test_enabled_composition_inserts_exactly_one_stage_after_normalize():
    composed = CPU_seg_train_transforms(patch_size=[160, 160, 96], runtime_target_spacing=[0.9, 0.9, NATIVE_SPACING])
    assert _stage_names(composed) == [
        "Torch_Normalize",
        "Torch_ResampleToSpacing",
        "Torch_Pad",
        "Torch_Crop",
        "Torch_Spatial",
        "Torch_Mirror",
    ]


def test_task4_sample_is_untouched_by_a_disabled_pipeline():
    """A real Task-4-shaped case through the frozen val pipeline, unchanged in geometry."""
    shape, spacing = (128, 128, 128), (1.0, 1.0, 1.0)
    data_dict, _ = make_case(shape, spacing, n_channels=1)
    out = CPU_seg_val_transforms(patch_size=[128, 128, 128])(data_dict)
    assert list(out["image"].shape[1:]) == [128, 128, 128]
    assert out["properties"]["new_spacing"] == [1.0, 1.0, 1.0]
    assert "runtime_resample" not in out["properties"]


# --------------------------------------------------------------------------------------------
# EVALUATION-TIME GEOMETRY: a fold must be scored on the geometry it was trained on
# --------------------------------------------------------------------------------------------


def test_evaluator_reads_training_spacing_from_the_checkpoint_config():
    """Otherwise a 0.9 mm-trained fold is scored on native volumes and the number is meaningless."""
    from finetuning.fomo26_inference.seg_ensemble import _runtime_target_spacing_of
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"transforms": {"runtime_target_spacing": [0.9, 0.9, NATIVE_SPACING]}})
    assert _runtime_target_spacing_of(cfg) == [0.9, 0.9, NATIVE_SPACING]


def test_evaluator_spacing_is_none_for_runs_predating_the_setting():
    from finetuning.fomo26_inference.seg_ensemble import _runtime_target_spacing_of
    from omegaconf import OmegaConf

    # A historical run: the key does not exist at all in its resolved config.
    assert _runtime_target_spacing_of(OmegaConf.create({"transforms": {"normalize": True}})) is None
    # And the disabled default.
    assert _runtime_target_spacing_of(OmegaConf.create({"transforms": {"runtime_target_spacing": None}})) is None


def test_evaluator_spacing_reaches_the_composed_test_pipeline():
    from finetuning.fomo26_inference.seg_ensemble import _runtime_target_spacing_of
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"transforms": {"runtime_target_spacing": [0.9, 0.9, NATIVE_SPACING]}})
    composed = CPU_seg_test_transforms(patch_size=[160, 160, 96], runtime_target_spacing=_runtime_target_spacing_of(cfg))
    assert _stage_names(composed) == ["Torch_Normalize", "Torch_ResampleToSpacing", "Torch_Pad"]
    assert composed.transforms[1].target_spacing == [0.9, 0.9, NATIVE_SPACING]
