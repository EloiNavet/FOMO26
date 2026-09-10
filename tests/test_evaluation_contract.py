"""Sliding-window normalisation, flip-TTA label safety and ensemble-space semantics.

Three gaps this pins down:

* the vendored sliding window sums logits into a zero canvas and never divides by the overlap
  count, so overlapped voxels are systematically sharpened before the softmax -- and the artefact is
  invisible whenever the volume fits in a single window;
* flip TTA mirrors predictions with no per-task statement that mirroring preserves the label space;
* fold/TTA averaging in probability space and in logit space are different estimators.

Each fix is opt-in, so the default path must stay bit-identical to what produced existing numbers.
"""

import json
import pytest
import torch
import torch.nn as nn
from asparagus.modules.lightning_modules.segmentation_module import SegmentationModule
from finetuning.fomo26_inference.tta_safety import FlipTTAUnsafe, assert_flip_tta_allowed, task_flip_policy
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[1] / "finetuning" / "fomo26_inference" / "task_definitions.json"


class _ConstantNet(nn.Module):
    """Emits the same logits for every window, so any overlap artefact is pure bookkeeping."""

    num_classes = 2

    def __init__(self, value=1.0):
        super().__init__()
        self.value = value
        self.param = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out = torch.zeros((x.shape[0], self.num_classes, *x.shape[2:]), dtype=torch.float32)
        out[:, 1] = self.value
        return out

    def sliding_window_predict(self, data, patch_size, overlap):
        from gardening_tools.modules.networks.utils import get_steps_for_sliding_window

        canvas = torch.zeros((1, self.num_classes, *data.shape[2:]))
        steps = get_steps_for_sliding_window(list(data.shape[2:]), patch_size, overlap)
        px, py, pz = patch_size
        for xs in steps[0]:
            for ys in steps[1]:
                for zs in steps[2]:
                    out = self.forward(data[:, :, xs : xs + px, ys : ys + py, zs : zs + pz])
                    canvas[:, :, xs : xs + px, ys : ys + py, zs : zs + pz] += out
        return canvas


def _module(**kwargs):
    return SegmentationModule(model=_ConstantNet(), inference_patch_size=[4, 4, 4], **kwargs)


def test_default_path_is_the_unnormalised_vendored_window(tmp_path):
    """The historical behaviour must remain reachable and unchanged, artefact included."""
    module = _module()
    assert module.window_blending == "none"
    assert module.sliding_window_overlap == 0.5

    x = torch.zeros(1, 1, 8, 4, 4)
    logits = module._sliding_window_predict_padded(x, [4, 4, 4])
    # Windows at x=0 and x=4 overlap at x in [2,4) with overlap=0.5 -> summed logits there.
    assert logits[0, 1].max().item() > 1.0, "the unnormalised artefact should still be present"


def test_normalised_window_makes_a_constant_field_constant():
    """A model emitting a constant logit must yield that constant everywhere, at any overlap."""
    module = _module(window_blending="uniform")
    x = torch.zeros(1, 1, 12, 4, 4)
    logits = module._normalized_sliding_window(x, [4, 4, 4])
    assert torch.allclose(logits[0, 1], torch.ones_like(logits[0, 1]), atol=1e-5)
    assert torch.allclose(logits[0, 0], torch.zeros_like(logits[0, 0]), atol=1e-5)


def test_gaussian_blending_also_preserves_a_constant_field():
    module = _module(window_blending="gaussian")
    x = torch.zeros(1, 1, 12, 4, 4)
    logits = module._normalized_sliding_window(x, [4, 4, 4])
    assert torch.allclose(logits[0, 1], torch.ones_like(logits[0, 1]), atol=1e-4)


@pytest.mark.parametrize("overlap", [0.0, 0.25, 0.5, 0.75])
def test_normalised_window_is_overlap_invariant_for_a_constant_field(overlap):
    """Overlap is an inference-cost knob; it must not move the prediction of a constant model."""
    module = _module(window_blending="uniform", sliding_window_overlap=overlap)
    x = torch.zeros(1, 1, 12, 4, 4)
    logits = module._normalized_sliding_window(x, [4, 4, 4])
    assert torch.allclose(logits[0, 1], torch.ones_like(logits[0, 1]), atol=1e-5)


def test_overlap_is_configurable_rather_than_a_literal():
    assert _module(sliding_window_overlap=0.25).sliding_window_overlap == 0.25


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_invalid_overlap_is_refused(bad):
    with pytest.raises(ValueError, match="sliding_window_overlap"):
        _module(sliding_window_overlap=bad)


def test_unknown_blending_mode_is_refused():
    with pytest.raises(ValueError, match="window_blending"):
        _module(window_blending="bilinear")


# --- flip TTA label safety -------------------------------------------------------------------


def test_every_challenge_task_declares_its_flip_invariance():
    tasks = json.loads(REGISTRY.read_text())["tasks"]
    for task in ("1", "2", "3", "4", "5"):
        policy = task_flip_policy(task)
        assert isinstance(policy["flip_invariant_labels"], bool)
        assert policy["rationale"], f"task {task} must say WHY mirroring is or is not label-preserving"
        assert "flip_invariance_rationale" in tasks[task]


@pytest.mark.parametrize("task", ["1", "2", "3", "4", "5"])
@pytest.mark.parametrize("tta", ["none", "flip3", "flip7", "auto"])
def test_flip_tta_is_allowed_for_todays_label_spaces(task, tta):
    """No current task encodes laterality, so this must be a no-op -- by assertion, not by luck."""
    assert_flip_tta_allowed(task, tta)


def test_flip_tta_is_refused_when_a_task_declares_laterality(monkeypatch):
    """A future label space that splits a class by side must fail here, not in a submission."""
    import finetuning.fomo26_inference.tta_safety as tta_safety

    monkeypatch.setattr(
        tta_safety,
        "task_flip_policy",
        lambda task: {"flip_invariant_labels": False, "flip_label_permutation": None, "rationale": "left/right nerve"},
    )
    with pytest.raises(FlipTTAUnsafe, match="without remapping laterality-encoded classes"):
        tta_safety.assert_flip_tta_allowed("4", "flip3")
    # tta=none never mirrors, so it stays allowed.
    tta_safety.assert_flip_tta_allowed("4", "none")


def test_a_declared_permutation_re_enables_flip_tta(monkeypatch):
    import finetuning.fomo26_inference.tta_safety as tta_safety

    monkeypatch.setattr(
        tta_safety,
        "task_flip_policy",
        lambda task: {"flip_invariant_labels": False, "flip_label_permutation": [0, 2, 1], "rationale": "L/R"},
    )
    tta_safety.assert_flip_tta_allowed("4", "flip3")


def test_an_undeclared_task_cannot_silently_use_flip_tta():
    with pytest.raises(FlipTTAUnsafe, match="not declared"):
        assert_flip_tta_allowed("99", "flip3")


# --- ensemble space --------------------------------------------------------------------------


def test_ensemble_space_default_is_probability_space():
    from finetuning.fomo26_inference.seg_ensemble import SegFoldEnsemble

    assert "prob" in SegFoldEnsemble.__init__.__defaults__


def test_ensemble_space_rejects_an_unknown_value():
    from finetuning.fomo26_inference.seg_ensemble import SegFoldEnsemble

    with pytest.raises(ValueError, match="ensemble_space"):
        SegFoldEnsemble.__init__(object.__new__(SegFoldEnsemble), [], n_modalities=1, n_classes=2, ensemble_space="geometric")
