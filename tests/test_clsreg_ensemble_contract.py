"""Wave 0A: characterisation of the cls/reg fold-ensemble aggregation contract.

Audit finding TEST-006 listed `finetuning/fomo26_inference/clsreg_ensemble.py` as untested. It
decides the numbers FOMO26 Tasks 1/3/4/5 submit: how fold predictions are combined, whether class
order survives the softmax, and whether TTA and cross-patch views are averaged deterministically.

`ClsRegFoldEnsemble.__init__` loads one checkpoint per fold from a run directory containing
`hydra/config.yaml`; reproducing that would mean writing several full finetune runs, which is out
of scope for a bounded CPU test. These tests therefore build the object with `object.__new__` and
populate exactly the attributes `__init__` would set, then exercise the **real**
`_one_module_output` and `predict_batch` implementations.

That is a deliberate line: the aggregation arithmetic under test is production code running on
real tensors. The stand-in fold modules are real `torch.nn.Module`s with known outputs -- not
mocks, and no test asserts "a mock was called". What is *not* covered here is checkpoint discovery
and module construction (`build_clsreg_module`); that gap is recorded in the Wave 0A report.

Bounded: CPU, tiny tensors, no checkpoints, no GPU, no network, no challenge data.
"""

from __future__ import annotations

import pytest
import torch
from finetuning.fomo26_inference import clsreg_ensemble as ens_mod
from finetuning.fomo26_inference.clsreg_ensemble import ClsRegFoldEnsemble

SPATIAL = (8, 8, 8)


class _ConstantLogits(torch.nn.Module):
    """A fold whose head always emits fixed logits, so fold averaging is exactly predictable."""

    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, x):  # noqa: D102
        return self.logits.unsqueeze(0).expand(x.shape[0], -1).clone()


class _MeanScalar(torch.nn.Module):
    """A fold that reports a scaled mean of its input, so TTA/crop averaging is observable."""

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = float(scale)

    def forward(self, x):  # noqa: D102
        return (x.flatten(1).mean(dim=1, keepdim=True) * self.scale).float()


class _FoldModule:
    """Stands in for a LightningModule: the ensemble only uses `.model` and the transform hook."""

    def __init__(self, model):
        self.model = model

    def _apply_test_transforms(self, batch):
        return batch


def _make_ensemble(models, *, kind, n_classes, tta="none", temperature=1.0, cross_patch="none"):
    ensemble = object.__new__(ClsRegFoldEnsemble)
    ensemble.device = "cpu"
    ensemble.kind = kind
    ensemble.n_classes = n_classes
    ensemble.temperature = float(temperature)
    ensemble.tta_flips = ens_mod._TTA_FLIPS[tta]
    ensemble.cross_patch = cross_patch
    ensemble.modules = [_FoldModule(m) for m in models]
    ensemble.target_size = list(SPATIAL)
    return ensemble


def _batch(batch_size=1, value=None):
    image = torch.full((batch_size, 1, *SPATIAL), float(value)) if value is not None else torch.randn(batch_size, 1, *SPATIAL)
    return {
        "image": image,
        "file_path": [f"case_{i:04d}" for i in range(batch_size)],
        "CLSREG_label": torch.arange(batch_size),
    }


# --------------------------------------------------------------------------------------------
# Classification: fold averaging and class-order preservation
# --------------------------------------------------------------------------------------------


def test_cls_probs_are_the_mean_of_per_fold_softmax():
    """The documented contract is 'averaged softmax', not 'softmax of averaged logits'."""
    logits_a, logits_b = [2.0, 0.0, -1.0], [-1.0, 1.0, 0.5]
    ensemble = _make_ensemble([_ConstantLogits(logits_a), _ConstantLogits(logits_b)], kind="cls", n_classes=3)

    probs = ensemble.predict_batch(_batch())["probs"]

    expected = (torch.softmax(torch.tensor(logits_a), dim=0) + torch.softmax(torch.tensor(logits_b), dim=0)) / 2
    assert torch.allclose(probs[0], expected, atol=1e-6)

    # Guard against the wrong-but-plausible implementation.
    logit_mean = torch.softmax((torch.tensor(logits_a) + torch.tensor(logits_b)) / 2, dim=0)
    assert not torch.allclose(probs[0], logit_mean, atol=1e-6)


def test_cls_probs_honor_frozen_validation_weights():
    logits_a, logits_b = [3.0, 0.0], [0.0, 2.0]
    ensemble = _make_ensemble([_ConstantLogits(logits_a), _ConstantLogits(logits_b)], kind="cls", n_classes=2)
    ensemble.ensemble_weights = [0.75, 0.25]

    probabilities = ensemble.predict_batch(_batch())["probs"]

    expected = 0.75 * torch.softmax(torch.tensor(logits_a), dim=0) + 0.25 * torch.softmax(torch.tensor(logits_b), dim=0)
    assert torch.allclose(probabilities[0], expected, atol=1e-6)


class _Poisoned(torch.nn.Module):
    """A staged-but-unused fold whose output would destroy any sum it entered."""

    def forward(self, x):  # noqa: D102
        return torch.full((x.shape[0], 2), float("nan"))


def test_cls_zero_weight_folds_are_staged_but_contribute_nothing():
    """Task-9 ships every trained fold; best_single gives the rest weight zero.

    Zero has to mean "not evaluated", not "multiplied by zero" -- otherwise a single unused fold
    emitting a non-finite value would silently poison the shipped prediction.
    """
    logits = [3.0, 0.0]
    ensemble = _make_ensemble([_ConstantLogits(logits), _Poisoned()], kind="cls", n_classes=2)
    ensemble.ensemble_weights = [1.0, 0.0]

    probabilities = ensemble.predict_batch(_batch())["probs"]

    assert torch.isfinite(probabilities).all(), "a zero-weight fold must not reach the accumulator"
    assert torch.allclose(probabilities[0], torch.softmax(torch.tensor(logits), dim=0), atol=1e-6)


def test_cls_weights_stay_bound_to_their_fold_order():
    """Weight i must apply to fold i; swapping the weights must change the answer."""
    logits_a, logits_b = [3.0, 0.0], [0.0, 2.0]
    models = [_ConstantLogits(logits_a), _ConstantLogits(logits_b)]
    forward = _make_ensemble(models, kind="cls", n_classes=2)
    forward.ensemble_weights = [0.75, 0.25]
    reversed_ = _make_ensemble(models, kind="cls", n_classes=2)
    reversed_.ensemble_weights = [0.25, 0.75]

    assert not torch.allclose(
        forward.predict_batch(_batch())["probs"][0],
        reversed_.predict_batch(_batch())["probs"][0],
        atol=1e-6,
    )


def test_cls_probs_are_a_distribution_over_the_declared_classes():
    ensemble = _make_ensemble([_ConstantLogits([1.0, 2.0, 3.0, 0.5])], kind="cls", n_classes=4)
    probs = ensemble.predict_batch(_batch(batch_size=2))["probs"]

    assert probs.shape == (2, 4), "probs must be [B, C]"
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)
    assert (probs >= 0).all()


def test_cls_class_order_is_preserved_not_sorted():
    """Class index identity must survive: argmax has to point at the highest *input* logit."""
    logits = [0.1, 5.0, 0.2, 0.3]
    ensemble = _make_ensemble([_ConstantLogits(logits)], kind="cls", n_classes=4)

    probs = ensemble.predict_batch(_batch())["probs"][0]

    assert int(probs.argmax()) == 1, "the ensemble reordered or re-ranked the class axis"
    # Softmax is strictly monotonic, so the full ranking must match the input logits exactly.
    assert torch.equal(probs.argsort(descending=True), torch.tensor(logits).argsort(descending=True))


def test_cls_temperature_flattens_the_distribution_without_changing_the_ranking():
    logits = [3.0, 1.0, 0.0]
    sharp = _make_ensemble([_ConstantLogits(logits)], kind="cls", n_classes=3, temperature=1.0)
    soft = _make_ensemble([_ConstantLogits(logits)], kind="cls", n_classes=3, temperature=10.0)

    p_sharp = sharp.predict_batch(_batch())["probs"][0]
    p_soft = soft.predict_batch(_batch())["probs"][0]

    assert int(p_sharp.argmax()) == int(p_soft.argmax()) == 0
    assert p_soft.max() < p_sharp.max(), "temperature > 1 must flatten the distribution"


# --------------------------------------------------------------------------------------------
# Regression: fold averaging and output shape
# --------------------------------------------------------------------------------------------


def test_reg_prediction_is_the_mean_over_folds():
    ensemble = _make_ensemble([_MeanScalar(scale=1.0), _MeanScalar(scale=3.0)], kind="reg", n_classes=1)
    batch = _batch(value=2.0)

    pred = ensemble.predict_batch(batch)["pred"]

    # each fold sees mean(input)=2.0 -> outputs 2.0 and 6.0 -> ensemble 4.0
    assert pred.shape == (1,), "reg predictions must be squeezed to [B]"
    assert torch.allclose(pred, torch.tensor([4.0]), atol=1e-6)


def test_reg_prediction_is_not_softmaxed():
    """The reg branch must pass raw outputs through; a stray softmax would clamp to 1.0."""
    ensemble = _make_ensemble([_MeanScalar(scale=1.0)], kind="reg", n_classes=1)
    pred = ensemble.predict_batch(_batch(value=7.5))["pred"]
    assert torch.allclose(pred, torch.tensor([7.5]), atol=1e-6)


# --------------------------------------------------------------------------------------------
# TTA / cross-patch view averaging and determinism
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tta,expected_views", [("none", 1), ("flip3", 4), ("flip7", 7)])
def test_tta_modes_average_the_documented_number_of_views(tta, expected_views):
    """`_TTA_FLIPS` sizes are part of the contract: flip3 -> 4 views (identity + 3 flips)."""
    assert len(ens_mod._TTA_FLIPS[tta]) == expected_views

    calls = []

    class _Counting(torch.nn.Module):
        def forward(self, x):
            calls.append(x.clone())
            return torch.zeros(x.shape[0], 1)

    ensemble = _make_ensemble([_Counting()], kind="reg", n_classes=1, tta=tta)
    ensemble.predict_batch(_batch())
    assert len(calls) == expected_views


def test_flip_tta_averages_over_flipped_inputs_and_stays_symmetric():
    """A flip-invariant input must give the same answer under every TTA mode."""
    uniform = _batch(value=1.0)
    results = {
        tta: _make_ensemble([_MeanScalar()], kind="reg", n_classes=1, tta=tta).predict_batch(uniform)["pred"]
        for tta in ("none", "flip3", "flip7")
    }
    assert torch.allclose(results["none"], results["flip3"], atol=1e-6)
    assert torch.allclose(results["none"], results["flip7"], atol=1e-6)


def test_prediction_is_deterministic_across_repeated_calls():
    ensemble = _make_ensemble(
        [_ConstantLogits([1.0, 2.0]), _ConstantLogits([0.0, 0.5])],
        kind="cls",
        n_classes=2,
        tta="flip7",
    )
    batch = _batch(batch_size=3)

    first = ensemble.predict_batch(batch)["probs"]
    second = ensemble.predict_batch(batch)["probs"]

    assert torch.equal(first, second), "ensemble inference is not deterministic"


def test_fold_order_does_not_change_the_ensemble_result():
    """Averaging must be order-independent, or fold enumeration order would leak into results."""
    a, b, c = _ConstantLogits([1.0, 2.0]), _ConstantLogits([0.0, 0.5]), _ConstantLogits([3.0, -1.0])
    forward = _make_ensemble([a, b, c], kind="cls", n_classes=2).predict_batch(_batch())["probs"]
    reverse = _make_ensemble([c, b, a], kind="cls", n_classes=2).predict_batch(_batch())["probs"]
    assert torch.allclose(forward, reverse, atol=1e-6)


# --------------------------------------------------------------------------------------------
# Identity propagation and fail-closed configuration
# --------------------------------------------------------------------------------------------


def test_case_identity_and_labels_are_propagated_unmodified():
    """Row identity must survive inference or predictions get attributed to the wrong case."""
    ensemble = _make_ensemble([_ConstantLogits([1.0, 0.0])], kind="cls", n_classes=2)
    batch = _batch(batch_size=3)

    result = ensemble.predict_batch(batch)

    assert result["file_path"] == batch["file_path"], "case identity was reordered or dropped"
    assert torch.equal(result["label"], batch["CLSREG_label"])
    assert result["probs"].shape[0] == len(batch["file_path"]), "row count changed during inference"


def test_unsupported_cross_patch_mode_is_rejected_by_the_constructor():
    """Fail-closed on an unknown view policy rather than silently using `none`."""
    with pytest.raises(ValueError, match="Unsupported cross_patch"):
        ClsRegFoldEnsemble(
            manifest_records=[],
            n_modalities=1,
            n_classes=2,
            kind="cls",
            device="cpu",
            cross_patch="cross13",
        )


def test_missing_fold_checkpoint_fails_closed(tmp_path):
    """A manifest naming a checkpoint that does not exist must raise, not silently drop the fold.

    This is the one `__init__` path reachable without a full run directory, and it is the one that
    matters: silently ensembling 4 of 5 folds would change the submitted numbers.
    """
    with pytest.raises(FileNotFoundError, match="Missing checkpoint"):
        ClsRegFoldEnsemble(
            manifest_records=[{"run_dir": str(tmp_path), "best_ckpt": str(tmp_path / "absent.ckpt"), "fold": 0}],
            n_modalities=1,
            n_classes=2,
            kind="cls",
            device="cpu",
        )
