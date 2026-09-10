"""The OOF evaluators must measure the inference policy they are asked for.

``oof_predictions`` used to accept ``tta`` and ignore it, running a bare ``module.model(x)``.
Records were therefore labelled with a policy they had not measured: a ``flip3`` run and a
``none`` run produced byte-identical metrics. These tests pin the policy actually reaching the
model, because an inference-policy comparison whose arms are secretly the same arm is worse
than no comparison at all.
"""

from __future__ import annotations

import pytest
import torch
from finetuning.fomo26_inference import clsreg_ensemble as ce, evaluate_clsreg as ec
from finetuning.fomo26_inference.seg_ensemble import SegFoldEnsemble

# ── a model whose output depends on orientation and on which crop it is shown ──


class _AsymmetricModule:
    """Flip- and crop-sensitive, so a policy that is silently dropped shows up as an equal number."""

    def __init__(self, kind, counter):
        self.kind = kind
        self.counter = counter

    def _apply_test_transforms(self, batch):
        return batch

    def model(self, x):
        self.counter["forwards"] += 1
        weight = torch.linspace(-1.0, 1.0, x[0, 0].numel()).reshape(x.shape[2:])
        # Mean, not sum: an unbounded score saturates the classification softmax, and a saturated
        # probability compares equal no matter which views produced it.
        score = (x[:, 0] * weight).mean(dim=(1, 2, 3))
        if self.kind == "cls":
            return torch.stack([torch.zeros_like(score), score], dim=1)
        return score.reshape(-1, 1)


@pytest.fixture
def oof_harness(monkeypatch, tmp_path):
    """Run the real ClsRegFoldEnsemble over one fold, counting forwards."""
    counter = {"forwards": 0}
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"")
    # Deliberately not a linear ramp: for a linear image the average over symmetrically-placed
    # cross-patch crops collapses back to the centre crop, which would make cross5 look inert.
    image = torch.rand((1, 1, 6, 6, 6), generator=torch.Generator().manual_seed(0))
    batch = {"image": image, "CLSREG_label": torch.tensor([1.0]), "file_path": ["s.pt"]}

    def run(kind, tta="none", cross_patch="none", target=(4, 4, 4)):
        counter["forwards"] = 0
        monkeypatch.setattr(ce, "build_clsreg_module", lambda *a, **k: (_AsymmetricModule(kind, counter), list(target)))
        monkeypatch.setattr(ec, "load_json", lambda path: [{"val": ["s.pt"], "train": []}])
        monkeypatch.setattr(ec, "get_data_path", lambda: str(tmp_path))
        monkeypatch.setattr(ec, "_loader", lambda files, target_size, **k: [batch for _ in files])
        record = {"fold": "0", "run_dir": str(tmp_path), "best_ckpt": str(ckpt)}
        raw, labels = ec.oof_predictions(
            [record],
            "split_kfold5_holdout70_15_15",
            "TASK",
            1,
            2,
            kind,
            "cpu",
            tta,
            0,
            cross_patch=cross_patch,
        )
        value = float(raw[0][1]) if kind == "cls" else float(raw[0])
        return value, counter["forwards"]

    return run


# ── the regression this defect was ──


@pytest.mark.parametrize("kind", ["cls", "reg"])
def test_flip_tta_reaches_the_model_in_oof(oof_harness, kind):
    """A flip7 OOF record must not be a none record wearing a flip7 label."""
    baseline, n_none = oof_harness(kind, tta="none")
    flipped, n_flip7 = oof_harness(kind, tta="flip7")

    assert n_none == 1, "tta=none is one view"
    assert n_flip7 == 7, "flip7 averages the 7 documented views"
    assert flipped != pytest.approx(baseline), (
        "flip7 produced the same number as none: TTA is being dropped before it reaches the model"
    )


@pytest.mark.parametrize("mode,views", [("cross5", 5), ("cross9", 9)])
def test_cross_patch_reaches_the_model_in_oof(oof_harness, mode, views):
    baseline, _ = oof_harness("reg", cross_patch="none")
    crossed, n_forwards = oof_harness("reg", cross_patch=mode)

    assert n_forwards == views
    assert crossed != pytest.approx(baseline)


def test_cross_patch_and_flip_tta_compose_multiplicatively(oof_harness):
    """The brief asks whether these can combine rather than being mutually exclusive. They can."""
    _, n_forwards = oof_harness("reg", tta="flip3", cross_patch="cross5")
    assert n_forwards == 20, "5 crop views x 4 flip views"


def test_a_zero_weight_fold_still_scores_its_own_partition(oof_harness, monkeypatch, tmp_path):
    """Weights describe how folds combine in an ensemble; in an OOF pool each fold stands alone."""
    counter = {"forwards": 0}
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"")
    batch = {"image": torch.ones(1, 1, 6, 6, 6), "CLSREG_label": torch.tensor([1.0]), "file_path": ["s.pt"]}
    monkeypatch.setattr(ce, "build_clsreg_module", lambda *a, **k: (_AsymmetricModule("reg", counter), [4, 4, 4]))
    monkeypatch.setattr(ec, "load_json", lambda path: [{"val": ["s.pt"], "train": []}])
    monkeypatch.setattr(ec, "get_data_path", lambda: str(tmp_path))
    monkeypatch.setattr(ec, "_loader", lambda files, target_size, **k: [batch for _ in files])

    record = {"fold": "0", "run_dir": str(tmp_path), "best_ckpt": str(ckpt), "ensemble_weight": 0.0}
    preds, labels = ec.oof_predictions([record], "split_kfold5_holdout70_15_15", "TASK", 1, 2, "reg", "cpu", "none", 0)
    assert len(preds) == 1, "a staged 0.0 weight must not delete the fold's own OOF prediction"


# ── the probability -> logit representation the record schema needs ──


@pytest.mark.parametrize("p", [0.5, 0.01, 0.937, 1e-9])
def test_pseudo_logits_reproduce_the_probability(p):
    probs = torch.tensor([1.0 - p, p])
    recovered = torch.softmax(ec._probs_to_logits(probs), 0)
    assert float(recovered[1]) == pytest.approx(p, abs=1e-6)


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.7])
def test_pseudo_logits_preserve_the_temperature_optimum(temperature):
    """log(softmax(z)) == z - logsumexp(z), and softmax ignores a shared additive shift.

    So temperature calibration fitted on these pseudo-logits finds the same optimum it would on
    the true logits. Without this, routing OOF through the ensemble would silently change what
    --calibrate reports.
    """
    true_logits = torch.tensor([-1.3, 2.4])
    pseudo = ec._probs_to_logits(torch.softmax(true_logits, 0))
    assert torch.allclose(torch.softmax(pseudo / temperature, 0), torch.softmax(true_logits / temperature, 0), atol=1e-6)


# ── the compute-neutral segmentation knob ──


def _seg_aggregate(logits_views, ensemble_space):
    """Run SegFoldEnsemble's own TTA aggregation for one member, without the model plumbing."""
    ens = object.__new__(SegFoldEnsemble)
    ens.ensemble_space = ensemble_space
    acc = None
    for logits in logits_views:
        view = logits.float() if ensemble_space == "logit" else torch.softmax(logits.float(), dim=1)
        acc = view if acc is None else acc + view
    out = acc / len(logits_views)
    return torch.softmax(out, dim=1) if ensemble_space == "logit" else out


def test_one_member_one_view_is_identical_in_both_spaces():
    """Why the P0 matrix skips `none x {prob,logit}`: there is nothing for the space to change."""
    logits = torch.randn(1, 3, 2, 2, 2)
    assert torch.allclose(_seg_aggregate([logits], "prob"), _seg_aggregate([logits], "logit"), atol=1e-6)


def test_multi_view_tta_genuinely_differs_between_the_two_spaces():
    """And why the arm is worth running once TTA is on: these are different estimators."""
    views = [torch.randn(1, 3, 2, 2, 2) for _ in range(7)]
    assert not torch.allclose(_seg_aggregate(views, "prob"), _seg_aggregate(views, "logit"), atol=1e-4)


def test_evaluate_seg_defaults_to_prob_and_forwards_the_requested_space(monkeypatch):
    """The historical numbers are prob-space; the flag must not silently move them."""
    import inspect
    from finetuning.fomo26_inference import evaluate_seg as es

    assert inspect.signature(es.evaluate).parameters["ensemble_space"].default == "prob"
    assert inspect.signature(es.out_of_fold_evaluate).parameters["ensemble_space"].default == "prob"

    seen = {}

    class _Ens:
        patch_size = [2, 2, 2]
        runtime_target_spacing = None

        def __init__(self, records, n_mod, n_cls, device="cuda", tta="none", ensemble_space="prob"):
            seen["ensemble_space"] = ensemble_space

    monkeypatch.setattr(es, "SegFoldEnsemble", _Ens)
    monkeypatch.setattr(es, "get_data_path", lambda: "/nonexistent")
    monkeypatch.setattr(es, "SegTestDataset", lambda files, transforms=None: [])
    monkeypatch.setattr(es, "CPU_seg_test_transforms", lambda **k: None)
    es.evaluate([], "TASK", None, 1, 1, "cpu", "none", 1.0, 0, test_files=[], ensemble_space="logit")
    assert seen["ensemble_space"] == "logit"


# ── the policy must be recoverable from the artifact ──


def test_the_oof_record_states_the_cross_patch_policy_it_measured():
    """Two records differing only in cross-patch must be distinguishable after the fact."""
    import inspect

    source = inspect.getsource(ec._run_oof)
    assert '"cross_patch": args.cross_patch' in source
    assert "cross_patch=args.cross_patch" in source
