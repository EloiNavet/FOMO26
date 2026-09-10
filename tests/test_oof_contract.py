"""Out-of-fold evaluation must be an OOF set, and must be scored the way everything else is.

The five Candidate-O folds were all evaluated on the SAME TEST_70_15_15 holdout, so those numbers
are common-holdout numbers. A legitimate OOF pool scores each fold only on the cases that fold
never trained on; it must never be conflated with the holdout, and it must not introduce a second
metric definition, or an OOF number and a holdout number stop being comparable.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from finetuning.fomo26_inference.evaluate_clsreg import oof_record


def _cases(paths, labels, folds):
    return [{"path": p, "label": lab, "fold": f, "checkpoint": "x"} for p, lab, f in zip(paths, labels, folds)]


def test_classification_oof_uses_the_canonical_metric_definitions():
    from finetuning.fomo26_inference.prediction_metrics import classification_record

    logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0], [0.5, 0.4], [-2.0, 1.0]])
    cases = _cases(["a/scan.pt", "b/scan.pt", "c/scan.pt", "d/scan.pt"], [0, 1, 0, 1], [0, 1, 2, 3])
    metrics, subjects = oof_record("cls", logits, torch.tensor([0, 1, 0, 1]), cases)

    payload = {
        case["path"]: {
            "label": case["label"],
            "logits": [float(v) for v in logits[i]],
            "prediction": int(float(torch.softmax(logits[i], 0)[1]) >= 0.5),
        }
        for i, case in enumerate(cases)
    }
    expected, _ = classification_record(payload)
    assert metrics == expected, "OOF must reuse the repository's metric, not define a parallel one"
    assert set(metrics) == {"auroc", "f1"}
    assert [s["fold"] for s in subjects] == [0, 1, 2, 3], "each OOF case records the fold that produced it"


def test_regression_oof_uses_the_canonical_metric_definitions():
    from finetuning.fomo26_inference.prediction_metrics import regression_record

    preds = np.array([10.0, 20.0, 30.0, 41.0])
    cases = _cases(["a/s.pt", "b/s.pt", "c/s.pt", "d/s.pt"], [12.0, 19.0, 33.0, 40.0], [0, 1, 2, 3])
    metrics, subjects = oof_record("reg", preds, np.array([12.0, 19.0, 33.0, 40.0]), cases)

    expected, _ = regression_record(
        {c["path"]: {"label": c["label"], "prediction": float(preds[i])} for i, c in enumerate(cases)}
    )
    assert metrics == expected
    assert set(metrics) == {"mae", "correlation"}
    # MAE keeps its direction: lower is better. Nothing here may flip it.
    assert metrics["mae"] == pytest.approx(np.mean(np.abs(preds - np.array([12.0, 19.0, 33.0, 40.0]))))
    assert len(subjects) == 4


def test_an_oof_pool_with_a_repeated_case_is_not_an_oof_pool(monkeypatch):
    """If two folds both score a subject, the validation partitions were not disjoint."""
    from finetuning.fomo26_inference import evaluate_clsreg as ec

    class Args:
        train_split = "split_kfold5_holdout70_15_15"
        task = "CLS901"
        kind = "cls"
        device = "cpu"
        tta = "none"
        cross_patch = "none"
        num_workers = 0
        output_json = None

    duplicated = _cases(["a/s.pt", "a/s.pt"], [0, 1], [0, 1])
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    monkeypatch.setattr(ec, "oof_predictions", lambda *a, **k: (logits, torch.tensor([0, 1]), duplicated))
    with pytest.raises(SystemExit, match="not an out-of-fold set"):
        ec._run_oof(Args(), 1, 2, [{"fold": 0}, {"fold": 1}])


def test_an_oof_record_is_never_labelled_as_the_holdout_split():
    """The two must stay distinguishable in the artifacts, or they will be compared as if equal."""
    import inspect

    source = inspect.getsource(__import__("finetuning.fomo26_inference.evaluate_clsreg", fromlist=["x"])._run_oof)
    assert '"split_id": f"{args.train_split}:val"' in source
    assert "TEST" not in source.split("split_id")[1][:80]
    assert 'fold="oof"' in source or 'fold="oof"' in source


def test_a_string_fold_index_still_selects_the_right_validation_partition(monkeypatch, tmp_path):
    """The Slurm rail records `fold` as a string; the split file is a list indexed by fold."""
    import numpy as np
    import torch
    from finetuning.fomo26_inference import evaluate_clsreg as ec

    folds = [{"val": [f"fold{i}_case.pt"], "train": []} for i in range(5)]
    monkeypatch.setattr(ec, "load_json", lambda path: folds)
    monkeypatch.setattr(ec, "get_data_path", lambda: str(tmp_path))

    class _Ensemble:
        """Stands in for ClsRegFoldEnsemble, which is now the seam OOF inference runs through."""

        target_size = (2, 2, 2)

        def __init__(self, *a, **k):
            pass

        def predict_batch(self, batch):
            return {"pred": torch.tensor([7.0]), "label": torch.tensor([1.0])}

    monkeypatch.setattr(ec, "ClsRegFoldEnsemble", _Ensemble)
    monkeypatch.setattr(ec, "_loader", lambda files, target, **k: [object() for _ in files])

    preds, labels, cases = ec.oof_predictions(
        [{"fold": "3", "run_dir": "r", "best_ckpt": "c"}],
        "split_kfold5_holdout70_15_15",
        "TASK",
        1,
        1,
        "reg",
        "cpu",
        "none",
        0,
        collect_cases=True,
    )
    assert [c["path"] for c in cases] == ["fold3_case.pt"], "a string fold must index the same partition as an int"
    assert np.allclose(preds, [7.0]) and list(labels) == [1.0]


# ── the metrics schema must be able to describe an out-of-fold result, and only honestly ──


def _oof_record(**provenance_overrides):
    from finetuning.fomo26_inference.metrics_io import build_record

    provenance = {
        "evaluated_checkpoint_role": "per_fold_out_of_fold",
        "evaluated_state": "per_fold_out_of_fold",
        "evaluated_fold_count": 5,
        "split_id": "split_kfold5_holdout70_15_15:val",
        "git_sha": "abc123",
        "prediction_path": "/x/task1_oof.json",
    }
    provenance.update(provenance_overrides)
    return build_record(
        task=1,
        fold="oof",
        status="completed",
        metrics={"auroc": 0.36, "f1": 0.5},
        task_name="CLS901_FOMO26_Task1_presence",
        kind="cls",
        provenance=provenance,
    )


def test_an_out_of_fold_record_is_a_valid_completed_record():
    from finetuning.fomo26_inference.metrics_io import validate_record

    assert validate_record(_oof_record())["fold"] == "oof"


def test_an_out_of_fold_record_may_not_claim_one_checkpoint():
    """Each case came from a different fold's weights; a single digest would fabricate an identity."""
    from finetuning.fomo26_inference.metrics_io import provenance_problems

    problems = provenance_problems(_oof_record(evaluated_checkpoint_sha256="deadbeef"))
    assert any("fabricate a checkpoint identity" in p for p in problems)

    assert any("positive int" in p for p in provenance_problems(_oof_record(evaluated_fold_count=None)))
    assert any("evaluated_state" in p for p in provenance_problems(_oof_record(evaluated_state="checkpoint_file")))


def test_an_out_of_fold_record_may_not_be_labelled_with_the_holdout_split():
    """The whole point of OOF is that it is not the holdout; the record must not blur that."""
    from finetuning.fomo26_inference.metrics_io import provenance_problems

    problems = provenance_problems(_oof_record(split_id="TEST_70_15_15"))
    assert any("held-out TEST split" in p for p in problems)


def test_the_ensemble_role_still_means_an_ensemble():
    """OOF got its own role rather than borrowing fold_ensemble, which averages members."""
    from finetuning.fomo26_inference.metrics_io import ENSEMBLE_ROLES, OUT_OF_FOLD_ROLES

    assert "per_fold_out_of_fold" not in ENSEMBLE_ROLES
    assert "fold_ensemble" not in OUT_OF_FOLD_ROLES
