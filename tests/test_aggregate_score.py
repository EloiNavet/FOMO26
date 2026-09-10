"""Offline FOMO26 score aggregator: normalization direction, weighting, and robustness."""

import json
import math
import pytest
from finetuning.fomo26_inference import aggregate_score as agg


def _write_task(root, task, obj, name="metrics.json"):
    task_dir = root / f"task{task}"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / name).write_text(json.dumps(obj))


def test_weights_match_challenge_and_sum_to_one():
    assert agg.TASK_WEIGHTS == {1: 0.10, 2: 0.25, 3: 0.10, 4: 0.25, 5: 0.10, 6: 0.10, 7: 0.10}
    assert math.isclose(sum(agg.TASK_WEIGHTS.values()), 1.0, abs_tol=1e-9)


def test_normalize_direction_higher_and_lower_is_better():
    # higher-is-better metrics pass through in [0, 1]
    assert agg.normalize_metric("dsc", 0.75) == pytest.approx(0.75)
    assert agg.normalize_metric("auroc", 1.0) == pytest.approx(1.0)
    # lower-is-better MAE: 0 -> 1.0 (perfect), 100 -> 0.0 (worst), 25 -> 0.75
    assert agg.normalize_metric("mae", 0.0) == pytest.approx(1.0)
    assert agg.normalize_metric("mae", 100.0) == pytest.approx(0.0)
    assert agg.normalize_metric("mae", 25.0) == pytest.approx(0.75)
    # correlation in [-1, 1] -> [0, 1]
    assert agg.normalize_metric("correlation", 1.0) == pytest.approx(1.0)
    assert agg.normalize_metric("correlation", 0.0) == pytest.approx(0.5)
    # out-of-range values clamp
    assert agg.normalize_metric("mae", 200.0) == pytest.approx(0.0)
    assert agg.normalize_metric("dsc", 1.5) == pytest.approx(1.0)


def test_complete_valid_results_penalize_policy(tmp_path):
    # A perfect run on all 7 tasks should score 1.0 with the full fixed weights.
    _write_task(tmp_path, 1, {"kind": "cls", "metrics": {"auroc": 1.0, "f1": 1.0}})
    _write_task(tmp_path, 2, {"kind": "seg", "metrics": {"dsc": 1.0, "nsd": 1.0}})
    _write_task(tmp_path, 3, {"kind": "reg", "metrics": {"mae": 0.0, "correlation": 1.0}})
    _write_task(tmp_path, 4, {"kind": "seg", "metrics": {"dsc": 1.0, "nsd": 1.0}})
    _write_task(tmp_path, 5, {"kind": "cls", "metrics": {"auroc": 1.0, "f1": 1.0}})
    _write_task(tmp_path, 6, {"kind": "cls", "metrics": {"auroc": 1.0, "f1": 1.0}})
    _write_task(tmp_path, 7, {"kind": "fairness", "metrics": {"fairness_score": 1.0}})

    report = agg.aggregate(tmp_path, missing_policy="penalize")
    assert report["offline_proxy_score"] == pytest.approx(1.0)
    assert report["missing_tasks"] == []
    assert report["invalid_tasks"] == []
    assert report["all_tasks_present_and_valid"] is True
    # official score is never claimed offline
    assert report["official_compatible_score"] is None
    assert "permutation" in report["official_compatible_reason"]


def test_multi_metric_segmentation_task_is_averaged(tmp_path):
    _write_task(tmp_path, 2, {"kind": "seg", "metrics": {"dsc": 0.8, "nsd": 0.6}})
    report = agg.aggregate(tmp_path, missing_policy="exclude")
    task2 = next(e for e in report["per_task"] if e["task"] == 2)
    assert task2["task_score"] == pytest.approx(0.7)  # (0.8 + 0.6) / 2


def test_missing_task_excluded_and_renormalized(tmp_path):
    # Only tasks 2 and 4 present (weights 0.25 each). Exclude policy renormalizes to 0.5 each.
    _write_task(tmp_path, 2, {"kind": "seg", "metrics": {"dsc": 0.8, "nsd": 0.8}})
    _write_task(tmp_path, 4, {"kind": "seg", "metrics": {"dsc": 0.4, "nsd": 0.4}})
    report = agg.aggregate(tmp_path, missing_policy="exclude")
    assert set(report["missing_tasks"]) == {1, 3, 5, 6, 7}
    # (0.5 * 0.8) + (0.5 * 0.4) = 0.6
    assert report["offline_proxy_score"] == pytest.approx(0.6)


def test_missing_task_penalized_scores_zero(tmp_path):
    _write_task(tmp_path, 2, {"kind": "seg", "metrics": {"dsc": 1.0, "nsd": 1.0}})
    report = agg.aggregate(tmp_path, missing_policy="penalize")
    # Only task 2 contributes its 0.25 weight; everything else missing -> 0.
    assert report["offline_proxy_score"] == pytest.approx(0.25)
    assert set(report["missing_tasks"]) == {1, 3, 4, 5, 6, 7}


def test_nan_metric_is_worst_case(tmp_path):
    _write_task(tmp_path, 2, {"kind": "seg", "metrics": {"dsc": float("nan"), "nsd": 0.5}})
    report = agg.aggregate(tmp_path, missing_policy="exclude")
    task2 = next(e for e in report["per_task"] if e["task"] == 2)
    # dsc NaN -> 0, nsd 0.5 -> mean 0.25
    assert task2["task_score"] == pytest.approx(0.25)
    assert any("NaN" in w for w in report["warnings"])


def test_failed_task_gets_zero(tmp_path):
    _write_task(tmp_path, 2, {"kind": "seg", "status": "failed", "metrics": {}})
    report = agg.aggregate(tmp_path, missing_policy="exclude")
    task2 = next(e for e in report["per_task"] if e["task"] == 2)
    assert task2["status"] == "failed"
    assert task2["valid"] is True
    assert task2["task_score"] == pytest.approx(0.0)


def test_malformed_result_file_is_reported_not_crashing(tmp_path):
    task_dir = tmp_path / "task2"
    task_dir.mkdir(parents=True)
    (task_dir / "metrics.json").write_text("{not valid json")
    report = agg.aggregate(tmp_path, missing_policy="exclude")
    task2 = next(e for e in report["per_task"] if e["task"] == 2)
    # Treated as failed -> worst case, and an explicit warning surfaces the file.
    assert task2["status"] == "failed"
    assert any("malformed" in w for w in report["warnings"])


def test_external_metrics_inject_tasks_6_and_7(tmp_path):
    _write_task(tmp_path, 2, {"kind": "seg", "metrics": {"dsc": 1.0, "nsd": 1.0}})
    external_path = tmp_path / "external.json"
    external_path.write_text(json.dumps({"6": {"metrics": {"auroc": 0.9}}, "task7": {"metrics": {"fairness_score": 0.8}}}))
    external = agg.load_external_metrics(external_path)
    report = agg.aggregate(tmp_path, external=external, missing_policy="exclude")
    present = {e["task"] for e in report["per_task"] if e["valid"]}
    assert {2, 6, 7}.issubset(present)
    assert 6 not in report["missing_tasks"]
    assert 7 not in report["missing_tasks"]


def test_flat_metric_form_supported(tmp_path):
    # metrics given as top-level keys without a "metrics" wrapper
    _write_task(tmp_path, 3, {"kind": "reg", "mae": 25.0, "correlation": 1.0})
    report = agg.aggregate(tmp_path, missing_policy="exclude")
    task3 = next(e for e in report["per_task"] if e["task"] == 3)
    # mae 25 -> 0.75, corr 1.0 -> 1.0, mean 0.875
    assert task3["task_score"] == pytest.approx(0.875)


def test_csv_written(tmp_path):
    _write_task(tmp_path, 2, {"kind": "seg", "metrics": {"dsc": 0.8, "nsd": 0.6}})
    report = agg.aggregate(tmp_path, missing_policy="exclude")
    csv_path = tmp_path / "score.csv"
    agg.write_csv(report, csv_path)
    text = csv_path.read_text()
    assert "offline_proxy_score" in text
    assert "task,kind,status" in text.splitlines()[0]
