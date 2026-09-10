"""Deterministic metrics.json contract: schema, seg reduction, and aggregate_score interop."""

import json
from finetuning.fomo26_inference import aggregate_score as agg, metrics_io

# A completed v2 record must carry the provenance its status claims, so the serialization tests
# below build one that does. `fold_ensemble` is the honest role for a multi-fold evaluation: it
# has no single evaluated checkpoint, and the schema forbids inventing one.
ENSEMBLE_PROVENANCE = {
    "evaluated_checkpoint_role": "fold_ensemble",
    "evaluated_state": "fold_ensemble",
    "evaluated_fold_count": 5,
    "prediction_path": "/runs/predictions/p.json",
    "split_id": "test",
    "git_sha": "0123456789abcdef",
}


def test_schema_fields_and_version(tmp_path):
    record = metrics_io.build_record(
        task=2,
        fold=0,
        status="completed",
        metrics={"dsc": 0.7, "nsd": 0.6},
        task_name="SEG902_FOMO26_Task2_lesion",
        kind="seg",
        provenance=ENSEMBLE_PROVENANCE,
    )
    assert record["schema_version"] == metrics_io.SCHEMA_VERSION
    assert record["task"] == 2
    assert record["fold"] == 0
    assert record["status"] == "completed"
    assert record["metrics"] == {"dsc": 0.7, "nsd": 0.6}
    path = metrics_io.write_metrics_json(tmp_path / "task2" / "metrics.json", record)
    assert path.is_file()


def test_write_is_deterministic(tmp_path):
    record = metrics_io.build_record(
        task=3, fold="ensemble", status="completed", metrics={"mae": 4.2, "correlation": 0.8}, provenance=ENSEMBLE_PROVENANCE
    )
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    metrics_io.write_metrics_json(a, record)
    metrics_io.write_metrics_json(b, record)
    assert a.read_text() == b.read_text()  # sorted keys -> byte-identical


def test_task_number_from_name():
    assert metrics_io.task_number_from_name("SEG902_FOMO26_Task2_lesion") == 2
    assert metrics_io.task_number_from_name("CLS905_FOMO26_Task5_ppmr") == 5
    assert metrics_io.task_number_from_name("nonsense") is None


def test_reduce_seg_summary_averages_foreground_classes():
    # task 4 multiclass: classes 1 and 2
    summary = {1: {"dsc": 0.8, "nsd": 0.6, "n": 10}, 2: {"dsc": 0.4, "nsd": 0.2, "n": 10}}
    reduced = metrics_io.reduce_seg_summary(summary)
    assert reduced == {"dsc": 0.6000000000000001, "nsd": 0.4}
    full = metrics_io.seg_metrics_with_per_class(summary)
    assert "per_class" in full and set(full["per_class"]) == {"1", "2"}


def test_string_keyed_summary_supported():
    # JSON round-trips class ids as strings.
    summary = {"1": {"dsc": 0.5, "nsd": 0.5}}
    assert metrics_io.reduce_seg_summary(summary) == {"dsc": 0.5, "nsd": 0.5}


def test_metrics_json_feeds_aggregate_score(tmp_path):
    # An evaluator-produced metrics.json must be consumable by the aggregator unchanged.
    record = metrics_io.build_record(
        task=2,
        fold="ensemble",
        status="completed",
        metrics={"dsc": 0.8, "nsd": 0.6},
        kind="seg",
        provenance=ENSEMBLE_PROVENANCE,
    )
    metrics_io.write_metrics_json(tmp_path / "task2" / "metrics.json", record)
    report = agg.aggregate(tmp_path, missing_policy="exclude")
    task2 = next(e for e in report["per_task"] if e["task"] == 2)
    assert task2["valid"] is True
    assert task2["task_score"] == 0.7  # (0.8 + 0.6) / 2
    written = json.loads((tmp_path / "task2" / "metrics.json").read_text())
    assert written["schema_version"] == metrics_io.SCHEMA_VERSION
