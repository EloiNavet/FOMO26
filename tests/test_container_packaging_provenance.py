"""A submission must be able to prove which weights it shipped.

``build_container.py`` previously wrote ``models/manifest.json`` with no digests at all, and the two
metadata files ``finetuning/container/container_contract.py`` consume
(``docker_metadata.json``, ``prediction_metadata.json``) had no producer anywhere in the repository
-- so the submission validator could never be run against a real build.
"""

import json
import pytest
import torch
from finetuning.container.build_container import main as build_main
from pathlib import Path


def _fold_run(root: Path, fold: int, weight: float) -> Path:
    run_dir = root / f"run__task2__fold{fold}"
    (run_dir / "hydra").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "hydra" / "config.yaml").write_text("task: SEG902\n")
    torch.save({"state_dict": {"w": torch.zeros(2) + weight}}, run_dir / "checkpoints" / "best.ckpt")
    return run_dir


def _stage(tmp_path, records, monkeypatch, task="2"):
    manifest = tmp_path / "fold_manifest.json"
    manifest.write_text(json.dumps(records))
    out = tmp_path / "ctx"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_container",
            "--task",
            task,
            "--manifest",
            str(manifest),
            "--out",
            str(out),
            "--link-repo",
        ],
    )
    build_main()
    return out


def test_model_manifest_carries_a_sha256_per_fold(tmp_path, monkeypatch):
    records = []
    for fold in range(3):
        run_dir = _fold_run(tmp_path, fold, weight=fold)
        records.append(
            {
                "returncode": 0,
                "run_dir": str(run_dir),
                "fold": fold,
                "pretrained_checkpoint_sha256": "a" * 64,
            }
        )
    out = _stage(tmp_path, records, monkeypatch)

    manifest = json.loads((out / "models" / "model_manifest.json").read_text())
    assert manifest["schema_version"] == "fomo26-model-manifest-v1"
    assert manifest["fold_count"] == 3
    assert manifest["pretrained_checkpoint_sha256"] == "a" * 64
    digests = [entry["sha256"] for entry in manifest["models"]]
    assert len(set(digests)) == 3, "distinct fold weights must yield distinct digests"
    assert all(len(digest) == 64 for digest in digests)


def test_digests_describe_the_bytes_inside_the_container(tmp_path, monkeypatch):
    run_dir = _fold_run(tmp_path, 0, weight=1.0)
    records = [{"returncode": 0, "run_dir": str(run_dir), "fold": 0}]
    out = _stage(tmp_path, records, monkeypatch)

    manifest = json.loads((out / "models" / "model_manifest.json").read_text())
    entry = manifest["models"][0]
    staged = out / "models" / "runs" / "fold0" / "checkpoints" / "best.ckpt"
    assert entry["path"] == "/app/models/runs/fold0/checkpoints/best.ckpt"

    import hashlib

    assert entry["sha256"] == hashlib.sha256(staged.read_bytes()).hexdigest()


def test_staging_refuses_weights_that_contradict_their_provenance_record(tmp_path, monkeypatch):
    """A fold manifest digest that does not match the copied bytes is a corrupted lineage."""
    run_dir = _fold_run(tmp_path, 0, weight=1.0)
    records = [
        {
            "returncode": 0,
            "run_dir": str(run_dir),
            "fold": 0,
            "fold_checkpoint_sha256": "f" * 64,  # deliberately wrong
        }
    ]
    with pytest.raises(SystemExit, match="do not match their provenance record"):
        _stage(tmp_path, records, monkeypatch)


def test_docker_metadata_declares_the_offline_contract(tmp_path, monkeypatch):
    run_dir = _fold_run(tmp_path, 0, weight=1.0)
    records = [{"returncode": 0, "run_dir": str(run_dir), "fold": 0}]
    out = _stage(tmp_path, records, monkeypatch)

    metadata = json.loads((out / "docker_metadata.json").read_text())
    assert metadata["network_required"] is False
    assert metadata["wandb_required"] is False
    assert metadata["entrypoints"] == ["2"]
    # build_executed stays false: staging a context is not building an image.
    assert metadata["build_executed"] is False


def test_mixed_backbones_are_recorded_as_a_conflict_not_silently_collapsed(tmp_path, monkeypatch):
    records = []
    for fold, sha in enumerate(("a" * 64, "b" * 64)):
        run_dir = _fold_run(tmp_path, fold, weight=fold)
        records.append({"returncode": 0, "run_dir": str(run_dir), "fold": fold, "pretrained_checkpoint_sha256": sha})
    out = _stage(tmp_path, records, monkeypatch)

    manifest = json.loads((out / "models" / "model_manifest.json").read_text())
    assert manifest["pretrained_checkpoint_sha256"] is None
    assert manifest["pretrained_checkpoint_sha256_conflict"] == ["a" * 64, "b" * 64]
