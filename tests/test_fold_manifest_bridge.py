"""The Slurm rail -> container rail bridge, and the provenance it must refuse to lose.

Before this bridge existed the only producer of a fold manifest was the workstation orchestrator,
which records no checkpoint digest and validates no provenance -- so the only path to a submittable
container bypassed every guarantee the Slurm rail provides.
"""

import json
import pytest
import torch
from finetuning.container.fomo_ensemble_predict import resolve_manifest_records
from finetuning.fomo26_inference.build_fold_manifest import (
    FoldManifestError,
    build_fold_manifest,
    discover_run_dirs,
    write_fold_manifest,
)
from pathlib import Path

BACKBONE_SHA = "a" * 64
OTHER_SHA = "b" * 64


def _make_run_dir(root: Path, task: int, fold: int, *, backbone_sha=BACKBONE_SHA, status="completed", run_id="r1"):
    run_dir = root / f"SEG90{task}_FOMO26_Task{task}_lesion" / f"run_{run_id}__task{task}__fold{fold}"
    (run_dir / "checkpoints").mkdir(parents=True)
    best = run_dir / "checkpoints" / "best.ckpt"
    torch.save({"state_dict": {"w": torch.zeros(2) + fold}}, best)
    torch.save({"state_dict": {}}, run_dir / "checkpoints" / "last.ckpt")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": "fomo26-run-manifest-v1",
                "run_id": run_id,
                "task": f"SEG90{task}_FOMO26_Task{task}_lesion",
                "fomo_task": task,
                "fold": fold,
                "status": status,
                "best_checkpoint": str(best),
                "pretrained_checkpoint": "/scratch/backbone.ckpt",
                "pretrained_checkpoint_sha256": backbone_sha,
                "git_commit": "deadbeef",
                "seed": 431027 + task * 100 + fold,
                "slurm_job_id": f"90{task}{fold}",
            }
        )
    )
    return run_dir


def test_bridge_output_is_consumable_by_the_container_rail(tmp_path):
    """The whole point: what this writes must load in resolve_manifest_records unchanged."""
    models = tmp_path / "models"
    for fold in range(5):
        _make_run_dir(models, 2, fold)

    manifest = build_fold_manifest(discover_run_dirs(models, 2), expected_folds=5)
    output = tmp_path / "fold_manifest.json"
    write_fold_manifest(manifest, output)

    records = resolve_manifest_records(str(output))
    assert len(records) == 5
    assert all(Path(record["best_ckpt"]).is_file() for record in records)
    assert sorted(record["fold"] for record in records) == [0, 1, 2, 3, 4]


def test_bridge_records_a_real_digest_for_every_fold_checkpoint(tmp_path):
    models = tmp_path / "models"
    for fold in range(3):
        _make_run_dir(models, 4, fold)

    manifest = build_fold_manifest(discover_run_dirs(models, 4))
    digests = {record["fold_checkpoint_sha256"] for record in manifest["records"]}
    assert len(digests) == 3, "distinct fold weights must produce distinct digests"
    assert all(len(digest) == 64 for digest in digests)
    assert manifest["pretrained_checkpoint_sha256"] == BACKBONE_SHA


def test_bridge_honours_an_explicit_non_best_checkpoint_name(tmp_path):
    models = tmp_path / "models"
    run_dir = _make_run_dir(models, 4, 0)
    manifest = build_fold_manifest([run_dir], checkpoint_name="last")
    assert Path(manifest["records"][0]["best_ckpt"]).name == "last.ckpt"


def test_bridge_refuses_folds_from_different_pretrained_checkpoints(tmp_path):
    """Ensembling across backbones silently changes what the submission is."""
    models = tmp_path / "models"
    _make_run_dir(models, 1, 0)
    _make_run_dir(models, 1, 1, backbone_sha=OTHER_SHA)

    with pytest.raises(FoldManifestError, match="different pretrained checkpoints"):
        build_fold_manifest(discover_run_dirs(models, 1))


def test_bridge_refuses_a_failed_fold_instead_of_silently_shrinking_the_ensemble(tmp_path):
    models = tmp_path / "models"
    _make_run_dir(models, 3, 0)
    _make_run_dir(models, 3, 1, status="failed")

    with pytest.raises(FoldManifestError, match="incomplete campaign"):
        build_fold_manifest(discover_run_dirs(models, 3))


def test_bridge_refuses_a_run_dir_without_slurm_provenance(tmp_path):
    models = tmp_path / "models"
    run_dir = _make_run_dir(models, 5, 0)
    (run_dir / "run_manifest.json").unlink()

    with pytest.raises(FoldManifestError, match="no run_manifest.json"):
        build_fold_manifest([run_dir])


def test_bridge_refuses_a_missing_fold_checkpoint(tmp_path):
    models = tmp_path / "models"
    run_dir = _make_run_dir(models, 2, 0)
    (run_dir / "checkpoints" / "best.ckpt").unlink()

    with pytest.raises(FoldManifestError, match="no model to ensemble"):
        build_fold_manifest([run_dir])


def test_expected_fold_count_is_enforced(tmp_path):
    models = tmp_path / "models"
    for fold in range(3):
        _make_run_dir(models, 2, fold)
    with pytest.raises(FoldManifestError, match="Expected 5 folds.*collected 3"):
        build_fold_manifest(discover_run_dirs(models, 2), expected_folds=5)


def test_expected_fold_identities_are_exact_and_unique(tmp_path):
    models = tmp_path / "models"
    run_dirs = [_make_run_dir(models, 2, fold) for fold in (0, 1, 2, 3, 5)]
    with pytest.raises(FoldManifestError, match="Expected 5 folds"):
        build_fold_manifest(run_dirs, expected_folds=5)


def test_bridge_refuses_a_missing_pretrained_checkpoint_digest(tmp_path):
    models = tmp_path / "models"
    run_dir = _make_run_dir(models, 2, 0, backbone_sha=None)
    with pytest.raises(FoldManifestError, match="no valid pretrained checkpoint SHA-256"):
        build_fold_manifest([run_dir])


def test_discovery_is_scoped_to_one_task_and_run_id(tmp_path):
    models = tmp_path / "models"
    _make_run_dir(models, 2, 0, run_id="lane_a")
    _make_run_dir(models, 2, 1, run_id="lane_b")
    _make_run_dir(models, 4, 0, run_id="lane_a")

    assert len(discover_run_dirs(models, 2)) == 2
    assert len(discover_run_dirs(models, 2, run_id="lane_a")) == 1
    assert len(discover_run_dirs(models, 4)) == 1


def test_provenance_sidecar_carries_the_campaign_identity(tmp_path):
    models = tmp_path / "models"
    for fold in range(2):
        _make_run_dir(models, 2, fold)
    manifest = build_fold_manifest(discover_run_dirs(models, 2))
    sidecar = write_fold_manifest(manifest, tmp_path / "fold_manifest.json")

    payload = json.loads(sidecar.read_text())
    assert payload["schema_version"] == "fomo26-fold-manifest-v1"
    assert payload["pretrained_checkpoint_sha256"] == BACKBONE_SHA
    assert all(record["git_commit"] == "deadbeef" for record in payload["records"])
