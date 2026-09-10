"""The fold manifest is the authority on campaign completeness; packaging consumes it.

Packaging must never restate how many folds a campaign trained. If the container derives its own
fold count from whatever happened to be staged, an incomplete campaign packages silently as a
complete one and the submission misrepresents what the ensemble is.
"""

from __future__ import annotations

import json
import pytest
from finetuning.fomo26_inference.build_fold_manifest import FoldManifestError, build_fold_manifest, write_fold_manifest
from pathlib import Path

BACKBONE_SHA = "f9d698d616d2c97756faa18e2aca5ce823a042c6ff318a9e52a4cc6821ced649"


def _run_dir(root: Path, fold: int, *, backbone: str = BACKBONE_SHA, status: str = "completed") -> Path:
    run_dir = root / "TASK" / f"run_cand__task2__fold{fold}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints" / "best.ckpt").write_bytes(b"weights-%d" % fold)
    (run_dir / "checkpoints" / "last.ckpt").write_bytes(b"last-%d" % fold)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "fold": fold,
                "status": status,
                "task": "SEG902",
                "fomo_task": "2",
                "run_id": "cand",
                "pretrained_checkpoint": "/pin/resenc_unet_b.ckpt",
                "pretrained_checkpoint_sha256": backbone,
                "git_commit": "abc123",
            }
        )
    )
    return run_dir


def test_a_complete_campaign_states_its_completeness_explicitly(tmp_path):
    manifest = build_fold_manifest(
        [_run_dir(tmp_path, f) for f in range(5)],
        expected_folds=5,
        candidate="O",
        inference_policy="task9_downstream_policy",
    )
    assert manifest["trained_fold_count"] == 5
    assert manifest["expected_folds"] == 5
    assert manifest["present_folds"] == ["0", "1", "2", "3", "4"]
    # The success path must make the positive claim, not merely omit the negative one.
    assert manifest["partial"] is False
    assert manifest["partial_reason"] is None
    assert manifest["candidate"] == "O"
    assert manifest["task"] == "2"
    assert manifest["pretrained_checkpoint_sha256"] == BACKBONE_SHA
    assert manifest["inference_policy"] == "task9_downstream_policy"
    assert manifest["git_commit"] == "abc123"
    assert all(record["fold_checkpoint_sha256"] for record in manifest["records"])


def test_every_fold_checkpoint_carries_its_own_digest(tmp_path):
    manifest = build_fold_manifest([_run_dir(tmp_path, f) for f in range(3)])
    digests = [r["fold_checkpoint_sha256"] for r in manifest["records"]]
    assert len(set(digests)) == 3, "distinct weights must hash distinctly"


def test_a_short_campaign_is_refused_rather_than_packaged(tmp_path):
    with pytest.raises(FoldManifestError, match="Expected 5 folds, collected 3"):
        build_fold_manifest([_run_dir(tmp_path, f) for f in range(3)], expected_folds=5)


def test_mixed_backbones_are_refused(tmp_path):
    runs = [_run_dir(tmp_path, 0), _run_dir(tmp_path, 1, backbone="0" * 64)]
    with pytest.raises(FoldManifestError, match="different pretrained checkpoints"):
        build_fold_manifest(runs)


def test_the_sidecar_is_what_packaging_reads(tmp_path):
    from finetuning.container.build_container import _assert_manifest_matches_staged, _read_provenance

    manifest = build_fold_manifest([_run_dir(tmp_path, f) for f in range(5)], expected_folds=5, candidate="O")
    output = tmp_path / "out" / "task2_fold_manifest.json"
    write_fold_manifest(manifest, output)

    # The primary file stays the bare list the inference rail has always consumed.
    assert isinstance(json.loads(output.read_text()), list)
    provenance = _read_provenance(output)
    assert provenance["trained_fold_count"] == 5
    assert provenance["candidate"] == "O"
    _assert_manifest_matches_staged(provenance, staged=5)


def test_packaging_refuses_when_staged_folds_contradict_the_manifest(tmp_path):
    from finetuning.container.build_container import _assert_manifest_matches_staged

    with pytest.raises(SystemExit, match="contradict its provenance"):
        _assert_manifest_matches_staged({"trained_fold_count": 5}, staged=4)


def test_packaging_refuses_a_partial_campaign_unless_it_is_acknowledged(tmp_path):
    from finetuning.container.build_container import _assert_manifest_matches_staged

    partial = {"trained_fold_count": 3, "expected_folds": 5, "partial": True, "partial_reason": "fold 4 failed"}
    with pytest.raises(SystemExit, match="marked partial"):
        _assert_manifest_matches_staged(partial, staged=3)
    # Acknowledged explicitly, it packages -- and the reason travels into the image metadata.
    _assert_manifest_matches_staged({**partial, "partial_acknowledged": True}, staged=3)


# ── the Tasks 6/7 image ships frozen pretrained weights, never a finetuned fold ───────────


def test_the_container_contract_accepts_a_tasks_6_7_image(tmp_path):
    from finetuning.container import container_contract as cc

    definition = tmp_path / "Apptainer.def"
    definition.write_text('Bootstrap: docker\nFrom: x\n%runscript\n    exec python /app/predict.py "$@"\n')
    model_file = tmp_path / "pretrained.ckpt"
    model_file.write_bytes(b"w")
    metadata = tmp_path / "meta.json"
    metadata.write_text(
        json.dumps(
            {
                "version": "1",
                "source_git_commit": "abc",
                "model_files": [str(model_file)],
                "task_entrypoints": {"6_7": "finetuning/container/predict_task6_7.py"},
            }
        )
    )
    report = cc.validate_container_contract(definition, metadata, require_tasks=("6_7",))
    assert report["status"] == "PASS"
    assert set(report["tasks"]) == {"6_7"}

    # The historical Tasks 1-5 requirement still bites when it is what the caller asked for.
    with pytest.raises(cc.ContainerContractInvalid, match=r"must declare entrypoints for tasks"):
        cc.validate_container_contract(definition, metadata)


def test_an_unknown_task_entrypoint_is_refused(tmp_path):
    from finetuning.container import container_contract as cc

    definition = tmp_path / "Apptainer.def"
    definition.write_text("Bootstrap: docker\n%runscript\n    exec python /app/predict.py\n")
    model_file = tmp_path / "m.ckpt"
    model_file.write_bytes(b"w")
    metadata = tmp_path / "meta.json"
    metadata.write_text(
        json.dumps(
            {
                "version": "1",
                "source_git_commit": "abc",
                "model_files": [str(model_file)],
                "task_entrypoints": {"99": "finetuning/container/predict_task1.py"},
            }
        )
    )
    with pytest.raises(cc.ContainerContractInvalid, match="unknown task entrypoints"):
        cc.validate_container_contract(definition, metadata, require_tasks=())


def test_the_shared_definition_no_longer_points_tasks_6_7_at_a_finetuned_run_dir():
    definition = Path("finetuning/container/Apptainer.def").read_text()
    assert "FOMO26_MODEL_DIR" not in definition, "Tasks 6/7 must not resolve a downstream run directory"
    assert "/app/models/env.sh" in definition, "per-task model identity must travel with the image"


def test_requesting_last_checkpoint_is_not_silently_answered_with_best(tmp_path):
    """best.ckpt was selected on each fold's own val set, so OOF must be able to ask for last."""
    run_dir = _run_dir(tmp_path, 0)
    record = json.loads((run_dir / "run_manifest.json").read_text())
    record["best_checkpoint"] = str(run_dir / "checkpoints" / "best.ckpt")
    (run_dir / "run_manifest.json").write_text(json.dumps(record))

    best = build_fold_manifest([run_dir], checkpoint_name="best")["records"][0]["best_ckpt"]
    last = build_fold_manifest([run_dir], checkpoint_name="last")["records"][0]["best_ckpt"]
    assert best.endswith("best.ckpt")
    assert last.endswith("last.ckpt"), "a declared best_checkpoint must not override an explicit --checkpoint-name"
