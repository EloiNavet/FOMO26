"""Focused software-contract tests for the resumable FOMO26 container release rail."""

from __future__ import annotations

import copy
import json
import os
import pytest
import shutil
import subprocess
import sys
from finetuning.container import handoff_adapter, release as rail, validator_bridge as bridge
from finetuning.container.release_manifest import (
    OFFICIAL_TASKS,
    ReleaseContractError,
    finalized_manifest,
    sha256_file,
    validate_release_manifest,
)
from finetuning.fomo26_inference.backbones import canonical_architecture
from pathlib import Path
from types import SimpleNamespace


def _artifact(path: Path, contents: bytes | str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents.encode() if isinstance(contents, str) else contents)
    return {"path": str(path), "sha256": sha256_file(path)}


def _valid_manifest(tmp_path: Path, *, mode: str = "final", selected_members: list[int] | None = None) -> dict:
    candidate = "candidate-a"
    architecture = "resenc_b"
    pretrained = _artifact(tmp_path / "source" / "pretrained.ckpt", b"pretrained")
    pretrained["step"] = 42
    tasks = {}
    for task_number in range(1, 6):
        task = f"task{task_number}"
        folds = []
        source_dirs = []
        fold_numbers = selected_members if selected_members is not None else list(range(5) if mode == "final" else range(2))
        for fold in fold_numbers:
            source_run_dir = f"/provenance/{candidate}/{task}/fold{fold}"
            source_dirs.append(source_run_dir)
            run_manifest = {
                "candidate_id": candidate,
                "architecture": architecture,
                "pretrained_checkpoint_sha256": pretrained["sha256"],
                "fomo_task": task_number,
                "fold": fold,
                "git_commit": "a" * 40,
            }
            folds.append(
                {
                    "fold": fold,
                    "checkpoint": _artifact(
                        tmp_path / "source" / task / f"fold{fold}" / "best.ckpt",
                        f"checkpoint-{task}-{fold}",
                    ),
                    "run_manifest": _artifact(
                        tmp_path / "source" / task / f"fold{fold}" / "run_manifest.json",
                        json.dumps(run_manifest),
                    ),
                    "hydra_config": _artifact(
                        tmp_path / "source" / task / f"fold{fold}" / "config.yaml",
                        f"task: {task}\nfold: {fold}\nmodel:\n  pretrain_net: resenc_unet_b_ssl\n",
                    ),
                    "source_run_dir": source_run_dir,
                    "downstream_run_git_commit": "a" * 40,
                }
            )
        policy = {
            "ensemble_members": list(fold_numbers),
            "member_selection": None if len(fold_numbers) == 1 else "time_budget_auto",
            "ensemble_method": None if len(fold_numbers) == 1 else "mean",
            "tta": "auto",
            "time_target_seconds": 110,
            "calibration": {"state": "none"},
        }
        if task in {"task1", "task3", "task5"}:
            policy["cross_patch"] = "none"
        else:
            policy.update({"window_policy": "checkpoint_config_overlap_0.5", "ensemble_space": "prob"})
        tasks[task] = {
            "candidate_id": candidate,
            "architecture": architecture,
            "pretrained_sha256": pretrained["sha256"],
            "submission_image": f"{task}.sif",
            "selected_members": list(fold_numbers),
            "folds": folds,
            "provenance": {
                "source_run_directories": source_dirs,
                "split_sha256": "1" * 64,
                "protocol_sha256": "2" * 64,
            },
            "policy": policy,
        }
    tasks["task6_and_7"] = {
        "candidate_id": candidate,
        "architecture": architecture,
        "pretrained_sha256": pretrained["sha256"],
        "submission_image": "task6_and_7.sif",
        "downstream_weights": [],
        "policy": {"patch_size": [128, 128, 128], "minimum_encoder_coverage": 0.98},
    }
    return finalized_manifest(
        {
            "schema_version": "fomo26-release-manifest-v1",
            "release_mode": mode,
            "candidate_id": candidate,
            "track": "main-track",
            "scientific_architecture": architecture,
            "architecture": architecture,
            "ssl_objective": "amaes",
            "checkpoint_source": "online",
            "scientific_authority": {
                **_artifact(tmp_path / "source" / "scientific_authority.json", "{}"),
                "schema_version": "test-scientific-authority-v1",
                "candidate_id": candidate,
            },
            "pretrained": pretrained,
            "pretrained_training_git_commit": "a" * 40,
            "packaging_git_commit": "b" * 40,
            "created_at": "2026-08-13T12:00:00Z",
            "protocol": {
                "split_sha256": "1" * 64,
                "protocol_sha256": "2" * 64,
                "qualification_limit_seconds": 110,
            },
            "container": {
                "cardinality": "one_sif_per_task_unit",
                "entrypoint": "/app/predict.py",
                "images": {task: f"{task}.sif" for task in OFFICIAL_TASKS},
            },
            "tasks": tasks,
        }
    )


def _release_dir(tmp_path: Path, manifest: dict, name: str = "release-a") -> Path:
    release = tmp_path / name
    release.mkdir()
    (release / "release_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    rail.write_sha256sums(release)
    return release


def _without_digest(payload: dict) -> dict:
    result = copy.deepcopy(payload)
    result.pop("manifest_sha256", None)
    return result


def _synthetic_validator(tmp_path, monkeypatch, *, files=None):
    """A controlled stand-in for the external validator.

    The real one is not in this repository and must not be downloaded during tests, so the default
    path builds a tiny checkout and a manifest that describes it exactly. That keeps the member-set
    and digest logic under test without a network call or an unlicensed copy.
    """
    import hashlib

    files = files if files is not None else {"container_validator/validate.py": b"print('ok')\n"}
    root = tmp_path / "validator"
    entries = []
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        entries.append(
            {
                "path": rel,
                "snapshot_sha256": hashlib.sha256(data).hexdigest(),
                "snapshot_size": len(data),
                "vendored": True,
            }
        )
    metadata = {
        "upstream_commit": bridge.PINNED_COMMIT,
        "upstream_tree": bridge.PINNED_TREE,
        "upstream_license": {"declared": False},
        "entries": entries,
    }
    monkeypatch.setattr(bridge, "_metadata", lambda: metadata)
    return root, metadata


def test_official_manifest_and_entrypoint_contract_are_pinned():
    """The validator is external now, so the offline gate is the manifest, not the source."""
    report = bridge.verify_manifest()
    assert report["commit"] == bridge.PINNED_COMMIT
    assert report["members"] == 84
    assert report["lfs_objects"] == 21
    assert report["redistributed_here"] is False
    assert set(bridge.verify_entrypoint_contract()) == set(OFFICIAL_TASKS)


def test_verify_snapshot_fails_closed_without_an_acquired_validator(monkeypatch):
    """Absence must be an error, never a pass and never a silent skip."""
    monkeypatch.delenv(bridge.ROOT_ENV, raising=False)
    with pytest.raises(bridge.ValidatorError, match="No official validator available"):
        bridge.verify_snapshot()


def test_verify_snapshot_accepts_an_exactly_matching_checkout(tmp_path, monkeypatch):
    root, _ = _synthetic_validator(tmp_path, monkeypatch)
    report = bridge.verify_snapshot(root)
    assert report["status"] == "VERIFIED" and report["vendored_files"] == 1


@pytest.mark.parametrize("damage", ["missing", "extra", "modified"])
def test_verify_snapshot_refuses_any_member_set_difference(tmp_path, monkeypatch, damage):
    """Set equality: a missing, extra or altered file are all refusals."""
    root, _ = _synthetic_validator(tmp_path, monkeypatch)
    target = root / "container_validator" / "validate.py"
    if damage == "missing":
        target.unlink()
    elif damage == "extra":
        (root / "container_validator" / "sneaky.py").write_bytes(b"x\n")
    else:
        target.write_bytes(b"print('tampered')\n")
    with pytest.raises(bridge.ValidatorError):
        bridge.verify_snapshot(root)


def test_apptainer_base_image_is_digest_only_and_matches_receipt_metadata():
    definition = Path("finetuning/container/Apptainer.def").read_text().splitlines()
    reference = next(line.removeprefix("From: ") for line in definition if line.startswith("From: "))
    repository, digest = reference.split("@", maxsplit=1)

    assert reference == rail.BASE_IMAGE
    assert ":" not in repository.rsplit("/", maxsplit=1)[-1]
    assert digest.startswith("sha256:") and len(digest) == len("sha256:") + 64


def test_apptainer_installs_no_natten_and_fails_closed_on_a_retired_architecture():
    """MedViT retired, and the NATTEN wheel it needed retired with it.

    This replaces the positive "NATTEN only for medvit" contract with its negative half. The image
    still reads the manifest architecture -- but now to refuse an unknown one, rather than to
    decide which extra wheel to fetch. Its wheel pins are preserved as reproduction evidence in
    finetuning/container/historical_sif_profile.json, so nothing is lost by removing the branch.
    """
    definition = Path("finetuning/container/Apptainer.def").read_text()

    assert 'json.load(open("/app/models/model_manifest.json"))["architecture"]' in definition
    # Match the branch and the install, not the word: the recipe's comment explains the history,
    # and a test that forbids naming what was removed forbids documenting it.
    executable = "\n".join(line for line in definition.splitlines() if not line.strip().startswith("#"))
    assert "medvit" not in executable.lower(), "a retired architecture branch came back"
    assert "natten" not in executable.lower(), "the public image must not install NATTEN"
    # The manifest architecture is still read, and now gates the build.
    assert "resenc_b|unet_m)" in definition
    assert "Unsupported architecture for the public release image" in definition


def test_the_historical_natten_wheel_pins_survive_outside_the_public_recipe():
    """Removing the branch must not lose the evidence needed to rebuild the submitted image."""
    profile = json.loads(Path("finetuning/container/historical_sif_profile.json").read_text())
    wheels = profile["natten_wheels"]["python"]
    assert sorted(wheels) == ["3.11", "3.12"]
    for url in wheels.values():
        assert "natten-0.21.0" in url and "sha256=" in url
    assert profile["dependencies"]["natten"] == "0.21.0"


def test_apptainer_pip_downloads_have_bounded_retries_and_timeout():
    definition = Path("finetuning/container/Apptainer.def").read_text()
    install_lines = [line.strip() for line in definition.splitlines() if line.strip().startswith("pip install")]

    # Two now, not three: the NATTEN wheel install went with MedViT. The count is asserted so a
    # silently added download is caught, but the invariant that matters is the one below -- every
    # pip invocation, however many there are, is bounded.
    assert len(install_lines) == 2
    assert all("--retries 10 --timeout 120" in line for line in install_lines)
    assert install_lines, "an unbounded-download guard that inspects nothing guards nothing"


def test_stale_validator_metadata_is_rejected(tmp_path, monkeypatch):
    root, metadata = _synthetic_validator(tmp_path, monkeypatch)
    metadata["upstream_commit"] = "0" * 40
    with pytest.raises(bridge.ValidatorError, match="approved commit/tree"):
        bridge.verify_snapshot(root)
    with pytest.raises(bridge.ValidatorError, match="approved commit/tree"):
        bridge.verify_manifest()


def test_final_release_manifest_accepts_complete_frozen_contract(tmp_path):
    manifest = _valid_manifest(tmp_path)
    assert validate_release_manifest(manifest)["manifest_sha256"] == manifest["manifest_sha256"]


def test_final_release_manifest_accepts_authoritative_single_member(tmp_path):
    manifest = _valid_manifest(tmp_path, selected_members=[0])
    assert validate_release_manifest(manifest)["tasks"]["task1"]["selected_members"] == [0]


def test_architecture_alias_is_explicit_and_preserves_scientific_identity(tmp_path):
    manifest = _without_digest(_valid_manifest(tmp_path, selected_members=[0]))
    manifest["scientific_architecture"] = "resenc_unet_b"
    assert canonical_architecture("resenc_unet_b") == "resenc_b"
    validate_release_manifest(manifest)

    manifest["scientific_architecture"] = "unet_m"
    with pytest.raises(ReleaseContractError, match="does not resolve"):
        validate_release_manifest(manifest)


def test_handoff_derivation_uses_commit_relative_paths_for_runtime_sources():
    source = Path("finetuning/container/predict_task1.py")
    receipt = handoff_adapter._derived("auto", "RUNTIME_CONTRACT_DERIVABLE", source, "test")
    assert receipt["source_path"] == source.as_posix()
    assert receipt["source_sha256"] == sha256_file(source)


def test_selected_member_contract_rejects_stage_drift_and_undeclared_extra(tmp_path):
    manifest = _without_digest(_valid_manifest(tmp_path / "different", selected_members=[0]))
    manifest["tasks"]["task3"]["selected_members"] = [1]
    with pytest.raises(ReleaseContractError, match="do not match authoritative selected_members"):
        validate_release_manifest(manifest)

    manifest = _without_digest(_valid_manifest(tmp_path / "extra", selected_members=[0]))
    extra = copy.deepcopy(manifest["tasks"]["task5"]["folds"][0])
    extra["fold"] = 1
    extra["source_run_dir"] += "-undeclared"
    manifest["tasks"]["task5"]["folds"].append(extra)
    manifest["tasks"]["task5"]["provenance"]["source_run_directories"].append(extra["source_run_dir"])
    with pytest.raises(ReleaseContractError, match="do not match authoritative selected_members"):
        validate_release_manifest(manifest, verify_files=False)


def test_single_member_policy_rejects_fabricated_ensemble_semantics(tmp_path):
    manifest = _without_digest(_valid_manifest(tmp_path, selected_members=[0]))
    manifest["tasks"]["task1"]["policy"]["ensemble_method"] = "mean"
    with pytest.raises(ReleaseContractError, match="single-member policy"):
        validate_release_manifest(manifest)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(candidate_id="../escape"), "unsafe characters"),
        (lambda value: value["tasks"]["task3"].update(architecture="unet_m"), "mixed candidate or architecture"),
        (lambda value: value["tasks"]["task4"].update(pretrained_sha256="f" * 64), "conflicting pretrained"),
        (
            lambda value: (
                value["tasks"]["task2"]["folds"].pop(),
                value["tasks"]["task2"]["provenance"]["source_run_directories"].pop(),
            ),
            "do not match authoritative selected_members",
        ),
        (
            lambda value: value["tasks"]["task1"]["folds"].append(copy.deepcopy(value["tasks"]["task1"]["folds"][0])),
            "duplicate fold",
        ),
        (lambda value: value["tasks"]["task5"].pop("policy"), "policy must be an object"),
        (lambda value: value["tasks"]["task3"].pop("provenance"), "provenance must be an object"),
        (lambda value: value["tasks"]["task6_and_7"].update(downstream_weights=["bad.ckpt"]), "empty list"),
    ],
    ids=[
        "unsafe-id",
        "mixed-architecture",
        "mixed-pretrained",
        "missing-fold",
        "duplicate-fold",
        "missing-policy",
        "missing-provenance",
        "task6-downstream-weight",
    ],
)
def test_final_manifest_refuses_identity_completeness_and_policy_failures(tmp_path, mutation, message):
    manifest = _without_digest(_valid_manifest(tmp_path))
    mutation(manifest)
    with pytest.raises(ReleaseContractError, match=message):
        validate_release_manifest(manifest)


def test_required_calibration_must_exist_and_match_digest(tmp_path):
    manifest = _without_digest(_valid_manifest(tmp_path))
    manifest["tasks"]["task1"]["policy"]["calibration"] = {
        "state": "required",
        "artifact": {"path": str(tmp_path / "missing.json"), "sha256": "c" * 64},
    }
    with pytest.raises(ReleaseContractError, match="calibration.artifact is missing"):
        validate_release_manifest(manifest)


def test_segmentation_calibration_cannot_be_declared_but_ignored(tmp_path):
    manifest = _without_digest(_valid_manifest(tmp_path))
    calibration = _artifact(tmp_path / "calibration.json", "{}")
    manifest["tasks"]["task2"]["policy"]["calibration"] = {"state": "required", "artifact": calibration}
    with pytest.raises(ReleaseContractError, match="supports only explicit calibration state 'none'"):
        validate_release_manifest(manifest)


def test_runtime_required_calibration_fails_closed(tmp_path):
    from finetuning.container.fomo_ensemble_predict import _load_calibration

    with pytest.raises(SystemExit, match="required by the frozen release policy"):
        _load_calibration(None, "cls", required=True)
    with pytest.raises(SystemExit, match="Required calibration artifact is missing"):
        _load_calibration(str(tmp_path / "missing.json"), "reg", required=True)


def test_checkpoint_and_run_manifest_identity_drift_are_rejected(tmp_path):
    manifest = _without_digest(_valid_manifest(tmp_path))
    checkpoint = Path(manifest["tasks"]["task1"]["folds"][0]["checkpoint"]["path"])
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ReleaseContractError, match="checkpoint drifted"):
        validate_release_manifest(manifest)

    manifest = _without_digest(_valid_manifest(tmp_path / "second"))
    run_spec = manifest["tasks"]["task2"]["folds"][0]["run_manifest"]
    run_path = Path(run_spec["path"])
    record = json.loads(run_path.read_text())
    record["candidate_id"] = "other"
    run_path.write_text(json.dumps(record))
    run_spec["sha256"] = sha256_file(run_path)
    with pytest.raises(ReleaseContractError, match="conflicting candidate_id"):
        validate_release_manifest(manifest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fold", 4, "conflicting fold"),
        ("git_commit", "9" * 40, "conflicting downstream run git commit"),
    ],
)
def test_run_manifest_fold_and_training_commit_are_bound(tmp_path, field, value, message):
    manifest = _without_digest(_valid_manifest(tmp_path))
    run_spec = manifest["tasks"]["task3"]["folds"][0]["run_manifest"]
    run_path = Path(run_spec["path"])
    record = json.loads(run_path.read_text())
    record[field] = value
    run_path.write_text(json.dumps(record))
    run_spec["sha256"] = sha256_file(run_path)
    with pytest.raises(ReleaseContractError, match=message):
        validate_release_manifest(manifest)


def test_hydra_architecture_and_auto_member_selection_are_explicit(tmp_path):
    manifest = _without_digest(_valid_manifest(tmp_path))
    config_spec = manifest["tasks"]["task4"]["folds"][0]["hydra_config"]
    config = Path(config_spec["path"])
    config.write_text("model:\n  pretrain_net: unet_m\n")
    config_spec["sha256"] = sha256_file(config)
    with pytest.raises(ReleaseContractError, match="architecture evidence"):
        validate_release_manifest(manifest)

    manifest = _without_digest(_valid_manifest(tmp_path / "selection"))
    manifest["tasks"]["task1"]["policy"]["member_selection"] = "fixed"
    with pytest.raises(ReleaseContractError, match="time_budget_auto"):
        validate_release_manifest(manifest)


def test_smoke_only_can_be_partial_but_never_ready(tmp_path):
    manifest = _valid_manifest(tmp_path, mode="smoke_only")
    validate_release_manifest(manifest)
    release = _release_dir(tmp_path, manifest)
    blockers = rail.readiness_blockers(release, manifest)
    assert "release_mode_is_smoke_only" in blockers


def test_smoke_only_manifest_cannot_be_reaudited_as_final(tmp_path):
    manifest = _valid_manifest(tmp_path / "inputs", mode="smoke_only")
    candidate_input = tmp_path / "smoke.json"
    candidate_input.write_text(json.dumps(manifest))
    with pytest.raises(rail.ReleaseError, match="permanently watermarked"):
        rail.audit_candidate(candidate_input, tmp_path / "promoted", mode="final")


def test_verify_release_writes_one_idempotent_pass_receipt(tmp_path):
    release = _release_dir(tmp_path, _valid_manifest(tmp_path / "inputs", mode="smoke_only"))

    first = rail.verify_release(release)
    second = rail.verify_release(release)

    assert first["status"] == "PASS"
    assert first["checksum_status"] == "VERIFIED"
    assert first["checked_files"] == 1
    assert second == {**first, "idempotent": True}
    assert len(list((release / "receipts" / "verify").glob("*.json"))) == 1


def test_export_is_normalized_atomic_and_idempotent(tmp_path, monkeypatch):
    source = _release_dir(tmp_path, _valid_manifest(tmp_path / "inputs"))
    destination_root = tmp_path / "exports"
    first = rail.export_release(source, destination_root)
    exported = destination_root / source.name
    normalized = json.loads((exported / "release_manifest.json").read_text())
    assert first["status"] == "PASS"
    assert normalized["pretrained"]["path"] == "artifacts/pretrained/pretrained.ckpt"
    assert not Path(normalized["pretrained"]["path"]).is_absolute()
    assert rail.verify_sha256sums(exported)["status"] == "VERIFIED"
    monkeypatch.setattr(rail, "_copy_artifact", lambda *args, **kwargs: pytest.fail("verified artifact recopied"))
    second = rail.export_release(source, destination_root)
    assert second["idempotent"] is True


def test_export_collision_and_checksum_mutation_fail_closed(tmp_path):
    source = _release_dir(tmp_path, _valid_manifest(tmp_path / "inputs"))
    destination = tmp_path / "exports" / source.name
    destination.mkdir(parents=True)
    (destination / "unrelated").write_text("collision")
    with pytest.raises(rail.ReleaseError, match="Missing SHA256SUMS"):
        rail.export_release(source, tmp_path / "exports")

    destination = tmp_path / "clean-exports"
    rail.export_release(source, destination)
    exported = destination / source.name
    Path(json.loads((exported / "release_manifest.json").read_text())["pretrained"]["path"])
    staged_pretrained = exported / "artifacts" / "pretrained" / "pretrained.ckpt"
    staged_pretrained.write_bytes(b"corrupt")
    with pytest.raises(rail.ReleaseError, match="Checksum drift"):
        rail.verify_sha256sums(exported)


def test_pull_uses_resumable_rsync_without_delete_and_is_idempotent(tmp_path, monkeypatch):
    source = _release_dir(tmp_path, _valid_manifest(tmp_path / "inputs"), name="release-pull")
    rail.write_sha256sums(source)
    script = tmp_path / "fake-rsync"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, shutil, sys\n"
        "source = pathlib.Path(os.environ['FAKE_RSYNC_SOURCE'])\n"
        "destination = pathlib.Path(sys.argv[-1])\n"
        "shutil.copytree(source, destination, dirs_exist_ok=True)\n"
    )
    script.chmod(0o755)
    monkeypatch.setenv("FAKE_RSYNC_SOURCE", str(source))
    dry = rail.pull_release(
        "host:/remote/release-pull",
        tmp_path / "pulled",
        release_id="release-pull",
        dry_run=True,
        rsync=str(script),
    )
    assert "--delete" not in dry["command"]
    first = rail.pull_release(
        "host:/remote/release-pull",
        tmp_path / "pulled",
        release_id="release-pull",
        rsync=str(script),
    )
    assert first["status"] == "PASS"
    second = rail.pull_release(
        "host:/remote/release-pull",
        tmp_path / "pulled",
        release_id="release-pull",
        rsync=str(script),
    )
    assert second["idempotent"] is True


def test_interrupted_pull_reuses_partial_destination_and_completes(tmp_path, monkeypatch):
    source = _release_dir(tmp_path, _valid_manifest(tmp_path / "inputs"), name="release-interrupted")
    rail.write_sha256sums(source)
    script = tmp_path / "flaky-rsync"
    marker = tmp_path / "first-attempt-failed"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, shutil, sys\n"
        "source = pathlib.Path(os.environ['FAKE_RSYNC_SOURCE'])\n"
        "destination = pathlib.Path(sys.argv[-1])\n"
        "destination.mkdir(parents=True, exist_ok=True)\n"
        "marker = pathlib.Path(os.environ['FAKE_RSYNC_MARKER'])\n"
        "if not marker.exists():\n"
        "    shutil.copy2(source / 'release_manifest.json', destination / 'release_manifest.json')\n"
        "    marker.write_text('failed')\n"
        "    raise SystemExit(23)\n"
        "shutil.copytree(source, destination, dirs_exist_ok=True)\n"
    )
    script.chmod(0o755)
    monkeypatch.setenv("FAKE_RSYNC_SOURCE", str(source))
    monkeypatch.setenv("FAKE_RSYNC_MARKER", str(marker))
    destination_root = tmp_path / "pulled"
    with pytest.raises(rail.ReleaseError, match="rsync failed"):
        rail.pull_release(
            "host:/remote/release-interrupted",
            destination_root,
            release_id="release-interrupted",
            attempts=1,
            rsync=str(script),
        )
    partial = destination_root / ".release-interrupted.partial"
    assert (partial / "release_manifest.json").is_file()

    result = rail.pull_release(
        "host:/remote/release-interrupted",
        destination_root,
        release_id="release-interrupted",
        attempts=1,
        rsync=str(script),
    )
    assert result["status"] == "PASS"
    assert not partial.exists()
    assert (destination_root / "release-interrupted" / "receipts" / "pull.json").is_file()


def test_stage_contexts_are_runtime_only_host_path_free_and_idempotent(tmp_path, monkeypatch):
    manifest = _valid_manifest(tmp_path / "inputs")
    release = _release_dir(tmp_path, manifest)
    monkeypatch.setattr(
        "finetuning.container.build_container._tracked_runtime_files",
        lambda: [Path("README.md"), Path("pyproject.toml")],
    )
    monkeypatch.setattr(
        "finetuning.container.build_container.audit_staged_runtime_imports",
        lambda context, task: {"status": "PASS", "task": task, "missing": []},
    )
    monkeypatch.setattr(rail, "_git_head", lambda: manifest["packaging_git_commit"])
    monkeypatch.setattr(rail, "_git_is_clean", lambda: True)

    def fake_export(destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("packaging==24.2 --hash=sha256:" + "a" * 64 + "\n")

    monkeypatch.setattr("finetuning.container.build_container._export_locked_requirements", fake_export)
    first = rail.stage_release(release)
    assert len(first) == 6
    task1 = release / "build-contexts" / "task1"
    internal_text = (task1 / "models" / "model_manifest.json").read_text()
    assert str(tmp_path) not in internal_text
    assert "/app/models/runs/fold0/checkpoints/best.ckpt" in internal_text
    assert not any(path.is_symlink() for path in task1.rglob("*"))
    task3_predict = (release / "build-contexts" / "task3" / "predict.py").read_bytes()
    task5_predict = (release / "build-contexts" / "task5" / "predict.py").read_bytes()
    assert task3_predict == Path("finetuning/container/predict_task3.py").read_bytes()
    assert task5_predict == Path("finetuning/container/predict_task5.py").read_bytes()
    assert task3_predict != task5_predict
    stage_receipts = {item["task"]: item for item in first}
    assert stage_receipts["task3"]["entrypoint_sha256"] != stage_receipts["task5"]["entrypoint_sha256"]
    task67 = json.loads((release / "build-contexts" / "task6_and_7" / "models/model_manifest.json").read_text())
    assert task67["downstream_finetuned_weights"] == []
    second = rail.stage_release(release)
    assert all(item["idempotent"] for item in second)


def test_staged_runtime_import_audit_rejects_missing_local_module(tmp_path):
    from finetuning.container.build_container import audit_staged_runtime_imports

    context = tmp_path / "context"
    (context / "asparagus_repo" / "asparagus").mkdir(parents=True)
    (context / "asparagus_repo" / "asparagus" / "__init__.py").write_text("")
    (context / "predict.py").write_text("from asparagus.paths import get_data_path\n")

    with pytest.raises(SystemExit, match=r"asparagus/paths\.py"):
        audit_staged_runtime_imports(context, "task1")


def test_final_staged_contexts_resolve_production_imports_without_checkout(tmp_path, monkeypatch):
    manifest = _valid_manifest(tmp_path / "inputs", selected_members=[0])
    release = _release_dir(tmp_path, manifest)
    monkeypatch.setattr(rail, "_git_head", lambda: manifest["packaging_git_commit"])
    monkeypatch.setattr(rail, "_git_is_clean", lambda: True)

    def fake_export(destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("packaging==24.2 --hash=sha256:" + "a" * 64 + "\n")

    monkeypatch.setattr("finetuning.container.build_container._export_locked_requirements", fake_export)
    receipts = rail.stage_release(release)
    assert len(receipts) == len(OFFICIAL_TASKS)

    isolated_cwd = tmp_path / "isolated-cwd"
    isolated_cwd.mkdir()
    probe = """
import importlib.util
import pathlib
import sys

runtime_root = pathlib.Path(sys.argv[1]).resolve()
entrypoint = pathlib.Path(sys.argv[2]).resolve()
spec = importlib.util.spec_from_file_location("fomo26_staged_predict", entrypoint)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
leaks = {}
for name, loaded in tuple(sys.modules.items()):
    if name != "asparagus" and name != "finetuning" and not name.startswith(("asparagus.", "finetuning.")):
        continue
    source = getattr(loaded, "__file__", None)
    if source is not None and not pathlib.Path(source).resolve().is_relative_to(runtime_root):
        leaks[name] = source
if leaks:
    raise RuntimeError(f"repository-local import leaked outside staged runtime: {leaks}")
"""
    for task in OFFICIAL_TASKS:
        context = release / "build-contexts" / task
        runtime_root = context / "asparagus_repo"
        audit = json.loads((context / "context_audit.json").read_text())["runtime_import_closure"]
        assert audit["status"] == "PASS"
        assert audit["missing"] == []
        assert (runtime_root / "asparagus" / "paths.py").is_file()
        # The closure is verified by the audit itself, which raises SystemExit when a file it
        # resolved is not staged; asserting one hardcoded member on top of that only pins whatever
        # the transitive imports happened to be. `metrics.py` used to appear in the Tasks 6/7
        # closure through the segmentation module; the multi-objective SSL imports that dragged it
        # there are not part of this distribution, so the closure is smaller and still complete.
        assert audit["local_files_verified"] > 0

        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(runtime_root)
        environment["PYTHONNOUSERSITE"] = "1"
        environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
        environment.pop("PYTHONHOME", None)
        result = subprocess.run(
            [sys.executable, "-c", probe, str(runtime_root), str(context / "predict.py")],
            cwd=isolated_cwd,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{task}: {result.stderr}"


def test_context_audit_rejects_host_paths_and_secret_material(tmp_path):
    context = tmp_path / "context"
    context.mkdir()
    (context / "metadata.json").write_text('{"path": "/home/user/checkpoint.ckpt"}')
    with pytest.raises(SystemExit, match="host_path"):
        __import__("finetuning.container.build_container", fromlist=["audit_release_context"]).audit_release_context(context)
    (context / "metadata.json").write_text("-----BEGIN OPENSSH PRIVATE KEY-----")
    with pytest.raises(SystemExit, match="secret_text"):
        __import__("finetuning.container.build_container", fromlist=["audit_release_context"]).audit_release_context(context)


def test_fake_apptainer_build_is_idempotent_and_collision_safe(tmp_path):
    manifest = _valid_manifest(tmp_path / "inputs")
    release = _release_dir(tmp_path, manifest)
    for task in OFFICIAL_TASKS:
        context = release / "build-contexts" / task
        context.mkdir(parents=True)
        (context / "Apptainer.def").write_text("Bootstrap: docker\nFrom: pinned\n")
        (context / "predict.py").write_text(f"# {task}\n")
        model_manifest = context / "models" / "model_manifest.json"
        model_manifest.parent.mkdir()
        model_manifest.write_text(
            json.dumps(
                {
                    "task": task,
                    "architecture": "resenc_b",
                    "pretrained_checkpoint_sha256": manifest["pretrained"]["sha256"],
                }
            )
        )
        receipt = release / "receipts" / "stage" / f"{task}.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "task": task,
                    "context_sha256": rail._tree_sha256(context),
                    "release_manifest_sha256": manifest["manifest_sha256"],
                }
            )
        )
    fake = tmp_path / "fake-apptainer"
    fake.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo \'apptainer version 1.4.0\'; exit 0; fi\n'
        'if [ "$1" = "build" ]; then printf \'fake-sif-%s\' "$5" > "$5"; printf \'%s\' "$PWD" > "$5.cwd"; exit 0; fi\n'
        "exit 2\n"
    )
    fake.chmod(0o755)
    first = rail.build_release(release, apptainer=str(fake))
    assert len(first) == 6
    assert all(item["dependency_versions"]["natten"] == "0.21.0" for item in first)
    assert all(item["dependency_versions"]["natten_backend"] == "python-only" for item in first)
    for task in OFFICIAL_TASKS:
        assert (release / "images" / f"{task}.sif.cwd").read_text() == str(release / "build-contexts" / task)
        receipt = json.loads((release / "receipts" / "build" / f"{task}.json").read_text())
        assert receipt["task"] == task
        assert receipt["pretrained_checkpoint_sha256"] == manifest["pretrained"]["sha256"]
    second = rail.build_release(release, apptainer=str(fake))
    assert all(item["idempotent"] for item in second)


def test_one_shot_engineering_smoke_is_timestamped_complete_and_resumable(tmp_path, monkeypatch):
    manifest = _valid_manifest(tmp_path / "inputs", mode="smoke_only")
    candidate_input = tmp_path / "candidate.json"
    candidate_input.write_text(json.dumps(manifest))
    monkeypatch.setattr(rail, "verify_snapshot", lambda root=None: {})
    monkeypatch.setattr(rail, "verify_entrypoint_contract", lambda: {})
    monkeypatch.setattr(rail, "_git_head", lambda: manifest["packaging_git_commit"])
    monkeypatch.setattr(rail, "_git_is_clean", lambda: True)
    monkeypatch.setattr(
        "finetuning.container.build_container._tracked_runtime_files",
        lambda: [Path("README.md"), Path("pyproject.toml")],
    )
    monkeypatch.setattr(
        "finetuning.container.build_container.audit_staged_runtime_imports",
        lambda context, task: {"status": "PASS", "task": task, "missing": []},
    )

    def fake_export(destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("packaging==24.2 --hash=sha256:" + "a" * 64 + "\n")

    monkeypatch.setattr("finetuning.container.build_container._export_locked_requirements", fake_export)
    fixture_digest = "f" * 64
    validator_digest = "v" * 64
    monkeypatch.setattr(rail, "verify_fixtures", lambda cache, root=None: {"fixture_digest": fixture_digest})
    monkeypatch.setattr(rail, "validator_sha256", lambda: validator_digest)

    def fake_validator(*, task, sif, fixture_cache, apptainer, no_gpu, timeout):
        digest = sha256_file(sif)
        return {
            "status": "PASS",
            "task": task,
            "structural_only": no_gpu,
            "returncode": 0,
            "success_summary": "ALL 12 TESTS PASSED",
            "stdout": "ALL 12 TESTS PASSED",
            "stderr": "",
            "sif_sha256_before": digest,
            "sif_sha256_after": digest,
            "validator_sha256": validator_digest,
            "fixture_digest": fixture_digest,
        }

    monkeypatch.setattr(rail, "run_official_validator", fake_validator)
    fake = tmp_path / "fake-apptainer"
    counter = tmp_path / "build-count"
    fake.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo \'apptainer version engineering-smoke\'; exit 0; fi\n'
        f'if [ "$1" = "build" ]; then echo x >> "{counter}"; printf \'engineering-sif-%s\' "$PWD" > "$5"; exit 0; fi\n'
        "exit 2\n"
    )
    fake.chmod(0o755)

    kwargs = {
        "input_path": candidate_input,
        "output_root": tmp_path / "releases",
        "mode": "smoke_only",
        "through": "finalize",
        "fixture_cache": tmp_path,
        "apptainer": str(fake),
        "no_gpu": True,
    }
    first = rail.orchestrate_release(**kwargs)
    release = Path(first["release_dir"])
    assert release.name == "2026-08-13_120000Z_candidate-a_main-track"
    assert first["completed_phase"] == "finalize"
    assert first["candidate_ready"] is False
    assert "release_mode_is_smoke_only" in first["blockers"]
    assert "release:missing_or_stale_verify_receipt" not in first["blockers"]
    assert sorted(path.name for path in (release / "images").glob("*.sif")) == [f"{task}.sif" for task in OFFICIAL_TASKS]
    assert (release / "README_RELEASE.md").is_file()
    assert "ENGINEERING_SMOKE_ONLY" in (release / "README_RELEASE.md").read_text()
    assert (release / "SUBMISSION_CHECKLIST.md").is_file()
    assert (release / "manifests" / "candidate_input.json").is_file()
    assert (release / "logs" / "finalize.json").is_file()
    assert len(counter.read_text().splitlines()) == 6
    before = {task: sha256_file(release / "images" / f"{task}.sif") for task in OFFICIAL_TASKS}

    second = rail.orchestrate_release(**kwargs)
    assert second["release_dir"] == first["release_dir"]
    assert len(counter.read_text().splitlines()) == 6
    assert before == {task: sha256_file(release / "images" / f"{task}.sif") for task in OFFICIAL_TASKS}
    assert all(item["idempotent"] for item in second["phases"]["build"])


def test_failed_task_build_retries_without_rebuilding_successful_images(tmp_path):
    manifest = _valid_manifest(tmp_path / "inputs")
    release = _release_dir(tmp_path, manifest)
    for task in OFFICIAL_TASKS:
        context = release / "build-contexts" / task
        context.mkdir(parents=True)
        (context / "Apptainer.def").write_text("Bootstrap: docker\nFrom: pinned\n")
        (context / "predict.py").write_text(f"# {task}\n")
        model_manifest = context / "models" / "model_manifest.json"
        model_manifest.parent.mkdir()
        model_manifest.write_text(
            json.dumps(
                {
                    "task": task,
                    "architecture": "resenc_b",
                    "pretrained_checkpoint_sha256": manifest["pretrained"]["sha256"],
                    "models": [],
                }
            )
        )
        stage = release / "receipts" / "stage" / f"{task}.json"
        stage.parent.mkdir(parents=True, exist_ok=True)
        stage.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "task": task,
                    "context_sha256": rail._tree_sha256(context),
                    "release_manifest_sha256": manifest["manifest_sha256"],
                }
            )
        )

    fake = tmp_path / "flaky-apptainer"
    failed_once = tmp_path / "failed-once"
    calls = tmp_path / "build-calls"
    fake.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo \'apptainer version 1.4.0\'; exit 0; fi\n'
        f'if [ "$1" = "build" ]; then echo "$PWD" >> "{calls}"; '
        f'case "$PWD" in */task3) if [ ! -f "{failed_once}" ]; then touch "{failed_once}"; '
        'printf incomplete > "$5"; exit 9; fi;; esac; printf \'complete-%s\' "$PWD" > "$5"; exit 0; fi\n'
        "exit 2\n"
    )
    fake.chmod(0o755)

    with pytest.raises(rail.ReleaseError, match="failed for task3"):
        rail.build_release(release, apptainer=str(fake))
    task1 = release / "images" / "task1.sif"
    task2 = release / "images" / "task2.sif"
    assert task1.is_file() and task2.is_file()
    assert not (release / "images" / "task3.sif").exists()
    before = {"task1": sha256_file(task1), "task2": sha256_file(task2)}
    assert list((release / "receipts" / "build" / "failures").glob("task3-*.json"))
    (release / "images" / "task3.sif").write_bytes(b"interrupted-unreceipted-build")

    receipts = rail.build_release(release, apptainer=str(fake))
    assert len(receipts) == 6
    assert receipts[0]["idempotent"] is True and receipts[1]["idempotent"] is True
    assert before == {"task1": sha256_file(task1), "task2": sha256_file(task2)}
    assert len(calls.read_text().splitlines()) == 7
    assert list((release / "receipts" / "build" / "recoveries").glob("task3-*.json"))


@pytest.mark.parametrize("mutate_sif", [False, True], ids=["malformed-summary", "sif-mutated"])
def test_official_validator_requires_success_summary_and_unchanged_sif(tmp_path, monkeypatch, mutate_sif):
    sif = tmp_path / "task1.sif"
    sif.write_bytes(b"sif")
    monkeypatch.setattr(bridge, "validator_root", lambda root=None: tmp_path)
    monkeypatch.setattr(bridge, "verify_snapshot", lambda root=None: {})
    monkeypatch.setattr(bridge, "verify_entrypoint_contract", lambda: {})
    monkeypatch.setattr(bridge, "verify_fixtures", lambda cache, root=None: {"fixture_digest": "d" * 64})

    def fake_run(*args, **kwargs):
        if mutate_sif:
            sif.write_bytes(b"changed")
            stdout = "ALL 12 TESTS PASSED"
        else:
            stdout = "validator exited without terminal result"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(bridge.subprocess, "run", fake_run)
    result = bridge.run_official_validator(task="task1", sif=sif, fixture_cache=tmp_path)
    assert result["status"] == "FAIL"


def test_cpu_structural_validation_cannot_mask_gpu_validation(tmp_path, monkeypatch):
    manifest = _valid_manifest(tmp_path / "inputs")
    release = _release_dir(tmp_path, manifest)
    image_dir = release / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for task in OFFICIAL_TASKS:
        (image_dir / f"{task}.sif").write_bytes(f"{task}-image".encode())
    fixture_digest = "f" * 64
    monkeypatch.setattr(rail, "verify_fixtures", lambda cache, root=None: {"fixture_digest": fixture_digest})
    monkeypatch.setattr(rail, "validator_sha256", lambda: "v" * 64)
    calls = []

    def fake_validator(*, task, sif, fixture_cache, apptainer, no_gpu, timeout):
        calls.append((task, no_gpu))
        digest = sha256_file(sif)
        return {
            "status": "PASS",
            "task": task,
            "structural_only": no_gpu,
            "returncode": 0,
            "success_summary": "ALL 12 TESTS PASSED",
            "stdout": "ALL 12 TESTS PASSED",
            "stderr": "",
            "sif_sha256_before": digest,
            "sif_sha256_after": digest,
            "validator_sha256": "v" * 64,
            "fixture_digest": fixture_digest,
        }

    monkeypatch.setattr(rail, "run_official_validator", fake_validator)
    rail.validate_images(release, fixture_cache=tmp_path, no_gpu=True)
    rail.validate_images(release, fixture_cache=tmp_path, no_gpu=False)
    assert len(calls) == 12
    assert all((release / "receipts" / "validator" / f"{task}-structural.json").is_file() for task in OFFICIAL_TASKS)
    assert all((release / "receipts" / "validator" / f"{task}.json").is_file() for task in OFFICIAL_TASKS)


def test_fixture_pointer_and_corruption_are_rejected(tmp_path, monkeypatch):
    manifest_bytes = b"inputs/task1/x.nii.gz\n"
    root, _ = _synthetic_validator(tmp_path, monkeypatch, files={"container_validator/data/manifest.yaml": manifest_bytes})
    monkeypatch.setenv(bridge.ROOT_ENV, str(root))
    (tmp_path / "manifest.yaml").write_bytes(manifest_bytes)
    entry = {
        "path": "container_validator/data/inputs/task1/x.nii.gz",
        "lfs": {"size": 12, "oid_sha256": "a" * 64},
    }
    monkeypatch.setattr(bridge, "_fixture_entries", lambda root=None: [entry])
    fixture = tmp_path / "inputs" / "task1" / "x.nii.gz"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(bridge.LFS_HEADER + b"\n")
    with pytest.raises(bridge.ValidatorError, match="LFS pointer"):
        bridge.verify_fixtures(tmp_path)
    fixture.write_bytes(b"x" * 12)
    with pytest.raises(bridge.ValidatorError, match="wrong SHA-256"):
        bridge.verify_fixtures(tmp_path)


def _write_ready_receipts(release: Path, manifest: dict) -> None:
    current_validator_sha = bridge.validator_sha256()
    image_dir = release / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for task in OFFICIAL_TASKS:
        image = image_dir / f"{task}.sif"
        image.write_bytes(f"{task}-image".encode())
        digest = sha256_file(image)
        build = release / "receipts" / "build" / f"{task}.json"
        build.parent.mkdir(parents=True, exist_ok=True)
        build.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "task": task,
                    "sif_sha256": digest,
                    "release_manifest_sha256": manifest["manifest_sha256"],
                    "pretrained_checkpoint_sha256": manifest["pretrained"]["sha256"],
                }
            )
        )
        for phase, payload in (
            (
                "validator",
                {
                    "status": "PASS",
                    "task": task,
                    "structural_only": False,
                    "success_summary": "ALL 12 TESTS PASSED",
                    "sif_sha256_before": digest,
                    "sif_sha256_after": digest,
                    "validator_sha256": current_validator_sha,
                    "fixture_digest": "f" * 64,
                    "release_manifest_sha256": manifest["manifest_sha256"],
                },
            ),
            (
                "qualification",
                {
                    "status": "PASS",
                    "task": task,
                    "gpu_name": "NVIDIA H100 80GB HBM3",
                    "wall_seconds": manifest["protocol"]["qualification_limit_seconds"] - 1,
                    "sif_sha256_before": digest,
                    "sif_sha256_after": digest,
                    "validator_sha256": current_validator_sha,
                    "fixture_digest": "f" * 64,
                    "release_manifest_sha256": manifest["manifest_sha256"],
                    "threshold_seconds": manifest["protocol"]["qualification_limit_seconds"],
                    "returncode": 0,
                    "success_summary": "ALL 12 TESTS PASSED",
                },
            ),
        ):
            path = release / "receipts" / phase / f"{task}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload))
    qualification_hashes = {
        task: sha256_file(release / "receipts" / "qualification" / f"{task}.json") for task in OFFICIAL_TASKS
    }
    (release / "receipts" / "qualification-collect-734291.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "release_manifest_sha256": manifest["manifest_sha256"],
                "receipt_sha256": qualification_hashes,
            }
        )
    )
    rail.write_sha256sums(release)
    sums = release / "SHA256SUMS"
    verify = {
        "status": "PASS",
        "release_manifest_sha256": manifest["manifest_sha256"],
        "sha256sums_sha256": sha256_file(sums),
    }
    (release / "receipts" / "verify.json").write_text(json.dumps(verify))


def test_finalize_requires_all_real_gates_and_detects_post_validation_sif_drift(tmp_path):
    manifest = _valid_manifest(tmp_path / "inputs")
    release = _release_dir(tmp_path, manifest)
    _write_ready_receipts(release, manifest)
    ready, blockers = rail.finalize_release(release)
    assert ready is True
    assert blockers == []
    assert "Submit as a Team" in (release / "SUBMISSION_CHECKLIST.md").read_text()
    (release / "images" / "task3.sif").write_bytes(b"mutated")
    ready, blockers = rail.finalize_release(release)
    assert ready is False
    assert "task3:post_validation_sif_drift" in blockers


def test_cpu_validator_receipt_and_missing_h100_never_become_ready(tmp_path):
    manifest = _valid_manifest(tmp_path / "inputs")
    release = _release_dir(tmp_path, manifest)
    _write_ready_receipts(release, manifest)
    validator = release / "receipts" / "validator" / "task1.json"
    payload = json.loads(validator.read_text())
    payload["structural_only"] = True
    validator.write_text(json.dumps(payload))
    (release / "receipts" / "qualification" / "task2.json").unlink()
    blockers = rail.readiness_blockers(release, manifest)
    assert "task1:official_gpu_validation_not_verified" in blockers
    assert "task2:h100_qualification_not_verified" in blockers


def test_finalization_rejects_extra_or_missing_task_sifs(tmp_path):
    manifest = _valid_manifest(tmp_path / "inputs")
    release = _release_dir(tmp_path, manifest)
    _write_ready_receipts(release, manifest)
    (release / "images" / "unexpected.sif").write_bytes(b"second-image")
    _, blockers = rail.finalize_release(release)
    assert "release:expected_exactly_six_task_sifs" in blockers

    (release / "images" / "unexpected.sif").unlink()
    (release / "images" / "task4.sif").unlink()
    _, blockers = rail.finalize_release(release)
    assert "release:expected_exactly_six_task_sifs" in blockers


def test_finalization_rejects_task_receipts_with_wrong_hash_or_task_binding(tmp_path):
    manifest = _valid_manifest(tmp_path / "inputs")
    release = _release_dir(tmp_path, manifest)
    _write_ready_receipts(release, manifest)
    validator = release / "receipts" / "validator" / "task5.json"
    payload = json.loads(validator.read_text())
    payload["sif_sha256_after"] = "e" * 64
    validator.write_text(json.dumps(payload))
    _, blockers = rail.finalize_release(release)
    assert "task5:post_validation_sif_drift" in blockers

    _write_ready_receipts(release, manifest)
    payload = json.loads(validator.read_text())
    payload["task"] = "task3"
    validator.write_text(json.dumps(payload))
    _, blockers = rail.finalize_release(release)
    assert "task5:official_gpu_validation_not_verified" in blockers

    _write_ready_receipts(release, manifest)
    qualification = release / "receipts" / "qualification" / "task5.json"
    payload = json.loads(qualification.read_text())
    payload["task"] = "task3"
    qualification.write_text(json.dumps(payload))
    _, blockers = rail.finalize_release(release)
    assert "task5:qualification_task_binding_drift" in blockers


def test_manifest_rejects_universal_or_duplicate_sif_topology(tmp_path):
    manifest = _without_digest(_valid_manifest(tmp_path / "inputs"))
    manifest["container"] = {
        "cardinality": "one_sif_per_track",
        "entrypoint": "/app/predict.py",
        "images": {task: "submission.sif" for task in OFFICIAL_TASKS},
    }
    with pytest.raises(ReleaseContractError, match="universal multi-task SIF"):
        validate_release_manifest(manifest)

    manifest = _without_digest(_valid_manifest(tmp_path / "duplicates"))
    manifest["container"]["images"]["task5"] = "task3.sif"
    manifest["tasks"]["task5"]["submission_image"] = "task3.sif"
    with pytest.raises(ReleaseContractError, match="distinct submission SIF"):
        validate_release_manifest(manifest)


def test_finalization_rejects_mixed_pretrained_build_lineage_and_universal_shape(tmp_path):
    manifest = _valid_manifest(tmp_path / "inputs")
    release = _release_dir(tmp_path, manifest)
    _write_ready_receipts(release, manifest)
    build = release / "receipts" / "build" / "task2.json"
    payload = json.loads(build.read_text())
    payload["pretrained_checkpoint_sha256"] = "e" * 64
    build.write_text(json.dumps(payload))
    _, blockers = rail.finalize_release(release)
    assert "task2:missing_stale_or_mixed_lineage_build_receipt" in blockers

    shutil.rmtree(release / "images")
    (release / "images").mkdir()
    (release / "images" / "submission.sif").write_bytes(b"universal")
    _, blockers = rail.finalize_release(release)
    assert "release:expected_exactly_six_task_sifs" in blockers


def test_run_local_receipts_are_per_invocation_and_idempotent(tmp_path, monkeypatch):
    manifest = _valid_manifest(tmp_path / "inputs")
    release = _release_dir(tmp_path, manifest)
    image = release / "images" / "task1.sif"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(rail.subprocess, "run", fake_run)
    first = rail.run_local(release, task="task1", arguments=["--output", "/tmp/result.txt"])
    second = rail.run_local(release, task="task1", arguments=["--output", "/tmp/result.txt"])
    third = rail.run_local(release, task="task1", arguments=["--output", "/tmp/other.txt"])
    assert first["status"] == "PASS"
    assert second["idempotent"] is True
    assert third["status"] == "PASS"
    assert len(calls) == 2
    assert len(list((release / "receipts" / "runtime").glob("task1-*.json"))) == 2


def test_qualification_is_dry_run_by_default_and_never_invokes_sbatch(tmp_path, monkeypatch):
    manifest = _valid_manifest(tmp_path / "inputs")
    release = _release_dir(tmp_path, manifest)
    _write_ready_receipts(release, manifest)
    monkeypatch.setattr(rail, "verify_fixtures", lambda cache, root=None: {"status": "VERIFIED", "fixture_digest": "f" * 64})
    result = rail.qualify_release(
        release,
        host="jeanzay",
        remote_dir="$SCRATCH/fomo26/releases/release-a/qualification",
        fixture_cache=tmp_path,
    )
    assert result["status"] == "DRY_RUN"
    assert result["bash_syntax"] == "PASS"
    assert result["image_names"] == {task: f"{task}.sif" for task in OFFICIAL_TASKS}
    assert len([command for command in result["commands"] if "idrcontmgr cp" in command]) == 6
    assert all("sbatch" not in command or command.startswith("ssh jeanzay sbatch") for command in result["commands"])
    assert not list((release / "receipts").glob("qualification-submit-*.json"))
    assert (
        'python3 "${SCRATCH}/fomo26/releases/release-a/qualification"/qualification_runner.py'
        in (release / "qualification" / "qualify.slurm").read_text()
    )


def test_generated_qualification_script_is_valid_bash(tmp_path):
    script = rail._qualification_script(
        "$SCRATCH/fomo26/release",
        110,
        release_manifest_sha="a" * 64,
        validator_digest="b" * 64,
        fixture_digest="c" * 64,
        image_digests={task: "d" * 64 for task in OFFICIAL_TASKS},
        image_names={task: f"{task}.sif" for task in OFFICIAL_TASKS},
    )
    path = tmp_path / "qualify.slurm"
    path.write_text(script)
    proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "set -e" not in script
    assert script.count("then fomo26_qualification_rc=1; fi") == 6
    assert "task3.sif" in script
    assert "task5.sif" in script


def test_explicit_qualification_submit_uses_fake_ssh_idrcontmgr_and_slurm(tmp_path, monkeypatch):
    # The validator is external: staging it needs a resolvable root, not a path in this repository.
    staged_validator = tmp_path / "external-validator"
    staged_validator.mkdir()
    monkeypatch.setattr(rail, "validator_root", lambda root=None: staged_validator)
    manifest = _valid_manifest(tmp_path / "inputs")
    release = _release_dir(tmp_path, manifest)
    _write_ready_receipts(release, manifest)
    monkeypatch.setattr(rail, "verify_fixtures", lambda cache, root=None: {"status": "VERIFIED", "fixture_digest": "f" * 64})
    commands = []

    def fake_run(command, *, attempts, timeout):
        commands.append(command)
        stdout = "734291\n" if "sbatch" in command else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(rail, "_run_retries", fake_run)
    result = rail.qualify_release(
        release,
        host="fake-jeanzay",
        remote_dir="/scratch/fomo26/release-a/qualification",
        fixture_cache=tmp_path,
        submit=True,
    )
    assert result["status"] == "SUBMITTED"
    assert result["job_id"] == "734291"
    assert len([command for command in commands if "idrcontmgr" in command]) == 6
    assert len([command for command in commands if "sbatch" in command]) == 1
    second = rail.qualify_release(
        release,
        host="fake-jeanzay",
        remote_dir="/scratch/fomo26/release-a/qualification",
        fixture_cache=tmp_path,
        submit=True,
    )
    assert second["idempotent"] is True
    assert len([command for command in commands if "sbatch" in command]) == 1


def test_qualification_collection_verifies_job_and_artifact_bindings(tmp_path, monkeypatch):
    manifest = _valid_manifest(tmp_path / "inputs")
    release = _release_dir(tmp_path, manifest)
    _write_ready_receipts(release, manifest)
    fixture_digest = "f" * 64
    monkeypatch.setattr(
        rail, "verify_fixtures", lambda cache, root=None: {"status": "VERIFIED", "fixture_digest": fixture_digest}
    )

    def fake_collect(command, *, attempts, timeout):
        destination = Path(command[-1])
        destination.mkdir(parents=True, exist_ok=True)
        for task in OFFICIAL_TASKS:
            image_digest = sha256_file(release / "images" / f"{task}.sif")
            payload = {
                "schema_version": "fomo26-h100-qualification-receipt-v1",
                "status": "PASS",
                "task": task,
                "slurm_job_id": "734291",
                "release_manifest_sha256": manifest["manifest_sha256"],
                "validator_sha256": bridge.validator_sha256(),
                "fixture_digest": fixture_digest,
                "sif_sha256_before": image_digest,
                "sif_sha256_after": image_digest,
                "threshold_seconds": manifest["protocol"]["qualification_limit_seconds"],
            }
            (destination / f"{task}.json").write_text(json.dumps(payload))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    qualification = release / "receipts" / "qualification"
    shutil.rmtree(qualification)
    (release / "receipts" / "qualification-collect-734291.json").unlink()
    monkeypatch.setattr(rail, "_run_retries", fake_collect)
    result = rail.qualify_release(
        release,
        host="fake-jeanzay",
        remote_dir="/scratch/fomo26/release-a/qualification",
        fixture_cache=tmp_path,
        collect=True,
        job_id="734291",
    )
    assert result["status"] == "PASS"
    assert all((qualification / f"{task}.json").is_file() for task in OFFICIAL_TASKS)
    second = rail.qualify_release(
        release,
        host="fake-jeanzay",
        remote_dir="/scratch/fomo26/release-a/qualification",
        fixture_cache=tmp_path,
        collect=True,
        job_id="734291",
    )
    assert second["idempotent"] is True
