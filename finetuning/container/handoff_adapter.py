"""Deterministic adapter from the immutable A/B scientific handoff to the release contract.

The adapter never selects a candidate, checkpoint, or inference policy. It accepts one named
candidate, maps only paths already declared by the handoff into a verified local backup, and freezes
the existing production predictor defaults as explicit release policy.
"""

from __future__ import annotations

import hashlib
import json
import re
import yaml
from finetuning.container.official_contract import OFFICIAL_TASKS
from finetuning.container.release_manifest import (
    DOWNSTREAM_TASKS,
    ReleaseContractError,
    canonical_json_bytes,
    finalized_manifest,
    sha256_file,
    validate_release_manifest,
)
from finetuning.fomo26_inference.backbones import canonical_architecture
from pathlib import Path
from typing import Any

HANDOFF_SCHEMA = "fomo26-docker-handoff-v1"
BACKUP_SCHEMA = "fomo26-jz-local-backup-manifest-v1"
DERIVATION_SCHEMA = "fomo26-release-derivation-v1"
CLASSIFICATIONS = {
    "MECHANICALLY_DERIVABLE",
    "RUNTIME_CONTRACT_DERIVABLE",
    "NOT_APPLICABLE",
    "TRUE_SCIENTIFIC_AUTHORITY_MISSING",
}
REPO = Path(__file__).resolve().parents[2]


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseContractError(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleaseContractError(f"{label} must be a JSON object: {path}.")
    return payload


def _source(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _source_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO).as_posix()
    except ValueError:
        return str(resolved)


def _derived(value: Any, classification: str, source: Path, rule: str) -> dict[str, Any]:
    if classification not in CLASSIFICATIONS:
        raise AssertionError(classification)
    return {
        "value": value,
        "classification": classification,
        "source_path": _source_path(source),
        "source_sha256": sha256_file(source),
        "derivation_rule": rule,
    }


def _runtime_default(path: Path, environment_name: str) -> str:
    pattern = re.compile(rf'os\.environ\.get\("{re.escape(environment_name)}",\s*"([^"]+)"\)')
    match = pattern.search(path.read_text())
    if match is None:
        raise ReleaseContractError(f"Production runtime does not declare a default for {environment_name} in {path}.")
    return match.group(1)


def _declared_tta(authority: dict, task: str, fallback: str) -> str:
    """The TTA a task actually runs: the handoff's declaration, else the wrapper default.

    ``auto`` lets the time-budget governor pick the highest admissible level, which is right when
    the scientific authority delegates the choice and wrong when it froze one. A candidate whose
    handoff says task3 runs flip3 and task2 runs none must get exactly that, so a declared value is
    validated against the task's registry ladder and then frozen.
    """
    declared = authority.get("tta")
    if declared is None:
        return fallback
    if not isinstance(declared, str):
        raise ReleaseContractError(f"{task} handoff declares a non-string TTA {declared!r}.")
    if declared != "auto":
        # Reuse the registry ladder rather than restating which levels a task may use.
        from finetuning.fomo26_inference.tta_safety import assert_tta_within_ladder

        assert_tta_within_ladder(task.removeprefix("task"), declared)
    return declared


def _write_json_idempotent(path: Path, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.is_file() and path.read_text() == data:
            return
        raise ReleaseContractError(f"Refusing to overwrite conflicting adapter output {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data)


class _Backup:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.manifest_path = self.root / "BACKUP_MANIFEST.json"
        self.manifest = _load_json(self.manifest_path, "backup manifest")
        if self.manifest.get("schema_version") != BACKUP_SCHEMA:
            raise ReleaseContractError(f"Unsupported backup schema in {self.manifest_path}.")
        if self.manifest.get("summary", {}).get("verified") is not True:
            raise ReleaseContractError("Local backup manifest is not VERIFIED.")
        records = self.manifest.get("records")
        if not isinstance(records, list):
            raise ReleaseContractError("Local backup manifest records must be a list.")
        self.by_remote: dict[str, dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("original_jz_path"), str):
                raise ReleaseContractError("Malformed local backup record.")
            remote = record["original_jz_path"]
            if remote in self.by_remote:
                raise ReleaseContractError(f"Duplicate remote path in local backup manifest: {remote}.")
            self.by_remote[remote] = record

    def artifact(self, remote_path: str, *, expected_sha256: str | None = None) -> dict[str, str]:
        record = self.by_remote.get(remote_path)
        if record is None:
            raise ReleaseContractError(f"Handoff path is absent from verified local backup: {remote_path}.")
        local = Path(str(record.get("local_path", ""))).resolve()
        if self.root not in local.parents or not local.is_file():
            raise ReleaseContractError(f"Backed-up artifact is missing or escapes the backup root: {local}.")
        recorded = record.get("sha256")
        if not isinstance(recorded, str) or not re.fullmatch(r"[0-9a-f]{64}", recorded):
            raise ReleaseContractError(f"Backup record has no valid SHA-256 for {remote_path}.")
        observed = sha256_file(local)
        if observed != recorded:
            raise ReleaseContractError(f"Backed-up artifact drifted: {local}; expected {recorded}, observed {observed}.")
        if expected_sha256 is not None and observed != expected_sha256:
            raise ReleaseContractError(
                f"Handoff/backup SHA mismatch for {remote_path}: expected {expected_sha256}, observed {observed}."
            )
        return {"path": str(local), "sha256": observed}

    def record(self, *, role: str, candidate: str | None, task: int | None = None) -> dict[str, Any]:
        matches = [
            record
            for record in self.manifest["records"]
            if record.get("role") == role and record.get("candidate") == candidate and record.get("task") == task
        ]
        if len(matches) != 1:
            raise ReleaseContractError(
                f"Expected one backup record for role={role}, candidate={candidate}, task={task}; found {len(matches)}."
            )
        return matches[0]


def adapt_handoff(
    *,
    handoff_path: str | Path,
    expected_handoff_sha256: str,
    backup_root: str | Path,
    candidate_id: str,
    packaging_git_commit: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Translate one explicitly named handoff candidate into a fully bound release manifest."""
    handoff_path = Path(handoff_path).resolve()
    observed_handoff_sha = sha256_file(handoff_path)
    if observed_handoff_sha != expected_handoff_sha256:
        raise ReleaseContractError(
            f"Scientific handoff SHA mismatch: expected {expected_handoff_sha256}, observed {observed_handoff_sha}."
        )
    handoff = _load_json(handoff_path, "scientific handoff")
    if handoff.get("schema_version") != HANDOFF_SCHEMA:
        raise ReleaseContractError(f"Unsupported scientific handoff schema in {handoff_path}.")
    candidates = handoff.get("candidates")
    if not isinstance(candidates, dict) or candidate_id not in candidates:
        raise ReleaseContractError(f"Candidate {candidate_id!r} is not declared by the handoff.")
    selected = candidates[candidate_id]
    if not isinstance(selected, dict) or selected.get("candidate") != candidate_id:
        raise ReleaseContractError(f"Malformed handoff candidate {candidate_id!r}.")

    backup = _Backup(Path(backup_root))
    scientific_architecture = str(selected.get("architecture", ""))
    architecture = canonical_architecture(scientific_architecture)
    if not scientific_architecture or (scientific_architecture == "resenc_unet_b" and architecture != "resenc_b"):
        raise ReleaseContractError(f"No explicit runtime architecture alias exists for {scientific_architecture!r}.")

    artifacts = selected.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get("tasks"), dict):
        raise ReleaseContractError(f"Candidate {candidate_id!r} has no authoritative artifact set.")
    pretrained_authority = artifacts.get("pretrained")
    if not isinstance(pretrained_authority, dict):
        raise ReleaseContractError(f"Candidate {candidate_id!r} has no pretrained authority.")
    pretrained = backup.artifact(
        str(pretrained_authority.get("path", "")),
        expected_sha256=str(pretrained_authority.get("sha256", "")),
    )
    pretrained["step"] = pretrained_authority.get("global_step")

    # A frozen campaign is the preferred evidence for track and objective, but not every
    # pretraining lineage produced one: the salvage lineages predate it. Where it is absent the
    # same three facts must be declared by the scientific handoff instead, and the manifest records
    # which of the two established them. Nothing is inferred from the checkpoint path.
    frozen_campaign_path = str(pretrained_authority.get("frozen_campaign", "")).strip()
    if frozen_campaign_path:
        campaign = backup.artifact(frozen_campaign_path)
        campaign_payload = _load_json(Path(campaign["path"]), "frozen campaign")
        lane_matches = [
            lane
            for lane in campaign_payload.get("lanes", [])
            if isinstance(lane, dict) and lane.get("candidate") == candidate_id
        ]
        if len(lane_matches) != 1:
            raise ReleaseContractError(f"Frozen campaign does not carry exactly one lane for {candidate_id}.")
        lane = lane_matches[0]
        if lane.get("architecture") != scientific_architecture:
            raise ReleaseContractError("Handoff and frozen campaign carry different scientific architectures.")
        dataset_name = str(campaign_payload.get("dataset", {}).get("name", ""))
        ssl_objective = str(lane.get("ssl_objective", ""))
        if not ssl_objective:
            raise ReleaseContractError("Frozen campaign lane has no SSL objective.")
        campaign_evidence: Path | str = Path(campaign["path"])
        campaign_basis = "frozen campaign"
    else:
        campaign = None
        dataset_name = str(pretrained_authority.get("pretraining_dataset", ""))
        ssl_objective = str(pretrained_authority.get("ssl_objective", ""))
        if not ssl_objective:
            raise ReleaseContractError(
                "Pretrained authority declares no frozen campaign and no ssl_objective; refusing to guess it."
            )
        if pretrained_authority.get("architecture") != scientific_architecture:
            raise ReleaseContractError("Handoff pretrained authority and candidate carry different architectures.")
        campaign_evidence = handoff_path
        campaign_basis = "scientific handoff (this pretraining lineage emitted no frozen campaign)"
    if not dataset_name.startswith("FOMO300K"):
        raise ReleaseContractError(
            f"Cannot derive Methods track: pretraining dataset {dataset_name!r} is not restricted to FOMO300K."
        )
    track = "methods"

    pretrained_git_record = backup.record(role="pretrained_git_commit", candidate=candidate_id)
    pretrained_git_artifact = backup.artifact(pretrained_git_record["original_jz_path"])
    pretrained_training_git_commit = Path(pretrained_git_artifact["path"]).read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", pretrained_training_git_commit):
        raise ReleaseContractError("Pretrained git_commit.txt does not contain a full commit SHA.")

    protocol_record = backup.record(role="downstream_protocol", candidate=None)
    protocol_artifact = backup.artifact(protocol_record["original_jz_path"])
    protocol_payload = yaml.safe_load(Path(protocol_artifact["path"]).read_text())
    if not isinstance(protocol_payload, dict):
        raise ReleaseContractError("Downstream protocol must be a YAML mapping.")

    split_artifacts: dict[str, dict[str, str]] = {}
    split_hashes: dict[str, str] = {}
    for task_index, task in enumerate(DOWNSTREAM_TASKS, start=1):
        split_record = backup.record(role="selected_split", candidate=None, task=task_index)
        split_artifacts[task] = backup.artifact(split_record["original_jz_path"])
        split_hashes[task] = split_artifacts[task]["sha256"]
    split_digest = hashlib.sha256(canonical_json_bytes(split_hashes)).hexdigest()

    wrapper_paths = {task: REPO / "finetuning" / "container" / f"predict_{task}.py" for task in DOWNSTREAM_TASKS}
    tta_values = {task: _runtime_default(path, "FOMO26_TTA") for task, path in wrapper_paths.items()}
    time_values = {task: float(_runtime_default(path, "FOMO26_TIME_TARGET_S")) for task, path in wrapper_paths.items()}
    if len(set(time_values.values())) != 1:
        raise ReleaseContractError("Production task wrappers do not share one explicit time contract.")
    # The wrapper default is the fallback, not the authority. A scientific handoff may freeze a
    # different TTA per task -- one candidate legitimately wants flip3 on task3 and none elsewhere
    # -- and a contract that can only express a single shared value cannot represent that policy.
    # Requiring the fallbacks to agree still catches a wrapper drifting on its own.
    if len(set(tta_values.values())) != 1:
        raise ReleaseContractError("Production task wrappers do not share one explicit TTA fallback.")
    qualification_limit = next(iter(time_values.values()))
    if not 0 < qualification_limit < 120:
        raise ReleaseContractError("Production runtime target is not strictly below the official 120-second limit.")

    tasks: dict[str, Any] = {}
    derivations: dict[str, Any] = {
        "track": _derived(
            track,
            "MECHANICALLY_DERIVABLE",
            campaign_evidence,
            f"The {campaign_basis} declares only FOMO300K for candidate pretraining; official Track 1 is Methods.",
        ),
        "ssl_objective": _derived(
            ssl_objective,
            "MECHANICALLY_DERIVABLE",
            campaign_evidence,
            f"Read the SSL objective the {campaign_basis} declares for {candidate_id}.",
        ),
        "protocol.split_sha256": _derived(
            split_digest,
            "MECHANICALLY_DERIVABLE",
            backup.manifest_path,
            "SHA256(canonical JSON task-to-verified-split-SHA256 mapping).",
        ),
        "protocol.protocol_sha256": _derived(
            protocol_artifact["sha256"],
            "MECHANICALLY_DERIVABLE",
            Path(protocol_artifact["path"]),
            "SHA256 of the declared downstream_policy_v2.yaml bytes.",
        ),
        "protocol.qualification_limit_seconds": _derived(
            qualification_limit,
            "RUNTIME_CONTRACT_DERIVABLE",
            wrapper_paths["task1"],
            "Common production FOMO26_TIME_TARGET_S default; verified equal across Tasks 1-5 and below 120.",
        ),
    }

    checkpoint_sources: set[str] = set()
    handoff_tasks = artifacts["tasks"]
    if set(handoff_tasks) != {str(index) for index in range(1, 6)}:
        raise ReleaseContractError("Handoff must declare exactly Tasks 1-5 for the selected model set.")
    for task_index, task in enumerate(DOWNSTREAM_TASKS, start=1):
        authority = handoff_tasks[str(task_index)]
        if not isinstance(authority, dict) or authority.get("task") != task_index:
            raise ReleaseContractError(f"Malformed handoff authority for {task}.")
        fold = authority.get("fold")
        if isinstance(fold, bool) or not isinstance(fold, int):
            raise ReleaseContractError(f"Handoff {task} has no explicit selected fold.")
        if authority.get("loading_contract", {}).get("architecture") != scientific_architecture:
            raise ReleaseContractError(f"Handoff {task} carries a mixed scientific architecture.")
        if authority.get("parent_pretrained_sha256") != pretrained["sha256"]:
            raise ReleaseContractError(f"Handoff {task} carries a mixed pretrained lineage.")

        checkpoint = backup.artifact(authority["deployment_checkpoint"], expected_sha256=authority["deployment_sha256"])
        run_manifest = backup.artifact(authority["run_manifest"])
        hydra_config = backup.artifact(authority["resolved_config"])
        run_payload = _load_json(Path(run_manifest["path"]), f"{task} run manifest")
        config_payload = yaml.safe_load(Path(hydra_config["path"]).read_text())
        if not isinstance(config_payload, dict):
            raise ReleaseContractError(f"{task} resolved config must be a YAML mapping.")
        checkpoint_source = config_payload.get("pretrained", {}).get("source")
        if not isinstance(checkpoint_source, str):
            raise ReleaseContractError(f"{task} resolved config has no pretrained.source.")
        checkpoint_sources.add(checkpoint_source)
        downstream_run_git_commit = str(run_payload.get("git_commit", ""))
        if downstream_run_git_commit != authority.get("git_commit"):
            raise ReleaseContractError(f"{task} handoff/run-manifest Git commit mismatch.")

        members = [fold]
        policy: dict[str, Any] = {
            "ensemble_members": members,
            "member_selection": None,
            "ensemble_method": None,
            "tta": _declared_tta(authority, task, tta_values[task]),
            "time_target_seconds": time_values[task],
            "calibration": {"state": "none"},
        }
        if task in {"task1", "task3", "task5"}:
            calibration_required = _runtime_default(wrapper_paths[task], "FOMO26_CALIBRATION_REQUIRED")
            if calibration_required != "0":
                raise ReleaseContractError(
                    f"{task} production runtime requires calibration but the handoff declares no artifact."
                )
        derivations[f"tasks.{task}.policy.ensemble_members"] = _derived(
            members,
            "MECHANICALLY_DERIVABLE",
            handoff_path,
            f"Use exactly the fold selected at candidates.{candidate_id}.artifacts.tasks.{task_index}.fold.",
        )
        for field in ("member_selection", "ensemble_method"):
            derivations[f"tasks.{task}.policy.{field}"] = _derived(
                None,
                "NOT_APPLICABLE",
                handoff_path,
                "Inter-member selection/aggregation is inapplicable because exactly one member is selected.",
            )
        declared_tta = authority.get("tta")
        derivations[f"tasks.{task}.policy.tta"] = _derived(
            policy["tta"],
            "MECHANICALLY_DERIVABLE" if declared_tta else "RUNTIME_CONTRACT_DERIVABLE",
            handoff_path if declared_tta else wrapper_paths[task],
            (
                f"Use exactly the TTA declared at candidates.{candidate_id}.artifacts.tasks.{task_index}.tta."
                if declared_tta
                else "Read the production FOMO26_TTA default and freeze it explicitly."
            ),
        )
        derivations[f"tasks.{task}.policy.time_target_seconds"] = _derived(
            policy["time_target_seconds"],
            "RUNTIME_CONTRACT_DERIVABLE",
            wrapper_paths[task],
            "Read the production FOMO26_TIME_TARGET_S default and freeze it explicitly.",
        )
        calibration_source = (
            wrapper_paths[task]
            if task in {"task1", "task3", "task5"}
            else REPO / "finetuning" / "container" / "fomo_ensemble_predict.py"
        )
        derivations[f"tasks.{task}.policy.calibration"] = _derived(
            policy["calibration"],
            "RUNTIME_CONTRACT_DERIVABLE",
            calibration_source,
            "No calibration artifact is declared; production runtime does not require one.",
        )
        if task in {"task1", "task3", "task5"}:
            policy["cross_patch"] = _runtime_default(wrapper_paths[task], "FOMO26_CROSS_PATCH")
            derivations[f"tasks.{task}.policy.cross_patch"] = _derived(
                policy["cross_patch"],
                "RUNTIME_CONTRACT_DERIVABLE",
                wrapper_paths[task],
                "Read the production FOMO26_CROSS_PATCH default and freeze it explicitly.",
            )
        else:
            policy["window_policy"] = _runtime_default(wrapper_paths[task], "FOMO26_WINDOW_POLICY")
            policy["ensemble_space"] = _runtime_default(wrapper_paths[task], "FOMO26_ENSEMBLE_SPACE")
            for field, environment in (
                ("window_policy", "FOMO26_WINDOW_POLICY"),
                ("ensemble_space", "FOMO26_ENSEMBLE_SPACE"),
            ):
                derivations[f"tasks.{task}.policy.{field}"] = _derived(
                    policy[field],
                    "RUNTIME_CONTRACT_DERIVABLE",
                    wrapper_paths[task],
                    f"Read the production {environment} default and freeze it explicitly.",
                )

        tasks[task] = {
            "candidate_id": candidate_id,
            "architecture": architecture,
            "pretrained_sha256": pretrained["sha256"],
            "submission_image": f"{task}.sif",
            "selected_members": members,
            "folds": [
                {
                    "fold": fold,
                    "checkpoint": checkpoint,
                    "run_manifest": run_manifest,
                    "hydra_config": hydra_config,
                    "downstream_run_git_commit": downstream_run_git_commit,
                    "source_run_dir": authority["run_dir"],
                }
            ],
            "provenance": {
                "source_run_directories": [authority["run_dir"]],
                "task_split_sha256": split_hashes[task],
                "split_sha256": split_digest,
                "protocol_sha256": protocol_artifact["sha256"],
            },
            "policy": policy,
        }

    if len(checkpoint_sources) != 1:
        raise ReleaseContractError(f"Selected task configs carry mixed checkpoint sources: {sorted(checkpoint_sources)}.")
    checkpoint_source = checkpoint_sources.pop()
    derivations["checkpoint_source"] = _derived(
        checkpoint_source,
        "RUNTIME_CONTRACT_DERIVABLE",
        Path(tasks["task1"]["folds"][0]["hydra_config"]["path"]),
        "Require the resolved pretrained.source to be identical across all five selected task configs.",
    )

    task6_wrapper = REPO / "finetuning" / "container" / "predict_task6_7.py"
    task6_authority = selected.get("tasks_6_7")
    if (
        not isinstance(task6_authority, dict)
        or task6_authority.get("architecture") != scientific_architecture
        or task6_authority.get("pretrained_sha256") != pretrained["sha256"]
        or task6_authority.get("pretrained_path") != pretrained_authority.get("path")
    ):
        raise ReleaseContractError("Tasks 6/7 authority is missing or conflicts with the common pretrained lineage.")
    patch_size = [int(value) for value in _runtime_default(task6_wrapper, "FOMO26_PATCH_SIZE").split(",")]
    minimum_coverage = float(_runtime_default(task6_wrapper, "FOMO26_MIN_ENCODER_COVERAGE"))
    if float(protocol_payload.get("minimum_encoder_loading_coverage", -1)) != minimum_coverage:
        raise ReleaseContractError("Task 6/7 runtime and downstream protocol disagree on encoder coverage.")
    tasks["task6_and_7"] = {
        "candidate_id": candidate_id,
        "architecture": architecture,
        "pretrained_sha256": pretrained["sha256"],
        "submission_image": "task6_and_7.sif",
        "downstream_weights": [],
        "policy": {"patch_size": patch_size, "minimum_encoder_coverage": minimum_coverage},
    }
    derivations["tasks.task6_and_7.policy.patch_size"] = _derived(
        patch_size,
        "RUNTIME_CONTRACT_DERIVABLE",
        task6_wrapper,
        "Read the production FOMO26_PATCH_SIZE default and freeze it explicitly.",
    )
    derivations["tasks.task6_and_7.policy.minimum_encoder_coverage"] = _derived(
        minimum_coverage,
        "RUNTIME_CONTRACT_DERIVABLE",
        task6_wrapper,
        "Read the production threshold and require equality with downstream_policy_v2.yaml.",
    )

    manifest = finalized_manifest(
        {
            "schema_version": "fomo26-release-manifest-v1",
            "release_mode": "final",
            "candidate_id": candidate_id,
            "track": track,
            "scientific_architecture": scientific_architecture,
            "architecture": architecture,
            "ssl_objective": ssl_objective,
            "checkpoint_source": checkpoint_source,
            "scientific_authority": {
                "path": str(handoff_path),
                "sha256": observed_handoff_sha,
                "schema_version": HANDOFF_SCHEMA,
                "candidate_id": candidate_id,
                "model_set_manifest_sha256": selected.get("_manifest_sha256"),
            },
            "pretrained": pretrained,
            "pretrained_training_git_commit": pretrained_training_git_commit,
            "packaging_git_commit": packaging_git_commit,
            "created_at": selected.get("created_utc") or handoff.get("created_utc"),
            "protocol": {
                "split_sha256": split_digest,
                "task_split_sha256": split_hashes,
                "protocol_sha256": protocol_artifact["sha256"],
                "qualification_limit_seconds": qualification_limit,
            },
            "container": {
                "cardinality": "one_sif_per_task_unit",
                "entrypoint": "/app/predict.py",
                "images": {task: f"{task}.sif" for task in OFFICIAL_TASKS},
            },
            "tasks": tasks,
            "derivation_receipt": {
                "schema_version": DERIVATION_SCHEMA,
                "authority_sha256": observed_handoff_sha,
                "backup_manifest": _source(backup.manifest_path),
                "candidate_id": candidate_id,
                "packaging_git_commit": packaging_git_commit,
                "fields": derivations,
                "unresolved_true_scientific_authority": [],
            },
        }
    )
    validate_release_manifest(manifest, verify_files=True)
    if output_path is not None:
        _write_json_idempotent(Path(output_path).resolve(), manifest)
    return manifest
