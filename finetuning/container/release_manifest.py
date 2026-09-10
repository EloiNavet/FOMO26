"""Strict manifest contract for reproducible FOMO26 container releases.

This module validates identity, provenance, fold completeness, frozen inference policy, and every
declared artifact digest.  It deliberately makes no scientific choice: it only freezes choices
that have already been finalized by the candidate owner.
"""

from __future__ import annotations

import hashlib
import json
import re
import yaml
from datetime import datetime
from finetuning.container.official_contract import OFFICIAL_TASKS
from finetuning.fomo26_inference.backbones import canonical_architecture, known_architectures
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "fomo26-release-manifest-v1"
DOWNSTREAM_TASKS = OFFICIAL_TASKS[:5]
TASK_CONTAINER_CARDINALITY = "one_sif_per_task_unit"
TASK_ENTRYPOINT = "/app/predict.py"
TASK_IMAGE_NAMES = {task: f"{task}.sif" for task in OFFICIAL_TASKS}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
ARCHITECTURE_MARKERS = {
    "resenc_b": ("resenc_unet_b", "resenc_b"),
    "unet_m": ("unet_m",),
}


class ReleaseContractError(ValueError):
    """The release manifest is incomplete, inconsistent, or has drifted."""


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def manifest_digest(payload: dict[str, Any]) -> str:
    digest_payload = dict(payload)
    digest_payload.pop("manifest_sha256", None)
    return hashlib.sha256(canonical_json_bytes(digest_payload)).hexdigest()


def load_manifest(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseContractError(f"Cannot read release manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleaseContractError("Release manifest must be a JSON object.")
    return payload


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReleaseContractError(f"{label} must be an object.")
    return value


def _require_nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReleaseContractError(f"{label} must be a non-empty string.")
    return value


def _require_sha(value: Any, label: str) -> str:
    value = _require_nonempty_text(value, label)
    if not SHA256.fullmatch(value):
        raise ReleaseContractError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _require_git_sha(value: Any, label: str) -> str:
    value = _require_nonempty_text(value, label)
    if not GIT_SHA.fullmatch(value):
        raise ReleaseContractError(f"{label} must be a full lowercase Git commit SHA.")
    return value


def _require_safe_id(value: Any, label: str) -> str:
    value = _require_nonempty_text(value, label)
    if not SAFE_ID.fullmatch(value) or ".." in value:
        raise ReleaseContractError(f"{label} contains unsafe characters: {value!r}.")
    return value


def _artifact(spec: Any, label: str, *, verify_files: bool, base_dir: Path | None = None) -> dict[str, Any]:
    artifact = _require_mapping(spec, label)
    path_text = _require_nonempty_text(artifact.get("path"), f"{label}.path")
    expected = _require_sha(artifact.get("sha256"), f"{label}.sha256")
    path = Path(path_text)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    if verify_files:
        if not path.is_file():
            raise ReleaseContractError(f"{label} is missing: {path}.")
        observed = sha256_file(path)
        if observed != expected:
            raise ReleaseContractError(f"{label} drifted: expected {expected}, observed {observed} at {path}.")
    return artifact


def _validate_created_at(value: Any) -> None:
    text = _require_nonempty_text(value, "created_at")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReleaseContractError("created_at must be an ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None:
        raise ReleaseContractError("created_at must include a timezone.")


def _validate_run_manifest_identity(
    artifact: dict[str, Any],
    *,
    label: str,
    candidate: str,
    architecture: str,
    scientific_architecture: str,
    pretrained_sha256: str,
    downstream_run_git_commit: str,
    fold: int,
    task: str,
    base_dir: Path | None,
) -> None:
    path = Path(artifact["path"])
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseContractError(f"{label} must be a readable JSON object: {exc}") from exc
    if not isinstance(record, dict):
        raise ReleaseContractError(f"{label} must be a JSON object.")
    required = {"fold", "fomo_task", "pretrained_checkpoint_sha256", "git_commit"}
    missing = sorted(required - record.keys())
    if missing:
        raise ReleaseContractError(f"{label} is missing required provenance fields: {missing}.")
    for key in ("candidate_id", "candidate"):
        if key in record and record[key] != candidate:
            raise ReleaseContractError(f"{label} carries conflicting {key}={record[key]!r}.")
    for key in ("architecture", "architecture_id"):
        if key in record and record[key] not in {architecture, scientific_architecture}:
            raise ReleaseContractError(f"{label} carries conflicting {key}={record[key]!r}.")
    if record["pretrained_checkpoint_sha256"] != pretrained_sha256:
        raise ReleaseContractError(f"{label} carries a conflicting pretrained checkpoint SHA.")
    expected_task = int(task.removeprefix("task"))
    if str(record["fomo_task"]) != str(expected_task):
        raise ReleaseContractError(f"{label} carries conflicting fomo_task={record['fomo_task']!r}.")
    if str(record["fold"]) != str(fold):
        raise ReleaseContractError(f"{label} carries conflicting fold={record['fold']!r}; expected {fold}.")
    if record["git_commit"] != downstream_run_git_commit:
        raise ReleaseContractError(f"{label} carries a conflicting downstream run git commit.")


def _validate_hydra_architecture(artifact: dict[str, Any], *, label: str, architecture: str, base_dir: Path | None) -> None:
    path = Path(artifact["path"])
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    try:
        document = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ReleaseContractError(f"{label} must be readable YAML: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("model"), dict):
        raise ReleaseContractError(f"{label} must contain a resolved model mapping.")
    model_text = json.dumps(document["model"], sort_keys=True).lower()
    observed = {
        name for name, markers in ARCHITECTURE_MARKERS.items() if any(marker.lower() in model_text for marker in markers)
    }
    if observed != {architecture}:
        raise ReleaseContractError(
            f"{label} model architecture evidence is {sorted(observed)}, expected exactly {architecture!r}."
        )


def _validate_policy(
    task: str,
    value: Any,
    present_folds: list[int],
    *,
    base_dir: Path | None,
) -> None:
    policy = _require_mapping(value, f"tasks.{task}.policy")
    required = {
        "ensemble_members",
        "member_selection",
        "ensemble_method",
        "tta",
        "time_target_seconds",
        "calibration",
    }
    required |= {"cross_patch"} if task in {"task1", "task3", "task5"} else {"window_policy", "ensemble_space"}
    missing = sorted(required - policy.keys())
    if missing:
        raise ReleaseContractError(f"tasks.{task}.policy is missing explicit choices: {missing}.")
    members = policy["ensemble_members"]
    if not isinstance(members, list) or any(not isinstance(member, int) for member in members):
        raise ReleaseContractError(f"tasks.{task}.policy.ensemble_members must be an integer list.")
    if members != present_folds:
        raise ReleaseContractError(
            f"tasks.{task}.policy.ensemble_members {members} does not match staged folds {present_folds}."
        )
    if policy["tta"] not in {"auto", "none", "flip3", "flip7"}:
        raise ReleaseContractError(f"tasks.{task}.policy.tta is unsupported.")
    if len(members) == 1:
        if policy["member_selection"] is not None or policy["ensemble_method"] is not None:
            raise ReleaseContractError(
                f"tasks.{task} single-member policy must mark member_selection and ensemble_method null."
            )
    else:
        if policy["ensemble_method"] != "mean":
            raise ReleaseContractError(f"tasks.{task}.policy.ensemble_method must be the implemented value 'mean'.")
        expected_selection = "time_budget_auto" if policy["tta"] == "auto" else "fixed"
        if policy["member_selection"] != expected_selection:
            raise ReleaseContractError(
                f"tasks.{task}.policy.member_selection must be {expected_selection!r} when tta={policy['tta']!r}."
            )
    target = policy["time_target_seconds"]
    if isinstance(target, bool) or not isinstance(target, int | float) or not (0 < float(target) < 120):
        raise ReleaseContractError(f"tasks.{task}.policy.time_target_seconds must be >0 and <120.")
    calibration = _require_mapping(policy["calibration"], f"tasks.{task}.policy.calibration")
    state = calibration.get("state")
    if state not in {"none", "required"}:
        raise ReleaseContractError(f"tasks.{task}.policy.calibration.state must be 'none' or 'required'.")
    if task in {"task2", "task4"} and state != "none":
        raise ReleaseContractError(f"tasks.{task} segmentation runtime supports only explicit calibration state 'none'.")
    if state == "required":
        _artifact(
            calibration.get("artifact"),
            f"tasks.{task}.policy.calibration.artifact",
            verify_files=False,
            base_dir=base_dir,
        )
    elif "artifact" in calibration and calibration["artifact"] is not None:
        raise ReleaseContractError(f"tasks.{task} calibration state is none but an artifact was supplied.")
    if task in {"task1", "task3", "task5"}:
        if policy["cross_patch"] not in {"none", "cross5", "cross9"}:
            raise ReleaseContractError(f"tasks.{task}.policy.cross_patch is unsupported.")
    else:
        if policy["window_policy"] != "checkpoint_config_overlap_0.5":
            raise ReleaseContractError(f"tasks.{task}.policy.window_policy must be 'checkpoint_config_overlap_0.5'.")
        if policy["ensemble_space"] not in {"prob", "logit"}:
            raise ReleaseContractError(f"tasks.{task}.policy.ensemble_space must be 'prob' or 'logit'.")


def _validate_task_containers(value: Any) -> dict[str, str]:
    container = _require_mapping(value, "container")
    if container.get("cardinality") != TASK_CONTAINER_CARDINALITY:
        raise ReleaseContractError(
            f"container.cardinality must be {TASK_CONTAINER_CARDINALITY!r}; "
            "a universal multi-task SIF is not a final submission artifact."
        )
    if container.get("entrypoint") != TASK_ENTRYPOINT:
        raise ReleaseContractError(f"container.entrypoint must be {TASK_ENTRYPOINT!r}.")
    images = _require_mapping(container.get("images"), "container.images")
    if set(images) != set(OFFICIAL_TASKS):
        raise ReleaseContractError(f"container.images must contain exactly {list(OFFICIAL_TASKS)}.")
    normalized: dict[str, str] = {}
    for task in OFFICIAL_TASKS:
        image_name = _require_nonempty_text(images.get(task), f"container.images.{task}")
        if Path(image_name).name != image_name or not image_name.endswith(".sif"):
            raise ReleaseContractError(f"container.images.{task} must be one safe .sif filename.")
        normalized[task] = image_name
    if len(set(normalized.values())) != len(OFFICIAL_TASKS):
        raise ReleaseContractError("Each official task unit must have one distinct submission SIF.")
    if normalized != TASK_IMAGE_NAMES:
        raise ReleaseContractError(f"container.images must be exactly {TASK_IMAGE_NAMES}.")
    return normalized


def validate_release_manifest(
    payload: dict[str, Any], *, verify_files: bool = True, base_dir: str | Path | None = None
) -> dict[str, Any]:
    """Validate and return ``payload``; raise :class:`ReleaseContractError` on any blocker."""
    base_dir = Path(base_dir) if base_dir is not None else None
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ReleaseContractError(f"schema_version must be {SCHEMA_VERSION!r}.")
    mode = payload.get("release_mode")
    if mode not in {"final", "smoke_only"}:
        raise ReleaseContractError("release_mode must be 'final' or 'smoke_only'.")
    candidate = _require_safe_id(payload.get("candidate_id"), "candidate_id")
    _require_safe_id(payload.get("track"), "track")
    if "release_id" in payload:
        _require_safe_id(payload.get("release_id"), "release_id")
    architecture = _require_nonempty_text(payload.get("architecture"), "architecture")
    if architecture not in known_architectures() or architecture not in ARCHITECTURE_MARKERS:
        raise ReleaseContractError(
            f"architecture {architecture!r} is not supported by the frozen runtime registry; "
            f"known: {sorted(ARCHITECTURE_MARKERS)}."
        )
    scientific_architecture = _require_nonempty_text(payload.get("scientific_architecture"), "scientific_architecture")
    if canonical_architecture(scientific_architecture) != architecture:
        raise ReleaseContractError(
            "scientific_architecture does not resolve to the declared runtime architecture: "
            f"{scientific_architecture!r} -> {canonical_architecture(scientific_architecture)!r}, "
            f"expected {architecture!r}."
        )
    _require_nonempty_text(payload.get("ssl_objective"), "ssl_objective")
    if payload.get("checkpoint_source") not in {"online", "ema", "ema_if_available"}:
        raise ReleaseContractError("checkpoint_source must be online, ema, or ema_if_available.")
    _require_git_sha(payload.get("pretrained_training_git_commit"), "pretrained_training_git_commit")
    _require_git_sha(payload.get("packaging_git_commit"), "packaging_git_commit")
    _validate_created_at(payload.get("created_at"))

    authority = _artifact(
        payload.get("scientific_authority"),
        "scientific_authority",
        verify_files=verify_files,
        base_dir=base_dir,
    )
    _require_nonempty_text(authority.get("schema_version"), "scientific_authority.schema_version")
    if authority.get("candidate_id") != candidate:
        raise ReleaseContractError("scientific_authority.candidate_id does not match candidate_id.")

    pretrained = _artifact(payload.get("pretrained"), "pretrained", verify_files=verify_files, base_dir=base_dir)
    step = pretrained.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ReleaseContractError("pretrained.step must be a non-negative integer.")
    common_pretrained_sha = pretrained["sha256"]

    protocol = _require_mapping(payload.get("protocol"), "protocol")
    split_digest = _require_sha(protocol.get("split_sha256"), "protocol.split_sha256")
    protocol_digest = _require_sha(protocol.get("protocol_sha256"), "protocol.protocol_sha256")
    qualification_limit = protocol.get("qualification_limit_seconds")
    if isinstance(qualification_limit, bool) or not isinstance(qualification_limit, int | float):
        raise ReleaseContractError("protocol.qualification_limit_seconds must be numeric.")
    if not (0 < float(qualification_limit) < 120):
        raise ReleaseContractError("protocol.qualification_limit_seconds must be >0 and strictly below 120.")

    task_images = _validate_task_containers(payload.get("container"))

    tasks = _require_mapping(payload.get("tasks"), "tasks")
    if set(tasks) != set(OFFICIAL_TASKS):
        raise ReleaseContractError(f"tasks must contain exactly {list(OFFICIAL_TASKS)}.")
    for task in DOWNSTREAM_TASKS:
        spec = _require_mapping(tasks[task], f"tasks.{task}")
        if spec.get("submission_image") != task_images[task]:
            raise ReleaseContractError(f"tasks.{task}.submission_image does not match container.images.{task}.")
        if spec.get("candidate_id") != candidate or spec.get("architecture") != architecture:
            raise ReleaseContractError(f"tasks.{task} has mixed candidate or architecture identity.")
        if spec.get("pretrained_sha256") != common_pretrained_sha:
            raise ReleaseContractError(f"tasks.{task} has a conflicting pretrained SHA.")
        provenance = _require_mapping(spec.get("provenance"), f"tasks.{task}.provenance")
        if provenance.get("split_sha256") != split_digest or provenance.get("protocol_sha256") != protocol_digest:
            raise ReleaseContractError(f"tasks.{task} has conflicting split/protocol provenance.")
        source_dirs = provenance.get("source_run_directories")
        if not isinstance(source_dirs, list) or not source_dirs or any(not str(item).strip() for item in source_dirs):
            raise ReleaseContractError(f"tasks.{task}.provenance.source_run_directories must be non-empty.")

        folds = spec.get("folds")
        if not isinstance(folds, list) or not folds:
            raise ReleaseContractError(f"tasks.{task}.folds must be a non-empty list.")
        by_fold: dict[int, dict[str, Any]] = {}
        for index, fold_spec in enumerate(folds):
            fold_spec = _require_mapping(fold_spec, f"tasks.{task}.folds[{index}]")
            fold = fold_spec.get("fold")
            if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(5):
                raise ReleaseContractError(f"tasks.{task}.folds[{index}].fold must be 0-4.")
            if fold in by_fold:
                raise ReleaseContractError(f"tasks.{task} contains duplicate fold {fold}.")
            by_fold[fold] = fold_spec
            downstream_run_git_commit = _require_git_sha(
                fold_spec.get("downstream_run_git_commit"),
                f"tasks.{task}.folds[{fold}].downstream_run_git_commit",
            )
            _artifact(
                fold_spec.get("checkpoint"),
                f"tasks.{task}.folds[{fold}].checkpoint",
                verify_files=verify_files,
                base_dir=base_dir,
            )
            _artifact(
                fold_spec.get("run_manifest"),
                f"tasks.{task}.folds[{fold}].run_manifest",
                verify_files=verify_files,
                base_dir=base_dir,
            )
            if verify_files:
                _validate_run_manifest_identity(
                    fold_spec["run_manifest"],
                    label=f"tasks.{task}.folds[{fold}].run_manifest",
                    candidate=candidate,
                    architecture=architecture,
                    scientific_architecture=scientific_architecture,
                    pretrained_sha256=common_pretrained_sha,
                    downstream_run_git_commit=downstream_run_git_commit,
                    fold=fold,
                    task=task,
                    base_dir=base_dir,
                )
            _artifact(
                fold_spec.get("hydra_config"),
                f"tasks.{task}.folds[{fold}].hydra_config",
                verify_files=verify_files,
                base_dir=base_dir,
            )
            if verify_files:
                _validate_hydra_architecture(
                    fold_spec["hydra_config"],
                    label=f"tasks.{task}.folds[{fold}].hydra_config",
                    architecture=architecture,
                    base_dir=base_dir,
                )
            _require_nonempty_text(fold_spec.get("source_run_dir"), f"tasks.{task}.folds[{fold}].source_run_dir")
        present = sorted(by_fold)
        selected_members = spec.get("selected_members")
        if (
            not isinstance(selected_members, list)
            or not selected_members
            or any(
                isinstance(member, bool) or not isinstance(member, int) or member not in range(5)
                for member in selected_members
            )
            or len(set(selected_members)) != len(selected_members)
            or selected_members != sorted(selected_members)
        ):
            raise ReleaseContractError(f"tasks.{task}.selected_members must be a sorted, unique, non-empty fold list.")
        declared_source_dirs = {str(item) for item in source_dirs}
        observed_source_dirs = {str(item["source_run_dir"]) for item in by_fold.values()}
        if declared_source_dirs != observed_source_dirs:
            raise ReleaseContractError(f"tasks.{task} source-run provenance does not match its fold records.")
        if present != selected_members:
            raise ReleaseContractError(
                f"tasks.{task} staged folds {present} do not match authoritative selected_members {selected_members}."
            )
        _validate_policy(task, spec.get("policy"), selected_members, base_dir=base_dir)
        calibration = spec["policy"]["calibration"]
        if calibration["state"] == "required":
            _artifact(
                calibration["artifact"],
                f"tasks.{task}.policy.calibration.artifact",
                verify_files=verify_files,
                base_dir=base_dir,
            )

    frozen = _require_mapping(tasks["task6_and_7"], "tasks.task6_and_7")
    if frozen.get("submission_image") != task_images["task6_and_7"]:
        raise ReleaseContractError("tasks.task6_and_7.submission_image does not match container.images.task6_and_7.")
    if frozen.get("candidate_id") != candidate or frozen.get("architecture") != architecture:
        raise ReleaseContractError("tasks.task6_and_7 has mixed candidate or architecture identity.")
    if frozen.get("pretrained_sha256") != common_pretrained_sha:
        raise ReleaseContractError("tasks.task6_and_7 has a conflicting pretrained SHA.")
    if frozen.get("downstream_weights") != []:
        raise ReleaseContractError("tasks.task6_and_7.downstream_weights must be an empty list.")
    policy = _require_mapping(frozen.get("policy"), "tasks.task6_and_7.policy")
    patch_size = policy.get("patch_size")
    if (
        not isinstance(patch_size, list)
        or len(patch_size) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in patch_size)
    ):
        raise ReleaseContractError("tasks.task6_and_7.policy.patch_size must contain three positive integers.")
    coverage = policy.get("minimum_encoder_coverage")
    if isinstance(coverage, bool) or not isinstance(coverage, int | float) or not 0 < float(coverage) <= 1:
        raise ReleaseContractError("tasks.task6_and_7.policy.minimum_encoder_coverage must be in (0, 1].")

    recorded_digest = payload.get("manifest_sha256")
    if recorded_digest is not None and recorded_digest != manifest_digest(payload):
        raise ReleaseContractError("release manifest digest does not match its canonical contents.")
    return payload


def finalized_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with its canonical immutable digest populated."""
    result = json.loads(json.dumps(payload))
    result["manifest_sha256"] = manifest_digest(result)
    return result
