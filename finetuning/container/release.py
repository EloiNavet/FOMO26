"""Resumable FOMO26 container release state machine.

The CLI never chooses a candidate or an inference policy.  It freezes and verifies declared
artifacts, stages deterministic runtime contexts, builds already-staged contexts, runs the pinned
official validator, optionally prepares Jean-Zay H100 qualification, and derives readiness solely
from immutable receipts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from finetuning.container.handoff_adapter import adapt_handoff
from finetuning.container.release_manifest import (
    DOWNSTREAM_TASKS,
    OFFICIAL_TASKS,
    ReleaseContractError,
    finalized_manifest,
    load_manifest,
    manifest_digest,
    sha256_file,
    validate_release_manifest,
)
from finetuning.container.validator_bridge import (
    ValidatorError,
    acquire_validator,
    bootstrap_fixtures,
    run_official_validator,
    upstream_head_drift,
    validator_root,
    validator_sha256,
    verify_entrypoint_contract,
    verify_fixtures,
    verify_manifest,
    verify_snapshot,
)
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
BASE_IMAGE = "pytorch/pytorch@sha256:27c3135420bc184e86977170b6158c6133be3c7cc5c35e9e4fa87bdda629dc2b"
RECEIPT_SCHEMA = "fomo26-release-receipt-v1"
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SAFE_REMOTE_PATH = re.compile(r"^(?:\$SCRATCH|/)[A-Za-z0-9_./-]+$")


class ReleaseError(RuntimeError):
    """A release phase cannot safely proceed or resume."""


def _image_path(release_dir: Path, payload: dict[str, Any], task: str) -> Path:
    return release_dir / "images" / payload["container"]["images"][task]


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReleaseError(f"Cannot resolve packaging Git commit: {exc}") from exc


def _git_is_clean() -> bool:
    proc = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0 and not proc.stdout.strip()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _write_bytes_immutable(path: Path, data: bytes) -> bool:
    """Write once; identical content is an idempotent success, any collision is fatal."""
    if path.exists():
        if path.is_file() and path.read_bytes() == data:
            return False
        raise ReleaseError(f"Immutable output collision at {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def _write_json_immutable(path: Path, payload: Any) -> bool:
    return _write_bytes_immutable(path, _json_bytes(payload))


def _replace_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _receipt(phase: str, **fields: Any) -> dict[str, Any]:
    return {"schema_version": RECEIPT_SCHEMA, "phase": phase, **fields}


def _safe_identifier(value: str, label: str) -> str:
    if not SAFE_IDENTIFIER.fullmatch(value) or ".." in value:
        raise ReleaseError(f"Unsafe {label} {value!r}.")
    return value


def _safe_remote_path(value: str) -> str:
    path_for_parts = value.replace("$SCRATCH", "/scratch")
    if not SAFE_REMOTE_PATH.fullmatch(value) or ".." in Path(path_for_parts).parts:
        raise ReleaseError(f"Unsafe remote path {value!r}.")
    return value.rstrip("/")


def _absolutize_artifact_paths(payload: dict[str, Any], base_dir: Path) -> None:
    specs = [payload.get("scientific_authority"), payload.get("pretrained")]
    for task in DOWNSTREAM_TASKS:
        task_spec = payload.get("tasks", {}).get(task, {})
        for fold in task_spec.get("folds", []):
            specs.extend(fold.get(key) for key in ("checkpoint", "run_manifest", "hydra_config"))
        calibration = task_spec.get("policy", {}).get("calibration", {})
        if calibration.get("state") == "required":
            specs.append(calibration.get("artifact"))
    for spec in specs:
        if not isinstance(spec, dict) or not isinstance(spec.get("path"), str):
            continue
        path = Path(spec["path"])
        if not path.is_absolute():
            spec["path"] = str((base_dir / path).resolve())


def _load_release(release_dir: str | Path, *, verify_files: bool = True) -> tuple[Path, dict[str, Any]]:
    release_dir = Path(release_dir).resolve()
    manifest_path = release_dir / "release_manifest.json"
    payload = load_manifest(manifest_path)
    validate_release_manifest(payload, verify_files=verify_files, base_dir=release_dir)
    return release_dir, payload


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _declared_files(release_dir: Path) -> list[Path]:
    paths = [release_dir / "release_manifest.json"]
    artifacts = release_dir / "artifacts"
    if artifacts.is_dir():
        paths.extend(sorted(path for path in artifacts.rglob("*") if path.is_file()))
    images = release_dir / "images"
    if images.is_dir():
        paths.extend(sorted(path for path in images.rglob("*.sif") if path.is_file()))
    return paths


def write_sha256sums(release_dir: Path) -> None:
    lines = [f"{sha256_file(path)}  {path.relative_to(release_dir).as_posix()}" for path in _declared_files(release_dir)]
    _replace_text(release_dir / "SHA256SUMS", "\n".join(lines) + "\n")


def verify_sha256sums(release_dir: str | Path) -> dict[str, Any]:
    release_dir = Path(release_dir).resolve()
    sums = release_dir / "SHA256SUMS"
    if not sums.is_file():
        raise ReleaseError(f"Missing SHA256SUMS at {sums}.")
    checked = 0
    for line_number, line in enumerate(sums.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\n]+)", line)
        if not match:
            raise ReleaseError(f"Malformed SHA256SUMS line {line_number}.")
        expected, relative = match.groups()
        path = release_dir / relative
        if path.resolve().parent != release_dir and release_dir not in path.resolve().parents:
            raise ReleaseError(f"SHA256SUMS path escapes release directory: {relative!r}.")
        if not path.is_file():
            raise ReleaseError(f"Checksummed file is missing: {relative}.")
        observed = sha256_file(path)
        if observed != expected:
            raise ReleaseError(f"Checksum drift for {relative}: expected {expected}, observed {observed}.")
        checked += 1
    return {"status": "VERIFIED", "checked_files": checked, "sha256sums_sha256": sha256_file(sums)}


def audit_candidate(input_path: str | Path, release_dir: str | Path, *, mode: str) -> dict[str, Any]:
    # The validator is external now, so the local gate is our manifest's integrity. Requiring an
    # acquired checkout here would make auditing a candidate depend on a third-party download.
    verify_manifest()
    verify_entrypoint_contract()
    input_path = Path(input_path).resolve()
    payload = load_manifest(input_path)
    payload = copy.deepcopy(payload)
    recorded_input_digest = payload.get("manifest_sha256")
    if recorded_input_digest is not None and recorded_input_digest != manifest_digest(payload):
        raise ReleaseError("Candidate input manifest digest is stale; refusing to re-sign changed scientific input.")
    if mode == "final" and payload.get("release_mode") == "smoke_only":
        raise ReleaseError("A smoke_only manifest is permanently watermarked and cannot be promoted to final.")
    _absolutize_artifact_paths(payload, input_path.parent)
    input_had_created_at = "created_at" in payload
    payload["schema_version"] = "fomo26-release-manifest-v1"
    payload["release_mode"] = mode
    release_dir = Path(release_dir).resolve()
    release_id = _safe_identifier(release_dir.name, "release id")
    recorded_release_id = payload.get("release_id")
    if recorded_release_id not in {None, release_id}:
        raise ReleaseError(
            f"Candidate input release_id {recorded_release_id!r} does not match output directory {release_id!r}."
        )
    payload["release_id"] = release_id
    head = _git_head()
    recorded_packaging = payload.get("packaging_git_commit")
    if recorded_packaging not in {None, head}:
        raise ReleaseError(f"Candidate input packaging commit {recorded_packaging} does not match HEAD {head}.")
    payload["packaging_git_commit"] = head
    payload.setdefault("created_at", _now())
    payload.pop("manifest_sha256", None)
    if mode == "final" and not _git_is_clean():
        raise ReleaseError("Final audit requires a clean working tree so packaging_git_commit binds every source byte.")
    existing_manifest = release_dir / "release_manifest.json"
    if existing_manifest.is_file():
        existing = load_manifest(existing_manifest)
        validate_release_manifest(existing, verify_files=True, base_dir=release_dir)
        comparable = copy.deepcopy(payload)
        if not input_had_created_at:
            comparable["created_at"] = existing["created_at"]
        comparable = finalized_manifest(comparable)
        if comparable != existing:
            raise ReleaseError(f"Existing audited release conflicts with candidate input at {release_dir}.")
        receipt_path = release_dir / "receipts" / "audit.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
        if receipt is None:
            receipt = _receipt(
                "audit",
                status="PASS",
                created_at=_now(),
                release_manifest_sha256=existing["manifest_sha256"],
                release_mode=mode,
                packaging_git_commit=head,
            )
            _write_json_immutable(receipt_path, receipt)
            write_sha256sums(release_dir)
            return {**receipt, "resumed": True}
        if not isinstance(receipt, dict) or receipt.get("release_manifest_sha256") != existing["manifest_sha256"]:
            raise ReleaseError(f"Existing audit receipt is missing or stale at {receipt_path}.")
        return {**receipt, "idempotent": True}
    validate_release_manifest(payload, verify_files=True)
    payload = finalized_manifest(payload)
    release_dir.mkdir(parents=True, exist_ok=True)
    _write_json_immutable(release_dir / "release_manifest.json", payload)
    audit_receipt = _receipt(
        "audit",
        status="PASS",
        created_at=_now(),
        release_manifest_sha256=payload["manifest_sha256"],
        release_mode=mode,
        packaging_git_commit=head,
    )
    _write_json_immutable(release_dir / "receipts" / "audit.json", audit_receipt)
    write_sha256sums(release_dir)
    return audit_receipt


def _resolve_artifact(source_release: Path, spec: dict[str, Any]) -> Path:
    path = Path(spec["path"])
    return path if path.is_absolute() else source_release / path


def _copy_artifact(source: Path, destination: Path, expected_sha: str) -> None:
    if not source.is_file() or sha256_file(source) != expected_sha:
        raise ReleaseError(f"Source artifact is missing or drifted: {source}.")
    if destination.exists():
        if destination.is_file() and sha256_file(destination) == expected_sha:
            return
        raise ReleaseError(f"Refusing to overwrite conflicting staged artifact {destination}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    if temporary.exists() and sha256_file(temporary) != expected_sha:
        raise ReleaseError(f"Conflicting partial artifact requires inspection: {temporary}.")
    if not temporary.exists():
        shutil.copyfile(source, temporary)
    if sha256_file(temporary) != expected_sha:
        raise ReleaseError(f"Artifact changed during copy: {source} -> {temporary}.")
    os.replace(temporary, destination)


def _normalize_and_copy_manifest(source_dir: Path, target_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(payload)
    source_digest = payload["manifest_sha256"]
    authority_rel = Path("artifacts/authority/scientific_handoff.json")
    _copy_artifact(
        _resolve_artifact(source_dir, payload["scientific_authority"]),
        target_dir / authority_rel,
        payload["scientific_authority"]["sha256"],
    )
    normalized["scientific_authority"]["path"] = authority_rel.as_posix()
    pretrained_rel = Path("artifacts/pretrained/pretrained.ckpt")
    _copy_artifact(
        _resolve_artifact(source_dir, payload["pretrained"]),
        target_dir / pretrained_rel,
        payload["pretrained"]["sha256"],
    )
    normalized["pretrained"]["path"] = pretrained_rel.as_posix()
    for task in DOWNSTREAM_TASKS:
        for fold in normalized["tasks"][task]["folds"]:
            source_fold = next(item for item in payload["tasks"][task]["folds"] if item["fold"] == fold["fold"])
            for key, filename in (
                ("checkpoint", "checkpoint.ckpt"),
                ("run_manifest", "run_manifest.json"),
                ("hydra_config", "config.yaml"),
            ):
                relative = Path(f"artifacts/{task}/fold{fold['fold']}/{filename}")
                _copy_artifact(
                    _resolve_artifact(source_dir, source_fold[key]),
                    target_dir / relative,
                    source_fold[key]["sha256"],
                )
                fold[key]["path"] = relative.as_posix()
        calibration = normalized["tasks"][task]["policy"]["calibration"]
        if calibration["state"] == "required":
            source_calibration = payload["tasks"][task]["policy"]["calibration"]["artifact"]
            relative = Path(f"artifacts/{task}/calibration.json")
            _copy_artifact(
                _resolve_artifact(source_dir, source_calibration),
                target_dir / relative,
                source_calibration["sha256"],
            )
            calibration["artifact"]["path"] = relative.as_posix()
    normalized["source_manifest_sha256"] = source_digest
    normalized = finalized_manifest(normalized)
    validate_release_manifest(normalized, verify_files=True, base_dir=target_dir)
    return normalized


def export_release(release_dir: str | Path, destination_root: str | Path) -> dict[str, Any]:
    source_dir, payload = _load_release(release_dir)
    release_id = _safe_identifier(source_dir.name, "release id")
    destination_root = Path(destination_root).resolve()
    destination = destination_root / release_id
    if destination.exists():
        report = verify_sha256sums(destination)
        receipt_path = destination / "receipts" / "export.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
        if receipt.get("source_manifest_sha256") != payload["manifest_sha256"]:
            raise ReleaseError(f"Export collision at {destination}: source manifest differs.")
        return {**receipt, "idempotent": True, "verification": report}
    destination_root.mkdir(parents=True, exist_ok=True)
    partial = destination_root / f".{release_id}.partial"
    partial.mkdir(parents=True, exist_ok=True)
    completed_receipt = partial / "receipts" / "export.json"
    if completed_receipt.is_file():
        receipt = json.loads(completed_receipt.read_text())
        if receipt.get("source_manifest_sha256") != payload["manifest_sha256"]:
            raise ReleaseError(f"Partial export receipt conflicts at {completed_receipt}.")
        verify_sha256sums(partial)
        os.replace(partial, destination)
        return {**receipt, "resumed": True}
    marker = {"source_manifest_sha256": payload["manifest_sha256"], "release_id": release_id}
    _write_json_immutable(partial / ".export-source.json", marker)
    normalized = _normalize_and_copy_manifest(source_dir, partial, payload)
    _write_json_immutable(partial / "release_manifest.json", normalized)
    write_sha256sums(partial)
    receipt = _receipt(
        "export",
        status="PASS",
        created_at=_now(),
        source_manifest_sha256=payload["manifest_sha256"],
        exported_manifest_sha256=normalized["manifest_sha256"],
        sha256sums_sha256=sha256_file(partial / "SHA256SUMS"),
        normalized_paths=True,
    )
    _write_json_immutable(partial / "receipts" / "export.json", receipt)
    verify_sha256sums(partial)
    (partial / ".export-source.json").unlink()
    os.replace(partial, destination)
    return receipt


def _run_retries(command: list[str], *, attempts: int, timeout: int) -> subprocess.CompletedProcess[str]:
    last = None
    for attempt in range(1, attempts + 1):
        try:
            last = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            last = subprocess.CompletedProcess(command, 124, exc.stdout or "", exc.stderr or f"timeout after {timeout}s")
        if last.returncode == 0:
            return last
        if attempt < attempts:
            time.sleep(min(2 ** (attempt - 1), 8))
    assert last is not None
    return last


def _ssh_command(host: str, *remote_arguments: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=2",
        host,
        *remote_arguments,
    ]


def pull_release(
    remote: str,
    destination_root: str | Path,
    *,
    release_id: str,
    attempts: int = 3,
    timeout: int = 300,
    dry_run: bool = False,
    rsync: str = "rsync",
) -> dict[str, Any]:
    _safe_identifier(release_id, "release id")
    if attempts <= 0 or timeout <= 0:
        raise ReleaseError("Pull attempts and timeout must be positive integers.")
    remote_match = re.fullmatch(r"([A-Za-z0-9_.@-]+):(.+)", remote)
    if not remote_match:
        raise ReleaseError("--remote must be a bounded host:path rsync source.")
    _safe_remote_path(remote_match.group(2))
    destination_root = Path(destination_root).resolve()
    destination = destination_root / release_id
    if destination.exists():
        report = verify_sha256sums(destination)
        return {"phase": "pull", "status": "PASS", "idempotent": True, "verification": report}
    partial = destination_root / f".{release_id}.partial"
    command = [
        rsync,
        "-a",
        "--checksum",
        "--partial",
        "--partial-dir=.rsync-partial",
        "-e",
        "ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=2",
    ]
    if dry_run:
        command.append("--dry-run")
    command += [remote.rstrip("/") + "/", str(partial) + "/"]
    if dry_run:
        return {"phase": "pull", "status": "DRY_RUN", "command": command}
    destination_root.mkdir(parents=True, exist_ok=True)
    partial.mkdir(parents=True, exist_ok=True)
    proc = _run_retries(command, attempts=attempts, timeout=timeout)
    if proc.returncode != 0:
        raise ReleaseError(f"rsync failed after {attempts} attempt(s), rc={proc.returncode}: {proc.stderr.strip()}")
    report = verify_sha256sums(partial)
    _load_release(partial)
    receipt = _receipt(
        "pull",
        status="PASS",
        created_at=_now(),
        remote=remote,
        attempts=attempts,
        sha256sums_sha256=report["sha256sums_sha256"],
    )
    _write_json_immutable(partial / "receipts" / "pull.json", receipt)
    os.replace(partial, destination)
    return receipt


def verify_release(release_dir: str | Path) -> dict[str, Any]:
    release_dir, payload = _load_release(release_dir)
    sums = verify_sha256sums(release_dir)
    receipt = _receipt(
        "verify",
        status="PASS",
        created_at=_now(),
        release_manifest_sha256=payload["manifest_sha256"],
        checksum_status=sums["status"],
        checked_files=sums["checked_files"],
        sha256sums_sha256=sums["sha256sums_sha256"],
    )
    receipt_id = sums["sha256sums_sha256"][:16]
    existing = release_dir / "receipts" / "verify" / f"{receipt_id}.json"
    if existing.is_file():
        prior = json.loads(existing.read_text())
        if (
            prior.get("release_manifest_sha256") == payload["manifest_sha256"]
            and prior.get("sha256sums_sha256") == sums["sha256sums_sha256"]
        ):
            return {**prior, "idempotent": True}
        raise ReleaseError(f"Immutable versioned verify receipt conflicts at {existing}.")
    _write_json_immutable(existing, receipt)
    return receipt


def stage_release(release_dir: str | Path) -> list[dict[str, Any]]:
    from finetuning.container.build_container import stage_release_context

    release_dir, payload = _load_release(release_dir)
    if payload["packaging_git_commit"] != _git_head():
        raise ReleaseError("Release packaging commit does not match the checked-out source commit.")
    if not _git_is_clean():
        raise ReleaseError("Release staging requires a clean tracked and untracked working tree.")
    results = []
    for task in OFFICIAL_TASKS:
        context = release_dir / "build-contexts" / task
        receipt_path = release_dir / "receipts" / "stage" / f"{task}.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            if (
                context.is_dir()
                and receipt.get("context_sha256") == _tree_sha256(context)
                and receipt.get("release_manifest_sha256") == payload["manifest_sha256"]
            ):
                results.append({**receipt, "idempotent": True})
                continue
            raise ReleaseError(f"Staged context drifted after immutable receipt: {context}.")
        stage_release_context(release_dir=release_dir, manifest=payload, task=task, out=context)
        receipt = _receipt(
            "stage",
            status="PASS",
            created_at=_now(),
            task=task,
            release_manifest_sha256=payload["manifest_sha256"],
            context_sha256=_tree_sha256(context),
            entrypoint_sha256=sha256_file(context / "predict.py"),
            model_manifest_sha256=sha256_file(context / "models" / "model_manifest.json"),
        )
        _write_json_immutable(receipt_path, receipt)
        results.append(receipt)
    return results


def _version(command: list[str]) -> str | None:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
    except OSError:
        return None
    output = (proc.stdout or proc.stderr).strip()
    return output if proc.returncode == 0 else None


def build_release(release_dir: str | Path, *, apptainer: str = "apptainer") -> list[dict[str, Any]]:
    from finetuning.container.build_container import audit_release_context

    release_dir, payload = _load_release(release_dir)
    apptainer_version = _version([apptainer, "--version"])
    if apptainer_version is None:
        raise ReleaseError(f"Apptainer executable is unavailable or unusable: {apptainer}.")
    results = []
    for task in OFFICIAL_TASKS:
        context = release_dir / "build-contexts" / task
        if not context.is_dir():
            raise ReleaseError(f"Missing staged context for {task}: run stage first.")
        context_sha = _tree_sha256(context)
        stage_receipt = _load_receipt(release_dir / "receipts" / "stage" / f"{task}.json")
        if (
            not stage_receipt
            or stage_receipt.get("status") != "PASS"
            or stage_receipt.get("task") != task
            or stage_receipt.get("context_sha256") != context_sha
        ):
            raise ReleaseError(f"Missing or stale stage receipt for {task}.")
        context_audit = audit_release_context(context)
        image = _image_path(release_dir, payload, task)
        receipt_path = release_dir / "receipts" / "build" / f"{task}.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            if (
                image.is_file()
                and receipt.get("task") == task
                and receipt.get("sif_sha256") == sha256_file(image)
                and receipt.get("context_sha256") == context_sha
                and receipt.get("release_manifest_sha256") == payload["manifest_sha256"]
                and receipt.get("packaging_git_commit") == payload["packaging_git_commit"]
                and receipt.get("pretrained_checkpoint_sha256") == payload["pretrained"]["sha256"]
            ):
                results.append({**receipt, "idempotent": True})
                continue
            raise ReleaseError(f"SIF or build receipt drifted for {task}: {image}.")
        in_progress_path = release_dir / "receipts" / "build" / "in-progress" / f"{task}-{context_sha[:16]}.json"
        if image.exists():
            in_progress = _load_receipt(in_progress_path)
            if (
                not image.is_file()
                or not in_progress
                or in_progress.get("task") != task
                or in_progress.get("context_sha256") != context_sha
                or in_progress.get("release_manifest_sha256") != payload["manifest_sha256"]
            ):
                raise ReleaseError(f"Refusing to overwrite unreceipted image without matching build marker: {image}.")
            recovery_id = manifest_digest(
                {"task": task, "context_sha256": context_sha, "discarded_sif_sha256": sha256_file(image)}
            )[:16]
            _write_json_immutable(
                release_dir / "receipts" / "build" / "recoveries" / f"{task}-{recovery_id}.json",
                _receipt(
                    "build-recovery",
                    status="RETRY_REQUIRED",
                    created_at=_now(),
                    task=task,
                    context_sha256=context_sha,
                    release_manifest_sha256=payload["manifest_sha256"],
                    discarded_unreceipted_sif_sha256=sha256_file(image),
                    discarded_unreceipted_sif_size=image.stat().st_size,
                ),
            )
            image.unlink()
        model_manifest_path = context / "models" / "model_manifest.json"
        if not model_manifest_path.is_file():
            raise ReleaseError(f"Missing staged model manifest for {task}: {model_manifest_path}.")
        model_manifest = json.loads(model_manifest_path.read_text())
        if model_manifest.get("task") != task:
            raise ReleaseError(f"Staged model manifest task mismatch for {task}.")
        pretrained_sha = model_manifest.get("pretrained_checkpoint_sha256")
        if pretrained_sha != payload["pretrained"]["sha256"]:
            raise ReleaseError(f"Staged model manifest has conflicting pretrained lineage for {task}.")
        architecture = str(model_manifest.get("architecture", "")).strip().lower()
        if not architecture:
            raise ReleaseError(f"Staged model manifest has no architecture for {task}.")
        natten_backend = "libnatten+torch270cu126" if architecture.startswith("medvit") else "python-only"
        image.parent.mkdir(parents=True, exist_ok=True)
        _write_json_immutable(
            in_progress_path,
            _receipt(
                "build-in-progress",
                status="STARTED",
                task=task,
                context_sha256=context_sha,
                release_manifest_sha256=payload["manifest_sha256"],
                packaging_git_commit=payload["packaging_git_commit"],
            ),
        )
        started = _now()
        started_monotonic = time.monotonic()
        command = [
            apptainer,
            "build",
            "--fakeroot",
            "--arch",
            "amd64",
            str(image),
            str(context / "Apptainer.def"),
        ]
        proc = subprocess.run(command, cwd=context, capture_output=True, text=True, check=False)
        finished = _now()
        duration_seconds = time.monotonic() - started_monotonic
        build_logs = release_dir / "logs" / "build"
        if proc.returncode == 0 and image.is_file():
            stdout_path = build_logs / f"{task}.stdout.txt"
            stderr_path = build_logs / f"{task}.stderr.txt"
        else:
            failure_id = manifest_digest(
                {"task": task, "started_at": started, "returncode": proc.returncode, "stderr": proc.stderr}
            )[:16]
            stdout_path = build_logs / "failures" / f"{task}-{failure_id}.stdout.txt"
            stderr_path = build_logs / "failures" / f"{task}-{failure_id}.stderr.txt"
        _write_bytes_immutable(stdout_path, proc.stdout.encode())
        _write_bytes_immutable(stderr_path, proc.stderr.encode())
        if proc.returncode != 0 or not image.is_file():
            incomplete_sha = sha256_file(image) if image.is_file() else None
            incomplete_size = image.stat().st_size if image.is_file() else None
            failure_receipt = _receipt(
                "build",
                status="FAIL",
                task=task,
                started_at=started,
                finished_at=finished,
                duration_seconds=duration_seconds,
                returncode=proc.returncode,
                release_manifest_sha256=payload["manifest_sha256"],
                packaging_git_commit=payload["packaging_git_commit"],
                context_sha256=context_sha,
                incomplete_sif_sha256=incomplete_sha,
                incomplete_sif_size=incomplete_size,
                stdout_path=stdout_path.relative_to(release_dir).as_posix(),
                stderr_path=stderr_path.relative_to(release_dir).as_posix(),
            )
            failure_path = release_dir / "receipts" / "build" / "failures" / f"{task}-{failure_id}.json"
            _write_json_immutable(failure_path, failure_receipt)
            if image.is_file():
                image.unlink()
            raise ReleaseError(f"Apptainer build failed for {task}, rc={proc.returncode}: {proc.stderr[-2000:]}")
        receipt = _receipt(
            "build",
            status="PASS",
            task=task,
            started_at=started,
            finished_at=finished,
            duration_seconds=duration_seconds,
            command=command,
            command_cwd=str(context),
            apptainer_version=apptainer_version,
            base_image=BASE_IMAGE,
            dependency_lock_sha256=sha256_file(REPO / "uv.lock"),
            dependency_versions={
                "torch": "2.7.0",
                "natten": "0.21.0",
                "natten_backend": natten_backend,
                "cuda": "12.6",
                "cudnn": "9",
            },
            host=platform.node(),
            host_platform=platform.platform(),
            release_manifest_sha256=payload["manifest_sha256"],
            packaging_git_commit=payload["packaging_git_commit"],
            context_sha256=context_sha,
            context_audit=context_audit,
            model_manifest_sha256=sha256_file(model_manifest_path),
            model_artifacts=model_manifest.get("models", []),
            entrypoint_sha256=sha256_file(context / "predict.py"),
            pretrained_checkpoint_sha256=pretrained_sha,
            sif_size=image.stat().st_size,
            sif_sha256=sha256_file(image),
            stdout_path=stdout_path.relative_to(release_dir).as_posix(),
            stderr_path=stderr_path.relative_to(release_dir).as_posix(),
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
        _write_json_immutable(receipt_path, receipt)
        results.append(receipt)
    write_sha256sums(release_dir)
    return results


def validate_images(
    release_dir: str | Path,
    *,
    fixture_cache: str | Path,
    apptainer: str = "apptainer",
    no_gpu: bool = False,
    timeout: int = 900,
) -> list[dict[str, Any]]:
    release_dir, payload = _load_release(release_dir)
    fixture_report = verify_fixtures(fixture_cache)
    current_validator_sha = validator_sha256()
    results = []
    for task in OFFICIAL_TASKS:
        image = _image_path(release_dir, payload, task)
        suffix = "-structural" if no_gpu else ""
        receipt_path = release_dir / "receipts" / "validator" / f"{task}{suffix}.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            if (
                image.is_file()
                and receipt.get("sif_sha256_before") == sha256_file(image)
                and receipt.get("sif_sha256_after") == sha256_file(image)
                and receipt.get("release_manifest_sha256") == payload["manifest_sha256"]
                and receipt.get("validator_sha256") == current_validator_sha
                and receipt.get("fixture_digest") == fixture_report["fixture_digest"]
                and receipt.get("structural_only") is no_gpu
            ):
                if receipt.get("status") != "PASS":
                    raise ReleaseError(f"Prior official validator receipt records failure for {task}.")
                results.append({**receipt, "idempotent": True})
                continue
            raise ReleaseError(f"SIF drifted after official validation receipt: {image}.")
        started = _now()
        result = run_official_validator(
            task=task,
            sif=image,
            fixture_cache=fixture_cache,
            apptainer=apptainer,
            no_gpu=no_gpu,
            timeout=timeout,
        )
        stdout, stderr = result.pop("stdout"), result.pop("stderr")
        finished = _now()
        if result["status"] == "PASS":
            receipt_target = receipt_path
            log_stem = f"{task}{suffix}"
        else:
            failure_id = manifest_digest({"task": task, "started_at": started, "result": result})[:16]
            receipt_target = release_dir / "receipts" / "validator" / "failures" / f"{task}{suffix}-{failure_id}.json"
            log_stem = f"failures/{task}{suffix}-{failure_id}"
        logs = release_dir / "receipts" / "validator" / "logs"
        _write_bytes_immutable(logs / f"{log_stem}.stdout.txt", stdout.encode())
        _write_bytes_immutable(logs / f"{log_stem}.stderr.txt", stderr.encode())
        receipt = _receipt(
            "validator",
            started_at=started,
            finished_at=finished,
            release_manifest_sha256=payload["manifest_sha256"],
            stdout_path=f"logs/{log_stem}.stdout.txt",
            stderr_path=f"logs/{log_stem}.stderr.txt",
            **result,
        )
        _write_json_immutable(receipt_target, receipt)
        results.append(receipt)
        if receipt["status"] != "PASS":
            raise ReleaseError(f"Official validator failed or produced a malformed success result for {task}.")
    return results


def run_local(
    release_dir: str | Path,
    *,
    task: str,
    arguments: list[str],
    apptainer: str = "apptainer",
    gpu: bool = False,
) -> dict[str, Any]:
    release_dir, payload = _load_release(release_dir)
    if task not in OFFICIAL_TASKS:
        raise ReleaseError(f"Unknown official task {task!r}.")
    image = _image_path(release_dir, payload, task)
    before = sha256_file(image)
    command = [apptainer, "run"]
    if gpu:
        command.append("--nv")
    command += [str(image), *arguments]
    invocation = {
        "task": task,
        "command": command,
        "gpu": gpu,
        "sif_sha256": before,
        "release_manifest_sha256": payload["manifest_sha256"],
    }
    invocation_id = manifest_digest(invocation)[:16]
    receipt_path = release_dir / "receipts" / "runtime" / f"{task}-{invocation_id}.json"
    if receipt_path.is_file():
        prior = _load_receipt(receipt_path)
        if prior and prior.get("sif_sha256_after") == before:
            if prior.get("status") != "PASS":
                raise ReleaseError(f"Prior local runtime receipt records failure at {receipt_path}.")
            return {**prior, "idempotent": True}
        raise ReleaseError(f"Runtime receipt or SIF drifted at {receipt_path}.")
    started = _now()
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    after = sha256_file(image)
    receipt = _receipt(
        "runtime",
        status="PASS" if proc.returncode == 0 and before == after else "FAIL",
        task=task,
        started_at=started,
        finished_at=_now(),
        command=command,
        returncode=proc.returncode,
        gpu=gpu,
        sif_sha256_before=before,
        sif_sha256_after=after,
        release_manifest_sha256=payload["manifest_sha256"],
        stdout=proc.stdout,
        stderr=proc.stderr,
    )
    _write_json_immutable(receipt_path, receipt)
    if receipt["status"] != "PASS":
        raise ReleaseError(f"Local runtime failed for {task}, rc={proc.returncode}; see {receipt_path}.")
    return receipt


def _remote_shell_path(value: str) -> str:
    if value == "$SCRATCH":
        return '"${SCRATCH}"'
    if value.startswith("$SCRATCH/"):
        return f'"${{SCRATCH}}/{value.removeprefix("$SCRATCH/")}"'
    return shlex.quote(value)


def _qualification_script(
    remote_dir: str,
    threshold: float,
    *,
    release_manifest_sha: str,
    validator_digest: str,
    fixture_digest: str,
    image_digests: dict[str, str],
    image_names: dict[str, str],
) -> str:
    remote = _remote_shell_path(remote_dir)
    commands = [
        "#!/bin/bash",
        "set -uo pipefail",
        "module purge",
        "module load arch/h100",
        "module load singularity",
        f"mkdir -p {remote}/qualification-receipts",
        "fomo26_qualification_rc=0",
    ]
    for task in OFFICIAL_TASKS:
        commands.append(
            f"if ! python3 {remote}/qualification_runner.py "
            f'--task {shlex.quote(task)} --sif "${{SINGULARITY_ALLOWED_DIR}}/{image_names[task]}" '
            f"--validator-root {remote}/validator "
            f"--fixture-cache {remote}/fixtures "
            f"--receipt {remote}/qualification-receipts/{task}.json "
            f"--threshold-seconds {threshold:g} "
            f"--release-manifest-sha256 {release_manifest_sha} "
            f"--validator-sha256 {validator_digest} --fixture-digest {fixture_digest} "
            f"--expected-sif-sha256 {image_digests[task]}; then fomo26_qualification_rc=1; fi"
        )
    commands.append('exit "${fomo26_qualification_rc}"')
    return "\n".join(commands) + "\n"


def qualify_release(
    release_dir: str | Path,
    *,
    host: str,
    remote_dir: str,
    fixture_cache: str | Path,
    submit: bool = False,
    collect: bool = False,
    job_id: str | None = None,
) -> dict[str, Any]:
    release_dir, payload = _load_release(release_dir)
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", host):
        raise ReleaseError(f"Unsafe qualification host {host!r}.")
    remote_dir = _safe_remote_path(remote_dir)
    threshold = float(payload["protocol"]["qualification_limit_seconds"])
    fixture_report = verify_fixtures(fixture_cache)
    image_names = payload["container"]["images"]
    expected_images = {_image_path(release_dir, payload, task) for task in OFFICIAL_TASKS}
    actual_images = {path for path in (release_dir / "images").glob("*.sif") if path.is_file()}
    if actual_images != expected_images:
        raise ReleaseError("Qualification requires exactly the six declared task-specific SIFs.")
    for task in OFFICIAL_TASKS:
        image = _image_path(release_dir, payload, task)
        if not image.is_file():
            raise ReleaseError(f"Missing already-built image for qualification: {image}.")
    qualification_dir = release_dir / "qualification"
    image_digests = {task: sha256_file(_image_path(release_dir, payload, task)) for task in OFFICIAL_TASKS}
    pinned_validator_sha = validator_sha256()
    script = _qualification_script(
        remote_dir,
        threshold,
        release_manifest_sha=payload["manifest_sha256"],
        validator_digest=pinned_validator_sha,
        fixture_digest=fixture_report["fixture_digest"],
        image_digests=image_digests,
        image_names=image_names,
    )
    script_path = qualification_dir / "qualify.slurm"
    _replace_text(script_path, script)
    syntax = subprocess.run(["bash", "-n", str(script_path)], capture_output=True, text=True, check=False)
    if syntax.returncode != 0:
        raise ReleaseError(f"Generated H100 qualification script failed bash -n: {syntax.stderr.strip()}")
    runner_source = Path(__file__).with_name("qualification_runner.py")
    _write_bytes_immutable(qualification_dir / "qualification_runner.py", runner_source.read_bytes())
    plan = {
        "schema_version": RECEIPT_SCHEMA,
        "phase": "qualify",
        "status": "DRY_RUN" if not submit and not collect else "PLANNED",
        "release_manifest_sha256": payload["manifest_sha256"],
        "host": host,
        "remote_dir": remote_dir,
        "threshold_seconds": threshold,
        "images": image_digests,
        "image_names": image_names,
        "validator_sha256": pinned_validator_sha,
        "fixture_digest": fixture_report["fixture_digest"],
        "qualification_script_sha256": sha256_file(script_path),
        "bash_syntax": "PASS",
        "resource_request": {"constraint": "h100", "gpus": 1, "cpus_per_task": 24, "time": "00:20:00"},
        "commands": [f"rsync -a --partial <release inputs> {host}:{remote_dir}/"]
        + [f"ssh {host} idrcontmgr cp {remote_dir}/images/{image_names[task]}" for task in OFFICIAL_TASKS]
        + [f"ssh {host} sbatch -C h100 --gres=gpu:1 --cpus-per-task=24 {remote_dir}/qualify.slurm"],
    }
    plan_id = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:16]
    _write_json_immutable(release_dir / "receipts" / f"qualification-plan-{plan_id}.json", plan)
    if collect:
        if not job_id or not re.fullmatch(r"[0-9]+", job_id):
            raise ReleaseError("--collect requires a numeric --job-id.")
        destination = release_dir / "receipts" / "qualification"
        collection_path = release_dir / "receipts" / f"qualification-collect-{job_id}.json"
        if collection_path.is_file():
            prior = _load_receipt(collection_path)
            if prior and all(
                (destination / f"{task}.json").is_file()
                and prior.get("receipt_sha256", {}).get(task) == sha256_file(destination / f"{task}.json")
                for task in OFFICIAL_TASKS
            ):
                return {**prior, "idempotent": True}
            raise ReleaseError(f"Qualification collection receipt is stale at {collection_path}.")
        with tempfile.TemporaryDirectory(prefix=".qualification-collect-", dir=release_dir / "receipts") as temporary:
            temporary_path = Path(temporary)
            command = [
                "rsync",
                "-a",
                "--partial",
                "-e",
                "ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=2",
                f"{host}:{remote_dir}/qualification-receipts/",
                str(temporary_path) + "/",
            ]
            proc = _run_retries(command, attempts=3, timeout=300)
            if proc.returncode != 0:
                raise ReleaseError(f"Qualification receipt collection failed: {proc.stderr.strip()}")
            collected = {}
            for task in OFFICIAL_TASKS:
                source = temporary_path / f"{task}.json"
                receipt = _load_receipt(source)
                expected = {
                    "task": task,
                    "slurm_job_id": job_id,
                    "release_manifest_sha256": payload["manifest_sha256"],
                    "validator_sha256": pinned_validator_sha,
                    "fixture_digest": fixture_report["fixture_digest"],
                    "sif_sha256_before": image_digests[task],
                    "sif_sha256_after": image_digests[task],
                }
                if not receipt or any(receipt.get(key) != value for key, value in expected.items()):
                    raise ReleaseError(f"Qualification receipt binding failed for {task}.")
                if receipt.get("threshold_seconds") != threshold:
                    raise ReleaseError(f"Qualification threshold drifted for {task}.")
                published = destination / f"{task}.json"
                _write_json_immutable(published, receipt)
                collected[task] = sha256_file(published)
        collection = _receipt(
            "qualification-collect",
            status="PASS",
            created_at=_now(),
            job_id=job_id,
            release_manifest_sha256=payload["manifest_sha256"],
            receipt_sha256=collected,
        )
        _write_json_immutable(collection_path, collection)
        return collection
    if not submit:
        return plan

    submit_receipt_path = release_dir / "receipts" / f"qualification-submit-{plan_id}.json"
    if submit_receipt_path.is_file():
        prior = _load_receipt(submit_receipt_path)
        if prior and prior.get("release_manifest_sha256") == payload["manifest_sha256"]:
            return {**prior, "idempotent": True}
        raise ReleaseError(f"Qualification submission receipt is stale at {submit_receipt_path}.")

    remote_work = remote_dir
    mkdir = _run_retries(_ssh_command(host, "mkdir", "-p", remote_work), attempts=3, timeout=60)
    if mkdir.returncode != 0:
        raise ReleaseError(f"Cannot create remote qualification directory: {mkdir.stderr.strip()}")
    transfers = [
        (release_dir / "images", f"{host}:{remote_work}/images/"),
        (Path(fixture_cache), f"{host}:{remote_work}/fixtures/"),
        (Path(__file__).with_name("qualification_runner.py"), f"{host}:{remote_work}/qualification_runner.py"),
        (validator_root(), f"{host}:{remote_work}/validator/"),
    ]
    for source, destination in transfers:
        source_arg = str(source) + ("/" if source.is_dir() else "")
        proc = _run_retries(
            [
                "rsync",
                "-a",
                "--partial",
                "-e",
                "ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=2",
                source_arg,
                destination,
            ],
            attempts=3,
            timeout=600,
        )
        if proc.returncode != 0:
            raise ReleaseError(f"Qualification transfer failed: {proc.stderr.strip()}")
    remote_script = _qualification_script(
        remote_work,
        threshold,
        release_manifest_sha=payload["manifest_sha256"],
        validator_digest=pinned_validator_sha,
        fixture_digest=fixture_report["fixture_digest"],
        image_digests=image_digests,
        image_names=image_names,
    )
    local_script = qualification_dir / "qualify.slurm"
    _replace_text(local_script, remote_script)
    proc = _run_retries(
        [
            "rsync",
            "-a",
            "-e",
            "ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=2",
            str(local_script),
            f"{host}:{remote_work}/qualify.slurm",
        ],
        attempts=3,
        timeout=120,
    )
    if proc.returncode != 0:
        raise ReleaseError(f"Qualification script transfer failed: {proc.stderr.strip()}")
    for task in OFFICIAL_TASKS:
        prep = _run_retries(
            _ssh_command(host, "idrcontmgr", "cp", f"{remote_work}/images/{image_names[task]}"),
            attempts=3,
            timeout=600,
        )
        if prep.returncode != 0:
            raise ReleaseError(f"idrcontmgr failed for {task}: {prep.stderr.strip()}")
    submit_command = _ssh_command(
        host,
        "sbatch",
        "--parsable",
        "-C",
        "h100",
        "--gres=gpu:1",
        "--cpus-per-task=24",
        "--time=00:20:00",
        f"{remote_work}/qualify.slurm",
    )
    proc = _run_retries(submit_command, attempts=1, timeout=60)
    if proc.returncode != 0:
        raise ReleaseError(f"Qualification submission failed: {proc.stderr.strip()}")
    returned_job = proc.stdout.strip().split(";")[0]
    if not re.fullmatch(r"[0-9]+", returned_job):
        raise ReleaseError(f"Slurm returned a malformed job id: {returned_job!r}.")
    receipt = _receipt(
        "qualify-submit",
        status="SUBMITTED",
        created_at=_now(),
        host=host,
        remote_dir=remote_work,
        job_id=returned_job,
        command=submit_command,
        release_manifest_sha256=payload["manifest_sha256"],
        images=image_digests,
        threshold_seconds=threshold,
        validator_sha256=pinned_validator_sha,
        fixture_digest=fixture_report["fixture_digest"],
    )
    _write_json_immutable(submit_receipt_path, receipt)
    return receipt


def _load_receipt(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _matching_verify_receipt(release_dir: Path, payload: dict[str, Any]) -> tuple[Path | None, dict[str, Any] | None]:
    sums_path = release_dir / "SHA256SUMS"
    if not sums_path.is_file():
        return None, None
    sums_sha = sha256_file(sums_path)
    candidates = [release_dir / "receipts" / "verify.json"]
    candidates.extend(sorted((release_dir / "receipts" / "verify").glob("*.json")))
    for path in candidates:
        receipt = _load_receipt(path)
        if (
            receipt
            and receipt.get("status") == "PASS"
            and receipt.get("release_manifest_sha256") == payload["manifest_sha256"]
            and receipt.get("sha256sums_sha256") == sums_sha
        ):
            return path, receipt
    return None, None


def readiness_blockers(release_dir: Path, payload: dict[str, Any]) -> list[str]:
    blockers = []
    if payload["release_mode"] != "final":
        blockers.append("release_mode_is_smoke_only")
    threshold = float(payload["protocol"]["qualification_limit_seconds"])
    current_validator_sha = validator_sha256()
    common_pretrained_sha = payload["pretrained"]["sha256"]
    _, verify_receipt = _matching_verify_receipt(release_dir, payload)
    if not verify_receipt:
        blockers.append("release:missing_or_stale_verify_receipt")
    collection_valid = False
    for collection_path in sorted((release_dir / "receipts").glob("qualification-collect-*.json")):
        collection = _load_receipt(collection_path)
        if (
            collection
            and collection.get("status") == "PASS"
            and collection.get("release_manifest_sha256") == payload["manifest_sha256"]
            and all(
                (release_dir / "receipts" / "qualification" / f"{task}.json").is_file()
                and collection.get("receipt_sha256", {}).get(task)
                == sha256_file(release_dir / "receipts" / "qualification" / f"{task}.json")
                for task in OFFICIAL_TASKS
            )
        ):
            collection_valid = True
            break
    if not collection_valid:
        blockers.append("release:missing_or_stale_qualification_collection")
    expected_images = {task: _image_path(release_dir, payload, task) for task in OFFICIAL_TASKS}
    expected_names = sorted(path.name for path in expected_images.values())
    actual_names = sorted(path.name for path in (release_dir / "images").glob("*.sif") if path.is_file())
    if actual_names != expected_names:
        blockers.append("release:expected_exactly_six_task_sifs")
    for task in OFFICIAL_TASKS:
        image = expected_images[task]
        if not image.is_file():
            blockers.append(f"{task}:missing_sif")
            continue
        observed = sha256_file(image)
        build = _load_receipt(release_dir / "receipts" / "build" / f"{task}.json")
        if (
            not build
            or build.get("status") != "PASS"
            or build.get("task") != task
            or build.get("sif_sha256") != observed
            or build.get("release_manifest_sha256") != payload["manifest_sha256"]
            or build.get("pretrained_checkpoint_sha256") != common_pretrained_sha
        ):
            blockers.append(f"{task}:missing_stale_or_mixed_lineage_build_receipt")
        validator = _load_receipt(release_dir / "receipts" / "validator" / f"{task}.json")
        if (
            not validator
            or validator.get("status") != "PASS"
            or validator.get("structural_only")
            or validator.get("release_manifest_sha256") != payload["manifest_sha256"]
            or validator.get("task") != task
        ):
            blockers.append(f"{task}:official_gpu_validation_not_verified")
        elif (
            validator.get("sif_sha256_before") != observed
            or validator.get("sif_sha256_after") != observed
            or not validator.get("success_summary")
        ):
            blockers.append(f"{task}:post_validation_sif_drift")
        elif validator.get("validator_sha256") != current_validator_sha:
            blockers.append(f"{task}:validator_snapshot_drift")
        qualification = _load_receipt(release_dir / "receipts" / "qualification" / f"{task}.json")
        if not qualification or qualification.get("status") != "PASS":
            blockers.append(f"{task}:h100_qualification_not_verified")
        else:
            hardware = str(qualification.get("gpu_name", "")).lower()
            if "h100" not in hardware:
                blockers.append(f"{task}:qualification_hardware_not_h100")
            if qualification.get("sif_sha256_after") != observed:
                blockers.append(f"{task}:post_qualification_sif_drift")
            if qualification.get("sif_sha256_before") != observed:
                blockers.append(f"{task}:pre_qualification_sif_drift")
            if qualification.get("release_manifest_sha256") != payload["manifest_sha256"]:
                blockers.append(f"{task}:qualification_release_manifest_drift")
            if qualification.get("validator_sha256") != current_validator_sha:
                blockers.append(f"{task}:qualification_validator_snapshot_drift")
            if validator and qualification.get("fixture_digest") != validator.get("fixture_digest"):
                blockers.append(f"{task}:qualification_fixture_drift")
            if qualification.get("task") != task:
                blockers.append(f"{task}:qualification_task_binding_drift")
            if qualification.get("threshold_seconds") != threshold:
                blockers.append(f"{task}:qualification_threshold_drift")
            if qualification.get("returncode") != 0 or not qualification.get("success_summary"):
                blockers.append(f"{task}:qualification_terminal_result_not_verified")
            wall = qualification.get("wall_seconds")
            if isinstance(wall, bool) or not isinstance(wall, int | float) or float(wall) >= threshold:
                blockers.append(f"{task}:qualification_margin_not_verified")
    return sorted(set(blockers))


def _checklist(release_dir: Path, payload: dict[str, Any], blockers: list[str]) -> str:
    rows = []
    common_sha = payload["pretrained"]["sha256"]
    for task in OFFICIAL_TASKS:
        image = _image_path(release_dir, payload, task)
        validator = _load_receipt(release_dir / "receipts" / "validator" / f"{task}.json") or {}
        qualification = _load_receipt(release_dir / "receipts" / "qualification" / f"{task}.json") or {}
        if image.is_file():
            rows.append(
                f"- `{task}` queue: `{image.name}` — {image.stat().st_size} bytes — `{sha256_file(image)}` — "
                f"validator `{validator.get('status', 'NOT_VERIFIED')}` — H100 runtime "
                f"`{qualification.get('status', 'NOT_VERIFIED')}`"
            )
        else:
            rows.append(f"- `{task}` queue: `{image.name}` — MISSING")
    readiness = "READY" if not blockers else "NOT READY: " + ", ".join(blockers)
    return f"""# FOMO26 Submission Checklist

Generated from immutable release receipts. This file is guidance only; this repository implements
no Synapse upload or submission command.

- Candidate: `{payload["candidate_id"]}`
- Track: `{payload["track"]}`
- Release manifest SHA-256: `{payload["manifest_sha256"]}`
- Common pretrained checkpoint SHA-256: `{common_sha}`
- Receipt-derived status: **{readiness}**
- Container cardinality: **six task-specific `.sif` files for this track**

## Expected task-specific SIFs

{chr(10).join(rows)}

## Mandatory submission checks

- Confirm each official validator and Jean-Zay H100 receipt matches its own task SIF SHA-256.
- Confirm every task build receipt records the common pretrained checkpoint SHA-256 above.
- Become a Synapse **Certified User** before submission.
- A team has only **three valid submission attempts per task and track**; use them deliberately.
- Create one Synapse project, upload the six task-specific `.sif` files, and submit each to its matching queue.
- **Submit as a Team. Individual submissions are invalid even when the file upload succeeded.**
- Do not reuse one universal multi-task SIF across the six queues.
- Do not upload checkpoints, source trees, or this release directory in place of the task SIFs.
"""


def finalize_release(release_dir: str | Path) -> tuple[bool, list[str]]:
    release_dir, payload = _load_release(release_dir)
    blockers = readiness_blockers(release_dir, payload)
    _replace_text(release_dir / "SUBMISSION_CHECKLIST.md", _checklist(release_dir, payload, blockers))
    verify_path, _ = _matching_verify_receipt(release_dir, payload)
    evidence_paths = {"verify": verify_path or release_dir / "receipts" / "verify.json"}
    for collection_path in sorted((release_dir / "receipts").glob("qualification-collect-*.json")):
        evidence_paths[f"qualification-collect:{collection_path.stem}"] = collection_path
    for task in OFFICIAL_TASKS:
        for phase in ("build", "validator", "qualification"):
            evidence_paths[f"{phase}:{task}"] = release_dir / "receipts" / phase / f"{task}.json"
    build_lineage = {
        task: (_load_receipt(release_dir / "receipts" / "build" / f"{task}.json") or {}).get("pretrained_checkpoint_sha256")
        for task in OFFICIAL_TASKS
    }
    gate_material = {
        "release_manifest_sha256": payload["manifest_sha256"],
        "common_pretrained_checkpoint_sha256": payload["pretrained"]["sha256"],
        "task_pretrained_lineage_sha256": build_lineage,
        "blockers": blockers,
        "evidence_sha256": {
            name: sha256_file(path) if path.is_file() else None for name, path in sorted(evidence_paths.items())
        },
        "image_sha256": {
            task: sha256_file(_image_path(release_dir, payload, task))
            if _image_path(release_dir, payload, task).is_file()
            else None
            for task in OFFICIAL_TASKS
        },
    }
    gate_id = manifest_digest(gate_material)[:16]
    receipt = _receipt(
        "release",
        status="READY" if not blockers else "NOT_READY",
        created_at=_now(),
        **gate_material,
    )
    receipt_path = release_dir / "receipts" / "release" / f"{gate_id}.json"
    if receipt_path.is_file():
        prior = _load_receipt(receipt_path)
        if not prior or any(prior.get(key) != value for key, value in gate_material.items()):
            raise ReleaseError(f"Immutable finalization receipt conflicts at {receipt_path}.")
    else:
        _write_json_immutable(receipt_path, receipt)
    return not blockers, blockers


def release_id_from_input(input_path: str | Path) -> str:
    """Derive the stable, UTC, human-readable release directory name from a finalized input."""
    payload = load_manifest(input_path)
    candidate_value = payload.get("candidate_id")
    track_value = payload.get("track")
    if not isinstance(candidate_value, str) or not isinstance(track_value, str):
        raise ReleaseError("Candidate input must contain string candidate_id and track identifiers.")
    candidate = _safe_identifier(candidate_value, "candidate id")
    track = _safe_identifier(track_value, "track")
    created_at = payload.get("created_at")
    if not isinstance(created_at, str) or not created_at.strip():
        raise ReleaseError("A top-level release input must contain created_at so resume keeps one timestamped directory.")
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReleaseError("Candidate input created_at must be a valid ISO-8601 timestamp.") from exc
    if created.tzinfo is None:
        raise ReleaseError("Candidate input created_at must include a timezone.")
    timestamp = created.astimezone(UTC).strftime("%Y-%m-%d_%H%M%SZ")
    release_id = f"{timestamp}_{candidate}_{track}"
    if len(release_id) > 128:
        identity_digest = manifest_digest({"candidate_id": candidate, "track": track})[:12]
        release_id = f"{timestamp}_{candidate[:40]}_{track[:40]}_{identity_digest}"
    return _safe_identifier(release_id, "derived release id")


def _write_phase_log(release_dir: Path, phase: str, result: Any) -> None:
    path = release_dir / "logs" / f"{phase}.json"
    if not path.exists():
        _write_json_immutable(path, {"phase": phase, "result": result})


def _operator_scaffold(release_dir: Path, *, input_path: Path | None = None) -> None:
    (release_dir / "logs").mkdir(parents=True, exist_ok=True)
    (release_dir / "manifests").mkdir(parents=True, exist_ok=True)
    manifest = release_dir / "release_manifest.json"
    if manifest.is_file():
        _write_bytes_immutable(release_dir / "manifests" / "release_manifest.snapshot.json", manifest.read_bytes())
    if input_path is not None:
        _write_bytes_immutable(release_dir / "manifests" / "candidate_input.json", input_path.read_bytes())


def _operator_readme(
    release_dir: Path,
    *,
    completed_phase: str,
    candidate_ready: bool = False,
    blockers: list[str] | None = None,
    failure: str | None = None,
) -> str:
    payload = load_manifest(release_dir / "release_manifest.json")
    mode = payload["release_mode"]
    if candidate_ready:
        status = "FOMO26_CANDIDATE_READY_TO_SUBMIT"
    elif failure:
        status = f"NOT_READY — {failure}"
    elif mode == "smoke_only":
        status = "ENGINEERING_SMOKE_ONLY — NOT_READY"
    elif completed_phase == "finalize":
        status = "NOT_READY — " + ", ".join(blockers or ["unknown_finalization_blocker"])
    else:
        status = f"IN_PROGRESS — completed through {completed_phase}"
    rows = []
    for task in OFFICIAL_TASKS:
        image = _image_path(release_dir, payload, task)
        if image.is_file():
            rows.append(f"- `{task}`: `{image.name}` — {image.stat().st_size} bytes — `{sha256_file(image)}`")
        else:
            rows.append(f"- `{task}`: `{image.name}` — NOT BUILT")
    blocker_lines = "\n".join(f"- `{blocker}`" for blocker in (blockers or [])) or "- None recorded at this phase."
    return f"""# FOMO26 release operator summary

This directory is the single resumable release workspace for candidate `{payload["candidate_id"]}`,
track `{payload["track"]}`. Its stable release id is `{payload.get("release_id", release_dir.name)}`.

- Release mode: `{mode}`
- Completed phase: `{completed_phase}`
- Status: **{status}**
- Release manifest SHA-256: `{payload["manifest_sha256"]}`
- Common pretrained checkpoint SHA-256: `{payload["pretrained"]["sha256"]}`

## Current blockers

{blocker_lines}

## Submission images

{chr(10).join(rows)}

Only the six files in `images/` are Synapse container upload artifacts. Inspect
`SUBMISSION_CHECKLIST.md` after finalization. Never upload a `smoke_only` release, and never interpret
structural CPU validation as H100 evidence. Detailed immutable evidence is under `receipts/`; operator
logs are under `logs/`; frozen manifest snapshots are under `manifests/`.
"""


def _pipeline_summary(
    release_dir: Path,
    *,
    completed_phase: str,
    phases: dict[str, Any],
    candidate_ready: bool = False,
    blockers: list[str] | None = None,
) -> dict[str, Any]:
    _replace_text(
        release_dir / "README_RELEASE.md",
        _operator_readme(
            release_dir,
            completed_phase=completed_phase,
            candidate_ready=candidate_ready,
            blockers=blockers,
        ),
    )
    payload = load_manifest(release_dir / "release_manifest.json")
    return {
        "phase": "release",
        "status": "PASS" if candidate_ready else "NOT_READY" if completed_phase == "finalize" else "IN_PROGRESS",
        "release_id": release_dir.name,
        "release_dir": str(release_dir),
        "release_mode": payload["release_mode"],
        "completed_phase": completed_phase,
        "candidate_ready": candidate_ready,
        "blockers": blockers or [],
        "phases": phases,
    }


def orchestrate_release(
    *,
    output_root: str | Path,
    input_path: str | Path | None = None,
    remote: str | None = None,
    release_id: str | None = None,
    mode: str | None = None,
    through: str = "finalize",
    export_root: str | Path | None = None,
    fixture_cache: str | Path | None = None,
    apptainer: str = "apptainer",
    no_gpu: bool = False,
    validator_timeout: int = 900,
    transfer_attempts: int = 3,
    transfer_timeout: int = 300,
    qualification_host: str = "jeanzay",
    qualification_remote_dir: str | None = None,
) -> dict[str, Any]:
    """Run existing release primitives in order without making any scientific choice."""
    if (input_path is None) == (remote is None):
        raise ReleaseError("Exactly one of input_path or remote is required.")
    valid_through = {"audit", "export", "pull", "verify", "stage", "build", "validate", "qualify-plan", "finalize"}
    if through not in valid_through:
        raise ReleaseError(f"Unsupported release completion phase {through!r}.")
    if through in {"validate", "qualify-plan", "finalize"} and fixture_cache is None:
        raise ReleaseError(f"fixture_cache is required when running through {through}.")
    if through == "qualify-plan" and not qualification_remote_dir:
        raise ReleaseError("qualification_remote_dir is required for --through qualify-plan.")

    output_root = Path(output_root).resolve()
    phases: dict[str, Any] = {}
    source_input: Path | None = None
    if input_path is not None:
        if through == "pull":
            raise ReleaseError("--through pull requires --remote, not --input.")
        source_input = Path(input_path).resolve()
        derived_id = release_id_from_input(source_input)
        if release_id is not None and release_id != derived_id:
            raise ReleaseError(f"Explicit release id {release_id!r} does not match derived id {derived_id!r}.")
        release_id = derived_id
    else:
        if through in {"audit", "export"}:
            raise ReleaseError(f"--through {through} requires --input, not --remote.")
        if mode is not None:
            raise ReleaseError("--mode belongs to the remote release manifest and cannot be overridden during pull.")
        if release_id is None:
            raise ReleaseError("release_id is required with remote pull; remote directory crawling is forbidden.")
        _safe_identifier(release_id, "release id")

    assert release_id is not None
    release_dir = output_root / release_id
    completed_phase = "none"
    try:
        if source_input is not None:
            release_mode = mode or "final"
            phases["audit"] = audit_candidate(source_input, release_dir, mode=release_mode)
            completed_phase = "audit"
            _operator_scaffold(release_dir, input_path=source_input)
            _write_phase_log(release_dir, "audit", phases["audit"])
            if through == "audit":
                return _pipeline_summary(release_dir, completed_phase=completed_phase, phases=phases)
            if export_root is not None:
                phases["export"] = export_release(release_dir, export_root)
                completed_phase = "export"
                _write_phase_log(release_dir, "export", phases["export"])
            if through == "export":
                if export_root is None:
                    raise ReleaseError("export_root is required for --through export.")
                return _pipeline_summary(release_dir, completed_phase=completed_phase, phases=phases)
        else:
            phases["pull"] = pull_release(
                remote or "",
                output_root,
                release_id=release_id,
                attempts=transfer_attempts,
                timeout=transfer_timeout,
            )
            completed_phase = "pull"
            _operator_scaffold(release_dir)
            pulled_payload = load_manifest(release_dir / "release_manifest.json")
            if pulled_payload.get("release_id") not in {None, release_id}:
                raise ReleaseError(
                    f"Pulled manifest release_id {pulled_payload.get('release_id')!r} does not match {release_id!r}."
                )
            _write_phase_log(release_dir, "pull", phases["pull"])
            if through == "pull":
                return _pipeline_summary(release_dir, completed_phase=completed_phase, phases=phases)

        phases["verify-source"] = verify_release(release_dir)
        completed_phase = "verify"
        _write_phase_log(release_dir, "verify-source", phases["verify-source"])
        if through == "verify":
            return _pipeline_summary(release_dir, completed_phase=completed_phase, phases=phases)

        phases["stage"] = stage_release(release_dir)
        completed_phase = "stage"
        _write_phase_log(release_dir, "stage", phases["stage"])
        if through == "stage":
            return _pipeline_summary(release_dir, completed_phase=completed_phase, phases=phases)

        phases["build"] = build_release(release_dir, apptainer=apptainer)
        phases["verify-images"] = verify_release(release_dir)
        completed_phase = "build"
        _write_phase_log(release_dir, "build", phases["build"])
        _write_phase_log(release_dir, "verify-images", phases["verify-images"])
        if through == "build":
            return _pipeline_summary(release_dir, completed_phase=completed_phase, phases=phases)

        phases["validate"] = validate_images(
            release_dir,
            fixture_cache=fixture_cache or "",
            apptainer=apptainer,
            no_gpu=no_gpu,
            timeout=validator_timeout,
        )
        completed_phase = "validate"
        _write_phase_log(release_dir, "validate", phases["validate"])
        if through == "validate":
            return _pipeline_summary(release_dir, completed_phase=completed_phase, phases=phases)

        if qualification_remote_dir:
            phases["qualify-plan"] = qualify_release(
                release_dir,
                host=qualification_host,
                remote_dir=qualification_remote_dir,
                fixture_cache=fixture_cache or "",
            )
            completed_phase = "qualify-plan"
            _write_phase_log(release_dir, "qualify-plan", phases["qualify-plan"])
        if through == "qualify-plan":
            return _pipeline_summary(release_dir, completed_phase=completed_phase, phases=phases)

        candidate_ready, blockers = finalize_release(release_dir)
        completed_phase = "finalize"
        phases["finalize"] = {"candidate_ready": candidate_ready, "blockers": blockers}
        _write_phase_log(release_dir, "finalize", phases["finalize"])
        return _pipeline_summary(
            release_dir,
            completed_phase=completed_phase,
            phases=phases,
            candidate_ready=candidate_ready,
            blockers=blockers,
        )
    except Exception as exc:
        if (release_dir / "release_manifest.json").is_file():
            failure_id = manifest_digest({"phase": completed_phase, "time": _now(), "error": str(exc)})[:16]
            _write_json_immutable(
                release_dir / "logs" / "failures" / f"{failure_id}.json",
                {"status": "FAIL", "completed_phase": completed_phase, "error": str(exc)},
            )
            _replace_text(
                release_dir / "README_RELEASE.md",
                _operator_readme(release_dir, completed_phase=completed_phase, failure=str(exc)),
            )
        raise


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fomo26-release", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validator = commands.add_parser("validator", help="Pinned official validator operations")
    validator_commands = validator.add_subparsers(dest="validator_command", required=True)
    validator_verify = validator_commands.add_parser("verify")
    validator_verify.add_argument("--check-upstream-head", action="store_true")
    # The validator is not in this repository: point at an acquired checkout, or set
    # FOMO26_VALIDATOR_ROOT. `release.py validator acquire` fetches the pinned commit.
    validator_verify.add_argument("--validator-root", type=Path, default=None)
    validator_bootstrap = validator_commands.add_parser("bootstrap-fixtures")
    validator_bootstrap.add_argument("--cache", type=Path, required=True)
    validator_bootstrap.add_argument("--validator-root", type=Path, default=None)
    validator_acquire = validator_commands.add_parser(
        "acquire", help="Download the pinned official validator commit and verify it"
    )
    validator_acquire.add_argument("--destination", type=Path, required=True)
    validator_acquire.add_argument("--timeout", type=int, default=120)

    adapter = commands.add_parser("adapt-handoff", help="Translate one immutable scientific model set")
    adapter.add_argument("--handoff", type=Path, required=True)
    adapter.add_argument("--expected-handoff-sha256", required=True)
    adapter.add_argument("--backup-root", type=Path, required=True)
    adapter.add_argument("--candidate", required=True)
    adapter.add_argument("--output", type=Path, required=True)

    complete = commands.add_parser(
        "release",
        help="Run the existing release phases as one resumable timestamped operator workflow",
    )
    source = complete.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Finalized candidate manifest with declared artifact paths")
    source.add_argument("--remote", help="Exact host:path of an exported release; no remote crawling")
    complete.add_argument("--output-root", type=Path, required=True)
    complete.add_argument("--release-id", help="Required with --remote; derived and checked with --input")
    complete.add_argument("--mode", choices=["final", "smoke_only"])
    complete.add_argument(
        "--through",
        choices=["audit", "export", "pull", "verify", "stage", "build", "validate", "qualify-plan", "finalize"],
        default="finalize",
    )
    complete.add_argument("--export-root", type=Path)
    complete.add_argument("--fixture-cache", type=Path)
    complete.add_argument("--apptainer", default="apptainer")
    complete.add_argument("--no-gpu", action="store_true")
    complete.add_argument("--validator-timeout", type=int, default=900)
    complete.add_argument("--transfer-attempts", type=int, default=3)
    complete.add_argument("--transfer-timeout", type=int, default=300)
    complete.add_argument("--qualification-host", default="jeanzay")
    complete.add_argument("--qualification-remote-dir")

    audit = commands.add_parser("audit")
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--release-dir", type=Path, required=True)
    audit.add_argument("--mode", choices=["final", "smoke_only"], required=True)

    export = commands.add_parser("export")
    export.add_argument("--release-dir", type=Path, required=True)
    export.add_argument("--destination-root", type=Path, default=None)

    pull = commands.add_parser("pull")
    pull.add_argument("--remote", required=True)
    pull.add_argument("--destination-root", type=Path, required=True)
    pull.add_argument("--release-id", required=True)
    pull.add_argument("--attempts", type=int, default=3)
    pull.add_argument("--timeout", type=int, default=300)
    pull.add_argument("--dry-run", action="store_true")

    verify = commands.add_parser("verify")
    verify.add_argument("--release-dir", type=Path, required=True)

    stage = commands.add_parser("stage")
    stage.add_argument("--release-dir", type=Path, required=True)

    build = commands.add_parser("build")
    build.add_argument("--release-dir", type=Path, required=True)
    build.add_argument("--apptainer", default="apptainer")

    validate = commands.add_parser("validate")
    validate.add_argument("--release-dir", type=Path, required=True)
    validate.add_argument("--fixture-cache", type=Path, required=True)
    validate.add_argument("--apptainer", default="apptainer")
    validate.add_argument("--no-gpu", action="store_true")
    validate.add_argument("--timeout", type=int, default=900)

    local = commands.add_parser("run-local")
    local.add_argument("--release-dir", type=Path, required=True)
    local.add_argument("--task", choices=OFFICIAL_TASKS, required=True)
    local.add_argument("--apptainer", default="apptainer")
    local.add_argument("--gpu", action="store_true")
    local.add_argument("arguments", nargs=argparse.REMAINDER)

    qualify = commands.add_parser("qualify")
    qualify.add_argument("--release-dir", type=Path, required=True)
    qualify.add_argument("--fixture-cache", type=Path, required=True)
    qualify.add_argument("--host", default="jeanzay")
    qualify.add_argument("--remote-dir", required=True)
    qualify.add_argument("--submit", action="store_true")
    qualify.add_argument("--collect", action="store_true")
    qualify.add_argument("--job-id")

    finalize = commands.add_parser("finalize")
    finalize.add_argument("--release-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validator":
            if args.validator_command == "verify":
                result = {
                    "manifest": verify_manifest(),
                    "snapshot": verify_snapshot(args.validator_root),
                    "entrypoints": verify_entrypoint_contract(),
                }
                if args.check_upstream_head:
                    result["upstream_head"] = upstream_head_drift()
            elif args.validator_command == "acquire":
                result = acquire_validator(args.destination, timeout=args.timeout)
            else:
                result = bootstrap_fixtures(args.cache, args.validator_root)
        elif args.command == "adapt-handoff":
            manifest = adapt_handoff(
                handoff_path=args.handoff,
                expected_handoff_sha256=args.expected_handoff_sha256,
                backup_root=args.backup_root,
                candidate_id=args.candidate,
                packaging_git_commit=_git_head(),
                output_path=args.output,
            )
            result = {
                "status": "PASS",
                "candidate_id": manifest["candidate_id"],
                "output": str(args.output.resolve()),
                "manifest_sha256": manifest["manifest_sha256"],
                "selected_members": {task: manifest["tasks"][task]["selected_members"] for task in DOWNSTREAM_TASKS},
                "unresolved_true_scientific_authority": manifest["derivation_receipt"]["unresolved_true_scientific_authority"],
            }
        elif args.command == "release":
            result = orchestrate_release(
                input_path=args.input,
                remote=args.remote,
                output_root=args.output_root,
                release_id=args.release_id,
                mode=args.mode,
                through=args.through,
                export_root=args.export_root,
                fixture_cache=args.fixture_cache,
                apptainer=args.apptainer,
                no_gpu=args.no_gpu,
                validator_timeout=args.validator_timeout,
                transfer_attempts=args.transfer_attempts,
                transfer_timeout=args.transfer_timeout,
                qualification_host=args.qualification_host,
                qualification_remote_dir=args.qualification_remote_dir,
            )
            _print_json(result)
            return 0 if result.get("candidate_ready") or result.get("completed_phase") != "finalize" else 2
        elif args.command == "audit":
            result = audit_candidate(args.input, args.release_dir, mode=args.mode)
        elif args.command == "export":
            root = args.destination_root
            if root is None:
                scratch = os.environ.get("SCRATCH")
                if not scratch:
                    raise ReleaseError("--destination-root is required when $SCRATCH is not set.")
                root = Path(scratch) / "fomo26" / "releases"
            result = export_release(args.release_dir, root)
        elif args.command == "pull":
            result = pull_release(
                args.remote,
                args.destination_root,
                release_id=args.release_id,
                attempts=args.attempts,
                timeout=args.timeout,
                dry_run=args.dry_run,
            )
        elif args.command == "verify":
            result = verify_release(args.release_dir)
        elif args.command == "stage":
            result = stage_release(args.release_dir)
        elif args.command == "build":
            result = build_release(args.release_dir, apptainer=args.apptainer)
        elif args.command == "validate":
            result = validate_images(
                args.release_dir,
                fixture_cache=args.fixture_cache,
                apptainer=args.apptainer,
                no_gpu=args.no_gpu,
                timeout=args.timeout,
            )
        elif args.command == "run-local":
            result = run_local(
                args.release_dir,
                task=args.task,
                arguments=args.arguments,
                apptainer=args.apptainer,
                gpu=args.gpu,
            )
        elif args.command == "qualify":
            result = qualify_release(
                args.release_dir,
                host=args.host,
                remote_dir=args.remote_dir,
                fixture_cache=args.fixture_cache,
                submit=args.submit,
                collect=args.collect,
                job_id=args.job_id,
            )
        else:
            ready, blockers = finalize_release(args.release_dir)
            if ready:
                print("FOMO26_CANDIDATE_READY_TO_SUBMIT")
                return 0
            print("NOT_READY — " + ", ".join(blockers))
            return 2
        _print_json(result)
        return 0
    except (ReleaseError, ReleaseContractError, ValidatorError, OSError, ValueError) as exc:
        print(f"NOT_READY — {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
