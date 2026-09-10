"""Bridge to the official FOMO26 container validator, which this repository does not contain.

The validator used to be vendored here. It is not any more: its upstream declares no licence, so
shipping 84 of its files inside a public release would be redistribution on undefined terms. What
survives is `validator_manifest.json` -- our own record of the pinned commit and the digest of
every member -- which is exactly what makes an independently acquired copy checkable.

Acquisition is explicit and user-invoked (`acquire_validator`). Nothing here touches the network
during import, test collection, CI, training, inference or a container build, and only the pinned
commit is ever fetched -- never main, HEAD, latest or a tag.

Absence is a failure, never a pass and never a silent skip: every entry point that needs the
validator resolves a root first and raises `ValidatorError` when there is not a verified one.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from finetuning.container.official_contract import official_contract
from finetuning.container.release_manifest import sha256_file
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).resolve().parent / "validator_manifest.json"
PINNED_COMMIT = "d442af2e9bdade58be20c2ee0cbabf8d0439e32b"
PINNED_TREE = "091c889c7c3d16204039f67b190e9a794e61a7f8"
UPSTREAM_URL = "https://github.com/fomo26/container-validator.git"
#: Where an acquired copy lives. Never inside this repository, and never defaulted silently.
ROOT_ENV = "FOMO26_VALIDATOR_ROOT"
#: A source checkout of this project is a few MB; the cap is generous but finite so a hostile or
#: broken response cannot fill the disk.
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 120
SUCCESS_RE = re.compile(r"ALL\s+(\d+)\s+TESTS\s+PASSED")
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"
WRAPPERS = {
    "task1": "predict_task1.py",
    "task2": "predict_task2.py",
    "task3": "predict_task3.py",
    "task4": "predict_task4.py",
    "task5": "predict_task5.py",
    "task6_and_7": "predict_task6_7.py",
}


class ValidatorError(RuntimeError):
    """Pinned validator files, fixtures, contracts, or execution are invalid."""


def _metadata() -> dict[str, Any]:
    try:
        payload = json.loads(MANIFEST.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidatorError(f"Cannot read validator manifest {MANIFEST}: {exc}") from exc
    return payload


def validator_root(root: str | Path | None = None) -> Path:
    """Resolve an acquired validator checkout, or say plainly that there is not one.

    Order: explicit argument, then ``$FOMO26_VALIDATOR_ROOT``. There is deliberately no in-repo
    default -- a default would be a path that cannot exist, and the resulting error would look like
    a bug rather than the missing prerequisite it is.
    """
    candidate = root if root is not None else os.environ.get(ROOT_ENV)
    if not candidate:
        raise ValidatorError(
            "No official validator available. This repository does not contain it: its upstream "
            f"declares no licence, so it is not redistributed here. Acquire it with "
            f"acquire_validator(<dir>) or set {ROOT_ENV} to an existing verified checkout of "
            f"{UPSTREAM_URL} at commit {PINNED_COMMIT}."
        )
    path = Path(candidate).expanduser().resolve()
    if not path.is_dir():
        raise ValidatorError(f"Validator root does not exist or is not a directory: {path}.")
    return path


def validator_sha256() -> str:
    """Digest the pinned validator identity from the manifest.

    Previously this walked the vendored tree. It now digests the manifest's own member list, which
    is the same identity by a different route: the same 84 paths and digests, in the same order.
    It stays computable with no validator present, so a release receipt can still record *which*
    validator it was pinned to.
    """
    metadata = _metadata()
    digest = hashlib.sha256()
    digest.update(metadata["upstream_commit"].encode())
    digest.update(b"\0")
    for entry in sorted(metadata["entries"], key=lambda item: item["path"]):
        digest.update(entry["path"].encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(entry["snapshot_sha256"]))
    return digest.hexdigest()


def verify_snapshot(root: str | Path | None = None) -> dict[str, Any]:
    """Verify an acquired checkout against the manifest: exact member set, exact bytes.

    Set equality, not containment. A missing file, an extra file and a modified file are all
    refusals, because "the parts I checked were fine" is not a statement about a validator.
    """
    base = validator_root(root)
    metadata = _metadata()
    if metadata.get("upstream_commit") != PINNED_COMMIT or metadata.get("upstream_tree") != PINNED_TREE:
        raise ValidatorError("Validator manifest does not contain the approved commit/tree pins.")
    entries = metadata.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValidatorError("Validator manifest has no file inventory.")
    by_path = {entry.get("path"): entry for entry in entries if isinstance(entry, dict)}
    if len(by_path) != len(entries):
        raise ValidatorError("Validator manifest contains a missing or duplicate path.")

    expected_files: set[str] = set()
    lfs_count = 0
    for rel, entry in by_path.items():
        if not isinstance(rel, str) or rel.startswith("/") or ".." in Path(rel).parts:
            raise ValidatorError(f"Unsafe validator inventory path: {rel!r}.")
        path = base / rel
        if entry.get("vendored"):
            expected_files.add(rel)
            if path.is_symlink():
                raise ValidatorError(f"Validator member is a symlink: {rel}.")
            if not path.is_file():
                raise ValidatorError(f"Validator file missing from the acquired checkout: {rel}.")
            if path.stat().st_size != entry.get("snapshot_size") or sha256_file(path) != entry.get("snapshot_sha256"):
                raise ValidatorError(f"Validator file does not match the pinned digest: {rel}.")
        else:
            lfs_count += 1
            lfs = entry.get("lfs")
            if not isinstance(lfs, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(lfs.get("oid_sha256", ""))):
                raise ValidatorError(f"Missing LFS OID for {rel}.")
            if not isinstance(lfs.get("size"), int) or lfs["size"] <= 0:
                raise ValidatorError(f"Missing LFS size for {rel}.")

    observed = {
        path.relative_to(base).as_posix()
        for path in base.rglob("*")
        if path.is_file() and not path.is_symlink() and ".git/" not in path.relative_to(base).as_posix()
    }
    # LFS payloads are pointers upstream; an acquired copy may or may not have resolved them, so
    # they are compared by record above and excluded from the member-set comparison here.
    observed -= {rel for rel, entry in by_path.items() if not entry.get("vendored")}
    if observed != expected_files:
        raise ValidatorError(
            f"Acquired validator file set does not match the manifest: "
            f"missing={sorted(expected_files - observed)[:5]}, extra={sorted(observed - expected_files)[:5]}."
        )
    return {
        "status": "VERIFIED",
        "root": str(base),
        "commit": PINNED_COMMIT,
        "tree": PINNED_TREE,
        "vendored_files": len(expected_files),
        "lfs_objects": lfs_count,
        "validator_sha256": validator_sha256(),
        "license_declared_upstream": metadata["upstream_license"]["declared"],
    }


def verify_manifest() -> dict[str, Any]:
    """Offline integrity of our own record, with no validator present.

    De-vendoring removed the thing `verify_snapshot` used to check locally, but it did not remove
    the need for a local gate: the manifest still has to be well-formed and still has to name the
    approved commit. This is that gate. It deliberately does NOT claim the validator is present or
    correct -- only `verify_snapshot(root)` can say that, and it needs an acquired copy.
    """
    metadata = _metadata()
    if metadata.get("upstream_commit") != PINNED_COMMIT or metadata.get("upstream_tree") != PINNED_TREE:
        raise ValidatorError("Validator manifest does not contain the approved commit/tree pins.")
    entries = metadata.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValidatorError("Validator manifest has no file inventory.")
    paths = [entry.get("path") for entry in entries if isinstance(entry, dict)]
    if len(set(paths)) != len(entries):
        raise ValidatorError("Validator manifest contains a missing or duplicate path.")
    vendored = sum(1 for entry in entries if entry.get("vendored"))
    return {
        "status": "VERIFIED",
        "scope": "manifest only; the validator itself is external and not checked here",
        "commit": PINNED_COMMIT,
        "members": vendored,
        "lfs_objects": len(entries) - vendored,
        "validator_sha256": validator_sha256(),
        "redistributed_here": False,
    }


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    """Extract with every unsafe member class refused rather than sanitised.

    Refusing is the point: a traversal path, an absolute path, a symlink or a hard link in an
    archive from a project that publishes no licence and no signature is not something to repair
    quietly -- it is a reason to stop.
    """
    root = destination.resolve()
    for member in archive.getmembers():
        if member.issym() or member.islnk():
            raise ValidatorError(f"Archive contains a link member, refusing: {member.name!r}.")
        if not (member.isfile() or member.isdir()):
            raise ValidatorError(f"Archive contains a special member, refusing: {member.name!r}.")
        name = member.name
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValidatorError(f"Archive member escapes the destination, refusing: {name!r}.")
        target = (root / name).resolve()
        if target != root and root not in target.parents:
            raise ValidatorError(f"Archive member resolves outside the destination, refusing: {name!r}.")
    archive.extractall(root)


def acquire_validator(
    destination: str | Path,
    *,
    timeout: int = DOWNLOAD_TIMEOUT_SECONDS,
    max_bytes: int = MAX_ARCHIVE_BYTES,
) -> dict[str, Any]:
    """Fetch the pinned validator commit into ``destination`` and verify it before returning.

    User-invoked only. The pinned commit is the sole thing ever requested -- resolving main, HEAD,
    latest or a tag would make "verified against the manifest" mean nothing, since the manifest
    describes one commit. Download goes to a temporary file, extraction to a temporary directory,
    and only a fully verified tree is moved into place, so a failure never leaves a half-tree that
    a later run might treat as real.
    """
    destination = Path(destination).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValidatorError(f"Refusing to overwrite a non-empty destination: {destination}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://codeload.github.com/fomo26/container-validator/tar.gz/{PINNED_COMMIT}"

    with tempfile.TemporaryDirectory(prefix=".fomo26-validator-", dir=destination.parent) as scratch:
        scratch_path = Path(scratch)
        archive_path = scratch_path / "validator.tar.gz"
        read = 0
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response, archive_path.open("wb") as stream:
                while chunk := response.read(1 << 20):
                    read += len(chunk)
                    if read > max_bytes:
                        raise ValidatorError(f"Validator archive exceeds {max_bytes} bytes; refusing.")
                    stream.write(chunk)
        except ValidatorError:
            raise
        except Exception as exc:  # noqa: BLE001 - urllib raises many unrelated types
            raise ValidatorError(f"Could not download the official validator from {url}: {exc}") from exc

        unpacked = scratch_path / "unpacked"
        unpacked.mkdir()
        with tarfile.open(archive_path, "r:gz") as archive:
            _safe_extract(archive, unpacked)
        roots = [child for child in unpacked.iterdir() if child.is_dir()]
        if len(roots) != 1:
            raise ValidatorError(f"Expected exactly one top-level directory in the archive, found {len(roots)}.")
        staged = scratch_path / "staged"
        os.replace(roots[0], staged)
        report = verify_snapshot(staged)
        if destination.exists():
            destination.rmdir()
        os.replace(staged, destination)

    report["root"] = str(destination)
    report["acquired_from"] = url
    return report


def upstream_head_drift() -> dict[str, Any]:
    """Report upstream HEAD without changing the snapshot or local Git state."""
    try:
        proc = subprocess.run(
            ["git", "ls-remote", "https://github.com/fomo26/container-validator.git", "refs/heads/main"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "NOT_VERIFIED", "error": str(exc), "pinned_commit": PINNED_COMMIT}
    head = proc.stdout.split()[0] if proc.returncode == 0 and proc.stdout.split() else None
    return {
        "status": "VERIFIED" if head else "NOT_VERIFIED",
        "pinned_commit": PINNED_COMMIT,
        "upstream_head": head,
        "drift": None if head is None else head != PINNED_COMMIT,
        "stderr": proc.stderr,
    }


def _declared_flags(wrapper: Path) -> set[str]:
    tree = ast.parse(wrapper.read_text(), filename=str(wrapper))
    flags = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str) and argument.value.startswith("--"):
                flags.add(argument.value)
    return flags


def _declared_output_format(wrapper: Path) -> str | None:
    tree = ast.parse(wrapper.read_text(), filename=str(wrapper))
    call_names = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    if "predict_seg" in call_names:
        return "nifti"
    if "predict_clsreg" in call_names:
        return "txt"
    if "extract_embedding" in call_names:
        return "numpy"
    return None


def verify_entrypoint_contract() -> dict[str, Any]:
    contract = official_contract()
    if set(contract) != set(WRAPPERS):
        raise ValidatorError(f"Official tasks and local wrappers differ: {sorted(contract)} vs {sorted(WRAPPERS)}.")
    compared = {}
    for task, filename in WRAPPERS.items():
        wrapper = REPO / "finetuning" / "container" / filename
        declared = _declared_flags(wrapper)
        required = {contract[task]["output_flag"]}
        for input_spec in contract[task]["inputs"]:
            required.update(input_spec["flags"])
        missing = sorted(required - declared)
        if missing:
            raise ValidatorError(f"{filename} is missing official flags {missing}.")
        declared_format = _declared_output_format(wrapper)
        if declared_format != contract[task]["output_format"]:
            raise ValidatorError(
                f"{filename} declares output format {declared_format!r}, official contract requires "
                f"{contract[task]['output_format']!r}."
            )
        compared[task] = {
            "wrapper": filename,
            "official_flags": sorted(required),
            "output_format": declared_format,
        }
    return compared


def _fixture_entries(root: str | Path | None = None) -> list[dict[str, Any]]:
    metadata = _metadata()
    base = validator_root(root)
    manifest = (base / "container_validator" / "data" / "manifest.yaml").read_text()
    paths = sorted(set(re.findall(r"inputs/[A-Za-z0-9_./-]+\.nii(?:\.gz)?", manifest)))
    by_path = {entry["path"]: entry for entry in metadata["entries"]}
    result = []
    for relative in paths:
        upstream_relative = f"container_validator/data/{relative}"
        entry = by_path.get(upstream_relative)
        if not entry or "lfs" not in entry:
            raise ValidatorError(f"Official fixture is absent from LFS inventory: {upstream_relative}.")
        result.append(entry)
    return result


def verify_fixtures(cache: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    cache = Path(cache)
    base = validator_root(root)
    fixture_entries = _fixture_entries(base)
    for entry in fixture_entries:
        relative = Path(entry["path"]).relative_to("container_validator/data")
        path = cache / relative
        if not path.is_file():
            raise ValidatorError(f"Official fixture missing from cache: {path}.")
        prefix = path.read_bytes()[: len(LFS_HEADER)]
        if prefix == LFS_HEADER:
            raise ValidatorError(f"Official fixture is still an LFS pointer: {path}.")
        if path.stat().st_size != entry["lfs"]["size"]:
            raise ValidatorError(f"Official fixture has wrong size: {path}.")
        if sha256_file(path) != entry["lfs"]["oid_sha256"]:
            raise ValidatorError(f"Official fixture has wrong SHA-256: {path}.")
    manifest = cache / "manifest.yaml"
    source_manifest = base / "container_validator" / "data" / "manifest.yaml"
    if not manifest.is_file() or sha256_file(manifest) != sha256_file(source_manifest):
        raise ValidatorError(f"Fixture cache has a missing or changed manifest: {manifest}.")
    digest = hashlib.sha256()
    for entry in fixture_entries:
        digest.update(entry["path"].encode())
        digest.update(bytes.fromhex(entry["lfs"]["oid_sha256"]))
    return {"status": "VERIFIED", "fixtures": len(fixture_entries), "fixture_digest": digest.hexdigest()}


def bootstrap_fixtures(cache: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    """Download exact-commit validator fixtures into ``cache`` and verify every object."""
    base = validator_root(root)
    verify_snapshot(base)
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    source_manifest = base / "container_validator" / "data" / "manifest.yaml"
    manifest = cache / "manifest.yaml"
    if manifest.exists() and sha256_file(manifest) != sha256_file(source_manifest):
        raise ValidatorError(f"Refusing to overwrite conflicting fixture manifest {manifest}.")
    if not manifest.exists():
        manifest.write_bytes(source_manifest.read_bytes())
    for entry in _fixture_entries(base):
        relative = Path(entry["path"]).relative_to("container_validator/data")
        destination = cache / relative
        expected_size = entry["lfs"]["size"]
        expected_sha = entry["lfs"]["oid_sha256"]
        if destination.is_file() and destination.stat().st_size == expected_size and sha256_file(destination) == expected_sha:
            continue
        if destination.exists():
            raise ValidatorError(f"Refusing to overwrite corrupt or unexpected fixture {destination}.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://media.githubusercontent.com/media/fomo26/container-validator/{PINNED_COMMIT}/{entry['path']}"
        fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
                while chunk := response.read(1 << 20):
                    stream.write(chunk)
            if temporary.read_bytes()[: len(LFS_HEADER)] == LFS_HEADER:
                raise ValidatorError(f"GitHub returned an LFS pointer instead of fixture bytes for {entry['path']}.")
            if temporary.stat().st_size != expected_size or sha256_file(temporary) != expected_sha:
                raise ValidatorError(f"Downloaded fixture failed SHA/size verification: {entry['path']}.")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    return verify_fixtures(cache, base)


def run_official_validator(
    *,
    task: str,
    sif: str | Path,
    fixture_cache: str | Path,
    apptainer: str = "apptainer",
    no_gpu: bool = False,
    timeout: int = 900,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the unchanged official CLI and require both rc=0 and its terminal pass summary."""
    base = validator_root(root)
    verify_snapshot(base)
    verify_entrypoint_contract()
    fixture_report = verify_fixtures(fixture_cache, base)
    if task not in official_contract():
        raise ValidatorError(f"Unknown official task {task!r}.")
    sif = Path(sif).resolve()
    if not sif.is_file():
        raise ValidatorError(f"SIF not found: {sif}.")
    before = sha256_file(sif)
    command = [
        sys.executable,
        str(base / "container_validator" / "validate.py"),
        "--task",
        task,
        "--sif",
        str(sif),
        "--manifest",
        str(Path(fixture_cache).resolve() / "manifest.yaml"),
        "--apptainer",
        apptainer,
        "--timeout",
        str(timeout),
    ]
    if no_gpu:
        command.append("--no-gpu")
    try:
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(timeout * 3, 60),
            check=False,
            env=environment,
        )
        stdout, stderr, returncode, timed_out = proc.stdout, proc.stderr, proc.returncode, False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        returncode, timed_out = 124, True
    after = sha256_file(sif)
    match = SUCCESS_RE.search(stdout)
    passed = returncode == 0 and match is not None and before == after and not timed_out
    return {
        "status": "PASS" if passed else "FAIL",
        "task": task,
        "structural_only": no_gpu,
        "returncode": returncode,
        "timed_out": timed_out,
        "success_summary": match.group(0) if match else None,
        "tests_passed": int(match.group(1)) if match else None,
        "stdout": stdout,
        "stderr": stderr,
        "sif_sha256_before": before,
        "sif_sha256_after": after,
        "validator_sha256": validator_sha256(),
        "fixture_digest": fixture_report["fixture_digest"],
        "command": command,
    }
