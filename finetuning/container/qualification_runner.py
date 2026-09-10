"""Run one official validator unit on a qualification node and emit a machine receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

SUCCESS_RE = re.compile(r"ALL\s+(\d+)\s+TESTS\s+PASSED")
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _fixture_digest(validator_root: Path, fixture_cache: Path) -> str:
    provenance = json.loads((validator_root / "PROVENANCE.json").read_text())
    entries = {entry["path"]: entry for entry in provenance["entries"]}
    source_manifest = validator_root / "upstream" / "container_validator" / "data" / "manifest.yaml"
    cache_manifest = fixture_cache / "manifest.yaml"
    if not cache_manifest.is_file() or _sha256(cache_manifest) != _sha256(source_manifest):
        raise SystemExit("Transferred fixture manifest is missing or drifted.")
    manifest_text = source_manifest.read_text()
    paths = sorted(set(re.findall(r"inputs/[A-Za-z0-9_./-]+\.nii(?:\.gz)?", manifest_text)))
    digest = hashlib.sha256()
    for relative in paths:
        inventory_path = f"container_validator/data/{relative}"
        entry = entries.get(inventory_path)
        if not isinstance(entry, dict) or not isinstance(entry.get("lfs"), dict):
            raise SystemExit(f"Fixture is absent from validator provenance: {inventory_path}")
        fixture = fixture_cache / relative
        if not fixture.is_file():
            raise SystemExit(f"Transferred fixture is missing: {fixture}")
        if fixture.read_bytes()[: len(LFS_HEADER)] == LFS_HEADER:
            raise SystemExit(f"Transferred fixture is an LFS pointer: {fixture}")
        expected_size = entry["lfs"]["size"]
        expected_sha = entry["lfs"]["oid_sha256"]
        if fixture.stat().st_size != expected_size or _sha256(fixture) != expected_sha:
            raise SystemExit(f"Transferred fixture bytes drifted: {fixture}")
        digest.update(inventory_path.encode())
        digest.update(bytes.fromhex(expected_sha))
    return digest.hexdigest()


def _command_text(command: list[str]) -> str | None:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
    except OSError:
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--sif", type=Path, required=True)
    parser.add_argument("--validator-root", type=Path, required=True)
    parser.add_argument("--fixture-cache", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--threshold-seconds", type=float, required=True)
    parser.add_argument("--release-manifest-sha256", required=True)
    parser.add_argument("--validator-sha256", required=True)
    parser.add_argument("--fixture-digest", required=True)
    parser.add_argument("--expected-sif-sha256", required=True)
    args = parser.parse_args(argv)

    before = _sha256(args.sif)
    if before != args.expected_sif_sha256:
        raise SystemExit(
            f"Prepared SIF digest differs from the local build: expected {args.expected_sif_sha256}, observed {before}"
        )
    observed_validator_sha = _tree_sha256(args.validator_root)
    if observed_validator_sha != args.validator_sha256:
        raise SystemExit(
            f"Transferred validator snapshot drifted: expected {args.validator_sha256}, observed {observed_validator_sha}"
        )
    observed_fixture_digest = _fixture_digest(args.validator_root, args.fixture_cache)
    if observed_fixture_digest != args.fixture_digest:
        raise SystemExit(
            f"Transferred fixture inventory drifted: expected {args.fixture_digest}, observed {observed_fixture_digest}"
        )
    validator = args.validator_root / "upstream" / "container_validator" / "validate.py"
    with tempfile.TemporaryDirectory(prefix="fomo26-singularity-compat-") as temporary_dir:
        compatibility = Path(temporary_dir) / "apptainer"
        compatibility.write_text(
            "#!/usr/bin/env python3\n"
            "import os, sys\n"
            "args = ['--nv' if arg == '--nvccli' else arg for arg in sys.argv[1:]]\n"
            "os.execvp('singularity', ['singularity', *args])\n"
        )
        compatibility.chmod(0o755)
        command = [
            sys.executable,
            str(validator),
            "--task",
            args.task,
            "--sif",
            str(args.sif),
            "--manifest",
            str(args.fixture_cache / "manifest.yaml"),
            "--apptainer",
            str(compatibility),
            "--timeout",
            str(max(1, int(args.threshold_seconds))),
        ]
        started_at = _now()
        started = time.monotonic()
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run(command, capture_output=True, text=True, check=False, env=environment)
    wall = time.monotonic() - started
    finished_at = _now()
    after = _sha256(args.sif)
    summary = SUCCESS_RE.search(proc.stdout)
    gpu_name = _command_text(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    cpu_count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    passed = (
        proc.returncode == 0
        and summary is not None
        and before == after
        and wall < args.threshold_seconds
        and gpu_name is not None
        and "h100" in gpu_name.lower()
    )
    receipt = {
        "schema_version": "fomo26-h100-qualification-receipt-v1",
        "status": "PASS" if passed else "FAIL",
        "task": args.task,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_seconds": wall,
        "threshold_seconds": args.threshold_seconds,
        "release_manifest_sha256": args.release_manifest_sha256,
        "returncode": proc.returncode,
        "success_summary": summary.group(0) if summary else None,
        "tests_passed": int(summary.group(1)) if summary else None,
        "sif_sha256_before": before,
        "sif_sha256_after": after,
        "validator_sha256": args.validator_sha256,
        "fixture_digest": args.fixture_digest,
        "gpu_name": gpu_name,
        "cpu_allocation": cpu_count,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "host": platform.node(),
        "peak_cuda_memory_mib": None,
        "peak_cuda_memory_observable": False,
        "command": command,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    if args.receipt.exists():
        raise SystemExit(f"Refusing to overwrite immutable qualification receipt {args.receipt}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{args.receipt.name}.", suffix=".partial", dir=args.receipt.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args.receipt)
    finally:
        if temporary.exists():
            temporary.unlink()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
