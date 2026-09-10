"""Stage an Apptainer build context for one FOMO26 task.

Rewrites the orchestrator manifest's absolute run dirs to the in-container /app/models/runs/<fold>
layout and copies each fold's hydra/config.yaml + checkpoints/<name>.ckpt. Then prints the
`apptainer build` command. Run the container-validator afterwards.

Example:
    python -m finetuning.container.build_container --task 2 \
        --manifest $ASPARAGUS_RESULTS/manifests/task2_amaes_final.json \
        --out $CONTAINER_ROOT/task2
    apptainer build --fakeroot --arch amd64 $CONTAINER_ROOT/task2.sif \
        $CONTAINER_ROOT/task2/Apptainer.def
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from finetuning.fomo26_inference.pretrained_embedding import SSL_OBJECTIVE_DEFAULT_SOURCE
from finetuning.fomo26_inference.runtime_geometry import task_runtime_target_spacing
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent  # <repo>

MODEL_MANIFEST_SCHEMA = "fomo26-model-manifest-v1"
DOCKER_METADATA_SCHEMA = "fomo26-docker-metadata-v1"


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _read_provenance(manifest_path: Path) -> dict:
    """The fold manifest's sidecar envelope, when the bridge produced one.

    ``build_fold_manifest`` writes the bare record list the inference rail reads, and puts the
    authoritative campaign state (candidate, expected/present folds, partial, digests) in a
    ``.provenance.json`` sidecar. Packaging must consume that state rather than re-deriving a fold
    count from whatever happened to be staged -- otherwise an incomplete campaign packages silently
    as a complete one.
    """
    if manifest_path.is_dir():
        return {}
    sidecar = manifest_path.with_suffix(".provenance.json")
    if not sidecar.is_file():
        return {}
    return json.loads(sidecar.read_text())


def _assert_manifest_matches_staged(provenance: dict, staged: int) -> None:
    declared = provenance.get("trained_fold_count")
    if declared is not None and int(declared) != staged:
        raise SystemExit(
            f"Fold manifest declares trained_fold_count={declared} but {staged} fold checkpoint(s) "
            "were staged. Refusing to package a container whose contents contradict its provenance."
        )
    expected = provenance.get("expected_folds")
    if provenance.get("partial") and not provenance.get("partial_acknowledged", False):
        raise SystemExit(
            f"Fold manifest is marked partial ({staged} of {expected} folds): {provenance.get('partial_reason')}. "
            "Pass --allow-partial to package an incomplete ensemble knowingly; it changes what the ensemble is."
        )


def _write_model_manifest(out: Path, args, new_records: list[dict], source_records: list[dict]) -> None:
    """Emit the digest-bearing model manifest and the docker metadata the validator consumes.

    ``finetuning/container/container_contract.py`` both *read*
    ``docker_metadata.json``, but nothing in the repository wrote one, so the submission validator
    could never run against a real build. These are the missing producers.
    """
    provenance = getattr(args, "_provenance", None) or {}
    backbones = {r.get("pretrained_checkpoint_sha256") for r in source_records if r.get("pretrained_checkpoint_sha256")}
    model_manifest = {
        "schema_version": MODEL_MANIFEST_SCHEMA,
        "task": str(args.task),
        "checkpoint_name": args.checkpoint_name,
        # Apptainer.def's %post reads model_manifest["architecture"] unconditionally to decide
        # which runtime extras are needed. The task 6_7 and release-rail writers both
        # emit it; this legacy Tasks 1-5 writer did not, so every Tasks 1-5 image failed to build
        # with KeyError: 'architecture'. Recording it here also makes the manifest state which
        # architecture the packaged weights belong to.
        "architecture": getattr(args, "architecture", None),
        "fold_count": len(new_records),
        # Consumed from the fold manifest, not restated: the manifest is the authority on how many
        # folds the campaign actually trained and whether it is complete.
        "candidate": provenance.get("candidate"),
        "expected_folds": provenance.get("expected_folds"),
        "present_folds": provenance.get("present_folds"),
        "trained_fold_count": provenance.get("trained_fold_count", len(new_records)),
        "partial": provenance.get("partial", False),
        "partial_reason": provenance.get("partial_reason"),
        "inference_policy": provenance.get("inference_policy"),
        "git_commit": _git_commit(),
        # One shared pretrained backbone across folds is what makes the ensemble one model family;
        # the fold-manifest bridge already refuses a mixed campaign, this records the outcome.
        "pretrained_checkpoint_sha256": sorted(backbones)[0] if len(backbones) == 1 else None,
        "pretrained_checkpoint_sha256_conflict": sorted(backbones) if len(backbones) > 1 else None,
        "models": [
            {
                "fold": record.get("fold"),
                "path": record["best_ckpt"],
                "sha256": record["checkpoint_sha256"],
                "source_run_dir": record.get("source_run_dir"),
            }
            for record in new_records
        ],
    }
    (out / "models" / "model_manifest.json").write_text(json.dumps(model_manifest, indent=2, sort_keys=True) + "\n")

    docker_metadata = {
        "schema_version": DOCKER_METADATA_SCHEMA,
        "entrypoints": [str(args.task)],
        # The challenge container runs offline on the evaluation platform. These are asserted by
        # container_contract.py; stating them here is what lets that assertion mean something.
        "network_required": False,
        "wandb_required": False,
        "build_host": platform.node(),
        "python_version": platform.python_version(),
        "git_commit": model_manifest["git_commit"],
        "model_manifest_sha256": _sha256_file(out / "models" / "model_manifest.json"),
        "build_executed": False,
    }
    (out / "docker_metadata.json").write_text(json.dumps(docker_metadata, indent=2, sort_keys=True) + "\n")
    print(f"Wrote models/model_manifest.json ({len(new_records)} digests) and docker_metadata.json", file=sys.stderr)


REPO_EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "wandb",
    "lightning_logs",
    "data",
    "build",
    "dist",
}
REPO_EXCLUDE_SUFFIXES = (
    ".ckpt",
    ".pt",
    ".nii",
    ".nii.gz",
    ".zip",
    ".sif",
    ".pyc",
)
FINETUNING_LOCAL_PREFIXES = (
    "Task_",
    "PPMR",
    "Zhang_Lingfeng_2022_PPMR_Dataset",
)

RELEASE_RUNTIME_PATHS = (
    "README.md",
    "pyproject.toml",
    "asparagus/__init__.py",
    "asparagus/functional",
    "asparagus/modules",
    "asparagus/pipeline/__init__.py",
    "asparagus/pipeline/auto_configuration",
    "finetuning/__init__.py",
    "finetuning/container/fomo_ensemble_predict.py",
    "finetuning/fomo26_inference/__init__.py",
    "finetuning/fomo26_inference/backbones.py",
    "finetuning/fomo26_inference/calibration.py",
    "finetuning/fomo26_inference/clsreg_ensemble.py",
    "finetuning/fomo26_inference/cross_patch.py",
    "finetuning/fomo26_inference/pretrained_embedding.py",
    "finetuning/fomo26_inference/runtime_geometry.py",
    "finetuning/fomo26_inference/seg_ensemble.py",
    "finetuning/fomo26_inference/time_budget.py",
    "finetuning/fomo26_inference/tta_safety.py",
    # tta_safety reads this registry at import time, and runtime_geometry reads the per-task
    # acquisition geometry out of the same file. Both resolve it next to themselves.
    # The import-closure audit only follows Python imports, so a data file the runtime opens has to
    # be named here or the packaged container raises when FOMO26_TTA=auto asks whether flip TTA is
    # admissible for the task, or when Task 3/4 resolves its canonicalization target.
    "finetuning/fomo26_inference/task_definitions.json",
)

PRODUCTION_ENTRYPOINTS = tuple(HERE / f"predict_task{task}.py" for task in ("1", "2", "3", "4", "5", "6_7"))
LOCAL_RUNTIME_PACKAGE_ROOTS = frozenset({"asparagus", "finetuning"})

_HOST_PATH_RE = re.compile(r"(?:/home/|/data/[a-z][a-z0-9_-]*/|/gpfs/|/lustre/|/scratch/)")
_SECRET_TEXT_RE = re.compile(r"(?:-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----|AKIA[0-9A-Z]{16}|gh[oprsu]_[A-Za-z0-9]{30,})")
_SECRET_NAMES = {".env", "id_rsa", "id_ed25519", "credentials", "credentials.json", "known_hosts"}


def _repo_ignore(dir_path, names):
    ignored = set()
    rel = Path(dir_path).resolve().relative_to(REPO)
    for name in names:
        if name in REPO_EXCLUDE_DIRS or name.startswith("tmp"):
            ignored.add(name)
            continue
        if rel.parts == ("finetuning",) and name.startswith(FINETUNING_LOCAL_PREFIXES):
            ignored.add(name)
            continue
        if any(name.endswith(suffix) for suffix in REPO_EXCLUDE_SUFFIXES):
            ignored.add(name)
    return ignored


def _copy_repo(repo_dst: Path):
    if repo_dst.is_symlink() or repo_dst.exists():
        if repo_dst.is_symlink() or repo_dst.is_file():
            repo_dst.unlink()
        else:
            shutil.rmtree(repo_dst)
    shutil.copytree(REPO, repo_dst, ignore=_repo_ignore)


def _write_immutable(path: Path, data: bytes) -> None:
    if path.exists():
        if path.is_file() and path.read_bytes() == data:
            return
        raise SystemExit(f"Refusing to overwrite conflicting staged file {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    if temporary.exists() and temporary.read_bytes() != data:
        raise SystemExit(f"Conflicting interrupted staging file requires inspection: {temporary}.")
    if not temporary.exists():
        temporary.write_bytes(data)
    os.replace(temporary, path)


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    if not source.is_file() or _sha256_file(source) != expected_sha256:
        raise SystemExit(f"Declared release artifact is missing or drifted: {source}.")
    if destination.exists():
        if destination.is_file() and _sha256_file(destination) == expected_sha256:
            return
        raise SystemExit(f"Refusing to overwrite conflicting staged artifact {destination}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    if not temporary.exists():
        shutil.copyfile(source, temporary)
    if _sha256_file(temporary) != expected_sha256:
        raise SystemExit(f"Artifact changed while staging {source}.")
    os.replace(temporary, destination)


def _tracked_python_modules() -> dict[str, Path]:
    """Map importable repository-local modules to their tracked source files."""
    command = ["git", "-C", str(REPO), "ls-files", "-z", "--", "*.py"]
    try:
        output = subprocess.run(command, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Cannot enumerate tracked Python modules: {exc}") from exc

    modules: dict[str, Path] = {}
    for raw_path in output.split(b"\0"):
        if not raw_path:
            continue
        path = Path(raw_path.decode())
        if not path.parts or path.parts[0] not in LOCAL_RUNTIME_PACKAGE_ROOTS:
            continue
        module = ".".join(path.parent.parts) if path.name == "__init__.py" else ".".join(path.with_suffix("").parts)
        modules[module] = path
    return modules


def _module_with_tracked_parents(module: str, modules: dict[str, Path]) -> set[str]:
    parts = module.split(".")
    return {candidate for index in range(1, len(parts) + 1) if (candidate := ".".join(parts[:index])) in modules}


def _local_imports(path: Path, module: str | None, modules: dict[str, Path]) -> set[str]:
    """Return tracked local modules imported anywhere in one production source file."""
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise SystemExit(f"Cannot inspect runtime imports in {path}: {exc}") from exc

    package = None
    if module:
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    imported: set[str] = set()
    for node in ast.walk(tree):
        candidates: list[str] = []
        if isinstance(node, ast.Import):
            candidates.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                if not package:
                    raise SystemExit(f"Relative import outside a package in runtime source {path}.")
                try:
                    base = importlib.util.resolve_name("." * node.level + base, package)
                except (ImportError, ValueError) as exc:
                    raise SystemExit(f"Cannot resolve relative import in runtime source {path}: {exc}") from exc
            if base:
                candidates.append(base)
            for alias in node.names:
                child = f"{base}.{alias.name}" if base else alias.name
                if child in modules:
                    candidates.append(child)
        for candidate in candidates:
            if candidate.split(".", 1)[0] in LOCAL_RUNTIME_PACKAGE_ROOTS:
                imported.update(_module_with_tracked_parents(candidate, modules))
    return imported


def _runtime_import_closure(entrypoint: Path) -> tuple[set[Path], set[str]]:
    """Resolve every statically reachable repository-local import from an entrypoint."""
    modules = _tracked_python_modules()
    queue = list(_local_imports(entrypoint, None, modules))
    visited: set[str] = set()
    while queue:
        module = queue.pop()
        if module in visited:
            continue
        visited.add(module)
        queue.extend(_local_imports(REPO / modules[module], module, modules) - visited)
    return {modules[module] for module in visited}, visited


def _tracked_runtime_files() -> list[Path]:
    command = ["git", "-C", str(REPO), "ls-files", "-z", "--", *RELEASE_RUNTIME_PATHS]
    try:
        output = subprocess.run(command, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Cannot enumerate the tracked runtime source set: {exc}") from exc
    paths = [Path(item.decode()) for item in output.split(b"\0") if item]
    missing = [
        path for path in RELEASE_RUNTIME_PATHS if not any(item == Path(path) or Path(path) in item.parents for item in paths)
    ]
    if missing:
        raise SystemExit(f"Required runtime paths are not tracked by packaging commit: {missing}.")
    runtime_paths = set(paths)
    for entrypoint in PRODUCTION_ENTRYPOINTS:
        dependencies, _ = _runtime_import_closure(entrypoint)
        runtime_paths.update(dependencies)
    return sorted(runtime_paths)


def _copy_runtime_source(destination: Path) -> None:
    """Copy the allowlisted, tracked runtime source set and no working-tree extras."""
    for relative in _tracked_runtime_files():
        source = REPO / relative
        target = destination / relative
        if source.is_symlink():
            raise SystemExit(f"Runtime source may not contain symlinks: {relative}.")
        _write_immutable(target, source.read_bytes())


def audit_staged_runtime_imports(context: Path, task: str) -> dict:
    """Fail before build when a staged entrypoint's local import closure is incomplete."""
    entrypoint = context / "predict.py"
    runtime_root = context / "asparagus_repo"
    if not entrypoint.is_file() or not runtime_root.is_dir():
        raise SystemExit(f"Staged runtime context is incomplete for {task}: {context}.")
    dependencies, modules = _runtime_import_closure(entrypoint)
    missing = sorted(path.as_posix() for path in dependencies if not (runtime_root / path).is_file())
    if missing:
        raise SystemExit(f"Staged runtime import closure is incomplete for {task}: missing {missing}.")
    return {
        "status": "PASS",
        "task": task,
        "local_modules_resolved": len(modules),
        "local_files_verified": len(dependencies),
        "missing": [],
    }


def _export_locked_requirements(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    if not temporary.exists():
        command = [
            "uv",
            "export",
            "--frozen",
            "--offline",
            "--no-default-groups",
            "--group",
            "dcai",
            "--no-emit-project",
            "--no-header",
            "--cache-dir",
            "/tmp/fomo26-container-uv-cache",
            "--format",
            "requirements.txt",
            "--output-file",
            str(temporary),
        ]
        proc = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise SystemExit(f"Locked dependency export failed: {proc.stderr}")
    _apply_release_runtime_overrides(temporary)
    if destination.exists():
        if destination.read_bytes() != temporary.read_bytes():
            raise SystemExit(f"Locked dependency export conflicts with existing {destination}.")
        temporary.unlink()
        return
    os.replace(temporary, destination)


#: Package name -> pinned version that the release runtime must contain, whatever the lock says.
#: The rationale for each pin lives beside its hashes in release_runtime_overrides.txt.
RELEASE_RUNTIME_OVERRIDES = {"gardening-tools": "0.3.2"}
RELEASE_RUNTIME_OVERRIDES_FILE = HERE / "release_runtime_overrides.txt"


def _override_blocks() -> dict[str, str]:
    """Parse release_runtime_overrides.txt into {package: full pinned requirement block}."""
    blocks, current, name = {}, [], None
    for line in RELEASE_RUNTIME_OVERRIDES_FILE.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and "==" in line:
            if name:
                blocks[name] = "\n".join(current)
            name = line.split("==", 1)[0].strip()
            current = [line]
        else:
            current.append(line)
    if name:
        blocks[name] = "\n".join(current)
    return blocks


def _apply_release_runtime_overrides(requirements: Path) -> None:
    """Rewrite the exported lock so the image provably installs the pinned release runtime.

    The override is applied to the requirements file itself rather than by installing a second
    time afterwards: a later `pip install` would depend on ordering and could be reordered or
    skipped, whereas a single hashed requirements file states one version and pip refuses anything
    else. The build fails loudly if an expected package is absent or the pin does not land.
    """
    blocks = _override_blocks()
    text = requirements.read_text()
    for package, version in RELEASE_RUNTIME_OVERRIDES.items():
        block = blocks.get(package)
        if block is None:
            raise SystemExit(f"No override block for {package!r} in {RELEASE_RUNTIME_OVERRIDES_FILE}.")
        # One exported entry spans its own line plus continuation/comment lines until the next
        # top-level requirement, so replace the whole span rather than just the version token.
        pattern = re.compile(
            rf"^{re.escape(package)}==[^\n]*\n(?:[ \t]+[^\n]*\n|[ \t]*#[^\n]*\n)*",
            re.MULTILINE,
        )
        text, count = pattern.subn(block + "\n", text)
        if count != 1:
            raise SystemExit(f"Expected exactly one {package} entry in the exported lock, found {count}.")
        if f"{package}=={version}" not in text:
            raise SystemExit(f"Release runtime override for {package} did not pin {version}.")
    requirements.write_text(text)


def _artifact_path(release_dir: Path, spec: dict) -> Path:
    path = Path(spec["path"])
    return path if path.is_absolute() else release_dir / path


def _env_line(name: str, value) -> str:
    return f"export {name}={shlex.quote(str(value))}\n"


#: What a geometry-declaring image does when an input's orientation is not the one the task was
#: fitted at. ``error`` refuses the case; ``skip`` falls back to native geometry, which is the
#: behaviour every pre-canonicalization release shipped. All 19 official validator fixtures are RAS,
#: but neither the official validator nor any organizer document asserts an orientation for the
#: hidden evaluation inputs, so an unexpected orientation must cost the canonicalization benefit for
#: that case rather than the case itself.
GEOMETRY_ORIENTATION_POLICY = "skip"


def _runtime_env(task: str, manifest: dict) -> dict[str, str | int | float]:
    if task == "task6_and_7":
        policy = manifest["tasks"][task]["policy"]
        return {
            "FOMO26_PRETRAINED_CHECKPOINT": "/app/models/pretrained/pretrained.ckpt",
            "FOMO26_ARCHITECTURE": manifest["architecture"],
            "FOMO26_SSL_OBJECTIVE": manifest["ssl_objective"],
            "FOMO26_CHECKPOINT_SOURCE": manifest["checkpoint_source"],
            "FOMO26_PATCH_SIZE": ",".join(str(value) for value in policy["patch_size"]),
            "FOMO26_MIN_ENCODER_COVERAGE": policy["minimum_encoder_coverage"],
        }
    policy = manifest["tasks"][task]["policy"]
    calibration = policy["calibration"]
    values: dict[str, str | int | float] = {
        "FOMO26_MANIFEST": "/app/models/manifest.json",
        "FOMO26_CHECKPOINT_NAME": "best",
        "FOMO26_TTA": policy["tta"],
        "FOMO26_MAX_MEMBERS": len(policy["ensemble_members"]),
        "FOMO26_TIME_TARGET_S": policy["time_target_seconds"],
        "FOMO26_CALIBRATION_REQUIRED": 1 if calibration["state"] == "required" else 0,
        "FOMO26_CALIBRATION_JSON": "/app/models/calibration.json" if calibration["state"] == "required" else "",
    }
    if task in {"task1", "task3", "task5"}:
        values["FOMO26_CROSS_PATCH"] = policy["cross_patch"]
    else:
        values["FOMO26_ENSEMBLE_SPACE"] = policy["ensemble_space"]
        values["FOMO26_WINDOW_POLICY"] = policy["window_policy"]
    # Only a task that declares a canonicalization target ever consults the orientation policy, so
    # the variable is frozen into exactly those images and left out of the ones where it is inert.
    if task_runtime_target_spacing(int(task.removeprefix("task"))) is not None:
        values["FOMO26_GEOMETRY_ORIENTATION_POLICY"] = GEOMETRY_ORIENTATION_POLICY
    return values


def audit_release_context(context: Path) -> dict:
    """Reject secrets, symlinks, and host-path dependencies before an image build."""
    findings = []
    files = 0
    for path in context.rglob("*"):
        relative = path.relative_to(context).as_posix()
        if path.is_symlink():
            findings.append(f"symlink:{relative}->{os.readlink(path)}")
            continue
        if not path.is_file():
            continue
        files += 1
        if path.name.lower() in _SECRET_NAMES:
            findings.append(f"secret_filename:{relative}")
        if path.suffix.lower() in {".ckpt", ".pt", ".npy", ".nii", ".gz", ".sif"}:
            continue
        if path.stat().st_size > 5 * 1024 * 1024:
            continue
        text = path.read_text(errors="ignore")
        if _HOST_PATH_RE.search(text):
            findings.append(f"host_path:{relative}")
        if _SECRET_TEXT_RE.search(text):
            findings.append(f"secret_text:{relative}")
    if findings:
        raise SystemExit(f"Release context audit failed: {findings}.")
    return {"status": "PASS", "files_scanned": files, "findings": []}


def stage_release_context(*, release_dir: Path, manifest: dict, task: str, out: Path) -> None:
    """Stage one final/smoke release task without legacy partial/link escape hatches."""
    if task not in {"task1", "task2", "task3", "task4", "task5", "task6_and_7"}:
        raise SystemExit(f"Unknown release task {task!r}.")
    if out.exists():
        raise SystemExit(f"Refusing to overwrite staged release context {out}.")
    partial = out.parent / f".{out.name}.partial"
    partial.mkdir(parents=True, exist_ok=True)
    marker = {"task": task, "release_manifest_sha256": manifest["manifest_sha256"]}
    _write_immutable(partial / ".release-source.json", (json.dumps(marker, sort_keys=True) + "\n").encode())
    legacy_task = "6_7" if task == "task6_and_7" else task.removeprefix("task")
    _write_immutable(partial / "predict.py", (HERE / f"predict_task{legacy_task}.py").read_bytes())
    _write_immutable(partial / "fomo_ensemble_predict.py", (HERE / "fomo_ensemble_predict.py").read_bytes())
    _write_immutable(partial / "Apptainer.def", (HERE / "Apptainer.def").read_bytes())
    _write_immutable(partial / "verify_runtime_contract.py", (HERE / "verify_runtime_contract.py").read_bytes())
    _copy_runtime_source(partial / "asparagus_repo")
    _export_locked_requirements(partial / "requirements-container.lock")

    runtime_env = _runtime_env(task, manifest)
    _write_immutable(
        partial / "models" / "env.sh",
        "".join(_env_line(name, value) for name, value in sorted(runtime_env.items())).encode(),
    )
    if task == "task6_and_7":
        pretrained = manifest["pretrained"]
        _copy_verified(
            _artifact_path(release_dir, pretrained),
            partial / "models" / "pretrained" / "pretrained.ckpt",
            pretrained["sha256"],
        )
        model_manifest = {
            "schema_version": MODEL_MANIFEST_SCHEMA,
            "task": task,
            "candidate": manifest["candidate_id"],
            "track": manifest["track"],
            "submission_image": manifest["container"]["images"][task],
            "architecture": manifest["architecture"],
            "scientific_architecture": manifest["scientific_architecture"],
            "ssl_objective": manifest["ssl_objective"],
            "checkpoint_source": manifest["checkpoint_source"],
            "frozen_pretrained": True,
            "pretrained_checkpoint_sha256": pretrained["sha256"],
            "downstream_finetuned_weights": [],
            "models": [{"path": "/app/models/pretrained/pretrained.ckpt", "sha256": pretrained["sha256"]}],
        }
    else:
        records = []
        model_records = []
        for fold_spec in sorted(manifest["tasks"][task]["folds"], key=lambda item: item["fold"]):
            fold = fold_spec["fold"]
            run_root = partial / "models" / "runs" / f"fold{fold}"
            _copy_verified(
                _artifact_path(release_dir, fold_spec["checkpoint"]),
                run_root / "checkpoints" / "best.ckpt",
                fold_spec["checkpoint"]["sha256"],
            )
            _copy_verified(
                _artifact_path(release_dir, fold_spec["hydra_config"]),
                run_root / "hydra" / "config.yaml",
                fold_spec["hydra_config"]["sha256"],
            )
            run_dir = f"/app/models/runs/fold{fold}"
            checkpoint = f"{run_dir}/checkpoints/best.ckpt"
            records.append(
                {"fold": fold, "returncode": 0, "run_dir": run_dir, "best_ckpt": checkpoint, "last_ckpt": checkpoint}
            )
            model_records.append(
                {
                    "fold": fold,
                    "path": checkpoint,
                    "sha256": fold_spec["checkpoint"]["sha256"],
                    "run_manifest_sha256": fold_spec["run_manifest"]["sha256"],
                    "hydra_config_sha256": fold_spec["hydra_config"]["sha256"],
                }
            )
        _write_immutable(partial / "models" / "manifest.json", _json_bytes(records))
        calibration = manifest["tasks"][task]["policy"]["calibration"]
        if calibration["state"] == "required":
            _copy_verified(
                _artifact_path(release_dir, calibration["artifact"]),
                partial / "models" / "calibration.json",
                calibration["artifact"]["sha256"],
            )
        internal_policy = json.loads(json.dumps(manifest["tasks"][task]["policy"]))
        if internal_policy["calibration"]["state"] == "required":
            internal_policy["calibration"]["artifact"]["path"] = "/app/models/calibration.json"
        model_manifest = {
            "schema_version": MODEL_MANIFEST_SCHEMA,
            "task": task,
            "candidate": manifest["candidate_id"],
            "track": manifest["track"],
            "submission_image": manifest["container"]["images"][task],
            "architecture": manifest["architecture"],
            "scientific_architecture": manifest["scientific_architecture"],
            "pretrained_checkpoint_sha256": manifest["pretrained"]["sha256"],
            "fold_count": len(records),
            "partial": manifest["release_mode"] != "final",
            "inference_policy": internal_policy,
            "downstream_finetuned_weights": [record["path"] for record in model_records],
            "models": model_records,
        }
    _write_immutable(partial / "models" / "model_manifest.json", _json_bytes(model_manifest))
    docker_metadata = {
        "schema_version": DOCKER_METADATA_SCHEMA,
        "entrypoints": [task],
        "network_required": False,
        "wandb_required": False,
        "packaging_git_commit": manifest["packaging_git_commit"],
        "release_manifest_sha256": manifest["manifest_sha256"],
        "model_manifest_sha256": _sha256_file(partial / "models" / "model_manifest.json"),
        "runtime_env": runtime_env,
    }
    _write_immutable(partial / "docker_metadata.json", _json_bytes(docker_metadata))
    context_audit = audit_release_context(partial)
    context_audit["runtime_import_closure"] = audit_staged_runtime_imports(partial, task)
    _write_immutable(partial / "context_audit.json", _json_bytes(context_audit))
    (partial / ".release-source.json").unlink()
    os.replace(partial, out)


def _stage_frozen_pretrained(out: Path, args) -> None:
    """Stage the Tasks 6/7 image: the PRETRAINED checkpoint and nothing finetuned.

    Tasks 6 and 7 are linear probing and fairness on frozen pretrained representations, so the only
    weights that may ship are the pretrained ones. Staging a finetuned fold here would submit a
    representation that has seen downstream labels.
    """
    if args.pretrained_checkpoint is None:
        raise SystemExit("--pretrained-checkpoint is required for task 6_7 (the frozen PRETRAINED checkpoint).")
    if not args.architecture:
        raise SystemExit("--architecture is required for task 6_7.")
    source = args.checkpoint_source or SSL_OBJECTIVE_DEFAULT_SOURCE.get(str(args.ssl_objective or "").lower())
    if not source:
        raise SystemExit("Provide --checkpoint-source, or an --ssl-objective whose contract supplies one.")

    src = args.pretrained_checkpoint
    if not src.is_file():
        raise SystemExit(f"Pretrained checkpoint not found: {src}")
    source_sha = _sha256_file(src)

    destination = out / "models" / "pretrained" / "pretrained.ckpt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, destination)
    staged_sha = _sha256_file(destination)
    if staged_sha != source_sha:
        raise SystemExit(f"Pretrained checkpoint changed during staging: {source_sha} -> {staged_sha}.")

    model_manifest = {
        "schema_version": MODEL_MANIFEST_SCHEMA,
        "task": "6_7",
        "candidate": args.candidate,
        "git_commit": _git_commit(),
        "frozen_pretrained": True,
        "architecture": args.architecture,
        "ssl_objective": args.ssl_objective,
        "checkpoint_source": source,
        "patch_size": [int(v) for v in args.patch_size],
        "pretrained_checkpoint_sha256": staged_sha,
        "pretrained_checkpoint_source_path": str(src),
        # Tasks 6/7 ensemble nothing and finetune nothing: one frozen encoder, one embedding.
        "fold_count": 0,
        "trained_fold_count": 0,
        "partial": False,
        "downstream_finetuned_weights": None,
        "models": [{"fold": None, "path": "/app/models/pretrained/pretrained.ckpt", "sha256": staged_sha}],
    }
    (out / "models" / "model_manifest.json").write_text(json.dumps(model_manifest, indent=2, sort_keys=True) + "\n")

    docker_metadata = {
        "schema_version": DOCKER_METADATA_SCHEMA,
        "entrypoints": ["6_7"],
        "network_required": False,
        "wandb_required": False,
        "build_host": platform.node(),
        "python_version": platform.python_version(),
        "git_commit": model_manifest["git_commit"],
        "model_manifest_sha256": _sha256_file(out / "models" / "model_manifest.json"),
        "build_executed": False,
        # The entrypoint reads these; recording them makes the image self-describing.
        "runtime_env": {
            "FOMO26_PRETRAINED_CHECKPOINT": "/app/models/pretrained/pretrained.ckpt",
            "FOMO26_ARCHITECTURE": args.architecture,
            "FOMO26_SSL_OBJECTIVE": args.ssl_objective or "",
            "FOMO26_CHECKPOINT_SOURCE": source,
            "FOMO26_PATCH_SIZE": ",".join(str(int(v)) for v in args.patch_size),
        },
    }
    (out / "docker_metadata.json").write_text(json.dumps(docker_metadata, indent=2, sort_keys=True) + "\n")
    # The image's %environment sources this, so the model identity travels with the container
    # instead of being retyped on the command line at evaluation time.
    (out / "models" / "env.sh").write_text(
        "".join(f'export {key}="{value}"\n' for key, value in sorted(docker_metadata["runtime_env"].items()))
    )

    repo_dst = out / "asparagus_repo"
    if args.link_repo:
        if not repo_dst.exists():
            repo_dst.symlink_to(REPO)
    else:
        _copy_repo(repo_dst)

    print(f"Staged Tasks 6/7 build context at {out} (frozen pretrained {args.architecture}, sha256 {staged_sha[:12]}).")
    sif = out.with_suffix(".sif")
    print(f"\nBuild:\n  apptainer build --fakeroot --arch amd64 {sif} {out / 'Apptainer.def'}")
    print("\nValidate:\n  python3 container_validator/validate.py --task task6_and_7 --sif " + str(sif))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["1", "2", "3", "4", "5", "6_7"])
    ap.add_argument("--manifest", type=Path, default=None, help="Orchestrator manifest (ensemble folds). Tasks 1-5 only.")
    ap.add_argument("--out", type=Path, required=True, help="Build-context directory to create.")
    ap.add_argument("--checkpoint-name", default="best")
    # Tasks 6/7 probe FROZEN pretrained representations, so they ship the pretrained checkpoint
    # itself -- the same artifact every other task transfers from -- and never a finetuned fold.
    ap.add_argument("--pretrained-checkpoint", type=Path, default=None, help="Tasks 6/7: the PRETRAINED SSL checkpoint.")
    ap.add_argument("--architecture", default=None, help="Tasks 6/7: resenc_b | unet_m | ...")
    ap.add_argument("--ssl-objective", default=None, help="Tasks 6/7: amaes (the retained public objective).")
    ap.add_argument("--checkpoint-source", default=None, choices=[None, "online", "ema", "ema_if_available"])
    ap.add_argument("--patch-size", nargs=3, type=int, default=[128, 128, 128], help="Tasks 6/7 inference patch size.")
    ap.add_argument("--candidate", default=None, help="Candidate identity recorded in the image metadata.")
    ap.add_argument("--allow-partial", action="store_true", help="Package an ensemble the fold manifest marks incomplete.")
    ap.add_argument(
        "--calibration-json", type=Path, help="Optional cls/reg calibration JSON to copy into /app/models/calibration.json."
    )
    ap.add_argument(
        "--link-repo", action="store_true", help="Symlink the repo instead of copying (smaller, local build only)."
    )
    args = ap.parse_args()

    out = args.out
    (out / "models" / "runs").mkdir(parents=True, exist_ok=True)
    _export_locked_requirements(out / "requirements-container.lock")

    # predict.py
    shutil.copy2(HERE / f"predict_task{args.task}.py", out / "predict.py")
    shutil.copy2(HERE / "fomo_ensemble_predict.py", out / "fomo_ensemble_predict.py")
    shutil.copy2(HERE / "Apptainer.def", out / "Apptainer.def")
    shutil.copy2(HERE / "verify_runtime_contract.py", out / "verify_runtime_contract.py")

    if args.task == "6_7":
        return _stage_frozen_pretrained(out, args)

    if args.manifest is None:
        raise SystemExit("--manifest is required for Tasks 1-5.")

    # checkpoints + rewritten manifest
    provenance = _read_provenance(args.manifest)
    if args.allow_partial:
        provenance["partial_acknowledged"] = True
    args._provenance = provenance
    if args.manifest.is_dir():
        records = [{"returncode": 0, "run_dir": str(args.manifest), "fold": 0}]
    else:
        records = json.loads(args.manifest.read_text())
    records = [r for r in records if r.get("returncode", 0) == 0 and r.get("run_dir")]
    new_records = []
    for i, r in enumerate(records):
        src = Path(r["run_dir"])
        dst = out / "models" / "runs" / f"fold{i}"
        (dst / "hydra").mkdir(parents=True, exist_ok=True)
        (dst / "checkpoints").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "hydra" / "config.yaml", dst / "hydra" / "config.yaml")
        shutil.copy2(
            src / "checkpoints" / f"{args.checkpoint_name}.ckpt", dst / "checkpoints" / f"{args.checkpoint_name}.ckpt"
        )
        ckpt_in_container = f"/app/models/runs/fold{i}/checkpoints/{args.checkpoint_name}.ckpt"
        # Hash what was actually copied, not the source: the digest must describe the bytes the
        # container will load. Without it the submission has no way to prove which weights shipped,
        # and a silently truncated copy is indistinguishable from a good one.
        copied_ckpt = dst / "checkpoints" / f"{args.checkpoint_name}.ckpt"
        checkpoint_sha256 = _sha256_file(copied_ckpt)
        source_sha256 = r.get("fold_checkpoint_sha256")
        if source_sha256 and source_sha256 != checkpoint_sha256:
            raise SystemExit(
                f"Fold {i} checkpoint changed during staging: the fold manifest recorded "
                f"{source_sha256}, the copy at {copied_ckpt} hashes to {checkpoint_sha256}. "
                "Refusing to package weights that do not match their provenance record."
            )
        new_records.append(
            {
                **r,
                "run_dir": f"/app/models/runs/fold{i}",
                "best_ckpt": ckpt_in_container,
                "last_ckpt": ckpt_in_container,
                "checkpoint_sha256": checkpoint_sha256,
                "source_run_dir": str(src),
            }
        )
    _assert_manifest_matches_staged(provenance, len(new_records))
    (out / "models" / "manifest.json").write_text(json.dumps(new_records, indent=2))
    _write_model_manifest(out, args, new_records, records)
    if args.calibration_json is not None:
        shutil.copy2(args.calibration_json, out / "models" / "calibration.json")
    legacy_runtime_env = {
        "FOMO26_MANIFEST": "/app/models/manifest.json",
        "FOMO26_CHECKPOINT_NAME": args.checkpoint_name,
        "FOMO26_TTA": "auto",
        "FOMO26_MAX_MEMBERS": 0,
        "FOMO26_TIME_TARGET_S": 115,
        "FOMO26_CALIBRATION_REQUIRED": 0,
        "FOMO26_CALIBRATION_JSON": "/app/models/calibration.json",
        "FOMO26_CROSS_PATCH": "none",
        "FOMO26_ENSEMBLE_SPACE": "prob",
    }
    (out / "models" / "env.sh").write_text(
        "".join(_env_line(name, value) for name, value in sorted(legacy_runtime_env.items()))
    )

    # repo
    repo_dst = out / "asparagus_repo"
    if args.link_repo:
        if not repo_dst.exists():
            repo_dst.symlink_to(REPO)
    else:
        _copy_repo(repo_dst)

    print(f"Staged build context at {out} ({len(new_records)} fold checkpoints).")
    sif = out.with_suffix(".sif")
    print(f"\nBuild:\n  apptainer build --fakeroot --arch amd64 {sif} {out / 'Apptainer.def'}")
    print(
        f"\nValidate (see https://github.com/fomo26/container-validator):\n\
            use fake_data/fomo26/fomo-task{args.task}-val/ as --data-dir"
    )


if __name__ == "__main__":
    main()
