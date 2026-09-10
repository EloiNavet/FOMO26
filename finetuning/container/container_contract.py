"""Lightweight local contract check for the existing FOMO26 Apptainer path.

It refuses a container definition that hardcodes a user or cluster path, that looks like it
carries a secret, or whose runscript reaches the network. It lived under
`experiments/fomo26_final/` while the campaign tree existed; the campaign tree is not part of
the public code release and this gate is, so it moved to the container rail it guards.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import torch
from finetuning.container.validator_bridge import official_contract
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DEFINITION = REPO / "finetuning/container/Apptainer.def"

#: Mechanically derived from the pinned official validator. Keep the historical internal spellings
#: used by this compatibility checker while eliminating a second handwritten task inventory.
_OFFICIAL_TO_LEGACY = {task: ("6_7" if task == "task6_and_7" else task.removeprefix("task")) for task in official_contract()}
KNOWN_TASKS = set(_OFFICIAL_TO_LEGACY.values())
#: What a submission must cover unless the caller narrows it -- preserves the historical default.
DEFAULT_REQUIRED_TASKS = ("1", "2", "3", "4", "5")


class ContainerContractInvalid(ValueError):
    """Raised when the local inference image contract is unsafe."""


def validate_container_contract(definition: Path, model_metadata: Path, require_tasks=DEFAULT_REQUIRED_TASKS) -> dict:
    text = definition.read_text()
    forbidden_paths = ("/home/", "/lustre/", "/gpfs/", "/scratch/")
    if any(path in text for path in forbidden_paths):
        raise ContainerContractInvalid("container definition contains a user/cluster-specific absolute path")
    if re.search(r"(?i)(api[_-]?key|access[_-]?token|secret)\s*=", text):
        raise ContainerContractInvalid("container definition appears to contain a secret")
    runscript = text.split("%runscript", 1)[-1]
    if re.search(r"\b(curl|wget|git|pip|wandb)\b", runscript):
        raise ContainerContractInvalid("container runtime requires network or W&B")
    metadata = json.loads(model_metadata.read_text())
    required = {"version", "source_git_commit", "model_files", "task_entrypoints"}
    missing = sorted(required - set(metadata))
    if missing:
        raise ContainerContractInvalid(f"model metadata is missing {missing}")
    if not metadata["model_files"]:
        raise ContainerContractInvalid("no model files are declared for image inclusion")
    for path in metadata["model_files"]:
        if not Path(path).is_file():
            raise ContainerContractInvalid(f"declared model file is missing: {path}")
    declared = set(metadata["task_entrypoints"])
    unknown = sorted(declared - KNOWN_TASKS)
    if unknown:
        raise ContainerContractInvalid(f"model metadata declares unknown task entrypoints {unknown}")
    missing_tasks = sorted(set(require_tasks) - declared)
    if missing_tasks:
        raise ContainerContractInvalid(f"model metadata must declare entrypoints for tasks {missing_tasks}")
    cli_results = {}
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    for task, entry in sorted(metadata["task_entrypoints"].items()):
        path = REPO / entry
        completed = subprocess.run(
            [sys.executable, str(path), "--help"],
            cwd=REPO,
            text=True,
            capture_output=True,
            env=environment,
            timeout=30,
        )
        if completed.returncode != 0:
            raise ContainerContractInvalid(f"Task {task} CPU startup/import failed: {completed.stderr[-500:]}")
        cli_results[task] = {"entrypoint": entry, "cpu_startup": "PASS"}
    return {
        "schema_version": "fomo26-container-contract-v1",
        "status": "PASS",
        "definition": str(definition),
        "version": metadata["version"],
        "source_git_commit": metadata["source_git_commit"],
        "model_file_count": len(metadata["model_files"]),
        "tasks": cli_results,
        "offline_runtime": True,
        "wandb_required": False,
        "network_required": False,
        "cpu_startup": "PASS",
        "gpu_startup": "LOCALLY_AVAILABLE" if torch.cuda.is_available() else "BLOCKED_NO_LOCAL_GPU",
        "memory_estimation": metadata.get("memory_estimation"),
        "timeout_seconds": metadata.get("timeout_seconds", 120),
        "deterministic": bool(metadata.get("deterministic", False)),
        "build_command": [
            "apptainer",
            "build",
            "--fakeroot",
            "<TASK_IMAGE>.sif",
            str(definition),
            "--arch",
            "amd64",
        ],
        "build_executed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definition", type=Path, default=DEFAULT_DEFINITION)
    parser.add_argument("--model-metadata", type=Path, required=True)
    parser.add_argument(
        "--require-tasks",
        default=",".join(DEFAULT_REQUIRED_TASKS),
        help="Comma-separated task entrypoints this metadata must declare (e.g. '6_7' for the Tasks 6/7 image).",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        required = tuple(t.strip() for t in str(args.require_tasks).split(",") if t.strip())
        report = validate_container_contract(args.definition, args.model_metadata, require_tasks=required)
    except (ContainerContractInvalid, OSError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        raise SystemExit(f"Container contract REFUSED: {exc}") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Container contract PASS -> {args.output}; image build not executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
