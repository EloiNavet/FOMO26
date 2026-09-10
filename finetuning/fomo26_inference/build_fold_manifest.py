#!/usr/bin/env python3
"""Bridge the Jean-Zay Slurm rail to the inference/container rail.

The two rails record a fold campaign in incompatible shapes:

* the Slurm rail writes one ``run_manifest.json`` *object* per run directory
  (``fomo26-run-manifest-v1``: task, fold, ``pretrained_checkpoint_sha256``, git state, Slurm ids);
* the inference and container rails consume a *list* of
  ``{run_dir, best_ckpt, returncode, fold}`` -- see
  ``finetuning/container/fomo_ensemble_predict.resolve_manifest_records`` and
  ``finetuning/container/build_container.py``.

Until now the only producer of that list was ``finetuning/run_finetune_folds.py``, the workstation
orchestrator, which records no checkpoint digest and validates no provenance. So the only path to a
submittable container bypassed every provenance guarantee the Slurm rail provides. This module is
that missing producer, and it adds the checks a submission depends on:

* every fold must name the **same** pretrained checkpoint SHA256 -- a fold accidentally fine-tuned
  from a different backbone silently changes what the ensemble is;
* every fold checkpoint is hashed, so the container manifest can carry real digests;
* a fold whose run failed, or whose checkpoint is missing, is reported rather than skipped.

Read-only with respect to run directories: it writes exactly one output file.

Usage::

    python -m finetuning.fomo26_inference.build_fold_manifest \\
        --models-root "$ASPARAGUS_MODELS" --task 2 --run-id safety_unet_32k \\
        --output "$ASPARAGUS_RESULTS/manifests/task2_fold_manifest.json"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

SCHEMA_VERSION = "fomo26-fold-manifest-v1"

#: ``submit_downstream.sh`` run-directory convention:
#: ``${ASPARAGUS_MODELS}/${task_name}/run_${run_id}__task${N}__fold${K}``.
RUN_DIR_RE = re.compile(r"^run_(?P<run_id>.+)__task(?P<task>\d+)__fold(?P<fold>\w+)$")


class FoldManifestError(RuntimeError):
    """A fold campaign that must not be packaged as-is."""


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_run_dirs(models_root: Path, task: int, run_id: str | None = None) -> list[Path]:
    """Return every ``run_*__task<N>__fold<K>`` directory for one task, sorted by fold."""
    found: list[tuple[str, Path]] = []
    for candidate in sorted(models_root.glob("*/run_*__task*__fold*")):
        if not candidate.is_dir():
            continue
        match = RUN_DIR_RE.match(candidate.name)
        if not match or int(match.group("task")) != int(task):
            continue
        if run_id is not None and match.group("run_id") != run_id:
            continue
        found.append((match.group("fold"), candidate))
    return [path for _fold, path in sorted(found, key=lambda item: item[0])]


def _read_run_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FoldManifestError(
            f"{run_dir} has no run_manifest.json. The Slurm rail writes one per run; its absence "
            "means this directory was not produced by jz_downstream.slurm, so its provenance "
            "cannot be established and it must not enter a submission."
        )
    return json.loads(manifest_path.read_text())


def _resolve_checkpoint(run_dir: Path, record: dict, checkpoint_name: str) -> Path:
    # ``best_checkpoint`` names the *best* checkpoint specifically. Honouring it for any requested
    # name silently returns best.ckpt when the caller asked for last.ckpt -- which matters because
    # best.ckpt was selected by monitoring each fold's own validation set, so it is exactly the
    # wrong checkpoint to score that same validation set with.
    declared = record.get("best_checkpoint") if checkpoint_name == "best" else None
    if declared:
        path = Path(declared)
        if not path.is_absolute():
            path = run_dir / path
        if path.is_file():
            return path
    fallback = run_dir / "checkpoints" / f"{checkpoint_name}.ckpt"
    if fallback.is_file():
        return fallback
    raise FoldManifestError(
        f"{run_dir} declares best_checkpoint={declared!r} and has no "
        f"checkpoints/{checkpoint_name}.ckpt; there is no model to ensemble for this fold."
    )


def build_fold_manifest(
    run_dirs,
    *,
    checkpoint_name: str = "best",
    expected_folds: int | None = None,
    require_uniform_backbone: bool = True,
    candidate: str | None = None,
    inference_policy: str | None = None,
) -> dict:
    """Collect fold records, refusing a campaign that cannot be packaged honestly."""
    run_dirs = [Path(item) for item in run_dirs]
    if not run_dirs:
        raise FoldManifestError("No fold run directories were found; nothing to package.")

    records: list[dict] = []
    backbones: dict[str, list[str]] = {}
    problems: list[str] = []

    for run_dir in run_dirs:
        record = _read_run_manifest(run_dir)
        fold = record.get("fold")
        status = str(record.get("status", "")).lower()
        if status and status not in {"completed", "success", "ok"}:
            problems.append(f"fold {fold} in {run_dir.name} has status={status!r}")
            continue
        checkpoint = _resolve_checkpoint(run_dir, record, checkpoint_name)
        backbone_sha = record.get("pretrained_checkpoint_sha256")
        if not isinstance(backbone_sha, str) or re.fullmatch(r"[0-9a-f]{64}", backbone_sha) is None:
            raise FoldManifestError(f"fold {fold} in {run_dir.name} has no valid pretrained checkpoint SHA-256")
        backbones.setdefault(str(backbone_sha), []).append(str(fold))
        records.append(
            {
                # --- the shape resolve_manifest_records()/build_container.py consume ---
                "run_dir": str(run_dir.resolve()),
                "best_ckpt": str(checkpoint.resolve()),
                "last_ckpt": str((run_dir / "checkpoints" / "last.ckpt").resolve()),
                "returncode": 0,
                "fold": fold,
                # --- provenance the workstation producer never recorded ---
                "backbone": record.get("pretrained_checkpoint"),
                "pretrained_checkpoint_sha256": backbone_sha,
                "pretrained_checkpoint_step": record.get("pretrained_checkpoint_step"),
                "architecture": record.get("architecture"),
                "ssl_objective": record.get("ssl_objective"),
                "transfer_source": record.get("transfer_source"),
                "transfer_scope": record.get("transfer_scope"),
                "source_git_commit": record.get("source_git_commit"),
                "launch_git_commit": record.get("launch_git_commit"),
                "split": record.get("split"),
                "fold_checkpoint_sha256": sha256_file(checkpoint),
                "task": record.get("task"),
                "fomo_task": record.get("fomo_task"),
                "run_id": record.get("run_id"),
                "evaluation_id": record.get("evaluation_id"),
                "seed": record.get("seed"),
                "git_commit": record.get("git_commit"),
                "slurm_job_id": record.get("slurm_job_id"),
            }
        )

    if problems:
        raise FoldManifestError(
            "Refusing to build a fold manifest from an incomplete campaign: "
            + "; ".join(problems)
            + ". Re-run the failed folds, or pass --allow-partial to package the survivors "
            "explicitly (which changes what the ensemble is)."
        )
    if require_uniform_backbone and len(backbones) > 1:
        detail = "; ".join(f"{sha}: folds {sorted(folds)}" for sha, folds in sorted(backbones.items()))
        raise FoldManifestError(
            "Folds were fine-tuned from different pretrained checkpoints, so ensembling them would "
            f"average models with different provenance: {detail}."
        )
    observed_folds = [str(item["fold"]) for item in records]
    # Two run directories claiming the same fold would pass a bare count check while silently
    # ensembling one fold twice and leaving another unrepresented.
    if len(observed_folds) != len(set(observed_folds)):
        raise FoldManifestError(f"Fold identities are duplicated: {sorted(observed_folds)}.")
    if expected_folds is not None and len(records) != int(expected_folds):
        raise FoldManifestError(
            f"Expected {expected_folds} folds, collected {len(records)} ({sorted(str(item['fold']) for item in records)})."
        )
    if expected_folds is not None:
        expected_fold_ids = {str(index) for index in range(int(expected_folds))}
        if set(observed_folds) != expected_fold_ids:
            raise FoldManifestError(
                f"Expected {expected_folds} folds {sorted(expected_fold_ids)}, collected "
                f"{len(records)} ({sorted(observed_folds)})."
            )
    present_folds = sorted(str(item["fold"]) for item in records)
    tasks = sorted({str(item["fomo_task"] or item["task"]) for item in records if item.get("fomo_task") or item.get("task")})
    # The manifest is the authority on how complete this campaign is. The packaging step must be
    # able to read that state instead of assuming a fold count, so it is always stated -- including
    # on the success path, where "partial: false" is the claim that makes the container honest.
    return {
        "schema_version": SCHEMA_VERSION,
        "records": records,
        "candidate": candidate,
        "task": tasks[0] if len(tasks) == 1 else None,
        "pretrained_checkpoint_sha256": next(iter(backbones)) if backbones else None,
        "pretrained_checkpoint": records[0].get("backbone") if records else None,
        "expected_folds": int(expected_folds) if expected_folds is not None else None,
        "present_folds": present_folds,
        "trained_fold_count": len(records),
        "fold_count": len(records),
        "partial": False if expected_folds is None else len(records) != int(expected_folds),
        "partial_reason": None,
        "inference_policy": inference_policy,
        "git_commit": records[0].get("git_commit") if records else None,
    }


def write_fold_manifest(manifest: dict, output: Path) -> Path:
    """Write the list-shaped manifest the inference rail reads, plus a sidecar with provenance.

    ``resolve_manifest_records`` expects a bare JSON list, so that is what the primary file holds;
    the richer envelope (digests, git state, backbone identity) goes next to it so the reports and
    the container model manifest can consume it without changing the historical contract.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest["records"], indent=2, sort_keys=True) + "\n")
    sidecar = output.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return sidecar


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--models-root", type=Path, help="ASPARAGUS_MODELS root to scan for fold run dirs.")
    source.add_argument("--run-dir", type=Path, nargs="+", help="Explicit fold run directories.")
    parser.add_argument("--task", type=int, help="Challenge task number (required with --models-root).")
    parser.add_argument("--run-id", help="Restrict discovery to one run id.")
    parser.add_argument("--checkpoint-name", default="best", help="Checkpoint stem inside checkpoints/.")
    parser.add_argument("--expected-folds", type=int, default=None, help="Fail unless exactly this many folds.")
    parser.add_argument("--candidate", default=None, help="Candidate identity this campaign belongs to.")
    parser.add_argument("--inference-policy", default=None, help="Frozen inference recipe this manifest is packaged under.")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Package only the successful folds. This changes what the ensemble is; it is recorded.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Where to write fold_manifest.json.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.models_root is not None:
        if args.task is None:
            raise SystemExit("--task is required with --models-root.")
        run_dirs = discover_run_dirs(args.models_root, args.task, args.run_id)
    else:
        run_dirs = list(args.run_dir)

    try:
        manifest = build_fold_manifest(
            run_dirs,
            checkpoint_name=args.checkpoint_name,
            expected_folds=args.expected_folds,
            candidate=args.candidate,
            inference_policy=args.inference_policy,
        )
    except FoldManifestError as error:
        if not args.allow_partial:
            print(f"FAIL: {error}", file=sys.stderr)
            return 2
        manifest = build_fold_manifest(
            [d for d in run_dirs if (d / "run_manifest.json").is_file()],
            checkpoint_name=args.checkpoint_name,
            candidate=args.candidate,
            inference_policy=args.inference_policy,
        )
        manifest["expected_folds"] = int(args.expected_folds) if args.expected_folds is not None else None
        manifest["partial"] = True
        manifest["partial_reason"] = str(error)

    sidecar = write_fold_manifest(manifest, args.output)
    print(f"Wrote {args.output} ({manifest['fold_count']} folds) and {sidecar}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
