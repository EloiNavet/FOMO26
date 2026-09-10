"""Orchestrate K-fold (and multi-backbone) finetuning for FOMO26.

For every (backbone, fold) pair it launches the matching ``asp_finetune_{seg,cls,reg}`` with
``data.train_split=<split> data.fold=<k>`` and ``FOMO26_BACKBONE_CHECKPOINT=<backbone>``, then
records the run directory and the resulting ``checkpoints/best.ckpt`` into a JSON *manifest*.
That manifest is the contract consumed by the inference engine (ensemble of fold checkpoints).

The challenge rule is "one pretrained checkpoint for all 7 tasks": pass a single ``--backbones``
for the real submission. Multiple backbones are supported only for screening / experiments.

Example (MVP, Task 2 seg, 5 folds, single backbone, both GPUs):
    python -m finetuning.run_finetune_folds \
        --task task2_lesion_ft --backbones amaes \
        --split split_kfold5_holdout70_15_15 --folds 0 1 2 3 4 \
        --gpus 0 1 --manifest $ASPARAGUS_RESULTS/manifests/task2_amaes.json \
        --extra "training.epochs=300 training.warmup_epochs=20"

Run with ``--dry-run`` first to print the planned commands.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# No default: this pointed at one machine's absolute path, which is a private path in a public
# tree and a silent wrong answer anywhere else. FOMO26_CKPT_DIR must be set explicitly.
CKPT_DIR = Path(os.environ["FOMO26_CKPT_DIR"]) if os.environ.get("FOMO26_CKPT_DIR") else None


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")[:120]


def is_done(record: dict) -> bool:
    """A run is 'done' (resume-skippable) only if it succeeded AND its best.ckpt still exists.

    A recorded best.ckpt path that has since disappeared is NOT done and must be retrained.
    """
    return bool(record.get("returncode") == 0 and record.get("best_ckpt") and Path(record["best_ckpt"]).exists())


# Map finetune config name -> (asp entry point, task kind). Extend as configs are added.
KIND_BY_PREFIX = {
    "task1_lesion": ("asp_finetune_seg", "seg"),
    "task2_lesion": ("asp_finetune_seg", "seg"),
    "task4_multiclass": ("asp_finetune_seg", "seg"),
    "task1_presence": ("asp_finetune_cls", "cls"),
    "task5_ppmr": ("asp_finetune_cls", "cls"),
    "task3_age": ("asp_finetune_reg", "reg"),
}

_RUN_DIR_RE = re.compile(r"Run dir:\s*(\S+)")
_manifest_lock = threading.Lock()


def infer_entrypoint(task: str, override: str | None) -> tuple[str, str]:
    if override:
        return {"seg": "asp_finetune_seg", "cls": "asp_finetune_cls", "reg": "asp_finetune_reg"}[override], override
    for prefix, val in KIND_BY_PREFIX.items():
        if task.startswith(prefix):
            return val
    raise SystemExit(f"Cannot infer kind for task '{task}'. Pass --kind seg|cls|reg.")


def backbone_path(name: str) -> Path:
    p = Path(name)
    if p.suffix == ".ckpt" and p.exists():
        return p.resolve()
    if CKPT_DIR is None:
        raise SystemExit(f"Backbone '{name}' is not a .ckpt path and FOMO26_CKPT_DIR is not set.")
    cand = CKPT_DIR / f"{name}.ckpt"
    if not cand.exists():
        raise SystemExit(f"Backbone '{name}' not found as a path or as {cand}.")
    return cand.resolve()


def resolve_entrypoint_command(entry: str) -> str:
    found = shutil.which(entry)
    if found:
        return found
    for python_path in (Path(sys.executable), Path(sys.executable).resolve()):
        venv_entry = python_path.parent / entry
        if venv_entry.is_file():
            return str(venv_entry)
    raise SystemExit(f"Entrypoint '{entry}' not found in PATH or next to {sys.executable}.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--task", required=True, help="Finetune config name under projects/fomo26/finetune/ (e.g. task2_lesion_ft)."
    )
    p.add_argument("--kind", choices=["seg", "cls", "reg"], default=None, help="Override the inferred task kind.")
    p.add_argument(
        "--backbones", nargs="+", required=True, help="Backbone names (resolved under FOMO26_CKPT_DIR) or .ckpt paths."
    )
    p.add_argument("--split", required=True, help="K-fold split name (without .json), e.g. split_kfold5_holdout70_15_15.")
    p.add_argument("--folds", nargs="+", type=int, required=True, help="Fold indices to train.")
    p.add_argument(
        "--gpus", nargs="+", type=int, default=[0], help="GPU ids to use; runs are pinned round-robin and run in parallel."
    )
    p.add_argument("--manifest", type=Path, required=True, help="Output manifest JSON (records run dirs + best.ckpt paths).")
    p.add_argument("--extra", default="", help="Extra hydra overrides appended to every run (single string).")
    p.add_argument("--config-group", default="projects/fomo26/finetune", help="Hydra config group holding the task config.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    entry, kind = infer_entrypoint(args.task, args.kind)
    entry_cmd = resolve_entrypoint_command(entry)
    config_name = f"{args.config_group}/{args.task}"
    backbones = {name: backbone_path(name) for name in args.backbones}

    jobs = [(bname, bpath, fold) for bname, bpath in backbones.items() for fold in args.folds]
    print(f"Planned {len(jobs)} runs: task={args.task} kind={kind} entry={entry}")
    print(f"  backbones={list(backbones)} folds={args.folds} gpus={args.gpus} split={args.split}")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text())
    # Resumable: skip (backbone, fold) pairs already trained successfully AND whose best.ckpt still
    # exists on disk (a recorded path that has since disappeared must be retrained, not skipped).
    done = {(r["backbone"], r["fold"]) for r in manifest if is_done(r)}
    logs_dir = args.manifest.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    git_commit = _git_commit()

    gpu_q: queue.Queue[int] = queue.Queue()
    for g in args.gpus:
        gpu_q.put(g)

    def run_one(job):
        bname, bpath, fold = job
        if (bname, fold) in done:
            print(f"[{bname} fold={fold}] SKIP (already in manifest)")
            return
        gpu = gpu_q.get()
        try:
            env = dict(os.environ)
            env["FOMO26_BACKBONE_CHECKPOINT"] = str(bpath)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env.setdefault("HYDRA_FULL_ERROR", "1")
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            cmd = [
                entry_cmd,
                "--config-name",
                config_name,
                f"data.train_split={args.split}",
                f"data.fold={fold}",
            ]
            if args.extra:
                cmd += args.extra.split()
            label = f"[{bname} fold={fold} gpu={gpu}]"
            if args.dry_run:
                print(label, "DRY:", "FOMO26_BACKBONE_CHECKPOINT=%s" % bpath, "CUDA_VISIBLE_DEVICES=%s" % gpu, " ".join(cmd))
                return
            print(label, "START", " ".join(cmd))
            t0 = time.time()
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            elapsed = round(time.time() - t0, 1)
            run_dir = None
            m = _RUN_DIR_RE.search(proc.stdout) or _RUN_DIR_RE.search(proc.stderr)
            if m:
                run_dir = m.group(1)
            # Persist stdout/stderr so a run is auditable without re-running it.
            stub = f"{_sanitize(args.task)}__{_sanitize(bname)}__fold{fold}"
            out_log = logs_dir / f"{stub}.out"
            err_log = logs_dir / f"{stub}.err"
            out_log.write_text(proc.stdout or "")
            err_log.write_text(proc.stderr or "")
            best_ckpt = str(Path(run_dir) / "checkpoints" / "best.ckpt") if run_dir else None
            best_exists = bool(best_ckpt) and Path(best_ckpt).exists()
            if proc.returncode != 0:
                status = "failed"
            elif not best_exists:
                status = "no_best_ckpt"
            else:
                status = "ok"
            entry_rec = {
                "task": args.task,
                "kind": kind,
                "backbone": bname,
                "backbone_path": str(bpath),
                "split": args.split,
                "fold": fold,
                "gpu": gpu,
                "returncode": proc.returncode,
                "status": status,
                "run_dir": run_dir,
                "best_ckpt": best_ckpt,
                "last_ckpt": str(Path(run_dir) / "checkpoints" / "last.ckpt") if run_dir else None,
                "command": cmd,
                "extra": args.extra,
                "git_commit": git_commit,
                "elapsed_sec": elapsed,
                "stdout_log": str(out_log),
                "stderr_log": str(err_log),
            }
            with _manifest_lock:
                manifest.append(entry_rec)
                args.manifest.write_text(json.dumps(manifest, indent=2))
            print(label, status.upper(), f"({elapsed}s)", "run_dir=", run_dir)
            if proc.returncode != 0:
                tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-15:])
                print(label, "stderr tail:\n", tail)
        finally:
            gpu_q.put(gpu)

    if args.dry_run:
        for job in jobs:
            run_one(job)
        return

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as ex:
        list(ex.map(run_one, jobs))

    ok = sum(1 for r in manifest if r.get("returncode") == 0)
    print(f"\nDone. {ok}/{len(jobs)} new/total runs ok. Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
