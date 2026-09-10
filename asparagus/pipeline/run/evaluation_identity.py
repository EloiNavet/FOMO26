"""The one machine-readable statement of what a downstream run actually evaluated.

Before this, the evaluated checkpoint had to be *re-inferred* after the fact, and every
consumer inferred it differently: the HPC downstream worker read the role back out of the
prediction filename and hashed ``checkpoints/best.ckpt`` whenever that file existed, one
collector took ``paths[0]`` of an unsorted glob, and ``eval_box.py`` walked ``os.listdir``.
Each of those is a guess, and each was wrong for at least one real configuration -- most
obviously a run that evaluates ``current`` weights, which would still have been stamped with
the digest of ``best.ckpt``.

The fine-tuning entrypoint now writes ``evaluation_identity.json`` next to the run, from
the same resolved object that chooses the weights handed to ``trainer.test`` and names the
prediction file. Everything downstream reads that sidecar instead of inspecting the
filesystem: the metrics record, the run manifest and the collectors all quote one identity
rather than each reconstructing their own.

Checkpoint-role semantics, in full
----------------------------------
``best``
    The exact file recorded by the monitored :class:`~lightning.pytorch.callbacks.ModelCheckpoint`
    (``best_model_path``). Evaluation fails if no such file was written; there is no fallback
    to the in-memory weights, because that would publish end-of-fit results labelled ``best``.
    ``checkpoint_path`` and ``checkpoint_sha256`` describe that file.

``last``
    The latest *periodic* checkpoint. Lightning does not offer one meaning of "last": a
    callback constructed with ``save_last=True`` records ``last_model_path``, while a
    callback with no ``monitor`` records the most recent periodic write in
    ``best_model_path`` (``_save_none_monitor_checkpoint`` assigns it there, whatever the
    attribute is called). ``resolve_checkpoint_path`` handles both explicitly and refuses a
    monitored callback, where ``best_model_path`` would mean "best" rather than "last".
    ``checkpoint_path`` and ``checkpoint_sha256`` describe that file.

``current``
    The in-memory end-of-fit weights. Lightning is passed ``ckpt_path=None`` and loads
    nothing. No checkpoint file is evaluated, so ``checkpoint_path`` and
    ``checkpoint_sha256`` are ``null`` -- deliberately, and never back-filled from
    ``best.ckpt`` or any other file that happens to be on disk. ``evaluated_state`` says
    ``in_memory_final``, and ``epoch``/``global_step`` are what identifies those weights.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

# Asparagus task names embed the FOMO26 task number (SEG902_FOMO26_Task2_lesion -> 2). Parsed
# here rather than imported from `finetuning`, which the pipeline package must not depend on.
_TASK_RE = re.compile(r"Task(\d+)", re.IGNORECASE)

SCHEMA_VERSION = "fomo26-evaluation-identity-v1"
FILENAME = "evaluation_identity.json"

# A file-backed role is one whose evaluated weights exist as a file that can be digested.
FILE_BACKED_ROLES = ("best", "last")
IN_MEMORY_ROLES = ("current",)

STATE_CHECKPOINT_FILE = "checkpoint_file"
STATE_IN_MEMORY_FINAL = "in_memory_final"

IDENTITY_FIELDS = (
    "schema_version",
    "status",
    "campaign_id",
    "evaluation_id",
    "evaluation_unit_id",
    "task",
    "task_name",
    "fold",
    "split",
    "checkpoint_role",
    "evaluated_state",
    "checkpoint_path",
    "checkpoint_sha256",
    "epoch",
    "global_step",
    "prediction_path",
    "git_sha",
    "launch_git_sha",
    "run_dir",
)


def sha256_file(path) -> str | None:
    """Digest a file, or return None when it is absent. Never invents a digest."""
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def executing_git_sha() -> str | None:
    """The commit of the tree this module is being imported from, or None.

    Resolved from the package's own location rather than from the working directory, so a job
    that runs with some other cwd still reports the code that actually executed. Returns None
    (never a guess) when the tree is not a Git checkout; a dirty tree is marked, because a
    commit id alone would overstate what was run.
    """
    import subprocess

    repo = Path(__file__).resolve().parent
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        if head.returncode != 0:
            return None
        sha = head.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        return f"{sha}-dirty" if dirty.returncode == 0 and dirty.stdout.strip() else sha
    except (OSError, subprocess.SubprocessError):
        return None


def clean_execution_commit(root=None) -> str | None:
    """Exact 40-hex HEAD of a checkout that has nothing uncommitted, or None.

    :func:`executing_git_sha` answers "what was run" for the record, and marks a dirty tree by
    suffixing ``-dirty``. That marker names the fact of dirt, not the dirt itself: two trees at the
    same HEAD with different uncommitted edits produce the same string. It is honest provenance and
    useless as an equality key.

    This is the stricter question -- "is the code in this tree exactly one published commit?" --
    and it answers only yes-with-the-commit or no. Callers that compare code between runs must use
    this one, so that an unresolvable or dirty tree can never satisfy an equality test.

    ``root`` defaults to the tree this module is imported from, so a caller asking about its own
    execution gets the same answer :func:`executing_git_sha` would resolve.
    """
    import subprocess

    repo = Path(root).resolve() if root is not None else Path(__file__).resolve().parent
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        if head.returncode != 0:
            return None
        sha = head.stdout.strip()
        if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha):
            return None
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=normal"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if status.returncode != 0 or status.stdout.strip():
            return None
        return sha
    except (OSError, subprocess.SubprocessError):
        return None


def task_number_from_name(task_name) -> int | None:
    """FOMO26 task number embedded in an asparagus task name, or None."""
    match = _TASK_RE.search(str(task_name or ""))
    return int(match.group(1)) if match else None


def evaluation_unit_id(evaluation_id, task, fold) -> str | None:
    """The canonical name for one (evaluation, task, fold) triple.

    Slash-separated rather than colon-separated: an evaluation id is free-form and a colon
    would make the value unsplittable once a path is involved.
    """
    if evaluation_id is None:
        return None
    return f"{evaluation_id}/task{task}/fold{fold}"


def _cfg_get(cfg, *names, default=None):
    """Read the first present of a dotted-path sequence out of a DictConfig or mapping."""
    for name in names:
        node = cfg
        for part in name.split("."):
            if node is None:
                break
            getter = getattr(node, "get", None)
            node = getter(part, None) if getter is not None else getattr(node, part, None)
        if node is not None:
            return node
    return default


def build_identity(
    cfg,
    *,
    role: str,
    checkpoint_path: str | None,
    prediction_path: str,
    run_dir: str,
    epoch: int | None,
    global_step: int | None,
    status: str,
    task_number=None,
) -> dict:
    """Assemble the identity from the already-resolved role. Digests the checkpoint once.

    ``checkpoint_path`` is what ``resolve_checkpoint_path`` returned for this run -- ``None``
    for ``current``. It is not re-derived here, and no other file is substituted for it.
    """
    if role in IN_MEMORY_ROLES and checkpoint_path is not None:
        raise ValueError(
            f"role {role!r} evaluates the in-memory weights, so it cannot carry checkpoint "
            f"path {checkpoint_path!r}. Attaching one would publish a digest for weights that "
            f"were never loaded."
        )
    if role in FILE_BACKED_ROLES and not checkpoint_path:
        raise ValueError(f"role {role!r} is file-backed but no checkpoint path was resolved")

    # Identity comes from the resolved configuration, never from the ambient environment. An
    # environment fallback would make the record depend on what happened to be exported into the
    # job, which is neither reviewable in `resolved_config.yaml` nor reproducible from it: two
    # runs of the same config could carry different evaluation ids and nothing would show it.
    # The launcher passes `+evaluation.id=` / `+evaluation.campaign_id=` as Hydra overrides.
    evaluation_id = _cfg_get(cfg, "evaluation.id")
    task_name = _cfg_get(cfg, "test_task")
    fold = _cfg_get(cfg, "data.fold")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "campaign_id": _cfg_get(cfg, "evaluation.campaign_id"),
        "evaluation_id": evaluation_id,
        "evaluation_unit_id": evaluation_unit_id(evaluation_id, task_number, fold),
        "task": task_number,
        "task_name": str(task_name) if task_name is not None else None,
        "fold": fold,
        "split": _cfg_get(cfg, "data.test_split"),
        "checkpoint_role": role,
        "evaluated_state": STATE_IN_MEMORY_FINAL if role in IN_MEMORY_ROLES else STATE_CHECKPOINT_FILE,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "epoch": int(epoch) if epoch is not None else None,
        "global_step": int(global_step) if global_step is not None else None,
        "prediction_path": str(prediction_path),
        # The commit whose working tree is EXECUTING, resolved from the module that is running.
        # `FOMO26_LAUNCH_GIT_COMMIT` is the submit-time commit and is recorded separately: the two
        # differ whenever a job is requeued, resumed, or starts after the checkout has moved, and
        # publishing the launch commit as the execution commit attributes results to code that did
        # not produce them.
        "git_sha": executing_git_sha(),
        "launch_git_sha": os.environ.get("FOMO26_LAUNCH_GIT_COMMIT") or None,
        "run_dir": str(run_dir),
    }


def write_identity(run_dir, identity: dict) -> Path:
    """Write the sidecar deterministically beside the run."""
    path = Path(run_dir) / FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    return path


def read_identity(path) -> dict:
    """Read a sidecar, rejecting an unrecognised schema and an internally inconsistent one."""
    identity = json.loads(Path(path).read_text())
    version = identity.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported evaluation-identity schema {version!r} in {path}; expected {SCHEMA_VERSION!r}")
    missing = [field for field in IDENTITY_FIELDS if field not in identity]
    if missing:
        raise ValueError(f"evaluation identity {path} is missing required field(s) {missing}")
    role = identity.get("checkpoint_role")
    state = identity.get("evaluated_state")
    if role in IN_MEMORY_ROLES:
        if state != STATE_IN_MEMORY_FINAL:
            raise ValueError(f"role {role!r} must record evaluated_state={STATE_IN_MEMORY_FINAL!r}, got {state!r}")
        if identity.get("checkpoint_path") or identity.get("checkpoint_sha256"):
            raise ValueError(
                f"role {role!r} evaluated the in-memory weights but the identity names a checkpoint "
                f"file; a digest here would describe weights that were never loaded"
            )
    elif role in FILE_BACKED_ROLES:
        if state != STATE_CHECKPOINT_FILE:
            raise ValueError(f"role {role!r} must record evaluated_state={STATE_CHECKPOINT_FILE!r}, got {state!r}")
        if not identity.get("checkpoint_path") or not identity.get("checkpoint_sha256"):
            raise ValueError(f"role {role!r} is file-backed and must record both a checkpoint path and its digest")
    else:
        raise ValueError(f"unknown checkpoint role {role!r} in {path}")
    return identity


def evaluate_and_record(
    cfg,
    *,
    trainer,
    model,
    datamodule,
    role: str,
    prediction_path: str,
    run_dir: str,
    best_ckpt_callback,
    last_ckpt_callback,
) -> dict:
    """Resolve the checkpoint once, evaluate it, and record what was evaluated.

    The single point where a role becomes a path, that path reaches ``trainer.test``, and both
    are written down. Segmentation, classification and regression all call this, so the three
    entrypoints cannot drift into three different notions of what they evaluated.

    ``epoch``/``global_step`` are read before the test loop, which resets them. For ``current``
    they identify the evaluated weights exactly; for ``best``/``last`` they describe the end of
    the fit that produced the checkpoint, whose own identity is ``checkpoint_sha256``.

    The sidecar is written *before* the evaluation (status ``evaluating``) so a run that dies
    mid-test still says what it was evaluating, and rewritten afterwards with the outcome.
    """
    from asparagus.pipeline.run.checkpoint_selection import resolve_checkpoint_path

    checkpoint_path = resolve_checkpoint_path(role, best_ckpt_callback, last_ckpt_callback)
    identity = build_identity(
        cfg,
        role=role,
        checkpoint_path=checkpoint_path,
        prediction_path=prediction_path,
        run_dir=run_dir,
        epoch=getattr(trainer, "current_epoch", None),
        global_step=getattr(trainer, "global_step", None),
        status="evaluating",
        task_number=task_number_from_name(_cfg_get(cfg, "test_task")),
    )
    write_identity(run_dir, identity)
    try:
        trainer.test(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)
    except BaseException:
        identity["status"] = "failed"
        write_identity(run_dir, identity)
        raise
    identity["status"] = "completed"
    write_identity(run_dir, identity)
    return identity


def main(argv: list[str] | None = None) -> int:
    """Read one field out of a run's identity sidecar, for shell callers.

    A launcher must not parse the sidecar with `sed`, and Jean-Zay does not guarantee `jq`.
    Exits non-zero when the sidecar is absent or invalid, so `set -e` fails the unit closed.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Print one field of a run's evaluation_identity.json.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--field", required=True, choices=IDENTITY_FIELDS)
    args = parser.parse_args(argv)
    identity = read_identity(Path(args.run_dir) / FILENAME)
    value = identity.get(args.field)
    print("" if value is None else value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
