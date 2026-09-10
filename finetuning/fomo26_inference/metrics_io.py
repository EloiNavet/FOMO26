"""Deterministic machine-readable metrics contract for FOMO26 downstream runs.

Infrastructure-only: this module *serializes* metrics into a stable ``metrics.json`` schema and
reduces the evaluators' per-class output to task-level numbers. It does not compute or alter any
metric. The metric names match the existing evaluators exactly:

* segmentation (``evaluate_seg.py``) -> ``dsc``, ``nsd``
* classification (``evaluate_clsreg.py``) -> ``auroc`` (and ``f1`` when provided)
* regression (``evaluate_clsreg.py``) -> ``mae``, ``correlation`` (the evaluator's Pearson ``r``)

Schema (``metrics.json``)::

    {
      "schema_version": "fomo26-metrics-v2",
      "task": 2,
      "task_name": "SEG902_FOMO26_Task2_lesion",
      "kind": "seg",
      "fold": 0,
      "status": "completed",
      "metrics": {"dsc": 0.0, "nsd": 0.0},
      "provenance": {...}   # v2 only; see PROVENANCE_FIELDS
    }

Schema versions
---------------
``fomo26-metrics-v2`` adds a ``provenance`` block that names, exactly, which artefacts produced
the record: the campaign and evaluation it belongs to, the checkpoint role that was evaluated,
the exact path and digest of the evaluated checkpoint (or an explicit in-memory marker with the
epoch and global step, for ``current``), the prediction file the metrics were computed from, the
split, and the Git commit. Before this, a metrics record carried no way to tell which checkpoint
or which code produced it, so a collector could only infer identity from directory and file
names.

Declaring the keys is not enough. A ``completed`` record is a claim that an evaluation happened,
so :func:`provenance_problems` requires the semantically applicable fields to be *filled in*, by
status and by role -- a completed record that cannot name its evaluation, its checkpoint or its
predictions is not evidence. Null stays correct exactly where the field does not apply:
``current`` has no checkpoint file to digest, and ``failed``/``missing``/``skipped``/``invalid``/
``incomplete`` records describe runs that produced no evaluation. v2 is not merged anywhere yet,
so its field set is corrected here rather than versioned again.

``fomo26-metrics-v1`` remains readable: it is the schema every historical FOMO26 run on disk was
written with, and re-running those evaluations is not possible. ``read_metrics_json`` accepts
both and rejects anything else, so an unrecognised schema fails loudly instead of being parsed
as if it were current.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import fmean

SCHEMA_VERSION = "fomo26-metrics-v2"
LEGACY_SCHEMA_VERSIONS = ("fomo26-metrics-v1",)
SUPPORTED_SCHEMA_VERSIONS = (SCHEMA_VERSION, *LEGACY_SCHEMA_VERSIONS)

# Every field a v2 record must carry, even when the value is unknown. The keys are always
# present so a missing provenance value is visibly null rather than silently absent.
PROVENANCE_FIELDS = (
    "campaign_id",
    "evaluation_id",
    "evaluation_unit_id",
    "source_checkpoint_sha256",
    "finetuning_checkpoint_sha256",
    "evaluated_checkpoint_role",
    "evaluated_checkpoint_path",
    "evaluated_checkpoint_sha256",
    "evaluated_state",
    "evaluated_epoch",
    "evaluated_global_step",
    "evaluated_fold_count",
    "prediction_path",
    "split_id",
    "split_sha256",
    "git_sha",
)

# Roles whose evaluated weights are a file, and therefore must name that file and its digest.
FILE_BACKED_ROLES = ("best", "last")
IN_MEMORY_ROLES = ("current",)
# A fold ensemble is evaluated by combining several fold checkpoints. There is no single
# evaluated checkpoint, so naming one would be a fabrication -- the record states how many folds
# were combined and leaves the singular checkpoint fields null.
ENSEMBLE_ROLES = ("fold_ensemble",)
#: An out-of-fold pass scores each case with the ONE fold checkpoint that never trained on it. It is
#: not an ensemble -- nothing is averaged -- but like an ensemble it spans several checkpoints, so no
#: single checkpoint identity describes it. It needs its own role rather than borrowing
#: ``fold_ensemble``, which would misdescribe how the number was produced.
OUT_OF_FOLD_ROLES = ("per_fold_out_of_fold",)
CHECKPOINT_ROLES = (*FILE_BACKED_ROLES, *IN_MEMORY_ROLES, *ENSEMBLE_ROLES, *OUT_OF_FOLD_ROLES)

STATE_CHECKPOINT_FILE = "checkpoint_file"
STATE_IN_MEMORY_FINAL = "in_memory_final"
STATE_FOLD_ENSEMBLE = "fold_ensemble"
STATE_OUT_OF_FOLD = "per_fold_out_of_fold"

# Provenance a *completed* v2 record must actually carry, not merely declare as a null key.
# Statuses outside this set describe a run that produced no evaluation, so their provenance is
# legitimately incomplete and is not required to be filled in.
COMPLETE_STATUS = "completed"
INCOMPLETE_STATUSES = ("failed", "missing", "skipped", "invalid", "incomplete")

REQUIRED_WHEN_COMPLETED = (
    "evaluation_id",
    "evaluation_unit_id",
    "evaluated_checkpoint_role",
    "evaluated_state",
    "prediction_path",
    "git_sha",
)

_TASK_RE = re.compile(r"Task(\d+)", re.IGNORECASE)


def task_number_from_name(task_name: str) -> int | None:
    """Extract the FOMO26 task number from an asparagus task name (e.g. SEG902_FOMO26_Task2 -> 2)."""
    match = _TASK_RE.search(task_name or "")
    return int(match.group(1)) if match else None


def build_record(
    task: int,
    fold: int | str | None,
    status: str,
    metrics: dict[str, float] | None,
    task_name: str | None = None,
    kind: str | None = None,
    extra: dict | None = None,
    provenance: dict | None = None,
) -> dict:
    """Assemble a metrics record in the stable schema. Pure; no I/O.

    Scalar metric values are coerced to float; nested breakdowns (e.g. seg ``per_class``) are kept as
    nested dicts of floats so ``evaluate_seg --output-json`` (which passes a ``per_class`` sub-dict)
    serialises instead of raising ``float(dict)``.
    """
    record = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "task_name": task_name,
        "kind": kind,
        "fold": fold,
        "status": status,
        "metrics": {k: _coerce_metric(v) for k, v in (metrics or {}).items()},
        "provenance": build_provenance(provenance),
    }
    if extra:
        record.update(extra)
    return record


def build_provenance(provenance: dict | None) -> dict:
    """Normalise a provenance mapping to exactly PROVENANCE_FIELDS.

    Unknown keys are rejected rather than passed through: provenance that varies field-by-field
    between writers cannot be joined across collectors, which is the situation this schema exists
    to end.
    """
    provided = dict(provenance or {})
    unknown = sorted(set(provided) - set(PROVENANCE_FIELDS))
    if unknown:
        raise ValueError(f"unknown provenance field(s) {unknown}; expected a subset of {list(PROVENANCE_FIELDS)}")
    return {field: provided.get(field) for field in PROVENANCE_FIELDS}


def provenance_problems(record: dict) -> list[str]:
    """Everything that stops this record from being a complete, self-consistent v2 record.

    Creating the provenance keys is not the same as filling them in. A record that says
    ``status: completed`` is a claim that an evaluation happened, and a claim that cannot name
    its evaluation, its checkpoint or its predictions is not usable as evidence -- so the
    applicable fields are required rather than permitted to stay null.

    Null remains correct where a value genuinely does not apply: ``current`` has no checkpoint
    file to digest, and a ``failed``/``missing``/``skipped``/``invalid``/``incomplete`` record
    describes a run that produced no evaluation at all. Returns a list of problems rather than
    raising, so a collector can report every one of them against the unit it belongs to.
    """
    problems: list[str] = []
    if record.get("schema_version") != SCHEMA_VERSION:
        return [f"not a {SCHEMA_VERSION} record (schema_version={record.get('schema_version')!r})"]
    status = record.get("status")
    if status != COMPLETE_STATUS:
        if status not in INCOMPLETE_STATUSES:
            problems.append(f"unknown status {status!r}; expected {COMPLETE_STATUS!r} or one of {list(INCOMPLETE_STATUSES)}")
        return problems

    for field in ("task", "fold"):
        if record.get(field) is None:
            problems.append(f"{field} is null on a completed record")
    provenance = record.get("provenance") or {}
    role_declared = provenance.get("evaluated_checkpoint_role")
    required = REQUIRED_WHEN_COMPLETED
    if role_declared in ENSEMBLE_ROLES + OUT_OF_FOLD_ROLES:
        # A standalone fold-ensemble or out-of-fold evaluation belongs to no campaign evaluation, so
        # it cannot invent an evaluation id or unit id. What it must not do -- claim a single
        # evaluated checkpoint -- is enforced by the role branch below.
        required = tuple(name for name in required if name not in ("evaluation_id", "evaluation_unit_id"))
    for field in required:
        if provenance.get(field) in (None, ""):
            problems.append(f"provenance.{field} is null on a completed record")
    if provenance.get("split_id") in (None, "") and provenance.get("split_sha256") in (None, ""):
        problems.append("neither provenance.split_id nor provenance.split_sha256 identifies the evaluated split")

    role = provenance.get("evaluated_checkpoint_role")
    state = provenance.get("evaluated_state")
    if role is not None and role not in CHECKPOINT_ROLES:
        problems.append(f"provenance.evaluated_checkpoint_role={role!r} is not one of {list(CHECKPOINT_ROLES)}")
    elif role in FILE_BACKED_ROLES:
        if state != STATE_CHECKPOINT_FILE:
            problems.append(f"role {role!r} must record evaluated_state={STATE_CHECKPOINT_FILE!r}, got {state!r}")
        for field in ("evaluated_checkpoint_path", "evaluated_checkpoint_sha256"):
            if provenance.get(field) in (None, ""):
                problems.append(f"provenance.{field} is null although role {role!r} evaluates a checkpoint file")
    elif role in ENSEMBLE_ROLES:
        if state != STATE_FOLD_ENSEMBLE:
            problems.append(f"role {role!r} must record evaluated_state={STATE_FOLD_ENSEMBLE!r}, got {state!r}")
        for field in ("evaluated_checkpoint_path", "evaluated_checkpoint_sha256"):
            if provenance.get(field) not in (None, ""):
                problems.append(
                    f"role {role!r} combines several fold checkpoints, so provenance.{field} must be "
                    f"null; a single value here would fabricate a checkpoint identity the ensemble "
                    f"does not have"
                )
        count = provenance.get("evaluated_fold_count")
        if not isinstance(count, int) or count < 1:
            problems.append(f"provenance.evaluated_fold_count must be a positive int for role {role!r}, got {count!r}")
    elif role in OUT_OF_FOLD_ROLES:
        if state != STATE_OUT_OF_FOLD:
            problems.append(f"role {role!r} must record evaluated_state={STATE_OUT_OF_FOLD!r}, got {state!r}")
        for field in ("evaluated_checkpoint_path", "evaluated_checkpoint_sha256"):
            if provenance.get(field) not in (None, ""):
                problems.append(
                    f"role {role!r} scores each case with a different fold checkpoint, so provenance.{field} "
                    f"must be null; a single value here would fabricate a checkpoint identity the pooled "
                    f"result does not have"
                )
        count = provenance.get("evaluated_fold_count")
        if not isinstance(count, int) or count < 1:
            problems.append(f"provenance.evaluated_fold_count must be a positive int for role {role!r}, got {count!r}")
        split = provenance.get("split_id") or ""
        if "TEST" in str(split).upper():
            problems.append(
                f"role {role!r} records split_id={split!r}, which names a held-out TEST split. An "
                f"out-of-fold result is scored on the per-fold validation partitions; labelling it "
                f"with the holdout would let the two be compared as if they were the same evidence"
            )
    elif role in IN_MEMORY_ROLES:
        if state != STATE_IN_MEMORY_FINAL:
            problems.append(f"role {role!r} must record evaluated_state={STATE_IN_MEMORY_FINAL!r}, got {state!r}")
        for field in ("evaluated_checkpoint_path", "evaluated_checkpoint_sha256"):
            if provenance.get(field) not in (None, ""):
                problems.append(
                    f"role {role!r} evaluated the in-memory weights, so provenance.{field} must be null; "
                    f"a value here describes weights that were never loaded"
                )
        for field in ("evaluated_epoch", "evaluated_global_step"):
            if provenance.get(field) is None:
                problems.append(f"provenance.{field} is null although role {role!r} has no checkpoint to identify it")
    return problems


def validate_record(record: dict) -> dict:
    """Return the record, or raise with every problem found. Use at write time."""
    problems = provenance_problems(record)
    if problems:
        raise ValueError("incomplete fomo26-metrics-v2 record:\n  - " + "\n  - ".join(problems))
    return record


def read_metrics_json(path, *, require_complete_v2: bool = False) -> dict:
    """Read a metrics record of any supported schema version, or fail loudly.

    v1 records are returned as written -- they carry no provenance block and none is invented for
    them. Callers that need provenance must ask for it: ``require_complete_v2`` rejects both a v1
    record and a v2 record whose completed status is not backed by usable provenance.
    """
    record = json.loads(Path(path).read_text())
    version = record.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported metrics schema {version!r} in {path}; this reader supports {list(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    if require_complete_v2:
        problems = provenance_problems(record)
        if problems:
            raise ValueError(f"{path} does not carry complete v2 provenance:\n  - " + "\n  - ".join(problems))
    return record


def _coerce_metric(value):
    """Coerce a metric value to JSON-safe floats, recursing into nested breakdowns (per_class)."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: _coerce_metric(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_metric(v) for v in value]
    return float(value)


def write_metrics_json(path: str | Path, record: dict) -> Path:
    """Write the record deterministically (sorted keys) and return the path.

    A ``completed`` v2 record that fails validation is refused here rather than written and
    caught later: once an under-specified record exists on disk, every consumer has to decide
    what to do with it, and at least one of them will guess.
    """
    validate_record(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True))
    return path


def reduce_seg_summary(summary: dict) -> dict[str, float]:
    """Reduce evaluate_seg's per-class ``{c: {dsc, nsd, n}}`` to task-level mean dsc/nsd.

    Mean is taken over foreground classes (the evaluator already excludes background). Keys may be
    ints or strings (JSON round-trips class ids as strings).
    """
    dsc_vals: list[float] = []
    nsd_vals: list[float] = []
    for _cls, vals in summary.items():
        if "dsc" in vals:
            dsc_vals.append(float(vals["dsc"]))
        if "nsd" in vals:
            nsd_vals.append(float(vals["nsd"]))
    out: dict[str, float] = {}
    if dsc_vals:
        out["dsc"] = float(fmean(dsc_vals))
    if nsd_vals:
        out["nsd"] = float(fmean(nsd_vals))
    return out


def seg_metrics_with_per_class(summary: dict) -> dict:
    """Task-level dsc/nsd plus the raw per-class breakdown (kept for provenance)."""
    reduced = reduce_seg_summary(summary)
    reduced["per_class"] = {str(c): {k: v for k, v in vals.items()} for c, vals in summary.items()}
    return reduced
