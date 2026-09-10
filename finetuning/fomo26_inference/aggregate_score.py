"""Offline unified FOMO26 score aggregator.

Reads per-task metric files, normalizes each metric to [0, 1] using the challenge's
worst-case conventions, and combines them with the official leaderboard task weights.

This is intentionally dependency-light (standard library only) so it runs anywhere,
including sandboxes without torch/monai. The heavy per-task metric *computation* lives in
``evaluate_seg.py`` / ``evaluate_clsreg.py`` / ``metrics.py``; this module only aggregates
their outputs.

Input contract
--------------
``--results-root`` is a directory containing one file per task, discovered as either
``task<N>/metrics.json`` or ``task<N>.json`` (N in 1..7). Each file is JSON:

    {
      "task": 2,                       # optional if the filename encodes it
      "kind": "seg",                   # optional; inferred from task number
      "status": "completed",           # completed | failed | missing (optional)
      "metrics": {"dsc": 0.72, "nsd": 0.65}
    }

``metrics`` may also be given as top-level keys (no "metrics" wrapper). Tasks 6 and 7 are
platform-evaluated (hidden labels); supply them through ``--external-metrics external.json``
(a mapping of task number -> the same per-task object) if you have platform numbers.

Two scores are emitted, kept explicitly distinct:

* ``offline_proxy_score`` -- weighted, normalized aggregate over the available tasks. Clearly
  a proxy, never the official leaderboard number.
* ``official_compatible_score`` -- ``null`` here, because the official leaderboard applies
  subject-wise 100,000-permutation rank normalization that cannot be reproduced from summary
  metrics. The field and its reason are always emitted so downstream code never mistakes the
  proxy for the official score.

Example:
    python -m finetuning.fomo26_inference.aggregate_score \
        --results-root $ASPARAGUS_RESULTS/downstream_metrics \
        --output-json score.json --output-csv score.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

# Bump when the weights, normalization, or aggregation change.
FORMULA_VERSION = "fomo26-offline-v1"

# Official leaderboard weighting, as published in the FOMO26 challenge description (section 4):
# segmentation Tasks 2 & 4 weigh 0.25 each; image-level/probing Tasks 1,3,5,6,7 weigh 0.10 each.
TASK_WEIGHTS: dict[int, float] = {1: 0.10, 2: 0.25, 3: 0.10, 4: 0.25, 5: 0.10, 6: 0.10, 7: 0.10}
CHALLENGE_SCREEN_TASKS = (1, 2, 3, 4, 5)
CHALLENGE_SCREEN_WEIGHT = sum(TASK_WEIGHTS[task] for task in CHALLENGE_SCREEN_TASKS)

TASK_KIND: dict[int, str] = {1: "cls", 2: "seg", 3: "reg", 4: "seg", 5: "cls", 6: "cls", 7: "fairness"}

# Metrics that compose each task's normalized score. Only the ones actually present in the
# metric file are used; a task needs at least one present metric to be scored.
TASK_METRICS: dict[int, list[str]] = {
    1: ["auroc", "f1"],
    2: ["dsc", "nsd"],
    3: ["mae", "correlation"],
    4: ["dsc", "nsd"],
    5: ["auroc", "f1"],
    6: ["auroc", "f1"],
    7: ["fairness_score"],
}

# Per-metric [0, 1] normalization: score = clamp((value - worst) / (best - worst), 0, 1).
# "best"/"worst" encode metric direction, so higher-is-better and lower-is-better share one rule.
# Worst-case values follow the challenge "Metrics & Penalties" section.
METRIC_SPECS: dict[str, dict[str, float]] = {
    "dsc": {"best": 1.0, "worst": 0.0},
    "nsd": {"best": 1.0, "worst": 0.0},
    "auroc": {"best": 1.0, "worst": 0.0},
    "f1": {"best": 1.0, "worst": 0.0},
    "mae": {"best": 0.0, "worst": 100.0},  # lower is better; worst AE = 100
    "correlation": {"best": 1.0, "worst": -1.0},  # Pearson r in [-1, 1]
    "fairness_score": {"best": 1.0, "worst": 0.0},  # (1/|V|) sum_v (1 - D_v)
}

RESERVED_KEYS = {"task", "task_name", "kind", "status", "metrics"}

# Canonical, mutually exclusive per-task outcomes. The distinction that matters is
# "ran and scored badly" vs "produced no result at all":
#   success  -> completed with usable, finite metrics
#   failed   -> ran and failed; the challenge applies a worst-case 0 (a real outcome)
#   invalid  -> ran and produced partial/non-finite metrics; worst-case 0 (a real outcome)
#   missing  -> attempted but no result/metrics were produced (pipeline gap)  [ABSENT]
#   not_run  -> never scheduled for this cell                                  [ABSENT]
# ABSENT outcomes are pipeline gaps, not model performance, so they must never be
# folded into a score as if the model had scored zero.
OUTCOME_SUCCESS = "success"
OUTCOME_FAILED = "failed"
OUTCOME_INVALID = "invalid"
OUTCOME_MISSING = "missing"
OUTCOME_NOT_RUN = "not_run"
ABSENT_OUTCOMES = frozenset({OUTCOME_MISSING, OUTCOME_NOT_RUN})


class TaskResult:
    """Aggregation outcome for a single task."""

    def __init__(self, task: int) -> None:
        self.task = task
        self.kind = TASK_KIND[task]
        self.weight = TASK_WEIGHTS[task]
        self.status = "missing"
        self.outcome = OUTCOME_MISSING  # canonical outcome (see ABSENT_OUTCOMES)
        self.present = False  # a metric file (or external entry) was supplied
        self.valid = False  # a finite normalized score was produced
        self.metrics: dict[str, float] = {}
        self.normalized_metrics: dict[str, float] = {}
        self.task_score: float | None = None
        self.warnings: list[str] = []

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "kind": self.kind,
            "status": self.status,
            "outcome": self.outcome,
            "absent": self.outcome in ABSENT_OUTCOMES,
            "present": self.present,
            "valid": self.valid,
            "weight": self.weight,
            "metrics": self.metrics,
            "normalized_metrics": self.normalized_metrics,
            "task_score": self.task_score,
            "warnings": self.warnings,
        }


def normalize_metric(name: str, value: float) -> float:
    spec = METRIC_SPECS[name]
    best, worst = spec["best"], spec["worst"]
    norm = (value - worst) / (best - worst)
    return max(0.0, min(1.0, norm))


def _coerce_metrics(payload: dict) -> dict[str, float]:
    """Pull the metrics mapping out of a task payload (wrapped or flat)."""
    raw = payload.get("metrics")
    if not isinstance(raw, dict):
        raw = {k: v for k, v in payload.items() if k not in RESERVED_KEYS}
    metrics: dict[str, float] = {}
    for key, val in raw.items():
        try:
            metrics[key] = float(val)
        except (TypeError, ValueError):
            metrics[key] = math.nan
    return metrics


def score_task(task: int, payload: dict | None) -> TaskResult:
    """Turn one task payload into a normalized [0, 1] score, tolerating bad input."""
    result = TaskResult(task)
    if payload is None:
        result.status = "missing"
        result.warnings.append("no metric file found for this task")
        return result

    result.present = True
    result.status = str(payload.get("status", "completed"))
    result.metrics = _coerce_metrics(payload)

    if result.status == "failed":
        # Ran but failed: challenge applies the absolute worst-case penalty (score 0).
        result.valid = True
        result.outcome = OUTCOME_FAILED
        result.task_score = 0.0
        result.warnings.append("task marked failed -> worst-case score 0")
        return result

    wanted = TASK_METRICS[task]
    normalized: list[float] = []
    for name in wanted:
        if name not in result.metrics:
            continue
        value = result.metrics[name]
        if name not in METRIC_SPECS:
            result.warnings.append(f"unknown metric '{name}' ignored")
            continue
        if not math.isfinite(value):
            # NaN/inf metric -> worst case for that metric, with a warning.
            result.normalized_metrics[name] = 0.0
            normalized.append(0.0)
            result.warnings.append(f"metric '{name}' is NaN/inf -> worst-case 0")
            continue
        norm = normalize_metric(name, value)
        result.normalized_metrics[name] = norm
        normalized.append(norm)

    if not normalized:
        result.valid = False
        # No metrics at all means the pipeline produced an empty result file (a gap),
        # whereas some-but-unusable metrics means it ran and produced nothing scorable.
        if not result.metrics:
            result.status = OUTCOME_MISSING
            result.outcome = OUTCOME_MISSING
            result.warnings.append("result file contains no metrics -> treated as absent, not as a zero score")
        else:
            result.status = "invalid"
            result.outcome = OUTCOME_INVALID
            result.warnings.append(f"no usable metric among {wanted}; expected one of them present")
        return result

    result.valid = True
    result.outcome = OUTCOME_SUCCESS
    result.task_score = sum(normalized) / len(normalized)
    return result


def discover_task_payload(results_root: Path, task: int) -> dict | None:
    """Load task<N>/metrics.json or task<N>.json, returning None if absent, a sentinel on error."""
    candidates = [results_root / f"task{task}" / "metrics.json", results_root / f"task{task}.json"]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            return {"status": "failed", "metrics": {}, "_error": f"malformed metric file {path}: {exc}"}
        if not isinstance(data, dict):
            return {"status": "failed", "metrics": {}, "_error": f"metric file {path} is not a JSON object"}
        return data
    return None


def load_external_metrics(path: Path | None) -> dict[int, dict]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    out: dict[int, dict] = {}
    for key, val in data.items():
        try:
            out[int(str(key).lower().replace("task", "").strip())] = val
        except ValueError:
            continue
    return out


def aggregate(
    results_root: Path,
    external: dict[int, dict] | None = None,
    missing_policy: str = "exclude",
) -> dict:
    """Aggregate per-task metrics into the offline proxy score.

    missing_policy:
        "exclude"  -> renormalize weights over present-and-valid tasks (partial score).
        "penalize" -> missing tasks score 0 at full weight (all-7 assumption).
    """
    external = external or {}
    results: list[TaskResult] = []
    warnings: list[str] = []

    for task in sorted(TASK_WEIGHTS):
        payload = external.get(task)
        if payload is None:
            payload = discover_task_payload(results_root, task)
        if isinstance(payload, dict) and payload.get("_error"):
            warnings.append(payload["_error"])
        results.append(score_task(task, payload))

    missing_tasks = [r.task for r in results if not r.present]
    invalid_tasks = [r.task for r in results if r.present and not r.valid]

    if missing_policy == "penalize":
        contributing = results  # every task counts; missing/invalid score 0
    elif missing_policy == "exclude":
        contributing = [r for r in results if r.valid]
    else:
        raise ValueError(f"unknown missing_policy: {missing_policy!r}")

    weight_sum = sum(r.weight for r in contributing)
    per_task: list[dict] = []
    proxy = 0.0
    for r in results:
        entry = r.to_dict()
        score = r.task_score if r.valid else 0.0
        if missing_policy == "penalize":
            contribution = r.weight * score
            normalized_weight = r.weight
        else:
            counts = r.valid
            normalized_weight = (r.weight / weight_sum) if (counts and weight_sum > 0) else 0.0
            contribution = normalized_weight * score if counts else 0.0
        entry["normalized_weight"] = normalized_weight
        entry["weighted_contribution"] = contribution
        proxy += contribution
        per_task.append(entry)

    for r in results:
        warnings.extend(f"task{r.task}: {w}" for w in r.warnings)

    all_present_valid = all(r.valid for r in results)
    official_reason = (
        "The official leaderboard applies subject-wise 100,000-permutation rank normalization "
        "(FOMO26_challenge.md section 4), which cannot be reproduced from summary metrics offline."
    )

    return {
        "formula_version": FORMULA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "results_root": str(results_root),
        "missing_policy": missing_policy,
        "weights": TASK_WEIGHTS,
        "per_task": per_task,
        "missing_tasks": missing_tasks,
        "invalid_tasks": invalid_tasks,
        "warnings": warnings,
        "offline_proxy_score": proxy,
        "official_compatible_score": None,
        "official_compatible_reason": official_reason,
        "all_tasks_present_and_valid": all_present_valid,
    }


def aggregate_tasks_1_5(payloads: dict[int, dict | None]) -> dict:
    """Fixed-weight challenge proxy used by the architecture campaign.

    Outcomes are classified as success / failed / invalid / missing / not_run.

    ``failed`` and ``invalid`` mean the task actually ran and produced nothing scorable;
    the challenge scores those 0 at full weight (weights are never redistributed).

    ``missing`` and ``not_run`` are ABSENT: the pipeline produced no result, which is not
    evidence that the model scores 0. An absent task therefore makes the record
    *incomplete* -- ``downstream_score_1_5`` is None rather than a zero-diluted number
    that would silently read as a genuine (bad) result. A task key absent from
    ``payloads`` is ``not_run``; a key present with no payload/metrics is ``missing``.
    """

    results = []
    for task in CHALLENGE_SCREEN_TASKS:
        scheduled = task in payloads
        result = score_task(task, payloads.get(task))
        if not result.present:
            result.outcome = OUTCOME_MISSING if scheduled else OUTCOME_NOT_RUN
            result.status = result.outcome
            result.task_score = None
            if not scheduled:
                result.warnings.append("task was never scheduled for this cell -> not_run (not scored)")
        elif result.outcome in (OUTCOME_FAILED, OUTCOME_MISSING, OUTCOME_INVALID):
            pass  # already classified by score_task (ran-and-failed, empty file, unusable metrics)
        elif result.status != "completed":
            # Non-terminal status (e.g. "running"): the job never produced a final result,
            # so it is absent rather than a legitimate zero -- even if partial metrics exist.
            result.outcome = OUTCOME_MISSING
            result.warnings.append(
                f"task status is {result.status!r}, not 'completed' -> treated as absent (not scored as zero)"
            )
        elif result.outcome == OUTCOME_SUCCESS:
            # The campaign proxy requires *every* metric of the task, not just one.
            required = TASK_METRICS[result.task]
            missing = [name for name in required if name not in result.metrics]
            nonfinite = [name for name in required if name in result.metrics and not math.isfinite(result.metrics[name])]
            if missing or nonfinite:
                result.valid = False
                result.status = "invalid"
                result.outcome = OUTCOME_INVALID
                result.task_score = None
                result.warnings.append(
                    f"campaign score requires every task metric; missing={missing}, nonfinite={nonfinite} -> task score 0"
                )
        if result.outcome in ABSENT_OUTCOMES:
            result.valid = False
            result.task_score = None
        results.append(result)

    by_outcome = {
        name: [r.task for r in results if r.outcome == name]
        for name in (OUTCOME_SUCCESS, OUTCOME_FAILED, OUTCOME_INVALID, OUTCOME_MISSING, OUTCOME_NOT_RUN)
    }
    absent_tasks = sorted(by_outcome[OUTCOME_MISSING] + by_outcome[OUTCOME_NOT_RUN])
    complete = not absent_tasks

    # Only tasks that actually ran contribute; absent tasks are excluded entirely rather
    # than counted as zero. Weights are still never redistributed: an incomplete record
    # simply has no comparable score.
    weighted_sum = sum(TASK_WEIGHTS[r.task] * (r.task_score or 0.0) for r in results if r.outcome not in ABSENT_OUTCOMES)
    promotable = complete and all(r.outcome == OUTCOME_SUCCESS for r in results)
    return {
        "formula_version": f"{FORMULA_VERSION}-tasks1-5",
        "task_weights": {task: TASK_WEIGHTS[task] for task in CHALLENGE_SCREEN_TASKS},
        "weighted_sum_raw": weighted_sum if complete else None,
        "weighted_sum_max": CHALLENGE_SCREEN_WEIGHT,
        "downstream_score_1_5": (weighted_sum / CHALLENGE_SCREEN_WEIGHT) if complete else None,
        "complete": complete,
        "promotable": promotable,
        "per_task": [result.to_dict() for result in results],
        "outcomes": {name: by_outcome[name] for name in by_outcome},
        "absent_tasks": absent_tasks,
        "success_tasks": by_outcome[OUTCOME_SUCCESS],
        "missing_tasks": by_outcome[OUTCOME_MISSING],
        "not_run_tasks": by_outcome[OUTCOME_NOT_RUN],
        "invalid_tasks": by_outcome[OUTCOME_INVALID],
        "failed_tasks": by_outcome[OUTCOME_FAILED],
    }


def write_csv(report: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["task", "kind", "status", "valid", "weight", "normalized_weight", "task_score", "weighted_contribution"]
        )
        for entry in report["per_task"]:
            writer.writerow(
                [
                    entry["task"],
                    entry["kind"],
                    entry["status"],
                    entry["valid"],
                    entry["weight"],
                    round(entry["normalized_weight"], 6),
                    "" if entry["task_score"] is None else round(entry["task_score"], 6),
                    round(entry["weighted_contribution"], 6),
                ]
            )
        writer.writerow(["TOTAL", "", "", "", "", "", "offline_proxy_score", round(report["offline_proxy_score"], 6)])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", type=Path, required=True, help="Directory with per-task metric files.")
    p.add_argument("--output-json", type=Path, required=True, help="Where to write the aggregated report JSON.")
    p.add_argument("--output-csv", type=Path, default=None, help="Optional CSV summary path.")
    p.add_argument(
        "--external-metrics",
        type=Path,
        default=None,
        help="JSON mapping task-number -> metric object (e.g. platform-provided Tasks 6/7).",
    )
    p.add_argument(
        "--missing-policy",
        choices=["exclude", "penalize"],
        default="exclude",
        help="exclude: renormalize weights over available tasks (default). penalize: missing tasks score 0.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.results_root.is_dir():
        raise SystemExit(f"results-root is not a directory: {args.results_root}")
    external = load_external_metrics(args.external_metrics)
    report = aggregate(args.results_root, external=external, missing_policy=args.missing_policy)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=False))
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(report, args.output_csv)

    print(f"formula_version    : {report['formula_version']}")
    print(f"offline_proxy_score: {report['offline_proxy_score']:.6f} (missing_policy={report['missing_policy']})")
    print(f"official_compatible: {report['official_compatible_score']} -- {report['official_compatible_reason']}")
    for entry in report["per_task"]:
        score = "n/a" if entry["task_score"] is None else f"{entry['task_score']:.4f}"
        print(
            f"  task{entry['task']} [{entry['kind']}] status={entry['status']:<9} "
            f"score={score} weight={entry['weight']} contrib={entry['weighted_contribution']:.4f}"
        )
    if report["missing_tasks"]:
        print(f"missing tasks: {report['missing_tasks']}")
    if report["invalid_tasks"]:
        print(f"invalid tasks: {report['invalid_tasks']}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
