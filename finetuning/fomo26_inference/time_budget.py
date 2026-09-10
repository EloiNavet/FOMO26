"""Per-subject time-budget governor for the submission container.

The challenge gives 120 s/case on an H100. A failed (timed-out) submission is wasted, so we must
stay UNDER 120 s but USE as much of it as possible (more ensemble members + TTA = better/steadier
scores). This governor measures the real cost of one inference pass at container start-up, then picks
the richest (n_members x TTA) configuration whose estimated time fits a safety target (~115 s).

One "pass" = one model evaluated once (one sliding-window or one forward). Total passes for a config
= n_members * n_tta_views. Estimated time = passes * t_per_pass + fixed_overhead.
"""

from __future__ import annotations

import time
import torch
from dataclasses import dataclass

# TTA view counts must match _TTA_FLIPS in the ensemble modules.
TTA_VIEWS = {"none": 1, "flip3": 4, "flip7": 7}


def measure_pass_time(fn, *args, warmup: int = 1, repeats: int = 2, **kwargs) -> float:
    """Median wall-time (seconds) of fn(*args), with CUDA sync and warm-up."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


@dataclass
class InferencePlan:
    n_members: int
    tta: str
    est_seconds_per_case: float
    passes: int


def plan_inference(
    t_per_pass: float,
    budget_s: float = 120.0,
    target_s: float = 115.0,
    max_members: int = 10,
    tta_levels: tuple[str, ...] = ("none", "flip3", "flip7"),
    fixed_overhead_s: float = 2.0,
    min_members: int = 1,
) -> InferencePlan:
    """Pick the config maximising passes (members x TTA views) under the time target.

    Prefers more ensemble members first (usually the biggest score gain), then richer TTA.
    """
    best = None
    for members in range(min_members, max_members + 1):
        for tta in tta_levels:
            passes = members * TTA_VIEWS[tta]
            est = passes * t_per_pass + fixed_overhead_s
            if est <= min(budget_s, target_s):
                cand = InferencePlan(members, tta, est, passes)
                # maximise passes; tie-break on more members (more decorrelated than more flips)
                if best is None or (cand.passes, cand.n_members) > (best.passes, best.n_members):
                    best = cand
    if best is None:
        # Even a single bare pass exceeds the budget: fall back to the cheapest possible config.
        best = InferencePlan(min_members, "none", t_per_pass * min_members + fixed_overhead_s, min_members)
    return best


def describe_plan(plan: InferencePlan, budget_s: float = 120.0) -> str:
    return (
        f"plan: {plan.n_members} member(s) x TTA={plan.tta} "
        f"({plan.passes} passes) ~{plan.est_seconds_per_case:.1f}s/case "
        f"({100 * plan.est_seconds_per_case / budget_s:.0f}% of {budget_s:.0f}s budget)"
    )
