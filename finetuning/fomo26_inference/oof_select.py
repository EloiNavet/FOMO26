#!/usr/bin/env python3
"""Choose the inference recipe from out-of-fold evidence, never from the final holdout.

The repository could already *compute* out-of-fold predictions for classification and regression,
but nothing consumed them to make a decision: TTA level, fold subset, sliding-window overlap and
ensemble space were all fixed by hand. That is the step where a holdout leak usually enters, because
the tempting way to pick a recipe is to try each one on the test split.

This module takes a list of OOF measurements -- one per candidate recipe -- and returns the winner
under a declared rule, with the whole comparison recorded. The output is the frozen inference recipe
the packaging stage consumes, so the submission can state exactly why it infers the way it does.

A recipe is only preferred over the incumbent when it beats it by more than ``--tie-margin``; ties
resolve to the cheaper recipe (fewer members x views), because an unjustified cost increase eats
into the 120 s/case budget for no measured gain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA_VERSION = "fomo26-oof-selection-v1"

#: Metric direction per task kind: True when larger is better.
HIGHER_IS_BETTER = {"dsc": True, "nsd": True, "auroc": True, "f1": True, "correlation": True, "mae": False}


class OOFSelectionError(RuntimeError):
    """The candidate set cannot support an honest selection."""


def _cost(recipe: dict) -> int:
    """Relative inference cost: ensemble members x TTA views x window passes."""
    views = {"none": 1, "flip3": 4, "flip7": 7}.get(str(recipe.get("tta", "none")), 1)
    members = int(recipe.get("n_members", 1))
    # Halving the overlap roughly doubles the number of windows per axis.
    overlap = float(recipe.get("overlap", 0.5))
    window_factor = max(1, round(1.0 / max(1e-6, 1.0 - overlap)))
    return members * views * window_factor


def select_recipe(candidates, *, metric: str, tie_margin: float = 0.0, incumbent: str | None = None) -> dict:
    """Pick the best recipe by OOF ``metric``, preferring the cheaper one on a tie."""
    if not candidates:
        raise OOFSelectionError("No candidate recipes were supplied.")
    if metric not in HIGHER_IS_BETTER:
        raise OOFSelectionError(f"Unknown metric {metric!r}; expected one of {sorted(HIGHER_IS_BETTER)}.")
    higher_better = HIGHER_IS_BETTER[metric]

    scored = []
    for candidate in candidates:
        candidate = dict(candidate)
        if candidate.get("source") == "holdout" or "TEST_" in json.dumps(candidate):
            raise OOFSelectionError(
                f"Candidate {candidate.get('name')!r} carries holdout-derived evidence. Recipe "
                "selection must run on out-of-fold predictions only; the holdout is read once, "
                "after this decision is frozen."
            )
        if metric not in candidate:
            raise OOFSelectionError(f"Candidate {candidate.get('name')!r} has no {metric!r} measurement.")
        candidate["cost"] = _cost(candidate)
        scored.append(candidate)

    def better(a, b) -> bool:
        """True when a beats b by more than the tie margin."""
        delta = a[metric] - b[metric]
        if not higher_better:
            delta = -delta
        if delta > tie_margin:
            return True
        if delta < -tie_margin:
            return False
        # Within the margin the two are indistinguishable; take the cheaper one.
        return a["cost"] < b["cost"]

    baseline = None
    if incumbent is not None:
        baseline = next((item for item in scored if item.get("name") == incumbent), None)
        if baseline is None:
            raise OOFSelectionError(f"Incumbent recipe {incumbent!r} is not among the candidates.")

    winner = baseline or scored[0]
    for candidate in scored:
        if candidate is winner:
            continue
        if better(candidate, winner):
            winner = candidate

    return {
        "schema_version": SCHEMA_VERSION,
        "metric": metric,
        "higher_is_better": higher_better,
        "tie_margin": tie_margin,
        "incumbent": incumbent,
        "selected": winner,
        "candidates": sorted(scored, key=lambda item: (-item[metric] if higher_better else item[metric], item["cost"])),
        "rule": (
            "Best out-of-fold metric; a challenger must beat the incumbent by more than tie_margin, "
            "otherwise the cheaper recipe wins. The final holdout is not consulted."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path, required=True, help="JSON list of OOF measurements.")
    parser.add_argument("--metric", required=True, choices=sorted(HIGHER_IS_BETTER))
    parser.add_argument("--tie-margin", type=float, default=0.0)
    parser.add_argument("--incumbent", default=None, help="Recipe that wins ties by default.")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        decision = select_recipe(
            json.loads(args.candidates.read_text()),
            metric=args.metric,
            tie_margin=args.tie_margin,
            incumbent=args.incumbent,
        )
    except OOFSelectionError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(f"Selected recipe {decision['selected'].get('name')} ({args.metric}={decision['selected'][args.metric]})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
