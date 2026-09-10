"""Refuse flip TTA on label spaces where mirroring is not label-preserving.

Flip TTA predicts on a mirrored volume and un-mirrors the result. That is only correct when no
class encodes laterality: if a task ever labels "left nerve" and "right nerve" as distinct classes,
the inverse flip puts the left prediction on the right side and the averaged ensemble is wrong in a
way no metric flags -- Dice stays plausible, the error is systematic and silent.

Today every FOMO26 task is flip-invariant (Task 2 has one meningioma foreground class, Task 4 labels
nerves vs vessels rather than sides, and 1/3/5 are image-level). So this module changes no numbers.
It exists so that stays true by assertion instead of by luck: the registry declares the property per
task, and a future label-space change that breaks it fails here rather than in a submission.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parent / "task_definitions.json"


class FlipTTAUnsafe(ValueError):
    """Flip TTA was requested for a label space that mirroring does not preserve."""


@lru_cache(maxsize=1)
def _task_definitions() -> dict:
    return json.loads(REGISTRY_PATH.read_text())["tasks"]


def task_flip_policy(task) -> dict:
    """Return the declared flip policy for one task, or raise if the task is undeclared."""
    definitions = _task_definitions()
    key = str(task)
    if key not in definitions:
        raise FlipTTAUnsafe(
            f"Task {key!r} is not declared in {REGISTRY_PATH}. Flip TTA cannot be shown to be "
            "label-preserving for an unknown label space; declare the task before enabling TTA."
        )
    entry = definitions[key]
    if "flip_invariant_labels" not in entry:
        raise FlipTTAUnsafe(
            f"Task {key} does not declare flip_invariant_labels. Add the declaration (with its "
            "rationale) before enabling flip TTA."
        )
    return {
        "flip_invariant_labels": bool(entry["flip_invariant_labels"]),
        "flip_label_permutation": entry.get("flip_label_permutation"),
        "rationale": entry.get("flip_invariance_rationale", ""),
    }


def assert_flip_tta_allowed(task, tta: str) -> None:
    """Fail closed when flip TTA would silently mislabel a laterality-encoding task.

    ``tta="none"`` is always allowed. ``tta="auto"`` is resolved by the time-budget governor into a
    concrete flip level, so it is treated as "flips may be used" and checked here too.
    """
    if str(tta) == "none":
        return
    policy = task_flip_policy(task)
    if policy["flip_invariant_labels"]:
        return
    if policy["flip_label_permutation"]:
        # A declared permutation is the other correct answer; the caller must apply it.
        return
    raise FlipTTAUnsafe(
        f"Task {task} declares flip_invariant_labels=false and no flip_label_permutation, so "
        f"tta={tta!r} would mirror predictions without remapping laterality-encoded classes. "
        "Use tta='none', or declare the class permutation that the inverse flip must apply."
    )


#: The ordered TTA ladder, cheapest first. ``time_budget.plan_inference`` walks it and takes the
#: richest level that fits the time target, so truncating it is what caps a task.
TTA_LADDER = ("none", "flip3", "flip7")


class TTALadderInvalid(ValueError):
    """A task declares a max_tta that is not a level of the ladder."""


def task_tta_ladder(task) -> tuple[str, ...]:
    """The TTA levels the governor may choose from for one task.

    A task declaring ``max_tta`` has evidence that the richer levels score *worse* on held-out
    data, so the ladder is truncated there and the governor can never spend spare time budget on a
    level the science rejected. A task declaring nothing keeps the full ladder -- this field only
    ever removes options, never adds them, and it never pins the choice: the governor is still free
    to select a cheaper level when the measured pass time says the capped one will not fit.
    """
    entry = _task_definitions().get(str(task), {})
    ceiling = entry.get("max_tta")
    if ceiling is None:
        return TTA_LADDER
    if ceiling not in TTA_LADDER:
        raise TTALadderInvalid(f"Task {task} declares max_tta={ceiling!r}, which is not one of {TTA_LADDER}.")
    return TTA_LADDER[: TTA_LADDER.index(ceiling) + 1]


def assert_tta_within_ladder(task, tta: str) -> None:
    """Fail closed when an explicit TTA request exceeds the task's declared ceiling.

    The governor respects the ceiling by construction, but ``FOMO26_TTA`` can also name a level
    directly. Honouring that silently would ship the configuration the held-out evidence rejected.
    """
    if task is None or str(tta) == "auto":
        return
    ladder = task_tta_ladder(task)
    if str(tta) not in ladder:
        raise FlipTTAUnsafe(
            f"Task {task} caps TTA at {ladder[-1]!r} on held-out evidence, so tta={tta!r} is not "
            f"available. Allowed levels: {ladder}."
        )


def flip_label_permutation(task) -> list[int] | None:
    """Class permutation to apply after the inverse flip, when the task declares one."""
    return task_flip_policy(task)["flip_label_permutation"]
