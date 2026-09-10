"""A task may cap the TTA ladder the time-budget governor chooses from.

Tasks 2 and 4 score *worse* under flip7 than flip3 on held-out folds, so leaving the governor free
to spend spare time budget on flip7 would ship the configuration the evidence rejected. The cap is
declared in the task registry, not in runtime logic, and it only ever removes levels: a task that
declares nothing keeps the full ladder, and the governor stays free to downgrade below the cap when
the measured pass time says the capped level will not fit.
"""

from __future__ import annotations

import pytest
from finetuning.fomo26_inference.time_budget import plan_inference
from finetuning.fomo26_inference.tta_safety import (
    TTA_LADDER,
    FlipTTAUnsafe,
    assert_tta_within_ladder,
    task_tta_ladder,
)

CAPPED = ("2", "4")
UNCAPPED = ("1", "3", "5")


@pytest.mark.parametrize("task", CAPPED)
def test_capped_tasks_expose_only_none_and_flip3(task: str) -> None:
    assert task_tta_ladder(task) == ("none", "flip3")


@pytest.mark.parametrize("task", UNCAPPED)
def test_uncapped_tasks_keep_the_full_ladder(task: str) -> None:
    assert task_tta_ladder(task) == TTA_LADDER
    assert "flip7" in task_tta_ladder(task)


@pytest.mark.parametrize("task", CAPPED)
def test_governor_can_never_select_flip7_for_a_capped_task(task: str) -> None:
    """Even with a budget large enough for flip7 many times over."""
    plan = plan_inference(
        t_per_pass=0.01,
        target_s=115.0,
        max_members=1,
        min_members=1,
        tta_levels=task_tta_ladder(task),
    )
    assert plan.tta == "flip3"


def test_governor_still_selects_flip7_for_task3() -> None:
    plan = plan_inference(
        t_per_pass=0.01,
        target_s=115.0,
        max_members=1,
        min_members=1,
        tta_levels=task_tta_ladder("3"),
    )
    assert plan.tta == "flip7"


@pytest.mark.parametrize("task", CAPPED)
def test_governor_remains_adaptive_under_the_cap(task: str) -> None:
    """A slow device must still be able to fall below the ceiling rather than overrun the budget."""
    ladder = task_tta_ladder(task)
    # 20 s per pass: flip3 is 4 views = 80 s + overhead, which fits 115 s but not 60 s.
    assert plan_inference(t_per_pass=20.0, target_s=115.0, max_members=1, min_members=1, tta_levels=ladder).tta == "flip3"
    assert plan_inference(t_per_pass=20.0, target_s=60.0, max_members=1, min_members=1, tta_levels=ladder).tta == "none"


@pytest.mark.parametrize("task", CAPPED)
def test_low_budget_downgrades_safely_rather_than_raising(task: str) -> None:
    """Below even one bare pass the governor returns the cheapest plan, it does not fail."""
    plan = plan_inference(
        t_per_pass=500.0,
        target_s=115.0,
        max_members=1,
        min_members=1,
        tta_levels=task_tta_ladder(task),
    )
    assert plan.tta == "none"
    assert plan.n_members == 1


@pytest.mark.parametrize("task", CAPPED)
def test_explicit_flip7_request_is_refused_for_a_capped_task(task: str) -> None:
    with pytest.raises(FlipTTAUnsafe, match="caps TTA"):
        assert_tta_within_ladder(task, "flip7")


@pytest.mark.parametrize("task", CAPPED)
def test_explicit_allowed_levels_are_accepted(task: str) -> None:
    assert_tta_within_ladder(task, "none")
    assert_tta_within_ladder(task, "flip3")


def test_explicit_flip7_is_accepted_where_no_cap_is_declared() -> None:
    assert_tta_within_ladder("3", "flip7")


def test_auto_is_never_refused_by_the_ladder_guard() -> None:
    """`auto` is resolved by the governor, which respects the ceiling by construction."""
    for task in CAPPED + UNCAPPED:
        assert_tta_within_ladder(task, "auto")
