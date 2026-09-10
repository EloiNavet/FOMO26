"""A geometry-declaring image must carry an explicit orientation policy.

`runtime_geometry.orientation_policy` defaults to `error`, which refuses a case whose orientation
is not the one the task was fitted at. That default is right for a library, but shipping it would
make one unexpected orientation cost the whole case. The release freezes `skip` instead, so such a
case falls back to native geometry -- the behaviour every pre-canonicalization release shipped.
"""

from __future__ import annotations

import pytest
from finetuning.container.build_container import GEOMETRY_ORIENTATION_POLICY, _runtime_env
from finetuning.fomo26_inference.runtime_geometry import (
    ORIENTATION_POLICIES,
    task_runtime_target_spacing,
)

GEOMETRY_TASKS = ("task3", "task4")
PLAIN_TASKS = ("task1", "task2", "task5")


def _manifest(task: str) -> dict:
    policy = {
        "ensemble_members": [0],
        "tta": "auto",
        "time_target_seconds": 115.0,
        "calibration": {"state": "none"},
        "cross_patch": "none",
        "ensemble_space": "prob",
        "window_policy": "checkpoint_config_overlap_0.5",
    }
    return {"tasks": {task: {"policy": policy}}}


def test_the_declared_policy_is_a_valid_one() -> None:
    assert GEOMETRY_ORIENTATION_POLICY in ORIENTATION_POLICIES


def test_release_does_not_ship_the_failing_default() -> None:
    assert GEOMETRY_ORIENTATION_POLICY == "skip"


@pytest.mark.parametrize("task", GEOMETRY_TASKS)
def test_geometry_tasks_freeze_the_policy_into_their_runtime_env(task: str) -> None:
    assert task_runtime_target_spacing(int(task.removeprefix("task"))) is not None
    env = _runtime_env(task, _manifest(task))
    assert env["FOMO26_GEOMETRY_ORIENTATION_POLICY"] == GEOMETRY_ORIENTATION_POLICY


@pytest.mark.parametrize("task", PLAIN_TASKS)
def test_tasks_without_geometry_do_not_carry_an_inert_variable(task: str) -> None:
    assert task_runtime_target_spacing(int(task.removeprefix("task"))) is None
    assert "FOMO26_GEOMETRY_ORIENTATION_POLICY" not in _runtime_env(task, _manifest(task))
