"""A scientific handoff may freeze a different TTA per task; the release contract must carry it.

``auto`` delegates the choice to the time-budget governor, which is correct when the scientific
authority delegates it and wrong when the authority froze one. A candidate whose handoff declares
flip3 on task3 and none on task2 cannot be represented by a single shared value, and shipping
``auto`` there would let the governor pick a level the science never selected.
"""

from __future__ import annotations

import pytest
from finetuning.container.handoff_adapter import ReleaseContractError, _declared_tta


def test_absent_declaration_falls_back_to_the_wrapper_default():
    assert _declared_tta({}, "task2", "auto") == "auto"


def test_declared_value_overrides_the_fallback():
    assert _declared_tta({"tta": "none"}, "task2", "auto") == "none"
    assert _declared_tta({"tta": "flip3"}, "task3", "auto") == "flip3"


def test_declared_auto_is_preserved_and_not_ladder_checked():
    assert _declared_tta({"tta": "auto"}, "task4", "none") == "auto"


def test_declaration_above_the_task_ladder_is_refused():
    # Task 2 and Task 4 cap at flip3 in the registry; a handoff cannot raise that ceiling.
    with pytest.raises(Exception):
        _declared_tta({"tta": "flip7"}, "task2", "auto")


def test_non_string_declaration_is_refused():
    with pytest.raises(ReleaseContractError, match="non-string TTA"):
        _declared_tta({"tta": 3}, "task2", "auto")
