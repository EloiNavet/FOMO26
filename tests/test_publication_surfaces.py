"""What a fresh public repository exposes on its very first push.

The workflows were inherited from the upstream project this code grew out of, and three of them
triggered `on: push`, required a `secrets.DEPLOY` key that will not exist, and pushed to
repositories belonging to someone else. On a public mirror that is not a failing job, it is a
public log of an attempt to write to a third party's repository.

These assertions are about the publication surface only -- what runs unattended, for whom, with
what credentials -- not about the substance of CI.
"""

from __future__ import annotations

import pytest
import re
import yaml
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((REPO / ".github" / "workflows").glob("*.y*ml"))

# asparagus/paths.py raises at import when these are unset, so a workflow that runs pytest without
# them fails at collection -- a red badge on the first push for a reason unrelated to the code.
REQUIRED_ENV = ("ASPARAGUS_DATA", "ASPARAGUS_MODELS", "ASPARAGUS_RESULTS", "ASPARAGUS_RAW_LABELS")


def test_there_is_at_least_one_workflow_and_all_parse():
    assert WORKFLOWS, "no workflows found; this test would otherwise pass vacuously"
    for path in WORKFLOWS:
        assert yaml.safe_load(path.read_text()), f"{path.name} is empty or unparseable"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_no_workflow_needs_a_secret(path):
    """`secrets.GITHUB_TOKEN` is provided automatically; anything else must be configured first."""
    used = set(re.findall(r"secrets\.([A-Za-z_][A-Za-z0-9_]*)", path.read_text()))
    assert used <= {"GITHUB_TOKEN"}, (
        f"{path.name} needs secret(s) that a new repository will not have: {sorted(used - {'GITHUB_TOKEN'})}"
    )


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_no_workflow_targets_a_third_party_repository(path):
    text = path.read_text()
    for owner in ("Sllambias",):
        assert owner not in text, f"{path.name} still references the upstream owner {owner!r}"
    assert "git@github.com:" not in text, f"{path.name} pushes over SSH to a named remote"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_no_workflow_requires_a_private_runner(path):
    doc = yaml.safe_load(path.read_text())
    for job in (doc.get("jobs") or {}).values():
        runs_on = job.get("runs-on")
        labels = [runs_on] if isinstance(runs_on, str) else list(runs_on or [])
        assert "self-hosted" not in labels, f"{path.name} wants a private runner: {labels}"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_a_workflow_that_runs_pytest_supplies_the_data_root_contract(path):
    """Otherwise the first push goes red at collection, not at a real failure."""
    text = path.read_text()
    if "pytest" not in text:
        pytest.skip(f"{path.name} runs no tests")
    for var in REQUIRED_ENV:
        assert var in text, f"{path.name} runs pytest without {var}; collection will fail"


def test_no_submodules_and_no_lfs_pointers():
    """A submodule or LFS object turns a clone into a fetch from somewhere else."""
    assert not (REPO / ".gitmodules").exists(), "a submodule would need its own remote to be public"
    attributes = REPO / ".gitattributes"
    if attributes.exists():
        assert "filter=lfs" not in attributes.read_text(), "LFS content is not carried by a plain clone"


def test_no_inherited_issue_automation():
    """A stale-bot inherited from another project would start closing this one's issues."""
    for path in WORKFLOWS:
        doc = yaml.safe_load(path.read_text())
        triggers = doc.get(True) or doc.get("on") or {}
        if isinstance(triggers, dict) and "schedule" in triggers:
            pytest.fail(f"{path.name} runs on a schedule unattended; confirm that is intended")
