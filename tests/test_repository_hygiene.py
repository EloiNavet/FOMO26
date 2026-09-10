"""Repository-level invariants that a review reads past but CI must not.

Both contracts here were broken once and only surfaced on GitHub Actions: compiled bytecode
became committable because a later negation pattern overrode the repository-wide rule, and two
test modules imported the test tree as a package, which resolves locally (the repository root is
on `sys.path` in a developer checkout) but not under a bare `uv run pytest` on a clean runner.
"""

from __future__ import annotations

import ast
import pytest
import re
import subprocess
from pathlib import Path

pytestmark = pytest.mark.reconciliation

REPO = Path(__file__).resolve().parents[1]
BYTECODE = re.compile(r"(?:^|/)__pycache__/|\.py[cod]$")
TEST_ROOTS = ("tests", "finetuning/tests")


def _tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True)
    return out.stdout.split("\n")


def test_no_compiled_python_bytecode_is_tracked():
    offenders = sorted(path for path in _tracked() if path and BYTECODE.search(path))
    assert not offenders, (
        "compiled Python bytecode is tracked; remove it and keep the repository-wide "
        "`__pycache__/` and `*.py[cod]` ignore rules in place:\n  " + "\n  ".join(offenders)
    )


def test_gitignore_ignores_bytecode_repository_wide():
    """Enumerating individual `.pyc` paths does not stop the next one from being committed."""

    rules = {line.strip() for line in (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()}
    assert "__pycache__/" in rules
    assert "*.py[cod]" in rules
    # A negation pattern placed after these rules used to win, because the last matching pattern
    # decides. These probes fail if that ever happens again, at any depth.
    probes = [
        "asparagus/__pycache__/probe.cpython-311.pyc",
        "finetuning/container/__pycache__/probe.cpython-311.pyc",
        "tests/nested/deeper/__pycache__/probe.cpython-311.pyc",
        "asparagus/probe.pyc",
        "tests/probe.pyo",
    ]
    for probe in probes:
        result = subprocess.run(["git", "check-ignore", "-q", "--no-index", probe], cwd=REPO, check=False)
        assert result.returncode == 0, f"{probe} would not be ignored"
    # …while ordinary sources stay tracked.
    kept = subprocess.run(["git", "check-ignore", "-q", "--no-index", "asparagus/paths.py"], cwd=REPO, check=False)
    assert kept.returncode == 1, "asparagus/paths.py must remain tracked"


def _test_modules() -> list[Path]:
    return sorted(path for root in TEST_ROOTS for path in (REPO / root).rglob("test_*.py"))


def test_no_test_module_imports_the_test_tree_as_a_package():
    """`tests/` has no `__init__.py`, so `import tests.x` depends on the repository root
    being on `sys.path`. A sibling `import test_x` works under pytest's default import mode,
    which is what CI runs."""

    offenders: list[str] = []
    for path in _test_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in {"tests"}:
                offenders.append(f"{path.relative_to(REPO)}:{node.lineno}: from {node.module} import ...")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "tests":
                        offenders.append(f"{path.relative_to(REPO)}:{node.lineno}: import {alias.name}")
    assert not offenders, "import the sibling module directly instead:\n  " + "\n  ".join(offenders)
