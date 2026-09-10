"""The packaged runtime must carry the data files it opens, not only the modules it imports.

``audit_staged_runtime_imports`` walks the Python import closure of each entrypoint, so a module
that reads a JSON registry at run time passes that audit while its data file is absent from the
build context. That is not hypothetical: a released Tasks 1-5 image failed on the qualification
node with ``FOMO26_TTA=auto`` because ``finetuning/fomo26_inference/task_definitions.json`` had not been
staged, and the image had to be repaired after it was built.
"""

from __future__ import annotations

from finetuning.container.build_container import RELEASE_RUNTIME_PATHS, _tracked_runtime_files
from finetuning.fomo26_inference import tta_safety
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_tta_registry_is_declared_as_a_runtime_path() -> None:
    """The registry tta_safety resolves must be in the allowlist under its repo-relative name."""
    relative = tta_safety.REGISTRY_PATH.relative_to(REPO).as_posix()
    assert relative in RELEASE_RUNTIME_PATHS


def test_tta_registry_is_staged_into_the_runtime_source_set() -> None:
    """It must also survive the tracked-file enumeration that actually populates the context."""
    relative = tta_safety.REGISTRY_PATH.relative_to(REPO)
    assert relative in set(_tracked_runtime_files())


def test_every_declared_runtime_path_exists() -> None:
    """An allowlisted path that no longer exists would silently drop from the packaged runtime."""
    missing = [path for path in RELEASE_RUNTIME_PATHS if not (REPO / path).exists()]
    assert missing == []
