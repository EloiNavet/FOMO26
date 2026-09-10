"""The release runtime must contain the dependency version that trained the packaged weights.

gardening_tools 0.3.5 renamed ResidualUNetDecoder's parameters, so a container built against it
loads a 0.3.2-trained segmentation checkpoint's encoder and silently leaves all 50 decoder tensors
randomly initialised. The image still builds, validates and meets its runtime budget while
predicting background everywhere, so the pin is what keeps the packaged model the trained model.
"""

from __future__ import annotations

import pytest
import re
from finetuning.container.build_container import (
    RELEASE_RUNTIME_OVERRIDES,
    _apply_release_runtime_overrides,
    _override_blocks,
)

LOCK = """\
fsspec==2025.3.0 \\
    --hash=sha256:aaaa
    # via torch
gardening-tools==0.3.5 \\
    --hash=sha256:93140862d775fd007e861b8a89edb1f47722f52057f36b2fbe112bbc8162839a \\
    --hash=sha256:c5290579d7f78c8236296be3776083b5f803f23a328b03227f600e01588d4f95
    # via asparagus
gitdb==4.0.12 \\
    --hash=sha256:bbbb
    # via gitpython
"""


def _entry(text: str, package: str) -> str:
    match = re.search(rf"^{re.escape(package)}==[^\n]*\n(?:[ \t]+[^\n]*\n|[ \t]*#[^\n]*\n)*", text, re.MULTILINE)
    return match.group(0) if match else ""


def test_override_pins_the_training_version(tmp_path):
    requirements = tmp_path / "req.txt"
    requirements.write_text(LOCK)
    _apply_release_runtime_overrides(requirements)
    text = requirements.read_text()
    assert "gardening-tools==0.3.2" in text
    assert "0.3.5" not in text


def test_override_carries_hashes_so_require_hashes_still_holds(tmp_path):
    requirements = tmp_path / "req.txt"
    requirements.write_text(LOCK)
    _apply_release_runtime_overrides(requirements)
    entry = _entry(requirements.read_text(), "gardening-tools")
    # --require-hashes rejects any unhashed requirement, so a pin without hashes would not install.
    assert entry.count("--hash=sha256:") >= 1


def test_override_leaves_other_requirements_untouched(tmp_path):
    requirements = tmp_path / "req.txt"
    requirements.write_text(LOCK)
    _apply_release_runtime_overrides(requirements)
    text = requirements.read_text()
    assert _entry(text, "fsspec") == _entry(LOCK, "fsspec")
    assert _entry(text, "gitdb") == _entry(LOCK, "gitdb")


def test_missing_package_fails_loudly_rather_than_silently_skipping(tmp_path):
    requirements = tmp_path / "req.txt"
    requirements.write_text("fsspec==2025.3.0 \\\n    --hash=sha256:aaaa\n")
    with pytest.raises(SystemExit, match="Expected exactly one gardening-tools entry"):
        _apply_release_runtime_overrides(requirements)


def test_every_declared_override_has_a_hashed_block():
    blocks = _override_blocks()
    for package, version in RELEASE_RUNTIME_OVERRIDES.items():
        assert package in blocks, f"{package} declared without a pinned block"
        assert f"{package}=={version}" in blocks[package]
        assert "--hash=sha256:" in blocks[package]
