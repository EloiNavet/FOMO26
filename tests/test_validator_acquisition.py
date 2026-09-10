"""Acquiring the external validator: what is fetched, what is refused, what absence means.

The validator is not in this repository. Its upstream declares no licence, so vendoring 84 of its
files into a public release would be redistribution on undefined terms. What remains here is a
manifest of paths and digests, and a way to fetch the pinned commit and check it against that.

Everything below is synthetic. No test reaches the network: the acquisition path is exercised
against locally built archives, because a suite that downloads from a third party to prove it can
download from a third party is not a test, it is a dependency.
"""

from __future__ import annotations

import hashlib
import io
import pytest
import tarfile
from finetuning.container import validator_bridge as bridge
from pathlib import Path


def _manifest_for(files: dict[str, bytes]) -> dict:
    return {
        "upstream_commit": bridge.PINNED_COMMIT,
        "upstream_tree": bridge.PINNED_TREE,
        "upstream_license": {"declared": False},
        "entries": [
            {
                "path": rel,
                "snapshot_sha256": hashlib.sha256(data).hexdigest(),
                "snapshot_size": len(data),
                "vendored": True,
            }
            for rel, data in files.items()
        ],
    }


def _archive(tmp_path: Path, files: dict[str, bytes], *, top: str = "container-validator-pinned") -> Path:
    path = tmp_path / "archive.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for rel, data in files.items():
            info = tarfile.TarInfo(f"{top}/{rel}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def _serve(monkeypatch, archive: Path):
    """Stand in for urlopen with a local file; the network is never touched."""

    class _Response:
        def __init__(self, data: bytes):
            self._stream = io.BytesIO(data)

        def read(self, size):
            return self._stream.read(size)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    data = archive.read_bytes()
    monkeypatch.setattr(bridge.urllib.request, "urlopen", lambda url, timeout=None: _Response(data))


# --------------------------------------------------------------------------- absence


def test_absence_is_an_error_not_a_pass(monkeypatch):
    monkeypatch.delenv(bridge.ROOT_ENV, raising=False)
    with pytest.raises(bridge.ValidatorError, match="No official validator available"):
        bridge.validator_root()


def test_the_error_says_how_to_obtain_it(monkeypatch):
    monkeypatch.delenv(bridge.ROOT_ENV, raising=False)
    with pytest.raises(bridge.ValidatorError) as excinfo:
        bridge.validator_root()
    message = str(excinfo.value)
    assert bridge.PINNED_COMMIT in message and "acquire_validator" in message


def test_a_root_that_does_not_exist_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv(bridge.ROOT_ENV, str(tmp_path / "nowhere"))
    with pytest.raises(bridge.ValidatorError, match="does not exist"):
        bridge.validator_root()


# --------------------------------------------------------------------------- acquisition


def test_acquisition_fetches_the_pinned_commit_and_verifies_it(tmp_path, monkeypatch):
    files = {"container_validator/validate.py": b"print('validator')\n"}
    monkeypatch.setattr(bridge, "_metadata", lambda: _manifest_for(files))
    _serve(monkeypatch, _archive(tmp_path, files))
    report = bridge.acquire_validator(tmp_path / "dest")
    assert report["status"] == "VERIFIED"
    assert report["commit"] == bridge.PINNED_COMMIT
    assert bridge.PINNED_COMMIT in report["acquired_from"]
    assert (tmp_path / "dest" / "container_validator" / "validate.py").is_file()


def test_only_the_pinned_commit_is_ever_requested(tmp_path, monkeypatch):
    """Never main, HEAD, latest or a tag: the manifest describes exactly one commit."""
    files = {"a.py": b"x\n"}
    monkeypatch.setattr(bridge, "_metadata", lambda: _manifest_for(files))
    seen = {}

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    data = _archive(tmp_path, files).read_bytes()

    def _urlopen(url, timeout=None):
        seen["url"] = url
        return _Response(data)

    monkeypatch.setattr(bridge.urllib.request, "urlopen", _urlopen)
    bridge.acquire_validator(tmp_path / "dest")
    assert bridge.PINNED_COMMIT in seen["url"]
    for floating in ("main", "HEAD", "latest", "refs/tags"):
        assert floating not in seen["url"]


def test_a_tampered_member_is_refused_and_nothing_is_installed(tmp_path, monkeypatch):
    files = {"a.py": b"correct\n"}
    monkeypatch.setattr(bridge, "_metadata", lambda: _manifest_for(files))
    _serve(monkeypatch, _archive(tmp_path, {"a.py": b"tampered\n"}))
    with pytest.raises(bridge.ValidatorError, match="pinned digest"):
        bridge.acquire_validator(tmp_path / "dest")
    assert not (tmp_path / "dest").exists(), "a failed acquisition must leave no half-tree behind"


def test_an_extra_member_is_refused(tmp_path, monkeypatch):
    files = {"a.py": b"x\n"}
    monkeypatch.setattr(bridge, "_metadata", lambda: _manifest_for(files))
    _serve(monkeypatch, _archive(tmp_path, {"a.py": b"x\n", "surprise.py": b"y\n"}))
    with pytest.raises(bridge.ValidatorError, match="does not match the manifest"):
        bridge.acquire_validator(tmp_path / "dest")


def test_an_oversized_archive_is_refused(tmp_path, monkeypatch):
    files = {"a.py": b"x\n"}
    monkeypatch.setattr(bridge, "_metadata", lambda: _manifest_for(files))
    _serve(monkeypatch, _archive(tmp_path, files))
    with pytest.raises(bridge.ValidatorError, match="exceeds"):
        bridge.acquire_validator(tmp_path / "dest", max_bytes=8)


def test_a_non_empty_destination_is_never_overwritten(tmp_path, monkeypatch):
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "mine.txt").write_text("do not clobber")
    with pytest.raises(bridge.ValidatorError, match="non-empty destination"):
        bridge.acquire_validator(dest)
    assert (dest / "mine.txt").read_text() == "do not clobber"


def test_a_download_failure_is_reported_as_a_validator_error(tmp_path, monkeypatch):
    def _boom(url, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr(bridge.urllib.request, "urlopen", _boom)
    with pytest.raises(bridge.ValidatorError, match="Could not download"):
        bridge.acquire_validator(tmp_path / "dest")


# --------------------------------------------------------------------------- archive safety
#
# The archive comes from a project that publishes no licence and no signature. An unsafe member is
# a reason to stop, not something to sanitise quietly.


@pytest.mark.parametrize("name", ["../escape.py", "/absolute.py", "nested/../../escape.py"])
def test_path_traversal_and_absolute_members_are_refused(tmp_path, name):
    path = tmp_path / "evil.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo(name)
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with tarfile.open(path, "r:gz") as archive:
        with pytest.raises(bridge.ValidatorError, match="refusing"):
            bridge._safe_extract(archive, tmp_path / "out")


def test_symlink_members_are_refused(tmp_path):
    path = tmp_path / "link.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)
    with tarfile.open(path, "r:gz") as archive:
        with pytest.raises(bridge.ValidatorError, match="link member"):
            bridge._safe_extract(archive, tmp_path / "out")


def test_hard_link_members_are_refused(tmp_path):
    path = tmp_path / "hard.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo("a")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
        link = tarfile.TarInfo("b")
        link.type = tarfile.LNKTYPE
        link.linkname = "a"
        tar.addfile(link)
    with tarfile.open(path, "r:gz") as archive:
        with pytest.raises(bridge.ValidatorError, match="link member"):
            bridge._safe_extract(archive, tmp_path / "out")


def test_device_and_fifo_members_are_refused(tmp_path):
    path = tmp_path / "dev.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo("fifo")
        info.type = tarfile.FIFOTYPE
        tar.addfile(info)
    with tarfile.open(path, "r:gz") as archive:
        with pytest.raises(bridge.ValidatorError, match="special member"):
            bridge._safe_extract(archive, tmp_path / "out")


# --------------------------------------------------------------------------- no redistribution


def test_this_repository_contains_no_upstream_validator_source():
    repo = Path(__file__).resolve().parents[1]
    assert not (repo / "third_party" / "fomo26_container_validator").exists()
    assert not list(repo.glob("**/container_validator/validate.py"))


def test_the_manifest_records_that_upstream_declares_no_licence():
    metadata = bridge._metadata()
    assert metadata["upstream_license"]["declared"] is False
    assert metadata["upstream_url"] == bridge.UPSTREAM_URL
    assert metadata["upstream_commit"] == bridge.PINNED_COMMIT
    assert metadata["member_count"] == 84
