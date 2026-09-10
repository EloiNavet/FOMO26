"""Wave 0A: characterisation of the FOMO26 submission-packaging contract.

Audit finding TEST-006 listed `finetuning/export_fomo26_submission.py` as having zero test
references. It is the last step before a challenge submission: it validates an official-schema
mapping and packages predictions into a zip. A silent failure here (wrong archive member name,
path traversal, a missing prediction copied as an empty entry) is discovered by the challenge
server, not by us.

These tests are pure filesystem work in `tmp_path`: no Docker daemon, no GPU, no network, no
challenge dataset, no credentials, and no official submission is produced.

Priorities covered, per the Wave 0A brief: submission member names / row identity / ordering (5),
fail-closed behaviour for missing predictions (6), and rejection of unsafe or inconsistent
entries (7).
"""

from __future__ import annotations

import json
import pytest
import zipfile
from finetuning import export_fomo26_submission as exporter
from pathlib import Path


def _schema(tmp_path: Path, entries) -> Path:
    path = tmp_path / "schema.json"
    path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    return path


def _prediction_file(tmp_path: Path, name: str = "predictions.json") -> Path:
    path = tmp_path / "src" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"case_0001": 0.5, "case_0002": 0.25}), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------------
# Schema validation must fail closed
# --------------------------------------------------------------------------------------------


def test_missing_schema_file_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="Submission schema not found"):
        exporter.load_schema(tmp_path / "absent.json")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"entries": []},
        {"entries": "not-a-list"},
    ],
    ids=["no-entries-key", "empty-entries", "entries-not-a-list"],
)
def test_schema_without_usable_entries_is_rejected(tmp_path, payload):
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty 'entries' list"):
        exporter.load_schema(path)


@pytest.mark.parametrize("missing", ["source", "target", "kind"])
def test_schema_entry_missing_a_required_key_is_rejected(tmp_path, missing):
    entry = {"source": "/tmp/x.json", "target": "Task_3/predictions.json", "kind": "file"}
    entry.pop(missing)
    with pytest.raises(ValueError, match=f"missing '{missing}'"):
        exporter.load_schema(_schema(tmp_path, [entry]))


def test_schema_entry_with_unsupported_kind_is_rejected(tmp_path):
    entry = {"source": "/tmp/x.json", "target": "Task_3/p.json", "kind": "symlink"}
    with pytest.raises(ValueError, match="unsupported kind"):
        exporter.load_schema(_schema(tmp_path, [entry]))


@pytest.mark.parametrize(
    "target",
    ["/absolute/Task_3/p.json", "../escape/p.json", "Task_3/../../escape.json"],
    ids=["absolute", "leading-parent", "embedded-parent"],
)
def test_schema_entry_with_unsafe_target_is_rejected(tmp_path, target):
    """Archive members must stay inside the submission root; no absolute paths, no traversal."""
    entry = {"source": "/tmp/x.json", "target": target, "kind": "file"}
    with pytest.raises(ValueError, match="unsafe target path"):
        exporter.load_schema(_schema(tmp_path, [entry]))


def test_valid_schema_round_trips_entries_in_order(tmp_path):
    entries = [
        {"source": "/tmp/a.json", "target": "Task_3/predictions.json", "kind": "file"},
        {"source": "/tmp/masks", "target": "Task_2", "kind": "directory"},
    ]
    loaded = exporter.load_schema(_schema(tmp_path, entries))
    assert loaded == entries, "load_schema must preserve entry identity and order"


# --------------------------------------------------------------------------------------------
# Copying must fail closed on missing predictions
# --------------------------------------------------------------------------------------------


def test_missing_source_file_fails_closed(tmp_path):
    entry = {"source": str(tmp_path / "nope.json"), "target": "Task_3/p.json", "kind": "file"}
    with pytest.raises(FileNotFoundError, match="Missing source file"):
        exporter.copy_entry(entry, tmp_path / "staging")


def test_missing_source_directory_fails_closed(tmp_path):
    entry = {"source": str(tmp_path / "nope"), "target": "Task_2", "kind": "directory"}
    with pytest.raises(FileNotFoundError, match="Missing source directory"):
        exporter.copy_entry(entry, tmp_path / "staging")


def test_kind_file_pointing_at_a_directory_fails_closed(tmp_path):
    """A prediction directory declared as `kind: file` must not be packaged as an empty member."""
    src = tmp_path / "src_dir"
    src.mkdir()
    entry = {"source": str(src), "target": "Task_3/p.json", "kind": "file"}
    with pytest.raises(FileNotFoundError, match="Missing source file"):
        exporter.copy_entry(entry, tmp_path / "staging")


def test_copy_entry_places_content_at_the_declared_target(tmp_path):
    source = _prediction_file(tmp_path)
    staging = tmp_path / "staging"
    exporter.copy_entry({"source": str(source), "target": "Task_3/predictions.json", "kind": "file"}, staging)
    copied = staging / "Task_3" / "predictions.json"
    assert copied.is_file()
    assert json.loads(copied.read_text()) == json.loads(source.read_text())


def test_copy_entry_directory_replaces_a_stale_target(tmp_path):
    """Re-packaging must not merge a previous run's masks into the new submission."""
    source = tmp_path / "masks"
    source.mkdir()
    (source / "case_0001.nii.gz").write_bytes(b"new")
    staging = tmp_path / "staging"
    stale = staging / "Task_2"
    stale.mkdir(parents=True)
    (stale / "case_9999.nii.gz").write_bytes(b"stale")

    exporter.copy_entry({"source": str(source), "target": "Task_2", "kind": "directory"}, staging)

    members = sorted(p.name for p in (staging / "Task_2").iterdir())
    assert members == ["case_0001.nii.gz"], "stale members survived re-packaging"


# --------------------------------------------------------------------------------------------
# Archive identity: member names and ordering
# --------------------------------------------------------------------------------------------


def test_zip_members_are_relative_sorted_and_complete(tmp_path):
    staging = tmp_path / "staging"
    (staging / "Task_2").mkdir(parents=True)
    (staging / "Task_3").mkdir(parents=True)
    (staging / "Task_3" / "predictions.json").write_text("{}", encoding="utf-8")
    (staging / "Task_2" / "case_0002.nii.gz").write_bytes(b"b")
    (staging / "Task_2" / "case_0001.nii.gz").write_bytes(b"a")

    output = tmp_path / "submission.zip"
    exporter.zip_directory(staging, output)

    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()

    assert names == sorted(names), "archive members must be written in sorted order"
    assert names == [
        "Task_2/case_0001.nii.gz",
        "Task_2/case_0002.nii.gz",
        "Task_3/predictions.json",
    ]
    assert not any(Path(n).is_absolute() for n in names), "archive leaked absolute paths"


def test_zip_directory_is_deterministic_across_runs(tmp_path):
    staging = tmp_path / "staging"
    (staging / "Task_1").mkdir(parents=True)
    for i in range(5):
        (staging / "Task_1" / f"case_{i:04d}.json").write_text(f'{{"v": {i}}}', encoding="utf-8")

    first, second = tmp_path / "a.zip", tmp_path / "b.zip"
    exporter.zip_directory(staging, first)
    exporter.zip_directory(staging, second)

    with zipfile.ZipFile(first) as a, zipfile.ZipFile(second) as b:
        assert a.namelist() == b.namelist()
        assert [a.read(n) for n in a.namelist()] == [b.read(n) for n in b.namelist()]


# --------------------------------------------------------------------------------------------
# The CLI, end to end, without producing an official submission
# --------------------------------------------------------------------------------------------


def test_cli_dry_run_validates_without_writing_an_archive(tmp_path, monkeypatch, capsys):
    source = _prediction_file(tmp_path)
    schema = _schema(tmp_path, [{"source": str(source), "target": "Task_3/predictions.json", "kind": "file"}])
    output = tmp_path / "submission.zip"
    monkeypatch.setattr(
        "sys.argv",
        ["export_fomo26_submission", "--schema", str(schema), "--output", str(output), "--dry-run"],
    )

    exporter.main()

    assert not output.exists(), "--dry-run must not write the submission archive"
    assert "Task_3/predictions.json" in capsys.readouterr().out


def test_cli_writes_an_archive_matching_the_schema(tmp_path, monkeypatch):
    predictions = _prediction_file(tmp_path)
    masks = tmp_path / "src" / "masks"
    masks.mkdir(parents=True)
    (masks / "case_0001.nii.gz").write_bytes(b"mask")

    schema = _schema(
        tmp_path,
        [
            {"source": str(predictions), "target": "Task_3/predictions.json", "kind": "file"},
            {"source": str(masks), "target": "Task_2", "kind": "directory"},
        ],
    )
    output = tmp_path / "out" / "submission.zip"
    monkeypatch.setattr(
        "sys.argv",
        ["export_fomo26_submission", "--schema", str(schema), "--output", str(output)],
    )

    exporter.main()

    assert output.is_file()
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == ["Task_2/case_0001.nii.gz", "Task_3/predictions.json"]
        assert json.loads(archive.read("Task_3/predictions.json")) == json.loads(predictions.read_text())


def test_cli_fails_closed_when_a_declared_prediction_is_absent(tmp_path, monkeypatch):
    """The failure that matters: an incomplete submission must never be produced silently."""
    schema = _schema(
        tmp_path,
        [{"source": str(tmp_path / "never_written.json"), "target": "Task_3/p.json", "kind": "file"}],
    )
    output = tmp_path / "submission.zip"
    monkeypatch.setattr(
        "sys.argv",
        ["export_fomo26_submission", "--schema", str(schema), "--output", str(output)],
    )

    with pytest.raises(FileNotFoundError):
        exporter.main()
    assert not output.exists(), "a partial submission archive was left behind"
