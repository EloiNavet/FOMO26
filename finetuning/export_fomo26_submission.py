#!/usr/bin/env python3
"""Package FOMO26 predictions according to an explicit challenge schema."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

SCHEMA_HELP = """Expected schema:
{
  "entries": [
    {"source": "/path/to/predictions.json", "target": "Task_3/predictions.json", "kind": "file"},
    {"source": "/path/to/prediction_masks", "target": "Task_2", "kind": "directory"}
  ]
}
The official challenge may require different target names. Put those exact paths in target.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, epilog=SCHEMA_HELP)
    parser.add_argument("--schema", type=Path, required=True, help="Official-schema mapping JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Submission zip path.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print copy plan without writing the zip.")
    return parser.parse_args()


def load_schema(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Submission schema not found: {path}. "
            f"The official schema is not in this repo; provide it explicitly.\n{SCHEMA_HELP}"
        )
    with path.open(encoding="utf-8") as file:
        schema = json.load(file)
    entries = schema.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} must contain a non-empty 'entries' list.\n{SCHEMA_HELP}")
    for index, entry in enumerate(entries):
        for key in ("source", "target", "kind"):
            if key not in entry:
                raise ValueError(f"Schema entry {index} is missing {key!r}.")
        if entry["kind"] not in {"file", "directory"}:
            raise ValueError(f"Schema entry {index} has unsupported kind={entry['kind']!r}.")
        if Path(entry["target"]).is_absolute() or ".." in Path(entry["target"]).parts:
            raise ValueError(f"Schema entry {index} has unsafe target path {entry['target']!r}.")
    return entries


def copy_entry(entry: dict[str, str], staging_dir: Path) -> None:
    source = Path(entry["source"]).expanduser().resolve()
    target = staging_dir / entry["target"]
    if entry["kind"] == "file":
        if not source.is_file():
            raise FileNotFoundError(f"Missing source file {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    else:
        if not source.is_dir():
            raise FileNotFoundError(f"Missing source directory {source}")
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)


def zip_directory(staging_dir: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(staging_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(staging_dir))


def main() -> None:
    args = parse_args()
    entries = load_schema(args.schema)
    for entry in entries:
        print(f"{entry['kind']}: {entry['source']} -> {entry['target']}")
    if args.dry_run:
        return
    with tempfile.TemporaryDirectory(prefix="fomo26_submission_") as tmp:
        staging_dir = Path(tmp)
        for entry in entries:
            copy_entry(entry, staging_dir)
        zip_directory(staging_dir, args.output.expanduser().resolve())
    print(f"Submission archive written to {args.output}")


if __name__ == "__main__":
    main()
