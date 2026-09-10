"""The canonical manifest as the source of truth for which sessions exist.

The FreeSurfer path discovers sessions with an ``os.walk`` of the cleaned tree. That is fine for
exploring a tree, but it cannot define a *corpus*: the walk would pick up every QA-rejected and
de-duplicated file the curation stage deliberately dropped, so the derivative's sample set would
not be the canonical 33,336 and a P0-vs-P1 comparison would silently compare different data.

This module therefore builds sessions from ``canonical_manifest_v2.tsv`` — the same file the P0
conversion is hash-contracted to (``78ce45fd…``) — so every canonical ``sample_id`` maps to
exactly one row of the derivative, and the two corpora are the same images by construction.

Identity is not taken on trust. :func:`identity_matches_path` re-derives
``(dataset, participant_id, session_id)`` from the BIDS components of ``cleaned_path`` and
refuses the row when they disagree, which is the same check the P0 converter already performs at
``run_upstream_convert.output_path``. A manifest row whose declared session does not match the
file it points at cannot be used to decide what shares an acquisition session.
"""

import csv
import hashlib
import logging
import os
from asparagus_preprocessing.session_coreg.discovery import Session
from pathlib import PurePosixPath
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Columns this module needs. ``canonical_manifest_v2.tsv`` carries 18; these are the load-bearing
#: ones, and a manifest missing any of them is rejected rather than silently half-read.
REQUIRED_FIELDS = (
    "sample_id",
    "dataset",
    "participant_id",
    "session_id",
    "cleaned_path",
    "modality_canonical",
)
OPTIONAL_FIELDS = ("session_n_scans", "shape", "pixdim", "orientation", "role", "derivation_type")


class CanonicalManifestError(ValueError):
    """Raised when the canonical manifest is missing, malformed, or not the contracted one."""


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    """SHA256 of a file, streamed. Mirrors ``upstream_manifest.sha256_of``."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def read_canonical_manifest(path: str, expected_sha256: Optional[str] = None) -> List[dict]:
    """Read the canonical manifest, optionally failing closed on a SHA mismatch.

    A different manifest is a different corpus; every derivative built from it would be
    provenance-linked to the wrong sample set, so the mismatch is fatal rather than a warning.
    """
    if not os.path.exists(path):
        raise CanonicalManifestError(f"canonical manifest not found: {path}")
    if expected_sha256:
        actual = sha256_of(path)
        if actual != expected_sha256:
            raise CanonicalManifestError(
                f"[FAIL-CLOSED] canonical manifest SHA256 mismatch.\n  expected {expected_sha256}\n  actual   {actual}\n"
                "This is not the contracted canonical corpus; refusing to build a derivative from it."
            )
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [f for f in REQUIRED_FIELDS if f not in (reader.fieldnames or [])]
        if missing:
            raise CanonicalManifestError(f"canonical manifest {path} is missing required columns: {missing}")
        rows = list(reader)
    if not rows:
        raise CanonicalManifestError(f"canonical manifest {path} has no rows")
    logger.info("Canonical manifest: %d samples from %s", len(rows), path)
    return rows


def session_key(row: dict) -> str:
    """``dataset/participant_id/session_id`` — the identity triple, as one stable path-like key."""
    return "/".join((row.get("dataset", ""), row.get("participant_id", ""), row.get("session_id", "")))


def identity_matches_path(row: dict) -> Tuple[bool, str]:
    """Check the manifest identity against the BIDS components of ``cleaned_path``.

    Returns ``(ok, reason)``. This is the same contract ``run_upstream_convert.output_path``
    enforces when writing P0 tensors, applied here *before* anything is grouped into a session.
    """
    dataset = str(row.get("dataset", "")).strip()
    participant = str(row.get("participant_id", "")).strip()
    session = str(row.get("session_id", "")).strip()
    source = str(row.get("cleaned_path", "")).strip()

    if not dataset or not participant or not session:
        return False, "empty dataset/participant_id/session_id"
    if not source:
        return False, "empty cleaned_path"

    dataset_parts = PurePosixPath(dataset).parts
    if PurePosixPath(dataset).is_absolute() or not dataset_parts or ".." in dataset_parts:
        return False, f"unsafe dataset component {dataset!r}"

    parts = PurePosixPath(source).parts
    start = next(
        (i for i in range(len(parts) - len(dataset_parts) + 1) if parts[i : i + len(dataset_parts)] == dataset_parts),
        None,
    )
    if start is None:
        return False, f"cleaned_path does not contain dataset component {dataset!r}"
    rel = parts[start:]
    if len(rel) < len(dataset_parts) + 3:
        return False, "cleaned_path is too shallow for dataset/participant/session/file"
    if rel[len(dataset_parts)] != participant:
        return False, f"path participant {rel[len(dataset_parts)]!r} != manifest {participant!r}"
    if rel[len(dataset_parts) + 1] != session:
        return False, f"path session {rel[len(dataset_parts) + 1]!r} != manifest {session!r}"
    return True, ""


def session_dir_of(row: dict) -> str:
    """Directory holding one session's scans: ``<cleaned_root>/<dataset>/<sub>/<ses>``."""
    source = str(row.get("cleaned_path", ""))
    dataset_parts = PurePosixPath(str(row.get("dataset", ""))).parts
    parts = PurePosixPath(source).parts
    start = next(
        (i for i in range(len(parts) - len(dataset_parts) + 1) if parts[i : i + len(dataset_parts)] == dataset_parts),
        None,
    )
    if start is None:
        return os.path.dirname(os.path.dirname(source))
    depth = start + len(dataset_parts) + 2  # dataset… / sub / ses
    return str(PurePosixPath(*parts[:depth]))


def group_by_session(rows: List[dict]) -> Dict[str, List[dict]]:
    """Group canonical rows by the identity triple, deterministically ordered."""
    groups: Dict[str, List[dict]] = {}
    for row in rows:
        groups.setdefault(session_key(row), []).append(row)
    for key in groups:
        groups[key].sort(key=lambda r: (r.get("cleaned_path", ""), r.get("sample_id", "")))
    return groups


def sessions_from_canonical(rows: List[dict], allowed_keys: Optional[set] = None) -> List[Session]:
    """Build :class:`Session` objects from canonical rows, in deterministic key order.

    ``allowed_keys`` restricts the result (used to drop sessions the identity audit rejected)
    without changing how any surviving session is constructed.
    """
    sessions: List[Session] = []
    for key, group in sorted(group_by_session(rows).items()):
        if allowed_keys is not None and key not in allowed_keys:
            continue
        first = group[0]
        sessions.append(
            Session(
                key=key,
                session_dir=session_dir_of(first),
                dataset=str(first.get("dataset", "")),
                subject=str(first.get("participant_id", "")),
                session=str(first.get("session_id", "")),
                image_paths=[str(r.get("cleaned_path", "")) for r in group],
                samples=list(group),
            )
        )
    return sessions


def infer_input_root(rows: List[dict]) -> str:
    """The cleaned corpus root implied by the manifest.

    Derived by stripping each row's ``dataset`` component off its ``cleaned_path``, **not** by
    taking a common ancestor: a chunk that happens to contain a single dataset would give a
    common ancestor one or more levels too deep, and every output path would then be mirrored
    into the wrong place. All rows must agree, or the manifest spans two corpora and is refused.
    """
    roots = set()
    for row in rows:
        source = str(row.get("cleaned_path", ""))
        dataset_parts = PurePosixPath(str(row.get("dataset", ""))).parts
        if not source or not dataset_parts:
            continue
        parts = PurePosixPath(source).parts
        start = next(
            (i for i in range(len(parts) - len(dataset_parts) + 1) if parts[i : i + len(dataset_parts)] == dataset_parts),
            None,
        )
        if start is None:
            continue
        roots.add(str(PurePosixPath(*parts[:start])) if start else os.sep)
    if not roots:
        raise CanonicalManifestError("cannot infer an input root: no cleaned_path contained its dataset component")
    if len(roots) > 1:
        raise CanonicalManifestError(f"canonical manifest spans several corpus roots, refusing: {sorted(roots)}")
    return roots.pop()
