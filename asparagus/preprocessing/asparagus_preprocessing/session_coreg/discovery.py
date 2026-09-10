"""Discover MRI sessions in a BIDS-style dataset tree.

A *session* is one acquisition of a subject that groups several co-acquired
modalities (``anat/``, ``dwi/`` ...). In the FOMO300K cleaned tree this is a
``PT***/sub-*/ses-*`` folder; some sources omit ``ses-*``. Discovery is
therefore keyed on the nearest ``ses-*`` ancestor, falling back to ``sub-*`` and
then the file's parent directory, so it is robust to both layouts.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)

_SES_RE = re.compile(r"(?:^|[/\\])(ses-[A-Za-z0-9]+)(?:[/\\]|$)")
_SUB_RE = re.compile(r"(?:^|[/\\])(sub-[A-Za-z0-9]+)(?:[/\\]|$)")
# Sidecars that live next to images but must never be treated as scans.
_SIDECAR_SUFFIXES = (".json", ".bval", ".bvec", ".tsv", ".txt", ".pkl", ".pk")


@dataclass
class Session:
    """A group of image files that share one subject/session."""

    key: str  # stable identifier, e.g. "PT029_OASIS2/sub-001/ses-01"
    session_dir: str  # deepest common directory of the grouped files
    dataset: str  # top-level dataset folder name, e.g. "PT029_OASIS2"
    subject: str
    session: str
    image_paths: List[str] = field(default_factory=list)
    #: Canonical-manifest rows backing ``image_paths``, in the same order, when the session came
    #: from the canonical manifest rather than a filesystem walk. Empty for filesystem discovery,
    #: which is what keeps the FreeSurfer path unaffected.
    samples: List[dict] = field(default_factory=list)
    #: False when the identity audit could not establish that these scans share one acquisition
    #: visit. Such a session is still processed — as passthrough — so the corpus keeps its
    #: samples; only permission to co-register is withdrawn.
    coreg_allowed: bool = True
    coreg_block_reason: str = ""

    def sample_for(self, path: str) -> dict:
        """Canonical row backing ``path``, or ``{}`` when this session is filesystem-derived."""
        for candidate, row in zip(self.image_paths, self.samples):
            if candidate == path:
                return row
        return {}

    def relpath(self, path: str) -> str:
        """Path of ``path`` relative to this session's directory."""
        return os.path.relpath(path, self.session_dir)


def _is_image(name: str, extensions: List[str]) -> bool:
    lower = name.lower()
    if lower.endswith(_SIDECAR_SUFFIXES):
        return False
    return any(lower.endswith(ext) for ext in extensions)


def _session_dir_for(path: str, root: str) -> str:
    """Nearest ``ses-*`` dir; else nearest ``sub-*`` dir; else the file's parent."""
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    for depth in range(len(parts) - 1, 0, -1):
        segment = parts[depth - 1]
        if segment.startswith("ses-"):
            return os.path.join(root, *parts[:depth])
    for depth in range(len(parts) - 1, 0, -1):
        segment = parts[depth - 1]
        if segment.startswith("sub-"):
            return os.path.join(root, *parts[:depth])
    return os.path.dirname(path)


def _extract_ids(session_dir: str) -> tuple:
    sub = _SUB_RE.search(session_dir + os.sep)
    ses = _SES_RE.search(session_dir + os.sep)
    subject = sub.group(1) if sub else os.path.basename(os.path.dirname(session_dir))
    session = ses.group(1) if ses else "ses-01"
    return subject, session


def find_sessions(root: str, extensions: List[str]) -> List[Session]:
    """Group every image under ``root`` into :class:`Session` objects.

    Sessions and their image lists are returned in a deterministic (sorted)
    order so multiprocessing shards and QC sampling are reproducible.
    """
    root = os.path.abspath(root)
    groups: Dict[str, List[str]] = {}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not _is_image(name, extensions):
                continue
            path = os.path.join(dirpath, name)
            session_dir = _session_dir_for(path, root)
            groups.setdefault(session_dir, []).append(path)

    sessions: List[Session] = []
    for session_dir in sorted(groups):
        image_paths = sorted(groups[session_dir])
        dataset = os.path.relpath(session_dir, root).split(os.sep)[0]
        subject, session = _extract_ids(session_dir)
        key = os.path.relpath(session_dir, root)
        sessions.append(
            Session(
                key=key,
                session_dir=session_dir,
                dataset=dataset,
                subject=subject,
                session=session,
                image_paths=image_paths,
            )
        )
    logger.info("Discovered %d sessions under %s", len(sessions), root)
    return sessions
