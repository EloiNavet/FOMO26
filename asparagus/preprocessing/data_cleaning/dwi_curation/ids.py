"""Robust parsing of subject/session/run identifiers and b-values from paths.

These helpers must tolerate the many FOMO300K naming variants: ``sub-001`` vs
``sub001``, sessions with/without ``ses-*``, ``run-*`` present or absent, b-values
encoded in the *new* filename (``dwi_bval1200``) or only in the *old* filename
(``..._bval_1200_...``, ``rec-ADC``), etc.
"""

from __future__ import annotations

import re
from pathlib import Path

_SUFFIX_RE = re.compile(r"\.nii(\.gz)?$", re.IGNORECASE)
_RUN_RE = re.compile(r"(?:^|[_/-])run-?(\d+)", re.IGNORECASE)
# b-value tokens, most-specific first. Ordered so ``bval1200`` wins over a bare
# ``b1200`` fragment, and ``bval_1200`` (old FOMO300K style) is caught too.
_BVAL_PATTERNS = (
    re.compile(r"bval[_-]?(\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"(?:^|[_-])b[-_]?(\d{2,5})(?:$|[_.])", re.IGNORECASE),
)


def _basename_no_ext(path: str | None) -> str:
    if not path:
        return ""
    name = Path(str(path)).name
    return _SUFFIX_RE.sub("", name)


def strip_nifti_ext(path: str | None) -> str:
    """Return the filename (no directories) without a ``.nii`` / ``.nii.gz`` suffix."""
    return _basename_no_ext(path)


def normalize_subject_id(value: str | None) -> str:
    """Canonical, comparison-friendly subject key.

    ``sub-001``, ``sub001``, ``Sub_001`` all normalise to ``sub001``. Used as a join
    key so metadata lookups survive inconsistent formatting across tables.
    """
    if value is None:
        return ""
    text = re.sub(r"[^0-9a-zA-Z]", "", str(value)).lower()
    return text


def normalize_session_id(value: str | None) -> str:
    """Canonical session key; empty string when no session is present."""
    if value is None:
        return ""
    return re.sub(r"[^0-9a-zA-Z]", "", str(value)).lower()


def parse_run_id(*paths: str | None) -> str | None:
    """Return a normalised ``run-<n>`` label from the first path that carries one."""
    for path in paths:
        if not path:
            continue
        match = _RUN_RE.search(str(path))
        if match:
            return f"run-{int(match.group(1))}"
    return None


def parse_b_value(*paths: str | None) -> float | None:
    """Parse a diffusion b-value from any of *paths* (new filename tried first).

    Only ``bval*`` tokens are trusted from the *new* filename; the looser ``b<NNN>``
    fallback is applied to catch old-filename encodings. Returns ``None`` when no
    b-value token is present (e.g. a generic ``_dwi.nii.gz`` or an ADC map).
    """
    for path in paths:
        stem = _basename_no_ext(path)
        if not stem:
            continue
        for pattern in _BVAL_PATTERNS:
            match = pattern.search(stem)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
    return None


def has_token(text: str | None, *tokens: str) -> bool:
    """Case-insensitive substring test used for adc/trace/apparent name evidence."""
    if not text:
        return False
    lowered = str(text).lower()
    return any(token in lowered for token in tokens)


def is_adc_name(*paths: str | None) -> bool:
    """True if any path filename encodes an ADC map (``adc`` / ``apparent``)."""
    return any(has_token(strip_nifti_ext(p), "adc", "apparent") for p in paths)


def is_trace_name(*paths: str | None) -> bool:
    """True if any path filename encodes a diffusion *trace* image."""
    return any(has_token(strip_nifti_ext(p), "trace") for p in paths)
