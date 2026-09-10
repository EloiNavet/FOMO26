"""Canonical MRI modality classification and reference priority.

Modality is inferred from the BIDS suffix of a file name and normalised to a
small canonical vocabulary. The priority order (used by the anisotropy-aware
reference policy) is::

    T1w > T1ce > T2w > FLAIR > PD > ADC > DWI > other
"""

import os

# Canonical label -> priority rank (lower = preferred as reference).
PRIORITY = {
    "T1w": 0,
    "T1ce": 1,
    "T2w": 2,
    "FLAIR": 3,
    "PD": 4,
    "ADC": 5,
    "DWI": 6,
    "other": 7,
}
OTHER_RANK = PRIORITY["other"]

# Substring (lowercase) -> canonical label. Order matters: the most specific
# tokens (t1ce/t1gd, dwi shells) must be tested before the generic ones (t1, t2).
_RULES = [
    ("flair", "FLAIR"),
    ("t1ce", "T1ce"),
    ("t1gd", "T1ce"),
    ("t1c", "T1ce"),
    ("ce_t1", "T1ce"),
    ("t1w", "T1w"),
    ("t1", "T1w"),
    ("mprage", "T1w"),
    ("mp2rage", "T1w"),
    ("unit1", "T1w"),
    ("t2starw", "other"),
    ("t2star", "other"),
    ("swi", "other"),
    ("t2w", "T2w"),
    ("t2", "T2w"),
    ("pdw", "PD"),
    ("pd", "PD"),
    ("adc", "ADC"),
    ("dwi_trace", "DWI"),
    ("dwi_b", "DWI"),
    ("trace", "DWI"),
    ("bval", "DWI"),
    ("dwi", "DWI"),
    ("dti", "DWI"),
    ("gre", "other"),
]


def suffix(path: str) -> str:
    """Return the BIDS-style suffix token, e.g. ``..._run-1_T1w.nii.gz`` -> ``T1w``."""
    base = os.path.basename(path)
    for ext in (".nii.gz", ".nii", ".mgz"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            break
    return (base.split("_")[-1] if "_" in base else base) or "unknown"


def classify(path: str) -> str:
    """Canonical modality label for a scan path (falls back to ``other``)."""
    stem = os.path.basename(path).lower()
    for token, label in _RULES:
        if token in stem:
            return label
    return "other"


def priority_rank(path: str) -> int:
    """Reference-priority rank for a scan (lower is preferred)."""
    return PRIORITY.get(classify(path), OTHER_RANK)


# --------------------------------------------------------------------------- #
# Canonical corpus vocabulary
# --------------------------------------------------------------------------- #
# ``canonical_manifest_v2.tsv`` labels contrasts with its own vocabulary (``T1c`` rather than
# ``T1ce``; the diffusion channels split into ``DWI_B0``/``DWI_B1000``/``DWI_TRACE``). When a run
# is manifest-driven that curated label is authoritative and must be preferred over guessing the
# contrast from a filename — ``classify()`` matches substrings, so a subject id containing "t1"
# can mislabel a T2w file.
CANONICAL_PRIORITY = {
    "T1w": 0,
    "T1c": 1,
    "T2w": 2,
    "FLAIR": 3,
    "PD": 4,
    "ADC": 5,
    "DWI_B0": 6,
    "DWI_B1000": 7,
    "DWI_TRACE": 8,
    "other": 9,
}
CANONICAL_OTHER_RANK = CANONICAL_PRIORITY["other"]

#: Legacy label -> canonical label, so both vocabularies rank consistently.
_LEGACY_TO_CANONICAL = {"T1ce": "T1c", "DWI": "DWI_B1000"}


def canonicalise(label: str) -> str:
    """Normalise a modality label onto the canonical corpus vocabulary."""
    text = (label or "").strip()
    if not text:
        return "other"
    if text in CANONICAL_PRIORITY:
        return text
    mapped = _LEGACY_TO_CANONICAL.get(text)
    if mapped:
        return mapped
    for known in CANONICAL_PRIORITY:
        if text.lower() == known.lower():
            return known
    return "other"


def priority_rank_for(label: str) -> int:
    """Reference-priority rank for an explicit modality label (lower is preferred)."""
    return CANONICAL_PRIORITY.get(canonicalise(label), CANONICAL_OTHER_RANK)
