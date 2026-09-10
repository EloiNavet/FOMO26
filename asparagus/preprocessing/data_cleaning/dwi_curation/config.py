"""Taxonomy constants, b-value bands, and thresholds for diffusion curation.

Single source of truth so classification rules live here rather than scattered
across regexes. Imported by :mod:`.classify`, :mod:`.select`, the audit/curate CLIs,
and (for the shared coarse view) the dataset overview figure.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Scan-level diffusion classes (Decision level A)
# ---------------------------------------------------------------------------
ADC = "ADC"
ADC_LIKELY = "ADC_LIKELY"
DWI_B1000 = "DWI_B1000"
DWI_NEAR_B1000 = "DWI_NEAR_B1000"
DWI_TRACE = "DWI_TRACE"
DWI_B0 = "DWI_B0"
DWI_LOW_B = "DWI_LOW_B"
DWI_HIGH_B = "DWI_HIGH_B"
DWI_EXTREME_HIGH_B = "DWI_EXTREME_HIGH_B"
DWI_DERIVED_UNKNOWN = "DWI_DERIVED_UNKNOWN"
DWI_EXCLUDED = "DWI_EXCLUDED"

DIFFUSION_CLASSES = (
    ADC,
    ADC_LIKELY,
    DWI_B1000,
    DWI_NEAR_B1000,
    DWI_TRACE,
    DWI_B0,
    DWI_LOW_B,
    DWI_HIGH_B,
    DWI_EXTREME_HIGH_B,
    DWI_DERIVED_UNKNOWN,
    DWI_EXCLUDED,
)

# ---------------------------------------------------------------------------
# Final curated diffusion output modalities (Decision level B)
# ---------------------------------------------------------------------------
OUT_ADC = "ADC"
OUT_DWI_B1000 = "DWI_B1000"
OUT_DWI_TRACE = "DWI_TRACE"
OUT_DWI_B0 = "DWI_B0"

# ---------------------------------------------------------------------------
# b-value bands (units: s/mm^2)
# ---------------------------------------------------------------------------
B0_MAX = 5.0  # b <= 5           -> DWI_B0
LOW_B_MAX = 800.0  # 5 < b < 800      -> DWI_LOW_B
NEAR_B1000_MIN = 800.0  # [800, 1200]      -> near-b1000 band
NEAR_B1000_MAX = 1200.0
EXACT_B1000_MIN = 990.0  # [990, 1010]     -> exact b1000
EXACT_B1000_MAX = 1010.0
HIGH_B_MAX = 4000.0  # 1200 < b < 4000  -> DWI_HIGH_B ; b >= 4000 -> DWI_EXTREME_HIGH_B

# Preferred near-b1000 shells for direct use / synthesis (nearest-to-1000 wins,
# with the explicit ordering exact > 900/1100 > 800/1200 encoded by distance).
PREFERRED_B1000_SHELLS = (1000.0, 900.0, 1100.0, 800.0, 1200.0)
CLINICAL_SHELL_MIN = 800.0
CLINICAL_SHELL_MAX = 1200.0

# ---------------------------------------------------------------------------
# Structural / default pretraining modalities (FOMO26-aligned)
# ---------------------------------------------------------------------------
# T2star and T2s are aliases of the same contrast.
STRUCTURAL_DEFAULT = ("T1w", "T2w", "FLAIR", "T2star", "SWI")
# GRE and T1c are NOT default for FOMO26-focused curation unless a flag enables them.
STRUCTURAL_OPTIONAL = ("GRE", "T1c")
# Full default pretraining vocabulary (structural + curated diffusion).
DEFAULT_PRETRAIN_MODALITIES = STRUCTURAL_DEFAULT + (OUT_DWI_B1000, OUT_ADC)

# Canonical structural suffix normalisation (lower-cased suffix -> canonical name).
STRUCTURAL_SUFFIX_CANON = {
    "t1w": "T1w",
    "t2w": "T2w",
    "flair": "FLAIR",
    "t2star": "T2star",
    "t2starw": "T2star",
    "t2s": "T2star",
    "swi": "SWI",
    "gre": "GRE",
    "t1c": "T1c",
    "t1ce": "T1c",
    "t1gd": "T1c",
}

# ---------------------------------------------------------------------------
# QC / derivation
# ---------------------------------------------------------------------------
MIN_VALID_VOXEL_FRACTION = 0.5  # below this the derived ADC/b1000 is flagged low-QC
ADC_SAVE_SCALE = 1e6  # ADC saved scaled by 1e6 (viewer-friendly ~600-1200)
RATIO_CLIP_MIN = 1e-10  # Sb/S0 clipped to [RATIO_CLIP_MIN, 1.0]
RATIO_CLIP_MAX = 1.0

# ADC-vs-DWI intensity heuristic (only used to promote a truly-generic ``_dwi`` to
# ADC_LIKELY when no name/metadata evidence exists). ADC maps have bright CSF but a
# *moderate* high-intensity tail (CSF ~2-3x parenchyma). A high-b DWI has a much
# heavier tail (dark parenchyma, ratio >> 5), so an UPPER bound is essential to avoid
# misclassifying high-b diffusion as ADC. Empirically every FOMO300K generic _dwi
# with p95/p50 > ~5 is a high-b research scan, not an ADC map.
ADC_LIKE_P95_OVER_P50_MIN = 1.5
ADC_LIKE_P95_OVER_P50_MAX = 4.5
ADC_LIKE_P99_OVER_P50_MIN = 1.8
ADC_LIKE_P99_OVER_P50_MAX = 6.5


def band_for_bvalue(b: float | None) -> str | None:
    """Map a numeric b-value to its raw diffusion band class (ignores name/metadata).

    Returns ``None`` when *b* is unknown so callers fall back to name/metadata/
    intensity evidence.
    """
    if b is None:
        return None
    if b <= B0_MAX:
        return DWI_B0
    if b < LOW_B_MAX:
        return DWI_LOW_B
    if EXACT_B1000_MIN <= b <= EXACT_B1000_MAX:
        return DWI_B1000
    if NEAR_B1000_MIN <= b <= NEAR_B1000_MAX:
        return DWI_NEAR_B1000
    if b < HIGH_B_MAX:
        return DWI_HIGH_B
    return DWI_EXTREME_HIGH_B


def is_clinical_shell(b: float | None) -> bool:
    """True if *b* lies in the clinical near-b1000 band usable for ADC/b1000 synth."""
    return b is not None and CLINICAL_SHELL_MIN <= b <= CLINICAL_SHELL_MAX


def b1000_preference_rank(b: float) -> tuple[int, float]:
    """Sort key selecting the best near-b1000 shell: exact 1000 > 900/1100 > 800/1200.

    Lower is better. Primary key is distance to 1000; ties broken deterministically.
    """
    return (round(abs(b - 1000.0)), b)
