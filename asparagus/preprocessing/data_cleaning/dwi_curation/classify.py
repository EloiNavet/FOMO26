"""Scan-level classification (Decision level A).

``classify_diffusion_scan`` fuses every available piece of evidence -- filename,
old (pre-clean) filename, parsed b-value, run id, NIfTI header/scaling, intensity
percentiles, and ``mri_info`` metadata (``SeriesDescription`` / ``ProtocolName``) --
into a single diffusion class with a confidence and a human-readable reason.

The evidence priority (highest first) is:
  1. Explicit ADC  (adc/apparent in a filename, or dADC/ADC in metadata)
  2. Trace provenance (``trace`` in a filename)  -> b1000 if near-1000, else trace
  3. Parsed b-value band
  4. Intensity heuristic for a truly-generic ``_dwi`` (ADC_LIKELY vs UNKNOWN)
"""

from __future__ import annotations

from . import config, ids
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScanEvidence:
    """All inputs available for classifying one diffusion scan.

    Only ``new_filename`` is strictly required; everything else is best-effort.
    ``intensity`` is a dict from :func:`nifti.compute_intensity_summary` and is only
    populated for ambiguous generic ``_dwi`` scans (targeted loading).
    """

    new_filename: str
    old_filename: str | None = None
    series_description: str | None = None
    protocol_name: str | None = None
    modality_folder: str | None = None  # "dwi" / "anat" from mapping.tsv
    intensity: dict[str, Any] | None = None
    header: dict[str, Any] | None = None


@dataclass
class ScanClassification:
    diffusion_class: str
    b_value: float | None
    is_trace: bool
    provenance_is_trace: bool
    is_adc_name: bool
    is_adc_metadata: bool
    confidence: str  # "high" | "medium" | "low"
    reason: str
    qc_flags: list[str] = field(default_factory=list)


_ADC_METADATA_TOKENS = ("adc", "apparent diffusion", "dadc")


def is_adc_metadata(series_description: str | None, protocol_name: str | None) -> bool:
    """True if ``SeriesDescription`` / ``ProtocolName`` indicate an ADC/dADC map."""
    return ids.has_token(series_description, *_ADC_METADATA_TOKENS) or ids.has_token(protocol_name, *_ADC_METADATA_TOKENS)


def _looks_like_adc_intensity(intensity: dict[str, Any] | None) -> bool:
    """ADC maps have bright CSF but a *moderate* tail: p95/p50 and p99/p50 sit in a
    bounded band. Below the band is a structural/dark map; above it is a high-b DWI.
    """
    if not intensity:
        return False
    p95r = intensity.get("p95_over_p50")
    p99r = intensity.get("p99_over_p50")
    try:
        return (
            p95r is not None
            and p99r is not None
            and config.ADC_LIKE_P95_OVER_P50_MIN <= p95r <= config.ADC_LIKE_P95_OVER_P50_MAX
            and config.ADC_LIKE_P99_OVER_P50_MIN <= p99r <= config.ADC_LIKE_P99_OVER_P50_MAX
        )
    except TypeError:
        return False


def classify_diffusion_scan(ev: ScanEvidence) -> ScanClassification:
    """Classify a single diffusion scan into a :data:`config.DIFFUSION_CLASSES` value."""
    paths = (ev.new_filename, ev.old_filename)
    b = ids.parse_b_value(*paths)
    adc_name = ids.is_adc_name(*paths)
    trace_name = ids.is_trace_name(*paths)
    adc_meta = is_adc_metadata(ev.series_description, ev.protocol_name)
    qc: list[str] = []

    # --- 1. Explicit ADC (name is strongest; metadata alone is medium confidence) ---
    if adc_name:
        return ScanClassification(
            config.ADC,
            b,
            False,
            False,
            True,
            adc_meta,
            "high",
            "explicit ADC in filename",
            qc,
        )
    if adc_meta and not trace_name:
        # dADC/ADC hidden in metadata (e.g. generic _dwi that is really an ADC map).
        # Intensity, when available, corroborates and lifts confidence.
        conf = "high" if _looks_like_adc_intensity(ev.intensity) else "medium"
        if conf == "medium":
            qc.append("adc_from_metadata_only")
        return ScanClassification(
            config.ADC,
            b,
            False,
            False,
            False,
            True,
            conf,
            "ADC/dADC in SeriesDescription/ProtocolName" + ("; ADC-like intensity" if conf == "high" else ""),
            qc,
        )

    # --- 2. Trace provenance ---
    # Trace is a *representation*, not a class of its own except when the b-value is
    # unknown. When b is known the band drives the class (so a high-b trace is still
    # excluded from clinical use); the trace flag is preserved for provenance.
    if trace_name:
        if b is not None and config.EXACT_B1000_MIN <= b <= config.EXACT_B1000_MAX:
            # Trace @ exactly b1000 IS the clinical isotropic/trace DWI b1000
            # (Decision 2); used directly. A near-but-not-exact clinical trace
            # (e.g. b1200) falls through to DWI_NEAR_B1000 so it is synthesized.
            return ScanClassification(
                config.DWI_B1000,
                b,
                True,
                True,
                False,
                False,
                "high",
                f"trace image at exact b={b:g} -> DWI_B1000 (trace provenance)",
                qc,
            )
        band = config.band_for_bvalue(b)
        if band is not None:
            qc.append("provenance_is_trace")
            if band == config.DWI_EXTREME_HIGH_B:
                qc.append("extreme_high_b_excluded_from_clinical")
            return ScanClassification(
                band,
                b,
                True,
                True,
                False,
                False,
                "high",
                f"trace image at b={b:g} -> {band} (band drives class)",
                qc,
            )
        # b-value unknown: this is the canonical distinct trace channel.
        return ScanClassification(
            config.DWI_TRACE,
            None,
            True,
            True,
            False,
            False,
            "high",
            "trace image, b-value unknown",
            qc,
        )

    # --- 3. Parsed b-value band ---
    band = config.band_for_bvalue(b)
    if band is not None:
        conf = "high"
        reason = f"b-value {b:g} -> {band}"
        if band == config.DWI_EXTREME_HIGH_B:
            qc.append("extreme_high_b_excluded_from_clinical")
        return ScanClassification(band, b, False, False, False, False, conf, reason, qc)

    # --- 4. Generic _dwi with no b-value / name / metadata evidence ---
    if _looks_like_adc_intensity(ev.intensity):
        qc.append("adc_from_intensity_heuristic")
        return ScanClassification(
            config.ADC_LIKELY,
            None,
            False,
            False,
            False,
            False,
            "low",
            "generic _dwi with ADC-like intensity (bright CSF tail)",
            qc,
        )
    qc.append("ambiguous_generic_dwi")
    reason = "generic _dwi: no b-value, ADC, or trace evidence"
    if ev.intensity is None:
        reason += " (intensity not inspected)"
    return ScanClassification(config.DWI_DERIVED_UNKNOWN, None, False, False, False, False, "low", reason, qc)


# ---------------------------------------------------------------------------
# Structural modality naming (for the pretrain-default selection + audit rows)
# ---------------------------------------------------------------------------
def structural_modality(new_filename: str) -> str | None:
    """Canonical structural modality name from a filename suffix, else ``None``."""
    stem = ids.strip_nifti_ext(new_filename)
    if "_" not in stem:
        suffix = stem
    else:
        suffix = stem.rsplit("_", 1)[-1]
    return config.STRUCTURAL_SUFFIX_CANON.get(suffix.lower())


# ---------------------------------------------------------------------------
# Lightweight, name-only classifier shared with the overview figure
# ---------------------------------------------------------------------------
def classify_diffusion_from_names(
    new_filename: str,
    old_filename: str | None = None,
    series_description: str | None = None,
    protocol_name: str | None = None,
) -> str:
    """Coarse diffusion class from names/metadata only (no image load).

    Used by the dataset overview figure so its diffusion taxonomy stays in lock-step
    with the curation classifier. Never returns ADC_LIKELY (which needs intensity);
    an unresolved generic ``_dwi`` maps to :data:`config.DWI_DERIVED_UNKNOWN`.
    """
    ev = ScanEvidence(
        new_filename=new_filename,
        old_filename=old_filename,
        series_description=series_description,
        protocol_name=protocol_name,
        intensity=None,
    )
    return classify_diffusion_scan(ev).diffusion_class
