"""One row per canonical sample: what happened to it, and where it went.

The derivative is anchored to the canonical manifest, so the only defensible accounting is
per-sample and exhaustive. Every one of the canonical samples gets exactly one row and exactly
one status, and :func:`assert_accounting` enforces the partition rather than reporting it:

    materialized (reference + registered + passthrough) + qc_rejected + failed + missing
        == canonical samples

A sample that quietly disappeared between the manifest and the derivative is the failure mode
this file exists to make impossible. "We processed 31,902 of 33,336 and the rest were probably
fine" is not an auditable statement; "1,434 missing, listed by sample_id with a reason" is.
"""

import csv
import json
import logging
import os
from asparagus_preprocessing.session_coreg import canonical, qc as qc_mod
from collections import Counter
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1

DERIVATIVE_FIELDS = (
    "sample_id",
    "dataset",
    "participant_id",
    "session_id",
    "session_key",
    "modality",
    "source_cleaned_path",
    "reference_sample_id",
    "reference_modality",
    "registered",
    "registration_method",
    "transform_fitted",
    "strict_pairing_eligible",
    "transform_path",
    "output_nifti",
    "status",
    "reason",
    "out_shape",
    "out_spacing",
    "translation_norm_mm",
    "rotation_deg",
    "determinant",
    "orthogonality_error",
    "nmi_before",
    "nmi_after",
    "nmi_improvement",
    "foreground_retained_frac",
)

#: Statuses whose sample exists on disk and belongs to the derivative corpus.
#:
#: ``header_only_aligned`` shares the session output lattice, but no transform was fitted. Its
#: anatomical alignment relies on scanner-header accuracy and must remain distinguishable from
#: optimised registration for downstream consumers. A common lattice is necessary for cross-modal
#: pairing but does not by itself prove anatomical co-registration, and the pilot300 header-prior
#: analysis showed scanner headers are not universally exact -- so these samples are materialised
#: and usable (unimodal SSL included), but ``strict_pairing_eligible`` is false for them.
MATERIALIZED = ("reference", "registered", "header_only_aligned", "passthrough_single_scan", "passthrough_ambiguous_session")
#: Statuses whose sample does not.
NOT_MATERIALIZED = ("qc_rejected", "failed", "missing")
ALL_STATUSES = MATERIALIZED + NOT_MATERIALIZED

#: ``registration_method`` values. Mirrors ``pipeline.HEADER_ONLY``; the two are asserted equal by
#: the test suite so the manifest cannot drift away from what the pipeline writes.
METHOD_HEADER_ONLY = "header_only"
METHOD_OPTIMISED = "optimised"
METHOD_NONE = "none"


class AccountingError(AssertionError):
    """Raised when the derivative does not account for every canonical sample exactly once."""


def _fmt(value, digits: int = 5):
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return value


def _modality_entry(qc: dict, source_path: str) -> dict:
    for mod in qc.get("modalities", []) or []:
        if mod.get("source_path") == source_path:
            return mod
    return {}


def build_records(
    canonical_rows: List[dict],
    verdicts_by_key: Dict[str, object],
    output_root: str,
) -> List[dict]:
    """Build the derivative manifest by joining canonical rows to each session's ``qc.json``.

    The canonical manifest drives the iteration, never the output tree: a sample whose session
    was never processed must still produce a row (``status=missing``), which is precisely what a
    tree walk would fail to notice.
    """
    records: List[dict] = []
    qc_cache: Dict[str, Optional[dict]] = {}

    # Indexed once rather than rescanned per row: resolving each reference by a linear search made
    # the join quadratic in the corpus size, which is 33,336 samples in production.
    # ``setdefault`` keeps the first row for a path, matching the search it replaces.
    rows_by_cleaned_path: Dict[str, dict] = {}
    for row in canonical_rows:
        rows_by_cleaned_path.setdefault(str(row.get("cleaned_path", "")), row)

    for row in canonical_rows:
        key = canonical.session_key(row)
        source = str(row.get("cleaned_path", ""))
        verdict = verdicts_by_key.get(key)

        if key not in qc_cache:
            qc_cache[key] = qc_mod.read_session_qc(os.path.join(output_root, key, "qc.json"))
        qc = qc_cache[key]

        record = {name: "" for name in DERIVATIVE_FIELDS}
        record.update(
            sample_id=str(row.get("sample_id", "")),
            dataset=str(row.get("dataset", "")),
            participant_id=str(row.get("participant_id", "")),
            session_id=str(row.get("session_id", "")),
            session_key=key,
            modality=str(row.get("modality_canonical", "")),
            source_cleaned_path=source,
            registered=False,
            registration_method=METHOD_NONE,
            transform_fitted=False,
            strict_pairing_eligible=False,
            status="missing",
            reason="session not processed",
        )

        if qc is None:
            records.append(record)
            continue
        if qc.get("status") == "dry_run":
            # A dry run simulates commands and writes no images. Counting it as materialized would
            # let a plan masquerade as a corpus.
            record["reason"] = "dry_run: commands simulated, no output written"
            records.append(record)
            continue

        reference_path = (qc.get("reference") or {}).get("source_path", "")
        ref_row = rows_by_cleaned_path.get(reference_path, {})
        record["reference_sample_id"] = str(ref_row.get("sample_id", ""))
        record["reference_modality"] = str(ref_row.get("modality_canonical", "")) or (qc.get("reference") or {}).get(
            "modality", ""
        )

        mod = _modality_entry(qc, source)
        if not mod:
            record["status"] = "missing"
            record["reason"] = f"no modality entry in session qc (session status={qc.get('status', '?')})"
            records.append(record)
            continue

        registration = mod.get("registration") or {}
        record.update(
            output_nifti=mod.get("output_path", ""),
            transform_path=mod.get("transform_path", ""),
            out_shape="x".join(str(int(s)) for s in (mod.get("output_shape") or [])),
            out_spacing="x".join(f"{float(v):g}" for v in (mod.get("output_spacing") or [])),
            translation_norm_mm=_fmt(registration.get("translation_norm_mm"), 4),
            rotation_deg=_fmt(registration.get("rotation_deg"), 4),
            determinant=_fmt(registration.get("determinant"), 6),
            orthogonality_error=_fmt(registration.get("orthogonality_error"), 8),
            nmi_before=_fmt(registration.get("nmi_before")),
            nmi_after=_fmt(registration.get("nmi_after")),
            nmi_improvement=_fmt(registration.get("nmi_improvement")),
            foreground_retained_frac=_fmt(mod.get("foreground_retained_frac"), 4),
        )

        is_reference = bool(mod.get("is_reference"))
        ok = mod.get("status") == "ok"
        # Snapshots written before ``registered`` was narrowed used it to mean "went through the
        # registration path". Read that older value as *attempted* only, and let the method say what
        # actually happened, so an existing qc.json still classifies correctly without recomputation.
        legacy_attempted = bool(mod.get("registered", not is_reference and qc.get("coregistration_applied")))
        attempted = bool(mod.get("coregistration_attempted", legacy_attempted))
        method = str(mod.get("registration_method") or (METHOD_OPTIMISED if legacy_attempted else METHOD_NONE))
        # An optimiser fitted this transform and QC accepted it. A header alignment also writes an
        # auditable matrix, so the existence of transform_path proves nothing on its own.
        fitted = bool(mod.get("transform_fitted", method == METHOD_OPTIMISED)) and ok and not is_reference
        record["registered"] = fitted
        record["transform_fitted"] = fitted
        record["registration_method"] = method if (attempted and not is_reference) else METHOD_NONE

        if not ok:
            record["status"] = "qc_rejected" if "output_qc" in str(mod.get("reason", "")) else "failed"
            record["reason"] = str(mod.get("reason", ""))[:300]
        elif is_reference:
            record["status"] = "reference"
            record["reason"] = (qc.get("reference") or {}).get("reason", "")
        elif method == METHOD_HEADER_ONLY:
            record["status"] = "header_only_aligned"
            record["reason"] = "scanner header alignment; no transform was fitted"
        elif fitted:
            record["status"] = "registered"
            record["reason"] = ""
        elif attempted:
            # The ladder ran, nothing was fitted, and yet the modality reports ok. That combination
            # has no defensible meaning, so it is failed rather than guessed into a passthrough.
            record["status"] = "failed"
            record["reason"] = f"registration attempted but no transform was fitted (method={method})"
        else:
            # Why co-registration did not happen is recorded in the session's own qc.json, so read
            # that first: the audit verdicts are supplied separately and a roll-up run without them
            # would otherwise label a five-modality session blocked for scanner conflict as
            # "single_scan_session" -- a statement that is simply false.
            block_reason = str(qc.get("coreg_block_reason") or "")
            blocked = bool(block_reason) or qc.get("coreg_allowed") is False
            verdict_blocked = verdict is not None and getattr(verdict, "status", "") == "ambiguous"
            if blocked or verdict_blocked:
                record["status"] = "passthrough_ambiguous_session"
                record["reason"] = block_reason or getattr(verdict, "reason", "") or "ambiguous_session"
            else:
                record["status"] = "passthrough_single_scan"
                record["reason"] = "single_scan_session"
        records.append(record)

    _assign_strict_pairing(records)
    return records


def _assign_strict_pairing(records: List[dict]) -> None:
    """Mark which samples a fitted-registration consumer may pair strictly.

    Eligibility is deliberately conservative and session-aware. An optimised moving scan is
    eligible. A reference is eligible only where its session actually produced at least one
    optimised registration -- it is the anchor those fits were measured against, so excluding it
    outright would leave every valid pair with no partner, while marking it eligible in a session
    that fitted nothing would assert a registration that never happened. Everything else, header
    alignments included, is false: usable, materialised, but never silently equivalent to a fit.
    """
    fitted_sessions = {r["session_key"] for r in records if r.get("status") == "registered"}
    for record in records:
        status = record.get("status")
        if status == "registered":
            record["strict_pairing_eligible"] = True
        elif status == "reference":
            record["strict_pairing_eligible"] = record.get("session_key") in fitted_sessions
        else:
            record["strict_pairing_eligible"] = False


def accounting(records: Iterable[dict]) -> dict:
    """Counts per status, plus the materialized / not-materialized split."""
    records = list(records)  # counted more than once below; a generator would silently zero the rest
    counts = Counter(r.get("status", "missing") for r in records)
    total = sum(counts.values())
    materialized = sum(counts.get(s, 0) for s in MATERIALIZED)
    return {
        "total_rows": total,
        "materialized": materialized,
        "reference": counts.get("reference", 0),
        "registered": counts.get("registered", 0),
        "header_only_aligned": counts.get("header_only_aligned", 0),
        "passthrough_single_scan": counts.get("passthrough_single_scan", 0),
        "passthrough_ambiguous_session": counts.get("passthrough_ambiguous_session", 0),
        "qc_rejected": counts.get("qc_rejected", 0),
        "failed": counts.get("failed", 0),
        "missing": counts.get("missing", 0),
        "transform_fitted": sum(1 for r in records if r.get("transform_fitted")),
        "strict_pairing_eligible": sum(1 for r in records if r.get("strict_pairing_eligible")),
        "unknown_status": {k: v for k, v in counts.items() if k not in ALL_STATUSES},
    }


def assert_accounting(records: List[dict], expected_samples: Optional[int] = None) -> dict:
    """Enforce the partition. Raises :class:`AccountingError` on any violation."""
    summary = accounting(records)

    if summary["unknown_status"]:
        raise AccountingError(f"derivative manifest carries unknown statuses: {summary['unknown_status']}")

    ids = [r.get("sample_id", "") for r in records]
    if "" in ids:
        raise AccountingError(f"{ids.count('')} derivative rows have an empty sample_id")
    duplicated = [sid for sid, n in Counter(ids).items() if n > 1]
    if duplicated:
        raise AccountingError(f"duplicate sample_id in the derivative manifest: {sorted(duplicated)[:10]}")

    parts = summary["materialized"] + summary["qc_rejected"] + summary["failed"] + summary["missing"]
    if parts != summary["total_rows"]:
        raise AccountingError(f"status partition does not cover every row: {parts} != {summary['total_rows']}")

    if expected_samples is not None and summary["total_rows"] != expected_samples:
        raise AccountingError(
            f"derivative manifest has {summary['total_rows']} rows but the canonical corpus has "
            f"{expected_samples}; a sample was added or lost."
        )
    return summary


def write(records: List[dict], path: str) -> str:
    """Write the derivative manifest deterministically (sorted by ``sample_id``)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ordered = sorted(records, key=lambda r: r.get("sample_id", ""))
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(DERIVATIVE_FIELDS), delimiter="\t", extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(ordered)
    os.replace(tmp, path)
    return canonical.sha256_of(path)


def write_metadata(payload: dict, path: str) -> str:
    """Write the derivative-level provenance/accounting JSON atomically."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return path


def tensorisation_rows(records: List[dict]) -> List[dict]:
    """Re-emit the materialized rows with canonical column names, for the ``.pt`` stage.

    Tensorisation is deliberately a separate stage: it consumes a manifest whose ``cleaned_path``
    points at the *registered* NIfTI, so the existing P0 conversion contract can be reused
    unchanged rather than forked.

    ``status``, ``registration_method``, ``transform_fitted`` and ``strict_pairing_eligible`` are
    carried through as extra columns. The conversion stage reads rows with ``csv.DictReader`` and
    ignores what it does not use, but a pairing consumer downstream of the ``.pt`` files has no
    other way to tell a fitted registration from a header alignment once the NIfTIs are gone.
    """
    out = []
    for record in records:
        if record.get("status") not in MATERIALIZED:
            continue
        out.append(
            {
                "sample_id": record["sample_id"],
                "dataset": record["dataset"],
                "participant_id": record["participant_id"],
                "session_id": record["session_id"],
                "cleaned_path": record["output_nifti"],
                "modality_canonical": record["modality"],
                "session_n_scans": "",
                "shape": record.get("out_shape", ""),
                "pixdim": record.get("out_spacing", ""),
                "orientation": "RAS",
                "status": record.get("status", ""),
                "registration_method": record.get("registration_method", METHOD_NONE),
                "transform_fitted": bool(record.get("transform_fitted")),
                "strict_pairing_eligible": bool(record.get("strict_pairing_eligible")),
            }
        )
    return out
