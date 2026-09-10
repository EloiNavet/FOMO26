"""Prove that a "session" is a genuine acquisition visit before co-registering it.

Rigidly aligning two scans asserts they show the same anatomy at the same moment. If the identity
triple ``(dataset, participant_id, session_id)`` has silently collapsed two *visits* — a
follow-up MRI months later, a different scanner, a different patient state — then registering
them is not noise reduction, it is fabricating a correspondence that never existed. Longitudinal
cohorts are present in this corpus by name (``PT029_OASIS2``,
``PT035_Yale_Brain_Mets_Longitudinal``), so this is a real risk, not a hypothetical one.

**What this corpus can and cannot support.** The cleaned manifest carries 46 columns and *none*
of them is an acquisition date — there is no ``AcquisitionDate``, ``StudyDate``, ``SeriesDate`` or
timestamp anywhere. Date-based longitudinal separation is therefore impossible here, and the
audit says so explicitly (``acquisition_dates_available: false``) rather than quietly omitting
the check and leaving a reader to assume it passed. Separation instead rests on ``session_id``
plus physical-acquisition evidence that must agree *within* a session:

* **HARD** signals block registration. Two different scanner models, manufacturers or field
  strengths under one session key mean two different scanning events. A contradictory subject age
  means two different visits. A missing/synthetic session id, or one that disagrees with the file
  path, means the identity is not established at all.
* **SOFT** signals are recorded only. ``SoftwareVersions`` and ``SeriesDescription`` legitimately
  vary between series of one visit (e.g. ``'3.2.3'`` vs ``'3.2.3\\3.2.3.4'``).

A blocked session is **not** a failure and is **not** dropped: it becomes a passthrough, still
materialised at RAS/1 mm so the canonical sample set is preserved. Fail-closed here means
"refuse to register", never "refuse to include".
"""

import csv
import json
import logging
import os
from asparagus_preprocessing.session_coreg import canonical
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

AUDIT_VERSION = 1

#: Columns whose disagreement inside one session key proves two distinct acquisition events.
HARD_CONSISTENCY_FIELDS = ("Manufacturer", "ManufacturersModelName", "field_strength_numeric", "age_numeric")
#: Columns that legitimately vary between series of a single visit.
SOFT_CONSISTENCY_FIELDS = ("SoftwareVersions", "SeriesDescription", "ProtocolName")

#: Values that carry no information and must never be compared as if they did.
_NULLS = {"", "nan", "n/a", "na", "none", "null", "unknown", "-"}

#: Session ids that indicate the source had none and one was fabricated downstream.
SYNTHETIC_SESSION_IDS = {"ses-01", "ses-1", "ses-unknown", "ses-na", "ses-none"}


class SessionIdentityError(ValueError):
    """Raised when the audit cannot be performed at all."""


@dataclass
class SessionVerdict:
    """One session's identity decision, with the evidence behind it."""

    session_key: str
    dataset: str
    participant_id: str
    session_id: str
    n_samples: int
    modalities: List[str] = field(default_factory=list)
    sample_ids: List[str] = field(default_factory=list)
    eligible: bool = False
    status: str = "single_scan"  # eligible | single_scan | ambiguous
    hard_reasons: List[str] = field(default_factory=list)
    soft_flags: List[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        if self.status == "ambiguous":
            return "ambiguous_session:" + ",".join(self.hard_reasons)
        if self.status == "single_scan":
            return "single_scan_session"
        return ""

    def to_row(self) -> dict:
        return {
            "session_key": self.session_key,
            "dataset": self.dataset,
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "n_samples": self.n_samples,
            "modalities": ";".join(self.modalities),
            "sample_ids": ";".join(self.sample_ids),
            "status": self.status,
            "eligible": self.eligible,
            "reason": self.reason,
            "soft_flags": ";".join(self.soft_flags),
        }


def _clean(value) -> str:
    text = str(value if value is not None else "").strip()
    return "" if text.lower() in _NULLS else text


def _distinct(rows: List[dict], field_name: str) -> List[str]:
    return sorted({v for v in (_clean(r.get(field_name)) for r in rows) if v})


def read_cleaned_metadata(path: str) -> Dict[str, dict]:
    """Index the cleaned-corpus ``manifest.tsv`` by ``cleaned_path``.

    This is where the scanner/demographic evidence lives; the canonical manifest does not carry
    it. Absent, the audit still runs but can only check structural identity, and says so.
    """
    if not path or not os.path.exists(path):
        return {}
    index: Dict[str, dict] = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = _clean(row.get("cleaned_path"))
            if key:
                index[key] = row
    logger.info("Cleaned metadata: %d rows indexed from %s", len(index), path)
    return index


def audit_session(
    key: str, rows: List[dict], metadata: Dict[str, dict], require_explicit_session: bool = False
) -> SessionVerdict:
    """Classify one session as eligible / single_scan / ambiguous."""
    first = rows[0]
    verdict = SessionVerdict(
        session_key=key,
        dataset=str(first.get("dataset", "")),
        participant_id=str(first.get("participant_id", "")),
        session_id=str(first.get("session_id", "")),
        n_samples=len(rows),
        modalities=sorted({str(r.get("modality_canonical", "")) for r in rows}),
        sample_ids=[str(r.get("sample_id", "")) for r in rows],
    )

    # --- structural identity (checked for every session, single-scan included) ---
    for row in rows:
        ok, why = canonical.identity_matches_path(row)
        if not ok:
            verdict.hard_reasons.append(f"identity_path_mismatch({why})")
            break
    if not _clean(verdict.session_id):
        verdict.hard_reasons.append("missing_session_id")
    elif require_explicit_session and verdict.session_id.lower() in SYNTHETIC_SESSION_IDS:
        verdict.hard_reasons.append(f"synthetic_session_id({verdict.session_id})")

    paths = [str(r.get("cleaned_path", "")) for r in rows]
    if len(set(paths)) != len(paths):
        duplicated = sorted({p for p in paths if paths.count(p) > 1})
        verdict.hard_reasons.append(f"duplicate_cleaned_path({len(duplicated)})")

    # --- acquisition-event consistency (only meaningful with >1 scan) ---
    meta_rows = [metadata.get(p, {}) for p in paths]
    have_metadata = any(meta_rows)
    if len(rows) > 1 and have_metadata:
        for name in HARD_CONSISTENCY_FIELDS:
            values = _distinct(meta_rows, name)
            if len(values) > 1:
                verdict.hard_reasons.append(f"{name}_conflict({'|'.join(values[:3])})")
        for name in SOFT_CONSISTENCY_FIELDS:
            values = _distinct(meta_rows, name)
            if len(values) > 1:
                verdict.soft_flags.append(f"{name}_varies({len(values)})")
    elif len(rows) > 1 and not have_metadata:
        verdict.soft_flags.append("no_cleaned_metadata_for_session")

    # --- verdict ---
    if verdict.hard_reasons:
        verdict.status = "ambiguous"
        verdict.eligible = False
    elif len(rows) < 2:
        verdict.status = "single_scan"
        verdict.eligible = False
    else:
        verdict.status = "eligible"
        verdict.eligible = True
    return verdict


def audit_sessions(
    canonical_rows: List[dict],
    cleaned_metadata_path: Optional[str] = None,
    require_explicit_session: bool = False,
) -> List[SessionVerdict]:
    """Audit every session in the canonical manifest, in deterministic key order."""
    metadata = read_cleaned_metadata(cleaned_metadata_path) if cleaned_metadata_path else {}
    groups = canonical.group_by_session(canonical_rows)
    return [
        audit_session(key, groups[key], metadata, require_explicit_session=require_explicit_session) for key in sorted(groups)
    ]


def eligible_keys(verdicts: List[SessionVerdict]) -> Set[str]:
    """Session keys that may be co-registered. Everything else is passthrough."""
    return {v.session_key for v in verdicts if v.eligible}


def summarise(verdicts: List[SessionVerdict], canonical_rows: List[dict], metadata_available: bool) -> dict:
    """Aggregate report answering the audit questions the derivative contract requires."""
    per_session_counts = Counter(v.n_samples for v in verdicts)
    participants = {(v.dataset, v.participant_id) for v in verdicts}
    sessions_per_participant = Counter()
    for v in verdicts:
        sessions_per_participant[(v.dataset, v.participant_id)] += 1

    modality_combos = Counter(" + ".join(v.modalities) for v in verdicts if v.n_samples > 1)
    datasets_multi = Counter(v.dataset for v in verdicts if v.n_samples > 1)
    hard_codes = Counter(r.split("(")[0] for v in verdicts for r in v.hard_reasons)
    soft_codes = Counter(f.split("(")[0] for v in verdicts for f in v.soft_flags)

    ambiguous = [v for v in verdicts if v.status == "ambiguous"]
    return {
        "audit_version": AUDIT_VERSION,
        "canonical_samples": len(canonical_rows),
        "subjects": len(participants),
        "session_keys": len(verdicts),
        "single_scan_sessions": sum(1 for v in verdicts if v.n_samples == 1),
        "multi_scan_sessions": sum(1 for v in verdicts if v.n_samples > 1),
        "scans_per_session": {str(k): v for k, v in sorted(per_session_counts.items())},
        "participants_with_multiple_sessions": sum(1 for c in sessions_per_participant.values() if c > 1),
        "sessions_eligible_for_registration": sum(1 for v in verdicts if v.eligible),
        "sessions_ambiguous": len(ambiguous),
        "samples_in_ambiguous_sessions": sum(v.n_samples for v in ambiguous),
        "ambiguity_codes": dict(hard_codes.most_common()),
        "soft_flag_codes": dict(soft_codes.most_common()),
        "modality_combinations": dict(modality_combos.most_common(30)),
        "datasets_contributing_multi_scan_sessions": dict(datasets_multi.most_common()),
        # Stated explicitly: this corpus carries no acquisition timestamps, so no date-based
        # longitudinal check was performed. Absence of the check is reported, not implied.
        "acquisition_dates_available": False,
        "acquisition_date_note": (
            "No AcquisitionDate/StudyDate/SeriesDate column exists in the cleaned manifest. "
            "Longitudinal separation rests on session_id plus scanner/demographic consistency."
        ),
        "cleaned_metadata_available": metadata_available,
        "hard_consistency_fields": list(HARD_CONSISTENCY_FIELDS),
        "soft_consistency_fields": list(SOFT_CONSISTENCY_FIELDS),
        "examples_ambiguous": [v.to_row() for v in ambiguous[:20]],
    }


AUDIT_ROW_FIELDS = (
    "session_key",
    "dataset",
    "participant_id",
    "session_id",
    "n_samples",
    "modalities",
    "sample_ids",
    "status",
    "eligible",
    "reason",
    "soft_flags",
)


def write_audit(verdicts: List[SessionVerdict], summary: dict, output_dir: str) -> dict:
    """Write ``session_audit.tsv``, ``session_ambiguous.tsv`` and ``session_audit.json``."""
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "sessions": os.path.join(output_dir, "session_audit.tsv"),
        "ambiguous": os.path.join(output_dir, "session_ambiguous.tsv"),
        "summary": os.path.join(output_dir, "session_audit.json"),
    }
    for name, subset in (("sessions", verdicts), ("ambiguous", [v for v in verdicts if v.status == "ambiguous"])):
        with open(paths[name], "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(AUDIT_ROW_FIELDS), delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(v.to_row() for v in subset)
    with open(paths["summary"], "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return paths


def group_examples(verdicts: List[SessionVerdict], limit: int = 5) -> Dict[str, List[str]]:
    """A few concrete session keys per ambiguity code, for a human to eyeball."""
    out: Dict[str, List[str]] = defaultdict(list)
    for v in verdicts:
        for reason in v.hard_reasons:
            code = reason.split("(")[0]
            if len(out[code]) < limit:
                out[code].append(v.session_key)
    return dict(out)
