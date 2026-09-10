"""Deterministic session manifest, chunking, and audits.

The manifest is the single source of truth for a run: it is built once, records
every session's geometry summary (3D/4D counts, modalities), and is sharded by
``chunk_id`` so a SLURM array can process disjoint slices with no coordination.
"""

import json
import logging
import math
import os
from asparagus_preprocessing.session_coreg import modalities
from asparagus_preprocessing.session_coreg.discovery import find_sessions
from asparagus_preprocessing.session_coreg.geometry import probe_session
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1


def build_records(input_root: str, extensions: List[str]) -> List[dict]:
    """Probe every session under ``input_root`` into deterministic manifest rows."""
    input_root = os.path.abspath(input_root)
    records: List[dict] = []
    for session in find_sessions(input_root, extensions):
        geoms = probe_session(session.image_paths)
        four_d = [g.path for g in geoms if g.is_4d]
        three_d = [g for g in geoms if not g.is_4d]
        mods = sorted({modalities.classify(g.path) for g in three_d})
        records.append(
            {
                "session_key": session.key,
                "dataset": session.dataset,
                "subject": session.subject,
                "session": session.session,
                "n_scans": len(session.image_paths),
                "n_3d": len(three_d),
                "n_4d": len(four_d),
                "modalities": mods,
                "four_d_paths": [os.path.relpath(p, input_root) for p in four_d],
                "image_paths": [os.path.relpath(p, input_root) for p in session.image_paths],
                "min_voxel_volume": round(min((g.voxel_volume for g in three_d), default=0.0), 5),
                "best_max_spacing": round(min((g.max_spacing for g in three_d), default=0.0), 4),
            }
        )
    records.sort(key=lambda r: r["session_key"])  # deterministic order == deterministic chunking
    logger.info("Manifest: %d sessions, %d with 4D scans", len(records), sum(1 for r in records if r["n_4d"]))
    return records


def write_manifest(records: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"version": MANIFEST_VERSION, "n_sessions": len(records), "records": records}
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)


def read_manifest(path: str) -> List[dict]:
    with open(path) as handle:
        payload = json.load(handle)
    return payload["records"]


def n_chunks(n_sessions: int, chunk_size: int) -> int:
    return int(math.ceil(n_sessions / chunk_size)) if chunk_size > 0 else 0


def chunk_records(records: List[dict], chunk_size: int, chunk_id: int) -> List[dict]:
    """Deterministic contiguous slice of the manifest for one array task."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    start = chunk_id * chunk_size
    return records[start : start + chunk_size]


def four_d_rows(records: List[dict]) -> List[dict]:
    """One row per 4D scan for the inventory report."""
    rows = []
    for record in records:
        for rel in record.get("four_d_paths", []):
            rows.append(
                {
                    "session_key": record["session_key"],
                    "dataset": record["dataset"],
                    "modality": modalities.classify(rel),
                    "path": rel,
                }
            )
    return rows


def inventory_summary(records: List[dict]) -> dict:
    datasets: Dict[str, dict] = {}
    total_3d = total_4d = 0
    for record in records:
        total_3d += record["n_3d"]
        total_4d += record["n_4d"]
        d = datasets.setdefault(record["dataset"], {"sessions": 0, "n_3d": 0, "n_4d": 0, "sessions_with_4d": 0})
        d["sessions"] += 1
        d["n_3d"] += record["n_3d"]
        d["n_4d"] += record["n_4d"]
        d["sessions_with_4d"] += 1 if record["n_4d"] else 0
    return {
        "n_sessions": len(records),
        "n_3d_scans": total_3d,
        "n_4d_scans": total_4d,
        "sessions_with_4d": sum(1 for r in records if r["n_4d"]),
        "by_dataset": datasets,
    }


def select_smoke(records: List[dict], max_per_category: int = 2) -> List[dict]:
    """Deterministic stratified smoke set covering the required scenarios.

    Categories: single-scan, T1w+T2w, T1w+FLAIR, thick-slice clinical, DWI/ADC,
    and >3-modality sessions. Picks up to ``max_per_category`` of each.
    """

    def has(rec, *mods):
        return all(m in rec["modalities"] for m in mods)

    categories = {
        "single_scan": lambda r: r["n_3d"] == 1,
        "t1_t2": lambda r: has(r, "T1w", "T2w"),
        "t1_flair": lambda r: has(r, "T1w", "FLAIR"),
        "thick_slice": lambda r: r["best_max_spacing"] >= 3.0,
        "dwi_adc": lambda r: ("DWI" in r["modalities"]) or ("ADC" in r["modalities"]),
        "multimodal_gt3": lambda r: r["n_3d"] > 3,
    }
    chosen: Dict[str, dict] = {}
    for name, pred in categories.items():
        picks = [r for r in records if pred(r)][:max_per_category]
        for r in picks:
            chosen.setdefault(r["session_key"], r).setdefault("smoke_categories", [])
            chosen[r["session_key"]]["smoke_categories"].append(name)
    return sorted(chosen.values(), key=lambda r: r["session_key"])


def records_to_sessions(records: List[dict], input_root: str):
    """Rehydrate manifest rows into Session objects for processing."""
    from asparagus_preprocessing.session_coreg.discovery import Session

    input_root = os.path.abspath(input_root)
    sessions = []
    for record in records:
        session_dir = os.path.join(input_root, record["session_key"])
        sessions.append(
            Session(
                key=record["session_key"],
                session_dir=session_dir,
                dataset=record["dataset"],
                subject=record["subject"],
                session=record["session"],
                image_paths=[os.path.join(input_root, rel) for rel in record["image_paths"]],
            )
        )
    return sessions


def find_optional(input_root: str, manifest_path: Optional[str], extensions: List[str]) -> List[dict]:
    """Return manifest records, reading a saved manifest when available."""
    if manifest_path and os.path.exists(manifest_path):
        logger.info("Using manifest %s", manifest_path)
        return read_manifest(manifest_path)
    return build_records(input_root, extensions)
