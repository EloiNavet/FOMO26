#!/usr/bin/env python3
"""Leak-free, per-dataset stratified 80/20 train/val split for FOMO pretraining.

Design goals (see docs/data-pipeline/dwi_curation.md and the project plan):

* **No subject leakage.** The split unit is the *subject* = ``(dataset, participant)``.
  Every session and every modality of a subject lands on the same side. A subject
  belongs to exactly one dataset, so grouping is collision-free even for nested
  OpenNeuro datasets (``PT030_OpenNeuro/ds004889``).
* **Representative on both sides.** The split is stratified *within each dataset*
  (so every dataset is 80/20, hence every pathology is ~80/20), balancing pathology
  family x sex x age-bin with a fallback ladder for small strata. Modality balance
  follows from dataset-scoped splitting and is verified in the report.

Output is the format the trainer consumes: a JSON list with a single fold
``[{"train": [paths...], "val": [paths...]}]`` (read as ``load_json(split)[fold]``),
plus a ``split_report.tsv`` for auditing balance and confirming the leak check.

Runnable standalone (local validation on TSVs) or on Jean Zay over a paths manifest::

    # local: drive the file list from a cleaned mapping.tsv
    python make_grouped_stratified_split.py \
        --from-mapping .../fomo300K/mapping.tsv \
        --participants .../fomo300K/participants.tsv \
        --output /tmp/split_80_20.json --report /tmp/split_report.tsv

    # Jean Zay: drive from the processed paths manifest
    python make_grouped_stratified_split.py \
        --paths $ASPARAGUS_DATA/PT900_FOMO300K/paths.json \
        --participants $CLEANED/participants.tsv \
        --output $ASPARAGUS_DATA/PT900_FOMO300K/split_80_20_grouped_stratified.json \
        --report $ASPARAGUS_DATA/PT900_FOMO300K/split_report.tsv
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pandas as pd
import re
from collections import Counter, defaultdict
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Optional

LOGGER = logging.getLogger("make_grouped_stratified_split")

_MISSING = {"", "#", "na", "n/a", "nan", "none", "null", "unknown", "not available"}
_DATASET_RE = re.compile(r"^(PT|SEG|CLS|REG)\d{3}")
_SUFFIX_RE = re.compile(r"\.(nii\.gz|nii|pt|pkl|pk|npy|npz)$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Pathology family (mirrors the dataset-analysis pathology mapping;
# kept inline so this preprocessing script is self-contained on the cluster).
# --------------------------------------------------------------------------- #
def pathology_family(raw: object) -> str:
    s = "" if raw is None else str(raw).strip().lower()
    if s in _MISSING:
        return "UNKNOWN"
    if (
        s
        in {"control", "cn", "hc", "healthy control", "neurotypical", "nh", "nondemented", "non-demented", "healthy", "normal"}
        or "control" in s
        or "typically develop" in s
    ):
        return "Control"
    if any(
        k in s
        for k in [
            "tumor",
            "glioma",
            "glioblastoma",
            "astrocytoma",
            "oligodendro",
            "meningioma",
            "adenoma",
            "pituitary",
            "ependymoma",
            "lymphoma",
            "neoplasm",
            "carcinoma",
            "metasta",
        ]
    ):
        return "Tumor"
    if any(k in s for k in ["stroke", "infarct", "ischemi", "aneurysm", "hemorrhage", "haemorrhage", "vascular"]):
        return "Stroke/Vascular"
    if any(k in s for k in ["alzheimer", "dement", "ftd", "mci", "converted"]) or s == "ad":
        return "Neurodeg"
    if any(k in s for k in ["parkinson", "motor neuron", "dystonia", "huntington"]) or s == "pd":
        return "Movement/MND"
    if any(k in s for k in ["epilep", "dysplasia", "seizure"]):
        return "Epilepsy"
    if any(
        k in s
        for k in ["depress", "anxiet", "psychiat", "mood", "adhd", "autism", "psychosis", "bipolar", "schizo", "ocd", "ptsd"]
    ):
        return "Psych/Neurodev"
    if s == "ms" or "multiple sclerosis" in s or "demyelin" in s:
        return "MS"
    return "Other"


# --------------------------------------------------------------------------- #
# Path -> (dataset, participant) parsing, nesting-safe
# --------------------------------------------------------------------------- #
def parse_subject(path: str) -> Optional[tuple[str, str]]:
    """Return ``(dataset, participant)`` from a BIDS-ish path, else ``None``.

    ``dataset`` spans every component from the first ``PT###/SEG###/...`` folder up to
    (but excluding) the ``sub-*`` folder, so nested OpenNeuro datasets stay distinct.
    """
    parts = [p for p in Path(str(path)).parts if p not in ("/", "")]
    sub_idx = next((i for i, p in enumerate(parts) if p.lower().startswith("sub-")), None)
    if sub_idx is None:
        return None
    start = next((i for i, p in enumerate(parts[:sub_idx]) if _DATASET_RE.match(p)), None)
    if start is None:
        return None
    dataset = "/".join(parts[start:sub_idx])
    return dataset, parts[sub_idx]


def modality_label(path: str) -> str:
    """Coarse curated-modality label from a filename, for the balance report only."""
    name = _SUFFIX_RE.sub("", Path(str(path)).name)
    tokens = [t for t in name.split("_") if not re.match(r"^(sub|ses|run)[-_]?", t, re.IGNORECASE)]
    series = "_".join(tokens).lower() if tokens else name.lower()
    for key in ("dwi_b1000", "dwi_b0", "dwi_trace", "adc"):
        if series == key or series.endswith("_" + key):
            return key.upper() if key.startswith("dwi") else "ADC"
    for key in (
        "t1w",
        "t2w",
        "flair",
        "t2star",
        "t2starw",
        "swi",
        "gre",
        "t1c",
        "pdw",
        "mp2rage",
        "unit1",
        "cbf",
        "asl",
        "m0scan",
    ):
        if series == key or series.endswith("_" + key):
            return {"t2starw": "T2star"}.get(key, key.upper() if len(key) <= 4 else key.capitalize())
    if series.startswith("dwi_bval") or series == "dwi" or "bval" in series:
        return "dwi_other"
    return "other"


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def _clean(v: object) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return None if s.lower() in _MISSING else s


def normalize_sex(v: object) -> str:
    s = (_clean(v) or "").upper()
    if s in {"M", "MALE", "1"}:
        return "M"
    if s in {"F", "FEMALE", "0", "2"}:
        return "F"
    return "U"


def parse_age(v: object) -> Optional[float]:
    s = _clean(v)
    if s is None:
        return None
    try:
        a = float(s)
    except ValueError:
        return None
    return a if math.isfinite(a) and 0 <= a <= 120 else None


def age_bin(age: Optional[float], width: int) -> str:
    if age is None:
        return "NA"
    return f"{int(age // width) * width}"


def load_participants(path: Path, age_bin_width: int) -> dict[tuple[str, str], dict]:
    """Per-subject demographics keyed by (dataset, participant); mode over sessions."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    per_subject: dict[tuple[str, str], dict] = {}
    agg: dict[tuple[str, str], dict] = defaultdict(lambda: {"sex": Counter(), "age": [], "fam": Counter()})
    for r in df.itertuples(index=False):
        d = r._asdict()
        key = (_clean(d.get("dataset")) or "", _clean(d.get("participant_id")) or "")
        if not key[0] or not key[1]:
            continue
        a = agg[key]
        a["sex"][normalize_sex(d.get("sex"))] += 1
        age = parse_age(d.get("age"))
        if age is not None:
            a["age"].append(age)
        a["fam"][pathology_family(d.get("group"))] += 1
    for key, a in agg.items():
        sex = a["sex"].most_common(1)[0][0] if a["sex"] else "U"
        fam = a["fam"].most_common(1)[0][0] if a["fam"] else "UNKNOWN"
        age = (sum(a["age"]) / len(a["age"])) if a["age"] else None
        per_subject[key] = {"sex": sex, "fam": fam, "age": age, "age_bin": age_bin(age, age_bin_width)}
    return per_subject


# --------------------------------------------------------------------------- #
# File list -> subjects
# --------------------------------------------------------------------------- #
def paths_from_mapping(mapping_path: Path) -> list[str]:
    df = pd.read_csv(mapping_path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    out = []
    for r in df.itertuples(index=False):
        d = r._asdict()
        ds, np_ = _clean(d.get("dataset")), _clean(d.get("new_path"))
        if ds and np_:
            out.append(f"{ds}/{np_}")
    return out


def group_files_by_subject(paths: list[str]) -> tuple[dict[tuple[str, str], list[str]], int]:
    by_subject: dict[tuple[str, str], list[str]] = defaultdict(list)
    unparsed = 0
    for p in paths:
        key = parse_subject(p)
        if key is None:
            unparsed += 1
            continue
        by_subject[key].append(p)
    return by_subject, unparsed


# --------------------------------------------------------------------------- #
# Stratified per-dataset split
# --------------------------------------------------------------------------- #
def _candidate_strata(subjects: list[tuple[str, str]], rec: dict) -> list[Optional[list[str]]]:
    comp = [f"{rec[s]['fam']}|{rec[s]['sex']}|{rec[s]['age_bin']}" for s in subjects]
    ps = [f"{rec[s]['fam']}|{rec[s]['sex']}" for s in subjects]
    fam = [rec[s]["fam"] for s in subjects]
    sex = [rec[s]["sex"] for s in subjects]
    return [comp, ps, fam, sex, None]


def _usable(labels: Optional[list[str]], n_val: int, n_tr: int) -> bool:
    if n_val < 1 or n_tr < 1:
        return False
    if labels is None:
        return True
    counts = Counter(labels)
    return min(counts.values()) >= 2 and len(counts) <= min(n_val, n_tr)


def split_subjects(
    by_subject: dict[tuple[str, str], list[str]],
    rec: dict,
    val_fraction: float,
    seed: int,
) -> tuple[set, set, list[dict]]:
    """Return (train_subjects, val_subjects, per_dataset_audit)."""
    datasets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in by_subject:
        datasets[key[0]].append(key)

    train: set = set()
    val: set = set()
    audit: list[dict] = []
    for dataset in sorted(datasets):
        subs = sorted(datasets[dataset])
        n = len(subs)
        n_val = max(1, int(round(n * val_fraction)))
        n_tr = n - n_val
        if n < 2 or n_tr < 1:
            train.update(subs)
            audit.append({"dataset": dataset, "n_subjects": n, "n_train": n, "n_val": 0, "strata": "all_train_small_dataset"})
            continue
        candidates = _candidate_strata(subs, rec)
        levels = ["fam|sex|age_bin", "fam|sex", "fam", "sex", "none"]
        used, strata = next((name, cand) for name, cand in zip(levels, candidates) if _usable(cand, n_val, n_tr))
        tr, va = train_test_split(subs, test_size=val_fraction, random_state=seed, stratify=strata)
        train.update(tr)
        val.update(va)
        audit.append({"dataset": dataset, "n_subjects": n, "n_train": len(tr), "n_val": len(va), "strata": used})
    return train, val, audit


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def build_report(by_subject, rec, train_subjects, val_subjects, per_dataset_audit, val_fraction) -> tuple[list[dict], dict]:
    rows: list[dict] = []

    def add_axis(axis: str, key_of):
        tr_c, va_c = Counter(), Counter()
        for s in train_subjects:
            tr_c[key_of(s)] += 1
        for s in val_subjects:
            va_c[key_of(s)] += 1
        for k in sorted(set(tr_c) | set(va_c), key=str):
            tot = tr_c[k] + va_c[k]
            rows.append(
                {
                    "axis": axis,
                    "key": k,
                    "n_train": tr_c[k],
                    "n_val": va_c[k],
                    "n_total": tot,
                    "val_fraction": round(va_c[k] / tot, 4) if tot else 0.0,
                }
            )

    add_axis("dataset", lambda s: s[0])
    add_axis("pathology", lambda s: rec.get(s, {}).get("fam", "UNKNOWN"))
    add_axis("sex", lambda s: rec.get(s, {}).get("sex", "U"))
    add_axis("age_bin", lambda s: rec.get(s, {}).get("age_bin", "NA"))

    # image-level modality balance
    mod_tr, mod_va = Counter(), Counter()
    for s, files in by_subject.items():
        target = mod_tr if s in train_subjects else mod_va
        for f in files:
            target[modality_label(f)] += 1
    for k in sorted(set(mod_tr) | set(mod_va)):
        tot = mod_tr[k] + mod_va[k]
        rows.append(
            {
                "axis": "modality_images",
                "key": k,
                "n_train": mod_tr[k],
                "n_val": mod_va[k],
                "n_total": tot,
                "val_fraction": round(mod_va[k] / tot, 4) if tot else 0.0,
            }
        )

    # age mean/std per side
    def age_stats(subs):
        ages = [rec[s]["age"] for s in subs if rec.get(s, {}).get("age") is not None]
        if not ages:
            return float("nan"), float("nan")
        m = sum(ages) / len(ages)
        sd = (sum((a - m) ** 2 for a in ages) / len(ages)) ** 0.5
        return m, sd

    tr_m, tr_sd = age_stats(train_subjects)
    va_m, va_sd = age_stats(val_subjects)
    rows.append(
        {
            "axis": "age_stats",
            "key": "train_mean_std",
            "n_train": len(train_subjects),
            "n_val": "",
            "n_total": "",
            "val_fraction": f"{tr_m:.1f}+/-{tr_sd:.1f}",
        }
    )
    rows.append(
        {
            "axis": "age_stats",
            "key": "val_mean_std",
            "n_train": "",
            "n_val": len(val_subjects),
            "n_total": "",
            "val_fraction": f"{va_m:.1f}+/-{va_sd:.1f}",
        }
    )

    # overall balance deviation on reasonably-sized dataset & pathology cells
    devs = [
        abs(r["val_fraction"] - val_fraction)
        for r in rows
        if r["axis"] in ("dataset", "pathology") and isinstance(r["n_total"], int) and r["n_total"] >= 20
    ]
    max_dev = max(devs) if devs else 0.0

    leak = train_subjects & val_subjects
    summary = {
        "n_train_subjects": len(train_subjects),
        "n_val_subjects": len(val_subjects),
        "val_fraction_subjects": round(len(val_subjects) / max(len(train_subjects) + len(val_subjects), 1), 4),
        "max_abs_val_fraction_deviation": round(max_dev, 4),
        "leak_check": "PASS" if not leak else f"FAIL ({len(leak)} overlapping subjects)",
    }
    return rows, summary


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_split(paths: list[str], participants: dict, val_fraction: float, seed: int):
    by_subject, unparsed = group_files_by_subject(paths)
    if unparsed:
        LOGGER.warning("%d paths could not be parsed into (dataset, subject) and were skipped", unparsed)
    # default demographics for subjects missing from participants.tsv
    rec = {}
    for key in by_subject:
        rec[key] = participants.get(key, {"sex": "U", "fam": "UNKNOWN", "age": None, "age_bin": "NA"})
    train_subjects, val_subjects, audit = split_subjects(by_subject, rec, val_fraction, seed)
    if train_subjects & val_subjects:
        raise RuntimeError("Subject leakage detected between train and val — aborting.")
    fold = {
        "train": sorted(p for s in train_subjects for p in by_subject[s]),
        "val": sorted(p for s in val_subjects for p in by_subject[s]),
    }
    return [fold], by_subject, rec, train_subjects, val_subjects, audit


def subject_side_from_reference(reference_path: Path) -> dict[tuple[str, str], str]:
    """Read a reference split (fold 0) and return ``{subject: 'train'|'val'}``.

    Lets derived eligibility-cohort splits (e.g. primary_structural) reuse the master
    subject assignment, so a subject stays on the SAME side across every cohort (no
    cross-cohort leakage), mirroring the old ``--reference-split`` behaviour.
    """
    data = json.loads(Path(reference_path).read_text())
    fold = data[0] if isinstance(data, list) else data
    side: dict[tuple[str, str], str] = {}
    for name in ("train", "val"):
        for p in fold.get(name, []):
            key = parse_subject(p)
            if key is not None:
                side[key] = name
    return side


def build_projected_split(paths: list[str], reference_side: dict[tuple[str, str], str]):
    """Partition *paths* by the reference subject assignment (unknown subjects -> train)."""
    by_subject, unparsed = group_files_by_subject(paths)
    if unparsed:
        LOGGER.warning("%d paths could not be parsed and were skipped", unparsed)
    train, val, unknown = [], [], 0
    for key, files in by_subject.items():
        if reference_side.get(key) == "val":
            val.extend(files)
        else:
            if key not in reference_side:
                unknown += 1
            train.extend(files)
    fold = {"train": sorted(train), "val": sorted(val)}
    return [fold], by_subject, unknown


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leak-free per-dataset stratified 80/20 split.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--paths", type=Path, help="paths.json manifest (list of file paths).")
    src.add_argument("--from-mapping", type=Path, help="mapping.tsv to derive the file list (local dev).")
    p.add_argument(
        "--participants", type=Path, default=None, help="participants.tsv (required unless --reference-split is given)."
    )
    p.add_argument(
        "--reference-split",
        type=Path,
        default=None,
        help="Project an existing master split onto --paths (keeps subjects on the "
        "same side across eligibility cohorts). Skips fresh stratification.",
    )
    p.add_argument("--output", type=Path, required=True, help="Split JSON (list with one fold).")
    p.add_argument("--report", type=Path, default=None, help="Optional split_report.tsv path.")
    p.add_argument("--val-fraction", type=float, default=0.20)
    p.add_argument("--age-bin-width", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    if not 0.0 < args.val_fraction < 1.0:
        raise SystemExit("--val-fraction must be in (0, 1).")

    if args.paths:
        paths = json.loads(Path(args.paths).read_text())
        if not isinstance(paths, list):
            raise SystemExit("--paths must be a JSON list of file paths.")
    else:
        paths = paths_from_mapping(args.from_mapping)
    LOGGER.info("Loaded %d file paths", len(paths))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # ---- Projection mode: reuse a master split's subject assignment ---------------
    if args.reference_split:
        reference_side = subject_side_from_reference(args.reference_split)
        folds, by_subject, unknown = build_projected_split(paths, reference_side)
        args.output.write_text(json.dumps(folds, indent=2))
        n_val = sum(1 for s in by_subject if reference_side.get(s) == "val")
        n_train = len(by_subject) - n_val
        LOGGER.info(
            "Projected %s onto %d subjects (%d train / %d val; %d not in reference -> train) -> %s",
            args.reference_split.name,
            len(by_subject),
            n_train,
            n_val,
            unknown,
            args.output,
        )
        return

    # ---- Master mode: fresh leak-free stratified split ----------------------------
    if args.participants is None:
        raise SystemExit("--participants is required unless --reference-split is given.")
    participants = load_participants(args.participants, args.age_bin_width)
    LOGGER.info("Loaded demographics for %d subjects", len(participants))

    folds, by_subject, rec, train_subjects, val_subjects, audit = build_split(
        paths, participants, args.val_fraction, args.seed
    )
    args.output.write_text(json.dumps(folds, indent=2))

    rows, summary = build_report(by_subject, rec, train_subjects, val_subjects, audit, args.val_fraction)
    if args.report:
        summary_rows = [
            {"axis": "SUMMARY", "key": k, "n_train": "", "n_val": "", "n_total": "", "val_fraction": v}
            for k, v in summary.items()
        ]
        write_tsv(args.report, summary_rows + rows)
        write_tsv(args.report.with_name(args.report.stem + "_by_dataset.tsv"), audit)

    LOGGER.info(
        "Split: %d train / %d val subjects (val_fraction=%.3f); leak_check=%s; max_dev=%.3f",
        summary["n_train_subjects"],
        summary["n_val_subjects"],
        summary["val_fraction_subjects"],
        summary["leak_check"],
        summary["max_abs_val_fraction_deviation"],
    )
    LOGGER.info("Wrote split -> %s", args.output)
    if summary["leak_check"] != "PASS":
        raise SystemExit("LEAK CHECK FAILED")


if __name__ == "__main__":
    main()
