"""Generate K-fold cross-validation split files for FOMO26 finetuning tasks.

Asparagus already supports K-fold: a split file is a JSON *list* of fold dicts
``[{"train": [...], "val": [...]}, ...]`` and the finetuning pipeline selects one
via ``data.fold=K`` (see ``experiment_setup.py``: ``load_json(train_split_path)[cfg.data.fold]``).

This script writes such a list for a converted task so that finetuning K folds is just:

    asp_finetune_seg --config-name projects/fomo26/finetune/task2_lesion \
        data.train_split=split_kfold5_holdout70_15_15 data.fold=0   # ... 1, 2, 3, 4

Stratification:
  * classification  -> stratify on the integer class label (source ``label.txt``/``labels.txt``)
  * regression      -> stratify on quantile bins of the target (e.g. brain age)
  * segmentation    -> stratify on lesion presence (mask max > 0); single-class -> plain KFold

Subjects are grouped implicitly: there is exactly one session (``ses-01``) per subject,
so each ``scan.pt`` path is one subject and never straddles train/val.

Example:
    python -m finetuning.make_folds --task SEG902_FOMO26_Task2_lesion --kfolds 5 \
        --holdout-test TEST_70_15_15
    python -m finetuning.make_folds --task REGR903_FOMO26_Task3_age --kfolds 10
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import os
from collections import Counter
from finetuning.prepare_fomo26_asparagus import TASK_SPECS, TaskSpec
from pathlib import Path

# output_name -> (task_key, spec)
_BY_OUTPUT = {spec.output_name: (key, spec) for key, spec in TASK_SPECS.items()}


def resolve_spec(task: str) -> TaskSpec:
    if task in TASK_SPECS:
        return TASK_SPECS[task]
    if task in _BY_OUTPUT:
        return _BY_OUTPUT[task][1]
    raise SystemExit(
        f"Unknown task '{task}'. Use a task key ({', '.join(TASK_SPECS)}) or an output name ({', '.join(_BY_OUTPUT)})."
    )


def subject_of(scan_path: str) -> str:
    # .../<output_name>/sub-XX/ses-01/scan.pt  ->  sub-XX
    return Path(scan_path).parent.parent.name


def load_json(path: Path):
    with open(path) as fh:
        return json.load(fh)


def read_scalar_label(path: Path, cast) -> float:
    return cast(Path(path).read_text().strip().split()[0])


def strat_key_classification(paths: list[str], source_root: Path, spec: TaskSpec) -> list[int]:
    keys = []
    for p in paths:
        label_file = source_root / spec.source_name / "labels" / subject_of(p) / "ses-01" / spec.label_name
        keys.append(int(read_scalar_label(label_file, float)))
    return keys


def strat_key_regression(paths: list[str], source_root: Path, spec: TaskSpec, n_bins: int) -> list[int]:
    values = []
    for p in paths:
        label_file = source_root / spec.source_name / "labels" / subject_of(p) / "ses-01" / spec.label_name
        values.append(read_scalar_label(label_file, float))
    values = np.asarray(values, dtype=float)
    # quantile bin edges; np.digitize -> integer bin index per sample
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
    return [int(b) for b in np.digitize(values, edges)]


def strat_key_segmentation(paths: list[str], raw_labels_root: Path, spec: TaskSpec) -> list[int] | None:
    """Presence (mask max > 0) per subject; None if raw labels unavailable."""
    import nibabel as nib

    keys = []
    for p in paths:
        raw = raw_labels_root / spec.output_name / subject_of(p) / "ses-01" / "scan_label.nii.gz"
        if not raw.is_file():
            return None
        keys.append(int(np.asanyarray(nib.load(str(raw)).dataobj).max() > 0))
    return keys


def make_folds(strat: list[int] | None, n_splits: int, seed: int) -> list[list[int]]:
    """Return a list of n_splits arrays of validation indices."""
    from sklearn.model_selection import KFold, StratifiedKFold

    n = len(strat) if strat is not None else None
    use_strat = strat is not None and min(Counter(strat).values()) >= n_splits and len(set(strat)) >= 2
    if use_strat:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = [val_idx for _, val_idx in splitter.split(np.zeros(len(strat)), strat)]
    else:
        if strat is not None and not use_strat:
            print("  [info] falling back to plain KFold (a class has fewer members than folds).")
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = [val_idx for _, val_idx in splitter.split(np.zeros(n if n else 0))]
    return [list(map(int, f)) for f in folds]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--task", required=True, help="Task key (e.g. task2_lesion) or output name (e.g. SEG902_FOMO26_Task2_lesion)."
    )
    p.add_argument("--kfolds", type=int, default=5, help="Number of CV folds.")
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("ASPARAGUS_DATA", "data/fomo26_finetune_asparagus")),
        help="Asparagus data root ($ASPARAGUS_DATA).",
    )
    p.add_argument(
        "--raw-labels-root",
        type=Path,
        default=Path(os.environ.get("ASPARAGUS_RAW_LABELS", "data/fomo26_finetune_raw_labels")),
        help="Asparagus raw-label root ($ASPARAGUS_RAW_LABELS), for seg stratification.",
    )
    p.add_argument("--source", type=Path, default=Path("finetuning"), help="Folder with Task_1 ... Task_5 (for labels).")
    p.add_argument(
        "--holdout-test",
        default=None,
        help="Name of a TEST split (e.g. TEST_70_15_15) whose subjects are EXCLUDED from the fold pool "
        "and kept as an untouched local test set. Omit to fold over ALL subjects (final-submission mode).",
    )
    p.add_argument("--reg-bins", type=int, default=5, help="Number of quantile bins for regression stratification.")
    p.add_argument("--seed", type=int, default=2606, help="Deterministic fold seed.")
    p.add_argument("--name", default=None, help="Override output filename stem (without .json).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec = resolve_spec(args.task)
    task_dir = args.data_root / spec.output_name
    if not task_dir.is_dir():
        raise SystemExit(f"Missing converted task at {task_dir}. Run prepare_fomo26_asparagus first.")

    paths: list[str] = load_json(task_dir / "paths.json")

    # Exclude held-out test subjects if requested.
    excluded: set[str] = set()
    if args.holdout_test:
        test_file = task_dir / f"{args.holdout_test}.json"
        if not test_file.is_file():
            raise SystemExit(f"Missing test split {test_file}.")
        excluded = {subject_of(p) for p in load_json(test_file)}
    pool = [p for p in paths if subject_of(p) not in excluded]
    if len(pool) < args.kfolds:
        raise SystemExit(f"Pool has {len(pool)} subjects but {args.kfolds} folds requested.")

    # Stratification key.
    if spec.kind == "classification":
        strat = strat_key_classification(pool, args.source, spec)
    elif spec.kind == "regression":
        strat = strat_key_regression(pool, args.source, spec, args.reg_bins)
    elif spec.kind == "segmentation":
        strat = strat_key_segmentation(pool, args.raw_labels_root, spec)
    else:
        strat = None

    val_folds = make_folds(strat, args.kfolds, args.seed)

    # Build the list-of-folds split structure.
    pool_arr = np.asarray(pool)
    all_idx = set(range(len(pool)))
    split = []
    for k, val_idx in enumerate(val_folds):
        val_set = set(val_idx)
        train_idx = sorted(all_idx - val_set)
        split.append(
            {
                "train": [str(pool_arr[i]) for i in train_idx],
                "val": [str(pool_arr[i]) for i in val_idx],
            }
        )

    stem = args.name or (
        f"split_kfold{args.kfolds}" + (f"_holdout{args.holdout_test.replace('TEST_', '')}" if args.holdout_test else "_all")
    )
    out = task_dir / f"{stem}.json"
    with open(out, "w") as fh:
        json.dump(split, fh, indent=2)

    # Report.
    print(f"Wrote {out}")
    print(f"  task={spec.output_name} kind={spec.kind} pool={len(pool)} excluded_test={len(excluded)}")
    for k, fold in enumerate(split):
        if strat is not None:
            val_subj = {subject_of(p) for p in fold["val"]}
            val_keys = [strat[i] for i, p in enumerate(pool) if subject_of(p) in val_subj]
            bal = dict(sorted(Counter(val_keys).items()))
        else:
            bal = "n/a"
        print(f"  fold {k}: train={len(fold['train'])} val={len(fold['val'])} val_strat={bal}")
    print(f"\nUse with:  data.train_split={stem} data.fold=0..{args.kfolds - 1}")


if __name__ == "__main__":
    main()
