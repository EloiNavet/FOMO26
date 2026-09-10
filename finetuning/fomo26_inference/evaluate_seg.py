"""Evaluate a fold-ensemble (and each fold alone) on the local held-out TEST split.

Reports mean DSC + NSD per foreground class so we can verify that
ensemble (+TTA) beats and stabilises single-fold models BEFORE building a submission.

Example:
    python -m finetuning.fomo26_inference.evaluate_seg \
        --manifest $ASPARAGUS_RESULTS/manifests/task2_amaes_mvp.json \
        --task SEG902_FOMO26_Task2_lesion --test-split TEST_70_15_15 --tta flip3
"""

from __future__ import annotations

import argparse
import json
import statistics
from asparagus.functional.collate import collate_return
from asparagus.modules.datasets.TrainDataset import SegTestDataset
from asparagus.modules.transforms.presets import CPU_seg_test_transforms
from asparagus.paths import get_data_path
from finetuning.fomo26_inference.metrics import dsc_nsd
from finetuning.fomo26_inference.seg_ensemble import SegFoldEnsemble
from gardening_tools.functional.paths.read import load_json
from pathlib import Path
from torch.utils.data import DataLoader


def _spacing_from_properties(properties) -> tuple[float, ...] | None:
    p = properties[0] if isinstance(properties, (list, tuple)) else properties
    for key in ("spacing", "itk_spacing", "original_spacing", "spacing_after_resampling"):
        if isinstance(p, dict) and key in p and p[key] is not None:
            return tuple(float(s) for s in p[key])
    return None


def evaluate(
    records,
    task,
    test_split,
    n_modalities,
    n_classes,
    device,
    tta,
    nsd_tol,
    num_workers,
    test_files=None,
    ensemble_space="prob",
):
    task_dir = Path(get_data_path()) / task
    if test_files is None:
        test_files = load_json(str(task_dir / f"{test_split}.json"))
    ens = SegFoldEnsemble(records, n_modalities, n_classes, device=device, tta=tta, ensemble_space=ensemble_space)
    # CPU_seg_test_transforms now requires the patch size (matches the official Task*_predict.py).
    # Spacing comes from the same place for the same reason: evaluating a fold on a geometry it was
    # not trained on measures the mismatch, not the model.
    loader = DataLoader(
        SegTestDataset(
            test_files,
            transforms=CPU_seg_test_transforms(
                patch_size=ens.patch_size,
                runtime_target_spacing=ens.runtime_target_spacing,
            ),
        ),
        batch_size=1,
        num_workers=num_workers,
        collate_fn=collate_return,
    )

    per_class = {c: [] for c in range(1, n_classes)}
    nsd_class = {c: [] for c in range(1, n_classes)}
    for batch in loader:
        out = ens.predict_batch(batch)
        pred = out["src_probs"].argmax(dim=1).cpu()  # [1, *spatial]
        gt = out["src_label"]
        gt = (gt[0] if isinstance(gt, (list, tuple)) else gt).cpu()
        if gt.ndim == pred.ndim + 1:
            gt = gt.squeeze(1)
        spacing = _spacing_from_properties(out["properties"])
        res = dsc_nsd(pred, gt, n_classes, spacing=spacing, nsd_tolerance_mm=nsd_tol)
        for c in range(1, n_classes):
            per_class[c].append(res[c]["dsc"])
            nsd_class[c].append(res[c]["nsd"])
    summary = {}
    for c in range(1, n_classes):
        summary[c] = {
            "dsc": round(statistics.fmean(per_class[c]), 4),
            "nsd": round(statistics.fmean(nsd_class[c]), 4),
            "n": len(per_class[c]),
        }
    return summary


def out_of_fold_evaluate(
    records, task, train_split, n_modalities, n_classes, device, tta, nsd_tol, num_workers, ensemble_space="prob"
):
    """Score every fold on its own validation partition and pool the result.

    Segmentation had no OOF path at all (only cls/reg did), which is why nothing could choose a TTA
    level, a fold subset or an overlap without either guessing or peeking at the holdout. The split
    file is a LIST indexed by fold -- the same contract ``experiment_setup`` uses -- so fold i's
    ``val`` entry is exactly the set of cases fold i never saw.
    """
    task_dir = Path(get_data_path()) / task
    split = load_json(str(task_dir / f"{train_split}.json"))
    if not isinstance(split, list):
        raise SystemExit(f"{train_split}.json must be a list indexed by fold; got {type(split).__name__}.")

    per_fold, pooled_dsc, pooled_nsd = {}, {c: [] for c in range(1, n_classes)}, {c: [] for c in range(1, n_classes)}
    for record in records:
        fold = int(record["fold"])
        if fold >= len(split):
            raise SystemExit(f"Fold {fold} is not present in {train_split}.json ({len(split)} folds).")
        val_files = split[fold]["val"]
        summary = evaluate(
            [record],
            task,
            None,
            n_modalities,
            n_classes,
            device,
            tta,
            nsd_tol,
            num_workers,
            test_files=val_files,
            ensemble_space=ensemble_space,
        )
        per_fold[fold] = summary
        for c in range(1, n_classes):
            # Weight by case count so pooling is the true per-case mean, not a mean of fold means.
            pooled_dsc[c].extend([summary[c]["dsc"]] * summary[c]["n"])
            pooled_nsd[c].extend([summary[c]["nsd"]] * summary[c]["n"])

    pooled = {
        c: {
            "dsc": round(statistics.fmean(pooled_dsc[c]), 4) if pooled_dsc[c] else None,
            "nsd": round(statistics.fmean(pooled_nsd[c]), 4) if pooled_nsd[c] else None,
            "n": len(pooled_dsc[c]),
        }
        for c in range(1, n_classes)
    }
    return {
        "schema_version": "fomo26-oof-seg-v1",
        "task": task,
        "train_split": train_split,
        "tta": tta,
        # Recorded because it changes the measurement: averaging TTA views as probabilities and
        # averaging them as logits are different numbers, and the payload must say which it is.
        "ensemble_space": ensemble_space,
        "folds": sorted(per_fold),
        "per_fold": per_fold,
        "pooled": pooled,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--task", required=True, help="Asparagus task name, e.g. SEG902_FOMO26_Task2_lesion.")
    p.add_argument("--test-split", default="TEST_70_15_15")
    p.add_argument("--tta", default="none", choices=["none", "flip3", "flip7"])
    p.add_argument("--nsd-tol", type=float, default=1.0, help="NSD tolerance in mm.")
    p.add_argument(
        "--ensemble-space",
        default="prob",
        choices=["prob", "logit"],
        help=(
            "Space in which TTA views and folds are averaged, mirroring the container's "
            "FOMO26_ENSEMBLE_SPACE. 'prob' (default) preserves the historical numbers."
        ),
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers. Keep 0 in sandbox/container smoke tests.")
    p.add_argument("--per-fold", action="store_true", help="Also evaluate each fold checkpoint alone.")
    p.add_argument("--output-json", type=Path, default=None, help="Also write a deterministic metrics.json.")
    p.add_argument("--fold", default="ensemble", help="Fold label recorded in metrics.json (default: ensemble).")
    p.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Evaluate only these fold indices (a real subset SELECTOR, unlike --fold which is just "
            "the label written into metrics.json). Use it to compare 1/3/5-fold ensembles."
        ),
    )
    p.add_argument(
        "--oof",
        action="store_true",
        help=(
            "Out-of-fold evaluation: score each fold on ITS OWN validation partition from "
            "--train-split instead of on the shared test split. This is the only honest basis for "
            "choosing TTA level, fold subset or overlap without touching the final holdout."
        ),
    )
    p.add_argument(
        "--train-split",
        default="split_kfold5_holdout70_15_15",
        help="K-fold split whose per-fold val partitions --oof scores.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    records = [r for r in json.loads(args.manifest.read_text()) if r.get("returncode") == 0 and r.get("best_ckpt")]
    if not records:
        raise SystemExit("No successful runs with best.ckpt in manifest.")
    if args.folds is not None:
        wanted = {int(fold) for fold in args.folds}
        records = [r for r in records if int(r.get("fold", -1)) in wanted]
        missing = wanted - {int(r.get("fold", -1)) for r in records}
        if missing:
            raise SystemExit(f"--folds asked for {sorted(wanted)} but the manifest has no {sorted(missing)}.")
    task_dir = Path(get_data_path()) / args.task
    dj = load_json(str(task_dir / "dataset.json"))
    n_mod = dj["dataset_config"]["n_modalities"]
    n_cls = dj["dataset_config"]["n_classes"]

    print(
        f"Task {args.task}: {len(records)} fold(s), n_modalities={n_mod}, n_classes={n_cls}, "
        f"tta={args.tta}, ensemble_space={args.ensemble_space}"
    )

    if args.oof:
        # Each fold scores only the cases it never trained on. Pooling these gives a selection
        # signal that is independent of the final holdout, which stays untouched until the
        # inference recipe is frozen.
        oof = out_of_fold_evaluate(
            records,
            args.task,
            args.train_split,
            n_mod,
            n_cls,
            args.device,
            args.tta,
            args.nsd_tol,
            args.num_workers,
            ensemble_space=args.ensemble_space,
        )
        for fold, summary in sorted(oof["per_fold"].items()):
            detail = " ".join(f"c{c} dsc={v['dsc']} nsd={v['nsd']} (n={v['n']})" for c, v in summary.items())
            print(f"  OOF fold {fold:>2}: {detail}")
        pooled_detail = " ".join(f"c{c} dsc={v['dsc']} nsd={v['nsd']} (n={v['n']})" for c, v in oof["pooled"].items())
        print(f"  OOF POOLED: {pooled_detail}")
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(oof, indent=2, sort_keys=True) + "\n")
        return

    if args.per_fold:
        for r in records:
            s = evaluate(
                [r],
                args.task,
                args.test_split,
                n_mod,
                n_cls,
                args.device,
                args.tta,
                args.nsd_tol,
                args.num_workers,
                ensemble_space=args.ensemble_space,
            )
            print(f"  fold {r['fold']:>2}: " + " ".join(f"c{c} dsc={v['dsc']} nsd={v['nsd']}" for c, v in s.items()))

    s = evaluate(
        records,
        args.task,
        args.test_split,
        n_mod,
        n_cls,
        args.device,
        args.tta,
        args.nsd_tol,
        args.num_workers,
        ensemble_space=args.ensemble_space,
    )
    print("  ENSEMBLE : " + " ".join(f"c{c} dsc={v['dsc']} nsd={v['nsd']} (n={v['n']})" for c, v in s.items()))

    if args.output_json is not None:
        # Additive serialization only; the metric computation above is unchanged.
        from asparagus.pipeline.run.evaluation_identity import executing_git_sha
        from finetuning.fomo26_inference.metrics_io import (
            build_record,
            seg_metrics_with_per_class,
            task_number_from_name,
            write_metrics_json,
        )

        task_num = task_number_from_name(args.task)
        record = build_record(
            task=task_num,
            fold=args.fold,
            status="completed",
            metrics=seg_metrics_with_per_class(s),
            task_name=args.task,
            kind="seg",
            extra={
                "test_split": args.test_split,
                "n_folds": len(records),
                "ensemble_space": args.ensemble_space,
            },
            provenance={
                # A fold ensemble has no single evaluated checkpoint. The role says so
                # explicitly and the singular checkpoint fields stay null rather than being
                # filled with one arbitrary fold's digest.
                "evaluated_checkpoint_role": "fold_ensemble",
                "evaluated_state": "fold_ensemble",
                "evaluated_fold_count": len(records),
                "prediction_path": str(args.output_json),
                "split_id": args.test_split,
                "git_sha": executing_git_sha(),
            },
        )
        write_metrics_json(args.output_json, record)
        print(f"  metrics.json -> {args.output_json}")


if __name__ == "__main__":
    main()
