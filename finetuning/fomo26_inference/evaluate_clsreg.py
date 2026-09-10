"""Evaluate a fold-ensemble (and each fold alone) for classification / regression on the local TEST.

cls -> AUROC (challenge metric). reg -> MAE + Pearson r. With --calibrate, fit temperature
(cls) or affine age de-biasing (reg) on out-of-fold predictions and report calibrated metrics too.

Example:
    python -m finetuning.fomo26_inference.evaluate_clsreg \
        --manifest $ASPARAGUS_RESULTS/manifests/task3_amaes.json \
        --task REGR903_FOMO26_Task3_age --kind reg --test-split TEST_70_15_15 --calibrate
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import torch
from asparagus.functional.collate import collate_return
from asparagus.modules.datasets.TrainDataset import ClsRegTestDataset
from asparagus.paths import get_data_path
from finetuning.fomo26_inference.calibration import (
    apply_age_bias,
    apply_temperature,
    fit_age_bias,
    fit_temperature,
)
from finetuning.fomo26_inference.clsreg_ensemble import ClsRegFoldEnsemble
from finetuning.fomo26_inference.cross_patch import CROSS_PATCH_CHOICES, clsreg_inference_transforms
from gardening_tools.functional.paths.read import load_json
from pathlib import Path
from torch.utils.data import DataLoader


def _loader(files, target_size, cross_patch="none", num_workers=0):
    return DataLoader(
        ClsRegTestDataset(files, transforms=clsreg_inference_transforms(target_size, cross_patch=cross_patch)),
        batch_size=1,
        num_workers=num_workers,
        collate_fn=collate_return,
    )


def auroc(scores, labels):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def mae(pred, true):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


def pearson(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return float(np.corrcoef(pred, true)[0, 1])


@torch.no_grad()
def ensemble_over(records, files, n_mod, n_cls, kind, device, tta, cross_patch, num_workers):
    ens = ClsRegFoldEnsemble(records, n_mod, n_cls, kind, device=device, tta=tta, cross_patch=cross_patch)
    loader = _loader(files, ens.target_size, cross_patch=cross_patch, num_workers=num_workers)
    scores, labels = [], []
    for batch in loader:
        out = ens.predict_batch(batch)
        lab = out["label"]
        lab = (lab[0] if isinstance(lab, (list, tuple)) else lab).reshape(-1)[0].item()
        if kind == "cls":
            scores.append(float(out["probs"][0, 1].item()))
        else:
            scores.append(float(out["pred"].reshape(-1)[0].item()))
        labels.append(lab)
    return np.array(scores), np.array(labels)


def _probs_to_logits(probs):
    """Represent an averaged probability vector as logits that ``oof_record`` can consume.

    ``predict_batch`` returns probabilities already averaged over TTA and cross-patch views, while
    the OOF record schema and temperature calibration are both defined on logits. ``log(p)`` is a
    faithful representation rather than a lossy one: softmax is shift-invariant and
    ``log(softmax(z)) == z - logsumexp(z)``, so these pseudo-logits reproduce ``p`` exactly under
    softmax *and* give ``fit_temperature`` the same optimum the true logits would. An average of
    several views has no single true logit vector, so there is nothing more exact to record.
    """
    p = torch.as_tensor(probs).detach().float().cpu().reshape(-1)
    return torch.log(p.clamp_min(1e-12))


@torch.no_grad()
def oof_predictions(
    records,
    split,
    task,
    n_mod,
    n_cls,
    kind,
    device,
    tta,
    num_workers,
    collect_cases=False,
    cross_patch="none",
):
    """Each fold scores its own validation fold -> OOF logits/preds + labels covering the train pool.

    Each fold runs through ``ClsRegFoldEnsemble`` as a single member, which is the same class the
    container uses at predict time, so flip-TTA and cross-patch views are applied here exactly as
    they are in deployment. The previous bare ``module.model(x)`` accepted ``tta`` and ignored it,
    so every OOF record labelled with a non-``none`` policy actually measured the ``none`` policy.
    """
    task_dir = Path(get_data_path()) / task
    folds = load_json(str(task_dir / f"{split}.json"))
    all_logits, all_preds, all_labels, cases = [], [], [], []
    for rec in records:
        # The fold index arrives as whatever the producing manifest recorded -- the Slurm rail
        # writes it as a string. The split file is a list, so it has to be an int either way.
        val_files = folds[int(rec["fold"])]["val"]
        # Ensemble weights describe how folds combine *in a fold ensemble*. In an OOF pool each
        # fold stands alone on its own partition, so a staged 0.0 weight must not delete the
        # predictions that partition depends on.
        member = {k: v for k, v in rec.items() if k != "ensemble_weight"}
        ens = ClsRegFoldEnsemble([member], n_mod, n_cls, kind, device=device, tta=tta, cross_patch=cross_patch)
        loader = _loader(val_files, ens.target_size, cross_patch=cross_patch, num_workers=num_workers)
        for path, batch in zip(val_files, loader):
            out = ens.predict_batch(batch)
            lab = out["label"]
            lab = (lab[0] if isinstance(lab, (list, tuple)) else lab).reshape(-1)[0].item()
            if kind == "cls":
                all_logits.append(_probs_to_logits(out["probs"][0]))
            else:
                all_preds.append(float(out["pred"].reshape(-1)[0].item()))
            all_labels.append(lab)
            if collect_cases:
                # Provenance per case: which fold produced it and from which checkpoint. Without
                # this an OOF pool cannot be audited for the leakage it exists to avoid.
                cases.append({"path": str(path), "fold": rec["fold"], "checkpoint": rec["best_ckpt"], "label": lab})
    if kind == "cls":
        stacked = torch.stack(all_logits)
        return (stacked, torch.tensor(all_labels), cases) if collect_cases else (stacked, torch.tensor(all_labels))
    preds = np.array(all_preds)
    return (preds, np.array(all_labels), cases) if collect_cases else (preds, np.array(all_labels))


def oof_record(kind, raw, labels, cases):
    """Score the OOF pool with the repository's canonical metric definitions, not new ones.

    ``prediction_metrics.classification_record`` / ``regression_record`` are what every other
    FOMO26 metric in this repository comes from; building their payload here keeps the OOF number
    directly comparable to the common-holdout number instead of introducing a second definition.
    """
    from finetuning.fomo26_inference.prediction_metrics import classification_record, regression_record

    payload = {}
    for index, case in enumerate(cases):
        if kind == "cls":
            logits = [float(v) for v in raw[index]]
            probability = float(torch.softmax(torch.as_tensor(raw[index]).detach().float(), 0)[1])
            payload[case["path"]] = {
                "label": int(case["label"]),
                "logits": logits,
                "prediction": int(probability >= 0.5),
            }
        else:
            payload[case["path"]] = {"label": float(case["label"]), "prediction": float(raw[index])}
    metrics, subjects = classification_record(payload) if kind == "cls" else regression_record(payload)
    for subject, case in zip(subjects, cases):
        subject["fold"] = case["fold"]
    return metrics, subjects


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--kind", required=True, choices=["cls", "reg"])
    p.add_argument("--test-split", default="TEST_70_15_15")
    p.add_argument("--tta", default="none", choices=["none", "flip3", "flip7"])
    p.add_argument(
        "--cross-patch",
        default="none",
        choices=CROSS_PATCH_CHOICES,
        help="Deterministic multi-crop views averaged with flip-TTA. Applies to --oof as well.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers. Keep 0 in sandbox/container smoke tests.")
    p.add_argument("--per-fold", action="store_true")
    p.add_argument("--calibrate", action="store_true", help="Fit temperature (cls) / age de-bias (reg) on OOF.")
    p.add_argument("--output-json", type=Path, default=None, help="Also write a deterministic metrics.json.")
    p.add_argument("--fold", default="ensemble", help="Fold label recorded in metrics.json (default: ensemble).")
    p.add_argument(
        "--oof",
        action="store_true",
        help=(
            "Out-of-fold evaluation: score each fold on ITS OWN validation partition from "
            "--train-split instead of on the shared test split. This is the only honest basis for "
            "choosing TTA level, fold subset or calibration without touching the final holdout."
        ),
    )
    p.add_argument(
        "--train-split",
        default="split_kfold5_holdout70_15_15",
        help="K-fold split whose per-fold val partitions --oof scores.",
    )
    p.add_argument("--evaluation-id", default=None, help="Campaign evaluation id recorded in provenance.")
    return p.parse_args()


def _run_oof(args, n_mod, n_cls, records):
    """Pool each fold's own validation predictions into one leakage-free evaluation set."""
    raw, labels, cases = oof_predictions(
        records,
        args.train_split,
        args.task,
        n_mod,
        n_cls,
        args.kind,
        args.device,
        args.tta,
        args.num_workers,
        collect_cases=True,
        cross_patch=args.cross_patch,
    )
    metrics, subjects = oof_record(args.kind, raw, labels, cases)

    seen = [case["path"] for case in cases]
    if len(set(seen)) != len(seen):
        raise SystemExit(
            f"OOF pool contains {len(seen) - len(set(seen))} duplicated case(s): the fold validation "
            "partitions are not disjoint, so this is not an out-of-fold set."
        )
    detail = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    print(f"  OOF POOLED: {detail} (n={len(subjects)}, folds={sorted({c['fold'] for c in cases})})")

    if args.output_json is not None:
        from asparagus.pipeline.run.evaluation_identity import executing_git_sha
        from finetuning.fomo26_inference.metrics_io import build_record, task_number_from_name, write_metrics_json

        record = build_record(
            task=task_number_from_name(args.task),
            fold="oof",
            status="completed",
            metrics=metrics,
            task_name=args.task,
            kind=args.kind,
            extra={
                "train_split": args.train_split,
                "n_folds": len(records),
                "n_oof": len(subjects),
                "tta": args.tta,
                # Recorded because it changes the measurement: a record that names only its TTA
                # cannot be told apart from one taken under a different cross-patch policy.
                "cross_patch": args.cross_patch,
            },
            provenance={
                "evaluated_checkpoint_role": "per_fold_out_of_fold",
                "evaluated_state": "per_fold_out_of_fold",
                "evaluated_fold_count": len(records),
                "evaluation_id": args.evaluation_id,
                "evaluation_unit_id": (f"{args.evaluation_id}__{args.task}__oof" if args.evaluation_id else None),
                "prediction_path": str(args.output_json),
                # Never TEST_70_15_15: an OOF record must not be mistaken for a holdout record.
                "split_id": f"{args.train_split}:val",
                "git_sha": executing_git_sha(),
            },
        )
        record["subjects"] = subjects
        write_metrics_json(args.output_json, record)
        print(f"  OOF metrics.json -> {args.output_json}")
    return None


def main():
    args = parse_args()
    records = [r for r in json.loads(args.manifest.read_text()) if r.get("returncode") == 0 and r.get("best_ckpt")]
    if not records:
        raise SystemExit("No successful runs with best.ckpt in manifest.")
    task_dir = Path(get_data_path()) / args.task
    dj = load_json(str(task_dir / "dataset.json"))
    n_mod = dj["dataset_config"]["n_modalities"]
    n_cls = dj["dataset_config"]["n_classes"]
    if args.oof:
        return _run_oof(args, n_mod, n_cls, records)

    test_files = load_json(str(task_dir / f"{args.test_split}.json"))
    print(
        f"Task {args.task} kind={args.kind}: {len(records)} fold(s), n_mod={n_mod}, "
        f"n_cls={n_cls}, tta={args.tta}, cross_patch={args.cross_patch}"
    )

    if args.per_fold:
        for r in records:
            s, y = ensemble_over(
                [r], test_files, n_mod, n_cls, args.kind, args.device, args.tta, args.cross_patch, args.num_workers
            )
            if args.kind == "cls":
                print(f"  fold {r['fold']:>2}: AUROC={auroc(s, y):.4f}")
            else:
                print(f"  fold {r['fold']:>2}: MAE={mae(s, y):.3f} r={pearson(s, y):.4f}")

    s, y = ensemble_over(
        records, test_files, n_mod, n_cls, args.kind, args.device, args.tta, args.cross_patch, args.num_workers
    )
    if args.kind == "cls":
        auroc_val = auroc(s, y)
        print(f"  ENSEMBLE : AUROC={auroc_val:.4f} (n={len(y)})")
        json_metrics = {"auroc": auroc_val}
    else:
        mae_val, r_val = mae(s, y), pearson(s, y)
        print(f"  ENSEMBLE : MAE={mae_val:.3f} r={r_val:.4f} (n={len(y)})")
        json_metrics = {"mae": mae_val, "correlation": r_val}

    if args.output_json is not None:
        # Additive serialization only; the metric computation above is unchanged.
        from asparagus.pipeline.run.evaluation_identity import executing_git_sha
        from finetuning.fomo26_inference.metrics_io import build_record, task_number_from_name, write_metrics_json

        record = build_record(
            task=task_number_from_name(args.task),
            fold=args.fold,
            status="completed",
            metrics=json_metrics,
            task_name=args.task,
            kind=args.kind,
            extra={"test_split": args.test_split, "n_folds": len(records), "n_test": int(len(y))},
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

    if args.calibrate:
        if args.kind == "cls":
            logits, lab = oof_predictions(
                records,
                records[0]["split"],
                args.task,
                n_mod,
                n_cls,
                "cls",
                args.device,
                args.tta,
                args.num_workers,
                cross_patch=args.cross_patch,
            )
            T = fit_temperature(logits, lab)
            # temperature scaling does not change argmax/AUROC ranking on its own; report T + Brier.
            cal = apply_temperature(logits, T)[:, 1].numpy()
            raw = torch.softmax(logits, 1)[:, 1].numpy()
            brier_raw = float(np.mean((raw - lab.numpy()) ** 2))
            brier_cal = float(np.mean((cal - lab.numpy()) ** 2))
            print(f"  CALIB    : T={T:.3f}  OOF Brier {brier_raw:.4f} -> {brier_cal:.4f}")
        else:
            pred, true = oof_predictions(
                records,
                records[0]["split"],
                args.task,
                n_mod,
                n_cls,
                "reg",
                args.device,
                args.tta,
                args.num_workers,
                cross_patch=args.cross_patch,
            )
            a, b = fit_age_bias(pred, true)
            s_cal = apply_age_bias(s, a, b)
            print(f"  CALIB    : age-bias a={a:.3f} b={b:.2f} -> TEST MAE {mae(s, y):.3f} -> {mae(s_cal, y):.3f}")


if __name__ == "__main__":
    main()
