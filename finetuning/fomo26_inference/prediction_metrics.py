"""Convert one downstream prediction JSON into challenge-like metrics plus subject rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numpy as np
import os
from finetuning.fomo26_inference.metrics_io import build_record, provenance_problems, write_metrics_json
from pathlib import Path
from statistics import fmean


def _subject_id(path: str) -> str:
    parts = Path(path).parts
    subject_index = next((index for index, part in enumerate(parts) if part.startswith("sub-")), None)
    if subject_index is None:
        return Path(path).stem
    identifiers = [Path(parts[subject_index]).stem]
    if subject_index + 1 < len(parts) and parts[subject_index + 1].startswith("ses-"):
        identifiers.append(parts[subject_index + 1])
    return "_".join(identifiers)


def _mean_finite(values) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(fmean(finite)) if finite else math.nan


def segmentation_record(payload: dict) -> tuple[dict, list[dict], list[dict]]:
    subjects = []
    subject_classes = []
    class_dsc: dict[str, list[float]] = {}
    class_nsd: dict[str, list[float]] = {}
    for path, result in payload.items():
        if path in {"mean", "metrics"} or not isinstance(result, dict):
            continue
        foreground = [(key, value) for key, value in result.items() if str(key) != "0" and isinstance(value, dict)]
        dsc = _mean_finite(value.get("dsc", value.get("dice", math.nan)) for _, value in foreground)
        nsd = _mean_finite(value.get("nsd", math.nan) for _, value in foreground)
        subjects.append({"subject_id": _subject_id(path), "dsc": dsc, "nsd": nsd})
        for key, value in foreground:
            class_dsc_value = float(value.get("dsc", value.get("dice", math.nan)))
            class_nsd_value = float(value.get("nsd", math.nan))
            class_dsc.setdefault(str(key), []).append(class_dsc_value)
            class_nsd.setdefault(str(key), []).append(class_nsd_value)
            subject_classes.append(
                {
                    "subject_id": _subject_id(path),
                    "class_index": int(key),
                    "dsc": class_dsc_value,
                    "nsd": class_nsd_value,
                }
            )
    per_class = {}
    for key, values in class_dsc.items():
        finite_count = sum(math.isfinite(float(value)) for value in values)
        per_class[key] = {
            "dsc": _mean_finite(values),
            "nsd": _mean_finite(class_nsd.get(key, [])),
            "n": finite_count,
            "n_total": len(values),
        }
    return (
        {
            "dsc": _mean_finite(subject["dsc"] for subject in subjects),
            "nsd": _mean_finite(subject["nsd"] for subject in subjects),
            "per_class": per_class,
        },
        subjects,
        subject_classes,
    )


def classification_record(payload: dict) -> tuple[dict[str, float], list[dict]]:
    from sklearn.metrics import f1_score, roc_auc_score

    subjects = []
    for path, result in payload.items():
        if path == "metrics" or not isinstance(result, dict) or "label" not in result:
            continue
        logits = np.asarray(result.get("logits", []), dtype=float)
        if logits.size >= 2:
            logits = logits - logits.max()
            exp_logits = np.exp(logits)
            probability = float(exp_logits[1] / exp_logits.sum())
        else:
            probability = float(result.get("prediction", 0))
        subjects.append(
            {
                "subject_id": _subject_id(path),
                "label": int(result["label"]),
                "prediction": int(result.get("prediction", probability >= 0.5)),
                "probability": probability,
            }
        )
    labels = np.asarray([row["label"] for row in subjects], dtype=int)
    predictions = np.asarray([row["prediction"] for row in subjects], dtype=int)
    probabilities = np.asarray([row["probability"] for row in subjects], dtype=float)
    auroc = float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) > 1 else math.nan
    f1 = float(f1_score(labels, predictions, average="macro", zero_division=0)) if len(labels) else math.nan
    return {"auroc": auroc, "f1": f1}, subjects


def regression_record(payload: dict) -> tuple[dict[str, float], list[dict]]:
    subjects = []
    for path, result in payload.items():
        if path == "metrics" or not isinstance(result, dict) or "label" not in result:
            continue
        subjects.append(
            {
                "subject_id": _subject_id(path),
                "label": float(result["label"]),
                "prediction": float(result["prediction"]),
            }
        )
    labels = np.asarray([row["label"] for row in subjects], dtype=float)
    predictions = np.asarray([row["prediction"] for row in subjects], dtype=float)
    mae = float(np.mean(np.abs(predictions - labels))) if len(labels) else math.nan
    correlation = (
        float(np.corrcoef(predictions, labels)[0, 1])
        if len(labels) > 1 and np.std(labels) > 0 and np.std(predictions) > 0
        else math.nan
    )
    return {"mae": mae, "correlation": correlation}, subjects


def record_from_predictions(
    payload: dict,
    *,
    task: int,
    fold: int,
    task_name: str,
    kind: str,
    status: str = "completed",
    provenance: dict | None = None,
) -> dict:
    if status != "completed":
        return build_record(task, fold, status, {}, task_name=task_name, kind=kind, extra={"subjects": []})
    if kind == "seg":
        metrics, subjects, subject_classes = segmentation_record(payload)
    else:
        builders = {"cls": classification_record, "reg": regression_record}
        metrics, subjects = builders[kind](payload)
        subject_classes = []
    return build_record(
        task,
        fold,
        status,
        metrics,
        task_name=task_name,
        kind=kind,
        extra={"subjects": subjects, "subject_classes": subject_classes, "n_subjects": len(subjects)},
        provenance=provenance,
    )


def _sha256_or_none(path) -> str | None:
    """Digest an artefact, or None when it was not supplied or does not exist.

    A missing artefact yields a visible null rather than an exception: the metrics record must
    still be written for a failed or incomplete run, and its provenance should say plainly that
    the artefact was absent.
    """
    if path is None or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_from_identity(identity: dict, *, predictions: Path) -> dict:
    """Turn a run's ``evaluation_identity.json`` into the record's checkpoint provenance.

    The sidecar is the single source of truth for what was evaluated, so the digest attached
    here is the digest of the file that was actually handed to ``trainer.test`` -- never
    ``best.ckpt`` because it happened to exist, and never nothing at all for ``current``.

    Fails closed when the identity and the predictions being scored disagree: a metrics record
    that quotes one run's identity over another run's numbers is worse than no record.
    """
    declared = identity.get("prediction_path")
    if declared and Path(declared).resolve() != Path(predictions).resolve():
        raise SystemExit(
            f"evaluation identity names prediction file {declared!r} but metrics are being computed "
            f"from {str(predictions)!r}. Refusing to attach one run's identity to another's numbers."
        )
    return {
        "evaluation_id": identity.get("evaluation_id"),
        "evaluation_unit_id": identity.get("evaluation_unit_id"),
        "campaign_id": identity.get("campaign_id"),
        "evaluated_checkpoint_role": identity.get("checkpoint_role"),
        "evaluated_checkpoint_path": identity.get("checkpoint_path"),
        "evaluated_checkpoint_sha256": identity.get("checkpoint_sha256"),
        "evaluated_state": identity.get("evaluated_state"),
        "evaluated_epoch": identity.get("epoch"),
        "evaluated_global_step": identity.get("global_step"),
        "prediction_path": declared,
        "split_id": identity.get("split"),
        "git_sha": identity.get("git_sha"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--kind", choices=("seg", "cls", "reg"), required=True)
    parser.add_argument("--status", default="completed")
    # The canonical provenance source: the sidecar the fine-tuning entrypoint wrote from the
    # same object that selected the evaluated weights. Everything it supplies is taken from it.
    parser.add_argument(
        "--evaluation-identity",
        type=Path,
        help="run's evaluation_identity.json; supplies the evaluated checkpoint, role, state and prediction path",
    )
    # Provenance a launcher knows but the run does not. Optional so a caller that cannot supply
    # a value writes a visible null rather than omitting the field.
    parser.add_argument("--campaign-id")
    parser.add_argument("--evaluation-id")
    parser.add_argument("--evaluation-unit-id")
    parser.add_argument("--source-checkpoint", type=Path, help="pretrained checkpoint the run started from")
    parser.add_argument("--finetuning-checkpoint", type=Path, help="checkpoint produced by fine-tuning")
    parser.add_argument("--split", type=Path, help="split definition used for this evaluation")
    parser.add_argument("--git-sha")
    args = parser.parse_args()

    identity_provenance: dict = {}
    if args.evaluation_identity is not None:
        from asparagus.pipeline.run.evaluation_identity import read_identity

        identity_provenance = provenance_from_identity(read_identity(args.evaluation_identity), predictions=args.predictions)

    provenance = {
        "campaign_id": args.campaign_id,
        "evaluation_id": args.evaluation_id,
        "evaluation_unit_id": args.evaluation_unit_id,
        "source_checkpoint_sha256": _sha256_or_none(args.source_checkpoint),
        "finetuning_checkpoint_sha256": _sha256_or_none(args.finetuning_checkpoint),
        "evaluated_checkpoint_role": None,
        "evaluated_checkpoint_path": None,
        "evaluated_checkpoint_sha256": None,
        "evaluated_state": None,
        "evaluated_epoch": None,
        "evaluated_global_step": None,
        "prediction_path": None,
        "split_id": None,
        "split_sha256": _sha256_or_none(args.split),
        "git_sha": args.git_sha or os.environ.get("FOMO26_LAUNCH_GIT_COMMIT"),
    }
    # The sidecar wins over anything a launcher passed on the command line: it was written by
    # the code that chose the weights, and the launcher was not.
    provenance.update({key: value for key, value in identity_provenance.items() if value is not None})

    payload = json.loads(args.predictions.read_text()) if args.predictions.is_file() else {}
    record = record_from_predictions(
        payload,
        task=args.task,
        fold=args.fold,
        task_name=args.task_name,
        kind=args.kind,
        status=args.status,
        provenance=provenance,
    )
    problems = provenance_problems(record)
    if problems:
        raise SystemExit(
            "refusing to write a completed metrics record without usable provenance:\n  - "
            + "\n  - ".join(problems)
            + f"\nPass --evaluation-identity {{run_dir}}/evaluation_identity.json, or record the "
            f"run's real status instead of {args.status!r}."
        )
    write_metrics_json(args.output, record)
    print(args.output)


if __name__ == "__main__":
    main()
