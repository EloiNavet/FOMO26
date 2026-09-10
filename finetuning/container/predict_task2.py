"""FOMO26 Task 2 (meningioma segmentation) container entrypoint.

Official I/O: --flair --dwi --t2s/--swi --output (binary mask .nii.gz, input affine).
Ensembles the fold checkpoints in MANIFEST. Channel order matches our prepare: (dwi, flair, susc).
"""

from __future__ import annotations

import argparse
import os
from finetuning.container.fomo_ensemble_predict import predict_seg, resolve_manifest_records

MANIFEST = os.environ.get("FOMO26_MANIFEST", "/app/models/manifest.json")
CHECKPOINT_NAME = os.environ.get("FOMO26_CHECKPOINT_NAME", "best")
TTA = os.environ.get("FOMO26_TTA", "auto")
MAX_MEMBERS = int(os.environ.get("FOMO26_MAX_MEMBERS", "0"))
TIME_TARGET_S = float(os.environ.get("FOMO26_TIME_TARGET_S", "115"))
CALIBRATION_JSON = os.environ.get("FOMO26_CALIBRATION_JSON", "/app/models/calibration.json")
ENSEMBLE_SPACE = os.environ.get("FOMO26_ENSEMBLE_SPACE", "prob")
WINDOW_POLICY = os.environ.get("FOMO26_WINDOW_POLICY", "checkpoint_config_overlap_0.5")


def main():
    ap = argparse.ArgumentParser(description="FOMO26 Task 2: Meningioma Segmentation")
    ap.add_argument("--flair", required=True)
    ap.add_argument("--dwi", required=True)
    ap.add_argument("--t2s", required=False)
    ap.add_argument("--swi", required=False)
    ap.add_argument("--output", required=True)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--checkpoint-name", default=CHECKPOINT_NAME)
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--tta", default=TTA, choices=["auto", "none", "flip3", "flip7"])
    ap.add_argument("--max-members", type=int, default=MAX_MEMBERS)
    ap.add_argument("--time-target-s", type=float, default=TIME_TARGET_S)
    ap.add_argument("--calibration-json", default=CALIBRATION_JSON)
    args = ap.parse_args()

    susc = args.t2s if args.t2s is not None else args.swi
    if susc is None:
        raise SystemExit("Task 2 needs one susceptibility image: pass --t2s or --swi.")
    data = [args.dwi, args.flair, susc]  # our prepare order: (dwi_b1000, flair, susceptibility)

    predict_seg(
        data=data,
        output_path=args.output,
        manifest_records=resolve_manifest_records(args.manifest, checkpoint_name=args.checkpoint_name),
        checkpoint_name=args.checkpoint_name,
        input_channels=3,
        output_channels=2,
        accelerator=args.accelerator,
        tta=args.tta,
        max_members=args.max_members,
        time_target_s=args.time_target_s,
        calibration_json=args.calibration_json,
        ensemble_space=ENSEMBLE_SPACE,
        window_policy=WINDOW_POLICY,
        task="2",
    )


if __name__ == "__main__":
    main()
