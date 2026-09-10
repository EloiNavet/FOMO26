"""FOMO26 Task 4 (trigeminal neuralgia, multiclass seg) container entrypoint.

Official I/O: --t2 --output (.nii.gz mask: 0=bg, 1=nerves, 2=vessels, input affine).
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
    ap = argparse.ArgumentParser(description="FOMO26 Task 4: Trigeminal Neuralgia Segmentation")
    ap.add_argument("--t2", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--checkpoint-name", default=CHECKPOINT_NAME)
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--tta", default=TTA, choices=["auto", "none", "flip3", "flip7"])
    ap.add_argument("--max-members", type=int, default=MAX_MEMBERS)
    ap.add_argument("--time-target-s", type=float, default=TIME_TARGET_S)
    ap.add_argument("--calibration-json", default=CALIBRATION_JSON)
    args = ap.parse_args()

    predict_seg(
        data=[args.t2],
        output_path=args.output,
        manifest_records=resolve_manifest_records(args.manifest, checkpoint_name=args.checkpoint_name),
        checkpoint_name=args.checkpoint_name,
        input_channels=1,
        output_channels=3,
        accelerator=args.accelerator,
        tta=args.tta,
        max_members=args.max_members,
        time_target_s=args.time_target_s,
        calibration_json=args.calibration_json,
        ensemble_space=ENSEMBLE_SPACE,
        window_policy=WINDOW_POLICY,
        task="4",
    )


if __name__ == "__main__":
    main()
