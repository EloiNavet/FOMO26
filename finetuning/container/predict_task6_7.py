"""FOMO26 Tasks 6 & 7 (linear probing / fairness) container entrypoint.

Official I/O: ``--input`` one NIfTI, ``--output`` one 1-D float ``.npy`` embedding.

Tasks 6 and 7 probe FROZEN pretrained representations, so the embedding comes from the pretrained
checkpoint itself -- the same artifact this candidate transfers into every other task, before any
downstream finetuning. No Task 1-5 run directory or finetuned checkpoint takes part.
"""

from __future__ import annotations

import argparse
import os
from finetuning.fomo26_inference.pretrained_embedding import extract_embedding

CHECKPOINT = os.environ.get("FOMO26_PRETRAINED_CHECKPOINT", "/app/models/pretrained/pretrained.ckpt")
ARCHITECTURE = os.environ.get("FOMO26_ARCHITECTURE", "")
SSL_OBJECTIVE = os.environ.get("FOMO26_SSL_OBJECTIVE", "")
CHECKPOINT_SOURCE = os.environ.get("FOMO26_CHECKPOINT_SOURCE", "")
PATCH_SIZE = os.environ.get("FOMO26_PATCH_SIZE", "128,128,128")
MIN_COVERAGE = float(os.environ.get("FOMO26_MIN_ENCODER_COVERAGE", "0.98"))


def _patch_size(text: str) -> tuple[int, int, int]:
    values = [int(v) for v in str(text).replace(" ", "").split(",") if v]
    if len(values) != 3:
        raise ValueError(f"FOMO26_PATCH_SIZE must be three comma-separated integers, got {text!r}")
    return tuple(values)


def main():
    ap = argparse.ArgumentParser(description="FOMO26 Tasks 6 & 7: frozen pretrained embedding")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--architecture", default=ARCHITECTURE or None)
    ap.add_argument("--ssl-objective", default=SSL_OBJECTIVE or None)
    ap.add_argument("--checkpoint-source", default=CHECKPOINT_SOURCE or None)
    ap.add_argument("--patch-size", default=PATCH_SIZE)
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--minimum-encoder-coverage", type=float, default=MIN_COVERAGE)
    args = ap.parse_args()

    if not args.architecture:
        raise SystemExit("Architecture is required: pass --architecture or set FOMO26_ARCHITECTURE in the image.")

    receipt = extract_embedding(
        checkpoint=args.checkpoint,
        architecture=args.architecture,
        input_path=args.input,
        output_path=args.output,
        patch_size=_patch_size(args.patch_size),
        ssl_objective=args.ssl_objective,
        checkpoint_source=args.checkpoint_source,
        accelerator=args.accelerator,
        minimum_encoder_coverage=args.minimum_encoder_coverage,
    )
    print(
        f"[embed] {receipt['embedding_dim']}-d frozen pretrained representation "
        f"(source={receipt['source_used']}, coverage={receipt['encoder_load_fraction']}) -> {args.output}"
    )


if __name__ == "__main__":
    main()
