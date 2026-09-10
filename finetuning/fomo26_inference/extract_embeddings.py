"""Frozen-backbone embedding extraction for FOMO26 Tasks 6 & 7 (deterministic, batched).

Reads the SAME pretrained checkpoint this candidate transfers into every other task, before any
downstream finetuning — Tasks 6 and 7 probe frozen pretrained representations, so no Task 1-5 run
directory or finetuned checkpoint may take part. The single-case official contract lives in
``pretrained_embedding.extract_embedding``; this module adds a deterministic multi-sample output
contract on top of it:

    embeddings.npy          # float array, shape (N, D), one row per input in sorted order
    sample_ids.csv          # header: row,sample_id  (stable id per row, same order as rows)
    embedding_manifest.json # shape, dtype, checkpoint identity + SHA256, source, coverage, ...

No local claim is made about official Task 6/7 performance (labels are platform-hidden).

The output-writing core (``write_embedding_outputs``) is pure and unit-tested; the actual
extraction imports torch lazily so this module loads on a login node.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import numpy as np
import time
from pathlib import Path

SCHEMA_VERSION = "fomo26-embeddings-v1"
NO_PERF_CLAIM = (
    "No official Task 6/7 performance is claimed locally; the platform trains linear probes on these "
    "embeddings against hidden labels. This file only records the frozen backbone embeddings."
)


def sha256_file(path: Path) -> str | None:
    if not path or not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_embedding_outputs(
    output_dir: str | Path,
    embeddings: np.ndarray,
    sample_ids: list[str],
    checkpoint_sha256: str | None = None,
    provenance: dict | None = None,
) -> dict:
    """Write embeddings.npy + sample_ids.csv + embedding_manifest.json deterministically. Pure I/O.

    Enforces one row per sample id, unique ids, and matching lengths.
    """
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D (N, D); got shape {embeddings.shape}")
    if len(sample_ids) != embeddings.shape[0]:
        raise ValueError(f"sample_ids ({len(sample_ids)}) must match embedding rows ({embeddings.shape[0]})")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must be unique (one stable id per embedding row)")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_path = output_dir / "embeddings.npy"
    ids_path = output_dir / "sample_ids.csv"
    manifest_path = output_dir / "embedding_manifest.json"

    np.save(emb_path, embeddings)
    with ids_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row", "sample_id"])
        for row, sid in enumerate(sample_ids):
            writer.writerow([row, sid])

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "num_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
        "shape": list(embeddings.shape),
        "embeddings_file": emb_path.name,
        "sample_ids_file": ids_path.name,
        "ordering": "sample_ids.csv row order matches embeddings.npy rows (sorted by input path)",
        "checkpoint_sha256": checkpoint_sha256,
        "backbone_frozen": True,
        "note": NO_PERF_CLAIM,
        **(provenance or {}),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _discover_inputs(args: argparse.Namespace) -> list[Path]:
    inputs: list[Path] = []
    if args.input_dir:
        root = Path(args.input_dir)
        for pattern in args.glob:
            inputs.extend(root.rglob(pattern))
    if args.input_list:
        for line in Path(args.input_list).read_text().splitlines():
            line = line.strip()
            if line:
                inputs.append(Path(line))
    inputs.extend(Path(p) for p in args.inputs)
    # Deterministic ordering by resolved absolute path.
    return sorted({p.resolve() for p in inputs})


def _sample_id(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii", ".npy", ".pt"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="*", default=[], help="Explicit input volumes.")
    p.add_argument("--input-dir", default=None, help="Directory to search for input volumes.")
    p.add_argument("--glob", nargs="+", default=["*.nii.gz"], help="Glob(s) under --input-dir.")
    p.add_argument("--input-list", default=None, help="Text file with one input path per line.")
    p.add_argument("--checkpoint", required=True, help="PRETRAINED SSL checkpoint (never a finetuned run).")
    p.add_argument("--architecture", required=True, help="resenc_b | unet_m | ...")
    p.add_argument("--ssl-objective", default=None, help="amaes | jepa (supplies the default source).")
    p.add_argument("--checkpoint-source", default=None, choices=[None, "online", "ema", "ema_if_available"])
    p.add_argument("--patch-size", nargs=3, type=int, required=True)
    p.add_argument("--minimum-encoder-coverage", type=float, default=0.98)
    p.add_argument("--output-dir", required=True, help="Where embeddings.npy / sample_ids.csv / manifest go.")
    p.add_argument("--accelerator", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    inputs = _discover_inputs(args)
    if not inputs:
        raise SystemExit("No input volumes found. Provide --inputs / --input-dir / --input-list.")

    sample_ids = [_sample_id(p) for p in inputs]
    if len(set(sample_ids)) != len(sample_ids):
        raise SystemExit("Derived sample ids are not unique; ensure input basenames are distinct.")

    checkpoint = Path(args.checkpoint)
    ckpt_sha = sha256_file(checkpoint)

    # Lazy import: extraction needs torch; keep module importable without it.
    from finetuning.fomo26_inference.pretrained_embedding import (
        embed_volume,
        load_frozen_pretrained_encoder,
        preprocess_volume,
        resolve_device,
    )

    # Load the frozen encoder once: every row must come from the identical parameters.
    frozen = load_frozen_pretrained_encoder(
        checkpoint,
        args.architecture,
        ssl_objective=args.ssl_objective,
        checkpoint_source=args.checkpoint_source,
        minimum_encoder_coverage=args.minimum_encoder_coverage,
    )
    device = resolve_device(args.accelerator)
    patch_size = tuple(args.patch_size)

    rows = []
    for path in inputs:
        image = preprocess_volume(path, patch_size, expected_channels=frozen.input_channels)
        rows.append(embed_volume(frozen, image, device=device))

    embeddings = np.stack(rows, axis=0)
    manifest = write_embedding_outputs(
        output_dir=args.output_dir,
        embeddings=embeddings,
        sample_ids=sample_ids,
        checkpoint_sha256=ckpt_sha,
        provenance={
            "patch_size": [int(v) for v in patch_size],
            "preprocessing": "CPU_val_transforms: volume_wise_znorm -> pad_to_patch -> center_crop",
            **frozen.provenance(),
        },
    )
    print(f"Wrote {manifest['num_samples']} x {manifest['embedding_dim']} embeddings to {args.output_dir}")


if __name__ == "__main__":
    main()
