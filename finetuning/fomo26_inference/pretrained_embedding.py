"""Frozen pretrained-encoder embeddings for the official FOMO26 Tasks 6 and 7.

Tasks 6 and 7 are linear probing and fairness evaluation on FROZEN pretrained representations.
The official container is handed one NIfTI per call and must return one finite 1-D float vector,
and the challenge requires the same pretrained checkpoint a candidate uses everywhere else.

The representation therefore comes from that pretrained checkpoint directly, *before* any
downstream finetuning. A Task 1-5 checkpoint must never take part: its encoder has been trained
on downstream labels, which is precisely what "frozen pretrained embedding" excludes. Routing
extraction through a finetuned run directory also makes the embedding inherit that task's input
modality count and its prediction head, neither of which has anything to do with the
representation.

Every piece of model handling here is the repository's existing primitive:
``build_backbone`` maps an architecture name to its SSL network, ``load_checkpoint_state_dict``
reads the file, ``extract_pretrained_encoder_state`` resolves online/EMA and enforces coverage,
and ``encode_representations(...)["h_global"]`` is the same pooled vector the linear probe,
the SSL validation monitor and the screening extractors all consume.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import torch
from asparagus.modules.datasets.TrainDataset import SingleSubjectPredictDataset
from asparagus.modules.transforms.presets.pretrain import CPU_val_transforms
from asparagus.pipeline.auto_configuration.checkpoint import load_checkpoint_state_dict
from asparagus.pipeline.auto_configuration.pretrained import extract_pretrained_encoder_state
from dataclasses import dataclass, field
from finetuning.fomo26_inference.backbones import build_backbone, known_architectures
from pathlib import Path
from typing import Any

#: Which saved encoder copy an objective's contract transfers from. ``ema_if_available`` is a
#: request the loader resolves against the checkpoint: EMA/target encoder when one was saved,
#: online otherwise. AMAES keeps no EMA target, so it is always online.
SSL_OBJECTIVE_DEFAULT_SOURCE = {"amaes": "online", "jepa": "ema_if_available"}

#: Stem weights whose second dimension is the modality count the pretrained encoder was built for.
#: ``conv.weight`` and ``all_modules.0.weight`` alias one ``nn.Parameter``; either answers.
_STEM_WEIGHT_SUFFIXES = ("encoder.stem.conv1.conv.weight", "encoder.stem.conv1.all_modules.0.weight")

#: The pretrained SSL reconstruction head is never executed here -- ``encode_representations``
#: stops at the encoder. Its width only has to build.
_UNUSED_SSL_HEAD_CHANNELS = 1


def stem_input_channels(state_dict, default: int = 1) -> int:
    """Read the modality count an encoder was built for, off its own stem.

    The channel contract belongs to the model, not to whoever calls it. Asking for a different
    count silently produces a stem the checkpoint cannot populate, which surfaces later as a
    channel mismatch against real data.
    """
    for key, value in (state_dict or {}).items():
        if any(str(key).endswith(suffix) for suffix in _STEM_WEIGHT_SUFFIXES):
            shape = getattr(value, "shape", None)
            if shape is not None and len(shape) >= 2:
                return int(shape[1])
    return default


@dataclass(frozen=True)
class FrozenEncoder:
    """A pretrained encoder plus the provenance that proves which artifact produced it."""

    model: Any
    architecture: str
    checkpoint: str
    requested_source: str
    source_used: str
    input_channels: int
    encoder_keys_expected: int
    encoder_keys_loaded: int
    load_fraction: float
    shape_mismatches: int
    scope_evidence: str
    report_text: str = field(repr=False, default="")

    def provenance(self) -> dict:
        return {
            "architecture": self.architecture,
            "checkpoint": self.checkpoint,
            "requested_source": self.requested_source,
            "source_used": self.source_used,
            "pretrained_input_channels": self.input_channels,
            "encoder_keys_expected": self.encoder_keys_expected,
            "encoder_keys_loaded": self.encoder_keys_loaded,
            "encoder_load_fraction": self.load_fraction,
            "shape_mismatches": self.shape_mismatches,
            "scope_evidence": self.scope_evidence,
            "downstream_finetuned_weights": None,
        }


def resolve_source(ssl_objective: str | None, checkpoint_source: str | None) -> str:
    """Explicit source wins; otherwise the objective's frozen transfer contract decides."""
    if checkpoint_source:
        return str(checkpoint_source)
    if not ssl_objective:
        raise ValueError("Provide --checkpoint-source, or --ssl-objective so the contract can supply one.")
    objective = str(ssl_objective).lower()
    if objective not in SSL_OBJECTIVE_DEFAULT_SOURCE:
        raise ValueError(
            f"Unknown ssl_objective {ssl_objective!r}; known: {sorted(SSL_OBJECTIVE_DEFAULT_SOURCE)}. "
            "Pass --checkpoint-source explicitly for an objective with no recorded default."
        )
    return SSL_OBJECTIVE_DEFAULT_SOURCE[objective]


def load_frozen_pretrained_encoder(
    checkpoint: str | Path,
    architecture: str,
    *,
    ssl_objective: str | None = None,
    checkpoint_source: str | None = None,
    minimum_encoder_coverage: float = 0.98,
    allow_unknown_encoder_scope: bool = False,
) -> FrozenEncoder:
    """Rebuild ``architecture`` and load the pretrained encoder from ``checkpoint``, strictly.

    Only the encoder is transferred (``load_scope=encoder_only``): the SSL reconstruction decoder
    plays no part in a representation, and no downstream task head is constructed at all.
    """
    if architecture not in known_architectures():
        raise ValueError(f"Unknown architecture {architecture!r}; registered: {known_architectures()}.")

    checkpoint = str(checkpoint)
    source = resolve_source(ssl_objective, checkpoint_source)
    raw_state = load_checkpoint_state_dict(checkpoint)
    input_channels = stem_input_channels(raw_state)

    model = build_backbone(architecture, input_channels=input_channels, output_channels=_UNUSED_SSL_HEAD_CHANNELS)
    model_state = model.state_dict()
    prefixes = tuple(getattr(model, "pretrained_backbone_prefixes", ("encoder.",)))
    target_keys = {f"model.{k}" for k in model_state if any(k.startswith(p) for p in prefixes)}
    if not target_keys:
        raise RuntimeError(f"{architecture} exposes no backbone keys under prefixes {prefixes}.")

    pretrained_cfg = {
        "source": source,
        "load_scope": "encoder_only",
        "strict_shapes": True,
        "fail_if_no_encoder_keys_loaded": True,
        "fail_if_missing_encoder_keys": True,
        "allow_unknown_encoder_scope": bool(allow_unknown_encoder_scope),
        "print_key_report": False,
    }
    filtered, report = extract_pretrained_encoder_state(
        raw_state,
        pretrained_cfg,
        target_keys,
        checkpoint_path=checkpoint,
        target_state_shapes={f"model.{k}": v.shape for k, v in model_state.items()},
        backbone_prefixes=prefixes,
    )

    # The loader speaks the Lightning namespace ("model.<...>"); the bare network does not.
    transferable = {k[len("model.") :]: v for k, v in filtered.items() if k.startswith("model.")}
    unexpected = sorted(set(transferable) - set(model_state))
    if unexpected:
        raise RuntimeError(f"{len(unexpected)} pretrained key(s) are unknown to {architecture}: {unexpected[:8]}")
    model.load_state_dict(transferable, strict=False)

    coverage = report.n_encoder_match / len(target_keys)
    if coverage < float(minimum_encoder_coverage):
        raise RuntimeError(
            f"Pretrained encoder coverage {coverage:.4f} is below the required "
            f"{float(minimum_encoder_coverage):.4f} ({report.n_encoder_match}/{len(target_keys)} keys). "
            "Refusing to emit embeddings from a partially initialised encoder."
        )

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return FrozenEncoder(
        model=model,
        architecture=architecture,
        checkpoint=checkpoint,
        requested_source=source,
        source_used=report.source_used,
        input_channels=input_channels,
        encoder_keys_expected=len(target_keys),
        encoder_keys_loaded=report.n_encoder_match,
        load_fraction=round(coverage, 6),
        shape_mismatches=len(report.shape_mismatches),
        scope_evidence=report.scope_evidence,
        report_text=report.format(),
    )


def preprocess_volume(input_path: str | Path, patch_size, expected_channels: int = 1) -> torch.Tensor:
    """One official single-NIfTI case -> the tensor the pretrained encoder was trained on.

    ``CPU_val_transforms`` is the repository's deterministic pretraining-domain inference
    transform: volume-wise z-score, pad up to the patch size, then centre-crop to it. Padding
    covers inputs smaller than a patch and the centre crop covers larger ones, so any input
    geometry resolves without a random draw.
    """
    dataset = SingleSubjectPredictDataset([str(input_path)], transforms=CPU_val_transforms(patch_size))
    image = dataset[0]["image"]
    if image.ndim != 4:
        raise ValueError(f"Expected a [C, D, H, W] volume from {input_path}, got shape {tuple(image.shape)}.")
    if int(image.shape[0]) != int(expected_channels):
        raise ValueError(
            f"{input_path} yielded {int(image.shape[0])} channel(s) but the pretrained encoder expects "
            f"{int(expected_channels)}. The official Tasks 6/7 input is one scalar NIfTI; do not stack, "
            "replicate or drop channels to bridge a mismatch."
        )
    return image.unsqueeze(0).float()


@torch.no_grad()
def embed_volume(frozen: FrozenEncoder, image: torch.Tensor, device: str = "cpu") -> np.ndarray:
    """Pooled global representation of one preprocessed volume, as a finite 1-D float32 vector."""
    model = frozen.model.to(device).eval()
    representations = model.encode_representations(image.to(device), use_modality_conditioning=False)
    embedding = representations["h_global"].detach().reshape(-1).float().cpu().numpy()
    if embedding.ndim != 1 or embedding.size == 0:
        raise ValueError(f"Embedding must be a non-empty 1-D vector, got shape {embedding.shape}.")
    if not np.isfinite(embedding).all():
        raise ValueError("Frozen pretrained embedding contains non-finite values.")
    return embedding.astype(np.float32, copy=False)


def resolve_device(accelerator: str) -> str:
    accelerator = (accelerator or "auto").lower()
    if accelerator in {"auto", "gpu", "cuda"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return accelerator


def extract_embedding(
    *,
    checkpoint: str | Path,
    architecture: str,
    input_path: str | Path,
    output_path: str | Path,
    patch_size,
    ssl_objective: str | None = None,
    checkpoint_source: str | None = None,
    accelerator: str = "auto",
    minimum_encoder_coverage: float = 0.98,
    frozen: FrozenEncoder | None = None,
) -> dict:
    """Full official Tasks 6/7 call: one NIfTI in, one 1-D ``.npy`` embedding out."""
    if frozen is None:
        frozen = load_frozen_pretrained_encoder(
            checkpoint,
            architecture,
            ssl_objective=ssl_objective,
            checkpoint_source=checkpoint_source,
            minimum_encoder_coverage=minimum_encoder_coverage,
        )
    image = preprocess_volume(input_path, patch_size, expected_channels=frozen.input_channels)
    embedding = embed_volume(frozen, image, device=resolve_device(accelerator))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embedding)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "embedding_dim": int(embedding.shape[0]),
        "dtype": str(embedding.dtype),
        "patch_size": [int(v) for v in patch_size],
        "preprocessing": "CPU_val_transforms: volume_wise_znorm -> pad_to_patch -> center_crop",
        **frozen.provenance(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="PRETRAINED SSL checkpoint (never a finetuned run)")
    parser.add_argument("--architecture", required=True, choices=known_architectures())
    parser.add_argument("--ssl-objective", default=None, help="amaes | jepa (supplies the default source)")
    parser.add_argument("--checkpoint-source", default=None, choices=[None, "online", "ema", "ema_if_available"])
    parser.add_argument("--input", required=True, help="one input NIfTI (official Tasks 6/7 contract)")
    parser.add_argument("--output", required=True, help="destination .npy for the 1-D embedding")
    parser.add_argument("--patch-size", nargs=3, type=int, required=True)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--minimum-encoder-coverage", type=float, default=0.98)
    parser.add_argument("--report-json", default="")
    args = parser.parse_args(argv)

    receipt = extract_embedding(
        checkpoint=args.checkpoint,
        architecture=args.architecture,
        input_path=args.input,
        output_path=args.output,
        patch_size=tuple(args.patch_size),
        ssl_objective=args.ssl_objective,
        checkpoint_source=args.checkpoint_source,
        accelerator=args.accelerator,
        minimum_encoder_coverage=args.minimum_encoder_coverage,
    )
    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
