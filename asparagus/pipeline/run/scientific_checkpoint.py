"""Checkpoint metadata used to approve exact technical resumes."""

from __future__ import annotations

import hashlib
import json
import os
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig, OmegaConf
from pathlib import Path


def _sha256_file(path: str) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(f"scientific initialization checkpoint is absent: {candidate}")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def downstream_scientific_invariant(cfg: DictConfig) -> tuple[dict, str]:
    """Bind the full resolved downstream contract and upstream initialization."""
    initialization = os.environ.get("DS_CHECKPOINT", "")
    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    scientific_config = {
        key: resolved.get(key)
        for key in ("model", "training", "data", "transforms", "lightning", "testing")
        if key in resolved
    }
    invariant = {
        "schema_version": "fomo26-downstream-scientific-invariant-v1",
        "source_git_commit": os.environ.get("DS_SOURCE_GIT_COMMIT", ""),
        "campaign_sha256": os.environ.get("DS_CAMPAIGN_SHA256", ""),
        "campaign_protocol": os.environ.get("DS_CAMPAIGN_PROTOCOL", ""),
        "candidate_id": os.environ.get("FOMO26_CANDIDATE", os.environ.get("DS_CAMPAIGN_OBJECTIVE", "")),
        "architecture": os.environ.get("DS_ARCHITECTURE", ""),
        "objective": os.environ.get("DS_CAMPAIGN_OBJECTIVE", ""),
        "task": os.environ.get("DS_FOMO_TASK", ""),
        "fold": os.environ.get("DS_FOLD", ""),
        "split": os.environ.get("DS_SPLIT", ""),
        "seed": os.environ.get("DS_SEED", str(cfg.training.seed)),
        "initialization_checkpoint": initialization or None,
        "initialization_checkpoint_sha256": _sha256_file(initialization),
        # Deliberately excludes technical resume/output identity fields at the
        # configuration root while retaining optimizer, scheduler, batching,
        # data/sampling, transforms, objective, precision, seed and evaluation.
        "resolved_scientific_config": scientific_config,
    }
    blob = json.dumps(invariant, sort_keys=True, separators=(",", ":"), default=str).encode()
    return invariant, hashlib.sha256(blob).hexdigest()


class DownstreamScientificInvariantCheckpoint(Callback):
    """Embed the invariant and digest into every best/last downstream checkpoint."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.invariant, self.digest = downstream_scientific_invariant(cfg)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint: dict) -> None:
        checkpoint["fomo26_scientific_invariant"] = self.invariant
        checkpoint["fomo26_scientific_invariant_sha256"] = self.digest
