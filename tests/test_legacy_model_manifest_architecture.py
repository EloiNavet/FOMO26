"""Regression: the legacy Tasks 1-5 packaging path must record `architecture`.

`Apptainer.def`'s %post reads `model_manifest["architecture"]` unconditionally to decide whether
the pinned NATTEN wheel is required. The Tasks 1-5 writer omitted the key, so every Tasks 1-5
image failed to build with `KeyError: 'architecture'` while task 6_7 built fine. These tests pin
the key's presence and prove the actual deployment checkpoint still lands under /app/models.
"""

from __future__ import annotations

import json
import re
from finetuning.container import build_container
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class _Args:
    task = "1"
    checkpoint_name = "best"
    architecture = "unet_m"
    candidate = "A"
    _provenance: dict = {}


def test_legacy_model_manifest_records_architecture(tmp_path):
    (tmp_path / "models").mkdir(parents=True)
    records = [
        {
            "fold": 0,
            "best_ckpt": "/app/models/runs/fold0/checkpoints/best.ckpt",
            "checkpoint_sha256": "a" * 64,
            "source_run_dir": "/remote/run",
            "pretrained_checkpoint_sha256": "b" * 64,
        }
    ]
    build_container._write_model_manifest(tmp_path, _Args(), records, records)
    manifest = json.loads((tmp_path / "models" / "model_manifest.json").read_text())

    assert "architecture" in manifest, "Apptainer.def %post reads this key unconditionally"
    assert manifest["architecture"] == "unet_m"


def test_apptainer_post_key_is_actually_present_for_legacy_path(tmp_path):
    """The %post expression must not KeyError against a legacy-written manifest."""
    definition = (REPO / "finetuning" / "container" / "Apptainer.def").read_text()
    key = re.search(r'model_manifest\.json"\)\)\["(\w+)"\]', definition)
    assert key, "Apptainer.def no longer reads a bare key from model_manifest.json"

    (tmp_path / "models").mkdir(parents=True)
    build_container._write_model_manifest(tmp_path, _Args(), [], [])
    manifest = json.loads((tmp_path / "models" / "model_manifest.json").read_text())
    # Mirrors the container's `json.load(...)[key]`, which raises on a missing key.
    assert manifest[key.group(1)] is not None


def test_deployment_checkpoint_still_lands_under_app_models(tmp_path):
    """The packaging fix must not disturb where the real weights are staged."""
    (tmp_path / "models").mkdir(parents=True)
    records = [
        {
            "fold": 0,
            "best_ckpt": "/app/models/runs/fold0/checkpoints/best.ckpt",
            "checkpoint_sha256": "c" * 64,
            "source_run_dir": "/remote/run",
        }
    ]
    build_container._write_model_manifest(tmp_path, _Args(), records, records)
    manifest = json.loads((tmp_path / "models" / "model_manifest.json").read_text())
    assert [m["path"] for m in manifest["models"]] == ["/app/models/runs/fold0/checkpoints/best.ckpt"]
    assert [m["sha256"] for m in manifest["models"]] == ["c" * 64]
