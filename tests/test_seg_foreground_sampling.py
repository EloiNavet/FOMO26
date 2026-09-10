"""Foreground crop probability must be configurable, and must reach the crop that uses it.

A segmentation target occupying ~1e-5 of the volume is almost never inside a uniformly random
patch, so how often crops are centred on foreground decides whether the network sees the target at
all. That knob was hard-coded; these tests pin both the default (so no existing run changes) and
the plumbing from config to transform.
"""

from __future__ import annotations

import inspect
import pytest
from asparagus.modules.transforms.presets.train import CPU_seg_train_transforms

DEFAULT_P = 0.33


def _crop_of(compose):
    crops = [t for t in compose.transforms if type(t).__name__ == "Torch_Crop"]
    assert len(crops) == 1, f"expected exactly one crop stage, got {[type(t).__name__ for t in compose.transforms]}"
    return crops[0]


def test_the_default_is_unchanged():
    """Every existing segmentation run must compose exactly as before."""
    assert inspect.signature(CPU_seg_train_transforms).parameters["p_oversample_foreground"].default == DEFAULT_P
    assert _crop_of(CPU_seg_train_transforms([64, 64, 64])).p_oversample_foreground == DEFAULT_P


@pytest.mark.parametrize("probability", [0.0, 0.33, 0.75, 1.0])
def test_the_requested_probability_reaches_the_crop(probability):
    compose = CPU_seg_train_transforms([64, 64, 64], p_oversample_foreground=probability)
    assert _crop_of(compose).p_oversample_foreground == probability


def test_the_finetune_entrypoint_passes_the_configured_value():
    """A knob nothing forwards is not a knob."""
    from asparagus.pipeline.run import finetune_seg

    source = inspect.getsource(finetune_seg)
    assert "p_oversample_foreground=cfg.transforms.p_oversample_foreground" in source


def test_the_seg_config_declares_the_default():
    import yaml
    from pathlib import Path

    config = yaml.safe_load(Path("configs/default_finetune_seg.yaml").read_text())
    assert config["transforms"]["p_oversample_foreground"] == DEFAULT_P
