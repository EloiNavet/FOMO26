"""PR8: downstream finetune configs must compose and resolve from committed configs alone.

Guards the config-hygiene fixes: the pretrain-only `momentum_transforms` / `_cpu_demo_tr_transforms`
nodes are no longer inherited by finetune, `training.num_sanity_val_steps` resolves, and post-fit
testing is an explicit `testing.run_after_fit` flag. Resolving the `transforms` and
`lightning._lightning_module` subtrees (rather than the whole config) avoids the `${hydra:run.dir}`
resolver that is only available inside a live Hydra run.
"""

import os
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

for _name, _fn in [("random", lambda a, b: 0), ("version", lambda: "t"), ("eval", eval)]:
    try:
        OmegaConf.register_new_resolver(_name, _fn)
    except Exception:
        pass

os.environ.setdefault("ASPARAGUS_DATA", "/tmp")
os.environ.setdefault("WANDB_ENTITY", "test")
_CFG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
_BASE = "projects/fomo26/finetune"

# (config name, pretrained.enabled expected)
_FINETUNE_CONFIGS = [
    ("task1_lesion_scratch", False),
    ("task1_lesion_amaes_encoder", True),
    ("task3_age_scratch", False),
    ("task3_age_amaes_encoder", True),
]


def _compose(name):
    with initialize_config_dir(version_base="1.2", config_dir=_CFG_DIR):
        return compose(config_name=name)


@pytest.mark.parametrize("name,pretrained_enabled", _FINETUNE_CONFIGS)
def test_finetune_config_composes_without_missing_interpolations(name, pretrained_enabled):
    cfg = _compose(f"{_BASE}/{name}")

    # The two subtrees that previously raised InterpolationKeyError must now resolve cleanly.
    OmegaConf.to_container(cfg.transforms, resolve=True, throw_on_missing=True)
    OmegaConf.to_container(cfg.lightning._lightning_module, resolve=True, throw_on_missing=True)

    # Pretrain-only nodes must NOT be inherited by finetune.
    assert "momentum_transforms" not in cfg.lightning._lightning_module
    assert "_cpu_demo_tr_transforms" not in cfg.transforms

    # num_sanity_val_steps must resolve (shared default from core/base).
    assert isinstance(cfg.training.num_sanity_val_steps, int)

    # Post-fit test is an explicit, default-off flag.
    assert cfg.testing.run_after_fit is False

    # pretrained block present with the expected enable flag.
    assert cfg.pretrained.enabled is pretrained_enabled
    if name.startswith("task3_age_"):
        assert cfg.model.encoder_pool == "max"


def test_fit_patch_size_never_floors_a_thin_axis_to_zero():
    """The post-fit sliding-window crash root cause: a sub-32 axis must not floor to a zero patch."""
    from asparagus.functional.utils import fit_patch_size_to_image_size

    assert fit_patch_size_to_image_size([160, 160, 32], [384, 512, 30]) == [160, 160, 32]
    assert all(p >= 32 for p in fit_patch_size_to_image_size([32, 32, 32], [16, 16, 16]))


def test_pretrain_still_has_momentum_and_cpu_demo_nodes():
    """Non-regression: the moved nodes remain available for pretraining."""
    cfg = _compose("default_pretrain")
    assert "momentum_transforms" in cfg.lightning._lightning_module
    assert "_cpu_demo_tr_transforms" in cfg.transforms
    assert cfg.training.num_sanity_val_steps == 2  # pretrain override preserved
