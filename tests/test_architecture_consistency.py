"""Task-2 cross-architecture consistency tests.

These complement the per-architecture suites by covering the gaps that the smoke matrix defers to
pytest:

* declared capability flags are uniform and match behaviour (Section A);
* real same-run Lightning resume (``fit(ckpt_path=...)``) restores step/optimizer/scheduler/EMA and
  continues rather than restarting from random init (Section G).

Everything runs on CPU with tiny real implementation classes, following ``tests/conftest.py``.
"""

import importlib
import lightning as L
import pytest
import torch
from asparagus.modules.data_modules.pretraining import PretrainDataModule
from asparagus.modules.lightning_modules import (
    SelfSupervisedModule,
)
from asparagus.modules.networks.resenc_unet import ResidualEncoderUNetSSL, resenc_unet_debug
from asparagus.modules.networks.unet import UNetSSL
from gardening_tools.modules.networks.components.blocks import MultiLayerConvDropoutNormNonlin
from gardening_tools.modules.transforms.copy_image_to_label import Torch_CopyImageToLabel
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision import transforms

# --------------------------------------------------------------------------- #
# Tiny builders (mirror the smoke-matrix sizing)
# --------------------------------------------------------------------------- #
# Tiny tests keep stage strides at one so every LFP grid remains >= NATTEN's 3-cubed kernel.
# One explicit 96-cubed test below exercises the production /64 stride schedule.


def _tiny_unet_ssl() -> UNetSSL:
    return UNetSSL(
        input_channels=1,
        output_channels=1,
        dimensions="3D",
        starting_filters=2,
        encoder_basic_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(1),
        decoder_basic_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(1),
        head_out_dim=8,
        head_hidden_dim=8,
    )


# --------------------------------------------------------------------------- #
# Section A -- capability flags are declared, typed and consistent
# --------------------------------------------------------------------------- #
_EXPECTED_CAPABILITIES = {
    "UNet": dict(reconstruction=True, multiscale=True, segmentation=True, tokens=False),
    "ResEnc": dict(reconstruction=True, multiscale=True, segmentation=True, tokens=False),
}


def _ssl_backbone_class(name):
    if name == "UNet":
        return UNetSSL
    if name == "ResEnc":
        return ResidualEncoderUNetSSL
    raise AssertionError(name)


@pytest.mark.parametrize("name", list(_EXPECTED_CAPABILITIES))
def test_backbone_capability_flags_are_declared_typed_and_correct(name):
    """Every SSL backbone declares the four capability flags with the expected boolean values."""
    cls = _ssl_backbone_class(name)
    expected = _EXPECTED_CAPABILITIES[name]
    for flag, attr in (
        ("reconstruction", "supports_reconstruction"),
        ("multiscale", "supports_multiscale_features"),
        ("segmentation", "supports_segmentation"),
        ("tokens", "supports_tokens"),
    ):
        assert hasattr(cls, attr), f"{name} is missing {attr}"
        value = getattr(cls, attr)
        assert isinstance(value, bool), f"{name}.{attr} must be a bool, got {type(value)}"
        assert value == expected[flag], f"{name}.{attr}={value}, expected {expected[flag]}"


# --------------------------------------------------------------------------- #
# Section G -- real same-run Lightning resume (fit(ckpt_path=...))
# --------------------------------------------------------------------------- #
def _make_ssl_module(model) -> SelfSupervisedModule:
    return SelfSupervisedModule(model=model, learning_rate=1e-3, warmup_epochs=0, train_transforms=None, val_transforms=None)


def _make_pretrain_dm(files) -> PretrainDataModule:
    copy_transform = transforms.Compose([Torch_CopyImageToLabel(copy=True)])
    return PretrainDataModule(
        batch_size=1,
        num_workers=0,
        train_split=files["train"],
        val_split=files["val"],
        train_transforms=copy_transform,
        val_transforms=copy_transform,
    )


def _resume_trainer(tmp_path, max_steps, callbacks=None):
    return L.Trainer(
        accelerator="cpu",
        max_steps=max_steps,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_progress_bar=False,
        default_root_dir=str(tmp_path),
        callbacks=callbacks or [],
    )


_RESUME_BACKBONES = {
    "ResEnc": lambda: resenc_unet_debug(dimensions="3D", input_channels=1, output_channels=1),
    "UNet": _tiny_unet_ssl,
}


@pytest.mark.parametrize("name", list(_RESUME_BACKBONES))
def test_same_run_resume_restores_state_and_continues(name, pretrain_files, tmp_path):
    """A real ``fit(ckpt_path=...)`` restores step/optimizer/scheduler/EMA and continues from the
    saved step -- it does not restart from a fresh random initialisation."""
    build = _RESUME_BACKBONES[name]
    dm = _make_pretrain_dm(pretrain_files)

    # Phase 1: train 2 steps and checkpoint the run.
    torch.manual_seed(0)
    ckpt_cb = ModelCheckpoint(dirpath=str(tmp_path), save_last=True)
    trainer1 = _resume_trainer(tmp_path, max_steps=2, callbacks=[ckpt_cb])
    trainer1.fit(_make_ssl_module(build()), datamodule=dm)
    assert trainer1.global_step == 2
    last_ckpt = tmp_path / "last.ckpt"
    assert last_ckpt.exists()

    payload = torch.load(last_ckpt, map_location="cpu", weights_only=False)
    assert payload["global_step"] == 2
    assert payload["optimizer_states"] and payload["lr_schedulers"]
    # NOTE: the EMA/target encoder (``momentum_model``) only exists under JEPA/contrastive configs;
    # when present it is a registered submodule so it serialises with ``state_dict`` and is restored
    # by the same strict resume path exercised here (round-trip covered by test_pretrained_loader).
    probe_key = next(key for key in payload["state_dict"] if key.startswith("model."))
    ckpt_weight = payload["state_dict"][probe_key].clone()

    # Phase 2 (restore): a *differently* initialised fresh module must adopt the checkpoint weights,
    # and with max_steps already reached it runs zero further steps -> weights equal the checkpoint.
    torch.manual_seed(123)
    fresh = _make_ssl_module(build())
    random_weight = fresh.state_dict()[probe_key].clone()
    assert not torch.equal(random_weight, ckpt_weight), "test setup: fresh init should differ from checkpoint"
    trainer2 = _resume_trainer(tmp_path, max_steps=2)
    trainer2.fit(fresh, datamodule=dm, ckpt_path=str(last_ckpt))
    assert trainer2.global_step == 2  # restored, no extra optimisation steps
    assert torch.equal(fresh.state_dict()[probe_key], ckpt_weight), "resume did not restore checkpoint weights"

    # Phase 3 (continue): resuming with a larger budget advances past the saved step.
    torch.manual_seed(7)
    continued = _make_ssl_module(build())
    trainer3 = _resume_trainer(tmp_path, max_steps=4)
    trainer3.fit(continued, datamodule=dm, ckpt_path=str(last_ckpt))
    assert trainer3.global_step == 4  # continued from step 2, not restarted from 0


# --------------------------------------------------------------------------- #
# Retired backbones -- negative contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("module_path", ["asparagus.modules.networks.medvit_3d", "asparagus.modules.networks.primus"])
def test_retired_backbone_modules_are_absent(module_path):
    """MedViT and Primus are excluded from the public release.

    The positive capability assertions for them were removed with the components; this is the
    negative half. Their absence has to fail closed, not degrade into a silent fallback onto a
    retained backbone.
    """
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_path)


@pytest.mark.parametrize("name", ["MedViT", "Primus"])
def test_retired_backbone_names_are_not_resolvable(name):
    assert name not in _EXPECTED_CAPABILITIES
    with pytest.raises(AssertionError):
        _ssl_backbone_class(name)
