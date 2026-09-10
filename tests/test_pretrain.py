"""Integration test for pipeline/run/pretrain.py components.

Uses SelfSupervisedModule + PretrainDataModule on synthetic volumes.
Torch_CopyImageToLabel adds batch["label"] so the SSL reconstruction loss can run.
Covers: unet_tiny 3D, ResidualEncoderUNet 3D, unet_tiny 2D.
"""

import pytest
import torch
from asparagus.modules.data_modules.pretraining import PretrainDataModule
from asparagus.modules.lightning_modules import SelfSupervisedModule
from asparagus.modules.networks.resenc_unet import resenc_unet_debug
from asparagus.modules.networks.unet import unet_tiny
from asparagus.modules.transforms.foreground_masking import Torch_ForegroundAwareMask
from gardening_tools.modules.transforms.copy_image_to_label import Torch_CopyImageToLabel
from torchvision import transforms
from types import SimpleNamespace


def make_pretrain_data_module(files, **kwargs):
    # CopyImageToLabel saves label = image before any GPU augmentation,
    # which is all the SSL reconstruction loss requires.
    copy_transform = transforms.Compose([Torch_CopyImageToLabel(copy=True)])
    return PretrainDataModule(
        batch_size=1,
        num_workers=1,
        train_split=files["train"],
        val_split=files["val"],
        train_transforms=copy_transform,
        val_transforms=copy_transform,
        **kwargs,
    )


def make_ssl_module(model):
    return SelfSupervisedModule(
        model=model,
        learning_rate=1e-3,
        warmup_epochs=0,
        train_transforms=None,
        val_transforms=None,
    )


def test_pretrain_resenc_unet_fit(pretrain_files, make_trainer):
    """SelfSupervisedModule fits with a tiny ResidualEncoderUNet on 3D synthetic data."""
    model = resenc_unet_debug(
        dimensions="3D",
        input_channels=1,
        output_channels=1,
    )
    make_trainer().fit(make_ssl_module(model), datamodule=make_pretrain_data_module(pretrain_files))


@pytest.mark.parametrize("foreground_mode", ["none", "bonus_voxel", "bonus_patch", "falcon_hard", "falcon_full"])
def test_resenc_amaes_foreground_modes_forward_backward(foreground_mode):
    model = resenc_unet_debug(dimensions="3D", input_channels=1, output_channels=1)
    module = SelfSupervisedModule(
        model=model,
        learning_rate=1e-3,
        mse_foreground_mode=foreground_mode,
        mse_foreground_bonus_weight=0.25,
        mse_foreground_bonus_patch_size=(4, 4, 4),
        mse_foreground_bonus_warmup_steps=0,
        mse_falcon_hard_weight=0.05,
        mse_falcon_spectral_weight=0.0,
        mse_falcon_patch_size=(4, 4, 4),
        mse_falcon_warmup_steps=0,
    )
    module._trainer = SimpleNamespace(global_step=0)
    transform = Torch_ForegroundAwareMask(
        ratio=0.6,
        token_size=(4, 4, 4),
        policy="target_foreground",
        foreground_bias=0.25,
        batched=True,
    )
    image = torch.zeros(1, 1, 16, 16, 16)
    image[..., :8, :, :] = 2.0
    batch = transform({"image": image.clone(), "label": image.clone()})

    pred, _ = module.model.forward_with_features(batch["image"])
    loss = module._rec_loss(pred, batch["label"], batch["mask"])
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


# `test_resenc_amaes_falcon_latent_forward_backward` drove the FALCON latent term through the
# multi-view SSL loss step. That term carries weight 0.0 in the published recipe and belongs to
# the multi-view path this distribution does not ship, so it went with it.


def test_pretrain_unet_2d_fit(pretrain_files_2d, make_trainer):
    """SelfSupervisedModule fits with a 2D UNet on synthetic 2D data."""
    model = unet_tiny(input_channels=1, output_channels=1, dimensions="2D")
    make_trainer().fit(make_ssl_module(model), datamodule=make_pretrain_data_module(pretrain_files_2d))
