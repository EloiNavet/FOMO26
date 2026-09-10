"""Unit tests for the decomposed per-objective SSL components (P3.9).

Each objective's pure math is now testable in isolation, without constructing the full
LightningModule or a network. These complement the end-to-end golden snapshots in
``test_ssl_characterization.py``.
"""

import math
import pytest
import torch
import torch.nn as nn
from asparagus.functional.frequency import (
    focal_frequency_weight_metrics,
    frequency_domain_loss,
    spectral_frequency_weight_diagnostics,
    spectral_residual_power_metrics,
)
from asparagus.functional.metrics import masking as masking_metrics
from asparagus.functional.scanner_targets import (
    SCANNER_IGNORE_INDEX,
    ScannerTargetConfig,
    ScannerTargetEncoder,
    canonicalize_field_strength,
    canonicalize_scanner_targets,
    spacing_bin_label,
)
from asparagus.functional.wavelet import masked_wavelet_loss, stationary_haar_coefficients, wavelet_residual_metrics
from asparagus.modules.lightning_modules.base_module import BaseModule
from asparagus.modules.lightning_modules.self_supervised import SelfSupervisedModule
from asparagus.modules.lightning_modules.ssl import schedules
from asparagus.modules.lightning_modules.ssl.objectives import reconstruction
from asparagus.modules.transforms.foreground_masking import Torch_ForegroundAwareMask, foreground_fraction_grid
from asparagus.modules.transforms.frepa_lite import Torch_FrepaLiteFrequencyCorruption
from types import SimpleNamespace


class _OptimizerTestModule(BaseModule):
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError


# --------------------------- schedules ---------------------------
def test_cosine_ramp_endpoints_and_midpoint():
    assert schedules.cosine_ramp(5, start_step=10, warmup_steps=10) == 0.0  # before start
    assert schedules.cosine_ramp(10, start_step=10, warmup_steps=10) == 0.0  # at start
    assert schedules.cosine_ramp(20, start_step=10, warmup_steps=10) == 1.0  # warmup complete
    assert schedules.cosine_ramp(100, start_step=10, warmup_steps=10) == 1.0  # clamped
    assert schedules.cosine_ramp(10, start_step=10, warmup_steps=0) == 1.0  # zero warmup -> on
    assert math.isclose(schedules.cosine_ramp(15, 10, 10), 0.5, abs_tol=1e-9)  # midpoint


def test_cosine_window_ramps_up_and_down():
    assert schedules.cosine_window(5, start_step=10, warmup_steps=10, end_step=30, decay_steps=10) == 0.0
    assert schedules.cosine_window(20, start_step=10, warmup_steps=10, end_step=30, decay_steps=10) == 1.0
    assert schedules.cosine_window(30, start_step=10, warmup_steps=10, end_step=30, decay_steps=10) == 1.0
    assert math.isclose(
        schedules.cosine_window(35, start_step=10, warmup_steps=10, end_step=30, decay_steps=10),
        0.5,
        abs_tol=1e-9,
    )
    assert schedules.cosine_window(40, start_step=10, warmup_steps=10, end_step=30, decay_steps=10) == 0.0
    assert schedules.cosine_window(30, start_step=10, warmup_steps=10, end_step=30, decay_steps=0) == 0.0
    assert schedules.cosine_window(100, start_step=10, warmup_steps=10, end_step=0, decay_steps=0) == 1.0


@pytest.mark.parametrize(
    ("gradient_clip_val", "expected_fused"),
    [(None, True), (0.0, True), (1.0, False)],
)
def test_adamw_disables_fused_mode_when_gradient_clipping_is_active(gradient_clip_val, expected_fused):
    module = _OptimizerTestModule(
        model=nn.Linear(2, 2),
        learning_rate=1e-3,
        warmup_epochs=0,
        optimizer="AdamW",
    )
    module._trainer = SimpleNamespace(
        gradient_clip_val=gradient_clip_val,
        max_epochs=-1,
        max_steps=10,
        limit_train_batches=10,
        accumulate_grad_batches=1,
    )

    optimizers, _ = module.configure_optimizers()

    assert optimizers[0].defaults["fused"] is expected_fused


# --------------------------- reconstruction ---------------------------
def test_rec_loss_masked_only_uses_hidden_region():
    fn = nn.MSELoss(reduction="mean")
    pred = torch.zeros(1, 1, 2, 2, 2)
    y = torch.ones(1, 1, 2, 2, 2)
    mask = torch.ones(1, 1, 2, 2, 2, dtype=torch.bool)  # all visible -> nothing hidden
    mask[0, 0, 0, 0, 0] = False  # one hidden voxel
    # masked loss only sees the single hidden voxel (error 1.0)
    assert torch.isclose(reconstruction.rec_loss(fn, pred, y, mask), torch.tensor(1.0))
    # full loss sees all voxels (also 1.0 here) and the no-mask path matches plain MSE
    assert torch.isclose(reconstruction.rec_loss(fn, pred, y, None), fn(pred, y))


def test_rec_loss_all_visible_returns_zero():
    fn = nn.MSELoss()
    pred, y = torch.zeros(1, 1, 2, 2, 2), torch.ones(1, 1, 2, 2, 2)
    mask = torch.ones(1, 1, 2, 2, 2, dtype=torch.bool)  # nothing hidden
    assert reconstruction.rec_loss(fn, pred, y, mask).item() == 0.0


def test_foreground_aware_mask_targets_foreground_tokens():
    image = torch.zeros(1, 8, 8, 8)
    image[:, :4] = 5.0
    data = {"image": image.clone()}
    torch.manual_seed(0)
    out = Torch_ForegroundAwareMask(
        ratio=0.5,
        token_size=(4, 4, 4),
        policy="target_foreground",
        foreground_bias=1.0,
        batched=False,
    )(data)
    hidden = ~out["mask"]
    assert hidden[:, :4].float().mean().item() == pytest.approx(1.0)
    assert hidden[:, 4:].float().mean().item() == pytest.approx(0.0)
    assert out["image"][hidden].abs().sum().item() == 0.0
    metrics = masking_metrics.compute(out["mask"].unsqueeze(0), image.unsqueeze(0))
    assert metrics["fg_masked_enrichment"] > 0


# --------------------------- Frepa-lite input corruption ---------------------------
def _frepa_band_mse(transform: Torch_FrepaLiteFrequencyCorruption, delta: torch.Tensor, spatial_rank: int):
    return transform._delta_band_mse(delta.float(), spatial_rank)


@pytest.mark.parametrize("enabled", [False, True])
def test_frepa_lite_disabled_or_probability_zero_preserves_image_and_label(enabled):
    image = torch.randn(2, 1, 8, 8, 8)
    label = image.clone() + 1.0
    transform = Torch_FrepaLiteFrequencyCorruption(enabled=enabled, p=0.0, ndim=3)

    out = transform({"image": image.clone(), "label": label.clone()})

    assert torch.equal(out["image"], image)
    assert torch.equal(out["label"], label)
    if enabled:
        assert out["frepa_lite/applied_fraction"].item() == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("shape", "ndim"),
    [
        ((2, 1, 8, 8), 2),
        ((1, 8, 8), 2),
        ((2, 1, 6, 6, 6), 3),
        ((1, 6, 6, 6), 3),
    ],
)
def test_frepa_lite_preserves_shape_dtype_device_and_finiteness(shape, ndim):
    image = torch.randn(*shape, dtype=torch.bfloat16)
    transform = Torch_FrepaLiteFrequencyCorruption(enabled=True, p=1.0, ndim=ndim, high_mask_ratio=0.25)

    out = transform({"image": image.clone()})

    assert out["image"].shape == image.shape
    assert out["image"].dtype == image.dtype
    assert out["image"].device == image.device
    assert torch.isfinite(out["image"].float()).all()
    assert out["frepa_lite/applied_fraction"].item() == pytest.approx(1.0)


def test_frepa_lite_small_dimensions_remain_finite():
    image = torch.randn(2, 1, 1, 1)
    transform = Torch_FrepaLiteFrequencyCorruption(enabled=True, p=1.0, ndim=2, low_noise_std=0.1, high_mask_ratio=0.5)

    out = transform({"image": image.clone()})

    assert torch.isfinite(out["image"]).all()
    assert torch.isfinite(out["frepa_lite/input_delta_mse"])


def test_frepa_lite_low_perturbation_changes_low_band_mostly():
    size = 32
    coords = torch.linspace(0.0, 2.0 * math.pi, size)
    image = torch.sin(coords).view(1, 1, size, 1).expand(1, 1, size, size)
    transform = Torch_FrepaLiteFrequencyCorruption(
        enabled=True,
        p=1.0,
        ndim=2,
        low_scale_min=1.3,
        low_scale_max=1.3,
        low_noise_std=0.0,
        high_mask_ratio=0.0,
    )

    out = transform({"image": image.clone()})
    low_mse, high_mse = _frepa_band_mse(transform, out["image"] - image, spatial_rank=2)

    assert low_mse > high_mse * 100.0
    assert out["frepa_lite/low_delta_mse"].item() > out["frepa_lite/high_delta_mse"].item()


def test_frepa_lite_high_masking_reduces_high_band_energy():
    size = 32
    coords = torch.arange(size)
    pattern = torch.where((coords[:, None] + coords[None, :]) % 2 == 0, 1.0, -1.0)
    image = pattern.view(1, 1, size, size)
    transform = Torch_FrepaLiteFrequencyCorruption(
        enabled=True,
        p=1.0,
        ndim=2,
        low_scale_min=1.0,
        low_scale_max=1.0,
        low_noise_std=0.0,
        high_mask_ratio=1.0,
    )

    out = transform({"image": image.clone()})
    _, high_before = _frepa_band_mse(transform, image, spatial_rank=2)
    _, high_after = _frepa_band_mse(transform, out["image"], spatial_rank=2)

    assert high_after < high_before * 0.01
    assert out["frepa_lite/high_keep_fraction"].item() == pytest.approx(0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"low_cutoff": 0.7, "high_cutoff": 0.6},
        {"p": -0.1},
        {"p": 1.1},
        {"low_scale_min": 0.0},
        {"low_scale_min": 1.2, "low_scale_max": 1.1},
        {"low_noise_std": -0.1},
        {"high_mask_ratio": -0.1},
        {"high_mask_ratio": 1.1},
        {"ndim": 4},
    ],
)
def test_frepa_lite_invalid_arguments_raise_clear_errors(kwargs):
    with pytest.raises(ValueError):
        Torch_FrepaLiteFrequencyCorruption(enabled=True, **kwargs)


def test_frepa_lite_is_deterministic_with_fixed_torch_seed():
    image = torch.randn(2, 1, 8, 8, 8)
    transform = Torch_FrepaLiteFrequencyCorruption(enabled=True, p=1.0, ndim=3)

    torch.manual_seed(123)
    out_a = transform({"image": image.clone()})["image"]
    torch.manual_seed(123)
    out_b = transform({"image": image.clone()})["image"]

    assert torch.equal(out_a, out_b)


def test_foreground_aware_mask_preserve_context_is_opposite_policy():
    image = torch.zeros(1, 8, 8, 8)
    image[:, :4] = 5.0
    torch.manual_seed(0)
    target = Torch_ForegroundAwareMask(
        ratio=0.5,
        token_size=(4, 4, 4),
        policy="target_foreground",
        foreground_bias=1.0,
        batched=False,
    )({"image": image.clone()})["mask"]
    torch.manual_seed(0)
    preserve = Torch_ForegroundAwareMask(
        ratio=0.5,
        token_size=(4, 4, 4),
        policy="preserve_context",
        foreground_bias=1.0,
        batched=False,
    )({"image": image.clone()})["mask"]
    assert (~target)[:, :4].float().mean() > (~preserve)[:, :4].float().mean()


def test_foreground_aware_mask_default_bias_keeps_some_foreground_context():
    image = torch.zeros(1, 16, 16, 16)
    image[:, :8] = 5.0
    torch.manual_seed(0)
    out = Torch_ForegroundAwareMask(ratio=0.6, token_size=(4, 4, 4), policy="target_foreground", batched=False)(
        {"image": image.clone()}
    )
    hidden_fg = (~out["mask"])[:, :8].float().mean().item()
    visible_fg = out["mask"][:, :8].float().mean().item()
    assert hidden_fg > 0.6
    assert visible_fg > 0.0


def test_foreground_fraction_grid_uses_fixed_threshold_when_degenerate():
    image = torch.ones(1, 8, 8, 8)
    fg, grid = foreground_fraction_grid(image, token_size=(4, 4, 4), threshold=2.0)
    assert grid == (2, 2, 2)
    assert fg.sum().item() == 0.0


def test_reconstruction_foreground_weights_use_configurable_threshold():
    image = torch.ones(1, 1, 4, 4, 4)
    weights = reconstruction.compute_foreground_voxel_weights(image, background_weight=0.2, threshold=2.0)
    assert torch.allclose(weights, torch.full_like(weights, 0.2))
    weights = reconstruction.compute_foreground_voxel_weights(image, background_weight=0.2, threshold=0.5)
    assert torch.allclose(weights, torch.ones_like(weights))


def test_foreground_reconstruction_metrics_split_hidden_mse():
    target = torch.zeros(1, 1, 4, 4, 4)
    target[:, :, :2] = 2.0
    pred = torch.zeros_like(target)
    pred[:, :, 2:] = 1.0
    mask = torch.zeros_like(target, dtype=torch.bool)

    metrics = reconstruction.foreground_reconstruction_mse_metrics(pred, target, mask)

    assert torch.isclose(metrics["loss_hidden_fg_unweighted"], torch.tensor(4.0))
    assert torch.isclose(metrics["loss_hidden_bg_unweighted"], torch.tensor(1.0))
    assert torch.isclose(metrics["loss_hidden_unweighted"], torch.tensor(2.5))
    assert torch.isclose(metrics["fg_fraction_hidden_loss"], torch.tensor(0.5))


def test_foreground_reconstruction_metrics_all_visible_returns_zero():
    target = torch.ones(1, 1, 4, 4, 4)
    pred = torch.zeros_like(target)
    mask = torch.ones_like(target, dtype=torch.bool)

    metrics = reconstruction.foreground_reconstruction_mse_metrics(pred, target, mask)

    assert all(value.item() == 0.0 for value in metrics.values())


def test_weighted_rec_loss_normalises_per_sample_by_default():
    pred = torch.zeros(2, 1, 1, 1, 2)
    target = torch.tensor([[[[[2.0, 1.0]]]], [[[[1.0, 1.0]]]]])
    mask = torch.zeros_like(target, dtype=torch.bool)
    weights = torch.tensor([[[[[1.0, 0.1]]]], [[[[0.1, 0.1]]]]])

    loss = reconstruction.weighted_rec_loss(pred, target, mask, weights)
    sample0 = (4.0 * 1.0 + 1.0 * 0.1) / 1.1
    sample1 = (1.0 * 0.1 + 1.0 * 0.1) / 0.2
    assert loss.item() == pytest.approx((sample0 + sample1) / 2.0)

    global_loss = reconstruction.weighted_rec_loss(pred, target, mask, weights, sample_normalize=False)
    assert global_loss.item() == pytest.approx((4.0 * 1.0 + 1.0 * 0.1 + 1.0 * 0.1 + 1.0 * 0.1) / 1.3)


def test_foreground_voxel_bonus_mse_matches_manual_per_sample():
    pred = torch.zeros(2, 1, 1, 1, 2)
    target = torch.tensor([[[[[2.0, 0.0]]]], [[[[0.0, 0.0]]]]])
    mask = torch.zeros_like(target, dtype=torch.bool)

    loss, active_fraction = reconstruction.foreground_voxel_bonus_mse(
        pred,
        target,
        mask,
        threshold=0.5,
        dynamic_quantiles=[0.0, 1.0],
        dynamic_scale=0.0,
    )

    assert loss.item() == pytest.approx(4.0)
    assert active_fraction.item() == pytest.approx(0.5)


def test_foreground_voxel_bonus_mse_empty_foreground_is_zero_and_finite():
    pred = torch.ones(2, 1, 2, 2, 2)
    target = torch.zeros_like(pred)
    mask = torch.zeros_like(pred, dtype=torch.bool)

    loss, active_fraction = reconstruction.foreground_voxel_bonus_mse(pred, target, mask, threshold=1.0)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert active_fraction.item() == 0.0


def test_foreground_patch_bonus_mse_selects_foreground_patches():
    pred = torch.zeros(1, 1, 4, 4, 4)
    target = torch.zeros_like(pred)
    target[..., :2, :2, :2] = 2.0
    mask = torch.zeros_like(pred, dtype=torch.bool)

    loss, active_fraction, patch_fg_fraction = reconstruction.foreground_patch_bonus_mse(
        pred,
        target,
        mask,
        patch_size=(2, 2, 2),
        min_foreground_fraction=0.1,
        threshold=0.5,
        dynamic_quantiles=[0.0, 1.0],
        dynamic_scale=0.0,
    )

    assert loss.item() == pytest.approx(4.0)
    assert active_fraction.item() == pytest.approx(1.0)
    assert patch_fg_fraction.item() == pytest.approx(1.0 / 8.0)


def test_foreground_patch_bonus_mse_requires_divisible_grid():
    with pytest.raises(ValueError, match="must be divisible"):
        reconstruction.foreground_patch_bonus_mse(
            torch.zeros(1, 1, 5, 4, 4),
            torch.zeros(1, 1, 5, 4, 4),
            patch_size=(2, 2, 2),
        )


def test_self_supervised_foreground_mode_none_matches_baseline_mse():
    module = SelfSupervisedModule(model=nn.Identity(), learning_rate=1e-3, mse_foreground_mode="none")
    pred = torch.zeros(1, 1, 1, 1, 2)
    target = torch.tensor([[[[[2.0, 0.0]]]]])
    mask = torch.zeros_like(target, dtype=torch.bool)

    loss = module._rec_loss(pred, target, mask)

    assert torch.isclose(loss, reconstruction.rec_loss(nn.MSELoss(reduction="mean"), pred, target, mask))


def test_self_supervised_bonus_voxel_adds_foreground_term_to_full_hidden_mse():
    module = SelfSupervisedModule(
        model=nn.Identity(),
        learning_rate=1e-3,
        mse_foreground_mode="bonus_voxel",
        mse_foreground_bonus_weight=0.25,
        mse_foreground_bonus_warmup_steps=0,
        mse_foreground_threshold=0.5,
        mse_foreground_dynamic_quantiles=[0.0, 1.0],
        mse_foreground_dynamic_scale=0.0,
    )
    module._trainer = SimpleNamespace(global_step=0)
    pred = torch.zeros(1, 1, 1, 1, 2)
    target = torch.tensor([[[[[2.0, 0.0]]]]])
    mask = torch.zeros_like(target, dtype=torch.bool)

    loss = module._rec_loss(pred, target, mask)

    assert loss.item() == pytest.approx(2.0 + 0.25 * 4.0)


def test_self_supervised_bonus_zero_weight_matches_baseline_mse():
    module = SelfSupervisedModule(
        model=nn.Identity(),
        learning_rate=1e-3,
        mse_foreground_mode="bonus_voxel",
        mse_foreground_bonus_weight=0.0,
    )
    module._trainer = SimpleNamespace(global_step=0)
    pred = torch.zeros(1, 1, 1, 1, 2)
    target = torch.tensor([[[[[2.0, 0.0]]]]])
    mask = torch.zeros_like(target, dtype=torch.bool)

    assert torch.isclose(module._rec_loss(pred, target, mask), reconstruction.rec_loss(nn.MSELoss(), pred, target, mask))


def test_self_supervised_wavelet_schedule_can_window_out():
    module = SelfSupervisedModule(
        model=nn.Identity(),
        learning_rate=1e-3,
        enable_wavelet_loss=True,
        wavelet_start_step=0,
        wavelet_warmup_steps=0,
        wavelet_end_step=10,
        wavelet_decay_steps=10,
    )

    module._trainer = SimpleNamespace(global_step=0)
    assert module.get_dynamic_wavelet_weight() == 1.0
    module._trainer = SimpleNamespace(global_step=15)
    assert module.get_dynamic_wavelet_weight() == pytest.approx(0.5)
    module._trainer = SimpleNamespace(global_step=20)
    assert module.get_dynamic_wavelet_weight() == 0.0


def test_self_supervised_legacy_foreground_aware_maps_to_weighted_replace():
    module = SelfSupervisedModule(
        model=nn.Identity(),
        learning_rate=1e-3,
        mse_foreground_aware=True,
        mse_foreground_mode="none",
        mse_background_weight=0.1,
        mse_foreground_threshold=0.5,
        mse_foreground_dynamic_quantiles=[0.0, 1.0],
        mse_foreground_dynamic_scale=0.0,
    )
    pred = torch.zeros(1, 1, 1, 1, 2)
    target = torch.tensor([[[[[2.0, 0.0]]]]])
    mask = torch.zeros_like(target, dtype=torch.bool)

    loss = module._rec_loss(pred, target, mask)

    assert module.mse_foreground_mode == "weighted_replace"
    assert loss.item() == pytest.approx((4.0 * 1.0 + 0.0 * 0.1) / 1.1)


def test_weighted_loss_gating_and_scaling():
    raw = torch.tensor(2.0)
    assert reconstruction.weighted_loss(raw, 0.5, 0.5, enabled=True).item() == pytest.approx(0.5)
    assert reconstruction.weighted_loss(raw, 0.5, 0.5, enabled=False).item() == 0.0


def test_frequency_wavelet_and_spatial_detail_disabled_return_zero():
    pred, y = torch.randn(1, 1, 4, 4, 4), torch.randn(1, 1, 4, 4, 4)
    assert reconstruction.frequency_loss(pred, y, None, False, 1.0, 2.0).item() == 0.0
    wavelet, diagnostics = reconstruction.wavelet_loss(
        pred,
        y,
        None,
        False,
        family="haar",
        levels=2,
        level_weights=[1.0, 1.0],
        include_lowpass=False,
        loss="l1",
        eps=1e-8,
        return_diagnostics=True,
    )
    assert wavelet.item() == 0.0
    assert diagnostics == {}
    assert reconstruction.spatial_detail(pred, y, None, False, 0.1).item() == 0.0


@pytest.mark.parametrize("shape", [(2, 2, 12, 16), (2, 2, 8, 10, 12)])
def test_masked_wavelet_loss_supports_2d_3d_anisotropic_and_channel_broadcast_mask(shape):
    torch.manual_seed(0)
    pred = torch.randn(*shape, requires_grad=True)
    target = torch.randn_like(pred)
    mask = torch.zeros(shape[0], 1, *shape[2:], dtype=torch.bool)
    mask[..., : shape[-1] // 2] = True

    loss, diagnostics = masked_wavelet_loss(pred, target, mask, return_diagnostics=True)
    loss.backward()

    assert loss.item() > 0.0
    assert set(diagnostics) == {"level1", "level2", "active_support_level1", "active_support_level2"}
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_masked_wavelet_loss_ignores_visible_errors_and_handles_extreme_masks():
    target = torch.zeros(1, 1, 8, 8, 8)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[..., 2:6, 2:6, 2:6] = False
    visible_error = torch.zeros_like(target)
    visible_error[mask] = 10.0
    hidden_error = torch.zeros_like(target)
    hidden_error[..., 3, 3, 3] = 2.0

    assert masked_wavelet_loss(visible_error, target, mask).item() == pytest.approx(0.0, abs=1e-7)
    assert masked_wavelet_loss(hidden_error, target, mask).item() > 0.0
    assert masked_wavelet_loss(hidden_error, target, torch.ones_like(mask)).item() == pytest.approx(0.0)
    assert torch.isfinite(masked_wavelet_loss(hidden_error, target, torch.zeros_like(mask)))


def test_interior_wavelet_support_excludes_artificial_mask_boundaries():
    target = torch.zeros(1, 1, 16, 16)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[..., 4:12, 4:12] = False
    pred = torch.zeros_like(target, requires_grad=True)
    with torch.no_grad():
        pred[..., 4:12, 4:12] = 1.0

    touched = masked_wavelet_loss(pred, target, mask, levels=2, support_mode="touched")
    interior, diagnostics = masked_wavelet_loss(
        pred,
        target,
        mask,
        levels=2,
        support_mode="interior",
        return_diagnostics=True,
    )
    interior.backward()

    assert touched.item() > 0.0
    assert interior.item() == pytest.approx(0.0, abs=1e-6)
    assert diagnostics["active_support_level1"] < 1.0 - mask.float().mean()
    assert diagnostics["active_support_level2"] < diagnostics["active_support_level1"]
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_interior_wavelet_support_ignores_visible_errors_and_detects_hidden_error():
    target = torch.zeros(1, 1, 12, 12, 12)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[..., 2:10, 2:10, 2:10] = False
    visible_error = torch.zeros_like(target)
    visible_error[mask] = 10.0
    hidden_error = torch.zeros_like(target)
    hidden_error[..., 5, 5, 5] = 2.0

    assert masked_wavelet_loss(visible_error, target, mask, support_mode="interior").item() == pytest.approx(0.0)
    assert masked_wavelet_loss(hidden_error, target, mask, support_mode="interior").item() > 0.0


def test_interior_wavelet_support_returns_differentiable_zero_when_empty():
    target = torch.zeros(1, 1, 8, 8)
    pred = torch.zeros_like(target, requires_grad=True)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[..., 4, 4] = False
    with torch.no_grad():
        pred[..., 4, 4] = 1.0

    loss, diagnostics = masked_wavelet_loss(
        pred,
        target,
        mask,
        levels=2,
        support_mode="interior",
        return_diagnostics=True,
    )
    loss.backward()

    assert loss.item() == pytest.approx(0.0)
    assert diagnostics["active_support_level1"].item() == 0.0
    assert diagnostics["active_support_level2"].item() == 0.0
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_detail_only_wavelet_loss_is_zero_for_constant_full_volume_residual():
    pred = torch.full((1, 1, 8, 10, 12), 3.0)
    target = torch.zeros_like(pred)

    detail_only = masked_wavelet_loss(pred, target, levels=2, include_lowpass=False)
    with_lowpass = masked_wavelet_loss(pred, target, levels=2, include_lowpass=True)

    assert detail_only.item() == pytest.approx(0.0, abs=1e-6)
    assert with_lowpass.item() > 0.0


def test_stationary_wavelet_metrics_separate_scale_and_orientation():
    size = 32
    coords = torch.arange(size, dtype=torch.float32)
    low = torch.sin(2 * torch.pi * 2 * coords / size).view(1, 1, 1, size).expand(1, 1, size, size)
    high = torch.sin(2 * torch.pi * 12 * coords / size).view(1, 1, 1, size).expand(1, 1, size, size)
    target = torch.zeros_like(low)

    low_metrics = wavelet_residual_metrics(low, target)
    high_metrics = wavelet_residual_metrics(high, target)
    assert low_metrics["level2"] > low_metrics["level1"]
    assert high_metrics["level1"] > high_metrics["level2"]

    volume = high.unsqueeze(2).expand(1, 1, size, size, size)
    bands = stationary_haar_coefficients(volume, levels=1)[0]
    assert bands["LLH"].abs().mean() > 100 * bands["LHL"].abs().mean()
    assert bands["LLH"].abs().mean() > 100 * bands["HLL"].abs().mean()


def test_wavelet_metrics_keep_weighted_and_unweighted_totals_separate():
    target = torch.zeros(1, 1, 16, 16)
    pred = torch.randn_like(target)
    metrics = wavelet_residual_metrics(pred, target, levels=2, level_weights=[1.0, 2.0])

    assert metrics["detail_total"] == pytest.approx((metrics["level1"] + 2.0 * metrics["level2"]) / 3.0)
    assert metrics["detail_total_unweighted"] == pytest.approx((metrics["level1"] + metrics["level2"]) / 2.0)


def test_masked_wavelet_loss_normalizes_repeated_active_support():
    target = torch.zeros(1, 1, 16, 16)
    one_error = torch.zeros_like(target)
    one_error[..., 4, 4] = 1.0
    one_mask = torch.ones_like(target, dtype=torch.bool)
    one_mask[..., 4, 4] = False

    two_errors = one_error.clone()
    two_errors[..., 11, 11] = 1.0
    two_mask = one_mask.clone()
    two_mask[..., 11, 11] = False

    one_loss = masked_wavelet_loss(one_error, target, one_mask)
    two_loss = masked_wavelet_loss(two_errors, target, two_mask)
    assert torch.isclose(one_loss, two_loss, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"family": "sym7"},
        {"levels": 0},
        {"levels": 2, "level_weights": [1.0]},
        {"level_weights": [1.0, -1.0]},
        {"support_mode": "boundary"},
        {"loss": "mse"},
        {"eps": 0.0},
    ],
)
def test_masked_wavelet_loss_rejects_invalid_configuration(kwargs):
    pred = torch.zeros(1, 1, 8, 8)
    with pytest.raises(ValueError):
        masked_wavelet_loss(pred, pred, **kwargs)


def test_masked_wavelet_loss_has_finite_bfloat16_autocast_gradients():
    pred = torch.randn(1, 1, 8, 8, 8, requires_grad=True)
    target = torch.randn_like(pred)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = masked_wavelet_loss(pred, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("shape", [(2, 1, 8, 8), (2, 1, 4, 5, 6)])
@pytest.mark.parametrize(
    "mode,kwargs",
    [
        ("residual_power_radial", {"weighting": "radial", "high_freq_weight": 0.0}),
        ("focal_frequency", {"weighting": "adaptive", "high_freq_weight": 0.0, "focal_alpha": 0.0}),
        (
            "residual_power_band",
            {"weighting": "band", "low_weight": 1.0, "mid_weight": 1.0, "high_band_weight": 1.0},
        ),
    ],
)
def test_residual_power_frequency_loss_unit_weights_match_hidden_mse(shape, mode, kwargs):
    torch.manual_seed(0)
    pred = torch.randn(*shape)
    target = torch.randn_like(pred)
    mask = torch.zeros_like(pred, dtype=torch.bool)
    mask[..., : shape[-1] // 2] = True

    loss = frequency_domain_loss(pred, target, mask=mask, mode=mode, **kwargs)
    hidden_mse = (pred - target).square()[~mask].mean()

    assert torch.isclose(loss, hidden_mse, rtol=1e-5, atol=1e-6)


def test_residual_power_frequency_loss_ignores_visible_errors_and_handles_empty_hidden_region():
    target = torch.zeros(1, 1, 4, 4, 4)
    visible_error = torch.zeros_like(target)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[..., 2:] = False
    visible_error[mask] = 10.0

    hidden_error = torch.zeros_like(target)
    hidden_error[~mask] = 2.0

    kwargs = dict(mode="residual_power_band", weighting="band", low_weight=1.0, mid_weight=1.0, high_band_weight=1.0)
    assert frequency_domain_loss(visible_error, target, mask=mask, **kwargs).item() == pytest.approx(0.0)
    assert frequency_domain_loss(hidden_error, target, mask=mask, **kwargs).item() == pytest.approx(4.0)
    assert frequency_domain_loss(hidden_error, target, mask=torch.ones_like(mask), **kwargs).item() == pytest.approx(0.0)
    assert torch.isfinite(frequency_domain_loss(hidden_error, target, mask=torch.zeros_like(mask), **kwargs))


def test_focal_frequency_loss_ignores_visible_errors_and_handles_empty_hidden_region():
    target = torch.zeros(1, 1, 4, 4, 4)
    visible_error = torch.zeros_like(target)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[..., 2:] = False
    visible_error[mask] = 10.0

    hidden_error = torch.zeros_like(target)
    hidden_error[~mask] = 2.0

    kwargs = dict(mode="focal_frequency", weighting="adaptive", high_freq_weight=0.0, focal_alpha=0.0)
    assert frequency_domain_loss(visible_error, target, mask=mask, **kwargs).item() == pytest.approx(0.0)
    assert frequency_domain_loss(hidden_error, target, mask=mask, **kwargs).item() == pytest.approx(4.0)
    assert frequency_domain_loss(hidden_error, target, mask=torch.ones_like(mask), **kwargs).item() == pytest.approx(0.0)
    assert torch.isfinite(frequency_domain_loss(hidden_error, target, mask=torch.zeros_like(mask), **kwargs))


@pytest.mark.parametrize("shape", [(2, 1, 8, 8), (2, 1, 4, 5, 6)])
def test_log_focal_frequency_loss_handles_2d_and_3d(shape):
    torch.manual_seed(0)
    pred = torch.randn(*shape)
    target = torch.randn_like(pred)
    mask = torch.zeros_like(pred, dtype=torch.bool)
    mask[..., : shape[-1] // 2] = True

    loss = frequency_domain_loss(
        pred,
        target,
        mask=mask,
        mode="log_focal_frequency",
        weighting="adaptive",
        high_freq_weight=0.0,
        focal_alpha=0.5,
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_log_focal_frequency_loss_ignores_visible_errors_and_handles_empty_hidden_region():
    target = torch.zeros(1, 1, 4, 4, 4)
    visible_error = torch.zeros_like(target)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[..., 2:] = False
    visible_error[mask] = 10.0

    hidden_error = torch.zeros_like(target)
    hidden_error[~mask] = 2.0

    kwargs = dict(mode="log_focal_frequency", weighting="adaptive", high_freq_weight=0.0, focal_alpha=0.5)
    assert frequency_domain_loss(visible_error, target, mask=mask, **kwargs).item() == pytest.approx(0.0)
    assert frequency_domain_loss(hidden_error, target, mask=mask, **kwargs).item() > 0.0
    assert frequency_domain_loss(hidden_error, target, mask=torch.ones_like(mask), **kwargs).item() == pytest.approx(0.0)
    assert torch.isfinite(frequency_domain_loss(hidden_error, target, mask=torch.zeros_like(mask), **kwargs))


def test_spectral_residual_power_metrics_separate_low_and_high_frequency_errors():
    size = 32
    coords = torch.arange(size, dtype=torch.float32)
    low_wave = torch.sin(2 * torch.pi * coords / size).view(1, 1, 1, size).expand(1, 1, size, size)
    high_wave = torch.cos(torch.pi * coords).view(1, 1, 1, size).expand(1, 1, size, size)
    target = torch.zeros_like(low_wave)

    low_metrics = spectral_residual_power_metrics(low_wave, target)
    high_metrics = spectral_residual_power_metrics(high_wave, target)

    assert low_metrics["low"] > low_metrics["high"]
    assert high_metrics["high"] > high_metrics["low"]


def test_focal_frequency_weight_metrics_follow_hard_frequency_band():
    size = 32
    coords = torch.arange(size, dtype=torch.float32)
    low_wave = torch.sin(2 * torch.pi * coords / size).view(1, 1, 1, size).expand(1, 1, size, size)
    high_wave = torch.cos(torch.pi * coords).view(1, 1, 1, size).expand(1, 1, size, size)
    target = torch.zeros_like(low_wave)

    low_metrics = focal_frequency_weight_metrics(low_wave, target, high_freq_weight=0.0, focal_alpha=1.0)
    high_metrics = focal_frequency_weight_metrics(high_wave, target, high_freq_weight=0.0, focal_alpha=1.0)

    assert low_metrics["low_weight_mean"] > low_metrics["high_weight_mean"]
    assert high_metrics["high_weight_mean"] > high_metrics["low_weight_mean"]
    assert torch.isfinite(low_metrics["high_low_weight_ratio"])
    assert torch.isfinite(high_metrics["high_low_weight_ratio"])


def test_log_focal_frequency_weight_metrics_follow_hard_frequency_band():
    size = 32
    coords = torch.arange(size, dtype=torch.float32)
    low_wave = torch.sin(2 * torch.pi * coords / size).view(1, 1, 1, size).expand(1, 1, size, size)
    high_wave = torch.cos(torch.pi * coords).view(1, 1, 1, size).expand(1, 1, size, size)
    target = torch.zeros_like(low_wave)

    low_metrics = focal_frequency_weight_metrics(
        low_wave,
        target,
        high_freq_weight=0.0,
        focal_alpha=1.0,
        log_focal=True,
    )
    high_metrics = focal_frequency_weight_metrics(
        high_wave,
        target,
        high_freq_weight=0.0,
        focal_alpha=1.0,
        log_focal=True,
    )

    assert low_metrics["low_weight_mean"] > low_metrics["high_weight_mean"]
    assert high_metrics["high_weight_mean"] > high_metrics["low_weight_mean"]
    assert low_metrics["low_error_mean"] > low_metrics["high_error_mean"]
    assert high_metrics["high_error_mean"] > high_metrics["low_error_mean"]
    assert torch.isfinite(low_metrics["error_mean"])
    assert torch.isfinite(high_metrics["high_low_weight_ratio"])


def test_spectral_frequency_weight_diagnostics_expose_effective_band_weights():
    diagnostics = spectral_frequency_weight_diagnostics(
        (96, 96, 96),
        mode="residual_power_band",
        weighting="band",
        low_weight=1.0,
        mid_weight=1.0,
        high_band_weight=4.0,
    )

    mass_total = (
        diagnostics["weighting/low_mass_fraction"]
        + diagnostics["weighting/mid_mass_fraction"]
        + diagnostics["weighting/high_mass_fraction"]
    )
    assert mass_total == pytest.approx(1.0)
    assert diagnostics["weighting/high_effective_weight"] > diagnostics["weighting/low_effective_weight"]
    assert diagnostics["weighting/low_effective_weight"] < 1.0


def test_high_only_band_weight_strength_is_controlled_by_loss_weight_not_high_band_weight():
    low_high = spectral_frequency_weight_diagnostics(
        (96, 96, 96),
        mode="residual_power_band",
        weighting="band",
        low_weight=0.0,
        mid_weight=0.0,
        high_band_weight=1.0,
    )
    high_high = spectral_frequency_weight_diagnostics(
        (96, 96, 96),
        mode="residual_power_band",
        weighting="band",
        low_weight=0.0,
        mid_weight=0.0,
        high_band_weight=4.0,
    )

    assert low_high["weighting/low_effective_weight"] == 0.0
    assert low_high["weighting/mid_effective_weight"] == 0.0
    assert low_high["weighting/high_effective_weight"] == pytest.approx(high_high["weighting/high_effective_weight"])
    assert low_high["weighting/high_effective_weight"] > 1.0


def test_spectral_frequency_weight_diagnostics_expose_focal_config():
    diagnostics = spectral_frequency_weight_diagnostics(
        (96, 96, 96),
        mode="focal_frequency",
        weighting="adaptive",
        high_freq_weight=2.0,
        frequency_power=2.0,
        focal_alpha=1.5,
        focal_weight_max=8.0,
    )

    assert diagnostics["weighting/adaptive_focal_alpha"] == pytest.approx(1.5)
    assert diagnostics["weighting/adaptive_focal_weight_max"] == pytest.approx(8.0)
    assert diagnostics["weighting/adaptive_prior_max_weight"] > diagnostics["weighting/adaptive_prior_min_weight"]


def test_spectral_frequency_weight_diagnostics_expose_log_focal_config():
    diagnostics = spectral_frequency_weight_diagnostics(
        (96, 96, 96),
        mode="log_focal_frequency",
        weighting="adaptive",
        high_freq_weight=0.0,
        frequency_power=2.0,
        focal_alpha=0.5,
        focal_weight_max=10.0,
        log_focal_eps=1e-7,
    )

    assert diagnostics["weighting/adaptive_focal_alpha"] == pytest.approx(0.5)
    assert diagnostics["weighting/adaptive_focal_weight_max"] == pytest.approx(10.0)
    assert diagnostics["weighting/adaptive_log_focal_eps"] == pytest.approx(1e-7)


def test_residual_power_frequency_loss_detects_spatial_shift_that_legacy_magnitude_ignores():
    target = torch.zeros(1, 1, 16, 16)
    target[..., 4, 4] = 1.0
    pred = target.roll(shifts=1, dims=-1)

    legacy = frequency_domain_loss(pred, target, mode="legacy_log_magnitude")
    residual = frequency_domain_loss(pred, target, mode="residual_power_radial", weighting="radial", high_freq_weight=0.0)
    focal = frequency_domain_loss(
        pred,
        target,
        mode="focal_frequency",
        weighting="adaptive",
        high_freq_weight=0.0,
        focal_alpha=1.0,
    )
    log_focal = frequency_domain_loss(
        pred,
        target,
        mode="log_focal_frequency",
        weighting="adaptive",
        high_freq_weight=0.0,
        focal_alpha=0.5,
    )

    assert legacy.item() == pytest.approx(0.0, abs=1e-10)
    assert residual.item() > 0.0
    assert focal.item() > 0.0
    assert log_focal.item() > 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "residual_power_radial", "weighting": "radial", "frequency_power": -1.0},
        {"mode": "residual_power_radial", "weighting": "radial", "high_freq_weight": -1.0},
        {"mode": "residual_power_band", "weighting": "band", "low_cutoff": 0.8, "high_cutoff": 0.2},
        {"mode": "residual_power_band", "weighting": "band", "low_weight": -1.0},
        {"mode": "residual_power_radial", "weighting": "band"},
        {"mode": "focal_frequency", "weighting": "radial"},
        {"mode": "focal_frequency", "weighting": "adaptive", "focal_alpha": -1.0},
        {"mode": "focal_frequency", "weighting": "adaptive", "focal_weight_max": 0.0},
        {"mode": "log_focal_frequency", "weighting": "radial"},
        {"mode": "log_focal_frequency", "weighting": "adaptive", "focal_alpha": -1.0},
        {"mode": "log_focal_frequency", "weighting": "adaptive", "focal_weight_max": 0.0},
        {"mode": "log_focal_frequency", "weighting": "adaptive", "log_focal_eps": 0.0},
    ],
)
def test_residual_power_frequency_loss_rejects_invalid_configuration(kwargs):
    pred = torch.zeros(1, 1, 4, 4)
    with pytest.raises(ValueError):
        frequency_domain_loss(pred, pred, **kwargs)


def test_residual_power_frequency_loss_has_finite_float32_and_bfloat16_gradients():
    torch.manual_seed(0)
    target = torch.randn(1, 1, 8, 8)
    pred = torch.randn_like(target, requires_grad=True)

    loss = frequency_domain_loss(pred, target, mode="residual_power_band", weighting="band")
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()

    pred_bf16 = pred.detach().to(torch.bfloat16).requires_grad_(True)
    target_bf16 = target.to(torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss_bf16 = frequency_domain_loss(pred_bf16, target_bf16, mode="residual_power_band", weighting="band")
    loss_bf16.backward()
    assert pred_bf16.grad is not None and torch.isfinite(pred_bf16.grad.float()).all()


def test_focal_frequency_loss_has_finite_float32_and_bfloat16_gradients():
    torch.manual_seed(0)
    target = torch.randn(1, 1, 8, 8)
    pred = torch.randn_like(target, requires_grad=True)

    loss = frequency_domain_loss(
        pred,
        target,
        mode="focal_frequency",
        weighting="adaptive",
        high_freq_weight=0.0,
        focal_alpha=1.0,
    )
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()

    pred_bf16 = pred.detach().to(torch.bfloat16).requires_grad_(True)
    target_bf16 = target.to(torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss_bf16 = frequency_domain_loss(
            pred_bf16,
            target_bf16,
            mode="focal_frequency",
            weighting="adaptive",
            high_freq_weight=0.0,
            focal_alpha=1.0,
        )
    loss_bf16.backward()
    assert pred_bf16.grad is not None and torch.isfinite(pred_bf16.grad.float()).all()


def test_log_focal_frequency_loss_has_finite_float32_and_bfloat16_gradients():
    torch.manual_seed(0)
    target = torch.randn(1, 1, 8, 8)
    pred = torch.randn_like(target, requires_grad=True)

    loss = frequency_domain_loss(
        pred,
        target,
        mode="log_focal_frequency",
        weighting="adaptive",
        high_freq_weight=0.0,
        focal_alpha=0.5,
    )
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()

    pred_bf16 = pred.detach().to(torch.bfloat16).requires_grad_(True)
    target_bf16 = target.to(torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss_bf16 = frequency_domain_loss(
            pred_bf16,
            target_bf16,
            mode="log_focal_frequency",
            weighting="adaptive",
            high_freq_weight=0.0,
            focal_alpha=0.5,
        )
    loss_bf16.backward()
    assert pred_bf16.grad is not None and torch.isfinite(pred_bf16.grad.float()).all()


def test_scanner_target_canonicalization_field_strength_spacing_and_missing():
    row = {
        "manufacturer": "Siemens Healthineers",
        "magneticfieldstrength": "3.0 Tesla",
        "pixdim": "[0.9, 0.9, 1.2]",
    }
    assert canonicalize_scanner_targets(row) == {
        "manufacturer": "siemens",
        "field_strength": "3t",
        "spacing_bin": "(1.0,1.5]",
    }
    assert canonicalize_field_strength("1.494") == "1.5t"
    assert canonicalize_field_strength("unknown") is None
    assert spacing_bin_label("[2.1, 2.2, 3.2]") == ">3.0"
    assert canonicalize_scanner_targets({"manufacturer": "n/a"})["manufacturer"] is None


def test_scanner_target_encoder_uses_train_vocab_and_unseen_validation_is_ignore():
    config = ScannerTargetConfig()
    train_rows = [
        {"manufacturer": "Siemens", "magneticfieldstrength": "3T", "pixdim": "1 1 1"},
        {"manufacturer": "Philips", "magneticfieldstrength": "1.5", "pixdim": "1.6 1.6 1.6"},
    ]
    encoder = ScannerTargetEncoder.fit(train_rows, config)
    assert encoder.encode({"manufacturer": "GE", "magneticfieldstrength": "7T", "pixdim": "4 4 4"}) == {
        "manufacturer": SCANNER_IGNORE_INDEX,
        "field_strength": SCANNER_IGNORE_INDEX,
        "spacing_bin": SCANNER_IGNORE_INDEX,
    }
    assert encoder.encode(train_rows[0])["manufacturer"] == encoder.vocab["manufacturer"]["siemens"]
    assert encoder.class_counts() == {"manufacturer": 2, "field_strength": 2, "spacing_bin": 2}
