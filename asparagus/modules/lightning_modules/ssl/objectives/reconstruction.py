"""Reconstruction objectives: masked MSE, spectral, wavelet, and spatial detail."""

import math
import torch
import torch.nn.functional as F
from asparagus.functional.frequency import frequency_domain_loss, spatial_detail_loss
from asparagus.functional.wavelet import masked_wavelet_loss


def rec_loss(rec_loss_fn, pred, y, mask=None):
    """Masked (hidden-region only) or full MSE reconstruction loss."""
    if tuple(pred.shape) != tuple(y.shape):
        raise ValueError(
            f"Reconstruction expects an exact prediction/target shape match, got {tuple(pred.shape)} and {tuple(y.shape)}."
        )
    if mask is not None:
        hidden = ~mask.to(dtype=torch.bool)
        if not hidden.any():
            return pred.sum() * 0.0
        return rec_loss_fn(pred[hidden], y[hidden])
    return rec_loss_fn(pred, y)


def foreground_voxel_mask(
    x: torch.Tensor,
    threshold: float = 0.0,
    dynamic_quantiles: tuple[float, float] | list[float] = (0.02, 0.98),
    dynamic_scale: float = 0.1,
) -> torch.Tensor:
    """Voxel-level foreground mask from robust per-sample thresholds."""

    if len(dynamic_quantiles) != 2:
        raise ValueError("dynamic_quantiles must contain exactly two values.")
    q_low, q_high = float(dynamic_quantiles[0]), float(dynamic_quantiles[1])
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError(f"Invalid dynamic_quantiles={dynamic_quantiles!r}; expected 0 <= low < high <= 1.")
    vol = x.abs().amax(dim=1, keepdim=True).float()
    b = vol.shape[0]
    flat = vol.flatten(1)
    stride = max(1, flat.shape[1] // 100000)
    sub = flat[:, ::stride]
    p_low = torch.quantile(sub, q_low, dim=1)
    p_high = torch.quantile(sub, q_high, dim=1)
    dyn_thr = float(dynamic_scale) * (p_high - p_low) + p_low
    fixed_thr = torch.full_like(dyn_thr, float(threshold))
    thr_shape = (b, 1, *([1] * (vol.ndim - 2)))
    thr = torch.where(p_high > p_low, dyn_thr, fixed_thr).view(thr_shape)
    return vol > thr


def compute_foreground_voxel_weights(
    x: torch.Tensor,
    background_weight: float = 0.1,
    threshold: float = 0.0,
    dynamic_quantiles: tuple[float, float] | list[float] = (0.02, 0.98),
    dynamic_scale: float = 0.1,
) -> torch.Tensor:
    """Voxel-level foreground weights ``[B,1,D,H,W]`` (fg=1, bg=background_weight).

    Robust per-volume threshold between configurable low/high intensity percentiles
    (channel-collapsed via abs-max); used for the foreground-aware AMAES reconstruction.
    """
    fg = foreground_voxel_mask(
        x,
        threshold=threshold,
        dynamic_quantiles=dynamic_quantiles,
        dynamic_scale=dynamic_scale,
    ).to(x.dtype)
    return background_weight + (1.0 - background_weight) * fg


def weighted_rec_loss(pred, y, mask=None, weights=None, eps: float = 1e-6, sample_normalize: bool = True):
    """Foreground-weighted MSE (weighted mean of squared error) over the hidden region.

    Equivalent to :func:`rec_loss` when ``weights`` is uniform; downweights background
    voxels by ``background_weight`` so the loss focuses on tissue.
    """
    se = (pred - y) ** 2
    w = weights if weights is not None else torch.ones_like(se)
    if w.shape != se.shape:
        w = w.expand_as(se)
    if mask is not None:
        hidden = ~mask.to(dtype=torch.bool)
        if not hidden.any():
            return pred.sum() * 0.0
        if sample_normalize and se.shape[0] > 0:
            hidden = hidden.expand_as(se)
            weighted_sum = (se * w * hidden).flatten(1).sum(dim=1)
            weight_sum = (w * hidden).flatten(1).sum(dim=1)
            valid = weight_sum > eps
            if not bool(valid.any()):
                return pred.sum() * 0.0
            return (weighted_sum[valid] / weight_sum[valid].clamp(min=eps)).mean()
        return (se[hidden] * w[hidden]).sum() / w[hidden].sum().clamp(min=eps)
    if sample_normalize and se.shape[0] > 0:
        weighted_sum = (se * w).flatten(1).sum(dim=1)
        weight_sum = w.flatten(1).sum(dim=1)
        valid = weight_sum > eps
        if not bool(valid.any()):
            return pred.sum() * 0.0
        return (weighted_sum[valid] / weight_sum[valid].clamp(min=eps)).mean()
    return (se * w).sum() / w.sum().clamp(min=eps)


def foreground_voxel_bonus_mse(
    pred: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor | None = None,
    threshold: float = 0.0,
    dynamic_quantiles: tuple[float, float] | list[float] = (0.02, 0.98),
    dynamic_scale: float = 0.1,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Foreground-only hidden-region MSE, normalised per sample.

    Returns ``(loss, active_fraction)`` where active samples have at least one hidden
    foreground voxel. If no sample is active, the loss is a differentiable zero.
    """

    se = (pred - y).square()
    selector = ~mask.to(dtype=torch.bool) if mask is not None else torch.ones_like(se, dtype=torch.bool)
    fg = foreground_voxel_mask(
        y,
        threshold=threshold,
        dynamic_quantiles=dynamic_quantiles,
        dynamic_scale=dynamic_scale,
    ).expand_as(se)
    selected = selector & fg
    if not bool(selected.any()):
        zero = pred.sum() * 0.0
        return zero, zero
    selected_f = selected.to(dtype=se.dtype)
    selected_count = selected_f.flatten(1).sum(dim=1)
    valid = selected_count > eps
    if not bool(valid.any()):
        zero = pred.sum() * 0.0
        return zero, zero
    sample_loss = (se * selected_f).flatten(1).sum(dim=1)[valid] / selected_count[valid].clamp(min=eps)
    active_fraction = valid.to(dtype=se.dtype).mean()
    return sample_loss.mean(), active_fraction


def _expand_spatial_size(values: tuple[int, ...] | list[int], ndim: int, name: str) -> tuple[int, ...]:
    size = tuple(int(v) for v in values)
    if len(size) == 1:
        size = size * ndim
    if len(size) != ndim or any(v <= 0 for v in size):
        raise ValueError(f"{name}={values!r} is incompatible with {ndim} spatial dims.")
    return size


def foreground_patch_bonus_mse(
    pred: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor | None = None,
    patch_size: tuple[int, ...] | list[int] = (4,),
    min_foreground_fraction: float = 0.1,
    threshold: float = 0.0,
    dynamic_quantiles: tuple[float, float] | list[float] = (0.02, 0.98),
    dynamic_scale: float = 0.1,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Patch-level foreground MSE over hidden regions.

    A patch contributes when it contains hidden target voxels and its foreground fraction
    is at least ``min_foreground_fraction``. Returns ``(loss, active_fraction,
    patch_fg_fraction_hidden)``.
    """

    spatial_ndim = pred.ndim - 2
    if spatial_ndim not in {2, 3}:
        raise ValueError(f"foreground_patch_bonus_mse expects BCHW or BCDHW tensors, got {tuple(pred.shape)}.")
    patch_size = _expand_spatial_size(patch_size, spatial_ndim, "patch_size")
    spatial = tuple(int(s) for s in pred.shape[2:])
    if any(s % p != 0 for s, p in zip(spatial, patch_size)):
        raise ValueError(f"Spatial shape {spatial} must be divisible by patch_size={patch_size}.")

    se = (pred - y).square().mean(dim=1, keepdim=True)
    selector = ~mask.to(dtype=torch.bool) if mask is not None else torch.ones_like(pred, dtype=torch.bool)
    hidden = selector.any(dim=1, keepdim=True).to(dtype=se.dtype)
    fg = foreground_voxel_mask(
        y,
        threshold=threshold,
        dynamic_quantiles=dynamic_quantiles,
        dynamic_scale=dynamic_scale,
    ).to(dtype=se.dtype)

    if spatial_ndim == 3:
        pool = F.avg_pool3d
    else:
        pool = F.avg_pool2d
    patch_volume = float(math.prod(patch_size))
    patch_se_sum = pool(se * hidden, kernel_size=patch_size, stride=patch_size) * patch_volume
    patch_hidden_count = pool(hidden, kernel_size=patch_size, stride=patch_size) * patch_volume
    patch_fg_fraction = pool(fg, kernel_size=patch_size, stride=patch_size)
    patch_selected = (patch_hidden_count > eps) & (patch_fg_fraction >= float(min_foreground_fraction))
    if not bool(patch_selected.any()):
        zero = pred.sum() * 0.0
        return zero, zero, zero

    patch_loss = patch_se_sum / patch_hidden_count.clamp(min=eps)
    selected_f = patch_selected.to(dtype=patch_loss.dtype)
    selected_count = selected_f.flatten(1).sum(dim=1)
    valid = selected_count > eps
    if not bool(valid.any()):
        zero = pred.sum() * 0.0
        return zero, zero, zero
    sample_loss = (patch_loss * selected_f).flatten(1).sum(dim=1)[valid] / selected_count[valid].clamp(min=eps)
    active_fraction = valid.to(dtype=patch_loss.dtype).mean()
    hidden_patch = patch_hidden_count > eps
    patch_fg_fraction_hidden = patch_selected.to(dtype=patch_loss.dtype).sum() / hidden_patch.to(
        dtype=patch_loss.dtype
    ).sum().clamp(min=1.0)
    return sample_loss.mean(), active_fraction, patch_fg_fraction_hidden


def _pool_nd(x: torch.Tensor, patch_size: tuple[int, ...]) -> torch.Tensor:
    pool = F.avg_pool3d if x.ndim == 5 else F.avg_pool2d
    return pool(x, kernel_size=patch_size, stride=patch_size)


def _patch_grid_inputs(
    pred: torch.Tensor,
    y: torch.Tensor,
    patch_size: tuple[int, ...] | list[int],
    name: str,
) -> tuple[int, tuple[int, ...], tuple[int, ...], float]:
    if pred.shape != y.shape:
        raise ValueError(f"pred and y must have identical shapes, got {tuple(pred.shape)} and {tuple(y.shape)}.")
    spatial_ndim = pred.ndim - 2
    if spatial_ndim not in {2, 3}:
        raise ValueError(f"{name} expects BCHW or BCDHW tensors, got {tuple(pred.shape)}.")
    patch_size = _expand_spatial_size(patch_size, spatial_ndim, "patch_size")
    spatial = tuple(int(s) for s in pred.shape[2:])
    if any(s % p != 0 for s, p in zip(spatial, patch_size)):
        raise ValueError(f"Spatial shape {spatial} must be divisible by patch_size={patch_size}.")
    return spatial_ndim, patch_size, spatial, float(math.prod(patch_size))


def _zero_falcon_metrics(pred: torch.Tensor) -> dict[str, torch.Tensor]:
    zero = pred.sum() * 0.0
    return {
        "active_fraction": zero,
        "weight_mean": zero,
        "weight_std": zero,
        "weight_max": zero,
        "topk_fraction_realized": zero,
        "fg_patch_fraction": zero,
        "selected_bg_error_ratio": zero,
        "fg_bg_grad_ratio": zero,
        "selected_patch_count": zero,
        "candidate_patch_count": zero,
        "hidden_patch_count": zero,
    }


def _safe_power(x: torch.Tensor, exponent: float, eps: float) -> torch.Tensor:
    if float(exponent) == 0.0:
        return torch.ones_like(x)
    return x.clamp_min(eps).pow(float(exponent))


def _sample_normalize_patches(values: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
    mask_f = mask.to(dtype=values.dtype)
    count = mask_f.flatten(1).sum(dim=1, keepdim=True)
    mean = (values * mask_f).flatten(1).sum(dim=1, keepdim=True) / count.clamp(min=eps)
    view_shape = (values.shape[0],) + (1,) * (values.ndim - 1)
    return values / mean.clamp(min=eps).view(view_shape)


def _anatomy_gradient_magnitude(y: torch.Tensor) -> torch.Tensor:
    vol = y.detach().abs().amax(dim=1, keepdim=True).float()
    grad = torch.zeros_like(vol)
    for dim in range(2, vol.ndim):
        diff = vol.diff(dim=dim).abs()
        pad = [0, 0] * (vol.ndim - 2)
        spatial_dim_from_end = vol.ndim - 1 - dim
        pad[2 * spatial_dim_from_end + 1] = 1
        grad = grad + F.pad(diff, tuple(pad))
    return grad


def _patch_loss_map(
    pred: torch.Tensor,
    y: torch.Tensor,
    hidden: torch.Tensor,
    patch_size: tuple[int, ...],
    patch_volume: float,
    loss_kind: str,
    charbonnier_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = pred - y
    if loss_kind == "mse":
        voxel_loss = residual.square().mean(dim=1, keepdim=True)
    elif loss_kind == "charbonnier":
        voxel_loss = torch.sqrt(residual.square() + float(charbonnier_eps) ** 2).mean(dim=1, keepdim=True)
    else:
        raise ValueError(f"Unknown FALCON hard loss_kind={loss_kind!r}; expected 'mse' or 'charbonnier'.")
    patch_loss_sum = _pool_nd(voxel_loss * hidden, patch_size) * patch_volume
    patch_hidden_count = _pool_nd(hidden, patch_size) * patch_volume
    return patch_loss_sum / patch_hidden_count.clamp(min=1e-12), patch_hidden_count


def _topk_mask_per_sample(scores: torch.Tensor, candidates: torch.Tensor, topk_fraction: float) -> torch.Tensor:
    scores_flat = scores.flatten(1)
    candidates_flat = candidates.flatten(1)
    selected_flat = torch.zeros_like(candidates_flat, dtype=torch.bool)
    fraction = min(1.0, max(0.0, float(topk_fraction)))
    for batch_idx in range(scores_flat.shape[0]):
        candidate_idx = candidates_flat[batch_idx].nonzero(as_tuple=True)[0]
        if candidate_idx.numel() == 0:
            continue
        k = max(1, min(candidate_idx.numel(), int(math.ceil(float(candidate_idx.numel()) * fraction))))
        local_scores = scores_flat[batch_idx, candidate_idx]
        top_local = torch.topk(local_scores, k=k, largest=True, sorted=False).indices
        selected_flat[batch_idx, candidate_idx[top_local]] = True
    return selected_flat.view_as(candidates)


def falcon_hard_patch_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor | None = None,
    patch_size: tuple[int, ...] | list[int] = (4,),
    topk_fraction: float = 0.25,
    min_foreground_fraction: float = 0.1,
    foreground_alpha: float = 1.0,
    error_gamma: float = 1.0,
    gradient_eta: float = 0.5,
    threshold: float = 0.0,
    dynamic_quantiles: tuple[float, float] | list[float] = (0.02, 0.98),
    dynamic_scale: float = 0.1,
    loss_kind: str = "mse",
    charbonnier_eps: float = 1e-3,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Foreground-adaptive hard patch reconstruction loss.

    Scores are detached and only choose/weight hard foreground patches; gradients flow
    through the selected patch losses, not through the difficulty estimate itself.
    """

    _, patch_size, _, patch_volume = _patch_grid_inputs(pred, y, patch_size, "falcon_hard_patch_loss")
    selector = ~mask.to(dtype=torch.bool) if mask is not None else torch.ones_like(pred, dtype=torch.bool)
    hidden = selector.any(dim=1, keepdim=True).to(dtype=pred.dtype)
    patch_loss, patch_hidden_count = _patch_loss_map(
        pred,
        y,
        hidden,
        patch_size,
        patch_volume,
        str(loss_kind),
        charbonnier_eps,
    )
    hidden_patch = patch_hidden_count > eps
    if not bool(hidden_patch.any()):
        zero = pred.sum() * 0.0
        return zero, _zero_falcon_metrics(pred)

    fg = foreground_voxel_mask(
        y,
        threshold=threshold,
        dynamic_quantiles=dynamic_quantiles,
        dynamic_scale=dynamic_scale,
    ).to(dtype=pred.dtype)
    patch_fg_fraction = _pool_nd(fg, patch_size)
    gradient = _anatomy_gradient_magnitude(y).to(dtype=pred.dtype)
    patch_gradient = _pool_nd(gradient, patch_size)
    candidates = hidden_patch & (patch_fg_fraction >= float(min_foreground_fraction))
    if not bool(candidates.any()):
        zero = pred.sum() * 0.0
        metrics = _zero_falcon_metrics(pred)
        metrics["hidden_patch_count"] = hidden_patch.to(dtype=pred.dtype).flatten(1).sum(dim=1).mean().detach()
        return zero, metrics

    error_score = _sample_normalize_patches(patch_loss.detach(), candidates, eps)
    gradient_score = _sample_normalize_patches(patch_gradient.detach(), candidates, eps)
    score = (
        _safe_power(patch_fg_fraction.detach(), foreground_alpha, eps)
        * _safe_power(error_score, error_gamma, eps)
        * _safe_power(gradient_score, gradient_eta, eps)
    )
    score = score * candidates.to(dtype=score.dtype)
    selected = _topk_mask_per_sample(score, candidates, topk_fraction)
    selected_f = selected.to(dtype=patch_loss.dtype)
    selected_count = selected_f.flatten(1).sum(dim=1)
    valid = selected_count > eps
    if not bool(valid.any()):
        zero = pred.sum() * 0.0
        return zero, _zero_falcon_metrics(pred)

    selected_weight = score * selected_f
    weight_sum = selected_weight.flatten(1).sum(dim=1)
    weighted_sample_loss = (patch_loss * selected_weight).flatten(1).sum(dim=1) / weight_sum.clamp(min=eps)
    uniform_sample_loss = (patch_loss * selected_f).flatten(1).sum(dim=1) / selected_count.clamp(min=eps)
    sample_loss = torch.where(weight_sum > eps, weighted_sample_loss, uniform_sample_loss)
    loss = sample_loss[valid].mean()

    candidate_count = candidates.to(dtype=patch_loss.dtype).flatten(1).sum(dim=1)
    hidden_count = hidden_patch.to(dtype=patch_loss.dtype).flatten(1).sum(dim=1)
    active_fraction = valid.to(dtype=patch_loss.dtype).mean()
    selected_scores = score[selected]
    zero = pred.sum() * 0.0
    foreground_proxy = uniform_sample_loss[valid].mean().detach()
    bg_patch = hidden_patch & (patch_fg_fraction < float(min_foreground_fraction))
    if bool(bg_patch.any()):
        bg_proxy = patch_loss.detach()[bg_patch].mean()
        fg_bg_grad_ratio = foreground_proxy / bg_proxy.clamp(min=eps)
    else:
        fg_bg_grad_ratio = zero.detach()
    metrics = {
        "active_fraction": active_fraction.detach(),
        "weight_mean": selected_scores.mean().detach() if selected_scores.numel() else zero.detach(),
        "weight_std": selected_scores.std(unbiased=False).detach() if selected_scores.numel() else zero.detach(),
        "weight_max": selected_scores.max().detach() if selected_scores.numel() else zero.detach(),
        "topk_fraction_realized": (selected_count[valid] / candidate_count[valid].clamp(min=eps)).mean().detach(),
        "fg_patch_fraction": (candidate_count[valid] / hidden_count[valid].clamp(min=eps)).mean().detach(),
        "selected_bg_error_ratio": fg_bg_grad_ratio.detach(),
        # Legacy alias kept so existing W&B panels and runs remain comparable.
        "fg_bg_grad_ratio": fg_bg_grad_ratio.detach(),
        "selected_patch_count": selected_count[valid].mean().detach(),
        "candidate_patch_count": candidate_count[valid].mean().detach(),
        "hidden_patch_count": hidden_count[valid].mean().detach(),
    }
    return loss, metrics


def falcon_foreground_focal_frequency_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor | None = None,
    threshold: float = 0.0,
    dynamic_quantiles: tuple[float, float] | list[float] = (0.02, 0.98),
    dynamic_scale: float = 0.1,
    high_weight: float = 1.0,
    power: float = 2.0,
    focal_alpha: float = 1.0,
    focal_weight_max: float = 10.0,
) -> torch.Tensor:
    """Focal frequency loss restricted to hidden foreground voxels."""

    selector = ~mask.to(dtype=torch.bool) if mask is not None else torch.ones_like(pred, dtype=torch.bool)
    fg = foreground_voxel_mask(
        y,
        threshold=threshold,
        dynamic_quantiles=dynamic_quantiles,
        dynamic_scale=dynamic_scale,
    ).expand_as(pred)
    selected = selector & fg
    if not bool(selected.any()):
        return pred.sum() * 0.0
    foreground_hidden_mask = ~selected
    return frequency_domain_loss(
        pred,
        y,
        mask=foreground_hidden_mask,
        high_freq_weight=high_weight,
        frequency_power=power,
        mode="focal_frequency",
        weighting="adaptive",
        focal_alpha=focal_alpha,
        focal_weight_max=focal_weight_max,
    )


def falcon_latent_feature_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor | None = None,
    min_foreground_fraction: float = 0.1,
    threshold: float = 0.0,
    dynamic_quantiles: tuple[float, float] | list[float] = (0.02, 0.98),
    dynamic_scale: float = 0.1,
    loss_kind: str = "mse",
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Local foreground latent consistency between masked student and clean EMA teacher."""

    if student.shape != teacher.shape:
        raise ValueError(
            f"student and teacher feature maps must match, got {tuple(student.shape)} and {tuple(teacher.shape)}."
        )
    if student.ndim not in {4, 5}:
        raise ValueError(f"falcon_latent_feature_loss expects BCHW or BCDHW feature maps, got {tuple(student.shape)}.")
    teacher = teacher.detach()
    if loss_kind == "mse":
        per_location = (student - teacher).square().mean(dim=1, keepdim=True)
    elif loss_kind == "smooth_l1":
        per_location = F.smooth_l1_loss(student, teacher, reduction="none").mean(dim=1, keepdim=True)
    else:
        raise ValueError(f"Unknown FALCON latent loss_kind={loss_kind!r}; expected 'mse' or 'smooth_l1'.")

    target_shape = tuple(int(size) for size in student.shape[2:])
    fg = foreground_voxel_mask(
        y,
        threshold=threshold,
        dynamic_quantiles=dynamic_quantiles,
        dynamic_scale=dynamic_scale,
    ).to(dtype=per_location.dtype)
    pool = F.adaptive_avg_pool3d if student.ndim == 5 else F.adaptive_avg_pool2d
    fg_fraction = pool(fg, target_shape)
    if mask is not None:
        hidden = (~mask.to(dtype=torch.bool)).any(dim=1, keepdim=True).to(dtype=per_location.dtype)
        hidden_fraction = pool(hidden, target_shape)
    else:
        hidden_fraction = torch.ones_like(fg_fraction)
    selected = (hidden_fraction > eps) & (fg_fraction >= float(min_foreground_fraction))
    if not bool(selected.any()):
        zero = student.sum() * 0.0
        return zero, {"active_fraction": zero, "fg_location_fraction": zero}

    selected_f = selected.to(dtype=per_location.dtype)
    selected_count = selected_f.flatten(1).sum(dim=1)
    valid = selected_count > eps
    if not bool(valid.any()):
        zero = student.sum() * 0.0
        return zero, {"active_fraction": zero, "fg_location_fraction": zero}
    sample_loss = (per_location * selected_f).flatten(1).sum(dim=1)[valid] / selected_count[valid].clamp(min=eps)
    hidden_count = (hidden_fraction > eps).to(dtype=per_location.dtype).flatten(1).sum(dim=1)[valid]
    metrics = {
        "active_fraction": valid.to(dtype=per_location.dtype).mean().detach(),
        "fg_location_fraction": (selected_count[valid] / hidden_count.clamp(min=eps)).mean().detach(),
    }
    return sample_loss.mean(), metrics


def foreground_reconstruction_mse_metrics(
    pred: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor | None = None,
    threshold: float = 0.0,
    dynamic_quantiles: tuple[float, float] | list[float] = (0.02, 0.98),
    dynamic_scale: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Unweighted hidden-region MSE split by foreground/background.

    These diagnostics keep reconstruction-quality comparisons comparable when the
    training objective uses foreground-weighted MSE.
    """
    se = (pred - y).square()
    if mask is not None:
        selector = ~mask.to(dtype=torch.bool)
    else:
        selector = torch.ones_like(se, dtype=torch.bool)
    if not bool(selector.any()):
        zero = pred.sum() * 0.0
        return {
            "loss_hidden_unweighted": zero,
            "loss_hidden_fg_unweighted": zero,
            "loss_hidden_bg_unweighted": zero,
            "fg_fraction_hidden_loss": zero,
        }

    fg = foreground_voxel_mask(
        y,
        threshold=threshold,
        dynamic_quantiles=dynamic_quantiles,
        dynamic_scale=dynamic_scale,
    ).expand_as(se)
    fg_selector = selector & fg
    bg_selector = selector & ~fg

    def mean_selected(selected: torch.Tensor) -> torch.Tensor:
        if not bool(selected.any()):
            return pred.sum() * 0.0
        return se[selected].mean()

    return {
        "loss_hidden_unweighted": se[selector].mean(),
        "loss_hidden_fg_unweighted": mean_selected(fg_selector),
        "loss_hidden_bg_unweighted": mean_selected(bg_selector),
        "fg_fraction_hidden_loss": fg_selector.float().sum() / selector.float().sum().clamp(min=1.0),
    }


def frequency_loss(
    pred,
    y,
    mask,
    enabled: bool,
    high_weight: float,
    power: float,
    mode: str = "legacy_log_magnitude",
    weighting: str | None = None,
    low_cutoff: float = 1.0 / 3.0,
    high_cutoff: float = 2.0 / 3.0,
    low_weight: float = 1.0,
    mid_weight: float = 1.0,
    high_band_weight: float = 2.0,
    focal_alpha: float = 1.0,
    focal_weight_max: float = 10.0,
    log_focal_eps: float = 1e-8,
):
    if not enabled:
        return pred.sum() * 0.0
    return frequency_domain_loss(
        pred,
        y,
        mask=mask,
        high_freq_weight=high_weight,
        frequency_power=power,
        mode=mode,
        weighting=weighting,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
        low_weight=low_weight,
        mid_weight=mid_weight,
        high_band_weight=high_band_weight,
        focal_alpha=focal_alpha,
        focal_weight_max=focal_weight_max,
        log_focal_eps=log_focal_eps,
    )


def wavelet_loss(
    pred,
    y,
    mask,
    enabled: bool,
    *,
    family: str,
    levels: int,
    level_weights,
    include_lowpass: bool,
    loss: str,
    eps: float,
    support_mode: str = "touched",
    return_diagnostics: bool = False,
):
    if not enabled:
        zero = pred.sum() * 0.0
        return (zero, {}) if return_diagnostics else zero
    return masked_wavelet_loss(
        pred,
        y,
        mask,
        family=family,
        levels=levels,
        level_weights=level_weights,
        support_mode=support_mode,
        include_lowpass=include_lowpass,
        loss=loss,
        eps=eps,
        return_diagnostics=return_diagnostics,
    )


def spatial_detail(pred, y, mask, enabled: bool, beta: float):
    if not enabled:
        return pred.sum() * 0.0
    return spatial_detail_loss(pred, y, mask=mask, beta=beta)


def weighted_loss(raw_loss: torch.Tensor, config_weight: float, schedule_weight: float, enabled: bool) -> torch.Tensor:
    if not enabled:
        return raw_loss.sum() * 0.0
    return float(config_weight) * float(schedule_weight) * raw_loss


def loss_metrics(total: torch.Tensor, components: dict, excluded: tuple[str, ...] = ()) -> dict:
    metrics = {"total": total.item()}
    for name, (raw, config_weight, schedule_weight, weighted, enabled) in components.items():
        if name in excluded:
            continue
        metrics[f"{name}/raw"] = raw.item()
        metrics[f"{name}/config_weight"] = float(config_weight)
        metrics[f"{name}/schedule_weight"] = float(schedule_weight)
        metrics[f"{name}/weighted"] = weighted.item()
        metrics[f"{name}/enabled"] = float(bool(enabled))
    return metrics
