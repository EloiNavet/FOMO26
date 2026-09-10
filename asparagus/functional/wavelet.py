"""Masked stationary Haar wavelet losses and reconstruction diagnostics."""

from __future__ import annotations

import itertools
import math
import torch
import torch.nn.functional as F
from collections.abc import Sequence


def _validate_wavelet_configuration(
    reference: torch.Tensor,
    *,
    family: str,
    levels: int,
    level_weights: Sequence[float] | None,
    support_mode: str,
    loss: str,
    eps: float,
) -> tuple[float, ...]:
    if reference.ndim not in (4, 5):
        raise ValueError(f"Stationary wavelet operations expect [B,C,H,W] or [B,C,D,H,W], got {tuple(reference.shape)}.")
    if family != "haar":
        raise ValueError(f"Unsupported wavelet family={family!r}; expected 'haar'.")
    if support_mode not in ("touched", "interior"):
        raise ValueError(f"Unsupported wavelet support_mode={support_mode!r}; expected 'touched' or 'interior'.")
    if not isinstance(levels, int) or isinstance(levels, bool) or levels <= 0:
        raise ValueError(f"Wavelet levels must be a positive integer, got {levels!r}.")
    if loss != "l1":
        raise ValueError(f"Unsupported wavelet loss={loss!r}; expected 'l1'.")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError(f"Wavelet eps must be positive, got {eps}.")

    weights = tuple(float(value) for value in (level_weights if level_weights is not None else [1.0] * levels))
    if len(weights) != levels:
        raise ValueError(f"level_weights must contain one value per level ({levels}), got {len(weights)}.")
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError(f"Wavelet level weights must be finite and non-negative, got {weights!r}.")
    if sum(weights) <= 0.0:
        raise ValueError("At least one wavelet level weight must be positive.")

    max_dilation = 2 ** (levels - 1)
    left = max_dilation // 2
    right = max_dilation - left
    minimum_size = max(left, right) + 1
    if any(size < minimum_size for size in reference.shape[2:]):
        raise ValueError(
            f"Spatial dimensions must be at least {minimum_size} for {levels} Haar SWT levels with reflect padding, "
            f"got {tuple(reference.shape[2:])}."
        )
    return weights


def _hidden_mask(reference: torch.Tensor, visible_mask: torch.Tensor | None) -> torch.Tensor:
    if visible_mask is None:
        return torch.ones_like(reference, dtype=torch.bool)
    if visible_mask.ndim != reference.ndim:
        raise ValueError(f"Wavelet mask must have the same rank as pred/target, got {visible_mask.ndim} and {reference.ndim}.")
    if visible_mask.shape[0] != reference.shape[0] or visible_mask.shape[2:] != reference.shape[2:]:
        raise ValueError(
            f"Wavelet mask batch/spatial shape {tuple(visible_mask.shape)} is incompatible with {tuple(reference.shape)}."
        )
    if visible_mask.shape[1] not in (1, reference.shape[1]):
        raise ValueError(f"Wavelet mask channels must be 1 or {reference.shape[1]}, got {visible_mask.shape[1]}.")
    return (~visible_mask.to(device=reference.device, dtype=torch.bool)).expand_as(reference)


def _haar_kernels(spatial_dims: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, tuple[str, ...]]:
    scale = 1.0 / math.sqrt(2.0)
    low = torch.tensor([scale, scale], device=device, dtype=dtype)
    high = torch.tensor([-scale, scale], device=device, dtype=dtype)
    kernels = []
    names = []
    for choices in itertools.product((0, 1), repeat=spatial_dims):
        kernel = low if choices[0] == 0 else high
        for choice in choices[1:]:
            kernel = kernel.unsqueeze(-1) * (low if choice == 0 else high)
        kernels.append(kernel)
        names.append("".join("L" if choice == 0 else "H" for choice in choices))
    return torch.stack(kernels).unsqueeze(1), tuple(names)


def _same_reflect_pad(value: torch.Tensor, dilation: int) -> torch.Tensor:
    left = dilation // 2
    right = dilation - left
    padding = []
    for _ in range(value.ndim - 2):
        padding.extend((left, right))
    return F.pad(value, tuple(padding), mode="reflect")


def _stationary_haar_level(value: torch.Tensor, dilation: int) -> tuple[torch.Tensor, tuple[str, ...]]:
    spatial_dims = value.ndim - 2
    kernels, names = _haar_kernels(spatial_dims, device=value.device, dtype=value.dtype)
    channels = value.shape[1]
    weight = kernels.repeat(channels, 1, *([1] * spatial_dims))
    padded = _same_reflect_pad(value, dilation)
    if spatial_dims == 2:
        output = F.conv2d(padded, weight, dilation=dilation, groups=channels)
    else:
        output = F.conv3d(padded, weight, dilation=dilation, groups=channels)
    return output.reshape(value.shape[0], channels, len(names), *value.shape[2:]), names


def _propagate_support(support: torch.Tensor, dilation: int, support_mode: str) -> torch.Tensor:
    spatial_dims = support.ndim - 2
    channels = support.shape[1]
    kernel = torch.ones((channels, 1, *([2] * spatial_dims)), device=support.device, dtype=support.dtype)
    padded = _same_reflect_pad(support, dilation)
    if spatial_dims == 2:
        propagated = F.conv2d(padded, kernel, dilation=dilation, groups=channels)
    else:
        propagated = F.conv3d(padded, kernel, dilation=dilation, groups=channels)
    if support_mode == "touched":
        selected = propagated > 0
    else:
        selected = propagated >= float(2**spatial_dims)
    return selected.to(dtype=support.dtype)


def stationary_haar_coefficients(value: torch.Tensor, levels: int) -> list[dict[str, torch.Tensor]]:
    """Return undecimated Haar coefficients for each level.

    The first mapping entry at every level is the all-low approximation; all
    remaining entries are spatially aligned detail subbands.
    """
    current = value.float()
    coefficients = []
    for level in range(levels):
        bands, names = _stationary_haar_level(current, dilation=2**level)
        level_coefficients = {name: bands[:, :, index] for index, name in enumerate(names)}
        coefficients.append(level_coefficients)
        current = level_coefficients["L" * (value.ndim - 2)]
    return coefficients


def wavelet_active_support(
    hidden: torch.Tensor,
    levels: int,
    *,
    support_mode: str = "touched",
) -> list[torch.Tensor]:
    """Return coefficients touched by, or fully contained in, hidden voxels."""
    if support_mode not in ("touched", "interior"):
        raise ValueError(f"Unsupported wavelet support_mode={support_mode!r}; expected 'touched' or 'interior'.")
    support = hidden.float()
    supports = []
    for level in range(levels):
        support = _propagate_support(support, dilation=2**level, support_mode=support_mode)
        supports.append(support)
    return supports


def _supported_means(values: torch.Tensor, support: torch.Tensor, eps: float) -> torch.Tensor:
    """Mean each band per sample/channel, then average non-empty entries."""
    band_count = values.shape[2]
    numerator = (values * support.unsqueeze(2)).reshape(values.shape[0], values.shape[1], band_count, -1).sum(-1)
    denominator = support.reshape(support.shape[0], support.shape[1], -1).sum(-1)
    means = numerator / denominator.unsqueeze(-1).clamp_min(eps)
    valid = denominator > 0
    if not bool(valid.any()):
        return values.reshape(values.shape[0], values.shape[1], band_count, -1).sum(dim=(0, 1, 3)) * 0.0
    return (means * valid.unsqueeze(-1)).sum(dim=(0, 1)) / valid.sum().to(means.dtype)


def masked_wavelet_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    family: str = "haar",
    levels: int = 2,
    level_weights: Sequence[float] | None = None,
    support_mode: str = "touched",
    include_lowpass: bool = False,
    loss: str = "l1",
    eps: float = 1e-8,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute detail fidelity on the hidden residual using stationary Haar bands."""
    if pred.shape != target.shape:
        raise ValueError(f"Wavelet pred and target shapes must match, got {tuple(pred.shape)} and {tuple(target.shape)}.")
    weights = _validate_wavelet_configuration(
        pred,
        family=family,
        levels=levels,
        level_weights=level_weights,
        support_mode=support_mode,
        loss=loss,
        eps=eps,
    )
    hidden = _hidden_mask(pred, mask)
    if not bool(hidden.any()):
        zero = pred.sum() * 0.0
        diagnostics = {f"level{level + 1}": zero.detach() for level in range(levels)}
        diagnostics.update({f"active_support_level{level + 1}": zero.detach() for level in range(levels)})
        return (zero, diagnostics) if return_diagnostics else zero

    residual = pred.float() - target.float()
    if support_mode == "touched":
        residual = residual * hidden
    coefficients = stationary_haar_coefficients(residual, levels)
    supports = wavelet_active_support(hidden, levels, support_mode=support_mode)
    low_name = "L" * (pred.ndim - 2)
    level_losses = []
    diagnostics = {}
    for level, (level_coefficients, support) in enumerate(zip(coefficients, supports, strict=True)):
        selected = [value.abs() for name, value in level_coefficients.items() if include_lowpass or name != low_name]
        band_means = _supported_means(torch.stack(selected, dim=2), support, eps)
        level_loss = band_means.mean()
        level_losses.append(level_loss)
        diagnostics[f"level{level + 1}"] = level_loss.detach()
        diagnostics[f"active_support_level{level + 1}"] = support.mean().detach()

    weight_tensor = residual.new_tensor(weights)
    total = (torch.stack(level_losses) * weight_tensor).sum() / weight_tensor.sum()
    return (total, diagnostics) if return_diagnostics else total


@torch.no_grad()
def wavelet_residual_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    family: str = "haar",
    levels: int = 2,
    level_weights: Sequence[float] | None = None,
    support_mode: str = "touched",
    include_lowpass: bool = False,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Measure masked residual error by stationary wavelet level and orientation."""
    if pred.shape != target.shape:
        raise ValueError(f"Wavelet pred and target shapes must match, got {tuple(pred.shape)} and {tuple(target.shape)}.")
    weights = _validate_wavelet_configuration(
        pred,
        family=family,
        levels=levels,
        level_weights=level_weights,
        support_mode=support_mode,
        loss="l1",
        eps=eps,
    )
    hidden = _hidden_mask(pred, mask)
    residual = pred.float() - target.float()
    target_values = target.float()
    if support_mode == "touched":
        residual = residual * hidden
        target_values = target_values * hidden
    residual_coefficients = stationary_haar_coefficients(residual, levels)
    target_coefficients = stationary_haar_coefficients(target_values, levels)
    supports = wavelet_active_support(hidden, levels, support_mode=support_mode)
    low_name = "L" * (pred.ndim - 2)
    metrics: dict[str, torch.Tensor] = {}
    level_errors = []
    for level, (residual_bands, target_bands, support) in enumerate(
        zip(residual_coefficients, target_coefficients, supports, strict=True)
    ):
        names = [name for name in residual_bands if include_lowpass or name != low_name]
        residual_values = torch.stack([residual_bands[name].abs() for name in names], dim=2)
        target_band_values = torch.stack([target_bands[name].abs() for name in names], dim=2)
        error_means = _supported_means(residual_values, support, eps)
        target_means = _supported_means(target_band_values, support, eps)
        level_error = error_means.mean()
        level_index = level + 1
        metrics[f"level{level_index}"] = level_error
        metrics[f"level{level_index}_relative_to_target"] = level_error / target_means.mean().clamp_min(eps)
        metrics[f"active_support_level{level_index}"] = support.mean()
        for name, value in zip(names, error_means, strict=True):
            metrics[f"level{level_index}/{name}"] = value
        level_errors.append(level_error)

    weight_tensor = pred.new_tensor(weights, dtype=torch.float32)
    metrics["detail_total"] = (torch.stack(level_errors) * weight_tensor).sum() / weight_tensor.sum()
    metrics["detail_total_unweighted"] = torch.stack(level_errors).mean()
    return metrics
