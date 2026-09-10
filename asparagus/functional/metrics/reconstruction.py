"""
Reconstruction quality metrics for SSL pretraining.

`Torch_Mask` marks visible voxels with True. Hidden-region metrics therefore
evaluate `~mask`; completed-image metrics insert predictions only there.
"""

import torch
import torch.nn.functional as F
from asparagus.functional.frequency import spectral_residual_power_metrics
from asparagus.functional.wavelet import wavelet_residual_metrics
from typing import Dict, Optional


def completed_reconstruction(masked_input: torch.Tensor, pred: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Return an image with observed voxels retained and hidden voxels reconstructed."""
    if mask is None:
        return pred
    return torch.where(mask.to(dtype=torch.bool), masked_input, pred)


def compute(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    masked_input: Optional[torch.Tensor] = None,
    data_range: float = 6.0,
    log_raw_full: bool = False,
    wavelet_family: str = "haar",
    wavelet_levels: int = 2,
    wavelet_level_weights: tuple[float, ...] | list[float] | None = None,
    wavelet_support_mode: str = "touched",
    wavelet_include_lowpass: bool = False,
    wavelet_eps: float = 1e-8,
) -> Dict[str, float | torch.Tensor]:
    """Compute hidden-objective and completed-image validation metrics."""
    if data_range <= 0:
        raise ValueError(f"SSIM data_range must be positive, got {data_range}.")
    completed = completed_reconstruction(masked_input, pred, mask) if masked_input is not None else pred

    hidden = compute_ssim_3d(pred, target, mask, n_slices=3, data_range=data_range)
    metrics: Dict[str, float | torch.Tensor] = {
        "ssim_3d_hidden": hidden.get("ssim_3d_hidden", hidden["ssim_3d"]),
        # Compatibility alias under corrected hidden-region semantics.
        "ssim_3d_masked": hidden.get("ssim_3d_hidden", hidden["ssim_3d"]),
        "ssim_3d_completed_full": compute_ssim_3d(completed, target, None, n_slices=3, data_range=data_range)["ssim_3d"],
    }

    hidden_edge = compute_edge_aware_error(pred, target, mask)
    metrics |= {
        "edge_error_hidden": hidden_edge.get("edge_error_hidden", hidden_edge["edge_error_total"]),
        "edge_error_masked": hidden_edge.get("edge_error_hidden", hidden_edge["edge_error_total"]),
        "edge_error_completed_full": compute_edge_aware_error(completed, target, None)["edge_error_total"],
    }
    hidden_spectral = compute_frequency_domain_error(pred, target, mask)
    metrics |= {f"spectral_error_hidden/{key}": value for key, value in hidden_spectral.items()}
    completed_spectral = compute_frequency_domain_error(completed, target, None)
    metrics |= {f"spectral_error_completed_full/{key}": value for key, value in completed_spectral.items()}
    hidden_wavelet = compute_wavelet_domain_error(
        pred,
        target,
        mask,
        family=wavelet_family,
        levels=wavelet_levels,
        level_weights=wavelet_level_weights,
        support_mode=wavelet_support_mode,
        include_lowpass=wavelet_include_lowpass,
        eps=wavelet_eps,
    )
    metrics |= {f"wavelet_error_hidden/{key}": value for key, value in hidden_wavelet.items()}
    completed_wavelet = compute_wavelet_domain_error(
        completed,
        target,
        None,
        family=wavelet_family,
        levels=wavelet_levels,
        level_weights=wavelet_level_weights,
        support_mode=wavelet_support_mode,
        include_lowpass=wavelet_include_lowpass,
        eps=wavelet_eps,
    )
    metrics |= {f"wavelet_error_completed_full/{key}": value for key, value in completed_wavelet.items()}
    # Compatibility aliases remain the hidden residual spectrum under current semantics.
    metrics |= hidden_spectral

    if log_raw_full:
        metrics["ssim_3d_raw_full"] = hidden["ssim_3d"]
        metrics["edge_error_raw_full"] = hidden_edge["edge_error_total"]
    return metrics


def compute_ssim_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    window_size: int = 11,
    n_slices: int = 5,
    data_range: float = 6.0,
) -> Dict[str, torch.Tensor]:
    """Compute axial-slice SSIM and, when provided, hidden-region SSIM."""
    if pred.dim() == 4:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)
        if mask is not None:
            mask = mask.unsqueeze(1)

    _, _, depth, _, _ = pred.shape

    def ssim_2d(img1, img2, return_map=False):
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32, device=img1.device)
        gauss = torch.exp(-(coords**2) / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        window = (gauss.unsqueeze(0) * gauss.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        mu1 = F.conv2d(img1.float(), window, padding=window_size // 2)
        mu2 = F.conv2d(img2.float(), window, padding=window_size // 2)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.square(), mu2.square(), mu1 * mu2
        sigma1_sq = F.conv2d(img1.float().square(), window, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(img2.float().square(), window, padding=window_size // 2) - mu2_sq
        sigma12 = F.conv2d(img1.float() * img2.float(), window, padding=window_size // 2) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return ssim_map if return_map else ssim_map.mean()

    indices = torch.linspace(0, depth - 1, n_slices, device=pred.device).long()
    full_values = [ssim_2d(pred[:, :, index], target[:, :, index]) for index in indices]
    metrics = {"ssim_3d": torch.stack(full_values).mean()}

    if mask is not None:
        hidden = ~mask.to(dtype=torch.bool)
        pred_hidden = pred * hidden
        target_hidden = target * hidden
        hidden_values = []
        for index in indices:
            slice_mask = hidden[:, :, index]
            if slice_mask.any():
                ssim_map = ssim_2d(pred_hidden[:, :, index], target_hidden[:, :, index], return_map=True)
                hidden_values.append(ssim_map[slice_mask].mean())
        metrics["ssim_3d_hidden"] = torch.stack(hidden_values).mean() if hidden_values else pred.new_tensor(0.0)
    return metrics


def compute_edge_aware_error(
    pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """Measure Sobel-gradient magnitude error, including the hidden region."""
    if pred.dim() == 4:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)
        if mask is not None:
            mask = mask.unsqueeze(1)

    kernels = [
        [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]], [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
        [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]], [[-2, -4, -2], [0, 0, 0], [2, 4, 2]], [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
        [[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[1, 2, 1], [2, 4, 2], [1, 2, 1]]],
    ]
    kernels = [torch.tensor(kernel, dtype=torch.float32, device=pred.device).unsqueeze(0).unsqueeze(0) for kernel in kernels]

    def errors(left, right):
        left_grads = [F.conv3d(left.float(), kernel, padding=1) for kernel in kernels]
        right_grads = [F.conv3d(right.float(), kernel, padding=1) for kernel in kernels]
        left_magnitude = torch.sqrt(sum(gradient.square() for gradient in left_grads) + 1e-8)
        right_magnitude = torch.sqrt(sum(gradient.square() for gradient in right_grads) + 1e-8)
        return F.mse_loss(left_magnitude, right_magnitude, reduction="none")

    all_errors = errors(pred, target)
    metrics = {"edge_error_total": all_errors.mean().item()}
    if mask is not None:
        hidden = (~mask.to(dtype=torch.bool)).to(dtype=pred.dtype)
        hidden_errors = errors(pred * hidden, target * hidden)
        metrics["edge_error_hidden"] = (hidden_errors * hidden).sum().div(hidden.sum().clamp(min=1)).item()
    return metrics


def compute_frequency_domain_error(
    pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """Evaluate residual Fourier power by frequency band."""
    if pred is None or target is None:
        return {}
    if pred.dim() not in (4, 5):
        return {}
    metrics = spectral_residual_power_metrics(pred.float(), target.float(), mask)
    out = {key: value.item() for key, value in metrics.items()}
    if out:
        out |= {
            "freq_domain_mse": out["residual_power_total"],
            "freq_domain_low_mse": out["low"],
            "freq_domain_mid_mse": out["mid"],
            "freq_domain_high_mse": out["high"],
            "freq_domain_high_low_ratio": out["high_low_ratio"],
        }
    return out


def compute_wavelet_domain_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    family: str = "haar",
    levels: int = 2,
    level_weights: tuple[float, ...] | list[float] | None = None,
    support_mode: str = "touched",
    include_lowpass: bool = False,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Evaluate spatially localized stationary-wavelet residual error."""
    if pred is None or target is None or pred.dim() not in (4, 5):
        return {}
    metrics = wavelet_residual_metrics(
        pred,
        target,
        mask,
        family=family,
        levels=levels,
        level_weights=level_weights,
        support_mode=support_mode,
        include_lowpass=include_lowpass,
        eps=eps,
    )
    return {key: value.item() for key, value in metrics.items()}
