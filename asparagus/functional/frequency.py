import torch
import torch.nn.functional as F

FREQUENCY_LOSS_MODES = {
    "legacy_log_magnitude",
    "residual_power_radial",
    "residual_power_band",
    "focal_frequency",
    "log_focal_frequency",
}
FREQUENCY_WEIGHTINGS = {"radial", "band", "adaptive"}


def frequency_domain_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    high_freq_weight: float = 1.0,
    frequency_power: float = 2.0,
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
) -> torch.Tensor:
    """Frequency-domain reconstruction objective.

    ``legacy_log_magnitude`` preserves the historical AMAES objective exactly:
    compare log-amplitude spectra and apply a radial high-frequency multiplier.
    The residual-power modes instead penalize the FFT power of ``pred - target``
    with orthonormal rFFT normalization and Hermitian weights, so unit spectral
    weights reduce to the spatial MSE over the active reconstruction region.
    ``focal_frequency`` keeps that residual-power contract but dynamically
    upweights hard frequencies with detached, per-sample normalized weights.
    ``log_focal_frequency`` uses log-space complex spectral differences for
    adaptive weights and a log-dampened complex residual error.
    """
    weighting = _resolve_weighting(mode, weighting)
    _validate_frequency_args(
        mode,
        weighting,
        high_freq_weight,
        frequency_power,
        low_cutoff,
        high_cutoff,
        low_weight,
        mid_weight,
        high_band_weight,
        focal_alpha,
        focal_weight_max,
        log_focal_eps,
    )

    if pred.ndim not in (4, 5):
        if mode != "legacy_log_magnitude":
            raise ValueError(f"Expected pred with shape BCHW or BCDHW, got {tuple(pred.shape)}.")
        return pred.sum() * 0.0

    if target.shape != pred.shape:
        raise ValueError(f"pred and target must have identical shapes, got {tuple(pred.shape)} and {tuple(target.shape)}.")

    if mode == "focal_frequency":
        return _focal_frequency_loss(
            pred,
            target,
            mask,
            high_freq_weight=high_freq_weight,
            frequency_power=frequency_power,
            focal_alpha=focal_alpha,
            focal_weight_max=focal_weight_max,
        )

    if mode == "log_focal_frequency":
        return _log_focal_frequency_loss(
            pred,
            target,
            mask,
            high_freq_weight=high_freq_weight,
            frequency_power=frequency_power,
            focal_alpha=focal_alpha,
            focal_weight_max=focal_weight_max,
            log_focal_eps=log_focal_eps,
        )

    if mode != "legacy_log_magnitude":
        return _residual_power_frequency_loss(
            pred,
            target,
            mask,
            high_freq_weight=high_freq_weight,
            frequency_power=frequency_power,
            weighting=weighting,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
            low_weight=low_weight,
            mid_weight=mid_weight,
            high_band_weight=high_band_weight,
        )

    if mask is not None:
        hidden = ~mask.to(dtype=torch.bool)
        residual = (pred - target) * hidden.to(dtype=pred.dtype)
        pred = residual
        target = torch.zeros_like(residual)

    spatial_dims = tuple(range(2, pred.ndim))
    pred_fft = torch.fft.rfftn(pred.float(), dim=spatial_dims)
    target_fft = torch.fft.rfftn(target.float(), dim=spatial_dims)

    pred_spectrum = torch.log1p(torch.abs(pred_fft))
    target_spectrum = torch.log1p(torch.abs(target_fft))
    weights = _radial_frequency_weights(
        pred.shape[2:],
        pred_spectrum.shape[2:],
        pred.device,
        pred_spectrum.dtype,
        high_freq_weight=high_freq_weight,
        frequency_power=frequency_power,
    )

    while weights.ndim < pred_spectrum.ndim:
        weights = weights.unsqueeze(0)

    return (weights * (pred_spectrum - target_spectrum).square()).mean()


def spectral_residual_power_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    low_cutoff: float = 1.0 / 3.0,
    high_cutoff: float = 2.0 / 3.0,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Residual Fourier power split into radial frequency bands.

    The returned residual powers use the same normalization as the residual-power
    loss. With no band weighting, ``residual_power_total`` equals spatial MSE over
    the active region.
    """
    if pred is None or target is None:
        return {}
    _validate_band_cutoffs(low_cutoff, high_cutoff)
    if pred.ndim not in (4, 5):
        return {}
    if target.shape != pred.shape:
        raise ValueError(f"pred and target must have identical shapes, got {tuple(pred.shape)} and {tuple(target.shape)}.")

    residual, hidden_count = _masked_residual(pred, target, mask)
    if hidden_count.item() <= 0:
        zero = pred.sum() * 0.0
        return {
            "residual_power_total": zero,
            "low": zero,
            "mid": zero,
            "high": zero,
            "low_fraction": zero,
            "mid_fraction": zero,
            "high_fraction": zero,
            "high_low_ratio": zero,
            "low_relative_to_target": zero,
            "mid_relative_to_target": zero,
            "high_relative_to_target": zero,
        }

    target_active = _active_target(target, mask).float()
    residual_power = _corrected_rfft_power(residual.float())
    target_power = _corrected_rfft_power(target_active)
    spatial_shape = pred.shape[2:]
    fft_shape = residual_power.shape[2:]
    low, mid, high = _frequency_band_masks(
        spatial_shape,
        fft_shape,
        pred.device,
        residual_power.dtype,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
    )
    scale = hidden_count.clamp_min(eps)

    total_energy = residual_power.sum()
    low_energy = _band_energy(residual_power, low)
    mid_energy = _band_energy(residual_power, mid)
    high_energy = _band_energy(residual_power, high)

    target_low = _band_energy(target_power, low)
    target_mid = _band_energy(target_power, mid)
    target_high = _band_energy(target_power, high)

    return {
        "residual_power_total": total_energy / scale,
        "low": low_energy / scale,
        "mid": mid_energy / scale,
        "high": high_energy / scale,
        "low_fraction": low_energy / total_energy.clamp_min(eps),
        "mid_fraction": mid_energy / total_energy.clamp_min(eps),
        "high_fraction": high_energy / total_energy.clamp_min(eps),
        "high_low_ratio": high_energy / low_energy.clamp_min(eps),
        "low_relative_to_target": low_energy / target_low.clamp_min(eps),
        "mid_relative_to_target": mid_energy / target_mid.clamp_min(eps),
        "high_relative_to_target": high_energy / target_high.clamp_min(eps),
    }


def spectral_frequency_weight_diagnostics(
    spatial_shape,
    high_freq_weight: float = 1.0,
    frequency_power: float = 2.0,
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
) -> dict[str, float]:
    """Describe spectral weighting for a concrete FFT shape.

    Residual-power losses divide by the Hermitian-corrected mean spectral
    weight. The returned effective weights are the per-band multipliers after
    that normalization; they are the quantities to inspect when deciding whether
    a run truly emphasizes high frequencies.
    """
    shape = tuple(int(size) for size in spatial_shape)
    if len(shape) not in (2, 3) or any(size <= 0 for size in shape):
        raise ValueError(f"Expected spatial_shape with 2 or 3 positive dimensions, got {spatial_shape}.")

    weighting = _resolve_weighting(mode, weighting)
    _validate_frequency_args(
        mode,
        weighting,
        high_freq_weight,
        frequency_power,
        low_cutoff,
        high_cutoff,
        low_weight,
        mid_weight,
        high_band_weight,
        focal_alpha,
        focal_weight_max,
        log_focal_eps,
    )

    device = torch.device("cpu")
    dtype = torch.float64
    fft_shape = shape[:-1] + (shape[-1] // 2 + 1,)
    correction = _rfft_hermitian_weights(shape, fft_shape, device, dtype)
    unweighted_mass = correction.sum().item()

    if weighting == "adaptive":
        prior = _radial_frequency_weights(
            shape,
            fft_shape,
            device,
            dtype,
            high_freq_weight=high_freq_weight,
            frequency_power=frequency_power,
        )
        weighted_mean = (prior * correction).sum().item() / unweighted_mass
        return {
            "weighting/adaptive_focal_alpha": float(focal_alpha),
            "weighting/adaptive_focal_weight_max": float(focal_weight_max),
            "weighting/adaptive_prior_mean_weight": float(weighted_mean),
            "weighting/adaptive_prior_min_weight": float(prior.min().item()),
            "weighting/adaptive_prior_max_weight": float(prior.max().item()),
            "weighting/adaptive_prior_high_freq_weight": float(high_freq_weight),
            "weighting/adaptive_prior_power": float(frequency_power),
            "weighting/adaptive_log_focal_eps": float(log_focal_eps),
        }

    if weighting == "radial":
        weights = _radial_frequency_weights(
            shape,
            fft_shape,
            device,
            dtype,
            high_freq_weight=high_freq_weight,
            frequency_power=frequency_power,
        )
        weighted_mean = (weights * correction).sum().item() / unweighted_mass
        return {
            "weighting/radial_mean_weight": float(weighted_mean),
            "weighting/radial_min_weight": float(weights.min().item()),
            "weighting/radial_max_weight": float(weights.max().item()),
            "weighting/radial_high_freq_weight": float(high_freq_weight),
            "weighting/radial_power": float(frequency_power),
        }

    weights = _band_frequency_weights(
        shape,
        fft_shape,
        device,
        dtype,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
        low_weight=low_weight,
        mid_weight=mid_weight,
        high_weight=high_band_weight,
    )
    low, mid, high = _frequency_band_masks(
        shape,
        fft_shape,
        device,
        dtype,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
    )
    masses = {
        "low": correction[low].sum().item(),
        "mid": correction[mid].sum().item(),
        "high": correction[high].sum().item(),
    }
    weighted_mean = (weights * correction).sum().item() / unweighted_mass
    if weighted_mean <= 0.0:
        raise ValueError("Spectral weights must contain positive mass.")

    config_weights = {
        "low": float(low_weight),
        "mid": float(mid_weight),
        "high": float(high_band_weight),
    }
    diagnostics = {
        "weighting/band_mean_weight": float(weighted_mean),
        "weighting/band_low_cutoff": float(low_cutoff),
        "weighting/band_high_cutoff": float(high_cutoff),
    }
    for band, mass in masses.items():
        diagnostics[f"weighting/{band}_mass_fraction"] = float(mass / unweighted_mass)
        diagnostics[f"weighting/{band}_config_weight"] = config_weights[band]
        diagnostics[f"weighting/{band}_effective_weight"] = float(config_weights[band] / weighted_mean)
    return diagnostics


def focal_frequency_weight_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    high_freq_weight: float = 1.0,
    frequency_power: float = 2.0,
    low_cutoff: float = 1.0 / 3.0,
    high_cutoff: float = 2.0 / 3.0,
    focal_alpha: float = 1.0,
    focal_weight_max: float = 10.0,
    eps: float = 1e-8,
    log_focal: bool = False,
    log_focal_eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Dynamic focal-frequency weight diagnostics for the current residual."""
    if pred is None or target is None:
        return {}
    if pred.ndim not in (4, 5):
        return {}
    if target.shape != pred.shape:
        raise ValueError(f"pred and target must have identical shapes, got {tuple(pred.shape)} and {tuple(target.shape)}.")

    _validate_frequency_args(
        "log_focal_frequency" if log_focal else "focal_frequency",
        "adaptive",
        high_freq_weight,
        frequency_power,
        low_cutoff,
        high_cutoff,
        1.0,
        1.0,
        1.0,
        focal_alpha,
        focal_weight_max,
        log_focal_eps,
    )
    if log_focal:
        active_pred, active_target, _ = _active_pair(pred, target, mask)
        pred_fft, target_fft = _rfftn_pair(active_pred.float(), active_target.float())
        weights, correction = _normalized_log_focal_frequency_weights(
            pred_fft,
            target_fft,
            pred.shape[2:],
            high_freq_weight=high_freq_weight,
            frequency_power=frequency_power,
            focal_alpha=focal_alpha,
            focal_weight_max=focal_weight_max,
            log_focal_eps=log_focal_eps,
        )
        error = torch.log1p(torch.abs(pred_fft - target_fft))
    else:
        residual, _ = _masked_residual(pred, target, mask)
        power = _corrected_rfft_power(residual.float())
        weights, correction = _normalized_focal_frequency_weights(
            power,
            pred.shape[2:],
            high_freq_weight=high_freq_weight,
            frequency_power=frequency_power,
            focal_alpha=focal_alpha,
            focal_weight_max=focal_weight_max,
        )
        error = None
    spatial_shape = pred.shape[2:]
    fft_shape = weights.shape[2:]
    low, mid, high = _frequency_band_masks(
        spatial_shape,
        fft_shape,
        pred.device,
        weights.dtype,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
    )
    reduce_dims = tuple(range(2, weights.ndim))
    correction_mass = correction.sum(dim=reduce_dims, keepdim=True).clamp_min(eps)
    per_channel_mean = (weights * correction).sum(dim=reduce_dims, keepdim=True) / correction_mass
    global_mass = correction.sum().clamp_min(eps) * weights.shape[0] * weights.shape[1]
    global_mean = (weights * correction).sum() / global_mass
    global_var = ((weights - per_channel_mean).square() * correction).sum() / global_mass

    def band_mean(band: torch.Tensor) -> torch.Tensor:
        while band.ndim < weights.ndim:
            band = band.unsqueeze(0)
        band_correction = correction * band.to(dtype=correction.dtype)
        denom = band_correction.sum().clamp_min(eps) * weights.shape[0] * weights.shape[1]
        return (weights * band_correction).sum() / denom

    def band_error_mean(band: torch.Tensor) -> torch.Tensor:
        if error is None:
            return weights.sum() * 0.0
        while band.ndim < error.ndim:
            band = band.unsqueeze(0)
        band_correction = correction * band.to(dtype=correction.dtype)
        denom = band_correction.sum().clamp_min(eps) * error.shape[0] * error.shape[1]
        return (error * band_correction).sum() / denom

    low_mean = band_mean(low)
    mid_mean = band_mean(mid)
    high_mean = band_mean(high)
    metrics = {
        "weight_mean": global_mean.detach(),
        "weight_std": global_var.clamp_min(0.0).sqrt().detach(),
        "weight_max": weights.max().detach(),
        "low_weight_mean": low_mean.detach(),
        "mid_weight_mean": mid_mean.detach(),
        "high_weight_mean": high_mean.detach(),
        "high_low_weight_ratio": (high_mean / low_mean.clamp_min(eps)).detach(),
    }
    if error is not None:
        error_mean = (error * correction).sum() / global_mass
        metrics.update(
            {
                "error_mean": error_mean.detach(),
                "low_error_mean": band_error_mean(low).detach(),
                "mid_error_mean": band_error_mean(mid).detach(),
                "high_error_mean": band_error_mean(high).detach(),
            }
        )
    return metrics


def spatial_detail_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    beta: float = 0.1,
) -> torch.Tensor:
    """Penalize missing local detail using finite differences in valid hidden neighborhoods."""
    if pred.ndim not in (4, 5):
        return pred.sum() * 0.0

    losses = []
    hidden = None if mask is None else ~mask.to(dtype=torch.bool)
    for axis in range(2, pred.ndim):
        pred_grad = pred.diff(dim=axis)
        target_grad = target.diff(dim=axis)
        per_voxel = F.smooth_l1_loss(pred_grad, target_grad, beta=beta, reduction="none")
        if hidden is not None:
            valid = hidden.narrow(axis, 1, hidden.shape[axis] - 1) & hidden.narrow(axis, 0, hidden.shape[axis] - 1)
            if valid.any():
                losses.append(per_voxel[valid].mean())
        else:
            losses.append(per_voxel.mean())

    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def _radial_frequency_weights(
    spatial_shape,
    fft_shape,
    device: torch.device,
    dtype: torch.dtype,
    high_freq_weight: float,
    frequency_power: float,
) -> torch.Tensor:
    frequencies = []
    for axis, size in enumerate(spatial_shape):
        if axis == len(spatial_shape) - 1:
            freq = torch.fft.rfftfreq(size, device=device, dtype=dtype)
        else:
            freq = torch.fft.fftfreq(size, device=device, dtype=dtype)
        frequencies.append(freq[: fft_shape[axis]].abs())

    grids = torch.meshgrid(*frequencies, indexing="ij")
    radius = torch.zeros_like(grids[0])
    for grid in grids:
        radius = radius + grid.square()
    radius = radius.sqrt()

    max_radius = radius.max().clamp_min(torch.finfo(dtype).eps)
    normalized_radius = radius / max_radius
    return 1.0 + float(high_freq_weight) * normalized_radius.pow(float(frequency_power))


def _residual_power_frequency_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    high_freq_weight: float,
    frequency_power: float,
    weighting: str,
    low_cutoff: float,
    high_cutoff: float,
    low_weight: float,
    mid_weight: float,
    high_band_weight: float,
) -> torch.Tensor:
    residual, hidden_count = _masked_residual(pred, target, mask)
    if hidden_count.item() <= 0:
        return pred.sum() * 0.0

    power = _corrected_rfft_power(residual.float())
    spatial_shape = pred.shape[2:]
    fft_shape = power.shape[2:]
    if weighting == "radial":
        weights = _radial_frequency_weights(
            spatial_shape,
            fft_shape,
            pred.device,
            power.dtype,
            high_freq_weight=high_freq_weight,
            frequency_power=frequency_power,
        )
    else:
        weights = _band_frequency_weights(
            spatial_shape,
            fft_shape,
            pred.device,
            power.dtype,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
            low_weight=low_weight,
            mid_weight=mid_weight,
            high_weight=high_band_weight,
        )

    correction = _rfft_hermitian_weights(spatial_shape, fft_shape, pred.device, power.dtype)
    weighted_mass = (weights * correction).sum()
    unweighted_mass = correction.sum()
    if weighted_mass.item() <= 0:
        raise ValueError("Spectral weights must contain positive mass.")

    while weights.ndim < power.ndim:
        weights = weights.unsqueeze(0)
    normalizer = hidden_count * (weighted_mass / unweighted_mass).to(dtype=hidden_count.dtype)
    return (power * weights).sum() / normalizer.clamp_min(torch.finfo(power.dtype).eps)


def _focal_frequency_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    high_freq_weight: float,
    frequency_power: float,
    focal_alpha: float,
    focal_weight_max: float,
) -> torch.Tensor:
    residual, hidden_count = _masked_residual(pred, target, mask)
    if hidden_count.item() <= 0:
        return pred.sum() * 0.0

    power = _corrected_rfft_power(residual.float())
    weights, _ = _normalized_focal_frequency_weights(
        power,
        pred.shape[2:],
        high_freq_weight=high_freq_weight,
        frequency_power=frequency_power,
        focal_alpha=focal_alpha,
        focal_weight_max=focal_weight_max,
    )
    return (power * weights).sum() / hidden_count.clamp_min(torch.finfo(power.dtype).eps)


def _log_focal_frequency_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    high_freq_weight: float,
    frequency_power: float,
    focal_alpha: float,
    focal_weight_max: float,
    log_focal_eps: float,
) -> torch.Tensor:
    active_pred, active_target, hidden_count = _active_pair(pred, target, mask)
    if hidden_count.item() <= 0:
        return pred.sum() * 0.0

    pred_fft, target_fft = _rfftn_pair(active_pred.float(), active_target.float())
    weights, correction = _normalized_log_focal_frequency_weights(
        pred_fft,
        target_fft,
        pred.shape[2:],
        high_freq_weight=high_freq_weight,
        frequency_power=frequency_power,
        focal_alpha=focal_alpha,
        focal_weight_max=focal_weight_max,
        log_focal_eps=log_focal_eps,
    )
    error = torch.log1p(torch.abs(pred_fft - target_fft))
    reduce_dims = tuple(range(2, error.ndim))
    correction_mass = correction.sum(dim=reduce_dims, keepdim=True).clamp_min(torch.finfo(error.dtype).eps)
    per_channel_loss = (error * weights * correction).sum(dim=reduce_dims, keepdim=True) / correction_mass
    return per_channel_loss.mean()


def _normalized_focal_frequency_weights(
    power: torch.Tensor,
    spatial_shape,
    high_freq_weight: float,
    frequency_power: float,
    focal_alpha: float,
    focal_weight_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    fft_shape = power.shape[2:]
    correction = _rfft_hermitian_weights(spatial_shape, fft_shape, power.device, power.dtype)
    while correction.ndim < power.ndim:
        correction = correction.unsqueeze(0)

    if float(focal_alpha) == 0.0:
        weights = torch.ones_like(power)
    else:
        weights = power.detach().clamp_min(0.0).sqrt().pow(float(focal_alpha))
        weights = weights.clamp(max=float(focal_weight_max))

    if float(high_freq_weight) > 0.0:
        prior = _radial_frequency_weights(
            spatial_shape,
            fft_shape,
            power.device,
            power.dtype,
            high_freq_weight=high_freq_weight,
            frequency_power=frequency_power,
        )
        while prior.ndim < weights.ndim:
            prior = prior.unsqueeze(0)
        weights = weights * prior

    reduce_dims = tuple(range(2, weights.ndim))
    mean_weight = (weights * correction).sum(dim=reduce_dims, keepdim=True) / correction.sum(
        dim=reduce_dims, keepdim=True
    ).clamp_min(torch.finfo(power.dtype).eps)
    weights = weights / mean_weight.clamp_min(torch.finfo(power.dtype).eps)
    return weights, correction


def _normalized_log_focal_frequency_weights(
    pred_fft: torch.Tensor,
    target_fft: torch.Tensor,
    spatial_shape,
    high_freq_weight: float,
    frequency_power: float,
    focal_alpha: float,
    focal_weight_max: float,
    log_focal_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    fft_shape = pred_fft.shape[2:]
    dtype = pred_fft.real.dtype
    correction = _rfft_hermitian_weights(spatial_shape, fft_shape, pred_fft.device, dtype)
    while correction.ndim < pred_fft.ndim:
        correction = correction.unsqueeze(0)

    if float(focal_alpha) == 0.0:
        weights = torch.ones_like(pred_fft.real)
    else:
        eps = float(log_focal_eps)
        delta_re = torch.log(pred_fft.real.detach().abs() + eps) - torch.log(target_fft.real.detach().abs() + eps)
        delta_im = torch.log(pred_fft.imag.detach().abs() + eps) - torch.log(target_fft.imag.detach().abs() + eps)
        weights = torch.sqrt(delta_re.square() + delta_im.square()).pow(float(focal_alpha))
        weights = weights.clamp(max=float(focal_weight_max))

    if float(high_freq_weight) > 0.0:
        prior = _radial_frequency_weights(
            spatial_shape,
            fft_shape,
            pred_fft.device,
            dtype,
            high_freq_weight=high_freq_weight,
            frequency_power=frequency_power,
        )
        while prior.ndim < weights.ndim:
            prior = prior.unsqueeze(0)
        weights = weights * prior

    reduce_dims = tuple(range(2, weights.ndim))
    eps = torch.finfo(dtype).eps
    correction_mass = correction.sum(dim=reduce_dims, keepdim=True).clamp_min(eps)
    mean_weight = (weights * correction).sum(dim=reduce_dims, keepdim=True) / correction_mass
    weights = torch.where(mean_weight > eps, weights / mean_weight.clamp_min(eps), torch.ones_like(weights))
    return weights, correction


def _active_pair(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mask is None:
        return pred, target, pred.new_tensor(float(pred.numel()))

    hidden = _hidden_mask_like(mask, pred)
    hidden_float = hidden.to(dtype=pred.dtype)
    return pred * hidden_float, target * hidden_float, hidden_float.sum()


def _masked_residual(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    residual = pred - target
    if mask is None:
        return residual, residual.new_tensor(float(residual.numel()))

    hidden = _hidden_mask_like(mask, residual)
    hidden_float = hidden.to(dtype=residual.dtype)
    return residual * hidden_float, hidden_float.sum()


def _rfftn_pair(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    spatial_dims = tuple(range(2, pred.ndim))
    pred = pred.contiguous()
    target = target.contiguous()
    return (
        torch.fft.rfftn(pred, dim=spatial_dims, norm="ortho"),
        torch.fft.rfftn(target, dim=spatial_dims, norm="ortho"),
    )


def _active_target(target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return target
    hidden = _hidden_mask_like(mask, target)
    return target * hidden.to(dtype=target.dtype)


def _hidden_mask_like(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    hidden = ~mask.to(device=reference.device, dtype=torch.bool)
    try:
        return hidden.expand_as(reference)
    except RuntimeError as error:
        raise ValueError(
            f"mask with shape {tuple(mask.shape)} cannot broadcast to prediction shape {tuple(reference.shape)}."
        ) from error


def _corrected_rfft_power(x: torch.Tensor) -> torch.Tensor:
    spatial_dims = tuple(range(2, x.ndim))
    spectrum = torch.fft.rfftn(x, dim=spatial_dims, norm="ortho")
    power = spectrum.abs().square()
    correction = _rfft_hermitian_weights(x.shape[2:], power.shape[2:], x.device, power.dtype)
    while correction.ndim < power.ndim:
        correction = correction.unsqueeze(0)
    return power * correction


def _rfft_hermitian_weights(spatial_shape, fft_shape, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    weights_1d = torch.ones(fft_shape[-1], device=device, dtype=dtype)
    last_size = int(spatial_shape[-1])
    if last_size > 1:
        if last_size % 2 == 0:
            if weights_1d.numel() > 2:
                weights_1d[1:-1] = 2.0
        elif weights_1d.numel() > 1:
            weights_1d[1:] = 2.0
    view_shape = [1] * (len(fft_shape) - 1) + [weights_1d.numel()]
    return weights_1d.view(view_shape).expand(tuple(fft_shape))


def _frequency_band_masks(
    spatial_shape,
    fft_shape,
    device: torch.device,
    dtype: torch.dtype,
    low_cutoff: float,
    high_cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    radius = _normalized_frequency_radius(spatial_shape, fft_shape, device, dtype)
    low = radius <= float(low_cutoff)
    high = radius >= float(high_cutoff)
    mid = ~(low | high)
    return low, mid, high


def _normalized_frequency_radius(spatial_shape, fft_shape, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    frequencies = []
    for axis, size in enumerate(spatial_shape):
        if axis == len(spatial_shape) - 1:
            freq = torch.fft.rfftfreq(size, device=device, dtype=dtype)
        else:
            freq = torch.fft.fftfreq(size, device=device, dtype=dtype)
        frequencies.append(freq[: fft_shape[axis]].abs())
    grids = torch.meshgrid(*frequencies, indexing="ij")
    radius = torch.zeros_like(grids[0])
    for grid in grids:
        radius = radius + grid.square()
    radius = radius.sqrt()
    return radius / radius.max().clamp_min(torch.finfo(dtype).eps)


def _band_frequency_weights(
    spatial_shape,
    fft_shape,
    device: torch.device,
    dtype: torch.dtype,
    low_cutoff: float,
    high_cutoff: float,
    low_weight: float,
    mid_weight: float,
    high_weight: float,
) -> torch.Tensor:
    low, mid, high = _frequency_band_masks(
        spatial_shape,
        fft_shape,
        device,
        dtype,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
    )
    weights = torch.zeros(tuple(fft_shape), device=device, dtype=dtype)
    weights = torch.where(low, weights.new_tensor(float(low_weight)), weights)
    weights = torch.where(mid, weights.new_tensor(float(mid_weight)), weights)
    weights = torch.where(high, weights.new_tensor(float(high_weight)), weights)
    return weights


def _band_energy(power: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < power.ndim:
        mask = mask.unsqueeze(0)
    return power.masked_select(mask.expand_as(power)).sum() if mask.any() else power.sum() * 0.0


def _resolve_weighting(mode: str, weighting: str | None) -> str:
    if mode not in FREQUENCY_LOSS_MODES:
        raise ValueError(f"Unknown frequency loss mode={mode!r}; expected one of {sorted(FREQUENCY_LOSS_MODES)}.")
    expected = {"residual_power_band": "band", "focal_frequency": "adaptive", "log_focal_frequency": "adaptive"}.get(
        mode, "radial"
    )
    if weighting is None:
        return expected
    if weighting not in FREQUENCY_WEIGHTINGS:
        raise ValueError(f"Unknown frequency weighting={weighting!r}; expected one of {sorted(FREQUENCY_WEIGHTINGS)}.")
    if mode != "legacy_log_magnitude" and weighting != expected:
        raise ValueError(f"mode={mode!r} requires weighting={expected!r}, got {weighting!r}.")
    return weighting


def _validate_frequency_args(
    mode: str,
    weighting: str,
    high_freq_weight: float,
    frequency_power: float,
    low_cutoff: float,
    high_cutoff: float,
    low_weight: float,
    mid_weight: float,
    high_band_weight: float,
    focal_alpha: float,
    focal_weight_max: float,
    log_focal_eps: float,
) -> None:
    if float(high_freq_weight) < 0.0:
        raise ValueError(f"high_freq_weight must be non-negative, got {high_freq_weight}.")
    if float(frequency_power) < 0.0:
        raise ValueError(f"frequency_power must be non-negative, got {frequency_power}.")
    if float(focal_alpha) < 0.0:
        raise ValueError(f"focal_alpha must be non-negative, got {focal_alpha}.")
    if float(focal_weight_max) <= 0.0:
        raise ValueError(f"focal_weight_max must be positive, got {focal_weight_max}.")
    if float(log_focal_eps) <= 0.0:
        raise ValueError(f"log_focal_eps must be positive, got {log_focal_eps}.")
    if mode in ("residual_power_band", "focal_frequency", "log_focal_frequency") or weighting in ("band", "adaptive"):
        _validate_band_cutoffs(low_cutoff, high_cutoff)
    if mode == "residual_power_band" or weighting == "band":
        band_weights = (float(low_weight), float(mid_weight), float(high_band_weight))
        if any(weight < 0.0 for weight in band_weights):
            raise ValueError(
                "Band frequency weights must be non-negative, got "
                f"low={low_weight}, mid={mid_weight}, high={high_band_weight}."
            )
        if sum(band_weights) <= 0.0:
            raise ValueError("At least one band frequency weight must be positive.")


def _validate_band_cutoffs(low_cutoff: float, high_cutoff: float) -> None:
    low, high = float(low_cutoff), float(high_cutoff)
    if not (0.0 <= low < high <= 1.0):
        raise ValueError(f"Expected 0 <= low_cutoff < high_cutoff <= 1, got {low_cutoff}, {high_cutoff}.")
