from __future__ import annotations

import torch
from collections.abc import MutableMapping


class Torch_FrepaLiteFrequencyCorruption:
    """Frepa-lite input corruption for AMAES pretraining.

    The transform perturbs low frequencies and randomly masks high-frequency
    coefficients in the input image only. Reconstruction labels are intentionally
    left untouched so the existing AMAES objective still targets the clean image.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        p: float = 1.0,
        ndim: int | None = None,
        low_cutoff: float = 1.0 / 3.0,
        high_cutoff: float = 2.0 / 3.0,
        low_scale_min: float = 0.7,
        low_scale_max: float = 1.3,
        low_noise_std: float = 0.03,
        high_mask_ratio: float = 0.25,
        eps: float = 1e-6,
    ):
        self.enabled = bool(enabled)
        self.p = float(p)
        self.ndim = None if ndim is None else int(ndim)
        self.low_cutoff = float(low_cutoff)
        self.high_cutoff = float(high_cutoff)
        self.low_scale_min = float(low_scale_min)
        self.low_scale_max = float(low_scale_max)
        self.low_noise_std = float(low_noise_std)
        self.high_mask_ratio = float(high_mask_ratio)
        self.eps = float(eps)
        self._validate()

    def _validate(self) -> None:
        if self.ndim is not None and self.ndim not in (2, 3):
            raise ValueError(f"ndim must be 2, 3 or None, got {self.ndim}.")
        if not (0.0 <= self.p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {self.p}.")
        if not (0.0 <= self.low_cutoff < self.high_cutoff <= 1.0):
            raise ValueError(f"Expected 0 <= low_cutoff < high_cutoff <= 1, got {self.low_cutoff}, {self.high_cutoff}.")
        if self.low_scale_min <= 0.0 or self.low_scale_max <= 0.0:
            raise ValueError(
                f"low_scale_min and low_scale_max must be positive, got {self.low_scale_min}, {self.low_scale_max}."
            )
        if self.low_scale_min > self.low_scale_max:
            raise ValueError(f"low_scale_min must be <= low_scale_max, got {self.low_scale_min}, {self.low_scale_max}.")
        if self.low_noise_std < 0.0:
            raise ValueError(f"low_noise_std must be non-negative, got {self.low_noise_std}.")
        if not (0.0 <= self.high_mask_ratio <= 1.0):
            raise ValueError(f"high_mask_ratio must be in [0, 1], got {self.high_mask_ratio}.")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}.")

    def __call__(self, data: MutableMapping):
        if not self.enabled:
            return data
        image = data.get("image")
        if image is None:
            return data
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected data['image'] to be a torch.Tensor, got {type(image)!r}.")

        batched, spatial_rank = self._infer_layout(image)
        x = image.unsqueeze(0) if not batched else image
        x_float = x.float()
        spatial_dims = tuple(range(x_float.ndim - spatial_rank, x_float.ndim))
        spatial_shape = tuple(int(x_float.shape[dim]) for dim in spatial_dims)

        if self.p <= 0.0:
            self._write_diagnostics(
                data,
                x_float,
                x_float,
                applied=x_float.new_zeros((x_float.shape[0], 1)),
                low_scale=x_float.new_ones((x_float.shape[0], x_float.shape[1])),
                high_keep_fraction=x_float.new_tensor(1.0),
                spatial_rank=spatial_rank,
            )
            return data

        spectrum = torch.fft.rfftn(x_float, dim=spatial_dims, norm="ortho")
        fft_shape = spectrum.shape[-spatial_rank:]
        low_band, _, high_band = self._frequency_bands(spatial_shape, fft_shape, x_float.device, x_float.dtype)
        low_band_broadcast = self._broadcast_band(low_band, spectrum.ndim)
        high_band_broadcast = self._broadcast_band(high_band, spectrum.ndim)

        corrupted_spectrum = spectrum.clone()
        low_scale = self._sample_low_scale(x_float)
        if self.low_scale_min != 1.0 or self.low_scale_max != 1.0:
            low_delta = (low_scale - 1.0).to(dtype=spectrum.dtype) * spectrum * low_band_broadcast
            corrupted_spectrum = corrupted_spectrum + low_delta

        if self.low_noise_std > 0.0:
            noise = torch.randn_like(x_float)
            reduce_dims = tuple(range(2, x_float.ndim))
            image_std = x_float.std(dim=reduce_dims, keepdim=True, unbiased=False).clamp_min(self.eps)
            noise = noise * image_std * self.low_noise_std
            noise_spectrum = torch.fft.rfftn(noise, dim=spatial_dims, norm="ortho") * low_band_broadcast
            corrupted_spectrum = corrupted_spectrum + noise_spectrum

        high_keep_fraction = x_float.new_tensor(1.0)
        if self.high_mask_ratio > 0.0 and high_band.any():
            keep_high = torch.rand_like(corrupted_spectrum.real) >= self.high_mask_ratio
            high_keep = torch.where(high_band_broadcast, keep_high, torch.ones_like(keep_high, dtype=torch.bool))
            corrupted_spectrum = corrupted_spectrum * high_keep.to(dtype=corrupted_spectrum.dtype)
            high_keep_fraction = keep_high.masked_select(high_band_broadcast.expand_as(keep_high)).float().mean()

        corrupted = torch.fft.irfftn(corrupted_spectrum, s=spatial_shape, dim=spatial_dims, norm="ortho").real
        applied = (torch.rand((x_float.shape[0], 1), device=x_float.device, dtype=x_float.dtype) < self.p).to(x_float.dtype)
        apply_shape = (x_float.shape[0],) + (1,) * (x_float.ndim - 1)
        applied_view = applied.view(apply_shape)
        out = torch.where(applied_view.bool(), corrupted, x_float)
        out = torch.nan_to_num(out, nan=0.0, posinf=4.0, neginf=-2.0).to(dtype=image.dtype)

        data["image"] = out.squeeze(0) if not batched else out
        effective_high_keep_fraction = applied.mean() * high_keep_fraction + (1.0 - applied.mean())
        self._write_diagnostics(
            data,
            x_float,
            out.float(),
            applied=applied,
            low_scale=low_scale.flatten(2).mean(dim=2),
            high_keep_fraction=effective_high_keep_fraction,
            spatial_rank=spatial_rank,
        )
        return data

    def _infer_layout(self, image: torch.Tensor) -> tuple[bool, int]:
        if self.ndim is not None:
            if image.ndim == self.ndim + 2:
                return True, self.ndim
            if image.ndim == self.ndim + 1:
                return False, self.ndim
            raise ValueError(f"Expected image with batched or unbatched {self.ndim}D shape, got {tuple(image.shape)}.")
        if image.ndim == 5:
            return True, 3
        if image.ndim == 4:
            return True, 2
        if image.ndim == 3:
            return False, 2
        raise ValueError(f"Expected image with shape BCHW, BCDHW, CHW or C(D)HW, got {tuple(image.shape)}.")

    def _sample_low_scale(self, x: torch.Tensor) -> torch.Tensor:
        shape = (x.shape[0], x.shape[1]) + (1,) * (x.ndim - 2)
        if self.low_scale_min == self.low_scale_max:
            return x.new_full(shape, self.low_scale_min)
        return x.new_empty(shape).uniform_(self.low_scale_min, self.low_scale_max)

    def _frequency_bands(
        self,
        spatial_shape: tuple[int, ...],
        fft_shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        radius = self._normalized_frequency_radius(spatial_shape, fft_shape, device, dtype)
        low = radius <= self.low_cutoff
        high = radius >= self.high_cutoff
        mid = ~(low | high)
        return low, mid, high

    @staticmethod
    def _normalized_frequency_radius(
        spatial_shape: tuple[int, ...],
        fft_shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
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
        return radius / radius.max().clamp_min(torch.finfo(dtype).eps)

    @staticmethod
    def _broadcast_band(band: torch.Tensor, target_ndim: int) -> torch.Tensor:
        while band.ndim < target_ndim:
            band = band.unsqueeze(0)
        return band

    def _write_diagnostics(
        self,
        data: MutableMapping,
        original: torch.Tensor,
        corrupted: torch.Tensor,
        *,
        applied: torch.Tensor,
        low_scale: torch.Tensor,
        high_keep_fraction: torch.Tensor,
        spatial_rank: int,
    ) -> None:
        delta = corrupted - original
        low_delta_mse, high_delta_mse = self._delta_band_mse(delta, spatial_rank)
        data["frepa_lite/applied_fraction"] = applied.mean().detach()
        data["frepa_lite/low_scale_mean"] = low_scale.mean().detach()
        data["frepa_lite/high_keep_fraction"] = high_keep_fraction.detach()
        data["frepa_lite/input_delta_mse"] = delta.square().mean().detach()
        data["frepa_lite/low_delta_mse"] = low_delta_mse.detach()
        data["frepa_lite/high_delta_mse"] = high_delta_mse.detach()

    def _delta_band_mse(self, delta: torch.Tensor, spatial_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_dims = tuple(range(delta.ndim - spatial_rank, delta.ndim))
        spatial_shape = tuple(int(delta.shape[dim]) for dim in spatial_dims)
        spectrum = torch.fft.rfftn(delta.float(), dim=spatial_dims, norm="ortho")
        power = spectrum.abs().square()
        correction = self._rfft_hermitian_weights(spatial_shape, power.shape[-spatial_rank:], power.device, power.dtype)
        while correction.ndim < power.ndim:
            correction = correction.unsqueeze(0)
        power = power * correction
        low, _, high = self._frequency_bands(spatial_shape, power.shape[-spatial_rank:], power.device, power.dtype)
        low_energy = self._band_energy(power, low)
        high_energy = self._band_energy(power, high)
        scale = delta.new_tensor(float(delta.numel())).clamp_min(self.eps)
        return low_energy / scale, high_energy / scale

    @staticmethod
    def _rfft_hermitian_weights(
        spatial_shape: tuple[int, ...],
        fft_shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
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

    @staticmethod
    def _band_energy(power: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        while mask.ndim < power.ndim:
            mask = mask.unsqueeze(0)
        return power.masked_select(mask.expand_as(power)).sum() if mask.any() else power.sum() * 0.0
