"""Foreground-aware masking transforms for AMAES-style reconstruction pretraining."""

import math
import torch
import torch.nn.functional as F


def _expand_token_size(token_size, ndim: int) -> tuple[int, ...]:
    values = [int(v) for v in list(token_size)]
    if len(values) == 1:
        values = values * ndim
    if len(values) != ndim:
        raise ValueError(f"token_size={token_size!r} is incompatible with {ndim} spatial dims.")
    if any(v <= 0 for v in values):
        raise ValueError(f"token_size values must be positive, got {token_size!r}.")
    return tuple(values)


def foreground_fraction_grid(
    image: torch.Tensor,
    token_size: tuple[int, ...] | list[int] = (4,),
    threshold: float = 0.0,
    dynamic_quantiles: tuple[float, float] | list[float] = (0.02, 0.98),
    dynamic_scale: float = 0.1,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Return foreground fraction per masking token for an unbatched ``[C,*spatial]`` image."""
    if image.ndim not in {3, 4}:
        raise ValueError(f"Expected unbatched image [C,H,W] or [C,D,H,W], got shape {tuple(image.shape)}.")
    ndim = image.ndim - 1
    token_size = _expand_token_size(token_size, ndim)
    if len(dynamic_quantiles) != 2:
        raise ValueError("dynamic_quantiles must contain exactly two values.")
    q_low, q_high = float(dynamic_quantiles[0]), float(dynamic_quantiles[1])
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError(f"Invalid dynamic_quantiles={dynamic_quantiles!r}; expected 0 <= low < high <= 1.")

    vol = image.abs().amax(dim=0, keepdim=False).float()
    flat = vol.flatten()
    stride = max(1, flat.numel() // 100000)
    sub = flat[::stride]
    p_low = torch.quantile(sub, q_low)
    p_high = torch.quantile(sub, q_high)
    dyn_thr = p_low + float(dynamic_scale) * (p_high - p_low)
    thr = dyn_thr if bool(p_high > p_low) else vol.new_tensor(float(threshold))
    voxel_fg = (vol > thr).float()

    spatial = tuple(int(s) for s in vol.shape)
    grid_dims = tuple(int(math.ceil(s / t)) for s, t in zip(spatial, token_size))
    padded_shape = tuple(g * t for g, t in zip(grid_dims, token_size))
    pad = []
    for size, padded in reversed(list(zip(spatial, padded_shape))):
        pad.extend([0, padded - size])
    padded = F.pad(voxel_fg, pad, mode="constant", value=0.0)

    if ndim == 3:
        pooled = F.avg_pool3d(padded[None, None], kernel_size=token_size, stride=token_size)[0, 0]
    else:
        pooled = F.avg_pool2d(padded[None, None], kernel_size=token_size, stride=token_size)[0, 0]
    return pooled.reshape(-1), grid_dims


def foreground_aware_mask(
    image: torch.Tensor,
    ratio: float = 0.6,
    token_size: tuple[int, ...] | list[int] = (4,),
    policy: str = "target_foreground",
    pixel_value: float = 0.0,
    foreground_bias: float = 0.5,
    threshold: float = 0.0,
    dynamic_quantiles: tuple[float, float] | list[float] = (0.02, 0.98),
    dynamic_scale: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask an unbatched ``[C,*spatial]`` image and return ``(image, visible_mask)``.

    ``visible_mask`` follows the repository convention: ``True`` means visible/context,
    ``False`` means hidden/reconstruction target.
    """
    if policy not in {"random", "target_foreground", "preserve_context"}:
        raise ValueError(f"Unknown foreground-aware mask policy={policy!r}.")

    fg_frac, grid_dims = foreground_fraction_grid(
        image,
        token_size=token_size,
        threshold=threshold,
        dynamic_quantiles=dynamic_quantiles,
        dynamic_scale=dynamic_scale,
    )
    grid_size = fg_frac.numel()
    hidden_count = int(grid_size * float(ratio))
    hidden_count = min(max(hidden_count, 0), grid_size)
    scores = torch.rand(grid_size, device=image.device, dtype=fg_frac.dtype)
    if policy == "target_foreground":
        scores = scores + float(foreground_bias) * fg_frac
    elif policy == "preserve_context":
        scores = scores - float(foreground_bias) * fg_frac

    visible_flat = torch.ones(grid_size, dtype=torch.bool, device=image.device)
    if hidden_count > 0:
        hidden_idx = torch.topk(scores, hidden_count).indices
        visible_flat[hidden_idx] = False

    visible_grid = visible_flat.view(*grid_dims)
    token_size = _expand_token_size(token_size, image.ndim - 1)
    for dim, size in enumerate(token_size):
        visible_grid = visible_grid.repeat_interleave(size, dim=dim)
    slices = tuple(slice(0, s) for s in image.shape[1:])
    visible_mask = visible_grid[slices].unsqueeze(0).expand_as(image)
    image = image.masked_fill(~visible_mask, float(pixel_value))
    return image, visible_mask


class Torch_ForegroundAwareMask:
    """Content-aware AMAES mask transform, drop-in compatible with ``Torch_Mask``."""

    def __init__(
        self,
        data_key: str = "image",
        mask_key: str = "mask",
        pixel_value: float = 0.0,
        ratio: float = 0.6,
        token_size: tuple[int, ...] | list[int] = (4,),
        policy: str = "target_foreground",
        foreground_bias: float = 0.5,
        threshold: float = 0.0,
        dynamic_quantiles: tuple[float, float] | list[float] = (0.02, 0.98),
        dynamic_scale: float = 0.1,
        batched: bool = True,
    ):
        self.data_key = data_key
        self.mask_key = mask_key
        self.pixel_value = float(pixel_value)
        self.ratio = float(ratio)
        self.token_size = tuple(token_size)
        self.policy = str(policy)
        self.foreground_bias = float(foreground_bias)
        self.threshold = float(threshold)
        self.dynamic_quantiles = tuple(dynamic_quantiles)
        self.dynamic_scale = float(dynamic_scale)
        self.batched = bool(batched)

    def _mask_one(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return foreground_aware_mask(
            image,
            ratio=self.ratio,
            token_size=self.token_size,
            policy=self.policy,
            pixel_value=self.pixel_value,
            foreground_bias=self.foreground_bias,
            threshold=self.threshold,
            dynamic_quantiles=self.dynamic_quantiles,
            dynamic_scale=self.dynamic_scale,
        )

    def __call__(self, data_dict):
        if not self.batched:
            image, mask = self._mask_one(data_dict[self.data_key])
            data_dict[self.data_key] = image
            data_dict[self.mask_key] = mask
            return data_dict

        image = data_dict[self.data_key]
        data_dict[self.mask_key] = torch.empty_like(image, dtype=torch.bool)
        for b in range(image.shape[0]):
            masked, mask = self._mask_one(image[b])
            image[b] = masked
            data_dict[self.mask_key][b] = mask
        return data_dict
