"""Deterministic cross-patch views for cls/reg inference.

FOMO25's ``self_supervised_crosspatch.py`` uses cross-reconstruction between
masked patches as a pretraining objective. For FOMO26 submission inference we
adapt the useful part of that idea conservatively: evaluate classification /
regression heads on several spatial crops of the same normalized volume, then
average the predictions together with fold ensembling and flip-TTA.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from asparagus.modules.transforms.pad import Torch_Pad
from collections.abc import Iterable, Iterator, Sequence
from gardening_tools.modules.transforms.normalize import Torch_Normalize
from torchvision import transforms

CROSS_PATCH_CHOICES = ("none", "cross5", "cross9")


def clsreg_inference_transforms(
    target_size: Sequence[int],
    cross_patch: str = "none",
    normalize: bool = True,
    runtime_target_spacing=None,
):
    """Return cls/reg test transforms.

    ``none`` keeps the historical center-crop path. Cross-patch modes keep the
    full padded volume so the ensemble can crop multiple deterministic views.

    ``runtime_target_spacing`` canonicalizes physical voxel spacing before any spatial op, so a
    fixed crop covers a fixed physical field of view. It is ``None`` by default and splices
    nothing, which leaves both branches composing exactly as they did before.
    """
    from asparagus.modules.transforms.presets import CPU_clsreg_val_test_transforms_crop
    from asparagus.modules.transforms.presets.train import _runtime_spacing_stage

    if cross_patch == "none":
        return CPU_clsreg_val_test_transforms_crop(
            target_size=target_size,
            normalize=normalize,
            runtime_target_spacing=runtime_target_spacing,
        )
    if cross_patch not in CROSS_PATCH_CHOICES:
        raise ValueError(f"Unsupported cross_patch={cross_patch!r}; expected one of {CROSS_PATCH_CHOICES}.")
    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            *_runtime_spacing_stage(runtime_target_spacing),
            Torch_Pad(patch_size=target_size),
        ]
    )


def _target_tuple(target_size: Sequence[int]) -> tuple[int, int, int]:
    target = tuple(int(v) for v in target_size)
    if len(target) != 3:
        raise ValueError(f"Cross-patch inference expects a 3D target size, got {target!r}.")
    return target


def _dedupe(offsets: Iterable[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    seen = set()
    out = []
    for offset in offsets:
        if offset not in seen:
            out.append(offset)
            seen.add(offset)
    return out


def cross_patch_offsets(
    spatial_shape: Sequence[int],
    target_size: Sequence[int],
    mode: str,
) -> list[tuple[int, int, int]]:
    """Return deterministic crop offsets for ``mode``.

    ``cross5`` = center + low/high shifts on the two axes with most available
    context. ``cross9`` adds low/high shifts on the third axis and two diagonal
    crops across the two largest axes. If the input already equals the target,
    duplicate offsets collapse back to the center crop.
    """
    if mode not in CROSS_PATCH_CHOICES:
        raise ValueError(f"Unsupported cross_patch={mode!r}; expected one of {CROSS_PATCH_CHOICES}.")

    spatial = tuple(int(v) for v in spatial_shape)
    target = _target_tuple(target_size)
    if len(spatial) != 3:
        raise ValueError(f"Cross-patch inference expects 3 spatial dims, got {spatial!r}.")

    spare = tuple(max(0, s - t) for s, t in zip(spatial, target, strict=True))
    center = tuple(v // 2 for v in spare)
    if mode == "none":
        return [center]

    axes_by_context = sorted(range(3), key=lambda axis: spare[axis], reverse=True)
    offsets: list[tuple[int, int, int]] = [center]

    def shifted(axis: int, value: int) -> tuple[int, int, int]:
        offset = list(center)
        offset[axis] = int(value)
        return tuple(offset)

    for axis in axes_by_context[:2]:
        offsets.append(shifted(axis, 0))
        offsets.append(shifted(axis, spare[axis]))

    if mode == "cross9":
        third_axis = axes_by_context[2]
        offsets.append(shifted(third_axis, 0))
        offsets.append(shifted(third_axis, spare[third_axis]))

        axis_a, axis_b = axes_by_context[:2]
        for value_a, value_b in ((0, 0), (spare[axis_a], spare[axis_b])):
            offset = list(center)
            offset[axis_a] = int(value_a)
            offset[axis_b] = int(value_b)
            offsets.append(tuple(offset))

    return _dedupe(offsets)


def _pad_to_target(x: torch.Tensor, target_size: tuple[int, int, int]) -> torch.Tensor:
    spatial = tuple(int(v) for v in x.shape[-3:])
    pad_d = max(0, target_size[0] - spatial[0])
    pad_h = max(0, target_size[1] - spatial[1])
    pad_w = max(0, target_size[2] - spatial[2])
    if pad_d == pad_h == pad_w == 0:
        return x

    # F.pad expects pads from last dimension backwards: W, H, D.
    pads = (
        pad_w // 2,
        pad_w - pad_w // 2,
        pad_h // 2,
        pad_h - pad_h // 2,
        pad_d // 2,
        pad_d - pad_d // 2,
    )
    return F.pad(x, pads)


def iter_cross_patch_views(
    x: torch.Tensor,
    target_size: Sequence[int],
    mode: str = "none",
) -> Iterator[torch.Tensor]:
    """Yield ``[B, C, D, H, W]`` crop views from a ``[B, C, D, H, W]`` tensor."""
    if x.ndim != 5:
        raise ValueError(f"Expected tensor [B,C,D,H,W], got shape {tuple(x.shape)}.")
    target = _target_tuple(target_size)
    x = _pad_to_target(x, target)
    for d, h, w in cross_patch_offsets(x.shape[-3:], target, mode):
        yield x[:, :, d : d + target[0], h : h + target[1], w : w + target[2]]


def make_cross_patch_batch(
    x: torch.Tensor,
    target_size: Sequence[int],
    mode: str = "none",
) -> torch.Tensor:
    """Stack cross-patch views as ``[V, B, C, D, H, W]`` for tests/debugging."""
    return torch.stack(list(iter_cross_patch_views(x, target_size, mode)), dim=0)
