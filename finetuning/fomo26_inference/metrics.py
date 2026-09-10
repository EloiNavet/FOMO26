"""Segmentation metrics matching the FOMO26 leaderboard: DSC and NSD (normalized surface dice)."""

from __future__ import annotations

import torch
from monai.metrics import compute_dice, compute_surface_dice


def _onehot(mask: torch.Tensor, n_classes: int) -> torch.Tensor:
    """mask [B, *spatial] int -> one-hot [B, C, *spatial] float."""
    return torch.movedim(torch.nn.functional.one_hot(mask.long(), n_classes), -1, 1).float()


def dsc_nsd(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    n_classes: int,
    spacing: tuple[float, ...] | None = None,
    nsd_tolerance_mm: float = 1.0,
) -> dict[int, dict[str, float]]:
    """Per-foreground-class DSC and NSD.

    pred_mask / gt_mask: [B, *spatial] integer label maps (B is typically 1).
    Returns {class_index: {"dsc": float, "nsd": float}} for classes 1..n_classes-1.
    """
    pred_oh = _onehot(pred_mask, n_classes)
    gt_oh = _onehot(gt_mask, n_classes)

    dsc = compute_dice(pred_oh, gt_oh, include_background=True)  # [B, C]
    thresholds = [nsd_tolerance_mm] * n_classes
    spacing_arg = tuple(spacing) if spacing is not None else None
    nsd = compute_surface_dice(
        pred_oh,
        gt_oh,
        class_thresholds=thresholds,
        include_background=True,
        spacing=spacing_arg,
    )  # [B, C]

    out: dict[int, dict[str, float]] = {}
    for c in range(1, n_classes):
        out[c] = {
            "dsc": float(torch.nanmean(dsc[:, c]).item()),
            "nsd": float(torch.nanmean(nsd[:, c]).item()),
        }
    return out
