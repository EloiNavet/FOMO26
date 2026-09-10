"""
Spatial and mask-related metrics for SSL pretraining.
"""

import torch
from typing import Any, Dict, Optional


def _foreground_mask(x: torch.Tensor) -> torch.Tensor:
    vol = x.abs().amax(dim=1, keepdim=True).float()
    flat = vol.flatten(1)
    stride = max(1, flat.shape[1] // 100000)
    sub = flat[:, ::stride]
    p2 = torch.quantile(sub, 0.02, dim=1)
    p98 = torch.quantile(sub, 0.98, dim=1)
    thr = 0.1 * (p98 - p2) + p2
    thr = torch.where(p98 > p2, thr, torch.zeros_like(thr)).view(x.shape[0], 1, 1, 1, 1)
    return vol > thr


def _safe_fraction(values: torch.Tensor, selector: torch.Tensor) -> float:
    if not bool(selector.any()):
        return 0.0
    return values.expand_as(selector)[selector].float().mean().item()


def compute(mask: Optional[torch.Tensor], x: torch.Tensor) -> Dict[str, Any]:
    """
    Masking strategy statistics for MAE-style pretraining.
    High mask_ratio_std indicates non-uniform masking patterns.
    """
    if mask is not None:
        mask_ratio_realized = (1 - mask.float().mean()).item()
        mask_ratio_std = mask.float().std().item()
        visible_tokens = mask.sum().item()
        masked_tokens = (~mask).sum().item()
    else:
        mask_ratio_realized = 0.0
        mask_ratio_std = 0.0
        visible_tokens = 0
        masked_tokens = 0
    fg = _foreground_mask(x)
    fg_fraction = fg.float().mean().item()
    if mask is not None:
        mask_bool = mask.to(dtype=torch.bool)
        fg_fraction_visible = _safe_fraction(fg, mask_bool)
        fg_fraction_masked = _safe_fraction(fg, ~mask_bool)
    else:
        fg_fraction_visible = 0.0
        fg_fraction_masked = 0.0

    return {
        "mask_ratio_realized": mask_ratio_realized,
        "mask_ratio_std": mask_ratio_std,
        "visible_tokens": visible_tokens,
        "masked_tokens": masked_tokens,
        "fg_fraction": fg_fraction,
        "fg_fraction_visible": fg_fraction_visible,
        "fg_fraction_masked": fg_fraction_masked,
        "fg_visible_enrichment": fg_fraction_visible - fg_fraction,
        "fg_masked_enrichment": fg_fraction_masked - fg_fraction,
    }
