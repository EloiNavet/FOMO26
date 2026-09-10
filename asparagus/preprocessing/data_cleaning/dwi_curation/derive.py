"""ADC and synthetic DWI b1000 derivation, plus provenance sidecars.

All maths operate on scaled voxel arrays (``scl_slope``/``scl_inter`` already
applied by :func:`nifti.load_scaled_data`). ADC is computed in physical units
(mm^2/s) internally so b1000 synthesis is consistent, then saved scaled by
:data:`config.ADC_SAVE_SCALE` for viewer-friendly values.
"""

from __future__ import annotations

import json
import numpy as np
from . import config
from pathlib import Path
from typing import Any


def _valid_mask(s0: np.ndarray, sb: np.ndarray) -> np.ndarray:
    """Voxels where the mono-exponential model is well-posed: 0 < Sb <= S0, finite."""
    return np.isfinite(s0) & np.isfinite(sb) & (s0 > 0) & (sb > 0) & (sb <= s0)


def compute_adc_from_b0_shell(
    s0: np.ndarray, sb: np.ndarray, b: float, scale: float = config.ADC_SAVE_SCALE
) -> tuple[np.ndarray, float]:
    """ADC = -ln(clip(Sb/S0)) / b.

    Returns ``(adc_scaled_float32, valid_voxel_fraction)``. ``adc_scaled`` is 0 where
    the model is invalid. ``valid_voxel_fraction`` is over the union foreground
    (S0>0 or Sb>0) so it reflects usable brain coverage.
    """
    s0 = s0.astype(np.float64, copy=False)
    sb = sb.astype(np.float64, copy=False)
    valid = _valid_mask(s0, sb)
    ratio = np.clip(
        np.divide(sb, s0, out=np.ones_like(s0), where=valid),
        config.RATIO_CLIP_MIN,
        config.RATIO_CLIP_MAX,
    )
    adc = np.zeros_like(s0)
    adc[valid] = -np.log(ratio[valid]) / float(b)  # physical units mm^2/s
    foreground = (s0 > 0) | (sb > 0)
    frac = float(valid.sum()) / float(max(int(foreground.sum()), 1))
    return (adc * scale).astype(np.float32), frac


def synthesize_b1000_from_shell(s0: np.ndarray, sb: np.ndarray, b: float) -> tuple[np.ndarray, float]:
    """Synthesise S(b=1000) from a b0 and a single near-b1000 shell.

    ADC = -ln(Sb/S0)/b ; S1000 = S0 * exp(-1000 * ADC). Returns
    ``(s1000_float32, valid_voxel_fraction)``.
    """
    adc_scaled, frac = compute_adc_from_b0_shell(s0, sb, b, scale=1.0)  # unscaled ADC
    s1000 = s0.astype(np.float64, copy=False) * np.exp(-1000.0 * adc_scaled)
    s1000 = np.where(np.isfinite(s1000), s1000, 0.0)
    return s1000.astype(np.float32), frac


def synthesize_b1000_loginterp(s_lo: np.ndarray, b_lo: float, s_hi: np.ndarray, b_hi: float) -> tuple[np.ndarray, float]:
    """Synthesise S(b=1000) by log-signal interpolation of two bracketing shells.

    log(S1000) = log(S_lo) + (1000 - b_lo)/(b_hi - b_lo) * (log(S_hi) - log(S_lo)).
    Requires ``b_lo < 1000 < b_hi``. Returns ``(s1000_float32, valid_voxel_fraction)``.
    """
    if not (b_lo < 1000.0 < b_hi):
        raise ValueError(f"shells must bracket 1000 (got b_lo={b_lo}, b_hi={b_hi})")
    s_lo = s_lo.astype(np.float64, copy=False)
    s_hi = s_hi.astype(np.float64, copy=False)
    valid = np.isfinite(s_lo) & np.isfinite(s_hi) & (s_lo > 0) & (s_hi > 0)
    w = (1000.0 - b_lo) / (b_hi - b_lo)
    log_s1000 = np.zeros_like(s_lo)
    log_s1000[valid] = np.log(s_lo[valid]) + w * (np.log(s_hi[valid]) - np.log(s_lo[valid]))
    s1000 = np.zeros_like(s_lo)
    s1000[valid] = np.exp(log_s1000[valid])
    foreground = (s_lo > 0) | (s_hi > 0)
    frac = float(valid.sum()) / float(max(int(foreground.sum()), 1))
    return s1000.astype(np.float32), frac


def write_derivation_provenance(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON sidecar describing a synthesized output (atomic write)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
