"""
Loss and reconstruction metrics for SSL pretraining.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional


# Frequency-based compute functions
def compute_train(
    loss: torch.Tensor,
    pred: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    loss_fn: nn.Module,
    masked_input: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    return compute_loss_metrics(loss, pred, y, mask, loss_fn, masked_input=masked_input)


def compute_val(
    loss: torch.Tensor,
    pred: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    loss_fn: nn.Module,
    masked_input: Optional[torch.Tensor] = None,
    data_range: float = 6.0,
    log_raw_full: bool = False,
) -> Dict[str, Any]:
    return compute_loss_metrics(
        loss,
        pred,
        y,
        mask,
        loss_fn,
        masked_input=masked_input,
        log_raw_full=log_raw_full,
    ) | compute_psnr_metrics(
        pred,
        y,
        mask,
        masked_input=masked_input,
        data_range=data_range,
        log_raw_full=log_raw_full,
    )


def compute_loss_metrics(
    loss: torch.Tensor,
    pred: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    loss_fn: nn.Module,
    masked_input: Optional[torch.Tensor] = None,
    log_raw_full: bool = False,
) -> Dict[str, Any]:
    metrics = {
        "loss": loss.item(),
        "loss_hidden": loss_fn(pred, y, mask).item(),
    }
    completed = _completed_reconstruction(masked_input, pred, mask)
    metrics["loss_completed_full"] = loss_fn(completed, y, None).item()
    if log_raw_full:
        metrics["loss_raw_full"] = loss_fn(pred, y, None).item()
    return metrics


def compute_psnr_metrics(
    pred: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor],
    masked_input: Optional[torch.Tensor] = None,
    data_range: float = 6.0,
    log_raw_full: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Peak Signal-to-Noise Ratio for reconstruction quality assessment.
    Higher PSNR indicates better perceptual quality (20-40 dB typical for SSL).
    """
    if data_range <= 0:
        raise ValueError(f"PSNR data_range must be positive, got {data_range}.")

    def psnr(mse: torch.Tensor) -> torch.Tensor:
        maximum = mse.new_tensor(float(data_range))
        return 20 * torch.log10(maximum / torch.sqrt(mse)) if mse > 0 else mse.new_tensor(100.0)

    with torch.no_grad():
        completed = _completed_reconstruction(masked_input, pred, mask)
        mse_completed = ((completed - y) ** 2).mean()
        psnr_completed = psnr(mse_completed)

        if mask is not None:
            hidden = ~mask.to(dtype=torch.bool)
            masked_pred = pred[hidden]
            masked_y = y[hidden]
            if masked_pred.numel() > 0:
                mse_masked = ((masked_pred - masked_y) ** 2).mean()
                psnr_masked = psnr(mse_masked)
            else:
                psnr_masked = pred.new_tensor(0.0)

            visible = mask.to(dtype=torch.bool)
            unmasked_pred = completed[visible]
            unmasked_y = y[mask]
            if unmasked_pred.numel() > 0:
                unmasked_mse = ((unmasked_pred - unmasked_y) ** 2).mean()
            else:
                unmasked_mse = pred.new_tensor(0.0)
        else:
            psnr_masked = psnr_completed
            unmasked_mse = pred.new_tensor(0.0)

        metrics = {
            "psnr_hidden": psnr_masked,
            "psnr_completed_full": psnr_completed,
            "completed_visible_mse": unmasked_mse,
        }
        if log_raw_full:
            metrics["psnr_raw_full"] = psnr(((pred - y) ** 2).mean())
        return metrics


def _completed_reconstruction(
    masked_input: Optional[torch.Tensor], pred: torch.Tensor, mask: Optional[torch.Tensor]
) -> torch.Tensor:
    if masked_input is None or mask is None:
        return pred
    visible = mask.to(dtype=torch.bool)
    return torch.where(visible, masked_input, pred)
