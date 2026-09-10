"""
Visualization generation for SSL pretraining logging.
"""

import torch
from typing import Optional


def create_visualizations(x: torch.Tensor, y: torch.Tensor, pred: torch.Tensor, mask: Optional[torch.Tensor], epoch: int):
    """
    Create visualization tensors for logging.

    Args:
        x: Input tensor
        y: Ground truth tensor
        pred: Prediction tensor
        mask: Optional mask tensor
        epoch: Current epoch number

    Returns:
        Tuple of (images, error_images) for logging
    """
    from asparagus.functional.visualization import get_logger_compatible_imgs

    # Hidden-only training does not constrain raw visible predictions. Display the
    # completed image: retained visible input with predictions inserted in hidden voxels.
    completed = torch.where(mask.to(dtype=torch.bool), x, pred) if mask is not None else pred
    images = get_logger_compatible_imgs(
        x,
        y,
        completed,
        slice_dim=1,
        n=1,
        desc=f"Epoch {epoch}",
        titles=["masked input", "target", "completed reconstruction"],
    )

    # Also create error maps
    with torch.no_grad():
        error = (completed - y).abs()
        if mask is not None:
            mask_viz = mask.float()
            # Ensure mask has same shape as error for visualization
            if len(mask_viz.shape) < len(error.shape):
                mask_viz = mask_viz.unsqueeze(1)  # Add channel dim if needed
        else:
            mask_viz = torch.ones_like(error)
        error_images = get_logger_compatible_imgs(
            x,
            error,
            mask_viz,
            slice_dim=1,
            n=1,
            desc=f"Error Map Epoch {epoch}",
            titles=["masked input", "completed abs error", "visible mask"],
        )

    return images, error_images
