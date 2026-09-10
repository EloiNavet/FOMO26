"""Post-hoc calibration fitted on out-of-fold (OOF) predictions.

* Classification: temperature scaling (single scalar T minimising OOF NLL).
* Regression: affine de-biasing for brain age. Age regressors regress toward the mean,
  so OOF predictions satisfy pred ~ a*true + b with a<1. We fit (a, b) and correct test
  predictions as (pred - b) / a, which removes the slope bias and typically lowers MAE.

Fit these on OOF (each fold scoring its own validation fold), then apply at test time.
"""

from __future__ import annotations

import numpy as np
import torch


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
    """Return T>0 minimising NLL of softmax(logits / T). logits [N, C], labels [N]."""
    logits = logits.detach().float()
    labels = labels.detach().long()
    log_T = torch.zeros(1, requires_grad=True)  # optimise log T to keep T>0
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / log_T.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_T.exp().item())


def apply_temperature(logits: torch.Tensor, T: float) -> torch.Tensor:
    return torch.softmax(logits.float() / T, dim=1)


def fit_age_bias(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    """Least-squares slope/intercept of pred ~ a*true + b."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    a, b = np.polyfit(true, pred, 1)
    return float(a), float(b)


def apply_age_bias(pred: np.ndarray, a: float, b: float) -> np.ndarray:
    """Invert pred = a*true + b -> corrected estimate of true."""
    a = a if abs(a) > 1e-6 else 1e-6
    return (np.asarray(pred, dtype=float) - b) / a
