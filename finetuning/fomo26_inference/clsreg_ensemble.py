"""Fold-ensemble classification / regression inference for FOMO26.

cls/reg use a single forward pass over a ``training.target_size`` crop (no sliding
window). We average softmax probabilities (classification) or raw outputs
(regression) across folds, optionally with flip-TTA and deterministic cross-patch
crop views.

Reuses the model + module instantiation from finetune_cls.py.
"""

from __future__ import annotations

import os
import torch
from asparagus.pipeline.auto_configuration.checkpoint import load_checkpoint_state_dict
from finetuning.fomo26_inference.cross_patch import CROSS_PATCH_CHOICES, iter_cross_patch_views
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pathlib import Path

_TTA_FLIPS = {
    "none": [()],
    "flip3": [(), (2,), (3,), (4,)],
    "flip7": [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4)],
}


def build_clsreg_module(run_dir: Path, ckpt_path: Path, n_modalities: int, n_classes: int, device: str):
    """Instantiate one Classification/Regression module with weights loaded, eval, on device.

    Returns (module, target_size).
    """
    ckpt_cfg = OmegaConf.load(run_dir / "hydra" / "config.yaml")
    model = instantiate(
        ckpt_cfg.model._cls_net,
        input_channels=n_modalities,
        output_channels=n_classes,
    )
    module = instantiate(
        ckpt_cfg.lightning._lightning_module,
        model=model,
        weights=load_checkpoint_state_dict(str(ckpt_path)),
        test_output_path=os.devnull,
    )
    module.eval()
    module.to(device)
    return module, list(ckpt_cfg.training.target_size)


class ClsRegFoldEnsemble:
    def __init__(
        self,
        manifest_records,
        n_modalities,
        n_classes,
        kind,
        device="cuda",
        tta="none",
        temperature=1.0,
        cross_patch="none",
    ):
        self.device = device
        self.kind = kind  # "cls" or "reg"
        self.n_classes = n_classes
        self.temperature = float(temperature)
        self.tta_flips = _TTA_FLIPS[tta]
        if cross_patch not in CROSS_PATCH_CHOICES:
            raise ValueError(f"Unsupported cross_patch={cross_patch!r}; expected one of {CROSS_PATCH_CHOICES}.")
        self.cross_patch = cross_patch
        self.modules = []
        # A released container stages every trained fold, so which folds contribute is stated by
        # weight rather than by absence. Weights are positional against ``manifest_records``, which
        # is also the order the modules are built in below, so each weight stays bound to its fold.
        raw_weights = [float(record.get("ensemble_weight", 1.0)) for record in manifest_records]
        if any(weight < 0 for weight in raw_weights) or sum(raw_weights) <= 0:
            raise ValueError("ensemble weights must be non-negative with a positive sum")
        total_weight = sum(raw_weights)
        self.ensemble_weights = [weight / total_weight for weight in raw_weights]
        self.ensemble_folds = [rec.get("fold") for rec in manifest_records]
        self.target_size = None
        for rec in manifest_records:
            ckpt = Path(rec.get("best_ckpt") or rec["last_ckpt"])
            if not ckpt.is_file():
                raise FileNotFoundError(f"Missing checkpoint {ckpt} for fold {rec.get('fold')}.")
            module, target = build_clsreg_module(Path(rec["run_dir"]), ckpt, n_modalities, n_classes, device)
            self.modules.append(module)
            self.target_size = target

    @torch.no_grad()
    def _one_module_output(self, module, x):
        """Mean over cross-patch views and flip-TTA for a single module."""
        acc = None
        n_views = 0
        for crop in iter_cross_patch_views(x, self.target_size, self.cross_patch):
            for flip in self.tta_flips:
                xf = torch.flip(crop, dims=flip) if flip else crop
                out = module.model(xf)
                out = torch.softmax(out.float() / self.temperature, dim=1) if self.kind == "cls" else out.float()
                acc = out if acc is None else acc + out
                n_views += 1
        return acc / n_views

    @torch.no_grad()
    def predict_batch(self, batch):
        """Return dict with ensemble output and the label.

        cls -> 'probs' [B, C] (averaged softmax); reg -> 'pred' [B] (averaged output).
        In ``cross_patch=none`` the dataset should apply the historical center-crop
        transform. In cross-patch modes it should apply normalize+pad only, so this
        class can crop deterministic spatial views.
        """
        m0 = self.modules[0]
        tbatch = m0._apply_test_transforms(batch)
        x = tbatch["image"].to(self.device)

        acc = None
        weights = getattr(self, "ensemble_weights", None)
        if weights is None:
            weights = [1.0 / len(self.modules)] * len(self.modules)
        for module, weight in zip(self.modules, weights, strict=True):
            # A zero-weight fold is staged for provenance, not for inference. Skipping it is what
            # multiplying by zero is supposed to mean, and unlike the multiplication it cannot let
            # a non-finite member leak into the sum.
            if weight == 0.0:
                continue
            out = self._one_module_output(module, x)
            weighted = out * weight
            acc = weighted if acc is None else acc + weighted
        ens = acc

        result = {"file_path": tbatch.get("file_path"), "label": tbatch.get("CLSREG_label")}
        if self.kind == "cls":
            result["probs"] = ens  # [B, C]
        else:
            result["pred"] = ens.squeeze(-1) if ens.ndim > 1 else ens  # [B]
        return result
