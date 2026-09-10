"""Fold-ensemble segmentation inference for FOMO26.

Loads N finetuned SegmentationModules (one per CV fold, all from the same single backbone),
runs Asparagus sliding-window inference for each, averages the softmax probabilities, optionally
with flip-TTA, then maps back to the original image space via reverse_preprocessing.

Reuses (does not reimplement):
  * model + module instantiation pattern from finetuning/predict_fomo26_seg.py
  * SegmentationModule._sliding_window_predict_padded (overlap=0.5) and reverse_preprocessing
  * SegTestDataset + CPU_seg_test_transforms for matched preprocessing

This is the inference core; a thin container I/O adapter (official FOMO format) plugs in on top.
"""

from __future__ import annotations

import os
import torch
from asparagus.functional.reverse_preprocessing import reverse_preprocessing
from asparagus.functional.utils import fit_patch_size_to_image_size
from asparagus.pipeline.auto_configuration.checkpoint import load_checkpoint_state_dict
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pathlib import Path


def _run_dir_of(record: dict) -> Path:
    return Path(record["run_dir"])


def build_seg_module(run_dir: Path, ckpt_path: Path, n_modalities: int, n_classes: int, device: str):
    """Instantiate one SegmentationModule with weights loaded, on device, in eval mode."""
    ckpt_cfg = OmegaConf.load(run_dir / "hydra" / "config.yaml")
    model = instantiate(
        ckpt_cfg.model._seg_net,
        input_channels=n_modalities,
        output_channels=n_classes,
    )
    module = instantiate(
        ckpt_cfg.lightning._lightning_module,
        model=model,
        weights=load_checkpoint_state_dict(str(ckpt_path)),
        inference_patch_size=ckpt_cfg.training.patch_size,
        test_output_path=os.devnull,
    )
    module.eval()
    module.to(device)
    return module, list(ckpt_cfg.training.patch_size), _runtime_target_spacing_of(ckpt_cfg)


def _runtime_target_spacing_of(ckpt_cfg):
    """Read the physical spacing a fold was trained at, from its own resolved config.

    Preprocessing geometry is part of the model contract: a network fitted on 0.9 mm voxels that is
    handed native-spacing volumes at inference is being asked a different question than the one it
    was trained on, and the resulting score measures the mismatch rather than the model. The patch
    size is already inherited from the checkpoint config for exactly this reason; spacing follows
    the same rule. Resolves to ``None`` for every run predating the setting.
    """
    spacing = OmegaConf.select(ckpt_cfg, "transforms.runtime_target_spacing", default=None)
    if spacing is None:
        return None
    return OmegaConf.to_container(spacing, resolve=True) if OmegaConf.is_config(spacing) else list(spacing)


# Flip-TTA axis sets over the 3 spatial dims (D, H, W) -> tensor dims (2, 3, 4).
_TTA_FLIPS = {
    "none": [()],
    "flip3": [(), (2,), (3,), (4,)],
    "flip7": [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4)],
}


class SegFoldEnsemble:
    def __init__(
        self,
        manifest_records,
        n_modalities,
        n_classes,
        device="cuda",
        tta="none",
        ensemble_space="prob",
        runtime_target_spacing=None,
    ):
        """manifest_records: list of dicts with 'run_dir' and 'best_ckpt' (e.g. from the orchestrator manifest).

        ``ensemble_space`` selects where folds are averaged. ``prob`` (the historical default)
        averages post-softmax probabilities; ``logit`` averages pre-softmax logits, which weights
        confident members more strongly. They are not equivalent, so the choice is explicit and the
        default preserves every existing number.

        ``runtime_target_spacing`` overrides the spacing the folds declare in their own configs.
        Every FOMO26 fold trained to date declares ``None``, so the geometry a task was actually
        fitted at cannot be recovered from the checkpoint config alone; it is supplied by the
        caller from the task registry. ``None`` keeps whatever the checkpoints declare, which is
        the historical behaviour.
        """
        self.device = device
        self.n_classes = n_classes
        if ensemble_space not in ("prob", "logit"):
            raise ValueError(f"ensemble_space must be 'prob' or 'logit', got {ensemble_space!r}.")
        self.ensemble_space = ensemble_space
        self.tta_flips = _TTA_FLIPS[tta]
        self.modules = []
        self.runtime_target_spacing = None
        spacings = []
        # A released container stages every trained fold, so which folds contribute is stated by
        # weight rather than by absence. Weights are positional against ``manifest_records``, which
        # is also the order the modules are built in below, so each weight stays bound to its fold.
        raw_weights = [float(record.get("ensemble_weight", 1.0)) for record in manifest_records]
        if any(weight < 0 for weight in raw_weights) or sum(raw_weights) <= 0:
            raise ValueError("ensemble weights must be non-negative with a positive sum")
        total_weight = sum(raw_weights)
        self.ensemble_weights = [weight / total_weight for weight in raw_weights]
        self.ensemble_folds = [rec.get("fold") for rec in manifest_records]
        for rec in manifest_records:
            ckpt = Path(rec.get("best_ckpt") or rec["last_ckpt"])
            if not ckpt.is_file():
                raise FileNotFoundError(f"Missing checkpoint {ckpt} for fold {rec.get('fold')}.")
            module, patch, spacing = build_seg_module(_run_dir_of(rec), ckpt, n_modalities, n_classes, device)
            self.modules.append(module)
            self.patch_size = patch  # all folds share the same config patch size
            spacings.append(spacing)
        # Folds averaged into one ensemble must have seen one geometry; otherwise their outputs are
        # not defined on a common grid and averaging them is meaningless.
        distinct = {repr(value) for value in spacings}
        if len(distinct) > 1:
            raise ValueError(
                f"Folds were trained at different runtime target spacings ({sorted(distinct)}); "
                "they cannot be ensembled or scored together."
            )
        declared = spacings[0] if spacings else None
        if runtime_target_spacing is not None:
            declared = [float(value) for value in runtime_target_spacing]
        self.runtime_target_spacing = declared

    @torch.no_grad()
    def _probs_one_module(self, module, x):
        """Mean flip-TTA output for a single module, in preprocessed space.

        TTA views are averaged in the configured space for the same reason folds are: averaging
        after the softmax and averaging before it are different estimators.
        """
        patch_size = fit_patch_size_to_image_size(module.inference_patch_size, list(x.shape[2:]))
        acc = None
        for flip in self.tta_flips:
            xf = torch.flip(x, dims=flip) if flip else x
            logits = module._sliding_window_predict_padded(xf, patch_size)
            if flip:
                # Un-mirror before accumulating. Safe only for flip-invariant label spaces, which
                # finetuning/fomo26_inference/tta_safety.py asserts per task before we get here.
                logits = torch.flip(logits, dims=flip)
            view = logits.float() if self.ensemble_space == "logit" else torch.softmax(logits.float(), dim=1)
            acc = view if acc is None else acc + view
        return acc / len(self.tta_flips)

    @torch.no_grad()
    def predict_batch(self, batch):
        """Return (src-space ensemble logits-as-probs, properties, src_label, file_path).

        batch is one item from a SegTestDataset DataLoader (keys: image, properties, src_label, file_path).
        """
        # Apply each module's test transforms once via the first module (deterministic, shared config).
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
            member = self._probs_one_module(module, x)
            weighted = member * weight
            acc = weighted if acc is None else acc + weighted
        ens_probs = acc
        if self.ensemble_space == "logit":
            # Averaged in logit space; convert once, at the end, so the returned tensor is always a
            # probability simplex regardless of how the members were combined.
            ens_probs = torch.softmax(ens_probs, dim=1)

        # Map averaged probabilities back to original image space (same target space for all folds).
        src_probs = reverse_preprocessing(ens_probs, tbatch["properties"])
        return {
            "src_probs": src_probs,
            "properties": tbatch["properties"],
            "src_label": tbatch.get("src_label"),
            "file_path": tbatch.get("file_path"),
        }
