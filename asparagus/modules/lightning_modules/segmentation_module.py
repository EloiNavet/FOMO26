import logging
import numpy as np
import os
import torch
import torch.nn as nn
import torchmetrics.functional
import wandb
from asparagus.functional.metrics.utils import format_multilabel_metrics
from asparagus.functional.reverse_preprocessing import reverse_preprocessing

# Use the asparagus copy of fit_patch_size_to_image_size: identical to gardening_tools' except it never
# floors a sub-32 spatial axis (e.g. a 30-slice volume) to a zero-size patch, which crashed sliding-window
# inference at test time. fit_image_to_patch_size gives the padded target so the image is never smaller
# than the patch.
from asparagus.functional.utils import fit_image_to_patch_size, fit_patch_size_to_image_size
from asparagus.modules.lightning_modules.base_module import BaseModule
from finetuning.fomo26_inference.metrics import dsc_nsd
from gardening_tools.functional.metrics import (
    FN,
    FP,
    TP,
    dice,
    f1,
    jaccard,
    precision,
    sensitivity,
    specificity,
    total_pos_gt,
    total_pos_pred,
    volume_similarity,
)
from gardening_tools.functional.paths.write import save_json
from gardening_tools.modules.losses.deep_supervision import DeepSupervisionLoss
from gardening_tools.modules.losses.DiceCE import DiceCE
from gardening_tools.modules.metrics import GeneralizedDiceScore
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassF1Score
from torchvision import transforms
from typing import Optional

#: Sliding-window logit blending. ``none`` is the historical vendored path (raw ``+=`` accumulation
#: with no normalisation); the other two divide by accumulated weight so overlap bands are not
#: systematically sharpened. Switching away from ``none`` moves every segmentation number.
_WINDOW_BLENDING_MODES = ("none", "uniform", "gaussian")


def symmetric_pad_to_spatial_shape(x: torch.Tensor, target: list[int] | tuple[int, ...]):
    """Symmetrically zero-pad a ``[B,C,*spatial]`` tensor and return its exact unpad slice."""
    spatial = [int(value) for value in x.shape[2:]]
    target = [int(value) for value in target]
    if len(spatial) != len(target) or any(wanted < current for current, wanted in zip(spatial, target)):
        raise ValueError(f"target spatial shape {target} must dominate input shape {spatial}")
    lower = [(wanted - current) // 2 for current, wanted in zip(spatial, target)]
    upper = [wanted - current - low for current, wanted, low in zip(spatial, target, lower)]
    padding = []
    for low, high in reversed(list(zip(lower, upper))):
        padding.extend((low, high))
    padded = nn.functional.pad(x, padding) if any(padding) else x
    crop = (slice(None), slice(None)) + tuple(slice(low, low + size) for low, size in zip(lower, spatial))
    return padded, crop, tuple(zip(lower, upper))


class SegmentationModule(BaseModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-2,
        warmup_epochs: int = 10,
        decoder_warmup_epochs: int = 0,
        cosine_period_ratio: float = 1,
        compile_mode: str = None,
        weights: dict = None,
        deep_supervision: bool = False,
        train_transforms: Optional[transforms.Compose] = None,
        test_transforms: Optional[transforms.Compose] = None,
        val_transforms: Optional[transforms.Compose] = None,
        optimizer: str = "SGD",
        inference_patch_size: list = [],
        inference_mode: str | None = None,
        sliding_window_overlap: float = 0.5,
        window_blending: str = "none",
        test_output_path: str = None,
        log_image_every_n_epochs: int = 50,
        weight_decay: float = 3e-5,
        nesterov: bool = True,
        momentum: float = 0.99,
        load_decoder: bool = True,
        repeat_stem_weights: bool = True,
    ):
        super().__init__(
            model=model,
            learning_rate=learning_rate,
            warmup_epochs=warmup_epochs,
            decoder_warmup_epochs=decoder_warmup_epochs,
            cosine_period_ratio=cosine_period_ratio,
            compile_mode=compile_mode,
            weights=weights,
            optimizer=optimizer,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            weight_decay=weight_decay,
            nesterov=nesterov,
            momentum=momentum,
            load_decoder=load_decoder,
            repeat_stem_weights=repeat_stem_weights,
        )
        self.inference_patch_size = inference_patch_size
        self.inference_mode = inference_mode
        self.sliding_window_overlap = float(sliding_window_overlap)
        if not 0.0 <= self.sliding_window_overlap < 1.0:
            raise ValueError(f"sliding_window_overlap must be in [0, 1), got {self.sliding_window_overlap}.")
        self.window_blending = str(window_blending)
        if self.window_blending not in _WINDOW_BLENDING_MODES:
            raise ValueError(f"window_blending must be one of {sorted(_WINDOW_BLENDING_MODES)}, got {self.window_blending!r}.")
        self.test_output_path = test_output_path
        self.num_classes = model.num_classes
        self.log_image_every_n_epochs = log_image_every_n_epochs
        self.deep_supervision = deep_supervision

        self.train_metrics = self.configure_metrics("train")
        self.val_metrics = self.configure_metrics("val")

        self.train_loss = DiceCE()
        self.val_loss = DiceCE()

        if self.deep_supervision:
            self.train_loss = DeepSupervisionLoss(loss=self.train_loss, weights=None)

    def configure_metrics(self, prefix: str):
        return MetricCollection(
            {
                f"{prefix}/dice": GeneralizedDiceScore(
                    num_classes=self.num_classes,
                    weight_type="linear",
                    per_class=True,
                    input_format="index",
                ),
                f"{prefix}/F1": MulticlassF1Score(
                    num_classes=self.num_classes,
                    ignore_index=0 if self.num_classes > 1 else None,
                    average=None,
                ),
            },
        )

    def training_step(self, batch, batch_idx):
        batch = self._apply_train_transforms(batch)
        x, y = batch["image"], batch["label"]

        pred = self.model(x)
        loss = self.train_loss(pred, y)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )

        if self.deep_supervision:
            # If deep_supervision is enabled output and target will be a list of
            # (downsampled) tensors. We only need the original ground truth and
            # its corresponding prediction which is always the first entry in each list.
            pred = pred[0]
            y = y[0]

        metrics = self.train_metrics(pred, y.squeeze(1))
        self.log_dict(
            format_multilabel_metrics(metrics, ignore_index=self.ignore_index_in_metrics),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )
        if (
            self.current_epoch > 0
            and batch_idx == 0
            and wandb.run is not None
            and self.current_epoch % self.log_image_every_n_epochs == 0
        ):
            self._log_dict_of_images_to_wandb(
                {
                    "input": x.detach().cpu().to(torch.float32).numpy(),
                    "target": y.detach().cpu().to(torch.float32).numpy(),
                    "output": pred.detach().cpu().to(torch.float32).numpy(),
                    "file": batch["file_path"],
                },
                log_key="train",
                task_type="segmentation",
            )

        return loss

    def validation_step(self, batch, batch_idx):
        batch = self._apply_val_transforms(batch)
        x, y = batch["image"], batch["label"]
        pred = self.model(x)
        loss = self.val_loss(pred, y)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )

        metrics = self.val_metrics(pred, y.squeeze(1))
        self.log_dict(
            format_multilabel_metrics(metrics, ignore_index=self.ignore_index_in_metrics),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )
        if (
            self.current_epoch > 0
            and batch_idx == 0
            and wandb.run is not None
            and self.current_epoch % self.log_image_every_n_epochs == 0
        ):
            self._log_dict_of_images_to_wandb(
                {
                    "input": x.detach().cpu().to(torch.float32).numpy(),
                    "target": y.detach().cpu().to(torch.float32).numpy(),
                    "output": pred.detach().cpu().to(torch.float32).numpy(),
                    "file": batch["file_path"],
                },
                log_key="val",
                task_type="segmentation",
            )

    def on_test_epoch_start(self):
        self.test_metrics = [
            dice,
            f1,
            jaccard,
            precision,
            sensitivity,
            specificity,
            TP,
            FP,
            FN,
            total_pos_gt,
            total_pos_pred,
            volume_similarity,
        ]
        self.results = {}
        return super().on_test_epoch_start()

    def _sliding_window_predict_padded(self, x, patch_size):
        """Sliding-window inference that pads the image up to the patch size before prediction.

        The encoder requires patches that are multiples of its total stride; when a spatial axis is
        smaller than the patch (e.g. a 30-slice volume vs a 32-deep patch) the sliding window extracts a
        degenerate boundary patch (negative start index) and crashes inside the residual blocks. Pad each
        axis up to at least the patch size, predict, then crop the logits back to the original extent.
        """
        spatial = list(x.shape[2:])
        target = [int(t) for t in fit_image_to_patch_size(patch_size, spatial)]
        crop = (slice(None), slice(None)) + tuple(slice(0, size) for size in spatial)
        if target != spatial:
            x, crop, pad_pairs = symmetric_pad_to_spatial_shape(x, target)
            logging.info(
                "Symmetrically padding image from %s to %s (pairs=%s) for sliding-window patch %s.",
                spatial,
                target,
                pad_pairs,
                list(patch_size),
            )
        if self.window_blending == "none":
            # Historical path, preserved bit-for-bit so existing numbers stay reproducible.
            logits = self.model.sliding_window_predict(data=x, patch_size=patch_size, overlap=self.sliding_window_overlap)
        else:
            logits = self._normalized_sliding_window(x, patch_size)
        return logits[crop]

    def _normalized_sliding_window(self, x, patch_size):
        """Sliding window that divides accumulated logits by their accumulated weight.

        The vendored ``gardening_tools`` window accumulates raw logits into a zero canvas with
        ``+=`` and never divides by an overlap count or weight map, so a voxel covered by k windows
        carries k summed logits before the softmax. That systematically sharpens predictions inside
        overlap bands, and it is invisible whenever the volume fits in a single window -- so small
        cases never reveal it.

        ``gaussian`` additionally down-weights each window's border, where the receptive field is
        truncated, which is the standard nnU-Net behaviour.

        NOTE: this changes every segmentation number relative to ``window_blending: none``. It is
        opt-in for exactly that reason; the fold-0 delta must be measured and recorded before a
        campaign adopts it.
        """
        # Same step generator the vendored window uses, so only the accumulation differs.
        from gardening_tools.modules.networks.utils import get_steps_for_sliding_window

        spatial = list(x.shape[2:])
        canvas = torch.zeros((1, self.num_classes, *spatial), device=x.device, dtype=torch.float32)
        weights = torch.zeros((1, 1, *spatial), device=x.device, dtype=torch.float32)
        window_weight = self._window_weight(patch_size, device=x.device)
        steps = get_steps_for_sliding_window(spatial, patch_size, self.sliding_window_overlap)
        px, py, pz = patch_size
        for xs in steps[0]:
            for ys in steps[1]:
                for zs in steps[2]:
                    view = (slice(None), slice(None), slice(xs, xs + px), slice(ys, ys + py), slice(zs, zs + pz))
                    out = self.model.forward(x[view]).float()
                    canvas[view] += out * window_weight
                    weights[:, :, xs : xs + px, ys : ys + py, zs : zs + pz] += window_weight[0, 0]
        # Every voxel is covered by at least one window, so the clamp only guards against a zero
        # produced by an all-zero Gaussian tail rather than by missing coverage.
        return canvas / weights.clamp_min(torch.finfo(torch.float32).eps)

    def _window_weight(self, patch_size, *, device):
        if self.window_blending == "uniform":
            return torch.ones((1, 1, *patch_size), device=device, dtype=torch.float32)
        # Separable Gaussian centred on the patch, sigma = 1/8 of the extent (nnU-Net's choice),
        # normalised to a peak of 1 so a single-window case is an exact no-op.
        axes = []
        for extent in patch_size:
            coords = torch.arange(extent, device=device, dtype=torch.float32)
            centre = (extent - 1) / 2.0
            sigma = max(extent / 8.0, 1e-6)
            axes.append(torch.exp(-((coords - centre) ** 2) / (2 * sigma**2)))
        weight = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
        weight = weight / weight.max()
        # A hard zero would make an uncovered voxel undefined; keep the tail strictly positive.
        return weight.clamp_min(1e-3)[None, None]

    def test_step(self, batch, batch_idx):
        batch = self._apply_test_transforms(batch)
        x = batch["image"]

        patch_size = fit_patch_size_to_image_size(self.inference_patch_size, list(x.shape[2:]))
        logits = self._sliding_window_predict_padded(x, patch_size)

        src_logits = reverse_preprocessing(logits, batch["properties"])
        src_label = batch["src_label"]
        metrics = self.compute_metrics_from_confusion_matrix(src_logits, src_label)
        properties = batch["properties"]
        properties = properties[0] if isinstance(properties, (list, tuple)) else properties
        spacing = None
        if isinstance(properties, dict):
            for key in ("spacing", "itk_spacing", "original_spacing", "spacing_after_resampling"):
                if properties.get(key) is not None:
                    spacing = tuple(float(value) for value in properties[key])
                    break
        target = src_label
        if isinstance(target, (list, tuple)):
            target = target[0]
        if target.ndim == src_logits.ndim:
            target = target.squeeze(1)
        challenge_metrics = dsc_nsd(
            src_logits.argmax(dim=1).cpu(),
            target.cpu(),
            src_logits.shape[1],
            spacing=spacing,
        )
        for class_index, values in challenge_metrics.items():
            metrics[str(class_index)].update(values)
        self.results[batch["file_path"]] = metrics

    def on_test_epoch_end(self):
        avg_results = {}
        first_file = list(self.results.keys())[0]
        logging.info(f"Test results for {len(self.results)} files:")
        for label in self.results[first_file].keys():
            avg_results[label] = {}
            for metric in self.results[first_file][label].keys():
                avg_results[label][metric] = round(
                    np.nanmean([self.results[path][label][metric] for path in self.results]),
                    4,
                )
                logging.info(f"{label} {metric}: {avg_results[label][metric]}")
        self.results["mean"] = avg_results
        os.makedirs(os.path.split(self.test_output_path)[0], exist_ok=True)
        save_json(self.results, self.test_output_path)

    def predict_step(self, batch, batch_idx):
        batch = self._apply_test_transforms(batch)
        x = batch["image"]
        patch_size = fit_patch_size_to_image_size(self.inference_patch_size, list(x.shape[2:]))
        logits = self._sliding_window_predict_padded(x, patch_size)
        logits = reverse_preprocessing(
            array=logits,
            image_properties=batch["properties"],
        )
        batch["logits"] = logits
        return batch

    def compute_metrics_from_confusion_matrix(self, logits, label):
        metrics = {}
        labels = logits.shape[1]
        cmat = torchmetrics.functional.confusion_matrix(
            logits, label.squeeze(1), task="multiclass", num_classes=logits.shape[1]
        )
        for label in range(labels):
            metrics_for_label = {}
            tp = cmat[label, label]
            fp = torch.sum(cmat[:, label]) - tp
            fn = torch.sum(cmat[label, :]) - tp
            tn = torch.sum(cmat) - tp - fp - fn
            for metric in self.test_metrics:
                metrics_for_label[metric.__name__] = float(metric(tp, fp, tn, fn))
            metrics[str(label)] = metrics_for_label
        return metrics
