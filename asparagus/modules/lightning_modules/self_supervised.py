import copy
import logging
import math
import torch
import torch.nn as nn
from asparagus.functional.metrics import (
    distribution as dist_metrics,
    embedding_projection as embedding_projection,
    features as feat_metrics,
    loss as loss_metrics,
    masking as masking,
    performance as perf_metrics,
    reconstruction as recon_metrics,
    stability as stability_metrics,
    visualization,
)
from asparagus.functional.scanner_targets import SCANNER_IGNORE_INDEX
from asparagus.functional.visualization import log_images_to_logger
from asparagus.modules.datasets.PretrainDataset import PretrainDataset
from asparagus.modules.lightning_modules.base_module import BaseModule
from asparagus.modules.lightning_modules.ssl import (
    SSLValidationMonitorMixin,
    metadata as ssl_metadata,
    schedules as ssl_schedules,
    views as ssl_views,
)
from asparagus.modules.lightning_modules.ssl.objectives import (
    reconstruction as obj_reconstruction,
)
from torchvision import transforms
from typing import NamedTuple, Optional, Sequence


class DemographicToken(NamedTuple):
    """A demographic candidate-set key.

    Structural modalities resolve to ``(name, modality_id, None, None)``; DWI b-value
    subtypes resolve to ``(name, dwi_id, bval_min, bval_max)`` so their candidate sets stay
    on a single diffusion contrast. The name (e.g. ``t1w`` or ``dwi_b1000``) is the W&B
    ``by_modality/<name>`` key and the loss-module dict key.
    """

    name: str
    modality_id: int
    bval_min: Optional[float]
    bval_max: Optional[float]

    @property
    def bval_range(self) -> Optional[tuple[float, float]]:
        if self.bval_min is None or self.bval_max is None:
            return None
        return (float(self.bval_min), float(self.bval_max))

    @property
    def loss_key(self) -> str:
        """Internal ``_contrastive_losses_demo`` / state-dict key.

        Structural modalities keep the legacy ``str(modality_id)`` key so existing
        demographic checkpoints/W&B state load unchanged; DWI subtypes (which share the
        ``dwi`` modality id) use their unique token name to avoid collisions.
        """
        return self.name if self.bval_range is not None else str(self.modality_id)


class SelfSupervisedModule(SSLValidationMonitorMixin, BaseModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        log_images_every_n_epoch: int = 5,
        warmup_epochs: int = 10,
        warmup_steps: int = None,
        cosine_period_ratio: float = 1,
        compile_mode: str = None,
        rec_loss_masked_only: bool = True,
        mse_foreground_aware: bool = False,
        mse_background_weight: float = 0.1,
        mse_foreground_threshold: float = 0.0,
        mse_foreground_dynamic_quantiles: Optional[tuple[float, float]] = None,
        mse_foreground_dynamic_scale: float = 0.1,
        mse_foreground_sample_normalize: bool = True,
        mse_foreground_mode: Optional[str] = None,
        mse_foreground_bonus_weight: float = 0.0,
        mse_foreground_bonus_patch_size: Optional[tuple[int, ...]] = None,
        mse_foreground_min_fraction: float = 0.1,
        mse_foreground_bonus_start_step: int = 0,
        mse_foreground_bonus_warmup_steps: int = 0,
        mse_falcon_hard_weight: float = 0.0,
        mse_falcon_latent_weight: float = 0.0,
        mse_falcon_spectral_weight: float = 0.0,
        mse_falcon_patch_size: Optional[tuple[int, ...]] = None,
        mse_falcon_topk_fraction: float = 0.25,
        mse_falcon_foreground_alpha: float = 1.0,
        mse_falcon_error_gamma: float = 1.0,
        mse_falcon_gradient_eta: float = 0.5,
        mse_falcon_hard_loss: str = "mse",
        mse_falcon_charbonnier_eps: float = 1e-3,
        mse_falcon_latent_level: int = -2,
        mse_falcon_latent_loss: str = "mse",
        mse_falcon_spectral_high_weight: float = 1.0,
        mse_falcon_spectral_power: float = 2.0,
        mse_falcon_spectral_focal_alpha: float = 1.0,
        mse_falcon_spectral_focal_weight_max: float = 10.0,
        mse_falcon_start_step: int = 0,
        mse_falcon_warmup_steps: int = 0,
        mse_falcon_end_step: int = 0,
        mse_falcon_decay_steps: int = 0,
        train_transforms: Optional[transforms.Compose] = None,
        unmasked_transforms: Optional[transforms.Compose] = None,
        test_transforms: Optional[transforms.Compose] = None,
        val_transforms: Optional[transforms.Compose] = None,
        momentum_transforms: Optional[transforms.Compose] = None,
        demo_cpu_transforms: Optional[transforms.Compose] = None,
        optimizer: str = "AdamW",
        mlflow_logging: bool = False,
        log_every_n_steps: int = 50,
        weight_decay: float = 3e-5,
        nesterov: bool = True,
        momentum: float = 0.99,
        moco_momentum: float = 0.999,
        loss_weight_mse: float = 1.0,
        loss_weight_demo: float = 0.1,
        loss_weight_patho: float = 0.1,
        loss_weight_scanner_pos: float = 0.0,
        loss_weight_inv_adv: float = 0.0,
        loss_weight_frequency: float = 0.0,
        loss_weight_wavelet: float = 0.0,
        loss_weight_spatial_detail: float = 0.0,
        loss_weight_stage1_anatomy: float = 0.0,
        loss_weight_stage1_variance: float = 0.0,
        loss_weight_stage1_covariance: float = 0.0,
        loss_weight_stage1_sigreg: float = 0.0,
        loss_weight_stage1_same_modality_hard_negative: float = 0.0,
        loss_weight_stage1_orthogonality: float = 0.0,
        loss_weight_stage1_modality_adversary: float = 0.0,
        loss_weight_stage2_anatomy: float = 0.0,
        loss_weight_stage2_modality: float = 0.0,
        loss_weight_stage2_sigreg: float = 0.0,
        loss_weight_stage2_adv_mod_on_anat: float = 0.0,
        loss_weight_stage2_xcov: float = 0.0,
        loss_weight_stage2_self_reconstruction: float = 0.0,
        loss_weight_stage2_cross_reconstruction: float = 0.0,
        enable_mse_loss: bool = True,
        enable_frequency_loss: bool = False,
        enable_wavelet_loss: bool = False,
        enable_spatial_detail_loss: bool = False,
        enable_demo_loss: bool = False,
        enable_patho_loss: bool = False,
        enable_scanner_pos_loss: bool = False,
        enable_inv_adv_loss: bool = False,
        enable_stage1_loss: bool = False,
        enable_stage1_orthogonality_loss: bool = False,
        enable_stage1_modality_adversary_loss: bool = False,
        enable_stage2_loss: bool = False,
        stage1_include_registered: bool = True,
        stage1_temperature: float = 0.1,
        stage1_queue_size: int = 4096,
        stage1_variance_target_std: float = 1.0,
        stage1_sigreg_enabled: bool = False,
        stage1_sigreg_num_slices: int = 256,
        stage1_cross_modal_candidates_only: bool = False,
        stage1_same_modality_hard_negative_margin: float = 0.0,
        stage1_longitudinal_positive_mode: str = "neutral",
        stage1_cross_session_different_modality_weight: float = 0.25,
        stage1_cross_session_same_modality_weight: float = 0.05,
        stage1_targeted_hard_negative_queue_enabled: bool = False,
        stage1_targeted_hard_negative_queue_size_per_modality: int = 128,
        stage1_targeted_hard_negative_top_k: int = 16,
        stage1_targeted_hard_negative_exclude_same_subject: bool = True,
        stage1_targeted_hard_negative_same_modality_only: bool = True,
        stage1_orthogonality_mode: str = "cross_correlation",
        stage1_orthogonality_margin: float = 0.0,
        stage1_modality_adversary_hidden_dim: int = 128,
        stage1_modality_adversary_grl_lambda: float = 1.0,
        stage1_modality_adversary_label_smoothing: float = 0.0,
        stage2_registered_only: bool = True,
        stage2_repel_margin: float = 0.0,
        stage2_anat_temperature: float = 0.1,
        stage2_sigreg_num_slices: int = 256,
        stage2_grl_lambda: float = 0.0,
        stage2_grl_schedule: str = "constant",
        stage2_grl_warmup_steps: int = 0,
        demographic_mode: str = "dufumier_yaware",
        demographic_symmetric: bool = False,
        demographic_eligible_pathology: int = 0,
        demographic_modalities: Optional[list[int]] = None,
        demographic_tokens: Optional[list[dict]] = None,
        pathology_modalities: Optional[list[int]] = None,
        pathology_eligible_classes: Optional[list[int]] = None,
        pathology_exclude_classes: Optional[list[int]] = None,
        pathology_unknown_ignore: bool = True,
        pathology_fine_label_source: Optional[str] = None,
        stage2_reconstruction_loss: str = "smooth_l1",
        stage2_smooth_l1_beta: float = 0.1,
        stage2_decoder_enabled: bool = False,
        stage2_reconstruction_foreground_only: bool = True,
        stage2_max_cross_pairs_per_anchor: int = 1,
        stage2_max_cross_pairs_per_batch: int = 8,
        stage2_cross_pair_sampling: str = "random",
        stage2_pair_max_offset_mm: float = 1.0,
        stage2_cross_reconstruction_start_step: int = 0,
        stage2_cross_reconstruction_warmup_steps: int = 0,
        frequency_loss_high_weight: float = 1.0,
        frequency_loss_power: float = 2.0,
        frequency_loss_mode: str = "legacy_log_magnitude",
        frequency_loss_weighting: str | None = None,
        frequency_loss_low_cutoff: float = 1.0 / 3.0,
        frequency_loss_high_cutoff: float = 2.0 / 3.0,
        frequency_loss_low_weight: float = 1.0,
        frequency_loss_mid_weight: float = 1.0,
        frequency_loss_high_band_weight: float = 2.0,
        frequency_loss_focal_alpha: float = 1.0,
        frequency_loss_focal_weight_max: float = 10.0,
        frequency_loss_log_focal_eps: float = 1e-8,
        wavelet_family: str = "haar",
        wavelet_levels: int = 2,
        wavelet_level_weights: Optional[Sequence[float]] = None,
        wavelet_support_mode: str = "touched",
        wavelet_include_lowpass: bool = False,
        wavelet_loss: str = "l1",
        wavelet_eps: float = 1e-8,
        spatial_detail_beta: float = 0.1,
        frequency_start_step: int = 0,
        frequency_warmup_steps: int = 1890,
        wavelet_start_step: int = 0,
        wavelet_warmup_steps: int = 1890,
        wavelet_end_step: int = 0,
        wavelet_decay_steps: int = 0,
        spatial_detail_start_step: int = 0,
        spatial_detail_warmup_steps: int = 1890,
        demo_start_step: int = 0,
        demo_warmup_steps: int = 1890,
        demographic_sigreg_enabled: bool = False,
        demographic_sigreg_weight: float = 0.0,
        demographic_sigreg_num_slices: int = 256,
        patho_start_step: int = 0,
        patho_warmup_steps: int = 1890,
        scanner_pos_start_step: int = 0,
        scanner_pos_warmup_steps: int = 0,
        inv_adv_start_step: int = 0,
        inv_adv_warmup_steps: int = 0,
        scanner_target_weights: Optional[dict[str, float]] = None,
        inv_target_weights: Optional[dict[str, float]] = None,
        scanner_pos_lambda_sigreg: float = 0.0,
        inv_adv_lambda_sigreg: float = 0.0,
        scanner_pos_lambda_norm_cap: float = 0.0,
        inv_adv_lambda_norm_cap: float = 0.0,
        scanner_pos_norm_cap: float = 0.0,
        inv_adv_norm_cap: float = 0.0,
        scanner_pos_label_smoothing: float = 0.0,
        inv_adv_label_smoothing: float = 0.0,
        scanner_ignore_index: int = SCANNER_IGNORE_INDEX,
        scanner_sigreg_num_slices: int = 256,
        inv_sigreg_num_slices: int = 256,
        enable_modality_loss: bool = False,
        loss_weight_modality: float = 0.0,
        modality_ce_enabled: bool = True,
        modality_ce_weight: float = 1.0,
        modality_label_smoothing: float = 0.0,
        modality_class_balancing: str = "none",
        modality_supcon_enabled: bool = False,
        modality_supcon_weight: float = 0.0,
        modality_supcon_temperature: float = 0.1,
        modality_supcon_require_diff_subject: bool = True,
        modality_sigreg_enabled: bool = False,
        modality_sigreg_weight: float = 0.0,
        modality_sigreg_num_slices: int = 256,
        modality_ignore_index: int = -100,
        modality_ignore_modality_ids: Optional[Sequence[int]] = None,
        inv_grl_lambda: float = 0.0,
        inv_grl_schedule: str = "constant",
        inv_grl_warmup_steps: int = 0,
        stage1_start_step: int = 0,
        stage1_warmup_steps: int = 1890,
        stage1_orthogonality_start_step: int = 0,
        stage1_orthogonality_warmup_steps: int = 0,
        stage1_modality_adversary_start_step: int = 0,
        stage1_modality_adversary_warmup_steps: int = 0,
        # Schedule for the stage-1 modality objective itself. Defaults reproduce the historical
        # behaviour exactly (active from step 0 at full weight), so no existing arm changes.
        modality_start_step: int = 0,
        modality_warmup_steps: int = 0,
        # Fail loudly when a stacked objective is enabled but never receives an eligible
        # batch. 0 disables the check, which is the historical behaviour.
        component_starvation_window: int = 0,
        batch_routing_log_every_n_steps: int = 0,
        stage2_start_step: int = 0,
        stage2_warmup_steps: int = 1890,
        validation_embedding_monitor_enabled: bool = False,
        validation_embedding_sources: Optional[list[str]] = None,
        validation_embedding_reducers: Optional[list[str]] = None,
        validation_embedding_max_points: int = 1024,
        validation_embedding_random_state: int = 0,
        demographic_silhouette_age_bin_years: int = 10,
        validation_mask_seed: int = 0,
        loss_gradient_norm_every_n_steps: int = 0,
        curriculum_schedules: dict | None = None,
        reconstruction_data_range: float = 6.0,
        log_raw_reconstruction_metrics: bool = False,
        metric_semantics_version: str = "current",
        probe_every_n_epoch: int = 5,
        probe_detailed_metrics: bool = False,
        exhaustive_evaluation: bool = False,
        matched_control: Optional[dict] = None,
        weights: dict = None,
    ):
        super().__init__(
            model=model,
            warmup_epochs=warmup_epochs,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            cosine_period_ratio=cosine_period_ratio,
            compile_mode=compile_mode,
            optimizer=optimizer,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            weight_decay=weight_decay,
            nesterov=nesterov,
            momentum=momentum,
            weights=weights,
        )

        # BaseModule compiles __call__; SSL uses custom forwards, so compile those routes explicitly.
        object.__setattr__(self, "_online_model", self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model)
        self._compiled_forward_with_features = None
        self._compiled_forward_multimodal_ssl = None
        if compile_mode is not None:
            self._compiled_forward_with_features = torch.compile(self._online_model.forward_with_features, mode=compile_mode)
            if hasattr(self._online_model, "forward_multimodal_ssl"):
                self._compiled_forward_multimodal_ssl = torch.compile(
                    self._online_model.forward_multimodal_ssl, mode=compile_mode
                )

        self._rec_loss_fn = nn.MSELoss(reduction="mean")
        self.rec_loss_masked_only = rec_loss_masked_only
        legacy_mse_foreground_aware = bool(mse_foreground_aware)
        requested_mse_foreground_mode = None if mse_foreground_mode is None else str(mse_foreground_mode)
        if requested_mse_foreground_mode is None or (requested_mse_foreground_mode == "none" and legacy_mse_foreground_aware):
            self.mse_foreground_mode = "weighted_replace" if legacy_mse_foreground_aware else "none"
            if legacy_mse_foreground_aware:
                logging.warning(
                    "losses.mse.foreground_aware=true is legacy; use "
                    "losses.mse.foreground_mode=weighted_replace for reproduction or bonus_voxel/bonus_patch for new runs."
                )
        else:
            self.mse_foreground_mode = requested_mse_foreground_mode
            if legacy_mse_foreground_aware and self.mse_foreground_mode != "weighted_replace":
                logging.warning(
                    "Ignoring legacy losses.mse.foreground_aware=true because losses.mse.foreground_mode=%s is set.",
                    self.mse_foreground_mode,
                )
        if self.mse_foreground_mode not in {
            "none",
            "weighted_replace",
            "bonus_voxel",
            "bonus_patch",
            "falcon_hard",
            "falcon_latent",
            "falcon_full",
        }:
            raise ValueError(
                "Unsupported mse_foreground_mode="
                f"{self.mse_foreground_mode!r}; expected none, weighted_replace, bonus_voxel, bonus_patch, "
                "falcon_hard, falcon_latent, or falcon_full."
            )
        self.mse_foreground_aware = self.mse_foreground_mode == "weighted_replace"
        self.mse_background_weight = float(mse_background_weight)
        self.mse_foreground_threshold = float(mse_foreground_threshold)
        self.mse_foreground_dynamic_quantiles = tuple(mse_foreground_dynamic_quantiles or (0.02, 0.98))
        self.mse_foreground_dynamic_scale = float(mse_foreground_dynamic_scale)
        self.mse_foreground_sample_normalize = bool(mse_foreground_sample_normalize)
        self.mse_foreground_bonus_weight = float(mse_foreground_bonus_weight)
        self.mse_foreground_bonus_patch_size = tuple(int(v) for v in (mse_foreground_bonus_patch_size or (4,)))
        self.mse_foreground_min_fraction = float(mse_foreground_min_fraction)
        self.mse_foreground_bonus_start_step = max(0, int(mse_foreground_bonus_start_step))
        self.mse_foreground_bonus_warmup_steps = max(0, int(mse_foreground_bonus_warmup_steps))
        self.mse_falcon_hard_weight = float(mse_falcon_hard_weight)
        self.mse_falcon_latent_weight = float(mse_falcon_latent_weight)
        self.mse_falcon_spectral_weight = float(mse_falcon_spectral_weight)
        self.mse_falcon_patch_size = tuple(int(v) for v in (mse_falcon_patch_size or mse_foreground_bonus_patch_size or (4,)))
        self.mse_falcon_topk_fraction = float(mse_falcon_topk_fraction)
        self.mse_falcon_foreground_alpha = float(mse_falcon_foreground_alpha)
        self.mse_falcon_error_gamma = float(mse_falcon_error_gamma)
        self.mse_falcon_gradient_eta = float(mse_falcon_gradient_eta)
        self.mse_falcon_hard_loss = str(mse_falcon_hard_loss)
        self.mse_falcon_charbonnier_eps = float(mse_falcon_charbonnier_eps)
        self.mse_falcon_latent_level = int(mse_falcon_latent_level)
        self.mse_falcon_latent_loss = str(mse_falcon_latent_loss)
        self.mse_falcon_spectral_high_weight = float(mse_falcon_spectral_high_weight)
        self.mse_falcon_spectral_power = float(mse_falcon_spectral_power)
        self.mse_falcon_spectral_focal_alpha = float(mse_falcon_spectral_focal_alpha)
        self.mse_falcon_spectral_focal_weight_max = float(mse_falcon_spectral_focal_weight_max)
        self.mse_falcon_start_step = max(0, int(mse_falcon_start_step))
        self.mse_falcon_warmup_steps = max(0, int(mse_falcon_warmup_steps))
        self.mse_falcon_end_step = max(0, int(mse_falcon_end_step))
        self.mse_falcon_decay_steps = max(0, int(mse_falcon_decay_steps))
        self.demographic_tokens = self._build_demographic_tokens(demographic_tokens, demographic_modalities)
        # The demographic / pathology contrastive objectives are not part of this distribution,
        # so no loss module can be supplied and both branches stay closed.
        self._contrastive_losses_demo = nn.ModuleDict({})
        self._contrastive_loss_patho = None
        self.unmasked_transforms = unmasked_transforms
        self.demo_cpu_transforms = demo_cpu_transforms
        self.log_images_every_n_epoch = log_images_every_n_epoch
        self.mlflow_logging = mlflow_logging
        self.momentum_transforms = momentum_transforms
        self.log_every_n_steps = log_every_n_steps
        self.loss_weight_mse = loss_weight_mse
        self.loss_weight_demo = loss_weight_demo
        self.loss_weight_patho = loss_weight_patho
        self.loss_weight_scanner_pos = float(loss_weight_scanner_pos)
        self.loss_weight_inv_adv = float(loss_weight_inv_adv)
        self.loss_weight_frequency = loss_weight_frequency
        self.loss_weight_wavelet = float(loss_weight_wavelet)
        self.loss_weight_spatial_detail = loss_weight_spatial_detail
        self.loss_weight_stage1_anatomy = loss_weight_stage1_anatomy
        self.loss_weight_stage1_variance = loss_weight_stage1_variance
        self.loss_weight_stage1_covariance = loss_weight_stage1_covariance
        self.loss_weight_stage1_sigreg = float(loss_weight_stage1_sigreg)
        self.loss_weight_stage1_same_modality_hard_negative = loss_weight_stage1_same_modality_hard_negative
        self.loss_weight_stage1_orthogonality = float(loss_weight_stage1_orthogonality)
        self.loss_weight_stage1_modality_adversary = float(loss_weight_stage1_modality_adversary)
        self.loss_weight_stage2_anatomy = loss_weight_stage2_anatomy
        self.loss_weight_stage2_modality = loss_weight_stage2_modality
        self.loss_weight_stage2_sigreg = float(loss_weight_stage2_sigreg)
        self.loss_weight_stage2_adv_mod_on_anat = float(loss_weight_stage2_adv_mod_on_anat)
        self.loss_weight_stage2_xcov = float(loss_weight_stage2_xcov)
        self.loss_weight_stage2_self_reconstruction = float(loss_weight_stage2_self_reconstruction)
        self.loss_weight_stage2_cross_reconstruction = float(loss_weight_stage2_cross_reconstruction)
        self.enable_mse_loss = enable_mse_loss
        self.enable_frequency_loss = enable_frequency_loss
        self.enable_wavelet_loss = bool(enable_wavelet_loss)
        self.enable_spatial_detail_loss = enable_spatial_detail_loss
        self.enable_demo_loss = False
        self.enable_patho_loss = False
        self.enable_scanner_pos_loss = bool(enable_scanner_pos_loss)
        self.enable_inv_adv_loss = bool(enable_inv_adv_loss)
        self.enable_stage1_loss = enable_stage1_loss
        self.enable_stage1_orthogonality_loss = bool(enable_stage1_orthogonality_loss)
        self.enable_stage1_modality_adversary_loss = bool(enable_stage1_modality_adversary_loss)
        self.enable_stage2_loss = enable_stage2_loss
        self.stage1_include_registered = bool(stage1_include_registered)
        self.stage1_temperature = stage1_temperature
        self.stage1_queue_size = int(stage1_queue_size)
        self.stage1_variance_target_std = float(stage1_variance_target_std)
        self.stage1_sigreg_enabled = bool(stage1_sigreg_enabled)
        self.stage1_sigreg_num_slices = int(stage1_sigreg_num_slices)
        self.stage1_cross_modal_candidates_only = bool(stage1_cross_modal_candidates_only)
        self.stage1_same_modality_hard_negative_margin = float(stage1_same_modality_hard_negative_margin)
        self.stage1_longitudinal_positive_mode = str(stage1_longitudinal_positive_mode)
        if self.stage1_longitudinal_positive_mode not in {"neutral", "weighted"}:
            raise ValueError(
                "Unsupported stage1_longitudinal_positive_mode="
                f"{stage1_longitudinal_positive_mode!r}; expected 'neutral' or 'weighted'."
            )
        self.stage1_cross_session_different_modality_weight = float(stage1_cross_session_different_modality_weight)
        self.stage1_cross_session_same_modality_weight = float(stage1_cross_session_same_modality_weight)
        self.stage1_targeted_hard_negative_queue_enabled = bool(stage1_targeted_hard_negative_queue_enabled)
        self.stage1_targeted_hard_negative_queue_size_per_modality = max(
            0,
            int(stage1_targeted_hard_negative_queue_size_per_modality),
        )
        self.stage1_targeted_hard_negative_top_k = max(1, int(stage1_targeted_hard_negative_top_k))
        self.stage1_targeted_hard_negative_exclude_same_subject = bool(stage1_targeted_hard_negative_exclude_same_subject)
        self.stage1_targeted_hard_negative_same_modality_only = bool(stage1_targeted_hard_negative_same_modality_only)
        self.stage2_registered_only = bool(stage2_registered_only)
        self.stage2_repel_margin = float(stage2_repel_margin)
        self.stage2_decoder_enabled = bool(stage2_decoder_enabled)
        self.stage2_reconstruction_foreground_only = bool(stage2_reconstruction_foreground_only)
        self.stage2_max_cross_pairs_per_anchor = max(0, int(stage2_max_cross_pairs_per_anchor))
        self.stage2_max_cross_pairs_per_batch = max(0, int(stage2_max_cross_pairs_per_batch))
        self.stage2_pair_max_offset_mm = float(stage2_pair_max_offset_mm)
        self.stage2_cross_pair_sampling = str(stage2_cross_pair_sampling)
        if self.stage2_cross_pair_sampling not in {"random", "deterministic"}:
            raise ValueError(
                "Unsupported Stage 2 cross-pair sampling="
                f"{self.stage2_cross_pair_sampling!r}; expected 'random' or 'deterministic'."
            )
        if (
            self.enable_stage2_loss
            and (self.loss_weight_stage2_self_reconstruction > 0.0 or self.loss_weight_stage2_cross_reconstruction > 0.0)
            and not self.stage2_decoder_enabled
        ):
            raise ValueError(
                "Stage 2 reconstruction weights require losses.multimodal_stage2.decoder_enabled=true; "
                "set weight_self_reconstruction=0 and weight_cross_reconstruction=0 for heads-only Stage 2."
            )
        if self.stage2_decoder_enabled and not self.enable_stage2_loss:
            raise ValueError("losses.multimodal_stage2.decoder_enabled=true requires losses.multimodal_stage2.enabled=true.")
        if self.stage2_decoder_enabled and not getattr(self._online_model, "stage2_factorized_head_enabled", False):
            raise ValueError(
                "losses.multimodal_stage2.decoder_enabled=true requires model.stage2_factorized_head.enabled=true."
            )
        self.stage2_anat_temperature = float(stage2_anat_temperature)
        self.stage2_sigreg_num_slices = int(stage2_sigreg_num_slices)
        self.stage2_grl_lambda = float(stage2_grl_lambda)
        self.stage2_grl_schedule = str(stage2_grl_schedule)
        if self.stage2_grl_schedule not in {"constant", "linear"}:
            raise ValueError(
                f"Unsupported Stage 2 GRL schedule {self.stage2_grl_schedule!r}; expected 'constant' or 'linear'."
            )
        self.stage2_grl_warmup_steps = max(0, int(stage2_grl_warmup_steps))
        if demographic_mode not in {"moco_yaware", "dufumier_yaware", "inbatch_yaware"}:
            raise ValueError(f"Unsupported demographic_mode={demographic_mode!r}.")
        self.demographic_mode = str(demographic_mode)
        self.demographic_symmetric = bool(demographic_symmetric)
        self.demographic_eligible_pathology = int(demographic_eligible_pathology)
        # Distinct modality ids spanned by the tokens (deduped, order-preserving). Kept for
        # callers that only need ids (e.g. mode/modality validation, FiLM cohorts).
        self.demographic_modalities = tuple(dict.fromkeys(token.modality_id for token in self.demographic_tokens))
        self.pathology_modalities = (
            PretrainDataset.normalize_modality_ids(pathology_modalities, default=()) if pathology_modalities else ()
        )
        self.pathology_eligible_classes = (
            tuple(int(value) for value in pathology_eligible_classes) if pathology_eligible_classes else ()
        )
        self.pathology_exclude_classes = (
            tuple(int(value) for value in pathology_exclude_classes) if pathology_exclude_classes else ()
        )
        self.pathology_unknown_ignore = bool(pathology_unknown_ignore)
        self.pathology_fine_label_source = None if pathology_fine_label_source is None else str(pathology_fine_label_source)
        if self.enable_demo_loss:
            missing_losses = [
                token.name for token in self.demographic_tokens if token.loss_key not in self._contrastive_losses_demo
            ]
            if missing_losses:
                raise ValueError(f"Missing demographic contrastive losses for tokens {missing_losses}.")
        self.demographic_token_names = tuple(token.name for token in self.demographic_tokens)
        self.stage2_reconstruction_loss = stage2_reconstruction_loss
        self.stage2_smooth_l1_beta = float(stage2_smooth_l1_beta)
        self._validate_stage2_configuration()
        self.frequency_loss_high_weight = frequency_loss_high_weight
        self.frequency_loss_power = frequency_loss_power
        self.frequency_loss_mode = str(frequency_loss_mode)
        self.frequency_loss_weighting = None if frequency_loss_weighting is None else str(frequency_loss_weighting)
        self.frequency_loss_low_cutoff = float(frequency_loss_low_cutoff)
        self.frequency_loss_high_cutoff = float(frequency_loss_high_cutoff)
        self.frequency_loss_low_weight = float(frequency_loss_low_weight)
        self.frequency_loss_mid_weight = float(frequency_loss_mid_weight)
        self.frequency_loss_high_band_weight = float(frequency_loss_high_band_weight)
        self.frequency_loss_focal_alpha = float(frequency_loss_focal_alpha)
        self.frequency_loss_focal_weight_max = float(frequency_loss_focal_weight_max)
        self.frequency_loss_log_focal_eps = float(frequency_loss_log_focal_eps)
        self.wavelet_family = str(wavelet_family)
        self.wavelet_levels = int(wavelet_levels)
        self.wavelet_level_weights = tuple(float(value) for value in (wavelet_level_weights or [1.0] * self.wavelet_levels))
        self.wavelet_support_mode = str(wavelet_support_mode)
        self.wavelet_include_lowpass = bool(wavelet_include_lowpass)
        self.wavelet_loss_name = str(wavelet_loss)
        self.wavelet_eps = float(wavelet_eps)
        self.spatial_detail_beta = float(spatial_detail_beta)
        self.frequency_start_step = frequency_start_step
        self.frequency_warmup_steps = frequency_warmup_steps
        self.wavelet_start_step = int(wavelet_start_step)
        self.wavelet_warmup_steps = int(wavelet_warmup_steps)
        self.wavelet_end_step = max(0, int(wavelet_end_step))
        self.wavelet_decay_steps = max(0, int(wavelet_decay_steps))
        self.spatial_detail_start_step = spatial_detail_start_step
        self.spatial_detail_warmup_steps = spatial_detail_warmup_steps
        self.demo_start_step = demo_start_step
        self.demo_warmup_steps = demo_warmup_steps
        self.demographic_sigreg_enabled = bool(demographic_sigreg_enabled)
        self.demographic_sigreg_weight = float(demographic_sigreg_weight)
        self.demographic_sigreg_num_slices = int(demographic_sigreg_num_slices)
        self.patho_start_step = patho_start_step
        self.patho_warmup_steps = patho_warmup_steps
        self.scanner_pos_start_step = max(0, int(scanner_pos_start_step))
        self.scanner_pos_warmup_steps = max(0, int(scanner_pos_warmup_steps))
        self.inv_adv_start_step = max(0, int(inv_adv_start_step))
        self.inv_adv_warmup_steps = max(0, int(inv_adv_warmup_steps))
        self.scanner_target_weights = {str(key): float(value) for key, value in (scanner_target_weights or {}).items()}
        self.inv_target_weights = {str(key): float(value) for key, value in (inv_target_weights or {}).items()}
        self.scanner_pos_lambda_sigreg = float(scanner_pos_lambda_sigreg)
        self.inv_adv_lambda_sigreg = float(inv_adv_lambda_sigreg)
        self.scanner_pos_lambda_norm_cap = float(scanner_pos_lambda_norm_cap)
        self.inv_adv_lambda_norm_cap = float(inv_adv_lambda_norm_cap)
        self.scanner_pos_norm_cap = float(scanner_pos_norm_cap)
        self.inv_adv_norm_cap = float(inv_adv_norm_cap)
        self.scanner_pos_label_smoothing = float(scanner_pos_label_smoothing)
        self.inv_adv_label_smoothing = float(inv_adv_label_smoothing)
        self.scanner_ignore_index = int(scanner_ignore_index)
        self.scanner_sigreg_num_slices = int(scanner_sigreg_num_slices)
        self.inv_sigreg_num_slices = int(inv_sigreg_num_slices)
        self.enable_modality_loss = bool(enable_modality_loss)
        self.loss_weight_modality = float(loss_weight_modality)
        self.modality_ce_enabled = bool(modality_ce_enabled)
        self.modality_ce_weight = float(modality_ce_weight)
        self.modality_label_smoothing = float(modality_label_smoothing)
        self.modality_class_balancing = str(modality_class_balancing)
        self.modality_supcon_enabled = bool(modality_supcon_enabled)
        self.modality_supcon_weight = float(modality_supcon_weight)
        self.modality_supcon_temperature = float(modality_supcon_temperature)
        self.modality_supcon_require_diff_subject = bool(modality_supcon_require_diff_subject)
        self.modality_sigreg_enabled = bool(modality_sigreg_enabled)
        self.modality_sigreg_weight = float(modality_sigreg_weight)
        self.modality_sigreg_num_slices = int(modality_sigreg_num_slices)
        self.modality_ignore_index = int(modality_ignore_index)
        self.modality_ignore_modality_ids = frozenset(int(i) for i in (modality_ignore_modality_ids or ()))
        self.stage1_orthogonality_mode = str(stage1_orthogonality_mode)
        self.stage1_orthogonality_margin = float(stage1_orthogonality_margin)
        if self.enable_stage1_orthogonality_loss and not (self.enable_stage1_loss and self.enable_modality_loss):
            raise ValueError("Stage-1 orthogonality requires both Stage-1 anatomy and Stage-1 modality losses enabled.")
        if self.enable_stage1_modality_adversary_loss and not (self.enable_stage1_loss and self.enable_modality_loss):
            raise ValueError("Stage-1 modality adversary requires both Stage-1 anatomy and Stage-1 modality losses enabled.")
        self.modality_start_step = int(modality_start_step)
        self.modality_warmup_steps = int(modality_warmup_steps)
        self.component_starvation_window = int(component_starvation_window)
        self.batch_routing_log_every_n_steps = int(batch_routing_log_every_n_steps)
        self._component_zero_streak: dict[str, int] = {}
        self.stage1_modality_adversary_hidden_dim = max(1, int(stage1_modality_adversary_hidden_dim))
        self.stage1_modality_adversary_grl_lambda = float(stage1_modality_adversary_grl_lambda)
        self.stage1_modality_adversary_label_smoothing = float(stage1_modality_adversary_label_smoothing)
        self.stage1_modality_adversary = (
            self._build_stage1_modality_adversary(self.stage1_modality_adversary_hidden_dim)
            if self.enable_stage1_modality_adversary_loss
            else None
        )
        self.inv_grl_lambda = float(inv_grl_lambda)
        self.inv_grl_schedule = str(inv_grl_schedule)
        if self.inv_grl_schedule not in {"constant", "linear"}:
            raise ValueError(f"Unsupported inv GRL schedule {self.inv_grl_schedule!r}; expected 'constant' or 'linear'.")
        self.inv_grl_warmup_steps = max(0, int(inv_grl_warmup_steps))
        self.stage1_start_step = stage1_start_step
        self.stage1_warmup_steps = stage1_warmup_steps
        self.stage1_orthogonality_start_step = max(0, int(stage1_orthogonality_start_step))
        self.stage1_orthogonality_warmup_steps = max(0, int(stage1_orthogonality_warmup_steps))
        self.stage1_modality_adversary_start_step = max(0, int(stage1_modality_adversary_start_step))
        self.stage1_modality_adversary_warmup_steps = max(0, int(stage1_modality_adversary_warmup_steps))
        self.stage2_start_step = stage2_start_step
        self.stage2_warmup_steps = stage2_warmup_steps
        self.stage2_cross_reconstruction_start_step = max(0, int(stage2_cross_reconstruction_start_step))
        self.stage2_cross_reconstruction_warmup_steps = max(0, int(stage2_cross_reconstruction_warmup_steps))
        self.validation_embedding_monitor_enabled = bool(validation_embedding_monitor_enabled)
        self.validation_embedding_sources = tuple(validation_embedding_sources or ["h", "z_patho"])
        self.validation_embedding_reducers = tuple(validation_embedding_reducers or ["umap", "tsne", "pca"])
        self.validation_embedding_max_points = int(validation_embedding_max_points)
        self.validation_embedding_random_state = int(validation_embedding_random_state)
        self.demographic_silhouette_age_bin_years = max(1, int(demographic_silhouette_age_bin_years))
        self.validation_mask_seed = int(validation_mask_seed)
        self.loss_gradient_norm_every_n_steps = int(loss_gradient_norm_every_n_steps)
        # Task-6 curriculum overrides, keyed by component slot. Empty by default: with no entry for
        # a slot, get_dynamic_weight() returns the historical cosine ramp unchanged, so every
        # Task-5 configuration keeps exactly the schedule it was frozen with.
        self._curriculum_specs = dict(curriculum_schedules or {})
        self.reconstruction_data_range = float(reconstruction_data_range)
        self.log_raw_reconstruction_metrics = bool(log_raw_reconstruction_metrics)
        self.metric_semantics_version = str(metric_semantics_version)
        self.probe_every_n_epoch = int(probe_every_n_epoch)
        self.probe_detailed_metrics = bool(probe_detailed_metrics)
        self.exhaustive_evaluation = bool(exhaustive_evaluation)
        self._val_embedding_batches = []
        self._projection_embedding_batches = []
        self._probe_reference_batches = []
        self._modality_probe_reference_batches = []
        self._demographic_probe_reference_batches = []
        self._val_metadata_batches = []
        self._val_stage1_batches = []
        self._initial_embedding_monitor_logged = False

        self.register_buffer("_stage1_queue_features", torch.empty(0, 0), persistent=False)
        self.register_buffer("_stage1_queue_subjects", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_stage1_queue_sessions", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_stage1_queue_modalities", torch.empty(0, dtype=torch.long), persistent=False)

        self._stage1_regularization_enabled = (
            self.loss_weight_stage1_variance > 0.0
            or self.loss_weight_stage1_covariance > 0.0
            or (self.stage1_sigreg_enabled and self.loss_weight_stage1_sigreg > 0.0)
            or self.loss_weight_stage1_same_modality_hard_negative > 0.0
            or self.enable_stage1_orthogonality_loss
            or self.enable_stage1_modality_adversary_loss
        )
        self._stage1_head_enabled = self.enable_stage1_loss or self._stage1_regularization_enabled
        self._multimodal_enabled = self._stage1_head_enabled or self.enable_stage2_loss
        # Matched-control forcing (screening campaigns only; both default False so no existing run
        # changes behaviour). A control arm differs from its treatment by ONE loss flag, but those
        # flags also decide the *execution path*: whether the three contrastive views are built (one
        # vs three GPU-augmentation passes per step, which desynchronises the masking RNG) and
        # whether a momentum encoder exists at all. Forcing the path here — and never a loss term —
        # is what makes "same data, same augmentations, same RNG, only the objective differs" true.
        _matched = dict(matched_control or {})
        self._force_contrastive_path = bool(_matched.get("force_contrastive_path", False))
        self._force_momentum_encoder = bool(_matched.get("force_momentum_encoder", False))

        self._contrastive_enabled = (
            self._force_contrastive_path
            or self.enable_demo_loss
            or (self.enable_patho_loss and self._contrastive_loss_patho is not None)
            or self.enable_scanner_pos_loss
            or self.enable_inv_adv_loss
            or self.enable_modality_loss
            or self._multimodal_enabled
        )
        self._uses_momentum_encoder = (
            self._force_momentum_encoder
            or (self.enable_demo_loss and self.demographic_mode == "moco_yaware")
            or self.enable_patho_loss
            or self._multimodal_enabled
        )

        reconstruction_requested = bool(
            self.enable_mse_loss or self.enable_frequency_loss or self.enable_wavelet_loss or self.enable_spatial_detail_loss
        )
        if reconstruction_requested and not getattr(self._online_model, "supports_reconstruction", True):
            raise ValueError(
                f"{type(self._online_model).__name__} is encoder-only and cannot run reconstruction losses. "
                "Disable MSE/frequency/wavelet/spatial-detail, or select a decoder-backed architecture."
            )

        self._set_inactive_ssl_heads_trainability()

        self.moco_momentum = float(moco_momentum)
        self._last_ema_momentum = float(moco_momentum)
        if self._uses_momentum_encoder:
            self.momentum_model = copy.deepcopy(model)
            target = self.momentum_model
            for param in target.parameters():
                param.requires_grad = False
            target.eval()

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Load new encoder-only targets and migrate legacy full EMA checkpoints."""
        migrated = dict(state_dict)
        changed = False
        if hasattr(self, "target_encoder") and any(key.startswith("momentum_model.") for key in state_dict):
            current_keys = set(self.state_dict())
            for key in list(migrated):
                if not key.startswith("momentum_model."):
                    continue
                relative = key[len("momentum_model.") :]
                target_key = "target_encoder." + relative
                if target_key in current_keys:
                    migrated[target_key] = migrated[key]
                del migrated[key]
            migrated.setdefault("_ema_update_count", self._ema_update_count.detach().clone())
            changed = True
        if changed:
            state_dict = migrated
        result = super().load_state_dict(state_dict, *args, **kwargs)
        if result is not None:
            return result

        # ``BaseModule.load_state_dict(..., strict=False)`` intentionally implements a permissive
        # transfer path and historically returned ``None`` after loading.  PyTorch's public
        # ``load_state_dict`` contract returns an ``_IncompatibleKeys`` report, and the frozen
        # Stage-2 parent loader consumes that report for provenance only.  Reconstruct it here
        # without touching any tensor, load decision, objective or optimizer state.
        current = self.state_dict()
        loadable = {
            key for key, value in state_dict.items() if key in current and tuple(value.shape) == tuple(current[key].shape)
        }
        return torch.nn.modules.module._IncompatibleKeys(
            missing_keys=sorted(set(current) - loadable),
            unexpected_keys=sorted(set(state_dict) - set(current)),
        )

    def _set_inactive_ssl_heads_trainability(self) -> None:
        """Freeze inactive auxiliary heads so DDP does not track unused trainable parameters."""
        if not bool(getattr(self._online_model, "modality_conditioning", False)):
            # These modules are bypassed by every AMAES/Stage 2 encoder forward in this
            # mode. Keeping them trainable makes strict DDP fail after the first backward.
            for module_name in ("modality_embedding", "encoder_films", "decoder_films"):
                self._set_module_trainability(module_name, False)
        self._set_module_trainability(
            "head_demo",
            self.enable_demo_loss,
        )
        self._set_module_trainability("head_patho", self.enable_patho_loss)
        self._set_module_trainability("scanner_embedding_head", self.enable_scanner_pos_loss)
        self._set_module_trainability("scanner_classifiers", self.enable_scanner_pos_loss)
        self._set_module_trainability("inv_embedding_head", self.enable_inv_adv_loss)
        self._set_module_trainability("inv_discriminators", self.enable_inv_adv_loss)
        self._set_module_trainability(
            "stage1_modality_head",
            self.enable_modality_loss or self.enable_stage1_orthogonality_loss,
        )
        self._set_module_trainability("head_stage1_anatomy", self._stage1_head_enabled)
        self._set_module_trainability("stage2_factorized_head", self.enable_stage2_loss)
        stage2_reconstruction_active = (
            self.enable_stage2_loss
            and self.stage2_decoder_enabled
            and (self.loss_weight_stage2_self_reconstruction > 0.0 or self.loss_weight_stage2_cross_reconstruction > 0.0)
        )
        self._set_module_trainability("stage2_anatomy_projector", stage2_reconstruction_active)
        self._set_module_trainability("stage2_decoder", stage2_reconstruction_active)

    def _validate_stage2_configuration(self) -> None:
        if not self.enable_stage2_loss:
            return
        if not self.stage2_registered_only:
            raise ValueError(
                "Stage 2 currently requires registered_only=true; unregistered scans must not contribute to "
                "factorization or reconstruction losses."
            )
        weights = {
            "weight_anatomy": self.loss_weight_stage2_anatomy,
            "weight_modality": self.loss_weight_stage2_modality,
            "weight_sigreg": self.loss_weight_stage2_sigreg,
            "weight_adv_mod_on_anat": self.loss_weight_stage2_adv_mod_on_anat,
            "weight_xcov": self.loss_weight_stage2_xcov,
            "weight_self_reconstruction": self.loss_weight_stage2_self_reconstruction,
            "weight_cross_reconstruction": self.loss_weight_stage2_cross_reconstruction,
        }
        nonfinite = [name for name, value in weights.items() if not math.isfinite(float(value))]
        if nonfinite:
            raise ValueError(f"Stage 2 loss weights must be finite; got {nonfinite}.")
        negative = [name for name, value in weights.items() if float(value) < 0.0]
        if negative:
            raise ValueError(f"Stage 2 loss weights must be non-negative; got {negative}.")
        if not math.isfinite(self.stage2_anat_temperature) or self.stage2_anat_temperature <= 0.0:
            raise ValueError("Stage 2 anatomy temperature must be > 0.")
        if self.stage2_sigreg_num_slices <= 0:
            raise ValueError("Stage 2 SIGReg num_slices must be > 0.")
        if not math.isfinite(self.stage2_grl_lambda) or self.stage2_grl_lambda < 0.0:
            raise ValueError("Stage 2 GRL lambda must be >= 0.")
        if self.stage2_reconstruction_loss not in {"smooth_l1", "mse"}:
            raise ValueError(
                f"Unsupported Stage 2 reconstruction loss {self.stage2_reconstruction_loss!r}; expected 'smooth_l1' or 'mse'."
            )
        if self.stage2_reconstruction_loss == "smooth_l1" and (
            not math.isfinite(self.stage2_smooth_l1_beta) or self.stage2_smooth_l1_beta <= 0.0
        ):
            raise ValueError("Stage 2 smooth_l1_beta must be > 0.")
        if self.loss_weight_stage2_cross_reconstruction > 0.0:
            if self.stage2_max_cross_pairs_per_anchor <= 0:
                raise ValueError("Stage 2 cross reconstruction requires max_cross_pairs_per_anchor > 0.")
            if self.stage2_max_cross_pairs_per_batch <= 0:
                raise ValueError("Stage 2 cross reconstruction requires max_cross_pairs_per_batch > 0.")

    def _set_module_trainability(self, module_name: str, trainable: bool) -> None:
        module = getattr(self._online_model, module_name, None)
        if module is None:
            return
        for parameter in module.parameters():
            parameter.requires_grad = bool(trainable)
        if module_name == "stage2_decoder" and trainable and getattr(module, "film_levels", "all") == "none":
            for film_name in ("bottleneck_film", "up_films"):
                film_module = getattr(module, film_name, None)
                if film_module is not None:
                    for parameter in film_module.parameters():
                        parameter.requires_grad = False

    def _build_stage1_modality_adversary(self, hidden_dim: int) -> nn.Module:
        z_dim = self._stage1_anatomy_output_dim()
        num_modalities = int(getattr(self._online_model, "num_modalities", 0))
        if num_modalities <= 0:
            raise ValueError("Stage-1 modality adversary requires model.num_modalities > 0.")
        return nn.Sequential(
            nn.Linear(z_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), num_modalities),
        )

    def _stage1_anatomy_output_dim(self) -> int:
        head = getattr(self._online_model, "head_stage1_anatomy", None)
        if head is None:
            raise ValueError("Stage-1 modality adversary requires model.head_stage1_anatomy.")
        for module in reversed(list(head.modules())):
            if isinstance(module, nn.Linear):
                return int(module.out_features)
        raise ValueError("Could not infer Stage-1 anatomy head output dimension from model.head_stage1_anatomy.")

    def _current_ema_momentum(self) -> float:
        """Fixed momentum for the contrastive objectives."""
        return self.moco_momentum

    def _ema_source_and_target(self) -> tuple[nn.Module, nn.Module]:
        target = self.momentum_model
        return self._online_model, target

    @torch.no_grad()
    def _update_momentum_encoder(self):
        m = self._current_ema_momentum()
        self._last_ema_momentum = float(m)
        source, target = self._ema_source_and_target()
        source_params = dict(source.named_parameters())
        target_params = dict(target.named_parameters())
        missing = sorted(set(target_params) - set(source_params))
        shape_mismatches = sorted(
            name for name in target_params if name in source_params and target_params[name].shape != source_params[name].shape
        )
        dtype_device_mismatches = sorted(
            name
            for name in target_params
            if name in source_params
            and (
                target_params[name].dtype != source_params[name].dtype
                or target_params[name].device != source_params[name].device
            )
        )
        if missing or shape_mismatches or dtype_device_mismatches:
            raise RuntimeError(
                "EMA source/target contract mismatch: "
                f"missing_online={missing}, shape_mismatches={shape_mismatches}, "
                f"dtype_device_mismatches={dtype_device_mismatches}."
            )
        ema_step = int(self._ema_update_count.item()) if hasattr(self, "_ema_update_count") else int(self.global_step)
        measure_distance = ema_step % max(1, self.log_every_n_steps) == 0
        squared_distance = None
        distance_elements = 0
        for name, param_k in target_params.items():
            if measure_distance:
                delta = (param_k.float() - source_params[name].float()).square().sum()
                squared_distance = delta if squared_distance is None else squared_distance + delta
                distance_elements += param_k.numel()
            param_k.mul_(m).add_(source_params[name], alpha=1.0 - m)

        source_buffers = dict(source.named_buffers())
        for name, buffer_k in target.named_buffers():
            buffer_q = source_buffers.get(name)
            if (
                buffer_q is None
                or buffer_q.shape != buffer_k.shape
                or buffer_q.dtype != buffer_k.dtype
                or buffer_q.device != buffer_k.device
            ):
                raise RuntimeError(f"EMA buffer contract mismatch for {name!r}.")
            if torch.is_floating_point(buffer_k) or torch.is_complex(buffer_k):
                buffer_k.mul_(m).add_(buffer_q, alpha=1.0 - m)
            else:
                buffer_k.copy_(buffer_q)
        if squared_distance is not None:
            self._last_ema_parameter_rms = (squared_distance / max(1, distance_elements)).sqrt().detach()
        if hasattr(self, "_ema_update_count"):
            self._ema_update_count.add_(1)

    def _get_ssl_model(self):
        return self._online_model

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        """Update EMA after the online optimizer step, once per effective batch."""
        trainer = getattr(self, "_trainer", None)
        precision_plugin = getattr(trainer, "precision_plugin", None)
        scaler = getattr(precision_plugin, "scaler", None)
        scale_before = float(scaler.get_scale()) if scaler is not None and hasattr(scaler, "get_scale") else None
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        scale_after = float(scaler.get_scale()) if scaler is not None and hasattr(scaler, "get_scale") else None
        # GradScaler lowers its scale when ``scaler.step`` skipped the optimizer
        # because of an Inf/NaN. In that case the online weights did not advance,
        # so neither the EMA target nor its schedule counter may advance.
        optimizer_was_skipped = scale_before is not None and scale_after is not None and scale_after < scale_before
        if self._uses_momentum_encoder and not optimizer_was_skipped:
            self._update_momentum_encoder()

    def training_step(self, batch, batch_idx):
        batch = self._apply_train_transforms(batch)
        x, y = batch["image"], batch["label"]

        if torch.isnan(y).any():
            logging.warning(f"Skipping batch {batch_idx} due to NaNs in input.")
            return None

        mask = batch.get("mask", None)
        pred, encoder_features = self._forward_with_features(self.model, x, modality_id=batch.get("modality_id"))

        loss_mse = self._rec_loss(pred, y, mask if self.rec_loss_masked_only else None)
        weighted_mse = self._weighted_loss(loss_mse, self.loss_weight_mse, 1.0, self.enable_mse_loss)
        component_values = {
            "mse": (loss_mse, self.loss_weight_mse, 1.0, weighted_mse, self.enable_mse_loss),
        }
        components = self._complete_loss_components(component_values, loss_mse)
        loss = sum((component[3] for component in components.values()), start=loss_mse.new_tensor(0.0))
        assert not torch.isnan(loss), "Reconstruction loss is NaN."
        self._maybe_log_loss_gradient_norms(components)
        self._check_component_starvation(components)

        # Logging
        with torch.no_grad():
            metrics = {}
            transforms_applied = batch.get("transforms_applied", None)

            core_metrics = {
                "loss": self._loss_metrics(
                    total=loss,
                    components=components,
                    excluded=("demographic",),
                ),
                "features": feat_metrics.compute_train(encoder_features),
                "stability": stability_metrics.compute_nan_inf_metrics(loss=loss, pred=pred, activations=encoder_features),
            }
            self.log_dict(
                self._format_metrics("train", core_metrics),
                on_step=True,
                on_epoch=False,
                sync_dist=False,
                batch_size=self.trainer.datamodule.batch_size,
            )

            if self.global_step % self.log_every_n_steps == 0:
                reconstruction_metrics = loss_metrics.compute_train(loss_mse, pred, y, mask, self._rec_loss, masked_input=x)
                reconstruction_metrics |= self._foreground_reconstruction_metrics(pred, y, mask)
                metrics = {
                    "reconstruction_loss": reconstruction_metrics,
                    "masking": masking.compute(mask, y),
                    "performance": perf_metrics.compute(transforms_applied, x.shape[0]),
                }
                self.log_dict(
                    self._format_metrics("train", metrics),
                    on_step=True,
                    on_epoch=False,
                    sync_dist=False,
                    batch_size=self.trainer.datamodule.batch_size,
                )

            if batch_idx == 0 and self.current_epoch % 10 == 0 and self.trainer.is_global_zero:
                images, error_images = visualization.create_visualizations(x, y, pred, mask, self.current_epoch)
                log_images_to_logger(
                    self.trainer.loggers,
                    images,
                    step=self.global_step,
                    prefix="images/train",
                )
                log_images_to_logger(
                    self.trainer.loggers,
                    error_images,
                    step=self.global_step,
                    prefix="images/train_error",
                )

        return loss

    def validation_step(self, batch, batch_idx):
        batch = self._apply_deterministic_val_transforms(batch)
        x, y = batch["image"], batch["label"]

        if torch.isnan(y).any():
            logging.warning(f"Skipping batch {batch_idx} due to NaNs in input.")
            return None

        mask = batch.get("mask", None)

        pred, encoder_features = self._forward_with_features(self.model, x, modality_id=batch.get("modality_id"))
        loss_mse = self._rec_loss(pred, y, mask if self.rec_loss_masked_only else None)
        weighted_mse = self._weighted_loss(loss_mse, self.loss_weight_mse, 1.0, self.enable_mse_loss)
        component_values = {
            "mse": (loss_mse, self.loss_weight_mse, 1.0, weighted_mse, self.enable_mse_loss),
        }
        components = self._complete_loss_components(component_values, loss_mse)
        loss = sum((component[3] for component in components.values()), start=loss_mse.new_tensor(0.0))
        assert not torch.isnan(loss), "Reconstruction loss is NaN."

        # Logging
        reconstruction_loss_metrics = loss_metrics.compute_val(
            loss_mse,
            pred,
            y,
            mask,
            self._rec_loss,
            masked_input=x,
            data_range=self.reconstruction_data_range,
            log_raw_full=self.log_raw_reconstruction_metrics,
        )
        reconstruction_loss_metrics |= self._foreground_reconstruction_metrics(pred, y, mask)
        metrics = {
            "loss": self._loss_metrics(total=loss, components=components, excluded=("demographic",)),
            "reconstruction_loss": reconstruction_loss_metrics,
            "features": feat_metrics.compute_val(encoder_features, self.model),
            "distribution": dist_metrics.compute(x, pred, y, encoder_features),
            "reconstruction": recon_metrics.compute(
                pred,
                y,
                mask,
                masked_input=x,
                data_range=self.reconstruction_data_range,
                log_raw_full=self.log_raw_reconstruction_metrics,
                wavelet_family=self.wavelet_family,
                wavelet_levels=self.wavelet_levels,
                wavelet_level_weights=self.wavelet_level_weights,
                wavelet_support_mode=self.wavelet_support_mode,
                wavelet_include_lowpass=self.wavelet_include_lowpass,
                wavelet_eps=self.wavelet_eps,
            ),
        }
        self._record_validation_metadata(self._contrastive_metadata(batch), x.shape[0], x.device)
        self.log_dict(
            self._format_metrics(self._validation_stage(), metrics),
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )

        # Rank zero only logging for images
        if self.trainer.is_global_zero and self.current_epoch % self.log_images_every_n_epoch == 0 and batch_idx == 0:
            images, error_images = visualization.create_visualizations(x, y, pred, mask, self.current_epoch)
            log_images_to_logger(self.trainer.loggers, images, step=self.global_step, prefix="images/val")
            log_images_to_logger(
                self.trainer.loggers,
                error_images,
                step=self.global_step,
                prefix="images/val_error_map",
            )

    def _rec_loss(self, pred, y, mask=None):
        if self.mse_foreground_mode == "weighted_replace":
            weights = obj_reconstruction.compute_foreground_voxel_weights(
                y,
                self.mse_background_weight,
                threshold=self.mse_foreground_threshold,
                dynamic_quantiles=self.mse_foreground_dynamic_quantiles,
                dynamic_scale=self.mse_foreground_dynamic_scale,
            )
            return obj_reconstruction.weighted_rec_loss(
                pred,
                y,
                mask,
                weights,
                sample_normalize=self.mse_foreground_sample_normalize,
            )
        loss = obj_reconstruction.rec_loss(self._rec_loss_fn, pred, y, mask)
        if self.mse_foreground_mode not in {"bonus_voxel", "bonus_patch"}:
            return loss
        bonus, _, _ = self._foreground_bonus_loss(pred, y, mask)
        schedule = self.get_dynamic_weight(self.mse_foreground_bonus_start_step, self.mse_foreground_bonus_warmup_steps)
        return loss + float(self.mse_foreground_bonus_weight) * float(schedule) * bonus

    def _foreground_bonus_loss(self, pred, y, mask=None):
        zero = pred.sum() * 0.0
        if self.mse_foreground_mode == "bonus_voxel":
            bonus, active_fraction = obj_reconstruction.foreground_voxel_bonus_mse(
                pred,
                y,
                mask,
                threshold=self.mse_foreground_threshold,
                dynamic_quantiles=self.mse_foreground_dynamic_quantiles,
                dynamic_scale=self.mse_foreground_dynamic_scale,
            )
            return bonus, active_fraction, zero
        if self.mse_foreground_mode == "bonus_patch":
            return obj_reconstruction.foreground_patch_bonus_mse(
                pred,
                y,
                mask,
                patch_size=self.mse_foreground_bonus_patch_size,
                min_foreground_fraction=self.mse_foreground_min_fraction,
                threshold=self.mse_foreground_threshold,
                dynamic_quantiles=self.mse_foreground_dynamic_quantiles,
                dynamic_scale=self.mse_foreground_dynamic_scale,
            )
        return zero, zero, zero

    def _foreground_reconstruction_metrics(self, pred, y, mask=None):
        metrics = obj_reconstruction.foreground_reconstruction_mse_metrics(
            pred,
            y,
            mask,
            threshold=self.mse_foreground_threshold,
            dynamic_quantiles=self.mse_foreground_dynamic_quantiles,
            dynamic_scale=self.mse_foreground_dynamic_scale,
        )
        bonus, active_fraction, patch_fg_fraction = self._foreground_bonus_loss(pred, y, mask)
        schedule = self.get_dynamic_weight(self.mse_foreground_bonus_start_step, self.mse_foreground_bonus_warmup_steps)
        weighted_bonus = float(self.mse_foreground_bonus_weight) * float(schedule) * bonus
        metrics.update(
            {
                "loss_hidden_fg_bonus": bonus.detach(),
                "loss_hidden_fg_bonus_weighted": weighted_bonus.detach(),
                "fg_bonus_active_fraction": active_fraction.detach(),
                "fg_bonus_config_weight": float(self.mse_foreground_bonus_weight),
                "fg_bonus_schedule_weight": float(schedule),
                "foreground_mode/none": float(self.mse_foreground_mode == "none"),
                "foreground_mode/weighted_replace": float(self.mse_foreground_mode == "weighted_replace"),
                "foreground_mode/bonus_voxel": float(self.mse_foreground_mode == "bonus_voxel"),
                "foreground_mode/bonus_patch": float(self.mse_foreground_mode == "bonus_patch"),
                "loss_hidden_patch_fg": bonus.detach() if self.mse_foreground_mode == "bonus_patch" else pred.sum() * 0.0,
                "patch_fg_fraction_hidden": patch_fg_fraction.detach(),
            }
        )
        return metrics

    @staticmethod
    def _weighted_loss(raw_loss: torch.Tensor, config_weight: float, schedule_weight: float, enabled: bool) -> torch.Tensor:
        return obj_reconstruction.weighted_loss(raw_loss, config_weight, schedule_weight, enabled)

    def _check_component_starvation(self, components: dict) -> None:
        """Fail loudly when an active objective never sees an eligible batch.

        Stacking several objectives onto one sampler can silently starve one of them: the loss is
        enabled and its schedule weight is positive, but every batch is ineligible, so the raw term
        is exactly zero forever and the component contributes nothing. That is indistinguishable
        from "disabled" in the logs, which is precisely the failure this campaign must not ship.

        A component is flagged only if it is enabled, its effective weight is strictly positive, and
        its raw loss was exactly zero for a whole window of consecutive optimizer steps.
        """
        window = int(self.component_starvation_window)
        if window <= 0:
            return
        for name, (raw, config_weight, schedule_weight, _weighted, enabled) in components.items():
            if not bool(enabled) or float(config_weight) * float(schedule_weight) <= 0.0:
                # Not active right now (e.g. before its start_step): reset rather than accumulate.
                self._component_zero_streak[name] = 0
                continue
            if float(raw.detach().abs().sum()) > 0.0:
                self._component_zero_streak[name] = 0
                continue
            streak = self._component_zero_streak.get(name, 0) + 1
            self._component_zero_streak[name] = streak
            if streak >= window:
                raise RuntimeError(
                    f"Loss component {name!r} is enabled with effective weight "
                    f"{float(config_weight) * float(schedule_weight):.6g} but its raw loss was exactly zero for "
                    f"{streak} consecutive optimizer steps. It is receiving no eligible batches, so it is "
                    "silently inactive. Fix the batch routing (sampler probabilities / eligibility) or "
                    "disable the component explicitly instead of shipping a recipe that does not run."
                )

    def _complete_loss_components(self, components: dict, reference: torch.Tensor) -> dict:
        """Keep the loss logging schema stable when an objective branch is inactive."""
        schedules = {
            "demographic": (
                self.loss_weight_demo,
                self.get_dynamic_weight(self.demo_start_step, self.demo_warmup_steps),
                self.enable_demo_loss,
            ),
            "pathology": (
                self.loss_weight_patho,
                self.get_dynamic_weight(self.patho_start_step, self.patho_warmup_steps),
                self.enable_patho_loss,
            ),
            "scanner_pos": (
                self.loss_weight_scanner_pos,
                self.get_dynamic_weight(self.scanner_pos_start_step, self.scanner_pos_warmup_steps),
                self.enable_scanner_pos_loss,
            ),
            "inv_adv": (
                self.loss_weight_inv_adv,
                self.get_dynamic_weight(self.inv_adv_start_step, self.inv_adv_warmup_steps),
                self.enable_inv_adv_loss,
            ),
            "stage1/modality": (
                self.loss_weight_modality,
                self.get_dynamic_weight(self.modality_start_step, self.modality_warmup_steps),
                self.enable_modality_loss,
            ),
            "stage1/anatomy": (
                self.loss_weight_stage1_anatomy,
                self.get_dynamic_weight(self.stage1_start_step, self.stage1_warmup_steps),
                self.enable_stage1_loss,
            ),
            "stage1/modality_adversary": (
                self.loss_weight_stage1_modality_adversary,
                self.get_dynamic_weight(
                    self.stage1_modality_adversary_start_step,
                    self.stage1_modality_adversary_warmup_steps,
                ),
                self.enable_stage1_modality_adversary_loss,
            ),
            "stage2/anatomy": (
                self.loss_weight_stage2_anatomy,
                self.get_dynamic_weight(self.stage2_start_step, self.stage2_warmup_steps),
                self.enable_stage2_loss,
            ),
            "stage2/modality": (
                self.loss_weight_stage2_modality,
                self.get_dynamic_weight(self.stage2_start_step, self.stage2_warmup_steps),
                self.enable_stage2_loss,
            ),
            "stage2/sigreg": (
                self.loss_weight_stage2_sigreg,
                self.get_dynamic_weight(self.stage2_start_step, self.stage2_warmup_steps),
                self.enable_stage2_loss and self.loss_weight_stage2_sigreg > 0.0,
            ),
            "stage2/adv_mod_on_anat": (
                self.loss_weight_stage2_adv_mod_on_anat,
                self.get_dynamic_weight(self.stage2_start_step, self.stage2_warmup_steps),
                self.enable_stage2_loss and self.loss_weight_stage2_adv_mod_on_anat > 0.0,
            ),
            "stage2/xcov": (
                self.loss_weight_stage2_xcov,
                self.get_dynamic_weight(self.stage2_start_step, self.stage2_warmup_steps),
                self.enable_stage2_loss and self.loss_weight_stage2_xcov > 0.0,
            ),
            "stage2/reconstruction/self": (
                self.loss_weight_stage2_self_reconstruction,
                self.get_dynamic_weight(self.stage2_start_step, self.stage2_warmup_steps),
                self.enable_stage2_loss and self.stage2_decoder_enabled and self.loss_weight_stage2_self_reconstruction > 0.0,
            ),
            "stage2/reconstruction/cross": (
                self.loss_weight_stage2_cross_reconstruction,
                self.get_dynamic_weight(self.stage2_start_step, self.stage2_warmup_steps)
                * self.get_dynamic_weight(
                    self.stage2_cross_reconstruction_start_step,
                    self.stage2_cross_reconstruction_warmup_steps,
                ),
                self.enable_stage2_loss and self.stage2_decoder_enabled and self.loss_weight_stage2_cross_reconstruction > 0.0,
            ),
            "stage2/self_reconstruction": (
                self.loss_weight_stage2_self_reconstruction,
                self.get_dynamic_weight(self.stage2_start_step, self.stage2_warmup_steps),
                False,
            ),
            "stage2/cross_reconstruction": (
                self.loss_weight_stage2_cross_reconstruction,
                self.get_dynamic_weight(self.stage2_start_step, self.stage2_warmup_steps),
                False,
            ),
        }
        completed = dict(components)
        zero = reference.sum() * 0.0
        for name, (weight, schedule, enabled) in schedules.items():
            if name not in completed:
                completed[name] = (zero, weight, schedule, zero, enabled)
        return completed

    @staticmethod
    def _loss_metrics(total: torch.Tensor, components: dict, excluded: tuple[str, ...] = ()) -> dict:
        return obj_reconstruction.loss_metrics(total, components, excluded)

    @staticmethod
    def _modality_diagnostics_for_sync_dist(diagnostics: dict[str, float]) -> dict[str, float]:
        """Remove conditional per-class keys before synchronized batch logging.

        Rank-local modality batches need not contain or predict the same classes.  Their
        ``recall/class_*`` and ``precision/class_*`` schemas can therefore differ, which makes
        Lightning issue different metric collectives on different ranks.  The fixed-size
        confusion matrix is reduced at epoch end and remains the authoritative per-class report.
        """
        rank_local_prefixes = ("support/class_", "recall/class_", "precision/class_")
        return {key: value for key, value in diagnostics.items() if not key.startswith(rank_local_prefixes)}

    def on_after_backward(self):
        grad_clip_val = self.trainer.gradient_clip_val if hasattr(self.trainer, "gradient_clip_val") else None
        metrics_grouped = {
            "stability": stability_metrics.compute_on_backward(self.model, grad_clip_val),
            "performance": perf_metrics.compute_on_backward(self.trainer),
        }
        self.log_dict(
            self._format_metrics("train", metrics_grouped),
            on_step=True,
            on_epoch=False,
            sync_dist=False,
            batch_size=self.trainer.datamodule.batch_size,
        )

    def _diagnostic_parameters(self) -> list:
        """The explicit parameter subset the diagnostics measure — the shared encoder.

        Gradient norms only compare across losses if they are measured on the *same* parameters, and
        the shared encoder is the thing every objective competes over. Falls back to all trainable
        parameters only when no encoder-named parameter exists.
        """
        parameters = [
            parameter for name, parameter in self.model.named_parameters() if "encoder" in name and parameter.requires_grad
        ]
        if not parameters:
            parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        return parameters

    def _maybe_log_loss_gradient_norms(self, components: dict):
        """Opt-in per-loss gradient-contribution diagnostics.

        Disabled unless ``logger.loss_gradient_norm_every_n_steps > 0``.  When enabled it must remain
        *observationally neutral* on training: it computes gradients with
        ``torch.autograd.grad(..., retain_graph=True)``, which populates no ``.grad`` buffer, takes no
        optimizer step, touches no AMP scaler and consumes no RNG.  The graph is released as soon as
        the block finishes because nothing here keeps a reference to it.

        These numbers are for calibration and interpretation.  They are deliberately *not* wired to
        any automatic promotion rule: a large gradient norm is not evidence that a component helps.
        """
        if self.loss_gradient_norm_every_n_steps <= 0:
            return
        if int(self.global_step) % self.loss_gradient_norm_every_n_steps != 0:
            return
        parameters = self._diagnostic_parameters()
        metrics = {}
        gradients_by_loss = {}
        norms_by_loss = {}
        for name, entry in components.items():
            _, config_weight, schedule_weight, weighted, enabled = entry
            effective = float(config_weight) * float(schedule_weight)
            metrics[f"train/grad_effective_weight/{name}"] = effective
            if not enabled or not torch.is_tensor(weighted) or not weighted.requires_grad:
                # An inactive component, or one with no valid anchors this step, is reported as
                # explicitly unavailable rather than as a zero contribution — those mean different
                # things and conflating them is how a dead objective looks merely small.
                metrics[f"train/grad_available/{name}"] = 0.0
                continue
            metrics[f"train/grad_available/{name}"] = 1.0
            gradients = torch.autograd.grad(weighted, parameters, retain_graph=True, allow_unused=True)
            finite_elements, total_elements = 0, 0
            squared_norm = weighted.new_tensor(0.0)
            for gradient in gradients:
                if gradient is None:
                    continue
                detached = gradient.detach().float()
                finite_mask = torch.isfinite(detached)
                finite_elements += int(finite_mask.sum().item())
                total_elements += detached.numel()
                squared_norm = squared_norm + detached.norm() ** 2
            norm = squared_norm.sqrt()
            metrics[f"train/grad_norm/{name}"] = norm
            metrics[f"train/grad_finite_fraction/{name}"] = finite_elements / total_elements if total_elements else 0.0
            # Weighted contribution is the norm as it enters the total loss; the unweighted one
            # divides the effective weight back out, so a component with a tiny weight and a huge
            # raw gradient is distinguishable from one that is genuinely quiet.
            metrics[f"train/grad_contribution_weighted/{name}"] = norm
            if effective > 0.0:
                metrics[f"train/grad_contribution_unweighted/{name}"] = norm / effective
            gradients_by_loss[name] = gradients
            norms_by_loss[name] = norm
        metrics.update(self._loss_gradient_alignment_metrics(gradients_by_loss, norms_by_loss))
        if metrics:
            self.log_dict(metrics, on_step=True, on_epoch=False, sync_dist=False)

    @staticmethod
    def _loss_gradient_alignment_metrics(gradients_by_loss: dict, norms_by_loss: dict) -> dict:
        """Pairwise gradient cosine similarity over every active component.

        Previously this compared each component against ``mse`` only, which cannot show two
        auxiliary objectives fighting each other.  Every unordered pair is now reported, with the
        ``_to_mse`` names preserved so existing dashboards keep working.
        """
        metrics = {}
        names = sorted(gradients_by_loss)
        for position, name in enumerate(names):
            for other in names[position + 1 :]:
                left, right = gradients_by_loss[name], gradients_by_loss[other]
                left_norm, right_norm = norms_by_loss[name], norms_by_loss[other]
                eps = torch.finfo(left_norm.dtype).eps
                dot = sum(
                    (
                        (a.float() * b.float()).sum()
                        for a, b in zip(left, right, strict=True)
                        if a is not None and b is not None
                    ),
                    start=left_norm.new_tensor(0.0),
                )
                denominator = left_norm * right_norm
                cosine = torch.where(
                    denominator > eps,
                    (dot / denominator.clamp_min(eps)).clamp(-1.0, 1.0),
                    denominator.new_zeros(()),
                )
                # "mse" sorts before most names, so the historical `<name>_to_mse` keys are
                # reproduced exactly; every other pair gets a symmetric `<a>_to_<b>` key.
                if name == "mse":
                    metrics[f"train/grad_cosine/{other}_to_mse"] = cosine
                    metrics[f"train/grad_norm_ratio/{other}_to_mse"] = right_norm / left_norm.clamp_min(eps)
                else:
                    metrics[f"train/grad_cosine/{name}_to_{other}"] = cosine
                    metrics[f"train/grad_norm_ratio/{name}_to_{other}"] = left_norm / right_norm.clamp_min(eps)
        return metrics

    def _prepare_contrastive_batch(self, batch, is_training: bool):
        return ssl_views.build_contrastive_views(
            batch,
            is_training=is_training,
            train_transforms=self.train_transforms,
            unmasked_transforms=self.unmasked_transforms,
            momentum_transforms=self.momentum_transforms,
            val_transforms=self.val_transforms,
            demo_cpu_transforms=self.demo_cpu_transforms,
            validation_mask_seed=self.validation_mask_seed,
        )

    def _format_metrics(self, stage, metric_groups):
        """
        Format metrics with hierarchical naming: stage/module/metric on wandb and stage_module/metric on MLflow.
        """
        # MLflow only supports one forward slash in the metric name
        metric_separator = "_" if self.mlflow_logging else "/"
        metrics = {}
        for module_name, metric_dict in metric_groups.items():
            for key, value in metric_dict.items():
                metrics[f"{stage}{metric_separator}{module_name}/{key}"] = value
        return metrics

    def _validation_stage(self) -> str:
        return "exhaustive" if self.exhaustive_evaluation else "val"

    def _apply_deterministic_val_transforms(self, batch):
        return ssl_views.apply_deterministic_val_transforms(batch, self.val_transforms, self.validation_mask_seed)

    def on_validation_epoch_start(self):
        self._val_embedding_batches = []
        self._projection_embedding_batches = []
        self._probe_reference_batches = []
        self._modality_probe_reference_batches = []
        self._demographic_probe_reference_batches = []
        self._val_metadata_batches = []
        self._val_stage1_batches = []
        num_modalities = int(getattr(self._online_model, "num_modalities", 0))
        self._val_modality_confusion = torch.zeros((num_modalities, num_modalities), device=self.device)
        self._val_demographic_cross_entropy_sum = torch.zeros((), device=self.device)
        self._val_demographic_weighted_cross_entropy_sum = torch.zeros((), device=self.device)
        self._val_demographic_target_entropy_sum = torch.zeros((), device=self.device)
        self._val_demographic_excess_cross_entropy_sum = torch.zeros((), device=self.device)
        self._val_demographic_candidate_count_sum = torch.zeros((), device=self.device)
        self._val_demographic_eligible_count = torch.zeros((), device=self.device)
        self._val_demographic_by_modality_totals = {
            token.name: {
                metric: torch.zeros((), device=self.device)
                for metric in (
                    "cross_entropy_sum",
                    "weighted_cross_entropy_sum",
                    "target_entropy_sum",
                    "excess_cross_entropy_sum",
                    "candidate_count_sum",
                    "logit_std_sum",
                    "eligible_count",
                    "active_count",
                )
            }
            for token in self.demographic_tokens
        }

    def on_train_start(self):
        if (
            not self.validation_embedding_monitor_enabled
            or self._initial_embedding_monitor_logged
            or int(getattr(self.trainer, "global_step", 0)) != 0
            or getattr(self.trainer, "sanity_checking", False)
        ):
            return

        was_training = self.training
        self.eval()
        try:
            self._run_probe_monitor_dataloaders()
            self._log_validation_embedding_monitor(direct_step=int(self.global_step))
            self._initial_embedding_monitor_logged = True
        finally:
            self.train(was_training)

    def on_validation_epoch_end(self):
        stage = self._validation_stage()
        if self.enable_demo_loss:
            demographic_totals = torch.stack(
                [
                    self._val_demographic_cross_entropy_sum,
                    self._val_demographic_weighted_cross_entropy_sum,
                    self._val_demographic_target_entropy_sum,
                    self._val_demographic_excess_cross_entropy_sum,
                    self._val_demographic_candidate_count_sum,
                    self._val_demographic_eligible_count,
                ]
            )
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(demographic_totals, op=torch.distributed.ReduceOp.SUM)
            if demographic_totals[5] > 0:
                self.log_dict(
                    {
                        f"{stage}/demographic/cross_entropy": demographic_totals[0] / demographic_totals[5],
                        f"{stage}/demographic/weighted_cross_entropy": demographic_totals[1] / demographic_totals[5],
                        f"{stage}/demographic/target_entropy": demographic_totals[2] / demographic_totals[5],
                        f"{stage}/demographic/excess_cross_entropy": demographic_totals[3] / demographic_totals[5],
                        f"{stage}/demographic/candidate_count": demographic_totals[4] / demographic_totals[5],
                        f"{stage}/demographic/eligible_count": demographic_totals[5],
                    },
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )
            for modality_name, totals in self._val_demographic_by_modality_totals.items():
                if torch.distributed.is_initialized():
                    for value in totals.values():
                        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
                eligible_count = totals["eligible_count"]
                if eligible_count <= 0:
                    continue
                prefix = f"{stage}/demographic/by_modality/{modality_name}"
                self.log_dict(
                    {
                        f"{prefix}/cross_entropy": totals["cross_entropy_sum"] / eligible_count,
                        f"{prefix}/weighted_cross_entropy": totals["weighted_cross_entropy_sum"] / eligible_count,
                        f"{prefix}/target/entropy": totals["target_entropy_sum"] / eligible_count,
                        f"{prefix}/objective/excess_cross_entropy": (totals["excess_cross_entropy_sum"] / eligible_count),
                        f"{prefix}/target/candidate_count": totals["candidate_count_sum"] / eligible_count,
                        f"{prefix}/prediction/logit_std": totals["logit_std_sum"] / eligible_count,
                        f"{prefix}/eligible_count": eligible_count,
                        f"{prefix}/active": totals["active_count"] / eligible_count,
                    },
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )
        if self._val_metadata_batches:
            combined = {
                key: torch.cat([batch[key] for batch in self._val_metadata_batches if key in batch], dim=0)
                for key in ("age", "sex", "pathology", "fine_pathology", "scanner_id", "modality_id", "dataset_id")
            }
            # ``contrastive_metadata_metrics`` deliberately emits class-specific and
            # top-k keys. Computing it independently on each DDP rank can therefore
            # produce a different metric schema per rank, which makes Lightning's
            # epoch-end metric collectives diverge. Gather the small metadata vectors
            # first so every rank computes the same full-cohort schema, then log the
            # already-global values without another distributed reduction.
            combined = {
                key: self._gather_monitor_tensor(value.to(self.device)).detach().cpu() for key, value in combined.items()
            }
            self.log_dict(
                self._format_metrics(stage, {"metadata_epoch": self._contrastive_metadata_metrics(combined)}),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
        if self._should_run_probe_monitor():
            self._run_probe_monitor_dataloaders()
        if self._val_stage1_batches:
            combined_stage1 = {
                key: torch.cat([batch[key] for batch in self._val_stage1_batches], dim=0)
                for key in ("features", "subject_id", "subject_session_id", "modality_id", "registered_subset")
            }
            stage1_metrics = dist_metrics.compute_cross_modal_retrieval(
                combined_stage1["features"],
                combined_stage1["subject_session_id"],
                combined_stage1["modality_id"],
                combined_stage1["registered_subset"],
                subject_ids=combined_stage1["subject_id"],
            )
            stage1_metrics_explicit = {f"z_anatomy/{key}": value for key, value in stage1_metrics.items()}
            stage1_cross_modal_metrics = dist_metrics.compute_cross_modal_retrieval(
                combined_stage1["features"],
                combined_stage1["subject_session_id"],
                combined_stage1["modality_id"],
                combined_stage1["registered_subset"],
                cross_modal_candidates_only=True,
                subject_ids=combined_stage1["subject_id"],
            )
            stage1_metrics_explicit.update(
                {f"z_anatomy_cross_modal_only/{key}": value for key, value in stage1_cross_modal_metrics.items()}
            )
            if all("h_features" in batch for batch in self._val_stage1_batches):
                h_features = torch.cat([batch["h_features"] for batch in self._val_stage1_batches], dim=0)
                h_metrics = dist_metrics.compute_cross_modal_retrieval(
                    h_features,
                    combined_stage1["subject_session_id"],
                    combined_stage1["modality_id"],
                    combined_stage1["registered_subset"],
                    subject_ids=combined_stage1["subject_id"],
                )
                stage1_metrics_explicit.update({f"h/{key}": value for key, value in h_metrics.items()})
                h_cross_modal_metrics = dist_metrics.compute_cross_modal_retrieval(
                    h_features,
                    combined_stage1["subject_session_id"],
                    combined_stage1["modality_id"],
                    combined_stage1["registered_subset"],
                    cross_modal_candidates_only=True,
                    subject_ids=combined_stage1["subject_id"],
                )
                stage1_metrics_explicit.update(
                    {f"h_cross_modal_only/{key}": value for key, value in h_cross_modal_metrics.items()}
                )
            z_mod_distribution = None
            if all("z_mod_features" in batch for batch in self._val_stage1_batches):
                z_mod_features = torch.cat([batch["z_mod_features"] for batch in self._val_stage1_batches], dim=0)
                z_mod_metrics = dist_metrics.compute_cross_modal_retrieval(
                    z_mod_features,
                    combined_stage1["subject_session_id"],
                    combined_stage1["modality_id"],
                    combined_stage1["registered_subset"],
                    subject_ids=combined_stage1["subject_id"],
                )
                stage1_metrics_explicit.update({f"z_mod/{key}": value for key, value in z_mod_metrics.items()})
                z_mod_cross_modal_metrics = dist_metrics.compute_cross_modal_retrieval(
                    z_mod_features,
                    combined_stage1["subject_session_id"],
                    combined_stage1["modality_id"],
                    combined_stage1["registered_subset"],
                    cross_modal_candidates_only=True,
                    subject_ids=combined_stage1["subject_id"],
                )
                stage1_metrics_explicit.update(
                    {f"z_mod_cross_modal_only/{key}": value for key, value in z_mod_cross_modal_metrics.items()}
                )
                z_mod_distribution = dist_metrics.compute_alignment_uniformity(z_mod_features)
                z_mod_distribution.update(feat_metrics.compute_collapse_score(z_mod_features))
                z_mod_distribution.update(feat_metrics.compute_participation_ratio(z_mod_features))
            stage1_distribution = dist_metrics.compute_alignment_uniformity(combined_stage1["features"])
            stage1_distribution.update(feat_metrics.compute_collapse_score(combined_stage1["features"]))
            stage1_distribution.update(feat_metrics.compute_participation_ratio(combined_stage1["features"]))
            logged_metrics = {
                "stage1_retrieval": stage1_metrics | stage1_metrics_explicit,
                "stage1_z_anatomy_distribution": stage1_distribution,
            }
            if z_mod_distribution is not None:
                logged_metrics["stage1_z_mod_distribution"] = z_mod_distribution
            self.log_dict(
                self._format_metrics(stage, logged_metrics),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
                rank_zero_only=True,
            )
        if self._should_run_probe_monitor():
            self._log_validation_embedding_monitor()

    @staticmethod
    def _build_demographic_tokens(tokens, modalities) -> tuple["DemographicToken", ...]:
        """Build the ordered demographic tokens from explicit descriptors or legacy ids.

        ``tokens`` (preferred) are ``{name, modality_id, bval_min, bval_max}`` descriptors
        resolved upstream (supports DWI b-value subtypes). Falls back to legacy
        ``demographic_modalities`` (ints/names of structural modalities) for backward
        compatibility with existing callers/tests.
        """
        if tokens:
            return tuple(
                DemographicToken(
                    name=str(token["name"]),
                    modality_id=int(token["modality_id"]),
                    bval_min=None if token.get("bval_min") is None else float(token["bval_min"]),
                    bval_max=None if token.get("bval_max") is None else float(token["bval_max"]),
                )
                for token in tokens
            )
        return tuple(
            DemographicToken(PretrainDataset.modality_name(modality_id), int(modality_id), None, None)
            for modality_id in PretrainDataset.normalize_modality_ids(modalities)
        )

    def _forward_with_features(self, model, x, modality_id=None):
        if (model is self.model or model is self._online_model) and self._compiled_forward_with_features is not None:
            return self._compiled_forward_with_features(x, modality_id=modality_id)
        try:
            return model.forward_with_features(x, modality_id=modality_id)
        except TypeError:
            return model.forward_with_features(x)

    def _pathology_eligible_mask(self, metadata: dict[str, torch.Tensor]) -> torch.Tensor:
        eligible = torch.ones_like(metadata["pathology"], dtype=torch.bool)
        if self.pathology_unknown_ignore:
            eligible &= metadata["pathology"] != -1
        if self.pathology_modalities:
            modality_match = torch.zeros_like(eligible, dtype=torch.bool)
            for modality_id in self.pathology_modalities:
                modality_match |= metadata["modality_id"] == int(modality_id)
            eligible &= modality_match
        if self.pathology_eligible_classes:
            class_match = torch.zeros_like(eligible, dtype=torch.bool)
            for class_id in self.pathology_eligible_classes:
                class_match |= metadata["pathology"] == int(class_id)
            eligible &= class_match
        for class_id in self.pathology_exclude_classes:
            eligible &= metadata["pathology"] != int(class_id)
        return eligible

    @staticmethod
    def _subject_hash_tensor(subject_session_key, device: torch.device) -> torch.Tensor:
        return ssl_metadata.subject_hash_tensor(subject_session_key, device)

    @staticmethod
    def _stable_int_hash(value) -> int:
        return ssl_metadata.stable_int_hash(value)

    def _gather_contrastive_metadata(self, batch, device, dtype):
        batch_size = self._metadata_batch_size(batch)
        age_local = self._to_float_tensor(batch.get("age"), device, dtype)
        if age_local is None:
            age_local = torch.full((batch_size,), float("nan"), dtype=dtype, device=device)
        sex_local = self._metadata_long_or_default(batch.get("sex"), batch_size, device, -1)
        patho_local = self._metadata_long_or_default(batch.get("pathology"), batch_size, device, -1)
        fine_patho_local = self._metadata_long_or_default(batch.get("fine_pathology"), batch_size, device, -1)
        modality_local = self._metadata_long_or_default(batch.get("modality_id"), batch_size, device, -1)
        scanner_local = self._metadata_long_or_default(batch.get("scanner_id"), batch_size, device, -1)
        dataset_local = self._metadata_long_or_default(batch.get("dataset_id"), batch_size, device, -1)
        # Diffusion b-value (NaN for non-DWI); used only by the demographic DWI subtype mask.
        dwi_bval_local = self._to_float_tensor(batch.get("dwi_bval"), device, dtype)
        if dwi_bval_local is None:
            dwi_bval_local = torch.full((batch_size,), float("nan"), dtype=dtype, device=device)

        local_metadata = {
            "age": age_local,
            "sex": sex_local,
            "pathology": patho_local,
            "fine_pathology": fine_patho_local,
            "modality_id": modality_local,
            "scanner_id": scanner_local,
            "dataset_id": dataset_local,
            "dwi_bval": dwi_bval_local,
        }

        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return local_metadata, local_metadata

        global_metadata = {
            "age": self.all_gather(age_local).view(-1),
            "sex": self.all_gather(sex_local).view(-1),
            "pathology": self.all_gather(patho_local).view(-1),
            "fine_pathology": self.all_gather(fine_patho_local).view(-1),
            "modality_id": self.all_gather(modality_local).view(-1),
            "scanner_id": self.all_gather(scanner_local).view(-1),
            "dataset_id": self.all_gather(dataset_local).view(-1),
            "dwi_bval": self.all_gather(dwi_bval_local).view(-1),
        }

        return local_metadata, global_metadata

    @staticmethod
    def _metadata_batch_size(batch) -> int:
        for key in ("pathology", "age", "sex", "modality_id", "image"):
            value = batch.get(key)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                return int(value.shape[0])
            return len(value)
        return 0

    @staticmethod
    def _metadata_long_or_default(values, batch_size: int, device: torch.device, default: int) -> torch.Tensor:
        if values is None:
            return torch.full((batch_size,), int(default), dtype=torch.long, device=device)
        return torch.as_tensor(values, dtype=torch.long, device=device).view(-1)

    @staticmethod
    def _to_float_tensor(values, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return ssl_metadata.to_float_tensor(values, device, dtype)

    @staticmethod
    def _contrastive_metadata(batch):
        return ssl_metadata.contrastive_metadata(batch)

    @staticmethod
    def _contrastive_metadata_metrics(metadata):
        return ssl_metadata.contrastive_metadata_metrics(metadata)

    def curriculum_spec(self, component: str | None):
        """The curriculum override for ``component``, or ``None`` when no curriculum is configured.

        The registry is empty unless a run explicitly supplies ``losses.curriculum``, so every
        historical configuration keeps its own schedule untouched.
        """
        if not component:
            return None
        return self._curriculum_specs.get(component)

    def get_dynamic_weight(self, start_step: int, warmup_steps: int, *, component: str | None = None) -> float:
        """Schedule multiplier at the current optimizer step.

        ``component`` names the curriculum slot this call belongs to. When a Task-6 curriculum
        supplies a :class:`~asparagus.modules.lightning_modules.ssl.schedules.ScheduleSpec` for that
        slot, it replaces the historical ``(start_step, warmup_steps)`` cosine ramp; otherwise the
        historical ramp is returned unchanged, byte for byte.
        """
        spec = self.curriculum_spec(component)
        if spec is not None:
            return ssl_schedules.schedule_fraction(spec, self.global_step)
        return ssl_schedules.cosine_ramp(self.global_step, start_step, warmup_steps)

    def get_dynamic_falcon_weight(self) -> float:
        spec = self.curriculum_spec("mse/falcon")
        if spec is not None:
            return ssl_schedules.schedule_fraction(spec, self.global_step)
        return ssl_schedules.cosine_window(
            self.global_step,
            self.mse_falcon_start_step,
            self.mse_falcon_warmup_steps,
            self.mse_falcon_end_step,
            self.mse_falcon_decay_steps,
        )

    def get_dynamic_frequency_weight(self) -> float:
        if not self.enable_frequency_loss:
            return 0.0
        return ssl_schedules.cosine_ramp(self.global_step, self.frequency_start_step, self.frequency_warmup_steps)

    def get_dynamic_wavelet_weight(self) -> float:
        return ssl_schedules.cosine_window(
            self.global_step,
            self.wavelet_start_step,
            self.wavelet_warmup_steps,
            self.wavelet_end_step,
            self.wavelet_decay_steps,
        )

    def get_inv_grl_lambda(self) -> float:
        if not self.enable_inv_adv_loss:
            return 0.0
        if self.inv_grl_schedule == "linear" and self.inv_grl_warmup_steps > 0:
            progress = min(1.0, max(0.0, float(self.global_step) / float(self.inv_grl_warmup_steps)))
            return self.inv_grl_lambda * progress
        return self.inv_grl_lambda

    def predict_step(self, batch, batch_idx):
        x = batch["image"]
        embeddings = self.model.encoder(x)[-1]
        return embeddings
