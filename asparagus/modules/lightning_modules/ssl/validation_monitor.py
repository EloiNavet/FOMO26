"""Validation-time representation monitoring for the SSL trainer.

Extracted verbatim from ``SelfSupervisedModule`` (P3.9 decomposition). These methods
are diagnostic only (probe dataloaders, kNN/silhouette probes, embedding projections)
and stay bound to the LightningModule via mixin inheritance, so behaviour and the
public method surface are unchanged.
"""

import logging
import numpy as np
import os
import torch
from asparagus.functional.metrics import embedding_projection
from asparagus.functional.representations import build_h_global, embedding_health_metrics
from asparagus.modules.datasets.PretrainDataset import PretrainDataset
from typing import Optional


class SSLValidationMonitorMixin:
    """Mixin providing the probe/embedding validation-monitor methods."""

    def _should_run_probe_monitor(self) -> bool:
        if not self.validation_embedding_monitor_enabled or self.trainer.sanity_checking:
            return False
        return self.exhaustive_evaluation or (
            self.probe_every_n_epoch > 0 and (int(self.current_epoch) + 1) % self.probe_every_n_epoch == 0
        )

    @torch.no_grad()
    def _run_probe_monitor_dataloaders(self):
        data_module = self.trainer.datamodule
        if not hasattr(data_module, "probe_reference_dataloader") or not hasattr(data_module, "probe_query_dataloader"):
            raise ValueError("Embedding monitoring requires probe_reference_dataloader and probe_query_dataloader.")
        self._probe_reference_batches = []
        self._modality_probe_reference_batches = []
        self._demographic_probe_reference_batches = []
        self._val_embedding_batches = []
        self._projection_embedding_batches = []
        for loader, collection in (
            (data_module.probe_reference_dataloader(), self._probe_reference_batches),
            (data_module.probe_query_dataloader(), self._val_embedding_batches),
        ):
            for batch_idx, batch in enumerate(loader):
                batch = self.transfer_batch_to_device(batch, self.device, batch_idx)
                embeddings = self._probe_embeddings(batch)
                self._collect_validation_embeddings(
                    embeddings,
                    self._contrastive_metadata(batch),
                    batch,
                    collection=collection,
                )
        if hasattr(data_module, "modality_probe_reference_dataloader"):
            for batch_idx, batch in enumerate(data_module.modality_probe_reference_dataloader()):
                batch = self.transfer_batch_to_device(batch, self.device, batch_idx)
                embeddings = self._probe_embeddings(batch)
                self._collect_validation_embeddings(
                    embeddings,
                    self._contrastive_metadata(batch),
                    batch,
                    collection=self._modality_probe_reference_batches,
                )
        if hasattr(data_module, "val_dataloader"):
            for batch_idx, batch in enumerate(data_module.val_dataloader()):
                batch = self.transfer_batch_to_device(batch, self.device, batch_idx)
                embeddings = self._probe_embeddings(batch)
                self._collect_validation_embeddings(
                    embeddings,
                    self._contrastive_metadata(batch),
                    batch,
                    collection=self._projection_embedding_batches,
                )
        if hasattr(data_module, "demographic_probe_reference_dataloader"):
            for batch_idx, batch in enumerate(data_module.demographic_probe_reference_dataloader()):
                batch = self.transfer_batch_to_device(batch, self.device, batch_idx)
                embeddings = self._probe_embeddings(batch)
                self._collect_validation_embeddings(
                    embeddings,
                    self._contrastive_metadata(batch),
                    batch,
                    collection=self._demographic_probe_reference_batches,
                )
        if self.enable_stage1_loss and hasattr(data_module, "stage1_monitor_dataloader"):
            self._val_stage1_batches = []
            for batch_idx, batch in enumerate(data_module.stage1_monitor_dataloader()):
                batch = self.transfer_batch_to_device(batch, self.device, batch_idx)
                embeddings = self._probe_embeddings(batch)
                if "z_anatomy" in embeddings:
                    self._record_stage1_validation_embeddings(embeddings, batch)

    @staticmethod
    def _probe_modality_id(batch: dict):
        """Modality id for the validation probe encoder call.

        Diagnostic-only: with ``ASPARAGUS_PROBE_IGNORE_MODALITY_ID`` set, probe the encoder with
        ``modality_id=None`` (FiLM "unknown" slot) to measure a FiLM-on checkpoint's representation
        the way the finetune path uses it (``model(x)`` with no modality id). Default keeps the real
        modality id, so training-run probes are unchanged.
        """
        if os.environ.get("ASPARAGUS_PROBE_IGNORE_MODALITY_ID", "").strip().lower() in {"1", "true", "yes", "on"}:
            return None
        return batch.get("modality_id")

    def _probe_embeddings(self, batch: dict) -> dict[str, torch.Tensor]:
        """Represent unmasked canonical images without invoking inactive heads."""
        model = self._get_ssl_model()
        encode_representations = getattr(model, "encode_representations", None)
        if callable(encode_representations):
            representations = encode_representations(batch["image"], modality_id=self._probe_modality_id(batch))
            h = representations["h_global"]
        else:
            representations = {}
            if not getattr(self, "_warned_legacy_probe_representation", False):
                logging.warning(
                    "%s lacks encode_representations; using deprecated validation-probe fallback.",
                    type(model).__name__,
                )
                self._warned_legacy_probe_representation = True
            _, encoder_features = self._forward_with_features(
                model, batch["image"], modality_id=self._probe_modality_id(batch)
            )
            h = self._legacy_global_features(encoder_features)
        embeddings = {"h": h}
        if self.enable_demo_loss:
            embeddings["z_demo"] = model.head_demo(h)
        if self.enable_patho_loss:
            embeddings["z_patho"] = model.head_patho(h)
        if getattr(self, "enable_scanner_pos_loss", False) and "z_scanner" in representations:
            embeddings["z_scanner"] = representations["z_scanner"]
        if getattr(self, "enable_inv_adv_loss", False) and "z_inv" in representations:
            embeddings["z_inv"] = representations["z_inv"]
        if getattr(self, "enable_modality_loss", False) and "z_mod" in representations:
            embeddings["z_mod"] = representations["z_mod"]
        if self.enable_stage1_loss:
            embeddings["z_anatomy"] = model.head_stage1_anatomy(h)
        return embeddings

    @staticmethod
    def _pool_feature_tensor(features: torch.Tensor) -> torch.Tensor:
        if features.ndim <= 2:
            return features
        return features.mean(dim=tuple(range(2, features.ndim)))

    @staticmethod
    def _legacy_global_features(features) -> torch.Tensor:
        try:
            return build_h_global(features)
        except (TypeError, ValueError):
            if not torch.is_tensor(features):
                raise
            return SSLValidationMonitorMixin._pool_feature_tensor(features)

    def _collect_validation_embeddings(
        self,
        embeddings: dict[str, torch.Tensor],
        metadata: dict,
        batch: Optional[dict] = None,
        collection: Optional[list] = None,
    ):
        if not self.validation_embedding_monitor_enabled or not embeddings or self.trainer.sanity_checking:
            return

        device = next(iter(embeddings.values())).device
        batch_size = next(iter(embeddings.values())).shape[0]
        pathology = self._metadata_tensor_or_default(metadata.get("pathology"), batch_size, device, torch.long, -1)
        fine_pathology = self._metadata_tensor_or_default(metadata.get("fine_pathology"), batch_size, device, torch.long, -1)
        age = self._metadata_tensor_or_default(metadata.get("age"), batch_size, device, torch.float32, float("nan"))
        sex = self._metadata_tensor_or_default(metadata.get("sex"), batch_size, device, torch.long, -1)
        scanner_id = self._metadata_tensor_or_default(metadata.get("scanner_id"), batch_size, device, torch.long, -1)
        dataset_id = self._metadata_tensor_or_default(metadata.get("dataset_id"), batch_size, device, torch.long, -1)

        gathered = {
            "pathology": self._gather_monitor_tensor(pathology).detach().cpu(),
            "fine_pathology": self._gather_monitor_tensor(fine_pathology).detach().cpu(),
            "age": self._gather_monitor_tensor(age).detach().cpu(),
            "sex": self._gather_monitor_tensor(sex).detach().cpu(),
            "scanner_id": self._gather_monitor_tensor(scanner_id).detach().cpu(),
            "dataset_id": self._gather_monitor_tensor(dataset_id).detach().cpu(),
            "features": {},
        }
        if batch is not None:
            modality_id = self._metadata_tensor_or_default(batch.get("modality_id"), batch_size, device, torch.long, -1)
            subject_id = self._subject_hash_tensor(batch.get("subject_key"), device)
            gathered["modality_id"] = self._gather_monitor_tensor(modality_id).detach().cpu()
            gathered["subject_id"] = self._gather_monitor_tensor(subject_id).detach().cpu()
            session_id = self._subject_hash_tensor(batch.get("subject_session_key"), device)
            gathered["session_id"] = self._gather_monitor_tensor(session_id).detach().cpu()
            # Diffusion b-value for per-DWI-subtype probe/projection cohorts (NaN for non-DWI).
            dwi_bval = self._metadata_tensor_or_default(batch.get("dwi_bval"), batch_size, device, torch.float32, float("nan"))
            gathered["dwi_bval"] = self._gather_monitor_tensor(dwi_bval).detach().cpu()
            field_strength_id = self._scanner_target_tensor_or_default(
                batch.get("scanner_targets"),
                "field_strength",
                batch_size,
                device,
            )
            gathered["field_strength_id"] = self._gather_monitor_tensor(field_strength_id).detach().cpu()
        for source in self._active_validation_embedding_sources():
            if source in embeddings:
                # Pool only for subject-level monitoring, before the DDP gather.
                feature = self._pool_feature_tensor(embeddings[source])
                gathered["features"][source] = self._gather_monitor_tensor(feature).detach().cpu()
        if gathered["features"]:
            (collection if collection is not None else self._val_embedding_batches).append(gathered)

    def _active_validation_embedding_sources(self) -> tuple[str, ...]:
        enabled = {"h"}
        if self.enable_demo_loss:
            enabled.add("z_demo")
        if self.enable_patho_loss:
            enabled.add("z_patho")
        if getattr(self, "enable_scanner_pos_loss", False):
            enabled.add("z_scanner")
        if getattr(self, "enable_inv_adv_loss", False):
            enabled.add("z_inv")
        if getattr(self, "enable_modality_loss", False):
            enabled.add("z_mod")
        if getattr(self, "enable_stage1_loss", False):
            enabled.add("z_anatomy")
        if getattr(self, "enable_stage2_loss", False):
            enabled.update({"z_anat_s2", "z_mod_s2"})
        requested = ("h", *self.validation_embedding_sources)
        return tuple(dict.fromkeys(source for source in requested if source in enabled))

    def _record_validation_metadata(self, metadata: dict, batch_size: int, device: torch.device):
        self._val_metadata_batches.append(
            {
                "pathology": self._metadata_tensor_or_default(metadata.get("pathology"), batch_size, device, torch.long, -1)
                .detach()
                .cpu(),
                "fine_pathology": self._metadata_tensor_or_default(
                    metadata.get("fine_pathology"), batch_size, device, torch.long, -1
                )
                .detach()
                .cpu(),
                "age": self._metadata_tensor_or_default(metadata.get("age"), batch_size, device, torch.float32, float("nan"))
                .detach()
                .cpu(),
                "sex": self._metadata_tensor_or_default(metadata.get("sex"), batch_size, device, torch.long, -1)
                .detach()
                .cpu(),
                "scanner_id": self._metadata_tensor_or_default(metadata.get("scanner_id"), batch_size, device, torch.long, -1)
                .detach()
                .cpu(),
                "modality_id": self._metadata_tensor_or_default(
                    metadata.get("modality_id"), batch_size, device, torch.long, -1
                )
                .detach()
                .cpu(),
                "dataset_id": self._metadata_tensor_or_default(metadata.get("dataset_id"), batch_size, device, torch.long, -1)
                .detach()
                .cpu(),
                "dwi_bval": self._metadata_tensor_or_default(
                    metadata.get("dwi_bval"), batch_size, device, torch.float32, float("nan")
                )
                .detach()
                .cpu(),
            }
        )

    def _record_stage1_validation_embeddings(self, embeddings: dict[str, torch.Tensor], batch: dict):
        features = embeddings["z_anatomy"]
        batch_size = features.shape[0]
        device = features.device
        modality_id = self._metadata_tensor_or_default(batch.get("modality_id"), batch_size, device, torch.long, -1)
        registered = self._metadata_tensor_or_default(batch.get("is_registered_subset"), batch_size, device, torch.bool, False)
        subject_key = batch.get("subject_key")
        if subject_key is None:
            subject_key = batch.get("subject_session_key")
        subject_id = self._subject_hash_tensor(subject_key, device)
        session_id = self._subject_hash_tensor(batch.get("subject_session_key"), device)
        row = {
            "features": self._gather_monitor_tensor(features.detach()).cpu(),
            "subject_id": self._gather_monitor_tensor(subject_id).detach().cpu(),
            "subject_session_id": self._gather_monitor_tensor(session_id).detach().cpu(),
            "modality_id": self._gather_monitor_tensor(modality_id).detach().cpu(),
            "registered_subset": self._gather_monitor_tensor(registered).detach().cpu(),
        }
        if "h" in embeddings:
            row["h_features"] = self._gather_monitor_tensor(embeddings["h"].detach()).cpu()
        if "z_mod" in embeddings:
            row["z_mod_features"] = self._gather_monitor_tensor(embeddings["z_mod"].detach()).cpu()
        self._val_stage1_batches.append(row)

    @staticmethod
    def _metadata_tensor_or_default(values, batch_size: int, device: torch.device, dtype: torch.dtype, default):
        if values is None:
            return torch.full((batch_size,), default, dtype=dtype, device=device)
        if isinstance(values, torch.Tensor):
            tensor = values.to(device=device, dtype=dtype).view(-1)
        else:
            tensor = torch.as_tensor(values, dtype=dtype, device=device).view(-1)
        if tensor.numel() == batch_size:
            return tensor
        if tensor.numel() == 1 and batch_size > 1:
            return tensor.expand(batch_size)
        raise ValueError(f"Metadata length {tensor.numel()} does not match validation batch size {batch_size}.")

    def _scanner_target_tensor_or_default(
        self,
        scanner_targets,
        target: str,
        batch_size: int,
        device: torch.device,
        default: int = -1,
    ) -> torch.Tensor:
        if not isinstance(scanner_targets, dict) or target not in scanner_targets:
            return torch.full((batch_size,), int(default), dtype=torch.long, device=device)
        return self._metadata_tensor_or_default(scanner_targets[target], batch_size, device, torch.long, default)

    def _gather_monitor_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return tensor
        if tensor.ndim == 0:
            tensor = tensor.view(1)
        tensor = tensor.contiguous()
        local_size = torch.tensor([tensor.shape[0]], dtype=torch.long, device=tensor.device)
        gathered_sizes = self.all_gather(local_size, sync_grads=False).view(-1)
        max_size = int(gathered_sizes.max().item())
        if max_size == 0:
            return tensor[:0]

        if tensor.shape[0] < max_size:
            pad_shape = (max_size - tensor.shape[0], *tensor.shape[1:])
            padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, padding], dim=0)

        gathered = self.all_gather(tensor, sync_grads=False).reshape(-1, max_size, *tensor.shape[1:])
        parts = [gathered[rank, : int(size.item())] for rank, size in enumerate(gathered_sizes)]
        return torch.cat(parts, dim=0)

    def _log_validation_embedding_monitor(
        self,
        on_step: bool = False,
        on_epoch: bool = True,
        direct_step: Optional[int] = None,
    ):
        if (
            not self.validation_embedding_monitor_enabled
            or not self._val_embedding_batches
            or not self._probe_reference_batches
        ):
            return

        query = self._monitor_collection_arrays(self._val_embedding_batches)
        projection_query = (
            self._monitor_collection_arrays(self._projection_embedding_batches)
            if getattr(self, "_projection_embedding_batches", None)
            else query
        )
        reference = self._monitor_collection_arrays(self._probe_reference_batches)
        modality_reference = (
            self._monitor_collection_arrays(self._modality_probe_reference_batches)
            if self._modality_probe_reference_batches
            else reference
        )
        demographic_reference = (
            self._monitor_collection_arrays(self._demographic_probe_reference_batches)
            if self._demographic_probe_reference_batches
            else reference
        )
        reference_subjects = set(reference["subject_id"].tolist())
        query_subjects = set(query["subject_id"].tolist())
        reference_sessions = set(reference["session_id"].tolist())
        query_sessions = set(query["session_id"].tolist())
        demographic_reference_subjects = set(demographic_reference["subject_id"].tolist())
        demographic_reference_sessions = set(demographic_reference["session_id"].tolist())
        modality_reference_subjects = set(modality_reference["subject_id"].tolist())
        modality_reference_sessions = set(modality_reference["session_id"].tolist())
        if (
            reference_subjects & query_subjects
            or reference_sessions & query_sessions
            or demographic_reference_subjects & query_subjects
            or demographic_reference_sessions & query_sessions
            or modality_reference_subjects & query_subjects
            or modality_reference_sessions & query_sessions
        ):
            raise ValueError("Probe leakage detected: training references overlap validation queries by subject or session.")

        stage = self._validation_stage()
        # Collapse and rank are objective-agnostic representation diagnostics, so the
        # reconstructive run gets the same non-collapse gate rather than a blind spot.
        health_by_source = {}
        for source, features in query["features"].items():
            health = embedding_health_metrics(torch.as_tensor(features))
            health_by_source[source] = health
            for metric, value in health.items():
                self._log_monitor_scalar(
                    f"{stage}/representation_health/{source}/{metric}",
                    value,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    direct_step=direct_step,
                )
            for modality_id in np.unique(query["modality_id"]):
                modality_mask = query["modality_id"] == modality_id
                if int(modality_mask.sum()) < 2:
                    continue
                modality_name = PretrainDataset.modality_name(int(modality_id))
                modality_health = embedding_health_metrics(torch.as_tensor(features[modality_mask]))
                for metric, value in modality_health.items():
                    self._log_monitor_scalar(
                        f"{stage}/representation_health/by_modality/{modality_name}/{source}/{metric}",
                        value,
                        on_step=on_step,
                        on_epoch=on_epoch,
                        direct_step=direct_step,
                    )

        t1w_id = PretrainDataset.MODALITY_TO_ID["t1w"]
        for token_name, modality_id, bval_range in self._demographic_token_specs():
            for namespace in self._modality_metric_namespaces(token_name):
                self._log_demographic_naive_baselines(
                    stage,
                    demographic_reference,
                    query,
                    modality_id=modality_id,
                    namespace=namespace,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    direct_step=direct_step,
                    bval_range=bval_range,
                )
        for source in self._active_validation_embedding_sources():
            if source not in query["features"] or source not in reference["features"]:
                continue
            query_features = query["features"][source]
            reference_features = reference["features"][source]
            t1w_query = query["modality_id"] == t1w_id
            t1w_reference = reference["modality_id"] == t1w_id
            pathology_silhouette = embedding_projection.compute_silhouette(
                query_features[t1w_query], query["pathology"][t1w_query]
            )
            self._log_silhouette_metrics(
                stage,
                source,
                "pathology",
                pathology_silhouette,
                on_step=on_step,
                on_epoch=on_epoch,
                direct_step=direct_step,
            )
            self._log_probe_metrics(
                stage,
                source,
                "site",
                embedding_projection.compute_reference_knn_classification_probe(
                    reference_features[t1w_reference],
                    reference["dataset_id"][t1w_reference],
                    query_features[t1w_query],
                    query["dataset_id"][t1w_query],
                    detailed=self.probe_detailed_metrics or self.exhaustive_evaluation,
                ),
                on_step=on_step,
                on_epoch=on_epoch,
                direct_step=direct_step,
            )
            self._log_probe_metrics(
                stage,
                source,
                "manufacturer",
                embedding_projection.compute_reference_knn_classification_probe(
                    reference_features[t1w_reference],
                    reference["scanner_id"][t1w_reference],
                    query_features[t1w_query],
                    query["scanner_id"][t1w_query],
                    detailed=self.probe_detailed_metrics or self.exhaustive_evaluation,
                ),
                on_step=on_step,
                on_epoch=on_epoch,
                direct_step=direct_step,
            )
            if source in {
                "h",
                "h_dense",
                "h_coarse",
                "h_global",
                "z_patho",
                "z_anatomy",
                "z_mod",
                "z_anat_s2",
                "z_mod_s2",
            }:
                self._log_probe_metrics(
                    stage,
                    source,
                    "pathology",
                    embedding_projection.compute_reference_knn_classification_probe(
                        reference_features,
                        reference["pathology"],
                        query_features[t1w_query],
                        query["pathology"][t1w_query],
                        detailed=self.probe_detailed_metrics or self.exhaustive_evaluation,
                    ),
                    on_step=on_step,
                    on_epoch=on_epoch,
                    direct_step=direct_step,
                )
                self._log_probe_metrics(
                    stage,
                    source,
                    "modality",
                    embedding_projection.compute_reference_knn_classification_probe(
                        modality_reference["features"].get(source, reference_features),
                        modality_reference["modality_id"]
                        if source in modality_reference["features"]
                        else reference["modality_id"],
                        query_features,
                        query["modality_id"],
                        detailed=self.probe_detailed_metrics or self.exhaustive_evaluation,
                    ),
                    on_step=on_step,
                    on_epoch=on_epoch,
                    direct_step=direct_step,
                )
                age_mae = embedding_projection.compute_reference_knn_age_mae(
                    reference_features,
                    reference["age"],
                    query_features,
                    query["age"],
                )
                self._log_monitor_scalar(
                    f"{stage}/probes/{source}/age/mae",
                    age_mae,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    direct_step=direct_step,
                )
                for metric_name, value in embedding_projection.compute_pathology_retrieval_metrics(
                    query_features,
                    query["pathology"],
                    fine_pathology=query["fine_pathology"],
                    dataset_id=query["dataset_id"],
                    modality_id=query["modality_id"],
                ).items():
                    self._log_monitor_scalar(
                        f"{stage}/retrieval/{source}/{metric_name}",
                        value,
                        on_step=on_step,
                        on_epoch=on_epoch,
                        direct_step=direct_step,
                    )
            if source in {"h", "h_dense", "h_coarse", "h_global", "z_demo"} and source in demographic_reference["features"]:
                for token_name, modality_id, bval_range in self._demographic_token_specs():
                    reference_eligible = self._demographic_probe_mask(demographic_reference, modality_id, bval_range)
                    query_eligible = self._demographic_probe_mask(query, modality_id, bval_range)
                    age_query_eligible = self._demographic_age_probe_mask(query, modality_id, bval_range)
                    sex_silhouette = embedding_projection.compute_silhouette(
                        query_features[query_eligible], query["sex"][query_eligible]
                    )
                    age_silhouette = embedding_projection.compute_silhouette(
                        query_features[age_query_eligible],
                        self._age_bin_labels(query["age"][age_query_eligible]),
                    )
                    sex_metrics = embedding_projection.compute_reference_knn_classification_probe(
                        demographic_reference["features"][source][reference_eligible],
                        demographic_reference["sex"][reference_eligible],
                        query_features[query_eligible],
                        query["sex"][query_eligible],
                        detailed=self.probe_detailed_metrics or self.exhaustive_evaluation,
                    )
                    age_mae = embedding_projection.compute_reference_knn_age_mae(
                        demographic_reference["features"][source][reference_eligible],
                        demographic_reference["age"][reference_eligible],
                        query_features[query_eligible],
                        query["age"][query_eligible],
                    )
                    for metric_namespace in self._modality_metric_namespaces(token_name):
                        self._log_silhouette_metrics(
                            stage,
                            source,
                            "sex",
                            sex_silhouette,
                            namespace=metric_namespace,
                            on_step=on_step,
                            on_epoch=on_epoch,
                            direct_step=direct_step,
                        )
                        self._log_silhouette_metrics(
                            stage,
                            source,
                            "age_bin",
                            age_silhouette,
                            namespace=metric_namespace,
                            on_step=on_step,
                            on_epoch=on_epoch,
                            direct_step=direct_step,
                        )
                        self._log_probe_metrics(
                            stage,
                            source,
                            "sex",
                            sex_metrics,
                            namespace=metric_namespace,
                            on_step=on_step,
                            on_epoch=on_epoch,
                            direct_step=direct_step,
                        )
                        probe_prefix = f"{stage}/probes/{source}"
                        if metric_namespace:
                            probe_prefix = f"{probe_prefix}/{metric_namespace}"
                        self._log_monitor_scalar(
                            f"{probe_prefix}/knn_age/mae",
                            age_mae,
                            on_step=on_step,
                            on_epoch=on_epoch,
                            direct_step=direct_step,
                        )
            if self.trainer.is_global_zero and self.validation_embedding_monitor_enabled:
                for token_name, modality_id, bval_range in self._monitor_token_specs():
                    modality_mask = (query["modality_id"] == modality_id) & self._bval_mask(query, bval_range)
                    control_projection_mask = self._control_projection_mask(query, modality_id, bval_range)
                    modality_control_mask = control_projection_mask[modality_mask]
                    for namespace_prefix in self._modality_metric_namespaces(token_name):
                        prefix = f"{namespace_prefix}/" if namespace_prefix else ""
                        self._log_embedding_projection_figures(
                            source,
                            query_features[control_projection_mask],
                            query["pathology"][control_projection_mask],
                            query["age"][control_projection_mask],
                            query["sex"][control_projection_mask],
                            query["modality_id"][control_projection_mask],
                            query["scanner_id"][control_projection_mask],
                            query["dataset_id"][control_projection_mask],
                            query["field_strength_id"][control_projection_mask],
                            namespace=f"{prefix}control_only",
                            style="control_age",
                            on_step=on_step,
                            on_epoch=on_epoch,
                            direct_step=direct_step,
                        )
                        self._log_embedding_projection_figures(
                            source,
                            query_features[modality_mask],
                            query["pathology"][modality_mask],
                            query["age"][modality_mask],
                            query["sex"][modality_mask],
                            query["modality_id"][modality_mask],
                            query["scanner_id"][modality_mask],
                            query["dataset_id"][modality_mask],
                            query["field_strength_id"][modality_mask],
                            namespace=f"{prefix}controls_with_pathologies",
                            style="pathology",
                            reference_mask=modality_control_mask,
                            on_step=on_step,
                            on_epoch=on_epoch,
                            direct_step=direct_step,
                        )
                projection_features = projection_query["features"].get(source)
                if projection_features is not None:
                    # Routine validation is stratified over modality/pathology/scanner, so these figures
                    # are the all-modality visual check while train-reference/query probes stay unchanged.
                    present_modalities = np.unique(projection_query["modality_id"]).tolist()
                    modality_names = {int(m): PretrainDataset.modality_name(int(m)) for m in present_modalities}
                    self._log_embedding_projection_figures(
                        source,
                        projection_features,
                        projection_query["pathology"],
                        projection_query["age"],
                        projection_query["sex"],
                        projection_query["modality_id"],
                        projection_query["scanner_id"],
                        projection_query["dataset_id"],
                        projection_query["field_strength_id"],
                        namespace="all_modalities",
                        style="modality",
                        modality_names=modality_names,
                        on_step=on_step,
                        on_epoch=on_epoch,
                        direct_step=direct_step,
                    )
                    self._log_embedding_projection_figures(
                        source,
                        projection_features,
                        projection_query["pathology"],
                        projection_query["age"],
                        projection_query["sex"],
                        projection_query["modality_id"],
                        projection_query["scanner_id"],
                        projection_query["dataset_id"],
                        projection_query["field_strength_id"],
                        namespace="scanner_acquisition",
                        style="scanner_acquisition",
                        modality_names=modality_names,
                        field_strength_names=self._scanner_target_label_names("field_strength"),
                        on_step=on_step,
                        on_epoch=on_epoch,
                        direct_step=direct_step,
                    )

    def _demographic_token_specs(self) -> tuple[tuple[str, int, Optional[tuple[float, float]]], ...]:
        """Per-token monitor specs ``(name, modality_id, bval_range)`` for the demographic probes."""
        return tuple((token.name, int(token.modality_id), token.bval_range) for token in self.demographic_tokens)

    def _monitor_token_specs(self) -> tuple[tuple[str, int, Optional[tuple[float, float]]], ...]:
        """Projection-cohort specs: always t1w, plus each demographic token (deduped by name)."""
        specs = [("t1w", PretrainDataset.MODALITY_TO_ID["t1w"], None)]
        seen = {"t1w"}
        for name, modality_id, bval_range in self._demographic_token_specs():
            if name in seen:
                continue
            seen.add(name)
            specs.append((name, modality_id, bval_range))
        return tuple(specs)

    def _monitor_modality_ids(self) -> tuple[int, ...]:
        """Distinct modality ids monitored (t1w plus the demographic tokens' modalities)."""
        return tuple(dict.fromkeys(modality_id for _, modality_id, _ in self._monitor_token_specs()))

    def _modality_metric_namespaces(self, name: str) -> tuple[str, ...]:
        namespace = f"by_modality/{name}"
        if name == "t1w":
            return namespace, ""
        return (namespace,)

    @staticmethod
    def _bval_mask(arrays: dict, bval_range: Optional[tuple[float, float]]) -> np.ndarray:
        if bval_range is None:
            return np.ones_like(arrays["modality_id"], dtype=bool)
        bval = arrays.get("dwi_bval")
        if bval is None:
            return np.zeros_like(arrays["modality_id"], dtype=bool)
        bval = np.asarray(bval, dtype=np.float32)
        bval_min, bval_max = float(bval_range[0]), float(bval_range[1])
        return np.isfinite(bval) & (bval >= bval_min) & (bval <= bval_max)

    @staticmethod
    def _monitor_collection_arrays(collection: list[dict]) -> dict:
        row_count = int(torch.cat([batch["pathology"] for batch in collection], dim=0).numel())
        float_keys = {"age", "dwi_bval"}
        values = {
            key: embedding_projection.tensor_to_1d_numpy(
                torch.cat([batch[key] for batch in collection], dim=0)
                if key in collection[0]
                else torch.full((row_count,), float("nan") if key in float_keys else -1),
                np.float32 if key in float_keys else np.int64,
            )
            for key in (
                "pathology",
                "fine_pathology",
                "age",
                "sex",
                "scanner_id",
                "dataset_id",
                "field_strength_id",
                "modality_id",
                "dwi_bval",
                "subject_id",
                "session_id",
            )
        }
        sources = set.intersection(*(set(batch["features"]) for batch in collection))
        values["features"] = {
            source: embedding_projection.tensor_to_2d_numpy(
                torch.cat([batch["features"][source] for batch in collection], dim=0)
            )
            for source in sources
        }
        return values

    def _scanner_target_label_names(self, target: str) -> dict[int, str]:
        data_module = getattr(self.trainer, "datamodule", None)
        encoder = getattr(data_module, "scanner_target_encoder", None)
        vocab = getattr(encoder, "vocab", None)
        if not isinstance(vocab, dict):
            return {}
        target_vocab = vocab.get(target, {})
        if not isinstance(target_vocab, dict):
            return {}
        return {int(index): str(label) for label, index in target_vocab.items()}

    def _demographic_probe_mask(
        self, arrays: dict, modality_id: Optional[int] = None, bval_range: Optional[tuple[float, float]] = None
    ) -> np.ndarray:
        modality_id = self.demographic_modalities[0] if modality_id is None else int(modality_id)
        return (
            (arrays["pathology"] == self.demographic_eligible_pathology)
            & (arrays["modality_id"] == modality_id)
            & np.isfinite(arrays["age"])
            & (arrays["sex"] >= 0)
            & self._bval_mask(arrays, bval_range)
        )

    def _demographic_age_probe_mask(
        self, arrays: dict, modality_id: Optional[int] = None, bval_range: Optional[tuple[float, float]] = None
    ) -> np.ndarray:
        modality_id = self.demographic_modalities[0] if modality_id is None else int(modality_id)
        return (
            (arrays["pathology"] == self.demographic_eligible_pathology)
            & (arrays["modality_id"] == modality_id)
            & np.isfinite(arrays["age"])
            & self._bval_mask(arrays, bval_range)
        )

    def _control_projection_mask(
        self, arrays: dict, modality_id: Optional[int] = None, bval_range: Optional[tuple[float, float]] = None
    ) -> np.ndarray:
        mask = (arrays["pathology"] == 0) & np.isfinite(arrays["age"]) & (arrays["sex"] >= 0)
        if modality_id is not None:
            mask &= arrays["modality_id"] == int(modality_id)
        return mask & self._bval_mask(arrays, bval_range)

    def _age_bin_labels(self, ages: np.ndarray) -> np.ndarray:
        labels = np.full(np.asarray(ages).shape, -1, dtype=np.int64)
        valid = np.isfinite(ages)
        labels[valid] = np.floor(ages[valid] / self.demographic_silhouette_age_bin_years).astype(np.int64)
        return labels

    def _log_monitor_scalar(
        self,
        name: str,
        value,
        on_step: bool,
        on_epoch: bool,
        direct_step: Optional[int] = None,
    ):
        if direct_step is None:
            self.log(
                name,
                value,
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=False,
                rank_zero_only=True,
            )
            return

        if not getattr(self.trainer, "is_global_zero", True):
            return
        for logger in getattr(self.trainer, "loggers", []):
            if hasattr(logger, "log_metrics"):
                logger.log_metrics({name: value}, step=direct_step)
            elif hasattr(logger, "experiment") and hasattr(logger.experiment, "log"):
                logger.experiment.log({name: value}, step=direct_step)

    def _log_silhouette_metrics(
        self,
        stage: str,
        source: str,
        target: str,
        silhouette: embedding_projection.SilhouetteResult,
        on_step: bool,
        on_epoch: bool,
        namespace: str = "",
        direct_step: Optional[int] = None,
    ):
        prefix = f"{stage}/probes/{source}"
        if namespace:
            prefix = f"{prefix}/{namespace}"
        self._log_monitor_scalar(
            f"{prefix}/silhouette_{target}",
            silhouette.score,
            on_step=on_step,
            on_epoch=on_epoch,
            direct_step=direct_step,
        )
        self._log_monitor_scalar(
            f"{prefix}/valid_{target}_count",
            float(silhouette.valid_count),
            on_step=on_step,
            on_epoch=on_epoch,
            direct_step=direct_step,
        )
        self._log_monitor_scalar(
            f"{prefix}/{target}_class_count",
            float(silhouette.class_count),
            on_step=on_step,
            on_epoch=on_epoch,
            direct_step=direct_step,
        )

    def _log_projection_silhouette_metrics(
        self,
        source: str,
        namespace: str,
        reducer: str,
        coords: np.ndarray,
        labels_by_target: dict[str, np.ndarray | None],
        on_step: bool,
        on_epoch: bool,
        direct_step: Optional[int] = None,
    ):
        if namespace != "all_modalities":
            return
        prefix = f"{self._validation_stage()}/embeddings/{source}/{namespace}/{reducer}"
        for target, labels in labels_by_target.items():
            if labels is None:
                continue
            silhouette = embedding_projection.compute_silhouette(coords, labels)
            self._log_monitor_scalar(
                f"{prefix}/silhouette_{target}",
                silhouette.score,
                on_step=on_step,
                on_epoch=on_epoch,
                direct_step=direct_step,
            )
            self._log_monitor_scalar(
                f"{prefix}/valid_{target}_count",
                float(silhouette.valid_count),
                on_step=on_step,
                on_epoch=on_epoch,
                direct_step=direct_step,
            )
            self._log_monitor_scalar(
                f"{prefix}/{target}_class_count",
                float(silhouette.class_count),
                on_step=on_step,
                on_epoch=on_epoch,
                direct_step=direct_step,
            )

    def _log_demographic_naive_baselines(
        self,
        stage: str,
        reference: dict,
        query: dict,
        modality_id: Optional[int] = None,
        namespace: str = "",
        on_step: bool = False,
        on_epoch: bool = True,
        direct_step: Optional[int] = None,
        bval_range: Optional[tuple[float, float]] = None,
    ):
        reference_eligible = self._demographic_probe_mask(reference, modality_id, bval_range)
        query_eligible = self._demographic_probe_mask(query, modality_id, bval_range)
        prefix = f"{stage}/probes/demographic_naive"
        if namespace:
            prefix = f"{prefix}/{namespace}"
        age_mae = embedding_projection.compute_reference_median_age_mae(
            reference["age"][reference_eligible],
            query["age"][query_eligible],
        )
        self._log_monitor_scalar(
            f"{prefix}/median_age/mae",
            age_mae,
            on_step=on_step,
            on_epoch=on_epoch,
            direct_step=direct_step,
        )
        for metric_name, value in embedding_projection.compute_reference_majority_classification_baseline(
            reference["sex"][reference_eligible],
            query["sex"][query_eligible],
        ).items():
            self._log_monitor_scalar(
                f"{prefix}/majority_sex/{metric_name}",
                value,
                on_step=on_step,
                on_epoch=on_epoch,
                direct_step=direct_step,
            )

    def _log_probe_metrics(
        self,
        stage: str,
        source: str,
        target: str,
        metrics: dict[str, float],
        on_step: bool = False,
        on_epoch: bool = True,
        namespace: str = "",
        direct_step: Optional[int] = None,
    ):
        probe_prefix = f"{stage}/probes/{source}"
        embedding_prefix = f"{stage}/embeddings/{source}"
        if namespace:
            probe_prefix = f"{probe_prefix}/{namespace}"
            embedding_prefix = f"{embedding_prefix}/{namespace}"
        for metric_name, value in metrics.items():
            self._log_monitor_scalar(
                f"{probe_prefix}/knn_{target}/{metric_name}",
                value,
                on_step=on_step,
                on_epoch=on_epoch,
                direct_step=direct_step,
            )
            # Mirror kNN probe scalars under embeddings for dashboards that group all representation diagnostics.
            self._log_monitor_scalar(
                f"{embedding_prefix}/knn_{target}/{metric_name}",
                value,
                on_step=on_step,
                on_epoch=on_epoch,
                direct_step=direct_step,
            )

    def _log_embedding_projection_figures(
        self,
        source: str,
        features_np,
        pathology_np,
        age_np,
        sex_np,
        modality_np,
        scanner_np=None,
        dataset_np=None,
        field_strength_np=None,
        namespace: str = "controls_with_pathologies",
        style: str = "pathology",
        reference_mask: Optional[np.ndarray] = None,
        modality_names: Optional[dict] = None,
        field_strength_names: Optional[dict] = None,
        on_step: bool = False,
        on_epoch: bool = True,
        direct_step: Optional[int] = None,
    ):
        if features_np.shape[0] < 2:
            return
        wandb_loggers = [
            logger for logger in getattr(self.trainer, "loggers", []) if "WandbLogger" in logger.__class__.__name__
        ]
        reducers = tuple(self.validation_embedding_reducers)
        if not wandb_loggers:
            if namespace != "all_modalities":
                return
            reducers = tuple(reducer for reducer in reducers if reducer.lower() == "pca")
            if not reducers:
                return

        indices = embedding_projection.deterministic_limit_indices(
            features_np.shape[0],
            self.validation_embedding_max_points,
            reference_mask=reference_mask,
            min_reference_points=3,
        )
        features_np = features_np[indices]
        pathology_np = pathology_np[indices]
        age_np = age_np[indices]
        sex_np = sex_np[indices]
        modality_np = modality_np[indices]
        scanner_np = scanner_np[indices] if scanner_np is not None else None
        dataset_np = dataset_np[indices] if dataset_np is not None else None
        field_strength_np = field_strength_np[indices] if field_strength_np is not None else None
        if reference_mask is not None:
            reference_mask = np.asarray(reference_mask, dtype=bool)[indices]
        current_epoch = int(getattr(self, "current_epoch", 0))
        for reducer in reducers:
            random_state = self.validation_embedding_random_state + current_epoch
            if reference_mask is None:
                coords = embedding_projection.reduce_embeddings(features_np, reducer=reducer, random_state=random_state)
            else:
                reference_count = int(reference_mask.sum())
                if reducer.lower() == "pca" and reference_count < 2:
                    continue
                if reducer.lower() == "umap" and reference_count < 3:
                    continue
                coords = embedding_projection.reduce_embeddings_projecting_reference(
                    features_np,
                    reducer=reducer,
                    random_state=random_state,
                    reference_mask=reference_mask,
                )
            self._log_projection_silhouette_metrics(
                source,
                namespace,
                reducer,
                coords,
                {
                    "modality_id": modality_np,
                    "scanner_id": scanner_np,
                    "dataset_id": dataset_np,
                    "field_strength_id": field_strength_np,
                },
                on_step=on_step,
                on_epoch=on_epoch,
                direct_step=direct_step,
            )
            if not wandb_loggers:
                continue
            import wandb

            title = f"Validation {source} {namespace} {reducer.upper()} epoch {current_epoch}"
            fig = embedding_projection.make_projection_figure(
                coords,
                pathology_np,
                age_np,
                sex_np,
                title,
                style=style,
                modality=modality_np,
                modality_names=modality_names,
                scanner_id=scanner_np,
                field_strength=field_strength_np,
                field_strength_names=field_strength_names,
            )
            rows = embedding_projection.make_projection_rows(
                coords, pathology_np, age_np, sex_np, modality_np, scanner_np, dataset_np, field_strength_np
            )
            table = wandb.Table(
                columns=embedding_projection.PROJECTION_ROW_COLUMNS,
                data=rows,
            )
            for logger in wandb_loggers:
                logger.experiment.log(
                    {
                        f"{self._validation_stage()}/embeddings/{source}/{namespace}/{reducer}": wandb.Image(fig),
                        f"{self._validation_stage()}/embeddings/{source}/{namespace}/{reducer}_table": table,
                    },
                    step=self.global_step,
                )
            import matplotlib.pyplot as plt

            plt.close(fig)
