"""
Validation embedding projection utilities for SSL pretraining diagnostics.
"""

# ruff: noqa: E402

from __future__ import annotations

import math
import os
import tempfile

# UMAP imports numba at module import time; set the cache before importing umap.
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "asparagus_numba_cache"))
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import numpy as np
import torch
import umap
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from typing import Iterable

PATHOLOGY_CLASS_NAMES = {
    -1: "Unknown",
    0: "Control",
    1: "Neurodegenerative",
    2: "Psychiatric / Neurodevelopmental",
    3: "Tumor / Oncology",
    4: "Vascular / Hemorrhage",
    5: "Other Structural",
}


@dataclass(frozen=True)
class SilhouetteResult:
    score: float
    valid_count: int
    class_count: int


def compute_knn_classification_probe(features: np.ndarray, labels: np.ndarray, n_neighbors: int = 5) -> dict[str, float]:
    """Leave-one-out kNN diagnostic; this is a validation monitor, not a final evaluation model."""
    valid = labels >= 0
    features = features[valid]
    labels = labels[valid].astype(np.int64, copy=False)
    classes = np.unique(labels)
    if features.shape[0] < 3 or classes.size < 2:
        return {}

    scaled = StandardScaler().fit_transform(features)
    distances = np.linalg.norm(scaled[:, None, :] - scaled[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    k = min(n_neighbors, features.shape[0] - 1)
    neighbor_indices = np.argsort(distances, axis=1)[:, :k]
    neighbor_distances = np.take_along_axis(distances, neighbor_indices, axis=1)
    weights = 1.0 / np.maximum(neighbor_distances, 1e-8)
    probabilities = np.zeros((labels.size, classes.size), dtype=np.float32)
    for class_index, class_id in enumerate(classes):
        probabilities[:, class_index] = (weights * (labels[neighbor_indices] == class_id)).sum(axis=1) / weights.sum(axis=1)
    predictions = classes[np.argmax(probabilities, axis=1)]
    one_hot = labels[:, None] == classes[None, :]
    metrics = {
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=classes, average="macro", zero_division=0)),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece": _expected_calibration_error(probabilities.max(axis=1), predictions == labels),
    }
    confusion = confusion_matrix(labels, predictions, labels=classes)
    for class_index, class_id in enumerate(classes):
        true_positive = confusion[class_index, class_index]
        false_negative = confusion[class_index, :].sum() - true_positive
        false_positive = confusion[:, class_index].sum() - true_positive
        true_negative = confusion.sum() - true_positive - false_negative - false_positive
        metrics[f"class_{int(class_id)}_recall"] = float(true_positive / max(1, true_positive + false_negative))
        metrics[f"class_{int(class_id)}_specificity"] = float(true_negative / max(1, true_negative + false_positive))
        for predicted_index, predicted_id in enumerate(classes):
            metrics[f"confusion/true_{int(class_id)}_pred_{int(predicted_id)}"] = float(
                confusion[class_index, predicted_index]
            )
    auprcs = []
    for class_index, class_id in enumerate(classes):
        class_target = one_hot[:, class_index].astype(np.int64)
        metrics[f"class_{int(class_id)}_f1"] = float(f1_score(class_target, predictions == class_id, zero_division=0))
        if np.unique(class_target).size > 1:
            score = float(average_precision_score(class_target, probabilities[:, class_index]))
            metrics[f"class_{int(class_id)}_auprc"] = score
            auprcs.append(score)
    if auprcs:
        metrics["macro_auprc"] = float(np.mean(auprcs))
    return metrics


def compute_reference_knn_classification_probe(
    reference_features: np.ndarray,
    reference_labels: np.ndarray,
    query_features: np.ndarray,
    query_labels: np.ndarray,
    n_neighbors: int = 5,
    detailed: bool = False,
) -> dict[str, float]:
    """Evaluate validation queries against training-only kNN references."""
    reference_valid = reference_labels >= 0
    query_valid = query_labels >= 0
    reference_features = reference_features[reference_valid]
    reference_labels = reference_labels[reference_valid].astype(np.int64, copy=False)
    query_features = query_features[query_valid]
    query_labels = query_labels[query_valid].astype(np.int64, copy=False)
    classes = np.unique(np.concatenate([reference_labels, query_labels])) if query_labels.size else np.array([])
    if reference_features.shape[0] < 1 or query_features.shape[0] < 1 or classes.size < 2:
        return {}

    scaler = StandardScaler().fit(reference_features)
    reference_scaled = scaler.transform(reference_features)
    query_scaled = scaler.transform(query_features)
    distances = np.linalg.norm(query_scaled[:, None, :] - reference_scaled[None, :, :], axis=-1)
    k = min(n_neighbors, reference_features.shape[0])
    neighbor_indices = np.argsort(distances, axis=1)[:, :k]
    neighbor_distances = np.take_along_axis(distances, neighbor_indices, axis=1)
    weights = 1.0 / np.maximum(neighbor_distances, 1e-8)
    probabilities = np.zeros((query_labels.size, classes.size), dtype=np.float32)
    for class_index, class_id in enumerate(classes):
        probabilities[:, class_index] = (weights * (reference_labels[neighbor_indices] == class_id)).sum(axis=1) / weights.sum(
            axis=1
        )
    predictions = classes[np.argmax(probabilities, axis=1)]
    one_hot = query_labels[:, None] == classes[None, :]
    metrics = {
        "mcc": float(matthews_corrcoef(query_labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(query_labels, predictions)),
        "macro_f1": float(f1_score(query_labels, predictions, labels=classes, average="macro", zero_division=0)),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece": _expected_calibration_error(probabilities.max(axis=1), predictions == query_labels),
        "reference_count": float(reference_features.shape[0]),
        "query_count": float(query_features.shape[0]),
    }
    auprcs = []
    for class_index, class_id in enumerate(classes):
        class_target = one_hot[:, class_index].astype(np.int64)
        if np.unique(class_target).size > 1:
            score = float(average_precision_score(class_target, probabilities[:, class_index]))
            if detailed:
                metrics[f"class_{int(class_id)}_auprc"] = score
            auprcs.append(score)
    if auprcs:
        metrics["macro_auprc"] = float(np.mean(auprcs))

    if detailed:
        confusion = confusion_matrix(query_labels, predictions, labels=classes)
        for class_index, class_id in enumerate(classes):
            class_target = one_hot[:, class_index].astype(np.int64)
            metrics[f"class_{int(class_id)}_f1"] = float(f1_score(class_target, predictions == class_id, zero_division=0))
            true_positive = confusion[class_index, class_index]
            false_negative = confusion[class_index, :].sum() - true_positive
            false_positive = confusion[:, class_index].sum() - true_positive
            true_negative = confusion.sum() - true_positive - false_negative - false_positive
            metrics[f"class_{int(class_id)}_recall"] = float(true_positive / max(1, true_positive + false_negative))
            metrics[f"class_{int(class_id)}_specificity"] = float(true_negative / max(1, true_negative + false_positive))
            for predicted_index, predicted_id in enumerate(classes):
                metrics[f"confusion/true_{int(class_id)}_pred_{int(predicted_id)}"] = float(
                    confusion[class_index, predicted_index]
                )
    return metrics


def compute_knn_age_mae(features: np.ndarray, ages: np.ndarray, n_neighbors: int = 5) -> float:
    valid = np.isfinite(ages)
    features = features[valid]
    ages = ages[valid]
    if features.shape[0] < 2:
        return float("nan")
    scaled = StandardScaler().fit_transform(features)
    distances = np.linalg.norm(scaled[:, None, :] - scaled[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    k = min(n_neighbors, features.shape[0] - 1)
    indices = np.argsort(distances, axis=1)[:, :k]
    weights = 1.0 / np.maximum(np.take_along_axis(distances, indices, axis=1), 1e-8)
    predictions = (weights * ages[indices]).sum(axis=1) / weights.sum(axis=1)
    return float(np.mean(np.abs(predictions - ages)))


def compute_reference_knn_age_mae(
    reference_features: np.ndarray,
    reference_ages: np.ndarray,
    query_features: np.ndarray,
    query_ages: np.ndarray,
    n_neighbors: int = 5,
) -> float:
    """Predict validation ages using only valid training-reference neighbors."""
    reference_valid = np.isfinite(reference_ages)
    query_valid = np.isfinite(query_ages)
    reference_features = reference_features[reference_valid]
    reference_ages = reference_ages[reference_valid]
    query_features = query_features[query_valid]
    query_ages = query_ages[query_valid]
    if reference_features.shape[0] < 1 or query_features.shape[0] < 1:
        return float("nan")
    scaler = StandardScaler().fit(reference_features)
    distances = np.linalg.norm(
        scaler.transform(query_features)[:, None, :] - scaler.transform(reference_features)[None, :, :],
        axis=-1,
    )
    k = min(n_neighbors, reference_features.shape[0])
    indices = np.argsort(distances, axis=1)[:, :k]
    weights = 1.0 / np.maximum(np.take_along_axis(distances, indices, axis=1), 1e-8)
    predictions = (weights * reference_ages[indices]).sum(axis=1) / weights.sum(axis=1)
    return float(np.mean(np.abs(predictions - query_ages)))


def compute_pathology_retrieval_metrics(
    features: np.ndarray,
    pathology: np.ndarray,
    fine_pathology: np.ndarray | None = None,
    dataset_id: np.ndarray | None = None,
    modality_id: np.ndarray | None = None,
    top_k: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """Nearest-neighbor retrieval diagnostics for pathology structure."""
    valid = pathology >= 0
    features = features[valid]
    pathology = pathology[valid].astype(np.int64, copy=False)
    if fine_pathology is not None:
        fine_pathology = fine_pathology[valid].astype(np.int64, copy=False)
    if dataset_id is not None:
        dataset_id = dataset_id[valid].astype(np.int64, copy=False)
    if modality_id is not None:
        modality_id = modality_id[valid].astype(np.int64, copy=False)
    if features.shape[0] < 2 or np.unique(pathology).size < 2:
        return {"eligible_anchor_count": 0.0}

    scaled = StandardScaler().fit_transform(features)
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    normalized = scaled / np.maximum(norms, 1e-8)
    similarities = normalized @ normalized.T
    np.fill_diagonal(similarities, -np.inf)
    candidates = np.isfinite(similarities)
    same_macro = pathology[:, None] == pathology[None, :]
    positive_masks = {"same_macro": same_macro & candidates}
    if fine_pathology is not None:
        same_fine = (fine_pathology[:, None] == fine_pathology[None, :]) & (fine_pathology[:, None] != -1)
        positive_masks["same_fine"] = same_fine & same_macro & candidates
    if dataset_id is not None:
        positive_masks["same_macro_different_site"] = positive_masks["same_macro"] & (
            dataset_id[:, None] != dataset_id[None, :]
        )
    if modality_id is not None:
        positive_masks["same_macro_different_modality"] = positive_masks["same_macro"] & (
            modality_id[:, None] != modality_id[None, :]
        )

    metrics = {"eligible_anchor_count": float(features.shape[0])}
    order = np.argsort(-similarities, axis=1)
    for name, positives in positive_masks.items():
        anchor_valid = positives.any(axis=1)
        metrics[f"{name}/eligible_anchor_count"] = float(anchor_valid.sum())
        if not anchor_valid.any():
            continue
        for k in top_k:
            kk = min(int(k), max(1, features.shape[0] - 1))
            hit = positives[np.arange(features.shape[0])[:, None], order[:, :kk]].any(axis=1)
            metrics[f"{name}/top{int(k)}"] = float(hit[anchor_valid].mean())
    return metrics


def compute_reference_median_age_mae(reference_ages: np.ndarray, query_ages: np.ndarray) -> float:
    """Null age predictor fit on training references and evaluated on validation queries."""
    reference_ages = reference_ages[np.isfinite(reference_ages)]
    query_ages = query_ages[np.isfinite(query_ages)]
    if reference_ages.shape[0] < 1 or query_ages.shape[0] < 1:
        return float("nan")
    prediction = float(np.median(reference_ages))
    return float(np.mean(np.abs(query_ages - prediction)))


def compute_reference_majority_classification_baseline(
    reference_labels: np.ndarray,
    query_labels: np.ndarray,
) -> dict[str, float]:
    """Null categorical predictor fit on training prevalence only."""
    reference_labels = reference_labels[reference_labels >= 0].astype(np.int64, copy=False)
    query_labels = query_labels[query_labels >= 0].astype(np.int64, copy=False)
    classes = np.unique(np.concatenate([reference_labels, query_labels])) if query_labels.size else np.array([])
    if reference_labels.shape[0] < 1 or query_labels.shape[0] < 1 or classes.size < 2:
        return {}
    train_classes, counts = np.unique(reference_labels, return_counts=True)
    majority_label = train_classes[np.argmax(counts)]
    predictions = np.full(query_labels.shape, majority_label, dtype=np.int64)
    return {
        "mcc": float(matthews_corrcoef(query_labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(query_labels, predictions)),
        "macro_f1": float(f1_score(query_labels, predictions, labels=classes, average="macro", zero_division=0)),
        "reference_count": float(reference_labels.shape[0]),
        "query_count": float(query_labels.shape[0]),
    }


def _expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    error = 0.0
    for low, high in zip(np.linspace(0.0, 1.0, n_bins, endpoint=False), np.linspace(0.1, 1.0, n_bins)):
        selected = (confidence > low) & (confidence <= high)
        if selected.any():
            error += float(selected.mean()) * abs(float(confidence[selected].mean()) - float(correct[selected].mean()))
    return error


def tensor_to_2d_numpy(features: torch.Tensor) -> np.ndarray:
    """Convert feature tensors to finite float32 [N, C] arrays."""
    if features.ndim > 2:
        features = features.flatten(1)
    array = features.detach().cpu().to(torch.float32).numpy()
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def tensor_to_1d_numpy(values: torch.Tensor | Iterable, dtype) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(list(values))
    return array.reshape(-1).astype(dtype, copy=False)


def deterministic_limit_indices(
    n_items: int,
    max_points: int,
    reference_mask: np.ndarray | None = None,
    min_reference_points: int = 3,
) -> np.ndarray:
    """Select a deterministic subset while preserving coverage across the validation stream."""
    if max_points <= 0:
        return np.empty((0,), dtype=np.int64)
    if n_items <= max_points:
        return np.arange(n_items, dtype=np.int64)
    base_indices = np.linspace(0, n_items - 1, num=max_points, dtype=np.int64)
    if reference_mask is None:
        return base_indices

    reference_mask = np.asarray(reference_mask, dtype=bool)
    if reference_mask.shape[0] != n_items:
        raise ValueError("reference_mask must have the same length as the number of items.")
    reference_indices = np.flatnonzero(reference_mask)
    required_reference_count = min(int(reference_indices.size), int(min_reference_points), int(max_points))
    if required_reference_count <= 0 or int(reference_mask[base_indices].sum()) >= required_reference_count:
        return base_indices

    forced_reference = reference_indices[
        np.linspace(0, reference_indices.size - 1, num=required_reference_count, dtype=np.int64)
    ]
    selected = set(int(index) for index in forced_reference)
    for index in base_indices:
        if len(selected) >= max_points:
            break
        selected.add(int(index))
    return np.asarray(sorted(selected), dtype=np.int64)


def scale_point_sizes(ages: np.ndarray, min_size: float = 24.0, max_size: float = 140.0) -> np.ndarray:
    valid = np.isfinite(ages)
    if not valid.any():
        return np.full(ages.shape, (min_size + max_size) / 2.0, dtype=np.float32)
    values = ages.astype(np.float32, copy=True)
    lo = float(np.nanmin(values[valid]))
    hi = float(np.nanmax(values[valid]))
    if math.isclose(lo, hi):
        sizes = np.full(values.shape, (min_size + max_size) / 2.0, dtype=np.float32)
    else:
        sizes = min_size + ((values - lo) / (hi - lo)) * (max_size - min_size)
    sizes[~valid] = min_size
    return sizes.astype(np.float32)


def scale_categorical_point_sizes(values: np.ndarray | None, min_size: float = 28.0, max_size: float = 128.0) -> np.ndarray:
    if values is None:
        return np.empty((0,), dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(values) & (values >= 0)
    if not valid.any():
        return np.full(values.shape, (min_size + max_size) / 2.0, dtype=np.float32)
    unique = np.unique(values[valid])
    if unique.size == 1:
        sizes = np.full(values.shape, (min_size + max_size) / 2.0, dtype=np.float32)
    else:
        ranks = {float(value): rank for rank, value in enumerate(sorted(unique.tolist()))}
        denom = max(1, unique.size - 1)
        sizes = np.full(values.shape, min_size, dtype=np.float32)
        for value, rank in ranks.items():
            sizes[values == value] = min_size + (rank / denom) * (max_size - min_size)
    sizes[~valid] = min_size
    return sizes.astype(np.float32)


def compute_silhouette(features: np.ndarray, pathology: np.ndarray) -> SilhouetteResult:
    valid = pathology != -1
    valid_count = int(valid.sum())
    valid_labels = pathology[valid]
    class_count = int(np.unique(valid_labels).size) if valid_count > 0 else 0
    if valid_count < 3 or class_count < 2 or class_count >= valid_count:
        return SilhouetteResult(score=float("nan"), valid_count=valid_count, class_count=class_count)

    scaled = StandardScaler().fit_transform(features[valid])
    return SilhouetteResult(
        score=float(silhouette_score(scaled, valid_labels)),
        valid_count=valid_count,
        class_count=class_count,
    )


def _umap_kwargs(n_points: int) -> dict:
    return {
        "n_components": 2,
        "n_neighbors": min(15, max(2, n_points - 1)),
        "min_dist": 0.1,
        "metric": "euclidean",
        "n_jobs": -1,
    }


def _ensure_2d_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32).reshape(len(coords), -1)
    if coords.shape[1] >= 2:
        return coords[:, :2]
    return np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))


def reduce_embeddings(features: np.ndarray, reducer: str, random_state: int) -> np.ndarray:
    if features.shape[0] < 2:
        return np.zeros((features.shape[0], 2), dtype=np.float32)
    scaled = StandardScaler().fit_transform(features)
    reducer = reducer.lower()
    if reducer == "tsne":
        if features.shape[0] < 3:
            return np.zeros((features.shape[0], 2), dtype=np.float32)
        perplexity = min(30.0, max(2.0, (features.shape[0] - 1) / 3.0))
        return (
            TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=random_state)
            .fit_transform(scaled)
            .astype(np.float32)
        )
    if reducer == "umap":
        return umap.UMAP(**_umap_kwargs(features.shape[0])).fit_transform(scaled).astype(np.float32)
    if reducer == "pca":
        n_components = min(2, scaled.shape[0], scaled.shape[1])
        return _ensure_2d_coords(PCA(n_components=n_components, random_state=random_state).fit_transform(scaled))
    raise ValueError(f"Unsupported embedding reducer: {reducer}")


def reduce_embeddings_projecting_reference(
    features: np.ndarray,
    reducer: str,
    random_state: int,
    reference_mask: np.ndarray,
) -> np.ndarray:
    """Fit PCA/UMAP on reference rows and transform all rows; t-SNE is joint."""
    reducer = reducer.lower()
    if reducer == "tsne":
        return reduce_embeddings(features, reducer, random_state)
    if features.shape[0] < 2:
        return np.zeros((features.shape[0], 2), dtype=np.float32)
    reference_mask = np.asarray(reference_mask, dtype=bool)
    n_ref = int(reference_mask.sum())
    if reference_mask.shape[0] != features.shape[0]:
        raise ValueError("reference_mask must have the same length as features.")
    if n_ref < 2 or (reducer == "umap" and n_ref < 3):
        raise ValueError(f"Not enough reference points to project {reducer}: got {n_ref}.")
    scaler = StandardScaler().fit(features[reference_mask])
    scaled_reference = scaler.transform(features[reference_mask])
    scaled_all = scaler.transform(features)
    if reducer == "pca":
        n_components = min(2, scaled_reference.shape[0], scaled_reference.shape[1])
        return _ensure_2d_coords(
            PCA(n_components=n_components, random_state=random_state).fit(scaled_reference).transform(scaled_all)
        )
    if reducer == "umap":
        return umap.UMAP(**_umap_kwargs(n_ref)).fit(scaled_reference).transform(scaled_all).astype(np.float32)
    raise ValueError(f"Unsupported embedding reducer: {reducer}")


def make_projection_figure(
    coords: np.ndarray,
    pathology: np.ndarray,
    ages: np.ndarray,
    sex: np.ndarray,
    title: str,
    style: str = "pathology",
    modality: np.ndarray | None = None,
    modality_names: dict | None = None,
    scanner_id: np.ndarray | None = None,
    scanner_names: dict | None = None,
    field_strength: np.ndarray | None = None,
    field_strength_names: dict | None = None,
):
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "asparagus_matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 5.5), dpi=140)
    markers = {0: "s", 1: "o"}
    sex_names = {0: "male", 1: "female"}
    if style == "control_age":
        scatter_for_colorbar = None
        for sex_id, marker in markers.items():
            mask = sex == sex_id
            if not mask.any():
                continue
            scatter_for_colorbar = ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=42,
                marker=marker,
                c=ages[mask],
                cmap="viridis",
                alpha=0.78,
                edgecolors="white",
                linewidths=0.35,
                label=sex_names[sex_id],
            )
        unknown_sex = ~np.isin(sex, [0, 1])
        if unknown_sex.any():
            scatter_for_colorbar = ax.scatter(
                coords[unknown_sex, 0],
                coords[unknown_sex, 1],
                s=42,
                marker="x",
                c=ages[unknown_sex],
                cmap="viridis",
                alpha=0.78,
                linewidths=0.8,
                label="sex=unknown",
            )
        if scatter_for_colorbar is not None:
            fig.colorbar(scatter_for_colorbar, ax=ax, label="age")
    elif style == "modality":
        # One color per modality over ALL points; color is keyed on the modality id so it is
        # stable across epochs even when some modalities are absent.
        names = modality_names or {}
        labels = modality if modality is not None else np.full((coords.shape[0],), -1)
        palette = plt.get_cmap("tab20")
        for modality_id in sorted(np.unique(labels).tolist()):
            mask = labels == modality_id
            if not mask.any():
                continue
            name = names.get(int(modality_id), str(int(modality_id)))
            color = "#9e9e9e" if int(modality_id) < 0 else palette(int(modality_id) % 20)
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=34,
                marker="o",
                color=color,
                alpha=0.75,
                edgecolors="white",
                linewidths=0.3,
                label=f"{name} (n={int(mask.sum())})",
            )
    elif style == "scanner_acquisition":
        from matplotlib.lines import Line2D

        labels = scanner_id if scanner_id is not None else np.full((coords.shape[0],), -1)
        labels = np.asarray(labels, dtype=np.int64)
        modality_labels = modality if modality is not None else np.full((coords.shape[0],), -1)
        modality_labels = np.asarray(modality_labels, dtype=np.int64)
        field_values = (
            np.asarray(field_strength, dtype=np.float32)
            if field_strength is not None
            else np.full((coords.shape[0],), -1.0, dtype=np.float32)
        )
        sizes = scale_categorical_point_sizes(field_values)
        scanner_name_map = {
            -1: "scanner=unknown",
            0: "Siemens",
            1: "Philips",
            2: "GE",
            3: "Canon/Toshiba",
            4: "Hitachi",
        }
        scanner_name_map.update(scanner_names or {})
        names = modality_names or {}
        field_names = field_strength_names or {}
        palette = plt.get_cmap("tab20")
        modality_markers = {
            -1: "x",
            0: "s",
            1: "o",
            2: "^",
            3: "D",
            4: "v",
            5: "P",
            6: "X",
            7: "*",
            8: "<",
            9: ">",
        }
        scanner_values = sorted(np.unique(labels).tolist())
        modality_values = sorted(np.unique(modality_labels).tolist())
        for scanner_value in scanner_values:
            scanner_mask = labels == scanner_value
            color = "#9e9e9e" if int(scanner_value) < 0 else palette(int(scanner_value) % 20)
            for modality_id in modality_values:
                mask = scanner_mask & (modality_labels == modality_id)
                if not mask.any():
                    continue
                marker = modality_markers.get(int(modality_id), "o")
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=sizes[mask],
                    marker=marker,
                    color=color,
                    alpha=0.72,
                    edgecolors="white",
                    linewidths=0.35,
                )

        scanner_handles = []
        for scanner_value in scanner_values:
            mask = labels == scanner_value
            if not mask.any():
                continue
            color = "#9e9e9e" if int(scanner_value) < 0 else palette(int(scanner_value) % 20)
            scanner_name = scanner_name_map.get(int(scanner_value), f"scanner={int(scanner_value)}")
            scanner_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor=color,
                    markeredgecolor="white",
                    label=f"{scanner_name} (n={int(mask.sum())})",
                    markersize=7,
                )
            )
        modality_handles = []
        for modality_id in modality_values:
            mask = modality_labels == modality_id
            if not mask.any():
                continue
            modality_name = names.get(int(modality_id), str(int(modality_id)))
            modality_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=modality_markers.get(int(modality_id), "o"),
                    linestyle="None",
                    color="#333333",
                    label=f"{modality_name} (n={int(mask.sum())})",
                    markersize=7,
                )
            )
        if scanner_handles:
            first_legend = ax.legend(
                handles=scanner_handles,
                title="scanner",
                fontsize=6,
                title_fontsize=7,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                frameon=True,
            )
            ax.add_artist(first_legend)
        legend_handles = modality_handles
        valid_field = np.isfinite(field_values) & (field_values >= 0)
        if valid_field.any():
            field_sizes = scale_categorical_point_sizes(field_values)
            for value in sorted(np.unique(field_values[valid_field]).tolist()):
                name = field_names.get(int(value), f"field_strength={int(value)}")
                marker_size = float(np.sqrt(np.mean(field_sizes[field_values == value])))
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="None",
                        color="#555555",
                        markerfacecolor="#bbbbbb",
                        alpha=0.8,
                        label=name,
                        markersize=max(4.5, marker_size),
                    )
                )
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                title="modality / field",
                fontsize=6,
                title_fontsize=7,
                loc="lower left",
                bbox_to_anchor=(1.01, 0.0),
                frameon=True,
            )
            fig.subplots_adjust(right=0.74)
    else:
        sizes = scale_point_sizes(ages)
        color_map = {
            -1: "#9e9e9e",
            0: "#1f77b4",
            1: "#d62728",
            2: "#2ca02c",
            3: "#9467bd",
            4: "#ff7f0e",
            5: "#17becf",
        }
        for class_id in sorted(np.unique(pathology).tolist()):
            for sex_id, marker in markers.items():
                mask = (pathology == class_id) & (sex == sex_id)
                if not mask.any():
                    continue
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=sizes[mask],
                    marker=marker,
                    c=color_map.get(int(class_id), "#7f7f7f"),
                    alpha=0.72,
                    edgecolors="white",
                    linewidths=0.35,
                    label=f"{PATHOLOGY_CLASS_NAMES.get(int(class_id), str(class_id))} / {sex_names[sex_id]}",
                )
            unknown_sex = ~np.isin(sex, [0, 1])
            mask = (pathology == class_id) & unknown_sex
            if mask.any():
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=sizes[mask],
                    marker="x",
                    c=color_map.get(int(class_id), "#7f7f7f"),
                    alpha=0.72,
                    linewidths=0.8,
                    label=f"{PATHOLOGY_CLASS_NAMES.get(int(class_id), str(class_id))} / sex=unknown",
                )

    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    if style != "scanner_acquisition":
        ax.legend(fontsize=6, loc="best", frameon=True, markerscale=0.75)
        fig.tight_layout()
    else:
        fig.tight_layout(rect=(0.0, 0.0, 0.74, 1.0))
    return fig


def make_projection_rows(
    coords: np.ndarray,
    pathology: np.ndarray,
    ages: np.ndarray,
    sex: np.ndarray,
    modality: np.ndarray | None = None,
    scanner_id: np.ndarray | None = None,
    dataset_id: np.ndarray | None = None,
    field_strength: np.ndarray | None = None,
) -> list[list]:
    rows = []
    if modality is None:
        modality = np.full((coords.shape[0],), -1, dtype=np.int64)
    if scanner_id is None:
        scanner_id = np.full((coords.shape[0],), -1, dtype=np.int64)
    if dataset_id is None:
        dataset_id = np.full((coords.shape[0],), -1, dtype=np.int64)
    if field_strength is None:
        field_strength = np.full((coords.shape[0],), -1, dtype=np.int64)
    for idx in range(coords.shape[0]):
        class_id = int(pathology[idx])
        rows.append(
            [
                int(idx),
                float(coords[idx, 0]),
                float(coords[idx, 1]),
                class_id,
                PATHOLOGY_CLASS_NAMES.get(class_id, str(class_id)),
                float(ages[idx]) if np.isfinite(ages[idx]) else None,
                int(sex[idx]),
                int(modality[idx]),
                int(scanner_id[idx]),
                int(dataset_id[idx]),
                int(field_strength[idx]),
            ]
        )
    return rows


PROJECTION_ROW_COLUMNS = [
    "index",
    "x",
    "y",
    "pathology",
    "pathology_name",
    "age",
    "sex",
    "modality_id",
    "scanner_id",
    "dataset_id",
    "field_strength_id",
]
