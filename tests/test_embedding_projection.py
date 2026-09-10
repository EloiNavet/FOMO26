import math
import numpy as np
import pytest
import torch
from asparagus.functional.metrics import embedding_projection
from asparagus.modules.datasets.PretrainDataset import PretrainDataset
from asparagus.modules.lightning_modules.self_supervised import SelfSupervisedModule


def test_embedding_projection_silhouette_excludes_unknown_pathology():
    features = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [4.0, 4.0],
            [4.1, 4.0],
            [20.0, 20.0],
        ],
        dtype=np.float32,
    )
    pathology = np.array([0, 0, 3, 3, -1], dtype=np.int64)

    result = embedding_projection.compute_silhouette(features, pathology)

    assert result.valid_count == 4
    assert result.class_count == 2
    assert not math.isnan(result.score)
    assert result.score > 0.0


def test_embedding_projection_silhouette_nan_when_all_unknown():
    features = np.eye(4, dtype=np.float32)
    pathology = np.full((4,), -1, dtype=np.int64)

    result = embedding_projection.compute_silhouette(features, pathology)

    assert math.isnan(result.score)
    assert result.valid_count == 0
    assert result.class_count == 0


def test_embedding_projection_silhouette_nan_when_single_valid_class():
    features = np.eye(4, dtype=np.float32)
    pathology = np.array([1, 1, 1, -1], dtype=np.int64)

    result = embedding_projection.compute_silhouette(features, pathology)

    assert math.isnan(result.score)
    assert result.valid_count == 3
    assert result.class_count == 1


def test_pathology_retrieval_reports_same_macro_and_different_site_hits():
    features = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
        ],
        dtype=np.float32,
    )
    pathology = np.array([0, 0, 3, 3], dtype=np.int64)
    fine_pathology = np.array([10, 10, 30, 31], dtype=np.int64)
    dataset_id = np.array([1, 2, 1, 2], dtype=np.int64)
    modality_id = np.array([0, 0, 0, 2], dtype=np.int64)

    metrics = embedding_projection.compute_pathology_retrieval_metrics(
        features,
        pathology,
        fine_pathology=fine_pathology,
        dataset_id=dataset_id,
        modality_id=modality_id,
    )

    assert metrics["same_macro/top1"] == pytest.approx(1.0)
    assert metrics["same_fine/top1"] == pytest.approx(1.0)
    assert metrics["same_macro_different_site/top1"] == pytest.approx(1.0)
    assert metrics["same_macro_different_modality/top1"] == pytest.approx(1.0)


def test_deterministic_limit_indices_are_stable_and_bounded():
    first = embedding_projection.deterministic_limit_indices(100, 10)
    second = embedding_projection.deterministic_limit_indices(100, 10)

    assert np.array_equal(first, second)
    assert first.shape == (10,)
    assert first[0] == 0
    assert first[-1] == 99


def test_embedding_projection_knn_diagnostic_reports_imbalance_metrics():
    features = np.array([[0.0], [0.1], [0.2], [4.0], [4.1], [4.2]], dtype=np.float32)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)

    metrics = embedding_projection.compute_knn_classification_probe(features, labels, n_neighbors=1)

    for key in ("mcc", "balanced_accuracy", "macro_f1", "macro_auprc", "brier", "ece"):
        assert key in metrics
    assert metrics["balanced_accuracy"] == 1.0
    assert "confusion/true_0_pred_0" in metrics


def test_embedding_projection_knn_age_reports_mae():
    features = np.array([[0.0], [0.1], [2.0], [2.1]], dtype=np.float32)
    ages = np.array([20.0, 21.0, 60.0, 61.0], dtype=np.float32)

    mae = embedding_projection.compute_knn_age_mae(features, ages, n_neighbors=1)

    assert mae == pytest.approx(1.0, abs=1e-5)


def test_reference_knn_probe_uses_train_references_for_validation_queries():
    reference_features = np.array([[0.0], [0.1], [4.0], [4.1]], dtype=np.float32)
    reference_labels = np.array([0, 0, 1, 1], dtype=np.int64)
    query_features = np.array([[0.05], [4.05]], dtype=np.float32)
    query_labels = np.array([0, 1], dtype=np.int64)

    metrics = embedding_projection.compute_reference_knn_classification_probe(
        reference_features, reference_labels, query_features, query_labels, n_neighbors=1
    )

    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["reference_count"] == 4.0
    assert metrics["query_count"] == 2.0
    assert not any(key.startswith("confusion/") for key in metrics)


def test_reference_knn_age_reports_validation_mae_from_training_references():
    reference_features = np.array([[0.0], [2.0]], dtype=np.float32)
    reference_ages = np.array([20.0, 60.0], dtype=np.float32)
    query_features = np.array([[0.1], [2.1]], dtype=np.float32)
    query_ages = np.array([21.0, 59.0], dtype=np.float32)

    mae = embedding_projection.compute_reference_knn_age_mae(
        reference_features, reference_ages, query_features, query_ages, n_neighbors=1
    )

    assert mae == pytest.approx(1.0, abs=1e-5)


def test_embedding_projection_supports_pca_and_reference_projection():
    features = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [4.0, 1.0, 0.0], [4.2, 1.0, 0.0]], dtype=np.float32)
    reference_mask = np.array([True, True, False, False])

    pca_coords = embedding_projection.reduce_embeddings(features, "pca", random_state=0)
    projected_coords = embedding_projection.reduce_embeddings_projecting_reference(
        features,
        "pca",
        random_state=0,
        reference_mask=reference_mask,
    )

    assert pca_coords.shape == (4, 2)
    assert projected_coords.shape == (4, 2)
    assert np.isfinite(projected_coords).all()


def test_embedding_projection_limit_preserves_reference_points():
    reference_mask = np.zeros((20,), dtype=bool)
    reference_mask[[1, 2, 3, 4]] = True

    indices = embedding_projection.deterministic_limit_indices(
        20,
        max_points=5,
        reference_mask=reference_mask,
        min_reference_points=3,
    )

    assert indices.shape == (5,)
    assert int(reference_mask[indices].sum()) >= 3


def test_embedding_projection_reference_projection_rejects_insufficient_reference():
    features = np.eye(4, dtype=np.float32)
    reference_mask = np.array([True, False, False, False])

    with pytest.raises(ValueError, match="Not enough reference points"):
        embedding_projection.reduce_embeddings_projecting_reference(
            features,
            "pca",
            random_state=0,
            reference_mask=reference_mask,
        )


def test_projection_figure_and_rows_support_control_age_style_and_modality():
    coords = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]], dtype=np.float32)
    pathology = np.array([0, 0, 0], dtype=np.int64)
    ages = np.array([20.0, 40.0, 60.0], dtype=np.float32)
    sex = np.array([0, 1, 0], dtype=np.int64)
    modality = np.array([0, 0, 4], dtype=np.int64)

    fig = embedding_projection.make_projection_figure(coords, pathology, ages, sex, "controls", style="control_age")
    rows = embedding_projection.make_projection_rows(coords, pathology, ages, sex, modality)

    assert len(fig.axes) >= 2
    modality_col = embedding_projection.PROJECTION_ROW_COLUMNS.index("modality_id")
    assert rows[0][modality_col] == 0
    assert rows[2][modality_col] == 4


def test_projection_figure_modality_style_colors_by_modality():
    rng = np.random.RandomState(0)
    coords = rng.randn(12, 2).astype(np.float32)
    pathology = np.full(12, -1, dtype=np.int64)
    ages = np.full(12, np.nan, dtype=np.float32)
    sex = np.full(12, -1, dtype=np.int64)
    modality = np.array([0, 1, 2, 3] * 3, dtype=np.int64)
    names = {i: PretrainDataset.modality_name(i) for i in (0, 1, 2, 3)}

    fig = embedding_projection.make_projection_figure(
        coords,
        pathology,
        ages,
        sex,
        "all modalities",
        style="modality",
        modality=modality,
        modality_names=names,
    )
    legend = fig.axes[0].get_legend()
    # one legend entry per present modality, labeled with the canonical modality name.
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert len(labels) == 4
    assert any(name in lbl for name in names.values() for lbl in labels)


def test_projection_figure_scanner_acquisition_style_encodes_scanner_modality_and_field_strength():
    rng = np.random.RandomState(1)
    coords = rng.randn(12, 2).astype(np.float32)
    pathology = np.full(12, -1, dtype=np.int64)
    ages = np.full(12, np.nan, dtype=np.float32)
    sex = np.full(12, -1, dtype=np.int64)
    modality = np.array([0, 1, 2] * 4, dtype=np.int64)
    scanner_id = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, -1, -1, -1], dtype=np.int64)
    field_strength = np.array([0, 1, 2] * 4, dtype=np.int64)

    fig = embedding_projection.make_projection_figure(
        coords,
        pathology,
        ages,
        sex,
        "scanner acquisition",
        style="scanner_acquisition",
        modality=modality,
        modality_names={0: "T1w", 1: "T2w", 2: "FLAIR"},
        scanner_id=scanner_id,
        field_strength=field_strength,
        field_strength_names={0: "1.5t", 1: "3t", 2: "7t"},
    )
    from matplotlib.legend import Legend

    labels = [text.get_text() for legend in fig.findobj(Legend) for text in legend.get_texts()]

    assert any("Siemens" in label for label in labels)
    assert any("T1w" in label for label in labels)
    assert any("3t" in label for label in labels)


def test_projection_rows_include_scanner_and_dataset_columns_for_leakage_coloring():
    coords = np.array([[0.0, 0.0], [1.0, 0.5]], dtype=np.float32)
    pathology = np.array([0, 3], dtype=np.int64)
    ages = np.array([20.0, 60.0], dtype=np.float32)
    sex = np.array([0, 1], dtype=np.int64)
    modality = np.array([0, 0], dtype=np.int64)
    scanner_id = np.array([7, 9], dtype=np.int64)
    dataset_id = np.array([101, 202], dtype=np.int64)
    field_strength = np.array([0, 1], dtype=np.int64)

    columns = embedding_projection.PROJECTION_ROW_COLUMNS
    rows = embedding_projection.make_projection_rows(
        coords, pathology, ages, sex, modality, scanner_id, dataset_id, field_strength
    )

    assert "scanner_id" in columns and "dataset_id" in columns
    assert "field_strength_id" in columns
    assert len(rows[0]) == len(columns)
    assert rows[0][columns.index("scanner_id")] == 7
    assert rows[1][columns.index("dataset_id")] == 202
    assert rows[1][columns.index("field_strength_id")] == 1


def test_stable_int_hash_fits_in_float64_mantissa():
    for value in ("pt033_wand", "Major depressive disorder", "control", "ds004889"):
        h = PretrainDataset._stable_int_hash(value)
        assert 0 <= h < 2**53
        assert int(float(h)) == h  # round-trips exactly through float64


def test_demographic_null_baselines_are_fit_on_training_references():
    reference_ages = np.array([20.0, 22.0, 80.0], dtype=np.float32)
    query_ages = np.array([21.0, 23.0], dtype=np.float32)
    reference_sex = np.array([1, 1, 0], dtype=np.int64)
    query_sex = np.array([0, 1], dtype=np.int64)

    age_mae = embedding_projection.compute_reference_median_age_mae(reference_ages, query_ages)
    sex_metrics = embedding_projection.compute_reference_majority_classification_baseline(reference_sex, query_sex)

    assert age_mae == pytest.approx(1.0)
    assert sex_metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert sex_metrics["reference_count"] == 3.0
    assert sex_metrics["query_count"] == 2.0


def test_ssl_metric_format_preserves_wandb_style_paths():
    module = SelfSupervisedModule(model=torch.nn.Identity(), learning_rate=1e-3)
    metrics = {
        "loss": {
            "total": 1.0,
            "mse/raw": 0.5,
        },
        "features": {
            "embedding_norm_mean": 2.0,
            "embedding_norm_std": 0.1,
        },
        "stability": {
            "nan_loss": 0.0,
            "nan_activations": 0.0,
            "inf_loss": 0.0,
            "inf_activations": 0.0,
            "gradient_clipping_events": 1.0,
        },
    }

    formatted = module._format_metrics("train", metrics)

    assert "train/loss/total" in formatted
    assert "train/loss/mse/raw" in formatted
    assert "train/features/embedding_norm_mean" in formatted
    assert "train/features/embedding_norm_std" in formatted
    assert "train/stability/nan_loss" in formatted
    assert "train/stability/nan_activations" in formatted
    assert "train/stability/inf_loss" in formatted
    assert "train/stability/inf_activations" in formatted
    assert "train/stability/gradient_clipping_events" in formatted
