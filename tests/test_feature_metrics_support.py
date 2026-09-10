import math
import torch
import torch.nn as nn
from asparagus.functional.metrics import features as feature_metrics


def _assert_all_finite(metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        assert math.isfinite(float(value)), key


def test_singleton_embedding_and_support_metadata_are_finite() -> None:
    encoder_features = torch.tensor([[3.0, 4.0]], dtype=torch.float32)

    metrics = feature_metrics.compute_val(encoder_features, nn.Identity())

    _assert_all_finite(metrics)
    assert metrics["embedding_sample_count"] == 1
    assert metrics["embedding_norm_std"] == 0.0
    assert metrics["feature_cov_sample_count"] == 1
    assert metrics["feature_cov_available"] == 0
    assert metrics["whitening_sample_count"] == 1
    assert metrics["whitening_available"] == 0

    for key in (
        "feature_cov_trace",
        "feature_cov_nonzero_variance",
        "feature_cov_condition_number",
        "feature_cov_max_eigenval",
        "feature_cov_min_eigenval",
        "feature_effective_rank",
        "whitening_offdiag_mean",
        "whitening_offdiag_std",
        "whitening_offdiag_max",
        "decorrelation_score",
        "whitening_diag_mean",
        "whitening_diag_std",
    ):
        assert key not in metrics


def test_multisample_embedding_keeps_regular_diagnostics_finite() -> None:
    encoder_features = torch.tensor(
        [[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [2.0, 1.0, 0.5]],
        dtype=torch.float32,
    )

    metrics = feature_metrics.compute_val(encoder_features, nn.Identity())

    _assert_all_finite(metrics)
    assert metrics["embedding_sample_count"] == 3
    assert metrics["feature_cov_sample_count"] == 3
    assert metrics["feature_cov_available"] == 1
    assert metrics["whitening_sample_count"] == 3
    assert metrics["whitening_available"] == 1
    assert "feature_cov_trace" in metrics
    assert "feature_cov_condition_number" in metrics or metrics["feature_cov_nonzero_variance"] == 0
    assert "whitening_offdiag_mean" in metrics
    assert "whitening_diag_mean" in metrics
    assert math.isclose(
        metrics["whitening_diag_mean"],
        1.0,
        rel_tol=1e-5,
        abs_tol=1e-5,
    )


def test_spatial_features_preserve_channel_sample_semantics() -> None:
    encoder_features = torch.arange(1 * 2 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 2, 3, 4)
    expected_channel_samples = encoder_features.permute(0, 2, 3, 4, 1).reshape(-1, 2)

    flattened = feature_metrics._to_channel_samples(encoder_features)
    metrics = feature_metrics.compute_val(encoder_features, nn.Identity())

    assert torch.equal(flattened, expected_channel_samples)
    _assert_all_finite(metrics)
    assert metrics["embedding_sample_count"] == expected_channel_samples.shape[0]
    assert metrics["feature_cov_sample_count"] == expected_channel_samples.shape[0]
    assert metrics["whitening_sample_count"] == expected_channel_samples.shape[0]
    assert metrics["feature_cov_available"] == 1
    assert metrics["whitening_available"] == 1


def test_zero_variance_features_remain_finite_and_explicit() -> None:
    encoder_features = torch.ones(2, 3, dtype=torch.float32)

    metrics = feature_metrics.compute_val(encoder_features, nn.Identity())

    _assert_all_finite(metrics)
    assert metrics["collapse_n_zero_var_dims"] == 3
    assert metrics["collapse_pct_zero_var_dims"] == 100.0
    assert metrics["collapse_var_mean"] == 0.0
    assert metrics["collapse_effective_dims"] == 0.0
    assert metrics["feature_cov_available"] == 1
    assert metrics["feature_cov_nonzero_variance"] == 0
    assert metrics["feature_cov_trace"] == 0.0
    assert "feature_cov_condition_number" not in metrics
    assert "feature_effective_rank" not in metrics
    assert metrics["whitening_available"] == 1
