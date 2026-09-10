"""
Statistical distribution and alignment metrics for SSL pretraining.
"""

import torch
from typing import Dict


# Frequency-based compute functions
def compute(
    x: torch.Tensor,
    pred: torch.Tensor,
    y: torch.Tensor,
    encoder_features: torch.Tensor,
    positive_features: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Metrics computed every validation step."""
    metrics = compute_input_statistics(x, prefix="input")
    metrics |= compute_input_statistics(pred, prefix="pred")
    metrics |= compute_intensity_distribution_match(pred, y)
    metrics |= compute_alignment_uniformity(encoder_features, positive_features=positive_features)
    return metrics


def compute_input_statistics(x: torch.Tensor, prefix: str = "input") -> Dict[str, torch.Tensor]:
    """
    Input intensity distribution monitoring for data drift detection.
    Significant shifts indicate preprocessing issues or domain shift.
    """
    x = x.float()
    return {
        f"{prefix}_mean": x.mean(),
        f"{prefix}_std": x.std(),
        f"{prefix}_min": x.min(),
        f"{prefix}_max": x.max(),
        f"{prefix}_p10": torch.quantile(x.flatten()[::10], 0.1),
        f"{prefix}_p50": torch.quantile(x.flatten()[::10], 0.5),
        f"{prefix}_p90": torch.quantile(x.flatten()[::10], 0.9),
    }


def compute_intensity_distribution_match(pred: torch.Tensor, target: torch.Tensor, n_bins: int = 50) -> Dict[str, float]:
    """
    KL divergence and Wasserstein distance between intensity histograms.
    KL < 0.1 indicates good distribution matching; >1.0 suggests mode collapse.
    """
    metrics = {}

    # Flatten tensors and convert to float32 for histc (required for BFloat16 compatibility)
    pred_flat = pred.flatten().float()
    target_flat = target.flatten().float()

    # Compute histograms
    hist_range = (min(pred_flat.min().item(), target_flat.min().item()), max(pred_flat.max().item(), target_flat.max().item()))
    pred_hist = torch.histc(pred_flat, bins=n_bins, min=hist_range[0], max=hist_range[1])
    target_hist = torch.histc(target_flat, bins=n_bins, min=hist_range[0], max=hist_range[1])

    # Normalize to probabilities
    pred_hist = pred_hist / pred_hist.sum()
    target_hist = target_hist / target_hist.sum()

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    pred_hist = pred_hist.clamp(min=eps)
    target_hist = target_hist.clamp(min=eps)

    # Compute KL divergence: KL(P||Q) = sum(P * log(P/Q))
    kl_div = (target_hist * torch.log(target_hist / pred_hist)).sum().item()

    # Also compute reverse KL
    kl_div_reverse = (pred_hist * torch.log(pred_hist / target_hist)).sum().item()

    # Symmetric KL (Jensen-Shannon divergence related)
    kl_symmetric = (kl_div + kl_div_reverse) / 2

    metrics["intensity_kl_divergence"] = kl_div
    metrics["intensity_kl_symmetric"] = kl_symmetric

    # Also compute histogram correlation
    pred_mean = pred_hist.mean()
    target_mean = target_hist.mean()

    cov = ((pred_hist - pred_mean) * (target_hist - target_mean)).mean()
    std_pred = pred_hist.std()
    std_target = target_hist.std()

    if std_pred > 0 and std_target > 0:
        correlation = cov / (std_pred * std_target)
        metrics["intensity_histogram_correlation"] = correlation.item()
    else:
        metrics["intensity_histogram_correlation"] = 0.0

    return metrics


def compute_alignment_uniformity(
    features: torch.Tensor,
    positive_features: torch.Tensor | None = None,
    temperature: float = 2.0,
    sample_size: int = 1000,
) -> Dict[str, float]:
    """
    Wang & Isola metrics for contrastive representation quality.
    Alignment < 1.0 good; Uniformity ~ -2 to -3 optimal for hypersphere coverage.
    """
    metrics = {}

    if features is None or features.numel() == 0:
        return metrics

    # Flatten spatial dimensions if present
    if features.dim() > 2:
        B = features.shape[0]
        features = features.reshape(B, -1).float()  # (B, D)
    else:
        features = features.float()

    # Sample if batch is too large (for efficiency)
    if features.shape[0] > sample_size:
        indices = torch.randperm(features.shape[0])[:sample_size]
        features = features[indices]
        if positive_features is not None:
            positive_features = positive_features[indices]

    # L2 normalize features
    features_normalized = torch.nn.functional.normalize(features, p=2, dim=1)

    if positive_features is not None and positive_features.shape[0] == features.shape[0]:
        if positive_features.dim() > 2:
            positive_features = positive_features.reshape(positive_features.shape[0], -1).float()
        else:
            positive_features = positive_features.float()
        positive_features = torch.nn.functional.normalize(positive_features, p=2, dim=1)
        metrics["alignment_loss"] = (features_normalized - positive_features).norm(dim=1).pow(2).mean().item()

    # Uniformity: log of average pairwise Gaussian potential
    # This measures how uniformly distributed features are
    if features.shape[0] > 1:
        # Compute pairwise distances
        sq_pdist = torch.pdist(features_normalized, p=2).pow(2)

        # Gaussian potential with temperature
        uniformity = sq_pdist.mul(-temperature).exp().mean().log()
        metrics["uniformity_loss"] = uniformity.item()

        # Additional uniformity statistics
        metrics["uniformity_mean_dist"] = sq_pdist.mean().item()
        # Only compute std if we have more than 1 distance
        if sq_pdist.numel() > 1:
            metrics["uniformity_std_dist"] = sq_pdist.std().item()
        else:
            metrics["uniformity_std_dist"] = 0.0
        metrics["uniformity_min_dist"] = sq_pdist.min().item()
        metrics["uniformity_max_dist"] = sq_pdist.max().item()

    # Compute cosine similarity statistics
    if features.shape[0] > 1:
        # Compute cosine similarity matrix
        cos_sim = torch.mm(features_normalized, features_normalized.t())

        # Get upper triangular part (excluding diagonal)
        mask = torch.triu(torch.ones_like(cos_sim, dtype=torch.bool), diagonal=1)
        cos_sim_values = cos_sim[mask]

        if cos_sim_values.numel() > 0:
            metrics["cosine_sim_mean"] = cos_sim_values.mean().item()
            # Only compute std if we have more than 1 similarity value
            if cos_sim_values.numel() > 1:
                metrics["cosine_sim_std"] = cos_sim_values.std().item()
            else:
                metrics["cosine_sim_std"] = 0.0
            metrics["cosine_sim_max"] = cos_sim_values.max().item()
            metrics["cosine_sim_min"] = cos_sim_values.min().item()

    return metrics


def compute_cross_modal_retrieval(
    features: torch.Tensor,
    subject_session_ids: torch.Tensor,
    modality_ids: torch.Tensor,
    registered_subset: torch.Tensor | None = None,
    cross_modal_candidates_only: bool = False,
    subject_ids: torch.Tensor | None = None,
) -> Dict[str, float]:
    """Evaluate same-session cross-modal retrieval with same-subject longitudinal scans neutralized."""
    if features is None or features.shape[0] < 2:
        return {"eligible_anchor_count": 0.0, "positive_pair_count": 0.0}
    features = torch.nn.functional.normalize(features.float(), p=2, dim=1)
    subject_session_ids = subject_session_ids.view(-1)
    modality_ids = modality_ids.view(-1)
    if subject_ids is None:
        subject_ids = subject_session_ids
    else:
        subject_ids = subject_ids.view(-1)
    valid = modality_ids >= 0
    same_session = subject_session_ids[:, None] == subject_session_ids[None, :]
    same_subject = subject_ids[:, None] == subject_ids[None, :]
    same_modality = modality_ids[:, None] == modality_ids[None, :]
    identity = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    positives = same_session & ~same_modality & valid[:, None] & valid[None, :]
    same_subject_cross_session = same_subject & ~same_session & valid[:, None] & valid[None, :]
    candidates = valid[:, None] & valid[None, :] & ~identity & ~(same_session & same_modality)
    candidates &= ~same_subject_cross_session
    if cross_modal_candidates_only:
        candidates &= ~same_modality
    valid_anchors = positives.any(dim=1)
    metrics = {
        "eligible_anchor_count": float(valid_anchors.sum().item()),
        "positive_pair_count": float(positives.sum().item()),
        "same_subject_cross_session_pair_count": float(same_subject_cross_session.sum().item()),
        "same_subject_cross_session_excluded_count": float(same_subject_cross_session.sum().item()),
        "subject_aware_negatives": float(subject_ids is not subject_session_ids),
    }
    if registered_subset is not None:
        registered_subset = registered_subset.to(device=features.device, dtype=torch.bool).view(-1)
        metrics["registered_eligible_anchor_count"] = float((valid_anchors & registered_subset).sum().item())
        metrics["nonregistered_eligible_anchor_count"] = float((valid_anchors & ~registered_subset).sum().item())
    if not valid_anchors.any():
        return metrics
    similarities = torch.matmul(features, features.T)
    metrics["alignment_cosine"] = float(similarities[positives].mean().item())
    if same_subject_cross_session.any():
        longitudinal_similarities = similarities[same_subject_cross_session]
        metrics["same_subject_cross_session_alignment_cosine"] = float(longitudinal_similarities.mean().item())
        longitudinal_candidates = valid[:, None] & valid[None, :] & ~identity & ~same_session
        longitudinal_valid_anchors = same_subject_cross_session.any(dim=1)
        metrics["same_subject_cross_session_eligible_anchor_count"] = float(longitudinal_valid_anchors.sum().item())
        longitudinal_nearest = similarities.masked_fill(~longitudinal_candidates, float("-inf")).argmax(dim=1)
        longitudinal_rows = torch.arange(features.shape[0], device=features.device)[longitudinal_valid_anchors]
        longitudinal_nearest_valid = longitudinal_nearest[longitudinal_valid_anchors]
        metrics["same_subject_cross_session_retrieval_at_1"] = float(
            same_subject_cross_session[longitudinal_rows, longitudinal_nearest_valid].float().mean().item()
        )
        longitudinal_anchor_similarities = similarities[longitudinal_valid_anchors].masked_fill(
            ~longitudinal_candidates[longitudinal_valid_anchors],
            float("-inf"),
        )
        longitudinal_positives = same_subject_cross_session[longitudinal_valid_anchors]
        best_longitudinal_positive = (
            longitudinal_anchor_similarities.masked_fill(
                ~longitudinal_positives,
                float("-inf"),
            )
            .max(dim=1)
            .values
        )
        longitudinal_ranks = (longitudinal_anchor_similarities >= best_longitudinal_positive[:, None]).sum(dim=1).float()
        metrics["same_subject_cross_session_retrieval_at_5"] = float((longitudinal_ranks <= 5).float().mean().item())
        different_subject = ~same_subject & valid[:, None] & valid[None, :] & ~identity
        if different_subject.any():
            different_subject_similarity = similarities[different_subject]
            metrics["different_subject_similarity_mean"] = float(different_subject_similarity.mean().item())
            metrics["same_subject_cross_session_vs_different_subject_margin_mean"] = float(
                (longitudinal_similarities.mean() - different_subject_similarity.mean()).item()
            )
    nearest = similarities.masked_fill(~candidates, float("-inf")).argmax(dim=1)
    rows = torch.arange(features.shape[0], device=features.device)
    valid_rows = rows[valid_anchors]
    nearest_valid = nearest[valid_anchors]
    metrics["candidate_pair_count"] = float(candidates[valid_anchors].sum().item())
    if metrics["positive_pair_count"] > 0.0:
        metrics["candidate_positive_ratio"] = metrics["candidate_pair_count"] / metrics["positive_pair_count"]
    top1_positive = positives[valid_rows, nearest_valid]
    metrics["retrieval_at_1"] = float(top1_positive.float().mean().item())
    metrics["top1_same_modality_fraction"] = float(same_modality[valid_rows, nearest_valid].float().mean().item())
    metrics["top1_cross_modal_fraction"] = float((~same_modality[valid_rows, nearest_valid]).float().mean().item())
    metrics["top1_same_session_fraction"] = float(same_session[valid_rows, nearest_valid].float().mean().item())
    anchor_similarities = similarities[valid_anchors].masked_fill(~candidates[valid_anchors], float("-inf"))
    anchor_positives = positives[valid_anchors]
    best_positive = anchor_similarities.masked_fill(~anchor_positives, float("-inf")).max(dim=1).values
    top_candidate = anchor_similarities.max(dim=1).values
    metrics["best_positive_similarity_mean"] = float(best_positive.mean().item())
    metrics["top_candidate_similarity_mean"] = float(top_candidate.mean().item())
    metrics["positive_top_candidate_margin_mean"] = float((best_positive - top_candidate).mean().item())
    negative_candidates = candidates[valid_anchors] & ~anchor_positives
    best_negative = anchor_similarities.masked_fill(~negative_candidates, float("-inf")).max(dim=1).values
    negative_available = torch.isfinite(best_negative)
    if negative_available.any():
        margin = best_positive[negative_available] - best_negative[negative_available]
        metrics["best_negative_similarity_mean"] = float(best_negative[negative_available].mean().item())
        metrics["positive_best_negative_margin_mean"] = float(margin.mean().item())
        metrics["positive_beats_best_negative_fraction"] = float((margin > 0).float().mean().item())
    positive_ranks = (anchor_similarities >= best_positive[:, None]).sum(dim=1).float()
    metrics["retrieval_at_5"] = float((positive_ranks <= 5).float().mean().item())
    metrics["retrieval_at_10"] = float((positive_ranks <= 10).float().mean().item())
    metrics["positive_rank_mean"] = float(positive_ranks.mean().item())
    metrics["positive_rank_median"] = float(positive_ranks.median().item())
    metrics["mean_reciprocal_rank"] = float((1.0 / positive_ranks.clamp_min(1.0)).mean().item())
    return metrics
