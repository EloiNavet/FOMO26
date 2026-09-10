"""Shared helpers for encoder representations used by SSL heads and probes."""

import math
import re
import torch
import torch.nn.functional as F
from collections.abc import Mapping

_TRAILING_INT_RE = re.compile(r"(\d+)$")


def _feature_order_index(key) -> int:
    if isinstance(key, int):
        return key
    if isinstance(key, str):
        match = _TRAILING_INT_RE.search(key)
        if match is not None:
            return int(match.group(1))
    raise ValueError(
        "Feature dictionaries must use integer keys or string keys ending in an integer (for example h1, stage2, encoder_3)."
    )


def ordered_feature_sequence(features) -> tuple[torch.Tensor, ...]:
    """Return encoder features in stable shallow-to-deep order.

    Lists and tuples are already ordered and are returned unchanged as tuples. Dicts are sorted by
    integer key or by the trailing integer in string keys, so h1/stage2/encoder_3 style names are
    deterministic. A single tensor is treated as a one-feature sequence for legacy callers.
    """
    if features is None:
        raise ValueError("Expected encoder features, got None.")
    if torch.is_tensor(features):
        return (features,)
    if isinstance(features, (list, tuple)):
        if len(features) == 0:
            raise ValueError("Expected at least one encoder feature, got an empty sequence.")
        return tuple(features)
    if isinstance(features, Mapping):
        if not features:
            raise ValueError("Expected at least one encoder feature, got an empty dict.")
        indexed = []
        seen_indices = set()
        for key, value in features.items():
            index = _feature_order_index(key)
            if index in seen_indices:
                raise ValueError(f"Feature dictionary has duplicate shallow-to-deep index {index}.")
            seen_indices.add(index)
            indexed.append((index, value))
        return tuple(value for _, value in sorted(indexed, key=lambda item: item[0]))
    raise TypeError(f"Expected tensor, list, tuple, or dict of encoder features, got {type(features).__name__}.")


def build_h_global(features) -> torch.Tensor:
    """Concatenate global-average-pooled 3D encoder feature maps.

    The operation preserves gradients, dtype and device. It intentionally does not detach, cast or
    move tensors. Every feature must be a 5D tensor [B, C, D, H, W] and all batch sizes must match.
    """
    ordered = ordered_feature_sequence(features)
    pooled = []
    batch_size = None
    for idx, feat in enumerate(ordered):
        if not torch.is_tensor(feat):
            raise TypeError(f"Encoder feature {idx} is {type(feat).__name__}, expected torch.Tensor.")
        if feat.ndim != 5:
            raise ValueError(f"Encoder feature {idx} must be 5D [B, C, D, H, W], got shape {tuple(feat.shape)}.")
        if batch_size is None:
            batch_size = int(feat.shape[0])
        elif int(feat.shape[0]) != batch_size:
            raise ValueError(
                f"All encoder features must share batch size {batch_size}; feature {idx} has batch size {feat.shape[0]}."
            )
        pooled.append(F.adaptive_avg_pool3d(feat, output_size=1).flatten(1))
    return torch.cat(pooled, dim=1)


def build_transfer_representations(features, dense_level: int = -2) -> dict[str, torch.Tensor | tuple]:
    """Build the task-facing representation contract from encoder feature grids.

    ``h_dense`` is the requested dense stage, ``h_coarse`` is the bottleneck grid,
    and ``h_global`` is the exact global-average-pooled bottleneck consumed by the
    ResEnc classification/regression head before dropout and its final linear layer.
    The complete shallow-to-deep feature tuple is retained as ``encoder_features``
    because segmentation decoders consume every grid.
    """
    ordered = ordered_feature_sequence(features)
    level = int(dense_level)
    index = level if level >= 0 else len(ordered) + level
    if not 0 <= index < len(ordered):
        raise ValueError(f"dense_level={level} is out of range for {len(ordered)} encoder features.")
    for feature_index, feature in enumerate(ordered):
        if not torch.is_tensor(feature) or feature.ndim != 5:
            shape = tuple(feature.shape) if torch.is_tensor(feature) else type(feature).__name__
            raise ValueError(f"Transfer feature {feature_index} must be a 5D tensor [B,C,D,H,W], got {shape}.")
    h_coarse = ordered[-1]
    h_global = F.adaptive_avg_pool3d(h_coarse, output_size=1).flatten(1)
    return {
        "encoder_features": ordered,
        "h_dense": ordered[index],
        "h_coarse": h_coarse,
        "h_global": h_global,
    }


def encode_transfer_representations(
    model,
    x: torch.Tensor,
    *,
    modality_id=None,
    dense_level: int = -2,
) -> dict[str, torch.Tensor | tuple]:
    """Execute the transferable encoder path and apply the shared task contract.

    The dispatch mirrors the real model families without invoking reconstruction
    decoders or SSL heads: SSL/EMA wrappers use ``_encode_skips``, classification
    wrappers use ``_encode``, and bare segmentation models use ``encoder``.
    """
    encode_skips = getattr(model, "_encode_skips", None)
    if callable(encode_skips):
        features = encode_skips(x, modality_id=modality_id)
    else:
        encode = getattr(model, "_encode", None)
        if callable(encode):
            features = encode(x)
        else:
            encoder = getattr(model, "encoder", None)
            if not callable(encoder):
                raise ValueError(
                    f"{type(model).__name__} has no transferable encoder path "
                    "(`_encode_skips`, `_encode`, or callable `encoder`)."
                )
            features = encoder(x)
    return build_transfer_representations(features, dense_level=dense_level)


# Moved here from the latent-prediction objective package, which this distribution does not
# ship. The metric is objective-agnostic: it is the non-collapse diagnostic the AMAES
# validation probe logs, so it must not live in an objective-specific module.
def embedding_health_metrics(embeddings: torch.Tensor, dead_std: float = 1e-3) -> dict[str, torch.Tensor]:
    """Inter-sample collapse diagnostics for one pooled embedding per scan.

    These statistics are defined *across samples*, so they mean nothing for a cohort of fewer than
    two. That degenerate case is reported explicitly through ``health_valid=0`` (with
    ``cohort_size``) instead of silently looking like a total collapse: at microbatch 1 the
    per-batch call returned dead_dim_fraction=1.0 and effective_rank=0.0, which is indistinguishable
    from a genuinely collapsed representation. ``representation_health_failures`` refuses to judge
    an invalid cohort, so a sentinel can neither trip nor mask the gate.

    The key schema is identical in both branches, which keeps DDP logging consistent across ranks.
    """
    x = embeddings.float()
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)
    cohort_size = x.new_tensor(float(x.shape[0]))
    finite_fraction = torch.isfinite(x).float().mean() if x.numel() else x.new_zeros(())
    # A non-finite cohort cannot be decomposed (torch.linalg.svdvals raises), and its rank
    # statistics would be meaningless anyway. Report it through finite_fraction and mark the
    # cohort invalid; a genuinely non-finite representation is caught by the dedicated
    # ssl.health.abort_on_nonfinite path, not by silently returning collapse-shaped numbers.
    if x.shape[0] < 2 or x.shape[1] == 0 or not bool(torch.isfinite(x).all()):
        zero = x.new_zeros(())
        return {
            "dim_std_mean": zero,
            "dim_std_median": zero,
            "dead_dim_fraction": x.new_ones(()),
            "effective_rank": zero,
            "effective_rank_normalized": zero,
            "singular_value_max": zero,
            "singular_value_median": zero,
            "singular_value_min": zero,
            "centered_pair_cosine_abs": zero,
            "feature_norm_mean": (
                x[torch.isfinite(x).all(dim=1)].norm(dim=1).mean() if x.numel() and bool(torch.isfinite(x).any()) else zero
            ),
            "finite_fraction": finite_fraction,
            "cohort_size": cohort_size,
            "health_valid": zero,
        }
    dim_std = x.std(dim=0, unbiased=False)
    centered = x - x.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered / math.sqrt(max(1, x.shape[0] - 1)))
    singular_mass = singular.sum()
    if bool((dim_std < float(dead_std)).all()):
        effective_rank = singular_mass.new_zeros(())
    else:
        probabilities = singular / singular_mass.clamp_min(1e-12)
        effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    max_rank = float(min(x.shape[0] - 1, x.shape[1]))
    normalized = effective_rank / max(1.0, max_rank)
    unit = F.normalize(centered, dim=1, eps=1e-12)
    cosine = unit @ unit.transpose(0, 1)
    off_diagonal = ~torch.eye(x.shape[0], dtype=torch.bool, device=x.device)
    return {
        "dim_std_mean": dim_std.mean(),
        "dim_std_median": dim_std.median(),
        "dead_dim_fraction": (dim_std < float(dead_std)).float().mean(),
        "effective_rank": effective_rank,
        "effective_rank_normalized": normalized,
        "singular_value_max": singular.max(),
        "singular_value_median": singular.median(),
        "singular_value_min": singular.min(),
        "centered_pair_cosine_abs": cosine[off_diagonal].abs().mean(),
        "feature_norm_mean": x.norm(dim=1).mean(),
        "finite_fraction": finite_fraction,
        "cohort_size": cohort_size,
        "health_valid": x.new_ones(()),
    }


def representation_health_failures(
    metrics: dict[str, torch.Tensor],
    *,
    dead_dim_fraction_max: float = 0.05,
    effective_rank_normalized_min: float = 0.25,
    centered_pair_cosine_abs_max: float = 0.95,
) -> tuple[str, ...]:
    """Return the failed Phase-2 gate names for one globally gathered representation.

    An invalid cohort (fewer than two samples) is not evidence of anything and is never reported as
    a failure; the caller is expected to require a real cohort size before gating.
    """
    if float(metrics.get("health_valid", 1.0)) < 1.0:
        return ()
    failures = []
    if float(metrics["dead_dim_fraction"]) > float(dead_dim_fraction_max):
        failures.append("dead_dim_fraction")
    if float(metrics["effective_rank_normalized"]) < float(effective_rank_normalized_min):
        failures.append("effective_rank_normalized")
    if float(metrics["centered_pair_cosine_abs"]) >= float(centered_pair_cosine_abs_max):
        failures.append("centered_pair_cosine_abs")
    return tuple(failures)
