"""Stateless metadata/tensor helpers for the SSL trainer (P3.9 decomposition).

Pure functions (no module state) used to coerce heterogeneous batch metadata into
tensors, hash subject/session identifiers, and summarise metadata for logging. The
``SelfSupervisedModule`` keeps thin delegating methods so existing ``self._x`` call
sites and the public surface are unchanged.
"""

import hashlib
import torch
from typing import Optional


def contrastive_metadata(batch) -> dict:
    keys = (
        "age",
        "sex",
        "pathology",
        "fine_pathology",
        "scanner_id",
        "domain_manufacturer_id",
        "modality_id",
        "dataset_id",
        "dwi_bval",
    )
    return {key: batch.get(key) for key in keys if batch.get(key) is not None}


def metadata_to_list(values) -> list:
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().tolist()
    return list(values)


def stable_int_hash(value) -> int:
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    # Mask to 52 bits (< 2^53) so ids round-trip exactly through a float64 mantissa
    # when logged. Must match PretrainDataset._stable_int_hash for consistent ids.
    return int.from_bytes(digest, byteorder="little", signed=False) & 0xFFFFFFFFFFFFF


def subject_hash_tensor(subject_session_key, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [stable_int_hash(value) for value in metadata_to_list(subject_session_key)],
        dtype=torch.long,
        device=device,
    )


def hash_tensor(values, device: torch.device) -> torch.Tensor:
    return torch.tensor([stable_int_hash(value) for value in metadata_to_list(values)], dtype=torch.long, device=device)


def to_long_tensor(values, device: torch.device) -> torch.Tensor:
    if values is None:
        return torch.empty(0, dtype=torch.long, device=device)
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=torch.long).view(-1)
    return torch.as_tensor(values, dtype=torch.long, device=device).view(-1)


def to_bool_tensor(values, device: torch.device) -> torch.Tensor:
    if values is None:
        return torch.zeros(0, dtype=torch.bool, device=device)
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=torch.bool).view(-1)
    return torch.as_tensor(values, dtype=torch.bool, device=device).view(-1)


def to_float_tensor(values, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=dtype)

    numeric_values = []
    for value in values:
        if value is None:
            numeric_values.append(float("nan"))
            continue
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            numeric_values.append(float("nan"))

    return torch.tensor(numeric_values, device=device, dtype=dtype)


def valid_sample_mask(mask: Optional[torch.Tensor], size: int, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.ones(size, dtype=torch.bool, device=device)
    return mask.to(device=device, dtype=torch.bool).view(-1)


def contrastive_metadata_metrics(metadata) -> dict:
    metrics = {}
    ages = metadata.get("age")
    if ages is not None:
        ages_tensor = torch.as_tensor(ages, dtype=torch.float32)
        valid_ages = ages_tensor[~torch.isnan(ages_tensor)]
        metrics["age/valid_count"] = int(valid_ages.numel())
        metrics["age/missing_fraction"] = float(1.0 - (valid_ages.numel() / max(1, ages_tensor.numel())))
        if valid_ages.numel() > 0:
            metrics["age/mean"] = valid_ages.mean().item()
            metrics["age/std"] = valid_ages.std(unbiased=False).item() if valid_ages.numel() > 1 else 0.0
            metrics["age/min"] = valid_ages.min().item()
            metrics["age/max"] = valid_ages.max().item()
            metrics["age/p10"] = torch.quantile(valid_ages, 0.10).item()
            metrics["age/p50"] = torch.quantile(valid_ages, 0.50).item()
            metrics["age/p90"] = torch.quantile(valid_ages, 0.90).item()
        else:
            for key in ("mean", "std", "min", "max", "p10", "p50", "p90"):
                metrics[f"age/{key}"] = float("nan")

    # Bounded enums get one fraction per class; high-cardinality keys (dataset_id holds
    # hashed site ids) get a bounded top-k summary instead to avoid a metric-key explosion.
    bounded_enum_keys = ("sex", "pathology", "scanner_id", "modality_id")
    high_cardinality_keys = ("dataset_id",)
    for key in (*bounded_enum_keys, *high_cardinality_keys):
        values = metadata.get(key)
        if values is None:
            continue
        values_tensor = torch.as_tensor(values, dtype=torch.long)
        valid = values_tensor != -1
        valid_values = values_tensor[valid]
        metrics[f"{key}/valid_count"] = int(valid_values.numel())
        metrics[f"{key}/missing_fraction"] = float(1.0 - (valid_values.numel() / max(1, values_tensor.numel())))
        metrics[f"{key}/unique_count"] = int(torch.unique(valid_values).numel()) if valid_values.numel() > 0 else 0
        if valid_values.numel() == 0:
            continue
        if key in high_cardinality_keys:
            unique_values, counts = torch.unique(valid_values, return_counts=True)
            order = torch.argsort(counts, descending=True)
            for rank, index in enumerate(order[:5].tolist(), start=1):
                metrics[f"{key}/top{rank}_fraction"] = float(counts[index].item() / valid_values.numel())
                metrics[f"{key}/top{rank}_id"] = float(unique_values[index].item())
        else:
            for class_id in torch.unique(valid_values).tolist():
                metrics[f"{key}/class_{int(class_id)}_fraction"] = (valid_values == int(class_id)).float().mean().item()

    return metrics
