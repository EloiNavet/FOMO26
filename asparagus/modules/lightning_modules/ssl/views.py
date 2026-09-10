"""Contrastive view generation for the SSL trainer (P3.9 decomposition).

Builds the three GPU views (masked + two augmented) needed for joint MAE + contrastive
learning, plus the deterministic validation-transform path. Pure functions parameterised
by the transform pipelines; ``SelfSupervisedModule`` keeps thin delegating methods.
"""

import logging
import torch
from asparagus.modules.lightning_modules.ssl import metadata as ssl_metadata


def apply_demo_cpu_transforms_to_image(image, demo_cpu_transforms) -> torch.Tensor:
    if isinstance(image, (list, tuple)):
        # raw_image is collated as a per-sample list (variable spatial size, not stacked by
        # pretrain_collate). demo_cpu_transforms pad + center-crop each sample to the common
        # patch size, after which the per-sample tensors can be stacked into a batch.
        if demo_cpu_transforms is None:
            raise ValueError(
                "raw_image arrived as a per-sample list but demo_cpu_transforms is None; "
                "cannot normalise heterogeneous shapes before stacking."
            )
        transformed_images = [demo_cpu_transforms({"image": sample.clone()})["image"] for sample in image]
        return torch.stack(transformed_images, dim=0)
    if demo_cpu_transforms is None:
        return image.clone()
    if image.ndim == 4:
        return demo_cpu_transforms({"image": image.clone()})["image"]
    if image.ndim == 5:
        transformed_images = [demo_cpu_transforms({"image": sample.clone()})["image"] for sample in image]
        return torch.stack(transformed_images, dim=0)
    raise ValueError(f"Expected demographic image with shape CXYZ or BCXYZ, got {tuple(image.shape)}.")


def prepare_demographic_base_view(batch, demo_cpu_transforms):
    source = dict(batch)
    if demo_cpu_transforms is not None and "raw_image" in batch:
        source["image"] = apply_demo_cpu_transforms_to_image(batch["raw_image"], demo_cpu_transforms)
    else:
        source["image"] = batch["image"].clone()
    return source


def apply_deterministic_val_transforms(batch, val_transforms, validation_mask_seed: int):
    if val_transforms is None:
        return batch
    identity = "|".join(str(value) for value in batch.get("file_path", []))
    seed = (validation_mask_seed + ssl_metadata.stable_int_hash(identity)) % (2**31 - 1)
    image = batch.get("image")
    devices = [image.device] if isinstance(image, torch.Tensor) and image.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return val_transforms(batch)


def build_contrastive_views(
    batch,
    *,
    is_training: bool,
    train_transforms,
    unmasked_transforms,
    momentum_transforms,
    val_transforms,
    demo_cpu_transforms,
    validation_mask_seed: int,
):
    """Generate the three augmented GPU views needed for MAE + contrastive learning."""
    view_masked = dict(batch)
    view_masked["image"] = batch["image"].clone()
    demo_base = prepare_demographic_base_view(batch, demo_cpu_transforms)
    view_aug_1 = dict(demo_base)
    view_aug_2 = dict(demo_base)
    view_aug_1["image"] = demo_base["image"].clone()
    view_aug_2["image"] = demo_base["image"].clone()

    metadata = ssl_metadata.contrastive_metadata(batch)
    view_masked["metadata"] = metadata
    view_aug_1["metadata"] = metadata
    view_aug_2["metadata"] = metadata

    if is_training:
        if train_transforms is not None:
            view_masked = train_transforms(view_masked)

        if unmasked_transforms is not None:
            view_aug_1 = unmasked_transforms(view_aug_1)
        elif momentum_transforms is not None:
            view_aug_1 = momentum_transforms(view_aug_1)
        elif train_transforms is not None:
            logging.warning("No unmasked transforms provided. Using masked transforms for view_aug_1.")
            view_aug_1 = train_transforms(view_aug_1)

        if momentum_transforms is not None:
            view_aug_2 = momentum_transforms(view_aug_2)
        elif unmasked_transforms is not None:
            view_aug_2 = unmasked_transforms(view_aug_2)
        elif train_transforms is not None:
            logging.warning("No momentum transforms provided. Using masked transforms for view_aug_2.")
            view_aug_2 = train_transforms(view_aug_2)
    else:
        if val_transforms is not None:
            view_masked = apply_deterministic_val_transforms(view_masked, val_transforms, validation_mask_seed)
        # Keep validation auxiliary views deterministic; training augmentations make
        # auxiliary validation losses and embedding diagnostics non-comparable.

    return {
        "view_masked": view_masked,
        "view_original": dict(batch),
        "view_aug_1": view_aug_1,
        "view_aug_2": view_aug_2,
        "metadata": metadata,
        "age": batch.get("age"),
        "sex": batch.get("sex"),
        "pathology": batch.get("pathology"),
        "fine_pathology": batch.get("fine_pathology"),
        "scanner_id": batch.get("scanner_id"),
        "scanner_targets": batch.get("scanner_targets"),
        "dataset_id": batch.get("dataset_id"),
        "domain_manufacturer_id": batch.get("domain_manufacturer_id"),
        "modality_id": batch.get("modality_id"),
        # Diffusion b-value must reach the loss step so DWI demographic subtypes (e.g.
        # dwi_b1000) can be filtered by b-value; without it their eligibility is empty.
        "dwi_bval": batch.get("dwi_bval"),
        "subject_key": batch.get("subject_key"),
        "subject_session_key": batch.get("subject_session_key"),
        "is_registered_subset": batch.get("is_registered_subset"),
    }
