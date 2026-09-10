"""RNG isolation for paired augmentation batches.

``legacy_global`` deliberately preserves the historical behaviour: GPU
augmentations consume the process-wide Python, NumPy and Torch RNG streams.
``paired_batch`` derives a stable seed from the logical batch coordinates and
runs the transforms in a temporary RNG context.  Masking remains outside
that context, so candidate arms may use different masking recipes without
changing the next batch's augmented input.
"""

import numpy as np
import random
import torch
from asparagus.modules.lightning_modules.ssl.metadata import stable_int_hash
from collections.abc import Iterator
from contextlib import contextmanager

LEGACY_GLOBAL = "legacy_global"
PAIRED_BATCH = "paired_batch"
SUPPORTED_AUGMENTATION_MODES = frozenset({LEGACY_GLOBAL, PAIRED_BATCH})
# These two strings are seed-derivation inputs, not labels: changing either changes every RNG
# stream derived from it, so they keep their original values even though the objective that
# named them is not part of this distribution.
PAIRED_BATCH_SEED_VERSION = "jepa_paired_batch_v1"
MASK_SEED_VERSION = "jepa_mask_v1"


def mask_batch_seed(
    *,
    run_seed: int,
    optimizer_step: int,
    batch_index: int,
    rank: int,
    version: str = MASK_SEED_VERSION,
) -> int:
    """Derive a stable seed for one rank-local mask draw.

    The key is stateless: it depends only on coordinates that a resumed run reconstructs exactly
    (run seed, restored optimizer-step counter, batch index within the epoch, rank). Nothing about
    the mask stream therefore has to be checkpointed, and a resumed run redraws the same masks.
    """
    coordinates = (
        str(version),
        int(run_seed),
        int(optimizer_step),
        int(batch_index),
        int(rank),
    )
    return stable_int_hash(coordinates)


def validate_augmentation_mode(mode: str) -> str:
    """Return a normalized augmentation mode or fail loudly."""
    normalized = str(mode).strip().lower()
    if normalized not in SUPPORTED_AUGMENTATION_MODES:
        raise ValueError(
            f"Unknown ssl.rng.augmentation_mode={mode!r}; expected one of {sorted(SUPPORTED_AUGMENTATION_MODES)}."
        )
    return normalized


def paired_batch_seed(
    *,
    run_seed: int,
    epoch: int,
    batch_index: int,
    rank: int,
    namespace: str,
    version: str = PAIRED_BATCH_SEED_VERSION,
) -> int:
    """Derive a stable seed for one rank-local augmented batch."""
    coordinates = (
        str(version),
        int(run_seed),
        int(epoch),
        int(batch_index),
        int(rank),
        str(namespace),
    )
    return stable_int_hash(coordinates)


def _batch_cuda_devices(batch) -> list[int]:
    image = batch.get("image") if isinstance(batch, dict) else None
    if not isinstance(image, torch.Tensor) or not image.is_cuda:
        return []
    index = image.device.index
    return [torch.cuda.current_device() if index is None else int(index)]


@contextmanager
def isolated_rng(seed: int, *, cuda_devices: list[int] | tuple[int, ...] = ()) -> Iterator[None]:
    """Temporarily seed and then restore Python, NumPy, Torch and used CUDA RNGs."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    devices = list(dict.fromkeys(int(device) for device in cuda_devices))
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(int(seed))
            np.random.seed(int(seed) % (2**32))
            # Seed the CPU generator without implicitly touching every visible CUDA
            # device; only the rank-local devices captured by ``fork_rng`` are seeded.
            torch.random.default_generator.manual_seed(int(seed))
            for device in devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(int(seed))
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def apply_batch_augmentation(
    batch,
    transforms,
    *,
    mode: str,
    run_seed: int,
    epoch: int,
    batch_index: int,
    rank: int,
    namespace: str,
):
    """Apply train transforms under the configured RNG contract."""
    mode = validate_augmentation_mode(mode)
    if transforms is None:
        return batch
    if mode == LEGACY_GLOBAL:
        return transforms(batch)

    seed = paired_batch_seed(
        run_seed=run_seed,
        epoch=epoch,
        batch_index=batch_index,
        rank=rank,
        namespace=namespace,
    )
    with isolated_rng(seed, cuda_devices=_batch_cuda_devices(batch)):
        return transforms(batch)
