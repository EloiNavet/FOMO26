import lightning as L
import pickle
import pytest
import torch


@pytest.fixture
def pretrain_files(tmp_path):
    """Three .pt files of shape [1, 32, 32, 32] for pretraining (raw image, no label).
    32^3 ensures the UNet bottleneck (4 max-pool stages) stays at 2x2x2, avoiding
    single-element BatchNorm errors with batch_size=1.
    """
    files = []
    for i in range(3):
        path = tmp_path / f"pre_{i:03d}.pt"
        torch.save(torch.randn(1, 32, 32, 32), path)
        files.append(str(path))
    return {"train": files[:2], "val": [files[2]]}


@pytest.fixture
def seg_files(tmp_path):
    """Three .pt + .pkl file pairs for segmentation. Shape [2, 32, 32, 32] = [image, label].
    32^3 ensures the UNet bottleneck (4 max-pool stages) stays at 2x2x2.
    """
    files = []
    for i in range(3):
        pt = tmp_path / f"seg_{i:03d}.pt"
        pkl = tmp_path / f"seg_{i:03d}.pkl"
        data = torch.zeros(2, 32, 32, 32)
        data[0] = torch.randn(32, 32, 32)
        data[1] = torch.randint(0, 2, (32, 32, 32)).float()
        torch.save(data, pt)
        with open(pkl, "wb") as f:
            pickle.dump({"foreground_locations": []}, f)
        files.append(str(pt))
    return {"train": files[:2], "val": [files[2]]}


@pytest.fixture
def clsreg_files(tmp_path):
    """Three .pt files containing (image[1,32,32,32], label_scalar) tuples.
    32^3 prevents single-element BatchNorm errors in the 4-stage UNet encoder.
    Labels are 0-dim int tensors; ClassificationModule.on_before_batch_transfer
    squeezes and converts to long before the training step.
    """
    files = []
    for i in range(3):
        path = tmp_path / f"cls_{i:03d}.pt"
        torch.save((torch.randn(1, 32, 32, 32), torch.tensor(i % 2)), path)
        files.append(str(path))
    return {"train": files[:2], "val": [files[2]], "test": [files[2]]}


@pytest.fixture
def reg_files(tmp_path):
    """Three .pt files containing (image[1,32,32,32], label[1]) tuples.
    Labels are 1D float tensors so they collate to [B, 1], matching the
    unet_clsreg_tiny output shape [B, 1] expected by MeanSquaredError.
    """
    files = []
    for i in range(3):
        path = tmp_path / f"reg_{i:03d}.pt"
        torch.save((torch.randn(1, 32, 32, 32), torch.tensor([float(i % 2)])), path)
        files.append(str(path))
    return {"train": files[:2], "val": [files[2]], "test": [files[2]]}


@pytest.fixture
def cls_probe_files(tmp_path):
    """Five .pt files for classification / linear-probe tests. 0-dim integer labels.
    2 train + 2 val gives full batches when batch_size=2, avoiding the squeeze()-to-scalar
    edge case in ClassificationModule.on_before_batch_transfer with batch_size=1.
    2 test files (labels 1, 0) ensure both classes are present for AUROC computation.
    """
    labels = [0, 1, 0, 1, 0, 1]
    files = []
    for i, lbl in enumerate(labels):
        path = tmp_path / f"clsp_{i:03d}.pt"
        torch.save((torch.randn(1, 32, 32, 32), torch.tensor(lbl)), path)
        files.append(str(path))
    return {"train": files[:2], "val": files[2:4], "test": files[4:6]}


@pytest.fixture
def pretrain_files_2d(tmp_path):
    """Three .pt files of shape [1, 32, 32] for 2D pretraining (raw image, no label).
    2D analogue of pretrain_files; 32^2 keeps the tiny UNet bottleneck at 8x8.
    """
    files = []
    for i in range(3):
        path = tmp_path / f"pre2d_{i:03d}.pt"
        torch.save(torch.randn(1, 32, 32), path)
        files.append(str(path))
    return {"train": files[:2], "val": [files[2]]}


@pytest.fixture
def seg_files_ds(tmp_path):
    """Segmentation pairs at 64^3, for deep-supervision tests only.

    Deep supervision compares five decoder outputs against a label pyramid with fixed factors
    (1, 1/2, 1/4, 1/8, 1/16, 1/16), so the network must genuinely downsample by 32 -- which means
    the volume cannot be smaller than 64^3 without the bottleneck reaching a single voxel and
    InstanceNorm3d refusing it in training mode.

    This is deliberately separate from `seg_files` rather than an enlargement of it. `seg_files`
    is shared with the Primus and MedViT segmentation tests, whose docstring records 32^3 as a
    considered choice for a four-stage bottleneck; growing it would hand those two eight times
    the voxels to satisfy a ResEnc constraint they do not have.
    """
    files = []
    for i in range(3):
        pt = tmp_path / f"segds_{i:03d}.pt"
        pkl = tmp_path / f"segds_{i:03d}.pkl"
        data = torch.zeros(2, 64, 64, 64)
        data[0] = torch.randn(64, 64, 64)
        data[1] = torch.randint(0, 2, (64, 64, 64)).float()
        torch.save(data, pt)
        with open(pkl, "wb") as f:
            pickle.dump({"foreground_locations": []}, f)
        files.append(str(pt))
    return {"train": files[:2], "val": [files[2]]}


@pytest.fixture
def seg_files_2d(tmp_path):
    """Three .pt + .pkl file pairs for 2D segmentation. Shape [2, 32, 32] = [image, label].
    2D analogue of seg_files; .pkl sidecar required by SegDataModule.
    """
    files = []
    for i in range(3):
        pt = tmp_path / f"seg2d_{i:03d}.pt"
        pkl = tmp_path / f"seg2d_{i:03d}.pkl"
        data = torch.zeros(2, 32, 32)
        data[0] = torch.randn(32, 32)
        data[1] = torch.randint(0, 2, (32, 32)).float()
        torch.save(data, pt)
        with open(pkl, "wb") as f:
            pickle.dump({"foreground_locations": []}, f)
        files.append(str(pt))
    return {"train": files[:2], "val": [files[2]]}


@pytest.fixture
def cls_probe_files_2d(tmp_path):
    """Six .pt files for 2D classification / linear-probe tests. 0-dim integer labels.
    2D analogue of cls_probe_files; same label pattern [0,1,0,1,0,1].
    """
    labels = [0, 1, 0, 1, 0, 1]
    files = []
    for i, lbl in enumerate(labels):
        path = tmp_path / f"clsp2d_{i:03d}.pt"
        torch.save((torch.randn(1, 32, 32), torch.tensor(lbl)), path)
        files.append(str(path))
    return {"train": files[:2], "val": files[2:4], "test": files[4:6]}


@pytest.fixture
def reg_files_2d(tmp_path):
    """Three .pt files containing (image[1,32,32], label[1]) tuples.
    2D analogue of reg_files; labels are 1D float tensors so they collate to [B, 1].
    """
    files = []
    for i in range(3):
        path = tmp_path / f"reg2d_{i:03d}.pt"
        torch.save((torch.randn(1, 32, 32), torch.tensor([float(i % 2)])), path)
        files.append(str(path))
    return {"train": files[:2], "val": [files[2]], "test": [files[2]]}


@pytest.fixture
def make_trainer(tmp_path):
    """Factory fixture that builds a minimal CPU Trainer for smoke tests."""

    def _make(**kwargs):
        defaults = dict(
            accelerator="cpu",
            max_epochs=1,
            limit_train_batches=5,
            limit_val_batches=5,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
        )
        defaults.update(kwargs)
        return L.Trainer(default_root_dir=str(tmp_path), **defaults)

    return _make


# ---------------------------------------------------------------------------------------------
# Test tiers
# ---------------------------------------------------------------------------------------------
# `FOMO26_TIER` selects which tier runs; the default is "fast". `FOMO26_TIER=all` restores the
# historical unfiltered behaviour exactly, so the collection facts recorded during the release audit
# remain reproducible.
#
# Two mechanisms, because there are two problems. The NATTEN modules cannot be *imported* without
# the extension, so they are dropped before import via `collect_ignore`; a marker would be
# evaluated too late. Everything else is a cost decision, so it is deselected after collection and
# the reason stays visible in the report.
import os as _os  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_TIER = _os.environ.get("FOMO26_TIER", "fast").strip() or "fast"
_REPO_ROOT = _Path(__file__).resolve().parents[1]


def _tier_assignments() -> dict:
    """Load the manifest by path, so conftest does not depend on test-package importability."""
    import importlib.util

    manifest = _Path(__file__).parent / "test_tier_contract.py"
    if not manifest.is_file():
        return {}
    spec = importlib.util.spec_from_file_location("_fomo26_tier_manifest", manifest)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TIER_ASSIGNMENTS


_ASSIGNMENTS = _tier_assignments()

collect_ignore = []
if _TIER not in {"all", "optional_natten"}:
    collect_ignore = [
        _Path(rel).name for rel, tier in sorted(_ASSIGNMENTS.items()) if tier == "optional_natten" and rel.startswith("tests/")
    ]


def pytest_collection_modifyitems(config, items):
    if _TIER == "all" or not _ASSIGNMENTS:
        return
    keep, drop = [], []
    for item in items:
        try:
            rel = _Path(str(item.fspath)).resolve().relative_to(_REPO_ROOT).as_posix()
        except ValueError:
            keep.append(item)
            continue
        (keep if _ASSIGNMENTS.get(rel, "fast") == _TIER else drop).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


def pytest_report_header(config):
    if not _ASSIGNMENTS:
        return None
    from collections import Counter

    counts = Counter(_ASSIGNMENTS.values())
    if _TIER == "all":
        return "fomo26 test tier: all (no tier filtering)"
    excluded = ", ".join(f"{t}={counts[t]}" for t in sorted(counts) if t != _TIER and counts[t])
    return f"fomo26 test tier: {_TIER} ({counts.get(_TIER, 0)} files)" + (
        f" | excluded opt-in tiers: {excluded}" if excluded else ""
    )
