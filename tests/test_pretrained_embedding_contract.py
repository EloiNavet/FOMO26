"""The official FOMO26 Tasks 6/7 contract: a frozen PRETRAINED encoder, one NIfTI, one 1-D vector.

Tasks 6 and 7 probe frozen pretrained representations. The embedding must therefore come from the
pretrained checkpoint a candidate transfers everywhere else, never from a Task 1-5 finetuned run:
a finetuned encoder has seen downstream labels, and it carries that task's modality count and
prediction head, neither of which describes the representation.

The output contract enforced here is the one the official validator applies at
container-validator d442af2e9bdade58be20c2ee0cbabf8d0439e32b (``container_validator/tasks.py``,
``output_numpy.py``): np.load-able without pickle, floating dtype, 1-D after squeeze, all finite,
and one consistent dimension across subjects.
"""

from __future__ import annotations

import inspect
import nibabel as nib
import numpy as np
import pytest
import torch
from finetuning.fomo26_inference import pretrained_embedding as pe
from pathlib import Path

RESENC_B_STEM_CHANNELS = 1


def _nifti(path: Path, shape=(32, 32, 16), seed: int = 0, spacing=1.0) -> Path:
    """A small RAS scalar volume shaped like the official validator's own fixtures."""
    rng = np.random.default_rng(seed)
    data = rng.normal(100.0, 25.0, size=shape).astype(np.float32)
    affine = np.diag([spacing, spacing, spacing, 1.0])
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


class _Conv1(torch.nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = torch.nn.Conv3d(cin, cout, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class _Stem(torch.nn.Module):
    """Named so its weight lands at ``encoder.stem.conv1.conv.weight``, like the real backbones."""

    def __init__(self, cin: int):
        super().__init__()
        self.conv1 = _Conv1(cin, 8)

    def forward(self, x):
        return self.conv1(x)


class _Encoder(torch.nn.Module):
    """Two feature grids, so build_h_global concatenates an 8+4 = 12-d pooled vector."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.stem = _Stem(in_channels)
        self.deeper = torch.nn.Conv3d(8, 4, 3, padding=1)

    def forward(self, x):
        shallow = self.stem(x)
        return [shallow, self.deeper(shallow)]


class _StubSSLNet(torch.nn.Module):
    """Stands in for a real SSL backbone: publishes encode_representations, no task head."""

    pretrained_backbone_prefixes = ("encoder.",)

    def __init__(self, input_channels: int = 1, output_channels: int = 1):
        super().__init__()
        self.encoder = _Encoder(input_channels)
        self.output_channels = output_channels

    def encode_representations(self, x, modality_id=None, use_modality_conditioning: bool = True):
        from asparagus.functional.representations import build_h_global

        features = self.encoder(x)
        return {"h_dense": features, "h_global": build_h_global(features)}


@pytest.fixture
def stub_architecture(monkeypatch):
    """Register a tiny architecture so the contract is exercised without a 102M-parameter net."""
    monkeypatch.setattr(pe, "known_architectures", lambda: ["stub_arch"])
    monkeypatch.setattr(
        pe,
        "build_backbone",
        lambda architecture, input_channels, output_channels: _StubSSLNet(input_channels, output_channels),
    )
    return "stub_arch"


def _stub_checkpoint(tmp_path: Path, input_channels: int = 1, prefix: str = "model.") -> Path:
    net = _StubSSLNet(input_channels)
    state = {f"{prefix}{k}": v for k, v in net.state_dict().items()}
    assert f"{prefix}encoder.stem.conv1.conv.weight" in state, "fixture must carry a readable stem"
    path = tmp_path / "pretrained.ckpt"
    torch.save({"state_dict": state, "global_step": 89775}, path)
    return path


# ── A. no downstream checkpoint can enter the canonical path ──────────────────────────────


def test_the_official_entrypoint_takes_a_pretrained_checkpoint_not_a_run_directory():
    from finetuning.container import predict_task6_7

    source = inspect.getsource(predict_task6_7)
    assert "--checkpoint" in source
    assert "model_dir" not in source, "a downstream run directory must not reach the Tasks 6/7 path"
    assert "model-dir" not in source
    assert "best.ckpt" not in source and "checkpoint_name" not in source


def test_no_official_tasks_6_7_module_imports_the_downstream_ensemble_predictor():
    """The finetuned-backbone embedding route is gone; nothing may quietly route back to it."""
    from finetuning.container import fomo_ensemble_predict

    assert not hasattr(fomo_ensemble_predict, "predict_embedding")

    for module_path in ("finetuning/container/predict_task6_7.py", "finetuning/fomo26_inference/extract_embeddings.py"):
        text = Path(module_path).read_text()
        assert "fomo_ensemble_predict" not in text, f"{module_path} must not reach the downstream predictor"
        assert "predict_embedding" not in text


def test_the_batch_extractor_also_requires_a_pretrained_checkpoint():
    from finetuning.fomo26_inference import extract_embeddings

    source = inspect.getsource(extract_embeddings.parse_args)
    assert '"--checkpoint"' in source and '"--architecture"' in source
    assert "--model-dir" not in source


# ── B. label data never reaches the encoder ───────────────────────────────────────────────


def test_a_stored_image_label_pair_never_passes_its_label_to_the_encoder(tmp_path, stub_architecture):
    """`.pt` samples are stored as [image, label]; only the image is an encoder input."""
    from asparagus.modules.datasets.TrainDataset import SingleSubjectPredictDataset
    from asparagus.modules.transforms.presets.pretrain import CPU_val_transforms

    image = torch.arange(8 * 8 * 8, dtype=torch.float32).reshape(1, 8, 8, 8)
    label = torch.full((1, 8, 8, 8), -999.0)
    sample_path = tmp_path / "scan.pt"
    torch.save([image, label], sample_path)

    dataset = SingleSubjectPredictDataset([str(sample_path)], transforms=CPU_val_transforms([8, 8, 8]))
    loaded = dataset[0]["image"]
    assert loaded.shape[0] == 1, "the label half must not become a second input channel"
    assert not torch.isclose(loaded, torch.tensor(-999.0)).any()


# ── C / F. the channel and coverage contract comes from the pretrained checkpoint ─────────


def test_the_input_channel_count_is_read_from_the_pretrained_stem():
    for suffix in ("encoder.stem.conv1.conv.weight", "encoder.stem.conv1.all_modules.0.weight"):
        assert pe.stem_input_channels({f"model.{suffix}": torch.ones(32, 1, 3, 3, 3)}) == 1
        assert pe.stem_input_channels({f"model.{suffix}": torch.ones(32, 4, 3, 3, 3)}) == 4
    # Nothing readable -> the documented default, never a downstream task's modality count.
    assert pe.stem_input_channels({"model.decoder.fc.weight": torch.ones(2, 8)}) == 1
    assert pe.stem_input_channels(None) == 1


def test_loading_reports_full_coverage_and_freezes_every_parameter(tmp_path, stub_architecture):
    frozen = pe.load_frozen_pretrained_encoder(_stub_checkpoint(tmp_path), stub_architecture, checkpoint_source="online")
    assert frozen.load_fraction == 1.0
    assert frozen.encoder_keys_loaded == frozen.encoder_keys_expected
    assert frozen.shape_mismatches == 0
    assert frozen.input_channels == 1
    assert not any(p.requires_grad for p in frozen.model.parameters())
    assert not frozen.model.training


def test_coverage_below_the_threshold_fails_closed(tmp_path, stub_architecture):
    path = _stub_checkpoint(tmp_path)
    payload = torch.load(path, weights_only=False)
    dropped = next(k for k in payload["state_dict"] if k.startswith("model.encoder.deeper"))
    del payload["state_dict"][dropped]
    torch.save(payload, path)

    with pytest.raises((RuntimeError, ValueError)):
        pe.load_frozen_pretrained_encoder(path, stub_architecture, checkpoint_source="online")


def test_a_multichannel_input_is_refused_rather_than_reshaped(tmp_path, stub_architecture):
    """Never stack, replicate, truncate or average channels to bridge a mismatch."""
    frozen = pe.load_frozen_pretrained_encoder(_stub_checkpoint(tmp_path), stub_architecture, checkpoint_source="online")
    four_channel = torch.zeros(4, 8, 8, 8)
    sample = tmp_path / "multi.pt"
    torch.save(four_channel, sample)
    with pytest.raises(ValueError, match="do not stack, replicate or drop channels"):
        pe.preprocess_volume(sample, (8, 8, 8), expected_channels=frozen.input_channels)


# ── D / E. one official NIfTI -> one finite, deterministic 1-D embedding ──────────────────


def test_one_official_nifti_produces_one_finite_1d_embedding(tmp_path, stub_architecture):
    receipt = pe.extract_embedding(
        checkpoint=_stub_checkpoint(tmp_path),
        architecture=stub_architecture,
        input_path=_nifti(tmp_path / "case_001" / "t2w.nii.gz"),
        output_path=tmp_path / "out" / "case_001.npy",
        patch_size=(16, 16, 16),
        checkpoint_source="online",
        accelerator="cpu",
    )
    # Exactly the checks container_validator/output_numpy.py applies.
    arr = np.load(tmp_path / "out" / "case_001.npy", allow_pickle=False)
    assert np.issubdtype(arr.dtype, np.floating)
    assert np.squeeze(arr).ndim == 1
    assert np.isfinite(arr).all()
    assert arr.shape[0] == receipt["embedding_dim"] > 0
    assert receipt["downstream_finetuned_weights"] is None


def test_embedding_dimension_is_consistent_across_subjects_and_input_shapes(tmp_path, stub_architecture):
    """The validator compares dimensions across subjects; pooling makes it geometry-independent."""
    checkpoint = _stub_checkpoint(tmp_path)
    frozen = pe.load_frozen_pretrained_encoder(checkpoint, stub_architecture, checkpoint_source="online")
    dims = set()
    for index, shape in enumerate([(32, 32, 16), (24, 30, 41), (8, 8, 8)]):
        image = pe.preprocess_volume(_nifti(tmp_path / f"s{index}.nii.gz", shape, seed=index), (16, 16, 16))
        dims.add(int(pe.embed_volume(frozen, image).shape[0]))
    assert len(dims) == 1, f"embedding dimension must not depend on input geometry, got {dims}"


def test_repeated_extraction_of_the_same_input_is_bitwise_identical(tmp_path, stub_architecture):
    """Deterministic policy only: eval mode, no dropout draw, no random crop, no TTA.

    Tolerance is exactly zero on CPU -- the same weights over the same tensor in the same order.
    """
    checkpoint = _stub_checkpoint(tmp_path)
    volume = _nifti(tmp_path / "case.nii.gz")
    first = pe.load_frozen_pretrained_encoder(checkpoint, stub_architecture, checkpoint_source="online")
    second = pe.load_frozen_pretrained_encoder(checkpoint, stub_architecture, checkpoint_source="online")

    a = pe.embed_volume(first, pe.preprocess_volume(volume, (16, 16, 16)))
    b = pe.embed_volume(first, pe.preprocess_volume(volume, (16, 16, 16)))
    c = pe.embed_volume(second, pe.preprocess_volume(volume, (16, 16, 16)))
    assert np.array_equal(a, b), "same encoder, same input, same bytes"
    assert np.array_equal(a, c), "a reloaded encoder must reproduce the embedding exactly"


def test_a_non_finite_embedding_is_refused(tmp_path, stub_architecture, monkeypatch):
    frozen = pe.load_frozen_pretrained_encoder(_stub_checkpoint(tmp_path), stub_architecture, checkpoint_source="online")
    monkeypatch.setattr(
        frozen.model, "encode_representations", lambda x, **kw: {"h_global": torch.tensor([[1.0, float("nan")]])}
    )
    with pytest.raises(ValueError, match="non-finite"):
        pe.embed_volume(frozen, torch.zeros(1, 1, 8, 8, 8))


# ── G. no downstream task predict_step is ever invoked ────────────────────────────────────


def test_extraction_never_routes_through_a_trainer_or_a_task_predict_step():
    source = inspect.getsource(pe)
    assert "Trainer" not in source
    assert "predict_step" not in source
    assert "_lightning_module" not in source
    assert "encode_representations" in source


def test_the_embedding_is_independent_of_any_head_width(tmp_path, stub_architecture, monkeypatch):
    """The pooled representation must not move when the unused SSL head width changes."""
    checkpoint = _stub_checkpoint(tmp_path)
    volume = _nifti(tmp_path / "case.nii.gz")
    embeddings = []
    for width in (1, 3):
        monkeypatch.setattr(pe, "_UNUSED_SSL_HEAD_CHANNELS", width)
        frozen = pe.load_frozen_pretrained_encoder(checkpoint, stub_architecture, checkpoint_source="online")
        assert frozen.model.output_channels == width
        embeddings.append(pe.embed_volume(frozen, pe.preprocess_volume(volume, (16, 16, 16))))
    assert np.array_equal(embeddings[0], embeddings[1])


# ── H / I. source selection and generic A/B/C composition ─────────────────────────────────


def test_online_and_ema_selection_follow_the_objective_contract():
    assert pe.resolve_source("amaes", None) == "online"
    assert pe.resolve_source("jepa", None) == "ema_if_available"
    # An explicit request always wins over the objective default.
    assert pe.resolve_source("amaes", "ema") == "ema"
    assert pe.resolve_source(None, "online") == "online"
    with pytest.raises(ValueError):
        pe.resolve_source(None, None)
    with pytest.raises(ValueError, match="Unknown ssl_objective"):
        pe.resolve_source("not_an_objective", None)


def test_requesting_ema_from_a_checkpoint_without_one_fails_loudly(tmp_path, stub_architecture):
    with pytest.raises(RuntimeError, match="no target_encoder"):
        pe.load_frozen_pretrained_encoder(_stub_checkpoint(tmp_path), stub_architecture, checkpoint_source="ema")


def test_an_ema_checkpoint_is_read_from_its_target_encoder(tmp_path, stub_architecture):
    """Candidate C's contract may specify EMA; the same generic API must serve it."""
    path = _stub_checkpoint(tmp_path)
    payload = torch.load(path, weights_only=False)
    online = dict(payload["state_dict"])
    # An EMA copy whose encoder weights differ from the online copy.
    for key, value in online.items():
        if key.startswith("model.encoder."):
            payload["state_dict"]["target_encoder." + key[len("model.") :]] = value + 1.0
    torch.save(payload, path)

    ema = pe.load_frozen_pretrained_encoder(path, stub_architecture, checkpoint_source="ema")
    online_encoder = pe.load_frozen_pretrained_encoder(path, stub_architecture, checkpoint_source="online")
    assert ema.source_used == "ema"
    assert online_encoder.source_used == "online"
    assert pe.load_frozen_pretrained_encoder(path, stub_architecture, ssl_objective="jepa").source_used == "ema"
    assert not torch.equal(ema.model.encoder.deeper.weight, online_encoder.model.encoder.deeper.weight)


def test_an_unknown_architecture_is_refused_rather_than_defaulted(tmp_path):
    with pytest.raises(ValueError, match="Unknown architecture"):
        pe.load_frozen_pretrained_encoder(tmp_path / "x.ckpt", "no_such_arch", checkpoint_source="online")


def test_the_registry_exposes_the_architectures_the_campaign_needs():
    from finetuning.fomo26_inference.backbones import known_architectures

    assert {"resenc_b", "unet_m"} <= set(known_architectures())
