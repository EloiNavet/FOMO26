"""Experiment A: the pure ResEnc-AMAES comparator must stay strictly reconstruction-only."""

import os
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

BASE = "projects/fomo26/safety/pretrain"
MANIFEST = "78ce45fd74d330687bd8a993d648b58504e6da8e3397f3716dedc6e4799a1eb9"
TENSORS = "ea4ada4400b78993ec2aea6e9ca0ee518c238a5e467834713f8e8a6bf4784702"


def _register():
    for name, fn in [("random", lambda a, b: 0), ("version", lambda: "test"), ("eval", eval)]:
        try:
            OmegaConf.register_new_resolver(name, fn)
        except Exception:
            pass


def _compose(name):
    _register()
    os.environ.setdefault("ASPARAGUS_DATA", "/tmp")
    cfgdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
    with initialize_config_dir(version_base="1.2", config_dir=cfgdir):
        return compose(config_name=f"{BASE}/{name}")


@pytest.mark.parametrize(
    "name,stop",
    [
        ("resenc_amaes_cal", 200),
        ("resenc_amaes_p1_32k", 32000),
        ("resenc_amaes_p2_96k", 96000),
        ("resenc_amaes_p3_187k", 187500),
    ],
)
def test_stages_share_the_curated_contract_and_horizon(name, stop):
    cfg = _compose(name)
    dc = cfg.campaign.dataset_contract
    assert dc.manifest_sha256 == MANIFEST
    assert dc.tensor_manifest_sha256 == TENSORS
    assert int(dc.accepted_tensors) == 33336 and int(dc.excluded_tensors) == 0
    assert int(cfg.training.global_batch_size) == 32
    assert int(cfg.training.steps) == 187500, "scheduler horizon must not be rescaled"
    assert int(cfg.training.warmup_steps) == 3750, "corrected optimizer-step warmup"
    assert int(cfg.training.seed) == 431027
    assert int(cfg.campaign.stop_at_checkpoint_step) == stop


def test_only_reconstruction_is_enabled():
    """AMAES reconstruction is the whole objective: nothing else is even declared.

    The published run closed every other objective explicitly. This distribution goes further --
    those objectives are not part of the code release at all -- so the assertion is now that the
    loss table contains exactly one entry rather than that the others are switched off.
    """
    cfg = _compose("resenc_amaes_p1_32k")
    assert set(cfg.losses) == {"mse"}, sorted(cfg.losses)
    assert cfg.losses.mse.enabled is True
    assert cfg.losses.mse.foreground_mode == "none", "no foreground weighting"
    assert cfg.losses.mse.foreground_aware is False


def test_no_jepa_and_no_modality_conditioning():
    cfg = _compose("resenc_amaes_p1_32k")
    assert set(cfg.ssl) == {"matched_control"}, "no latent objective is declarable here"
    assert cfg.model.modality_conditioning is False, "FiLM is unavailable downstream"
    assert cfg.campaign.architecture == "resenc_unet_b"
    assert float(cfg.data.multimodal_batch_probability) == 0.0
    assert float(cfg.data.demographic_batch_probability) == 0.0


def test_objective_contract_accepts_every_stage():
    from asparagus.pipeline.run.pretrain import _validate_ssl_objective_contract

    for name in ("resenc_amaes_cal", "resenc_amaes_p1_32k", "resenc_amaes_p2_96k", "resenc_amaes_p3_187k"):
        _validate_ssl_objective_contract(_compose(name))


def test_resenc_amaes_forward_backward_and_save_load(tmp_path):
    """A real optimizer step on the ResEnc-B SSL trunk, then an exact state round-trip."""
    from asparagus.modules.networks.resenc_unet import resenc_unet_b_ssl

    torch.manual_seed(0)
    net = resenc_unet_b_ssl(
        dimensions="3D", input_channels=1, output_channels=1, use_skip_connections=True, modality_conditioning=False
    )
    # 64^3 keeps the 6-stage ResEnc bottleneck at 2^3; 32^3 collapses it to 1^3 and
    # InstanceNorm refuses a single spatial element in training mode.
    x = torch.randn(2, 1, 64, 64, 64)
    out = net(x)
    pred = out[0] if isinstance(out, (tuple, list)) else out
    assert torch.isfinite(pred).all()
    loss = torch.nn.functional.mse_loss(pred, x)
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads, "no gradients reached the trunk"
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads), "all gradients were exactly zero"

    ckpt = tmp_path / "amaes.pt"
    torch.save(net.state_dict(), ckpt)
    restored = resenc_unet_b_ssl(
        dimensions="3D", input_channels=1, output_channels=1, use_skip_connections=True, modality_conditioning=False
    )
    restored.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    for (ka, va), (kb, vb) in zip(net.state_dict().items(), restored.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)


def test_encoder_transfers_to_the_downstream_resenc_trunk():
    """The pretrained encoder keys must land on the downstream ResEnc segmentation trunk."""
    from asparagus.modules.networks.resenc_unet import resenc_unet_b, resenc_unet_b_ssl

    ssl_net = resenc_unet_b_ssl(
        dimensions="3D", input_channels=1, output_channels=1, use_skip_connections=True, modality_conditioning=False
    )
    seg_net = resenc_unet_b(dimensions="3D", input_channels=1, output_channels=2)
    ssl_encoder = {k: v for k, v in ssl_net.state_dict().items() if k.startswith("encoder.")}
    seg_encoder = {k: v for k, v in seg_net.state_dict().items() if k.startswith("encoder.")}
    assert ssl_encoder, "SSL trunk exposes no encoder.* parameters"
    shared = set(ssl_encoder) & set(seg_encoder)
    assert shared, "no encoder parameter names are shared with the downstream trunk"
    mismatched = [k for k in shared if ssl_encoder[k].shape != seg_encoder[k].shape]
    assert not mismatched, f"shape mismatch on transfer: {mismatched[:5]}"
    # The overwhelming majority of the downstream encoder must be covered by the pretrained one.
    assert len(shared) / len(seg_encoder) > 0.9, f"only {len(shared)}/{len(seg_encoder)} encoder keys transfer"
