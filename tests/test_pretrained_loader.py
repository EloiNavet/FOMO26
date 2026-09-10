"""Tests for the SSL->downstream pretrained-encoder loader (PR7).

Covers source selection (online / ema / ema_if_available + fallback), encoder-only scoping,
JEPA/objective key dropping, torch.compile prefix stripping, the fail-loud behaviour, the report
counts, and one integration test against the real ResEnc downstream net.
"""

import os
import pytest
import torch
from asparagus.pipeline.auto_configuration.pretrained import extract_pretrained_encoder_state


def _fake_ssl_state_dict():
    """A realistic SSL (jepa+deep_sup) checkpoint: online + EMA backbones + jepa predictors."""
    return {
        # online backbone
        "model.encoder.stem.weight": torch.randn(4, 1, 3, 3, 3),
        "model.encoder.stages.0.conv.weight": torch.randn(8, 4, 3, 3, 3),
        "model.encoder_films.0.weight": torch.randn(8),  # FiLM -> not encoder.* -> dropped (encoder_only)
        "model.decoder.0.conv.weight": torch.randn(2, 8, 1, 1, 1),  # decoder -> dropped
        "model.head_demo.fc.weight": torch.randn(4, 8),  # SSL head -> dropped
        "model.modality_embedding.weight": torch.randn(5, 8),  # dropped
        # EMA / target backbone
        "momentum_model.encoder.stem.weight": torch.randn(4, 1, 3, 3, 3),
        "momentum_model.encoder.stages.0.conv.weight": torch.randn(8, 4, 3, 3, 3),
        # JEPA-only predictor heads
        "jepa.predictors.-2.predictor_embed.weight": torch.randn(16, 8),
        "jepa.predictors.-1.mask_token": torch.randn(1, 1, 16),
    }


_TARGET = {"model.encoder.stem.weight", "model.encoder.stages.0.conv.weight"}


def test_online_encoder_only():
    sd, report = extract_pretrained_encoder_state(_fake_ssl_state_dict(), {"source": "online"}, _TARGET)
    assert set(sd.keys()) == _TARGET  # only encoder, remapped to model.encoder.*
    assert report.source_used == "online" and report.n_encoder == 2 and report.n_encoder_match == 2
    assert report.n_skipped_jepa == 2  # both jepa.predictors keys
    assert report.n_skipped_other_source == 2  # the 2 momentum_model.encoder keys
    assert report.n_skipped_decoder_head == 4  # encoder_films + decoder + head_demo + modality_embedding


def test_ema_source_remaps_momentum_to_model():
    sd, report = extract_pretrained_encoder_state(_fake_ssl_state_dict(), {"source": "ema"}, _TARGET)
    assert set(sd.keys()) == _TARGET  # momentum_model.encoder.* remapped to model.encoder.*
    assert report.source_used == "ema" and report.n_encoder_match == 2
    assert report.n_skipped_other_source == 6  # all 6 model.* keys are the non-selected source now


def test_ema_source_accepts_new_encoder_only_target_prefix():
    state = {
        "model.encoder.stem.weight": torch.randn(4, 1, 3, 3, 3),
        "target_encoder.encoder.stem.weight": torch.randn(4, 1, 3, 3, 3),
        "target_encoder.encoder.stages.0.conv.weight": torch.randn(8, 4, 3, 3, 3),
        "jepa.predictors.-2.mask_token": torch.randn(1, 1, 8),
    }
    sd, report = extract_pretrained_encoder_state(state, {"source": "ema"}, _TARGET)
    assert set(sd) == _TARGET
    assert report.source_used == "ema" and report.n_encoder_match == 2


def test_ema_if_available_prefers_ema_then_falls_back():
    sd, report = extract_pretrained_encoder_state(_fake_ssl_state_dict(), {"source": "ema_if_available"}, _TARGET)
    assert report.source_used == "ema"  # momentum present
    online_only = {k: v for k, v in _fake_ssl_state_dict().items() if not k.startswith("momentum_model.")}
    _, report2 = extract_pretrained_encoder_state(online_only, {"source": "ema_if_available"}, _TARGET)
    assert report2.source_used == "online"  # no EMA -> fall back


def test_ema_source_without_ema_raises():
    online_only = {k: v for k, v in _fake_ssl_state_dict().items() if not k.startswith("momentum_model.")}
    with pytest.raises(RuntimeError, match="no target_encoder"):
        extract_pretrained_encoder_state(online_only, {"source": "ema"}, _TARGET)


def test_strips_torch_compile_prefix():
    sd = {"model._orig_mod.encoder.stem.weight": torch.randn(2), "jepa.predictors.-2.x": torch.randn(2)}
    out, report = extract_pretrained_encoder_state(sd, {"source": "online"}, {"model.encoder.stem.weight"})
    assert "model.encoder.stem.weight" in out and report.n_encoder_match == 1


def test_encoder_decoder_scope_keeps_decoder():
    sd, report = extract_pretrained_encoder_state(
        _fake_ssl_state_dict(), {"source": "online", "load_scope": "encoder_decoder"}, _TARGET
    )
    assert "model.decoder.0.conv.weight" in sd  # decoder kept under the AMAES-full ablation scope
    assert "model.head_demo.fc.weight" not in sd  # heads still dropped


def test_no_skip_decoder_kernel_zero_pads_skip_half_exactly():
    encoder = torch.randn(2, 2)
    decoder = torch.randn(3, 4, 3, 3, 3)
    raw = {
        "model.encoder.weight": encoder,
        "model.decoder.decoder_conv1.conv1.conv.weight": decoder,
    }
    target_shapes = {
        "model.encoder.weight": encoder.shape,
        "model.decoder.decoder_conv1.conv1.conv.weight": (3, 8, 3, 3, 3),
    }

    filtered, report = extract_pretrained_encoder_state(
        raw,
        {
            "source": "online",
            "load_scope": "encoder_decoder",
            "adapt_decoder_skip_channels": True,
        },
        {"model.encoder.weight"},
        target_state_shapes=target_shapes,
    )

    adapted = filtered["model.decoder.decoder_conv1.conv1.conv.weight"]
    assert torch.equal(adapted[:, :4], decoder)
    assert torch.count_nonzero(adapted[:, 4:]) == 0
    assert report.n_decoder_skip_zero_pad == 1
    assert report.shape_mismatches == []
    assert "decoder skip-channel zero-pad adaptations: 1" in report.format()


def test_no_skip_decoder_kernel_mismatch_stays_strict_without_opt_in():
    raw = {
        "model.encoder.weight": torch.randn(2, 2),
        "model.decoder.decoder_conv1.conv1.conv.weight": torch.randn(3, 4, 3, 3, 3),
    }
    target_shapes = {
        "model.encoder.weight": (2, 2),
        "model.decoder.decoder_conv1.conv1.conv.weight": (3, 8, 3, 3, 3),
    }
    with pytest.raises(RuntimeError, match="incompatible shapes"):
        extract_pretrained_encoder_state(
            raw,
            {"source": "online", "load_scope": "encoder_decoder"},
            {"model.encoder.weight"},
            target_state_shapes=target_shapes,
        )


def test_real_resenc_no_skip_checkpoint_initializes_skip_decoder_function_exactly():
    from asparagus.modules.networks.resenc_unet import ResidualEncoderUNetSSL
    from gardening_tools.modules.networks.resunet import ResidualEncoderUNet

    common = {
        "dimensions": "2D",
        "input_channels": 1,
        "output_channels": 1,
        "kernel_size": 3,
        "stride": 2,
        "features_per_stage": (2, 4, 8, 16, 16, 16),
        "n_blocks_per_stage": (1, 1, 1, 1, 1, 1),
        "n_conv_per_stage_decoder": (1, 1, 1, 1, 1),
    }
    torch.manual_seed(7)
    pretrained = ResidualEncoderUNetSSL(
        **common,
        use_skip_connections=False,
        modality_conditioning=False,
    ).eval()
    torch.manual_seed(11)
    downstream = ResidualEncoderUNet(**common, use_skip_connections=True).eval()

    raw = {f"model.{key}": value for key, value in pretrained.state_dict().items()}
    target_state = downstream.state_dict()
    target_encoder = {f"model.{key}" for key in target_state if key.startswith("encoder.")}
    filtered, report = extract_pretrained_encoder_state(
        raw,
        {
            "source": "online",
            "load_scope": "encoder_decoder",
            "adapt_decoder_skip_channels": True,
            "allow_missing_decoder": False,
        },
        target_encoder,
        target_state_shapes={f"model.{key}": value.shape for key, value in target_state.items()},
    )

    downstream.load_state_dict(
        {key.removeprefix("model."): value for key, value in filtered.items()},
        strict=False,
    )
    assert report.n_decoder_skip_zero_pad == 10
    assert report.shape_mismatches == []

    x = torch.randn(1, 1, 128, 128)
    with torch.no_grad():
        expected = pretrained(x)
        actual = downstream(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_fail_if_no_encoder_keys_loaded():
    with pytest.raises(RuntimeError, match="No encoder keys matched"):
        extract_pretrained_encoder_state(_fake_ssl_state_dict(), {"source": "online"}, {"model.encoder.MISSING"})
    # disabling the guard returns an empty-match dict without raising
    sd, report = extract_pretrained_encoder_state(
        _fake_ssl_state_dict(),
        {"source": "online", "fail_if_no_encoder_keys_loaded": False, "fail_if_missing_encoder_keys": False},
        {"model.encoder.MISSING"},
    )
    assert report.n_encoder_match == 0


def test_partial_encoder_export_fails_loudly_by_default():
    partial = _fake_ssl_state_dict()
    del partial["momentum_model.encoder.stages.0.conv.weight"]
    with pytest.raises(RuntimeError, match=r"encoder is incomplete.*stages\.0\.conv\.weight"):
        extract_pretrained_encoder_state(partial, {"source": "ema"}, _TARGET)

    _, report = extract_pretrained_encoder_state(partial, {"source": "ema", "fail_if_missing_encoder_keys": False}, _TARGET)
    assert report.n_encoder_match == 1 and report.missing_target == 1


def test_unknown_source_and_scope_raise():
    with pytest.raises(ValueError, match="source"):
        extract_pretrained_encoder_state(_fake_ssl_state_dict(), {"source": "bogus"}, _TARGET)
    with pytest.raises(ValueError, match="load_scope"):
        extract_pretrained_encoder_state(_fake_ssl_state_dict(), {"source": "online", "load_scope": "bogus"}, _TARGET)


_SHAPES_OK = {"model.encoder.stem.weight": (4, 1, 3, 3, 3), "model.encoder.stages.0.conv.weight": (8, 4, 3, 3, 3)}


def test_report_states_zero_shape_mismatches_explicitly():
    _, report = extract_pretrained_encoder_state(
        _fake_ssl_state_dict(), {"source": "online"}, _TARGET, target_state_shapes=_SHAPES_OK
    )
    assert report.shape_mismatches == [] and report.n_stem_repeat == 0
    assert "shape_mismatches=0" in report.format()  # explicit, never an ambiguous "Wrong shape"


def test_stem_repeat_not_counted_as_mismatch():
    shapes = dict(_SHAPES_OK, **{"model.encoder.stem.weight": (4, 4, 3, 3, 3)})  # model has 4 input channels
    _, report = extract_pretrained_encoder_state(
        _fake_ssl_state_dict(), {"source": "online"}, _TARGET, target_state_shapes=shapes
    )
    assert report.n_stem_repeat == 1 and report.shape_mismatches == []
    assert "shape_mismatches=0" in report.format()


def test_real_shape_mismatch_reported_with_exact_shapes_and_strict_raises():
    shapes = dict(_SHAPES_OK, **{"model.encoder.stages.0.conv.weight": (16, 4, 3, 3, 3)})  # 8 != 16 out channels
    # strict_shapes default True -> raises, naming the key and both shapes
    with pytest.raises(RuntimeError, match=r"stages.0.conv.weight.*\(8, 4, 3, 3, 3\).*\(16, 4, 3, 3, 3\)"):
        extract_pretrained_encoder_state(_fake_ssl_state_dict(), {"source": "online"}, _TARGET, target_state_shapes=shapes)
    # strict_shapes False -> recorded in the report (key + exact shapes), no raise
    _, report = extract_pretrained_encoder_state(
        _fake_ssl_state_dict(), {"source": "online", "strict_shapes": False}, _TARGET, target_state_shapes=shapes
    )
    assert len(report.shape_mismatches) == 1
    key, ckpt_shape, model_shape = report.shape_mismatches[0]
    assert key == "model.encoder.stages.0.conv.weight" and ckpt_shape == (8, 4, 3, 3, 3) and model_shape == (16, 4, 3, 3, 3)
    assert "shape_mismatches=1" in report.format()


# --------------------------------------------------------------------------- #
# Integration: real downstream ResEnc seg net + synthetic SSL checkpoint
# --------------------------------------------------------------------------- #
def test_integration_transfers_encoder_keeps_head_random():
    from asparagus.modules.networks.resenc_unet import resenc_unet_b

    torch.manual_seed(0)
    pretrained_net = resenc_unet_b(dimensions="3D", input_channels=1, output_channels=2)
    torch.manual_seed(1)
    downstream = resenc_unet_b(dimensions="3D", input_channels=1, output_channels=2)

    # Build a realistic SSL checkpoint from the pretrained net (online + EMA + jepa predictors).
    ssl_ckpt = {f"model.{k}": v for k, v in pretrained_net.state_dict().items()}
    ssl_ckpt.update({f"momentum_model.{k}": v for k, v in pretrained_net.state_dict().items()})
    ssl_ckpt["jepa.predictors.-2.predictor_embed.weight"] = torch.randn(16, 8)
    # Multi-scale supervision reaching the bottleneck: without a level -1 predictor the scope
    # check correctly reads this checkpoint as "stage 5 never trained" and refuses it.
    ssl_ckpt["jepa.predictors.-1.predictor_embed.weight"] = torch.randn(16, 8)

    target = {f"model.{k}" for k in downstream.state_dict() if k.startswith("encoder.")}
    filtered, report = extract_pretrained_encoder_state(ssl_ckpt, {"source": "online", "load_scope": "encoder_only"}, target)
    assert report.n_encoder_match == len(target) and report.missing_target == 0 and report.n_skipped_jepa == 2
    assert report.trained_encoder_stages == report.checkpoint_encoder_stages and report.scope_evidence == "inferred"

    # Snapshot a decoder weight, then load the filtered (encoder-only) dict into the downstream net.
    dec_key = next(k for k in downstream.state_dict() if k.startswith("decoder."))
    dec_before = downstream.state_dict()[dec_key].clone()
    # strip the module `model.` prefix to load the encoder weights into the bare net
    downstream.load_state_dict({k[len("model.") :]: v for k, v in filtered.items()}, strict=False)

    # encoder transferred (equal to pretrained), decoder untouched (still random / != pretrained)
    enc_key = next(k for k in downstream.state_dict() if k.startswith("encoder."))
    assert torch.equal(downstream.state_dict()[enc_key], pretrained_net.state_dict()[enc_key])
    assert torch.equal(downstream.state_dict()[dec_key], dec_before)
    assert not torch.equal(downstream.state_dict()[dec_key], pretrained_net.state_dict()[dec_key])


def test_multi_module_backbone_transfer_keeps_primus_eva_weights():
    """Encoder-only means the declared trainable trunk, not literally only ``encoder.*``."""
    raw = {
        "model.encoder.proj.weight": torch.randn(4, 1, 2, 2, 2),
        "model.eva.blocks.0.weight": torch.randn(4, 4),
        "model.decoder.head.weight": torch.randn(2, 4),
        "jepa.predictors.-1.weight": torch.randn(4, 4),
    }
    target = {"model.encoder.proj.weight", "model.eva.blocks.0.weight"}
    filtered, report = extract_pretrained_encoder_state(
        raw,
        {"source": "online", "load_scope": "encoder_only"},
        target,
        backbone_prefixes=("encoder.", "eva."),
    )
    assert set(filtered) == target
    assert report.n_encoder == report.n_encoder_match == 2
    assert report.missing_target == 0


def test_learned_representation_projection_is_transferable_backbone_state():
    raw = {
        "model.encoder.stem.weight": torch.randn(4, 1, 3, 3, 3),
        "model.h_global_projector.weight": torch.randn(8, 4),
        "model.h_global_projector.bias": torch.randn(8),
        "model.head.weight": torch.randn(2, 8),
    }
    prefixes = ("encoder.", "h_global_projector.", "h_global_norm.")
    target = {
        "model.encoder.stem.weight",
        "model.h_global_projector.weight",
        "model.h_global_projector.bias",
    }
    filtered, report = extract_pretrained_encoder_state(
        raw,
        {"source": "online", "load_scope": "encoder_only"},
        target,
        backbone_prefixes=prefixes,
    )
    assert set(filtered) == target
    assert report.n_encoder_match == 3


def test_jepa_resenc_clsreg_transfer_rejects_legacy_maxpool():
    from asparagus.modules.networks.resenc_unet import ResidualEncoderUNetCLSREG
    from asparagus.pipeline.auto_configuration.pretrained import resolve_pretrained_weights
    from omegaconf import OmegaConf

    model = ResidualEncoderUNetCLSREG(
        dimensions="3D",
        input_channels=1,
        output_channels=1,
        kernel_size=3,
        stride=2,
        features_per_stage=(2, 2, 2, 2, 2, 2),
        n_blocks_per_stage=(1, 1, 1, 1, 1, 1),
        encoder_pool="max",
    )
    cfg = OmegaConf.create(
        {
            "training": {"load_decoder": True},
            "pretrained": {
                "enabled": True,
                "source": "ema",
                "load_scope": "encoder_only",
            },
        }
    )
    raw = {
        "target_encoder.encoder.stem.conv1.conv.weight": torch.randn(2, 1, 3, 3, 3),
        "jepa.predictors.-2.mask_token": torch.randn(1, 1, 8),
    }
    with pytest.raises(RuntimeError, match=r"requires model\.encoder_pool=avg"):
        resolve_pretrained_weights(cfg, model, raw)


def test_old_ssl_head_shape_mismatch_is_skipped_while_encoder_loads():
    from asparagus.modules.lightning_modules.self_supervised import SelfSupervisedModule
    from asparagus.modules.networks.resenc_unet import ResidualEncoderUNetSSL

    model = ResidualEncoderUNetSSL(
        dimensions="3D",
        input_channels=1,
        output_channels=1,
        kernel_size=3,
        stride=2,
        features_per_stage=(2, 2, 2, 2, 2, 2),
        n_blocks_per_stage=(1, 1, 1, 1, 1, 1),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        head_out_dim=16,
        head_hidden_dim=16,
    )
    module = SelfSupervisedModule(model=model, learning_rate=1e-3)
    state_before = module.state_dict()
    encoder_key = next(
        key for key, value in state_before.items() if key.startswith("model.encoder.") and value.is_floating_point()
    )
    head_key = "model.head_demo.net.0.weight"
    encoder_value = torch.full_like(state_before[encoder_key], 0.12345)
    old_head_value = torch.randn(16, 2)

    module.load_state_dict(
        {
            encoder_key: encoder_value,
            head_key: old_head_value,
        },
        strict=False,
    )

    state_after = module.state_dict()
    assert torch.equal(state_after[encoder_key], encoder_value)
    assert state_after[head_key].shape[1] == model.global_feature_dim
    assert torch.equal(state_after[head_key], state_before[head_key])


# --------------------------------------------------------------------------- #
# Config: the pretrained variants compose and expose the expected fields
# --------------------------------------------------------------------------- #
def test_pretrained_finetune_configs_build():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    for name, fn in [("random", lambda a, b: 0), ("version", lambda: "t"), ("eval", eval)]:
        try:
            OmegaConf.register_new_resolver(name, fn)
        except Exception:
            pass
    os.environ.setdefault("ASPARAGUS_DATA", "/tmp")
    os.environ.setdefault("WANDB_ENTITY", "x")
    cfgdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
    base = "projects/fomo26/finetune"
    with initialize_config_dir(version_base="1.2", config_dir=cfgdir):
        scratch = compose(config_name=f"{base}/task1_lesion_scratch")
        assert scratch.pretrained.enabled is False

        # The JEPA-parent finetune lanes were retired with the objective; composing one must fail
        # closed rather than resolve onto the AMAES lane that replaced it.
        with pytest.raises(Exception):
            compose(config_name=f"{base}/task1_lesion_jepa_deepsup")

        amaes = compose(config_name=f"{base}/task3_age_amaes_encoder")
        assert amaes.pretrained.enabled is True and amaes.pretrained.source == "online"


# --------------------------------------------------------------------------- #
# Export-scope contract (JPA-005): a stage the SSL run never trained must not be
# consumed downstream as if it had been pretrained.
# --------------------------------------------------------------------------- #
def _staged_ssl_state_dict(trained_stages=None, n_stages=4):
    state = {}
    for stage in range(n_stages):
        state[f"model.encoder.stages.{stage}.conv.weight"] = torch.ones(2, 2, 3, 3, 3)
        state[f"target_encoder.encoder.stages.{stage}.conv.weight"] = torch.ones(2, 2, 3, 3, 3)
    if trained_stages is not None:
        state["_jepa_trained_encoder_stages"] = torch.tensor(trained_stages)
    return state


def _staged_target(n_stages=4):
    return {f"model.encoder.stages.{stage}.conv.weight" for stage in range(n_stages)}


def test_partially_trained_encoder_is_refused_by_default():
    with pytest.raises(ValueError, match="frozen at random initialisation"):
        extract_pretrained_encoder_state(_staged_ssl_state_dict(trained_stages=3), {"source": "ema"}, _staged_target())


def test_partially_trained_encoder_can_be_accepted_knowingly():
    _, report = extract_pretrained_encoder_state(
        _staged_ssl_state_dict(trained_stages=3),
        {"source": "ema", "allow_partial_encoder": True},
        _staged_target(),
    )
    assert report.trained_encoder_stages == 3
    assert report.checkpoint_encoder_stages == 4
    assert "3 of 4 stages" in report.format()


def test_explicit_full_scope_loads_unchanged():
    _, full = extract_pretrained_encoder_state(_staged_ssl_state_dict(trained_stages=4), {"source": "ema"}, _staged_target())
    assert full.trained_encoder_stages == 4
    assert full.scope_evidence == "explicit"
    assert "4 of 4 stages" in full.format()


def test_a_historical_checkpoint_scope_is_inferred_from_its_jepa_predictors():
    """No scope buffer, but the saved predictors prove which prefix was supervised."""
    state = _staged_ssl_state_dict()
    state["jepa.predictors.-1.predictor_embed.weight"] = torch.randn(4, 4)
    _, report = extract_pretrained_encoder_state(state, {"source": "ema"}, _staged_target())
    assert report.trained_encoder_stages == 4 and report.scope_evidence == "inferred"

    partial = _staged_ssl_state_dict()
    partial["jepa.predictors.-2.predictor_embed.weight"] = torch.randn(4, 4)
    with pytest.raises(ValueError, match="frozen at random initialisation"):
        extract_pretrained_encoder_state(partial, {"source": "ema"}, _staged_target())


def test_a_non_jepa_checkpoint_is_inferred_to_have_trained_the_whole_trunk():
    """A reconstruction/contrastive run drives every stage through the decoder."""
    state = {f"model.encoder.stages.{i}.conv.weight": torch.ones(2, 2, 3, 3, 3) for i in range(4)}
    state.update({f"momentum_model.encoder.stages.{i}.conv.weight": torch.ones(2, 2, 3, 3, 3) for i in range(4)})
    _, report = extract_pretrained_encoder_state(state, {"source": "online"}, _staged_target())
    assert report.trained_encoder_stages == 4 and report.scope_evidence == "inferred"


def test_an_unknown_scope_is_refused_and_never_reported_as_all():
    """V-1: absence of scope metadata is not evidence that every stage was pretrained."""
    with pytest.raises(ValueError, match="UNKNOWN encoder scope"):
        extract_pretrained_encoder_state(_staged_ssl_state_dict(), {"source": "ema"}, _staged_target())

    _, report = extract_pretrained_encoder_state(
        _staged_ssl_state_dict(),
        {"source": "ema", "allow_unknown_encoder_scope": True},
        _staged_target(),
    )
    assert report.trained_encoder_stages is None and report.scope_evidence == "unknown"
    rendered = report.format()
    assert "unknown of 4 stages" in rendered
    assert "all" not in rendered.split("pretrained encoder scope:")[1].splitlines()[0]


def test_an_unstaged_trunk_has_no_partial_scope_question():
    state = {"model.encoder.proj.weight": torch.randn(4, 4), "jepa.predictors.-1.weight": torch.randn(4, 4)}
    _, report = extract_pretrained_encoder_state(state, {"source": "online"}, {"model.encoder.proj.weight"})
    assert report.scope_evidence == "not_applicable" and report.checkpoint_encoder_stages == 0
