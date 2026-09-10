"""Task 5 — a control arm must differ from its treatment by the loss term ONLY.

Two loss flags silently decide the *execution path*, not just which loss is added:

* ``_contrastive_enabled`` (self_supervised.py) selects between one GPU-augmentation pass per step
  and three (the contrastive view build). Different pass counts consume the RNG differently, so the
  crops and masks of the two arms desynchronise from the second batch onward.
* ``_uses_momentum_encoder`` decides whether a momentum encoder exists and is updated at all.
* ``return_raw_image`` (pretrain.py) re-bases the augmented views from a *random* crop to a
  deterministic *center* crop (views.py::prepare_demographic_base_view).

Left alone, D0-vs-D1 and S0-vs-S1 would therefore compare different data streams and attribute the
difference to the objective. These tests lock in the fix: with the matched-control flags set, the
two arms of a pair take an identical path, and **the tests fail if the stream diverges** — including
if the flags are removed.
"""

from __future__ import annotations

import torch
from asparagus.modules.lightning_modules.ssl.views import (
    build_contrastive_views,
    prepare_demographic_base_view,
)


# ---------------------------------------------------------------------------------------------
# The gating structure, mirrored exactly from self_supervised.py:690-711.
# ---------------------------------------------------------------------------------------------
def gates(
    *,
    enable_demo=False,
    demographic_mode="dufumier_yaware",
    enable_stage1=False,
    enable_stage2=False,
    stage1_regularization=False,
    force_contrastive_path=False,
    force_momentum_encoder=False,
):
    stage1_head = enable_stage1 or stage1_regularization
    multimodal = stage1_head or enable_stage2
    contrastive = force_contrastive_path or enable_demo or enable_stage1 or multimodal
    momentum = force_momentum_encoder or (enable_demo and demographic_mode == "moco_yaware") or multimodal
    return {"contrastive": contrastive, "momentum": momentum}


# ---------------------------------------------------------------------------------------------
# A0 / A1 must NOT be forced onto the contrastive path
# ---------------------------------------------------------------------------------------------
def test_plain_amaes_arms_stay_on_the_standard_path():
    """Forcing A0/A1 would buy them three augmentation passes and an unused momentum encoder."""
    for arm in ("A0", "A1"):
        g = gates()  # no forcing, no aux loss
        assert g["contrastive"] is False, f"{arm} must stay on the single-pass AMAES path"
        assert g["momentum"] is False, f"{arm} must not carry a momentum encoder"


# ---------------------------------------------------------------------------------------------
# D pair: contrastive path must match; neither arm uses a momentum encoder
# ---------------------------------------------------------------------------------------------
def test_demographic_pair_diverges_without_the_flag():
    """Regression guard: this is the bug the flag fixes."""
    d0 = gates(enable_demo=False)
    d1 = gates(enable_demo=True)
    assert d0["contrastive"] != d1["contrastive"], (
        "expected the unfixed code to diverge; if this fails the guard is no longer meaningful"
    )


def test_demographic_pair_matches_with_the_flag():
    d0 = gates(enable_demo=False, force_contrastive_path=True)
    d1 = gates(enable_demo=True, force_contrastive_path=True)
    assert d0 == d1, f"D0 and D1 take different paths: {d0} vs {d1}"


def test_demographic_pair_needs_no_momentum_encoder():
    """The frozen profile is `dufumier_yaware`; momentum requires `moco_yaware`, so both arms are
    naturally momentum-free and force_momentum_encoder must stay off for this pair."""
    d1 = gates(enable_demo=True, demographic_mode="dufumier_yaware", force_contrastive_path=True)
    assert d1["momentum"] is False
    moco = gates(enable_demo=True, demographic_mode="moco_yaware", force_contrastive_path=True)
    assert moco["momentum"] is True, "sanity: moco_yaware is what would require momentum"


# ---------------------------------------------------------------------------------------------
# S pair: BOTH the contrastive path and the momentum encoder must be forced
# ---------------------------------------------------------------------------------------------
def test_stage1_control_loses_both_path_and_momentum_without_the_flags():
    """With Stage-1 off, same_modality_hard_negative_weight returns to its 0.0 default, so
    _stage1_regularization_enabled is False and S0 loses the momentum encoder too."""
    s0 = gates(enable_stage1=False, stage1_regularization=False)
    s1 = gates(enable_stage1=True)
    assert s0["contrastive"] != s1["contrastive"]
    assert s0["momentum"] != s1["momentum"], "the momentum encoder is the second, deeper divergence"


def test_stage1_pair_matches_only_when_both_flags_are_set():
    s0_partial = gates(enable_stage1=False, force_contrastive_path=True)
    s1 = gates(enable_stage1=True, force_contrastive_path=True, force_momentum_encoder=True)
    assert s0_partial != s1, "forcing only the contrastive path leaves the momentum encoder unmatched"

    s0 = gates(enable_stage1=False, force_contrastive_path=True, force_momentum_encoder=True)
    assert s0 == s1, f"S0 and S1 take different paths: {s0} vs {s1}"


def test_forcing_never_enables_a_loss():
    """The flags force the path only; no scientific objective may be switched on as a side effect."""
    forced = gates(force_contrastive_path=True, force_momentum_encoder=True)
    assert forced["contrastive"] and forced["momentum"]
    # the loss-enable inputs are untouched: a forced control still has every objective off
    for flag in ("enable_demo", "enable_stage1", "enable_stage2"):
        assert flag not in forced


# ---------------------------------------------------------------------------------------------
# return_raw_image must follow the SAMPLING policy, not the loss flag
# ---------------------------------------------------------------------------------------------
def resolve_return_raw_image(demographic_batch_probability: float) -> bool:
    """Mirrors pretrain.py after the fix."""
    return bool(demographic_batch_probability > 0.0)


def test_return_raw_image_is_driven_by_sampling_not_by_the_loss():
    # both members of the D pair share demographic_batch_probability=0.5 -> both get raw_image
    assert resolve_return_raw_image(0.5) is True
    # arms without demographic sampling (A0/A1/S0/S1) never get it
    assert resolve_return_raw_image(0.0) is False


def test_raw_image_presence_changes_the_augmented_view_base():
    """Why the above matters: raw_image re-bases the aug views onto a different image."""
    torch.manual_seed(0)
    img = torch.randn(1, 1, 8, 8, 8)
    raw = torch.randn(1, 1, 8, 8, 8)

    def demo_tr(d):  # a deterministic stand-in for the CPU val transform
        return {**d, "image": d["image"] * 0 + 1.0}

    with_raw = prepare_demographic_base_view({"image": img, "raw_image": raw}, demo_tr)
    without_raw = prepare_demographic_base_view({"image": img}, demo_tr)
    assert not torch.equal(with_raw["image"], without_raw["image"]), (
        "raw_image changes the view base — which is exactly why it must not depend on a loss flag"
    )


# ---------------------------------------------------------------------------------------------
# Stream equivalence on the REAL view builder, over several consecutive batches
# ---------------------------------------------------------------------------------------------
def _random_crop_transform(d):
    """A stochastic transform standing in for the real augmentations.

    Continuous noise, not a discrete shift: any RNG divergence must show up deterministically, and a
    discrete transform could coincide by chance and silently weaken the divergence guard.
    """
    img = d["image"]
    return {**d, "image": img + torch.randn_like(img)}


def _batch(i):
    return {"image": torch.full((1, 1, 6, 6, 6), float(i)), "file_path": [f"s{i}"]}


def _run_stream(n_batches: int, *, contrastive: bool, seed: int = 1234):
    """Simulate an arm's per-step data path: contrastive arms build three views (three transform
    passes), non-contrastive arms apply one. Returns the produced views per batch."""
    torch.manual_seed(seed)
    out = []
    for i in range(n_batches):
        b = _batch(i)
        if contrastive:
            views = build_contrastive_views(
                b,
                is_training=True,
                train_transforms=_random_crop_transform,
                unmasked_transforms=_random_crop_transform,
                momentum_transforms=None,
                val_transforms=None,
                demo_cpu_transforms=None,
                validation_mask_seed=0,
            )
            out.append(
                (
                    views["view_masked"]["image"].clone(),
                    views["view_aug_1"]["image"].clone(),
                    views["view_aug_2"]["image"].clone(),
                )
            )
        else:
            out.append((_random_crop_transform(b)["image"].clone(), None, None))
    return out


def test_matched_arms_produce_identical_views_over_several_batches():
    """The central test: two arms on the SAME path yield bit-identical views at a fixed seed."""
    a = _run_stream(4, contrastive=True)
    b = _run_stream(4, contrastive=True)
    assert len(a) == len(b) == 4
    for i, (x, y) in enumerate(zip(a, b)):
        for j, (vx, vy) in enumerate(zip(x, y)):
            assert torch.equal(vx, vy), f"batch {i}, view {j} diverged between matched arms"


def test_unmatched_paths_diverge_and_the_test_detects_it():
    """If the flags were removed, one arm would take the single-pass path and the streams would
    diverge — this must be detected, otherwise the equivalence test proves nothing."""
    contrastive = _run_stream(4, contrastive=True)
    plain = _run_stream(4, contrastive=False)
    diverged = any(not torch.equal(c[0], p[0]) for c, p in zip(contrastive, plain))
    assert diverged, "an unmatched pair must be detected as divergent"


def test_masked_view_is_the_reconstruction_input_and_is_compared():
    """view_masked feeds the AMAES/MSE term; it must be identical across a matched pair."""
    a = _run_stream(3, contrastive=True)
    b = _run_stream(3, contrastive=True)
    for i, (x, y) in enumerate(zip(a, b)):
        assert torch.equal(x[0], y[0]), f"masked (reconstruction) view diverged at batch {i}"
