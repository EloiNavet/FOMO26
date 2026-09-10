import math
from torch.optim.lr_scheduler import LambdaLR
from typing import Any, Dict, List


def _is_no_decay_param(name: str, param) -> bool:
    """Parameters that should not be weight-decayed.

    Following MAE/ViT practice we exclude all 1-D parameters (biases and the
    scale/shift of every norm layer) as well as positional and learned tokens,
    which are sensitive to shrinkage.
    """
    if param.ndim <= 1:
        return True
    lowered = name.lower()
    return any(tag in lowered for tag in ("pos_embed", "register_token", "cls_token", "mask_token"))


def build_param_groups(named_parameters, weight_decay: float, separate_encoder_decoder: bool = False) -> List[Dict[str, Any]]:
    """Build optimizer parameter groups with weight-decay-aware bucketing.

    Norm/bias/positional parameters land in ``*_no_decay`` groups with
    ``weight_decay=0``. When ``separate_encoder_decoder`` is set the groups are
    additionally split into encoder/decoder roles (decoder = anything under
    ``model.decoder``; everything else, including the stem, is encoder) so the
    sawtooth schedule can drive them with different warmups. Group order is
    deterministic: encoder before decoder, decay before no-decay.
    """
    buckets: Dict[tuple, list] = {}
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        clean = name.replace("_orig_mod.", "")  # if params are compiled we laugh and do this
        if separate_encoder_decoder:
            role = "decoder" if "model.decoder" in clean else "encoder"
        else:
            role = "params"
        decay = not _is_no_decay_param(clean, param)
        buckets.setdefault((role, decay), []).append(param)

    role_order = ["encoder", "decoder"] if separate_encoder_decoder else ["params"]
    if separate_encoder_decoder:
        # All hail the almighty assert which saved my ass. Twice.
        has_encoder = any(role == "encoder" for role, _ in buckets)
        has_decoder = any(role == "decoder" for role, _ in buckets)
        assert has_encoder and has_decoder, "Encoder or decoder parameters not found."

    groups: List[Dict[str, Any]] = []
    for role in role_order:
        for decay in (True, False):
            params = buckets.get((role, decay))
            if not params:
                continue
            groups.append(
                {
                    "params": params,
                    "name": role if decay else f"{role}_no_decay",
                    "weight_decay": weight_decay if decay else 0.0,
                }
            )
    return groups


def separate_encoder_decoder_weights(named_parameters) -> List[Dict[str, Any]]:
    """Separate the encoder and decoder weights of a model (no weight-decay split).

    Retained for backwards compatibility; prefer :func:`build_param_groups`.
    """
    encoder_params = []
    decoder_params = []
    for name, param in named_parameters:
        name = name.replace("_orig_mod.", "")  # if params are compilled we laugh and do this
        if "model.decoder" in name:
            decoder_params.append(param)
        else:
            # Default to encoder params for any other parameters (e.g., stem)
            encoder_params.append(param)

    # All hail the almighty assert which saved my ass. Twice.
    assert len(encoder_params) > 0 and len(decoder_params) > 0, "Encoder or decoder parameters not found."
    return [
        {"params": encoder_params, "name": "encoder"},
        {"params": decoder_params, "name": "decoder"},
    ]


def sawtooth_warmup_cosine_decay_schedule(
    optimizer,
    decoder_warmup_epochs,
    warmup_epochs,
    steps_per_epoch,
    cosine_period_ratio,  # cosine_half_period is from max to min
    max_epochs,
):
    """
    Phase 1: Decoder warmup, encoder frozen
    Phase 2: Both encoder and decoder warmup
    Phase 3: Cosine annealing for both
    """
    assert max_epochs > 0 and steps_per_epoch > 0, "max_epochs and steps_per_epoch must be greater than 0"
    print(f"Using separate warmup: decoder for {decoder_warmup_epochs} epochs, then both for {warmup_epochs} epochs")

    decoder_warmup_steps = int(decoder_warmup_epochs * steps_per_epoch)
    encoder_decoder_warmup_steps = int(warmup_epochs * steps_per_epoch)
    total_warmup_steps = decoder_warmup_steps + encoder_decoder_warmup_steps
    cosine_steps = int(cosine_period_ratio * (max_epochs * steps_per_epoch - total_warmup_steps))

    def encoder_phase1_lambda(_step):
        return 0.0  # Encoder frozen during phase 1

    def decoder_phase1_lambda(step):
        return 0.999 * step / decoder_warmup_steps + 0.001

    # LambdaLR needs one lambda per param group; assign by role so it is robust
    # to extra (e.g. no-decay) groups rather than relying on positional order.
    group_names = [group.get("name", "") for group in optimizer.param_groups]
    assert any(name.startswith("encoder") for name in group_names) and any(
        name.startswith("decoder") for name in group_names
    ), f"Expected encoder/decoder param groups, got {group_names}."

    def cosine_lambda(step):
        if cosine_steps <= 0:
            return 1.0
        progress = min(max((step - total_warmup_steps) / cosine_steps, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def encoder_lambda(step):
        if step < decoder_warmup_steps:
            return encoder_phase1_lambda(step)
        if step < total_warmup_steps:
            progress = (step - decoder_warmup_steps) / max(1, encoder_decoder_warmup_steps)
            return 0.001 + 0.999 * progress
        return cosine_lambda(step)

    def decoder_lambda(step):
        if step < decoder_warmup_steps:
            return decoder_phase1_lambda(step)
        if step < total_warmup_steps:
            return 1.0
        return cosine_lambda(step)

    lr_lambdas = [decoder_lambda if name.startswith("decoder") else encoder_lambda for name in group_names]
    return LambdaLR(optimizer, lr_lambda=lr_lambdas)


def simple_warmup_cosine_decay_schedule(
    optimizer,
    warmup_epochs,
    steps_per_epoch,
    cosine_period_ratio,
    max_epochs=-1,
    max_steps=-1,
    warmup_steps=None,
):
    """
    Phase 1: Warmup for both encoder and decoder
    Phase 2: Cosine annealing for both

    ``warmup_steps`` states the warmup directly in optimizer steps and takes precedence over
    ``warmup_epochs``. Prefer it: an epoch-denominated warmup is only accumulation-independent if
    the caller derived the epoch count in optimizer steps, and the historical config expression did
    not, which silently scaled the warmup with accumulate_grad_batches. ``warmup_epochs`` remains
    the default so existing configs keep their exact schedule.
    """
    assert warmup_epochs >= 0, "Warmup epochs must be greater than or equal to 0."
    assert cosine_period_ratio > 0, "Cosine period ratio must be greater than 0."
    assert steps_per_epoch > 0, "Steps per epoch must be greater than 0."
    assert max_epochs > 0 or max_steps > 0, "Either max_epochs or max_steps must be greater than 0."

    # cosine_half_period is from max to min
    if max_epochs > 0:
        max_steps = max_epochs * steps_per_epoch

    if warmup_steps is not None:
        assert int(warmup_steps) >= 0, "Warmup steps must be greater than or equal to 0."
        requested_warmup_steps = int(warmup_steps)
        warmup_source = f"{requested_warmup_steps} optimizer steps (explicit warmup_steps)"
    else:
        requested_warmup_steps = int(warmup_epochs * steps_per_epoch)
        warmup_source = f"{warmup_epochs} epochs x {steps_per_epoch} optimizer steps/epoch"
    total_warmup_steps = min(requested_warmup_steps, max(0, max_steps - 1))
    cosine_steps = max(1, int(cosine_period_ratio * (max_steps - total_warmup_steps)))

    print(f"Using warmup from {warmup_source} -> effective_warmup_steps={total_warmup_steps}")
    if total_warmup_steps != requested_warmup_steps:
        print(f"Clipped warmup from {requested_warmup_steps} steps to leave one decay step in a short run")
    print(f"Cosine decay for {cosine_steps} steps after warmup")

    def lr_lambda(step):
        if total_warmup_steps > 0 and step < total_warmup_steps:
            return 0.001 + 0.999 * (step / max(1, total_warmup_steps))
        progress = min(max((step - total_warmup_steps) / max(1, cosine_steps), 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def cosine_decay_schedule(optimizer, steps_per_epoch, cosine_period_ratio, max_epochs=-1, max_steps=-1):
    """
    Phase 1: Cosine annealing for both encoder and decoder
    """
    # cosine_half_period is from max to min
    if max_epochs > 0:
        max_steps = max_epochs * steps_per_epoch
    cosine_steps = int(cosine_period_ratio * max_steps)
    assert cosine_steps > 0, "Cosine steps must be greater than 0 for cosine decay schedule."

    def lr_lambda(step):
        progress = min(max(step / max(1, cosine_steps), 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
