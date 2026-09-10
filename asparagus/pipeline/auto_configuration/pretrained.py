"""Load an SSL-pretrained encoder into a downstream model.

SSL checkpoints contain ``model.*`` (online backbone), ``target_encoder.*`` (current encoder-only
EMA target), legacy ``momentum_model.*`` targets, and objective-only predictor heads. Downstream finetuning only
wants the *backbone*. This utility selects the online or EMA backbone, keeps backbone-only weights by
default, drops objective-only keys, remaps the chosen source into the downstream ``model.*``
namespace, and fails loudly if no backbone key matches the downstream model. CNNs use ``encoder.*``;
architectures such as Primus explicitly declare additional trunk prefixes (``eva.*``). Positional-embedding
interpolation and shape filtering remain in ``BaseModule.load_state_dict``; this stage never silently keeps
unrelated weights.

Input-channel (stem) adaptation is performed *here*, on every key this stage classifies as
``stem_repeat``. That classification is exactly the set of shape differences this stage already
excuses from ``strict_shapes``, so adapting the same set closes the gap between what is excused and
what is actually transferred. ``BaseModule.load_state_dict`` only ever adapted the single key named by
``model.stem_weight_name``, which several downstream networks
(``ResidualEncoderUNetCLSREG``, ``UNet``/``UNetCLSREG``) do not declare at all: their multichannel stems
were excused here, dropped there, and silently left at random initialisation while the loader still
logged a successful transfer. Adapting centrally fixes every architecture at once, covers aliased state
keys (``conv.weight`` and ``all_modules.0.weight`` name one ``nn.Parameter``), and is idempotent -- once
adapted, the shape matches the target, so the ``BaseModule`` stem branch can no longer fire.
"""

import logging
import re
from dataclasses import dataclass, field

_ONLINE_PREFIX = "model."
_EMA_PREFIXES = ("target_encoder.", "momentum_model.")
_VALID_SOURCES = ("online", "ema", "ema_if_available")
_VALID_SCOPES = ("encoder_only", "encoder_decoder")


@dataclass
class PretrainedLoadReport:
    checkpoint: str
    requested_source: str
    source_used: str
    scope: str
    n_encoder: int = 0  # encoder keys extracted (remapped to model.encoder.*)
    n_encoder_match: int = 0  # of those, names present in the downstream model
    n_decoder: int = 0  # decoder-body keys extracted under encoder_decoder scope
    n_decoder_match: int = 0
    n_skipped_jepa: int = 0  # jepa.* / predictors (objective-only)
    n_skipped_other_source: int = 0  # the non-selected encoder copy + unrelated top-level state
    n_skipped_decoder_head: int = 0  # decoder / heads / FiLM / modality under encoder_only
    missing_target: int = 0  # downstream encoder keys not covered by the checkpoint
    missing_decoder_target: int = 0  # downstream decoder-body keys not covered (task heads excluded)
    n_stem_repeat: int = 0  # single-channel stems expanded to the downstream modality count here
    n_decoder_skip_zero_pad: int = 0  # no-skip decoder kernels expanded with zero-initialised skip channels
    n_skipped_task_head: int = 0  # output heads intentionally reinitialised for downstream classes
    # real shape mismatches against the downstream model: (key, checkpoint_shape, model_shape)
    shape_mismatches: list = field(default_factory=list)
    # Export scope: how many leading encoder stages the SSL run actually trained, and how
    # many the checkpoint contains. ``None`` means UNKNOWN — the run recorded no scope and
    # the checkpoint holds no evidence from which one can be derived. Unknown is never
    # reported or treated as "all stages pretrained".
    trained_encoder_stages: int | None = None
    checkpoint_encoder_stages: int = 0
    # How ``trained_encoder_stages`` was obtained: "explicit" (recorded by the SSL run),
    # "inferred" (derived from sufficient saved evidence) or "unknown" (no evidence).
    scope_evidence: str = "unknown"

    def format(self) -> str:
        lines = [
            "Pretrained checkpoint loading report:",
            f"  checkpoint: {self.checkpoint}",
            f"  source: {self.requested_source} -> {self.source_used}",
            f"  scope: {self.scope}",
            f"  encoder keys extracted: {self.n_encoder} (matching downstream model: {self.n_encoder_match})",
            f"  decoder keys extracted: {self.n_decoder} (matching downstream model: {self.n_decoder_match})",
            f"  skipped JEPA-only keys: {self.n_skipped_jepa}",
            f"  skipped other-source/objective keys: {self.n_skipped_other_source}",
            f"  skipped decoder/head/non-encoder keys: {self.n_skipped_decoder_head}",
            f"  downstream encoder keys missing from checkpoint: {self.missing_target}",
            f"  downstream decoder-body keys missing from checkpoint: {self.missing_decoder_target}",
            f"  stem input-channel expansions applied: {self.n_stem_repeat}",
            f"  decoder skip-channel zero-pad adaptations: {self.n_decoder_skip_zero_pad}",
            f"  task-head keys reinitialised: {self.n_skipped_task_head}",
            f"  pretrained encoder scope: {'unknown' if self.trained_encoder_stages is None else self.trained_encoder_stages} "
            f"of {self.checkpoint_encoder_stages or 'unknown'} stages "
            f"(evidence: {self.scope_evidence})",
            f"  shape_mismatches={len(self.shape_mismatches)}",
        ]
        for key, ckpt_shape, model_shape in self.shape_mismatches:
            lines.append(f"    - {key}: checkpoint {tuple(ckpt_shape)} != model {tuple(model_shape)}")
        return "\n".join(lines)


def _strip_compile(state_dict: dict) -> dict:
    """Remove the torch.compile ``_orig_mod.`` segment (online backbone is compiled at pretrain time)."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def _has_ema_encoder(state_dict: dict) -> bool:
    return any(k.startswith(prefix + "encoder.") for prefix in _EMA_PREFIXES for k in state_dict)


def _ema_prefix(state_dict: dict) -> str | None:
    for prefix in _EMA_PREFIXES:
        if any(key.startswith(prefix + "encoder.") for key in state_dict):
            return prefix
    return None


def _shape_diff_kind(ckpt_shape, model_shape) -> str:
    """Classify a checkpoint-vs-model shape pair: match / stem_repeat / mismatch."""
    ckpt_shape, model_shape = tuple(ckpt_shape), tuple(model_shape)
    if ckpt_shape == model_shape:
        return "match"
    # stem input-channel repeat: identical except dim 1, where the checkpoint has a single channel.
    # ``adapt_stem_input_channels`` expands it to the downstream modality count. Not a real mismatch.
    if (
        len(ckpt_shape) == len(model_shape) >= 2
        and ckpt_shape[1] == 1
        and model_shape[1] > 1
        and ckpt_shape[:1] == model_shape[:1]
        and ckpt_shape[2:] == model_shape[2:]
    ):
        return "stem_repeat"
    return "mismatch"


def adapt_stem_input_channels(value, model_shape):
    """Expand a single-input-channel stem kernel to the downstream modality count.

    Uses the arithmetic ``BaseModule.load_state_dict`` has always applied to
    ``model.<stem_weight_name>``: repeat the single input channel across the target modalities
    and divide by their count. Dividing keeps the pre-activation scale the pretrained kernel was
    trained to produce, since the repeated channels sum over the input dimension.

    Only a 1 -> N widening is supported; every other channel change is refused so it stays a real
    shape mismatch rather than being silently reshaped.
    """
    model_shape = tuple(model_shape)
    source_shape = tuple(value.shape)
    if _shape_diff_kind(source_shape, model_shape) != "stem_repeat":
        raise RuntimeError(
            f"refusing to adapt input channels {source_shape} -> {model_shape}: only a single-channel "
            "stem may be expanded to the downstream modality count"
        )
    channels = model_shape[1]
    repeats = [1] * value.ndim
    repeats[1] = channels
    adapted = value.repeat(*repeats) / channels
    if tuple(adapted.shape) != model_shape:  # pragma: no cover - guarded by _shape_diff_kind above
        raise RuntimeError(f"stem adaptation produced {tuple(adapted.shape)}, expected {model_shape}")
    return adapted


def _is_decoder_task_head(key: str) -> bool:
    lowered = key.lower()
    return key.startswith("model.decoder.") and any(
        token in lowered
        for token in (
            ".head.",
            ".heads.",
            "auxiliary_heads",
            "seg_layer",
            "out_conv",
            "output_conv",
            "final_conv",
        )
    )


def _can_zero_pad_decoder_skip_channels(key: str, ckpt_shape, model_shape) -> bool:
    """Whether a no-skip decoder kernel can exactly initialize its skip-enabled counterpart.

    ResEnc decoders concatenate ``[upsampled_decoder, encoder_skip]``. A no-skip convolution
    therefore maps exactly to the first half of a skip-enabled convolution; zeroing the second
    half preserves the pretrained function until finetuning learns to use skip features.
    """
    ckpt_shape, model_shape = tuple(ckpt_shape), tuple(model_shape)
    return (
        key.startswith("model.decoder.decoder_conv")
        and key.endswith("weight")
        and len(ckpt_shape) >= 3
        and len(ckpt_shape) == len(model_shape)
        and model_shape[0] == ckpt_shape[0]
        and model_shape[1] == 2 * ckpt_shape[1]
        and model_shape[2:] == ckpt_shape[2:]
    )


_STAGE_KEY = re.compile(r"(?:^|\.)encoder\.stages\.(\d+)\.")


def _count_encoder_stages(state_dict: dict) -> int:
    """Highest ``encoder.stages.<i>`` index present, +1. ``0`` when the trunk is not staged."""
    indices = {int(match.group(1)) for key in state_dict if (match := _STAGE_KEY.search(key))}
    return max(indices) + 1 if indices else 0


_JEPA_PREDICTOR_KEY = re.compile(r"(?:^|\.)jepa\.predictors\.(-?\d+)\.")
# ``momentum_model`` is also used by non-JEPA contrastive objectives, so its
# presence cannot establish that a checkpoint is JEPA.  JEPA checkpoints use
# the objective/predictor namespace or the dedicated target encoder.
_JEPA_MARKER_PREFIXES = ("jepa.", "target_encoder.")


def _resolve_pretrained_scope(state_dict: dict, total: int) -> tuple[int | None, str]:
    """Determine how many leading encoder stages a checkpoint actually had trained.

    Returns ``(stages, evidence)``. ``evidence`` is:

    * ``"explicit"`` — the SSL run recorded ``_jepa_trained_encoder_stages``;
    * ``"inferred"`` — the run predates that buffer, but the checkpoint *proves* its scope:
      either it carries no JEPA marker at all (a reconstruction objective drives the whole
      trunk through the decoder, so every stage was trained), or it carries the per-level
      JEPA predictors, whose levels determine the supervised prefix exactly as
      ``SelfSupervisedModule._trained_encoder_stage_count`` computes it at training time;
    * ``"unknown"`` — a JEPA checkpoint with neither the buffer nor the predictors. Absence
      of evidence is not evidence of full pretraining, so this is *not* reported as "all".
    """
    if total == 0:
        # No indexable ``encoder.stages`` list (tokenised ViT trunks, flat backbones): the
        # whole trunk always runs, so there is no partial-scope question to answer.
        return 0, "not_applicable"
    recorded = state_dict.get("_jepa_trained_encoder_stages")
    if recorded is not None:
        return int(recorded.item() if hasattr(recorded, "item") else recorded), "explicit"
    if not any(key.startswith(prefix) for prefix in _JEPA_MARKER_PREFIXES for key in state_dict):
        # No JEPA anywhere: a reconstruction/contrastive SSL run trained the whole trunk.
        return total, "inferred"
    levels = {int(match.group(1)) for key in state_dict if (match := _JEPA_PREDICTOR_KEY.search(key))}
    if levels and total > 0:
        deepest = max(level if level >= 0 else total + level for level in levels)
        return int(min(deepest + 1, total)), "inferred"
    return None, "unknown"


def _check_pretrained_scope(state_dict: dict, pretrained_cfg, report: PretrainedLoadReport) -> None:
    """Refuse to consume encoder stages the SSL run never trained — or never proved it did.

    Training only a prefix of the encoder is a legitimate choice (a shallow JEPA feature
    level leaves deeper stages frozen at initialisation). What must not happen is a
    downstream run loading such a checkpoint and treating every stage as pretrained.

    A checkpoint whose scope cannot be established is reported as ``unknown`` and refused by
    default, because "we did not record it" is not the same claim as "all of it was
    trained". ``pretrained.allow_unknown_encoder_scope=true`` is the explicit exploratory
    override.
    """
    # A backbone whose `stages` is a derived view (MedViT3D) exposes no `encoder.stages.<i>`
    # keys, so the count recorded by the SSL run takes precedence over the key scan.
    recorded_total = state_dict.get("_jepa_encoder_stage_total")
    total = int(recorded_total.item() if hasattr(recorded_total, "item") else recorded_total or 0)
    total = total or _count_encoder_stages(state_dict)
    report.checkpoint_encoder_stages = total
    trained, evidence = _resolve_pretrained_scope(state_dict, total)
    report.trained_encoder_stages = trained
    report.scope_evidence = evidence
    if evidence == "unknown":
        if bool(pretrained_cfg.get("allow_unknown_encoder_scope", False)):
            logging.warning(
                "PRETRAINED SCOPE UNKNOWN: this checkpoint records no trained-encoder scope and carries "
                "no evidence from which one can be derived, so it is NOT known whether all %d encoder "
                "stages were pretrained. Loading anyway because pretrained.allow_unknown_encoder_scope=true; "
                "results from this checkpoint must not be reported as fully pretrained.",
                total,
            )
            return
        raise ValueError(
            "Pretrained checkpoint has an UNKNOWN encoder scope: it records no "
            "`_jepa_trained_encoder_stages` buffer and carries no JEPA predictors from which the "
            "supervised prefix could be derived, so it is unproven that all "
            f"{total} encoder stages were pretrained. Absence of the metadata is not evidence of full "
            "pretraining. Re-export the checkpoint from a run that records its scope, or set "
            "pretrained.allow_unknown_encoder_scope=true to load it knowingly for exploratory work."
        )
    if trained is None or trained <= 0 or total <= 0 or trained >= total:
        return
    if bool(pretrained_cfg.get("allow_partial_encoder", False)):
        logging.warning(
            "Pretrained checkpoint trained only %d of %d encoder stages; stages %d.. are at "
            "their SSL initialisation. Proceeding because pretrained.allow_partial_encoder=true.",
            trained,
            total,
            trained,
        )
        return
    raise ValueError(
        f"Pretrained checkpoint trained only {trained} of {total} encoder stages: stages "
        f"{trained}..{total - 1} were frozen at random initialisation during SSL and are NOT "
        "pretrained. Supervise the bottleneck during pretraining, truncate the downstream "
        "encoder to the pretrained scope, or set pretrained.allow_partial_encoder=true to "
        "accept randomly initialised deep stages knowingly."
    )


def _resolve_source(requested: str, state_dict: dict) -> str:
    if requested not in _VALID_SOURCES:
        raise ValueError(f"Unknown pretrained.source={requested!r}; expected one of {_VALID_SOURCES}.")
    if requested == "online":
        return "online"
    if requested == "ema":
        if not _has_ema_encoder(state_dict):
            raise RuntimeError(
                "pretrained.source=ema but the checkpoint has no target_encoder.encoder.* or "
                "momentum_model.encoder.* keys "
                "(no EMA/target encoder was saved). Use source=online or ema_if_available."
            )
        return "ema"
    return "ema" if _has_ema_encoder(state_dict) else "online"  # ema_if_available


def extract_pretrained_encoder_state(
    raw_state_dict,
    pretrained_cfg,
    target_encoder_keys,
    checkpoint_path: str = "",
    target_state_shapes=None,
    backbone_prefixes=("encoder.",),
):
    """Filter+remap an SSL checkpoint to the downstream encoder namespace.

    :param raw_state_dict: the Lightning checkpoint ``state_dict`` (model.* / momentum_model.* / jepa.*).
    :param pretrained_cfg: mapping with ``source``, ``load_scope``, ``strict_shapes``,
        ``fail_if_no_encoder_keys_loaded``, and ``fail_if_missing_encoder_keys``.
    :param target_encoder_keys: set of downstream module encoder keys, e.g. ``{"model.encoder.<...>"}``.
    :param target_state_shapes: optional ``{key: shape}`` of the downstream module; when given, the report
        records exact shape mismatches (stem input-channel differences are classified as expected stem-repeats),
        and ``strict_shapes`` raises on any real mismatch.
    :returns: ``(filtered_state_dict, PretrainedLoadReport)`` — keys remapped to the ``model.<...>`` namespace.
    """
    state_dict = _strip_compile(raw_state_dict)
    requested = str(pretrained_cfg.get("source", "ema_if_available"))
    scope = str(pretrained_cfg.get("load_scope", "encoder_only"))
    if scope not in _VALID_SCOPES:
        raise ValueError(f"Unknown pretrained.load_scope={scope!r}; expected one of {_VALID_SCOPES}.")
    source = _resolve_source(requested, state_dict)
    src_prefix = _ema_prefix(state_dict) if source == "ema" else _ONLINE_PREFIX
    other_prefixes = (_ONLINE_PREFIX,) if source == "ema" else _EMA_PREFIXES
    backbone_prefixes = tuple(str(prefix) for prefix in backbone_prefixes)
    if not backbone_prefixes or any(not prefix.endswith(".") for prefix in backbone_prefixes):
        raise ValueError("backbone_prefixes must be a non-empty sequence of dotted module prefixes.")
    keep_rel = backbone_prefixes if scope == "encoder_only" else (*backbone_prefixes, "decoder.")

    report = PretrainedLoadReport(checkpoint=str(checkpoint_path), requested_source=requested, source_used=source, scope=scope)
    _check_pretrained_scope(state_dict, pretrained_cfg, report)
    filtered = {}
    for key, value in state_dict.items():
        if key.startswith("jepa."):
            report.n_skipped_jepa += 1
            continue
        if not key.startswith(src_prefix) or any(key.startswith(prefix) for prefix in other_prefixes):
            # the non-selected encoder copy, predictors, loss buffers, etc.
            report.n_skipped_other_source += 1
            continue
        rel = key[len(src_prefix) :]  # e.g. "encoder.stages...", "decoder...", "encoder_films...", "head_demo..."
        if any(rel.startswith(p) for p in keep_rel):
            new_key = _ONLINE_PREFIX + rel  # downstream namespace
            filtered[new_key] = value
            if any(rel.startswith(prefix) for prefix in backbone_prefixes):
                report.n_encoder += 1
            elif rel.startswith("decoder."):
                report.n_decoder += 1
        else:
            report.n_skipped_decoder_head += 1

    target_encoder_keys = set(target_encoder_keys)
    report.n_encoder_match = sum(1 for k in filtered if k in target_encoder_keys)
    target_decoder_keys = set()
    if target_state_shapes is not None:
        target_decoder_keys = {
            key for key in target_state_shapes if key.startswith("model.decoder.") and not _is_decoder_task_head(key)
        }
    report.n_decoder_match = sum(1 for key in filtered if key in target_decoder_keys)
    report.missing_target = sum(1 for k in target_encoder_keys if k not in filtered)

    if target_state_shapes is not None:
        for key, value in list(filtered.items()):
            if key not in target_state_shapes:
                continue
            kind = _shape_diff_kind(value.shape, target_state_shapes[key])
            if kind == "stem_repeat":
                # Adapt here rather than excusing the difference and hoping a later stage handles
                # it: this is the only place that sees every such key together with its target shape.
                filtered[key] = adapt_stem_input_channels(value, target_state_shapes[key])
                report.n_stem_repeat += 1
            elif kind == "mismatch":
                if bool(pretrained_cfg.get("adapt_decoder_skip_channels", False)) and _can_zero_pad_decoder_skip_channels(
                    key, value.shape, target_state_shapes[key]
                ):
                    adapted = value.new_zeros(target_state_shapes[key])
                    adapted[:, : value.shape[1], ...] = value
                    filtered[key] = adapted
                    report.n_decoder_skip_zero_pad += 1
                    continue
                if bool(pretrained_cfg.get("allow_missing_head", True)) and _is_decoder_task_head(key):
                    filtered.pop(key)
                    report.n_skipped_task_head += 1
                    # Task output heads are deliberately excluded from ``target_decoder_keys``.
                    # Do not decrement the decoder-body match count for a head that was never
                    # included in that count.
                    if key in target_decoder_keys:
                        report.n_decoder_match = max(0, report.n_decoder_match - 1)
                    continue
                report.shape_mismatches.append((key, tuple(value.shape), tuple(target_state_shapes[key])))

    if scope == "encoder_decoder":
        report.missing_decoder_target = sum(1 for key in target_decoder_keys if key not in filtered)

    if bool(pretrained_cfg.get("fail_if_no_encoder_keys_loaded", True)) and report.n_encoder_match == 0:
        raise RuntimeError(
            f"No encoder keys matched the downstream model (source={source}, scope={scope}): extracted "
            f"{report.n_encoder} encoder keys, 0 matched the {len(target_encoder_keys)} downstream encoder keys. "
            "Wrong checkpoint or incompatible architecture?"
        )
    if bool(pretrained_cfg.get("fail_if_missing_encoder_keys", True)) and report.missing_target:
        missing = sorted(target_encoder_keys - set(filtered))
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise RuntimeError(
            f"Pretrained encoder is incomplete: {report.missing_target} of {len(target_encoder_keys)} downstream "
            f"encoder keys are missing ({preview}{suffix}). Set fail_if_missing_encoder_keys=false only for an "
            "explicit partial-transfer ablation."
        )
    fail_if_missing_decoder = bool(pretrained_cfg.get("fail_if_missing_decoder_keys", True)) and not bool(
        pretrained_cfg.get("allow_missing_decoder", False)
    )
    if scope == "encoder_decoder" and fail_if_missing_decoder and report.missing_decoder_target:
        missing = sorted(target_decoder_keys - set(filtered))
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise RuntimeError(
            f"Pretrained decoder body is incomplete: {report.missing_decoder_target} of "
            f"{len(target_decoder_keys)} downstream decoder keys are missing ({preview}{suffix}). "
            "Only task-specific output heads may be reinitialised."
        )
    if bool(pretrained_cfg.get("strict_shapes", True)) and report.shape_mismatches:
        raise RuntimeError(
            f"strict_shapes=true and {len(report.shape_mismatches)} encoder weight(s) have incompatible shapes:\n"
            + "\n".join(f"  {k}: checkpoint {c} != model {m}" for k, c, m in report.shape_mismatches)
        )
    if bool(pretrained_cfg.get("print_key_report", True)):
        logging.info(report.format())
    return filtered, report


def resolve_pretrained_weights(cfg, model, resolved_weights):
    """Entrypoint helper: if ``cfg.pretrained.enabled``, filter the resolved SSL checkpoint to the
    downstream encoder. Returns ``(weights, load_decoder)`` — ``load_decoder`` is forced False under
    the encoder_only scope so the downstream decoder/head stay randomly initialised (fair transfer).
    """
    from asparagus.pipeline.auto_configuration.checkpoint import load_checkpoint_state_dict

    pcfg = cfg.get("pretrained", None)
    # reg/cls finetune configs don't define training.load_decoder (only seg does); default True.
    load_decoder = bool(cfg.training.get("load_decoder", True))
    if not pcfg or not bool(pcfg.get("enabled", False)):
        return resolved_weights, load_decoder

    raw = resolved_weights
    explicit_path = pcfg.get("checkpoint_path", None)
    if explicit_path:
        raw = load_checkpoint_state_dict(explicit_path)
    if raw is None:
        raise ValueError(
            "pretrained.enabled=true but no checkpoint was resolved. Set `checkpoint_path` (or "
            "`pretrained.checkpoint_path`) to the SSL checkpoint to transfer from."
        )
    if any(str(key).startswith("jepa.") for key in raw) and hasattr(model, "encoder_pool"):
        encoder_pool = str(getattr(model, "encoder_pool")).strip().lower()
        if encoder_pool != "avg":
            raise RuntimeError(
                "JEPA ResEnc classification/regression transfer requires model.encoder_pool=avg: "
                "the pretrained/EMA encoder downsamples with AvgPool, while the legacy downstream "
                f"default is {encoder_pool!r}. Refusing a numerically non-isomorphic transfer."
            )
    model_state = model.state_dict()
    backbone_prefixes = tuple(getattr(model, "pretrained_backbone_prefixes", ("encoder.",)))
    target_encoder_keys = {
        f"model.{key}" for key in model_state if any(key.startswith(prefix) for prefix in backbone_prefixes)
    }
    target_state_shapes = {f"model.{k}": tuple(v.shape) for k, v in model_state.items()}
    weights, _report = extract_pretrained_encoder_state(
        raw,
        pcfg,
        target_encoder_keys,
        checkpoint_path=str(explicit_path or "<resolved checkpoint>"),
        target_state_shapes=target_state_shapes,
        backbone_prefixes=backbone_prefixes,
    )
    if str(pcfg.get("load_scope", "encoder_only")) == "encoder_only":
        load_decoder = False
    else:
        load_decoder = True
    return weights, load_decoder
