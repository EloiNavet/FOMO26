"""One canonical checkpoint-selection contract shared by segmentation, classification and regression.

Downstream runs previously carried two independent sources of truth for the same decision.
``cfg.test_checkpoint`` named the prediction file, while ``cfg.testing.checkpoint`` (falling back
to ``cfg.test_checkpoint``) chose the weights actually handed to ``trainer.test``. Nothing kept
them in agreement, so a config that set ``testing.checkpoint: current`` while inheriting
``test_checkpoint: best`` from the finetune defaults evaluated end-of-fit weights and wrote
the result to ``<task>__<split>__best.json``. Six architecture-comparison configs did exactly
that; that campaign is not part of the public code release, but the failure mode is a
property of the two-field design, not of those six files. That filename is then parsed back
as the checkpoint label by ``asparagus/pipeline/run/eval_box.py``, so every collector downstream
reported ``best`` for a run evaluated on ``current``.

The resolution here is deliberately strict rather than a precedence rule: when both fields are
present and disagree there is no way to tell which one the author meant, and guessing is what
produced mislabelled results in the first place. A config must say the same thing twice or say it
once.

The returned role is the *only* value permitted to name a checkpoint anywhere downstream -- the
weights loaded for ``trainer.test``, the prediction filename, the prediction metadata, the run
manifest, the metrics record and the collectors all read it, so a run's checkpoint identity is
the same string everywhere it appears.
"""

from __future__ import annotations

VALID_CHECKPOINT_ROLES = ("best", "last", "current")

LEGACY_FIELD = "test_checkpoint"
CANONICAL_FIELD = "testing.checkpoint"


def resolve_test_checkpoint_role(cfg) -> str:
    """Return the single checkpoint role for this run, or fail loudly.

    ``best`` and ``last`` load the checkpoint saved by the corresponding callback. ``current``
    evaluates the in-memory end-of-fit weights and loads nothing.
    """
    legacy = cfg.get(LEGACY_FIELD, None)
    testing = cfg.get("testing", None)
    canonical = testing.get("checkpoint", None) if testing is not None else None

    if canonical is not None and legacy is not None and str(canonical) != str(legacy):
        raise ValueError(
            f"conflicting checkpoint selection: {CANONICAL_FIELD}={canonical!r} but "
            f"{LEGACY_FIELD}={legacy!r}. These name the same decision and must agree -- the "
            f"resolved role drives the evaluated weights, the prediction filename, the run "
            f"manifest and every collector. Set both to the intended role, or set only one."
        )

    role = canonical if canonical is not None else legacy
    if role is None:
        raise ValueError(
            f"no checkpoint selection: set {CANONICAL_FIELD} (or the legacy {LEGACY_FIELD}) to one of {VALID_CHECKPOINT_ROLES}"
        )

    role = str(role)
    if role not in VALID_CHECKPOINT_ROLES:
        raise ValueError(f"{CANONICAL_FIELD}/{LEGACY_FIELD} must be one of {VALID_CHECKPOINT_ROLES}, got {role!r}")
    return role


def prediction_filename(cfg, role: str) -> str:
    """Prediction filename for a resolved role.

    Built from the same ``role`` that selects the evaluated weights, so the name can be parsed
    back into a truthful checkpoint identity.
    """
    return f"{cfg.test_task}__{cfg.data.test_split}__{role}.json"


def _best_checkpoint_path(callback) -> str:
    """The file the *monitored* callback selected as best.

    Requires a monitor: on a callback with ``monitor=None`` Lightning writes the most recent
    checkpoint into ``best_model_path`` (see ``_save_none_monitor_checkpoint``), so reading that
    attribute off an unmonitored callback would report the latest checkpoint as the best one.
    """
    if getattr(callback, "monitor", None) is None:
        raise ValueError(
            "the 'best' checkpoint callback has no monitor, so it does not select a best "
            "checkpoint -- Lightning records the most recent write in best_model_path when "
            "monitor is None. Configure a monitor, or evaluate 'last' instead."
        )
    path = getattr(callback, "best_model_path", "")
    if not path:
        raise ValueError(
            "no 'best' checkpoint was saved, so it cannot be evaluated. Refusing to fall back "
            "to the end-of-fit weights, which would be reported as 'best'."
        )
    return path


def _last_checkpoint_path(callback) -> str:
    """The latest periodic checkpoint.

    'last' is defined here as *the most recent checkpoint this callback wrote*, and Lightning
    records that in two different attributes depending on how the callback was built:

    * ``save_last=True`` -> the dedicated ``last_model_path``;
    * ``monitor=None``   -> ``best_model_path``, because ``_save_none_monitor_checkpoint``
      assigns the just-written path there regardless of the attribute's name. This is the
      shape the finetune entrypoints construct (``every_n_epochs``, ``save_top_k=1``,
      ``filename="last"``).

    A monitored callback without ``save_last`` is refused rather than guessed at: its
    ``best_model_path`` means "best by metric", which is not what 'last' promises.
    """
    if getattr(callback, "save_last", False):
        path = getattr(callback, "last_model_path", "")
        if path:
            return path
        raise ValueError(
            "the 'last' checkpoint callback declares save_last but recorded no last_model_path, "
            "so no last checkpoint was written and none can be evaluated."
        )
    if getattr(callback, "monitor", None) is not None:
        raise ValueError(
            "the 'last' checkpoint callback is monitored and does not save_last, so it exposes "
            "no path to the latest checkpoint -- best_model_path would be the best by metric, "
            "not the last. Set save_last=True on it, or drop its monitor."
        )
    path = getattr(callback, "best_model_path", "")
    if not path:
        raise ValueError(
            "no 'last' checkpoint was saved, so it cannot be evaluated. Refusing to fall back "
            "to the end-of-fit weights, which would be reported as 'last'."
        )
    return path


def resolve_checkpoint_path(role: str, best_ckpt_callback, last_ckpt_callback) -> str | None:
    """The exact file passed to ``trainer.test`` for a resolved role, or ``None``.

    ``current`` returns ``None``: Lightning then tests the weights already in memory, and no
    checkpoint file is involved -- so nothing downstream may attach one. ``best`` and ``last``
    return the exact path recorded by the matching callback, and fail closed when no checkpoint
    was written rather than silently falling back to the end-of-fit weights.

    See ``asparagus.pipeline.run.evaluation_identity`` for the full role semantics; this is the
    only function that turns a role into a path.
    """
    if role == "current":
        return None
    if role == "best":
        return _best_checkpoint_path(best_ckpt_callback)
    if role == "last":
        return _last_checkpoint_path(last_ckpt_callback)
    raise ValueError(f"unknown checkpoint role {role!r}; expected one of {VALID_CHECKPOINT_ROLES}")
