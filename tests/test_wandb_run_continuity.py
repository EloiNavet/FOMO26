"""A lane's segments must land in ONE W&B run, or fail rather than split the history.

`resume="allow"` silently opens a *new* run when the previous id cannot be resolved. On a chained
multi-segment lane that turns one 32k-step history into several disjoint runs, which is only
noticeable after the fact and cannot be repaired offline.
"""

import pytest
from asparagus.pipeline.auto_configuration.logging import logging as build_loggers


def _loggers(**overrides):
    kwargs = dict(
        ckpt_wandb_id=None,
        ckpt_mlflow_id=None,
        log_file_name="job",
        run_dir="/tmp/does-not-need-to-exist",
        version="v1",
        wandb_experiment="exp",
        wandb_logging=True,
        log_to_stdout=False,
    )
    kwargs.update(overrides)
    return build_loggers(**kwargs)


def _wandb_logger(loggers):
    return next(logger for logger in loggers if hasattr(logger, "_wandb_init"))


def test_continuation_without_a_resolvable_run_id_fails_closed():
    with pytest.raises(ValueError, match="W&B run continuity was required"):
        _loggers(ckpt_wandb_id=None, wandb_require_run_continuity=True)


def test_continuation_with_a_run_id_resumes_must():
    loggers = _loggers(ckpt_wandb_id="abc123", wandb_require_run_continuity=True)
    assert _wandb_logger(loggers)._wandb_init["resume"] == "must"


def test_default_behaviour_is_unchanged_for_existing_runs():
    """Every recipe that does not opt in keeps the historical resume="allow" semantics."""
    loggers = _loggers(ckpt_wandb_id="abc123")
    assert _wandb_logger(loggers)._wandb_init["resume"] == "allow"


def test_fresh_run_without_continuity_requirement_does_not_resume():
    loggers = _loggers(ckpt_wandb_id=None)
    assert _wandb_logger(loggers)._wandb_init["resume"] is None
