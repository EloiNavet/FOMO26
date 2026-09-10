"""Warmup must be accumulation-independent, and collapse health must need a real cohort.

Both regressions were observed on Jean Zay:

* jobs 533085/533090 ran the same nominal 2% warmup with accumulation 8 and 4 and got 300 and 900
  optimizer steps respectively, because ``training.warmup_epochs`` divides optimizer steps by
  microbatches-per-epoch;
* job 533085 logged dead_dim_fraction=1.0 and effective_rank=0 for all 800 steps at microbatch 1,
  which is the "<2 samples" sentinel, not a collapse.
"""

import math
import pytest
import torch
from asparagus.functional.lr_scheduling import simple_warmup_cosine_decay_schedule
from asparagus.functional.representations import (
    embedding_health_metrics,
    representation_health_failures,
)

HORIZON = 187_500
WARMUP_RATIO = 0.02
EXPECTED_WARMUP = int(round(HORIZON * WARMUP_RATIO))  # 3750


def _schedule(warmup_steps=None, warmup_epochs=0, optimizer_steps_per_epoch=1000):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    return simple_warmup_cosine_decay_schedule(
        optimizer,
        warmup_epochs,
        optimizer_steps_per_epoch,
        1.0,
        -1,
        HORIZON,
        warmup_steps=warmup_steps,
    )


def _lr_at(scheduler, step):
    return scheduler.lr_lambdas[0](step)


# ------------------------- 1. accumulation independence -------------------------


@pytest.mark.parametrize("accumulation", [1, 2, 4, 8])
def test_explicit_warmup_steps_is_identical_for_every_accumulation(accumulation):
    """The corrected contract is stated in optimizer steps, so accumulation cannot move it."""
    # Whatever the geometry, an epoch is 1000 optimizer steps; only the microbatch count changes.
    scheduler = _schedule(warmup_steps=EXPECTED_WARMUP, optimizer_steps_per_epoch=1000)
    assert _lr_at(scheduler, EXPECTED_WARMUP - 1) < 1.0
    assert _lr_at(scheduler, EXPECTED_WARMUP) == pytest.approx(1.0, abs=1e-9)
    # Halfway through warmup the multiplier is halfway between the 0.001 floor and 1.0.
    half = _lr_at(scheduler, EXPECTED_WARMUP // 2)
    assert half == pytest.approx(0.001 + 0.999 * 0.5, abs=1e-3)


def test_legacy_epoch_warmup_reproduces_the_observed_accumulation_dependence():
    """Pin the defect the fix removes, so nobody reintroduces the epoch-denominated contract."""
    # Calibration geometry: 300 optimizer steps per epoch.
    # accumulation 8 resolved to warmup_epochs=1 -> 300 steps; accumulation 4 to 3 -> 900 steps.
    accum8 = _schedule(warmup_epochs=1, optimizer_steps_per_epoch=300)
    accum4 = _schedule(warmup_epochs=3, optimizer_steps_per_epoch=300)
    assert _lr_at(accum8, 400) == pytest.approx(1.0, abs=1e-5), "accumulation 8 was already at peak by step 400"
    assert _lr_at(accum4, 800) == pytest.approx(0.001 + 0.999 * (800 / 900), abs=1e-6)
    # The two geometries disagree at step 800: exactly the reported symptom.
    assert _lr_at(accum8, 800) != pytest.approx(_lr_at(accum4, 800), abs=1e-6)


def test_warmup_steps_takes_precedence_over_warmup_epochs():
    scheduler = _schedule(warmup_steps=EXPECTED_WARMUP, warmup_epochs=1, optimizer_steps_per_epoch=300)
    assert _lr_at(scheduler, 300) < 1.0, "the 1-epoch legacy warmup must not win"
    assert _lr_at(scheduler, EXPECTED_WARMUP) == pytest.approx(1.0, abs=1e-9)


def test_warmup_epochs_still_applies_when_warmup_steps_is_none():
    """Backward compatibility: existing configs keep their exact schedule."""
    scheduler = _schedule(warmup_steps=None, warmup_epochs=2, optimizer_steps_per_epoch=300)
    assert _lr_at(scheduler, 600) == pytest.approx(1.0, abs=1e-9)


# ------------------------- 2. resume continuity -------------------------


def test_warmup_is_a_pure_function_of_global_step_so_resume_is_exact():
    """A resumed run must land on the same multiplier as an uninterrupted one."""
    uninterrupted = _schedule(warmup_steps=EXPECTED_WARMUP)
    resumed = _schedule(warmup_steps=EXPECTED_WARMUP)
    for step in (0, 1, 800, EXPECTED_WARMUP - 1, EXPECTED_WARMUP, EXPECTED_WARMUP + 10, 32_000):
        assert _lr_at(uninterrupted, step) == _lr_at(resumed, step)
    # And it is monotone through the warmup boundary, so a resume cannot jump.
    values = [_lr_at(uninterrupted, s) for s in range(EXPECTED_WARMUP - 3, EXPECTED_WARMUP + 1)]
    assert values == sorted(values)


# ------------------------- 3. health cohort validity -------------------------


def test_microbatch_one_reports_an_invalid_cohort_rather_than_collapse():
    metrics = embedding_health_metrics(torch.randn(1, 64))
    assert float(metrics["health_valid"]) == 0.0
    assert float(metrics["cohort_size"]) == 1.0
    # The sentinel must never be treated as a real failure.
    assert representation_health_failures(metrics) == ()


def test_schema_is_identical_for_valid_and_invalid_cohorts():
    """DDP requires every rank to log the same keys regardless of its local cohort size."""
    assert set(embedding_health_metrics(torch.randn(1, 64))) == set(embedding_health_metrics(torch.randn(128, 64)))


def test_healthy_cohort_passes_the_gate():
    torch.manual_seed(0)
    metrics = embedding_health_metrics(torch.randn(128, 64))
    assert float(metrics["health_valid"]) == 1.0
    assert float(metrics["cohort_size"]) == 128.0
    assert float(metrics["finite_fraction"]) == 1.0
    assert float(metrics["dead_dim_fraction"]) == 0.0
    assert float(metrics["effective_rank_normalized"]) > 0.25
    assert representation_health_failures(metrics) == ()


def test_collapsed_cohort_fails_the_gate():
    collapsed = torch.ones(128, 64) * 3.0  # every sample identical -> no inter-sample variation
    metrics = embedding_health_metrics(collapsed)
    assert float(metrics["health_valid"]) == 1.0
    assert float(metrics["dead_dim_fraction"]) == 1.0
    assert float(metrics["effective_rank_normalized"]) == 0.0
    failures = representation_health_failures(metrics)
    assert "dead_dim_fraction" in failures
    assert "effective_rank_normalized" in failures


def test_rank_one_cohort_is_detected_as_low_rank_not_as_a_healthy_one():
    """A single shared direction is real collapse, not a sentinel."""
    torch.manual_seed(0)
    direction = torch.randn(1, 64)
    cohort = torch.randn(128, 1) * direction
    metrics = embedding_health_metrics(cohort)
    assert float(metrics["health_valid"]) == 1.0
    assert float(metrics["effective_rank_normalized"]) < 0.25
    assert "effective_rank_normalized" in representation_health_failures(metrics)


def test_non_finite_embeddings_are_reported_and_do_not_crash_the_decomposition():
    cohort = torch.randn(128, 64)
    cohort[0, 0] = float("nan")
    metrics = embedding_health_metrics(cohort)
    assert math.isclose(float(metrics["finite_fraction"]), 1 - 1 / (128 * 64), rel_tol=1e-6)
    # Rank statistics are undefined on a non-finite cohort, so it is reported invalid rather
    # than raising out of torch.linalg.svdvals or being scored as a collapse.
    assert float(metrics["health_valid"]) == 0.0
    assert representation_health_failures(metrics) == ()
