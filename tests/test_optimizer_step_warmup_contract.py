"""Regression tests for accumulation-independent optimizer-step warmup."""

import copy
import pytest
import torch
from asparagus.functional.lr_scheduling import simple_warmup_cosine_decay_schedule

HORIZON = 250_000
WARMUP_STEPS = 5_000


def _schedule(*, warmup_steps=None, warmup_epochs=0, optimizer_steps_per_epoch=1_000):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    return simple_warmup_cosine_decay_schedule(
        optimizer,
        warmup_epochs,
        optimizer_steps_per_epoch,
        1.0,
        max_steps=HORIZON,
        warmup_steps=warmup_steps,
    )


def _lr_at(scheduler, step):
    return scheduler.lr_lambdas[0](step)


@pytest.mark.parametrize("accumulation", [1, 2, 4, 8])
def test_explicit_optimizer_step_warmup_is_accumulation_independent(accumulation):
    scheduler = _schedule(warmup_steps=WARMUP_STEPS, optimizer_steps_per_epoch=1_000 // accumulation)
    assert _lr_at(scheduler, WARMUP_STEPS - 1) < 1.0
    assert _lr_at(scheduler, WARMUP_STEPS) == pytest.approx(1.0, abs=1e-9)


def test_explicit_warmup_takes_precedence_over_legacy_epochs():
    scheduler = _schedule(warmup_steps=WARMUP_STEPS, warmup_epochs=1, optimizer_steps_per_epoch=300)
    assert _lr_at(scheduler, 300) < 1.0
    assert _lr_at(scheduler, WARMUP_STEPS) == pytest.approx(1.0, abs=1e-9)


def test_legacy_epoch_warmup_remains_backward_compatible():
    scheduler = _schedule(warmup_epochs=2, optimizer_steps_per_epoch=300)
    assert _lr_at(scheduler, 600) == pytest.approx(1.0, abs=1e-9)


def test_warmup_is_a_pure_function_of_global_step_across_resume():
    uninterrupted = _schedule(warmup_steps=WARMUP_STEPS)
    resumed = _schedule(warmup_steps=WARMUP_STEPS)
    for step in (0, 1, 800, WARMUP_STEPS - 1, WARMUP_STEPS, WARMUP_STEPS + 10, 32_000):
        assert _lr_at(uninterrupted, step) == _lr_at(resumed, step)


def test_optimizer_and_scheduler_state_continue_identically_across_cutoff():
    def make_pair():
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD([parameter], lr=1.0, momentum=0.9)
        scheduler = simple_warmup_cosine_decay_schedule(
            optimizer,
            warmup_epochs=0,
            steps_per_epoch=1_000,
            cosine_period_ratio=1.0,
            max_steps=HORIZON,
            warmup_steps=WARMUP_STEPS,
        )
        return parameter, optimizer, scheduler

    uninterrupted_parameter, uninterrupted_optimizer, uninterrupted_scheduler = make_pair()
    uninterrupted_parameter.grad = torch.tensor([0.25])
    uninterrupted_optimizer.step()
    uninterrupted_scheduler.step(31_999)

    resumed_parameter, resumed_optimizer, resumed_scheduler = make_pair()
    resumed_parameter.data.copy_(uninterrupted_parameter.data)
    resumed_optimizer.load_state_dict(copy.deepcopy(uninterrupted_optimizer.state_dict()))
    resumed_scheduler.load_state_dict(copy.deepcopy(uninterrupted_scheduler.state_dict()))

    for step in (32_000, 32_001):
        uninterrupted_parameter.grad = torch.tensor([0.25])
        resumed_parameter.grad = torch.tensor([0.25])
        uninterrupted_optimizer.step()
        resumed_optimizer.step()
        uninterrupted_scheduler.step(step)
        resumed_scheduler.step(step)
        assert uninterrupted_parameter.detach().item() == pytest.approx(resumed_parameter.detach().item(), abs=0.0)
        assert uninterrupted_optimizer.state_dict() == resumed_optimizer.state_dict()
        assert uninterrupted_scheduler.state_dict() == resumed_scheduler.state_dict()
        assert uninterrupted_optimizer.param_groups[0]["lr"] == resumed_optimizer.param_groups[0]["lr"]
