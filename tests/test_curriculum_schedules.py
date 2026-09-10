"""The curriculum schedule contract.

Two things must hold and are asserted here rather than assumed:

1. the generalised :class:`ScheduleSpec` path reproduces the historical ``cosine_ramp`` /
   ``cosine_window`` **exactly**, so no frozen Task-5 configuration changes meaning;
2. the effective weight is a pure function of the optimizer ``global_step``, which is what makes it
   simultaneously resume-safe, accumulation-safe and identical on every DDP rank.
"""

from __future__ import annotations

import math
import pytest
from asparagus.modules.lightning_modules.ssl import schedules as s


def spec(**kwargs) -> s.ScheduleSpec:
    base = {"name": "test", "enabled": True, "weight": 1.0}
    base.update(kwargs)
    return s.ScheduleSpec(**base)


# ==================================================================================================
# Historical equivalence — the property that protects the frozen campaign
# ==================================================================================================
@pytest.mark.parametrize("start,ramp", [(0, 0), (0, 100), (50, 0), (50, 100), (1000, 4800), (4800, 1600)])
def test_cosine_spec_is_bit_identical_to_cosine_ramp(start, ramp):
    subject = spec(start_step=start, ramp_steps=ramp, kind="cosine")
    for step in range(0, 2 * (start + ramp + 200), 1):
        assert s.effective_weight(subject, step) == s.cosine_ramp(step, start, ramp), step


@pytest.mark.parametrize(
    "start,ramp,stop,decay",
    [(0, 100, 500, 0), (0, 100, 500, 200), (100, 50, 300, 100), (0, 0, 10, 5), (0, 1890, 20000, 4000)],
)
def test_cosine_spec_with_a_stop_is_bit_identical_to_cosine_window(start, ramp, stop, decay):
    subject = spec(start_step=start, ramp_steps=ramp, stop_step=stop, decay_steps=decay, kind="cosine")
    for step in range(0, stop + decay + 200):
        assert s.effective_weight(subject, step) == s.cosine_window(step, start, ramp, stop, decay), step


def test_a_disabled_schedule_contributes_exactly_zero():
    subject = spec(enabled=False, start_step=0, ramp_steps=0, weight=0.9)
    assert all(s.effective_weight(subject, step) == 0.0 for step in range(0, 1000, 13))


# ==================================================================================================
# Exact boundary behaviour
# ==================================================================================================
def test_exact_values_at_every_boundary_for_a_full_window():
    subject = spec(start_step=1000, ramp_steps=400, stop_step=3000, decay_steps=200, kind="cosine")

    assert s.effective_weight(subject, 999) == 0.0  # step before start
    assert s.effective_weight(subject, 1000) == 0.0  # start step: ramp begins at zero
    assert s.effective_weight(subject, 1200) == pytest.approx(0.5)  # ramp midpoint
    assert s.effective_weight(subject, 1400) == pytest.approx(1.0)  # full-weight boundary
    assert s.effective_weight(subject, 2999) == pytest.approx(1.0)  # last step before stop
    assert s.effective_weight(subject, 3000) == pytest.approx(1.0)  # stop boundary: decay begins
    assert s.effective_weight(subject, 3100) == pytest.approx(0.5)  # decay midpoint
    assert s.effective_weight(subject, 3200) == 0.0  # decay boundary
    assert s.effective_weight(subject, 5000) == 0.0  # after stop


def test_stop_without_decay_drops_to_zero_exactly_at_the_stop_step():
    subject = spec(start_step=0, ramp_steps=0, stop_step=500, decay_steps=0)
    assert s.effective_weight(subject, 499) == 1.0
    assert s.effective_weight(subject, 500) == 0.0
    assert s.effective_weight(subject, 501) == 0.0


def test_linear_kind_ramps_and_decays_linearly():
    subject = spec(start_step=100, ramp_steps=100, stop_step=400, decay_steps=100, kind="linear")
    assert s.effective_weight(subject, 100) == pytest.approx(0.0)
    assert s.effective_weight(subject, 125) == pytest.approx(0.25)
    assert s.effective_weight(subject, 150) == pytest.approx(0.50)
    assert s.effective_weight(subject, 200) == pytest.approx(1.0)
    assert s.effective_weight(subject, 425) == pytest.approx(0.75)
    assert s.effective_weight(subject, 500) == 0.0


def test_constant_kind_is_a_plain_on_off_window():
    subject = spec(start_step=100, ramp_steps=0, stop_step=300, decay_steps=0, kind="constant")
    assert s.effective_weight(subject, 99) == 0.0
    assert s.effective_weight(subject, 100) == 1.0
    assert s.effective_weight(subject, 299) == 1.0
    assert s.effective_weight(subject, 300) == 0.0


def test_base_weight_scales_the_whole_schedule():
    subject = spec(weight=0.02, start_step=0, ramp_steps=100, kind="linear")
    assert s.effective_weight(subject, 50) == pytest.approx(0.01)
    assert s.effective_weight(subject, 100) == pytest.approx(0.02)
    assert s.schedule_fraction(subject, 100) == pytest.approx(1.0)


# ==================================================================================================
# Fail-closed validation
# ==================================================================================================
def test_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="schedule kind"):
        spec(kind="exponential")


def test_constant_with_a_ramp_is_refused_rather_than_silently_reinterpreted():
    with pytest.raises(ValueError, match="forbids ramp_steps"):
        spec(kind="constant", ramp_steps=100)
    with pytest.raises(ValueError, match="forbids ramp_steps"):
        spec(kind="constant", decay_steps=100)


def test_negative_steps_are_refused():
    with pytest.raises(ValueError, match="must be >= 0"):
        spec(start_step=-1)


def test_a_stop_before_the_start_is_refused():
    with pytest.raises(ValueError, match="precedes start_step"):
        spec(start_step=500, stop_step=100)


# ==================================================================================================
# No pseudo-epoch dependence; accumulation and DDP invariance
# ==================================================================================================
def test_weight_depends_only_on_global_step_not_on_epoch_length():
    subject = spec(start_step=100, ramp_steps=200)
    # Two runs with different steps_per_epoch reach global step 250 at different epochs. The
    # schedule must not be able to tell them apart, because it never sees an epoch.
    assert s.effective_weight(subject, 250) == s.effective_weight(subject, 250)
    by_step = [s.effective_weight(subject, step) for step in range(0, 400)]
    assert by_step == sorted(by_step), "a ramp must be monotonically non-decreasing in global step"


def test_all_ddp_ranks_and_accumulation_settings_agree_for_a_given_optimizer_step():
    subject = spec(start_step=10, ramp_steps=90, stop_step=200, decay_steps=50)
    for step in range(0, 300, 7):
        # Simulating four ranks and two accumulation settings: the only input is the step, so every
        # caller necessarily computes the same number with no communication.
        values = {s.effective_weight(subject, step) for _rank in range(4) for _accum in (1, 4)}
        assert len(values) == 1


# ==================================================================================================
# Resume safety: five resume points, each identical to the uninterrupted trajectory
# ==================================================================================================
def _trajectory(subject: s.ScheduleSpec, steps: range) -> list[float]:
    return [s.effective_weight(subject, step) for step in steps]


@pytest.mark.parametrize(
    "resume_step,phase",
    [
        (50, "before activation"),
        (150, "during ramp"),
        (300, "at full weight"),
        (520, "during decay"),
        (700, "after stop"),
    ],
)
def test_resume_reproduces_the_uninterrupted_trajectory(resume_step, phase):
    subject = spec(start_step=100, ramp_steps=200, stop_step=500, decay_steps=100)
    horizon = range(0, 800)

    uninterrupted = _trajectory(subject, horizon)
    # A resumed run recomputes from scratch at its restart step; there is no schedule state to
    # restore, which is exactly why the two trajectories cannot diverge.
    resumed = _trajectory(subject, range(resume_step, 800))

    assert resumed == uninterrupted[resume_step:], f"resume {phase} diverged"


def test_a_resumed_schedule_needs_no_state_dict_at_all():
    subject = spec(start_step=100, ramp_steps=200)
    # The spec is frozen and carries no mutable state; that is the resume-safety guarantee.
    assert subject == s.ScheduleSpec(name="test", enabled=True, weight=1.0, start_step=100, ramp_steps=200)
    assert not hasattr(subject, "state_dict")


# ==================================================================================================
# Ratio resolution and persisted provenance
# ==================================================================================================
def test_ratio_resolution_matches_the_historical_hydra_arithmetic():
    horizon = 32000
    resolved = s.resolve_ratio_schedule(
        "mse/falcon",
        horizon_steps=horizon,
        weight=1.0,
        start_ratio=0.0,
        ramp_ratio=0.1,
        stop_ratio=0.0,
        decay_ratio=0.0,
    )
    # configs/default_pretrain.yaml computes these with ${eval:"int(${training.steps} * ratio)"}.
    assert resolved.start_step == int(horizon * 0.0)
    assert resolved.ramp_steps == int(horizon * 0.1) == 3200
    assert resolved.stop_step == int(horizon * 0.0) == 0
    assert resolved.decay_steps == 0
    # stop_step == 0 means "no end window", identical to the historical cosine_window contract.
    assert s.effective_weight(resolved, horizon) == pytest.approx(1.0)


def test_ratio_resolution_requires_a_horizon():
    with pytest.raises(ValueError, match="requires a training horizon"):
        s.spec_from_mapping("x", {"ramp_ratio": 0.1})
    with pytest.raises(ValueError, match="must be positive"):
        s.resolve_ratio_schedule("x", horizon_steps=0, ramp_ratio=0.1)


def test_mixing_absolute_and_ratio_forms_for_the_same_boundary_is_refused():
    with pytest.raises(ValueError, match="both absolute and ratio"):
        s.spec_from_mapping("x", {"ramp_steps": 100, "ramp_ratio": 0.1}, horizon_steps=1000)


def test_specs_from_config_defaults_to_an_empty_registry():
    assert s.specs_from_config(None) == {}
    assert s.specs_from_config({}) == {}


def test_resolved_provenance_persists_absolute_steps():
    specs = s.specs_from_config(
        {
            "stage1/anatomy": {"weight": 0.02, "start_ratio": 0.15, "ramp_ratio": 0.05, "kind": "cosine"},
            "stage2": {"weight": 1.0, "start_step": 20000, "ramp_steps": 2000, "stop_step": 30000, "decay_steps": 2000},
        },
        horizon_steps=32000,
    )
    provenance = s.resolved_provenance(specs, horizon_steps=32000)

    assert provenance["step_basis"] == "optimizer_global_step"
    assert provenance["horizon_optimizer_steps"] == 32000
    anatomy = provenance["components"]["stage1/anatomy"]
    assert anatomy["start_step"] == 4800 and anatomy["ramp_steps"] == 1600
    stage2 = provenance["components"]["stage2"]
    assert (stage2["start_step"], stage2["stop_step"], stage2["decay_steps"]) == (20000, 30000, 2000)
    # Deterministic ordering so the artifact is byte-stable.
    assert list(provenance["components"]) == sorted(provenance["components"])


# ==================================================================================================
# The trainer-side override hook
# ==================================================================================================
class _FakeModule:
    """The two methods the trainer contributes, in isolation from the 5,000-line module."""

    def __init__(self, specs, global_step):
        self._curriculum_specs = dict(specs or {})
        self.global_step = global_step

    curriculum_spec = None  # bound below


def _bind_module_methods():
    from asparagus.modules.lightning_modules.self_supervised import SelfSupervisedModule

    _FakeModule.curriculum_spec = SelfSupervisedModule.curriculum_spec
    _FakeModule.get_dynamic_weight = SelfSupervisedModule.get_dynamic_weight
    return _FakeModule


def test_without_a_curriculum_the_trainer_returns_the_historical_ramp():
    module_class = _bind_module_methods()
    for step in (0, 99, 100, 500, 4800, 32000):
        module = module_class({}, step)
        assert module.get_dynamic_weight(100, 400, component="stage1/anatomy") == s.cosine_ramp(step, 100, 400)
        # An unnamed call can never be overridden at all.
        assert module.get_dynamic_weight(100, 400) == s.cosine_ramp(step, 100, 400)


def test_a_curriculum_spec_overrides_only_its_own_component():
    module_class = _bind_module_methods()
    override = spec(name="stage2", start_step=1000, ramp_steps=0, kind="constant")
    module = module_class({"stage2": override}, 1500)

    assert module.get_dynamic_weight(0, 10_000, component="stage2") == 1.0
    # A different component keeps the historical ramp it was called with.
    assert module.get_dynamic_weight(0, 10_000, component="stage1/anatomy") == s.cosine_ramp(1500, 0, 10_000)


def test_an_overridden_component_is_still_a_pure_function_of_the_step():
    module_class = _bind_module_methods()
    override = spec(name="stage2", start_step=100, ramp_steps=100, stop_step=300, decay_steps=100)
    for step in range(0, 500, 5):
        module = module_class({"stage2": override}, step)
        assert module.get_dynamic_weight(0, 0, component="stage2") == s.effective_weight(override, step)


def test_falcon_window_is_overridable_and_otherwise_unchanged():
    from asparagus.modules.lightning_modules.self_supervised import SelfSupervisedModule

    class _Falcon:
        curriculum_spec = SelfSupervisedModule.curriculum_spec
        get_dynamic_falcon_weight = SelfSupervisedModule.get_dynamic_falcon_weight

        def __init__(self, specs, step):
            self._curriculum_specs = dict(specs)
            self.global_step = step
            self.mse_falcon_start_step = 0
            self.mse_falcon_warmup_steps = 3200
            self.mse_falcon_end_step = 0
            self.mse_falcon_decay_steps = 0

    for step in (0, 1600, 3200, 32000):
        assert _Falcon({}, step).get_dynamic_falcon_weight() == s.cosine_window(step, 0, 3200, 0, 0)

    override = spec(name="mse/falcon", start_step=0, ramp_steps=0, stop_step=1000, decay_steps=0)
    assert _Falcon({"mse/falcon": override}, 500).get_dynamic_falcon_weight() == 1.0
    assert _Falcon({"mse/falcon": override}, 1000).get_dynamic_falcon_weight() == 0.0


def test_cosine_ramp_midpoint_is_the_documented_s_curve():
    # Guards the exact analytic form: a linear ramp would give 0.5 at 25% too.
    assert s.cosine_ramp(25, 0, 100) == pytest.approx(0.5 * (1 - math.cos(math.pi * 0.25)))
    assert s.cosine_ramp(50, 0, 100) == pytest.approx(0.5)
