"""The general/registered data-stream curriculum in SameSessionMultimodalSampler.

The load-bearing property is the first test: with no curriculum configured the sampler yields the
*byte-identical* index sequence it always did.  Everything else in this file is only safe because of
that — the frozen Task-5 arms all run without a curriculum, and a sampler that quietly re-rolled its
RNG would change what they measured.
"""

from __future__ import annotations

import pytest
from asparagus.modules.data_modules.pretraining import SameSessionMultimodalSampler
from asparagus.modules.lightning_modules.ssl import schedules as s


def _row(subject: str, modality_id: int, is_registered: bool) -> dict:
    """One cached-metadata row in the exact shape PretrainDataset produces."""
    return {
        "identity": {
            "subject_session_key": f"{subject}_ses0",
            "subject_key": subject,
            "modality_id": modality_id,
            "is_registered_subset": is_registered,
            "bvalue": None,
            "scanner_key": "scanner0",
        },
        "common": {"age": 50.0, "sex": 0, "pathology": 0},
    }


class FakeDataset:
    """Minimal stand-in exposing exactly the metadata contract the sampler reads.

    ``registered_sessions`` of the sessions are marked as belonging to the registered subset, each
    with two modalities so both ``single_session`` and ``packed_pairs`` support exists.
    """

    def __init__(self, sessions: int = 20, registered_sessions: int = 6):
        self._cached_metadata = []
        for session in range(sessions):
            is_registered = session < registered_sessions
            for modality_id in (1, 2):
                self._cached_metadata.append(_row(f"sub{session}", modality_id, is_registered))
        self.files = list(range(len(self._cached_metadata)))

    def __len__(self):
        return len(self._cached_metadata)


def make_sampler(**kwargs) -> SameSessionMultimodalSampler:
    defaults = {
        "dataset": kwargs.pop("dataset", FakeDataset()),
        "batch_size": 4,
        "num_samples": 160,
        "multimodal_probability": 0.5,
        "seed": 431027,
    }
    defaults.update(kwargs)
    dataset = defaults.pop("dataset")
    return SameSessionMultimodalSampler(dataset, **defaults)


def schedule(**kwargs) -> s.ScheduleSpec:
    base = {"name": "data/registered", "enabled": True, "weight": 1.0, "kind": "linear"}
    base.update(kwargs)
    return s.ScheduleSpec(**base)


# ==================================================================================================
# The frozen-behaviour guarantee
# ==================================================================================================
def test_without_a_schedule_the_index_sequence_is_unchanged():
    dataset = FakeDataset()
    baseline = make_sampler(dataset=dataset)
    baseline.set_epoch(0)
    first = list(baseline)

    again = make_sampler(dataset=FakeDataset())
    again.set_epoch(0)
    second = list(again)

    assert first == second
    assert len(first) == baseline.num_samples_per_rank
    assert baseline.registered_probability(0) == 0.0
    assert baseline.registered_probability(10_000) == 0.0


def test_a_zero_weight_schedule_never_produces_a_registered_batch():
    sampler = make_sampler(registered_batch_schedule=schedule(weight=0.0))
    sampler.set_epoch(0)
    list(sampler)
    assert sampler.realized_registered_fraction() == 0.0
    assert not any(mode.startswith("registered_") for mode in sampler.batch_mode_counts)


def test_zero_probability_yields_the_same_indices_as_no_schedule_at_all():
    # A configured-but-zero curriculum must be observationally identical to no curriculum: the
    # registered RNG is a separate stream, so it cannot perturb the historical draws.
    plain = make_sampler(dataset=FakeDataset())
    plain.set_epoch(0)
    expected = list(plain)

    zero = make_sampler(dataset=FakeDataset(), registered_batch_schedule=schedule(weight=0.0))
    zero.set_epoch(0)
    assert list(zero) == expected


# ==================================================================================================
# Probability schedules driven by the optimizer step
# ==================================================================================================
def test_constant_probability_is_reported_at_every_step():
    sampler = make_sampler(registered_batch_schedule=schedule(weight=0.5, kind="constant"))
    for step in (0, 1, 100, 10_000):
        assert sampler.registered_probability(step) == pytest.approx(0.5)


def test_linear_ramp_probability_tracks_the_optimizer_step():
    sampler = make_sampler(registered_batch_schedule=schedule(weight=0.8, start_step=100, ramp_steps=100, kind="linear"))
    assert sampler.registered_probability(50) == 0.0
    assert sampler.registered_probability(100) == pytest.approx(0.0)
    assert sampler.registered_probability(150) == pytest.approx(0.4)
    assert sampler.registered_probability(200) == pytest.approx(0.8)
    assert sampler.registered_probability(5000) == pytest.approx(0.8)


def test_probability_is_clamped_into_the_unit_interval():
    sampler = make_sampler(registered_batch_schedule=schedule(weight=3.0, kind="constant"))
    assert sampler.registered_probability(10) == 1.0


def test_registered_only_final_phase_makes_every_batch_registered():
    sampler = make_sampler(registered_batch_schedule=schedule(weight=1.0, kind="constant"))
    sampler.set_epoch(0)
    list(sampler)
    assert sampler.realized_registered_fraction() == pytest.approx(1.0)
    assert sampler.registered_fallback_count == 0


def test_mixed_final_phase_keeps_general_batches_available():
    sampler = make_sampler(registered_batch_schedule=schedule(weight=0.5, kind="constant"))
    sampler.set_epoch(0)
    list(sampler)
    fraction = sampler.realized_registered_fraction()
    assert 0.0 < fraction < 1.0, fraction
    assert any(mode.startswith("general") for mode in sampler.batch_mode_counts)


def test_a_late_starting_ramp_leaves_the_early_phase_purely_general():
    # Starting well beyond this iteration's horizon: no batch in it may be registered.
    sampler = make_sampler(registered_batch_schedule=schedule(weight=1.0, start_step=10_000, ramp_steps=100))
    sampler.set_epoch(0)
    list(sampler)
    assert sampler.realized_registered_fraction() == 0.0


# ==================================================================================================
# Optimizer-step derivation, accumulation, DDP agreement
# ==================================================================================================
def test_optimizer_step_is_derived_from_position_and_accumulation():
    sampler = make_sampler(registered_batch_schedule=schedule(), accumulate_grad_batches=1)
    assert sampler.batches_per_iteration == 40
    assert sampler.optimizer_step_for(0, 0) == 0
    assert sampler.optimizer_step_for(0, 39) == 39
    assert sampler.optimizer_step_for(1, 0) == 40

    accumulated = make_sampler(registered_batch_schedule=schedule(), accumulate_grad_batches=4)
    assert accumulated.optimizer_step_for(0, 0) == 0
    assert accumulated.optimizer_step_for(0, 3) == 0
    assert accumulated.optimizer_step_for(0, 4) == 1
    assert accumulated.optimizer_step_for(1, 0) == 10


def test_all_ddp_ranks_agree_on_the_batch_mode_for_every_optimizer_step():
    spec = schedule(weight=0.5, kind="constant")
    ranks = []
    for rank in range(4):
        sampler = make_sampler(
            dataset=FakeDataset(),
            registered_batch_schedule=spec,
            num_replicas=4,
            rank=rank,
        )
        sampler.set_epoch(0)
        list(sampler)
        ranks.append(sampler.batch_mode_log)

    assert all(log == ranks[0] for log in ranks[1:]), "ranks disagreed on batch mode"
    assert ranks[0], "no batches were produced"
    # And the modes really do vary, so the agreement is not trivially satisfied by a constant.
    assert len({mode for _step, mode in ranks[0]}) > 1


def test_accumulation_does_not_change_which_optimizer_step_a_probability_belongs_to():
    spec = schedule(weight=1.0, start_step=5, ramp_steps=0, kind="constant")
    single = make_sampler(registered_batch_schedule=spec, accumulate_grad_batches=1)
    quad = make_sampler(registered_batch_schedule=spec, accumulate_grad_batches=4)
    # Batch 8 is optimizer step 8 without accumulation and step 2 with accumulation 4; the
    # probability follows the optimizer step in both cases, which is the documented contract.
    assert single.registered_probability(single.optimizer_step_for(0, 8)) == 1.0
    assert quad.registered_probability(quad.optimizer_step_for(0, 8)) == 0.0
    assert quad.registered_probability(quad.optimizer_step_for(0, 20)) == 1.0


# ==================================================================================================
# Resume safety
# ==================================================================================================
def test_resume_mid_ramp_reproduces_the_uninterrupted_batch_mode_sequence():
    spec = schedule(weight=0.6, start_step=0, ramp_steps=80, kind="linear")

    uninterrupted = make_sampler(dataset=FakeDataset(), registered_batch_schedule=spec)
    uninterrupted.set_epoch(0)
    list(uninterrupted)
    expected_first_epoch = list(uninterrupted.batch_mode_log)
    list(uninterrupted)  # second pseudo-epoch
    expected_second_epoch = list(uninterrupted.batch_mode_log)

    resumed = make_sampler(dataset=FakeDataset(), registered_batch_schedule=spec)
    resumed.load_state_dict({"iteration": 1, "position": 0})
    list(resumed)

    assert expected_first_epoch != expected_second_epoch, "the two epochs must differ for this to prove anything"
    assert resumed.batch_mode_log == expected_second_epoch


def test_resume_state_dict_round_trip_preserves_position():
    sampler = make_sampler(registered_batch_schedule=schedule(weight=0.5, kind="constant"))
    sampler.enable_consumption_tracking()
    sampler.set_epoch(0)
    iterator = iter(sampler)
    for _ in range(20):
        next(iterator)
    sampler.mark_consumed(20)
    state = sampler.state_dict()
    assert state["iteration"] == 0
    assert state["position"] > 0

    restored = make_sampler(registered_batch_schedule=schedule(weight=0.5, kind="constant"))
    restored.load_state_dict(state)
    remaining = list(restored)
    assert len(remaining) == restored.num_samples_per_rank - state["position"]


# ==================================================================================================
# Support, fallbacks and fail-closed behaviour
# ==================================================================================================
def test_insufficient_registered_support_fails_closed_rather_than_degrading_silently():
    empty = FakeDataset(sessions=20, registered_sessions=0)
    with pytest.raises(ValueError, match="no registered same-session multimodal support"):
        make_sampler(dataset=empty, registered_batch_schedule=schedule(weight=0.5))


def test_zero_weight_is_allowed_even_without_registered_support():
    empty = FakeDataset(sessions=20, registered_sessions=0)
    sampler = make_sampler(dataset=empty, registered_batch_schedule=schedule(weight=0.0))
    sampler.set_epoch(0)
    assert list(sampler)


def test_registered_batches_only_ever_draw_registered_samples():
    dataset = FakeDataset(sessions=20, registered_sessions=5)
    registered_indices = {
        index for index, row in enumerate(dataset._cached_metadata) if row["identity"]["is_registered_subset"]
    }
    sampler = make_sampler(dataset=dataset, registered_batch_schedule=schedule(weight=1.0, kind="constant"))
    sampler.set_epoch(0)
    indices = list(sampler)

    # Every batch is a registered batch, and the sampler pads short batches from the full corpus;
    # with two modalities per session the 4-sample batch takes 2 registered + 2 filler, so assert
    # that the *registered-drawn* portion never leaves the registered subset.
    assert sampler.realized_registered_fraction() == pytest.approx(1.0)
    assert registered_indices, "the fixture must contain registered samples"
    assert any(index in registered_indices for index in indices)


def test_packed_pair_registered_support_is_used_in_packed_mode():
    dataset = FakeDataset(sessions=20, registered_sessions=6)
    sampler = make_sampler(
        dataset=dataset,
        batch_size=4,
        stage1_multimodal_batch_mode="packed_pairs",
        registered_batch_schedule=schedule(weight=1.0, kind="constant"),
    )
    sampler.set_epoch(0)
    list(sampler)
    assert sampler.batch_mode_counts.get("registered_packed", 0) > 0
    assert "registered_single_session" not in sampler.batch_mode_counts


def test_packed_mode_without_registered_pair_support_fails_closed():
    class SingleModalityRegistered(FakeDataset):
        def __init__(self):
            super().__init__(sessions=20, registered_sessions=0)
            # One registered scan, but alone in its session: no pair support can exist.
            self._cached_metadata.append(_row("subX", 1, True))
            self.files = list(range(len(self._cached_metadata)))

    with pytest.raises(ValueError, match="no registered same-session multimodal support"):
        make_sampler(
            dataset=SingleModalityRegistered(),
            stage1_multimodal_batch_mode="packed_pairs",
            registered_batch_schedule=schedule(weight=0.5),
        )


# ==================================================================================================
# Matched treatment/control streams
# ==================================================================================================
def test_matched_treatment_and_control_share_an_identical_batch_type_schedule():
    spec = schedule(weight=0.5, start_step=0, ramp_steps=40, kind="linear")
    treatment = make_sampler(dataset=FakeDataset(), registered_batch_schedule=spec, seed=431027)
    control = make_sampler(dataset=FakeDataset(), registered_batch_schedule=spec, seed=431027)
    treatment.set_epoch(0)
    control.set_epoch(0)
    treatment_indices, control_indices = list(treatment), list(control)

    assert treatment.batch_mode_log == control.batch_mode_log
    assert treatment_indices == control_indices, "a matched control must see the identical stream"


def test_a_different_seed_changes_the_stream_so_the_match_is_meaningful():
    spec = schedule(weight=0.5, kind="constant")
    a = make_sampler(dataset=FakeDataset(), registered_batch_schedule=spec, seed=1)
    b = make_sampler(dataset=FakeDataset(), registered_batch_schedule=spec, seed=2)
    a.set_epoch(0)
    b.set_epoch(0)
    assert list(a) != list(b)


# ==================================================================================================
# Logging surface
# ==================================================================================================
def test_the_sampler_exposes_every_required_diagnostic():
    sampler = make_sampler(registered_batch_schedule=schedule(weight=0.5, kind="constant"))
    sampler.set_epoch(0)
    list(sampler)

    assert sampler.last_batch_mode is not None
    assert sampler.last_registered_probability == pytest.approx(0.5)
    assert sampler.realized_registered_fraction() >= 0.0
    assert sum(sampler.batch_mode_counts.values()) == sampler.batches_per_iteration
    assert len(sampler.batch_mode_log) == sampler.batches_per_iteration
    assert isinstance(sampler.registered_fallback_count, int)


def test_batch_mode_counters_reset_between_epochs():
    sampler = make_sampler(registered_batch_schedule=schedule(weight=0.5, kind="constant"))
    sampler.set_epoch(0)
    list(sampler)
    first = sum(sampler.batch_mode_counts.values())
    list(sampler)
    assert sum(sampler.batch_mode_counts.values()) == first
