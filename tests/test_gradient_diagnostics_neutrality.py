"""Per-loss gradient-contribution diagnostics, and the proof that enabling them changes nothing.

The decisive test in this file is
:func:`test_enabling_diagnostics_leaves_the_optimizer_update_numerically_equivalent`.  A diagnostic
that perturbs the run it measures is worse than no diagnostic, because it invalidates the very
comparison it was added to inform.
"""

from __future__ import annotations

import pytest
import torch
from asparagus.modules.lightning_modules.self_supervised import SelfSupervisedModule


class TinyEncoderModel(torch.nn.Module):
    """A model whose parameters are named so the diagnostics' encoder subset selection applies."""

    def __init__(self, width: int = 6):
        super().__init__()
        self.encoder = torch.nn.Linear(width, width, bias=False)
        self.decoder_head = torch.nn.Linear(width, width, bias=False)

    def forward(self, x):
        return self.decoder_head(self.encoder(x))


class DiagnosticHarness:
    """Binds only the diagnostic methods, so the 5,000-line module is not constructed."""

    _maybe_log_loss_gradient_norms = SelfSupervisedModule._maybe_log_loss_gradient_norms
    # staticmethod on the real class; re-wrap so binding here does not inject `self`.
    _loss_gradient_alignment_metrics = staticmethod(SelfSupervisedModule._loss_gradient_alignment_metrics)
    _diagnostic_parameters = SelfSupervisedModule._diagnostic_parameters

    def __init__(self, model, *, every_n_steps: int, global_step: int = 0):
        self.model = model
        self.loss_gradient_norm_every_n_steps = every_n_steps
        self.global_step = global_step
        self.logged: dict = {}

    def log_dict(self, metrics, **_kwargs):
        self.logged.update({key: float(value) for key, value in metrics.items()})


def build_components(model, batch, *, include_dead: bool = False) -> dict:
    """Three live objectives sharing one encoder, in the module's component-tuple shape.

    Each entry is ``(raw, config_weight, schedule_weight, weighted, enabled)``.
    """
    features = model.encoder(batch)
    mse_raw = features.pow(2).mean()
    aux_raw = features.mean().abs()
    # Deliberately anti-aligned with mse so the pairwise cosine has something real to report.
    opposed_raw = -features.pow(2).mean() + features.abs().mean()

    components = {
        "mse": (mse_raw, 1.0, 1.0, 1.0 * 1.0 * mse_raw, True),
        "aux": (aux_raw, 0.5, 0.4, 0.5 * 0.4 * aux_raw, True),
        "opposed": (opposed_raw, 0.25, 1.0, 0.25 * 1.0 * opposed_raw, True),
    }
    if include_dead:
        zero = torch.zeros((), dtype=batch.dtype)
        components["stage2/anatomy"] = (zero, 1.0, 0.0, zero, False)
    return components


def total_loss(components: dict) -> torch.Tensor:
    return sum(entry[3] for entry in components.values())


def run_optimizer_steps(*, diagnostics_every: int, steps: int = 5, seed: int = 431027):
    """Run a short training loop, optionally with diagnostics enabled, and return the final state."""
    torch.manual_seed(seed)
    model = TinyEncoderModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    harness = DiagnosticHarness(model, every_n_steps=diagnostics_every)

    torch.manual_seed(seed + 1)
    batches = [torch.randn(4, 6) for _ in range(steps)]

    for step, batch in enumerate(batches):
        harness.global_step = step
        optimizer.zero_grad(set_to_none=True)
        components = build_components(model, batch, include_dead=True)
        loss = total_loss(components)
        # Diagnostics run exactly where the production module runs them: after the loss is built and
        # before backward, on a graph that backward will still need.
        harness._maybe_log_loss_gradient_norms(components)
        loss.backward()
        optimizer.step()

    return {
        "parameters": {name: parameter.detach().clone() for name, parameter in model.named_parameters()},
        "optimizer_state": optimizer.state_dict(),
        "rng": torch.get_rng_state(),
        "logged": harness.logged,
    }


# ==================================================================================================
# The neutrality proof
# ==================================================================================================
def test_enabling_diagnostics_leaves_the_optimizer_update_numerically_equivalent():
    without = run_optimizer_steps(diagnostics_every=0)
    with_diagnostics = run_optimizer_steps(diagnostics_every=1)

    assert without["logged"] == {}, "diagnostics must be silent when disabled"
    assert with_diagnostics["logged"], "diagnostics must actually produce metrics when enabled"

    for name, expected in without["parameters"].items():
        observed = with_diagnostics["parameters"][name]
        assert torch.allclose(observed, expected, rtol=0.0, atol=1e-6), name
        # On this deterministic CPU path the updates are in fact bit-identical; the tolerance above
        # is the contract, this is the observed strength.
        assert torch.equal(observed, expected), f"{name} was perturbed by the diagnostics"


def test_diagnostics_do_not_advance_the_rng_stream():
    without = run_optimizer_steps(diagnostics_every=0)
    with_diagnostics = run_optimizer_steps(diagnostics_every=1)
    assert torch.equal(without["rng"], with_diagnostics["rng"])


def test_diagnostics_do_not_change_optimizer_state_or_step_count():
    without = run_optimizer_steps(diagnostics_every=0)
    with_diagnostics = run_optimizer_steps(diagnostics_every=1)

    left = without["optimizer_state"]["state"]
    right = with_diagnostics["optimizer_state"]["state"]
    assert set(left) == set(right)
    for key in left:
        assert int(left[key]["step"].item()) == int(right[key]["step"].item()) == 5
        assert torch.equal(left[key]["exp_avg"], right[key]["exp_avg"])
        assert torch.equal(left[key]["exp_avg_sq"], right[key]["exp_avg_sq"])


def test_diagnostics_populate_no_grad_buffer_of_their_own():
    torch.manual_seed(0)
    model = TinyEncoderModel()
    harness = DiagnosticHarness(model, every_n_steps=1)
    batch = torch.randn(4, 6)

    components = build_components(model, batch)
    harness._maybe_log_loss_gradient_norms(components)

    # torch.autograd.grad returns gradients; it never accumulates into .grad.
    assert all(parameter.grad is None for parameter in model.parameters())


def test_diagnostics_do_not_take_a_second_optimizer_step():
    torch.manual_seed(0)
    model = TinyEncoderModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    steps = []

    original_step = optimizer.step

    def counting_step(*args, **kwargs):
        steps.append(1)
        return original_step(*args, **kwargs)

    optimizer.step = counting_step
    harness = DiagnosticHarness(model, every_n_steps=1)

    batch = torch.randn(4, 6)
    components = build_components(model, batch)
    loss = total_loss(components)
    harness._maybe_log_loss_gradient_norms(components)
    loss.backward()
    optimizer.step()

    assert steps == [1]


def test_diagnostics_do_not_touch_the_amp_scaler():
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    before = scaler.state_dict()

    torch.manual_seed(0)
    model = TinyEncoderModel()
    harness = DiagnosticHarness(model, every_n_steps=1)
    components = build_components(model, torch.randn(4, 6))
    harness._maybe_log_loss_gradient_norms(components)

    assert scaler.state_dict() == before


def test_backward_still_succeeds_after_the_diagnostic_block_so_no_graph_is_lost():
    torch.manual_seed(0)
    model = TinyEncoderModel()
    harness = DiagnosticHarness(model, every_n_steps=1)
    components = build_components(model, torch.randn(4, 6))
    loss = total_loss(components)

    harness._maybe_log_loss_gradient_norms(components)
    loss.backward()  # would raise if retain_graph had not preserved it

    assert any(parameter.grad is not None for parameter in model.parameters())


def test_the_graph_is_released_once_the_step_completes():
    torch.manual_seed(0)
    model = TinyEncoderModel()
    harness = DiagnosticHarness(model, every_n_steps=1)
    components = build_components(model, torch.randn(4, 6))
    loss = total_loss(components)
    harness._maybe_log_loss_gradient_norms(components)
    loss.backward()

    # A retained graph would let a second backward succeed; the buffers must be freed exactly as in
    # an undiagnosed step.
    with pytest.raises(RuntimeError, match="second time"):
        loss.backward()


# ==================================================================================================
# Frequency bound
# ==================================================================================================
@pytest.mark.parametrize("every,step,expected", [(0, 0, False), (0, 10, False), (10, 5, False), (10, 10, True), (1, 7, True)])
def test_diagnostics_respect_the_configured_frequency(every, step, expected):
    torch.manual_seed(0)
    model = TinyEncoderModel()
    harness = DiagnosticHarness(model, every_n_steps=every, global_step=step)
    harness._maybe_log_loss_gradient_norms(build_components(model, torch.randn(4, 6)))
    assert bool(harness.logged) is expected


# ==================================================================================================
# What the diagnostics report
# ==================================================================================================
def test_every_required_metric_family_is_emitted():
    torch.manual_seed(0)
    model = TinyEncoderModel()
    harness = DiagnosticHarness(model, every_n_steps=1)
    harness._maybe_log_loss_gradient_norms(build_components(model, torch.randn(4, 6), include_dead=True))
    keys = harness.logged

    for family in (
        "train/grad_norm/",
        "train/grad_finite_fraction/",
        "train/grad_effective_weight/",
        "train/grad_contribution_weighted/",
        "train/grad_contribution_unweighted/",
        "train/grad_available/",
        "train/grad_cosine/",
        "train/grad_norm_ratio/",
    ):
        assert any(key.startswith(family) for key in keys), family


def test_an_inactive_component_is_marked_unavailable_not_reported_as_zero():
    torch.manual_seed(0)
    model = TinyEncoderModel()
    harness = DiagnosticHarness(model, every_n_steps=1)
    harness._maybe_log_loss_gradient_norms(build_components(model, torch.randn(4, 6), include_dead=True))

    assert harness.logged["train/grad_available/stage2/anatomy"] == 0.0
    assert harness.logged["train/grad_available/mse"] == 1.0
    # An unavailable component must not contribute a misleading zero norm.
    assert "train/grad_norm/stage2/anatomy" not in harness.logged


def test_finite_fraction_is_one_for_a_healthy_component():
    torch.manual_seed(0)
    model = TinyEncoderModel()
    harness = DiagnosticHarness(model, every_n_steps=1)
    harness._maybe_log_loss_gradient_norms(build_components(model, torch.randn(4, 6)))
    assert harness.logged["train/grad_finite_fraction/mse"] == 1.0


def test_effective_weight_is_the_product_of_config_and_schedule_weight():
    torch.manual_seed(0)
    model = TinyEncoderModel()
    harness = DiagnosticHarness(model, every_n_steps=1)
    harness._maybe_log_loss_gradient_norms(build_components(model, torch.randn(4, 6)))
    assert harness.logged["train/grad_effective_weight/aux"] == pytest.approx(0.5 * 0.4)
    assert harness.logged["train/grad_effective_weight/mse"] == pytest.approx(1.0)


def test_unweighted_contribution_divides_the_effective_weight_back_out():
    torch.manual_seed(0)
    model = TinyEncoderModel()
    harness = DiagnosticHarness(model, every_n_steps=1)
    harness._maybe_log_loss_gradient_norms(build_components(model, torch.randn(4, 6)))
    weighted = harness.logged["train/grad_contribution_weighted/aux"]
    unweighted = harness.logged["train/grad_contribution_unweighted/aux"]
    assert unweighted == pytest.approx(weighted / (0.5 * 0.4))


def test_pairwise_cosines_cover_every_pair_and_keep_the_historical_mse_names():
    torch.manual_seed(0)
    model = TinyEncoderModel()
    harness = DiagnosticHarness(model, every_n_steps=1)
    harness._maybe_log_loss_gradient_norms(build_components(model, torch.randn(4, 6)))

    # Historical keys preserved so existing dashboards keep resolving.
    assert "train/grad_cosine/aux_to_mse" in harness.logged
    assert "train/grad_cosine/opposed_to_mse" in harness.logged
    # And the pair the old mse-only version could never show.
    assert "train/grad_cosine/aux_to_opposed" in harness.logged
    for key, value in harness.logged.items():
        if key.startswith("train/grad_cosine/"):
            assert -1.0 <= value <= 1.0, key


def test_an_opposed_objective_shows_a_negative_cosine_against_mse():
    torch.manual_seed(0)
    model = TinyEncoderModel()
    harness = DiagnosticHarness(model, every_n_steps=1)
    harness._maybe_log_loss_gradient_norms(build_components(model, torch.randn(4, 6)))
    assert harness.logged["train/grad_cosine/opposed_to_mse"] < 0.0


def test_diagnostics_measure_the_encoder_subset_only():
    torch.manual_seed(0)
    model = TinyEncoderModel()
    harness = DiagnosticHarness(model, every_n_steps=1)
    parameters = harness._diagnostic_parameters()
    encoder_parameters = [parameter for name, parameter in model.named_parameters() if "encoder" in name]
    assert len(parameters) == len(encoder_parameters) == 1
    assert parameters[0] is model.encoder.weight


def test_diagnostics_survive_a_component_that_shares_no_parameters_with_the_encoder():
    torch.manual_seed(0)
    model = TinyEncoderModel()
    harness = DiagnosticHarness(model, every_n_steps=1)
    detached = model.decoder_head.weight.pow(2).mean()
    components = {
        "mse": (detached, 1.0, 1.0, detached, True),
    }
    # allow_unused=True must absorb this rather than raising.
    harness._maybe_log_loss_gradient_norms(components)
    assert harness.logged["train/grad_norm/mse"] == 0.0
