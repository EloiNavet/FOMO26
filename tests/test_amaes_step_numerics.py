"""The published AMAES step is pinned numerically, not merely structurally.

Reducing the pretraining stack to the reconstruction objective was a large edit to the module
that computes the published step. Imports and type checks cannot tell whether such an edit
changed what a step *computes*: a reordered sum or a dropped term still imports cleanly.

What was measured, and what this test does and does not claim
------------------------------------------------------------
An A/B comparison was run between the implementation before the reduction and the one here, in
the same process and the same pinned environment. Losses, every parameter gradient and every
parameter after three optimizer steps were bit-identical; the only difference was that 113 logged
scalars belonging to objectives that are no longer shipped stopped being reported, and none was a
core reconstruction metric. That is an exact same-environment equivalence result.

It is **not** portable. Floating-point addition is not associative, so the summation order inside
the convolution kernels -- which depends on thread count, CPU instruction set and BLAS build --
changes the last few digits. On this machine alone, moving from one thread to two shifts the
step-2 loss from 10.823636054992676 to 10.823689460754395. `torch.use_deterministic_algorithms`
makes a run repeatable on one configuration; it does not make two configurations agree bit for
bit, and this test does not assert that they do.

So this test is a **portable regression contract**: exact on everything structural, and
tolerance-based on the arithmetic, with the tolerance derived from measurement rather than
chosen for comfort. It still fails if the reconstruction loss changes, a term is dropped, the
gradients change, the optimizer update changes, a parameter is added, removed or renamed, a
gradient or parameter becomes non-finite, or a parameter silently stops receiving gradient.

The arithmetic is compared at two scales, because they are not equally well conditioned. A
step-0 gradient of one small bias is a sum whose terms cancel almost entirely, so its last digits
belong to the reduction order and not to the model; two GitHub-hosted runners of the same image
disagree on it by as much as a factor of two. The same gradients summed over the whole model
agree to about one part in a million. The per-tensor comparison is therefore deliberately the
looser of the two, and the model-wide one carries the tight tolerance. Read together they say:
every tensor is in the right place and the right magnitude class, and the gradient as a whole is
the same gradient.

Neither this test nor the A/B result is a reproduction of the real pretraining run: that would
need the real corpus, multiple GPUs and a full schedule.
"""

from __future__ import annotations

import json
import os
import pytest
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "tests" / "amaes_step_probe.py"
REFERENCE = REPO / "tests" / "amaes_step_reference.json"

# Tolerances, each derived from deviations actually measured between environments, then rounded up
# to the next power of ten. What was measured, on this machine unless stated:
#
#   three repeated runs, same pinned environment          bit-identical
#   pre-reduction versus post-reduction, same process     bit-identical
#   one thread versus two, four and eight                 losses            4.9e-06 relative
#                                                         gradients         6.5e-06 relative
#                                                         final parameters  1.1e-06 relative
#                                                         update totals     3.8e-05 relative
#   GitHub-hosted runner A versus this machine            losses            2.4e-06 relative
#   GitHub-hosted runner B versus this machine            losses, final parameters and update
#                                                         totals all within the tolerances below;
#                                                         step-0 per-tensor gradients up to
#                                                         2.5e-04 relative above the floor, and up
#                                                         to 1.2e+00 relative below it
#
# Runner B is why the gradient tolerances are not the same as the loss tolerance. Two hosted
# runners of the same image differ in instruction set, so the reduction order inside the conv
# backward differs, and float32 gradients differ accordingly. Nothing about the result is wrong;
# a tolerance that only one runner class satisfies is what was wrong.
#
# Per-tensor comparisons carry an absolute floor proportional to the model-wide magnitude of the
# quantity compared. Without it a bias whose gradient squared-norm is 4.4e-14 would demand a
# relative tolerance of 0.46 to survive a thread-count change, and the test would be blind. Bias
# gradients are sums over a batch and a volume whose terms cancel almost completely: the surviving
# value is orders of magnitude below the terms that produced it, so its relative error is set by
# those terms, not by itself. The floor is what keeps such a value from demanding an O(1) relative
# tolerance for everyone else. Gradients now use the same floor as parameters, which is the floor
# that already survived both runner classes.
LOSS_RTOL = 1e-4
GRADIENT_RTOL = 1e-3
GRADIENT_ATOL_FRACTION = 1e-8
PARAMETER_RTOL = 1e-4
PARAMETER_ATOL_FRACTION = 1e-8
# The update is summarised model-wide; see the probe for why per-tensor is not portable. 5e-4 is
# thirteen times the largest deviation measured, and still catches a learning rate or a beta
# changed in the fourth significant figure.
UPDATE_RTOL = 5e-4
# Summed over all 269 gradient tensors. A total is far better conditioned than its terms: the
# deviations that force GRADIENT_RTOL to 1e-3 are absolute quantities around 1e-05, and they are
# being compared against a total of 2.2e+03. Measured headroom against runner B is about two
# orders of magnitude, so this stays tight and recovers the sensitivity the per-tensor loosening
# gives up -- a uniform drift too small for any single tensor to report still moves the total.
GRADIENT_TOTAL_RTOL = 1e-4

GRADIENT_STATS = ("abs_sum", "sq_norm", "max_abs")
# `abs_sum` is excluded for parameters: after three steps a small bias can flip the sign of
# its own update, which moves the absolute sum by 1.4e-02 relative between thread counts
# while the squared norm moves by 1.1e-06. Squared norm and maximum absolute value are the
# summaries that survive.
PARAMETER_STATS = ("sq_norm", "max_abs")


def _run_probe() -> dict:
    """Measure in a subprocess: the thread pinning must precede the first Torch import."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(REPO), *(p for p in [environment.get("PYTHONPATH")] if p)])
    completed = subprocess.run(
        [sys.executable, str(PROBE)],
        cwd=str(REPO),
        env=environment,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if completed.returncode != 0:
        raise AssertionError(f"probe failed ({completed.returncode}):\n{completed.stderr[-4000:]}")
    return json.loads(completed.stdout)


@pytest.fixture(scope="module")
def observed() -> dict:
    return _run_probe()


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def _worst(observed_block: dict, reference_block: dict, stats, rtol: float, atol_fraction: float) -> list[str]:
    """Compare every tensor and every summary, and report every violation rather than the first."""
    scale = sum(entry["abs_sum"] for entry in reference_block.values())
    floor = atol_fraction * scale
    problems: list[str] = []
    # Fail closed on a set mismatch: a tensor that vanished must be reported as such, not raise a
    # KeyError from inside the comparison, and a tensor that appeared must not pass unexamined.
    missing = sorted(set(reference_block) - set(observed_block))
    extra = sorted(set(observed_block) - set(reference_block))
    problems += [f"{name}: absent from the observed run" for name in missing]
    problems += [f"{name}: present in the observed run but not in the reference" for name in extra]
    for name, expected in sorted(reference_block.items()):
        if name not in observed_block:
            continue
        actual = observed_block[name]
        for stat in stats:
            got, want = actual[stat], expected[stat]
            if abs(got - want) <= floor + rtol * abs(want):
                continue
            relative = abs(got - want) / max(abs(want), 1e-300)
            problems.append(f"{name}.{stat}: got {got!r} want {want!r} (relative {relative:.3e})")
    return problems


def test_reference_manifest_is_self_describing(reference):
    """A manifest whose provenance cannot be read is not evidence."""
    assert reference["schema"] == "fomo26-amaes-step-reference-v1"
    assert reference["seed"] == 20260908
    assert reference["steps"] == 3
    assert reference["input_shape"] == [2, 1, 64, 64, 64]


def test_parameter_identity_is_exact(observed, reference):
    """Names, shapes and counts are structure, not arithmetic: they must match exactly."""
    assert observed["shapes"] == reference["shapes"]
    assert len(observed["shapes"]) == 299
    assert set(observed["gradients_step0"]) == set(reference["gradients_step0"])
    assert len(observed["gradients_step0"]) == 269


def test_every_gradient_and_parameter_is_finite(observed):
    """A NaN or an infinity must fail here, not surface as a silently ruined run."""
    assert observed["finite"] == {"gradients": True, "parameters": True}


def test_gradient_participation_classes_are_unchanged(observed, reference):
    """Which parameters carry gradient, and which do not, is pinned exactly.

    In this configuration the batch supplies no ``modality_id``, so the modality-conditioning
    path is bypassed: the FiLM layers and the projection heads receive no gradient at all, and the
    modality embedding receives one that is identically zero. That is a property of the
    configuration, not a defect, and it is stable -- so it is recorded rather than asserted away.

    Comparing the three classes exactly is what makes this a disconnection detector. A parameter
    that stops receiving signal moves between classes, and no rounding difference can move it:
    an identically zero gradient is not a rounding of a non-zero one.
    """

    def classify(payload):
        every = set(payload["shapes"])
        graded = set(payload["gradients_step0"])
        zero = {n for n, s in payload["gradients_step0"].items() if s["abs_sum"] == 0.0}
        return {
            "no_gradient": sorted(every - graded),
            "zero_gradient": sorted(zero),
            "n_carrying_signal": len(graded - zero),
        }

    got, want = classify(observed), classify(reference)
    assert got["no_gradient"] == want["no_gradient"]
    assert got["zero_gradient"] == want["zero_gradient"]
    assert got["n_carrying_signal"] == want["n_carrying_signal"] == 268
    # Guard the guard: if the recorded classes ever swallowed the whole model the test above
    # would pass vacuously.
    assert len(want["no_gradient"]) == 30
    assert len(want["zero_gradient"]) == 1


def test_reconstruction_losses_match_the_reference(observed, reference):
    """The loss of every step, within the measured cross-environment margin."""
    got, want = observed["losses"], reference["losses"]
    assert len(got) == len(want) == 3
    offenders = [
        f"step {i}: got {a!r} want {b!r} (relative {abs(a - b) / abs(b):.3e})"
        for i, (a, b) in enumerate(zip(got, want))
        if abs(a - b) > LOSS_RTOL * abs(b)
    ]
    assert not offenders, "\n".join(offenders)


def test_gradients_match_the_reference(observed, reference):
    """Every per-tensor gradient summary, at the first step, before the optimizer amplifies."""
    problems = _worst(
        observed["gradients_step0"],
        reference["gradients_step0"],
        GRADIENT_STATS,
        GRADIENT_RTOL,
        GRADIENT_ATOL_FRACTION,
    )
    assert not problems, "gradient summaries diverged:\n" + "\n".join(problems[:20])


def test_gradient_totals_match_the_reference(observed, reference):
    """The model-wide gradient, where the arithmetic is well conditioned and the tolerance can be tight.

    Loosening the per-tensor tolerance to survive a second runner class costs sensitivity: a
    uniform drift of a few parts in ten thousand is now below the per-tensor threshold on every
    tensor individually. It is not below this one. Summing 269 tensors turns per-tensor deviations
    of order 1e-05 absolute into a relative deviation of order 1e-06 against a total of 2.2e+03, so
    a change that moves every gradient together shows up here at full strength while reduction-order
    noise does not.
    """

    def totals(payload):
        block = payload["gradients_step0"]
        return {
            "abs_sum": sum(entry["abs_sum"] for entry in block.values()),
            "sq_norm": sum(entry["sq_norm"] for entry in block.values()),
        }

    got, want = totals(observed), totals(reference)
    for stat in ("abs_sum", "sq_norm"):
        assert abs(got[stat] - want[stat]) <= GRADIENT_TOTAL_RTOL * abs(want[stat]), (
            f"model-wide gradient {stat}: got {got[stat]!r} want {want[stat]!r} "
            f"(relative {abs(got[stat] - want[stat]) / abs(want[stat]):.3e}, tolerance {GRADIENT_TOTAL_RTOL:.0e})"
        )


def test_optimizer_update_matches_the_reference(observed, reference):
    """The update the optimizer produced, summed over the model, and the weights it left behind."""
    for stat in ("abs_sum", "sq_norm"):
        got, want = observed["update_totals"][stat], reference["update_totals"][stat]
        assert abs(got - want) <= UPDATE_RTOL * abs(want), (
            f"total update {stat}: got {got!r} want {want!r} "
            f"(relative {abs(got - want) / abs(want):.3e}, tolerance {UPDATE_RTOL:.0e})"
        )
    problems = _worst(
        observed["parameters_after_steps"],
        reference["parameters_after_steps"],
        PARAMETER_STATS,
        PARAMETER_RTOL,
        PARAMETER_ATOL_FRACTION,
    )
    assert not problems, "parameter summaries diverged after three steps:\n" + "\n".join(problems[:20])


def test_the_step_still_reports_its_reconstruction_terms(observed, reference):
    """A dropped term that still computes the same loss would otherwise pass unnoticed."""
    missing = sorted(set(reference["logged_keys"]) - set(observed["logged_keys"]))
    assert not missing, f"the step stopped reporting: {missing}"
    for key in (
        "train/loss/mse/raw",
        "train/loss/mse/weighted",
        "train/reconstruction_loss/loss",
        "train/reconstruction_loss/loss_hidden",
        "train/reconstruction_loss/foreground_mode/none",
    ):
        assert key in observed["logged_keys"], key
