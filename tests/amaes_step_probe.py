"""Measure one AMAES pretraining step, reproducibly enough to compare across machines.

Run as a subprocess, never imported into a pytest process that has already loaded Torch: the
thread-count environment below only takes effect before the first Torch import, and thread count
changes the summation order inside the convolution kernels. That is not a correctness problem -- it is
ordinary floating-point non-associativity -- but it does mean an unpinned run is not comparable
with any other run.

The summaries are chosen to be numerically stable. A signed sum is deliberately NOT among them:
for a tensor whose gradient is near zero (several biases here have a gradient squared-norm around
1e-14) the signed sum is dominated by cancellation, and its relative error across two thread
counts on the same machine reaches 5.9e+01. Absolute sum, squared norm and maximum absolute value
accumulate magnitudes instead, so they stay stable to a few parts in a million.

`python tests/amaes_step_probe.py --write tests/amaes_step_reference.json` regenerates the
reference. Every input is synthetic and seeded; no corpus, checkpoint or absolute path is involved.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
for _name in _THREAD_VARS:
    os.environ[_name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)

SEED = 20260908
SHAPE = (2, 1, 64, 64, 64)
STEPS = 3
SCHEMA = "fomo26-amaes-step-reference-v1"


def _seed() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)


def _summary(tensor: torch.Tensor) -> dict[str, float]:
    """Scale-stable summaries, accumulated in float64."""
    value = tensor.detach().to(torch.float64)
    return {
        "abs_sum": float(value.abs().sum()),
        "sq_norm": float((value * value).sum()),
        "max_abs": float(value.abs().max()),
    }


def _build():
    from asparagus.modules.lightning_modules.self_supervised import SelfSupervisedModule
    from asparagus.modules.networks.resenc_unet import resenc_unet_b_ssl

    _seed()
    model = resenc_unet_b_ssl(dimensions="3D", input_channels=1, output_channels=1)
    _seed()
    module = SelfSupervisedModule(
        model=model,
        learning_rate=1e-3,
        enable_mse_loss=True,
        loss_weight_mse=10.0,
        rec_loss_masked_only=True,
        mse_foreground_mode="none",
        mse_foreground_aware=False,
        log_every_n_steps=1,
    )
    return module


def measure() -> dict:
    from types import SimpleNamespace

    module = _build()
    logged: dict[str, object] = {}
    module.log_dict = lambda payload, *a, **k: logged.update(dict(payload))
    module.log = lambda name, value, *a, **k: logged.__setitem__(name, value)
    module._trainer = SimpleNamespace(
        global_step=0,
        max_steps=10,
        estimated_stepping_batches=10,
        is_global_zero=True,
        current_epoch=1,
        datamodule=SimpleNamespace(batch_size=2),
        precision_plugin=SimpleNamespace(scaler=None),
        logger=None,
        loggers=[],
    )
    module.train()

    optimizer = torch.optim.AdamW(module.model.parameters(), lr=1e-3)
    initial = {name: param.detach().clone() for name, param in module.model.named_parameters()}
    _seed()
    losses: list[float] = []
    gradients: dict[str, dict[str, float]] = {}
    finite = {"gradients": True, "parameters": True}

    for step in range(STEPS):
        batch = {
            "image": torch.randn(*SHAPE),
            "label": torch.randn(*SHAPE),
            "mask": torch.rand(*SHAPE) > 0.4,
        }
        optimizer.zero_grad(set_to_none=True)
        loss = module.training_step(batch, step)
        loss.backward()
        if step == 0:
            for name, param in module.model.named_parameters():
                if param.grad is None:
                    continue
                gradients[name] = _summary(param.grad)
                if not bool(torch.isfinite(param.grad).all()):
                    finite["gradients"] = False
        optimizer.step()
        losses.append(float(loss.detach()))

    # The optimizer update is summarised model-wide rather than per tensor. Per tensor it is not
    # portable: for a bias whose gradient is noise, Adam's direction is decided by that noise, so
    # the same tensor's update can flip sign between two thread counts on one machine (relative
    # deviation up to 2.0e-01 measured). Summed over the model the update is dominated by the
    # tensors that carry signal and is stable to 3.8e-05, which is what makes it assertable.
    update_totals = {"abs_sum": 0.0, "sq_norm": 0.0}
    parameters = {}
    shapes = {}
    for name, param in module.model.named_parameters():
        delta = _summary(param.detach() - initial[name])
        update_totals["abs_sum"] += delta["abs_sum"]
        update_totals["sq_norm"] += delta["sq_norm"]
        parameters[name] = _summary(param)
        shapes[name] = list(param.shape)
        if not bool(torch.isfinite(param).all()):
            finite["parameters"] = False

    return {
        "schema": SCHEMA,
        "seed": SEED,
        "input_shape": list(SHAPE),
        "steps": STEPS,
        "losses": losses,
        "shapes": shapes,
        "gradients_step0": gradients,
        "parameters_after_steps": parameters,
        "update_totals": update_totals,
        "finite": finite,
        "logged_keys": sorted(logged),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", metavar="PATH", help="write the reference manifest to PATH")
    args = parser.parse_args()
    payload = measure()
    text = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    if args.write:
        with open(args.write, "w", encoding="utf-8") as handle:
            handle.write(text)
        return 0
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
