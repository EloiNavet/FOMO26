"""DDP contract for the retained public v4 AMAES stack (gloo, CPU, 2 ranks).

The authoritative v4 config `projects/fomo26/safety/pretrain/resenc_amaes_p2_96k` resolves to
`strategy: ddp`, `num_devices: 4`, `num_nodes: 1`, so distributed training is part of the public
contract and not merely historical provenance. The tier that used to hold this guarantee was
emptied when the JEPA-specific test retired with its objective; this restores it for the stack the
release actually ships.

What it proves: two ranks reach a rendezvous, the retained ResEnc-B/AMAES module constructs on
both, one bounded forward/backward completes, the gradients are finite, and a deterministic
parameter digest agrees across ranks -- a per-rank divergence desynchronises every collective that
follows, which hangs rather than raises.

The AMAES path must also stay AMAES: the dormant JEPA compatibility remainder is imported by
`self_supervised` at module scope, so its presence in `sys.modules` proves nothing. The assertion
that matters is that the objective is not *enabled*.

Gloo on CPU, file rendezvous, finite process-group timeout, every phase announced -- so a stall
raises where the traceback is instead of blocking where it is not. No GPU, Slurm or network
service is required, and no worker is left behind.

Run directly::

    python tests/run_test_tier.py ddp
"""

import os
import pytest
import queue as queue_module
import tempfile
import time
import torch
import torch.multiprocessing as mp
import traceback
from _pytest.outcomes import Failed
from datetime import timedelta
from pathlib import Path

# A cold spawned interpreter has to import torch, lightning and asparagus before it can do
# anything, which is minutes rather than seconds on a two-core runner. Budget for that, and
# rely on the phase reporting -- not on the budget -- to tell a slow import from a hang.
TOTAL_TIMEOUT = 600.0
POLL_INTERVAL = 5.0
JOIN_TIMEOUT = 30.0
# Finite, and far below TOTAL_TIMEOUT: a rendezvous or collective that cannot complete must
# raise inside the child (where the traceback is) instead of blocking until the parent
# gives up (where it is not). torch's default for gloo is 30 minutes. Read from the
# environment -- which a spawned child inherits -- so the harness self-test below can prove
# the mechanism in seconds instead of minutes.
PROCESS_GROUP_TIMEOUT = timedelta(seconds=float(os.environ.get("FOMO26_DDP_PG_TIMEOUT_S", "120")))

PHASES = ("spawned", "imported", "ready", "result")


def _init(rank, world_size, rendezvous):
    import torch.distributed as dist

    # A runner with several interfaces can bind a gloo transport the peer cannot reach,
    # which stalls the first collective rather than failing it. Loopback is the only
    # interface these two ranks ever need.
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=PROCESS_GROUP_TIMEOUT,
    )


def _child(body, rank, world_size, rendezvous, channel):
    """Announce each phase, and never let a child exception die unseen in the child."""

    import torch.distributed as dist

    try:
        channel.put(("spawned", rank, None))
        # The expensive import, made explicit so a slow one is visibly a slow import.
        import asparagus.modules.lightning_modules.self_supervised  # noqa: F401

        channel.put(("imported", rank, None))
        _init(rank, world_size, rendezvous)
        channel.put(("ready", rank, None))
        channel.put(("result", rank, body(rank)))
    except BaseException:  # noqa: BLE001 - the parent must see every failure mode
        channel.put(("failed", rank, traceback.format_exc()))
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _diagnosis(reason, processes, phases):
    lines = [reason]
    for rank, process in enumerate(processes):
        lines.append(f"  rank {rank}: last phase={phases[rank]!r} alive={process.is_alive()} exitcode={process.exitcode}")
    lines.append("  phases: spawned -> imported (asparagus loaded) -> ready (process group joined) -> result.")
    lines.append(
        "  One rank at 'result' while another is stuck at 'ready' is a genuine one-sided collective. "
        "Every rank stuck at 'spawned'/'imported' is a slow or failed import or rendezvous, not a hang. "
        "A non-zero exitcode with no 'failed' message means the child died without reporting."
    )
    return "\n".join(lines)


def _shutdown(processes):
    """Leave no orphan behind, whatever happened above."""

    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=JOIN_TIMEOUT)
        if process.is_alive():
            process.kill()
            process.join(timeout=JOIN_TIMEOUT)


def _collect(processes, channel, world_size):
    phases = {rank: "spawned" for rank in range(world_size)}
    results = {}
    deadline = time.monotonic() + TOTAL_TIMEOUT
    while len(results) < world_size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            pytest.fail(_diagnosis("timed out waiting for every rank to report", processes, phases))
        try:
            kind, rank, payload = channel.get(timeout=min(remaining, POLL_INTERVAL))
        except queue_module.Empty:
            if all(not process.is_alive() for process in processes):
                pytest.fail(_diagnosis("every child exited before reporting a result", processes, phases))
            continue
        if kind == "failed":
            pytest.fail(f"rank {rank} raised after phase {phases[rank]!r}:\n{payload}")
        phases[rank] = kind
        if kind == "result":
            results[rank] = payload
    return results


def _spawn(body, world_size=2):
    context = mp.get_context("spawn")
    channel = context.Queue()
    with tempfile.TemporaryDirectory(prefix="amaes-ddp-") as directory:
        # A file rendezvous needs no port, so no run can collide with another suite, with a
        # leftover process, or with itself on a rerun.
        rendezvous = Path(directory) / "rendezvous"
        processes = [
            context.Process(target=_child, args=(body, rank, world_size, str(rendezvous), channel), daemon=True)
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        try:
            results = _collect(processes, channel, world_size)
            for rank, process in enumerate(processes):
                process.join(timeout=JOIN_TIMEOUT)
                assert process.exitcode == 0, f"rank {rank} exited with {process.exitcode}"
            return results
        finally:
            _shutdown(processes)


def _amaes_step(rank):
    """Construct the retained AMAES module, take one DDP step, and report a digest."""

    import hashlib
    import torch.distributed as dist
    from asparagus.modules.lightning_modules import SelfSupervisedModule
    from asparagus.modules.networks.resenc_unet import resenc_unet_debug
    from torch.nn.parallel import DistributedDataParallel

    torch.manual_seed(0)
    net = resenc_unet_debug(dimensions="3D", input_channels=1, output_channels=1)
    module = SelfSupervisedModule(model=net, learning_rate=1e-3, warmup_epochs=0, train_transforms=None, val_transforms=None)
    # The public stack is AMAES-only. The JEPA remainder is importable but must not be selected.
    jepa_enabled = bool(getattr(module, "_jepa_enabled", False))

    ddp = DistributedDataParallel(net)
    x = torch.randn(2, 1, 16, 16, 16, generator=torch.Generator().manual_seed(1234 + rank))
    out = ddp(x)
    pred = out[0] if isinstance(out, (list, tuple)) else out
    if isinstance(pred, dict):
        pred = next(iter(pred.values()))
    loss = torch.nn.functional.mse_loss(pred, torch.zeros_like(pred))
    loss.backward()

    grads = [p.grad for p in ddp.parameters() if p.grad is not None]
    finite = all(bool(torch.isfinite(g).all()) for g in grads)

    # DDP all-reduces gradients, so after backward every rank must hold the same gradient state.
    digest = hashlib.sha256()
    for name, parameter in sorted(ddp.module.named_parameters()):
        gradient = parameter.grad
        digest.update(name.encode())
        digest.update(b"none" if gradient is None else gradient.detach().to(torch.float64).numpy().tobytes())

    world = torch.tensor([float(dist.get_world_size())])
    dist.all_reduce(world)
    return {
        "rank": rank,
        "jepa_enabled": jepa_enabled,
        "gradients": len(grads),
        "gradients_finite": finite,
        "grad_digest": digest.hexdigest(),
        "all_reduce_total": float(world.item()),
    }


def test_two_ranks_take_one_amaes_step_and_agree():
    results = _spawn(_amaes_step, world_size=2)
    assert sorted(results) == [0, 1]
    for rank, r in sorted(results.items()):
        assert r["jepa_enabled"] is False, f"rank {rank} selected the retired JEPA objective"
        assert r["gradients"] > 0, f"rank {rank} produced no gradients"
        assert r["gradients_finite"], f"rank {rank} produced a non-finite gradient"
        assert r["all_reduce_total"] == 4.0, f"rank {rank} saw a broken collective"
    assert results[0]["grad_digest"] == results[1]["grad_digest"], (
        "the ranks hold different gradients after the DDP all-reduce; every collective after this point would desynchronise"
    )


def _one_sided_failure(rank):
    """Module level: `spawn` pickles the body, and a local closure cannot be pickled."""
    if rank == 1:
        raise RuntimeError("deliberate rank-1 failure")
    return {"rank": rank}


def test_a_failure_in_one_rank_is_reported_rather_than_hanging():
    """The harness must surface a one-sided failure instead of blocking until the budget expires."""
    with pytest.raises(Failed, match="deliberate rank-1 failure"):
        _spawn(_one_sided_failure, world_size=2)
