"""Wave 0A: bounded CPU smoke coverage for the Task-5 downstream finetune entrypoints.

Audit finding TEST-006 (`docs/audits/FOMO26_CODEBASE_FORENSIC_AUDIT_2026-07-31.md`): nothing in
either test suite referenced ``asparagus/pipeline/run/finetune_cls.py`` or ``finetune_reg.py``,
even though the recorded Task-5 downstream units ran exactly those entrypoints -- their run
directories contain ``script=finetune_cls``. ``tests/test_finetune_seg.py``
exercises *components* (SegmentationModule + a data module) and never imports the entrypoint, so
the Hydra composition, ``prepare_standard_experiment``, ``resolve_checkpoint``,
``resolve_pretrained_weights`` and ``write_resource_metrics`` wiring was untested.

These tests drive the **real** entrypoint function. ``@hydra.main`` wraps ``main`` with
``functools.wraps``, so ``main.__wrapped__`` is the undecorated production function; composing the
real config and calling it needs **no testability seam**. Nothing here mocks ``Trainer.fit`` and
nothing reimplements checkpoint loading -- the transfer assertions read back what the production
loader actually put into the saved checkpoint.

Bounded by construction: CPU only, synthetic 32^3 volumes, ``unet_tiny``, one epoch, one train
batch, one val batch, no network (``WANDB_MODE=offline``, ``wandb_logging=false``), no Docker, no
real FOMO26 data, no Jean-Zay path.

Two harness-only settings are worth stating explicitly, because they are *not* scientific changes:

* ``hardware.compile_mode=null`` -- the default ``"default"`` triggers ``torch.compile``, which
  aborts the CPU process with a native ``free(): invalid size`` during validation teardown.
  ``configs/hardware/osx.yaml`` ships the same empty value, so this is an existing supported
  hardware profile, not a new one.
* ``training.learning_rate=0`` in the transfer tests -- AdamW with lr=0 (decoupled weight decay
  included) leaves parameters bit-identical, which is what lets the test prove that a specific
  encoder tensor came from the pretrained checkpoint rather than from random init.

Two production defects this file surfaced were fixed in the same wave, and are guarded here:
``test_entrypoints_are_mutually_importable`` (resolvers now registered with ``replace=True``) and
``test_finetune_entrypoint_runs_with_the_progress_bar_disabled`` (the progress bar is now attached
only when ``logger.progress_bar`` is set).
"""

from __future__ import annotations

import importlib
import json
import os
import pytest
import torch
from omegaconf import OmegaConf
from pathlib import Path

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs"

# A value no random initialisation will produce, used to prove a real transfer happened.
SENTINEL = 0.036_912_5
PATCH = [32, 32, 32]


def _import_entrypoint(name: str):
    """Import an entrypoint module directly.

    This needs no shim: every module under ``asparagus/pipeline/run/`` registers its OmegaConf
    resolvers with ``replace=True``, matching the pattern ``pretrain.py`` already used, so
    importing several entrypoints into one interpreter is safe.
    ``test_entrypoints_are_mutually_importable`` guards that property.
    """
    return importlib.import_module(f"asparagus.pipeline.run.{name}")


# --------------------------------------------------------------------------------------------
# Synthetic dataset in the exact layout prepare_standard_experiment() expects
# --------------------------------------------------------------------------------------------


def _write_dataset(data_root: Path, task: str, *, regression: bool) -> dict:
    """Create $ASPARAGUS_DATA/<task>/{dataset.json,split_smoke.json,TEST_smoke.json} + samples.

    Layout comes from configs/core/base.yaml:82-84 (`${ASPARAGUS_DATA}/${task}/${split}.json`)
    and versioning.pathing() (`${data.data_path}/dataset.json`).
    """
    task_dir = data_root / task
    sample_dir = task_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []
    for i in range(4):
        # Label conventions follow tests/conftest.py: classification labels are 0-dim ints
        # (ClassificationModule.on_before_batch_transfer squeezes them), regression labels are
        # 1-D floats so they collate to [B, 1] and match the clsreg head's output shape.
        label = torch.tensor([float(i % 2)]) if regression else torch.tensor(i % 2)
        path = sample_dir / f"case_{i:03d}.pt"
        torch.save((torch.randn(1, *PATCH), label), path)
        paths.append(str(path))

    n_classes = 1 if regression else 2
    (task_dir / "dataset.json").write_text(json.dumps({"metadata": {"n_modalities": 1, "n_classes": n_classes}}))
    # train_split_path is indexed by cfg.data.fold, so this is a list of folds.
    (task_dir / "split_smoke.json").write_text(json.dumps([{"train": paths[:2], "val": paths[2:3]}]))
    (task_dir / "TEST_smoke.json").write_text(json.dumps(paths[3:]))
    return {"task_dir": task_dir, "paths": paths, "n_classes": n_classes}


def _write_ssl_checkpoint(path: Path, *, n_classes: int) -> dict[str, torch.Tensor]:
    """Write a minimal Lightning-style SSL checkpoint whose encoder tensors are all SENTINEL.

    Built from the *same* factory the downstream run uses, so every encoder shape matches and the
    transfer is a genuine one rather than a shape-coincidence. Keys use the ``model.<...>``
    namespace that `extract_pretrained_encoder_state` expects from an SSL Lightning checkpoint.
    """
    from asparagus.modules.networks.unet import unet_clsreg_tiny

    reference = unet_clsreg_tiny(input_channels=1, output_channels=n_classes, dimensions="3D")
    encoder_state = {
        f"model.{key}": torch.full_like(value, SENTINEL)
        for key, value in reference.state_dict().items()
        if key.startswith("encoder.")
    }
    assert encoder_state, "reference model exposes no encoder.* parameters"

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": encoder_state, "global_step": 1000, "epoch": 1}, path)
    return encoder_state


# --------------------------------------------------------------------------------------------
# Driving the real entrypoint
# --------------------------------------------------------------------------------------------


def _base_overrides(task: str, out_dir: Path) -> list[str]:
    return [
        f"task={task}",
        f"test_task={task}",
        "data.train_split=split_smoke",
        "data.test_split=TEST_smoke",
        "data.fold=0",
        "+model=unet_tiny",
        "training.epochs=1",
        "training.batch_size=1",
        "training.limit_train_batches=1",
        "training.limit_val_batches=1",
        "training.check_val_every_n_epoch=1",
        "training.warmup_epochs=0",
        "++training.decoder_warmup_epochs=0",
        f"training.target_size=[{','.join(str(x) for x in PATCH)}]",
        "training.seed=1234",
        "logger.wandb_logging=false",
        "logger.mlflow_logging=false",
        "logger.log_to_stdout=false",
        "hardware.accelerator=cpu",
        "hardware.num_devices=1",
        "hardware.trainer_devices=1",
        "hardware.num_workers=0",
        "hardware.precision=32",
        "hardware.strategy=auto",
        "~hardware.compile_mode",
        "+hardware.compile_mode=null",  # see module docstring
        f"hydra.run.dir={out_dir}",
    ]


def _run_entrypoint(module, config_name: str, overrides: list[str], out_dir: Path):
    """Compose the real config and call the real, undecorated entrypoint. Returns the config."""
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import open_dict

    out_dir.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=config_name, overrides=overrides, return_hydra_config=True)
        # compose() leaves hydra.runtime.output_dir MISSING; pathing() reads it.
        with open_dict(cfg):
            cfg.hydra.runtime.output_dir = str(out_dir)
            cfg.hydra.run.dir = str(out_dir)
            cfg.hydra.job.name = config_name
            cfg.hydra.job.config_name = config_name
        HydraConfig.instance().set_config(cfg)
        with open_dict(cfg):
            cfg.pop("hydra", None)
        module.main.__wrapped__(cfg)
    return cfg


@pytest.fixture
def smoke_env(tmp_path, monkeypatch):
    """Point the ASPARAGUS_* variables at a throwaway tree and keep the run fully offline."""
    for name, sub in (
        ("ASPARAGUS_DATA", "data"),
        ("ASPARAGUS_MODELS", "models"),
        ("ASPARAGUS_RESULTS", "results"),
    ):
        target = tmp_path / sub
        target.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(name, str(target))
    monkeypatch.setenv("ASPARAGUS_CONFIGS", str(CONFIG_DIR))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_ENTITY", "test")
    return tmp_path


ENTRYPOINTS = [
    pytest.param("finetune_cls", "default_finetune_cls", "CLS999_Wave0aSmoke", False, id="cls"),
    pytest.param("finetune_reg", "default_finetune_reg", "REGR999_Wave0aSmoke", True, id="reg"),
]

# Every module under asparagus/pipeline/run/ that registers OmegaConf resolvers at import time.
#
# This list is read by `importlib.import_module`, so a name here is a dependency that no import
# statement records and no static scan can see. It named five modules -- train_cls, train_reg,
# train_seg, test_cls and test_seg -- for some time after they were removed from
# `asparagus/pipeline/run/`, and `test_entrypoints_are_mutually_importable` raised
# ModuleNotFoundError throughout. It stayed invisible because the failure was present both before
# and after every subsequent change, so any comparison against a control showed no difference.
#
# The list stays a literal: it is the *expectation*, and deriving it from the directory would
# make the drift check below assert the tree against itself. The directory scan is the
# observation, and `test_entrypoint_module_list_is_exact` compares the two.
RETIRED_ENTRYPOINT_MODULES = (
    "train_cls",
    "train_reg",
    "train_seg",
    "test_cls",
    "test_seg",
)

ALL_ENTRYPOINT_MODULES = [
    "pretrain",
    "finetune_cls",
    "finetune_reg",
    "finetune_seg",
    "linear_probe",
    "eval_box",
]


def _resolver_registering_modules_on_disk() -> set[str]:
    """Observation, not expectation: what `asparagus/pipeline/run/` actually contains."""
    run_dir = REPO / "asparagus" / "pipeline" / "run"
    return {
        path.stem
        for path in run_dir.glob("*.py")
        if path.stem != "__init__" and "register_new_resolver" in path.read_text(encoding="utf-8", errors="replace")
    }


def test_entrypoint_module_list_is_exact():
    """The literal expectation must equal what is on disk, in both directions.

    A module removed from the tree must leave this list, and a new entrypoint must be added to it
    rather than silently escaping the mutual-import check below.
    """
    on_disk = _resolver_registering_modules_on_disk()
    assert set(ALL_ENTRYPOINT_MODULES) == on_disk, (
        "the entrypoint list drifted from asparagus/pipeline/run/.\n"
        f"  listed but absent   : {sorted(set(ALL_ENTRYPOINT_MODULES) - on_disk)}\n"
        f"  present but unlisted: {sorted(on_disk - set(ALL_ENTRYPOINT_MODULES))}"
    )
    assert len(ALL_ENTRYPOINT_MODULES) == len(set(ALL_ENTRYPOINT_MODULES)), "duplicate entrypoint name"


def test_retired_entrypoints_are_gone_and_stay_out_of_the_list():
    """Ratchet on the five modules an authorized deletion wave removed.

    Fails if one is reinstated without review, and fails if one is put back into the list while
    still absent -- which is how the stale list caused a permanent failure in the first place.
    """
    for name in RETIRED_ENTRYPOINT_MODULES:
        assert name not in ALL_ENTRYPOINT_MODULES, f"{name} is retired but is listed as an entrypoint"
        assert not (REPO / "asparagus" / "pipeline" / "run" / f"{name}.py").exists(), (
            f"{name} is back on disk; if that is intended, remove it from RETIRED_ENTRYPOINT_MODULES "
            "and add it to ALL_ENTRYPOINT_MODULES in the same change"
        )


def test_entrypoints_are_mutually_importable():
    """Importing every entrypoint into one interpreter must not raise.

    Each module calls ``OmegaConf.register_new_resolver`` for ``random``/``version``/``eval`` at
    import time. Without ``replace=True`` the second import raises
    ``ValueError: resolver 'random' is already registered``. That is invisible under
    ``@hydra.main`` (one entrypoint per process) but makes the modules mutually exclusive in any
    test session, and it silently blocked coverage of this whole family.
    """
    for name in ALL_ENTRYPOINT_MODULES:
        _import_entrypoint(name)

    for resolver in ("random", "version", "eval"):
        assert OmegaConf.has_resolver(resolver), f"resolver {resolver!r} was not registered"


# --------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("entrypoint,config_name,task,regression", ENTRYPOINTS)
def test_finetune_entrypoint_runs_end_to_end_on_cpu(smoke_env, entrypoint, config_name, task, regression):
    """The production entrypoint completes one bounded train+validation epoch and writes its run.

    Covers, through the real code path: Hydra composition -> prepare_standard_experiment ->
    resolve_checkpoint -> seed_everything -> transforms -> data module -> model construction ->
    resolve_pretrained_weights -> lightning module -> Trainer -> fit -> write_resource_metrics.
    """
    module = _import_entrypoint(entrypoint)
    data_root = Path(os.environ["ASPARAGUS_DATA"])
    _write_dataset(data_root, task, regression=regression)
    out_dir = smoke_env / "run"

    cfg = _run_entrypoint(module, config_name, _base_overrides(task, out_dir), out_dir)

    # Task identity and fold must survive composition into the run.
    assert cfg.task == task
    assert cfg.data.fold == 0
    assert cfg.training.seed == 1234
    assert str(cfg.data.data_path).endswith(task)

    # The run must have produced a real checkpoint, not an empty directory.
    best = out_dir / "checkpoints" / "best.ckpt"
    assert best.is_file(), f"{entrypoint} produced no best.ckpt in {out_dir}"
    payload = torch.load(best, map_location="cpu", weights_only=False)
    assert payload["state_dict"], "saved checkpoint has an empty state_dict"
    assert payload["global_step"] >= 1, "no optimizer step was taken"


@pytest.mark.parametrize("entrypoint,config_name,task,regression", ENTRYPOINTS)
def test_finetune_entrypoint_serialises_resource_metrics(smoke_env, entrypoint, config_name, task, regression):
    """write_resource_metrics() is part of the entrypoint contract; check what it emits."""
    module = _import_entrypoint(entrypoint)
    data_root = Path(os.environ["ASPARAGUS_DATA"])
    _write_dataset(data_root, task, regression=regression)
    out_dir = smoke_env / "run"

    _run_entrypoint(module, config_name, _base_overrides(task, out_dir), out_dir)

    metrics_path = out_dir / "resource_metrics.json"
    assert metrics_path.is_file(), f"{entrypoint} wrote no resource_metrics.json"
    record = json.loads(metrics_path.read_text())

    # Characterisation: pin the fields collectors read, so a writer-side change is visible.
    for field in ("optimizer_steps", "samples_processed"):
        assert field in record, f"resource_metrics.json lost the '{field}' field"
    assert int(record["optimizer_steps"]) >= 1
    assert int(record["samples_processed"]) >= 1
    # Serialised content must be JSON round-trippable (collectors parse it with json.loads).
    assert json.loads(json.dumps(record)) == record


@pytest.mark.parametrize("entrypoint,config_name,task,regression", ENTRYPOINTS)
def test_finetune_entrypoint_loads_pretrained_encoder_weights(smoke_env, entrypoint, config_name, task, regression):
    """Prove the SSL->downstream transfer really happened, via the production loader.

    ``training.learning_rate=0`` freezes every parameter (AdamW with lr=0 applies no update and no
    decoupled decay), so any encoder tensor equal to SENTINEL in the *saved* checkpoint can only
    have come from ``resolve_pretrained_weights`` reading the SSL checkpoint. The decoder head is
    asserted to be untouched, which is the encoder_only contract.
    """
    module = _import_entrypoint(entrypoint)
    data_root = Path(os.environ["ASPARAGUS_DATA"])
    dataset = _write_dataset(data_root, task, regression=regression)
    ssl_ckpt = smoke_env / "ssl" / "pretrained.ckpt"
    encoder_state = _write_ssl_checkpoint(ssl_ckpt, n_classes=dataset["n_classes"])
    out_dir = smoke_env / "run"

    overrides = _base_overrides(task, out_dir) + [
        f"checkpoint_path={ssl_ckpt}",
        "pretrained.enabled=true",
        "pretrained.source=online",
        "pretrained.load_scope=encoder_only",
        "++training.learning_rate=0.0",
        "++training.weight_decay=0.0",
    ]
    _run_entrypoint(module, config_name, overrides, out_dir)

    saved = torch.load(out_dir / "checkpoints" / "best.ckpt", map_location="cpu", weights_only=False)
    state = saved["state_dict"]

    transferred = [k for k in encoder_state if k in state]
    assert transferred, (
        "no encoder key from the SSL checkpoint appears in the finetuned checkpoint; the "
        f"namespaces diverged (ssl sample: {sorted(encoder_state)[:2]}, "
        f"downstream sample: {sorted(state)[:2]})"
    )

    matched = [k for k in transferred if torch.allclose(state[k], encoder_state[k], atol=0, rtol=0)]
    assert matched, (
        f"{entrypoint}: {len(transferred)} encoder keys were present but none retained the "
        "pretrained values - resolve_pretrained_weights did not transfer them"
    )
    # A real transfer moves the whole backbone, not one incidental buffer.
    assert len(matched) >= 0.9 * len(transferred), (
        f"{entrypoint}: only {len(matched)}/{len(transferred)} encoder tensors carry the pretrained values"
    )

    # encoder_only must leave the downstream head randomly initialised.
    head = [k for k in state if ".decoder." in k or k.startswith("model.decoder.")]
    assert head, "downstream module exposes no decoder/head parameters to check"
    assert not any(
        torch.allclose(state[k], torch.full_like(state[k], SENTINEL), atol=0, rtol=0) for k in head if state[k].numel() > 1
    ), "encoder_only transfer leaked pretrained values into the downstream head"


@pytest.mark.parametrize("entrypoint,config_name,task,regression", ENTRYPOINTS)
def test_finetune_entrypoint_runs_with_the_progress_bar_disabled(smoke_env, entrypoint, config_name, task, regression):
    """`logger.progress_bar=false` must be a usable setting.

    `configs/core/base.yaml` maps it to Lightning's `enable_progress_bar`, but the entrypoints
    used to append a TQDMProgressBar to `callbacks` unconditionally, so Lightning rejected the
    combination with `MisconfigurationException` at Trainer construction. Every run entrypoint
    now attaches the bar only when it is enabled.
    """
    module = _import_entrypoint(entrypoint)
    data_root = Path(os.environ["ASPARAGUS_DATA"])
    _write_dataset(data_root, task, regression=regression)
    out_dir = smoke_env / "run"

    overrides = _base_overrides(task, out_dir) + ["logger.progress_bar=false"]
    _run_entrypoint(module, config_name, overrides, out_dir)

    assert (out_dir / "checkpoints" / "best.ckpt").is_file(), (
        f"{entrypoint} produced no checkpoint with the progress bar disabled"
    )


@pytest.mark.parametrize("entrypoint,config_name,task,regression", ENTRYPOINTS)
def test_finetune_entrypoint_fails_closed_when_pretrained_checkpoint_is_missing(
    smoke_env, entrypoint, config_name, task, regression
):
    """`pretrained.enabled=true` with no resolvable checkpoint must raise, never train from scratch.

    This is the negative case for the transfer contract: silently falling back to random init
    would produce a scientifically mislabelled 'pretrained' run.
    """
    module = _import_entrypoint(entrypoint)
    data_root = Path(os.environ["ASPARAGUS_DATA"])
    _write_dataset(data_root, task, regression=regression)
    out_dir = smoke_env / "run"

    overrides = _base_overrides(task, out_dir) + [
        "pretrained.enabled=true",
        "pretrained.source=online",
        "pretrained.load_scope=encoder_only",
    ]
    with pytest.raises(ValueError, match="no checkpoint was resolved"):
        _run_entrypoint(module, config_name, overrides, out_dir)
