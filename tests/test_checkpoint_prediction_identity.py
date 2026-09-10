"""One checkpoint identity, from the evaluated weights through to the collectors.

Runs used to carry two independent answers to "which checkpoint was evaluated?":
``cfg.test_checkpoint`` named the prediction file, ``cfg.testing.checkpoint`` chose the weights.
Nothing kept them in agreement, and ``eval_box`` parses the filename back into the reported
checkpoint label -- so a config asking for ``current`` evaluated end-of-fit weights and published
them as ``best``. These tests pin the single resolved role and the deterministic prediction
identity that replaced that.
"""

from __future__ import annotations

import hashlib
import json
import pytest
import subprocess
import sys
from asparagus.pipeline.run import evaluation_identity as identity_module
from asparagus.pipeline.run.checkpoint_selection import (
    VALID_CHECKPOINT_ROLES,
    prediction_filename,
    resolve_checkpoint_path,
    resolve_test_checkpoint_role,
)
from finetuning.fomo26_inference import prediction_metrics
from finetuning.fomo26_inference.metrics_io import (
    LEGACY_SCHEMA_VERSIONS,
    PROVENANCE_FIELDS,
    SCHEMA_VERSION,
    build_record,
    provenance_problems,
    read_metrics_json,
    write_metrics_json,
)
from omegaconf import OmegaConf
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.reconciliation


def _cfg(**overrides):
    base = {"test_task": "SEG902_FOMO26_Task2_lesion", "data": {"test_split": "test"}}
    base.update(overrides)
    return OmegaConf.create(base)


# --- role resolution -------------------------------------------------------------------------


@pytest.mark.parametrize("role", VALID_CHECKPOINT_ROLES)
def test_agreeing_fields_resolve_to_that_role(role):
    cfg = _cfg(test_checkpoint=role, testing={"checkpoint": role})
    assert resolve_test_checkpoint_role(cfg) == role


@pytest.mark.parametrize("role", VALID_CHECKPOINT_ROLES)
def test_either_field_alone_resolves(role):
    assert resolve_test_checkpoint_role(_cfg(test_checkpoint=role)) == role
    assert resolve_test_checkpoint_role(_cfg(testing={"checkpoint": role})) == role


def test_disagreeing_fields_fail_instead_of_picking_one():
    """The defect this contract exists to prevent, in its original shape."""
    cfg = _cfg(test_checkpoint="best", testing={"checkpoint": "current"})
    with pytest.raises(ValueError, match="conflicting checkpoint selection"):
        resolve_test_checkpoint_role(cfg)


def test_absent_selection_fails():
    with pytest.raises(ValueError, match="no checkpoint selection"):
        resolve_test_checkpoint_role(_cfg())


def test_unknown_role_fails():
    with pytest.raises(ValueError, match="must be one of"):
        resolve_test_checkpoint_role(_cfg(test_checkpoint="penultimate"))


# --- prediction identity ---------------------------------------------------------------------


def test_prediction_filename_carries_the_resolved_role():
    cfg = _cfg(test_checkpoint="current", testing={"checkpoint": "current"})
    role = resolve_test_checkpoint_role(cfg)
    assert prediction_filename(cfg, role) == "SEG902_FOMO26_Task2_lesion__test__current.json"


def test_prediction_filename_is_not_derived_from_the_legacy_field():
    """A run evaluated on `current` must never be published as `best`."""
    cfg = _cfg(testing={"checkpoint": "current"})
    assert prediction_filename(cfg, resolve_test_checkpoint_role(cfg)).endswith("__current.json")


class _Callback:
    """Stands in for a ModelCheckpoint. `monitor` and `save_last` decide what its paths mean."""

    def __init__(self, best_model_path="", *, monitor=None, save_last=False, last_model_path=""):
        self.best_model_path = best_model_path
        self.last_model_path = last_model_path
        self.monitor = monitor
        self.save_last = save_last


def _monitored(path):
    return _Callback(path, monitor="val/loss")


def _periodic(path):
    """The unmonitored, every_n_epochs callback the finetune entrypoints construct."""
    return _Callback(path)


def test_current_role_loads_nothing():
    assert resolve_checkpoint_path("current", _monitored("/b.ckpt"), _periodic("/l.ckpt")) is None


def test_best_and_last_roles_select_their_own_callback():
    best, last = _monitored("/b.ckpt"), _periodic("/l.ckpt")
    assert resolve_checkpoint_path("best", best, last) == "/b.ckpt"
    assert resolve_checkpoint_path("last", best, last) == "/l.ckpt"


def test_last_prefers_the_dedicated_last_model_path_when_save_last_is_set():
    last = _Callback("/periodic.ckpt", save_last=True, last_model_path="/last.ckpt")
    assert resolve_checkpoint_path("last", _monitored("/b.ckpt"), last) == "/last.ckpt"


def test_best_refuses_an_unmonitored_callback():
    """With monitor=None Lightning writes the *latest* checkpoint into best_model_path."""
    with pytest.raises(ValueError, match="has no monitor"):
        resolve_checkpoint_path("best", _periodic("/latest.ckpt"), _periodic("/l.ckpt"))


def test_last_refuses_a_monitored_callback_that_does_not_save_last():
    """Its best_model_path is the best by metric, which is not what 'last' promises."""
    with pytest.raises(ValueError, match="monitored and does not save_last"):
        resolve_checkpoint_path("last", _monitored("/b.ckpt"), _monitored("/best-by-metric.ckpt"))


def test_missing_checkpoint_fails_instead_of_silently_using_end_of_fit_weights():
    with pytest.raises(ValueError, match="Refusing to fall back"):
        resolve_checkpoint_path("best", _monitored(""), _periodic("/l.ckpt"))
    with pytest.raises(ValueError, match="Refusing to fall back"):
        resolve_checkpoint_path("last", _monitored("/b.ckpt"), _periodic(""))


@pytest.mark.slow
def test_real_model_checkpoint_callbacks_resolve_to_the_files_they_wrote(tmp_path):
    """The 'best'/'last' contract against real Lightning callbacks, not a stand-in.

    The entrypoints build `last_ckpt_callback` with no monitor, so its latest write lands in
    `best_model_path` -- an attribute name that means "best" on the other callback. This drives
    a real two-epoch fit and asserts the two roles resolve to two different real files with the
    meanings they claim.
    """
    import lightning as pl
    import torch
    from lightning.pytorch.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader, TensorDataset

    class Tiny(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 1)
            self.losses = iter([1.0, 0.1, 5.0])  # epoch index 1 is the best; epoch index 2 is the last

        def training_step(self, batch, _):
            return self.layer(batch[0]).mean() * 0.0 + self.layer.weight.sum() * 0.0 + 1.0

        def validation_step(self, batch, _):
            return None

        def on_validation_epoch_end(self):
            # Logged per epoch, not per batch: batching must not shift which epoch is best.
            self.log("val/loss", next(self.losses, 9.0))

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.0)

    data = DataLoader(TensorDataset(torch.zeros(4, 4)), batch_size=2)
    ckpt_dir = tmp_path / "checkpoints"
    best_ckpt_callback = ModelCheckpoint(
        dirpath=ckpt_dir, monitor="val/loss", mode="min", save_top_k=1, filename="best", enable_version_counter=False
    )
    last_ckpt_callback = ModelCheckpoint(
        dirpath=ckpt_dir, every_n_epochs=1, save_top_k=1, filename="last", enable_version_counter=False
    )
    trainer = pl.Trainer(
        max_epochs=3,
        num_sanity_val_steps=0,  # a sanity pass would consume the first loss and shift every epoch
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[last_ckpt_callback, best_ckpt_callback],
    )
    trainer.fit(Tiny(), train_dataloaders=data, val_dataloaders=data)

    best = resolve_checkpoint_path("best", best_ckpt_callback, last_ckpt_callback)
    last = resolve_checkpoint_path("last", best_ckpt_callback, last_ckpt_callback)
    assert Path(best).is_file() and Path(last).is_file()
    assert Path(best).name == "best.ckpt"
    assert Path(last).name == "last.ckpt"
    assert best != last, "'best' and 'last' resolved to the same file"
    # The best checkpoint is the epoch that minimised val/loss, not the final one.
    assert torch.load(best, map_location="cpu", weights_only=False)["epoch"] == 1
    assert torch.load(last, map_location="cpu", weights_only=False)["epoch"] == 2


# --- shipped configs -------------------------------------------------------------------------


def test_every_finetune_config_declares_a_consistent_checkpoint_selection():
    """No shipped config may reintroduce the disagreement.

    Reads the YAML directly rather than composing it: this must hold for the file as written, so a
    later edit that sets only one of the two fields to a new value is caught here.
    """
    import yaml

    offenders = []
    for path in sorted((REPO / "configs").rglob("*.yaml")):
        try:
            doc = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue
        if not isinstance(doc, dict):
            continue
        legacy = doc.get("test_checkpoint")
        canonical = (doc.get("testing") or {}).get("checkpoint") if isinstance(doc.get("testing"), dict) else None
        if legacy is not None and canonical is not None and str(legacy) != str(canonical):
            offenders.append((str(path.relative_to(REPO)), legacy, canonical))
    assert not offenders, f"configs declare conflicting checkpoint selections: {offenders}"


# `test_archcmp_configs_declare_current_in_both_fields` pinned the six architecture-comparison
# finetune configs. That campaign is not part of the public code release. The invariant it
# guarded -- that a config never declares two conflicting checkpoint selections -- is still
# enforced above, over every retained config rather than those six.


# --- metrics provenance ----------------------------------------------------------------------


def test_new_records_carry_every_provenance_field(tmp_path):
    record = build_record(
        2,
        0,
        "completed",
        {"dsc": 0.5},
        task_name="SEG902_FOMO26_Task2_lesion",
        kind="seg",
        provenance=_complete_provenance(),
    )
    assert record["schema_version"] == SCHEMA_VERSION
    assert set(record["provenance"]) == set(PROVENANCE_FIELDS)
    assert record["provenance"]["evaluated_checkpoint_role"] == "best"
    # unsupplied fields are visibly null, not absent
    assert record["provenance"]["source_checkpoint_sha256"] is None

    out = tmp_path / "metrics.json"
    write_metrics_json(out, record)
    assert read_metrics_json(out)["provenance"]["evaluation_id"] == "eval_x"


def test_unknown_provenance_field_is_rejected():
    with pytest.raises(ValueError, match="unknown provenance field"):
        build_record(2, 0, "completed", {}, provenance={"checkpoint": "best"})


def test_legacy_v1_records_remain_readable(tmp_path):
    """Historical runs on disk are v1 and cannot be re-evaluated."""
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps(
            {
                "schema_version": LEGACY_SCHEMA_VERSIONS[0],
                "task": 2,
                "task_name": "SEG902_FOMO26_Task2_lesion",
                "kind": "seg",
                "fold": 0,
                "status": "completed",
                "metrics": {"dsc": 0.42},
            }
        )
    )
    record = read_metrics_json(legacy)
    assert record["schema_version"] == "fomo26-metrics-v1"
    assert record["metrics"]["dsc"] == 0.42
    assert "provenance" not in record


def test_unsupported_schema_is_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "fomo26-metrics-v99", "metrics": {}}))
    with pytest.raises(ValueError, match="unsupported metrics schema"):
        read_metrics_json(bad)


# --- launcher determinism --------------------------------------------------------------------


# --- the evaluation identity sidecar -----------------------------------------------------------


def _identity_cfg(**overrides):
    base = {
        "test_task": "SEG902_FOMO26_Task2_lesion",
        "data": {"test_split": "test", "fold": 0},
        "evaluation": {"id": "eval_x", "campaign_id": "d0"},
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _checkpoint(tmp_path, name, payload=b"weights"):
    path = tmp_path / name
    path.write_bytes(payload)
    return path


def test_identity_records_the_exact_evaluated_file_for_a_file_backed_role(tmp_path):
    ckpt = _checkpoint(tmp_path, "best.ckpt")
    identity = identity_module.build_identity(
        _identity_cfg(),
        role="best",
        checkpoint_path=str(ckpt),
        prediction_path=str(tmp_path / "p.json"),
        run_dir=str(tmp_path),
        epoch=7,
        global_step=700,
        status="completed",
        task_number=2,
    )
    assert identity["checkpoint_path"] == str(ckpt)
    assert identity["checkpoint_sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert identity["evaluated_state"] == identity_module.STATE_CHECKPOINT_FILE
    assert identity["evaluation_unit_id"] == "eval_x/task2/fold0"
    assert identity["task"] == 2 and identity["fold"] == 0 and identity["split"] == "test"


def test_current_carries_no_checkpoint_digest_at_all(tmp_path):
    """`current` evaluates weights that were never a file. Nothing may be attached to it."""
    _checkpoint(tmp_path, "best.ckpt")  # exists, and must still not be picked up
    identity = identity_module.build_identity(
        _identity_cfg(),
        role="current",
        checkpoint_path=None,
        prediction_path=str(tmp_path / "p.json"),
        run_dir=str(tmp_path),
        epoch=9,
        global_step=900,
        status="completed",
        task_number=2,
    )
    assert identity["checkpoint_path"] is None
    assert identity["checkpoint_sha256"] is None
    assert identity["evaluated_state"] == identity_module.STATE_IN_MEMORY_FINAL
    assert (identity["epoch"], identity["global_step"]) == (9, 900)


def test_attaching_a_checkpoint_to_current_is_refused(tmp_path):
    with pytest.raises(ValueError, match="in-memory weights"):
        identity_module.build_identity(
            _identity_cfg(),
            role="current",
            checkpoint_path=str(_checkpoint(tmp_path, "best.ckpt")),
            prediction_path=str(tmp_path / "p.json"),
            run_dir=str(tmp_path),
            epoch=1,
            global_step=1,
            status="completed",
        )


def test_reading_back_an_inconsistent_identity_fails(tmp_path):
    identity = identity_module.build_identity(
        _identity_cfg(),
        role="current",
        checkpoint_path=None,
        prediction_path=str(tmp_path / "p.json"),
        run_dir=str(tmp_path),
        epoch=1,
        global_step=1,
        status="completed",
        task_number=2,
    )
    identity["checkpoint_sha256"] = "f" * 64  # tampered: current has no checkpoint
    identity_module.write_identity(tmp_path, identity)
    with pytest.raises(ValueError, match="never loaded"):
        identity_module.read_identity(tmp_path / identity_module.FILENAME)


def test_identity_field_cli_prints_one_field(tmp_path):
    """The launcher reads the sidecar through this, not through sed or jq."""
    identity = identity_module.build_identity(
        _identity_cfg(),
        role="best",
        checkpoint_path=str(_checkpoint(tmp_path, "best.ckpt")),
        prediction_path=str(tmp_path / "predictions" / "p.json"),
        run_dir=str(tmp_path),
        epoch=3,
        global_step=30,
        status="completed",
        task_number=2,
    )
    identity_module.write_identity(tmp_path, identity)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "asparagus.pipeline.run.evaluation_identity",
            "--run-dir",
            str(tmp_path),
            "--field",
            "prediction_path",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == str(tmp_path / "predictions" / "p.json")


# --- provenance flows from the sidecar, and completeness is enforced ---------------------------


def _identity_file(tmp_path, role, checkpoint_path, prediction_path):
    identity = identity_module.build_identity(
        _identity_cfg(),
        role=role,
        checkpoint_path=checkpoint_path,
        prediction_path=str(prediction_path),
        run_dir=str(tmp_path),
        epoch=5,
        global_step=500,
        status="completed",
        task_number=2,
    )
    return identity_module.write_identity(tmp_path, identity)


def test_metrics_provenance_is_taken_from_the_sidecar(tmp_path):
    ckpt = _checkpoint(tmp_path, "last.ckpt", b"the-last-weights")
    predictions = tmp_path / "p.json"
    predictions.write_text("{}")
    path = _identity_file(tmp_path, "last", str(ckpt), predictions)
    provenance = prediction_metrics.provenance_from_identity(identity_module.read_identity(path), predictions=predictions)
    assert provenance["evaluated_checkpoint_role"] == "last"
    assert provenance["evaluated_checkpoint_path"] == str(ckpt)
    assert provenance["evaluated_checkpoint_sha256"] == hashlib.sha256(b"the-last-weights").hexdigest()
    assert provenance["evaluation_unit_id"] == "eval_x/task2/fold0"


def test_a_stale_best_ckpt_cannot_contaminate_current_or_last_provenance(tmp_path):
    """The exact defect: best.ckpt on disk was hashed as the evaluated checkpoint regardless."""
    stale = _checkpoint(tmp_path, "best.ckpt", b"stale-best-from-an-earlier-attempt")
    predictions = tmp_path / "p.json"
    predictions.write_text("{}")
    stale_digest = hashlib.sha256(stale.read_bytes()).hexdigest()

    current = prediction_metrics.provenance_from_identity(
        identity_module.read_identity(_identity_file(tmp_path, "current", None, predictions)),
        predictions=predictions,
    )
    assert current["evaluated_checkpoint_sha256"] is None
    assert current["evaluated_state"] == identity_module.STATE_IN_MEMORY_FINAL

    last = _checkpoint(tmp_path, "last.ckpt", b"the-actually-evaluated-weights")
    resolved = prediction_metrics.provenance_from_identity(
        identity_module.read_identity(_identity_file(tmp_path, "last", str(last), predictions)),
        predictions=predictions,
    )
    assert resolved["evaluated_checkpoint_sha256"] == hashlib.sha256(last.read_bytes()).hexdigest()
    assert resolved["evaluated_checkpoint_sha256"] != stale_digest


def test_provenance_from_a_different_run_is_refused(tmp_path):
    predictions = tmp_path / "p.json"
    predictions.write_text("{}")
    other = tmp_path / "someone_elses.json"
    other.write_text("{}")
    path = _identity_file(tmp_path, "current", None, predictions)
    with pytest.raises(SystemExit, match="Refusing to attach"):
        prediction_metrics.provenance_from_identity(identity_module.read_identity(path), predictions=other)


def _complete_provenance(**overrides):
    provenance = {
        "campaign_id": "d0",
        "evaluation_id": "eval_x",
        "evaluation_unit_id": "eval_x/task2/fold0",
        "evaluated_checkpoint_role": "best",
        "evaluated_checkpoint_path": "/runs/checkpoints/best.ckpt",
        "evaluated_checkpoint_sha256": "a" * 64,
        "evaluated_state": "checkpoint_file",
        "evaluated_epoch": 5,
        "evaluated_global_step": 500,
        "prediction_path": "/runs/predictions/p.json",
        "split_id": "test",
        "git_sha": "abc123",
    }
    provenance.update(overrides)
    return provenance


def test_a_complete_record_passes_validation():
    record = build_record(2, 0, "completed", {"dsc": 0.5}, kind="seg", provenance=_complete_provenance())
    assert provenance_problems(record) == []


@pytest.mark.parametrize(
    "field",
    ["evaluation_id", "evaluation_unit_id", "evaluated_checkpoint_role", "prediction_path", "git_sha"],
)
def test_a_completed_record_may_not_leave_an_applicable_field_null(field):
    """Creating the key is not filling it in."""
    record = build_record(2, 0, "completed", {"dsc": 0.5}, kind="seg", provenance=_complete_provenance(**{field: None}))
    assert any(field in problem for problem in provenance_problems(record))


def test_a_file_backed_role_must_name_its_checkpoint_and_digest():
    record = build_record(
        2, 0, "completed", {"dsc": 0.5}, kind="seg", provenance=_complete_provenance(evaluated_checkpoint_sha256=None)
    )
    assert any("evaluates a checkpoint file" in problem for problem in provenance_problems(record))


def test_current_must_not_carry_a_checkpoint_digest():
    record = build_record(
        2,
        0,
        "completed",
        {"dsc": 0.5},
        kind="seg",
        provenance=_complete_provenance(evaluated_checkpoint_role="current", evaluated_state="in_memory_final"),
    )
    problems = provenance_problems(record)
    assert any("never loaded" in problem for problem in problems)


def test_current_is_complete_with_epoch_and_step_and_no_checkpoint():
    record = build_record(
        2,
        0,
        "completed",
        {"dsc": 0.5},
        kind="seg",
        provenance=_complete_provenance(
            evaluated_checkpoint_role="current",
            evaluated_state="in_memory_final",
            evaluated_checkpoint_path=None,
            evaluated_checkpoint_sha256=None,
        ),
    )
    assert provenance_problems(record) == []


@pytest.mark.parametrize("status", ["failed", "missing", "skipped", "invalid", "incomplete"])
def test_a_run_that_produced_no_evaluation_needs_no_provenance(status):
    """Null is right where the field genuinely does not apply."""
    record = build_record(2, 0, status, {}, kind="seg", provenance=None)
    assert provenance_problems(record) == []


def test_an_invalid_completed_record_cannot_be_written(tmp_path):
    """The guarantee is at the write, not at the read: an under-specified record never lands."""
    with pytest.raises(ValueError, match="incomplete fomo26-metrics-v2 record"):
        write_metrics_json(tmp_path / "metrics.json", build_record(2, 0, "completed", {"dsc": 0.5}, kind="seg"))
    assert not (tmp_path / "metrics.json").exists()


def test_read_can_demand_complete_v2_provenance(tmp_path):
    """Historical records already on disk predate the write-time guarantee, so readers can ask."""
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(build_record(2, 0, "completed", {"dsc": 0.5}, kind="seg")))
    assert read_metrics_json(path)["metrics"]["dsc"] == 0.5  # permissive read still works
    with pytest.raises(ValueError, match="does not carry complete v2 provenance"):
        read_metrics_json(path, require_complete_v2=True)


# --- identity must not depend on the ambient environment ----------------------------------------


def test_evaluation_identity_comes_from_the_config_not_the_environment(tmp_path, monkeypatch):
    """An environment fallback made the record depend on what the job happened to inherit.

    `resolved_config.yaml` is what a reviewer reads; an id that lives only in the environment is
    invisible there and cannot be reproduced from the config.
    """
    monkeypatch.setenv("FOMO26_EVALUATION_ID", "ambient_should_be_ignored")
    monkeypatch.setenv("FOMO26_CAMPAIGN_ID", "ambient_should_be_ignored")
    cfg = OmegaConf.create({"test_task": "SEG902_FOMO26_Task2_lesion", "data": {"test_split": "test", "fold": 0}})
    identity = identity_module.build_identity(
        cfg,
        role="current",
        checkpoint_path=None,
        prediction_path=str(tmp_path / "p.json"),
        run_dir=str(tmp_path),
        epoch=1,
        global_step=1,
        status="completed",
        task_number=2,
    )
    assert identity["evaluation_id"] is None
    assert identity["campaign_id"] is None
    assert identity["evaluation_unit_id"] is None


def test_identity_records_the_executing_commit_separately_from_the_launch_commit(tmp_path, monkeypatch):
    """A requeued or resumed job executes a tree that may differ from the submit-time commit."""
    monkeypatch.setenv("FOMO26_LAUNCH_GIT_COMMIT", "1111111111111111111111111111111111111111")
    cfg = OmegaConf.create({"test_task": "SEG902_FOMO26_Task2_lesion", "data": {"test_split": "test", "fold": 0}})
    identity = identity_module.build_identity(
        cfg,
        role="current",
        checkpoint_path=None,
        prediction_path=str(tmp_path / "p.json"),
        run_dir=str(tmp_path),
        epoch=1,
        global_step=1,
        status="completed",
        task_number=2,
    )
    assert identity["launch_git_sha"] == "1111111111111111111111111111111111111111"
    # The executing SHA is resolved from this checkout, so it is a real commit and not the label.
    assert identity["git_sha"] != identity["launch_git_sha"]
    assert identity["git_sha"] and len(identity["git_sha"].removesuffix("-dirty")) == 40


# --- the launcher must fail closed --------------------------------------------------------------


# Five tests here drove `misc/jean_zay/submit_downstream.sh` and its collector: they asserted that
# the launcher reads the evaluated-checkpoint identity instead of rediscovering artefacts. The HPC
# orchestration rail is not published, so they were removed with it. The identity contract itself
# is still covered by the remaining tests in this module, which read the config and the manifest.
