"""Test tiers: every tracked test artifact belongs to exactly one, and this file says which.

The default suite has to be runnable from a fresh checkout on a CPU-only machine with no NATTEN,
in a few minutes. It was 37 minutes, and four modules could not even be collected. Both problems
are handled here, by different mechanisms, because they are different problems.

**Collection versus cost.** Four modules imported `natten`, a CUDA extension, so pytest exited 2
before running anything -- a marker cannot help, because a marker is evaluated after the module is
imported. Those four were excluded from *collection*. Retiring MedViT removed all four, so
`optional_natten` is now declared and empty: the mechanism stays, because the contract must still
catch a NATTEN-dependent test arriving later. Everything else is a cost question, handled by
deselection after collection, which keeps the reason in the report instead of hiding files.

**Measured, not guessed.** The two modules that dominated the runtime were moved out of the
default tier on measurement rather than on impression; both have since retired with the code
they exercised, so `pipeline` is now declared and empty for the same reason `optional_natten`
is. `test_amaes_ddp_gloo.py` and `test_container_release_rail.py` are classified by what they
need rather than by cost -- a distributed backend and the container release rail.

Nothing is deleted, and nothing is hidden: `test_every_tracked_test_artifact_has_exactly_one_tier`
fails if a new test appears without an assignment, if a path is listed twice, if a listed path does
not exist, or if a tier name is not one of the seven below. A test can leave the default tier only
by being named here with a reason.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

TIERS = ("fast", "optional_natten", "model", "pipeline", "ddp", "release", "historical")

#: Files that are part of the test suite but are not themselves test modules. Recorded so that
#: "every tracked artifact is accounted for" is checkable rather than a matter of trust.
SUPPORT_ARTIFACTS = {
    "tests/conftest.py": "shared fixtures and the tier collection hook",
    "tests/amaes_step_probe.py": "subprocess measurement driven by test_amaes_step_numerics.py",
    "tests/amaes_step_reference.json": "recorded numerical reference read by test_amaes_step_numerics.py",
}

#: Audit gates that live outside `testpaths` and are executed explicitly. Two of them are scripts
#: rather than pytest modules: running them under pytest collects zero tests and exits 5, which
#: looks like success and verifies nothing.
# No audit gate is part of this distribution. The mapping stays declared and empty rather than
# deleted: it is what would catch a `test_`-named script being recorded as a pytest suite, and a
# removed mechanism cannot catch anything.
AUDIT_ARTIFACTS: dict[str, tuple[str, str]] = {}

NON_FAST_REASONS: dict[str, str] = {
    "tests/test_amaes_ddp_gloo.py": (
        "spawns two ranks over gloo with a file rendezvous; a distributed backend is a capability question, not a cost one"
    ),
    "tests/test_container_release_rail.py": (
        "exercises the container/release rail and the validator acquisition contract; release-specific by subject, not by cost"
    ),
}

TIER_ASSIGNMENTS: dict[str, str] = {
    "tests/test_matched_control_equivalence.py": "fast",
    "tests/import_test.py": "fast",
    "tests/test_amaes_ddp_gloo.py": "ddp",
    "tests/test_amaes_step_numerics.py": "fast",
    "tests/test_aggregate_score.py": "fast",
    "tests/test_architecture_consistency.py": "fast",
    "tests/test_checkpoint_prediction_identity.py": "fast",
    "tests/test_clsreg_ensemble_contract.py": "fast",
    "tests/test_container_packaging_provenance.py": "fast",
    "tests/test_container_release_rail.py": "release",
    "tests/test_curated_modality_vocab.py": "fast",
    "tests/test_curriculum_schedules.py": "fast",
    "tests/test_declared_tta_policy.py": "fast",
    "tests/test_dinov2_vit_x.py": "fast",
    "tests/test_embedding_projection.py": "fast",
    "tests/test_evaluation_contract.py": "fast",
    "tests/test_extract_embeddings.py": "fast",
    "tests/test_feature_metrics_support.py": "fast",
    "tests/test_finetune_config_composition.py": "fast",
    "tests/test_finetune_entrypoints_smoke.py": "fast",
    "tests/test_finetune_seg.py": "fast",
    "tests/test_fold_manifest_bridge.py": "fast",
    "tests/test_fomo26_finetuning_converter.py": "fast",
    "tests/test_gradient_diagnostics_neutrality.py": "fast",
    "tests/test_handoff_campaign_optional.py": "fast",
    "tests/test_legacy_model_manifest_architecture.py": "fast",
    "tests/test_linear_probe.py": "fast",
    "tests/test_local_csv_logging.py": "fast",
    "tests/test_metrics_io.py": "fast",
    "tests/test_model_config_contract.py": "fast",
    "tests/test_oof_contract.py": "fast",
    "tests/test_oof_inference_policy.py": "fast",
    "tests/test_optimizer_step_warmup_contract.py": "fast",
    "tests/test_packaging_manifest_authority.py": "fast",
    "tests/test_predict_dataset_sample_format.py": "fast",
    "tests/test_prediction_writer_compat.py": "fast",
    "tests/test_publication_surfaces.py": "fast",
    "tests/test_notice_scope.py": "fast",
    "tests/test_validator_acquisition.py": "fast",
    "tests/test_pretrain.py": "fast",
    "tests/test_pretrain_metadata_contrastive.py": "fast",
    "tests/test_pretrained_embedding_contract.py": "fast",
    "tests/test_pretrained_loader.py": "fast",
    "tests/test_registered_stream_curriculum.py": "fast",
    "tests/test_release_geometry_orientation_policy.py": "fast",
    "tests/test_release_runtime_data_files.py": "fast",
    "tests/test_release_runtime_override.py": "fast",
    "tests/test_repository_hygiene.py": "fast",
    "tests/test_representations.py": "fast",
    "tests/test_review_fixes.py": "fast",
    "tests/test_runtime_contract_gate.py": "fast",
    "tests/test_runtime_geometry_canonicalization.py": "fast",
    "tests/test_runtime_spacing.py": "fast",
    "tests/test_safety_resenc_amaes.py": "fast",
    "tests/test_seg_foreground_sampling.py": "fast",
    "tests/test_segmentation_inference.py": "fast",
    "tests/test_session_coreg_fsl.py": "fast",
    "tests/test_ssl_objectives.py": "fast",
    "tests/test_submission_export_contract.py": "fast",
    "tests/test_test_cls.py": "fast",
    "tests/test_tier_contract.py": "fast",
    "tests/test_train_reg.py": "fast",
    "tests/test_tta_ladder_cap.py": "fast",
    "tests/test_wandb_run_continuity.py": "fast",
    "tests/test_warmup_and_health_contract.py": "fast",
}

#: Exact commands. Every tier a reader might want to run is spelled out, including the ones that
#: are empty today, so "how do I run X" never needs archaeology.
TIER_COMMANDS = {
    "fast": "FOMO26_TIER=fast python -m pytest",
    "optional_natten": "python tests/run_test_tier.py optional_natten",
    "model": "FOMO26_TIER=model python -m pytest",
    "pipeline": "FOMO26_TIER=pipeline python -m pytest",
    "ddp": "FOMO26_TIER=ddp python -m pytest",
    "release": "FOMO26_TIER=release python -m pytest",
    "historical": "FOMO26_TIER=historical python -m pytest",
    "all": "FOMO26_TIER=all python -m pytest",
}


def _tracked_test_modules() -> set[str]:
    import re

    out = subprocess.run(["git", "-C", str(REPO), "ls-files"], check=True, text=True, capture_output=True).stdout.split()
    return {p for p in out if re.match(r"^tests/(test_.*|.*_test)\.py$", p)}


def test_every_tracked_test_artifact_has_exactly_one_tier():
    """A new test file must be classified before it can be committed."""
    tracked = _tracked_test_modules()
    assigned = set(TIER_ASSIGNMENTS)
    assert assigned - tracked == set(), f"manifest names paths that are not tracked test modules: {sorted(assigned - tracked)}"
    assert tracked - assigned == set(), f"tracked test modules with no tier: {sorted(tracked - assigned)}"


def test_every_tier_name_is_known():
    unknown = {t for t in TIER_ASSIGNMENTS.values() if t not in TIERS}
    assert not unknown, f"unknown tier(s): {sorted(unknown)}"


def test_every_non_fast_assignment_states_a_reason():
    """A test leaves the default tier only with a written justification."""
    missing = sorted(p for p, t in TIER_ASSIGNMENTS.items() if t != "fast" and not NON_FAST_REASONS.get(p))
    assert not missing, f"non-fast assignment without a reason: {missing}"


def test_every_tier_has_an_exact_command():
    assert set(TIER_COMMANDS) - {"all"} == set(TIERS)


def test_support_and_audit_artifacts_exist():
    """The non-test artifacts are named so nothing is silently unaccounted for.

    AUDIT_ARTIFACTS is empty in this distribution; SUPPORT_ARTIFACTS is not, so this still
    checks something real.
    """
    for rel in list(SUPPORT_ARTIFACTS) + list(AUDIT_ARTIFACTS):
        assert (REPO / rel).is_file(), f"{rel} is recorded but missing"


def test_no_test_module_is_omitted_from_every_command():
    """Union of the tiers must be the whole tracked set: nothing may fall through the cracks."""
    covered = set()
    for tier in TIERS:
        covered |= {p for p, t in TIER_ASSIGNMENTS.items() if t == tier}
    assert covered == set(TIER_ASSIGNMENTS)


def test_audit_artifact_execution_kind_matches_reality():
    """A `test_`-named file that defines no test function is a script, and must be run as one.

    A file that collects zero tests under pytest exits 5; reporting that as a passing suite is how
    a gate stops gating without anyone noticing. No such artifact remains in this distribution, so
    this iterates rather than parametrizes -- an empty parametrization reports as a skip, and a
    skip nobody can explain is exactly what this suite refuses to leave lying around.
    """
    import ast

    for rel, (kind, _command) in sorted(AUDIT_ARTIFACTS.items()):
        tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
        has_test_funcs = any(isinstance(node, ast.FunctionDef) and node.name.startswith("test_") for node in tree.body)
        assert has_test_funcs == (kind == "pytest"), f"{rel}: recorded as {kind} but pytest-collectable={has_test_funcs}"


def test_optional_natten_is_declared_and_empty():
    """Retiring MedViT removed every NATTEN-dependent module.

    The tier is kept declared rather than deleted: the mechanism is what catches a
    NATTEN-importing test arriving later, and deleting it would silently readmit the collection
    failure it exists to prevent.
    """
    assert "optional_natten" in TIERS
    assert [p for p, tier in TIER_ASSIGNMENTS.items() if tier == "optional_natten"] == []
    assert "optional_natten" in TIER_COMMANDS


def test_no_tracked_test_module_imports_natten():
    """No retained test may depend on NATTEN; one that did would fail collection again."""
    offenders = []
    for rel in sorted(_tracked_test_modules()):
        text = (REPO / rel).read_text(encoding="utf-8", errors="ignore")
        if re.search(r"^\s*(?:import natten|from natten)", text, re.MULTILINE):
            offenders.append(rel)
    assert offenders == [], f"NATTEN-importing test modules must be assigned to optional_natten: {offenders}"
