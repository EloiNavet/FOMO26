"""The build-time gate that refuses an image whose finetuned weights do not fully load."""

from __future__ import annotations

import importlib.util
import pytest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "finetuning" / "container" / "verify_runtime_contract.py"
spec = importlib.util.spec_from_file_location("verify_runtime_contract", MODULE)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def test_segmentation_tasks_are_the_ones_carrying_finetuned_decoders():
    # Tasks 1/3/5 ship a ClsRegHead and load 450/450; only the segmentation images carry a decoder
    # whose parameter names can drift, so only they need the completeness check.
    assert set(verify.SEG_TASKS) == {"task2", "task4"}
    assert verify.SEG_TASKS["task2"] == 2
    assert verify.SEG_TASKS["task4"] == 3


def test_the_pinned_runtime_version_is_the_one_that_trained_the_checkpoints():
    assert verify.REQUIRED_RUNTIME_VERSIONS["gardening_tools"] == "0.3.2"


def test_fail_exits_nonzero_so_the_build_cannot_be_sealed():
    with pytest.raises(SystemExit) as excinfo:
        verify._fail("decoder would stay random")
    assert excinfo.value.code == 1


def test_input_channels_come_from_the_runtime_modality_contract():
    # Task 2 stacks three modalities at runtime while its checkpoint config reports one. Building
    # the model from that fallback fabricates a stem-shaped mismatch the real runtime never has,
    # which would fail a sound image (or, with a different fallback, mask a real defect).
    source = MODULE.read_text()
    assert "MODALITY_ORDER" in source
    # The name may still appear in prose explaining why it is wrong; what must be gone is the
    # lookup that built the model from it.
    assert 'OmegaConf.select(config, "data.n_modalities"' not in source


def test_gate_requires_shape_compatibility_not_just_matching_names():
    # A name that matches with the wrong shape is dropped by load_state_dict exactly as silently
    # as a name that does not match at all, so names alone cannot establish a complete load.
    source = MODULE.read_text()
    assert "shape_mismatches" in source
    assert "transferable != len(model_keys)" in source
