"""The container must write its prediction to exactly the path the organizers ask for.

The release runtime is pinned to gardening_tools 0.3.2 (the version that trained the packaged
checkpoints). Its writer appends the extension to the path it is given, where 0.3.5 writes the path
verbatim. Handing 0.3.2 the official ``/output/<case>.nii.gz`` therefore produces
``<case>.nii.gz.nii.gz``: predict.py exits 0, the validator reports
``prediction_runs_successfully`` and then ``output_file_exists`` fails, because nothing was written
where anyone looks for it.
"""

from __future__ import annotations

import pytest
from finetuning.container import fomo_ensemble_predict as predict


def test_new_api_receives_the_path_verbatim(monkeypatch):
    seen = {}

    def writer(logits, outpath, properties, compression=9):
        seen["outpath"] = outpath

    monkeypatch.setattr(predict, "save_prediction_from_logits", writer)
    predict._save_prediction([0], "/output/case_a.nii.gz", {})
    assert seen["outpath"] == "/output/case_a.nii.gz"


def test_old_api_receives_the_stem_so_the_file_lands_on_the_asked_path(monkeypatch):
    seen = {}

    def writer(logits, outpath, properties, save_format="nii.gz", compression=9):
        # Mirrors 0.3.2, which appends the extension itself.
        seen["written"] = outpath + ".nii.gz"
        seen["save_format"] = save_format

    monkeypatch.setattr(predict, "save_prediction_from_logits", writer)
    predict._save_prediction([0], "/output/case_a.nii.gz", {})
    assert seen["written"] == "/output/case_a.nii.gz"
    assert seen["save_format"] == "nii.gz"


def test_unexpected_extension_is_refused_rather_than_written_somewhere_else(monkeypatch):
    def writer(logits, outpath, properties, save_format="nii.gz", compression=9):
        raise AssertionError("must not be called")

    monkeypatch.setattr(predict, "save_prediction_from_logits", writer)
    with pytest.raises(ValueError, match="Expected a .nii.gz output path"):
        predict._save_prediction([0], "/output/case_a.nii", {})
