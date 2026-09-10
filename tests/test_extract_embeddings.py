"""Embedding output contract: deterministic ordering, sample-id alignment, manifest fields."""

import csv
import json
import numpy as np
import pytest
from finetuning.fomo26_inference import extract_embeddings as ee


def test_write_embedding_outputs_files_and_manifest(tmp_path):
    emb = np.arange(6, dtype=np.float32).reshape(3, 2)
    ids = ["subA", "subB", "subC"]
    manifest = ee.write_embedding_outputs(
        tmp_path,
        emb,
        ids,
        checkpoint_sha256="deadbeef",
        provenance={"architecture": "resenc_b", "source_used": "online", "downstream_finetuned_weights": None},
    )
    assert (tmp_path / "embeddings.npy").is_file()
    assert (tmp_path / "sample_ids.csv").is_file()
    assert (tmp_path / "embedding_manifest.json").is_file()

    assert manifest["schema_version"] == ee.SCHEMA_VERSION
    assert manifest["num_samples"] == 3
    assert manifest["embedding_dim"] == 2
    assert manifest["shape"] == [3, 2]
    assert manifest["dtype"] == "float32"
    assert manifest["backbone_frozen"] is True
    assert manifest["checkpoint_sha256"] == "deadbeef"
    assert manifest["architecture"] == "resenc_b"
    assert manifest["source_used"] == "online"
    # Tasks 6/7 embeddings come from the pretrained checkpoint; no finetuned weights take part.
    assert manifest["downstream_finetuned_weights"] is None
    assert "No official Task 6/7 performance" in manifest["note"]


def test_sample_ids_align_with_rows(tmp_path):
    emb = np.array([[1.0], [2.0], [3.0]])
    ids = ["s0", "s1", "s2"]
    ee.write_embedding_outputs(tmp_path, emb, ids)
    with (tmp_path / "sample_ids.csv").open() as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["row", "sample_id"]
    assert [r[1] for r in rows[1:]] == ids
    assert [int(r[0]) for r in rows[1:]] == [0, 1, 2]
    loaded = np.load(tmp_path / "embeddings.npy")
    assert loaded.shape == (3, 1)


def test_row_count_mismatch_rejected(tmp_path):
    with pytest.raises(ValueError, match="must match embedding rows"):
        ee.write_embedding_outputs(tmp_path, np.zeros((2, 4)), ["only_one"])


def test_duplicate_sample_ids_rejected(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        ee.write_embedding_outputs(tmp_path, np.zeros((2, 4)), ["dup", "dup"])


def test_non_2d_embeddings_rejected(tmp_path):
    with pytest.raises(ValueError, match="2-D"):
        ee.write_embedding_outputs(tmp_path, np.zeros((5,)), ["a", "b", "c", "d", "e"])


def test_manifest_is_deterministic(tmp_path):
    emb = np.ones((2, 3))
    ee.write_embedding_outputs(tmp_path / "a", emb, ["x", "y"])
    ee.write_embedding_outputs(tmp_path / "b", emb, ["x", "y"])
    a = json.loads((tmp_path / "a" / "embedding_manifest.json").read_text())
    b = json.loads((tmp_path / "b" / "embedding_manifest.json").read_text())
    a.pop("created_at")
    b.pop("created_at")
    assert a == b


def test_sample_id_strips_medical_suffixes():
    from pathlib import Path

    assert ee._sample_id(Path("/x/sub-01_flair.nii.gz")) == "sub-01_flair"
    assert ee._sample_id(Path("/x/sub-02.npy")) == "sub-02"
