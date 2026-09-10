"""Stage 5: FOMO26 curated diffusion modality vocabulary in PretrainDataset.

Verifies that the curated vocab is (a) fully backward-compatible when disabled
(default), and (b) resolves ADC / DWI_B1000 / DWI_B0 / DWI_TRACE as *distinct*
modalities -- never the collapsed generic ``dwi`` -- when explicitly enabled. GRE/T1c
stay out of the default pretraining set unless explicitly enabled.
"""

import pytest

torch = pytest.importorskip("torch")
from asparagus.modules.datasets.PretrainDataset import PretrainDataset  # noqa: E402


@pytest.fixture
def curated_vocab():
    """Enable the curated vocab for a test, then restore the previous state."""
    prev = PretrainDataset._USE_CURATED_VOCAB
    PretrainDataset.use_curated_diffusion_vocab(True)
    try:
        yield
    finally:
        PretrainDataset._USE_CURATED_VOCAB = prev


# --------------------------------------------------------------------------- #
# Backward compatibility (curated vocab disabled = current behaviour)
# --------------------------------------------------------------------------- #
def test_legacy_vocab_unchanged_by_default():
    assert PretrainDataset.curated_diffusion_vocab_enabled() is False
    vocab = PretrainDataset.modality_vocab()
    assert vocab["dwi"] == 4 and vocab["adc"] == 6 and vocab["dwi_trace"] == 5
    assert "dwi_b1000" not in vocab and "dwi_b0" not in vocab and "t2star" not in vocab


def test_legacy_dwi_bval_collapses_to_dwi():
    assert PretrainDataset._infer_modality_from_filename("sub-1_ses-1_dwi_bval1000.nii.gz") == "dwi"
    assert PretrainDataset._modality_id_from_filename("sub-1_ses-1_dwi_bval1000.nii.gz") == 4


# --------------------------------------------------------------------------- #
# Curated vocab enabled
# --------------------------------------------------------------------------- #
def test_curated_ids_are_additive_and_distinct(curated_vocab):
    vocab = PretrainDataset.modality_vocab()
    # Legacy ids are never renumbered.
    assert vocab["t1w"] == 0 and vocab["dwi"] == 4 and vocab["adc"] == 6 and vocab["dwi_trace"] == 5
    # ADC and DWI_B1000 are distinct ids, and neither equals the generic dwi id.
    assert vocab["dwi_b1000"] == 15 and vocab["dwi_b0"] == 16 and vocab["t2star"] == 17
    assert vocab["adc"] != vocab["dwi"] and vocab["dwi_b1000"] != vocab["dwi"]
    assert vocab["adc"] != vocab["dwi_b1000"]


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("sub-1_ses-1_DWI_B1000.nii.gz", "dwi_b1000"),
        ("sub-1_ses-1_ADC.nii.gz", "adc"),
        ("sub-1_ses-1_DWI_B0.nii.gz", "dwi_b0"),
        ("sub-1_ses-1_DWI_TRACE.nii.gz", "dwi_trace"),
        ("sub-1_ses-1_T2star.nii.gz", "t2star"),
        ("sub-1_ses-1_FLAIR.nii.gz", "flair"),
    ],
)
def test_curated_filename_inference(curated_vocab, filename, expected):
    assert PretrainDataset._infer_modality_from_filename(filename) == expected


def test_curated_adc_and_b1000_get_different_ids(curated_vocab):
    adc_id = PretrainDataset._modality_id_from_filename("sub-1_ses-1_ADC.nii.gz")
    b1000_id = PretrainDataset._modality_id_from_filename("sub-1_ses-1_DWI_B1000.nii.gz")
    dwi_id = PretrainDataset.MODALITY_TO_ID["dwi"]
    assert adc_id != b1000_id
    assert adc_id != dwi_id and b1000_id != dwi_id


def test_curated_legacy_dwi_bval_still_collapses(curated_vocab):
    # Even with curated vocab on, a raw legacy dwi_bval file collapses to dwi (not b1000).
    assert PretrainDataset._infer_modality_from_filename("sub-1_ses-1_dwi_bval1000.nii.gz") == "dwi"


def test_t2s_alias_normalizes(curated_vocab):
    ids = PretrainDataset.normalize_modality_ids(["t2s"])
    assert ids == (PretrainDataset._CURATED_MODALITY_ADDITIONS["t2star"],)


def test_normalize_modality_ids_accepts_curated_names(curated_vocab):
    ids = PretrainDataset.normalize_modality_ids(["dwi_b1000", "adc", "flair"])
    assert ids == (15, 6, 2)


def test_normalize_rejects_curated_names_when_disabled():
    with pytest.raises(ValueError):
        PretrainDataset.normalize_modality_ids(["dwi_b1000"])


# --------------------------------------------------------------------------- #
# Default pretraining modality set (GRE/T1c gating)
# --------------------------------------------------------------------------- #
def test_default_pretrain_modalities_excludes_gre_t1c():
    default = PretrainDataset.default_pretrain_modalities()
    assert default == ("t1w", "t2w", "flair", "t2star", "swi", "dwi_b1000", "adc")
    assert "gre" not in default and "t1c" not in default


def test_default_pretrain_modalities_can_enable_gre_t1c():
    default = PretrainDataset.default_pretrain_modalities(enable_gre_t1c=True)
    assert default[-2:] == ("gre", "t1c")
