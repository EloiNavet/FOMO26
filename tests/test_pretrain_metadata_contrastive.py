import math
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch
from asparagus.functional.frequency import frequency_domain_loss, spatial_detail_loss
from asparagus.functional.lr_scheduling import simple_warmup_cosine_decay_schedule
from asparagus.functional.metrics import (
    distribution as distribution_metrics,
    loss as loss_metrics,
    reconstruction as reconstruction_metrics,
    stability as stability_metrics,
)
from asparagus.functional.versioning import detect_id
from asparagus.modules.callbacks import ProfilerCallback
from asparagus.modules.data_modules.pretraining import (
    PretrainDataModule,
    SameSessionMultimodalSampler,
    StatefulReplacementSampler,
    _distributed_sampler_context,
)
from asparagus.modules.datasets.PretrainDataset import PretrainDataset
from asparagus.modules.lightning_modules.self_supervised import SelfSupervisedModule
from asparagus.modules.lightning_modules.ssl import views as ssl_views
from asparagus.modules.transforms.frepa_lite import Torch_FrepaLiteFrequencyCorruption
from asparagus.pipeline.auto_configuration.checkpoint import (
    resolve_training_resume_checkpoint,
    resolve_training_resume_seed,
)
from asparagus.pipeline.auto_configuration.experiment_setup import _is_rank_zero_process as setup_is_rank_zero_process
from asparagus.pipeline.auto_configuration.logging import HydraConfigWandbLogger
from asparagus.pipeline.run.pretrain import (
    StopAfterValidationStep,
    _is_rank_zero_process as pretrain_is_rank_zero_process,
    _pretrain_progress_callbacks,
    _resolve_pretrain_modality_ids,
)
from collections import Counter, defaultdict
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pathlib import Path
from types import SimpleNamespace

# 1. Raw Data Profiles from FOMO300K (Sampled from your EDA)
RAW_PATHOLOGIES = [
    # Controls
    "Control",
    "Typically Developing",
    "CN",
    "normal",
    "Normal",
    "nondemented",
    "Nondemented",
    "control",
    "NeuroTypical",
    "HC",
    "healthy control",
    "nh",
    # Alzheimer / Dementia
    "Dementia",
    "very_mild_dementia",
    "mild_dementia",
    "moderate_dementia",
    "Demented",
    "Converted",
    "AD",
    # Parkinson / Movement
    "Parkinson",
    "PD",
    "Parkinson’s disease",
    "Parkinson’s - no/mild hyposmia",
    "Upper limb dystonia",
    "Parkinson's disease - mild cognitive impairment",
    "Cervical dystonia",
    # Psych / Neurodevelopmental
    "ADHD-MPH",
    "ADHD-NAIVE",
    "ADHD",
    "ADHD-Inattentive",
    "Autism Spectrum Disorder",
    "Schizophrenia without current auditory hallucinations",
    "First-episode schizophrenia",
    "Schizophrenia/Schizoaffective",
    "SCHZ",
    "Bipolar",
    "BIPOLAR",
    "BP",
    "Depression - no treatment",
    "Major depressive disorder",
    "Obsessive compulsive disorder",
    "dyslexia",
    "Cocaine use disorder",
    "Cocaine use disorder - Sham",
    "Cocaine use disorder - Treatment",
    # Tumor / Oncology
    "Brain Tumor",
    "GMB",
    "neuroectodermal tumor",
    "Tumor",
    "Pons Gliom",
    "pylocytic astrocytom",
    "DNET",
    "medulloblastoma",
    "astrocytoma",
    "oligodendroglioma",
    "glioblastoma",
    "Meningioma",
    "Ependymoma",
    "Ganglioglioma",
    "Pituitary adenomas",
    "Diffuse large B-cell lymphoma",
    # Vascular (Stroke / Infarct / Hemorrhage)
    "Stroke",
    "Infarct",
    "HIE",
    "Intracerebral hemorrhage",
    "subdurale hemorrhagy",
    "Premature Infarct",
    "intraventricular hemorrhage",
    "Cortical cerebral infarction",
    "Brain aneurysm(s)",
    "Perinatal stroke",
    # Other Structural / Epilepsy / Injury
    "Hydrocephalus",
    "Premature PVL",
    "encephalocele",
    "macrocephaly",
    "PVL",
    "Congenital CMV infection",
    "Arachnoidal cyst",
    "focal cortical dysplasia",
    "Temporal or parietal lobe epilepsy",
    "Epilepsy",
    "TBI",
    "Traumatic brain injury",
    "Multiple sclerosis",
    "Brain abscess",
    "dysplas",
    "sclerosis",
    "Hearing loss",
    # Known non-pathological exclusion / non-brain disease labels
    "Motion artefact",
    "Fibromyalgia",
    "Osteoarthritis",
]

RAW_MANUFACTURERS = [
    "Siemens",
    "SIEMENS",
    "Siemens Healthineers",
    "SIEMENS ",
    "Siemens healthcare gmbh",
    "Philips",
    "Philips Medical Systems",
    "Philips Healthcare",
    "Philips medical systems",
    "GE",
    "GE MEDICAL SYSTEMS",
    "General Electrics",
    "Ge medical systems",
    "Toshiba",
    "Toshiba_mec",
    "Canon_mec",
    "Hitachi",
    "Hitachi medical corporation",
    "Ningbo Xingaoyi",
    "Mediso",
    "Brucker",
    "Visage pr",
    "Fuji film co., ltd.",
    None,
]


PATHOLOGY_RULES = [
    {
        "name": "movement",
        "class_id": 1,
        "keywords": ["parkinson", "movement", "dystonia"],
        "equals": ["pd"],
    },
    {
        "name": "control",
        "class_id": 0,
        "keywords": [
            "control",
            "normal",
            "nondemented",
            "neurotypical",
            "typically developing",
        ],
        "equals": ["cn", "hc", "nh"],
    },
    {
        "name": "ad_dementia",
        "class_id": 1,
        "keywords": ["alzheimer", "dement"],
        "equals": ["ad", "converted"],
    },
    {
        "name": "psych_neurodev",
        "class_id": 2,
        "keywords": [
            "schiz",
            "schizo",
            "schz",
            "bipolar",
            "bp",
            "adhd",
            "autism",
            "depress",
            "depressive",
            "obsessive",
            "ocd",
            "psychosis",
            "psych",
            "anxiety",
            "dyslex",
        ],
        "equals": [],
    },
    {
        "name": "tumor_oncology",
        "class_id": 3,
        "keywords": [
            "tumor",
            "gliom",
            "astrocytom",
            "meningiom",
            "ependymom",
            "medulloblastoma",
            "lymphoma",
            "adenoma",
            "oncolog",
            "metasta",
            "neuroectoderm",
            "gangliogliom",
            "pylocytic",
            "dnet",
            "gbm",
            "gmb",
            "pituitary",
        ],
        "equals": [],
    },
    {
        "name": "vascular",
        "class_id": 4,
        "keywords": ["stroke", "infarct", "hemorrhag", "aneurysm", "hie"],
        "equals": [],
    },
    {
        "name": "other_structural",
        "class_id": 5,
        "keywords": [
            "epilepsy",
            "seizure",
            "hydrocephalus",
            "cyst",
            "dysplas",
            "sclerosis",
            "malformation",
            "atrophy",
            "heterotopia",
            "injury",
            "lesion",
            "tbi",
            "structural",
            "developmental",
            "encephalocele",
            "macrocephaly",
            "pvl",
            "cmv",
            "abscess",
        ],
        "equals": [],
    },
]


MANUFACTURER_RULES = [
    {"name": "siemens", "class_id": 0, "keywords": ["siemens"]},
    {"name": "philips", "class_id": 1, "keywords": ["philips"]},
    {
        "name": "ge",
        "class_id": 2,
        "keywords": ["ge", "general electric", "general electrics"],
    },
    {"name": "toshiba_canon", "class_id": 3, "keywords": ["toshiba", "canon"]},
    {"name": "hitachi", "class_id": 4, "keywords": ["hitachi"]},
]


def test_pathology_mapping_coverage():
    """
    Tests that raw pathology strings map correctly to the 6 Macro-Classes (0-5)
    while explicit non-pathological or missing values remain ignored.
    """
    mapped_values = set()

    for raw_val in RAW_PATHOLOGIES:
        if str(raw_val).strip().lower() in PretrainDataset.KNOWN_NON_PATHOLOGY_GROUPS:
            continue
        sample_metadata = {"group": raw_val}
        result = PretrainDataset._select_common_metadata_fields(sample_metadata)
        mapped_values.add(result["pathology"])

    # Assert exhaustiveness: all pathology target buckets must be hit.
    expected_buckets = {0, 1, 2, 3, 4, 5}
    assert mapped_values == expected_buckets, f"Missing buckets! Got {mapped_values}, expected {expected_buckets}"

    # Assert specific strict boundary conditions to ensure substrings don't bleed
    def get_val(g):
        return PretrainDataset._select_common_metadata_fields({"group": g})["pathology"]

    # 0: Controls
    assert get_val("Control") == 0
    assert get_val("Typically Developing") == 0
    assert get_val("nondemented") == 0  # Tricky: Must map to 0, not 1 (demented)

    # 1: Neurodegenerative AD/PD/Dementia
    assert get_val("very_mild_dementia") == 1
    assert get_val("Converted") == 1
    assert get_val("FTD") == 1

    assert get_val("Parkinson's disease - normal cognition") == 1  # Tricky: Contains "normal", must override control
    assert get_val("Upper limb dystonia") == 1

    # 2: Psych
    assert get_val("BIPOLAR") == 2
    assert get_val("Autism Spectrum Disorder") == 2
    assert get_val("Major depressive disorder") == 2
    assert get_val("Obsessive compulsive disorder") == 2
    assert get_val("dyslexia") == 2
    assert get_val("Cocaine use disorder") == 2
    assert get_val("Cocaine use disorder - Sham") == 2
    assert get_val("Cocaine use disorder - Treatment") == 2

    # 3: Tumor
    assert get_val("pylocytic astrocytom") == 3

    # 4: Vascular
    assert get_val("subdurale hemorrhagy") == 4
    assert get_val("HIE") == 4

    # 5: Other
    assert get_val("Traumatic brain injury") == 5
    assert get_val("Premature PVL") == 5
    assert get_val("Brain abscess") == 5
    assert get_val("Hearing loss") == 5
    assert get_val("Fibromyalgia") == 5
    assert get_val("Osteoarthritis") == 5

    # -1 is reserved for missing or explicit non-pathological exclusions only.
    assert get_val("Motion artefact") == -1
    assert get_val(None) == -1
    assert get_val(float("nan")) == -1


def test_pathology_mapping_rejects_nonempty_unclassified_group():
    with pytest.raises(ValueError, match="Unclassified pathology group"):
        PretrainDataset._select_common_metadata_fields({"group": "Definitely New Disease"})


def test_pretrain_dataset_rejects_unclassified_pathology_metadata(tmp_path):
    sample_path = tmp_path / "PT001_A" / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.pt"
    sample_path.parent.mkdir(parents=True)
    torch.save(torch.ones(1, 4, 4, 4), sample_path)

    pd.DataFrame(
        [
            {
                "dataset": "PT001_A",
                "participant_id": "sub-01",
                "session_id": "ses-01",
                "age": 42,
                "sex": "M",
                "group": "Definitely New Disease",
            }
        ]
    ).to_csv(tmp_path / "participants.tsv", sep="\t", index=False)

    with pytest.raises(ValueError, match="Unclassified pathology group"):
        PretrainDataset([str(sample_path)], metadata_paths=[str(tmp_path / "participants.tsv")])


def test_manufacturer_mapping_coverage():
    """
    Tests that the 26 raw manufacturer strings correctly map to the 5 designated scanners
    or the Unmapped class (-1).
    """
    mapped_values = set()

    for raw_val in RAW_MANUFACTURERS:
        sample_metadata = {"manufacturer": raw_val}
        result = PretrainDataset._select_common_metadata_fields(sample_metadata)
        mapped_values.add(result["scanner_id"])

    # Assert exhaustiveness
    expected_buckets = {-1, 0, 1, 2, 3, 4}
    assert mapped_values == expected_buckets, f"Missing buckets! Got {mapped_values}, expected {expected_buckets}"

    # Assert specific string parsing robustness
    def get_val(m):
        return PretrainDataset._select_common_metadata_fields({"manufacturer": m})["scanner_id"]

    assert get_val("Siemens Healthineers") == 0
    assert get_val("Philips medical systems") == 1
    assert get_val("General Electrics") == 2
    assert get_val("Canon_mec") == 3
    assert get_val("Hitachi") == 4
    assert get_val("Brucker") == -1
    assert get_val("Visage pr") == -1
    assert get_val(None) == -1


def test_age_filtering():
    """
    Tests robust parsing of continuous variables, including noisy string formats.
    """

    def get_val(a):
        return PretrainDataset._select_common_metadata_fields({"age": a})["age"]

    # Valid integers and floats
    assert get_val(42) == 42.0
    assert get_val(42.5) == 42.5
    assert get_val("65") == 65.0
    assert get_val(" 80 ") == 80.0

    # Missing / Unparseable
    assert math.isnan(get_val("NaN"))
    assert math.isnan(get_val(None))
    assert math.isnan(get_val("Unknown"))


def test_sex_filtering():
    """
    Tests strict categorical mapping for demographic sex.
    """

    def get_val(s):
        return PretrainDataset._select_common_metadata_fields({"sex": s})["sex"]

    assert get_val("M") == 0
    assert get_val("m") == 0
    assert get_val("F") == 1
    assert get_val(" f ") == 1
    assert get_val("NaN") == -1
    assert get_val("Other") == -1
    assert get_val(None) == -1


def test_pretrain_dataset_loads_tsv_metadata(tmp_path):
    sample_path = tmp_path / "PT001_A" / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.pt"
    sample_path.parent.mkdir(parents=True)
    torch.save(torch.ones(1, 4, 4, 4), sample_path)

    participants = pd.DataFrame(
        [
            {
                "dataset": "PT001_A",
                "participant_id": "sub-01",
                "session_id": "ses-01",
                "Age": 42,
                "Sex": "M",
                "Group": "Control",
                "Manufacturer": "Siemens",
            }
        ]
    )
    participants.to_csv(tmp_path / "participants.tsv", sep="\t", index=False)

    dataset = PretrainDataset([str(sample_path)], metadata_paths=[str(tmp_path / "participants.tsv")])
    item = dataset[0]

    assert item["age"] == 42.0
    assert item["sex"] == 0
    assert item["pathology"] == 0
    assert item["fine_pathology"] == PretrainDataset._stable_int_hash("control")
    assert item["scanner_id"] == 0
    assert item["metadata"]["age"] == 42
    assert item["metadata"]["sex"] == "M"


def test_pretrain_dataset_reuses_unchanged_metadata_table_cache(tmp_path):
    sample_path = tmp_path / "PT001_A" / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.pt"
    sample_path.parent.mkdir(parents=True)
    torch.save(torch.ones(1, 2, 2, 2), sample_path)
    pd.DataFrame(
        [
            {
                "dataset": "PT001_A",
                "participant_id": "sub-01",
                "session_id": "ses-01",
                "age": 42,
                "sex": "M",
                "group": "Control",
            }
        ]
    ).to_csv(tmp_path / "participants.tsv", sep="\t", index=False)

    first = PretrainDataset([str(sample_path)], metadata_paths=[str(tmp_path / "participants.tsv")])
    second = PretrainDataset([str(sample_path)], metadata_paths=[str(tmp_path / "participants.tsv")])

    assert first.metadata_dict is second.metadata_dict


def test_pretrain_dataset_rejects_legacy_participant_only_metadata(tmp_path):
    sample_path = tmp_path / "PT001_A" / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.pt"
    sample_path.parent.mkdir(parents=True)
    torch.save(torch.ones(1, 4, 4, 4), sample_path)

    participants = pd.DataFrame(
        [
            {
                "participant_id": "sub-01",
                "age": 42,
                "sex": "M",
                "group": "Control",
            }
        ]
    )
    participants.to_csv(tmp_path / "participants.tsv", sep="\t", index=False)

    with pytest.raises(KeyError):
        PretrainDataset([str(sample_path)], metadata_paths=[str(tmp_path / "participants.tsv")])


def test_pretrain_dataset_marks_registered_subset_from_50k_mapping(tmp_path):
    file_registered = tmp_path / "PT028_OASIS1" / "sub-001" / "ses-01" / "anat" / "sub-001_ses-01_run-4_T1w.pt"
    file_unregistered = tmp_path / "PT028_OASIS1" / "sub-002" / "ses-01" / "anat" / "sub-002_ses-01_run-1_T1w.pt"
    file_registered.parent.mkdir(parents=True)
    file_unregistered.parent.mkdir(parents=True)
    torch.save(torch.ones(1, 4, 4, 4), file_registered)
    torch.save(torch.ones(1, 4, 4, 4), file_unregistered)

    participants = pd.DataFrame(
        [
            {
                "dataset": "PT028_OASIS1",
                "participant_id": "sub-001",
                "session_id": "ses-01",
                "age": 70,
                "sex": "F",
                "group": "Control",
            },
            {
                "dataset": "PT028_OASIS1",
                "participant_id": "sub-002",
                "session_id": "ses-01",
                "age": 71,
                "sex": "M",
                "group": "Control",
            },
        ]
    )
    mapping = pd.DataFrame(
        [
            {
                "dataset_60K": "PT001_OASIS1",
                "subject_60K": "sub_1",
                "session_60K": "ses_1",
                "filename_60K": "t1_2.nii.gz",
                "dataset_300K": "PT028_OASIS1",
                "subject_300K": "ses-01",
                "session_300K": "anat",
                "filename_300K": "sub-001_ses-01_run-4_T1w.nii.gz",
            }
        ]
    )
    participants.to_csv(tmp_path / "participants.tsv", sep="\t", index=False)
    mapping.to_csv(tmp_path / "FOMO50K_300K_mapping.tsv", sep="\t", index=False)

    dataset = PretrainDataset(
        [str(file_registered), str(file_unregistered)],
        metadata_paths=[str(tmp_path / "participants.tsv")],
        registered_mapping_path=str(tmp_path / "FOMO50K_300K_mapping.tsv"),
    )
    assert dataset[0]["is_registered_subset"] is True
    assert dataset[1]["is_registered_subset"] is False

    registered_only = PretrainDataset(
        [str(file_registered), str(file_unregistered)],
        metadata_paths=[str(tmp_path / "participants.tsv")],
        registered_mapping_path=str(tmp_path / "FOMO50K_300K_mapping.tsv"),
        registered_only=True,
    )
    assert len(registered_only) == 1
    assert registered_only[0]["participant_id"] == "sub-001"


def test_pretrain_dataset_accepts_generic_fomo50k_mapping_manifest(tmp_path):
    fomo50k_root = tmp_path / "FOMO50K"
    file_registered = fomo50k_root / "PT028_OASIS1" / "sub-001" / "ses-01" / "anat" / "sub-001_ses-01_T1w.pt"
    file_unregistered = fomo50k_root / "PT028_OASIS1" / "sub-002" / "ses-01" / "anat" / "sub-002_ses-01_T1w.pt"
    file_registered.parent.mkdir(parents=True)
    file_unregistered.parent.mkdir(parents=True)
    torch.save(torch.ones(1, 4, 4, 4), file_registered)
    torch.save(torch.ones(1, 4, 4, 4), file_unregistered)

    participants = pd.DataFrame(
        [
            {
                "dataset": "PT028_OASIS1",
                "participant_id": "sub-001",
                "session_id": "ses-01",
                "age": 70,
                "sex": "F",
                "group": "Control",
            },
            {
                "dataset": "PT028_OASIS1",
                "participant_id": "sub-002",
                "session_id": "ses-01",
                "age": 71,
                "sex": "M",
                "group": "Control",
            },
        ]
    )
    mapping = pd.DataFrame(
        [
            {
                "dataset": "PT028_OASIS1",
                "old_path": "source/path",
                "new_path": "sub-001/ses-01/anat/sub-001_ses-01_T1w.pt",
                "old_filename": "source.nii.gz",
                "new_filename": "sub-001_ses-01_T1w.pt",
                "participant_id": "sub-001",
                "session_id": "ses-01",
                "modality": "t1w",
            }
        ]
    )
    participants.to_csv(tmp_path / "participants.tsv", sep="\t", index=False)
    mapping_path = fomo50k_root / "mapping.tsv"
    mapping.to_csv(mapping_path, sep="\t", index=False)

    dataset = PretrainDataset(
        [str(file_registered), str(file_unregistered)],
        metadata_paths=[str(tmp_path / "participants.tsv")],
        registered_mapping_path=str(mapping_path),
    )

    assert dataset[0]["is_registered_subset"] is True
    assert dataset[1]["is_registered_subset"] is False


def test_pretrain_dataset_rejects_generic_fomo300k_mapping_as_registered_manifest(tmp_path):
    fomo300k_root = tmp_path / "FOMO300K"
    file_registered = fomo300k_root / "PT028_OASIS1" / "sub-001" / "ses-01" / "anat" / "sub-001_ses-01_T1w.pt"
    file_registered.parent.mkdir(parents=True)
    torch.save(torch.ones(1, 4, 4, 4), file_registered)

    participants = pd.DataFrame(
        [
            {
                "dataset": "PT028_OASIS1",
                "participant_id": "sub-001",
                "session_id": "ses-01",
                "age": 70,
                "sex": "F",
                "group": "Control",
            }
        ]
    )
    mapping = pd.DataFrame(
        [
            {
                "dataset": "PT028_OASIS1",
                "old_path": "source/path",
                "new_path": "sub-001/ses-01/anat/sub-001_ses-01_T1w.pt",
                "old_filename": "source.nii.gz",
                "new_filename": "sub-001_ses-01_T1w.pt",
                "participant_id": "sub-001",
                "session_id": "ses-01",
                "modality": "t1w",
            }
        ]
    )
    participants.to_csv(tmp_path / "participants.tsv", sep="\t", index=False)
    mapping_path = fomo300k_root / "mapping.tsv"
    mapping.to_csv(mapping_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="dataset_300K and filename_300K"):
        PretrainDataset(
            [str(file_registered)],
            metadata_paths=[str(tmp_path / "participants.tsv")],
            registered_mapping_path=str(mapping_path),
        )


def test_pretrain_dataset_uses_composite_metadata_keys(tmp_path):
    file_a = tmp_path / "PT001_A" / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.pt"
    file_b = tmp_path / "PT002_B" / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.pt"
    file_a.parent.mkdir(parents=True)
    file_b.parent.mkdir(parents=True)
    torch.save(torch.ones(1, 4, 4, 4), file_a)
    torch.save(torch.ones(1, 4, 4, 4), file_b)

    participants = pd.DataFrame(
        [
            {
                "dataset": "PT001_A",
                "participant_id": "sub-01",
                "session_id": "ses-01",
                "age": 21,
                "sex": "M",
                "group": "Control",
            },
            {
                "dataset": "PT002_B",
                "participant_id": "sub-01",
                "session_id": "ses-01",
                "age": 72,
                "sex": "F",
                "group": "Dementia",
            },
        ]
    )
    participants.to_csv(tmp_path / "participants.tsv", sep="\t", index=False)

    dataset = PretrainDataset(
        [str(file_a), str(file_b)],
        metadata_paths=[str(tmp_path / "participants.tsv")],
    )

    item_a = dataset[0]
    item_b = dataset[1]
    assert item_a["age"] == 21.0
    assert item_a["sex"] == 0
    assert item_a["pathology"] == 0
    assert item_b["age"] == 72.0
    assert item_b["sex"] == 1
    assert item_b["pathology"] == 1


def test_pretrain_dataset_uses_session_specific_metadata(tmp_path):
    file_a = tmp_path / "PT001_A" / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.pt"
    file_b = tmp_path / "PT001_A" / "sub-01" / "ses-02" / "anat" / "sub-01_ses-02_T1w.pt"
    file_a.parent.mkdir(parents=True)
    file_b.parent.mkdir(parents=True)
    torch.save(torch.ones(1, 4, 4, 4), file_a)
    torch.save(torch.ones(1, 4, 4, 4), file_b)

    participants = pd.DataFrame(
        [
            {
                "dataset": "PT001_A",
                "participant_id": "sub-01",
                "session_id": "ses-01",
                "age": 40,
                "sex": "M",
                "group": "Control",
            },
            {
                "dataset": "PT001_A",
                "participant_id": "sub-01",
                "session_id": "ses-02",
                "age": 41,
                "sex": "M",
                "group": "Control",
            },
        ]
    )
    participants.to_csv(tmp_path / "participants.tsv", sep="\t", index=False)

    dataset = PretrainDataset(
        [str(file_a), str(file_b)],
        metadata_paths=[str(tmp_path / "participants.tsv")],
    )

    assert dataset[0]["age"] == 40.0
    assert dataset[1]["age"] == 41.0


def test_pretrain_dataset_uses_filename_level_scan_metadata(tmp_path):
    file_t1 = tmp_path / "PT001_A" / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.pt"
    file_flair = tmp_path / "PT001_A" / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_FLAIR.pt"
    file_t1.parent.mkdir(parents=True)
    torch.save(torch.ones(1, 4, 4, 4), file_t1)
    torch.save(torch.ones(1, 4, 4, 4), file_flair)

    participants = pd.DataFrame(
        [
            {
                "dataset": "PT001_A",
                "participant_id": "sub-01",
                "session_id": "ses-01",
                "age": 40,
                "sex": "M",
                "group": "Control",
            }
        ]
    )
    mri_info = pd.DataFrame(
        [
            {
                "dataset": "PT001_A",
                "participant_id": "sub-01",
                "session_id": "ses-01",
                "filename": "sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz",
                "Manufacturer": "Siemens",
            },
            {
                "dataset": "PT001_A",
                "participant_id": "sub-01",
                "session_id": "ses-01",
                "filename": "sub-01/ses-01/anat/sub-01_ses-01_FLAIR.nii.gz",
                "Manufacturer": "Philips",
            },
        ]
    )
    participants.to_csv(tmp_path / "participants.tsv", sep="\t", index=False)
    mri_info.to_csv(tmp_path / "mri_info.tsv", sep="\t", index=False)

    dataset = PretrainDataset(
        [str(file_t1), str(file_flair)],
        metadata_paths=[
            str(tmp_path / "participants.tsv"),
            str(tmp_path / "mri_info.tsv"),
        ],
    )

    assert dataset[0]["scanner_id"] == 0
    assert dataset[1]["scanner_id"] == 1
    assert dataset[0]["modality"] == "t1w"
    assert dataset[0]["modality_id"] == 0
    assert dataset[1]["modality"] == "flair"
    assert dataset[1]["modality_id"] == 2
    assert dataset[0]["subject_session_key"] == "pt001_a|sub-01|ses-01"


def test_pretrain_modality_taxonomy_covers_supported_structural_and_quantitative_modalities():
    assert PretrainDataset.MODALITY_TO_ID == {
        "t1w": 0,
        "t2w": 1,
        "flair": 2,
        "t1c": 3,
        "dwi": 4,
        "dwi_trace": 5,
        "adc": 6,
        "swi": 7,
        "gre": 8,
        "asl": 9,
        "m0scan": 10,
        "cbf": 11,
        "pdw": 12,
        "mp2rage": 13,
        "unit1": 14,
    }


def test_pretrain_dataset_fails_on_unclassified_modality(tmp_path):
    image_path = tmp_path / "PT001_A" / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_PET.pt"
    image_path.parent.mkdir(parents=True)
    torch.save(torch.ones(1, 4, 4, 4), image_path)
    pd.DataFrame(
        [
            {
                "dataset": "PT001_A",
                "participant_id": "sub-01",
                "session_id": "ses-01",
                "age": 40,
                "sex": "M",
                "group": "Control",
            }
        ]
    ).to_csv(tmp_path / "participants.tsv", sep="\t", index=False)

    with pytest.raises(ValueError, match="Unsupported or unclassified MRI modality"):
        PretrainDataset([str(image_path)], metadata_paths=[str(tmp_path / "participants.tsv")])


def test_pretrain_datamodule_builds_train_only_scanner_target_vocab(tmp_path):
    rows = []
    paths = []
    specs = [
        ("sub-01", "Siemens", "3T", "1.0 1.0 1.0"),
        ("sub-02", "Philips", "1.5", "1.6 1.6 1.6"),
        ("sub-03", "GE", "7T", "4.0 4.0 4.0"),
    ]
    for subject, manufacturer, field_strength, pixdim in specs:
        path = tmp_path / "PT001_A" / subject / "ses-01" / "anat" / f"{subject}_ses-01_T1w.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.ones(1, 2, 2, 2), path)
        paths.append(str(path))
        rows.append(
            {
                "dataset": "PT001_A",
                "participant_id": subject,
                "session_id": "ses-01",
                "filename": f"{subject}/ses-01/anat/{subject}_ses-01_T1w.pt",
                "age": 30,
                "sex": "M",
                "group": "control",
                "manufacturer": manufacturer,
                "MagneticFieldStrength": field_strength,
                "pixdim": pixdim,
            }
        )
    metadata_path = tmp_path / "metadata.tsv"
    pd.DataFrame(rows).to_csv(metadata_path, sep="\t", index=False)

    module = PretrainDataModule(
        batch_size=2,
        num_workers=0,
        train_split=paths[:2],
        val_split=paths[2:],
        metadata_paths=[str(metadata_path)],
        same_session_multimodal_batches=False,
        validate_split_disjointness=True,
        scanner_targets_enabled=True,
        save_scanner_target_vocab=False,
    )
    module.setup("fit")

    assert module.scanner_target_encoder.vocab["manufacturer"] == {"philips": 0, "siemens": 1}
    assert module.scanner_target_class_counts == {"manufacturer": 2, "field_strength": 2, "spacing_bin": 2}
    train_targets = [row["common"]["scanner_targets"] for row in module.train_dataset._cached_metadata]
    assert all(target["manufacturer"] != -100 for target in train_targets)
    assert all(target["field_strength"] != -100 for target in train_targets)
    assert all(target["spacing_bin"] != -100 for target in train_targets)
    assert module.val_dataset._cached_metadata[0]["common"]["scanner_targets"] == {
        "manufacturer": -100,
        "field_strength": -100,
        "spacing_bin": -100,
    }


def test_scanner_spacing_uses_only_pixdim_from_sibling_manifest(tmp_path):
    paths = []
    mri_rows = []
    manifest_rows = []
    specs = [
        ("sub-01", "Siemens", "3T", "1x1x1"),
        ("sub-02", "Philips", "1.5T", "1.6x1.6x1.6"),
        ("sub-03", "GE", "7T", "4x4x4"),
    ]
    for subject, manufacturer, field_strength, pixdim in specs:
        filename = f"{subject}/ses-01/anat/{subject}_ses-01_T1w.pt"
        path = tmp_path / "PT001_A" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.ones(1, 2, 2, 2), path)
        paths.append(str(path))
        identity = {
            "dataset": "PT001_A",
            "participant_id": subject,
            "session_id": "ses-01",
            "filename": filename,
        }
        mri_rows.append(
            {
                **identity,
                "Manufacturer": manufacturer,
                "MagneticFieldStrength": field_strength,
            }
        )
        manifest_rows.append({**identity, "pixdim": pixdim, "Manufacturer": "must-not-override"})
    mri_path = tmp_path / "mri_info.tsv"
    manifest_path = tmp_path / "manifest.tsv"
    pd.DataFrame(mri_rows).to_csv(mri_path, sep="\t", index=False)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, sep="\t", index=False)

    module = PretrainDataModule(
        batch_size=2,
        num_workers=0,
        train_split=paths[:2],
        val_split=paths[2:],
        metadata_paths=[str(mri_path)],
        same_session_multimodal_batches=False,
        validate_split_disjointness=True,
        scanner_targets_enabled=True,
        save_scanner_target_vocab=False,
    )
    module.setup("fit")

    assert module.scanner_spacing_metadata_path == str(manifest_path.resolve())
    assert module.scanner_target_encoder.vocab["manufacturer"] == {"philips": 0, "siemens": 1}
    assert module.scanner_target_encoder.vocab["spacing_bin"] == {"<=1.0": 0, "(1.5,2.0]": 1}
    assert {row["raw"]["manufacturer"] for row in module.train_dataset._cached_metadata} == {"Siemens", "Philips"}
    assert {row["raw"]["pixdim"] for row in module.train_dataset._cached_metadata} == {"1x1x1", "1.6x1.6x1.6"}


def test_pretrain_modality_config_rejects_unknown_names_and_numeric_ids():
    assert _resolve_pretrain_modality_ids(["t1w", "t2", 4]) == [0, 1, 4]
    assert PretrainDataset.normalize_modality_ids(["t2", "t2w", 2]) == (1, 2)
    assert PretrainDataset.modality_name(1) == "t2w"
    with pytest.raises(ValueError, match="Unsupported pretraining modality"):
        _resolve_pretrain_modality_ids(["pet"])
    with pytest.raises(ValueError, match="Unsupported pretraining modality"):
        _resolve_pretrain_modality_ids([99])


def test_pretrain_collate_keeps_raw_metadata_as_list():
    from asparagus.modules.data_modules.pretraining import pretrain_collate

    batch = [
        {
            "image": torch.ones(1, 4, 4, 4),
            "label": torch.ones(1, 4, 4, 4),
            "age": 42.0,
            "metadata": {"age": 42.0, "sex": "M", "manufacturer": float("nan")},
        },
        {
            "image": torch.zeros(1, 4, 4, 4),
            "label": torch.zeros(1, 4, 4, 4),
            "age": float("nan"),
            "metadata": {
                "age": "Unknown",
                "sex": float("nan"),
                "manufacturer": "Siemens",
            },
        },
    ]

    collated = pretrain_collate(batch)

    assert collated["image"].shape == (2, 1, 4, 4, 4)
    assert isinstance(collated["metadata"], list)
    assert collated["metadata"][0]["sex"] == "M"
    assert collated["metadata"][1]["manufacturer"] == "Siemens"


def test_pretrain_datamodule_rejects_subject_leakage(tmp_path):
    train_path = tmp_path / "PT001_A" / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.pt"
    val_path = tmp_path / "PT001_A" / "sub-01" / "ses-02" / "anat" / "sub-01_ses-02_T1w.pt"
    participants = pd.DataFrame(
        [
            {
                "dataset": "PT001_A",
                "participant_id": "sub-01",
                "session_id": "ses-01",
                "age": 40,
                "sex": "F",
                "group": "Control",
            },
            {
                "dataset": "PT001_A",
                "participant_id": "sub-01",
                "session_id": "ses-02",
                "age": 41,
                "sex": "F",
                "group": "Control",
            },
        ]
    )
    participants.to_csv(tmp_path / "participants.tsv", sep="\t", index=False)
    module = PretrainDataModule(
        batch_size=1,
        num_workers=0,
        train_split=[str(train_path)],
        val_split=[str(val_path)],
        metadata_paths=[str(tmp_path / "participants.tsv")],
    )

    with pytest.raises(ValueError, match="split leakage"):
        module.setup_fit()


def test_pretrain_validation_loader_is_deterministic_and_complete(tmp_path):
    paths = []
    rows = []
    for subject in ("sub-01", "sub-02", "sub-03"):
        path = tmp_path / "PT001_A" / subject / "ses-01" / "anat" / f"{subject}_ses-01_T1w.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.ones(1, 2, 2, 2), path)
        paths.append(str(path))
        rows.append(
            {
                "dataset": "PT001_A",
                "participant_id": subject,
                "session_id": "ses-01",
                "age": 40,
                "sex": "F",
                "group": "Control",
            }
        )
    pd.DataFrame(rows).to_csv(tmp_path / "participants.tsv", sep="\t", index=False)
    module = PretrainDataModule(
        batch_size=2,
        num_workers=0,
        train_split=[paths[0]],
        val_split=paths[1:],
        train_transforms=None,
        val_transforms=None,
        metadata_paths=[str(tmp_path / "participants.tsv")],
    )
    module.setup_fit()

    first = [batch["file_path"] for batch in module.val_dataloader()]
    second = [batch["file_path"] for batch in module.val_dataloader()]
    assert first == second
    assert sum(len(batch) for batch in first) == 2


def test_pretrain_validation_transform_uses_deterministic_center_crop():
    from asparagus.modules.transforms.presets.pretrain import CPU_val_transforms

    image = torch.arange(8 * 8 * 8, dtype=torch.float32).reshape(1, 8, 8, 8)
    transform = CPU_val_transforms([4, 4, 4])
    first = transform({"image": image.clone()})["image"]
    second = transform({"image": image.clone()})["image"]

    assert torch.equal(first, second)
    assert first.shape == (1, 4, 4, 4)


def test_pretrain_cpu_transforms_accept_hydra_listconfig_patch_size():
    from asparagus.modules.transforms.presets.pretrain import CPU_train_transforms, CPU_val_transforms

    cfg = OmegaConf.create({"patch_size": [8, 8, 8]})
    image = torch.arange(10 * 10 * 10, dtype=torch.float32).reshape(1, 10, 10, 10)

    val_output = CPU_val_transforms(cfg.patch_size)({"image": image.clone()})
    train_output = CPU_train_transforms(cfg.patch_size)({"image": image.clone()})

    assert val_output["image"].shape == (1, 8, 8, 8)
    assert train_output["image"].shape == (1, 8, 8, 8)


def test_pretrain_datamodule_supports_validation_only_setup(tmp_path):
    train_path = tmp_path / "PT001_A" / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.pt"
    val_path = tmp_path / "PT001_A" / "sub-02" / "ses-01" / "anat" / "sub-02_ses-01_T1w.pt"
    for path in (train_path, val_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.ones(1, 2, 2, 2), path)
    pd.DataFrame(
        [
            {
                "dataset": "PT001_A",
                "participant_id": "sub-01",
                "session_id": "ses-01",
                "group": "Control",
            },
            {
                "dataset": "PT001_A",
                "participant_id": "sub-02",
                "session_id": "ses-01",
                "group": "Control",
            },
        ]
    ).to_csv(tmp_path / "participants.tsv", sep="\t", index=False)
    module = PretrainDataModule(
        batch_size=1,
        num_workers=0,
        train_split=[str(train_path)],
        val_split=[str(val_path)],
        metadata_paths=[str(tmp_path / "participants.tsv")],
        complete_validation=True,
    )

    module.setup("validate")

    assert len(module.val_dataloader().dataset) == 1


def test_pretrain_monitor_cohorts_are_fixed_canonical_and_seed_independent(tmp_path):
    train_paths = []
    val_paths = []
    rows = []
    for split, subjects in (("train", range(6)), ("val", range(6, 12))):
        for number in subjects:
            subject = f"sub-{number:02d}"
            for modality in ("T1w", "FLAIR"):
                path = tmp_path / "PT001_A" / subject / "ses-01" / "anat" / f"{subject}_ses-01_{modality}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(torch.ones(1, 2, 2, 2), path)
                (train_paths if split == "train" else val_paths).append(str(path))
            rows.append(
                {
                    "dataset": "PT001_A",
                    "participant_id": subject,
                    "session_id": "ses-01",
                    "age": 30 + number,
                    "sex": "F" if number % 2 else "M",
                    "group": "Control" if number % 2 else "AD",
                    "manufacturer": "Siemens" if number % 2 else "Philips",
                }
            )
    metadata_path = tmp_path / "participants.tsv"
    pd.DataFrame(rows).to_csv(metadata_path, sep="\t", index=False)

    def configured_module(training_seed, demographic_modalities=None):
        module = PretrainDataModule(
            batch_size=2,
            num_workers=0,
            train_split=train_paths,
            val_split=val_paths,
            metadata_paths=[str(metadata_path)],
            sampler_seed=training_seed,
            monitor_seed=17,
            routine_scan_count=4,
            probe_train_max_subjects=3,
            demographic_probe_train_max_subjects=2,
            demographic_modalities=demographic_modalities,
            stage1_pair_session_count=2,
        )
        module.setup_fit()
        return module

    first = configured_module(1)
    second = configured_module(999)
    first_routine = [first.routine_val_dataset.dataset.files[index] for index in first.routine_val_dataset.indices]
    second_routine = [second.routine_val_dataset.dataset.files[index] for index in second.routine_val_dataset.indices]
    assert first_routine == second_routine
    assert len(first_routine) == 4

    query_rows = first.probe_query_dataset.dataset._cached_metadata
    assert len(query_rows) == 6
    assert {row["identity"]["modality_id"] for row in query_rows} == {0}
    assert len(first.probe_reference_dataset) == 3
    modality_reference_rows = first.modality_probe_reference_dataset.dataset._cached_metadata
    assert {row["identity"]["modality_id"] for row in modality_reference_rows} == {0, 2}
    demographic_rows = first.demographic_probe_reference_dataset.dataset._cached_metadata
    assert len(demographic_rows) == 2
    assert all(row["identity"]["modality_id"] == 0 and row["common"]["pathology"] == 0 for row in demographic_rows)
    assert len(first.stage1_monitor_dataset) == 4

    reference_subjects = {row["identity"]["subject_key"] for row in first.probe_reference_dataset.dataset._cached_metadata}
    query_subjects = {row["identity"]["subject_key"] for row in query_rows}
    assert reference_subjects.isdisjoint(query_subjects)

    flair_only = configured_module(1, demographic_modalities=[2])
    flair_query_rows = flair_only.probe_query_dataset.dataset._cached_metadata
    assert {row["identity"]["modality_id"] for row in flair_query_rows} == {0, 2}
    flair_demographic_rows = flair_only.demographic_probe_reference_dataset.dataset._cached_metadata
    assert flair_demographic_rows
    assert all(row["identity"]["modality_id"] == 2 for row in flair_demographic_rows)


def test_demographic_objective_requires_eligible_samples_in_both_splits(tmp_path):
    train_path = tmp_path / "PT001_A" / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.pt"
    val_path = tmp_path / "PT001_A" / "sub-02" / "ses-01" / "anat" / "sub-02_ses-01_T1w.pt"
    for path in (train_path, val_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.ones(1, 2, 2, 2), path)
    pd.DataFrame(
        [
            {
                "dataset": "PT001_A",
                "participant_id": "sub-01",
                "session_id": "ses-01",
                "age": 20,
                "sex": "M",
                "group": "Control",
            },
            {
                "dataset": "PT001_A",
                "participant_id": "sub-02",
                "session_id": "ses-01",
                "age": 50,
                "sex": "F",
                "group": "AD",
            },
        ]
    ).to_csv(tmp_path / "participants.tsv", sep="\t", index=False)
    module = PretrainDataModule(
        batch_size=1,
        num_workers=0,
        train_split=[str(train_path)],
        val_split=[str(val_path)],
        metadata_paths=[str(tmp_path / "participants.tsv")],
        require_demographic_samples=True,
    )

    with pytest.raises(ValueError, match="Demographic objective requires eligible"):
        module.setup_fit()


def test_multimodal_objectives_require_objective_aware_sampling():
    with pytest.raises(ValueError, match="same_session_multimodal_batches=true"):
        PretrainDataModule(
            batch_size=2,
            num_workers=0,
            train_split=[],
            val_split=[],
            metadata_paths=["unused.tsv"],
            same_session_multimodal_batches=False,
            multimodal_batch_probability=0.0,
            require_multimodal_samples=True,
        )


def test_stage1_objective_requires_multimodal_samples_in_train_and_val(tmp_path):
    train_paths = []
    val_paths = []
    rows = []
    for subject, split, modalities in (
        ("sub-01", "train", ("T1w", "FLAIR")),
        ("sub-02", "val", ("T1w",)),
    ):
        rows.append(
            {
                "dataset": "PT001_A",
                "participant_id": subject,
                "session_id": "ses-01",
                "age": 30,
                "sex": "F",
                "group": "Control",
            }
        )
        for modality in modalities:
            path = tmp_path / "PT001_A" / subject / "ses-01" / "anat" / f"{subject}_ses-01_{modality}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.ones(1, 2, 2, 2), path)
            (train_paths if split == "train" else val_paths).append(str(path))
    metadata_path = tmp_path / "participants.tsv"
    pd.DataFrame(rows).to_csv(metadata_path, sep="\t", index=False)

    module = PretrainDataModule(
        batch_size=2,
        num_workers=0,
        train_split=train_paths,
        val_split=val_paths,
        metadata_paths=[str(metadata_path)],
        same_session_multimodal_batches=True,
        multimodal_batch_probability=1.0,
        require_multimodal_samples=True,
    )

    with pytest.raises(ValueError, match="Multimodal Stage 1 requires"):
        module.setup_fit()


def test_stage2_objective_requires_registered_multimodal_samples_in_train_and_val(tmp_path):
    train_paths = []
    val_paths = []
    rows = []
    mapping_rows = []
    for subject, split in (("sub-01", "train"), ("sub-02", "val")):
        rows.append(
            {
                "dataset": "PT001_A",
                "participant_id": subject,
                "session_id": "ses-01",
                "age": 30,
                "sex": "F",
                "group": "Control",
            }
        )
        for modality in ("T1w", "FLAIR"):
            filename = f"{subject}_ses-01_{modality}.pt"
            path = tmp_path / "PT001_A" / subject / "ses-01" / "anat" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.ones(1, 2, 2, 2), path)
            (train_paths if split == "train" else val_paths).append(str(path))
            if modality == "T1w":
                mapping_rows.append({"dataset_300K": "PT001_A", "filename_300K": filename})
    metadata_path = tmp_path / "participants.tsv"
    mapping_path = tmp_path / "mapping.tsv"
    pd.DataFrame(rows).to_csv(metadata_path, sep="\t", index=False)
    pd.DataFrame(mapping_rows).to_csv(mapping_path, sep="\t", index=False)

    module = PretrainDataModule(
        batch_size=2,
        num_workers=0,
        train_split=train_paths,
        val_split=val_paths,
        metadata_paths=[str(metadata_path)],
        registered_mapping_path=str(mapping_path),
        same_session_multimodal_batches=True,
        multimodal_batch_probability=1.0,
        require_registered_multimodal_samples=True,
    )

    with pytest.raises(ValueError, match="Multimodal Stage 2 requires registered"):
        module.setup_fit()


# The two diagnostic-split-generator tests drove `asparagus/scripts/make_pretrain_diagnostic_split.py`,
# a corpus-curation helper outside the published D1 rail. Script and tests were removed together.
# The Jean-Zay CUDA-preflight test went with `misc/jean_zay/jz_pretrain.slurm` for the same reason.


def test_3d_dwi_bval_file_processes_without_bval_bvec_sidecars(tmp_path):
    from asparagus_preprocessing.configs.preprocessing_presets import (
        get_FOMO300K_saving_config,
        get_noresampling_preprocessing_config,
    )
    from asparagus_preprocessing.utils.process_case import process_dwi_case

    image_path = tmp_path / "sub-001_ses-01_dwi_bval1200.nii.gz"
    output_base = tmp_path / "processed" / "sub-001_ses-01_dwi_bval1200"
    data = np.random.default_rng(0).normal(size=(16, 16, 16)).astype(np.float32)
    nib.save(nib.Nifti1Image(data, affine=np.eye(4)), image_path)

    process_dwi_case(
        str(image_path),
        str(tmp_path / "missing.bval"),
        str(tmp_path / "missing.bvec"),
        str(output_base),
        get_noresampling_preprocessing_config(),
        get_FOMO300K_saving_config(save_as_tensor=True, save_dset_metadata=False, bidsify=False),
        use_trace_computation=True,
        strict=False,
    )

    assert (tmp_path / "processed" / "sub-001_ses-01_dwi_bval1200.pt").exists()
    assert not (tmp_path / "processed" / "sub-001_ses-01_dwi_bval1200_bval_.pt").exists()


def test_update_paths_skips_empty_subsample_dataset_without_warning(tmp_path, caplog):
    from asparagus_preprocessing.utils.detect import update_paths
    from asparagus_preprocessing.utils.saving import enhanced_save_json

    dataset_dir = tmp_path / "PT999_EMPTY"
    dataset_dir.mkdir()
    enhanced_save_json(
        {
            "dataset_config": {"split": None},
            "metadata": {"files_source_directory_total": 0},
            "saving_config": {"save_as_tensor": True},
        },
        str(dataset_dir / "dataset.json"),
    )
    enhanced_save_json([], str(dataset_dir / "paths.json"))

    caplog.clear()
    with caplog.at_level("WARNING"):
        update_paths(str(dataset_dir))

    assert "No files found in target directory" not in caplog.text


def test_demographic_views_use_raw_image_and_dedicated_cpu_transform():
    class AddTen:
        def __call__(self, batch):
            batch = dict(batch)
            batch["image"] = batch["image"] + 10.0
            return batch

    module = SelfSupervisedModule(
        model=torch.nn.Identity(),
        learning_rate=1e-3,
        demo_cpu_transforms=AddTen(),
    )
    batch = {
        "image": torch.zeros(1, 1, 2, 2, 2),
        "raw_image": torch.ones(1, 1, 2, 2, 2),
        "age": torch.tensor([30.0]),
        "sex": torch.tensor([1]),
        "pathology": torch.tensor([0]),
        "modality_id": torch.tensor([0]),
    }

    prepared = module._prepare_contrastive_batch(batch, is_training=True)

    assert torch.equal(prepared["view_masked"]["image"], batch["image"])
    assert torch.equal(prepared["view_aug_1"]["image"], batch["raw_image"] + 10.0)
    assert torch.equal(prepared["view_aug_2"]["image"], batch["raw_image"] + 10.0)


def test_demographic_cpu_transform_handles_batched_raw_images():
    class AssertSingleImage:
        def __call__(self, batch):
            batch = dict(batch)
            assert batch["image"].ndim == 4
            batch["image"] = batch["image"] + 1.0
            return batch

    module = SelfSupervisedModule(
        model=torch.nn.Identity(),
        learning_rate=1e-3,
        demo_cpu_transforms=AssertSingleImage(),
    )
    batch = {
        "image": torch.zeros(2, 1, 2, 2, 2),
        "raw_image": torch.ones(2, 1, 2, 2, 2),
        "age": torch.tensor([30.0, 40.0]),
        "sex": torch.tensor([1, 0]),
        "pathology": torch.tensor([0, 0]),
        "modality_id": torch.tensor([0, 0]),
    }

    prepared = module._prepare_contrastive_batch(batch, is_training=True)

    assert torch.equal(prepared["view_masked"]["image"], batch["image"])
    assert torch.equal(prepared["view_aug_1"]["image"], batch["raw_image"] + 1.0)
    assert torch.equal(prepared["view_aug_2"]["image"], batch["raw_image"] + 1.0)


def test_dataset_id_metadata_metrics_use_topk_summary_not_per_hash_keys():
    from asparagus.modules.lightning_modules.ssl import metadata as ssl_metadata

    metrics = ssl_metadata.contrastive_metadata_metrics(
        {
            "dataset_id": torch.tensor([10, 10, 10, 20, 20, 30]),
            "pathology": torch.tensor([0, 0, 1, 1, 1, -1]),
        }
    )

    # High-cardinality dataset_id is summarized, not expanded one key per hash.
    assert not any(key.startswith("dataset_id/class_") for key in metrics)
    assert metrics["dataset_id/unique_count"] == 3
    assert metrics["dataset_id/top1_fraction"] == pytest.approx(0.5)
    assert metrics["dataset_id/top1_id"] == pytest.approx(10.0)
    # Bounded enums still expand per class.
    assert "pathology/class_0_fraction" in metrics


def test_validation_epoch_metadata_metrics_gather_before_rank_stable_logging():
    from asparagus.modules.lightning_modules.ssl import metadata as ssl_metadata

    local = {
        "age": torch.tensor([20.0, 21.0]),
        "sex": torch.tensor([0, 0]),
        "pathology": torch.tensor([0, 0]),
        "fine_pathology": torch.tensor([0, 0]),
        "scanner_id": torch.tensor([0, 0]),
        "modality_id": torch.tensor([0, 0]),
        "dataset_id": torch.tensor([10, 10]),
    }
    remote = {
        "age": torch.tensor([70.0, 71.0]),
        "sex": torch.tensor([1, 1]),
        "pathology": torch.tensor([1, 1]),
        "fine_pathology": torch.tensor([1, 1]),
        "scanner_id": torch.tensor([1, 1]),
        "modality_id": torch.tensor([1, 1]),
        "dataset_id": torch.tensor([20, 20]),
    }
    # This is the job-382957 failure mode: rank-local class support yields
    # different dynamic metric keys and cannot be passed to sync_dist=True.
    assert set(ssl_metadata.contrastive_metadata_metrics(local)) != set(ssl_metadata.contrastive_metadata_metrics(remote))

    module = SelfSupervisedModule(model=torch.nn.Identity(), learning_rate=1e-3)
    module._val_metadata_batches = [local]
    remote_values = iter(remote.values())
    gather_calls = []

    def gather(local_value):
        gather_calls.append(local_value.clone())
        return torch.cat((local_value, next(remote_values).to(local_value.device)))

    logged = {}
    log_kwargs = {}
    module._gather_monitor_tensor = gather
    module.log_dict = lambda metrics, **kwargs: (logged.update(metrics), log_kwargs.update(kwargs))

    module.on_validation_epoch_end()

    assert len(gather_calls) == len(local)
    assert logged["val/metadata_epoch/pathology/class_0_fraction"] == pytest.approx(0.5)
    assert logged["val/metadata_epoch/pathology/class_1_fraction"] == pytest.approx(0.5)
    assert logged["val/metadata_epoch/dataset_id/unique_count"] == 2
    assert log_kwargs["sync_dist"] is False


def test_pathology_eligibility_applies_modality_and_class_filters():
    module = SelfSupervisedModule(
        model=torch.nn.Identity(),
        learning_rate=1e-3,
        pathology_modalities=[0],
        pathology_eligible_classes=[0, 1, 3, 4],
        pathology_exclude_classes=[2, 5],
    )
    metadata = {
        "pathology": torch.tensor([0, 1, 2, 3, 4, 5, -1, 1]),
        "modality_id": torch.tensor([0, 0, 0, 1, 0, 0, 0, 2]),
    }

    assert module._pathology_eligible_mask(metadata).tolist() == [
        True,
        True,
        False,
        False,
        True,
        False,
        False,
        False,
    ]


def test_pathology_eligibility_can_count_unknowns_as_skipped_rows():
    module = SelfSupervisedModule(
        model=torch.nn.Identity(),
        learning_rate=1e-3,
        pathology_unknown_ignore=False,
    )
    metadata = {
        "pathology": torch.tensor([0, -1, 1]),
        "modality_id": torch.zeros(3, dtype=torch.long),
    }

    assert module._pathology_eligible_mask(metadata).tolist() == [True, True, True]


def test_contrastive_metadata_gathers_fine_pathology():
    module = SelfSupervisedModule(model=torch.nn.Identity(), learning_rate=1e-3)
    batch = {
        "age": torch.tensor([20.0, 70.0]),
        "sex": torch.tensor([0, 1]),
        "pathology": torch.tensor([0, 1]),
        "fine_pathology": torch.tensor([101, 202]),
        "modality_id": torch.tensor([0, 0]),
        "scanner_id": torch.tensor([0, 1]),
        "dataset_id": torch.tensor([10, 20]),
    }

    local, global_metadata = module._gather_contrastive_metadata(batch, torch.device("cpu"), torch.float32)

    assert local["fine_pathology"].tolist() == [101, 202]
    assert global_metadata["fine_pathology"].tolist() == [101, 202]


def test_contrastive_views_preserve_fine_pathology_and_dataset_id_top_level():
    batch = {
        "image": torch.randn(2, 1, 4, 4, 4),
        "age": torch.tensor([20.0, 70.0]),
        "sex": torch.tensor([0, 1]),
        "pathology": torch.tensor([0, 1]),
        "fine_pathology": torch.tensor([101, 202]),
        "modality_id": torch.tensor([0, 0]),
        "scanner_id": torch.tensor([0, 1]),
        "dataset_id": torch.tensor([10, 20]),
    }

    prepared = ssl_views.build_contrastive_views(
        batch,
        is_training=False,
        train_transforms=None,
        unmasked_transforms=None,
        momentum_transforms=None,
        val_transforms=None,
        demo_cpu_transforms=None,
        validation_mask_seed=0,
    )

    assert prepared["fine_pathology"].tolist() == [101, 202]
    assert prepared["dataset_id"].tolist() == [10, 20]
    assert prepared["metadata"]["fine_pathology"].tolist() == [101, 202]
    assert prepared["metadata"]["dataset_id"].tolist() == [10, 20]


def test_frepa_lite_contrastive_views_apply_only_to_training_masked_view():
    class IdentityTransform:
        def __call__(self, data):
            return data

    batch = {
        "image": torch.randn(2, 1, 8, 8, 8),
        "raw_image": torch.randn(2, 1, 8, 8, 8),
        "age": torch.tensor([20.0, 70.0]),
        "sex": torch.tensor([0, 1]),
        "pathology": torch.tensor([0, 1]),
        "modality_id": torch.tensor([0, 0]),
    }
    frepa = Torch_FrepaLiteFrequencyCorruption(
        enabled=True,
        p=1.0,
        ndim=3,
        low_scale_min=1.2,
        low_scale_max=1.2,
        low_noise_std=0.0,
        high_mask_ratio=0.0,
    )

    train_views = ssl_views.build_contrastive_views(
        batch,
        is_training=True,
        train_transforms=frepa,
        unmasked_transforms=IdentityTransform(),
        momentum_transforms=IdentityTransform(),
        val_transforms=None,
        demo_cpu_transforms=None,
        validation_mask_seed=0,
    )
    val_views = ssl_views.build_contrastive_views(
        batch,
        is_training=False,
        train_transforms=frepa,
        unmasked_transforms=IdentityTransform(),
        momentum_transforms=IdentityTransform(),
        val_transforms=None,
        demo_cpu_transforms=None,
        validation_mask_seed=0,
    )

    assert "frepa_lite/applied_fraction" in train_views["view_masked"]
    assert "frepa_lite/applied_fraction" not in train_views["view_aug_1"]
    assert "frepa_lite/applied_fraction" not in train_views["view_aug_2"]
    assert "frepa_lite/applied_fraction" not in val_views["view_masked"]
    assert not torch.equal(train_views["view_masked"]["image"], batch["image"])
    assert torch.equal(train_views["view_aug_1"]["image"], batch["image"])
    assert torch.equal(train_views["view_aug_2"]["image"], batch["image"])


def test_stop_after_validation_step_preserves_trainer_horizon():
    callback = StopAfterValidationStep(3150)
    trainer = SimpleNamespace(global_step=3149, max_steps=12000, should_stop=False)

    callback.on_validation_end(trainer, None)
    assert trainer.should_stop is False
    assert trainer.max_steps == 12000

    trainer.global_step = 3150
    callback.on_validation_end(trainer, None)
    assert trainer.should_stop is True
    assert trainer.max_steps == 12000


def test_wandb_uses_hydra_config_without_duplicate_lightning_hparams():
    logger = object.__new__(HydraConfigWandbLogger)

    assert logger.log_hyperparams({"loss_weight_demo": 0.5}) is None


def test_pretrain_trainer_receives_configured_distributed_strategy():
    with initialize_config_dir(version_base=None, config_dir=str(Path("configs").resolve())):
        cfg = compose(
            config_name="projects/fomo26/safety/pretrain/resenc_amaes",
            overrides=["hardware.strategy=ddp_find_unused_parameters_true"],
        )

    assert cfg.lightning._trainer.strategy == "ddp_find_unused_parameters_true"


def test_pretrain_progress_callback_follows_progress_bar_config():
    with initialize_config_dir(version_base=None, config_dir=str(Path("configs").resolve())):
        disabled = compose(
            config_name="projects/fomo26/safety/pretrain/resenc_amaes",
            overrides=["logger.progress_bar=false"],
        )
        enabled = compose(
            config_name="projects/fomo26/safety/pretrain/resenc_amaes",
            overrides=["logger.progress_bar=true"],
        )

    assert _pretrain_progress_callbacks(disabled) == []
    assert len(_pretrain_progress_callbacks(enabled)) == 1


def test_rank_zero_detection_uses_distributed_environment(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert pretrain_is_rank_zero_process()
    assert setup_is_rank_zero_process()

    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert not pretrain_is_rank_zero_process()
    assert not setup_is_rank_zero_process()


def test_detect_id_supports_custom_run_underscore_directory_layout(tmp_path):
    run_dir = tmp_path / "PT900_FOMO300K" / "run_320133"
    run_dir.mkdir(parents=True)

    assert detect_id("320133", model_dir=str(tmp_path)) == str(run_dir)


def test_detect_id_still_supports_default_run_id_directory_layout(tmp_path):
    run_dir = tmp_path / "some" / "nested" / "run_id=320133"
    run_dir.mkdir(parents=True)

    assert detect_id("320133", model_dir=str(tmp_path)) == str(run_dir)


def test_resolve_training_resume_checkpoint_requires_same_run_last_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    assert resolve_training_resume_checkpoint(str(checkpoint_dir), required=False) is None
    with pytest.raises(FileNotFoundError, match="resume_training=true"):
        resolve_training_resume_checkpoint(str(checkpoint_dir), required=True)

    checkpoint_path = checkpoint_dir / "last.ckpt"
    checkpoint_path.write_bytes(b"checkpoint")
    assert resolve_training_resume_checkpoint(str(checkpoint_dir), required=True) == str(checkpoint_path)


def test_resolve_training_resume_seed_reads_saved_hparams_without_loading_weights(
    tmp_path,
):
    (tmp_path / "hparams.yaml").write_text(
        "validation_mask_seed: 175422\nrecursive: &node\n  self: *node\n",
        encoding="utf-8",
    )

    assert resolve_training_resume_seed(str(tmp_path)) == 175422


def test_strict_lightning_resume_rejects_incomplete_state_dict():
    module = SelfSupervisedModule(model=torch.nn.Linear(2, 2), learning_rate=1e-3)
    state_dict = module.state_dict()
    incomplete_state_dict = {key: value for key, value in state_dict.items() if not key.endswith("bias")}

    with pytest.raises(RuntimeError, match="Missing key"):
        module.load_state_dict(incomplete_state_dict, strict=True)


def test_permissive_self_supervised_transfer_returns_incompatible_keys_report():
    module = SelfSupervisedModule(model=torch.nn.Linear(2, 2), learning_rate=1e-3)
    state_dict = module.state_dict()
    transferred = {key: value.clone() + 1 for key, value in state_dict.items() if not key.endswith("bias")}

    report = module.load_state_dict(transferred, strict=False)

    assert report is not None
    assert sorted(report.missing_keys) == sorted(set(state_dict) - set(transferred))
    assert report.unexpected_keys == []


def test_profiler_callback_configuration_is_instantiable():
    cfg = OmegaConf.load("configs/core/base.yaml")

    callback = instantiate(cfg.profiler._callback)

    assert callback.profile_memory is True
    assert callback.warmup_steps == 10


def test_profiler_supports_current_device_time_event_api(monkeypatch):
    class Event:
        key = "forward"
        cpu_time_total = 1000.0
        device_time_total = 2000.0

    class FakeProfiler:
        @staticmethod
        def key_averages():
            return [Event()]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)

    metrics = ProfilerCallback()._extract_timing_metrics(FakeProfiler())

    assert metrics["forward_time"] == pytest.approx(0.002)
    assert metrics["step_time"] == pytest.approx(0.002)
    assert "gpu_memory_peak_allocated_gb" in metrics


def test_warmup_schedule_handles_single_pseudo_epoch_smoke_run():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1e-3)

    scheduler = simple_warmup_cosine_decay_schedule(
        optimizer,
        warmup_epochs=1,
        steps_per_epoch=6,
        cosine_period_ratio=1.0,
        max_steps=6,
    )
    for _ in range(6):
        optimizer.step()
        scheduler.step()

    assert math.isfinite(optimizer.param_groups[0]["lr"])


class _Stage1SigRegSmokeModel(torch.nn.Module):
    def forward_with_features(self, x, modality_id=None):
        return torch.zeros_like(x), {"h": x.flatten(1)}

    def forward_multimodal_ssl(self, x, modality_id=None, return_logits=True, grl_lambda=0.0):
        flat = x.flatten(1).float()
        base = torch.cat(
            [
                flat,
                flat.square() + 0.1,
                torch.sin(flat),
                torch.cos(flat),
            ],
            dim=1,
        )
        return {
            "h": base[:, :4],
            "z_demo": base[:, :4],
            "z_patho": base[:, :4],
            "z_anatomy": base[:, :4],
        }


def test_masked_reconstruction_and_detail_losses_use_hidden_voxels_only():
    module = SelfSupervisedModule(model=torch.nn.Identity(), learning_rate=1e-3)
    target = torch.zeros(1, 1, 2, 2, 2)
    mask = torch.tensor([[[[[True, True], [False, False]], [[True, True], [False, False]]]]])
    pred_visible_error = torch.zeros_like(target)
    pred_visible_error[mask] = 10.0
    pred_hidden_error = torch.zeros_like(target)
    pred_hidden_error[~mask] = 2.0

    assert module._rec_loss(pred_visible_error, target, mask).item() == 0.0
    assert module._rec_loss(pred_hidden_error, target, mask).item() == pytest.approx(4.0)
    assert frequency_domain_loss(pred_visible_error, target, mask=mask).item() == pytest.approx(0.0)
    assert spatial_detail_loss(pred_visible_error, target, mask=mask).item() == pytest.approx(0.0)


def test_hidden_reconstruction_metrics_ignore_visible_prediction_errors():
    target = torch.zeros(1, 1, 8, 8, 8)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[:, :, 2:6, 2:6, 2:6] = False
    pred = torch.zeros_like(target)
    pred[mask] = 10.0

    masked_input = torch.zeros_like(target)
    metrics = reconstruction_metrics.compute(pred, target, mask, masked_input=masked_input)

    assert metrics["ssim_3d_hidden"].item() == pytest.approx(1.0)
    assert metrics["ssim_3d_masked"].item() == pytest.approx(metrics["ssim_3d_hidden"].item())
    assert metrics["edge_error_hidden"] == pytest.approx(0.0)
    assert metrics["edge_error_masked"] == pytest.approx(metrics["edge_error_hidden"])
    assert metrics["freq_domain_mse"] == pytest.approx(0.0)
    assert metrics["spectral_error_hidden/residual_power_total"] == pytest.approx(0.0)
    assert metrics["spectral_error_hidden/low"] == pytest.approx(0.0)
    assert metrics["spectral_error_hidden/mid"] == pytest.approx(0.0)
    assert metrics["spectral_error_hidden/high"] == pytest.approx(0.0)
    assert metrics["spectral_error_completed_full/residual_power_total"] == pytest.approx(0.0)


def test_completed_reconstruction_uses_visible_input_and_hidden_prediction():
    masked_input = torch.tensor([[[[[1.0, 0.0]]]]])
    pred = torch.tensor([[[[[9.0, 2.0]]]]])
    mask = torch.tensor([[[[[True, False]]]]])

    completed = reconstruction_metrics.completed_reconstruction(masked_input, pred, mask)

    assert torch.equal(completed, torch.tensor([[[[[1.0, 2.0]]]]]))


def test_psnr_uses_configured_clamped_intensity_range():
    target = torch.zeros(1, 1, 1, 1, 2)
    pred = torch.ones_like(target)
    mask = torch.zeros_like(target, dtype=torch.bool)

    metrics = loss_metrics.compute_psnr_metrics(pred, target, mask, data_range=6.0)

    assert metrics["psnr_hidden"].item() == pytest.approx(20.0 * math.log10(6.0))


def test_reconstruction_only_loss_schema_omits_retired_demographic_aliases():
    module = SelfSupervisedModule(model=torch.nn.Identity(), learning_rate=1e-3)
    mse = torch.tensor(2.0)
    components = module._complete_loss_components(
        {"mse": (mse, 1.0, 1.0, mse, True)},
        mse,
    )
    metrics = module._loss_metrics(mse, components, excluded=("demographic",))

    assert metrics["stage2/cross_reconstruction/enabled"] == 0.0
    assert metrics["stage2/cross_reconstruction/raw"] == 0.0
    assert "demographic/enabled" not in metrics


def test_stability_names_parameter_tensor_and_scalar_gradient_counts():
    model = torch.nn.Linear(3, 2)
    model(torch.ones(1, 3)).sum().backward()

    metrics = stability_metrics.compute_gradient_metrics(model)

    assert metrics["num_params_with_grad"] == metrics["num_parameter_tensors_with_grad"] == 2
    assert metrics["num_scalar_parameters_with_grad"] == 8


def test_reconstruction_only_probe_does_not_invoke_inactive_projection_heads():
    class FailingHead(torch.nn.Module):
        def forward(self, _features):
            raise AssertionError("inactive projection head was invoked")

    class ProbeModel(torch.nn.Module):
        head_demo = FailingHead()
        head_patho = FailingHead()
        head_stage1_anatomy = FailingHead()

        def forward_with_features(self, x, modality_id=None):
            return x, x

    module = SelfSupervisedModule(model=ProbeModel(), learning_rate=1e-3)
    embeddings = module._probe_embeddings({"image": torch.ones(2, 1, 2, 2, 2), "modality_id": torch.zeros(2)})

    assert set(embeddings) == {"h"}


def test_linear_probe_uses_canonical_h_global_representation():
    from asparagus.modules.lightning_modules.linear_probe_module import LinearProbeModule

    class ProbeBackbone(torch.nn.Module):
        global_feature_dim = 3

        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Module()
            self.decoder.fc = torch.nn.Linear(99, 1)

        def encode_representations(self, x):
            return {"h_dense": [], "h_global": torch.full((x.shape[0], 3), 7.0, device=x.device)}

    module = LinearProbeModule(model=ProbeBackbone(), learning_rates=[1e-3], num_classes=2)

    features = module._get_features(torch.zeros(4, 1, 2, 2, 2))

    assert features.shape == (4, 3)
    assert torch.equal(features, torch.full((4, 3), 7.0))
    assert next(iter(module.heads.values())).in_features == 3


def test_encoder_monitor_source_cannot_be_disabled_by_projection_source_config():
    module = SelfSupervisedModule(
        model=torch.nn.Identity(),
        learning_rate=1e-3,
        validation_embedding_sources=["z_demo"],
    )

    assert module._active_validation_embedding_sources() == ("h",)


def test_reconstruction_only_monitor_logs_demographic_probe_on_shared_encoder():
    module = SelfSupervisedModule(
        model=torch.nn.Identity(),
        learning_rate=1e-3,
        validation_embedding_monitor_enabled=True,
        validation_embedding_sources=["h", "z_demo"],
    )
    module._trainer = SimpleNamespace(is_global_zero=True)
    logged = {}
    module.log = lambda name, value, **kwargs: logged.__setitem__(name, (value, kwargs))

    def collected(features, pathology, age, sex, subject, session):
        count = len(pathology)
        return {
            "features": {"h": torch.tensor(features, dtype=torch.float32)},
            "pathology": torch.tensor(pathology),
            "age": torch.tensor(age, dtype=torch.float32),
            "sex": torch.tensor(sex),
            "scanner_id": torch.zeros(count, dtype=torch.long),
            "modality_id": torch.zeros(count, dtype=torch.long),
            "subject_id": torch.tensor(subject),
            "session_id": torch.tensor(session),
        }

    module._probe_reference_batches = [
        collected(
            [[0.0], [1.0], [5.0], [6.0]],
            [0, 0, 1, 1],
            [20, 40, 60, 70],
            [0, 1, 0, 1],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        )
    ]
    module._demographic_probe_reference_batches = [collected([[0.0], [1.0]], [0, 0], [20, 40], [0, 1], [1, 2], [1, 2])]
    module._val_embedding_batches = [
        collected(
            [[0.1], [0.9], [5.1], [5.9]],
            [0, 0, 1, 1],
            [21, 39, 61, 69],
            [0, 1, 0, 1],
            [11, 12, 13, 14],
            [11, 12, 13, 14],
        )
    ]

    module._log_validation_embedding_monitor(on_step=True, on_epoch=False)

    assert "val/probes/h/knn_age/mae" in logged
    assert "val/probes/h/knn_sex/balanced_accuracy" in logged
    assert "val/probes/h/silhouette_sex" in logged
    assert "val/probes/h/valid_sex_count" in logged
    assert "val/probes/h/silhouette_age_bin" in logged
    assert "val/probes/h/valid_age_bin_count" in logged
    assert "val/probes/demographic_naive/median_age/mae" in logged
    assert "val/representation_health/h/dead_dim_fraction" in logged
    assert "val/representation_health/h/effective_rank_normalized" in logged
    assert not any("/z_demo/" in name for name in logged)
    assert logged["val/probes/h/knn_age/mae"][1]["on_step"] is True
    assert logged["val/probes/h/knn_age/mae"][1]["on_epoch"] is False


def test_multimodal_monitor_logs_separate_demographic_probes_by_modality():
    module = SelfSupervisedModule(
        model=torch.nn.Identity(),
        learning_rate=1e-3,
        demographic_modalities=[0, 1],
        validation_embedding_monitor_enabled=True,
        validation_embedding_sources=["h"],
    )
    module._trainer = SimpleNamespace(is_global_zero=True)
    logged = {}
    module.log = lambda name, value, **kwargs: logged.__setitem__(name, (value, kwargs))

    def collected(features, modality, subject_offset):
        count = len(features)
        return {
            "features": {"h": torch.tensor(features, dtype=torch.float32)},
            "pathology": torch.zeros(count, dtype=torch.long),
            "age": torch.tensor([20.0, 40.0, 22.0, 42.0]),
            "sex": torch.tensor([0, 1, 0, 1]),
            "scanner_id": torch.zeros(count, dtype=torch.long),
            "modality_id": torch.tensor(modality),
            "subject_id": torch.arange(subject_offset, subject_offset + count),
            "session_id": torch.arange(subject_offset, subject_offset + count),
        }

    reference = collected([[0.0], [1.0], [10.0], [11.0]], [0, 0, 1, 1], 0)
    query = collected([[0.1], [0.9], [10.1], [10.9]], [0, 0, 1, 1], 10)
    module._probe_reference_batches = [reference]
    module._demographic_probe_reference_batches = [reference]
    module._val_embedding_batches = [query]

    module._log_validation_embedding_monitor(on_step=True, on_epoch=False)

    assert "val/probes/h/by_modality/t1w/knn_age/mae" in logged
    assert "val/probes/h/by_modality/t2w/knn_age/mae" in logged
    assert "val/probes/h/by_modality/t1w/knn_sex/balanced_accuracy" in logged
    assert "val/probes/h/by_modality/t2w/knn_sex/balanced_accuracy" in logged


def test_all_modalities_projection_logs_silhouette_scalars_without_wandb():
    module = SelfSupervisedModule(
        model=torch.nn.Identity(),
        learning_rate=1e-3,
        validation_embedding_monitor_enabled=True,
        validation_embedding_reducers=["pca"],
    )
    module._trainer = SimpleNamespace(is_global_zero=True, loggers=[])
    logged = {}
    module.log = lambda name, value, **kwargs: logged.__setitem__(name, (value, kwargs))

    features = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]], dtype=np.float32)
    pathology = np.zeros(4, dtype=np.int64)
    age = np.full(4, np.nan, dtype=np.float32)
    sex = np.full(4, -1, dtype=np.int64)
    modality = np.array([0, 0, 1, 1], dtype=np.int64)
    scanner = np.array([0, 1, 0, 1], dtype=np.int64)
    dataset = np.array([0, 0, 1, 1], dtype=np.int64)
    field_strength = np.array([0, 0, 1, 1], dtype=np.int64)

    module._log_embedding_projection_figures(
        "z_mod",
        features,
        pathology,
        age,
        sex,
        modality,
        scanner,
        dataset,
        field_strength,
        namespace="all_modalities",
        style="modality",
    )

    key = "val/embeddings/z_mod/all_modalities/pca/silhouette_modality_id"
    assert key in logged
    assert np.isfinite(logged[key][0])
    assert logged["val/embeddings/z_mod/all_modalities/pca/modality_id_class_count"][0] == 2.0


def test_t2w_only_demographic_monitor_keeps_t1w_pathology_cohort_without_legacy_demo_aliases():
    module = SelfSupervisedModule(
        model=torch.nn.Identity(),
        learning_rate=1e-3,
        demographic_modalities=[1],
        validation_embedding_monitor_enabled=True,
        validation_embedding_sources=["h"],
    )
    module._trainer = SimpleNamespace(is_global_zero=True)
    logged = {}
    module.log = lambda name, value, **kwargs: logged.__setitem__(name, (value, kwargs))

    def collected(features, pathology, age, sex, modality, subject_offset):
        count = len(features)
        return {
            "features": {"h": torch.tensor(features, dtype=torch.float32)},
            "pathology": torch.tensor(pathology),
            "age": torch.tensor(age, dtype=torch.float32),
            "sex": torch.tensor(sex),
            "scanner_id": torch.zeros(count, dtype=torch.long),
            "modality_id": torch.tensor(modality),
            "subject_id": torch.arange(subject_offset, subject_offset + count),
            "session_id": torch.arange(subject_offset, subject_offset + count),
        }

    module._probe_reference_batches = [
        collected(
            [[0.0], [0.5], [5.0], [5.5]],
            [0, 0, 1, 1],
            [20, 30, 60, 70],
            [0, 1, 0, 1],
            [0, 0, 0, 0],
            0,
        )
    ]
    module._demographic_probe_reference_batches = [
        collected(
            [[10.0], [11.0], [12.0], [13.0]],
            [0, 0, 0, 0],
            [20, 30, 60, 70],
            [0, 1, 0, 1],
            [1, 1, 1, 1],
            10,
        )
    ]
    module._val_embedding_batches = [
        collected(
            [[0.1], [0.4], [5.1], [5.4], [10.1], [10.9], [12.1], [12.9]],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [21, 31, 61, 71, 21, 31, 61, 71],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 1, 1, 1],
            20,
        )
    ]

    module._log_validation_embedding_monitor(on_step=True, on_epoch=False)

    assert np.isfinite(logged["val/probes/h/knn_pathology/balanced_accuracy"][0])
    assert "val/probes/h/by_modality/t2w/knn_age/mae" in logged
    assert "val/probes/h/knn_age/mae" not in logged
    assert "val/probes/demographic_naive/by_modality/t2w/median_age/mae" in logged
    assert "val/probes/demographic_naive/median_age/mae" not in logged
    assert module._monitor_modality_ids() == (0, 1)


def test_stage2_monitor_logs_factorized_branch_probes():
    module = SelfSupervisedModule(
        model=torch.nn.Identity(),
        learning_rate=1e-3,
        enable_stage2_loss=True,
        validation_embedding_monitor_enabled=True,
        validation_embedding_sources=["z_anat_s2", "z_mod_s2"],
    )
    module._trainer = SimpleNamespace(is_global_zero=True)
    logged = {}
    module.log = lambda name, value, **kwargs: logged.__setitem__(name, (value, kwargs))

    def collected(features, modality_id, subject_offset):
        count = len(features)
        tensor = torch.tensor(features, dtype=torch.float32)
        return {
            "features": {"h": tensor, "z_anat_s2": tensor, "z_mod_s2": tensor},
            "pathology": torch.tensor([0, 0, 1, 1]),
            "fine_pathology": torch.tensor([0, 0, 1, 1]),
            "age": torch.tensor([20.0, 22.0, 60.0, 62.0], dtype=torch.float32),
            "sex": torch.tensor([0, 1, 0, 1]),
            "scanner_id": torch.tensor([0, 0, 1, 1]),
            "dataset_id": torch.tensor([0, 0, 1, 1]),
            "field_strength_id": torch.tensor([0, 0, 1, 1]),
            "modality_id": torch.tensor(modality_id),
            "subject_id": torch.arange(subject_offset, subject_offset + count),
            "session_id": torch.arange(subject_offset, subject_offset + count),
        }

    module._probe_reference_batches = [
        collected(
            [[0.0], [0.2], [5.0], [5.2]],
            [0, 1, 0, 1],
            0,
        )
    ]
    module._val_embedding_batches = [
        collected(
            [[0.1], [0.3], [5.1], [5.3]],
            [0, 1, 0, 1],
            10,
        )
    ]
    module._demographic_probe_reference_batches = []

    module._log_validation_embedding_monitor(on_step=True, on_epoch=False)

    assert "val/probes/z_anat_s2/knn_modality/balanced_accuracy" in logged
    assert "val/probes/z_mod_s2/knn_modality/balanced_accuracy" in logged
    assert "val/probes/z_mod_s2/knn_manufacturer/balanced_accuracy" in logged
    assert "val/embeddings/z_mod_s2/all_modalities/pca/silhouette_modality_id" in logged


def test_stage1_monitor_logs_factorized_branch_leakage_probes():
    module = SelfSupervisedModule(
        model=torch.nn.Identity(),
        learning_rate=1e-3,
        enable_stage1_loss=True,
        enable_modality_loss=True,
        validation_embedding_monitor_enabled=True,
        validation_embedding_sources=["z_anatomy", "z_mod"],
    )
    module._trainer = SimpleNamespace(is_global_zero=True)
    logged = {}
    module.log = lambda name, value, **kwargs: logged.__setitem__(name, (value, kwargs))

    def collected(features, modality_id, subject_offset):
        count = len(features)
        tensor = torch.tensor(features, dtype=torch.float32)
        return {
            "features": {"h": tensor, "z_anatomy": tensor, "z_mod": tensor},
            "pathology": torch.tensor([0, 0, 1, 1]),
            "fine_pathology": torch.tensor([0, 0, 1, 1]),
            "age": torch.tensor([20.0, 22.0, 60.0, 62.0], dtype=torch.float32),
            "sex": torch.tensor([0, 1, 0, 1]),
            "scanner_id": torch.tensor([0, 0, 1, 1]),
            "dataset_id": torch.tensor([0, 0, 1, 1]),
            "field_strength_id": torch.tensor([0, 0, 1, 1]),
            "modality_id": torch.tensor(modality_id),
            "subject_id": torch.arange(subject_offset, subject_offset + count),
            "session_id": torch.arange(subject_offset, subject_offset + count),
        }

    module._probe_reference_batches = [collected([[0.0], [0.2], [5.0], [5.2]], [0, 1, 0, 1], 0)]
    module._modality_probe_reference_batches = [collected([[0.0], [5.0], [0.2], [5.2]], [0, 1, 0, 1], 20)]
    module._val_embedding_batches = [collected([[0.1], [5.1], [0.3], [5.3]], [0, 1, 0, 1], 10)]
    module._demographic_probe_reference_batches = []

    module._log_validation_embedding_monitor(on_step=True, on_epoch=False)

    assert "val/probes/z_anatomy/knn_modality/balanced_accuracy" in logged
    assert "val/probes/z_mod/knn_modality/balanced_accuracy" in logged
    assert logged["val/probes/z_mod/knn_modality/balanced_accuracy"][0] > 0.5
    assert "val/retrieval/z_anatomy/same_macro/top1" in logged
    assert "val/retrieval/z_mod/same_macro/top1" in logged


def test_stage1_epoch_monitor_logs_z_mod_subject_retrieval_and_distribution():
    module = SelfSupervisedModule(model=torch.nn.Identity(), learning_rate=1e-3)
    logged = {}
    module.log_dict = lambda metrics, **kwargs: logged.update(metrics)
    module._val_stage1_batches = [
        {
            "features": torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
            "z_mod_features": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
            "subject_id": torch.tensor([1, 1, 2, 2]),
            "subject_session_id": torch.tensor([11, 11, 22, 22]),
            "modality_id": torch.tensor([0, 1, 0, 1]),
            "registered_subset": torch.ones(4, dtype=torch.bool),
        }
    ]

    module.on_validation_epoch_end()

    assert "val/stage1_retrieval/z_mod/retrieval_at_1" in logged
    assert "val/stage1_retrieval/z_mod_cross_modal_only/retrieval_at_1" in logged
    assert "val/stage1_z_mod_distribution/effective_rank_ratio" in logged


def test_scanner_acquisition_projection_uses_routine_projection_collection():
    module = SelfSupervisedModule(
        model=torch.nn.Identity(),
        learning_rate=1e-3,
        validation_embedding_monitor_enabled=True,
        validation_embedding_sources=["h"],
    )
    module._trainer = SimpleNamespace(is_global_zero=True)
    module.log = lambda *args, **kwargs: None

    def collected(features, modality, subject_offset):
        count = len(features)
        return {
            "features": {"h": torch.tensor(features, dtype=torch.float32)},
            "pathology": torch.zeros(count, dtype=torch.long),
            "age": torch.tensor([20.0, 22.0, 24.0, 26.0][:count], dtype=torch.float32),
            "sex": torch.tensor([0, 1, 0, 1][:count], dtype=torch.long),
            "scanner_id": torch.tensor([0, 1, 0, 1][:count], dtype=torch.long),
            "dataset_id": torch.tensor([2, 2, 3, 3][:count], dtype=torch.long),
            "field_strength_id": torch.tensor([0, 1, 0, 1][:count], dtype=torch.long),
            "modality_id": torch.tensor(modality, dtype=torch.long),
            "subject_id": torch.arange(subject_offset, subject_offset + count),
            "session_id": torch.arange(subject_offset, subject_offset + count),
        }

    module._probe_reference_batches = [collected([[0.0], [0.5], [4.0], [4.5]], [0, 0, 0, 0], 0)]
    module._val_embedding_batches = [collected([[0.1], [0.4], [4.1], [4.4]], [0, 0, 0, 0], 10)]
    module._projection_embedding_batches = [collected([[0.1], [0.4], [4.1], [4.4]], [0, 1, 0, 1], 20)]
    module._demographic_probe_reference_batches = []

    calls = []

    def record_projection(*args, **kwargs):
        calls.append(
            {
                "namespace": kwargs.get("namespace"),
                "style": kwargs.get("style"),
                "modality": np.asarray(args[5]).copy(),
                "field_strength": np.asarray(args[8]).copy(),
            }
        )

    module._log_embedding_projection_figures = record_projection

    module._log_validation_embedding_monitor(on_step=True, on_epoch=False)

    scanner_calls = [call for call in calls if call["namespace"] == "scanner_acquisition"]
    assert scanner_calls
    assert scanner_calls[0]["style"] == "scanner_acquisition"
    assert set(scanner_calls[0]["modality"].tolist()) == {0, 1}
    assert set(scanner_calls[0]["field_strength"].tolist()) == {0, 1}


def test_initial_embedding_monitor_runs_once_at_step_zero():
    module = SelfSupervisedModule(
        model=torch.nn.Identity(),
        learning_rate=1e-3,
        validation_embedding_monitor_enabled=True,
    )
    module._trainer = SimpleNamespace(global_step=0, sanity_checking=False)
    calls = []
    module._run_probe_monitor_dataloaders = lambda: calls.append(("collect", None))
    module._log_validation_embedding_monitor = lambda **kwargs: calls.append(("log", kwargs))

    module.on_train_start()
    module.on_train_start()

    assert calls == [
        ("collect", None),
        ("log", {"direct_step": 0}),
    ]
    assert module._initial_embedding_monitor_logged is True


def test_initial_monitor_scalar_direct_logging_uses_explicit_step():
    class ScalarLogger:
        def __init__(self):
            self.metrics = []

        def log_metrics(self, values, step=None):
            self.metrics.append((values, step))

    logger = ScalarLogger()
    module = SelfSupervisedModule(model=torch.nn.Identity(), learning_rate=1e-3)
    module._trainer = SimpleNamespace(is_global_zero=True, loggers=[logger])

    module._log_monitor_scalar("val/probes/h/silhouette_age_bin", 0.25, on_step=False, on_epoch=False, direct_step=0)

    assert logger.metrics == [({"val/probes/h/silhouette_age_bin": 0.25}, 0)]


def test_embedding_projection_figures_log_with_current_global_step():
    pytest.importorskip("wandb")

    class WandbLogger:
        def __init__(self):
            self.steps = []
            self.experiment = self

        def log(self, _data, step=None):
            self.steps.append(step)

    logger = WandbLogger()
    module = SelfSupervisedModule(
        model=torch.nn.Identity(),
        learning_rate=1e-3,
        validation_embedding_monitor_enabled=True,
        validation_embedding_reducers=["pca"],
    )
    module._trainer = SimpleNamespace(loggers=[logger], global_step=17, current_epoch=2)

    module._log_embedding_projection_figures(
        "h",
        np.array([[0.0, 0.0], [0.1, 0.2], [3.0, 3.0]], dtype=np.float32),
        np.array([0, 0, 0], dtype=np.int64),
        np.array([20.0, 30.0, 40.0], dtype=np.float32),
        np.array([0, 1, 0], dtype=np.int64),
        np.array([0, 0, 0], dtype=np.int64),
        namespace="control_only",
        style="control_age",
    )

    assert logger.steps == [17]


def test_validation_mask_transform_is_deterministic_per_batch_identity():
    class RandomMask:
        def __call__(self, batch):
            batch = dict(batch)
            batch["mask"] = torch.rand_like(batch["image"]) > 0.5
            return batch

    module = SelfSupervisedModule(
        model=torch.nn.Identity(),
        learning_rate=1e-3,
        val_transforms=RandomMask(),
        validation_mask_seed=13,
    )
    batch = {"image": torch.ones(2, 1, 3, 3, 3), "file_path": ["a", "b"]}
    first = module._apply_deterministic_val_transforms(dict(batch))["mask"]
    second = module._apply_deterministic_val_transforms(dict(batch))["mask"]
    assert torch.equal(first, second)


def test_stateful_replacement_sampler_resumes_at_exact_next_index():
    sampler = StatefulReplacementSampler(list(range(11)), num_samples=20, seed=37)
    iterator = iter(sampler)
    consumed = [next(iterator) for _ in range(7)]
    state = sampler.state_dict()
    expected_remainder = list(iterator)

    resumed = StatefulReplacementSampler(list(range(11)), num_samples=20, seed=37)
    resumed.load_state_dict(state)
    assert len(consumed) == 7
    assert list(resumed) == expected_remainder


def test_stateful_sampler_checkpoints_consumed_not_prefetched_position():
    baseline = list(StatefulReplacementSampler(list(range(11)), num_samples=20, seed=37))
    sampler = StatefulReplacementSampler(list(range(11)), num_samples=20, seed=37)
    sampler.enable_consumption_tracking()
    iterator = iter(sampler)
    prefetched = [next(iterator) for _ in range(7)]
    sampler.mark_consumed(3)
    state = sampler.state_dict()

    resumed = StatefulReplacementSampler(list(range(11)), num_samples=20, seed=37)
    resumed.load_state_dict(state)
    assert prefetched == baseline[:7]
    assert list(resumed) == baseline[3:]


def test_stateful_sampler_pending_resume_survives_lightning_set_epoch():
    baseline = list(StatefulReplacementSampler(list(range(97)), num_samples=40, seed=37))
    resumed = StatefulReplacementSampler(list(range(97)), num_samples=40, seed=37)
    resumed.load_state_dict({"iteration": 0, "position": 12})
    # Lightning calls set_epoch before the restored iterator yields its first item.
    resumed.set_epoch(0)
    assert list(resumed) == baseline[12:]


def test_demographic_sampler_selects_one_modality_and_unique_global_subjects():
    rows = [(modality_id, 20.0 + index, index % 2, 0) for modality_id in (0, 1) for index in range(12)]
    dataset = SimpleNamespace(
        files=[f"scan_{index}.pt" for index in range(len(rows))],
        _cached_metadata=[
            {
                "identity": {
                    "modality_id": modality_id,
                    "modality": "t1w" if modality_id == 0 else "t2w",
                    "subject_key": f"sub-{index}",
                    "subject_session_key": f"sub-{index}|ses-01",
                },
                "common": {"age": age, "sex": sex, "pathology": pathology, "scanner_id": 0},
            }
            for index, (modality_id, age, sex, pathology) in enumerate(rows)
        ],
    )

    sampler = SameSessionMultimodalSampler(
        dataset,
        batch_size=4,
        num_samples=64,
        multimodal_probability=0.0,
        demographic_probability=1.0,
        demographic_modalities=[0, 1],
        seed=7,
        num_replicas=2,
        rank=0,
    )
    peer = SameSessionMultimodalSampler(
        dataset,
        batch_size=4,
        num_samples=64,
        multimodal_probability=0.0,
        demographic_probability=1.0,
        demographic_modalities=[0, 1],
        seed=7,
        num_replicas=2,
        rank=1,
    )
    assert all(
        isinstance(sex, int) and isinstance(age_bin, int)
        for strata in sampler._demographic_strata_by_token.values()
        for sex, age_bin in strata
    )

    sampled_by_rank = [list(sampler), list(peer)]
    modality_counts = Counter()
    for offset in range(0, len(sampled_by_rank[0]), 4):
        global_batch = sampled_by_rank[0][offset : offset + 4] + sampled_by_rank[1][offset : offset + 4]
        modalities = {dataset._cached_metadata[index]["identity"]["modality_id"] for index in global_batch}
        subjects = {dataset._cached_metadata[index]["identity"]["subject_key"] for index in global_batch}
        assert len(modalities) == 1
        assert len(subjects) == len(global_batch)
        modality_counts[next(iter(modalities))] += 1
    assert set(modality_counts) == {0, 1}


def test_packed_stage1_sampler_fills_batch_with_allowed_pairs():
    rows = []
    modality_names = {0: "t1w", 1: "t2w", 2: "flair"}
    for session_index in range(12):
        for modality_id in (0, 1, 2):
            rows.append((session_index, modality_id))
    dataset = SimpleNamespace(
        files=[f"sub-{session:02d}_ses-01_{modality_names[modality_id]}.pt" for session, modality_id in rows],
        _cached_metadata=[
            {
                "identity": {
                    "modality_id": modality_id,
                    "modality": modality_names[modality_id],
                    "subject_key": f"sub-{session:02d}",
                    "subject_session_key": f"sub-{session:02d}|ses-01",
                    "is_registered_subset": False,
                },
                "common": {"age": 30.0, "sex": 0, "pathology": 0, "scanner_id": 0},
            }
            for session, modality_id in rows
        ],
    )

    sampler = SameSessionMultimodalSampler(
        dataset,
        batch_size=16,
        num_samples=16,
        multimodal_probability=1.0,
        demographic_probability=0.0,
        stage1_multimodal_batch_mode="packed_pairs",
        stage1_multimodal_allowed_pairs=["t1w-t2w"],
        seed=17,
    )

    indices = list(sampler)
    by_session = defaultdict(list)
    for index in indices:
        identity = dataset._cached_metadata[index]["identity"]
        by_session[identity["subject_session_key"]].append(identity["modality_id"])

    assert len(indices) == 16
    assert len(by_session) == 8
    assert all(sorted(modalities) == [0, 1] for modalities in by_session.values())


def test_packed_stage1_sampler_respects_structural_allowed_pair_subset():
    rows = []
    modality_names = {0: "t1w", 1: "t2w", 2: "flair"}
    for session_index in range(10):
        for modality_id in (0, 1, 2):
            rows.append((session_index, modality_id))
    dataset = SimpleNamespace(
        files=[f"sub-{session:02d}_ses-01_{modality_names[modality_id]}.pt" for session, modality_id in rows],
        _cached_metadata=[
            {
                "identity": {
                    "modality_id": modality_id,
                    "modality": modality_names[modality_id],
                    "subject_key": f"sub-{session:02d}",
                    "subject_session_key": f"sub-{session:02d}|ses-01",
                    "is_registered_subset": False,
                },
                "common": {"age": 30.0, "sex": 0, "pathology": 0, "scanner_id": 0},
            }
            for session, modality_id in rows
        ],
    )

    sampler = SameSessionMultimodalSampler(
        dataset,
        batch_size=12,
        num_samples=12,
        multimodal_probability=1.0,
        demographic_probability=0.0,
        stage1_multimodal_batch_mode="packed_pairs",
        stage1_multimodal_allowed_pairs=["t1w-flair"],
        seed=19,
    )

    indices = list(sampler)
    by_session = defaultdict(list)
    for index in indices:
        identity = dataset._cached_metadata[index]["identity"]
        by_session[identity["subject_session_key"]].append(identity["modality_id"])

    assert len(by_session) == 6
    assert all(sorted(modalities) == [0, 2] for modalities in by_session.values())

    allowed_pairs = SameSessionMultimodalSampler._normalize_modality_pairs(["t1w-flair"])
    monitor_indices = PretrainDataModule._stage1_pair_indices(dataset, session_count=6, seed=23, allowed_pairs=allowed_pairs)
    monitor_by_session = defaultdict(list)
    for index in monitor_indices:
        identity = dataset._cached_metadata[index]["identity"]
        monitor_by_session[identity["subject_session_key"]].append(identity["modality_id"])
    assert len(monitor_by_session) == 6
    assert all(sorted(modalities) == [0, 2] for modalities in monitor_by_session.values())


def test_packed_stage1_setup_rejects_missing_allowed_pair_support(tmp_path):
    train_paths = []
    val_paths = []
    rows = []
    for subject, split in (("sub-01", "train"), ("sub-02", "val")):
        rows.append(
            {
                "dataset": "PT001_A",
                "participant_id": subject,
                "session_id": "ses-01",
                "age": 30,
                "sex": "F",
                "group": "Control",
            }
        )
        for modality in ("T1w", "FLAIR"):
            path = tmp_path / "PT001_A" / subject / "ses-01" / "anat" / f"{subject}_ses-01_{modality}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.ones(1, 2, 2, 2), path)
            (train_paths if split == "train" else val_paths).append(str(path))
    metadata_path = tmp_path / "participants.tsv"
    pd.DataFrame(rows).to_csv(metadata_path, sep="\t", index=False)

    module = PretrainDataModule(
        batch_size=4,
        num_workers=0,
        train_split=train_paths,
        val_split=val_paths,
        metadata_paths=[str(metadata_path)],
        same_session_multimodal_batches=True,
        multimodal_batch_probability=1.0,
        stage1_multimodal_batch_mode="packed_pairs",
        stage1_multimodal_allowed_pairs=["t1w-t2w"],
        require_multimodal_samples=True,
    )

    with pytest.raises(ValueError, match="allowed_pairs=t1w-t2w"):
        module.setup_fit()


def test_distributed_sampler_keeps_demographic_batch_schedule_aligned_across_ranks():
    rows = [(0, 20.0 + index, index % 2, 0) for index in range(8)]
    rows.extend([(2, 30.0, 0, 1), (4, 45.0, 1, 1)])
    dataset = SimpleNamespace(
        files=[f"scan_{index}.pt" for index in range(len(rows))],
        _cached_metadata=[
            {
                "identity": {
                    "modality_id": modality_id,
                    "modality": {0: "t1w", 2: "flair", 4: "dwi"}[modality_id],
                    "subject_key": f"sub-{index}",
                    "subject_session_key": f"sub-{index}|ses-01",
                },
                "common": {"age": age, "sex": sex, "pathology": pathology, "scanner_id": 0},
            }
            for index, (modality_id, age, sex, pathology) in enumerate(rows)
        ],
    )

    samplers = [
        SameSessionMultimodalSampler(
            dataset,
            batch_size=4,
            num_samples=32,
            multimodal_probability=0.0,
            demographic_probability=0.5,
            demographic_modalities=[0],
            seed=0,
            num_replicas=2,
            rank=rank,
        )
        for rank in range(2)
    ]
    for sampler in samplers:
        # Make non-demographic fallback batches unambiguous for this schedule test.
        sampler._all_indices = [8, 9]

    sampled_by_rank = [list(sampler) for sampler in samplers]
    assert [len(indices) for indices in sampled_by_rank] == [16, 16]
    observed_kinds = []
    for offset in range(0, 16, 4):
        rank_batches = [indices[offset : offset + 4] for indices in sampled_by_rank]
        pathology_sets = [
            {dataset._cached_metadata[index]["common"]["pathology"] for index in batch} for batch in rank_batches
        ]
        assert pathology_sets[0] == pathology_sets[1]
        assert pathology_sets[0] in ({0}, {1})
        if pathology_sets[0] == {0}:
            assert all(
                dataset._cached_metadata[index]["identity"]["modality_id"] == 0 for batch in rank_batches for index in batch
            )
        observed_kinds.append(next(iter(pathology_sets[0])))
    assert set(observed_kinds) == {0, 1}


def test_distributed_sampler_context_broadcasts_rank_zero_seed(monkeypatch):
    monkeypatch.setattr("asparagus.modules.data_modules.pretraining.dist.is_available", lambda: True)
    monkeypatch.setattr("asparagus.modules.data_modules.pretraining.dist.is_initialized", lambda: True)
    monkeypatch.setattr("asparagus.modules.data_modules.pretraining.dist.get_world_size", lambda: 2)
    monkeypatch.setattr("asparagus.modules.data_modules.pretraining.dist.get_rank", lambda: 1)

    def broadcast(values, src):
        assert src == 0
        values[0] = 431027

    monkeypatch.setattr("asparagus.modules.data_modules.pretraining.dist.broadcast_object_list", broadcast)

    assert _distributed_sampler_context(999999) == (2, 1, 431027)


def test_multimodal_sampler_rejects_probability_sum_above_one():
    dataset = SimpleNamespace(files=[])

    with pytest.raises(ValueError, match="sum to at most 1.0"):
        SameSessionMultimodalSampler(
            dataset,
            batch_size=4,
            num_samples=4,
            multimodal_probability=0.7,
            demographic_probability=0.4,
        )


def test_modality_sync_diagnostics_have_rank_invariant_schema():
    rank0 = {
        "loss": 0.2,
        "accuracy": 0.5,
        "support/class_0": 2.0,
        "recall/class_0": 0.5,
        "precision/class_1": 1.0,
    }
    rank1 = {
        "loss": 0.3,
        "accuracy": 0.75,
        "support/class_1": 2.0,
        "recall/class_1": 0.5,
        "precision/class_0": 1.0,
    }

    filtered0 = SelfSupervisedModule._modality_diagnostics_for_sync_dist(rank0)
    filtered1 = SelfSupervisedModule._modality_diagnostics_for_sync_dist(rank1)

    assert set(filtered0) == set(filtered1) == {"loss", "accuracy"}


def test_demographic_sampler_rejects_modality_without_enough_unique_global_subjects():
    dataset = SimpleNamespace(
        files=[f"scan_{index}.pt" for index in range(3)],
        _cached_metadata=[
            {
                "identity": {
                    "modality_id": 2,
                    "modality": "flair",
                    "subject_key": f"sub-{index}",
                    "subject_session_key": f"sub-{index}|ses-01",
                },
                "common": {"age": 30.0 + index, "sex": index % 2, "pathology": 0, "scanner_id": 0},
            }
            for index in range(3)
        ],
    )

    with pytest.raises(ValueError, match="unique eligible subject") as error:
        SameSessionMultimodalSampler(
            dataset,
            batch_size=2,
            num_samples=8,
            multimodal_probability=0.0,
            demographic_probability=1.0,
            demographic_modalities=[2],
            num_replicas=2,
            rank=0,
        )
    assert "available={'flair': 3}" in str(error.value)
    assert "subject replacement is intentionally disabled" in str(error.value)


def test_stage1_validation_reports_cross_modal_retrieval_for_registered_and_unregistered_scans():
    features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [1.0, 0.0]])
    sessions = torch.tensor([1, 1, 2, 2, 1])
    modalities = torch.tensor([0, 1, 0, 2, 0])
    registered = torch.tensor([True, True, False, False, False])

    metrics = distribution_metrics.compute_cross_modal_retrieval(features, sessions, modalities, registered)

    assert metrics["eligible_anchor_count"] == 5.0
    assert metrics["registered_eligible_anchor_count"] == 2.0
    assert metrics["nonregistered_eligible_anchor_count"] == 3.0
    assert metrics["retrieval_at_1"] == pytest.approx(1.0)
    assert metrics["retrieval_at_5"] == pytest.approx(1.0)
    assert metrics["mean_reciprocal_rank"] > 0.0
    assert metrics["alignment_cosine"] > 0.9


def test_stage1_retrieval_treats_high_alignment_without_retrieval_as_failure():
    features = torch.ones(6, 3)
    sessions = torch.tensor([1, 1, 2, 2, 3, 3])
    modalities = torch.tensor([0, 1, 0, 1, 0, 1])

    metrics = distribution_metrics.compute_cross_modal_retrieval(features, sessions, modalities)

    assert metrics["alignment_cosine"] == pytest.approx(1.0)
    assert metrics["retrieval_at_1"] < 1.0
    assert metrics["positive_rank_mean"] > 1.0
    assert metrics["mean_reciprocal_rank"] < 1.0


def test_stage1_retrieval_can_restrict_candidates_to_cross_modal_scans():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.99, 0.01],
            [0.0, 1.0],
        ]
    )
    sessions = torch.tensor([1, 1, 2, 2])
    modalities = torch.tensor([0, 1, 0, 1])

    full = distribution_metrics.compute_cross_modal_retrieval(features, sessions, modalities)
    cross_modal_only = distribution_metrics.compute_cross_modal_retrieval(
        features,
        sessions,
        modalities,
        cross_modal_candidates_only=True,
    )

    assert full["retrieval_at_1"] < 1.0
    assert cross_modal_only["retrieval_at_1"] > full["retrieval_at_1"]
    assert full["top1_same_modality_fraction"] > 0.0
    assert cross_modal_only["top1_same_modality_fraction"] == pytest.approx(0.0)
    assert full["candidate_positive_ratio"] > cross_modal_only["candidate_positive_ratio"]


def test_stage1_retrieval_reports_same_modality_top1_dominance():
    features = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.99, 0.0, 0.05],
            [0.2, 0.9, 0.0],
        ]
    )
    sessions = torch.tensor([1, 1, 2, 2])
    modalities = torch.tensor([0, 1, 0, 1])

    metrics = distribution_metrics.compute_cross_modal_retrieval(features, sessions, modalities)

    assert metrics["candidate_pair_count"] > metrics["positive_pair_count"]
    assert metrics["top1_same_modality_fraction"] > 0.0
    assert metrics["top1_cross_modal_fraction"] < 1.0
    assert metrics["positive_best_negative_margin_mean"] < 0.0
    assert metrics["positive_beats_best_negative_fraction"] < 1.0


def test_stage1_retrieval_neutralizes_same_subject_different_session_candidates():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.99, 0.01],
            [0.0, 1.0],
        ]
    )
    subjects = torch.tensor([1, 1, 1, 2])
    sessions = torch.tensor([10, 10, 11, 20])
    modalities = torch.tensor([0, 1, 0, 1])

    legacy = distribution_metrics.compute_cross_modal_retrieval(features, sessions, modalities)
    subject_aware = distribution_metrics.compute_cross_modal_retrieval(
        features,
        sessions,
        modalities,
        subject_ids=subjects,
    )

    assert legacy["retrieval_at_1"] < 1.0
    assert subject_aware["retrieval_at_1"] == pytest.approx(1.0)
    assert subject_aware["same_subject_cross_session_pair_count"] == 4.0
    assert subject_aware["same_subject_cross_session_excluded_count"] == 4.0
    assert subject_aware["same_subject_cross_session_alignment_cosine"] > 0.0
    assert subject_aware["same_subject_cross_session_retrieval_at_1"] > 0.0


def test_online_model_alias_does_not_duplicate_state_dict_keys():
    module = SelfSupervisedModule(model=torch.nn.Linear(2, 2), learning_rate=1e-3)

    assert not any(key.startswith("_online_model.") for key in module.state_dict())


def test_stage2_rejects_registered_only_false_instead_of_silently_masking_it():
    with pytest.raises(ValueError, match="registered_only=true"):
        SelfSupervisedModule(
            model=torch.nn.Identity(),
            learning_rate=1e-3,
            enable_stage2_loss=True,
            stage2_registered_only=False,
        )


def test_demographic_dwi_subtype_taxonomy_resolution():
    dwi_id = PretrainDataset.MODALITY_TO_ID["dwi"]
    # Structural tokens: no b-value constraint, legacy modality ids preserved.
    for name in ("t1w", "t2w", "flair"):
        token = PretrainDataset.resolve_demographic_token(name)
        assert token == {
            "name": name,
            "modality_id": PretrainDataset.MODALITY_TO_ID[name],
            "bval_min": None,
            "bval_max": None,
        }
    # DWI subtype keeps the dwi modality id (FiLM/Stage-1 untouched) + explicit b-value band.
    token = PretrainDataset.resolve_demographic_token("dwi_b1000")
    assert token == {"name": "dwi_b1000", "modality_id": dwi_id, "bval_min": 900.0, "bval_max": 1100.0}
    assert PretrainDataset.demographic_dwi_subtype_names() == ("dwi_b1000",)
    # The collapsed `dwi` class is NOT a real modality token change: modality inference is
    # untouched (dwi_bval* still collapses to `dwi`) so MODALITY_TO_ID / Stage-1 vocab is stable.
    assert PretrainDataset._infer_modality_from_filename("sub-1_ses-1_dwi_bval1000.nii.gz") == "dwi"
    assert PretrainDataset._infer_dwi_bval_from_filename("sub-1_ses-1_dwi_bval1000.nii.gz") == 1000.0
    assert PretrainDataset._infer_dwi_bval_from_filename("sub-1_ses-1_dwi_bval2200.nii.gz") == 2200.0
    assert math.isnan(PretrainDataset._infer_dwi_bval_from_filename("sub-1_ses-1_t1w.nii.gz"))
    assert math.isnan(PretrainDataset._infer_dwi_bval_from_filename("sub-1_ses-1_dwi.nii.gz"))


def test_demographic_collapsed_dwi_and_unvalidated_subtypes_are_rejected():
    # Collapsed `dwi` mixes heterogeneous b-values -> blocked with an actionable message.
    with pytest.raises(ValueError, match="collapsed 'dwi'|validated DWI subtype"):
        PretrainDataset.resolve_demographic_token("dwi")
    # A DWI subtype outside the validated allowlist is rejected (no implicit taxonomy).
    with pytest.raises(ValueError, match="validated b-value subtype|validated DWI subtype"):
        PretrainDataset.resolve_demographic_token("dwi_b2000")
    with pytest.raises(ValueError):
        PretrainDataset.resolve_demographic_token("dwi_trace")
    # Non-demographic structural modalities (e.g. t1c) are not eligible for the objective.
    with pytest.raises(ValueError, match="restricted to"):
        PretrainDataset.resolve_demographic_token("t1c")


def test_demographic_sampler_separates_dwi_subtype_by_bvalue():
    dwi_id = PretrainDataset.MODALITY_TO_ID["dwi"]
    # 6 in-band (b1000) control subjects + 6 out-of-band (b2000/b0) control subjects.
    rows = []
    for index in range(6):
        rows.append((dwi_id, 1000.0, 20.0 + index, index % 2))
    for index in range(6):
        rows.append((dwi_id, 2000.0 if index % 2 == 0 else 0.0, 20.0 + index, index % 2))
    dataset = SimpleNamespace(
        files=[f"scan_{index}.pt" for index in range(len(rows))],
        _cached_metadata=[
            {
                "identity": {
                    "modality_id": modality_id,
                    "modality": "dwi",
                    "dwi_bval": bval,
                    "subject_key": f"sub-{index}",
                    "subject_session_key": f"sub-{index}|ses-01",
                },
                "common": {"age": age, "sex": sex, "pathology": 0, "scanner_id": 0},
            }
            for index, (modality_id, bval, age, sex) in enumerate(rows)
        ],
    )
    sampler = SameSessionMultimodalSampler(
        dataset,
        batch_size=2,
        num_samples=16,
        multimodal_probability=0.0,
        demographic_probability=1.0,
        demographic_tokens=[{"name": "dwi_b1000", "modality_id": dwi_id, "bval_min": 900.0, "bval_max": 1100.0}],
        seed=11,
        num_replicas=1,
        rank=0,
    )
    # Only the b1000 token is configured; its strata must contain exactly the 6 in-band subjects.
    assert set(sampler._demographic_strata_by_token) == {"dwi_b1000"}
    in_band_subjects = {
        subject for subjects in sampler._demographic_strata_by_token["dwi_b1000"].values() for subject in subjects
    }
    assert in_band_subjects == {f"sub-{index}" for index in range(6)}
    # Every sampled scan is an in-band b1000 control (never a b2000/b0 scan).
    for index in sampler:
        assert dataset._cached_metadata[index]["identity"]["dwi_bval"] == 1000.0


@pytest.fixture
def curated_diffusion_vocab():
    """Enable the FOMO26 curated diffusion vocab for a test, restoring the prior state after."""
    prior = PretrainDataset.curated_diffusion_vocab_enabled()
    PretrainDataset.use_curated_diffusion_vocab(True)
    try:
        yield
    finally:
        PretrainDataset.use_curated_diffusion_vocab(prior)


def test_curated_vocab_resolves_dwi_b1000_modality_and_id(curated_diffusion_vocab):
    # Curated tree names DWI channels explicitly (`_DWI_B1000`) rather than `dwi_bvalN`. With the
    # curated vocab on they must resolve to their own additive ids (15/16), not crash or collapse
    # to generic `dwi` / t1w.
    vocab = PretrainDataset.active_modality_vocab()
    assert vocab["dwi_b1000"] == 15 and vocab["dwi_b0"] == 16
    assert PretrainDataset._infer_modality_from_filename("sub-1_ses-01_DWI_B1000.nii.gz") == "dwi_b1000"
    assert PretrainDataset._infer_modality_from_filename("sub-1_ses-01_DWI_B0.pt") == "dwi_b0"
    # A curated `_DWI_B1000` has no b-value token, so the legacy bval parse is NaN by design.
    assert math.isnan(PretrainDataset._infer_dwi_bval_from_filename("sub-1_ses-01_DWI_B1000.nii.gz"))


def test_curated_vocab_disabled_keeps_legacy_15_class_vocab():
    # Default (curated off): the legacy 15-class vocab is unchanged and `_DWI_B1000` is unknown.
    assert PretrainDataset.curated_diffusion_vocab_enabled() is False
    assert set(PretrainDataset.active_modality_vocab().values()) == set(range(15))
    assert "dwi_b1000" not in PretrainDataset.active_modality_vocab()


def test_resolve_demographic_token_curated_is_modality_based_without_bval_filter(curated_diffusion_vocab):
    # Curated: dwi_b1000/dwi_b0 are clean explicit modalities (id 15/16) with NO b-value filter.
    b1000 = PretrainDataset.resolve_demographic_token("dwi_b1000")
    assert b1000 == {"name": "dwi_b1000", "modality_id": 15, "bval_min": None, "bval_max": None}
    b0 = PretrainDataset.resolve_demographic_token("dwi_b0")
    assert b0 == {"name": "dwi_b0", "modality_id": 16, "bval_min": None, "bval_max": None}
    # Structural still resolves normally; collapsed/non-validated DWI still rejected.
    assert PretrainDataset.resolve_demographic_token("t1w")["modality_id"] == 0
    with pytest.raises(ValueError):
        PretrainDataset.resolve_demographic_token("dwi")
    with pytest.raises(ValueError):
        PretrainDataset.resolve_demographic_token("adc")


def test_resolve_demographic_token_legacy_is_bval_band_on_generic_dwi():
    # Legacy (curated off): dwi_b1000 is the b-value band [900,1100] on the collapsed `dwi` id 4.
    assert PretrainDataset.curated_diffusion_vocab_enabled() is False
    token = PretrainDataset.resolve_demographic_token("dwi_b1000")
    assert token["modality_id"] == PretrainDataset.MODALITY_TO_ID["dwi"]
    assert (token["bval_min"], token["bval_max"]) == (900.0, 1100.0)
    # dwi_b0 is only a curated subtype; it is not a legacy demographic token.
    with pytest.raises(ValueError):
        PretrainDataset.resolve_demographic_token("dwi_b0")
