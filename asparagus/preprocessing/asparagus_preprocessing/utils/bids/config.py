"""Default configuration for BIDS organization."""

# File extensions
SUPPORTED_FILE_EXTENSIONS = (".nii.gz", ".nii", ".pt")
EXTENSION_LENGTHS = {".nii.gz": 7, ".nii": 4, ".pt": 3}

# Session matching delimiters
SESSION_DELIMITERS = ["_", "-", "T", " "]

# Metadata fields to extract
DEFAULT_MRI_METADATA_FIELDS = [
    "Modality",
    "MagneticFieldStrength",
    "Manufacturer",
    "ManufacturersModelName",
    "SoftwareVersions",
    "MRAcquisitionType",
    "SeriesDescription",
    "ProtocolName",
    "ScanningSequence",
    "SequenceVariant",
    "ScanOptions",
    "SequenceName",
    "EchoTime",
    "SliceThickness",
    "RepetitionTime",
    "InversionTime",
    "FlipAngle",
]

# Default columns to keep in demographics
DEFAULT_COLUMNS_TO_KEEP = ["participant_id", "age", "sex"]

# Default session extraction patterns
DEFAULT_SESSION_PATTERNS = [
    r"ses-[_-]?(run-[A-Za-z0-9]+)",
    r"ses-([A-Za-z0-9]+)",
    r"session[_-]?([A-Za-z0-9]+)",
    r"visit[_-]?([A-Za-z0-9]+)",
    r"time[_-]?([A-Za-z0-9]+)",
    r"tp[_-]?([A-Za-z0-9]+)",
]

# Keywords for detecting session-like columns (to avoid false matches)
SESSION_KEYWORDS = ["session", "visit", "timepoint", "scanid", "study_datetime"]

# Default column mappings for demographics files
DEFAULT_COLUMN_MAP = {
    "participant_id": [
        "participant_id",
        "patient",
        "subjectid",
        "subject_id",
        "subject id",
        "image_id",
        "mri_id",
        "scanname",
        "subject",
        "participant",
        "subid",
        "patid",
        "preschoolid",
        "name",
        "id",
    ],
    "age": ["agegroup", "age_at_scan", "age"],
    "sex": ["gender", "sexmf", "sex", "m/f"],
    "group": ["diagnosis", "dx", "group"],
    "disease_duration": ["disease_duration"],
    "site": ["site", "center", "scanner"],
    "session": ["session", "visit", "timepoint", "scanid", "study_datetime"],
    "handedness": [
        "handedness",
        "handed",
        "hand",
        "Handedness 1=Right 2=Left",
        "laterality",
    ],
    "ethnicity": ["ethnicity", "ethnic"],
    "race": ["race"],
    "weight": ["weight"],
    "height": ["height"],
    "occupation": ["occupation"],
    "marital": ["marital"],
    "qualification": ["qualification"],
}

# Default mappings for standardization
DEFAULT_SEX_MAPPINGS = {
    "m": "M",
    "male": "M",
    "man": "M",
    "boy": "M",
    "f": "F",
    "female": "F",
    "woman": "F",
    "girl": "F",
    "w": "F",
}

DEFAULT_HANDEDNESS_MAPPINGS = {
    "r": "R",
    "right": "R",
    "right-handed": "R",
    "right handed": "R",
    "righthanded": "R",
    "dextral": "R",
    "l": "L",
    "left": "L",
    "left-handed": "L",
    "left handed": "L",
    "lefthanded": "L",
    "sinistral": "L",
    "a": "A",
    "ambidextrous": "A",
    "ambi": "A",
    "both": "A",
    "ambiguous": "A",
    "r+l": "A",
    "l+r": "A",
}

# Default BIDS modality patterns
DEFAULT_MODALITY_PATTERNS = {
    # Diffusion sequences
    "dwi": {"suffix": "dwi", "folder": "dwi"},
    "dti": {"suffix": "dwi", "folder": "dwi"},
    "diffusion": {"suffix": "dwi", "folder": "dwi"},
    "bval": {"suffix": "dwi", "folder": "dwi"},
    "trace": {"suffix": "dwi", "folder": "dwi"},
    "adc": {"suffix": "adc", "folder": "dwi"},
    "adc_map": {"suffix": "adc", "folder": "dwi"},
    "apparent": {"suffix": "adc", "folder": "dwi"},
    "sbref": {"suffix": "sbref", "folder": "dwi"},
    "b1000": {"suffix": "dwi", "folder": "dwi"},
    "b2000": {"suffix": "dwi", "folder": "dwi"},
    "b3000": {"suffix": "dwi", "folder": "dwi"},
    # Perfusion sequences
    "cbf": {"suffix": "cbf", "folder": "perf"},
    "cerebral_blood_flow": {"suffix": "cbf", "folder": "perf"},
    "asl": {"suffix": "asl", "folder": "perf"},
    "perf": {"suffix": "perf", "folder": "perf"},
    "perfusion": {"suffix": "perf", "folder": "perf"},
    "mzero": {"suffix": "m0scan", "folder": "perf"},
    "m0scan": {"suffix": "m0scan", "folder": "perf"},
    "basil_att": {"suffix": "basil_att", "folder": "perf"},
    # Anatomical sequences
    "t1ce": {"suffix": "T1c", "folder": "anat"},
    "t1c": {"suffix": "T1c", "folder": "anat"},
    "gad": {"suffix": "T1c", "folder": "anat"},
    "t1gd": {"suffix": "T1c", "folder": "anat"},
    "t1+gad": {"suffix": "T1c", "folder": "anat"},
    "t1+c": {"suffix": "T1c", "folder": "anat"},
    "post": {"suffix": "T1c", "folder": "anat"},
    "contrast": {"suffix": "T1c", "folder": "anat"},
    "mprage+c": {"suffix": "T1c", "folder": "anat"},
    "mprage_post": {"suffix": "T1c", "folder": "anat"},
    "flair": {"suffix": "FLAIR", "folder": "anat"},
    "t2f": {"suffix": "FLAIR", "folder": "anat"},
    "t2_flair": {"suffix": "FLAIR", "folder": "anat"},
    "t2flair": {"suffix": "FLAIR", "folder": "anat"},
    "t2-flair": {"suffix": "FLAIR", "folder": "anat"},
    "mp2rage": {"suffix": "MP2RAGE", "folder": "anat"},
    "mprage": {"suffix": "T1w", "folder": "anat"},
    "bravo": {"suffix": "T1w", "folder": "anat"},
    "fspgr": {"suffix": "T1w", "folder": "anat"},
    "spgr": {"suffix": "T1w", "folder": "anat"},
    "spgrir": {"suffix": "T1w", "folder": "anat"},
    "ir": {"suffix": "T1w", "folder": "anat"},
    "mpr": {"suffix": "T1w", "folder": "anat"},
    "flash": {"suffix": "FLASH", "folder": "anat"},
    "t2": {"suffix": "T2w", "folder": "anat"},
    "t2w": {"suffix": "T2w", "folder": "anat"},
    "pd": {"suffix": "PDw", "folder": "anat"},
    "pdw": {"suffix": "PDw", "folder": "anat"},
    "proton": {"suffix": "PDw", "folder": "anat"},
    "t1map": {"suffix": "T1map", "folder": "anat"},
    "t1_map": {"suffix": "T1map", "folder": "anat"},
    "mtr": {"suffix": "MTRmap", "folder": "anat"},
    "t1relaxometry": {"suffix": "T1map", "folder": "anat"},
    "t2map": {"suffix": "T2map", "folder": "anat"},
    "t2_map": {"suffix": "T2map", "folder": "anat"},
    "t2relaxometry": {"suffix": "T2map", "folder": "anat"},
    "r1map": {"suffix": "R1map", "folder": "anat"},
    "r2map": {"suffix": "R2map", "folder": "anat"},
    "r2star": {"suffix": "R2starmap", "folder": "anat"},
    "r2starmap": {"suffix": "R2starmap", "folder": "anat"},
    "t2starmap": {"suffix": "T2starmap", "folder": "anat"},
    "ute": {"suffix": "UTE", "folder": "anat"},
    "t2star": {"suffix": "T2starw", "folder": "anat"},
    "t2*": {"suffix": "T2starw", "folder": "anat"},
    "t2s": {"suffix": "T2starw", "folder": "anat"},
    "gre": {"suffix": "gre", "folder": "anat"},
    "swi": {"suffix": "swi", "folder": "anat"},
    "susceptibility": {"suffix": "swi", "folder": "anat"},
    "minip": {"suffix": "swi", "folder": "anat"},
    "swan": {"suffix": "swi", "folder": "anat"},
    "mra": {"suffix": "angio", "folder": "anat"},
    "angio": {"suffix": "angio", "folder": "anat"},
    "veno": {"suffix": "angio", "folder": "anat"},
    "vaso": {"suffix": "angio", "folder": "anat"},
    "unit1": {"suffix": "UNIT1", "folder": "anat"},
    "tof": {"suffix": "angio", "folder": "anat"},
    "t1": {"suffix": "T1w", "folder": "anat"},
    "mese": {"suffix": "MESE", "folder": "anat"},
    # PET sequences
    "pet": {"suffix": "pet", "folder": "pet"},
}
