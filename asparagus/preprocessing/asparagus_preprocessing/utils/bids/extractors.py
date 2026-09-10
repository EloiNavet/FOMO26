import json
import logging
import os
import pandas as pd
import re
from .config import (
    DEFAULT_COLUMN_MAP,
    DEFAULT_MODALITY_PATTERNS,
    DEFAULT_MRI_METADATA_FIELDS,
    DEFAULT_SESSION_PATTERNS,
    SESSION_KEYWORDS,
)
from typing import Dict, List, Tuple


def extract_subject_id_from_path(file_path: str, patterns: List[str] = None) -> str:
    """Extract subject ID from file path with simple pattern matching."""

    if patterns:
        for pattern in patterns:
            match = re.search(pattern, file_path, re.IGNORECASE)
            if match:
                if match.groups():
                    # Take only the first group (subject ID), ignore possible session
                    subject_id = match.group(1)
                else:
                    subject_id = match.group(0)

                if not subject_id.startswith("sub-"):
                    return f"sub-{subject_id}"
                return subject_id

    match = re.search(r"sub[ject]*[-_]?(\w+)", file_path, re.IGNORECASE)
    if match:
        return f"sub-{match.group(1)}"

    basename = os.path.basename(file_path).split(".")[0]
    cleaned = re.sub(r"[^a-zA-Z0-9]", "", basename)
    return f"sub-{cleaned}"


def extract_session_from_path(file_path: str, patterns: List[str] = None) -> str:
    """Extract session information from file path."""

    # First try the custom patterns if provided
    if patterns:
        for pattern in patterns:
            match = re.search(pattern, file_path, re.IGNORECASE)
            if match and len(match.groups()) >= 2:
                return match.group(2)  # Second capture group is session

    # Fall back to default session patterns from config
    for part in file_path.split(os.sep):
        for pattern in DEFAULT_SESSION_PATTERNS:
            match = re.search(pattern, part, re.IGNORECASE)
            if match:
                return match.group(1)

    return "ses-1"


def find_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Find relevant columns in demographics DataFrame."""
    found_cols = {}
    used_columns = set()

    # Check if there are multiple session-like columns
    session_columns = [col for col in df.columns for kw in SESSION_KEYWORDS if kw in col.lower()]
    skip_session = len(session_columns) > 1

    # Use default column map from config
    col_map = DEFAULT_COLUMN_MAP.copy()

    # If multiple session columns found, clear session keywords to skip matching
    if skip_session:
        col_map["session"] = []

    for key, keywords in col_map.items():
        for col in df.columns:
            if col not in used_columns and any(kw in col.lower() for kw in keywords):
                found_cols[key] = col
                used_columns.add(col)
                break
    return found_cols


def generate_session_variants(session: str) -> List[str]:
    """Generate all possible variants of a session identifier for matching."""
    variants = [
        session,
        session.lower(),
        session.upper(),
        session.replace("ses-", ""),
        session.replace("ses-", "").lower(),
        session.replace("ses-", "").upper(),
    ]

    if not session.startswith("ses-"):
        variants.extend([f"ses-{session}", f"ses-{session.lower()}", f"ses-{session.upper()}"])

    return variants


def sessions_match(file_session: str, demo_session: str, delimiters: List[str] = None) -> bool:
    """
    Check if file session matches demographic session.

    Handles:
    - Exact matches (case-insensitive)
    - Numeric equivalence (e.g., "01" == "1")
    - Prefix matches with delimiters (e.g., "ses-1" matches "ses-1_baseline")

    Args:
        file_session: Session ID from file path
        demo_session: Session ID from demographics file
        delimiters: Optional list of delimiter characters (defaults to SESSION_DELIMITERS)

    Returns:
        True if sessions match, False otherwise
    """
    from .config import SESSION_DELIMITERS

    if delimiters is None:
        delimiters = SESSION_DELIMITERS

    file_variants = generate_session_variants(file_session)
    demo_variants = generate_session_variants(demo_session)

    for fv in file_variants:
        for dsv in demo_variants:
            # Exact match
            if fv == dsv:
                return True

            # Numeric equivalence (handles leading zeros)
            if fv.isdigit() and dsv.isdigit() and int(fv) == int(dsv):
                return True

            # Prefix match with delimiter
            if dsv.startswith(fv) and len(dsv) > len(fv):
                next_char = dsv[len(fv)]
                if next_char in delimiters:
                    return True

    return False


def match_numeric_subject_ids(demo_subjects: set, processed_subjects: set) -> set:
    """
    Match numeric subject IDs between demographics and processed files.

    If demographics has 'sub-001' and processed files have 'sub-1',
    this function matches them by comparing the numeric part.

    Args:
        demo_subjects: Set of subject IDs from demographics
        processed_subjects: Set of subject IDs from processed files

    Returns:
        Set of processed subject IDs, with numeric IDs updated to match demographics format
    """
    demo_nums = {int(s[4:]): s for s in demo_subjects if s.startswith("sub-") and s[4:].isdigit()}
    return {
        demo_nums.get(int(s[4:]), s) if s.startswith("sub-") and s[4:].isdigit() and int(s[4:]) in demo_nums else s
        for s in processed_subjects
    }


def detect_modality_info(filename: str, custom_patterns: dict = None) -> Tuple[str, str]:
    """Detect modality suffix and folder from filename."""
    patterns = DEFAULT_MODALITY_PATTERNS.copy()
    if custom_patterns:
        patterns.update(custom_patterns)

    filename_lower = filename.lower()
    for pattern, info in patterns.items():
        if pattern in filename_lower:
            return info["suffix"], info["folder"]

    return "scan", "undefined"


def get_json_path_for_image(image_path: str, custom_json_mapping: dict = None) -> str:
    """
    Find the corresponding JSON file for an image file.

    Args:
        image_path: Path to image file (.nii, .nii.gz, .pt)
        custom_json_mapping: Dict mapping path patterns to JSON file paths

    Returns:
        Path to JSON file (may not exist)
    """
    # Check for custom JSON mapping based on path patterns
    if custom_json_mapping:
        image_path_lower = image_path.lower()
        for pattern, json_path in custom_json_mapping.items():
            if pattern.lower() in image_path_lower:
                return json_path

    base_path = image_path

    # Remove extensions
    if base_path.endswith(".nii.gz"):
        base_path = base_path[:-7]
    elif base_path.endswith(".nii"):
        base_path = base_path[:-4]
    elif base_path.endswith(".pt"):
        base_path = base_path[:-3]

    # Default: same name but .json extension, or any JSON in folder
    default_json = base_path + ".json"
    if os.path.exists(default_json):
        return default_json

    # Fallback: look for any JSON file in the same directory
    folder = os.path.dirname(base_path)
    for file in os.listdir(folder):
        if file.endswith(".json"):
            return os.path.join(folder, file)

    return default_json


def extract_json_metadata(
    json_path: str,
    hardcoded_metadata: dict = None,
    filename: str = "",
    file_path: str = "",
) -> Dict[str, str]:
    """
    Extract MRI metadata from JSON file or use hardcoded metadata.

    Args:
        json_path: Path to JSON file
        hardcoded_metadata: Dict with hardcoded metadata based on modality patterns
        filename: Original filename for pattern matching
        file_path: Full file path for pattern matching

    Returns dict with standardized keys, all values as strings.
    """
    result = {}

    # First try hardcoded metadata if provided
    if hardcoded_metadata:
        # Check against full path first if provided
        if file_path:
            file_path_lower = file_path.lower()
            for pattern, metadata in hardcoded_metadata.items():
                if re.search(pattern.lower(), file_path_lower):
                    # Use hardcoded metadata for this pattern
                    for field in DEFAULT_MRI_METADATA_FIELDS:
                        value = metadata.get(field)
                        if value is not None:
                            if isinstance(value, list):
                                result[field] = ", ".join(str(v) for v in value)
                            else:
                                result[field] = str(value)
                        else:
                            result[field] = None
                    logging.info(f"Using hardcoded metadata for path pattern '{pattern}' in file {file_path}")
                    return result

        # Fallback to filename check
        elif filename:
            filename_lower = filename.lower()
            for pattern, metadata in hardcoded_metadata.items():
                if re.search(pattern.lower(), filename_lower):
                    # Use hardcoded metadata for this pattern
                    for field in DEFAULT_MRI_METADATA_FIELDS:
                        value = metadata.get(field)
                        if value is not None:
                            if isinstance(value, list):
                                result[field] = ", ".join(str(v) for v in value)
                            else:
                                result[field] = str(value)
                        else:
                            result[field] = None
                    logging.info(f"Using hardcoded metadata for filename pattern '{pattern}' in file {filename}")
                    return result

    # If no hardcoded metadata matched or provided, try JSON file
    if not os.path.exists(json_path):
        logging.warning(f"JSON file not found: {json_path}")
        return {field: None for field in DEFAULT_MRI_METADATA_FIELDS}

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        # Check if metadata is nested under 'meta' key (common format)
        if "meta" in json_data and isinstance(json_data["meta"], dict):
            json_data = json_data["meta"]
            logging.info(f"Found metadata nested under 'meta' key in {json_path}")

        # Extract each field, converting to string and handling lists/arrays
        for field in DEFAULT_MRI_METADATA_FIELDS:
            value = json_data.get(field)
            if value is None:
                result[field] = None
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                result[field] = ", ".join(str(v) for v in value)
            else:
                result[field] = str(value)

        # Log successful extraction
        non_none_fields = [f for f in DEFAULT_MRI_METADATA_FIELDS if result.get(f) is not None]
        if non_none_fields:
            logging.info(f"Successfully extracted {len(non_none_fields)} metadata fields from {json_path}")
        else:
            logging.warning(f"No metadata fields found in {json_path}")

    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logging.error(f"JSON parsing error in {json_path}: {e}")
        result = {field: None for field in DEFAULT_MRI_METADATA_FIELDS}
    except Exception as e:
        logging.error(f"Unexpected error reading JSON file {json_path}: {e}")
        result = {field: None for field in DEFAULT_MRI_METADATA_FIELDS}

    return result
