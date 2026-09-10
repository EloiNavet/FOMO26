import pandas as pd
from .config import DEFAULT_HANDEDNESS_MAPPINGS, DEFAULT_SEX_MAPPINGS
from typing import Optional


def _apply_mapping(value, custom_mapping: dict, default_mapping: dict) -> Optional[str]:
    """
    Helper to apply custom or default mapping to a value.

    Args:
        value: Value to standardize
        custom_mapping: Optional custom mapping dict
        default_mapping: Default mapping dict

    Returns:
        Standardized value or None if no match found
    """
    if pd.isna(value):
        return value

    value_str = str(value).strip()

    # Try custom mapping first
    if custom_mapping:
        # Handle numeric mappings
        try:
            numeric_key = int(float(value_str))
            if numeric_key in custom_mapping:
                return custom_mapping[numeric_key]
        except (ValueError, TypeError):
            pass

        # Handle string mappings
        if value_str in custom_mapping:
            return custom_mapping[value_str]

    # Use default mapping
    return default_mapping.get(value_str.lower(), None)


def standardize_sex(sex_value, sex_mapping: dict = None) -> str:
    """
    Standardize sex values to M or F.

    Args:
        sex_value: Original sex value
        sex_mapping: Optional custom mapping dict, e.g., {0: 'M', 1: 'F'}

    Returns:
        Standardized sex as 'M' or 'F', or None value if can't be standardized
    """
    return _apply_mapping(sex_value, sex_mapping, DEFAULT_SEX_MAPPINGS)


def standardize_handedness(handedness_value, handedness_mapping: dict = None) -> str:
    """
    Standardize handedness values to R, L, or A.

    Args:
        handedness_value: Original handedness value
        handedness_mapping: Optional custom mapping dict, e.g., {0: 'R', 1: 'L'}
    Returns:
        Standardized handedness as 'R', 'L', or 'A', or None value if can't be standardized
    """
    return _apply_mapping(handedness_value, handedness_mapping, DEFAULT_HANDEDNESS_MAPPINGS)


def standardize_group(group_value, group_mapping: dict = None, default_group: str = None) -> str:
    """
    Standardize group values.

    Args:
        group_value: Original group value
        group_mapping: Optional custom mapping dict, e.g., {0: 'Control', 1: 'Patient'}
        default_group: Default value to use if no group column exists or value is missing

    Returns:
        Standardized group or default value
    """
    if pd.isna(group_value):
        return default_group if default_group is not None else group_value

    # Convert to string and strip whitespace
    group_str = str(group_value).strip()

    # If custom mapping provided, use it first
    if group_mapping:
        # Handle numeric mappings
        try:
            numeric_key = int(float(group_str))
            if numeric_key in group_mapping:
                return group_mapping[numeric_key]
        except (ValueError, TypeError):
            pass

        # Handle string mappings
        if group_str in group_mapping:
            return group_mapping[group_str]

    # Return original value if no mapping found
    return group_str


def standardize_subject_id(subject_id: str) -> str:
    """Standardize subject ID format to BIDS convention (sub-XXX)."""
    cleaned = str(subject_id).strip()

    if cleaned.lower().startswith("sub-"):
        return cleaned
    elif cleaned.lower().startswith("subject_"):
        return cleaned.replace("subject_", "sub-", 1)
    else:
        return f"sub-{cleaned}"
