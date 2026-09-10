import logging
import numpy as np
import os
import pandas as pd
import re
import shutil
from .config import (
    DEFAULT_MRI_METADATA_FIELDS,
    EXTENSION_LENGTHS,
    SUPPORTED_FILE_EXTENSIONS,
)
from .extractors import (
    detect_modality_info,
    extract_json_metadata,
    extract_session_from_path,
    extract_subject_id_from_path,
    find_columns,
    get_json_path_for_image,
    match_numeric_subject_ids,
    sessions_match,
)
from .standardizers import (
    standardize_group,
    standardize_handedness,
    standardize_sex,
    standardize_subject_id,
)
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def load_demographics(
    demographics_csv_path: str,
) -> Tuple[Optional[pd.DataFrame], bool]:
    """Load demographics file and return DataFrame and success flag."""
    if not os.path.exists(demographics_csv_path):
        logging.warning(f"Demographics file not found: {demographics_csv_path}")
        return None, False

    try:
        if demographics_csv_path.endswith((".xlsx", ".xls")):
            demo_df = pd.read_excel(demographics_csv_path)
        else:
            separator = "\t" if demographics_csv_path.endswith(".tsv") else ","
            try:
                demo_df = pd.read_csv(demographics_csv_path, sep=separator)
            except pd.errors.ParserError:
                # Try semicolon if comma fails
                demo_df = pd.read_csv(demographics_csv_path, sep=";")

        if len(demo_df) == 0:
            logging.warning("Demographics file is empty")
            return None, False

        logging.info(f"Loaded {len(demo_df)} subjects from demographics")
        return demo_df, True

    except Exception as e:
        logging.error(f"Error loading demographics: {e}")
        return None, False


def create_subject_session_mapping(
    processed_ids: set, subject_sessions: Dict[str, set]
) -> Tuple[Dict[str, str], Dict[Tuple[str, str], str]]:
    """Create consistent subject and session mappings with zero-padding (min 2 digits)."""
    unique_subjects = sorted(processed_ids)

    # Always at least 2 digits, expands if needed
    num_subject_digits = max(2, len(str(len(unique_subjects))))
    subject_mapping = {old_id: f"sub-{i + 1:0{num_subject_digits}d}" for i, old_id in enumerate(unique_subjects)}

    session_mapping = {}
    for original_id, sessions in subject_sessions.items():
        sorted_sessions = sorted(sessions)

        # Always at least 2 digits, expands if needed
        num_session_digits = max(2, len(str(len(sorted_sessions))))
        for i, session in enumerate(sorted_sessions):
            session_mapping[(original_id, session)] = f"ses-{i + 1:0{num_session_digits}d}"

    return subject_mapping, session_mapping


def _extract_demographic_sessions(
    subject_demo_rows: pd.DataFrame,
) -> Tuple[Dict[str, pd.Series], Optional[pd.Series]]:
    """
    Extract available demographic sessions for a subject.

    Returns:
        Tuple of (available_demo_sessions dict, nan_demo_row or None)
    """
    available_demo_sessions = {}
    nan_demo_row = None

    for _, demo_row in subject_demo_rows.iterrows():
        if "session_id" in demo_row:
            if pd.notna(demo_row["session_id"]):
                demo_session = str(demo_row["session_id"])
                available_demo_sessions[demo_session] = demo_row
            else:
                # Session is NaN - store for later assignment to unmatched sessions
                nan_demo_row = demo_row
        else:
            # No session column exists - use this row for all sessions
            nan_demo_row = demo_row
            break

    return available_demo_sessions, nan_demo_row


def _match_file_sessions_to_demographics(
    file_sessions: List[str],
    available_demo_sessions: Dict[str, pd.Series],
    nan_demo_row: Optional[pd.Series],
) -> Dict[str, pd.Series]:
    """
    Match file sessions to demographic sessions using fuzzy matching.

    Args:
        file_sessions: List of session IDs extracted from files
        available_demo_sessions: Dict mapping demo session IDs to demographic rows
        nan_demo_row: Demographic row with NaN/missing session (or None)

    Returns:
        Dictionary mapping file session -> demographic row
    """
    file_to_demo_mapping = {}

    # First pass: match file sessions to available demographic sessions
    for file_session in file_sessions:
        if available_demo_sessions:
            for demo_session, demo_row in available_demo_sessions.items():
                if sessions_match(file_session, demo_session):
                    file_to_demo_mapping[file_session] = demo_row
                    logging.debug(f"Matched file session '{file_session}' to demo session '{demo_session}'")
                    break

    # Second pass: assign NaN session row to ALL unmatched file sessions
    # This is intentional - if demographics has a row with no session info,
    # it applies to all file sessions that didn't match a specific demo session
    if nan_demo_row is not None:
        for file_session in file_sessions:
            if file_session not in file_to_demo_mapping:
                file_to_demo_mapping[file_session] = nan_demo_row
                logging.debug(f"Assigned NaN demographic row to unmatched file session '{file_session}'")

    return file_to_demo_mapping


def _create_row_without_demographics(
    subject_mapping: Dict[str, str],
    original_id: str,
    session_number: int,
    original_session: str,
    default_group: Optional[str],
    columns_to_keep: List[str],
) -> Dict:
    """Create a minimal row for sessions without demographic data."""
    new_row = {
        "participant_id": subject_mapping[original_id],
        "session_id": f"ses-{session_number:02d}",
        "original_participant_id": original_id,
        "original_session_id": original_session,
    }

    if default_group is not None and "group" in columns_to_keep:
        new_row["group"] = default_group

    return new_row


def create_expanded_dataframe(
    matched_df: Optional[pd.DataFrame],
    processed_ids: set,
    subject_sessions: Dict[str, set],
    subject_mapping: Dict[str, str],
    session_mapping: Dict[Tuple[str, str], str],
    columns_to_keep: List[str],
    default_group: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create expanded DataFrame with all subject-session combinations.

    This function matches file sessions to demographic data (if available) and creates
    a complete DataFrame with one row per subject-session combination.
    """
    expanded_rows = []

    if matched_df is None:
        # No demographics file - create minimal rows from processed files only
        for original_id in processed_ids:
            original_sessions = subject_sessions.get(original_id, {"ses-01"})
            for original_session in sorted(original_sessions):
                new_row = {
                    "participant_id": subject_mapping[original_id],
                    "session_id": session_mapping.get((original_id, original_session), "ses-01"),
                    "original_participant_id": original_id,
                    "original_session_id": original_session,
                }
                if default_group is not None and "group" in columns_to_keep:
                    new_row["group"] = default_group
                expanded_rows.append(new_row)

    else:
        # Demographics file exists - match file sessions to demographics
        for original_id in processed_ids:
            original_sessions = sorted(subject_sessions.get(original_id, {"ses-01"}))
            subject_demo_rows = matched_df[matched_df["original_participant_id"] == original_id]

            if not subject_demo_rows.empty:
                # Extract available demographic sessions
                available_demo_sessions, nan_demo_row = _extract_demographic_sessions(subject_demo_rows)

                # Match file sessions to demographic sessions
                file_to_demo_mapping = _match_file_sessions_to_demographics(
                    original_sessions, available_demo_sessions, nan_demo_row
                )

                # Create rows with sequential session numbering
                session_counter = 1
                for file_session in original_sessions:
                    if file_session in file_to_demo_mapping:
                        # Use matched demographic data
                        demo_row = file_to_demo_mapping[file_session]
                        new_row = demo_row.copy()
                        new_row["participant_id"] = subject_mapping[original_id]
                        new_row["session_id"] = f"ses-{session_counter:02d}"
                        new_row["original_session_id"] = file_session
                    else:
                        # No demographic data for this session
                        new_row = _create_row_without_demographics(
                            subject_mapping,
                            original_id,
                            session_counter,
                            file_session,
                            default_group,
                            columns_to_keep,
                        )

                    session_counter += 1
                    expanded_rows.append(new_row)

            else:
                # No demographics for this subject at all
                for session_index, original_session in enumerate(original_sessions):
                    new_row = _create_row_without_demographics(
                        subject_mapping,
                        original_id,
                        session_index + 1,
                        original_session,
                        default_group,
                        columns_to_keep,
                    )
                    expanded_rows.append(new_row)

    # Create DataFrames - keep the to_dict() check for Series objects
    expanded_df = pd.DataFrame([row.to_dict() if hasattr(row, "to_dict") else row for row in expanded_rows])

    # Create display DataFrame with selected columns
    available_columns = [col for col in columns_to_keep if col in expanded_df.columns]
    if not available_columns:
        available_columns = ["participant_id", "session_id"]

    display_df = expanded_df[available_columns].copy()

    # Sort both DataFrames consistently
    sort_columns = ["participant_id"]
    if "session_id" in display_df.columns:
        sort_columns.append("session_id")

    display_df = display_df.sort_values(sort_columns).reset_index(drop=True)
    expanded_df = expanded_df.sort_values(sort_columns).reset_index(drop=True)

    return expanded_df, display_df


def generate_clean_BIDS_filename(
    subject_id: str,
    session_id: str,
    original_filename: str,
    existing_files: set = None,
    modality_patterns: dict = None,
) -> str:
    """Generate clean, standardized BIDS filename with subject and session."""
    if existing_files is None:
        existing_files = set()

    # Check for supported extensions using config
    ext = None
    base = None
    for extension, length in EXTENSION_LENGTHS.items():
        if original_filename.endswith(extension):
            ext = extension
            base = original_filename[:-length]
            break

    if ext is None:
        return original_filename  # Unsupported extension

    # Use the configurable modality detection
    suffix, _ = detect_modality_info(original_filename, modality_patterns)

    # Handle special cases like bval and trace
    base_lower = base.lower()
    bval_match = re.search(r"bval[_-]?(\d+)", base_lower)
    if bval_match:
        suffix += f"_bval{bval_match.group(1)}"

    if "trace" in base_lower and suffix == "dwi":
        suffix += "_trace"

    base_filename = f"{subject_id}_{session_id}_{suffix}{ext}"

    # Handle filename conflicts with BIDS run- format
    if base_filename in existing_files:
        run_number = 1
        while f"{subject_id}_{session_id}_run-{run_number}_{suffix}{ext}" in existing_files:
            run_number += 1
        base_filename = f"{subject_id}_{session_id}_run-{run_number}_{suffix}{ext}"

    return base_filename


def remove_empty_dirs(path: str, source_dir: str) -> None:
    """Remove empty directories."""
    try:
        if os.path.exists(path) and os.path.isdir(path) and path != source_dir:
            if not os.listdir(path):
                os.rmdir(path)
                logging.debug(f"Removed empty directory: {path}")
    except (OSError, PermissionError) as e:
        logging.warning(f"Could not remove directory {path}: {e}")


def create_mri_info_dataframe(
    expanded_df: pd.DataFrame,
    file_mapping: Dict[Tuple[str, str], List[str]],
    common_base: str,
    custom_json_mapping: dict = None,
    hardcoded_metadata: dict = None,
    source_dir: str = None,
    target_dir: str = None,
) -> pd.DataFrame:
    """Create MRI info dataframe with metadata from JSON files or hardcoded metadata."""
    mri_info_rows = []

    for _, row in expanded_df.iterrows():
        subject_id = row["participant_id"]
        session_id = row["session_id"]
        original_subject_id = row["original_participant_id"]
        original_session_id = row["original_session_id"]

        key = (original_subject_id, original_session_id)
        if key in file_mapping:
            file_paths = file_mapping[key]

            # Group by modality
            modality_groups = defaultdict(list)
            for file_path in file_paths:
                original_filename = os.path.basename(file_path)
                _, modality = detect_modality_info(original_filename)
                modality_groups[modality].append((file_path, original_filename))

            # Process each modality group
            for modality, mod_files in modality_groups.items():
                # Group files by suffix+extension to detect duplicates FIRST
                suffix_groups = defaultdict(list)

                for file_path, original_filename in mod_files:
                    # Get extension
                    ext = None
                    base = None
                    for extension, length in EXTENSION_LENGTHS.items():
                        if original_filename.endswith(extension):
                            ext = extension
                            base = original_filename[:-length]
                            break

                    if ext is None:
                        continue

                    # Get suffix
                    suffix, _ = detect_modality_info(original_filename)

                    # Handle special cases
                    base_lower = base.lower()
                    bval_match = re.search(r"bval[_-]?(\d+)", base_lower)
                    if bval_match:
                        suffix += f"_bval{bval_match.group(1)}"
                    if "trace" in base_lower and suffix == "dwi":
                        suffix += "_trace"

                    suffix_key = (suffix, ext)
                    suffix_groups[suffix_key].append((file_path, original_filename))

                # Assign filenames with BIDS-compliant run numbers
                for (suffix, ext), path_file_list in suffix_groups.items():
                    if len(path_file_list) == 1:
                        # Single file - no run number
                        file_path, original_filename = path_file_list[0]
                        clean_filename = f"{subject_id}_{session_id}_{suffix}{ext}"
                        rel_filename = os.path.join(subject_id, session_id, modality, clean_filename)

                        # Get metadata
                        if source_dir and target_dir:
                            source_file_path = file_path.replace(target_dir, source_dir, 1)
                            json_path = get_json_path_for_image(source_file_path, custom_json_mapping)
                        else:
                            json_path = get_json_path_for_image(file_path, custom_json_mapping)

                        metadata = extract_json_metadata(json_path, hardcoded_metadata, original_filename, file_path)

                        mri_row = {
                            "participant_id": subject_id,
                            "session_id": session_id,
                            "filename": rel_filename,
                            **metadata,
                        }
                        mri_info_rows.append(mri_row)
                    else:
                        # Multiple files - assign run-1, run-2, etc.
                        for run_num, (file_path, original_filename) in enumerate(path_file_list, start=1):
                            clean_filename = f"{subject_id}_{session_id}_run-{run_num}_{suffix}{ext}"
                            rel_filename = os.path.join(subject_id, session_id, modality, clean_filename)

                            # Get metadata
                            if source_dir and target_dir:
                                source_file_path = file_path.replace(target_dir, source_dir, 1)
                                json_path = get_json_path_for_image(source_file_path, custom_json_mapping)
                            else:
                                json_path = get_json_path_for_image(file_path, custom_json_mapping)

                            metadata = extract_json_metadata(
                                json_path,
                                hardcoded_metadata,
                                original_filename,
                                file_path,
                            )

                            mri_row = {
                                "participant_id": subject_id,
                                "session_id": session_id,
                                "filename": rel_filename,
                                **metadata,
                            }
                            mri_info_rows.append(mri_row)

    if mri_info_rows:
        mri_df = pd.DataFrame(mri_info_rows)
        mri_df = mri_df.sort_values(["participant_id", "session_id", "filename"]).reset_index(drop=True)
        return mri_df
    else:
        columns = [
            "participant_id",
            "session_id",
            "filename",
        ] + DEFAULT_MRI_METADATA_FIELDS
        return pd.DataFrame(columns=columns)


def extract_demographics(
    processed_files: List[str],
    demographics_csv_path: str,
    columns_to_keep: List[str] = ["participant_id", "age", "sex"],
    custom_patterns: List[str] = None,
    modality_patterns: List[str] = None,
    sex_mapping: dict = None,
    handedness_mapping: dict = None,
    group_mapping: dict = None,
    default_group: str = None,
    custom_json_mapping: dict = None,
    hardcoded_metadata: dict = None,
    source_dir: str = None,
    target_dir: str = None,
) -> Optional[pd.DataFrame]:
    """Extract and standardize demographics for processed subjects."""

    if not processed_files:
        logging.warning("No processed files")
        return None

    processed_ids = set()
    subject_sessions = {}
    file_mapping = {}  # Store file paths directly

    for file_path in processed_files:
        original_id = extract_subject_id_from_path(file_path, custom_patterns)
        session_name = extract_session_from_path(file_path, custom_patterns)
        processed_ids.add(original_id)

        if original_id not in subject_sessions:
            subject_sessions[original_id] = set()
        subject_sessions[original_id].add(session_name)

        # Store the file path mapping directly here
        key = (original_id, session_name)
        if key not in file_mapping:
            file_mapping[key] = []
        file_mapping[key].append(file_path)

    logging.info(f"Extracted {len(processed_ids)} unique subject IDs")

    demo_df, use_demographics = load_demographics(demographics_csv_path)
    matched_df = None

    if use_demographics:
        found_cols = find_columns(demo_df)
        logging.info(f"Column mapping: {found_cols}")

        if "participant_id" not in found_cols:
            logging.warning("No participant_id column found, creating participant_id from file patterns only")
        else:
            demo_df["participant_id_standardized"] = demo_df[found_cols["participant_id"]].apply(standardize_subject_id)

            # Auto-match subject ID formats
            demo_subjects = set(demo_df["participant_id_standardized"])
            matched_processed_ids = match_numeric_subject_ids(demo_subjects, processed_ids)

            # Update subject_sessions and file_mapping to use matched IDs
            matched_subject_sessions = {}
            matched_file_mapping = {}
            for old_id, sessions in subject_sessions.items():
                matched_id = match_numeric_subject_ids(demo_subjects, {old_id}).__iter__().__next__()
                matched_subject_sessions[matched_id] = sessions

                # Also update file_mapping
                for session in sessions:
                    old_key = (old_id, session)
                    new_key = (matched_id, session)
                    if old_key in file_mapping:
                        matched_file_mapping[new_key] = file_mapping[old_key]

            processed_ids = matched_processed_ids
            subject_sessions = matched_subject_sessions
            file_mapping = matched_file_mapping

            matched_df = demo_df[demo_df["participant_id_standardized"].isin(processed_ids)].copy()

            missing_subjects = processed_ids - set(matched_df["participant_id_standardized"])
            if missing_subjects:
                empty_row_template = {
                    col: (None if col in [found_cols["participant_id"], "participant_id_standardized"] else np.nan)
                    for col in demo_df.columns
                }

                missing_rows = []
                for missing_id in missing_subjects:
                    row = empty_row_template.copy()
                    row[found_cols["participant_id"]] = missing_id
                    row["participant_id_standardized"] = missing_id
                    missing_rows.append(row)

                missing_df = pd.DataFrame(missing_rows)
                matched_df = pd.concat([matched_df, missing_df], ignore_index=True)

            rename_map = {found_cols[col]: col for col in found_cols if col in columns_to_keep and col != "participant_id"}
            matched_df = matched_df.rename(columns=rename_map)

            if "sex" in matched_df.columns:
                matched_df["sex"] = matched_df["sex"].apply(lambda x: standardize_sex(x, sex_mapping))
                logging.info("Standardized sex column")

            if "handedness" in matched_df.columns:
                matched_df["handedness"] = matched_df["handedness"].apply(
                    lambda x: standardize_handedness(x, handedness_mapping)
                )
                logging.info("Standardized handedness column")

            if "group" in matched_df.columns:
                matched_df["group"] = matched_df["group"].apply(lambda x: standardize_group(x, group_mapping, default_group))
                logging.info("Standardized group column")

            matched_df["original_participant_id"] = matched_df["participant_id_standardized"].copy()

            logging.info(f"Matched {len(matched_df)}/{len(processed_ids)} subjects")

    if default_group is not None and "group" in columns_to_keep:
        if matched_df is None:
            logging.info(f"No demographics file - will add default group: {default_group}")
        else:
            if "group" not in matched_df.columns:
                matched_df["group"] = default_group
                logging.info(f"Added group column with default value: {default_group}")

    subject_mapping, session_mapping = create_subject_session_mapping(processed_ids, subject_sessions)

    expanded_df, display_df = create_expanded_dataframe(
        matched_df,
        processed_ids,
        subject_sessions,
        subject_mapping,
        session_mapping,
        columns_to_keep,
        default_group,
    )

    if "participant_id" in display_df.columns:
        sort_columns = ["participant_id"]
        if "session_id" in display_df.columns:
            sort_columns.append("session_id")
        display_df = display_df.sort_values(sort_columns)

    # Add filenames to expanded_df using pre-built file mapping
    if processed_files:
        common_base = os.path.commonpath(processed_files)

        # Assign files to expanded_df rows using pre-built mapping
        for i, row in expanded_df.iterrows():
            key = (row["original_participant_id"], row["original_session_id"])
            if key in file_mapping:
                rel_paths = [os.path.relpath(file_path, common_base) for file_path in file_mapping[key]]
                expanded_df.at[i, "filenames"] = ",".join(rel_paths)

        mri_info_df = create_mri_info_dataframe(
            expanded_df,
            file_mapping,
            common_base,
            custom_json_mapping,
            hardcoded_metadata,
            source_dir,
            target_dir,
        )

        return expanded_df, display_df, mri_info_df
    else:
        # Create empty MRI info dataframe when no processed files
        columns = [
            "participant_id",
            "session_id",
            "filename",
        ] + DEFAULT_MRI_METADATA_FIELDS
        mri_info_df = pd.DataFrame(columns=columns)

        return expanded_df, display_df, mri_info_df


def rename_files_with_mapping(
    target_dir: str,
    expanded_df: pd.DataFrame,
    modality_patterns: dict = None,
) -> pd.DataFrame:
    """Reorganize files into BIDS structure using the provided expanded_df mapping."""
    import uuid

    mapping_data = []
    old_directories = set()

    # Create file lookup from expanded_df
    file_lookup = {}
    for _, row in expanded_df.iterrows():
        if pd.notna(row.get("filenames")):
            filepaths = row["filenames"].split(",")
            for filepath in filepaths:
                filepath = filepath.strip()
                file_lookup[filepath] = {
                    "new_participant_id": row["participant_id"],
                    "new_session_id": row["session_id"],
                }

    # Collect all files using os.walk
    all_file_paths = []
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if not file.endswith(SUPPORTED_FILE_EXTENSIONS):
                continue
            old_path = os.path.join(root, file)
            all_file_paths.append(old_path)
            old_directories.add(os.path.dirname(old_path))

    if not all_file_paths:
        logging.warning("No files found to process")
        return pd.DataFrame()

    common_base = os.path.commonpath(all_file_paths)
    files_found = len(all_file_paths)

    # STEP 1: Move all files to temporary names to avoid collisions
    temp_mapping = {}  # old_path -> temp_path

    for old_path in all_file_paths:
        file = os.path.basename(old_path)
        rel_path = os.path.relpath(old_path, common_base)

        if rel_path not in file_lookup:
            continue  # Skip files not in our mapping

        # Move to temp name
        temp_name = f".temp_{uuid.uuid4().hex}{os.path.splitext(old_path)[1]}"
        temp_path = os.path.join(os.path.dirname(old_path), temp_name)

        try:
            shutil.move(old_path, temp_path)
            temp_mapping[old_path] = (temp_path, rel_path)
        except Exception as e:
            logging.error(f"Error moving {old_path} to temp: {e}")
            continue

    # STEP 2: Group temp files by subject/session/modality
    subject_session_modality_files = defaultdict(list)

    for old_path, (temp_path, rel_path) in temp_mapping.items():
        file = os.path.basename(old_path)
        mapping = file_lookup[rel_path]
        new_subject_id = mapping["new_participant_id"]
        new_session_id = mapping["new_session_id"]

        _, modality = detect_modality_info(file, modality_patterns)

        key = (new_subject_id, new_session_id, modality)
        subject_session_modality_files[key].append((temp_path, old_path, file))

    # STEP 3: Organize files with BIDS-compliant run numbers
    files_organized = 0
    for (
        subject_id,
        session_id,
        modality,
    ), file_list in subject_session_modality_files.items():
        modality_dir = os.path.join(target_dir, subject_id, session_id, modality)
        os.makedirs(modality_dir, exist_ok=True)

        # Group by suffix+extension
        suffix_groups = defaultdict(list)
        for temp_path, old_path, file in file_list:
            ext = None
            base = None
            for extension, length in EXTENSION_LENGTHS.items():
                if file.endswith(extension):
                    ext = extension
                    base = file[:-length]
                    break

            if ext is None:
                continue

            suffix, _ = detect_modality_info(file, modality_patterns)

            base_lower = base.lower()
            bval_match = re.search(r"bval[_-]?(\d+)", base_lower)
            if bval_match:
                suffix += f"_bval{bval_match.group(1)}"
            if "trace" in base_lower and suffix == "dwi":
                suffix += "_trace"

            key = (suffix, ext)
            suffix_groups[key].append((temp_path, old_path, file))

        # Assign filenames with run numbers
        for (suffix, ext), path_file_list in suffix_groups.items():
            if len(path_file_list) == 1:
                temp_path, old_path, old_filename = path_file_list[0]
                clean_filename = f"{subject_id}_{session_id}_{suffix}{ext}"
                new_path = os.path.join(modality_dir, clean_filename)
                old_rel_path = os.path.relpath(old_path, target_dir)
                new_rel_path = os.path.relpath(new_path, target_dir)

                try:
                    shutil.move(temp_path, new_path)
                    files_organized += 1
                except Exception as e:
                    logging.error(f"Error moving {temp_path} to {new_path}: {e}")
                    continue

                mapping_data.append(
                    {
                        "old_path": old_rel_path,
                        "new_path": new_rel_path,
                        "old_filename": old_filename,
                        "new_filename": clean_filename,
                        "participant_id": subject_id,
                        "session_id": session_id,
                        "modality": modality,
                    }
                )
            else:
                for run_num, (temp_path, old_path, old_filename) in enumerate(path_file_list, start=1):
                    clean_filename = f"{subject_id}_{session_id}_run-{run_num}_{suffix}{ext}"
                    new_path = os.path.join(modality_dir, clean_filename)
                    old_rel_path = os.path.relpath(old_path, target_dir)
                    new_rel_path = os.path.relpath(new_path, target_dir)

                    try:
                        shutil.move(temp_path, new_path)
                        files_organized += 1
                    except Exception as e:
                        logging.error(f"Error moving {temp_path} to {new_path}: {e}")
                        continue

                    mapping_data.append(
                        {
                            "old_path": old_rel_path,
                            "new_path": new_rel_path,
                            "old_filename": old_filename,
                            "new_filename": clean_filename,
                            "participant_id": subject_id,
                            "session_id": session_id,
                            "modality": modality,
                        }
                    )

    # Remove empty directories
    all_paths_to_check = set()
    for old_dir in old_directories:
        current_path = old_dir
        while current_path != target_dir and current_path != os.path.dirname(current_path):
            all_paths_to_check.add(current_path)
            current_path = os.path.dirname(current_path)

    for path_to_check in sorted(all_paths_to_check, key=lambda x: x.count(os.sep), reverse=True):
        remove_empty_dirs(path_to_check, target_dir)

    logging.info(f"Found {files_found} files, organized {files_organized} files")

    mapping_df = pd.DataFrame(mapping_data)
    if not mapping_df.empty:
        mapping_path = os.path.join(target_dir, "mapping.tsv")
        mapping_df.to_csv(mapping_path, sep="\t", index=False)
        logging.info(f"Saved mapping to {mapping_path}")

    return mapping_df
