import asparagus_preprocessing
import logging
import os
import re
from asparagus_preprocessing.paths import get_data_path
from asparagus_preprocessing.utils.loading import load_json
from asparagus_preprocessing.utils.saving import enhanced_save_json
from asparagus_preprocessing.utils.splitting import split
from itertools import repeat
from multiprocessing import Manager, Pool
from time import time
from typing import Any


def simple_recursive_find_and_group_files(
    base_path: str,
    extensions: list[str],
    patterns_exclusion: list[str] = [],
    processes: int = 2,
) -> tuple[list[str], list[str]]:
    """
    Simple version of file finding and grouping that only separates excluded files.
    Args:
        base_path: Root directory to search
        extensions: File extensions to filter by
        patterns_exclusion: Patterns to exclude
        processes: Number of worker processes
    Returns:
        Tuple of 2 lists: (regular, excluded) file paths
    """
    regular, _, _, _, excluded = recursive_find_and_group_files(
        base_path=base_path,
        extensions=extensions,
        patterns_dwi=[],
        patterns_pet=[],
        patterns_perfusion=[],
        patterns_exclusion=patterns_exclusion,
        processes=processes,
    )
    return list(regular), list(excluded)


def recursive_find_and_group_files(
    base_path: str,
    extensions: list[str],
    patterns_dwi: list[str],
    patterns_pet: list[str],
    patterns_perfusion: list[str],
    patterns_exclusion: list[str] = [],
    processes: int = 2,
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    """
    Find and group files by pattern matching across multiple processes.

    Args:
        base_path: Root directory to search
        extensions: File extensions to filter by
        patterns_dwi: Patterns to match DWI files
        patterns_pet: Patterns to match PET files
        patterns_perfusion: Patterns to match perfusion files
        patterns_exclusion: Patterns to exclude
        processes: Number of worker processes

    Returns:
        Tuple of 5 lists: (regular, dwi, pet, perfusion, excluded) file paths
    """

    # 1. Find all files
    all_cases = recursive_find_files(base_path, extensions)
    logging.info(f"Found {len(all_cases)} files with extensions {extensions} in {base_path}")

    # 2. Filter all files in parallel
    time_start = time()
    if processes <= 1:
        regular = []
        dwi = []
        pet = []
        perfusion = []
        excluded = []
        categories = [regular, dwi, pet, perfusion, excluded]
        for file in all_cases:
            filter_files(
                file,
                categories,
                set(patterns_dwi),
                set(patterns_pet),
                set(patterns_perfusion),
                set(patterns_exclusion),
            )
        time_end = time()
        logging.info(f"Filtering took {time_end - time_start:.0f} seconds")
        return regular, dwi, pet, perfusion, excluded

    manager = Manager()
    L = [manager.list(), manager.list(), manager.list(), manager.list(), manager.list()]
    p = Pool(processes)
    p.starmap(
        filter_files,
        zip(
            all_cases,
            repeat(L),
            repeat(set(patterns_dwi)),
            repeat(set(patterns_pet)),
            repeat(set(patterns_perfusion)),
            repeat(set(patterns_exclusion)),
        ),
    )
    p.close()
    p.join()
    time_end = time()
    logging.info(f"Filtering took {time_end - time_start:.0f} seconds")
    return L[0], L[1], L[2], L[3], L[4]  # regular, DWI, PET, perfusion, excluded


def filter_files(
    file: str,
    L: list[Any],
    patterns_dwi: set[str],
    patterns_pet: set[str],
    patterns_perfusion: set[str],
    patterns_exclusion: set[str],
) -> None:
    """
    Filter a single file into appropriate category list based on patterns.

    Args:
        file: File path to filter
        L: List of 5 multiprocessing Manager lists for categorization
        patterns_dwi: Set of DWI patterns
        patterns_pet: Set of PET patterns
        patterns_perfusion: Set of perfusion patterns
        patterns_exclusion: Set of exclusion patterns
    """
    if any(exclusion_pattern in file for exclusion_pattern in patterns_exclusion):
        L[4].append(file)
    elif any(pat1 in file for pat1 in patterns_dwi):
        L[1].append(file)
    elif any(pat2 in file for pat2 in patterns_pet):
        L[2].append(file)
    elif any(pat3 in file for pat3 in patterns_perfusion):
        L[3].append(file)
    else:
        L[0].append(file)


def recursive_find_files(path: str, extensions: list[str]) -> list[str]:
    all_cases = []
    # 1. Build list of all files in the directory recursively.
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                all_cases.append(os.path.join(dirpath, filename))
    return all_cases


def get_bvals_and_bvecs_v1(files, extensions, bval_suffix=".bval", bvec_suffix=".bvec"):
    """
    V1 assumes bvals and bvecs are in the same directory as the DWI files

    Args:
        files: List of DWI file paths
        file_suffix: File suffix to replace (e.g., '.nii.gz')
        bval_suffix: Extension for bval files (default: '.bval', can use '.bvals')
        bvec_suffix: Extension for bvec files (default: '.bvec', can use '.bvecs')
    """
    bvals_out = []
    bvecs_out = []
    for f in files:
        for ext in extensions:
            if f.endswith(ext):
                bvals_out.append(f.replace(ext, bval_suffix))
                bvecs_out.append(f.replace(ext, bvec_suffix))
                break
        else:
            raise ValueError(f"File {f} does not end with any of the specified extensions: {extensions}")
    return bvals_out, bvecs_out


def get_bvals_and_bvecs_v2(files, bval_file, bvec_file):
    """
    V2 assumes all files share the same bval and bvec files.
    """
    bvals_out = [bval_file for f in files]
    bvecs_out = [bvec_file for f in files]
    return bvals_out, bvecs_out


def get_bvals_and_bvecs_v3(files, extensions):
    """
    V3 assumes bvals and bvecs are in separate 'bval' and 'bvec' folders
    at the same level as the nifti files directory

    Directory structure expected:
    parent_dir/
    ├── nifti/
    │   ├── file1.nii.gz
    │   └── file2.nii.gz
    ├── bval/
    │   ├── file1.bval
    │   └── file2.bval
    └── bvec/
        ├── file1.bvec
        └── file2.bvec
    """
    bvals_out = []
    bvecs_out = []

    for f in files:
        # Get the directory containing the nifti file
        nifti_dir = os.path.dirname(f)
        # Go up one level to get parent directory
        parent_dir = os.path.dirname(nifti_dir)
        # Get filename without the suffix
        filename = os.path.basename(f)
        for ext in extensions:
            if f.endswith(ext):
                filename_no_ext = filename.replace(ext, "")
                break
        # Construct paths to bval and bvec files
        bval_path = os.path.join(parent_dir, "bval", filename_no_ext + ".bval")
        bvec_path = os.path.join(parent_dir, "bvec", filename_no_ext + ".bvec")

        bvals_out.append(bval_path)
        bvecs_out.append(bvec_path)

    return bvals_out, bvecs_out


def get_bvals_and_bvecs_v4(files):
    """
    V4 assumes bvals.txt and bvecs.txt are in parent directory

    Directory structure:
    parent_dir/
    ├── bvals.txt
    ├── bvecs.txt
    └── mri/
        └── diff.nii.gz
    """
    bvals_out = []
    bvecs_out = []

    for f in files:
        parent_dir = os.path.dirname(os.path.dirname(f))
        bvals_out.append(os.path.join(parent_dir, "bvals.txt"))
        bvecs_out.append(os.path.join(parent_dir, "bvecs.txt"))

    return bvals_out, bvecs_out


def find_processed_dataset(dataset_ID):
    dataset_ID = str(dataset_ID)
    datasets = os.listdir(get_data_path())
    if dataset_ID in datasets:
        return dataset_ID
    for dataset in datasets:
        if dataset.startswith(dataset_ID + "_"):
            return dataset

    raise LookupError(f"Dataset {dataset_ID} not found in {get_data_path()}. These are valid datasets: {datasets}")


def update_paths(dataset_dir):
    dataset_json_path = os.path.join(dataset_dir, "dataset.json")
    if not os.path.exists(dataset_json_path):
        logging.warning(f"No dataset.json found in {dataset_dir}; skipping update_paths.")
        return

    dataset_cfg = load_json(dataset_json_path)

    if os.path.exists(os.path.join(dataset_dir, "paths.json")):
        paths_path = os.path.join(dataset_dir, "paths.json")
        paths_permissions = os.stat(paths_path).st_mode
        old_paths = load_json(os.path.join(dataset_dir, "paths.json"))
    else:
        paths_permissions = 0o777
        old_paths = []

    extension = ".pt" if dataset_cfg["saving_config"]["save_as_tensor"] else ".nii.gz"
    target_all_files = recursive_find_files(dataset_dir, extensions=extension)

    if len(extension) == 0 or len(target_all_files) == 0:
        source_total = dataset_cfg.get("metadata", {}).get("files_source_directory_total")
        if source_total == 0:
            logging.info(f"No source files recorded for {dataset_dir}; skipping update_paths for empty dataset.")
            return
        logging.warning(
            f"No files found in target directory {dataset_dir} with extension {extension}. If this is a dataset collection you need to rerun its preprocessing script"  # noqa: E501
        )
        return
    if len(old_paths) == 0:
        logging.warning(
            f"No old paths provided, can not verify that the {len(target_all_files)} files found correspond to the old number of files"  # noqa: E501
        )
    if len(target_all_files) != len(old_paths):
        logging.warning(
            f"Number of files in target directory {len(target_all_files)} does not match number of old paths {len(old_paths)}"
        )

    enhanced_save_json(
        old_paths,
        os.path.join(dataset_dir, "paths_backup.json"),
        permissions=paths_permissions,
    )
    enhanced_save_json(target_all_files, os.path.join(dataset_dir, "paths.json"))

    split_name = dataset_cfg["dataset_config"].get("split")
    if split_name is not None:
        split_fn = getattr(asparagus_preprocessing.utils.splitting, split_name)
        split(
            files=target_all_files,
            fn=split_fn,
            save_path=os.path.join(dataset_dir, split_name + ".json"),
        )


def find_and_add_train_splits(dataset_dir, all_train_splits):
    regex = re.compile(r"^.*split.*\.json$", re.IGNORECASE)
    matching_files = [filename for filename in os.listdir(dataset_dir) if regex.match(filename)]
    assert len(matching_files) == 1, f"Expected exactly one split JSON file in {dataset_dir}, but found {len(matching_files)}."
    train_splits = load_json(os.path.join(dataset_dir, matching_files[0]))

    for i in range(5):
        all_train_splits[i]["train"] += train_splits[i]["train"]
        all_train_splits[i]["val"] += train_splits[i]["val"]

    return all_train_splits


def find_and_add_test_splits(dataset_dir, all_test_splits):
    regex = re.compile(r"^.*TEST.*\.json$", re.IGNORECASE)
    matching_files = [filename for filename in os.listdir(dataset_dir) if regex.match(filename)]
    assert len(matching_files) == 1, f"Expected exactly one split JSON file in {dataset_dir}, but found {len(matching_files)}."

    test_splits = load_json(os.path.join(dataset_dir, matching_files[0]))
    all_test_splits += test_splits

    return all_test_splits
