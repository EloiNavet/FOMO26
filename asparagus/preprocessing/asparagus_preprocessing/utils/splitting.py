import logging
import re
from asparagus_preprocessing.utils.saving import enhanced_save_json
from collections import defaultdict
from sklearn.model_selection import train_test_split


def split_40_10_50(files: list, test: bool = False, seed_increment: int = 0):
    return non_stratified_split(files, 0.40, 0.10, 0.50, test, seed_increment, base_seed=28300211)


def BIDSsplit_40_10_50(files: list, test: bool = False, seed_increment: int = 0):
    sub_pattern = r"/sub-\d+/"
    return stratified_split(files, 0.40, 0.10, 0.50, test, sub_pattern, seed_increment, base_seed=283123111)


def ABVIBsplit_40_10_50(files: list, test: bool = False, seed_increment: int = 0):
    sub_pattern = r"ABVIB/\d+/"
    return stratified_split(files, 0.40, 0.10, 0.50, test, sub_pattern, seed_increment, base_seed=283123111)


def PatientIDsplit_40_10_50(files: list, test: bool = False, seed_increment: int = 0):
    sub_pattern = r"/PatientID_[0-9]+/"
    return stratified_split(files, 0.40, 0.10, 0.50, test, sub_pattern, seed_increment, base_seed=283123111)


def MCSAsplit_40_10_50(files: list, test: bool = False, seed_increment: int = 0):
    sub_pattern = r"/MCSA_\d+/"
    return stratified_split(files, 0.40, 0.10, 0.50, test, sub_pattern, seed_increment, base_seed=283123111)


def UCSDsplit_40_10_50(files: list, test: bool = False, seed_increment: int = 0):
    sub_pattern = r"/UCSD-PTGBM-\d+_"
    return stratified_split(files, 0.40, 0.10, 0.50, test, sub_pattern, seed_increment, base_seed=283123111)


def non_stratified_split(
    files: list,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    test: bool = False,
    seed_increment: int = 0,
    base_seed: int = 0,
):
    assert train_ratio + val_ratio + test_ratio == 1.0, (
        "Train, validation, and test ratios must sum to 1, but got {}, {}, {}".format(train_ratio, val_ratio, test_ratio)
    )

    files = sorted(files)

    if test:
        if test_ratio == 0.0:
            return files, []
        return train_test_split(files, test_size=test_ratio, random_state=base_seed)

    return train_test_split(
        files,
        test_size=val_ratio / (val_ratio + train_ratio),
        random_state=base_seed + seed_increment + 1,
    )


def split_group_key(file: str, group_by: str) -> str:
    if group_by == "file":
        return file

    parts = [part for part in file.replace("\\", "/").split("/") if part]
    dataset_idx = next((idx for idx, part in enumerate(parts) if re.match(r"^(PT|SEG|CLS|REG)\d{3}", part)), None)
    subject_idx = next((idx for idx, part in enumerate(parts) if part.startswith("sub-")), None)
    dataset = parts[dataset_idx] if dataset_idx is not None else None
    subject = parts[subject_idx] if subject_idx is not None else None
    session = next((part for part in parts if part.startswith("ses-")), None)
    if subject is None:
        subject_match = re.search(r"sub-[A-Za-z0-9]+", file)
        subject = subject_match.group(0) if subject_match else None
    if session is None:
        session_match = re.search(r"ses-[A-Za-z0-9]+", file)
        session = session_match.group(0) if session_match else None

    if subject is None:
        raise ValueError(f"Could not extract subject id from path: {file}")

    if dataset_idx is not None and subject_idx is not None and dataset_idx < subject_idx:
        prefix = "/".join(parts[dataset_idx : subject_idx + 1])
    else:
        prefix = f"{dataset or 'unknown_dataset'}/{subject}"
    if group_by == "subject":
        return prefix
    if group_by == "session":
        return f"{prefix}/{session or 'no-session'}"
    raise ValueError(f"Unsupported group_by value: {group_by}")


def grouped_non_stratified_split(
    files: list,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    group_by: str,
    test: bool = False,
    seed_increment: int = 0,
    base_seed: int = 0,
):
    assert train_ratio + val_ratio + test_ratio == 1.0, (
        "Train, validation, and test ratios must sum to 1, but got {}, {}, {}".format(train_ratio, val_ratio, test_ratio)
    )

    files = sorted(files)
    group_to_files = defaultdict(list)
    for file in files:
        group_to_files[split_group_key(file, group_by)].append(file)
    groups = sorted(group_to_files)

    if test:
        if test_ratio == 0.0:
            return files, []
        train_groups, test_groups = train_test_split(groups, test_size=test_ratio, random_state=base_seed)
        train_groups = set(train_groups)
        test_groups = set(test_groups)
        return (
            [file for group in groups if group in train_groups for file in group_to_files[group]],
            [file for group in groups if group in test_groups for file in group_to_files[group]],
        )

    if val_ratio == 0.0:
        return files, []
    train_groups, val_groups = train_test_split(
        groups,
        test_size=val_ratio / (val_ratio + train_ratio),
        random_state=base_seed + seed_increment + 1,
    )
    train_groups = set(train_groups)
    val_groups = set(val_groups)
    return (
        [file for group in groups if group in train_groups for file in group_to_files[group]],
        [file for group in groups if group in val_groups for file in group_to_files[group]],
    )


def stratified_split(
    files: list,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    test: bool,
    pattern: str,
    seed_increment: int,
    base_seed: int,
):
    """Subject-level split that assigns files by substring containment of the matched id.

    .. deprecated::
        Files are assigned with ``any(subject in file ...)``, which is fragile: an
        id that is a substring of another (e.g. "sub-1" vs "sub-10") can leak across
        splits unless the ``pattern`` embeds delimiters. Prefer
        :func:`grouped_non_stratified_split` (group_by="subject"/"session"), which
        uses exact group membership via :func:`split_group_key`.
    """
    logging.warning(
        "stratified_split uses substring file matching and can leak subjects whose ids are "
        "substrings of one another. Prefer grouped_non_stratified_split / dynamic_grouped_split."
    )
    assert train_ratio + val_ratio + test_ratio == 1.0, (
        "Train, validation, and test ratios must sum to 1, but got {}, {}, {}".format(train_ratio, val_ratio, test_ratio)
    )

    subjects = []
    for i in files:
        subjects.append(re.findall(pattern, i)[0])

    subjects = sorted(list(set(subjects)))

    if test:
        train_subs, test_subs = train_test_split(subjects, test_size=test_ratio, random_state=base_seed)
        train_sub_files = [file for file in files if any(tr_sub in file for tr_sub in train_subs)]
        test_sub_files = [file for file in files if any(test_sub in file for test_sub in test_subs)]
        return train_sub_files, test_sub_files

    train_subs, val_subs = train_test_split(
        subjects,
        test_size=val_ratio / (val_ratio + train_ratio),
        random_state=base_seed + seed_increment,
    )
    train_sub_files = [file for file in files if any(tr_sub in file for tr_sub in train_subs)]
    val_sub_files = [file for file in files if any(val_sub in file for val_sub in val_subs)]
    return train_sub_files, val_sub_files


def split(files: list, fn: callable, folds=5, save_path: str = None, split_pattern: str = None):
    """Build ``folds`` train/val splits plus a held-out test set and save them to JSON.

    The JSON at ``save_path`` (all folds) and its ``TEST_`` sibling are the canonical
    outputs. The returned ``(train, val, test)`` is only the *last* fold, provided as a
    convenience for quick inspection — do not treat it as the full cross-validation split.
    """
    if len(re.findall(r"/sub-\d+/", files[0])) > 0:
        logging.warning(
            "BIDS format detected. Consider switching to BIDSsplit_XXX to avoid data leakage (same subject in train/val/test) \
                if you are not already doing it."
        )

    splits_trval = []
    trainval, test = fn(files, test=True)

    for i in range(folds):
        train, val = fn(trainval, test=False, seed_increment=i)
        splits_trval.append({"train": train, "val": val})
    if save_path is not None:
        enhanced_save_json(obj=splits_trval, file=save_path)
        enhanced_save_json(obj=test, file=save_path.replace("split_", "TEST_"))
    return train, val, test


def dynamic_split(
    files: list,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    folds: int = 5,
    save_path: str = None,
):
    splits_trval = []
    trainval, test = non_stratified_split(files, train_ratio, val_ratio, test_ratio, test=True)

    for i in range(folds):
        train, val = non_stratified_split(trainval, train_ratio, val_ratio, test_ratio, test=False, seed_increment=i)
        splits_trval.append({"train": train, "val": val})

    if save_path is not None:
        enhanced_save_json(obj=splits_trval, file=save_path)
        enhanced_save_json(obj=test, file=save_path.replace("split_", "TEST_"))

    return train, val, test


def dynamic_grouped_split(
    files: list,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    group_by: str,
    folds: int = 5,
    save_path: str = None,
):
    splits_trval = []
    trainval, test = grouped_non_stratified_split(
        files,
        train_ratio,
        val_ratio,
        test_ratio,
        group_by=group_by,
        test=True,
    )

    for i in range(folds):
        train, val = grouped_non_stratified_split(
            trainval,
            train_ratio,
            val_ratio,
            test_ratio,
            group_by=group_by,
            test=False,
            seed_increment=i,
        )
        splits_trval.append({"train": train, "val": val})

    if save_path is not None:
        enhanced_save_json(obj=splits_trval, file=save_path)
        enhanced_save_json(obj=test, file=save_path.replace("split_", "TEST_"))

    return train, val, test


def subset_split_assignments(
    splits_trval: list[dict[str, list[str]]],
    test: list[str],
    files: list[str],
) -> tuple[list[dict[str, list[str]]], list[str]]:
    """Restrict an existing split to eligible files without changing subject assignments."""
    selected = set(files)
    reference_files = set(test)
    for fold in splits_trval:
        reference_files.update(fold["train"])
        reference_files.update(fold["val"])
    missing = selected - reference_files
    if missing:
        raise ValueError(f"{len(missing)} selected files are absent from the reference split.")

    subset_splits = [
        {
            "train": [path for path in fold["train"] if path in selected],
            "val": [path for path in fold["val"] if path in selected],
        }
        for fold in splits_trval
    ]
    subset_test = [path for path in test if path in selected]
    return subset_splits, subset_test
