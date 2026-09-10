import asparagus_preprocessing
import json
import logging
import os
import pandas as pd
from asparagus_preprocessing.paths import get_data_path
from asparagus_preprocessing.utils.dataclasses import (
    DatasetConfig,
    PreprocessingConfig,
    SavingConfig,
)
from asparagus_preprocessing.utils.detect import (
    find_and_add_test_splits,
    find_and_add_train_splits,
    find_processed_dataset,
    recursive_find_and_group_files,
    recursive_find_files,
)
from asparagus_preprocessing.utils.loading import load_json
from asparagus_preprocessing.utils.saving import enhanced_save_json
from asparagus_preprocessing.utils.splitting import split
from pathlib import Path


def generate_dataset_json(
    output_file: str,
    dataset_name: str,
    metadata: dict = {},
    dataset_config: DatasetConfig = None,
    saving_config: SavingConfig = None,
    preprocessing_config: PreprocessingConfig = None,
) -> None:
    json_dict = {}
    json_dict["name"] = dataset_name
    json_dict["metadata"] = metadata
    json_dict["preprocessing_config"] = preprocessing_config
    json_dict["saving_config"] = saving_config
    json_dict["dataset_config"] = dataset_config

    enhanced_save_json(json_dict, os.path.join(output_file))


def simple_postprocess_standard_dataset(
    dataset_config: DatasetConfig,
    preprocessing_config: PreprocessingConfig,
    saving_config: SavingConfig,
    target_dir: str,
    source_files_standard: list[str],
    source_files_excluded: list[str],
    processes: int = 12,
) -> None:
    postprocess_standard_dataset(
        dataset_config=dataset_config,
        preprocessing_config=preprocessing_config,
        saving_config=saving_config,
        target_dir=target_dir,
        source_files_standard=source_files_standard,
        source_files_DWI=[],
        source_files_PET=[],
        source_files_Perf=[],
        source_files_excluded=source_files_excluded,
        processes=processes,
    )


def postprocess_standard_dataset(
    dataset_config: DatasetConfig,
    preprocessing_config: PreprocessingConfig,
    saving_config: SavingConfig,
    target_dir: str,
    source_files_standard: list[str],
    source_files_DWI: list[str],
    source_files_PET: list[str],
    source_files_Perf: list[str],
    source_files_excluded: list[str],
    processes: int = 12,
) -> None:
    source_all_files = (
        len(source_files_standard)
        + len(source_files_DWI)
        + len(source_files_PET)
        + len(source_files_Perf)
        + len(source_files_excluded)
    )
    target_all_files = recursive_find_files(target_dir, extensions=dataset_config.in_extensions + [".pt"])

    files_delta = len(target_all_files) - source_all_files

    target_files_standard, target_files_DWI, target_files_PET, target_files_Perf, _ = recursive_find_and_group_files(
        base_path=target_dir,
        extensions=dataset_config.in_extensions + [".pt"],
        patterns_dwi=dataset_config.patterns_DWI,
        patterns_pet=dataset_config.patterns_PET,
        patterns_perfusion=dataset_config.patterns_perfusion,
        patterns_exclusion=dataset_config.patterns_exclusion,
        processes=processes,
    )
    geometry_metadata = generate_geometry_qc_manifests(target_dir)
    generate_dataset_json(
        os.path.join(target_dir, "dataset.json"),
        dataset_name=dataset_config.task_name,
        preprocessing_config=preprocessing_config,
        saving_config=saving_config,
        dataset_config=dataset_config,
        metadata={
            "files_source_directory_total": source_all_files,
            "files_source_directory_standard": len(source_files_standard),
            "files_source_directory_DWI": len(source_files_DWI),
            "files_source_directory_PET": len(source_files_PET),
            "files_source_directory_Perfusion": len(source_files_Perf),
            "files_source_directory_excluded": len(source_files_excluded),
            "files_target_directory_total": len(target_all_files),
            "files_target_directory_standard": len(target_files_standard),
            "files_target_directory_DWI": len(target_files_DWI),
            "files_target_directory_PET": len(target_files_PET),
            "files_target_directory_Perfusion": len(target_files_Perf),
            "files_delta_after_processing": files_delta,
            "n_classes": dataset_config.n_classes,
            "n_modalities": dataset_config.n_modalities,
            **geometry_metadata,
        },
    )
    enhanced_save_json(target_all_files, os.path.join(target_dir, "paths.json"))

    if dataset_config.split is not None:
        save_path = os.path.join(target_dir, dataset_config.split + ".json")
        split(
            files=target_all_files,
            fn=getattr(asparagus_preprocessing.utils.splitting, dataset_config.split),
            save_path=save_path,
        )


def generate_geometry_qc_manifests(target_dir: str) -> dict:
    """Write physical-geometry audit tables and cohort path manifests when available."""
    records = []
    for qc_path in sorted(Path(target_dir).rglob("*.geometry_qc.json")):
        with qc_path.open("r", encoding="utf-8") as file:
            record = json.load(file)
        base_path = record.get("output_base_path", str(qc_path).removesuffix(".geometry_qc.json"))
        candidates = [base_path + ".pt", base_path + ".nii.gz", base_path + ".nii"]
        existing_output = next((path for path in candidates if os.path.exists(path)), None)
        record["output_path"] = existing_output or candidates[0]
        records.append(record)

    if not records:
        return {}

    table = pd.DataFrame(records)
    table.to_csv(os.path.join(target_dir, "geometry_qc.tsv"), sep="\t", index=False)

    geometry_valid = sorted(
        record["output_path"] for record in records if record.get("included") and os.path.exists(record["output_path"])
    )
    primary_structural = sorted(
        record["output_path"]
        for record in records
        if record.get("primary_structural_eligible") and os.path.exists(record["output_path"])
    )
    anisotropy_ablation = sorted(
        record["output_path"]
        for record in records
        if (
            record.get("included")
            and record.get("structural_modality")
            and record.get("primary_structural_exclusion_reason") == "native_spacing_above_2p5mm"
            and os.path.exists(record["output_path"])
        )
    )
    enhanced_save_json(geometry_valid, os.path.join(target_dir, "paths_geometry_valid.json"))
    enhanced_save_json(primary_structural, os.path.join(target_dir, "paths_primary_structural.json"))
    enhanced_save_json(anisotropy_ablation, os.path.join(target_dir, "paths_structural_anisotropy_ablation.json"))
    enhanced_save_json(geometry_valid, os.path.join(target_dir, "paths_multimodal_geometry_valid.json"))
    exclusions = table.loc[~table["included"], "exclusion_reason"].value_counts().to_dict()
    return {
        "geometry_qc_manifest": "geometry_qc.tsv",
        "geometry_valid_files": len(geometry_valid),
        "primary_structural_files": len(primary_structural),
        "structural_anisotropy_ablation_files": len(anisotropy_ablation),
        "geometry_excluded_files": int((~table["included"]).sum()),
        "geometry_exclusion_reasons": {str(key): int(value) for key, value in exclusions.items()},
    }


def combine_geometry_qc_manifests(dataset_collection: list[str], target_dir: str) -> dict:
    frames = []
    manifest_names = (
        "paths_geometry_valid.json",
        "paths_primary_structural.json",
        "paths_structural_anisotropy_ablation.json",
        "paths_multimodal_geometry_valid.json",
    )
    combined_paths = {name: [] for name in manifest_names}
    for dataset in dataset_collection:
        dataset_dir = os.path.join(get_data_path(), dataset)
        qc_path = os.path.join(dataset_dir, "geometry_qc.tsv")
        if os.path.exists(qc_path):
            frames.append(pd.read_csv(qc_path, sep="\t"))
        for name in manifest_names:
            path = os.path.join(dataset_dir, name)
            if os.path.exists(path):
                combined_paths[name].extend(load_json(path))
    if not frames:
        return {}
    qc = pd.concat(frames, ignore_index=True)
    qc.to_csv(os.path.join(target_dir, "geometry_qc.tsv"), sep="\t", index=False)
    for name, paths in combined_paths.items():
        enhanced_save_json(sorted(paths), os.path.join(target_dir, name))
    exclusions = qc.loc[~qc["included"], "exclusion_reason"].value_counts().to_dict()
    return {
        "geometry_qc_manifest": "geometry_qc.tsv",
        "geometry_valid_files": len(combined_paths["paths_geometry_valid.json"]),
        "primary_structural_files": len(combined_paths["paths_primary_structural.json"]),
        "structural_anisotropy_ablation_files": len(combined_paths["paths_structural_anisotropy_ablation.json"]),
        "geometry_excluded_files": int((~qc["included"]).sum()),
        "geometry_exclusion_reasons": {str(key): int(value) for key, value in exclusions.items()},
    }


def combine_datasets_with_splits(
    dataset_collection: list[str],
) -> tuple[dict[str, dict], list[str], list[dict[str, list]], list[dict]]:
    all_dataset_json = {}
    all_files = []
    all_train_splits = [
        {"train": [], "val": []},
        {"train": [], "val": []},
        {"train": [], "val": []},
        {"train": [], "val": []},
        {"train": [], "val": []},
    ]
    all_test_splits = []

    for dataset in dataset_collection:
        dataset = find_processed_dataset(dataset)
        dataset_dir = os.path.join(get_data_path(), dataset)
        dataset_json = load_json(os.path.join(dataset_dir, "dataset.json"))
        paths_json = load_json(os.path.join(dataset_dir, "paths.json"))
        all_dataset_json[dataset] = dataset_json
        all_files += paths_json
        all_train_splits = find_and_add_train_splits(dataset_dir, all_train_splits)
        all_test_splits = find_and_add_test_splits(dataset_dir, all_test_splits)
    return all_dataset_json, all_files, all_train_splits, all_test_splits


def missing_dataset_prefixes(requested_datasets: list[str], available_datasets: list[str]) -> list[str]:
    return [
        dataset
        for dataset in requested_datasets
        if not any(
            available_dataset == dataset or available_dataset.startswith(f"{dataset}_")
            for available_dataset in available_datasets
        )
    ]


def combine_datasets_without_splits(
    dataset_collection: list[str],
    allow_missing: bool = False,
) -> tuple[dict[str, dict], list[str]]:
    all_dataset_json = {}
    all_files = []

    for dataset in dataset_collection:
        try:
            dataset = find_processed_dataset(dataset)
        except LookupError:
            if allow_missing:
                logging.warning("Skipping missing processed dataset %s", dataset)
                continue
            raise

        dataset_dir = os.path.join(get_data_path(), dataset)
        dataset_json_path = os.path.join(dataset_dir, "dataset.json")
        paths_json_path = os.path.join(dataset_dir, "paths.json")
        missing_metadata = [path for path in [dataset_json_path, paths_json_path] if not os.path.exists(path)]
        if missing_metadata:
            if allow_missing:
                logging.warning(
                    "Skipping processed dataset %s because required metadata is missing: %s",
                    dataset,
                    ", ".join(missing_metadata),
                )
                continue
            raise FileNotFoundError(
                f"Required metadata missing for processed dataset {dataset}: {', '.join(missing_metadata)}"
            )

        dataset_json = load_json(dataset_json_path)
        paths_json = load_json(paths_json_path)
        all_dataset_json[dataset] = dataset_json
        all_files += paths_json
    return all_dataset_json, all_files
