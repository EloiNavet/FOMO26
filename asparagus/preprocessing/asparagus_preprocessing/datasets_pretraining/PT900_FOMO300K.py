import os
from asparagus_preprocessing.configs.preprocessing_presets import (
    get_FOMO300K_saving_config,
    get_noresampling_preprocessing_config,
)
from asparagus_preprocessing.paths import get_data_path
from asparagus_preprocessing.utils.dataclasses import DatasetConfig
from asparagus_preprocessing.utils.metadata_generation import (
    combine_datasets_without_splits,
    combine_geometry_qc_manifests,
    generate_dataset_json,
    missing_dataset_prefixes,
)
from asparagus_preprocessing.utils.parser import asparagus_parser
from asparagus_preprocessing.utils.saving import enhanced_save_json


def main(
    path: str = "",
    subdir: str = "",
    processes=12,
    bidsify=False,
    save_dset_metadata=False,
    save_as_tensor=False,
):
    dataset_collection = [
        "PT001",
        "PT002",
        "PT003",
        "PT004",
        "PT005",
        "PT006",
        "PT007",
        "PT008",
        "PT009",
        "PT010",
        "PT011",
        "PT012",
        "PT013",
        "PT014",
        "PT015",
        "PT016",
        "PT017",
        "PT018",
        "PT019",
        "PT020",
        "PT021",
        "PT022",
        "PT023",
        "PT024",
        "PT025",
        "PT026",
        "PT027",
        "PT028",
        "PT029",
        "PT030",
        "PT031",
        "PT032",
        "PT033",
        "PT034",
        "PT035",
        "PT036",
    ]
    dataset_config = DatasetConfig(
        task_name="PT900_FOMO300K",
        n_classes=1,
        n_modalities=1,
        in_extensions=[],
        split=None,
        patterns_exclusion=[],
        patterns_DWI=[],
        patterns_PET=[],
        patterns_perfusion=[],
        patterns_m0=[],
        patterns_bidsify=[],
        df_columns=[],
    )
    preprocessing_config = get_noresampling_preprocessing_config()
    saving_config = get_FOMO300K_saving_config(
        save_as_tensor=save_as_tensor,
        save_dset_metadata=save_dset_metadata,
        bidsify=bidsify,
    )

    target_dir = os.path.join(get_data_path(), dataset_config.task_name)
    os.makedirs(target_dir, exist_ok=True)

    all_dataset_json, all_files = combine_datasets_without_splits(dataset_collection, allow_missing=True)
    available_dataset_collection = list(all_dataset_json)
    missing_datasets = missing_dataset_prefixes(dataset_collection, available_dataset_collection)
    if not all_files:
        raise LookupError(
            f"No processed datasets from {dataset_collection} were found under {get_data_path()}. "
            "Run the per-dataset preprocessing first, or point ASPARAGUS_DATA to the processed toy dataset."
        )

    geometry_metadata = combine_geometry_qc_manifests(available_dataset_collection, target_dir)
    generate_dataset_json(
        os.path.join(target_dir, "dataset.json"),
        dataset_name=dataset_config.task_name,
        preprocessing_config=preprocessing_config,
        saving_config=saving_config,
        dataset_config=dataset_config,
        metadata={
            "dataset_collection_jsons": all_dataset_json,
            "dataset_collection": available_dataset_collection,
            "dataset_collection_requested": dataset_collection,
            "dataset_collection_missing": missing_datasets,
            "final_files": len(all_files),
            **geometry_metadata,
        },
    )
    enhanced_save_json(all_files, os.path.join(target_dir, "paths.json"))


if __name__ == "__main__":
    args = asparagus_parser.parse_args()
    main(
        processes=args.num_workers,
        bidsify=args.bidsify,
        save_dset_metadata=args.save_dset_metadata,
        save_as_tensor=args.save_as_tensor,
    )
