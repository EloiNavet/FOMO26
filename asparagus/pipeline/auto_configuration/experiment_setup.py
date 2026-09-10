import logging
import os
from asparagus.functional.hydra import fast_instantiate
from asparagus.modules.dataclasses import DataFiles
from asparagus.pipeline.auto_configuration.versioning import pathing, versioning
from gardening_tools.functional.paths.read import load_json
from pathlib import Path


def _is_rank_zero_process() -> bool:
    return os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) == "0"


def relocate_dataset_paths(paths, data_path):
    """Rebase portable split entries onto the configured dataset directory.

    Split JSON files historically contain absolute paths from the machine that
    generated them. The dataset directory name is stable across installations,
    so preserve the path below that directory while replacing its old root.
    """
    dataset_root = Path(data_path)
    dataset_name = dataset_root.name

    def relocate(value):
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, tuple):
            return tuple(relocate(item) for item in value)
        if not isinstance(value, str):
            return value

        path = Path(value)
        try:
            dataset_index = len(path.parts) - 1 - tuple(reversed(path.parts)).index(dataset_name)
        except ValueError:
            if path.is_absolute():
                return value
            return str(dataset_root / path)
        return str(dataset_root.joinpath(*path.parts[dataset_index + 1 :]))

    return relocate(paths)


def prepare_standard_experiment(cfg):
    pathingcfg = pathing(cfg, train=True)
    versioncfg = versioning(cfg)
    if not os.path.isfile(cfg.test_split_path):
        if _is_rank_zero_process():
            logging.warning("No test split found")
        test = None
    else:
        test = load_json(cfg.test_split_path)

    filecfg = DataFiles(
        dataset_json=load_json(pathingcfg.dataset_json_path),
        splits=relocate_dataset_paths(
            load_json(cfg.train_split_path)[cfg.data.fold],
            cfg.data.data_path,
        ),
        test=relocate_dataset_paths(test, cfg.data.data_path) if test is not None else None,
    )
    if _is_rank_zero_process():
        logging.warning(f"###RUN-ID={versioncfg.version}###")
    return filecfg, pathingcfg, versioncfg


def prepare_inference(cfg):
    pathingcfg = pathing(cfg, train=False)
    filecfg = DataFiles(
        dataset_json=load_json(pathingcfg.dataset_json_path),
        splits=None,
        test=relocate_dataset_paths(
            load_json(cfg.data.test_split_path),
            cfg.data.test_data_path,
        ),
    )
    return filecfg, pathingcfg


def prepare_ssl_plugins(cfg):
    plugins = []
    if cfg.plugins is None:
        return plugins
    if cfg.plugins.seg is not None:
        plugins.append(prepare_online_segmentation(cfg))
    return plugins


def prepare_online_segmentation(cfg):
    dataset_json = load_json(cfg.plugins.seg.dataset_json_path)
    splits = relocate_dataset_paths(
        load_json(cfg.plugins.seg.splits_path)[cfg.plugins.seg.data.fold],
        Path(cfg.plugins.seg.dataset_json_path).parent,
    )

    num_classes = dataset_json["metadata"]["n_classes"]
    num_modalities = dataset_json["metadata"]["n_modalities"]
    cpu_transforms = fast_instantiate(cfg.plugins.seg._cpu_transforms)
    seg_data_module = fast_instantiate(
        cfg.plugins.seg._data_module,
        train_split=splits["train"],
        val_split=splits["val"],
        train_transforms=cpu_transforms,
        val_transforms=cpu_transforms,
    )

    model = fast_instantiate(
        cfg.model._plugin_seg_net,
        input_channels=num_modalities,
        output_channels=num_classes,
    )

    plugin = fast_instantiate(
        cfg.plugins.seg._plugin,
        model=model,
        data_module=seg_data_module,
        output_channels=num_classes,
    )

    return plugin
