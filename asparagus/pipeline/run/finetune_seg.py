import hydra
import lightning as pl
import os
import random
import time
from asparagus.functional.hydra import fast_instantiate
from asparagus.functional.versioning import generate_unused_run_id
from asparagus.modules.hydra.plugins.searchpath_plugins import FinetuneSearchpathPlugin
from asparagus.modules.transforms.presets import CPU_seg_test_transforms
from asparagus.paths import get_config_path
from asparagus.pipeline.auto_configuration.checkpoint import resolve_checkpoint
from asparagus.pipeline.auto_configuration.experiment_setup import (
    prepare_standard_experiment,
)
from asparagus.pipeline.auto_configuration.logging import logging
from asparagus.pipeline.auto_configuration.pretrained import resolve_pretrained_weights
from asparagus.pipeline.run.checkpoint_selection import (
    prediction_filename,
    resolve_test_checkpoint_role,
)
from asparagus.pipeline.run.evaluation_identity import evaluate_and_record
from asparagus.pipeline.run.scientific_checkpoint import DownstreamScientificInvariantCheckpoint
from dotenv import load_dotenv
from gardening_tools.modules.networks.components.weight_init import set_params_to_zero
from hydra.core.hydra_config import HydraConfig
from hydra.core.plugins import Plugins
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from omegaconf import DictConfig, OmegaConf

if os.environ.get("FOMO26_DISABLE_DOTENV") != "1":
    load_dotenv()

OmegaConf.register_new_resolver("random", lambda min, max: random.randint(min, max), replace=True)
OmegaConf.register_new_resolver(
    "version",
    lambda resume_training, run_dir: generate_unused_run_id(resume_training=resume_training, run_dir=run_dir),
    use_cache=True,
    replace=True,
)
OmegaConf.register_new_resolver("eval", eval, replace=True)
Plugins.instance().register(FinetuneSearchpathPlugin)


@hydra.main(
    config_path=get_config_path(),
    config_name="default_finetune_seg",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    # The Jean-Zay child-runtime fingerprint check lived here. It compared the interpreter,
    # venv and pinned torch/numpy build against one specific cluster installation, and armed
    # itself only from FOMO26_RUNTIME_FINGERPRINT, which that cluster's bootstrap set. The HPC
    # orchestration rail is not part of the public code release, so the check went with it
    # rather than being kept as a stub that would always pass and guarantee nothing.
    run_started_at = time.perf_counter()
    print(f"{OmegaConf.to_yaml(cfg)}\n Version: {cfg.run_id}\n Run dir: {HydraConfig.get().run.dir}\n")
    logging_safe_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    file_store, path_store, version_store = prepare_standard_experiment(cfg)
    weights = resolve_checkpoint(cfg)
    pl.seed_everything(seed=cfg.training.seed, workers=True)

    assert "load_checkpoint_name" in cfg.keys(), "load_checkpoint_name not in config. Did you supply a scratch config?"

    loggers = logging(
        ckpt_wandb_id=version_store.wandb_id,
        ckpt_mlflow_id=version_store.mlflow_id,
        log_file_name=HydraConfig.get().job.name,
        run_dir=path_store.run_dir,
        version=version_store.version,
        wandb_config=logging_safe_cfg,
        wandb_experiment=HydraConfig.get().job.config_name,
        wandb_project=cfg.logger.wandb_project,
        wandb_logging=cfg.logger.wandb_logging,
        csv_logging=cfg.logger.get("csv_logging", False),
        mlflow_logging=cfg.logger.mlflow_logging,
        log_to_stdout=cfg.logger.log_to_stdout,
    )

    best_ckpt_callback = ModelCheckpoint(
        dirpath=path_store.ckpt_save_dir,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best",
        enable_version_counter=False,
    )
    last_ckpt_callback = ModelCheckpoint(
        dirpath=path_store.ckpt_save_dir,
        every_n_epochs=cfg.model.ckpt_every_n_epoch,
        save_top_k=1,
        filename="last",
        enable_version_counter=False,
    )

    # Lightning raises MisconfigurationException if a TQDMProgressBar is passed while
    # enable_progress_bar is False, and configs/core/base.yaml sets
    # `enable_progress_bar: ${logger.progress_bar}`. Only attach the bar when it is enabled;
    # the default is progress_bar: True, so production behaviour is unchanged.
    progressbar_callbacks = [TQDMProgressBar(refresh_rate=cfg.logger.log_every_n_steps)] if cfg.logger.progress_bar else []
    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch", log_momentum=True)
    scientific_invariant_callback = DownstreamScientificInvariantCheckpoint(cfg)
    profilers = None

    # Physical-scale normalization has to reach train, validation and test identically: a model
    # trained on 0.9 mm voxels that is evaluated on native ones is being asked a different question.
    runtime_target_spacing = cfg.transforms.runtime_target_spacing
    if runtime_target_spacing is not None:
        runtime_target_spacing = OmegaConf.to_container(runtime_target_spacing, resolve=True)
    cpu_tr_transforms = fast_instantiate(
        cfg.transforms._cpu_tr_transforms,
        patch_size=cfg.training.patch_size,
        p_oversample_foreground=cfg.transforms.p_oversample_foreground,
        runtime_target_spacing=runtime_target_spacing,
    )
    cpu_val_transforms = fast_instantiate(
        cfg.transforms._cpu_val_transforms,
        patch_size=cfg.training.patch_size,
        runtime_target_spacing=runtime_target_spacing,
    )
    gpu_tr_transforms = fast_instantiate(
        cfg.transforms._gpu_tr_transforms,
        ndim=len(cfg.training.patch_size),
        deep_supervision=cfg.model.deep_supervision,
    )

    data_module = fast_instantiate(
        cfg.lightning._data_module,
        train_split=file_store.splits["train"],
        val_split=file_store.splits["val"],
        train_transforms=cpu_tr_transforms,
        val_transforms=cpu_val_transforms,
        test_samples=file_store.test,
        test_transforms=CPU_seg_test_transforms(
            patch_size=cfg.training.patch_size,
            runtime_target_spacing=runtime_target_spacing,
        ),
    )

    model = fast_instantiate(
        cfg.model._seg_net,
        input_channels=file_store.dataset_json["metadata"]["n_modalities"],
        output_channels=file_store.dataset_json["metadata"]["n_classes"],
    )

    # SSL (AMAES/JEPA/deep-sup) encoder transfer: filter the checkpoint to the downstream encoder.
    weights, load_decoder = resolve_pretrained_weights(cfg, model, weights)

    # Resolve the checkpoint role once, before anything can name a checkpoint. The same value
    # drives the prediction filename below and the weights handed to trainer.test further down.
    test_checkpoint_role = resolve_test_checkpoint_role(cfg)
    test_output_path = os.path.join(path_store.run_dir, "predictions", prediction_filename(cfg, test_checkpoint_role))

    model_module = fast_instantiate(
        cfg.lightning._lightning_module,
        model=model,
        warmup_epochs=cfg.training.warmup_epochs,
        decoder_warmup_epochs=cfg.training.decoder_warmup_epochs,
        weights=weights,
        train_transforms=gpu_tr_transforms,
        val_transforms=None,
        optimizer=cfg.training.get("optimizer", cfg.model.finetune_optim),
        learning_rate=cfg.training.get("learning_rate", cfg.model.finetune_lr),
        weight_decay=cfg.training.get("weight_decay", 3e-5),
        deep_supervision=cfg.model.deep_supervision,
        inference_patch_size=cfg.training.patch_size,
        test_output_path=test_output_path,
        load_decoder=load_decoder,
        repeat_stem_weights=cfg.training.repeat_stem_weights,
    )

    trainer = fast_instantiate(
        cfg.lightning._trainer,
        callbacks=[
            last_ckpt_callback,
            best_ckpt_callback,
            scientific_invariant_callback,
            *progressbar_callbacks,
            lr_monitor_callback,
        ],
        log_every_n_steps=cfg.logger.log_every_n_steps,
        logger=loggers,
        profiler=profilers,
        default_root_dir=path_store.run_dir,
        max_epochs=cfg.training.epochs,
        limit_train_batches=cfg.training.train_batches_per_epoch_per_device,
        limit_val_batches=cfg.training.val_batches_per_epoch_per_device,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        use_distributed_sampler=False,
    )

    trainer.fit(
        model=model_module,
        datamodule=data_module,
    )

    if cfg.get("testing", {}).get("run_after_fit", False):
        if test_checkpoint_role != "current":
            model_module.model.apply(set_params_to_zero)
        # Resolves the checkpoint, evaluates it and writes evaluation_identity.json from that
        # one object. Nothing downstream re-infers the evaluated checkpoint from a filename,
        # a glob or the presence of best.ckpt.
        evaluate_and_record(
            cfg,
            trainer=trainer,
            model=model_module,
            datamodule=data_module,
            role=test_checkpoint_role,
            prediction_path=test_output_path,
            run_dir=path_store.run_dir,
            best_ckpt_callback=best_ckpt_callback,
            last_ckpt_callback=last_ckpt_callback,
        )
    else:
        print("Post-fit test skipped (testing.run_after_fit=false). Set testing.run_after_fit=true to evaluate.")
    from asparagus.pipeline.resource_metrics import write_resource_metrics

    write_resource_metrics(
        path_store.run_dir,
        model_module,
        run_started_at,
        devices=int(cfg.hardware.num_devices),
        samples_processed=int(trainer.global_step)
        * int(cfg.training.batch_size)
        * int(cfg.training.accumulate_grad_batches)
        * int(cfg.hardware.num_devices),
        optimizer_steps=int(trainer.global_step),
    )


if __name__ == "__main__":
    main()
