import math
from asparagus.functional.decorators import depends_on_mlflow
from asparagus.modules.callbacks.loggers import BaseLogger
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger, WandbLogger
from typing import Optional, Union


@depends_on_mlflow()
class SafeMLFlowLogger(MLFlowLogger):
    def log_metrics(self, metrics, step=None):
        safe_metrics = {
            k.replace("/", "_"): (-99999 if isinstance(v, float) and math.isnan(v) else v) for k, v in metrics.items()
        }
        super().log_metrics(safe_metrics, step)


class HydraConfigWandbLogger(WandbLogger):
    """Keep the structured Hydra config as W&B's single hyperparameter source."""

    def log_hyperparams(self, params) -> None:
        return None


def logging(
    ckpt_wandb_id: Union[str, None],
    ckpt_mlflow_id: Union[str, None],
    log_file_name: str,
    run_dir: str,
    version: Union[int, str],
    wandb_experiment: str,
    wandb_run_description: str = None,
    wandb_project: str = "Asparagus",
    wandb_entity: Optional[str] = None,
    wandb_log_model: Union[bool, str] = False,
    wandb_logging: bool = True,
    csv_logging: bool = False,
    wandb_config: dict = None,
    mlflow_logging: bool = False,
    log_to_stdout: bool = True,
    wandb_require_run_continuity: bool = False,
):
    """
    Configure and return loggers for training.

    Args:
        ckpt_wandb_id: ID for checkpoint (used by wandb)
        run_dir: Directory to save logs
        version: Version identifier
        wandb_experiment: Experiment name
        logger_type: Type of logger to use ('wandb' or 'mlflow')

    Returns:
        list: Configured loggers
    """
    loggers = [BaseLogger(save_dir=run_dir, file_name=log_file_name, log_to_stdout=log_to_stdout)]

    if csv_logging:
        loggers.append(CSVLogger(save_dir=run_dir, name="local_history", version=""))

    if wandb_logging:
        # A multi-segment lane must land every segment in ONE W&B run with a monotonically
        # increasing global step. `resume="allow"` silently starts a *new* run when the id is
        # missing or unusable, which splits a lane's history across runs and is only noticed after
        # the fact. When the caller declares that this run must be continuous
        # (`wandb_require_run_continuity=True`), resume with "must" so W&B refuses instead.
        wandb_resume = None
        if ckpt_wandb_id:
            wandb_resume = "must" if wandb_require_run_continuity else "allow"
        elif wandb_require_run_continuity:
            raise ValueError(
                "W&B run continuity was required for this run, but no previous run id could be "
                f"resolved from {run_dir}/wandb/latest-run. Continuing would start a second W&B "
                "run for the same lane and split its step history. Restore the offline run "
                "directory, pass the run id explicitly, or set "
                "`logger.wandb_require_run_continuity=false` for a genuinely fresh run."
            )
        loggers.append(
            HydraConfigWandbLogger(
                name=f"{wandb_experiment}_{version}",
                notes=wandb_run_description,
                save_dir=run_dir,
                project=wandb_project,
                group=wandb_experiment,
                log_model=wandb_log_model,
                version=ckpt_wandb_id if ckpt_wandb_id else None,
                resume=wandb_resume,
                entity=wandb_entity,
                config=wandb_config,
            )
        )

    if mlflow_logging:
        loggers.append(
            SafeMLFlowLogger(
                experiment_name=wandb_experiment,
                tracking_uri=f"file:{run_dir}/mlruns",
                run_id=ckpt_mlflow_id if ckpt_mlflow_id else None,
            )
        )

    return loggers
