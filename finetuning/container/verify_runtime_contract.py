"""Build-time contract check, executed INSIDE the image before it is sealed.

Two failures motivate this gate, both of which produced an image that built cleanly, passed the
official validator and met the runtime budget while predicting background on every voxel:

1. The release runtime silently resolved a different ``gardening_tools`` than the one that trained
   the packaged checkpoints.
2. Because that release renamed the decoder's parameters, every decoder tensor missed by NAME.
   ``BaseModule.load_state_dict`` only asserts that *some* weight loaded, which is correct for SSL
   pretraining (where the decoder is deliberately dropped) but useless for a finetuned downstream
   checkpoint, where a partially loaded model is simply broken.

So a task that ships finetuned segmentation weights must load ALL of them, and the check runs
here, at build time, rather than being left to a real-case probe nobody may run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED_RUNTIME_VERSIONS = {"gardening_tools": "0.3.2"}
#: Tasks whose image ships finetuned segmentation weights and therefore must load them completely.
SEG_TASKS = {"task2": 2, "task4": 3}


def _runtime_input_channels(task: str) -> int:
    """How many channels predict.py actually stacks for this task.

    Not ``data.n_modalities`` from the checkpoint config: Task 2 reports 1 there while the runtime
    stacks three modalities. Instantiating from that fallback fabricates a stem-shaped mismatch the
    real runtime never has, which would either mask a genuine defect or fail a sound image. The
    module that assembles the inputs is the only authority on their count.
    """
    sys.path.insert(0, "/app/asparagus_repo/finetuning/container")
    from fomo_ensemble_predict import MODALITY_ORDER

    if task not in MODALITY_ORDER:
        _fail(f"{task} has no declared modality order; refusing to guess its input channel count")
    return len(MODALITY_ORDER[task])


def _fail(message: str) -> None:
    print(f"RUNTIME_CONTRACT_FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    import importlib.metadata as metadata

    for package, expected in REQUIRED_RUNTIME_VERSIONS.items():
        found = metadata.version(package)
        if found != expected:
            _fail(f"{package}=={found} installed, release runtime requires {expected}")
        print(f"RUNTIME_CONTRACT: {package}=={found}")

    manifest = json.loads(Path("/app/models/model_manifest.json").read_text())
    task = str(manifest.get("task"))
    if task not in SEG_TASKS:
        print(f"RUNTIME_CONTRACT: {task} ships no finetuned segmentation decoder; parameter check not applicable")
        return

    from asparagus.pipeline.auto_configuration.checkpoint import load_checkpoint_state_dict
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    records = manifest.get("models") or []
    if not records:
        _fail(f"{task} declares no packaged model records")
    for record in records:
        checkpoint = Path(record["path"])
        run_dir = checkpoint.parent.parent
        config = OmegaConf.load(run_dir / "hydra" / "config.yaml")
        channels = _runtime_input_channels(task)
        model = instantiate(
            config.model._seg_net,
            input_channels=channels,
            output_channels=SEG_TASKS[task],
        )
        model_state = model.state_dict()
        checkpoint_state = {
            key.removeprefix("model."): value for key, value in load_checkpoint_state_dict(str(checkpoint)).items()
        }
        model_keys, checkpoint_keys = set(model_state), set(checkpoint_state)
        unfilled = sorted(model_keys - checkpoint_keys)
        unmatched = sorted(checkpoint_keys - model_keys)
        # Matching names is not enough: a name that matches with a different shape is dropped by
        # load_state_dict just as silently as a name that does not match at all.
        shape_mismatches = sorted(
            key for key in model_keys & checkpoint_keys if tuple(model_state[key].shape) != tuple(checkpoint_state[key].shape)
        )
        transferable = len(model_keys & checkpoint_keys) - len(shape_mismatches)
        fold = record.get("fold")
        if unfilled or unmatched or shape_mismatches or transferable != len(model_keys):
            _fail(
                f"{task} fold {fold}: {transferable} of {len(model_keys)} model parameters are "
                f"shape-compatible with the checkpoint's {len(checkpoint_keys)}. "
                f"{len(unfilled)} would stay at their random initialisation, "
                f"{len(unmatched)} checkpoint tensors would be dropped, and "
                f"{len(shape_mismatches)} match by name but not by shape. "
                f"unfilled={unfilled[:3]} unmatched={unmatched[:3]} shape={shape_mismatches[:3]}"
            )
        print(
            f"RUNTIME_CONTRACT: {task} fold {fold} loads completely "
            f"({transferable}/{len(model_keys)} shape-compatible, 0 unfilled, 0 unmatched, "
            f"0 shape mismatches, input_channels={channels})"
        )


if __name__ == "__main__":
    main()
