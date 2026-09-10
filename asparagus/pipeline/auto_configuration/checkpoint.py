import os
import torch
import yaml
from asparagus.functional.huggingface import download_hf_checkpoint
from asparagus.functional.versioning import detect_id
from hydra.utils import get_class


def load_checkpoint_state_dict(path):
    """Load a checkpoint file and return the state_dict."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if "state_dict" in ckpt:
        print(f"Loading weights trained for {ckpt.get('global_step', '?')} steps / {ckpt.get('epoch', '?')} epochs.")
        return ckpt["state_dict"]
    elif "network_weights" in ckpt:
        print("Loading weights from external checkpoint (network_weights key).")
        return ckpt["network_weights"]
    else:
        raise ValueError("Unsupported checkpoint format. Expected 'state_dict' or 'network_weights' key.")


def resolve_checkpoint_path(cfg):
    """Resolve checkpoint file path from config. Returns path or None."""
    if cfg.checkpoint_run_id:
        folder = detect_id(cfg.checkpoint_run_id)
        return os.path.join(folder, "checkpoints", cfg.load_checkpoint_name)
    if cfg.checkpoint_path:
        return cfg.checkpoint_path
    return None


def _is_path_within(path, parent):
    path = os.path.abspath(os.fspath(path))
    parent = os.path.abspath(os.fspath(parent))
    try:
        return os.path.commonpath([path, parent]) == parent
    except ValueError:
        return False


def _validate_explicit_resume_checkpoint(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise ValueError(f"resume_checkpoint_path is not a readable Torch checkpoint: {path}") from exc

    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"resume_checkpoint_path must be a Lightning checkpoint with a state_dict: {path}")


def resolve_training_resume_checkpoint(checkpoint_dir, required=False, resume_checkpoint_path=None):
    """Return the Lightning checkpoint used to continue a run in its own directory."""
    if resume_checkpoint_path:
        checkpoint_dir = os.path.abspath(os.fspath(checkpoint_dir))
        resume_checkpoint_path = os.path.abspath(os.fspath(resume_checkpoint_path))
        if not _is_path_within(resume_checkpoint_path, checkpoint_dir):
            raise ValueError(
                "resume_checkpoint_path must point inside this run's checkpoint directory: "
                f"{checkpoint_dir}. Got: {resume_checkpoint_path}"
            )
        if not os.path.isfile(resume_checkpoint_path):
            raise FileNotFoundError(f"resume_checkpoint_path does not exist: {resume_checkpoint_path}")
        _validate_explicit_resume_checkpoint(resume_checkpoint_path)
        return resume_checkpoint_path

    path = os.path.join(checkpoint_dir, "last.ckpt")
    if os.path.isfile(path):
        return path
    if required:
        raise FileNotFoundError(
            "resume_training=true requires an existing same-run checkpoint at "
            f"{path}. Use the original command with the original run_id and output-directory convention."
        )
    return None


def resolve_training_resume_seed(run_dir):
    """Read the saved training seed without constructing checkpoint tensors."""
    hparams_path = os.path.join(run_dir, "hparams.yaml")
    if not os.path.isfile(hparams_path):
        return None
    with open(hparams_path, encoding="utf-8") as stream:
        hparams_node = yaml.compose(stream, Loader=yaml.BaseLoader)
    if hparams_node is None:
        return None
    for key_node, value_node in hparams_node.value:
        if key_node.value == "validation_mask_seed":
            return int(value_node.value)
    return None


def resolve_checkpoint(cfg):
    """Resolve and load checkpoint from config. Returns a state_dict or None."""
    hf_id = getattr(cfg, "hf_model_id", None) or None
    ckpt_path = resolve_checkpoint_path(cfg)

    sources = [s for s in [ckpt_path, hf_id] if s]
    if len(sources) > 1:
        raise ValueError("Provide only one of: checkpoint_run_id, checkpoint_path, hf_model_id")
    if len(sources) == 0:
        return None

    if ckpt_path:
        state_dict = load_checkpoint_state_dict(ckpt_path)
        # A published third-party backbone keeps its own naming conventions whether it arrives
        # over the network or was staged to disk first. Compute nodes on an air-gapped cluster
        # can only ever see the staged copy, so the remap must be reachable from a local path
        # too -- otherwise the same bytes transfer through `hf_model_id` and silently fail to
        # transfer through `checkpoint_path`.
        external_format = getattr(cfg, "external_weight_format", None) or None
        if external_format:
            weight_mapper = get_class(external_format)
            state_dict = weight_mapper(state_dict).remap_keys()
        return state_dict

    path = download_hf_checkpoint(hf_id)
    state_dict = load_checkpoint_state_dict(path)

    weight_mapper = get_class(cfg.hf_weight_format)
    return weight_mapper(state_dict).remap_keys()
