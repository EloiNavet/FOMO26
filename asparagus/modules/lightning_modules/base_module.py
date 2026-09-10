import copy
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from abc import abstractmethod
from asparagus.functional.lr_scheduling import (
    build_param_groups,
    cosine_decay_schedule,
    sawtooth_warmup_cosine_decay_schedule,
    simple_warmup_cosine_decay_schedule,
)
from asparagus.functional.pos_embed import resize_pos_embed_3d
from asparagus.functional.visualization import (
    get_logger_compatible_image_output_target,
    log_image_output_target_to_mlflow,
    log_image_output_target_to_wandb,
)
from torch.optim import SGD, AdamW
from torchvision import transforms
from typing import Optional


class BaseModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        warmup_epochs: int = None,
        warmup_steps: int = None,
        decoder_warmup_epochs: int = 0,
        cosine_period_ratio: float = 1,
        compile_mode: str = None,
        weights: dict = None,
        load_decoder: bool = True,
        optimizer: str = "SGD",
        train_transforms: Optional[transforms.Compose] = None,
        test_transforms: Optional[transforms.Compose] = None,
        val_transforms: Optional[transforms.Compose] = None,
        weight_decay: float = 3e-5,
        nesterov: bool = True,
        momentum: float = 0.99,
        repeat_stem_weights: bool = True,
        pretrained_target_size: Optional[tuple] = None,
        target_size: Optional[tuple] = None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.val_transforms = val_transforms
        self.pretrained_target_size = pretrained_target_size
        self.target_size = target_size

        self.loss = None
        self.train_metrics = None
        self.val_metrics = None
        self.warmup_epochs = warmup_epochs
        # Warmup stated directly in optimizer steps. Takes precedence over warmup_epochs when set;
        # None keeps the historical epoch-denominated behaviour for every existing config.
        self.warmup_steps = warmup_steps
        self.decoder_warmup_epochs = decoder_warmup_epochs
        self.ignore_index_in_metrics = 0
        self.cosine_period_ratio = cosine_period_ratio
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.momentum = momentum
        self.repeat_stem_weights = repeat_stem_weights
        assert 0 < cosine_period_ratio <= 1

        self.save_hyperparameters(
            ignore=[
                "model",
                "weights",
                "train_transforms",
                "val_transforms",
                "test_transforms",
                "contrastive_loss_demo",
                "contrastive_loss_patho",
                # Task-6 keeps these as frozen ScheduleSpec instances at runtime and writes their
                # resolved records to curriculum_schedules.json. Lightning's YAML logger attempts
                # to mutate dataclass fields while serialising hparams, so the typed runtime copy
                # must not be captured here.
                "curriculum_schedules",
            ]
        )
        self.model = model

        if weights is not None:
            self.load_state_dict(weights, load_decoder=load_decoder, strict=False)

        self.model = torch.compile(model, mode=compile_mode) if compile_mode is not None else model

    @abstractmethod
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        # Build parameter groups: norm/bias/positional params are excluded from
        # weight decay (MAE/ViT practice), and encoder/decoder are split into
        # separate groups when the sawtooth (decoder-warmup) schedule is used.
        param_groups = build_param_groups(
            self.named_parameters(),
            weight_decay=self.weight_decay,
            separate_encoder_decoder=self.decoder_warmup_epochs > 0,
        )
        no_decay = sum(g["weight_decay"] == 0.0 for g in param_groups)
        print(f"Optimizer parameter groups: {[g['name'] for g in param_groups]} ({no_decay} without weight decay)")

        if self.optimizer == "SGD":
            optimizer = SGD(
                param_groups,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
                nesterov=self.nesterov,
            )
        elif self.optimizer == "AdamW":
            gradient_clip_val = getattr(self.trainer, "gradient_clip_val", None)
            use_fused_adamw = gradient_clip_val is None or float(gradient_clip_val) <= 0.0
            optimizer = AdamW(
                param_groups,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                amsgrad=False,
                betas=(0.9, 0.98),
                fused=use_fused_adamw,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        print(
            f"Using optimizer {optimizer.__class__.__name__} with learning rate {self.learning_rate} "
            f"(fused={optimizer.defaults.get('fused', False)})"
        )

        # Calculate steps per epoch based on trainer configuration
        # if max_epochs is *not* set (i.e., set to -1), we are probably using max_steps
        # if max_epochs is set, we can calculate steps per epoch based on estimated_stepping_batches
        if self.trainer.max_epochs <= 0:
            optimizer_steps_per_epoch = self.trainer.limit_train_batches // self.trainer.accumulate_grad_batches
        else:
            optimizer_steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs

        # Scheduler option 1: Three-phase schedule with separate decoder/joint warmup
        if self.decoder_warmup_epochs > 0:
            scheduler = sawtooth_warmup_cosine_decay_schedule(
                optimizer,
                self.decoder_warmup_epochs,
                self.warmup_epochs,
                optimizer_steps_per_epoch,
                self.cosine_period_ratio,
                self.trainer.max_epochs,  # may be -1, if using max_steps
            )
        # Scheduler option 2: Two-phase schedule with joint warmup
        # `or 0` because warmup_epochs is None whenever a config states warmup_steps instead;
        # `None > 0` raises.
        elif (self.warmup_epochs or 0) > 0 or (self.warmup_steps is not None and int(self.warmup_steps) > 0):
            scheduler = simple_warmup_cosine_decay_schedule(
                optimizer,
                self.warmup_epochs or 0,
                optimizer_steps_per_epoch,
                self.cosine_period_ratio,
                self.trainer.max_epochs,  # may be -1, if using max_steps
                self.trainer.max_steps,  # may be -1, if using max_epochs
                warmup_steps=self.warmup_steps,
            )
        # Scheduler option 3: Just cosine annealing
        else:
            scheduler = cosine_decay_schedule(
                optimizer,
                optimizer_steps_per_epoch,
                self.cosine_period_ratio,
                self.trainer.max_epochs,  # may be -1, if using max_steps
                self.trainer.max_steps,  # may be -1, if using max_epochs
            )

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,  # scheduler is updated after each batch
        }

        return [optimizer], [scheduler_config]

    def load_state_dict(self, state_dict, load_decoder=True, *args, **kwargs):
        strict_requested = kwargs.get("strict", args[0] if args else True)
        if load_decoder and strict_requested is not False:
            # Lightning resume must restore an identical training state. The
            # permissive transfer route below is only for explicit pretraining
            # weight initialisation, which calls this method with strict=False.
            return super().load_state_dict(state_dict, *args, **kwargs)

        old_params = copy.deepcopy(self.state_dict())

        target_compiled = "_orig" in next(iter(old_params.keys()))
        source_compiled = "_orig" in next(iter(state_dict.keys()))

        print(f"Target compiled: {target_compiled}, source compiled: {source_compiled}")

        if not target_compiled and source_compiled:
            print("Source state_dict is compiled, but target model is not. Removing _orig suffix from state_dict keys.")
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        # Repeat stem weights when state_dict num_channels is smaller than new_state_dict num_channels
        if hasattr(self.model, "stem_weight_name") and self.model.stem_weight_name is not None and self.repeat_stem_weights:
            prefix = "model._orig_mod." if "_orig_mod" in list(state_dict.keys())[0] else "model."
            stem_name = f"{prefix}{self.model.stem_weight_name}"
            pt_input_channels = state_dict[stem_name].shape[1]
            ft_input_channels = old_params[stem_name].shape[1]
            if pt_input_channels < ft_input_channels:
                assert pt_input_channels == 1, (
                    "Stem weights can only be repeated if the input channels in the state_dict is 1."
                )
                print(f"Repeating stem weights from {pt_input_channels} to {ft_input_channels} channels for {stem_name}.")
                state_dict[stem_name] = state_dict[stem_name].repeat(1, ft_input_channels, 1, 1, 1) / ft_input_channels

        # Interpolate positional embeddings when spatial dimensions differ
        if self.pretrained_target_size is not None and self.target_size is not None:
            for key in list(state_dict.keys()):
                if key not in old_params or old_params[key].shape == state_dict[key].shape:
                    continue
                if key.endswith("pos_embed"):
                    num_prefix_tokens = getattr(self.model.eva, "num_prefix_tokens", 0)
                    patch_embed_size = tuple(self.model.encoder.proj.weight.shape[2:])
                    print(f"Interpolating {key}: {state_dict[key].shape} -> {old_params[key].shape}")
                    state_dict[key] = resize_pos_embed_3d(
                        state_dict[key],
                        old_params[key],
                        num_prefix_tokens=num_prefix_tokens,
                        pretrained_target_size=self.pretrained_target_size,
                        target_size=self.target_size,
                        patch_embed_size=patch_embed_size,
                    )

        # Fail loudly on any remaining positional-embedding mismatch. Silently
        # dropping it (below) would finetune at a new resolution with freshly
        # initialised positional embeddings, quietly degrading the model.
        unresolved_pos_embed = [
            key
            for key in state_dict
            if key.endswith("pos_embed") and key in old_params and old_params[key].shape != state_dict[key].shape
        ]
        if unresolved_pos_embed:
            raise ValueError(
                "Positional embedding shape mismatch that was not interpolated: "
                + ", ".join(
                    f"{k} (checkpoint {tuple(state_dict[k].shape)} vs model {tuple(old_params[k].shape)})"
                    for k in unresolved_pos_embed
                )
                + ". Set `pretrained_target_size` and `target_size` so they can be interpolated."
            )

        # Classify incoming keys against the current model BEFORE filtering, so the
        # reported lists are meaningful (previously they were computed after the
        # filter and were therefore always empty).
        incoming_keys = list(state_dict.keys())
        rejected_keys_new = sorted(k for k in incoming_keys if k not in old_params)
        rejected_keys_shape = sorted(
            k for k in incoming_keys if k in old_params and old_params[k].shape != state_dict[k].shape
        )
        rejected_keys_decoder = sorted(k for k in incoming_keys if not load_decoder and k.startswith("model.decoder"))

        def should_load_key(key):
            # reject all decoder keys regardless of their shape
            if not load_decoder and key.startswith("model.decoder"):
                return False
            # accept all keys that are in the old state dict and have the same shape
            return (key in old_params) and (old_params[key].shape == state_dict[key].shape)

        loadable_keys = [k for k in incoming_keys if should_load_key(k)]
        state_dict = {k: state_dict[k] for k in loadable_keys}

        # Load the state dict
        kwargs["strict"] = False
        super().load_state_dict(state_dict, *args, **kwargs)

        # Verify the intended keys actually changed value after loading. A loadable
        # key whose tensor is byte-identical afterwards is genuinely suspicious
        # (note: a value that coincidentally matched the prior init can false-flag).
        new_params = self.state_dict()
        rejected_keys_data = sorted(k for k in loadable_keys if torch.equal(old_params[k], new_params[k]))
        successful = len(loadable_keys) - len(rejected_keys_data)

        print(
            f"Successfully transferred weights for {successful}/{len(loadable_keys)} loadable layers "
            f"({len(old_params)} params in model)."
        )
        print(
            "Rejected the following keys:\n"
            f"Not in model: {rejected_keys_new}.\n"
            f"Wrong shape: {rejected_keys_shape}.\n"
            f"Decoder skipped: {rejected_keys_decoder}.\n"
            f"Loaded but unchanged (suspicious): {rejected_keys_data}."
        )
        if not load_decoder:
            print("Decoder weights were not loaded, as requested. If you want to load them, set `load_decoder=True`.")
            print(f"Rejected decoder keys: {rejected_keys_decoder}.")
        else:
            print("Warning! Also loaded the decoder. If you are finetuning, this might not be what you want.")

        assert successful > 0, "No weights were loaded. Check the state_dict and the model architecture."

    def _log_dict_of_images_to_wandb(self, imagedict: dict, log_key: str, task_type: str = ""):
        """
        Log a random image from the imagedict to wandb
        """
        batch_idx = np.random.randint(0, imagedict["input"].shape[0])
        image, output, target = get_logger_compatible_image_output_target(
            image=imagedict["input"][batch_idx],
            output=imagedict["output"][batch_idx],
            target=imagedict["target"][batch_idx],
            task_type=task_type,
        )
        for logger in self.trainer.loggers:
            if "WandbLogger" in logger.__class__.__name__:
                log_image_output_target_to_wandb(
                    logger=logger,
                    image=image,
                    output=output,
                    target=target,
                    log_key=log_key,
                    fig_title=imagedict["file"][batch_idx].split("/Task")[-1],
                    step=self.global_step,
                    task_type=task_type,
                )
            if "MLFlowLogger" in logger.__class__.__name__:
                log_image_output_target_to_mlflow(
                    logger=logger,
                    image=image,
                    output=output,
                    target=target,
                    log_key=log_key,
                    fig_title=imagedict["file"][batch_idx].split("/Task")[-1],
                    step=self.global_step,
                    task_type=task_type,
                )

    def _apply_train_transforms(self, batch):
        if self.train_transforms is not None:
            return self.train_transforms(batch)
        return batch

    def _apply_val_transforms(self, batch):
        if self.val_transforms is not None:
            return self.val_transforms(batch)
        return batch

    def _apply_test_transforms(self, batch):
        if self.test_transforms is not None:
            return self.test_transforms(batch)
        return batch
