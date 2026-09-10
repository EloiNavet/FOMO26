import math
from asparagus.modules.transforms import Torch_ClampTarget
from asparagus.modules.transforms.crop import Torch_Crop
from asparagus.modules.transforms.foreground_masking import Torch_ForegroundAwareMask
from asparagus.modules.transforms.frepa_lite import Torch_FrepaLiteFrequencyCorruption
from asparagus.modules.transforms.pad import Torch_Pad
from gardening_tools.functional.transforms.spatial import get_max_rotated_size
from gardening_tools.modules.transforms.bias_field import Torch_BiasField
from gardening_tools.modules.transforms.blur import Torch_Blur
from gardening_tools.modules.transforms.copy_image_to_label import Torch_CopyImageToLabel
from gardening_tools.modules.transforms.cropping_and_padding import Torch_CenterCrop
from gardening_tools.modules.transforms.gamma import Torch_Gamma
from gardening_tools.modules.transforms.masking import Torch_Mask
from gardening_tools.modules.transforms.motion_ghosting import Torch_MotionGhosting
from gardening_tools.modules.transforms.noise import Torch_AdditiveNoise, Torch_MultiplicativeNoise
from gardening_tools.modules.transforms.normalize import Torch_Normalize
from gardening_tools.modules.transforms.ringing import Torch_GibbsRinging
from gardening_tools.modules.transforms.sampling import Torch_SimulateLowres
from gardening_tools.modules.transforms.spatial import Torch_Spatial
from torchvision import transforms


def _normalize_patch_size(patch_size):
    return [int(math.ceil(float(value))) for value in list(patch_size)]


def CPU_val_transforms(patch_size):
    patch_size = _normalize_patch_size(patch_size)
    return transforms.Compose(
        [
            Torch_Normalize(normalize=True),
            Torch_Pad(patch_size=patch_size),
            Torch_CenterCrop(target_size=patch_size),
            Torch_CopyImageToLabel(copy=True),
            Torch_ClampTarget(clamp=True, min_value=-2.0, max_value=4.0),
        ]
    )


def CPU_train_transforms(patch_size):
    patch_size = _normalize_patch_size(patch_size)
    p_rot_all_channel = 0.2
    p_scale_all_channel = 0.2

    if p_rot_all_channel > 0 or p_scale_all_channel > 0:
        pre_aug_patch_size = _normalize_patch_size(get_max_rotated_size(patch_size))
    else:
        pre_aug_patch_size = patch_size

    return transforms.Compose(
        [
            Torch_Normalize(normalize=True),
            Torch_Pad(patch_size=pre_aug_patch_size),
            Torch_Crop(patch_size=pre_aug_patch_size, p_oversample_foreground=0.0),
            Torch_Spatial(
                patch_size=patch_size,
                p_deform_all_channel=0.0,
                p_rot_all_channel=p_rot_all_channel,
                p_rot_per_axis=0.3,
                p_scale_all_channel=p_scale_all_channel,
                clip_to_input_range=False,
                skip_label=False,
            ),
            Torch_CopyImageToLabel(copy=True),
            Torch_ClampTarget(clamp=True, min_value=-2.0, max_value=4.0),
        ]
    )


def GPU_train_transforms(
    masking=False,
    ndim=3,
    mask_ratio=0.6,
    mask_policy="random",
    mask_token_size=(4,),
    mask_foreground_bias=0.5,
    mask_foreground_threshold=0.0,
    mask_foreground_dynamic_quantiles=(0.02, 0.98),
    mask_foreground_dynamic_scale=0.1,
    frepa_lite=None,
):
    axes = (0, ndim)
    tforms = transforms.Compose(
        [
            Torch_Blur(p_per_channel=0.1),
            Torch_BiasField(p_per_channel=0.2),
            Torch_Gamma(p_all_channel=0.2),
            Torch_MotionGhosting(p_per_channel=0.1, axes=axes),
            Torch_GibbsRinging(p_per_channel=0.1, axes=axes),
            Torch_SimulateLowres(p_per_channel=0.1, p_per_axis=0.3),
            Torch_MultiplicativeNoise(p_per_channel=0.1),
            Torch_AdditiveNoise(p_per_channel=0.1),
        ]
    )

    if frepa_lite is not None and bool(frepa_lite.get("enabled", False)):
        tforms.transforms.append(
            Torch_FrepaLiteFrequencyCorruption(
                enabled=True,
                p=frepa_lite.get("p", 1.0),
                ndim=ndim,
                low_cutoff=frepa_lite.get("low_cutoff", 1.0 / 3.0),
                high_cutoff=frepa_lite.get("high_cutoff", 2.0 / 3.0),
                low_scale_min=frepa_lite.get("low_scale_min", 0.7),
                low_scale_max=frepa_lite.get("low_scale_max", 1.3),
                low_noise_std=frepa_lite.get("low_noise_std", 0.03),
                high_mask_ratio=frepa_lite.get("high_mask_ratio", 0.25),
                eps=frepa_lite.get("eps", 1e-6),
            )
        )

    if masking:
        if str(mask_policy) == "random":
            tforms.transforms.append(Torch_Mask(ratio=mask_ratio, token_size=list(mask_token_size)))
        else:
            tforms.transforms.append(
                Torch_ForegroundAwareMask(
                    ratio=mask_ratio,
                    token_size=mask_token_size,
                    policy=str(mask_policy),
                    foreground_bias=mask_foreground_bias,
                    threshold=mask_foreground_threshold,
                    dynamic_quantiles=mask_foreground_dynamic_quantiles,
                    dynamic_scale=mask_foreground_dynamic_scale,
                )
            )

    return tforms


def GPU_train_transforms_unmasked(
    ndim=3,
    mask_ratio=0.6,
    mask_policy="random",
    mask_token_size=(4,),
    mask_foreground_bias=0.5,
    mask_foreground_threshold=0.0,
    mask_foreground_dynamic_quantiles=(0.02, 0.98),
    mask_foreground_dynamic_scale=0.1,
    frepa_lite=None,
):
    del mask_policy, mask_token_size, mask_foreground_bias, mask_foreground_threshold
    del mask_foreground_dynamic_quantiles, mask_foreground_dynamic_scale, frepa_lite
    return GPU_train_transforms(masking=False, ndim=ndim, mask_ratio=mask_ratio)


def GPU_identity_transforms(
    ndim=3,
    mask_ratio=0.6,
    mask_policy="random",
    mask_token_size=(4,),
    mask_foreground_bias=0.5,
    mask_foreground_threshold=0.0,
    mask_foreground_dynamic_quantiles=(0.02, 0.98),
    mask_foreground_dynamic_scale=0.1,
):
    del ndim, mask_ratio, mask_policy, mask_token_size, mask_foreground_bias, mask_foreground_threshold
    del mask_foreground_dynamic_quantiles, mask_foreground_dynamic_scale
    return transforms.Compose([])


def GPU_val_transforms(
    masking=False,
    mask_ratio=0.6,
    mask_policy="random",
    mask_token_size=(4,),
    mask_foreground_bias=0.5,
    mask_foreground_threshold=0.0,
    mask_foreground_dynamic_quantiles=(0.02, 0.98),
    mask_foreground_dynamic_scale=0.1,
):
    if masking:
        if str(mask_policy) == "random":
            mask_transform = Torch_Mask(ratio=mask_ratio, token_size=list(mask_token_size))
        else:
            mask_transform = Torch_ForegroundAwareMask(
                ratio=mask_ratio,
                token_size=mask_token_size,
                policy=str(mask_policy),
                foreground_bias=mask_foreground_bias,
                threshold=mask_foreground_threshold,
                dynamic_quantiles=mask_foreground_dynamic_quantiles,
                dynamic_scale=mask_foreground_dynamic_scale,
            )
        return transforms.Compose(
            [
                mask_transform,
            ]
        )
    return None
