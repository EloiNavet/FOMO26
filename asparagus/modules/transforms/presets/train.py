from asparagus.modules.transforms.crop import Torch_Crop
from asparagus.modules.transforms.pad import Torch_Pad
from asparagus.modules.transforms.spacing import Torch_ResampleToSpacing
from gardening_tools.functional.transforms.spatial import get_max_rotated_size
from gardening_tools.modules.transforms.bias_field import Torch_BiasField
from gardening_tools.modules.transforms.blur import Torch_Blur
from gardening_tools.modules.transforms.cropping_and_padding import Torch_CenterCrop
from gardening_tools.modules.transforms.deep_supervision import Torch_DownsampleSegForDS
from gardening_tools.modules.transforms.gamma import Torch_Gamma
from gardening_tools.modules.transforms.mirror import Torch_Mirror
from gardening_tools.modules.transforms.motion_ghosting import Torch_MotionGhosting
from gardening_tools.modules.transforms.noise import Torch_AdditiveNoise, Torch_MultiplicativeNoise
from gardening_tools.modules.transforms.normalize import Torch_CT_NormalizeC0, Torch_Normalize
from gardening_tools.modules.transforms.ringing import Torch_GibbsRinging
from gardening_tools.modules.transforms.sampling import Torch_Resize, Torch_SimulateLowres
from gardening_tools.modules.transforms.spatial import Torch_Spatial
from torchvision import transforms


def none(ndim=3, deep_supervision=False, **_unused):
    return None


def _runtime_spacing_stage(runtime_target_spacing):
    """Return the resampling stage as a list so a disabled run composes exactly as before.

    Splicing an empty list leaves the historical ``Compose`` byte-identical, which matters because
    every already-validated segmentation run must keep composing the way it did when it was scored.
    """
    if runtime_target_spacing is None:
        return []
    return [Torch_ResampleToSpacing(target_spacing=runtime_target_spacing)]


def CPU_seg_train_transforms(patch_size, normalize=True, p_oversample_foreground=0.33, runtime_target_spacing=None):
    """Segmentation training crops.

    ``p_oversample_foreground`` is the probability that a crop is centred on a foreground voxel.
    It is exposed because it is the only lever over how often the network sees the target at all:
    for a structure occupying ~1e-5 of the volume, uniformly random crops almost never contain
    one, and the network converges to predicting background everywhere. The default is unchanged.

    ``runtime_target_spacing`` normalizes physical voxel spacing before any spatial op, so a fixed
    patch covers a fixed physical field of view across a cohort with heterogeneous acquisition
    geometry. Default ``None`` leaves the pipeline unchanged.
    """
    if len(patch_size) == 2:
        axes = (0, 1)
    else:
        axes = (0, 1, 2)
    p_rot_all_channel = 0.2
    p_scale_all_channel = 0.2

    if p_rot_all_channel > 0 or p_scale_all_channel > 0:
        pre_aug_patch_size = get_max_rotated_size(patch_size)
    else:
        pre_aug_patch_size = patch_size

    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            *_runtime_spacing_stage(runtime_target_spacing),
            Torch_Pad(patch_size=pre_aug_patch_size),
            Torch_Crop(patch_size=pre_aug_patch_size, p_oversample_foreground=p_oversample_foreground),
            Torch_Spatial(
                patch_size=patch_size,
                p_deform_all_channel=0.0,
                p_rot_all_channel=p_rot_all_channel,
                p_rot_per_axis=0.3,
                x_rot_in_degrees=(-30.0, 30.0),
                y_rot_in_degrees=(-30.0, 30.0),
                z_rot_in_degrees=(-30.0, 30.0),
                scale_factor=(0.7, 1.4),
            ),
            Torch_Mirror(
                p_per_sample=1.0,
                p_mirror_per_axis=0.5,
                axes=axes,
            ),
        ]
    )


def CPU_clsreg_train_transforms_crop(target_size, normalize=True):
    if len(target_size) == 2:
        axes = (0, 1)
    else:
        axes = (0, 1, 2)
    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            Torch_Pad(patch_size=target_size),
            Torch_CenterCrop(target_size=target_size),
            Torch_Spatial(
                patch_size=target_size,
                p_deform_all_channel=0.0,
                p_rot_all_channel=0.2,
                p_rot_per_axis=0.3,
                p_scale_all_channel=0.2,
                x_rot_in_degrees=(-30.0, 30.0),
                y_rot_in_degrees=(-30.0, 30.0),
                z_rot_in_degrees=(-30.0, 30.0),
                scale_factor=(0.7, 1.4),
                crop=False,
                clip_to_input_range=False,
                skip_label=True,
            ),
            Torch_Mirror(
                p_per_sample=1.0,
                p_mirror_per_axis=0.5,
                axes=axes,
            ),
        ]
    )


def CPU_CT_C0_clsreg_train_transforms_crop(target_size, normalize=True):
    if len(target_size) == 2:
        axes = (0, 1)
    else:
        axes = (0, 1, 2)
    return transforms.Compose(
        [
            Torch_CT_NormalizeC0(normalize=normalize),
            Torch_Pad(patch_size=target_size),
            Torch_CenterCrop(target_size=target_size),
            Torch_Spatial(
                patch_size=target_size,
                p_deform_all_channel=0.0,
                p_rot_all_channel=0.2,
                p_rot_per_axis=0.3,
                p_scale_all_channel=0.2,
                x_rot_in_degrees=(-30.0, 30.0),
                y_rot_in_degrees=(-30.0, 30.0),
                z_rot_in_degrees=(-30.0, 30.0),
                scale_factor=(0.7, 1.4),
                crop=False,
                clip_to_input_range=False,
                skip_label=True,
            ),
            Torch_Mirror(
                p_per_sample=1.0,
                p_mirror_per_axis=0.5,
                axes=axes,
            ),
        ]
    )


def CPU_clsreg_train_transforms_resize(target_size, normalize=True):
    if len(target_size) == 2:
        axes = (0, 1)
    else:
        axes = (0, 1, 2)
    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            Torch_Resize(target_size=target_size),
            Torch_Spatial(
                patch_size=target_size,
                p_deform_all_channel=0.0,
                p_rot_all_channel=0.2,
                p_rot_per_axis=0.3,
                p_scale_all_channel=0.2,
                x_rot_in_degrees=(-30.0, 30.0),
                y_rot_in_degrees=(-30.0, 30.0),
                z_rot_in_degrees=(-30.0, 30.0),
                scale_factor=(0.7, 1.4),
                crop=False,
                clip_to_input_range=False,
                skip_label=True,
            ),
            Torch_Mirror(
                p_per_sample=1.0,
                p_mirror_per_axis=0.5,
                axes=axes,
            ),
        ]
    )


def CPU_clsreg_val_test_transforms_crop(target_size, normalize=True, runtime_target_spacing=None):
    """Deterministic cls/reg inference crop.

    ``runtime_target_spacing`` is spliced in the same position the segmentation presets use it --
    after normalization, before pad/crop -- so a fixed-size centre crop covers a fixed *physical*
    field of view instead of a fixed voxel count. Disabled by default, in which case
    ``_runtime_spacing_stage`` splices nothing and this composes exactly as it always has.
    """
    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            *_runtime_spacing_stage(runtime_target_spacing),
            Torch_Pad(patch_size=target_size),
            Torch_CenterCrop(target_size=target_size),
        ]
    )


def CPU_CT_C0_clsreg_val_test_transforms_crop(target_size, normalize=True):
    return transforms.Compose(
        [
            Torch_CT_NormalizeC0(normalize=normalize),
            Torch_Pad(patch_size=target_size),
            Torch_CenterCrop(target_size=target_size),
        ]
    )


def CPU_seg_val_transforms(patch_size, normalize=True, runtime_target_spacing=None):
    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            *_runtime_spacing_stage(runtime_target_spacing),
            Torch_Pad(patch_size=patch_size),
            Torch_Crop(patch_size=patch_size, p_oversample_foreground=0.0),
        ]
    )


def CPU_seg_test_transforms(patch_size, normalize=True, runtime_target_spacing=None):
    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            *_runtime_spacing_stage(runtime_target_spacing),
            Torch_Pad(patch_size=patch_size),
        ]
    )


def GPU_all_train_transforms(ndim=3, deep_supervision=False):
    axes = (0, ndim)
    tforms = transforms.Compose(
        [
            Torch_Blur(p_per_channel=0.15),
            Torch_BiasField(p_per_channel=0.2),
            Torch_Gamma(p_all_channel=0.15),
            Torch_MotionGhosting(p_per_channel=0.1, axes=axes),
            Torch_GibbsRinging(p_per_channel=0.1, axes=axes),
            Torch_SimulateLowres(p_per_channel=0.5, p_per_axis=0.25),
            Torch_MultiplicativeNoise(p_per_channel=0.1),
            Torch_AdditiveNoise(p_per_channel=0.1),
        ]
    )

    if deep_supervision:
        tforms.transforms.append(Torch_DownsampleSegForDS(deep_supervision=True))

    return tforms
