"""PR9: segmentation post-fit sliding-window inference must be robust to thin/odd spatial axes.

The failing case: a volume with a z-axis (30) smaller than the inference patch depth (32) makes the
sliding window extract a degenerate boundary patch and crash inside the encoder residual blocks. The
fix pads the image up to the patch size, predicts, then crops the logits back.
"""

import os
import torch
from asparagus.functional.utils import fit_patch_size_to_image_size
from asparagus.modules.lightning_modules.segmentation_module import SegmentationModule
from asparagus.modules.networks.resenc_unet import resenc_unet_b


def _seg_module(in_ch=2, out_ch=2, patch=(64, 64, 32)):
    net = resenc_unet_b(dimensions="3D", input_channels=in_ch, output_channels=out_ch)
    return SegmentationModule(model=net, inference_patch_size=list(patch), inference_mode="3D").eval()


def test_fit_patch_size_never_floors_a_thin_axis_to_zero():
    assert fit_patch_size_to_image_size([160, 160, 32], [384, 512, 30]) == [160, 160, 32]
    assert all(p >= 32 for p in fit_patch_size_to_image_size([32, 32, 32], [16, 16, 16]))


def test_padded_sliding_window_handles_thin_z_axis():
    torch.manual_seed(0)
    mod = _seg_module(in_ch=2, out_ch=2, patch=(64, 64, 32))
    x = torch.randn(1, 2, 64, 64, 30)  # z=30 (not a multiple of 32) — the geometry that crashed
    patch = fit_patch_size_to_image_size(mod.inference_patch_size, list(x.shape[2:]))
    assert patch[2] == 32 and patch[2] > x.shape[-1]  # patch depth exceeds the image depth
    with torch.no_grad():
        logits = mod._sliding_window_predict_padded(x, patch)
    assert logits.shape == (1, 2, 64, 64, 30)  # cropped back to the original spatial extent
    assert torch.isfinite(logits).all()


def test_clean_patch_forward_works():
    torch.manual_seed(1)
    net = resenc_unet_b(dimensions="3D", input_channels=2, output_channels=2).eval()
    with torch.no_grad():
        out = net(torch.randn(1, 2, 64, 64, 32))
    out = out[0] if isinstance(out, (list, tuple)) else out
    assert out.shape[:2] == (1, 2) and torch.isfinite(out).all()


def test_task1_lesion_composes_with_run_after_fit_true():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    for n, f in [("random", lambda a, b: 0), ("version", lambda: "t"), ("eval", eval)]:
        try:
            OmegaConf.register_new_resolver(n, f)
        except Exception:
            pass
    os.environ.setdefault("ASPARAGUS_DATA", "/tmp")
    os.environ.setdefault("WANDB_ENTITY", "test")
    cfg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
    with initialize_config_dir(version_base="1.2", config_dir=cfg_dir):
        cfg = compose(
            config_name="projects/fomo26/finetune/task1_lesion_scratch",
            overrides=["testing.run_after_fit=true"],
        )
    assert cfg.testing.run_after_fit is True
