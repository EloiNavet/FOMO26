"""Core ensemble inference for FOMO26 submission containers.

The per-task wrappers parse the official CLI arguments, assemble modalities in the order used by
``prepare_fomo26_asparagus``, then call these helpers. This module deliberately reuses the same
fold-ensemble implementations as local evaluation so the container path exercises the same TTA and
sliding-window behavior.
"""

from __future__ import annotations

import gc
import inspect
import json
import numpy as np
import torch
from asparagus.functional.collate import collate_return
from asparagus.modules.datasets.TrainDataset import SingleSubjectPredictDataset
from asparagus.modules.transforms.presets import CPU_seg_test_transforms
from finetuning.fomo26_inference.calibration import apply_age_bias
from finetuning.fomo26_inference.clsreg_ensemble import ClsRegFoldEnsemble
from finetuning.fomo26_inference.cross_patch import CROSS_PATCH_CHOICES, clsreg_inference_transforms
from finetuning.fomo26_inference.runtime_geometry import resolve_for_inputs
from finetuning.fomo26_inference.seg_ensemble import SegFoldEnsemble
from finetuning.fomo26_inference.time_budget import describe_plan, measure_pass_time, plan_inference
from finetuning.fomo26_inference.tta_safety import (
    assert_flip_tta_allowed,
    assert_tta_within_ladder,
    task_tta_ladder,
)
from gardening_tools.functional.paths.write import save_prediction_from_logits
from omegaconf import OmegaConf
from pathlib import Path


def _save_prediction(logits, output_path, properties) -> None:
    """Write a segmentation prediction to exactly ``output_path`` on either gardening_tools API.

    The release runtime is pinned to 0.3.2, the version that trained the packaged checkpoints, and
    its writer differs from 0.3.5's in a way that silently misplaces the file rather than raising:

      0.3.5  save_prediction_from_logits(logits, outpath, properties)      -> writes outpath
      0.3.2  save_prediction_from_logits(logits, outpath, properties,
                                         save_format="nii.gz")             -> writes outpath + ".nii.gz"

    Passing the official ``/output/<case>.nii.gz`` straight through to 0.3.2 therefore produces
    ``<case>.nii.gz.nii.gz``, and the container exits 0 having written nothing the organizers will
    find. Dispatching on the signature keeps one call site correct on both versions instead of
    encoding a guess about which one is installed.
    """
    if "save_format" in inspect.signature(save_prediction_from_logits).parameters:
        suffix = ".nii.gz"
        if not str(output_path).endswith(suffix):
            raise ValueError(f"Expected a {suffix} output path, got {output_path!r}.")
        save_prediction_from_logits(
            logits,
            str(output_path)[: -len(suffix)],
            properties=properties,
            save_format="nii.gz",
        )
        return
    save_prediction_from_logits(logits, str(output_path), properties=properties)


# Channel order our prepare_fomo26_asparagus used (TASK_SPECS). The wrappers assemble inputs to match.
MODALITY_ORDER = {
    "task1": ["adc", "dwi", "flair", "t2s_or_swi"],
    "task2": ["dwi", "flair", "t2s_or_swi"],
    "task3": ["t1"],
    "task4": ["t2"],
    "task5": ["t1"],
}

TTA_CHOICES = {"auto", "none", "flip3", "flip7"}
#: Ladder used when no task identity is available to consult the registry with.
TTA_LADDER_DEFAULT = ("none", "flip3", "flip7")


def _device_from_accelerator(accelerator: str) -> str:
    acc = (accelerator or "auto").lower()
    if acc in {"auto", "gpu", "cuda"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return acc


def _load_cfg(model_dir: Path):
    return OmegaConf.load(model_dir / "hydra" / "config.yaml")


def _ckpt(model_dir: Path, name: str) -> str:
    return str(model_dir / "checkpoints" / f"{name}.ckpt")


def _single_subject_batch(data, transforms):
    dataset = SingleSubjectPredictDataset([str(p) for p in data], transforms=transforms)
    return collate_return([dataset[0]])


def _first_properties(properties):
    return properties[0] if isinstance(properties, (list, tuple)) else properties


def _limit_records(records: list[dict], max_members: int | None) -> list[dict]:
    if max_members is None or int(max_members) <= 0:
        return records
    return records[: min(int(max_members), len(records))]


def _select_plan(
    kind: str,
    records: list[dict],
    tta: str,
    max_members: int,
    time_target_s: float,
    make_ensemble,
    batch,
    *,
    cross_patch: str = "none",
    task: str | int | None = None,
):
    if tta not in TTA_CHOICES:
        raise ValueError(f"Unsupported TTA level {tta!r}; expected one of {sorted(TTA_CHOICES)}.")
    if task is not None:
        # Flip TTA mirrors the volume. Refuse it for any task whose label space encodes laterality
        # and declares no class permutation -- that error is systematic and invisible to the metric.
        assert_flip_tta_allowed(task, tta)
        # A task may also cap how rich the ladder goes, on held-out evidence that the richer
        # levels score worse. An explicitly named level must respect that ceiling too.
        assert_tta_within_ladder(task, tta)

    limited = _limit_records(records, max_members)
    if not limited:
        raise SystemExit("No checkpoint records available for inference.")
    if tta != "auto":
        suffix = f", cross-patch={cross_patch}" if cross_patch != "none" else ""
        print(f"[{kind}] plan: {len(limited)} member(s) x TTA={tta}{suffix} (manual)")
        return limited, tta

    probe = make_ensemble(limited[:1], "none")
    try:
        t_per_pass = measure_pass_time(probe.predict_batch, batch, warmup=0, repeats=1)
    finally:
        del probe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # The governor still measures this device and picks the richest level that fits; the ladder it
    # may pick from is narrowed to what the task's held-out evidence supports, so spare time budget
    # can never buy a level that scored worse. A task declaring no ceiling keeps the full ladder.
    plan = plan_inference(
        t_per_pass=t_per_pass,
        target_s=float(time_target_s),
        max_members=len(limited),
        min_members=1,
        tta_levels=task_tta_ladder(task) if task is not None else TTA_LADDER_DEFAULT,
    )
    suffix = f", cross-patch={cross_patch}" if cross_patch != "none" else ""
    print(f"[{kind}] {describe_plan(plan)}{suffix}; measured one-member pass={t_per_pass:.2f}s")
    return limited[: plan.n_members], plan.tta


def _load_calibration(calibration_json: str | None, expected_kind: str, *, required: bool = False) -> dict:
    if not calibration_json:
        if required:
            raise SystemExit(f"A {expected_kind} calibration artifact is required by the frozen release policy.")
        return {}
    path = Path(calibration_json)
    if not path.is_file():
        if required:
            raise SystemExit(f"Required calibration artifact is missing: {path}.")
        return {}
    spec = json.loads(path.read_text())
    kind = spec.get("kind")
    if kind is not None and kind != expected_kind:
        raise SystemExit(f"Calibration file {path} is for kind={kind!r}, expected {expected_kind!r}.")
    return spec


def predict_seg(
    data,
    output_path,
    manifest_records,
    *,
    checkpoint_name="best",
    input_channels,
    output_channels,
    accelerator,
    tta="auto",
    max_members=0,
    time_target_s=115.0,
    calibration_json=None,
    ensemble_space="prob",
    window_policy="checkpoint_config_overlap_0.5",
    task=None,
    runtime_target_spacing=None,
):
    """Fold-ensemble segmentation over raw NIfTIs -> mask in the input image space."""
    del checkpoint_name, calibration_json  # records already carry the checkpoint paths; seg has no calibration.
    if window_policy != "checkpoint_config_overlap_0.5":
        raise ValueError(f"Unsupported segmentation window policy {window_policy!r}.")
    records = list(manifest_records)
    if not records:
        raise SystemExit("No checkpoint records available for segmentation inference.")
    cfg0 = _load_cfg(Path(records[0]["run_dir"]))
    # Canonicalize physical geometry before pad/crop when the task declares a fitted geometry.
    # reverse_preprocessing undoes it through ``size_before_resample``, so the mask still lands in
    # the input volume's own space. Tasks that declare nothing resolve to None and compose as before.
    if runtime_target_spacing is None:
        runtime_target_spacing, geometry_note = resolve_for_inputs(task, data)
    else:
        geometry_note = "caller-supplied runtime target spacing"
    print(f"[seg] geometry: {geometry_note} (target_spacing={runtime_target_spacing})")
    batch = _single_subject_batch(
        data,
        CPU_seg_test_transforms(
            patch_size=cfg0.training.patch_size,
            runtime_target_spacing=runtime_target_spacing,
        ),
    )
    device = _device_from_accelerator(accelerator)

    def make_ensemble(recs, chosen_tta):
        return SegFoldEnsemble(
            recs,
            input_channels,
            output_channels,
            device=device,
            tta=chosen_tta,
            ensemble_space=ensemble_space,
            runtime_target_spacing=runtime_target_spacing,
        )

    selected, chosen_tta = _select_plan("seg", records, tta, max_members, time_target_s, make_ensemble, batch, task=task)
    ensemble = make_ensemble(selected, chosen_tta)
    out = ensemble.predict_batch(batch)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    _save_prediction(
        out["src_probs"].detach().cpu().numpy(),
        output_path,
        _first_properties(out["properties"]),
    )
    print(f"[seg] ensemble of {len(selected)} fold(s), TTA={chosen_tta} -> {output_path}")


def predict_clsreg(
    data,
    output_path,
    manifest_records,
    *,
    checkpoint_name="best",
    input_channels,
    output_channels,
    accelerator,
    kind,
    tta="auto",
    max_members=0,
    time_target_s=115.0,
    calibration_json=None,
    calibration_required=False,
    cross_patch="none",
    task=None,
    runtime_target_spacing=None,
):
    """Fold-ensemble classification/regression over raw NIfTIs -> scalar .txt output."""
    del checkpoint_name  # records already carry the checkpoint paths.
    records = list(manifest_records)
    if not records:
        raise SystemExit(f"No checkpoint records available for {kind} inference.")
    if cross_patch not in CROSS_PATCH_CHOICES:
        raise ValueError(f"Unsupported cross_patch={cross_patch!r}; expected one of {CROSS_PATCH_CHOICES}.")
    cfg0 = _load_cfg(Path(records[0]["run_dir"]))
    # See predict_seg: the target is the geometry this task was actually fitted at, declared per
    # task in the registry and absent for every task that did not opt in.
    if runtime_target_spacing is None:
        runtime_target_spacing, geometry_note = resolve_for_inputs(task, data)
    else:
        geometry_note = "caller-supplied runtime target spacing"
    print(f"[{kind}] geometry: {geometry_note} (target_spacing={runtime_target_spacing})")
    batch = _single_subject_batch(
        data,
        clsreg_inference_transforms(
            cfg0.training.target_size,
            cross_patch=cross_patch,
            runtime_target_spacing=runtime_target_spacing,
        ),
    )
    device = _device_from_accelerator(accelerator)
    calibration = _load_calibration(calibration_json, kind, required=calibration_required)
    temperature = float(calibration.get("temperature", 1.0)) if kind == "cls" else 1.0

    def make_ensemble(recs, chosen_tta):
        return ClsRegFoldEnsemble(
            recs,
            input_channels,
            output_channels,
            kind,
            device=device,
            tta=chosen_tta,
            temperature=temperature,
            cross_patch=cross_patch,
        )

    selected, chosen_tta = _select_plan(
        kind,
        records,
        tta,
        max_members,
        time_target_s,
        make_ensemble,
        batch,
        cross_patch=cross_patch,
        task=task,
    )
    ensemble = make_ensemble(selected, chosen_tta)
    out = ensemble.predict_batch(batch)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if kind == "cls":
        value = float(out["probs"][0, 1].detach().cpu().item())
        np.savetxt(output_path, np.array([value]), fmt="%.8f")
    else:
        pred = np.array([float(out["pred"].reshape(-1)[0].detach().cpu().item())])
        if "age_bias" in calibration:
            bias = calibration["age_bias"]
            pred = apply_age_bias(pred, float(bias["a"]), float(bias["b"]))
        np.savetxt(output_path, pred, fmt="%.6f")
    suffix = f", cross-patch={cross_patch}" if cross_patch != "none" else ""
    print(f"[{kind}] ensemble of {len(selected)} fold(s), TTA={chosen_tta}{suffix} -> {output_path}")


def resolve_manifest_records(manifest_or_dir: str, checkpoint_name: str = "best") -> list[dict]:
    """Return fold records with in-use checkpoint paths from an orchestrator manifest or one run dir."""
    p = Path(manifest_or_dir)
    if p.is_dir():
        return [{"run_dir": str(p), "best_ckpt": _ckpt(p, checkpoint_name), "returncode": 0, "fold": 0}]

    records = json.loads(p.read_text())
    resolved = []
    for rec in records:
        if rec.get("returncode", 0) != 0 or not rec.get("run_dir"):
            continue
        run_dir = Path(rec["run_dir"])
        best_ckpt = rec.get("best_ckpt") or _ckpt(run_dir, checkpoint_name)
        resolved.append({**rec, "run_dir": str(run_dir), "best_ckpt": str(best_ckpt)})
    if not resolved:
        raise SystemExit(f"No successful run dirs in manifest {p}.")
    return resolved


def resolve_model_dirs(manifest_or_dir: str) -> list[Path]:
    """Backward-compatible helper for older wrappers."""
    return [Path(r["run_dir"]) for r in resolve_manifest_records(manifest_or_dir)]
