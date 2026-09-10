"""Small run-level compute sidecar used by architecture comparisons."""

from __future__ import annotations

import json
import re
import time
import torch
from datetime import datetime
from pathlib import Path


def _parameter_counts(model) -> dict[str, int]:
    named = list(model.named_parameters())
    encoder_named = [(name, parameter) for name, parameter in named if ".encoder." in f".{name}."]
    ema_encoder_named = [
        (name, parameter) for name, parameter in encoder_named if "target_encoder." in name or "momentum_model." in name
    ]
    return {
        "parameters": sum(parameter.numel() for _, parameter in named),
        "trainable_parameters": sum(parameter.numel() for _, parameter in named if parameter.requires_grad),
        "encoder_parameters": sum(parameter.numel() for _, parameter in encoder_named),
        "online_encoder_parameters": sum(
            parameter.numel()
            for name, parameter in encoder_named
            if "target_encoder." not in name and "momentum_model." not in name
        ),
        "ema_encoder_parameters": sum(parameter.numel() for _, parameter in ema_encoder_named),
        "decoder_parameters": sum(parameter.numel() for name, parameter in named if ".decoder." in f".{name}."),
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_resource_metrics(
    run_dir,
    model,
    started_at: float,
    *,
    devices: int = 1,
    samples_processed: int | None = None,
    optimizer_steps: int | None = None,
) -> dict:
    elapsed = max(0.0, time.perf_counter() - float(started_at))
    path = Path(run_dir) / "resource_metrics.json"
    previous = {}
    if path.is_file():
        try:
            previous = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            previous = {}
    cumulative_elapsed = float(previous.get("elapsed_seconds", 0.0)) + elapsed
    cumulative_samples = (
        max(int(previous.get("samples_processed", 0)), int(samples_processed))
        if samples_processed is not None
        else int(previous.get("samples_processed", 0))
    )
    payload = {
        "elapsed_seconds": cumulative_elapsed,
        "h100_hours": float(previous.get("h100_hours", 0.0)) + elapsed * max(1, int(devices)) / 3600.0,
        "segments": int(previous.get("segments", 0)) + 1,
        "optimizer_steps": (
            max(int(previous.get("optimizer_steps", 0)), int(optimizer_steps))
            if optimizer_steps is not None
            else previous.get("optimizer_steps")
        ),
        "samples_processed": cumulative_samples,
        "samples_per_second": cumulative_samples / cumulative_elapsed if cumulative_elapsed > 0 else None,
        **_parameter_counts(model),
        "cuda_max_memory_allocated_gib": None,
        "cuda_max_memory_reserved_gib": None,
        "cuda_device_total_memory_gib": None,
        "distributed_world_size": max(1, int(devices)),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
    }
    if torch.cuda.is_available():
        payload["cuda_max_memory_allocated_gib"] = max(
            float(previous.get("cuda_max_memory_allocated_gib") or 0.0),
            torch.cuda.max_memory_allocated() / 1024**3,
        )
        payload["cuda_max_memory_reserved_gib"] = max(
            float(previous.get("cuda_max_memory_reserved_gib") or 0.0),
            torch.cuda.max_memory_reserved() / 1024**3,
        )
        payload["cuda_device_total_memory_gib"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, payload)
    return payload


def write_completion_validation_artifacts(
    run_dir,
    model,
    completion_started_at: float,
    *,
    checkpoint_path,
    checkpoint_sha256: str,
    checkpoint_step: int,
    checkpoint_epoch: int,
    global_batch_size: int,
    devices: int,
    source_git_commit: str,
    slurm_job_id: str | None,
    cuda_max_memory_allocated_gib: float | None,
    cuda_max_memory_reserved_gib: float | None,
    cuda_device_total_memory_gib: float | None,
) -> tuple[dict, dict]:
    """Finalize a zero-optimizer-step calibration validation segment.

    The original calibration process failed after its exact-step checkpoint, before
    ``write_resource_metrics`` could run. Recover the original segment wall time from
    its first durable "Starting model training" timestamp and the authoritative
    checkpoint mtime; the completion validation duration is recorded separately.
    """

    run_path = Path(run_dir)
    checkpoint = Path(checkpoint_path)
    log_path = run_path / "pretrain.log"
    if not log_path.is_file():
        raise FileNotFoundError(f"calibration log is missing: {log_path}")
    match = re.search(
        r"(?m)^(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}) Starting model training\s*$",
        log_path.read_text(errors="replace"),
    )
    if match is None:
        raise ValueError(f"could not recover calibration start timestamp from {log_path}")
    training_started = datetime.strptime(match.group(1), "%Y_%m_%d_%H_%M_%S")
    checkpoint_written = datetime.fromtimestamp(checkpoint.stat().st_mtime)
    training_elapsed = (checkpoint_written - training_started).total_seconds()
    if training_elapsed <= 0.0:
        raise ValueError(f"invalid calibration timing: checkpoint {checkpoint_written} is not after start {training_started}")

    completion_elapsed = max(0.0, time.perf_counter() - float(completion_started_at))
    samples_processed = int(checkpoint_step) * int(global_batch_size)
    seconds_per_optimizer_step = training_elapsed / int(checkpoint_step)
    projected_32k_seconds = seconds_per_optimizer_step * 32_000
    previous = {}
    previous_path = run_path / "resource_metrics.json"
    if previous_path.is_file():
        try:
            previous = json.loads(previous_path.read_text())
        except (OSError, json.JSONDecodeError):
            previous = {}
    training_peak_allocated = previous.get("cuda_max_memory_allocated_gib")
    training_peak_reserved = previous.get("cuda_max_memory_reserved_gib")

    def peak(first, second):
        values = [float(value) for value in (first, second) if value is not None]
        return max(values) if values else None

    resource_payload = {
        "elapsed_seconds": training_elapsed + completion_elapsed,
        "training_elapsed_seconds": training_elapsed,
        "completion_validation_seconds": completion_elapsed,
        "h100_hours": (training_elapsed + completion_elapsed) * max(1, int(devices)) / 3600.0,
        "segments": 2,
        "optimizer_steps": int(checkpoint_step),
        "optimizer_steps_in_completion": 0,
        "samples_processed": samples_processed,
        "samples_per_second": samples_processed / training_elapsed,
        "seconds_per_optimizer_step": seconds_per_optimizer_step,
        "projected_a0_32000_steps_seconds": projected_32k_seconds,
        "projected_a0_32000_steps_hours": projected_32k_seconds / 3600.0,
        **_parameter_counts(model),
        "cuda_max_memory_allocated_gib": peak(training_peak_allocated, cuda_max_memory_allocated_gib),
        "cuda_max_memory_reserved_gib": peak(training_peak_reserved, cuda_max_memory_reserved_gib),
        "validation_cuda_max_memory_allocated_gib": cuda_max_memory_allocated_gib,
        "validation_cuda_max_memory_reserved_gib": cuda_max_memory_reserved_gib,
        "cuda_device_total_memory_gib": cuda_device_total_memory_gib,
        "memory_measurement_phase": "completion_validation",
        "distributed_world_size": max(1, int(devices)),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
    }
    completion_payload = {
        "schema_version": "fomo26-calibration-completion-v1",
        "status": "ok",
        "mode": "validation_only",
        "optimizer_steps_executed": 0,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": str(checkpoint_sha256),
        "checkpoint_global_step": int(checkpoint_step),
        "checkpoint_epoch": int(checkpoint_epoch),
        "global_batch_size": int(global_batch_size),
        "devices": int(devices),
        "source_git_commit": str(source_git_commit),
        "slurm_job_id": None if slurm_job_id is None else str(slurm_job_id),
        "validation_metrics_written": True,
        "checkpoint_synchronization_completed": True,
        "resource_metrics_path": str(run_path / "resource_metrics.json"),
    }

    _atomic_write_json(run_path / "resource_metrics.json", resource_payload)
    _atomic_write_json(run_path / "calibration_completion.json", completion_payload)
    marker = run_path / "TRAINING_PAUSED"
    temporary_marker = marker.with_name(f".{marker.name}.tmp")
    temporary_marker.write_text(
        f"completion_validation_global_step={int(checkpoint_step)}\n"
        "optimizer_steps_executed=0\n"
        f"checkpoint_sha256={checkpoint_sha256}\n"
    )
    temporary_marker.replace(marker)
    return resource_payload, completion_payload
