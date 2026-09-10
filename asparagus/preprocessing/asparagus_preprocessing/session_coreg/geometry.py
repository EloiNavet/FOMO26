"""Voxel-geometry probing and session-reference selection.

Header reading uses nibabel only (no FreeSurfer), so both reference policies are
unit-testable without an MRI toolbox. Two policies are supported:

* ``fomo50k_legacy`` — minimum voxel *volume*, reproducing ``pre_process.sh``.
* ``anisotropy_aware`` — the FOMO26 default: penalises thick slices and
  anisotropy and (optionally) prefers structural modalities, so a
  0.49x0.49x6.5 mm FLAIR does not win over a reasonable isotropic T1w.
"""

import logging
import nibabel as nib
import numpy as np
import os
from asparagus_preprocessing.session_coreg import modalities
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ScanGeometry:
    """Header-derived geometry for a single scan."""

    path: str
    shape: tuple
    voxel_sizes: tuple  # mm, from the affine (matches FreeSurfer ``mri_info``)
    ndim: int
    #: Curated contrast from the canonical manifest, when the run is manifest-driven. It is
    #: authoritative: ``modalities.classify`` guesses from the filename and can be fooled by a
    #: subject id, which would then steer reference selection.
    modality_label: Optional[str] = None

    @property
    def voxel_volume(self) -> float:
        return float(np.prod(self.voxel_sizes))

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.shape[:3]))

    @property
    def max_spacing(self) -> float:
        return float(np.max(self.voxel_sizes))

    @property
    def min_spacing(self) -> float:
        return float(np.min(self.voxel_sizes))

    @property
    def anisotropy(self) -> float:
        lo = self.min_spacing
        return float(self.max_spacing / lo) if lo > 0 else float("inf")

    @property
    def modality(self) -> str:
        return self.modality_label or modalities.classify(self.path)

    @property
    def priority(self) -> int:
        """Reference priority: from the curated label when known, else from the filename."""
        if self.modality_label:
            return modalities.priority_rank_for(self.modality_label)
        return modalities.priority_rank(self.path)

    @property
    def is_4d(self) -> bool:
        return self.ndim > 3


def probe_geometry(path: str, modality_label: Optional[str] = None) -> ScanGeometry:
    """Read a scan's spatial geometry from its header without loading voxels."""
    img = nib.load(path)
    shape = tuple(int(s) for s in img.shape)
    voxel_sizes = tuple(float(v) for v in nib.affines.voxel_sizes(img.affine)[:3])
    return ScanGeometry(path=path, shape=shape, voxel_sizes=voxel_sizes, ndim=len(shape), modality_label=modality_label)


def probe_session(paths: List[str]) -> List[ScanGeometry]:
    geometries: List[ScanGeometry] = []
    for path in sorted(paths):
        try:
            geometries.append(probe_geometry(path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read geometry for %s: %s", path, exc)
    return geometries


def select_reference(geometries: List[ScanGeometry], config) -> Tuple[Optional[ScanGeometry], dict]:
    """Choose the session reference and return a full decision log.

    Returns ``(chosen, log)`` where ``log`` records the policy, every candidate's
    score, and the human-readable reason. ``chosen`` is ``None`` only when there
    are no 3D candidates.
    """
    candidates = [g for g in geometries if not g.is_4d]
    log = {
        "policy": config.reference_policy,
        "reference_string": config.reference_string,
        "max_reference_spacing": config.max_reference_spacing,
        "candidates": [],
        "reason": "",
    }
    if not candidates:
        log["reason"] = "no 3D candidates"
        return None, log

    # Explicit reference-string override wins under either policy.
    if config.reference_string:
        for geometry in sorted(candidates, key=lambda g: os.path.basename(g.path)):
            if config.reference_string in os.path.basename(geometry.path):
                log["reason"] = f"reference_string match: {config.reference_string!r}"
                log["candidates"] = [_candidate_row(geometry, config, chosen=True)]
                return geometry, log

    if config.reference_policy == "fomo50k_legacy":
        chosen = min(candidates, key=lambda g: (g.voxel_volume, -g.n_voxels, os.path.basename(g.path)))
        log["reason"] = "fomo50k_legacy: minimum voxel volume"
    else:
        chosen = _select_anisotropy_aware(candidates, config)
        log["reason"] = "anisotropy_aware: lowest penalty score"

    log["candidates"] = sorted(
        (_candidate_row(g, config, chosen=(g.path == chosen.path)) for g in candidates),
        key=lambda r: r["score"],
    )
    for row in log["candidates"]:
        logger.info(
            "reference candidate [%s] %s score=%.3f max_spacing=%.2f aniso=%.2f prio=%d%s",
            config.reference_policy,
            os.path.basename(row["path"]),
            row["score"],
            row["max_spacing"],
            row["anisotropy"],
            row["priority"],
            " <== CHOSEN" if row["chosen"] else "",
        )
    return chosen, log


def _score(geometry: ScanGeometry, config) -> float:
    """Anisotropy-aware penalty (lower is better)."""
    score = (
        config.w_max_spacing * geometry.max_spacing
        + config.w_anisotropy * (geometry.anisotropy - 1.0)
        + config.w_priority * geometry.priority
    )
    if geometry.max_spacing > config.max_reference_spacing:
        score += config.reference_gate_penalty
    return float(score)


def _select_anisotropy_aware(candidates: List[ScanGeometry], config) -> ScanGeometry:
    # Deterministic: score, then voxel volume, then name.
    return min(candidates, key=lambda g: (_score(g, config), g.voxel_volume, os.path.basename(g.path)))


def _candidate_row(geometry: ScanGeometry, config, chosen: bool) -> dict:
    return {
        "path": geometry.path,
        "modality": geometry.modality,
        "voxel_sizes": list(geometry.voxel_sizes),
        "voxel_volume_mm3": round(geometry.voxel_volume, 5),
        "max_spacing": round(geometry.max_spacing, 4),
        "anisotropy": round(geometry.anisotropy, 4),
        "priority": geometry.priority,
        "gated": bool(geometry.max_spacing > config.max_reference_spacing),
        "score": round(_score(geometry, config), 5),
        "chosen": bool(chosen),
    }
