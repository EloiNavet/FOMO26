"""Task-scoped inference-time geometry canonicalization.

Why this exists
---------------
``prepare_fomo26_asparagus.py`` stores every downstream corpus at native acquisition geometry
(``target_spacing: None``, ``target_orientation: "native"``), and no inference path resamples
anything. For a task whose corpus happens to hold exactly one geometry that combination is a trap:
the network is fitted at a single physical voxel scale, and an incoming volume at any other spacing
reaches it at a scale it has never seen. Measured on held-out folds, a 30% spacing deviation costs
Task 3 +3.8 MAE and 0.13 correlation; a 20% deviation costs Task 4 -0.06 mean DSC.

This module answers one question -- *what physical geometry was this task fitted at* -- from the
task registry, so the answer is declared once, per task, and is absent by default. Tasks that do
not declare a geometry resolve to ``None`` and keep the historical no-resampling behaviour exactly.

What it deliberately does not do
--------------------------------
It introduces no transform. The resampling itself is
:class:`asparagus.modules.transforms.spacing.Torch_ResampleToSpacing`, which already exists, is
already tested, and already records its inverse through ``size_before_resample`` so
``reverse_preprocessing`` restores a segmentation into the original input geometry. This module
only decides *whether* and *at what target* that existing stage is spliced in.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parent / "task_definitions.json"

#: Environment override for the orientation policy. The default refuses to canonicalize a volume
#: whose axis order differs from the one the task was fitted at, because a per-axis spacing target
#: applied under a different axis order silently resamples the wrong axes.
ORIENTATION_POLICY_ENV = "FOMO26_GEOMETRY_ORIENTATION_POLICY"
ORIENTATION_POLICIES = ("error", "skip")

#: Deployment override for the target itself: a spacing triplet, or ``none`` to switch
#: canonicalization off. It is read only for tasks that already declare a geometry.
TARGET_SPACING_ENV = "FOMO26_RUNTIME_TARGET_SPACING"


class RuntimeGeometryError(ValueError):
    """A geometry that must not be canonicalized silently."""


@lru_cache(maxsize=1)
def _task_definitions() -> dict:
    return json.loads(REGISTRY_PATH.read_text())["tasks"]


def task_runtime_geometry(task) -> dict | None:
    """Return the declared inference geometry for one task, or ``None`` if it declares none.

    ``task=None`` also resolves to ``None``: a caller that does not say which task it is cannot be
    given a task-scoped geometry, and the safe answer is the historical behaviour.
    """
    if task is None:
        return None
    definitions = _task_definitions()
    key = str(task)
    if key not in definitions:
        raise RuntimeGeometryError(
            f"Task {key!r} is not declared in {REGISTRY_PATH}. Refusing to guess whether it has a "
            "single fitted acquisition geometry."
        )
    entry = definitions[key]
    spacing = entry.get("runtime_target_spacing_mm")
    if spacing is None:
        return None
    spacing = [float(value) for value in spacing]
    if len(spacing) != 3 or any(not (value > 0) for value in spacing):
        raise RuntimeGeometryError(
            f"Task {key} declares an unusable runtime_target_spacing_mm={entry.get('runtime_target_spacing_mm')!r}; "
            "it must be three positive numbers in millimetres."
        )
    return {
        "target_spacing": spacing,
        "target_orientation": entry.get("runtime_target_orientation"),
        "rationale": entry.get("runtime_geometry_rationale", ""),
    }


def task_runtime_target_spacing(task) -> list[float] | None:
    """The per-axis target spacing for a task, or ``None`` when the task declares none."""
    geometry = task_runtime_geometry(task)
    return None if geometry is None else list(geometry["target_spacing"])


def orientation_policy() -> str:
    policy = os.environ.get(ORIENTATION_POLICY_ENV, "error").strip().lower()
    if policy not in ORIENTATION_POLICIES:
        raise RuntimeGeometryError(f"{ORIENTATION_POLICY_ENV}={policy!r} is not one of {ORIENTATION_POLICIES}.")
    return policy


def resolve_target_spacing_for_case(task, case_orientation, *, policy: str | None = None):
    """Decide the target spacing to apply to one case, honouring the orientation contract.

    A spacing target is expressed per array axis. Applying it to a volume stored in a different
    axis order resamples the wrong axes and produces a confidently wrong prediction that no metric
    flags, so a mismatch is refused rather than absorbed. ``policy="skip"`` downgrades the refusal
    to "leave this case at native geometry", which is exactly the historical behaviour and is the
    conservative choice for a submission container that must not crash a case.

    Returns ``(target_spacing_or_None, note)``.
    """
    geometry = task_runtime_geometry(task)
    if geometry is None:
        return None, "task declares no runtime geometry; native geometry kept"

    expected = geometry.get("target_orientation")
    if expected is None or case_orientation is None or str(case_orientation) == str(expected):
        return list(geometry["target_spacing"]), "canonicalizing to the fitted acquisition geometry"

    message = (
        f"Task {task} was fitted on {expected} volumes but this case is {case_orientation}. A "
        "per-axis spacing target applied under a different axis order resamples the wrong axes."
    )
    effective = policy or orientation_policy()
    if effective == "error":
        raise RuntimeGeometryError(message + f" Set {ORIENTATION_POLICY_ENV}=skip to fall back to native geometry instead.")
    return None, message + " Falling back to native geometry (historical behaviour)."


def _peek_orientation(path) -> str | None:
    """Axis codes of a NIfTI without reading its voxel data, or ``None`` if not determinable.

    ``nibabel`` loads lazily, so the affine is available from the header alone -- the orientation
    can therefore be checked before the transform pipeline is composed, which is where the target
    spacing has to be decided.
    """
    text = str(path)
    if not (text.endswith(".nii") or text.endswith(".nii.gz")):
        return None
    try:
        import nibabel as nib

        return "".join(nib.aff2axcodes(nib.load(text).affine))
    except Exception:
        # An unreadable header is not this function's error to raise: the loader that actually
        # reads the volume reports it with far better context a moment later.
        return None


def _env_override():
    """Parse the deployment override, or ``None`` when unset.

    Returns ``"disabled"``, a three-element spacing list, or ``None`` for "not set".
    """
    raw = os.environ.get(TARGET_SPACING_ENV)
    if raw is None or not raw.strip():
        return None
    text = raw.strip().lower()
    if text in {"none", "off", "null", "disabled"}:
        return "disabled"
    parts = [piece for piece in raw.replace(":", ",").split(",") if piece.strip()]
    try:
        spacing = [float(piece) for piece in parts]
    except ValueError as exc:
        raise RuntimeGeometryError(f"{TARGET_SPACING_ENV}={raw!r} is not a spacing triplet.") from exc
    if len(spacing) != 3 or any(not (value > 0) for value in spacing):
        raise RuntimeGeometryError(f"{TARGET_SPACING_ENV}={raw!r} must be three positive millimetre values, or 'none'.")
    return spacing


def resolve_for_inputs(task, data_paths, *, policy: str | None = None):
    """Target spacing to apply for one case, given the files that case will be assembled from.

    Returns ``(target_spacing_or_None, note)``. Orientation is taken from the first input, which
    is the volume whose affine the prediction is written back into.

    The deployment override is consulted only *after* the task has been shown to declare a
    geometry. That ordering is the scope guarantee: the override can retarget or switch off a task
    that opted in, and can never switch canonicalization on for a task that did not.
    """
    if task_runtime_geometry(task) is None:
        return None, "task declares no runtime geometry; native geometry kept"

    override = _env_override()
    if override == "disabled":
        return None, f"{TARGET_SPACING_ENV} disabled canonicalization; native geometry kept"

    paths = list(data_paths or [])
    orientation = _peek_orientation(paths[0]) if paths else None
    spacing, note = resolve_target_spacing_for_case(task, orientation, policy=policy)
    if spacing is not None and isinstance(override, list):
        return list(override), f"{note} ({TARGET_SPACING_ENV} override)"
    return spacing, note
