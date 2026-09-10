"""QC for the co-registration pipeline: strict output validation, metrics,
flat rows for aggregation, and visual montages (including before/after
registration)."""

import json
import logging
import numpy as np
import os
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

QC_SCHEMA_VERSION = 3


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def write_session_qc(qc: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(qc, handle, indent=2, sort_keys=True, default=_json_default)
    os.replace(tmp, path)  # atomic: a partial file never looks like a done session


def read_session_qc(path: str) -> Optional[dict]:
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


# --------------------------------------------------------------------------- #
# Image statistics & validation
# --------------------------------------------------------------------------- #
def brain_volume_cm3(mask_path: str) -> Optional[float]:
    import nibabel as nib

    try:
        img = nib.load(mask_path)
        data = np.asanyarray(img.dataobj)
        voxel_mm3 = float(np.prod(nib.affines.voxel_sizes(img.affine)[:3]))
        return float((data > 0).sum()) * voxel_mm3 / 1000.0
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not compute brain volume for %s: %s", mask_path, exc)
        return None


def image_stats(path: str) -> dict:
    """Load an output image and compute the statistics strict QC needs."""
    import nibabel as nib

    img = nib.load(path)
    data = np.asanyarray(img.dataobj).astype(np.float32)
    finite = np.isfinite(data)
    nonzero = finite & (data != 0)
    n = int(data.size)
    fg = data[nonzero]
    return {
        "shape": [int(s) for s in data.shape],
        "ndim": int(data.ndim),
        "affine": img.affine.tolist(),
        "voxel_sizes": [float(v) for v in nib.affines.voxel_sizes(img.affine)[:3]],
        "orientation": "".join(nib.aff2axcodes(img.affine)),
        "all_finite": bool(finite.all()),
        "nonzero_fraction": float(nonzero.sum() / n) if n else 0.0,
        "p50": float(np.percentile(fg, 50)) if fg.size else 0.0,
        "p99": float(np.percentile(fg, 99)) if fg.size else 0.0,
    }


def validate_output_image(path: str, config) -> Tuple[dict, List[str], List[str]]:
    """Return ``(stats, rejects, warns)`` for one final output image."""
    rejects: List[str] = []
    warns: List[str] = []
    if not os.path.exists(path):
        return {"exists": False}, ["missing_output"], warns
    try:
        stats = image_stats(path)
    except Exception as exc:  # noqa: BLE001
        return {"exists": True, "readable": False, "error": str(exc)}, ["unreadable_output"], warns
    stats["exists"] = True
    stats["readable"] = True

    if stats["ndim"] != 3:
        rejects.append(f"not_3d(ndim={stats['ndim']})")
    if not stats["all_finite"]:
        rejects.append("non_finite")
    if stats["nonzero_fraction"] <= 0:
        rejects.append("empty")
    if config.expected_orientation and stats["orientation"] != config.expected_orientation:
        rejects.append(f"orientation={stats['orientation']}")
    if config.target_iso_spacing is not None:
        target = np.asarray(config.target_iso_spacing)
        if not np.allclose(stats["voxel_sizes"], target, atol=config.qc.spacing_tol_mm):
            rejects.append(f"spacing={[round(v, 3) for v in stats['voxel_sizes']]}")
    nz = stats["nonzero_fraction"]
    if nz < config.qc.min_nonzero_fraction:
        warns.append(f"low_nonzero_fraction({nz:.4f})")
    if nz > config.qc.max_nonzero_fraction:
        warns.append(f"high_nonzero_fraction({nz:.4f})")
    return stats, rejects, warns


def image_under_test(mod: dict) -> str:
    """The image this modality's output QC should read.

    Before publication the finished image is still the work-dir candidate, and that is what must
    be judged -- publication is what passing this check earns. Afterwards the candidate has been
    renamed onto the canonical path, so re-validating an already-published session (or reading a
    session written by an older run that had no candidates) falls back to ``output_path``.
    """
    candidate = mod.get("candidate_path") or ""
    if candidate and os.path.exists(candidate):
        return candidate
    return mod.get("output_path", "") or candidate


def validate_session_outputs(qc: dict, config) -> dict:
    """Validate all outputs of a session; annotate ``qc`` and return a summary.

    Downgrades a modality to ``failed`` if it fails a rejection-level check, and
    records session-level shared-shape/affine and brain-mask checks. Also emits
    ``flags`` used to select sessions for visual QC.
    """
    per_image = {}
    session_rejects: List[str] = []
    session_warns: List[str] = []
    shapes, affines = [], []

    for mod in qc.get("modalities", []):
        if mod.get("status") != "ok":
            continue
        stats, rejects, warns = validate_output_image(image_under_test(mod), config)
        mod["p50"] = stats.get("p50")
        mod["p99"] = stats.get("p99")
        mod["nonzero_fraction"] = stats.get("nonzero_fraction")
        mod["output_orientation"] = stats.get("orientation")
        mod["output_spacing"] = stats.get("voxel_sizes")
        mod["output_shape"] = stats.get("shape")
        per_image[mod["name"]] = {"rejects": rejects, "warns": warns}
        if rejects:
            mod["status"] = "failed"
            mod["reason"] = (mod.get("reason", "") + "; output_qc:" + ",".join(rejects)).strip("; ")
        else:
            if stats.get("shape"):
                shapes.append(tuple(stats["shape"]))
                affines.append(np.asarray(stats["affine"]))
        session_warns.extend(f"{mod['name']}:{w}" for w in warns)

    # Shared geometry across the session's surviving outputs.
    #
    # This is the co-registration contract, so it only applies where co-registration was actually
    # performed. A session the identity audit refused to register, or a single-scan session, is
    # passed through on each scan's own P0 grid *by design*: its scans were never promised a
    # common lattice, and reporting shape_mismatch against them describes the intended behaviour
    # as a defect.
    if qc.get("coregistration_applied", True):
        if len({s for s in shapes}) > 1:
            session_rejects.append(f"shape_mismatch:{sorted(set(shapes))}")
        if affines and not all(np.allclose(a, affines[0], atol=1e-3) for a in affines):
            session_rejects.append("affine_mismatch")

    # Brain mask checks.
    flags = {}
    bm = qc.get("brain_mask", {})
    if config.do_skull_strip and config.save_brain_mask:
        mask_path = bm.get("path")
        if not mask_path or not os.path.exists(mask_path):
            session_rejects.append("missing_brain_mask")
        else:
            vol = bm.get("brain_volume_cm3")
            if vol is not None:
                if vol < config.qc.min_brain_volume_cm3:
                    session_warns.append(f"small_brain({vol:.0f}cm3)")
                    flags["small_mask"] = True
                if vol > config.qc.max_brain_volume_cm3:
                    session_warns.append(f"large_brain({vol:.0f}cm3)")
                    flags["large_mask"] = True
            frac = _brain_fraction(qc, config)
            if frac is not None:
                bm["brain_fraction"] = round(frac, 4)
                if frac < config.qc.min_brain_fraction:
                    session_warns.append(f"low_brain_fraction({frac:.3f})")

    # Registration-motion / NMI warnings feed the suspicious-session flags.
    for mod in qc.get("modalities", []):
        reg = mod.get("registration", {})
        if reg.get("translation_norm_mm", 0) > config.qc.translation_warn_mm:
            session_warns.append(f"{mod['name']}:large_translation")
            flags["large_translation"] = True
        if reg.get("rotation_deg", 0) > config.qc.rotation_warn_deg:
            session_warns.append(f"{mod['name']}:large_rotation")
            flags["large_rotation"] = True
        imp = reg.get("nmi_improvement")
        if imp is not None and imp < nmi_warn_threshold(mod, config):
            session_warns.append(f"{mod['name']}:nmi_worsened({imp:.3f})")
            flags["worst_nmi"] = True
        retained = mod.get("foreground_retained_frac")
        if retained is not None and retained < config.qc.min_foreground_retained_frac:
            session_warns.append(f"{mod['name']}:clipped_by_reference_fov({retained:.3f})")
            flags["fov_clipped"] = True

        # Zero overlap is REJECTION level, and it is not a retention threshold: it says that none
        # of the source-defined foreground exists on the target lattice, so the published volume
        # holds no anatomy at all. That is categorically different from the clipping warning
        # above, which cannot tell lost brain from lost non-brain coverage and so stays a warning.
        #
        # Decided on the voxel counts, never on foreground_retained_frac, which is rounded for
        # reporting; and never on "is the image nonzero", because a spline resample of a
        # non-overlapping volume leaves dust of order 1e-11 that is nonzero and means nothing.
        #
        # It is checked for every modality regardless of how the output was produced. An optimised
        # fit that lands in a wrong basin can miss the reference just as completely as a header
        # alignment between two scans whose headers disagree.
        src_foreground = mod.get("source_foreground_voxel_count")
        out_foreground = mod.get("output_foreground_voxel_count")
        if (
            mod.get("status") == "ok"
            and src_foreground is not None
            and out_foreground is not None
            and src_foreground > 0
            and out_foreground == 0
        ):
            mod["status"] = "failed"
            mod["reason"] = (mod.get("reason", "") + "; output_qc:zero_foreground_overlap").strip("; ")
            session_rejects.append(f"{mod['name']}:zero_foreground_overlap")
            flags["zero_foreground_overlap"] = True

        # Rigid sanity is REJECTION level: a reflection, scale or shear means the transform
        # absorbed anatomy, and the output is not what "rigid co-registration" promised.
        #
        # The flag is named for what the check actually establishes. A 6-DOF fit that lands in a
        # wrong basin is still an exactly rigid matrix -- determinant 1 to machine precision --
        # it is its *magnitude* that is implausible, so calling it "non rigid" misdescribes the
        # evidence for every reader downstream.
        if mod.get("status") == "ok":
            violations = rigid_violations(reg, config)
            if violations:
                mod["status"] = "failed"
                mod["reason"] = (mod.get("reason", "") + "; transform_qc:" + ",".join(violations)).strip("; ")
                session_rejects.extend(f"{mod['name']}:{reason}" for reason in violations)
                flags["implausible_rigid_transform"] = True

    summary = {
        "per_image": per_image,
        "session_rejects": session_rejects,
        "session_warns": session_warns,
        "n_reject": len(session_rejects) + sum(1 for v in per_image.values() if v["rejects"]),
        "n_warn": len(session_warns),
        "flags": flags,
    }
    qc["output_qc"] = summary
    return summary


def is_diffusion(mod: dict) -> bool:
    """True when a modality entry is a diffusion contrast (ADC / DWI_*)."""
    from asparagus_preprocessing.session_coreg.config import DWI_MODALITIES

    label = str(mod.get("modality_canonical") or mod.get("modality") or "").upper()
    return any(label == m.upper() or label.startswith("DWI") for m in DWI_MODALITIES)


def nmi_threshold_for(mod: dict, config) -> float:
    """Similarity threshold for one modality.

    Diffusion gets its own, looser threshold because rigid registration cannot correct EPI
    distortion: a modest NMI drop on ADC/DWI is expected physics, not a failed fit. Making this
    explicit is the alternative to quietly weakening the global threshold for everyone.
    """
    return config.qc.nmi_improvement_warn_dwi if is_diffusion(mod) else config.qc.nmi_improvement_warn


# Backwards-compatible alias used by :func:`validate_session_outputs`.
nmi_warn_threshold = nmi_threshold_for


def rigid_violations(reg: dict, config) -> List[str]:
    """Rejection reasons for a transform that is not a rigid body motion.

    Only checks what was actually measured: a transform whose metrics could not be parsed, or
    which was decomposed in FSL space rather than world space, is not judged here (the shared
    shape/affine check still applies to its output).
    """
    reasons: List[str] = []
    if not reg or not reg.get("parsed", True):
        return reasons
    if reg.get("world_space") is False and "fsl_matrix" not in reg:
        # A raw FSL-space decomposition has an ambiguous determinant sign; do not fail on it.
        return reasons

    det = reg.get("determinant")
    if det is not None:
        if det < 0:
            reasons.append(f"reflection(det={det:.4f})")
        elif abs(det - 1.0) > config.qc.rigid_determinant_tol:
            reasons.append(f"volume_change(det={det:.4f})")
    scale_err = reg.get("max_scale_error")
    if scale_err is not None and scale_err > config.qc.rigid_scale_tol:
        reasons.append(f"scale(max_scale_error={scale_err:.4g})")
    ortho = reg.get("orthogonality_error")
    if ortho is not None and ortho > config.qc.rigid_orthogonality_tol:
        reasons.append(f"shear(orthogonality_error={ortho:.4g})")
    translation = reg.get("translation_norm_mm")
    if translation is not None and translation > config.qc.translation_reject_mm:
        reasons.append(f"implausible_translation({translation:.1f}mm)")
    rotation = reg.get("rotation_deg")
    if rotation is not None and rotation > config.qc.rotation_reject_deg:
        reasons.append(f"implausible_rotation({rotation:.1f}deg)")
    return reasons


def _brain_fraction(qc: dict, config) -> Optional[float]:
    """Brain-mask voxels divided by the reference output's nonzero voxels."""
    import nibabel as nib

    mask_path = qc.get("brain_mask", {}).get("path")
    ref_mod = next((m for m in qc.get("modalities", []) if m.get("is_reference") and m.get("status") == "ok"), None)
    # The reference is judged before it is published, so read whichever copy exists right now.
    ref_path = image_under_test(ref_mod) if ref_mod else ""
    if not mask_path or not ref_path or not os.path.exists(mask_path) or not os.path.exists(ref_path):
        return None
    try:
        mask = np.asanyarray(nib.load(mask_path).dataobj) > 0
        ref = np.asanyarray(nib.load(ref_path).dataobj)
        nz = int((ref != 0).sum())
        return float(mask.sum() / nz) if nz else None
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Flat rows for the aggregate TSV
# --------------------------------------------------------------------------- #
def session_qc_rows(qc: dict) -> List[dict]:
    common = {
        "session_key": qc.get("session_key"),
        "dataset": qc.get("dataset"),
        "subject": qc.get("subject"),
        "session": qc.get("session"),
        "status": qc.get("status"),
        "reason": qc.get("reason", ""),
        "reference": qc.get("reference", {}).get("name"),
        "reference_policy": qc.get("reference", {}).get("policy"),
        "reference_selection": qc.get("reference", {}).get("reason"),
        "n_scans": qc.get("n_scans"),
        "brain_volume_cm3": qc.get("brain_mask", {}).get("brain_volume_cm3"),
        "n_output_reject": qc.get("output_qc", {}).get("n_reject"),
        "n_output_warn": qc.get("output_qc", {}).get("n_warn"),
    }
    modalities = qc.get("modalities", [])
    if not modalities:
        return [{**common, "modality": None}]
    rows = []
    for mod in modalities:
        reg = mod.get("registration", {})
        rows.append(
            {
                **common,
                "modality": mod.get("name"),
                "modality_canonical": mod.get("modality_canonical", ""),
                "is_reference": mod.get("is_reference"),
                "modality_status": mod.get("status"),
                "modality_reason": mod.get("reason", ""),
                "source_voxel_volume_mm3": _round(mod.get("source_voxel_volume")),
                "source_shape": _fmt(mod.get("source_shape")),
                "coreg_cost": _round(mod.get("coreg_cost")),
                "translation_norm_mm": _round(reg.get("translation_norm_mm")),
                "rotation_deg": _round(reg.get("rotation_deg")),
                "determinant": _round(reg.get("determinant"), 6),
                "orthogonality_error": _round(reg.get("orthogonality_error"), 8),
                "nmi_before": _round(reg.get("nmi_before")),
                "nmi_after": _round(reg.get("nmi_after")),
                "nmi_improvement": _round(reg.get("nmi_improvement")),
                # Continuous diagnostic, deliberately uncalibrated. See QCThresholds.
                "foreground_retained_frac": _round(mod.get("foreground_retained_frac")),
                "output_path": mod.get("output_path"),
            }
        )
    return rows


def _round(value, ndigits: int = 4):
    return round(value, ndigits) if isinstance(value, (int, float)) else value


def _fmt(seq):
    return "x".join(str(int(v)) for v in seq) if seq else ""


# --------------------------------------------------------------------------- #
# Visual QC
# --------------------------------------------------------------------------- #
def _take_slice(volume, axis: int, index: int):
    if volume is None:
        return None
    index = min(max(index, 0), volume.shape[axis] - 1)
    return np.take(volume, index, axis=axis)


def _load(path):
    import nibabel as nib

    return np.squeeze(np.asanyarray(nib.load(path).dataobj).astype(np.float32))


def render_registration_qc(
    ref_path: str,
    moving_before: Optional[str],
    moving_after: str,
    mask_path: Optional[str],
    out_png: str,
    label: str = "",
    title: str = "",
) -> bool:
    """Reference / moving-before / moving-after with reference-edge and mask overlays.

    The reference's edges (green) are overlaid on the moving image before and
    after registration so misalignment is obvious; the brain-mask contour (red)
    checks the skull-strip. Returns ``True`` if a PNG was written.

    All three orthogonal planes are rendered. Axial alone cannot answer the question the
    reference-FOV clipping metric raises — whether the foreground a moving scan lost was brain
    or merely wider non-brain acquisition coverage. That is a superior/inferior and
    anterior/posterior question, so coronal and sagittal have to be visible too.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        ref = _load(ref_path)
        after = _load(moving_after)
    except Exception as exc:  # noqa: BLE001
        logger.warning("registration QC load failed (%s): %s", out_png, exc)
        return False
    before = None
    if moving_before and os.path.exists(moving_before):
        try:
            before = _load(moving_before)
        except Exception:  # noqa: BLE001
            before = None
    mask = None
    if mask_path and os.path.exists(mask_path):
        try:
            mask = _load(mask_path) > 0
        except Exception:  # noqa: BLE001
            mask = None

    ref_edges = _edge_map(ref)
    cols = [
        ("reference", ref, None, mask),
        ("moving before", before if before is not None else after, ref_edges, None),
        ("moving after", after, ref_edges, mask),
    ]
    axes_names = ["axial", "coronal", "sagittal"]
    planes = [2, 1, 0]
    fig, axes = plt.subplots(len(planes), len(cols), figsize=(3 * len(cols), 3 * len(planes)), squeeze=False)
    for r, plane in enumerate(planes):
        for c, (name, vol, edges, msk) in enumerate(cols):
            ax = axes[r][c]
            ax.axis("off")
            if vol is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center")
                continue
            idx = vol.shape[plane] // 2
            sl = _take_slice(vol, plane, idx)
            vmax = np.percentile(sl, 99) if np.any(sl) else 1.0
            ax.imshow(np.rot90(sl), cmap="gray", vmin=0, vmax=max(vmax, 1e-6))
            if edges is not None:
                esl = _take_slice(edges, plane, min(idx, edges.shape[plane] - 1))
                ax.contour(np.rot90(esl), levels=[0.5], colors="lime", linewidths=0.5)
            if msk is not None:
                msl = _take_slice(msk, plane, min(idx, msk.shape[plane] - 1))
                if msl is not None and msl.any():
                    ax.contour(np.rot90(msl), levels=[0.5], colors="red", linewidths=0.6)
            if r == 0:
                ax.set_title(name, fontsize=9)
            if c == 0:
                ax.text(-0.05, 0.5, axes_names[r], transform=ax.transAxes, rotation=90, va="center", fontsize=8)
    fig.suptitle((title + "  " + label).strip(), fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return True


def _edge_map(volume: np.ndarray, percentile: float = 92.0) -> np.ndarray:
    grads = np.gradient(volume.astype(np.float64))
    mag = np.sqrt(np.sum([g**2 for g in grads], axis=0))
    fg = mag[volume != 0]
    thr = np.percentile(fg, percentile) if fg.size else 0.0
    return (mag >= thr).astype(np.float32)


def render_qc_montage(
    image_paths: List[str], labels: List[str], mask_path: Optional[str], out_png: str, max_modalities: int = 4, title: str = ""
) -> bool:
    """Orthogonal mid-slices per modality with the brain-mask contour (fallback view)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = list(zip(image_paths, labels))[:max_modalities]
    if not pairs:
        return False
    mask = None
    if mask_path and os.path.exists(mask_path):
        try:
            mask = _load(mask_path) > 0
        except Exception:  # noqa: BLE001
            mask = None
    fig, axes = plt.subplots(len(pairs), 3, figsize=(9, 3 * len(pairs)), squeeze=False)
    for row, (path, label) in enumerate(pairs):
        try:
            data = _load(path)
        except Exception as exc:  # noqa: BLE001
            for col in range(3):
                axes[row][col].text(0.5, 0.5, f"load failed\n{exc}", ha="center", va="center", fontsize=7)
                axes[row][col].axis("off")
            continue
        centers = [s // 2 for s in data.shape[:3]]
        for col in range(3):
            sl = _take_slice(data, col, centers[col])
            msl = _take_slice(mask, col, centers[col]) if mask is not None else None
            ax = axes[row][col]
            vmax = np.percentile(sl, 99) if np.any(sl) else 1.0
            ax.imshow(np.rot90(sl), cmap="gray", vmin=0, vmax=max(vmax, 1e-6))
            if msl is not None and msl.any():
                ax.contour(np.rot90(msl), levels=[0.5], colors="red", linewidths=0.6)
            ax.axis("off")
            ax.set_title(f"{label} — {['sagittal', 'coronal', 'axial'][col]}", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return True
