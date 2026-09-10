"""Per-session orchestration: RAS -> co-register -> skull-strip -> 1 mm iso.

Faithful to FOMO50K ``pre_process.sh`` (co-register every scan to a single
reference, SynthSeg mask on the reference shared across modalities), with FOMO26
hardening: anisotropy-aware reference selection, unified interpolation, strict
output QC + registration metrics, full command provenance, explicit 4D handling,
and safe never-overwrite I/O.

Statuses: ``success`` (all expected modalities ok and output QC clean),
``partial`` (some ok), ``failed`` (nothing valid / critical error), ``empty``
(no usable input), ``dry_run`` (commands only simulated).
"""

import logging
import numpy as np
import os
import shutil
import time
from asparagus_preprocessing.session_coreg import metrics as metrics_mod, modalities, provenance, qc as qc_mod
from asparagus_preprocessing.session_coreg.backend import BackendError
from asparagus_preprocessing.session_coreg.config import CoregConfig
from asparagus_preprocessing.session_coreg.discovery import Session
from asparagus_preprocessing.session_coreg.freesurfer import FreeSurferError, FreeSurferRunner
from asparagus_preprocessing.session_coreg.geometry import ScanGeometry, probe_geometry, select_reference
from asparagus_preprocessing.session_coreg.paths import assert_safe_io
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class FourDInputError(RuntimeError):
    """Raised when 4D scans are present and ``fail_on_4d`` is set."""


def _safe_stem(session: Session, path: str) -> str:
    rel = session.relpath(path)
    for ext in (".nii.gz", ".nii", ".mgz"):
        if rel.lower().endswith(ext):
            rel = rel[: -len(ext)]
            break
    return rel.replace(os.sep, "__")


def output_path_for(scan_path: str, input_root: str, output_root: str) -> str:
    return os.path.join(output_root, os.path.relpath(scan_path, input_root))


def session_out_dir(output_root: str, session: Session) -> str:
    return os.path.join(output_root, session.key)


def process_session(
    session: Session,
    config: CoregConfig,
    input_root: str,
    output_root: str,
    runner: Optional[FreeSurferRunner] = None,
    render_qc_image: bool = False,
    qc_image_dir: Optional[str] = None,
) -> dict:
    """Process one session end-to-end and return its QC record. Never raises."""
    input_root, output_root = assert_safe_io(input_root, output_root)

    runner = runner or FreeSurferRunner(
        threads=config.threads, freesurfer_home=config.freesurfer_home, max_log_chars=config.max_log_chars
    )
    out_dir = session_out_dir(output_root, session)
    work_dir = os.path.join(out_dir, "_work")
    qc: dict = {
        "schema_version": qc_mod.QC_SCHEMA_VERSION,
        "session_key": session.key,
        "dataset": session.dataset,
        "subject": session.subject,
        "session": session.session,
        "config": _config_snapshot(config),
        "n_scans": len(session.image_paths),
        "modalities": [],
        "timings_s": {},
        "status": "failed",
        "reason": "",
        "provenance": provenance.environment_provenance(config.freesurfer_home or ""),
    }
    started = time.perf_counter()
    state: dict = {}
    try:
        os.makedirs(work_dir, exist_ok=True)
        _run_session(session, config, input_root, output_root, runner, work_dir, qc, state)
    except FourDInputError as exc:
        qc["status"] = "failed"
        qc["reason"] = f"4d_input: {exc}"
        logger.error("Session %s has 4D input: %s", session.key, exc)
    except BackendError as exc:
        qc["status"] = "failed"
        qc["reason"] = f"backend: {exc}"
        logger.error("Session %s failed: %s", session.key, exc)
    except Exception as exc:  # noqa: BLE001 - one bad session must not kill the pool
        qc["status"] = "failed"
        qc["reason"] = f"{type(exc).__name__}: {exc}"
        logger.exception("Session %s crashed", session.key)
    finally:
        if runner.dry_run:
            qc["status"] = "dry_run"
            qc["reason"] = "dry-run: commands simulated, no outputs written"
        qc["timings_s"]["total"] = round(time.perf_counter() - started, 3)
        qc["commands"] = {
            "count": len(runner.command_log),
            "total_duration_s": round(sum(c["duration_s"] for c in runner.command_log), 3),
        }
        _write_commands(runner, out_dir)
        if render_qc_image and qc_image_dir and not runner.dry_run:
            _maybe_render_qc(qc, config, state, qc_image_dir, forced=render_qc_image)
        _cleanup_work(config, runner, qc, work_dir)
        qc_mod.write_session_qc(qc, os.path.join(out_dir, "qc.json"))
    return qc


def _config_snapshot(config: CoregConfig) -> dict:
    return {
        "reference_policy": config.reference_policy,
        "reference_string": config.reference_string,
        "max_reference_spacing": config.max_reference_spacing,
        "do_coregister": config.do_coregister,
        "do_skull_strip": config.do_skull_strip,
        "synthseg_robust": config.synthseg_robust,
        "target_iso_spacing": list(config.target_iso_spacing) if config.target_iso_spacing else None,
        "image_interp": config.image_interp,
        "mask_interp": config.mask_interp,
        "fail_on_4d": config.fail_on_4d,
        "compute_registration_metrics": config.compute_registration_metrics,
        "backend": config.backend,
        "dof": config.dof,
        "cost": config.cost,
        "init_from_header": config.init_from_header,
        # Recorded even when None, so a run's provenance always states which optimiser search
        # produced it and the paired ablation can be told apart from the baseline by file alone.
        "angular_search_deg": list(config.angular_search_deg) if config.angular_search_deg else None,
        "fallback_unconstrained_retry": config.fallback_unconstrained_retry,
        "inverse_direction_rescue": config.inverse_direction_rescue,
        "header_only_fallback": config.header_only_fallback,
    }


def _run_session(session, config, input_root, output_root, runner, work_dir, qc, state) -> None:
    ras_dir = os.path.join(work_dir, "ras")
    os.makedirs(ras_dir, exist_ok=True)

    # 1. Probe geometry; classify 3D vs 4D. Never silently drop 4D.
    usable: List[ScanGeometry] = []
    dropped: List[dict] = []
    four_d: List[str] = []
    for path in session.image_paths:
        sample = session.sample_for(path)
        try:
            geom = probe_geometry(path, modality_label=sample.get("modality_canonical") or None)
        except Exception as exc:  # noqa: BLE001
            dropped.append({"path": path, "reason": f"unreadable: {exc}"})
            continue
        divergence = _manifest_geometry_divergence(geom, sample)
        if divergence:
            # The header is authoritative, but a manifest that disagrees with it means the two
            # describe different files. Grouping such a scan into a session would register data
            # the corpus definition does not actually describe, so it is refused.
            dropped.append({"path": path, "reason": f"manifest_geometry_mismatch: {divergence}"})
            continue
        if geom.is_4d:
            four_d.append(path)
        else:
            usable.append(geom)
    qc["dropped"] = dropped
    qc["four_d_scans"] = four_d
    if four_d and config.fail_on_4d:
        raise FourDInputError(f"{len(four_d)} 4D scan(s): {four_d}")
    for path in four_d:  # only reached when fail_on_4d is False
        dropped.append({"path": path, "reason": "4d_dropped (fail_on_4d disabled)"})

    if not usable:
        qc["status"] = "empty"
        qc["reason"] = "no usable 3D scans in session"
        return

    # 2. Reference selection (policy-driven, fully logged).
    reference, ref_log = select_reference(usable, config)
    qc["reference"] = {
        "name": modalities.suffix(reference.path),
        "modality": reference.modality,
        "source_path": reference.path,
        "voxel_sizes": list(reference.voxel_sizes),
        "voxel_volume_mm3": round(reference.voxel_volume, 5),
        "max_spacing": round(reference.max_spacing, 4),
        "anisotropy": round(reference.anisotropy, 4),
        "shape": list(reference.shape),
        "policy": ref_log["policy"],
        "reason": ref_log["reason"],
        "candidates": ref_log["candidates"],
        "anisotropic_reference": bool(reference.max_spacing > config.max_reference_spacing),
    }

    # 3. Reorient to RAS.
    t0 = time.perf_counter()
    ras_paths: Dict[str, str] = {}
    for geom in usable:
        ras = os.path.join(ras_dir, _safe_stem(session, geom.path) + ".nii.gz")
        runner.reorient_to_ras(geom.path, ras)
        ras_paths[geom.path] = ras
    qc["timings_s"]["ras"] = round(time.perf_counter() - t0, 3)

    ref_ras = ras_paths[reference.path]
    # A session the identity audit could not vouch for is processed as passthrough: its scans are
    # still materialised at RAS/1 mm, they are simply never aligned to each other.
    do_coreg = config.do_coregister and len(usable) >= config.min_scans_for_coreg and session.coreg_allowed
    qc["coregistration_applied"] = do_coreg
    qc["coreg_allowed"] = session.coreg_allowed
    qc["coreg_block_reason"] = session.coreg_block_reason
    iso = config.target_iso_spacing is not None

    # 4. Target grid (reference resampled to iso, or its own RAS grid).
    if iso and do_coreg:
        target_grid = os.path.join(work_dir, "ref_iso.nii.gz")
        runner.make_iso_target(ref_ras, target_grid, config.target_iso_spacing, interp=config.image_interp)
    else:
        target_grid = ref_ras

    # 5. Bring every scan into the target space; compute registration metrics.
    t0 = time.perf_counter()
    pre_mask: Dict[str, str] = {}
    reg_artifacts: Dict[str, dict] = {}
    for geom in usable:
        sample = session.sample_for(geom.path)
        mod = {
            "name": modalities.suffix(geom.path),
            "modality": geom.modality,
            # Canonical identity, present only for manifest-driven runs. Carried into QC so the
            # derivative manifest joins on sample_id rather than on a path, and so diffusion-
            # specific thresholds key off the curated label instead of a filename guess.
            "sample_id": sample.get("sample_id", ""),
            "modality_canonical": sample.get("modality_canonical", ""),
            "source_path": geom.path,
            "source_shape": list(geom.shape),
            "source_voxel_sizes": list(geom.voxel_sizes),
            "source_voxel_volume": round(geom.voxel_volume, 5),
            "is_reference": geom.path == reference.path,
            # ``registered`` means an optimised transform was fitted AND accepted -- not "this scan
            # went through the registration path". ``transform_fitted`` says the same thing about
            # the transform's provenance and is the field downstream consumers should read, because
            # a transform_path can exist without an optimiser ever having run (header alignment).
            "registered": False,
            "transform_fitted": False,
            "registration_method": "none",
            # Whether the attempt ladder ran at all. References and passthrough scans never enter
            # it, and that is what separates them from a moving scan whose every attempt failed.
            "coregistration_attempted": False,
            "transform_path": "",
            "coreg_cost": None,
            "registration": {},
            "status": "ok",
            "reason": "",
            "output_path": output_path_for(geom.path, input_root, output_root),
        }
        out_pre = os.path.join(work_dir, _safe_stem(session, geom.path) + "_pre.nii.gz")
        try:
            artifacts = _bring_into_space(
                geom,
                reference,
                ras_paths[geom.path],
                ref_ras,
                target_grid,
                out_pre,
                work_dir,
                session,
                config,
                runner,
                mod,
                do_coreg,
            )
            pre_mask[geom.path] = out_pre
            reg_artifacts[geom.path] = artifacts
        except BackendError as exc:
            mod["status"] = "failed"
            mod["reason"] = str(exc)[:500]
        qc["modalities"].append(mod)
    qc["timings_s"]["register"] = round(time.perf_counter() - t0, 3)

    if not pre_mask:
        qc["status"] = "failed"
        qc["reason"] = "all scans failed registration/resampling"
        return

    # 6. Skull-strip (shared reference mask when co-registered) and write outputs.
    t0 = time.perf_counter()
    brain_mask_out = (
        os.path.join(session_out_dir(output_root, session), "brainmask.nii.gz") if config.save_brain_mask else None
    )
    _skull_strip_and_write(session, config, runner, work_dir, target_grid, reference, pre_mask, qc, do_coreg, brain_mask_out)
    qc["timings_s"]["skull_strip"] = round(time.perf_counter() - t0, 3)

    # 7. Strict output QC (dry-run has no outputs to validate).
    state["reg_artifacts"] = reg_artifacts
    state["target_grid"] = target_grid
    state["reference_path"] = reference.path
    qc["session_output_dir"] = session_out_dir(output_root, session)
    if not runner.dry_run:
        t0 = time.perf_counter()
        qc_mod.validate_session_outputs(qc, config)
        qc["timings_s"]["output_qc"] = round(time.perf_counter() - t0, 3)
        # Only now, with every candidate judged, may anything reach a canonical path.
        _publish_outputs(qc, config, runner, state)

    _finalize_status(qc, config, runner)


def _bring_into_space(
    geom, reference, ras, ref_ras, target_grid, out_pre, work_dir, session, config, runner, mod, do_coreg
) -> dict:
    """Produce one scan's pre-mask output on the target grid; return QC artifacts."""
    is_ref = geom.path == reference.path
    iso = config.target_iso_spacing is not None
    artifacts: dict = {"after": out_pre, "before": None}

    if do_coreg and not is_ref:
        _register_with_fallback(geom, ras, ref_ras, target_grid, out_pre, work_dir, session, config, runner, mod, artifacts)
    elif do_coreg and is_ref:
        _copy_if_exists(target_grid if iso else ref_ras, out_pre)
    else:  # no coregistration: iso-resample each scan in its own frame
        if iso:
            runner.make_iso_target(ras, out_pre, config.target_iso_spacing, interp=config.image_interp)
        else:
            _copy_if_exists(ras, out_pre)
    return artifacts


#: Attempt 1 uses whatever the runner is already configured with; passing nothing keeps that the
#: single source of truth. Only the retry overrides it, and only to remove the bound.
_RUNNER_DEFAULT = object()

#: Attempt kinds. ``FORWARD`` estimates moving -> target grid, which is what every published P1
#: registration to date used. ``INVERSE_RESCUE`` estimates the same rigid fit with the two images
#: swapped and inverts it; the *result* still means moving -> target grid, so everything
#: downstream -- application, metrics, QC, publication -- is byte-for-byte the same operation.
FORWARD = "forward"
INVERSE_RESCUE = "inverse_direction_rescue"
#: Not an estimation at all: publish the scan where its scanner header places it. Always last.
HEADER_ONLY = "header_only"


def _attempt_ladder(config) -> List[dict]:
    """The attempt ladder, in order.

    ``search`` is ``_RUNNER_DEFAULT`` ("use the runner's own bound") or ``None`` ("unconstrained").
    The unconstrained retry is only meaningful when attempt 1 is bounded: without a bound the two
    attempts would issue an identical command, so there is nothing to fall back to.

    The inverse-direction rescue always comes last and always carries the runner's own angular
    bound, so it differs from attempt 1 in exactly one variable -- the estimation direction.
    """
    ladder: List[dict] = [{"type": FORWARD, "search": _RUNNER_DEFAULT}]
    if config.fallback_unconstrained_retry and config.angular_search_deg is not None:
        ladder.append({"type": FORWARD, "search": None})
    if getattr(config, "inverse_direction_rescue", False):
        ladder.append({"type": INVERSE_RESCUE, "search": _RUNNER_DEFAULT})
    if getattr(config, "header_only_fallback", False):
        # Always last: it optimises nothing, so it can only ever be the answer once every
        # optimised attempt has already been judged and lost.
        ladder.append({"type": HEADER_ONLY, "search": None})
    return ladder


def _estimate_inverse_direction(ras, reg_ref, transform, reverse_matrix, runner, policy):
    """Fit ``target grid -> moving``, then invert so ``transform`` still means ``moving -> grid``.

    The swap is the whole intervention. Both images keep their roles everywhere else: ``ras`` is
    still the original moving NIfTI, ``reg_ref`` is still the session's P0-derived 1 mm target
    grid, and the caller still applies ``transform`` with ``-in ras -ref target_grid``, so the
    moving scan can never become the output lattice and the final resampling count stays at one.

    FSL's frames make this a pure re-estimation rather than a composition: a FLIRT matrix lives in
    the ``-in``/``-ref`` pair's scaled-mm spaces, and ``convert_xfm -inverse`` returns the matrix
    for that pair swapped -- which is exactly the pair the application needs.
    """
    kwargs = {} if policy is _RUNNER_DEFAULT else {"angular_search_deg": policy}
    result, cost = runner.coregister(reg_ref, ras, reverse_matrix, **kwargs)
    runner.invert_transform(reverse_matrix, transform)
    return result, cost


def _register_with_fallback(
    geom, ras, ref_ras, target_grid, out_pre, work_dir, session, config, runner, mod, artifacts
) -> None:
    """Register one moving scan, retrying once with an unconstrained search if QC rejects it.

    This is failure recovery, not model selection. Attempt 1 is accepted the moment it passes the
    existing QC and attempt 2 is never run; attempt 2 exists only to rescue a scan attempt 1
    already lost. Two passing candidates are therefore never compared, by NMI or otherwise, and
    no threshold is relaxed for either attempt -- both are judged by exactly the checks
    :func:`qc.validate_session_outputs` will re-apply to whichever one is selected.

    Each attempt writes its own image and transform, so a retry can never overwrite a valid
    attempt-1 result, and both attempts' provenance is kept whenever a retry happened.
    """
    # FSL matrices are only valid for the exact ``-in``/``-ref`` pair they were estimated on, so
    # that backend must fit against the final target grid; FreeSurfer LTAs carry their own
    # geometry and keep fitting against the reference's native grid, unchanged.
    reg_ref = target_grid if getattr(runner, "register_against_target_grid", False) else ref_ras
    suffix = getattr(runner, "transform_suffix", ".lta")
    stem = os.path.splitext(out_pre)[0]
    ladder = _attempt_ladder(config)
    if any(spec["type"] == INVERSE_RESCUE for spec in ladder) and not getattr(runner, "supports_inverse_direction", False):
        raise BackendError(
            f"inverse_direction_rescue was requested but backend {getattr(runner, 'name', '?')!r} does not "
            "expose an inversion contract; refusing to silently fall back to the forward-only ladder."
        )
    attempts: List[dict] = []
    selected: Optional[dict] = None

    last_backend_error: Optional[BackendError] = None
    for number, spec in enumerate(ladder, start=1):
        policy, kind = spec["search"], spec["type"]
        # Distinct paths per attempt: a retry must never write over attempt 1's candidate.
        image = f"{stem}_attempt{number}.nii.gz" if len(ladder) > 1 else out_pre
        transform = f"{stem}_attempt{number}{suffix}" if len(ladder) > 1 else stem + suffix
        reverse_matrix = f"{stem}_attempt{number}_reverse{suffix}"
        # A stand-in modality entry, so each attempt is measured exactly as a real one would be
        # without any attempt's numbers leaking into ``mod`` before one is selected.
        scratch: dict = {
            "name": mod.get("name"),
            "modality_canonical": mod.get("modality_canonical"),
            "source_path": mod.get("source_path"),
            "is_reference": False,
            "registration": {},
        }
        effective = config.angular_search_deg if policy is _RUNNER_DEFAULT else policy
        inverse = kind == INVERSE_RESCUE
        header_only = kind == HEADER_ONLY
        record = {
            "attempt": number,
            "attempt_type": kind,
            "angular_search_deg": None if header_only else (list(effective) if effective else None),
            "transform_path": transform,
            "image_path": image,
            # Which image played which role in the *estimation*. The application below is
            # identical for every attempt, so this is the only thing that varies.
            "estimation_direction": (
                "none:scanner_header" if header_only else ("reference_to_moving" if inverse else "moving_to_reference")
            ),
            "estimation_in": "" if header_only else (reg_ref if inverse else ras),
            "estimation_ref": "" if header_only else (ras if inverse else reg_ref),
            "forward_matrix_path": reverse_matrix if inverse else "",
            "apply_in": ras,
            "apply_ref": target_grid,
        }
        issued = len(runner.command_log)
        try:
            # Attempt 1 passes no override so the runner's own configuration governs it.
            if header_only:
                # No estimation: one command writes both the header transform and the image.
                runner.header_alignment(ras, target_grid, image, transform, interp=config.image_interp)
                cost = None
            elif inverse:
                _, cost = _estimate_inverse_direction(ras, reg_ref, transform, reverse_matrix, runner, policy)
            else:
                kwargs = {} if policy is _RUNNER_DEFAULT else {"angular_search_deg": policy}
                _, cost = runner.coregister(ras, reg_ref, transform, **kwargs)
            estimation = [] if header_only else [entry["argv"] for entry in runner.command_log[issued:]]
            if not header_only:
                runner.apply_transform(ras, target_grid, image, lta=transform, interp=config.image_interp)
            applied = runner.command_log[-1]["argv"] if runner.command_log else None
            attempt_artifacts = {"after": image, "before": None, "lta": transform}
            _registration_metrics(
                scratch, attempt_artifacts, ras, target_grid, image, transform, work_dir, session, config, runner
            )
            _foreground_retention(scratch, geom.path, image, runner)
            record["coreg_cost"] = cost
            record["estimate_command"] = estimation[0] if estimation else None
            # convert_xfm, on the rescue only; None elsewhere so the two paths stay distinguishable.
            record["inversion_command"] = estimation[1] if len(estimation) > 1 else None
            record["apply_command"] = applied
            # Unchanged from before the rescue existed: the last command issued at this point.
            # Kept verbatim so provenance written by earlier revisions stays comparable.
            record["command"] = runner.command_log[-1]["argv"] if runner.command_log else None
            record.update(_attempt_metrics(scratch))
            rejects = _modality_qc_rejects(scratch, image, config, runner)
            record["qc_rejects"] = rejects
            record["qc_outcome"] = "pass" if not rejects else "fail"
            record["reason"] = "" if not rejects else ",".join(rejects)
        except BackendError as exc:
            last_backend_error = exc
            record["qc_outcome"] = "fail"
            record["reason"] = f"backend: {str(exc)[:200]}"
            record["qc_rejects"] = ["backend_error"]
            attempts.append(record)
            continue

        attempts.append(record)
        if record["qc_outcome"] == "pass":
            selected = record
            mod["coreg_cost"] = cost
            mod["registration"] = scratch["registration"]
            _copy_foreground_fields(scratch, mod)
            artifacts.update(attempt_artifacts)
            break  # attempt 1 passing is final: attempt 2 is never run

    # The ladder ran. That is a different fact from whether it succeeded, and both are recorded:
    # a passthrough scan never reaches here, so this is what tells the two apart downstream.
    mod["coregistration_attempted"] = True
    mod["registration_attempts"] = attempts
    mod["selected_attempt"] = selected["attempt"] if selected else None
    mod["retried"] = len(attempts) > 1
    # A header-only fallback is NOT a registration and must never be counted as one. The scan does
    # share the session output lattice, but no transform was fitted: its anatomical alignment rests
    # on scanner-header accuracy, which the pilot300 header-prior analysis showed is not universally
    # exact. So it stays filterable, and it is deliberately not ``registered``.
    chosen = (selected or {}).get("attempt_type", FORWARD)
    mod["registration_method"] = HEADER_ONLY if chosen == HEADER_ONLY else ("optimised" if selected else "none")
    mod["transform_fitted"] = selected is not None and chosen != HEADER_ONLY
    mod["registered"] = mod["transform_fitted"]

    if selected is None:
        _persist_attempt_transforms(attempts, work_dir, session, mod, runner)
        if last_backend_error is not None and not any(a.get("qc_rejects") != ["backend_error"] for a in attempts):
            # Every attempt died in the toolbox, so no image was ever produced. Raise, exactly as
            # before the ladder existed, so the caller records a structured backend failure.
            raise last_backend_error
        # Nothing passed QC. Keep the last attempt's evidence so the failure can be reviewed, and
        # let the normal QC path fail the modality; no image is published either way.
        last = attempts[-1] if attempts else {}
        fallback_image = last.get("image_path")
        if fallback_image and os.path.exists(fallback_image) and fallback_image != out_pre:
            _copy_if_exists(fallback_image, out_pre)
        artifacts["after"] = out_pre if os.path.exists(out_pre) else fallback_image
        artifacts["lta"] = last.get("transform_path")
        mod["registration"] = last.get("metrics_registration", {}) or mod.get("registration", {})
        mod["transform_path"] = _persist_transform(last.get("transform_path", ""), work_dir, session, mod, runner)
        return

    if selected["image_path"] != out_pre:
        os.replace(selected["image_path"], out_pre)
        artifacts["after"] = out_pre
    mod["transform_path"] = _persist_transform(selected["transform_path"], work_dir, session, mod, runner)
    _persist_attempt_transforms(attempts, work_dir, session, mod, runner)


def _persist_attempt_transforms(attempts: List[dict], work_dir: str, session, mod: dict, runner) -> None:
    """Copy every attempt's transform out of the work dir, which is deleted on success.

    Only done when a retry actually happened: with a single attempt the transform is already
    persisted under its modality name. Provenance that points into a deleted directory is not
    provenance, so each record is rewritten to where its transform now lives.
    """
    if len(attempts) < 2 or runner.dry_run:
        return
    label = mod.get("modality_canonical") or mod.get("name") or "moving"
    destination_dir = os.path.join(os.path.dirname(work_dir), "transforms")
    for record in attempts:
        outcome = record.get("qc_outcome", "unknown")
        # The rescue's reverse-direction fit is kept alongside the inverted matrix it produced:
        # without it the inversion cannot be re-derived or checked after the fact.
        for key, tag in (("transform_path", ""), ("forward_matrix_path", "_reverse")):
            source = record.get(key) or ""
            if not source or not os.path.exists(source):
                continue
            suffix = os.path.splitext(source)[1] or ".mat"
            destination = os.path.join(destination_dir, f"{label}_attempt{record['attempt']}_{outcome}{tag}{suffix}")
            try:
                os.makedirs(destination_dir, exist_ok=True)
                shutil.copyfile(source, destination)
                record[key] = destination
            except OSError as exc:  # noqa: BLE001 - keep the work-dir path rather than losing the record
                logger.warning("Could not persist attempt transform for %s: %s", session.key, exc)


def _attempt_metrics(scratch: dict) -> dict:
    """The measured quantities an attempt must record, whether or not it is selected."""
    reg = scratch.get("registration") or {}
    return {
        "translation_norm_mm": reg.get("translation_norm_mm"),
        "rotation_deg": reg.get("rotation_deg"),
        "determinant": reg.get("determinant"),
        "orthogonality_error": reg.get("orthogonality_error"),
        "nmi_before": reg.get("nmi_before"),
        "nmi_after": reg.get("nmi_after"),
        "nmi_improvement": reg.get("nmi_improvement"),
        "foreground_retained_frac": scratch.get("foreground_retained_frac"),
        "metrics_registration": reg,
    }


def _modality_qc_rejects(scratch: dict, image: str, config, runner) -> List[str]:
    """Rejection-level QC for one attempt, using the session validator's own checks.

    Deliberately the same two functions :func:`qc.validate_session_outputs` applies per modality,
    so an accepted attempt cannot later be rejected by a stricter standard, and a retry cannot be
    triggered by a laxer one. The session-level shared-grid check is not included: it is a
    property of the session, identical for both attempts by construction (both resample onto the
    same target grid), and not attributable to a single scan.
    """
    if runner.dry_run:
        return []
    _, rejects, _ = qc_mod.validate_output_image(image, config)
    return list(rejects) + qc_mod.rigid_violations(scratch.get("registration") or {}, config)


def _persist_transform(transform: str, work_dir: str, session: Session, mod: dict, runner) -> str:
    """Copy a transform out of the work dir into ``<session>/transforms/`` and return its path.

    Named after the modality rather than the temporary stem, so the artifact is readable and
    stable across reruns.
    """
    suffix = os.path.splitext(transform)[1] or getattr(runner, "transform_suffix", ".lta")
    label = mod.get("modality_canonical") or mod.get("name") or "moving"
    destination = os.path.join(os.path.dirname(work_dir), "transforms", f"{label}_to_reference{suffix}")
    if runner.dry_run:
        return destination
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copyfile(transform, destination)
        return destination
    except OSError as exc:  # noqa: BLE001 - report, but keep the in-work path usable
        logger.warning("Could not persist transform for %s: %s", session.key, exc)
        return transform


def _parse_xsep(value) -> Optional[tuple]:
    """Parse the manifest's ``AxBxC`` geometry encoding; ``None`` when absent/unparseable."""
    text = str(value or "").strip().replace("X", "x")
    if not text:
        return None
    try:
        return tuple(float(part) for part in text.split("x") if part)
    except ValueError:
        return None


#: Spacing tolerance for the manifest cross-check. The manifest stores geometry with limited
#: precision (``'1x1x1.2'``), so this is deliberately loose: it must catch a real disagreement
#: (1 mm vs 3 mm, a transposed shape) without firing on a rounded decimal.
MANIFEST_SPACING_TOL_MM = 0.01


def _manifest_geometry_divergence(geom: ScanGeometry, sample: dict) -> str:
    """Compare probed header geometry with the canonical manifest row. ``""`` when consistent."""
    if not sample:
        return ""  # filesystem-driven run: there is no manifest claim to check against
    shape = _parse_xsep(sample.get("shape"))
    if shape and tuple(int(s) for s in shape[:3]) != tuple(int(s) for s in geom.shape[:3]):
        return f"shape header={list(geom.shape[:3])} manifest={[int(s) for s in shape[:3]]}"
    pixdim = _parse_xsep(sample.get("pixdim"))
    if pixdim and len(pixdim) >= 3:
        deltas = [abs(a - b) for a, b in zip(geom.voxel_sizes, pixdim[:3])]
        if max(deltas) > MANIFEST_SPACING_TOL_MM:
            return f"spacing header={[round(v, 4) for v in geom.voxel_sizes]} manifest={list(pixdim[:3])}"
    return ""


#: Everything :func:`_foreground_retention` records. Kept as one list because the counts are what
#: the zero-overlap invariant is decided on: carrying the ratio forward without them would leave
#: the invariant unable to fire on the very path that produced the value.
FOREGROUND_FIELDS = (
    "foreground_retained_frac",
    "foreground_retained_frac_raw",
    "source_foreground_voxel_count",
    "output_foreground_voxel_count",
    "source_foreground_volume_mm3",
    "output_foreground_volume_mm3",
)


def _copy_foreground_fields(scratch: dict, mod: dict) -> None:
    """Carry the retention evidence from an attempt's scratch entry onto the modality."""
    for field in FOREGROUND_FIELDS:
        if field in scratch:
            mod[field] = scratch[field]


def _foreground_retention(mod: dict, source_path: str, resampled_path: str, runner) -> None:
    """Fraction of the moving scan's foreground volume still present after resampling.

    The session lattice is the reference scan's field of view, so a moving scan that extends
    beyond it (a neck-covering FLAIR against a brain-only T1w, say) loses that region. This
    measures the loss instead of assuming it is negligible.

    Both sides are thresholded at the *same absolute* intensity, derived from the source, which
    is meaningful because interpolation preserves intensity scale. Edge blurring can push the
    ratio marginally above 1, so only the low side is acted on.

    The voxel counts either side of that threshold are recorded alongside the ratio. They are the
    structural evidence: ``foreground_retained_frac`` is rounded for reporting, so it cannot be
    used to decide whether *any* foreground survived, and a spline resample of a non-overlapping
    volume leaves interpolation dust that is nonzero but carries no anatomy. The counts say
    plainly whether the source-defined foreground exists on the target lattice at all.
    """
    if runner.dry_run:
        return
    try:
        import nibabel as nib

        src_img = nib.load(source_path)
        src = np.asanyarray(src_img.dataobj, dtype=np.float32)
        positive = src[np.isfinite(src) & (src > 0)]
        if positive.size < 100:
            return
        threshold = float(np.percentile(positive, 40)) * 0.5
        src_voxel_mm3 = float(np.prod(nib.affines.voxel_sizes(src_img.affine)[:3]))
        src_count = int((src > threshold).sum())
        src_volume = float(src_count) * src_voxel_mm3
        if src_volume <= 0:
            return
        out_img = nib.load(resampled_path)
        out = np.asanyarray(out_img.dataobj, dtype=np.float32)
        out_voxel_mm3 = float(np.prod(nib.affines.voxel_sizes(out_img.affine)[:3]))
        out_count = int((out > threshold).sum())
        out_volume = float(out_count) * out_voxel_mm3
        mod["source_foreground_voxel_count"] = src_count
        mod["output_foreground_voxel_count"] = out_count
        mod["source_foreground_volume_mm3"] = src_volume
        mod["output_foreground_volume_mm3"] = out_volume
        mod["foreground_retained_frac_raw"] = out_volume / src_volume
        mod["foreground_retained_frac"] = round(out_volume / src_volume, 4)
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never fail a session
        mod["foreground_retention_error"] = str(exc)[:200]


def _registration_metrics(mod, artifacts, ras, target_grid, after, lta, work_dir, session, config, runner) -> None:
    """Transform decomposition + NMI (and optional edge overlap) before/after registration."""
    reg = mod["registration"]
    reg.update(runner.transform_metrics(lta, moving=mod["source_path"], reference=target_grid))
    if not config.compute_registration_metrics or runner.dry_run:
        return
    before = os.path.join(work_dir, _safe_stem(session, mod["source_path"]) + "_before.nii.gz")
    try:
        runner.apply_transform(ras, target_grid, before, lta=None, interp=config.image_interp)
        artifacts["before"] = before
        ref_on_grid = target_grid  # reference already lives on the target grid
        a = qc_mod._load(ref_on_grid)
        b_before = qc_mod._load(before)
        b_after = qc_mod._load(after)
        reg["nmi_before"] = metrics_mod.normalized_mutual_information(a, b_before)
        reg["nmi_after"] = metrics_mod.normalized_mutual_information(a, b_after)
        if reg.get("nmi_before") is not None and reg.get("nmi_after") is not None:
            reg["nmi_improvement"] = round(reg["nmi_after"] - reg["nmi_before"], 5)
        if config.compute_edge_overlap:
            reg["edge_overlap_before"] = metrics_mod.edge_overlap(a, b_before)
            reg["edge_overlap_after"] = metrics_mod.edge_overlap(a, b_after)
    except Exception as exc:  # noqa: BLE001 - metrics are best-effort, never fatal
        reg["metrics_error"] = str(exc)[:200]


def _skull_strip_and_write(
    session, config, runner, work_dir, target_grid, reference, pre_mask, qc, do_coreg, brain_mask_out
) -> None:
    """Produce each modality's *candidate* image inside the work dir.

    Nothing here writes to a canonical derivative path. Candidates are validated first and only
    then published by :func:`_publish_outputs`, so an image that fails QC never becomes part of
    the derivative. See that function for the invariant.
    """
    mods_by_path = {m["source_path"]: m for m in qc["modalities"]}
    candidate_dir = os.path.join(work_dir, "candidates")
    os.makedirs(candidate_dir, exist_ok=True)

    def candidate_for(path: str) -> str:
        return os.path.join(candidate_dir, os.path.basename(mods_by_path[path]["output_path"]))

    if not config.do_skull_strip:
        for path, pre in pre_mask.items():
            # The pre-mask image already *is* the finished candidate for this backend.
            mods_by_path[path]["candidate_path"] = pre
        return

    if do_coreg:
        ref_pre = pre_mask.get(reference.path)
        if ref_pre is None:
            raise FreeSurferError("reference scan has no pre-mask output; cannot build shared brain mask")
        mask = _make_brain_mask(runner, config, work_dir, ref_pre, target_grid, stem="ref")
        for path, pre in pre_mask.items():
            cand = candidate_for(path)
            try:
                runner.apply_mask(pre, mask, cand) if not runner.dry_run else _finalize(pre, cand)
                mods_by_path[path]["candidate_path"] = cand
            except BackendError as exc:
                mods_by_path[path]["status"] = "failed"
                mods_by_path[path]["reason"] = str(exc)[:500]
        _record_brain_mask(qc, mask, brain_mask_out, runner)
    else:
        first_mask = None
        for path, pre in pre_mask.items():
            cand = candidate_for(path)
            try:
                mask = _make_brain_mask(runner, config, work_dir, pre, pre, stem=_safe_stem(session, path))
                runner.apply_mask(pre, mask, cand) if not runner.dry_run else _finalize(pre, cand)
                mods_by_path[path]["candidate_path"] = cand
                first_mask = first_mask or mask
            except BackendError as exc:
                mods_by_path[path]["status"] = "failed"
                mods_by_path[path]["reason"] = str(exc)[:500]
        _record_brain_mask(qc, first_mask, brain_mask_out, runner)


def _publish_outputs(qc: dict, config, runner, state: dict) -> None:
    """Publish QC-passing candidates to their canonical paths, and only those.

    The invariant this establishes is::

        a canonical output exists  =>  that sample passed output QC

    A modality that failed keeps its transform, metrics, provenance and QC reason, but loses its
    ``output_path``: the derivative manifest must never carry a path to an image that did not
    pass. Any canonical output left by an earlier attempt is removed (or quarantined) here too,
    so a re-run cannot inherit a stale invalid image.

    Publication is a same-filesystem :func:`os.replace`, which is atomic: a reader either sees
    the previous file or the new one, never a partially written image.
    """
    if runner.dry_run:
        return
    session_dir = qc.get("session_output_dir") or ""
    quarantine_dir = os.path.join(session_dir, "_quarantine") if session_dir else ""
    published, quarantined, stale_removed = 0, 0, 0

    for mod in qc.get("modalities", []):
        out = mod.get("output_path") or ""
        cand = mod.get("candidate_path") or ""
        passed = mod.get("status") == "ok"

        if passed and cand and os.path.exists(cand):
            os.makedirs(os.path.dirname(out), exist_ok=True)
            try:
                os.replace(cand, out)
            except OSError:  # different filesystem: fall back to a copy + unlink
                shutil.copyfile(cand, out)
                _unlink_quietly(cand)
            mod["published"] = True
            published += 1
            # The rendered montage reads the post-registration image; repoint it at the
            # published file, since the candidate no longer exists under its old name.
            artifacts = (state.get("reg_artifacts") or {}).get(mod.get("source_path"))
            if isinstance(artifacts, dict) and artifacts.get("after") == cand:
                artifacts["after"] = out
            continue

        mod["published"] = False
        if passed:
            # Marked ok but nothing to publish: that is a failure, not a silent empty sample.
            mod["status"] = "failed"
            mod["reason"] = (mod.get("reason", "") + "; publication:missing_candidate").strip("; ")
        if out and os.path.exists(out):
            stale_removed += 1
            _quarantine_or_remove(out, quarantine_dir, config, suffix=".stale")
        if cand and os.path.exists(cand):
            quarantined += 1
            moved_to = _quarantine_or_remove(cand, quarantine_dir, config, suffix=".rejected")
            mod["quarantined_path"] = moved_to
            # The rejected image is the evidence a human needs in order to see *how* it failed,
            # so keep the before/after montage pointing at it rather than at a path we just
            # emptied. Losing the render would hide exactly the cases worth reviewing.
            artifacts = (state.get("reg_artifacts") or {}).get(mod.get("source_path"))
            if isinstance(artifacts, dict) and artifacts.get("after") == cand and moved_to:
                artifacts["after"] = moved_to
        # Keep the intended location for debugging, but never as a usable output path.
        mod["unpublished_output_path"] = out
        mod["output_path"] = ""

    qc["publication"] = {
        "published": published,
        "rejected_candidates": quarantined,
        "stale_outputs_cleared": stale_removed,
        "quarantine_dir": quarantine_dir if config.quarantine_failed_outputs else "",
    }


def _quarantine_or_remove(path: str, quarantine_dir: str, config, suffix: str) -> str:
    """Move an invalid image out of the derivative; return where it went, or "" if deleted."""
    if config.quarantine_failed_outputs and quarantine_dir:
        os.makedirs(quarantine_dir, exist_ok=True)
        dest = os.path.join(quarantine_dir, _mark_basename(os.path.basename(path), suffix))
        try:
            os.replace(path, dest)
            return dest
        except OSError:
            pass
    _unlink_quietly(path)
    return ""


def _mark_basename(basename: str, marker: str) -> str:
    """Insert ``marker`` before the image extension, keeping the file loadable.

    ``foo.nii.gz`` + ``.rejected`` -> ``foo.rejected.nii.gz``. Appending after the extension
    instead would make nibabel refuse the file ("cannot work out file type"), and a quarantined
    image that cannot be opened is useless as evidence -- it is exactly the image a reviewer
    needs to look at.
    """
    for ext in (".nii.gz", ".nii", ".mgz"):
        if basename.lower().endswith(ext):
            return basename[: -len(ext)] + marker + ext
    return basename + marker


def _unlink_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _make_brain_mask(runner, config, work_dir, source, target_grid, stem: str) -> str:
    seg = os.path.join(work_dir, f"{stem}_seg.nii.gz")
    seg_on_grid = os.path.join(work_dir, f"{stem}_seg_grid.nii.gz")
    mask = os.path.join(work_dir, f"{stem}_brainmask.nii.gz")
    runner.synthseg(source, seg, robust=config.synthseg_robust)
    runner.apply_transform(seg, target_grid, seg_on_grid, lta=None, interp=config.mask_interp)
    runner.binarize(seg_on_grid, mask, min_value=1.0)
    return mask


def _record_brain_mask(qc, mask, brain_mask_out, runner) -> None:
    if mask is None:
        return
    if brain_mask_out and not runner.dry_run:
        os.makedirs(os.path.dirname(brain_mask_out), exist_ok=True)
        shutil.copyfile(mask, brain_mask_out)
    entry = {"path": brain_mask_out}
    if not runner.dry_run:
        entry["brain_volume_cm3"] = qc_mod.brain_volume_cm3(brain_mask_out or mask)
    qc["brain_mask"] = entry


def _finalize_status(qc: dict, config, runner) -> None:
    if runner.dry_run:
        qc["status"] = "dry_run"
        return
    mods = qc["modalities"]
    ok = [m for m in mods if m["status"] == "ok"]
    failed = [m for m in mods if m["status"] == "failed"]
    session_rejects = qc.get("output_qc", {}).get("session_rejects", [])
    warns_block = config.qc.escalate_warnings and qc.get("output_qc", {}).get("n_warn", 0) > 0

    if not ok:
        qc["status"] = "failed"
        qc["reason"] = "no modality produced a valid output"
    elif failed or session_rejects or warns_block:
        qc["status"] = "partial"
        reasons = []
        if failed:
            reasons.append(f"{len(failed)} modality/ies failed")
        if session_rejects:
            reasons.append("output_qc:" + ",".join(session_rejects))
        if warns_block:
            reasons.append("escalated_warnings")
        qc["reason"] = "; ".join(reasons)
    else:
        qc["status"] = "success"
        qc["reason"] = ""


def _maybe_render_qc(qc, config, state, qc_image_dir, forced: bool) -> None:
    """Render visual QC for sampled or suspicious sessions."""
    flags = qc.get("output_qc", {}).get("flags", {})
    suspicious = (
        qc.get("status") in {"partial", "failed"} or bool(flags) or qc.get("reference", {}).get("anisotropic_reference", False)
    )
    if not (forced or (config.qc_render_suspicious and suspicious)):
        return
    base = os.path.join(qc_image_dir, qc["dataset"], f"{qc['subject']}_{qc['session']}")

    # Multi-modality montage of the final outputs.
    ok_mods = sorted(
        (m for m in qc["modalities"] if m["status"] == "ok" and os.path.exists(m.get("output_path", ""))),
        key=lambda m: (not m["is_reference"], m["name"]),
    )
    if ok_mods:
        qc_mod.render_qc_montage(
            [m["output_path"] for m in ok_mods],
            [m["name"] + (" (ref)" if m["is_reference"] else "") for m in ok_mods],
            qc.get("brain_mask", {}).get("path"),
            base + ".png",
            title=qc["session_key"],
        )
    # Before/after registration montages. Normally only the most-suspicious moving modality is
    # rendered; under ``qc_image_render_all`` (smoke/pilot) every one is, because the case worth
    # looking at — the lowest foreground_retained_frac — is not knowable until after processing
    # and is not necessarily the one with the largest motion.
    reg_artifacts = state.get("reg_artifacts", {})
    if config.qc_image_render_all:
        targets = [m for m in qc["modalities"] if not m["is_reference"] and m.get("registration")]
    else:
        worst = _worst_moving(qc)
        targets = [worst] if worst is not None else []

    for mod in targets:
        art = reg_artifacts.get(mod["source_path"], {})
        if not art.get("after"):
            continue
        suffix = f"_reg_{mod.get('modality_canonical') or mod['name']}" if config.qc_image_render_all else "_reg"
        retained = mod.get("foreground_retained_frac")
        label = mod["name"] if retained is None else f"{mod['name']}  retained={retained:.3f}"
        qc_mod.render_registration_qc(
            state.get("target_grid"),
            art.get("before"),
            art["after"],
            qc.get("brain_mask", {}).get("path"),
            base + suffix + ".png",
            label=label,
            title=qc["session_key"],
        )


def _worst_moving(qc: dict):
    """The moving modality most worth eyeballing: worst motion, similarity, or FOV retention."""
    moving = [m for m in qc["modalities"] if not m["is_reference"] and m.get("registration")]
    if not moving:
        return None

    def severity(m):
        reg = m["registration"]
        imp = reg.get("nmi_improvement")
        retained = m.get("foreground_retained_frac")
        return (
            reg.get("translation_norm_mm", 0.0),
            reg.get("rotation_deg", 0.0),
            -(imp if imp is not None else 0.0),
            -(retained if retained is not None else 1.0),  # lower retention == more suspicious
        )

    return max(moving, key=severity)


def _write_commands(runner, out_dir: str) -> None:
    if not runner.command_log:
        return
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "commands.json"), "w") as handle:
            import json

            json.dump(runner.command_log, handle, indent=2)
    except OSError as exc:
        logger.warning("Could not write command log to %s: %s", out_dir, exc)


def _cleanup_work(config, runner, qc, work_dir: str) -> None:
    if not os.path.isdir(work_dir):
        return
    if runner.dry_run:
        shutil.rmtree(work_dir, ignore_errors=True)
        return
    success = qc.get("status") == "success"
    if success:
        if config.clean_work_on_success:
            shutil.rmtree(work_dir, ignore_errors=True)
    else:
        if not config.preserve_work_on_failure:
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            logger.info("Preserving work dir for %s (status=%s): %s", qc["session_key"], qc["status"], work_dir)


def _finalize(src: str, dst: str) -> None:
    _copy_if_exists(src, dst)


def _copy_if_exists(src: str, dst: str) -> None:
    if not os.path.exists(src):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
