"""CLI for intra-session multimodal co-registration (FOMO26).

Subcommands
-----------
    manifest    Build the deterministic session manifest + 4D inventory.
    inventory   Print the 3D/4D inventory summary (alias of manifest report).
    run         Process sessions (optionally one array shard); resumable.
    aggregate   Collect per-session QC into global TSV/JSON + suspicious list.
    resubmit    List chunk ids still needing work (partial/failed/empty/missing).
    check-tools Report FreeSurfer tool availability and exit.

A SLURM array calls ``run --manifest ... --chunk-size N --chunk-id $TASK`` so
shards never write shared files; ``aggregate`` runs once afterwards.
"""

import argparse
import json
import logging
import os
import sys
from asparagus_preprocessing.session_coreg import (
    canonical as canonical_mod,
    derivative_manifest as derivative_mod,
    fsl as fsl_mod,
    manifest as manifest_mod,
    pipeline as pipeline_mod,
    provenance as provenance_mod,
    qc as qc_mod,
    session_audit as audit_mod,
)
from asparagus_preprocessing.session_coreg.config import (
    BACKENDS,
    FSL_INTERP_CHOICES,
    INTERP_CHOICES,
    CoregConfig,
    QCThresholds,
)
from asparagus_preprocessing.session_coreg.discovery import Session, find_sessions
from asparagus_preprocessing.session_coreg.freesurfer import FreeSurferRunner, check_tools
from functools import partial
from multiprocessing import Pool
from typing import List, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Argument wiring
# --------------------------------------------------------------------------- #
def add_io_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--input-root",
        default=os.environ.get("ASPARAGUS_COREG_SOURCE"),
        help="Root of the cleaned BIDS tree (never modified).",
    )
    p.add_argument(
        "--output-root",
        default=os.environ.get("ASPARAGUS_COREG_OUTPUT"),
        help="Root of the co-registered output tree (must differ from --input-root).",
    )
    p.add_argument("--freesurfer-home", default=os.environ.get("FREESURFER_HOME"))
    p.add_argument("--log-level", default="INFO")


def add_backend_args(p: argparse.ArgumentParser) -> None:
    """Backend + corpus-definition flags, shared by every subcommand that touches data."""
    p.add_argument(
        "--backend",
        choices=list(BACKENDS),
        default=os.environ.get("ASPARAGUS_COREG_BACKEND", "freesurfer"),
        help="Registration toolbox. Use 'fsl' on Jean Zay (FreeSurfer is not installed there).",
    )
    p.add_argument("--dof", type=int, default=6, help="Degrees of freedom; the production contract is rigid 6.")
    p.add_argument("--cost", default="normmi", help="Registration cost function (default normmi, multimodal).")
    p.add_argument(
        "--allow-affine-ablation",
        action="store_true",
        help="Permit --dof != 6. Named ablation only; never the production derivative.",
    )
    p.add_argument("--no-header-init", action="store_true", help="FSL: do not pass -usesqform.")
    p.add_argument(
        "--fallback-unconstrained-retry",
        action="store_true",
        help=(
            "Retry a moving scan once with FLIRT's unconstrained search when the constrained "
            "attempt fails output QC. Failure recovery only: a passing first attempt is final."
        ),
    )
    p.add_argument(
        "--inverse-direction-rescue",
        action="store_true",
        help=(
            "FSL: after every forward attempt has failed output QC, estimate the same rigid fit "
            "once more with the images swapped (target grid as -in, moving scan as -ref), invert "
            "it with convert_xfm -inverse, and apply the inverse to the original moving scan onto "
            "the same target grid. Failure recovery only; the output lattice never changes."
        ),
    )
    p.add_argument(
        "--header-only-fallback",
        action="store_true",
        help=(
            "FSL: after every optimised attempt has failed output QC, publish the scan where its "
            "scanner header places it (-applyxfm -usesqform) on the same target grid. Recorded as "
            "registration_method=header_only and status=header_only_aligned; never as a registration."
        ),
    )
    p.add_argument(
        "--angular-search-deg",
        nargs=2,
        type=int,
        metavar=("LO", "HI"),
        default=None,
        help=(
            "FSL: bound the optimiser's rotation search on x/y/z (-searchrx/-searchry/-searchrz). "
            "Omit for FLIRT's own default. A search constraint, not a QC threshold."
        ),
    )
    p.add_argument(
        "--canonical-manifest",
        default=os.environ.get("ASPARAGUS_COREG_CANONICAL_MANIFEST"),
        help="canonical_manifest_v2.tsv. When set it DEFINES the corpus; no filesystem walk is used.",
    )
    p.add_argument("--expect-manifest-sha", default=None, help="Fail closed unless the manifest hashes to this.")
    p.add_argument(
        "--cleaned-metadata",
        default=os.environ.get("ASPARAGUS_COREG_CLEANED_METADATA"),
        help="FOMO300K_cleaned/manifest.tsv, source of the scanner/demographic identity evidence.",
    )
    p.add_argument(
        "--require-explicit-session",
        action="store_true",
        help="Treat ses-01-style fallback ids as unproven identity (blocks registration, keeps passthrough).",
    )


def add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--reference-policy", choices=["anisotropy_aware", "fomo50k_legacy"], default="anisotropy_aware")
    p.add_argument("--reference-string", default=None)
    p.add_argument("--max-reference-spacing", type=float, default=3.0)
    p.add_argument("--no-coreg", action="store_true")
    p.add_argument("--no-skull-strip", action="store_true")
    p.add_argument("--no-synthseg-robust", action="store_true")
    p.add_argument("--iso-spacing", nargs=3, type=float, default=[1.0, 1.0, 1.0], metavar=("X", "Y", "Z"))
    p.add_argument("--keep-native-spacing", action="store_true")
    p.add_argument(
        "--image-interp",
        choices=sorted(set(INTERP_CHOICES) | set(FSL_INTERP_CHOICES)),
        default=None,
        help="Default depends on the backend: 'trilin' (FreeSurfer), 'spline' (FSL, matching P0's cubic kernel).",
    )
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--allow-4d", action="store_true", help="Drop 4D scans with a record instead of failing the session.")
    p.add_argument("--no-registration-metrics", action="store_true", help="Skip NMI before/after (faster).")
    p.add_argument("--edge-overlap", action="store_true", help="Also compute edge-overlap metrics (slower).")
    p.add_argument("--escalate-warnings", action="store_true", help="Treat QC warnings as blocking success.")
    p.add_argument("--keep-work-on-success", action="store_true", help="Do not delete work dir on success.")
    p.add_argument("--clean-work-on-failure", action="store_true", help="Delete work dir even on failure.")
    p.add_argument("--qc-images-per-dataset", type=int, default=25)
    p.add_argument("--qc-stride", type=int, default=20)
    p.add_argument(
        "--qc-images-all",
        action="store_true",
        help="Render tri-planar QC for EVERY processed session (use for the smoke and the pilot).",
    )


def config_from_args(args) -> CoregConfig:
    return CoregConfig(
        backend=getattr(args, "backend", "freesurfer"),
        dof=getattr(args, "dof", 6),
        cost=getattr(args, "cost", "normmi"),
        allow_affine_ablation=getattr(args, "allow_affine_ablation", False),
        init_from_header=not getattr(args, "no_header_init", False),
        angular_search_deg=(tuple(a) if (a := getattr(args, "angular_search_deg", None)) else None),
        fallback_unconstrained_retry=getattr(args, "fallback_unconstrained_retry", False),
        inverse_direction_rescue=getattr(args, "inverse_direction_rescue", False),
        header_only_fallback=getattr(args, "header_only_fallback", False),
        canonical_manifest=getattr(args, "canonical_manifest", None),
        expected_manifest_sha256=getattr(args, "expect_manifest_sha", None),
        cleaned_metadata=getattr(args, "cleaned_metadata", None),
        require_explicit_session=getattr(args, "require_explicit_session", False),
        reference_policy=args.reference_policy,
        reference_string=args.reference_string,
        max_reference_spacing=args.max_reference_spacing,
        do_coregister=not args.no_coreg,
        do_skull_strip=not args.no_skull_strip,
        synthseg_robust=not args.no_synthseg_robust,
        target_iso_spacing=None if args.keep_native_spacing else tuple(args.iso_spacing),
        image_interp=args.image_interp,
        threads=args.threads,
        fail_on_4d=not args.allow_4d,
        compute_registration_metrics=not args.no_registration_metrics,
        compute_edge_overlap=args.edge_overlap,
        qc=QCThresholds(escalate_warnings=args.escalate_warnings),
        clean_work_on_success=not args.keep_work_on_success,
        preserve_work_on_failure=not args.clean_work_on_failure,
        qc_image_max_per_dataset=args.qc_images_per_dataset,
        qc_image_stride=args.qc_stride,
        qc_image_render_all=getattr(args, "qc_images_all", False),
        overwrite=getattr(args, "overwrite", False),
        freesurfer_home=args.freesurfer_home,
    )


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


def _require_roots(args) -> tuple:
    if not args.input_root or not args.output_root:
        raise SystemExit("--input-root and --output-root are required (or set ASPARAGUS_COREG_SOURCE/OUTPUT).")
    return os.path.abspath(args.input_root), os.path.abspath(args.output_root)


def _manifest_path(output_root: str) -> str:
    return os.path.join(output_root, "manifest.json")


# --------------------------------------------------------------------------- #
# Resume: strict "done" validation
# --------------------------------------------------------------------------- #
#: Config fields that define *what the derivative is*. If any of them differs from the snapshot a
#: session was produced under, that session's outputs are not the ones the current run is asking
#: for, and reusing them would silently mix two policies inside one derivative.
RESUME_INVALIDATING_FIELDS = (
    "backend",
    "dof",
    "cost",
    "init_from_header",
    "image_interp",
    "angular_search_deg",
    "fallback_unconstrained_retry",
    "inverse_direction_rescue",
    "header_only_fallback",
    "target_iso_spacing",
    "reference_policy",
    "do_coregister",
    "do_skull_strip",
)


def _config_differs(stored: dict, config: CoregConfig) -> Optional[str]:
    """Name the first registration-defining field whose stored value differs, else None.

    A field *missing* from the snapshot counts as a difference, not as compatibility. Older
    revisions wrote fewer fields, so an absent ``angular_search_deg`` means the session was
    produced before the bounded search existed -- which is exactly the policy difference this
    guard is for. Treating absence as "no objection" let a real production root keep 32 sessions
    registered with no angular search and no fallback retry.
    """
    wanted = pipeline_mod._config_snapshot(config)
    for field in RESUME_INVALIDATING_FIELDS:
        if field not in stored:
            return f"{field}: absent from the stored snapshot (produced before this field existed)"
        # Snapshots round-trip through JSON, so tuples come back as lists; compare by value.
        if _as_value(stored.get(field)) != _as_value(wanted.get(field)):
            return f"{field}: stored={stored.get(field)!r} requested={wanted.get(field)!r}"
    return None


def _as_value(v):
    return tuple(v) if isinstance(v, (list, tuple)) else v


def is_session_done(output_root: str, session_key: str, config: CoregConfig) -> bool:
    """A session counts as done only if it is a validated success produced under this config."""
    import nibabel as nib

    out_dir = os.path.join(output_root, session_key)
    qc = qc_mod.read_session_qc(os.path.join(out_dir, "qc.json"))
    if not qc or qc.get("status") != "success":
        return False
    changed = _config_differs(qc.get("config") or {}, config)
    if changed is not None:
        logger.info("Session %s was produced under a different policy (%s); reprocessing.", session_key, changed)
        return False
    mods = qc.get("modalities", [])
    if not mods or any(m.get("status") != "ok" for m in mods):
        return False
    shapes = []
    for mod in mods:
        path = mod.get("output_path", "")
        if not path or not os.path.exists(path):
            return False
        try:
            img = nib.load(path)
            if len(img.shape) != 3:
                return False
            shapes.append(tuple(int(s) for s in img.shape))
        except Exception:  # noqa: BLE001
            return False
    # Only a *registered* session is contracted to share one lattice. A single-scan or
    # audit-blocked session is an intentional passthrough whose scans keep their own P0 grids, so
    # requiring one shape here would mark it permanently unfinished and reprocess it on every
    # resume. This mirrors the same exemption in the output-QC shared-grid check.
    if qc.get("coregistration_applied", True) and len(set(shapes)) > 1:
        return False
    if config.do_skull_strip and config.save_brain_mask:
        bm = qc.get("brain_mask", {}).get("path")
        if not bm or not os.path.exists(bm):
            return False
    return True


# --------------------------------------------------------------------------- #
# Sessions to process
# --------------------------------------------------------------------------- #
def load_canonical(config: CoregConfig) -> List[dict]:
    """Read the canonical manifest declared by the config, verifying its SHA when contracted."""
    return canonical_mod.read_canonical_manifest(config.canonical_manifest, config.expected_manifest_sha256)


def audit_from_config(config: CoregConfig, rows: List[dict]):
    """Run the session-identity audit that gates multi-scan registration."""
    return audit_mod.audit_sessions(
        rows,
        cleaned_metadata_path=config.cleaned_metadata,
        require_explicit_session=config.require_explicit_session,
    )


def _canonical_sessions(config: CoregConfig) -> List[Session]:
    """Sessions defined by the canonical manifest, with ambiguous ones marked non-eligible.

    An ambiguous session is *not* removed: it still yields a Session whose scans are processed
    as passthrough, so the derivative keeps the canonical sample set. What the audit removes is
    permission to co-register, never the data.
    """
    rows = load_canonical(config)
    verdicts = audit_from_config(config, rows)
    eligible = audit_mod.eligible_keys(verdicts)
    reasons = {v.session_key: v.reason for v in verdicts}
    sessions = canonical_mod.sessions_from_canonical(rows)
    blocked = 0
    for session in sessions:
        if session.key not in eligible and len(session.image_paths) > 1:
            session.coreg_allowed = False
            session.coreg_block_reason = reasons.get(session.key, "ambiguous_session")
            blocked += 1
    logger.info(
        "Canonical corpus: %d samples, %d sessions, %d eligible for registration, %d multi-scan blocked by the audit.",
        len(rows),
        len(sessions),
        len(eligible),
        blocked,
    )
    return sessions


def canonical_smoke_keys(
    sessions: List[Session], seed: int = 0, per_category: int = 3, max_sessions: Optional[int] = 40
) -> List[str]:
    """Deterministic stratified smoke set over canonical sessions.

    Strata are the scenarios a rigid intra-session pipeline can actually fail on: single-scan
    passthrough, the common structural pairs, diffusion (rigid does not fix EPI distortion),
    anisotropic geometry, rich multi-modal sessions, and per-dataset coverage. Selection is
    seeded and order-independent so the same smoke set is reproducible from the manifest alone.

    Scenario coverage is filled first and dataset breadth second, so truncating to
    ``max_sessions`` can only cost breadth — never a scenario. This matters because the corpus
    has ~170 distinct ``dataset`` values (each OpenNeuro accession counts separately), and
    covering them all would produce a "smoke" set far larger than one meant to be eyeballed.
    """
    import random

    def mods(session: Session) -> set:
        return {str(s.get("modality_canonical", "")) for s in session.samples}

    def anisotropic(session: Session) -> bool:
        for sample in session.samples:
            pix = pipeline_mod._parse_xsep(sample.get("pixdim"))
            if pix and len(pix) >= 3 and max(pix[:3]) / max(min(pix[:3]), 1e-6) >= 3.0:
                return True
        return False

    dwi = set(("ADC", "DWI_B0", "DWI_B1000", "DWI_TRACE", "DWI"))
    categories = {
        "single_scan": lambda s: len(s.image_paths) == 1,
        "t1_only_multi": lambda s: mods(s) == {"T1w"} and len(s.image_paths) > 1,
        "t1_t2": lambda s: {"T1w", "T2w"} <= mods(s),
        "t1_flair": lambda s: {"T1w", "FLAIR"} <= mods(s),
        "t1_dwi": lambda s: "T1w" in mods(s) and bool(mods(s) & (dwi - {"ADC"})),
        "t1_adc": lambda s: "T1w" in mods(s) and "ADC" in mods(s),
        "ge3_modalities": lambda s: len(mods(s)) >= 3,
        "anisotropic": lambda s: anisotropic(s) and len(s.image_paths) > 1,
        "anisotropic_diffusion": lambda s: anisotropic(s) and bool(mods(s) & dwi),
        "blocked_by_audit": lambda s: not s.coreg_allowed,
    }
    rng = random.Random(seed)
    chosen: List[str] = []
    seen = set()
    # Larger requests deepen every stratum rather than only widening the top-up tail, so a pilot
    # is a scaled-up version of the smoke and not the smoke plus an arbitrary remainder. The
    # floor of 3 keeps the established 20-50 session smoke selection bit-for-bit unchanged.
    depth = per_category if max_sessions is None else max(per_category, max_sessions // 20)
    for name in sorted(categories):
        matching = sorted(s.key for s in sessions if categories[name](s))
        rng.shuffle(matching)
        for key in matching[:depth]:
            if key not in seen:
                seen.add(key)
                chosen.append(key)
    scenario_count = len(chosen)
    # Then dataset breadth, shuffled so a truncated smoke is not always the alphabetical head.
    by_dataset: dict = {}
    for session in sessions:
        if len(session.image_paths) > 1:
            by_dataset.setdefault(session.dataset, []).append(session.key)
    datasets = sorted(by_dataset)
    rng.shuffle(datasets)
    for dataset in datasets:
        key = sorted(by_dataset[dataset])[0]
        if key not in seen:
            seen.add(key)
            chosen.append(key)

    if max_sessions is not None and len(chosen) > max_sessions:
        if scenario_count > max_sessions:
            logger.warning(
                "stratified smoke: %d sessions are needed for scenario coverage but max_sessions=%d; "
                "keeping all scenarios and ignoring the cap.",
                scenario_count,
                max_sessions,
            )
        chosen = chosen[: max(max_sessions, scenario_count)]
    elif max_sessions is not None and len(chosen) < max_sessions:
        # Scenario coverage plus one session per dataset is a floor, not a target. A larger
        # request (the pilot) tops up from the remaining registerable sessions, shuffled under
        # the same seed so the set stays reproducible from the manifest alone.
        remaining = sorted(s.key for s in sessions if s.key not in seen and len(s.image_paths) > 1)
        rng.shuffle(remaining)
        chosen.extend(remaining[: max_sessions - len(chosen)])
    return sorted(chosen)


def _apply_session_filters(args, sessions: List[Session], config: CoregConfig) -> List[Session]:
    """Shared --datasets / --stratified-smoke / --smoke / --limit / chunk filtering."""
    if getattr(args, "datasets", None):
        wanted = set(args.datasets)
        sessions = [s for s in sessions if s.dataset in wanted]
    if getattr(args, "stratified_smoke", False):
        keys = set(canonical_smoke_keys(sessions, seed=getattr(args, "seed", 0), max_sessions=getattr(args, "smoke_size", 40)))
        sessions = [s for s in sessions if s.key in keys]
    if getattr(args, "chunk_id", None) is not None:
        sessions = manifest_mod.chunk_records(sessions, args.chunk_size, args.chunk_id)
    if getattr(args, "smoke", None):
        sessions = sessions[: args.smoke]
    if getattr(args, "limit", None) is not None:
        sessions = sessions[: args.limit]
    return sessions


def _resolve_sessions(args, input_root: str, output_root: str, config: CoregConfig) -> List[Session]:
    if config.canonical_manifest:
        # The canonical manifest defines the corpus; sessions are already deterministically
        # ordered by key, so chunking it directly gives disjoint, gap-free array shards.
        return _apply_session_filters(args, _canonical_sessions(config), config)

    manifest_file = getattr(args, "manifest", None) or _manifest_path(output_root)
    if getattr(args, "chunk_id", None) is not None:
        if not os.path.exists(manifest_file):
            raise SystemExit(f"chunked run requires a manifest; build it first: {manifest_file}")
        records = manifest_mod.read_manifest(manifest_file)
        records = manifest_mod.chunk_records(records, args.chunk_size, args.chunk_id)
        sessions = manifest_mod.records_to_sessions(records, input_root)
    elif os.path.exists(manifest_file):
        sessions = manifest_mod.records_to_sessions(manifest_mod.read_manifest(manifest_file), input_root)
    else:
        sessions = find_sessions(input_root, config.extensions)

    if getattr(args, "datasets", None):
        wanted = set(args.datasets)
        sessions = [s for s in sessions if s.dataset in wanted]
    if getattr(args, "stratified_smoke", False):
        records = manifest_mod.find_optional(
            input_root, manifest_file if os.path.exists(manifest_file) else None, config.extensions
        )
        keys = {r["session_key"] for r in manifest_mod.select_smoke(records)}
        sessions = [s for s in sessions if s.key in keys]
    if getattr(args, "smoke", None):
        sessions = sessions[: args.smoke]
    if getattr(args, "limit", None) is not None:
        sessions = sessions[: args.limit]
    return sessions


# --------------------------------------------------------------------------- #
# Subcommand: manifest / inventory
# --------------------------------------------------------------------------- #
def cmd_manifest(args) -> int:
    import pandas as pd

    _setup_logging(args.log_level)
    input_root, output_root = _require_roots(args)
    os.makedirs(output_root, exist_ok=True)
    records = manifest_mod.build_records(input_root, CoregConfig().extensions)
    manifest_mod.write_manifest(records, _manifest_path(output_root))
    summary = manifest_mod.inventory_summary(records)
    with open(os.path.join(output_root, "inventory_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    four_d = manifest_mod.four_d_rows(records)
    pd.DataFrame(four_d).to_csv(os.path.join(output_root, "inventory_4d.tsv"), sep="\t", index=False)
    print(json.dumps(summary, indent=2))
    if four_d:
        logger.warning(
            "%d 4D scans found across %d sessions (see inventory_4d.tsv). "
            "With --fail-on-4d (default) these sessions fail rather than drop data.",
            len(four_d),
            summary["sessions_with_4d"],
        )
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: run
# --------------------------------------------------------------------------- #
def make_runner(config: CoregConfig, dry_run: bool):
    """Build the registration runner for the configured backend.

    Both runners expose the same method surface, so nothing downstream branches on the backend.
    """
    if config.backend == "fsl":
        return fsl_mod.FSLRunner(
            dof=config.dof,
            cost=config.cost,
            interp=config.image_interp,
            target_spacing=float(config.target_iso_spacing[0]),
            init_from_header=config.init_from_header,
            allow_affine_ablation=config.allow_affine_ablation,
            angular_search_deg=config.angular_search_deg,
            threads=config.threads,
            dry_run=dry_run,
            max_log_chars=config.max_log_chars,
        )
    return FreeSurferRunner(
        threads=config.threads, dry_run=dry_run, freesurfer_home=config.freesurfer_home, max_log_chars=config.max_log_chars
    )


def backend_tool_report(config: CoregConfig) -> dict:
    """Tool-availability report for the configured backend."""
    return fsl_mod.check_tools() if config.backend == "fsl" else check_tools(config.freesurfer_home)


def _process_one(
    session: Session, render_qc: bool, config: CoregConfig, input_root: str, output_root: str, dry_run: bool
) -> dict:
    runner = make_runner(config, dry_run)
    qc_image_dir = os.path.join(output_root, "qc_images")
    return pipeline_mod.process_session(
        session=session,
        config=config,
        input_root=input_root,
        output_root=output_root,
        runner=runner,
        render_qc_image=render_qc,
        qc_image_dir=qc_image_dir if render_qc else None,
    )


def _qc_image_keys(sessions: List[Session], config: CoregConfig) -> set:
    if config.qc_image_render_all:
        # Smoke/pilot: render everything. The cases most worth looking at (lowest
        # foreground_retained_frac, worst NMI) are only identifiable *after* processing, so they
        # cannot be sampled for in advance.
        return {s.key for s in sessions}
    if config.qc_image_max_per_dataset <= 0:
        return set()
    chosen: set = set()
    by_dataset: dict = {}
    for s in sessions:
        by_dataset.setdefault(s.dataset, []).append(s)
    stride = max(config.qc_image_stride, 1)
    for group in by_dataset.values():
        chosen.update(s.key for s in group[::stride][: config.qc_image_max_per_dataset])
    return chosen


def cmd_run(args) -> int:
    _setup_logging(args.log_level)
    input_root, output_root = _require_roots(args)
    if os.path.realpath(input_root) == os.path.realpath(output_root):
        raise SystemExit("--output-root must differ from --input-root (raw data is never overwritten).")

    config = config_from_args(args)
    report = backend_tool_report(config)
    if not report["ok"] and not args.dry_run and not args.skip_tool_check:
        logger.error("%s backend not fully available: %s", config.backend, json.dumps(report["tools"]))
        logger.error("Load the toolbox (`module load fsl/6.0.4` for --backend fsl) or use --dry-run / --skip-tool-check.")
        return 1

    sessions = _resolve_sessions(args, input_root, output_root, config)
    todo = sessions if config.overwrite else [s for s in sessions if not is_session_done(output_root, s.key, config)]
    logger.info("%d sessions selected, %d to process, %d already done.", len(sessions), len(todo), len(sessions) - len(todo))
    if not todo:
        _write_shard_summary(args, output_root, [])
        return 0

    os.makedirs(output_root, exist_ok=True)
    qc_keys = _qc_image_keys(todo, config)
    worker = partial(_process_one, config=config, input_root=input_root, output_root=output_root, dry_run=args.dry_run)
    tasks = [(s, s.key in qc_keys) for s in todo]

    if args.num_workers <= 1 or args.dry_run:
        results = [worker(s, render) for s, render in tasks]
    else:
        with Pool(args.num_workers) as pool:
            results = pool.starmap(worker, tasks)

    counts: dict = {}
    for qc in results:
        counts[qc["status"]] = counts.get(qc["status"], 0) + 1
    logger.info("Shard complete: %s", json.dumps(counts))
    _write_shard_summary(args, output_root, results)
    return 0


def _write_shard_summary(args, output_root: str, results: List[dict]) -> None:
    """Per-shard status file (no shared writes across array tasks)."""
    chunk_id = getattr(args, "chunk_id", None)
    shard_dir = os.path.join(output_root, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    name = f"chunk_{chunk_id:05d}.json" if chunk_id is not None else "run_local.json"
    counts: dict = {}
    for qc in results:
        counts[qc["status"]] = counts.get(qc["status"], 0) + 1
    payload = {
        "chunk_id": chunk_id,
        "n": len(results),
        "counts": counts,
        "sessions": [{"session_key": r["session_key"], "status": r["status"], "reason": r.get("reason", "")} for r in results],
    }
    with open(os.path.join(shard_dir, name), "w") as handle:
        json.dump(payload, handle, indent=2)


# --------------------------------------------------------------------------- #
# Subcommand: aggregate
# --------------------------------------------------------------------------- #
def cmd_aggregate(args) -> int:
    import pandas as pd

    _setup_logging(args.log_level)
    input_root, output_root = _require_roots(args)
    config = config_from_args(args) if hasattr(args, "backend") else CoregConfig()
    manifest_file = _manifest_path(output_root)
    if config.canonical_manifest:
        # The corpus is defined by the canonical manifest, so the aggregate must iterate it too.
        # Falling back to a filesystem walk here would silently roll up whatever happens to be on
        # disk instead of the sample set the derivative is contracted to.
        keys = sorted(canonical_mod.group_by_session(load_canonical(config)))
    elif os.path.exists(manifest_file):
        keys = [r["session_key"] for r in manifest_mod.read_manifest(manifest_file)]
    else:
        keys = [s.key for s in find_sessions(input_root, CoregConfig().extensions)]

    rows, rejections, suspicious, statuses = [], [], [], {}
    for key in keys:
        qc = qc_mod.read_session_qc(os.path.join(output_root, key, "qc.json"))
        if qc is None:
            statuses["missing"] = statuses.get("missing", 0) + 1
            continue
        statuses[qc.get("status", "unknown")] = statuses.get(qc.get("status", "unknown"), 0) + 1
        rows.extend(qc_mod.session_qc_rows(qc))
        if qc.get("status") not in {"success"}:
            rejections.append(
                {"session_key": key, "dataset": qc.get("dataset"), "status": qc.get("status"), "reason": qc.get("reason", "")}
            )
        _collect_suspicious(qc, suspicious)

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(output_root, "coreg_qc.tsv"), sep="\t", index=False)
    pd.DataFrame(rejections).to_csv(os.path.join(output_root, "coreg_rejections.tsv"), sep="\t", index=False)
    pd.DataFrame(suspicious).to_csv(os.path.join(output_root, "coreg_suspicious.tsv"), sep="\t", index=False)
    fov = _fov_review_rows(rows, output_root)
    pd.DataFrame(fov).to_csv(os.path.join(output_root, "coreg_fov_review.tsv"), sep="\t", index=False)

    summary = {
        "statuses": statuses,
        "n_sessions": len(keys),
        "n_output_images": len(rows),
        "n_suspicious": len(suspicious),
        "foreground_retained_frac": _distribution([r.get("foreground_retained_frac") for r in rows]),
    }
    with open(os.path.join(output_root, "coreg_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    if fov:
        print(
            f"\ncoreg_fov_review.tsv: {len(fov)} moving scans ranked by foreground_retained_frac (lowest first).\n"
            "Open the montages of the lowest rows in all three planes and decide whether the lost\n"
            "foreground is brain or non-brain coverage. Thresholds are NOT calibrated yet."
        )
    return 0


def _distribution(values: List) -> dict:
    """Min/quantiles/max of a continuous diagnostic, for calibrating a threshold from evidence."""
    numeric = sorted(float(v) for v in values if isinstance(v, (int, float)))
    if not numeric:
        return {"n": 0}

    def q(p: float) -> float:
        return round(numeric[min(int(p * (len(numeric) - 1)), len(numeric) - 1)], 4)

    return {
        "n": len(numeric),
        "min": round(numeric[0], 4),
        "p05": q(0.05),
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "max": round(numeric[-1], 4),
    }


def _fov_review_rows(rows: List[dict], output_root: str) -> List[dict]:
    """Moving scans ranked by ``foreground_retained_frac``, lowest first, with their montage path.

    This is the worklist for calibrating the reference-FOV question: the metric alone cannot say
    whether a low value means lost brain or merely a wider non-brain acquisition box, so each row
    points at the tri-planar render that can.
    """
    review = []
    for row in rows:
        value = row.get("foreground_retained_frac")
        if row.get("is_reference") or not isinstance(value, (int, float)):
            continue
        base = os.path.join(
            output_root, "qc_images", str(row.get("dataset", "")), f"{row.get('subject')}_{row.get('session')}"
        )
        review.append(
            {
                "foreground_retained_frac": value,
                "session_key": row.get("session_key"),
                "dataset": row.get("dataset"),
                "modality": row.get("modality_canonical") or row.get("modality"),
                "reference": row.get("reference"),
                "modality_status": row.get("modality_status"),
                "translation_norm_mm": row.get("translation_norm_mm"),
                "rotation_deg": row.get("rotation_deg"),
                "nmi_improvement": row.get("nmi_improvement"),
                # Matches both layouts: "<base>_reg.png" (sampled) and "<base>_reg_<MOD>.png"
                # (rendered per moving modality under --qc-images-all).
                "qc_montage_glob": f"{base}*_reg*.png",
                "output_path": row.get("output_path"),
            }
        )
    return sorted(review, key=lambda r: r["foreground_retained_frac"])


def _collect_suspicious(qc: dict, out: List[dict]) -> None:
    reasons = []
    if qc.get("status") in {"partial", "failed"}:
        reasons.append(qc["status"])
    if qc.get("reference", {}).get("anisotropic_reference"):
        reasons.append("anisotropic_reference")
    for flag in qc.get("output_qc", {}).get("flags", {}):
        reasons.append(flag)
    if reasons:
        out.append(
            {
                "session_key": qc.get("session_key"),
                "dataset": qc.get("dataset"),
                "status": qc.get("status"),
                "reasons": ",".join(sorted(set(reasons))),
            }
        )


# --------------------------------------------------------------------------- #
# Subcommand: resubmit
# --------------------------------------------------------------------------- #
def cmd_resubmit(args) -> int:
    _setup_logging(args.log_level)
    input_root, output_root = _require_roots(args)
    config = config_from_args(args)
    manifest_file = _manifest_path(output_root)
    if not os.path.exists(manifest_file):
        raise SystemExit(f"no manifest at {manifest_file}; run `manifest` first.")
    records = manifest_mod.read_manifest(manifest_file)
    chunk_size = args.chunk_size
    pending_chunks = set()
    pending_sessions = []
    for idx, record in enumerate(records):
        if not is_session_done(output_root, record["session_key"], config):
            pending_sessions.append(record["session_key"])
            pending_chunks.add(idx // chunk_size)
    out = {
        "n_pending_sessions": len(pending_sessions),
        "n_total_chunks": manifest_mod.n_chunks(len(records), chunk_size),
        "pending_chunks": sorted(pending_chunks),
        "array_spec": _array_spec(sorted(pending_chunks)),
    }
    print(json.dumps(out, indent=2))
    if args.write:
        with open(os.path.join(output_root, "pending_chunks.txt"), "w") as handle:
            handle.write(",".join(str(c) for c in sorted(pending_chunks)))
    return 0


def _array_spec(chunks: List[int]) -> str:
    return ",".join(str(c) for c in chunks)


def cmd_check_tools(args) -> int:
    report = fsl_mod.check_tools() if getattr(args, "backend", "freesurfer") == "fsl" else check_tools(args.freesurfer_home)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


# --------------------------------------------------------------------------- #
# Subcommand: audit-sessions
# --------------------------------------------------------------------------- #
def cmd_audit_sessions(args) -> int:
    """Audit ``(dataset, participant_id, session_id)`` before any registration is permitted."""
    _setup_logging(args.log_level)
    config = config_from_args(args)
    if not config.canonical_manifest:
        raise SystemExit("audit-sessions requires --canonical-manifest: the corpus must be defined, not discovered.")
    output_root = os.path.abspath(args.output_root) if args.output_root else os.path.dirname(config.canonical_manifest)

    rows = load_canonical(config)
    verdicts = audit_from_config(config, rows)
    summary = audit_mod.summarise(verdicts, rows, metadata_available=bool(config.cleaned_metadata))
    summary["canonical_manifest"] = config.canonical_manifest
    summary["canonical_manifest_sha256"] = canonical_mod.sha256_of(config.canonical_manifest)
    summary["examples_by_code"] = audit_mod.group_examples(verdicts)

    paths = audit_mod.write_audit(verdicts, summary, os.path.join(output_root, "manifests"))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("examples_ambiguous",)}, indent=2))
    print(f"\nwrote {paths['sessions']}\n      {paths['ambiguous']}\n      {paths['summary']}")
    if not config.cleaned_metadata:
        logger.warning(
            "No --cleaned-metadata given: only structural identity was checked. Scanner/demographic "
            "contradictions inside a session key CANNOT be detected without it."
        )
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: derivative-manifest
# --------------------------------------------------------------------------- #
def cmd_derivative_manifest(args) -> int:
    """Emit one derivative row per canonical sample and enforce the accounting invariant."""
    _setup_logging(args.log_level)
    config = config_from_args(args)
    if not config.canonical_manifest:
        raise SystemExit("derivative-manifest requires --canonical-manifest.")
    _, output_root = _require_roots(args)

    rows = load_canonical(config)
    verdicts = {v.session_key: v for v in audit_from_config(config, rows)}
    records = derivative_mod.build_records(rows, verdicts, output_root)

    manifest_dir = os.path.join(output_root, "manifests")
    manifest_path = os.path.join(manifest_dir, "derivative_manifest.tsv")
    try:
        summary = derivative_mod.assert_accounting(records, expected_samples=len(rows))
        accounting_ok, accounting_error = True, ""
    except derivative_mod.AccountingError as exc:
        summary = derivative_mod.accounting(records)
        accounting_ok, accounting_error = False, str(exc)

    sha = derivative_mod.write(records, manifest_path)
    metadata = {
        "derivative_manifest_version": derivative_mod.MANIFEST_VERSION,
        "derivative_manifest_sha256": sha,
        "source_canonical_manifest": config.canonical_manifest,
        "source_canonical_manifest_sha256": canonical_mod.sha256_of(config.canonical_manifest),
        "expected_samples": len(rows),
        "accounting": summary,
        "accounting_ok": accounting_ok,
        "accounting_error": accounting_error,
        "config": _config_provenance(config),
        "environment": provenance_mod.environment_provenance(config.freesurfer_home or ""),
    }
    derivative_mod.write_metadata(metadata, os.path.join(manifest_dir, "derivative_metadata.json"))

    print(json.dumps({"accounting": summary, "accounting_ok": accounting_ok, "sha256": sha}, indent=2))
    print(f"\nwrote {manifest_path}\n      {os.path.join(manifest_dir, 'derivative_metadata.json')}")
    if not accounting_ok:
        logger.error("ACCOUNTING VIOLATION: %s", accounting_error)
        return 1
    return 0


def _config_provenance(config: CoregConfig) -> dict:
    """The configuration fields that change what the derivative *is*."""
    return {
        "backend": config.backend,
        "dof": config.dof,
        "cost": config.cost,
        "image_interp": config.image_interp,
        "init_from_header": config.init_from_header,
        "angular_search_deg": list(config.angular_search_deg) if config.angular_search_deg else None,
        "fallback_unconstrained_retry": config.fallback_unconstrained_retry,
        "inverse_direction_rescue": config.inverse_direction_rescue,
        "header_only_fallback": config.header_only_fallback,
        "target_iso_spacing": list(config.target_iso_spacing) if config.target_iso_spacing else None,
        "do_skull_strip": config.do_skull_strip,
        "reference_policy": config.reference_policy,
        "max_reference_spacing": config.max_reference_spacing,
        "require_explicit_session": config.require_explicit_session,
        "expected_manifest_sha256": config.expected_manifest_sha256,
    }


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Intra-session multimodal MRI co-registration (FOMO26).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_man = sub.add_parser("manifest", help="Build the session manifest + 4D inventory.")
    add_io_args(p_man)
    p_man.set_defaults(func=cmd_manifest)

    p_inv = sub.add_parser("inventory", help="Alias of manifest (build + print inventory).")
    add_io_args(p_inv)
    p_inv.set_defaults(func=cmd_manifest)

    p_aud = sub.add_parser("audit-sessions", help="Audit session identity before permitting registration.")
    add_io_args(p_aud)
    add_backend_args(p_aud)
    add_config_args(p_aud)
    p_aud.set_defaults(func=cmd_audit_sessions)

    p_der = sub.add_parser("derivative-manifest", help="One row per canonical sample + accounting invariant.")
    add_io_args(p_der)
    add_backend_args(p_der)
    add_config_args(p_der)
    p_der.set_defaults(func=cmd_derivative_manifest)

    p_run = sub.add_parser("run", help="Process sessions (optionally one array shard).")
    add_io_args(p_run)
    add_backend_args(p_run)
    add_config_args(p_run)
    p_run.add_argument("--seed", type=int, default=0, help="Seed for the stratified smoke selection.")
    p_run.add_argument(
        "--smoke-size",
        type=int,
        default=40,
        help="Cap on --stratified-smoke sessions (scenario coverage is never sacrificed to it).",
    )
    p_run.add_argument("--manifest", default=None, help="Manifest path (default: <output-root>/manifest.json).")
    p_run.add_argument("--chunk-size", type=int, default=200)
    p_run.add_argument("--chunk-id", type=int, default=None, help="Array shard id; requires a manifest.")
    p_run.add_argument("--datasets", nargs="*", default=None)
    p_run.add_argument("--smoke", type=int, default=None, help="Process only the first N sessions.")
    p_run.add_argument("--stratified-smoke", action="store_true", help="Process the stratified smoke set.")
    p_run.add_argument("--limit", type=int, default=None)
    p_run.add_argument("--num-workers", type=int, default=8)
    p_run.add_argument("--overwrite", action="store_true")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--skip-tool-check", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_agg = sub.add_parser("aggregate", help="Collect per-session QC into global TSV/JSON.")
    add_io_args(p_agg)
    p_agg.set_defaults(func=cmd_aggregate)

    p_res = sub.add_parser("resubmit", help="List chunk ids still needing work.")
    add_io_args(p_res)
    add_backend_args(p_res)
    add_config_args(p_res)
    p_res.add_argument("--chunk-size", type=int, default=200)
    p_res.add_argument("--write", action="store_true", help="Also write pending_chunks.txt.")
    p_res.set_defaults(func=cmd_resubmit)

    p_chk = sub.add_parser("check-tools", help="Report backend tool availability (FreeSurfer or FSL).")
    add_io_args(p_chk)
    add_backend_args(p_chk)
    p_chk.set_defaults(func=cmd_check_tools)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
