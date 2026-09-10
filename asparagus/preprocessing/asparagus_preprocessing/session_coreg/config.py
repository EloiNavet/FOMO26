from dataclasses import dataclass, field
from typing import List, Optional, Tuple

REFERENCE_POLICIES = ("anisotropy_aware", "fomo50k_legacy")
INTERP_CHOICES = ("trilin", "cubic", "nearest")
#: FSL speaks a different interpolation vocabulary; ``spline`` is cubic, matching the P0 kernel.
FSL_INTERP_CHOICES = ("spline", "trilinear", "nearestneighbour", "sinc")
BACKENDS = ("freesurfer", "fsl")
#: Per-backend default interpolation. FreeSurfer keeps its historical ``trilin``; FSL defaults to
#: cubic so a P1 output differs from its P0 counterpart by registration alone, not by kernel.
DEFAULT_INTERP = {"freesurfer": "trilin", "fsl": "spline"}
BACKEND_INTERP_CHOICES = {"freesurfer": INTERP_CHOICES, "fsl": FSL_INTERP_CHOICES}

#: Diffusion contrasts. Rigid registration does not correct EPI distortion, so these are judged
#: with their own similarity threshold and are never allowed to justify more degrees of freedom.
DWI_MODALITIES = ("ADC", "DWI_B0", "DWI_B1000", "DWI_TRACE", "DWI")


@dataclass
class QCThresholds:
    """Warning/rejection thresholds for strict output QC.

    *Rejection-level* checks (a session cannot be ``success`` if any fail):
    output unreadable, not 3D, non-finite, empty, wrong orientation, spacing off
    the isotropic target, or shape/affine not shared across the session.

    *Warning-level* checks (recorded, flag the session for visual QC, but do not
    block ``success`` unless ``escalate_warnings`` is set): implausible brain
    volume/fraction, extreme nonzero fraction, worsened NMI, large motion.
    """

    spacing_tol_mm: float = 0.05
    min_nonzero_fraction: float = 0.005
    max_nonzero_fraction: float = 0.999
    min_brain_volume_cm3: float = 50.0  # low bound admits infants/paediatrics
    max_brain_volume_cm3: float = 3000.0
    min_brain_fraction: float = 0.03  # brain voxels / nonzero voxels
    nmi_improvement_warn: float = 0.0  # warn if after - before < this
    #: Diffusion is judged separately: EPI distortion is geometric, unmodelled by a rigid fit, so
    #: a small NMI drop on ADC/DWI is expected and must not be read as a failed registration.
    nmi_improvement_warn_dwi: float = -0.02
    translation_warn_mm: float = 30.0
    rotation_warn_deg: float = 20.0

    # --- rigid-transform sanity (REJECTION level) ---------------------------
    # A 6-DOF fit must be a rotation plus a translation. A reflection, a scale or a shear is a
    # failed registration, not a warning: it means the transform absorbed anatomy. det alone is
    # not sufficient — a shear can have det exactly 1 — so orthogonality is checked directly.
    rigid_determinant_tol: float = 1e-3  # |det - 1|
    rigid_scale_tol: float = 1e-3  # max |singular value - 1|
    rigid_orthogonality_tol: float = 1e-4  # max |L'L - I|
    translation_reject_mm: float = 60.0
    rotation_reject_deg: float = 45.0
    #: PROVISIONAL AND NON-SCIENTIFIC. Do not cite this number.
    #:
    #: ``foreground_retained_frac`` is recorded per moving scan as a **continuous diagnostic**;
    #: this value only decides when the aggregate flags a session for a human to look at. It was
    #: not derived from any experiment. The only evidence behind it is three real clinical
    #: sessions (PT002_Nigerian_Clinical, T1w reference with FLAIR/T2w/DWI moving) whose measured
    #: values were 0.54-0.96 — enough to show that a near-1.0 threshold fires on everything and
    #: carries no signal, and nothing more.
    #:
    #: It is a WARNING and must stay one until the real-FSL smoke has been reviewed: the metric
    #: cannot yet distinguish "lost brain" from "lost non-brain acquisition coverage", and only
    #: the tri-planar renders can. Do not promote it to a rejection, and do not run the smoke
    #: with ``--escalate-warnings`` (which would make every warning block success).
    min_foreground_retained_frac: float = 0.60

    escalate_warnings: bool = False  # if True, warnings also block success


@dataclass
class CoregConfig:
    """Configuration for the intra-session co-registration pipeline.

    Defaults target FOMO26: anisotropy-aware reference selection and 1 mm
    isotropic outputs. Set ``reference_policy='fomo50k_legacy'`` and
    ``target_iso_spacing=None`` to reproduce plain FOMO50K.
    """

    extensions: List[str] = field(default_factory=lambda: [".nii.gz", ".nii"])

    # --- backend ------------------------------------------------------------
    backend: str = "freesurfer"  # "fsl" is the Jean-Zay rail; FreeSurfer is unavailable there
    dof: int = 6  # rigid; passed explicitly so no toolbox default can decide it
    cost: str = "normmi"
    init_from_header: bool = True  # FSL: -usesqform (inputs are RAS with valid sforms)
    allow_affine_ablation: bool = False  # the ONLY way to reach dof != 6, and never a default
    #: FSL: bound on the optimiser's rotation search, in degrees, applied to x/y/z alike
    #: (``-searchrx/-searchry/-searchrz lo hi``). ``None`` keeps FLIRT's own +/-90 default.
    #:
    #: This is a *search* constraint, not a QC threshold: it changes where the optimiser is
    #: allowed to look, never what counts as an acceptable result. It exists because a rigid
    #: intra-session fit that needs more than a few tens of degrees is not a plausible fit but a
    #: wrong basin, and normmi on a low-information contrast (a derived ADC map, say) can reach
    #: one. Keep it explicit and revertible so it can be ablated against the unconstrained run.
    angular_search_deg: Optional[Tuple[int, int]] = None
    #: When ``angular_search_deg`` is set and a moving scan fails output QC, retry that scan once
    #: with FLIRT's unconstrained search and keep the retry only if it passes the same checks.
    #:
    #: This is failure recovery, not model selection. A passing first attempt is final and the
    #: retry never runs, so two acceptable registrations are never compared and no threshold is
    #: relaxed for either. It exists because the paired smoke showed the constrained and
    #: unconstrained searches failing on *disjoint* scans -- 5 wrong-basin failures fixed by the
    #: bound, 1 caused by it -- so neither search alone dominates, while attempting the second
    #: only after the first is already lost costs nothing on the scans that succeed.
    fallback_unconstrained_retry: bool = False
    #: Last-resort optimiser rescue, run **only** after every forward attempt has already failed
    #: output QC: estimate the rigid fit with the two images swapped (the session's target grid as
    #: FLIRT ``-in``, the moving scan as ``-ref``), invert that matrix with ``convert_xfm
    #: -inverse``, and apply the inverse to the *original* moving scan onto the *same* target grid.
    #:
    #: This changes the estimation direction and nothing else -- same DOF, same cost, same angular
    #: search as attempt 1, same interpolation, same single final resampling, same target lattice,
    #: same QC. The moving scan never becomes the output lattice, so full reference FOV is kept.
    #:
    #: It exists because the paired reference-policy gate showed the two directions are not
    #: equally conditioned: with a thick slab as the *moving* image the optimiser can slide it
    #: along its own thick axis, and ``fomo50k_legacy`` avoided that failure only by electing the
    #: slab as reference -- which costs roughly half the session's field of view. Estimating in the
    #: better-conditioned direction and inverting is meant to buy the robustness without the FOV.
    #:
    #: Failure recovery, not model selection: a passing forward attempt is final and this never
    #: runs, so two acceptable registrations are never compared and no threshold is relaxed.
    inverse_direction_rescue: bool = False
    #: Last resort, run **only** after every optimised attempt has failed output QC: publish the
    #: moving scan where its own scanner header places it (``-applyxfm -usesqform``), resampled
    #: onto the same P0-derived target grid with the same kernel. No intensity optimisation.
    #:
    #: This is **not a registration** and is never recorded as one: the scan is marked
    #: ``registration_method='header_only'`` so the derivative can always be filtered on it.
    #:
    #: Why it is defensible where optimisation is not. On the 8 scans the pipeline definitively
    #: lost, an independent comparator (``fomo50k_legacy`` fitted the same pair in the opposite
    #: direction, with the slab as its reference) puts the true relationship 0.30-1.35 mm and
    #: 0.16-2.56 deg from the header -- and tri-planar QC confirms the header places ventricles,
    #: corpus callosum, brainstem and skull inside the reference's contours in all 8. Meanwhile
    #: FLIRT's own translation-only schedule and ``simple3D.sch`` both drove the brain out of the
    #: field of view entirely (165-394 mm, retention 0.0-0.28).
    #:
    #: Its failure mode is *under*-correction, never mis-correction: it cannot invent a rotation,
    #: which is exactly how the rejected inverse-direction rescue produced QC-passing anatomy
    #: errors of 11-27 deg. Note this does not make the header universally right -- across 187
    #: accepted thick-slice scans the median header-to-final correction is 3.46 deg (p90 11.15) --
    #: which is why this only ever runs on scans already lost, never in place of a fit.
    header_only_fallback: bool = False

    # --- corpus definition --------------------------------------------------
    # When set, sessions come from the canonical manifest instead of a filesystem walk, so the
    # derivative's sample set is the canonical corpus by construction.
    canonical_manifest: Optional[str] = None
    expected_manifest_sha256: Optional[str] = None
    cleaned_metadata: Optional[str] = None  # FOMO300K_cleaned/manifest.tsv, for the identity audit
    require_explicit_session: bool = False  # treat ses-01-style fallbacks as unproven identity

    # --- reference selection ------------------------------------------------
    reference_policy: str = "anisotropy_aware"
    reference_string: Optional[str] = None
    max_reference_spacing: float = 3.0  # mm; gate against thick-slice references
    w_max_spacing: float = 1.0  # anisotropy_aware score weights
    w_anisotropy: float = 1.0
    w_priority: float = 0.5
    reference_gate_penalty: float = 1000.0

    # --- steps --------------------------------------------------------------
    do_coregister: bool = True
    do_skull_strip: bool = True
    synthseg_robust: bool = True

    target_iso_spacing: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0)
    #: One policy for all intensity images. ``None`` resolves to the backend's default so the
    #: FreeSurfer vocabulary is never silently handed to FSL, or vice versa.
    image_interp: Optional[str] = None
    mask_interp: str = "nearest"  # masks/segmentations stay nearest

    threads: int = 1
    min_scans_for_coreg: int = 2

    # --- 4D handling --------------------------------------------------------
    max_ndim: int = 3
    fail_on_4d: bool = True  # production default: never silently drop 4D

    # --- QC -----------------------------------------------------------------
    save_brain_mask: bool = True
    compute_registration_metrics: bool = True  # NMI before/after (needs a pre-coreg resample)
    compute_edge_overlap: bool = False  # extra, slower structural check
    qc: QCThresholds = field(default_factory=QCThresholds)
    expected_orientation: str = "RAS"

    qc_image_max_per_dataset: int = 25
    qc_image_stride: int = 20
    qc_render_suspicious: bool = True  # always render partial/failed/flagged sessions
    #: Render every processed session, ignoring the sampling stride. Meant for the smoke and the
    #: pilot: with ~40 sessions the strided sample would silently skip most of them, and the whole
    #: point of the smoke is to eyeball the full spread — in particular the *lowest*
    #: ``foreground_retained_frac`` cases, which cannot be selected in advance because the metric
    #: only exists after a scan has been resampled.
    qc_image_render_all: bool = False

    # --- bookkeeping --------------------------------------------------------
    overwrite: bool = False
    #: Keep a rejected candidate image under ``<session>/_quarantine/`` instead of deleting it.
    #: Either way it never occupies a canonical derivative path; this only decides whether the
    #: evidence survives for inspection.
    quarantine_failed_outputs: bool = True
    clean_work_on_success: bool = True
    preserve_work_on_failure: bool = True
    max_log_chars: int = 40000
    freesurfer_home: Optional[str] = None

    def __post_init__(self) -> None:
        if self.reference_policy not in REFERENCE_POLICIES:
            raise ValueError(f"reference_policy must be one of {REFERENCE_POLICIES}, got {self.reference_policy!r}.")
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}.")
        if self.image_interp is None:
            self.image_interp = DEFAULT_INTERP[self.backend]
        allowed = BACKEND_INTERP_CHOICES[self.backend]
        if self.image_interp not in allowed:
            raise ValueError(f"image_interp for backend {self.backend!r} must be one of {allowed}, got {self.image_interp!r}.")
        if self.dof != 6 and not self.allow_affine_ablation:
            raise ValueError(
                f"dof={self.dof} requested but the production contract is rigid 6-DOF. Set "
                "allow_affine_ablation=True to run an explicitly named, non-default ablation."
            )
        if self.inverse_direction_rescue:
            # Fail closed rather than silently degrading to the forward-only ladder: the rescue is
            # defined by FSL's own frame convention and inversion tool, and a FreeSurfer LTA
            # carries its own geometry, so the operation is not the same one there.
            if self.backend != "fsl":
                raise ValueError(
                    "inverse_direction_rescue is defined for the FSL backend only (it inverts a "
                    f"FLIRT matrix with convert_xfm); backend={self.backend!r}."
                )
            if not self.do_coregister:
                raise ValueError("inverse_direction_rescue requires do_coregister=True; there is nothing to rescue.")
        if self.header_only_fallback:
            if self.backend != "fsl":
                raise ValueError(f"header_only_fallback is defined for the FSL backend only; backend={self.backend!r}.")
            if not self.do_coregister:
                raise ValueError("header_only_fallback requires do_coregister=True; there is nothing to fall back from.")
            if not self.init_from_header:
                # Without -usesqform the fallback would resample from FLIRT's corner-origin frames
                # rather than from the scanner header, which is a different operation entirely.
                raise ValueError("header_only_fallback requires init_from_header=True (it IS the header alignment).")
        if self.backend == "fsl":
            if self.do_skull_strip:
                raise ValueError(
                    "the FSL backend never skull-strips: the primary registered derivative is "
                    "registration-only (no SynthSeg, no masking, no N4, no defacing). Pass "
                    "do_skull_strip=False (CLI: --no-skull-strip)."
                )
            if self.target_iso_spacing is None:
                raise ValueError("the FSL backend requires an isotropic target so sessions share one lattice.")
        if self.mask_interp != "nearest":
            raise ValueError("mask_interp must be 'nearest' to keep masks/segmentations binary.")
        if self.target_iso_spacing is not None:
            if len(self.target_iso_spacing) != 3 or any(s <= 0 for s in self.target_iso_spacing):
                raise ValueError("target_iso_spacing must be three positive values or None.")
            self.target_iso_spacing = tuple(float(s) for s in self.target_iso_spacing)
        if self.threads < 1:
            raise ValueError("threads must be >= 1.")
        if self.max_reference_spacing <= 0:
            raise ValueError("max_reference_spacing must be positive.")
