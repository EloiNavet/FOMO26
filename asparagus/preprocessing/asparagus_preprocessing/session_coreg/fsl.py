"""FSL backend for the intra-session pipeline: rigid 6-DOF FLIRT on Jean Zay.

FreeSurfer is **not available on Jean Zay** (only ``ants/2.3.x`` and ``fsl/6.0.4``), so the
FreeSurfer path in :mod:`~asparagus_preprocessing.session_coreg.freesurfer` cannot run there.
:class:`FSLRunner` exposes the *same method surface* as ``FreeSurferRunner`` so
:mod:`~asparagus_preprocessing.session_coreg.pipeline` can drive either without knowing which
toolbox produced a transform. No orchestration, QC, manifest, sharding or resume logic is
duplicated.

Scientific contract of this backend
-----------------------------------

**Rigid, 6 DOF, always.** ``-dof 6`` is passed explicitly; FSL's own default is never relied on.
A 12-DOF affine is reachable only through an explicitly named ablation
(``allow_affine_ablation=True``) and is never the production path. Scale and shear are not
"more flexible" here — they silently absorb real anatomical difference into the transform.

**One interpolation.** The session's shared lattice is built by the P0 kernel
(:mod:`asparagus_preprocessing.p0_kernel`) from the reference scan, so the reference and every
single-scan passthrough are bit-for-bit identical to their P0 tensors. A moving scan is then
resampled **once**, straight from its native NIfTI onto that lattice. There is no
native -> iso1 -> registration -> iso1 chain.

**No matrix composition.** A FLIRT matrix is only meaningful for the exact ``-in``/``-ref`` pair
it was fitted on. Estimating against the native reference and re-applying against a 1 mm grid
would require composing matrices across two different FSL frames. Instead both the fit and the
application use the *same* ``-ref`` (the final grid), so ``-applyxfm -init`` is exactly correct
and needs no composition. This is why :attr:`register_against_target_grid` is True.

**Nothing but registration.** No skull-stripping, no masking, no bias correction, no defacing,
no intensity harmonisation, no atlas/MNI step, no deformable warp. ``mask``-style flags are
rejected rather than ignored. DWI/ADC volumes are the already-curated scalar maps and are never
regenerated; a rigid fit does not correct EPI distortion and this backend never escalates the
degrees of freedom to make diffusion "look" better aligned.
"""

import logging
import os
import re
import shutil
import subprocess
import time
from asparagus_preprocessing import p0_kernel
from asparagus_preprocessing.session_coreg.backend import BackendError, CommandResult
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

REQUIRED_TOOLS = ("flirt",)
#: Needed only when ``inverse_direction_rescue`` is enabled, so it is reported separately rather
#: than making every FSL run depend on it.
INVERSE_RESCUE_TOOLS = ("convert_xfm",)

#: Production defaults. ``dof`` is 6 and is asserted, not defaulted-into.
FLIRT_DOF = 6
FLIRT_COST = "normmi"
#: Cubic spline, matching the P0 kernel's ``order=3``. Using trilinear here would make P1 differ
#: from P0 by both registration *and* a smoother interpolation kernel, so a downstream P0-vs-P1
#: difference could not be attributed to registration alone.
FLIRT_INTERP = "spline"

INTERP_CHOICES = ("spline", "trilinear", "nearestneighbour", "sinc")
COST_CHOICES = ("normmi", "mutualinfo", "corratio", "normcorr", "leastsq", "labeldiff", "bbr")

#: Flags that would turn this into something other than a pure registration.
FORBIDDEN_FLAGS = ("-inweight", "-refweight", "-wmseg", "-fieldmap", "-nosearch")

_COST_RE = re.compile(r"final(?:\s+\w+)?\s*cost\s*[:=]?\s*([0-9.eE+-]+)", re.IGNORECASE)


class FSLError(BackendError):
    """Raised when an FSL command fails or produces no output."""


def flirt_version() -> str:
    """Best-effort FSL version string, from ``$FSLDIR/etc/fslversion`` or ``flirt -version``."""
    fsldir = os.environ.get("FSLDIR", "")
    stamp = os.path.join(fsldir, "etc", "fslversion") if fsldir else ""
    if stamp and os.path.exists(stamp):
        try:
            with open(stamp) as handle:
                return handle.read().strip()
        except OSError:
            pass
    try:
        out = subprocess.run(["flirt", "-version"], capture_output=True, text=True, timeout=30)
        line = (out.stdout or out.stderr).strip().splitlines()
        if line:
            return line[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def check_tools(fsldir: Optional[str] = None) -> dict:
    """Report which FSL tools are resolvable, without running a registration."""
    home = fsldir or os.environ.get("FSLDIR")
    report = {"fsldir": home, "fsl_version": flirt_version(), "tools": {}}

    def resolve(tool: str) -> Optional[str]:
        found = shutil.which(tool)
        if found is None and home:
            candidate = os.path.join(home, "bin", tool)
            found = candidate if os.path.exists(candidate) else None
        return found

    for tool in REQUIRED_TOOLS:
        report["tools"][tool] = resolve(tool)
    report["ok"] = all(report["tools"].values())
    report["inverse_rescue_tools"] = {tool: resolve(tool) for tool in INVERSE_RESCUE_TOOLS}
    report["inverse_rescue_ok"] = all(report["inverse_rescue_tools"].values())
    return report


#: Sentinel: ``None`` means "unconstrained search", so it cannot double as "unspecified".
_USE_RUNNER_DEFAULT = object()


def flirt_available() -> bool:
    return shutil.which("flirt") is not None


def flirt_estimate_command(
    moving: str,
    reference: str,
    out_matrix: str,
    dof: int = FLIRT_DOF,
    cost: str = FLIRT_COST,
    init_from_header: bool = True,
    angular_search_deg: Optional[Tuple[int, int]] = None,
) -> List[str]:
    """The exact FLIRT command that *estimates* the transform.

    ``-omat`` is mandatory: a registration whose transform was not saved cannot be audited,
    re-applied, or decomposed for QC. No output image is requested here — the resampling happens
    once, in :func:`flirt_apply_command`, onto the final grid.

    ``angular_search_deg`` bounds the optimiser's rotation search on all three axes. ``None``
    leaves FLIRT's own default (+/-90 deg per axis) untouched, which is the pre-ablation
    behaviour; a tuple emits ``-searchrx/-searchry/-searchrz lo hi``. It constrains where the
    optimiser may look, and changes no QC threshold, cost function, DOF or interpolation.
    """
    cmd = [
        "flirt",
        "-in",
        str(moving),
        "-ref",
        str(reference),
        "-omat",
        str(out_matrix),
        "-dof",
        str(int(dof)),
        "-cost",
        str(cost),
    ]
    if angular_search_deg is not None:
        lo, hi = normalise_angular_search(angular_search_deg)
        for axis in ("x", "y", "z"):
            cmd += [f"-searchr{axis}", str(lo), str(hi)]
    if init_from_header:
        # Both images are RAS with a valid sform and share the scanner frame, so the headers are
        # the correct starting point. Without this FLIRT starts from corner-origin FSL frames and
        # can begin far off when the two fields of view differ.
        cmd.append("-usesqform")
    return cmd


def normalise_angular_search(value) -> Tuple[int, int]:
    """Validate an angular-search bound and return it as ``(lo, hi)`` integers.

    FLIRT takes whole degrees. The bound is symmetric in practice but is not forced to be, so an
    asymmetric ablation stays expressible; ``lo < hi`` and both within +/-180 are required so a
    typo cannot silently produce an unsearchable or wrap-around range.
    """
    try:
        lo, hi = (int(round(float(v))) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"angular_search_deg must be a (lo, hi) pair of degrees, got {value!r}") from exc
    if lo >= hi:
        raise ValueError(f"angular_search_deg must satisfy lo < hi, got ({lo}, {hi})")
    if lo < -180 or hi > 180:
        raise ValueError(f"angular_search_deg must lie within [-180, 180], got ({lo}, {hi})")
    return lo, hi


def flirt_apply_command(
    moving: str,
    target_grid: str,
    out_image: str,
    matrix: Optional[str],
    interp: str = FLIRT_INTERP,
    init_from_header: bool = True,
) -> List[str]:
    """The exact FLIRT command that resamples ``moving`` onto ``target_grid``.

    ``matrix=None`` means "header alignment only", which is what the NMI-before volume needs:
    the same grid, the same interpolation, no registration.
    """
    cmd = [
        "flirt",
        "-in",
        str(moving),
        "-ref",
        str(target_grid),
        "-out",
        str(out_image),
        "-applyxfm",
        "-interp",
        str(interp),
    ]
    if matrix:
        cmd += ["-init", str(matrix)]
    elif init_from_header:
        cmd.append("-usesqform")
    return cmd


def convert_xfm_invert_command(in_matrix: str, out_matrix: str) -> List[str]:
    """FSL's own documented inversion: ``convert_xfm -omat <out> -inverse <in>``.

    ``convert_xfm``'s usage text states the contract in one line -- *"-inverse (Reference image
    must be the one originally used)"* -- i.e. the inverted matrix's ``-in``/``-ref`` pair is the
    original pair swapped. That is exactly what the rescue needs: a matrix estimated with the
    target grid as ``-in`` and the moving scan as ``-ref`` inverts to one that FLIRT will accept
    with ``-in <moving> -ref <target grid> -applyxfm -init``.
    """
    return ["convert_xfm", "-omat", str(out_matrix), "-inverse", str(in_matrix)]


#: The inverted matrix is cross-checked against a plain numerical inverse. FSL writes ``.mat``
#: files with limited precision, so this is a contract check ("convert_xfm did the operation we
#: believe it does"), not a precision requirement; a real disagreement means the semantics are not
#: what this pipeline assumes and the scan must fail rather than be published.
INVERSION_CROSSCHECK_ATOL = 1e-4


@dataclass
class FSLRunner:
    """Executes FSL commands with logging, timing, provenance and dry-run.

    Method names mirror :class:`~asparagus_preprocessing.session_coreg.freesurfer.FreeSurferRunner`
    so the pipeline can hold either.
    """

    transform_suffix: ClassVar[str] = ".mat"
    #: A FLIRT matrix is bound to the ``-ref`` it was fitted against, so the fit must happen on
    #: the final target grid. See the module docstring.
    register_against_target_grid: ClassVar[bool] = True
    #: FSL exposes a documented matrix inversion (``convert_xfm -inverse``) whose frames are the
    #: fitted pair swapped, which is what makes the reverse-direction rescue a pure re-estimation
    #: rather than a composition. A backend without that contract must not be handed the rescue.
    supports_inverse_direction: ClassVar[bool] = True
    name: ClassVar[str] = "fsl"

    dof: int = FLIRT_DOF
    cost: str = FLIRT_COST
    interp: str = FLIRT_INTERP
    target_spacing: float = p0_kernel.TARGET_SPACING_MM
    init_from_header: bool = True
    allow_affine_ablation: bool = False
    #: Optimiser rotation bound in degrees, applied to all three axes. ``None`` = FLIRT's own
    #: default search. This is a search constraint, never a QC threshold.
    angular_search_deg: Optional[Tuple[int, int]] = None

    threads: int = 1
    dry_run: bool = False
    fsldir: Optional[str] = None
    timeout: Optional[float] = None
    max_log_chars: int = 40000
    command_log: List[dict] = field(default_factory=list)
    _env: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.dof != FLIRT_DOF and not self.allow_affine_ablation:
            raise ValueError(
                f"dof={self.dof} requested but the production contract is rigid {FLIRT_DOF}-DOF. "
                "Set allow_affine_ablation=True to run a named, non-default ablation."
            )
        if self.cost not in COST_CHOICES:
            raise ValueError(f"cost must be one of {COST_CHOICES}, got {self.cost!r}.")
        if self.interp not in INTERP_CHOICES:
            raise ValueError(f"interp must be one of {INTERP_CHOICES}, got {self.interp!r}.")
        if self.target_spacing <= 0:
            raise ValueError("target_spacing must be positive.")
        if self.angular_search_deg is not None:
            # Normalise once, at construction, so provenance and the emitted command cannot
            # disagree and an invalid bound fails before any registration runs.
            self.angular_search_deg = normalise_angular_search(self.angular_search_deg)
        self._env = dict(os.environ)
        if self.fsldir:
            self._env["FSLDIR"] = self.fsldir
        self._env.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")
        self._env.setdefault("OMP_NUM_THREADS", str(self.threads))

    # --- command execution ------------------------------------------------

    def run(self, cmd: Sequence[str], expect_output: Optional[str] = None) -> CommandResult:
        """Run one command; record provenance; raise on failure or missing output."""
        cmd = [str(part) for part in cmd]
        forbidden = [flag for flag in FORBIDDEN_FLAGS if flag in cmd]
        if forbidden:
            raise FSLError(f"refusing a command carrying non-registration flags {forbidden}: {' '.join(cmd)}")
        logger.debug("RUN: %s", " ".join(cmd))
        if self.dry_run:
            result = CommandResult(cmd=cmd, returncode=0, skipped=True)
            self.command_log.append(result.to_record(self.max_log_chars))
            return result

        started = time.perf_counter()
        proc = subprocess.run(cmd, env=self._env, capture_output=True, text=True, timeout=self.timeout)
        result = CommandResult(
            cmd=cmd,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_s=time.perf_counter() - started,
        )
        self.command_log.append(result.to_record(self.max_log_chars))
        if proc.returncode != 0:
            raise FSLError(f"{cmd[0]} failed (exit {proc.returncode}): {' '.join(cmd)}\n{proc.stderr[-2000:]}")
        if expect_output is not None and not _output_exists(expect_output):
            raise FSLError(f"{cmd[0]} produced no output at {expect_output}: {' '.join(cmd)}")
        return result

    def _record_local(self, argv: List[str], started: float) -> CommandResult:
        """Log an in-process (nibabel) step so provenance covers non-subprocess work too."""
        result = CommandResult(cmd=argv, returncode=0, duration_s=time.perf_counter() - started, skipped=self.dry_run)
        self.command_log.append(result.to_record(self.max_log_chars))
        return result

    # --- pipeline steps ---------------------------------------------------

    def reorient_to_ras(self, src: str, dst: str, is_label: bool = False) -> CommandResult:
        """Materialise ``dst`` for ``src`` **without reorienting or interpolating**.

        The P0 kernel applies ``as_closest_canonical`` at resample time, so reorienting here
        would only add a NIfTI round-trip whose float32 header re-encoding perturbs the affine —
        and that would break the bit-for-bit equality between a P1 reference and its P0 tensor.
        A hard link (copy on fallback) keeps the file byte-identical while giving the pipeline
        the path it expects.
        """
        started = time.perf_counter()
        if not self.dry_run:
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            if os.path.exists(dst):
                os.remove(dst)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copyfile(src, dst)
        return self._record_local(["<passthrough:ras-already-canonical>", src, dst], started)

    def make_iso_target(self, ref: str, dst: str, spacing: Sequence[float], interp: str = FLIRT_INTERP) -> CommandResult:
        """Build the session's shared lattice with the **P0 kernel**, not with FSL.

        ``spacing`` must be isotropic and equal to the P0 target; anything else would silently
        create a corpus that is not comparable to P0.
        """
        values = [float(s) for s in spacing]
        if len({round(v, 9) for v in values}) != 1:
            raise FSLError(f"target spacing must be isotropic to match the P0 contract, got {values}.")
        if abs(values[0] - self.target_spacing) > 1e-9:
            raise FSLError(f"target spacing {values[0]} != backend target_spacing {self.target_spacing}.")
        started = time.perf_counter()
        if not self.dry_run:
            p0_kernel.write_p0_resampled(ref, dst, values[0])
        result = self._record_local(["<p0_kernel:p0_resample>", ref, dst, f"spacing={values[0]}", "order=3"], started)
        if not self.dry_run and not _output_exists(dst):
            raise FSLError(f"p0_kernel produced no output at {dst}")
        return result

    def coregister(self, mov: str, ref: str, lta: str, angular_search_deg=_USE_RUNNER_DEFAULT) -> tuple:
        """Estimate the rigid transform ``mov -> ref`` and save it to ``lta`` (a ``.mat``).

        ``ref`` is the final target grid (see :attr:`register_against_target_grid`).

        ``angular_search_deg`` overrides the runner's own bound for this call alone, which is how
        the fallback ladder runs a second, unconstrained attempt without mutating shared state (a
        runner is reused across the scans of a session). ``None`` is a meaningful value here --
        "no bound" -- so the default is a sentinel, not ``None``.
        """
        policy = self.angular_search_deg if angular_search_deg is _USE_RUNNER_DEFAULT else angular_search_deg
        cmd = flirt_estimate_command(
            mov,
            ref,
            lta,
            dof=self.dof,
            cost=self.cost,
            init_from_header=self.init_from_header,
            angular_search_deg=policy,
        )
        result = self.run(cmd, expect_output=lta)
        cost = None
        matches = _COST_RE.findall(result.stdout + "\n" + result.stderr)
        if matches:
            try:
                cost = float(matches[-1])
            except ValueError:
                cost = None
        return result, cost

    def header_alignment(self, mov: str, targ: str, out: str, out_matrix: str, interp: str = FLIRT_INTERP) -> CommandResult:
        """Resample ``mov`` onto ``targ`` using the scanner headers alone, recording the transform.

        No intensity optimisation happens: ``-applyxfm -usesqform`` places the scan where its
        sform says it is, which for two scans of one session is the relationship the scanner
        itself recorded. ``-omat`` is written in the same call, so the resulting transform is
        auditable and decomposable exactly like a fitted one -- it decomposes to the world-space
        identity, which is precisely the claim being made.

        One command, one interpolation, the same target grid and the same kernel as every other
        attempt. This is not a registration and must never be recorded as one.
        """
        cmd = flirt_apply_command(mov, targ, out, matrix=None, interp=interp, init_from_header=True)
        cmd += ["-omat", str(out_matrix)]
        return self.run(cmd, expect_output=out)

    def invert_transform(self, transform: str, out_matrix: str) -> CommandResult:
        """Invert a FLIRT ``.mat`` with ``convert_xfm -inverse``, then verify it really inverted.

        The tool is FSL's own, so the operation is theirs to define; the cross-check exists
        because the whole rescue rests on the inverted matrix meaning "moving -> target grid", and
        a silently wrong matrix would still produce a plausible-looking image. Any disagreement
        beyond :data:`INVERSION_CROSSCHECK_ATOL` raises instead of publishing.
        """
        result = self.run(convert_xfm_invert_command(transform, out_matrix), expect_output=out_matrix)
        if self.dry_run:
            return result

        import numpy as np
        from asparagus_preprocessing.session_coreg import metrics as metrics_mod

        forward = metrics_mod.parse_flirt_mat(transform)
        inverse = metrics_mod.parse_flirt_mat(out_matrix)
        if forward is None or inverse is None:
            raise FSLError(f"could not parse the matrices around convert_xfm ({transform} -> {out_matrix}).")
        try:
            expected = np.linalg.inv(forward)
        except np.linalg.LinAlgError as exc:
            raise FSLError(f"the estimated matrix {transform} is singular and cannot be inverted: {exc}") from exc
        drift = float(np.abs(np.asarray(inverse) - expected).max())
        if drift > INVERSION_CROSSCHECK_ATOL:
            raise FSLError(
                f"convert_xfm -inverse disagrees with a numerical inverse by {drift:.3e} "
                f"(> {INVERSION_CROSSCHECK_ATOL:.0e}) for {transform}; refusing to apply it."
            )
        return result

    def apply_transform(
        self, mov: str, targ: str, out: str, lta: Optional[str] = None, interp: str = FLIRT_INTERP
    ) -> CommandResult:
        """Resample ``mov`` onto ``targ``. This is the single stored interpolation."""
        cmd = flirt_apply_command(mov, targ, out, lta, interp=self.interp, init_from_header=self.init_from_header)
        return self.run(cmd, expect_output=out)

    # --- transform QC -----------------------------------------------------

    def transform_metrics(self, transform: str, moving: Optional[str] = None, reference: Optional[str] = None) -> dict:
        """World-space decomposition of a FLIRT ``.mat`` (see ``metrics.flirt_metrics``)."""
        from asparagus_preprocessing.session_coreg import metrics as metrics_mod

        if self.dry_run:
            return {"parsed": False, "world_space": False, "dry_run": True}
        return metrics_mod.flirt_metrics(transform, moving=moving, reference=reference)

    # --- steps this backend refuses --------------------------------------

    def synthseg(self, *args, **kwargs):
        raise FSLError(
            "the FSL backend never skull-strips: the primary registered derivative must stay "
            "non-destructive. Run with skull stripping disabled (do_skull_strip=False)."
        )

    binarize = synthseg
    apply_mask = synthseg

    def environment(self) -> dict:
        """Backend identity for the provenance record."""
        return {"backend": self.name, "fsl_version": flirt_version(), "fsldir": self.fsldir or os.environ.get("FSLDIR")}


def _output_exists(path: str) -> bool:
    """True if FSL wrote ``path`` under any of the extensions ``FSLOUTPUTTYPE`` may pick."""
    if os.path.exists(path):
        return True
    stem = path
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    return any(os.path.exists(stem + ext) for ext in (".nii.gz", ".nii"))
