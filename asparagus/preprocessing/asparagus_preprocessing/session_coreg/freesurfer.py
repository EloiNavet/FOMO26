"""Thin, testable wrappers around the FreeSurfer command-line tools.

Every external command goes through :meth:`FreeSurferRunner.run`, which records
full provenance (argv, return code, stdout, stderr, duration) into
``command_log``. Exact commands follow FOMO50K ``pre_process.sh``; the isotropic
target and the unified interpolation policy are the FOMO26 additions.

Interpolation is configured once (``config.image_interp``) and mapped to each
tool's own flag name so the reference and moving images are resampled
identically; masks/segmentations always use nearest-neighbour.
"""

import logging
import os
import re
import shutil
import subprocess
import time
from asparagus_preprocessing.session_coreg import provenance
from asparagus_preprocessing.session_coreg.backend import BackendError, CommandResult  # noqa: F401 (re-export)
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Sequence

logger = logging.getLogger(__name__)

REQUIRED_TOOLS = ("mri_convert", "mri_coreg", "mri_vol2vol", "mri_synthseg", "mri_mask", "mri_binarize")

_COST_RE = re.compile(r"(?:final cost|mincost)\s*[:=]?\s*([0-9.eE+-]+)", re.IGNORECASE)

# Canonical interpolation -> per-tool flag values.
_VOL2VOL_INTERP = {"trilin": "trilin", "cubic": "cubic", "nearest": "nearest"}
_CONVERT_RESAMPLE = {"trilin": "interpolate", "cubic": "cubic", "nearest": "nearest"}


def vol2vol_interp(canonical: str) -> str:
    return _VOL2VOL_INTERP[canonical]


def convert_resample_type(canonical: str) -> str:
    return _CONVERT_RESAMPLE[canonical]


class FreeSurferError(BackendError):
    """Raised when a FreeSurfer command fails or produces no output.

    Subclasses :class:`~asparagus_preprocessing.session_coreg.backend.BackendError` so the
    pipeline can catch either toolbox's failure with one clause; ``except FreeSurferError``
    elsewhere keeps working unchanged.
    """


def check_tools(freesurfer_home: Optional[str] = None) -> dict:
    """Report which required tools are resolvable, without running them."""
    home = freesurfer_home or os.environ.get("FREESURFER_HOME")
    report = {"freesurfer_home": home, "freesurfer_version": provenance.freesurfer_version(home or ""), "tools": {}}
    for tool in REQUIRED_TOOLS:
        resolved = shutil.which(tool)
        if resolved is None and home:
            candidate = os.path.join(home, "bin", tool)
            resolved = candidate if os.path.exists(candidate) else None
        report["tools"][tool] = resolved
    report["ok"] = bool(home) and all(report["tools"].values())
    return report


@dataclass
class FreeSurferRunner:
    """Executes FreeSurfer commands with logging, timing, provenance and dry-run."""

    #: Transform-file extension this toolbox writes (FSL writes ``.mat``).
    transform_suffix: ClassVar[str] = ".lta"
    #: Whether the transform must be estimated against the final target grid rather than the
    #: reference's native grid. FreeSurfer LTAs carry their own source/destination geometry, so
    #: ``mri_vol2vol`` can retarget them freely and this stays False. FSL matrices cannot: they
    #: live in the ``-in``/``-ref`` pair's own coordinate frames.
    register_against_target_grid: ClassVar[bool] = False

    threads: int = 1
    dry_run: bool = False
    freesurfer_home: Optional[str] = None
    timeout: Optional[float] = None
    max_log_chars: int = 40000
    command_log: List[dict] = field(default_factory=list)
    _env: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._env = dict(os.environ)
        if self.freesurfer_home:
            self._env["FREESURFER_HOME"] = self.freesurfer_home
        self._env.setdefault("OMP_NUM_THREADS", str(self.threads))

    def run(self, cmd: Sequence[str], expect_output: Optional[str] = None) -> CommandResult:
        """Run one command; record provenance; raise on failure or missing output."""
        cmd = [str(part) for part in cmd]
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
            raise FreeSurferError(f"{result.name} failed (exit {proc.returncode}): {' '.join(cmd)}\n{proc.stderr[-2000:]}")
        if expect_output is not None and not os.path.exists(expect_output):
            raise FreeSurferError(f"{result.name} produced no output at {expect_output}: {' '.join(cmd)}")
        return result

    # --- pipeline steps -------------------------------------------------

    def reorient_to_ras(self, src: str, dst: str, is_label: bool = False) -> CommandResult:
        cmd = ["mri_convert", src, dst, "--out_orientation", "RAS"]
        if is_label:
            cmd += ["--resample_type", "nearest"]
        return self.run(cmd, expect_output=dst)

    def make_iso_target(self, ref: str, dst: str, spacing: Sequence[float], interp: str = "trilin") -> CommandResult:
        """Resample ``ref`` onto an isotropic grid (same interp policy as moving images)."""
        cmd = [
            "mri_convert",
            ref,
            dst,
            "--voxsize",
            *[f"{s:g}" for s in spacing],
            "--resample_type",
            convert_resample_type(interp),
        ]
        return self.run(cmd, expect_output=dst)

    def coregister(self, mov: str, ref: str, lta: str, angular_search_deg=None) -> tuple[CommandResult, Optional[float]]:
        # ``angular_search_deg`` is part of the shared runner surface because the FSL backend can
        # bound FLIRT's rotation search. mri_coreg exposes no equivalent, so it is accepted and
        # ignored here rather than silently changing what this backend does.
        del angular_search_deg
        cmd = ["mri_coreg", "--mov", mov, "--ref", ref, "--reg", lta, "--threads", str(self.threads)]
        result = self.run(cmd, expect_output=lta)
        cost = None
        matches = _COST_RE.findall(result.stdout + "\n" + result.stderr)
        if matches:
            try:
                cost = float(matches[-1])
            except ValueError:
                cost = None
        return result, cost

    def apply_transform(
        self, mov: str, targ: str, out: str, lta: Optional[str] = None, interp: str = "trilin"
    ) -> CommandResult:
        cmd = ["mri_vol2vol", "--mov", mov, "--o", out, "--targ", targ, "--interp", vol2vol_interp(interp)]
        cmd += ["--reg", lta] if lta else ["--regheader"]
        return self.run(cmd, expect_output=out)

    def synthseg(self, src: str, seg: str, robust: bool = True) -> CommandResult:
        cmd = ["mri_synthseg", "--i", src, "--o", seg, "--threads", str(self.threads)]
        if robust:
            cmd.append("--robust")
        return self.run(cmd, expect_output=seg)

    def binarize(self, src: str, out: str, min_value: float = 1.0) -> CommandResult:
        cmd = ["mri_binarize", "--i", src, "--o", out, "--min", f"{min_value:g}", "--binval", "1"]
        return self.run(cmd, expect_output=out)

    def apply_mask(self, src: str, mask: str, out: str) -> CommandResult:
        return self.run(["mri_mask", src, mask, out], expect_output=out)

    # --- transform QC ---------------------------------------------------

    def transform_metrics(self, transform: str, moving: Optional[str] = None, reference: Optional[str] = None) -> dict:
        """Decomposed metrics for a transform this backend produced.

        An LTA already encodes a world-space transform, so ``moving``/``reference`` are unused
        here; they exist because FSL matrices live in a voxel-scaled frame and need both images
        to be converted back to millimetres.
        """
        from asparagus_preprocessing.session_coreg import metrics as metrics_mod

        return metrics_mod.lta_metrics(transform)
