"""Provenance capture: environment, versions, and per-command logs.

Every external command's full invocation is recorded (see
:class:`~asparagus_preprocessing.session_coreg.freesurfer.CommandResult`), and
each session stores the environment it ran in so a result can be reproduced or
audited later.
"""

import getpass
import logging
import os
import platform
import subprocess
import sys
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def freesurfer_version(freesurfer_home: str = "") -> str:
    """Best-effort FreeSurfer version string (from build-stamp or the binary)."""
    home = freesurfer_home or os.environ.get("FREESURFER_HOME", "")
    stamp = os.path.join(home, "build-stamp.txt") if home else ""
    if stamp and os.path.exists(stamp):
        try:
            with open(stamp) as handle:
                return handle.read().strip()
        except OSError:
            pass
    try:
        out = subprocess.run(["mri_convert", "--version"], capture_output=True, text=True, timeout=30)
        line = (out.stdout or out.stderr).strip().splitlines()
        if line:
            return line[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


@lru_cache(maxsize=1)
def git_sha() -> str:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(
            ["git", "-C", here, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


@lru_cache(maxsize=1)
def package_version() -> str:
    try:
        import asparagus_preprocessing  # noqa: F401

        return getattr(sys.modules.get("asparagus_preprocessing"), "__version__", "0.1.6")
    except Exception:  # noqa: BLE001
        return "unknown"


def environment_provenance(freesurfer_home: str = "") -> dict:
    """Snapshot of the runtime environment for a session's QC record."""
    return {
        "hostname": platform.node(),
        "user": _safe(getpass.getuser),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "freesurfer_version": freesurfer_version(freesurfer_home),
        "git_sha": git_sha(),
        "package_version": package_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "argv": " ".join(sys.argv),
    }


def _safe(fn):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None
