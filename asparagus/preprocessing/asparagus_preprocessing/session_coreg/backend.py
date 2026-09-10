"""The two things every registration backend shares: a command record and an error type.

Deliberately not a plugin framework. The pipeline drives a backend by duck typing — FreeSurfer and
FSL simply expose the same method names — and the only thing that genuinely has to be common is
the exception it raises, so a failing external command fails *one modality* rather than crashing
the whole session. Before this existed, ``FSLError`` slipped past ``except FreeSurferError`` and
took the session down with it.
"""

from dataclasses import dataclass
from typing import List


class BackendError(RuntimeError):
    """A registration toolbox command failed or produced no output."""


@dataclass
class CommandResult:
    """One external (or in-process) step, with everything needed to reproduce and audit it."""

    cmd: List[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    skipped: bool = False  # True in dry-run

    @property
    def name(self) -> str:
        return self.cmd[0] if self.cmd else ""

    def to_record(self, max_chars: int = 40000) -> dict:
        return {
            "argv": self.cmd,
            "returncode": self.returncode,
            "duration_s": round(self.duration_s, 3),
            "skipped": self.skipped,
            "stdout": self.stdout[-max_chars:],
            "stderr": self.stderr[-max_chars:],
        }
