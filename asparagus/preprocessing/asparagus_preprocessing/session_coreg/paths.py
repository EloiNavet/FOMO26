"""Path-safety helpers.

The pipeline must never be able to write into the raw input tree. All checks
resolve symlinks with ``realpath`` and use ``commonpath`` so that neither
nesting direction nor symlink aliasing can slip through.
"""

import os


class UnsafeIOError(ValueError):
    """Raised when the input/output roots are the same or nested/aliased."""


def _real(path: str) -> str:
    return os.path.realpath(os.path.abspath(path))


def _is_within(child: str, parent: str) -> bool:
    """True if ``child`` is ``parent`` or lives inside it (post-realpath)."""
    try:
        return os.path.commonpath([child, parent]) == parent
    except ValueError:
        # Different drives / uncomparable -> not nested.
        return False


def assert_safe_io(input_root: str, output_root: str) -> tuple:
    """Validate the input/output roots, returning their realpaths.

    Rejects, after resolving symlinks:
      * ``output == input`` (including symlink-equivalent paths);
      * ``output`` nested inside ``input`` (would write into raw data);
      * ``input`` nested inside ``output`` (raw data inside the write target).
    """
    real_in = _real(input_root)
    real_out = _real(output_root)
    if real_in == real_out:
        raise UnsafeIOError(f"output_root and input_root resolve to the same path ({real_in}); raw data is never overwritten.")
    if _is_within(real_out, real_in):
        raise UnsafeIOError(
            f"output_root ({real_out}) is nested inside input_root ({real_in}); refusing to write into raw data."
        )
    if _is_within(real_in, real_out):
        raise UnsafeIOError(
            f"input_root ({real_in}) is nested inside output_root ({real_out}); refusing (raw data under write target)."
        )
    return real_in, real_out
