"""Run one test tier, with a preflight that fails loudly instead of quietly.

The tier a reader is most likely to get wrong is `optional_natten`. Those four modules import a
CUDA extension, and without it pytest exits 2 with four collection errors -- which is a failure,
but an opaque one, and the temptation is to "fix" it with a skip. A skipped suite reports success
while testing nothing, so this runner refuses instead: it checks that natten is importable *and*
pinned to the expected version before pytest is invoked at all, and exits non-zero with a sentence
saying what is wrong.

Every other tier is a plain pytest invocation with FOMO26_TIER set; they are routed through here
so that "how do I run tier X" has one answer.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REQUIRED_NATTEN = "0.21.0"


def _load_manifest() -> dict:
    import importlib.util

    spec = importlib.util.spec_from_file_location("_fomo26_tier_manifest", REPO / "tests" / "test_tier_contract.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TIER_ASSIGNMENTS, module.TIERS


def _natten_preflight() -> str | None:
    """Return an error sentence, or None when natten is present at the pinned version."""
    try:
        import natten  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - a native extension can fail in many ways
        return (
            f"natten is not importable ({type(exc).__name__}: {exc}). The optional_natten tier "
            f"requires natten=={REQUIRED_NATTEN}; install it or run the fast tier instead. "
            "A missing dependency is not a passing test run."
        )
    try:
        from importlib.metadata import version

        found = version("natten")
    except Exception as exc:  # noqa: BLE001
        return f"natten is importable but its version could not be determined ({exc})."
    if found != REQUIRED_NATTEN:
        return f"natten {found} is installed but this tier is pinned to {REQUIRED_NATTEN}."
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tier")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("pytest_args", nargs="*", default=[])
    args = parser.parse_args(argv)

    assignments, declared = _load_manifest()
    members = sorted(p for p, t in assignments.items() if t == args.tier)
    # Validate against the *declared* tiers, not against the tiers that happen to have members:
    # a tier can legitimately be declared and empty, and that is not an unknown tier.
    if args.tier not in declared:
        print(f"unknown tier {args.tier!r}; declared tiers: {sorted(declared)}", file=sys.stderr)
        return 2

    if not members:
        # Retiring MedViT removed every NATTEN-dependent module. The tier stays declared so the
        # contract still catches one arriving later, but an empty tier has nothing to preflight:
        # a missing optional dependency is not a failure when nothing asks for it.
        print(f"tier={args.tier} files=0\n0 assigned tests; nothing to run.")
        return 0

    if args.tier == "optional_natten":
        problem = _natten_preflight()
        if problem:
            print(f"optional_natten preflight failed: {problem}", file=sys.stderr)
            return 1
        # Name the four files explicitly rather than relying on tier deselection, so this command
        # runs exactly the modules the manifest assigns and nothing else.
        cmd = [sys.executable, "-m", "pytest", *members]
    else:
        cmd = [sys.executable, "-m", "pytest"]

    if args.collect_only:
        cmd.append("--co")
    cmd.extend(args.pytest_args)

    env = {**os.environ, "FOMO26_TIER": args.tier}
    env.setdefault("PYTHONPATH", str(REPO))
    print(f"tier={args.tier} files={len(members)}\ncommand: FOMO26_TIER={args.tier} {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
