"""Session-level diffusion output selection (Decision level B).

Given every classified diffusion scan for one ``(dataset, participant, session)``,
decide which curated outputs should exist -- final ``ADC``, final ``DWI_B1000``, and
optional ``DWI_TRACE`` / ``DWI_B0`` -- and how each should be produced (direct copy
vs. synthesis). Source files are preferred within a single run and recorded for
provenance. Shells at b >= 4000 are never used for clinical outputs.
"""

from __future__ import annotations

from . import config
from .classify import ScanClassification
from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class ScanRecord:
    """One classified diffusion scan within a session."""

    relpath: str  # curated-relative path (identifier)
    cls: ScanClassification
    run_id: Optional[str] = None
    abspath: Optional[str] = None

    @property
    def diffusion_class(self) -> str:
        return self.cls.diffusion_class

    @property
    def b_value(self) -> Optional[float]:
        return self.cls.b_value


# Derivations that use an existing acquisition directly (single source, no computation).
# Everything else combines several shells and goes through the synthesis path.
DIRECT_USE_DERIVATIONS = ("copy", "copy_trace_as_b1000", "near_shell_direct")


@dataclass
class OutputPlan:
    output_modality: str
    derivation_type: str  # copy | copy_trace_as_b1000 | near_shell_direct |
    # synth_b1000_from_b0_shell | synth_b1000_loginterp |
    # synth_adc_from_b0_shell
    source_records: list[ScanRecord]
    source_bvals: list[Optional[float]]
    run_id: Optional[str]
    confidence: str
    reason: str
    qc_flags: list[str] = field(default_factory=list)

    @property
    def source_relpaths(self) -> list[str]:
        return [r.relpath for r in self.source_records]

    @property
    def synthesized(self) -> bool:
        """True only when the output is *computed* from several shells.

        ``near_shell_direct`` uses a single existing near-b1000 acquisition as-is — it is a direct
        use, like ``copy``, not a synthesis. Classifying it as synthesized sent it down the
        multi-source synthesis path, where ``len(sources) < 2`` made it fail as
        ``skipped_missing_source``, so the channel was silently never materialized.
        """
        return self.derivation_type not in DIRECT_USE_DERIVATIONS


@dataclass
class SessionSelection:
    adc: Optional[OutputPlan] = None
    b1000: Optional[OutputPlan] = None
    trace: Optional[OutputPlan] = None
    b0: Optional[OutputPlan] = None

    def outputs(self) -> list[OutputPlan]:
        return [p for p in (self.adc, self.b1000, self.trace, self.b0) if p is not None]


def _by_class(scans: Sequence[ScanRecord], *classes: str) -> list[ScanRecord]:
    wanted = set(classes)
    out = [s for s in scans if s.diffusion_class in wanted]
    return sorted(out, key=lambda s: (s.run_id or "", s.relpath))


def _pick_same_run(primary: ScanRecord, pool: Sequence[ScanRecord]) -> Optional[ScanRecord]:
    """Prefer a pool member sharing ``primary``'s run; else the first available."""
    if not pool:
        return None
    same = [s for s in pool if primary.run_id is not None and s.run_id == primary.run_id]
    return (same or list(pool))[0]


def _nearest_shell(shells: Sequence[ScanRecord]) -> Optional[ScanRecord]:
    usable = [s for s in shells if s.b_value is not None]
    if not usable:
        return None
    return min(usable, key=lambda s: config.b1000_preference_rank(s.b_value))


def _select_b1000(scans: Sequence[ScanRecord]) -> Optional[OutputPlan]:
    exact = _by_class(scans, config.DWI_B1000)
    exact_pure = [s for s in exact if not s.cls.provenance_is_trace]
    exact_trace = [s for s in exact if s.cls.provenance_is_trace]
    near = _by_class(scans, config.DWI_NEAR_B1000)
    b0s = _by_class(scans, config.DWI_B0)

    # 1. Prefer a genuine (non-trace) exact b1000, then a trace-provenance b1000.
    if exact_pure:
        src = exact_pure[0]
        return OutputPlan(
            config.OUT_DWI_B1000, "copy", [src], [src.b_value], src.run_id, "high", "exact b1000 available; used directly"
        )
    if exact_trace:
        src = exact_trace[0]
        return OutputPlan(
            config.OUT_DWI_B1000,
            "copy_trace_as_b1000",
            [src],
            [src.b_value],
            src.run_id,
            "high",
            "trace image at clinical b1000 used directly as DWI_B1000",
            ["provenance_is_trace"],
        )

    # 2. Near-b1000 shells: synth from b0 if available, else log-interp a bracketing
    #    pair, else use the nearest shell directly (approximate).
    if near:
        best = _nearest_shell(near)
        if b0s and best is not None:
            b0 = _pick_same_run(best, b0s)
            qc = [] if b0.run_id == best.run_id else ["cross_run_b0_shell"]
            return OutputPlan(
                config.OUT_DWI_B1000,
                "synth_b1000_from_b0_shell",
                [b0, best],
                [b0.b_value, best.b_value],
                best.run_id,
                "high",
                f"synthesized b1000 from b0 + b{best.b_value:g}",
                qc,
            )
        below = [s for s in near if s.b_value is not None and s.b_value < 1000.0]
        above = [s for s in near if s.b_value is not None and s.b_value > 1000.0]
        if below and above:
            lo = max(below, key=lambda s: s.b_value)
            hi = min(above, key=lambda s: s.b_value)
            return OutputPlan(
                config.OUT_DWI_B1000,
                "synth_b1000_loginterp",
                [lo, hi],
                [lo.b_value, hi.b_value],
                lo.run_id or hi.run_id,
                "medium",
                f"log-interpolated b1000 from b{lo.b_value:g} and b{hi.b_value:g} (no b0 available)",
                ["no_b0_for_synth"],
            )
        if best is not None:
            return OutputPlan(
                config.OUT_DWI_B1000,
                "near_shell_direct",
                [best],
                [best.b_value],
                best.run_id,
                "medium",
                f"nearest clinical shell b{best.b_value:g} used directly (no b0)",
                ["approximate_b1000_no_b0"],
            )
    return None


def _select_adc(scans: Sequence[ScanRecord]) -> Optional[OutputPlan]:
    explicit = _by_class(scans, config.ADC)
    likely = _by_class(scans, config.ADC_LIKELY)
    near = _by_class(scans, config.DWI_NEAR_B1000, config.DWI_B1000)
    b0s = _by_class(scans, config.DWI_B0)

    # 1. Explicit ADC (highest confidence first), then ADC_LIKELY.
    if explicit:
        src = sorted(explicit, key=lambda s: (0 if s.cls.confidence == "high" else 1, s.relpath))[0]
        return OutputPlan(
            config.OUT_ADC, "copy", [src], [src.b_value], src.run_id, src.cls.confidence, "explicit ADC used directly"
        )
    if likely:
        src = likely[0]
        return OutputPlan(
            config.OUT_ADC,
            "copy",
            [src],
            [src.b_value],
            src.run_id,
            "low",
            "ADC_LIKELY (intensity heuristic) used as ADC",
            ["adc_likely_low_confidence"],
        )

    # 2. Synthesize ADC from b0 + nearest clinical shell.
    shells = [s for s in near if config.is_clinical_shell(s.b_value)]
    if b0s and shells:
        best = _nearest_shell(shells)
        b0 = _pick_same_run(best, b0s)
        qc = [] if b0.run_id == best.run_id else ["cross_run_b0_shell"]
        return OutputPlan(
            config.OUT_ADC,
            "synth_adc_from_b0_shell",
            [b0, best],
            [b0.b_value, best.b_value],
            best.run_id,
            "high",
            f"synthesized ADC from b0 + b{best.b_value:g}",
            qc,
        )
    return None


def _select_optional(scans: Sequence[ScanRecord], cls: str, out_mod: str) -> Optional[OutputPlan]:
    pool = _by_class(scans, cls)
    if not pool:
        return None
    src = pool[0]
    return OutputPlan(out_mod, "copy", [src], [src.b_value], src.run_id, "high", f"{cls} kept as optional {out_mod}")


def select_diffusion_outputs_for_session(scans: Sequence[ScanRecord]) -> SessionSelection:
    """Return the curated diffusion outputs for a single session."""
    return SessionSelection(
        adc=_select_adc(scans),
        b1000=_select_b1000(scans),
        trace=_select_optional(scans, config.DWI_TRACE, config.OUT_DWI_TRACE),
        b0=_select_optional(scans, config.DWI_B0, config.OUT_DWI_B0),
    )
