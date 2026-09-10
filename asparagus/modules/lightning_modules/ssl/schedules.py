"""Dynamic loss-weight schedules for the SSL trainer (P3.9 decomposition).

A single cosine ramp-up underlies every auxiliary-objective schedule; the module's
``get_dynamic_*`` methods are thin wrappers over :func:`cosine_ramp`.

Curriculum contract (added for Task 6)
--------------------------------------
:class:`ScheduleSpec` generalises the two functions below into one declarative per-component
contract — ``constant`` / ``linear`` / ``cosine``, with an absolute start step, a ramp, an optional
stop step and an optional decay.  It is a *generalisation*, not a replacement:
:func:`schedule_fraction` is proven bit-identical to :func:`cosine_ramp` and :func:`cosine_window`
for ``kind="cosine"``, so no historical schedule changes meaning.

Three properties matter and are all consequences of one design choice — the effective weight is a
pure function of the optimizer ``global_step``:

* **no pseudo-epoch dependence** — nothing here reads an epoch, a dataloader length, or
  ``steps_per_epoch``;
* **gradient-accumulation and DDP safe** — every rank computes the same number from the same step,
  with no communication;
* **resume-safe** — a resumed run at step *n* recomputes exactly what an uninterrupted run had at
  step *n*, because there is no schedule state to restore.

Ratios stay ratios in user configuration.  :func:`resolve_ratio_schedule` resolves them **once**
against the full planned training horizon using the same ``int(horizon * ratio)`` arithmetic as the
historical Hydra ``${eval:...}`` expressions, and :func:`resolved_provenance` serialises the
resulting absolute steps so a reader never has to recompute them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: The three admissible shapes. ``constant`` is a plain on/off window with no ramp and no decay.
SCHEDULE_KINDS = ("constant", "linear", "cosine")


def cosine_ramp(global_step: int, start_step: int, warmup_steps: int) -> float:
    """Cosine S-curve ramp from 0.0 to 1.0 between ``start_step`` and ``start_step + warmup_steps``.

    Returns 0.0 before ``start_step`` and 1.0 once warmup is complete (or immediately
    when ``warmup_steps <= 0``).
    """
    if global_step < start_step:
        return 0.0
    if warmup_steps <= 0:
        return 1.0
    progress = (global_step - start_step) / float(warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 - math.cos(math.pi * progress))


def cosine_window(
    global_step: int,
    start_step: int,
    warmup_steps: int,
    end_step: int = 0,
    decay_steps: int = 0,
) -> float:
    """Cosine ramp-up with an optional cosine ramp-down window.

    ``end_step <= 0`` disables the end window. With ``decay_steps <= 0`` the weight
    drops to zero at ``end_step``. Otherwise it decays smoothly to zero over
    ``[end_step, end_step + decay_steps]``.
    """

    weight = cosine_ramp(global_step, start_step, warmup_steps)
    if end_step <= 0:
        return weight
    if global_step < end_step:
        return weight
    if decay_steps <= 0:
        return 0.0
    if global_step >= end_step + decay_steps:
        return 0.0
    progress = (global_step - end_step) / float(decay_steps)
    progress = min(max(progress, 0.0), 1.0)
    return weight * 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass(frozen=True)
class ScheduleSpec:
    """One component's curriculum, in absolute optimizer steps.

    ``weight`` is the base (maximum) weight the component reaches at full ramp. The *fraction*
    returned by :func:`schedule_fraction` is always in ``[0, 1]``; :func:`effective_weight`
    multiplies it by ``weight``.

    ``stop_step is None`` (or ``<= 0``) means the component never stops. With a stop step and
    ``decay_steps <= 0`` the weight drops to zero exactly at ``stop_step``; otherwise it decays to
    zero over ``[stop_step, stop_step + decay_steps]``.
    """

    name: str = ""
    enabled: bool = False
    weight: float = 0.0
    start_step: int = 0
    ramp_steps: int = 0
    stop_step: int | None = None
    decay_steps: int = 0
    kind: str = "cosine"

    def __post_init__(self) -> None:
        if self.kind not in SCHEDULE_KINDS:
            raise ValueError(f"schedule kind must be one of {SCHEDULE_KINDS}, got {self.kind!r}")
        for field_name in ("start_step", "ramp_steps", "decay_steps"):
            if int(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} must be >= 0, got {getattr(self, field_name)!r}")
        if self.stop_step is not None and int(self.stop_step) > 0 and int(self.stop_step) < int(self.start_step):
            raise ValueError(f"stop_step {self.stop_step} precedes start_step {self.start_step}")
        # A "constant" schedule with a ramp or a decay is ambiguous — it would silently behave as a
        # linear or cosine one. Refuse it instead of guessing which the author meant.
        if self.kind == "constant" and (int(self.ramp_steps) or int(self.decay_steps)):
            raise ValueError(
                f"schedule {self.name!r}: kind='constant' forbids ramp_steps/decay_steps "
                f"(got ramp_steps={self.ramp_steps}, decay_steps={self.decay_steps}); "
                "use kind='linear' or kind='cosine' for a ramped schedule"
            )

    def as_record(self) -> dict:
        """JSON-serialisable provenance for this component's resolved absolute steps."""
        return {
            "name": self.name,
            "enabled": bool(self.enabled),
            "weight": float(self.weight),
            "start_step": int(self.start_step),
            "ramp_steps": int(self.ramp_steps),
            "stop_step": None if self.stop_step is None else int(self.stop_step),
            "decay_steps": int(self.decay_steps),
            "kind": self.kind,
        }


def _ramp_fraction(progress: float, kind: str) -> float:
    if kind == "constant":
        return 1.0
    if kind == "linear":
        return progress
    return 0.5 * (1.0 - math.cos(math.pi * progress))


def _decay_fraction(progress: float, kind: str) -> float:
    if kind == "constant":
        return 1.0
    if kind == "linear":
        return 1.0 - progress
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def schedule_fraction(spec: ScheduleSpec, global_step: int) -> float:
    """The component's schedule multiplier in ``[0, 1]`` at ``global_step``.

    Deliberately a pure function of the step: that is what makes the schedule resume-safe,
    accumulation-safe and identical on every DDP rank without any communication.

    For ``kind="cosine"`` this reproduces :func:`cosine_ramp` exactly when no stop step is set, and
    :func:`cosine_window` exactly when one is.
    """
    if not spec.enabled:
        return 0.0
    step = int(global_step)
    start = int(spec.start_step)
    if step < start:
        return 0.0

    ramp_steps = int(spec.ramp_steps)
    if ramp_steps <= 0:
        weight = 1.0
    else:
        progress = min(max((step - start) / float(ramp_steps), 0.0), 1.0)
        weight = _ramp_fraction(progress, spec.kind)

    stop = spec.stop_step
    if stop is None or int(stop) <= 0 or step < int(stop):
        return weight

    decay_steps = int(spec.decay_steps)
    if decay_steps <= 0:
        return 0.0
    if step >= int(stop) + decay_steps:
        return 0.0
    progress = min(max((step - int(stop)) / float(decay_steps), 0.0), 1.0)
    return weight * _decay_fraction(progress, spec.kind)


def effective_weight(spec: ScheduleSpec, global_step: int) -> float:
    """``spec.weight`` scaled by the schedule multiplier at ``global_step``."""
    return float(spec.weight) * schedule_fraction(spec, global_step)


def resolve_ratio_schedule(
    name: str,
    *,
    horizon_steps: int,
    enabled: bool = True,
    weight: float = 0.0,
    start_ratio: float = 0.0,
    ramp_ratio: float = 0.0,
    stop_ratio: float | None = None,
    decay_ratio: float = 0.0,
    kind: str = "cosine",
) -> ScheduleSpec:
    """Resolve ratio-of-horizon configuration into absolute optimizer steps, once.

    The arithmetic is deliberately ``int(horizon * ratio)`` — byte-identical to the historical Hydra
    ``${eval:"int(${training.steps} * ratio)"}`` expressions in ``configs/default_pretrain.yaml``.
    Changing it would silently re-interpret every existing ratio-configured run, so it does not
    change.
    """
    if int(horizon_steps) <= 0:
        raise ValueError(f"schedule {name!r}: horizon_steps must be positive, got {horizon_steps!r}")
    horizon = int(horizon_steps)
    return ScheduleSpec(
        name=name,
        enabled=bool(enabled),
        weight=float(weight),
        start_step=int(horizon * float(start_ratio)),
        ramp_steps=int(horizon * float(ramp_ratio)),
        stop_step=None if stop_ratio is None else int(horizon * float(stop_ratio)),
        decay_steps=int(horizon * float(decay_ratio)),
        kind=kind,
    )


def spec_from_mapping(name: str, mapping, *, horizon_steps: int | None = None) -> ScheduleSpec:
    """Build a spec from a plain mapping, accepting either absolute steps or ratios.

    Absolute keys win over ratio keys; mixing an absolute step with the matching ratio for the same
    boundary is refused rather than silently resolved one way.
    """
    data = dict(mapping or {})
    # Explicit pairing: naive string surgery on "ramp_steps" produces "ramp_ratios", which would
    # make the collision check silently never fire.
    equivalents = {
        "start_step": "start_ratio",
        "ramp_steps": "ramp_ratio",
        "stop_step": "stop_ratio",
        "decay_steps": "decay_ratio",
    }
    ratios = set(equivalents.values())
    collisions = sorted(
        f"{absolute_key}/{ratio_key}"
        for absolute_key, ratio_key in equivalents.items()
        if absolute_key in data and ratio_key in data
    )
    if collisions:
        raise ValueError(f"schedule {name!r}: both absolute and ratio forms given for {collisions}")

    if ratios & set(data):
        if horizon_steps is None:
            raise ValueError(f"schedule {name!r}: ratio configuration requires a training horizon")
        return resolve_ratio_schedule(
            name,
            horizon_steps=horizon_steps,
            enabled=data.get("enabled", True),
            weight=data.get("weight", 0.0),
            start_ratio=data.get("start_ratio", 0.0),
            ramp_ratio=data.get("ramp_ratio", 0.0),
            stop_ratio=data.get("stop_ratio"),
            decay_ratio=data.get("decay_ratio", 0.0),
            kind=data.get("kind", "cosine"),
        )
    return ScheduleSpec(
        name=name,
        enabled=bool(data.get("enabled", True)),
        weight=float(data.get("weight", 0.0)),
        start_step=int(data.get("start_step", 0)),
        ramp_steps=int(data.get("ramp_steps", 0)),
        stop_step=None if data.get("stop_step") is None else int(data["stop_step"]),
        decay_steps=int(data.get("decay_steps", 0)),
        kind=str(data.get("kind", "cosine")),
    )


def specs_from_config(config, *, horizon_steps: int | None = None) -> dict[str, ScheduleSpec]:
    """Turn a ``{component_name: {...}}`` config block into specs. ``None``/empty yields ``{}``.

    An empty result is the default and means "every component keeps its historical schedule": the
    trainer only consults this registry for components it finds in it.
    """
    if not config:
        return {}
    items = config.items() if hasattr(config, "items") else dict(config).items()
    return {str(name): spec_from_mapping(str(name), value, horizon_steps=horizon_steps) for name, value in items}


def resolved_provenance(specs: dict[str, ScheduleSpec], *, horizon_steps: int) -> dict:
    """Serialise resolved absolute steps so a run's curriculum never has to be recomputed."""
    return {
        "schema_version": "fomo26-curriculum-schedule-provenance-v1",
        "horizon_optimizer_steps": int(horizon_steps),
        "step_basis": "optimizer_global_step",
        "components": {name: specs[name].as_record() for name in sorted(specs)},
    }
