"""Scanner/acquisition target canonicalization and train-vocabulary encoding."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

SCANNER_TARGET_NAMES = ("manufacturer", "field_strength", "spacing_bin")
SCANNER_IGNORE_INDEX = -100

_MISSING_STRINGS = {"", "nan", "none", "null", "na", "n/a", "unknown", "unk", "not available"}
_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        if bool(value != value):
            return True
    except Exception:
        pass
    if isinstance(value, str):
        return value.strip().lower() in _MISSING_STRINGS
    return False


def _text(value) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip().lower()
    return None if text in _MISSING_STRINGS else text


def _first_present(row: Mapping, keys: Sequence[str]):
    for key in keys:
        value = row.get(key)
        if not _is_missing(value):
            return value
    return None


def canonicalize_manufacturer(value) -> str | None:
    text = _text(value)
    if text is None:
        return None
    if "siemens" in text:
        return "siemens"
    if "philips" in text:
        return "philips"
    if text == "ge" or text.startswith("ge ") or "ge medical" in text:
        return "ge"
    if "general electric" in text or "general electrics" in text:
        return "ge"
    return text


def canonicalize_field_strength(value) -> str | None:
    text = _text(value)
    if text is None:
        return None
    match = _NUMBER_RE.search(text)
    if match is None:
        return None
    try:
        tesla = float(match.group(0))
    except ValueError:
        return None
    if not math.isfinite(tesla) or tesla <= 0.0 or tesla > 15.0:
        return None
    if abs(tesla - 1.5) <= 0.25:
        return "1.5t"
    if abs(tesla - 3.0) <= 0.35:
        return "3t"
    if abs(tesla - 7.0) <= 0.5:
        return "7t"
    return None


def parse_spacing_values(value) -> tuple[float, ...] | None:
    if _is_missing(value):
        return None
    if isinstance(value, (list, tuple)):
        numbers = value
    else:
        numbers = _NUMBER_RE.findall(str(value))
    parsed = []
    for number in numbers:
        try:
            spacing = float(number)
        except (TypeError, ValueError):
            continue
        if math.isfinite(spacing) and 0.0 < spacing <= 20.0:
            parsed.append(spacing)
    return tuple(parsed) if parsed else None


def spacing_bin_label(value, bins: Sequence[float] = (1.0, 1.5, 2.0, 3.0), summary: str = "max") -> str | None:
    spacings = parse_spacing_values(value)
    if not spacings:
        return None
    if summary == "mean":
        scalar = sum(spacings) / len(spacings)
    elif summary == "max":
        scalar = max(spacings)
    else:
        raise ValueError(f"Unsupported spacing summary {summary!r}; expected 'max' or 'mean'.")
    if not math.isfinite(scalar) or scalar <= 0.0:
        return None
    ordered = [float(bound) for bound in bins]
    previous = None
    for bound in ordered:
        if scalar <= bound:
            return (
                f"<={_format_spacing_bound(bound)}"
                if previous is None
                else f"({_format_spacing_bound(previous)},{_format_spacing_bound(bound)}]"
            )
        previous = bound
    return f">{_format_spacing_bound(ordered[-1])}" if ordered else None


@dataclass(frozen=True)
class ScannerTargetConfig:
    keys: tuple[str, ...] = SCANNER_TARGET_NAMES
    ignore_index: int = SCANNER_IGNORE_INDEX
    spacing_bins: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)
    spacing_summary: str = "max"
    spacing_columns: tuple[str, ...] = (
        "original_spacing",
        "pixdim",
        "source_spacing",
        "native_spacing",
        "voxel_spacing",
    )

    def __post_init__(self):
        if tuple(self.keys) != SCANNER_TARGET_NAMES:
            raise ValueError(
                "Scanner targets must be exactly "
                f"{list(SCANNER_TARGET_NAMES)} for this acquisition-only branch, got {list(self.keys)}."
            )


class ScannerTargetEncoder:
    """Frozen scanner/acquisition vocabularies built from training metadata."""

    def __init__(
        self,
        vocab: Mapping[str, Mapping[str, int]],
        config: ScannerTargetConfig | None = None,
    ):
        self.config = config or ScannerTargetConfig()
        self.vocab = {
            target: {str(label): int(index) for label, index in sorted(mapping.items(), key=lambda item: item[1])}
            for target, mapping in vocab.items()
        }

    @classmethod
    def fit(cls, rows: Sequence[Mapping], config: ScannerTargetConfig | None = None) -> "ScannerTargetEncoder":
        config = config or ScannerTargetConfig()
        values = {target: set() for target in config.keys}
        for row in rows:
            canonical = canonicalize_scanner_targets(row, config)
            for target, label in canonical.items():
                if target in values and label is not None:
                    values[target].add(label)
        vocab = {}
        for target in config.keys:
            labels = _ordered_labels(target, values[target], config)
            vocab[target] = {label: index for index, label in enumerate(labels)}
        return cls(vocab, config)

    def encode(self, row: Mapping) -> dict[str, int]:
        canonical = canonicalize_scanner_targets(row, self.config)
        encoded = {}
        for target in self.config.keys:
            label = canonical.get(target)
            encoded[target] = self.vocab.get(target, {}).get(label, int(self.config.ignore_index))
        return encoded

    def class_counts(self, minimum: int = 1) -> dict[str, int]:
        return {target: max(int(minimum), len(self.vocab.get(target, {}))) for target in self.config.keys}

    def to_dict(self) -> dict:
        return {
            "keys": list(self.config.keys),
            "ignore_index": int(self.config.ignore_index),
            "spacing_bins": list(self.config.spacing_bins),
            "spacing_summary": self.config.spacing_summary,
            "spacing_columns": list(self.config.spacing_columns),
            "vocab": {target: dict(mapping) for target, mapping in self.vocab.items()},
        }


def canonicalize_scanner_targets(row: Mapping, config: ScannerTargetConfig | None = None) -> dict[str, str | None]:
    config = config or ScannerTargetConfig()
    manufacturer = canonicalize_manufacturer(_first_present(row, ("manufacturer", "manufacturers")))
    field_strength = canonicalize_field_strength(
        _first_present(row, ("field_strength", "magneticfieldstrength", "magnetic_field_strength"))
    )
    spacing_source = _first_present(row, config.spacing_columns)
    spacing_bin = spacing_bin_label(spacing_source, config.spacing_bins, config.spacing_summary)
    return {
        "manufacturer": manufacturer,
        "field_strength": field_strength,
        "spacing_bin": spacing_bin,
    }


def _ordered_labels(target: str, labels: set[str], config: ScannerTargetConfig) -> list[str]:
    if target == "field_strength":
        preferred = ["1.5t", "3t", "7t"]
        return [label for label in preferred if label in labels] + sorted(labels - set(preferred))
    if target == "spacing_bin":
        preferred = []
        previous = None
        for bound in config.spacing_bins:
            preferred.append(
                f"<={_format_spacing_bound(float(bound))}"
                if previous is None
                else f"({_format_spacing_bound(float(previous))},{_format_spacing_bound(float(bound))}]"
            )
            previous = bound
        if config.spacing_bins:
            preferred.append(f">{_format_spacing_bound(float(config.spacing_bins[-1]))}")
        return [label for label in preferred if label in labels] + sorted(labels - set(preferred))
    return sorted(labels)


def _format_spacing_bound(value: float) -> str:
    return f"{float(value):.1f}"
