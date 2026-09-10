"""Runtime-neutral registry for reconstructing frozen FOMO26 SSL backbones."""

from __future__ import annotations

from collections.abc import Callable

_ARCH_FACTORIES: dict[str, Callable[..., object]] = {}
ARCHITECTURE_ALIASES = {"resenc_unet_b": "resenc_b"}


def register_architecture(name: str):
    """Register a lazy ``(input_channels, output_channels)`` backbone factory."""

    def _wrap(fn):
        _ARCH_FACTORIES[name] = fn
        return fn

    return _wrap


def _lazy_unet_m(input_channels: int, output_channels: int):
    from asparagus.modules.networks.unet import unet_m

    return unet_m(input_channels=input_channels, output_channels=output_channels)


def _lazy_resenc_b(input_channels: int, output_channels: int):
    from asparagus.modules.networks.resenc_unet import resenc_unet_b_ssl

    return resenc_unet_b_ssl(dimensions="3D", input_channels=input_channels, output_channels=output_channels)


_ARCH_FACTORIES.update({"unet_m": _lazy_unet_m, "resenc_b": _lazy_resenc_b})


def known_architectures() -> list[str]:
    return sorted(_ARCH_FACTORIES)


def canonical_architecture(architecture: str) -> str:
    """Return the explicit runtime identity for a scientific architecture name."""
    return ARCHITECTURE_ALIASES.get(architecture, architecture)


def build_backbone(architecture: str, input_channels: int, output_channels: int = 2):
    """Build an explicitly registered SSL backbone without a fallback."""
    architecture = canonical_architecture(architecture)
    if architecture not in _ARCH_FACTORIES:
        raise ValueError(
            f"Unknown architecture {architecture!r}; registered: {known_architectures()}. "
            "Register an explicit runtime factory before using this checkpoint."
        )
    return _ARCH_FACTORIES[architecture](input_channels=input_channels, output_channels=output_channels)
