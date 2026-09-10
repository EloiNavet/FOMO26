"""Normalize physical voxel spacing at runtime.

The preprocessed FOMO26 segmentation corpora are stored at each subject's native acquisition
geometry: ``prepare_fomo26_asparagus.py`` records ``original_spacing == new_spacing`` and performs
no resampling. For a cohort acquired at one resolution that is harmless. For Task 2 it is not --
its in-plane spacing spans 0.43-0.90 mm across 23 subjects, so a fixed-size voxel patch covers
between a half and a whole of the physical field of view depending on which subject it came from,
and the network is asked to recognise the same physical lesion at two different scales.

The preprocessing rail already solves this with :func:`resample_case_to_spacing`, but it needs the
raw NIfTI inputs and an affine. Those are not always available next to a preprocessed corpus, so
this transform does the equivalent resampling at load time from the geometry the ``.pkl`` sidecar
already carries.

It is disabled by default (``target_spacing=None``), in which case ``__call__`` returns the input
dictionary untouched and every existing experiment composes exactly as before.
"""

import numpy as np
import torch
import torch.nn.functional as F
from gardening_tools.modules.transforms.BaseTransform import BaseTransform

#: Per-axis sentinel meaning "keep this axis at whatever physical spacing it already has".
#: Task 2 is acquired with 5.2-7.5 mm slices; forcing those to an in-plane target would invent
#: through-plane detail that was never measured.
NATIVE_SPACING = "native"


class RuntimeSpacingError(ValueError):
    """A spacing request or a geometry record that must not be resampled silently."""


def resolve_target_spacing(target_spacing, source_spacing) -> list[float]:
    """Resolve a per-axis spacing request against the spacing an image actually has.

    Each requested axis is either a positive finite number or :data:`NATIVE_SPACING`. Anything
    else -- zero, negative, NaN, infinity, a wrong axis count -- is refused rather than coerced,
    because every one of those silently produces a differently scaled image.
    """
    source = np.asarray(source_spacing, dtype=float)
    if source.shape != (len(target_spacing),):
        raise RuntimeSpacingError(
            f"target_spacing has {len(target_spacing)} axes but the image geometry declares "
            f"{source.shape[0] if source.ndim else 0}; they must describe the same image."
        )
    if not np.all(np.isfinite(source)) or np.any(source <= 0):
        raise RuntimeSpacingError(f"source spacing must be positive and finite, got {source_spacing!r}.")

    resolved: list[float] = []
    for axis, request in enumerate(target_spacing):
        if isinstance(request, str):
            if request != NATIVE_SPACING:
                raise RuntimeSpacingError(
                    f"axis {axis}: unknown spacing sentinel {request!r}; expected a positive number or {NATIVE_SPACING!r}."
                )
            resolved.append(float(source[axis]))
            continue
        if isinstance(request, bool) or request is None:
            raise RuntimeSpacingError(f"axis {axis}: {request!r} is not a spacing.")
        value = float(request)
        if not np.isfinite(value) or value <= 0:
            raise RuntimeSpacingError(f"axis {axis}: spacing must be positive and finite, got {request!r}.")
        resolved.append(value)
    return resolved


def target_shape_for_spacing(source_shape, source_spacing, target_spacing) -> list[int]:
    """Return the voxel grid that preserves physical coverage at ``target_spacing``.

    Same rule as :func:`resample_case_to_spacing`: ``round(extent / spacing)``, floored at one
    voxel so a very thin axis cannot vanish.
    """
    source_shape = np.asarray(source_shape, dtype=float)
    ratio = np.asarray(source_spacing, dtype=float) / np.asarray(target_spacing, dtype=float)
    return [int(value) for value in np.maximum(np.round(source_shape * ratio), 1).astype(int)]


def _remap_foreground_locations(foreground_locations, source_shape, target_shape):
    """Rescale cached foreground voxel coordinates onto the resampled grid.

    These coordinates drive :class:`Torch_Crop`'s foreground oversampling. Left unscaled after a
    0.45 mm -> 0.9 mm resample they address voxels that no longer exist, and ``torch_crop`` derives
    a ``np.random.randint(low, high)`` window from them -- which raises once ``low`` exceeds
    ``high``. So this is required for correctness, not merely for tidiness.
    """
    scale = np.asarray(target_shape, dtype=float) / np.asarray(source_shape, dtype=float)
    bound = np.asarray(target_shape, dtype=int) - 1

    def remap(locations):
        if len(locations) == 0:
            return locations
        moved = np.rint(np.asarray(locations, dtype=float) * scale).astype(int)
        return np.clip(moved, 0, bound).tolist()

    if isinstance(foreground_locations, dict):
        return {key: remap(value) for key, value in foreground_locations.items()}
    return remap(foreground_locations)


class Torch_ResampleToSpacing(BaseTransform):
    """Resample a loaded case onto a target physical voxel spacing.

    Must run *after* intensity normalization and *before* padding/cropping, which is how the
    preprocessing rail orders the same two operations (:func:`resample_case_to_spacing` normalizes,
    then resamples) and what leaves the pad/crop provenance describing the tensor the model
    actually sees.
    """

    def __init__(
        self,
        target_spacing=None,
        data_key: str = "image",
        label_key: str = "label",
        properties_key: str = "properties",
        source_spacing_key: str = "source_spacing",
        image_mode: str = "trilinear",
    ):
        self.target_spacing = list(target_spacing) if target_spacing is not None else None
        self.data_key = data_key
        self.label_key = label_key
        self.properties_key = properties_key
        self.source_spacing_key = source_spacing_key
        self.image_mode = image_mode

    def _source_spacing(self, data_dict):
        """Find the spacing that describes the tensor currently in memory.

        ``new_spacing`` is that spacing by construction: preprocessing writes it as the output
        geometry, and it equals ``original_spacing`` exactly when no resampling was applied.
        """
        properties = data_dict.get(self.properties_key)
        if isinstance(properties, dict) and properties.get("new_spacing") is not None:
            return properties["new_spacing"]
        spacing = data_dict.get(self.source_spacing_key)
        if spacing is not None:
            return spacing
        raise RuntimeSpacingError(
            "Runtime spacing normalization was requested but the sample carries no geometry: "
            f"neither {self.properties_key!r}['new_spacing'] nor {self.source_spacing_key!r} is "
            "present. Refusing to guess the spacing of a medical image."
        )

    @staticmethod
    def _record_provenance(properties, source_shape, source_spacing, resolved_spacing):
        """Hand the inverse to the existing ``size_before_resample`` contract.

        ``reverse_preprocessing`` already interpolates a prediction back to
        ``size_before_resample`` between unpadding and uncropping, which is exactly where an
        undo of this transform belongs. That field is therefore reused rather than duplicated --
        but only after proving it holds nothing else. A corpus that was genuinely resampled during
        preprocessing needs a composable representation of *two* resamples, which this does not
        implement, so such a corpus is refused rather than silently flattened.
        """
        spatial = list(source_shape)
        recorded = properties.get("size_before_resample")
        if recorded is not None and [int(value) for value in recorded] != [int(value) for value in spatial]:
            raise RuntimeSpacingError(
                f"size_before_resample={list(recorded)} already describes an earlier resample from "
                f"the current shape {spatial}. Composing two resamples into one inverse field would "
                "silently corrupt prediction restoration; this transform only supports a corpus "
                "stored at its pre-resample geometry."
            )
        properties["size_before_resample"] = [int(value) for value in spatial]
        properties["new_spacing"] = [float(value) for value in resolved_spacing]
        # Deliberately namespaced so it cannot be mistaken for source-space geometry. The metric
        # path resolves spacing via ("spacing", "itk_spacing", "original_spacing", ...), so
        # original_spacing stays untouched and DSC/NSD keep measuring in source space.
        properties["runtime_resample"] = {
            "source_shape": [int(value) for value in spatial],
            "source_spacing": [float(value) for value in source_spacing],
            "target_spacing": [float(value) for value in resolved_spacing],
        }

    def __call__(self, data_dict: dict) -> dict:
        if self.target_spacing is None:
            return data_dict

        image = data_dict[self.data_key]
        label = data_dict.get(self.label_key)
        source_shape = list(image.shape[1:])
        source_spacing = self._source_spacing(data_dict)
        resolved = resolve_target_spacing(self.target_spacing, source_spacing)
        target_shape = target_shape_for_spacing(source_shape, source_spacing, resolved)

        properties = data_dict.get(self.properties_key)
        if isinstance(properties, dict):
            self._record_provenance(properties, source_shape, source_spacing, resolved)

        if target_shape == source_shape:
            return data_dict

        # One grid for the whole case: every modality is a channel of this tensor, so they cannot
        # drift apart, and the label rides the identical target shape.
        data_dict[self.data_key] = F.interpolate(
            image.unsqueeze(0).to(torch.float32),
            size=target_shape,
            mode=self.image_mode,
            align_corners=False,
        ).squeeze(0)
        if label is not None:
            resampled_label = F.interpolate(
                label.unsqueeze(0).to(torch.float32),
                size=target_shape,
                mode="nearest",
            ).squeeze(0)
            data_dict[self.label_key] = resampled_label.to(label.dtype)

        if data_dict.get("foreground_locations") is not None:
            data_dict["foreground_locations"] = _remap_foreground_locations(
                data_dict["foreground_locations"], source_shape, target_shape
            )
        return data_dict
