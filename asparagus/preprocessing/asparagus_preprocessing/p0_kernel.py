"""The P0 geometry kernel: RAS -> 1 mm isotropic, full FOV, no crop, no mask.

This is the **single definition** of the geometry contract that produced
``FOMO300K_curated_iso1p0_v1`` (33,336 samples, canonical manifest SHA256
``78ce45fd…``). It was extracted verbatim from
the P0 conversion entrypoint so that the P1
co-registered derivative can reuse it rather than re-implement it: a second
implementation would be a second contract, and the P0-vs-P1 ablation would then
confound registration with a preprocessing difference nobody declared.

Two callers, one kernel:

* **P0** — ``upstream_pilot.convert_one`` resamples every canonical sample.
* **P1** — ``session_coreg`` builds each session's shared 1 mm lattice from its
  reference scan, and materialises single-scan / passthrough sessions.

Because both go through :func:`p0_resample`, a P1 reference scan and a P1
single-scan passthrough are **bit-for-bit identical** to their P0 counterparts
(``tests/test_session_coreg_fsl.py`` asserts exactly this). P1 therefore differs
from P0 only on the non-reference scans of genuinely multi-scan sessions.

Nothing here masks, crops, bias-corrects or normalises. Interpolation is cubic
(``order=3``) and out-of-field voxels are filled with the source minimum, not
zero, so padding cannot invent signal brighter than the background.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Union

#: Isotropic target spacing of the canonical corpus, in millimetres.
TARGET_SPACING_MM = 1.0
#: Interpolation order used by the contract (cubic spline).
RESAMPLE_ORDER = 3
#: Orientation every output must carry.
REQUIRED_ORIENTATION = "RAS"

ImageOrPath = Union[str, "object"]


def _load(src: ImageOrPath):
    """Return a nibabel image for a path or pass an already-loaded image through."""
    import nibabel as nib

    return nib.load(src) if isinstance(src, (str, bytes)) else src


def ras_view(img, data: np.ndarray = None):
    """Reorient to closest-canonical (RAS) without interpolating.

    Mirrors P0: the voxel array is materialised as float32 *first* (which applies
    any ``scl_slope``/``scl_inter``), and the reorientation is then a pure axis
    permutation and flip — no resampling, no value change.

    ``data`` lets a caller that has already decoded the volume hand it in, so a
    corpus-wide run does not decode every file twice. It must be exactly
    ``np.asanyarray(img.dataobj, dtype=np.float32)``.
    """
    import nibabel as nib

    src = np.asanyarray(img.dataobj, dtype=np.float32) if data is None else data
    return nib.as_closest_canonical(nib.Nifti1Image(src, img.affine, img.header))


def iso_grid(ras_img, spacing: float = TARGET_SPACING_MM) -> Tuple[tuple, np.ndarray]:
    """Target ``(shape, affine)`` for an isotropic resample that keeps the full FOV.

    The shape is ceil'ed so no physical extent is ever cut, and the affine's
    linear block is rescaled column-wise, which preserves the world origin and
    the direction cosines exactly.
    """
    zooms = tuple(float(z) for z in ras_img.header.get_zooms()[:3])
    out_shape = tuple(int(np.ceil(s * z / spacing)) for s, z in zip(ras_img.shape[:3], zooms))
    affine = ras_img.affine.copy()
    affine[:3, :3] = affine[:3, :3] @ np.diag([spacing / z for z in zooms])
    return out_shape, affine


def resample_to_grid(ras_img, shape: tuple, affine: np.ndarray, order: int = RESAMPLE_ORDER, cval: float = None):
    """Resample ``ras_img`` onto an explicit ``(shape, affine)`` lattice.

    ``cval`` defaults to the source minimum, matching P0: padding with 0 would
    be brighter than background on images whose air is negative.
    """
    from nibabel.processing import resample_from_to

    fill = float(np.min(ras_img.dataobj)) if cval is None else float(cval)
    return resample_from_to(ras_img, (shape, affine), order=order, mode="constant", cval=fill)


def p0_resample(src: ImageOrPath, spacing: float = TARGET_SPACING_MM, data: np.ndarray = None):
    """The exact P0 kernel: RAS -> isotropic ``spacing``, full FOV, cubic, no crop.

    Returns a nibabel image. This is the function both P0 conversion and P1
    co-registration must call; see the module docstring for why. ``data`` is the
    optional pre-decoded float32 array (see :func:`ras_view`).
    """
    ras = ras_view(_load(src), data=data)
    shape, affine = iso_grid(ras, spacing)
    return resample_to_grid(ras, shape, affine)


def write_p0_resampled(src: ImageOrPath, dst: str, spacing: float = TARGET_SPACING_MM) -> str:
    """Run :func:`p0_resample` and save the result to ``dst`` as float32 NIfTI.

    The output dtype is pinned to float32 with scaling disabled so the stored
    file round-trips exactly; a scaled integer container would quantise the
    session's shared lattice differently per modality.
    """
    import nibabel as nib
    import os

    out = p0_resample(src, spacing)
    data = np.asanyarray(out.dataobj, dtype=np.float32)
    image = nib.Nifti1Image(data, out.affine)
    image.set_data_dtype(np.float32)
    image.header.set_slope_inter(1.0, 0.0)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    nib.save(image, dst)
    return dst


def grid_signature(img) -> dict:
    """Shape / spacing / orientation / affine of an image, for QC and provenance."""
    import nibabel as nib

    return {
        "shape": [int(s) for s in img.shape[:3]],
        "voxel_sizes": [float(v) for v in nib.affines.voxel_sizes(img.affine)[:3]],
        "orientation": "".join(nib.aff2axcodes(img.affine)),
        "affine": img.affine.tolist(),
    }
