"""Registration-quality and affine metrics (pure numpy; no FreeSurfer needed).

Used to judge co-registration beyond ``mri_coreg``'s internal cost:
  * normalized mutual information (NMI) before/after, the standard multimodal
    similarity measure;
  * edge overlap (Dice of gradient-magnitude edges), a complementary structural
    check;
  * affine/LTA decomposition into determinant, translation norm (mm) and
    rotation magnitude (degrees).
"""

import logging
import numpy as np
import re
from typing import Optional

logger = logging.getLogger(__name__)


def normalized_mutual_information(
    a: np.ndarray, b: np.ndarray, bins: int = 32, mask: Optional[np.ndarray] = None
) -> Optional[float]:
    """NMI = (H(a) + H(b)) / H(a, b), computed over shared foreground.

    Returns a value in ``[1, 2]`` (1 = independent, higher = more dependent), or
    ``None`` if there is not enough overlapping signal.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return None
    fg = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        fg &= mask.astype(bool)
    else:
        fg &= (a != 0) | (b != 0)
    if fg.sum() < 100:
        return None
    av, bv = a[fg], b[fg]
    hist, _, _ = np.histogram2d(av, bv, bins=bins)
    pab = hist / hist.sum()
    pa = pab.sum(axis=1)
    pb = pab.sum(axis=0)
    h_a = _entropy(pa)
    h_b = _entropy(pb)
    h_ab = _entropy(pab.ravel())
    if h_ab <= 0:
        return None
    return float((h_a + h_b) / h_ab)


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _gradient_magnitude(volume: np.ndarray) -> np.ndarray:
    grads = np.gradient(volume.astype(np.float64))
    return np.sqrt(np.sum([g**2 for g in grads], axis=0))


def edge_overlap(a: np.ndarray, b: np.ndarray, percentile: float = 90.0, mask: Optional[np.ndarray] = None) -> Optional[float]:
    """Dice overlap of the strongest gradient-magnitude edges of ``a`` and ``b``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return None
    ga, gb = _gradient_magnitude(a), _gradient_magnitude(b)
    region = np.ones(a.shape, dtype=bool) if mask is None else mask.astype(bool)
    if region.sum() < 100:
        return None
    ta = np.percentile(ga[region], percentile)
    tb = np.percentile(gb[region], percentile)
    ea = (ga >= ta) & region
    eb = (gb >= tb) & region
    denom = ea.sum() + eb.sum()
    if denom == 0:
        return None
    return float(2.0 * np.logical_and(ea, eb).sum() / denom)


def orthogonality_error(linear: np.ndarray) -> float:
    """``max|LᵀL - I|`` for a 3x3 linear block: 0 for a pure rotation.

    This is the direct test of "no scale and no shear". It is reported alongside
    the singular values because it collapses both failure modes into one number
    a threshold can be set on.
    """
    linear = np.asarray(linear, dtype=np.float64)
    return float(np.abs(linear.T @ linear - np.eye(3)).max())


def decompose_affine(affine: np.ndarray) -> dict:
    """Decompose a 4x4 affine into determinant, translation and rotation.

    Rotation is extracted via SVD of the 3x3 linear block (scale-invariant), and
    reported as the geodesic angle in degrees. ``orthogonality_error`` and the
    singular values are what distinguish a genuinely rigid 6-DOF transform from
    an affine that happens to have a plausible determinant: a shear can preserve
    volume exactly while destroying the geometry.
    """
    affine = np.asarray(affine, dtype=np.float64)
    linear = affine[:3, :3]
    translation = affine[:3, 3]
    det = float(np.linalg.det(linear))
    u, s, vt = np.linalg.svd(linear)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:  # reflection -> flip to a proper rotation
        u = u.copy()
        u[:, -1] *= -1
        rotation = u @ vt
    angle_rad = np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return {
        "determinant": det,
        "translation_mm": [float(t) for t in translation],
        "translation_norm_mm": float(np.linalg.norm(translation)),
        "rotation_deg": float(np.degrees(angle_rad)),
        "scales": [float(x) for x in s],
        "orthogonality_error": orthogonality_error(linear),
        "max_scale_error": float(np.abs(np.asarray(s) - 1.0).max()),
    }


# --------------------------------------------------------------------------- #
# FSL / FLIRT transforms
# --------------------------------------------------------------------------- #
def parse_flirt_mat(path: str) -> Optional[np.ndarray]:
    """Parse a FLIRT ``.mat``: four whitespace-separated rows of four numbers."""
    try:
        with open(path) as handle:
            text = handle.read()
    except OSError:
        return None
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            return None
        if len(rows) == 4:
            return np.array(rows, dtype=np.float64)
    return None


def world_to_fsl(img) -> np.ndarray:
    """World (scanner mm) -> FSL coordinates for one image.

    FSL works in a millimetre space built from voxel indices times the voxel
    sizes, with the first axis flipped when the image's affine is
    "neurological" (positive determinant).

    Because that space is already in millimetres, the voxel scaling cancels and
    a 6-DOF FLIRT matrix *is* orthogonal in FSL space. The conversion still
    matters for two reasons: the FSL axes are not the scanner axes, so a
    translation vector read straight off the ``.mat`` is expressed in the wrong
    frame; and when exactly one of the two images is neurological, FSL flips one
    and not the other, so the raw determinant's sign does not tell you whether
    the anatomical transform preserves handedness. Reporting in world space
    makes translation_mm, the rotation axis and the determinant sign mean what a
    reader assumes they mean.
    """
    import nibabel as nib

    zooms = [float(z) for z in nib.affines.voxel_sizes(img.affine)[:3]]
    scale = np.diag([*zooms, 1.0])
    if np.linalg.det(img.affine[:3, :3]) > 0:
        flip = np.eye(4)
        flip[0, 0] = -1.0
        flip[0, 3] = (int(img.shape[0]) - 1) * zooms[0]
        scale = flip @ scale
    return scale @ np.linalg.inv(img.affine)


def flirt_world_transform(mat: np.ndarray, moving, reference) -> np.ndarray:
    """Convert a FLIRT ``.mat`` into the equivalent world-space (mm) transform.

    ``T_world = world_to_fsl(ref)⁻¹ · M · world_to_fsl(moving)``.
    """
    a = world_to_fsl(moving)
    b = world_to_fsl(reference)
    return np.linalg.inv(b) @ np.asarray(mat, dtype=np.float64) @ a


def flirt_metrics(path: str, moving: Optional[str] = None, reference: Optional[str] = None) -> dict:
    """Decomposed world-space metrics for a FLIRT transform.

    Without both images the matrix cannot be de-scaled, so the raw FSL-space
    decomposition is returned and explicitly marked ``world_space=False`` rather
    than being silently interpreted as millimetres.
    """
    mat = parse_flirt_mat(path)
    if mat is None:
        return {"parsed": False, "world_space": False}
    if not moving or not reference:
        out = decompose_affine(mat)
        out.update(parsed=True, world_space=False)
        return out
    try:
        import nibabel as nib

        world = flirt_world_transform(mat, nib.load(moving), nib.load(reference))
    except Exception as exc:  # noqa: BLE001 - fall back to the frame we can still report
        out = decompose_affine(mat)
        out.update(parsed=True, world_space=False, world_error=str(exc)[:200])
        return out
    out = decompose_affine(world)
    out.update(parsed=True, world_space=True, fsl_matrix=mat.tolist())
    return out


_LTA_ROW = re.compile(r"^\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$")


def parse_lta(path: str) -> Optional[np.ndarray]:
    """Parse the 4x4 transform matrix from a FreeSurfer ``.lta`` file."""
    try:
        with open(path) as handle:
            lines = handle.readlines()
    except OSError:
        return None
    # The matrix follows a "1 4 4" (nrows ncols ...) header line.
    for i, line in enumerate(lines):
        if re.match(r"^\s*1\s+4\s+4\s*$", line):
            rows = []
            for row_line in lines[i + 1 : i + 5]:
                m = _LTA_ROW.match(row_line)
                if not m:
                    break
                rows.append([float(x) for x in m.groups()])
            if len(rows) == 4:
                return np.array(rows, dtype=np.float64)
    # Fallback: first 4 consecutive parseable 4-number rows.
    rows = []
    for line in lines:
        m = _LTA_ROW.match(line)
        if m:
            rows.append([float(x) for x in m.groups()])
            if len(rows) == 4:
                return np.array(rows, dtype=np.float64)
        elif rows:
            rows = []
    return None


def lta_metrics(path: str) -> dict:
    """Decomposed rigid/affine metrics for an LTA transform file."""
    mat = parse_lta(path)
    if mat is None:
        return {"parsed": False}
    out = decompose_affine(mat)
    out["parsed"] = True
    return out
