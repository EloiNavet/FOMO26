from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PreprocessingConfig:
    """
    Configuration for image preprocessing operations applied before training.

    Fields are grouped by concern: what the output space looks like (spacing /
    size / orientation), how voxel intensities are handled (normalization), and
    what quality filters are applied (NaN removal, minimum slice count).

    Attributes:
        normalization_operation: One normalization scheme per modality, applied
            in channel order. E.g. ``["volume_wise_znorm", "no_norm"]`` normalizes
            the first channel and leaves the second untouched. Accepted values:
            ``"no_norm"``, ``"volume_wise_znorm"``, ``"minmax"``, ``"standardize"``,
            ``"clipping"``, ``"ct"``, ``"255to1"``.
        target_spacing: Voxel spacing (mm) to resample all images to, e.g.
            ``[1.0, 1.0, 1.0]`` for isotropic 1 mm. Set to ``None`` to preserve
            native spacing. Mutually exclusive with ``target_size``.
        target_size: Fixed spatial dimensions ``[H, W, D]`` to resample all
            images to. Set to ``None`` to let spacing drive the output size.
            Mutually exclusive with ``target_spacing`` and ``downsample_factor``.
            This is a legacy voxel-index operation and must not be used on
            anisotropic medical images unless explicitly acknowledged.
        output_size: Fixed output envelope applied after physical-space
            resampling, e.g. ``[96, 96, 96]`` after 2.5 mm isotropic
            resampling. Smaller images are centered and padded. Images that
            would require cropping are rejected when ``overflow_policy`` is
            ``"reject"``.
        downsample_factor: Uniform voxel-grid downsampling factor. For example,
            ``2.0`` halves each spatial dimension and doubles the effective
            voxel spacing. This preserves image proportions and physical field
            of view, while producing variable output shapes across subjects.
            Mutually exclusive with ``target_spacing`` and ``target_size``.
        target_orientation: Anatomical orientation code to reorient images to
            before any other operation, e.g. ``"RAS"``. Set to ``None`` to skip
            reorientation.
        keep_aspect_ratio_when_using_target_size: When ``target_size`` is set,
            scale uniformly so that one dimension reaches the target and zero-pad
            the rest rather than stretching. Introduces black bars but avoids
            voxel-index distortion only; it is not a physical-space guarantee.
        overflow_policy: Policy for a physically resampled image that exceeds
            ``output_size``. ``"reject"`` records and excludes it instead of
            silently truncating tissue.
        allow_legacy_anisotropic_target_size: Explicit opt-in for using the
            legacy ``target_size`` path on input whose voxel spacing is
            anisotropic.
        crop_to_nonzero: Remove background slices before resampling by cropping
            to the smallest bounding box that contains all non-background voxels.
            Reduces memory and compute.
        background_pixel_value: Voxel value treated as background when
            ``crop_to_nonzero`` is ``True``.
        intensities: Dataset-wide intensity statistics used by normalization
            schemes that require global context (e.g. ``"minmax"``,
            ``"standardize"``, ``"ct"``). ``None`` for volume-wise schemes.
        min_slices: Discard any image whose smallest spatial dimension is below
            this threshold. Useful to filter out corrupt or near-2D volumes.
        remove_nans: Raise an error and skip images that contain NaN values
            rather than propagating them silently into training.
        image_properties: Internal container populated during preprocessing with
            per-image metadata (spacing, orientation, crop box, etc.). Should
            not be set manually.
    """

    # Output space
    normalization_operation: List[str]
    target_spacing: Optional[List[float]]
    target_size: Optional[List[int]] = None
    output_size: Optional[List[int]] = None
    downsample_factor: Optional[float] = None
    target_orientation: Optional[str] = "RAS"
    keep_aspect_ratio_when_using_target_size: bool = False
    overflow_policy: str = "reject"
    allow_legacy_anisotropic_target_size: bool = False

    # Intensity handling
    intensities: Optional[List] = None

    # Cropping
    crop_to_nonzero: bool = True
    background_pixel_value: int = 0

    # Quality filters
    min_slices: int = 0
    remove_nans: bool = True

    # Internal
    image_properties: Optional[dict] = field(default_factory=dict)


@dataclass
class SavingConfig:
    """
    Configuration controlling how processed images and metadata are persisted.

    Attributes:
        save_as_tensor: Save output as a ``.pt`` PyTorch tensor. If ``False``,
            output is saved as ``.nii.gz``. Tensor format is required for
            multi-modal data (stacked along the channel dimension).
        tensor_dtype: NumPy / PyTorch dtype string used when ``save_as_tensor``
            is ``True``, e.g. ``"float32"``.
        save_file_metadata: Save a per-image ``.pkl`` sidecar containing
            preprocessing provenance (original spacing, crop box, orientation,
            etc.). **Required for segmentation tasks**, where foreground
            locations stored in the sidecar are used for class-balanced sampling.
        save_dset_metadata: Save dataset-level aggregate metadata (participant
            demographics, MRI acquisition parameters) to ``mri_info.tsv``.
            Only relevant for pretraining datasets that expose a ``bidsify``
            workflow.
        bidsify: Reorganise the output directory into a BIDS-compliant folder
            structure and rename files accordingly. Requires dataset-specific
            ``patterns_bidsify`` and a demographics file.
    """

    # Output format
    save_as_tensor: bool
    tensor_dtype: str

    # Metadata sidecars
    save_file_metadata: bool
    save_dset_metadata: bool

    # BIDS
    bidsify: bool


@dataclass
class DatasetConfig:
    """
    Configuration describing the structure and content of a single dataset.

    Required fields encode facts that are always dataset-specific and have no
    safe default. Optional fields default to empty lists and only need to be
    overridden when the dataset requires filtering or special handling.

    Attributes:
        task_name: Unique identifier for this dataset following the convention
            ``<TYPE><NNN>_<Description>``, e.g. ``"PT038_Oslo7T"``.
            Valid type prefixes: ``PT`` (pretraining), ``SEG`` (segmentation),
            ``CLS`` (classification), ``REG`` (regression).
        n_classes: Number of target classes. Use ``1`` for pretraining and
            regression. For segmentation this includes the background class.
        n_modalities: Number of image channels expected per sample. Determines
            the ``C`` dimension of the output tensor ``[C, H, W, D]``.
        in_extensions: File extensions to search for in the source directory,
            e.g. ``[".nii.gz"]`` or ``[".nii", ".nii.gz"]``.
        split: Name of the splitting function to use (defined in
            ``asparagus_preprocessing/utils/splitting.py``), e.g.
            ``"split_40_10_50"``. ``None`` skips automatic split generation.
        df_columns: Columns to retain from the demographics file when
            ``save_dset_metadata`` or ``bidsify`` is enabled, e.g.
            ``["participant_id", "session_id", "age", "sex", "group"]``.
        patterns_exclusion: Substrings matched against the full file path.
            Any file whose path contains one of these strings is excluded
            entirely from processing, e.g. ``["func", "mask", "derivatives"]``.
        patterns_DWI: Substrings that identify diffusion-weighted images.
            Matched files are routed to the DWI-specific processing pipeline
            (4D → 3D reduction, bval/bvec handling).
        patterns_PET: Substrings that identify PET images. Matched files are
            routed to the PET-specific processing pipeline (4D → 3D mean).
        patterns_perfusion: Substrings that identify perfusion / ASL images.
            Matched files are routed to the perfusion-specific pipeline.
        patterns_m0: Substrings that identify M0 reference scans within the
            perfusion file set. M0 scans receive a different reduction strategy
            (mean across time) than labelled perfusion images.
        patterns_bidsify: Regular expressions used to extract subject and
            session identifiers from file paths when ``bidsify`` is enabled,
            e.g. ``[r"sub-([A-Za-z0-9]+)"]``.
    """

    # Identity
    task_name: str
    n_classes: int
    n_modalities: int
    in_extensions: List[str]

    # Splitting
    split: Optional[str] = None

    # File filtering
    patterns_exclusion: List[str] = field(default_factory=list)
    patterns_DWI: List[str] = field(default_factory=list)
    patterns_PET: List[str] = field(default_factory=list)
    patterns_perfusion: List[str] = field(default_factory=list)
    patterns_m0: List[str] = field(default_factory=list)

    # Metadata / BIDS
    df_columns: List[str] = field(default_factory=list)
    patterns_bidsify: List[str] = field(default_factory=list)
