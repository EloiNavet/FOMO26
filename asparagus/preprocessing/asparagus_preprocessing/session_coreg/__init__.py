"""Intra-session multimodal affine co-registration for FOMO26.

This subpackage adapts the FOMO60K/FOMO50K session preprocessing pipeline
(``FGA-DIKU/fomo_mri_datasets``) to the FOMO26 challenge. For each MRI session
(one ``sub-*/ses-*`` folder of a BIDS-style dataset) it:

1. reorients every scan to RAS;
2. selects the highest-spatial-resolution scan as the session reference;
3. affine co-registers every other scan to that reference with FreeSurfer
   ``mri_coreg`` (default parameters);
4. skull-strips using a SynthSeg brain mask computed on the reference and shared
   across the co-registered modalities;
5. resamples all outputs to 1 mm isotropic (FOMO26 target grid);
6. writes strict per-session QC metadata and, for a sampled subset, visual QC
   montages;
7. never writes into the raw input tree.

The heavy lifting is delegated to FreeSurfer command-line tools; the Python code
here is the orchestration, reference selection, QC, and bookkeeping layer.
"""

from asparagus_preprocessing.session_coreg.config import CoregConfig, QCThresholds

__all__ = ["CoregConfig", "QCThresholds"]
