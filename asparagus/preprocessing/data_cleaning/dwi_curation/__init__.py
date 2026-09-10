"""FOMO300K diffusion (DWI/ADC) curation layer.

A downstream, auditable curation layer that runs on the already-produced
``FOMO300K_cleaned`` tree (plus ``mapping.tsv`` / ``mri_info.tsv`` provenance) and
resolves the clinical diffusion modalities FOMO26 needs -- ``DWI_B1000`` and
``ADC`` -- as *distinct* channels, instead of the single generic ``dwi`` class the
current cleaning pipeline emits.

The package never modifies ``FOMO300K_cleaned``. Stage 1 (:mod:`.audit`) produces a
per-scan report; Stage 3 (:mod:`.curate`) plans (dry-run) or writes a new
``FOMO300K_curated_v2`` tree.

See ``docs/data-pipeline/dwi_curation.md`` for the design note.
"""

from . import classify, config, derive, ids, selection  # noqa: F401

__all__ = ["config", "ids", "classify", "selection", "derive"]
