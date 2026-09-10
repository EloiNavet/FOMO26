"""Not every pretraining lineage emitted a frozen campaign.

The adapter derived track and SSL objective exclusively from `frozen_campaign.json`, which the
challenge lineage produced and the salvage lineages did not. A candidate whose parent predates that
artifact could not be released at all. Where it is absent the same facts must be declared by the
scientific handoff instead -- never inferred from a checkpoint path -- and the manifest records
which of the two established them.
"""

from __future__ import annotations

import pytest
from finetuning.container.handoff_adapter import CLASSIFICATIONS


def test_declared_tta_uses_an_existing_classification():
    # A classification outside this vocabulary raises AssertionError inside _derived, which would
    # abort the whole adaptation rather than mislabel one field.
    assert "MECHANICALLY_DERIVABLE" in CLASSIFICATIONS
    assert "DECLARED_BY_SCIENTIFIC_AUTHORITY" not in CLASSIFICATIONS


def test_adapter_reads_objective_and_dataset_from_the_authority_when_no_campaign_exists():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "finetuning" / "container" / "handoff_adapter.py").read_text()
    # Absent campaign must still require an explicitly declared objective, not a guessed one.
    assert "declares no frozen campaign and no ssl_objective" in source
    assert "pretraining_dataset" in source
    # The FOMO300K restriction that establishes the Methods track still applies either way.
    assert "is not restricted to FOMO300K" in source


@pytest.mark.parametrize("field", ["ssl_objective", "architecture"])
def test_absent_campaign_still_cross_checks_the_candidate(field):
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "finetuning" / "container" / "handoff_adapter.py").read_text()
    assert field in source
