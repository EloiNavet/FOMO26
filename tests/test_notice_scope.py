"""What NOTICE must cover, and why that is narrower than the lockfile.

The lock has 169 entries -- one first-party (this project) and 168 third-party resolutions --
and this repository redistributes none of them. It publishes source, and the
reader resolves dependencies from public indexes under their own licences. Listing every lock entry
as though it were redistributed would overstate what is happening; listing none would leave the
reader without the inventory the courtesy scope exists to provide.

These assertions pin that scope so it cannot quietly drift into either error.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTICE = REPO / "NOTICE"
PROJECT_NAME = "asparagus"
INVENTORY_HEADER = "THIRD-PARTY LOCK INVENTORY"
MANIFEST = REPO / "finetuning" / "container" / "validator_manifest.json"


def test_notice_exists_and_declares_its_scope():
    text = NOTICE.read_text()
    assert "SCOPE" in text
    assert "distributes SOURCE CODE" in text
    for scope in ("A.", "B.", "C."):
        assert scope in text, f"scope section {scope} missing"


def test_this_repository_redistributes_no_third_party_source():
    """Scope A is empty by construction, and the tree must agree with the claim."""
    text = NOTICE.read_text()
    section = text[text.index("A. THIRD-PARTY SOURCE") : text.index("B. DECLARED DEPENDENCIES")]
    assert "None." in section
    assert not (REPO / "third_party").exists(), "scope A says none; the tree says otherwise"


def test_the_notice_explains_why_the_validator_is_not_vendored():
    """Removal is the resolution, so the reasoning has to survive the file that recorded it."""
    text = " ".join(NOTICE.read_text().split())
    assert "declares no licence" in text
    assert "validator_manifest.json" in text
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["upstream_license"]["declared"] is False
    assert manifest["upstream_commit"] in text, "the pinned commit must be stated, not implied"


def test_the_manifest_is_a_record_not_a_copy():
    """Paths and digests are ours; upstream source is not here."""
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["member_count"] == 84
    for entry in manifest["entries"]:
        assert set(entry) <= {"path", "snapshot_sha256", "snapshot_size", "vendored", "lfs"}


def test_the_owner_decision_is_not_dressed_up_as_legal_review():
    """The owner may approve a source release; that is not, and must not read as, legal advice."""
    text = " ".join(NOTICE.read_text().split())  # the prose wraps; the claim must not depend on where
    assert "SOURCE-RELEASE APPROVED BY THE PROJECT OWNER" in text
    assert "not legal-counsel review" in text
    assert "not legal certainty" in text
    assert "constitutes a legal opinion" in text


def test_no_binary_distribution_or_sif_claim_is_made():
    text = " ".join(NOTICE.read_text().split())
    assert "ships no dependency wheels and no SIF" in text
    assert "no binary-identical SIF was rebuilt or validated, and none is claimed" in text


def _locked_third_party() -> Counter:
    """Every [[package]] block in the lock except this project's own, as a (name, version) multiset.

    A multiset, not a set: uv resolves a name conditionally when different Python ranges need
    different versions, and the lock then carries one block per resolution. Collapsing those to a
    name loses exactly the information the inventory exists to carry.
    """
    lock = (REPO / "uv.lock").read_text()
    blocks = re.findall(r'^\[\[package\]\]\nname = "([^"]+)"\nversion = "([^"]+)"', lock, re.MULTILINE)
    assert blocks, "no packages parsed from the lock"
    return Counter((name, version) for name, version in blocks if name != PROJECT_NAME)


def _inventory() -> Counter:
    """The inventory section of NOTICE, as the same kind of multiset.

    Lines may carry trailing annotations (`[runtime]`, a marker); only the first two fields are
    the identity.
    """
    text = NOTICE.read_text()
    start = text.index(INVENTORY_HEADER)
    body = text[start + len(INVENTORY_HEADER) :]
    entries = Counter()
    for line in body.splitlines():
        if not line.startswith("  ") or not line.strip():
            continue
        fields = line.split()
        if len(fields) < 2 or not re.match(r"^[0-9]", fields[1]):
            continue  # prose in the section preamble, not an inventory row
        entries[(fields[0], fields[1])] += 1
    return entries


def test_the_inventory_is_exactly_the_third_party_lock_resolutions():
    """Name-level agreement is not enough: a dropped conditional resolution passes that and lies.

    The inventory claims to be complete. This compares it to the lock as a multiset of
    (name, version), in both directions, so a resolution that is silently collapsed into another
    fails here rather than being discovered by a reader.
    """
    locked, listed = _locked_third_party(), _inventory()
    missing = sorted((locked - listed).elements())
    extra = sorted((listed - locked).elements())
    assert not missing, f"locked resolutions absent from the NOTICE inventory: {missing}"
    assert not extra, f"NOTICE inventory rows that the lock does not contain: {extra}"


def test_the_first_party_entry_is_excluded_and_said_to_be():
    """`asparagus` is what this repository publishes, not something it depends on."""
    lock = (REPO / "uv.lock").read_text()
    assert f'name = "{PROJECT_NAME}"' in lock, "the project's own lock entry vanished"
    assert (PROJECT_NAME, "0.4.6") not in _inventory(), "the first-party entry must not be listed as a dependency"
    assert "first-party" in NOTICE.read_text(), "the exclusion must be stated, not silently applied"


def test_a_conditionally_resolved_name_carries_its_marker():
    """Two rows with one name and no markers would be indistinguishable from a duplication bug."""
    listed = _inventory()
    text = NOTICE.read_text()
    per_name = Counter(name for name, _ in listed.elements())
    duplicated = {name for name, count in per_name.items() if count > 1}
    assert duplicated, "no name resolves more than once; this test would otherwise pass vacuously"
    for name in sorted(duplicated):
        rows = [line for line in text.splitlines() if line.startswith(f"  {name} ")]
        for row in rows:
            assert "marker:" in row, f"{name} resolves more than once; each row must name its marker: {row!r}"


def test_every_runtime_dependency_appears_in_the_inventory():
    """Scope B is a courtesy list, but it must at least be complete for what actually installs."""
    lock = (REPO / "uv.lock").read_text()
    block = re.search(r'\[\[package\]\]\nname = "asparagus"\n(?:.*\n)*?dependencies = \[\n((?:.*\n)*?)\]\n', lock)
    assert block, "could not locate the project's own dependency list in the lock"
    runtime = set(re.findall(r'name = "([^"]+)"', block.group(1)))
    assert runtime, "no runtime dependencies parsed"
    text = NOTICE.read_text()
    missing = sorted(name for name in runtime if f"  {name} " not in text)
    assert not missing, f"runtime dependencies absent from the NOTICE inventory: {missing}"


def test_natten_is_absent_from_the_public_notice():
    """No retained module imports it; a NOTICE naming it would describe the wrong environment."""
    assert "natten" not in NOTICE.read_text().lower()
