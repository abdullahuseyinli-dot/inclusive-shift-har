from __future__ import annotations

import json
from pathlib import Path

import pytest

from inclusive_shift_har.models.third_party_specs import (
    THIRD_PARTY_BASELINE_SPECS,
    get_third_party_spec,
)


def test_audited_repository_commits_are_full_sha1_values() -> None:
    for spec in THIRD_PARTY_BASELINE_SPECS.values():
        if spec.repository_url is not None:
            assert spec.audited_commit is not None
            assert len(spec.audited_commit) == 40
            assert all(character in "0123456789abcdef" for character in spec.audited_commit)


def test_missing_software_licenses_never_allow_reuse() -> None:
    for spec in THIRD_PARTY_BASELINE_SPECS.values():
        if spec.software_license.startswith("NOASSERTION"):
            assert spec.reuse_decision == "blocked_no_software_license"


def test_registry_never_claims_a_faithful_local_implementation() -> None:
    assert all(
        not spec.faithful_local_implementation for spec in THIRD_PARTY_BASELINE_SPECS.values()
    )


def test_ccil_is_explicitly_paper_derived_and_has_no_official_repository() -> None:
    ccil = get_third_party_spec("ccil")

    assert ccil.reuse_decision == "paper_derived_loss_only"
    assert not ccil.official_code_available
    assert ccil.repository_url is None
    assert ccil.audited_commit is None


def test_liteway_current_revision_is_recorded_but_unlicensed() -> None:
    liteway = get_third_party_spec("liteway")

    assert liteway.audited_commit == "982100053db3a81b10a10d225711829473ac1f3d"
    assert liteway.reuse_decision == "blocked_no_software_license"
    assert "nine-channel" in liteway.decision_note


def test_unknown_baseline_error_lists_known_identifiers() -> None:
    with pytest.raises(KeyError, match="known baselines"):
        get_third_party_spec("not-a-baseline")


def test_machine_provenance_matches_registry_and_keeps_claims_blocked(
    repository_root: Path,
) -> None:
    path = repository_root / "docs/baselines/third_party_baseline_provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = {record["id"]: record for record in payload["baselines"]}

    assert set(records) == set(THIRD_PARTY_BASELINE_SPECS)
    for key, spec in THIRD_PARTY_BASELINE_SPECS.items():
        record = records[key]
        assert record["source_code"]["audited_commit"] == spec.audited_commit
        assert record["faithful_local_implementation"] is False
