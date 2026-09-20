from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from inclusive_shift_har.manifests.canonical import canonical_json_sha256


def _json(root: Path, relative: str) -> dict[str, Any]:
    value = json.loads((root / relative).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_current_publication_index_matches_canonical_results(repository_root: Path) -> None:
    index = _json(repository_root, "results/research/current_publication_evidence_v1.json")
    record_hash = index.pop("record_sha256")
    assert record_hash == canonical_json_sha256(index)

    target = _json(repository_root, "results/confirmatory/zero_shot_v1/publication_report_v1.json")
    target_rows = {row["model_id"]: row for row in target["model_rows"]}
    locked = index["locked_target_opening"]
    assert locked["compact_dann_mean_participant_macro_f1"] == pytest.approx(
        target_rows["compact-dann"]["mean_participant_macro_f1"]
    )
    assert locked["compact_coral_mean_participant_macro_f1"] == pytest.approx(
        target_rows["compact-coral"]["mean_participant_macro_f1"]
    )
    assert locked["more_har_full_mean_participant_macro_f1"] == pytest.approx(
        target_rows["more-har-full"]["mean_participant_macro_f1"]
    )

    ctgr = _json(repository_root, "results/development/max_rnd_secondary_v1_summary.json")
    hera = _json(repository_root, "results/development/hera_ctgr_retrospective_v1_summary.json")
    hera_v2 = _json(
        repository_root, "results/development/hera_ctgr_v2_retrospective_v1_summary.json"
    )
    source = index["matched_source_development"]
    assert source["rmrp_6ch_mean_participant_macro_f1"] == pytest.approx(
        ctgr["ctgr"]["five_seed_summary"]["flat_rmrp"]["mean_participant_macro_f1"]
    )
    assert source["ctgr_9ch_mean_participant_macro_f1"] == pytest.approx(
        ctgr["ctgr"]["five_seed_summary"]["ctgr"]["mean_participant_macro_f1"]
    )
    assert source["strict_hera_v1_9ch_mean_participant_macro_f1"] == pytest.approx(
        hera["five_seed_mean_results"]["hera_ctgr_strict"]["mean_participant_macro_f1"]
    )
    assert source["hera_v2_full_9ch_mean_participant_macro_f1"] == pytest.approx(
        hera_v2["five_seed_mean_results"]["hera_ctgr_v2_full"]["mean_participant_macro_f1"]
    )


def test_current_evidence_role_ledger_keeps_consumed_cohorts_closed(
    repository_root: Path,
) -> None:
    ledger = yaml.safe_load(
        (repository_root / "configs/datasets/evidence_roles_20260920_v4.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert ledger["datasets"]["inclusivehar_v4"]["target_reopening_permitted"] is False
    assert "consumed" in ledger["datasets"]["aicos_har_v1"]["provider_test_role"]
    assert (
        "consumed" in (ledger["datasets"]["aicos_har_v1"]["provider_development_folds_1_to_5_role"])
    )
    assert ledger["claim_policy"]["state_of_the_art_claim_supported"] is False


def test_public_entry_documents_point_to_current_evidence(repository_root: Path) -> None:
    readme = (repository_root / "README.md").read_text(encoding="utf-8")
    status = (repository_root / "docs/PROJECT_STATUS.md").read_text(encoding="utf-8")
    results = (repository_root / "results/README.md").read_text(encoding="utf-8")
    for text in (readme, status, results):
        assert "EVIDENCE_INDEX.md" in text
    assert "PUBLICATION_CHECKLIST.md" in readme
    assert "EVIDENCE_SUPERSESSION.md" in readme
