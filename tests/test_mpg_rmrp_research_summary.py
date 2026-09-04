from __future__ import annotations

from pathlib import Path

from inclusive_shift_har.manifests.canonical import (
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)


def test_mpg_rmrp_committed_summary_is_honest_and_self_hashed(repository_root: Path) -> None:
    path = repository_root / "results/development/mpg_rmrp_v1_summary.json"
    record = load_json_strict(path)
    claimed = record.pop("record_sha256")
    assert claimed == canonical_json_sha256(record)
    assert record["advancement_gate_passed"] is False
    assert record["across_seed_summary"]["positive_seed_count"] == 0
    assert record["across_seed_summary"]["positive_fold_count"] == 0
    assert record["claim_boundary"]["confirmatory_claim_allowed"] is False
    assert record["claim_boundary"]["independent_new_ability_relevant_cohort_required"] is True
    assert record["across_seed_summary"]["mpg_rmrp"]["mean_participant_macro_f1"] < 0.60
    assert record["across_seed_summary"]["flat_rmrp"]["mean_participant_macro_f1"] > 0.83


def test_mpg_rmrp_committed_summary_matches_local_audit_when_available(
    repository_root: Path,
) -> None:
    summary = load_json_strict(repository_root / "results/development/mpg_rmrp_v1_summary.json")
    if not (repository_root / ".audit/v3/mpg-rmrp").is_dir():
        return
    for field in ("selection", "multiseed_evidence"):
        reference = summary[field]
        path = repository_root / reference["path"]
        record = load_json_strict(path)
        claimed = record.pop("record_sha256")
        assert claimed == canonical_json_sha256(record)
        assert reference["record_sha256"] == claimed
        assert reference["file_sha256"] == sha256_file(path)


def test_mpg_rmrp_result_narrative_is_present(repository_root: Path) -> None:
    assert (repository_root / "docs/research/MPG_RMRP_V1_RESULTS.md").is_file()
