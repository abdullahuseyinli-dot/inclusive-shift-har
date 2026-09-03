from __future__ import annotations

from pathlib import Path

from inclusive_shift_har.manifests.canonical import canonical_json_sha256, load_json_strict


def test_v2_research_summary_is_self_hashed_and_claim_bounded(
    repository_root: Path,
) -> None:
    path = repository_root / "results/development/fuse_reframe_v2/research_summary_v1.json"
    record = load_json_strict(path)
    claimed_hash = record.pop("record_sha256")
    assert claimed_hash == canonical_json_sha256(record)

    assert record["repository_scope"].startswith("inclusive_shift_har_smartphone_wearable_imu")
    assert record["protocol"]["inclusivehar_target_participants_reopened"] is False
    assert (
        record["robust_multiscale_residual_pyramid"]["mean_participant_macro_f1"]
        == 0.8379032940215063
    )
    assert record["paired_gsp_to_rmrp"]["engineering_advance_gate_passed"] is False
    assert record["claim_policy"]["breakthrough_confirmed"] is False
    assert record["claim_policy"]["new_zero_shot_target_claim_allowed"] is False
    assert record["daghar_external"]["frozen_candidate_changed_after_evaluation"] is False
    assert record["har_pmd"]["status"] == "not_downloaded_not_evaluated"


def test_v2_publication_narrative_files_are_present(repository_root: Path) -> None:
    assert (repository_root / "docs/research/FUSE_REFRAME_V2_RESEARCH_REPORT.md").is_file()
    assert (repository_root / "paper/FUSE_REFRAME_V2_ADDENDUM.md").is_file()
