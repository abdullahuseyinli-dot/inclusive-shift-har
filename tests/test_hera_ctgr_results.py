from __future__ import annotations

from pathlib import Path

from inclusive_shift_har.manifests.canonical import (
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)


def test_hera_ctgr_summary_is_self_hashed_and_claim_safe(repository_root: Path) -> None:
    path = repository_root / "results/development/hera_ctgr_retrospective_v1_summary.json"
    record = load_json_strict(path)
    claimed = record.pop("record_sha256")
    assert claimed == canonical_json_sha256(record)
    assert record["best_real_method"]["name"] == "hera_ctgr_strict"
    assert record["best_real_method"]["mean_participant_macro_f1"] == 0.8684917479119463
    assert record["best_real_method"]["breakthrough"] is False
    assert record["advancement_gate"]["strict_hera_passed"] is False
    assert record["advancement_gate"]["full_hera_passed"] is False
    assert record["claim_boundary"]["participants_11_through_20_loaded"] is False
    assert record["claim_boundary"]["daghar_loaded"] is False
    assert record["claim_boundary"]["state_of_the_art_claim_allowed"] is False


def test_hera_ctgr_summary_matches_preserved_audit_when_available(
    repository_root: Path,
) -> None:
    summary = load_json_strict(
        repository_root / "results/development/hera_ctgr_retrospective_v1_summary.json"
    )
    protocol = summary["protocol"]
    assert sha256_file(repository_root / protocol["path"]) == protocol["sha256"]
    run = summary["run"]
    run_path = repository_root / run["path"]
    prediction_path = repository_root / run["predictions_path"]
    if not run_path.exists():
        return
    assert sha256_file(run_path) == run["sha256"]
    result = load_json_strict(run_path)
    claimed = result.pop("record_sha256")
    assert claimed == canonical_json_sha256(result)
    assert claimed == run["record_sha256"]
    assert result["outer_labels_used_for_training_selection_calibration_or_routing"] is False
    assert result["participants_11_through_20_loaded"] is False
    assert result["daghar_loaded"] is False
    assert sha256_file(prediction_path) == run["predictions_sha256"]


def test_hera_ctgr_result_narrative_is_present(repository_root: Path) -> None:
    assert (repository_root / "docs/research/HERA_CTGR_RETROSPECTIVE_V1_RESULTS.md").is_file()
