from __future__ import annotations

from pathlib import Path

import numpy as np

from inclusive_shift_har.manifests.canonical import (
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)


def test_hera_ctgr_v2_summary_is_self_hashed_and_claim_safe(
    repository_root: Path,
) -> None:
    path = repository_root / "results/development/hera_ctgr_v2_retrospective_v1_summary.json"
    record = load_json_strict(path)
    claimed = record.pop("record_sha256")
    assert claimed == canonical_json_sha256(record)
    result = record["five_seed_mean_results"]
    assert result["hera_ctgr_v2_full"]["mean_participant_macro_f1"] == 0.867493040748332
    assert result["frozen_hera_ctgr_v1_strict"]["mean_participant_macro_f1"] == (0.8684917479119463)
    assert record["v2_full_comparison"]["breakthrough"] is False
    assert record["advancement_gate"]["overall_passed"] is False
    assert record["claim_boundary"]["participants_11_through_20_loaded"] is False
    assert record["claim_boundary"]["daghar_loaded"] is False
    assert record["claim_boundary"]["state_of_the_art_claim_allowed"] is False


def test_hera_ctgr_v2_summary_matches_preserved_audit_when_available(
    repository_root: Path,
) -> None:
    summary = load_json_strict(
        repository_root / "results/development/hera_ctgr_v2_retrospective_v1_summary.json"
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
    assert result["outer_labels_used_for_zero_shot_training_selection_or_routing"] is False
    assert result["participants_11_through_20_loaded"] is False
    assert result["daghar_loaded"] is False
    assert result["advancement_gate"]["overall_passed"] is False
    assert result["breakthrough_gate"]["overall_passed"] is False
    assert sha256_file(prediction_path) == run["predictions_sha256"]


def test_hera_ctgr_v2_predictions_obey_fallback_contract_when_available(
    repository_root: Path,
) -> None:
    summary = load_json_strict(
        repository_root / "results/development/hera_ctgr_v2_retrospective_v1_summary.json"
    )
    prediction_path = repository_root / summary["run"]["predictions_path"]
    if not prediction_path.exists():
        return
    with np.load(prediction_path, allow_pickle=False) as payload:
        assert payload["labels"].shape == (725,)
        assert set(payload["participant_ids"].tolist()) == {str(value) for value in range(1, 11)}
        for seed in (11, 23, 47, 89, 131):
            core = payload[f"seed_{seed}__decision_separated_core_probabilities"]
            full = payload[f"seed_{seed}__hera_ctgr_v2_full_probabilities"]
            routed = payload[f"seed_{seed}__hera_ctgr_v2_full_routed"]
            assert core.shape == (725, 3)
            assert np.isfinite(core).all()
            np.testing.assert_allclose(core.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
            np.testing.assert_array_equal(full, core)
            assert not routed.any()


def test_hera_ctgr_v2_result_narrative_is_present(repository_root: Path) -> None:
    assert (repository_root / "docs/research/HERA_CTGR_V2_RETROSPECTIVE_V1_RESULTS.md").is_file()
