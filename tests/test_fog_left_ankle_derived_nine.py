from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

import inclusive_shift_har.experiments.fog_left_ankle_derived_nine as l9v
from inclusive_shift_har.experiments.fog_rf_feature_weight_factorial import (
    _write_json_create_only,
    method_report,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/fog_left_ankle_derived_nine_v1.yaml"


def test_config_is_frozen_and_mutations_fail() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    l9v.validate_config(config)
    assert config["experiment_id"] == "fog-left-ankle-derived-nine-probe-v1"
    assert l9v.RUN_DIRECTORY_NAME == "fog-left-ankle-derived-nine-seed11-20260908-003"
    assert config["resources"]["maximum_fit_attempts"] == 5
    assert config["resources"]["compute_wall_seconds"] == 900
    assert config["weighting"]["expected_training_rows"] == [940, 825, 947, 829, 1075]
    mutated = copy.deepcopy(config)
    mutated["estimator"]["n_estimators"] = 501
    with pytest.raises(ValueError, match="config changed"):
        l9v.validate_config(mutated)


def test_frozen_hash_conventions_and_exact_lv_partitions_are_distinct() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["reference"]["availability_mask_sha256"] == (
        "27c4d82a970975d93784cce15aa67799b07300eeacf7bb82d1ee62ec7f6b9a3d"
    )
    assert config["reference"]["availability_mask_generic_array_sha256"] == (
        l9v.EXPECTED_GENERIC_MASK_SHA256
    )
    assert l9v.METHOD_ORDER[l9v.B0_METHOD_INDEX] == "b0"
    fallback_ids = l9v.fallback_source_method_ids(122)
    assert fallback_ids.dtype == np.dtype("<U2")
    assert fallback_ids.tolist() == ["b0"] * 122
    assert l9v.EXPECTED_L9V_FEATURE_SHA256 == (
        "b1b38b52e66d10b31c6d70a689f90e917f81e269340534fb422d25d9514fdc70"
    )
    assert len(set(l9v.EXPECTED_TRAINING_ID_HASHES)) == 5
    assert len(set(l9v.EXPECTED_EVALUATION_ID_HASHES)) == 5
    assert len(set(l9v.EXPECTED_WEIGHT_HASHES)) == 5


def test_shared_six_channel_features_are_selected_by_name_not_position() -> None:
    values = np.asarray([[30.0, 10.0, 40.0, 20.0], [31.0, 11.0, 41.0, 21.0]])
    selected = l9v.name_aligned_features(
        values,
        ("feature_c", "feature_a", "feature_d", "feature_b"),
        ("feature_a", "feature_b"),
    )
    assert np.array_equal(selected, np.asarray([[10.0, 20.0], [11.0, 21.0]]))
    with pytest.raises(ValueError, match="requested feature missing"):
        l9v.name_aligned_features(values, ("c", "a", "d", "b"), ("a", "missing"))


def test_legacy_coverage_contract_does_not_require_an_invented_self_hash() -> None:
    rows = [{"window_id": f"window-{index}"} for index in range(1939)]
    coverage = {
        "record_kind": "fog_spatial_information_coverage_audit",
        "status": "complete_zero_fit_observable_coverage_audit",
        "model_fits": 0,
        "activity_fog_clinical_outcome_columns_accessed": False,
        "raw_sha256": l9v.RAW_SHA256,
        "raw_size_bytes": l9v.RAW_SIZE,
        "candidate_availability": rows,
    }
    assert l9v.coverage_rows_from_audit(coverage) is rows
    corrupted = copy.deepcopy(coverage)
    corrupted["activity_fog_clinical_outcome_columns_accessed"] = True
    with pytest.raises(ValueError, match="coverage audit contract changed"):
        l9v.coverage_rows_from_audit(corrupted)


def test_gate_requires_every_frozen_condition() -> None:
    comparison = {
        "mean_difference": 0.015,
        "participant_wins": 14,
        "bottom_30_percent_difference": -0.01,
        "worst_participant_difference": -0.03,
        "class_recall_differences": {
            "mobility": -0.01,
            "sitting": -0.02,
            "standing": -0.02,
        },
        "leave_one_participant_out_mean_differences": [0.001] * 22,
    }
    leave_fold = [{"mean_participant_difference": 0.001} for _ in range(5)]
    assert l9v._gate(comparison, leave_fold, source_reference_exact=True)["status"] == "pass"
    failed = copy.deepcopy(comparison)
    failed["participant_wins"] = 13
    result = l9v._gate(failed, leave_fold, source_reference_exact=True)
    assert result["status"] == "fail"
    assert result["checks"]["participant_wins"] is False


def test_analysis_requires_all_four_controls_and_does_not_build_architecture() -> None:
    roster = [f"fogstar:{index:03d}" for index in range(1, 23)]
    people = np.repeat(np.asarray(roster), 3)
    labels = np.tile(np.arange(3, dtype=np.int64), 22)
    strong = np.eye(3, dtype=np.float64)[labels]
    weak_labels = np.zeros_like(labels)
    weak = np.eye(3, dtype=np.float64)[weak_labels]
    probabilities = {"l9v": strong, "lv": weak, "bv": weak, "b0": weak, "f3": weak}
    reports = {
        name: method_report(
            labels=labels, probabilities=value, participant_ids=people, roster=roster
        )
        for name, value in probabilities.items()
    }
    result = l9v.analyse(
        reports,
        labels=labels,
        probabilities=probabilities,
        participants=people,
        participant_folds=np.asarray([min(index // 5, 4) for index in range(22)]),
        roster=roster,
        source_reference_exact=True,
    )
    assert set(result["gates"]) == {
        "l9v_minus_lv",
        "l9v_minus_bv",
        "l9v_minus_b0",
        "l9v_minus_f3",
    }
    assert result["advancement"]["status"] == "pass"
    assert result["architecture_built"] is False
    summary = l9v._outcome_summary(result)
    assert "l9v_minus_lv" in summary
    assert "PASS" not in summary


def test_sixth_fit_attempt_is_rejected_before_estimator_work(tmp_path: Path) -> None:
    empty_values = np.empty((0, 120), dtype=np.float64)
    empty_labels = np.empty(0, dtype=np.int64)
    empty_people = np.empty(0, dtype=np.str_)
    empty_indices = np.empty(0, dtype=np.int64)
    with pytest.raises(ValueError, match="five-attempt budget"):
        l9v._fit_l9v(
            training_values=empty_values,
            training_labels=empty_labels,
            training_participants=empty_people,
            evaluation_values=empty_values,
            evaluation_participants=empty_people,
            training_global_indices=empty_indices,
            evaluation_global_indices=empty_indices,
            outer_fold=0,
            attempt_number=6,
            feature_names=tuple(f"f{index}" for index in range(120)),
            output_directory=tmp_path,
            deadline=float("inf"),
        )
    assert list(tmp_path.iterdir()) == []


def test_prevalidation_manifest_allows_only_declared_lifecycle_files(tmp_path: Path) -> None:
    (tmp_path / "evidence.txt").write_text("evidence", encoding="utf-8")
    _write_json_create_only(tmp_path / "artifact_manifest.json", l9v._artifact_manifest(tmp_path))
    assert l9v._verify_artifact_manifest(tmp_path) == 1
    (tmp_path / "validation_worker_shutdown.json").write_text("{}", encoding="utf-8")
    (tmp_path / "validation.json").write_text("{}", encoding="utf-8")
    (tmp_path / "completion_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "controller_final_receipt.json").write_text("{}", encoding="utf-8")
    assert l9v._verify_artifact_manifest(tmp_path) == 1
    (tmp_path / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="coverage changed"):
        l9v._verify_artifact_manifest(tmp_path)
