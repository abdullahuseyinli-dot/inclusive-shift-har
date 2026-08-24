from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from inclusive_shift_har.evaluation.secondary_aggregation import (
    SecondaryAggregationError,
    aggregate_efficiency_profiles,
    aggregate_sensor_stress,
    build_parser,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file


def _write(path: Path, value: dict[str, Any], *, hash_field: str = "record_sha256") -> None:
    value[hash_field] = canonical_json_sha256(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_efficiency_aggregation_preserves_seed_rows_and_operator_scope(tmp_path: Path) -> None:
    entries: list[dict[str, Any]] = []
    for seed, latency in ((11, 1.0), (29, 1.2)):
        path = tmp_path / f"profile-{seed}.json"
        profile: dict[str, Any] = {
            "schema_version": "1.0.0",
            "record_kind": "cuda_neural_efficiency_profile",
            "status": "profile_complete",
            "model_id": "compact",
            "seed": seed,
            "precision": "float32",
            "input_shape": [1, 128, 6],
            "final_freeze_inventory_sha256": "a" * 64,
            "parameters": {
                "total": 100,
                "trainable": 90,
                "state_tensor_bytes": 400,
            },
            "complexity": {
                "supported_operator_macs_per_window": 1000,
                "estimated_flops_from_supported_macs_per_window": 2000,
                "coverage_status": "supported_operator_subset_only",
            },
            "latency_ms_per_batch": {
                "mean": latency,
                "median": latency,
                "mean_per_window": latency,
                "percentiles": {"95.0": latency + 0.1},
            },
            "vram_bytes": {"peak_allocated": 1000 + seed, "peak_reserved": 2000 + seed},
            "model_size": {"checkpoint_file_bytes": 800},
            "model_selection_use": False,
        }
        _write(path, profile)
        entries.append(
            {
                "model_id": "compact",
                "seed": seed,
                "batch_size": 1,
                "precision": "float32",
                "path": path.relative_to(tmp_path).as_posix(),
                "file_sha256": sha256_file(path),
                "record_sha256": profile["record_sha256"],
            }
        )
    index_path = tmp_path / "efficiency-index.json"
    index: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "frozen_neural_efficiency_profile_index",
        "status": "complete_create_only",
        "profile_config_sha256": "b" * 64,
        "final_freeze_inventory_sha256": "a" * 64,
        "profile_count": 2,
        "profiles": entries,
        "target_signals_or_metrics_accessed": False,
        "model_selection_use": False,
    }
    _write(index_path, index)
    with pytest.raises(SecondaryAggregationError, match="index contract mismatch"):
        aggregate_efficiency_profiles(
            index_path,
            artifact_root=tmp_path,
            destination=tmp_path / "aggregate.json",
            created_at_utc="2099-01-01T00:00:00Z",
        )


def _participant_report(participant: str, macro_f1: float) -> dict[str, Any]:
    return {
        "participants": [
            {
                "participant_id": participant,
                "macro_f1": macro_f1,
                "balanced_accuracy": macro_f1,
            }
        ]
    }


def test_sensor_stress_aggregation_builds_participant_and_cohort_delta_tables(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source-clean.json"
    source_record: dict[str, Any] = {
        "record_kind": "secondary_source_clean_reference",
        "participant_level_report": _participant_report("source-1", 0.8),
    }
    _write(source_path, source_record)
    target_path = tmp_path / "target-clean.json"
    target_record: dict[str, Any] = {
        "record_kind": "locked_target_per_seed_result",
        "participant_level_report": _participant_report("target-1", 0.6),
    }
    _write(target_path, target_record)
    source_reference = {
        "model_id": "compact",
        "seed": 11,
        "cohort": "source",
        "record_path": source_path.name,
        "record_file_sha256": sha256_file(source_path),
        "record_sha256": source_record["record_sha256"],
        "prediction_sha256": "1" * 64,
    }
    target_reference = {
        "model_id": "compact",
        "seed": 11,
        "cohort": "target",
        "record_path": target_path.name,
        "record_file_sha256": sha256_file(target_path),
        "record_sha256": target_record["record_sha256"],
        "prediction_sha256": "2" * 64,
    }
    conditions = [f"condition-{index}" for index in range(6)]
    stress_entries: list[dict[str, Any]] = []
    for cohort, participant, clean_value, stress_value in (
        ("source", "source-1", 0.8, 0.7),
        ("target", "target-1", 0.6, 0.4),
    ):
        for condition in conditions:
            path = tmp_path / f"{cohort}-{condition}.json"
            record: dict[str, Any] = {
                "record_kind": "secondary_sensor_reliability_evaluation",
                "model_id": "compact",
                "seed": 11,
                "cohort": cohort,
                "track_role": "secondary_post_confirmatory",
                "primary_claim_eligible": False,
                "used_for_model_selection": False,
                "target_based_tuning": False,
                "stress_config_sha256": "3" * 64,
                "condition": {"condition_id": condition},
                "participant_level_report": _participant_report(participant, stress_value),
            }
            _write(path, record)
            stress_entries.append(
                {
                    "model_id": "compact",
                    "seed": 11,
                    "cohort": cohort,
                    "condition_id": condition,
                    "record_path": path.name,
                    "record_file_sha256": sha256_file(path),
                    "record_sha256": record["record_sha256"],
                    "prediction_sha256": "4" * 64,
                    "clean_value_for_test": clean_value,
                }
            )
    index_path = tmp_path / "stress-index.json"
    index: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "secondary_sensor_reliability_index",
        "status": "complete_create_only",
        "track_role": "secondary_post_confirmatory",
        "primary_claim_eligible": False,
        "used_for_model_selection": False,
        "target_clean_inference_rerun": False,
        "raw_target_materialization_invoked": False,
        "opening_or_unlock_invoked": False,
        "stress_config_sha256": "3" * 64,
        "final_freeze_inventory_sha256": "5" * 64,
        "locked_target_index_record_sha256": "6" * 64,
        "model_seed_count": 1,
        "condition_count": 6,
        "stress_result_count": 12,
        "source_clean_references": [source_reference],
        "target_clean_references": [target_reference],
        "stress_results": stress_entries,
    }
    _write(index_path, index)
    result = aggregate_sensor_stress(
        index_path,
        artifact_root=tmp_path,
        destination=tmp_path / "stress-aggregate.json",
        created_at_utc="2099-01-01T00:00:00Z",
    )
    assert len(result["tables"]["participant_seed_deltas"]) == 12
    source_row = next(
        row
        for row in result["tables"]["cohort_condition_aggregates"]
        if row["cohort"] == "source" and row["condition_id"] == "condition-0"
    )
    target_row = next(
        row
        for row in result["tables"]["cohort_condition_aggregates"]
        if row["cohort"] == "target" and row["condition_id"] == "condition-0"
    )
    assert source_row["mean_delta_participant_macro_f1"] == pytest.approx(-0.1)
    assert target_row["mean_delta_participant_macro_f1"] == pytest.approx(-0.2)
    gap = result["tables"]["source_target_gap_deltas"][0]
    assert gap["gap_change_under_stress"] == pytest.approx(0.1)


def test_aggregate_module_parser_has_both_operations() -> None:
    parser = build_parser()
    for operation in ("efficiency", "sensor-stress"):
        parsed = parser.parse_args(
            [
                operation,
                "--index",
                "index.json",
                "--artifact-root",
                ".",
                "--destination",
                "aggregate.json",
                "--created-at-utc",
                "2099-01-01T00:00:00Z",
            ]
        )
        assert parsed.operation == operation


def test_sensor_stress_index_rejects_absolute_result_record_references(tmp_path: Path) -> None:
    record_path = tmp_path / "source-clean.json"
    record: dict[str, Any] = {
        "record_kind": "secondary_source_clean_reference",
        "participant_level_report": _participant_report("source-1", 0.8),
    }
    _write(record_path, record)
    index_path = tmp_path / "stress-index.json"
    index: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "secondary_sensor_reliability_index",
        "status": "complete_create_only",
        "track_role": "secondary_post_confirmatory",
        "primary_claim_eligible": False,
        "used_for_model_selection": False,
        "target_clean_inference_rerun": False,
        "raw_target_materialization_invoked": False,
        "opening_or_unlock_invoked": False,
        "stress_config_sha256": "3" * 64,
        "final_freeze_inventory_sha256": "5" * 64,
        "locked_target_index_record_sha256": "6" * 64,
        "model_seed_count": 1,
        "condition_count": 6,
        "stress_result_count": 12,
        "source_clean_references": [
            {
                "model_id": "compact",
                "seed": 11,
                "cohort": "source",
                "record_path": str(record_path.resolve()),
                "record_file_sha256": sha256_file(record_path),
                "record_sha256": record["record_sha256"],
                "prediction_sha256": "1" * 64,
            }
        ],
        "target_clean_references": [],
        "stress_results": [],
    }
    _write(index_path, index)

    with pytest.raises(SecondaryAggregationError, match="artifact-root-relative"):
        aggregate_sensor_stress(
            index_path,
            artifact_root=tmp_path,
            destination=tmp_path / "aggregate.json",
            created_at_utc="2099-01-01T00:00:00Z",
        )
