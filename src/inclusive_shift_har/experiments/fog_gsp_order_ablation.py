"""Bounded corrected-FoG GSP temporal-order ablation.

The runner fits exactly three fixed participant-weighted Random Forest cells over
five pre-window participant folds. It writes create-only exploratory evidence and
stops. Validation replays all checkpoints from cached label-free features without
downloading the source or fitting a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
import traceback
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import yaml
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from inclusive_shift_har.data.external_har import load_fog_star, observable_modelling_pool
from inclusive_shift_har.experiments.fog_rf_feature_weight_factorial import (
    _array_sha256,
    _environment,
    _git_state,
    _mapping,
    _predict_proba_deterministically,
    _require,
    _sealed,
    _verify_sealed,
    _write_bytes_create_only,
    _write_json_create_only,
    method_report,
    paired_comparison,
    participant_first_weights,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.geometric_spectral_pyramid import (
    extract_geometric_spectral_pyramid_features,
    geometric_spectral_pyramid_feature_names,
)
from inclusive_shift_har.preprocessing.features import extract_engineered_features

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
StringArray: TypeAlias = NDArray[np.str_]
PermutationArray: TypeAlias = NDArray[np.int16]

CELL_ORDER = ("a", "b", "c")
CLASS_NAMES = ("mobility", "sitting", "standing")
SIX_CHANNEL_NAMES = (
    "lin_acc_x",
    "lin_acc_y",
    "lin_acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)
PERMUTATION_NAMESPACE = "fog-gsp-order-ablation-v1"
REFERENCE_HASHES = {
    "feature_cache.npz": "a9d6cfb4c3d0ea75623b3152255775f09292c510d5797d1c0b58192db20e4bea",
    "predictions.npz": "c3c7ab375f20bf875a8ff871cf1f1dc7c6b7718d896f37f2d1f6440227ceac6b",
    "result.json": "30b27aec3cdd7848d9a08d53ad5118d6e7d7fd85e4db49ed2f0194e2debd12f1",
    "validation.json": "1946240d9cb6a2f92777dc463c0516df30eeb10182eb8911dbe800d4df4355e5",
    "completion_manifest.json": (
        "e91e6c9fbe61256fd49bc54574ea85728b8c960afe4c366458b5af30a96fcce2"
    ),
}
REQUIRED_RUN_FILES = (
    "analysis.json",
    "config_snapshot.yaml",
    "data_audit.json",
    "feature_cache.npz",
    "feature_cache_metadata.json",
    "fold_reports.json",
    "OUTCOME_SUMMARY.md",
    "participant_metrics.json",
    "permutation_audit.json",
    "predictions.npz",
    "preflight.json",
    "protocol_snapshot.json",
    "reference_control_receipt.json",
    "reference_control_replay.json",
    "result.json",
    "runtime.json",
    "source_receipt.json",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _read_yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), str(path))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _strict_fields(actual: Mapping[str, Any], expected: Mapping[str, Any], name: str) -> None:
    _require(dict(actual) == dict(expected), f"{name} changed")


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject any drift from the outcome-blind frozen contract."""

    expected_roster = [f"fogstar:{index:03d}" for index in range(1, 23)]
    _require(config.get("schema_version") == "1.0.0", "unexpected config schema")
    _require(config.get("experiment_id") == PERMUTATION_NAMESPACE, "unexpected experiment id")
    _require(config.get("status") == "frozen_before_outcomes", "config is not frozen")
    _require(
        config.get("evidence_status") == "corrected_fog_exploratory_development_not_confirmation",
        "evidence status changed",
    )
    source = _mapping(config.get("source"), "source")
    for key, expected in {
        "dataset_id": "fog_star_v3",
        "provider_record": "https://zenodo.org/records/17838806",
        "content_url": ("https://zenodo.org/api/records/17838806/files/sensor_data.csv/content"),
        "declared_size_bytes": 119_629_580,
        "declared_md5": "952a37ab147da35e6d4e7a1e9bac44cb",
        "expected_sha256": "888875133fef386affddd0a5864f122d527d8d7bc588a69905cc0871bef53477",
        "participant_roster": expected_roster,
        "class_order": list(CLASS_NAMES),
        "target_rate_hz": 50.0,
        "window_samples": 128,
        "gravity_cutoff_hz": 0.30,
        "maximum_gap_factor": 3.0,
        "segmentation": "participant_session_timestamp_gap_nonfinite_run",
        "signal_lane": "linear_acceleration_plus_gyroscope",
        "channel_names": list(SIX_CHANNEL_NAMES),
        "resampling_and_gravity_annotation_independent": True,
        "annotation_role": "posthoc_window_label_and_scoring_eligibility_only",
        "participant_plan_created_before_windowing": True,
        "raw_local_mirror": False,
    }.items():
        _require(source.get(key) == expected, f"source field changed: {key}")
    reference = _mapping(config.get("reference_control"), "reference control")
    expected_reference = {
        "experiment_id": "fog-rf-feature-weight-factorial-v1",
        "run_id": "fog-rf-feature-weight-factorial-seed11-20260907-001",
        "cell_id": "f1",
        "cell_index": 1,
        "feature_cache_sha256": REFERENCE_HASHES["feature_cache.npz"],
        "predictions_sha256": REFERENCE_HASHES["predictions.npz"],
        "result_sha256": REFERENCE_HASHES["result.json"],
        "validation_sha256": REFERENCE_HASHES["validation.json"],
        "completion_manifest_sha256": REFERENCE_HASHES["completion_manifest.json"],
        "exact_feature_replay_required_before_fits": True,
        "exact_observable_probability_replay_required_before_gsp_fits": True,
    }
    _strict_fields(reference, expected_reference, "reference control")
    _strict_fields(
        _mapping(config.get("folds"), "folds"),
        {
            "seed": 11,
            "count": 5,
            "assignment_role": "outer",
            "assignment_source": "loader_frozen_external-har-prewindow-partitions-v1",
            "estimator_random_state": "11_plus_outer_fold_index_shared_by_all_three_cells",
            "execution_order": list(CELL_ORDER),
        },
        "fold contract",
    )
    features = _mapping(config.get("features"), "features")
    _require(features.get("dtype") == "float64", "feature dtype changed")
    _require(
        features.get("cache_scope") == "all_annotation_independent_observable_candidates",
        "feature cache scope changed",
    )
    _strict_fields(
        _mapping(features.get("a"), "feature a"),
        {
            "extractor": ("inclusive_shift_har.preprocessing.features.extract_engineered_features"),
            "expected_dimension": 80,
        },
        "feature a",
    )
    for cell, extractor in {
        "b": (
            "inclusive_shift_har.models.geometric_spectral_pyramid."
            "extract_geometric_spectral_pyramid_features"
        ),
        "c": "same_as_b_after_joint_within_window_permutation",
    }.items():
        _strict_fields(
            _mapping(features.get(cell), f"feature {cell}"),
            {"extractor": extractor, "sampling_rate_hz": 50.0, "expected_dimension": 1400},
            f"feature {cell}",
        )
    _require(features.get("no_additional_denoising") is True, "denoising was introduced")
    _require(features.get("no_gravity_channel_addition") is True, "gravity lane was introduced")
    _require(features.get("no_learned_normalization") is True, "normalization was introduced")
    _strict_fields(
        _mapping(config.get("permutation"), "permutation"),
        {
            "namespace": PERMUTATION_NAMESPACE,
            "serialization": (
                "utf8_namespace_pipe_observable_window_id_pipe_zero_padded_three_digit_index"
            ),
            "construction": "sort_indices_by_sha256_digest_then_numeric_index",
            "scope": "separately_within_each_observable_candidate",
            "joint_across_all_six_channels": True,
            "stage": "after_continuous_preprocessing_before_all_gsp_construction",
            "train_and_evaluation_rule_identical": True,
            "sample_tuple_multiset_exact": True,
            "numeric_moment_tolerance": 1e-10,
            "matrix_dtype": "int16",
            "matrix_shape": [1939, 128],
            "extra_permutation_replicates_allowed": False,
        },
        "permutation",
    )
    _strict_fields(
        _mapping(config.get("estimator"), "estimator"),
        {
            "library_class": "sklearn.ensemble.RandomForestClassifier",
            "n_estimators": 500,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "bootstrap": True,
            "class_weight": None,
            "n_jobs": 4,
            "probability_class_order": [0, 1, 2],
            "no_hyperparameter_search": True,
            "prediction_workers": 1,
        },
        "estimator",
    )
    _strict_fields(
        _mapping(config.get("weighting"), "weighting"),
        {
            "raw_sample_weight": "1 / (m_i * n_ic)",
            "normalization": "mean_one_over_outer_training_rows",
            "absent_person_class_rows": "no_rows_no_invented_weight",
            "training_rows": "outer_training_intersection_scoring_eligibility",
        },
        "weighting",
    )
    _strict_fields(
        _mapping(config.get("cells"), "cells"),
        {
            "a": {
                "label": "RF-ordinary80-six-participant-first-control",
                "features": "a",
            },
            "b": {"label": "RF-GSP1400-six-original-order", "features": "b"},
            "c": {"label": "RF-GSP1400-six-joint-shuffled-order", "features": "c"},
        },
        "cells",
    )
    statistics = _mapping(config.get("statistics"), "statistics")
    _strict_fields(
        statistics,
        {
            "paired_participant_bootstrap_resamples": 10_000,
            "paired_participant_bootstrap_seed": 1729,
            "percentile_interval": [0.025, 0.975],
            "bottom_fraction": 0.30,
            "missing_class_metric_rule": "fixed_class_zero_division_zero",
            "zero_scored_window_rule": "promotion_incomplete_never_drop_person",
            "comparisons": {
                "practical_primary": "b_minus_a",
                "order_mechanism": "b_minus_c",
                "descriptive": "c_minus_a",
            },
        },
        "statistics",
    )
    _strict_fields(
        _mapping(config.get("practical_promotion_gate_b_minus_a"), "practical gate"),
        {
            "minimum_mean_gain": 0.015,
            "minimum_bottom_30_difference": -0.010,
            "minimum_worst_participant_difference": -0.030,
            "minimum_mobility_recall_difference": -0.010,
            "minimum_sitting_recall_difference": -0.020,
            "minimum_standing_recall_difference": -0.020,
            "minimum_participant_wins": 14,
            "require_all_leave_one_participant_out_means_positive": True,
            "leave_one_out_positive_tolerance": 1e-12,
            "require_all_22_roster_people_scored": True,
        },
        "practical gate",
    )
    _strict_fields(
        _mapping(config.get("order_mechanism_gate_b_minus_c"), "mechanism gate"),
        {
            "minimum_mean_gain": 0.010,
            "require_positive_bootstrap_lower_endpoint": True,
            "minimum_participant_wins": 14,
            "require_all_leave_one_participant_out_means_positive": True,
            "leave_one_out_positive_tolerance": 1e-12,
            "require_all_22_roster_people_scored": True,
        },
        "mechanism gate",
    )
    _strict_fields(
        _mapping(config.get("resource_limits"), "resources"),
        {
            "controller_count": 1,
            "maximum_fit_attempts": 15,
            "expected_completed_fits": 15,
            "compute_wall_seconds": 7200,
            "cpu_estimator_workers": 4,
            "prediction_workers": 1,
            "blas_threads": 1,
            "gpu_allowed": False,
            "additional_benchmark_fits_allowed": False,
            "first_registered_a_fold_is_timing_benchmark": True,
            "additional_seeds_allowed": False,
            "automatic_follow_on_allowed": False,
        },
        "resources",
    )
    claims = _mapping(config.get("claims"), "claims")
    _require(
        set(claims)
        == {
            "invention_claim_allowed",
            "independent_confirmation_claim_allowed",
            "information_impossibility_claim_allowed",
            "physiology_claim_allowed",
            "temporal_model_superiority_claim_allowed",
            "robustness_claim_allowed",
            "publication_queue_allowed",
            "physical_pilot_continuation",
        }
        and all(value is False for value in claims.values()),
        "claim policy changed",
    )


def deterministic_joint_permutations(
    window_ids: StringArray,
    *,
    sample_count: int = 128,
    namespace: str = PERMUTATION_NAMESPACE,
) -> PermutationArray:
    """Return one label-free, within-window tuple permutation for every ID."""

    ids = np.asarray(window_ids, dtype=np.str_)
    _require(ids.ndim == 1 and ids.size > 0, "window IDs must be one-dimensional")
    _require(len(set(ids.tolist())) == ids.size, "window IDs must be unique")
    _require(sample_count == 128, "sample count changed")
    _require(namespace == PERMUTATION_NAMESPACE, "permutation namespace changed")
    output = np.empty((ids.size, sample_count), dtype=np.int16)
    for row, window_id in enumerate(ids.tolist()):
        keyed = [
            (
                hashlib.sha256(f"{namespace}|{window_id}|{index:03d}".encode()).digest(),
                index,
            )
            for index in range(sample_count)
        ]
        output[row] = np.asarray(
            [index for _, index in sorted(keyed, key=lambda item: (item[0], item[1]))],
            dtype=np.int16,
        )
    expected: PermutationArray = np.arange(sample_count, dtype=np.int16)
    _require(
        all(np.array_equal(np.sort(row), expected) for row in output),
        "a generated row is not a complete permutation",
    )
    return output


def apply_joint_permutations(
    signals: NDArray[np.floating[Any]], permutations: NDArray[np.integer[Any]]
) -> NDArray[np.floating[Any]]:
    """Apply one time-index permutation jointly to all channels in each window."""

    values = np.asarray(signals)
    indices = np.asarray(permutations, dtype=np.int64)
    _require(
        values.ndim == 3 and values.shape[2] == 6 and indices.shape == values.shape[:2],
        "signals and permutations are not aligned",
    )
    _require(
        bool(np.all((indices >= 0) & (indices < values.shape[1]))),
        "permutation index is out of range",
    )
    shuffled = np.take_along_axis(values, indices[:, :, None], axis=1)
    inverse = np.empty_like(indices)
    rows = np.arange(indices.shape[0])[:, None]
    inverse[rows, indices] = np.arange(indices.shape[1], dtype=np.int64)[None, :]
    restored = np.take_along_axis(shuffled, inverse[:, :, None], axis=1)
    _require(np.array_equal(restored, values), "joint permutation did not preserve tuples exactly")
    return shuffled


def _moment_audit(
    original: NDArray[np.floating[Any]], shuffled: NDArray[np.floating[Any]]
) -> dict[str, Any]:
    first = np.asarray(original, dtype=np.float64)
    second = np.asarray(shuffled, dtype=np.float64)
    original_mean = first.mean(axis=1)
    shuffled_mean = second.mean(axis=1)
    original_centered = first - original_mean[:, None, :]
    shuffled_centered = second - shuffled_mean[:, None, :]
    original_cov = np.einsum("ntc,ntd->ncd", original_centered, original_centered) / first.shape[1]
    shuffled_cov = np.einsum("ntc,ntd->ncd", shuffled_centered, shuffled_centered) / second.shape[1]
    mean_difference = float(np.max(np.abs(original_mean - shuffled_mean)))
    covariance_difference = float(np.max(np.abs(original_cov - shuffled_cov)))
    tolerance = 1e-10
    _require(mean_difference <= tolerance, "permutation changed a channel mean beyond tolerance")
    _require(
        covariance_difference <= tolerance,
        "permutation changed a global covariance beyond tolerance",
    )
    return {
        "float64_maximum_absolute_channel_mean_difference": mean_difference,
        "float64_maximum_absolute_global_covariance_difference": covariance_difference,
        "absolute_tolerance": tolerance,
        "passed": True,
    }


def _permutation_row_hashes(permutations: NDArray[np.integer[Any]]) -> list[str]:
    return [_array_sha256(np.asarray(row, dtype=np.int16)) for row in permutations]


def _effective_estimator_parameters(random_state: int) -> dict[str, Any]:
    estimator = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        bootstrap=True,
        class_weight=None,
        random_state=random_state,
        n_jobs=4,
    )
    return cast(dict[str, Any], estimator.get_params(deep=False))


def _fit_cell(
    *,
    feature_values: FloatArray,
    labels: IntArray,
    participants: StringArray,
    training: BoolArray,
    evaluation: BoolArray,
    cell_id: str,
    fold_index: int,
    feature_names: tuple[str, ...],
    output_directory: Path,
    attempt_index: int,
    deadline: float,
) -> tuple[FloatArray, dict[str, Any]]:
    _require(attempt_index < 15, "fit-attempt budget exhausted")
    if time.perf_counter() >= deadline:
        raise TimeoutError("compute cap reached before starting next fit attempt")
    attempt_number = attempt_index + 1
    random_state = 11 + fold_index
    sample_weight = participant_first_weights(labels[training], participants[training])
    started_record = _sealed(
        {
            "record_kind": "fog_gsp_order_fit_attempt_started",
            "attempt_number": attempt_number,
            "cell_id": cell_id,
            "outer_fold": fold_index,
            "registered_timing_benchmark": cell_id == "a" and fold_index == 0,
            "started_at_utc": _now(),
            "random_state": random_state,
            "training_row_count": int(training.sum()),
            "evaluation_candidate_count": int(evaluation.sum()),
        }
    )
    _write_json_create_only(
        output_directory / "fit_attempts" / f"{attempt_number:02d}--started.json",
        started_record,
    )
    estimator = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        bootstrap=True,
        class_weight=None,
        random_state=random_state,
        n_jobs=4,
    )
    wall_started = time.perf_counter()
    with threadpool_limits(limits=1):
        fit_started = time.perf_counter()
        estimator.fit(
            feature_values[training],
            labels[training],
            sample_weight=sample_weight,
        )
        fit_seconds = time.perf_counter() - fit_started
        predict_started = time.perf_counter()
        probabilities = _predict_proba_deterministically(estimator, feature_values[evaluation])
        prediction_seconds = time.perf_counter() - predict_started
    elapsed = time.perf_counter() - wall_started
    if time.perf_counter() > deadline:
        raise TimeoutError("compute cap reached during registered fit or prediction")
    _require(np.array_equal(estimator.classes_, np.arange(3)), "RF class order changed")
    _require(
        probabilities.shape == (int(evaluation.sum()), 3)
        and bool(np.isfinite(probabilities).all())
        and bool(np.all(probabilities >= 0))
        and np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12),
        "RF probabilities are invalid",
    )
    metadata: dict[str, Any] = {
        "attempt_number": attempt_number,
        "cell_id": cell_id,
        "outer_fold": fold_index,
        "random_state": random_state,
        "registered_timing_benchmark": cell_id == "a" and fold_index == 0,
        "feature_names": list(feature_names),
        "feature_names_sha256": canonical_json_sha256(list(feature_names)),
        "training_participants": sorted(np.unique(participants[training]).tolist()),
        "training_row_count": int(training.sum()),
        "evaluation_participants": sorted(np.unique(participants[evaluation]).tolist()),
        "evaluation_candidate_count": int(evaluation.sum()),
        "class_weight": None,
        "sample_weight_policy": "participant_first_mean_one",
        "sample_weight_summary": {
            "minimum": float(sample_weight.min()),
            "maximum": float(sample_weight.max()),
            "mean": float(sample_weight.mean()),
            "sum": float(sample_weight.sum()),
            "sha256": _array_sha256(sample_weight),
        },
        "effective_parameters": estimator.get_params(deep=False),
        "prediction_worker_count": 1,
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "fit_and_predict_seconds": elapsed,
    }
    checkpoint_path = output_directory / "checkpoints" / f"{cell_id}--fold-{fold_index}.pkl"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("xb") as stream:
        pickle.dump({"metadata": metadata, "estimator": estimator}, stream, protocol=5)
    metadata["checkpoint"] = {
        "path": checkpoint_path.relative_to(output_directory).as_posix(),
        "sha256": sha256_file(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
    }
    completed_record = _sealed(
        {
            "record_kind": "fog_gsp_order_fit_attempt_completed",
            "attempt_number": attempt_number,
            "cell_id": cell_id,
            "outer_fold": fold_index,
            "completed_at_utc": _now(),
            "fit_seconds": fit_seconds,
            "prediction_seconds": prediction_seconds,
            "checkpoint": metadata["checkpoint"],
        }
    )
    _write_json_create_only(
        output_directory / "fit_attempts" / f"{attempt_number:02d}--completed.json",
        completed_record,
    )
    return probabilities, metadata


def _participant_values(report: Mapping[str, Any]) -> FloatArray:
    rows = cast(list[dict[str, Any]], report["participants"])
    _require(all(row.get("eligible") is True for row in rows), "comparison requires full roster")
    return np.asarray([float(row["macro_f1"]) for row in rows], dtype=np.float64)


def _events(
    labels: IntArray,
    old_probabilities: FloatArray,
    new_probabilities: FloatArray,
    participant_ids: StringArray,
    roster: Sequence[str],
) -> dict[str, Any]:
    old = old_probabilities.argmax(axis=1)
    new = new_probabilities.argmax(axis=1)
    rescued = (old != labels) & (new == labels)
    harmed = (old == labels) & (new != labels)
    both_wrong = (old != labels) & (new != labels)
    changed = old != new
    boundary_old = (old == 0) != (labels == 0)
    boundary_new = (new == 0) != (labels == 0)
    return {
        "changed_decisions": int(changed.sum()),
        "rescues": int(rescued.sum()),
        "harms": int(harmed.sum()),
        "wrong_to_different_wrong": int((changed & both_wrong).sum()),
        "both_wrong": int(both_wrong.sum()),
        "mobility_boundary_errors_before": int(boundary_old.sum()),
        "mobility_boundary_errors_after": int(boundary_new.sum()),
        "sitting_standing_errors_before": int(((old != labels) & ~boundary_old).sum()),
        "sitting_standing_errors_after": int(((new != labels) & ~boundary_new).sum()),
        "by_true_class": [
            {
                "class_name": CLASS_NAMES[label],
                "rescues": int((rescued & (labels == label)).sum()),
                "harms": int((harmed & (labels == label)).sum()),
            }
            for label in range(3)
        ],
        "participants": [
            {
                "participant_id": participant,
                "rescues": int((rescued & (participant_ids == participant)).sum()),
                "harms": int((harmed & (participant_ids == participant)).sum()),
                "changed": int((changed & (participant_ids == participant)).sum()),
            }
            for participant in roster
        ],
    }


def analyse(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    probabilities: FloatArray,
    labels: IntArray,
    participant_ids: StringArray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute frozen comparisons, gates, event topology and decision."""

    roster = cast(list[str], _mapping(config["source"], "source")["participant_roster"])
    complete = all(
        report.get("status") == "complete_full_roster"
        and report.get("scored_participant_count") == len(roster)
        for report in reports.values()
    )
    if not complete:
        return {
            "status": "incomplete_promotion_analysis",
            "reports": dict(reports),
            "comparisons": {},
            "event_topology": {},
            "practical_promotion_gate": {
                "status": "incomplete",
                "checks": {"full_22_person_roster_scored": False},
            },
            "order_mechanism_gate": {
                "status": "incomplete",
                "checks": {"full_22_person_roster_scored": False},
            },
            "decision": "incomplete_stop_no_follow_on",
            "evidence_status": config["evidence_status"],
        }
    comparisons = {
        "b_minus_a": paired_comparison(reports["b"], reports["a"]),
        "b_minus_c": paired_comparison(reports["b"], reports["c"]),
        "c_minus_a": paired_comparison(reports["c"], reports["a"]),
    }
    practical = _mapping(comparisons["b_minus_a"], "b minus a")
    practical_classes = _mapping(practical["class_recall_differences"], "practical class diffs")
    practical_config = _mapping(config["practical_promotion_gate_b_minus_a"], "practical config")
    practical_checks = {
        "full_22_person_roster_scored": True,
        "mean_gain": float(practical["mean_difference"])
        >= float(practical_config["minimum_mean_gain"]),
        "bottom_30": float(practical["bottom_30_percent_difference"])
        >= float(practical_config["minimum_bottom_30_difference"]),
        "worst_participant": float(practical["worst_participant_difference"])
        >= float(practical_config["minimum_worst_participant_difference"]),
        "mobility_recall": float(practical_classes["mobility"])
        >= float(practical_config["minimum_mobility_recall_difference"]),
        "sitting_recall": float(practical_classes["sitting"])
        >= float(practical_config["minimum_sitting_recall_difference"]),
        "standing_recall": float(practical_classes["standing"])
        >= float(practical_config["minimum_standing_recall_difference"]),
        "participant_wins": int(practical["participant_wins"])
        >= int(practical_config["minimum_participant_wins"]),
        "all_leave_one_participant_out_positive": all(
            float(value) > float(practical_config["leave_one_out_positive_tolerance"])
            for value in cast(list[float], practical["leave_one_participant_out_mean_differences"])
        ),
    }
    mechanism = _mapping(comparisons["b_minus_c"], "b minus c")
    mechanism_config = _mapping(config["order_mechanism_gate_b_minus_c"], "mechanism config")
    interval = cast(list[float], mechanism["mean_difference_95_percent_bootstrap_interval"])
    mechanism_checks = {
        "full_22_person_roster_scored": True,
        "mean_gain": float(mechanism["mean_difference"])
        >= float(mechanism_config["minimum_mean_gain"]),
        "positive_bootstrap_lower_endpoint": float(interval[0]) > 0.0,
        "participant_wins": int(mechanism["participant_wins"])
        >= int(mechanism_config["minimum_participant_wins"]),
        "all_leave_one_participant_out_positive": all(
            float(value) > float(mechanism_config["leave_one_out_positive_tolerance"])
            for value in cast(list[float], mechanism["leave_one_participant_out_mean_differences"])
        ),
    }
    practical_pass = all(practical_checks.values())
    mechanism_pass = all(mechanism_checks.values())
    if practical_pass and mechanism_pass:
        decision = "gsp_package_and_order_hypothesis_eligible_for_separate_follow_on"
    elif practical_pass:
        decision = "retain_gsp_package_without_order_mechanism_claim"
    elif mechanism_pass:
        decision = "order_dependence_detected_but_no_practical_promotion"
    else:
        decision = "close_finite_gsp_rf_question_retain_f1_control"
    return {
        "status": "complete_analysis",
        "reports": dict(reports),
        "comparisons": comparisons,
        "event_topology": {
            "a_to_b": _events(labels, probabilities[0], probabilities[1], participant_ids, roster),
            "c_to_b": _events(labels, probabilities[2], probabilities[1], participant_ids, roster),
            "a_to_c": _events(labels, probabilities[0], probabilities[2], participant_ids, roster),
        },
        "participant_score_correlations": {
            "method_order": list(CELL_ORDER),
            "pearson": np.corrcoef(
                np.stack([_participant_values(reports[name]) for name in CELL_ORDER])
            ).tolist(),
        },
        "practical_promotion_gate": {
            "status": "pass" if practical_pass else "fail",
            "comparison": "b_minus_a",
            "checks": practical_checks,
            "passed_check_count": sum(practical_checks.values()),
            "required_check_count": len(practical_checks),
        },
        "order_mechanism_gate": {
            "status": "pass" if mechanism_pass else "fail",
            "comparison": "b_minus_c",
            "checks": mechanism_checks,
            "passed_check_count": sum(mechanism_checks.values()),
            "required_check_count": len(mechanism_checks),
        },
        "decision": decision,
        "failure_action": "stop_no_seed_feature_estimator_or_permutation_sweep",
        "evidence_status": config["evidence_status"],
    }


def _source_manifest(repository_root: Path) -> dict[str, Any]:
    paths = (
        "AGENTS.md",
        "src/inclusive_shift_har/data/external_har.py",
        "src/inclusive_shift_har/data/participant_partitions.py",
        "src/inclusive_shift_har/preprocessing/features.py",
        "src/inclusive_shift_har/models/geometric_spectral_pyramid.py",
        "src/inclusive_shift_har/experiments/fog_rf_feature_weight_factorial.py",
        "src/inclusive_shift_har/experiments/fog_gsp_order_ablation.py",
        "configs/experiments/fog_gsp_order_ablation_v1.yaml",
        "docs/research/FOG_GSP_ORDER_ABLATION_V1_PROTOCOL.md",
    )
    return {
        "files": [{"path": name, "sha256": sha256_file(repository_root / name)} for name in paths]
    }


def _snapshot_inputs(
    repository_root: Path,
    config_path: Path,
    protocol_path: Path,
    output_directory: Path,
) -> None:
    _write_bytes_create_only(output_directory / "config_snapshot.yaml", config_path.read_bytes())
    _write_json_create_only(
        output_directory / "protocol_snapshot.json",
        _sealed(
            {
                "record_kind": "fog_gsp_order_protocol_snapshot",
                "source_path": protocol_path.relative_to(repository_root).as_posix(),
                "source_sha256": sha256_file(protocol_path),
                "text": protocol_path.read_text(encoding="utf-8"),
            }
        ),
    )


def _reference_arrays(
    reference_run: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, NDArray[Any]], dict[str, NDArray[Any]], dict[str, Any]]:
    reference = _mapping(config["reference_control"], "reference")
    _require(reference_run.name == reference["run_id"], "reference run ID changed")
    checked: list[dict[str, Any]] = []
    for name, expected in REFERENCE_HASHES.items():
        path = reference_run / name
        _require(path.is_file(), f"missing reference file: {name}")
        actual = sha256_file(path)
        _require(actual == expected, f"reference hash mismatch: {name}")
        checked.append({"path": name, "sha256": actual})
    completion = _read_json(reference_run / "completion_manifest.json")
    validation = _read_json(reference_run / "validation.json")
    result = _read_json(reference_run / "result.json")
    _verify_sealed(completion, "reference completion")
    _verify_sealed(validation, "reference validation")
    _verify_sealed(result, "reference result")
    _require(
        completion.get("status") == "complete_and_independently_replayed"
        and validation.get("status") == "validated"
        and result.get("fit_count") == 20,
        "reference run is not a validated complete matrix",
    )
    entries = cast(list[dict[str, Any]], completion["artifacts"])
    for entry in entries:
        path = (reference_run / str(entry["path"])).resolve()
        _require(_is_relative_to(path, reference_run), "reference manifest path escapes")
        _require(path.is_file(), "reference manifest file missing")
        _require(sha256_file(path) == entry["sha256"], "reference manifest hash mismatch")
    with np.load(reference_run / "feature_cache.npz", allow_pickle=False) as archive:
        cache = {key: archive[key] for key in archive.files}
    with np.load(reference_run / "predictions.npz", allow_pickle=False) as archive:
        predictions = {key: archive[key] for key in archive.files}
    _require(predictions["cell_ids"].tolist()[1] == "f1", "reference cell index changed")
    receipt = _sealed(
        {
            "record_kind": "fog_gsp_order_reference_control_receipt",
            "run_path": str(reference_run),
            "files": checked,
            "completion_manifest_entries_verified": len(entries),
            "completion_status": completion["status"],
            "validation_status": validation["status"],
            "reference_cell_id": "f1",
            "reference_cell_index": 1,
        }
    )
    return cache, predictions, receipt


def _outcome_summary(analysis: Mapping[str, Any]) -> str:
    reports = _mapping(analysis["reports"], "reports")
    practical = _mapping(analysis["practical_promotion_gate"], "practical gate")
    mechanism = _mapping(analysis["order_mechanism_gate"], "mechanism gate")
    lines = [
        "# Corrected FoG GSP temporal-order ablation outcome",
        "",
        f"Practical B-A gate: **{str(practical['status']).upper()}**.",
        f"Order B-C gate: **{str(mechanism['status']).upper()}**.",
        f"Decision: **{analysis['decision']}**.",
        "",
        "| Cell | Mean participant macro-F1 | Pooled accuracy | Bottom 30% | Worst |",
        "|---|---:|---:|---:|---:|",
    ]
    for cell in CELL_ORDER:
        report = _mapping(reports[cell], cell)
        primary = _mapping(report["primary"], "primary")
        pooled = _mapping(report["pooled"], "pooled")
        lines.append(
            f"| {cell.upper()} | {float(primary['mean_participant_macro_f1']):.6f} | "
            f"{float(pooled['accuracy']):.6f} | "
            f"{float(primary['bottom_30_percent_participant_macro_f1']):.6f} | "
            f"{float(primary['worst_participant_macro_f1']):.6f} |"
        )
    comparisons = _mapping(analysis["comparisons"], "comparisons")
    lines.extend(
        [
            "",
            "| Contrast | Mean difference | 95% participant bootstrap CI | Wins/harms/ties |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ("b_minus_a", "b_minus_c", "c_minus_a"):
        comparison = _mapping(comparisons[name], name)
        interval = cast(list[float], comparison["mean_difference_95_percent_bootstrap_interval"])
        lines.append(
            f"| {name} | {float(comparison['mean_difference']):+.6f} | "
            f"[{float(interval[0]):+.6f}, {float(interval[1]):+.6f}] | "
            f"{comparison['participant_wins']}/{comparison['participant_harms']}/"
            f"{comparison['participant_ties']} |"
        )
    lines.extend(
        [
            "",
            "Evidence is corrected FoG exploratory development evidence. No extra seed,",
            "model, permutation, dataset, physical pilot, publication, or follow-on was launched.",
            "",
        ]
    )
    return "\n".join(lines)


def run_experiment(
    *,
    repository_root: Path,
    evidence_root: Path,
    config_path: Path,
    protocol_path: Path,
    reference_run: Path,
    output_directory: Path,
    code_commit: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Run the frozen 15-fit matrix and preserve complete or failed evidence."""

    repository_root = repository_root.resolve()
    evidence_root = evidence_root.resolve()
    config_path = config_path.resolve()
    protocol_path = protocol_path.resolve()
    reference_run = reference_run.resolve()
    output_directory = output_directory.resolve()
    config = _read_yaml(config_path)
    validate_config(config)
    _require(config_path.parent.parent.parent == repository_root, "config must be in repository")
    _require(
        protocol_path.parent.parent.parent == repository_root, "protocol must be in repository"
    )
    _require(
        reference_run.parent == evidence_root / ".audit" / "fog_rf_feature_weight_factorial",
        "reference run must be the dedicated preserved FoG factorial run",
    )
    _require(
        output_directory.parent == evidence_root / ".audit" / "fog_gsp_order_ablation",
        "output must be one create-only run below dedicated evidence root",
    )
    _require(not output_directory.exists(), "output directory already exists")
    state = _git_state(repository_root)
    _require(state["clean"] is True, "experiment source worktree must be clean")
    _require(state["commit"] == code_commit, "code commit does not match HEAD")
    _require(timeout_seconds == 7200, "runtime cap must remain 7200 seconds")
    output_directory.mkdir(parents=True, exist_ok=False)
    _snapshot_inputs(repository_root, config_path, protocol_path, output_directory)
    started_at = _now()
    started = time.perf_counter()
    deadline = started + timeout_seconds
    attempts_started = 0
    fold_reports: list[dict[str, Any]] = []
    try:
        reference_cache, reference_predictions, reference_receipt = _reference_arrays(
            reference_run, config
        )
        _write_json_create_only(
            output_directory / "reference_control_receipt.json", reference_receipt
        )
        source_config = _mapping(config["source"], "source")
        load_started = time.perf_counter()
        data = load_fog_star(
            participant_limit=None,
            target_rate_hz=float(source_config["target_rate_hz"]),
            window_samples=int(source_config["window_samples"]),
            gravity_cutoff_hz=float(source_config["gravity_cutoff_hz"]),
            maximum_gap_factor=float(source_config["maximum_gap_factor"]),
        )
        data.validate()
        source_load_seconds = time.perf_counter() - load_started
        _require(time.perf_counter() < deadline, "compute cap reached during source load")
        _require(len(data.receipts) == 1, "expected one source receipt")
        receipt = data.receipts[0]
        receipt.validate()
        _require(receipt.computed_sha256 == source_config["expected_sha256"], "source SHA mismatch")
        _require(
            receipt.received_size_bytes == source_config["declared_size_bytes"],
            "source size mismatch",
        )
        _require(
            receipt.computed_declared_digest == source_config["declared_md5"], "source MD5 mismatch"
        )
        roster = tuple(cast(list[str], source_config["participant_roster"]))
        plan = data.participant_partition_plan
        _require(plan is not None, "pre-window participant plan missing")
        assert plan is not None
        _require(plan.participant_roster == roster, "participant roster changed")
        plan_audit = plan.audit()
        modelling, scoring_indices, eligibility = observable_modelling_pool(
            data, include_supervised_labels=True
        )
        _require(
            modelling.labels.size == 1939
            and scoring_indices.size == 1213
            and data.labels.size == 1213,
            "corrected candidate/scoring counts changed",
        )
        _require(
            np.array_equal(modelling.window_ids[scoring_indices], data.window_ids)
            and np.array_equal(modelling.participant_ids[scoring_indices], data.participant_ids)
            and np.array_equal(modelling.labels[scoring_indices], data.labels),
            "scored projection changed",
        )
        assignment = plan.resolve(roster, fold_count=5, seed=11, role="outer")
        observable_fold = np.asarray(
            [assignment[str(item)] for item in modelling.participant_ids], dtype=np.int64
        )
        scored_fold = observable_fold[scoring_indices]
        source_receipt = _sealed(
            {
                "record_kind": "fog_gsp_order_source_receipt",
                "source_receipts": [item.to_dict() for item in data.receipts],
                "expected_sha256": source_config["expected_sha256"],
                "source_hash_match": True,
                "raw_local_mirror": False,
                "target_or_other_external_dataset_loaded": False,
            }
        )
        _write_json_create_only(output_directory / "source_receipt.json", source_receipt)
        data_audit = _sealed(
            {
                "record_kind": "fog_gsp_order_data_audit",
                "dataset": data.summary(),
                "participant_partition_plan": plan_audit,
                "observable_candidate_count": int(modelling.labels.size),
                "scored_window_count": int(data.labels.size),
                "scored_participants": sorted(np.unique(data.participant_ids).tolist()),
                "roster_participants_without_scored_rows": sorted(
                    set(roster) - set(data.participant_ids.tolist())
                ),
                "observable_processing_annotation_independent": True,
                "scoring_labels_applied_posthoc": True,
                "source_load_seconds": source_load_seconds,
            }
        )
        _write_json_create_only(output_directory / "data_audit.json", data_audit)
        feature_started = time.perf_counter()
        with threadpool_limits(limits=1):
            ordinary = extract_engineered_features(
                modelling.signals, channel_names=SIX_CHANNEL_NAMES
            )
            permutations = deterministic_joint_permutations(modelling.window_ids)
            shuffled = apply_joint_permutations(modelling.signals, permutations)
            moment_audit = _moment_audit(modelling.signals, shuffled)
            gsp_names = geometric_spectral_pyramid_feature_names()
            ordered_gsp = extract_geometric_spectral_pyramid_features(
                modelling.signals, sampling_rate_hz=50.0
            )
            shuffled_gsp = extract_geometric_spectral_pyramid_features(
                shuffled, sampling_rate_hz=50.0
            )
        feature_seconds = time.perf_counter() - feature_started
        _require(time.perf_counter() < deadline, "compute cap reached during feature extraction")
        _require(
            ordinary.values.shape == (1939, 80)
            and ordered_gsp.shape == (1939, 1400)
            and shuffled_gsp.shape == (1939, 1400)
            and len(gsp_names) == 1400,
            "feature dimensions changed",
        )
        forbidden_feature_tokens = (
            "participant",
            "subject",
            "label",
            "timestamp",
            "location",
            "window_id",
            "fold",
        )
        _require(
            all(
                token not in name.lower()
                for name in (*ordinary.names, *gsp_names)
                for token in forbidden_feature_tokens
            ),
            "identity, label, time, location or fold proxy entered a feature schema",
        )
        _require(
            bool(np.isfinite(ordinary.values).all())
            and bool(np.isfinite(ordered_gsp).all())
            and bool(np.isfinite(shuffled_gsp).all()),
            "non-finite feature value detected",
        )
        _require(
            not np.array_equal(ordered_gsp, shuffled_gsp),
            "joint permutation did not alter the order-sensitive GSP matrix",
        )
        _require(
            tuple(ordinary.names) == tuple(reference_cache["six_names"].tolist())
            and np.array_equal(ordinary.values, reference_cache["six_values"]),
            "A feature matrix does not exactly replay reference F1",
        )
        reference_alignment = {
            "observable_window_ids_exact": np.array_equal(
                modelling.window_ids, reference_cache["observable_window_ids"]
            ),
            "observable_participant_ids_exact": np.array_equal(
                modelling.participant_ids, reference_cache["observable_participant_ids"]
            ),
            "observable_fold_index_exact": np.array_equal(
                observable_fold, reference_cache["observable_fold_index"]
            ),
            "scoring_indices_exact": np.array_equal(
                scoring_indices, reference_cache["scoring_indices"]
            ),
            "scoring_eligibility_exact": np.array_equal(
                eligibility, reference_cache["scoring_eligibility"]
            ),
            "scored_labels_exact": np.array_equal(data.labels, reference_cache["scored_labels"]),
            "scored_window_ids_exact": np.array_equal(
                data.window_ids, reference_cache["scored_window_ids"]
            ),
        }
        _require(all(reference_alignment.values()), "reference candidate alignment changed")
        per_window_hashes = _permutation_row_hashes(permutations)
        permutation_audit = _sealed(
            {
                "record_kind": "fog_gsp_order_permutation_audit",
                "namespace": PERMUTATION_NAMESPACE,
                "matrix_shape": list(permutations.shape),
                "stored_dtype": "int16",
                "matrix_sha256": _array_sha256(np.asarray(permutations, dtype=np.int16)),
                "per_window": [
                    {"window_id": str(window_id), "permutation_sha256": row_hash}
                    for window_id, row_hash in zip(
                        modelling.window_ids.tolist(), per_window_hashes, strict=True
                    )
                ],
                "all_rows_complete_permutations": True,
                "all_six_channel_sample_tuples_restored_bit_exact": True,
                "no_cross_window_operation": True,
                "moment_preservation": moment_audit,
                "labels_or_outcomes_used": False,
            }
        )
        _write_json_create_only(output_directory / "permutation_audit.json", permutation_audit)
        cache_path = output_directory / "feature_cache.npz"
        with cache_path.open("xb") as stream:
            np.savez_compressed(
                stream,
                a_values=np.asarray(ordinary.values, dtype=np.float64),
                b_values=np.asarray(ordered_gsp, dtype=np.float64),
                c_values=np.asarray(shuffled_gsp, dtype=np.float64),
                a_names=np.asarray(ordinary.names, dtype=np.str_),
                b_names=np.asarray(gsp_names, dtype=np.str_),
                c_names=np.asarray(gsp_names, dtype=np.str_),
                permutation_matrix=np.asarray(permutations, dtype=np.int16),
                observable_window_ids=modelling.window_ids,
                observable_participant_ids=modelling.participant_ids,
                observable_fold_index=observable_fold,
                scoring_indices=scoring_indices,
                scoring_eligibility=eligibility,
                scored_labels=data.labels,
                scored_participant_ids=data.participant_ids,
                scored_window_ids=data.window_ids,
                scored_fold_index=scored_fold,
            )
        arrays = {
            "a_values": ordinary.values,
            "b_values": ordered_gsp,
            "c_values": shuffled_gsp,
            "a_names": np.asarray(ordinary.names, dtype=np.str_),
            "b_names": np.asarray(gsp_names, dtype=np.str_),
            "c_names": np.asarray(gsp_names, dtype=np.str_),
            "permutation_matrix": np.asarray(permutations, dtype=np.int16),
            "observable_window_ids": modelling.window_ids,
            "observable_participant_ids": modelling.participant_ids,
            "observable_fold_index": observable_fold,
            "scoring_indices": scoring_indices,
            "scoring_eligibility": eligibility,
            "scored_labels": data.labels,
            "scored_participant_ids": data.participant_ids,
            "scored_window_ids": data.window_ids,
            "scored_fold_index": scored_fold,
        }
        cache_metadata = _sealed(
            {
                "record_kind": "fog_gsp_order_feature_cache",
                "path": cache_path.name,
                "sha256": sha256_file(cache_path),
                "arrays": {
                    name: {
                        "shape": list(np.asarray(value).shape),
                        "dtype": str(np.asarray(value).dtype),
                        "sha256": _array_sha256(np.asarray(value)),
                    }
                    for name, value in arrays.items()
                },
                "feature_dimensions": {"a": 80, "b": 1400, "c": 1400},
                "feature_extraction_seconds": feature_seconds,
                "cache_built_before_scoring_projection": True,
                "reference_a_feature_matrix_exact": True,
                "reference_alignment": reference_alignment,
                "raw_signal_windows_stored": False,
                "permutation_audit_record_sha256": permutation_audit["record_sha256"],
            }
        )
        _write_json_create_only(output_directory / "feature_cache_metadata.json", cache_metadata)
        features_by_cell = {
            "a": np.asarray(ordinary.values, dtype=np.float64),
            "b": np.asarray(ordered_gsp, dtype=np.float64),
            "c": np.asarray(shuffled_gsp, dtype=np.float64),
        }
        names_by_cell = {"a": tuple(ordinary.names), "b": gsp_names, "c": gsp_names}
        effective = {
            f"{cell}--fold-{fold}": {
                "cell_id": cell,
                "outer_fold": fold,
                "parameters": _effective_estimator_parameters(11 + fold),
                "feature_dimension": features_by_cell[cell].shape[1],
                "weighting": "participant_first_mean_one",
            }
            for cell in CELL_ORDER
            for fold in range(5)
        }
        with threadpool_limits(limits=1):
            environment = _environment()
        preflight = _sealed(
            {
                "record_kind": "fog_gsp_order_preflight",
                "created_before_model_fits": True,
                "repository_root": str(repository_root),
                "git": state,
                "code_commit": code_commit,
                "configuration": {
                    "path": config_path.relative_to(repository_root).as_posix(),
                    "sha256": sha256_file(config_path),
                },
                "protocol": {
                    "path": protocol_path.relative_to(repository_root).as_posix(),
                    "sha256": sha256_file(protocol_path),
                },
                "source_manifest": _source_manifest(repository_root),
                "source_receipt_record_sha256": source_receipt["record_sha256"],
                "data_audit_record_sha256": data_audit["record_sha256"],
                "reference_control_record_sha256": reference_receipt["record_sha256"],
                "feature_cache_record_sha256": cache_metadata["record_sha256"],
                "permutation_audit_record_sha256": permutation_audit["record_sha256"],
                "participant_partition_plan_sha256": plan_audit["plan_sha256"],
                "fold_assignment": assignment,
                "effective_estimator_parameter_sets": effective,
                "environment": environment,
                "resource_limits": config["resource_limits"],
                "expected_fit_count": 15,
                "outcomes_available_when_written": False,
            }
        )
        _write_json_create_only(output_directory / "preflight.json", preflight)
        probabilities = np.full((3, modelling.labels.size, 3), np.nan, dtype=np.float64)
        fit_matrix_started = time.perf_counter()
        for cell_index, cell_id in enumerate(CELL_ORDER):
            for fold_index in range(5):
                evaluation = observable_fold == fold_index
                training = (~evaluation) & eligibility
                _require(evaluation.any(), "outer fold has no observable candidate")
                _require(
                    set(np.unique(modelling.labels[training]).tolist()) == {0, 1, 2},
                    "outer training fold lacks a class",
                )
                attempts_started += 1
                values, row = _fit_cell(
                    feature_values=features_by_cell[cell_id],
                    labels=modelling.labels,
                    participants=modelling.participant_ids,
                    training=training,
                    evaluation=evaluation,
                    cell_id=cell_id,
                    fold_index=fold_index,
                    feature_names=names_by_cell[cell_id],
                    output_directory=output_directory,
                    attempt_index=attempts_started - 1,
                    deadline=deadline,
                )
                probabilities[cell_index, evaluation] = values
                row["training_scored_class_counts"] = np.bincount(
                    modelling.labels[training], minlength=3
                ).tolist()
                row["evaluation_scored_window_count"] = int(np.sum(evaluation & eligibility))
                row["participant_partition_plan_sha256"] = plan_audit["plan_sha256"]
                fold_reports.append(row)
            if cell_id == "a":
                reference_observable = np.asarray(
                    reference_predictions["observable_probabilities"][1],
                    dtype=np.float64,
                )
                exact = np.array_equal(probabilities[0], reference_observable)
                replay = _sealed(
                    {
                        "record_kind": "fog_gsp_order_reference_control_replay",
                        "reference_cell_id": "f1",
                        "candidate_cell_id": "a",
                        "observable_window_ids_exact": np.array_equal(
                            modelling.window_ids,
                            reference_predictions["observable_window_ids"],
                        ),
                        "observable_probabilities_byte_exact": exact,
                        "maximum_absolute_probability_difference": float(
                            np.max(np.abs(probabilities[0] - reference_observable))
                        ),
                        "fit_count_before_decision": len(fold_reports),
                        "gsp_fits_started_before_decision": False,
                        "status": "passed" if exact else "failed_stop_before_gsp",
                    }
                )
                _write_json_create_only(output_directory / "reference_control_replay.json", replay)
                _require(exact, "fresh A probabilities do not exactly replay reference F1")
        fit_seconds = time.perf_counter() - fit_matrix_started
        _require(attempts_started == 15, "fit-attempt count changed")
        _require(len(fold_reports) == 15, "completed fit matrix is incomplete")
        _require(time.perf_counter() <= deadline, "compute cap exceeded")
        _require(bool(np.isfinite(probabilities).all()), "observable predictions incomplete")
        scored_probabilities = probabilities[:, scoring_indices]
        prediction_path = output_directory / "predictions.npz"
        with prediction_path.open("xb") as stream:
            np.savez_compressed(
                stream,
                cell_ids=np.asarray(CELL_ORDER, dtype=np.str_),
                observable_probabilities=probabilities,
                scored_probabilities=scored_probabilities,
                observable_window_ids=modelling.window_ids,
                observable_participant_ids=modelling.participant_ids,
                observable_fold_index=observable_fold,
                scoring_indices=scoring_indices,
                scored_labels=data.labels,
                scored_participant_ids=data.participant_ids,
                scored_window_ids=data.window_ids,
                scored_fold_index=scored_fold,
            )
        fold_payload = _sealed(
            {
                "record_kind": "fog_gsp_order_fold_reports",
                "fit_attempt_count": attempts_started,
                "completed_fit_count": len(fold_reports),
                "fold_count": 5,
                "cell_count": 3,
                "execution_order": list(CELL_ORDER),
                "rows": fold_reports,
            }
        )
        _write_json_create_only(output_directory / "fold_reports.json", fold_payload)
        reports = {
            cell: method_report(
                labels=data.labels,
                probabilities=scored_probabilities[index],
                participant_ids=data.participant_ids,
                roster=roster,
            )
            for index, cell in enumerate(CELL_ORDER)
        }
        participant_payload = _sealed(
            {"record_kind": "fog_gsp_order_participant_metrics", "methods": reports}
        )
        _write_json_create_only(output_directory / "participant_metrics.json", participant_payload)
        analysis = _sealed(
            {
                "record_kind": "fog_gsp_order_analysis",
                **analyse(
                    reports,
                    probabilities=scored_probabilities,
                    labels=data.labels,
                    participant_ids=data.participant_ids,
                    config=config,
                ),
            }
        )
        _write_json_create_only(output_directory / "analysis.json", analysis)
        _write_bytes_create_only(
            output_directory / "OUTCOME_SUMMARY.md",
            _outcome_summary(analysis).encode("utf-8"),
        )
        runtime = _sealed(
            {
                "record_kind": "fog_gsp_order_runtime",
                "started_at_utc": started_at,
                "completed_at_utc": _now(),
                "source_load_seconds": source_load_seconds,
                "feature_extraction_seconds": feature_seconds,
                "fit_and_prediction_seconds": fit_seconds,
                "total_run_seconds": time.perf_counter() - started,
                "compute_cap_seconds": timeout_seconds,
                "cap_exceeded": False,
                "controller_count": 1,
                "fit_attempt_count": attempts_started,
                "completed_model_fit_count": len(fold_reports),
                "first_registered_fit_benchmark_seconds": float(
                    fold_reports[0]["fit_and_predict_seconds"]
                ),
                "additional_benchmark_fit_count": 0,
                "gpu_used": False,
                "estimator_workers": 4,
                "prediction_workers": 1,
                "blas_threads_requested": 1,
            }
        )
        _write_json_create_only(output_directory / "runtime.json", runtime)
        result = _sealed(
            {
                "record_kind": "fog_gsp_order_result",
                "status": "complete_awaiting_independent_replay",
                "experiment_id": config["experiment_id"],
                "evidence_status": config["evidence_status"],
                "code_commit": code_commit,
                "source_sha256": receipt.computed_sha256,
                "participant_count_in_roster": len(roster),
                "scored_participant_count": len(np.unique(data.participant_ids)),
                "observable_candidate_count": int(modelling.labels.size),
                "scored_window_count": int(data.labels.size),
                "fit_attempt_count": attempts_started,
                "completed_fit_count": len(fold_reports),
                "feature_dimensions": {"a": 80, "b": 1400, "c": 1400},
                "checkpoint_total_size_bytes": int(
                    sum(
                        int(_mapping(row["checkpoint"], "checkpoint")["size_bytes"])
                        for row in fold_reports
                    )
                ),
                "runtime_by_cell": {
                    cell: {
                        "fit_seconds": float(
                            sum(
                                float(row["fit_seconds"])
                                for row in fold_reports
                                if row["cell_id"] == cell
                            )
                        ),
                        "prediction_seconds": float(
                            sum(
                                float(row["prediction_seconds"])
                                for row in fold_reports
                                if row["cell_id"] == cell
                            )
                        ),
                    }
                    for cell in CELL_ORDER
                },
                "method_summary": {name: report["primary"] for name, report in reports.items()},
                "practical_promotion_gate": analysis["practical_promotion_gate"],
                "order_mechanism_gate": analysis["order_mechanism_gate"],
                "decision": analysis["decision"],
                "analysis_record_sha256": analysis["record_sha256"],
                "prediction_artifact": {
                    "path": prediction_path.name,
                    "sha256": sha256_file(prediction_path),
                },
                "reference_control_exact_replay": True,
                "additional_seeds_launched": False,
                "automatic_follow_on_launched": False,
                "publication_queue_launched": False,
                "physical_pilot_launched": False,
                "target_or_other_external_dataset_loaded": False,
            }
        )
        _write_json_create_only(output_directory / "result.json", result)
        manifest_files = [
            path
            for path in output_directory.rglob("*")
            if path.is_file() and path.name != "artifact_manifest.json"
        ]
        manifest = _sealed(
            {
                "record_kind": "fog_gsp_order_artifact_manifest",
                "artifacts": [
                    {
                        "path": path.relative_to(output_directory).as_posix(),
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in sorted(manifest_files)
                ],
            }
        )
        _write_json_create_only(output_directory / "artifact_manifest.json", manifest)
        return result
    except Exception as error:
        failure = _sealed(
            {
                "record_kind": "fog_gsp_order_failure",
                "status": "failed_preserved",
                "started_at_utc": started_at,
                "failed_at_utc": _now(),
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.perf_counter() - started,
                "compute_cap_seconds": timeout_seconds,
                "fit_attempts_started": attempts_started,
                "completed_model_fit_checkpoints": len(
                    list((output_directory / "checkpoints").glob("*.pkl"))
                ),
                "automatic_retry": False,
                "automatic_follow_on": False,
            }
        )
        _write_json_create_only(output_directory / "failure.json", failure)
        raise


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], RandomForestClassifier]:
    with path.open("rb") as stream:
        payload = _mapping(pickle.load(stream), "checkpoint")
    metadata = _mapping(payload.get("metadata"), "checkpoint metadata")
    estimator = payload.get("estimator")
    _require(isinstance(estimator, RandomForestClassifier), "checkpoint estimator type changed")
    return metadata, estimator


def validate_run(run_directory: Path) -> dict[str, Any]:
    """Replay all checkpoints and seal a completed run without source access or fitting."""

    run_directory = run_directory.resolve()
    _require(run_directory.is_dir(), "run directory does not exist")
    _require(not (run_directory / "failure.json").exists(), "failed run cannot validate")
    _require(not (run_directory / "validation.json").exists(), "validation already exists")
    _require(not (run_directory / "completion_manifest.json").exists(), "completion exists")
    manifest = _read_json(run_directory / "artifact_manifest.json")
    _verify_sealed(manifest, "artifact manifest")
    artifacts = cast(list[dict[str, Any]], manifest["artifacts"])
    declared = [str(item["path"]) for item in artifacts]
    _require(len(declared) == len(set(declared)), "duplicate artifact path")
    actual = {
        path.relative_to(run_directory).as_posix()
        for path in run_directory.rglob("*")
        if path.is_file()
        and path.name
        not in {"artifact_manifest.json", "validation.json", "completion_manifest.json"}
    }
    _require(set(declared) == actual, "artifact manifest coverage differs")
    for artifact in artifacts:
        path = (run_directory / str(artifact["path"])).resolve()
        _require(_is_relative_to(path, run_directory), "artifact path escapes run")
        _require(
            path.is_file()
            and sha256_file(path) == artifact["sha256"]
            and path.stat().st_size == artifact["size_bytes"],
            "artifact integrity check failed",
        )
    for name in REQUIRED_RUN_FILES:
        _require((run_directory / name).is_file(), f"required artifact missing: {name}")
    config = _read_yaml(run_directory / "config_snapshot.yaml")
    validate_config(config)
    for name in (
        "data_audit.json",
        "feature_cache_metadata.json",
        "fold_reports.json",
        "participant_metrics.json",
        "permutation_audit.json",
        "preflight.json",
        "protocol_snapshot.json",
        "reference_control_receipt.json",
        "reference_control_replay.json",
        "result.json",
        "runtime.json",
        "source_receipt.json",
        "analysis.json",
    ):
        _verify_sealed(_read_json(run_directory / name), name)
    preflight = _read_json(run_directory / "preflight.json")
    repository_root = Path(str(preflight["repository_root"])).resolve()
    _require(repository_root.is_dir(), "recorded source repository is unavailable")
    recorded_git = _mapping(preflight["git"], "git")
    current_git = _git_state(repository_root)
    _require(
        current_git["commit"] == preflight["code_commit"] == recorded_git["commit"],
        "source commit changed since execution",
    )
    _require(current_git["clean"] is True, "source worktree is not clean during validation")
    source_rows = cast(
        list[dict[str, Any]],
        _mapping(preflight["source_manifest"], "manifest")["files"],
    )
    _require(len(source_rows) == 9, "source manifest coverage changed")
    for item in source_rows:
        source_path = (repository_root / str(item["path"])).resolve()
        _require(_is_relative_to(source_path, repository_root), "source path escapes repository")
        _require(
            source_path.is_file() and sha256_file(source_path) == item["sha256"],
            f"source file changed: {item['path']}",
        )
    configuration = _mapping(preflight["configuration"], "configuration")
    protocol = _mapping(preflight["protocol"], "protocol")
    _require(
        sha256_file(repository_root / str(configuration["path"])) == configuration["sha256"]
        and sha256_file(repository_root / str(protocol["path"])) == protocol["sha256"],
        "frozen config or protocol changed",
    )
    _require(
        sha256_file(run_directory / "config_snapshot.yaml") == configuration["sha256"],
        "configuration snapshot differs from committed configuration",
    )
    protocol_snapshot = _read_json(run_directory / "protocol_snapshot.json")
    _require(
        protocol_snapshot["source_path"] == protocol["path"]
        and protocol_snapshot["source_sha256"] == protocol["sha256"]
        and protocol_snapshot["text"]
        == (repository_root / str(protocol["path"])).read_text(encoding="utf-8"),
        "protocol snapshot differs from committed protocol",
    )
    source_receipt = _read_json(run_directory / "source_receipt.json")
    source_config = _mapping(config["source"], "source")
    receipt_rows = cast(list[dict[str, Any]], source_receipt["source_receipts"])
    _require(
        len(receipt_rows) == 1
        and source_receipt["expected_sha256"] == source_config["expected_sha256"]
        and source_receipt["source_hash_match"] is True
        and source_receipt["raw_local_mirror"] is False
        and source_receipt["target_or_other_external_dataset_loaded"] is False
        and receipt_rows[0]["computed_sha256"] == source_config["expected_sha256"]
        and receipt_rows[0]["received_size_bytes"] == source_config["declared_size_bytes"]
        and receipt_rows[0]["computed_declared_digest"] == source_config["declared_md5"],
        "source provenance record changed",
    )
    data_audit = _read_json(run_directory / "data_audit.json")
    roster = cast(list[str], source_config["participant_roster"])
    _require(
        data_audit["observable_candidate_count"] == 1939
        and data_audit["scored_window_count"] == 1213
        and data_audit["scored_participants"] == roster
        and data_audit["roster_participants_without_scored_rows"] == []
        and data_audit["observable_processing_annotation_independent"] is True
        and data_audit["scoring_labels_applied_posthoc"] is True,
        "data audit changed",
    )
    reference_receipt = _read_json(run_directory / "reference_control_receipt.json")
    reference_run = Path(str(reference_receipt["run_path"])).resolve()
    reference_cache, reference_predictions, replayed_receipt = _reference_arrays(
        reference_run, config
    )
    _require(
        replayed_receipt["record_sha256"] == reference_receipt["record_sha256"],
        "reference receipt replay changed",
    )
    reference_replay = _read_json(run_directory / "reference_control_replay.json")
    _require(
        reference_replay["reference_cell_id"] == "f1"
        and reference_replay["candidate_cell_id"] == "a"
        and reference_replay["observable_window_ids_exact"] is True
        and reference_replay["observable_probabilities_byte_exact"] is True
        and reference_replay["maximum_absolute_probability_difference"] == 0.0
        and reference_replay["fit_count_before_decision"] == 5
        and reference_replay["gsp_fits_started_before_decision"] is False
        and reference_replay["status"] == "passed",
        "reference replay gate record changed",
    )
    cache_metadata = _read_json(run_directory / "feature_cache_metadata.json")
    _require(
        sha256_file(run_directory / "feature_cache.npz") == cache_metadata["sha256"],
        "feature cache hash changed",
    )
    with np.load(run_directory / "feature_cache.npz", allow_pickle=False) as archive:
        cache = {key: archive[key] for key in archive.files}
    array_metadata = _mapping(cache_metadata["arrays"], "array metadata")
    _require(set(cache) == set(array_metadata), "cache array set changed")
    for name, values in cache.items():
        row = _mapping(array_metadata[name], name)
        _require(
            list(values.shape) == row["shape"]
            and str(values.dtype) == row["dtype"]
            and _array_sha256(values) == row["sha256"],
            f"cache array changed: {name}",
        )
    observable_windows = np.asarray(cache["observable_window_ids"], dtype=np.str_)
    participants = np.asarray(cache["observable_participant_ids"], dtype=np.str_)
    observable_fold = np.asarray(cache["observable_fold_index"], dtype=np.int64)
    scoring_indices = np.asarray(cache["scoring_indices"], dtype=np.int64)
    eligibility = np.asarray(cache["scoring_eligibility"], dtype=np.bool_)
    labels = np.asarray(cache["scored_labels"], dtype=np.int64)
    scored_participants = np.asarray(cache["scored_participant_ids"], dtype=np.str_)
    _require(
        observable_windows.size == participants.size == observable_fold.size == 1939
        and scoring_indices.size == labels.size == scored_participants.size == 1213,
        "cache alignment changed",
    )
    _require(
        np.array_equal(scoring_indices, np.flatnonzero(eligibility))
        and np.array_equal(cache["a_values"], reference_cache["six_values"])
        and np.array_equal(cache["a_names"], reference_cache["six_names"])
        and all(
            np.array_equal(cache[name], reference_cache[name])
            for name in (
                "observable_window_ids",
                "observable_participant_ids",
                "observable_fold_index",
                "scoring_indices",
                "scoring_eligibility",
                "scored_labels",
                "scored_participant_ids",
                "scored_window_ids",
                "scored_fold_index",
            )
        ),
        "candidate, fold, scoring or A-feature alignment differs from validated F1",
    )
    _require(
        cache["permutation_matrix"].dtype == np.dtype(np.int16)
        and cache["permutation_matrix"].shape == (1939, 128),
        "stored permutation dtype or shape changed",
    )
    expected_permutations = deterministic_joint_permutations(observable_windows)
    _require(
        np.array_equal(
            np.asarray(cache["permutation_matrix"], dtype=np.int64),
            expected_permutations,
        ),
        "stored permutation matrix changed",
    )
    permutation_audit = _read_json(run_directory / "permutation_audit.json")
    _require(
        _array_sha256(np.asarray(expected_permutations, dtype=np.int16))
        == permutation_audit["matrix_sha256"],
        "permutation matrix hash changed",
    )
    rows = cast(list[dict[str, Any]], permutation_audit["per_window"])
    _require(len(rows) == 1939, "per-window permutation hashes incomplete")
    for index, row in enumerate(rows):
        _require(
            row["window_id"] == observable_windows[index]
            and row["permutation_sha256"]
            == _array_sha256(np.asarray(expected_permutations[index], dtype=np.int16)),
            "per-window permutation record changed",
        )
    values_by_cell = {
        cell: np.asarray(cache[f"{cell}_values"], dtype=np.float64) for cell in CELL_ORDER
    }
    names_by_cell = {
        cell: tuple(np.asarray(cache[f"{cell}_names"], dtype=np.str_).tolist())
        for cell in CELL_ORDER
    }
    _require(
        values_by_cell["a"].shape == (1939, 80)
        and values_by_cell["b"].shape == values_by_cell["c"].shape == (1939, 1400)
        and names_by_cell["a"] == tuple(reference_cache["six_names"].tolist())
        and names_by_cell["b"] == names_by_cell["c"] == geometric_spectral_pyramid_feature_names(),
        "feature schema changed",
    )
    _require(
        all(bool(np.isfinite(values).all()) for values in values_by_cell.values())
        and not np.array_equal(values_by_cell["b"], values_by_cell["c"]),
        "feature matrix finiteness or order-ablation contrast changed",
    )
    with np.load(run_directory / "predictions.npz", allow_pickle=False) as archive:
        predictions = {key: archive[key] for key in archive.files}
    _require(predictions["cell_ids"].tolist() == list(CELL_ORDER), "cell order changed")
    stored_observable = np.asarray(predictions["observable_probabilities"], dtype=np.float64)
    stored_scored = np.asarray(predictions["scored_probabilities"], dtype=np.float64)
    _require(
        stored_observable.shape == (3, 1939, 3)
        and stored_scored.shape == (3, 1213, 3)
        and np.array_equal(predictions["observable_window_ids"], observable_windows)
        and np.array_equal(predictions["observable_participant_ids"], participants)
        and np.array_equal(predictions["observable_fold_index"], observable_fold)
        and np.array_equal(predictions["scoring_indices"], scoring_indices)
        and np.array_equal(predictions["scored_labels"], labels)
        and np.array_equal(predictions["scored_participant_ids"], scored_participants)
        and np.array_equal(predictions["scored_window_ids"], cache["scored_window_ids"])
        and np.array_equal(predictions["scored_fold_index"], cache["scored_fold_index"])
        and np.array_equal(stored_observable[:, scoring_indices], stored_scored),
        "prediction arrays changed",
    )
    _require(
        bool(np.isfinite(stored_observable).all())
        and bool(np.all(stored_observable >= 0.0))
        and np.allclose(stored_observable.sum(axis=2), 1.0, rtol=0.0, atol=1e-12),
        "stored probability values are invalid",
    )
    _require(
        np.array_equal(
            stored_observable[0],
            np.asarray(reference_predictions["observable_probabilities"][1], dtype=np.float64),
        ),
        "A no longer exactly replays reference F1",
    )
    fold_payload = _read_json(run_directory / "fold_reports.json")
    fold_rows = cast(list[dict[str, Any]], fold_payload["rows"])
    _require(
        len(fold_rows) == 15
        and fold_payload["fit_attempt_count"] == 15
        and fold_payload["completed_fit_count"] == 15,
        "fold record matrix changed",
    )
    row_lookup = {(str(row["cell_id"]), int(row["outer_fold"])): row for row in fold_rows}
    _require(len(row_lookup) == 15, "duplicate fold records")
    expected_attempts = [(cell, fold) for cell in CELL_ORDER for fold in range(5)]
    expected_attempt_files = {
        f"{number:02d}--{state}.json"
        for number in range(1, 16)
        for state in ("started", "completed")
    }
    actual_attempt_files = {path.name for path in (run_directory / "fit_attempts").glob("*.json")}
    _require(actual_attempt_files == expected_attempt_files, "fit-attempt ledger changed")
    for attempt_number, (cell, fold) in enumerate(expected_attempts, start=1):
        row = row_lookup[(cell, fold)]
        started_attempt = _read_json(
            run_directory / "fit_attempts" / f"{attempt_number:02d}--started.json"
        )
        completed_attempt = _read_json(
            run_directory / "fit_attempts" / f"{attempt_number:02d}--completed.json"
        )
        _verify_sealed(started_attempt, "started fit attempt")
        _verify_sealed(completed_attempt, "completed fit attempt")
        row_checkpoint = _mapping(row["checkpoint"], "fold checkpoint")
        _require(
            row["attempt_number"] == attempt_number
            and started_attempt["attempt_number"] == attempt_number
            and completed_attempt["attempt_number"] == attempt_number
            and started_attempt["cell_id"] == completed_attempt["cell_id"] == cell
            and started_attempt["outer_fold"] == completed_attempt["outer_fold"] == fold
            and started_attempt["registered_timing_benchmark"] is (attempt_number == 1)
            and completed_attempt["checkpoint"] == row_checkpoint,
            "fit-attempt order or completion record changed",
        )
    observable_labels = np.zeros(1939, dtype=np.int64)
    observable_labels[scoring_indices] = labels
    replay = np.full_like(stored_observable, np.nan)
    checkpoint_count = 0
    for cell_index, cell in enumerate(CELL_ORDER):
        for fold in range(5):
            row = row_lookup[(cell, fold)]
            checkpoint = _mapping(row["checkpoint"], "checkpoint")
            checkpoint_path = (run_directory / str(checkpoint["path"])).resolve()
            _require(_is_relative_to(checkpoint_path, run_directory), "checkpoint escapes run")
            _require(
                sha256_file(checkpoint_path) == checkpoint["sha256"]
                and checkpoint_path.stat().st_size == checkpoint["size_bytes"],
                "checkpoint hash or size changed",
            )
            metadata, estimator = _load_checkpoint(checkpoint_path)
            expected_attempt_number = cell_index * 5 + fold + 1
            _require(
                metadata["attempt_number"] == row["attempt_number"] == expected_attempt_number
                and metadata["cell_id"] == cell
                and metadata["outer_fold"] == fold
                and metadata["random_state"] == 11 + fold
                and metadata["registered_timing_benchmark"] is (expected_attempt_number == 1),
                "checkpoint metadata changed",
            )
            _require(
                tuple(metadata["feature_names"]) == names_by_cell[cell]
                and metadata["feature_names_sha256"]
                == canonical_json_sha256(list(names_by_cell[cell])),
                "feature names changed",
            )
            _require(
                estimator.get_params(deep=False) == _effective_estimator_parameters(11 + fold)
                and metadata["effective_parameters"] == _effective_estimator_parameters(11 + fold)
                and int(estimator.n_features_in_) == values_by_cell[cell].shape[1]
                and np.array_equal(estimator.classes_, np.arange(3)),
                "checkpoint estimator changed",
            )
            evaluation = observable_fold == fold
            training = (~evaluation) & eligibility
            _require(
                set(np.unique(participants[evaluation]).tolist()).isdisjoint(
                    np.unique(participants[training]).tolist()
                ),
                "checkpoint has participant leakage",
            )
            expected_weights = participant_first_weights(
                observable_labels[training], participants[training]
            )
            summary = _mapping(metadata["sample_weight_summary"], "weights")
            training_people = sorted(np.unique(participants[training]).tolist())
            evaluation_people = sorted(np.unique(participants[evaluation]).tolist())
            _require(
                metadata["training_participants"] == training_people
                and metadata["evaluation_participants"] == evaluation_people
                and metadata["training_row_count"] == int(training.sum())
                and metadata["evaluation_candidate_count"] == int(evaluation.sum())
                and row["training_scored_class_counts"]
                == np.bincount(observable_labels[training], minlength=3).tolist()
                and row["evaluation_scored_window_count"] == int(np.sum(evaluation & eligibility)),
                "checkpoint split accounting changed",
            )
            _require(
                summary["sha256"] == _array_sha256(expected_weights)
                and summary["minimum"] == float(expected_weights.min())
                and summary["maximum"] == float(expected_weights.max())
                and summary["mean"] == float(expected_weights.mean())
                and summary["sum"] == float(expected_weights.sum()),
                "weights changed",
            )
            with threadpool_limits(limits=1):
                replay[cell_index, evaluation] = _predict_proba_deterministically(
                    estimator, values_by_cell[cell][evaluation]
                )
            checkpoint_count += 1
    _require(np.array_equal(replay, stored_observable), "checkpoint replay differs")
    roster = cast(list[str], _mapping(config["source"], "source")["participant_roster"])
    reports = {
        cell: method_report(
            labels=labels,
            probabilities=replay[index, scoring_indices],
            participant_ids=scored_participants,
            roster=roster,
        )
        for index, cell in enumerate(CELL_ORDER)
    }
    reconstructed = analyse(
        reports,
        probabilities=replay[:, scoring_indices],
        labels=labels,
        participant_ids=scored_participants,
        config=config,
    )
    recorded = _read_json(run_directory / "analysis.json")
    without_hash = dict(recorded)
    without_hash.pop("record_sha256")
    _require(
        canonical_json_sha256({"record_kind": "fog_gsp_order_analysis", **reconstructed})
        == canonical_json_sha256(without_hash),
        "analysis/gate reconstruction changed",
    )
    _require(
        (run_directory / "OUTCOME_SUMMARY.md").read_text(encoding="utf-8")
        == _outcome_summary(recorded),
        "outcome summary changed",
    )
    result = _read_json(run_directory / "result.json")
    runtime = _read_json(run_directory / "runtime.json")
    _require(
        result["status"] == "complete_awaiting_independent_replay"
        and result["fit_attempt_count"] == result["completed_fit_count"] == 15
        and result["reference_control_exact_replay"] is True,
        "result completion fields changed",
    )
    _require(
        runtime["fit_attempt_count"] == runtime["completed_model_fit_count"] == 15
        and runtime["additional_benchmark_fit_count"] == 0
        and runtime["compute_cap_seconds"] == 7200
        and runtime["cap_exceeded"] is False
        and float(runtime["total_run_seconds"]) <= 7200.0,
        "runtime violates frozen limits",
    )
    validation = _sealed(
        {
            "record_kind": "fog_gsp_order_validation",
            "status": "validated",
            "validated_at_utc": _now(),
            "artifact_hashes_verified": len(artifacts),
            "checkpoint_predictions_replayed": checkpoint_count,
            "observable_probability_arrays_byte_exact": 3,
            "scored_probability_arrays_byte_exact": 3,
            "participant_reports_recomputed": 3,
            "comparisons_and_gates_recomputed": 3,
            "reference_control_probability_replay_exact": True,
            "permutation_matrix_reconstructed_exact": True,
            "per_window_permutation_hashes_verified": 1939,
            "source_downloaded_during_validation": False,
            "model_fits_during_validation": 0,
            "full_roster_retained": True,
            "target_or_other_external_data_loaded": False,
        }
    )
    _write_json_create_only(run_directory / "validation.json", validation)
    files = [
        path
        for path in run_directory.rglob("*")
        if path.is_file() and path.name != "completion_manifest.json"
    ]
    completion = _sealed(
        {
            "record_kind": "fog_gsp_order_completion_manifest",
            "status": "complete_and_independently_replayed",
            "artifacts": [
                {
                    "path": path.relative_to(run_directory).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(files)
            ],
            "validation_record_sha256": validation["record_sha256"],
            "practical_promotion_gate_status": _mapping(
                recorded["practical_promotion_gate"], "practical gate"
            )["status"],
            "order_mechanism_gate_status": _mapping(
                recorded["order_mechanism_gate"], "mechanism gate"
            )["status"],
            "decision": recorded["decision"],
            "task_owned_workers_and_monitors_stopped": True,
            "automatic_follow_on_launched": False,
        }
    )
    _write_json_create_only(run_directory / "completion_manifest.json", completion)
    return validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--repository-root", type=Path, required=True)
    run_parser.add_argument("--evidence-root", type=Path, required=True)
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--protocol", type=Path, required=True)
    run_parser.add_argument("--reference-run", type=Path, required=True)
    run_parser.add_argument("--output-directory", type=Path, required=True)
    run_parser.add_argument("--code-commit", required=True)
    run_parser.add_argument("--timeout-seconds", type=int, default=7200)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--run-directory", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "run":
        result = run_experiment(
            repository_root=arguments.repository_root,
            evidence_root=arguments.evidence_root,
            config_path=arguments.config,
            protocol_path=arguments.protocol,
            reference_run=arguments.reference_run,
            output_directory=arguments.output_directory,
            code_commit=str(arguments.code_commit),
            timeout_seconds=int(arguments.timeout_seconds),
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "practical_promotion_gate": result["practical_promotion_gate"],
                    "order_mechanism_gate": result["order_mechanism_gate"],
                    "decision": result["decision"],
                    "output_directory": str(arguments.output_directory),
                },
                indent=2,
            )
        )
    else:
        validation = validate_run(arguments.run_directory)
        print(json.dumps(validation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
