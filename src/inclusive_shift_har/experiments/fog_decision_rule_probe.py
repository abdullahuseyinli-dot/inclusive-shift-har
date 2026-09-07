"""Bounded corrected-FoG decision-rule probe.

The runner fits five exact outer F1 controls and twenty participant-nested inner
F1 models.  It selects one of nine fixed class multipliers per outer fold using
only inner-OOF rows.  D0 and D1 share raw probabilities; D1 classification
metrics consume explicit decoded labels.  Validation replays all checkpoints
without fitting or loading any raw dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import pickle
import threading
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

from inclusive_shift_har.experiments.fog_rf_feature_weight_factorial import (
    _array_sha256,
    _bottom,
    _confusion,
    _environment,
    _git_state,
    _mapping,
    _predict_proba_deterministically,
    _require,
    _scores_from_confusion,
    _sealed,
    _verify_sealed,
    _write_bytes_create_only,
    _write_json_create_only,
    paired_comparison,
    participant_first_weights,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
StringArray: TypeAlias = NDArray[np.str_]

EXPERIMENT_ID = "fog-decision-rule-probe-v1"
CLASS_NAMES = ("mobility", "sitting", "standing")
MULTIPLIERS = (0.5, 1.0, 2.0)
GRID = tuple((sitting, standing) for sitting in MULTIPLIERS for standing in MULTIPLIERS)
OUTER_FOLDS = tuple(range(5))
MAXIMUM_FIT_ATTEMPTS = 25
PINNED_PYTHON_PREFIX = "3.11.9 "
PINNED_NUMPY_VERSION = "2.3.5"
PINNED_SKLEARN_VERSION = "1.8.0"

GSP_REFERENCE_HASHES = {
    "feature_cache.npz": "6aabaae394bbf38e5ab84e738e9598c1d977e119d965f0f0eb17f7575aa7960a",
    "predictions.npz": "499219539c5105776c7dc396fbf87ee312a6ccf90a3f9b8a52bff8187876e489",
    "result.json": "b3fcd9f9399df66e01948ac48cee4b67f2b8d234af40f8586954ce5df2d24fc4",
    "validation.json": "f769d2c2c4d7f2ef2e30f8fd8a37de765b3918251265c9d1feca72ca2fbfdf13",
    "completion_manifest.json": (
        "10d844a7ebd77930f2bb19d6ed08fd90433205ff0303c8ccfc659655db417698"
    ),
}
FACTORIAL_REFERENCE_HASHES = {
    "feature_cache.npz": "a9d6cfb4c3d0ea75623b3152255775f09292c510d5797d1c0b58192db20e4bea",
    "predictions.npz": "c3c7ab375f20bf875a8ff871cf1f1dc7c6b7718d896f37f2d1f6440227ceac6b",
    "participant_metrics.json": (
        "417ee43e8b1ccf875805204b99a48322c5823939d086d053fc54704315e7b322"
    ),
    "result.json": "30b27aec3cdd7848d9a08d53ad5118d6e7d7fd85e4db49ed2f0194e2debd12f1",
    "validation.json": "1946240d9cb6a2f92777dc463c0516df30eeb10182eb8911dbe800d4df4355e5",
    "completion_manifest.json": (
        "e91e6c9fbe61256fd49bc54574ea85728b8c960afe4c366458b5af30a96fcce2"
    ),
}
REQUIRED_RUN_FILES = (
    "analysis.json",
    "artifact_manifest.json",
    "config_snapshot.yaml",
    "feature_cache.npz",
    "feature_cache_metadata.json",
    "fit_reports.json",
    "inner_predictions.npz",
    "OUTCOME_SUMMARY.md",
    "participant_metrics.json",
    "partition_preflight.json",
    "policy_freezes.json",
    "predictions.npz",
    "protocol_snapshot.json",
    "reference_receipts.json",
    "reference_replay.json",
    "result.json",
    "runtime.json",
    "source_manifest.json",
    "worker_shutdown.json",
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


def _read_npz(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _write_npz_create_only(path: Path, **arrays: NDArray[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]


def _expected_config() -> dict[str, Any]:
    roster = [f"fogstar:{index:03d}" for index in range(1, 23)]
    return {
        "schema_version": "1.0.0",
        "experiment_id": EXPERIMENT_ID,
        "status": "frozen_before_outcomes",
        "evidence_status": "corrected_fog_exploratory_development_not_confirmation",
        "source": {
            "dataset_id": "fog_star_v3",
            "expected_sha256": ("888875133fef386affddd0a5864f122d527d8d7bc588a69905cc0871bef53477"),
            "observable_candidate_count": 1939,
            "scored_window_count": 1213,
            "participant_roster": roster,
            "class_order": list(CLASS_NAMES),
            "scored_class_support": [954, 74, 185],
            "outer_fold_count": 5,
            "outer_fold_participant_counts": [5, 5, 4, 4, 4],
            "candidate_features": "six_channel_ordinary80",
            "annotation_role": "posthoc_window_label_and_scoring_eligibility_only",
        },
        "references": {
            "gsp_run": {
                "run_id": "fog-gsp-order-ablation-seed11-20260907-001",
                "feature_cache_sha256": GSP_REFERENCE_HASHES["feature_cache.npz"],
                "predictions_sha256": GSP_REFERENCE_HASHES["predictions.npz"],
                "result_sha256": GSP_REFERENCE_HASHES["result.json"],
                "validation_sha256": GSP_REFERENCE_HASHES["validation.json"],
                "completion_manifest_sha256": GSP_REFERENCE_HASHES["completion_manifest.json"],
                "feature_key": "a_values",
                "feature_name_key": "a_names",
                "probability_cell_id": "a",
                "probability_cell_index": 0,
            },
            "factorial_run": {
                "run_id": "fog-rf-feature-weight-factorial-seed11-20260907-001",
                "feature_cache_sha256": FACTORIAL_REFERENCE_HASHES["feature_cache.npz"],
                "predictions_sha256": FACTORIAL_REFERENCE_HASHES["predictions.npz"],
                "participant_metrics_sha256": FACTORIAL_REFERENCE_HASHES[
                    "participant_metrics.json"
                ],
                "result_sha256": FACTORIAL_REFERENCE_HASHES["result.json"],
                "validation_sha256": FACTORIAL_REFERENCE_HASHES["validation.json"],
                "completion_manifest_sha256": FACTORIAL_REFERENCE_HASHES[
                    "completion_manifest.json"
                ],
                "f1_cell_index": 1,
                "f3_cell_index": 3,
            },
        },
        "estimator": {
            "class": "sklearn.ensemble.RandomForestClassifier",
            "n_estimators": 500,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "bootstrap": True,
            "class_weight": None,
            "fit_workers": 4,
            "prediction_workers": 1,
            "outer_random_state": "11_plus_outer_fold_index",
            "inner_random_state": ("1100_plus_10_times_outer_fold_plus_inner_validation_fold"),
        },
        "weighting": {
            "policy": "participant_first_mean_one",
            "raw_sample_weight": "1 / (m_i * n_ic)",
            "training_eligible_rows_only": True,
        },
        "decision_rule": {
            "mobility_multiplier": 1.0,
            "sitting_multipliers": list(MULTIPLIERS),
            "standing_multipliers": list(MULTIPLIERS),
            "tie_order": list(CLASS_NAMES),
            "output_kind": "unnormalized_decision_scores_not_probabilities",
            "preserve_raw_probabilities": True,
        },
        "inner_selection": {
            "score": "equal_participant_fixed_three_class_macro_f1",
            "comparison": "candidate_minus_identity",
            "minimum_bottom_30_difference": -0.01,
            "minimum_worst_participant_difference": -0.03,
            "minimum_mobility_recall_difference": -0.01,
            "minimum_sitting_recall_difference": -0.02,
            "minimum_standing_recall_difference": -0.02,
            "tie_tolerance": 1e-12,
            "tie_break": [
                "minimum_absolute_log2_offset_l1",
                "lexicographic_log2_sitting_then_standing",
            ],
            "identity_if_no_nonidentity_gain_above_tolerance": True,
        },
        "final_gate": {
            "comparisons": ["d1_minus_d0", "d1_minus_f3"],
            "minimum_mean_gain": 0.015,
            "minimum_participant_wins": 14,
            "minimum_bottom_30_difference": -0.01,
            "minimum_worst_participant_difference": -0.03,
            "minimum_mobility_recall_difference": -0.01,
            "minimum_sitting_recall_difference": -0.02,
            "minimum_standing_recall_difference": -0.02,
            "leave_one_participant_positive_tolerance": 1e-12,
            "require_every_leave_one_fold_mean_positive": True,
        },
        "metrics": {
            "primary": "fixed_three_class_macro_f1_within_person_then_equal_person_mean",
            "missing_class_rule": "fixed_class_zero_division_zero",
            "bottom_fraction": 0.30,
            "bootstrap_resamples": 10_000,
            "bootstrap_seed": 1729,
            "probability_clip": 1e-12,
        },
        "resources": {
            "controller_count": 1,
            "maximum_fit_attempts": MAXIMUM_FIT_ATTEMPTS,
            "expected_outer_fits": 5,
            "expected_inner_fits": 20,
            "compute_wall_seconds": 3600,
            "implementation_and_checks_wall_seconds": 5400,
            "analysis_wall_seconds": 1800,
            "session_wall_seconds": 10800,
            "blas_threads": 1,
            "gpu_allowed": False,
            "additional_benchmark_fits_allowed": False,
            "additional_seeds_allowed": False,
            "other_datasets_allowed": False,
            "automatic_follow_on_allowed": False,
        },
        "runtime_contract": {
            "python_version_prefix": PINNED_PYTHON_PREFIX,
            "numpy_version": PINNED_NUMPY_VERSION,
            "scikit_learn_version": PINNED_SKLEARN_VERSION,
            "python_executable_must_match_gsp_reference": True,
            "effective_estimator_parameters_must_match_gsp_reference": True,
        },
        "claims": {
            "novelty": False,
            "confirmation": False,
            "calibration_improvement": False,
            "physiological_mechanism": False,
            "universal_temporal_failure": False,
        },
    }


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject any drift from the fully frozen contract."""

    _require(dict(config) == _expected_config(), "frozen decision-rule config changed")


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


def decode_with_multipliers(
    probabilities: FloatArray, *, sitting: float, standing: float
) -> tuple[FloatArray, IntArray, FloatArray]:
    """Decode fixed class scores and retain confidence of the emitted raw class."""

    p = np.asarray(probabilities, dtype=np.float64)
    _require(
        p.ndim == 2
        and p.shape[1] == 3
        and bool(np.isfinite(p).all())
        and bool(np.all(p >= 0))
        and np.allclose(p.sum(axis=1), 1.0, rtol=0.0, atol=1e-12),
        "decision probabilities are invalid",
    )
    _require(sitting in MULTIPLIERS and standing in MULTIPLIERS, "multiplier outside grid")
    multiplier = np.asarray([1.0, sitting, standing], dtype=np.float64)
    scores = p * multiplier[None, :]
    decisions = scores.argmax(axis=1).astype(np.int64)
    emitted_probability = p[np.arange(p.shape[0]), decisions]
    return scores, decisions, emitted_probability


def method_report_from_decisions(
    *,
    labels: IntArray,
    raw_probabilities: FloatArray,
    decisions: IntArray,
    participant_ids: StringArray,
    roster: Sequence[str],
) -> dict[str, Any]:
    """Report explicit hard labels while proper scores use unchanged probabilities."""

    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(raw_probabilities, dtype=np.float64)
    predicted = np.asarray(decisions, dtype=np.int64)
    people = np.asarray(participant_ids, dtype=np.str_)
    expected_roster = tuple(roster)
    _require(
        y.ndim == predicted.ndim == people.ndim == 1
        and y.size > 0
        and predicted.shape == people.shape == y.shape
        and p.shape == (y.size, 3),
        "explicit report arrays are not aligned",
    )
    _require(
        set(np.unique(y).tolist()).issubset({0, 1, 2})
        and set(np.unique(predicted).tolist()).issubset({0, 1, 2}),
        "report classes leave schema",
    )
    _require(
        set(np.unique(people).tolist()).issubset(expected_roster),
        "report participant leaves roster",
    )
    _require(
        bool(np.isfinite(p).all() and np.all(p >= 0))
        and np.allclose(p.sum(axis=1), 1.0, rtol=0.0, atol=1e-12),
        "raw report probabilities are invalid",
    )
    rows: list[dict[str, Any]] = []
    values: list[float] = []
    recalls: list[FloatArray] = []
    nll_values: list[float] = []
    brier_values: list[float] = []
    present_f1_values: list[float] = []
    all_class_count = 0
    for participant in expected_roster:
        selected = people == participant
        if not selected.any():
            rows.append(
                {
                    "participant_id": participant,
                    "eligible": False,
                    "window_count": 0,
                    "class_support": [0, 0, 0],
                    "macro_f1": None,
                    "class_recall": None,
                    "nll": None,
                    "multiclass_brier": None,
                    "present_class_macro_f1": None,
                    "mean_emitted_label_probability": None,
                }
            )
            continue
        confusion = _confusion(y[selected], predicted[selected])
        precision, recall, class_f1 = _scores_from_confusion(confusion)
        selected_labels = y[selected]
        selected_probabilities = p[selected]
        nll = float(
            -np.log(
                np.clip(selected_probabilities, 1e-12, 1.0)[
                    np.arange(selected_labels.size), selected_labels
                ]
            ).mean()
        )
        one_hot = np.eye(3, dtype=np.float64)[selected_labels]
        brier = float(np.sum((selected_probabilities - one_hot) ** 2, axis=1).mean())
        value = float(class_f1.mean())
        present = confusion.sum(axis=1) > 0
        present_f1 = float(class_f1[present].mean())
        emitted = selected_probabilities[np.arange(int(selected.sum())), predicted[selected]]
        all_class_count += int(present.all())
        values.append(value)
        recalls.append(recall)
        nll_values.append(nll)
        brier_values.append(brier)
        present_f1_values.append(present_f1)
        rows.append(
            {
                "participant_id": participant,
                "eligible": True,
                "window_count": int(selected.sum()),
                "class_support": confusion.sum(axis=1).tolist(),
                "confusion_matrix": confusion.tolist(),
                "class_precision": precision.tolist(),
                "class_recall": recall.tolist(),
                "class_f1": class_f1.tolist(),
                "macro_f1": value,
                "present_class_macro_f1": present_f1,
                "nll": nll,
                "multiclass_brier": brier,
                "mean_emitted_label_probability": float(emitted.mean()),
            }
        )
    pooled = _confusion(y, predicted)
    pooled_precision, pooled_recall, pooled_f1 = _scores_from_confusion(pooled)
    clipped = np.clip(p, 1e-12, 1.0)
    emitted = p[np.arange(y.size), predicted]
    missing = [str(row["participant_id"]) for row in rows if not bool(row["eligible"])]
    complete = not missing and len(values) == len(expected_roster)
    output: dict[str, Any] = {
        "status": (
            "complete_full_roster" if complete else "incomplete_zero_scored_roster_participant"
        ),
        "roster_count": len(expected_roster),
        "scored_participant_count": len(values),
        "participants_without_scored_rows": missing,
        "sample_count": int(y.size),
        "class_names": list(CLASS_NAMES),
        "participants": rows,
        "pooled": {
            "confusion_matrix": pooled.tolist(),
            "class_precision": dict(zip(CLASS_NAMES, pooled_precision.tolist(), strict=True)),
            "class_recall": dict(zip(CLASS_NAMES, pooled_recall.tolist(), strict=True)),
            "class_f1": dict(zip(CLASS_NAMES, pooled_f1.tolist(), strict=True)),
            "macro_f1": float(pooled_f1.mean()),
            "accuracy": float(np.mean(predicted == y)),
            "nll": float(-np.log(clipped[np.arange(y.size), y]).mean()),
            "multiclass_brier": float(
                np.mean(np.sum((p - np.eye(3, dtype=np.float64)[y]) ** 2, axis=1))
            ),
            "mean_emitted_label_probability": float(emitted.mean()),
        },
        "fixed_class_rule": "zero_division_zero",
        "hard_label_source": "explicit_decisions_not_probability_argmax",
        "proper_score_source": "unchanged_raw_probabilities",
    }
    if complete:
        participant_values = np.asarray(values, dtype=np.float64)
        participant_recalls = np.stack(recalls)
        output["primary"] = {
            "mean_participant_macro_f1": float(participant_values.mean()),
            "median_participant_macro_f1": float(np.median(participant_values)),
            "quartiles_participant_macro_f1": [
                float(np.quantile(participant_values, 0.25)),
                float(np.quantile(participant_values, 0.75)),
            ],
            "bottom_30_percent_participant_macro_f1": _bottom(participant_values),
            "bottom_30_participant_count": math.ceil(0.30 * participant_values.size),
            "worst_participant_macro_f1": float(participant_values.min()),
            "participant_macro_class_recall": dict(
                zip(CLASS_NAMES, participant_recalls.mean(axis=0).tolist(), strict=True)
            ),
            "mean_participant_nll": float(np.mean(nll_values)),
            "mean_participant_multiclass_brier": float(np.mean(brier_values)),
            "mean_present_class_participant_macro_f1": float(np.mean(present_f1_values)),
            "participants_with_all_three_classes": all_class_count,
        }
    else:
        output["primary"] = None
    return output


def _participant_scores(report: Mapping[str, Any]) -> FloatArray:
    rows = cast(list[dict[str, Any]], report["participants"])
    _require(all(row.get("eligible") is True for row in rows), "report roster is incomplete")
    return np.asarray([float(row["macro_f1"]) for row in rows], dtype=np.float64)


def _assert_report_alignment(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    left_rows = cast(list[dict[str, Any]], left["participants"])
    right_rows = cast(list[dict[str, Any]], right["participants"])
    _require(
        [row["participant_id"] for row in left_rows]
        == [row["participant_id"] for row in right_rows],
        "participant report ordering differs",
    )


def _selection_difference(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    candidate_primary = _mapping(candidate["primary"], "candidate primary")
    baseline_primary = _mapping(baseline["primary"], "baseline primary")
    candidate_recall = _mapping(
        candidate_primary["participant_macro_class_recall"], "candidate recall"
    )
    baseline_recall = _mapping(
        baseline_primary["participant_macro_class_recall"], "baseline recall"
    )
    candidate_scores = _participant_scores(candidate)
    baseline_scores = _participant_scores(baseline)
    differences = candidate_scores - baseline_scores
    return {
        "mean_difference": float(differences.mean()),
        "bottom_30_percent_difference": float(
            candidate_primary["bottom_30_percent_participant_macro_f1"]
            - baseline_primary["bottom_30_percent_participant_macro_f1"]
        ),
        "worst_participant_difference": float(
            candidate_primary["worst_participant_macro_f1"]
            - baseline_primary["worst_participant_macro_f1"]
        ),
        "class_recall_differences": {
            name: float(candidate_recall[name] - baseline_recall[name]) for name in CLASS_NAMES
        },
        "participant_differences": differences.tolist(),
        "participant_wins": int(np.sum(differences > 1e-12)),
        "participant_harms": int(np.sum(differences < -1e-12)),
        "participant_ties": int(np.sum(np.abs(differences) <= 1e-12)),
    }


def select_policy(
    probabilities: FloatArray,
    labels: IntArray,
    participant_ids: StringArray,
    roster: Sequence[str],
) -> dict[str, Any]:
    """Select only from supplied inner-OOF arrays; no outer state is accepted."""

    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    people = np.asarray(participant_ids, dtype=np.str_)
    _require(p.shape == (y.size, 3) and people.shape == y.shape, "selector arrays misalign")
    _require(set(np.unique(people).tolist()) == set(roster), "selector roster mismatch")
    _, identity_labels, _ = decode_with_multipliers(p, sitting=1.0, standing=1.0)
    baseline = method_report_from_decisions(
        labels=y,
        raw_probabilities=p,
        decisions=identity_labels,
        participant_ids=people,
        roster=roster,
    )
    baseline_mean = float(
        _mapping(baseline["primary"], "baseline primary")["mean_participant_macro_f1"]
    )
    rows: list[dict[str, Any]] = []
    for sitting, standing in GRID:
        _, decisions, _ = decode_with_multipliers(p, sitting=sitting, standing=standing)
        report = method_report_from_decisions(
            labels=y,
            raw_probabilities=p,
            decisions=decisions,
            participant_ids=people,
            roster=roster,
        )
        difference = _selection_difference(report, baseline)
        recalls = _mapping(difference["class_recall_differences"], "inner recall differences")
        checks = {
            "bottom_30": float(difference["bottom_30_percent_difference"]) >= -0.01,
            "worst_participant": float(difference["worst_participant_difference"]) >= -0.03,
            "mobility_recall": float(recalls["mobility"]) >= -0.01,
            "sitting_recall": float(recalls["sitting"]) >= -0.02,
            "standing_recall": float(recalls["standing"]) >= -0.02,
        }
        if sitting == standing == 1.0:
            _require(all(checks.values()), "identity must remain feasible")
        primary = _mapping(report["primary"], "candidate primary")
        sitting_exponent = round(math.log2(sitting))
        standing_exponent = round(math.log2(standing))
        rows.append(
            {
                "rule_id": f"sitting_{sitting:g}__standing_{standing:g}",
                "multipliers": [1.0, sitting, standing],
                "log2_posture_offsets": [sitting_exponent, standing_exponent],
                "offset_l1": abs(sitting_exponent) + abs(standing_exponent),
                "mean_participant_macro_f1": float(primary["mean_participant_macro_f1"]),
                "bottom_30_percent_participant_macro_f1": float(
                    primary["bottom_30_percent_participant_macro_f1"]
                ),
                "worst_participant_macro_f1": float(primary["worst_participant_macro_f1"]),
                "participant_macro_class_recall": primary["participant_macro_class_recall"],
                "participant_macro_f1": _participant_scores(report).tolist(),
                "difference_from_identity": difference,
                "feasibility_checks": checks,
                "feasible": all(checks.values()),
                "changed_decisions_from_identity": int(np.sum(decisions != identity_labels)),
            }
        )
    feasible = [row for row in rows if bool(row["feasible"])]
    _require(bool(feasible), "identity feasibility was lost")
    tolerance = 1e-12
    improving_nonidentity = [
        row
        for row in feasible
        if row["multipliers"] != [1.0, 1.0, 1.0]
        and float(row["mean_participant_macro_f1"]) > baseline_mean + tolerance
    ]
    if not improving_nonidentity:
        selected = next(row for row in rows if row["multipliers"] == [1.0, 1.0, 1.0])
        reason = "identity_no_feasible_nonidentity_gain_above_tolerance"
    else:
        best = max(float(row["mean_participant_macro_f1"]) for row in feasible)
        tied: list[dict[str, Any]] = [
            row for row in feasible if float(row["mean_participant_macro_f1"]) >= best - tolerance
        ]
        selected = min(
            tied,
            key=lambda row: (
                int(row["offset_l1"]),
                int(cast(list[int], row["log2_posture_offsets"])[0]),
                int(cast(list[int], row["log2_posture_offsets"])[1]),
            ),
        )
        reason = "maximum_feasible_mean_then_fixed_tie_break"
    return {
        "selector_kind": "inner_oof_only_fixed_nine_rule",
        "sample_count": int(y.size),
        "participant_roster": list(roster),
        "participant_count": len(roster),
        "bottom_30_participant_count": math.ceil(0.30 * len(roster)),
        "candidate_count": len(rows),
        "identity_mean_participant_macro_f1": baseline_mean,
        "candidates": rows,
        "selected_rule_id": selected["rule_id"],
        "selected_multipliers": selected["multipliers"],
        "selected_mean_participant_macro_f1": selected["mean_participant_macro_f1"],
        "selection_reason": reason,
        "tie_tolerance": tolerance,
    }


def _weight_summary(
    labels: IntArray, participants: StringArray, training: BoolArray
) -> dict[str, Any]:
    weights = participant_first_weights(labels[training], participants[training])
    training_people = np.unique(participants[training])
    totals = np.asarray(
        [weights[participants[training] == person].sum() for person in training_people],
        dtype=np.float64,
    )
    _require(float(np.max(totals) - np.min(totals)) <= 1e-10, "participant weight totals differ")
    return {
        "minimum": float(weights.min()),
        "maximum": float(weights.max()),
        "mean": float(weights.mean()),
        "sum": float(weights.sum()),
        "sha256": _array_sha256(weights),
        "participant_total_minimum": float(totals.min()),
        "participant_total_maximum": float(totals.max()),
    }


def partition_preflight(
    *, labels: IntArray, participants: StringArray, folds: IntArray, eligibility: BoolArray
) -> dict[str, Any]:
    """Audit all 25 partitions before fitting."""

    _require(
        labels.shape == participants.shape == folds.shape == eligibility.shape,
        "preflight arrays misalign",
    )
    rows: list[dict[str, Any]] = []
    for outer in OUTER_FOLDS:
        training = (folds != outer) & eligibility
        evaluation = folds == outer
        rows.append(
            _partition_row(
                role="outer",
                outer_fold=outer,
                inner_validation_fold=None,
                labels=labels,
                participants=participants,
                eligibility=eligibility,
                training=training,
                evaluation=evaluation,
            )
        )
    for outer in OUTER_FOLDS:
        for inner in OUTER_FOLDS:
            if inner == outer:
                continue
            training = (folds != outer) & (folds != inner) & eligibility
            evaluation = folds == inner
            rows.append(
                _partition_row(
                    role="inner",
                    outer_fold=outer,
                    inner_validation_fold=inner,
                    labels=labels,
                    participants=participants,
                    eligibility=eligibility,
                    training=training,
                    evaluation=evaluation,
                )
            )
    _require(len(rows) == MAXIMUM_FIT_ATTEMPTS, "preflight partition count changed")
    return {
        "record_kind": "fog_decision_rule_partition_preflight",
        "created_before_fits": True,
        "partition_count": len(rows),
        "outer_partition_count": 5,
        "inner_partition_count": 20,
        "all_training_partitions_have_three_classes": True,
        "all_training_evaluation_participants_disjoint": True,
        "rows": rows,
    }


def _partition_row(
    *,
    role: str,
    outer_fold: int,
    inner_validation_fold: int | None,
    labels: IntArray,
    participants: StringArray,
    eligibility: BoolArray,
    training: BoolArray,
    evaluation: BoolArray,
) -> dict[str, Any]:
    training_people = sorted(np.unique(participants[training]).tolist())
    evaluation_people = sorted(np.unique(participants[evaluation]).tolist())
    _require(set(training_people).isdisjoint(evaluation_people), "participant leakage")
    class_counts = np.bincount(labels[training], minlength=3)
    _require(bool(np.all(class_counts > 0)), "training partition lacks a global class")
    training_participant_class_counts = [
        {
            "participant_id": participant,
            "class_counts": np.bincount(
                labels[training & (participants == participant)], minlength=3
            ).tolist(),
        }
        for participant in training_people
    ]
    evaluation_participant_class_counts = [
        {
            "participant_id": participant,
            "class_counts": np.bincount(
                labels[evaluation & eligibility & (participants == participant)], minlength=3
            ).tolist(),
        }
        for participant in evaluation_people
    ]
    return {
        "role": role,
        "outer_fold": outer_fold,
        "inner_validation_fold": inner_validation_fold,
        "training_participants": training_people,
        "evaluation_participants": evaluation_people,
        "training_scored_row_count": int(training.sum()),
        "evaluation_observable_candidate_count": int(evaluation.sum()),
        "evaluation_scored_row_count": int(np.sum(evaluation & eligibility)),
        "training_class_counts": class_counts.tolist(),
        "training_participant_class_counts": training_participant_class_counts,
        "evaluation_scored_participant_class_counts": evaluation_participant_class_counts,
        "training_mask_sha256": _array_sha256(training),
        "evaluation_mask_sha256": _array_sha256(evaluation),
        "sample_weight_summary": _weight_summary(labels, participants, training),
    }


def _verify_reference(
    run_directory: Path,
    *,
    expected_run_id: str,
    hashes: Mapping[str, str],
) -> tuple[dict[str, NDArray[Any]], dict[str, NDArray[Any]], dict[str, Any]]:
    _require(run_directory.name == expected_run_id, "reference run ID changed")
    checked: list[dict[str, Any]] = []
    for name, expected_hash in hashes.items():
        path = run_directory / name
        _require(
            path.is_file() and sha256_file(path) == expected_hash, f"reference changed: {name}"
        )
        checked.append({"path": name, "sha256": expected_hash, "size_bytes": path.stat().st_size})
    completion = _read_json(run_directory / "completion_manifest.json")
    validation = _read_json(run_directory / "validation.json")
    _verify_sealed(completion, "reference completion")
    _verify_sealed(validation, "reference validation")
    _require(
        completion.get("status") == "complete_and_independently_replayed"
        and validation.get("status") == "validated",
        "reference is not independently validated",
    )
    entries = cast(list[dict[str, Any]], completion["artifacts"])
    for entry in entries:
        path = (run_directory / str(entry["path"])).resolve()
        _require(_is_relative_to(path, run_directory), "reference manifest escapes run")
        _require(
            path.is_file()
            and sha256_file(path) == entry["sha256"]
            and path.stat().st_size == entry["size_bytes"],
            "reference manifest entry changed",
        )
    cache = _read_npz(run_directory / "feature_cache.npz")
    predictions = _read_npz(run_directory / "predictions.npz")
    receipt = {
        "run_path": str(run_directory),
        "run_id": expected_run_id,
        "checked_files": checked,
        "completion_manifest_entries_verified": len(entries),
        "completion_status": completion["status"],
        "validation_status": validation["status"],
    }
    return cache, predictions, receipt


def _validate_gsp_cache_arrays(run_directory: Path, cache: Mapping[str, NDArray[Any]]) -> int:
    metadata = _read_json(run_directory / "feature_cache_metadata.json")
    arrays = _mapping(metadata["arrays"], "GSP cache arrays")
    _require(set(arrays) == set(cache), "GSP cache array set changed")
    for name, values in cache.items():
        row = _mapping(arrays[name], name)
        _require(
            row["shape"] == list(values.shape)
            and row["dtype"] == str(values.dtype)
            and row["sha256"] == _array_sha256(values),
            f"GSP cache metadata mismatch: {name}",
        )
    return len(arrays)


def _load_references(
    gsp_run: Path, factorial_run: Path, config: Mapping[str, Any]
) -> tuple[dict[str, NDArray[Any]], dict[str, NDArray[Any]], dict[str, Any]]:
    references = _mapping(config["references"], "references")
    gsp_config = _mapping(references["gsp_run"], "GSP reference")
    factorial_config = _mapping(references["factorial_run"], "factorial reference")
    gsp_cache, gsp_predictions, gsp_receipt = _verify_reference(
        gsp_run,
        expected_run_id=str(gsp_config["run_id"]),
        hashes=GSP_REFERENCE_HASHES,
    )
    factorial_cache, factorial_predictions, factorial_receipt = _verify_reference(
        factorial_run,
        expected_run_id=str(factorial_config["run_id"]),
        hashes=FACTORIAL_REFERENCE_HASHES,
    )
    gsp_receipt["cache_arrays_verified"] = _validate_gsp_cache_arrays(gsp_run, gsp_cache)
    alignment_keys = (
        "observable_window_ids",
        "observable_participant_ids",
        "observable_fold_index",
        "scoring_indices",
        "scored_labels",
        "scored_participant_ids",
        "scored_window_ids",
        "scored_fold_index",
    )
    for key in alignment_keys:
        _require(
            np.array_equal(gsp_predictions[key], factorial_predictions[key]),
            f"reference prediction alignment differs: {key}",
        )
    cache_alignment_keys = (*alignment_keys, "scoring_eligibility")
    for key in cache_alignment_keys:
        _require(
            np.array_equal(gsp_cache[key], factorial_cache[key]),
            f"reference cache alignment differs: {key}",
        )
    _require(
        gsp_predictions["cell_ids"].tolist()[0] == "a"
        and factorial_predictions["cell_ids"].tolist()[1] == "f1"
        and factorial_predictions["cell_ids"].tolist()[3] == "f3",
        "reference cell order changed",
    )
    _require(
        np.array_equal(gsp_cache["a_values"], factorial_cache["six_values"])
        and np.array_equal(gsp_cache["a_names"], factorial_cache["six_names"])
        and np.array_equal(
            gsp_predictions["observable_probabilities"][0],
            factorial_predictions["observable_probabilities"][1],
        ),
        "A/F1 references no longer agree exactly",
    )
    arrays = {
        "feature_values": np.asarray(gsp_cache["a_values"], dtype=np.float64),
        "feature_names": np.asarray(gsp_cache["a_names"], dtype=np.str_),
        "observable_window_ids": np.asarray(gsp_cache["observable_window_ids"], dtype=np.str_),
        "observable_participant_ids": np.asarray(
            gsp_cache["observable_participant_ids"], dtype=np.str_
        ),
        "observable_fold_index": np.asarray(gsp_cache["observable_fold_index"], dtype=np.int64),
        "scoring_indices": np.asarray(gsp_cache["scoring_indices"], dtype=np.int64),
        "scoring_eligibility": np.asarray(gsp_cache["scoring_eligibility"], dtype=np.bool_),
        "scored_labels": np.asarray(gsp_cache["scored_labels"], dtype=np.int64),
        "scored_participant_ids": np.asarray(gsp_cache["scored_participant_ids"], dtype=np.str_),
        "scored_window_ids": np.asarray(gsp_cache["scored_window_ids"], dtype=np.str_),
        "scored_fold_index": np.asarray(gsp_cache["scored_fold_index"], dtype=np.int64),
    }
    controls = {
        "gsp_a_observable_probabilities": np.asarray(
            gsp_predictions["observable_probabilities"][0], dtype=np.float64
        ),
        "factorial_f1_observable_probabilities": np.asarray(
            factorial_predictions["observable_probabilities"][1], dtype=np.float64
        ),
        "f3_observable_probabilities": np.asarray(
            factorial_predictions["observable_probabilities"][3], dtype=np.float64
        ),
    }
    receipt = _sealed(
        {
            "record_kind": "fog_decision_rule_reference_receipts",
            "gsp": gsp_receipt,
            "factorial": factorial_receipt,
            "candidate_feature_matrix_exact_across_references": True,
            "f1_probabilities_exact_across_references": True,
            "alignment_keys_exact": list(alignment_keys),
        }
    )
    return arrays, controls, receipt


def _validate_runtime_contract(gsp_run: Path) -> dict[str, Any]:
    """Bind dependency versions and RF defaults to the sealed GSP execution."""

    reference_preflight = _read_json(gsp_run / "preflight.json")
    _verify_sealed(reference_preflight, "GSP reference preflight")
    reference_environment = _mapping(reference_preflight["environment"], "GSP environment")
    current_environment = _environment()
    _require(
        str(reference_environment["python"]).startswith(PINNED_PYTHON_PREFIX)
        and reference_environment["numpy"] == PINNED_NUMPY_VERSION
        and reference_environment["scikit_learn"] == PINNED_SKLEARN_VERSION,
        "GSP reference environment does not match the frozen runtime contract",
    )
    _require(
        str(current_environment["python"]).startswith(PINNED_PYTHON_PREFIX)
        and current_environment["numpy"] == PINNED_NUMPY_VERSION
        and current_environment["scikit_learn"] == PINNED_SKLEARN_VERSION
        and current_environment["python"] == reference_environment["python"]
        and Path(str(current_environment["python_executable"])).resolve()
        == Path(str(reference_environment["python_executable"])).resolve(),
        "current interpreter does not match the frozen validated GSP environment",
    )
    parameter_sets = _mapping(
        reference_preflight["effective_estimator_parameter_sets"],
        "GSP effective estimator parameters",
    )
    for outer in OUTER_FOLDS:
        reference_row = _mapping(parameter_sets[f"a--fold-{outer}"], "GSP A parameter row")
        _require(
            reference_row["cell_id"] == "a"
            and reference_row["outer_fold"] == outer
            and reference_row["feature_dimension"] == 80
            and reference_row["weighting"] == "participant_first_mean_one"
            and reference_row["parameters"] == _effective_estimator_parameters(11 + outer),
            "effective Random Forest parameters differ from sealed GSP A",
        )
    return current_environment


def _worker_shutdown_receipt(completed_fits: int) -> dict[str, Any]:
    """Record observable Python worker state after every registered fit returns."""

    active_children = [child for child in multiprocessing.active_children() if child.is_alive()]
    main_thread = threading.main_thread()
    active_threads = [
        thread
        for thread in threading.enumerate()
        if thread is not main_thread and thread.is_alive()
    ]
    _require(completed_fits == MAXIMUM_FIT_ATTEMPTS, "shutdown checked before fit completion")
    _require(not active_children, "active multiprocessing fit worker remains")
    _require(not active_threads, "active Python fit worker thread remains")
    return _sealed(
        {
            "record_kind": "fog_decision_rule_worker_shutdown",
            "observed_at_utc": _now(),
            "completed_fit_count": completed_fits,
            "active_multiprocessing_child_count": 0,
            "active_non_main_python_thread_count": 0,
            "active_multiprocessing_children": [],
            "active_non_main_python_threads": [],
            "task_owned_fit_workers_and_monitors_stopped": True,
        }
    )


def _fit_model(
    *,
    feature_values: FloatArray,
    labels: IntArray,
    participants: StringArray,
    training: BoolArray,
    evaluation: BoolArray,
    role: str,
    outer_fold: int,
    inner_validation_fold: int | None,
    attempt_number: int,
    output_directory: Path,
    feature_names: tuple[str, ...],
    deadline: float,
) -> tuple[FloatArray, dict[str, Any]]:
    _require(1 <= attempt_number <= MAXIMUM_FIT_ATTEMPTS, "fit-attempt budget exhausted")
    if time.perf_counter() >= deadline:
        raise TimeoutError("compute cap reached before next registered fit")
    random_state = (
        11 + outer_fold
        if role == "outer"
        else 1100 + 10 * outer_fold + cast(int, inner_validation_fold)
    )
    benchmark = role == "outer" and outer_fold == 0
    sample_weight = participant_first_weights(labels[training], participants[training])
    started_record = _sealed(
        {
            "record_kind": "fog_decision_rule_fit_attempt_started",
            "attempt_number": attempt_number,
            "role": role,
            "outer_fold": outer_fold,
            "inner_validation_fold": inner_validation_fold,
            "registered_timing_benchmark": benchmark,
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
        estimator.fit(feature_values[training], labels[training], sample_weight=sample_weight)
        fit_seconds = time.perf_counter() - fit_started
        prediction_started = time.perf_counter()
        probabilities = _predict_proba_deterministically(estimator, feature_values[evaluation])
        prediction_seconds = time.perf_counter() - prediction_started
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
        "role": role,
        "outer_fold": outer_fold,
        "inner_validation_fold": inner_validation_fold,
        "random_state": random_state,
        "registered_timing_benchmark": benchmark,
        "feature_names": list(feature_names),
        "feature_names_sha256": canonical_json_sha256(list(feature_names)),
        "training_participants": sorted(np.unique(participants[training]).tolist()),
        "training_row_count": int(training.sum()),
        "training_class_counts": np.bincount(labels[training], minlength=3).tolist(),
        "evaluation_participants": sorted(np.unique(participants[evaluation]).tolist()),
        "evaluation_candidate_count": int(evaluation.sum()),
        "class_weight": None,
        "sample_weight_policy": "participant_first_mean_one",
        "sample_weight_summary": _weight_summary(labels, participants, training),
        "effective_parameters": estimator.get_params(deep=False),
        "prediction_worker_count": 1,
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "fit_and_predict_seconds": elapsed,
    }
    checkpoint_name = (
        f"outer--fold-{outer_fold}.pkl"
        if role == "outer"
        else f"inner--outer-{outer_fold}--validation-{inner_validation_fold}.pkl"
    )
    checkpoint_path = output_directory / "checkpoints" / checkpoint_name
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
            "record_kind": "fog_decision_rule_fit_attempt_completed",
            "attempt_number": attempt_number,
            "role": role,
            "outer_fold": outer_fold,
            "inner_validation_fold": inner_validation_fold,
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


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], RandomForestClassifier]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    _require(isinstance(payload, dict), "checkpoint payload is not a mapping")
    metadata = _mapping(payload.get("metadata"), "checkpoint metadata")
    estimator = payload.get("estimator")
    _require(isinstance(estimator, RandomForestClassifier), "checkpoint estimator type changed")
    return metadata, estimator


def _decision_events(
    labels: IntArray,
    old_decisions: IntArray,
    new_decisions: IntArray,
    participant_ids: StringArray,
    roster: Sequence[str],
) -> dict[str, Any]:
    old = np.asarray(old_decisions, dtype=np.int64)
    new = np.asarray(new_decisions, dtype=np.int64)
    rescued = (old != labels) & (new == labels)
    harmed = (old == labels) & (new != labels)
    changed = old != new
    both_wrong = (old != labels) & (new != labels)
    return {
        "changed_decisions": int(changed.sum()),
        "rescues": int(rescued.sum()),
        "harms": int(harmed.sum()),
        "wrong_to_different_wrong": int(np.sum(changed & both_wrong)),
        "both_wrong": int(both_wrong.sum()),
        "by_true_class": [
            {
                "class_name": CLASS_NAMES[label],
                "rescues": int(np.sum(rescued & (labels == label))),
                "harms": int(np.sum(harmed & (labels == label))),
                "changed": int(np.sum(changed & (labels == label))),
            }
            for label in range(3)
        ],
        "participants": [
            {
                "participant_id": participant,
                "rescues": int(np.sum(rescued & (participant_ids == participant))),
                "harms": int(np.sum(harmed & (participant_ids == participant))),
                "changed": int(np.sum(changed & (participant_ids == participant))),
            }
            for participant in roster
        ],
    }


def _leave_one_fold(
    left: Mapping[str, Any], right: Mapping[str, Any], participant_folds: IntArray
) -> list[dict[str, Any]]:
    _assert_report_alignment(left, right)
    differences = _participant_scores(left) - _participant_scores(right)
    _require(participant_folds.shape == differences.shape, "participant fold vector misaligns")
    return [
        {
            "omitted_outer_fold": fold,
            "remaining_participant_count": int(np.sum(participant_folds != fold)),
            "mean_participant_difference": float(differences[participant_folds != fold].mean()),
        }
        for fold in OUTER_FOLDS
    ]


def _gate(
    comparison: Mapping[str, Any], leave_fold: Sequence[Mapping[str, Any]], *, replay_exact: bool
) -> dict[str, Any]:
    recall = _mapping(comparison["class_recall_differences"], "class recall differences")
    checks = {
        "full_22_person_roster_scored": True,
        "fresh_control_exact_replay": replay_exact,
        "mean_gain": float(comparison["mean_difference"]) >= 0.015,
        "participant_wins": int(comparison["participant_wins"]) >= 14,
        "bottom_30": float(comparison["bottom_30_percent_difference"]) >= -0.01,
        "worst_participant": float(comparison["worst_participant_difference"]) >= -0.03,
        "mobility_recall": float(recall["mobility"]) >= -0.01,
        "sitting_recall": float(recall["sitting"]) >= -0.02,
        "standing_recall": float(recall["standing"]) >= -0.02,
        "all_leave_one_participant_out_positive": all(
            float(value) > 1e-12
            for value in cast(list[float], comparison["leave_one_participant_out_mean_differences"])
        ),
        "all_leave_one_fold_means_positive": all(
            float(row["mean_participant_difference"]) > 0 for row in leave_fold
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "passed_check_count": sum(checks.values()),
        "required_check_count": len(checks),
    }


def analyse(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    labels: IntArray,
    decisions: Mapping[str, IntArray],
    participant_ids: StringArray,
    participant_folds: IntArray,
    roster: Sequence[str],
    control_replay_exact: bool,
) -> dict[str, Any]:
    """Recompute both comparisons, fold sensitivity and prospective gates."""

    _require(set(reports) == {"d0", "d1", "f3"}, "report set changed")
    for reference in ("d0", "f3"):
        _assert_report_alignment(reports["d1"], reports[reference])
    comparisons = {
        "d1_minus_d0": paired_comparison(reports["d1"], reports["d0"]),
        "d1_minus_f3": paired_comparison(reports["d1"], reports["f3"]),
    }
    leave_fold = {
        "d1_minus_d0": _leave_one_fold(reports["d1"], reports["d0"], participant_folds),
        "d1_minus_f3": _leave_one_fold(reports["d1"], reports["f3"], participant_folds),
    }
    gates = {
        name: _gate(comparisons[name], leave_fold[name], replay_exact=control_replay_exact)
        for name in comparisons
    }
    if all(gate["status"] == "pass" for gate in gates.values()):
        decision = "pass_both_propose_separately_authorized_frozen_policy_seed_replication"
    elif gates["d1_minus_d0"]["status"] == "pass":
        decision = "retain_narrow_decision_policy_result_f3_remains_practical_reference"
    else:
        decision = "close_finite_decision_rule_family_retain_existing_baselines"
    return {
        "status": "complete_analysis",
        "evidence_status": "corrected_fog_exploratory_development_not_confirmation",
        "reports": dict(reports),
        "comparisons": comparisons,
        "leave_one_outer_fold_sensitivity": leave_fold,
        "gates": gates,
        "event_topology": {
            "d0_to_d1": _decision_events(
                labels,
                decisions["d0"],
                decisions["d1"],
                participant_ids,
                roster,
            ),
            "f3_to_d1": _decision_events(
                labels,
                decisions["f3"],
                decisions["d1"],
                participant_ids,
                roster,
            ),
        },
        "decision": decision,
        "failure_action": "stop_no_grid_seed_gsp_router_or_dataset_extension",
        "bootstrap_scope": (
            "participant resampling conditional on fitted overlapping-fold models and "
            "adaptively consumed development cohort"
        ),
    }


def _source_manifest(repository_root: Path) -> dict[str, Any]:
    paths = (
        "AGENTS.md",
        "src/inclusive_shift_har/experiments/fog_rf_feature_weight_factorial.py",
        "src/inclusive_shift_har/experiments/fog_decision_rule_probe.py",
        "configs/experiments/fog_decision_rule_probe_v1.yaml",
        "docs/research/FOG_DECISION_RULE_PROBE_V1_PROTOCOL.md",
    )
    return {
        "record_kind": "fog_decision_rule_source_manifest",
        "files": [{"path": name, "sha256": sha256_file(repository_root / name)} for name in paths],
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
                "record_kind": "fog_decision_rule_protocol_snapshot",
                "source_path": protocol_path.relative_to(repository_root).as_posix(),
                "source_sha256": sha256_file(protocol_path),
                "text": protocol_path.read_text(encoding="utf-8"),
            }
        ),
    )


def _cache_metadata(arrays: Mapping[str, NDArray[Any]]) -> dict[str, Any]:
    return _sealed(
        {
            "record_kind": "fog_decision_rule_feature_cache_metadata",
            "label_free_feature_values": True,
            "feature_dimension": 80,
            "arrays": {
                name: {
                    "shape": list(values.shape),
                    "dtype": str(values.dtype),
                    "sha256": _array_sha256(values),
                }
                for name, values in arrays.items()
            },
        }
    )


def _outcome_summary(analysis: Mapping[str, Any], policies: Mapping[str, Any]) -> str:
    reports = _mapping(analysis["reports"], "reports")
    gates = _mapping(analysis["gates"], "gates")
    lines = [
        "# Corrected FoG bounded decision-rule outcome",
        "",
        f"D1-D0 gate: **{str(_mapping(gates['d1_minus_d0'], 'gate')['status']).upper()}**.",
        f"D1-F3 gate: **{str(_mapping(gates['d1_minus_f3'], 'gate')['status']).upper()}**.",
        f"Decision: **{analysis['decision']}**.",
        "",
        "| Output | Mean participant F1 | Pooled accuracy | Bottom 30% | Worst |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("d0", "d1", "f3"):
        report = _mapping(reports[name], name)
        primary = _mapping(report["primary"], "primary")
        pooled = _mapping(report["pooled"], "pooled")
        lines.append(
            f"| {name.upper()} | {float(primary['mean_participant_macro_f1']):.6f} | "
            f"{float(pooled['accuracy']):.6f} | "
            f"{float(primary['bottom_30_percent_participant_macro_f1']):.6f} | "
            f"{float(primary['worst_participant_macro_f1']):.6f} |"
        )
    lines.extend(
        [
            "",
            "| Outer fold | Selected [mobility, sitting, standing] multipliers | Reason |",
            "|---:|---|---|",
        ]
    )
    for row in cast(list[dict[str, Any]], policies["policies"]):
        selection = _mapping(row["selection"], "selection")
        lines.append(
            f"| {row['outer_fold']} | {selection['selected_multipliers']} | "
            f"{selection['selection_reason']} |"
        )
    comparisons = _mapping(analysis["comparisons"], "comparisons")
    lines.extend(
        [
            "",
            "| Contrast | Mean difference | 95% participant bootstrap CI | Wins/harms/ties |",
            "|---|---:|---|---|",
        ]
    )
    for name in ("d1_minus_d0", "d1_minus_f3"):
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
            "D0 and D1 share byte-identical raw probabilities. Decision scores are not",
            "probabilities. Evidence remains corrected FoG exploratory development.",
            "No additional seed, grid, GSP, dataset, physical or publication task launched.",
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
    gsp_reference_run: Path,
    factorial_reference_run: Path,
    output_directory: Path,
    code_commit: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Execute the frozen 25-fit probe and preserve complete or failed evidence."""

    repository_root = repository_root.resolve()
    evidence_root = evidence_root.resolve()
    config_path = config_path.resolve()
    protocol_path = protocol_path.resolve()
    gsp_reference_run = gsp_reference_run.resolve()
    factorial_reference_run = factorial_reference_run.resolve()
    output_directory = output_directory.resolve()
    config = _read_yaml(config_path)
    validate_config(config)
    _require(timeout_seconds == 3600, "compute timeout must remain 3600 seconds")
    _require(
        output_directory.parent == evidence_root / ".audit" / "fog_decision_rule_probe",
        "output must be in dedicated evidence family",
    )
    _require(not output_directory.exists(), "output directory already exists")
    _require(config_path.parent == repository_root / "configs" / "experiments", "wrong config path")
    _require(protocol_path.parent == repository_root / "docs" / "research", "wrong protocol path")
    git_state = _git_state(repository_root)
    _require(git_state["clean"] is True, "source worktree must be clean")
    _require(git_state["commit"] == code_commit, "code commit argument does not match HEAD")
    _require(
        gsp_reference_run.parent == evidence_root / ".audit" / "fog_gsp_order_ablation"
        and factorial_reference_run.parent
        == evidence_root / ".audit" / "fog_rf_feature_weight_factorial",
        "reference path is outside dedicated preserved evidence families",
    )
    output_directory.mkdir(parents=True, exist_ok=False)
    run_started = time.perf_counter()
    deadline = run_started + timeout_seconds
    attempts_started = 0
    completed_fits = 0
    try:
        _snapshot_inputs(repository_root, config_path, protocol_path, output_directory)
        source_manifest = _sealed(_source_manifest(repository_root))
        _write_json_create_only(output_directory / "source_manifest.json", source_manifest)
        reference_started = time.perf_counter()
        arrays, controls, reference_receipt = _load_references(
            gsp_reference_run, factorial_reference_run, config
        )
        runtime_environment = _validate_runtime_contract(gsp_reference_run)
        reference_seconds = time.perf_counter() - reference_started
        _write_json_create_only(output_directory / "reference_receipts.json", reference_receipt)
        feature_values = np.asarray(arrays["feature_values"], dtype=np.float64)
        feature_names = tuple(np.asarray(arrays["feature_names"], dtype=np.str_).tolist())
        participants = np.asarray(arrays["observable_participant_ids"], dtype=np.str_)
        folds = np.asarray(arrays["observable_fold_index"], dtype=np.int64)
        eligibility = np.asarray(arrays["scoring_eligibility"], dtype=np.bool_)
        scoring_indices = np.asarray(arrays["scoring_indices"], dtype=np.int64)
        scored_labels = np.asarray(arrays["scored_labels"], dtype=np.int64)
        observable_labels = np.zeros(participants.size, dtype=np.int64)
        observable_labels[scoring_indices] = scored_labels
        _require(
            feature_values.shape == (1939, 80)
            and participants.shape == folds.shape == eligibility.shape == (1939,)
            and scoring_indices.shape == scored_labels.shape == (1213,)
            and np.array_equal(scoring_indices, np.flatnonzero(eligibility))
            and len(feature_names) == len(set(feature_names)) == 80,
            "cache dimensions or projection changed",
        )
        _require(
            len(np.unique(arrays["observable_window_ids"])) == 1939
            and len(np.unique(arrays["scored_window_ids"])) == 1213
            and np.bincount(scored_labels, minlength=3).tolist() == [954, 74, 185],
            "window identity or class support changed",
        )
        cached_arrays = {name: np.asarray(values) for name, values in arrays.items()}
        _write_npz_create_only(output_directory / "feature_cache.npz", **cached_arrays)
        _write_json_create_only(
            output_directory / "feature_cache_metadata.json", _cache_metadata(cached_arrays)
        )
        preflight_started = time.perf_counter()
        preflight = _sealed(
            {
                **partition_preflight(
                    labels=observable_labels,
                    participants=participants,
                    folds=folds,
                    eligibility=eligibility,
                ),
                "created_at_utc": _now(),
                "source_manifest_record_sha256": source_manifest["record_sha256"],
                "reference_receipts_record_sha256": reference_receipt["record_sha256"],
                "git_state": git_state,
                "repository_root": str(repository_root),
                "config_source_path": config_path.relative_to(repository_root).as_posix(),
                "protocol_source_path": protocol_path.relative_to(repository_root).as_posix(),
                "environment": runtime_environment,
                "runtime_contract_validated_before_fits": True,
                "maximum_fit_attempts": MAXIMUM_FIT_ATTEMPTS,
                "compute_deadline_seconds": timeout_seconds,
                "expected_inner_observable_predictions": 7756,
                "expected_inner_scored_predictions": 4852,
            }
        )
        preflight_seconds = time.perf_counter() - preflight_started
        _write_json_create_only(output_directory / "partition_preflight.json", preflight)
        outer_probabilities = np.full((1939, 3), np.nan, dtype=np.float64)
        inner_probabilities = np.full((5, 1939, 3), np.nan, dtype=np.float64)
        fit_reports: list[dict[str, Any]] = []
        fit_matrix_started = time.perf_counter()
        for outer in OUTER_FOLDS:
            training = (folds != outer) & eligibility
            evaluation = folds == outer
            attempts_started += 1
            probabilities, report = _fit_model(
                feature_values=feature_values,
                labels=observable_labels,
                participants=participants,
                training=training,
                evaluation=evaluation,
                role="outer",
                outer_fold=outer,
                inner_validation_fold=None,
                attempt_number=attempts_started,
                output_directory=output_directory,
                feature_names=feature_names,
                deadline=deadline,
            )
            completed_fits += 1
            outer_probabilities[evaluation] = probabilities
            fit_reports.append(report)
        _require(bool(np.isfinite(outer_probabilities).all()), "outer predictions incomplete")
        exact_gsp = bool(
            np.array_equal(outer_probabilities, controls["gsp_a_observable_probabilities"])
        )
        exact_factorial = bool(
            np.array_equal(outer_probabilities, controls["factorial_f1_observable_probabilities"])
        )
        replay = _sealed(
            {
                "record_kind": "fog_decision_rule_reference_replay",
                "checked_before_inner_fits": True,
                "outer_fit_count_before_decision": completed_fits,
                "inner_fits_started_before_decision": False,
                "gsp_a_observable_probabilities_exact": exact_gsp,
                "factorial_f1_observable_probabilities_exact": exact_factorial,
                "maximum_absolute_difference": float(
                    np.max(np.abs(outer_probabilities - controls["gsp_a_observable_probabilities"]))
                ),
                "observable_candidate_count": 1939,
                "feature_matrix_exact_across_references": True,
                "candidate_alignment_exact": True,
            }
        )
        _write_json_create_only(output_directory / "reference_replay.json", replay)
        _require(
            exact_gsp and exact_factorial, "fresh outer F1 probabilities do not replay exactly"
        )
        for outer in OUTER_FOLDS:
            for inner in OUTER_FOLDS:
                if inner == outer:
                    continue
                training = (folds != outer) & (folds != inner) & eligibility
                evaluation = folds == inner
                attempts_started += 1
                probabilities, report = _fit_model(
                    feature_values=feature_values,
                    labels=observable_labels,
                    participants=participants,
                    training=training,
                    evaluation=evaluation,
                    role="inner",
                    outer_fold=outer,
                    inner_validation_fold=inner,
                    attempt_number=attempts_started,
                    output_directory=output_directory,
                    feature_names=feature_names,
                    deadline=deadline,
                )
                completed_fits += 1
                inner_probabilities[outer, evaluation] = probabilities
                fit_reports.append(report)
        fit_seconds = time.perf_counter() - fit_matrix_started
        _require(
            attempts_started == completed_fits == MAXIMUM_FIT_ATTEMPTS,
            "fit matrix incomplete",
        )
        finite_rows = np.asarray(np.isfinite(inner_probabilities).all(axis=2), dtype=np.bool_)
        _require(
            int(finite_rows.sum()) == 7756
            and int(np.sum(finite_rows & eligibility[None, :])) == 4852
            and all(np.array_equal(finite_rows[outer], folds != outer) for outer in OUTER_FOLDS),
            "inner-OOF coverage changed",
        )
        _write_npz_create_only(
            output_directory / "inner_predictions.npz",
            inner_observable_probabilities=inner_probabilities,
            outer_training_prediction_mask=finite_rows,
            observable_window_ids=arrays["observable_window_ids"],
            observable_participant_ids=participants,
            observable_fold_index=folds,
            scoring_indices=scoring_indices,
            scoring_eligibility=eligibility,
            scored_labels=scored_labels,
            scored_participant_ids=arrays["scored_participant_ids"],
            scored_window_ids=arrays["scored_window_ids"],
            scored_fold_index=arrays["scored_fold_index"],
        )
        selection_started = time.perf_counter()
        policy_rows: list[dict[str, Any]] = []
        selected_by_outer: dict[int, list[float]] = {}
        for outer in OUTER_FOLDS:
            selected_rows = (folds != outer) & eligibility
            inner_roster = sorted(np.unique(participants[selected_rows]).tolist())
            selection = select_policy(
                inner_probabilities[outer, selected_rows],
                observable_labels[selected_rows],
                participants[selected_rows],
                inner_roster,
            )
            _require(
                selection["participant_count"] == (17 if outer in {0, 1} else 18)
                and selection["bottom_30_participant_count"] == 6,
                "inner participant or bottom-tail count changed",
            )
            freeze = _sealed(
                {
                    "record_kind": "fog_decision_rule_policy_freeze",
                    "outer_fold": outer,
                    "created_at_utc": _now(),
                    "created_before_any_outer_d1_decoding_or_metrics": True,
                    "outer_evaluation_participants": sorted(
                        np.unique(participants[folds == outer]).tolist()
                    ),
                    "outer_training_participants": inner_roster,
                    "outer_evaluation_rows_excluded_from_selection": True,
                    "selection": selection,
                }
            )
            path = output_directory / "policy_freezes" / f"outer-fold-{outer}.json"
            _write_json_create_only(path, freeze)
            policy_rows.append(
                {
                    "outer_fold": outer,
                    "path": path.relative_to(output_directory).as_posix(),
                    "sha256": sha256_file(path),
                    "record_sha256": freeze["record_sha256"],
                    "selection": selection,
                }
            )
            selected_by_outer[outer] = cast(list[float], selection["selected_multipliers"])
        policies = _sealed(
            {
                "record_kind": "fog_decision_rule_policy_freezes",
                "all_five_policies_frozen_before_outer_d1_decoding_or_metrics": True,
                "policy_count": 5,
                "inner_rule_evaluation_count": 45,
                "policies": policy_rows,
            }
        )
        _write_json_create_only(output_directory / "policy_freezes.json", policies)
        selection_seconds = time.perf_counter() - selection_started
        decision_scores = np.empty_like(outer_probabilities)
        d1_decisions = np.empty(1939, dtype=np.int64)
        d1_emitted_probability = np.empty(1939, dtype=np.float64)
        candidate_multipliers = np.empty((1939, 3), dtype=np.float64)
        for outer in OUTER_FOLDS:
            evaluation = folds == outer
            selected = selected_by_outer[outer]
            scores, decoded, emitted = decode_with_multipliers(
                outer_probabilities[evaluation], sitting=selected[1], standing=selected[2]
            )
            decision_scores[evaluation] = scores
            d1_decisions[evaluation] = decoded
            d1_emitted_probability[evaluation] = emitted
            candidate_multipliers[evaluation] = np.asarray(selected, dtype=np.float64)
        d0_decisions = outer_probabilities.argmax(axis=1).astype(np.int64)
        d0_emitted_probability = outer_probabilities[
            np.arange(outer_probabilities.shape[0]), d0_decisions
        ]
        f3_probabilities = np.asarray(controls["f3_observable_probabilities"], dtype=np.float64)
        f3_decisions = f3_probabilities.argmax(axis=1).astype(np.int64)
        _write_npz_create_only(
            output_directory / "predictions.npz",
            raw_outer_probabilities=outer_probabilities,
            d0_decisions=d0_decisions,
            d1_decisions=d1_decisions,
            d1_decision_scores=decision_scores,
            d0_emitted_label_probability=d0_emitted_probability,
            d1_emitted_label_probability=d1_emitted_probability,
            candidate_multipliers=candidate_multipliers,
            f3_reference_probabilities=f3_probabilities,
            f3_reference_decisions=f3_decisions,
            observable_window_ids=arrays["observable_window_ids"],
            observable_participant_ids=participants,
            observable_fold_index=folds,
            scoring_indices=scoring_indices,
            scoring_eligibility=eligibility,
            scored_labels=scored_labels,
            scored_participant_ids=arrays["scored_participant_ids"],
            scored_window_ids=arrays["scored_window_ids"],
            scored_fold_index=arrays["scored_fold_index"],
        )
        scored_people = np.asarray(arrays["scored_participant_ids"], dtype=np.str_)
        scored_folds = np.asarray(arrays["scored_fold_index"], dtype=np.int64)
        roster = cast(list[str], _mapping(config["source"], "source")["participant_roster"])
        reports = {
            "d0": method_report_from_decisions(
                labels=scored_labels,
                raw_probabilities=outer_probabilities[scoring_indices],
                decisions=d0_decisions[scoring_indices],
                participant_ids=scored_people,
                roster=roster,
            ),
            "d1": method_report_from_decisions(
                labels=scored_labels,
                raw_probabilities=outer_probabilities[scoring_indices],
                decisions=d1_decisions[scoring_indices],
                participant_ids=scored_people,
                roster=roster,
            ),
            "f3": method_report_from_decisions(
                labels=scored_labels,
                raw_probabilities=f3_probabilities[scoring_indices],
                decisions=f3_decisions[scoring_indices],
                participant_ids=scored_people,
                roster=roster,
            ),
        }
        for field in ("nll", "multiclass_brier"):
            _require(
                _mapping(reports["d0"]["pooled"], "d0 pooled")[field]
                == _mapping(reports["d1"]["pooled"], "d1 pooled")[field],
                "D0/D1 proper scores differ",
            )
        participant_fold_vector = np.asarray(
            [
                int(np.unique(scored_folds[scored_people == participant]).item())
                for participant in roster
            ],
            dtype=np.int64,
        )
        analysis = analyse(
            reports,
            labels=scored_labels,
            decisions={
                "d0": d0_decisions[scoring_indices],
                "d1": d1_decisions[scoring_indices],
                "f3": f3_decisions[scoring_indices],
            },
            participant_ids=scored_people,
            participant_folds=participant_fold_vector,
            roster=roster,
            control_replay_exact=True,
        )
        participant_payload = _sealed(
            {"record_kind": "fog_decision_rule_participant_metrics", "methods": reports}
        )
        analysis_payload = _sealed({"record_kind": "fog_decision_rule_analysis", **analysis})
        _write_json_create_only(output_directory / "participant_metrics.json", participant_payload)
        _write_json_create_only(output_directory / "analysis.json", analysis_payload)
        fit_payload = _sealed(
            {
                "record_kind": "fog_decision_rule_fit_reports",
                "fit_attempt_count": attempts_started,
                "completed_fit_count": completed_fits,
                "rows": fit_reports,
            }
        )
        _write_json_create_only(output_directory / "fit_reports.json", fit_payload)
        total_seconds = time.perf_counter() - run_started
        runtime = _sealed(
            {
                "record_kind": "fog_decision_rule_runtime",
                "reference_verification_seconds": reference_seconds,
                "partition_preflight_seconds": preflight_seconds,
                "fit_and_prediction_seconds": fit_seconds,
                "selection_and_freeze_seconds": selection_seconds,
                "total_run_seconds": total_seconds,
                "compute_cap_seconds": timeout_seconds,
                "cap_exceeded": total_seconds > timeout_seconds,
                "fit_attempt_count": attempts_started,
                "completed_model_fit_count": completed_fits,
                "outer_fit_count": 5,
                "inner_fit_count": 20,
                "first_registered_fit_benchmark_seconds": float(
                    fit_reports[0]["fit_and_predict_seconds"]
                ),
                "additional_benchmark_fit_count": 0,
                "fit_worker_count": 4,
                "prediction_worker_count": 1,
                "blas_thread_limit": 1,
                "execution_parallelism": "sequential_models_single_controller",
            }
        )
        _require(total_seconds <= timeout_seconds, "complete run exceeded compute cap")
        _write_json_create_only(output_directory / "runtime.json", runtime)
        result = _sealed(
            {
                "record_kind": "fog_decision_rule_result",
                "status": "complete_awaiting_independent_replay",
                "experiment_id": EXPERIMENT_ID,
                "evidence_status": config["evidence_status"],
                "code_commit": code_commit,
                "source_sha256": _mapping(config["source"], "source")["expected_sha256"],
                "observable_candidate_count": 1939,
                "scored_window_count": 1213,
                "participant_count": 22,
                "feature_dimension": 80,
                "fit_attempt_count": attempts_started,
                "completed_fit_count": completed_fits,
                "inner_rule_evaluation_count": 45,
                "policy_count": 5,
                "selected_policies": [selected_by_outer[outer] for outer in OUTER_FOLDS],
                "method_summary": {
                    name: _mapping(report["primary"], "primary") for name, report in reports.items()
                },
                "gates": analysis["gates"],
                "decision": analysis["decision"],
                "raw_probability_identity_d0_d1": True,
                "automatic_follow_on_launched": False,
                "additional_seeds_launched": False,
                "other_dataset_loaded": False,
                "physical_pilot_launched": False,
                "publication_queue_launched": False,
            }
        )
        _write_json_create_only(output_directory / "result.json", result)
        _write_bytes_create_only(
            output_directory / "OUTCOME_SUMMARY.md",
            _outcome_summary(analysis, policies).encode("utf-8"),
        )
        worker_shutdown = _worker_shutdown_receipt(completed_fits)
        _write_json_create_only(output_directory / "worker_shutdown.json", worker_shutdown)
        files = [
            path
            for path in output_directory.rglob("*")
            if path.is_file()
            and path.name
            not in {"artifact_manifest.json", "validation.json", "completion_manifest.json"}
        ]
        artifact_manifest = _sealed(
            {
                "record_kind": "fog_decision_rule_artifact_manifest",
                "status": "complete_prevalidation",
                "artifacts": [
                    {
                        "path": path.relative_to(output_directory).as_posix(),
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in sorted(files)
                ],
            }
        )
        _write_json_create_only(output_directory / "artifact_manifest.json", artifact_manifest)
        return result
    except BaseException as error:
        failure = {
            "record_kind": "fog_decision_rule_failure",
            "status": "incomplete_stop_no_follow_on",
            "failed_at_utc": _now(),
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "fit_attempts_started": attempts_started,
            "completed_fits": completed_fits,
            "elapsed_seconds": time.perf_counter() - run_started,
            "compute_cap_seconds": timeout_seconds,
            "automatic_retry_allowed": False,
            "automatic_follow_on_launched": False,
        }
        failure_path = output_directory / "failure.json"
        if not failure_path.exists():
            _write_json_create_only(failure_path, _sealed(failure))
        raise


def _verify_manifest_entries(run_directory: Path, manifest: Mapping[str, Any]) -> int:
    _verify_sealed(manifest, "artifact manifest")
    _require(
        manifest["record_kind"] == "fog_decision_rule_artifact_manifest"
        and manifest["status"] == "complete_prevalidation",
        "artifact manifest status changed",
    )
    entries = cast(list[dict[str, Any]], manifest["artifacts"])
    declared_paths = [str(entry["path"]) for entry in entries]
    _require(len(declared_paths) == len(set(declared_paths)), "duplicate artifact path")
    actual_paths = {
        path.relative_to(run_directory).as_posix()
        for path in run_directory.rglob("*")
        if path.is_file()
        and path.name
        not in {"artifact_manifest.json", "validation.json", "completion_manifest.json"}
    }
    _require(set(declared_paths) == actual_paths, "artifact manifest coverage changed")
    for entry in entries:
        path = (run_directory / str(entry["path"])).resolve()
        _require(_is_relative_to(path, run_directory), "artifact path escapes run")
        _require(
            path.is_file()
            and sha256_file(path) == entry["sha256"]
            and path.stat().st_size == entry["size_bytes"],
            "artifact manifest entry changed",
        )
    return len(entries)


def _reconstruct_analysis(
    *,
    config: Mapping[str, Any],
    arrays: Mapping[str, NDArray[Any]],
    raw_probabilities: FloatArray,
    f3_probabilities: FloatArray,
    d0_decisions: IntArray,
    d1_decisions: IntArray,
    f3_decisions: IntArray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scoring = np.asarray(arrays["scoring_indices"], dtype=np.int64)
    labels = np.asarray(arrays["scored_labels"], dtype=np.int64)
    people = np.asarray(arrays["scored_participant_ids"], dtype=np.str_)
    scored_folds = np.asarray(arrays["scored_fold_index"], dtype=np.int64)
    roster = cast(list[str], _mapping(config["source"], "source")["participant_roster"])
    reports = {
        "d0": method_report_from_decisions(
            labels=labels,
            raw_probabilities=raw_probabilities[scoring],
            decisions=d0_decisions[scoring],
            participant_ids=people,
            roster=roster,
        ),
        "d1": method_report_from_decisions(
            labels=labels,
            raw_probabilities=raw_probabilities[scoring],
            decisions=d1_decisions[scoring],
            participant_ids=people,
            roster=roster,
        ),
        "f3": method_report_from_decisions(
            labels=labels,
            raw_probabilities=f3_probabilities[scoring],
            decisions=f3_decisions[scoring],
            participant_ids=people,
            roster=roster,
        ),
    }
    participant_folds = np.asarray(
        [int(np.unique(scored_folds[people == participant]).item()) for participant in roster],
        dtype=np.int64,
    )
    analysis = analyse(
        reports,
        labels=labels,
        decisions={
            "d0": d0_decisions[scoring],
            "d1": d1_decisions[scoring],
            "f3": f3_decisions[scoring],
        },
        participant_ids=people,
        participant_folds=participant_folds,
        roster=roster,
        control_replay_exact=True,
    )
    return reports, analysis


def validate_run(run_directory: Path) -> dict[str, Any]:
    """Replay 25 checkpoints and all policy/report logic without fitting."""

    run_directory = run_directory.resolve()
    _require(run_directory.is_dir(), "run directory does not exist")
    _require(not (run_directory / "failure.json").exists(), "failed run cannot validate")
    _require(not (run_directory / "validation.json").exists(), "validation already exists")
    _require(not (run_directory / "completion_manifest.json").exists(), "completion already exists")
    for name in REQUIRED_RUN_FILES:
        _require((run_directory / name).is_file(), f"required run file missing: {name}")
    manifest = _read_json(run_directory / "artifact_manifest.json")
    artifact_count = _verify_manifest_entries(run_directory, manifest)
    config = _read_yaml(run_directory / "config_snapshot.yaml")
    validate_config(config)
    source_manifest = _read_json(run_directory / "source_manifest.json")
    stored_preflight = _read_json(run_directory / "partition_preflight.json")
    protocol_snapshot = _read_json(run_directory / "protocol_snapshot.json")
    _verify_sealed(source_manifest, "source manifest")
    _verify_sealed(stored_preflight, "partition preflight")
    _verify_sealed(protocol_snapshot, "protocol snapshot")
    repository_root = Path(str(stored_preflight["repository_root"])).resolve()
    _require(repository_root.is_dir(), "recorded source worktree is unavailable")
    recorded_git = _mapping(stored_preflight["git_state"], "recorded git state")
    current_git = _git_state(repository_root)
    _require(
        current_git["clean"] is True and current_git["commit"] == recorded_git["commit"],
        "recorded source worktree commit or cleanliness changed",
    )
    expected_source_manifest = _sealed(_source_manifest(repository_root))
    _require(source_manifest == expected_source_manifest, "source manifest reconstruction changed")
    _require(
        stored_preflight["source_manifest_record_sha256"] == source_manifest["record_sha256"],
        "preflight source-manifest binding changed",
    )
    config_source = (repository_root / str(stored_preflight["config_source_path"])).resolve()
    protocol_source = (repository_root / str(stored_preflight["protocol_source_path"])).resolve()
    _require(
        _is_relative_to(config_source, repository_root)
        and _is_relative_to(protocol_source, repository_root)
        and config_source.is_file()
        and protocol_source.is_file(),
        "recorded config or protocol path is invalid",
    )
    _require(
        config_source.read_bytes() == (run_directory / "config_snapshot.yaml").read_bytes(),
        "configuration snapshot differs from committed configuration",
    )
    _require(
        protocol_snapshot["source_path"] == stored_preflight["protocol_source_path"]
        and protocol_snapshot["source_sha256"] == sha256_file(protocol_source)
        and protocol_snapshot["text"] == protocol_source.read_text(encoding="utf-8"),
        "protocol snapshot differs from committed protocol",
    )
    feature_cache = _read_npz(run_directory / "feature_cache.npz")
    cache_metadata = _read_json(run_directory / "feature_cache_metadata.json")
    _verify_sealed(cache_metadata, "feature cache metadata")
    metadata_arrays = _mapping(cache_metadata["arrays"], "cache arrays")
    _require(set(feature_cache) == set(metadata_arrays), "cache array set changed")
    for name, values in feature_cache.items():
        row = _mapping(metadata_arrays[name], name)
        _require(
            row["shape"] == list(values.shape)
            and row["dtype"] == str(values.dtype)
            and row["sha256"] == _array_sha256(values),
            f"cache changed: {name}",
        )
    reference_receipt = _read_json(run_directory / "reference_receipts.json")
    _verify_sealed(reference_receipt, "reference receipt")
    gsp_path = Path(str(_mapping(reference_receipt["gsp"], "GSP receipt")["run_path"]))
    factorial_path = Path(
        str(_mapping(reference_receipt["factorial"], "factorial receipt")["run_path"])
    )
    reference_arrays, controls, reconstructed_receipt = _load_references(
        gsp_path, factorial_path, config
    )
    _require(
        reconstructed_receipt == reference_receipt,
        "reference receipt reconstruction changed",
    )
    _require(
        stored_preflight["reference_receipts_record_sha256"] == reference_receipt["record_sha256"],
        "preflight reference binding changed",
    )
    current_environment = _validate_runtime_contract(gsp_path)
    recorded_environment = _mapping(stored_preflight["environment"], "recorded environment")
    for key in ("python", "numpy", "scikit_learn"):
        _require(
            recorded_environment[key] == current_environment[key],
            f"recorded runtime changed: {key}",
        )
    _require(
        Path(str(recorded_environment["python_executable"])).resolve()
        == Path(str(current_environment["python_executable"])).resolve()
        and stored_preflight["runtime_contract_validated_before_fits"] is True,
        "recorded interpreter or pre-fit runtime validation changed",
    )
    _require(set(feature_cache) == set(reference_arrays), "reference cache key set changed")
    for name, reference_values in reference_arrays.items():
        local_values = feature_cache[name]
        _require(
            local_values.dtype == reference_values.dtype
            and local_values.shape == reference_values.shape
            and np.array_equal(local_values, reference_values),
            f"local cache differs from pinned reference: {name}",
        )
    reference_replay = _read_json(run_directory / "reference_replay.json")
    _verify_sealed(reference_replay, "reference replay")
    _require(
        reference_replay["checked_before_inner_fits"] is True
        and reference_replay["outer_fit_count_before_decision"] == 5
        and reference_replay["inner_fits_started_before_decision"] is False
        and reference_replay["gsp_a_observable_probabilities_exact"] is True
        and reference_replay["factorial_f1_observable_probabilities_exact"] is True
        and reference_replay["maximum_absolute_difference"] == 0.0
        and reference_replay["observable_candidate_count"] == 1939
        and reference_replay["feature_matrix_exact_across_references"] is True
        and reference_replay["candidate_alignment_exact"] is True,
        "control-before-inner replay receipt changed",
    )
    values = np.asarray(feature_cache["feature_values"], dtype=np.float64)
    names = tuple(np.asarray(feature_cache["feature_names"], dtype=np.str_).tolist())
    participants = np.asarray(feature_cache["observable_participant_ids"], dtype=np.str_)
    folds = np.asarray(feature_cache["observable_fold_index"], dtype=np.int64)
    eligibility = np.asarray(feature_cache["scoring_eligibility"], dtype=np.bool_)
    scoring = np.asarray(feature_cache["scoring_indices"], dtype=np.int64)
    labels = np.zeros(1939, dtype=np.int64)
    labels[scoring] = np.asarray(feature_cache["scored_labels"], dtype=np.int64)
    recomputed_preflight = partition_preflight(
        labels=labels, participants=participants, folds=folds, eligibility=eligibility
    )
    for key in (
        "partition_count",
        "outer_partition_count",
        "inner_partition_count",
        "all_training_partitions_have_three_classes",
        "all_training_evaluation_participants_disjoint",
        "rows",
    ):
        _require(stored_preflight[key] == recomputed_preflight[key], "preflight changed")
    fit_payload = _read_json(run_directory / "fit_reports.json")
    _verify_sealed(fit_payload, "fit reports")
    fit_rows = cast(list[dict[str, Any]], fit_payload["rows"])
    _require(
        fit_payload["fit_attempt_count"] == fit_payload["completed_fit_count"] == 25
        and len(fit_rows) == 25
        and fit_payload["record_kind"] == "fog_decision_rule_fit_reports",
        "fit report count changed",
    )
    expected_attempts: list[tuple[str, int, int | None]] = [
        ("outer", outer, None) for outer in OUTER_FOLDS
    ]
    expected_attempts.extend(
        ("inner", outer, inner) for outer in OUTER_FOLDS for inner in OUTER_FOLDS if inner != outer
    )
    expected_attempt_files = {
        f"{attempt:02d}--{state}.json"
        for attempt in range(1, MAXIMUM_FIT_ATTEMPTS + 1)
        for state in ("started", "completed")
    }
    _require(
        {path.name for path in (run_directory / "fit_attempts").glob("*") if path.is_file()}
        == expected_attempt_files,
        "fit-attempt artifact family changed",
    )
    expected_checkpoint_files = {
        *(f"outer--fold-{outer}.pkl" for outer in OUTER_FOLDS),
        *(
            f"inner--outer-{outer}--validation-{inner}.pkl"
            for outer in OUTER_FOLDS
            for inner in OUTER_FOLDS
            if inner != outer
        ),
    }
    _require(
        {path.name for path in (run_directory / "checkpoints").glob("*") if path.is_file()}
        == expected_checkpoint_files,
        "checkpoint artifact family changed",
    )
    _require(
        {path.name for path in (run_directory / "policy_freezes").glob("*") if path.is_file()}
        == {f"outer-fold-{outer}.json" for outer in OUTER_FOLDS},
        "policy-freeze artifact family changed",
    )
    outer_replay = np.full((1939, 3), np.nan, dtype=np.float64)
    inner_replay = np.full((5, 1939, 3), np.nan, dtype=np.float64)
    checkpoint_count = 0
    for attempt_number, (expected_role, outer, inner) in enumerate(expected_attempts, start=1):
        row = fit_rows[attempt_number - 1]
        random_state = (
            11 + outer if expected_role == "outer" else 1100 + 10 * outer + cast(int, inner)
        )
        _require(
            row["attempt_number"] == attempt_number
            and row["role"] == expected_role
            and row["outer_fold"] == outer
            and row["inner_validation_fold"] == inner,
            "fit order changed",
        )
        started = _read_json(run_directory / "fit_attempts" / f"{attempt_number:02d}--started.json")
        completed = _read_json(
            run_directory / "fit_attempts" / f"{attempt_number:02d}--completed.json"
        )
        _verify_sealed(started, "started attempt")
        _verify_sealed(completed, "completed attempt")
        training = (
            ((folds != outer) & eligibility)
            if expected_role == "outer"
            else ((folds != outer) & (folds != cast(int, inner)) & eligibility)
        )
        evaluation = folds == (outer if expected_role == "outer" else cast(int, inner))
        checkpoint = _mapping(row["checkpoint"], "checkpoint")
        _require(
            started["attempt_number"] == completed["attempt_number"] == attempt_number
            and started["role"] == completed["role"] == expected_role
            and started["outer_fold"] == completed["outer_fold"] == outer
            and started["inner_validation_fold"] == completed["inner_validation_fold"] == inner
            and started["registered_timing_benchmark"] is (attempt_number == 1)
            and started["random_state"] == random_state
            and started["training_row_count"] == int(training.sum())
            and started["evaluation_candidate_count"] == int(evaluation.sum())
            and completed["checkpoint"] == checkpoint
            and math.isfinite(float(completed["fit_seconds"]))
            and float(completed["fit_seconds"]) >= 0.0
            and math.isfinite(float(completed["prediction_seconds"]))
            and float(completed["prediction_seconds"]) >= 0.0,
            "attempt ledger changed",
        )
        checkpoint_path = (run_directory / str(checkpoint["path"])).resolve()
        _require(_is_relative_to(checkpoint_path, run_directory), "checkpoint escapes run")
        _require(
            checkpoint_path.is_file()
            and sha256_file(checkpoint_path) == checkpoint["sha256"]
            and checkpoint_path.stat().st_size == checkpoint["size_bytes"],
            "checkpoint changed",
        )
        stored_metadata, estimator = _load_checkpoint(checkpoint_path)
        expected_weight = participant_first_weights(labels[training], participants[training])
        weight_summary = _mapping(stored_metadata["sample_weight_summary"], "weights")
        row_without_checkpoint = dict(row)
        row_without_checkpoint.pop("checkpoint")
        _require(
            row_without_checkpoint == stored_metadata
            and stored_metadata["attempt_number"] == attempt_number
            and stored_metadata["role"] == expected_role
            and stored_metadata["outer_fold"] == outer
            and stored_metadata["inner_validation_fold"] == inner
            and stored_metadata["random_state"] == random_state
            and tuple(stored_metadata["feature_names"]) == names
            and stored_metadata["feature_names_sha256"] == canonical_json_sha256(list(names))
            and stored_metadata["training_row_count"] == int(training.sum())
            and stored_metadata["evaluation_candidate_count"] == int(evaluation.sum())
            and stored_metadata["training_participants"]
            == sorted(np.unique(participants[training]).tolist())
            and stored_metadata["evaluation_participants"]
            == sorted(np.unique(participants[evaluation]).tolist())
            and stored_metadata["training_class_counts"]
            == np.bincount(labels[training], minlength=3).tolist()
            and weight_summary == _weight_summary(labels, participants, training)
            and weight_summary["sha256"] == _array_sha256(expected_weight)
            and stored_metadata["effective_parameters"]
            == _effective_estimator_parameters(random_state)
            and estimator.get_params(deep=False) == _effective_estimator_parameters(random_state)
            and np.array_equal(estimator.classes_, np.arange(3))
            and int(estimator.n_features_in_) == 80,
            "checkpoint metadata, split, weights or estimator changed",
        )
        with threadpool_limits(limits=1):
            replay = _predict_proba_deterministically(estimator, values[evaluation])
        if expected_role == "outer":
            outer_replay[evaluation] = replay
        else:
            inner_replay[outer, evaluation] = replay
        checkpoint_count += 1
    predictions = _read_npz(run_directory / "predictions.npz")
    inner_predictions = _read_npz(run_directory / "inner_predictions.npz")
    alignment_keys = (
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
    expected_prediction_keys = {
        "raw_outer_probabilities",
        "d0_decisions",
        "d1_decisions",
        "d1_decision_scores",
        "d0_emitted_label_probability",
        "d1_emitted_label_probability",
        "candidate_multipliers",
        "f3_reference_probabilities",
        "f3_reference_decisions",
        *alignment_keys,
    }
    expected_inner_keys = {
        "inner_observable_probabilities",
        "outer_training_prediction_mask",
        *alignment_keys,
    }
    _require(
        set(predictions) == expected_prediction_keys
        and set(inner_predictions) == expected_inner_keys,
        "prediction archive key set changed",
    )
    for key in alignment_keys:
        _require(
            predictions[key].dtype == feature_cache[key].dtype
            and inner_predictions[key].dtype == feature_cache[key].dtype
            and np.array_equal(predictions[key], feature_cache[key])
            and np.array_equal(inner_predictions[key], feature_cache[key]),
            f"prediction metadata alignment changed: {key}",
        )
    stored_outer = np.asarray(predictions["raw_outer_probabilities"], dtype=np.float64)
    stored_inner = np.asarray(inner_predictions["inner_observable_probabilities"], dtype=np.float64)
    stored_inner_mask = np.asarray(
        inner_predictions["outer_training_prediction_mask"], dtype=np.bool_
    )
    finite_inner_rows = np.isfinite(stored_inner).all(axis=2)
    all_nan_inner_rows = np.isnan(stored_inner).all(axis=2)
    _require(
        stored_outer.shape == (1939, 3)
        and stored_inner.shape == (5, 1939, 3)
        and stored_inner_mask.shape == (5, 1939)
        and bool(np.isfinite(stored_outer).all())
        and bool(np.all(stored_outer >= 0.0))
        and np.allclose(stored_outer.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
        and np.array_equal(finite_inner_rows, stored_inner_mask)
        and np.array_equal(all_nan_inner_rows, ~stored_inner_mask)
        and int(stored_inner_mask.sum()) == 7756
        and int(np.sum(stored_inner_mask & eligibility[None, :])) == 4852
        and all(np.array_equal(stored_inner_mask[outer], folds != outer) for outer in OUTER_FOLDS)
        and bool(np.all(stored_inner[stored_inner_mask] >= 0.0))
        and np.allclose(stored_inner[stored_inner_mask].sum(axis=1), 1.0, rtol=0.0, atol=1e-12),
        "prediction probability values or inner coverage changed",
    )
    _require(
        np.array_equal(outer_replay, stored_outer)
        and np.array_equal(inner_replay, stored_inner, equal_nan=True)
        and np.array_equal(outer_replay, controls["gsp_a_observable_probabilities"])
        and np.array_equal(outer_replay, controls["factorial_f1_observable_probabilities"]),
        "checkpoint probability replay differs",
    )
    policy_payload = _read_json(run_directory / "policy_freezes.json")
    _verify_sealed(policy_payload, "policy freezes")
    policy_rows = cast(list[dict[str, Any]], policy_payload["policies"])
    _require(
        policy_payload["record_kind"] == "fog_decision_rule_policy_freezes"
        and policy_payload["all_five_policies_frozen_before_outer_d1_decoding_or_metrics"] is True
        and policy_payload["policy_count"] == len(policy_rows) == 5
        and policy_payload["inner_rule_evaluation_count"] == 45,
        "policy count changed",
    )
    selected: dict[int, list[float]] = {}
    for outer, row in enumerate(policy_rows):
        _require(row["outer_fold"] == outer, "policy order changed")
        policy_path = (run_directory / str(row["path"])).resolve()
        _require(
            _is_relative_to(policy_path, run_directory)
            and sha256_file(policy_path) == row["sha256"],
            "policy freeze changed",
        )
        freeze = _read_json(policy_path)
        _verify_sealed(freeze, "policy freeze")
        selected_rows = (folds != outer) & eligibility
        inner_roster = sorted(np.unique(participants[selected_rows]).tolist())
        evaluation_roster = sorted(np.unique(participants[folds == outer]).tolist())
        reconstructed = select_policy(
            inner_replay[outer, selected_rows],
            labels[selected_rows],
            participants[selected_rows],
            inner_roster,
        )
        _require(
            freeze["record_kind"] == "fog_decision_rule_policy_freeze"
            and freeze["outer_fold"] == outer
            and freeze["created_before_any_outer_d1_decoding_or_metrics"] is True
            and freeze["outer_evaluation_participants"] == evaluation_roster
            and freeze["outer_training_participants"] == inner_roster
            and freeze["outer_evaluation_rows_excluded_from_selection"] is True
            and freeze["selection"] == reconstructed
            and row["record_sha256"] == freeze["record_sha256"]
            and row["selection"] == reconstructed,
            "policy selection changed",
        )
        selected[outer] = cast(list[float], reconstructed["selected_multipliers"])
    decision_scores = np.empty_like(outer_replay)
    d1 = np.empty(1939, dtype=np.int64)
    d1_emitted = np.empty(1939, dtype=np.float64)
    candidate_multipliers = np.empty((1939, 3), dtype=np.float64)
    for outer in OUTER_FOLDS:
        evaluation = folds == outer
        multiplier = selected[outer]
        scores, decoded, emitted = decode_with_multipliers(
            outer_replay[evaluation], sitting=multiplier[1], standing=multiplier[2]
        )
        decision_scores[evaluation] = scores
        d1[evaluation] = decoded
        d1_emitted[evaluation] = emitted
        candidate_multipliers[evaluation] = multiplier
    d0 = outer_replay.argmax(axis=1).astype(np.int64)
    d0_emitted = outer_replay[np.arange(1939), d0]
    f3_probabilities = np.asarray(controls["f3_observable_probabilities"], dtype=np.float64)
    f3 = f3_probabilities.argmax(axis=1).astype(np.int64)
    _require(
        np.array_equal(predictions["d0_decisions"], d0)
        and np.array_equal(predictions["d1_decisions"], d1)
        and np.array_equal(predictions["d1_decision_scores"], decision_scores)
        and np.array_equal(predictions["d0_emitted_label_probability"], d0_emitted)
        and np.array_equal(predictions["d1_emitted_label_probability"], d1_emitted)
        and np.array_equal(predictions["candidate_multipliers"], candidate_multipliers)
        and np.array_equal(predictions["f3_reference_probabilities"], f3_probabilities)
        and np.array_equal(predictions["f3_reference_decisions"], f3),
        "stored decision arrays changed",
    )
    _require(
        predictions["raw_outer_probabilities"].dtype == np.dtype(np.float64)
        and predictions["d0_decisions"].dtype == np.dtype(np.int64)
        and predictions["d1_decisions"].dtype == np.dtype(np.int64)
        and predictions["d1_decision_scores"].dtype == np.dtype(np.float64)
        and predictions["d0_emitted_label_probability"].dtype == np.dtype(np.float64)
        and predictions["d1_emitted_label_probability"].dtype == np.dtype(np.float64)
        and predictions["candidate_multipliers"].dtype == np.dtype(np.float64)
        and predictions["f3_reference_probabilities"].dtype == np.dtype(np.float64)
        and predictions["f3_reference_decisions"].dtype == np.dtype(np.int64)
        and inner_predictions["inner_observable_probabilities"].dtype == np.dtype(np.float64)
        and inner_predictions["outer_training_prediction_mask"].dtype == np.dtype(np.bool_),
        "prediction array dtype changed",
    )
    reports, analysis = _reconstruct_analysis(
        config=config,
        arrays=feature_cache,
        raw_probabilities=outer_replay,
        f3_probabilities=f3_probabilities,
        d0_decisions=d0,
        d1_decisions=d1,
        f3_decisions=f3,
    )
    stored_reports = _read_json(run_directory / "participant_metrics.json")
    stored_analysis = _read_json(run_directory / "analysis.json")
    _verify_sealed(stored_reports, "participant reports")
    _verify_sealed(stored_analysis, "analysis")
    _require(stored_reports["methods"] == reports, "participant reports changed")
    analysis_without_hash = dict(stored_analysis)
    analysis_without_hash.pop("record_sha256")
    _require(
        analysis_without_hash == {"record_kind": "fog_decision_rule_analysis", **analysis},
        "analysis reconstruction changed",
    )
    _require(
        (run_directory / "OUTCOME_SUMMARY.md").read_text(encoding="utf-8")
        == _outcome_summary(analysis, policy_payload),
        "outcome summary changed",
    )
    result = _read_json(run_directory / "result.json")
    runtime = _read_json(run_directory / "runtime.json")
    worker_shutdown = _read_json(run_directory / "worker_shutdown.json")
    _verify_sealed(result, "result")
    _verify_sealed(runtime, "runtime")
    _verify_sealed(worker_shutdown, "worker shutdown")
    _require(
        worker_shutdown["record_kind"] == "fog_decision_rule_worker_shutdown"
        and worker_shutdown["completed_fit_count"] == 25
        and worker_shutdown["active_multiprocessing_child_count"] == 0
        and worker_shutdown["active_non_main_python_thread_count"] == 0
        and worker_shutdown["active_multiprocessing_children"] == []
        and worker_shutdown["active_non_main_python_threads"] == []
        and worker_shutdown["task_owned_fit_workers_and_monitors_stopped"] is True,
        "fit-worker shutdown receipt changed",
    )
    expected_result = {
        "record_kind": "fog_decision_rule_result",
        "status": "complete_awaiting_independent_replay",
        "experiment_id": EXPERIMENT_ID,
        "evidence_status": config["evidence_status"],
        "code_commit": recorded_git["commit"],
        "source_sha256": _mapping(config["source"], "source")["expected_sha256"],
        "observable_candidate_count": 1939,
        "scored_window_count": 1213,
        "participant_count": 22,
        "feature_dimension": 80,
        "fit_attempt_count": 25,
        "completed_fit_count": 25,
        "inner_rule_evaluation_count": 45,
        "policy_count": 5,
        "selected_policies": [selected[outer] for outer in OUTER_FOLDS],
        "method_summary": {
            name: _mapping(report["primary"], f"{name} primary") for name, report in reports.items()
        },
        "gates": analysis["gates"],
        "decision": analysis["decision"],
        "raw_probability_identity_d0_d1": True,
        "automatic_follow_on_launched": False,
        "additional_seeds_launched": False,
        "other_dataset_loaded": False,
        "physical_pilot_launched": False,
        "publication_queue_launched": False,
    }
    result_without_hash = dict(result)
    result_without_hash.pop("record_sha256")
    _require(result_without_hash == expected_result, "result reconstruction changed")
    timing_fields = (
        "reference_verification_seconds",
        "partition_preflight_seconds",
        "fit_and_prediction_seconds",
        "selection_and_freeze_seconds",
        "total_run_seconds",
        "first_registered_fit_benchmark_seconds",
    )
    timings = [float(runtime[field]) for field in timing_fields]
    _require(
        runtime["record_kind"] == "fog_decision_rule_runtime"
        and all(math.isfinite(value) and value >= 0.0 for value in timings)
        and sum(timings[:4]) <= float(runtime["total_run_seconds"])
        and runtime["first_registered_fit_benchmark_seconds"]
        == fit_rows[0]["fit_and_predict_seconds"]
        and runtime["fit_attempt_count"] == runtime["completed_model_fit_count"] == 25
        and runtime["outer_fit_count"] == 5
        and runtime["inner_fit_count"] == 20
        and runtime["additional_benchmark_fit_count"] == 0
        and runtime["compute_cap_seconds"] == 3600
        and runtime["cap_exceeded"] is False
        and runtime["fit_worker_count"] == 4
        and runtime["prediction_worker_count"] == 1
        and runtime["blas_thread_limit"] == 1
        and runtime["execution_parallelism"] == "sequential_models_single_controller"
        and float(runtime["total_run_seconds"]) <= 3600,
        "result or runtime completion fields changed",
    )
    validation = _sealed(
        {
            "record_kind": "fog_decision_rule_validation",
            "status": "validated",
            "validated_at_utc": _now(),
            "artifact_hashes_verified": artifact_count,
            "reference_manifests_reverified": 2,
            "checkpoint_predictions_replayed": checkpoint_count,
            "model_fits_during_validation": 0,
            "outer_probability_replay_exact": True,
            "inner_probability_replay_exact": True,
            "fresh_f1_reference_probability_replay_exact": True,
            "inner_observable_predictions_replayed": 7756,
            "inner_scored_predictions_replayed": 4852,
            "inner_rule_rows_recomputed": 45,
            "policy_freezes_recomputed": 5,
            "explicit_decisions_recomputed": 2,
            "participant_reports_recomputed": 3,
            "comparisons_and_gates_recomputed": 2,
            "raw_d0_d1_probabilities_identical": True,
            "raw_d0_d1_nll_and_brier_identical": True,
            "selected_label_probability_rule_verified": True,
            "full_22_person_roster_retained": True,
            "source_or_other_dataset_loaded": False,
            "automatic_follow_on_launched": False,
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
            "record_kind": "fog_decision_rule_completion_manifest",
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
            "gates": analysis["gates"],
            "decision": analysis["decision"],
            "worker_shutdown_record_sha256": worker_shutdown["record_sha256"],
            "task_owned_workers_and_monitors_stopped": worker_shutdown[
                "task_owned_fit_workers_and_monitors_stopped"
            ],
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
    run_parser.add_argument("--gsp-reference-run", type=Path, required=True)
    run_parser.add_argument("--factorial-reference-run", type=Path, required=True)
    run_parser.add_argument("--output-directory", type=Path, required=True)
    run_parser.add_argument("--code-commit", required=True)
    run_parser.add_argument("--timeout-seconds", type=int, default=3600)
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
            gsp_reference_run=arguments.gsp_reference_run,
            factorial_reference_run=arguments.factorial_reference_run,
            output_directory=arguments.output_directory,
            code_commit=str(arguments.code_commit),
            timeout_seconds=int(arguments.timeout_seconds),
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "gates": result["gates"],
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
