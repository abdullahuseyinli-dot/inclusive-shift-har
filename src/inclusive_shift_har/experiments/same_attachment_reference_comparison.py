"""Finite schedule, admission, weighting and gates for the prospective reference trial.

The controller is intentionally unable to turn a synthetic or merely complete physical
run into human-performance fitting. Numerical model primitives live in the models module;
this module defines the evidence boundary and deterministic analysis contract.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.models.same_attachment_reference_training import ReferenceTrainingError

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

ARM_IDS = ("Z", "A", "B", "C", "D", "D-additive", "D-constant", "K", "MAP")
MOTION_FITS = (
    "factorized",
    "point_joint",
    "directional_joint",
    "directional_additive",
    "directional_constant",
    "polynomial_capacity",
)
POSTURE_FITS = ("point", "directional", "directional_constant")
CLASS_NAMES = ("mobility", "sitting", "standing")
CLASS_CONDITIONS = {
    0: ("usual", "slow", "turning"),
    1: ("quiet", "upper_body_motion"),
    2: ("quiet", "upper_body_motion"),
}


@dataclass(frozen=True)
class FitAttempt:
    attempt_index: int
    fold_index: int
    object_kind: str
    object_id: str


def build_fit_schedule() -> tuple[FitAttempt, ...]:
    attempts: list[FitAttempt] = []
    for fold in range(6):
        for name in MOTION_FITS:
            attempts.append(FitAttempt(len(attempts), fold, "binary_motion", name))
        for name in POSTURE_FITS:
            attempts.append(FitAttempt(len(attempts), fold, "binary_posture", name))
        attempts.append(FitAttempt(len(attempts), fold, "multinomial", "Z"))
        attempts.append(FitAttempt(len(attempts), fold, "scalar_concentration", "variable_tau"))
        attempts.append(FitAttempt(len(attempts), fold, "scalar_concentration", "common_kappa"))
    return tuple(attempts)


FIT_SCHEDULE = build_fit_schedule()


def validate_supervised_admission(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Reject synthetic/template completion and require both independent information gates."""

    required_true = (
        "signal_run_independently_validated",
        "qualification_manifest_verified",
        "canonicalization_receipt_verified",
        "human_authorization_and_consent_verified",
        "canonical_recordings_bound",
        "adjudicated_annotations_bound",
        "processing_frozen_before_human_outcomes",
        "original_pilot_gate_passed",
        "pooled_companion_gate_passed",
    )
    failures = [name for name in required_true if receipt.get(name) is not True]
    if receipt.get("evidence_status") != "real_physical_development_observations":
        failures.append("evidence_status_is_not_real_physical_development_observations")
    exact = {
        "wearer_count": 6,
        "attachment_block_count": 24,
        "support_bout_count": 96,
        "query_bout_count": 168,
        "total_bout_count": 264,
    }
    for name, expected in exact.items():
        if receipt.get(name) != expected:
            failures.append(f"{name}_must_equal_{expected}")
    if receipt.get("source_checkout_guard_passed") is not True:
        failures.append("source_checkout_guard_not_passed")
    return {
        "status": "admitted" if not failures else "blocked",
        "failures": failures,
        "fit_authorized": not failures,
        "synthetic_or_template_completion_sufficient": False,
        "historical_claim_flags_modified": False,
    }


def training_window_weights(
    *,
    wearer_ids: Sequence[str],
    class_indices: Any,
    condition_ids: Sequence[str],
    block_ids: Sequence[str],
    bout_ids: Sequence[str],
) -> FloatArray:
    """Return equal wearer/class/condition/block/bout-window weights summing to one."""

    classes = np.asarray(class_indices)
    size = len(wearer_ids)
    if (
        classes.shape != (size,)
        or len(condition_ids) != size
        or len(block_ids) != size
        or len(bout_ids) != size
    ):
        raise ReferenceTrainingError("training metadata must be row aligned")
    if size == 0 or not np.all(np.isin(classes, [0, 1, 2])):
        raise ReferenceTrainingError("training classes must be nonempty indices 0,1,2")
    wearers = tuple(str(value) for value in wearer_ids)
    unique_wearers = sorted(set(wearers))
    if len(unique_wearers) != 5:
        raise ReferenceTrainingError("each outer-training fold requires exactly five wearers")
    by_bout: dict[str, list[int]] = defaultdict(list)
    for index, bout in enumerate(bout_ids):
        by_bout[str(bout)].append(index)
    weights = np.zeros(size, dtype=np.float64)
    for wearer in unique_wearers:
        for class_index in range(3):
            expected_conditions = CLASS_CONDITIONS[class_index]
            observed_conditions = {
                str(condition_ids[index])
                for index in range(size)
                if wearers[index] == wearer and int(classes[index]) == class_index
            }
            if observed_conditions != set(expected_conditions):
                raise ReferenceTrainingError(
                    "training condition roster is incomplete or unexpected"
                )
            for condition in expected_conditions:
                selected_bouts = sorted(
                    {
                        str(bout_ids[index])
                        for index in range(size)
                        if wearers[index] == wearer
                        and int(classes[index]) == class_index
                        and condition_ids[index] == condition
                    }
                )
                if len(selected_bouts) != 4:
                    raise ReferenceTrainingError(
                        "each wearer/class/condition requires four physical bouts"
                    )
                if len({block_ids[by_bout[bout][0]] for bout in selected_bouts}) != 4:
                    raise ReferenceTrainingError(
                        "condition bouts must come from four attachment blocks"
                    )
                for bout in selected_bouts:
                    indices = by_bout[bout]
                    if any(
                        wearers[index] != wearer
                        or int(classes[index]) != class_index
                        or condition_ids[index] != condition
                        for index in indices
                    ):
                        raise ReferenceTrainingError("bout identifier crosses a metadata group")
                    value = 1.0 / (
                        len(unique_wearers) * 3 * len(expected_conditions) * 4 * len(indices)
                    )
                    weights[indices] = value
    if np.any(weights <= 0.0) or not math.isclose(float(weights.sum()), 1.0, abs_tol=1.0e-12):
        raise ReferenceTrainingError("training weights do not cover the exact schedule")
    return weights


def fixed_class_macro_f1(labels: IntArray, predictions: IntArray, weights: FloatArray) -> float:
    if labels.shape != predictions.shape or weights.shape != labels.shape:
        raise ReferenceTrainingError("metric arrays must align")
    scores = []
    for class_index in range(3):
        true_positive = float(weights[(labels == class_index) & (predictions == class_index)].sum())
        false_positive = float(
            weights[(labels != class_index) & (predictions == class_index)].sum()
        )
        false_negative = float(
            weights[(labels == class_index) & (predictions != class_index)].sum()
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0.0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def evaluate_equal_bout(
    *,
    wearer_ids: Sequence[str],
    bout_ids: Sequence[str],
    labels: Any,
    probabilities: Any,
) -> dict[str, Any]:
    target = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    size = len(wearer_ids)
    if target.shape != (size,) or probability.shape != (size, 3) or len(bout_ids) != size:
        raise ReferenceTrainingError("evaluation inputs must be row aligned")
    if (
        not np.isfinite(probability).all()
        or np.any(probability < 0.0)
        or not np.allclose(probability.sum(axis=1), 1.0)
    ):
        raise ReferenceTrainingError("evaluation probabilities are invalid")
    wearers = sorted(set(str(value) for value in wearer_ids))
    if len(wearers) != 6:
        raise ReferenceTrainingError("primary evaluation requires exactly six wearers")
    participant_rows: list[dict[str, Any]] = []
    for wearer in wearers:
        indices = [index for index, value in enumerate(wearer_ids) if str(value) == wearer]
        wearer_bouts = sorted({str(bout_ids[index]) for index in indices})
        if len(wearer_bouts) != 28:
            raise ReferenceTrainingError("each wearer must contribute exactly 28 query bouts")
        weights = np.zeros(size, dtype=np.float64)
        for bout in wearer_bouts:
            rows = [index for index in indices if str(bout_ids[index]) == bout]
            if not rows:
                raise ReferenceTrainingError("empty evaluation bout")
            weights[rows] = 1.0 / (28 * len(rows))
        predictions = np.argmax(probability, axis=1).astype(np.int64)
        score = fixed_class_macro_f1(target[indices], predictions[indices], weights[indices])
        recalls = {}
        for class_index, name in enumerate(CLASS_NAMES):
            denominator = float(
                weights[(target == class_index) & (np.isin(np.arange(size), indices))].sum()
            )
            numerator = float(
                weights[
                    (target == class_index)
                    & (predictions == class_index)
                    & (np.isin(np.arange(size), indices))
                ].sum()
            )
            recalls[name] = 0.0 if denominator == 0.0 else numerator / denominator
        participant_rows.append({"wearer_id": wearer, "macro_f1": score, "class_recalls": recalls})
    values = np.array([row["macro_f1"] for row in participant_rows], dtype=np.float64)
    return {
        "participant_rows": participant_rows,
        "mean_participant_macro_f1": float(values.mean()),
        "bottom_two_participant_macro_f1": float(np.sort(values)[:2].mean()),
        "primary_weighting": "equal_wearer_equal_bout_equal_window_within_bout",
    }


def exhaustive_six_wearer_interval(differences: Any) -> list[float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ReferenceTrainingError("exhaustive interval requires six finite wearer differences")
    indices = np.asarray(list(itertools.product(range(6), repeat=6)), dtype=np.int64)
    draws = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975], method="linear")]


def compare_reports(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    candidate_rows = {str(row["wearer_id"]): row for row in candidate["participant_rows"]}
    baseline_rows = {str(row["wearer_id"]): row for row in baseline["participant_rows"]}
    if set(candidate_rows) != set(baseline_rows) or len(candidate_rows) != 6:
        raise ReferenceTrainingError("paired reports require the same six wearers")
    order = sorted(candidate_rows)
    differences = np.array(
        [
            float(candidate_rows[name]["macro_f1"]) - float(baseline_rows[name]["macro_f1"])
            for name in order
        ]
    )
    recall_differences = {
        class_name: float(
            np.mean(
                [
                    float(candidate_rows[name]["class_recalls"][class_name])
                    - float(baseline_rows[name]["class_recalls"][class_name])
                    for name in order
                ]
            )
        )
        for class_name in CLASS_NAMES
    }
    return {
        "wearer_order": order,
        "wearer_differences": differences.tolist(),
        "mean_difference": float(differences.mean()),
        "paired_95_percent_exhaustive_bootstrap_interval": exhaustive_six_wearer_interval(
            differences
        ),
        "strict_wins": int(np.sum(differences > 0.0)),
        "ties": int(np.sum(differences == 0.0)),
        "harms": int(np.sum(differences < 0.0)),
        "minimum_wearer_difference": float(differences.min()),
        "leave_one_wearer_out_mean_differences": [
            float(np.delete(differences, index).mean()) for index in range(6)
        ],
        "bottom_two_difference": float(candidate["bottom_two_participant_macro_f1"])
        - float(baseline["bottom_two_participant_macro_f1"]),
        "class_recall_differences": recall_differences,
    }


def practical_gate(comparisons: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    required = ("Z", "A", "MAP")
    failures: list[str] = []
    for baseline in required:
        row = comparisons.get(baseline)
        if row is None:
            failures.append(f"{baseline}:comparison_missing")
            continue
        checks = {
            "mean_gain_at_least_3pp": float(row["mean_difference"]) >= 0.03,
            "interval_lower_positive": float(
                row["paired_95_percent_exhaustive_bootstrap_interval"][0]
            )
            > 0.0,
            "five_of_six_wins": int(row["strict_wins"]) >= 5,
            "all_leave_one_out_positive": all(
                float(value) > 0.0 for value in row["leave_one_wearer_out_mean_differences"]
            ),
            "maximum_harm_within_3pp": float(row["minimum_wearer_difference"]) >= -0.03,
            "bottom_two_not_worse": float(row["bottom_two_difference"]) >= 0.0,
            "mobility_recall_within_1pp": float(row["class_recall_differences"]["mobility"])
            >= -0.01,
            "sitting_recall_within_2pp": float(row["class_recall_differences"]["sitting"]) >= -0.02,
            "standing_recall_within_2pp": float(row["class_recall_differences"]["standing"])
            >= -0.02,
        }
        failures.extend(f"{baseline}:{name}" for name, passed in checks.items() if not passed)
    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "comparators": list(required),
        "adaptive_best_arm_substitution_allowed": False,
    }


def schedule_receipt() -> dict[str, Any]:
    counts = Counter(attempt.object_kind for attempt in FIT_SCHEDULE)
    return {
        "total_fitted_optimization_objects": len(FIT_SCHEDULE),
        "counts": dict(sorted(counts.items())),
        "arm_fold_outputs": len(ARM_IDS) * 6,
        "encoder_fits": 0,
        "attempts": [attempt.__dict__ for attempt in FIT_SCHEDULE],
    }
