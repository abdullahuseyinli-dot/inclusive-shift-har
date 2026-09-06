"""Frozen Sole-HARmony observable-session temporal evaluation.

The primary lane predicts every label-independent session candidate with the fixed
XGBoost-6ch outer-fold control, updates causal state on that complete stream, and
only then slices homogeneous annotated windows for scoring. The historical camera-
bout materialization remains available solely as a separately written oracle-boundary
diagnostic; its numbers are never a matched comparator for the observable lane.
"""

from __future__ import annotations

import argparse
import json
import re
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from numpy.typing import NDArray
from sklearn.metrics import (  # type: ignore[import-untyped]
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from inclusive_shift_har.artifacts.research_provenance import (
    _external_evidence_status,
    _publication_artifact_contract,
    _publication_launch_context_binding,
    _resolve_publication_launch_context,
    _runtime_environment,
    _source_manifest_commit_errors,
    _write_launch_failure_envelope,
    _write_self_hashed_json_create_only,
)
from inclusive_shift_har.data.external_har import (
    CORE_CLASS_NAMES,
    OBSERVABLE_SCORING_ELIGIBILITY_POLICY,
    ExternalHARWindows,
    _array_sha256,
    load_sole_harmony,
    observable_modelling_pool,
)
from inclusive_shift_har.evaluation.external_statistics import seed_evidence
from inclusive_shift_har.evaluation.inference_contracts import OBSERVABLE_CONTEXT_PROTOCOL
from inclusive_shift_har.evaluation.metrics import classification_report, expected_calibration_error
from inclusive_shift_har.experiments.cross_dataset_har import (
    _git_state,
    _observable_classical_seed_predictions,
    _write_json_create_only,
)
from inclusive_shift_har.experiments.cross_dataset_har import (
    _source_input_manifest as _source_input_manifest,
)
from inclusive_shift_har.experiments.external_evidence_validate import (
    validate_and_record_run_directory,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.hera_ctgr_v2 import accumulate_bout_intervention_evidence

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]

PROTOCOL_ID = "sole-harmony-observable-session-temporal-v1"
PROTOCOL_PATH = "configs/protocols/sole_harmony_observable_session_temporal_v1.yaml"
BASE_METHOD = "XGBoost-6ch-unsmoothed"
PRIMARY_METHOD = "XGBoost-6ch-causal-probability-w5"
_WIDTHS = (3, 5, 7)
_HYSTERESIS_CONFIRMATIONS = (2, 3)
_SHUFFLE_REPLICATES = 100
_BOOTSTRAP_REPLICATES = 10_000
_BOOTSTRAP_SEED = 20260905


def _temporal_report(
    labels: IntArray,
    probability: FloatArray,
    participants: StringArray,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    """Build a Sole report with the frozen integer-count lower-tail estimand."""

    report = classification_report(
        labels,
        probability,
        participants.tolist(),
        class_names=class_names,
    )
    participant_rows = cast(list[dict[str, Any]], report["participants"])
    values = np.sort(
        np.asarray([float(row["macro_f1"]) for row in participant_rows], dtype=np.float64)
    )
    if values.size == 0:
        raise ValueError("Sole temporal report requires at least one participant")
    tail_count = max(1, int(np.ceil(0.30 * values.size)))
    primary = cast(dict[str, Any], report["primary"])
    primary["bottom_30_percent_participant_macro_f1"] = float(values[:tail_count].mean())
    return report


def _read_protocol(repository_root: Path) -> dict[str, Any]:
    path = repository_root / PROTOCOL_PATH
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("Sole-HARmony temporal protocol must be an object")
    protocol = cast(dict[str, Any], value)
    budget = cast(dict[str, Any], protocol.get("budget", {}))
    methods = cast(dict[str, Any], protocol.get("methods", {}))
    cohort = cast(dict[str, Any], protocol.get("cohort", {}))
    preprocessing = cast(dict[str, Any], protocol.get("preprocessing", {}))
    base_model = cast(dict[str, Any], protocol.get("base_model", {}))
    statistics = cast(dict[str, Any], protocol.get("statistics", {}))
    evidence_role = cast(dict[str, Any], protocol.get("evidence_role", {}))
    expected_gate = (
        "primary paired mean 95-percent CI lower bound strictly greater than zero",
        "secondary independently ranked bottom-30-percent distribution-contrast "
        "95-percent CI lower bound at least zero",
        "mobility and sitting recall do not worsen",
        "participant-mean NLL does not worsen",
        "causal result exceeds the median of its shuffled-time negative controls",
        "no oracle-boundary or excluded-window shortcut contributes to the result",
    )
    if (
        protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("status") != "FROZEN_BEFORE_SESSION_OBSERVABLE_IMPLEMENTATION_OR_OUTCOMES"
        or evidence_role.get("role") != "development temporal diagnostic"
        or evidence_role.get("confirmation_allowed") is not False
        or evidence_role.get("camera_bout_lane_role")
        != "oracle-boundary upper-bound diagnostic only"
        or cohort.get("participants") != "ordered provider participants C001-C012"
        or cohort.get("sessions_per_participant") != "first two ordered sessions"
        or cohort.get("independent_unit") != "participant"
        or cohort.get("repetitions_or_sessions_increase_independent_n") is not False
        or preprocessing.get("observable_lane") != "session_observable"
        or preprocessing.get("window_samples") != 128
        or preprocessing.get("target_sampling_rate_hz") != 50
        or preprocessing.get("stride_samples") != 128
        or base_model.get("primary") != "XGBoost-6ch"
        or base_model.get("rationale")
        != (
            "Representative high-capacity six-channel tree control fixed before v4 "
            "replacement outcomes; this predeclaration makes no strongest-control claim."
        )
        or base_model.get("outer_training_participant_labels_used_for_supervised_fit") is not True
        or base_model.get(
            "outer_evaluation_participant_labels_used_for_fit_selection_or_calibration"
        )
        is not False
        or tuple(budget.get("seeds", ())) != (11, 23, 47)
        or budget.get("outer_folds") != 5
        or budget.get("hyperparameter_trials") != 0
        or budget.get("candidate_redesigns_after_outcomes") != 0
        or tuple(budget.get("temporal_widths", ())) != _WIDTHS
        or tuple(budget.get("hysteresis_confirmation_windows", ())) != _HYSTERESIS_CONFIRMATIONS
        or budget.get("shuffled_time_replicates_per_probability_width") != _SHUFFLE_REPLICATES
        or methods.get("primary_control") != BASE_METHOD
        or methods.get("primary_candidate") != PRIMARY_METHOD
        or methods.get("future_window_access") is not False
        or statistics.get("primary_metric")
        != "mean across seeds of equally weighted participant fixed-three-class macro-F1"
        or statistics.get("uncertainty") != "paired participant-cluster bootstrap, 10000 draws"
        or statistics.get("lower_tail")
        != {
            "per_method_summary": "bottom ceil(0.30*N) participant macro-F1",
            "advancement_contrast": (
                "Secondary jointly participant-resampled, independently ranked difference "
                "between candidate and control bottom-ceil(0.30*N) means. This is a "
                "lower-tail distribution contrast, not a paired fixed-participant effect."
            ),
        }
        or tuple(protocol.get("advancement_gate", ())) != expected_gate
    ):
        raise ValueError("Sole-HARmony frozen protocol differs from the implemented contract")
    return protocol


def _temporal_block_ids(data: ExternalHARWindows) -> StringArray:
    """Return the physical run prefix encoded before each deterministic window ID."""

    identifiers: list[str] = []
    for value in data.window_ids.tolist():
        prefix, separator, suffix = str(value).rpartition("/window-")
        if (
            not separator
            or not prefix
            or re.fullmatch(r"[0-9]{6}", suffix) is None
            or re.search(r"/run-[0-9]{4}-finite-[0-9]{3}$", prefix) is None
        ):
            raise ValueError("Sole-HARmony window identifiers lack physical-run provenance")
        identifiers.append(prefix)
    return np.asarray(identifiers, dtype=np.str_)


def _block_slices(block_ids: StringArray) -> tuple[IntArray, ...]:
    if block_ids.ndim != 1 or block_ids.size == 0:
        raise ValueError("Sole-HARmony temporal block identifiers must be a non-empty vector")
    boundaries = np.flatnonzero(np.concatenate(([True], block_ids[1:] != block_ids[:-1])))
    stops = np.concatenate((boundaries[1:], np.array([block_ids.size], dtype=np.int64)))
    slices = tuple(
        np.arange(start, stop, dtype=np.int64)
        for start, stop in zip(boundaries, stops, strict=True)
    )
    if len({str(block_ids[indices[0]]) for indices in slices}) != len(slices):
        raise ValueError("a temporal block appears in multiple non-contiguous stream positions")
    return slices


def _validate_probability_stream(probabilities: FloatArray, block_ids: StringArray) -> FloatArray:
    values = np.asarray(probabilities, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] != block_ids.size
        or values.shape[1] != len(CORE_CLASS_NAMES)
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
    ):
        raise ValueError("temporal probabilities must be aligned finite three-class rows")
    _block_slices(block_ids)
    return values


def _causal_probability(
    probabilities: FloatArray,
    bout_ids: StringArray,
    *,
    width: int,
) -> FloatArray:
    """Apply a trailing probability mean within each observable physical block."""

    values = _validate_probability_stream(probabilities, bout_ids)
    smoothed = np.column_stack(
        [
            accumulate_bout_intervention_evidence(values[:, class_index], bout_ids, width=width)
            for class_index in range(values.shape[1])
        ]
    )
    return np.asarray(smoothed / smoothed.sum(axis=1, keepdims=True), dtype=np.float64)


def _causal_majority_probability(
    probabilities: FloatArray,
    block_ids: StringArray,
    *,
    width: int,
) -> FloatArray:
    """Return causal hard-label majority decisions as explicit one-hot rows.

    The most recent label wins a count tie. This fixed tie rule prevents class-index
    ordering from becoming an accidental control advantage.
    """

    values = _validate_probability_stream(probabilities, block_ids)
    if width < 1:
        raise ValueError("causal majority width must be positive")
    hard = values.argmax(axis=1)
    output = np.zeros_like(values)
    for indices in _block_slices(block_ids):
        for local_index, index in enumerate(indices.tolist()):
            left = max(0, local_index - width + 1)
            history = hard[indices[left : local_index + 1]]
            counts = np.bincount(history, minlength=values.shape[1])
            tied = np.flatnonzero(counts == counts.max())
            latest = int(history[-1])
            selected = latest if latest in tied else int(tied[0])
            output[index, selected] = 1.0
    return output


def _causal_hysteresis_probability(
    probabilities: FloatArray,
    block_ids: StringArray,
    *,
    confirmation_windows: int,
) -> FloatArray:
    """Switch hard state only after consecutive causal challenger confirmations."""

    values = _validate_probability_stream(probabilities, block_ids)
    if confirmation_windows < 2:
        raise ValueError("hysteresis requires at least two confirmation windows")
    hard = values.argmax(axis=1)
    output = np.zeros_like(values)
    for indices in _block_slices(block_ids):
        state = int(hard[indices[0]])
        challenger = -1
        count = 0
        for index in indices.tolist():
            proposed = int(hard[index])
            if proposed == state:
                challenger = -1
                count = 0
            else:
                if proposed == challenger:
                    count += 1
                else:
                    challenger = proposed
                    count = 1
                if count >= confirmation_windows:
                    state = challenger
                    challenger = -1
                    count = 0
            output[index, state] = 1.0
    return output


def _shuffled_time_probability(
    probabilities: FloatArray,
    block_ids: StringArray,
    *,
    width: int,
    seed: int,
) -> FloatArray:
    """Destroy within-block time order, smooth causally, then restore item identity."""

    values = _validate_probability_stream(probabilities, block_ids)
    generator = np.random.default_rng(seed)
    result = np.empty_like(values)
    for indices in _block_slices(block_ids):
        order = generator.permutation(indices.size)
        shuffled = values[indices][order]
        shuffled_ids = np.full(indices.size, str(block_ids[indices[0]]), dtype=np.str_)
        smoothed = _causal_probability(shuffled, shuffled_ids, width=width)
        result[indices[order]] = smoothed
    return result


def _temporal_methods(base: FloatArray, block_ids: StringArray) -> dict[str, FloatArray]:
    methods = {BASE_METHOD: np.asarray(base, dtype=np.float64).copy()}
    for width in _WIDTHS:
        methods[f"XGBoost-6ch-causal-probability-w{width}"] = _causal_probability(
            base, block_ids, width=width
        )
        methods[f"XGBoost-6ch-causal-majority-w{width}"] = _causal_majority_probability(
            base, block_ids, width=width
        )
    for confirmation in _HYSTERESIS_CONFIRMATIONS:
        methods[f"XGBoost-6ch-hysteresis-c{confirmation}"] = _causal_hysteresis_probability(
            base,
            block_ids,
            confirmation_windows=confirmation,
        )
    return methods


def _seed_averaged_participant_score(
    data: ExternalHARWindows, per_seed: dict[int, FloatArray]
) -> float:
    participants = np.unique(data.participant_ids)
    scores: list[float] = []
    class_indices = np.arange(len(data.class_names), dtype=np.int64)
    for seed in sorted(per_seed):
        predicted = per_seed[seed].argmax(axis=1)
        for participant in participants:
            selected = data.participant_ids == participant
            scores.append(
                float(
                    f1_score(
                        data.labels[selected],
                        predicted[selected],
                        labels=class_indices,
                        average="macro",
                        zero_division=0,
                    )
                )
            )
    return float(np.mean(scores))


def _participant_distribution(
    values: FloatArray,
    participants: StringArray,
    *,
    eligible: NDArray[np.bool_] | None = None,
) -> dict[str, Any]:
    """Summarize a seed-averaged participant vector with participant-cluster uncertainty."""

    if values.ndim != 1 or values.shape != participants.shape or not np.isfinite(values).all():
        raise ValueError("participant distribution requires aligned finite vectors")
    selected_mask = (
        np.ones(values.shape, dtype=np.bool_)
        if eligible is None
        else np.asarray(eligible, dtype=np.bool_)
    )
    if selected_mask.shape != values.shape:
        raise ValueError("participant eligibility must align with values")
    selected_values = values[selected_mask]
    selected_participants = participants[selected_mask]
    base = {
        "participant_n": int(selected_values.size),
        "eligible_participants": selected_participants.tolist(),
        "ineligible_participants": participants[~selected_mask].tolist(),
        "participant_values": dict(
            zip(selected_participants.tolist(), selected_values.tolist(), strict=True)
        ),
    }
    if selected_values.size == 0:
        return {**base, "summary": None}
    draws = np.random.default_rng(_BOOTSTRAP_SEED).integers(
        0,
        selected_values.size,
        size=(_BOOTSTRAP_REPLICATES, selected_values.size),
    )
    quartiles = np.quantile(selected_values, [0.25, 0.50, 0.75])
    tail_count = max(1, int(np.ceil(0.30 * selected_values.size)))
    return {
        **base,
        "summary": {
            "mean": float(selected_values.mean()),
            "median": float(quartiles[1]),
            "quartiles_25_50_75": quartiles.tolist(),
            "minimum": float(selected_values.min()),
            "maximum": float(selected_values.max()),
            "bottom_30_percent_mean": float(np.sort(selected_values)[:tail_count].mean()),
            "participant_cluster_bootstrap_95_percent_ci": np.quantile(
                selected_values[draws].mean(axis=1), [0.025, 0.975]
            ).tolist(),
        },
    }


def _participant_statistical_supplement(
    data: ExternalHARWindows,
    per_seed: dict[int, dict[str, FloatArray]],
) -> dict[str, Any]:
    """Participant-balanced fixed-class, sensitivity, confusion, and calibration summaries."""

    if not per_seed:
        raise ValueError("temporal statistics require at least one seed")
    seeds = sorted(per_seed)
    method_names = sorted(per_seed[seeds[0]])
    if not method_names or any(set(per_seed[seed]) != set(method_names) for seed in seeds):
        raise ValueError("every temporal seed must retain the same non-empty method set")
    participants = np.unique(data.participant_ids)
    class_count = len(data.class_names)
    class_indices = np.arange(class_count, dtype=np.int64)
    support = np.asarray(
        [
            np.bincount(data.labels[data.participant_ids == participant], minlength=class_count)
            for participant in participants
        ],
        dtype=np.int64,
    )
    present = support > 0
    complete = np.asarray(present.all(axis=1), dtype=np.bool_)
    methods: dict[str, Any] = {}
    for method in method_names:
        precision_values = np.zeros((len(seeds), participants.size, class_count), dtype=np.float64)
        recall_values = np.zeros_like(precision_values)
        f1_values = np.zeros_like(precision_values)
        normalized_confusions = np.zeros(
            (len(seeds), participants.size, class_count, class_count), dtype=np.float64
        )
        nll_values = np.zeros((len(seeds), participants.size), dtype=np.float64)
        brier_values = np.zeros_like(nll_values)
        ece_values = np.zeros_like(nll_values)
        for seed_index, seed in enumerate(seeds):
            probability = np.asarray(per_seed[seed][method], dtype=np.float64)
            if (
                probability.shape != (data.labels.size, class_count)
                or not np.isfinite(probability).all()
                or np.any(probability < 0.0)
                or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
            ):
                raise ValueError(f"invalid scored probability rows for seed {seed}, {method}")
            predicted = probability.argmax(axis=1)
            for participant_index, participant in enumerate(participants):
                selected = data.participant_ids == participant
                labels = data.labels[selected]
                participant_probability = probability[selected]
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels,
                    predicted[selected],
                    labels=class_indices,
                    zero_division=0,
                )
                precision_values[seed_index, participant_index] = precision
                recall_values[seed_index, participant_index] = recall
                f1_values[seed_index, participant_index] = f1
                matrix = confusion_matrix(labels, predicted[selected], labels=class_indices).astype(
                    np.float64
                )
                row_sums = matrix.sum(axis=1, keepdims=True)
                normalized_confusions[seed_index, participant_index] = np.divide(
                    matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0.0
                )
                selected_probability = np.clip(
                    participant_probability[np.arange(labels.size), labels], 1e-12, 1.0
                )
                nll_values[seed_index, participant_index] = float(
                    -np.log(selected_probability).mean()
                )
                one_hot = np.eye(class_count, dtype=np.float64)[labels]
                brier_values[seed_index, participant_index] = float(
                    np.sum((participant_probability - one_hot) ** 2, axis=1).mean()
                )
                ece_values[seed_index, participant_index] = expected_calibration_error(
                    labels, participant_probability, bins=15
                )

        seed_averaged_precision = precision_values.mean(axis=0)
        seed_averaged_recall = recall_values.mean(axis=0)
        seed_averaged_f1 = f1_values.mean(axis=0)
        fixed_macro_f1 = seed_averaged_f1.mean(axis=1)
        present_macro_f1 = np.divide(
            np.where(present, seed_averaged_f1, 0.0).sum(axis=1),
            present.sum(axis=1),
        )
        per_class: dict[str, Any] = {}
        for class_index, name in enumerate(data.class_names):
            eligible = present[:, class_index]
            per_class[name] = {
                "precision": float(seed_averaged_precision[:, class_index].mean()),
                "recall": float(seed_averaged_recall[:, class_index].mean()),
                "f1": float(seed_averaged_f1[:, class_index].mean()),
                "fixed_cohort_zero_division_0": True,
                "true_support_window_count": int(support[:, class_index].sum()),
                "participants_with_true_support": int(eligible.sum()),
                "participants_without_true_support": participants[~eligible].tolist(),
                "true_support_eligible_participant_sensitivity": {
                    "participant_n": int(eligible.sum()),
                    "precision": float(seed_averaged_precision[eligible, class_index].mean())
                    if eligible.any()
                    else None,
                    "recall": float(seed_averaged_recall[eligible, class_index].mean())
                    if eligible.any()
                    else None,
                    "f1": float(seed_averaged_f1[eligible, class_index].mean())
                    if eligible.any()
                    else None,
                },
            }
        seed_averaged_confusion = normalized_confusions.mean(axis=0)
        confusion_rows: list[list[float] | None] = []
        for class_index in range(class_count):
            eligible = present[:, class_index]
            confusion_rows.append(
                seed_averaged_confusion[eligible, class_index, :].mean(axis=0).tolist()
                if eligible.any()
                else None
            )
        mean_nll = nll_values.mean(axis=0)
        mean_brier = brier_values.mean(axis=0)
        mean_ece = ece_values.mean(axis=0)
        methods[method] = {
            "fixed_cohort_participant_count": int(participants.size),
            "eligible_participant_count": int(participants.size),
            "eligible_participant_count_definition": (
                "participants with at least one scored window; class-specific true-support "
                "eligibility is reported separately for every class"
            ),
            "participant_seed_observation_count": int(len(seeds) * participants.size),
            "fixed_class_participant_macro_f1": _participant_distribution(
                fixed_macro_f1, participants
            ),
            "present_true_class_participant_macro_f1_sensitivity": (
                _participant_distribution(present_macro_f1, participants)
            ),
            "all_declared_classes_supported_participant_sensitivity": (
                _participant_distribution(fixed_macro_f1, participants, eligible=complete)
            ),
            "per_class_participant_balanced": per_class,
            "participant_normalized_confusion": confusion_rows,
            "participant_normalized_confusion_eligible_n_by_true_class": dict(
                zip(data.class_names, present.sum(axis=0).tolist(), strict=True)
            ),
            "participant_mean_calibration": {
                "negative_log_likelihood": float(mean_nll.mean()),
                "multiclass_brier_score": float(mean_brier.mean()),
                "ece": float(mean_ece.mean()),
                "negative_log_likelihood_participant_distribution": (
                    _participant_distribution(mean_nll, participants)
                ),
                "multiclass_brier_participant_distribution": (
                    _participant_distribution(mean_brier, participants)
                ),
                "ece_participant_distribution": _participant_distribution(mean_ece, participants),
                "ece_bins": 15,
                "ece_note": (
                    "ECE is computed within participant and seed before equal-participant "
                    "aggregation; with small participant sample sizes it is a secondary, "
                    "sample-size-sensitive diagnostic"
                ),
                "hard_control_note": (
                    "majority and hysteresis emit deterministic one-hot decisions; their "
                    "calibration values are degenerate diagnostics, not calibrated probabilities"
                    if "majority" in method or "hysteresis" in method
                    else None
                ),
            },
        }
    return {
        "statistical_unit": "participant",
        "seeds": seeds,
        "participant_count": int(participants.size),
        "aggregation": (
            "average frozen seeds within participant first, then give each participant "
            "equal weight; participant bootstrap retains all seeds within each cluster"
        ),
        "fixed_class_primary_cross_reference": "primary_seed_averaged",
        "perfect_prediction_fixed_class_mean_ceiling": float(present.mean(axis=1).mean()),
        "participants_with_all_declared_classes": int(complete.sum()),
        "missing_class_rule": (
            "Primary fixed-three-class macro-F1 retains zero for undefined classes. "
            "Present-true-class and all-declared-classes-supported results are secondary, "
            "label-defined sensitivities and never replace the primary."
        ),
        "confusion_estimand": (
            "Normalize each true-class row within participant and seed; average seeds "
            "within participant; then average only participants with true support for that row."
        ),
        "methods": methods,
    }


def _paired_primary_comparison(primary: dict[str, Any]) -> dict[str, Any]:
    methods = cast(dict[str, Any], primary["methods"])
    candidate_map = cast(dict[str, float], methods[PRIMARY_METHOD]["participant_values"])
    control_map = cast(dict[str, float], methods[BASE_METHOD]["participant_values"])
    if set(candidate_map) != set(control_map):
        raise ValueError("primary temporal comparison participant sets differ")
    participants = sorted(candidate_map)
    if not participants:
        raise ValueError("primary temporal comparison requires participants")
    candidate = np.asarray([candidate_map[item] for item in participants], dtype=np.float64)
    control = np.asarray([control_map[item] for item in participants], dtype=np.float64)
    if not np.isfinite(candidate).all() or not np.isfinite(control).all():
        raise ValueError("primary temporal comparison values must be finite")
    difference = candidate - control
    generator = np.random.default_rng(_BOOTSTRAP_SEED)
    draw = generator.integers(0, len(participants), size=(_BOOTSTRAP_REPLICATES, len(participants)))
    mean_draw = difference[draw].mean(axis=1)
    tail_count = max(1, int(np.ceil(0.30 * len(participants))))
    candidate_tail = np.sort(candidate[draw], axis=1)[:, :tail_count].mean(axis=1)
    control_tail = np.sort(control[draw], axis=1)[:, :tail_count].mean(axis=1)
    tail_draw = candidate_tail - control_tail
    tolerance = 1e-12
    return {
        "candidate": PRIMARY_METHOD,
        "comparator": BASE_METHOD,
        "difference_direction": "candidate_minus_comparator",
        "participant_count": len(participants),
        "paired_participant_ids_sha256": canonical_json_sha256(participants),
        "participant_difference_values": dict(zip(participants, difference.tolist(), strict=True)),
        "mean_difference": float(difference.mean()),
        "paired_participant_bootstrap_95_percent_ci": np.quantile(
            mean_draw, [0.025, 0.975]
        ).tolist(),
        "secondary_independently_ranked_bottom_30_percent_distribution_difference": float(
            np.sort(candidate)[:tail_count].mean() - np.sort(control)[:tail_count].mean()
        ),
        "secondary_independently_ranked_bottom_30_percent_distribution_difference_95_percent_ci": np.quantile(
            tail_draw, [0.025, 0.975]
        ).tolist(),
        "rescue_count": int(np.sum(difference > tolerance)),
        "harm_count": int(np.sum(difference < -tolerance)),
        "tie_count": int(np.sum(np.abs(difference) <= tolerance)),
        "tie_tolerance_absolute": tolerance,
        "secondary_tail_estimand": (
            "Jointly resample aligned participants, independently rank candidate and "
            "comparator scores within each draw, and subtract bottom-ceil(0.30*N) means; "
            "this contrasts lower-tail distributions rather than a fixed-person tail subset"
        ),
        "seed_aggregation_before_bootstrap": (
            "all frozen seed scores are averaged within participant before participant draws"
        ),
        "bootstrap_replicates": _BOOTSTRAP_REPLICATES,
        "bootstrap_seed": _BOOTSTRAP_SEED,
    }


def _advancement_gate(
    *,
    boundary_mode: str,
    primary: dict[str, Any],
    supplement: dict[str, Any],
    comparison: dict[str, Any],
    shuffled: dict[str, Any],
) -> dict[str, Any]:
    if boundary_mode != "session_observable":
        return {
            "applicable": False,
            "overall_passed": False,
            "reason": "camera-bout oracle evidence is not a matched advancement contrast",
            "candidate": PRIMARY_METHOD,
            "comparator": BASE_METHOD,
        }
    if (
        comparison.get("candidate") != PRIMARY_METHOD
        or comparison.get("comparator") != BASE_METHOD
        or comparison.get("difference_direction") != "candidate_minus_comparator"
        or comparison.get("participant_count") != 12
        or comparison.get("bootstrap_replicates") != _BOOTSTRAP_REPLICATES
        or comparison.get("bootstrap_seed") != _BOOTSTRAP_SEED
    ):
        raise ValueError("primary comparison does not match the frozen temporal contract")
    if (
        primary.get("participant_count") != 12
        or tuple(primary.get("seeds", ())) != (11, 23, 47)
        or supplement.get("statistical_unit") != "participant"
        or supplement.get("participant_count") != 12
    ):
        raise ValueError("participant statistics do not match the frozen temporal cohort")
    methods = cast(dict[str, Any], supplement["methods"])
    candidate = cast(dict[str, Any], methods[PRIMARY_METHOD])
    control = cast(dict[str, Any], methods[BASE_METHOD])
    candidate_class = cast(dict[str, Any], candidate["per_class_participant_balanced"])
    control_class = cast(dict[str, Any], control["per_class_participant_balanced"])
    candidate_calibration = cast(dict[str, Any], candidate["participant_mean_calibration"])
    control_calibration = cast(dict[str, Any], control["participant_mean_calibration"])
    for method_name, method_supplement in (
        (PRIMARY_METHOD, candidate),
        (BASE_METHOD, control),
    ):
        fixed_distribution = cast(
            dict[str, Any], method_supplement["fixed_class_participant_macro_f1"]
        )
        fixed_summary = cast(dict[str, Any], fixed_distribution["summary"])
        retained_method = cast(dict[str, Any], primary["methods"])[method_name]
        if not np.isclose(
            float(fixed_summary["mean"]),
            float(retained_method["mean_participant_macro_f1"]),
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(f"fixed-class participant statistic mismatch for {method_name}")
        present_distribution = cast(
            dict[str, Any],
            method_supplement["present_true_class_participant_macro_f1_sensitivity"],
        )
        present_summary = cast(dict[str, Any], present_distribution["summary"])
        if not np.isclose(
            float(present_summary["mean"]),
            float(retained_method["present_class_sensitivity_mean"]),
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(f"present-class participant statistic mismatch for {method_name}")
    mean_lower = float(comparison["paired_participant_bootstrap_95_percent_ci"][0])
    tail_lower = float(
        comparison[
            "secondary_independently_ranked_bottom_30_percent_distribution_difference_95_percent_ci"
        ][0]
    )
    mobility_difference = float(candidate_class["mobility"]["recall"]) - float(
        control_class["mobility"]["recall"]
    )
    sitting_difference = float(candidate_class["sitting"]["recall"]) - float(
        control_class["sitting"]["recall"]
    )
    nll_difference = float(candidate_calibration["negative_log_likelihood"]) - float(
        control_calibration["negative_log_likelihood"]
    )
    causal_score = float(primary["methods"][PRIMARY_METHOD]["mean_participant_macro_f1"])
    shuffled_width_5 = cast(dict[str, Any], shuffled.get("width_5", {}))
    shuffled_values = np.asarray(
        shuffled_width_5.get("replicate_seed_averaged_participant_macro_f1", ()),
        dtype=np.float64,
    )
    if (
        shuffled_width_5.get("replicate_count") != _SHUFFLE_REPLICATES
        or shuffled_values.shape != (_SHUFFLE_REPLICATES,)
        or not np.isfinite(shuffled_values).all()
        or not np.isclose(
            float(shuffled_width_5.get("causal_mean_participant_macro_f1", np.nan)),
            causal_score,
            atol=1e-12,
            rtol=0.0,
        )
        or not np.isclose(
            float(shuffled_width_5.get("shuffled_median", np.nan)),
            float(np.median(shuffled_values)),
            atol=1e-12,
            rtol=0.0,
        )
    ):
        raise ValueError("width-five shuffled-time evidence does not match the frozen contract")
    shuffled_median = float(np.median(shuffled_values))
    criteria: dict[str, dict[str, Any]] = {
        "paired_mean_ci_lower_strictly_above_zero": {
            "observed": mean_lower,
            "passed": mean_lower > 0.0,
        },
        "secondary_independently_ranked_bottom_30_ci_lower_at_least_zero": {
            "observed": tail_lower,
            "passed": tail_lower >= 0.0,
        },
        "mobility_recall_not_worse": {
            "observed_difference": mobility_difference,
            "passed": mobility_difference >= -1e-12,
        },
        "sitting_recall_not_worse": {
            "observed_difference": sitting_difference,
            "passed": sitting_difference >= -1e-12,
        },
        "participant_mean_nll_not_worse": {
            "observed_difference": nll_difference,
            "passed": nll_difference <= 1e-12,
        },
        "causal_exceeds_shuffled_time_median": {
            "causal": causal_score,
            "shuffled_median": shuffled_median,
            "passed": causal_score > shuffled_median,
        },
        "no_oracle_or_excluded_window_shortcut": {
            "passed": True,
            "basis": (
                "session-observable grid and full-candidate state update verified before "
                "scoring slice"
            ),
        },
    }
    return {
        "applicable": True,
        "candidate": PRIMARY_METHOD,
        "comparator": BASE_METHOD,
        "criteria": criteria,
        "overall_passed": all(bool(item["passed"]) for item in criteria.values()),
        "comparison_direction": "candidate_minus_comparator; positive favors causal width five",
        "recall_gate_estimand": (
            "seed-averaged participant-balanced recall over the fixed participant cohort; "
            "participants without true class support retain zero in this gate"
        ),
        "calibration_gate_estimand": (
            "seed-averaged within-participant NLL, equally averaged over participants; "
            "candidate minus comparator, so non-positive is not worse"
        ),
        "shuffled_control_gate_estimand": (
            "primary fixed-class participant macro-F1 exceeds the median of 100 "
            "predeclared within-block shuffled-time replicates"
        ),
        "stop_rule": "stop expansion after any failed criterion; do not select an alternate width",
    }


def _validate_lane(
    data: ExternalHARWindows,
    *,
    boundary_mode: str,
) -> tuple[ExternalHARWindows, IntArray, StringArray]:
    data.validate()
    if data.dataset_id != "sole_harmony_v1" or data.class_names != CORE_CLASS_NAMES:
        raise ValueError("temporal lane requires three-class Sole-HARmony")
    if data.sampling_rate_hz != 50.0 or data.signals.shape[1:] != (128, 6):
        raise ValueError("temporal lane requires the frozen 128-sample, 50-Hz, six-channel input")
    if data.participant_partition_plan is None:
        raise PermissionError("Sole-HARmony requires a pre-window participant partition plan")
    expected_participants = tuple(f"sole:C{index:03d}" for index in range(1, 13))
    if data.participant_partition_plan.participant_roster != expected_participants:
        raise ValueError("temporal protocol requires the frozen ordered C001-C012 cohort")
    boundary = data.boundary_provenance
    if not isinstance(boundary, dict) or boundary.get("boundary_mode") != boundary_mode:
        raise ValueError("Sole-HARmony boundary provenance does not match the requested lane")
    if boundary_mode == "session_observable":
        if (
            boundary.get("repository_signal_grid_annotation_independent") is not True
            or boundary.get("provider_upstream_annotation_conditioned") is not False
            or data.observable_candidates is None
        ):
            raise PermissionError("session-observable signal/grid isolation is not established")
    elif boundary_mode == "camera_bout_oracle":
        if (
            boundary.get("repository_signal_grid_annotation_independent") is not False
            or data.observable_candidates is not None
        ):
            raise PermissionError("camera-bout lane must remain explicitly annotation-conditioned")
    else:
        raise ValueError("unknown Sole-HARmony temporal boundary lane")

    modelling, scoring_indices, _eligibility = observable_modelling_pool(
        data, include_supervised_labels=True
    )
    participants = np.unique(modelling.participant_ids)
    if tuple(sorted(participants.tolist())) != expected_participants:
        raise ValueError("every frozen participant must have observable temporal candidates")
    if tuple(sorted(np.unique(data.participant_ids).tolist())) != expected_participants:
        raise ValueError("every frozen participant must contribute eligible scoring windows")
    if any(
        np.unique(modelling.session_ids[modelling.participant_ids == participant]).size != 2
        for participant in participants
    ):
        raise ValueError("temporal protocol requires exactly two sessions per participant")
    blocks = _temporal_block_ids(modelling)
    for indices in _block_slices(blocks):
        window_positions = np.asarray(
            [
                int(str(modelling.window_ids[index]).rsplit("/window-", 1)[1])
                for index in indices.tolist()
            ],
            dtype=np.int64,
        )
        if (
            np.unique(modelling.participant_ids[indices]).size != 1
            or np.unique(modelling.session_ids[indices]).size != 1
            or np.unique(modelling.trial_ids[indices]).size != 1
            or not np.array_equal(window_positions, np.arange(indices.size, dtype=np.int64))
        ):
            raise ValueError(
                "a temporal block crosses metadata or lacks its complete chronological grid"
            )
    if boundary_mode == "session_observable" and any(
        "camera-bout" in str(value) for value in blocks.tolist()
    ):
        raise PermissionError("session-observable temporal state contains an oracle camera reset")
    if boundary_mode == "session_observable" and any(
        str(trial) != f"{session}:session-recording"
        for session, trial in zip(
            modelling.session_ids.tolist(), modelling.trial_ids.tolist(), strict=True
        )
    ):
        raise PermissionError("session-observable state contains a non-session trial reset")
    if boundary_mode == "camera_bout_oracle" and any(
        "camera-bout-" not in str(value) for value in blocks.tolist()
    ):
        raise PermissionError("camera-bout oracle lane lacks explicit camera reset provenance")
    return modelling, scoring_indices, blocks


def evaluate_temporal_lane(
    data: ExternalHARWindows,
    *,
    seeds: tuple[int, ...],
    repository_root: Path,
    n_jobs: int,
) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    """Evaluate one explicit Sole-HARmony boundary lane under the frozen budget."""

    protocol = _read_protocol(repository_root)
    if seeds != tuple(protocol["budget"]["seeds"]):
        raise ValueError("Sole-HARmony temporal evaluation requires frozen seeds 11, 23, and 47")
    if n_jobs == 0:
        raise ValueError("n_jobs may not be zero")
    boundary = cast(dict[str, Any], data.boundary_provenance)
    boundary_mode = str(boundary.get("boundary_mode"))
    modelling, scoring_indices, temporal_blocks = _validate_lane(data, boundary_mode=boundary_mode)

    scored_by_seed: dict[int, dict[str, FloatArray]] = {}
    observable_by_seed: dict[int, dict[str, FloatArray]] = {}
    fold_records: list[dict[str, Any]] = []
    for seed in seeds:
        base = _observable_classical_seed_predictions(
            data,
            seed=seed,
            methods=("XGBoost-6ch",),
            outer_fold_count=5,
            inner_fold_count=4,
        )
        if not np.array_equal(base.data.window_ids, modelling.window_ids) or not np.array_equal(
            base.scoring_indices, scoring_indices
        ):
            raise AssertionError("per-seed observable population or scoring slice changed")
        temporal = _temporal_methods(base.probabilities["XGBoost-6ch"], temporal_blocks)
        observable_by_seed[seed] = temporal
        scored_by_seed[seed] = {
            method: probability[scoring_indices] for method, probability in temporal.items()
        }
        record = base.record
        record["temporal_state_input_window_count"] = int(modelling.labels.size)
        record["temporal_scoring_slice_window_count"] = int(scoring_indices.size)
        record["temporal_state_updated_on_every_observable_candidate"] = True
        record["temporal_labels_or_eligibility_used_before_state_update"] = False
        record["boundary_mode"] = boundary_mode
        for fold in cast(list[dict[str, Any]], record["folds"]):
            fold["temporal_state_updated_on_every_evaluation_candidate"] = True
            fold["evaluation_scoring_eligibility_used_before_prediction"] = False
        fold_records.append(record)

    primary, seed_prediction_archive = seed_evidence(
        data,
        scored_by_seed,
        primary_contrast_eligible=False,
    )
    ensemble = {
        method: np.mean(np.stack([scored_by_seed[seed][method] for seed in seeds], axis=0), axis=0)
        for method in scored_by_seed[seeds[0]]
    }
    observable_ensemble = {
        method: np.mean(
            np.stack([observable_by_seed[seed][method] for seed in seeds], axis=0), axis=0
        )
        for method in observable_by_seed[seeds[0]]
    }
    reports = {
        method: _temporal_report(
            data.labels,
            probability,
            data.participant_ids,
            data.class_names,
        )
        for method, probability in ensemble.items()
    }
    supplement = _participant_statistical_supplement(data, scored_by_seed)
    primary_comparison = _paired_primary_comparison(primary)

    shuffled_controls: dict[str, Any] = {}
    for width in _WIDTHS:
        replicate_scores: list[float] = []
        replicate_seed_schedule: list[dict[str, Any]] = []
        for replicate in range(_SHUFFLE_REPLICATES):
            shuffled_scored: dict[int, FloatArray] = {}
            random_seeds: dict[str, int] = {}
            for model_seed in seeds:
                random_seed = 202_609_050 + width * 100_000 + model_seed * 1_000 + replicate
                random_seeds[str(model_seed)] = random_seed
                shuffled_full = _shuffled_time_probability(
                    observable_by_seed[model_seed][BASE_METHOD],
                    temporal_blocks,
                    width=width,
                    seed=random_seed,
                )
                shuffled_scored[model_seed] = shuffled_full[scoring_indices]
            replicate_scores.append(_seed_averaged_participant_score(data, shuffled_scored))
            replicate_seed_schedule.append(
                {"replicate": replicate, "random_seeds_by_model_seed": random_seeds}
            )
        values = np.asarray(replicate_scores, dtype=np.float64)
        causal_name = f"XGBoost-6ch-causal-probability-w{width}"
        causal_score = float(primary["methods"][causal_name]["mean_participant_macro_f1"])
        shuffled_controls[f"width_{width}"] = {
            "replicate_count": _SHUFFLE_REPLICATES,
            "replicate_seed_averaged_participant_macro_f1": replicate_scores,
            "random_seed_schedule": replicate_seed_schedule,
            "causal_mean_participant_macro_f1": causal_score,
            "shuffled_mean": float(values.mean()),
            "shuffled_median": float(np.median(values)),
            "shuffled_95_percent_interval": np.quantile(values, [0.025, 0.975]).tolist(),
            "causal_minus_shuffled_median": causal_score - float(np.median(values)),
            "permutation_tail_fraction_shuffled_at_least_causal": float(
                (1 + np.sum(values >= causal_score)) / (1 + values.size)
            ),
            "interpretation": (
                "predeclared within-observable-block negative control; replicates are not "
                "independent participants and the tail fraction is not a confirmatory p-value"
            ),
        }

    gate = _advancement_gate(
        boundary_mode=boundary_mode,
        primary=primary,
        supplement=supplement,
        comparison=primary_comparison,
        shuffled=shuffled_controls,
    )
    if data.participant_partition_plan is None:
        raise AssertionError("validated temporal data lost its participant plan")
    plan_sha256 = str(data.participant_partition_plan.audit()["plan_sha256"])
    observable = boundary_mode == "session_observable"
    scoring_eligibility = np.zeros(modelling.labels.size, dtype=np.bool_)
    scoring_eligibility[scoring_indices] = True
    scored_prediction_archive = {**ensemble, **seed_prediction_archive}
    observable_prediction_archive = {
        **observable_ensemble,
        **{
            f"seed-{seed}__{method}": probability
            for seed, methods in observable_by_seed.items()
            for method, probability in methods.items()
        },
    }
    dataset_summary = data.summary()
    observable_pool_audit = dataset_summary.get("observable_candidate_pool")
    observable_identifiers = {
        name: getattr(modelling, name)
        for name in ("participant_ids", "session_ids", "trial_ids", "window_ids")
    }
    scored_identifiers = {
        name: getattr(data, name)
        for name in ("participant_ids", "session_ids", "trial_ids", "window_ids")
    }
    result = {
        "schema_version": "1.0.0",
        "experiment_id": (
            "sole-harmony-observable-session-temporal-v1"
            if observable
            else "sole-harmony-camera-bout-oracle-temporal-v2"
        ),
        "protocol_id": PROTOCOL_ID,
        "observable_context_protocol": OBSERVABLE_CONTEXT_PROTOCOL,
        "evidence_status": (
            "EXTERNAL_TEMPORAL_DEVELOPMENT_NOT_CONFIRMATORY"
            if observable
            else "ORACLE_BOUNDARY_DIAGNOSTIC_ONLY"
        ),
        "dataset": dataset_summary,
        "seeds": list(seeds),
        "primary_seed_averaged": primary,
        "temporal_reports": reports,
        "participant_statistical_supplement": supplement,
        "primary_paired_participant_bootstrap": primary_comparison,
        "shuffled_time_negative_controls": shuffled_controls,
        "advancement_gate": gate,
        "fold_records": fold_records,
        "participant_partition_plan_sha256": plan_sha256,
        "base_model_contract": {
            "method": "XGBoost-6ch",
            "features": "fixed six-channel engineered-feature interface",
            "outer_folds": 5,
            "hyperparameter_trials": 0,
            "inner_fold_assignments": "retained for split audit but unused by fixed control",
            "held_out_predictions_cover_every_observable_candidate": True,
            "outer_training_participant_labels_used_for_supervised_fit": True,
            "outer_evaluation_participant_labels_used_for_fit_selection_or_calibration": False,
        },
        "temporal_contract": {
            "protocol_id": PROTOCOL_ID,
            "boundary_mode": boundary_mode,
            "window_order": (
                "provider participant/session, timestamp-gap or finite-run block, chronological window"
                if observable
                else "provider participant/session, camera-labelled bout, chronological window"
            ),
            "state_reset": (
                "provider session, timestamp discontinuity, or finite-sensor run only"
                if observable
                else "camera-labelled bout or finite sub-run (oracle)"
            ),
            "future_window_access": False,
            "state_updates_before_scoring_slice": True,
            "state_input_population": (
                "all annotation-independent observable candidates"
                if observable
                else "camera-bout-selected oracle candidates"
            ),
            "scoring_eligibility_applied_after_all_temporal_predictions": True,
            "widths_reported_without_outcome_selection": list(_WIDTHS),
            "hysteresis_confirmations_reported_without_outcome_selection": list(
                _HYSTERESIS_CONFIRMATIONS
            ),
            "primary_candidate": PRIMARY_METHOD,
            "primary_control": BASE_METHOD,
            "offline_symmetric_resampling_precludes_zero_lookahead_claim": True,
        },
        "prediction_contract": {
            "scored_window_count": int(data.labels.size),
            "observable_candidate_window_count": int(modelling.labels.size),
            "scoring_eligibility_policy": OBSERVABLE_SCORING_ELIGIBILITY_POLICY,
            "scoring_indices_sha256": canonical_json_sha256(scoring_indices.tolist()),
            "scoring_indices_array_sha256": _array_sha256(scoring_indices),
            "observable_scoring_eligibility_sha256": _array_sha256(scoring_eligibility),
            "observable_candidate_pool_audit_sha256": (
                canonical_json_sha256(observable_pool_audit)
                if isinstance(observable_pool_audit, dict)
                else None
            ),
            "observable_identifier_array_sha256": {
                name: _array_sha256(values) for name, values in observable_identifiers.items()
            },
            "scored_identifier_array_sha256": {
                name: _array_sha256(values) for name, values in scored_identifiers.items()
            },
            "observable_probability_array_sha256": {
                name: _array_sha256(values)
                for name, values in observable_prediction_archive.items()
            },
            "scored_probability_array_sha256": {
                name: _array_sha256(values) for name, values in scored_prediction_archive.items()
            },
            "observable_probabilities_retained_for_every_seed_and_method": True,
            "scored_probabilities_retained_for_every_seed_and_method": True,
            "participant_partition_plan_sha256": plan_sha256,
        },
        "comparison_contract": {
            "within_lane_matched_methods": sorted(reports),
            "camera_bout_lane_comparable_to_session_observable": False,
            "camera_bout_lane_role": "oracle-boundary upper-bound diagnostic only",
            "cross_lane_before_after_language_allowed": False,
        },
        "claim_policy": {
            "confirmatory_claim_allowed": False,
            "state_of_the_art_claim_allowed": False,
            "clinical_claim_allowed": False,
            "zero_lookahead_streaming_claim_allowed": False,
            "camera_bout_lane_is_deployable": False,
            "alternate_width_selection_after_primary_failure_allowed": False,
            "camera_annotations_define_preprocessing_boundaries": not observable,
            "evaluation_activity_labels_are_model_features_or_fit_targets": False,
        },
        "requested_orchestration_n_jobs": n_jobs,
    }
    predictions: dict[str, FloatArray] = dict(scored_prediction_archive)
    predictions.update(
        {
            f"observable-{name}": probability
            for name, probability in observable_prediction_archive.items()
        }
    )
    return result, predictions


def run_and_write(
    *,
    data: ExternalHARWindows,
    output_directory: Path,
    repository_root: Path,
    seeds: tuple[int, ...],
    n_jobs: int,
    inherited_launch_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one explicit lane in a create-only, self-contained evidence directory."""

    if seeds != (11, 23, 47):
        raise ValueError("external publication evidence requires frozen seeds 11, 23, and 47")
    git_at_launch, source_input_manifest, launch_context = _resolve_publication_launch_context(
        repository_root=repository_root,
        output_directory=output_directory,
        current_git_state=_git_state(repository_root),
        current_source_manifest=_source_input_manifest(repository_root),
        manifest_commit_validator=_source_manifest_commit_errors,
        inherited_launch_context=inherited_launch_context,
    )
    launch_context_binding = _publication_launch_context_binding(launch_context)
    output_directory.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC).isoformat()
    dataset_summary = data.summary()
    evidence_status = _external_evidence_status(dataset_summary)
    audit = {
        "schema_version": "1.0.0",
        "created_at": started,
        "dataset": dataset_summary,
        "artifact_evidence_status": evidence_status,
        "source_receipts": [receipt.to_dict() for receipt in data.receipts],
        "storage_disclosure": {
            "raw_local_mirror": False,
            "processing": "streamed provider bytes and in-memory materialization",
        },
        "source_input_manifest": source_input_manifest,
        "git_at_launch": git_at_launch,
        "publication_launch_context": launch_context_binding,
    }
    data_audit_artifact = _write_self_hashed_json_create_only(
        output_directory / "data_audit.json", audit
    )
    artifact_contract = _publication_artifact_contract(source_input_manifest, data_audit_artifact)
    try:
        result, predictions = evaluate_temporal_lane(
            data, seeds=seeds, repository_root=repository_root, n_jobs=n_jobs
        )
        modelling, scoring_indices, _blocks = _validate_lane(
            data,
            boundary_mode=str(cast(dict[str, Any], data.boundary_provenance)["boundary_mode"]),
        )
        _modelling_check, _scoring_check, eligibility = observable_modelling_pool(
            data, include_supervised_labels=True
        )
        if not np.array_equal(
            modelling.window_ids, _modelling_check.window_ids
        ) or not np.array_equal(scoring_indices, _scoring_check):
            raise AssertionError("prediction archive scoring indices changed after evaluation")
        prediction_path = output_directory / "predictions.npz"
        if prediction_path.exists():
            raise FileExistsError(prediction_path)
        standard = {
            f"probability__{name}": value
            for name, value in predictions.items()
            if not name.startswith("observable-")
        }
        observable = {
            f"observable_probability__{name.removeprefix('observable-')}": value
            for name, value in predictions.items()
            if name.startswith("observable-")
        }
        archive_values: dict[str, Any] = {
            "labels": data.labels,
            "participant_ids": data.participant_ids,
            "session_ids": data.session_ids,
            "trial_ids": data.trial_ids,
            "window_ids": data.window_ids,
            "observable_participant_ids": modelling.participant_ids,
            "observable_session_ids": modelling.session_ids,
            "observable_trial_ids": modelling.trial_ids,
            "observable_window_ids": modelling.window_ids,
            "observable_scoring_indices": scoring_indices,
            "observable_scoring_eligibility": eligibility,
        }
        archive_values.update(standard)
        archive_values.update(observable)
        np.savez_compressed(prediction_path, **archive_values)
        _resolve_publication_launch_context(
            repository_root=repository_root,
            output_directory=output_directory,
            current_git_state=_git_state(repository_root),
            current_source_manifest=_source_input_manifest(repository_root),
            manifest_commit_validator=_source_manifest_commit_errors,
            inherited_launch_context=launch_context,
        )
        result["started_at"] = started
        result["created_at"] = datetime.now(UTC).isoformat()
        result["git"] = _git_state(repository_root)
        result["git_at_launch"] = git_at_launch
        result["source_input_manifest"] = source_input_manifest
        result["publication_launch_context"] = launch_context_binding
        result["environment"] = _runtime_environment()
        result["artifact_evidence_status"] = evidence_status
        result["data_audit_artifact"] = data_audit_artifact
        result["artifact_contract"] = artifact_contract
        result["inputs"] = {
            name: {"path": path, "sha256": sha256_file(repository_root / path)}
            for name, path in {
                "portfolio_config": "configs/datasets/external_har_portfolio_v1.yaml",
                "temporal_protocol": PROTOCOL_PATH,
            }.items()
        }
        result["prediction_artifact"] = {
            "path": prediction_path.name,
            "sha256": sha256_file(prediction_path),
        }
        result["result_payload_sha256_before_serialization"] = canonical_json_sha256(result)
        _write_json_create_only(output_directory / "result.json", result)
    except Exception as exc:
        _write_self_hashed_json_create_only(
            output_directory / "failure.json",
            {
                "schema_version": "1.0.0",
                "status": "FAILED_PRESERVED",
                "started_at": started,
                "failed_at": datetime.now(UTC).isoformat(),
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
                "git": _git_state(repository_root),
                "git_at_launch": git_at_launch,
                "source_input_manifest": source_input_manifest,
                "publication_launch_context": launch_context_binding,
                "environment": _runtime_environment(),
                "artifact_evidence_status": evidence_status,
                "data_audit_artifact": data_audit_artifact,
                "artifact_contract": artifact_contract,
            },
            hash_field="failure_payload_sha256_before_serialization",
        )
        validate_and_record_run_directory(output_directory, repository_root)
        raise
    validate_and_record_run_directory(output_directory, repository_root)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    parser.add_argument("--participants", type=int, default=12)
    parser.add_argument("--sessions-per-participant", type=int, default=2)
    parser.add_argument(
        "--boundary-mode",
        choices=("session_observable", "camera_bout_oracle"),
        default="session_observable",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--audit-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repository_root.resolve()
    output_directory = args.output_directory.resolve()
    git_at_launch, source_input_manifest, launch_context = _resolve_publication_launch_context(
        repository_root=root,
        output_directory=output_directory,
        current_git_state=_git_state(root),
        current_source_manifest=_source_input_manifest(root),
        manifest_commit_validator=_source_manifest_commit_errors,
    )
    launch_context_binding = _publication_launch_context_binding(launch_context)
    if args.audit_only:
        output_directory.mkdir(parents=True, exist_ok=False)
        started = datetime.now(UTC).isoformat()
        try:
            data = load_sole_harmony(
                participant_limit=args.participants,
                sessions_per_participant=args.sessions_per_participant,
                boundary_mode=args.boundary_mode,
            )
            _resolve_publication_launch_context(
                repository_root=root,
                output_directory=output_directory,
                current_git_state=_git_state(root),
                current_source_manifest=_source_input_manifest(root),
                manifest_commit_validator=_source_manifest_commit_errors,
                inherited_launch_context=launch_context,
            )
            audit = {
                "schema_version": "1.0.0",
                "created_at": datetime.now(UTC).isoformat(),
                "started_at": started,
                "status": "DATA_AUDIT_COMPLETE_NO_MODELS_RUN",
                "dataset": data.summary(),
                "source_receipts": [receipt.to_dict() for receipt in data.receipts],
                "storage_disclosure": {"raw_local_mirror": False},
                "git": _git_state(root),
                "git_at_launch": git_at_launch,
                "source_input_manifest": source_input_manifest,
                "publication_launch_context": launch_context_binding,
            }
            _write_self_hashed_json_create_only(output_directory / "data_audit.json", audit)
            print(json.dumps(data.summary(), indent=2, sort_keys=True))
            return 0
        except Exception as exc:
            _write_self_hashed_json_create_only(
                output_directory / "failure.json",
                {
                    "schema_version": "1.0.0",
                    "status": "FAILED_PRESERVED",
                    "failure_scope": "pre_writer_configuration_or_dataset_acquisition",
                    "stage": "audit_only_dataset_acquisition_or_materialization",
                    "started_at": started,
                    "failed_at": datetime.now(UTC).isoformat(),
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "git": _git_state(root),
                    "git_at_launch": git_at_launch,
                    "source_input_manifest": source_input_manifest,
                    "publication_launch_context": launch_context_binding,
                    "environment": _runtime_environment(),
                },
                hash_field="failure_payload_sha256_before_serialization",
            )
            validate_and_record_run_directory(output_directory, root)
            raise
    started = datetime.now(UTC).isoformat()
    stage = "configuration"
    try:
        if tuple(args.seeds) != (11, 23, 47):
            raise ValueError("external publication evidence requires frozen seeds 11, 23, and 47")
        stage = "dataset_acquisition"
        data = load_sole_harmony(
            participant_limit=args.participants,
            sessions_per_participant=args.sessions_per_participant,
            boundary_mode=args.boundary_mode,
        )
        stage = "experiment_writer"
        result = run_and_write(
            data=data,
            output_directory=output_directory,
            repository_root=root,
            seeds=tuple(args.seeds),
            n_jobs=args.n_jobs,
            inherited_launch_context=launch_context,
        )
    except Exception as error:
        if not output_directory.exists():
            _write_launch_failure_envelope(
                repository_root=root,
                output_directory=output_directory,
                launch_context=launch_context,
                started_at=started,
                stage=stage,
                exception=error,
                traceback_text=traceback.format_exc(),
            )
        raise
    print(
        json.dumps(
            {
                name: method["mean_participant_macro_f1"]
                for name, method in result["primary_seed_averaged"]["methods"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
