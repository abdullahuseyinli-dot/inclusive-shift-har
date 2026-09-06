"""Create participant-balanced statistical supplements from retained predictions.

This module is reconstruction-only: it validates immutable run packages through the
matched-table gate, opens their scored prediction arrays, and writes a new directory.
It never fits a model or opens source sensor signals.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix  # type: ignore[import-untyped]

from inclusive_shift_har.artifacts.research_provenance import (
    _git_state,
    _source_input_manifest,
    _write_json_create_only,
)
from inclusive_shift_har.evaluation.metrics import expected_calibration_error
from inclusive_shift_har.experiments import publication_table
from inclusive_shift_har.experiments.external_evidence_validate import _safe_run_artifact_path
from inclusive_shift_har.manifests.canonical import canonical_json_sha256

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]

_BOOTSTRAP_REPLICATES = 10_000
_BOOTSTRAP_SEED = 20_260_905
_ECE_BINS = 15
_IDENTITY_ARRAYS = ("labels", "participant_ids", "session_ids", "trial_ids", "window_ids")


class _ParticipantBootstrap:
    """Deterministic equal-participant bootstrap, cached by eligible cohort size."""

    def __init__(self, replicates: int, seed: int) -> None:
        if replicates < 1:
            raise ValueError("bootstrap_replicates must be positive")
        self.replicates = replicates
        self.seed = seed
        self._draws: dict[int, IntArray] = {}

    def draws(self, participant_count: int) -> IntArray:
        if participant_count < 1:
            raise ValueError("a bootstrap summary requires at least one participant")
        if participant_count not in self._draws:
            self._draws[participant_count] = np.random.default_rng(self.seed).integers(
                0,
                participant_count,
                size=(self.replicates, participant_count),
                dtype=np.int64,
            )
        return self._draws[participant_count]


def _descriptive_summary(values: FloatArray) -> dict[str, Any]:
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("participant summaries require a finite, non-empty vector")
    quartiles = np.quantile(values, [0.25, 0.50, 0.75])
    return {
        "mean": float(values.mean()),
        "median": float(quartiles[1]),
        "quartiles_25_50_75": quartiles.tolist(),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "participant_n": int(values.size),
    }


def _bootstrap_summary(values: FloatArray, bootstrap: _ParticipantBootstrap) -> dict[str, Any]:
    summary = _descriptive_summary(values)
    draws = bootstrap.draws(values.size)
    summary["participant_cluster_bootstrap_95_percent_ci"] = np.quantile(
        values[draws].mean(axis=1), [0.025, 0.975]
    ).tolist()
    return summary


def _participant_subset_summary(
    values: FloatArray,
    eligible: NDArray[np.bool_],
    participants: StringArray,
    bootstrap: _ParticipantBootstrap | None,
) -> dict[str, Any]:
    """Summarize a label-defined participant subset without changing its denominator."""
    if values.ndim != 1 or values.shape != eligible.shape or values.shape != participants.shape:
        raise ValueError("participant subset values, eligibility, and identifiers must align")
    selected = values[eligible]
    summary = (
        None
        if selected.size == 0
        else _descriptive_summary(selected)
        if bootstrap is None
        else _bootstrap_summary(selected, bootstrap)
    )
    return {
        "summary": summary,
        "eligible_participant_n": int(eligible.sum()),
        "eligible_participants": participants[eligible].tolist(),
        "ineligible_participants": participants[~eligible].tolist(),
        "participant_values": dict(
            zip(participants[eligible].tolist(), selected.tolist(), strict=True)
        ),
    }


def _validate_inputs(
    labels: IntArray,
    participant_ids: StringArray,
    class_names: tuple[str, ...],
    probabilities_by_seed: dict[int, dict[str, FloatArray]],
) -> tuple[list[int], list[str], StringArray]:
    if labels.ndim != 1 or participant_ids.ndim != 1 or labels.shape != participant_ids.shape:
        raise ValueError("labels and participant_ids must be aligned one-dimensional arrays")
    if labels.size == 0:
        raise ValueError("at least one scored prediction is required")
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("class_names must be non-empty and unique")
    if np.any(labels < 0) or np.any(labels >= len(class_names)):
        raise ValueError("labels do not align with the declared class columns")
    if not probabilities_by_seed:
        raise ValueError("at least one retained seed is required")
    seeds = sorted(probabilities_by_seed)
    methods = sorted(probabilities_by_seed[seeds[0]])
    if not methods or any(set(probabilities_by_seed[seed]) != set(methods) for seed in seeds):
        raise ValueError("every seed must contain the same non-empty method set")
    for seed in seeds:
        for method in methods:
            probability = probabilities_by_seed[seed][method]
            if probability.shape != (labels.size, len(class_names)):
                raise ValueError(f"invalid probability shape for seed {seed}, method {method}")
            if (
                not np.isfinite(probability).all()
                or np.any(probability < 0.0)
                or np.any(probability > 1.0)
                or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
            ):
                raise ValueError(f"invalid probabilities for seed {seed}, method {method}")
    participants = np.unique(participant_ids)
    if np.any(np.asarray([str(value) == "" for value in participants])):
        raise ValueError("participant identifiers must not be empty")
    return seeds, methods, participants


def _class_metric_summary(
    values: FloatArray,
    eligible: NDArray[np.bool_],
    participants: StringArray,
    bootstrap: _ParticipantBootstrap | None,
    *,
    eligible_values: FloatArray | None = None,
) -> dict[str, Any]:
    if values.shape != eligible.shape or values.ndim != 1:
        raise ValueError("class metric values and eligibility must align")
    sensitivity_values = values if eligible_values is None else eligible_values
    if sensitivity_values.shape != values.shape:
        raise ValueError("eligible sensitivity values must align with fixed-cohort values")
    all_summary = (
        _descriptive_summary(values) if bootstrap is None else _bootstrap_summary(values, bootstrap)
    )
    selected_values = sensitivity_values[eligible]
    eligible_summary = (
        None
        if selected_values.size == 0
        else _descriptive_summary(selected_values)
        if bootstrap is None
        else _bootstrap_summary(selected_values, bootstrap)
    )
    return {
        "all_participants_zero_division_0": all_summary,
        "eligible_participant_sensitivity": eligible_summary,
        "eligible_participant_n": int(eligible.sum()),
        "ineligible_participants": participants[~eligible].tolist(),
        "participant_values_zero_division_0": dict(
            zip(participants.tolist(), values.tolist(), strict=True)
        ),
        "eligible_participant_values": dict(
            zip(
                participants[eligible].tolist(),
                sensitivity_values[eligible].tolist(),
                strict=True,
            )
        ),
    }


def _per_class_summary(
    values: FloatArray,
    eligible: NDArray[np.bool_],
    participants: StringArray,
    class_names: tuple[str, ...],
    bootstrap: _ParticipantBootstrap | None,
    *,
    eligible_values: FloatArray | None = None,
) -> dict[str, Any]:
    return {
        class_name: _class_metric_summary(
            values[:, class_index],
            eligible[:, class_index],
            participants,
            bootstrap,
            eligible_values=None if eligible_values is None else eligible_values[:, class_index],
        )
        for class_index, class_name in enumerate(class_names)
    }


def _normalized_confusion_summary(
    rows: FloatArray,
    support: IntArray,
    class_names: tuple[str, ...],
    bootstrap: _ParticipantBootstrap | None,
) -> dict[str, Any]:
    """Average true-class-normalized confusion rows over eligible participants."""
    class_count = len(class_names)
    if rows.shape != (support.shape[0], class_count, class_count):
        raise ValueError("participant confusion rows and support do not align")
    matrix: list[list[float | None]] = []
    lower: list[list[float | None]] = []
    upper: list[list[float | None]] = []
    eligible_counts: list[int] = []
    for true_class in range(class_count):
        eligible = support[:, true_class] > 0
        eligible_counts.append(int(eligible.sum()))
        if not eligible.any():
            matrix.append([None] * class_count)
            lower.append([None] * class_count)
            upper.append([None] * class_count)
            continue
        values = rows[eligible, true_class, :]
        matrix.append(values.mean(axis=0).tolist())
        if bootstrap is None:
            lower.append([None] * class_count)
            upper.append([None] * class_count)
            continue
        draws = bootstrap.draws(values.shape[0])
        intervals = np.quantile(values[draws].mean(axis=1), [0.025, 0.975], axis=0)
        lower.append(intervals[0].tolist())
        upper.append(intervals[1].tolist())
    return {
        "row_labels_true_class": list(class_names),
        "column_labels_predicted_class": list(class_names),
        "mean_matrix": matrix,
        "participant_cluster_bootstrap_95_percent_ci_lower": lower,
        "participant_cluster_bootstrap_95_percent_ci_upper": upper,
        "eligible_participant_n_by_true_class": dict(
            zip(class_names, eligible_counts, strict=True)
        ),
        "estimand": (
            "Within each participant, divide every true-class confusion row by that "
            "participant's support for the class; then average participants with support."
        ),
    }


def _pooled_window_diagnostics(
    labels: IntArray, probability: FloatArray, class_names: tuple[str, ...]
) -> dict[str, Any]:
    predicted = probability.argmax(axis=1)
    matrix = confusion_matrix(labels, predicted, labels=np.arange(len(class_names))).astype(
        np.int64
    )
    true_count = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    true_positive = matrix.diagonal()
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros(len(class_names), dtype=np.float64),
        where=predicted_count > 0,
    )
    recall = np.divide(
        true_positive,
        true_count,
        out=np.zeros(len(class_names), dtype=np.float64),
        where=true_count > 0,
    )
    f1 = np.divide(
        2.0 * true_positive,
        true_count + predicted_count,
        out=np.zeros(len(class_names), dtype=np.float64),
        where=(true_count + predicted_count) > 0,
    )
    selected = np.clip(probability[np.arange(labels.size), labels], 1e-12, 1.0)
    one_hot = np.eye(len(class_names), dtype=np.float64)[labels]
    return {
        "inferential_status": (
            "descriptive pooled-window diagnostic only; windows are not independent "
            "statistical units and no window bootstrap is reported"
        ),
        "window_n": int(labels.size),
        "per_class": {
            class_name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(true_count[index]),
                "predicted_count": int(predicted_count[index]),
            }
            for index, class_name in enumerate(class_names)
        },
        "confusion_counts": matrix.tolist(),
        "calibration": {
            "negative_log_likelihood": float(-np.log(selected).mean()),
            "multiclass_brier": float(np.square(probability - one_hot).sum(axis=1).mean()),
            "top_label_ece_15_equal_width_bins": expected_calibration_error(
                labels, probability, bins=_ECE_BINS
            ),
        },
    }


def _seed_averaged_metric(
    values: FloatArray, eligible: NDArray[np.bool_]
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    """Return fixed-seed means and eligible-seed participant sensitivities."""
    if values.shape != eligible.shape or values.ndim != 3:
        raise ValueError("seed metric values and eligibility must be seed x participant x class")
    counts = eligible.sum(axis=0)
    summed = np.where(eligible, values, 0.0).sum(axis=0)
    eligible_averaged = np.divide(summed, counts, out=np.zeros_like(summed), where=counts > 0)
    return values.mean(axis=0), eligible_averaged, counts > 0


def _method_statistics(
    labels: IntArray,
    participant_ids: StringArray,
    participants: StringArray,
    class_names: tuple[str, ...],
    seeds: list[int],
    per_seed: dict[int, FloatArray],
    bootstrap: _ParticipantBootstrap,
) -> dict[str, Any]:
    seed_count = len(seeds)
    participant_count = participants.size
    class_count = len(class_names)
    support = np.asarray(
        [
            np.bincount(labels[participant_ids == participant], minlength=class_count)
            for participant in participants
        ],
        dtype=np.int64,
    )
    precision = np.zeros((seed_count, participant_count, class_count), dtype=np.float64)
    recall = np.zeros_like(precision)
    f1 = np.zeros_like(precision)
    precision_eligible = np.zeros_like(precision, dtype=np.bool_)
    recall_eligible = np.broadcast_to(support > 0, precision.shape).copy()
    f1_eligible = np.zeros_like(precision, dtype=np.bool_)
    confusion_rows = np.zeros(
        (seed_count, participant_count, class_count, class_count), dtype=np.float64
    )
    nll = np.zeros((seed_count, participant_count), dtype=np.float64)
    brier = np.zeros_like(nll)
    ece = np.zeros_like(nll)
    macro_f1 = np.zeros_like(nll)
    present_class_macro_f1 = np.zeros_like(nll)
    complete_class_participants = np.asarray((support > 0).all(axis=1), dtype=np.bool_)
    seed_records: dict[str, Any] = {}

    for seed_index, seed in enumerate(seeds):
        probability = per_seed[seed]
        predicted = probability.argmax(axis=1)
        participant_records: dict[str, Any] = {}
        for participant_index, participant in enumerate(participants):
            selected = participant_ids == participant
            truth = labels[selected]
            local_probability = probability[selected]
            local_predicted = predicted[selected]
            matrix = confusion_matrix(truth, local_predicted, labels=np.arange(class_count)).astype(
                np.int64
            )
            true_count = matrix.sum(axis=1)
            predicted_count = matrix.sum(axis=0)
            true_positive = matrix.diagonal()
            local_precision = np.divide(
                true_positive,
                predicted_count,
                out=np.zeros(class_count, dtype=np.float64),
                where=predicted_count > 0,
            )
            local_recall = np.divide(
                true_positive,
                true_count,
                out=np.zeros(class_count, dtype=np.float64),
                where=true_count > 0,
            )
            local_f1 = np.divide(
                2.0 * true_positive,
                true_count + predicted_count,
                out=np.zeros(class_count, dtype=np.float64),
                where=(true_count + predicted_count) > 0,
            )
            precision[seed_index, participant_index] = local_precision
            recall[seed_index, participant_index] = local_recall
            f1[seed_index, participant_index] = local_f1
            precision_eligible[seed_index, participant_index] = predicted_count > 0
            f1_eligible[seed_index, participant_index] = (true_count + predicted_count) > 0
            confusion_rows[seed_index, participant_index] = np.divide(
                matrix,
                true_count[:, None],
                out=np.zeros((class_count, class_count), dtype=np.float64),
                where=true_count[:, None] > 0,
            )
            selected_probability = np.clip(
                local_probability[np.arange(truth.size), truth], 1e-12, 1.0
            )
            one_hot = np.eye(class_count, dtype=np.float64)[truth]
            nll[seed_index, participant_index] = float(-np.log(selected_probability).mean())
            brier[seed_index, participant_index] = float(
                np.square(local_probability - one_hot).sum(axis=1).mean()
            )
            ece[seed_index, participant_index] = expected_calibration_error(
                truth, local_probability, bins=_ECE_BINS
            )
            macro_f1[seed_index, participant_index] = float(local_f1.mean())
            true_class_present = true_count > 0
            if not true_class_present.any():
                raise ValueError("every scored participant must have at least one true class")
            present_class_macro_f1[seed_index, participant_index] = float(
                local_f1[true_class_present].mean()
            )
            participant_records[str(participant)] = {
                "window_count": int(selected.sum()),
                "fixed_class_macro_f1": float(local_f1.mean()),
                "present_true_class_macro_f1_sensitivity": float(
                    local_f1[true_class_present].mean()
                ),
                "has_all_declared_classes": bool(true_class_present.all()),
                "per_class": {
                    class_name: {
                        "precision": float(local_precision[class_index]),
                        "recall": float(local_recall[class_index]),
                        "f1": float(local_f1[class_index]),
                        "true_support": int(true_count[class_index]),
                        "predicted_count": int(predicted_count[class_index]),
                        "precision_eligible": bool(predicted_count[class_index] > 0),
                        "recall_eligible": bool(true_count[class_index] > 0),
                        "f1_eligible": bool(
                            true_count[class_index] + predicted_count[class_index] > 0
                        ),
                    }
                    for class_index, class_name in enumerate(class_names)
                },
                "confusion_counts": matrix.tolist(),
                "calibration": {
                    "negative_log_likelihood": float(nll[seed_index, participant_index]),
                    "multiclass_brier": float(brier[seed_index, participant_index]),
                    "top_label_ece_15_equal_width_bins": float(ece[seed_index, participant_index]),
                },
            }
        seed_primary = _descriptive_summary(macro_f1[seed_index])
        seed_primary["bottom_30_percent_mean"] = float(
            np.sort(macro_f1[seed_index])[: max(1, int(np.ceil(0.30 * participant_count)))].mean()
        )
        seed_primary["worst_participant_ids"] = participants[
            np.isclose(
                macro_f1[seed_index],
                macro_f1[seed_index].min(),
                atol=1e-12,
                rtol=0.0,
            )
        ].tolist()
        seed_primary["participant_values"] = dict(
            zip(participants.tolist(), macro_f1[seed_index].tolist(), strict=True)
        )
        seed_present_class = _descriptive_summary(present_class_macro_f1[seed_index])
        seed_present_class["participant_values"] = dict(
            zip(
                participants.tolist(),
                present_class_macro_f1[seed_index].tolist(),
                strict=True,
            )
        )
        seed_records[str(seed)] = {
            "participant_details": participant_records,
            "participant_fixed_class_macro_f1": seed_primary,
            "participant_present_true_class_macro_f1_sensitivity": seed_present_class,
            "all_declared_classes_supported_participant_sensitivity": (
                _participant_subset_summary(
                    macro_f1[seed_index],
                    complete_class_participants,
                    participants,
                    None,
                )
            ),
            "participant_balanced_per_class": {
                "precision": _per_class_summary(
                    precision[seed_index],
                    precision_eligible[seed_index],
                    participants,
                    class_names,
                    None,
                ),
                "recall": _per_class_summary(
                    recall[seed_index],
                    recall_eligible[seed_index],
                    participants,
                    class_names,
                    None,
                ),
                "f1": _per_class_summary(
                    f1[seed_index],
                    f1_eligible[seed_index],
                    participants,
                    class_names,
                    None,
                ),
            },
            "participant_normalized_confusion": _normalized_confusion_summary(
                confusion_rows[seed_index], support, class_names, None
            ),
            "participant_mean_calibration": {
                "negative_log_likelihood": _descriptive_summary(nll[seed_index]),
                "multiclass_brier": _descriptive_summary(brier[seed_index]),
                "top_label_ece_15_equal_width_bins": _descriptive_summary(ece[seed_index]),
            },
            "pooled_window_diagnostics": _pooled_window_diagnostics(
                labels, probability, class_names
            ),
            "uncertainty_note": (
                "Per-seed rows are descriptive. Participant-cluster intervals are on the "
                "frozen-seed aggregate below."
            ),
        }

    fixed_precision, eligible_precision, precision_any = _seed_averaged_metric(
        precision, precision_eligible
    )
    fixed_recall, eligible_recall, recall_any = _seed_averaged_metric(recall, recall_eligible)
    fixed_f1, eligible_f1, f1_any = _seed_averaged_metric(f1, f1_eligible)
    averaged_confusion = confusion_rows.mean(axis=0)
    averaged_macro_f1 = macro_f1.mean(axis=0)
    averaged_present_class_macro_f1 = present_class_macro_f1.mean(axis=0)
    tail_count = max(1, int(np.ceil(0.30 * participant_count)))
    primary_summary = _bootstrap_summary(averaged_macro_f1, bootstrap)
    primary_summary["bottom_30_percent_mean"] = float(
        np.sort(averaged_macro_f1)[:tail_count].mean()
    )
    bootstrap_values = averaged_macro_f1[bootstrap.draws(participant_count)]
    primary_summary["bottom_30_percent_participant_cluster_bootstrap_95_percent_ci"] = np.quantile(
        np.sort(bootstrap_values, axis=1)[:, :tail_count].mean(axis=1),
        [0.025, 0.975],
    ).tolist()
    primary_summary["worst_participant_ids"] = participants[
        np.isclose(
            averaged_macro_f1,
            averaged_macro_f1.min(),
            atol=1e-12,
            rtol=0.0,
        )
    ].tolist()
    primary_summary["participant_values"] = dict(
        zip(participants.tolist(), averaged_macro_f1.tolist(), strict=True)
    )
    present_class_summary = _bootstrap_summary(averaged_present_class_macro_f1, bootstrap)
    present_class_summary["participant_values"] = dict(
        zip(participants.tolist(), averaged_present_class_macro_f1.tolist(), strict=True)
    )
    return {
        "participant_seed_details": seed_records,
        "seed_averaged_participant_cluster_statistics": {
            "primary_fixed_class_macro_f1": primary_summary,
            "participant_present_true_class_macro_f1_sensitivity": present_class_summary,
            "all_declared_classes_supported_participant_sensitivity": (
                _participant_subset_summary(
                    averaged_macro_f1,
                    complete_class_participants,
                    participants,
                    bootstrap,
                )
            ),
            "participant_balanced_per_class": {
                "precision": _per_class_summary(
                    fixed_precision,
                    precision_any,
                    participants,
                    class_names,
                    bootstrap,
                    eligible_values=eligible_precision,
                ),
                "recall": _per_class_summary(
                    fixed_recall,
                    recall_any,
                    participants,
                    class_names,
                    bootstrap,
                    eligible_values=eligible_recall,
                ),
                "f1": _per_class_summary(
                    fixed_f1,
                    f1_any,
                    participants,
                    class_names,
                    bootstrap,
                    eligible_values=eligible_f1,
                ),
            },
            "participant_normalized_confusion": _normalized_confusion_summary(
                averaged_confusion, support, class_names, bootstrap
            ),
            "participant_mean_calibration": {
                "negative_log_likelihood": _bootstrap_summary(nll.mean(axis=0), bootstrap),
                "multiclass_brier": _bootstrap_summary(brier.mean(axis=0), bootstrap),
                "top_label_ece_15_equal_width_bins": _bootstrap_summary(
                    ece.mean(axis=0), bootstrap
                ),
            },
        },
    }


def compute_participant_statistical_supplement(
    labels: IntArray,
    participant_ids: StringArray,
    class_names: tuple[str, ...],
    probabilities_by_seed: dict[int, dict[str, FloatArray]],
    *,
    bootstrap_replicates: int = _BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = _BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute fixed-cohort and eligible-participant statistics without file access."""
    seeds, methods, participants = _validate_inputs(
        labels, participant_ids, class_names, probabilities_by_seed
    )
    bootstrap = _ParticipantBootstrap(bootstrap_replicates, bootstrap_seed)
    class_count = len(class_names)
    support = np.asarray(
        [
            np.bincount(labels[participant_ids == participant], minlength=class_count)
            for participant in participants
        ],
        dtype=np.int64,
    )
    result: dict[str, Any] = {
        "statistical_unit": "participant",
        "seed_aggregation": (
            "First average all frozen seed values within participant, then give each "
            "participant equal weight. Bootstrap resamples participants as clusters and "
            "retains all their seed observations."
        ),
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "seeds": seeds,
        "participant_count": int(participants.size),
        "participants_with_all_declared_classes": int((support > 0).all(axis=1).sum()),
        "participants_missing_at_least_one_declared_class": participants[
            ~(support > 0).all(axis=1)
        ].tolist(),
        "perfect_prediction_fixed_class_mean_ceiling": float((support > 0).mean(axis=1).mean()),
        "scored_window_count": int(labels.size),
        "class_support": {
            class_name: {
                "scored_windows": int(support[:, class_index].sum()),
                "participants_with_true_support": int((support[:, class_index] > 0).sum()),
                "participants_without_true_support": participants[
                    support[:, class_index] == 0
                ].tolist(),
            }
            for class_index, class_name in enumerate(class_names)
        },
        "participant_class_support": {
            str(participant): dict(zip(class_names, support[index].tolist(), strict=True))
            for index, participant in enumerate(participants)
        },
        "estimand_definitions": {
            "fixed_cohort_per_class": (
                "Mean of participant-level class precision/recall/F1 over every participant; "
                "undefined ratios are zero. This preserves a fixed denominator."
            ),
            "eligible_participant_sensitivity": (
                "Precision includes participants predicting the class; recall includes "
                "participants with true support; F1 includes their union. Each eligible "
                "participant is weighted equally after eligible seeds are averaged. "
                "Precision and F1 eligibility can therefore depend on model predictions; "
                "these are secondary sensitivities, never the fixed-cohort primary."
            ),
            "participant_present_true_class_macro_f1_sensitivity": (
                "Within each participant and seed, average class F1 only over declared "
                "classes with true scored-window support for that participant. The "
                "label-defined participant cohort is unchanged across methods and seeds; "
                "the varying class denominator makes this a secondary missing-class "
                "sensitivity, not the primary estimand."
            ),
            "all_declared_classes_supported_participant_sensitivity": (
                "Apply the primary fixed-class macro-F1 definition only to participants "
                "with true scored-window support for every declared class. Eligibility is "
                "label-defined and fixed across methods and seeds. Empty cohorts are "
                "reported as null, never imputed."
            ),
            "participant_normalized_confusion": (
                "Rows are normalized within participant and true class, then averaged only "
                "over participants with support for that true class."
            ),
            "participant_mean_calibration": (
                "NLL, multiclass Brier, and 15-bin equal-width top-label ECE are calculated "
                "within participant and then equally averaged. Participant-mean ECE is a "
                "secondary, sample-size-sensitive calibration diagnostic."
            ),
            "pooled_window_diagnostics": (
                "Pooled metrics are descriptive only and are never assigned participant-"
                "level uncertainty or treated as independent-window inference."
            ),
        },
        "methods": {},
    }
    method_results: dict[str, Any] = result["methods"]
    for method in methods:
        method_results[method] = _method_statistics(
            labels,
            participant_ids,
            participants,
            class_names,
            seeds,
            {seed: probabilities_by_seed[seed][method] for seed in seeds},
            bootstrap,
        )
    return result


def _paired_comparison_groups(
    source_groups: list[dict[str, Any]], statistics: dict[str, Any]
) -> list[dict[str, Any]]:
    """Rebuild matched comparisons by participant ID and cross-check the source table.

    Dictionary insertion order is not a scientific pairing contract. This routine aligns
    every candidate and control by explicit participant identifier before calculating the
    paired bootstrap, lower-tail contrast, and rescue/harm/tie counts.
    """
    bootstrap = _ParticipantBootstrap(
        int(statistics["bootstrap_replicates"]), int(statistics["bootstrap_seed"])
    )
    methods = statistics["methods"]
    rebuilt_groups: list[dict[str, Any]] = []
    for source_group in source_groups:
        rebuilt_group = {key: value for key, value in source_group.items() if key != "comparisons"}
        source_comparison = source_group["comparisons"]
        if "pairs" not in source_comparison:
            rebuilt_group["comparisons"] = source_comparison
            rebuilt_groups.append(rebuilt_group)
            continue
        control = str(source_comparison["control"])
        if control not in methods:
            raise ValueError(f"paired comparison control is absent from statistics: {control}")
        control_values = methods[control]["seed_averaged_participant_cluster_statistics"][
            "primary_fixed_class_macro_f1"
        ]["participant_values"]
        participant_order = sorted(control_values)
        if not participant_order:
            raise ValueError("paired comparisons require at least one participant")
        base = np.asarray([control_values[name] for name in participant_order], dtype=np.float64)
        draws = bootstrap.draws(base.size)
        tail_count = max(1, int(np.ceil(0.30 * base.size)))
        rebuilt_pairs: dict[str, Any] = {}
        for method, source_pair in sorted(source_comparison["pairs"].items()):
            if method not in methods:
                raise ValueError(f"paired comparison method is absent from statistics: {method}")
            candidate_values = methods[method]["seed_averaged_participant_cluster_statistics"][
                "primary_fixed_class_macro_f1"
            ]["participant_values"]
            if set(candidate_values) != set(participant_order):
                raise ValueError(f"paired comparison participant mismatch: {method} vs {control}")
            candidate = np.asarray(
                [candidate_values[name] for name in participant_order], dtype=np.float64
            )
            delta = candidate - base
            mean_ci = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975]).tolist()
            tail_ci = np.quantile(
                np.sort(candidate[draws], axis=1)[:, :tail_count].mean(axis=1)
                - np.sort(base[draws], axis=1)[:, :tail_count].mean(axis=1),
                [0.025, 0.975],
            ).tolist()
            calculated = {
                "mean_difference": float(delta.mean()),
                "paired_participant_bootstrap_95_percent_ci": mean_ci,
                "bottom_30_percent_difference_95_percent_ci": tail_ci,
                "rescue_count": int(np.sum(delta > 1e-12)),
                "harm_count": int(np.sum(delta < -1e-12)),
                "tie_count": int(np.sum(np.abs(delta) <= 1e-12)),
            }
            for field in ("rescue_count", "harm_count", "tie_count"):
                if int(source_pair[field]) != calculated[field]:
                    raise ValueError(f"source paired-comparison {field} mismatch for {method}")
            for field in (
                "mean_difference",
                "paired_participant_bootstrap_95_percent_ci",
                "bottom_30_percent_difference_95_percent_ci",
            ):
                if not np.allclose(
                    np.asarray(source_pair[field], dtype=np.float64),
                    np.asarray(calculated[field], dtype=np.float64),
                    atol=1e-12,
                    rtol=0.0,
                ):
                    raise ValueError(f"source paired-comparison {field} mismatch for {method}")
            rebuilt_pairs[method] = {
                **{
                    key: value
                    for key, value in source_pair.items()
                    if key
                    not in {
                        "mean_difference",
                        "paired_participant_bootstrap_95_percent_ci",
                        "bottom_30_percent_difference_95_percent_ci",
                        "rescue_count",
                        "harm_count",
                        "tie_count",
                    }
                },
                **calculated,
                "difference_direction": "candidate_minus_control",
                "tie_tolerance_absolute": 1e-12,
                "paired_participant_ids_sha256": canonical_json_sha256(participant_order),
                "participant_delta_values": dict(
                    zip(participant_order, delta.tolist(), strict=True)
                ),
                "source_comparison_cross_check_passed": True,
            }
        rebuilt_group["comparisons"] = {
            **{key: value for key, value in source_comparison.items() if key != "pairs"},
            "pairing_contract": (
                "Candidate minus control after exact participant-ID alignment. A positive "
                "delta is a rescue, a negative delta is harm, and absolute delta <= 1e-12 "
                "is a tie. Bootstrap draws jointly resample aligned participant rows while "
                "retaining every frozen seed within participant."
            ),
            "bottom_30_percent_contract": (
                "Within each paired participant bootstrap draw, independently rank candidate "
                "and control participant scores and subtract their bottom-30% means. This is "
                "a lower-tail distribution contrast, not a fixed-person rescue contrast."
            ),
            "pairs": rebuilt_pairs,
        }
        rebuilt_groups.append(rebuilt_group)
    return rebuilt_groups


def _path_locator(path: Path, repository_root: Path) -> dict[str, str]:
    """Describe a run location without an ambiguous machine-specific path."""
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(repository_root.resolve(strict=True)).as_posix()
    except ValueError:
        return {"path_kind": "external_absolute", "path": str(resolved)}
    return {"path_kind": "repository_relative", "path": relative or "."}


def _regular_file_snapshot(path: Path, description: str) -> tuple[bytes, str]:
    """Read one non-symlink file once so the digest covers the bytes subsequently used."""
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{description} must be a regular non-symlink file: {path}")
    payload = path.read_bytes()
    return payload, hashlib.sha256(payload).hexdigest()


def _expected_table_source(
    source: Any,
    directory: Path,
    repository_root: Path,
) -> dict[str, Any]:
    if not isinstance(source, dict):
        raise ValueError("matched table source must be an object")
    expected_directory = publication_table._repository_relative_path(directory, repository_root)
    if source.get("run_directory") != expected_directory:
        raise ValueError("matched table source does not identify the requested run directory")
    return source


def _load_prediction_evidence(
    run_directories: list[Path],
    seeds: list[int],
    identical_reference_methods: tuple[str, ...],
    table_sources: list[dict[str, Any]],
    repository_root: Path,
) -> tuple[
    dict[str, NDArray[Any]],
    dict[int, dict[str, FloatArray]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    if len(table_sources) != len(run_directories):
        raise ValueError("matched table source count differs from requested run count")
    identity: dict[str, NDArray[Any]] = {}
    per_seed: dict[int, dict[str, FloatArray]] = {seed: {} for seed in seeds}
    sources: list[dict[str, Any]] = []
    checked_table_sources: list[dict[str, Any]] = []
    for directory_value, table_source_value in zip(run_directories, table_sources, strict=True):
        directory = directory_value.resolve(strict=True)
        if not directory.is_dir():
            raise ValueError(f"run directory is not a directory: {directory}")
        table_source = _expected_table_source(table_source_value, directory, repository_root)
        result_path = _safe_run_artifact_path(directory, "result.json")
        audit_path = _safe_run_artifact_path(directory, "data_audit.json")
        if result_path is None or audit_path is None:  # fixed names; defensive fail-closed guard
            raise ValueError("run contains an unsafe terminal or data-audit path")
        result_payload, result_sha256 = _regular_file_snapshot(result_path, "run result")
        _audit_payload, audit_sha256 = _regular_file_snapshot(audit_path, "run data audit")
        if result_sha256 != table_source.get("result_sha256"):
            raise ValueError("run result SHA-256 differs from the matched table source")
        if audit_sha256 != table_source.get("data_audit_sha256"):
            raise ValueError("run data-audit SHA-256 differs from the matched table source")
        result = json.loads(result_payload)
        if not isinstance(result, dict):
            raise ValueError(f"run result must contain a JSON object: {result_path}")
        methods = sorted(result["primary_seed_averaged"]["methods"])
        prediction_record = result.get("prediction_artifact")
        declared_prediction_path = (
            prediction_record.get("path") if isinstance(prediction_record, dict) else None
        )
        prediction_path = _safe_run_artifact_path(directory, declared_prediction_path)
        if prediction_path is None:
            raise ValueError("prediction artifact must be one direct run-directory file")
        prediction_payload, predictions_sha256 = _regular_file_snapshot(
            prediction_path, "prediction artifact"
        )
        if predictions_sha256 != table_source.get("predictions_sha256"):
            raise ValueError("prediction SHA-256 differs from the matched table source")
        declared_prediction_sha256 = (
            prediction_record.get("sha256") if isinstance(prediction_record, dict) else None
        )
        if (
            declared_prediction_sha256 is not None
            and declared_prediction_sha256 != predictions_sha256
        ):
            raise ValueError("prediction SHA-256 differs from the result declaration")
        with np.load(io.BytesIO(prediction_payload), allow_pickle=False) as archive:
            for name in _IDENTITY_ARRAYS:
                values = np.asarray(archive[name])
                if name in identity and not np.array_equal(identity[name], values):
                    raise ValueError(f"non-comparable prediction identity: {name}")
                identity.setdefault(name, values.copy())
            for seed in seeds:
                for method in methods:
                    values = np.asarray(
                        archive[f"probability__seed-{seed}__{method}"], dtype=np.float64
                    )
                    if method in per_seed[seed]:
                        existing = per_seed[seed][method]
                        if (
                            method not in identical_reference_methods
                            or existing.shape != values.shape
                            or existing.tobytes() != values.tobytes()
                        ):
                            raise ValueError(f"duplicate prediction method differs: {method}")
                        continue
                    per_seed[seed][method] = values.copy()
        sources.append(
            {
                "run_directory": _path_locator(directory, repository_root),
                "result_artifact": {
                    "path_kind": "run_directory_relative",
                    "path": "result.json",
                    "sha256": result_sha256,
                },
                "prediction_artifact": {
                    "path_kind": "run_directory_relative",
                    "path": prediction_path.name,
                    "sha256": predictions_sha256,
                },
                "data_audit_artifact": {
                    "path_kind": "run_directory_relative",
                    "path": "data_audit.json",
                    "sha256": audit_sha256,
                },
                "methods": methods,
                "matched_table_source_hash_cross_check_passed": True,
                "prediction_loaded_from_hashed_byte_snapshot": True,
            }
        )
        checked_table_source = dict(table_source)
        checked_table_source["run_directory"] = _path_locator(directory, repository_root)
        checked_table_source["supplement_source_hash_cross_check_passed"] = True
        checked_table_sources.append(checked_table_source)
    return identity, per_seed, sources, checked_table_sources


def reconstruct_statistical_supplement(
    run_directories: list[Path],
    repository_root: Path,
    *,
    identical_reference_methods: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Reconstruct a supplement after the existing matched-publication gate passes."""
    table = publication_table.reconstruct_matched_table(
        run_directories,
        repository_root,
        identical_reference_methods=identical_reference_methods,
    )
    seeds = [int(seed) for seed in table["comparison_contract"]["seeds"]]
    identity, probabilities, prediction_sources, validated_run_sources = _load_prediction_evidence(
        run_directories,
        seeds,
        identical_reference_methods,
        table["sources"],
        repository_root,
    )
    raw_labels = np.asarray(identity["labels"])
    if not np.issubdtype(raw_labels.dtype, np.integer):
        raise ValueError("prediction labels must use an integer dtype")
    identity_sha256 = canonical_json_sha256(
        {name: np.asarray(identity[name]).tolist() for name in _IDENTITY_ARRAYS}
    )
    if identity_sha256 != table["prediction_identity_sha256"]:
        raise ValueError("prediction identity digest does not match the validated table")
    labels = np.asarray(raw_labels, dtype=np.int64)
    participant_ids = np.asarray(identity["participant_ids"], dtype=np.str_)
    class_names = tuple(str(name) for name in table["comparison_contract"]["class_names"])
    statistics = compute_participant_statistical_supplement(
        labels, participant_ids, class_names, probabilities
    )
    if set(statistics["methods"]) != set(table["statistics"]["methods"]):
        raise ValueError("supplement methods do not match the validated table")
    for method, values in statistics["methods"].items():
        calculated = values["seed_averaged_participant_cluster_statistics"][
            "primary_fixed_class_macro_f1"
        ]["participant_values"]
        retained = table["statistics"]["methods"][method]["participant_values"]
        if set(calculated) != set(retained) or not np.allclose(
            [calculated[name] for name in sorted(calculated)],
            [retained[name] for name in sorted(retained)],
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(f"participant metric cross-check failed for {method}")
    compact_contract = {
        key: table["comparison_contract"].get(key)
        for key in (
            "class_names",
            "seeds",
            "endpoint",
            "protocol_id",
            "estimand",
            "personalization_budget",
            "inference_unit",
            "bootstrap_unit",
        )
    }
    compact_contract["dataset_id"] = table["comparison_contract"]["dataset"]["dataset_id"]
    compact_contract["source_dataset"] = table["comparison_contract"].get("source_dataset")
    paired_comparison_groups = _paired_comparison_groups(
        table["descriptive_comparisons_vs_strongest_same_input_and_context_control"],
        statistics,
    )
    record: dict[str, Any] = {
        "schema_version": "1.1.0",
        "record_kind": "publication_statistical_supplement",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "evidence_status": table["evidence_status"],
        "comparison_contract": compact_contract,
        "comparison_contract_sha256": table["comparison_contract_sha256"],
        "prediction_identity_sha256": table["prediction_identity_sha256"],
        "matched_table_record_sha256": table["record_sha256"],
        "method_inference_contracts": table["method_inference_contracts"],
        "method_evidence_statuses": table["method_evidence_statuses"],
        "statistics": statistics,
        "strongest_same_input_and_context_budget_comparisons": paired_comparison_groups,
        "comparison_qualification": (
            "Same-budget comparator selection is descriptive and post-selection. Groups "
            "without an applicable retained control remain explicitly unmatched; no "
            "cross-channel or cross-context before/after claim is licensed."
        ),
        "missing_data_limitation": {
            "scope": (
                "Prediction artifacts contain only scored, post-eligibility windows. They "
                "cannot identify performance on excluded non-finite, missing-label, "
                "unsupported-label, mixed-label, or dropped-tail candidates."
            ),
            "consequence": (
                "Reported participant and class denominators are conditional on scored "
                "windows; candidate/exclusion accounting must be read from the validated "
                "dataset audits and is not an imputed performance estimate."
            ),
            "imputation_performed": False,
        },
        "compute_and_model_size_limitation": (
            "The retained prediction/result packages do not expose a common, comparable "
            "fit-time, inference-time, peak-memory, serialized-size, or parameter-count "
            "contract for every method. This supplement does not invent those values."
        ),
        "reporting_git_state": _git_state(repository_root),
        "reporting_source_manifest": _source_input_manifest(repository_root),
        "prediction_sources": prediction_sources,
        "validated_run_sources": validated_run_sources,
        "reconstruction_only_no_training_or_raw_signal_access": True,
        "independent_confirmation_or_sota_claim_allowed": False,
        "scientific_status_inheritance": {
            "dataset_evidence_status_copied_verbatim_from_validated_table": True,
            "method_evidence_statuses_copied_verbatim_from_validated_table": True,
            "reporting_reconstruction_can_upgrade_scientific_status": False,
            "paired_comparisons_are_descriptive_post_selection": True,
        },
    }
    record["record_sha256"] = canonical_json_sha256(record)
    return record


def supplement_markdown(record: dict[str, Any]) -> str:
    statistics = record["statistics"]
    lines = [
        "# Participant-balanced statistical supplement",
        "",
        f"Evidence status: {record['evidence_status']}. Statistical unit: participant. "
        f"Participants: {statistics['participant_count']}; scored windows: "
        f"{statistics['scored_window_count']}; seeds: "
        f"{', '.join(map(str, statistics['seeds']))}.",
        "",
        "Intervals resample participants as clusters after all frozen seeds are retained "
        "within participant. Pooled-window values are descriptive diagnostics only.",
        "",
        "| Method | Fixed-class macro-F1 [95% CI] | Present-true-class sensitivity | "
        "All-classes-supported sensitivity (N) | Participant-mean NLL | Brier | ECE | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for method, values in sorted(statistics["methods"].items()):
        aggregate = values["seed_averaged_participant_cluster_statistics"]
        primary = aggregate["primary_fixed_class_macro_f1"]
        present_class = aggregate["participant_present_true_class_macro_f1_sensitivity"]
        complete_class = aggregate["all_declared_classes_supported_participant_sensitivity"]
        calibration = aggregate["participant_mean_calibration"]
        low, high = primary["participant_cluster_bootstrap_95_percent_ci"]
        complete_summary = complete_class["summary"]
        complete_text = (
            f"{complete_summary['mean']:.6f} ({complete_class['eligible_participant_n']})"
            if complete_summary is not None
            else "not estimable (0)"
        )
        lines.append(
            f"| {method} | {primary['mean']:.6f} [{low:.6f}, {high:.6f}] "
            f"| {present_class['mean']:.6f} | {complete_text} "
            f"| {calibration['negative_log_likelihood']['mean']:.6f} "
            f"| {calibration['multiclass_brier']['mean']:.6f} "
            f"| {calibration['top_label_ece_15_equal_width_bins']['mean']:.6f} "
            f"| {record['method_evidence_statuses'][method]} |"
        )
    lines.extend(
        [
            "",
            "Per-class fixed-cohort and eligible-participant estimates, participant-"
            "normalized confusion matrices, per-seed details, participant distributions, "
            "bootstrap intervals, and matched rescue/harm/tie comparisons are in the JSON.",
            "The present-class and all-classes-supported columns are secondary, label-defined "
            "missing-class sensitivities. They do not replace the fixed-class primary.",
            "Paired comparisons are descriptive and cannot upgrade any inherited evidence status.",
            "",
            f"Missing-data limitation: {record['missing_data_limitation']['scope']}",
            "",
            f"Record SHA-256: `{record['record_sha256']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_statistical_supplement(
    run_directories: list[Path],
    repository_root: Path,
    output: Path,
    *,
    identical_reference_methods: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Write JSON and Markdown to a new directory, refusing replacement."""
    record = reconstruct_statistical_supplement(
        run_directories,
        repository_root,
        identical_reference_methods=identical_reference_methods,
    )
    markdown = supplement_markdown(record)
    output.mkdir(parents=True, exist_ok=False)
    _write_json_create_only(output / "statistical_supplement.json", record)
    with (output / "statistical_supplement.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(markdown)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, action="append", required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identical-reference-method", action="append", default=[])
    args = parser.parse_args(argv)
    record = write_statistical_supplement(
        args.run_directory,
        args.repository_root.resolve(),
        args.output,
        identical_reference_methods=tuple(args.identical_reference_method),
    )
    print(
        json.dumps(
            {
                "status": "STATISTICAL_SUPPLEMENT_CREATED",
                "record_sha256": record["record_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
