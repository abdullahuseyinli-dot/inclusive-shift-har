"""Participant-aware classification, calibration, and selective-risk metrics."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (  # type: ignore[import-untyped]
    confusion_matrix,
    f1_score,
    recall_score,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _validate_predictions(
    y_true: Sequence[int] | IntArray,
    probabilities: Sequence[Sequence[float]] | FloatArray,
    participant_ids: Sequence[str],
) -> tuple[IntArray, FloatArray, NDArray[np.str_]]:
    labels = np.asarray(y_true, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    participants = np.asarray(participant_ids, dtype=np.str_)
    if labels.ndim != 1 or probability.ndim != 2 or participants.ndim != 1:
        raise ValueError("labels/participants must be vectors and probabilities must be a matrix")
    if labels.shape[0] != probability.shape[0] or labels.shape[0] != participants.shape[0]:
        raise ValueError("predictions, labels, and participant ids are not aligned")
    if labels.size == 0 or probability.shape[1] < 2:
        raise ValueError("at least one sample and two classes are required")
    if not np.isfinite(probability).all() or np.any(probability < 0):
        raise ValueError("probabilities must be finite and non-negative")
    row_sums = probability.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("probability rows must sum to one")
    if labels.min() < 0 or labels.max() >= probability.shape[1]:
        raise ValueError("label index lies outside probability columns")
    if np.any(np.char.str_len(participants) == 0):
        raise ValueError("participant ids must be non-empty")
    return labels, probability, participants


def expected_calibration_error(
    y_true: IntArray,
    probabilities: FloatArray,
    *,
    bins: int = 15,
) -> float:
    """Equal-width top-label ECE, retained as a secondary calibration diagnostic."""

    if bins < 2:
        raise ValueError("ECE requires at least two bins")
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correctness = predictions == y_true
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index, (lower, upper) in enumerate(pairwise(boundaries)):
        selected = (confidence >= lower if index == 0 else confidence > lower) & (
            confidence <= upper
        )
        if selected.any():
            error += float(selected.mean()) * abs(
                float(correctness[selected].mean()) - float(confidence[selected].mean())
            )
    return error


def selective_risk(probabilities: FloatArray, y_true: IntArray) -> dict[str, Any]:
    """Return empirical risk-coverage points and trapezoidal AURC."""

    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    order = np.argsort(-confidence, kind="stable")
    errors = (predictions[order] != y_true[order]).astype(np.float64)
    counts = np.arange(1, errors.size + 1, dtype=np.float64)
    coverage = counts / errors.size
    risk = np.cumsum(errors) / counts
    coverage_with_origin = np.concatenate((np.array([0.0]), coverage))
    risk_with_origin = np.concatenate((np.array([0.0]), risk))
    aurc = float(np.trapezoid(risk_with_origin, coverage_with_origin))
    indices = np.unique(np.linspace(0, errors.size - 1, min(errors.size, 101), dtype=int))
    return {
        "aurc": aurc,
        "points": [
            {"coverage": float(coverage[index]), "risk": float(risk[index])} for index in indices
        ],
    }


def classification_report(
    y_true: Sequence[int] | IntArray,
    probabilities: Sequence[Sequence[float]] | FloatArray,
    participant_ids: Sequence[str],
    *,
    class_names: Sequence[str],
    cohort_by_participant: dict[str, str] | None = None,
    ece_bins: int = 15,
) -> dict[str, Any]:
    """Compute window diagnostics and equally weighted participant-level endpoints."""

    labels, probability, participants = _validate_predictions(
        y_true, probabilities, participant_ids
    )
    if len(class_names) != probability.shape[1] or len(set(class_names)) != len(class_names):
        raise ValueError("class_names must uniquely align with probability columns")
    predictions = probability.argmax(axis=1)
    class_indices = np.arange(probability.shape[1], dtype=np.int64)
    recalls = recall_score(
        labels,
        predictions,
        labels=class_indices,
        average=None,
        zero_division=0,
    )
    participant_rows: list[dict[str, Any]] = []
    for participant in sorted(np.unique(participants).tolist()):
        selected = participants == participant
        participant_macro_f1 = f1_score(
            labels[selected],
            predictions[selected],
            labels=class_indices,
            average="macro",
            zero_division=0,
        )
        participant_rows.append(
            {
                "participant_id": participant,
                "window_count": int(selected.sum()),
                "macro_f1": float(participant_macro_f1),
                "balanced_accuracy": float(
                    recall_score(
                        labels[selected],
                        predictions[selected],
                        labels=class_indices,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "cohort": None
                if cohort_by_participant is None
                else cohort_by_participant.get(participant),
            }
        )
    participant_values = np.asarray([row["macro_f1"] for row in participant_rows], dtype=np.float64)
    cohort_summary: dict[str, Any] | None = None
    if cohort_by_participant is not None:
        unknown = sorted(
            participant
            for participant in np.unique(participants).tolist()
            if participant not in cohort_by_participant
        )
        if unknown:
            raise ValueError(f"cohort mapping lacks participants: {unknown}")
        grouped: dict[str, list[float]] = {}
        for row in participant_rows:
            cohort = str(row["cohort"])
            grouped.setdefault(cohort, []).append(float(row["macro_f1"]))
        cohort_summary = {
            cohort: {
                "participant_count": len(values),
                "mean_participant_macro_f1": float(np.mean(values)),
            }
            for cohort, values in sorted(grouped.items())
        }
        if "source" in cohort_summary and "target" in cohort_summary:
            cohort_summary["source_minus_target_gap"] = float(
                cohort_summary["source"]["mean_participant_macro_f1"]
                - cohort_summary["target"]["mean_participant_macro_f1"]
            )

    clipped = np.clip(probability, 1e-12, 1.0)
    one_hot = np.eye(probability.shape[1], dtype=np.float64)[labels]
    return {
        "schema_version": "1.0.0",
        "sample_count": int(labels.size),
        "participant_count": len(participant_rows),
        "class_names": list(class_names),
        "primary": {
            "mean_participant_macro_f1": float(participant_values.mean()),
            "worst_participant_macro_f1": float(participant_values.min()),
            "lower_decile_participant_macro_f1": float(
                np.quantile(participant_values, 0.1, method="linear")
            ),
        },
        "window_level_diagnostics": {
            "accuracy": float(np.mean(labels == predictions)),
            "macro_f1": float(
                f1_score(
                    labels,
                    predictions,
                    labels=class_indices,
                    average="macro",
                    zero_division=0,
                )
            ),
            "balanced_accuracy": float(recalls.mean()),
            "per_class_recall": {
                name: float(value) for name, value in zip(class_names, recalls, strict=True)
            },
            "confusion_matrix": confusion_matrix(
                labels, predictions, labels=class_indices
            ).tolist(),
        },
        "calibration": {
            "negative_log_likelihood": float(
                -np.log(clipped[np.arange(labels.size), labels]).mean()
            ),
            "multiclass_brier_score": float(np.sum((probability - one_hot) ** 2, axis=1).mean()),
            "ece": expected_calibration_error(labels, probability, bins=ece_bins),
            "ece_bins": ece_bins,
            "ece_status": "secondary_diagnostic",
        },
        "selective_risk": selective_risk(probability, labels),
        "participants": participant_rows,
        "cohorts": cohort_summary,
        "independence_note": "windows are not treated as independent statistical units",
    }
