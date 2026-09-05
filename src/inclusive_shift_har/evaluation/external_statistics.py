"""Frozen participant-cluster reporting for the external physical-grid correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (  # type: ignore[import-untyped]
    confusion_matrix,
    precision_recall_fscore_support,
)

from inclusive_shift_har.data.external_har import ExternalHARWindows
from inclusive_shift_har.evaluation.metrics import classification_report

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ParticipantMetricInputs:
    """Only retained annotations/grouping needed to reproduce reported metrics."""

    dataset_id: str
    class_names: tuple[str, ...]
    labels: NDArray[np.int64]
    participant_ids: NDArray[np.str_]


def seed_evidence(
    data: ExternalHARWindows | ParticipantMetricInputs,
    per_seed: dict[int, dict[str, FloatArray]],
    *,
    primary_contrast_eligible: bool = True,
) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    """Preserve every seed and report primary seed-averaged participant metrics.

    Bootstrap resampling uses participants as clusters and retains all their seeds.
    Class diagnostics are explicitly pooled-window diagnostics, not independent N.
    """

    if not per_seed:
        raise ValueError("seed evidence requires at least one attempted seed")
    seeds = sorted(per_seed)
    methods = sorted(per_seed[seeds[0]])
    if any(set(per_seed[seed]) != set(methods) for seed in seeds):
        raise ValueError("every seed must contain the same frozen comparison set")
    participants = np.unique(data.participant_ids)
    class_count = len(data.class_names)
    support = np.asarray(
        [
            np.bincount(data.labels[data.participant_ids == person], minlength=class_count)
            for person in participants
        ],
        dtype=np.int64,
    )
    present = support > 0
    complete = present.all(axis=1)
    bootstrap = np.random.default_rng(20260905).integers(
        0, len(participants), size=(10000, len(participants))
    )
    primary: dict[str, Any] = {}
    archive: dict[str, FloatArray] = {}
    for method in methods:
        fixed: list[list[float]] = []
        present_scores: list[list[float]] = []
        per_seed_reports: dict[str, Any] = {}
        for seed in seeds:
            probability = per_seed[seed][method]
            report = classification_report(
                data.labels,
                probability,
                data.participant_ids.tolist(),
                class_names=data.class_names,
            )
            predicted = probability.argmax(axis=1)
            precision, recall, f1, counts = precision_recall_fscore_support(
                data.labels,
                predicted,
                labels=np.arange(class_count),
                zero_division=0,
            )
            report["per_class"] = {
                name: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(counts[i]),
                }
                for i, name in enumerate(data.class_names)
            }
            fixed.append([float(row["macro_f1"]) for row in report["participants"]])
            seed_present: list[float] = []
            for person, mask in zip(participants, present, strict=True):
                selected = data.participant_ids == person
                matrix = confusion_matrix(
                    data.labels[selected], predicted[selected], labels=np.arange(class_count)
                )
                denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
                scores = np.divide(
                    2.0 * matrix.diagonal(),
                    denominator,
                    out=np.zeros(class_count),
                    where=denominator > 0,
                )
                seed_present.append(float(scores[mask].mean()))
            present_scores.append(seed_present)
            per_seed_reports[str(seed)] = report
            archive[f"seed-{seed}__{method}"] = probability
        values = np.mean(fixed, axis=0)
        tail_count = max(1, int(np.ceil(0.30 * values.size)))
        primary[method] = {
            "mean_participant_macro_f1": float(values.mean()),
            "participant_bootstrap_95_percent_ci": np.quantile(
                values[bootstrap].mean(axis=1), [0.025, 0.975]
            ).tolist(),
            "worst_participant_macro_f1": float(values.min()),
            "bottom_30_percent_participant_macro_f1": float(np.sort(values)[:tail_count].mean()),
            "participant_quartiles": np.quantile(values, [0.25, 0.5, 0.75]).tolist(),
            "participant_values": dict(zip(participants.tolist(), values.tolist(), strict=True)),
            "present_class_sensitivity_mean": float(np.mean(present_scores)),
            "complete_class_participant_count": int(complete.sum()),
            "complete_class_participant_sensitivity_mean": float(values[complete].mean())
            if complete.any()
            else None,
            "reports_by_seed": per_seed_reports,
        }
    comparisons: dict[str, Any] = {}
    if "XGBoost-6ch" in primary:
        base = np.asarray(list(primary["XGBoost-6ch"]["participant_values"].values()))
        for method in methods:
            if method == "XGBoost-6ch":
                continue
            candidate = np.asarray(list(primary[method]["participant_values"].values()))
            difference = candidate - base
            candidate_bootstrap, base_bootstrap = candidate[bootstrap], base[bootstrap]
            tail = max(1, int(np.ceil(0.30 * base.size)))
            tail_differences = np.sort(candidate_bootstrap, axis=1)[:, :tail].mean(
                axis=1
            ) - np.sort(base_bootstrap, axis=1)[:, :tail].mean(axis=1)
            mean_ci = np.quantile(difference[bootstrap].mean(axis=1), [0.025, 0.975])
            tail_ci = np.quantile(tail_differences, [0.025, 0.975])
            comparisons[method] = {
                "comparator": "XGBoost-6ch",
                "mean_difference": float(difference.mean()),
                "paired_participant_bootstrap_95_percent_ci": mean_ci.tolist(),
                "bottom_30_percent_difference_95_percent_ci": tail_ci.tolist(),
                "rescue_count": int(np.sum(difference > 1e-12)),
                "harm_count": int(np.sum(difference < -1e-12)),
                "tie_count": int(np.sum(np.abs(difference) <= 1e-12)),
                "primary_contrast": primary_contrast_eligible
                and method == "HERA-DG-full"
                and data.dataset_id == "fog_star_v3",
                "mean_and_tail_gate_passed": bool(mean_ci[0] > 0.0 and tail_ci[0] >= 0.0),
                "multiplicity_note": "only the declared FoG HERA-full contrast is primary; all others descriptive",
            }
    return {
        "estimand": "participant fixed-class macro-F1 averaged over all frozen seeds",
        "seeds": seeds,
        "participant_count": len(participants),
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 20260905,
        "participant_class_support": {
            person: dict(zip(data.class_names, counts.tolist(), strict=True))
            for person, counts in zip(participants.tolist(), support, strict=True)
        },
        "perfect_prediction_fixed_class_mean_ceiling": float(present.mean()),
        "methods": primary,
        "comparisons_vs_xgboost_6ch": comparisons,
        "seed_probability_ensemble_is_secondary": True,
    }, archive
