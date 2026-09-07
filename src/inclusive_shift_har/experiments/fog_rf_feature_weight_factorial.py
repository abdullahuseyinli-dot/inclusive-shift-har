"""Bounded corrected FoG Random Forest feature-by-weighting factorial.

The runner is intentionally separate from the canceled publication queue. It streams
the frozen FoG source once, materializes both deterministic feature lanes once, fits
exactly four fixed Random Forest cells in each of five pre-window participant folds,
and writes create-only development evidence. Validation replays saved checkpoints
against cached held-out features without downloading source data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import sklearn  # type: ignore[import-untyped]
import yaml
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
from threadpoolctl import threadpool_info, threadpool_limits  # type: ignore[import-untyped]

from inclusive_shift_har.data.external_har import load_fog_star, observable_modelling_pool
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.preprocessing.features import extract_engineered_features

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
StringArray = NDArray[np.str_]

CELL_ORDER = ("f0", "f1", "f2", "f3")
CONTROL_ORDER = ("f0", "f1", "f2")
CLASS_NAMES = ("mobility", "sitting", "standing")
SIX_CHANNEL_NAMES = (
    "lin_acc_x",
    "lin_acc_y",
    "lin_acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)
NINE_CHANNEL_NAMES = (*SIX_CHANNEL_NAMES, "gravity_x", "gravity_y", "gravity_z")
REQUIRED_RUN_FILES = (
    "analysis.json",
    "config_snapshot.yaml",
    "data_audit.json",
    "feature_cache.npz",
    "feature_cache_metadata.json",
    "fold_reports.json",
    "OUTCOME_SUMMARY.md",
    "participant_metrics.json",
    "predictions.npz",
    "preflight.json",
    "protocol_snapshot.json",
    "result.json",
    "runtime.json",
    "source_receipt.json",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value, str(path))


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(value, str(path))


def _write_bytes_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _write_json_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    body = (
        json.dumps(
            dict(payload), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        + b"\n"
    )
    _write_bytes_create_only(path, body)


def _sealed(payload: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(payload)
    _require("record_sha256" not in output, "record_sha256 is reserved")
    output["record_sha256"] = canonical_json_sha256(output)
    return output


def _verify_sealed(payload: Mapping[str, Any], name: str) -> None:
    body = dict(payload)
    declared = body.pop("record_sha256", None)
    _require(
        isinstance(declared, str) and declared == canonical_json_sha256(body),
        f"{name} self-hash is invalid",
    )


def _array_sha256(values: NDArray[Any]) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"|")
    digest.update(",".join(str(item) for item in array.shape).encode("ascii"))
    digest.update(b"|")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _git_output(repository_root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments), cwd=repository_root, text=True, encoding="utf-8"
    ).strip()


def _git_state(repository_root: Path) -> dict[str, Any]:
    status = _git_output(repository_root, "status", "--porcelain=v1")
    return {
        "branch": _git_output(repository_root, "branch", "--show-current"),
        "commit": _git_output(repository_root, "rev-parse", "HEAD"),
        "clean": not status,
        "porcelain": status.splitlines() if status else [],
    }


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_config(config: Mapping[str, Any]) -> None:
    source = _mapping(config.get("source"), "source")
    folds = _mapping(config.get("folds"), "folds")
    features = _mapping(config.get("features"), "features")
    estimator = _mapping(config.get("estimator"), "estimator")
    weighting = _mapping(config.get("weighting"), "weighting")
    cells = _mapping(config.get("cells"), "cells")
    primary = _mapping(config.get("primary"), "primary")
    statistics = _mapping(config.get("statistics"), "statistics")
    gate = _mapping(config.get("promotion_gate"), "promotion_gate")
    resources = _mapping(config.get("resource_limits"), "resource_limits")
    claims = _mapping(config.get("claims"), "claims")
    expected_roster = [f"fogstar:{index:03d}" for index in range(1, 23)]
    _require(config.get("schema_version") == "1.0.0", "unexpected config schema")
    _require(
        config.get("experiment_id") == "fog-rf-feature-weight-factorial-v1",
        "unexpected experiment id",
    )
    _require(config.get("status") == "frozen_before_outcomes", "config is not frozen")
    _require(source.get("dataset_id") == "fog_star_v3", "wrong source dataset")
    _require(
        source.get("provider_record") == "https://zenodo.org/records/17838806",
        "wrong provider record",
    )
    _require(
        source.get("content_url")
        == "https://zenodo.org/api/records/17838806/files/sensor_data.csv/content",
        "wrong source URL",
    )
    _require(source.get("declared_size_bytes") == 119_629_580, "wrong source size")
    _require(
        source.get("declared_md5") == "952a37ab147da35e6d4e7a1e9bac44cb",
        "wrong source MD5",
    )
    _require(source.get("participant_roster") == expected_roster, "wrong source roster")
    _require(source.get("class_order") == list(CLASS_NAMES), "wrong class order")
    _require(
        source.get("expected_sha256")
        == "888875133fef386affddd0a5864f122d527d8d7bc588a69905cc0871bef53477",
        "wrong source hash",
    )
    _require(source.get("target_rate_hz") == 50.0, "wrong target rate")
    _require(source.get("window_samples") == 128, "wrong window size")
    _require(source.get("gravity_cutoff_hz") == 0.30, "wrong gravity cutoff")
    _require(source.get("maximum_gap_factor") == 3.0, "wrong gap factor")
    _require(
        source.get("segmentation") == "participant_session_timestamp_gap_nonfinite_run",
        "segmentation contract changed",
    )
    _require(
        source.get("annotation_role") == "posthoc_window_label_and_scoring_eligibility_only",
        "annotation role changed",
    )
    _require(source.get("raw_local_mirror") is False, "raw mirror is forbidden")
    _require(
        source.get("participant_plan_created_before_windowing") is True, "pre-window plan required"
    )
    _require(
        source.get("resampling_and_gravity_annotation_independent") is True,
        "annotation-independent signal processing required",
    )
    _require(
        folds
        == {
            "seed": 11,
            "count": 5,
            "assignment_role": "outer",
            "assignment_source": "loader_frozen_external-har-prewindow-partitions-v1",
            "estimator_random_state": "11_plus_outer_fold_index_shared_by_all_four_cells",
        },
        "fold contract changed",
    )
    six = _mapping(features.get("six_channel"), "six_channel")
    nine = _mapping(features.get("derived_nine"), "derived_nine")
    _require(
        features.get("extractor")
        == "inclusive_shift_har.preprocessing.features.extract_engineered_features",
        "feature extractor changed",
    )
    _require(features.get("dtype") == "float64", "feature dtype changed")
    _require(
        features.get("cache_scope") == "all_annotation_independent_observable_candidates",
        "feature cache scope changed",
    )
    _require(
        six.get("signal") == "linear_acceleration_plus_gyroscope",
        "six-channel signal changed",
    )
    _require(
        nine.get("signal") == "six_channel_plus_causal_gravity_derived_from_total_acceleration",
        "derived-nine signal changed",
    )
    _require(
        tuple(cast(list[str], six.get("channel_names"))) == SIX_CHANNEL_NAMES,
        "six-channel order changed",
    )
    _require(
        tuple(cast(list[str], nine.get("channel_names"))) == NINE_CHANNEL_NAMES,
        "nine-channel order changed",
    )
    _require(
        estimator.get("library_class") == "sklearn.ensemble.RandomForestClassifier",
        "wrong estimator",
    )
    for key, expected in {
        "n_estimators": 500,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "bootstrap": True,
        "n_jobs": 4,
        "no_hyperparameter_search": True,
    }.items():
        _require(estimator.get(key) == expected, f"estimator field changed: {key}")
    _require(
        estimator.get("probability_class_order") == [0, 1, 2],
        "probability class order changed",
    )
    _require(
        _mapping(weighting.get("ordinary"), "ordinary")
        == {"class_weight": "balanced_subsample", "sample_weight": None},
        "ordinary weighting changed",
    )
    participant_first = _mapping(weighting.get("participant_first"), "participant_first")
    _require(
        participant_first.get("class_weight") is None, "participant-first class_weight must be off"
    )
    _require(
        participant_first.get("raw_sample_weight") == "1 / (m_i * n_ic)",
        "participant-first formula changed",
    )
    _require(
        participant_first.get("normalization") == "mean_one_over_outer_training_rows",
        "participant-first normalization changed",
    )
    _require(
        participant_first.get("absent_person_class_rows") == "no_rows_no_invented_weight",
        "absent participant-class handling changed",
    )
    _require(set(cells) == set(CELL_ORDER), "cell set changed")
    expected_cells = {
        "f0": ("six_channel", "ordinary"),
        "f1": ("six_channel", "participant_first"),
        "f2": ("derived_nine", "ordinary"),
        "f3": ("derived_nine", "participant_first"),
    }
    for name, (lane, weights) in expected_cells.items():
        cell = _mapping(cells[name], name)
        _require(
            (cell.get("features"), cell.get("weighting")) == (lane, weights),
            f"cell changed: {name}",
        )
    _require(primary.get("candidate") == "f3", "primary candidate changed")
    _require(primary.get("strongest_control_candidates") == list(CONTROL_ORDER), "controls changed")
    _require(
        primary.get("strongest_control_tie_priority") == list(CONTROL_ORDER),
        "control tie order changed",
    )
    _require(primary.get("tie_tolerance") == 1e-12, "tie tolerance changed")
    _require(
        statistics.get("paired_participant_bootstrap_resamples") == 10_000,
        "bootstrap count changed",
    )
    _require(statistics.get("paired_participant_bootstrap_seed") == 1729, "bootstrap seed changed")
    _require(
        statistics.get("percentile_interval") == [0.025, 0.975],
        "bootstrap interval changed",
    )
    _require(statistics.get("bottom_fraction") == 0.30, "bottom fraction changed")
    _require(
        statistics.get("missing_class_metric_rule") == "fixed_class_zero_division_zero",
        "missing-class rule changed",
    )
    _require(
        statistics.get("zero_scored_window_rule") == "promotion_incomplete_never_drop_person",
        "zero-window rule changed",
    )
    for key, expected in {
        "minimum_mean_gain": 0.015,
        "minimum_bottom_30_difference": -0.010,
        "minimum_worst_participant_difference": -0.030,
        "minimum_mobility_recall_difference": -0.010,
        "minimum_sitting_recall_difference": -0.020,
        "minimum_standing_recall_difference": -0.020,
        "minimum_participant_wins_if_full_roster": 14,
        "leave_one_out_positive_tolerance": 1e-12,
        "require_all_leave_one_participant_out_means_positive": True,
        "require_all_22_roster_people_scored": True,
    }.items():
        _require(gate.get(key) == expected, f"gate field changed: {key}")
    _require(resources.get("controller_count") == 1, "one controller required")
    _require(resources.get("maximum_model_fits") == 20, "fit budget changed")
    _require(resources.get("compute_wall_seconds") == 7200, "compute cap changed")
    _require(resources.get("cpu_estimator_workers") == 4, "worker count changed")
    _require(resources.get("prediction_workers") == 1, "prediction worker count changed")
    _require(resources.get("blas_threads") == 1, "BLAS thread count changed")
    _require(resources.get("gpu_allowed") is False, "GPU is not allowed")
    _require(resources.get("additional_seeds_allowed") is False, "additional seeds forbidden")
    _require(resources.get("automatic_follow_on_allowed") is False, "automatic follow-on forbidden")
    _require(all(value is False for value in claims.values()), "claim policy changed")


def participant_first_weights(labels: IntArray, participant_ids: StringArray) -> FloatArray:
    """Return exact mean-one 1/(m_i*n_ic) outer-training weights."""

    y = np.asarray(labels, dtype=np.int64)
    people = np.asarray(participant_ids, dtype=np.str_)
    _require(
        y.ndim == people.ndim == 1 and y.size == people.size and y.size > 0,
        "weight arrays are not aligned",
    )
    _require(set(np.unique(y).tolist()) == {0, 1, 2}, "training partition lacks a class")
    weights = np.empty(y.size, dtype=np.float64)
    for participant in np.unique(people):
        selected = people == participant
        present = np.unique(y[selected])
        class_count = int(present.size)
        for label in present:
            cell = selected & (y == label)
            weights[cell] = 1.0 / (class_count * int(cell.sum()))
    weights /= weights.mean()
    _require(
        bool(np.isfinite(weights).all() and np.all(weights > 0)),
        "participant-first weights are invalid",
    )
    return weights


def _confusion(labels: IntArray, predictions: IntArray) -> IntArray:
    _require(
        labels.shape == predictions.shape and labels.ndim == 1, "confusion arrays are not aligned"
    )
    return np.bincount(3 * labels + predictions, minlength=9).reshape(3, 3).astype(np.int64)


def _scores_from_confusion(confusion: IntArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    true_support = confusion.sum(axis=1).astype(np.float64)
    predicted_support = confusion.sum(axis=0).astype(np.float64)
    diagonal = np.diag(confusion).astype(np.float64)
    recall = np.divide(diagonal, true_support, out=np.zeros(3), where=true_support > 0)
    precision = np.divide(diagonal, predicted_support, out=np.zeros(3), where=predicted_support > 0)
    denominator = recall + precision
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros(3), where=denominator > 0)
    return precision, recall, f1


def _bottom(values: FloatArray, fraction: float = 0.30) -> float:
    count = math.ceil(fraction * values.size)
    return float(np.mean(np.sort(values)[:count]))


def method_report(
    *,
    labels: IntArray,
    probabilities: FloatArray,
    participant_ids: StringArray,
    roster: Sequence[str],
) -> dict[str, Any]:
    """Compute fixed-three-class, participant-first reporting over the full roster."""

    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    people = np.asarray(participant_ids, dtype=np.str_)
    expected_roster = tuple(roster)
    _require(
        y.ndim == 1 and y.size > 0 and p.shape == (y.size, 3) and people.shape == y.shape,
        "report arrays are not aligned",
    )
    _require(
        set(np.unique(people).tolist()).issubset(expected_roster),
        "scored participant is outside the frozen roster",
    )
    _require(bool(np.isfinite(p).all() and np.all(p >= 0)), "probabilities are invalid")
    _require(
        np.allclose(p.sum(axis=1), 1.0, rtol=0.0, atol=1e-12), "probabilities do not sum to one"
    )
    _require(set(np.unique(y).tolist()).issubset({0, 1, 2}), "labels leave class schema")
    prediction = p.argmax(axis=1).astype(np.int64)
    rows: list[dict[str, Any]] = []
    available_values: list[float] = []
    available_recalls: list[FloatArray] = []
    available_nll: list[float] = []
    available_brier: list[float] = []
    available_present_class_f1: list[float] = []
    all_class_participant_count = 0
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
                }
            )
            continue
        confusion = _confusion(y[selected], prediction[selected])
        precision, recall, f1 = _scores_from_confusion(confusion)
        clipped = np.clip(p[selected], 1e-12, 1.0)
        selected_labels = y[selected]
        nll = float(-np.log(clipped[np.arange(selected_labels.size), selected_labels]).mean())
        one_hot = np.eye(3, dtype=np.float64)[selected_labels]
        brier = float(np.sum((p[selected] - one_hot) ** 2, axis=1).mean())
        value = float(f1.mean())
        present = confusion.sum(axis=1) > 0
        present_class_value = float(f1[present].mean())
        all_class_participant_count += int(present.all())
        available_values.append(value)
        available_recalls.append(recall)
        available_nll.append(nll)
        available_brier.append(brier)
        available_present_class_f1.append(present_class_value)
        rows.append(
            {
                "participant_id": participant,
                "eligible": True,
                "window_count": int(selected.sum()),
                "class_support": confusion.sum(axis=1).tolist(),
                "confusion_matrix": confusion.tolist(),
                "class_precision": precision.tolist(),
                "class_recall": recall.tolist(),
                "class_f1": f1.tolist(),
                "macro_f1": value,
                "present_class_macro_f1": present_class_value,
                "nll": nll,
                "multiclass_brier": brier,
            }
        )
    pooled = _confusion(y, prediction)
    pooled_precision, pooled_recall, pooled_f1 = _scores_from_confusion(pooled)
    missing = [str(row["participant_id"]) for row in rows if not bool(row["eligible"])]
    complete = not missing and len(available_values) == len(expected_roster)
    output: dict[str, Any] = {
        "status": "complete_full_roster"
        if complete
        else "incomplete_zero_scored_roster_participant",
        "roster_count": len(expected_roster),
        "scored_participant_count": len(available_values),
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
            "accuracy": float(np.mean(prediction == y)),
            "nll": float(-np.log(np.clip(p, 1e-12, 1.0)[np.arange(y.size), y]).mean()),
            "multiclass_brier": float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1))),
        },
        "fixed_class_rule": "zero_division_zero",
    }
    if complete:
        values = np.asarray(available_values, dtype=np.float64)
        recalls = np.stack(available_recalls)
        output["primary"] = {
            "mean_participant_macro_f1": float(values.mean()),
            "median_participant_macro_f1": float(np.median(values)),
            "quartiles_participant_macro_f1": [
                float(np.quantile(values, 0.25)),
                float(np.quantile(values, 0.75)),
            ],
            "bottom_30_percent_participant_macro_f1": _bottom(values),
            "bottom_30_participant_count": math.ceil(0.30 * values.size),
            "worst_participant_macro_f1": float(values.min()),
            "participant_macro_class_recall": dict(
                zip(CLASS_NAMES, np.mean(recalls, axis=0).tolist(), strict=True)
            ),
            "mean_participant_nll": float(np.mean(available_nll)),
            "mean_participant_multiclass_brier": float(np.mean(available_brier)),
            "mean_present_class_participant_macro_f1": float(np.mean(available_present_class_f1)),
            "participants_with_all_three_classes": all_class_participant_count,
        }
    else:
        output["primary"] = None
    return output


def _participant_values(report: Mapping[str, Any]) -> FloatArray:
    rows = cast(list[dict[str, Any]], report["participants"])
    _require(all(row.get("eligible") is True for row in rows), "comparison requires full roster")
    return np.asarray([float(row["macro_f1"]) for row in rows], dtype=np.float64)


def paired_comparison(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    bootstrap_seed: int = 1729,
    resamples: int = 10_000,
) -> dict[str, Any]:
    left_values = _participant_values(left)
    right_values = _participant_values(right)
    _require(left_values.shape == right_values.shape, "participant comparison shapes differ")
    differences = left_values - right_values
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, differences.size, size=(resamples, differences.size))
    left_draw = left_values[indices]
    right_draw = right_values[indices]
    difference_draw = left_draw - right_draw
    mean_draw = np.mean(difference_draw, axis=1)
    count = math.ceil(0.30 * differences.size)
    bottom_draw = np.mean(np.sort(left_draw, axis=1)[:, :count], axis=1) - np.mean(
        np.sort(right_draw, axis=1)[:, :count], axis=1
    )
    worst_draw = np.min(left_draw, axis=1) - np.min(right_draw, axis=1)
    tolerance = 1e-12
    left_primary = _mapping(left["primary"], "left primary")
    right_primary = _mapping(right["primary"], "right primary")
    left_recall = _mapping(left_primary["participant_macro_class_recall"], "left recall")
    right_recall = _mapping(right_primary["participant_macro_class_recall"], "right recall")
    loo = [float(np.delete(differences, index).mean()) for index in range(differences.size)]
    return {
        "participant_differences": differences.tolist(),
        "mean_difference": float(differences.mean()),
        "median_paired_difference": float(np.median(differences)),
        "mean_difference_95_percent_bootstrap_interval": [
            float(np.quantile(mean_draw, 0.025)),
            float(np.quantile(mean_draw, 0.975)),
        ],
        "bottom_30_percent_difference": float(
            left_primary["bottom_30_percent_participant_macro_f1"]
            - right_primary["bottom_30_percent_participant_macro_f1"]
        ),
        "bottom_30_difference_95_percent_bootstrap_interval": [
            float(np.quantile(bottom_draw, 0.025)),
            float(np.quantile(bottom_draw, 0.975)),
        ],
        "worst_participant_difference": float(
            left_primary["worst_participant_macro_f1"] - right_primary["worst_participant_macro_f1"]
        ),
        "worst_difference_95_percent_bootstrap_interval": [
            float(np.quantile(worst_draw, 0.025)),
            float(np.quantile(worst_draw, 0.975)),
        ],
        "class_recall_differences": {
            name: float(left_recall[name] - right_recall[name]) for name in CLASS_NAMES
        },
        "participant_wins": int(np.sum(differences > tolerance)),
        "participant_harms": int(np.sum(differences < -tolerance)),
        "participant_ties": int(np.sum(np.abs(differences) <= tolerance)),
        "minimum_paired_participant_difference": float(differences.min()),
        "maximum_paired_participant_difference": float(differences.max()),
        "leave_one_participant_out_mean_differences": loo,
        "leave_one_out_minimum": float(min(loo)),
        "bootstrap": {
            "unit": "participant",
            "resamples": resamples,
            "seed": bootstrap_seed,
            "interval": "95_percentile_linear",
        },
    }


def analyse(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct fixed comparisons, factorial contrasts and the promotion gate."""

    roster = cast(list[str], _mapping(config["source"], "source")["participant_roster"])
    complete = all(
        report.get("status") == "complete_full_roster"
        and report.get("scored_participant_count") == len(roster)
        for report in reports.values()
    )
    comparisons: dict[str, Any] = {}
    factorial: dict[str, Any] = {}
    strongest: str | None = None
    gate: dict[str, Any]
    if complete:
        means = {
            name: float(_mapping(reports[name]["primary"], "primary")["mean_participant_macro_f1"])
            for name in CELL_ORDER
        }
        best = max(means[name] for name in CONTROL_ORDER)
        tolerance = 1e-12
        strongest = next(name for name in CONTROL_ORDER if means[name] >= best - tolerance)
        comparisons = {
            f"f3_minus_{name}": paired_comparison(reports["f3"], reports[name])
            for name in CONTROL_ORDER
        }
        participant_values = {name: _participant_values(reports[name]) for name in CELL_ORDER}

        def contrast(values: FloatArray) -> dict[str, Any]:
            rng = np.random.default_rng(1729)
            indices = rng.integers(0, values.size, size=(10_000, values.size))
            draw = values[indices].mean(axis=1)
            return {
                "participant_differences": values.tolist(),
                "mean_difference": float(values.mean()),
                "mean_difference_95_percent_bootstrap_interval": [
                    float(np.quantile(draw, 0.025)),
                    float(np.quantile(draw, 0.975)),
                ],
                "participant_wins": int(np.sum(values > 1e-12)),
                "participant_harms": int(np.sum(values < -1e-12)),
                "participant_ties": int(np.sum(np.abs(values) <= 1e-12)),
            }

        factorial = {
            "derived_nine_minus_six_ordinary": contrast(
                participant_values["f2"] - participant_values["f0"]
            ),
            "derived_nine_minus_six_participant_first": contrast(
                participant_values["f3"] - participant_values["f1"]
            ),
            "participant_first_minus_ordinary_six": contrast(
                participant_values["f1"] - participant_values["f0"]
            ),
            "participant_first_minus_ordinary_derived_nine": contrast(
                participant_values["f3"] - participant_values["f2"]
            ),
            "difference_in_differences": contrast(
                (participant_values["f3"] - participant_values["f1"])
                - (participant_values["f2"] - participant_values["f0"])
            ),
        }
        primary = _mapping(comparisons[f"f3_minus_{strongest}"], "primary comparison")
        class_differences = _mapping(primary["class_recall_differences"], "class diffs")
        gate_config = _mapping(config["promotion_gate"], "promotion gate")
        checks = {
            "full_22_person_roster_scored": True,
            "mean_gain": float(primary["mean_difference"])
            >= float(gate_config["minimum_mean_gain"]),
            "bottom_30": float(primary["bottom_30_percent_difference"])
            >= float(gate_config["minimum_bottom_30_difference"]),
            "worst_participant": float(primary["worst_participant_difference"])
            >= float(gate_config["minimum_worst_participant_difference"]),
            "mobility_recall": float(class_differences["mobility"])
            >= float(gate_config["minimum_mobility_recall_difference"]),
            "sitting_recall": float(class_differences["sitting"])
            >= float(gate_config["minimum_sitting_recall_difference"]),
            "standing_recall": float(class_differences["standing"])
            >= float(gate_config["minimum_standing_recall_difference"]),
            "participant_wins": int(primary["participant_wins"])
            >= int(gate_config["minimum_participant_wins_if_full_roster"]),
            "all_leave_one_participant_out_positive": all(
                float(value) > float(gate_config["leave_one_out_positive_tolerance"])
                for value in cast(
                    list[float], primary["leave_one_participant_out_mean_differences"]
                )
            ),
        }
        gate = {
            "status": "pass" if all(checks.values()) else "fail",
            "candidate": "f3",
            "strongest_control": strongest,
            "checks": checks,
            "passed_check_count": sum(checks.values()),
            "required_check_count": len(checks),
            "failure_action": "close_factorial_no_additional_seed_or_automatic_follow_on",
        }
    else:
        gate = {
            "status": "incomplete",
            "candidate": "f3",
            "strongest_control": None,
            "checks": {"full_22_person_roster_scored": False},
            "passed_check_count": 0,
            "required_check_count": 9,
            "failure_action": "retain_complete_matrix_but_do_not_promote_or_drop_zero_window_people",
        }
    return {
        "status": "complete_analysis" if complete else "incomplete_promotion_analysis",
        "strongest_control": strongest,
        "reports": dict(reports),
        "comparisons": comparisons,
        "factorial_contrasts": factorial,
        "promotion_gate": gate,
        "evidence_status": "corrected_fog_development_baseline_not_confirmation",
    }


def _effective_estimator_parameters(
    *, class_weight: str | None, random_state: int, n_jobs: int
) -> dict[str, Any]:
    estimator = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        bootstrap=True,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return cast(dict[str, Any], estimator.get_params(deep=False))


def _predict_proba_deterministically(
    estimator: RandomForestClassifier, feature_values: FloatArray
) -> FloatArray:
    """Use a fixed tree-reduction order while retaining four-worker fitting."""

    fit_worker_count = estimator.n_jobs
    try:
        estimator.n_jobs = 1
        return np.asarray(estimator.predict_proba(feature_values), dtype=np.float64)
    finally:
        estimator.n_jobs = fit_worker_count


def _fit_cell(
    *,
    feature_values: FloatArray,
    labels: IntArray,
    participants: StringArray,
    training: BoolArray,
    evaluation: BoolArray,
    cell_id: str,
    fold_index: int,
    n_jobs: int,
    output_directory: Path,
    feature_names: tuple[str, ...],
    deadline: float,
) -> tuple[FloatArray, dict[str, Any]]:
    if time.perf_counter() >= deadline:
        raise TimeoutError("compute cap reached before starting the next registered fit")
    participant_first = cell_id in {"f1", "f3"}
    class_weight = None if participant_first else "balanced_subsample"
    random_state = 11 + fold_index
    estimator = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        bootstrap=True,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    sample_weight = (
        participant_first_weights(labels[training], participants[training])
        if participant_first
        else None
    )
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        fit_started = time.perf_counter()
        estimator.fit(
            feature_values[training],
            labels[training],
            sample_weight=sample_weight,
        )
        fit_seconds = time.perf_counter() - fit_started
        prediction_started = time.perf_counter()
        probabilities = _predict_proba_deterministically(estimator, feature_values[evaluation])
        prediction_seconds = time.perf_counter() - prediction_started
    elapsed = time.perf_counter() - started
    if time.perf_counter() > deadline:
        raise TimeoutError("compute cap reached during registered fit or prediction")
    _require(np.array_equal(estimator.classes_, np.arange(3)), "RF class order changed")
    _require(
        bool(
            probabilities.shape == (int(evaluation.sum()), 3)
            and np.isfinite(probabilities).all()
            and np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
        ),
        "RF probabilities are invalid",
    )
    metadata = {
        "cell_id": cell_id,
        "outer_fold": fold_index,
        "random_state": random_state,
        "feature_names": list(feature_names),
        "feature_names_sha256": canonical_json_sha256(list(feature_names)),
        "training_participants": sorted(np.unique(participants[training]).tolist()),
        "training_row_count": int(training.sum()),
        "evaluation_participants": sorted(np.unique(participants[evaluation]).tolist()),
        "evaluation_candidate_count": int(evaluation.sum()),
        "class_weight": class_weight,
        "sample_weight_policy": "participant_first_mean_one" if participant_first else None,
        "sample_weight_summary": None
        if sample_weight is None
        else {
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
    payload = {"metadata": metadata, "estimator": estimator}
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("xb") as stream:
        pickle.dump(payload, stream, protocol=5)
    metadata["checkpoint"] = {
        "path": checkpoint_path.relative_to(output_directory).as_posix(),
        "sha256": sha256_file(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
    }
    return probabilities, metadata


def _outcome_summary(analysis: Mapping[str, Any]) -> str:
    gate = _mapping(analysis["promotion_gate"], "gate")
    lines = [
        "# Corrected FoG RF feature-by-weighting factorial outcome",
        "",
        f"Promotion status: **{str(gate['status']).upper()}**.",
        "",
        "| Cell | Mean participant macro-F1 | Bottom 30% | Worst participant |",
        "|---|---:|---:|---:|",
    ]
    reports = _mapping(analysis["reports"], "reports")
    for name in CELL_ORDER:
        report = _mapping(reports[name], name)
        primary = report.get("primary")
        if isinstance(primary, Mapping):
            lines.append(
                f"| {name.upper()} | {float(primary['mean_participant_macro_f1']):.6f} | "
                f"{float(primary['bottom_30_percent_participant_macro_f1']):.6f} | "
                f"{float(primary['worst_participant_macro_f1']):.6f} |"
            )
        else:
            lines.append(f"| {name.upper()} | unavailable | unavailable | unavailable |")
    lines.extend(
        [
            "",
            f"Strongest control: {gate.get('strongest_control')}.",
            "",
            "Evidence is corrected FoG development baseline evidence. No additional seed,",
            "neural model, transfer job, or publication queue was launched.",
            "",
        ]
    )
    return "\n".join(lines)


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "threadpool_info": threadpool_info(),
        "cpu_count": os.cpu_count(),
    }


def _snapshot_inputs(
    *, repository_root: Path, config_path: Path, protocol_path: Path, output_directory: Path
) -> None:
    _write_bytes_create_only(output_directory / "config_snapshot.yaml", config_path.read_bytes())
    protocol_text = protocol_path.read_text(encoding="utf-8")
    _write_json_create_only(
        output_directory / "protocol_snapshot.json",
        _sealed(
            {
                "record_kind": "fog_rf_factorial_protocol_snapshot",
                "source_path": protocol_path.relative_to(repository_root).as_posix(),
                "source_sha256": sha256_file(protocol_path),
                "text": protocol_text,
            }
        ),
    )


def _source_manifest(repository_root: Path) -> dict[str, Any]:
    paths = [
        "AGENTS.md",
        "src/inclusive_shift_har/data/external_har.py",
        "src/inclusive_shift_har/data/participant_partitions.py",
        "src/inclusive_shift_har/preprocessing/features.py",
        "src/inclusive_shift_har/experiments/fog_rf_feature_weight_factorial.py",
        "configs/experiments/fog_rf_feature_weight_factorial_v1.yaml",
        "docs/research/FOG_RF_FEATURE_WEIGHT_FACTORIAL_V1_PROTOCOL.md",
    ]
    return {
        "files": [{"path": name, "sha256": sha256_file(repository_root / name)} for name in paths]
    }


def _feature_cache_key(
    *,
    source_sha256: str,
    participant_plan_sha256: str,
    observable_window_ids_sha256: str,
    config_sha256: str,
    feature_code_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema": "fog-rf-feature-cache-key-v1",
        "source_sha256": source_sha256,
        "participant_plan_sha256": participant_plan_sha256,
        "observable_window_ids_sha256": observable_window_ids_sha256,
        "config_sha256": config_sha256,
        "feature_code_sha256": feature_code_sha256,
        "lanes": {
            "six_channel": list(SIX_CHANNEL_NAMES),
            "derived_nine": list(NINE_CHANNEL_NAMES),
        },
        "dtype": "float64",
    }
    return {**payload, "cache_key_sha256": canonical_json_sha256(payload)}


def run_experiment(
    *,
    repository_root: Path,
    evidence_root: Path,
    config_path: Path,
    protocol_path: Path,
    output_directory: Path,
    code_commit: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Run the exact frozen matrix and write create-only artifacts."""

    repository_root = repository_root.resolve()
    evidence_root = evidence_root.resolve()
    config_path = config_path.resolve()
    protocol_path = protocol_path.resolve()
    output_directory = output_directory.resolve()
    config = _read_yaml(config_path)
    validate_config(config)
    _require(
        config_path.parent.parent.parent == repository_root, "config must be inside repository"
    )
    _require(
        protocol_path.parent.parent.parent == repository_root, "protocol must be inside repository"
    )
    allowed_output = evidence_root / ".audit" / "fog_rf_feature_weight_factorial"
    _require(
        output_directory.parent == allowed_output,
        "output must be one create-only run below the dedicated evidence root",
    )
    _require(not output_directory.exists(), "output directory already exists")
    state = _git_state(repository_root)
    _require(state["clean"] is True, "experiment source worktree must be clean")
    _require(state["commit"] == code_commit, "code commit does not match HEAD")
    _require(timeout_seconds == 7200, "runtime cap must remain 7200 seconds")
    output_directory.mkdir(parents=True, exist_ok=False)
    started_at = _now()
    started = time.perf_counter()
    deadline = started + timeout_seconds
    _snapshot_inputs(
        repository_root=repository_root,
        config_path=config_path,
        protocol_path=protocol_path,
        output_directory=output_directory,
    )
    try:
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
        _require(len(data.receipts) == 1, "expected exactly one FoG source receipt")
        receipt = data.receipts[0]
        receipt.validate()
        _require(
            receipt.computed_sha256 == source_config["expected_sha256"], "source SHA-256 mismatch"
        )
        _require(
            receipt.received_size_bytes == source_config["declared_size_bytes"],
            "source byte count mismatch",
        )
        _require(
            receipt.computed_declared_digest == source_config["declared_md5"],
            "source MD5 mismatch",
        )
        roster = tuple(cast(list[str], source_config["participant_roster"]))
        plan = data.participant_partition_plan
        if plan is None:
            raise ValueError("pre-window participant plan missing")
        _require(plan.participant_roster == roster, "pre-window roster differs from protocol")
        plan_audit = plan.audit()
        modelling, scoring_indices, eligibility = observable_modelling_pool(
            data, include_supervised_labels=True
        )
        _require(
            scoring_indices.size == data.labels.size
            and np.array_equal(modelling.window_ids[scoring_indices], data.window_ids)
            and np.array_equal(modelling.participant_ids[scoring_indices], data.participant_ids)
            and np.array_equal(modelling.labels[scoring_indices], data.labels),
            "scored-to-observable alignment failed",
        )
        assignment = plan.resolve(roster, fold_count=5, seed=11, role="outer")
        observable_fold = np.asarray(
            [assignment[str(item)] for item in modelling.participant_ids], dtype=np.int64
        )
        scored_fold = observable_fold[scoring_indices]
        source_receipt = _sealed(
            {
                "record_kind": "fog_rf_factorial_source_receipt",
                "source_receipts": [item.to_dict() for item in data.receipts],
                "expected_sha256": source_config["expected_sha256"],
                "source_hash_match": True,
                "raw_local_mirror": False,
                "target_or_other_external_dataset_loaded": False,
            }
        )
        _write_json_create_only(output_directory / "source_receipt.json", source_receipt)
        audit = _sealed(
            {
                "record_kind": "fog_rf_factorial_data_audit",
                "dataset": data.summary(),
                "participant_partition_plan": plan_audit,
                "observable_candidate_count": int(modelling.labels.size),
                "scored_window_count": int(scoring_indices.size),
                "scored_participants": sorted(np.unique(data.participant_ids).tolist()),
                "roster_participants_without_scored_rows": sorted(
                    set(roster) - set(data.participant_ids.tolist())
                ),
                "observable_processing_annotation_independent": True,
                "scoring_labels_applied_posthoc": True,
                "source_load_seconds": source_load_seconds,
            }
        )
        _write_json_create_only(output_directory / "data_audit.json", audit)

        feature_started = time.perf_counter()
        six = extract_engineered_features(modelling.signals, channel_names=SIX_CHANNEL_NAMES)
        nine = extract_engineered_features(
            modelling.nine_channel_signals, channel_names=NINE_CHANNEL_NAMES
        )
        feature_seconds = time.perf_counter() - feature_started
        _require(time.perf_counter() < deadline, "compute cap reached during feature extraction")
        cache_path = output_directory / "feature_cache.npz"
        with cache_path.open("xb") as stream:
            np.savez_compressed(
                stream,
                six_values=np.asarray(six.values, dtype=np.float64),
                nine_values=np.asarray(nine.values, dtype=np.float64),
                six_names=np.asarray(six.names, dtype=np.str_),
                nine_names=np.asarray(nine.names, dtype=np.str_),
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
        feature_metadata = _sealed(
            {
                "record_kind": "fog_rf_factorial_feature_cache",
                "path": cache_path.name,
                "sha256": sha256_file(cache_path),
                "six_shape": list(six.values.shape),
                "nine_shape": list(nine.values.shape),
                "six_names": list(six.names),
                "nine_names": list(nine.names),
                "six_values_sha256": _array_sha256(six.values),
                "nine_values_sha256": _array_sha256(nine.values),
                "observable_window_ids_sha256": _array_sha256(modelling.window_ids),
                "scoring_indices_sha256": _array_sha256(scoring_indices),
                "scoring_eligibility_sha256": _array_sha256(eligibility),
                "scored_labels_sha256": _array_sha256(data.labels),
                "scored_participant_ids_sha256": _array_sha256(data.participant_ids),
                "cache_key": _feature_cache_key(
                    source_sha256=receipt.computed_sha256,
                    participant_plan_sha256=str(plan_audit["plan_sha256"]),
                    observable_window_ids_sha256=_array_sha256(modelling.window_ids),
                    config_sha256=sha256_file(config_path),
                    feature_code_sha256=sha256_file(
                        repository_root / "src/inclusive_shift_har/preprocessing/features.py"
                    ),
                ),
                "feature_extraction_seconds": feature_seconds,
                "cache_rebuilt_for_corrected_run": True,
                "old_feature_cache_used": False,
            }
        )
        _write_json_create_only(output_directory / "feature_cache_metadata.json", feature_metadata)

        cell_config = _mapping(config["cells"], "cells")
        effective = {
            f"{cell_id}--fold-{fold}": {
                "cell_id": cell_id,
                "outer_fold": fold,
                "feature_lane": _mapping(cell_config[cell_id], cell_id)["features"],
                "weighting": _mapping(cell_config[cell_id], cell_id)["weighting"],
                "parameters": _effective_estimator_parameters(
                    class_weight=None if cell_id in {"f1", "f3"} else "balanced_subsample",
                    random_state=11 + fold,
                    n_jobs=4,
                ),
            }
            for cell_id in CELL_ORDER
            for fold in range(5)
        }
        preflight = _sealed(
            {
                "record_kind": "fog_rf_factorial_preflight",
                "created_before_model_fits": True,
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
                "data_audit_record_sha256": audit["record_sha256"],
                "feature_cache_record_sha256": feature_metadata["record_sha256"],
                "participant_partition_plan_sha256": plan_audit["plan_sha256"],
                "fold_assignment": assignment,
                "effective_estimator_parameter_sets": effective,
                "environment": _environment(),
                "resource_limits": config["resource_limits"],
                "expected_fit_count": 20,
                "outcomes_available_when_written": False,
            }
        )
        _write_json_create_only(output_directory / "preflight.json", preflight)

        probabilities = np.full((4, modelling.labels.size, 3), np.nan, dtype=np.float64)
        fold_reports: list[dict[str, Any]] = []
        fit_count = 0
        fit_started = time.perf_counter()
        feature_by_cell = {
            "f0": np.asarray(six.values, dtype=np.float64),
            "f1": np.asarray(six.values, dtype=np.float64),
            "f2": np.asarray(nine.values, dtype=np.float64),
            "f3": np.asarray(nine.values, dtype=np.float64),
        }
        names_by_cell = {
            "f0": six.names,
            "f1": six.names,
            "f2": nine.names,
            "f3": nine.names,
        }
        for fold_index in range(5):
            evaluation = observable_fold == fold_index
            training = (~evaluation) & eligibility
            _require(evaluation.any(), "outer fold has no observable candidate")
            _require(
                set(np.unique(modelling.labels[training]).tolist()) == {0, 1, 2},
                "outer training fold lacks a class",
            )
            for cell_index, cell_id in enumerate(CELL_ORDER):
                values, fold_record = _fit_cell(
                    feature_values=feature_by_cell[cell_id],
                    labels=modelling.labels,
                    participants=modelling.participant_ids,
                    training=training,
                    evaluation=evaluation,
                    cell_id=cell_id,
                    fold_index=fold_index,
                    n_jobs=4,
                    output_directory=output_directory,
                    feature_names=names_by_cell[cell_id],
                    deadline=deadline,
                )
                probabilities[cell_index, evaluation] = values
                fold_record["training_scored_class_counts"] = np.bincount(
                    modelling.labels[training], minlength=3
                ).tolist()
                fold_record["evaluation_scored_window_count"] = int(
                    np.sum(evaluation & eligibility)
                )
                fold_record["participant_partition_plan_sha256"] = plan_audit["plan_sha256"]
                fold_reports.append(fold_record)
                fit_count += 1
        fit_seconds = time.perf_counter() - fit_started
        _require(fit_count == 20, "registered fit matrix is incomplete")
        _require(time.perf_counter() <= deadline, "compute cap exceeded by registered matrix")
        _require(
            bool(np.isfinite(probabilities).all()),
            "observable predictions are incomplete",
        )
        scored_probability = probabilities[:, scoring_indices]
        prediction_path = output_directory / "predictions.npz"
        with prediction_path.open("xb") as stream:
            np.savez_compressed(
                stream,
                cell_ids=np.asarray(CELL_ORDER, dtype=np.str_),
                observable_probabilities=probabilities,
                scored_probabilities=scored_probability,
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
                "record_kind": "fog_rf_factorial_fold_reports",
                "fit_count": fit_count,
                "fold_count": 5,
                "cell_count": 4,
                "rows": fold_reports,
            }
        )
        _write_json_create_only(output_directory / "fold_reports.json", fold_payload)
        reports = {
            cell_id: method_report(
                labels=data.labels,
                probabilities=scored_probability[index],
                participant_ids=data.participant_ids,
                roster=roster,
            )
            for index, cell_id in enumerate(CELL_ORDER)
        }
        participant_payload = _sealed(
            {
                "record_kind": "fog_rf_factorial_participant_metrics",
                "methods": reports,
            }
        )
        _write_json_create_only(output_directory / "participant_metrics.json", participant_payload)
        analysis = _sealed(
            {
                "record_kind": "fog_rf_factorial_analysis",
                **analyse(reports, config=config),
            }
        )
        _write_json_create_only(output_directory / "analysis.json", analysis)
        summary = _outcome_summary(analysis)
        _write_bytes_create_only(output_directory / "OUTCOME_SUMMARY.md", summary.encode("utf-8"))
        runtime = _sealed(
            {
                "record_kind": "fog_rf_factorial_runtime",
                "started_at_utc": started_at,
                "completed_at_utc": _now(),
                "source_load_seconds": source_load_seconds,
                "feature_extraction_seconds": feature_seconds,
                "fit_and_prediction_seconds": fit_seconds,
                "total_run_seconds": time.perf_counter() - started,
                "compute_cap_seconds": timeout_seconds,
                "cap_exceeded": False,
                "controller_count": 1,
                "model_fit_count": fit_count,
                "gpu_used": False,
                "estimator_workers": 4,
                "blas_threads_requested": 1,
            }
        )
        _write_json_create_only(output_directory / "runtime.json", runtime)
        result = _sealed(
            {
                "record_kind": "fog_rf_factorial_result",
                "status": "complete_awaiting_independent_replay",
                "experiment_id": config["experiment_id"],
                "evidence_status": config["evidence_status"],
                "code_commit": code_commit,
                "source_sha256": receipt.computed_sha256,
                "participant_count_in_roster": len(roster),
                "scored_participant_count": len(np.unique(data.participant_ids)),
                "observable_candidate_count": int(modelling.labels.size),
                "scored_window_count": int(data.labels.size),
                "fit_count": fit_count,
                "feature_dimensions": {
                    "six_channel": int(six.values.shape[1]),
                    "derived_nine": int(nine.values.shape[1]),
                },
                "checkpoint_total_size_bytes": int(
                    sum(
                        int(_mapping(row["checkpoint"], "checkpoint")["size_bytes"])
                        for row in fold_reports
                    )
                ),
                "runtime_by_cell": {
                    cell_id: {
                        "fit_seconds": float(
                            sum(
                                float(row["fit_seconds"])
                                for row in fold_reports
                                if row["cell_id"] == cell_id
                            )
                        ),
                        "prediction_seconds": float(
                            sum(
                                float(row["prediction_seconds"])
                                for row in fold_reports
                                if row["cell_id"] == cell_id
                            )
                        ),
                    }
                    for cell_id in CELL_ORDER
                },
                "method_summary": {name: report["primary"] for name, report in reports.items()},
                "strongest_control": analysis["strongest_control"],
                "promotion_gate": analysis["promotion_gate"],
                "analysis_record_sha256": analysis["record_sha256"],
                "prediction_artifact": {
                    "path": prediction_path.name,
                    "sha256": sha256_file(prediction_path),
                },
                "additional_seeds_launched": False,
                "automatic_follow_on_launched": False,
                "publication_queue_launched": False,
                "prompted_c3_continued": False,
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
                "record_kind": "fog_rf_factorial_run_artifact_manifest",
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
                "record_kind": "fog_rf_factorial_failure",
                "status": "failed_preserved",
                "started_at_utc": started_at,
                "failed_at_utc": _now(),
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.perf_counter() - started,
                "compute_cap_seconds": timeout_seconds,
                "automatic_retry": False,
                "completed_model_fit_checkpoints": len(
                    list((output_directory / "checkpoints").glob("*.pkl"))
                ),
            }
        )
        _write_json_create_only(output_directory / "failure.json", failure)
        raise


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], RandomForestClassifier]:
    with path.open("rb") as stream:
        value = pickle.load(stream)
    payload = _mapping(value, "checkpoint")
    metadata = _mapping(payload.get("metadata"), "checkpoint metadata")
    estimator = payload.get("estimator")
    _require(isinstance(estimator, RandomForestClassifier), "checkpoint estimator type changed")
    return metadata, estimator


def validate_run(run_directory: Path) -> dict[str, Any]:
    """Replay the completed matrix from saved models/features and seal validation."""

    run_directory = run_directory.resolve()
    _require(run_directory.is_dir(), "run directory does not exist")
    _require(
        not (run_directory / "failure.json").exists(), "failed run cannot be validated as complete"
    )
    _require(not (run_directory / "validation.json").exists(), "validation already exists")
    _require(not (run_directory / "completion_manifest.json").exists(), "completion already exists")
    manifest = _read_json(run_directory / "artifact_manifest.json")
    _verify_sealed(manifest, "artifact manifest")
    artifacts = cast(list[dict[str, Any]], manifest["artifacts"])
    declared_paths = [str(item["path"]) for item in artifacts]
    _require(
        len(declared_paths) == len(set(declared_paths)),
        "artifact manifest contains duplicate paths",
    )
    actual_paths = {
        path.relative_to(run_directory).as_posix()
        for path in run_directory.rglob("*")
        if path.is_file()
        and path.name
        not in {"artifact_manifest.json", "validation.json", "completion_manifest.json"}
    }
    _require(
        set(declared_paths) == actual_paths,
        "artifact manifest does not exactly cover pre-validation run files",
    )
    for artifact in artifacts:
        path = (run_directory / str(artifact["path"])).resolve()
        _require(_is_relative_to(path, run_directory), "manifest path escapes run")
        _require(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == int(artifact["size_bytes"])
            and sha256_file(path) == artifact["sha256"],
            f"artifact hash/size mismatch: {artifact['path']}",
        )
    _require(
        set(REQUIRED_RUN_FILES).issubset({str(item["path"]) for item in artifacts}),
        "required run artifact missing",
    )
    _require(
        sum(path.startswith("checkpoints/") for path in declared_paths) == 20,
        "artifact manifest does not contain exactly 20 checkpoints",
    )
    config = _read_yaml(run_directory / "config_snapshot.yaml")
    validate_config(config)
    for name in (
        "analysis.json",
        "data_audit.json",
        "feature_cache_metadata.json",
        "fold_reports.json",
        "participant_metrics.json",
        "preflight.json",
        "protocol_snapshot.json",
        "result.json",
        "runtime.json",
        "source_receipt.json",
    ):
        _verify_sealed(_read_json(run_directory / name), name)
    feature_metadata = _read_json(run_directory / "feature_cache_metadata.json")
    _require(
        sha256_file(run_directory / "feature_cache.npz") == feature_metadata["sha256"],
        "feature cache hash mismatch",
    )
    with np.load(run_directory / "feature_cache.npz", allow_pickle=False) as cache:
        six_values = np.asarray(cache["six_values"], dtype=np.float64)
        nine_values = np.asarray(cache["nine_values"], dtype=np.float64)
        six_names = tuple(cache["six_names"].tolist())
        nine_names = tuple(cache["nine_names"].tolist())
        observable_ids = np.asarray(cache["observable_participant_ids"], dtype=np.str_)
        observable_windows = np.asarray(cache["observable_window_ids"], dtype=np.str_)
        observable_fold = np.asarray(cache["observable_fold_index"], dtype=np.int64)
        scoring_indices = np.asarray(cache["scoring_indices"], dtype=np.int64)
        eligibility = np.asarray(cache["scoring_eligibility"], dtype=np.bool_)
        labels = np.asarray(cache["scored_labels"], dtype=np.int64)
        participants = np.asarray(cache["scored_participant_ids"], dtype=np.str_)
        windows = np.asarray(cache["scored_window_ids"], dtype=np.str_)
        scored_fold = np.asarray(cache["scored_fold_index"], dtype=np.int64)
    _require(
        _array_sha256(six_values) == feature_metadata["six_values_sha256"], "six features changed"
    )
    _require(
        _array_sha256(nine_values) == feature_metadata["nine_values_sha256"],
        "nine features changed",
    )
    _require(tuple(feature_metadata["six_names"]) == six_names, "six feature names changed")
    _require(tuple(feature_metadata["nine_names"]) == nine_names, "nine feature names changed")
    _require(
        _array_sha256(observable_windows) == feature_metadata["observable_window_ids_sha256"],
        "observable window IDs changed",
    )
    _require(
        _array_sha256(scoring_indices) == feature_metadata["scoring_indices_sha256"],
        "scoring indices changed",
    )
    _require(
        _array_sha256(eligibility) == feature_metadata["scoring_eligibility_sha256"],
        "scoring eligibility changed",
    )
    _require(
        _array_sha256(labels) == feature_metadata["scored_labels_sha256"],
        "scored labels changed",
    )
    _require(
        _array_sha256(participants) == feature_metadata["scored_participant_ids_sha256"],
        "scored participant IDs changed",
    )
    _require(
        six_values.shape[0]
        == nine_values.shape[0]
        == observable_ids.size
        == observable_windows.size
        == observable_fold.size
        == eligibility.size,
        "observable feature-cache arrays are not aligned",
    )
    _require(
        scoring_indices.ndim == 1
        and labels.shape == participants.shape == windows.shape == scored_fold.shape
        and scoring_indices.size == labels.size
        and np.array_equal(np.flatnonzero(eligibility), scoring_indices),
        "scored feature-cache arrays are not aligned",
    )
    _require(
        np.array_equal(observable_windows[scoring_indices], windows),
        "scoring window alignment failed",
    )
    _require(
        np.array_equal(observable_ids[scoring_indices], participants),
        "scoring participant alignment failed",
    )
    _require(
        np.array_equal(observable_fold[scoring_indices], scored_fold),
        "scored fold alignment failed",
    )

    preflight = _read_json(run_directory / "preflight.json")
    source_receipt = _read_json(run_directory / "source_receipt.json")
    data_audit = _read_json(run_directory / "data_audit.json")
    source_receipts = cast(list[dict[str, Any]], source_receipt["source_receipts"])
    _require(len(source_receipts) == 1, "source receipt cardinality changed")
    plan_audit = _mapping(data_audit["participant_partition_plan"], "participant partition plan")
    source_manifest = _mapping(preflight["source_manifest"], "source manifest")
    source_files = cast(list[dict[str, Any]], source_manifest["files"])
    feature_code = [
        item
        for item in source_files
        if item["path"] == "src/inclusive_shift_har/preprocessing/features.py"
    ]
    _require(len(feature_code) == 1, "feature code is absent from source manifest")
    expected_cache_key = _feature_cache_key(
        source_sha256=str(source_receipts[0]["computed_sha256"]),
        participant_plan_sha256=str(plan_audit["plan_sha256"]),
        observable_window_ids_sha256=_array_sha256(observable_windows),
        config_sha256=sha256_file(run_directory / "config_snapshot.yaml"),
        feature_code_sha256=str(feature_code[0]["sha256"]),
    )
    _require(
        _mapping(feature_metadata["cache_key"], "feature cache key") == expected_cache_key,
        "feature cache key cannot be reconstructed",
    )
    _require(
        _mapping(preflight["configuration"], "preflight configuration")["sha256"]
        == sha256_file(run_directory / "config_snapshot.yaml"),
        "preflight configuration hash changed",
    )
    _require(
        _mapping(preflight["protocol"], "preflight protocol")["sha256"]
        == _read_json(run_directory / "protocol_snapshot.json")["source_sha256"],
        "preflight protocol hash changed",
    )
    effective = _mapping(
        preflight["effective_estimator_parameter_sets"],
        "effective estimator parameter sets",
    )
    _require(
        set(effective)
        == {f"{cell_id}--fold-{fold_index}" for cell_id in CELL_ORDER for fold_index in range(5)},
        "preflight does not contain all 20 effective parameter sets",
    )

    with np.load(run_directory / "predictions.npz", allow_pickle=False) as predictions:
        _require(
            predictions["cell_ids"].tolist() == list(CELL_ORDER), "prediction cell order changed"
        )
        stored_observable = np.asarray(predictions["observable_probabilities"], dtype=np.float64)
        stored_scored = np.asarray(predictions["scored_probabilities"], dtype=np.float64)
        _require(
            np.array_equal(predictions["observable_window_ids"], observable_windows),
            "prediction observable IDs changed",
        )
        _require(
            np.array_equal(predictions["scoring_indices"], scoring_indices),
            "prediction scoring indices changed",
        )
        _require(np.array_equal(predictions["scored_labels"], labels), "prediction labels changed")
        _require(
            np.array_equal(predictions["scored_participant_ids"], participants),
            "prediction participants changed",
        )
        _require(
            np.array_equal(predictions["observable_participant_ids"], observable_ids),
            "prediction observable participant IDs changed",
        )
        _require(
            np.array_equal(predictions["observable_fold_index"], observable_fold),
            "prediction observable folds changed",
        )
        _require(
            np.array_equal(predictions["scored_window_ids"], windows),
            "prediction scored window IDs changed",
        )
        _require(
            np.array_equal(predictions["scored_fold_index"], scored_fold),
            "prediction scored folds changed",
        )
    replay = np.full_like(stored_observable, np.nan)
    fold_payload = _read_json(run_directory / "fold_reports.json")
    rows = cast(list[dict[str, Any]], fold_payload["rows"])
    _require(len(rows) == 20 and fold_payload["fit_count"] == 20, "fold report count changed")
    row_lookup = {(str(row["cell_id"]), int(row["outer_fold"])): row for row in rows}
    _require(len(row_lookup) == 20, "duplicate fold reports")
    observable_labels = np.zeros(observable_ids.size, dtype=np.int64)
    observable_labels[scoring_indices] = labels
    for cell_index, cell_id in enumerate(CELL_ORDER):
        feature_values = six_values if cell_id in {"f0", "f1"} else nine_values
        feature_names = six_names if cell_id in {"f0", "f1"} else nine_names
        for fold_index in range(5):
            row = row_lookup[(cell_id, fold_index)]
            checkpoint = _mapping(row["checkpoint"], "checkpoint reference")
            checkpoint_path = (run_directory / str(checkpoint["path"])).resolve()
            _require(
                _is_relative_to(checkpoint_path, run_directory),
                "checkpoint path escapes run directory",
            )
            _require(
                sha256_file(checkpoint_path) == checkpoint["sha256"], "checkpoint hash changed"
            )
            metadata, estimator = _load_checkpoint(checkpoint_path)
            _require(
                metadata["cell_id"] == cell_id and metadata["outer_fold"] == fold_index,
                "checkpoint metadata mismatch",
            )
            _require(
                tuple(metadata["feature_names"]) == feature_names,
                "checkpoint feature schema mismatch",
            )
            _require(
                np.array_equal(estimator.classes_, np.arange(3)), "checkpoint class order mismatch"
            )
            evaluation = observable_fold == fold_index
            training = (~evaluation) & eligibility
            _require(
                set(np.unique(observable_ids[evaluation]).tolist()).isdisjoint(
                    np.unique(observable_ids[training]).tolist()
                ),
                "checkpoint fold has participant overlap",
            )
            _require(
                metadata["training_participants"]
                == sorted(np.unique(observable_ids[training]).tolist())
                and metadata["evaluation_participants"]
                == sorted(np.unique(observable_ids[evaluation]).tolist())
                and metadata["training_row_count"] == int(training.sum())
                and metadata["evaluation_candidate_count"] == int(evaluation.sum()),
                "checkpoint fold provenance mismatch",
            )
            expected_parameters = _effective_estimator_parameters(
                class_weight=None if cell_id in {"f1", "f3"} else "balanced_subsample",
                random_state=11 + fold_index,
                n_jobs=4,
            )
            _require(
                estimator.get_params(deep=False) == expected_parameters
                and metadata["effective_parameters"] == expected_parameters,
                "checkpoint estimator parameters changed",
            )
            _require(
                int(estimator.n_features_in_) == feature_values.shape[1],
                "checkpoint feature dimension changed",
            )
            if cell_id in {"f1", "f3"}:
                expected_weights = participant_first_weights(
                    observable_labels[training], observable_ids[training]
                )
                weight_summary = _mapping(
                    metadata["sample_weight_summary"], "sample-weight summary"
                )
                _require(
                    weight_summary["sha256"] == _array_sha256(expected_weights),
                    "participant-first training weights changed",
                )
            else:
                _require(
                    metadata["sample_weight_summary"] is None,
                    "ordinary cell unexpectedly has explicit sample weights",
                )
            with threadpool_limits(limits=1):
                values = _predict_proba_deterministically(estimator, feature_values[evaluation])
            replay[cell_index, evaluation] = values
    _require(
        np.array_equal(replay, stored_observable),
        "checkpoint replay differs from stored observable predictions",
    )
    replay_scored = replay[:, scoring_indices]
    _require(np.array_equal(replay_scored, stored_scored), "scored prediction replay differs")
    roster = cast(list[str], _mapping(config["source"], "source")["participant_roster"])
    reports = {
        cell_id: method_report(
            labels=labels,
            probabilities=replay_scored[index],
            participant_ids=participants,
            roster=roster,
        )
        for index, cell_id in enumerate(CELL_ORDER)
    }
    reconstructed = analyse(reports, config=config)
    recorded = _read_json(run_directory / "analysis.json")
    recorded_without_hash = dict(recorded)
    recorded_without_hash.pop("record_sha256")
    _require(
        canonical_json_sha256({"record_kind": "fog_rf_factorial_analysis", **reconstructed})
        == canonical_json_sha256(recorded_without_hash),
        "analysis/gate replay differs",
    )
    _require(
        (run_directory / "OUTCOME_SUMMARY.md").read_text(encoding="utf-8")
        == _outcome_summary(recorded),
        "outcome summary differs",
    )
    result = _read_json(run_directory / "result.json")
    runtime = _read_json(run_directory / "runtime.json")
    _require(
        result["status"] == "complete_awaiting_independent_replay"
        and result["fit_count"] == 20
        and result["code_commit"] == preflight["code_commit"]
        and result["source_sha256"] == source_receipts[0]["computed_sha256"],
        "result completion/provenance fields changed",
    )
    _require(
        runtime["model_fit_count"] == 20
        and runtime["compute_cap_seconds"] == 7200
        and runtime["cap_exceeded"] is False
        and float(runtime["total_run_seconds"]) <= 7200.0,
        "runtime record violates the frozen resource contract",
    )
    validation = _sealed(
        {
            "record_kind": "fog_rf_factorial_validation",
            "status": "validated",
            "validated_at_utc": _now(),
            "artifact_hashes_verified": len(artifacts),
            "checkpoint_predictions_replayed": 20,
            "observable_probability_arrays_byte_exact": 4,
            "scored_probability_arrays_byte_exact": 4,
            "participant_reports_recomputed": 4,
            "factorial_contrasts_recomputed": 5 if reconstructed["factorial_contrasts"] else 0,
            "promotion_gate_recomputed": True,
            "full_roster_retained": True,
            "source_downloaded_during_validation": False,
            "model_fits_during_validation": 0,
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
            "record_kind": "fog_rf_factorial_completion_manifest",
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
            "promotion_gate_status": _mapping(recorded["promotion_gate"], "promotion gate")[
                "status"
            ],
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
            output_directory=arguments.output_directory,
            code_commit=str(arguments.code_commit),
            timeout_seconds=int(arguments.timeout_seconds),
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "promotion_gate": result["promotion_gate"],
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
