"""Decision-separated, evidence-marginalized extensions for HERA-CTGR v2.

This module contains deterministic feature, calibration, routing, and
one-query building blocks.  It does not load datasets.  Evaluation labels are
accepted only by explicitly named training-target or personalization helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]


@dataclass(frozen=True, slots=True)
class DualFrameFeatures:
    """Posture representation retaining device and gravity-aligned frames."""

    features: FloatArray
    compact_context: FloatArray
    feature_names: tuple[str, ...]
    compact_context_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CandidateMarginalization:
    """Consensus probabilities and label-free candidate uncertainty."""

    probabilities: FloatArray
    candidate_weights: FloatArray
    disagreement: FloatArray


@dataclass(frozen=True, slots=True)
class RidgeLogisticModel:
    """Small participant-balanced probabilistic routing head."""

    center: FloatArray
    scale: FloatArray
    coefficients: FloatArray
    intercept: float

    def predict_probability(self, features: NDArray[np.floating[Any]]) -> FloatArray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.center.size:
            raise ValueError("binary-head features do not match the fitted model")
        if not np.isfinite(values).all():
            raise ValueError("binary-head features must be finite")
        logits = (values - self.center) / self.scale @ self.coefficients + self.intercept
        return np.asarray(1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0))), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class BinaryJackknifePrediction:
    """Participant-jackknife probability interval for one routing event."""

    mean: FloatArray
    standard_deviation: FloatArray
    lower_bound: FloatArray
    upper_bound: FloatArray
    model_count: int


@dataclass(frozen=True, slots=True)
class RescueHarmTargets:
    """Training-only counterfactual outcome labels."""

    disagreement: BoolArray
    rescue: FloatArray
    harm: FloatArray


@dataclass(frozen=True, slots=True)
class SentinelRoutingResult:
    """Decision-separated probabilities and complete routing diagnostics."""

    probabilities: FloatArray
    routed: BoolArray
    eligible: BoolArray
    rescue_lower_bound: FloatArray
    harm_upper_bound: FloatArray
    net_benefit_lower_bound: FloatArray
    support_distance: FloatArray


@dataclass(frozen=True, slots=True)
class OneQueryStateResult:
    """One-query OFF/NORMAL/INVERTED posture-gauge result."""

    probabilities: FloatArray
    evaluation_mask: BoolArray
    queried_index: int | None
    selected_state: str
    query_score: FloatArray


_DUAL_SERIES_NAMES = (
    "gravity_x",
    "gravity_y",
    "gravity_z",
    "gravity_unit_x",
    "gravity_unit_y",
    "gravity_unit_z",
    "user_acceleration_x",
    "user_acceleration_y",
    "user_acceleration_z",
    "gyroscope_x",
    "gyroscope_y",
    "gyroscope_z",
    "user_acceleration_tangent_1",
    "user_acceleration_tangent_2",
    "user_acceleration_parallel_gravity",
    "gyroscope_tangent_1",
    "gyroscope_tangent_2",
    "gyroscope_parallel_gravity",
    "user_acceleration_horizontal_norm",
    "gyroscope_horizontal_norm",
    "total_acceleration_parallel_gravity",
    "total_acceleration_horizontal_norm",
    "gravity_norm",
    "gravity_direction_speed",
    "gravity_gyro_residual",
    "normalized_gravity_gyro_residual",
)

_SUMMARY_NAMES = (
    "mean",
    "std",
    "quantile_10",
    "median",
    "quantile_90",
    "iqr",
    "rms",
    "mean_absolute_difference",
    "linear_slope_per_second",
    "first_half_minus_second_half",
)

_COMPACT_CONTEXT_NAMES = (
    "gravity_unit_x_median",
    "gravity_unit_y_median",
    "gravity_unit_z_median",
    "gravity_abs_sorted_low_median",
    "gravity_abs_sorted_mid_median",
    "gravity_abs_sorted_high_median",
    "user_acceleration_vertical_median",
    "user_acceleration_vertical_iqr",
    "user_acceleration_horizontal_rms",
    "gyroscope_vertical_median",
    "gyroscope_horizontal_rms",
    "gravity_direction_speed_p90",
    "normalized_kinematic_residual_p90",
    "gravity_norm_mad",
    "gravity_direction_dispersion",
)


def _probabilities(values: NDArray[np.floating[Any]], *, name: str) -> FloatArray:
    probability = np.asarray(values, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[0] == 0 or probability.shape[1] != 3:
        raise ValueError(f"{name} must be a non-empty [sample,3] matrix")
    if not np.isfinite(probability).all() or np.any(probability < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-7, rtol=1e-7):
        raise ValueError(f"{name} rows must sum to one")
    return probability


def _validate_dual_inputs(
    primary_signals: NDArray[np.floating[Any]],
    gravity: NDArray[np.floating[Any]],
    sampling_rate_hz: float,
) -> tuple[FloatArray, FloatArray]:
    primary = np.asarray(primary_signals, dtype=np.float64)
    gravity_values = np.asarray(gravity, dtype=np.float64)
    if primary.ndim != 3 or primary.shape[0] == 0 or primary.shape[1] < 32 or primary.shape[2] != 6:
        raise ValueError("dual-frame features require [window,time>=32,6] primary signals")
    if gravity_values.shape != (*primary.shape[:2], 3):
        raise ValueError("gravity must align with primary signals as [window,time,3]")
    if not np.isfinite(primary).all() or not np.isfinite(gravity_values).all():
        raise ValueError("dual-frame feature inputs must be finite")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0.0:
        raise ValueError("sampling rate must be finite and positive")
    return primary, gravity_values


def _dual_frame_series(
    primary_signals: NDArray[np.floating[Any]],
    gravity: NDArray[np.floating[Any]],
    *,
    sampling_rate_hz: float,
) -> FloatArray:
    primary, gravity_values = _validate_dual_inputs(primary_signals, gravity, sampling_rate_hz)
    gravity_norm = np.linalg.norm(gravity_values, axis=2)
    unit = gravity_values / np.maximum(gravity_norm[:, :, None], 1e-10)

    x_axis = np.zeros_like(unit)
    x_axis[:, :, 0] = 1.0
    y_axis = np.zeros_like(unit)
    y_axis[:, :, 1] = 1.0
    tangent_1 = x_axis - np.sum(x_axis * unit, axis=2)[:, :, None] * unit
    tangent_norm = np.linalg.norm(tangent_1, axis=2)
    fallback = tangent_norm < 0.10
    tangent_y = y_axis - np.sum(y_axis * unit, axis=2)[:, :, None] * unit
    tangent_1[fallback] = tangent_y[fallback]
    tangent_norm = np.linalg.norm(tangent_1, axis=2)
    tangent_1 /= np.maximum(tangent_norm[:, :, None], 1e-10)
    tangent_2 = np.cross(unit, tangent_1)

    acceleration = primary[:, :, :3]
    gyroscope = primary[:, :, 3:]
    acc_t1 = np.sum(acceleration * tangent_1, axis=2)
    acc_t2 = np.sum(acceleration * tangent_2, axis=2)
    acc_vertical = np.sum(acceleration * unit, axis=2)
    gyro_t1 = np.sum(gyroscope * tangent_1, axis=2)
    gyro_t2 = np.sum(gyroscope * tangent_2, axis=2)
    gyro_vertical = np.sum(gyroscope * unit, axis=2)
    acc_horizontal = np.sqrt(acc_t1**2 + acc_t2**2)
    gyro_horizontal = np.sqrt(gyro_t1**2 + gyro_t2**2)
    total = acceleration + gravity_values
    total_vertical = np.sum(total * unit, axis=2)
    total_horizontal = np.linalg.norm(total - total_vertical[:, :, None] * unit, axis=2)

    derivative = np.zeros_like(unit)
    derivative[:, 1:] = (unit[:, 1:] - unit[:, :-1]) * sampling_rate_hz
    predicted = -np.cross(gyroscope, unit)
    derivative_speed = np.linalg.norm(derivative, axis=2)
    predicted_speed = np.linalg.norm(predicted, axis=2)
    residual = np.linalg.norm(derivative - predicted, axis=2)
    normalized_residual = residual / np.maximum(0.25 + derivative_speed + predicted_speed, 1e-10)

    series = np.concatenate(
        (
            gravity_values,
            unit,
            acceleration,
            gyroscope,
            acc_t1[:, :, None],
            acc_t2[:, :, None],
            acc_vertical[:, :, None],
            gyro_t1[:, :, None],
            gyro_t2[:, :, None],
            gyro_vertical[:, :, None],
            acc_horizontal[:, :, None],
            gyro_horizontal[:, :, None],
            total_vertical[:, :, None],
            total_horizontal[:, :, None],
            gravity_norm[:, :, None],
            derivative_speed[:, :, None],
            residual[:, :, None],
            normalized_residual[:, :, None],
        ),
        axis=2,
    )
    if series.shape[2] != len(_DUAL_SERIES_NAMES) or not np.isfinite(series).all():
        raise AssertionError("dual-frame time-series contract changed")
    return np.asarray(series, dtype=np.float64)


@lru_cache(maxsize=1)
def dual_frame_feature_names() -> tuple[str, ...]:
    """Return the immutable flattened feature order."""

    return tuple(
        f"{series}__{summary}" for series in _DUAL_SERIES_NAMES for summary in _SUMMARY_NAMES
    )


def extract_dual_frame_posture_features(
    primary_signals: NDArray[np.floating[Any]],
    gravity: NDArray[np.floating[Any]],
    *,
    sampling_rate_hz: float,
) -> DualFrameFeatures:
    """Retain device-frame posture while adding gravity-aligned dynamics."""

    series = _dual_frame_series(primary_signals, gravity, sampling_rate_hz=sampling_rate_hz)
    windows, time_steps, channels = series.shape
    quantiles = np.quantile(series, (0.10, 0.25, 0.50, 0.75, 0.90), axis=1)
    time = np.arange(time_steps, dtype=np.float64) / sampling_rate_hz
    centered_time = time - time.mean()
    centered_series = series - series.mean(axis=1, keepdims=True)
    slopes = np.einsum("t,ntc->nc", centered_time, centered_series) / float(
        np.sum(centered_time**2)
    )
    midpoint = time_steps // 2
    summaries = (
        series.mean(axis=1),
        series.std(axis=1),
        quantiles[0],
        quantiles[2],
        quantiles[4],
        quantiles[3] - quantiles[1],
        np.sqrt(np.mean(series**2, axis=1)),
        np.mean(np.abs(np.diff(series, axis=1)), axis=1),
        slopes,
        series[:, :midpoint].mean(axis=1) - series[:, midpoint:].mean(axis=1),
    )
    features = np.stack(summaries, axis=2).reshape(windows, channels * len(_SUMMARY_NAMES))

    unit_median = quantiles[2, :, 3:6]
    sorted_absolute = np.sort(np.abs(unit_median), axis=1)
    gravity_median = quantiles[2, :, 22]
    gravity_mad = np.median(np.abs(series[:, :, 22] - gravity_median[:, None]), axis=1)
    compact = np.column_stack(
        (
            unit_median,
            sorted_absolute,
            quantiles[2, :, 14],
            quantiles[3, :, 14] - quantiles[1, :, 14],
            np.sqrt(np.mean(series[:, :, 18] ** 2, axis=1)),
            quantiles[2, :, 17],
            np.sqrt(np.mean(series[:, :, 19] ** 2, axis=1)),
            quantiles[4, :, 23],
            quantiles[4, :, 25],
            gravity_mad,
            np.linalg.norm(series[:, :, 3:6].std(axis=1), axis=1),
        )
    )
    if features.shape != (windows, len(dual_frame_feature_names())):
        raise AssertionError("dual-frame flattened feature contract changed")
    if compact.shape != (windows, len(_COMPACT_CONTEXT_NAMES)):
        raise AssertionError("dual-frame compact-context contract changed")
    if not np.isfinite(features).all() or not np.isfinite(compact).all():
        raise AssertionError("dual-frame features must be finite")
    return DualFrameFeatures(
        features=np.asarray(features, dtype=np.float64),
        compact_context=np.asarray(compact, dtype=np.float64),
        feature_names=dual_frame_feature_names(),
        compact_context_names=_COMPACT_CONTEXT_NAMES,
    )


def _jensen_shannon(left: FloatArray, right: FloatArray) -> FloatArray:
    midpoint = 0.5 * (left + right)
    return np.asarray(
        0.5
        * np.sum(
            left * (np.log(np.clip(left, 1e-12, 1.0)) - np.log(np.clip(midpoint, 1e-12, 1.0)))
            + right * (np.log(np.clip(right, 1e-12, 1.0)) - np.log(np.clip(midpoint, 1e-12, 1.0))),
            axis=-1,
        ),
        dtype=np.float64,
    )


def stability_weighted_candidate_marginalization(
    candidates: NDArray[np.floating[Any]],
    *,
    divergence_temperature: float,
    minimum_candidate_weight: float,
) -> CandidateMarginalization:
    """Downweight candidate outliers without outcome-dependent weights."""

    values = np.asarray(candidates, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[2] != 3:
        raise ValueError("candidate probabilities must have shape [candidate,sample,3]")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("candidate probabilities must be finite and non-negative")
    if not np.allclose(values.sum(axis=2), 1.0, atol=1e-7, rtol=1e-7):
        raise ValueError("every candidate probability row must sum to one")
    if not np.isfinite(divergence_temperature) or divergence_temperature <= 0.0:
        raise ValueError("divergence temperature must be finite and positive")
    if not 0.0 <= minimum_candidate_weight < 1.0 / values.shape[0]:
        raise ValueError("minimum candidate weight is incompatible with candidate count")

    consensus = values.mean(axis=0)
    divergence = np.stack(
        [_jensen_shannon(values[index], consensus) for index in range(values.shape[0])], axis=0
    )
    raw_weight = np.exp(-np.clip(divergence / divergence_temperature, 0.0, 40.0))
    raw_weight /= raw_weight.sum(axis=0, keepdims=True)
    remaining = 1.0 - values.shape[0] * minimum_candidate_weight
    weight = minimum_candidate_weight + remaining * raw_weight
    probability = np.sum(weight[:, :, None] * values, axis=0)
    probability /= probability.sum(axis=1, keepdims=True)
    disagreement = divergence.mean(axis=0)
    return CandidateMarginalization(
        probabilities=np.asarray(probability, dtype=np.float64),
        candidate_weights=np.asarray(weight, dtype=np.float64),
        disagreement=np.asarray(disagreement, dtype=np.float64),
    )


def temperature_scale_probabilities(
    probabilities: NDArray[np.floating[Any]], *, temperature: float
) -> FloatArray:
    """Apply rank-preserving scalar temperature calibration."""

    values = _probabilities(probabilities, name="probabilities")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    logits = np.log(np.clip(values, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    result /= result.sum(axis=1, keepdims=True)
    if not np.array_equal(result.argmax(axis=1), values.argmax(axis=1)):
        raise AssertionError("scalar temperature changed a class decision")
    return np.asarray(result, dtype=np.float64)


def apply_rank_locked_posture_offset(
    probabilities: NDArray[np.floating[Any]],
    *,
    posture_logit_offset: float,
) -> tuple[FloatArray, BoolArray]:
    """Permit sitting/standing corrections while forbidding mobility flips."""

    values = _probabilities(probabilities, name="probabilities")
    if not np.isfinite(posture_logit_offset):
        raise ValueError("posture offset must be finite")
    stationary = values[:, 1] + values[:, 2]
    log_odds = np.log(np.clip(values[:, 1], 1e-12, 1.0)) - np.log(np.clip(values[:, 2], 1e-12, 1.0))
    sitting = 1.0 / (1.0 + np.exp(-np.clip(log_odds + posture_logit_offset, -40.0, 40.0)))
    candidate = values.copy()
    candidate[:, 1] = stationary * sitting
    candidate[:, 2] = stationary * (1.0 - sitting)
    original_class = values.argmax(axis=1)
    candidate_class = candidate.argmax(axis=1)
    allowed = (original_class != 0) & (candidate_class != 0)
    result = values.copy()
    result[allowed] = candidate[allowed]
    changed = result.argmax(axis=1) != original_class
    if np.any((result.argmax(axis=1) == 0) != (original_class == 0)):
        raise AssertionError("rank-locked posture offset changed mobility membership")
    np.testing.assert_allclose(result[:, 0], values[:, 0], atol=0.0, rtol=0.0)
    return np.asarray(result, dtype=np.float64), np.asarray(changed, dtype=np.bool_)


def build_rescue_harm_features(
    base_probability: NDArray[np.floating[Any]],
    intervention_probability: NDArray[np.floating[Any]],
    *,
    candidate_disagreement: NDArray[np.floating[Any]],
    physics_reliability: NDArray[np.floating[Any]],
    dual_frame_context: NDArray[np.floating[Any]],
) -> tuple[FloatArray, tuple[str, ...]]:
    """Build label-free evidence for distinct rescue and harm heads."""

    base = _probabilities(base_probability, name="base probability")
    intervention = _probabilities(intervention_probability, name="intervention probability")
    disagreement = np.asarray(candidate_disagreement, dtype=np.float64)
    reliability = np.asarray(physics_reliability, dtype=np.float64)
    context = np.asarray(dual_frame_context, dtype=np.float64)
    if intervention.shape != base.shape:
        raise ValueError("base and intervention probabilities must align")
    if disagreement.shape != (base.shape[0],) or reliability.shape != disagreement.shape:
        raise ValueError("candidate disagreement and reliability must align")
    if context.ndim != 2 or context.shape[0] != base.shape[0]:
        raise ValueError("dual-frame context must align with probabilities")
    if (
        not np.isfinite(disagreement).all()
        or not np.isfinite(reliability).all()
        or not np.isfinite(context).all()
        or np.any((reliability < 0.0) | (reliability > 1.0))
    ):
        raise ValueError("rescue-harm evidence must be finite and reliability bounded")

    def margin(probability: FloatArray) -> FloatArray:
        ordered = np.sort(probability, axis=1)
        return ordered[:, -1] - ordered[:, -2]

    def entropy(probability: FloatArray) -> FloatArray:
        clipped = np.clip(probability, 1e-12, 1.0)
        return np.asarray(-np.sum(clipped * np.log(clipped), axis=1), dtype=np.float64)

    base_odds = np.log(np.clip(base[:, 1], 1e-12, 1.0)) - np.log(np.clip(base[:, 2], 1e-12, 1.0))
    intervention_odds = np.log(np.clip(intervention[:, 1], 1e-12, 1.0)) - np.log(
        np.clip(intervention[:, 2], 1e-12, 1.0)
    )
    js = _jensen_shannon(base, intervention)
    derived = np.column_stack(
        (
            base.max(axis=1),
            margin(base),
            entropy(base),
            base[:, 1] + base[:, 2],
            base_odds,
            intervention.max(axis=1),
            margin(intervention),
            entropy(intervention),
            intervention[:, 1] + intervention[:, 2],
            intervention_odds,
            intervention_odds - base_odds,
            np.sum(np.abs(intervention - base), axis=1),
            js,
            disagreement,
            reliability,
        )
    )
    names = (
        "base_max_probability",
        "base_margin",
        "base_entropy",
        "base_stationary_mass",
        "base_posture_log_odds",
        "intervention_max_probability",
        "intervention_margin",
        "intervention_entropy",
        "intervention_stationary_mass",
        "intervention_posture_log_odds",
        "posture_log_odds_residual",
        "base_intervention_l1",
        "base_intervention_jensen_shannon",
        "candidate_ensemble_disagreement",
        "physics_reliability",
        *(f"dual_context_{index:02d}" for index in range(context.shape[1])),
    )
    features = np.concatenate((derived, context), axis=1)
    if features.shape != (base.shape[0], len(names)) or not np.isfinite(features).all():
        raise AssertionError("rescue-harm feature contract changed")
    return np.asarray(features, dtype=np.float64), names


def counterfactual_rescue_harm_targets(
    base_probability: NDArray[np.floating[Any]],
    intervention_probability: NDArray[np.floating[Any]],
    labels: NDArray[np.integer[Any]],
) -> RescueHarmTargets:
    """Label training disagreements as rescue, harm, or neither."""

    base = _probabilities(base_probability, name="base probability")
    intervention = _probabilities(intervention_probability, name="intervention probability")
    truth = np.asarray(labels, dtype=np.int64)
    if intervention.shape != base.shape or truth.shape != (base.shape[0],):
        raise ValueError("counterfactual routing arrays must align")
    if truth.size == 0 or truth.min() < 0 or truth.max() > 2:
        raise ValueError("counterfactual labels must lie in [0,2]")
    base_label = base.argmax(axis=1)
    intervention_label = intervention.argmax(axis=1)
    disagreement = base_label != intervention_label
    rescue = disagreement & (base_label != truth) & (intervention_label == truth)
    harm = disagreement & (base_label == truth) & (intervention_label != truth)
    return RescueHarmTargets(
        disagreement=np.asarray(disagreement, dtype=np.bool_),
        rescue=np.asarray(rescue, dtype=np.float64),
        harm=np.asarray(harm, dtype=np.float64),
    )


def _participant_weights(participants: StringArray, target: FloatArray) -> FloatArray:
    weights = np.empty(participants.size, dtype=np.float64)
    for participant in np.unique(participants):
        selected = participants == participant
        weights[selected] = 1.0 / float(selected.sum())
    for outcome in (0.0, 1.0):
        selected = target == outcome
        if selected.any():
            weights[selected] *= 0.5 / float(weights[selected].sum())
    weights *= participants.size / weights.sum()
    return weights


def fit_participant_balanced_logistic(
    features: NDArray[np.floating[Any]],
    targets: NDArray[np.floating[Any]],
    participant_ids: NDArray[np.str_],
    *,
    ridge_penalty: float,
    maximum_iterations: int = 80,
) -> RidgeLogisticModel:
    """Fit a deterministic participant- and class-balanced ridge logistic head."""

    values = np.asarray(features, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    participants = np.asarray(participant_ids, dtype=np.str_)
    if (
        values.ndim != 2
        or values.shape[0] < 2
        or target.shape != (values.shape[0],)
        or participants.shape != target.shape
    ):
        raise ValueError("binary-head training arrays must align and be non-empty")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(target).all()
        or np.any((target < 0) | (target > 1))
    ):
        raise ValueError("binary-head values and targets must be finite and bounded")
    if np.unique(participants).size < 2:
        raise ValueError("binary head requires at least two participants")
    if not np.isfinite(ridge_penalty) or ridge_penalty <= 0.0:
        raise ValueError("ridge penalty must be finite and positive")
    if maximum_iterations < 1:
        raise ValueError("maximum iterations must be positive")

    weights = _participant_weights(participants, target)
    normalized = weights / weights.sum()
    center = np.sum(values * normalized[:, None], axis=0)
    variance = np.sum((values - center) ** 2 * normalized[:, None], axis=0)
    scale = np.sqrt(np.maximum(variance, 1e-12))
    standardized = (values - center) / scale
    design = np.column_stack((np.ones(values.shape[0]), standardized))
    prevalence = float((target.sum() + 1.0) / (target.size + 2.0))
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    coefficients[0] = np.log(prevalence) - np.log(1.0 - prevalence)
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge_penalty
    penalty[0, 0] = 0.0
    for _ in range(maximum_iterations):
        logits = design @ coefficients
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        gradient = design.T @ (weights * (probability - target)) + penalty @ coefficients
        curvature = np.maximum(probability * (1.0 - probability), 1e-8)
        hessian = design.T @ (design * (weights * curvature)[:, None]) + penalty
        step = np.linalg.pinv(hessian) @ gradient
        coefficients -= step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return RidgeLogisticModel(
        center=np.asarray(center, dtype=np.float64),
        scale=np.asarray(scale, dtype=np.float64),
        coefficients=np.asarray(coefficients[1:], dtype=np.float64),
        intercept=float(coefficients[0]),
    )


def participant_jackknife_binary_prediction(
    training_features: NDArray[np.floating[Any]],
    training_targets: NDArray[np.floating[Any]],
    training_participant_ids: NDArray[np.str_],
    evaluation_features: NDArray[np.floating[Any]],
    *,
    ridge_penalty: float,
    interval_z: float,
) -> BinaryJackknifePrediction:
    """Estimate a binary routing event with whole-participant jackknifing."""

    train = np.asarray(training_features, dtype=np.float64)
    target = np.asarray(training_targets, dtype=np.float64)
    participants = np.asarray(training_participant_ids, dtype=np.str_)
    evaluate = np.asarray(evaluation_features, dtype=np.float64)
    unique = np.unique(participants)
    if unique.size < 4:
        raise ValueError("binary jackknife requires at least four training participants")
    if target.shape != (train.shape[0],) or participants.shape != target.shape:
        raise ValueError("binary jackknife training arrays do not align")
    if evaluate.ndim != 2 or evaluate.shape[1] != train.shape[1]:
        raise ValueError("binary jackknife evaluation features do not align")
    if not np.isfinite(interval_z) or interval_z < 0.0:
        raise ValueError("binary jackknife interval multiplier must be non-negative")
    predictions: list[FloatArray] = []
    for excluded in unique:
        retained = participants != excluded
        model = fit_participant_balanced_logistic(
            train[retained],
            target[retained],
            participants[retained],
            ridge_penalty=ridge_penalty,
        )
        predictions.append(model.predict_probability(evaluate))
    stack = np.stack(predictions, axis=0)
    mean = stack.mean(axis=0)
    deviation = stack.std(axis=0, ddof=1)
    return BinaryJackknifePrediction(
        mean=np.asarray(mean, dtype=np.float64),
        standard_deviation=np.asarray(deviation, dtype=np.float64),
        lower_bound=np.asarray(np.clip(mean - interval_z * deviation, 0.0, 1.0), dtype=np.float64),
        upper_bound=np.asarray(np.clip(mean + interval_z * deviation, 0.0, 1.0), dtype=np.float64),
        model_count=len(predictions),
    )


def apply_rescue_harm_sentinel(
    base_probability: NDArray[np.floating[Any]],
    intervention_probability: NDArray[np.floating[Any]],
    rescue_prediction: BinaryJackknifePrediction,
    harm_prediction: BinaryJackknifePrediction,
    *,
    physics_reliability: NDArray[np.floating[Any]],
    physics_trusted: NDArray[np.bool_],
    support_distance: NDArray[np.floating[Any]],
    minimum_rescue_lower_bound: float,
    maximum_harm_upper_bound: float,
    harm_penalty: float,
    minimum_net_benefit: float,
    minimum_physics_reliability: float,
    maximum_support_distance: float,
) -> SentinelRoutingResult:
    """Route only high-evidence sitting/standing corrections with exact fallback."""

    base = _probabilities(base_probability, name="base probability")
    intervention = _probabilities(intervention_probability, name="intervention probability")
    reliability = np.asarray(physics_reliability, dtype=np.float64)
    trusted = np.asarray(physics_trusted, dtype=np.bool_)
    support = np.asarray(support_distance, dtype=np.float64)
    vectors = (
        rescue_prediction.lower_bound,
        harm_prediction.upper_bound,
        reliability,
        support,
    )
    if intervention.shape != base.shape or any(item.shape != (base.shape[0],) for item in vectors):
        raise ValueError("sentinel arrays must align")
    if trusted.shape != (base.shape[0],) or not all(np.isfinite(item).all() for item in vectors):
        raise ValueError("sentinel trust and score arrays must be finite and aligned")
    settings = (
        minimum_rescue_lower_bound,
        maximum_harm_upper_bound,
        harm_penalty,
        minimum_net_benefit,
        minimum_physics_reliability,
        maximum_support_distance,
    )
    if not all(np.isfinite(item) for item in settings) or harm_penalty < 0.0:
        raise ValueError("sentinel thresholds must be finite and harm penalty non-negative")
    if not 0.0 <= minimum_rescue_lower_bound <= 1.0 or not 0.0 <= maximum_harm_upper_bound <= 1.0:
        raise ValueError("sentinel probability thresholds must lie in [0,1]")
    if not 0.0 <= minimum_physics_reliability <= 1.0 or maximum_support_distance <= 0.0:
        raise ValueError("sentinel reliability/support thresholds are invalid")

    base_class = base.argmax(axis=1)
    intervention_class = intervention.argmax(axis=1)
    semantic = (base_class != 0) & (intervention_class != 0) & (base_class != intervention_class)
    eligible = (
        semantic
        & trusted
        & (reliability >= minimum_physics_reliability)
        & (support <= maximum_support_distance)
    )
    net = rescue_prediction.lower_bound - harm_penalty * harm_prediction.upper_bound
    routed = (
        eligible
        & (rescue_prediction.lower_bound >= minimum_rescue_lower_bound)
        & (harm_prediction.upper_bound <= maximum_harm_upper_bound)
        & (net >= minimum_net_benefit)
    )
    result = base.copy()
    result[routed] = intervention[routed]
    if np.any((result.argmax(axis=1) == 0) != (base_class == 0)):
        raise AssertionError("rescue-harm sentinel changed mobility membership")
    return SentinelRoutingResult(
        probabilities=np.asarray(result, dtype=np.float64),
        routed=np.asarray(routed, dtype=np.bool_),
        eligible=np.asarray(eligible, dtype=np.bool_),
        rescue_lower_bound=np.asarray(rescue_prediction.lower_bound, dtype=np.float64),
        harm_upper_bound=np.asarray(harm_prediction.upper_bound, dtype=np.float64),
        net_benefit_lower_bound=np.asarray(net, dtype=np.float64),
        support_distance=np.asarray(support, dtype=np.float64),
    )


def _swap_posture(probability: FloatArray) -> FloatArray:
    swapped = probability.copy()
    swapped[:, [1, 2]] = swapped[:, [2, 1]]
    return swapped


def apply_one_query_state_selection(
    base_probability: NDArray[np.floating[Any]],
    intervention_probability: NDArray[np.floating[Any]],
    physics_reliability: NDArray[np.floating[Any]],
    window_ids: NDArray[np.str_],
    labels: NDArray[np.integer[Any]],
    *,
    minimum_stationary_mass: float,
) -> OneQueryStateResult:
    """Select OFF/NORMAL/INVERTED from one label chosen without seeing labels."""

    base = _probabilities(base_probability, name="base probability")
    intervention = _probabilities(intervention_probability, name="intervention probability")
    reliability = np.asarray(physics_reliability, dtype=np.float64)
    windows = np.asarray(window_ids, dtype=np.str_)
    truth = np.asarray(labels, dtype=np.int64)
    if intervention.shape != base.shape or reliability.shape != (base.shape[0],):
        raise ValueError("one-query probability and reliability arrays must align")
    if windows.shape != reliability.shape or truth.shape != reliability.shape:
        raise ValueError("one-query identifiers and labels must align")
    if len(set(windows.tolist())) != windows.size or not np.isfinite(reliability).all():
        raise ValueError("one-query identifiers must be unique and reliability finite")
    if truth.min() < 0 or truth.max() > 2 or not 0.0 <= minimum_stationary_mass <= 1.0:
        raise ValueError("one-query labels or stationary threshold are invalid")

    base_stationary = base[:, 1] + base[:, 2]
    intervention_stationary = intervention[:, 1] + intervention[:, 2]
    base_odds = np.log(np.clip(base[:, 1], 1e-12, 1.0)) - np.log(np.clip(base[:, 2], 1e-12, 1.0))
    intervention_odds = np.log(np.clip(intervention[:, 1], 1e-12, 1.0)) - np.log(
        np.clip(intervention[:, 2], 1e-12, 1.0)
    )
    score = (
        np.minimum(base_stationary, intervention_stationary)
        * reliability
        * np.abs(intervention_odds - base_odds)
    )
    eligible = (
        (base_stationary >= minimum_stationary_mass)
        & (intervention_stationary >= minimum_stationary_mass)
        & (base.argmax(axis=1) != intervention.argmax(axis=1))
        & (base.argmax(axis=1) != 0)
        & (intervention.argmax(axis=1) != 0)
    )
    candidates = np.flatnonzero(eligible).tolist()
    if not candidates:
        return OneQueryStateResult(
            probabilities=base.copy(),
            evaluation_mask=np.ones(base.shape[0], dtype=np.bool_),
            queried_index=None,
            selected_state="OFF",
            query_score=np.asarray(score, dtype=np.float64),
        )
    query = min(candidates, key=lambda index: (-float(score[index]), str(windows[index])))
    inverted = _swap_posture(intervention)
    paths = (base, intervention, inverted)
    names = ("OFF", "NORMAL", "INVERTED")
    observed = int(truth[query])
    likelihood = np.asarray([item[query, observed] for item in paths], dtype=np.float64)
    selected_index = int(np.argmax(likelihood))
    result = base.copy()
    selected = paths[selected_index]
    posture_rows = (base.argmax(axis=1) != 0) & (selected.argmax(axis=1) != 0)
    result[posture_rows] = selected[posture_rows]
    evaluation = np.ones(base.shape[0], dtype=np.bool_)
    evaluation[query] = False
    return OneQueryStateResult(
        probabilities=np.asarray(result, dtype=np.float64),
        evaluation_mask=evaluation,
        queried_index=query,
        selected_state=names[selected_index],
        query_score=np.asarray(score, dtype=np.float64),
    )


def accumulate_bout_intervention_evidence(
    scores: NDArray[np.floating[Any]],
    bout_ids: NDArray[np.str_],
    *,
    width: int,
) -> FloatArray:
    """Causal trailing evidence mean that resets at every declared bout."""

    values = np.asarray(scores, dtype=np.float64)
    bouts = np.asarray(bout_ids, dtype=np.str_)
    if values.ndim != 1 or bouts.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("bout scores and identifiers must be aligned finite vectors")
    if width < 1:
        raise ValueError("bout accumulator width must be positive")
    result = np.empty_like(values)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and bouts[end] == bouts[start]:
            end += 1
        for index in range(start, end):
            left = max(start, index - width + 1)
            result[index] = float(np.mean(values[left : index + 1]))
        start = end
    return np.asarray(result, dtype=np.float64)


__all__ = [
    "BinaryJackknifePrediction",
    "CandidateMarginalization",
    "DualFrameFeatures",
    "OneQueryStateResult",
    "RescueHarmTargets",
    "RidgeLogisticModel",
    "SentinelRoutingResult",
    "accumulate_bout_intervention_evidence",
    "apply_one_query_state_selection",
    "apply_rank_locked_posture_offset",
    "apply_rescue_harm_sentinel",
    "build_rescue_harm_features",
    "counterfactual_rescue_harm_targets",
    "dual_frame_feature_names",
    "extract_dual_frame_posture_features",
    "fit_participant_balanced_logistic",
    "participant_jackknife_binary_prediction",
    "stability_weighted_candidate_marginalization",
    "temperature_scale_probabilities",
]
