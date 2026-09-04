"""Counterfactual Advantage-Gated Experts for inertial HAR.

The module contains deterministic, participant-agnostic building blocks for the
prospective CAGE-HAR development lane.  It deliberately does not load datasets,
participant metadata, or labels except when constructing training-only
counterfactual advantage targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]


@dataclass(frozen=True, slots=True)
class CageContext:
    """Compact physical context and a label-free reliability score."""

    features: FloatArray
    reliability: FloatArray
    feature_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RidgeAdvantageModel:
    """Transparent regularized linear model for intervention advantage."""

    center: FloatArray
    scale: FloatArray
    coefficients: FloatArray
    intercept: float

    def predict(self, features: NDArray[np.floating[Any]]) -> FloatArray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.center.size:
            raise ValueError("advantage features do not match the fitted router")
        if not np.isfinite(values).all():
            raise ValueError("advantage features must be finite")
        standardized = (values - self.center) / self.scale
        return np.asarray(standardized @ self.coefficients + self.intercept, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class AdvantagePrediction:
    """Jackknife mean, dispersion, and conservative stability bound."""

    mean: FloatArray
    standard_deviation: FloatArray
    lower_bound: FloatArray
    model_count: int


@dataclass(frozen=True, slots=True)
class CageRoutingResult:
    """Probabilities and complete per-window routing diagnostics."""

    probabilities: FloatArray
    routed: BoolArray
    chosen_expert: IntArray
    mix_weight: FloatArray
    achieved_kl: FloatArray
    predicted_advantage_lower_bound: FloatArray


@dataclass(frozen=True, slots=True)
class SemanticGaugeResult:
    """Soft sitting/standing gauge repair after a label-safe query selection."""

    probabilities: FloatArray
    posterior_swap_probability: float
    log_bayes_factor_swap_over_identity: float


_GAUGE_FEATURE_NAMES = (
    "gravity_valid_fraction",
    "gravity_norm_median",
    "gravity_norm_mad",
    "gravity_direction_deviation_median_rad",
    "gravity_direction_deviation_p90_rad",
    "gravity_angular_speed_median_rad_s",
    "gravity_angular_speed_p90_rad_s",
    "central_gravity_x",
    "central_gravity_y",
    "central_gravity_z",
    "central_gravity_abs_sorted_low",
    "central_gravity_abs_sorted_mid",
    "central_gravity_abs_sorted_high",
    "user_acceleration_parallel_mean",
    "user_acceleration_parallel_std",
    "user_acceleration_perpendicular_mean",
    "user_acceleration_perpendicular_std",
    "gyroscope_parallel_mean",
    "gyroscope_parallel_std",
    "gyroscope_perpendicular_mean",
    "gyroscope_perpendicular_std",
    "total_acceleration_parallel_median",
    "total_acceleration_parallel_std",
    "instantaneous_central_acceleration_projection_disagreement",
    "instantaneous_central_gyroscope_projection_disagreement",
)

_HEALTH_METRICS = (
    "valid_fraction_mean",
    "valid_fraction_min",
    "longest_gap_fraction_max",
    "time_since_last_valid_fraction_max",
    "stuck_fraction_max",
    "saturation_fraction_max",
    "endpoint_drift_scaled_max",
    "high_frequency_scaled_median",
)


def _validate_sensor_inputs(
    primary_signals: NDArray[np.floating[Any]],
    gravity: NDArray[np.floating[Any]],
    validity_mask: NDArray[np.bool_] | None,
    sampling_rate_hz: float,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    primary = np.asarray(primary_signals, dtype=np.float64)
    gravity_values = np.asarray(gravity, dtype=np.float64)
    if primary.ndim != 3 or primary.shape[1] < 8 or primary.shape[2] != 6:
        raise ValueError("CAGE-HAR requires primary input shaped [window,time>=8,6]")
    if gravity_values.shape != (*primary.shape[:2], 3):
        raise ValueError("gravity must align with primary input as [window,time,3]")
    if primary.shape[0] == 0 or not np.isfinite(primary).all():
        raise ValueError("primary signals must be non-empty and finite")
    if not np.isfinite(gravity_values).all():
        raise ValueError("gravity values must be finite; missingness belongs in the mask")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0.0:
        raise ValueError("sampling rate must be finite and positive")
    if validity_mask is None:
        validity = np.ones((*primary.shape[:2], 9), dtype=np.bool_)
    else:
        validity = np.asarray(validity_mask, dtype=np.bool_)
        if validity.shape != (*primary.shape[:2], 9):
            raise ValueError("validity mask must align as [window,time,9]")
    return primary, gravity_values, validity


def _masked_values(values: FloatArray, valid: BoolArray, row: int) -> FloatArray:
    selected = np.asarray(values[row][valid[row]], dtype=np.float64)
    return selected if selected.size else np.zeros(1, dtype=np.float64)


def _masked_mean_std(values: FloatArray, valid: BoolArray) -> tuple[FloatArray, FloatArray]:
    means = np.empty(values.shape[0], dtype=np.float64)
    deviations = np.empty(values.shape[0], dtype=np.float64)
    for row in range(values.shape[0]):
        selected = _masked_values(values, valid, row)
        means[row] = float(np.mean(selected))
        deviations[row] = float(np.std(selected))
    return means, deviations


def _masked_median(values: FloatArray, valid: BoolArray) -> FloatArray:
    result = np.empty(values.shape[0], dtype=np.float64)
    for row in range(values.shape[0]):
        result[row] = float(np.median(_masked_values(values, valid, row)))
    return result


def _masked_quantile(values: FloatArray, valid: BoolArray, quantile: float) -> FloatArray:
    result = np.empty(values.shape[0], dtype=np.float64)
    for row in range(values.shape[0]):
        result[row] = float(np.quantile(_masked_values(values, valid, row), quantile))
    return result


def extract_cage_gauge_context(
    primary_signals: NDArray[np.floating[Any]],
    gravity: NDArray[np.floating[Any]],
    *,
    sampling_rate_hz: float,
    validity_mask: NDArray[np.bool_] | None = None,
) -> CageContext:
    """Extract a partial-invariant gravity gauge without erasing posture orientation.

    Scalar projections are invariant to a shared proper rotation of all sensor
    vectors.  The central gravity coordinates are retained separately because
    absolute device orientation can contain posture information.  Reliability
    falls to zero when no native gravity sample is available.
    """

    primary, gravity_values, validity = _validate_sensor_inputs(
        primary_signals, gravity, validity_mask, sampling_rate_hz
    )
    gravity_valid = np.all(validity[:, :, 6:9], axis=2)
    gravity_norm = np.linalg.norm(gravity_values, axis=2)
    gravity_valid &= gravity_norm > 1e-8
    window_count = gravity_valid.shape[0]

    central = np.zeros((window_count, 3), dtype=np.float64)
    central[:, 2] = 1.0
    for row in range(window_count):
        if gravity_valid[row].any():
            # A vector mean is rotation-equivariant; component-wise medians are
            # not.  Robustness to bad gravity samples is represented explicitly
            # through norm/direction dispersion and handled by the router.
            candidate = np.mean(gravity_values[row, gravity_valid[row]], axis=0)
            norm = float(np.linalg.norm(candidate))
            if norm > 1e-8:
                central[row] = candidate / norm

    native_unit = gravity_values / np.maximum(gravity_norm[:, :, None], 1e-8)
    unit = np.broadcast_to(central[:, None, :], gravity_values.shape).copy()
    unit[gravity_valid] = native_unit[gravity_valid]
    cosine_to_center = np.sum(unit * central[:, None, :], axis=2)
    direction_deviation = np.arccos(np.clip(cosine_to_center, -1.0, 1.0))
    previous = np.concatenate((unit[:, :1], unit[:, :-1]), axis=1)
    angular_speed = (
        np.arccos(np.clip(np.sum(previous * unit, axis=2), -1.0, 1.0)) * sampling_rate_hz
    )

    acceleration = primary[:, :, :3]
    gyroscope = primary[:, :, 3:]
    acceleration_valid = np.all(validity[:, :, :3], axis=2) & gravity_valid
    gyroscope_valid = np.all(validity[:, :, 3:6], axis=2) & gravity_valid
    acceleration_parallel = np.sum(acceleration * unit, axis=2)
    acceleration_central = np.sum(acceleration * central[:, None, :], axis=2)
    acceleration_perpendicular = np.linalg.norm(
        acceleration - acceleration_parallel[:, :, None] * unit, axis=2
    )
    gyroscope_parallel = np.sum(gyroscope * unit, axis=2)
    gyroscope_central = np.sum(gyroscope * central[:, None, :], axis=2)
    gyroscope_perpendicular = np.linalg.norm(
        gyroscope - gyroscope_parallel[:, :, None] * unit, axis=2
    )
    total_acceleration = acceleration + gravity_values
    total_parallel = np.sum(total_acceleration * unit, axis=2)

    gravity_fraction = gravity_valid.mean(axis=1)
    gravity_norm_median = _masked_median(gravity_norm, gravity_valid)
    gravity_norm_mad = _masked_median(
        np.abs(gravity_norm - gravity_norm_median[:, None]), gravity_valid
    )
    deviation_median = _masked_median(direction_deviation, gravity_valid)
    deviation_p90 = _masked_quantile(direction_deviation, gravity_valid, 0.90)
    angular_median = _masked_median(angular_speed, gravity_valid)
    angular_p90 = _masked_quantile(angular_speed, gravity_valid, 0.90)
    acc_parallel_mean, acc_parallel_std = _masked_mean_std(
        acceleration_parallel, acceleration_valid
    )
    acc_perpendicular_mean, acc_perpendicular_std = _masked_mean_std(
        acceleration_perpendicular, acceleration_valid
    )
    gyro_parallel_mean, gyro_parallel_std = _masked_mean_std(gyroscope_parallel, gyroscope_valid)
    gyro_perpendicular_mean, gyro_perpendicular_std = _masked_mean_std(
        gyroscope_perpendicular, gyroscope_valid
    )
    total_parallel_median = _masked_median(total_parallel, acceleration_valid)
    _, total_parallel_std = _masked_mean_std(total_parallel, acceleration_valid)
    acceleration_disagreement = _masked_median(
        np.abs(acceleration_parallel - acceleration_central), acceleration_valid
    )
    gyroscope_disagreement = _masked_median(
        np.abs(gyroscope_parallel - gyroscope_central), gyroscope_valid
    )
    sorted_central = np.sort(np.abs(central), axis=1)
    features = np.column_stack(
        (
            gravity_fraction,
            gravity_norm_median,
            gravity_norm_mad,
            deviation_median,
            deviation_p90,
            angular_median,
            angular_p90,
            central,
            sorted_central,
            acc_parallel_mean,
            acc_parallel_std,
            acc_perpendicular_mean,
            acc_perpendicular_std,
            gyro_parallel_mean,
            gyro_parallel_std,
            gyro_perpendicular_mean,
            gyro_perpendicular_std,
            total_parallel_median,
            total_parallel_std,
            acceleration_disagreement,
            gyroscope_disagreement,
        )
    )
    norm_scale = np.maximum(np.abs(gravity_norm_median), 1e-6)
    norm_stability = np.exp(-gravity_norm_mad / norm_scale)
    direction_stability = np.exp(-deviation_p90)
    primary_fraction = validity[:, :, :6].mean(axis=(1, 2))
    reliability = np.clip(
        gravity_fraction * primary_fraction * norm_stability * direction_stability,
        0.0,
        1.0,
    )
    if features.shape != (window_count, len(_GAUGE_FEATURE_NAMES)):
        raise AssertionError("CAGE gravity-gauge feature contract changed")
    if not np.isfinite(features).all() or not np.isfinite(reliability).all():
        raise AssertionError("CAGE gravity-gauge output must be finite")
    return CageContext(
        np.asarray(features, dtype=np.float64),
        np.asarray(reliability, dtype=np.float64),
        _GAUGE_FEATURE_NAMES,
    )


def _longest_false_run(mask: BoolArray) -> int:
    longest = 0
    current = 0
    for value in mask.tolist():
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def extract_cage_sensor_health(
    signals: NDArray[np.floating[Any]],
    *,
    validity_mask: NDArray[np.bool_] | None = None,
    channel_scale: NDArray[np.floating[Any]] | None = None,
    saturation_threshold: NDArray[np.floating[Any]] | None = None,
    modality_slices: tuple[tuple[str, int, int], ...] = (
        ("accelerometer", 0, 3),
        ("gyroscope", 3, 6),
        ("gravity", 6, 9),
    ),
) -> CageContext:
    """Summarize explicit missingness and observable sensor-health symptoms."""

    values = np.asarray(signals, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] == 0 or values.shape[1] < 8:
        raise ValueError("sensor health requires [window,time>=8,channel] input")
    if not np.isfinite(values).all():
        raise ValueError("sensor values must be finite; missingness belongs in the mask")
    if validity_mask is None:
        validity = np.ones(values.shape, dtype=np.bool_)
    else:
        validity = np.asarray(validity_mask, dtype=np.bool_)
        if validity.shape != values.shape:
            raise ValueError("sensor validity mask must align with signals")
    channels = values.shape[2]
    if channel_scale is None:
        scale = np.median(np.abs(values), axis=(0, 1))
        scale = np.maximum(scale, 1e-6)
    else:
        scale = np.asarray(channel_scale, dtype=np.float64)
        if scale.shape != (channels,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("channel scale must contain one finite positive value per channel")
    if saturation_threshold is None:
        saturation = np.full(channels, np.inf, dtype=np.float64)
    else:
        saturation = np.asarray(saturation_threshold, dtype=np.float64)
        if (
            saturation.shape != (channels,)
            or np.any(np.isnan(saturation))
            or np.any(saturation <= 0)
        ):
            raise ValueError("saturation thresholds must be positive and align with channels")
    covered: list[int] = []
    names: list[str] = []
    feature_blocks: list[FloatArray] = []
    modality_reliability: list[FloatArray] = []
    time_steps = values.shape[1]
    for modality, start, stop in modality_slices:
        if start < 0 or stop <= start or stop > channels:
            raise ValueError("modality slices must be non-empty and inside the channel tensor")
        covered.extend(range(start, stop))
        current_values = values[:, :, start:stop]
        current_valid = validity[:, :, start:stop]
        valid_fraction = current_valid.mean(axis=1)
        longest_gap = np.empty((values.shape[0], stop - start), dtype=np.float64)
        since_last = np.empty_like(longest_gap)
        for row in range(values.shape[0]):
            for column in range(stop - start):
                mask = current_valid[row, :, column]
                longest_gap[row, column] = _longest_false_run(mask) / time_steps
                valid_indices = np.flatnonzero(mask)
                since_last[row, column] = (
                    1.0
                    if valid_indices.size == 0
                    else float(time_steps - 1 - valid_indices[-1]) / time_steps
                )
        differences = np.diff(current_values, axis=1)
        valid_pairs = current_valid[:, 1:] & current_valid[:, :-1]
        scaled_tolerance = scale[None, None, start:stop] * 1e-4
        stuck = np.sum(
            (np.abs(differences) <= scaled_tolerance) & valid_pairs, axis=1
        ) / np.maximum(np.sum(valid_pairs, axis=1), 1)
        saturated = np.sum(
            (np.abs(current_values) >= saturation[None, None, start:stop]) & current_valid,
            axis=1,
        ) / np.maximum(np.sum(current_valid, axis=1), 1)
        quarter = max(1, time_steps // 4)
        drift = (
            np.abs(
                np.median(current_values[:, -quarter:], axis=1)
                - np.median(current_values[:, :quarter], axis=1)
            )
            / scale[None, start:stop]
        )
        high_frequency = np.median(np.abs(differences), axis=1) / scale[None, start:stop]
        block = np.column_stack(
            (
                valid_fraction.mean(axis=1),
                valid_fraction.min(axis=1),
                longest_gap.max(axis=1),
                since_last.max(axis=1),
                stuck.max(axis=1),
                saturated.max(axis=1),
                drift.max(axis=1),
                np.median(high_frequency, axis=1),
            )
        )
        feature_blocks.append(np.asarray(block, dtype=np.float64))
        names.extend(f"{modality}__{metric}" for metric in _HEALTH_METRICS)
        # Flat or slowly drifting signals can be physically genuine within one
        # short HAR window.  Expose those symptoms to the learned router, but do
        # not turn them into a hard validity gate without longitudinal evidence.
        direct_health = (
            block[:, 0]
            * (1.0 - block[:, 2])
            * (1.0 - block[:, 3])
            * (1.0 - np.clip(block[:, 5], 0.0, 1.0))
        )
        modality_reliability.append(np.clip(direct_health, 0.0, 1.0))
    if sorted(covered) != list(range(channels)) or len(set(covered)) != channels:
        raise ValueError("modality slices must cover each sensor channel exactly once")
    features = np.concatenate(feature_blocks, axis=1)
    reliability = np.min(np.stack(modality_reliability, axis=1), axis=1)
    if features.shape != (values.shape[0], len(names)) or not np.isfinite(features).all():
        raise AssertionError("CAGE sensor-health feature contract changed")
    return CageContext(features, reliability, tuple(names))


def cage_expert_reliability(
    sensor_health: CageContext,
    *,
    required_modalities: tuple[str, ...],
    gravity_gauge: CageContext | None = None,
) -> FloatArray:
    """Derive an expert-specific hard reliability gate from explicit evidence."""

    if not required_modalities or len(set(required_modalities)) != len(required_modalities):
        raise ValueError("an expert must declare unique required modalities")
    if sensor_health.features.ndim != 2 or sensor_health.reliability.shape != (
        sensor_health.features.shape[0],
    ):
        raise ValueError("sensor-health context arrays are not aligned")
    names = {name: index for index, name in enumerate(sensor_health.feature_names)}
    modality_scores: list[FloatArray] = []
    for modality in required_modalities:
        required_names = (
            f"{modality}__valid_fraction_min",
            f"{modality}__longest_gap_fraction_max",
            f"{modality}__time_since_last_valid_fraction_max",
            f"{modality}__saturation_fraction_max",
        )
        if any(name not in names for name in required_names):
            raise ValueError(f"sensor-health context lacks modality {modality!r}")
        valid = sensor_health.features[:, names[required_names[0]]]
        longest_gap = sensor_health.features[:, names[required_names[1]]]
        since_last = sensor_health.features[:, names[required_names[2]]]
        saturation = sensor_health.features[:, names[required_names[3]]]
        modality_scores.append(
            np.clip(
                valid
                * (1.0 - np.clip(longest_gap, 0.0, 1.0))
                * (1.0 - np.clip(since_last, 0.0, 1.0))
                * (1.0 - np.clip(saturation, 0.0, 1.0)),
                0.0,
                1.0,
            )
        )
    reliability = np.min(np.stack(modality_scores, axis=1), axis=1)
    if "gravity" in required_modalities:
        if gravity_gauge is None:
            raise ValueError("a gravity-dependent expert requires gravity-gauge reliability")
        if gravity_gauge.reliability.shape != reliability.shape:
            raise ValueError("gravity-gauge reliability does not align with sensor health")
        reliability = np.minimum(reliability, gravity_gauge.reliability)
    return np.asarray(reliability, dtype=np.float64)


def _probabilities(values: NDArray[np.floating[Any]], *, name: str) -> FloatArray:
    probability = np.asarray(values, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] != 3 or probability.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty [sample,3] matrix")
    if not np.isfinite(probability).all() or np.any(probability < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError(f"{name} rows must sum to one")
    return probability


def counterfactual_log_loss_advantage(
    base_probability: NDArray[np.floating[Any]],
    expert_probability: NDArray[np.floating[Any]],
    labels: NDArray[np.integer[Any]],
) -> FloatArray:
    """Return training-only base loss minus expert loss for each sample."""

    base = _probabilities(base_probability, name="base probability")
    expert = _probabilities(expert_probability, name="expert probability")
    truth = np.asarray(labels, dtype=np.int64)
    if expert.shape != base.shape or truth.shape != (base.shape[0],):
        raise ValueError("counterfactual advantage inputs are not aligned")
    if truth.min() < 0 or truth.max() > 2:
        raise ValueError("counterfactual labels must lie in [0,2]")
    rows = np.arange(truth.size)
    return np.asarray(
        np.log(np.clip(expert[rows, truth], 1e-12, 1.0))
        - np.log(np.clip(base[rows, truth], 1e-12, 1.0)),
        dtype=np.float64,
    )


def select_semantic_gauge_query(
    base_probability: NDArray[np.floating[Any]],
    gauge_expert_probability: NDArray[np.floating[Any]],
    expert_reliability: NDArray[np.floating[Any]],
    window_ids: NDArray[np.str_],
    *,
    minimum_stationary_probability: float,
    minimum_suspicion_score: float,
) -> tuple[int | None, FloatArray]:
    """Select one semantic query from predictions only, without reading labels."""

    base = _probabilities(base_probability, name="base probability")
    expert = _probabilities(gauge_expert_probability, name="gauge expert probability")
    reliability = np.asarray(expert_reliability, dtype=np.float64)
    windows = np.asarray(window_ids, dtype=np.str_)
    if expert.shape != base.shape or reliability.shape != (base.shape[0],):
        raise ValueError("semantic-gauge predictions and reliability do not align")
    if windows.shape != reliability.shape or len(set(windows.tolist())) != windows.size:
        raise ValueError("semantic-gauge window identifiers must align and be unique")
    if (
        not np.isfinite(reliability).all()
        or np.any((reliability < 0.0) | (reliability > 1.0))
        or not 0.0 <= minimum_stationary_probability <= 1.0
        or not np.isfinite(minimum_suspicion_score)
        or minimum_suspicion_score < 0.0
    ):
        raise ValueError("semantic-gauge query settings are invalid")
    base_stationary = base[:, 1] + base[:, 2]
    expert_stationary = expert[:, 1] + expert[:, 2]
    base_log_odds = np.log(np.clip(base[:, 1], 1e-12, 1.0)) - np.log(
        np.clip(base[:, 2], 1e-12, 1.0)
    )
    expert_log_odds = np.log(np.clip(expert[:, 1], 1e-12, 1.0)) - np.log(
        np.clip(expert[:, 2], 1e-12, 1.0)
    )
    opposite_direction = np.maximum(0.0, -(base_log_odds * expert_log_odds))
    score = (
        np.minimum(base_stationary, expert_stationary) * reliability * np.sqrt(opposite_direction)
    )
    eligible = (
        (base_stationary >= minimum_stationary_probability)
        & (expert_stationary >= minimum_stationary_probability)
        & (score >= minimum_suspicion_score)
    )
    indices = np.flatnonzero(eligible).tolist()
    if not indices:
        return None, np.asarray(score, dtype=np.float64)
    selected = min(indices, key=lambda index: (-float(score[index]), str(windows[index])))
    return selected, np.asarray(score, dtype=np.float64)


def apply_soft_semantic_gauge(
    probabilities: NDArray[np.floating[Any]],
    *,
    queried_index: int,
    observed_label: int,
    prior_swap_probability: float,
    maximum_swap_weight: float,
    evidence_temperature: float,
) -> SemanticGaugeResult:
    """Softly blend identity and posture-swap hypotheses after one observed label."""

    values = _probabilities(probabilities, name="semantic-gauge probability")
    if queried_index < 0 or queried_index >= values.shape[0]:
        raise ValueError("semantic-gauge query index is out of range")
    if observed_label not in {0, 1, 2}:
        raise ValueError("semantic-gauge observed label must lie in [0,2]")
    if (
        not 0.0 < prior_swap_probability < 1.0
        or not 0.0 <= maximum_swap_weight <= 1.0
        or not np.isfinite(evidence_temperature)
        or evidence_temperature <= 0.0
    ):
        raise ValueError("semantic-gauge posterior settings are invalid")
    log_bayes = 0.0
    if observed_label in {1, 2}:
        other = 2 if observed_label == 1 else 1
        log_bayes = float(
            np.log(max(float(values[queried_index, other]), 1e-12))
            - np.log(max(float(values[queried_index, observed_label]), 1e-12))
        )
    prior_log_odds = np.log(prior_swap_probability) - np.log(1.0 - prior_swap_probability)
    posterior_log_odds = prior_log_odds + log_bayes / evidence_temperature
    posterior = float(1.0 / (1.0 + np.exp(-np.clip(posterior_log_odds, -40.0, 40.0))))
    swap_weight = maximum_swap_weight * posterior
    swapped = values.copy()
    swapped[:, [1, 2]] = swapped[:, [2, 1]]
    adjusted = (1.0 - swap_weight) * values + swap_weight * swapped
    adjusted[:, 0] = values[:, 0]
    stationary = 1.0 - values[:, 0]
    adjusted[:, 1:] *= stationary[:, None] / np.maximum(adjusted[:, 1:].sum(axis=1)[:, None], 1e-12)
    return SemanticGaugeResult(
        probabilities=np.asarray(adjusted, dtype=np.float64),
        posterior_swap_probability=posterior,
        log_bayes_factor_swap_over_identity=log_bayes,
    )


def build_advantage_router_features(
    base_probability: NDArray[np.floating[Any]],
    expert_probability: NDArray[np.floating[Any]],
    *,
    context_features: NDArray[np.floating[Any]],
    expert_reliability: NDArray[np.floating[Any]],
) -> tuple[FloatArray, tuple[str, ...]]:
    """Construct label-free competence features for one base/expert pair."""

    base = _probabilities(base_probability, name="base probability")
    expert = _probabilities(expert_probability, name="expert probability")
    context = np.asarray(context_features, dtype=np.float64)
    reliability = np.asarray(expert_reliability, dtype=np.float64)
    if expert.shape != base.shape:
        raise ValueError("base and expert probabilities must align")
    if context.ndim != 2 or context.shape[0] != base.shape[0]:
        raise ValueError("router context features must align with probabilities")
    if reliability.shape != (base.shape[0],) or not np.isfinite(reliability).all():
        raise ValueError("expert reliability must align with probabilities")
    if not np.isfinite(context).all() or np.any((reliability < 0.0) | (reliability > 1.0)):
        raise ValueError("router context and reliability must be finite and bounded")

    def margin(probability: FloatArray) -> FloatArray:
        ordered = np.sort(probability, axis=1)
        return ordered[:, -1] - ordered[:, -2]

    def entropy(probability: FloatArray) -> FloatArray:
        clipped = np.clip(probability, 1e-12, 1.0)
        return np.asarray(-np.sum(clipped * np.log(clipped), axis=1), dtype=np.float64)

    midpoint = 0.5 * (base + expert)
    js = 0.5 * np.sum(
        base * (np.log(np.clip(base, 1e-12, 1.0)) - np.log(np.clip(midpoint, 1e-12, 1.0)))
        + expert * (np.log(np.clip(expert, 1e-12, 1.0)) - np.log(np.clip(midpoint, 1e-12, 1.0))),
        axis=1,
    )
    derived = np.column_stack(
        (
            base.max(axis=1),
            margin(base),
            entropy(base),
            base[:, 1] + base[:, 2],
            base[:, 1] - base[:, 2],
            expert.max(axis=1),
            margin(expert),
            entropy(expert),
            expert[:, 1] + expert[:, 2],
            expert[:, 1] - expert[:, 2],
            np.sum(np.abs(expert - base), axis=1),
            js,
            (base.argmax(axis=1) != expert.argmax(axis=1)).astype(np.float64),
            reliability,
        )
    )
    names = (
        "base_max_probability",
        "base_margin",
        "base_entropy",
        "base_stationary_mass",
        "base_signed_sitting_minus_standing",
        "expert_max_probability",
        "expert_margin",
        "expert_entropy",
        "expert_stationary_mass",
        "expert_signed_sitting_minus_standing",
        "base_expert_l1_disagreement",
        "base_expert_jensen_shannon",
        "base_expert_label_disagreement",
        "expert_reliability",
        *(f"context_{index:03d}" for index in range(context.shape[1])),
    )
    features = np.concatenate((derived, context), axis=1)
    return np.asarray(features, dtype=np.float64), names


def _participant_weights(participants: StringArray) -> FloatArray:
    weights = np.empty(participants.size, dtype=np.float64)
    for participant in np.unique(participants):
        selected = participants == participant
        weights[selected] = 1.0 / float(selected.sum())
    weights *= participants.size / weights.sum()
    return weights


def fit_ridge_advantage_model(
    features: NDArray[np.floating[Any]],
    advantage: NDArray[np.floating[Any]],
    participant_ids: NDArray[np.str_],
    *,
    ridge_penalty: float,
    sample_weight: NDArray[np.floating[Any]] | None = None,
) -> RidgeAdvantageModel:
    """Fit a participant-balanced ridge model with an unpenalized intercept."""

    values = np.asarray(features, dtype=np.float64)
    target = np.asarray(advantage, dtype=np.float64)
    participants = np.asarray(participant_ids, dtype=np.str_)
    if (
        values.ndim != 2
        or values.shape[0] < 2
        or target.shape != (values.shape[0],)
        or participants.shape != target.shape
    ):
        raise ValueError("ridge router inputs must be aligned and non-empty")
    if not np.isfinite(values).all() or not np.isfinite(target).all():
        raise ValueError("ridge router inputs must be finite")
    if np.unique(participants).size < 2:
        raise ValueError("ridge router needs at least two participants")
    if not np.isfinite(ridge_penalty) or ridge_penalty <= 0.0:
        raise ValueError("ridge penalty must be finite and positive")
    weights = _participant_weights(participants)
    if sample_weight is not None:
        supplied_weight = np.asarray(sample_weight, dtype=np.float64)
        if (
            supplied_weight.shape != target.shape
            or not np.isfinite(supplied_weight).all()
            or np.any(supplied_weight < 0.0)
            or not np.any(supplied_weight > 0.0)
        ):
            raise ValueError("router sample weights must be aligned, finite, and non-negative")
        weights *= supplied_weight
        weights *= target.size / weights.sum()
    normalized_weights = weights / weights.sum()
    center = np.sum(values * normalized_weights[:, None], axis=0)
    variance = np.sum((values - center) ** 2 * normalized_weights[:, None], axis=0)
    scale = np.sqrt(np.maximum(variance, 1e-12))
    standardized = (values - center) / scale
    design = np.column_stack((np.ones(values.shape[0]), standardized))
    weighted_design = design * np.sqrt(weights[:, None])
    weighted_target = target * np.sqrt(weights)
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge_penalty
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(weighted_design.T @ weighted_design + penalty) @ (
        weighted_design.T @ weighted_target
    )
    return RidgeAdvantageModel(
        center=np.asarray(center, dtype=np.float64),
        scale=np.asarray(scale, dtype=np.float64),
        coefficients=np.asarray(coefficients[1:], dtype=np.float64),
        intercept=float(coefficients[0]),
    )


def participant_jackknife_advantage_prediction(
    training_features: NDArray[np.floating[Any]],
    training_advantage: NDArray[np.floating[Any]],
    training_participant_ids: NDArray[np.str_],
    evaluation_features: NDArray[np.floating[Any]],
    *,
    ridge_penalty: float,
    lower_bound_z: float,
    training_sample_weight: NDArray[np.floating[Any]] | None = None,
) -> AdvantagePrediction:
    """Predict intervention advantage with participant-jackknife stability bounds."""

    train = np.asarray(training_features, dtype=np.float64)
    target = np.asarray(training_advantage, dtype=np.float64)
    participants = np.asarray(training_participant_ids, dtype=np.str_)
    evaluate = np.asarray(evaluation_features, dtype=np.float64)
    unique = np.unique(participants)
    if unique.size < 4:
        raise ValueError("advantage jackknife requires at least four training participants")
    if target.shape != (train.shape[0],) or participants.shape != target.shape:
        raise ValueError("advantage jackknife training arrays are not aligned")
    supplied_weight: FloatArray | None = None
    if training_sample_weight is not None:
        supplied_weight = np.asarray(training_sample_weight, dtype=np.float64)
        if supplied_weight.shape != target.shape:
            raise ValueError("advantage jackknife sample weights do not align")
    if evaluate.ndim != 2 or evaluate.shape[1] != train.shape[1]:
        raise ValueError("advantage jackknife evaluation features do not align")
    if not np.isfinite(lower_bound_z) or lower_bound_z < 0.0:
        raise ValueError("lower-bound multiplier must be finite and non-negative")
    predictions: list[FloatArray] = []
    for excluded in unique:
        retained = participants != excluded
        model = fit_ridge_advantage_model(
            train[retained],
            target[retained],
            participants[retained],
            ridge_penalty=ridge_penalty,
            sample_weight=None if supplied_weight is None else supplied_weight[retained],
        )
        predictions.append(model.predict(evaluate))
    stack = np.stack(predictions, axis=0)
    mean = stack.mean(axis=0)
    deviation = stack.std(axis=0, ddof=1)
    return AdvantagePrediction(
        mean=np.asarray(mean, dtype=np.float64),
        standard_deviation=np.asarray(deviation, dtype=np.float64),
        lower_bound=np.asarray(mean - lower_bound_z * deviation, dtype=np.float64),
        model_count=len(predictions),
    )


def _kl_divergence(candidate: FloatArray, reference: FloatArray) -> FloatArray:
    left = np.clip(candidate, 1e-12, 1.0)
    right = np.clip(reference, 1e-12, 1.0)
    return np.asarray(np.sum(left * (np.log(left) - np.log(right)), axis=1), dtype=np.float64)


def _mix_one(
    base: FloatArray,
    expert: FloatArray,
    weight: FloatArray,
    posture_only: BoolArray,
) -> FloatArray:
    result = (1.0 - weight[:, None]) * base + weight[:, None] * expert
    if posture_only.any():
        rows = np.flatnonzero(posture_only)
        stationary_mass = 1.0 - base[rows, 0]
        base_conditional = base[rows, 1:] / np.maximum(base[rows, 1:].sum(axis=1)[:, None], 1e-12)
        expert_conditional = expert[rows, 1:] / np.maximum(
            expert[rows, 1:].sum(axis=1)[:, None], 1e-12
        )
        conditional = (1.0 - weight[rows, None]) * base_conditional + weight[
            rows, None
        ] * expert_conditional
        conditional /= conditional.sum(axis=1, keepdims=True)
        result[rows, 0] = base[rows, 0]
        result[rows, 1:] = stationary_mass[:, None] * conditional
    result /= result.sum(axis=1, keepdims=True)
    return np.asarray(result, dtype=np.float64)


def apply_cage_trust_region(
    base_probability: NDArray[np.floating[Any]],
    expert_probabilities: NDArray[np.floating[Any]],
    predicted_advantage_lower_bounds: NDArray[np.floating[Any]],
    expert_reliability: NDArray[np.floating[Any]],
    expert_posture_only: NDArray[np.bool_],
    *,
    minimum_advantage: float,
    minimum_reliability: float,
    maximum_mix_weight: float,
    advantage_scale: float,
    maximum_kl: float,
) -> CageRoutingResult:
    """Apply the best eligible expert while bounding probability displacement."""

    base = _probabilities(base_probability, name="base probability")
    experts = np.asarray(expert_probabilities, dtype=np.float64)
    bounds = np.asarray(predicted_advantage_lower_bounds, dtype=np.float64)
    reliability = np.asarray(expert_reliability, dtype=np.float64)
    posture_only = np.asarray(expert_posture_only, dtype=np.bool_)
    if experts.ndim != 3 or experts.shape[0] != base.shape[0] or experts.shape[2] != 3:
        raise ValueError("expert probabilities must be [sample,expert,3]")
    if bounds.shape != experts.shape[:2] or reliability.shape != bounds.shape:
        raise ValueError("advantage bounds and reliability must align with experts")
    if posture_only.shape != (experts.shape[1],):
        raise ValueError("posture-only flags must contain one value per expert")
    if (
        not np.isfinite(experts).all()
        or np.any(experts < 0.0)
        or not np.allclose(experts.sum(axis=2), 1.0, atol=1e-6, rtol=1e-6)
        or not np.isfinite(bounds).all()
        or not np.isfinite(reliability).all()
        or np.any((reliability < 0.0) | (reliability > 1.0))
    ):
        raise ValueError("expert routing arrays are invalid")
    scalars = (
        minimum_advantage,
        minimum_reliability,
        maximum_mix_weight,
        advantage_scale,
        maximum_kl,
    )
    if not all(np.isfinite(value) for value in scalars):
        raise ValueError("trust-region settings must be finite")
    if (
        minimum_reliability < 0.0
        or minimum_reliability > 1.0
        or maximum_mix_weight < 0.0
        or maximum_mix_weight > 1.0
        or advantage_scale <= 0.0
        or maximum_kl < 0.0
    ):
        raise ValueError("trust-region settings are outside their allowed ranges")

    eligible_bounds = np.where(reliability >= minimum_reliability, bounds, -np.inf)
    chosen = np.argmax(eligible_bounds, axis=1).astype(np.int64)
    rows = np.arange(base.shape[0])
    chosen_bound = eligible_bounds[rows, chosen]
    routed = chosen_bound > minimum_advantage
    raw_weight = maximum_mix_weight * np.clip(
        (chosen_bound - minimum_advantage) / advantage_scale, 0.0, 1.0
    )
    weight = np.where(routed, raw_weight, 0.0)
    chosen_expert = experts[rows, chosen]
    chosen_posture = posture_only[chosen]

    candidate = _mix_one(base, chosen_expert, weight, chosen_posture)
    if maximum_kl == 0.0:
        weight[:] = 0.0
        candidate = base.copy()
    else:
        over = _kl_divergence(candidate, base) > maximum_kl
        if over.any():
            low = np.zeros(int(over.sum()), dtype=np.float64)
            high = weight[over].copy()
            over_base = base[over]
            over_expert = chosen_expert[over]
            over_posture = chosen_posture[over]
            for _ in range(32):
                middle = 0.5 * (low + high)
                trial = _mix_one(over_base, over_expert, middle, over_posture)
                acceptable = _kl_divergence(trial, over_base) <= maximum_kl
                low = np.where(acceptable, middle, low)
                high = np.where(acceptable, high, middle)
            weight[over] = low
            candidate[over] = _mix_one(over_base, over_expert, low, over_posture)
    routed &= weight > 0.0
    chosen = np.where(routed, chosen, -1).astype(np.int64)
    achieved_kl = _kl_divergence(candidate, base)
    if not np.allclose(candidate.sum(axis=1), 1.0, atol=1e-9, rtol=1e-9):
        raise AssertionError("CAGE trust-region probabilities do not sum to one")
    if np.any(achieved_kl > maximum_kl + 1e-9):
        raise AssertionError("CAGE trust-region KL cap was exceeded")
    return CageRoutingResult(
        probabilities=np.asarray(candidate, dtype=np.float64),
        routed=np.asarray(routed, dtype=np.bool_),
        chosen_expert=chosen,
        mix_weight=np.asarray(weight, dtype=np.float64),
        achieved_kl=np.asarray(achieved_kl, dtype=np.float64),
        predicted_advantage_lower_bound=np.asarray(chosen_bound, dtype=np.float64),
    )


__all__ = [
    "AdvantagePrediction",
    "CageContext",
    "CageRoutingResult",
    "RidgeAdvantageModel",
    "SemanticGaugeResult",
    "apply_cage_trust_region",
    "apply_soft_semantic_gauge",
    "build_advantage_router_features",
    "cage_expert_reliability",
    "counterfactual_log_loss_advantage",
    "extract_cage_gauge_context",
    "extract_cage_sensor_health",
    "fit_ridge_advantage_model",
    "participant_jackknife_advantage_prediction",
    "select_semantic_gauge_query",
]
