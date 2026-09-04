"""Train-only posture microstates and transition-graph features for short IMU windows."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans  # type: ignore[import-untyped]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
AssignmentMode = Literal["soft", "hard"]

_STATE_VECTOR_NAMES = (
    "detrended_acceleration_norm",
    "detrended_gyroscope_norm",
    "detrended_acceleration_derivative_norm",
    "detrended_gyroscope_derivative_norm",
    "normalized_acceleration_gyroscope_dot",
    "normalized_acceleration_gyroscope_cross_norm",
)


@dataclass(frozen=True)
class MicrostateFeatureSpec:
    """Fixed feature-family switches used by the primary method and frozen ablations."""

    transition_lags: tuple[int, ...] = (1, 2, 4, 8, 16)
    assignment: AssignmentMode = "soft"
    include_transitions: bool = True
    include_persistence_dwell: bool = True
    include_half_direction: bool = True

    def __post_init__(self) -> None:
        if (
            not self.transition_lags
            or any(lag < 1 for lag in self.transition_lags)
            or len(set(self.transition_lags)) != len(self.transition_lags)
        ):
            raise ValueError("transition lags must be unique positive integers")
        if self.assignment not in {"soft", "hard"}:
            raise ValueError("assignment must be 'soft' or 'hard'")


_PRIMARY_FEATURE_SPEC = MicrostateFeatureSpec()


@dataclass(frozen=True)
class MicrostateCodebook:
    """Training-partition-only robust scaler, K-means centres, and soft temperature."""

    centers: FloatArray
    location: FloatArray
    scale: FloatArray
    temperature: float
    cluster_count: int
    detrend_span_seconds: float
    sampling_rate_hz: float
    seed: int
    training_participants: tuple[str, ...]
    training_stationary_window_count: int

    def __post_init__(self) -> None:
        if self.centers.shape != (self.cluster_count, len(_STATE_VECTOR_NAMES)):
            raise ValueError("microstate centre dimensions do not match the codebook")
        if self.location.shape != (len(_STATE_VECTOR_NAMES),) or self.scale.shape != (
            len(_STATE_VECTOR_NAMES),
        ):
            raise ValueError("microstate scaler dimensions are invalid")
        if (
            self.cluster_count < 2
            or not np.isfinite(self.centers).all()
            or not np.isfinite(self.location).all()
            or not np.isfinite(self.scale).all()
            or np.any(self.scale <= 0)
            or not np.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ValueError("microstate codebook contains invalid fitted values")

    def audit_record(self) -> dict[str, Any]:
        """Return a JSON-safe description without embedding fitted arrays."""

        return {
            "cluster_count": self.cluster_count,
            "state_vector_count": len(_STATE_VECTOR_NAMES),
            "detrend_span_seconds": self.detrend_span_seconds,
            "sampling_rate_hz": self.sampling_rate_hz,
            "temperature": self.temperature,
            "seed": self.seed,
            "training_participants": list(self.training_participants),
            "training_stationary_window_count": self.training_stationary_window_count,
        }


def _validated_signals(signals: NDArray[np.floating[Any]]) -> FloatArray:
    values = np.asarray(signals, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] < 32 or values.shape[2] != 6:
        raise ValueError("microstate posture features require [window,time>=32,6] input")
    if values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("microstate posture input must be non-empty and finite")
    return values


def _odd_span(seconds: float, sampling_rate_hz: float, time_steps: int) -> int:
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("detrending span must be finite and positive")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling rate must be finite and positive")
    samples = max(3, round(seconds * sampling_rate_hz))
    if samples % 2 == 0:
        samples += 1
    largest_odd = time_steps if time_steps % 2 == 1 else time_steps - 1
    return min(samples, largest_odd)


def _rolling_spatial_median(triad: FloatArray, span: int) -> FloatArray:
    """Return an SO(3)-equivariant rolling geometric median for one sensor triad."""

    radius = span // 2
    padded = np.pad(triad, ((0, 0), (radius, radius), (0, 0)), mode="edge")
    result = np.empty_like(triad)
    for step in range(triad.shape[1]):
        neighbourhood = padded[:, step : step + span, :]
        estimate = neighbourhood.mean(axis=1)
        for _ in range(24):
            difference = neighbourhood - estimate[:, None, :]
            distance = np.linalg.norm(difference, axis=2)
            weight = 1.0 / np.maximum(distance, 1e-10)
            updated = np.sum(weight[:, :, None] * neighbourhood, axis=1) / weight.sum(
                axis=1, keepdims=True
            )
            estimate = updated
        result[:, step, :] = estimate
    return result


def extract_microstate_state_vectors(
    signals: NDArray[np.floating[Any]],
    *,
    sampling_rate_hz: float,
    detrend_span_seconds: float,
) -> FloatArray:
    """Map each timestamp to a rotation-invariant, locally detrended six-vector."""

    values = _validated_signals(signals)
    span = _odd_span(detrend_span_seconds, sampling_rate_hz, values.shape[1])
    trend = np.concatenate(
        (
            _rolling_spatial_median(values[:, :, :3], span),
            _rolling_spatial_median(values[:, :, 3:], span),
        ),
        axis=2,
    )
    detrended = values - trend
    acceleration = detrended[:, :, :3]
    gyroscope = detrended[:, :, 3:]
    acceleration_delta = np.diff(acceleration, axis=1, prepend=acceleration[:, :1, :])
    gyroscope_delta = np.diff(gyroscope, axis=1, prepend=gyroscope[:, :1, :])
    acceleration_norm = np.linalg.norm(acceleration, axis=2)
    gyroscope_norm = np.linalg.norm(gyroscope, axis=2)
    joint_scale = acceleration_norm * gyroscope_norm
    well_conditioned = joint_scale > 1e-8
    safe_scale = np.where(well_conditioned, joint_scale, 1.0)
    dot = np.where(well_conditioned, np.sum(acceleration * gyroscope, axis=2) / safe_scale, 0.0)
    cross = np.where(
        well_conditioned,
        np.linalg.norm(np.cross(acceleration, gyroscope), axis=2) / safe_scale,
        0.0,
    )
    vectors = np.stack(
        (
            acceleration_norm,
            gyroscope_norm,
            np.linalg.norm(acceleration_delta, axis=2) * sampling_rate_hz,
            np.linalg.norm(gyroscope_delta, axis=2) * sampling_rate_hz,
            dot,
            cross,
        ),
        axis=2,
    )
    if not np.isfinite(vectors).all():
        raise ValueError("microstate state-vector extraction produced non-finite values")
    return np.asarray(vectors, dtype=np.float64)


def fit_microstate_codebook(
    state_vectors: NDArray[np.floating[Any]],
    labels: NDArray[np.integer[Any]],
    participants: NDArray[np.str_],
    *,
    cluster_count: int,
    detrend_span_seconds: float,
    sampling_rate_hz: float,
    seed: int,
) -> MicrostateCodebook:
    """Fit a robustly scaled, participant-balanced codebook on stationary training rows."""

    vectors = np.asarray(state_vectors, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    people = np.asarray(participants, dtype=np.str_)
    if (
        vectors.ndim != 3
        or vectors.shape[2] != len(_STATE_VECTOR_NAMES)
        or y.shape != (vectors.shape[0],)
        or people.shape != y.shape
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("codebook fitting requires aligned finite state vectors and labels")
    if cluster_count not in {4, 6, 8}:
        raise ValueError("the locked microstate grid permits K in {4,6,8}")
    stationary = y != 0
    if not stationary.any() or set(y[stationary].tolist()) != {1, 2}:
        raise ValueError("codebook training requires both sitting and standing windows")
    stationary_vectors = vectors[stationary]
    stationary_people = people[stationary]
    flattened = stationary_vectors.reshape(-1, stationary_vectors.shape[2])
    location = np.median(flattened, axis=0)
    lower, upper = np.quantile(flattened, (0.25, 0.75), axis=0)
    scale = np.where(upper - lower > 1e-9, upper - lower, 1.0)
    standardized = (flattened - location) / scale
    point_people = np.repeat(stationary_people, stationary_vectors.shape[1])
    weights = np.empty(point_people.size, dtype=np.float64)
    unique_people = sorted(set(point_people.tolist()), key=int)
    for participant in unique_people:
        selected = point_people == participant
        weights[selected] = 1.0 / float(selected.sum())
    weights *= weights.size / weights.sum()
    model = KMeans(
        n_clusters=cluster_count,
        n_init=20,
        max_iter=300,
        algorithm="lloyd",
        random_state=seed,
    )
    model.fit(standardized, sample_weight=weights)
    distances = np.sum((standardized - model.cluster_centers_[model.labels_]) ** 2, axis=1)
    positive = distances[distances > np.finfo(np.float64).eps]
    temperature = float(np.median(positive)) if positive.size else 1.0
    return MicrostateCodebook(
        centers=np.asarray(model.cluster_centers_, dtype=np.float64),
        location=np.asarray(location, dtype=np.float64),
        scale=np.asarray(scale, dtype=np.float64),
        temperature=max(float(temperature), float(np.finfo(np.float64).eps)),
        cluster_count=cluster_count,
        detrend_span_seconds=float(detrend_span_seconds),
        sampling_rate_hz=float(sampling_rate_hz),
        seed=int(seed),
        training_participants=tuple(unique_people),
        training_stationary_window_count=int(stationary.sum()),
    )


def microstate_assignments(
    state_vectors: NDArray[np.floating[Any]],
    codebook: MicrostateCodebook,
    *,
    mode: AssignmentMode = "soft",
) -> FloatArray:
    """Return normalized soft or hard assignments for every window timestamp."""

    vectors = np.asarray(state_vectors, dtype=np.float64)
    if (
        vectors.ndim != 3
        or vectors.shape[2] != len(_STATE_VECTOR_NAMES)
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("microstate assignment requires finite [window,time,6] vectors")
    standardized = (vectors - codebook.location[None, None, :]) / codebook.scale[None, None, :]
    squared_distance = np.sum(
        (standardized[:, :, None, :] - codebook.centers[None, None, :, :]) ** 2,
        axis=3,
    )
    if mode == "hard":
        closest = np.argmin(squared_distance, axis=2)
        return np.asarray(
            np.eye(codebook.cluster_count, dtype=np.float64)[closest], dtype=np.float64
        )
    if mode != "soft":
        raise ValueError("assignment mode must be 'soft' or 'hard'")
    shifted = squared_distance - squared_distance.min(axis=2, keepdims=True)
    unnormalized = np.exp(-shifted / codebook.temperature)
    return np.asarray(unnormalized / unnormalized.sum(axis=2, keepdims=True), dtype=np.float64)


def _dwell_features(assignments: FloatArray, observed: BoolArray) -> tuple[FloatArray, FloatArray]:
    windows, time_steps, clusters = assignments.shape
    mean_dwell = np.zeros((windows, clusters), dtype=np.float64)
    max_dwell = np.zeros((windows, clusters), dtype=np.float64)
    states = np.argmax(assignments, axis=2)
    for window in range(windows):
        runs: list[list[int]] = [[] for _ in range(clusters)]
        active_state = -1
        active_length = 0
        for step in range(time_steps):
            state = int(states[window, step]) if observed[window, step] else -1
            if state == active_state and state >= 0:
                active_length += 1
                continue
            if active_state >= 0:
                runs[active_state].append(active_length)
            active_state = state
            active_length = 1 if state >= 0 else 0
        if active_state >= 0:
            runs[active_state].append(active_length)
        for cluster, lengths in enumerate(runs):
            if lengths:
                mean_dwell[window, cluster] = float(np.mean(lengths)) / time_steps
                max_dwell[window, cluster] = float(np.max(lengths)) / time_steps
    return mean_dwell, max_dwell


def extract_microstate_posture_features(
    state_vectors: NDArray[np.floating[Any]],
    codebook: MicrostateCodebook,
    *,
    feature_spec: MicrostateFeatureSpec = _PRIMARY_FEATURE_SPEC,
    validity_mask: NDArray[np.bool_] | None = None,
) -> FloatArray:
    """Summarize occupancies, transitions, persistence, dwell, direction, and coverage."""

    vectors = np.asarray(state_vectors, dtype=np.float64)
    if vectors.ndim != 3 or vectors.shape[2] != len(_STATE_VECTOR_NAMES):
        raise ValueError("posture graph requires [window,time,6] state vectors")
    windows, time_steps, _ = vectors.shape
    if validity_mask is None:
        channel_validity = np.ones((windows, time_steps, 6), dtype=np.bool_)
    else:
        channel_validity = np.asarray(validity_mask, dtype=np.bool_)
        if channel_validity.shape != (windows, time_steps, 6):
            raise ValueError("validity mask must align as [window,time,6]")
    observed = np.all(channel_validity, axis=2)
    assignment = microstate_assignments(vectors, codebook, mode=feature_spec.assignment)
    weighted = assignment * observed[:, :, None]
    observed_count = np.maximum(observed.sum(axis=1, keepdims=True), 1)
    outputs: list[FloatArray] = [weighted.sum(axis=1) / observed_count]

    if feature_spec.include_transitions:
        for lag in feature_spec.transition_lags:
            if lag >= time_steps:
                raise ValueError("transition lag must be shorter than the window")
            pair_validity = observed[:, :-lag] & observed[:, lag:]
            transition = np.einsum(
                "ntk,ntl,nt->nkl",
                assignment[:, :-lag],
                assignment[:, lag:],
                pair_validity,
            )
            denominator = np.maximum(transition.sum(axis=(1, 2), keepdims=True), 1.0)
            outputs.append((transition / denominator).reshape(windows, -1))

    if feature_spec.include_persistence_dwell:
        pair_validity = observed[:, :-1] & observed[:, 1:]
        persistence_numerator = np.sum(
            assignment[:, :-1] * assignment[:, 1:] * pair_validity[:, :, None], axis=1
        )
        persistence_denominator = np.maximum(
            np.sum(assignment[:, :-1] * pair_validity[:, :, None], axis=1), 1e-12
        )
        mean_dwell, max_dwell = _dwell_features(assignment, observed)
        outputs.extend((persistence_numerator / persistence_denominator, mean_dwell, max_dwell))

    if feature_spec.include_half_direction:
        midpoint = time_steps // 2
        first_observed = np.maximum(observed[:, :midpoint].sum(axis=1, keepdims=True), 1)
        second_observed = np.maximum(observed[:, midpoint:].sum(axis=1, keepdims=True), 1)
        first = weighted[:, :midpoint].sum(axis=1) / first_observed
        second = weighted[:, midpoint:].sum(axis=1) / second_observed
        outputs.append(first - second)

    outputs.append(channel_validity.mean(axis=1))
    outputs.append(observed.mean(axis=1, keepdims=True))
    features = np.concatenate(outputs, axis=1)
    expected = len(microstate_posture_feature_names(codebook.cluster_count, feature_spec))
    if features.shape != (windows, expected) or not np.isfinite(features).all():
        raise AssertionError("microstate posture feature contract changed")
    return np.asarray(features, dtype=np.float64)


@lru_cache(maxsize=64)
def microstate_posture_feature_names(
    cluster_count: int,
    feature_spec: MicrostateFeatureSpec = _PRIMARY_FEATURE_SPEC,
) -> tuple[str, ...]:
    """Return the immutable ordered graph-feature schema for a codebook size."""

    if cluster_count not in {4, 6, 8}:
        raise ValueError("the locked microstate grid permits K in {4,6,8}")
    names = [f"occupancy_{state}" for state in range(cluster_count)]
    if feature_spec.include_transitions:
        for lag in feature_spec.transition_lags:
            names.extend(
                f"transition_lag_{lag}_{source}_{target}"
                for source in range(cluster_count)
                for target in range(cluster_count)
            )
    if feature_spec.include_persistence_dwell:
        names.extend(f"persistence_{state}" for state in range(cluster_count))
        names.extend(f"mean_dwell_fraction_{state}" for state in range(cluster_count))
        names.extend(f"max_dwell_fraction_{state}" for state in range(cluster_count))
    if feature_spec.include_half_direction:
        names.extend(f"first_minus_second_occupancy_{state}" for state in range(cluster_count))
    names.extend(f"observed_channel_fraction_{channel}" for channel in range(6))
    names.append("fully_observed_timestep_fraction")
    return tuple(names)


def compose_mobility_posture_probabilities(
    mobility_probability: NDArray[np.floating[Any]],
    sitting_given_stationary_probability: NDArray[np.floating[Any]],
) -> FloatArray:
    """Compose P(mobility), P(sitting), and P(standing) from two binary heads."""

    mobility = np.asarray(mobility_probability, dtype=np.float64)
    sitting = np.asarray(sitting_given_stationary_probability, dtype=np.float64)
    if mobility.ndim != 1 or sitting.shape != mobility.shape or mobility.size == 0:
        raise ValueError("hierarchical probabilities must be aligned non-empty vectors")
    if (
        not np.isfinite(mobility).all()
        or not np.isfinite(sitting).all()
        or np.any((mobility < 0) | (mobility > 1))
        or np.any((sitting < 0) | (sitting > 1))
    ):
        raise ValueError("hierarchical probabilities must lie in [0,1]")
    stationary = 1.0 - mobility
    result = np.stack((mobility, stationary * sitting, stationary * (1.0 - sitting)), axis=1)
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-12):
        raise AssertionError("hierarchical probability composition is not normalized")
    return np.asarray(result, dtype=np.float64)


def microstate_state_vector_names() -> tuple[str, ...]:
    """Return the six rotation-invariant timestamp-vector names."""

    return _STATE_VECTOR_NAMES
