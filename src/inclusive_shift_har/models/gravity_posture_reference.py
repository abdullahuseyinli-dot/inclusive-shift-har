"""Gravity-referenced posture features for the nine-channel sensor-sufficiency lane."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_DERIVED_NAMES = (
    "gravity_unit_x",
    "gravity_unit_y",
    "gravity_unit_z",
    "gravity_abs_x",
    "gravity_abs_y",
    "gravity_abs_z",
    "gravity_abs_sorted_low",
    "gravity_abs_sorted_mid",
    "gravity_abs_sorted_high",
    "user_acceleration_parallel_gravity",
    "user_acceleration_perpendicular_norm",
    "gyroscope_parallel_gravity",
    "gyroscope_perpendicular_norm",
    "total_acceleration_parallel_gravity",
    "total_acceleration_perpendicular_norm",
    "gravity_direction_angular_speed",
    "user_acceleration_norm",
    "gyroscope_norm",
    "normalized_acceleration_gyroscope_dot",
)
_SUMMARY_NAMES = (
    "mean",
    "std",
    "minimum",
    "maximum",
    "quantile_10",
    "quantile_25",
    "median",
    "quantile_75",
    "quantile_90",
    "iqr",
    "rms",
    "mean_absolute_difference",
    "linear_slope_per_second",
    "first_half_minus_second_half",
    "last_minus_first",
)


def _validate_inputs(
    primary_signals: NDArray[np.floating[Any]],
    gravity: NDArray[np.floating[Any]],
    sampling_rate_hz: float,
) -> tuple[FloatArray, FloatArray]:
    primary = np.asarray(primary_signals, dtype=np.float64)
    gravity_values = np.asarray(gravity, dtype=np.float64)
    if primary.ndim != 3 or primary.shape[1] < 32 or primary.shape[2] != 6:
        raise ValueError("gravity posture features require [window,time>=32,6] primary input")
    if gravity_values.shape != (*primary.shape[:2], 3):
        raise ValueError("gravity must align with primary input as [window,time,3]")
    if (
        primary.shape[0] == 0
        or not np.isfinite(primary).all()
        or not np.isfinite(gravity_values).all()
    ):
        raise ValueError("gravity posture inputs must be non-empty and finite")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling rate must be finite and positive")
    return primary, gravity_values


def gravity_reference_time_series(
    primary_signals: NDArray[np.floating[Any]],
    gravity: NDArray[np.floating[Any]],
    *,
    sampling_rate_hz: float,
) -> FloatArray:
    """Construct physical projections and device-orientation gravity coordinates."""

    primary, gravity_values = _validate_inputs(primary_signals, gravity, sampling_rate_hz)
    gravity_norm = np.linalg.norm(gravity_values, axis=2, keepdims=True)
    unit = gravity_values / np.maximum(gravity_norm, 1e-10)
    acceleration = primary[:, :, :3]
    gyroscope = primary[:, :, 3:]
    acceleration_parallel = np.sum(acceleration * unit, axis=2)
    gyroscope_parallel = np.sum(gyroscope * unit, axis=2)
    acceleration_perpendicular = np.linalg.norm(
        acceleration - acceleration_parallel[:, :, None] * unit, axis=2
    )
    gyroscope_perpendicular = np.linalg.norm(
        gyroscope - gyroscope_parallel[:, :, None] * unit, axis=2
    )
    total_acceleration = acceleration + gravity_values
    total_parallel = np.sum(total_acceleration * unit, axis=2)
    total_perpendicular = np.linalg.norm(
        total_acceleration - total_parallel[:, :, None] * unit, axis=2
    )
    previous = np.concatenate((unit[:, :1], unit[:, :-1]), axis=1)
    angular_speed = (
        np.arccos(np.clip(np.sum(previous * unit, axis=2), -1.0, 1.0)) * sampling_rate_hz
    )
    acceleration_norm = np.linalg.norm(acceleration, axis=2)
    gyroscope_norm = np.linalg.norm(gyroscope, axis=2)
    joint_norm = acceleration_norm * gyroscope_norm
    dot = np.where(
        joint_norm > 1e-10,
        np.sum(acceleration * gyroscope, axis=2) / np.maximum(joint_norm, 1e-10),
        0.0,
    )
    derived = np.concatenate(
        (
            unit,
            np.abs(unit),
            np.sort(np.abs(unit), axis=2),
            acceleration_parallel[:, :, None],
            acceleration_perpendicular[:, :, None],
            gyroscope_parallel[:, :, None],
            gyroscope_perpendicular[:, :, None],
            total_parallel[:, :, None],
            total_perpendicular[:, :, None],
            angular_speed[:, :, None],
            acceleration_norm[:, :, None],
            gyroscope_norm[:, :, None],
            dot[:, :, None],
        ),
        axis=2,
    )
    if derived.shape[2] != len(_DERIVED_NAMES) or not np.isfinite(derived).all():
        raise AssertionError("gravity reference time-series contract changed")
    return np.asarray(derived, dtype=np.float64)


def extract_gravity_posture_reference_features(
    primary_signals: NDArray[np.floating[Any]],
    gravity: NDArray[np.floating[Any]],
    *,
    sampling_rate_hz: float,
) -> FloatArray:
    """Summarize the gravity-reference series without participant-dependent fitting."""

    series = gravity_reference_time_series(
        primary_signals, gravity, sampling_rate_hz=sampling_rate_hz
    )
    windows, time_steps, channels = series.shape
    quantiles = np.quantile(series, (0.10, 0.25, 0.50, 0.75, 0.90), axis=1)
    time = np.arange(time_steps, dtype=np.float64) / sampling_rate_hz
    centered_time = time - time.mean()
    slope_denominator = float(np.sum(centered_time**2))
    centered_series = series - series.mean(axis=1, keepdims=True)
    slopes = np.einsum("t,ntc->nc", centered_time, centered_series) / slope_denominator
    midpoint = time_steps // 2
    summaries = (
        series.mean(axis=1),
        series.std(axis=1),
        series.min(axis=1),
        series.max(axis=1),
        quantiles[0],
        quantiles[1],
        quantiles[2],
        quantiles[3],
        quantiles[4],
        quantiles[3] - quantiles[1],
        np.sqrt(np.mean(series**2, axis=1)),
        np.mean(np.abs(np.diff(series, axis=1)), axis=1),
        slopes,
        series[:, :midpoint].mean(axis=1) - series[:, midpoint:].mean(axis=1),
        series[:, -1] - series[:, 0],
    )
    features = np.stack(summaries, axis=2).reshape(windows, channels * len(_SUMMARY_NAMES))
    expected = len(gravity_posture_reference_feature_names())
    if features.shape != (windows, expected) or not np.isfinite(features).all():
        raise AssertionError("gravity posture feature contract changed")
    return np.asarray(features, dtype=np.float64)


@lru_cache(maxsize=1)
def gravity_posture_reference_feature_names() -> tuple[str, ...]:
    """Return the immutable feature order."""

    return tuple(
        f"{derived}__{summary}" for derived in _DERIVED_NAMES for summary in _SUMMARY_NAMES
    )


def gravity_reference_time_series_names() -> tuple[str, ...]:
    """Return the derived timestamp-channel order."""

    return _DERIVED_NAMES
