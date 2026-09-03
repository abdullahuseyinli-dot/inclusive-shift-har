"""Deterministic time, spectral, and structural descriptors for short IMU windows."""

from __future__ import annotations

from functools import lru_cache
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.preprocessing.v2 import invariant_features_numpy

FloatArray = NDArray[np.float64]

_STREAM_NAMES = (
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
    "gyroscope_x",
    "gyroscope_y",
    "gyroscope_z",
    "acceleration_norm",
    "gyroscope_norm",
    "acceleration_delta_norm",
    "gyroscope_delta_norm",
    "normalized_cross_sensor_dot",
    "cross_sensor_cross_norm",
    "acceleration_lag_dot",
    "gyroscope_lag_dot",
)

_PER_STREAM_NAMES = (
    "mean",
    "sd",
    "rms",
    "median",
    "q10",
    "q25",
    "q75",
    "q90",
    "minimum",
    "maximum",
    "mad",
    "mean_absolute_delta",
    "delta_sd",
    "skewness",
    "excess_kurtosis",
    "centered_zero_crossing_rate",
    "range",
    "spectral_fraction_000_005",
    "spectral_fraction_005_010",
    "spectral_fraction_010_020",
    "spectral_fraction_020_035",
    "spectral_fraction_035_050",
    "spectral_centroid",
    "spectral_entropy",
    "dominant_frequency",
    "spectral_low_high_ratio",
    "autocorrelation_lag_1",
    "autocorrelation_lag_4",
    "autocorrelation_lag_8",
    "autocorrelation_lag_16",
    "autocorrelation_lag_32",
)


@lru_cache(maxsize=1)
def spectral_shape_feature_names() -> tuple[str, ...]:
    names = [f"{stream}__{feature}" for stream in _STREAM_NAMES for feature in _PER_STREAM_NAMES]
    for first in range(6):
        for second in range(first + 1, 6):
            names.append(f"raw_correlation_{first}_{second}")
    return tuple(names)


def _stream_features(streams: FloatArray) -> FloatArray:
    mean = streams.mean(axis=2)
    centered = streams - mean[:, :, None]
    variance = np.mean(centered**2, axis=2)
    sd = np.sqrt(np.maximum(variance, 1e-12))
    rms = np.sqrt(np.mean(streams**2, axis=2))
    quantiles = np.quantile(streams, (0.1, 0.25, 0.5, 0.75, 0.9), axis=2)
    minimum = streams.min(axis=2)
    maximum = streams.max(axis=2)
    mad = np.median(np.abs(streams - quantiles[2][..., None]), axis=2)
    delta = np.diff(streams, axis=2)
    standardized = centered / sd[:, :, None]
    skewness = np.mean(standardized**3, axis=2)
    kurtosis = np.mean(standardized**4, axis=2) - 3.0
    crossing = np.mean(centered[:, :, :-1] * centered[:, :, 1:] < 0, axis=2)

    power = np.abs(np.fft.rfft(centered, axis=2)) ** 2
    frequencies = np.fft.rfftfreq(streams.shape[2])
    power[:, :, 0] = 0.0
    power_sum = np.maximum(power.sum(axis=2), 1e-12)
    band_edges = (0.0, 0.05, 0.10, 0.20, 0.35, 0.5000001)
    band_fractions = []
    for lower, upper in pairwise(band_edges):
        selected = (frequencies >= lower) & (frequencies < upper)
        band_fractions.append(power[:, :, selected].sum(axis=2) / power_sum)
    normalized_power = power / power_sum[:, :, None]
    spectral_centroid = np.sum(normalized_power * frequencies[None, None, :], axis=2)
    spectral_entropy = -np.sum(
        normalized_power * np.log(np.maximum(normalized_power, 1e-12)), axis=2
    ) / np.log(power.shape[2])
    dominant = frequencies[np.argmax(power, axis=2)]
    low_high_ratio = (band_fractions[0] + band_fractions[1]) / np.maximum(
        band_fractions[3] + band_fractions[4], 1e-12
    )

    denominator = np.maximum(np.sum(centered**2, axis=2), 1e-12)
    autocorrelations = []
    for lag in (1, 4, 8, 16, 32):
        autocorrelations.append(
            np.sum(centered[:, :, :-lag] * centered[:, :, lag:], axis=2) / denominator
        )
    ordered = (
        mean,
        sd,
        rms,
        quantiles[2],
        quantiles[0],
        quantiles[1],
        quantiles[3],
        quantiles[4],
        minimum,
        maximum,
        mad,
        np.mean(np.abs(delta), axis=2),
        delta.std(axis=2),
        skewness,
        kurtosis,
        crossing,
        maximum - minimum,
        *band_fractions,
        spectral_centroid,
        spectral_entropy,
        dominant,
        low_high_ratio,
        *autocorrelations,
    )
    if len(ordered) != len(_PER_STREAM_NAMES):
        raise AssertionError("per-stream spectral feature contract changed")
    return np.stack(ordered, axis=2).reshape(streams.shape[0], -1)


def extract_spectral_shape_features(signals: NDArray[np.floating[Any]]) -> FloatArray:
    """Return finite per-window features with no fitted or cross-window parameters."""

    values = np.asarray(signals, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (128, 6):
        raise ValueError("spectral-shape features require [window,128,6] input")
    if values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("spectral-shape input must be non-empty and finite")
    invariant = np.asarray(invariant_features_numpy(values), dtype=np.float64)
    streams = np.concatenate((values, invariant), axis=2).transpose(0, 2, 1)
    per_stream = _stream_features(streams)

    raw_centered = values - values.mean(axis=1, keepdims=True)
    covariance = np.einsum("ntc,ntd->ncd", raw_centered, raw_centered) / values.shape[1]
    raw_sd = np.sqrt(np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 1e-12))
    correlation = covariance / (raw_sd[:, :, None] * raw_sd[:, None, :])
    cross_features = np.stack(
        [correlation[:, first, second] for first in range(6) for second in range(first + 1, 6)],
        axis=1,
    )
    features = np.concatenate((per_stream, cross_features), axis=1)
    if features.shape[1] != len(spectral_shape_feature_names()):
        raise AssertionError("spectral-shape feature/name dimensions differ")
    if not np.isfinite(features).all():
        raise ValueError("spectral-shape extraction produced non-finite features")
    return np.asarray(features, dtype=np.float64)
