"""Deterministic multi-scale geometry and spectrum map for short triaxial IMU windows."""

from __future__ import annotations

from functools import lru_cache
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.preprocessing.v2 import invariant_features_numpy

FloatArray = NDArray[np.float64]

_BASE_STREAM_NAMES = (
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
_CANONICAL_STREAM_NAMES = tuple(f"canonical_{name}" for name in _BASE_STREAM_NAMES[:6])
_STREAM_NAMES = _BASE_STREAM_NAMES + _CANONICAL_STREAM_NAMES
_STREAM_FEATURE_NAMES = (
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
    "physical_power_000_050_hz",
    "physical_power_050_100_hz",
    "physical_power_100_200_hz",
    "physical_power_200_400_hz",
    "physical_power_400_nyquist_hz",
    "normalized_power_000_005",
    "normalized_power_005_010",
    "normalized_power_010_020",
    "normalized_power_020_035",
    "normalized_power_035_050",
    "spectral_centroid_hz",
    "spectral_centroid_fraction",
    "spectral_entropy",
    "dominant_frequency_hz",
    "dominant_frequency_fraction",
    "spectral_low_high_ratio",
    "autocorrelation_005_seconds",
    "autocorrelation_010_seconds",
    "autocorrelation_025_seconds",
    "autocorrelation_050_seconds",
    "autocorrelation_100_seconds",
    "haar_detail_energy_level_1",
    "haar_detail_energy_level_2",
    "haar_detail_energy_level_3",
    "haar_detail_energy_level_4",
    "haar_approximation_energy_level_4",
)
_LOCAL_FEATURE_NAMES = (
    "mean",
    "sd",
    "rms",
    "mean_absolute_delta",
    "centered_zero_crossing_rate",
)
_SEGMENT_NAMES = (
    "global",
    "half_0",
    "half_1",
    "quarter_0",
    "quarter_1",
    "quarter_2",
    "quarter_3",
)
_GEOMETRY_FEATURE_NAMES = (
    "acceleration_covariance_eigen_fraction_0",
    "acceleration_covariance_eigen_fraction_1",
    "acceleration_covariance_eigen_fraction_2",
    "gyroscope_covariance_eigen_fraction_0",
    "gyroscope_covariance_eigen_fraction_1",
    "gyroscope_covariance_eigen_fraction_2",
    "cross_covariance_singular_fraction_0",
    "cross_covariance_singular_fraction_1",
    "cross_covariance_singular_fraction_2",
    "acceleration_eigen_gap_01",
    "acceleration_eigen_gap_12",
    "gyroscope_eigen_gap_01",
    "gyroscope_eigen_gap_12",
    "normalized_cross_covariance_frobenius",
    "normalized_cross_covariance_determinant",
)


@lru_cache(maxsize=1)
def geometric_spectral_pyramid_feature_names() -> tuple[str, ...]:
    """Return the immutable ordered feature schema."""

    names = [
        f"{stream}__{feature}" for stream in _STREAM_NAMES for feature in _STREAM_FEATURE_NAMES
    ]
    for stream in _BASE_STREAM_NAMES:
        for scale, pieces in (("halves", 2), ("quarters", 4)):
            for piece in range(pieces):
                for feature in _LOCAL_FEATURE_NAMES:
                    names.append(f"{scale}_{piece}__{stream}__{feature}")
    for first in range(6):
        for second in range(first + 1, 6):
            names.append(f"global_raw_correlation_{first}_{second}")
    for segment in _SEGMENT_NAMES:
        names.extend(f"{segment}__{feature}" for feature in _GEOMETRY_FEATURE_NAMES)
    return tuple(names)


def _canonical_triaxial_view(values: FloatArray) -> FloatArray:
    acceleration = values[:, :, :3]
    centered = acceleration - acceleration.mean(axis=1, keepdims=True)
    covariance = np.einsum("ntc,ntd->ncd", centered, centered) / values.shape[1]
    _, eigenvectors = np.linalg.eigh(covariance)
    basis = eigenvectors[:, :, ::-1]
    determinant = np.linalg.det(basis)
    basis[:, :, 2] *= np.where(determinant < 0, -1.0, 1.0)[:, None]
    canonical = np.einsum("ntc,ncd->ntd", values[:, :, :3], basis)
    canonical_gyro = np.einsum("ntc,ncd->ntd", values[:, :, 3:], basis)
    centered_canonical = canonical - canonical.mean(axis=1, keepdims=True)
    third_moment = np.mean(centered_canonical**3, axis=1)
    largest_index = np.abs(centered_canonical).argmax(axis=1)
    fallback = np.take_along_axis(
        centered_canonical,
        largest_index[:, None, :],
        axis=1,
    )[:, 0, :]
    direction = np.where(np.abs(third_moment) > 1e-12, third_moment, fallback)
    signs = np.where(direction < 0, -1.0, 1.0)
    return np.concatenate(
        (canonical * signs[:, None, :], canonical_gyro * signs[:, None, :]), axis=2
    )


def _haar_energy_features(streams: FloatArray) -> list[FloatArray]:
    approximation = streams - streams.mean(axis=2, keepdims=True)
    total_energy = np.maximum(np.sum(approximation**2, axis=2), 1e-12)
    details: list[FloatArray] = []
    for _ in range(4):
        usable = approximation.shape[2] - approximation.shape[2] % 2
        even = approximation[:, :, :usable:2]
        odd = approximation[:, :, 1:usable:2]
        detail = (even - odd) / np.sqrt(2.0)
        approximation = (even + odd) / np.sqrt(2.0)
        details.append(np.sum(detail**2, axis=2) / total_energy)
    details.append(np.sum(approximation**2, axis=2) / total_energy)
    return details


def _stream_features(streams: FloatArray, *, sampling_rate_hz: float) -> FloatArray:
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
    crossing = np.mean(centered[:, :, :-1] * centered[:, :, 1:] < 0, axis=2)

    power = np.abs(np.fft.rfft(centered, axis=2)) ** 2
    frequencies = np.fft.rfftfreq(streams.shape[2], d=1.0 / sampling_rate_hz)
    normalized_frequencies = frequencies / sampling_rate_hz
    power[:, :, 0] = 0.0
    power_sum = np.maximum(power.sum(axis=2), 1e-12)
    physical_bands: list[FloatArray] = []
    for lower, upper in ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, np.inf)):
        selected = (frequencies >= lower) & (frequencies < upper)
        physical_bands.append(power[:, :, selected].sum(axis=2) / power_sum)
    normalized_bands: list[FloatArray] = []
    for lower, upper in pairwise((0.0, 0.05, 0.10, 0.20, 0.35, 0.5000001)):
        selected = (normalized_frequencies >= lower) & (normalized_frequencies < upper)
        normalized_bands.append(power[:, :, selected].sum(axis=2) / power_sum)
    normalized_power = power / power_sum[:, :, None]
    centroid_hz = np.sum(normalized_power * frequencies[None, None, :], axis=2)
    entropy = -np.sum(
        normalized_power * np.log(np.maximum(normalized_power, 1e-12)), axis=2
    ) / np.log(power.shape[2])
    dominant_hz = frequencies[np.argmax(power, axis=2)]
    low_high_ratio = (physical_bands[0] + physical_bands[1]) / np.maximum(
        physical_bands[3] + physical_bands[4], 1e-12
    )
    denominator = np.maximum(np.sum(centered**2, axis=2), 1e-12)
    autocorrelations: list[FloatArray] = []
    for seconds in (0.05, 0.10, 0.25, 0.50, 1.00):
        lag = min(max(1, round(seconds * sampling_rate_hz)), streams.shape[2] - 1)
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
        np.mean(standardized**3, axis=2),
        np.mean(standardized**4, axis=2) - 3.0,
        crossing,
        maximum - minimum,
        *physical_bands,
        *normalized_bands,
        centroid_hz,
        centroid_hz / sampling_rate_hz,
        entropy,
        dominant_hz,
        dominant_hz / sampling_rate_hz,
        low_high_ratio,
        *autocorrelations,
        *_haar_energy_features(streams),
    )
    if len(ordered) != len(_STREAM_FEATURE_NAMES):
        raise AssertionError("geometric spectral per-stream contract changed")
    return np.stack(ordered, axis=2).reshape(streams.shape[0], -1)


def _local_features(streams: FloatArray) -> FloatArray:
    outputs: list[FloatArray] = []
    for pieces in (2, 4):
        for segment in np.array_split(streams, pieces, axis=2):
            mean = segment.mean(axis=2)
            centered = segment - mean[:, :, None]
            delta = np.diff(segment, axis=2)
            outputs.extend(
                (
                    mean,
                    np.sqrt(np.mean(centered**2, axis=2)),
                    np.sqrt(np.mean(segment**2, axis=2)),
                    np.mean(np.abs(delta), axis=2),
                    np.mean(centered[:, :, :-1] * centered[:, :, 1:] < 0, axis=2),
                )
            )
    return np.stack(outputs, axis=2).reshape(streams.shape[0], -1)


def _geometry_features(values: FloatArray) -> FloatArray:
    segments = [values]
    segments.extend(np.array_split(values, 2, axis=1))
    segments.extend(np.array_split(values, 4, axis=1))
    outputs: list[FloatArray] = []
    for segment in segments:
        acceleration = segment[:, :, :3] - segment[:, :, :3].mean(axis=1, keepdims=True)
        gyroscope = segment[:, :, 3:] - segment[:, :, 3:].mean(axis=1, keepdims=True)
        acc_cov = np.einsum("ntc,ntd->ncd", acceleration, acceleration) / segment.shape[1]
        gyro_cov = np.einsum("ntc,ntd->ncd", gyroscope, gyroscope) / segment.shape[1]
        cross_cov = np.einsum("ntc,ntd->ncd", acceleration, gyroscope) / segment.shape[1]
        acc_eigen = np.maximum(np.linalg.eigvalsh(acc_cov)[:, ::-1], 0.0)
        gyro_eigen = np.maximum(np.linalg.eigvalsh(gyro_cov)[:, ::-1], 0.0)
        acc_fraction = acc_eigen / np.maximum(acc_eigen.sum(axis=1, keepdims=True), 1e-12)
        gyro_fraction = gyro_eigen / np.maximum(gyro_eigen.sum(axis=1, keepdims=True), 1e-12)
        singular = np.linalg.svd(cross_cov, compute_uv=False)
        cross_scale = np.maximum(np.sqrt(acc_eigen.sum(axis=1) * gyro_eigen.sum(axis=1)), 1e-12)
        singular_fraction = singular / cross_scale[:, None]
        determinant = np.linalg.det(cross_cov) / np.maximum(cross_scale**3, 1e-12)
        outputs.append(
            np.concatenate(
                (
                    acc_fraction,
                    gyro_fraction,
                    singular_fraction,
                    acc_fraction[:, :2] - acc_fraction[:, 1:],
                    gyro_fraction[:, :2] - gyro_fraction[:, 1:],
                    np.linalg.norm(cross_cov, axis=(1, 2))[:, None] / cross_scale[:, None],
                    determinant[:, None],
                ),
                axis=1,
            )
        )
    return np.concatenate(outputs, axis=1)


def extract_geometric_spectral_pyramid_features(
    signals: NDArray[np.floating[Any]],
    *,
    sampling_rate_hz: float,
) -> FloatArray:
    """Map finite six-axis windows to a fixed multi-view feature vector."""

    values = np.asarray(signals, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 6 or values.shape[1] < 32:
        raise ValueError("geometric spectral features require [window,time>=32,6] input")
    if values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("geometric spectral input must be non-empty and finite")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling rate must be finite and positive")
    invariant = np.asarray(invariant_features_numpy(values), dtype=np.float64)
    base_streams = np.concatenate((values, invariant), axis=2).transpose(0, 2, 1)
    canonical_streams = _canonical_triaxial_view(values).transpose(0, 2, 1)
    global_features = _stream_features(
        np.concatenate((base_streams, canonical_streams), axis=1),
        sampling_rate_hz=sampling_rate_hz,
    )
    local_features = _local_features(base_streams)
    centered = values - values.mean(axis=1, keepdims=True)
    covariance = np.einsum("ntc,ntd->ncd", centered, centered) / values.shape[1]
    raw_sd = np.sqrt(np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 1e-12))
    correlation = covariance / (raw_sd[:, :, None] * raw_sd[:, None, :])
    correlations = np.stack(
        [correlation[:, first, second] for first in range(6) for second in range(first + 1, 6)],
        axis=1,
    )
    features = np.concatenate(
        (global_features, local_features, correlations, _geometry_features(values)), axis=1
    )
    if features.shape[1] != len(geometric_spectral_pyramid_feature_names()):
        raise AssertionError("geometric spectral feature/name dimensions differ")
    if not np.isfinite(features).all():
        raise ValueError("geometric spectral extraction produced non-finite features")
    return np.asarray(features, dtype=np.float64)
