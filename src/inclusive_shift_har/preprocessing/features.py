"""Deterministic engineered inertial features for non-neural baselines."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EngineeredFeatureBatch:
    values: NDArray[np.float64]
    names: tuple[str, ...]


def _spectral_features(channel: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
    centered = channel - channel.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    if power.shape[1] > 1:
        power[:, 0] = 0.0
    total = power.sum(axis=1, keepdims=True)
    normalized = np.divide(power, total, out=np.zeros_like(power), where=total > 0)
    dominant = np.argmax(power, axis=1).astype(np.float64)
    dominant /= max(power.shape[1] - 1, 1)
    entropy = -np.sum(normalized * np.log(np.clip(normalized, 1e-15, 1.0)), axis=1)
    entropy /= np.log(max(power.shape[1], 2))
    return dominant, entropy


def extract_engineered_features(
    windows: NDArray[np.floating],
    *,
    channel_names: tuple[str, ...],
) -> EngineeredFeatureBatch:
    """Extract time/frequency summaries, vector norms, and within-modality correlations."""

    array = np.asarray(windows, dtype=np.float64)
    if array.ndim != 3 or array.shape[1] != 128 or array.shape[2] != len(channel_names):
        raise ValueError("feature extraction expects [window,128,channel] and aligned names")
    if array.shape[0] == 0 or not np.isfinite(array).all():
        raise ValueError("feature extraction input must be non-empty and finite")
    if len(channel_names) != 6 or len(set(channel_names)) != 6:
        raise ValueError("the primary feature schema requires six unique inertial channels")

    feature_columns: list[NDArray[np.float64]] = []
    feature_names: list[str] = []
    for channel_index, channel_name in enumerate(channel_names):
        channel = array[:, :, channel_index]
        difference = np.diff(channel, axis=1)
        centered = channel - channel.mean(axis=1, keepdims=True)
        zero_crossing = np.mean(centered[:, :-1] * centered[:, 1:] < 0, axis=1)
        dominant, entropy = _spectral_features(channel)
        summaries = (
            ("mean", channel.mean(axis=1)),
            ("std", channel.std(axis=1, ddof=0)),
            ("min", channel.min(axis=1)),
            ("max", channel.max(axis=1)),
            ("median", np.median(channel, axis=1)),
            ("iqr", np.quantile(channel, 0.75, axis=1) - np.quantile(channel, 0.25, axis=1)),
            ("rms", np.sqrt(np.mean(channel**2, axis=1))),
            ("mean_abs_difference", np.mean(np.abs(difference), axis=1)),
            ("zero_crossing_rate", zero_crossing),
            ("dominant_frequency_bin_fraction", dominant),
            ("spectral_entropy", entropy),
        )
        for statistic, values in summaries:
            feature_columns.append(np.asarray(values, dtype=np.float64))
            feature_names.append(f"{channel_name}__{statistic}")

    for modality_name, indices in (("accelerometer", (0, 1, 2)), ("gyroscope", (3, 4, 5))):
        vector_norm = np.linalg.vector_norm(array[:, :, indices], axis=2)
        norm_summaries = (
            ("mean", vector_norm.mean(axis=1)),
            ("std", vector_norm.std(axis=1, ddof=0)),
            ("rms", np.sqrt(np.mean(vector_norm**2, axis=1))),
            ("p95", np.quantile(vector_norm, 0.95, axis=1)),
        )
        for statistic, values in norm_summaries:
            feature_columns.append(np.asarray(values, dtype=np.float64))
            feature_names.append(f"{modality_name}_norm__{statistic}")
        for left, right in combinations(indices, 2):
            left_centered = array[:, :, left] - array[:, :, left].mean(axis=1, keepdims=True)
            right_centered = array[:, :, right] - array[:, :, right].mean(axis=1, keepdims=True)
            numerator = np.mean(left_centered * right_centered, axis=1)
            denominator = np.sqrt(
                np.mean(left_centered**2, axis=1) * np.mean(right_centered**2, axis=1)
            )
            correlation = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > 1e-15,
            )
            feature_columns.append(correlation)
            feature_names.append(f"{channel_names[left]}__{channel_names[right]}__correlation")

    values = np.stack(feature_columns, axis=1)
    if not np.isfinite(values).all():
        raise AssertionError("engineered feature implementation produced non-finite values")
    return EngineeredFeatureBatch(values=values, names=tuple(feature_names))
