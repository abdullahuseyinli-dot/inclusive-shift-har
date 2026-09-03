"""Deterministic noise- and drift-aware views for the Geometric Spectral Pyramid."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter  # type: ignore[import-untyped]

from inclusive_shift_har.models.geometric_spectral_pyramid import (
    extract_geometric_spectral_pyramid_features,
    geometric_spectral_pyramid_feature_names,
)

FloatArray = NDArray[np.float64]


def _odd_span(samples: int, maximum: int, *, minimum: int = 5) -> int:
    span = max(minimum, samples)
    if span % 2 == 0:
        span += 1
    largest_odd = maximum if maximum % 2 == 1 else maximum - 1
    return min(span, largest_odd)


def robust_multiscale_signal_views(
    signals: NDArray[np.floating[Any]], *, sampling_rate_hz: float
) -> dict[str, FloatArray]:
    """Return original, short-scale denoised, and long-trend residual signal views."""

    values = np.asarray(signals, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 6 or values.shape[1] < 32:
        raise ValueError("robust multiscale views require [window,time>=32,6] input")
    if values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("robust multiscale input must be non-empty and finite")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling rate must be finite and positive")
    short_span = _odd_span(round(0.20 * sampling_rate_hz), values.shape[1])
    trend_span = _odd_span(round(1.00 * sampling_rate_hz), values.shape[1])
    denoised = savgol_filter(values, window_length=short_span, polyorder=2, axis=1, mode="interp")
    trend = savgol_filter(values, window_length=trend_span, polyorder=2, axis=1, mode="interp")
    residual = values - trend
    return {
        "base": values,
        "denoised": np.asarray(denoised, dtype=np.float64),
        "residual": np.asarray(residual, dtype=np.float64),
    }


def robust_multiscale_feature_views(
    signals: NDArray[np.floating[Any]], *, sampling_rate_hz: float
) -> dict[str, FloatArray]:
    """Extract fixed Geometric Spectral Pyramid features from every signal view."""

    signal_views = robust_multiscale_signal_views(signals, sampling_rate_hz=sampling_rate_hz)
    base_views = {
        name: extract_geometric_spectral_pyramid_features(values, sampling_rate_hz=sampling_rate_hz)
        for name, values in signal_views.items()
    }
    return {
        **base_views,
        "base_denoised": np.concatenate((base_views["base"], base_views["denoised"]), axis=1),
        "tri_view": np.concatenate(
            (base_views["base"], base_views["denoised"], base_views["residual"]), axis=1
        ),
    }


@lru_cache(maxsize=5)
def robust_multiscale_feature_names(view: str) -> tuple[str, ...]:
    """Return immutable names for one fixed multiscale feature view."""

    base = geometric_spectral_pyramid_feature_names()
    components = {
        "base": ("base",),
        "denoised": ("denoised",),
        "residual": ("residual",),
        "base_denoised": ("base", "denoised"),
        "tri_view": ("base", "denoised", "residual"),
    }
    if view not in components:
        raise ValueError(f"unknown robust multiscale feature view {view!r}")
    return tuple(f"{component}__{feature}" for component in components[view] for feature in base)
