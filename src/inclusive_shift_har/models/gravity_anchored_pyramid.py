"""Gravity-anchored dual-view extension of the Geometric Spectral Pyramid."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.models.geometric_spectral_pyramid import (
    extract_geometric_spectral_pyramid_features,
    geometric_spectral_pyramid_feature_names,
)

FloatArray = NDArray[np.float64]


def gravity_anchored_feature_views(
    primary_signals: NDArray[np.floating[Any]],
    gravity: NDArray[np.floating[Any]],
    *,
    sampling_rate_hz: float,
) -> dict[str, FloatArray]:
    """Return dynamic, reconstructed-total, and concatenated deterministic views."""

    primary = np.asarray(primary_signals, dtype=np.float64)
    gravity_values = np.asarray(gravity, dtype=np.float64)
    if primary.ndim != 3 or primary.shape[1] < 32 or primary.shape[2] != 6:
        raise ValueError("primary signals must have shape [window,time,6] with time >= 32")
    if gravity_values.shape != (*primary.shape[:2], 3):
        raise ValueError("gravity must align as [window,time,3]")
    if not np.isfinite(primary).all() or not np.isfinite(gravity_values).all():
        raise ValueError("gravity-anchored feature inputs must be finite")
    total = primary.copy()
    total[:, :, :3] += gravity_values
    dynamic_features = extract_geometric_spectral_pyramid_features(
        primary, sampling_rate_hz=sampling_rate_hz
    )
    total_features = extract_geometric_spectral_pyramid_features(
        total, sampling_rate_hz=sampling_rate_hz
    )
    return {
        "dynamic": dynamic_features,
        "total": total_features,
        "dual": np.concatenate((dynamic_features, total_features), axis=1),
    }


@lru_cache(maxsize=3)
def gravity_anchored_feature_names(view: str) -> tuple[str, ...]:
    """Return the immutable feature names for one declared view."""

    base = geometric_spectral_pyramid_feature_names()
    if view == "dynamic":
        return tuple(f"dynamic__{item}" for item in base)
    if view == "total":
        return tuple(f"total__{item}" for item in base)
    if view == "dual":
        return gravity_anchored_feature_names("dynamic") + gravity_anchored_feature_names("total")
    raise ValueError(f"unknown gravity-anchored feature view {view!r}")
