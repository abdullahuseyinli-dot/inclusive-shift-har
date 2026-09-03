"""Source-fitted physical and invariant preprocessing for v2 research models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.preprocessing.normalization import ChannelStandardizer


def invariant_features_numpy(signals: NDArray[np.floating[Any]]) -> NDArray[np.float32]:
    """Return eight rotation-invariant time-series features from native six-channel IMU data."""

    values = np.asarray(signals, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 6 or values.shape[1] < 2:
        raise ValueError("invariant features require [window,time,6] input with time >= 2")
    if not np.isfinite(values).all():
        raise ValueError("invariant feature input must be finite")
    acceleration = values[:, :, :3]
    gyroscope = values[:, :, 3:]
    acceleration_norm = np.linalg.norm(acceleration, axis=2)
    gyroscope_norm = np.linalg.norm(gyroscope, axis=2)
    acceleration_delta = np.diff(acceleration, axis=1, prepend=acceleration[:, :1])
    gyroscope_delta = np.diff(gyroscope, axis=1, prepend=gyroscope[:, :1])
    acceleration_delta_norm = np.linalg.norm(acceleration_delta, axis=2)
    gyroscope_delta_norm = np.linalg.norm(gyroscope_delta, axis=2)
    denominator = np.maximum(acceleration_norm * gyroscope_norm, 1e-8)
    normalized_cross_sensor_dot = np.sum(acceleration * gyroscope, axis=2) / denominator
    cross_sensor_cross_norm = np.linalg.norm(np.cross(acceleration, gyroscope), axis=2)
    previous_acceleration = np.concatenate((acceleration[:, :1], acceleration[:, :-1]), axis=1)
    previous_gyroscope = np.concatenate((gyroscope[:, :1], gyroscope[:, :-1]), axis=1)
    acceleration_lag_dot = np.sum(acceleration * previous_acceleration, axis=2)
    gyroscope_lag_dot = np.sum(gyroscope * previous_gyroscope, axis=2)
    return np.asarray(
        np.stack(
            (
                acceleration_norm,
                gyroscope_norm,
                acceleration_delta_norm,
                gyroscope_delta_norm,
                normalized_cross_sensor_dot,
                cross_sensor_cross_norm,
                acceleration_lag_dot,
                gyroscope_lag_dot,
            ),
            axis=2,
        ),
        dtype=np.float32,
    )


@dataclass(frozen=True)
class V2PhysicalPreprocessor:
    """Training-partition-only moments embedded into a FuSE-ReFrame checkpoint."""

    raw: ChannelStandardizer
    invariant_mean: NDArray[np.float64]
    invariant_scale: NDArray[np.float64]
    clipping_thresholds: NDArray[np.float64]
    clipping_quantile: float

    @classmethod
    def fit(
        cls,
        signals: NDArray[np.floating[Any]],
        participant_ids: list[str],
        *,
        declared_training_participants: set[str],
        split_manifest_sha256: str,
        channel_names: tuple[str, ...],
        clipping_quantile: float = 0.999,
    ) -> V2PhysicalPreprocessor:
        """Fit all moments on the declared source-training partition only."""

        values = np.asarray(signals, dtype=np.float32)
        if not 0.95 <= clipping_quantile < 1.0:
            raise ValueError("clipping_quantile must lie in [0.95, 1.0)")
        raw = ChannelStandardizer.fit(
            values,
            participant_ids,
            declared_training_participants=declared_training_participants,
            split_manifest_sha256=split_manifest_sha256,
            channel_names=channel_names,
        )
        invariant = np.asarray(invariant_features_numpy(values), dtype=np.float64)
        invariant_mean = invariant.mean(axis=(0, 1))
        invariant_scale = invariant.std(axis=(0, 1))
        invariant_scale = np.where(invariant_scale < 1e-8, 1.0, invariant_scale)
        clipping_thresholds = np.quantile(
            np.abs(np.asarray(values, dtype=np.float64)),
            clipping_quantile,
            axis=(0, 1),
        )
        clipping_thresholds = np.maximum(clipping_thresholds, 1e-6)
        for array in (invariant_mean, invariant_scale, clipping_thresholds):
            array.setflags(write=False)
        return cls(
            raw=raw,
            invariant_mean=invariant_mean,
            invariant_scale=invariant_scale,
            clipping_thresholds=clipping_thresholds,
            clipping_quantile=clipping_quantile,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable, provenance-preserving record."""

        return {
            "schema_version": "1.0.0",
            "method": "native_physical_plus_rotation_invariant_source_standardization",
            "raw": self.raw.to_dict(),
            "invariant_feature_order": [
                "acceleration_norm",
                "gyroscope_norm",
                "acceleration_delta_norm",
                "gyroscope_delta_norm",
                "normalized_cross_sensor_dot",
                "cross_sensor_cross_norm",
                "acceleration_lag_dot",
                "gyroscope_lag_dot",
            ],
            "invariant_mean": self.invariant_mean.tolist(),
            "invariant_scale": self.invariant_scale.tolist(),
            "clipping_thresholds": self.clipping_thresholds.tolist(),
            "clipping_quantile": self.clipping_quantile,
            "fit_scope": "training_partition_only",
            "augmentation_order": "native_physical_transform_then_source_fitted_normalization",
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> V2PhysicalPreprocessor:
        """Restore and validate a serialized v2 preprocessor."""

        invariant_mean = np.asarray(payload["invariant_mean"], dtype=np.float64)
        invariant_scale = np.asarray(payload["invariant_scale"], dtype=np.float64)
        clipping_thresholds = np.asarray(payload["clipping_thresholds"], dtype=np.float64)
        if invariant_mean.shape != (8,) or invariant_scale.shape != (8,):
            raise ValueError("v2 invariant moments must each contain eight values")
        if clipping_thresholds.shape != (6,):
            raise ValueError("v2 clipping thresholds must contain six values")
        if np.any(invariant_scale <= 0) or np.any(clipping_thresholds <= 0):
            raise ValueError("v2 scales and clipping thresholds must be positive")
        for array in (invariant_mean, invariant_scale, clipping_thresholds):
            array.setflags(write=False)
        return cls(
            raw=ChannelStandardizer.from_dict(dict(payload["raw"])),
            invariant_mean=invariant_mean,
            invariant_scale=invariant_scale,
            clipping_thresholds=clipping_thresholds,
            clipping_quantile=float(payload["clipping_quantile"]),
        )
