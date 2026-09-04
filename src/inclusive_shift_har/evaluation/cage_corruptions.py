"""Deterministic physical fault simulations for prospective CAGE-HAR development.

These transformations are secondary engineering stressors.  Severity values are
dimensionless until calibrated against real sensor-fault episodes and therefore
cannot support real-world robustness claims by themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
CorruptionKind: TypeAlias = Literal[
    "clean",
    "constant_bias",
    "random_walk_drift",
    "gain_error",
    "colored_noise",
    "contiguous_gap",
    "axis_dropout",
    "accelerometer_dropout",
    "gyroscope_dropout",
    "gravity_dropout",
    "stuck_at",
    "saturation",
    "orientation_jump",
    "axis_sign_permutation",
    "timestamp_jitter",
    "quantization",
    "combined_bias_noise_gap",
]


@dataclass(frozen=True, slots=True)
class CageCorruptionSpec:
    """One prospective fault family and a dimensionless engineering severity."""

    kind: CorruptionKind
    severity: float


@dataclass(frozen=True, slots=True)
class CageCorruption:
    """New arrays plus explicit validity and fault lineage."""

    primary_signals: FloatArray
    gravity: FloatArray
    validity_mask: BoolArray
    timestamps_seconds: FloatArray
    metadata: dict[str, Any]


def _validate(
    primary_signals: NDArray[np.floating[Any]],
    gravity: NDArray[np.floating[Any]],
    validity_mask: NDArray[np.bool_] | None,
    timestamps_seconds: NDArray[np.floating[Any]] | None,
    sampling_rate_hz: float,
) -> tuple[FloatArray, FloatArray, BoolArray, FloatArray]:
    primary = np.asarray(primary_signals, dtype=np.float64)
    gravity_values = np.asarray(gravity, dtype=np.float64)
    if primary.ndim != 3 or primary.shape[1] < 8 or primary.shape[2] != 6:
        raise ValueError("CAGE corruptions require [window,time>=8,6] primary signals")
    if gravity_values.shape != (*primary.shape[:2], 3):
        raise ValueError("gravity must align as [window,time,3]")
    if primary.shape[0] == 0 or not np.isfinite(primary).all():
        raise ValueError("primary signals must be non-empty and finite")
    if not np.isfinite(gravity_values).all():
        raise ValueError("gravity must be finite; missing samples belong in the mask")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0.0:
        raise ValueError("sampling rate must be finite and positive")
    if validity_mask is None:
        validity = np.ones((*primary.shape[:2], 9), dtype=np.bool_)
    else:
        validity = np.asarray(validity_mask, dtype=np.bool_)
        if validity.shape != (*primary.shape[:2], 9):
            raise ValueError("validity mask must align as [window,time,9]")
    if timestamps_seconds is None:
        timestamp = np.broadcast_to(
            np.arange(primary.shape[1], dtype=np.float64)[None, :] / sampling_rate_hz,
            primary.shape[:2],
        ).copy()
    else:
        timestamp = np.asarray(timestamps_seconds, dtype=np.float64)
        if timestamp.shape != primary.shape[:2] or not np.isfinite(timestamp).all():
            raise ValueError("timestamps must align as [window,time] and be finite")
        if np.any(np.diff(timestamp, axis=1) <= 0.0):
            raise ValueError("input timestamps must be strictly increasing within each window")
    return primary.copy(), gravity_values.copy(), validity.copy(), timestamp.copy()


def _channel_scale(signals: FloatArray) -> FloatArray:
    scale = np.median(np.abs(signals - np.median(signals, axis=1, keepdims=True)), axis=(0, 1))
    fallback = np.median(np.abs(signals), axis=(0, 1))
    return np.asarray(np.maximum(np.maximum(scale, 0.1 * fallback), 1e-6), dtype=np.float64)


def _proper_rotation(axis: FloatArray, angle: float) -> FloatArray:
    unit = axis / np.maximum(np.linalg.norm(axis), 1e-12)
    x, y, z = unit.tolist()
    cross = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.asarray(
        np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross),
        dtype=np.float64,
    )


def _apply_shared_rotation(signals: FloatArray, rotation: FloatArray, start: int) -> None:
    for offset in (0, 3, 6):
        signals[:, start:, offset : offset + 3] = (
            signals[:, start:, offset : offset + 3] @ rotation.T
        )


def _apply_gap(
    signals: FloatArray,
    validity: BoolArray,
    *,
    fraction: float,
    generator: np.random.Generator,
) -> None:
    time_steps = signals.shape[1]
    duration = max(1, min(time_steps, int(np.ceil(fraction * time_steps))))
    for row in range(signals.shape[0]):
        start = int(generator.integers(0, time_steps - duration + 1))
        signals[row, start : start + duration] = 0.0
        validity[row, start : start + duration] = False


def apply_cage_corruption(
    primary_signals: NDArray[np.floating[Any]],
    gravity: NDArray[np.floating[Any]],
    spec: CageCorruptionSpec,
    *,
    sampling_rate_hz: float,
    seed: int,
    validity_mask: NDArray[np.bool_] | None = None,
    timestamps_seconds: NDArray[np.floating[Any]] | None = None,
) -> CageCorruption:
    """Return a deterministic fault without crossing existing window boundaries."""

    if not np.isfinite(spec.severity) or not 0.0 <= spec.severity <= 1.0:
        raise ValueError("CAGE corruption severity must lie in [0,1]")
    primary, gravity_values, validity, timestamps = _validate(
        primary_signals,
        gravity,
        validity_mask,
        timestamps_seconds,
        sampling_rate_hz,
    )
    combined = np.concatenate((primary, gravity_values), axis=2)
    generator = np.random.default_rng(seed)
    scale = _channel_scale(combined)
    severity = float(spec.severity)
    time_steps = combined.shape[1]

    if spec.kind == "clean":
        pass
    elif spec.kind == "constant_bias":
        direction = generator.choice(np.asarray([-1.0, 1.0]), size=combined.shape[2])
        combined += severity * scale[None, None, :] * direction[None, None, :]
    elif spec.kind == "random_walk_drift":
        increments = generator.normal(size=combined.shape)
        walk = np.cumsum(increments, axis=1)
        walk -= walk[:, :1]
        endpoint_scale = np.std(walk[:, -1], axis=0)
        walk /= np.maximum(endpoint_scale[None, None, :], 1e-6)
        combined += severity * scale[None, None, :] * walk
    elif spec.kind == "gain_error":
        direction = generator.choice(np.asarray([-1.0, 1.0]), size=combined.shape[2])
        gain = 1.0 + severity * direction
        combined *= gain[None, None, :]
    elif spec.kind == "colored_noise":
        innovations = generator.normal(size=combined.shape)
        noise = np.zeros_like(combined)
        for index in range(1, time_steps):
            noise[:, index] = 0.85 * noise[:, index - 1] + innovations[:, index]
        noise /= np.maximum(np.std(noise, axis=(0, 1), keepdims=True), 1e-6)
        combined += severity * scale[None, None, :] * noise
    elif spec.kind == "contiguous_gap":
        _apply_gap(combined, validity, fraction=severity, generator=generator)
    elif spec.kind == "axis_dropout":
        axis = int(generator.integers(0, 3))
        for offset in (0, 3, 6):
            combined[:, :, offset + axis] = 0.0
            validity[:, :, offset + axis] = False
    elif spec.kind in {"accelerometer_dropout", "gyroscope_dropout", "gravity_dropout"}:
        start = {"accelerometer_dropout": 0, "gyroscope_dropout": 3, "gravity_dropout": 6}[
            spec.kind
        ]
        combined[:, :, start : start + 3] = 0.0
        validity[:, :, start : start + 3] = False
    elif spec.kind == "stuck_at":
        duration = max(1, min(time_steps - 1, int(np.ceil(severity * time_steps))))
        start = time_steps - duration
        combined[:, start:] = combined[:, start - 1 : start]
    elif spec.kind == "saturation":
        threshold = np.maximum((3.5 - 3.0 * severity) * scale, 0.25 * scale)
        combined = np.clip(combined, -threshold[None, None, :], threshold[None, None, :])
    elif spec.kind == "orientation_jump":
        rotation_axis = generator.normal(size=3)
        rotation = _proper_rotation(np.asarray(rotation_axis, dtype=np.float64), severity * np.pi)
        _apply_shared_rotation(combined, rotation, time_steps // 2)
    elif spec.kind == "axis_sign_permutation":
        permutation = np.eye(3)[generator.permutation(3)]
        signs = np.diag(generator.choice(np.asarray([-1.0, 1.0]), size=3))
        transform = signs @ permutation
        _apply_shared_rotation(combined, transform, 0)
    elif spec.kind == "timestamp_jitter":
        base_interval = 1.0 / sampling_rate_hz
        intervals = generator.normal(
            loc=base_interval,
            scale=severity * 0.45 * base_interval,
            size=(combined.shape[0], time_steps - 1),
        )
        intervals = np.maximum(intervals, 0.05 * base_interval)
        timestamps[:, 1:] = timestamps[:, :1] + np.cumsum(intervals, axis=1)
    elif spec.kind == "quantization":
        step = np.maximum(severity * scale, 1e-8)
        combined = np.round(combined / step[None, None, :]) * step[None, None, :]
    elif spec.kind == "combined_bias_noise_gap":
        direction = generator.choice(np.asarray([-1.0, 1.0]), size=combined.shape[2])
        combined += 0.5 * severity * scale[None, None, :] * direction[None, None, :]
        innovations = generator.normal(size=combined.shape)
        noise = np.zeros_like(combined)
        for index in range(1, time_steps):
            noise[:, index] = 0.75 * noise[:, index - 1] + innovations[:, index]
        noise /= np.maximum(np.std(noise, axis=(0, 1), keepdims=True), 1e-6)
        combined += 0.5 * severity * scale[None, None, :] * noise
        _apply_gap(combined, validity, fraction=0.5 * severity, generator=generator)
    else:
        raise ValueError(f"unsupported CAGE corruption kind {spec.kind!r}")

    combined[~validity] = 0.0
    invalid_fraction = float((~validity).mean())
    metadata: dict[str, Any] = {
        "kind": spec.kind,
        "severity": severity,
        "seed": seed,
        "invalid_fraction": invalid_fraction,
        "timestamps_strictly_increasing": bool(np.all(np.diff(timestamps, axis=1) > 0.0)),
        "severity_empirically_calibrated": False,
        "primary_claim_eligible": False,
        "window_boundaries_crossed": False,
    }
    if not np.isfinite(combined).all() or not np.isfinite(timestamps).all():
        raise AssertionError("CAGE corruption produced non-finite output")
    return CageCorruption(
        primary_signals=np.asarray(combined[:, :, :6], dtype=np.float64),
        gravity=np.asarray(combined[:, :, 6:9], dtype=np.float64),
        validity_mask=np.asarray(validity, dtype=np.bool_),
        timestamps_seconds=np.asarray(timestamps, dtype=np.float64),
        metadata=metadata,
    )


__all__ = ["CageCorruption", "CageCorruptionSpec", "CorruptionKind", "apply_cage_corruption"]
