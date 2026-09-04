"""Deterministic corruption suite with explicit per-value provenance validity masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
CorruptionName = Literal[
    "clean",
    "axis_dropout",
    "modality_dropout",
    "contiguous_gap",
    "stuck_at",
    "saturation",
    "bias_drift",
    "scale_drift",
    "gaussian_noise",
    "constrained_rotation",
]


@dataclass(frozen=True)
class ProvenanceCorruptionSpec:
    """One corruption with a normalized severity and explicit validity semantics."""

    name: CorruptionName
    severity: float

    def __post_init__(self) -> None:
        if self.name == "clean":
            if self.severity != 0.0:
                raise ValueError("clean severity must be zero")
        elif not 0.0 < self.severity <= 1.0:
            raise ValueError("corruption severity must lie in (0,1]")


@dataclass(frozen=True)
class ProvenanceCorruption:
    """Corrupted signals, explicit validity mask, and injection diagnostics."""

    signals: FloatArray
    validity_mask: BoolArray
    affected_fraction: float
    invalid_fraction: float


def _rotation_matrix(axis: FloatArray, angle: float) -> FloatArray:
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    x, y, z = axis
    cosine, sine = np.cos(angle), np.sin(angle)
    cross = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.asarray(
        cosine * np.eye(3) + sine * cross + (1.0 - cosine) * np.outer(axis, axis),
        dtype=np.float64,
    )


def apply_provenance_corruption(
    signals: NDArray[np.floating[Any]],
    spec: ProvenanceCorruptionSpec,
    *,
    channel_scale: NDArray[np.floating[Any]],
    saturation_threshold: NDArray[np.floating[Any]],
    seed: int,
) -> ProvenanceCorruption:
    """Inject a corruption while distinguishing absent/invalid from merely degraded data."""

    values = np.asarray(signals, dtype=np.float64)
    scale = np.asarray(channel_scale, dtype=np.float64)
    threshold = np.asarray(saturation_threshold, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] == 0 or values.shape[2] != 6:
        raise ValueError("provenance corruption requires non-empty [window,time,6] input")
    if (
        scale.shape != (6,)
        or threshold.shape != (6,)
        or not np.isfinite(values).all()
        or np.any(scale <= 0)
        or np.any(threshold <= 0)
    ):
        raise ValueError("provenance corruption inputs or train-only scales are invalid")
    generator = np.random.default_rng(seed)
    result = values.copy()
    validity = np.ones(values.shape, dtype=np.bool_)
    affected = np.zeros(values.shape, dtype=np.bool_)
    windows, time_steps, _ = values.shape
    if spec.name == "clean":
        pass
    elif spec.name == "axis_dropout":
        count = min(3, max(1, round(3 * spec.severity)))
        for window in range(windows):
            selected = generator.choice(6, size=count, replace=False)
            result[window, :, selected] = 0.0
            validity[window, :, selected] = False
            affected[window, :, selected] = True
    elif spec.name == "modality_dropout":
        for window in range(windows):
            start = 0 if int(generator.integers(0, 2)) == 0 else 3
            result[window, :, start : start + 3] = 0.0
            validity[window, :, start : start + 3] = False
            affected[window, :, start : start + 3] = True
    elif spec.name in {"contiguous_gap", "stuck_at"}:
        length = min(time_steps, max(1, round(time_steps * spec.severity)))
        for window in range(windows):
            start = int(generator.integers(0, time_steps - length + 1))
            if spec.name == "contiguous_gap":
                result[window, start : start + length, :] = 0.0
                validity[window, start : start + length, :] = False
                affected[window, start : start + length, :] = True
            else:
                channel = int(generator.integers(0, 6))
                anchor = result[window, max(0, start - 1), channel]
                result[window, start : start + length, channel] = anchor
                validity[window, start : start + length, channel] = False
                affected[window, start : start + length, channel] = True
    elif spec.name == "saturation":
        limit = threshold * (1.0 - 0.9 * spec.severity)
        clipped = np.clip(result, -limit[None, None, :], limit[None, None, :])
        affected = clipped != result
        validity[affected] = False
        result = clipped
    elif spec.name in {"bias_drift", "scale_drift"}:
        signs = generator.choice(np.asarray([-1.0, 1.0]), size=(windows, 1, 6))
        ramp = np.linspace(-1.0, 1.0, time_steps)[None, :, None]
        if spec.name == "bias_drift":
            result += signs * ramp * spec.severity * scale[None, None, :]
        else:
            result *= 1.0 + signs * ramp * spec.severity
        affected[:] = True
    elif spec.name == "gaussian_noise":
        result += generator.normal(size=result.shape) * spec.severity * scale[None, None, :]
        affected[:] = True
    elif spec.name == "constrained_rotation":
        maximum_angle = np.deg2rad(45.0 * spec.severity)
        for window in range(windows):
            axis = generator.normal(size=3)
            angle = float(generator.uniform(-maximum_angle, maximum_angle))
            rotation = _rotation_matrix(np.asarray(axis, dtype=np.float64), angle)
            result[window, :, :3] = result[window, :, :3] @ rotation.T
            result[window, :, 3:] = result[window, :, 3:] @ rotation.T
        affected[:] = True
    else:
        raise AssertionError(f"unhandled provenance corruption {spec.name!r}")
    if not np.isfinite(result).all():
        raise AssertionError("provenance corruption produced non-finite signals")
    return ProvenanceCorruption(
        signals=np.asarray(result, dtype=np.float64),
        validity_mask=validity,
        affected_fraction=float(affected.mean()),
        invalid_fraction=float((~validity).mean()),
    )


def interpolate_from_validity_mask(
    signals: NDArray[np.floating[Any]],
    validity_mask: NDArray[np.bool_],
    *,
    training_channel_median: NDArray[np.floating[Any]],
) -> FloatArray:
    """Interpolate invalid values; use train-only channel medians for absent channels."""

    values = np.asarray(signals, dtype=np.float64)
    validity = np.asarray(validity_mask, dtype=np.bool_)
    fallback = np.asarray(training_channel_median, dtype=np.float64)
    if (
        values.ndim != 3
        or values.shape != validity.shape
        or values.shape[2] != 6
        or fallback.shape != (6,)
        or not np.isfinite(values).all()
        or not np.isfinite(fallback).all()
    ):
        raise ValueError("mask interpolation inputs are invalid or misaligned")
    result = values.copy()
    time = np.arange(values.shape[1], dtype=np.float64)
    for window in range(values.shape[0]):
        for channel in range(6):
            observed = validity[window, :, channel]
            if observed.all():
                continue
            if not observed.any():
                result[window, :, channel] = fallback[channel]
                continue
            result[window, ~observed, channel] = np.interp(
                time[~observed], time[observed], result[window, observed, channel]
            )
    if not np.isfinite(result).all():
        raise AssertionError("mask interpolation produced non-finite signals")
    return np.asarray(result, dtype=np.float64)
