"""Deterministic physical-space augmentation for FuSE-ReFrame development."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from inclusive_shift_har.models.common import require_window_tensor


@dataclass(frozen=True)
class PhysicalAugmentationConfig:
    """Predeclared hypothesis bounds for native-unit transformations."""

    amplitude_low: float = 0.9
    amplitude_high: float = 1.1
    time_scale_low: float = 0.92
    time_scale_high: float = 1.08
    rotation_degrees: float = 20.0
    acceleration_noise_sd_g: float = 0.01
    gyroscope_noise_sd_rad_s: float = 0.01
    acceleration_bias_g: float = 0.01
    gyroscope_bias_rad_s: float = 0.01
    acceleration_drift_g: float = 0.01
    gyroscope_drift_rad_s: float = 0.01
    channel_dropout_probability: float = 0.02

    def __post_init__(self) -> None:
        if not 0 < self.amplitude_low <= self.amplitude_high:
            raise ValueError("amplitude bounds must be ordered and positive")
        if not 0 < self.time_scale_low <= self.time_scale_high:
            raise ValueError("time-scale bounds must be ordered and positive")
        nonnegative = (
            self.rotation_degrees,
            self.acceleration_noise_sd_g,
            self.gyroscope_noise_sd_rad_s,
            self.acceleration_bias_g,
            self.gyroscope_bias_rad_s,
            self.acceleration_drift_g,
            self.gyroscope_drift_rad_s,
            self.channel_dropout_probability,
        )
        if any(value < 0 for value in nonnegative):
            raise ValueError("physical augmentation magnitudes must be non-negative")
        if self.channel_dropout_probability >= 1:
            raise ValueError("channel-dropout probability must be below one")


@dataclass(frozen=True)
class PhysicalAugmentedBatch:
    """Augmented native signals and known clean-to-augmented transform parameters."""

    signals: Tensor
    transform_parameters: Tensor
    dropped_channels: Tensor


def _uniform(
    shape: tuple[int, ...],
    *,
    low: float,
    high: float,
    reference: Tensor,
    generator: torch.Generator,
) -> Tensor:
    values = torch.rand(shape, dtype=reference.dtype, device=reference.device, generator=generator)
    return low + (high - low) * values


def _rotation_matrices(
    batch: int,
    *,
    reference: Tensor,
    generator: torch.Generator,
    maximum_degrees: float,
) -> tuple[Tensor, Tensor]:
    axis = torch.randn(
        (batch, 3),
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )
    axis = axis / torch.clamp(torch.linalg.vector_norm(axis, dim=1, keepdim=True), min=1e-8)
    angle = _uniform(
        (batch, 1),
        low=-maximum_degrees * torch.pi / 180.0,
        high=maximum_degrees * torch.pi / 180.0,
        reference=reference,
        generator=generator,
    )
    x, y, z = axis.unbind(dim=1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        (
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ),
        dim=1,
    ).reshape(batch, 3, 3)
    identity = torch.eye(3, dtype=reference.dtype, device=reference.device).expand(batch, -1, -1)
    sine = torch.sin(angle).reshape(batch, 1, 1)
    cosine = torch.cos(angle).reshape(batch, 1, 1)
    rotation = identity + sine * skew + (1.0 - cosine) * torch.bmm(skew, skew)
    return rotation, torch.cat((axis, angle), dim=1)


def _time_warp(signals: Tensor, scale: Tensor) -> Tensor:
    _, time, _ = signals.shape
    base = torch.linspace(-1.0, 1.0, time, dtype=signals.dtype, device=signals.device)
    coordinates = torch.clamp(scale * base.unsqueeze(0), -1.0, 1.0)
    grid = torch.stack((coordinates, torch.zeros_like(coordinates)), dim=-1).unsqueeze(1)
    return (
        F.grid_sample(
            signals.transpose(1, 2).unsqueeze(2),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        .squeeze(2)
        .transpose(1, 2)
    )


def augment_native_signals(
    signals: Tensor,
    *,
    generator: torch.Generator,
    config: PhysicalAugmentationConfig | None = None,
) -> PhysicalAugmentedBatch:
    """Apply replayable, bounded transforms before any source-fitted normalization."""

    require_window_tensor(signals)
    if config is None:
        config = PhysicalAugmentationConfig()
    batch, time, _ = signals.shape
    time_scale = _uniform(
        (batch, 1),
        low=config.time_scale_low,
        high=config.time_scale_high,
        reference=signals,
        generator=generator,
    )
    warped = _time_warp(signals, time_scale)
    rotation, rotation_parameters = _rotation_matrices(
        batch,
        reference=signals,
        generator=generator,
        maximum_degrees=config.rotation_degrees,
    )
    acceleration = torch.bmm(warped[:, :, :3], rotation.transpose(1, 2))
    gyroscope = torch.bmm(warped[:, :, 3:], rotation.transpose(1, 2))
    acceleration_gain = _uniform(
        (batch, 1, 1),
        low=config.amplitude_low,
        high=config.amplitude_high,
        reference=signals,
        generator=generator,
    )
    gyroscope_gain = _uniform(
        (batch, 1, 1),
        low=config.amplitude_low,
        high=config.amplitude_high,
        reference=signals,
        generator=generator,
    )
    acceleration_bias = _uniform(
        (batch, 1, 3),
        low=-config.acceleration_bias_g,
        high=config.acceleration_bias_g,
        reference=signals,
        generator=generator,
    )
    gyroscope_bias = _uniform(
        (batch, 1, 3),
        low=-config.gyroscope_bias_rad_s,
        high=config.gyroscope_bias_rad_s,
        reference=signals,
        generator=generator,
    )
    acceleration_drift = _uniform(
        (batch, 1, 3),
        low=-config.acceleration_drift_g,
        high=config.acceleration_drift_g,
        reference=signals,
        generator=generator,
    )
    gyroscope_drift = _uniform(
        (batch, 1, 3),
        low=-config.gyroscope_drift_rad_s,
        high=config.gyroscope_drift_rad_s,
        reference=signals,
        generator=generator,
    )
    ramp = torch.linspace(-1.0, 1.0, time, dtype=signals.dtype, device=signals.device)
    ramp = ramp.reshape(1, time, 1)
    acceleration = acceleration * acceleration_gain + acceleration_bias + ramp * acceleration_drift
    gyroscope = gyroscope * gyroscope_gain + gyroscope_bias + ramp * gyroscope_drift
    acceleration_noise = (
        torch.randn(
            acceleration.shape,
            dtype=signals.dtype,
            device=signals.device,
            generator=generator,
        )
        * config.acceleration_noise_sd_g
    )
    gyroscope_noise = (
        torch.randn(
            gyroscope.shape,
            dtype=signals.dtype,
            device=signals.device,
            generator=generator,
        )
        * config.gyroscope_noise_sd_rad_s
    )
    transformed = torch.cat(
        (acceleration + acceleration_noise, gyroscope + gyroscope_noise),
        dim=2,
    )
    dropped = (
        torch.rand(
            (batch, 6),
            device=signals.device,
            generator=generator,
        )
        < config.channel_dropout_probability
    )
    transformed = transformed.masked_fill(dropped.unsqueeze(1), 0.0)
    parameters = torch.cat(
        (
            time_scale,
            rotation_parameters,
            acceleration_gain.reshape(batch, 1),
            gyroscope_gain.reshape(batch, 1),
            acceleration_bias.reshape(batch, 3),
            gyroscope_bias.reshape(batch, 3),
            acceleration_drift.reshape(batch, 3),
            gyroscope_drift.reshape(batch, 3),
        ),
        dim=1,
    )
    return PhysicalAugmentedBatch(
        signals=transformed,
        transform_parameters=parameters,
        dropped_channels=dropped,
    )
