"""Deterministic, label-preserving inertial augmentation primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from inclusive_shift_har.models.common import require_window_tensor


@dataclass(frozen=True)
class AugmentedBatch:
    """Augmented signals and the sampled non-sensitive transformation parameters."""

    signals: Tensor
    parameters: Tensor


def signal_descriptors(x: Tensor) -> Tensor:
    """Compute eight measurable motion descriptors without identity/group metadata."""

    require_window_tensor(x)
    acc = x[:, :, :3]
    gyro = x[:, :, 3:]
    acc_norm = torch.linalg.vector_norm(acc, dim=2)
    gyro_norm = torch.linalg.vector_norm(gyro, dim=2)
    acc_jerk = torch.diff(acc, dim=1)
    gyro_jerk = torch.diff(gyro, dim=1)
    return torch.stack(
        (
            acc_norm.mean(dim=1),
            acc_norm.std(dim=1, unbiased=False),
            gyro_norm.mean(dim=1),
            gyro_norm.std(dim=1, unbiased=False),
            acc.square().mean(dim=(1, 2)),
            gyro.square().mean(dim=(1, 2)),
            acc_jerk.abs().mean(dim=(1, 2)),
            gyro_jerk.abs().mean(dim=(1, 2)),
        ),
        dim=1,
    )


def _uniform(
    shape: tuple[int, ...],
    *,
    low: float,
    high: float,
    reference: Tensor,
    generator: torch.Generator,
) -> Tensor:
    values = torch.rand(
        shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )
    return low + (high - low) * values


def physically_plausible_augmentation(
    x: Tensor,
    *,
    generator: torch.Generator,
    amplitude_range: tuple[float, float] = (0.85, 1.15),
    time_scale_range: tuple[float, float] = (0.9, 1.1),
    yaw_degrees: float = 15.0,
    noise_standard_deviation: float = 0.01,
) -> AugmentedBatch:
    """Apply bounded amplitude, cadence, orientation, and additive-noise transforms.

    The same rigid yaw rotation is applied to accelerometer and gyroscope triads. Time scaling uses
    border padding and retains the original 128-sample interface. Parameters are sampled per window
    by the caller-provided device-local generator, making the transform replayable from RNG state.
    """

    require_window_tensor(x)
    if amplitude_range[0] <= 0 or amplitude_range[0] > amplitude_range[1]:
        raise ValueError("amplitude_range must be ordered and strictly positive")
    if time_scale_range[0] <= 0 or time_scale_range[0] > time_scale_range[1]:
        raise ValueError("time_scale_range must be ordered and strictly positive")
    if yaw_degrees < 0 or noise_standard_deviation < 0:
        raise ValueError("yaw and noise bounds must be non-negative")

    batch, time, _ = x.shape
    acc_scale = _uniform(
        (batch, 1, 1),
        low=amplitude_range[0],
        high=amplitude_range[1],
        reference=x,
        generator=generator,
    )
    gyro_scale = _uniform(
        (batch, 1, 1),
        low=amplitude_range[0],
        high=amplitude_range[1],
        reference=x,
        generator=generator,
    )
    time_scale = _uniform(
        (batch, 1),
        low=time_scale_range[0],
        high=time_scale_range[1],
        reference=x,
        generator=generator,
    )
    yaw_limit = torch.pi * yaw_degrees / 180.0
    yaw = _uniform(
        (batch, 1),
        low=-float(yaw_limit),
        high=float(yaw_limit),
        reference=x,
        generator=generator,
    )

    base = torch.linspace(-1.0, 1.0, time, device=x.device, dtype=x.dtype)
    temporal_coordinates = torch.clamp(time_scale * base.unsqueeze(0), -1.0, 1.0)
    zeros = torch.zeros_like(temporal_coordinates)
    grid = torch.stack((temporal_coordinates, zeros), dim=-1).unsqueeze(1)
    warped = (
        F.grid_sample(
            x.transpose(1, 2).unsqueeze(2),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        .squeeze(2)
        .transpose(1, 2)
    )

    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    row_x = torch.cat((cosine, -sine, torch.zeros_like(cosine)), dim=1)
    row_y = torch.cat((sine, cosine, torch.zeros_like(cosine)), dim=1)
    row_z = torch.cat(
        (torch.zeros_like(cosine), torch.zeros_like(cosine), torch.ones_like(cosine)),
        dim=1,
    )
    rotation = torch.stack((row_x, row_y, row_z), dim=1)
    acc = torch.bmm(warped[:, :, :3], rotation.transpose(1, 2)) * acc_scale
    gyro = torch.bmm(warped[:, :, 3:], rotation.transpose(1, 2)) * gyro_scale
    transformed = torch.cat((acc, gyro), dim=2)
    if noise_standard_deviation:
        noise = torch.randn(
            transformed.shape,
            device=transformed.device,
            dtype=transformed.dtype,
            generator=generator,
        )
        transformed = transformed + noise_standard_deviation * noise

    parameters = torch.cat(
        (
            torch.log(acc_scale[:, 0]),
            torch.log(gyro_scale[:, 0]),
            time_scale - 1.0,
            torch.sin(yaw),
            torch.cos(yaw) - 1.0,
        ),
        dim=1,
    )
    return AugmentedBatch(signals=transformed, parameters=parameters)
