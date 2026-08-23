"""MoRe-HAR: a compact motion-realization factorization hypothesis."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from inclusive_shift_har.models.common import (
    AttentiveStatisticalPooling,
    ConvNormAct,
    DepthwiseSeparableConv1d,
    HAROutput,
    ResidualTemporalBlock,
    require_window_tensor,
    trainable_parameter_count,
)


class MultiResolutionTemporal(nn.Module):
    """Parallel short-, medium-, and long-context separable temporal filters."""

    def __init__(self, in_channels: int, branch_channels: int, out_channels: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            DepthwiseSeparableConv1d(in_channels, branch_channels, kernel) for kernel in (3, 7, 15)
        )
        self.fuse = ConvNormAct(branch_channels * 3, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1)))


class MoReHAR(nn.Module):
    """Factorized activity-content and physical-realization representation model.

    The inference classifier consumes only ``z_content``. ``z_realization`` is supervised with
    non-sensitive descriptors or augmentation parameters during development and can be discarded
    at inference. Disability status, assistive-device metadata, and participant identity are never
    accepted as inputs.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        stem_channels: int = 32,
        temporal_channels: int = 96,
        content_dim: int = 96,
        realization_dim: int = 48,
        descriptor_dim: int = 8,
        dropout: float = 0.1,
        parameter_limit: int = 2_000_000,
    ) -> None:
        super().__init__()
        self.parameter_limit = parameter_limit
        self.accelerometer_stem = ConvNormAct(3, stem_channels, 7)
        self.gyroscope_stem = ConvNormAct(3, stem_channels, 7)
        self.multi_resolution = MultiResolutionTemporal(
            stem_channels * 2,
            branch_channels=temporal_channels // 2,
            out_channels=temporal_channels,
        )
        self.temporal = nn.Sequential(
            *(
                ResidualTemporalBlock(
                    temporal_channels,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in (1, 2, 4, 8)
            )
        )
        self.pool = AttentiveStatisticalPooling(temporal_channels)
        pooled_dim = temporal_channels * 2
        self.content_projection = nn.Sequential(
            nn.Linear(pooled_dim, content_dim),
            nn.LayerNorm(content_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.realization_projection = nn.Sequential(
            nn.Linear(pooled_dim, realization_dim),
            nn.LayerNorm(realization_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.activity_classifier = nn.Linear(content_dim, num_classes)
        self.descriptor_head = nn.Linear(realization_dim, descriptor_dim)
        parameter_count = trainable_parameter_count(self)
        if parameter_count >= parameter_limit:
            raise ValueError(
                f"MoRe-HAR has {parameter_count:,} parameters, violating limit {parameter_limit:,}"
            )

    def forward(self, x: Tensor) -> HAROutput:
        require_window_tensor(x)
        channels_first = x.transpose(1, 2)
        acc = self.accelerometer_stem(channels_first[:, :3])
        gyro = self.gyroscope_stem(channels_first[:, 3:])
        features = self.multi_resolution(torch.cat((acc, gyro), dim=1))
        pooled = self.pool(self.temporal(features))
        content = self.content_projection(pooled)
        realization = self.realization_projection(pooled)
        return HAROutput(
            logits=self.activity_classifier(content),
            content=content,
            realization=realization,
            descriptor_prediction=self.descriptor_head(realization),
        )
