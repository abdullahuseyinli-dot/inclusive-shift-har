"""Shared neural-network components for six-channel inertial HAR models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class HAROutput:
    """Standardized model output used by every neural training objective."""

    logits: Tensor
    content: Tensor | None = None
    realization: Tensor | None = None
    descriptor_prediction: Tensor | None = None
    domain_logits: Tensor | None = None


def require_window_tensor(x: Tensor, *, channels: int = 6) -> None:
    """Validate the public ``[batch, time, channels]`` tensor contract."""

    if x.ndim != 3:
        raise ValueError(f"expected [batch, time, channels], received shape {tuple(x.shape)}")
    if x.shape[1] < 2:
        raise ValueError("a HAR window must contain at least two time samples")
    if x.shape[2] != channels:
        raise ValueError(f"expected {channels} channels, received {x.shape[2]}")
    if not x.is_floating_point():
        raise TypeError("HAR inputs must be floating-point tensors")


def trainable_parameter_count(module: nn.Module) -> int:
    """Return the exact number of trainable scalar parameters."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class ConvNormAct(nn.Sequential):
    """Length-preserving Conv1d, BatchNorm and GELU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        groups: int = 1,
        stride: int = 1,
    ) -> None:
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise temporal filtering followed by pointwise channel mixing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.depthwise = ConvNormAct(
            in_channels,
            in_channels,
            kernel_size,
            dilation=dilation,
            groups=in_channels,
        )
        self.pointwise = ConvNormAct(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.pointwise(self.depthwise(x)))


class ResidualTemporalBlock(nn.Module):
    """Compact residual dilated block with separable convolutions."""

    def __init__(self, channels: int, *, dilation: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv1d(channels, channels, 5, dilation=dilation),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.activation(x + self.block(x)))


class AttentiveStatisticalPooling(nn.Module):
    """Learned temporal pooling returning weighted mean and standard deviation."""

    def __init__(self, channels: int, attention_channels: int | None = None) -> None:
        super().__init__()
        hidden = attention_channels or max(channels // 2, 8)
        self.attention = nn.Sequential(
            nn.Conv1d(channels, hidden, 1),
            nn.Tanh(),
            nn.Conv1d(hidden, 1, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        weights = torch.softmax(self.attention(x), dim=-1)
        mean = torch.sum(weights * x, dim=-1)
        second_moment = torch.sum(weights * x.square(), dim=-1)
        variance = torch.clamp(second_moment - mean.square(), min=1e-6)
        return torch.cat((mean, torch.sqrt(variance)), dim=1)
