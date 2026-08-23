"""Compact, locally specified neural baselines for controlled HAR comparisons.

These implementations are repository baselines, not claims of bit-for-bit reproduction of
third-party code. Architecture provenance and tuning budgets travel with experiment records.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from inclusive_shift_har.models.common import (
    ConvNormAct,
    HAROutput,
    ResidualTemporalBlock,
    require_window_tensor,
)


class CNN1D(nn.Module):
    """Three-block one-dimensional convolutional baseline."""

    def __init__(self, num_classes: int, *, input_channels: int = 6, width: int = 64) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.encoder = nn.Sequential(
            ConvNormAct(input_channels, width, 7),
            nn.MaxPool1d(2),
            ConvNormAct(width, width * 2, 5),
            nn.MaxPool1d(2),
            ConvNormAct(width * 2, width * 2, 3),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(width * 2, num_classes)

    def forward(self, x: Tensor) -> HAROutput:
        require_window_tensor(x, channels=self.input_channels)
        embedding = self.encoder(x.transpose(1, 2)).squeeze(-1)
        return HAROutput(logits=self.classifier(embedding), content=embedding)


class BiLSTM(nn.Module):
    """Bidirectional recurrent baseline with mean temporal pooling."""

    def __init__(
        self,
        num_classes: int,
        *,
        input_channels: int = 6,
        hidden_size: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.encoder = nn.LSTM(
            input_channels,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: Tensor) -> HAROutput:
        require_window_tensor(x, channels=self.input_channels)
        sequence, _ = self.encoder(x)
        embedding = sequence.mean(dim=1)
        return HAROutput(logits=self.classifier(embedding), content=embedding)


class JointCNNBiLSTM(nn.Module):
    """Jointly trained feature-fusion model; it is deliberately not called an ensemble."""

    def __init__(
        self,
        num_classes: int,
        *,
        input_channels: int = 6,
        width: int = 48,
        hidden_size: int = 48,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.cnn = nn.Sequential(
            ConvNormAct(input_channels, width, 7),
            nn.MaxPool1d(2),
            ConvNormAct(width, width * 2, 5),
            nn.AdaptiveAvgPool1d(1),
        )
        self.recurrent = nn.LSTM(
            input_channels,
            hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        fused_width = width * 2 + hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(fused_width, fused_width // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fused_width // 2, num_classes),
        )

    def forward(self, x: Tensor) -> HAROutput:
        require_window_tensor(x, channels=self.input_channels)
        cnn_embedding = self.cnn(x.transpose(1, 2)).squeeze(-1)
        recurrent_sequence, _ = self.recurrent(x)
        recurrent_embedding = recurrent_sequence.mean(dim=1)
        embedding = torch.cat((cnn_embedding, recurrent_embedding), dim=1)
        return HAROutput(logits=self.classifier(embedding), content=embedding)


class DeepConvLSTM(nn.Module):
    """Deep convolutional recurrent baseline in the DeepConvLSTM family."""

    def __init__(
        self,
        num_classes: int,
        *,
        input_channels: int = 6,
        conv_width: int = 64,
        hidden_size: int = 96,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.convolutions = nn.Sequential(
            ConvNormAct(input_channels, conv_width, 5),
            ConvNormAct(conv_width, conv_width, 5),
            ConvNormAct(conv_width, conv_width, 5),
            ConvNormAct(conv_width, conv_width, 5),
        )
        self.recurrent = nn.LSTM(
            conv_width,
            hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: Tensor) -> HAROutput:
        require_window_tensor(x, channels=self.input_channels)
        features = self.convolutions(x.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.recurrent(features)
        embedding = sequence[:, -1]
        return HAROutput(logits=self.classifier(embedding), content=embedding)


class CompactResidualHAR(nn.Module):
    """CNN-HAR/TinyHAR-style compact residual temporal baseline."""

    def __init__(
        self,
        num_classes: int,
        *,
        input_channels: int = 6,
        width: int = 64,
        blocks: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.stem = ConvNormAct(input_channels, width, 7)
        self.blocks = nn.Sequential(
            *(
                ResidualTemporalBlock(width, dilation=2**index, dropout=dropout)
                for index in range(blocks)
            )
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(width, num_classes)

    def forward(self, x: Tensor) -> HAROutput:
        require_window_tensor(x, channels=self.input_channels)
        features = self.blocks(self.stem(x.transpose(1, 2)))
        embedding = self.pool(features).squeeze(-1)
        return HAROutput(logits=self.classifier(embedding), content=embedding)


class StaticDualBranchHAR(nn.Module):
    """Parameter-controlled static accelerometer/gyroscope fusion baseline."""

    def __init__(self, num_classes: int, *, width: int = 96, dropout: float = 0.1) -> None:
        super().__init__()
        if width % 2:
            raise ValueError("width must be even")
        branch_width = width // 2
        self.accelerometer = nn.Sequential(
            ConvNormAct(3, branch_width, 7),
            ResidualTemporalBlock(branch_width, dilation=1, dropout=dropout),
            ResidualTemporalBlock(branch_width, dilation=4, dropout=dropout),
        )
        self.gyroscope = nn.Sequential(
            ConvNormAct(3, branch_width, 7),
            ResidualTemporalBlock(branch_width, dilation=1, dropout=dropout),
            ResidualTemporalBlock(branch_width, dilation=4, dropout=dropout),
        )
        self.fusion = nn.Sequential(
            ConvNormAct(width, width, 5),
            ResidualTemporalBlock(width, dilation=2, dropout=dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(width, num_classes)

    def forward(self, x: Tensor) -> HAROutput:
        require_window_tensor(x)
        channels_first = x.transpose(1, 2)
        acc = self.accelerometer(channels_first[:, :3])
        gyro = self.gyroscope(channels_first[:, 3:])
        embedding = self.fusion(torch.cat((acc, gyro), dim=1)).squeeze(-1)
        return HAROutput(logits=self.classifier(embedding), content=embedding)


BaselineFactory = Callable[[int, int], nn.Module]


def build_baseline(name: str, *, num_classes: int, input_channels: int = 6) -> nn.Module:
    """Build a locked local baseline by stable registry name."""

    factories: dict[str, BaselineFactory] = {
        "cnn1d": lambda classes, channels: CNN1D(classes, input_channels=channels),
        "bilstm": lambda classes, channels: BiLSTM(classes, input_channels=channels),
        "joint_cnn_bilstm": lambda classes, channels: JointCNNBiLSTM(
            classes, input_channels=channels
        ),
        "deepconvlstm": lambda classes, channels: DeepConvLSTM(classes, input_channels=channels),
        "compact_residual_96": lambda classes, channels: CompactResidualHAR(
            classes, input_channels=channels, width=96, blocks=5
        ),
        "compact_residual_64": lambda classes, channels: CompactResidualHAR(
            classes, input_channels=channels, width=64, blocks=4
        ),
        "compact_residual_32": lambda classes, channels: CompactResidualHAR(
            classes, input_channels=channels, width=32, blocks=3
        ),
        "static_dual_branch": lambda classes, channels: (
            StaticDualBranchHAR(classes) if channels == 6 else _raise_dual_branch_channels(channels)
        ),
        "static_dual_branch_matched": lambda classes, channels: (
            StaticDualBranchHAR(classes, width=120)
            if channels == 6
            else _raise_dual_branch_channels(channels)
        ),
    }
    try:
        factory = factories[name.casefold()]
    except KeyError as exc:
        raise ValueError(f"unknown baseline {name!r}; choices: {sorted(factories)}") from exc
    return factory(num_classes, input_channels)


def _raise_dual_branch_channels(channels: int) -> nn.Module:
    raise ValueError(f"static_dual_branch requires six channels, received {channels}")
