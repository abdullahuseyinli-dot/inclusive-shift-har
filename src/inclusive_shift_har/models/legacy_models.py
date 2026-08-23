"""Runnable, notebook-exact coursework HAR architecture reconstruction.

The layer graphs come from notebook cells 10, 12, 14, and 26.  Engineering
around data, splitting, normalization, checkpointing, and evaluation is not
copied from the notebook because those parts require the corrected protocol.
The dual graph is a jointly trained logit-fusion model, not an ensemble of
independently trained estimators.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor, nn

from inclusive_shift_har.models.common import HAROutput, require_window_tensor

LEGACY_MODEL_SOURCE_CELLS: Final[dict[str, tuple[str, ...]]] = {
    "lstm": ("10/11/4929a1ae", "26/27/12004a09"),
    "cnn1d": ("12/13/cc39cfc2", "26/27/12004a09"),
    "joint_logit_fusion": (
        "10/11/4929a1ae",
        "12/13/cc39cfc2",
        "14/15/8fcda24e",
        "26/27/12004a09",
    ),
}


class LegacyLSTMNet(nn.Module):
    """Exact LSTM/BiLSTM graph from notebook cell 10."""

    def __init__(
        self,
        *,
        input_channels: int,
        hidden_size: int,
        layers: int,
        num_classes: int,
        bidirectional: bool = True,
        dropout: float = 0.3,
        temporal_head: bool = False,
    ) -> None:
        super().__init__()
        if min(input_channels, hidden_size, layers, num_classes) <= 0:
            raise ValueError("all architecture dimensions must be positive")
        self.input_channels = input_channels
        self.bidirectional = bidirectional
        self.temporal_head = temporal_head
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        output_width = hidden_size * (2 if bidirectional else 1)
        head_width = output_width * (2 if temporal_head else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(head_width),
            nn.Dropout(dropout),
            nn.Linear(head_width, num_classes),
        )

    def forward(self, x: Tensor) -> HAROutput:
        require_window_tensor(x, channels=self.input_channels)
        sequence, (hidden, _) = self.lstm(x)
        if self.bidirectional:
            last = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            last = hidden[-1]
        if self.temporal_head:
            temporal_max = sequence.max(dim=1).values
            embedding = torch.cat((last, temporal_max), dim=1)
        else:
            embedding = last
        return HAROutput(logits=self.head(embedding), content=embedding)


class LegacyCNN1D(nn.Module):
    """Exact three-block CNN graph from notebook cell 12."""

    def __init__(
        self,
        *,
        input_channels: int,
        num_classes: int,
        hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(input_channels, hidden, num_classes) <= 0:
            raise ValueError("all architecture dimensions must be positive")
        self.input_channels = input_channels
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden, hidden * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden * 2, hidden * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, num_classes),
        )

    def forward(self, x: Tensor) -> HAROutput:
        require_window_tensor(x, channels=self.input_channels)
        pooled = self.net(x.transpose(1, 2))
        embedding = pooled.flatten(1)
        return HAROutput(logits=self.head(pooled), content=embedding)


class LegacyJointLogitFusion(nn.Module):
    """Jointly trained BiLSTM/CNN arithmetic-logit fusion from cell 14."""

    def __init__(self, bilstm: LegacyLSTMNet, cnn1d: LegacyCNN1D) -> None:
        super().__init__()
        if bilstm.input_channels != cnn1d.input_channels:
            raise ValueError("joint branches must use the same input channel count")
        self.input_channels = bilstm.input_channels
        self.bilstm = bilstm
        self.cnn1d = cnn1d

    def forward(self, x: Tensor) -> HAROutput:
        require_window_tensor(x, channels=self.input_channels)
        recurrent = self.bilstm(x)
        convolutional = self.cnn1d(x)
        logits = 0.5 * (recurrent.logits + convolutional.logits)
        content = torch.cat((recurrent.content, convolutional.content), dim=1)
        return HAROutput(logits=logits, content=content)


def build_exact_legacy_model(
    variant: str,
    *,
    num_classes: int = 6,
    input_channels: int = 6,
) -> nn.Module:
    """Build a stable notebook-exact variant for corrected reproduction."""

    key = variant.casefold()
    if key == "legacy_bilstm_h192":
        return LegacyLSTMNet(
            input_channels=input_channels,
            hidden_size=192,
            layers=2,
            num_classes=num_classes,
            bidirectional=True,
            dropout=0.3,
            temporal_head=False,
        )
    if key == "legacy_bilstm_h192_temporal":
        return LegacyLSTMNet(
            input_channels=input_channels,
            hidden_size=192,
            layers=2,
            num_classes=num_classes,
            bidirectional=True,
            dropout=0.3,
            temporal_head=True,
        )
    if key == "legacy_bilstm_h256_temporal":
        return LegacyLSTMNet(
            input_channels=input_channels,
            hidden_size=256,
            layers=2,
            num_classes=num_classes,
            bidirectional=True,
            dropout=0.3,
            temporal_head=True,
        )
    if key == "legacy_cnn1d_h128":
        return LegacyCNN1D(
            input_channels=input_channels,
            num_classes=num_classes,
            hidden=128,
            dropout=0.15,
        )
    if key == "legacy_joint_bilstm256_cnn128":
        return LegacyJointLogitFusion(
            LegacyLSTMNet(
                input_channels=input_channels,
                hidden_size=256,
                layers=2,
                num_classes=num_classes,
                bidirectional=True,
                dropout=0.3,
                temporal_head=True,
            ),
            LegacyCNN1D(
                input_channels=input_channels,
                num_classes=num_classes,
                hidden=128,
                dropout=0.15,
            ),
        )
    if key == "legacy_joint_bilstm512_cnn256":
        return LegacyJointLogitFusion(
            LegacyLSTMNet(
                input_channels=input_channels,
                hidden_size=512,
                layers=2,
                num_classes=num_classes,
                bidirectional=True,
                dropout=0.3,
                temporal_head=True,
            ),
            LegacyCNN1D(
                input_channels=input_channels,
                num_classes=num_classes,
                hidden=256,
                dropout=0.15,
            ),
        )
    choices = (
        "legacy_bilstm_h192",
        "legacy_bilstm_h192_temporal",
        "legacy_bilstm_h256_temporal",
        "legacy_cnn1d_h128",
        "legacy_joint_bilstm256_cnn128",
        "legacy_joint_bilstm512_cnn256",
    )
    raise ValueError(f"unknown exact legacy variant {variant!r}; choices: {choices}")
