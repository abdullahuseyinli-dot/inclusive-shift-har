"""Framework-independent reconstruction of the coursework HAR architectures.

These specifications preserve the layer graph from notebook cells 10, 12, 14,
and 26 without importing a training framework at package import time.  The
dual-branch graph is correctly named as a jointly trained logit-average model,
not an ensemble of independently trained estimators.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

LEGACY_NOTEBOOK_SHA256 = "c25d78ad1a21e485a7bd0d9dd5cc74bba4d3872465be300d7be56d8b829f636"


@dataclass(frozen=True)
class LegacyArchitectureSpec:
    architecture_id: str
    family: Literal["bilstm", "cnn1d", "joint_dual_branch"]
    input_channels: int
    class_count: int
    hidden: int
    layers: int | None
    dropout: float
    temporal_head: bool | None
    cnn_hidden: int | None
    source_cells: tuple[str, ...]
    evidence_status: str = "architecture_reconstruction_no_model_result"

    def parameter_count(self) -> int:
        if self.family == "bilstm":
            assert self.layers is not None and self.temporal_head is not None
            return bilstm_parameter_count(
                input_size=self.input_channels,
                hidden=self.hidden,
                layers=self.layers,
                classes=self.class_count,
                temporal_head=self.temporal_head,
            )
        if self.family == "cnn1d":
            assert self.cnn_hidden is not None
            return cnn1d_parameter_count(
                input_size=self.input_channels,
                hidden=self.cnn_hidden,
                classes=self.class_count,
            )
        assert self.layers is not None
        assert self.temporal_head is not None
        assert self.cnn_hidden is not None
        return bilstm_parameter_count(
            input_size=self.input_channels,
            hidden=self.hidden,
            layers=self.layers,
            classes=self.class_count,
            temporal_head=self.temporal_head,
        ) + cnn1d_parameter_count(
            input_size=self.input_channels,
            hidden=self.cnn_hidden,
            classes=self.class_count,
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["parameter_count"] = self.parameter_count()
        if self.family == "joint_dual_branch":
            payload["aggregation"] = "arithmetic_mean_of_branch_logits"
            payload["training_semantics"] = "joint_end_to_end_single_objective"
        return payload


def bilstm_parameter_count(
    *,
    input_size: int,
    hidden: int,
    layers: int,
    classes: int,
    temporal_head: bool,
) -> int:
    """Count trainable parameters in the exact bidirectional LSTM graph."""

    if min(input_size, hidden, layers, classes) <= 0:
        raise ValueError("all BiLSTM dimensions must be positive")
    directions = 2
    recurrent = 0
    layer_input = input_size
    for _ in range(layers):
        # Per direction: weight_ih, weight_hh, bias_ih, and bias_hh.
        recurrent += directions * (4 * hidden * (layer_input + hidden) + 8 * hidden)
        layer_input = directions * hidden
    recurrent_output = directions * hidden
    head_input = recurrent_output * (2 if temporal_head else 1)
    layer_norm = 2 * head_input
    linear = head_input * classes + classes
    return recurrent + layer_norm + linear


def cnn1d_parameter_count(*, input_size: int, hidden: int, classes: int) -> int:
    """Count trainable parameters in the exact three-block CNN1D graph."""

    if min(input_size, hidden, classes) <= 0:
        raise ValueError("all CNN dimensions must be positive")
    widths = (hidden, hidden * 2, hidden * 4)
    kernels = (7, 5, 3)
    inputs = (input_size, widths[0], widths[1])
    convolution_and_batch_norm = sum(
        output_width * input_width * kernel
        + output_width  # convolution bias
        + 2 * output_width  # BatchNorm scale and bias
        for input_width, output_width, kernel in zip(inputs, widths, kernels, strict=True)
    )
    classifier = widths[-1] * classes + classes
    return convolution_and_batch_norm + classifier


def corrected_legacy_architectures(
    *, input_channels: int = 6, class_count: int = 6
) -> tuple[LegacyArchitectureSpec, ...]:
    """Return the corrected architecture registry for official primary-six input."""

    return (
        LegacyArchitectureSpec(
            architecture_id="legacy_bilstm_h192",
            family="bilstm",
            input_channels=input_channels,
            class_count=class_count,
            hidden=192,
            layers=2,
            dropout=0.3,
            temporal_head=False,
            cnn_hidden=None,
            source_cells=("10/11/4929a1ae", "26/27/12004a09"),
        ),
        LegacyArchitectureSpec(
            architecture_id="legacy_bilstm_h192_temporal",
            family="bilstm",
            input_channels=input_channels,
            class_count=class_count,
            hidden=192,
            layers=2,
            dropout=0.3,
            temporal_head=True,
            cnn_hidden=None,
            source_cells=("10/11/4929a1ae", "26/27/12004a09"),
        ),
        LegacyArchitectureSpec(
            architecture_id="legacy_cnn1d_h128",
            family="cnn1d",
            input_channels=input_channels,
            class_count=class_count,
            hidden=128,
            layers=None,
            dropout=0.15,
            temporal_head=None,
            cnn_hidden=128,
            source_cells=("12/13/cc39cfc2", "26/27/12004a09"),
        ),
        LegacyArchitectureSpec(
            architecture_id="legacy_bilstm_h256_temporal",
            family="bilstm",
            input_channels=input_channels,
            class_count=class_count,
            hidden=256,
            layers=2,
            dropout=0.3,
            temporal_head=True,
            cnn_hidden=None,
            source_cells=("10/11/4929a1ae", "26/27/12004a09"),
        ),
        LegacyArchitectureSpec(
            architecture_id="legacy_joint_bilstm256_cnn128",
            family="joint_dual_branch",
            input_channels=input_channels,
            class_count=class_count,
            hidden=256,
            layers=2,
            dropout=0.3,
            temporal_head=True,
            cnn_hidden=128,
            source_cells=(
                "10/11/4929a1ae",
                "12/13/cc39cfc2",
                "14/15/8fcda24e",
                "26/27/12004a09",
            ),
        ),
        LegacyArchitectureSpec(
            architecture_id="legacy_joint_bilstm512_cnn256",
            family="joint_dual_branch",
            input_channels=input_channels,
            class_count=class_count,
            hidden=512,
            layers=2,
            dropout=0.3,
            temporal_head=True,
            cnn_hidden=256,
            source_cells=(
                "10/11/4929a1ae",
                "12/13/cc39cfc2",
                "14/15/8fcda24e",
                "26/27/12004a09",
            ),
        ),
    )
