"""Model registry and architecture specifications."""

from inclusive_shift_har.models.baselines import (
    CNN1D,
    BiLSTM,
    CompactResidualHAR,
    DeepConvLSTM,
    JointCNNBiLSTM,
    StaticDualBranchHAR,
    build_baseline,
)
from inclusive_shift_har.models.common import HAROutput, trainable_parameter_count
from inclusive_shift_har.models.legacy_models import build_exact_legacy_model
from inclusive_shift_har.models.more_har import MoReHAR

__all__ = [
    "CNN1D",
    "BiLSTM",
    "CompactResidualHAR",
    "DeepConvLSTM",
    "HAROutput",
    "JointCNNBiLSTM",
    "MoReHAR",
    "StaticDualBranchHAR",
    "build_baseline",
    "build_exact_legacy_model",
    "trainable_parameter_count",
]
