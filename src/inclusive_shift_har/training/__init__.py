"""Training utilities and objectives."""

from inclusive_shift_har.training.augmentation import (
    AugmentedBatch,
    physically_plausible_augmentation,
    signal_descriptors,
)
from inclusive_shift_har.training.engine import (
    TrainingConfig,
    TrainingLineage,
    configure_determinism,
    predict_model,
    reconstruct_checkpoint,
    train_source_model,
    training_config_from_dict,
)
from inclusive_shift_har.training.final_checkpoint import validate_fixed_epoch_checkpoint
from inclusive_shift_har.training.objectives import (
    GroupDROState,
    MoReObjectiveWeights,
    coral_loss,
    cross_covariance_loss,
    more_har_objective,
    supervised_contrastive_loss,
    symmetric_consistency_loss,
)

__all__ = [
    "AugmentedBatch",
    "GroupDROState",
    "MoReObjectiveWeights",
    "TrainingConfig",
    "TrainingLineage",
    "configure_determinism",
    "coral_loss",
    "cross_covariance_loss",
    "more_har_objective",
    "physically_plausible_augmentation",
    "predict_model",
    "reconstruct_checkpoint",
    "signal_descriptors",
    "supervised_contrastive_loss",
    "symmetric_consistency_loss",
    "train_source_model",
    "training_config_from_dict",
    "validate_fixed_epoch_checkpoint",
]
