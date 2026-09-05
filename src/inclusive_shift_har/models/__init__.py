"""Model registry and architecture specifications."""

from inclusive_shift_har.models.baselines import (
    CNN1D,
    BiLSTM,
    CompactResidualHAR,
    DeepConvLSTM,
    JointCNNBiLSTM,
    StaticDualBranchHAR,
    TinyHAR,
    build_baseline,
)
from inclusive_shift_har.models.cage_har import (
    AdvantagePrediction,
    CageContext,
    CageRoutingResult,
    RidgeAdvantageModel,
    SemanticGaugeResult,
    apply_cage_trust_region,
    apply_soft_semantic_gauge,
    build_advantage_router_features,
    cage_expert_reliability,
    counterfactual_log_loss_advantage,
    extract_cage_gauge_context,
    extract_cage_sensor_health,
    fit_ridge_advantage_model,
    participant_jackknife_advantage_prediction,
    select_semantic_gauge_query,
)
from inclusive_shift_har.models.common import HAROutput, trainable_parameter_count
from inclusive_shift_har.models.domain_adversarial import (
    DANNCompactResidualHAR,
    gradient_reverse,
)
from inclusive_shift_har.models.fuse_reframe import (
    BackboneMode,
    FunctionalHierarchy,
    FuSEReFrameHAR,
    FuSEReFrameOutput,
    invariant_features_torch,
)
from inclusive_shift_har.models.geometric_spectral_pyramid import (
    extract_geometric_spectral_pyramid_features,
    geometric_spectral_pyramid_feature_names,
)
from inclusive_shift_har.models.gravity_anchored_pyramid import (
    gravity_anchored_feature_names,
    gravity_anchored_feature_views,
)
from inclusive_shift_har.models.gravity_posture_reference import (
    extract_gravity_posture_reference_features,
    gravity_posture_reference_feature_names,
    gravity_reference_time_series,
    gravity_reference_time_series_names,
)
from inclusive_shift_har.models.legacy_models import build_exact_legacy_model
from inclusive_shift_har.models.microstate_posture_graph import (
    MicrostateCodebook,
    MicrostateFeatureSpec,
    compose_mobility_posture_probabilities,
    extract_microstate_posture_features,
    extract_microstate_state_vectors,
    extract_undetrended_microstate_state_vectors,
    fit_microstate_codebook,
    microstate_posture_feature_names,
    microstate_state_vector_names,
)
from inclusive_shift_har.models.more_har import MoReHAR
from inclusive_shift_har.models.robust_multiscale_residual_pyramid import (
    robust_multiscale_feature_names,
    robust_multiscale_feature_views,
    robust_multiscale_signal_views,
)

__all__ = [
    "CNN1D",
    "AdvantagePrediction",
    "BackboneMode",
    "BiLSTM",
    "CageContext",
    "CageRoutingResult",
    "CompactResidualHAR",
    "DANNCompactResidualHAR",
    "DeepConvLSTM",
    "FuSEReFrameHAR",
    "FuSEReFrameOutput",
    "FunctionalHierarchy",
    "HAROutput",
    "JointCNNBiLSTM",
    "MicrostateCodebook",
    "MicrostateFeatureSpec",
    "MoReHAR",
    "RidgeAdvantageModel",
    "SemanticGaugeResult",
    "StaticDualBranchHAR",
    "TinyHAR",
    "apply_cage_trust_region",
    "apply_soft_semantic_gauge",
    "build_advantage_router_features",
    "build_baseline",
    "build_exact_legacy_model",
    "cage_expert_reliability",
    "compose_mobility_posture_probabilities",
    "counterfactual_log_loss_advantage",
    "extract_cage_gauge_context",
    "extract_cage_sensor_health",
    "extract_geometric_spectral_pyramid_features",
    "extract_gravity_posture_reference_features",
    "extract_microstate_posture_features",
    "extract_microstate_state_vectors",
    "extract_undetrended_microstate_state_vectors",
    "fit_microstate_codebook",
    "fit_ridge_advantage_model",
    "geometric_spectral_pyramid_feature_names",
    "gradient_reverse",
    "gravity_anchored_feature_names",
    "gravity_anchored_feature_views",
    "gravity_posture_reference_feature_names",
    "gravity_reference_time_series",
    "gravity_reference_time_series_names",
    "invariant_features_torch",
    "microstate_posture_feature_names",
    "microstate_state_vector_names",
    "participant_jackknife_advantage_prediction",
    "robust_multiscale_feature_names",
    "robust_multiscale_feature_views",
    "robust_multiscale_signal_views",
    "select_semantic_gauge_query",
    "trainable_parameter_count",
]
