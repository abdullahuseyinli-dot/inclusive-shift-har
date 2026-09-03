from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from inclusive_shift_har.data.daghar import daghar_signal_columns
from inclusive_shift_har.evaluation.v2_corruptions import (
    CorruptionSpec,
    apply_sensor_corruption,
    default_corruption_suite,
)
from inclusive_shift_har.experiments.aeon_source_controls import _load_control
from inclusive_shift_har.experiments.crossfit_expert_stack import stack_features
from inclusive_shift_har.experiments.daghar_augmented_geometric_nested import (
    _balanced_domain_weights,
    _deterministic_cell_cap,
)
from inclusive_shift_har.experiments.daghar_external_evaluation import (
    _load_config as load_daghar_external_config,
)
from inclusive_shift_har.experiments.daghar_external_evaluation import (
    external_domain_summary,
)
from inclusive_shift_har.experiments.fixed_oof_fusion import fixed_probability_pools
from inclusive_shift_har.experiments.fuse_reframe_nested import (
    _candidate_summary,
    _lower_fraction_mean,
)
from inclusive_shift_har.experiments.fuse_reframe_router import (
    GATE_FEATURE_NAMES,
    _fit_gate,
    _gate_probability,
    router_features,
    sensor_health_features,
)
from inclusive_shift_har.experiments.geometric_pyramid_corruptions import (
    _load_config as load_geometric_corruption_config,
)
from inclusive_shift_har.experiments.geometric_pyramid_corruptions import (
    train_only_corruption_scales,
)
from inclusive_shift_har.experiments.geometric_pyramid_seed_ensemble import (
    _load_config as load_geometric_seed_ensemble_config,
)
from inclusive_shift_har.experiments.geometric_pyramid_seed_ensemble import (
    equal_seed_probability_mean,
)
from inclusive_shift_har.experiments.multirocket_source import (
    MultiRocketControlConfig,
    _aligned_probabilities,
)
from inclusive_shift_har.experiments.semantic_anchor_reconciliation import (
    reconcile_posture_semantics,
    select_semantic_anchors,
)
from inclusive_shift_har.models import FuSEReFrameHAR, invariant_features_torch
from inclusive_shift_har.models.geometric_spectral_pyramid import (
    extract_geometric_spectral_pyramid_features,
    geometric_spectral_pyramid_feature_names,
)
from inclusive_shift_har.models.gravity_anchored_pyramid import (
    gravity_anchored_feature_names,
    gravity_anchored_feature_views,
)
from inclusive_shift_har.models.robust_multiscale_residual_pyramid import (
    robust_multiscale_feature_names,
    robust_multiscale_feature_views,
    robust_multiscale_signal_views,
)
from inclusive_shift_har.models.spectral_shape import (
    extract_spectral_shape_features,
    spectral_shape_feature_names,
)
from inclusive_shift_har.preprocessing.v2 import V2PhysicalPreprocessor, invariant_features_numpy
from inclusive_shift_har.training.sampling import ParticipantClassSampler
from inclusive_shift_har.training.v2_augmentation import (
    PhysicalAugmentationConfig,
    augment_native_signals,
)
from inclusive_shift_har.training.v2_engine import (
    V2TrainingConfig,
    _allowed_class_set_ids,
    exact_allowed_classes,
    train_fuse_reframe,
    train_fuse_reframe_fixed_evaluation,
)
from inclusive_shift_har.training.v2_objectives import (
    exact_nll,
    participant_cvar,
    reliability_gate_target,
    set_valued_nll,
    vicreg_loss,
)
from inclusive_shift_har.training.weight_averaging import (
    EMATracker,
    SWADTracker,
    UniformAveragingTracker,
    average_state_dicts,
)


def _signals(seed: int = 7, windows: int = 12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(windows, 128, 6)).astype(np.float32)
    values[:, :, :3] *= 0.2
    values[:, :, 3:] *= 0.5
    return values


def test_daghar_signal_column_contract_is_axis_major_and_unique() -> None:
    columns = daghar_signal_columns()
    assert len(columns) == 360
    assert columns[:2] == ("accel-x-0", "accel-x-1")
    assert columns[59:61] == ("accel-x-59", "accel-y-0")
    assert columns[-1] == "gyro-z-59"
    assert len(set(columns)) == len(columns)


def test_external_domain_weights_and_hash_cap_are_balanced_and_replayable() -> None:
    labels = np.asarray([0, 0, 1, 0, 1, 1, 0, 0], dtype=np.int64)
    participants = np.asarray(["a", "a", "a", "b", "b", "b", "c", "c"])
    domains = np.asarray(["x", "x", "x", "x", "x", "x", "y", "y"])
    windows = np.asarray(["8", "1", "4", "2", "7", "3", "6", "5"])
    weights = _balanced_domain_weights(labels, participants, domains)
    assert weights[domains == "x"].sum() == pytest.approx(weights[domains == "y"].sum())
    cap = _deterministic_cell_cap(
        labels,
        participants,
        domains,
        windows,
        maximum=1,
    )
    assert np.flatnonzero(cap).tolist() == [1, 2, 3, 5, 7]
    np.testing.assert_array_equal(
        cap,
        _deterministic_cell_cap(labels, participants, domains, windows, maximum=1),
    )


def test_daghar_external_protocol_and_equal_domain_summary_are_fixed() -> None:
    config = load_daghar_external_config(
        Path("configs/experiments/daghar_external_evaluation_v1.yaml")
    )
    assert config["external_evaluation"]["domains_in_frozen_order"] == [
        "MotionSense",
        "KuHar",
        "WISDM",
    ]
    reports = {
        "small": {"primary": {"mean_participant_macro_f1": 0.9}},
        "large": {"primary": {"mean_participant_macro_f1": 0.3}},
    }
    summary = external_domain_summary(reports)
    assert summary["equal_domain_mean_of_participant_macro_f1"] == pytest.approx(0.6)
    assert summary["worst_domain_mean_participant_macro_f1"] == pytest.approx(0.3)


def _preprocessor(values: np.ndarray) -> V2PhysicalPreprocessor:
    participants = [str(index // 3 + 1) for index in range(values.shape[0])]
    return V2PhysicalPreprocessor.fit(
        values,
        participants,
        declared_training_participants=set(participants),
        split_manifest_sha256="a" * 64,
        channel_names=("ax", "ay", "az", "gx", "gy", "gz"),
    )


def _model(
    preprocessor: V2PhysicalPreprocessor,
    *,
    fusion_mode: str = "gated",
    hierarchical: bool = True,
    normalization: str = "group",
    backbone: str = "multiscale",
    num_source_domains: int | None = None,
) -> FuSEReFrameHAR:
    return FuSEReFrameHAR(
        raw_mean=torch.from_numpy(preprocessor.raw.mean.copy()).float(),
        raw_scale=torch.from_numpy(preprocessor.raw.scale.copy()).float(),
        invariant_mean=torch.from_numpy(preprocessor.invariant_mean.copy()).float(),
        invariant_scale=torch.from_numpy(preprocessor.invariant_scale.copy()).float(),
        clipping_thresholds=torch.from_numpy(preprocessor.clipping_thresholds.copy()).float(),
        hidden_channels=16,
        embedding_dim=24,
        fusion_mode=fusion_mode,  # type: ignore[arg-type]
        hierarchical=hierarchical,
        normalization=normalization,  # type: ignore[arg-type]
        backbone=backbone,  # type: ignore[arg-type]
        num_source_domains=num_source_domains,
    )


def test_numpy_and_torch_invariant_features_match() -> None:
    values = _signals(windows=3)
    observed = invariant_features_torch(torch.from_numpy(values)).detach().numpy()
    expected = invariant_features_numpy(values)
    np.testing.assert_allclose(observed, expected, rtol=1e-5, atol=1e-6)


def test_spectral_shape_features_are_deterministic_finite_and_named() -> None:
    values = _signals(windows=3)
    first = extract_spectral_shape_features(values)
    second = extract_spectral_shape_features(values.copy())
    np.testing.assert_array_equal(first, second)
    assert first.shape == (3, len(spectral_shape_feature_names()))
    assert len(set(spectral_shape_feature_names())) == first.shape[1]
    assert np.isfinite(first).all()


def test_geometric_spectral_pyramid_is_deterministic_cross_rate_and_named() -> None:
    values = _signals(windows=2)
    first = extract_geometric_spectral_pyramid_features(values, sampling_rate_hz=50.0)
    second = extract_geometric_spectral_pyramid_features(values.copy(), sampling_rate_hz=50.0)
    short = extract_geometric_spectral_pyramid_features(values[:, :60], sampling_rate_hz=20.0)
    np.testing.assert_array_equal(first, second)
    assert (
        first.shape
        == short.shape
        == (
            2,
            len(geometric_spectral_pyramid_feature_names()),
        )
    )
    assert len(set(geometric_spectral_pyramid_feature_names())) == first.shape[1]
    assert np.isfinite(first).all() and np.isfinite(short).all()


def test_geometric_corruption_contract_is_fixed_and_train_scales_are_finite() -> None:
    config, specs = load_geometric_corruption_config(
        Path("configs/experiments/geometric_spectral_pyramid_corruptions_v1.yaml")
    )
    assert config["selection_use_prohibited"] is True
    assert specs == default_corruption_suite()
    scale, clipping = train_only_corruption_scales(_signals(), clipping_quantile=0.999)
    assert scale.shape == clipping.shape == (6,)
    assert np.isfinite(scale).all() and np.isfinite(clipping).all()
    assert (scale > 0).all() and (clipping > 0).all()
    robust_config, robust_specs = load_geometric_corruption_config(
        Path("configs/experiments/robust_multiscale_residual_corruptions_v1.yaml")
    )
    assert robust_config["selection_use_prohibited"] is True
    assert robust_specs == specs


def test_geometric_seed_ensemble_contract_is_fixed_equal_and_normalized() -> None:
    config = load_geometric_seed_ensemble_config(
        Path("configs/experiments/geometric_spectral_pyramid_seed_ensemble_v1.yaml")
    )
    assert config["ensemble"]["seeds"] == [11, 23, 47, 89, 131]
    first = np.asarray([[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]], dtype=np.float64)
    second = np.asarray([[0.4, 0.4, 0.2], [0.1, 0.8, 0.1]], dtype=np.float64)
    observed = equal_seed_probability_mean(np.stack((first, second)))
    np.testing.assert_allclose(observed, 0.5 * (first + second))
    np.testing.assert_allclose(observed.sum(axis=1), 1.0)
    with pytest.raises(ValueError, match="invalid"):
        equal_seed_probability_mean(np.stack((first * 2.0, second)))


def test_robust_multiscale_residual_pyramid_is_deterministic_and_named() -> None:
    values = _signals(windows=2)
    signals = robust_multiscale_signal_views(values, sampling_rate_hz=50.0)
    assert signals.keys() == {"base", "denoised", "residual"}
    assert signals["base"].shape == signals["denoised"].shape == signals["residual"].shape
    features = robust_multiscale_feature_views(values, sampling_rate_hz=50.0)
    replay = robust_multiscale_feature_views(values.copy(), sampling_rate_hz=50.0)
    assert features.keys() == {"base", "denoised", "residual", "base_denoised", "tri_view"}
    for view, feature_values in features.items():
        np.testing.assert_array_equal(feature_values, replay[view])
        assert feature_values.shape == (2, len(robust_multiscale_feature_names(view)))
        assert np.isfinite(feature_values).all()
    np.testing.assert_array_equal(
        features["tri_view"],
        np.concatenate((features["base"], features["denoised"], features["residual"]), axis=1),
    )


def test_gravity_anchored_views_reconstruct_total_acceleration_deterministically() -> None:
    primary = _signals(windows=2)
    gravity = np.zeros((2, 128, 3), dtype=np.float32)
    gravity[:, :, 2] = -1.0
    first = gravity_anchored_feature_views(primary, gravity, sampling_rate_hz=50.0)
    second = gravity_anchored_feature_views(primary.copy(), gravity, sampling_rate_hz=50.0)
    assert first.keys() == {"dynamic", "total", "dual"}
    for view in first:
        np.testing.assert_array_equal(first[view], second[view])
        assert first[view].shape == (2, len(gravity_anchored_feature_names(view)))
        assert np.isfinite(first[view]).all()
    np.testing.assert_array_equal(
        first["dual"], np.concatenate((first["dynamic"], first["total"]), axis=1)
    )
    assert not np.array_equal(first["dynamic"], first["total"])


def test_semantic_anchor_reconciliation_uses_only_anchors_and_detects_swap() -> None:
    labels = np.asarray([0, 1, 2, 1, 2, 1, 2], dtype=np.int64)
    participants = np.asarray(["1"] * labels.size)
    windows = np.asarray([f"window-{index}" for index in range(labels.size)])
    anchors = select_semantic_anchors(
        labels,
        participants,
        windows,
        participant_id="1",
        budget_per_class=1,
        seed=17,
    )
    assert anchors.sum() == 2
    assert set(labels[anchors].tolist()) == {1, 2}
    probabilities = np.full((labels.size, 3), 0.05, dtype=np.float64)
    probabilities[:, 0] = np.where(labels == 0, 0.9, 0.05)
    probabilities[:, 1] = np.where(labels == 2, 0.9, 0.05)
    probabilities[:, 2] = np.where(labels == 1, 0.9, 0.05)
    reconciled, decision = reconcile_posture_semantics(
        probabilities,
        labels,
        anchors,
        minimum_log_bayes_factor=np.log(3.0),
    )
    assert decision["decision"] == "swap_sitting_standing"
    np.testing.assert_array_equal(reconciled.argmax(axis=1), labels)


def test_crossfit_stack_features_are_fixed_finite_and_validate_probabilities() -> None:
    first = np.asarray([[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]], dtype=np.float64)
    second = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2]], dtype=np.float64)
    third = np.asarray([[0.5, 0.2, 0.3], [0.3, 0.3, 0.4]], dtype=np.float64)
    features = stack_features((first, second, third))
    assert features.shape == (2, 30)
    assert np.isfinite(features).all()
    with pytest.raises(ValueError, match="probabilities are invalid"):
        stack_features((first * 2.0, second, third))


def test_fixed_probability_pools_are_normalized_and_symmetric() -> None:
    first = np.asarray([[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]], dtype=np.float64)
    second = np.asarray([[0.4, 0.4, 0.2], [0.1, 0.8, 0.1]], dtype=np.float64)
    forward = fixed_probability_pools(first, second)
    reverse = fixed_probability_pools(second, first)
    assert forward.keys() == reverse.keys()
    for name in forward:
        np.testing.assert_allclose(forward[name], reverse[name])
        np.testing.assert_allclose(forward[name].sum(axis=1), 1.0)


def test_aeon_control_config_rejects_unlisted_algorithm() -> None:
    config = Path("configs/experiments/aeon_time_series_controls_v1.yaml")
    with pytest.raises(ValueError, match="not predeclared"):
        _load_control(config, "invented_after_results")


def test_invariant_features_are_rotation_invariant() -> None:
    values = torch.from_numpy(_signals(windows=3))
    angle = torch.tensor(0.71)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = torch.cat(
        (
            values[:, :, :3] @ rotation.T,
            values[:, :, 3:] @ rotation.T,
        ),
        dim=2,
    )
    torch.testing.assert_close(
        invariant_features_torch(values),
        invariant_features_torch(rotated),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("fusion_mode", ["raw", "invariant", "static", "gated"])
@pytest.mark.parametrize("hierarchical", [False, True])
def test_fuse_reframe_outputs_normalized_probabilities(
    fusion_mode: str,
    hierarchical: bool,
) -> None:
    values = _signals(windows=4)
    model = _model(_preprocessor(values), fusion_mode=fusion_mode, hierarchical=hierarchical)
    output = model(torch.from_numpy(values))
    assert output.logits.shape == (4, 3)
    assert output.reconstruction.shape == values.shape
    assert output.gate.shape == (4, 1)
    torch.testing.assert_close(output.logits.exp().sum(dim=1), torch.ones(4))
    assert torch.all((output.gate >= 0) & (output.gate <= 1))


def test_layer_normalized_model_outputs_normalized_probabilities() -> None:
    values = _signals(windows=4)
    model = _model(_preprocessor(values), normalization="layer")
    output = model(torch.from_numpy(values))
    torch.testing.assert_close(output.logits.exp().sum(dim=1), torch.ones(4))


@pytest.mark.parametrize("backbone", ["compact_residual", "inception", "tinyhar"])
def test_research_backbones_preserve_output_contract(backbone: str) -> None:
    values = _signals(windows=2)
    model = _model(
        _preprocessor(values),
        normalization="batch",
        backbone=backbone,
        num_source_domains=2,
    )
    output = model(torch.from_numpy(values), grl_strength=0.5)
    assert output.logits.shape == (2, 3)
    assert output.reconstruction.shape == values.shape
    assert output.domain_logits is not None
    assert output.domain_logits.shape == (2, 2)


def test_native_augmentation_is_deterministic_and_shares_rotation() -> None:
    values = torch.from_numpy(_signals(windows=3))
    config = PhysicalAugmentationConfig(
        amplitude_low=1.0,
        amplitude_high=1.0,
        time_scale_low=1.0,
        time_scale_high=1.0,
        acceleration_noise_sd_g=0.0,
        gyroscope_noise_sd_rad_s=0.0,
        acceleration_bias_g=0.0,
        gyroscope_bias_rad_s=0.0,
        acceleration_drift_g=0.0,
        gyroscope_drift_rad_s=0.0,
        channel_dropout_probability=0.0,
    )
    first = augment_native_signals(
        values,
        generator=torch.Generator().manual_seed(91),
        config=config,
    )
    second = augment_native_signals(
        values,
        generator=torch.Generator().manual_seed(91),
        config=config,
    )
    torch.testing.assert_close(first.signals, second.signals)
    torch.testing.assert_close(first.transform_parameters, second.transform_parameters)
    before = invariant_features_torch(values)
    after = invariant_features_torch(first.signals)
    torch.testing.assert_close(before, after, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize(
    "spec",
    [
        CorruptionSpec("clean", 0.0),
        CorruptionSpec("rotation", 0.5),
        CorruptionSpec("noise", 0.2),
        CorruptionSpec("channel_dropout", 1 / 6),
        CorruptionSpec("temporal_gap", 0.25),
        CorruptionSpec("clipping", 0.5),
    ],
)
def test_sensor_corruptions_are_replayable(spec: CorruptionSpec) -> None:
    values = torch.from_numpy(_signals(windows=3))
    channel_scale = torch.ones(6)
    clipping_thresholds = torch.full((6,), 0.5)
    first = apply_sensor_corruption(
        values,
        spec,
        channel_scale=channel_scale,
        clipping_thresholds=clipping_thresholds,
        seed=19,
    )
    second = apply_sensor_corruption(
        values,
        spec,
        channel_scale=channel_scale,
        clipping_thresholds=clipping_thresholds,
        seed=19,
    )
    torch.testing.assert_close(first.signals, second.signals)
    assert 0 <= first.affected_fraction <= 1
    if spec.name == "clean":
        torch.testing.assert_close(first.signals, values)


def test_default_corruption_suite_has_unique_named_severity_cells() -> None:
    suite = default_corruption_suite()
    assert suite[0] == CorruptionSpec("clean", 0.0)
    assert len({(item.name, item.severity) for item in suite}) == len(suite)


def test_partial_label_loss_does_not_invent_stationary_leaf() -> None:
    log_probabilities = torch.log(torch.tensor([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]]))
    allowed = torch.tensor([[False, True, True], [True, False, False]])
    losses = set_valued_nll(log_probabilities, allowed)
    torch.testing.assert_close(losses, -torch.log(torch.tensor([0.8, 0.7])))
    exact = exact_nll(log_probabilities, torch.tensor([2, 0]))
    torch.testing.assert_close(exact, -torch.log(torch.tensor([0.5, 0.7])))


def test_reliability_oracle_prefers_lower_loss_branch() -> None:
    raw = torch.log(torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]))
    invariant = torch.log(torch.tensor([[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]]))
    allowed = torch.tensor([[True, False, False], [True, False, False]])
    target = reliability_gate_target(raw, invariant, allowed, temperature=0.1)
    assert target[0] > 0.99
    assert target[1] < 0.01
    assert not target.requires_grad


def test_vicreg_is_finite_for_regular_and_singleton_batches() -> None:
    first = torch.randn(8, 12, requires_grad=True)
    second = first.detach() + 0.1 * torch.randn(8, 12)
    loss = vicreg_loss(first, second)
    assert torch.isfinite(loss)
    loss.backward()  # type: ignore[no-untyped-call]
    singleton = vicreg_loss(torch.zeros(1, 3), torch.ones(1, 3))
    assert torch.isfinite(singleton)


def test_partial_label_sampler_ids_preserve_allowed_sets() -> None:
    allowed = np.asarray(
        [[True, False, False], [False, True, True], [False, False, True]],
        dtype=np.bool_,
    )
    assert _allowed_class_set_ids(allowed) == [1, 6, 4]


def test_multirocket_control_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError, match="invalid MultiRocket"):
        MultiRocketControlConfig(n_kernels=0)


def test_multirocket_probability_alignment() -> None:
    class StubClassifier:
        classes_ = np.asarray([2, 0, 1], dtype=np.int64)

        def predict(self, signals: np.ndarray) -> np.ndarray:
            del signals
            return np.asarray([0, 2], dtype=np.int64)

        def predict_proba(self, signals: np.ndarray) -> np.ndarray:
            del signals
            return np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    signals = np.zeros((2, 6, 128), dtype=np.float32)
    probabilities, status = _aligned_probabilities(
        StubClassifier(),  # type: ignore[arg-type]
        signals,
        num_classes=3,
    )
    assert probabilities.argmax(axis=1).tolist() == [0, 2]
    assert status.startswith("unsupported_one_hot")


def test_router_features_are_finite_and_identity_free() -> None:
    signals = _signals(windows=4)
    first = np.asarray(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5], [0.5, 0.3, 0.2]],
        dtype=np.float64,
    )
    second = np.asarray(
        [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.1, 0.7, 0.2], [0.3, 0.4, 0.3]],
        dtype=np.float64,
    )
    health = sensor_health_features(signals, np.ones(6, dtype=np.float64))
    features = router_features(first, second, signals, np.ones(6, dtype=np.float64))
    assert health.shape == (4, 10)
    assert features.shape == (4, len(GATE_FEATURE_NAMES))
    assert np.isfinite(features).all()
    assert all(
        "participant" not in name and "disability" not in name for name in GATE_FEATURE_NAMES
    )


def test_router_fits_only_decisive_oof_rows() -> None:
    features = np.arange(8 * len(GATE_FEATURE_NAMES), dtype=np.float64).reshape(8, -1)
    labels = np.asarray([0, 1, 0, 1, 2, 1, 0, 2], dtype=np.int64)
    prediction_a = np.asarray([0, 0, 0, 2, 2, 0, 1, 2], dtype=np.int64)
    prediction_b = np.asarray([1, 1, 2, 1, 0, 1, 0, 0], dtype=np.int64)
    first = np.eye(3, dtype=np.float64)[prediction_a]
    second = np.eye(3, dtype=np.float64)[prediction_b]
    model, constant, record = _fit_gate(features, labels, first, second, seed=7)
    probabilities = _gate_probability(model, constant, features)
    assert record["outer_label_used"] is False
    assert record["decisive_training_row_count"] > 0
    assert probabilities.shape == (8,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_participant_cvar_uses_highest_loss_participant_tail() -> None:
    losses = torch.tensor([0.1, 0.3, 0.8, 1.0, 0.2, 0.4])
    participants = torch.tensor([0, 0, 1, 1, 2, 2])
    observed = participant_cvar(losses, participants, tail_fraction=1 / 3)
    assert observed.item() == pytest.approx(0.9)


def test_lower_fraction_mean_uses_fractional_boundary_participant() -> None:
    assert _lower_fraction_mean([0.1, 0.2, 0.3, 0.9], 0.3) == pytest.approx((0.1 + 0.2 * 0.2) / 1.2)


def test_nested_candidate_summary_reads_current_selection_epoch_schema() -> None:
    records = []
    for participant, score in (("1", 0.7), ("2", 0.8)):
        records.append(
            {
                "validation_report": {
                    "participants": [{"participant_id": participant, "macro_f1": score}],
                    "window_level_diagnostics": {
                        "confusion_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
                    },
                },
                "configuration": {"learning_rate": 0.001},
                "configuration_sha256": "a" * 64,
                "parameter_count": 10,
                "selection": {"method": "source_validation_best", "epoch": 7},
                "history": [{}] * 9,
                "_result_reference": {"path": "ignored", "sha256": "b" * 64},
            }
        )
    summary = _candidate_summary("candidate", records)
    assert summary["inner_selected_epochs"] == [7, 7]


def test_participant_class_sampler_balances_cells_and_replays() -> None:
    participants = ["1", "1", "1", "2", "2", "2"]
    labels = [0, 0, 1, 0, 1, 1]
    first = ParticipantClassSampler(participants, labels, seed=11, sample_count=12)
    second = ParticipantClassSampler(participants, labels, seed=11, sample_count=12)
    indices = list(first)
    assert indices == list(second)
    cells = [(participants[index], labels[index]) for index in indices]
    assert set(cells) == {("1", 0), ("1", 1), ("2", 0), ("2", 1)}
    assert {cell: cells.count(cell) for cell in set(cells)} == {
        ("1", 0): 3,
        ("1", 1): 3,
        ("2", 0): 3,
        ("2", 1): 3,
    }


def test_weight_averaging_and_trackers() -> None:
    first = {"weight": torch.tensor([1.0]), "count": torch.tensor(2)}
    second = {"weight": torch.tensor([3.0]), "count": torch.tensor(2)}
    averaged = average_state_dicts([first, second])
    torch.testing.assert_close(averaged["weight"], torch.tensor([2.0]))
    assert averaged["count"].item() == 2

    module = torch.nn.Linear(1, 1, bias=False)
    ema = EMATracker(decay=0.5)
    with torch.no_grad():
        module.weight.fill_(1.0)
    ema.update(module)
    with torch.no_grad():
        module.weight.fill_(3.0)
    ema.update(module)
    assert ema.state is not None
    torch.testing.assert_close(ema.state["weight"], torch.tensor([[2.0]]))

    swa = UniformAveragingTracker()
    with torch.no_grad():
        module.weight.fill_(1.0)
    swa.update(module)
    with torch.no_grad():
        module.weight.fill_(3.0)
    swa.update(module)
    assert swa.state is not None
    assert swa.update_count == 2
    torch.testing.assert_close(swa.state["weight"], torch.tensor([[2.0]]))

    swad = SWADTracker(optimum_patience=2, overfit_patience=3, tolerance_rate=1.2)
    for epoch, error in enumerate((0.4, 0.5, 0.45, 0.8, 0.9, 1.0), start=1):
        with torch.no_grad():
            module.weight.fill_(float(epoch))
        swad.update(epoch, error, module)
    result = swad.result()
    assert result is not None
    assert result.start_epoch == 1
    assert result.end_epoch == 3
    assert result.state_count == 3
    torch.testing.assert_close(result.state["weight"], torch.tensor([[2.0]]))


def test_v2_preprocessor_round_trip() -> None:
    values = _signals()
    fitted = _preprocessor(values)
    restored = V2PhysicalPreprocessor.from_dict(fitted.to_dict())
    np.testing.assert_array_equal(restored.raw.mean, fitted.raw.mean)
    np.testing.assert_array_equal(restored.invariant_mean, fitted.invariant_mean)
    np.testing.assert_array_equal(restored.clipping_thresholds, fitted.clipping_thresholds)


def test_v2_engine_runs_target_sealed_and_writes_create_only(tmp_path: Path) -> None:
    train = _signals(seed=101, windows=18)
    validation = _signals(seed=102, windows=6)
    train_labels = np.tile(np.arange(3, dtype=np.int64), 6)
    validation_labels = np.tile(np.arange(3, dtype=np.int64), 2)
    train_participants = [str(index // 3 + 1) for index in range(18)]
    validation_participants = ["8"] * 3 + ["10"] * 3
    preprocessor = V2PhysicalPreprocessor.fit(
        train,
        train_participants,
        declared_training_participants=set(train_participants),
        split_manifest_sha256="b" * 64,
        channel_names=("ax", "ay", "az", "gx", "gy", "gz"),
    )
    output = tmp_path / "run"
    result = train_fuse_reframe(
        train,
        exact_allowed_classes(train_labels),
        train_participants,
        validation,
        validation_labels,
        validation_participants,
        preprocessor=preprocessor,
        config=V2TrainingConfig(
            seed=3,
            epochs=2,
            batch_size=6,
            patience=2,
            minimum_epochs=1,
            hidden_channels=8,
            embedding_dim=8,
            use_physical_augmentation=False,
            reconstruction_weight=0.0,
            weight_averaging="none",
            mixed_precision=False,
        ),
        class_names=("mobility", "sitting", "standing"),
        lineage={
            "target_subject_or_window_records_loaded": False,
            "target_performance_or_prediction_accessed": False,
        },
        output_directory=output,
        device=torch.device("cpu"),
    )
    assert result["status"] == "source_development_complete_target_sealed"
    assert result["target_subject_or_window_records_loaded"] is False
    assert (output / "selected.pt").is_file()
    assert (output / "source_validation_predictions.npz").is_file()
    assert (output / "result.json").is_file()
    with pytest.raises(FileExistsError):
        train_fuse_reframe(
            train,
            exact_allowed_classes(train_labels),
            train_participants,
            validation,
            validation_labels,
            validation_participants,
            preprocessor=preprocessor,
            config=V2TrainingConfig(
                seed=3,
                epochs=1,
                batch_size=6,
                patience=1,
                minimum_epochs=1,
                hidden_channels=8,
                embedding_dim=8,
                use_physical_augmentation=False,
                reconstruction_weight=0.0,
                weight_averaging="none",
                mixed_precision=False,
            ),
            class_names=("mobility", "sitting", "standing"),
            lineage={
                "target_subject_or_window_records_loaded": False,
                "target_performance_or_prediction_accessed": False,
            },
            output_directory=output,
            device=torch.device("cpu"),
        )


def test_fixed_fit_does_not_access_outer_evaluation_during_training(tmp_path: Path) -> None:
    train = _signals(seed=201, windows=9)
    evaluation = _signals(seed=202, windows=3)
    train_labels = np.tile(np.arange(3, dtype=np.int64), 3)
    evaluation_labels = np.arange(3, dtype=np.int64)
    train_participants = ["1"] * 3 + ["2"] * 3 + ["3"] * 3
    evaluation_participants = ["4"] * 3
    preprocessor = V2PhysicalPreprocessor.fit(
        train,
        train_participants,
        declared_training_participants={"1", "2", "3"},
        split_manifest_sha256="c" * 64,
        channel_names=("ax", "ay", "az", "gx", "gy", "gz"),
    )
    result = train_fuse_reframe_fixed_evaluation(
        train,
        exact_allowed_classes(train_labels),
        train_participants,
        evaluation,
        evaluation_labels,
        evaluation_participants,
        preprocessor=preprocessor,
        config=V2TrainingConfig(
            seed=5,
            epochs=1,
            batch_size=3,
            patience=1,
            minimum_epochs=1,
            hidden_channels=8,
            embedding_dim=8,
            use_physical_augmentation=False,
            reconstruction_weight=0.0,
            weight_averaging="none",
            mixed_precision=False,
        ),
        fixed_epochs=1,
        averaging_interval=None,
        class_names=("mobility", "sitting", "standing"),
        lineage={
            "fixed_schedule_derived_without_outer_evaluation": True,
            "target_subject_or_window_records_loaded": False,
            "target_performance_or_prediction_accessed": False,
        },
        output_directory=tmp_path / "fixed",
        device=torch.device("cpu"),
    )
    assert result["outer_evaluation_access_count_during_training"] == 0
    assert result["outer_evaluation_access_count_after_training"] == 1
    assert all(item["outer_evaluation_accessed"] is False for item in result["history"])
