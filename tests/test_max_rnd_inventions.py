from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from inclusive_shift_har.evaluation.provenance_corruptions import (
    ProvenanceCorruptionSpec,
    apply_provenance_corruption,
    interpolate_from_validity_mask,
)
from inclusive_shift_har.experiments.active_semantic_gauge_sentinel import (
    _load_config as load_sentinel_config,
)
from inclusive_shift_har.experiments.active_semantic_gauge_sentinel import (
    run_semantic_gauge_sentinel_for_participant,
)
from inclusive_shift_har.experiments.confidence_triggered_gravity_residual import (
    _candidates as ctgr_candidates,
)
from inclusive_shift_har.experiments.confidence_triggered_gravity_residual import (
    _load_config as load_ctgr_config,
)
from inclusive_shift_har.experiments.confidence_triggered_gravity_residual import (
    apply_confidence_triggered_gravity_residual,
)
from inclusive_shift_har.experiments.provenance_mask_robustness import (
    _load_config as load_mask_config,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.gravity_posture_reference import (
    extract_gravity_posture_reference_features,
    gravity_posture_reference_feature_names,
    gravity_reference_time_series,
    gravity_reference_time_series_names,
)


def _signals(seed: int = 81) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    primary = generator.normal(size=(4, 128, 6))
    gravity = generator.normal(size=(4, 128, 3))
    gravity /= np.linalg.norm(gravity, axis=2, keepdims=True)
    return primary, gravity


def test_gravity_reference_features_are_deterministic_finite_and_named() -> None:
    primary, gravity = _signals()
    series = gravity_reference_time_series(primary, gravity, sampling_rate_hz=50.0)
    features = extract_gravity_posture_reference_features(primary, gravity, sampling_rate_hz=50.0)
    replay = extract_gravity_posture_reference_features(
        primary.copy(), gravity.copy(), sampling_rate_hz=50.0
    )
    assert series.shape == (4, 128, len(gravity_reference_time_series_names()))
    assert features.shape == (4, len(gravity_posture_reference_feature_names()))
    assert len(gravity_posture_reference_feature_names()) == len(
        set(gravity_posture_reference_feature_names())
    )
    np.testing.assert_array_equal(features, replay)
    assert np.isfinite(features).all()


def test_gravity_physical_projections_survive_shared_proper_rotation() -> None:
    primary, gravity = _signals()
    angle = 0.61
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = np.concatenate(
        (primary[:, :, :3] @ rotation.T, primary[:, :, 3:] @ rotation.T), axis=2
    )
    original_series = gravity_reference_time_series(primary, gravity, sampling_rate_hz=50.0)
    rotated_series = gravity_reference_time_series(
        rotated, gravity @ rotation.T, sampling_rate_hz=50.0
    )
    np.testing.assert_allclose(
        original_series[:, :, 9:], rotated_series[:, :, 9:], rtol=1e-7, atol=1e-6
    )


def test_ctgr_grid_and_trigger_preserve_confident_base() -> None:
    config = load_ctgr_config(
        Path("configs/experiments/confidence_triggered_gravity_residual_v1.yaml")
    )
    candidates = ctgr_candidates(config)
    assert len(candidates) == 121
    assert candidates[0]["id"] == "base_no_route"
    assert all(candidate["participant_class_weighted"] is True for candidate in candidates)
    base = np.asarray([[0.9, 0.05, 0.05], [0.4, 0.35, 0.25]])
    expert = np.asarray([[0.2, 0.7, 0.1], [0.4, 0.1, 0.5]])
    fused, trigger = apply_confidence_triggered_gravity_residual(
        base, expert, confidence_threshold=0.55, blend_weight=0.75
    )
    np.testing.assert_array_equal(trigger, [False, True])
    np.testing.assert_array_equal(fused[0], base[0])
    np.testing.assert_allclose(fused[1], 0.25 * base[1] + 0.75 * expert[1])
    np.testing.assert_allclose(fused.sum(axis=1), 1.0)


def test_semantic_sentinel_queries_before_label_and_can_swap() -> None:
    load_sentinel_config(Path("configs/experiments/active_semantic_gauge_sentinel_v1.yaml"))
    probability = np.asarray(
        [[0.01, 0.98, 0.01], [0.05, 0.90, 0.05], [0.05, 0.10, 0.85]], dtype=np.float64
    )
    labels = np.asarray([2, 1, 2], dtype=np.int64)
    windows = np.asarray(["a", "b", "c"])
    adjusted, queried, record = run_semantic_gauge_sentinel_for_participant(
        probability,
        labels,
        windows,
        maximum_queries=1,
        minimum_absolute_log_bayes_factor=np.log(3.0),
        minimum_stationary_probability=0.5,
        strategy="active_information",
        seed=7,
    )
    np.testing.assert_array_equal(queried, [True, False, False])
    assert record["decision"] == "swap_sitting_standing"
    np.testing.assert_array_equal(adjusted[:, 1], probability[:, 2])
    changed_unqueried_labels = labels.copy()
    changed_unqueried_labels[1:] = [2, 1]
    _, replay_query, _ = run_semantic_gauge_sentinel_for_participant(
        probability,
        changed_unqueried_labels,
        windows,
        maximum_queries=1,
        minimum_absolute_log_bayes_factor=np.log(3.0),
        minimum_stationary_probability=0.5,
        strategy="active_information",
        seed=7,
    )
    np.testing.assert_array_equal(replay_query, queried)


def test_provenance_mask_never_infers_missingness_from_zero_and_interpolates() -> None:
    load_mask_config(Path("configs/experiments/provenance_mask_robustness_v1.yaml"))
    primary, _ = _signals()
    primary[0, 0, 0] = 0.0
    scale = np.ones(6)
    threshold = np.full(6, 2.0)
    clean = apply_provenance_corruption(
        primary,
        ProvenanceCorruptionSpec("clean", 0.0),
        channel_scale=scale,
        saturation_threshold=threshold,
        seed=3,
    )
    assert clean.validity_mask[0, 0, 0]
    dropped = apply_provenance_corruption(
        primary,
        ProvenanceCorruptionSpec("contiguous_gap", 0.25),
        channel_scale=scale,
        saturation_threshold=threshold,
        seed=3,
    )
    assert dropped.invalid_fraction == 0.25
    reconstructed = interpolate_from_validity_mask(
        dropped.signals,
        dropped.validity_mask,
        training_channel_median=np.zeros(6),
    )
    assert reconstructed.shape == primary.shape
    assert np.isfinite(reconstructed).all()
    for severity, expected_missing_channels in ((1 / 3, 1), (2 / 3, 2)):
        axis_dropout = apply_provenance_corruption(
            primary,
            ProvenanceCorruptionSpec("axis_dropout", severity),
            channel_scale=scale,
            saturation_threshold=threshold,
            seed=3,
        )
        np.testing.assert_array_equal(
            (~axis_dropout.validity_mask).sum(axis=(1, 2)),
            np.full(primary.shape[0], 128 * expected_missing_channels),
        )
    rotated = apply_provenance_corruption(
        primary,
        ProvenanceCorruptionSpec("constrained_rotation", 0.5),
        channel_scale=scale,
        saturation_threshold=threshold,
        seed=3,
    )
    assert rotated.affected_fraction == 1.0
    assert rotated.invalid_fraction == 0.0


@pytest.mark.parametrize(
    "path",
    [
        Path("results/protocol/confidence_triggered_gravity_residual_v1.json"),
        Path("results/protocol/active_semantic_gauge_sentinel_v1.json"),
        Path("results/protocol/provenance_mask_robustness_v1.json"),
    ],
)
def test_max_rnd_protocol_lock_self_hash_and_file_lineage(path: Path) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    claimed = record.pop("record_sha256")
    assert claimed == canonical_json_sha256(record)

    def validate_references(value: object) -> None:
        if isinstance(value, dict):
            if set(("path", "sha256")) <= value.keys():
                referenced = Path(str(value["path"]))
                assert referenced.is_file()
                assert sha256_file(referenced) == value["sha256"]
            for child in value.values():
                validate_references(child)
        elif isinstance(value, list):
            for child in value:
                validate_references(child)

    validate_references(record)
