from __future__ import annotations

import numpy as np

from inclusive_shift_har.models.microstate_posture_graph import (
    MicrostateCodebook,
    MicrostateFeatureSpec,
    compose_mobility_posture_probabilities,
    extract_microstate_posture_features,
    extract_microstate_state_vectors,
    extract_undetrended_microstate_state_vectors,
    fit_microstate_codebook,
    microstate_assignments,
    microstate_posture_feature_names,
    microstate_state_vector_names,
)


def _signals(*, windows: int = 8, seed: int = 41) -> np.ndarray:
    generator = np.random.default_rng(seed)
    time = np.arange(128, dtype=np.float64) / 50.0
    values = generator.normal(0.0, 0.05, size=(windows, 128, 6))
    for index in range(windows):
        frequency = 0.4 + 0.15 * (index % 4)
        values[index, :, :3] += np.sin(2 * np.pi * frequency * time)[:, None]
        values[index, :, 3:] += 0.3 * np.cos(2 * np.pi * frequency * time)[:, None]
    return values


def _fitted_codebook(values: np.ndarray) -> tuple[np.ndarray, MicrostateCodebook]:
    vectors = extract_microstate_state_vectors(
        values,
        sampling_rate_hz=50.0,
        detrend_span_seconds=0.2,
    )
    labels = np.asarray([0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    participants = np.asarray(["1", "1", "2", "2", "3", "3", "4", "4"])
    codebook = fit_microstate_codebook(
        vectors,
        labels,
        participants,
        cluster_count=4,
        detrend_span_seconds=0.2,
        sampling_rate_hz=50.0,
        seed=11,
    )
    return vectors, codebook


def test_microstate_vectors_are_deterministic_rotation_invariant_and_named() -> None:
    values = _signals()
    angle = 0.73
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = np.concatenate((values[:, :, :3] @ rotation.T, values[:, :, 3:] @ rotation.T), axis=2)
    first = extract_microstate_state_vectors(
        values, sampling_rate_hz=50.0, detrend_span_seconds=0.2
    )
    replay = extract_microstate_state_vectors(
        values.copy(), sampling_rate_hz=50.0, detrend_span_seconds=0.2
    )
    transformed = extract_microstate_state_vectors(
        rotated, sampling_rate_hz=50.0, detrend_span_seconds=0.2
    )
    np.testing.assert_array_equal(first, replay)
    np.testing.assert_allclose(first, transformed, rtol=1e-9, atol=1e-9)
    assert first.shape == (8, 128, len(microstate_state_vector_names()))


def test_undetrended_microstate_vectors_preserve_offsets_for_frozen_ablation() -> None:
    values = _signals()
    shifted = values.copy()
    shifted[:, :, 0] += 3.0
    detrended = extract_microstate_state_vectors(
        shifted, sampling_rate_hz=50.0, detrend_span_seconds=0.2
    )
    undetrended = extract_undetrended_microstate_state_vectors(shifted, sampling_rate_hz=50.0)
    assert undetrended.shape == detrended.shape == (8, 128, 6)
    assert np.isfinite(undetrended).all()
    assert float(undetrended[:, :, 0].mean()) > float(detrended[:, :, 0].mean()) + 2.0


def test_codebook_ignores_mobility_rows_and_soft_assignments_normalize() -> None:
    values = _signals()
    vectors, codebook = _fitted_codebook(values)
    changed = vectors.copy()
    changed[:2] += 1000.0
    labels = np.asarray([0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    participants = np.asarray(["1", "1", "2", "2", "3", "3", "4", "4"])
    replay = fit_microstate_codebook(
        changed,
        labels,
        participants,
        cluster_count=4,
        detrend_span_seconds=0.2,
        sampling_rate_hz=50.0,
        seed=11,
    )
    np.testing.assert_allclose(codebook.centers, replay.centers, rtol=0.0, atol=1e-15)
    assignment = microstate_assignments(vectors, codebook, mode="soft")
    hard = microstate_assignments(vectors, codebook, mode="hard")
    np.testing.assert_allclose(assignment.sum(axis=2), 1.0)
    np.testing.assert_array_equal(hard.sum(axis=2), 1.0)
    assert np.any((assignment > 0.0) & (assignment < 1.0))


def test_microstate_graph_features_are_finite_named_and_mask_explicit() -> None:
    values = _signals()
    vectors, codebook = _fitted_codebook(values)
    features = extract_microstate_posture_features(vectors, codebook)
    replay = extract_microstate_posture_features(vectors.copy(), codebook)
    names = microstate_posture_feature_names(4)
    np.testing.assert_array_equal(features, replay)
    assert features.shape == (8, len(names))
    assert len(names) == len(set(names))
    assert np.isfinite(features).all()
    np.testing.assert_array_equal(features[:, -7:], 1.0)

    zero_vectors = extract_microstate_state_vectors(
        np.zeros_like(values), sampling_rate_hz=50.0, detrend_span_seconds=0.2
    )
    zero_without_mask = extract_microstate_posture_features(zero_vectors, codebook)
    explicit_mask = np.ones_like(values, dtype=np.bool_)
    explicit_mask[:, :, 0] = False
    zero_with_mask = extract_microstate_posture_features(
        zero_vectors, codebook, validity_mask=explicit_mask
    )
    np.testing.assert_array_equal(zero_without_mask[:, -7:], 1.0)
    np.testing.assert_array_equal(zero_with_mask[:, -7], 0.0)
    np.testing.assert_array_equal(zero_with_mask[:, -1], 0.0)


def test_microstate_ablation_schema_and_hierarchical_composition() -> None:
    occupancy_only = MicrostateFeatureSpec(
        transition_lags=(1,),
        include_transitions=False,
        include_persistence_dwell=False,
        include_half_direction=False,
    )
    assert microstate_posture_feature_names(4, occupancy_only) == (
        "occupancy_0",
        "occupancy_1",
        "occupancy_2",
        "occupancy_3",
        "observed_channel_fraction_0",
        "observed_channel_fraction_1",
        "observed_channel_fraction_2",
        "observed_channel_fraction_3",
        "observed_channel_fraction_4",
        "observed_channel_fraction_5",
        "fully_observed_timestep_fraction",
    )
    mobility = np.asarray([0.8, 0.1], dtype=np.float64)
    sitting = np.asarray([0.25, 0.75], dtype=np.float64)
    probability = compose_mobility_posture_probabilities(mobility, sitting)
    np.testing.assert_allclose(probability, [[0.8, 0.05, 0.15], [0.1, 0.675, 0.225]])
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)
