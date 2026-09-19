from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")

from inclusive_shift_har.experiments.harth_crsp_back import (
    _base_features,
    _fold_assignment,
    _physics_features,
    _rich_features,
    _temporal_features,
    _train_feature_map,
)


def test_feature_maps_have_fixed_shapes_and_finite_values() -> None:
    window = np.random.default_rng(5).normal(size=(250, 3))
    assert _base_features(window).shape == (24,)
    assert _rich_features(window).shape == (161,)
    assert _temporal_features(window).shape == (192,)
    assert _physics_features(window).shape == (14,)
    for function in (_base_features, _rich_features, _temporal_features, _physics_features):
        assert np.isfinite(function(window)).all()


def test_fold_assignment_is_deterministic_and_exclusive() -> None:
    participants = ["S006", "S008", "S009", "S010", "S012", "S013", "S014", "S015", "S016", "S017"]
    first = _fold_assignment(participants)
    second = _fold_assignment(participants)
    assert first == second
    assert set(first) == set(participants)
    assert set(first.values()) == {0, 1, 2, 3, 4}


def test_paired_map_uses_back_features_at_inference() -> None:
    rng = np.random.default_rng(7)
    train_back = rng.normal(size=(30, 8))
    test_back = rng.normal(size=(6, 8))
    train_thigh = rng.normal(size=(30, 5))
    test_thigh = rng.normal(size=(6, 5))
    train_out, test_out, receipt = _train_feature_map(
        train_back,
        test_back,
        "paired_reconstruction",
        train_thigh=train_thigh,
        test_thigh=test_thigh,
    )
    assert train_out.shape == (30, 13)
    assert test_out.shape == (6, 13)
    assert receipt["map"] == "back_to_thigh_ridge"
    altered_test_thigh = test_thigh + 1000.0
    _, altered_out, _ = _train_feature_map(
        train_back,
        test_back,
        "paired_reconstruction",
        train_thigh=train_thigh,
        test_thigh=altered_test_thigh,
    )
    np.testing.assert_allclose(test_out, altered_out)
