from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")

from inclusive_shift_har.experiments.harth_gravity_hierarchy import (
    _decode_sequence,
    _gravity_geometry_features,
    _prior_calibrate,
)


def test_gravity_geometry_is_fixed_and_finite() -> None:
    window = np.random.default_rng(12).normal(size=(250, 3))
    features = _gravity_geometry_features(window)
    assert features.shape == (30,)
    assert np.isfinite(features).all()


def test_prior_calibration_uses_training_prior_and_is_finite() -> None:
    probabilities = np.asarray([[0.5, 0.5], [0.2, 0.8]], dtype=np.float64)
    calibrated, delta = _prior_calibrate(probabilities, np.asarray([0, 0, 0, 1]))
    assert delta < 0.0
    assert calibrated.shape == probabilities.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert np.isfinite(calibrated).all()


def test_state_decoder_respects_minimum_dwell() -> None:
    probability = np.asarray(
        [[0.95, 0.05], [0.05, 0.95], [0.95, 0.05], [0.95, 0.05]],
        dtype=np.float64,
    )
    transition = np.log(np.asarray([[0.98, 0.02], [0.02, 0.98]], dtype=np.float64))
    decoded = _decode_sequence(probability, transition)
    assert decoded.tolist() == [0, 0, 0, 0]
