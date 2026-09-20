from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from inclusive_shift_har.experiments.confidence_triggered_gravity_residual import _feature_views
from inclusive_shift_har.experiments.ctgr_routing_validation import (
    alignment,
    numerical_gate,
    unconditional,
)


def test_unconditional_preserves_selected_weight_and_base_mobility_mass() -> None:
    base = np.asarray([[0.2, 0.7, 0.1], [0.8, 0.15, 0.05]])
    expert = np.asarray([[0.2, 0.1, 0.7], [0.8, 0.02, 0.18]])
    candidate = {"id": "fixed-fold-recipe", "blend_weight": 0.5}
    result = unconditional(base, expert, candidate)
    assert np.allclose(result, [[0.2, 0.4, 0.4], [0.8, 0.085, 0.115]])
    assert np.allclose(result[:, 0], base[:, 0])
    assert np.array_equal(unconditional(base, expert, {"id": "base_no_route"}), base)


def test_pairing_reorders_by_identity_and_rejects_duplicates_or_missing_rows() -> None:
    reference = np.asarray(["a", "b", "c"])
    assert alignment(reference, np.asarray(["c", "a", "b"])).tolist() == [1, 2, 0]
    with pytest.raises(ValueError, match="duplicate"):
        alignment(reference, np.asarray(["a", "a", "c"]))
    with pytest.raises(ValueError, match="coverage"):
        alignment(reference, np.asarray(["a", "b", "d"]))


def test_mean_improvement_cannot_override_person_or_standing_harm() -> None:
    effect: dict[str, Any] = {
        "mean_delta": 0.04,
        "bottom_tail_delta": 0.01,
        "worst_paired_harm": -0.01,
        "paired_95_percent_interval": [0.01, 0.08],
        "recall_deltas": {
            "pooled": {"mobility": 0.0, "sitting": 0.03, "standing": 0.02},
            "participant_mean": {"mobility": 0.0, "sitting": 0.03, "standing": 0.02},
        },
    }
    assert numerical_gate(effect, external=True)["passed"]
    harmed = {**effect, "worst_paired_harm": -0.04}
    assert (
        "no_participant_harm_beyond_3pp" in numerical_gate(harmed, external=True)["failed_checks"]
    )
    effect["recall_deltas"]["participant_mean"]["standing"] = -0.01
    assert (
        "participant_mean_standing_recall" in numerical_gate(effect, external=True)["failed_checks"]
    )


def test_computational_batching_does_not_change_window_feature_values() -> None:
    rng = np.random.default_rng(918)
    signals = rng.normal(size=(5, 128, 6))
    gravity = np.zeros((5, 128, 3))
    gravity[:, :, 2] = 1.0
    base, views = _feature_views(signals, gravity, sampling_rate_hz=50.0)
    first, first_views = _feature_views(signals[:2], gravity[:2], sampling_rate_hz=50.0)
    second, second_views = _feature_views(signals[2:], gravity[2:], sampling_rate_hz=50.0)
    np.testing.assert_allclose(base, np.concatenate([first, second]), atol=1e-12, rtol=1e-12)
    for key in views:
        np.testing.assert_allclose(
            views[key],
            np.concatenate([first_views[key], second_views[key]]),
            atol=1e-12,
            rtol=1e-12,
        )
