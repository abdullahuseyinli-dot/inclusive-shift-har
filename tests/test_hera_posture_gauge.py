from __future__ import annotations

import numpy as np

from inclusive_shift_har.experiments.hera_posture_gauge import (
    _participant_gauge,
    _query_order,
    _soft_adjust,
)


def test_soft_gauge_preserves_mobility_and_probability_rows() -> None:
    probabilities = np.array([[0.2, 0.7, 0.1], [0.8, 0.1, 0.1]], dtype=np.float64)
    adjusted = _soft_adjust(probabilities, 0.35)
    assert np.allclose(adjusted[:, 0], probabilities[:, 0])
    assert np.allclose(adjusted.sum(axis=1), 1.0)
    assert np.isfinite(adjusted).all()


def test_query_order_does_not_read_labels() -> None:
    probabilities = np.array([[0.2, 0.7, 0.1], [0.2, 0.1, 0.7], [0.8, 0.1, 0.1]], dtype=np.float64)
    windows = np.asarray(["b", "a", "c"], dtype=np.str_)
    first = _query_order(probabilities, windows, strategy="active_information", seed=271828)
    second = _query_order(probabilities, windows, strategy="active_information", seed=271828)
    assert first == second


def test_participant_gauge_excludes_selected_query_and_can_swap() -> None:
    probabilities = np.array([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]], dtype=np.float64)
    labels = np.asarray([2, 2, 0], dtype=np.int64)
    windows = np.asarray(["a", "b", "c"], dtype=np.str_)
    hard, soft, queried, record = _participant_gauge(
        probabilities,
        labels,
        windows,
        maximum_queries=1,
        strategy="active_information",
        seed=271828,
    )
    assert queried.sum() == 1
    assert record["decision"] == "swap_sitting_standing"
    assert np.allclose(hard[0], [0.1, 0.1, 0.8])
    assert np.allclose(hard[1], [0.1, 0.8, 0.1])
    assert np.allclose(soft.sum(axis=1), 1.0)
