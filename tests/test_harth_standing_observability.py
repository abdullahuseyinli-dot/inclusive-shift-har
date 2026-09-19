import numpy as np

from inclusive_shift_har.experiments.harth_standing_observability import (
    _choose_gate_threshold,
    _gate_prediction,
)


def test_gate_only_overrides_sitting_back_decisions() -> None:
    back = np.array([0, 1, 0, 1], dtype=np.int64)
    thigh = np.array([[0.1, 0.9], [0.1, 0.9], [0.8, 0.2], [0.1, 0.9]])
    assert np.array_equal(_gate_prediction(back, thigh, 0.8), np.array([1, 1, 0, 1]))


def test_gate_threshold_is_train_only_and_limits_sitting_loss() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    back = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]]
    )
    thigh = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.4, 0.6], [0.3, 0.7], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]
    )
    selected = _choose_gate_threshold(labels, back, thigh)
    assert 0.0 <= selected["threshold"] <= 1.0
    assert selected["sitting_recall_loss"] <= 0.02 + 1.0e-12
