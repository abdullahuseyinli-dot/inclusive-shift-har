from __future__ import annotations

import numpy as np

from inclusive_shift_har.experiments.gravity_invariant_followup import (
    INVARIANT_GROUPS,
    participant_first_weights,
    select_invariant_features,
)


def test_invariant_selector_is_predeclared_and_compact() -> None:
    names = np.asarray(
        [f"native__{group}__stat_{index}" for group in INVARIANT_GROUPS for index in range(15)],
        dtype=np.str_,
    )
    values = np.ones((2, names.size), dtype=np.float64)
    selected, selected_names, indices = select_invariant_features(values, names)
    assert selected.shape == (2, 150)
    assert len(selected_names) == len(indices) == 150


def test_weights_equalize_person_class_cells() -> None:
    labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    people = np.asarray(["a", "a", "a", "b", "b", "b"], dtype=np.str_)
    weights = participant_first_weights(labels, people)
    assert np.isclose(weights.mean(), 1.0)
    assert np.isclose(weights[0], weights[1])
    assert np.isclose(weights[2], weights[3])


def test_selector_rejects_axis_features() -> None:
    names = np.asarray(
        [f"native__{group}__stat_{index}" for group in INVARIANT_GROUPS for index in range(15)]
        + ["native__gravity_unit_x__mean"],
        dtype=np.str_,
    )
    values = np.ones((1, names.size), dtype=np.float64)
    selected, selected_names, _ = select_invariant_features(values, names)
    assert selected.shape == (1, 150)
    assert all("gravity_unit_" not in name for name in selected_names)
