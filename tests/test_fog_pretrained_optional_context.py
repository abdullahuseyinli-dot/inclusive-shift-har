from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from inclusive_shift_har.experiments import fog_pretrained_optional_context as optional


def test_native_history_grid_is_right_aligned_30hz_without_future() -> None:
    grid = optional._native_history_grid(20.0)
    assert grid.shape == (300,)
    assert grid[0] == pytest.approx(10.0)
    assert grid[-1] == pytest.approx(20.0 - 1.0 / 30.0)
    np.testing.assert_allclose(np.diff(grid), 1.0 / 30.0, rtol=0.0, atol=2e-15)
    assert np.all(grid < 20.0)


def test_native_interpolation_has_no_extrapolation_and_preserves_axes() -> None:
    timestamps = np.arange(0, 12 * 60, dtype=np.float64) / 60.0
    acceleration = np.column_stack((timestamps, 2 * timestamps, -timestamps))
    run = optional.NativeSensorRun(timestamps=timestamps, total_acceleration_g=acceleration)
    grid = optional._native_history_grid(11.0)
    result = optional._interpolate_history(run, grid)
    np.testing.assert_allclose(result[:, 0], grid, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(result[:, 1], 2 * grid, rtol=0.0, atol=4e-14)
    np.testing.assert_allclose(result[:, 2], -grid, rtol=0.0, atol=2e-14)
    with pytest.raises(ValueError, match="extrapolate"):
        optional._interpolate_history(run, optional._native_history_grid(9.0))


def test_optional_embedding_scaler_uses_only_real_outer_training_history() -> None:
    embeddings = np.asarray([[1.0, 5.0], [3.0, 5.0], [100.0, 100.0], [9.0, 7.0]], dtype=np.float32)
    full = np.asarray([True, True, False, True])
    training = np.asarray([True, True, True, False])
    transformed, mean, scale = optional.standardized_optional_embedding(embeddings, full, training)
    np.testing.assert_array_equal(mean, [2.0, 5.0])
    np.testing.assert_array_equal(scale, [1.0, 1.0])
    np.testing.assert_array_equal(transformed[0], [-1.0, 0.0])
    np.testing.assert_array_equal(transformed[1], [1.0, 0.0])
    np.testing.assert_array_equal(transformed[2], [0.0, 0.0])
    np.testing.assert_array_equal(transformed[3], [7.0, 2.0])


def test_headroom_oracle_is_diagnostic_and_counts_only_strict_improvements() -> None:
    labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    people = np.asarray(["A", "A", "A", "B", "B", "B"], dtype=np.str_)
    probabilities = np.asarray(
        [
            [0.2, 0.7, 0.1],
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ],
        dtype=np.float64,
    )
    result = optional.prospective_motion_headroom(
        labels=labels,
        participant_ids=people,
        l9v_probabilities=probabilities,
        editable_mask=np.ones(6, dtype=np.bool_),
        roster=["A", "B"],
    )
    assert result["diagnostic_label_oracle_only"] is True
    assert result["not_model_performance"] is True
    assert result["editable_rows"] == 6
    assert result["maximum_possible_strict_participant_wins"] == 1
    assert [row["strict_improvement_possible"] for row in result["rows"]] == [True, False]


def test_frozen_extractor_rejects_wrong_parameter_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    class TinyComplete(torch.nn.Module):
        def __init__(self, **_kwargs: Any) -> None:
            super().__init__()
            self.feature_extractor = torch.nn.Conv1d(3, 4, 1)

    class Module:
        Resnet = TinyComplete

    monkeypatch.setattr(optional, "_load_harnet_module", lambda _root: Module())
    with pytest.raises(ValueError, match="parameter count"):
        optional.instantiate_harnet_extractor(
            source_root=Path("unused"), seed=11, checkpoint_path=None
        )
