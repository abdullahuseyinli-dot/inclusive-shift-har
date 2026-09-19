from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "src")

from inclusive_shift_har.experiments.harth_sage_x import (
    SageModel,
    _participant_weights,
    _standardize,
)


def test_sage_model_returns_back_only_logits_and_paired_targets() -> None:
    rng = np.random.default_rng(4)
    back = torch.from_numpy(rng.normal(size=(4, 250, 3)).astype(np.float32)).transpose(1, 2)
    thigh = torch.from_numpy(rng.normal(size=(4, 250, 3)).astype(np.float32)).transpose(1, 2)
    for arm in (
        "supervised_temporal",
        "same_sensor_ssl",
        "paired_csmr",
        "paired_contrastive",
        "paired_distill",
        "paired_distill_contrastive",
    ):
        model = SageModel(arm)
        result = model(back, thigh if arm.startswith("paired") else None)
        assert result["logits"].shape == (4, 2)
        assert result["embedding"].shape == (4, 64)
        if arm != "supervised_temporal":
            assert result["reconstruction"].shape == (4, 75)
        if arm.startswith("paired"):
            assert result["thigh_embedding"].shape == (4, 64)
        if arm.startswith("paired_distill"):
            assert result["thigh_logits"].shape == (4, 2)
    inference_result = SageModel("paired_csmr")(back, None)
    assert inference_result["logits"].shape == (4, 2)
    assert "thigh_embedding" not in inference_result


def test_weights_are_participant_aware_and_mean_one() -> None:
    labels = np.asarray([0, 0, 1, 1, 1, 0], dtype=np.int64)
    participants = np.asarray(["a", "a", "a", "b", "b", "b"])
    indices = np.arange(labels.size)
    weights = _participant_weights(labels, participants, indices)
    assert np.isclose(weights.mean(), 1.0)
    assert np.isfinite(weights).all()
    assert weights[0] < weights[2]
    assert weights[5] > weights[3]


def test_standardization_is_finite_and_train_only() -> None:
    rng = np.random.default_rng(9)
    train = rng.normal(size=(5, 250, 3)).astype(np.float32)
    test = rng.normal(size=(2, 250, 3)).astype(np.float32)
    transformed_train, transformed_test, mean, scale = _standardize(train, test)
    assert transformed_train.shape == train.shape
    assert transformed_test.shape == test.shape
    assert mean.shape == (1, 1, 3)
    assert scale.shape == (1, 1, 3)
    assert np.isfinite(transformed_train).all()
    assert np.isfinite(transformed_test).all()
