from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inclusive_shift_har.data.external_har import ExternalHARWindows, SourceReceipt
from inclusive_shift_har.evaluation.external_statistics import (
    ParticipantMetricInputs,
    seed_evidence,
)
from inclusive_shift_har.experiments import hierarchical_posture as runner
from inclusive_shift_har.experiments.hierarchical_posture import posture_advancement_gate
from inclusive_shift_har.models.classical import ClassicalConfig, _build_estimator
from inclusive_shift_har.models.hierarchical_posture import (
    VARIANTS,
    fit_posture_forest,
    participant_class_weights,
)
from inclusive_shift_har.preprocessing.features import extract_engineered_features


def test_participant_class_mass_is_equal_even_with_missing_classes() -> None:
    labels = np.array([0, 0, 1, 2, 0, 0, 0, 1], dtype=np.int64)
    participants = np.array(["a"] * 4 + ["b"] * 4)
    weights = participant_class_weights(labels, participants)
    assert weights.mean() == pytest.approx(1)
    assert weights[:4].sum() == pytest.approx(weights[4:].sum())
    for person in ("a", "b"):
        masses = [
            weights[(participants == person) & (labels == label)].sum()
            for label in np.unique(labels[participants == person])
        ]
        np.testing.assert_allclose(masses, masses[0])


@pytest.mark.parametrize("variant", VARIANTS)
def test_fixed_tree_budget_probability_order_and_label_free_inference(variant: str) -> None:
    features = np.random.default_rng(11).normal(size=(60, 8))
    labels = np.tile(np.arange(3), 20)
    participants = np.repeat(["a", "b", "c", "d"], 15)
    model = fit_posture_forest(features, labels, participants, variant=variant, seed=11)
    prediction = model.predict(features[:6])
    assert prediction.shape == (6, 3)
    np.testing.assert_allclose(prediction.sum(axis=1), 1)
    assert model.size_summary()["tree_count"] == 500
    assert set(inspect.signature(model.predict).parameters) == {"features"}
    np.testing.assert_array_equal(prediction, model.predict(features[:6].copy()))
    if variant == "RandomForest-6ch":
        base = _build_estimator(ClassicalConfig("random_forest", num_classes=3, seed=11))
        base.fit(features, labels)
        np.testing.assert_array_equal(prediction, base.predict_proba(features[:6]))
    if variant == "PB-HPF-shuffled-posture":
        assert model.shuffled_training_labels > 0


@pytest.mark.parametrize("channels", [6, 9])
def test_cached_window_features_equal_partition_local_features(channels: int) -> None:
    signals = np.random.default_rng(12).normal(size=(20, 128, channels)).astype(np.float32)
    names = tuple(f"channel_{index}" for index in range(channels))
    selected = np.arange(20) % 3 != 0
    full = extract_engineered_features(signals, channel_names=names)
    local = extract_engineered_features(signals[selected], channel_names=names)
    np.testing.assert_array_equal(full.values[selected], local.values)
    assert full.names == local.names


@pytest.mark.parametrize("candidate_correct", [False, True])
def test_advancement_gate_is_not_replaced_by_best_ablation(candidate_correct: bool) -> None:
    labels = np.tile(np.arange(3), 12)
    inputs = ParticipantMetricInputs(
        "fog_star_v3",
        ("mobility", "sitting", "standing"),
        labels,
        np.repeat([f"p{i:02d}" for i in range(12)], 3),
    )
    correct = np.eye(3)[labels] * 0.9 + 0.1 / 3
    wrong = np.eye(3)[(labels + 1) % 3] * 0.9 + 0.1 / 3
    probabilities = {
        seed: {
            name: correct
            if name == "PB-RF-D9" or (candidate_correct and name == "PB-HPF")
            else wrong
            for name in VARIANTS
        }
        for seed in (11, 23, 47)
    }
    statistics, _ = seed_evidence(inputs, probabilities)
    gate = posture_advancement_gate(statistics)
    assert gate["all_advancement_gates_passed"] is candidate_correct
    assert gate["comparisons"]["PB-RF-D9"]["input_tuple_matched"] is False
    if not candidate_correct:
        assert gate["decision"].startswith("STOP_CANDIDATE_EXPANSION")


class SyntheticForest:
    """Training seam only; runner serialization and statistics stay real."""

    def __init__(self, participants: np.ndarray[Any, Any]) -> None:
        self.training_participants = tuple(np.unique(participants).tolist())

    def predict(self, features: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.tile([0.6, 0.2, 0.2], (len(features), 1))

    def size_summary(self) -> dict[str, Any]:
        return {"synthetic_training_seam": True}

    def mechanism(self, _features: np.ndarray[Any, Any]) -> dict[str, Any]:
        return {"synthetic_training_seam": True}


def test_posture_suite_keeps_every_seed_fold_prediction_and_independent_partitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = np.tile(np.arange(3), 12)
    people = np.repeat([f"p{i:02d}" for i in range(12)], 3)
    values = np.random.default_rng(93).normal(size=(36, 128, 9)).astype(np.float32)
    data = ExternalHARWindows(
        dataset_id="synthetic_posture_suite",
        channel_lane="derived-gravity-9ch",
        sampling_rate_hz=50,
        signals=values[:, :, :6],
        gravity=values[:, :, 6:],
        labels=labels,
        participant_ids=people,
        session_ids=people,
        trial_ids=people,
        window_ids=np.array([f"w{i}" for i in range(36)]),
        receipts=(
            SourceReceipt(
                "synthetic_posture_suite", "synthetic://no-source-data", None, 1, 1, "0" * 64
            ),
        ),
    )
    monkeypatch.setattr(
        runner,
        "fit_posture_forest",
        lambda _features, _labels, participants, **_kwargs: SyntheticForest(participants),
    )
    result, probabilities = runner.evaluate_posture_forests(data, output=tmp_path)
    assert result["seeds"] == [11, 23, 47] and len(result["fold_records"]) == 90
    assert result["advancement_gate"]["all_advancement_gates_passed"] is False
    assert len(list(tmp_path.glob("*__predictions.npz"))) == 15
    for record in result["fold_records"]:
        assert not set(record["training_participants"]) & set(record["evaluation_participants"])
    for seed in (11, 23, 47):
        for name in VARIANTS:
            assert probabilities[f"seed-{seed}__{name}"].shape == (36, 3)
