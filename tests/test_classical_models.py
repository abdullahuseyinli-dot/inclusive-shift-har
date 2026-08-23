from __future__ import annotations

from pathlib import Path

import numpy as np

from inclusive_shift_har.models.classical import (
    ClassicalConfig,
    fit_classical_model,
    load_classical_checkpoint,
    predict_classical_model,
    save_classical_checkpoint_create_only,
)

CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
CLASSES = ("mobility", "sitting", "standing")


def _separable_windows(count: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    labels = np.arange(count, dtype=np.int64) % 3
    windows = generator.normal(scale=0.1, size=(count, 128, 6)).astype(np.float32)
    windows[:, :, 0] += labels[:, None]
    return windows, labels


def test_logistic_baseline_probability_alignment_and_checkpoint(tmp_path: Path) -> None:
    train_windows, train_labels = _separable_windows(30, seed=1)
    validation_windows, validation_labels = _separable_windows(12, seed=2)
    fitted = fit_classical_model(
        train_windows,
        train_labels,
        ["p1"] * 15 + ["p2"] * 15,
        config=ClassicalConfig("logistic_regression", num_classes=3, seed=5),
        channel_names=CHANNELS,
        lineage={"split_manifest_sha256": "a" * 64},
    )
    _, probabilities, report = predict_classical_model(
        fitted,
        validation_windows,
        validation_labels,
        ["v1"] * 6 + ["v2"] * 6,
        class_names=CLASSES,
    )
    assert probabilities.shape == (12, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert report["participant_count"] == 2
    record = save_classical_checkpoint_create_only(fitted, tmp_path / "model.pkl")
    assert len(record["sha256"]) == 64
    restored = load_classical_checkpoint(tmp_path / "model.pkl")
    _, restored_probabilities, _ = predict_classical_model(
        restored,
        validation_windows,
        validation_labels,
        ["v1"] * 6 + ["v2"] * 6,
        class_names=CLASSES,
    )
    assert np.array_equal(probabilities, restored_probabilities)
