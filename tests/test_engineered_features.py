from __future__ import annotations

import numpy as np

from inclusive_shift_har.preprocessing import extract_engineered_features

CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")


def test_engineered_feature_schema_is_stable_and_finite() -> None:
    windows = np.zeros((2, 128, 6), dtype=np.float32)
    windows[1, :, 0] = np.sin(np.linspace(0, 8 * np.pi, 128))
    batch = extract_engineered_features(windows, channel_names=CHANNELS)
    assert batch.values.shape == (2, 80)
    assert len(batch.names) == 80
    assert len(set(batch.names)) == 80
    assert np.isfinite(batch.values).all()
    assert "acc_x__dominant_frequency_bin_fraction" in batch.names
    assert "acc_x__acc_y__correlation" in batch.names


def test_feature_extraction_is_byte_deterministic() -> None:
    windows = np.random.default_rng(9).normal(size=(3, 128, 6)).astype(np.float32)
    first = extract_engineered_features(windows, channel_names=CHANNELS)
    second = extract_engineered_features(windows, channel_names=CHANNELS)
    assert first.names == second.names
    assert first.values.tobytes() == second.values.tobytes()


def test_native_nine_channel_schema_adds_gravity_summaries() -> None:
    names = (*CHANNELS, "gravity_x", "gravity_y", "gravity_z")
    windows = np.random.default_rng(10).normal(size=(2, 128, 9)).astype(np.float32)
    batch = extract_engineered_features(windows, channel_names=names)
    assert batch.values.shape == (2, 120)
    assert len(batch.names) == len(set(batch.names)) == 120
    assert "gravity_norm__p95" in batch.names
    assert "gravity_x__gravity_y__correlation" in batch.names
    assert np.isfinite(batch.values).all()
