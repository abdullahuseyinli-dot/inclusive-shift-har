from __future__ import annotations

import numpy as np
import pytest

from inclusive_shift_har.preprocessing import ChannelStandardizer


def test_normalization_uses_exact_declared_training_participants() -> None:
    windows = np.arange(4 * 8 * 2, dtype=np.float32).reshape(4, 8, 2)
    standardizer = ChannelStandardizer.fit(
        windows,
        ["p1", "p1", "p2", "p2"],
        declared_training_participants={"p1", "p2"},
        split_manifest_sha256="f" * 64,
        channel_names=("acc_x", "gyro_x"),
    )
    transformed = standardizer.transform(windows)
    assert np.allclose(transformed.mean(axis=(0, 1)), 0.0, atol=1e-6)
    assert np.allclose(transformed.std(axis=(0, 1)), 1.0, atol=1e-6)
    assert standardizer.training_participants == ("p1", "p2")


def test_normalization_rejects_undeclared_or_missing_participants() -> None:
    with pytest.raises(ValueError, match="differ"):
        ChannelStandardizer.fit(
            np.ones((2, 8, 2), dtype=np.float32),
            ["p1", "target"],
            declared_training_participants={"p1"},
            split_manifest_sha256="f" * 64,
            channel_names=("acc_x", "gyro_x"),
        )


def test_normalization_round_trip_state() -> None:
    windows = np.arange(2 * 4 * 2, dtype=np.float32).reshape(2, 4, 2)
    fitted = ChannelStandardizer.fit(
        windows,
        ["p1", "p1"],
        declared_training_participants={"p1"},
        split_manifest_sha256="0" * 64,
        channel_names=("a", "b"),
    )
    reconstructed = ChannelStandardizer.from_dict(fitted.to_dict())
    assert np.array_equal(fitted.transform(windows), reconstructed.transform(windows))
