from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import yaml

from inclusive_shift_har.evaluation.cage_corruptions import (
    CageCorruptionSpec,
    apply_cage_corruption,
)


def _signals(seed: int = 450) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    primary = generator.normal(size=(4, 64, 6))
    gravity = generator.normal(size=(4, 64, 3))
    gravity /= np.maximum(np.linalg.norm(gravity, axis=2, keepdims=True), 1e-12)
    return primary, gravity


@pytest.mark.parametrize(
    "kind",
    [
        "clean",
        "constant_bias",
        "random_walk_drift",
        "gain_error",
        "colored_noise",
        "contiguous_gap",
        "axis_dropout",
        "accelerometer_dropout",
        "gyroscope_dropout",
        "gravity_dropout",
        "stuck_at",
        "saturation",
        "orientation_jump",
        "axis_sign_permutation",
        "timestamp_jitter",
        "quantization",
        "combined_bias_noise_gap",
    ],
)
def test_cage_corruptions_are_deterministic_finite_and_non_mutating(kind: str) -> None:
    primary, gravity = _signals()
    original_primary = primary.copy()
    original_gravity = gravity.copy()
    spec = CageCorruptionSpec(kind=kind, severity=0.25)  # type: ignore[arg-type]
    first = apply_cage_corruption(
        primary,
        gravity,
        spec,
        sampling_rate_hz=50.0,
        seed=91,
    )
    second = apply_cage_corruption(
        primary,
        gravity,
        spec,
        sampling_rate_hz=50.0,
        seed=91,
    )
    np.testing.assert_array_equal(primary, original_primary)
    np.testing.assert_array_equal(gravity, original_gravity)
    np.testing.assert_array_equal(first.primary_signals, second.primary_signals)
    np.testing.assert_array_equal(first.gravity, second.gravity)
    np.testing.assert_array_equal(first.validity_mask, second.validity_mask)
    np.testing.assert_array_equal(first.timestamps_seconds, second.timestamps_seconds)
    assert np.isfinite(first.primary_signals).all()
    assert np.isfinite(first.gravity).all()
    assert first.metadata["primary_claim_eligible"] is False
    assert first.metadata["severity_empirically_calibrated"] is False


def test_shared_orientation_jump_preserves_vector_geometry() -> None:
    primary, gravity = _signals()
    result = apply_cage_corruption(
        primary,
        gravity,
        CageCorruptionSpec("orientation_jump", 0.6),
        sampling_rate_hz=50.0,
        seed=7,
    )
    start = primary.shape[1] // 2
    np.testing.assert_allclose(
        np.linalg.norm(result.primary_signals[:, start:, :3], axis=2),
        np.linalg.norm(primary[:, start:, :3], axis=2),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.sum(result.primary_signals[:, start:, :3] * result.gravity[:, start:], axis=2),
        np.sum(primary[:, start:, :3] * gravity[:, start:], axis=2),
        atol=1e-10,
    )


def test_explicit_dropouts_set_validity_and_do_not_infer_from_zero() -> None:
    primary, gravity = _signals()
    primary[0, 0, 0] = 0.0
    clean = apply_cage_corruption(
        primary,
        gravity,
        CageCorruptionSpec("clean", 0.0),
        sampling_rate_hz=50.0,
        seed=1,
    )
    assert clean.validity_mask[0, 0, 0]
    dropped = apply_cage_corruption(
        primary,
        gravity,
        CageCorruptionSpec("gravity_dropout", 1.0),
        sampling_rate_hz=50.0,
        seed=1,
    )
    assert np.all(~dropped.validity_mask[:, :, 6:9])
    assert np.all(dropped.gravity == 0.0)
    assert dropped.metadata["invalid_fraction"] == pytest.approx(1.0 / 3.0)


def test_timestamp_jitter_remains_causal_and_monotonic() -> None:
    primary, gravity = _signals()
    result = apply_cage_corruption(
        primary,
        gravity,
        CageCorruptionSpec("timestamp_jitter", 1.0),
        sampling_rate_hz=50.0,
        seed=9,
    )
    assert np.all(np.diff(result.timestamps_seconds, axis=1) > 0.0)
    assert result.metadata["timestamps_strictly_increasing"] is True
    assert not np.allclose(np.diff(result.timestamps_seconds, axis=1), 1.0 / 50.0)


def test_cage_corruption_rejects_invalid_severity() -> None:
    primary, gravity = _signals()
    with pytest.raises(ValueError, match="severity"):
        apply_cage_corruption(
            primary,
            gravity,
            CageCorruptionSpec("colored_noise", 1.1),
            sampling_rate_hz=50.0,
            seed=1,
        )


def test_cage_corruption_config_is_explicitly_uncalibrated_and_complete() -> None:
    config = cast(
        dict[str, Any],
        yaml.safe_load(
            Path("configs/experiments/cage_har_corruptions_v1.yaml").read_text(encoding="utf-8")
        ),
    )
    assert config["severity_empirically_calibrated"] is False
    assert config["primary_claim_eligible"] is False
    conditions = config["conditions"]
    assert len({item["id"] for item in conditions}) == len(conditions)
    kinds = {item["kind"] for item in conditions}
    assert {
        "constant_bias",
        "random_walk_drift",
        "gain_error",
        "colored_noise",
        "contiguous_gap",
        "axis_dropout",
        "accelerometer_dropout",
        "gyroscope_dropout",
        "gravity_dropout",
        "stuck_at",
        "saturation",
        "orientation_jump",
        "axis_sign_permutation",
        "timestamp_jitter",
        "quantization",
        "combined_bias_noise_gap",
    } <= kinds
    assert config["reporting"]["real_fault_atlas_required_before_robustness_claim"] is True
