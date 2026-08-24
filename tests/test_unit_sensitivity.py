from __future__ import annotations

import numpy as np
import pytest

from inclusive_shift_har.evaluation.unit_sensitivity import (
    build_acceleration_unit_sensitivity_record,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256


def test_si_conversion_is_numerically_invariant_after_source_only_zscore() -> None:
    generator = np.random.default_rng(17)
    source = generator.normal(size=(8, 128, 6)).astype(np.float32)
    evaluation = generator.normal(size=(3, 128, 6)).astype(np.float32)
    participants = ["1"] * 4 + ["2"] * 4

    record = build_acceleration_unit_sensitivity_record(
        source,
        evaluation,
        participants,
        declared_source_participants={"1", "2"},
        split_manifest_sha256="a" * 64,
        channel_names=("ax", "ay", "az", "gx", "gy", "gz"),
        lineage={"test": True},
    )

    assert record["status"] == "numerically_equivalent_under_training_only_zscore"
    assert record["normalized_tensor_diagnostics"]["within_tolerance"] is True
    assert record["normalized_tensor_diagnostics"]["overall_maximum_absolute_difference"] <= 1e-6
    unhashed = dict(record)
    claimed = unhashed.pop("record_sha256")
    assert claimed == canonical_json_sha256(unhashed)


def test_unit_sensitivity_validates_shape_and_source_identity() -> None:
    source = np.ones((2, 128, 6), dtype=np.float32)
    evaluation = np.ones((1, 128, 6), dtype=np.float32)
    kwargs = {
        "declared_source_participants": {"1", "2"},
        "split_manifest_sha256": "b" * 64,
        "channel_names": ("ax", "ay", "az", "gx", "gy", "gz"),
        "lineage": {},
    }
    with pytest.raises(ValueError, match="align"):
        build_acceleration_unit_sensitivity_record(
            source,
            evaluation,
            ["1"],
            **kwargs,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="finite"):
        invalid = source.copy()
        invalid[0, 0, 0] = np.nan
        build_acceleration_unit_sensitivity_record(
            invalid,
            evaluation,
            ["1", "2"],
            **kwargs,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="non-empty"):
        build_acceleration_unit_sensitivity_record(
            source,
            np.empty((0, 128, 6), dtype=np.float32),
            ["1", "2"],
            **kwargs,  # type: ignore[arg-type]
        )


def test_unit_sensitivity_requires_full_split_hash_and_nonempty_participants() -> None:
    source = np.ones((2, 4, 6), dtype=np.float32)
    evaluation = np.ones((1, 4, 6), dtype=np.float32)
    common = {
        "declared_source_participants": {"1", "2"},
        "channel_names": ("ax", "ay", "az", "gx", "gy", "gz"),
        "lineage": {},
    }

    with pytest.raises(ValueError, match="full SHA-256"):
        build_acceleration_unit_sensitivity_record(
            source,
            evaluation,
            ["1", "2"],
            split_manifest_sha256="not-a-hash",
            **common,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="non-empty strings"):
        build_acceleration_unit_sensitivity_record(
            source,
            evaluation,
            ["1", ""],
            split_manifest_sha256="c" * 64,
            **common,  # type: ignore[arg-type]
        )
