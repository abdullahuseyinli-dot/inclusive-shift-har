"""Training-only normalization and cache invalidation contracts."""

from __future__ import annotations

import numpy as np
import pytest

from inclusive_shift_har.preprocessing.cache import (
    preprocessing_cache_key,
    validate_cache_metadata,
)
from inclusive_shift_har.preprocessing.normalization import ChannelStandardizer


def test_normalization_fits_source_training_rows_only() -> None:
    values = np.asarray([[[1.0, 2.0], [3.0, 2.0]], [[5.0, 2.0], [7.0, 2.0]]])
    stats = ChannelStandardizer.fit(
        values,
        ["1", "2"],
        declared_training_participants={"1", "2"},
        channel_names=("acc_x", "gyro_x"),
        split_manifest_sha256="a" * 64,
    )

    assert stats.training_participants == ("1", "2")
    transformed = stats.transform(values)
    assert np.allclose(transformed.mean(axis=(0, 1)), [0.0, 0.0], atol=1.0e-7)


@pytest.mark.parametrize("forbidden_partition", ["source_validation", "target_sealed"])
def test_normalization_rejects_validation_and_target_rows(forbidden_partition: str) -> None:
    forbidden_subject = "8" if forbidden_partition == "source_validation" else "11"
    with pytest.raises(ValueError, match="declared training partition"):
        ChannelStandardizer.fit(
            np.asarray([[[1.0]], [[2.0]]]),
            ["1", forbidden_subject],
            declared_training_participants={"1", "2"},
            channel_names=("acc_x",),
            split_manifest_sha256="a" * 64,
        )


def test_cache_key_invalidates_on_every_locked_input() -> None:
    base = {
        "source_artifact_sha256": "a" * 64,
        "preprocessing_config_sha256": "b" * 64,
        "split_manifest_sha256": "c" * 64,
        "code_version": "windowing-v1",
    }
    key = preprocessing_cache_key(**base)
    metadata = {"cache_key": key, "schema_version": "1.0.0"}
    assert validate_cache_metadata(metadata, **base)
    variants = [
        {**base, "source_artifact_sha256": "d" * 64},
        {**base, "preprocessing_config_sha256": "d" * 64},
        {**base, "split_manifest_sha256": "d" * 64},
        {**base, "code_version": "windowing-v2"},
    ]
    assert all(not validate_cache_metadata(metadata, **variant) for variant in variants)
