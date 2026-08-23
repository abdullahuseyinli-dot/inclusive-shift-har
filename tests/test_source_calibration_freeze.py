from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from inclusive_shift_har.evaluation.source_calibration import (
    build_source_temperature_calibrator_record,
    load_source_temperature_calibrator_file,
    validate_source_temperature_calibrator_record,
    write_source_temperature_calibrator_new,
)


def _calibrator_record() -> dict[str, object]:
    logits = np.asarray(
        [[5.0, 0.0, -1.0], [0.0, 5.0, -1.0], [-1.0, 0.0, 5.0], [4.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    labels = np.asarray([0, 1, 2, 1], dtype=np.int64)
    return build_source_temperature_calibrator_record(
        logits,
        labels,
        ["source-window-1", "source-window-2", "source-window-3", "source-window-4"],
        fit_partition="source_validation",
        checkpoint_sha256="a" * 64,
        training_configuration_sha256="b" * 64,
        split_manifest_sha256="c" * 64,
        class_names=("mobility", "sitting", "standing"),
    )


def test_source_temperature_calibrator_is_create_only_and_lineage_bound(
    tmp_path: Path,
) -> None:
    record = _calibrator_record()
    result = write_source_temperature_calibrator_new(
        record,
        tmp_path / "calibrator.json",
        allowed_root=tmp_path,
    )
    assert len(result["file_sha256"]) == 64
    loaded, calibrator = load_source_temperature_calibrator_file(
        tmp_path / "calibrator.json",
        expected_checkpoint_sha256="a" * 64,
        expected_training_configuration_sha256="b" * 64,
        expected_split_manifest_sha256="c" * 64,
    )
    assert loaded["fit_partition"] == "source_validation"
    assert calibrator.sample_count == 4
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_source_temperature_calibrator_new(
            record,
            tmp_path / "calibrator.json",
            allowed_root=tmp_path,
        )


def test_temperature_calibration_rejects_target_partition() -> None:
    logits = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    labels = np.asarray([0, 1], dtype=np.int64)
    with pytest.raises(PermissionError, match="source_validation"):
        build_source_temperature_calibrator_record(
            logits,
            labels,
            ["target-window-1", "target-window-2"],
            fit_partition="target_sealed",
            checkpoint_sha256="a" * 64,
            training_configuration_sha256="b" * 64,
            split_manifest_sha256="c" * 64,
            class_names=("mobility", "sitting"),
        )


def test_temperature_calibrator_tamper_breaks_self_hash() -> None:
    tampered = deepcopy(_calibrator_record())
    tampered["temperature"] = float(cast(float, tampered["temperature"])) + 0.1
    with pytest.raises(ValueError, match="self-hash"):
        validate_source_temperature_calibrator_record(tampered)
