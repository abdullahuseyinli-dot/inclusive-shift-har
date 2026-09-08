from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
import yaml

import inclusive_shift_har.experiments.fog_raw_cache_integrity as integrity
from inclusive_shift_har.experiments.fog_raw_cache_integrity import (
    _cache_array_sha256,
    _external_array_sha256,
    _segment_compare,
    _write_bytes_create_only,
    independent_annotation_admission,
    independent_gravity,
    independent_magnitude_descriptors,
    independent_outer_assignment,
    independent_physical_runs,
    independent_resample_segment,
    independent_si_channels,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/fog_raw_cache_integrity_v1.yaml"


def _config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_frozen_config_rejects_reconstruction_or_fit_drift() -> None:
    config = _config()
    validate_config(config)
    changed = copy.deepcopy(config)
    changed["reconstruction"]["gravity_cutoff_hz"] = 0.31
    with pytest.raises(ValueError, match="reconstruction"):
        validate_config(changed)
    changed = copy.deepcopy(config)
    changed["resources"]["model_fit_budget"] = 1
    with pytest.raises(ValueError, match="resource"):
        validate_config(changed)


def test_si_conversion_uses_declared_units_and_operation_order() -> None:
    frame = pd.DataFrame(
        {
            "back_acc_x": [1.0],
            "back_acc_y": [0.0],
            "back_acc_z": [-1.0],
            "back_gyro_x": [180.0],
            "back_gyro_y": [0.0],
            "back_gyro_z": [-180.0],
        }
    )
    acceleration, gyroscope = independent_si_channels(frame)
    np.testing.assert_array_equal(acceleration, [[9.80665, 0.0, -9.80665]])
    np.testing.assert_allclose(gyroscope, [[np.pi, 0.0, -np.pi]], rtol=0.0, atol=0.0)


def test_gravity_constant_and_step_response_has_first_sample_state() -> None:
    values = np.asarray(
        [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [3.0, 2.0, 1.0]],
        dtype=np.float64,
    )
    result = independent_gravity(values, sampling_rate_hz=60.0, cutoff_hz=0.30)
    alpha = 1.0 - np.exp(-2.0 * np.pi * 0.30 / 60.0)
    expected_second = values[0] + alpha * (values[1] - values[0])
    np.testing.assert_array_equal(result[0], values[0])
    np.testing.assert_allclose(result[1], expected_second, rtol=0.0, atol=0.0)
    constant = independent_gravity(
        np.repeat(values[:1], 20, axis=0),
        sampling_rate_hz=60.0,
        cutoff_hz=0.30,
    )
    np.testing.assert_array_equal(constant, np.repeat(values[:1], 20, axis=0))


def test_physical_runs_split_nonfinite_backstep_and_strict_gap_only() -> None:
    timestamps = np.asarray(
        [0.0, 0.01, 0.06, 0.110001, 0.12, 0.12, 0.13, 1.0, 1.01],
        dtype=np.float64,
    )
    signals = np.zeros((timestamps.size, 6), dtype=np.float64)
    signals[6, 0] = np.nan
    runs = independent_physical_runs(
        timestamps,
        signals,
        nominal_rate_hz=60.0,
        maximum_gap_factor=3.0,
    )
    assert runs == [(0, 3), (3, 5), (5, 6), (7, 9)]


def test_full_segment_filter_resample_is_not_equivalent_to_window_resets() -> None:
    count = 420
    timestamps = np.arange(count, dtype=np.float64) / 60.0
    total = np.zeros((count, 3), dtype=np.float64)
    total[:, 2] = 9.80665
    total[175:, 0] = 2.0
    gyro = np.zeros_like(total)
    full = independent_resample_segment(
        timestamps=timestamps,
        total_acceleration=total,
        gyroscope=gyro,
        source_rate_hz=60.0,
        target_rate_hz=50.0,
        gravity_cutoff_hz=0.30,
        window_samples=128,
    )
    left = independent_resample_segment(
        timestamps=timestamps[:210],
        total_acceleration=total[:210],
        gyroscope=gyro[:210],
        source_rate_hz=60.0,
        target_rate_hz=50.0,
        gravity_cutoff_hz=0.30,
        window_samples=128,
    )
    right = independent_resample_segment(
        timestamps=timestamps[210:],
        total_acceleration=total[210:],
        gyroscope=gyro[210:],
        source_rate_hz=60.0,
        target_rate_hz=50.0,
        gravity_cutoff_hz=0.30,
        window_samples=128,
    )
    reset_gravity = np.concatenate((left["gravity"], right["gravity"]), axis=0)
    full_gravity = np.asarray(full["gravity"])[: reset_gravity.shape[0]]
    assert not np.allclose(full_gravity, reset_gravity, rtol=0.0, atol=1e-12)


def test_annotation_admission_checks_hidden_source_codes_before_mapping() -> None:
    source_times = np.arange(156, dtype=np.float64) / 60.0
    target_times = np.arange(130, dtype=np.float64) / 50.0
    starts = np.asarray([0], dtype=np.int64)
    labels = np.ones(source_times.size, dtype=np.float64)
    labels[61] = 6.0
    admitted, rejected, mapped = independent_annotation_admission(
        source_timestamps=source_times,
        source_labels=labels,
        target_timestamps=target_times,
        candidate_starts=starts,
        target_rate_hz=50.0,
        window_samples=128,
    )
    assert admitted == []
    assert mapped == []
    assert rejected["mixed_annotation"] == [0]

    labels[61] = 5.0
    admitted, rejected, mapped = independent_annotation_admission(
        source_timestamps=source_times,
        source_labels=labels,
        target_timestamps=target_times,
        candidate_starts=starts,
        target_rate_hz=50.0,
        window_samples=128,
    )
    assert admitted == []
    assert mapped == []
    assert rejected["missing_or_unsupported_annotation"] == [0]


def test_descriptor_contract_materializes_float32_before_float64_reduction() -> None:
    source = np.linspace(0.123456789, 2.123456789, 128 * 6, dtype=np.float64).reshape(1, 128, 6)
    materialized = source.astype(np.float32)
    actual = independent_magnitude_descriptors(materialized)
    values = materialized.astype(np.float64)
    acc = np.sqrt(np.sum(values[:, :, :3] ** 2, axis=2))
    gyro = np.sqrt(np.sum(values[:, :, 3:] ** 2, axis=2))
    expected = np.column_stack(
        (
            np.sqrt(np.mean(acc**2, axis=1)),
            np.std(acc, axis=1),
            np.sqrt(np.mean(gyro**2, axis=1)),
            np.std(gyro, axis=1),
        )
    )
    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(
        actual, independent_magnitude_descriptors(source.astype(np.float32) + np.float32(1e-5))
    )


def test_segment_comparison_detects_poisoned_gravity_hash() -> None:
    timestamps = np.arange(180, dtype=np.float64) / 60.0
    total = np.column_stack(
        (
            np.sin(timestamps),
            np.cos(timestamps),
            np.full(timestamps.size, 9.80665),
        )
    )
    segment = independent_resample_segment(
        timestamps=timestamps,
        total_acceleration=total,
        gyroscope=np.zeros_like(total),
        source_rate_hz=60.0,
        target_rate_hz=50.0,
        gravity_cutoff_hz=0.30,
        window_samples=128,
    )
    first = _segment_compare(
        segment,
        participant="fogstar:001",
        session="fogstar:001:session-001",
        trial="fogstar:001:session-001:recording",
        run_index=0,
        admitted=[0],
        rejected={"missing_or_unsupported_annotation": [], "mixed_annotation": []},
        reference=None,
        source_rate_hz=60.0,
        target_rate_hz=50.0,
        window_samples=128,
    )
    reference = copy.deepcopy(first["actual"])
    matched = _segment_compare(
        segment,
        participant="fogstar:001",
        session="fogstar:001:session-001",
        trial="fogstar:001:session-001:recording",
        run_index=0,
        admitted=[0],
        rejected={"missing_or_unsupported_annotation": [], "mixed_annotation": []},
        reference=reference,
        source_rate_hz=60.0,
        target_rate_hz=50.0,
        window_samples=128,
    )
    assert matched["status"] == "no_defect_detected_under_frozen_contract"
    reference["gravity_sha256"] = "0" * 64
    poisoned = _segment_compare(
        segment,
        participant="fogstar:001",
        session="fogstar:001:session-001",
        trial="fogstar:001:session-001:recording",
        run_index=0,
        admitted=[0],
        rejected={"missing_or_unsupported_annotation": [], "mixed_annotation": []},
        reference=reference,
        source_rate_hz=60.0,
        target_rate_hz=50.0,
        window_samples=128,
    )
    assert poisoned["status"] == "defect_detected"
    assert poisoned["checks"]["gravity_sha256"] is False


def test_hash_conventions_and_fold_assignment_are_explicit() -> None:
    values = np.asarray([[1.0, 2.0]], dtype=np.float32)
    assert _external_array_sha256(values) != _cache_array_sha256(values)
    roster = [f"p{index}" for index in range(7)]
    first = independent_outer_assignment(roster, seed=11, count=5)
    second = independent_outer_assignment(list(reversed(roster)), seed=11, count=5)
    assert first == second
    assert set(first) == set(roster)
    assert set(first.values()) == set(range(5))


def test_auditor_import_graph_excludes_production_transforms_and_estimators() -> None:
    tree = ast.parse(Path(integrity.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    forbidden = (
        "inclusive_shift_har.data.external_har",
        "inclusive_shift_har.preprocessing.features",
        "sklearn",
        "xgboost",
        "torch",
    )
    assert not any(name.startswith(forbidden) for name in imported)


def test_create_only_writer_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "evidence.bin"
    _write_bytes_create_only(path, b"first")
    with pytest.raises(FileExistsError):
        _write_bytes_create_only(path, b"second")
