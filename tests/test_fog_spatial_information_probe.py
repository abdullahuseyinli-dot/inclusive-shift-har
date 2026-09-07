from __future__ import annotations

import copy
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
import yaml

import inclusive_shift_har.experiments.fog_spatial_information_probe as spatial
from inclusive_shift_har.data.external_har import resample_physical_segment
from inclusive_shift_har.experiments.fog_rf_feature_weight_factorial import (
    _array_sha256,
    method_report,
)
from inclusive_shift_har.experiments.fog_spatial_information_probe import (
    analyse,
    apply_back_fallback,
    assert_b0_replay,
    availability_mask_from_rows,
    fold_masks,
    make_feature_blocks,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/fog_spatial_information_probe_v1.yaml"


def _config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_frozen_config_rejects_scientific_and_resource_drift() -> None:
    config = _config()
    validate_config(config)
    mutations = (
        ("availability", "fixed_sensor", "ankleR"),
        ("availability", "timestamp_atol_seconds", 1e-5),
        ("estimator", "n_estimators", 501),
        ("estimator", "monotonic_cst", [0] * 80),
        ("gate", "minimum_participant_wins", 13),
        ("resources", "maximum_fit_attempts", 26),
        ("claims", "standalone_ankle_replacement", True),
    )
    for section, key, value in mutations:
        changed = copy.deepcopy(config)
        changed[section][key] = value
        with pytest.raises(ValueError, match="config changed"):
            validate_config(changed)


def test_availability_contract_has_dedicated_hash_and_rejects_ambiguity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "window_id": f"w{index}",
            "ankle_finite_run_contains_window": True,
            "ankle_own_continuous_50hz_grid_matches_all128_times": index < 1817,
            "ankle_matching_grid_run_indices": [0] if index < 1817 else [],
        }
        for index in range(1939)
    ]
    expected = np.arange(1939) < 1817
    monkeypatch.setattr(spatial, "MASK_SHA256", spatial._availability_array_sha256(expected))
    actual = availability_mask_from_rows(rows)
    assert np.array_equal(actual, expected)
    assert spatial._availability_array_sha256(actual) != _array_sha256(actual)
    rows[0]["ankle_matching_grid_run_indices"] = [0, 1]
    with pytest.raises(ValueError, match="availability count changed"):
        availability_mask_from_rows(rows)


def test_materializer_interface_cannot_accept_annotations() -> None:
    signature = inspect.signature(spatial._materialize_ankle_windows)
    assert tuple(signature.parameters) == ("raw_path", "coverage_rows", "availability")
    source = inspect.getsource(spatial._materialize_ankle_windows)
    for forbidden in ("activity", "fog_severity", "taskID", "scored_labels"):
        assert forbidden not in source


def test_materializer_splits_gaps_aligns_unique_runs_and_preserves_coverage_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sensor_columns = [
        f"{sensor}_{modality}_{axis}"
        for sensor in ("back", "ankleL")
        for modality in ("acc", "gyro")
        for axis in "xyz"
    ]
    rows: list[dict[str, Any]] = []

    def add_session(subject: int, session: int, first: float, second: float) -> None:
        for run_index, origin in enumerate((first, second)):
            for sample in range(180):
                row: dict[str, Any] = {
                    "timestamp": origin + sample / 60.0,
                    "subjectID": subject,
                    "sessionID": session,
                    "activity": f"annotation-{run_index}",
                    "fog_severity": 99,
                    "taskID": "must-not-be-read",
                }
                row.update({name: 0.0 for name in sensor_columns})
                row["ankleL_acc_z"] = 1.0
                row["ankleL_gyro_x"] = float(subject + run_index)
                rows.append(row)
            gap: dict[str, Any] = {
                "timestamp": origin + 3.0,
                "subjectID": subject,
                "sessionID": session,
                "activity": "annotation-gap",
                "fog_severity": 99,
                "taskID": "must-not-be-read",
            }
            gap.update({name: 0.0 for name in sensor_columns})
            gap["ankleL_acc_x"] = np.nan
            rows.append(gap)

    # Write session 2 first. groupby(sort=True) will process session 1 first, so
    # the final result can only follow coverage order if the explicit index sort runs.
    add_session(2, 2, 100.0, 110.0)
    add_session(1, 1, 0.0, 10.0)
    raw_path = tmp_path / "raw.csv"
    pd.DataFrame(rows).to_csv(raw_path, index=False)

    coverage: list[dict[str, Any]] = [
        {
            "window_id": "coverage-first-session-two",
            "participant_id": "fogstar:002",
            "session_id": "fogstar:002:session-002",
            "ankle_matching_grid_run_indices": [0],
            "start_time": 100.0 + 3 / 50.0,
        },
        {
            "window_id": "coverage-second-session-one",
            "participant_id": "fogstar:001",
            "session_id": "fogstar:001:session-001",
            "ankle_matching_grid_run_indices": [1],
            "start_time": 10.0 + 2 / 50.0,
        },
    ]
    availability = np.ones(2, dtype=np.bool_)

    requested_columns: list[set[str]] = []
    real_read_csv = pd.read_csv

    def guarded_read_csv(*args: Any, **kwargs: Any) -> pd.DataFrame:
        requested = set(kwargs["usecols"])
        requested_columns.append(requested)
        assert {"activity", "fog_severity", "taskID"}.isdisjoint(requested)
        return real_read_csv(*args, **kwargs)

    real_resample = resample_physical_segment
    resampled_grids: dict[float, np.ndarray] = {}

    def capture_resample(**kwargs: Any) -> Any:
        segment = real_resample(**kwargs)
        resampled_grids[float(kwargs["timestamps"][0])] = segment.timestamps.copy()
        return segment

    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
    monkeypatch.setattr(
        "inclusive_shift_har.experiments.fog_spatial_information_probe.resample_physical_segment",
        capture_resample,
    )

    windows, gravity, indices, receipts, segment_receipts = spatial._materialize_ankle_windows(
        raw_path, coverage, availability
    )

    assert requested_columns
    assert windows.shape == (2, 128, 6)
    assert gravity.shape == (2, 128, 3)
    assert all("gravity_sha256" in row for row in receipts)
    assert len(segment_receipts) == 4
    assert all("gravity_sha256" in row["audit"] for row in segment_receipts)
    assert indices.tolist() == [0, 1]
    assert [row["window_id"] for row in receipts] == [
        "coverage-first-session-two",
        "coverage-second-session-one",
    ]
    assert [(row["ankle_run_index"], row["ankle_offset"]) for row in receipts] == [
        (0, 3),
        (1, 2),
    ]
    assert set(resampled_grids) == {0.0, 10.0, 100.0, 110.0}
    for origin, grid in resampled_grids.items():
        assert np.array_equal(grid, origin + np.arange(grid.size, dtype=np.float64) / 50.0)
    expected_timestamps = (
        100.0 + np.arange(3, 131, dtype=np.float64) / 50.0,
        10.0 + np.arange(2, 130, dtype=np.float64) / 50.0,
    )
    assert [row["timestamp_sha256"] for row in receipts] == [
        _array_sha256(values) for values in expected_timestamps
    ]

    ambiguous = copy.deepcopy(coverage)
    ambiguous[1]["ankle_matching_grid_run_indices"] = [0, 1]
    with pytest.raises(ValueError, match="ankle run match changed"):
        spatial._materialize_ankle_windows(raw_path, ambiguous, availability)

    off_grid = copy.deepcopy(coverage)
    off_grid[1]["start_time"] += 0.005
    with pytest.raises(ValueError, match="timestamp correspondence changed"):
        spatial._materialize_ankle_windows(raw_path, off_grid, availability)


def test_feature_blocks_and_fallback_preserve_exact_contract() -> None:
    back = np.arange(4 * 80, dtype=np.float64).reshape(4, 80)
    ankle = back + 1000.0
    names = tuple(f"f{index}" for index in range(80))
    blocks = make_feature_blocks(back, ankle, names, names)
    assert set(blocks) == set(spatial.RESTRICTED_CELLS)
    assert blocks["bv"][0].shape == blocks["lv"][0].shape == (4, 80)
    assert blocks["blv"][0].shape == blocks["bbv"][0].shape == (4, 160)
    assert np.array_equal(blocks["blv"][0][:, :80], back)
    assert np.array_equal(blocks["blv"][0][:, 80:], ankle)
    assert np.array_equal(blocks["bbv"][0][:, :80], blocks["bbv"][0][:, 80:])
    b0 = np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    mask = np.asarray([True, False, True])
    branch = np.asarray([[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]])
    final = apply_back_fallback(b0, mask, branch)
    assert np.array_equal(final[~mask], b0[~mask])
    assert np.array_equal(final[mask], branch)


def test_fold_masks_make_all_restricted_cells_share_rows_and_exclude_evaluation() -> None:
    folds = np.repeat(np.arange(5), 6)
    eligibility = np.tile([True, True, True, False, True, True], 5)
    availability = np.tile([True, False, True, True, True, False], 5)
    for outer in range(5):
        masks = fold_masks(folds, eligibility, availability, outer)
        assert not np.any(masks["restricted_training"] & masks["b0_evaluation"])
        assert np.all(masks["restricted_training"] <= masks["b0_training"])
        assert np.all(masks["restricted_evaluation"] <= masks["b0_evaluation"])
        for _cell in spatial.RESTRICTED_CELLS:
            assert np.array_equal(
                masks["restricted_training"],
                (~masks["b0_evaluation"]) & eligibility & availability,
            )


def test_b0_replay_barrier_is_byte_exact_and_precedes_attempt_six() -> None:
    expected = np.tile(np.asarray([[0.8, 0.1, 0.1]]), (1939, 1))
    record = assert_b0_replay(expected.copy(), expected)
    assert record["attempt_count_before_decision"] == 5
    changed = expected.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], 1.0)
    with pytest.raises(ValueError, match="probability replay mismatch"):
        assert_b0_replay(changed, expected)


def test_cooperative_cancellation_is_explicit() -> None:
    sentinel = ROOT / "tests" / "not-present-cancel-sentinel"
    spatial._check_cancel(sentinel, "test")
    with pytest.raises(KeyboardInterrupt, match="cooperative cancellation"):
        spatial._check_cancel(Path(__file__), "test")


def test_analysis_requires_every_matched_control() -> None:
    roster = [f"p{index:02d}" for index in range(22)]
    people = np.repeat(np.asarray(roster), 3)
    labels = np.tile(np.arange(3, dtype=np.int64), 22)
    weak_labels = labels.copy()
    weak_labels[labels == 1] = 0
    weak = np.eye(3, dtype=np.float64)[weak_labels]
    strong = np.eye(3, dtype=np.float64)[labels]
    probability = {
        "b0": weak,
        "bv": weak,
        "lv": strong,
        "blv": strong,
        "bbv": weak,
        "f3": weak,
    }
    reports = {
        cell: method_report(
            labels=labels, probabilities=value, participant_ids=people, roster=roster
        )
        for cell, value in probability.items()
    }
    result = analyse(
        reports,
        labels=labels,
        probabilities=probability,
        participants=people,
        participant_folds=np.asarray([min(index // 5, 4) for index in range(22)]),
        roster=roster,
        b0_replay_exact=True,
    )
    assert result["advancement"]["lv"]["status"] == "pass"
    assert result["advancement"]["blv"]["status"] == "fail"
    assert set(result["comparisons"]) == set(spatial.CONTRAST_PAIRS)
    assert "bbv_minus_bv" in result["comparisons"]
