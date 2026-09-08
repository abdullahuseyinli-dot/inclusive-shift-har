from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import inclusive_shift_har.experiments.physical_information_signal as runner
from inclusive_shift_har.data.physical_information_pilot import build_plan
from inclusive_shift_har.data.physical_information_signal import (
    CanonicalRecording,
    SignalContractError,
    WindowTable,
    analyze_physical_information,
    build_window_table,
    canonical_recording_from_arrays,
    continuous_run_slices,
    parse_device_signal_contract,
    project_bouts,
    student_t_summary,
    window_table_hash,
)
from inclusive_shift_har.manifests.canonical import (
    CanonicalJSONError,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiments/physical_information_identifiability_v1.yaml"


def _contract(
    *,
    native: bool = True,
    gravity_source: str = "provider_native_gravity",
    timestamp_unit: str = "s",
    acceleration_unit: str = "m/s^2",
    angular_velocity_unit: str = "rad/s",
    semantics: str = "total_acceleration_including_gravity",
    source_rate: float = 1.0,
    target_rate: float = 1.0,
    window_samples: int = 2,
    stride: int = 1,
) -> dict[str, Any]:
    channels = {
        "sample_id": "sample_id",
        "timestamp": "timestamp",
        "recording_break_before": "break_before",
        "ax": "acc_x",
        "ay": "acc_y",
        "az": "acc_z",
        "gx": "gyro_x",
        "gy": "gyro_y",
        "gz": "gyro_z",
    }
    if native:
        channels.update(
            {
                "gravity_x": "gravity_x",
                "gravity_y": "gravity_y",
                "gravity_z": "gravity_z",
            }
        )
    return {
        "record_kind": "physical_information_signal_contract",
        "schema_version": "1.0.0",
        "status": "qualified_and_processing_frozen",
        "evidence_status": "synthetic_fixture_not_physical_evidence",
        "device_id": "synthetic-device",
        "manufacturer": "synthetic",
        "model": "fixture",
        "firmware": "fixture-1",
        "logger": "fixture-logger",
        "axis_convention": "right-handed-fixture-frame",
        "clock_domain": "synthetic-monotonic",
        "channel_names": channels,
        "units": {
            "timestamp": timestamp_unit,
            "acceleration": acceleration_unit,
            "angular_velocity": angular_velocity_unit,
            "gravity": acceleration_unit,
        },
        "acceleration_semantics": semantics,
        "native_gravity_available": native,
        "processing_contract": {
            "source_rate_hz": source_rate,
            "target_rate_hz": target_rate,
            "gravity_source": gravity_source,
            "gravity_cutoff_hz": None if gravity_source == "provider_native_gravity" else 0.1,
            "maximum_gap_seconds": 1.5 / source_rate,
            "resampling_method": "linear_interpolation_v1",
            "window_samples": window_samples,
            "window_stride_samples": stride,
        },
        "frame_conditioning_limit": 10.0,
        "equipment_receipt_sha256": "d" * 64,
        "placement_instruction_sha256": "a" * 64,
        "bench_qualification_sha256": "b" * 64,
        "canonicalization_receipt_sha256": "c" * 64,
        "processing_frozen_at_utc": "2026-09-08T12:00:00Z",
    }


def _recording(
    contract_value: dict[str, Any],
    *,
    timestamps: np.ndarray | None = None,
    acceleration: np.ndarray | None = None,
    gravity: np.ndarray | None = None,
    gyro: np.ndarray | None = None,
    breaks: np.ndarray | None = None,
) -> CanonicalRecording:
    count = 6 if timestamps is None else len(timestamps)
    timestamps = np.arange(count, dtype=np.float64) if timestamps is None else timestamps
    gravity = np.tile(np.array([0.0, 0.0, 9.80665]), (count, 1)) if gravity is None else gravity
    acceleration = gravity.copy() if acceleration is None else acceleration
    gyro = np.zeros((count, 3), dtype=np.float64) if gyro is None else gyro
    breaks = np.zeros(count, dtype=np.bool_) if breaks is None else breaks
    contract = parse_device_signal_contract(contract_value)
    arrays: dict[str, np.ndarray] = {
        "sample_ids": np.arange(count, dtype=np.int64),
        "timestamps": timestamps,
        "acceleration_xyz": acceleration,
        "gyroscope_xyz": gyro,
        "recording_break_before": breaks,
    }
    if contract.native_gravity_available:
        arrays["native_gravity_xyz"] = gravity
    return canonical_recording_from_arrays(
        metadata={
            "recording_id": "recording-01",
            "block_id": "block-01",
            "wearer_id": "pilot:001",
            "visit_number": 1,
            "attachment_number": 1,
        },
        arrays=arrays,
        contract=contract,
    )


def test_contract_rejects_ambiguous_channels_units_and_gravity_modes() -> None:
    parse_device_signal_contract(_contract())
    missing_channel = _contract()
    del missing_channel["channel_names"]["timestamp"]
    with pytest.raises(SignalContractError, match="fields missing"):
        parse_device_signal_contract(missing_channel)
    bad_unit = _contract()
    bad_unit["units"]["acceleration"] = "counts"
    with pytest.raises(SignalContractError, match="units"):
        parse_device_signal_contract(bad_unit)
    missing_native = _contract(native=False)
    with pytest.raises(SignalContractError, match="native gravity"):
        parse_device_signal_contract(missing_native)
    derived_linear = _contract(
        native=False,
        gravity_source="derived_causal_lowpass",
        semantics="linear_acceleration_gravity_removed",
        source_rate=10.0,
        target_rate=10.0,
    )
    with pytest.raises(SignalContractError, match="total acceleration"):
        parse_device_signal_contract(derived_linear)


def test_unit_conversion_and_native_linear_semantics_are_explicit() -> None:
    si_value = _contract(source_rate=1.0, target_rate=1.0)
    si = _recording(
        si_value,
        gyro=np.tile(np.array([math.pi, 0.0, 0.0]), (6, 1)),
    )
    device_value = _contract(
        timestamp_unit="ms",
        acceleration_unit="g",
        angular_velocity_unit="deg/s",
        source_rate=1.0,
        target_rate=1.0,
    )
    device = _recording(
        device_value,
        timestamps=np.arange(6, dtype=np.float64) * 1000.0,
        acceleration=np.tile(np.array([0.0, 0.0, 1.0]), (6, 1)),
        gravity=np.tile(np.array([0.0, 0.0, 1.0]), (6, 1)),
        gyro=np.tile(np.array([180.0, 0.0, 0.0]), (6, 1)),
    )
    assert np.allclose(si.timestamps, device.timestamps)
    assert np.allclose(si.acceleration_xyz, device.acceleration_xyz)
    assert np.allclose(si.gyroscope_xyz, device.gyroscope_xyz)
    linear_value = _contract(semantics="linear_acceleration_gravity_removed")
    linear = _recording(
        linear_value,
        acceleration=np.zeros((6, 3)),
    )
    table = build_window_table([linear], parse_device_signal_contract(linear_value))
    assert np.allclose(table.dynamic_acceleration_rms, 0.0)
    assert table.valid.all()


def test_continuous_runs_split_on_break_gap_nonfinite_and_nonincreasing() -> None:
    value = _contract(source_rate=2.0, target_rate=2.0)
    timestamps = np.array([0.0, 0.5, 1.0, 1.5, 3.0, 3.5, 4.0, 3.5, 4.0, 4.5])
    acceleration = np.tile(np.array([0.0, 0.0, 9.80665]), (10, 1))
    acceleration[3, 0] = np.nan
    breaks = np.zeros(10, dtype=np.bool_)
    breaks[2] = True
    recording = _recording(value, timestamps=timestamps, acceleration=acceleration, breaks=breaks)
    assert continuous_run_slices(recording, maximum_gap_seconds=0.75) == [
        (0, 2),
        (2, 3),
        (4, 7),
        (7, 10),
    ]


def test_window_stride_trim_and_annotation_invariance() -> None:
    value = _contract(window_samples=2, stride=1)
    contract = parse_device_signal_contract(value)
    recording = _recording(value)
    windows = build_window_table([recording], contract)
    assert windows.starts_seconds.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert windows.ends_seconds.tolist() == [2.0, 3.0, 4.0, 5.0, 6.0]
    original_hash = window_table_hash(windows)
    planned = {
        "bout_id": "bout-01",
        "block_id": "block-01",
        "wearer_id": "pilot:001",
        "role": "query",
        "activity": "sitting",
        "motion": "quiet",
        "interior_trim_seconds_each_end": 1,
    }
    annotation = {
        "bout_id": "bout-01",
        "recording_id": "recording-01",
        "start_sample": 0,
        "stop_sample_exclusive": 6,
        "observed_activity": "sitting",
        "observed_motion": "quiet",
        "adjudication_status": "adjudicated",
    }
    projection = project_bouts(
        plan_bouts=[planned],
        annotations=[annotation],
        recordings=[recording],
        windows=windows,
    )[0]
    assert projection["candidate_window_count"] == 3
    assert projection["valid_window_count"] == 3
    assert projection["eligible"] is True
    mutated = dict(annotation, observed_activity="standing")
    assert window_table_hash(build_window_table([recording], contract)) == original_hash
    rejected = project_bouts(
        plan_bouts=[planned],
        annotations=[mutated],
        recordings=[recording],
        windows=windows,
    )[0]
    assert rejected["eligibility_reason"] == "adjudicated_label_does_not_match_plan"


def test_derived_gravity_resets_at_signal_run_boundaries() -> None:
    value = _contract(
        native=False,
        gravity_source="derived_causal_lowpass",
        source_rate=10.0,
        target_rate=10.0,
        window_samples=2,
        stride=2,
    )
    acceleration = np.vstack(
        [
            np.tile(np.array([0.0, 0.0, 9.80665]), (5, 1)),
            np.tile(np.array([9.80665, 0.0, 0.0]), (5, 1)),
        ]
    )
    breaks = np.zeros(10, dtype=np.bool_)
    breaks[5] = True
    recording = _recording(
        value,
        timestamps=np.arange(10, dtype=np.float64) / 10.0,
        acceleration=acceleration,
        breaks=breaks,
    )
    table = build_window_table([recording], parse_device_signal_contract(value))
    assert set(table.run_indices.tolist()) == {0, 1}
    first_second_run = int(np.flatnonzero(table.run_indices == 1)[0])
    assert np.allclose(table.gravity_directions[first_second_run], [1.0, 0.0, 0.0])


def test_derived_lane_does_not_use_an_available_native_gravity_channel() -> None:
    value = _contract(
        native=True,
        gravity_source="derived_causal_lowpass",
        source_rate=10.0,
        target_rate=10.0,
        window_samples=2,
        stride=2,
    )
    native_gravity = np.full((6, 3), np.nan)
    recording = _recording(
        value,
        timestamps=np.arange(6, dtype=np.float64) / 10.0,
        gravity=native_gravity,
        acceleration=np.tile(np.array([0.0, 0.0, 9.80665]), (6, 1)),
    )
    table = build_window_table([recording], parse_device_signal_contract(value))
    assert len(table.window_ids) == 3
    assert table.valid.all()


def test_invalid_gravity_window_stays_in_bout_denominator_with_reason() -> None:
    value = _contract(window_samples=2, stride=2)
    gravity = np.array(
        [
            [0.0, 0.0, 9.80665],
            [0.0, 0.0, -9.80665],
            [0.0, 0.0, 9.80665],
            [0.0, 0.0, 9.80665],
        ]
    )
    recording = _recording(
        value,
        timestamps=np.arange(4, dtype=np.float64),
        gravity=gravity,
        acceleration=gravity,
    )
    windows = build_window_table([recording], parse_device_signal_contract(value))
    assert windows.valid.tolist() == [False, True]
    assert windows.invalid_reasons[0] == "gravity_mean_norm_at_or_below_epsilon"
    planned = {
        "bout_id": "bout-01",
        "block_id": "block-01",
        "wearer_id": "pilot:001",
        "role": "query",
        "activity": "sitting",
        "motion": "quiet",
        "interior_trim_seconds_each_end": 0,
    }
    projection = project_bouts(
        plan_bouts=[planned],
        annotations=[
            {
                "bout_id": "bout-01",
                "recording_id": "recording-01",
                "start_sample": 0,
                "stop_sample_exclusive": 4,
                "observed_activity": "sitting",
                "observed_motion": "quiet",
                "adjudication_status": "adjudicated",
            }
        ],
        recordings=[recording],
        windows=windows,
        minimum_valid_window_count=1,
        minimum_valid_window_fraction=0.8,
        source_rate_hz=1.0,
    )[0]
    assert projection["candidate_window_count"] == 2
    assert projection["valid_window_count"] == 1
    assert projection["valid_window_fraction"] == 0.5
    assert projection["eligibility_reason"] == "valid_window_fraction_below_minimum"


def test_reset_clock_windows_are_bound_to_annotation_source_sample_support() -> None:
    value = _contract(window_samples=2, stride=1)
    timestamps = np.array([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0])
    gravity = np.vstack(
        [
            np.tile(np.array([0.0, 0.0, 9.80665]), (4, 1)),
            np.tile(np.array([9.80665, 0.0, 0.0]), (4, 1)),
        ]
    )
    breaks = np.zeros(8, dtype=np.bool_)
    breaks[4] = True
    recording = _recording(
        value,
        timestamps=timestamps,
        acceleration=gravity,
        gravity=gravity,
        breaks=breaks,
    )
    windows = build_window_table([recording], parse_device_signal_contract(value))
    assert len(windows.window_ids) == 6
    projection = project_bouts(
        plan_bouts=[
            {
                "bout_id": "first-epoch",
                "block_id": "block-01",
                "wearer_id": "pilot:001",
                "role": "query",
                "activity": "sitting",
                "motion": "quiet",
                "interior_trim_seconds_each_end": 0,
            }
        ],
        annotations=[
            {
                "bout_id": "first-epoch",
                "recording_id": "recording-01",
                "start_sample": 0,
                "stop_sample_exclusive": 4,
                "observed_activity": "sitting",
                "observed_motion": "quiet",
                "adjudication_status": "adjudicated",
            }
        ],
        recordings=[recording],
        windows=windows,
        minimum_valid_window_count=1,
        source_rate_hz=1.0,
    )[0]
    assert projection["candidate_window_count"] == 3
    selected = [windows.window_ids.index(item) for item in projection["candidate_window_ids"]]
    assert all(windows.source_stop_sample_indices_exclusive[index] <= 4 for index in selected)
    assert all(windows.run_indices[index] == 0 for index in selected)


def test_nonfinite_dropout_remains_in_nominal_grid_denominator() -> None:
    value = _contract(window_samples=2, stride=1)
    gravity = np.tile(np.array([0.0, 0.0, 9.80665]), (10, 1))
    acceleration = gravity.copy()
    acceleration[2:7] = np.nan
    recording = _recording(
        value,
        timestamps=np.arange(10, dtype=np.float64),
        acceleration=acceleration,
        gravity=gravity,
    )
    windows = build_window_table([recording], parse_device_signal_contract(value))
    assert len(windows.window_ids) == 9
    projection = project_bouts(
        plan_bouts=[
            {
                "bout_id": "dropout-bout",
                "block_id": "block-01",
                "wearer_id": "pilot:001",
                "role": "query",
                "activity": "sitting",
                "motion": "quiet",
                "interior_trim_seconds_each_end": 0,
            }
        ],
        annotations=[
            {
                "bout_id": "dropout-bout",
                "recording_id": "recording-01",
                "start_sample": 0,
                "stop_sample_exclusive": 10,
                "observed_activity": "sitting",
                "observed_motion": "quiet",
                "adjudication_status": "adjudicated",
            }
        ],
        recordings=[recording],
        windows=windows,
        minimum_valid_window_count=1,
        minimum_valid_window_fraction=0.8,
        source_rate_hz=1.0,
    )[0]
    assert projection["candidate_window_count"] == 9
    assert projection["valid_window_count"] == 3
    assert projection["valid_window_fraction"] == pytest.approx(1.0 / 3.0)
    assert projection["candidate_invalid_reason_counts"] == {
        "nonfinite_inertial_or_gravity_sample": 6
    }
    assert projection["eligibility_reason"] == "valid_window_fraction_below_minimum"


def test_timestamp_gap_remains_an_explicit_invalid_candidate() -> None:
    value = _contract(window_samples=2, stride=1)
    timestamps = np.array([0.0, 1.0, 5.0, 6.0])
    gravity = np.tile(np.array([0.0, 0.0, 9.80665]), (4, 1))
    recording = _recording(value, timestamps=timestamps, acceleration=gravity, gravity=gravity)
    windows = build_window_table([recording], parse_device_signal_contract(value))
    assert len(windows.window_ids) == 6
    assert windows.invalid_reasons.count("timestamp_gap_above_frozen_maximum") == 4


def _analysis_fixture(
    *, standing: np.ndarray | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]], WindowTable]:
    sitting_vector = np.array([0.0, 0.0, 1.0])
    standing_vector = np.array([1.0, 0.0, 0.0]) if standing is None else standing
    wearers = [f"pilot:{index:03d}" for index in range(1, 7)]
    bouts: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    projections: list[dict[str, Any]] = []
    window_ids: list[str] = []
    recording_ids: list[str] = []
    block_ids: list[str] = []
    wearer_ids: list[str] = []
    directions: list[np.ndarray] = []
    for wearer in wearers:
        previous_support: list[str] | None = None
        for block_index in range(4):
            block = f"{wearer}/block-{block_index + 1}"
            support_ids: list[str] = []
            query_ids: list[str] = []
            for role in ("support", "query"):
                for activity, motion in (
                    ("sitting", "quiet"),
                    ("standing", "quiet"),
                    ("sitting", "upper_body_motion"),
                    ("standing", "upper_body_motion"),
                ):
                    bout_id = f"{block}/{role}-{activity}-{motion}"
                    bouts.append(
                        {
                            "bout_id": bout_id,
                            "block_id": block,
                            "wearer_id": wearer,
                            "role": role,
                            "activity": activity,
                            "motion": motion,
                        }
                    )
                    (support_ids if role == "support" else query_ids).append(bout_id)
                    window_id = f"window/{len(window_ids):04d}"
                    window_ids.append(window_id)
                    recording_ids.append(f"recording/{block}")
                    block_ids.append(block)
                    wearer_ids.append(wearer)
                    directions.append(sitting_vector if activity == "sitting" else standing_vector)
                    projections.append(
                        {
                            "bout_id": bout_id,
                            "wearer_id": wearer,
                            "activity": activity,
                            "motion": motion,
                            "eligible": True,
                            "valid_window_ids": [window_id],
                        }
                    )
            comparisons.append(
                {
                    "comparison_id": f"fresh::{block}",
                    "comparison_type": "fresh",
                    "query_bout_ids": query_ids,
                    "fresh_support_bout_ids": support_ids,
                    "stale_support_bout_ids": None,
                }
            )
            if previous_support is not None and block_index in (1, 3):
                comparisons.append(
                    {
                        "comparison_id": f"attachment-stale::{block}",
                        "comparison_type": "attachment_stale",
                        "query_bout_ids": query_ids,
                        "fresh_support_bout_ids": support_ids,
                        "stale_support_bout_ids": previous_support,
                    }
                )
            previous_support = support_ids
    count = len(window_ids)
    windows = WindowTable(
        window_ids=tuple(window_ids),
        recording_ids=tuple(recording_ids),
        block_ids=tuple(block_ids),
        wearer_ids=tuple(wearer_ids),
        run_indices=np.zeros(count, dtype=np.int64),
        source_run_indices=np.zeros(count, dtype=np.int64),
        source_start_sample_indices=np.arange(count, dtype=np.int64),
        source_stop_sample_indices_exclusive=np.arange(count, dtype=np.int64) + 1,
        starts_seconds=np.arange(count, dtype=np.float64),
        ends_seconds=np.arange(count, dtype=np.float64) + 1.0,
        gravity_directions=np.vstack(directions),
        dynamic_acceleration_rms=np.linspace(0.0, 1.0, count),
        gyroscope_rms=np.linspace(0.0, 2.0, count),
        valid=np.ones(count, dtype=np.bool_),
        invalid_reasons=("valid",) * count,
    )
    return {"wearers": wearers, "bouts": bouts, "comparisons": comparisons}, projections, windows


def test_prototypes_stale_intersections_all_six_gate_and_synthetic_claim_limit() -> None:
    plan, projections, windows = _analysis_fixture()
    analysis = analyze_physical_information(
        plan=plan,
        projections=projections,
        windows=windows,
        contract=parse_device_signal_contract(_contract()),
    )
    assert analysis["physical_identifiability_gate"]["status"] == "pass"
    assert analysis["claims"]["physical_identifiability_demonstrated"] is False
    assert len(analysis["stale_comparison_rows"]) == 48
    assert all(
        row["identical_query_window_count"] == 1
        and len(row["identical_query_window_ids_sha256"]) == 64
        for row in analysis["stale_comparison_rows"]
    )
    damaged = copy.deepcopy(projections)
    damaged[4]["eligible"] = False
    analysis_damaged = analyze_physical_information(
        plan=plan,
        projections=damaged,
        windows=windows,
        contract=parse_device_signal_contract(_contract()),
    )
    assert analysis_damaged["physical_identifiability_gate"]["status"] == "fail"
    assert (
        "not_all_six_frozen_roster_wearers_complete"
        in analysis_damaged["physical_identifiability_gate"]["failures"]
    )


def test_fresh_and_stale_summaries_exclude_incomplete_estimands_explicitly() -> None:
    plan, projections, windows = _analysis_fixture()
    damaged = copy.deepcopy(projections)
    damaged[0]["eligible"] = False
    analysis = analyze_physical_information(
        plan=plan,
        projections=damaged,
        windows=windows,
        contract=parse_device_signal_contract(_contract()),
    )
    assert analysis["wearer_completeness"][0]["complete"] is True
    quiet = analysis["fresh_aggregation"]["equal_wearer_summaries"]["quiet"]
    assert quiet["count"] == 5
    assert "pilot:001" not in quiet["included_wearer_ids"]
    assert quiet["excluded_wearers"][0]["wearer_id"] == "pilot:001"
    quiet_wearer = next(
        row
        for row in analysis["fresh_aggregation"]["condition_to_wearer_rows"]
        if row["wearer_id"] == "pilot:001" and row["motion"] == "quiet"
    )
    assert quiet_wearer["wearer_mean"] is None
    assert quiet_wearer["estimand_complete"] is False
    stale = analysis["stale_descriptive_summaries"]["attachment_stale"]
    assert stale["equal_wearer_summary"]["count"] == 5
    excluded = stale["equal_wearer_summary"]["excluded_wearers"]
    assert excluded[0]["wearer_id"] == "pilot:001"
    assert stale["wearer_rows"][0]["wearer_mean"] is None


def test_stale_summary_excludes_wearer_with_unrelated_missing_fresh_query() -> None:
    plan, projections, windows = _analysis_fixture()
    damaged = copy.deepcopy(projections)
    # The first block is never an attachment-stale query block. Its missing query
    # therefore leaves all eight attachment-stale contrasts available while making
    # the protocol's 16-query wearer completeness rule fail.
    damaged[4]["eligible"] = False
    analysis = analyze_physical_information(
        plan=plan,
        projections=damaged,
        windows=windows,
        contract=parse_device_signal_contract(_contract()),
    )
    assert analysis["wearer_completeness"][0]["complete"] is False
    stale = analysis["stale_descriptive_summaries"]["attachment_stale"]
    wearer = stale["wearer_rows"][0]
    assert wearer["observed_condition_reference_count"] == 8
    assert wearer["eligible_condition_reference_count"] == 8
    assert wearer["estimand_complete"] is False
    assert wearer["wearer_mean"] is None
    assert {item["reason"] for item in wearer["exclusions"]} == {
        "wearer_signal_completeness_failed"
    }
    assert stale["equal_wearer_summary"]["count"] == 5
    assert "pilot:001" not in stale["equal_wearer_summary"]["included_wearer_ids"]


@pytest.mark.parametrize(
    "standing",
    [np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])],
)
def test_degenerate_and_antiparallel_frames_are_not_declared_eligible(
    standing: np.ndarray,
) -> None:
    plan, projections, windows = _analysis_fixture(standing=standing)
    analysis = analyze_physical_information(
        plan=plan,
        projections=projections,
        windows=windows,
        contract=parse_device_signal_contract(_contract()),
    )
    first = analysis["reference_records"][0]["fresh_prototypes"]["motion_strata"]["quiet"]
    assert first["numerically_nondegenerate"] is False
    assert first["inverse_absolute_sine_condition_number"] is None
    assert first["frame_eligible"] is False


def test_student_t_interval_is_suppressed_below_two_independent_wearers() -> None:
    one = student_t_summary([0.5])
    assert one["student_t_95_interval"] is None
    assert one["interval_suppression_reason"] == "suppressed_below_two_complete_wearers"
    two = student_t_summary([0.0, 1.0])
    assert two["interval_degrees_of_freedom"] == 1
    assert two["student_t_95_interval"] is not None


def _frozen_plan() -> tuple[dict[str, Any], dict[str, Any]]:
    config_value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(config_value, dict)
    plan = build_plan(
        config_value,
        created_at_utc="2026-09-08T12:00:00.000000Z",
        code_commit="d" * 40,
        config_reference={"path": "config", "sha256": "a" * 64},
        protocol_reference={"path": "protocol", "sha256": "b" * 64},
        schema_reference={"path": "schema", "sha256": "c" * 64},
    )
    return plan, config_value


def test_blocker_mode_is_create_only_self_hashed_and_has_no_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, config_value = _frozen_plan()
    plan_path = tmp_path / "plan.json"
    config_path = tmp_path / "config.yaml"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    config_path.write_text(yaml.safe_dump(config_value), encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "_validate_source_lock",
        lambda _root, _commit: {"commit": "d" * 40, "clean": True, "status_lines": []},
    )
    output = tmp_path / "run-001"
    result = runner.emit_blocker_record(
        repository_root=ROOT,
        plan_path=plan_path,
        config_path=config_path,
        output_directory=output,
        code_commit="d" * 40,
    )
    assert result["status"] == "incomplete_blocker_sealed"
    blocker_value = load_json_strict(output / "INCOMPLETE.json")
    assert isinstance(blocker_value, dict)
    declared = blocker_value.pop("record_sha256")
    assert declared == canonical_json_sha256(blocker_value)
    assert blocker_value["participant_outcomes_loaded"] is False
    assert blocker_value["physical_identifiability_gate"]["status"] == "not_evaluated"
    assert set(runner.SIGNAL_SOFTWARE_PREREQUISITES) <= set(blocker_value["blocking_prerequisites"])
    with pytest.raises(FileExistsError):
        runner.emit_blocker_record(
            repository_root=ROOT,
            plan_path=plan_path,
            config_path=config_path,
            output_directory=output,
            code_commit="d" * 40,
        )


def test_zero_fit_execute_package_replays_from_copied_canonical_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, config_value = _frozen_plan()
    plan_path = tmp_path / "plan.json"
    config_path = tmp_path / "config.yaml"
    contract_path = tmp_path / "contract.json"
    annotations_path = tmp_path / "annotations.json"
    collection_path = tmp_path / "collection.json"
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    config_path.write_text(yaml.safe_dump(config_value), encoding="utf-8")
    contract_value = _contract()
    contract_path.write_text(json.dumps(contract_value), encoding="utf-8")
    recording_by_block: dict[str, str] = {}
    recording_rows: list[dict[str, Any]] = []
    for index, block in enumerate(plan["blocks"]):
        recording_id = f"recording-{index:02d}"
        recording_by_block[block["block_id"]] = recording_id
        relative = f"recording-{index:02d}.npz"
        path = canonical_root / relative
        sample_count = 22
        clock_offset = 30.0 * (int(block["attachment_number"]) - 1)
        with path.open("xb") as stream:
            np.savez_compressed(
                stream,
                sample_ids=np.arange(sample_count, dtype=np.int64),
                timestamps=np.arange(sample_count, dtype=np.float64) + clock_offset,
                acceleration_xyz=np.tile(np.array([0.0, 0.0, 9.80665]), (sample_count, 1)),
                gyroscope_xyz=np.zeros((sample_count, 3), dtype=np.float64),
                recording_break_before=np.zeros(sample_count, dtype=np.bool_),
                native_gravity_xyz=np.tile(np.array([0.0, 0.0, 9.80665]), (sample_count, 1)),
            )
        recording_rows.append(
            {
                "recording_id": recording_id,
                "block_id": block["block_id"],
                "wearer_id": block["wearer_id"],
                "visit_number": block["visit_number"],
                "attachment_number": block["attachment_number"],
                "device_id": "synthetic-device",
                "canonical_path": relative,
                "canonical_sha256": sha256_file(path),
                "canonical_size_bytes": path.stat().st_size,
                "visit_started_at_utc": (
                    "2026-09-09T09:00:00Z"
                    if int(block["visit_number"]) == 1
                    else "2026-09-16T09:00:00Z"
                ),
                "started_at": clock_offset,
                "ended_at": clock_offset + sample_count,
                "source_clock_reset_observed": False,
                "clock_mapping_receipt": None,
            }
        )
    collection: dict[str, Any] = {
        "record_kind": "physical_information_canonical_collection_manifest",
        "schema_version": "1.0.0",
        "recordings": recording_rows,
    }
    collection["record_sha256"] = canonical_json_sha256(collection)
    collection_path.write_text(json.dumps(collection), encoding="utf-8")
    annotations = [
        {
            "bout_id": bout["bout_id"],
            "recording_id": recording_by_block[bout["block_id"]],
            "start_sample": 2 * (int(bout["sequence_in_block"]) - 1),
            "stop_sample_exclusive": 2 * int(bout["sequence_in_block"]),
            "observed_activity": bout["activity"],
            "observed_motion": bout["motion"],
            "adjudication_status": "adjudicated",
            "adjudicator_id": "synthetic-fixture",
            "adjudicated_at_utc": "2026-09-08T12:00:00Z",
        }
        for bout in plan["bouts"]
    ]
    annotations_path.write_text(json.dumps(annotations), encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "_validate_source_lock",
        lambda _root, _commit: {"commit": "d" * 40, "clean": True, "status_lines": []},
    )
    output = tmp_path / "signal-run"
    final = runner.execute_signal_run(
        repository_root=ROOT,
        plan_path=plan_path,
        config_path=config_path,
        contract_path=contract_path,
        collection_manifest_path=collection_path,
        annotations_path=annotations_path,
        canonical_root=canonical_root,
        output_directory=output,
        code_commit="d" * 40,
    )
    assert final["status"] == "complete"
    assert final["model_fit_count"] == 0
    assert final["physical_identifiability_gate"]["status"] == "fail"
    validation = runner.validate_signal_run(output)
    assert validation["valid"] is True
    assert validation["model_fit_count"] == 0
    assert (output / "signal_audit.json").is_file()
    assert (output / "result.json").is_file()
    assert len(list((output / "canonical_inputs").glob("*.npz"))) == 24


def test_missing_device_contract_produces_sealed_incomplete_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, config_value = _frozen_plan()
    plan_path = tmp_path / "plan.json"
    config_path = tmp_path / "config.yaml"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    config_path.write_text(yaml.safe_dump(config_value), encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "_validate_source_lock",
        lambda _root, _commit: {"commit": "d" * 40, "clean": True, "status_lines": []},
    )
    output = tmp_path / "failed-run"
    with pytest.raises(CanonicalJSONError, match="contract"):
        runner.execute_signal_run(
            repository_root=ROOT,
            plan_path=plan_path,
            config_path=config_path,
            contract_path=tmp_path / "missing-contract.json",
            collection_manifest_path=tmp_path / "missing-collection.json",
            annotations_path=tmp_path / "missing-annotations.json",
            canonical_root=tmp_path,
            output_directory=output,
            code_commit="d" * 40,
        )
    incomplete = load_json_strict(output / "INCOMPLETE.json")
    assert isinstance(incomplete, dict)
    declared = incomplete.pop("record_sha256")
    assert declared == canonical_json_sha256(incomplete)
    assert incomplete["physical_identifiability_gate"]["status"] == "not_evaluated"
    assert incomplete["model_fit_count"] == 0
    assert (output / "failure_manifest.json").exists()
    assert (output / "failure_shutdown.json").exists()


def test_source_lock_rejects_unrelated_checkout(tmp_path: Path) -> None:
    with pytest.raises(SignalContractError, match="executing controller"):
        runner._validate_source_lock(tmp_path, "d" * 40)


def test_changed_during_copy_is_rejected_and_failure_sealed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = runner._copy_create_only

    def changed(source: Path, target: Path, *, root: Path) -> None:
        original(source, target, root=root)
        with target.open("ab") as stream:
            stream.write(b"changed during copy")

    monkeypatch.setattr(runner, "_copy_create_only", changed)
    with pytest.raises(SignalContractError, match="copied canonical bytes"):
        test_zero_fit_execute_package_replays_from_copied_canonical_inputs(tmp_path, monkeypatch)
    assert (tmp_path / "signal-run/failure_final_receipt.json").exists()


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_interruption_retains_shutdown_and_failure_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: type[BaseException],
) -> None:
    def interrupted(*args: Any, **kwargs: Any) -> Any:
        raise interruption("injected interruption")

    monkeypatch.setattr(runner, "build_window_table", interrupted)
    with pytest.raises(interruption):
        test_zero_fit_execute_package_replays_from_copied_canonical_inputs(tmp_path, monkeypatch)
    output = tmp_path / "signal-run"
    for name in (
        "INCOMPLETE.json",
        "failure_runtime.json",
        "failure_shutdown.json",
        "failure_manifest.json",
        "failure_final_receipt.json",
    ):
        record = load_json_strict(output / name)
        runner._verify_sealed(record, name)


@pytest.mark.parametrize(
    "filename", ["completion_manifest.json", "controller_final_receipt.json", "validation.json"]
)
def test_completed_lifecycle_corruption_fails_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    test_zero_fit_execute_package_replays_from_copied_canonical_inputs(tmp_path, monkeypatch)
    output = tmp_path / "signal-run"
    (output / filename).write_text("{}", encoding="utf-8")
    with pytest.raises(SignalContractError):
        runner.validate_signal_run(output)


@pytest.mark.parametrize(
    "defect", ["extra_file", "duplicate_entry", "missing_completion", "wrong_result", "wrong_audit"]
)
def test_manifest_completeness_and_exposed_summary_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    test_zero_fit_execute_package_replays_from_copied_canonical_inputs(tmp_path, monkeypatch)
    output = tmp_path / "signal-run"
    if defect == "extra_file":
        (output / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    elif defect == "missing_completion":
        (output / "completion_manifest.json").rename(output / "retained_original_completion.json")
    elif defect == "duplicate_entry":
        path = output / "artifact_manifest.json"
        manifest = load_json_strict(path)
        manifest.pop("record_sha256")
        manifest["artifacts"].append(manifest["artifacts"][0])
        path.write_text(json.dumps(runner._sealed(manifest)), encoding="utf-8")
    else:
        monkeypatch.setattr(runner, "_verify_artifact_manifest", lambda *_: None)
        monkeypatch.setattr(runner, "_verify_completion", lambda *_, **__: None)
        path = output / ("result.json" if defect == "wrong_result" else "signal_audit.json")
        value = load_json_strict(path)
        value.pop("record_sha256")
        value["recording_count"] = -1
        path.write_text(json.dumps(runner._sealed(value)), encoding="utf-8")
    with pytest.raises(SignalContractError):
        runner.validate_signal_run(output)


def test_common_visit_chronology_rejects_reset_and_requires_mapping() -> None:
    contract = parse_device_signal_contract(_contract())
    rows = [
        {
            "wearer_id": "pilot:001",
            "visit_number": 1,
            "attachment_number": index + 1,
            "visit_started_at_utc": "2026-09-09T12:00:00Z",
            "started_at": index * 30.0,
            "ended_at": index * 30.0 + 20,
            "source_clock_reset_observed": False,
            "clock_mapping_receipt": None,
        }
        for index in range(2)
    ]
    runner._collection_chronology(rows, contract)
    reset = copy.deepcopy(rows)
    reset[1]["started_at"] = 0.0
    with pytest.raises(SignalContractError, match="reset/overlap"):
        runner._collection_chronology(reset, contract)
    reset = copy.deepcopy(rows)
    reset[1]["source_clock_reset_observed"] = True
    with pytest.raises(SignalContractError, match="reviewed mapping"):
        runner._collection_chronology(reset, contract)
    reset = copy.deepcopy(rows)
    reset[1]["visit_started_at_utc"] = "2026-09-08T11:00:00Z"
    with pytest.raises(SignalContractError, match="after qualified"):
        runner._collection_chronology(reset, contract)


def test_real_mode_requires_actual_qualification_files(tmp_path: Path) -> None:
    value = _contract()
    value["evidence_status"] = "development_physical_measurement"
    plan, config = _frozen_plan()
    with pytest.raises(SignalContractError, match="actual qualification"):
        runner._qualification(
            contract=parse_device_signal_contract(value),
            plan=plan,
            config=config,
            manifest_path=None,
            root=tmp_path,
            copy_root=None,
        )


def _qualification_fixture(root: Path) -> tuple[dict[str, Any], Path]:
    value = _contract()
    value["evidence_status"] = "development_physical_measurement"
    for stem in ("placement", "bench"):
        (root / f"{stem}.txt").write_text(f"synthetic {stem} evidence fixture", encoding="utf-8")
    value["placement_instruction_sha256"] = sha256_file(root / "placement.txt")
    value["bench_qualification_sha256"] = sha256_file(root / "bench.txt")
    equipment = {
        key: value[key]
        for key in (
            "device_id",
            "manufacturer",
            "model",
            "firmware",
            "logger",
            "axis_convention",
            "clock_domain",
            "native_gravity_available",
            "processing_contract",
            "frame_conditioning_limit",
            "processing_frozen_at_utc",
        )
    }
    equipment.update(
        {
            "record_kind": "physical_information_equipment_receipt",
            "receipt_id": "synthetic-equipment",
            "status": "qualified",
            "qualified_at_utc": "2026-09-08T08:00:00Z",
            "placement_instruction_path": "placement.txt",
            "placement_instruction_sha256": value["placement_instruction_sha256"],
            "bench_qualification_path": "bench.txt",
            "bench_qualification_sha256": value["bench_qualification_sha256"],
            "channel_map": value["channel_names"],
            "units": value["units"],
            "nominal_rate_hz": 1.0,
            "observed_interval_summary": {
                "sample_count": 20,
                "median_seconds": 1.0,
                "p05_seconds": 1.0,
                "p95_seconds": 1.0,
                "nonpositive_interval_count": 0,
                "gap_count": 0,
            },
        }
    )
    (root / "equipment.json").write_text(json.dumps(equipment), encoding="utf-8")
    value["equipment_receipt_sha256"] = sha256_file(root / "equipment.json")
    adapter = runner._sealed(
        {
            "record_kind": "physical_information_canonicalization_receipt",
            "status": "reviewed",
            "device_id": value["device_id"],
            "equipment_receipt_sha256": value["equipment_receipt_sha256"],
            "processing_contract_sha256": canonical_json_sha256(value["processing_contract"]),
            "reviewer_id": "synthetic-reviewer",
            "adapter_code_sha256": "a" * 64,
            "reviewed_at_utc": "2026-09-08T09:00:00Z",
        }
    )
    (root / "adapter.json").write_text(json.dumps(adapter), encoding="utf-8")
    value["canonicalization_receipt_sha256"] = sha256_file(root / "adapter.json")
    manifest = runner._sealed(
        {
            "record_kind": "physical_information_signal_qualification_manifest",
            "schema_version": "1.0.0",
            "equipment": {"path": "equipment.json", "sha256": value["equipment_receipt_sha256"]},
            "canonicalization": {
                "path": "adapter.json",
                "sha256": value["canonicalization_receipt_sha256"],
            },
        }
    )
    path = root / "qualification.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return value, path


def test_existing_equipment_schema_hashes_and_freeze_are_bound(tmp_path: Path) -> None:
    value, path = _qualification_fixture(tmp_path)
    plan, config = _frozen_plan()
    copied = tmp_path / "copied"
    copied.mkdir()
    result = runner._qualification(
        contract=parse_device_signal_contract(value),
        plan=plan,
        config=config,
        manifest_path=path,
        root=tmp_path,
        copy_root=copied,
    )
    assert result["equipment_evidence_verified"] is True
    assert result["physical_evidence_qualified"] is False
    assert result["study_prerequisites_verified"] is False
    assert len(list(copied.iterdir())) == 4
    (tmp_path / "bench.txt").write_text("corrupt fixture", encoding="utf-8")
    with pytest.raises(SignalContractError, match="equipment qualification invalid"):
        runner._qualification(
            contract=parse_device_signal_contract(value),
            plan=plan,
            config=config,
            manifest_path=path,
            root=tmp_path,
            copy_root=None,
        )


def test_adapter_review_after_freeze_cannot_qualify(tmp_path: Path) -> None:
    value, path = _qualification_fixture(tmp_path)
    adapter = load_json_strict(tmp_path / "adapter.json")
    adapter.pop("record_sha256")
    adapter["reviewed_at_utc"] = "2026-09-08T13:00:00Z"
    (tmp_path / "adapter.json").write_text(json.dumps(runner._sealed(adapter)), encoding="utf-8")
    value["canonicalization_receipt_sha256"] = sha256_file(tmp_path / "adapter.json")
    manifest = load_json_strict(path)
    manifest.pop("record_sha256")
    manifest["canonicalization"]["sha256"] = value["canonicalization_receipt_sha256"]
    path.write_text(json.dumps(runner._sealed(manifest)), encoding="utf-8")
    plan, config = _frozen_plan()
    with pytest.raises(SignalContractError, match="precede processing freeze"):
        runner._qualification(
            contract=parse_device_signal_contract(value),
            plan=plan,
            config=config,
            manifest_path=path,
            root=tmp_path,
            copy_root=None,
        )


def test_blocker_interruption_retains_failure_and_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_write = runner._write_json

    def interrupt_validation(path: Path, payload: Any, *, root: Path) -> None:
        if path.name == "validation.json":
            raise KeyboardInterrupt("synthetic blocker interruption")
        original_write(path, payload, root=root)

    monkeypatch.setattr(runner, "_write_json", interrupt_validation)
    with pytest.raises(KeyboardInterrupt):
        test_blocker_mode_is_create_only_self_hashed_and_has_no_outcomes(tmp_path, monkeypatch)
    output = tmp_path / "run-001"
    assert load_json_strict(output / "INCOMPLETE.json")["status"] == (
        "incomplete_external_prerequisites_missing"
    )
    assert load_json_strict(output / "failure_reason.json")["reason_type"] == "KeyboardInterrupt"
    assert load_json_strict(output / "failure_shutdown.json")["workers_remaining"] == 0
    assert load_json_strict(output / "failure_final_receipt.json")["model_fit_count"] == 0
