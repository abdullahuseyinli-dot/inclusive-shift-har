from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
import yaml

from inclusive_shift_har.artifacts.research_provenance import (
    IMU_HAR_IL_AVAILABLE_VALID_PARTICIPANT_ROSTER,
    IMU_HAR_IL_COMPLETE_CORE_PARTICIPANT_ROSTER,
)
from inclusive_shift_har.data import external_har as data


def test_fixed_inventory_protocol_keeps_lanes_and_independent_n_separate() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = yaml.safe_load(
        (root / "configs/protocols/imu_har_il_fixed_inventory_v1.yaml").read_text(encoding="utf-8")
    )
    assert protocol["protocol_id"] == "imu-har-il-fixed-inventory-v1"
    assert protocol["dataset"]["provider_reported_participant_count"] == 50
    assert protocol["dataset"]["requested_trial_count"] == 600
    complete = protocol["lanes"]["complete_requested_core"]
    available = protocol["lanes"]["available_valid_trials"]
    assert (complete["retained_participant_count"], complete["retained_trial_count"]) == (
        19,
        228,
    )
    assert (available["retained_participant_count"], available["retained_trial_count"]) == (
        47,
        457,
    )
    assert complete["retained_participants"] == list(IMU_HAR_IL_COMPLETE_CORE_PARTICIPANT_ROSTER)
    assert available["retained_participants"] == list(IMU_HAR_IL_AVAILABLE_VALID_PARTICIPANT_ROSTER)
    assert complete["source_payload_inventory_sha256"] == (
        "cba6f7bb1dcaea8fc60a319d73e9123aeb478f9f09752c462a6bd94086b4d78c"
    )
    assert available["source_payload_inventory_sha256"] == (
        "ca81875cd3bcab424aa3821b7723c8b6f42d170ea8931a507dad47a16e1b1c25"
    )
    policy = protocol["comparison_and_claim_policy"]
    assert policy["lanes_are_matched_before_after_comparisons"] is False
    assert policy["repetitions_increase_independent_participant_n"] is False
    assert policy["provider_presegmented_diagnostic_status_preserved"] is True
    basis = protocol["evidence_basis"]
    assert basis["complete_core_data_audit"]["sha256"] == (
        "04e259bb84967bf888141308566842f3ff79edca60cee7795d76213e7dd1f471"
    )
    assert basis["available_trial_data_audit"]["sha256"] == (
        "a667b55b5cc9f6c24e3b86435fe7149a7705eefeaf4e318eb5199cceb99741d6"
    )
    assert "intentionally not committed or required in a clean clone" in basis["scope"]


def _sources() -> tuple[data._IMUSourceFile, ...]:
    return tuple(
        data._IMUSourceFile(
            participant=person,
            repetition="Repetition_1",
            activity=activity,
            file_id=index,
            filename=f"{person}/Repetition_1/{activity}/Body-WT.csv",
            file_size=0,
            locator=f"https://example.invalid/{person}/{activity}",
        )
        for index, (person, activity) in enumerate(
            (person, activity)
            for person in ("P_01", "P_02")
            for activity in ("Walk", "Sit", "Stand")
        )
    )


def test_inventory_retains_existing_trials_without_inventing_missing_repetitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _sources()
    specs = [
        (s.participant, s.repetition, s.activity, s.filename.rsplit("/", 1)[0]) for s in sources
    ]
    monkeypatch.setattr(data, "_request_json", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(data, "_imu_folder_paths", lambda *_args, **_kwargs: specs)
    lookup = dict(zip(specs, sources, strict=True))
    monkeypatch.setattr(
        data,
        "_resolve_imu_file_optional",
        lambda spec: (spec, None if spec == specs[-1] else lookup[spec]),
    )
    common: dict[str, Any] = dict(
        participant_limit=None, repetition_limit=4, activities=("Walk", "Sit", "Stand"), workers=1
    )
    complete, omitted = data._imu_har_il_inventory(**common)
    available, missing = data._imu_har_il_inventory(**common, require_complete_core=False)
    assert len(complete) == 3 and len(available) == 5
    assert all(s.participant == "P_01" for s in complete)
    assert missing[0]["missing_folders"] == omitted[0]["missing_folders"]
    assert "other available trials retained" in missing[0]["action"]


@pytest.mark.parametrize(
    "weakness", ["nonfinite_signal", "conflicting_label", "nonfinite_label", "missing_label"]
)
def test_available_policy_localizes_quarantine_and_preserves_retained_signal_grids(
    monkeypatch: pytest.MonkeyPatch, weakness: str
) -> None:
    sources = _sources()
    monkeypatch.setattr(data, "_imu_har_il_inventory", lambda **_kwargs: (sources, ()))
    mode = "clean"

    def download(source: data._IMUSourceFile) -> tuple[data._IMUSourceFile, bytes]:
        rows = 320
        values = np.sin(np.arange(rows) / 20.0) + source.file_id
        frame = pd.DataFrame(
            {name: values.copy() for name in ("Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z")}
        )
        frame["Activity_label"] = {"Walk": 5, "Sit": 1, "Stand": 2}[source.activity]
        if source.participant == "P_02" and source.activity == "Walk":
            if mode == "nonfinite_signal":
                frame["Acc_X"] = np.nan
            elif mode == "conflicting_label":
                frame["Activity_label"] = 1
            elif mode == "nonfinite_label":
                frame.loc[17, "Activity_label"] = np.nan
            elif mode == "missing_label":
                frame = frame.drop(columns=["Activity_label"])
        payload = frame.to_csv(index=False).encode()
        return replace(source, file_size=len(payload)), payload

    monkeypatch.setattr(data, "_download_imu_payload", download)
    reference = data.load_imu_har_il(
        selection_policy="available_valid_trials", download_workers=1, repetition_limit=4
    )
    mode = weakness
    available = data.load_imu_har_il(
        selection_policy="available_valid_trials", download_workers=1, repetition_limit=4
    )
    complete = data.load_imu_har_il(download_workers=1, repetition_limit=4)
    assert set(available.participant_ids) == {"imuharil:P_01", "imuharil:P_02"}
    audit = available.summary()["cohort_audit"]
    assert audit["requested_participant_count"] == audit["retained_participant_count"] == 2
    assert audit["protocol_id"] == "imu-har-il-available-trials-v1"
    assert audit["inventory_protocol_id"] == "imu-har-il-fixed-inventory-v1"
    assert audit["requested_trial_count"] == 24
    assert audit["inventory_selected_source_file_count"] == 6
    assert len(audit["source_inventory_sha256"]) == 64
    assert len(audit["source_payload_inventory_sha256"]) == 64
    if weakness in {"missing_label", "conflicting_label", "nonfinite_label"}:
        assert audit["quarantined_trial_count"] == 0
        np.testing.assert_array_equal(reference.window_ids, available.window_ids)
        np.testing.assert_array_equal(reference.labels, available.labels)
        np.testing.assert_array_equal(reference.signals, available.signals)
        np.testing.assert_array_equal(reference.gravity, available.gravity)
        assert set(complete.participant_ids) == {"imuharil:P_01", "imuharil:P_02"}
        assert any(
            issue["member"].endswith("P_02/Repetition_1/Walk/Body-WT.csv")
            for issue in available.source_issues
        )
    else:
        assert audit["quarantined_trial_count"] == 1
        assert set(complete.participant_ids) == {"imuharil:P_01"}
        assert available.summary()["participants_with_all_classes"] == 1
        assert available.summary()[
            "perfect_prediction_mean_participant_macro_f1_ceiling"
        ] == pytest.approx(5 / 6)
    reference_index = {name: index for index, name in enumerate(reference.window_ids)}
    for index, name in enumerate(available.window_ids):
        old = reference_index[name]
        np.testing.assert_array_equal(available.signals[index], reference.signals[old])
        np.testing.assert_array_equal(available.gravity[index], reference.gravity[old])
    complete_audit = complete.cohort_audit
    assert complete_audit is not None
    assert complete_audit["protocol_id"] == "imu-har-il-complete-requested-core-v1"
    assert complete_audit["selection_policy"] == "complete_requested_core"
    assert complete_audit["independent_unit"] == (
        "participant; repetitions do not increase independent N"
    )


def test_unknown_selection_policy_fails_before_provider_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("invalid policy must not access the provider")

    monkeypatch.setattr(data, "_imu_har_il_inventory", forbidden)
    with pytest.raises(ValueError, match="selection policy"):
        data.load_imu_har_il(selection_policy="choose_highest_score")


def test_loader_preserves_retained_trial_boundary_and_tail_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _sources()
    monkeypatch.setattr(data, "_imu_har_il_inventory", lambda **_kwargs: (sources, ()))

    def download(source: data._IMUSourceFile) -> tuple[data._IMUSourceFile, bytes]:
        rows = 400
        values = np.sin(np.arange(rows) / 20.0) + source.file_id
        frame = pd.DataFrame(
            {name: values.copy() for name in ("Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z")}
        )
        frame["Activity_label"] = {"Walk": 5, "Sit": 1, "Stand": 2}[source.activity]
        if source.participant == "P_01" and source.activity == "Walk":
            frame.loc[[2, 5], "Acc_X"] = np.nan
        payload = frame.to_csv(index=False).encode()
        return replace(source, file_size=len(payload)), payload

    monkeypatch.setattr(data, "_download_imu_payload", download)
    result = data.load_imu_har_il(selection_policy="available_valid_trials", download_workers=1)
    trial = "imuharil:P_01:Repetition_1:Walk"
    trial_exclusions = [item for item in result.exclusions if item.get("trial_id") == trial]
    reasons = [item["reason"] for item in trial_exclusions]
    assert reasons.count("nonfinite_sensor") == 1
    assert reasons.count("short_finite_signal_run") == 2
    assert reasons.count("incomplete_resampled_tail") == 1
    audited_rows = sum(
        int(item["source_samples"])
        for item in result.preprocessing_audit
        if item.get("trial_id") == trial
    )
    excluded_rows = sum(
        int(item["source_rows"])
        for item in trial_exclusions
        if item["reason"] in {"nonfinite_sensor", "short_finite_signal_run"}
    )
    assert audited_rows + excluded_rows == 400
