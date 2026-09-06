"""Metamorphic checks for provider-boundary and annotation-isolation contracts."""

from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from inclusive_shift_har.data import external_har


def _imu_sources(participant_count: int = 5) -> tuple[external_har._IMUSourceFile, ...]:
    return tuple(
        external_har._IMUSourceFile(
            participant=participant,
            repetition="Repetition_1",
            activity=activity,
            file_id=index,
            filename=f"HAR_IMU_IL/{participant}/Repetition_1/{activity}/Body-WT.csv",
            file_size=0,
            locator=f"https://example.invalid/{participant}/{activity}",
        )
        for index, (participant, activity) in enumerate(
            (f"P_{person:02d}", activity)
            for person in range(1, participant_count + 1)
            for activity in ("Walk", "Sit", "Stand")
        )
    )


def test_fog_participant_with_no_scored_labels_stays_in_raw_roster_and_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows: list[pd.DataFrame] = []
    for person in range(1, 6):
        timestamps = np.arange(900, dtype=np.float64) / 60.0
        labels = np.repeat(np.array([1.0, 2.0, 3.0]), 300)
        if person == 5:
            labels[:] = np.nan
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "back_acc_x": np.sin(timestamps + person),
                    "back_acc_y": np.cos(timestamps),
                    "back_acc_z": np.ones(timestamps.size),
                    "back_gyro_x": np.sin(timestamps / 2.0),
                    "back_gyro_y": np.cos(timestamps / 3.0),
                    "back_gyro_z": np.zeros(timestamps.size),
                    "activity": labels,
                    "subjectID": person,
                    "sessionID": 1,
                }
            )
        )
    payload = pd.concat(rows, ignore_index=True).to_csv(index=False).encode("utf-8")
    monkeypatch.setattr(external_har, "_request_bytes", lambda *_args, **_kwargs: payload)
    monkeypatch.setattr(
        external_har,
        "_receipt",
        lambda **kwargs: external_har.SourceReceipt(
            kwargs["dataset_id"],
            kwargs["locator"],
            None,
            len(payload),
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        ),
    )
    result = external_har.load_fog_star(target_rate_hz=60.0)
    assert result.participant_partition_plan is not None
    assert result.participant_partition_plan.participant_roster == tuple(
        f"fogstar:{person:03d}" for person in range(1, 6)
    )
    assert "fogstar:005" not in result.participant_ids
    assert result.observable_candidates is not None
    assert "fogstar:005" in result.observable_candidates.participant_ids
    modelling, _, eligible = external_har.observable_modelling_pool(
        result, include_supervised_labels=True
    )
    fifth = modelling.participant_ids == "fogstar:005"
    assert fifth.any() and not eligible[fifth].any()
    boundary = result.summary()["boundary_provenance"]
    assert boundary["repository_signal_grid_annotation_independent"] is True
    assert boundary["provider_upstream_annotation_conditioned"] is False
    assert boundary["zero_lookahead_streaming_valid"] is False


@pytest.mark.parametrize("mutation", ["conflict", "nonfinite", "missing", "permuted"])
def test_imu_local_numeric_annotations_cannot_change_grid_or_partition_plan(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    sources = _imu_sources()
    monkeypatch.setattr(external_har, "_imu_har_il_inventory", lambda **_kwargs: (sources, ()))
    mode = "clean"

    def download(
        source: external_har._IMUSourceFile,
    ) -> tuple[external_har._IMUSourceFile, bytes]:
        rows = 384
        time = np.arange(rows, dtype=np.float64)
        base = np.sin(time / 17.0 + source.file_id)
        frame = pd.DataFrame(
            {
                name: base + channel / 10.0
                for channel, name in enumerate(
                    ("Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z")
                )
            }
        )
        expected = {"Walk": 5.0, "Sit": 1.0, "Stand": 2.0}[source.activity]
        frame["Activity_label"] = expected
        if source.file_id == 0:
            if mode == "conflict":
                frame["Activity_label"] = 1.0
            elif mode == "nonfinite":
                frame.loc[31, "Activity_label"] = np.nan
            elif mode == "missing":
                frame = frame.drop(columns=["Activity_label"])
            elif mode == "permuted":
                frame["Activity_label"] = np.resize(np.array([1.0, 2.0, 5.0]), rows)
        payload = frame.to_csv(index=False).encode("utf-8")
        return replace(source, file_size=len(payload)), payload

    monkeypatch.setattr(external_har, "_download_imu_payload", download)
    baseline = external_har.load_imu_har_il(download_workers=1)
    mode = mutation
    changed = external_har.load_imu_har_il(download_workers=1)

    for name in (
        "signals",
        "gravity",
        "labels",
        "participant_ids",
        "session_ids",
        "trial_ids",
        "window_ids",
    ):
        np.testing.assert_array_equal(getattr(baseline, name), getattr(changed, name))
    assert baseline.participant_partition_plan is not None
    assert changed.participant_partition_plan is not None
    assert baseline.participant_partition_plan.audit() == changed.participant_partition_plan.audit()
    assert len(changed.participant_partition_plan.participant_roster) == 5
    assert changed.source_issues
    boundary = changed.summary()["boundary_provenance"]
    assert boundary["provider_upstream_annotation_conditioned"] is True
    assert boundary["repository_signal_grid_annotation_independent"] is False
    assert boundary["local_numeric_annotation_affects_signal_grid"] is False
    assert boundary["zero_lookahead_streaming_valid"] is False


def test_imu_missing_inventory_member_cannot_remove_person_from_prewindow_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _imu_sources(4)
    missing = (
        {
            "participant_id": "imuharil:P_05",
            "reason": "missing Body-WT.csv in at least one requested core trial",
            "action": "participant excluded before windowing to retain complete core support",
            "missing_folders": ["HAR_IMU_IL/P_05/Repetition_1/Stand"],
        },
    )
    monkeypatch.setattr(
        external_har,
        "_imu_har_il_inventory",
        lambda **_kwargs: (sources, missing),
    )

    def download(
        source: external_har._IMUSourceFile,
    ) -> tuple[external_har._IMUSourceFile, bytes]:
        rows = 384
        values = np.sin(np.arange(rows, dtype=np.float64) / 17.0 + source.file_id)
        frame = pd.DataFrame(
            {name: values for name in ("Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z")}
        )
        frame["Activity_label"] = {"Walk": 5, "Sit": 1, "Stand": 2}[source.activity]
        payload = frame.to_csv(index=False).encode("utf-8")
        return replace(source, file_size=len(payload)), payload

    monkeypatch.setattr(external_har, "_download_imu_payload", download)
    result = external_har.load_imu_har_il(download_workers=1)
    assert result.participant_partition_plan is not None
    assert "imuharil:P_05" in result.participant_partition_plan.participant_roster
    observation = result.summary()["participant_partition_observation"]
    assert observation["participants_without_retained_windows"] == ["imuharil:P_05"]


def _har_pmd_archive(path: Path) -> Path:
    timestamps = np.arange(420, dtype=np.float64) / 60.0
    timestamps[2] = np.nan
    timestamps[200:] += 1.0
    timestamps[330:] -= 3.0
    frame = pd.DataFrame({"Time": timestamps})
    for channel, name in enumerate(
        ("LAccX", "LAccY", "LAccZ", "GyrX", "GyrY", "GyrZ", "GraX", "GraY", "GraZ")
    ):
        frame[name] = np.sin(np.arange(timestamps.size) / 13.0 + channel)
    frame.loc[[205, 208], "LAccX"] = np.nan
    with ZipFile(path, "x") as archive:
        for activity in ("still", "walking", "crutches", "walker", "manual"):
            archive.writestr(
                f"data_publish/1/phone/1_{activity}_phone_indoor.csv",
                frame.to_csv(index=False),
            )
    return path


def test_har_pmd_accounts_for_every_source_row_target_sample_and_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _har_pmd_archive(tmp_path / "har-pmd-accounting.zip")
    monkeypatch.setattr(external_har, "RemoteZip", lambda _url: ZipFile(path))
    result = external_har.load_har_pmd_native(
        participant_limit=1,
        environments=("indoor",),
        target_rate_hz=50.0,
        window_samples=64,
    )
    assert len(result.preprocessing_audit) == 5
    for audit in result.preprocessing_audit:
        assert audit["source_rows"] == audit["source_rows_accounted"] == 420
        assert audit["interpolated_target_samples"] == audit["target_samples_accounted"]
        assert audit["timestamp_gap_boundary_count"] >= 1
        assert audit["timestamp_nonincreasing_boundary_count"] >= 1
        assert audit["nonfinite_timestamp_rows"] == 1
        assert audit["short_timestamp_run_rows"] == 2
        assert audit["nonfinite_sensor_rows"] == 2
        assert audit["short_finite_signal_run_rows"] >= 2
        assert audit["dropped_tail_target_samples"] > 0
        assert audit["candidate_window_count"] > 0
    assert {
        "nonfinite_timestamp",
        "unusable_timestamp_run",
        "nonfinite_sensor",
        "short_finite_signal_run",
        "incomplete_resampled_tail",
    }.issubset({item["reason"] for item in result.exclusions})
    boundary = result.summary()["boundary_provenance"]
    assert boundary["source_boundary_unit"].startswith("provider scripted")
    assert boundary["repository_signal_grid_annotation_independent"] is True
    assert boundary["zero_lookahead_streaming_valid"] is False
    assert "stress test only" in boundary["evidence_scope"]


def _mock_sole_session(
    monkeypatch: pytest.MonkeyPatch,
    labels: np.ndarray[Any, Any],
    *,
    sensor_mutation: str | None = None,
) -> None:
    prefix = "C001/session"
    metadata = json.dumps(
        {
            "recording": {"sampling_hz": 270},
            "units": {"acc": "m/s^2", "gyr": "rad/s"},
            "session_id": "synthetic",
        }
    ).encode("utf-8")
    payloads = {f"{prefix}/meta.json": metadata, f"{prefix}/DataStruct.mat": b"synthetic"}

    class Archive:
        def __init__(self, _url: str) -> None:
            pass

        def __enter__(self) -> Archive:
            return self

        def __exit__(self, *_args: Any) -> None:
            pass

        def namelist(self) -> list[str]:
            return list(payloads)

        def getinfo(self, name: str) -> SimpleNamespace:
            payload = payloads[name]
            return SimpleNamespace(file_size=len(payload), CRC=zlib.crc32(payload))

        def read(self, name: str) -> bytes:
            return payloads[name]

    monkeypatch.setattr(external_har, "RemoteZip", Archive)
    monkeypatch.setattr(
        external_har,
        "_request_json",
        lambda *_args, **_kwargs: {
            "files": [{"key": "C001.zip", "links": {"content": "https://example.invalid/C001.zip"}}]
        },
    )
    timestamps = np.arange(8100, dtype=np.float64) * (1000.0 / 270.0)
    if sensor_mutation == "gap":
        timestamps[4050:] += 500.0
    elif sensor_mutation == "reset":
        timestamps[4050:] -= 5000.0
    base = np.column_stack(
        (
            np.sin(timestamps / 170.0),
            np.cos(timestamps / 110.0),
            timestamps / 10000.0,
        )
    )
    if sensor_mutation == "nonfinite":
        base[4050, 0] = np.nan
    monkeypatch.setattr(
        external_har,
        "_sole_session_arrays",
        lambda _payload: (
            timestamps,
            base,
            base / 20.0,
            base + np.array([0.0, 0.0, 9.81]),
            labels,
        ),
    )


def test_sole_session_observable_grid_is_invariant_to_camera_boundary_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_labels = np.array([[0.0, 0.0, 10000.0], [1.0, 10000.0, 20000.0], [2.0, 20000.0, 30000.0]])
    _mock_sole_session(monkeypatch, first_labels)
    first = external_har.load_sole_harmony(
        participant_limit=1,
        sessions_per_participant=1,
        boundary_mode="session_observable",
    )
    changed_labels = np.array([[0.0, 0.0, 9000.0], [1.0, 9000.0, 21000.0], [2.0, 21000.0, 30000.0]])
    _mock_sole_session(monkeypatch, changed_labels)
    changed = external_har.load_sole_harmony(
        participant_limit=1,
        sessions_per_participant=1,
        boundary_mode="session_observable",
    )
    assert first.observable_candidates is not None
    assert changed.observable_candidates is not None
    assert first.observable_candidates.audit() == changed.observable_candidates.audit()
    assert first.preprocessing_audit != changed.preprocessing_audit
    boundary = changed.summary()["boundary_provenance"]
    assert boundary["boundary_mode"] == "session_observable"
    assert boundary["repository_signal_grid_annotation_independent"] is True
    assert "post hoc" in boundary["evidence_scope"]
    assert all(":session-recording/" in value for value in changed.window_ids)


@pytest.mark.parametrize("sensor_mutation", ["gap", "reset", "nonfinite"])
def test_sole_session_observable_mode_uses_only_observable_run_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    sensor_mutation: str,
) -> None:
    labels = np.array([[0.0, 0.0, 10000.0], [1.0, 10000.0, 20000.0], [2.0, 20000.0, 31000.0]])
    _mock_sole_session(monkeypatch, labels, sensor_mutation=sensor_mutation)
    result = external_har.load_sole_harmony(
        participant_limit=1,
        sessions_per_participant=1,
        boundary_mode="session_observable",
    )
    assert len(result.preprocessing_audit) == 2
    assert result.observable_candidates is not None
    assert len(np.unique(result.observable_candidates.trial_ids)) == 1
    assert all(
        audit["within_declared_segment_transform_annotation_dependency"] is False
        and audit["segment_boundary_annotation_conditioned"] is False
        for audit in result.preprocessing_audit
    )


def test_sole_default_remains_explicit_camera_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = np.array([[0.0, 0.0, 10000.0], [1.0, 10000.0, 20000.0], [2.0, 20000.0, 30000.0]])
    _mock_sole_session(monkeypatch, labels)
    result = external_har.load_sole_harmony(participant_limit=1, sessions_per_participant=1)
    boundary = result.summary()["boundary_provenance"]
    assert boundary["boundary_mode"] == "camera_bout_oracle"
    assert boundary["repository_signal_grid_annotation_independent"] is False
    assert "not deployable" in boundary["evidence_scope"]
    assert result.preprocessing_audit
    assert all(
        audit["within_declared_segment_transform_annotation_dependency"] is False
        and audit["segment_boundary_annotation_conditioned"] is True
        for audit in result.preprocessing_audit
    )
    assert result.observable_candidates is None


def test_unknown_sole_boundary_mode_fails_before_provider_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("unknown boundary mode must fail before provider access")

    monkeypatch.setattr(external_har, "_request_json", forbidden)
    with pytest.raises(ValueError, match="boundary mode"):
        external_har.load_sole_harmony(boundary_mode="camera_labels_are_deployable")


def test_boundary_provenance_rejects_an_incomplete_claim() -> None:
    count = 3
    data = external_har.ExternalHARWindows(
        dataset_id="synthetic",
        channel_lane="derived-gravity-9ch",
        sampling_rate_hz=50.0,
        signals=np.zeros((count, 128, 6), dtype=np.float32),
        gravity=np.ones((count, 128, 3), dtype=np.float32),
        labels=np.arange(count, dtype=np.int64),
        participant_ids=np.array(["p0", "p1", "p2"]),
        session_ids=np.array(["s0", "s1", "s2"]),
        trial_ids=np.array(["t0", "t1", "t2"]),
        window_ids=np.array(["w0", "w1", "w2"]),
        receipts=(
            external_har.SourceReceipt(
                "synthetic",
                "https://example.invalid/source",
                None,
                1,
                1,
                hashlib.sha256(b"x").hexdigest(),
            ),
        ),
        boundary_provenance={"protocol_id": external_har.BOUNDARY_PROVENANCE_PROTOCOL},
    )
    with pytest.raises(ValueError, match="boundary provenance"):
        data.validate()
