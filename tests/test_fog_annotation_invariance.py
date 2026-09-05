"""Regression evidence for the annotation-independent FoG signal contract."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from inclusive_shift_har.data import external_har


def _fog_frame(labels: np.ndarray[Any, Any]) -> pd.DataFrame:
    time = np.arange(labels.size, dtype=np.float64) / 60.0
    return pd.DataFrame(
        {
            "timestamp": time,
            "back_acc_x": np.sin(2.3 * time),
            "back_acc_y": np.cos(1.7 * time),
            "back_acc_z": 1.0 + 0.1 * np.sin(3.1 * time),
            "back_gyro_x": np.sin(8.2 * time),
            "back_gyro_y": np.cos(4.7 * time),
            "back_gyro_z": np.sin(0.4 * time),
            "activity": labels,
            "subjectID": 1,
            "sessionID": 1,
            "taskID": 1,
        }
    )


def _install_fog_payload(monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame) -> None:
    payload = frame.to_csv(index=False).encode("utf-8")
    monkeypatch.setattr(external_har, "_request_bytes", lambda *_args, **_kwargs: payload)
    receipt = external_har.SourceReceipt(
        dataset_id="fog_star_v3",
        locator="https://example.invalid/synthetic-fog",
        member=None,
        declared_size_bytes=len(payload),
        received_size_bytes=len(payload),
        computed_sha256=hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(external_har, "_receipt", lambda **_kwargs: replace(receipt))


def test_fog_resampling_inputs_do_not_depend_on_annotation_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = external_har.resample_uniform
    calls: list[np.ndarray[Any, Any]] = []

    def capture(values: np.ndarray[Any, Any], **kwargs: Any) -> np.ndarray[Any, Any]:
        calls.append(values.copy())
        return original(values, **kwargs)

    monkeypatch.setattr(external_har, "resample_uniform", capture)
    _install_fog_payload(monkeypatch, _fog_frame(np.repeat([1.0, 2.0, 3.0], 480)))
    external_har.load_fog_star()
    first = calls.copy()
    calls.clear()
    changed = np.concatenate((np.full(511, 1.0), np.full(449, 2.0), np.full(480, 3.0)))
    changed[700] = np.nan
    _install_fog_payload(monkeypatch, _fog_frame(changed))
    external_har.load_fog_star()
    assert len(first) == len(calls)
    for left, right in zip(first, calls, strict=True):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("target_rate", [30.0, 50.0, 60.0, 90.0])
@pytest.mark.parametrize("label_change", ["shift", "permute", "missing", "remove"])
def test_physical_grid_and_features_are_invariant_to_all_annotation_changes(
    target_rate: float,
    label_change: str,
) -> None:
    frame = _fog_frame(np.repeat([1.0, 2.0, 3.0], 480))
    total = frame[["back_acc_x", "back_acc_y", "back_acc_z"]].to_numpy(dtype=np.float64)
    segment = external_har.resample_physical_segment(
        timestamps=frame["timestamp"].to_numpy(dtype=np.float64),
        acceleration=total,
        gyroscope=total * 0.03,
        gravity=None,
        source_rate_hz=60.0,
        target_rate_hz=target_rate,
        gravity_cutoff_hz=0.3,
        window_samples=128,
    )
    baseline_audit = segment.audit()
    original_signals, original_gravity = segment.signals.copy(), segment.gravity.copy()
    labels = frame["activity"].to_numpy(dtype=np.float64)
    if label_change == "shift":
        labels = np.roll(labels, 31)
    elif label_change == "permute":
        labels = np.random.default_rng(123).permutation(labels)
    elif label_change == "missing":
        labels[127:133] = np.nan
    else:
        labels[:] = np.nan
    accumulator = external_har._WindowAccumulator.empty()
    accumulator.add_annotated_segment(
        segment,
        source_labels=labels,
        label_map={1: 0, 2: 1, 3: 2},
        participant="p1",
        session="s1",
        trial="t1",
        run_index=0,
    )
    assert segment.audit() == baseline_audit
    np.testing.assert_array_equal(segment.signals, original_signals)
    np.testing.assert_array_equal(segment.gravity, original_gravity)
    for key, value in baseline_audit.items():
        assert accumulator.preprocessing_audit[0][key] == value
    report = accumulator.preprocessing_audit[0]
    excluded = report["excluded_candidate_indices"]
    assert len(accumulator.windows) + sum(len(items) for items in excluded.values()) == len(
        segment.candidate_starts
    )
    if label_change == "remove":
        assert not accumulator.windows


@pytest.mark.parametrize(
    "boundary", ["timestamp_gap", "timestamp_reset", "nan_signal", "nan_timestamp"]
)
def test_fog_loader_retains_observable_boundaries_and_global_candidate_grid(
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    labels = np.repeat([1.0, 2.0, 3.0], 600)
    frame = _fog_frame(labels)
    if boundary == "timestamp_gap":
        frame.loc[300:, "timestamp"] += 1.0
    elif boundary == "timestamp_reset":
        frame.loc[300:, "timestamp"] -= 5.0
    elif boundary == "nan_signal":
        frame.loc[300, "back_acc_x"] = np.nan
    else:
        frame.loc[300, "timestamp"] = np.nan
    _install_fog_payload(monkeypatch, frame)
    first = external_har.load_fog_star()
    frame.loc[700:710, "activity"] = np.nan
    frame.loc[620:680, "activity"] = 3.0
    _install_fog_payload(monkeypatch, frame)
    second = external_har.load_fog_star()
    assert len(first.preprocessing_audit) == len(second.preprocessing_audit) == 2
    for left, right in zip(first.preprocessing_audit, second.preprocessing_audit, strict=True):
        for key in left:
            if key not in {"admitted_candidate_indices", "excluded_candidate_indices"}:
                assert left[key] == right[key]
    common = set(first.window_ids) & set(second.window_ids)
    assert common
    for identifier in common:
        left_index = int(np.flatnonzero(first.window_ids == identifier)[0])
        right_index = int(np.flatnonzero(second.window_ids == identifier)[0])
        np.testing.assert_array_equal(first.signals[left_index], second.signals[right_index])
        np.testing.assert_array_equal(first.gravity[left_index], second.gravity[right_index])


def test_downsampling_does_not_hide_a_short_missing_annotation() -> None:
    frame = _fog_frame(np.ones(600))
    segment = external_har.resample_physical_segment(
        timestamps=frame["timestamp"].to_numpy(dtype=np.float64),
        acceleration=np.ones((600, 3)),
        gyroscope=np.zeros((600, 3)),
        gravity=None,
        source_rate_hz=60.0,
        target_rate_hz=30.0,
        gravity_cutoff_hz=0.3,
        window_samples=128,
    )
    labels = np.ones(600)
    labels[1] = np.nan  # This source sample is absent from the 30 Hz projected grid.
    accumulator = external_har._WindowAccumulator.empty()
    accumulator.add_annotated_segment(
        segment,
        source_labels=labels,
        label_map={1: 0},
        participant="p",
        session="s",
        trial="t",
        run_index=0,
    )
    assert accumulator.preprocessing_audit[0]["excluded_candidate_indices"][
        "missing_or_unsupported_annotation"
    ] == [0]
    assert accumulator.preprocessing_audit[0]["admitted_candidate_indices"] == [1]


@pytest.mark.parametrize("change", ["shift", "permute", "missing", "remove"])
def test_task_annotations_cannot_reset_the_fog_signal_pipeline(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    frame = _fog_frame(np.repeat([1.0, 2.0, 3.0], 600))
    frame["taskID"] = np.repeat([1.0, 2.0, 3.0], 600)
    _install_fog_payload(monkeypatch, frame)
    original = external_har.load_fog_star()
    if change == "shift":
        frame["taskID"] = np.roll(frame["taskID"].to_numpy(), 71)
    elif change == "permute":
        frame["taskID"] = np.random.default_rng(123).permutation(frame["taskID"].to_numpy())
    elif change == "missing":
        frame["taskID"] = np.nan
    else:
        frame = frame.drop(columns=["taskID"])
    _install_fog_payload(monkeypatch, frame)
    changed = external_har.load_fog_star()
    assert len(original.preprocessing_audit) == 1
    assert original.preprocessing_audit == changed.preprocessing_audit
    np.testing.assert_array_equal(original.signals, changed.signals)
    np.testing.assert_array_equal(original.gravity, changed.gravity)
    np.testing.assert_array_equal(original.window_ids, changed.window_ids)
    np.testing.assert_array_equal(original.labels, changed.labels)
