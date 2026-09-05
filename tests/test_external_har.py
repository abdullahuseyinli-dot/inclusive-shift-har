from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

import inclusive_shift_har.data.external_har as external_har
from inclusive_shift_har.data.external_har import (
    ExternalHARWindows,
    SourceReceipt,
    _contiguous_label_runs,
    _contiguous_signal_runs,
    _IMUSourceFile,
    _sole_session_arrays,
    _WindowAccumulator,
    causal_gravity_lowpass,
    concatenate_external_windows,
    participant_fold_assignment,
    resample_uniform,
)
from inclusive_shift_har.experiments.cross_dataset_neural import _InferenceWindows
from inclusive_shift_har.experiments.sole_harmony_temporal import (
    _causal_probability,
    _temporal_block_ids,
)
from inclusive_shift_har.models import build_baseline


def _receipt(dataset_id: str) -> SourceReceipt:
    return SourceReceipt(
        dataset_id=dataset_id,
        locator="https://example.invalid/immutable",
        member=None,
        declared_size_bytes=1,
        received_size_bytes=1,
        computed_sha256="0" * 64,
    )


def _windows(dataset_id: str, lane: str = "derived-gravity-9ch") -> ExternalHARWindows:
    labels = np.array([0, 1, 2], dtype=np.int64)
    return ExternalHARWindows(
        dataset_id=dataset_id,
        channel_lane=lane,
        sampling_rate_hz=50.0,
        signals=np.zeros((3, 128, 6), dtype=np.float32),
        gravity=np.ones((3, 128, 3), dtype=np.float32),
        labels=labels,
        participant_ids=np.array([f"{dataset_id}:p{index}" for index in range(3)]),
        session_ids=np.array([f"{dataset_id}:s{index}" for index in range(3)]),
        trial_ids=np.array([f"{dataset_id}:t{index}" for index in range(3)]),
        window_ids=np.array([f"{dataset_id}:w{index}" for index in range(3)]),
        receipts=(_receipt(dataset_id),),
    )


def test_causal_gravity_is_trial_local_and_reconstructs_total_acceleration() -> None:
    total = np.column_stack(
        (
            np.linspace(0.0, 2.0, 180),
            np.zeros(180),
            np.full(180, 9.80665),
        )
    )
    gravity = causal_gravity_lowpass(total, sampling_rate_hz=60.0, cutoff_hz=0.3)
    linear = total - gravity
    np.testing.assert_allclose(linear + gravity, total, rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(gravity[0], total[0])
    assert gravity.shape == total.shape


def test_causal_gravity_state_is_not_reset_at_an_activity_boundary() -> None:
    total = np.zeros((128, 3), dtype=np.float64)
    total[64:, 2] = 9.80665
    full_trial = causal_gravity_lowpass(total, sampling_rate_hz=50.0, cutoff_hz=0.3)
    label_reset = np.concatenate(
        (
            causal_gravity_lowpass(total[:64], sampling_rate_hz=50.0, cutoff_hz=0.3),
            causal_gravity_lowpass(total[64:], sampling_rate_hz=50.0, cutoff_hz=0.3),
        ),
        axis=0,
    )
    assert full_trial[64, 2] < 1.0
    assert label_reset[64, 2] == pytest.approx(9.80665)


def test_contiguous_runs_break_on_label_reset_and_timestamp_gap() -> None:
    timestamps = np.array([0.0, 0.02, 0.04, 0.06, 0.50, 0.52, 0.01, 0.03])
    labels = np.array([0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    assert _contiguous_label_runs(
        timestamps, labels, nominal_rate_hz=50.0, maximum_gap_factor=3.0
    ) == [(0, 2), (2, 4), (4, 6), (6, 8)]


def test_physical_signal_runs_do_not_use_activity_boundaries() -> None:
    timestamps = np.arange(8, dtype=np.float64) / 50.0
    signals = np.ones((8, 3), dtype=np.float64)
    assert _contiguous_signal_runs(
        timestamps,
        signals,
        nominal_rate_hz=50.0,
        maximum_gap_factor=3.0,
    ) == [(0, 8)]


def test_physical_signal_runs_ignore_missing_activity_annotations() -> None:
    timestamps = np.arange(8, dtype=np.float64) / 50.0
    signals = np.ones((8, 3), dtype=np.float64)
    activities = np.array([0.0, 0.0, np.nan, np.nan, 1.0, 1.0, 2.0, 2.0])
    # Activity values are intentionally not an input to the physical-run detector.
    assert _contiguous_signal_runs(
        timestamps,
        signals,
        nominal_rate_hz=50.0,
        maximum_gap_factor=3.0,
    ) == [(0, activities.size)]


def test_uniform_windowing_never_crosses_the_supplied_trial() -> None:
    samples = 360
    total = np.column_stack((np.zeros(samples), np.zeros(samples), np.full(samples, 9.80665)))
    gyro = np.zeros_like(total)
    accumulator = _WindowAccumulator.empty()
    accumulator.add_uniform_trial(
        total_acceleration=total,
        gyroscope=gyro,
        gravity=None,
        source_rate_hz=60.0,
        target_rate_hz=50.0,
        gravity_cutoff_hz=0.3,
        label=1,
        participant="source:p1",
        session="source:p1:s1",
        trial="source:p1:s1:t1",
        run_index=0,
        window_samples=128,
    )
    assert len(accumulator.windows) == 2
    assert all("source:p1:s1:t1" in item for item in accumulator.windows)
    assert all(item.shape == (128, 6) for item in accumulator.signals)


def test_participant_folds_are_deterministic_and_exclusive() -> None:
    participants = [f"p{index:02d}" for index in range(15)]
    first = participant_fold_assignment(participants, fold_count=5, seed=11)
    second = participant_fold_assignment(list(reversed(participants)), fold_count=5, seed=11)
    assert first == second
    assert set(first) == set(participants)
    assert set(first.values()) == set(range(5))
    assert all(list(first.values()).count(fold) == 3 for fold in range(5))


def test_dataset_validation_and_lane_separation() -> None:
    first = replace(
        _windows("one"),
        gravity_source="causal_lowpass_from_total_acceleration",
        gravity_cutoff_hz=0.3,
    )
    second = replace(
        _windows("two"),
        gravity_source="causal_lowpass_from_total_acceleration",
        gravity_cutoff_hz=0.3,
    )
    first.validate()
    combined = concatenate_external_windows((first, second), dataset_id="combined")
    assert combined.labels.size == 6
    assert combined.gravity_cutoff_hz == 0.3
    assert combined.summary()["gravity_preprocessing"] == {
        "source": "causal_lowpass_from_total_acceleration",
        "causal_lowpass_cutoff_hz": 0.3,
    }
    with pytest.raises(ValueError, match="one aligned lane"):
        concatenate_external_windows(
            (first, _windows("native", "native-gravity-9ch")), dataset_id="invalid"
        )
    with pytest.raises(ValueError, match="one aligned lane"):
        concatenate_external_windows(
            (first, replace(second, gravity_cutoff_hz=0.4)), dataset_id="invalid-cutoff"
        )


def test_dataset_validation_rejects_an_invalid_declared_gravity_cutoff() -> None:
    with pytest.raises(ValueError, match="gravity cutoff"):
        replace(_windows("bad-cutoff"), gravity_cutoff_hz=25.0).validate()


def test_dataset_summary_respects_a_declared_five_class_ontology() -> None:
    labels = np.arange(5, dtype=np.int64)
    data = ExternalHARWindows(
        dataset_id="five-class",
        channel_lane="native-gravity-9ch",
        sampling_rate_hz=50.0,
        signals=np.zeros((5, 128, 6), dtype=np.float32),
        gravity=np.ones((5, 128, 3), dtype=np.float32),
        labels=labels,
        participant_ids=np.array([f"p{index}" for index in labels]),
        session_ids=np.array([f"s{index}" for index in labels]),
        trial_ids=np.array([f"t{index}" for index in labels]),
        window_ids=np.array([f"w{index}" for index in labels]),
        receipts=(_receipt("five-class"),),
        class_names=("a", "b", "c", "d", "e"),
    )
    data.validate()
    summary = data.summary()
    assert summary["class_window_counts"] == {
        "a": 1,
        "b": 1,
        "c": 1,
        "d": 1,
        "e": 1,
    }
    assert summary["participants_with_all_classes"] == 0
    assert summary["perfect_prediction_mean_participant_macro_f1_ceiling"] == 0.2


def test_uniform_resampling_uses_expected_length() -> None:
    values = np.arange(600, dtype=np.float64).reshape(200, 3)
    result = resample_uniform(values, source_rate_hz=60.0, target_rate_hz=50.0)
    assert result.shape == (167, 3)
    assert np.isfinite(result).all()


def test_tinyhar_standalone_baseline_contract() -> None:
    import torch

    model = build_baseline("tinyhar", num_classes=3)
    output = model(torch.randn(2, 128, 6))
    assert output.logits.shape == (2, 3)
    assert output.content is not None and output.content.shape == (2, 96)


def test_neural_inference_view_has_no_label_or_weight_channel() -> None:
    import torch

    signals = np.arange(2 * 128 * 6, dtype=np.float32).reshape(2, 128, 6)
    dataset = _InferenceWindows(
        signals,
        np.array([1], dtype=np.int64),
        np.zeros((1, 6), dtype=np.float32),
        np.ones((1, 6), dtype=np.float32),
    )
    item = dataset[0]
    assert isinstance(item, torch.Tensor)
    np.testing.assert_array_equal(item.numpy(), signals[1])


def test_imu_loader_excludes_an_entire_participant_before_windowing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_labels = {"Walk": 5, "Sit": 1, "Stand": 2}
    sources = tuple(
        _IMUSourceFile(
            participant=participant,
            repetition="Repetition_1",
            activity=activity,
            file_id=index,
            filename=f"HAR_IMU_IL/{participant}/Repetition_1/{activity}/Body-WT.csv",
            file_size=0,
            locator=f"https://example.invalid/{participant}/{activity}",
        )
        for index, (participant, activity) in enumerate(
            (participant, activity)
            for participant in ("P_01", "P_02")
            for activity in ("Walk", "Sit", "Stand")
        )
    )

    def payload(source: _IMUSourceFile) -> tuple[_IMUSourceFile, bytes]:
        row_count = 160
        invalid = source.participant == "P_02"
        values = np.full(row_count, np.nan if invalid else 1.0)
        frame = pd.DataFrame(
            {
                "Acc_X": values,
                "Acc_Y": values,
                "Acc_Z": values,
                "Gyr_X": values,
                "Gyr_Y": values,
                "Gyr_Z": values,
                "Activity Label": np.full(row_count, provider_labels[source.activity]),
            }
        )
        encoded = frame.to_csv(index=False).encode("utf-8")
        return replace(source, file_size=len(encoded)), encoded

    monkeypatch.setattr(
        external_har,
        "_imu_har_il_inventory",
        lambda **_kwargs: (sources, ()),
    )
    monkeypatch.setattr(external_har, "_download_imu_payload", payload)
    result = external_har.load_imu_har_il(download_workers=1)
    assert set(result.participant_ids.tolist()) == {"imuharil:P_01"}
    assert set(result.labels.tolist()) == {0, 1, 2}
    assert result.exclusions[0]["participant_id"] == "imuharil:P_02"
    assert len(result.exclusions[0]["invalid_trials"]) == 3
    assert len(result.source_issues) == 6


def test_sole_harmony_matlab_73_schema_is_transposed_and_aligned() -> None:
    import io

    import h5py  # type: ignore[import-untyped]

    stream = io.BytesIO()
    with h5py.File(stream, "w") as handle:
        root = handle.create_group("DataStruct")
        right = root.create_group("InsoleR")
        for name, values in {
            "t_ms": np.arange(6, dtype=np.float64),
            "lin_acc_x": np.arange(6, dtype=np.float64),
            "lin_acc_y": np.arange(6, dtype=np.float64) + 1.0,
            "lin_acc_z": np.arange(6, dtype=np.float64) + 2.0,
            "gyr_x": np.zeros(6),
            "gyr_y": np.ones(6),
            "gyr_z": np.full(6, 2.0),
            "raw_acc_x": np.arange(6, dtype=np.float64) + 3.0,
            "raw_acc_y": np.arange(6, dtype=np.float64) + 4.0,
            "raw_acc_z": np.arange(6, dtype=np.float64) + 5.0,
        }.items():
            right.create_dataset(name, data=values.reshape(1, -1))
        root.create_dataset(
            "labelsCam",
            data=np.array([[0.0, 1.0], [0.0, 3.0], [3.0, 6.0]], dtype=np.float64),
        )
    timestamps, linear, gyro, raw, labels = _sole_session_arrays(stream.getvalue())
    assert timestamps.shape == (6,)
    assert linear.shape == gyro.shape == raw.shape == (6, 3)
    np.testing.assert_array_equal(labels, np.array([[0.0, 0.0, 3.0], [1.0, 3.0, 6.0]]))


def test_causal_temporal_probability_uses_only_past_values_and_resets() -> None:
    data = _windows("temporal")
    data = replace(
        data,
        window_ids=np.array(
            [
                "p/s/t/run-0000-finite-000/window-000000",
                "p/s/t/run-0000-finite-000/window-000001",
                "p/s/t/run-0001-finite-000/window-000000",
            ]
        ),
    )
    probabilities = np.array([[0.9, 0.1, 0.0], [0.3, 0.7, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    blocks = _temporal_block_ids(data)
    smoothed = _causal_probability(probabilities, blocks, width=3)
    np.testing.assert_allclose(smoothed[0], probabilities[0])
    np.testing.assert_allclose(smoothed[1], np.array([0.6, 0.4, 0.0]))
    np.testing.assert_allclose(smoothed[2], probabilities[2])
