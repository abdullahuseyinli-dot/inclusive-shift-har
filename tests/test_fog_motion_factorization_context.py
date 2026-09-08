from __future__ import annotations

from copy import deepcopy

import numpy as np

from inclusive_shift_har.data.external_har import UniformPhysicalSegment
from inclusive_shift_har.experiments.fog_motion_factorization import (
    context_reason_counts,
    materialize_fog_motion_contexts,
)


def _segment(*, start: float, samples: int, marker: float) -> UniformPhysicalSegment:
    timestamps = start + np.arange(samples, dtype=np.float64) / 50.0
    position = np.arange(samples, dtype=np.float64)
    signals = np.column_stack([marker + position + 1_000.0 * channel for channel in range(6)])
    gravity = np.column_stack([marker + position + 10_000.0 * channel for channel in range(3)])
    return UniformPhysicalSegment(
        signals=signals,
        gravity=gravity,
        source_timestamps=timestamps.copy(),
        timestamps=timestamps,
        candidate_starts=np.arange(samples // 128, dtype=np.int64) * 128,
        source_rate_hz=60.0,
        target_rate_hz=50.0,
        window_samples=128,
    )


def _row(
    *,
    run: int,
    window: int,
    start_time: float,
    ankle_run: int | None,
    activity: str = "ignored",
) -> dict[str, object]:
    participant = "fogstar:001"
    session = f"{participant}:session-001"
    available = ankle_run is not None
    return {
        "window_id": (
            f"{participant}/{session}/{session}:recording/"
            f"run-{run:04d}-finite-000/window-{window:06d}"
        ),
        "participant_id": participant,
        "session_id": session,
        "start_time": start_time,
        "end_time_exclusive": start_time + 128.0 / 50.0,
        "ankle_finite_run_contains_window": available,
        "ankle_own_continuous_50hz_grid_matches_all128_times": available,
        "ankle_containing_run_indices": [] if ankle_run is None else [ankle_run],
        "ankle_matching_grid_run_indices": [] if ankle_run is None else [ankle_run],
        # This deliberately proves annotation-like extras cannot define boundaries.
        "activity": activity,
    }


def test_context_is_right_aligned_and_never_borrows_future_samples() -> None:
    key = ("fogstar:001", "fogstar:001:session-001")
    back = _segment(start=0.0, samples=700, marker=10.0)
    ankle = _segment(start=0.0, samples=700, marker=20.0)
    row = _row(run=0, window=3, start_time=384.0 / 50.0, ankle_run=0)

    result = materialize_fog_motion_contexts(
        [row],
        np.asarray([True], dtype=np.bool_),
        {key: {0: back}},
        {key: {0: ankle}},
    )

    # The current query ends at sample 512, so [e-500,e) is exactly [12,512).
    assert result.signals.shape == (1, 500, 6)
    assert np.array_equal(result.signals[0], ankle.signals[12:512])
    assert np.array_equal(result.derived_gravity[0], ankle.gravity[12:512])
    assert np.array_equal(result.timestamps[0], ankle.timestamps[12:512])
    assert result.observation_mask[0].all()
    assert result.full_context_mask.tolist() == [True]
    assert np.array_equal(result.signals[0, -128:], ankle.signals[384:512])
    assert float(result.signals[0, -1, 0]) == float(ankle.signals[511, 0])
    assert float(ankle.signals[512, 0]) not in result.signals[0, :, 0]


def test_overlapping_reset_clocks_use_explicit_physical_run_only() -> None:
    key = ("fogstar:001", "fogstar:001:session-001")
    # Both runs reset to the same clock. Only run 1 is named by the observable ID
    # and ankle matching receipt; timestamp matching alone would be ambiguous.
    back_zero = _segment(start=0.0, samples=700, marker=1_000.0)
    back_one = _segment(start=0.0, samples=700, marker=2_000.0)
    ankle_zero = _segment(start=0.0, samples=700, marker=3_000.0)
    ankle_one = _segment(start=0.0, samples=700, marker=4_000.0)
    row = _row(run=1, window=3, start_time=384.0 / 50.0, ankle_run=1)

    result = materialize_fog_motion_contexts(
        [row],
        np.asarray([True], dtype=np.bool_),
        {key: {0: back_zero, 1: back_one}},
        {key: {0: ankle_zero, 1: ankle_one}},
    )

    assert np.array_equal(result.signals[0], ankle_one.signals[12:512])
    assert not np.array_equal(result.signals[0], ankle_zero.signals[12:512])
    assert result.alignment_receipts[0]["back_run_index"] == 1
    assert result.alignment_receipts[0]["ankle_run_index"] == 1


def test_back_break_clips_continuous_ankle_history_and_preserves_current_suffix() -> None:
    key = ("fogstar:001", "fogstar:001:session-001")
    # The ankle has ten seconds before the back run begins. The observable query is
    # the first back window, so those ankle samples must stay masked.
    back = _segment(start=10.0, samples=300, marker=100.0)
    ankle = _segment(start=0.0, samples=900, marker=200.0)
    row = _row(run=0, window=0, start_time=10.0, ankle_run=0)

    result = materialize_fog_motion_contexts(
        [row],
        np.asarray([True], dtype=np.bool_),
        {key: {0: back}},
        {key: {0: ankle}},
    )

    assert result.full_context_mask.tolist() == [False]
    assert not result.observation_mask[0, :-128].any()
    assert result.observation_mask[0, -128:].all()
    assert np.all(result.signals[0, :-128] == 0.0)
    assert np.array_equal(result.signals[0, -128:], ankle.signals[500:628])
    assert result.alignment_receipts[0]["history_sample_count"] == 128
    assert result.alignment_receipts[0]["status"] == "insufficient_back_run_history"


def test_ankle_break_and_unavailable_rows_remain_explicit_in_masks() -> None:
    key = ("fogstar:001", "fogstar:001:session-001")
    back = _segment(start=0.0, samples=700, marker=100.0)
    # At the fourth back window the ankle run has only 312 samples of history.
    ankle = _segment(start=4.0, samples=500, marker=200.0)
    rows = [
        _row(run=0, window=3, start_time=384.0 / 50.0, ankle_run=0),
        _row(run=0, window=4, start_time=512.0 / 50.0, ankle_run=None),
    ]

    result = materialize_fog_motion_contexts(
        rows,
        np.asarray([True, False], dtype=np.bool_),
        {key: {0: back}},
        {key: {0: ankle}},
    )

    assert result.observation_mask[0, -312:].all()
    assert not result.observation_mask[0, :-312].any()
    assert result.alignment_receipts[0]["status"] == "insufficient_ankle_run_history"
    assert not result.observation_mask[1].any()
    assert np.all(result.signals[1] == 0.0)
    assert result.alignment_receipts[1]["status"] == "current_ankle_unavailable"
    assert context_reason_counts(result) == {
        "current_ankle_unavailable": 1,
        "insufficient_ankle_run_history": 1,
    }


def test_context_boundaries_and_values_are_annotation_invariant() -> None:
    key = ("fogstar:001", "fogstar:001:session-001")
    segment = _segment(start=0.0, samples=700, marker=20.0)
    original = _row(
        run=0,
        window=3,
        start_time=384.0 / 50.0,
        ankle_run=0,
        activity="walking",
    )
    changed = deepcopy(original)
    changed["activity"] = "standing"
    changed["fog_label"] = 99

    first = materialize_fog_motion_contexts(
        [original],
        np.asarray([True], dtype=np.bool_),
        {key: {0: segment}},
        {key: {0: segment}},
    )
    second = materialize_fog_motion_contexts(
        [changed],
        np.asarray([True], dtype=np.bool_),
        {key: {0: segment}},
        {key: {0: segment}},
    )

    assert np.array_equal(first.signals, second.signals)
    assert np.array_equal(first.timestamps, second.timestamps)
    assert np.array_equal(first.observation_mask, second.observation_mask)
    assert first.alignment_receipts == second.alignment_receipts
