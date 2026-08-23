"""Boundary and identity tests for non-overlapping released-block windows."""

from __future__ import annotations

from inclusive_shift_har.data.windowing import ReleasedBlock, windows_from_released_block


def _window_block() -> ReleasedBlock:
    return ReleasedBlock(
        released_run_id="released_run_001",
        subject_id="1",
        activity_label="Walking",
        start_row_inclusive=1001,
        end_row_inclusive=1300,
        row_count=300,
    )


def test_nonoverlapping_windows_stay_inside_block_and_drop_only_tail() -> None:
    result = windows_from_released_block(
        _window_block(),
        dataset_id="inclusivehar_v4",
        source_artifact_sha256="a" * 64,
        protocol_sha256="b" * 64,
        partition="source_train",
        canonical_labels={"functional_core": "mobility"},
    )

    assert [
        (window.start_row_inclusive, window.end_row_inclusive) for window in result.windows
    ] == [
        (1001, 1128),
        (1129, 1256),
    ]
    assert result.used_rows == 256
    assert result.dropped_tail_rows == 44
    assert result.windows[-1].end_row_inclusive < result.block.end_row_inclusive
    assert result.conditional_hidden_join_crossing_window_bound == 2
    assert all(window.trial_id is None for window in result.windows)
    assert all(window.trial_status == "unrecoverable" for window in result.windows)


def test_window_ids_are_deterministic_and_protocol_bound() -> None:
    first = windows_from_released_block(
        _window_block(),
        dataset_id="inclusivehar_v4",
        source_artifact_sha256="a" * 64,
        protocol_sha256="b" * 64,
        partition="source_train",
        canonical_labels={"functional_core": "mobility"},
    )
    second = windows_from_released_block(
        _window_block(),
        dataset_id="inclusivehar_v4",
        source_artifact_sha256="a" * 64,
        protocol_sha256="b" * 64,
        partition="source_train",
        canonical_labels={"functional_core": "mobility"},
    )
    changed_protocol = windows_from_released_block(
        _window_block(),
        dataset_id="inclusivehar_v4",
        source_artifact_sha256="a" * 64,
        protocol_sha256="c" * 64,
        partition="source_train",
        canonical_labels={"functional_core": "mobility"},
    )

    assert [window.window_id for window in first.windows] == [
        window.window_id for window in second.windows
    ]
    assert [window.window_id for window in first.windows] != [
        window.window_id for window in changed_protocol.windows
    ]


def test_overlapping_window_stride_is_rejected() -> None:
    try:
        windows_from_released_block(
            _window_block(),
            dataset_id="inclusivehar_v4",
            source_artifact_sha256="a" * 64,
            protocol_sha256="b" * 64,
            partition="source_train",
            canonical_labels={},
            stride_samples=64,
        )
    except ValueError as exc:
        assert "stride equal" in str(exc)
    else:
        raise AssertionError("overlapping stride was accepted")
