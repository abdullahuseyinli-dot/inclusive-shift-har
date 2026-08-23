"""Deterministic non-overlapping windows over released InclusiveHAR blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inclusive_shift_har.manifests.canonical import canonical_json_sha256


@dataclass(frozen=True)
class ReleasedBlock:
    released_run_id: str
    subject_id: str
    activity_label: str
    start_row_inclusive: int
    end_row_inclusive: int
    row_count: int

    def validate(self) -> None:
        if not self.released_run_id or not self.subject_id or not self.activity_label:
            raise ValueError("released block identifiers must be non-empty")
        if self.start_row_inclusive <= 0:
            raise ValueError("released block start row must be positive")
        if self.end_row_inclusive < self.start_row_inclusive:
            raise ValueError("released block end precedes start")
        observed = self.end_row_inclusive - self.start_row_inclusive + 1
        if observed != self.row_count:
            raise ValueError(
                f"released block row_count mismatch: declared {self.row_count}, interval {observed}"
            )


@dataclass(frozen=True)
class WindowRecord:
    window_id: str
    raw_interval_id: str
    released_run_id: str
    subject_id: str
    activity_label: str
    canonical_labels: dict[str, str]
    partition: str
    window_ordinal: int
    start_row_inclusive: int
    end_row_inclusive: int
    length_samples: int
    stride_samples: int
    trial_id: None
    trial_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "activity_label": self.activity_label,
            "canonical_labels": dict(sorted(self.canonical_labels.items())),
            "end_row_inclusive": self.end_row_inclusive,
            "length_samples": self.length_samples,
            "partition": self.partition,
            "raw_interval_id": self.raw_interval_id,
            "released_run_id": self.released_run_id,
            "start_row_inclusive": self.start_row_inclusive,
            "subject_id": self.subject_id,
            "trial_id": self.trial_id,
            "trial_status": self.trial_status,
            "window_id": self.window_id,
            "window_ordinal": self.window_ordinal,
            "stride_samples": self.stride_samples,
        }


@dataclass(frozen=True)
class BlockWindowResult:
    block: ReleasedBlock
    windows: tuple[WindowRecord, ...]
    dropped_tail_rows: int
    used_rows: int
    conditional_hidden_join_crossing_window_bound: int

    def summary(self) -> dict[str, Any]:
        return {
            "activity_label": self.block.activity_label,
            "conditional_hidden_join_crossing_window_bound": self.conditional_hidden_join_crossing_window_bound,
            "dropped_tail_rows": self.dropped_tail_rows,
            "end_row_inclusive": self.block.end_row_inclusive,
            "released_run_id": self.block.released_run_id,
            "row_count": self.block.row_count,
            "start_row_inclusive": self.block.start_row_inclusive,
            "subject_id": self.block.subject_id,
            "used_rows": self.used_rows,
            "window_count": len(self.windows),
        }


def _stable_window_id(
    *,
    dataset_id: str,
    source_artifact_sha256: str,
    protocol_sha256: str,
    block: ReleasedBlock,
    start_row_inclusive: int,
    end_row_inclusive: int,
    length_samples: int,
    stride_samples: int,
    window_ordinal: int,
) -> str:
    identity = {
        "activity_label": block.activity_label,
        "dataset_id": dataset_id,
        "end_row_inclusive": end_row_inclusive,
        "length_samples": length_samples,
        "local_offset_samples": window_ordinal * stride_samples,
        "protocol_sha256": protocol_sha256,
        "released_run_id": block.released_run_id,
        "source_artifact_sha256": source_artifact_sha256,
        "start_row_inclusive": start_row_inclusive,
        "subject_id": block.subject_id,
        "stride_samples": stride_samples,
    }
    return f"w_{canonical_json_sha256(identity)}"


def windows_from_released_block(
    block: ReleasedBlock,
    *,
    dataset_id: str,
    source_artifact_sha256: str,
    protocol_sha256: str,
    partition: str,
    canonical_labels: dict[str, str],
    length_samples: int = 128,
    stride_samples: int = 128,
    conditional_hidden_joins: int = 2,
) -> BlockWindowResult:
    """Create windows without crossing a released block or overlapping samples."""

    block.validate()
    if not dataset_id or not source_artifact_sha256 or not protocol_sha256 or not partition:
        raise ValueError("dataset, source hash, protocol hash, and partition must be non-empty")
    if length_samples <= 0 or stride_samples <= 0:
        raise ValueError("window length and stride must be positive")
    if stride_samples != length_samples:
        raise ValueError("released-block protocol requires stride equal to window length")
    if conditional_hidden_joins < 0:
        raise ValueError("conditional hidden-join count cannot be negative")

    window_count = block.row_count // length_samples
    windows: list[WindowRecord] = []
    for ordinal in range(window_count):
        start = block.start_row_inclusive + (ordinal * stride_samples)
        end = start + length_samples - 1
        if end > block.end_row_inclusive:
            raise AssertionError("window generation crossed a released block")
        window_id = _stable_window_id(
            dataset_id=dataset_id,
            source_artifact_sha256=source_artifact_sha256,
            protocol_sha256=protocol_sha256,
            block=block,
            start_row_inclusive=start,
            end_row_inclusive=end,
            length_samples=length_samples,
            stride_samples=stride_samples,
            window_ordinal=ordinal,
        )
        windows.append(
            WindowRecord(
                window_id=window_id,
                raw_interval_id=(
                    f"{dataset_id}:{source_artifact_sha256}:rows:{start:09d}-{end:09d}"
                ),
                released_run_id=block.released_run_id,
                subject_id=block.subject_id,
                activity_label=block.activity_label,
                canonical_labels=canonical_labels,
                partition=partition,
                window_ordinal=ordinal,
                start_row_inclusive=start,
                end_row_inclusive=end,
                length_samples=length_samples,
                stride_samples=stride_samples,
                trial_id=None,
                trial_status="unrecoverable",
            )
        )
    used_rows = window_count * length_samples
    return BlockWindowResult(
        block=block,
        windows=tuple(windows),
        dropped_tail_rows=block.row_count - used_rows,
        used_rows=used_rows,
        conditional_hidden_join_crossing_window_bound=min(conditional_hidden_joins, window_count),
    )
