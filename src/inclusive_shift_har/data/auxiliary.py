"""Source-only materialization for explicitly allowlisted auxiliary sensor channels."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.data.windowing import WindowRecord
from inclusive_shift_har.manifests.canonical import sha256_file

INCLUSIVEHAR_GRAVITY_CHANNELS = (
    "motionGravityX",
    "motionGravityY",
    "motionGravityZ",
)


@dataclass(frozen=True)
class AuxiliaryWindows:
    """Auxiliary values aligned to source-window identifiers."""

    signals: NDArray[np.float32]
    window_ids: tuple[str, ...]
    participant_ids: tuple[str, ...]


def materialize_inclusivehar_source_gravity(
    csv_path: Path,
    records: tuple[WindowRecord, ...],
    *,
    expected_source_sha256: str,
    ontology_track: str,
    allowed_partitions: set[str],
) -> AuxiliaryWindows:
    """Read only gravity for source partitions already fixed by window records."""

    if any(item.casefold().startswith("target") for item in allowed_partitions):
        raise PermissionError("source auxiliary materializer refuses target partitions")
    if sha256_file(csv_path) != expected_source_sha256:
        raise ValueError("InclusiveHAR source artifact hash mismatch before materialization")
    selected = [
        record
        for record in records
        if record.partition in allowed_partitions and ontology_track in record.canonical_labels
    ]
    selected.sort(key=lambda item: item.start_row_inclusive)
    if not selected:
        raise ValueError("no source windows match the auxiliary materialization request")
    for previous, current in pairwise(selected):
        if current.start_row_inclusive <= previous.end_row_inclusive:
            raise ValueError("selected auxiliary raw intervals overlap or are duplicated")

    signals = np.empty((len(selected), 128, 3), dtype=np.float32)
    window_ids: list[str] = []
    participant_ids: list[str] = []
    next_window = 0
    active_values: list[list[float]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("InclusiveHAR CSV is empty") from exc
        header_index = {name: index for index, name in enumerate(header)}
        required = set(INCLUSIVEHAR_GRAVITY_CHANNELS) | {"label", "UserID"}
        missing = sorted(required - set(header_index))
        if missing:
            raise ValueError(f"InclusiveHAR CSV lacks gravity columns: {missing}")
        channel_indices = [header_index[name] for name in INCLUSIVEHAR_GRAVITY_CHANNELS]
        for data_row, row in enumerate(reader, start=1):
            if next_window >= len(selected):
                break
            active = selected[next_window]
            if data_row < active.start_row_inclusive:
                continue
            if data_row > active.end_row_inclusive:
                raise AssertionError("gravity materializer skipped an expected window row")
            if row[header_index["label"]].strip() != active.activity_label:
                raise ValueError(f"gravity window {active.window_id} mismatches its label")
            try:
                observed_subject = str(int(row[header_index["UserID"]].strip()))
            except ValueError as exc:
                raise ValueError(f"gravity window {active.window_id} has invalid subject") from exc
            if observed_subject != active.subject_id:
                raise ValueError(f"gravity window {active.window_id} mismatches its participant")
            try:
                active_values.append([float(row[index]) for index in channel_indices])
            except ValueError as exc:
                raise ValueError(f"gravity window {active.window_id} is nonnumeric") from exc
            if data_row == active.end_row_inclusive:
                values = np.asarray(active_values, dtype=np.float32)
                if values.shape != (128, 3) or not np.isfinite(values).all():
                    raise ValueError(
                        f"gravity window {active.window_id} violates finite [128,3] contract"
                    )
                signals[next_window] = values
                window_ids.append(active.window_id)
                participant_ids.append(active.subject_id)
                next_window += 1
                active_values = []
    if next_window != len(selected):
        raise ValueError(
            f"CSV ended after {next_window} of {len(selected)} authorized gravity windows"
        )
    return AuxiliaryWindows(
        signals=signals,
        window_ids=tuple(window_ids),
        participant_ids=tuple(participant_ids),
    )
