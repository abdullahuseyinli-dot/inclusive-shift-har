"""Read-only materialization of locked released-block windows into ignored caches."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.data.inclusivehar import INCLUSIVEHAR_PRIMARY_CHANNELS
from inclusive_shift_har.data.windowing import WindowRecord
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.protocols.seal import validate_target_unlock_record


@dataclass(frozen=True)
class MaterializedWindows:
    signals: NDArray[np.float32]
    labels: NDArray[np.int64]
    window_ids: tuple[str, ...]
    participant_ids: tuple[str, ...]
    released_labels: tuple[str, ...]
    partitions: tuple[str, ...]
    class_names: tuple[str, ...]
    ontology_track: str


def materialization_cache_key(
    *,
    source_artifact_sha256: str,
    preprocessing_config_sha256: str,
    split_manifest_sha256: str,
    code_version: str,
    ontology_track: str,
    partitions: Sequence[str],
) -> str:
    """Bind a cache to all raw/config/code inputs that can alter its values."""

    hashes = (source_artifact_sha256, preprocessing_config_sha256, split_manifest_sha256)
    if any(len(value) != 64 for value in hashes):
        raise ValueError("source, preprocessing, and split hashes must be full SHA-256 values")
    if not code_version or not ontology_track or not partitions:
        raise ValueError("cache key requires code version, ontology track, and partitions")
    return canonical_json_sha256(
        {
            "algorithm_version": "inclusivehar-materialization-v1",
            "code_version": code_version,
            "ontology_track": ontology_track,
            "partitions": sorted(set(partitions)),
            "preprocessing_config_sha256": preprocessing_config_sha256,
            "source_artifact_sha256": source_artifact_sha256,
            "split_manifest_sha256": split_manifest_sha256,
        }
    )


def _selected_records(
    records: Sequence[WindowRecord],
    *,
    ontology_track: str,
    allowed_partitions: set[str],
) -> list[WindowRecord]:
    selected: list[WindowRecord] = []
    for record in records:
        if record.partition not in allowed_partitions:
            continue
        if ontology_track in record.canonical_labels:
            selected.append(record)
    selected.sort(key=lambda item: item.start_row_inclusive)
    if not selected:
        raise ValueError("no windows match the authorized partitions and locked ontology track")
    for previous, current in pairwise(selected):
        if current.start_row_inclusive <= previous.end_row_inclusive:
            raise ValueError("selected raw intervals overlap or are duplicated")
    return selected


def materialize_inclusivehar_windows(
    csv_path: Path,
    records: Sequence[WindowRecord],
    *,
    expected_source_sha256: str,
    ontology_track: str,
    class_names: tuple[str, ...],
    allowed_partitions: set[str],
    target_unlock: dict[str, Any] | None = None,
    expected_target_seal_id: str | None = None,
    expected_split_manifest_sha256: str | None = None,
) -> MaterializedWindows:
    """Read only the six allowlisted channels for already-authorized split records.

    Target-named partitions additionally require a matching, approved one-time unlock record.
    Numeric labels come solely from ``class_names``.
    """

    if sha256_file(csv_path) != expected_source_sha256:
        raise ValueError("InclusiveHAR source artifact hash mismatch before materialization")
    if len(class_names) < 2 or len(set(class_names)) != len(class_names):
        raise ValueError("class_names must be a locked, unique, ordered schema")
    target_requested = any(
        partition.casefold().startswith("target") for partition in allowed_partitions
    )
    if target_requested:
        if expected_target_seal_id is None or len(expected_target_seal_id) != 64:
            raise PermissionError("target materialization requires the locked target seal id")
        if expected_split_manifest_sha256 is None:
            raise PermissionError("target materialization requires the locked split hash")
        validate_target_unlock_record(
            target_unlock,
            split_manifest_sha256=expected_split_manifest_sha256,
            target_seal_id=expected_target_seal_id,
        )
    selected = _selected_records(
        records,
        ontology_track=ontology_track,
        allowed_partitions=allowed_partitions,
    )
    class_to_index = {label: index for index, label in enumerate(class_names)}
    signals = np.empty((len(selected), 128, 6), dtype=np.float32)
    numeric_labels = np.empty(len(selected), dtype=np.int64)
    window_ids: list[str] = []
    participant_ids: list[str] = []
    released_labels: list[str] = []
    partitions: list[str] = []
    next_window = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("InclusiveHAR CSV is empty") from exc
        header_index = {name: index for index, name in enumerate(header)}
        required = set(INCLUSIVEHAR_PRIMARY_CHANNELS) | {"label", "UserID"}
        missing = sorted(required - set(header_index))
        if missing:
            raise ValueError(f"InclusiveHAR CSV lacks materialization columns: {missing}")
        channel_indices = [header_index[name] for name in INCLUSIVEHAR_PRIMARY_CHANNELS]
        active = selected[next_window]
        active_values: list[list[float]] = []
        for data_row, row in enumerate(reader, start=1):
            if next_window >= len(selected):
                break
            active = selected[next_window]
            if data_row < active.start_row_inclusive:
                continue
            if data_row > active.end_row_inclusive:
                raise AssertionError("materializer skipped an expected window row")
            if row[header_index["label"]].strip() != active.activity_label:
                raise ValueError(
                    f"window {active.window_id} crosses or mismatches an activity label"
                )
            try:
                observed_subject = str(int(row[header_index["UserID"]].strip()))
            except ValueError as exc:
                raise ValueError(f"window {active.window_id} has an invalid subject id") from exc
            if observed_subject != active.subject_id:
                raise ValueError(f"window {active.window_id} crosses or mismatches a participant")
            try:
                active_values.append([float(row[index]) for index in channel_indices])
            except ValueError as exc:
                raise ValueError(
                    f"window {active.window_id} contains a nonnumeric model input"
                ) from exc
            if data_row == active.end_row_inclusive:
                values = np.asarray(active_values, dtype=np.float32)
                if values.shape != (128, 6) or not np.isfinite(values).all():
                    raise ValueError(
                        f"window {active.window_id} violates the finite [128,6] contract"
                    )
                canonical_label = active.canonical_labels[ontology_track]
                try:
                    numeric_label = class_to_index[canonical_label]
                except KeyError as exc:
                    raise ValueError(
                        f"window {active.window_id} has canonical label outside locked class schema"
                    ) from exc
                signals[next_window] = values
                numeric_labels[next_window] = numeric_label
                window_ids.append(active.window_id)
                participant_ids.append(active.subject_id)
                released_labels.append(active.activity_label)
                partitions.append(active.partition)
                next_window += 1
                active_values = []
    if next_window != len(selected):
        raise ValueError(f"CSV ended after {next_window} of {len(selected)} authorized windows")
    return MaterializedWindows(
        signals=signals,
        labels=numeric_labels,
        window_ids=tuple(window_ids),
        participant_ids=tuple(participant_ids),
        released_labels=tuple(released_labels),
        partitions=tuple(partitions),
        class_names=class_names,
        ontology_track=ontology_track,
    )


def save_materialized_cache_create_only(
    batch: MaterializedWindows,
    cache_directory: Path,
    *,
    cache_key: str,
    lineage: dict[str, Any],
) -> dict[str, Any]:
    """Write ignored, create-only NPZ content and a canonical metadata sidecar."""

    if len(cache_key) != 64:
        raise ValueError("cache key must be a full SHA-256")
    cache_directory.mkdir(parents=True, exist_ok=True)
    array_path = cache_directory / f"{cache_key}.npz"
    metadata_path = cache_directory / f"{cache_key}.json"
    with array_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            signals=batch.signals,
            labels=batch.labels,
            window_ids=np.asarray(batch.window_ids),
            participant_ids=np.asarray(batch.participant_ids),
            released_labels=np.asarray(batch.released_labels),
            partitions=np.asarray(batch.partitions),
        )
    metadata = {
        "schema_version": "1.0.0",
        "cache_key": cache_key,
        "array_sha256": sha256_file(array_path),
        "window_count": batch.signals.shape[0],
        "shape": list(batch.signals.shape),
        "class_names": list(batch.class_names),
        "ontology_track": batch.ontology_track,
        "lineage": lineage,
    }
    with metadata_path.open("xb") as stream:
        stream.write(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        stream.write(b"\n")
    return {
        "cache_key": cache_key,
        "array_path": str(array_path),
        "array_sha256": metadata["array_sha256"],
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
    }
