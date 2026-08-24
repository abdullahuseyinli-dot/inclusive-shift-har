"""Create hash-pinned primary-channel caches after consumed target opening 1.

This command has no unlock-record or target-opening interface. It validates the
already-consumed receipt/index, reads the immutable CSV once, and publishes
separate conventional ``MaterializedWindows`` caches for source training,
source validation, and the sealed target partition.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from inclusive_shift_har.data.inclusivehar import INCLUSIVEHAR_PRIMARY_CHANNELS
from inclusive_shift_har.data.materialize import (
    MaterializedWindows,
    materialization_cache_key,
    save_materialized_cache_create_only,
)
from inclusive_shift_har.data.windowing import WindowRecord
from inclusive_shift_har.evaluation.postconfirmatory_inputs import (
    MaterializedCacheEvidence,
    load_consumed_target_context,
    load_materialized_cache_evidence,
)
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)

PRIMARY_CACHE_SCHEMA_VERSION = "1.0.0"
PRIMARY_CACHE_RECORD_KIND = "postconfirmatory_primary_channel_cache_index"
PRIMARY_CACHE_EVIDENCE_STATUS = "post_confirmatory_cache_consumed_opening_1"
FUNCTIONAL_CORE_CLASS_NAMES = ("mobility", "sitting", "standing")
EXPECTED_PARTITION_COUNTS = {
    "source_train": 582,
    "source_validation": 143,
    "target_sealed": 807,
}
EXPECTED_PARTICIPANTS = {
    "source_train": frozenset({"1", "2", "3", "4", "5", "6", "7", "9"}),
    "source_validation": frozenset({"8", "10"}),
    "target_sealed": frozenset(str(value) for value in range(11, 21)),
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CACHE_ARRAY_KEYS = {
    "signals",
    "labels",
    "window_ids",
    "participant_ids",
    "released_labels",
    "partitions",
}


class PostconfirmatoryCacheError(RuntimeError):
    """Raised when cache preparation or loading violates opening-1 lineage."""


@dataclass(frozen=True, slots=True)
class PreparedPrimaryCacheEvidence:
    record_path: Path
    record_file_sha256: str
    record_sha256: str
    record: Mapping[str, Any]
    cache: MaterializedCacheEvidence


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PostconfirmatoryCacheError(f"{name} must be an object")
    return value


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PostconfirmatoryCacheError(f"{name} must be a lowercase SHA-256")
    return value


def _self_hash(record: Mapping[str, Any], *, field: str, name: str) -> str:
    body = dict(record)
    claimed = _sha256(body.pop(field, None), name=f"{name} {field}")
    if claimed != canonical_json_sha256(body):
        raise PostconfirmatoryCacheError(f"{name} self-hash does not validate")
    return claimed


def _existing_file(value: str | Path, *, root: Path, name: str) -> Path:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    if candidate.is_symlink():
        raise PostconfirmatoryCacheError(f"{name} may not be a symlink")
    path = candidate.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PostconfirmatoryCacheError(f"{name} escapes artifact_root") from exc
    if path.is_symlink() or not path.is_file():
        raise PostconfirmatoryCacheError(f"{name} must be a regular file")
    return path


def _new_path(value: str | Path, *, root: Path, name: str) -> Path:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    path = candidate.resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PostconfirmatoryCacheError(f"{name} escapes its allowed root") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = path.parent.resolve(strict=True)
    if parent != path.parent or parent.is_symlink():
        raise PostconfirmatoryCacheError(f"{name} parent is unsafe")
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite {name}: {path}")
    return path


def _load_split(
    split_manifest_path: str | Path, *, artifact_root: Path
) -> tuple[Mapping[str, Any], Path, str, str]:
    path = _existing_file(split_manifest_path, root=artifact_root, name="split manifest")
    split = _mapping(load_json_strict(path), name="split manifest")
    split_hash = _self_hash(split, field="split_manifest_sha256", name="split manifest")
    required = {
        "dataset_id": "inclusivehar_v4",
        "subject_assignment_before_windowing": True,
        "target_performance_or_prediction_accessed": False,
        "window_length_samples": 128,
        "window_stride_samples": 128,
    }
    mismatches = [key for key, expected in required.items() if split.get(key) != expected]
    protocol = _mapping(split.get("protocol"), name="split protocol")
    if mismatches or protocol.get("protocol_id") != "inclusivehar-released-block-v1.2":
        raise PostconfirmatoryCacheError(f"split contract differs: {mismatches}")
    return split, path, split_hash, sha256_file(path)


def _validate_ontology(split: Mapping[str, Any]) -> tuple[str, str]:
    ontology = _mapping(split.get("ontology"), name="split ontology")
    schemas = _mapping(ontology.get("runnable_track_schemas"), name="ontology schemas")
    schema = _mapping(schemas.get("functional_core"), name="functional-core schema")
    body = dict(schema)
    schema_hash = _sha256(body.pop("class_schema_sha256", None), name="functional-core schema hash")
    required = {
        "track": "functional_core",
        "class_count": 3,
        "class_order": list(FUNCTIONAL_CORE_CLASS_NAMES),
        "index_by_class": {name: index for index, name in enumerate(FUNCTIONAL_CORE_CLASS_NAMES)},
    }
    if any(body.get(key) != expected for key, expected in required.items()) or schema_hash != (
        canonical_json_sha256(body)
    ):
        raise PostconfirmatoryCacheError("functional-core ontology/class order changed")
    return _sha256(ontology.get("config_sha256"), name="ontology config hash"), schema_hash


def _source_hash(split: Mapping[str, Any]) -> str:
    source = _mapping(split.get("source_evidence"), name="split source evidence")
    return _sha256(source.get("sensor_artifact_sha256"), name="raw sensor artifact hash")


def _preprocessing_lineage(
    split: Mapping[str, Any], *, artifact_root: Path
) -> tuple[Mapping[str, Any], Path, str]:
    preprocessing = _mapping(split.get("preprocessing"), name="split preprocessing")
    if preprocessing.get("preprocessing_id") != "inclusivehar-primary-six-128-v1":
        raise PostconfirmatoryCacheError("split does not use the primary six-channel transform")
    config_path_value = preprocessing.get("config_path")
    if not isinstance(config_path_value, str):
        raise PostconfirmatoryCacheError("preprocessing config path is invalid")
    config_path = _existing_file(config_path_value, root=artifact_root, name="preprocessing config")
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = _mapping(parsed, name="preprocessing config")
    config_hash = _sha256(preprocessing.get("config_sha256"), name="preprocessing config hash")
    if canonical_json_sha256(config) != config_hash:
        raise PostconfirmatoryCacheError("preprocessing config bytes/content changed")
    policy = _mapping(split.get("model_input_policy"), name="model input policy")
    if policy.get("allowed_channels_exact_order") != list(INCLUSIVEHAR_PRIMARY_CHANNELS):
        raise PostconfirmatoryCacheError("primary channel order changed")
    return config, config_path, config_hash


def _partition_records(split: Mapping[str, Any], partition: str) -> tuple[WindowRecord, ...]:
    values = split.get("windows")
    if not isinstance(values, list):
        raise PostconfirmatoryCacheError("split lacks window records")
    records: list[WindowRecord] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise PostconfirmatoryCacheError("split window record is not an object")
        labels = value.get("canonical_labels")
        if value.get("partition") != partition or not isinstance(labels, Mapping):
            continue
        if "functional_core" not in labels:
            continue
        try:
            record = WindowRecord(**dict(value))
        except TypeError as exc:
            raise PostconfirmatoryCacheError(f"{partition} window schema changed") from exc
        if (
            record.length_samples != 128
            or record.stride_samples != 128
            or record.canonical_labels["functional_core"] not in FUNCTIONAL_CORE_CLASS_NAMES
        ):
            raise PostconfirmatoryCacheError(f"{partition} functional-core window is invalid")
        records.append(record)
    records.sort(key=lambda value: value.start_row_inclusive)
    expected_count = EXPECTED_PARTITION_COUNTS[partition]
    if len(records) != expected_count:
        raise PostconfirmatoryCacheError(
            f"{partition} requires exactly {expected_count} functional-core windows"
        )
    ids = [record.window_id for record in records]
    if len(set(ids)) != expected_count:
        raise PostconfirmatoryCacheError(f"{partition} window identities are duplicated")
    participants = {record.subject_id for record in records}
    if participants != EXPECTED_PARTICIPANTS[partition]:
        raise PostconfirmatoryCacheError(f"{partition} participant assignment changed")
    for previous, current in pairwise(records):
        if current.start_row_inclusive <= previous.end_row_inclusive:
            raise PostconfirmatoryCacheError(f"{partition} raw window intervals overlap")
    return tuple(records)


def _validate_functional_partition_universe(split: Mapping[str, Any]) -> None:
    values = split.get("windows")
    if not isinstance(values, list):
        raise PostconfirmatoryCacheError("split lacks window records")
    observed = 0
    for value in values:
        if not isinstance(value, Mapping):
            raise PostconfirmatoryCacheError("split window record is not an object")
        labels = value.get("canonical_labels")
        if isinstance(labels, Mapping) and "functional_core" in labels:
            observed += 1
            if value.get("partition") not in EXPECTED_PARTITION_COUNTS:
                raise PostconfirmatoryCacheError(
                    "functional-core window has an unauthorized partition"
                )
    if observed != sum(EXPECTED_PARTITION_COUNTS.values()):
        raise PostconfirmatoryCacheError("functional-core partition universe count changed")


def _materialize_once(
    csv_path: Path,
    records: Sequence[WindowRecord],
) -> MaterializedWindows:
    ordered = sorted(records, key=lambda value: value.start_row_inclusive)
    if not ordered:
        raise PostconfirmatoryCacheError("no authorized records were selected")
    for previous, current in pairwise(ordered):
        if current.start_row_inclusive <= previous.end_row_inclusive:
            raise PostconfirmatoryCacheError("combined cache windows overlap")
    class_to_index = {label: index for index, label in enumerate(FUNCTIONAL_CORE_CLASS_NAMES)}
    signals = np.empty((len(ordered), 128, 6), dtype=np.float32)
    labels = np.empty(len(ordered), dtype=np.int64)
    window_ids: list[str] = []
    participant_ids: list[str] = []
    released_labels: list[str] = []
    partitions: list[str] = []
    next_window = 0
    active_values: list[list[float]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise PostconfirmatoryCacheError("InclusiveHAR CSV is empty") from exc
        header_index = {name: index for index, name in enumerate(header)}
        required = set(INCLUSIVEHAR_PRIMARY_CHANNELS) | {"label", "UserID"}
        missing = sorted(required - set(header_index))
        if missing:
            raise PostconfirmatoryCacheError(f"InclusiveHAR CSV lacks columns: {missing}")
        channel_indices = [header_index[name] for name in INCLUSIVEHAR_PRIMARY_CHANNELS]
        for row_index, row in enumerate(reader, start=1):
            if next_window >= len(ordered):
                break
            record = ordered[next_window]
            if row_index < record.start_row_inclusive:
                continue
            if row_index > record.end_row_inclusive:
                raise PostconfirmatoryCacheError("materializer skipped an authorized window")
            try:
                observed_subject = str(int(row[header_index["UserID"]].strip()))
                observed_label = row[header_index["label"]].strip()
                active_values.append([float(row[index]) for index in channel_indices])
            except (IndexError, ValueError) as exc:
                raise PostconfirmatoryCacheError(
                    "authorized window contains malformed primary-channel data"
                ) from exc
            if observed_subject != record.subject_id or observed_label != record.activity_label:
                raise PostconfirmatoryCacheError(
                    "authorized window crosses a participant/activity boundary"
                )
            if row_index == record.end_row_inclusive:
                values = np.asarray(active_values, dtype=np.float32)
                if values.shape != (128, 6) or not np.isfinite(values).all():
                    raise PostconfirmatoryCacheError("window violates finite [128,6]")
                signals[next_window] = values
                labels[next_window] = class_to_index[record.canonical_labels["functional_core"]]
                window_ids.append(record.window_id)
                participant_ids.append(record.subject_id)
                released_labels.append(record.activity_label)
                partitions.append(record.partition)
                active_values = []
                next_window += 1
    if next_window != len(ordered):
        raise PostconfirmatoryCacheError(
            f"CSV ended after {next_window} of {len(ordered)} authorized windows"
        )
    return MaterializedWindows(
        signals=signals,
        labels=labels,
        window_ids=tuple(window_ids),
        participant_ids=tuple(participant_ids),
        released_labels=tuple(released_labels),
        partitions=tuple(partitions),
        class_names=FUNCTIONAL_CORE_CLASS_NAMES,
        ontology_track="functional_core",
    )


def _select_partition(batch: MaterializedWindows, partition: str) -> MaterializedWindows:
    indices = np.asarray(
        [index for index, value in enumerate(batch.partitions) if value == partition],
        dtype=np.int64,
    )
    if indices.size != EXPECTED_PARTITION_COUNTS[partition]:
        raise PostconfirmatoryCacheError(f"materialized {partition} count changed")
    positions = indices.tolist()
    return MaterializedWindows(
        signals=np.asarray(batch.signals[indices], dtype=np.float32),
        labels=np.asarray(batch.labels[indices], dtype=np.int64),
        window_ids=tuple(batch.window_ids[index] for index in positions),
        participant_ids=tuple(batch.participant_ids[index] for index in positions),
        released_labels=tuple(batch.released_labels[index] for index in positions),
        partitions=tuple(batch.partitions[index] for index in positions),
        class_names=batch.class_names,
        ontology_track=batch.ontology_track,
    )


def _select_target_participants(
    batch: MaterializedWindows, participants: frozenset[str]
) -> MaterializedWindows:
    if not participants or not participants <= EXPECTED_PARTICIPANTS["target_sealed"]:
        raise PostconfirmatoryCacheError("target shard participant selection is invalid")
    indices = np.asarray(
        [
            index
            for index, (partition, participant) in enumerate(
                zip(batch.partitions, batch.participant_ids, strict=True)
            )
            if partition == "target_sealed" and participant in participants
        ],
        dtype=np.int64,
    )
    if indices.size < 1:
        raise PostconfirmatoryCacheError("target participant shard is empty")
    positions = indices.tolist()
    selected = MaterializedWindows(
        signals=np.asarray(batch.signals[indices], dtype=np.float32),
        labels=np.asarray(batch.labels[indices], dtype=np.int64),
        window_ids=tuple(batch.window_ids[index] for index in positions),
        participant_ids=tuple(batch.participant_ids[index] for index in positions),
        released_labels=tuple(batch.released_labels[index] for index in positions),
        partitions=tuple(batch.partitions[index] for index in positions),
        class_names=batch.class_names,
        ontology_track=batch.ontology_track,
    )
    if set(selected.participant_ids) != set(participants):
        raise PostconfirmatoryCacheError("target participant shard coverage differs")
    return selected


def _cache_reference(
    cache: Mapping[str, Any],
    *,
    artifact_root: Path,
    partition: str,
    expected_count: int | None = None,
    participant_id: str | None = None,
) -> dict[str, Any]:
    array_path = Path(str(cache["array_path"])).resolve(strict=True)
    metadata_path = Path(str(cache["metadata_path"])).resolve(strict=True)
    metadata = _mapping(load_json_strict(metadata_path), name=f"{partition} cache metadata")
    lineage = _mapping(metadata.get("lineage"), name=f"{partition} cache lineage")
    count = EXPECTED_PARTITION_COUNTS[partition] if expected_count is None else expected_count
    reference = {
        "partition": partition,
        "window_count": count,
        "window_ids_sha256": lineage["window_ids_sha256"],
        "ordered_window_ids_sha256": lineage["ordered_window_ids_sha256"],
        "ordered_participant_ids_sha256": lineage["ordered_participant_ids_sha256"],
        "ordered_labels_sha256": lineage["ordered_labels_sha256"],
        "array_path": array_path.relative_to(artifact_root).as_posix(),
        "array_sha256": cache["array_sha256"],
        "metadata_path": metadata_path.relative_to(artifact_root).as_posix(),
        "metadata_sha256": cache["metadata_sha256"],
    }
    if participant_id is not None:
        reference["participant_id"] = participant_id
    return reference


def _load_source_train_cache_evidence(
    *,
    array_path: str | Path,
    metadata_path: str | Path,
    expected_array_sha256: str,
    expected_metadata_sha256: str,
    artifact_root: Path,
    expected_split_manifest_sha256: str,
) -> MaterializedCacheEvidence:
    """Validate the conventional cache form for the source-training partition."""

    array_file = _existing_file(array_path, root=artifact_root, name="source-train cache")
    metadata_file = _existing_file(
        metadata_path, root=artifact_root, name="source-train cache metadata"
    )
    array_hash = _sha256(expected_array_sha256, name="source-train array hash")
    metadata_hash = _sha256(expected_metadata_sha256, name="source-train metadata hash")
    if sha256_file(array_file) != array_hash or sha256_file(metadata_file) != metadata_hash:
        raise PostconfirmatoryCacheError("source-train cache file hash changed")
    metadata = _mapping(load_json_strict(metadata_file), name="source-train cache metadata")
    lineage = _mapping(metadata.get("lineage"), name="source-train cache lineage")
    if (
        metadata.get("array_sha256") != array_hash
        or metadata.get("class_names") != list(FUNCTIONAL_CORE_CLASS_NAMES)
        or metadata.get("ontology_track") != "functional_core"
        or metadata.get("window_count") != EXPECTED_PARTITION_COUNTS["source_train"]
        or lineage.get("split_manifest_sha256") != expected_split_manifest_sha256
        or lineage.get("partition") != "source_train"
    ):
        raise PostconfirmatoryCacheError("source-train cache metadata changed")
    with np.load(array_file, allow_pickle=False) as arrays:
        if set(arrays.files) != _CACHE_ARRAY_KEYS:
            raise PostconfirmatoryCacheError("source-train cache keys changed")
        raw_signals = np.asarray(arrays["signals"])
        raw_labels = np.asarray(arrays["labels"])
        if raw_signals.dtype != np.dtype(np.float32) or raw_labels.dtype != np.dtype(np.int64):
            raise PostconfirmatoryCacheError("source-train cache dtype changed")
        signals = np.asarray(raw_signals, dtype=np.float32)
        labels = np.asarray(raw_labels, dtype=np.int64)
        window_ids = tuple(str(value) for value in arrays["window_ids"].tolist())
        participant_ids = tuple(str(value) for value in arrays["participant_ids"].tolist())
        released_labels = tuple(str(value) for value in arrays["released_labels"].tolist())
        partitions = tuple(str(value) for value in arrays["partitions"].tolist())
    count = EXPECTED_PARTITION_COUNTS["source_train"]
    if (
        signals.shape != (count, 128, 6)
        or labels.shape != (count,)
        or metadata.get("shape") != [count, 128, 6]
        or not np.isfinite(signals).all()
        or len(window_ids) != count
        or len(set(window_ids)) != count
        or len(participant_ids) != count
        or len(released_labels) != count
        or len(partitions) != count
        or set(participant_ids) != EXPECTED_PARTICIPANTS["source_train"]
        or set(partitions) != {"source_train"}
        or labels.min() < 0
        or labels.max() >= len(FUNCTIONAL_CORE_CLASS_NAMES)
    ):
        raise PostconfirmatoryCacheError("source-train cache alignment changed")
    batch = MaterializedWindows(
        signals=signals,
        labels=labels,
        window_ids=window_ids,
        participant_ids=participant_ids,
        released_labels=released_labels,
        partitions=partitions,
        class_names=FUNCTIONAL_CORE_CLASS_NAMES,
        ontology_track="functional_core",
    )
    return MaterializedCacheEvidence(
        batch=batch,
        array_path=array_file,
        array_sha256=array_hash,
        metadata_path=metadata_file,
        metadata_sha256=metadata_hash,
        metadata=metadata,
    )


def prepare_postconfirmatory_primary_caches(
    *,
    split_manifest_path: str | Path,
    opening_receipt_path: str | Path,
    locked_target_index_path: str | Path,
    raw_csv_path: str | Path,
    artifact_root: str | Path,
    cache_directory: str | Path,
    cache_root: str | Path,
    record_output: str | Path,
    record_root: str | Path,
    code_commit: str,
    created_at_utc: str,
) -> dict[str, Any]:
    """Materialize both authorized partitions once and publish immutable cache evidence."""

    commit = code_commit.casefold()
    if _COMMIT_RE.fullmatch(commit) is None:
        raise PostconfirmatoryCacheError("code_commit must be a full 40-character Git ID")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", created_at_utc) is None:
        raise PostconfirmatoryCacheError("created_at_utc must use second-precision UTC Z format")
    root = Path(artifact_root).resolve(strict=True)
    cache_allowed = Path(cache_root).resolve(strict=True)
    record_allowed = Path(record_root).resolve(strict=True)
    for allowed, name in ((cache_allowed, "cache_root"), (record_allowed, "record_root")):
        try:
            allowed.relative_to(root)
        except ValueError as exc:
            raise PostconfirmatoryCacheError(f"{name} escapes artifact_root") from exc
    cache_path = _new_path(cache_directory, root=cache_allowed, name="cache directory")
    record_path = _new_path(record_output, root=record_allowed, name="cache record")
    split, split_path, split_hash, split_file_hash = _load_split(
        split_manifest_path, artifact_root=root
    )
    context = load_consumed_target_context(
        opening_receipt_path=opening_receipt_path,
        locked_target_index_path=locked_target_index_path,
        artifact_root=root,
    )
    if (
        context.receipt.get("split_manifest_sha256") != split_hash
        or context.index.get("split_manifest_sha256") != split_hash
        or context.index.get("class_names") != list(FUNCTIONAL_CORE_CLASS_NAMES)
        or context.index.get("window_count") != EXPECTED_PARTITION_COUNTS["target_sealed"]
        or context.index.get("participant_count") != len(EXPECTED_PARTICIPANTS["target_sealed"])
    ):
        raise PostconfirmatoryCacheError("consumed opening-1 evidence differs from split/schema")
    ontology_hash, ontology_schema_hash = _validate_ontology(split)
    preprocessing, preprocessing_path, preprocessing_hash = _preprocessing_lineage(
        split, artifact_root=root
    )
    source_hash = _source_hash(split)
    raw_path = _existing_file(raw_csv_path, root=root, name="raw sensor CSV")
    _validate_functional_partition_universe(split)
    source_train_records = _partition_records(split, "source_train")
    source_records = _partition_records(split, "source_validation")
    target_records = _partition_records(split, "target_sealed")
    partition_id_sets = [
        {record.window_id for record in records}
        for records in (source_train_records, source_records, target_records)
    ]
    if any(
        left & right
        for index, left in enumerate(partition_id_sets)
        for right in partition_id_sets[index + 1 :]
    ):
        raise PostconfirmatoryCacheError("cache partition window identities overlap")
    if sha256_file(raw_path) != source_hash:
        raise PostconfirmatoryCacheError("raw sensor CSV hash differs from split lineage")
    combined = _materialize_once(
        raw_path, (*source_train_records, *source_records, *target_records)
    )
    batches = {
        "source_train": _select_partition(combined, "source_train"),
        "source_validation": _select_partition(combined, "source_validation"),
        "target_sealed": _select_partition(combined, "target_sealed"),
    }
    target_shard_batches = {
        participant: _select_target_participants(combined, frozenset({participant}))
        for participant in sorted(EXPECTED_PARTICIPANTS["target_sealed"], key=int)
    }
    shard_id_sets = [set(batch.window_ids) for batch in target_shard_batches.values()]
    if any(
        left & right
        for index, left in enumerate(shard_id_sets)
        for right in shard_id_sets[index + 1 :]
    ):
        raise PostconfirmatoryCacheError("target participant shard window identities overlap")
    shard_union = set().union(*shard_id_sets)
    target_ids = set(batches["target_sealed"].window_ids)
    if shard_union != target_ids or sum(len(values) for values in shard_id_sets) != len(target_ids):
        raise PostconfirmatoryCacheError("target participant shards are not an exact target union")

    cache_path.mkdir(exist_ok=False)
    cache_records: dict[str, Mapping[str, Any]] = {}
    target_shard_records: dict[str, Mapping[str, Any]] = {}
    common_lineage = {
        "split_manifest_sha256": split_hash,
        "split_manifest_file_sha256": split_file_hash,
        "source_artifact_sha256": source_hash,
        "preprocessing_config_sha256": preprocessing_hash,
        "ontology_config_sha256": ontology_hash,
        "ontology_class_schema_sha256": ontology_schema_hash,
        "opening_receipt_record_sha256": context.receipt["record_sha256"],
        "opening_receipt_file_sha256": context.receipt_file_sha256,
        "locked_target_index_record_sha256": context.index["record_sha256"],
        "locked_target_index_file_sha256": context.index_file_sha256,
        "target_seal_id": context.index["target_seal_id"],
        "code_commit": commit,
    }
    try:
        for partition, batch in batches.items():
            records = {
                "source_train": source_train_records,
                "source_validation": source_records,
                "target_sealed": target_records,
            }[partition]
            lineage = {
                **common_lineage,
                "partition": partition,
                "window_count": len(records),
                "window_ids_sha256": canonical_json_sha256(
                    sorted(record.window_id for record in records)
                ),
                "ordered_window_ids_sha256": canonical_json_sha256(list(batch.window_ids)),
                "ordered_participant_ids_sha256": canonical_json_sha256(
                    list(batch.participant_ids)
                ),
                "ordered_labels_sha256": canonical_json_sha256(batch.labels.tolist()),
            }
            key = materialization_cache_key(
                source_artifact_sha256=source_hash,
                preprocessing_config_sha256=preprocessing_hash,
                split_manifest_sha256=split_hash,
                code_version=commit,
                ontology_track="functional_core",
                partitions=(partition,),
            )
            saved = save_materialized_cache_create_only(
                batch, cache_path, cache_key=key, lineage=lineage
            )
            cache_records[partition] = _cache_reference(
                saved, artifact_root=root, partition=partition
            )
        target_base_key = materialization_cache_key(
            source_artifact_sha256=source_hash,
            preprocessing_config_sha256=preprocessing_hash,
            split_manifest_sha256=split_hash,
            code_version=commit,
            ontology_track="functional_core",
            partitions=("target_sealed",),
        )
        for participant, batch in target_shard_batches.items():
            records = tuple(record for record in target_records if record.subject_id == participant)
            if len(records) != len(batch.window_ids):
                raise PostconfirmatoryCacheError(
                    f"target participant {participant} shard count differs from split"
                )
            lineage = {
                **common_lineage,
                "partition": "target_sealed",
                "cache_role": "few_person_target_participant_shard",
                "participant_id": participant,
                "window_count": len(records),
                "window_ids_sha256": canonical_json_sha256(
                    sorted(record.window_id for record in records)
                ),
                "ordered_window_ids_sha256": canonical_json_sha256(list(batch.window_ids)),
                "ordered_participant_ids_sha256": canonical_json_sha256(
                    list(batch.participant_ids)
                ),
                "ordered_labels_sha256": canonical_json_sha256(batch.labels.tolist()),
            }
            shard_key = canonical_json_sha256(
                {
                    "cache_kind": "few_person_target_participant_shard_v1",
                    "base_cache_key": target_base_key,
                    "participant_id": participant,
                }
            )
            saved = save_materialized_cache_create_only(
                batch, cache_path, cache_key=shard_key, lineage=lineage
            )
            target_shard_records[participant] = _cache_reference(
                saved,
                artifact_root=root,
                partition="target_sealed",
                expected_count=len(records),
                participant_id=participant,
            )
        source_train_ref = cache_records["source_train"]
        source_ref = cache_records["source_validation"]
        target_ref = cache_records["target_sealed"]
        _load_source_train_cache_evidence(
            array_path=cast(str, source_train_ref["array_path"]),
            metadata_path=cast(str, source_train_ref["metadata_path"]),
            expected_array_sha256=cast(str, source_train_ref["array_sha256"]),
            expected_metadata_sha256=cast(str, source_train_ref["metadata_sha256"]),
            artifact_root=root,
            expected_split_manifest_sha256=split_hash,
        )
        load_materialized_cache_evidence(
            array_path=source_ref["array_path"],
            metadata_path=source_ref["metadata_path"],
            expected_array_sha256=str(source_ref["array_sha256"]),
            expected_metadata_sha256=str(source_ref["metadata_sha256"]),
            artifact_root=root,
            expected_partition="source_validation",
            expected_split_manifest_sha256=split_hash,
            expected_class_names=FUNCTIONAL_CORE_CLASS_NAMES,
        )
        load_materialized_cache_evidence(
            array_path=target_ref["array_path"],
            metadata_path=target_ref["metadata_path"],
            expected_array_sha256=str(target_ref["array_sha256"]),
            expected_metadata_sha256=str(target_ref["metadata_sha256"]),
            artifact_root=root,
            expected_partition="target_sealed",
            expected_split_manifest_sha256=split_hash,
            expected_class_names=FUNCTIONAL_CORE_CLASS_NAMES,
            target_context=context,
        )
        validated_shard_ids: set[str] = set()
        for participant, shard in target_shard_records.items():
            evidence = load_materialized_cache_evidence(
                array_path=shard["array_path"],
                metadata_path=shard["metadata_path"],
                expected_array_sha256=str(shard["array_sha256"]),
                expected_metadata_sha256=str(shard["metadata_sha256"]),
                artifact_root=root,
                expected_partition="target_sealed",
                expected_split_manifest_sha256=split_hash,
                expected_class_names=FUNCTIONAL_CORE_CLASS_NAMES,
                target_context=context,
                expected_target_participants=frozenset({participant}),
            )
            shard_ids = set(evidence.batch.window_ids)
            if validated_shard_ids & shard_ids:
                raise PostconfirmatoryCacheError("validated target participant shards overlap")
            validated_shard_ids.update(shard_ids)
        if validated_shard_ids != target_ids:
            raise PostconfirmatoryCacheError(
                "validated target participant shards do not cover the full target cache"
            )
        record: dict[str, Any] = {
            "schema_version": PRIMARY_CACHE_SCHEMA_VERSION,
            "record_kind": PRIMARY_CACHE_RECORD_KIND,
            "status": "complete_create_only",
            "evidence_status": PRIMARY_CACHE_EVIDENCE_STATUS,
            "created_at_utc": created_at_utc,
            "code_commit": commit,
            "split_manifest": {
                "path": split_path.relative_to(root).as_posix(),
                "record_sha256": split_hash,
                "file_sha256": split_file_hash,
            },
            "raw_sensor_csv": {
                "path": raw_path.relative_to(root).as_posix(),
                "sha256": source_hash,
            },
            "preprocessing": {
                "path": preprocessing_path.relative_to(root).as_posix(),
                "config_sha256": preprocessing_hash,
                "file_sha256": sha256_file(preprocessing_path),
                "config": dict(preprocessing),
            },
            "ontology_track": "functional_core",
            "class_names": list(FUNCTIONAL_CORE_CLASS_NAMES),
            "primary_channels": list(INCLUSIVEHAR_PRIMARY_CHANNELS),
            "opening_1": {
                "receipt_record_sha256": context.receipt["record_sha256"],
                "receipt_file_sha256": context.receipt_file_sha256,
                "index_record_sha256": context.index["record_sha256"],
                "index_file_sha256": context.index_file_sha256,
                "target_seal_id": context.index["target_seal_id"],
            },
            "caches": dict(cache_records),
            "target_participant_shards": dict(target_shard_records),
            "target_participant_shard_audit": {
                "status": "exact_disjoint_union_validated_against_opening_1",
                "participant_ids": sorted(target_shard_records, key=int),
                "shard_count": len(target_shard_records),
                "total_window_count": len(validated_shard_ids),
                "union_window_ids_sha256": canonical_json_sha256(sorted(validated_shard_ids)),
                "whole_target_window_ids_sha256": target_ref["window_ids_sha256"],
                "pairwise_window_id_disjoint": True,
                "exact_whole_target_union": True,
            },
            "target_cache_ordered_alignment_validated_against_opening_1": True,
            "unlock_api_called": False,
            "new_target_opening_created": False,
            "gpu_execution_performed": False,
        }
        record["record_sha256"] = canonical_json_sha256(record)
        atomic_write_json_new(record, record_path, allowed_root=record_allowed)
        return record
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": PRIMARY_CACHE_SCHEMA_VERSION,
            "record_kind": "postconfirmatory_primary_channel_cache_failure",
            "status": "failed_preserved_create_only",
            "evidence_status": "failed_cache_preparation_not_evidence",
            "split_manifest_sha256": split_hash,
            "opening_receipt_record_sha256": context.receipt["record_sha256"],
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "partial_cache_directory": cache_path.relative_to(root).as_posix(),
            "unlock_api_called": False,
            "new_target_opening_created": False,
        }
        failure["record_sha256"] = canonical_json_sha256(failure)
        atomic_write_json_new(failure, cache_path / "failure.json", allowed_root=cache_allowed)
        raise PostconfirmatoryCacheError(
            f"cache preparation failed; evidence preserved at {cache_path / 'failure.json'}"
        ) from exc


def _validate_target_participant_shard_index(
    record: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    value = _mapping(record.get("target_participant_shards"), name="target participant shard index")
    expected = EXPECTED_PARTICIPANTS["target_sealed"]
    if set(value) != set(expected):
        raise PostconfirmatoryCacheError("target participant shard subject set changed")
    shards: dict[str, Mapping[str, Any]] = {}
    total = 0
    array_paths: set[str] = set()
    metadata_paths: set[str] = set()
    for participant in sorted(expected, key=int):
        shard = _mapping(value.get(participant), name=f"target shard {participant}")
        count = shard.get("window_count")
        if (
            shard.get("participant_id") != participant
            or shard.get("partition") != "target_sealed"
            or not isinstance(count, int)
            or count < 1
        ):
            raise PostconfirmatoryCacheError(
                f"target participant shard {participant} contract changed"
            )
        for field in (
            "window_ids_sha256",
            "ordered_window_ids_sha256",
            "ordered_participant_ids_sha256",
            "ordered_labels_sha256",
            "array_sha256",
            "metadata_sha256",
        ):
            _sha256(shard.get(field), name=f"target shard {participant} {field}")
        array_path = shard.get("array_path")
        metadata_path = shard.get("metadata_path")
        if (
            not isinstance(array_path, str)
            or not array_path
            or not isinstance(metadata_path, str)
            or not metadata_path
            or array_path in array_paths
            or metadata_path in metadata_paths
        ):
            raise PostconfirmatoryCacheError("target shard paths are invalid or reused")
        array_paths.add(array_path)
        metadata_paths.add(metadata_path)
        total += count
        shards[participant] = shard
    whole = _mapping(
        _mapping(record.get("caches"), name="primary cache references").get("target_sealed"),
        name="whole target cache reference",
    )
    audit = _mapping(
        record.get("target_participant_shard_audit"), name="target participant shard audit"
    )
    required_audit = {
        "status": "exact_disjoint_union_validated_against_opening_1",
        "participant_ids": sorted(expected, key=int),
        "shard_count": len(expected),
        "total_window_count": EXPECTED_PARTITION_COUNTS["target_sealed"],
        "union_window_ids_sha256": whole.get("window_ids_sha256"),
        "whole_target_window_ids_sha256": whole.get("window_ids_sha256"),
        "pairwise_window_id_disjoint": True,
        "exact_whole_target_union": True,
    }
    if total != EXPECTED_PARTITION_COUNTS["target_sealed"] or any(
        audit.get(key) != expected_value for key, expected_value in required_audit.items()
    ):
        raise PostconfirmatoryCacheError("target participant shard union audit changed")
    return shards


def load_prepared_primary_cache(
    *,
    record_path: str | Path,
    expected_record_file_sha256: str,
    partition: str,
    opening_receipt_path: str | Path,
    locked_target_index_path: str | Path,
    artifact_root: str | Path,
    expected_split_manifest_sha256: str,
    expected_source_artifact_sha256: str,
) -> PreparedPrimaryCacheEvidence:
    """Load one cache only when its self-hashed index file is externally pinned."""

    if partition not in EXPECTED_PARTITION_COUNTS:
        raise PostconfirmatoryCacheError("unsupported primary cache partition")
    root = Path(artifact_root).resolve(strict=True)
    path = _existing_file(record_path, root=root, name="primary cache record")
    expected_file_hash = _sha256(
        expected_record_file_sha256, name="expected primary cache record file hash"
    )
    if sha256_file(path) != expected_file_hash:
        raise PostconfirmatoryCacheError("primary cache record file hash changed")
    record = _mapping(load_json_strict(path), name="primary cache record")
    record_hash = _self_hash(record, field="record_sha256", name="primary cache record")
    required = {
        "schema_version": PRIMARY_CACHE_SCHEMA_VERSION,
        "record_kind": PRIMARY_CACHE_RECORD_KIND,
        "status": "complete_create_only",
        "evidence_status": PRIMARY_CACHE_EVIDENCE_STATUS,
        "ontology_track": "functional_core",
        "class_names": list(FUNCTIONAL_CORE_CLASS_NAMES),
        "primary_channels": list(INCLUSIVEHAR_PRIMARY_CHANNELS),
        "target_cache_ordered_alignment_validated_against_opening_1": True,
        "unlock_api_called": False,
        "new_target_opening_created": False,
    }
    mismatches = [key for key, expected in required.items() if record.get(key) != expected]
    split = _mapping(record.get("split_manifest"), name="cache split lineage")
    raw = _mapping(record.get("raw_sensor_csv"), name="cache raw lineage")
    if (
        mismatches
        or split.get("record_sha256") != expected_split_manifest_sha256
        or raw.get("sha256") != expected_source_artifact_sha256
    ):
        raise PostconfirmatoryCacheError(f"primary cache record lineage differs: {mismatches}")
    _validate_target_participant_shard_index(record)
    context = load_consumed_target_context(
        opening_receipt_path=opening_receipt_path,
        locked_target_index_path=locked_target_index_path,
        artifact_root=root,
    )
    opening = _mapping(record.get("opening_1"), name="cache opening-1 lineage")
    if (
        opening.get("receipt_record_sha256") != context.receipt["record_sha256"]
        or opening.get("receipt_file_sha256") != context.receipt_file_sha256
        or opening.get("index_record_sha256") != context.index["record_sha256"]
        or opening.get("index_file_sha256") != context.index_file_sha256
        or opening.get("target_seal_id") != context.index["target_seal_id"]
    ):
        raise PostconfirmatoryCacheError("primary cache record differs from consumed opening 1")
    caches = _mapping(record.get("caches"), name="primary cache references")
    cache = _mapping(caches.get(partition), name=f"{partition} cache reference")
    if (
        cache.get("partition") != partition
        or cache.get("window_count") != EXPECTED_PARTITION_COUNTS[partition]
    ):
        raise PostconfirmatoryCacheError(f"{partition} cache reference changed")
    if partition == "source_train":
        evidence = _load_source_train_cache_evidence(
            array_path=cast(str, cache.get("array_path")),
            metadata_path=cast(str, cache.get("metadata_path")),
            expected_array_sha256=cast(str, cache.get("array_sha256")),
            expected_metadata_sha256=cast(str, cache.get("metadata_sha256")),
            artifact_root=root,
            expected_split_manifest_sha256=expected_split_manifest_sha256,
        )
    else:
        evidence = load_materialized_cache_evidence(
            array_path=cast(str, cache.get("array_path")),
            metadata_path=cast(str, cache.get("metadata_path")),
            expected_array_sha256=cast(str, cache.get("array_sha256")),
            expected_metadata_sha256=cast(str, cache.get("metadata_sha256")),
            artifact_root=root,
            expected_partition=partition,
            expected_split_manifest_sha256=expected_split_manifest_sha256,
            expected_class_names=FUNCTIONAL_CORE_CLASS_NAMES,
            target_context=context if partition == "target_sealed" else None,
        )
    lineage = _mapping(evidence.metadata.get("lineage"), name=f"{partition} cache lineage")
    for field in (
        "window_ids_sha256",
        "ordered_window_ids_sha256",
        "ordered_participant_ids_sha256",
        "ordered_labels_sha256",
    ):
        if cache.get(field) != lineage.get(field):
            raise PostconfirmatoryCacheError(
                f"{partition} cache identity hash differs from metadata"
            )
    return PreparedPrimaryCacheEvidence(
        record_path=path,
        record_file_sha256=expected_file_hash,
        record_sha256=record_hash,
        record=record,
        cache=evidence,
    )


def load_prepared_primary_participant_shard(
    *,
    record_path: str | Path,
    expected_record_file_sha256: str,
    participant_id: str,
    opening_receipt_path: str | Path,
    locked_target_index_path: str | Path,
    artifact_root: str | Path,
    expected_split_manifest_sha256: str,
    expected_source_artifact_sha256: str,
) -> PreparedPrimaryCacheEvidence:
    """Load exactly one target participant shard from consumed-opening evidence."""

    if participant_id not in EXPECTED_PARTICIPANTS["target_sealed"]:
        raise PostconfirmatoryCacheError("unsupported target shard participant")
    root = Path(artifact_root).resolve(strict=True)
    path = _existing_file(record_path, root=root, name="primary cache record")
    expected_file_hash = _sha256(
        expected_record_file_sha256, name="expected primary cache record file hash"
    )
    if sha256_file(path) != expected_file_hash:
        raise PostconfirmatoryCacheError("primary cache record file hash changed")
    record = _mapping(load_json_strict(path), name="primary cache record")
    record_hash = _self_hash(record, field="record_sha256", name="primary cache record")
    required = {
        "schema_version": PRIMARY_CACHE_SCHEMA_VERSION,
        "record_kind": PRIMARY_CACHE_RECORD_KIND,
        "status": "complete_create_only",
        "evidence_status": PRIMARY_CACHE_EVIDENCE_STATUS,
        "ontology_track": "functional_core",
        "class_names": list(FUNCTIONAL_CORE_CLASS_NAMES),
        "primary_channels": list(INCLUSIVEHAR_PRIMARY_CHANNELS),
        "target_cache_ordered_alignment_validated_against_opening_1": True,
        "unlock_api_called": False,
        "new_target_opening_created": False,
    }
    mismatches = [key for key, expected in required.items() if record.get(key) != expected]
    split = _mapping(record.get("split_manifest"), name="cache split lineage")
    raw = _mapping(record.get("raw_sensor_csv"), name="cache raw lineage")
    if (
        mismatches
        or split.get("record_sha256") != expected_split_manifest_sha256
        or raw.get("sha256") != expected_source_artifact_sha256
    ):
        raise PostconfirmatoryCacheError(f"primary cache record lineage differs: {mismatches}")
    split_path = _existing_file(
        cast(str, split.get("path")), root=root, name="cache split manifest"
    )
    split_record = _mapping(load_json_strict(split_path), name="cache split manifest")
    if (
        sha256_file(split_path) != split.get("file_sha256")
        or _self_hash(split_record, field="split_manifest_sha256", name="cache split manifest")
        != expected_split_manifest_sha256
    ):
        raise PostconfirmatoryCacheError("cache split-manifest file lineage changed")
    shards = _validate_target_participant_shard_index(record)
    shard = shards[participant_id]
    context = load_consumed_target_context(
        opening_receipt_path=opening_receipt_path,
        locked_target_index_path=locked_target_index_path,
        artifact_root=root,
    )
    opening = _mapping(record.get("opening_1"), name="cache opening-1 lineage")
    if (
        opening.get("receipt_record_sha256") != context.receipt["record_sha256"]
        or opening.get("receipt_file_sha256") != context.receipt_file_sha256
        or opening.get("index_record_sha256") != context.index["record_sha256"]
        or opening.get("index_file_sha256") != context.index_file_sha256
        or opening.get("target_seal_id") != context.index["target_seal_id"]
    ):
        raise PostconfirmatoryCacheError("primary cache record differs from consumed opening 1")
    evidence = load_materialized_cache_evidence(
        array_path=cast(str, shard.get("array_path")),
        metadata_path=cast(str, shard.get("metadata_path")),
        expected_array_sha256=cast(str, shard.get("array_sha256")),
        expected_metadata_sha256=cast(str, shard.get("metadata_sha256")),
        artifact_root=root,
        expected_partition="target_sealed",
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        expected_class_names=FUNCTIONAL_CORE_CLASS_NAMES,
        target_context=context,
        expected_target_participants=frozenset({participant_id}),
    )
    lineage = _mapping(evidence.metadata.get("lineage"), name="target shard lineage")
    if (
        lineage.get("cache_role") != "few_person_target_participant_shard"
        or lineage.get("participant_id") != participant_id
        or lineage.get("window_count") != shard.get("window_count")
        or set(evidence.batch.participant_ids) != {participant_id}
    ):
        raise PostconfirmatoryCacheError("target participant shard lineage changed")
    for field in (
        "window_ids_sha256",
        "ordered_window_ids_sha256",
        "ordered_participant_ids_sha256",
        "ordered_labels_sha256",
    ):
        if shard.get(field) != lineage.get(field):
            raise PostconfirmatoryCacheError(
                f"target participant shard {field} differs from metadata"
            )
    return PreparedPrimaryCacheEvidence(
        record_path=path,
        record_file_sha256=expected_file_hash,
        record_sha256=record_hash,
        record=record,
        cache=evidence,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--opening-receipt", type=Path, required=True)
    parser.add_argument("--locked-target-index", type=Path, required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--record-output", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--created-at-utc", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        record = prepare_postconfirmatory_primary_caches(
            split_manifest_path=args.split_manifest,
            opening_receipt_path=args.opening_receipt,
            locked_target_index_path=args.locked_target_index,
            raw_csv_path=args.raw_csv,
            artifact_root=args.artifact_root,
            cache_directory=args.cache_directory,
            cache_root=args.cache_root,
            record_output=args.record_output,
            record_root=args.record_root,
            code_commit=args.code_commit,
            created_at_utc=args.created_at_utc,
        )
        record_file = _existing_file(
            args.record_output,
            root=args.record_root.resolve(strict=True),
            name="cache record",
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "fail", "message": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": record["status"],
                "record_sha256": record["record_sha256"],
                "record_file_sha256": sha256_file(record_file),
                "caches": record["caches"],
                "unlock_api_called": False,
                "new_target_opening_created": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
