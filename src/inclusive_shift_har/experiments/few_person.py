"""One-scenario CUDA runner for the post-confirmatory few-person curve."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from inclusive_shift_har.data.inclusivehar import INCLUSIVEHAR_PRIMARY_CHANNELS
from inclusive_shift_har.data.materialize import MaterializedWindows
from inclusive_shift_har.data.windowing import WindowRecord
from inclusive_shift_har.evaluation._strict_config import require_utc_timestamp
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.evaluation.source_calibration import (
    load_source_temperature_calibrator_file,
)
from inclusive_shift_har.experiments.postconfirmatory_cache import (
    FUNCTIONAL_CORE_CLASS_NAMES,
    PRIMARY_CACHE_EVIDENCE_STATUS,
    PRIMARY_CACHE_RECORD_KIND,
    PRIMARY_CACHE_SCHEMA_VERSION,
    PreparedPrimaryCacheEvidence,
)
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)
from inclusive_shift_har.models.common import HAROutput, trainable_parameter_count
from inclusive_shift_har.preprocessing.normalization import ChannelStandardizer
from inclusive_shift_har.protocols.few_person import (
    FEW_PERSON_PROTOCOL_ID,
    FewPersonProtocolError,
)
from inclusive_shift_har.protocols.few_person_v1_1 import (
    FEW_PERSON_V1_1_EVIDENCE_STATUS,
    FEW_PERSON_V1_1_PROTOCOL_ID,
    build_few_person_v1_1_manifest,
    validate_few_person_v1_1_manifest_assignments,
    write_few_person_v1_1_manifest_new,
)
from inclusive_shift_har.training.engine import (
    TrainingConfig,
    configure_determinism,
    predict_model,
    reconstruct_checkpoint,
    training_config_from_dict,
)


class FewPersonRunError(RuntimeError):
    """Raised when a few-person run violates its frozen boundary."""


@dataclass(frozen=True, slots=True)
class _PrimaryCacheMetadataPreflight:
    record_path: Path
    record_file_sha256: str
    record_sha256: str
    participant_shards: Mapping[str, Mapping[str, Any]]


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FewPersonRunError(f"{name} must be an object")
    return value


def _json_object(path: str | Path, *, name: str) -> dict[str, Any]:
    return dict(_mapping(load_json_strict(path), name=name))


def _self_hash(record: Mapping[str, Any], *, field: str, name: str) -> str:
    claimed = record.get(field)
    body = dict(record)
    body.pop(field, None)
    if not isinstance(claimed, str) or claimed != canonical_json_sha256(body):
        raise FewPersonRunError(f"{name} self-hash does not validate")
    return claimed


def _full_commit(value: str) -> str:
    normalized = value.casefold()
    if len(normalized) != 40 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise FewPersonRunError("code commit must be a full 40-character Git object ID")
    return normalized


def _repository_head(repository_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FewPersonRunError("artifact root is not a readable Git checkout") from exc
    return _full_commit(completed.stdout.strip())


def _require_repository_head(repository_root: Path, supplied_commit: str) -> str:
    commit = _full_commit(supplied_commit)
    observed = _repository_head(repository_root)
    if observed != commit:
        raise FewPersonRunError(
            f"supplied code commit {commit} differs from repository HEAD {observed}"
        )
    return commit


def _execution_environment(
    device: torch.device, *, cudnn_enabled_during_run: bool
) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "torch_cuda_runtime": torch.version.cuda,
        "device_type": device.type,
        "device_index": torch.cuda.current_device() if device.index is None else device.index,
        "device_name": properties.name,
        "device_total_memory_bytes": int(properties.total_memory),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "cudnn_enabled_during_run": cudnn_enabled_during_run,
        "cudnn_version": torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
        "hostname_recorded": False,
        "process_id_recorded": False,
    }


def _confined(path: Path, *, root: Path, name: str, must_exist: bool) -> Path:
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=must_exist)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise FewPersonRunError(f"{name} escapes its declared root") from exc
    return resolved


def _sha256_digest(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FewPersonRunError(f"{name} must be a lowercase SHA-256")
    return value


def _preflight_primary_cache_metadata(
    *,
    record_path: Path,
    expected_record_file_sha256: str,
    artifact_root: Path,
    split_manifest_path: Path,
    split_manifest_sha256: str,
    split_manifest_file_sha256: str,
    source_artifact_sha256: str,
    plan: Mapping[str, Any],
) -> _PrimaryCacheMetadataPreflight:
    """Validate target-cache lineage and bytes without loading any signal array."""

    root = artifact_root.resolve(strict=True)
    path = _confined(record_path, root=root, name="primary cache record", must_exist=True)
    pinned_file_hash = _sha256_digest(
        expected_record_file_sha256, name="primary cache record file hash"
    )
    if sha256_file(path) != pinned_file_hash:
        raise FewPersonRunError("primary cache record file hash changed")
    record = _json_object(path, name="primary cache record")
    record_sha256 = _self_hash(record, field="record_sha256", name="primary cache record")
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
    if mismatches:
        raise FewPersonRunError(f"primary cache record contract differs: {mismatches}")

    split_reference = _mapping(record.get("split_manifest"), name="cache split lineage")
    referenced_split = _confined(
        root / str(split_reference.get("path")),
        root=root,
        name="cache-referenced split manifest",
        must_exist=True,
    )
    supplied_split = split_manifest_path.resolve(strict=True)
    if (
        referenced_split != supplied_split
        or split_reference.get("record_sha256") != split_manifest_sha256
        or split_reference.get("file_sha256") != split_manifest_file_sha256
        or sha256_file(referenced_split) != split_manifest_file_sha256
    ):
        raise FewPersonRunError("primary cache does not reference the exact parent split file")

    raw_reference = _mapping(record.get("raw_sensor_csv"), name="cache raw lineage")
    if raw_reference.get("sha256") != source_artifact_sha256:
        raise FewPersonRunError("primary cache raw-source lineage changed")

    opening = _mapping(record.get("opening_1"), name="cache opening-1 lineage")
    opening_required = {
        "receipt_record_sha256": plan["opening_receipt_record_sha256"],
        "receipt_file_sha256": plan["opening_receipt_file_sha256"],
        "index_record_sha256": plan["zero_shot_index_record_sha256"],
        "index_file_sha256": plan["zero_shot_index_file_sha256"],
        "target_seal_id": plan["target_seal_id"],
    }
    if any(opening.get(key) != value for key, value in opening_required.items()):
        raise FewPersonRunError("primary cache differs from consumed opening 1")

    caches = _mapping(record.get("caches"), name="primary cache references")
    target = _mapping(caches.get("target_sealed"), name="target cache reference")
    target_count = plan.get("functional_core_target_window_count")
    target_ids_hash = plan.get("functional_core_target_window_ids_sha256")
    if (
        not isinstance(target_count, int)
        or target_count <= 0
        or target.get("partition") != "target_sealed"
        or target.get("window_count") != target_count
        or target.get("window_ids_sha256") != target_ids_hash
    ):
        raise FewPersonRunError("target cache coverage differs from the v1.1 manifest")
    for field in ("window_ids_sha256",):
        _sha256_digest(target.get(field), name=f"target cache {field}")
    shard_values = _mapping(
        record.get("target_participant_shards"), name="target participant shard index"
    )
    target_subjects = {str(value) for value in range(11, 21)}
    if set(shard_values) != target_subjects:
        raise FewPersonRunError("target participant shard subject set changed")
    split_record = _json_object(referenced_split, name="cache-referenced split manifest")
    windows = split_record.get("windows")
    if not isinstance(windows, list):
        raise FewPersonRunError("cache-referenced split lacks windows")
    expected_ids_by_subject = {
        participant: sorted(
            str(value["window_id"])
            for value in windows
            if isinstance(value, Mapping)
            and value.get("partition") == "target_sealed"
            and value.get("subject_id") == participant
            and isinstance(value.get("canonical_labels"), Mapping)
            and "functional_core" in cast(Mapping[str, Any], value["canonical_labels"])
        )
        for participant in sorted(target_subjects, key=int)
    }
    participant_shards: dict[str, Mapping[str, Any]] = {}
    all_ids: set[str] = set()
    total = 0
    array_paths: set[Path] = set()
    metadata_paths: set[Path] = set()
    for participant, expected_ids in expected_ids_by_subject.items():
        shard = _mapping(shard_values.get(participant), name=f"target shard {participant}")
        array_path = _confined(
            root / str(shard.get("array_path")),
            root=root,
            name=f"target shard {participant} array",
            must_exist=True,
        )
        metadata_path = _confined(
            root / str(shard.get("metadata_path")),
            root=root,
            name=f"target shard {participant} metadata",
            must_exist=True,
        )
        if array_path in array_paths or metadata_path in metadata_paths:
            raise FewPersonRunError("target participant shard paths are reused")
        array_paths.add(array_path)
        metadata_paths.add(metadata_path)
        expected_hash = canonical_json_sha256(expected_ids)
        if (
            not expected_ids
            or shard.get("participant_id") != participant
            or shard.get("partition") != "target_sealed"
            or shard.get("window_count") != len(expected_ids)
            or shard.get("window_ids_sha256") != expected_hash
        ):
            raise FewPersonRunError(f"target shard {participant} differs from the split")
        for field in (
            "window_ids_sha256",
            "ordered_window_ids_sha256",
            "ordered_participant_ids_sha256",
            "ordered_labels_sha256",
            "array_sha256",
            "metadata_sha256",
        ):
            _sha256_digest(shard.get(field), name=f"target shard {participant} {field}")
        if sha256_file(metadata_path) != shard.get("metadata_sha256"):
            raise FewPersonRunError(f"target shard {participant} metadata hash changed")
        metadata = _json_object(metadata_path, name=f"target shard {participant} metadata")
        if (
            metadata.get("schema_version") != "1.0.0"
            or metadata.get("array_sha256") != shard.get("array_sha256")
            or metadata.get("window_count") != len(expected_ids)
            or metadata.get("shape") != [len(expected_ids), 128, 6]
            or metadata.get("class_names") != list(FUNCTIONAL_CORE_CLASS_NAMES)
            or metadata.get("ontology_track") != "functional_core"
        ):
            raise FewPersonRunError(f"target shard {participant} metadata changed")
        lineage = _mapping(metadata.get("lineage"), name=f"target shard {participant} lineage")
        lineage_required = {
            "split_manifest_sha256": split_manifest_sha256,
            "source_artifact_sha256": source_artifact_sha256,
            "opening_receipt_record_sha256": plan["opening_receipt_record_sha256"],
            "opening_receipt_file_sha256": plan["opening_receipt_file_sha256"],
            "locked_target_index_record_sha256": plan["zero_shot_index_record_sha256"],
            "locked_target_index_file_sha256": plan["zero_shot_index_file_sha256"],
            "target_seal_id": plan["target_seal_id"],
            "partition": "target_sealed",
            "cache_role": "few_person_target_participant_shard",
            "participant_id": participant,
            "window_count": len(expected_ids),
            "window_ids_sha256": expected_hash,
        }
        if any(lineage.get(key) != value for key, value in lineage_required.items()):
            raise FewPersonRunError(f"target shard {participant} lineage changed")
        for field in (
            "window_ids_sha256",
            "ordered_window_ids_sha256",
            "ordered_participant_ids_sha256",
            "ordered_labels_sha256",
        ):
            if shard.get(field) != lineage.get(field):
                raise FewPersonRunError(f"target shard {participant} {field} differs from metadata")
        participant_shards[participant] = {
            "participant_id": participant,
            "window_count": len(expected_ids),
            "window_ids_sha256": expected_hash,
            "array_path": array_path.relative_to(root).as_posix(),
            "array_sha256": shard["array_sha256"],
            "metadata_path": metadata_path.relative_to(root).as_posix(),
            "metadata_sha256": shard["metadata_sha256"],
        }
        overlap = all_ids & set(expected_ids)
        if overlap:
            raise FewPersonRunError("target participant shard split identities overlap")
        all_ids.update(expected_ids)
        total += len(expected_ids)
    audit = _mapping(
        record.get("target_participant_shard_audit"), name="target participant shard audit"
    )
    if (
        total != target_count
        or canonical_json_sha256(sorted(all_ids)) != target_ids_hash
        or audit.get("status") != "exact_disjoint_union_validated_against_opening_1"
        or audit.get("participant_ids") != sorted(target_subjects, key=int)
        or audit.get("shard_count") != len(target_subjects)
        or audit.get("total_window_count") != target_count
        or audit.get("union_window_ids_sha256") != target_ids_hash
        or audit.get("whole_target_window_ids_sha256") != target_ids_hash
        or audit.get("pairwise_window_id_disjoint") is not True
        or audit.get("exact_whole_target_union") is not True
    ):
        raise FewPersonRunError("target participant shard union audit changed")
    return _PrimaryCacheMetadataPreflight(
        record_path=path,
        record_file_sha256=pinned_file_hash,
        record_sha256=record_sha256,
        participant_shards=participant_shards,
    )


def _raw_input_lineage(path: Path, *, root: Path, source_sha256: str) -> dict[str, Any]:
    return {
        "mode": "immutable_raw_csv_authorized_windows_only",
        "source_path": path.relative_to(root).as_posix(),
        "source_artifact_sha256": source_sha256,
    }


def _selected(
    plan: Mapping[str, Any], *, fold_id: str, k: int, model_id: str, seed: int
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    scenarios = plan.get("scenarios")
    models = plan.get("models")
    if not isinstance(scenarios, list) or not isinstance(models, list):
        raise FewPersonRunError("few-person manifest lacks scenarios or models")
    scenario = next(
        (
            _mapping(value, name="few-person scenario")
            for value in scenarios
            if isinstance(value, Mapping)
            and value.get("fold_id") == fold_id
            and value.get("k") == k
        ),
        None,
    )
    model = next(
        (
            _mapping(value, name="few-person model")
            for value in models
            if isinstance(value, Mapping)
            and value.get("model_id") == model_id
            and value.get("seed") == seed
        ),
        None,
    )
    if scenario is None:
        raise FewPersonRunError(f"scenario {fold_id} k={k} is not predeclared")
    if model is None:
        raise FewPersonRunError(f"model/seed {model_id}/{seed} is not predeclared")
    inclusion = set(cast(list[str], scenario["target_inclusion_subjects"]))
    evaluation = set(cast(list[str], scenario["evaluation_subjects"]))
    unused = set(cast(list[str], scenario["unused_target_subjects"]))
    if inclusion & evaluation or inclusion & unused or evaluation & unused:
        raise FewPersonRunError("target scenario participant sets overlap")
    if len(inclusion) != k or inclusion | evaluation | unused != {
        str(value) for value in range(11, 21)
    }:
        raise FewPersonRunError("target scenario does not partition the target cohort")
    for forbidden in (
        "validation_subjects",
        "calibration_subjects",
        "threshold_selection_subjects",
    ):
        if scenario.get(forbidden) != []:
            raise FewPersonRunError(f"evaluation isolation changed: {forbidden}")
    return scenario, model


def _window_records(
    split: Mapping[str, Any],
    *,
    subjects: set[str],
    ontology_track: str,
    expected_count: int,
    expected_ids_sha256: str,
) -> tuple[WindowRecord, ...]:
    windows = split.get("windows")
    if not isinstance(windows, list):
        raise FewPersonRunError("parent split lacks window records")
    records = tuple(
        WindowRecord(**dict(value))
        for value in windows
        if isinstance(value, Mapping)
        and value.get("partition") == "target_sealed"
        and value.get("subject_id") in subjects
        and ontology_track in cast(Mapping[str, str], value.get("canonical_labels", {}))
    )
    records = tuple(sorted(records, key=lambda item: item.start_row_inclusive))
    ids = sorted(record.window_id for record in records)
    if len(records) != expected_count or canonical_json_sha256(ids) != expected_ids_sha256:
        raise FewPersonRunError("scenario window count/hash differs from the parent split")
    for previous, current in pairwise(records):
        if current.start_row_inclusive <= previous.end_row_inclusive:
            raise FewPersonRunError("selected target windows overlap")
    return records


def _materialize_authorized_records(
    csv_path: Path,
    records: Sequence[WindowRecord],
    *,
    expected_source_sha256: str,
    class_names: tuple[str, ...],
    ontology_track: str,
) -> MaterializedWindows:
    """Read only records authorized by the post-confirmatory scenario manifest."""

    if sha256_file(csv_path) != expected_source_sha256:
        raise FewPersonRunError("InclusiveHAR source artifact hash mismatch")
    if not records:
        raise FewPersonRunError("authorized target record set is empty")
    class_to_index = {name: index for index, name in enumerate(class_names)}
    signals = np.empty((len(records), 128, 6), dtype=np.float32)
    labels = np.empty(len(records), dtype=np.int64)
    window_ids: list[str] = []
    participant_ids: list[str] = []
    released_labels: list[str] = []
    next_window = 0
    active_values: list[list[float]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise FewPersonRunError("InclusiveHAR CSV is empty") from exc
        header_index = {name: index for index, name in enumerate(header)}
        required = set(INCLUSIVEHAR_PRIMARY_CHANNELS) | {"label", "UserID"}
        missing = sorted(required - set(header_index))
        if missing:
            raise FewPersonRunError(f"InclusiveHAR CSV lacks columns: {missing}")
        channel_indices = [header_index[name] for name in INCLUSIVEHAR_PRIMARY_CHANNELS]
        for row_index, row in enumerate(reader, start=1):
            if next_window >= len(records):
                break
            active = records[next_window]
            if row_index < active.start_row_inclusive:
                continue
            if row_index > active.end_row_inclusive:
                raise FewPersonRunError("materializer skipped an authorized target window")
            if row[header_index["label"]].strip() != active.activity_label:
                raise FewPersonRunError("target window crosses an activity boundary")
            try:
                observed_subject = str(int(row[header_index["UserID"]].strip()))
                active_values.append([float(row[index]) for index in channel_indices])
            except ValueError as exc:
                raise FewPersonRunError("target window contains invalid model input") from exc
            if observed_subject != active.subject_id:
                raise FewPersonRunError("target window crosses a participant boundary")
            if row_index == active.end_row_inclusive:
                values = np.asarray(active_values, dtype=np.float32)
                if values.shape != (128, 6) or not np.isfinite(values).all():
                    raise FewPersonRunError("target window violates finite [128,6]")
                canonical = active.canonical_labels[ontology_track]
                if canonical not in class_to_index:
                    raise FewPersonRunError("target label lies outside the frozen schema")
                signals[next_window] = values
                labels[next_window] = class_to_index[canonical]
                window_ids.append(active.window_id)
                participant_ids.append(active.subject_id)
                released_labels.append(active.activity_label)
                next_window += 1
                active_values = []
    if next_window != len(records):
        raise FewPersonRunError("CSV ended before all authorized windows were read")
    return MaterializedWindows(
        signals=signals,
        labels=labels,
        window_ids=tuple(window_ids),
        participant_ids=tuple(participant_ids),
        released_labels=tuple(released_labels),
        partitions=("target_postconfirmatory",) * len(records),
        class_names=class_names,
        ontology_track=ontology_track,
    )


def _select_cached_records(
    evidence: PreparedPrimaryCacheEvidence,
    records: Sequence[WindowRecord],
    *,
    class_names: tuple[str, ...],
) -> MaterializedWindows:
    batch = evidence.cache.batch
    if batch.class_names != class_names or batch.ontology_track != "functional_core":
        raise FewPersonRunError("target cache class/ontology schema changed")
    index_by_id = {window_id: index for index, window_id in enumerate(batch.window_ids)}
    if len(index_by_id) != len(batch.window_ids):
        raise FewPersonRunError("target cache window identities are duplicated")
    ordered = tuple(sorted(records, key=lambda value: value.start_row_inclusive))
    try:
        indices = np.asarray([index_by_id[record.window_id] for record in ordered], dtype=np.int64)
    except KeyError as exc:
        raise FewPersonRunError("target cache lacks a predeclared scenario window") from exc
    class_to_index = {name: index for index, name in enumerate(class_names)}
    for record, index in zip(ordered, indices.tolist(), strict=True):
        if (
            batch.participant_ids[index] != record.subject_id
            or batch.released_labels[index] != record.activity_label
            or batch.partitions[index] != "target_sealed"
            or int(batch.labels[index])
            != class_to_index[record.canonical_labels["functional_core"]]
        ):
            raise FewPersonRunError("target cache row metadata differs from the split")
    return MaterializedWindows(
        signals=np.asarray(batch.signals[indices], dtype=np.float32),
        labels=np.asarray(batch.labels[indices], dtype=np.int64),
        window_ids=tuple(batch.window_ids[index] for index in indices.tolist()),
        participant_ids=tuple(batch.participant_ids[index] for index in indices.tolist()),
        released_labels=tuple(batch.released_labels[index] for index in indices.tolist()),
        partitions=tuple(batch.partitions[index] for index in indices.tolist()),
        class_names=batch.class_names,
        ontology_track=batch.ontology_track,
    )


def _shard_input_lineage(
    preflight: _PrimaryCacheMetadataPreflight,
    participants: set[str],
    records: Sequence[WindowRecord],
    *,
    root: Path,
) -> dict[str, Any]:
    selected = {
        participant: dict(preflight.participant_shards[participant])
        for participant in sorted(participants, key=int)
    }
    expected_count = sum(int(value["window_count"]) for value in selected.values())
    if (
        set(record.subject_id for record in records) != participants
        or len(records) != expected_count
    ):
        raise FewPersonRunError("scenario records differ from participant-shard lineage")
    return {
        "mode": "hash_pinned_target_participant_shards",
        "record_path": preflight.record_path.relative_to(root).as_posix(),
        "record_file_sha256": preflight.record_file_sha256,
        "record_sha256": preflight.record_sha256,
        "participant_ids": sorted(participants, key=int),
        "participant_shards": selected,
        "selected_window_count": expected_count,
        "selected_window_ids_sha256": canonical_json_sha256(
            sorted(record.window_id for record in records)
        ),
    }


def _select_across_cached_shards(
    batches: Sequence[MaterializedWindows],
    records: Sequence[WindowRecord],
    *,
    class_names: tuple[str, ...],
) -> MaterializedWindows:
    rows: dict[str, tuple[NDArray[np.float32], int, str, str, str]] = {}
    for batch in batches:
        if batch.class_names != class_names or batch.ontology_track != "functional_core":
            raise FewPersonRunError("target shard class/ontology schema changed")
        for index, window_id in enumerate(batch.window_ids):
            if window_id in rows:
                raise FewPersonRunError("loaded target participant shards overlap")
            rows[window_id] = (
                np.asarray(batch.signals[index], dtype=np.float32),
                int(batch.labels[index]),
                batch.participant_ids[index],
                batch.released_labels[index],
                batch.partitions[index],
            )
    ordered = tuple(sorted(records, key=lambda value: value.start_row_inclusive))
    if set(rows) != {record.window_id for record in ordered}:
        raise FewPersonRunError("loaded participant shards differ from scenario window identities")
    class_to_index = {name: index for index, name in enumerate(class_names)}
    signals: list[NDArray[np.float32]] = []
    labels: list[int] = []
    participants: list[str] = []
    released: list[str] = []
    partitions: list[str] = []
    for record in ordered:
        signal, label, participant, released_label, partition = rows[record.window_id]
        if (
            participant != record.subject_id
            or released_label != record.activity_label
            or partition != "target_sealed"
            or label != class_to_index[record.canonical_labels["functional_core"]]
        ):
            raise FewPersonRunError("target shard row metadata differs from split")
        signals.append(signal)
        labels.append(label)
        participants.append(participant)
        released.append(released_label)
        partitions.append(partition)
    return MaterializedWindows(
        signals=np.asarray(signals, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        window_ids=tuple(record.window_id for record in ordered),
        participant_ids=tuple(participants),
        released_labels=tuple(released),
        partitions=tuple(partitions),
        class_names=class_names,
        ontology_track="functional_core",
    )


def _load_preflight_shard_array(
    reference: Mapping[str, Any],
    *,
    artifact_root: Path,
    class_names: tuple[str, ...],
) -> MaterializedWindows:
    root = artifact_root.resolve(strict=True)
    participant = str(reference["participant_id"])
    count = int(reference["window_count"])
    array_path = _confined(
        root / str(reference["array_path"]),
        root=root,
        name=f"target shard {participant} array",
        must_exist=True,
    )
    if sha256_file(array_path) != reference.get("array_sha256"):
        raise FewPersonRunError(f"target shard {participant} array hash changed")
    with np.load(array_path, allow_pickle=False) as arrays:
        required = {
            "signals",
            "labels",
            "window_ids",
            "participant_ids",
            "released_labels",
            "partitions",
        }
        if set(arrays.files) != required:
            raise FewPersonRunError(f"target shard {participant} array keys changed")
        raw_signals = np.asarray(arrays["signals"])
        raw_labels = np.asarray(arrays["labels"])
        if raw_signals.dtype != np.dtype(np.float32) or raw_labels.dtype != np.dtype(np.int64):
            raise FewPersonRunError(f"target shard {participant} dtypes changed")
        signals = np.asarray(raw_signals, dtype=np.float32)
        labels = np.asarray(raw_labels, dtype=np.int64)
        window_ids = tuple(str(value) for value in arrays["window_ids"].tolist())
        participant_ids = tuple(str(value) for value in arrays["participant_ids"].tolist())
        released_labels = tuple(str(value) for value in arrays["released_labels"].tolist())
        partitions = tuple(str(value) for value in arrays["partitions"].tolist())
    if (
        signals.shape != (count, 128, 6)
        or labels.shape != (count,)
        or not np.isfinite(signals).all()
        or len(window_ids) != count
        or len(set(window_ids)) != count
        or len(participant_ids) != count
        or len(released_labels) != count
        or len(partitions) != count
        or set(participant_ids) != {participant}
        or set(partitions) != {"target_sealed"}
        or labels.min() < 0
        or labels.max() >= len(class_names)
        or canonical_json_sha256(sorted(window_ids)) != reference["window_ids_sha256"]
    ):
        raise FewPersonRunError(f"target shard {participant} array alignment changed")
    return MaterializedWindows(
        signals=signals,
        labels=labels,
        window_ids=window_ids,
        participant_ids=participant_ids,
        released_labels=released_labels,
        partitions=partitions,
        class_names=class_names,
        ontology_track="functional_core",
    )


def _load_cached_participant_shards(
    *,
    preflight: _PrimaryCacheMetadataPreflight,
    participants: set[str],
    records: Sequence[WindowRecord],
    class_names: tuple[str, ...],
    artifact_root: Path,
    adapted_checkpoint_path: Path | None = None,
    expected_adapted_checkpoint_sha256: str | None = None,
) -> MaterializedWindows:
    """Load only declared subject shards, optionally behind a checkpoint barrier."""

    checkpoint_required = adapted_checkpoint_path is not None
    if checkpoint_required != (expected_adapted_checkpoint_sha256 is not None):
        raise FewPersonRunError("adapted checkpoint path/hash must be supplied together")
    if checkpoint_required and (
        adapted_checkpoint_path is None
        or not adapted_checkpoint_path.is_file()
        or sha256_file(adapted_checkpoint_path) != expected_adapted_checkpoint_sha256
    ):
        raise FewPersonRunError("held-out cache access requires the exact fixed adapted checkpoint")
    if set(record.subject_id for record in records) != participants:
        raise FewPersonRunError("scenario records differ from requested participant shards")
    batches: list[MaterializedWindows] = []
    for participant in sorted(participants, key=int):
        reference = preflight.participant_shards[participant]
        batch = _load_preflight_shard_array(
            reference,
            artifact_root=artifact_root,
            class_names=class_names,
        )
        batches.append(batch)
    return _select_across_cached_shards(batches, records, class_names=class_names)


def _adaptation_seed(base_seed: int, fold_id: str, k: int) -> int:
    token = f"inclusive-shift-har|few-person-v1|{base_seed}|{fold_id}|k={k}"
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "big")


def _ablate(signals: Tensor, indices: tuple[int, ...]) -> Tensor:
    if not indices:
        return signals
    result = signals.clone()
    result[:, :, list(indices)] = 0.0
    return result


def _fine_tune_fixed(
    model: nn.Module,
    windows: NDArray[np.float32],
    labels: NDArray[np.int64],
    *,
    config: TrainingConfig,
    adaptation_seed: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], torch.optim.Optimizer]:
    configure_determinism(adaptation_seed)
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(windows)),
        torch.from_numpy(np.ascontiguousarray(labels)),
    )
    generator = torch.Generator().manual_seed(adaptation_seed + 1)
    loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        cast(Any, dataset),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision == "float16")  # type: ignore[attr-defined]
    amp_dtype = torch.bfloat16 if config.mixed_precision == "bfloat16" else torch.float16
    amp_enabled = config.mixed_precision != "disabled"
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        examples = 0
        for signals, targets in loader:
            signals = _ablate(signals.to(device), config.zero_channel_indices)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                output = cast(HAROutput, model(signals))
                loss = F.cross_entropy(output.logits, targets)
            cast(Any, scaler.scale(loss)).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            count = targets.numel()
            total_loss += float(loss.detach()) * count
            examples += count
        history.append(
            {
                "epoch": epoch,
                "training_cross_entropy": total_loss / examples,
                "checkpoint_selected": epoch == config.epochs,
                "validation_accessed": False,
            }
        )
    return history, optimizer


def _write_npz_new(path: Path, arrays: Mapping[str, NDArray[Any]]) -> str:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **cast(dict[str, Any], dict(arrays)))
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_file(path)


def run_few_person_scenario(
    *,
    manifest_path: Path,
    split_manifest_path: Path,
    opening_receipt_path: Path,
    zero_shot_index_path: Path,
    final_freeze_inventory_path: Path,
    raw_csv_path: Path | None,
    primary_cache_record_path: Path | None = None,
    primary_cache_record_sha256: str | None = None,
    artifact_root: Path,
    fold_id: str,
    k: int,
    model_id: str,
    seed: int,
    code_commit: str,
    created_at_utc: str,
    output_directory: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Adapt one frozen neural model and evaluate one untouched outer pair."""

    plan = _json_object(manifest_path, name="few-person manifest")
    plan_sha256 = _self_hash(plan, field="manifest_sha256", name="few-person manifest")
    protocol_id = str(plan.get("protocol_id"))
    if protocol_id == FEW_PERSON_PROTOCOL_ID:
        raise FewPersonRunError(
            "few-person v1 is preserved but scientifically incompatible; use v1.1"
        )
    if protocol_id != FEW_PERSON_V1_1_PROTOCOL_ID:
        raise FewPersonRunError("few-person manifest protocol is unsupported")
    expected_status = "ready_postconfirmatory_functional_core_v1_1_no_scenario_run"
    evidence_status = FEW_PERSON_V1_1_EVIDENCE_STATUS
    required = {
        "status": expected_status,
        "evidence_status": evidence_status,
        "target_metrics_or_predictions_used_for_design_or_selection": False,
        "target_raw_values_accessed_during_manifest_build": False,
    }
    if any(plan.get(key) != value for key, value in required.items()):
        raise FewPersonRunError("few-person manifest contract mismatch")
    cache_requested = primary_cache_record_path is not None
    if (raw_csv_path is None) == (not cache_requested):
        raise FewPersonRunError("select exactly one input mode: raw CSV or primary cache")
    if cache_requested != (primary_cache_record_sha256 is not None):
        raise FewPersonRunError("primary cache record and pinned file hash are both required")
    if cache_requested and protocol_id != FEW_PERSON_V1_1_PROTOCOL_ID:
        raise FewPersonRunError("primary cache input is restricted to functional-core v1.1")
    for path, field in (
        (opening_receipt_path, "opening_receipt_file_sha256"),
        (zero_shot_index_path, "zero_shot_index_file_sha256"),
        (final_freeze_inventory_path, "final_freeze_inventory_file_sha256"),
    ):
        if sha256_file(path) != plan.get(field):
            raise FewPersonRunError(f"post-confirmatory evidence hash changed: {field}")
    split = _json_object(split_manifest_path, name="parent split manifest")
    split_sha256 = _self_hash(split, field="split_manifest_sha256", name="parent split")
    if split_sha256 != plan.get("split_manifest_sha256"):
        raise FewPersonRunError("parent split differs from few-person manifest")
    try:
        validate_few_person_v1_1_manifest_assignments(plan, split)
    except FewPersonProtocolError as exc:
        raise FewPersonRunError(str(exc)) from exc
    scenario, model_entry = _selected(plan, fold_id=fold_id, k=k, model_id=model_id, seed=seed)
    timestamp = require_utc_timestamp(created_at_utc, location="created_at_utc")
    if not torch.cuda.is_available():
        raise FewPersonRunError(
            "few-person neural execution requires CUDA; CPU fallback is forbidden"
        )

    root = artifact_root.resolve(strict=True)
    commit = _require_repository_head(root, code_commit)
    resolved_output_root = output_root.resolve(strict=True)
    try:
        resolved_output_root.relative_to(root)
    except ValueError as exc:
        raise FewPersonRunError("few-person output root escapes artifact root") from exc
    resolved_split_path = _confined(
        split_manifest_path, root=root, name="parent split manifest", must_exist=True
    )
    split_file_sha256 = sha256_file(resolved_split_path)
    checkpoint_record = _mapping(model_entry["checkpoint"], name="checkpoint")
    calibrator_record = _mapping(model_entry["calibrator"], name="calibrator")
    checkpoint_path = _confined(
        root / str(checkpoint_record["path"]), root=root, name="checkpoint", must_exist=True
    )
    calibrator_path = _confined(
        root / str(calibrator_record["path"]), root=root, name="calibrator", must_exist=True
    )
    if sha256_file(checkpoint_path) != checkpoint_record.get("sha256"):
        raise FewPersonRunError("base checkpoint hash changed")
    if sha256_file(calibrator_path) != calibrator_record.get("sha256"):
        raise FewPersonRunError("source calibrator hash changed")
    device = torch.device("cuda")
    model, checkpoint = reconstruct_checkpoint(checkpoint_path, device=device)
    base_configuration = _mapping(checkpoint.get("configuration"), name="base configuration")
    if canonical_json_sha256(base_configuration) != model_entry.get(
        "training_configuration_sha256"
    ):
        raise FewPersonRunError("base checkpoint configuration differs from freeze")
    config = training_config_from_dict(dict(base_configuration))
    if config.checkpoint_selection_rule != "fixed_last_epoch" or config.seed != seed:
        raise FewPersonRunError("base checkpoint is not the predeclared fixed source model")
    normalizer = ChannelStandardizer.from_dict(
        dict(_mapping(checkpoint.get("normalization"), name="checkpoint normalization"))
    )
    expected_normalization = set(cast(list[str], scenario["normalization_fit_subjects"]))
    if set(normalizer.training_participants) != expected_normalization or any(
        participant in {str(value) for value in range(11, 21)}
        for participant in normalizer.training_participants
    ):
        raise FewPersonRunError("normalization is not the frozen source-training-only transform")
    class_names = tuple(str(value) for value in checkpoint["label_schema"])
    if protocol_id == FEW_PERSON_V1_1_PROTOCOL_ID and (
        list(class_names) != plan.get("class_names")
        or canonical_json_sha256(list(class_names)) != plan.get("class_schema_sha256")
    ):
        raise FewPersonRunError("checkpoint class order differs from v1.1 ontology binding")
    calibrator_payload, calibrator = load_source_temperature_calibrator_file(
        calibrator_path,
        expected_checkpoint_sha256=str(checkpoint_record["sha256"]),
        expected_training_configuration_sha256=str(model_entry["training_configuration_sha256"]),
        expected_split_manifest_sha256=split_sha256,
    )
    if calibrator_payload.get("target_subject_or_window_records_used") is not False:
        raise FewPersonRunError("calibrator is not source-only")
    environment = _execution_environment(device, cudnn_enabled_during_run=not config.disable_cudnn)

    inclusion = set(cast(list[str], scenario["target_inclusion_subjects"]))
    evaluation = set(cast(list[str], scenario["evaluation_subjects"]))
    ontology_track = str(plan["ontology_track"])
    train_records = _window_records(
        split,
        subjects=inclusion,
        ontology_track=ontology_track,
        expected_count=int(scenario["target_inclusion_window_count"]),
        expected_ids_sha256=str(scenario["target_inclusion_window_ids_sha256"]),
    )
    evaluation_records = _window_records(
        split,
        subjects=evaluation,
        ontology_track=ontology_track,
        expected_count=int(scenario["evaluation_window_count"]),
        expected_ids_sha256=str(scenario["evaluation_window_ids_sha256"]),
    )
    if {record.window_id for record in train_records} & {
        record.window_id for record in evaluation_records
    }:
        raise FewPersonRunError("inclusion/evaluation window identities overlap")
    source_sha256 = split.get("source_artifact_sha256")
    if source_sha256 is None:
        source_sha256 = _mapping(
            split.get("source_evidence"), name="parent split source evidence"
        ).get("sensor_artifact_sha256")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise FewPersonRunError("parent split source artifact hash is invalid")
    cache_preflight: _PrimaryCacheMetadataPreflight | None = None
    if cache_requested:
        assert primary_cache_record_path is not None
        assert primary_cache_record_sha256 is not None
        cache_preflight = _preflight_primary_cache_metadata(
            record_path=primary_cache_record_path,
            expected_record_file_sha256=primary_cache_record_sha256,
            artifact_root=artifact_root,
            split_manifest_path=resolved_split_path,
            split_manifest_sha256=split_sha256,
            split_manifest_file_sha256=split_file_sha256,
            source_artifact_sha256=source_sha256,
            plan=plan,
        )
        training_lineage = _shard_input_lineage(
            cache_preflight, inclusion, train_records, root=root
        )
        evaluation_lineage = _shard_input_lineage(
            cache_preflight, evaluation, evaluation_records, root=root
        )
        training_raw_path: Path | None = None
    else:
        assert raw_csv_path is not None
        training_raw_path = _confined(
            raw_csv_path, root=root, name="raw sensor CSV", must_exist=True
        )
        if sha256_file(training_raw_path) != source_sha256:
            raise FewPersonRunError("InclusiveHAR source artifact hash mismatch")
        evaluation_lineage = _raw_input_lineage(
            training_raw_path, root=root, source_sha256=source_sha256
        )
        training_lineage = _raw_input_lineage(
            training_raw_path, root=root, source_sha256=source_sha256
        )
    split_lineage = {
        "path": resolved_split_path.relative_to(root).as_posix(),
        "record_sha256": split_sha256,
        "file_sha256": split_file_sha256,
    }
    checkpoint_input_lineage = {
        "schema_version": "1.0.0",
        "split_manifest": split_lineage,
        "training": training_lineage,
        "evaluation": evaluation_lineage,
        "access_barrier": {
            "evaluation_metadata_preflight_only_before_adaptation": cache_requested,
            "evaluation_signals_accessed_before_adapted_checkpoint_fixed": False,
            "adapted_checkpoint_fixed_before_evaluation": False,
        },
    }
    started = time.perf_counter()
    original_cudnn = torch.backends.cudnn.enabled
    output = _confined(
        output_directory,
        root=resolved_output_root,
        name="output directory",
        must_exist=False,
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite few-person output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    try:
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cudnn.enabled = not config.disable_cudnn
        # Cache mode loads only the k inclusion-participant shards. It never
        # opens the whole-target cache or either held-out participant shard.
        if cache_preflight is not None:
            train_batch = _load_cached_participant_shards(
                preflight=cache_preflight,
                participants=inclusion,
                records=train_records,
                class_names=class_names,
                artifact_root=artifact_root,
            )
        else:
            assert training_raw_path is not None
            train_batch = _materialize_authorized_records(
                training_raw_path,
                train_records,
                expected_source_sha256=source_sha256,
                class_names=class_names,
                ontology_track=ontology_track,
            )
        if set(train_batch.participant_ids) != inclusion:
            raise FewPersonRunError("training materialization differs from inclusion participants")
        adaptation_seed = _adaptation_seed(seed, fold_id, k)
        history, optimizer = _fine_tune_fixed(
            model,
            normalizer.transform(train_batch.signals),
            train_batch.labels,
            config=config,
            adaptation_seed=adaptation_seed,
            device=device,
        )
        checkpoint_output = output / "adapted.pt"
        checkpoint_payload = {
            "schema_version": "1.0.0",
            "record_kind": "few_person_adapted_neural_checkpoint",
            "created_at_utc": timestamp,
            "evidence_status": evidence_status,
            "protocol_id": protocol_id,
            "model_id": model_id,
            "seed": seed,
            "adaptation_seed": adaptation_seed,
            "fold_id": fold_id,
            "k": k,
            "inclusion_subjects": sorted(inclusion, key=int),
            "inclusion_window_count": len(train_records),
            "inclusion_window_ids_sha256": canonical_json_sha256(
                sorted(record.window_id for record in train_records)
            ),
            "evaluation_subjects_excluded_from_training": sorted(evaluation, key=int),
            "normalization_fit_subjects": list(normalizer.training_participants),
            "target_validation_performed": False,
            "calibration_refit_on_target": False,
            "threshold_selection_performed": False,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "normalization": normalizer.to_dict(),
            "label_schema": list(class_names),
            "base_checkpoint_sha256": checkpoint_record["sha256"],
            "base_training_configuration": dict(base_configuration),
            "base_training_configuration_sha256": model_entry["training_configuration_sha256"],
            "few_person_manifest_sha256": plan_sha256,
            "split_manifest_sha256": split_sha256,
            "code_commit": commit,
            "environment": environment,
            "checkpoint_selection_rule": "fixed_last_epoch_no_validation",
            "training_history": history,
            "rng_states": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all(),
            },
            "parameter_count": trainable_parameter_count(model),
            "input_materialization": checkpoint_input_lineage,
        }
        with checkpoint_output.open("xb") as stream:
            torch.save(checkpoint_payload, stream)
        checkpoint_sha256 = sha256_file(checkpoint_output)

        # No held-out signal array is loaded or sliced until the adapted
        # checkpoint has been durably written and hash-fixed above.
        if cache_preflight is not None:
            evaluation_batch = _load_cached_participant_shards(
                preflight=cache_preflight,
                participants=evaluation,
                records=evaluation_records,
                class_names=class_names,
                artifact_root=artifact_root,
                adapted_checkpoint_path=checkpoint_output,
                expected_adapted_checkpoint_sha256=checkpoint_sha256,
            )
        else:
            assert training_raw_path is not None
            evaluation_batch = _materialize_authorized_records(
                training_raw_path,
                evaluation_records,
                expected_source_sha256=source_sha256,
                class_names=class_names,
                ontology_track=ontology_track,
            )
        if set(evaluation_batch.participant_ids) != evaluation:
            raise FewPersonRunError("evaluation materialization differs from outer participants")
        logits, uncalibrated, _ = predict_model(
            model,
            normalizer.transform(evaluation_batch.signals),
            evaluation_batch.labels,
            list(evaluation_batch.participant_ids),
            class_names=class_names,
            batch_size=config.batch_size,
            device=device,
            mixed_precision=config.mixed_precision,
            zero_channel_indices=config.zero_channel_indices,
        )
        calibrated = calibrator.probabilities(logits)
        report = classification_report(
            evaluation_batch.labels,
            calibrated,
            list(evaluation_batch.participant_ids),
            class_names=class_names,
        )
        prediction_path = output / "predictions.npz"
        prediction_sha256 = _write_npz_new(
            prediction_path,
            {
                "window_ids": np.asarray(evaluation_batch.window_ids),
                "participant_ids": np.asarray(evaluation_batch.participant_ids),
                "true_labels": evaluation_batch.labels,
                "logits": logits,
                "uncalibrated_probabilities": uncalibrated,
                "source_temperature_probabilities": calibrated,
                "predicted_labels": calibrated.argmax(axis=1).astype(np.int64),
            },
        )
        torch.cuda.synchronize(device)
        result_input_lineage = {
            "schema_version": "1.0.0",
            "split_manifest": split_lineage,
            "training": training_lineage,
            "evaluation": evaluation_lineage,
            "access_barrier": {
                "evaluation_metadata_preflight_only_before_adaptation": cache_requested,
                "evaluation_signals_accessed_before_adapted_checkpoint_fixed": False,
                "adapted_checkpoint_fixed_before_evaluation": True,
                "adapted_checkpoint_sha256": checkpoint_sha256,
            },
        }
        result: dict[str, Any] = {
            "schema_version": "1.0.0",
            "record_kind": "few_person_outer_fold_result",
            "created_at_utc": timestamp,
            "status": "complete_create_only_postconfirmatory_secondary",
            "evidence_status": evidence_status,
            "protocol_id": protocol_id,
            "model_id": model_id,
            "seed": seed,
            "adaptation_seed": adaptation_seed,
            "fold_id": fold_id,
            "k": k,
            "few_person_manifest_sha256": plan_sha256,
            "split_manifest_sha256": split_sha256,
            "opening_receipt_record_sha256": plan["opening_receipt_record_sha256"],
            "zero_shot_index_record_sha256": plan["zero_shot_index_record_sha256"],
            "base_checkpoint_sha256": checkpoint_record["sha256"],
            "base_source_calibrator_sha256": calibrator_record["sha256"],
            "base_training_configuration_sha256": model_entry["training_configuration_sha256"],
            "source_hyperparameters_inherited": {
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "gradient_clip_norm": config.gradient_clip_norm,
                "mixed_precision": config.mixed_precision,
                "disable_cudnn": config.disable_cudnn,
            },
            "adaptation_objective": "supervised_cross_entropy_only",
            "checkpoint_selection_rule": "fixed_last_epoch_no_validation",
            "normalization_refit_on_target": False,
            "calibration_refit_on_target": False,
            "threshold_selection_performed": False,
            "target_validation_performed": False,
            "inclusion_subjects": sorted(inclusion, key=int),
            "evaluation_subjects": sorted(evaluation, key=int),
            "unused_target_subjects": scenario["unused_target_subjects"],
            "adapted_checkpoint": {
                "path": checkpoint_output.relative_to(root).as_posix(),
                "sha256": checkpoint_sha256,
            },
            "prediction_artifact": {
                "path": prediction_path.relative_to(root).as_posix(),
                "sha256": prediction_sha256,
            },
            "participant_level_report": report,
            "statistical_unit": "participant",
            "target_information_used_for_model_or_hyperparameter_selection": False,
            "input_materialization": result_input_lineage,
            "device": {
                "type": "cuda",
                "name": torch.cuda.get_device_properties(device).name,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cudnn_enabled": torch.backends.cudnn.enabled,
            },
            "code_commit": commit,
            "environment": environment,
            "elapsed_seconds": time.perf_counter() - started,
        }
        result["record_sha256"] = canonical_json_sha256(result)
        atomic_write_json_new(result, output / "result.json", allowed_root=output_root)
        return result
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": "1.0.0",
            "record_kind": "few_person_failed_run",
            "created_at_utc": timestamp,
            "status": "failed_preserved_postconfirmatory_secondary",
            "evidence_status": "failed_run_not_result",
            "protocol_id": protocol_id,
            "fold_id": fold_id,
            "k": k,
            "model_id": model_id,
            "seed": seed,
            "few_person_manifest_sha256": plan_sha256,
            "code_commit": commit,
            "environment": environment,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "input_materialization": checkpoint_input_lineage,
            "elapsed_seconds": time.perf_counter() - started,
        }
        failure["record_sha256"] = canonical_json_sha256(failure)
        atomic_write_json_new(failure, output / "failure.json", allowed_root=output_root)
        raise FewPersonRunError(
            f"few-person scenario failed; evidence preserved at {output / 'failure.json'}"
        ) from exc
    finally:
        torch.backends.cudnn.enabled = original_cudnn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser(
        "build-manifest", help="build a create-only v1.1 manifest; v1 regeneration is disabled"
    )
    build.add_argument("--split-manifest", type=Path, required=True)
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--opening-receipt", type=Path, required=True)
    build.add_argument("--zero-shot-index", type=Path, required=True)
    build.add_argument("--final-freeze-inventory", type=Path, required=True)
    build.add_argument("--superseded-v1-manifest", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--allowed-root", type=Path, required=True)
    run = subparsers.add_parser("run-scenario")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--split-manifest", type=Path, required=True)
    run.add_argument("--opening-receipt", type=Path, required=True)
    run.add_argument("--zero-shot-index", type=Path, required=True)
    run.add_argument("--final-freeze-inventory", type=Path, required=True)
    inputs = run.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--raw-csv", type=Path)
    inputs.add_argument("--primary-cache-record", type=Path)
    run.add_argument("--primary-cache-record-sha256")
    run.add_argument("--artifact-root", type=Path, required=True)
    run.add_argument("--fold-id", required=True)
    run.add_argument("--k", type=int, choices=(1, 2, 4), required=True)
    run.add_argument("--model-id", required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--code-commit", required=True)
    run.add_argument("--created-at-utc", required=True)
    run.add_argument("--output-directory", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build-manifest":
            manifest = build_few_person_v1_1_manifest(
                split_manifest_path=args.split_manifest,
                config_path=args.config,
                opening_receipt_path=args.opening_receipt,
                zero_shot_index_path=args.zero_shot_index,
                final_freeze_inventory_path=args.final_freeze_inventory,
                superseded_v1_manifest_path=args.superseded_v1_manifest,
            )
            write_few_person_v1_1_manifest_new(
                manifest, args.output, allowed_root=args.allowed_root
            )
            payload: Mapping[str, Any] = {
                "status": manifest["status"],
                "manifest_sha256": manifest["manifest_sha256"],
                "output": str(args.output),
                "target_metrics_or_predictions_used_for_design_or_selection": False,
                "target_raw_values_accessed_during_manifest_build": False,
            }
        else:
            result = run_few_person_scenario(
                manifest_path=args.manifest,
                split_manifest_path=args.split_manifest,
                opening_receipt_path=args.opening_receipt,
                zero_shot_index_path=args.zero_shot_index,
                final_freeze_inventory_path=args.final_freeze_inventory,
                raw_csv_path=args.raw_csv,
                primary_cache_record_path=args.primary_cache_record,
                primary_cache_record_sha256=args.primary_cache_record_sha256,
                artifact_root=args.artifact_root,
                fold_id=args.fold_id,
                k=args.k,
                model_id=args.model_id,
                seed=args.seed,
                code_commit=args.code_commit,
                created_at_utc=args.created_at_utc,
                output_directory=args.output_directory,
                output_root=args.output_root,
            )
            payload = {
                "status": result["status"],
                "record_sha256": result["record_sha256"],
                "fold_id": result["fold_id"],
                "k": result["k"],
                "model_id": result["model_id"],
                "seed": result["seed"],
            }
    except (
        FewPersonProtocolError,
        FewPersonRunError,
        FileExistsError,
        OSError,
        ValueError,
    ) as exc:
        print(json.dumps({"status": "fail", "message": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
