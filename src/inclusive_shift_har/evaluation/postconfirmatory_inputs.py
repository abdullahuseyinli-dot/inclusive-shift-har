"""Validated inputs for secondary operations after the single target opening.

No function in this module accepts an unlock record, raw CSV path, or opening
acknowledgement.  Target signals must already exist in a hash-pinned ignored
materialization cache whose ordered identity matches opening-1 evidence.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from inclusive_shift_har.data.materialize import MaterializedWindows
from inclusive_shift_har.evaluation.locked_target import LOCKED_TARGET_EVIDENCE_STATUS
from inclusive_shift_har.manifests.canonical import (
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LOCKED_ARRAY_KEYS = {
    "evidence_status",
    "window_ids",
    "participant_ids",
    "true_labels",
    "logits",
    "uncalibrated_probabilities",
    "calibrated_probabilities",
    "predicted_labels",
}
_CACHE_ARRAY_KEYS = {
    "signals",
    "labels",
    "window_ids",
    "participant_ids",
    "released_labels",
    "partitions",
}


class PostconfirmatoryInputError(ValueError):
    """Raised when consumed-opening or materialized-cache lineage is incomplete."""


@dataclass(frozen=True, slots=True)
class ConsumedTargetContext:
    artifact_root: Path
    receipt_path: Path
    receipt_file_sha256: str
    receipt: Mapping[str, Any]
    index_path: Path
    index_file_sha256: str
    index: Mapping[str, Any]
    class_names: tuple[str, ...]
    result_entries: Mapping[tuple[str, int], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class MaterializedCacheEvidence:
    batch: MaterializedWindows
    array_path: Path
    array_sha256: str
    metadata_path: Path
    metadata_sha256: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TargetCleanReference:
    model_id: str
    seed: int
    prediction_sha256: str
    record_sha256: str
    record_file_sha256: str
    record_path: Path
    participant_level_report: Mapping[str, Any]


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PostconfirmatoryInputError(f"{name} must be an object")
    return value


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PostconfirmatoryInputError(f"{name} must be a lowercase SHA-256")
    return value


def _validate_self_hash(record: Mapping[str, Any], *, field: str, name: str) -> str:
    claimed = _sha256(record.get(field), name=f"{name} {field}")
    body = dict(record)
    body.pop(field, None)
    if claimed != canonical_json_sha256(body):
        raise PostconfirmatoryInputError(f"{name} self-hash does not validate")
    return claimed


def _resolve_file(value: str | Path, *, root: Path, name: str) -> Path:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    if candidate.is_symlink():
        raise PostconfirmatoryInputError(f"{name} may not be a symlink")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PostconfirmatoryInputError(f"{name} escapes artifact_root") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise PostconfirmatoryInputError(f"{name} must be a regular non-symlink file")
    return resolved


def load_consumed_target_context(
    *,
    opening_receipt_path: str | Path,
    locked_target_index_path: str | Path,
    artifact_root: str | Path,
) -> ConsumedTargetContext:
    """Validate completion of opening 1 without creating or consuming an unlock."""

    root = Path(artifact_root).resolve(strict=True)
    receipt_path = _resolve_file(opening_receipt_path, root=root, name="opening receipt")
    index_path = _resolve_file(locked_target_index_path, root=root, name="locked target index")
    receipt = _mapping(load_json_strict(receipt_path), name="opening receipt")
    index = _mapping(load_json_strict(index_path), name="locked target index")
    receipt_hash = _validate_self_hash(receipt, field="record_sha256", name="opening receipt")
    _validate_self_hash(index, field="record_sha256", name="locked target index")
    receipt_required = {
        "record_kind": "confirmatory_target_opening_receipt",
        "status": "unlock_consumed_before_materialization",
        "target_opening_number": 1,
    }
    index_required = {
        "record_kind": "locked_target_evaluation_index",
        "status": "complete_create_only",
        "evidence_status": LOCKED_TARGET_EVIDENCE_STATUS,
        "target_opening_number": 1,
        "target_information_used_for_model_selection": False,
    }
    if any(receipt.get(key) != expected for key, expected in receipt_required.items()):
        raise PostconfirmatoryInputError("receipt is not the consumed opening-1 receipt")
    if any(index.get(key) != expected for key, expected in index_required.items()):
        raise PostconfirmatoryInputError("index is not completed opening-1 evidence")
    for key in ("split_manifest_sha256", "target_seal_id"):
        if _sha256(receipt.get(key), name=f"receipt {key}") != _sha256(
            index.get(key), name=f"index {key}"
        ):
            raise PostconfirmatoryInputError(f"receipt and index disagree on {key}")
    receipt_file_hash = sha256_file(receipt_path)
    if (
        index.get("opening_receipt_record_sha256") != receipt_hash
        or index.get("opening_receipt_file_sha256") != receipt_file_hash
    ):
        raise PostconfirmatoryInputError("locked index is not bound to the exact receipt file")
    class_names_value = index.get("class_names")
    if (
        not isinstance(class_names_value, list)
        or len(class_names_value) < 2
        or any(not isinstance(value, str) or not value for value in class_names_value)
        or len(set(class_names_value)) != len(class_names_value)
    ):
        raise PostconfirmatoryInputError("locked target index class schema is invalid")
    entries_value = index.get("results")
    if not isinstance(entries_value, list) or len(entries_value) != index.get(
        "model_seed_result_count"
    ):
        raise PostconfirmatoryInputError("locked target index result list is incomplete")
    entries: dict[tuple[str, int], Mapping[str, Any]] = {}
    for value in entries_value:
        entry = _mapping(value, name="locked target index entry")
        model_id = entry.get("model_id")
        seed = entry.get("seed")
        if (
            not isinstance(model_id, str)
            or not model_id
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or (model_id, seed) in entries
        ):
            raise PostconfirmatoryInputError("locked target index model/seed keys are invalid")
        entries[(model_id, seed)] = entry
    _sha256(index.get("final_freeze_inventory_sha256"), name="freeze inventory hash")
    return ConsumedTargetContext(
        artifact_root=root,
        receipt_path=receipt_path,
        receipt_file_sha256=receipt_file_hash,
        receipt=receipt,
        index_path=index_path,
        index_file_sha256=sha256_file(index_path),
        index=index,
        class_names=tuple(class_names_value),
        result_entries=entries,
    )


def load_target_clean_reference(
    context: ConsumedTargetContext,
    *,
    model_id: str,
    seed: int,
) -> TargetCleanReference:
    """Load a stored clean report; never rerun target clean inference."""

    try:
        entry = context.result_entries[(model_id, seed)]
    except KeyError as exc:
        raise PostconfirmatoryInputError(
            f"opening-1 index lacks target clean reference for {model_id} seed {seed}"
        ) from exc
    record_path_value = entry.get("record_path")
    if not isinstance(record_path_value, str):
        raise PostconfirmatoryInputError("target result record path is invalid")
    record_path = _resolve_file(
        record_path_value, root=context.artifact_root, name="target clean result record"
    )
    record_file_hash = sha256_file(record_path)
    if record_file_hash != entry.get("record_file_sha256"):
        raise PostconfirmatoryInputError("target clean result record file hash changed")
    record = _mapping(load_json_strict(record_path), name="target clean result record")
    record_hash = _validate_self_hash(
        record, field="record_sha256", name="target clean result record"
    )
    required = {
        "record_kind": "locked_target_per_seed_result",
        "status": "complete_create_only",
        "evidence_status": LOCKED_TARGET_EVIDENCE_STATUS,
        "target_opening_number": 1,
        "model_id": model_id,
        "seed": seed,
        "class_names": list(context.class_names),
        "target_information_used_for_model_selection": False,
        "opening_receipt_record_sha256": context.receipt["record_sha256"],
        "final_freeze_inventory_sha256": context.index["final_freeze_inventory_sha256"],
    }
    mismatches = [key for key, expected in required.items() if record.get(key) != expected]
    if mismatches or record_hash != entry.get("record_sha256"):
        raise PostconfirmatoryInputError(
            f"target clean result differs from opening-1 index: {mismatches}"
        )
    prediction = _mapping(record.get("prediction_array"), name="target prediction array")
    prediction_hash = _sha256(prediction.get("sha256"), name="target prediction hash")
    if prediction_hash != entry.get("array_sha256"):
        raise PostconfirmatoryInputError("target prediction hash differs from opening-1 index")
    indexed_array_path = entry.get("array_path")
    if not isinstance(indexed_array_path, str):
        raise PostconfirmatoryInputError("target prediction index path is invalid")
    array_path = _resolve_file(
        indexed_array_path, root=context.artifact_root, name="target clean prediction array"
    )
    if sha256_file(array_path) != prediction_hash:
        raise PostconfirmatoryInputError("target clean prediction array bytes changed")
    report = _mapping(record.get("participant_level_report"), name="target participant report")
    participants = report.get("participants")
    if not isinstance(participants, list) or not participants:
        raise PostconfirmatoryInputError("target participant report is empty")
    return TargetCleanReference(
        model_id=model_id,
        seed=seed,
        prediction_sha256=prediction_hash,
        record_sha256=record_hash,
        record_file_sha256=record_file_hash,
        record_path=record_path,
        participant_level_report=report,
    )


def _load_target_alignment(
    context: ConsumedTargetContext,
) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    if not context.result_entries:
        raise PostconfirmatoryInputError("locked target index has no result entries")
    key = sorted(context.result_entries, key=lambda value: (value[0].casefold(), value[1]))[0]
    entry = context.result_entries[key]
    load_target_clean_reference(context, model_id=key[0], seed=key[1])
    prediction_path_value = entry.get("array_path")
    if not isinstance(prediction_path_value, str):
        raise PostconfirmatoryInputError("target alignment index path is invalid")
    array_path = _resolve_file(
        prediction_path_value, root=context.artifact_root, name="target alignment prediction"
    )
    if sha256_file(array_path) != entry.get("array_sha256"):
        raise PostconfirmatoryInputError("target alignment prediction bytes changed")
    with np.load(array_path, allow_pickle=False) as arrays:
        if set(arrays.files) != _LOCKED_ARRAY_KEYS:
            raise PostconfirmatoryInputError("target alignment prediction keys changed")
        window_ids = tuple(str(value) for value in arrays["window_ids"].tolist())
        participant_ids = tuple(str(value) for value in arrays["participant_ids"].tolist())
        labels = np.asarray(arrays["true_labels"], dtype=np.int64)
    return window_ids, participant_ids, labels


def load_materialized_cache_evidence(
    *,
    array_path: str | Path,
    metadata_path: str | Path,
    expected_array_sha256: str,
    expected_metadata_sha256: str,
    artifact_root: str | Path,
    expected_partition: str,
    expected_split_manifest_sha256: str,
    expected_class_names: tuple[str, ...],
    target_context: ConsumedTargetContext | None = None,
    expected_target_participants: frozenset[str] | None = None,
) -> MaterializedCacheEvidence:
    """Load a hash-pinned existing cache; raw materialization is intentionally absent."""

    if expected_partition not in {"source_validation", "target_sealed"}:
        raise PostconfirmatoryInputError("unsupported secondary cache partition")
    if (expected_partition == "target_sealed") != (target_context is not None):
        raise PostconfirmatoryInputError("target cache requires the consumed target context")
    if expected_target_participants is not None and (
        target_context is None or not expected_target_participants
    ):
        raise PostconfirmatoryInputError(
            "target participant subset requires a non-empty consumed-target selection"
        )
    root = Path(artifact_root).resolve(strict=True)
    array_file = _resolve_file(array_path, root=root, name="materialized cache array")
    metadata_file = _resolve_file(metadata_path, root=root, name="materialized cache metadata")
    array_hash = _sha256(expected_array_sha256, name="expected cache array hash")
    metadata_hash = _sha256(expected_metadata_sha256, name="expected cache metadata hash")
    if sha256_file(array_file) != array_hash or sha256_file(metadata_file) != metadata_hash:
        raise PostconfirmatoryInputError("materialized cache file hash mismatch")
    metadata = _mapping(load_json_strict(metadata_file), name="materialized cache metadata")
    if metadata.get("array_sha256") != array_hash:
        raise PostconfirmatoryInputError("cache metadata points to different array bytes")
    if metadata.get("class_names") != list(expected_class_names):
        raise PostconfirmatoryInputError("cache class schema differs from frozen evaluation")
    if metadata.get("ontology_track") != "functional_core":
        raise PostconfirmatoryInputError("cache ontology track is not functional_core")
    lineage = _mapping(metadata.get("lineage"), name="materialized cache lineage")
    if lineage.get("split_manifest_sha256") != expected_split_manifest_sha256:
        raise PostconfirmatoryInputError("cache split lineage differs from the freeze")
    with np.load(array_file, allow_pickle=False) as arrays:
        if set(arrays.files) != _CACHE_ARRAY_KEYS:
            raise PostconfirmatoryInputError("materialized cache keys differ from convention")
        raw_signals = np.asarray(arrays["signals"])
        raw_labels = np.asarray(arrays["labels"])
        if raw_signals.dtype != np.dtype(np.float32) or raw_labels.dtype != np.dtype(np.int64):
            raise PostconfirmatoryInputError("cache signal/label dtypes differ from convention")
        signals = np.asarray(raw_signals, dtype=np.float32)
        labels = np.asarray(raw_labels, dtype=np.int64)
        window_ids = tuple(str(value) for value in arrays["window_ids"].tolist())
        participant_ids = tuple(str(value) for value in arrays["participant_ids"].tolist())
        released_labels = tuple(str(value) for value in arrays["released_labels"].tolist())
        partitions = tuple(str(value) for value in arrays["partitions"].tolist())
    count = signals.shape[0]
    if signals.shape != (count, 128, 6) or labels.shape != (count,) or count < 1:
        raise PostconfirmatoryInputError("cache violates aligned [N,128,6] shape")
    if (
        not np.isfinite(signals).all()
        or len(set(window_ids)) != count
        or any(not value for value in window_ids + participant_ids)
    ):
        raise PostconfirmatoryInputError("cache signals or stable window IDs are invalid")
    if any(
        len(values) != count
        for values in (window_ids, participant_ids, released_labels, partitions)
    ):
        raise PostconfirmatoryInputError("cache arrays are not row-aligned")
    if set(partitions) != {expected_partition}:
        raise PostconfirmatoryInputError("cache contains an unauthorized partition")
    if labels.min() < 0 or labels.max() >= len(expected_class_names):
        raise PostconfirmatoryInputError("cache labels lie outside the frozen class schema")
    if metadata.get("window_count") != count or metadata.get("shape") != list(signals.shape):
        raise PostconfirmatoryInputError("cache metadata count/shape differs from arrays")
    if target_context is not None:
        if (
            lineage.get("opening_receipt_record_sha256") != target_context.receipt["record_sha256"]
            or lineage.get("target_seal_id") != target_context.index["target_seal_id"]
        ):
            raise PostconfirmatoryInputError("target cache is not bound to consumed opening 1")
        expected_windows, expected_participants, expected_labels = _load_target_alignment(
            target_context
        )
        if expected_target_participants is not None:
            positions = [
                index
                for index, participant in enumerate(expected_participants)
                if participant in expected_target_participants
            ]
            if set(expected_participants[index] for index in positions) != set(
                expected_target_participants
            ):
                raise PostconfirmatoryInputError(
                    "requested target participants are absent from opening-1 alignment"
                )
            expected_windows = tuple(expected_windows[index] for index in positions)
            expected_participants = tuple(expected_participants[index] for index in positions)
            expected_labels = np.asarray(expected_labels[positions], dtype=np.int64)
        if (
            window_ids != expected_windows
            or participant_ids != expected_participants
            or not np.array_equal(labels, expected_labels)
        ):
            raise PostconfirmatoryInputError(
                "target cache identity differs from opening-1 prediction alignment"
            )
    batch = MaterializedWindows(
        signals=signals,
        labels=labels,
        window_ids=window_ids,
        participant_ids=participant_ids,
        released_labels=released_labels,
        partitions=partitions,
        class_names=expected_class_names,
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
