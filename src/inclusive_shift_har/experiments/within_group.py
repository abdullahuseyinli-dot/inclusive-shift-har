"""One-cell CUDA runner for post-confirmatory disabled-cohort evaluation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from inclusive_shift_har.calibration import fit_temperature
from inclusive_shift_har.data.inclusivehar import INCLUSIVEHAR_PRIMARY_CHANNELS
from inclusive_shift_har.data.materialize import MaterializedWindows
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.evaluation.postconfirmatory_inputs import (
    ConsumedTargetContext,
    load_consumed_target_context,
)
from inclusive_shift_har.experiments.postconfirmatory_cache import (
    PRIMARY_CACHE_EVIDENCE_STATUS,
    PRIMARY_CACHE_RECORD_KIND,
    PRIMARY_CACHE_SCHEMA_VERSION,
)
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_bytes,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)
from inclusive_shift_har.models.common import HAROutput, trainable_parameter_count
from inclusive_shift_har.preprocessing.normalization import ChannelStandardizer
from inclusive_shift_har.protocols.within_group import (
    FUNCTIONAL_CORE_CLASS_ORDER,
    REQUIRED_SEEDS,
    SUPPORTED_MODEL_IDS,
    TARGET_PARTICIPANTS,
    WITHIN_GROUP_EVIDENCE_STATUS,
    WITHIN_GROUP_PROTOCOL_ID,
    WITHIN_GROUP_SCHEMA_VERSION,
)
from inclusive_shift_har.training.engine import (
    TrainingConfig,
    build_model,
    configure_determinism,
    predict_model,
    training_config_from_dict,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_CACHE_ARRAY_KEYS = {
    "signals",
    "labels",
    "window_ids",
    "participant_ids",
    "released_labels",
    "partitions",
}


class WithinGroupRunError(RuntimeError):
    """Raised when a within-group cell violates protocol or evidence lineage."""


@dataclass(frozen=True, slots=True)
class ShardReference:
    participant_id: str
    window_count: int
    window_ids_sha256: str
    ordered_window_ids_sha256: str
    ordered_participant_ids_sha256: str
    ordered_labels_sha256: str
    array_path: Path
    array_sha256: str
    metadata_path: Path
    metadata_sha256: str


@dataclass(frozen=True, slots=True)
class ParticipantShardStore:
    artifact_root: Path
    record_path: Path
    record_file_sha256: str
    record_sha256: str
    split_manifest_sha256: str
    source_artifact_sha256: str
    opening_receipt_record_sha256: str
    opening_index_record_sha256: str
    shards: Mapping[str, ShardReference]
    split_records_by_participant: Mapping[str, tuple[Mapping[str, Any], ...]]

    def load_role(
        self, participants: Sequence[str], *, role: str
    ) -> tuple[MaterializedWindows, dict[str, Any]]:
        """Parse only the participant shards declared for one fold role."""

        expected_participants = tuple(participants)
        if (
            not expected_participants
            or len(set(expected_participants)) != len(expected_participants)
            or not set(expected_participants) <= TARGET_PARTICIPANTS
        ):
            raise WithinGroupRunError(f"{role} participant selection is invalid")
        rows: dict[str, tuple[NDArray[np.float32], int, str, str, str]] = {}
        references: list[dict[str, Any]] = []
        expected_records: list[Mapping[str, Any]] = []
        for participant in expected_participants:
            reference = self.shards[participant]
            expected_records.extend(self.split_records_by_participant[participant])
            with np.load(reference.array_path, allow_pickle=False) as arrays:
                if set(arrays.files) != _CACHE_ARRAY_KEYS:
                    raise WithinGroupRunError(f"target shard {participant} array keys changed")
                raw_signals = np.asarray(arrays["signals"])
                raw_labels = np.asarray(arrays["labels"])
                if raw_signals.dtype != np.dtype(np.float32) or raw_labels.dtype != np.dtype(
                    np.int64
                ):
                    raise WithinGroupRunError(f"target shard {participant} dtypes changed")
                signals = np.asarray(raw_signals, dtype=np.float32)
                labels = np.asarray(raw_labels, dtype=np.int64)
                window_ids = tuple(str(value) for value in arrays["window_ids"].tolist())
                participant_ids = tuple(str(value) for value in arrays["participant_ids"].tolist())
                released_labels = tuple(str(value) for value in arrays["released_labels"].tolist())
                partitions = tuple(str(value) for value in arrays["partitions"].tolist())
            count = reference.window_count
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
                or labels.max() >= len(FUNCTIONAL_CORE_CLASS_ORDER)
                or canonical_json_sha256(sorted(window_ids)) != reference.window_ids_sha256
                or canonical_json_sha256(list(window_ids)) != reference.ordered_window_ids_sha256
                or canonical_json_sha256(list(participant_ids))
                != reference.ordered_participant_ids_sha256
                or canonical_json_sha256(labels.tolist()) != reference.ordered_labels_sha256
            ):
                raise WithinGroupRunError(f"target shard {participant} alignment changed")
            for index, window_id in enumerate(window_ids):
                if window_id in rows:
                    raise WithinGroupRunError("selected target shards overlap")
                rows[window_id] = (
                    np.asarray(signals[index], dtype=np.float32),
                    int(labels[index]),
                    participant_ids[index],
                    released_labels[index],
                    partitions[index],
                )
            references.append(
                {
                    "participant_id": participant,
                    "window_count": count,
                    "window_ids_sha256": reference.window_ids_sha256,
                    "array_path": reference.array_path.relative_to(self.artifact_root).as_posix(),
                    "array_sha256": reference.array_sha256,
                    "metadata_path": reference.metadata_path.relative_to(
                        self.artifact_root
                    ).as_posix(),
                    "metadata_sha256": reference.metadata_sha256,
                }
            )
        expected_records.sort(key=lambda item: int(cast(int, item["start_row_inclusive"])))
        expected_ids = [str(record["window_id"]) for record in expected_records]
        if set(rows) != set(expected_ids):
            raise WithinGroupRunError(f"{role} shard union differs from the parent split")
        class_to_index = {name: index for index, name in enumerate(FUNCTIONAL_CORE_CLASS_ORDER)}
        ordered_signals: list[NDArray[np.float32]] = []
        ordered_labels: list[int] = []
        ordered_participants: list[str] = []
        ordered_released: list[str] = []
        for record in expected_records:
            window_id = str(record["window_id"])
            signal, label, participant, released, partition = rows[window_id]
            canonical = str(
                _mapping(record["canonical_labels"], name="window labels")["functional_core"]
            )
            if (
                participant != record["subject_id"]
                or released != record["activity_label"]
                or partition != "target_sealed"
                or label != class_to_index[canonical]
            ):
                raise WithinGroupRunError(f"{role} shard metadata differs from split")
            ordered_signals.append(signal)
            ordered_labels.append(label)
            ordered_participants.append(participant)
            ordered_released.append(released)
        batch = MaterializedWindows(
            signals=np.asarray(ordered_signals, dtype=np.float32),
            labels=np.asarray(ordered_labels, dtype=np.int64),
            window_ids=tuple(expected_ids),
            participant_ids=tuple(ordered_participants),
            released_labels=tuple(ordered_released),
            partitions=("target_sealed",) * len(expected_ids),
            class_names=FUNCTIONAL_CORE_CLASS_ORDER,
            ontology_track="functional_core",
        )
        lineage = {
            "mode": "hash_pinned_target_participant_shards",
            "role": role,
            "participant_ids": list(expected_participants),
            "window_count": len(expected_ids),
            "window_ids_sha256": canonical_json_sha256(sorted(expected_ids)),
            "ordered_window_ids_sha256": canonical_json_sha256(expected_ids),
            "shards": references,
        }
        return batch, lineage


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WithinGroupRunError(f"{name} must be an object")
    return value


def _json_object(path: str | Path, *, name: str) -> dict[str, Any]:
    return dict(_mapping(load_json_strict(path), name=name))


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise WithinGroupRunError(f"{name} must be a lowercase SHA-256")
    return value


def _self_hash(record: Mapping[str, Any], *, field: str, name: str) -> str:
    claimed = _sha256(record.get(field), name=f"{name} {field}")
    body = dict(record)
    body.pop(field, None)
    if claimed != canonical_json_sha256(body):
        raise WithinGroupRunError(f"{name} self-hash does not validate")
    return claimed


def _resolve_file(value: str | Path, *, root: Path, name: str) -> Path:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    if candidate.is_symlink():
        raise WithinGroupRunError(f"{name} may not be a symlink")
    path = candidate.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise WithinGroupRunError(f"{name} escapes artifact_root") from exc
    if path.is_symlink() or not path.is_file():
        raise WithinGroupRunError(f"{name} must be a regular file")
    return path


def _record_file(value: Any, *, root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise WithinGroupRunError(f"{name} must be a portable relative path")
    return _resolve_file(value, root=root, name=name)


def _array_sha256(array: NDArray[Any], *, dtype: np.dtype[Any]) -> str:
    canonical = np.ascontiguousarray(array, dtype=dtype)
    digest = hashlib.sha256()
    digest.update(
        canonical_json_bytes(
            {
                "algorithm": "numpy-c-contiguous-little-endian-v1",
                "dtype": canonical.dtype.str,
                "shape": list(canonical.shape),
            }
        )
    )
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _functional_records_by_participant(
    split: Mapping[str, Any],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    windows = split.get("windows")
    if not isinstance(windows, list):
        raise WithinGroupRunError("parent split lacks windows")
    grouped: dict[str, list[Mapping[str, Any]]] = {
        participant: [] for participant in TARGET_PARTICIPANTS
    }
    all_ids: set[str] = set()
    for value in windows:
        if not isinstance(value, Mapping):
            raise WithinGroupRunError("parent split window is not an object")
        labels = value.get("canonical_labels")
        participant = value.get("subject_id")
        if (
            value.get("partition") != "target_sealed"
            or participant not in TARGET_PARTICIPANTS
            or not isinstance(labels, Mapping)
            or labels.get("functional_core") not in FUNCTIONAL_CORE_CLASS_ORDER
        ):
            continue
        window_id = value.get("window_id")
        if not isinstance(window_id, str) or not window_id or window_id in all_ids:
            raise WithinGroupRunError("target functional-core window identities are invalid")
        all_ids.add(window_id)
        grouped[str(participant)].append(value)
    result: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for participant, records in grouped.items():
        records.sort(key=lambda item: int(cast(int, item["start_row_inclusive"])))
        if not records:
            raise WithinGroupRunError(f"target participant {participant} has no windows")
        result[participant] = tuple(records)
    if len(all_ids) != 807:
        raise WithinGroupRunError("target functional-core universe is not 807 windows")
    return result


def preflight_participant_shard_store(
    *,
    record_path: str | Path,
    expected_record_file_sha256: str,
    split_manifest_path: str | Path,
    split: Mapping[str, Any],
    plan: Mapping[str, Any],
    context: ConsumedTargetContext,
    artifact_root: str | Path,
) -> ParticipantShardStore:
    """Validate every shard and sidecar without parsing any signal array."""

    root = Path(artifact_root).resolve(strict=True)
    path = _resolve_file(record_path, root=root, name="primary cache record")
    pinned_file_hash = _sha256(expected_record_file_sha256, name="primary cache record file hash")
    if sha256_file(path) != pinned_file_hash:
        raise WithinGroupRunError("primary cache record file hash changed")
    record = _json_object(path, name="primary cache record")
    record_sha256 = _self_hash(record, field="record_sha256", name="primary cache record")
    required = {
        "schema_version": PRIMARY_CACHE_SCHEMA_VERSION,
        "record_kind": PRIMARY_CACHE_RECORD_KIND,
        "status": "complete_create_only",
        "evidence_status": PRIMARY_CACHE_EVIDENCE_STATUS,
        "ontology_track": "functional_core",
        "class_names": list(FUNCTIONAL_CORE_CLASS_ORDER),
        "primary_channels": list(INCLUSIVEHAR_PRIMARY_CHANNELS),
        "target_cache_ordered_alignment_validated_against_opening_1": True,
        "unlock_api_called": False,
        "new_target_opening_created": False,
    }
    mismatches = [key for key, expected in required.items() if record.get(key) != expected]
    if mismatches:
        raise WithinGroupRunError(f"primary cache contract differs: {mismatches}")
    plan_split = _mapping(plan.get("split_manifest"), name="plan split")
    cache_split = _mapping(record.get("split_manifest"), name="cache split")
    supplied_split = _resolve_file(split_manifest_path, root=root, name="split manifest")
    referenced_split = _record_file(cache_split.get("path"), root=root, name="cache split")
    if (
        referenced_split != supplied_split
        or cache_split.get("record_sha256") != plan_split.get("record_sha256")
        or cache_split.get("file_sha256") != plan_split.get("file_sha256")
        or sha256_file(referenced_split) != plan_split.get("file_sha256")
    ):
        raise WithinGroupRunError("cache does not reference the exact manifest split")
    source_hash = _sha256(plan.get("source_artifact_sha256"), name="source artifact hash")
    raw_lineage = _mapping(record.get("raw_sensor_csv"), name="cache raw lineage")
    if raw_lineage.get("sha256") != source_hash:
        raise WithinGroupRunError("cache source-artifact lineage changed")
    opening = _mapping(record.get("opening_1"), name="cache opening lineage")
    plan_opening = _mapping(plan.get("opening_1"), name="plan opening lineage")
    opening_required = {
        "receipt_record_sha256": context.receipt["record_sha256"],
        "receipt_file_sha256": context.receipt_file_sha256,
        "index_record_sha256": context.index["record_sha256"],
        "index_file_sha256": context.index_file_sha256,
        "target_seal_id": context.index["target_seal_id"],
    }
    if any(
        opening.get(key) != value or plan_opening.get(key) != value
        for key, value in opening_required.items()
    ):
        raise WithinGroupRunError("cache/manifest differ from consumed opening 1")

    records_by_participant = _functional_records_by_participant(split)
    shard_values = _mapping(record.get("target_participant_shards"), name="participant shard index")
    if set(shard_values) != TARGET_PARTICIPANTS:
        raise WithinGroupRunError("participant shard subject set changed")
    shards: dict[str, ShardReference] = {}
    union_ids: set[str] = set()
    for participant in sorted(TARGET_PARTICIPANTS, key=int):
        value = _mapping(shard_values[participant], name=f"target shard {participant}")
        records = records_by_participant[participant]
        ids = [str(item["window_id"]) for item in records]
        count = len(ids)
        expected_hash = canonical_json_sha256(sorted(ids))
        array_path = _record_file(
            value.get("array_path"), root=root, name=f"target shard {participant} array"
        )
        metadata_path = _record_file(
            value.get("metadata_path"),
            root=root,
            name=f"target shard {participant} metadata",
        )
        array_hash = _sha256(value.get("array_sha256"), name="target shard array hash")
        metadata_hash = _sha256(value.get("metadata_sha256"), name="target shard metadata hash")
        if sha256_file(array_path) != array_hash or sha256_file(metadata_path) != metadata_hash:
            raise WithinGroupRunError(f"target shard {participant} file hash changed")
        metadata = _mapping(load_json_strict(metadata_path), name="target shard metadata")
        lineage = _mapping(metadata.get("lineage"), name="target shard lineage")
        required_reference = {
            "participant_id": participant,
            "partition": "target_sealed",
            "window_count": count,
            "window_ids_sha256": expected_hash,
        }
        lineage_required = {
            "cache_role": "few_person_target_participant_shard",
            "participant_id": participant,
            "partition": "target_sealed",
            "window_count": count,
            "window_ids_sha256": expected_hash,
            "split_manifest_sha256": plan_split["record_sha256"],
            "source_artifact_sha256": source_hash,
            "opening_receipt_record_sha256": context.receipt["record_sha256"],
            "opening_receipt_file_sha256": context.receipt_file_sha256,
            "locked_target_index_record_sha256": context.index["record_sha256"],
            "locked_target_index_file_sha256": context.index_file_sha256,
            "target_seal_id": context.index["target_seal_id"],
        }
        if (
            any(value.get(key) != expected for key, expected in required_reference.items())
            or any(lineage.get(key) != expected for key, expected in lineage_required.items())
            or metadata.get("array_sha256") != array_hash
            or metadata.get("window_count") != count
            or metadata.get("shape") != [count, 128, 6]
            or metadata.get("class_names") != list(FUNCTIONAL_CORE_CLASS_ORDER)
            or metadata.get("ontology_track") != "functional_core"
        ):
            raise WithinGroupRunError(f"target shard {participant} metadata changed")
        ordered_window_hash = _sha256(
            value.get("ordered_window_ids_sha256"), name="ordered window hash"
        )
        ordered_participant_hash = _sha256(
            value.get("ordered_participant_ids_sha256"), name="ordered participant hash"
        )
        ordered_label_hash = _sha256(value.get("ordered_labels_sha256"), name="ordered label hash")
        if (
            lineage.get("ordered_window_ids_sha256") != ordered_window_hash
            or lineage.get("ordered_participant_ids_sha256") != ordered_participant_hash
            or lineage.get("ordered_labels_sha256") != ordered_label_hash
        ):
            raise WithinGroupRunError(f"target shard {participant} ordered lineage changed")
        if union_ids & set(ids):
            raise WithinGroupRunError("participant shard split identities overlap")
        union_ids.update(ids)
        shards[participant] = ShardReference(
            participant_id=participant,
            window_count=count,
            window_ids_sha256=expected_hash,
            ordered_window_ids_sha256=ordered_window_hash,
            ordered_participant_ids_sha256=ordered_participant_hash,
            ordered_labels_sha256=ordered_label_hash,
            array_path=array_path,
            array_sha256=array_hash,
            metadata_path=metadata_path,
            metadata_sha256=metadata_hash,
        )
    audit = _mapping(record.get("target_participant_shard_audit"), name="shard audit")
    union_hash = canonical_json_sha256(sorted(union_ids))
    if (
        len(union_ids) != 807
        or audit.get("status") != "exact_disjoint_union_validated_against_opening_1"
        or audit.get("participant_ids") != sorted(TARGET_PARTICIPANTS, key=int)
        or audit.get("shard_count") != 10
        or audit.get("total_window_count") != 807
        or audit.get("union_window_ids_sha256") != union_hash
        or audit.get("pairwise_window_id_disjoint") is not True
        or audit.get("exact_whole_target_union") is not True
    ):
        raise WithinGroupRunError("participant shard union audit changed")
    return ParticipantShardStore(
        artifact_root=root,
        record_path=path,
        record_file_sha256=pinned_file_hash,
        record_sha256=record_sha256,
        split_manifest_sha256=str(plan_split["record_sha256"]),
        source_artifact_sha256=source_hash,
        opening_receipt_record_sha256=str(context.receipt["record_sha256"]),
        opening_index_record_sha256=str(context.index["record_sha256"]),
        shards=shards,
        split_records_by_participant=records_by_participant,
    )


def _load_plan_and_cell(
    *,
    manifest_path: Path,
    split_manifest_path: Path,
    opening_receipt_path: Path,
    locked_target_index_path: Path,
    final_freeze_inventory_path: Path,
    artifact_root: Path,
    fold_id: str,
    model_id: str,
    seed: int,
) -> tuple[
    dict[str, Any],
    str,
    dict[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    ConsumedTargetContext,
]:
    root = artifact_root.resolve(strict=True)
    manifest_file = _resolve_file(manifest_path, root=root, name="within-group manifest")
    plan = _json_object(manifest_file, name="within-group manifest")
    plan_sha256 = _self_hash(plan, field="manifest_sha256", name="within-group manifest")
    required = {
        "schema_version": WITHIN_GROUP_SCHEMA_VERSION,
        "manifest_kind": "postconfirmatory_disabled_within_group_cross_subject",
        "protocol_id": WITHIN_GROUP_PROTOCOL_ID,
        "status": "ready_no_cell_run",
        "evidence_status": WITHIN_GROUP_EVIDENCE_STATUS,
        "scientific_role": "secondary_descriptive_not_locked_confirmatory",
        "model_order": list(SUPPORTED_MODEL_IDS),
        "seed_order": list(REQUIRED_SEEDS),
        "expected_cell_count": 75,
        "raw_dataset_file_accessed_during_manifest_build": False,
        "target_signals_accessed_during_manifest_build": False,
        "target_predictions_or_performance_used_for_fold_or_model_design": False,
        "new_target_opening_created": False,
    }
    mismatch = [key for key, expected in required.items() if plan.get(key) != expected]
    if mismatch:
        raise WithinGroupRunError(f"within-group manifest contract differs: {mismatch}")
    supplied = {
        "split_manifest": _resolve_file(
            split_manifest_path, root=root, name="parent split manifest"
        ),
        "opening_receipt": _resolve_file(opening_receipt_path, root=root, name="opening receipt"),
        "locked_target_index": _resolve_file(
            locked_target_index_path, root=root, name="locked target index"
        ),
        "final_freeze_inventory": _resolve_file(
            final_freeze_inventory_path, root=root, name="final freeze inventory"
        ),
    }
    split_reference = _mapping(plan.get("split_manifest"), name="manifest split")
    opening_reference = _mapping(plan.get("opening_1"), name="manifest opening")
    freeze_reference = _mapping(
        plan.get("final_freeze_inventory"), name="manifest freeze inventory"
    )
    expected_file_hashes = {
        "split_manifest": split_reference.get("file_sha256"),
        "opening_receipt": opening_reference.get("receipt_file_sha256"),
        "locked_target_index": opening_reference.get("index_file_sha256"),
        "final_freeze_inventory": freeze_reference.get("file_sha256"),
    }
    for name, path in supplied.items():
        if sha256_file(path) != expected_file_hashes[name]:
            raise WithinGroupRunError(f"{name} file differs from the manifest")
    split = _json_object(supplied["split_manifest"], name="parent split")
    if _self_hash(split, field="split_manifest_sha256", name="parent split") != (
        split_reference.get("record_sha256")
    ):
        raise WithinGroupRunError("parent split record differs from manifest")
    context = load_consumed_target_context(
        opening_receipt_path=supplied["opening_receipt"],
        locked_target_index_path=supplied["locked_target_index"],
        artifact_root=root,
    )
    if (
        context.receipt.get("record_sha256") != opening_reference.get("receipt_record_sha256")
        or context.index.get("record_sha256") != opening_reference.get("index_record_sha256")
        or context.index.get("split_manifest_sha256") != split_reference.get("record_sha256")
    ):
        raise WithinGroupRunError("consumed opening context differs from manifest")
    inventory = _json_object(supplied["final_freeze_inventory"], name="freeze inventory")
    if _self_hash(inventory, field="inventory_sha256", name="freeze inventory") != (
        freeze_reference.get("record_sha256")
    ):
        raise WithinGroupRunError("freeze inventory differs from manifest")
    folds = plan.get("folds")
    models = plan.get("models")
    if (
        not isinstance(folds, list)
        or len(folds) != 5
        or not isinstance(models, list)
        or len(models) != len(SUPPORTED_MODEL_IDS) * len(REQUIRED_SEEDS)
    ):
        raise WithinGroupRunError("manifest lacks folds/models")
    fold_ids = [_mapping(value, name="fold").get("fold_id") for value in folds]
    model_keys = [
        (
            _mapping(value, name="model entry").get("model_id"),
            _mapping(value, name="model entry").get("seed"),
        )
        for value in models
    ]
    expected_model_keys = {
        (declared_model, declared_seed)
        for declared_model in SUPPORTED_MODEL_IDS
        for declared_seed in REQUIRED_SEEDS
    }
    if (
        len(set(fold_ids)) != len(fold_ids)
        or set(model_keys) != expected_model_keys
        or len(set(model_keys)) != len(model_keys)
    ):
        raise WithinGroupRunError("manifest fold/model inventory changed")
    fold = next(
        (
            _mapping(value, name="fold")
            for value in folds
            if isinstance(value, Mapping) and value.get("fold_id") == fold_id
        ),
        None,
    )
    model = next(
        (
            _mapping(value, name="model cell")
            for value in models
            if isinstance(value, Mapping)
            and value.get("model_id") == model_id
            and value.get("seed") == seed
        ),
        None,
    )
    if fold is None or model is None:
        raise WithinGroupRunError(f"cell {fold_id}/{model_id}/{seed} is not predeclared")
    role_sets: list[set[str]] = []
    for role, expected_count in (("training", 6), ("validation", 2), ("evaluation", 2)):
        role_record = _mapping(fold.get(role), name=f"fold {role}")
        participants = role_record.get("participant_ids")
        if (
            not isinstance(participants, list)
            or len(participants) != expected_count
            or any(not isinstance(value, str) for value in participants)
        ):
            raise WithinGroupRunError(f"fold {role} participant assignment changed")
        role_sets.append(set(cast(list[str], participants)))
    if (
        any(
            left & right for index, left in enumerate(role_sets) for right in role_sets[index + 1 :]
        )
        or set().union(*role_sets) != TARGET_PARTICIPANTS
    ):
        raise WithinGroupRunError("fold roles do not partition target participants")
    configuration = dict(
        _mapping(
            model.get("source_locked_training_configuration"),
            name="source-locked training configuration",
        )
    )
    if canonical_json_sha256(configuration) != model.get(
        "source_locked_training_configuration_sha256"
    ):
        raise WithinGroupRunError("source-locked configuration hash changed")
    return plan, plan_sha256, split, fold, model, context


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _rng_states() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def _device_environment(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "device_type": device.type,
        "device_name": properties.name,
        "device_total_memory_bytes": properties.total_memory,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_version": cast(Any, torch.backends.cudnn).version(),
    }


def require_cuda_device() -> torch.device:
    """Return the sole authorized neural device or fail without a CPU fallback."""

    if not torch.cuda.is_available():
        raise WithinGroupRunError(
            "within-group neural execution requires CUDA; CPU fallback is forbidden"
        )
    return torch.device("cuda")


def _write_torch_new(path: Path, payload: Mapping[str, Any]) -> str:
    with path.open("xb") as stream:
        torch.save(dict(payload), stream)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_file(path)


def _write_npz_new(path: Path, arrays: Mapping[str, NDArray[Any]]) -> str:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **cast(dict[str, Any], dict(arrays)))
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_file(path)


def _train_fixed_epoch(
    batch: MaterializedWindows,
    *,
    normalizer: ChannelStandardizer,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if config.checkpoint_selection_rule != "fixed_last_epoch":
        raise WithinGroupRunError("within-group training requires fixed-last-epoch selection")
    configure_determinism(config.seed)
    torch.backends.cudnn.enabled = not config.disable_cudnn
    normalized = normalizer.transform(batch.signals)
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(normalized)),
        torch.from_numpy(np.ascontiguousarray(batch.labels)),
    )
    generator = torch.Generator().manual_seed(config.seed + 1)
    loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        cast(Any, dataset),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
        pin_memory=True,
    )
    model = build_model(config).to(device)
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise WithinGroupRunError("model parameters are not resident on CUDA")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision == "float16")  # type: ignore[attr-defined]
    amp_dtype = torch.bfloat16 if config.mixed_precision == "bfloat16" else torch.float16
    amp_enabled = config.mixed_precision != "disabled"
    history: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        example_count = 0
        optimizer_updates = 0
        for signals, labels in loader:
            signals = signals.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if signals.device.type != "cuda" or labels.device.type != "cuda":
                raise WithinGroupRunError("neural batch is not resident on CUDA")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                output = cast(HAROutput, model(signals))
                loss = F.cross_entropy(output.logits, labels)
            cast(Any, scaler.scale(loss)).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            count = labels.numel()
            loss_sum += float(loss.detach()) * count
            example_count += count
            optimizer_updates += 1
        if example_count < 1 or optimizer_updates < 1:
            raise WithinGroupRunError("training epoch performed no optimizer update")
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "training_cross_entropy": loss_sum / example_count,
                "optimizer_update_count": optimizer_updates,
                "checkpoint_selected": epoch == config.epochs,
                "validation_accessed": False,
                "evaluation_accessed": False,
                "learning_rate_after_epoch": float(optimizer.param_groups[0]["lr"]),
            }
        )
    torch.cuda.synchronize(device)
    runtime = {
        "training_elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_training_performed": True,
        "cpu_neural_training_performed": False,
        "mixed_precision": config.mixed_precision,
        "cudnn_disabled_for_run": config.disable_cudnn,
    }
    states = {
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "scheduler_state": _cpu_tree(scheduler.state_dict()),
        "scaler_state": _cpu_tree(scaler.state_dict()),
        "rng_states": _cpu_tree(_rng_states()),
        "loader_generator_state": generator.get_state(),
    }
    return model, states, history, runtime


def _role_participants(fold: Mapping[str, Any], role: str) -> list[str]:
    value = _mapping(fold.get(role), name=f"fold {role}").get("participant_ids")
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise WithinGroupRunError(f"fold {role} participants changed")
    return cast(list[str], value)


def _validate_loaded_role(
    batch: MaterializedWindows, lineage: Mapping[str, Any], fold: Mapping[str, Any], role: str
) -> None:
    expected = _mapping(fold.get(role), name=f"fold {role}")
    if (
        lineage.get("participant_ids") != expected.get("participant_ids")
        or lineage.get("window_count") != expected.get("window_count")
        or lineage.get("window_ids_sha256") != expected.get("window_ids_sha256")
        or lineage.get("ordered_window_ids_sha256") != expected.get("ordered_window_ids_sha256")
        or set(batch.participant_ids) != set(cast(list[str], expected["participant_ids"]))
    ):
        raise WithinGroupRunError(f"loaded {role} data differ from the manifest")


def _portable(path: Path, root: Path) -> str:
    return path.resolve(strict=True).relative_to(root).as_posix()


def _failure_record(
    *,
    directory: Path,
    root: Path,
    plan_sha256: str,
    fold_id: str,
    model_id: str,
    seed: int,
    created_at_utc: str,
    code_commit: str,
    exc: BaseException,
) -> None:
    artifacts = []
    for path in sorted(directory.iterdir(), key=lambda value: value.name):
        if path.is_file() and not path.is_symlink() and path.name != "failure.json":
            artifacts.append(
                {
                    "path": _portable(path, root),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    failure: dict[str, Any] = {
        "schema_version": WITHIN_GROUP_SCHEMA_VERSION,
        "record_kind": "within_group_cell_failure",
        "status": "failed_preserved_create_only",
        "evidence_status": "failed_within_group_cell_not_result_evidence",
        "protocol_id": WITHIN_GROUP_PROTOCOL_ID,
        "manifest_sha256": plan_sha256,
        "fold_id": fold_id,
        "model_id": model_id,
        "seed": seed,
        "created_at_utc": created_at_utc,
        "code_commit": code_commit,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "preserved_partial_artifacts": artifacts,
        "raw_dataset_file_accessed": False,
        "new_target_opening_created": False,
        "cpu_neural_fallback_used": False,
    }
    failure["record_sha256"] = canonical_json_sha256(failure)
    atomic_write_json_new(failure, directory / "failure.json", allowed_root=directory)


def run_within_group_cell(
    *,
    manifest_path: Path,
    split_manifest_path: Path,
    opening_receipt_path: Path,
    locked_target_index_path: Path,
    final_freeze_inventory_path: Path,
    primary_cache_record_path: Path,
    primary_cache_record_file_sha256: str,
    artifact_root: Path,
    output_root: Path,
    fold_id: str,
    model_id: str,
    seed: int,
    code_commit: str,
    created_at_utc: str,
) -> dict[str, Any]:
    """Train and evaluate exactly one predeclared fold/model/seed cell."""

    root = artifact_root.resolve(strict=True)
    resolved_output_root = output_root.resolve(strict=True)
    try:
        resolved_output_root.relative_to(root)
    except ValueError as exc:
        raise WithinGroupRunError("output root escapes artifact root") from exc
    commit = code_commit.casefold()
    if _COMMIT_RE.fullmatch(commit) is None:
        raise WithinGroupRunError("code_commit must be a full Git object ID")
    if _UTC_RE.fullmatch(created_at_utc) is None:
        raise WithinGroupRunError("created_at_utc must be an explicit UTC Z timestamp")
    plan, plan_sha256, split, fold, model_entry, context = _load_plan_and_cell(
        manifest_path=manifest_path,
        split_manifest_path=split_manifest_path,
        opening_receipt_path=opening_receipt_path,
        locked_target_index_path=locked_target_index_path,
        final_freeze_inventory_path=final_freeze_inventory_path,
        artifact_root=root,
        fold_id=fold_id,
        model_id=model_id,
        seed=seed,
    )
    cell_directory = (resolved_output_root / fold_id / model_id / f"seed-{seed}").resolve(
        strict=False
    )
    try:
        cell_directory.relative_to(resolved_output_root)
    except ValueError as exc:
        raise WithinGroupRunError("derived cell path escapes output root") from exc
    if os.path.lexists(cell_directory):
        raise FileExistsError(f"refusing to overwrite cell directory: {cell_directory}")
    cell_directory.mkdir(parents=True, exist_ok=False)
    original_cudnn = torch.backends.cudnn.enabled
    try:
        device = require_cuda_device()
        store = preflight_participant_shard_store(
            record_path=primary_cache_record_path,
            expected_record_file_sha256=primary_cache_record_file_sha256,
            split_manifest_path=split_manifest_path,
            split=split,
            plan=plan,
            context=context,
            artifact_root=root,
        )
        configuration = dict(
            _mapping(
                model_entry["source_locked_training_configuration"],
                name="training configuration",
            )
        )
        config = training_config_from_dict(configuration)
        if config.seed != seed or config.model_name == "dann_compact_residual_96":
            raise WithinGroupRunError("cell training configuration identity changed")
        training_participants = _role_participants(fold, "training")
        validation_participants = _role_participants(fold, "validation")
        evaluation_participants = _role_participants(fold, "evaluation")

        training_batch, training_input = store.load_role(training_participants, role="training")
        _validate_loaded_role(training_batch, training_input, fold, "training")
        normalizer = ChannelStandardizer.fit(
            training_batch.signals,
            list(training_batch.participant_ids),
            declared_training_participants=set(training_participants),
            split_manifest_sha256=store.split_manifest_sha256,
            channel_names=INCLUSIVEHAR_PRIMARY_CHANNELS,
            minimum_scale=1e-8,
        )
        model, states, history, runtime = _train_fixed_epoch(
            training_batch,
            normalizer=normalizer,
            config=config,
            device=device,
        )
        checkpoint_path = cell_directory / "selected.pt"
        checkpoint_payload = {
            "schema_version": WITHIN_GROUP_SCHEMA_VERSION,
            "record_kind": "within_group_fixed_epoch_checkpoint",
            "status": "frozen_before_validation_or_evaluation_loading",
            "evidence_status": WITHIN_GROUP_EVIDENCE_STATUS,
            "protocol_id": WITHIN_GROUP_PROTOCOL_ID,
            "fold_id": fold_id,
            "model_id": model_id,
            "seed": seed,
            "epoch": config.epochs,
            "selected_epoch": config.epochs,
            "checkpoint_selection_rule": "fixed_last_epoch",
            "model_state": _cpu_tree(model.state_dict()),
            **states,
            "normalization": normalizer.to_dict(),
            "label_schema": list(FUNCTIONAL_CORE_CLASS_ORDER),
            "configuration": asdict(config),
            "configuration_sha256": canonical_json_sha256(asdict(config)),
            "manifest_sha256": plan_sha256,
            "split_manifest_sha256": store.split_manifest_sha256,
            "source_artifact_sha256": store.source_artifact_sha256,
            "primary_cache_record_sha256": store.record_sha256,
            "primary_cache_record_file_sha256": store.record_file_sha256,
            "training_input": training_input,
            "training_participants": training_participants,
            "validation_participants_declared_not_loaded": validation_participants,
            "evaluation_participants_declared_not_loaded": evaluation_participants,
            "history": history,
            "parameter_count": trainable_parameter_count(model),
            "runtime": runtime,
            "environment": _device_environment(device),
            "validation_arrays_parsed_before_checkpoint_freeze": False,
            "evaluation_arrays_parsed_before_checkpoint_freeze": False,
            "all_shard_bytes_hashed_before_training_for_integrity_only": True,
            "code_commit": commit,
            "created_at_utc": created_at_utc,
        }
        checkpoint_sha256 = _write_torch_new(checkpoint_path, checkpoint_payload)

        validation_batch, validation_input = store.load_role(
            validation_participants, role="validation"
        )
        _validate_loaded_role(validation_batch, validation_input, fold, "validation")
        validation_normalized = normalizer.transform(validation_batch.signals)
        validation_logits, _, validation_report_uncalibrated = predict_model(
            model,
            validation_normalized,
            validation_batch.labels,
            list(validation_batch.participant_ids),
            class_names=FUNCTIONAL_CORE_CLASS_ORDER,
            batch_size=config.batch_size,
            device=device,
            mixed_precision=config.mixed_precision,
            zero_channel_indices=config.zero_channel_indices,
        )
        validation_window_hash = canonical_json_sha256(sorted(validation_batch.window_ids))
        calibrator = fit_temperature(
            validation_logits,
            validation_batch.labels,
            validation_split_sha256=validation_window_hash,
        )
        calibrator_record: dict[str, Any] = {
            "schema_version": WITHIN_GROUP_SCHEMA_VERSION,
            "record_kind": "within_group_validation_temperature_calibrator",
            "status": "frozen_before_evaluation_loading",
            "evidence_status": WITHIN_GROUP_EVIDENCE_STATUS,
            "protocol_id": WITHIN_GROUP_PROTOCOL_ID,
            "fold_id": fold_id,
            "model_id": model_id,
            "seed": seed,
            "method": "scalar_temperature",
            "fit_role": "validation",
            "fit_participants": validation_participants,
            "fit_window_count": validation_batch.labels.size,
            "fit_window_ids_sha256": validation_window_hash,
            "fit_ordered_window_ids_sha256": canonical_json_sha256(
                list(validation_batch.window_ids)
            ),
            "fit_logits_sha256": _array_sha256(validation_logits, dtype=np.dtype("<f8")),
            "fit_labels_sha256": _array_sha256(validation_batch.labels, dtype=np.dtype("<i8")),
            "temperature": calibrator.temperature,
            "nll_before": calibrator.nll_before,
            "nll_after": calibrator.nll_after,
            "checkpoint_sha256": checkpoint_sha256,
            "configuration_sha256": canonical_json_sha256(asdict(config)),
            "manifest_sha256": plan_sha256,
            "split_manifest_sha256": store.split_manifest_sha256,
            "primary_cache_record_sha256": store.record_sha256,
            "validation_input": validation_input,
            "validation_uncalibrated_report": validation_report_uncalibrated,
            "evaluation_participants_declared_not_loaded": evaluation_participants,
            "evaluation_arrays_parsed_before_calibrator_freeze": False,
            "checkpoint_selection_used_validation": False,
            "target_validation_used_for_calibration": True,
            "code_commit": commit,
            "created_at_utc": created_at_utc,
        }
        calibrator_record["record_sha256"] = canonical_json_sha256(calibrator_record)
        calibrator_path = atomic_write_json_new(
            calibrator_record, cell_directory / "calibrator.json", allowed_root=cell_directory
        )
        calibrator_file_sha256 = sha256_file(calibrator_path)
        if (
            sha256_file(checkpoint_path) != checkpoint_sha256
            or sha256_file(calibrator_path) != calibrator_file_sha256
        ):
            raise WithinGroupRunError("checkpoint/calibrator barrier file changed")

        evaluation_batch, evaluation_input = store.load_role(
            evaluation_participants, role="evaluation"
        )
        _validate_loaded_role(evaluation_batch, evaluation_input, fold, "evaluation")
        evaluation_normalized = normalizer.transform(evaluation_batch.signals)
        logits, probabilities, uncalibrated_report = predict_model(
            model,
            evaluation_normalized,
            evaluation_batch.labels,
            list(evaluation_batch.participant_ids),
            class_names=FUNCTIONAL_CORE_CLASS_ORDER,
            batch_size=config.batch_size,
            device=device,
            mixed_precision=config.mixed_precision,
            zero_channel_indices=config.zero_channel_indices,
        )
        calibrated_probabilities = calibrator.probabilities(logits)
        calibrated_report = classification_report(
            evaluation_batch.labels,
            calibrated_probabilities,
            evaluation_batch.participant_ids,
            class_names=FUNCTIONAL_CORE_CLASS_ORDER,
        )
        predictions_path = cell_directory / "predictions.npz"
        predictions_sha256 = _write_npz_new(
            predictions_path,
            {
                "evidence_status": np.asarray([WITHIN_GROUP_EVIDENCE_STATUS]),
                "window_ids": np.asarray(evaluation_batch.window_ids),
                "participant_ids": np.asarray(evaluation_batch.participant_ids),
                "true_labels": evaluation_batch.labels,
                "logits": logits,
                "uncalibrated_probabilities": probabilities,
                "calibrated_probabilities": calibrated_probabilities,
                "predicted_labels": calibrated_probabilities.argmax(axis=1).astype(np.int64),
            },
        )
        result: dict[str, Any] = {
            "schema_version": WITHIN_GROUP_SCHEMA_VERSION,
            "record_kind": "within_group_fold_model_seed_result",
            "status": "complete_create_only",
            "evidence_status": WITHIN_GROUP_EVIDENCE_STATUS,
            "scientific_role": "secondary_descriptive_not_locked_confirmatory",
            "protocol_id": WITHIN_GROUP_PROTOCOL_ID,
            "fold_id": fold_id,
            "model_id": model_id,
            "seed": seed,
            "created_at_utc": created_at_utc,
            "code_commit": commit,
            "manifest_sha256": plan_sha256,
            "split_manifest_sha256": store.split_manifest_sha256,
            "source_artifact_sha256": store.source_artifact_sha256,
            "opening_receipt_record_sha256": store.opening_receipt_record_sha256,
            "opening_index_record_sha256": store.opening_index_record_sha256,
            "primary_cache": {
                "record_path": _portable(store.record_path, root),
                "record_sha256": store.record_sha256,
                "record_file_sha256": store.record_file_sha256,
            },
            "class_names": list(FUNCTIONAL_CORE_CLASS_ORDER),
            "configuration": asdict(config),
            "configuration_sha256": canonical_json_sha256(asdict(config)),
            "fold_roles": {
                "training": training_input,
                "validation": validation_input,
                "evaluation": evaluation_input,
            },
            "normalization": normalizer.to_dict(),
            "checkpoint": {
                "path": _portable(checkpoint_path, root),
                "sha256": checkpoint_sha256,
                "size_bytes": checkpoint_path.stat().st_size,
                "selection_rule": "fixed_last_epoch",
                "selected_epoch": config.epochs,
            },
            "calibrator": {
                "path": _portable(calibrator_path, root),
                "file_sha256": calibrator_file_sha256,
                "record_sha256": calibrator_record["record_sha256"],
                "temperature": calibrator.temperature,
            },
            "prediction_array": {
                "path": _portable(predictions_path, root),
                "sha256": predictions_sha256,
                "size_bytes": predictions_path.stat().st_size,
                "format": "npz",
            },
            "participant_level_report": calibrated_report,
            "uncalibrated_report": uncalibrated_report,
            "runtime": runtime,
            "parameter_count": trainable_parameter_count(model),
            "cuda_execution": _device_environment(device),
            "participant_assignment_before_array_loading": True,
            "normalization_fit_on_training_only": True,
            "checkpoint_frozen_before_validation_loading": True,
            "calibration_fit_on_validation_only": True,
            "checkpoint_and_calibrator_frozen_before_evaluation_loading": True,
            "evaluation_used_for_training_selection_or_calibration": False,
            "disability_or_assistive_device_metadata_used_as_model_input": False,
            "raw_dataset_file_accessed": False,
            "whole_target_cache_array_accessed": False,
            "opening_1_prediction_array_accessed": False,
            "new_target_opening_created": False,
            "cpu_neural_fallback_used": False,
        }
        result["record_sha256"] = canonical_json_sha256(result)
        atomic_write_json_new(result, cell_directory / "result.json", allowed_root=cell_directory)
        return result
    except Exception as exc:
        _failure_record(
            directory=cell_directory,
            root=root,
            plan_sha256=plan_sha256,
            fold_id=fold_id,
            model_id=model_id,
            seed=seed,
            created_at_utc=created_at_utc,
            code_commit=commit,
            exc=exc,
        )
        raise WithinGroupRunError(
            f"within-group cell failed; evidence preserved at {cell_directory / 'failure.json'}"
        ) from exc
    finally:
        torch.backends.cudnn.enabled = original_cudnn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--opening-receipt", type=Path, required=True)
    parser.add_argument("--locked-target-index", type=Path, required=True)
    parser.add_argument("--final-freeze-inventory", type=Path, required=True)
    parser.add_argument("--primary-cache-record", type=Path, required=True)
    parser.add_argument("--primary-cache-record-file-sha256", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--created-at-utc", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_within_group_cell(
            manifest_path=args.manifest,
            split_manifest_path=args.split_manifest,
            opening_receipt_path=args.opening_receipt,
            locked_target_index_path=args.locked_target_index,
            final_freeze_inventory_path=args.final_freeze_inventory,
            primary_cache_record_path=args.primary_cache_record,
            primary_cache_record_file_sha256=args.primary_cache_record_file_sha256,
            artifact_root=args.artifact_root,
            output_root=args.output_root,
            fold_id=args.fold_id,
            model_id=args.model_id,
            seed=args.seed,
            code_commit=args.code_commit,
            created_at_utc=args.created_at_utc,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "fail", "message": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "record_sha256": result["record_sha256"],
                "fold_id": result["fold_id"],
                "model_id": result["model_id"],
                "seed": result["seed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
