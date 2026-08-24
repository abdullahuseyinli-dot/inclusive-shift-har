from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inclusive_shift_har.data.inclusivehar import INCLUSIVEHAR_PRIMARY_CHANNELS
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.experiments.few_person import _select_cached_records
from inclusive_shift_har.experiments.postconfirmatory_cache import (
    EXPECTED_PARTICIPANTS,
    EXPECTED_PARTITION_COUNTS,
    FUNCTIONAL_CORE_CLASS_NAMES,
    PostconfirmatoryCacheError,
    _partition_records,
    build_parser,
    load_prepared_primary_cache,
    load_prepared_primary_participant_shard,
    prepare_postconfirmatory_primary_caches,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file


def _write_hashed(path: Path, payload: dict[str, Any], field: str) -> None:
    payload[field] = canonical_json_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    label_pairs = (
        ("Walking", "mobility"),
        ("Sitting", "sitting"),
        ("Standing", "standing"),
    )
    participants = {
        partition: sorted(values, key=int) for partition, values in EXPECTED_PARTICIPANTS.items()
    }
    rows = [",".join((*INCLUSIVEHAR_PRIMARY_CHANNELS, "label", "UserID")) + "\n"]
    windows: list[dict[str, Any]] = []
    row_index = 1
    partition_ordinal = 0
    for partition in ("source_train", "source_validation", "target_sealed"):
        subjects = participants[partition]
        for ordinal in range(EXPECTED_PARTITION_COUNTS[partition]):
            subject = subjects[ordinal % len(subjects)]
            released_label, canonical_label = label_pairs[ordinal % len(label_pairs)]
            start = row_index
            end = start + 127
            window_id = f"{partition}-window-{ordinal:04d}"
            windows.append(
                {
                    "window_id": window_id,
                    "raw_interval_id": f"raw-{partition_ordinal:04d}",
                    "released_run_id": f"run-{partition_ordinal:04d}",
                    "subject_id": subject,
                    "activity_label": released_label,
                    "canonical_labels": {"functional_core": canonical_label},
                    "partition": partition,
                    "window_ordinal": ordinal,
                    "start_row_inclusive": start,
                    "end_row_inclusive": end,
                    "length_samples": 128,
                    "stride_samples": 128,
                    "trial_id": None,
                    "trial_status": "released_block_surrogate",
                }
            )
            row = f"0,0,0,0,0,0,{released_label},{subject}\n"
            rows.extend([row] * 128)
            row_index = end + 1
            partition_ordinal += 1
    raw_path = tmp_path / "data" / "raw" / "inclusivehar.csv"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text("".join(rows), encoding="utf-8")

    preprocessing = {
        "schema_version": "1.0.0",
        "preprocessing_id": "inclusivehar-primary-six-128-v1",
        "channels": list(INCLUSIVEHAR_PRIMARY_CHANNELS),
    }
    preprocessing_path = tmp_path / "configs" / "preprocessing.json"
    preprocessing_path.parent.mkdir(parents=True)
    preprocessing_path.write_text(json.dumps(preprocessing), encoding="utf-8")
    functional_schema: dict[str, Any] = {
        "track": "functional_core",
        "class_count": 3,
        "class_order": list(FUNCTIONAL_CORE_CLASS_NAMES),
        "index_by_class": {name: index for index, name in enumerate(FUNCTIONAL_CORE_CLASS_NAMES)},
    }
    functional_schema["class_schema_sha256"] = canonical_json_sha256(functional_schema)
    seal_id = "b" * 64
    split: dict[str, Any] = {
        "schema_version": "1.0.0",
        "dataset_id": "inclusivehar_v4",
        "protocol": {"protocol_id": "inclusivehar-released-block-v1.2"},
        "subject_assignment_before_windowing": True,
        "target_performance_or_prediction_accessed": False,
        "window_length_samples": 128,
        "window_stride_samples": 128,
        "target_seal": {"seal_id": seal_id},
        "source_evidence": {"sensor_artifact_sha256": sha256_file(raw_path)},
        "preprocessing": {
            "preprocessing_id": "inclusivehar-primary-six-128-v1",
            "config_path": preprocessing_path.relative_to(tmp_path).as_posix(),
            "config_sha256": canonical_json_sha256(preprocessing),
        },
        "model_input_policy": {"allowed_channels_exact_order": list(INCLUSIVEHAR_PRIMARY_CHANNELS)},
        "ontology": {
            "config_sha256": "c" * 64,
            "runnable_track_schemas": {"functional_core": functional_schema},
        },
        "windows": windows,
    }
    split_path = tmp_path / "results" / "split.json"
    _write_hashed(split_path, split, "split_manifest_sha256")

    receipt: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "confirmatory_target_opening_receipt",
        "status": "unlock_consumed_before_materialization",
        "target_opening_number": 1,
        "split_manifest_sha256": split["split_manifest_sha256"],
        "target_seal_id": seal_id,
    }
    receipt_path = tmp_path / "results" / "opening.json"
    _write_hashed(receipt_path, receipt, "record_sha256")

    target_windows = [value for value in windows if value["partition"] == "target_sealed"]
    target_ids = tuple(str(value["window_id"]) for value in target_windows)
    target_participants = tuple(str(value["subject_id"]) for value in target_windows)
    class_to_index = {name: index for index, name in enumerate(FUNCTIONAL_CORE_CLASS_NAMES)}
    labels = np.asarray(
        [class_to_index[value["canonical_labels"]["functional_core"]] for value in target_windows],
        dtype=np.int64,
    )
    probabilities = np.full((labels.size, 3), 0.05, dtype=np.float64)
    probabilities[np.arange(labels.size), labels] = 0.9
    prediction_path = tmp_path / "results" / "target.predictions.npz"
    with prediction_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            evidence_status=np.asarray(["locked_confirmatory_target_opening_1"]),
            window_ids=np.asarray(target_ids),
            participant_ids=np.asarray(target_participants),
            true_labels=labels,
            logits=np.log(probabilities),
            uncalibrated_probabilities=probabilities,
            calibrated_probabilities=probabilities,
            predicted_labels=labels,
        )
    freeze_hash = "d" * 64
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "locked_target_per_seed_result",
        "status": "complete_create_only",
        "evidence_status": "locked_confirmatory_target_opening_1",
        "target_opening_number": 1,
        "model_id": "synthetic",
        "seed": 11,
        "class_names": list(FUNCTIONAL_CORE_CLASS_NAMES),
        "opening_receipt_record_sha256": receipt["record_sha256"],
        "final_freeze_inventory_sha256": freeze_hash,
        "prediction_array": {
            "path": prediction_path.relative_to(tmp_path).as_posix(),
            "sha256": sha256_file(prediction_path),
            "format": "npz",
        },
        "participant_level_report": classification_report(
            labels,
            probabilities,
            target_participants,
            class_names=FUNCTIONAL_CORE_CLASS_NAMES,
        ),
        "target_information_used_for_model_selection": False,
    }
    result_path = tmp_path / "results" / "target.result.json"
    _write_hashed(result_path, result, "record_sha256")
    index: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "locked_target_evaluation_index",
        "status": "complete_create_only",
        "evidence_status": "locked_confirmatory_target_opening_1",
        "target_opening_number": 1,
        "split_manifest_sha256": split["split_manifest_sha256"],
        "target_seal_id": seal_id,
        "final_freeze_inventory_sha256": freeze_hash,
        "opening_receipt_record_sha256": receipt["record_sha256"],
        "opening_receipt_file_sha256": sha256_file(receipt_path),
        "class_names": list(FUNCTIONAL_CORE_CLASS_NAMES),
        "window_count": labels.size,
        "participant_count": len(set(target_participants)),
        "model_seed_result_count": 1,
        "results": [
            {
                "model_id": "synthetic",
                "seed": 11,
                "array_path": prediction_path.relative_to(tmp_path).as_posix(),
                "array_sha256": sha256_file(prediction_path),
                "record_path": result_path.relative_to(tmp_path).as_posix(),
                "record_file_sha256": sha256_file(result_path),
                "record_sha256": result["record_sha256"],
            }
        ],
        "target_information_used_for_model_selection": False,
    }
    index_path = tmp_path / "results" / "index.json"
    _write_hashed(index_path, index, "record_sha256")
    cache_root = tmp_path / "data" / "cache"
    record_root = tmp_path / "results" / "cache"
    cache_root.mkdir(parents=True)
    record_root.mkdir(parents=True)
    return {
        "raw": raw_path,
        "split": split_path,
        "receipt": receipt_path,
        "index": index_path,
        "cache_root": cache_root,
        "record_root": record_root,
    }


def test_create_only_primary_caches_cover_all_three_exact_partitions(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    record = prepare_postconfirmatory_primary_caches(
        split_manifest_path=paths["split"],
        opening_receipt_path=paths["receipt"],
        locked_target_index_path=paths["index"],
        raw_csv_path=paths["raw"],
        artifact_root=tmp_path,
        cache_directory="primary-v1",
        cache_root=paths["cache_root"],
        record_output="primary-cache.json",
        record_root=paths["record_root"],
        code_commit="a" * 40,
        created_at_utc="2099-01-01T00:00:00Z",
    )
    record_path = paths["record_root"] / "primary-cache.json"
    record_file_hash = sha256_file(record_path)

    assert set(record["caches"]) == {
        "source_train",
        "source_validation",
        "target_sealed",
    }
    assert record["target_cache_ordered_alignment_validated_against_opening_1"] is True
    assert set(record["target_participant_shards"]) == EXPECTED_PARTICIPANTS["target_sealed"]
    assert record["target_participant_shard_audit"]["exact_whole_target_union"] is True
    assert record["unlock_api_called"] is False
    target_evidence = None
    for partition, count in EXPECTED_PARTITION_COUNTS.items():
        evidence = load_prepared_primary_cache(
            record_path=record_path,
            expected_record_file_sha256=record_file_hash,
            partition=partition,
            opening_receipt_path=paths["receipt"],
            locked_target_index_path=paths["index"],
            artifact_root=tmp_path,
            expected_split_manifest_sha256=record["split_manifest"]["record_sha256"],
            expected_source_artifact_sha256=record["raw_sensor_csv"]["sha256"],
        )
        assert evidence.cache.batch.signals.shape == (count, 128, 6)
        assert set(evidence.cache.batch.partitions) == {partition}
        assert set(evidence.cache.batch.participant_ids) == EXPECTED_PARTICIPANTS[partition]
        if partition == "target_sealed":
            target_evidence = evidence

    assert target_evidence is not None
    split = json.loads(paths["split"].read_text(encoding="utf-8"))
    target_records = _partition_records(split, "target_sealed")[:4]
    selected = _select_cached_records(
        target_evidence,
        target_records,
        class_names=FUNCTIONAL_CORE_CLASS_NAMES,
    )
    assert selected.window_ids == tuple(record.window_id for record in target_records)
    assert selected.signals.shape == (4, 128, 6)

    shard_union: set[str] = set()
    for participant in sorted(EXPECTED_PARTICIPANTS["target_sealed"], key=int):
        shard = load_prepared_primary_participant_shard(
            record_path=record_path,
            expected_record_file_sha256=record_file_hash,
            participant_id=participant,
            opening_receipt_path=paths["receipt"],
            locked_target_index_path=paths["index"],
            artifact_root=tmp_path,
            expected_split_manifest_sha256=record["split_manifest"]["record_sha256"],
            expected_source_artifact_sha256=record["raw_sensor_csv"]["sha256"],
        )
        assert set(shard.cache.batch.participant_ids) == {participant}
        shard_ids = set(shard.cache.batch.window_ids)
        assert not shard_union & shard_ids
        shard_union.update(shard_ids)
    assert len(shard_union) == EXPECTED_PARTITION_COUNTS["target_sealed"]
    assert (
        canonical_json_sha256(sorted(shard_union))
        == record["target_participant_shard_audit"]["union_window_ids_sha256"]
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_postconfirmatory_primary_caches(
            split_manifest_path=paths["split"],
            opening_receipt_path=paths["receipt"],
            locked_target_index_path=paths["index"],
            raw_csv_path=paths["raw"],
            artifact_root=tmp_path,
            cache_directory="primary-v1",
            cache_root=paths["cache_root"],
            record_output="primary-cache.json",
            record_root=paths["record_root"],
            code_commit="a" * 40,
            created_at_utc="2099-01-01T00:00:00Z",
        )


def test_cache_record_requires_external_file_hash_pin(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    record = prepare_postconfirmatory_primary_caches(
        split_manifest_path=paths["split"],
        opening_receipt_path=paths["receipt"],
        locked_target_index_path=paths["index"],
        raw_csv_path=paths["raw"],
        artifact_root=tmp_path,
        cache_directory="primary-v1",
        cache_root=paths["cache_root"],
        record_output="primary-cache.json",
        record_root=paths["record_root"],
        code_commit="a" * 40,
        created_at_utc="2099-01-01T00:00:00Z",
    )
    with pytest.raises(PostconfirmatoryCacheError, match="file hash changed"):
        load_prepared_primary_cache(
            record_path=paths["record_root"] / "primary-cache.json",
            expected_record_file_sha256="f" * 64,
            partition="target_sealed",
            opening_receipt_path=paths["receipt"],
            locked_target_index_path=paths["index"],
            artifact_root=tmp_path,
            expected_split_manifest_sha256=record["split_manifest"]["record_sha256"],
            expected_source_artifact_sha256=record["raw_sensor_csv"]["sha256"],
        )


def test_target_shard_index_mutation_fails_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    record = prepare_postconfirmatory_primary_caches(
        split_manifest_path=paths["split"],
        opening_receipt_path=paths["receipt"],
        locked_target_index_path=paths["index"],
        raw_csv_path=paths["raw"],
        artifact_root=tmp_path,
        cache_directory="primary-v1",
        cache_root=paths["cache_root"],
        record_output="primary-cache.json",
        record_root=paths["record_root"],
        code_commit="a" * 40,
        created_at_utc="2099-01-01T00:00:00Z",
    )
    record_path = paths["record_root"] / "primary-cache.json"
    mutated = json.loads(record_path.read_text(encoding="utf-8"))
    mutated["target_participant_shards"]["11"]["participant_id"] = "12"
    mutated.pop("record_sha256")
    mutated["record_sha256"] = canonical_json_sha256(mutated)
    record_path.write_text(json.dumps(mutated), encoding="utf-8")

    with pytest.raises(PostconfirmatoryCacheError, match="shard 11 contract changed"):
        load_prepared_primary_participant_shard(
            record_path=record_path,
            expected_record_file_sha256=sha256_file(record_path),
            participant_id="11",
            opening_receipt_path=paths["receipt"],
            locked_target_index_path=paths["index"],
            artifact_root=tmp_path,
            expected_split_manifest_sha256=record["split_manifest"]["record_sha256"],
            expected_source_artifact_sha256=record["raw_sensor_csv"]["sha256"],
        )


def test_partition_audit_rejects_missing_functional_window(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    split = json.loads(paths["split"].read_text(encoding="utf-8"))
    target = next(value for value in split["windows"] if value["partition"] == "source_validation")
    target["canonical_labels"] = {"inclusive_native": "walking"}
    with pytest.raises(PostconfirmatoryCacheError, match="exactly 143"):
        _partition_records(split, "source_validation")


def test_cache_preparer_exposes_no_unlock_or_new_opening_interface() -> None:
    help_text = build_parser().format_help().casefold()
    assert "--unlock-record" not in help_text
    assert "--opening-acknowledgement" not in help_text
    assert "--opening-receipt" in help_text
