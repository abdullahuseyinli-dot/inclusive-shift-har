from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inclusive_shift_har.data.inclusivehar import INCLUSIVEHAR_PRIMARY_CHANNELS
from inclusive_shift_har.data.materialize import (
    materialization_cache_key,
    materialize_inclusivehar_windows,
    save_materialized_cache_create_only,
)
from inclusive_shift_har.data.windowing import WindowRecord


def _write_synthetic_csv(path: Path) -> str:
    header = [*INCLUSIVEHAR_PRIMARY_CHANNELS, "label", "UserID"]
    lines = [",".join(header)]
    for row in range(256):
        label = "Walking" if row < 128 else "Sitting"
        subject = "1" if row < 128 else "2"
        lines.append(",".join([*(str(row + channel) for channel in range(6)), label, subject]))
    content = ("\n".join(lines) + "\n").encode()
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _record(start: int, *, subject: str, label: str, partition: str) -> WindowRecord:
    end = start + 127
    return WindowRecord(
        window_id=f"w-{start}",
        raw_interval_id=f"rows-{start}-{end}",
        released_run_id=f"run-{start}",
        subject_id=subject,
        activity_label=label,
        canonical_labels={"functional_core": "mobility" if label == "Walking" else "sitting"},
        partition=partition,
        window_ordinal=0,
        start_row_inclusive=start,
        end_row_inclusive=end,
        length_samples=128,
        stride_samples=128,
        trial_id=None,
        trial_status="unrecoverable",
    )


def test_materialization_reads_only_authorized_partition(tmp_path: Path) -> None:
    csv_path = tmp_path / "synthetic.csv"
    source_hash = _write_synthetic_csv(csv_path)
    batch = materialize_inclusivehar_windows(
        csv_path,
        [
            _record(1, subject="1", label="Walking", partition="source"),
            _record(129, subject="2", label="Sitting", partition="target_sealed"),
        ],
        expected_source_sha256=source_hash,
        ontology_track="functional_core",
        class_names=("mobility", "sitting", "standing"),
        allowed_partitions={"source"},
    )
    assert batch.signals.shape == (1, 128, 6)
    assert batch.participant_ids == ("1",)
    assert batch.labels.tolist() == [0]


def test_target_partition_requires_matching_one_time_unlock(tmp_path: Path) -> None:
    csv_path = tmp_path / "synthetic.csv"
    source_hash = _write_synthetic_csv(csv_path)
    target_record = _record(129, subject="2", label="Sitting", partition="target_sealed")
    with pytest.raises(PermissionError, match="unlock"):
        materialize_inclusivehar_windows(
            csv_path,
            [target_record],
            expected_source_sha256=source_hash,
            ontology_track="functional_core",
            class_names=("mobility", "sitting", "standing"),
            allowed_partitions={"target_sealed"},
            expected_target_seal_id="a" * 64,
            expected_split_manifest_sha256="b" * 64,
        )
    batch = materialize_inclusivehar_windows(
        csv_path,
        [target_record],
        expected_source_sha256=source_hash,
        ontology_track="functional_core",
        class_names=("mobility", "sitting", "standing"),
        allowed_partitions={"target_sealed"},
        expected_target_seal_id="a" * 64,
        expected_split_manifest_sha256="b" * 64,
        target_unlock={
            "approved_at_utc": "2099-01-01T00:00:00Z",
            "code_commit": "synthetic-test-commit",
            "final_gates": {
                "artifact_validation_passed": True,
                "configuration_validation_passed": True,
                "manifest_validation_passed": True,
                "protocol_lock_present": True,
                "split_audit_passed": True,
                "tests_passed": True,
                "type_checks_passed": True,
                "working_tree_clean_or_documented": True,
            },
            "gate": "final_evaluation_unlock",
            "protocol_lock_sha256": "c" * 64,
            "reason": "locked_confirmatory_evaluation",
            "schema_version": "1.0.0",
            "split_manifest_sha256": "b" * 64,
            "status": "approved",
            "target_opening_number": 1,
            "target_performance_previously_accessed": False,
            "target_seal_id": "a" * 64,
        },
    )
    assert batch.participant_ids == ("2",)


def test_cache_key_invalidates_on_every_declared_input() -> None:
    base: dict[str, Any] = {
        "source_artifact_sha256": "a" * 64,
        "preprocessing_config_sha256": "b" * 64,
        "split_manifest_sha256": "c" * 64,
        "code_version": "commit-1",
        "ontology_track": "functional_core",
        "partitions": ["source"],
    }
    first = materialization_cache_key(**base)
    for field, replacement in (
        ("source_artifact_sha256", "d" * 64),
        ("preprocessing_config_sha256", "e" * 64),
        ("split_manifest_sha256", "f" * 64),
        ("code_version", "commit-2"),
        ("ontology_track", "inclusive_native"),
        ("partitions", ["target_unlocked"]),
    ):
        changed = dict(base)
        changed[field] = replacement
        assert materialization_cache_key(**changed) != first


def test_cache_is_create_only_and_hashes_content(tmp_path: Path) -> None:
    csv_path = tmp_path / "synthetic.csv"
    source_hash = _write_synthetic_csv(csv_path)
    batch = materialize_inclusivehar_windows(
        csv_path,
        [_record(1, subject="1", label="Walking", partition="source")],
        expected_source_sha256=source_hash,
        ontology_track="functional_core",
        class_names=("mobility", "sitting", "standing"),
        allowed_partitions={"source"},
    )
    result = save_materialized_cache_create_only(
        batch,
        tmp_path / "cache",
        cache_key="1" * 64,
        lineage={"test": True},
    )
    assert len(result["array_sha256"]) == 64
    assert len(result["metadata_sha256"]) == 64
    with np.load(result["array_path"], allow_pickle=False) as payload:
        assert payload["signals"].shape == (1, 128, 6)
