"""Synthetic aggregate-only split construction and leakage audits."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from inclusive_shift_har.manifests.canonical import canonical_json_sha256
from inclusive_shift_har.protocols.audit import audit_released_block_split_manifest
from inclusive_shift_har.protocols.splits import (
    build_released_block_split_manifest,
    build_source_window_manifest,
    write_split_manifest_new,
)

from ._protocol_synthetic import materialize_synthetic_protocol_inputs


def test_split_build_is_deterministic_and_participant_exclusive(
    tmp_path: Path, repository_root: Path
) -> None:
    inputs = materialize_synthetic_protocol_inputs(
        tmp_path / "inputs", repository_root=repository_root
    )
    first = build_released_block_split_manifest(**inputs)
    second = build_released_block_split_manifest(**inputs)

    assert first == second
    assert first["split_manifest_sha256"] == second["split_manifest_sha256"]
    assert first["window_count"] == 240
    assert first["remainder_accounting"] == {
        "dropped_tail_rows": 120,
        "policy": "drop_tail_shorter_than_128_per_released_block",
        "source_data_rows": 30_840,
        "used_window_rows": 30_720,
    }
    assignments = {
        subject: {
            window["partition"] for window in first["windows"] if window["subject_id"] == subject
        }
        for subject in [str(value) for value in range(1, 21)]
    }
    assert all(len(partitions) == 1 for partitions in assignments.values())
    report = audit_released_block_split_manifest(first, expected_manifest=second)
    assert report["valid"] is True
    assert report["status"] == "pass_conditional_released_block"
    assert len(first["source_nested_cv"]) == 5
    assert all(len(fold["inner_folds"]) == 4 for fold in first["source_nested_cv"])
    assert first["ontology"]["runnable_track_schemas"]["functional_core"] == {
        "class_count": 3,
        "class_order": ["mobility", "sitting", "standing"],
        "class_schema_sha256": "8fd23761b03ea39ebfc1fffc23430b31add0968fe6fc8522d035e7d0d9d704b4",
        "index_by_class": {"mobility": 0, "sitting": 1, "standing": 2},
        "track": "functional_core",
    }


def test_source_materialization_manifest_excludes_every_target_record(
    tmp_path: Path, repository_root: Path
) -> None:
    inputs = materialize_synthetic_protocol_inputs(
        tmp_path / "inputs", repository_root=repository_root
    )
    split = build_released_block_split_manifest(**inputs)
    source = build_source_window_manifest(split)

    assert source["source_window_count"] == 120
    assert source["target_subject_or_window_records_included"] is False
    assert source["target_performance_or_prediction_accessed"] is False
    assert {window["subject_id"] for window in source["windows"]} == {
        str(value) for value in range(1, 11)
    }
    assert {window["partition"] for window in source["windows"]} == {
        "source_train",
        "source_validation",
    }


def test_split_manifest_publication_is_byte_deterministic(
    tmp_path: Path, repository_root: Path
) -> None:
    inputs = materialize_synthetic_protocol_inputs(
        tmp_path / "inputs", repository_root=repository_root
    )
    manifest = build_released_block_split_manifest(**inputs)
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    first = write_split_manifest_new(manifest, output_root / "one.json", allowed_root=output_root)
    second = write_split_manifest_new(manifest, output_root / "two.json", allowed_root=output_root)

    assert first.read_bytes() == second.read_bytes()


def test_split_audit_detects_raw_overlap_even_with_recomputed_self_hash(
    tmp_path: Path, repository_root: Path
) -> None:
    inputs = materialize_synthetic_protocol_inputs(
        tmp_path / "inputs", repository_root=repository_root
    )
    manifest = build_released_block_split_manifest(**inputs)
    tampered = deepcopy(manifest)
    tampered["windows"][1]["start_row_inclusive"] = tampered["windows"][0]["start_row_inclusive"]
    tampered["windows"][1]["end_row_inclusive"] = tampered["windows"][0]["end_row_inclusive"]
    tampered.pop("split_manifest_sha256")
    tampered["split_manifest_sha256"] = canonical_json_sha256(tampered)

    report = audit_released_block_split_manifest(tampered)
    codes = {error["code"] for error in report["errors"]}
    assert report["valid"] is False
    assert "RAW_SAMPLE_OVERLAP" in codes
    assert "BLOCK_BOUNDARY_OR_ORDINAL" in codes


def test_split_audit_detects_subject_partition_overlap(
    tmp_path: Path, repository_root: Path
) -> None:
    inputs = materialize_synthetic_protocol_inputs(
        tmp_path / "inputs", repository_root=repository_root
    )
    tampered = build_released_block_split_manifest(**inputs)
    tampered = deepcopy(tampered)
    tampered["windows"][0]["partition"] = "source_validation"
    tampered.pop("split_manifest_sha256")
    tampered["split_manifest_sha256"] = canonical_json_sha256(tampered)

    report = audit_released_block_split_manifest(tampered)
    assert "SUBJECT_PARTITION_OVERLAP" in {error["code"] for error in report["errors"]}
