"""Checksum and read-access gate tests using tiny local bytes only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from inclusive_shift_har.data.audit import (
    DataAuditMode,
    DataGateError,
    audit_manifest_data,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256

from ._synthetic import materialize_manifest_payload, synthetic_dataset_manifest


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _matching_gate(manifest: dict[str, Any]) -> dict[str, str]:
    return {
        "schema_version": "1.0.0",
        "gate": "raw_data_read_access",
        "status": "approved",
        "dataset_id": manifest["dataset_id"],
        "manifest_sha256": canonical_json_sha256(manifest),
        "approved_at_utc": "2026-08-23T00:00:00Z",
    }


def test_dry_run_reads_no_raw_file(tmp_path: Path) -> None:
    manifest = synthetic_dataset_manifest()
    manifest_path = _write_json(tmp_path / "manifest.json", manifest)

    report = audit_manifest_data(manifest_path, mode=DataAuditMode.DRY_RUN)

    assert report.valid
    assert report.data_access == "none"
    assert report.observations[0].status == "planned_no_filesystem_access"
    assert "dry-run performed no filesystem" in report.warnings[0]


def test_read_only_audit_requires_matching_explicit_gate(tmp_path: Path) -> None:
    manifest = synthetic_dataset_manifest()
    manifest_path = _write_json(tmp_path / "manifest.json", manifest)
    data_root = tmp_path / "raw-root"
    data_root.mkdir()

    with pytest.raises(DataGateError, match="gate record"):
        audit_manifest_data(
            manifest_path,
            mode=DataAuditMode.READ_ONLY,
            data_root=data_root,
        )

    wrong_gate = _matching_gate(manifest)
    wrong_gate["manifest_sha256"] = "0" * 64
    gate_path = _write_json(tmp_path / "wrong-gate.json", wrong_gate)
    with pytest.raises(DataGateError, match="does not match"):
        audit_manifest_data(
            manifest_path,
            mode=DataAuditMode.READ_ONLY,
            data_root=data_root,
            gate_record_path=gate_path,
        )


def test_read_only_audit_verifies_size_and_sha256(tmp_path: Path) -> None:
    payload = b"tiny synthetic inertial fixture\n"
    manifest = synthetic_dataset_manifest(payload=payload)
    manifest_path = _write_json(tmp_path / "manifest.json", manifest)
    data_root = tmp_path / "raw-root"
    data_root.mkdir()
    materialize_manifest_payload(data_root, manifest, payload)
    gate_path = _write_json(tmp_path / "gate.json", _matching_gate(manifest))

    report = audit_manifest_data(
        manifest_path,
        mode=DataAuditMode.READ_ONLY,
        data_root=data_root,
        gate_record_path=gate_path,
    )

    assert report.valid, report.to_dict()
    observation = report.observations[0]
    assert observation.status == "pass"
    assert observation.observed_size_bytes == len(payload)
    assert observation.observed_sha256 == manifest["artifacts"][0]["expected_sha256"]
    assert report.to_dict()["report_sha256"] == report.to_dict()["report_sha256"]


def test_read_only_audit_detects_same_size_content_mutation(tmp_path: Path) -> None:
    expected = b"abcdefgh"
    corrupted = b"abcdEfgh"
    assert len(expected) == len(corrupted)
    manifest = synthetic_dataset_manifest(payload=expected)
    manifest_path = _write_json(tmp_path / "manifest.json", manifest)
    data_root = tmp_path / "raw-root"
    data_root.mkdir()
    materialize_manifest_payload(data_root, manifest, corrupted)
    gate_path = _write_json(tmp_path / "gate.json", _matching_gate(manifest))

    report = audit_manifest_data(
        manifest_path,
        mode="read-only",
        data_root=data_root,
        gate_record_path=gate_path,
    )

    assert not report.valid
    assert any("SHA-256 mismatch" in error for error in report.errors)
    assert not any("size mismatch" in error for error in report.errors)
