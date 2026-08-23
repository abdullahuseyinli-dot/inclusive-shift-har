"""Offline coverage for the full privacy-safe InclusiveHAR audit profile."""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

from inclusive_shift_har.data.inclusivehar import (
    INCLUSIVEHAR_EXPECTED_LABELS,
    INCLUSIVEHAR_V4_HEADERS,
    audit_inclusivehar_v4_dataset,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256

from ._synthetic import synthetic_dataset_manifest


def _synthetic_csv() -> bytes:
    text = StringIO(newline="")
    writer = csv.writer(text, lineterminator="\r\n")
    writer.writerow(INCLUSIVEHAR_V4_HEADERS)
    numeric_columns = len(INCLUSIVEHAR_V4_HEADERS) - 3
    for subject_id in range(1, 21):
        for label_index, label in enumerate(INCLUSIVEHAR_EXPECTED_LABELS):
            numeric = [f"{(subject_id * 0.001) + (label_index * 0.0001):.6f}"] * (numeric_columns)
            numeric[0] = "51.5074"  # private synthetic location sentinel
            writer.writerow([*numeric, label, subject_id, int(subject_id > 10)])
    return text.getvalue().encode("utf-8")


def _synthetic_docx() -> bytes:
    output = BytesIO()
    document = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>PRIVATE_SYNTHETIC_DESCRIPTION</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", document)
    return output.getvalue()


def test_full_audit_is_segment_aware_private_and_protocol_quarantined(
    tmp_path: Path,
) -> None:
    csv_payload = _synthetic_csv()
    docx_payload = _synthetic_docx()
    manifest: dict[str, Any] = synthetic_dataset_manifest(payload=csv_payload)
    manifest["expected_data"].update(
        {
            "reported_row_count": 121,
            "reported_sampling_rate_hz": 50,
        }
    )
    manifest["artifacts"] = [
        {
            "artifact_id": "inclusivehar_v4_sensor_csv",
            "role": "raw_sensor_table",
            "provider_filename": "sensor.csv",
            "download_url": "https://example.invalid/sensor.csv",
            "storage_path": "inclusivehar/v4/sensor.csv",
            "media_type": "text/csv",
            "expected_size_bytes": len(csv_payload),
            "expected_sha256": hashlib.sha256(csv_payload).hexdigest(),
            "hash_provenance": "provider_sha256",
            "git_policy": "exclude_raw_from_git",
        },
        {
            "artifact_id": "inclusivehar_v4_disability_description_docx",
            "role": "sensitive_participant_metadata",
            "provider_filename": "metadata.docx",
            "download_url": "https://example.invalid/metadata.docx",
            "storage_path": "inclusivehar/v4/metadata.docx",
            "media_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "expected_size_bytes": len(docx_payload),
            "expected_sha256": hashlib.sha256(docx_payload).hexdigest(),
            "hash_provenance": "provider_sha256",
            "git_policy": "exclude_raw_from_git",
        },
    ]

    raw_root = tmp_path / "raw"
    raw_directory = raw_root / "inclusivehar" / "v4"
    raw_directory.mkdir(parents=True)
    (raw_directory / "sensor.csv").write_bytes(csv_payload)
    (raw_directory / "metadata.docx").write_bytes(docx_payload)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha256 = canonical_json_sha256(manifest)
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "gate": "raw_data_read_access",
                "status": "approved",
                "dataset_id": "inclusivehar_v4",
                "manifest_sha256": manifest_sha256,
                "approved_at_utc": "2026-08-23T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    report = audit_inclusivehar_v4_dataset(
        manifest_path,
        data_root=raw_root,
        gate_record_path=gate_path,
        completed_at_utc="2026-08-23T00:01:00Z",
    )

    csv_audit = report["csv_audit"]
    assert report["status"] == "integrity_pass_protocol_quarantine"
    assert report["acceptance_gate"]["split_and_window_readiness"] == "blocked"
    assert csv_audit["row_structure"]["data_row_count"] == 120
    assert csv_audit["row_structure"]["headers"][-3:] == ["label", "UserID", "disabled"]
    assert csv_audit["released_order_and_boundaries"]["released_run_count"] == 120
    assert csv_audit["released_order_and_boundaries"][
        "one_contiguous_released_run_per_subject_activity"
    ]
    assert (
        csv_audit["released_order_and_boundaries"]["timestamp_gap_reset_monotonicity_status"]
        == "not_auditable_no_timestamp_column"
    )
    assert report["privacy"]["gps_coordinate_values_exported"] is False
    assert report["sensitive_metadata_audit"]["privacy_safe_manual_consistency_review"] == {
        "status": "not_applicable_nonofficial_fixture"
    }
    public_report = json.dumps(report, sort_keys=True)
    assert "51.5074" not in public_report
    assert "PRIVATE_SYNTHETIC_DESCRIPTION" not in public_report
