"""Tiny aggregate-only Stage 4 fixture; it contains no sensor measurements."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from inclusive_shift_har.manifests.canonical import (
    canonical_json_bytes,
    canonical_json_sha256,
    sha256_file,
)


def materialize_synthetic_protocol_inputs(
    root: Path,
    *,
    repository_root: Path,
) -> dict[str, Path]:
    """Create a 20-participant/6-label aggregate audit without raw signals."""

    root.mkdir(parents=True, exist_ok=True)
    labels = ["Ramp descent", "Ramp ascent", "Sitting", "Standing", "Walking", "jogging"]
    runs: list[dict[str, Any]] = []
    next_row = 1
    for subject in range(1, 21):
        for label in labels:
            runs.append(
                {
                    "activity_label": label,
                    "end_data_row_inclusive": next_row + 256,
                    "released_run_id": f"released_run_{len(runs) + 1:03d}",
                    "row_count": 257,
                    "start_data_row_inclusive": next_row,
                    "subject_id": str(subject),
                }
            )
            next_row += 257
    audit: dict[str, Any] = {
        "acquisition": {
            "artifact_verification": {
                "observations": [
                    {
                        "artifact_id": "inclusivehar_v4_sensor_csv",
                        "observed_sha256": "a" * 64,
                    }
                ]
            }
        },
        "csv_audit": {"released_order_and_boundaries": {"released_runs": runs}},
        "dataset_id": "inclusivehar_v4",
        "status": "integrity_pass_protocol_quarantine",
    }
    audit["report_sha256"] = canonical_json_sha256(audit)
    audit_path = root / "synthetic_stage3_audit.json"
    audit_path.write_bytes(canonical_json_bytes(audit) + b"\n")

    protocol = yaml.safe_load(
        (repository_root / "configs/protocols/inclusivehar_released_block_v1_2.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(protocol, dict)
    protocol["stage3_audit_report_sha256"] = audit["report_sha256"]
    protocol["stage3_audit_file_sha256"] = sha256_file(audit_path)
    protocol_path = root / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")

    authorization = json.loads(
        (
            repository_root / "results/protocol/released_block_deviation_authorization.json"
        ).read_text(encoding="utf-8")
    )
    assert isinstance(authorization, dict)
    authorization["stage3_audit_report_sha256"] = audit["report_sha256"]
    authorization_path = root / "authorization.json"
    authorization_path.write_bytes(canonical_json_bytes(authorization) + b"\n")
    return {
        "audit_report_path": audit_path,
        "authorization_path": authorization_path,
        "ontology_config_path": repository_root / "configs/ontologies/inclusivehar_v1_1.yaml",
        "preprocessing_config_path": repository_root
        / "configs/preprocessing/inclusivehar_primary_128.yaml",
        "protocol_config_path": protocol_path,
    }
