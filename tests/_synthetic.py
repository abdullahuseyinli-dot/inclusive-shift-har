"""Tiny manifest and artifact builders used only by tests."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

PRIMARY_SIX_CHANNELS = [
    "motionUserAccelerationX",
    "motionUserAccelerationY",
    "motionUserAccelerationZ",
    "motionRotationRateX",
    "motionRotationRateY",
    "motionRotationRateZ",
]

SENSITIVE_EXACT_NAMES = [
    "disabled",
    "userId",
    "Label",
    "label",
    "disability",
    "disability_status",
    "assistive_device",
    "participant_id",
    "activity_label",
    "timestamp",
]

CHECKPOINT_FILE_ROLES = [
    "model_state",
    "optimizer_state",
    "preprocessing",
    "normalization",
    "label_schema",
    "rng_states",
    "environment_metadata",
]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def synthetic_dataset_manifest(
    *,
    payload: bytes = b"tiny synthetic inertial fixture\n",
    storage_path: str = "data/raw/synthetic/v1/sensor.csv",
) -> dict[str, Any]:
    """Build the smallest valid dataset manifest accepted by Stage 2."""

    return {
        "schema_version": "1.0.0",
        "manifest_kind": "dataset",
        # Use the registered InclusiveHAR profile while keeping every byte synthetic.
        "dataset_id": "inclusivehar_v4",
        "title": "Synthetic HAR validation fixture",
        "record_status": "locked_starter",
        "release": {
            "version": "1.0",
            "doi": "10.0000/synthetic.not-a-publication",
            "landing_url": "https://example.invalid/synthetic-har",
            "published_date": "2026-01-01",
        },
        "provenance": {
            "official_repository_url": "https://example.invalid/synthetic-har",
            "metadata_endpoint": "https://example.invalid/synthetic-har/metadata.json",
            "metadata_verified_date": "2026-08-23",
        },
        "license": {
            "status": "clear",
            "notices": [
                {
                    "source": "synthetic fixture",
                    "text": "Generated test bytes; no third-party data.",
                }
            ],
            "redistribution_policy": "do_not_redistribute_raw",
        },
        "artifacts": [
            {
                "artifact_id": "synthetic_sensor_csv",
                "role": "raw_sensor_table",
                "provider_filename": "sensor.csv",
                "download_url": "https://example.invalid/synthetic-har/sensor.csv",
                "storage_path": storage_path,
                "media_type": "text/csv",
                "expected_size_bytes": len(payload),
                "expected_sha256": sha256_bytes(payload),
                "hash_provenance": "provider_sha256",
                "git_policy": "exclude_raw_from_git",
            }
        ],
        "expected_data": {
            "sample_rate_hz": 50,
            "window_length_samples": 128,
            "model_tensor_shape": ["batch", 128, 6],
        },
        "feature_policy": {
            "model_input_policy": "exact_allowlist_only",
            "primary_six_channel_allowlist": list(PRIMARY_SIX_CHANNELS),
            "sensitive_and_non_model_exclusions": {
                "exact_names": list(SENSITIVE_EXACT_NAMES),
                "casefold_prefixes": ["gps", "location", "latitude", "longitude"],
            },
        },
        "gate_policy": {
            "raw_data_read_requires": "raw_data_read_access_approved",
            "split_build_requires": "data_audit_pass",
            "training_requires": "split_audit_pass",
            "confirmatory_evaluation_requires": "final_evaluation_unlock",
        },
    }


def with_second_artifact(manifest: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(manifest)
    second = deepcopy(result["artifacts"][0])
    second.update(
        {
            "artifact_id": "synthetic_sensor_csv_2",
            "provider_filename": "sensor-2.csv",
            "storage_path": "data/raw/synthetic/v1/sensor-2.csv",
        }
    )
    result["artifacts"].append(second)
    return result


def materialize_manifest_payload(
    root: Path,
    manifest: dict[str, Any],
    payload: bytes = b"tiny synthetic inertial fixture\n",
) -> Path:
    storage_path = manifest["artifacts"][0]["storage_path"]
    target = root / Path(storage_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return target


def synthetic_checkpoint_manifest() -> tuple[dict[str, Any], dict[str, bytes]]:
    payloads = {role: f"synthetic {role}\n".encode() for role in CHECKPOINT_FILE_ROLES}
    files = [
        {
            "role": role,
            "path": f"run-001/{role}.bin",
            "size_bytes": len(payloads[role]),
            "sha256": sha256_bytes(payloads[role]),
        }
        for role in CHECKPOINT_FILE_ROLES
    ]
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "manifest_kind": "artifact",
        "artifact_id": "synthetic-checkpoint-run-001",
        "artifact_type": "checkpoint",
        "status": "complete",
        "evidence_status": "development_source_only",
        "created_at_utc": "2026-08-23T00:00:00Z",
        "provenance": {
            "code_commit": "1" * 40,
            "configuration_sha256": "2" * 64,
            "dataset_manifest_sha256": "3" * 64,
            "split_manifest_sha256": "4" * 64,
        },
        "files": files,
    }
    return manifest, payloads


def materialize_checkpoint_artifact(root: Path) -> tuple[Path, dict[str, Any]]:
    import json

    manifest, payloads = synthetic_checkpoint_manifest()
    run_root = root / "run-001"
    run_root.mkdir(parents=True, exist_ok=True)
    for role, payload in payloads.items():
        (run_root / f"{role}.bin").write_bytes(payload)
    manifest_path = run_root / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest_path, manifest
