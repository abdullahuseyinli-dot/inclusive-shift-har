from __future__ import annotations

from copy import deepcopy

import pytest

from inclusive_shift_har.manifests.canonical import canonical_json_sha256
from inclusive_shift_har.protocols.cage_cohort import (
    CageCohortContractError,
    validate_cage_cohort_record,
)


def _record() -> dict[str, object]:
    channels = [
        "motionUserAccelerationX",
        "motionUserAccelerationY",
        "motionUserAccelerationZ",
        "motionRotationRateX",
        "motionRotationRateY",
        "motionRotationRateZ",
        "motionGravityX",
        "motionGravityY",
        "motionGravityZ",
    ]
    record_keys = [
        "monotonic_timestamp",
        "participant_id",
        "site_id",
        "session_id",
        "trial_id",
        "repetition_id",
        "activity_label",
        "device_model",
        "operating_system",
        "app_version",
        "placement",
        "orientation",
        "sensor_accuracy",
        "sensor_provenance",
    ]
    record: dict[str, object] = {
        "schema_version": "1.0.0",
        "record_kind": "cage_har_cohort",
        "status": "available_for_development",
        "created_at_utc": "2026-09-04T12:00:00Z",
        "dataset_id": "new-cage-development-v1",
        "cohort_role": "new_development",
        "participant_count": 2,
        "site_ids": ["site-a"],
        "movement_realization_strata": ["unaided", "walking-aid"],
        "minimum_sessions_per_participant": 2,
        "sensor_contract": {
            "channels": channels,
            "units_by_channel": {channel: "declared-unit" for channel in channels},
            "coordinate_frame_by_modality": {
                "user_acceleration": "device-right-handed",
                "rotation_rate": "device-right-handed",
                "gravity": "device-right-handed",
            },
            "native_gravity_recorded": True,
            "true_sampling_intervals_preserved": True,
        },
        "record_key_columns": {key: key for key in record_keys},
        "model_input_exclusions": [
            "participant_id",
            "disability",
            "assistive_device",
            "monotonic_timestamp",
            "site_id",
            "session_id",
            "trial_id",
            "location",
        ],
        "participants": [
            {
                "participant_id": "new-001",
                "site_id": "site-a",
                "movement_realization_stratum": "unaided",
                "session_ids": ["day-1", "day-2"],
            },
            {
                "participant_id": "new-002",
                "site_id": "site-a",
                "movement_realization_stratum": "walking-aid",
                "session_ids": ["day-1", "day-2"],
            },
        ],
        "evidence_boundary": {
            "includes_inclusivehar_participants_1_through_20": False,
            "includes_consumed_daghar_targets": False,
            "disability_or_identity_used_as_model_input": False,
            "partition_before_windowing": True,
            "windows_cross_trial_session_or_timestamp_discontinuity": False,
            "raw_evidence_preserved": True,
            "labels_or_predictions_opened_for_confirmatory_selection": False,
        },
    }
    record["record_sha256"] = canonical_json_sha256(record)
    return record


def _rehash(record: dict[str, object]) -> None:
    record.pop("record_sha256", None)
    record["record_sha256"] = canonical_json_sha256(record)


def test_cage_cohort_contract_accepts_complete_new_development_record() -> None:
    validated = validate_cage_cohort_record(_record())
    assert validated["validated_participant_ids"] == ["new-001", "new-002"]
    assert validated["cohort_role"] == "new_development"


def test_cage_cohort_contract_rejects_consumed_participant() -> None:
    record = deepcopy(_record())
    participants = record["participants"]
    assert isinstance(participants, list)
    assert isinstance(participants[0], dict)
    participants[0]["participant_id"] = "10"
    _rehash(record)
    with pytest.raises(CageCohortContractError, match="consumed"):
        validate_cage_cohort_record(record)


def test_cage_cohort_contract_rejects_missing_second_session() -> None:
    record = deepcopy(_record())
    participants = record["participants"]
    assert isinstance(participants, list)
    assert isinstance(participants[0], dict)
    participants[0]["session_ids"] = ["day-1"]
    _rehash(record)
    with pytest.raises(CageCohortContractError, match="too few"):
        validate_cage_cohort_record(record)


def test_cage_cohort_contract_rejects_boundary_relaxation() -> None:
    record = deepcopy(_record())
    boundary = record["evidence_boundary"]
    assert isinstance(boundary, dict)
    boundary["raw_evidence_preserved"] = False
    _rehash(record)
    with pytest.raises(CageCohortContractError, match="fail closed"):
        validate_cage_cohort_record(record)


def test_cage_cohort_contract_rejects_stale_self_hash() -> None:
    record = _record()
    record["dataset_id"] = "changed-after-hash"
    with pytest.raises(CageCohortContractError, match="self-hash"):
        validate_cage_cohort_record(record)
