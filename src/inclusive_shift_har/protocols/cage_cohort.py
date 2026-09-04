"""Fail-closed manifest contract for future CAGE-HAR cohorts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from inclusive_shift_har.manifests.canonical import canonical_json_sha256

_CHANNELS = (
    "motionUserAccelerationX",
    "motionUserAccelerationY",
    "motionUserAccelerationZ",
    "motionRotationRateX",
    "motionRotationRateY",
    "motionRotationRateZ",
    "motionGravityX",
    "motionGravityY",
    "motionGravityZ",
)
_RECORD_KEYS = {
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
}
_MODEL_EXCLUSIONS = {
    "participant_id",
    "disability",
    "assistive_device",
    "monotonic_timestamp",
    "site_id",
    "session_id",
    "trial_id",
    "location",
}
_BOUNDARY_KEYS = {
    "includes_inclusivehar_participants_1_through_20",
    "includes_consumed_daghar_targets",
    "disability_or_identity_used_as_model_input",
    "partition_before_windowing",
    "windows_cross_trial_session_or_timestamp_discontinuity",
    "raw_evidence_preserved",
    "labels_or_predictions_opened_for_confirmatory_selection",
}


class CageCohortContractError(ValueError):
    """Raised when a future cohort lacks publication-critical provenance."""


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise CageCohortContractError(f"{name} must be an object with string keys")
    return cast(dict[str, Any], value)


def _unique_strings(value: object, *, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise CageCohortContractError(f"{name} must be a non-empty list of strings")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise CageCohortContractError(f"{name} must not contain duplicates")
    return result


def validate_cage_cohort_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate and return a future development or still-sealed cohort record."""

    required = {
        "schema_version",
        "record_kind",
        "status",
        "created_at_utc",
        "dataset_id",
        "cohort_role",
        "participant_count",
        "site_ids",
        "movement_realization_strata",
        "minimum_sessions_per_participant",
        "sensor_contract",
        "record_key_columns",
        "model_input_exclusions",
        "participants",
        "evidence_boundary",
        "record_sha256",
    }
    if set(record) != required:
        raise CageCohortContractError("CAGE cohort manifest keys changed")
    claimed_hash = record["record_sha256"]
    unhashed = dict(record)
    unhashed.pop("record_sha256")
    if not isinstance(claimed_hash, str) or claimed_hash != canonical_json_sha256(unhashed):
        raise CageCohortContractError("CAGE cohort manifest self-hash mismatch")
    if record["schema_version"] != "1.0.0" or record["record_kind"] != "cage_har_cohort":
        raise CageCohortContractError("CAGE cohort schema or record kind changed")
    if record["cohort_role"] not in {"new_development", "sealed_confirmatory"}:
        raise CageCohortContractError("CAGE cohort role must be new development or sealed")
    expected_status = {
        "new_development": "available_for_development",
        "sealed_confirmatory": "sealed_not_opened",
    }[str(record["cohort_role"])]
    if record["status"] != expected_status:
        raise CageCohortContractError("CAGE cohort status is inconsistent with its role")
    timestamp = record["created_at_utc"]
    if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
        raise CageCohortContractError("created_at_utc must be a UTC Z timestamp")
    try:
        datetime.fromisoformat(timestamp.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise CageCohortContractError("created_at_utc is invalid") from exc
    dataset_id = record["dataset_id"]
    if (
        not isinstance(dataset_id, str)
        or not dataset_id
        or dataset_id
        in {
            "inclusivehar_v4",
            "daghar_v2",
        }
    ):
        raise CageCohortContractError("dataset_id is empty or identifies consumed evidence")
    sites = _unique_strings(record["site_ids"], name="site_ids")
    strata = _unique_strings(
        record["movement_realization_strata"], name="movement_realization_strata"
    )
    minimum_sessions = record["minimum_sessions_per_participant"]
    if isinstance(minimum_sessions, bool) or not isinstance(minimum_sessions, int):
        raise CageCohortContractError("minimum sessions must be an integer")
    if minimum_sessions < 2:
        raise CageCohortContractError("CAGE cohort requires at least two sessions per participant")

    sensor = _mapping(record["sensor_contract"], name="sensor contract")
    if set(sensor) != {
        "channels",
        "units_by_channel",
        "coordinate_frame_by_modality",
        "native_gravity_recorded",
        "true_sampling_intervals_preserved",
    }:
        raise CageCohortContractError("sensor contract keys changed")
    if tuple(sensor["channels"]) != _CHANNELS:
        raise CageCohortContractError("sensor contract must contain the locked nine-channel order")
    units = _mapping(sensor["units_by_channel"], name="units by channel")
    if set(units) != set(_CHANNELS) or any(
        not isinstance(value, str) or not value for value in units.values()
    ):
        raise CageCohortContractError("every sensor channel requires an explicit unit")
    frames = _mapping(sensor["coordinate_frame_by_modality"], name="coordinate frames")
    if set(frames) != {"user_acceleration", "rotation_rate", "gravity"} or any(
        not isinstance(value, str) or not value for value in frames.values()
    ):
        raise CageCohortContractError("every modality requires an explicit coordinate frame")
    if (
        sensor["native_gravity_recorded"] is not True
        or sensor["true_sampling_intervals_preserved"] is not True
    ):
        raise CageCohortContractError("native gravity and true sampling intervals are required")

    record_keys = _mapping(record["record_key_columns"], name="record key columns")
    if set(record_keys) != _RECORD_KEYS or any(
        not isinstance(value, str) or not value for value in record_keys.values()
    ):
        raise CageCohortContractError("all required record-key columns must be declared")
    exclusions = set(_unique_strings(record["model_input_exclusions"], name="model exclusions"))
    if not _MODEL_EXCLUSIONS <= exclusions:
        raise CageCohortContractError("model input exclusions omit protected metadata")

    participant_count = record["participant_count"]
    raw_participants = record["participants"]
    if isinstance(participant_count, bool) or not isinstance(participant_count, int):
        raise CageCohortContractError("participant_count must be an integer")
    if not isinstance(raw_participants, list) or len(raw_participants) != participant_count:
        raise CageCohortContractError("participant records do not match participant_count")
    participant_ids: list[str] = []
    for index, raw_participant in enumerate(raw_participants):
        participant = _mapping(raw_participant, name=f"participants[{index}]")
        if set(participant) != {
            "participant_id",
            "site_id",
            "movement_realization_stratum",
            "session_ids",
        }:
            raise CageCohortContractError("participant record keys changed")
        participant_id = participant["participant_id"]
        if not isinstance(participant_id, str) or not participant_id:
            raise CageCohortContractError("participant_id must be a non-empty string")
        if participant_id in {str(item) for item in range(1, 21)}:
            raise CageCohortContractError("consumed InclusiveHAR participant id is forbidden")
        if participant["site_id"] not in sites:
            raise CageCohortContractError("participant site is not declared")
        if participant["movement_realization_stratum"] not in strata:
            raise CageCohortContractError("participant realization stratum is not declared")
        sessions = _unique_strings(
            participant["session_ids"], name=f"participants[{index}].session_ids"
        )
        if len(sessions) < minimum_sessions:
            raise CageCohortContractError("participant has too few distinct sessions")
        participant_ids.append(participant_id)
    if len(set(participant_ids)) != len(participant_ids):
        raise CageCohortContractError("participant identifiers must be unique")

    boundary = _mapping(record["evidence_boundary"], name="evidence boundary")
    if set(boundary) != _BOUNDARY_KEYS:
        raise CageCohortContractError("evidence boundary keys changed")
    expected_boundary = {
        "includes_inclusivehar_participants_1_through_20": False,
        "includes_consumed_daghar_targets": False,
        "disability_or_identity_used_as_model_input": False,
        "partition_before_windowing": True,
        "windows_cross_trial_session_or_timestamp_discontinuity": False,
        "raw_evidence_preserved": True,
        "labels_or_predictions_opened_for_confirmatory_selection": False,
    }
    if boundary != expected_boundary:
        raise CageCohortContractError("CAGE cohort evidence boundary does not fail closed")
    result = dict(record)
    result["validated_participant_ids"] = sorted(participant_ids)
    return result


def load_cage_cohort_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a future CAGE-HAR cohort manifest."""

    parsed = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_cage_cohort_record(_mapping(parsed, name="CAGE cohort manifest"))


__all__ = [
    "CageCohortContractError",
    "load_cage_cohort_manifest",
    "validate_cage_cohort_record",
]
