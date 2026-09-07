"""Plan and evidence contracts for the physical information pilot.

The module creates a complete acquisition schedule and validates later collection
records. It does not communicate with a device, acquire human data, or provide
device-specific preprocessing defaults.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from inclusive_shift_har.manifests.canonical import (
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)

SUPPORT_CONDITIONS = (
    ("sitting", "quiet"),
    ("sitting", "upper_body_motion"),
    ("standing", "quiet"),
    ("standing", "upper_body_motion"),
)
QUERY_CONDITIONS = (
    ("sitting", "quiet"),
    ("sitting", "upper_body_motion"),
    ("standing", "quiet"),
    ("standing", "upper_body_motion"),
    ("mobility", "usual"),
    ("mobility", "slow"),
    ("mobility", "turning"),
)
PENDING_PREREQUISITES = (
    "human_study_authorization",
    "consent_process",
    "participant_access",
    "qualified_device",
    "placement_instruction",
    "bench_uncertainty",
    "device_specific_processing_freeze",
)
COLLECTION_STATUSES = {
    "planned",
    "recorded",
    "partial",
    "failed",
    "withdrawn",
    "unresolved",
}


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(value, str(path))


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject changes to the fixed study size, chronology, or decision limits."""

    population = _mapping(config.get("population"), "population")
    schedule = _mapping(config.get("schedule"), "schedule")
    comparisons = _mapping(config.get("comparison_contract"), "comparison_contract")
    recording = _mapping(config.get("recording_contract"), "recording_contract")
    analysis = _mapping(config.get("analysis_contract"), "analysis_contract")
    resources = _mapping(config.get("resources"), "resources")
    claims = _mapping(config.get("claims"), "claims")
    roster = [f"pilot:{index:03d}" for index in range(1, 7)]
    if config.get("schema_version") != "1.0.0":
        raise ValueError("unexpected config schema")
    if config.get("study_id") != "physical-information-identifiability-v1":
        raise ValueError("unexpected study id")
    if config.get("status") != "frozen_plan_device_contract_pending":
        raise ValueError("configuration is not frozen at the correct readiness state")
    if population.get("wearer_ids") != roster:
        raise ValueError("wearer roster changed")
    expected_population = {
        "planned_wearer_count": 6,
        "wearer_ids": roster,
        "visits_per_wearer": 2,
        "attachments_per_visit": 2,
        "replacement_recruitment_to_improve_outcomes": False,
        "independent_unit": "wearer",
    }
    if population != expected_population:
        raise ValueError("population contract changed")
    expected_schedule = {
        "support_precedes_query": True,
        "counterbalance": "cyclic_rotation_by_global_block_plus_wearer_index",
        "posture_duration_seconds": 30,
        "mobility_duration_seconds": 45,
        "interior_trim_seconds_each_end": 5,
        "support_conditions": [
            {"activity": activity, "motion": motion} for activity, motion in SUPPORT_CONDITIONS
        ],
        "query_conditions": [
            {"activity": activity, "motion": motion} for activity, motion in QUERY_CONDITIONS
        ],
        "expected_blocks": 24,
        "expected_bouts_per_block": 11,
        "expected_bouts": 264,
        "expected_nominal_recording_seconds": 9000,
        "appointment_cap_minutes_per_visit": 30,
    }
    if schedule != expected_schedule:
        raise ValueError("schedule contract changed")
    if (
        _mapping(comparisons.get("fresh"), "fresh").get("expected_query_blocks") != 24
        or _mapping(comparisons.get("attachment_stale"), "attachment stale").get(
            "expected_query_blocks"
        )
        != 12
        or _mapping(comparisons.get("visit_stale"), "visit stale").get("expected_query_blocks")
        != 12
        or comparisons.get("identical_query_ids_and_validity_masks_required") is not True
        or comparisons.get("future_support_forbidden") is not True
    ):
        raise ValueError("comparison contract changed")
    if recording.get("processing_freeze_point") != (
        "after_device_bench_qualification_before_first_human_outcome"
    ):
        raise ValueError("device-specific processing freeze changed")
    if recording.get("preprocessing_reset_boundaries") != [
        "attachment_change",
        "recording_break",
        "timestamp_discontinuity",
        "nonfinite_sensor_run",
    ]:
        raise ValueError("preprocessing reset boundaries changed")
    if recording.get("preprocessing_non_boundaries") != [
        "activity_annotation",
        "support_query_role",
        "bout_boundary",
        "scoring_admission",
    ]:
        raise ValueError("annotation-independent preprocessing contract changed")
    expected_analysis = {
        "aggregation_order": [
            "window_to_bout",
            "bout_to_condition_and_reference_scope",
            "condition_to_wearer",
            "equal_wearer_mean",
        ],
        "signal_projection": {
            "window_grid": "left_aligned_continuous_recording_no_bout_reset",
            "window_length_source": "equipment.processing_contract.window_samples",
            "window_stride_source": "equipment.processing_contract.window_stride_samples",
            "bout_admission": ("full_window_inside_fixed_five_second_trimmed_adjudicated_interval"),
            "gravity_direction": "normalize(arithmetic_mean(gravity_xyz_samples_in_window))",
            "numerical_norm_epsilon": 1.0e-12,
            "invalid_window_if": [
                "nonfinite_inertial_or_gravity_sample",
                "timestamp_gap_above_frozen_maximum",
                "gravity_mean_norm_at_or_below_epsilon",
            ],
        },
        "bout_eligibility": {
            "minimum_valid_window_count": 3,
            "minimum_valid_window_fraction": 0.80,
            "denominator": "all_grid_windows_fully_inside_fixed_trimmed_interval",
            "imputation_allowed": False,
            "quiet_interval_search_allowed": False,
        },
        "prototypes": {
            "posture_classes": ["sitting", "standing"],
            "motion_strata": ["quiet", "upper_body_motion"],
            "construction": ("normalize(arithmetic_mean(valid_support_window_gravity_directions))"),
            "support_scope": (
                "one_adjudicated_support_bout_per_posture_and_motion_stratum_per_reference_block"
            ),
            "invalid_if": [
                "support_bout_ineligible",
                "prototype_resultant_norm_at_or_below_epsilon",
            ],
        },
        "estimands": {
            "angular_distance": ("acos(clip(dot(unit_vector_a,unit_vector_b),-1,1))_radians"),
            "within_posture_angular_spread": [
                "median_window_to_prototype_angle",
                "p90_window_to_prototype_angle",
            ],
            "between_posture_angular_separation": (
                "angle(sitting_prototype,standing_prototype)_within_motion_stratum_and_reference_block"
            ),
            "signed_query_margin": (
                "wrong_posture_angle_minus_correct_posture_angle_per_valid_query_window_then_bout_median"
            ),
            "fresh_stale_difference": (
                "fresh_bout_median_signed_margin_minus_stale_bout_median_signed_margin_on_identical_query_windows"
            ),
            "anchor_angle": "between_posture_angular_separation",
            "inverse_absolute_sine_condition_number": (
                "1/abs(sin(anchor_angle)); infinity_when_denominator_at_or_below_epsilon"
            ),
            "dynamic_acceleration_rms": (
                "sqrt(mean(sum((acceleration_xyz-gravity_xyz)^2)))_per_window"
            ),
            "gyroscope_rms": "sqrt(mean(sum(gyroscope_xyz^2)))_per_window",
            "motion_overlap_description": (
                "within_wearer_common_language_probability_mobility_bout_median_exceeds_"
                "upper_body_motion_posture_bout_median_reported_per_energy_feature"
            ),
        },
        "completeness": {
            "complete_wearer_requires": (
                "all_16_fresh_posture_query_bouts_metadata_and_signal_eligible"
            ),
            "incomplete_wearer_imputation": "none",
            "failed_withdrawn_unresolved_retained": True,
            "stale_validity_mask": (
                "intersection_of_identical_query_window_ids_for_fresh_and_stale"
            ),
        },
        "uncertainty": {
            "independent_unit": "wearer",
            "report": [
                "all_wearer_values",
                "equal_wearer_mean",
                "sample_standard_deviation",
                "two_sided_95_percent_student_t_interval",
            ],
            "interval_degrees_of_freedom": "complete_wearer_count_minus_one",
            "interval_suppressed_below_complete_wearers": 2,
            "confirmatory_p_values": False,
        },
        "prototype_prerequisite_rule": {
            "complete_wearers_required": 6,
            "positive_mean_margin_required_for": [
                "quiet_posture_queries",
                "upper_body_motion_posture_queries",
            ],
            "application": (
                "every_complete_wearer_must_have_strictly_positive_fresh_mean_margin_"
                "in_each_motion_stratum"
            ),
            "decision_scope": "permits_new_protocol_draft_only",
        },
        "frame_rule": {
            "numerical_nondegeneracy_sufficient": False,
            "bench_uncertainty_limit_status": "pending_device_qualification",
        },
        "motion_rule": {
            "overlap_rejects_simple_energy_interpretation_only": True,
            "automatic_boundary_model_eligibility": False,
        },
        "c3_reopening_allowed": False,
        "population_superiority_claim_allowed": False,
    }
    if analysis != expected_analysis:
        raise ValueError("analysis contract changed")
    if resources != {
        "setup_engineering_wall_seconds": 7200,
        "initial_analysis_compute_wall_seconds": 3600,
        "total_participant_hours": 6,
        "model_fit_budget": 0,
        "automatic_architecture_experiment_allowed": False,
        "automatic_external_dataset_allowed": False,
    }:
        raise ValueError("resource contract changed")
    if any(value is not False for value in claims.values()):
        raise ValueError("claims must remain false before collection")


def _rotate(values: Sequence[tuple[str, str]], offset: int) -> list[tuple[str, str]]:
    index = offset % len(values)
    return list(values[index:]) + list(values[:index])


def _visit_id(wearer: str, visit: int) -> str:
    return f"{wearer}/visit-{visit}"


def _block_id(wearer: str, visit: int, attachment: int) -> str:
    return f"{_visit_id(wearer, visit)}/attachment-{attachment}"


def _bout_id(
    block: str,
    *,
    sequence: int,
    role: str,
    activity: str,
    motion: str,
) -> str:
    return f"{block}/bout-{sequence:02d}-{role}-{activity}-{motion}"


def build_plan(
    config: Mapping[str, Any],
    *,
    created_at_utc: str,
    code_commit: str,
    config_reference: Mapping[str, str],
    protocol_reference: Mapping[str, str],
    schema_reference: Mapping[str, str],
) -> dict[str, Any]:
    """Build the exact 264-bout plan and immutable matched comparisons."""

    validate_config(config)
    population = _mapping(config["population"], "population")
    schedule = _mapping(config["schedule"], "schedule")
    wearers = cast(list[str], population["wearer_ids"])
    visits: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    bouts: list[dict[str, Any]] = []
    block_support: dict[str, list[str]] = {}
    block_query: dict[str, list[str]] = {}
    global_block = 0
    for wearer_index, wearer in enumerate(wearers):
        for visit_number in (1, 2):
            visit_id = _visit_id(wearer, visit_number)
            visit_blocks = [_block_id(wearer, visit_number, attachment) for attachment in (1, 2)]
            visits.append(
                {
                    "visit_id": visit_id,
                    "wearer_id": wearer,
                    "visit_number": visit_number,
                    "appointment_cap_minutes": 30,
                    "block_ids": visit_blocks,
                    "actual_visit_started_at_utc": None,
                    "status": "planned",
                }
            )
            for attachment in (1, 2):
                block_id = _block_id(wearer, visit_number, attachment)
                # Adding the wearer index avoids repeating the same order in every
                # visit/attachment cell (four blocks per wearer would otherwise
                # alias a four-condition support rotation).
                rotation_offset = global_block + wearer_index
                support_order = _rotate(SUPPORT_CONDITIONS, rotation_offset)
                query_order = _rotate(QUERY_CONDITIONS, rotation_offset)
                block_bouts: list[str] = []
                support_ids: list[str] = []
                query_ids: list[str] = []
                for sequence, (activity, motion) in enumerate(support_order, start=1):
                    identifier = _bout_id(
                        block_id,
                        sequence=sequence,
                        role="support",
                        activity=activity,
                        motion=motion,
                    )
                    support_ids.append(identifier)
                    block_bouts.append(identifier)
                    bouts.append(
                        {
                            "bout_id": identifier,
                            "block_id": block_id,
                            "wearer_id": wearer,
                            "visit_number": visit_number,
                            "attachment_number": attachment,
                            "sequence_in_block": sequence,
                            "role": "support",
                            "activity": activity,
                            "motion": motion,
                            "expected_duration_seconds": 30,
                            "interior_trim_seconds_each_end": int(
                                schedule["interior_trim_seconds_each_end"]
                            ),
                            "prompt_id": f"support-{activity}-{motion}",
                            "status": "planned",
                        }
                    )
                for sequence, (activity, motion) in enumerate(query_order, start=5):
                    identifier = _bout_id(
                        block_id,
                        sequence=sequence,
                        role="query",
                        activity=activity,
                        motion=motion,
                    )
                    query_ids.append(identifier)
                    block_bouts.append(identifier)
                    bouts.append(
                        {
                            "bout_id": identifier,
                            "block_id": block_id,
                            "wearer_id": wearer,
                            "visit_number": visit_number,
                            "attachment_number": attachment,
                            "sequence_in_block": sequence,
                            "role": "query",
                            "activity": activity,
                            "motion": motion,
                            "expected_duration_seconds": 45 if activity == "mobility" else 30,
                            "interior_trim_seconds_each_end": int(
                                schedule["interior_trim_seconds_each_end"]
                            ),
                            "prompt_id": f"query-{activity}-{motion}",
                            "status": "planned",
                        }
                    )
                block_support[block_id] = support_ids
                block_query[block_id] = query_ids
                blocks.append(
                    {
                        "block_id": block_id,
                        "wearer_id": wearer,
                        "visit_id": visit_id,
                        "visit_number": visit_number,
                        "attachment_number": attachment,
                        "global_block_index": global_block,
                        "counterbalance_rotation_offset": rotation_offset,
                        "bout_ids": block_bouts,
                        "equipment_receipt_id": None,
                        "recording_id": None,
                        "status": "planned",
                    }
                )
                global_block += 1

    comparisons: list[dict[str, Any]] = []
    for block in blocks:
        block_id = str(block["block_id"])
        wearer = str(block["wearer_id"])
        visit = int(block["visit_number"])
        attachment = int(block["attachment_number"])
        comparisons.append(
            {
                "comparison_id": f"fresh::{block_id}",
                "comparison_type": "fresh",
                "query_block_id": block_id,
                "query_bout_ids": block_query[block_id],
                "fresh_support_bout_ids": block_support[block_id],
                "stale_support_bout_ids": None,
                "identical_query_ids_required": True,
                "validity_mask_policy": "same_query_ids_and_intersection_validity_mask",
            }
        )
        if attachment == 2:
            stale_block = _block_id(wearer, visit, 1)
            comparisons.append(
                {
                    "comparison_id": f"attachment-stale::{block_id}",
                    "comparison_type": "attachment_stale",
                    "query_block_id": block_id,
                    "query_bout_ids": block_query[block_id],
                    "fresh_support_bout_ids": block_support[block_id],
                    "stale_support_bout_ids": block_support[stale_block],
                    "identical_query_ids_required": True,
                    "validity_mask_policy": "same_query_ids_and_intersection_validity_mask",
                }
            )
        if visit == 2:
            stale_block = _block_id(wearer, 1, attachment)
            comparisons.append(
                {
                    "comparison_id": f"visit-stale::{block_id}",
                    "comparison_type": "visit_stale",
                    "query_block_id": block_id,
                    "query_bout_ids": block_query[block_id],
                    "fresh_support_bout_ids": block_support[block_id],
                    "stale_support_bout_ids": block_support[stale_block],
                    "identical_query_ids_required": True,
                    "validity_mask_policy": "same_query_ids_and_intersection_validity_mask",
                }
            )
    counts = {
        "wearers": len(wearers),
        "visits": len(visits),
        "blocks": len(blocks),
        "bouts": len(bouts),
        "support_bouts": sum(item["role"] == "support" for item in bouts),
        "query_bouts": sum(item["role"] == "query" for item in bouts),
        "nominal_recording_seconds": sum(int(item["expected_duration_seconds"]) for item in bouts),
        "fresh_query_blocks": sum(item["comparison_type"] == "fresh" for item in comparisons),
        "attachment_stale_query_blocks": sum(
            item["comparison_type"] == "attachment_stale" for item in comparisons
        ),
        "visit_stale_query_blocks": sum(
            item["comparison_type"] == "visit_stale" for item in comparisons
        ),
    }
    plan: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "physical_information_pilot_plan",
        "study_id": config["study_id"],
        "evidence_status": "planned",
        "readiness_status": "plan_valid_collection_pending",
        "created_at_utc": created_at_utc,
        "lineage": {
            "code_commit": code_commit,
            "config": dict(config_reference),
            "protocol": dict(protocol_reference),
            "schema": dict(schema_reference),
        },
        "counts": counts,
        "wearers": wearers,
        "visits": visits,
        "blocks": blocks,
        "bouts": bouts,
        "comparisons": comparisons,
        "pending_prerequisites": list(PENDING_PREREQUISITES),
        "claims": dict(_mapping(config["claims"], "claims")),
    }
    plan["record_sha256"] = canonical_json_sha256(plan)
    return plan


def verify_self_hash(record: Mapping[str, Any]) -> bool:
    body = dict(record)
    declared = body.pop("record_sha256", None)
    return isinstance(declared, str) and declared == canonical_json_sha256(body)


def validate_plan(plan: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate exact static schedule equality plus explicit chronology invariants."""

    errors: list[str] = []
    try:
        validate_config(config)
    except ValueError as exc:
        errors.append(str(exc))
    if not verify_self_hash(plan):
        errors.append("plan self-hash mismatch")
    lineage = _mapping(plan.get("lineage"), "plan lineage")
    try:
        expected = build_plan(
            config,
            created_at_utc=str(plan.get("created_at_utc")),
            code_commit=str(lineage.get("code_commit")),
            config_reference=_mapping(lineage.get("config"), "config reference"),
            protocol_reference=_mapping(lineage.get("protocol"), "protocol reference"),
            schema_reference=_mapping(lineage.get("schema"), "schema reference"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"plan reconstruction failed: {exc}")
        expected = {}
    for field in (
        "schema_version",
        "record_kind",
        "study_id",
        "evidence_status",
        "readiness_status",
        "counts",
        "wearers",
        "visits",
        "blocks",
        "bouts",
        "comparisons",
        "pending_prerequisites",
        "claims",
    ):
        if expected and plan.get(field) != expected.get(field):
            errors.append(f"plan field differs from deterministic schedule: {field}")
    bouts = cast(list[dict[str, Any]], plan.get("bouts", []))
    bout_ids = [str(item.get("bout_id")) for item in bouts]
    if len(bout_ids) != len(set(bout_ids)):
        errors.append("bout identifiers are not unique")
    by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for bout in bouts:
        by_block[str(bout.get("block_id"))].append(bout)
    for block, rows in by_block.items():
        rows.sort(key=lambda item: int(item.get("sequence_in_block", 0)))
        if len(rows) != 11:
            errors.append(f"block does not contain 11 bouts: {block}")
            continue
        roles = [str(item.get("role")) for item in rows]
        if roles != ["support"] * 4 + ["query"] * 7:
            errors.append(f"support does not precede every query: {block}")
    block_order = {
        str(item["block_id"]): int(item["global_block_index"])
        for item in cast(list[dict[str, Any]], plan.get("blocks", []))
    }
    bout_to_block = {str(item["bout_id"]): str(item["block_id"]) for item in bouts}
    for comparison in cast(list[dict[str, Any]], plan.get("comparisons", [])):
        query_block = str(comparison.get("query_block_id"))
        stale = comparison.get("stale_support_bout_ids")
        if isinstance(stale, list):
            for support_id in stale:
                support_block = bout_to_block.get(str(support_id))
                if (
                    support_block is None
                    or query_block not in block_order
                    or block_order[support_block] >= block_order[query_block]
                ):
                    errors.append(
                        f"stale support is missing or not earlier: {comparison.get('comparison_id')}"
                    )
                    break
    return {
        "valid": not errors,
        "errors": errors,
        "counts": plan.get("counts"),
        "support_precedes_query": not any("support does not precede" in item for item in errors),
        "future_support_absent": not any("not earlier" in item for item in errors),
        "self_hash_valid": verify_self_hash(plan),
    }


def _safe_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and not value.endswith("/")


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite_number(value: object, *, positive: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return math.isfinite(number) and (not positive or number > 0)


def _integer(value: object, *, minimum: int | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    return minimum is None or value >= minimum


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _parse_utc_z(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.endswith("Z"):
        return None
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        return None
    return parsed.astimezone(UTC)


def _resolve_relative_path(root: Path, value: object) -> tuple[Path | None, str | None]:
    if not _safe_relative_path(value):
        return None, "unsafe"
    relative = PurePosixPath(cast(str, value))
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return None, "symlink"
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None, "escapes root"
    return resolved, None


def validate_collection_records(
    *,
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    equipment: Mapping[str, Any] | None,
    recordings: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    raw_root: Path,
) -> dict[str, Any]:
    """Validate later physical records without inventing absent observations."""

    errors: list[str] = []
    try:
        plan_report = validate_plan(plan, config)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        plan_report = {"valid": False, "errors": [f"validation raised: {exc}"]}
    errors.extend(f"plan: {item}" for item in cast(list[str], plan_report["errors"]))
    equipment_error_start = len(errors)
    equipment_required = {
        "record_kind",
        "receipt_id",
        "device_id",
        "manufacturer",
        "model",
        "firmware",
        "logger",
        "placement_instruction_path",
        "placement_instruction_sha256",
        "axis_convention",
        "channel_map",
        "units",
        "clock_domain",
        "nominal_rate_hz",
        "observed_interval_summary",
        "native_gravity_available",
        "processing_contract",
        "frame_conditioning_limit",
        "bench_qualification_path",
        "bench_qualification_sha256",
        "qualified_at_utc",
        "processing_frozen_at_utc",
        "status",
    }
    raw_root = raw_root.resolve()
    equipment_qualified_at: datetime | None = None
    processing_frozen_at: datetime | None = None
    if equipment is None:
        errors.append("equipment receipt is pending")
    else:
        missing = sorted(equipment_required - set(equipment))
        if missing:
            errors.append(f"equipment fields missing: {missing}")
        unexpected = sorted(set(equipment) - equipment_required)
        if unexpected:
            errors.append(f"unexpected equipment fields: {unexpected}")
        if equipment.get("record_kind") != "physical_information_equipment_receipt":
            errors.append("equipment record kind is invalid")
        if equipment.get("status") != "qualified":
            errors.append("equipment is not qualified")
        for name in (
            "receipt_id",
            "device_id",
            "manufacturer",
            "model",
            "firmware",
            "logger",
            "axis_convention",
            "clock_domain",
        ):
            if not _nonempty_string(equipment.get(name)):
                errors.append(f"equipment field is empty or invalid: {name}")
        for path_name, hash_name in (
            ("placement_instruction_path", "placement_instruction_sha256"),
            ("bench_qualification_path", "bench_qualification_sha256"),
        ):
            path, path_error = _resolve_relative_path(raw_root, equipment.get(path_name))
            if path_error is not None:
                errors.append(f"equipment {path_name} path is {path_error}")
            elif path is None or not path.is_file():
                errors.append(f"equipment evidence file missing: {path_name}")
            elif not _valid_sha256(equipment.get(hash_name)):
                errors.append(f"equipment evidence hash is invalid: {hash_name}")
            elif sha256_file(path) != equipment.get(hash_name):
                errors.append(f"equipment evidence hash mismatch: {path_name}")
        channel_map = equipment.get("channel_map")
        required_channels = {"ax", "ay", "az", "gx", "gy", "gz"}
        if not isinstance(channel_map, Mapping) or not required_channels.issubset(channel_map):
            errors.append("equipment channel map is incomplete")
        elif any(
            not _nonempty_string(key) or not _nonempty_string(value)
            for key, value in channel_map.items()
        ):
            errors.append("equipment channel map contains an empty key or value")
        units = equipment.get("units")
        required_units = {"acceleration", "angular_velocity"}
        if not isinstance(units, Mapping) or not required_units.issubset(units):
            errors.append("equipment units are incomplete")
        elif any(
            not _nonempty_string(key) or not _nonempty_string(value) for key, value in units.items()
        ):
            errors.append("equipment units contain an empty key or value")
        interval_summary = equipment.get("observed_interval_summary")
        interval_fields = {
            "sample_count",
            "median_seconds",
            "p05_seconds",
            "p95_seconds",
            "nonpositive_interval_count",
            "gap_count",
        }
        if not isinstance(interval_summary, Mapping) or set(interval_summary) != interval_fields:
            errors.append("observed interval summary is incomplete")
        else:
            if not _integer(interval_summary.get("sample_count"), minimum=2):
                errors.append("observed interval sample count is invalid")
            for name in ("median_seconds", "p05_seconds", "p95_seconds"):
                if not _finite_number(interval_summary.get(name), positive=True):
                    errors.append(f"observed interval value is invalid: {name}")
            if all(
                _finite_number(interval_summary.get(name), positive=True)
                for name in ("median_seconds", "p05_seconds", "p95_seconds")
            ) and not (
                float(interval_summary["p05_seconds"])
                <= float(interval_summary["median_seconds"])
                <= float(interval_summary["p95_seconds"])
            ):
                errors.append("observed interval percentiles are not ordered")
            for name in ("nonpositive_interval_count", "gap_count"):
                if not _integer(interval_summary.get(name), minimum=0):
                    errors.append(f"observed interval count is invalid: {name}")
        processing = equipment.get("processing_contract")
        required_processing = {
            "source_rate_hz",
            "target_rate_hz",
            "gravity_source",
            "gravity_cutoff_hz",
            "maximum_gap_seconds",
            "resampling_method",
            "window_samples",
            "window_stride_samples",
        }
        if not isinstance(processing, Mapping) or set(processing) != required_processing:
            errors.append("device-specific processing contract is incomplete")
        else:
            for name in ("source_rate_hz", "target_rate_hz", "maximum_gap_seconds"):
                if not _finite_number(processing.get(name), positive=True):
                    errors.append(f"processing value is invalid: {name}")
            gravity_source = processing.get("gravity_source")
            cutoff = processing.get("gravity_cutoff_hz")
            if not isinstance(gravity_source, str) or gravity_source not in {
                "provider_native_gravity",
                "derived_causal_lowpass",
            }:
                errors.append("processing gravity source is invalid")
            elif gravity_source == "provider_native_gravity" and cutoff is not None:
                errors.append("native gravity must not declare a derived cutoff")
            elif gravity_source == "derived_causal_lowpass":
                source_rate = processing.get("source_rate_hz")
                if (
                    not _finite_number(cutoff, positive=True)
                    or not _finite_number(source_rate, positive=True)
                    or float(cast(int | float, cutoff))
                    >= float(cast(int | float, source_rate)) / 2.0
                ):
                    errors.append("derived gravity cutoff is invalid")
            if not _nonempty_string(processing.get("resampling_method")):
                errors.append("processing resampling method is invalid")
            if not _integer(processing.get("window_samples"), minimum=1):
                errors.append("processing window size is invalid")
            if not _integer(processing.get("window_stride_samples"), minimum=1):
                errors.append("processing window stride is invalid")
            elif _integer(processing.get("window_samples"), minimum=1) and int(
                processing["window_stride_samples"]
            ) > int(processing["window_samples"]):
                errors.append("processing window stride exceeds the window size")
        rate = equipment.get("nominal_rate_hz")
        if not _finite_number(rate, positive=True):
            errors.append("equipment nominal rate is invalid")
        elif (
            isinstance(processing, Mapping)
            and _finite_number(processing.get("source_rate_hz"), positive=True)
            and not math.isclose(
                float(cast(int | float, rate)),
                float(cast(int | float, processing["source_rate_hz"])),
                rel_tol=1e-9,
                abs_tol=0.0,
            )
        ):
            errors.append("equipment nominal and processing source rates differ")
        native_gravity_available = equipment.get("native_gravity_available")
        if not isinstance(native_gravity_available, bool):
            errors.append("native gravity availability is invalid")
        if (
            isinstance(processing, Mapping)
            and processing.get("gravity_source") == "provider_native_gravity"
        ):
            if native_gravity_available is not True:
                errors.append("provider-native gravity was selected but is unavailable")
            if not isinstance(channel_map, Mapping) or not {
                "gravity_x",
                "gravity_y",
                "gravity_z",
            }.issubset(channel_map):
                errors.append("provider-native gravity channels are not mapped")
        frame_limit = equipment.get("frame_conditioning_limit")
        if frame_limit is not None and not _finite_number(frame_limit, positive=True):
            errors.append("frame conditioning limit is invalid")
        equipment_qualified_at = _parse_utc_z(equipment.get("qualified_at_utc"))
        processing_frozen_at = _parse_utc_z(equipment.get("processing_frozen_at_utc"))
        if equipment_qualified_at is None:
            errors.append("equipment qualification UTC time is invalid")
        if processing_frozen_at is None:
            errors.append("processing freeze UTC time is invalid")
        if (
            equipment_qualified_at is not None
            and processing_frozen_at is not None
            and processing_frozen_at < equipment_qualified_at
        ):
            errors.append("processing was frozen before equipment qualification")
    equipment_ready = equipment is not None and len(errors) == equipment_error_start
    planned_blocks = {
        str(item["block_id"]): item for item in cast(list[dict[str, Any]], plan.get("blocks", []))
    }
    planned_bouts = {
        str(item["bout_id"]): item for item in cast(list[dict[str, Any]], plan.get("bouts", []))
    }
    recording_by_id: dict[str, Mapping[str, Any]] = {}
    claimed_block_ids: set[str] = set()
    recorded_block_ids: set[str] = set()
    recording_by_block: dict[str, Mapping[str, Any]] = {}
    recording_integrity_ids: set[str] = set()
    raw_paths: set[str] = set()
    visit_times: dict[tuple[str, int], datetime] = {}
    recording_intervals: dict[str, tuple[datetime, datetime]] = {}
    recording_status_counts = {status: 0 for status in sorted(COLLECTION_STATUSES)}
    invalid_recording_status_count = 0
    raw_required = {
        "recording_id",
        "relative_path",
        "sha256",
        "size_bytes",
        "sample_count",
        "device_id",
        "wearer_id",
        "visit_number",
        "attachment_number",
        "clock_domain",
        "visit_started_at_utc",
        "started_at",
        "ended_at",
        "status",
    }
    for record in recordings:
        record_error_start = len(errors)
        identifier_value = record.get("recording_id")
        identifier = identifier_value if isinstance(identifier_value, str) else ""
        if not _nonempty_string(identifier) or identifier in recording_by_id:
            errors.append(f"missing or duplicate recording id: {identifier!r}")
            continue
        recording_by_id[identifier] = record
        missing = sorted(raw_required - set(record))
        if missing:
            errors.append(f"recording fields missing for {identifier}: {missing}")
        unexpected = sorted(set(record) - raw_required)
        if unexpected:
            errors.append(f"unexpected recording fields for {identifier}: {unexpected}")
        status = record.get("status")
        if not isinstance(status, str) or status not in COLLECTION_STATUSES:
            errors.append(f"invalid recording status: {identifier}")
            invalid_recording_status_count += 1
        else:
            recording_status_counts[status] += 1
        for name in ("device_id", "wearer_id", "clock_domain"):
            if not _nonempty_string(record.get(name)):
                errors.append(f"recording field is empty or invalid for {identifier}: {name}")
        if equipment is not None:
            if record.get("device_id") != equipment.get("device_id"):
                errors.append(f"recording device differs from equipment receipt: {identifier}")
            if record.get("clock_domain") != equipment.get("clock_domain"):
                errors.append(f"recording clock differs from equipment receipt: {identifier}")
        if not _valid_sha256(record.get("sha256")):
            errors.append(f"raw hash value is invalid: {identifier}")
        if not _integer(record.get("size_bytes"), minimum=1):
            errors.append(f"invalid recording byte size: {identifier}")
        sample_count = record.get("sample_count")
        if not _integer(sample_count, minimum=1):
            errors.append(f"invalid recording sample count: {identifier}")
        relative = record.get("relative_path")
        path, path_error = _resolve_relative_path(raw_root, relative)
        if path_error is not None:
            errors.append(f"unsafe raw path ({path_error}): {identifier}")
        elif path is not None:
            normalized_path = str(path).casefold()
            if normalized_path in raw_paths:
                errors.append(f"multiple recordings reuse one raw object: {identifier}")
            raw_paths.add(normalized_path)
            if status == "recorded":
                if not path.is_file():
                    errors.append(f"recorded raw file missing: {identifier}")
                else:
                    if record.get("sha256") != sha256_file(path):
                        errors.append(f"raw hash mismatch: {identifier}")
                    if record.get("size_bytes") != path.stat().st_size:
                        errors.append(f"raw size mismatch: {identifier}")
        visit_number = record.get("visit_number")
        attachment_number = record.get("attachment_number")
        valid_coordinates = _integer(visit_number) and _integer(attachment_number)
        block_id = ""
        if valid_coordinates:
            block_id = _block_id(
                str(record.get("wearer_id")),
                cast(int, visit_number),
                cast(int, attachment_number),
            )
        if not valid_coordinates or block_id not in planned_blocks:
            errors.append(f"recording is outside planned block roster: {identifier}")
        elif block_id in claimed_block_ids:
            errors.append(f"multiple recordings claim one attachment block: {block_id}")
        else:
            claimed_block_ids.add(block_id)
            recording_by_block[block_id] = record
        visit_key = (str(record.get("wearer_id")), cast(int, visit_number or 0))
        visit_started = record.get("visit_started_at_utc")
        visit_datetime = _parse_utc_z(visit_started)
        if visit_datetime is None:
            errors.append(f"recording visit UTC time is invalid: {identifier}")
        elif visit_key in visit_times and visit_times[visit_key] != visit_datetime:
            errors.append(f"inconsistent visit UTC time across attachment blocks: {visit_key}")
        else:
            visit_times[visit_key] = visit_datetime
        if (
            visit_datetime is not None
            and processing_frozen_at is not None
            and visit_datetime <= processing_frozen_at
        ):
            errors.append(f"recording predates the processing freeze: {identifier}")
        started = record.get("started_at")
        ended = record.get("ended_at")
        interval_values_valid = (
            _finite_number(started)
            and _finite_number(ended)
            and float(cast(int | float, started)) >= 0.0
            and float(cast(int | float, ended)) > float(cast(int | float, started))
        )
        if not interval_values_valid:
            errors.append(f"recording time interval is invalid: {identifier}")
        elif visit_datetime is not None:
            try:
                absolute_interval = (
                    visit_datetime + timedelta(seconds=float(cast(int | float, started))),
                    visit_datetime + timedelta(seconds=float(cast(int | float, ended))),
                )
            except OverflowError:
                errors.append(f"recording time interval overflows UTC range: {identifier}")
            else:
                recording_intervals[identifier] = absolute_interval
                if (
                    processing_frozen_at is not None
                    and absolute_interval[0] <= processing_frozen_at
                ):
                    errors.append(f"recording starts before the processing freeze: {identifier}")
        if len(errors) == record_error_start and status == "recorded":
            recording_integrity_ids.add(identifier)
            recorded_block_ids.add(block_id)
    for wearer in cast(list[str], plan.get("wearers", [])):
        for visit_number in (1, 2):
            first_id = _block_id(wearer, visit_number, 1)
            second_id = _block_id(wearer, visit_number, 2)
            first = recording_by_block.get(first_id)
            second = recording_by_block.get(second_id)
            if first is None or second is None:
                continue
            first_interval = recording_intervals.get(str(first.get("recording_id")))
            second_interval = recording_intervals.get(str(second.get("recording_id")))
            if (
                first_interval is not None
                and second_interval is not None
                and second_interval[0] < first_interval[1]
            ):
                errors.append(f"attachment chronology violated: {wearer}/visit-{visit_number}")
    annotations_by_recording: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_bouts: set[str] = set()
    metadata_eligible_bouts: set[str] = set()
    annotation_statuses = ("adjudicated", "failed", "withdrawn", "unresolved")
    annotation_status_counts = {status: 0 for status in annotation_statuses}
    invalid_annotation_status_count = 0
    annotation_required = {
        "bout_id",
        "recording_id",
        "start_sample",
        "stop_sample_exclusive",
        "observed_activity",
        "observed_motion",
        "adjudication_status",
        "adjudicator_id",
        "adjudicated_at_utc",
    }
    allowed_observed_activity = {"sitting", "standing", "mobility", "other", "unresolved"}
    allowed_observed_motion = {
        "quiet",
        "upper_body_motion",
        "usual",
        "slow",
        "turning",
        "other",
        "unresolved",
    }
    for annotation in annotations:
        bout_value = annotation.get("bout_id")
        recording_value = annotation.get("recording_id")
        bout_id = bout_value if isinstance(bout_value, str) else ""
        recording_id = recording_value if isinstance(recording_value, str) else ""
        if not _nonempty_string(bout_id) or bout_id in seen_bouts:
            errors.append(f"missing or duplicate annotated bout: {bout_id!r}")
            continue
        seen_bouts.add(bout_id)
        missing = sorted(annotation_required - set(annotation))
        if missing:
            errors.append(f"annotation fields missing for {bout_id}: {missing}")
        unexpected = sorted(set(annotation) - annotation_required)
        if unexpected:
            errors.append(f"unexpected annotation fields for {bout_id}: {unexpected}")
        if (
            not isinstance(annotation.get("observed_activity"), str)
            or annotation.get("observed_activity") not in allowed_observed_activity
        ):
            errors.append(f"invalid observed activity: {bout_id}")
        if (
            not isinstance(annotation.get("observed_motion"), str)
            or annotation.get("observed_motion") not in allowed_observed_motion
        ):
            errors.append(f"invalid observed motion: {bout_id}")
        annotation_status = annotation.get("adjudication_status")
        if not isinstance(annotation_status, str) or annotation_status not in annotation_statuses:
            errors.append(f"invalid adjudication status: {bout_id}")
            invalid_annotation_status_count += 1
        else:
            annotation_status_counts[annotation_status] += 1
        if not _nonempty_string(annotation.get("adjudicator_id")):
            errors.append(f"missing adjudicator provenance: {bout_id}")
        adjudicated_at = annotation.get("adjudicated_at_utc")
        adjudicated_datetime = _parse_utc_z(adjudicated_at)
        if adjudicated_datetime is None:
            errors.append(f"invalid adjudication UTC time: {bout_id}")
        if bout_id not in planned_bouts:
            errors.append(f"annotation is outside plan: {bout_id}")
            continue
        if recording_id not in recording_by_id:
            errors.append(f"annotation recording is missing: {bout_id}")
            continue
        recording = recording_by_id[recording_id]
        planned = planned_bouts[bout_id]
        visit_number = recording.get("visit_number")
        attachment_number = recording.get("attachment_number")
        expected_block = (
            _block_id(
                str(recording.get("wearer_id")),
                cast(int, visit_number),
                cast(int, attachment_number),
            )
            if _integer(visit_number) and _integer(attachment_number)
            else ""
        )
        if planned.get("block_id") != expected_block:
            errors.append(f"annotation assigned to wrong recording block: {bout_id}")
        start = annotation.get("start_sample")
        stop = annotation.get("stop_sample_exclusive")
        sample_count = recording.get("sample_count")
        interval_valid = (
            _integer(start, minimum=0)
            and _integer(stop, minimum=1)
            and cast(int, stop) > cast(int, start)
            and _integer(sample_count, minimum=1)
            and cast(int, stop) <= cast(int, sample_count)
        )
        if not interval_valid:
            errors.append(f"annotation sample interval invalid: {bout_id}")
            continue
        annotations_by_recording[recording_id].append(annotation)
        if (
            recording_id in recording_integrity_ids
            and annotation_status == "adjudicated"
            and annotation.get("observed_activity") == planned.get("activity")
            and annotation.get("observed_motion") == planned.get("motion")
            and adjudicated_datetime is not None
            and planned.get("block_id") == expected_block
        ):
            metadata_eligible_bouts.add(bout_id)
    for _recording_id, rows in annotations_by_recording.items():
        ordered = sorted(rows, key=lambda item: int(item["start_sample"]))
        for left, right in pairwise(ordered):
            if int(left["stop_sample_exclusive"]) > int(right["start_sample"]):
                errors.append(
                    f"annotation sample overlap: {left['bout_id']} and {right['bout_id']}"
                )
        by_block: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in ordered:
            by_block[str(planned_bouts[str(row["bout_id"])]["block_id"])].append(row)
        for block_id, block_rows in by_block.items():
            observed_sequences = [
                int(planned_bouts[str(row["bout_id"])]["sequence_in_block"]) for row in block_rows
            ]
            if observed_sequences != sorted(observed_sequences):
                errors.append(f"bout sequence chronology violated: {block_id}")
            support_stops = [
                int(row["stop_sample_exclusive"])
                for row in block_rows
                if planned_bouts[str(row["bout_id"])]["role"] == "support"
            ]
            query_starts = [
                int(row["start_sample"])
                for row in block_rows
                if planned_bouts[str(row["bout_id"])]["role"] == "query"
            ]
            if support_stops and query_starts and max(support_stops) > min(query_starts):
                errors.append(f"support/query chronology violated: {block_id}")
    for wearer in cast(list[str], plan.get("wearers", [])):
        visit_1 = visit_times.get((wearer, 1))
        visit_2 = visit_times.get((wearer, 2))
        if visit_1 is not None and visit_2 is not None and visit_2 <= visit_1:
            errors.append(f"visit chronology violated: {wearer}")
    for comparison in cast(list[dict[str, Any]], plan.get("comparisons", [])):
        stale_ids = comparison.get("stale_support_bout_ids")
        if not isinstance(stale_ids, list) or not stale_ids:
            continue
        query_block_id = str(comparison.get("query_block_id"))
        stale_bout = planned_bouts.get(str(stale_ids[0]))
        stale_block_id = str(stale_bout.get("block_id")) if stale_bout is not None else ""
        stale_recording = recording_by_block.get(stale_block_id)
        query_recording = recording_by_block.get(query_block_id)
        if stale_recording is None or query_recording is None:
            continue
        stale_interval = recording_intervals.get(str(stale_recording.get("recording_id")))
        query_interval = recording_intervals.get(str(query_recording.get("recording_id")))
        if (
            stale_interval is not None
            and query_interval is not None
            and stale_interval[1] > query_interval[0]
        ):
            errors.append(f"stale reference chronology violated: {comparison.get('comparison_id')}")
    expected_block_ids = set(planned_blocks)
    expected_bout_ids = set(planned_bouts)
    recorded_evidence_complete = (
        equipment_ready
        and not errors
        and len(recordings) == len(expected_block_ids) == 24
        and claimed_block_ids == expected_block_ids
        and recorded_block_ids == expected_block_ids
        and recording_status_counts["recorded"] == 24
        and len(annotations) == len(expected_bout_ids) == 264
        and seen_bouts == expected_bout_ids
        and annotation_status_counts["adjudicated"] == 264
        and metadata_eligible_bouts == expected_bout_ids
    )
    study_prerequisites_verified = False
    collection_complete = recorded_evidence_complete and study_prerequisites_verified
    only_pending_equipment = (
        equipment is None
        and not recordings
        and not annotations
        and len(errors) == equipment_error_start + 1
        and errors[-1] == "equipment receipt is pending"
    )
    if only_pending_equipment:
        status = "collection_not_ready"
    elif errors:
        status = "collection_records_invalid"
    elif recorded_evidence_complete:
        status = "recorded_adjudicated_evidence_complete_signal_eligibility_pending"
    elif not equipment_ready:
        status = "collection_not_ready"
    elif recordings or annotations:
        status = "partial_recorded_evidence"
    else:
        status = "equipment_evidence_verified_collection_prerequisites_unverified"
    if equipment_ready:
        remaining_prerequisites = list(PENDING_PREREQUISITES[:3])
    else:
        remaining_prerequisites = list(PENDING_PREREQUISITES)
    return {
        "valid": not errors,
        "status": status,
        "errors": errors,
        "equipment_evidence_verified": equipment_ready,
        "study_prerequisites_verified": study_prerequisites_verified,
        "remaining_prerequisites": remaining_prerequisites,
        "recording_count": len(recordings),
        "recorded_block_count": len(recorded_block_ids),
        "claimed_block_count": len(claimed_block_ids),
        "recording_status_counts": {
            **recording_status_counts,
            "invalid": invalid_recording_status_count,
        },
        "annotation_count": len(annotations),
        "adjudicated_bout_count": annotation_status_counts["adjudicated"],
        "metadata_eligible_bout_count": len(metadata_eligible_bouts),
        "signal_eligibility_evaluated": False,
        "analysis_eligible_bout_count": 0,
        "annotation_status_counts": {
            **annotation_status_counts,
            "invalid": invalid_annotation_status_count,
        },
        "planned_block_count": len(planned_blocks),
        "planned_bout_count": len(planned_bouts),
        "missing_recorded_block_ids": sorted(expected_block_ids - recorded_block_ids),
        "missing_annotation_bout_ids": sorted(expected_bout_ids - seen_bouts),
        "metadata_ineligible_bout_ids": sorted(expected_bout_ids - metadata_eligible_bouts),
        "analysis_ready": False,
        "recorded_evidence_complete": recorded_evidence_complete,
        "collection_complete": collection_complete,
        "collection_completed_claim": False,
        "model_fit_count": 0,
    }


def load_plan(path: Path) -> dict[str, Any]:
    return _mapping(load_json_strict(path), str(path))
