"""Create-only zero-fit qualification for same-attachment directional algebra."""

from __future__ import annotations

import argparse
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import numpy as np
import yaml

from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)
from inclusive_shift_har.models.same_attachment_directional_reference import (
    DirectionalReferenceError,
    DirectionalSupport,
    JointReferenceHeadParameters,
    build_directional_support,
    build_joint_reference_head_parameters,
    compose_same_attachment_probabilities,
    directional_reference_evidence,
    summarize_support_bout,
)

FIXTURE_NAMES = (
    "common_proper_rotation_invariance",
    "sitting_standing_swap_equivariance",
    "uniform_broad_distribution_limit",
    "equal_odds_distinct_compatibility",
    "identical_anchor_nondiscrimination",
    "antipodal_anchor_likelihood_validity",
    "support_bout_rotation_equivariance",
    "zero_resultant_rejection",
    "nonfinite_direction_norm_rejection",
    "explicit_query_validity_fallback",
    "non_float64_baseline_rejection",
    "exact_missing_support_fallback",
    "exact_stale_support_fallback",
    "exact_invalid_support_fallback",
    "exact_attachment_mismatch_fallback",
    "probability_normalization_and_finiteness",
    "nonzero_joint_interaction_execution",
)

SOURCE_ROLES = (
    "runner",
    "directional_module",
    "tests",
    "runner_tests",
    "configuration",
    "protocol",
    "governing_specification",
    "controller_synthesis",
    "physical_pilot_protocol",
    "physical_pilot_config",
)


class GeometryQualificationError(ValueError):
    """Raised when the zero-fit qualification contract is violated."""


def _now_z() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GeometryQualificationError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _exact_fields(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise GeometryQualificationError(
            f"{name} fields changed; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _sealed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    if "record_sha256" in result:
        raise GeometryQualificationError("payload is already sealed")
    result["record_sha256"] = canonical_json_sha256(result)
    return result


def _verify_sealed(value: Mapping[str, Any], name: str) -> None:
    body = dict(value)
    declared = body.pop("record_sha256", None)
    if not isinstance(declared, str) or declared != canonical_json_sha256(body):
        raise GeometryQualificationError(f"{name} self-hash mismatch")


def _write_json(path: Path, value: Mapping[str, Any], *, root: Path) -> None:
    atomic_write_json_new(value, path, allowed_root=root)


def _write_bytes(path: Path, value: bytes, *, root: Path) -> None:
    resolved_root = root.resolve(strict=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = path.parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(resolved_root)
    except ValueError as exc:
        raise GeometryQualificationError(f"output path escapes run directory: {path}") from exc
    target = resolved_parent / path.name
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
    with target.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _load_config(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise GeometryQualificationError(f"configuration is missing or symbolic: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        loaded = yaml.safe_load(handle)
    config = _mapping(loaded, "configuration")
    _exact_fields(
        config,
        {
            "schema_version",
            "experiment_id",
            "status",
            "evidence_role",
            "class_order",
            "posture_order",
            "support_bouts_per_posture",
            "support_bouts_total",
            "equal_bout_mixture_weights",
            "numerics",
            "fixed_fixture_head",
            "required_fixtures",
            "resources",
            "claims",
        },
        "configuration",
    )
    if config.get("schema_version") != "1.0.0":
        raise GeometryQualificationError("unexpected configuration schema_version")
    if config.get("experiment_id") != "same-attachment-directional-reference-geometry-v1":
        raise GeometryQualificationError("unexpected experiment_id")
    if config.get("status") != "zero_fit_software_qualification":
        raise GeometryQualificationError("configuration status is not zero-fit qualification")
    if config.get("evidence_role") != "synthetic_fixture_not_physical_evidence":
        raise GeometryQualificationError("configuration evidence role changed")
    if config.get("class_order") != ["mobility", "sitting", "standing"]:
        raise GeometryQualificationError("class order changed")
    if config.get("posture_order") != ["sitting", "standing"]:
        raise GeometryQualificationError("posture order changed")
    if config.get("support_bouts_per_posture") != 2 or config.get("support_bouts_total") != 4:
        raise GeometryQualificationError("support-bout budget changed")
    if config.get("equal_bout_mixture_weights") is not True:
        raise GeometryQualificationError("equal-bout weighting is required")
    if config.get("required_fixtures") != list(FIXTURE_NAMES):
        raise GeometryQualificationError("required fixture schedule changed")
    resources = _mapping(config.get("resources"), "resources")
    _exact_fields(
        resources,
        {
            "model_fit_budget",
            "encoder_fit_budget",
            "target_cohort_access_allowed",
            "external_dataset_access_allowed",
            "workers_allowed",
            "monitors_allowed",
        },
        "resources",
    )
    if (
        resources.get("model_fit_budget") != 0
        or resources.get("encoder_fit_budget") != 0
        or resources.get("workers_allowed") != 0
        or resources.get("monitors_allowed") != 0
        or resources.get("target_cohort_access_allowed") is not False
        or resources.get("external_dataset_access_allowed") is not False
    ):
        raise GeometryQualificationError("zero-fit resource contract changed")
    claims = _mapping(config.get("claims"), "claims")
    _exact_fields(
        claims,
        {
            "physical_measurement_used",
            "physical_identifiability_demonstrated",
            "reattachment_robustness_demonstrated",
            "model_performance_demonstrated",
            "architecture_promotion_allowed",
        },
        "claims",
    )
    if any(value is not False for value in claims.values()):
        raise GeometryQualificationError("all qualification claims must remain false")
    numerics = _mapping(config.get("numerics"), "numerics")
    _exact_fields(
        numerics,
        {
            "norm_epsilon",
            "maximum_concentration",
            "comparison_atol",
            "probability_sum_atol",
        },
        "numerics",
    )
    for name in (
        "norm_epsilon",
        "maximum_concentration",
        "comparison_atol",
        "probability_sum_atol",
    ):
        value = numerics.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise GeometryQualificationError(f"numerics.{name} must be positive")
    head = _mapping(config.get("fixed_fixture_head"), "fixed_fixture_head")
    _exact_fields(
        head,
        {
            "motion_intercept",
            "motion_dynamics_coefficients",
            "motion_compatibility_coefficient",
            "motion_dispersion_coefficient",
            "motion_interaction_coefficients",
            "posture_ratio_coefficient",
            "posture_dispersion_interaction_coefficient",
            "posture_intercept_allowed",
        },
        "fixed_fixture_head",
    )
    if head.get("posture_intercept_allowed") is not False:
        raise GeometryQualificationError("fixture posture intercept must remain forbidden")
    return config


def _support(
    config: Mapping[str, Any],
    *,
    status: Literal["fresh", "stale", "invalid"] = "fresh",
    sitting_concentrations: np.ndarray[Any, Any] | None = None,
    standing_concentrations: np.ndarray[Any, Any] | None = None,
) -> DirectionalSupport:
    numerics = _mapping(config.get("numerics"), "numerics")
    return build_directional_support(
        attachment_id="attachment-01",
        sitting_directions=np.asarray([[0.0, 0.0, 1.0], [0.2, 0.0, 0.98]]),
        standing_directions=np.asarray([[0.0, 1.0, 0.0], [0.0, 0.98, 0.2]]),
        sitting_concentrations=(
            np.asarray([8.0, 4.0]) if sitting_concentrations is None else sitting_concentrations
        ),
        standing_concentrations=(
            np.asarray([5.0, 3.0]) if standing_concentrations is None else standing_concentrations
        ),
        sitting_dispersions_radians=np.asarray([0.10, 0.20]),
        standing_dispersions_radians=np.asarray([0.30, 0.40]),
        status=status,
        norm_epsilon=float(numerics["norm_epsilon"]),
        maximum_concentration=float(numerics["maximum_concentration"]),
    )


def _parameters(config: Mapping[str, Any], *, scale: float = 1.0) -> JointReferenceHeadParameters:
    head = _mapping(config.get("fixed_fixture_head"), "fixed_fixture_head")
    return build_joint_reference_head_parameters(
        motion_intercept=float(head["motion_intercept"]) * scale,
        motion_dynamics_coefficients=np.asarray(
            head["motion_dynamics_coefficients"], dtype=np.float64
        )
        * scale,
        motion_compatibility_coefficient=float(head["motion_compatibility_coefficient"]) * scale,
        motion_dispersion_coefficient=float(head["motion_dispersion_coefficient"]) * scale,
        motion_interaction_coefficients=np.asarray(
            head["motion_interaction_coefficients"], dtype=np.float64
        )
        * scale,
        posture_ratio_coefficient=float(head["posture_ratio_coefficient"]) * scale,
        posture_dispersion_interaction_coefficient=float(
            head["posture_dispersion_interaction_coefficient"]
        )
        * scale,
    )


def _baseline(rows: int) -> np.ndarray[Any, np.dtype[np.float64]]:
    result = np.tile(np.asarray([0.2, 0.3, 0.5]), (rows, 1))
    if rows:
        result[-1] = [1.0, 0.0, 0.0]
    return np.asarray(result, dtype=np.float64)


def _rotation() -> np.ndarray[Any, np.dtype[np.float64]]:
    axis = np.asarray([1.0, -2.0, 0.5], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    angle = 0.83
    skew = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.asarray(
        np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew),
        dtype=np.float64,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GeometryQualificationError(message)


def _rotation_fixture(config: Mapping[str, Any]) -> dict[str, Any]:
    support = _support(config)
    queries = np.asarray([[0.1, 0.2, 0.97], [-0.2, 0.9, 0.1], [0.8, 0.2, -0.1]])
    dynamics = np.asarray([[0.2, -0.1], [1.1, 0.7], [-0.5, 0.9]])
    rotation = _rotation()
    rotated_support = build_directional_support(
        attachment_id=support.attachment_id,
        sitting_directions=support.directions[0] @ rotation.T,
        standing_directions=support.directions[1] @ rotation.T,
        sitting_concentrations=support.concentrations[0],
        standing_concentrations=support.concentrations[1],
        sitting_dispersions_radians=support.dispersions_radians[0],
        standing_dispersions_radians=support.dispersions_radians[1],
    )
    original_evidence = directional_reference_evidence(queries, support)
    rotated_evidence = directional_reference_evidence(queries @ rotation.T, rotated_support)
    original_output = compose_same_attachment_probabilities(
        dynamics=dynamics,
        query_directions=queries,
        query_attachment_ids=[support.attachment_id] * len(queries),
        baseline_probabilities=_baseline(len(queries)),
        parameters=_parameters(config),
        support=support,
        query_valid=np.ones(len(queries), dtype=np.bool_),
    )
    rotated_output = compose_same_attachment_probabilities(
        dynamics=dynamics,
        query_directions=queries @ rotation.T,
        query_attachment_ids=[support.attachment_id] * len(queries),
        baseline_probabilities=_baseline(len(queries)),
        parameters=_parameters(config),
        support=rotated_support,
        query_valid=np.ones(len(queries), dtype=np.bool_),
    )
    error = float(
        max(
            np.max(np.abs(original_evidence.features - rotated_evidence.features)),
            np.max(np.abs(original_output.probabilities - rotated_output.probabilities)),
        )
    )
    tolerance = float(_mapping(config.get("numerics"), "numerics")["comparison_atol"])
    _require(error <= tolerance, "common-rotation error exceeds tolerance")
    return {"maximum_absolute_error": error, "tolerance": tolerance}


def _swap_fixture(config: Mapping[str, Any]) -> dict[str, Any]:
    support = _support(config)
    swapped = build_directional_support(
        attachment_id=support.attachment_id,
        sitting_directions=support.directions[1],
        standing_directions=support.directions[0],
        sitting_concentrations=support.concentrations[1],
        standing_concentrations=support.concentrations[0],
        sitting_dispersions_radians=support.dispersions_radians[1],
        standing_dispersions_radians=support.dispersions_radians[0],
    )
    queries = np.asarray([[0.0, 0.1, 1.0], [0.0, 1.0, 0.1]])
    dynamics = np.asarray([[0.4, -0.2], [0.7, 0.3]])
    baseline = _baseline(2)
    original_evidence = directional_reference_evidence(queries, support)
    swapped_evidence = directional_reference_evidence(queries, swapped)
    original = compose_same_attachment_probabilities(
        dynamics=dynamics,
        query_directions=queries,
        query_attachment_ids=[support.attachment_id] * 2,
        baseline_probabilities=baseline,
        parameters=_parameters(config),
        support=support,
        query_valid=np.ones(2, dtype=np.bool_),
    )
    relabeled = compose_same_attachment_probabilities(
        dynamics=dynamics,
        query_directions=queries,
        query_attachment_ids=[support.attachment_id] * 2,
        baseline_probabilities=baseline[:, [0, 2, 1]],
        parameters=_parameters(config),
        support=swapped,
        query_valid=np.ones(2, dtype=np.bool_),
    )
    error = float(
        max(
            np.max(
                np.abs(
                    swapped_evidence.log_likelihoods - original_evidence.log_likelihoods[:, ::-1]
                )
            ),
            np.max(np.abs(swapped_evidence.features[:, 0] + original_evidence.features[:, 0])),
            np.max(np.abs(swapped_evidence.features[:, 1:] - original_evidence.features[:, 1:])),
            np.max(np.abs(relabeled.probabilities - original.probabilities[:, [0, 2, 1]])),
        )
    )
    tolerance = float(_mapping(config.get("numerics"), "numerics")["comparison_atol"])
    _require(error <= tolerance, "class-swap error exceeds tolerance")
    return {"maximum_absolute_error": error, "tolerance": tolerance}


def _uniform_fixture(config: Mapping[str, Any]) -> dict[str, Any]:
    support = _support(
        config,
        sitting_concentrations=np.zeros(2),
        standing_concentrations=np.zeros(2),
    )
    evidence = directional_reference_evidence(
        np.asarray([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]), support
    )
    error = float(np.max(np.abs(evidence.features[:, :2])))
    tolerance = float(_mapping(config.get("numerics"), "numerics")["comparison_atol"])
    _require(error <= tolerance, "uniform limit is not neutral")
    return {"maximum_absolute_r_or_e": error, "mean_dispersion": float(evidence.features[0, 2])}


def _equal_odds_fixture(config: Mapping[str, Any]) -> dict[str, Any]:
    broad = _support(
        config,
        sitting_concentrations=np.zeros(2),
        standing_concentrations=np.zeros(2),
    )
    improbable = build_directional_support(
        attachment_id="attachment-01",
        sitting_directions=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        standing_directions=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        sitting_concentrations=np.asarray([100.0, 100.0]),
        standing_concentrations=np.asarray([100.0, 100.0]),
        sitting_dispersions_radians=np.asarray([0.1, 0.1]),
        standing_dispersions_radians=np.asarray([0.1, 0.1]),
    )
    query = np.asarray([[0.0, 0.0, -1.0]])
    broad_evidence = directional_reference_evidence(query, broad).features[0]
    improbable_evidence = directional_reference_evidence(query, improbable).features[0]
    _require(abs(float(broad_evidence[0])) <= 1.0e-12, "broad ratio is not zero")
    _require(abs(float(improbable_evidence[0])) <= 1.0e-12, "improbable ratio is not zero")
    _require(float(improbable_evidence[1]) < -100.0, "compatibility does not retain improbability")
    return {
        "broad_r": float(broad_evidence[0]),
        "broad_e": float(broad_evidence[1]),
        "improbable_r": float(improbable_evidence[0]),
        "improbable_e": float(improbable_evidence[1]),
    }


def _identical_fixture(config: Mapping[str, Any]) -> dict[str, Any]:
    support = build_directional_support(
        attachment_id="attachment-01",
        sitting_directions=np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        standing_directions=np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        sitting_concentrations=np.asarray([4.0, 4.0]),
        standing_concentrations=np.asarray([4.0, 4.0]),
        sitting_dispersions_radians=np.asarray([0.2, 0.2]),
        standing_dispersions_radians=np.asarray([0.2, 0.2]),
    )
    evidence = directional_reference_evidence(np.asarray([[1.0, 0.0, 0.0]]), support)
    ratio = float(evidence.features[0, 0])
    _require(abs(ratio) <= 1.0e-12, "identical anchors are spuriously discriminative")
    return {"posture_ratio": ratio, "finite": bool(np.isfinite(evidence.features).all())}


def _antipodal_fixture(config: Mapping[str, Any]) -> dict[str, Any]:
    support = build_directional_support(
        attachment_id="attachment-01",
        sitting_directions=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        standing_directions=np.asarray([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        sitting_concentrations=np.asarray([20.0, 20.0]),
        standing_concentrations=np.asarray([20.0, 20.0]),
        sitting_dispersions_radians=np.zeros(2),
        standing_dispersions_radians=np.zeros(2),
    )
    evidence = directional_reference_evidence(
        np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]), support
    )
    ratios = evidence.features[:, 0]
    _require(bool(np.isfinite(evidence.features).all()), "antipodal evidence is non-finite")
    _require(float(ratios[0]) > 30.0 and float(ratios[1]) < -30.0, "antipodal sign failed")
    return {"sitting_query_ratio": float(ratios[0]), "standing_query_ratio": float(ratios[1])}


def _bout_rotation_fixture(config: Mapping[str, Any]) -> dict[str, Any]:
    del config
    directions = np.asarray([[0.0, 0.0, 1.0], [0.1, 0.0, 0.995], [-0.1, 0.0, 0.995]])
    rotation = _rotation()
    original, median, p90 = summarize_support_bout(directions)
    rotated, rotated_median, rotated_p90 = summarize_support_bout(directions @ rotation.T)
    error = float(
        max(
            np.max(np.abs(rotated - original @ rotation.T)),
            abs(rotated_median - median),
            abs(rotated_p90 - p90),
        )
    )
    _require(error <= 1.0e-12, "support summary is not rotation equivariant")
    return {"maximum_absolute_error": error, "median_dispersion": median, "p90_dispersion": p90}


def _zero_resultant_fixture(config: Mapping[str, Any]) -> dict[str, Any]:
    del config
    rejected = False
    message = ""
    try:
        summarize_support_bout(np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]))
    except DirectionalReferenceError as exc:
        rejected = True
        message = str(exc)
    _require(rejected, "zero-resultant support was not rejected")
    return {"rejected": rejected, "reason": message}


def _nonfinite_norm_fixture(config: Mapping[str, Any]) -> dict[str, Any]:
    rejected = False
    message = ""
    try:
        directional_reference_evidence(
            np.asarray([[1.0e308, 1.0e308, 1.0e308]]),
            _support(config),
        )
    except DirectionalReferenceError as exc:
        rejected = True
        message = str(exc)
    _require(rejected and "non-finite computed norm" in message, "overflowed norm did not fail")
    baseline = _baseline(1)
    output = compose_same_attachment_probabilities(
        dynamics=np.zeros((1, 2)),
        query_directions=np.asarray([[1.0e308, 1.0e308, 1.0e308]]),
        query_attachment_ids=["attachment-01"],
        baseline_probabilities=baseline,
        parameters=_parameters(config),
        support=_support(config),
        query_valid=np.ones(1, dtype=np.bool_),
    )
    fallback_exact = output.probabilities.tobytes() == baseline.tobytes()
    _require(
        fallback_exact and output.fallback_reasons == ("invalid_query_direction",),
        "overflowed query did not take exact invalid-direction fallback",
    )
    return {"strict_rejected": rejected, "reason": message, "fallback_bytes_exact": fallback_exact}


def _explicit_validity_fixture(config: Mapping[str, Any]) -> dict[str, Any]:
    baseline = _baseline(2)
    output = compose_same_attachment_probabilities(
        dynamics=np.zeros((2, 2)),
        query_directions=np.tile(np.asarray([0.0, 0.0, 1.0]), (2, 1)),
        query_attachment_ids=["attachment-01", "attachment-01"],
        baseline_probabilities=baseline,
        parameters=_parameters(config),
        support=_support(config),
        query_valid=np.zeros(2, dtype=np.bool_),
    )
    exact = output.probabilities.dtype == baseline.dtype and (
        output.probabilities.tobytes() == baseline.tobytes()
    )
    _require(exact, "explicit-invalid query fallback changed")
    _require(
        output.fallback_reasons == ("query_marked_invalid", "query_marked_invalid"),
        "explicit-invalid query reason changed",
    )
    return {"baseline_bytes_exact": exact, "used_reference_count": int(output.used_reference.sum())}


def _float64_baseline_fixture(config: Mapping[str, Any]) -> dict[str, Any]:
    rejected = False
    message = ""
    try:
        compose_same_attachment_probabilities(
            dynamics=np.zeros((1, 2)),
            query_directions=np.asarray([[0.0, 0.0, 1.0]]),
            query_attachment_ids=["attachment-01"],
            baseline_probabilities=np.asarray([[0.2, 0.3, 0.5]], dtype=np.float32),
            parameters=_parameters(config),
            support=None,
            query_valid=np.ones(1, dtype=np.bool_),
        )
    except DirectionalReferenceError as exc:
        rejected = True
        message = str(exc)
    _require(rejected and "native float64" in message, "non-float64 baseline did not fail")
    return {"rejected": rejected, "reason": message}


def _fallback_fixture(config: Mapping[str, Any], status: str) -> dict[str, Any]:
    row_count = 4
    baseline = _baseline(row_count)
    support: DirectionalSupport | None
    if status == "missing":
        support = None
        expected_reason = "missing_support"
    else:
        typed_status = cast(Literal["stale", "invalid"], status)
        support = _support(config, status=typed_status)
        expected_reason = f"{status}_support"
    output = compose_same_attachment_probabilities(
        dynamics=np.zeros((row_count, 2)),
        query_directions=np.tile(np.asarray([0.0, 0.0, 1.0]), (row_count, 1)),
        query_attachment_ids=["attachment-01"] * row_count,
        baseline_probabilities=baseline,
        parameters=_parameters(config),
        support=support,
        query_valid=np.ones(row_count, dtype=np.bool_),
    )
    exact = output.probabilities.dtype == baseline.dtype and (
        output.probabilities.tobytes() == baseline.tobytes()
    )
    reasons_exact = output.fallback_reasons == tuple(expected_reason for _ in range(row_count))
    _require(exact and reasons_exact and not np.any(output.used_reference), "fallback changed")
    return {
        "baseline_bytes_exact": exact,
        "reasons_exact": reasons_exact,
        "used_reference_count": int(output.used_reference.sum()),
    }


def _attachment_mismatch_fixture(config: Mapping[str, Any]) -> dict[str, Any]:
    baseline = _baseline(2)
    output = compose_same_attachment_probabilities(
        dynamics=np.zeros((2, 2)),
        query_directions=np.tile(np.asarray([0.0, 0.0, 1.0]), (2, 1)),
        query_attachment_ids=["wrong-01", "wrong-02"],
        baseline_probabilities=baseline,
        parameters=_parameters(config),
        support=_support(config),
        query_valid=np.ones(2, dtype=np.bool_),
    )
    exact = output.probabilities.dtype == baseline.dtype and (
        output.probabilities.tobytes() == baseline.tobytes()
    )
    _require(exact, "attachment-mismatch fallback changed")
    _require(
        output.fallback_reasons == ("attachment_mismatch", "attachment_mismatch"),
        "attachment mismatch reason changed",
    )
    return {"baseline_bytes_exact": exact, "used_reference_count": int(output.used_reference.sum())}


def _normalization_fixture(config: Mapping[str, Any]) -> dict[str, Any]:
    row_count = 4
    output = compose_same_attachment_probabilities(
        dynamics=np.asarray([[100.0, -100.0], [-100.0, 100.0], [0.0, 0.0], [2.0, 3.0]]),
        query_directions=np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        ),
        query_attachment_ids=["attachment-01"] * row_count,
        baseline_probabilities=_baseline(row_count),
        parameters=_parameters(config, scale=100.0),
        support=_support(config),
        query_valid=np.ones(row_count, dtype=np.bool_),
    )
    sum_error = float(np.max(np.abs(output.probabilities.sum(axis=1) - 1.0)))
    finite = bool(np.isfinite(output.probabilities).all())
    bounded = bool(np.all((output.probabilities >= 0.0) & (output.probabilities <= 1.0)))
    tolerance = float(_mapping(config.get("numerics"), "numerics")["probability_sum_atol"])
    _require(finite and bounded and sum_error <= tolerance, "probability validity failed")
    return {
        "finite": finite,
        "bounded": bounded,
        "maximum_probability_sum_error": sum_error,
        "tolerance": tolerance,
    }


def _interaction_fixture(config: Mapping[str, Any]) -> dict[str, Any]:
    support = _support(config)
    dynamics = np.asarray([[0.4, -0.7]])
    query = np.asarray([[0.1, 0.2, 0.97]])
    parameters = _parameters(config)
    evidence = directional_reference_evidence(query, support).features[0]
    output = compose_same_attachment_probabilities(
        dynamics=dynamics,
        query_directions=query,
        query_attachment_ids=[support.attachment_id],
        baseline_probabilities=_baseline(1),
        parameters=parameters,
        support=support,
        query_valid=np.ones(1, dtype=np.bool_),
    )
    ratio, compatibility, dispersion = (float(item) for item in evidence)
    interaction = compatibility * float(dynamics[0] @ parameters.motion_interaction_coefficients)
    motion_logit = (
        parameters.motion_intercept
        + float(dynamics[0] @ parameters.motion_dynamics_coefficients)
        + parameters.motion_compatibility_coefficient * compatibility
        + parameters.motion_dispersion_coefficient * dispersion
        + interaction
    )
    expected_motion = 1.0 / (1.0 + np.exp(-motion_logit))
    posture_logit = ratio * (
        parameters.posture_ratio_coefficient
        + parameters.posture_dispersion_interaction_coefficient * dispersion
    )
    expected_sitting_conditional = 1.0 / (1.0 + np.exp(-posture_logit))
    errors = (
        abs(float(output.probabilities[0, 0]) - float(expected_motion)),
        abs(
            float(output.probabilities[0, 1])
            - (1.0 - float(expected_motion)) * float(expected_sitting_conditional)
        ),
    )
    _require(interaction != 0.0, "fixture interaction is accidentally zero")
    _require(max(errors) <= 1.0e-12, "joint interaction hand calculation failed")
    return {
        "interaction_term": interaction,
        "motion_probability": float(output.probabilities[0, 0]),
        "sitting_probability": float(output.probabilities[0, 1]),
        "maximum_hand_calculation_error": max(errors),
    }


def _fixture_functions() -> dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]]:
    return {
        "common_proper_rotation_invariance": _rotation_fixture,
        "sitting_standing_swap_equivariance": _swap_fixture,
        "uniform_broad_distribution_limit": _uniform_fixture,
        "equal_odds_distinct_compatibility": _equal_odds_fixture,
        "identical_anchor_nondiscrimination": _identical_fixture,
        "antipodal_anchor_likelihood_validity": _antipodal_fixture,
        "support_bout_rotation_equivariance": _bout_rotation_fixture,
        "zero_resultant_rejection": _zero_resultant_fixture,
        "nonfinite_direction_norm_rejection": _nonfinite_norm_fixture,
        "explicit_query_validity_fallback": _explicit_validity_fixture,
        "non_float64_baseline_rejection": _float64_baseline_fixture,
        "exact_missing_support_fallback": lambda config: _fallback_fixture(config, "missing"),
        "exact_stale_support_fallback": lambda config: _fallback_fixture(config, "stale"),
        "exact_invalid_support_fallback": lambda config: _fallback_fixture(config, "invalid"),
        "exact_attachment_mismatch_fallback": _attachment_mismatch_fixture,
        "probability_normalization_and_finiteness": _normalization_fixture,
        "nonzero_joint_interaction_execution": _interaction_fixture,
    }


def build_geometry_report(config: Mapping[str, Any]) -> dict[str, Any]:
    """Execute all deterministic fixtures and return a JSON-safe zero-fit report."""

    functions = _fixture_functions()
    rows: list[dict[str, Any]] = []
    for name in FIXTURE_NAMES:
        try:
            measurements = functions[name](config)
        except Exception as exc:
            rows.append(
                {
                    "fixture": name,
                    "status": "fail",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "measurements": None,
                }
            )
        else:
            rows.append(
                {
                    "fixture": name,
                    "status": "pass",
                    "error_type": None,
                    "error": None,
                    "measurements": measurements,
                }
            )
    passed = sum(row["status"] == "pass" for row in rows)
    return {
        "record_kind": "same_attachment_directional_reference_geometry_report",
        "schema_version": "1.0.0",
        "status": "pass" if passed == len(rows) else "fail",
        "evidence_role": "synthetic_fixture_not_physical_evidence",
        "class_order": ["mobility", "sitting", "standing"],
        "posture_order": ["sitting", "standing"],
        "fixture_count": len(rows),
        "passed_fixture_count": passed,
        "failed_fixture_count": len(rows) - passed,
        "fixtures": rows,
        "model_fit_count": 0,
        "encoder_fit_count": 0,
        "target_cohort_loaded": False,
        "external_dataset_loaded": False,
        "physical_measurement_loaded": False,
        "automatic_follow_on_launched": False,
        "claims": {
            "software_algebra_qualified": passed == len(rows),
            "physical_identifiability_demonstrated": False,
            "reattachment_robustness_demonstrated": False,
            "model_performance_demonstrated": False,
            "architecture_promoted": False,
        },
    }


def _git_state(repository_root: Path) -> dict[str, Any]:
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"branch": branch, "commit": commit, "clean": not status, "status_lines": status}


def _snapshot_sources(
    *,
    run_directory: Path,
    repository_root: Path,
    config_path: Path,
    protocol_path: Path,
    governing_spec_path: Path,
    synthesis_path: Path,
) -> dict[str, Any]:
    runner_path = Path(__file__).resolve(strict=True)
    module_path = (
        repository_root / "src/inclusive_shift_har/models/same_attachment_directional_reference.py"
    ).resolve(strict=True)
    records = (
        ("runner", runner_path, "runner.py"),
        ("directional_module", module_path, "directional_module.py"),
        (
            "tests",
            (repository_root / "tests/test_same_attachment_directional_reference.py").resolve(
                strict=True
            ),
            "tests.py",
        ),
        (
            "runner_tests",
            (
                repository_root / "tests/test_same_attachment_directional_reference_geometry.py"
            ).resolve(strict=True),
            "runner_tests.py",
        ),
        ("configuration", config_path.resolve(strict=True), "config.yaml"),
        ("protocol", protocol_path.resolve(strict=True), "protocol.md"),
        ("governing_specification", governing_spec_path.resolve(strict=True), "governing_spec.md"),
        ("controller_synthesis", synthesis_path.resolve(strict=True), "controller_synthesis.md"),
        (
            "physical_pilot_protocol",
            (
                repository_root
                / "docs/research/PHYSICAL_INFORMATION_IDENTIFIABILITY_V1_PROTOCOL.md"
            ).resolve(strict=True),
            "physical_pilot_protocol.md",
        ),
        (
            "physical_pilot_config",
            (
                repository_root / "configs/experiments/physical_information_identifiability_v1.yaml"
            ).resolve(strict=True),
            "physical_pilot_config.yaml",
        ),
    )
    rows: list[dict[str, Any]] = []
    for role, source, snapshot_name in records:
        payload = source.read_bytes()
        target = run_directory / "snapshots" / snapshot_name
        _write_bytes(target, payload, root=run_directory)
        rows.append(
            {
                "role": role,
                "source_path": str(source),
                "snapshot_path": target.relative_to(run_directory).as_posix(),
                "size_bytes": len(payload),
                "sha256": sha256_file(target),
            }
        )
    return _sealed(
        {
            "record_kind": "same_attachment_directional_reference_source_manifest",
            "schema_version": "1.0.0",
            "repository": _git_state(repository_root),
            "sources": rows,
        }
    )


def _source_by_role(source_manifest: Mapping[str, Any], role: str, run_directory: Path) -> Path:
    rows = _validated_source_rows(source_manifest, run_directory)
    matched = [row for row in rows if row["role"] == role]
    if len(matched) != 1:
        raise GeometryQualificationError(f"source role is not unique: {role}")
    return cast(Path, matched[0]["resolved_path"])


def _validated_source_rows(
    source_manifest: Mapping[str, Any], run_directory: Path
) -> list[dict[str, Any]]:
    values = source_manifest.get("sources")
    if not isinstance(values, list):
        raise GeometryQualificationError("source manifest sources must be a list")
    rows = [_mapping(value, "source row") for value in values]
    roles = [row.get("role") for row in rows]
    if (
        any(not isinstance(role, str) for role in roles)
        or len(roles) != len(set(cast(list[str], roles)))
        or set(roles) != set(SOURCE_ROLES)
    ):
        raise GeometryQualificationError("source manifest roles are missing, extra, or duplicated")
    root = run_directory.resolve(strict=True)
    validated: list[dict[str, Any]] = []
    for row in rows:
        if set(row) != {"role", "source_path", "snapshot_path", "size_bytes", "sha256"}:
            raise GeometryQualificationError("source row fields changed")
        if not isinstance(row["source_path"], str) or not row["source_path"]:
            raise GeometryQualificationError("source path declaration is invalid")
        relative = row["snapshot_path"]
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise GeometryQualificationError("source snapshot path is not safe POSIX relative")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
            raise GeometryQualificationError("source snapshot path is not safe POSIX relative")
        path = root.joinpath(*pure.parts).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise GeometryQualificationError("source snapshot escapes run directory") from exc
        if path.is_symlink() or not path.is_file():
            raise GeometryQualificationError("source snapshot must be a regular non-symbolic file")
        size = row["size_bytes"]
        digest = row["sha256"]
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise GeometryQualificationError("source snapshot size/hash declaration is invalid")
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise GeometryQualificationError(f"source snapshot hash/size mismatch: {relative}")
        validated.append({**row, "resolved_path": path})
    return validated


def _internal_validation(
    run_directory: Path,
    source_manifest: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    config = _load_config(_source_by_role(source_manifest, "configuration", run_directory))
    recomputed = build_geometry_report(config)
    report_body = dict(report)
    report_body.pop("record_sha256", None)
    _validated_source_rows(source_manifest, run_directory)
    checks = {
        "source_snapshot_hashes_valid": True,
        "fixture_report_exact_replay": canonical_json_sha256(recomputed)
        == canonical_json_sha256(report_body),
        "all_required_fixtures_pass": recomputed["status"] == "pass"
        and recomputed["fixture_count"] == len(FIXTURE_NAMES),
        "zero_model_fits": recomputed["model_fit_count"] == 0,
        "zero_encoder_fits": recomputed["encoder_fit_count"] == 0,
        "no_target_or_external_or_physical_data": not any(
            recomputed[key]
            for key in (
                "target_cohort_loaded",
                "external_dataset_loaded",
                "physical_measurement_loaded",
            )
        ),
        "no_automatic_follow_on": recomputed["automatic_follow_on_launched"] is False,
    }
    return _sealed(
        {
            "record_kind": "same_attachment_directional_reference_geometry_validation",
            "schema_version": "1.0.0",
            "status": "pass" if all(checks.values()) else "fail",
            "checks": checks,
            "validation_fit_count": 0,
        }
    )


def _artifact_rows(run_directory: Path) -> list[dict[str, Any]]:
    excluded = {"artifact_manifest.json", "completion_manifest.json"}
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in run_directory.rglob("*") if item.is_file()):
        relative = path.relative_to(run_directory).as_posix()
        if relative in excluded:
            continue
        rows.append(
            {"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return rows


def _summary(report: Mapping[str, Any]) -> str:
    return (
        "# Same-attachment directional-reference geometry v1\n\n"
        f"Status: **{str(report['status']).upper()}**\n\n"
        f"All {report['passed_fixture_count']} of {report['fixture_count']} required deterministic "
        "fixtures passed with zero model fits and zero encoder fits. The result qualifies only the "
        "software algebra. Physical identifiability, reattachment robustness, model performance, "
        "architecture promotion, and novelty remain unestablished.\n\n"
        "The next external gate remains the frozen six-wearer physical information pilot. No "
        "supervised follow-on was launched.\n"
    )


def run_geometry_qualification(
    *,
    repository_root: Path,
    config_path: Path,
    protocol_path: Path,
    governing_spec_path: Path,
    synthesis_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Create and seal one zero-fit geometry qualification package."""

    root = repository_root.resolve(strict=True)
    if Path(__file__).resolve(strict=True).parents[3] != root:
        raise GeometryQualificationError("repository_root is not the executing source checkout")
    if os.path.lexists(output_directory):
        raise FileExistsError(f"refusing to overwrite output directory: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir()
    started = time.perf_counter()
    receipt = _sealed(
        {
            "record_kind": "same_attachment_directional_reference_geometry_command_receipt",
            "schema_version": "1.0.0",
            "status": "started",
            "started_at_utc": _now_z(),
            "output_directory": str(output_directory.resolve(strict=True)),
            "controller_pid": os.getpid(),
            "model_fit_budget": 0,
            "encoder_fit_budget": 0,
        }
    )
    _write_json(output_directory / "command_receipt.json", receipt, root=output_directory)
    try:
        config = _load_config(config_path.resolve(strict=True))
        source_manifest = _snapshot_sources(
            run_directory=output_directory,
            repository_root=root,
            config_path=config_path,
            protocol_path=protocol_path,
            governing_spec_path=governing_spec_path,
            synthesis_path=synthesis_path,
        )
        _write_json(
            output_directory / "source_manifest.json",
            source_manifest,
            root=output_directory,
        )
        environment = _sealed(
            {
                "record_kind": "same_attachment_directional_reference_geometry_environment",
                "schema_version": "1.0.0",
                "python_version": sys.version,
                "python_implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "numpy_version": np.__version__,
                "pyyaml_version": yaml.__version__,
                "model_fit_count": 0,
                "encoder_fit_count": 0,
            }
        )
        _write_json(
            output_directory / "environment.json",
            environment,
            root=output_directory,
        )
        fixture_started = time.perf_counter()
        report = _sealed(build_geometry_report(config))
        fixture_seconds = time.perf_counter() - fixture_started
        _write_json(output_directory / "geometry_report.json", report, root=output_directory)
        worker_shutdown = _sealed(
            {
                "record_kind": "same_attachment_directional_reference_worker_shutdown",
                "schema_version": "1.0.0",
                "status": "pass",
                "task_owned_workers_remaining": 0,
                "task_owned_monitors_remaining": 0,
                "workers_launched": 0,
                "monitors_launched": 0,
            }
        )
        _write_json(
            output_directory / "worker_shutdown.json",
            worker_shutdown,
            root=output_directory,
        )
        validation = _internal_validation(output_directory, source_manifest, report)
        _write_json(output_directory / "validation.json", validation, root=output_directory)
        runtime = _sealed(
            {
                "record_kind": "same_attachment_directional_reference_geometry_runtime",
                "schema_version": "1.0.0",
                "fixture_seconds": fixture_seconds,
                "total_pre_manifest_seconds": time.perf_counter() - started,
                "model_fit_seconds": 0.0,
                "encoder_fit_seconds": 0.0,
            }
        )
        _write_json(output_directory / "runtime.json", runtime, root=output_directory)
        _write_bytes(
            output_directory / "OUTCOME_SUMMARY.md",
            _summary(report).encode("utf-8"),
            root=output_directory,
        )
        artifact_manifest = _sealed(
            {
                "record_kind": "same_attachment_directional_reference_artifact_manifest",
                "schema_version": "1.0.0",
                "status": "complete",
                "files": _artifact_rows(output_directory),
            }
        )
        _write_json(
            output_directory / "artifact_manifest.json",
            artifact_manifest,
            root=output_directory,
        )
        completion = _sealed(
            {
                "record_kind": "same_attachment_directional_reference_completion_manifest",
                "schema_version": "1.0.0",
                "status": "complete" if report["status"] == "pass" else "complete_failed_gate",
                "completed_at_utc": _now_z(),
                "artifact_manifest_sha256": sha256_file(
                    output_directory / "artifact_manifest.json"
                ),
                "geometry_report_sha256": sha256_file(output_directory / "geometry_report.json"),
                "validation_sha256": sha256_file(output_directory / "validation.json"),
                "model_fit_count": 0,
                "encoder_fit_count": 0,
                "automatic_follow_on_launched": False,
            }
        )
        _write_json(
            output_directory / "completion_manifest.json",
            completion,
            root=output_directory,
        )
        return validate_geometry_qualification(output_directory)
    except BaseException as exc:
        failure = _sealed(
            {
                "record_kind": "same_attachment_directional_reference_geometry_failure",
                "schema_version": "1.0.0",
                "status": "incomplete",
                "failed_at_utc": _now_z(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "model_fit_count": 0,
                "encoder_fit_count": 0,
                "workers_remaining": 0,
                "monitors_remaining": 0,
            }
        )
        if not os.path.lexists(output_directory / "INCOMPLETE.json"):
            _write_json(output_directory / "INCOMPLETE.json", failure, root=output_directory)
        raise


def validate_geometry_qualification(run_directory: Path) -> dict[str, Any]:
    """Read-only validate artifacts and recompute every deterministic fixture."""

    root = run_directory.resolve(strict=True)
    if not root.is_dir() or (root / "INCOMPLETE.json").exists():
        raise GeometryQualificationError("run directory is missing or incomplete")
    artifact_manifest = _mapping(
        load_json_strict(root / "artifact_manifest.json"), "artifact manifest"
    )
    completion = _mapping(load_json_strict(root / "completion_manifest.json"), "completion")
    source_manifest = _mapping(load_json_strict(root / "source_manifest.json"), "source manifest")
    report = _mapping(load_json_strict(root / "geometry_report.json"), "geometry report")
    stored_validation = _mapping(load_json_strict(root / "validation.json"), "validation")
    environment = _mapping(load_json_strict(root / "environment.json"), "environment")
    command_receipt = _mapping(load_json_strict(root / "command_receipt.json"), "command receipt")
    worker_shutdown = _mapping(load_json_strict(root / "worker_shutdown.json"), "worker shutdown")
    runtime = _mapping(load_json_strict(root / "runtime.json"), "runtime")
    for name, record in (
        ("artifact manifest", artifact_manifest),
        ("completion", completion),
        ("source manifest", source_manifest),
        ("geometry report", report),
        ("validation", stored_validation),
        ("environment", environment),
        ("command receipt", command_receipt),
        ("worker shutdown", worker_shutdown),
        ("runtime", runtime),
    ):
        _verify_sealed(record, name)
    zero_fit_checks = {
        "environment_model_fits": environment.get("model_fit_count") == 0,
        "environment_encoder_fits": environment.get("encoder_fit_count") == 0,
        "receipt_status": command_receipt.get("status") == "started",
        "receipt_model_fit_budget": command_receipt.get("model_fit_budget") == 0,
        "receipt_encoder_fit_budget": command_receipt.get("encoder_fit_budget") == 0,
        "runtime_model_fit_seconds": runtime.get("model_fit_seconds") == 0.0,
        "runtime_encoder_fit_seconds": runtime.get("encoder_fit_seconds") == 0.0,
        "completion_model_fits": completion.get("model_fit_count") == 0,
        "completion_encoder_fits": completion.get("encoder_fit_count") == 0,
        "completion_no_follow_on": completion.get("automatic_follow_on_launched") is False,
        "report_model_fits": report.get("model_fit_count") == 0,
        "report_encoder_fits": report.get("encoder_fit_count") == 0,
        "report_no_follow_on": report.get("automatic_follow_on_launched") is False,
        "validation_fits": stored_validation.get("validation_fit_count") == 0,
        "shutdown_status": worker_shutdown.get("status") == "pass",
        "shutdown_workers": worker_shutdown.get("task_owned_workers_remaining") == 0,
        "shutdown_monitors": worker_shutdown.get("task_owned_monitors_remaining") == 0,
        "launched_workers": worker_shutdown.get("workers_launched") == 0,
        "launched_monitors": worker_shutdown.get("monitors_launched") == 0,
    }
    failed_zero_fit_checks = [name for name, passed in zero_fit_checks.items() if not passed]
    if failed_zero_fit_checks:
        raise GeometryQualificationError(
            f"zero-fit/shutdown receipt checks failed: {failed_zero_fit_checks}"
        )
    rows = artifact_manifest.get("files")
    if not isinstance(rows, list):
        raise GeometryQualificationError("artifact manifest files must be a list")
    declared_paths: set[str] = set()
    for row_value in rows:
        row = _mapping(row_value, "artifact row")
        relative = row.get("path")
        if not isinstance(relative, str) or relative in declared_paths or "\\" in relative:
            raise GeometryQualificationError("artifact path is invalid or duplicated")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
            raise GeometryQualificationError("artifact path is not safe POSIX relative")
        path = root.joinpath(*pure.parts).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise GeometryQualificationError("artifact path escapes run directory") from exc
        if path.is_symlink() or not path.is_file():
            raise GeometryQualificationError("artifact must be a regular non-symbolic file")
        if path.stat().st_size != row.get("size_bytes") or sha256_file(path) != row.get("sha256"):
            raise GeometryQualificationError(f"artifact hash/size mismatch: {relative}")
        declared_paths.add(relative)
    actual_paths = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    expected_paths = declared_paths | {"artifact_manifest.json", "completion_manifest.json"}
    if actual_paths != expected_paths:
        raise GeometryQualificationError("artifact inventory does not exactly cover run files")
    if completion.get("artifact_manifest_sha256") != sha256_file(root / "artifact_manifest.json"):
        raise GeometryQualificationError("completion does not bind artifact manifest")
    if completion.get("geometry_report_sha256") != sha256_file(root / "geometry_report.json"):
        raise GeometryQualificationError("completion does not bind geometry report")
    if completion.get("validation_sha256") != sha256_file(root / "validation.json"):
        raise GeometryQualificationError("completion does not bind validation")
    replay = _internal_validation(root, source_manifest, report)
    if replay["status"] != "pass" or canonical_json_sha256(replay) != canonical_json_sha256(
        stored_validation
    ):
        raise GeometryQualificationError("stored validation does not match exact replay")
    if report.get("status") != "pass" or completion.get("status") != "complete":
        raise GeometryQualificationError("software qualification gate did not pass")
    return {
        "status": "complete_read_only_revalidation",
        "run_directory": str(root),
        "fixture_count": report["fixture_count"],
        "passed_fixture_count": report["passed_fixture_count"],
        "model_fit_count": completion["model_fit_count"],
        "encoder_fit_count": completion["encoder_fit_count"],
        "validation_fit_count": stored_validation["validation_fit_count"],
        "task_owned_workers_remaining": worker_shutdown["task_owned_workers_remaining"],
        "task_owned_monitors_remaining": worker_shutdown["task_owned_monitors_remaining"],
        "automatic_follow_on_launched": completion["automatic_follow_on_launched"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--repository-root", required=True, type=Path)
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--protocol", required=True, type=Path)
    run_parser.add_argument("--governing-spec", required=True, type=Path)
    run_parser.add_argument("--synthesis", required=True, type=Path)
    run_parser.add_argument("--output", required=True, type=Path)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--run-directory", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        result = run_geometry_qualification(
            repository_root=args.repository_root,
            config_path=args.config,
            protocol_path=args.protocol,
            governing_spec_path=args.governing_spec,
            synthesis_path=args.synthesis,
            output_directory=args.output,
        )
    else:
        result = validate_geometry_qualification(args.run_directory)
    print(yaml.safe_dump(result, sort_keys=True).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
