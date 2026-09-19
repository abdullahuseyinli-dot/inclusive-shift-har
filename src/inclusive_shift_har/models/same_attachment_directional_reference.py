"""No-fit same-attachment directional evidence and joint probability algebra.

This module deliberately contains no estimator fitting and accepts no activity labels.
It turns four explicitly qualified support-bout summaries into rotation-invariant
directional evidence, then optionally applies fixed coefficients for a small joint
motion/posture probability head.  Attachment freshness and validity are caller-supplied
metadata; they are never inferred from sensor values.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

POSTURE_ORDER = ("sitting", "standing")
REFERENCE_FEATURE_NAMES = (
    "posture_log_likelihood_ratio",
    "stationary_reference_compatibility",
    "mean_support_dispersion_radians",
)

_LOG_FOUR_PI = math.log(4.0 * math.pi)
_LOG_TWO = math.log(2.0)


class DirectionalReferenceError(ValueError):
    """Raised when directional-reference inputs violate the explicit contract."""


@dataclass(frozen=True)
class DirectionalSupport:
    """Exactly two independent support-bout components for each posture class."""

    attachment_id: str
    directions: FloatArray
    concentrations: FloatArray
    dispersions_radians: FloatArray
    status: Literal["fresh", "stale", "invalid"]


@dataclass(frozen=True)
class DirectionalEvidence:
    """Per-query directional likelihoods and three compact reference features."""

    log_likelihoods: FloatArray
    features: FloatArray


@dataclass(frozen=True)
class JointReferenceHeadParameters:
    """Fixed coefficients for the reviewable dynamics/reference probability head."""

    motion_intercept: float
    motion_dynamics_coefficients: FloatArray
    motion_compatibility_coefficient: float
    motion_dispersion_coefficient: float
    motion_interaction_coefficients: FloatArray
    posture_ratio_coefficient: float
    posture_dispersion_interaction_coefficient: float


@dataclass(frozen=True)
class ReferenceProbabilityOutput:
    """Probabilities plus explicit reference-use and fallback diagnostics."""

    probabilities: FloatArray
    reference_features: FloatArray
    used_reference: BoolArray
    fallback_reasons: tuple[str, ...]


def _finite_positive_scalar(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise DirectionalReferenceError(f"{name} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DirectionalReferenceError(f"{name} must be finite and positive") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise DirectionalReferenceError(f"{name} must be finite and positive")
    return result


def _as_finite_array(value: Any, shape: tuple[int, ...], name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise DirectionalReferenceError(f"{name} must have shape {shape}")
    if not np.isfinite(array).all():
        raise DirectionalReferenceError(f"{name} must be finite")
    return np.asarray(array, dtype=np.float64)


def _readonly_copy(value: FloatArray) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _normalize_rows(
    value: Any,
    *,
    expected_rows: int,
    name: str,
    norm_epsilon: float,
) -> FloatArray:
    norm_epsilon = _finite_positive_scalar(norm_epsilon, "norm_epsilon")
    rows = _as_finite_array(value, (expected_rows, 3), name)
    with np.errstate(over="ignore", invalid="ignore"):
        norms = np.linalg.norm(rows, axis=1)
    if not np.isfinite(norms).all():
        raise DirectionalReferenceError(f"{name} contains a non-finite computed norm")
    if np.any(norms <= norm_epsilon):
        raise DirectionalReferenceError(f"{name} contains a norm at or below epsilon")
    unit = np.asarray(rows / norms[:, None], dtype=np.float64)
    unit_norms = np.linalg.norm(unit, axis=1)
    if not np.isfinite(unit).all() or not np.allclose(unit_norms, 1.0, atol=1.0e-12, rtol=0.0):
        raise DirectionalReferenceError(f"{name} could not be normalized to unit length")
    return unit


def build_directional_support(
    *,
    attachment_id: str,
    sitting_directions: Any,
    standing_directions: Any,
    sitting_concentrations: Any,
    standing_concentrations: Any,
    sitting_dispersions_radians: Any,
    standing_dispersions_radians: Any,
    status: Literal["fresh", "stale", "invalid"] = "fresh",
    norm_epsilon: float = 1.0e-12,
    maximum_concentration: float = 500.0,
) -> DirectionalSupport:
    """Validate and bind four equal-bout directional mixture components.

    Concentrations are explicit inputs because their future mapping from measured
    dispersion must be trained and frozen on outer-training participants.  This
    no-fit module neither estimates nor silently clips them.  A concentration of
    zero is retained as the exact uniform-distribution limit.
    """

    if not isinstance(attachment_id, str) or not attachment_id.strip():
        raise DirectionalReferenceError("attachment_id must be a non-empty string")
    if status not in {"fresh", "stale", "invalid"}:
        raise DirectionalReferenceError("unsupported support status")
    norm_epsilon = _finite_positive_scalar(norm_epsilon, "norm_epsilon")
    maximum_concentration = _finite_positive_scalar(maximum_concentration, "maximum_concentration")

    directions = np.stack(
        (
            _normalize_rows(
                sitting_directions,
                expected_rows=2,
                name="sitting_directions",
                norm_epsilon=norm_epsilon,
            ),
            _normalize_rows(
                standing_directions,
                expected_rows=2,
                name="standing_directions",
                norm_epsilon=norm_epsilon,
            ),
        ),
        axis=0,
    )
    concentrations = np.stack(
        (
            _as_finite_array(sitting_concentrations, (2,), "sitting_concentrations"),
            _as_finite_array(standing_concentrations, (2,), "standing_concentrations"),
        ),
        axis=0,
    )
    if np.any(concentrations < 0.0) or np.any(concentrations > maximum_concentration):
        raise DirectionalReferenceError(
            "concentrations must be between zero and maximum_concentration"
        )
    dispersions = np.stack(
        (
            _as_finite_array(
                sitting_dispersions_radians,
                (2,),
                "sitting_dispersions_radians",
            ),
            _as_finite_array(
                standing_dispersions_radians,
                (2,),
                "standing_dispersions_radians",
            ),
        ),
        axis=0,
    )
    if np.any(dispersions < 0.0) or np.any(dispersions > math.pi):
        raise DirectionalReferenceError("dispersions must lie in [0, pi] radians")
    return DirectionalSupport(
        attachment_id=attachment_id,
        directions=_readonly_copy(np.asarray(directions, dtype=np.float64)),
        concentrations=_readonly_copy(np.asarray(concentrations, dtype=np.float64)),
        dispersions_radians=_readonly_copy(np.asarray(dispersions, dtype=np.float64)),
        status=status,
    )


def summarize_support_bout(
    window_directions: Any,
    *,
    norm_epsilon: float = 1.0e-12,
) -> tuple[FloatArray, float, float]:
    """Return normalized mean direction and median/p90 angular dispersion.

    This is a deterministic physical summary, not a concentration estimator.
    A zero resultant is invalid even though a valid vMF component may separately
    use concentration zero to express the broad uniform limit.
    """

    values = np.asarray(window_directions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
        raise DirectionalReferenceError("window_directions must have shape [window>=1, 3]")
    unit = _normalize_rows(
        values,
        expected_rows=values.shape[0],
        name="window_directions",
        norm_epsilon=norm_epsilon,
    )
    resultant = unit.mean(axis=0)
    norm = float(np.linalg.norm(resultant))
    if not math.isfinite(norm) or norm <= norm_epsilon:
        raise DirectionalReferenceError("support-bout resultant norm is at or below epsilon")
    direction = np.asarray(resultant / norm, dtype=np.float64)
    angles = np.arccos(np.clip(unit @ direction, -1.0, 1.0))
    return (
        _readonly_copy(direction),
        float(np.median(angles)),
        float(np.percentile(angles, 90.0)),
    )


def log_vmf_normalizer_3d(concentration: Any) -> FloatArray:
    """Return stable log(kappa / (4*pi*sinh(kappa))) for kappa >= 0."""

    values = np.asarray(concentration, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise DirectionalReferenceError("concentration must be finite and nonnegative")
    result = np.empty_like(values, dtype=np.float64)
    zero = values == 0.0
    small = (values > 0.0) & (values <= 1.0e-3)
    regular = values > 1.0e-3
    result[zero] = -_LOG_FOUR_PI
    if np.any(small):
        squared = values[small] ** 2
        result[small] = -_LOG_FOUR_PI - (squared / 6.0 - squared**2 / 180.0 + squared**3 / 2835.0)
    if np.any(regular):
        selected = values[regular]
        log_sinh = selected + np.log1p(-np.exp(-2.0 * selected)) - _LOG_TWO
        result[regular] = np.log(selected) - _LOG_FOUR_PI - log_sinh
    if not np.isfinite(result).all():
        raise DirectionalReferenceError("vMF normalizer produced a non-finite value")
    return np.asarray(result, dtype=np.float64)


def directional_reference_evidence(
    query_directions: Any,
    support: DirectionalSupport,
    *,
    norm_epsilon: float = 1.0e-12,
) -> DirectionalEvidence:
    """Compute equal-bout sitting/standing likelihoods and invariant features."""

    queries = np.asarray(query_directions, dtype=np.float64)
    if queries.ndim != 2 or queries.shape[1] != 3 or queries.shape[0] == 0:
        raise DirectionalReferenceError("query_directions must have shape [query>=1, 3]")
    unit_queries = _normalize_rows(
        queries,
        expected_rows=queries.shape[0],
        name="query_directions",
        norm_epsilon=norm_epsilon,
    )
    if support.status != "fresh":
        raise DirectionalReferenceError("directional evidence requires fresh valid support")
    if support.directions.shape != (2, 2, 3):
        raise DirectionalReferenceError("support directions contract changed")
    if support.concentrations.shape != (2, 2):
        raise DirectionalReferenceError("support concentrations contract changed")
    if support.dispersions_radians.shape != (2, 2):
        raise DirectionalReferenceError("support dispersions contract changed")

    class_log_likelihoods: list[FloatArray] = []
    for class_index in range(2):
        concentration = support.concentrations[class_index]
        component_log_density = (unit_queries @ support.directions[class_index].T) * concentration[
            None, :
        ] + log_vmf_normalizer_3d(concentration)[None, :]
        class_log_likelihoods.append(
            np.logaddexp(component_log_density[:, 0], component_log_density[:, 1]) - _LOG_TWO
        )
    log_likelihoods = np.stack(class_log_likelihoods, axis=1)
    ratio = log_likelihoods[:, 0] - log_likelihoods[:, 1]
    compatibility = (
        np.logaddexp(log_likelihoods[:, 0], log_likelihoods[:, 1]) - _LOG_TWO + _LOG_FOUR_PI
    )
    symmetric_dispersion = float(np.mean(support.dispersions_radians))
    features = np.column_stack(
        (
            ratio,
            compatibility,
            np.full(unit_queries.shape[0], symmetric_dispersion, dtype=np.float64),
        )
    )
    if not np.isfinite(log_likelihoods).all() or not np.isfinite(features).all():
        raise DirectionalReferenceError("directional evidence must be finite")
    return DirectionalEvidence(
        log_likelihoods=np.asarray(log_likelihoods, dtype=np.float64),
        features=np.asarray(features, dtype=np.float64),
    )


def build_joint_reference_head_parameters(
    *,
    motion_intercept: float,
    motion_dynamics_coefficients: Any,
    motion_compatibility_coefficient: float,
    motion_dispersion_coefficient: float,
    motion_interaction_coefficients: Any,
    posture_ratio_coefficient: float,
    posture_dispersion_interaction_coefficient: float,
) -> JointReferenceHeadParameters:
    """Validate fixed coefficients without fitting or hidden normalization."""

    scalars = (
        motion_intercept,
        motion_compatibility_coefficient,
        motion_dispersion_coefficient,
        posture_ratio_coefficient,
        posture_dispersion_interaction_coefficient,
    )
    if not all(math.isfinite(float(value)) for value in scalars):
        raise DirectionalReferenceError("joint-head scalar coefficients must be finite")
    dynamics = _as_finite_array(
        motion_dynamics_coefficients,
        (2,),
        "motion_dynamics_coefficients",
    )
    interactions = _as_finite_array(
        motion_interaction_coefficients,
        (2,),
        "motion_interaction_coefficients",
    )
    return JointReferenceHeadParameters(
        motion_intercept=float(motion_intercept),
        motion_dynamics_coefficients=_readonly_copy(dynamics),
        motion_compatibility_coefficient=float(motion_compatibility_coefficient),
        motion_dispersion_coefficient=float(motion_dispersion_coefficient),
        motion_interaction_coefficients=_readonly_copy(interactions),
        posture_ratio_coefficient=float(posture_ratio_coefficient),
        posture_dispersion_interaction_coefficient=float(
            posture_dispersion_interaction_coefficient
        ),
    )


def _sigmoid(value: FloatArray) -> FloatArray:
    result = np.empty_like(value, dtype=np.float64)
    nonnegative = value >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-value[nonnegative]))
    exponential = np.exp(value[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return np.asarray(result, dtype=np.float64)


def _validate_baseline_probabilities(value: Any, row_count: int) -> FloatArray:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.float64):
        raise DirectionalReferenceError(
            "baseline_probabilities must be a native float64 ndarray for exact fallback"
        )
    baseline = _as_finite_array(value, (row_count, 3), "baseline_probabilities")
    if np.any(baseline < 0.0) or np.any(baseline > 1.0):
        raise DirectionalReferenceError("baseline probabilities must lie in [0, 1]")
    if not np.allclose(baseline.sum(axis=1), 1.0, atol=1.0e-12, rtol=0.0):
        raise DirectionalReferenceError("baseline probability rows must sum to one")
    return baseline


def compose_same_attachment_probabilities(
    *,
    dynamics: Any,
    query_directions: Any,
    query_attachment_ids: Sequence[str],
    baseline_probabilities: Any,
    parameters: JointReferenceHeadParameters,
    support: DirectionalSupport | None,
    query_valid: Any,
    norm_epsilon: float = 1.0e-12,
) -> ReferenceProbabilityOutput:
    """Apply the fixed joint head only where fresh same-attachment support is valid.

    Class order is ``mobility, sitting, standing``.  The posture logit is odd in
    the sitting-vs-standing likelihood ratio and has no free intercept, which
    gives fixed-parameter sitting/standing label-swap equivariance.  Every row
    outside the explicit reference-use mask is copied byte-for-byte from the
    supplied baseline probability array.
    """

    norm_epsilon = _finite_positive_scalar(norm_epsilon, "norm_epsilon")

    dynamic_values = np.asarray(dynamics, dtype=np.float64)
    directions = np.asarray(query_directions, dtype=np.float64)
    if dynamic_values.ndim != 2 or dynamic_values.shape[1] != 2:
        raise DirectionalReferenceError("dynamics must have shape [query, 2]")
    row_count = dynamic_values.shape[0]
    if directions.shape != (row_count, 3):
        raise DirectionalReferenceError("query_directions must align as [query, 3]")
    if len(query_attachment_ids) != row_count or any(
        not isinstance(item, str) or not item for item in query_attachment_ids
    ):
        raise DirectionalReferenceError("query_attachment_ids must identify every query")
    baseline = _validate_baseline_probabilities(baseline_probabilities, row_count)
    declared_valid: BoolArray
    declared_valid = np.asarray(query_valid)
    if declared_valid.shape != (row_count,) or declared_valid.dtype.kind != "b":
        raise DirectionalReferenceError("query_valid must be a boolean [query] array")
    declared_valid = np.asarray(declared_valid, dtype=np.bool_)

    result = np.array(baseline, dtype=np.float64, copy=True)
    reference_features = np.zeros((row_count, len(REFERENCE_FEATURE_NAMES)), dtype=np.float64)
    used = np.zeros(row_count, dtype=np.bool_)
    reasons = ["missing_support"] * row_count
    if support is None:
        return ReferenceProbabilityOutput(result, reference_features, used, tuple(reasons))
    if support.status != "fresh":
        reason = "stale_support" if support.status == "stale" else "invalid_support"
        return ReferenceProbabilityOutput(
            result,
            reference_features,
            used,
            tuple(reason for _ in range(row_count)),
        )

    finite_direction = np.asarray(np.isfinite(directions).all(axis=1), dtype=np.bool_)
    with np.errstate(over="ignore", invalid="ignore"):
        direction_norm = np.linalg.norm(
            np.where(finite_direction[:, None], directions, 0.0),
            axis=1,
        )
    finite_direction_norm = np.asarray(np.isfinite(direction_norm), dtype=np.bool_)
    finite_dynamics = np.asarray(np.isfinite(dynamic_values).all(axis=1), dtype=np.bool_)
    attachment_match = np.asarray(
        [item == support.attachment_id for item in query_attachment_ids],
        dtype=np.bool_,
    )
    used = np.asarray(
        declared_valid
        & finite_direction
        & finite_direction_norm
        & (direction_norm > norm_epsilon)
        & finite_dynamics
        & attachment_match,
        dtype=np.bool_,
    )
    for index in range(row_count):
        if used[index]:
            reasons[index] = "reference_used"
        elif not declared_valid[index]:
            reasons[index] = "query_marked_invalid"
        elif (
            not finite_direction[index]
            or not finite_direction_norm[index]
            or direction_norm[index] <= norm_epsilon
        ):
            reasons[index] = "invalid_query_direction"
        elif not finite_dynamics[index]:
            reasons[index] = "invalid_dynamics"
        elif not attachment_match[index]:
            reasons[index] = "attachment_mismatch"
        else:
            raise AssertionError("unclassified reference fallback")

    if np.any(used):
        evidence = directional_reference_evidence(
            directions[used],
            support,
            norm_epsilon=norm_epsilon,
        )
        reference_features[used] = evidence.features
        active_dynamics = dynamic_values[used]
        ratio = evidence.features[:, 0]
        compatibility = evidence.features[:, 1]
        dispersion = evidence.features[:, 2]
        motion_logit = (
            parameters.motion_intercept
            + active_dynamics @ parameters.motion_dynamics_coefficients
            + parameters.motion_compatibility_coefficient * compatibility
            + parameters.motion_dispersion_coefficient * dispersion
            + compatibility * (active_dynamics @ parameters.motion_interaction_coefficients)
        )
        posture_logit = ratio * (
            parameters.posture_ratio_coefficient
            + parameters.posture_dispersion_interaction_coefficient * dispersion
        )
        mobility = _sigmoid(np.asarray(motion_logit, dtype=np.float64))
        sitting_given_stationary = _sigmoid(np.asarray(posture_logit, dtype=np.float64))
        active = np.empty((int(used.sum()), 3), dtype=np.float64)
        active[:, 0] = mobility
        active[:, 1] = (1.0 - mobility) * sitting_given_stationary
        active[:, 2] = 1.0 - active[:, 0] - active[:, 1]
        if (
            not np.isfinite(active).all()
            or np.any(active < 0.0)
            or np.any(active > 1.0)
            or not np.allclose(active.sum(axis=1), 1.0, atol=1.0e-15, rtol=0.0)
        ):
            raise DirectionalReferenceError("joint head produced invalid probabilities")
        result[used] = active

    if not np.array_equal(result[~used], baseline[~used]):
        raise AssertionError("fallback probabilities changed")
    return ReferenceProbabilityOutput(
        probabilities=np.asarray(result, dtype=np.float64),
        reference_features=reference_features,
        used_reference=used,
        fallback_reasons=tuple(reasons),
    )
