"""Hierarchical evidence-regularized extensions for frozen CTGR predictions.

The functions in this module are label-free at inference time.  Labels are used
only by experiment code to construct participant-level training utilities and
to fit calibration on participant-exclusive out-of-fold predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]


@dataclass(frozen=True, slots=True)
class GravityKinematicContext:
    """Per-window consistency between gravity direction and angular velocity."""

    features: FloatArray
    normalized_residual_p90: FloatArray
    valid_pair_fraction: FloatArray
    feature_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PhysicsReference:
    """Training-only threshold used for deterministic gravity intervention vetoes."""

    normalized_residual_p90_threshold: float
    minimum_valid_pair_fraction: float
    quantile: float


@dataclass(frozen=True, slots=True)
class PhysicsTrust:
    """Label-free reliability and hard trust decision for each window."""

    reliability: FloatArray
    trusted: BoolArray


@dataclass(frozen=True, slots=True)
class ResponderSignatures:
    """Robust unlabeled summaries, one row per participant or session."""

    participant_ids: StringArray
    features: FloatArray
    feature_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HeraRoutingResult:
    """Tri-state responder routing output and complete label-free diagnostics."""

    probabilities: FloatArray
    state_index: IntArray
    physics_vetoed: BoolArray
    participant_state: dict[str, str]
    participant_support_distance: dict[str, float]
    participant_base_advantage_lower_bound: dict[str, float]
    participant_broad_advantage_lower_bound: dict[str, float]


_KINEMATIC_FEATURE_NAMES = (
    "valid_pair_fraction",
    "kinematic_residual_median_rad_s",
    "kinematic_residual_p90_rad_s",
    "normalized_kinematic_residual_median",
    "normalized_kinematic_residual_p90",
    "gravity_direction_speed_median_rad_s",
    "gyro_implied_direction_speed_median_rad_s",
    "direction_prediction_alignment_error_median",
)

_RESPONDER_FEATURE_NAMES = (
    "ctgr_intervention_fraction",
    "broad_hard_change_fraction",
    "base_low_confidence_fraction",
    "base_confidence_p10",
    "base_stationary_mass_mean",
    "base_expert_hard_disagreement_fraction",
    "base_expert_js_mean",
    "base_expert_js_p90",
    "posture_log_odds_residual_median",
    "posture_log_odds_residual_iqr",
    "gravity_reliability_mean",
    "gravity_reliability_p10",
    "kinematic_residual_median",
    "kinematic_residual_p90",
    "physics_trusted_fraction",
)


def _probabilities(values: NDArray[np.floating[Any]], *, name: str) -> FloatArray:
    probability = np.asarray(values, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[0] == 0 or probability.shape[1] != 3:
        raise ValueError(f"{name} must be a non-empty [sample,3] matrix")
    if not np.isfinite(probability).all() or np.any(probability < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-7, rtol=1e-7):
        raise ValueError(f"{name} rows must sum to one")
    return probability


def _masked_quantile(values: FloatArray, mask: BoolArray, quantile: float) -> FloatArray:
    result = np.empty(values.shape[0], dtype=np.float64)
    for row in range(values.shape[0]):
        selected = values[row, mask[row]]
        result[row] = float(np.quantile(selected, quantile)) if selected.size else 1.0
    return result


def extract_gravity_kinematic_context(
    primary_signals: NDArray[np.floating[Any]],
    gravity: NDArray[np.floating[Any]],
    *,
    sampling_rate_hz: float,
    validity_mask: NDArray[np.bool_] | None = None,
) -> GravityKinematicContext:
    """Measure whether ``d(g_hat)/dt`` agrees with ``-omega x g_hat``.

    The relation holds for an inertial gravity vector represented in a rotating
    device frame.  It provides information that gravity angular speed alone
    cannot: whether the recorded gravity change is plausible given the gyroscope.
    """

    primary = np.asarray(primary_signals, dtype=np.float64)
    gravity_values = np.asarray(gravity, dtype=np.float64)
    if primary.ndim != 3 or primary.shape[1] < 3 or primary.shape[2] != 6:
        raise ValueError("primary signals must have shape [window,time>=3,6]")
    if gravity_values.shape != (*primary.shape[:2], 3):
        raise ValueError("gravity must align with primary signals as [window,time,3]")
    if not np.isfinite(primary).all() or not np.isfinite(gravity_values).all():
        raise ValueError("signals must be finite; missing samples belong in the validity mask")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0.0:
        raise ValueError("sampling rate must be finite and positive")
    if validity_mask is None:
        validity = np.ones((*primary.shape[:2], 9), dtype=np.bool_)
    else:
        validity = np.asarray(validity_mask, dtype=np.bool_)
        if validity.shape != (*primary.shape[:2], 9):
            raise ValueError("validity mask must align as [window,time,9]")

    gravity_norm = np.linalg.norm(gravity_values, axis=2)
    gravity_valid = np.all(validity[:, :, 6:9], axis=2) & (gravity_norm > 1e-8)
    gyro_valid = np.all(validity[:, :, 3:6], axis=2)
    unit = np.zeros_like(gravity_values)
    unit[gravity_valid] = gravity_values[gravity_valid] / gravity_norm[gravity_valid, None]

    pair_valid = np.zeros(gravity_valid.shape, dtype=np.bool_)
    pair_valid[:, 1:] = gravity_valid[:, 1:] & gravity_valid[:, :-1] & gyro_valid[:, 1:]
    derivative = np.zeros_like(unit)
    derivative[:, 1:] = (unit[:, 1:] - unit[:, :-1]) * sampling_rate_hz
    gyro = primary[:, :, 3:6]
    gyro_cross_gravity = np.cross(gyro, unit)
    predicted_derivative = -gyro_cross_gravity
    residual = np.linalg.norm(derivative - predicted_derivative, axis=2)
    derivative_speed = np.linalg.norm(derivative, axis=2)
    predicted_speed = np.linalg.norm(predicted_derivative, axis=2)
    normalized_residual = residual / np.maximum(0.25 + derivative_speed + predicted_speed, 1e-8)

    product = derivative_speed * predicted_speed
    cosine = np.ones_like(product)
    active = pair_valid & (product > 1e-10)
    cosine[active] = (
        np.sum(derivative[active] * predicted_derivative[active], axis=1) / product[active]
    )
    alignment_error = 0.5 * (1.0 - np.clip(cosine, -1.0, 1.0))

    valid_fraction = pair_valid.mean(axis=1)
    residual_median = _masked_quantile(residual, pair_valid, 0.50)
    residual_p90 = _masked_quantile(residual, pair_valid, 0.90)
    normalized_median = _masked_quantile(normalized_residual, pair_valid, 0.50)
    normalized_p90 = _masked_quantile(normalized_residual, pair_valid, 0.90)
    direction_speed_median = _masked_quantile(derivative_speed, pair_valid, 0.50)
    predicted_speed_median = _masked_quantile(predicted_speed, pair_valid, 0.50)
    alignment_median = _masked_quantile(alignment_error, pair_valid, 0.50)
    features = np.column_stack(
        (
            valid_fraction,
            residual_median,
            residual_p90,
            normalized_median,
            normalized_p90,
            direction_speed_median,
            predicted_speed_median,
            alignment_median,
        )
    )
    if features.shape != (primary.shape[0], len(_KINEMATIC_FEATURE_NAMES)):
        raise AssertionError("HERA kinematic feature contract changed")
    if not np.isfinite(features).all():
        raise AssertionError("HERA kinematic features must be finite")
    return GravityKinematicContext(
        features=np.asarray(features, dtype=np.float64),
        normalized_residual_p90=np.asarray(normalized_p90, dtype=np.float64),
        valid_pair_fraction=np.asarray(valid_fraction, dtype=np.float64),
        feature_names=_KINEMATIC_FEATURE_NAMES,
    )


def fit_physics_reference(
    training_context: GravityKinematicContext,
    *,
    veto_quantile: float,
    minimum_valid_pair_fraction: float,
) -> PhysicsReference:
    """Fit one label-free physical-consistency threshold on training windows only."""

    if not 0.5 <= veto_quantile < 1.0:
        raise ValueError("physics-veto quantile must lie in [0.5,1)")
    if not 0.0 <= minimum_valid_pair_fraction <= 1.0:
        raise ValueError("minimum valid-pair fraction must lie in [0,1]")
    eligible = training_context.valid_pair_fraction >= minimum_valid_pair_fraction
    values = training_context.normalized_residual_p90[eligible]
    if values.size == 0:
        raise ValueError("training data contain no windows eligible for a physics reference")
    threshold = max(float(np.quantile(values, veto_quantile)), 1e-6)
    return PhysicsReference(threshold, minimum_valid_pair_fraction, veto_quantile)


def apply_physics_reference(
    context: GravityKinematicContext,
    reference: PhysicsReference,
) -> PhysicsTrust:
    """Apply a training-derived reference without labels or evaluation fitting."""

    ratio = context.normalized_residual_p90 / reference.normalized_residual_p90_threshold
    reliability = context.valid_pair_fraction * np.exp(-np.clip(ratio, 0.0, 40.0))
    trusted = (context.valid_pair_fraction >= reference.minimum_valid_pair_fraction) & (
        context.normalized_residual_p90 <= reference.normalized_residual_p90_threshold
    )
    return PhysicsTrust(
        reliability=np.asarray(np.clip(reliability, 0.0, 1.0), dtype=np.float64),
        trusted=np.asarray(trusted, dtype=np.bool_),
    )


def apply_posture_conditional_calibration(
    probabilities: NDArray[np.floating[Any]],
    *,
    temperature: float,
    posture_logit_offset: float,
) -> FloatArray:
    """Calibrate sitting/standing odds while preserving mobility mass exactly."""

    values = _probabilities(probabilities, name="probabilities")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not np.isfinite(posture_logit_offset):
        raise ValueError("posture logit offset must be finite")
    stationary = values[:, 1] + values[:, 2]
    log_odds = (
        np.log(np.clip(values[:, 1], 1e-12, 1.0)) - np.log(np.clip(values[:, 2], 1e-12, 1.0))
    ) / temperature + posture_logit_offset
    sitting_conditional = 1.0 / (1.0 + np.exp(-np.clip(log_odds, -40.0, 40.0)))
    result = values.copy()
    result[:, 1] = stationary * sitting_conditional
    result[:, 2] = stationary * (1.0 - sitting_conditional)
    result /= result.sum(axis=1, keepdims=True)
    return np.asarray(result, dtype=np.float64)


def average_probability_candidates(
    candidates: NDArray[np.floating[Any]],
) -> FloatArray:
    """Equal-average a fixed candidate set to marginalize selection uncertainty."""

    values = np.asarray(candidates, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] == 0 or values.shape[2] != 3:
        raise ValueError("candidate probabilities must have shape [candidate,sample,3]")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("candidate probabilities must be finite and non-negative")
    if not np.allclose(values.sum(axis=2), 1.0, atol=1e-7, rtol=1e-7):
        raise ValueError("every candidate probability row must sum to one")
    result = values.mean(axis=0)
    result /= result.sum(axis=1, keepdims=True)
    return np.asarray(result, dtype=np.float64)


def _jensen_shannon(left: FloatArray, right: FloatArray) -> FloatArray:
    midpoint = 0.5 * (left + right)
    return np.asarray(
        0.5
        * np.sum(
            left * (np.log(np.clip(left, 1e-12, 1.0)) - np.log(np.clip(midpoint, 1e-12, 1.0)))
            + right * (np.log(np.clip(right, 1e-12, 1.0)) - np.log(np.clip(midpoint, 1e-12, 1.0))),
            axis=1,
        ),
        dtype=np.float64,
    )


def build_responder_signatures(
    base_probability: NDArray[np.floating[Any]],
    expert_probability: NDArray[np.floating[Any]],
    ctgr_probability: NDArray[np.floating[Any]],
    broad_probability: NDArray[np.floating[Any]],
    participant_ids: NDArray[np.str_],
    *,
    gravity_reliability: NDArray[np.floating[Any]],
    kinematic_context: GravityKinematicContext,
    physics_trusted: NDArray[np.bool_],
    low_confidence_threshold: float,
) -> ResponderSignatures:
    """Pool label-free evidence at participant/session granularity."""

    base = _probabilities(base_probability, name="base probability")
    expert = _probabilities(expert_probability, name="expert probability")
    ctgr = _probabilities(ctgr_probability, name="CTGR probability")
    broad = _probabilities(broad_probability, name="broad probability")
    participants = np.asarray(participant_ids, dtype=np.str_)
    reliability = np.asarray(gravity_reliability, dtype=np.float64)
    trusted = np.asarray(physics_trusted, dtype=np.bool_)
    if expert.shape != base.shape or ctgr.shape != base.shape or broad.shape != base.shape:
        raise ValueError("responder probability paths must align")
    if participants.shape != (base.shape[0],) or reliability.shape != participants.shape:
        raise ValueError("participant IDs and reliability must align with probabilities")
    if trusted.shape != participants.shape or kinematic_context.features.shape[0] != base.shape[0]:
        raise ValueError("physics context must align with probabilities")
    if (
        not np.isfinite(reliability).all()
        or np.any((reliability < 0.0) | (reliability > 1.0))
        or not 0.0 <= low_confidence_threshold <= 1.0
    ):
        raise ValueError("responder reliability or threshold is invalid")

    js = _jensen_shannon(base, expert)
    base_posture_log_odds = np.log(np.clip(base[:, 1], 1e-12, 1.0)) - np.log(
        np.clip(base[:, 2], 1e-12, 1.0)
    )
    expert_posture_log_odds = np.log(np.clip(expert[:, 1], 1e-12, 1.0)) - np.log(
        np.clip(expert[:, 2], 1e-12, 1.0)
    )
    residual = expert_posture_log_odds - base_posture_log_odds
    unique = np.unique(participants)
    rows: list[list[float]] = []
    for participant in unique:
        selected = participants == participant
        base_confidence = base[selected].max(axis=1)
        participant_residual = residual[selected]
        participant_js = js[selected]
        participant_reliability = reliability[selected]
        participant_kinematic = kinematic_context.normalized_residual_p90[selected]
        rows.append(
            [
                float(np.mean(np.max(np.abs(ctgr[selected] - base[selected]), axis=1) > 1e-12)),
                float(np.mean(broad[selected].argmax(axis=1) != base[selected].argmax(axis=1))),
                float(np.mean(base_confidence < low_confidence_threshold)),
                float(np.quantile(base_confidence, 0.10)),
                float(np.mean(base[selected, 1] + base[selected, 2])),
                float(np.mean(base[selected].argmax(axis=1) != expert[selected].argmax(axis=1))),
                float(np.mean(participant_js)),
                float(np.quantile(participant_js, 0.90)),
                float(np.median(participant_residual)),
                float(
                    np.quantile(participant_residual, 0.75)
                    - np.quantile(participant_residual, 0.25)
                ),
                float(np.mean(participant_reliability)),
                float(np.quantile(participant_reliability, 0.10)),
                float(np.median(participant_kinematic)),
                float(np.quantile(participant_kinematic, 0.90)),
                float(np.mean(trusted[selected])),
            ]
        )
    features = np.asarray(rows, dtype=np.float64)
    if features.shape != (unique.size, len(_RESPONDER_FEATURE_NAMES)):
        raise AssertionError("HERA responder-signature feature contract changed")
    if not np.isfinite(features).all():
        raise AssertionError("HERA responder signatures must be finite")
    return ResponderSignatures(
        participant_ids=np.asarray(unique, dtype=np.str_),
        features=features,
        feature_names=_RESPONDER_FEATURE_NAMES,
    )


def standardized_nearest_support_distance(
    training_features: NDArray[np.floating[Any]],
    evaluation_features: NDArray[np.floating[Any]],
) -> FloatArray:
    """Return robust standardized nearest-neighbour distance divided by sqrt(d)."""

    training = np.asarray(training_features, dtype=np.float64)
    evaluation = np.asarray(evaluation_features, dtype=np.float64)
    if (
        training.ndim != 2
        or evaluation.ndim != 2
        or training.shape[0] < 2
        or training.shape[1] == 0
        or evaluation.shape[1] != training.shape[1]
    ):
        raise ValueError("support-distance feature matrices are incompatible")
    if not np.isfinite(training).all() or not np.isfinite(evaluation).all():
        raise ValueError("support-distance features must be finite")
    center = np.median(training, axis=0)
    q25 = np.quantile(training, 0.25, axis=0)
    q75 = np.quantile(training, 0.75, axis=0)
    robust_scale = (q75 - q25) / 1.349
    standard_scale = np.std(training, axis=0)
    scale = np.maximum(np.maximum(robust_scale, standard_scale), 1e-6)
    train_z = (training - center) / scale
    evaluate_z = (evaluation - center) / scale
    distance = np.linalg.norm(evaluate_z[:, None, :] - train_z[None, :, :], axis=2)
    return np.asarray(distance.min(axis=1) / np.sqrt(training.shape[1]), dtype=np.float64)


def apply_physics_veto(
    base_probability: NDArray[np.floating[Any]],
    candidate_probability: NDArray[np.floating[Any]],
    physics_trusted: NDArray[np.bool_],
) -> tuple[FloatArray, BoolArray]:
    """Fall back exactly to the base wherever gravity evidence is untrusted."""

    base = _probabilities(base_probability, name="base probability")
    candidate = _probabilities(candidate_probability, name="candidate probability")
    trusted = np.asarray(physics_trusted, dtype=np.bool_)
    if candidate.shape != base.shape or trusted.shape != (base.shape[0],):
        raise ValueError("physics-veto inputs must align")
    intervention = np.max(np.abs(candidate - base), axis=1) > 1e-12
    vetoed = intervention & ~trusted
    result = candidate.copy()
    result[vetoed] = base[vetoed]
    return np.asarray(result, dtype=np.float64), np.asarray(vetoed, dtype=np.bool_)


def apply_tri_state_responder_controller(
    base_probability: NDArray[np.floating[Any]],
    pulse_probability: NDArray[np.floating[Any]],
    broad_probability: NDArray[np.floating[Any]],
    participant_ids: NDArray[np.str_],
    signature_participant_ids: NDArray[np.str_],
    base_advantage_lower_bound: NDArray[np.floating[Any]],
    broad_advantage_lower_bound: NDArray[np.floating[Any]],
    support_distance: NDArray[np.floating[Any]],
    physics_trusted: NDArray[np.bool_],
    *,
    minimum_advantage: float,
    maximum_support_distance: float,
) -> HeraRoutingResult:
    """Choose OFF/PULSE/BROAD per participant, defaulting safely to PULSE."""

    base = _probabilities(base_probability, name="base probability")
    pulse = _probabilities(pulse_probability, name="pulse probability")
    broad = _probabilities(broad_probability, name="broad probability")
    participants = np.asarray(participant_ids, dtype=np.str_)
    signature_ids = np.asarray(signature_participant_ids, dtype=np.str_)
    base_bound = np.asarray(base_advantage_lower_bound, dtype=np.float64)
    broad_bound = np.asarray(broad_advantage_lower_bound, dtype=np.float64)
    support = np.asarray(support_distance, dtype=np.float64)
    trusted = np.asarray(physics_trusted, dtype=np.bool_)
    if pulse.shape != base.shape or broad.shape != base.shape:
        raise ValueError("tri-state probability paths must align")
    if participants.shape != (base.shape[0],) or trusted.shape != participants.shape:
        raise ValueError("tri-state participant and physics arrays must align")
    if (
        signature_ids.ndim != 1
        or len(set(signature_ids.tolist())) != signature_ids.size
        or base_bound.shape != signature_ids.shape
        or broad_bound.shape != signature_ids.shape
        or support.shape != signature_ids.shape
    ):
        raise ValueError("tri-state participant-level arrays must align and be unique")
    if set(participants.tolist()) != set(signature_ids.tolist()):
        raise ValueError("tri-state signatures must cover evaluation participants exactly")
    if (
        not np.isfinite(base_bound).all()
        or not np.isfinite(broad_bound).all()
        or not np.isfinite(support).all()
        or not np.isfinite(minimum_advantage)
        or minimum_advantage < 0.0
        or not np.isfinite(maximum_support_distance)
        or maximum_support_distance <= 0.0
    ):
        raise ValueError("tri-state routing settings must be finite and valid")

    result = pulse.copy()
    state_index = np.ones(base.shape[0], dtype=np.int64)
    participant_state: dict[str, str] = {}
    distance_record: dict[str, float] = {}
    base_record: dict[str, float] = {}
    broad_record: dict[str, float] = {}
    for index, participant in enumerate(signature_ids.tolist()):
        mask = participants == participant
        state = "PULSE"
        if support[index] <= maximum_support_distance:
            best = max(float(base_bound[index]), float(broad_bound[index]))
            if best > minimum_advantage:
                if float(base_bound[index]) >= float(broad_bound[index]):
                    state = "OFF"
                    result[mask] = base[mask]
                    state_index[mask] = 0
                else:
                    state = "BROAD"
                    result[mask] = broad[mask]
                    state_index[mask] = 2
        participant_state[participant] = state
        distance_record[participant] = float(support[index])
        base_record[participant] = float(base_bound[index])
        broad_record[participant] = float(broad_bound[index])

    intervention = np.max(np.abs(result - base), axis=1) > 1e-12
    vetoed = intervention & ~trusted
    result[vetoed] = base[vetoed]
    return HeraRoutingResult(
        probabilities=np.asarray(result, dtype=np.float64),
        state_index=state_index,
        physics_vetoed=np.asarray(vetoed, dtype=np.bool_),
        participant_state=participant_state,
        participant_support_distance=distance_record,
        participant_base_advantage_lower_bound=base_record,
        participant_broad_advantage_lower_bound=broad_record,
    )


__all__ = [
    "GravityKinematicContext",
    "HeraRoutingResult",
    "PhysicsReference",
    "PhysicsTrust",
    "ResponderSignatures",
    "apply_physics_reference",
    "apply_physics_veto",
    "apply_posture_conditional_calibration",
    "apply_tri_state_responder_controller",
    "average_probability_candidates",
    "build_responder_signatures",
    "extract_gravity_kinematic_context",
    "fit_physics_reference",
    "standardized_nearest_support_distance",
]
