from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from inclusive_shift_har.models.same_attachment_directional_reference import (
    REFERENCE_FEATURE_NAMES,
    DirectionalReferenceError,
    DirectionalSupport,
    JointReferenceHeadParameters,
    build_directional_support,
    build_joint_reference_head_parameters,
    compose_same_attachment_probabilities,
    directional_reference_evidence,
    log_vmf_normalizer_3d,
    summarize_support_bout,
)


def _support(
    *,
    status: Literal["fresh", "stale", "invalid"] = "fresh",
    sitting_concentrations: np.ndarray | None = None,
    standing_concentrations: np.ndarray | None = None,
) -> DirectionalSupport:
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
    )


def _parameters(*, scale: float = 1.0) -> JointReferenceHeadParameters:
    return build_joint_reference_head_parameters(
        motion_intercept=-0.3 * scale,
        motion_dynamics_coefficients=np.asarray([0.7, -0.2]) * scale,
        motion_compatibility_coefficient=0.4 * scale,
        motion_dispersion_coefficient=-0.1 * scale,
        motion_interaction_coefficients=np.asarray([0.25, -0.15]) * scale,
        posture_ratio_coefficient=0.8 * scale,
        posture_dispersion_interaction_coefficient=-0.2 * scale,
    )


def _baseline(rows: int) -> NDArray[np.float64]:
    values = np.tile(np.asarray([0.2, 0.3, 0.5]), (rows, 1))
    if rows:
        values[-1] = [1.0, 0.0, 0.0]
    return values


def _proper_rotation() -> NDArray[np.float64]:
    axis = np.asarray([1.0, -2.0, 0.5], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    angle = 0.83
    skew = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.asarray(
        np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew),
        dtype=np.float64,
    )


def test_vmf_normalizer_is_finite_at_uniform_small_and_large_limits() -> None:
    concentration = np.asarray([0.0, 1.0e-10, 1.0e-3, 1.0, 500.0])
    result = log_vmf_normalizer_3d(concentration)
    assert np.isfinite(result).all()
    assert result[0] == -math.log(4.0 * math.pi)
    assert result[1] == pytest.approx(result[0], abs=1.0e-15)
    assert np.all(np.diff(result) <= 0.0)
    assert np.all(np.diff(result[1:]) < 0.0)
    with pytest.raises(DirectionalReferenceError, match="nonnegative"):
        log_vmf_normalizer_3d(np.asarray([-1.0]))


def test_uniform_broad_distribution_has_exact_neutral_ratio_and_compatibility() -> None:
    support = _support(
        sitting_concentrations=np.zeros(2),
        standing_concentrations=np.zeros(2),
    )
    queries = np.asarray([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    evidence = directional_reference_evidence(queries, support)
    np.testing.assert_allclose(evidence.features[:, :2], 0.0, atol=1.0e-15, rtol=0.0)
    assert evidence.features[:, 2].tolist() == pytest.approx([0.25, 0.25])


def test_equal_odds_retain_absolute_compatibility_as_separate_evidence() -> None:
    broad = _support(
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
    broad_evidence = directional_reference_evidence(query, broad)
    improbable_evidence = directional_reference_evidence(query, improbable)
    assert broad_evidence.features[0, 0] == pytest.approx(0.0)
    assert improbable_evidence.features[0, 0] == pytest.approx(0.0)
    assert broad_evidence.features[0, 1] == pytest.approx(0.0)
    assert improbable_evidence.features[0, 1] < -100.0


def test_common_proper_rotation_preserves_evidence_and_probabilities() -> None:
    support = _support()
    queries = np.asarray([[0.1, 0.2, 0.97], [-0.2, 0.9, 0.1], [0.8, 0.2, -0.1]])
    dynamics = np.asarray([[0.2, -0.1], [1.1, 0.7], [-0.5, 0.9]])
    rotation = _proper_rotation()
    rotated = build_directional_support(
        attachment_id=support.attachment_id,
        sitting_directions=support.directions[0] @ rotation.T,
        standing_directions=support.directions[1] @ rotation.T,
        sitting_concentrations=support.concentrations[0],
        standing_concentrations=support.concentrations[1],
        sitting_dispersions_radians=support.dispersions_radians[0],
        standing_dispersions_radians=support.dispersions_radians[1],
    )
    original_evidence = directional_reference_evidence(queries, support)
    rotated_evidence = directional_reference_evidence(queries @ rotation.T, rotated)
    np.testing.assert_allclose(
        original_evidence.log_likelihoods,
        rotated_evidence.log_likelihoods,
        atol=1.0e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        original_evidence.features,
        rotated_evidence.features,
        atol=1.0e-12,
        rtol=0.0,
    )
    original = compose_same_attachment_probabilities(
        dynamics=dynamics,
        query_directions=queries,
        query_attachment_ids=[support.attachment_id] * 3,
        baseline_probabilities=_baseline(3),
        parameters=_parameters(),
        support=support,
        query_valid=np.ones(3, dtype=np.bool_),
    )
    rotated_output = compose_same_attachment_probabilities(
        dynamics=dynamics,
        query_directions=queries @ rotation.T,
        query_attachment_ids=[support.attachment_id] * 3,
        baseline_probabilities=_baseline(3),
        parameters=_parameters(),
        support=rotated,
        query_valid=np.ones(3, dtype=np.bool_),
    )
    np.testing.assert_allclose(original.probabilities, rotated_output.probabilities, atol=1.0e-12)


def test_sitting_standing_swap_is_equivariant_with_fixed_odd_posture_head() -> None:
    support = _support()
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
    original_evidence = directional_reference_evidence(queries, support)
    swapped_evidence = directional_reference_evidence(queries, swapped)
    np.testing.assert_allclose(
        swapped_evidence.log_likelihoods,
        original_evidence.log_likelihoods[:, ::-1],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        swapped_evidence.features[:, 0],
        -original_evidence.features[:, 0],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        swapped_evidence.features[:, 1:],
        original_evidence.features[:, 1:],
        atol=1.0e-12,
    )
    original = compose_same_attachment_probabilities(
        dynamics=dynamics,
        query_directions=queries,
        query_attachment_ids=[support.attachment_id] * 2,
        baseline_probabilities=_baseline(2),
        parameters=_parameters(),
        support=support,
        query_valid=np.ones(2, dtype=np.bool_),
    )
    relabeled = compose_same_attachment_probabilities(
        dynamics=dynamics,
        query_directions=queries,
        query_attachment_ids=[support.attachment_id] * 2,
        baseline_probabilities=_baseline(2)[:, [0, 2, 1]],
        parameters=_parameters(),
        support=swapped,
        query_valid=np.ones(2, dtype=np.bool_),
    )
    np.testing.assert_allclose(relabeled.probabilities, original.probabilities[:, [0, 2, 1]])


def test_antipodal_anchors_are_valid_for_likelihood_despite_frame_degeneracy() -> None:
    support = build_directional_support(
        attachment_id="attachment-01",
        sitting_directions=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        standing_directions=np.asarray([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        sitting_concentrations=np.asarray([20.0, 20.0]),
        standing_concentrations=np.asarray([20.0, 20.0]),
        sitting_dispersions_radians=np.asarray([0.0, 0.0]),
        standing_dispersions_radians=np.asarray([0.0, 0.0]),
    )
    evidence = directional_reference_evidence(
        np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
        support,
    )
    assert evidence.features[0, 0] > 30.0
    assert evidence.features[1, 0] < -30.0
    assert np.isfinite(evidence.features).all()


def test_identical_class_anchors_are_finite_but_non_discriminative() -> None:
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
    assert evidence.features[0, 0] == pytest.approx(0.0)
    assert np.isfinite(evidence.features).all()


def test_joint_head_exercises_compatibility_dispersion_and_interactions() -> None:
    support = _support()
    dynamics = np.asarray([[0.4, -0.7]])
    query = np.asarray([[0.1, 0.2, 0.97]])
    parameters = _parameters()
    evidence = directional_reference_evidence(query, support)
    output = compose_same_attachment_probabilities(
        dynamics=dynamics,
        query_directions=query,
        query_attachment_ids=[support.attachment_id],
        baseline_probabilities=_baseline(1),
        parameters=parameters,
        support=support,
        query_valid=np.ones(1, dtype=np.bool_),
    )
    ratio, compatibility, dispersion = evidence.features[0]
    expected_motion_logit = (
        parameters.motion_intercept
        + float(dynamics[0] @ parameters.motion_dynamics_coefficients)
        + parameters.motion_compatibility_coefficient * compatibility
        + parameters.motion_dispersion_coefficient * dispersion
        + compatibility * float(dynamics[0] @ parameters.motion_interaction_coefficients)
    )
    expected_motion = 1.0 / (1.0 + math.exp(-expected_motion_logit))
    expected_posture_logit = ratio * (
        parameters.posture_ratio_coefficient
        + parameters.posture_dispersion_interaction_coefficient * dispersion
    )
    expected_sitting = 1.0 / (1.0 + math.exp(-expected_posture_logit))
    assert output.probabilities[0, 0] == pytest.approx(expected_motion)
    assert output.probabilities[0, 1] == pytest.approx((1.0 - expected_motion) * expected_sitting)
    assert output.reference_features.shape == (1, len(REFERENCE_FEATURE_NAMES))


def test_missing_stale_invalid_and_row_level_failures_copy_baseline_exactly() -> None:
    baseline = _baseline(5)
    dynamics = np.asarray([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [np.nan, 0.0], [0.0, 0.0]])
    directions = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    query_valid = np.asarray([True, False, True, True, True])
    identifiers = ["attachment-01", "attachment-01", "attachment-01", "attachment-01", "wrong"]
    active = compose_same_attachment_probabilities(
        dynamics=dynamics,
        query_directions=directions,
        query_attachment_ids=identifiers,
        baseline_probabilities=baseline,
        parameters=_parameters(),
        support=_support(),
        query_valid=query_valid,
    )
    assert active.used_reference.tolist() == [True, False, False, False, False]
    assert active.fallback_reasons == (
        "reference_used",
        "query_marked_invalid",
        "invalid_query_direction",
        "invalid_dynamics",
        "attachment_mismatch",
    )
    assert np.array_equal(
        active.probabilities[~active.used_reference], baseline[~active.used_reference]
    )
    assert (
        active.probabilities[~active.used_reference].tobytes()
        == baseline[~active.used_reference].tobytes()
    )
    for support, reason in (
        (None, "missing_support"),
        (_support(status="stale"), "stale_support"),
        (_support(status="invalid"), "invalid_support"),
    ):
        output = compose_same_attachment_probabilities(
            dynamics=np.zeros((5, 2)),
            query_directions=np.tile(np.asarray([0.0, 0.0, 1.0]), (5, 1)),
            query_attachment_ids=["attachment-01"] * 5,
            baseline_probabilities=baseline,
            parameters=_parameters(),
            support=support,
            query_valid=np.ones(5, dtype=np.bool_),
        )
        assert output.probabilities.dtype == baseline.dtype
        assert output.probabilities.tobytes() == baseline.tobytes()
        assert output.fallback_reasons == tuple(reason for _ in range(5))


def test_probability_normalization_is_finite_for_extreme_fixed_coefficients() -> None:
    rows = 4
    output = compose_same_attachment_probabilities(
        dynamics=np.asarray([[100.0, -100.0], [-100.0, 100.0], [0.0, 0.0], [2.0, 3.0]]),
        query_directions=np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        ),
        query_attachment_ids=["attachment-01"] * rows,
        baseline_probabilities=_baseline(rows),
        parameters=_parameters(scale=100.0),
        support=_support(),
        query_valid=np.ones(rows, dtype=np.bool_),
    )
    assert np.isfinite(output.probabilities).all()
    assert np.all((output.probabilities >= 0.0) & (output.probabilities <= 1.0))
    np.testing.assert_allclose(output.probabilities.sum(axis=1), 1.0, atol=1.0e-15)


def test_support_bout_summary_is_rotation_equivariant_and_rejects_zero_resultant() -> None:
    directions = np.asarray([[0.0, 0.0, 1.0], [0.1, 0.0, 0.995], [-0.1, 0.0, 0.995]])
    rotation = _proper_rotation()
    original, median, p90 = summarize_support_bout(directions)
    rotated, rotated_median, rotated_p90 = summarize_support_bout(directions @ rotation.T)
    np.testing.assert_allclose(rotated, original @ rotation.T, atol=1.0e-12)
    assert rotated_median == pytest.approx(median, abs=1.0e-12)
    assert rotated_p90 == pytest.approx(p90, abs=1.0e-12)
    with pytest.raises(DirectionalReferenceError, match="resultant"):
        summarize_support_bout(np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]))


def test_support_contract_rejects_malformed_or_silently_uncapped_inputs() -> None:
    with pytest.raises(DirectionalReferenceError, match="norm"):
        build_directional_support(
            attachment_id="attachment-01",
            sitting_directions=np.zeros((2, 3)),
            standing_directions=np.asarray([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
            sitting_concentrations=np.ones(2),
            standing_concentrations=np.ones(2),
            sitting_dispersions_radians=np.ones(2),
            standing_dispersions_radians=np.ones(2),
        )
    with pytest.raises(DirectionalReferenceError, match="maximum_concentration"):
        build_directional_support(
            attachment_id="attachment-01",
            sitting_directions=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            standing_directions=np.asarray([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
            sitting_concentrations=np.asarray([1.0, 501.0]),
            standing_concentrations=np.ones(2),
            sitting_dispersions_radians=np.ones(2),
            standing_dispersions_radians=np.ones(2),
        )
    with pytest.raises(DirectionalReferenceError, match="sum to one"):
        compose_same_attachment_probabilities(
            dynamics=np.zeros((1, 2)),
            query_directions=np.asarray([[0.0, 0.0, 1.0]]),
            query_attachment_ids=["attachment-01"],
            baseline_probabilities=np.asarray([[0.2, 0.2, 0.2]]),
            parameters=_parameters(),
            support=_support(),
            query_valid=np.ones(1, dtype=np.bool_),
        )


def test_float32_baseline_is_rejected_before_exact_fallback_claim() -> None:
    with pytest.raises(DirectionalReferenceError, match="native float64"):
        compose_same_attachment_probabilities(
            dynamics=np.zeros((1, 2)),
            query_directions=np.asarray([[0.0, 0.0, 1.0]]),
            query_attachment_ids=["attachment-01"],
            baseline_probabilities=np.asarray([[0.2, 0.3, 0.5]], dtype=np.float32),
            parameters=_parameters(),
            support=None,
            query_valid=np.ones(1, dtype=np.bool_),
        )


def test_nonfinite_computed_direction_norm_is_rejected() -> None:
    huge = np.asarray([[1.0e308, 1.0e308, 1.0e308]])
    with pytest.raises(DirectionalReferenceError, match="non-finite computed norm"):
        directional_reference_evidence(huge, _support())
    baseline = _baseline(1)
    output = compose_same_attachment_probabilities(
        dynamics=np.zeros((1, 2)),
        query_directions=huge,
        query_attachment_ids=["attachment-01"],
        baseline_probabilities=baseline,
        parameters=_parameters(),
        support=_support(),
        query_valid=np.ones(1, dtype=np.bool_),
    )
    assert output.fallback_reasons == ("invalid_query_direction",)
    assert output.probabilities.tobytes() == baseline.tobytes()


def test_nonfinite_or_boolean_norm_epsilon_is_rejected_at_public_boundaries() -> None:
    query = np.asarray([[0.0, 0.0, 1.0]])
    for invalid in (math.nan, math.inf, True):
        with pytest.raises(DirectionalReferenceError, match="finite and positive"):
            directional_reference_evidence(query, _support(), norm_epsilon=invalid)
        with pytest.raises(DirectionalReferenceError, match="finite and positive"):
            compose_same_attachment_probabilities(
                dynamics=np.zeros((1, 2)),
                query_directions=query,
                query_attachment_ids=["attachment-01"],
                baseline_probabilities=_baseline(1),
                parameters=_parameters(),
                support=_support(),
                query_valid=np.ones(1, dtype=np.bool_),
                norm_epsilon=invalid,
            )
