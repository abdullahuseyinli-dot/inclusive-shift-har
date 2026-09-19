from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from inclusive_shift_har.models.same_attachment_directional_reference import (
    build_directional_support,
    build_joint_reference_head_parameters,
    compose_same_attachment_probabilities,
    directional_reference_evidence,
)
from inclusive_shift_har.models.same_attachment_reference_training import (
    MapPrior,
    OptimizerFit,
    WeightedScaler,
    apply_scaler,
    binary_objective_gradient,
    compile_directional_head,
    compose_probabilities,
    fit_map_prior,
    fit_multinomial_zero_sum,
    fit_weighted_scaler,
    map_sitting_probability,
    multinomial_objective_gradient,
    point_reference_features,
    predict_multinomial,
    update_map_posterior,
)


def _finite_difference(
    function: Callable[[np.ndarray], tuple[float, np.ndarray]], parameters: np.ndarray
) -> np.ndarray:
    result = np.empty_like(parameters)
    for index in range(parameters.size):
        step = np.zeros_like(parameters)
        step[index] = 1.0e-6
        result[index] = (function(parameters + step)[0] - function(parameters - step)[0]) / 2.0e-6
    return result


def test_weighted_scaler_uses_only_supplied_rows_and_weights() -> None:
    values = np.array([[0.0, 1.0], [2.0, 3.0], [100.0, 100.0]])
    scaler = fit_weighted_scaler(values[:2], [0.25, 0.75])
    assert np.allclose(scaler.mean, [1.5, 2.5])
    assert np.allclose(apply_scaler(values[:2], scaler).T @ np.array([0.25, 0.75]), 0.0)
    assert not np.any(scaler.mean == values[2])


def test_binary_gradient_matches_finite_difference() -> None:
    x = np.array([[0.1, -0.2], [1.2, 0.5], [-0.7, 1.4]])
    y = np.array([0, 1, 0])
    w = np.array([0.2, 0.3, 0.5])
    parameters = np.array([0.3, -0.4, 0.8])

    def function(value: np.ndarray) -> tuple[float, np.ndarray]:
        return binary_objective_gradient(value, x, y, w)

    _, analytic = function(parameters)
    assert np.allclose(analytic, _finite_difference(function, parameters), atol=1.0e-7)


def test_zero_sum_multinomial_gradient_includes_third_row_chain_rule() -> None:
    x = np.array([[0.1, -0.2], [1.2, 0.5], [-0.7, 1.4], [0.2, 0.8]])
    y = np.array([0, 1, 2, 2])
    w = np.array([0.2, 0.3, 0.1, 0.4])
    parameters = np.linspace(-0.4, 0.6, 6)

    def function(value: np.ndarray) -> tuple[float, np.ndarray]:
        return multinomial_objective_gradient(value, x, y, w)

    _, analytic = function(parameters)
    assert np.allclose(analytic, _finite_difference(function, parameters), atol=1.0e-7)


def test_zero_sum_fit_stores_all_eighteen_coefficients_for_five_features() -> None:
    rng = np.random.default_rng(20260914)
    x = rng.normal(size=(90, 5))
    y = np.tile(np.arange(3), 30)
    fit = fit_multinomial_zero_sum(x, y, np.full(90, 1 / 90))
    assert fit.coefficients.shape == (3, 6)
    assert np.allclose(fit.coefficients.sum(axis=0), 0.0, atol=1.0e-12)
    probabilities = predict_multinomial(x, fit)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_point_reference_uses_two_equal_components_per_posture() -> None:
    query = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    features = point_reference_features(
        query,
        [[1.0, 0.0, 0.0], [1.0, 0.1, 0.0]],
        [[0.0, 1.0, 0.0], [0.1, 1.0, 0.0]],
    )
    assert features[0, 0] > 0.0
    assert features[1, 0] < 0.0
    assert np.all(features[:, 1] <= 0.0)


def test_map_update_matches_unequal_variance_and_duplicate_support_fixtures() -> None:
    identity = WeightedScaler(np.zeros(5), np.ones(5))
    prior = MapPrior(
        means=np.array([[0, 2, 0, 2, 0], [0, 2, 0, 2, 0]], dtype=float),
        variances=np.array([[4, 1, 4, 1, 1], [4, 1, 4, 1, 1]], dtype=float),
        scaler=identity,
    )
    unequal = np.array([[2, 0, 2, 0, -1], [4, 2, 4, 2, 1]], dtype=float)
    result = update_map_posterior(prior, unequal, unequal)
    assert np.allclose(result.variances[0], [0.8, 0.5, 0.8, 0.5, 0.5])
    assert np.allclose(result.means[0], [2.4, 1.5, 2.4, 1.5, 0.0])
    duplicate = np.tile(np.array([3, -1, 0, 2, 0], dtype=float), (2, 1))
    duplicate_result = update_map_posterior(prior, duplicate, duplicate)
    assert duplicate_result.variances[0, 0] == pytest.approx(4 / 80001)
    assert duplicate_result.variances[0, 1] == pytest.approx(1 / 20001)
    assert duplicate_result.means[0, 0] == pytest.approx(240000 / 80001)
    assert duplicate_result.means[0, 1] == pytest.approx(-19998 / 20001)
    assert duplicate_result.variances.min() < 1.0e-4  # posterior is deliberately not floored


def test_weighted_map_prior_uses_bessel_denominator() -> None:
    base = np.array([0.0, 2.0, 4.0])
    features = np.column_stack((base, np.zeros((3, 4))))
    # Duplicate the fixture for both posture classes and at least two wearers.
    values = np.vstack((features, features))
    labels = np.array([0, 0, 0, 1, 1, 1])
    wearers = np.array(["a", "b", "c", "a", "b", "c"])
    weights = np.array([0.25, 0.25, 0.5, 0.25, 0.25, 0.5])
    prior = fit_map_prior(values, labels, wearers, weights)
    standardized = apply_scaler(values, prior.scaler)
    expected = np.sum(np.array([0.25, 0.25, 0.5]) * standardized[:3, 0] ** 2) / 0.625
    assert prior.variances[0, 0] == pytest.approx(expected)


def test_map_class_swap_requires_swapping_priors_and_supports() -> None:
    identity = WeightedScaler(np.zeros(5), np.ones(5))
    prior = MapPrior(
        np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=float), np.ones((2, 5)), identity
    )
    sit = np.tile(prior.means[0], (2, 1))
    stand = np.tile(prior.means[1], (2, 1))
    posterior = update_map_posterior(prior, sit, stand)
    q = map_sitting_probability(np.zeros((1, 5)), prior, posterior)[0]
    swapped_prior = MapPrior(prior.means[::-1].copy(), prior.variances[::-1].copy(), identity)
    swapped = update_map_posterior(swapped_prior, stand, sit)
    swapped_q = map_sitting_probability(np.zeros((1, 5)), swapped_prior, swapped)[0]
    assert q == pytest.approx(1.0 - swapped_q)
    assert compose_probabilities([0.2], [q]).sum() == pytest.approx(1.0)


def test_normalized_directional_coefficients_compile_to_existing_compositor() -> None:
    support = build_directional_support(
        attachment_id="a",
        sitting_directions=[[1, 0, 0], [1, 0.2, 0]],
        standing_directions=[[0, 1, 0], [0.1, 1, 0.1]],
        sitting_concentrations=[3, 8],
        standing_concentrations=[5, 9],
        sitting_dispersions_radians=[0.13, 0.22],
        standing_dispersions_radians=[0.27, 0.41],
    )
    query = np.array([[1, 0.3, 0.2], [0.1, 1, 0.4], [0.5, 0.2, 1.0]], dtype=float)
    raw_x = np.array([[1.2, 0.4], [2.1, 1.4], [0.7, 2.4]])
    mu_x, scale_x = np.array([0.4, 1.3]), np.array([1.4, 0.7])
    t = (raw_x - mu_x) / scale_x
    reference = directional_reference_evidence(query, support).features
    r, e, h_raw = reference.T
    design = np.column_stack((t, e, h_raw / math.pi, e * t[:, 0], e * t[:, 1]))
    mean = np.array([0.2, -0.3, 0.4, 0.1, -0.6, 0.7])
    scale = np.array([1.1, 0.8, 1.4, 0.3, 0.9, 1.3])
    motion = OptimizerFit(np.r_[-0.3, [0.2, -0.4, 0.5, 0.3, -0.6, 0.7]], 0, True, 0, 0, "fixture")
    posture = OptimizerFit(np.array([0.3, 0.8]), 0, True, 0, 0, "fixture")
    compiled = compile_directional_head(
        motion_fit=motion,
        motion_scaler=WeightedScaler(mean, scale),
        x_mean=mu_x,
        x_scale=scale_x,
        posture_fit=posture,
        ratio_rms=2.2,
        design_columns=("x1", "x2", "e", "h", "e_x1", "e_x2"),
    )
    parameters = build_joint_reference_head_parameters(**compiled)
    output = compose_same_attachment_probabilities(
        dynamics=raw_x,
        query_directions=query,
        query_attachment_ids=["a"] * 3,
        baseline_probabilities=np.full((3, 3), 1 / 3, dtype=np.float64),
        parameters=parameters,
        support=support,
        query_valid=np.ones(3, dtype=bool),
    ).probabilities
    motion_logit = motion.coefficients[0] + ((design - mean) / scale) @ motion.coefficients[1:]
    m = 1 / (1 + np.exp(-motion_logit))
    q = 1 / (1 + np.exp(-(r / 2.2 * ((1 - h_raw / math.pi) * 0.3 + h_raw / math.pi * 0.8))))
    assert np.max(np.abs(output - compose_probabilities(m, q))) < 1.0e-12
