"""Prospective compact training primitives for same-attachment reference HAR.

These functions are cohort-agnostic mathematical components. They do not load data,
select participants, choose supports, or authorize an experiment. Callers must enforce
the physical, participant-exclusive and attempt-accounting contracts.
"""

# mypy: disable-error-code="import-untyped"

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.models.same_attachment_directional_reference import (
    log_vmf_normalizer_3d,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class ReferenceTrainingError(ValueError):
    """Raised when a prospective fitting contract is invalid or incomplete."""


@dataclass(frozen=True)
class WeightedScaler:
    mean: FloatArray
    scale: FloatArray


@dataclass(frozen=True)
class OptimizerFit:
    coefficients: FloatArray
    objective: float
    converged: bool
    iterations: int
    evaluations: int
    message: str


@dataclass(frozen=True)
class ConcentrationCalibration:
    kind: str
    value: float
    objective: float
    converged: bool
    evaluations: tuple[tuple[float, float], ...]
    cap_fraction: float
    message: str


@dataclass(frozen=True)
class MapPrior:
    means: FloatArray
    variances: FloatArray
    scaler: WeightedScaler


@dataclass(frozen=True)
class MapPosterior:
    means: FloatArray
    variances: FloatArray


def _float_matrix(value: Any, name: str, columns: int | None = None) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or (columns is not None and result.shape[1] != columns):
        suffix = "" if columns is None else f" with {columns} columns"
        raise ReferenceTrainingError(f"{name} must be a matrix{suffix}")
    if not np.isfinite(result).all():
        raise ReferenceTrainingError(f"{name} must be finite")
    return result


def _weights(value: Any, rows: int) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (rows,) or not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ReferenceTrainingError("weights must be finite, positive and row-aligned")
    return result


def _binary_labels(value: Any, rows: int) -> FloatArray:
    result = np.asarray(value)
    if result.shape != (rows,) or not np.all(np.isin(result, [0, 1])):
        raise ReferenceTrainingError("binary labels must be row-aligned zeros and ones")
    return result.astype(np.float64)


def fit_weighted_scaler(values: Any, weights: Any) -> WeightedScaler:
    matrix = _float_matrix(values, "values")
    row_weights = _weights(weights, matrix.shape[0])
    normalized = row_weights / row_weights.sum()
    mean = normalized @ matrix
    variance = normalized @ ((matrix - mean) ** 2)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale = np.where(scale == 0.0, 1.0, scale)
    return WeightedScaler(np.asarray(mean, dtype=np.float64), np.asarray(scale, dtype=np.float64))


def apply_scaler(values: Any, scaler: WeightedScaler) -> FloatArray:
    matrix = _float_matrix(values, "values", scaler.mean.size)
    if scaler.scale.shape != scaler.mean.shape or np.any(scaler.scale <= 0.0):
        raise ReferenceTrainingError("scaler is invalid")
    result = (matrix - scaler.mean) / scaler.scale
    if not np.isfinite(result).all():
        raise ReferenceTrainingError("scaled values are not finite")
    return np.asarray(result, dtype=np.float64)


def binary_objective_gradient(
    parameters: Any,
    features: Any,
    labels: Any,
    weights: Any,
    *,
    penalty: float = 0.01,
    include_intercept: bool = True,
) -> tuple[float, FloatArray]:
    matrix = _float_matrix(features, "features")
    target = _binary_labels(labels, matrix.shape[0])
    row_weights = _weights(weights, matrix.shape[0])
    coefficients = np.asarray(parameters, dtype=np.float64)
    expected = matrix.shape[1] + int(include_intercept)
    if coefficients.shape != (expected,) or not np.isfinite(coefficients).all():
        raise ReferenceTrainingError("binary coefficients have the wrong shape or are nonfinite")
    if not math.isfinite(penalty) or penalty < 0.0:
        raise ReferenceTrainingError("penalty must be finite and nonnegative")
    slopes = coefficients[1:] if include_intercept else coefficients
    logits = matrix @ slopes + (coefficients[0] if include_intercept else 0.0)
    residual = (1.0 / (1.0 + np.exp(-np.clip(logits, -709.0, 709.0))) - target) * row_weights
    loss = float(np.sum(row_weights * (np.logaddexp(0.0, logits) - target * logits)))
    loss += 0.5 * penalty * float(slopes @ slopes)
    slope_gradient = matrix.T @ residual + penalty * slopes
    gradient = (
        np.concatenate(([float(residual.sum())], slope_gradient))
        if include_intercept
        else slope_gradient
    )
    return loss, np.asarray(gradient, dtype=np.float64)


def fit_binary_logistic(
    features: Any,
    labels: Any,
    weights: Any,
    *,
    penalty: float = 0.01,
    include_intercept: bool = True,
    nonnegative: bool = False,
) -> OptimizerFit:
    matrix = _float_matrix(features, "features")
    target = _binary_labels(labels, matrix.shape[0])
    row_weights = _weights(weights, matrix.shape[0])
    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover - dependency qualification handles this
        raise ReferenceTrainingError("SciPy is required for the frozen optimizer") from exc
    size = matrix.shape[1] + int(include_intercept)
    bounds = [(0.0, None)] * size if nonnegative else None

    def objective(parameters: FloatArray) -> tuple[float, FloatArray]:
        return binary_objective_gradient(
            parameters,
            matrix,
            target,
            row_weights,
            penalty=penalty,
            include_intercept=include_intercept,
        )

    result = minimize(
        objective,
        np.zeros(size, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 1000, "gtol": 1.0e-8, "ftol": 1.0e-12},
    )
    fit = OptimizerFit(
        coefficients=np.asarray(result.x, dtype=np.float64),
        objective=float(result.fun),
        converged=bool(result.success),
        iterations=int(result.nit),
        evaluations=int(result.nfev),
        message=str(result.message),
    )
    if (
        not fit.converged
        or not math.isfinite(fit.objective)
        or not np.isfinite(fit.coefficients).all()
    ):
        raise ReferenceTrainingError(f"binary optimization incomplete: {fit.message}")
    return fit


def fit_constrained_posture(
    ratio: Any, dispersion_fraction: Any, sitting_labels: Any, weights: Any
) -> OptimizerFit:
    r = np.asarray(ratio, dtype=np.float64)
    h = np.asarray(dispersion_fraction, dtype=np.float64)
    if r.ndim != 1 or h.shape != r.shape or not np.isfinite(r).all() or not np.isfinite(h).all():
        raise ReferenceTrainingError("posture ratio and dispersion must be finite aligned vectors")
    if np.any((h < 0.0) | (h > 1.0)):
        raise ReferenceTrainingError("dispersion fraction must lie in [0,1]")
    design = np.column_stack((r * (1.0 - h), r * h))
    return fit_binary_logistic(
        design, sitting_labels, weights, include_intercept=False, nonnegative=True
    )


def predict_binary(
    features: Any, fit: OptimizerFit, *, include_intercept: bool = True
) -> FloatArray:
    matrix = _float_matrix(features, "features")
    expected = matrix.shape[1] + int(include_intercept)
    if fit.coefficients.shape != (expected,):
        raise ReferenceTrainingError("fit and prediction feature shapes differ")
    logits = matrix @ (fit.coefficients[1:] if include_intercept else fit.coefficients)
    if include_intercept:
        logits += fit.coefficients[0]
    result = np.empty_like(logits)
    nonnegative = logits >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
    exponential = np.exp(logits[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return np.asarray(result, dtype=np.float64)


def multinomial_objective_gradient(
    parameters: Any, features: Any, labels: Any, weights: Any, *, penalty: float = 0.01
) -> tuple[float, FloatArray]:
    matrix = _float_matrix(features, "features")
    target = np.asarray(labels)
    if target.shape != (matrix.shape[0],) or not np.all(np.isin(target, [0, 1, 2])):
        raise ReferenceTrainingError("multinomial labels must be class indices 0,1,2")
    target = target.astype(np.int64)
    row_weights = _weights(weights, matrix.shape[0])
    free = np.asarray(parameters, dtype=np.float64)
    if free.shape != (2 * (matrix.shape[1] + 1),) or not np.isfinite(free).all():
        raise ReferenceTrainingError("multinomial free parameters have the wrong shape")
    augmented = np.column_stack((np.ones(matrix.shape[0]), matrix))
    first_two = free.reshape(2, augmented.shape[1])
    stored = np.vstack((first_two, -first_two.sum(axis=0)))
    logits = augmented @ stored.T
    logits -= logits.max(axis=1, keepdims=True)
    log_denominator = np.log(np.exp(logits).sum(axis=1))
    log_probability = logits - log_denominator[:, None]
    loss = -float(np.sum(row_weights * log_probability[np.arange(matrix.shape[0]), target]))
    loss += 0.5 * penalty * float(np.sum(stored[:, 1:] ** 2))
    probability = np.exp(log_probability)
    probability[np.arange(matrix.shape[0]), target] -= 1.0
    stored_gradient = (probability * row_weights[:, None]).T @ augmented
    stored_gradient[:, 1:] += penalty * stored[:, 1:]
    free_gradient = stored_gradient[:2] - stored_gradient[2]
    return loss, np.asarray(free_gradient.ravel(), dtype=np.float64)


def fit_multinomial_zero_sum(features: Any, labels: Any, weights: Any) -> OptimizerFit:
    matrix = _float_matrix(features, "features")
    target = np.asarray(labels)
    row_weights = _weights(weights, matrix.shape[0])
    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover
        raise ReferenceTrainingError("SciPy is required for the frozen optimizer") from exc

    def objective(parameters: FloatArray) -> tuple[float, FloatArray]:
        return multinomial_objective_gradient(parameters, matrix, target, row_weights)

    result = minimize(
        objective,
        np.zeros(2 * (matrix.shape[1] + 1), dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000, "gtol": 1.0e-8, "ftol": 1.0e-12},
    )
    free = np.asarray(result.x, dtype=np.float64).reshape(2, matrix.shape[1] + 1)
    stored = np.vstack((free, -free.sum(axis=0)))
    fit = OptimizerFit(
        stored,
        float(result.fun),
        bool(result.success),
        int(result.nit),
        int(result.nfev),
        str(result.message),
    )
    if not fit.converged or not np.isfinite(fit.coefficients).all():
        raise ReferenceTrainingError(f"multinomial optimization incomplete: {fit.message}")
    return fit


def predict_multinomial(features: Any, fit: OptimizerFit) -> FloatArray:
    matrix = _float_matrix(features, "features")
    if fit.coefficients.shape != (3, matrix.shape[1] + 1):
        raise ReferenceTrainingError("multinomial stored coefficients have the wrong shape")
    logits = np.column_stack((np.ones(matrix.shape[0]), matrix)) @ fit.coefficients.T
    logits -= logits.max(axis=1, keepdims=True)
    exponent = np.exp(logits)
    return np.asarray(exponent / exponent.sum(axis=1, keepdims=True), dtype=np.float64)


def compose_probabilities(mobility_probability: Any, sitting_probability: Any) -> FloatArray:
    mobility = np.asarray(mobility_probability, dtype=np.float64)
    sitting = np.asarray(sitting_probability, dtype=np.float64)
    if mobility.ndim != 1 or sitting.shape != mobility.shape:
        raise ReferenceTrainingError("motion and posture probabilities must be aligned vectors")
    if not np.isfinite(mobility).all() or not np.isfinite(sitting).all():
        raise ReferenceTrainingError("probabilities must be finite")
    if np.any((mobility < 0) | (mobility > 1) | (sitting < 0) | (sitting > 1)):
        raise ReferenceTrainingError("probabilities must lie in [0,1]")
    result = np.column_stack(
        (mobility, (1.0 - mobility) * sitting, (1.0 - mobility) * (1.0 - sitting))
    )
    return np.asarray(result, dtype=np.float64)


def point_reference_features(
    query_directions: Any,
    sitting_components: Any,
    standing_components: Any,
    *,
    norm_epsilon: float = 1.0e-12,
) -> FloatArray:
    query = _float_matrix(query_directions, "query_directions", 3)
    sitting = _float_matrix(sitting_components, "sitting_components", 3)
    standing = _float_matrix(standing_components, "standing_components", 3)
    if sitting.shape[0] != 2 or standing.shape[0] != 2:
        raise ReferenceTrainingError("point references require two components per posture")

    def normalized_mean(values: FloatArray) -> FloatArray:
        mean = values.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if not math.isfinite(norm) or norm <= norm_epsilon:
            raise ReferenceTrainingError("point reference resultant is degenerate")
        return np.asarray(mean / norm, dtype=np.float64)

    sit = normalized_mean(sitting)
    stand = normalized_mean(standing)
    qnorm = np.linalg.norm(query, axis=1)
    if np.any(qnorm <= norm_epsilon):
        raise ReferenceTrainingError("query direction is degenerate")
    unit = query / qnorm[:, None]
    d_sit = np.arccos(np.clip(unit @ sit, -1.0, 1.0))
    d_stand = np.arccos(np.clip(unit @ stand, -1.0, 1.0))
    return np.column_stack((d_stand - d_sit, -np.minimum(d_sit, d_stand)))


def _mixture_log_density(query: FloatArray, support: FloatArray, kappa: FloatArray) -> FloatArray:
    normalizer = log_vmf_normalizer_3d(kappa)
    components = (query[:, None, :] * support).sum(axis=2) * kappa + normalizer
    return np.asarray(
        np.logaddexp(components[:, 0], components[:, 1]) - math.log(2.0),
        dtype=np.float64,
    )


def directional_features_batch(
    query_directions: Any,
    support_directions: Any,
    concentrations: Any,
    dispersions_radians: Any,
) -> FloatArray:
    query = _float_matrix(query_directions, "query_directions", 3)
    support = np.asarray(support_directions, dtype=np.float64)
    kappa = np.asarray(concentrations, dtype=np.float64)
    dispersion = np.asarray(dispersions_radians, dtype=np.float64)
    expected_support = (query.shape[0], 2, 2, 3)
    if support.shape != expected_support or kappa.shape != (query.shape[0], 2, 2):
        raise ReferenceTrainingError("directional support arrays have the wrong shape")
    if dispersion.shape != (query.shape[0], 2, 2):
        raise ReferenceTrainingError("directional dispersion array has the wrong shape")
    if (
        not np.isfinite(support).all()
        or not np.isfinite(kappa).all()
        or not np.isfinite(dispersion).all()
    ):
        raise ReferenceTrainingError("directional inputs must be finite")
    norms = np.linalg.norm(query, axis=1)
    support_norms = np.linalg.norm(support, axis=3)
    if np.any(norms <= 1.0e-12) or not np.allclose(support_norms, 1.0, atol=1.0e-12, rtol=0):
        raise ReferenceTrainingError("directional inputs must be unit-normalizable")
    unit = query / norms[:, None]
    sit = _mixture_log_density(unit, support[:, 0], kappa[:, 0])
    stand = _mixture_log_density(unit, support[:, 1], kappa[:, 1])
    ratio = sit - stand
    compatibility = np.logaddexp(sit, stand) - math.log(2.0) + math.log(4.0 * math.pi)
    return np.column_stack((ratio, compatibility, dispersion.mean(axis=(1, 2))))


def kappa_from_spread(spread: Any, tau: float, bench_delta: float) -> FloatArray:
    sigma = np.asarray(spread, dtype=np.float64)
    if not np.isfinite(sigma).all() or np.any((sigma < 0.0) | (sigma > math.pi)):
        raise ReferenceTrainingError("support spread must lie in [0,pi]")
    if not math.isfinite(tau) or tau < 0.0 or not math.isfinite(bench_delta) or bench_delta <= 0.0:
        raise ReferenceTrainingError("tau must be nonnegative and bench_delta positive")
    return np.asarray(np.minimum(500.0, tau / (sigma**2 + bench_delta**2)), dtype=np.float64)


def _calibrate_scalar(
    objective: Callable[[float], float],
    *,
    lower: float,
    upper: float,
    kind: str,
    value_transform: Callable[[float], float],
    cap_fraction: Callable[[float], float],
) -> ConcentrationCalibration:
    try:
        from scipy.optimize import minimize_scalar
    except ImportError as exc:  # pragma: no cover
        raise ReferenceTrainingError("SciPy is required for concentration calibration") from exc
    evaluations: list[tuple[float, float]] = []

    def recorded(raw: float) -> float:
        value = value_transform(float(raw))
        score = float(objective(value))
        if not math.isfinite(score):
            raise ReferenceTrainingError("concentration objective is nonfinite")
        evaluations.append((value, score))
        return score

    result = minimize_scalar(
        recorded, bounds=(lower, upper), method="bounded", options={"xatol": 1.0e-8, "maxiter": 500}
    )
    candidates = list(evaluations)
    for raw in (lower, upper):
        recorded(raw)
    if kind == "variable_tau":
        candidates.append((0.0, float(objective(0.0))))
        evaluations.append(candidates[-1])
    best_value, best_score = min(candidates, key=lambda item: (item[1], item[0]))
    converged = bool(result.success)
    calibration = ConcentrationCalibration(
        kind,
        best_value,
        best_score,
        converged,
        tuple(evaluations),
        cap_fraction(best_value),
        str(result.message),
    )
    if not converged:
        raise ReferenceTrainingError(f"concentration calibration incomplete: {result.message}")
    return calibration


def calibrate_variable_concentration(
    query_directions: Any,
    support_directions: Any,
    support_spreads: Any,
    posture_labels: Any,
    weights: Any,
    *,
    bench_delta: float,
) -> ConcentrationCalibration:
    query = _float_matrix(query_directions, "query_directions", 3)
    support = np.asarray(support_directions, dtype=np.float64)
    spread = np.asarray(support_spreads, dtype=np.float64)
    labels = np.asarray(posture_labels)
    row_weights = _weights(weights, query.shape[0])
    if support.shape != (query.shape[0], 2, 2, 3) or spread.shape != (query.shape[0], 2, 2):
        raise ReferenceTrainingError("calibration supports are misaligned")
    if labels.shape != (query.shape[0],) or not np.all(np.isin(labels, [0, 1])):
        raise ReferenceTrainingError("posture labels must use sitting=0, standing=1")
    upper_tau = 500.0 * (math.pi**2 + bench_delta**2)

    def objective(tau: float) -> float:
        kappa = kappa_from_spread(spread, tau, bench_delta)
        unit = query / np.linalg.norm(query, axis=1)[:, None]
        sit = _mixture_log_density(unit, support[:, 0], kappa[:, 0])
        stand = _mixture_log_density(unit, support[:, 1], kappa[:, 1])
        chosen = np.where(labels == 0, sit, stand)
        return -float(np.sum(row_weights * chosen))

    return _calibrate_scalar(
        objective,
        lower=-12.0,
        upper=math.log(upper_tau),
        kind="variable_tau",
        value_transform=math.exp,
        cap_fraction=lambda tau: float(
            np.mean(kappa_from_spread(spread, tau, bench_delta) >= 500.0)
        ),
    )


def calibrate_common_concentration(
    query_directions: Any,
    support_directions: Any,
    posture_labels: Any,
    weights: Any,
) -> ConcentrationCalibration:
    query = _float_matrix(query_directions, "query_directions", 3)
    support = np.asarray(support_directions, dtype=np.float64)
    labels = np.asarray(posture_labels)
    row_weights = _weights(weights, query.shape[0])
    if support.shape != (query.shape[0], 2, 2, 3):
        raise ReferenceTrainingError("calibration supports are misaligned")
    if labels.shape != (query.shape[0],) or not np.all(np.isin(labels, [0, 1])):
        raise ReferenceTrainingError("posture labels must use sitting=0, standing=1")

    def objective(value: float) -> float:
        kappa = np.full((query.shape[0], 2, 2), value, dtype=np.float64)
        unit = query / np.linalg.norm(query, axis=1)[:, None]
        chosen = np.where(
            labels == 0,
            _mixture_log_density(unit, support[:, 0], kappa[:, 0]),
            _mixture_log_density(unit, support[:, 1], kappa[:, 1]),
        )
        return -float(np.sum(row_weights * chosen))

    return _calibrate_scalar(
        objective,
        lower=0.0,
        upper=500.0,
        kind="common_kappa",
        value_transform=float,
        cap_fraction=lambda value: float(value >= 500.0),
    )


def fit_map_prior(
    bout_features: Any,
    posture_labels: Any,
    wearer_ids: Any,
    bout_weights: Any,
    *,
    variance_floor: float = 1.0e-4,
) -> MapPrior:
    features = _float_matrix(bout_features, "bout_features", 5)
    labels = np.asarray(posture_labels)
    wearers = np.asarray(wearer_ids)
    weights = _weights(bout_weights, features.shape[0])
    if labels.shape != (features.shape[0],) or not np.all(np.isin(labels, [0, 1])):
        raise ReferenceTrainingError("MAP posture labels must be 0/1")
    if wearers.shape != (features.shape[0],) or len(set(wearers.tolist())) < 2:
        raise ReferenceTrainingError(
            "MAP prior needs aligned observations from at least two wearers"
        )
    scaler = fit_weighted_scaler(features, weights)
    standardized = apply_scaler(features, scaler)
    means: list[FloatArray] = []
    variances: list[FloatArray] = []
    for class_index in (0, 1):
        mask = labels == class_index
        selected_weights = weights[mask]
        selected_weights = selected_weights / selected_weights.sum()
        denominator = 1.0 - float(selected_weights @ selected_weights)
        if denominator <= 0.0:
            raise ReferenceTrainingError("MAP prior variance denominator is nonpositive")
        mean = selected_weights @ standardized[mask]
        variance = (selected_weights @ ((standardized[mask] - mean) ** 2)) / denominator
        means.append(mean)
        variances.append(np.maximum(variance, variance_floor))
    return MapPrior(
        np.asarray(means, dtype=np.float64), np.asarray(variances, dtype=np.float64), scaler
    )


def update_map_posterior(
    prior: MapPrior,
    sitting_support: Any,
    standing_support: Any,
    *,
    variance_floor: float = 1.0e-4,
) -> MapPosterior:
    means: list[FloatArray] = []
    variances: list[FloatArray] = []
    for support in (sitting_support, standing_support):
        values = apply_scaler(_float_matrix(support, "support", 5), prior.scaler)
        if values.shape[0] != 2:
            raise ReferenceTrainingError("MAP requires exactly two support bouts per posture")
        support_mean = values.mean(axis=0)
        support_variance = np.maximum(values.var(axis=0, ddof=1), variance_floor)
        class_index = len(means)
        posterior_variance = 1.0 / (1.0 / prior.variances[class_index] + 2.0 / support_variance)
        posterior_mean = posterior_variance * (
            prior.means[class_index] / prior.variances[class_index]
            + 2.0 * support_mean / support_variance
        )
        means.append(posterior_mean)
        variances.append(posterior_variance)
    return MapPosterior(
        np.asarray(means, dtype=np.float64), np.asarray(variances, dtype=np.float64)
    )


def map_sitting_probability(
    query_features: Any, prior: MapPrior, posterior: MapPosterior
) -> FloatArray:
    values = apply_scaler(_float_matrix(query_features, "query_features", 5), prior.scaler)
    if posterior.means.shape != (2, 5):
        raise ReferenceTrainingError("MAP posterior means have the wrong shape")
    logits = -np.sum((values[:, None, :] - posterior.means[None, :, :]) ** 2, axis=2)
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    probability /= probability.sum(axis=1, keepdims=True)
    return np.asarray(probability[:, 0], dtype=np.float64)


def compile_directional_head(
    *,
    motion_fit: OptimizerFit,
    motion_scaler: WeightedScaler,
    x_mean: Any,
    x_scale: Any,
    posture_fit: OptimizerFit,
    ratio_rms: float,
    design_columns: tuple[str, ...],
) -> dict[str, Any]:
    """Compile normalized B/D/D-additive/D-constant coefficients to raw module form."""
    allowed = ("x1", "x2", "e", "h", "e_x1", "e_x2")
    if any(column not in allowed for column in design_columns) or len(set(design_columns)) != len(
        design_columns
    ):
        raise ReferenceTrainingError("unsupported or duplicate directional design column")
    if motion_fit.coefficients.shape != (len(design_columns) + 1,):
        raise ReferenceTrainingError("motion coefficient shape differs from declared columns")
    if motion_scaler.mean.shape != (len(design_columns),):
        raise ReferenceTrainingError("motion scaler shape differs from declared columns")
    mu = np.asarray(x_mean, dtype=np.float64)
    scale = np.asarray(x_scale, dtype=np.float64)
    if mu.shape != (2,) or scale.shape != (2,) or np.any(scale <= 0.0):
        raise ReferenceTrainingError("x normalization must contain two positive scales")
    a = motion_fit.coefficients[1:] / motion_scaler.scale
    by_name = dict(zip(design_columns, a, strict=True))
    intercept = float(motion_fit.coefficients[0] - a @ motion_scaler.mean)
    for index, name in enumerate(("x1", "x2")):
        intercept -= by_name.get(name, 0.0) * mu[index] / scale[index]
    compatibility = by_name.get("e", 0.0)
    for index, name in enumerate(("e_x1", "e_x2")):
        compatibility -= by_name.get(name, 0.0) * mu[index] / scale[index]
    if posture_fit.coefficients.shape != (2,) or ratio_rms <= 0.0:
        raise ReferenceTrainingError("posture compilation inputs are invalid")
    return {
        "motion_intercept": intercept,
        "motion_dynamics_coefficients": [
            by_name.get("x1", 0.0) / scale[0],
            by_name.get("x2", 0.0) / scale[1],
        ],
        "motion_compatibility_coefficient": compatibility,
        "motion_dispersion_coefficient": by_name.get("h", 0.0) / math.pi,
        "motion_interaction_coefficients": [
            by_name.get("e_x1", 0.0) / scale[0],
            by_name.get("e_x2", 0.0) / scale[1],
        ],
        "posture_ratio_coefficient": float(posture_fit.coefficients[0] / ratio_rms),
        "posture_dispersion_interaction_coefficient": float(
            (posture_fit.coefficients[1] - posture_fit.coefficients[0]) / (math.pi * ratio_rms)
        ),
    }
