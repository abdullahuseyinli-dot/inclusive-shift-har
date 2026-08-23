"""Validation-only scalar temperature calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]
from scipy.special import logsumexp, softmax  # type: ignore[import-untyped]


@dataclass(frozen=True)
class TemperatureCalibrator:
    """A fitted scalar temperature with mandatory validation-split lineage."""

    temperature: float
    validation_split_sha256: str
    sample_count: int
    nll_before: float
    nll_after: float

    def transform_logits(self, logits: NDArray[np.float64]) -> NDArray[np.float64]:
        array = np.asarray(logits, dtype=np.float64)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError("logits must be a finite rank-two array")
        return array / self.temperature

    def probabilities(self, logits: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(softmax(self.transform_logits(logits), axis=1), dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nll(logits: NDArray[np.float64], labels: NDArray[np.int64], temperature: float) -> float:
    scaled = logits / temperature
    return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(labels.size), labels]))


def fit_temperature(
    validation_logits: NDArray[np.float64],
    validation_labels: NDArray[np.int64],
    *,
    validation_split_sha256: str,
) -> TemperatureCalibrator:
    """Fit on validation logits only; the caller must supply immutable split provenance."""

    logits = np.asarray(validation_logits, dtype=np.float64)
    labels = np.asarray(validation_labels, dtype=np.int64)
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.size:
        raise ValueError("validation logits and labels are not aligned")
    if labels.size == 0 or not np.isfinite(logits).all():
        raise ValueError("validation calibration data must be non-empty and finite")
    if labels.min() < 0 or labels.max() >= logits.shape[1]:
        raise ValueError("validation label index lies outside logits")
    if len(validation_split_sha256) != 64:
        raise ValueError("validation split lineage must be a full SHA-256")

    result = minimize_scalar(
        lambda log_temperature: _nll(logits, labels, float(np.exp(log_temperature))),
        bounds=(-4.0, 4.0),
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"temperature optimization failed: {result.message}")
    temperature = float(np.exp(result.x))
    return TemperatureCalibrator(
        temperature=temperature,
        validation_split_sha256=validation_split_sha256,
        sample_count=labels.size,
        nll_before=_nll(logits, labels, 1.0),
        nll_after=_nll(logits, labels, temperature),
    )
