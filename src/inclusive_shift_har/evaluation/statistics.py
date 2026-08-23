"""Participant-level uncertainty, paired tests, effect sizes, and multiplicity control."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata, wilcoxon  # type: ignore[import-untyped]


def participant_bootstrap_interval(
    participant_values: Mapping[str, float],
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 1729,
) -> dict[str, Any]:
    """Percentile CI for an equally weighted participant mean."""

    if len(participant_values) < 2 or resamples < 100:
        raise ValueError("bootstrap requires at least two participants and 100 resamples")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    participant_ids = sorted(participant_values)
    values = np.asarray([participant_values[key] for key in participant_ids], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("participant metrics must be finite")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(resamples, values.size))
    bootstrap_means = values[indices].mean(axis=1)
    alpha = 1.0 - confidence
    return {
        "estimate": float(values.mean()),
        "confidence": confidence,
        "lower": float(np.quantile(bootstrap_means, alpha / 2.0)),
        "upper": float(np.quantile(bootstrap_means, 1.0 - alpha / 2.0)),
        "resamples": resamples,
        "seed": seed,
        "cluster_unit": "participant",
        "participant_count": values.size,
    }


def _paired_arrays(
    reference: Mapping[str, float], candidate: Mapping[str, float]
) -> tuple[list[str], NDArray[np.float64], NDArray[np.float64]]:
    if set(reference) != set(candidate) or len(reference) < 2:
        raise ValueError("paired comparisons require the same two or more participant ids")
    participants = sorted(reference)
    baseline = np.asarray([reference[key] for key in participants], dtype=np.float64)
    proposed = np.asarray([candidate[key] for key in participants], dtype=np.float64)
    if not np.isfinite(baseline).all() or not np.isfinite(proposed).all():
        raise ValueError("paired metrics must be finite")
    return participants, baseline, proposed


def paired_participant_comparison(
    reference: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    permutation_samples: int = 100_000,
    seed: int = 2718,
) -> dict[str, Any]:
    """Two-sided paired sign-flip test, Wilcoxon test, and paired effect sizes."""

    participants, baseline, proposed = _paired_arrays(reference, candidate)
    differences = proposed - baseline
    observed = abs(float(differences.mean()))
    if differences.size <= 16:
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=differences.size)))
        permuted = np.abs((signs * differences).mean(axis=1))
        permutation_p = float(np.mean(permuted >= observed - 1e-15))
        permutation_mode = "exact_sign_flip"
        effective_samples = int(signs.shape[0])
    else:
        if permutation_samples < 1_000:
            raise ValueError("Monte Carlo paired test requires at least 1,000 samples")
        generator = np.random.default_rng(seed)
        signs = generator.choice(
            np.array([-1.0, 1.0]), size=(permutation_samples, differences.size)
        )
        permuted = np.abs((signs * differences).mean(axis=1))
        permutation_p = float((np.sum(permuted >= observed) + 1) / (permutation_samples + 1))
        permutation_mode = "monte_carlo_sign_flip"
        effective_samples = permutation_samples

    if np.allclose(differences, 0.0):
        wilcoxon_statistic = 0.0
        wilcoxon_p = 1.0
    else:
        wilcoxon_result = wilcoxon(
            differences,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )
        wilcoxon_statistic = float(wilcoxon_result.statistic)
        wilcoxon_p = float(wilcoxon_result.pvalue)
    standard_deviation = float(differences.std(ddof=1))
    paired_standardized_effect = (
        float(differences.mean() / standard_deviation) if standard_deviation > 0 else None
    )
    nonzero = differences[differences != 0]
    rank_biserial = None
    if nonzero.size:
        ranks = np.asarray(rankdata(np.abs(nonzero), method="average"), dtype=np.float64)
        positive = float(ranks[nonzero > 0].sum())
        negative = float(ranks[nonzero < 0].sum())
        rank_biserial = float((positive - negative) / (positive + negative))
    return {
        "participant_ids": participants,
        "participant_count": len(participants),
        "candidate_minus_reference_mean": float(differences.mean()),
        "candidate_minus_reference_median": float(np.median(differences)),
        "paired_standardized_effect": paired_standardized_effect,
        "rank_biserial_effect": rank_biserial,
        "permutation": {
            "mode": permutation_mode,
            "two_sided_p_value": permutation_p,
            "samples": effective_samples,
            "seed": None if permutation_mode == "exact_sign_flip" else seed,
        },
        "wilcoxon": {
            "statistic": wilcoxon_statistic,
            "two_sided_p_value": wilcoxon_p,
        },
        "unit": "participant",
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm family-wise error-rate adjustment in original input order."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a non-empty vector in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.maximum.accumulate((values.size - np.arange(values.size)) * values[order])
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return [float(value) for value in adjusted]
