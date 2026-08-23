from __future__ import annotations

import numpy as np
import pytest

from inclusive_shift_har.calibration import fit_temperature
from inclusive_shift_har.evaluation import (
    classification_report,
    holm_adjust,
    paired_participant_comparison,
    participant_bootstrap_interval,
)


def test_participant_report_weights_participants_equally() -> None:
    labels = np.array([0, 1, 0, 1, 0, 1])
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.2, 0.8],
            [0.7, 0.3],
        ]
    )
    participants = ["p1", "p1", "p1", "p1", "p2", "p2"]
    report = classification_report(
        labels,
        probabilities,
        participants,
        class_names=("sitting", "standing"),
    )
    assert report["primary"]["mean_participant_macro_f1"] == pytest.approx(0.5)
    assert report["primary"]["worst_participant_macro_f1"] == 0.0
    assert report["sample_count"] == 6
    assert report["participant_count"] == 2


def test_prediction_probability_alignment_is_enforced() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        classification_report(
            [0],
            [[0.8, 0.8]],
            ["p1"],
            class_names=("a", "b"),
        )


def test_participant_bootstrap_is_deterministic() -> None:
    values = {"p1": 0.4, "p2": 0.6, "p3": 0.8}
    first = participant_bootstrap_interval(values, resamples=500, seed=3)
    second = participant_bootstrap_interval(values, resamples=500, seed=3)
    assert first == second
    assert first["estimate"] == pytest.approx(0.6)


def test_paired_statistics_operate_on_participants() -> None:
    comparison = paired_participant_comparison(
        {"p1": 0.4, "p2": 0.5, "p3": 0.6},
        {"p1": 0.5, "p2": 0.7, "p3": 0.9},
    )
    assert comparison["participant_count"] == 3
    assert comparison["candidate_minus_reference_mean"] == pytest.approx(0.2)
    assert comparison["permutation"]["mode"] == "exact_sign_flip"


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    adjusted = holm_adjust([0.03, 0.01, 0.5])
    assert adjusted == pytest.approx([0.06, 0.03, 0.5])


def test_temperature_scaling_records_validation_lineage_and_reduces_nll() -> None:
    logits = np.array([[8.0, -1.0], [4.0, -2.0], [3.0, -1.0], [-2.0, 3.0]])
    labels = np.array([0, 1, 0, 1])
    calibrator = fit_temperature(
        logits,
        labels,
        validation_split_sha256="a" * 64,
    )
    probabilities = calibrator.probabilities(logits)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert calibrator.nll_after <= calibrator.nll_before
    assert calibrator.validation_split_sha256 == "a" * 64
