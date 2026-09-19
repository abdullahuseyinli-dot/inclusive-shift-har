from __future__ import annotations

from typing import Any

import numpy as np

from inclusive_shift_har.experiments.same_attachment_reference_comparison import (
    ARM_IDS,
    FIT_SCHEDULE,
    compare_reports,
    evaluate_equal_bout,
    practical_gate,
    schedule_receipt,
    training_window_weights,
    validate_supervised_admission,
)


def test_exact_finite_schedule_and_sharing_count() -> None:
    receipt = schedule_receipt()
    assert len(ARM_IDS) == 9
    assert len(FIT_SCHEDULE) == 72
    assert receipt["counts"] == {
        "binary_motion": 36,
        "binary_posture": 18,
        "multinomial": 6,
        "scalar_concentration": 12,
    }
    assert receipt["arm_fold_outputs"] == 54
    assert receipt["encoder_fits"] == 0
    assert [attempt.attempt_index for attempt in FIT_SCHEDULE] == list(range(72))


def test_synthetic_or_incomplete_physical_receipt_cannot_authorize_fits() -> None:
    blocked = validate_supervised_admission(
        {"evidence_status": "synthetic_fixture_not_physical_evidence"}
    )
    assert blocked["status"] == "blocked"
    assert blocked["fit_authorized"] is False
    receipt: dict[str, Any] = {
        name: True
        for name in (
            "signal_run_independently_validated",
            "qualification_manifest_verified",
            "canonicalization_receipt_verified",
            "human_authorization_and_consent_verified",
            "canonical_recordings_bound",
            "adjudicated_annotations_bound",
            "processing_frozen_before_human_outcomes",
            "original_pilot_gate_passed",
            "pooled_companion_gate_passed",
            "source_checkout_guard_passed",
        )
    }
    receipt.update(
        {
            "evidence_status": "real_physical_development_observations",
            "wearer_count": 6,
            "attachment_block_count": 24,
            "support_bout_count": 96,
            "query_bout_count": 168,
            "total_bout_count": 264,
        }
    )
    assert validate_supervised_admission(receipt)["status"] == "admitted"
    receipt["pooled_companion_gate_passed"] = False
    assert validate_supervised_admission(receipt)["status"] == "blocked"


def test_training_weights_are_equal_across_hierarchy_despite_unequal_windows() -> None:
    wearer_ids: list[str] = []
    classes: list[int] = []
    conditions: list[str] = []
    blocks: list[str] = []
    bouts: list[str] = []
    condition_map = {
        0: ("usual", "slow", "turning"),
        1: ("quiet", "upper_body_motion"),
        2: ("quiet", "upper_body_motion"),
    }
    for wearer_index in range(5):
        for class_index, names in condition_map.items():
            for condition in names:
                for block in range(4):
                    bout = f"w{wearer_index}:c{class_index}:{condition}:b{block}"
                    for _ in range(1 + block):
                        wearer_ids.append(f"w{wearer_index}")
                        classes.append(class_index)
                        conditions.append(condition)
                        blocks.append(f"w{wearer_index}:b{block}")
                        bouts.append(bout)
    weights = training_window_weights(
        wearer_ids=wearer_ids,
        class_indices=classes,
        condition_ids=conditions,
        block_ids=blocks,
        bout_ids=bouts,
    )
    assert np.isclose(weights.sum(), 1.0, atol=1.0e-12)
    by_wearer = {
        wearer: weights[np.array(wearer_ids) == wearer].sum() for wearer in set(wearer_ids)
    }
    assert np.allclose(list(by_wearer.values()), 1 / 5)
    unique_bouts = sorted(set(bouts))
    for bout in unique_bouts:
        count = sum(value == bout for value in bouts)
        row_weights = weights[np.array(bouts) == bout]
        assert np.allclose(row_weights, row_weights[0])
        assert row_weights.sum() == row_weights[0] * count


def _evaluation(probability_shift: float = 0.0) -> dict[str, object]:
    wearers: list[str] = []
    bouts: list[str] = []
    labels: list[int] = []
    probabilities: list[list[float]] = []
    for wearer in range(6):
        for bout in range(28):
            label = bout % 3
            for _ in range(1 + bout % 2):
                wearers.append(f"w{wearer}")
                bouts.append(f"w{wearer}:b{bout}")
                labels.append(label)
                row = np.full(3, 0.05)
                row[label] = 0.9 + probability_shift
                row /= row.sum()
                probabilities.append(row.tolist())
    return evaluate_equal_bout(
        wearer_ids=wearers, bout_ids=bouts, labels=labels, probabilities=probabilities
    )


def test_equal_bout_metric_and_paired_gate_contract() -> None:
    report = _evaluation()
    assert report["mean_participant_macro_f1"] == 1.0
    comparison = compare_reports(report, report)
    assert comparison["mean_difference"] == 0.0
    assert comparison["paired_95_percent_exhaustive_bootstrap_interval"] == [0.0, 0.0]
    assert practical_gate({name: comparison for name in ("Z", "A", "MAP")})["status"] == "fail"
