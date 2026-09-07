from __future__ import annotations

import copy
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from inclusive_shift_har.experiments.fog_decision_rule_probe import (
    GRID,
    analyse,
    decode_with_multipliers,
    method_report_from_decisions,
    partition_preflight,
    select_policy,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/fog_decision_rule_probe_v1.yaml"


def _config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _one_hot(labels: np.ndarray[Any, np.dtype[np.int64]]) -> np.ndarray[Any, Any]:
    return np.asarray(np.eye(3, dtype=np.float64)[labels], dtype=np.float64)


def test_frozen_config_rejects_grid_seed_and_gate_drift() -> None:
    config = _config()
    validate_config(config)
    mutations = (
        ("decision_rule", "standing_multipliers", [0.25, 1.0, 2.0]),
        ("resources", "maximum_fit_attempts", 26),
        ("final_gate", "minimum_participant_wins", 13),
        ("estimator", "inner_random_state", "outcome_selected"),
        ("runtime_contract", "numpy_version", "2.3.4"),
    )
    for section, key, value in mutations:
        changed = copy.deepcopy(config)
        changed[section][key] = value
        with pytest.raises(ValueError, match="config changed"):
            validate_config(changed)


def test_decoder_uses_explicit_label_confidence_and_fixed_ties() -> None:
    probabilities = np.asarray([[0.60, 0.35, 0.05], [0.50, 0.25, 0.25]])
    scores, decisions, confidence = decode_with_multipliers(
        probabilities, sitting=2.0, standing=1.0
    )
    assert np.array_equal(decisions, [1, 0])
    assert np.array_equal(confidence, [0.35, 0.50])
    assert scores[0, 1] == 0.70
    _, tied, tied_confidence = decode_with_multipliers(probabilities[1:], sitting=2.0, standing=2.0)
    assert np.array_equal(tied, [0])
    assert np.array_equal(tied_confidence, [0.50])


def test_explicit_report_changes_classification_but_not_proper_scores() -> None:
    roster = ["p1", "p2"]
    people = np.repeat(np.asarray(roster), 3)
    labels = np.tile(np.arange(3, dtype=np.int64), 2)
    probabilities = np.asarray(
        [
            [0.60, 0.35, 0.05],
            [0.51, 0.48, 0.01],
            [0.05, 0.10, 0.85],
            [0.60, 0.35, 0.05],
            [0.51, 0.48, 0.01],
            [0.05, 0.10, 0.85],
        ]
    )
    d0 = probabilities.argmax(axis=1)
    _, d1, _ = decode_with_multipliers(probabilities, sitting=2.0, standing=1.0)
    report0 = method_report_from_decisions(
        labels=labels,
        raw_probabilities=probabilities,
        decisions=d0,
        participant_ids=people,
        roster=roster,
    )
    report1 = method_report_from_decisions(
        labels=labels,
        raw_probabilities=probabilities,
        decisions=d1,
        participant_ids=people,
        roster=roster,
    )
    assert report0["pooled"]["confusion_matrix"] != report1["pooled"]["confusion_matrix"]
    assert report0["pooled"]["nll"] == report1["pooled"]["nll"]
    assert report0["pooled"]["multiclass_brier"] == report1["pooled"]["multiclass_brier"]
    assert report1["hard_label_source"] == "explicit_decisions_not_probability_argmax"


def test_selector_is_inner_only_and_identity_falls_back_deterministically() -> None:
    signature = inspect.signature(select_policy)
    assert tuple(signature.parameters) == (
        "probabilities",
        "labels",
        "participant_ids",
        "roster",
    )
    assert len(GRID) == len(set(GRID)) == 9
    roster = [f"p{i}" for i in range(6)]
    labels = np.tile(np.arange(3, dtype=np.int64), len(roster))
    people = np.repeat(np.asarray(roster), 3)
    selected = select_policy(_one_hot(labels), labels, people, roster)
    assert selected["candidate_count"] == 9
    assert selected["selected_multipliers"] == [1.0, 1.0, 1.0]
    assert selected["selection_reason"] == "identity_no_feasible_nonidentity_gain_above_tolerance"
    assert selected["bottom_30_participant_count"] == 2


def test_partition_preflight_has_exact_nesting_and_weights() -> None:
    folds = np.repeat(np.arange(5, dtype=np.int64), 6)
    participants = np.asarray([f"p{fold}-{row // 3}" for fold in range(5) for row in range(6)])
    labels = np.tile(np.arange(3, dtype=np.int64), 10)
    eligibility = np.ones(30, dtype=np.bool_)
    report = partition_preflight(
        labels=labels, participants=participants, folds=folds, eligibility=eligibility
    )
    assert report["partition_count"] == 25
    assert [row["role"] for row in report["rows"][:5]] == ["outer"] * 5
    assert [row["role"] for row in report["rows"][5:]] == ["inner"] * 20
    for row in report["rows"]:
        assert set(row["training_participants"]).isdisjoint(row["evaluation_participants"])
        assert row["training_class_counts"][0] > 0
        assert len(row["training_participant_class_counts"]) == len(row["training_participants"])
        assert len(row["evaluation_scored_participant_class_counts"]) == len(
            row["evaluation_participants"]
        )
        weights = row["sample_weight_summary"]
        assert weights["mean"] == pytest.approx(1.0)
        assert weights["participant_total_minimum"] == pytest.approx(
            weights["participant_total_maximum"]
        )


def test_analysis_uses_explicit_decisions_and_requires_both_controls() -> None:
    roster = [f"p{i:02d}" for i in range(22)]
    people = np.repeat(np.asarray(roster), 3)
    labels = np.tile(np.arange(3, dtype=np.int64), 22)
    weak = labels.copy()
    weak[labels == 1] = 0
    probabilities = _one_hot(weak)
    report_weak = method_report_from_decisions(
        labels=labels,
        raw_probabilities=probabilities,
        decisions=weak,
        participant_ids=people,
        roster=roster,
    )
    report_good = method_report_from_decisions(
        labels=labels,
        raw_probabilities=probabilities,
        decisions=labels,
        participant_ids=people,
        roster=roster,
    )
    participant_folds = np.asarray([min(index // 5, 4) for index in range(22)])
    result = analyse(
        {"d0": report_weak, "d1": report_good, "f3": report_weak},
        labels=labels,
        decisions={"d0": weak, "d1": labels, "f3": weak},
        participant_ids=people,
        participant_folds=participant_folds,
        roster=roster,
        control_replay_exact=True,
    )
    assert result["gates"]["d1_minus_d0"]["status"] == "pass"
    assert result["gates"]["d1_minus_f3"]["status"] == "pass"
    assert result["event_topology"]["d0_to_d1"]["rescues"] == 22
