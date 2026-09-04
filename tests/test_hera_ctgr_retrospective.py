from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from inclusive_shift_har.experiments.hera_ctgr_retrospective import (
    _apply_calibration,
    _label_informed_oracle,
    _random_state_control,
    _select_posture_calibration,
    _top_ranked_candidates,
    _utility_targets,
    load_hera_retrospective_config,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _fixed() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        load_hera_retrospective_config(
            _root() / "configs/experiments/hera_ctgr_retrospective_v1.yaml"
        )["fixed_method"],
    )


def _probabilities(labels: np.ndarray, confidence: float = 0.75) -> np.ndarray:
    result = np.full((labels.size, 3), (1.0 - confidence) / 2.0, dtype=np.float64)
    result[np.arange(labels.size), labels] = confidence
    return result


def test_retrospective_config_locks_methods_nesting_and_claim_boundary() -> None:
    config = load_hera_retrospective_config(
        _root() / "configs/experiments/hera_ctgr_retrospective_v1.yaml"
    )
    assert config["fixed_method"]["candidate_top_k"] == 3
    assert (
        config["nesting_contract"][
            "outer_labels_used_for_training_selection_calibration_or_routing"
        ]
        is False
    )
    assert config["claim_policy"]["retrospective_result_is_hypothesis_generation_only"]
    assert config["claim_policy"]["state_of_the_art_claim_allowed"] is False


def test_retrospective_protocol_record_is_self_hashed_and_anchored() -> None:
    path = _root() / "results/protocol/hera_ctgr_retrospective_v1.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    supplied = record.pop("record_sha256")
    assert supplied == canonical_json_sha256(record)
    for reference in (
        record["protocol_document"],
        record["experiment_config"],
        record["implementation"],
        record["focused_test"],
    ):
        assert sha256_file(_root() / reference["path"]) == reference["sha256"]


def test_top_three_comes_only_from_frozen_eligible_ranking() -> None:
    candidates = {name: {"id": name} for name in ("a", "b", "c", "d")}
    folds = []
    for index in range(5):
        folds.append(
            {
                "outer_fold_id": f"f{index}",
                "selected_candidate_id": "a",
                "eligible_within_mean_tolerance": ["a", "b", "c"],
                "candidate_ranking": [
                    {"candidate_id": "a"},
                    {"candidate_id": "b"},
                    {"candidate_id": "c"},
                    {"candidate_id": "d"},
                ],
            }
        )
    selected = _top_ranked_candidates({"folds": folds}, candidates)
    assert tuple(item["id"] for item in selected["f0"]) == ("a", "b", "c")


def test_conditional_calibration_selection_and_restoration_preserve_contract() -> None:
    labels = np.tile(np.arange(3, dtype=np.int64), 4)
    participants = np.repeat(np.asarray(["a", "b", "c", "d"]), 3)
    probabilities = _probabilities(labels, confidence=0.65)
    base = probabilities.copy()
    restore = np.zeros(labels.size, dtype=np.bool_)
    restore[[0, 5, 9]] = True
    selection, candidates = _select_posture_calibration(
        probabilities,
        labels,
        participants,
        _fixed(),
        base_probability=base,
        restore_mask=restore,
    )
    calibrated = _apply_calibration(
        probabilities,
        selection,
        base_probability=base,
        restore_mask=restore,
    )
    assert len(candidates) == 9
    np.testing.assert_allclose(calibrated[restore], base[restore])
    np.testing.assert_allclose(calibrated[:, 0], probabilities[:, 0])
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0)


def test_participant_utility_targets_are_macro_f1_differences() -> None:
    labels = np.tile(np.arange(3, dtype=np.int64), 4)
    participants = np.repeat(np.asarray(["a", "b", "c", "d"]), 3)
    pulse = _probabilities(labels)
    base = pulse.copy()
    broad = pulse.copy()
    base[:3] = np.roll(base[:3], 1, axis=1)
    broad[3:6] = np.roll(broad[3:6], 1, axis=1)
    base_utility, broad_utility = _utility_targets(
        labels,
        participants,
        base,
        pulse,
        broad,
        np.asarray(["a", "b", "c", "d"]),
    )
    assert base_utility[0] < 0.0
    assert np.allclose(base_utility[1:], 0.0)
    assert broad_utility[1] < 0.0
    assert np.allclose(broad_utility[[0, 2, 3]], 0.0)


def test_random_control_preserves_state_counts_and_exact_off_fallback() -> None:
    labels = np.tile(np.arange(3, dtype=np.int64), 2)
    participants = np.repeat(np.asarray(["a", "b"]), 3)
    base = _probabilities(labels, confidence=0.60)
    pulse = _probabilities(labels, confidence=0.70)
    broad = _probabilities(labels, confidence=0.80)
    output, state, _, assigned = _random_state_control(
        base=base,
        pulse=pulse,
        broad=broad,
        participants=participants,
        participant_state={"a": "OFF", "b": "BROAD"},
        physics_trusted=np.ones(labels.size, dtype=np.bool_),
        calibration={"temperature": 1.0, "posture_logit_offset": 0.0},
        seed=7,
    )
    assert sorted(assigned.values()) == ["BROAD", "OFF"]
    off_rows = state == 0
    np.testing.assert_allclose(output[off_rows], base[off_rows])


def test_oracle_is_explicit_label_informed_upper_diagnostic() -> None:
    labels = np.tile(np.arange(3, dtype=np.int64), 2)
    participants = np.repeat(np.asarray(["a", "b"]), 3)
    perfect = _probabilities(labels)
    wrong = np.roll(perfect, 1, axis=1)
    base = perfect.copy()
    base[3:] = wrong[3:]
    pulse = wrong.copy()
    broad = wrong.copy()
    broad[3:] = perfect[3:]
    oracle, choices = _label_informed_oracle(labels, participants, (base, pulse, broad))
    assert choices == {"a": "OFF", "b": "BROAD"}
    assert np.array_equal(oracle.argmax(axis=1), labels)
