from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from inclusive_shift_har.experiments.hera_ctgr_v2_retrospective import (
    _apply_dual_candidate,
    _participant_oracle,
    _rank_lock_candidate,
    _select_dual_candidate,
    _select_offset,
    _select_temperature,
    _window_oracle,
    load_hera_v2_retrospective_config,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _fixed() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        load_hera_v2_retrospective_config(
            _root() / "configs/experiments/hera_ctgr_v2_retrospective_v1.yaml"
        )["fixed_method"],
    )


def _probabilities(labels: np.ndarray, confidence: float = 0.75) -> np.ndarray:
    result = np.full((labels.size, 3), (1.0 - confidence) / 2.0, dtype=np.float64)
    result[np.arange(labels.size), labels] = confidence
    return result


def test_v2_retrospective_config_locks_nesting_and_claim_boundary() -> None:
    config = load_hera_v2_retrospective_config(
        _root() / "configs/experiments/hera_ctgr_v2_retrospective_v1.yaml"
    )
    assert config["method_freeze"]["tag"] == "hera-ctgr-v2-implementation-v1"
    assert config["fixed_method"]["harm_penalty"] == 2.0
    assert config["nesting_contract"][
        "binary_heads_trained_on_participant_exclusive_oof_disagreements"
    ]
    assert config["claim_policy"]["state_of_the_art_claim_allowed"] is False


def test_v2_retrospective_protocol_record_is_self_hashed_when_present() -> None:
    path = _root() / "results/protocol/hera_ctgr_v2_retrospective_v1.json"
    if not path.exists():
        return
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


def test_rank_lock_restores_any_mobility_boundary_change() -> None:
    base = np.asarray([[0.6, 0.3, 0.1], [0.1, 0.6, 0.3], [0.1, 0.6, 0.3]])
    candidate = np.asarray([[0.2, 0.7, 0.1], [0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    result = _rank_lock_candidate(base, candidate)
    np.testing.assert_array_equal(result[:2], base[:2])
    np.testing.assert_array_equal(result[2], candidate[2])


def test_temperature_and_offset_are_selected_separately() -> None:
    labels = np.tile(np.arange(3, dtype=np.int64), 4)
    participants = np.repeat(np.asarray(["a", "b", "c", "d"]), 3)
    probability = _probabilities(labels, confidence=0.60)
    temperature, temperature_candidates = _select_temperature(
        probability, labels, participants, _fixed()
    )
    offset, offset_candidates = _select_offset(probability, labels, participants, _fixed())
    assert temperature in {0.75, 0.9, 1.0, 1.1, 1.25}
    assert offset == 0.0
    assert len(temperature_candidates) == 5
    assert len(offset_candidates) == 5


def test_dual_candidate_selection_obeys_frozen_grid_and_rank_lock() -> None:
    labels = np.tile(np.arange(3, dtype=np.int64), 4)
    participants = np.repeat(np.asarray(["a", "b", "c", "d"]), 3)
    core = _probabilities(labels, confidence=0.60)
    expert = core.copy()
    expert[:, [1, 2]] = expert[:, [2, 1]]
    selection, candidates = _select_dual_candidate(core, expert, labels, participants, _fixed())
    result = _apply_dual_candidate(core, expert, selection)
    assert len(candidates) == 16
    assert selection["confidence_threshold"] in {0.5, 0.55, 0.6, 0.65}
    assert selection["blend_weight"] in {0.25, 0.5, 0.75, 1.0}
    np.testing.assert_array_equal(result.argmax(axis=1) == 0, core.argmax(axis=1) == 0)


def test_participant_and_window_oracles_are_explicit_hindsight_diagnostics() -> None:
    labels = np.tile(np.arange(3, dtype=np.int64), 2)
    participants = np.repeat(np.asarray(["a", "b"]), 3)
    perfect = _probabilities(labels)
    wrong = np.roll(perfect, 1, axis=1)
    first = perfect.copy()
    first[3:] = wrong[3:]
    second = wrong.copy()
    second[3:] = perfect[3:]
    participant_oracle, choices = _participant_oracle(
        labels, participants, (first, second), ("FIRST", "SECOND")
    )
    window_oracle = _window_oracle(labels, (first, second))
    assert choices == {"a": "FIRST", "b": "SECOND"}
    np.testing.assert_array_equal(participant_oracle.argmax(axis=1), labels)
    np.testing.assert_array_equal(window_oracle.argmax(axis=1), labels)
