from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.hera_ctgr_v2 import (
    BinaryJackknifePrediction,
    accumulate_bout_intervention_evidence,
    apply_one_query_state_selection,
    apply_rank_locked_posture_offset,
    apply_rescue_harm_sentinel,
    build_rescue_harm_features,
    counterfactual_rescue_harm_targets,
    extract_dual_frame_posture_features,
    fit_participant_balanced_logistic,
    participant_jackknife_binary_prediction,
    stability_weighted_candidate_marginalization,
    temperature_scale_probabilities,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _binary_prediction(lower: list[float], upper: list[float]) -> BinaryJackknifePrediction:
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    return BinaryJackknifePrediction(
        mean=0.5 * (low + high),
        standard_deviation=0.5 * (high - low),
        lower_bound=low,
        upper_bound=high,
        model_count=4,
    )


def test_config_freezes_decision_separation_and_claim_boundary() -> None:
    path = _root() / "configs/experiments/hera_ctgr_v2.yaml"
    if not path.exists():
        pytest.skip("v2 protocol files are added after core implementation")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["status"] == "implemented_awaiting_new_development_cohort"
    assert config["decision_separation"]["mobility_flip_from_posture_path_allowed"] is False
    assert config["rescue_harm_sentinel"]["separate_binary_heads"] is True
    assert config["claim_policy"]["confirmatory_claim_allowed"] is False


def test_implementation_record_is_self_hashed_when_present() -> None:
    path = _root() / "results/protocol/hera_ctgr_v2.json"
    if not path.exists():
        pytest.skip("v2 protocol record is assembled after core implementation")
    record = json.loads(path.read_text(encoding="utf-8"))
    supplied = record.pop("record_sha256")
    assert supplied == canonical_json_sha256(record)
    references = (
        record["protocol_document"],
        record["experiment_config"],
        *record["implementation"],
        *record["focused_tests"],
    )
    for reference in references:
        assert sha256_file(_root() / reference["path"]) == reference["sha256"]


def test_dual_frame_features_preserve_absolute_gravity_and_are_finite() -> None:
    primary = np.zeros((2, 64, 6), dtype=np.float64)
    gravity = np.zeros((2, 64, 3), dtype=np.float64)
    gravity[0, :, 2] = 1.0
    gravity[1, :, 0] = 1.0
    features = extract_dual_frame_posture_features(primary, gravity, sampling_rate_hz=50.0)
    assert features.features.shape == (2, 260)
    assert features.compact_context.shape == (2, 15)
    assert np.isfinite(features.features).all()
    assert features.features[0, features.feature_names.index("gravity_unit_z__mean")] == 1.0
    assert features.features[1, features.feature_names.index("gravity_unit_x__mean")] == 1.0
    assert not np.array_equal(features.features[0], features.features[1])


def test_stability_weighting_downweights_an_outlier() -> None:
    candidates = np.asarray(
        [
            [[0.10, 0.80, 0.10], [0.70, 0.20, 0.10]],
            [[0.10, 0.78, 0.12], [0.68, 0.22, 0.10]],
            [[0.10, 0.10, 0.80], [0.05, 0.05, 0.90]],
        ],
        dtype=np.float64,
    )
    result = stability_weighted_candidate_marginalization(
        candidates,
        divergence_temperature=0.05,
        minimum_candidate_weight=0.05,
    )
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(result.candidate_weights.sum(axis=0), 1.0)
    assert np.all(result.candidate_weights[2] < result.candidate_weights[0])
    assert np.all(result.disagreement > 0.0)


def test_temperature_is_rank_preserving() -> None:
    probability = np.asarray(
        [[0.51, 0.30, 0.19], [0.10, 0.44, 0.46], [0.20, 0.70, 0.10]], dtype=np.float64
    )
    calibrated = temperature_scale_probabilities(probability, temperature=0.75)
    np.testing.assert_array_equal(calibrated.argmax(axis=1), probability.argmax(axis=1))
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0)


def test_posture_offset_never_changes_mobility_membership() -> None:
    probability = np.asarray(
        [[0.60, 0.21, 0.19], [0.10, 0.46, 0.44], [0.10, 0.44, 0.46]], dtype=np.float64
    )
    adjusted, changed = apply_rank_locked_posture_offset(probability, posture_logit_offset=0.50)
    np.testing.assert_array_equal(adjusted[:, 0], probability[:, 0])
    np.testing.assert_array_equal(adjusted.argmax(axis=1) == 0, probability.argmax(axis=1) == 0)
    assert not changed[0]


def test_counterfactual_targets_separate_rescues_and_harms() -> None:
    base = np.asarray([[0.1, 0.7, 0.2], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7]], dtype=np.float64)
    intervention = np.asarray([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7]], dtype=np.float64)
    targets = counterfactual_rescue_harm_targets(
        base, intervention, np.asarray([2, 1, 2], dtype=np.int64)
    )
    np.testing.assert_array_equal(targets.disagreement, [True, True, False])
    np.testing.assert_array_equal(targets.rescue, [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(targets.harm, [0.0, 1.0, 0.0])


def test_participant_balanced_binary_head_and_jackknife_are_bounded() -> None:
    participants = np.repeat(np.asarray(["a", "b", "c", "d", "e", "f"]), 4)
    target = np.tile(np.asarray([0.0, 0.0, 1.0, 1.0]), 6)
    features = np.column_stack((target * 2.0 - 1.0, np.arange(target.size) % 2))
    model = fit_participant_balanced_logistic(features, target, participants, ridge_penalty=1.0)
    probability = model.predict_probability(features)
    assert float(probability[target == 1].mean()) > float(probability[target == 0].mean())
    prediction = participant_jackknife_binary_prediction(
        features,
        target,
        participants,
        features[:3],
        ridge_penalty=1.0,
        interval_z=1.0,
    )
    assert prediction.model_count == 6
    assert np.all((prediction.lower_bound >= 0.0) & (prediction.upper_bound <= 1.0))
    assert np.all(prediction.lower_bound <= prediction.upper_bound)


def test_sentinel_routes_only_safe_posture_disagreements() -> None:
    base = np.asarray([[0.1, 0.6, 0.3], [0.1, 0.6, 0.3], [0.6, 0.3, 0.1]], dtype=np.float64)
    intervention = np.asarray([[0.1, 0.3, 0.6], [0.1, 0.3, 0.6], [0.1, 0.2, 0.7]], dtype=np.float64)
    rescue = _binary_prediction([0.8, 0.8, 0.9], [0.9, 0.9, 1.0])
    harm = _binary_prediction([0.0, 0.0, 0.0], [0.05, 0.4, 0.05])
    result = apply_rescue_harm_sentinel(
        base,
        intervention,
        rescue,
        harm,
        physics_reliability=np.ones(3),
        physics_trusted=np.ones(3, dtype=np.bool_),
        support_distance=np.zeros(3),
        minimum_rescue_lower_bound=0.5,
        maximum_harm_upper_bound=0.1,
        harm_penalty=2.0,
        minimum_net_benefit=0.2,
        minimum_physics_reliability=0.5,
        maximum_support_distance=3.0,
    )
    np.testing.assert_array_equal(result.routed, [True, False, False])
    np.testing.assert_array_equal(result.probabilities.argmax(axis=1) == 0, [False, False, True])


def test_rescue_harm_features_are_label_free_and_aligned() -> None:
    base = np.tile(np.asarray([[0.1, 0.6, 0.3]]), (4, 1))
    intervention = np.tile(np.asarray([[0.1, 0.3, 0.6]]), (4, 1))
    features, names = build_rescue_harm_features(
        base,
        intervention,
        candidate_disagreement=np.linspace(0.0, 0.1, 4),
        physics_reliability=np.ones(4),
        dual_frame_context=np.ones((4, 3)),
    )
    assert features.shape == (4, len(names))
    assert len(names) == 18
    assert np.isfinite(features).all()


def test_one_query_selects_state_and_excludes_query() -> None:
    base = np.asarray([[0.1, 0.8, 0.1], [0.1, 0.7, 0.2], [0.1, 0.6, 0.3]], dtype=np.float64)
    normal = np.asarray([[0.1, 0.1, 0.8], [0.1, 0.2, 0.7], [0.1, 0.3, 0.6]], dtype=np.float64)
    result = apply_one_query_state_selection(
        base,
        normal,
        np.ones(3),
        np.asarray(["w1", "w2", "w3"]),
        np.asarray([1, 1, 1]),
        minimum_stationary_mass=0.8,
    )
    assert result.queried_index == 0
    assert result.selected_state == "OFF"
    assert not result.evaluation_mask[0]
    np.testing.assert_array_equal(result.probabilities, base)


def test_bout_accumulator_is_causal_and_resets() -> None:
    result = accumulate_bout_intervention_evidence(
        np.asarray([1.0, 3.0, 5.0, 10.0, 14.0]),
        np.asarray(["a", "a", "a", "b", "b"]),
        width=2,
    )
    np.testing.assert_allclose(result, [1.0, 2.0, 4.0, 10.0, 12.0])


def test_invalid_candidate_floor_fails_closed() -> None:
    candidates = np.tile(np.asarray([[[0.1, 0.6, 0.3]]]), (3, 1, 1))
    with pytest.raises(ValueError, match="minimum candidate weight"):
        stability_weighted_candidate_marginalization(
            candidates,
            divergence_temperature=0.05,
            minimum_candidate_weight=0.34,
        )
