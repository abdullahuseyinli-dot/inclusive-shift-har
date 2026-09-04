from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.hera_ctgr import (
    apply_physics_reference,
    apply_physics_veto,
    apply_posture_conditional_calibration,
    apply_tri_state_responder_controller,
    average_probability_candidates,
    build_responder_signatures,
    extract_gravity_kinematic_context,
    fit_physics_reference,
    standardized_nearest_support_distance,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _probabilities(rows: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = np.tile(np.asarray([[0.10, 0.55, 0.35]], dtype=np.float64), (rows, 1))
    expert = np.tile(np.asarray([[0.10, 0.25, 0.65]], dtype=np.float64), (rows, 1))
    ctgr = 0.5 * base + 0.5 * expert
    broad = 0.25 * base + 0.75 * expert
    return base, expert, ctgr, broad


def test_config_declares_consumed_evidence_and_separate_lanes() -> None:
    config = yaml.safe_load(
        (_root() / "configs/experiments/hera_ctgr_v1.yaml").read_text(encoding="utf-8")
    )
    assert config["status"] == "implemented_awaiting_new_development_cohort"
    assert config["frozen_paths"]["uncertain_default"] == "pulse"
    assert config["input_contract"]["forbidden_dataset_ids"] == [
        "inclusivehar_v4",
        "daghar_v2",
    ]
    assert config["claim_policy"]["unlabeled_participant_context_reported_as_transductive"]
    assert not config["claim_policy"]["confirmatory_claim_allowed"]


def test_implementation_record_is_self_hashed_and_anchored() -> None:
    record_path = _root() / "results/protocol/hera_ctgr_v1.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    supplied_hash = record.pop("record_sha256")
    assert supplied_hash == canonical_json_sha256(record)
    for reference in (
        record["protocol_document"],
        record["experiment_config"],
        *record["implementation"],
        *record["focused_tests"],
    ):
        path = _root() / reference["path"]
        assert sha256_file(path) == reference["sha256"]


def test_zero_motion_has_zero_kinematic_residual() -> None:
    primary = np.zeros((3, 16, 6), dtype=np.float64)
    gravity = np.zeros((3, 16, 3), dtype=np.float64)
    gravity[:, :, 2] = 1.0
    context = extract_gravity_kinematic_context(primary, gravity, sampling_rate_hz=50.0)
    np.testing.assert_allclose(context.normalized_residual_p90, 0.0)
    reference = fit_physics_reference(
        context,
        veto_quantile=0.99,
        minimum_valid_pair_fraction=0.90,
    )
    trust = apply_physics_reference(context, reference)
    assert trust.trusted.all()
    assert np.isfinite(context.features).all()


def test_inconsistent_gravity_change_increases_residual() -> None:
    primary = np.zeros((2, 32, 6), dtype=np.float64)
    gravity = np.zeros((2, 32, 3), dtype=np.float64)
    gravity[:, :, 2] = 1.0
    angles = np.linspace(0.0, np.pi / 2.0, 32)
    gravity[1, :, 0] = np.sin(angles)
    gravity[1, :, 2] = np.cos(angles)
    context = extract_gravity_kinematic_context(primary, gravity, sampling_rate_hz=50.0)
    assert context.normalized_residual_p90[1] > context.normalized_residual_p90[0]


def test_posture_calibration_preserves_mobility_mass() -> None:
    base, _, _, _ = _probabilities()
    calibrated = apply_posture_conditional_calibration(
        base,
        temperature=0.8,
        posture_logit_offset=0.15,
    )
    np.testing.assert_array_equal(calibrated[:, 0], base[:, 0])
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0)
    assert not np.array_equal(calibrated[:, 1:], base[:, 1:])


def test_candidate_average_is_a_probability_matrix() -> None:
    base, expert, ctgr, _ = _probabilities()
    averaged = average_probability_candidates(np.stack((base, expert, ctgr), axis=0))
    np.testing.assert_allclose(averaged.sum(axis=1), 1.0)
    np.testing.assert_allclose(averaged, (base + expert + ctgr) / 3.0)


def test_responder_signature_is_invariant_to_within_participant_order() -> None:
    base, expert, ctgr, broad = _probabilities(8)
    participants = np.asarray(["a"] * 4 + ["b"] * 4, dtype=np.str_)
    primary = np.zeros((8, 12, 6), dtype=np.float64)
    gravity = np.zeros((8, 12, 3), dtype=np.float64)
    gravity[:, :, 2] = 1.0
    context = extract_gravity_kinematic_context(primary, gravity, sampling_rate_hz=50.0)
    trust = np.ones(8, dtype=np.bool_)
    reliability = np.ones(8, dtype=np.float64)
    first = build_responder_signatures(
        base,
        expert,
        ctgr,
        broad,
        participants,
        gravity_reliability=reliability,
        kinematic_context=context,
        physics_trusted=trust,
        low_confidence_threshold=0.60,
    )
    order = np.asarray([3, 1, 0, 2, 7, 5, 4, 6])
    second = build_responder_signatures(
        base[order],
        expert[order],
        ctgr[order],
        broad[order],
        participants[order],
        gravity_reliability=reliability[order],
        kinematic_context=type(context)(
            features=context.features[order],
            normalized_residual_p90=context.normalized_residual_p90[order],
            valid_pair_fraction=context.valid_pair_fraction[order],
            feature_names=context.feature_names,
        ),
        physics_trusted=trust[order],
        low_confidence_threshold=0.60,
    )
    assert first.feature_names == second.feature_names
    np.testing.assert_allclose(first.features, second.features)


def test_physics_veto_restores_base_exactly() -> None:
    base, _, ctgr, _ = _probabilities()
    trusted = np.asarray([True, False, True, False, True, False])
    result, vetoed = apply_physics_veto(base, ctgr, trusted)
    np.testing.assert_array_equal(result[~trusted], base[~trusted])
    np.testing.assert_array_equal(result[trusted], ctgr[trusted])
    np.testing.assert_array_equal(vetoed, ~trusted)


def test_tri_state_controller_defaults_to_pulse_and_obeys_bounds() -> None:
    base, _, pulse, broad = _probabilities()
    participants = np.asarray(["off", "off", "pulse", "pulse", "broad", "broad"])
    signature_ids = np.asarray(["broad", "off", "pulse"])
    result = apply_tri_state_responder_controller(
        base,
        pulse,
        broad,
        participants,
        signature_ids,
        base_advantage_lower_bound=np.asarray([-0.1, 0.2, 0.0]),
        broad_advantage_lower_bound=np.asarray([0.2, -0.1, 0.0]),
        support_distance=np.asarray([0.5, 0.5, 0.5]),
        physics_trusted=np.ones(6, dtype=np.bool_),
        minimum_advantage=0.01,
        maximum_support_distance=3.0,
    )
    np.testing.assert_array_equal(result.probabilities[participants == "off"], base[:2])
    np.testing.assert_array_equal(result.probabilities[participants == "pulse"], pulse[:2])
    np.testing.assert_array_equal(result.probabilities[participants == "broad"], broad[:2])
    assert result.participant_state == {"broad": "BROAD", "off": "OFF", "pulse": "PULSE"}


def test_tri_state_controller_uses_pulse_for_ood_signature() -> None:
    base, _, pulse, broad = _probabilities(2)
    result = apply_tri_state_responder_controller(
        base,
        pulse,
        broad,
        np.asarray(["p", "p"]),
        np.asarray(["p"]),
        base_advantage_lower_bound=np.asarray([1.0]),
        broad_advantage_lower_bound=np.asarray([2.0]),
        support_distance=np.asarray([4.0]),
        physics_trusted=np.ones(2, dtype=np.bool_),
        minimum_advantage=0.01,
        maximum_support_distance=3.0,
    )
    np.testing.assert_array_equal(result.probabilities, pulse)
    assert result.participant_state["p"] == "PULSE"


def test_support_distance_validates_shapes_and_is_zero_for_seen_rows() -> None:
    training = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    distance = standardized_nearest_support_distance(training, training[[1]])
    np.testing.assert_allclose(distance, 0.0)
    with pytest.raises(ValueError, match="incompatible"):
        standardized_nearest_support_distance(training, np.ones((1, 3)))
