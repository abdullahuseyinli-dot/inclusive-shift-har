from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from inclusive_shift_har.experiments.cage_har import (
    _load_config,
    apply_probability_repair,
    run_cage_cross_fitted_arrays,
    run_cage_development,
    run_cage_synthetic_smoke,
    synthetic_cage_bundle,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.cage_har import (
    apply_cage_trust_region,
    apply_soft_semantic_gauge,
    build_advantage_router_features,
    cage_expert_reliability,
    counterfactual_log_loss_advantage,
    extract_cage_gauge_context,
    extract_cage_sensor_health,
    fit_ridge_advantage_model,
    participant_jackknife_advantage_prediction,
    select_semantic_gauge_query,
)


def _sensor_windows(seed: int = 917) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    primary = generator.normal(scale=0.2, size=(5, 64, 6))
    gravity = generator.normal(size=(5, 64, 3))
    gravity /= np.maximum(np.linalg.norm(gravity, axis=2, keepdims=True), 1e-12)
    return primary, gravity


def test_cage_gauge_is_partial_invariant_and_retains_orientation() -> None:
    primary, gravity = _sensor_windows()
    angle = 0.73
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated_primary = np.concatenate(
        (primary[:, :, :3] @ rotation.T, primary[:, :, 3:] @ rotation.T), axis=2
    )
    original = extract_cage_gauge_context(primary, gravity, sampling_rate_hz=50.0)
    rotated = extract_cage_gauge_context(
        rotated_primary, gravity @ rotation.T, sampling_rate_hz=50.0
    )
    assert original.features.shape == (5, len(original.feature_names))
    assert original.feature_names == rotated.feature_names
    np.testing.assert_allclose(original.features[:, :7], rotated.features[:, :7], atol=1e-9)
    np.testing.assert_allclose(original.features[:, 13:], rotated.features[:, 13:], atol=1e-9)
    np.testing.assert_allclose(original.reliability, rotated.reliability, atol=1e-9)


def test_cage_gauge_missingness_is_explicit_and_fails_to_zero_reliability() -> None:
    primary, gravity = _sensor_windows()
    validity = np.ones((*primary.shape[:2], 9), dtype=np.bool_)
    validity[0, :, 6:9] = False
    validity[1, :32, 6:9] = False
    context = extract_cage_gauge_context(
        primary,
        gravity,
        sampling_rate_hz=50.0,
        validity_mask=validity,
    )
    assert context.reliability[0] == 0.0
    assert 0.0 < context.reliability[1] < context.reliability[2]
    assert np.isfinite(context.features).all()


def test_cage_sensor_health_exposes_faults_without_treating_zero_as_missing() -> None:
    primary, gravity = _sensor_windows()
    signals = np.concatenate((primary, gravity), axis=2)
    signals[0, 0, 0] = 0.0
    validity = np.ones_like(signals, dtype=np.bool_)
    clean = extract_cage_sensor_health(signals, validity_mask=validity)
    assert clean.reliability[0] > 0.99
    validity[1, :, 3:6] = False
    dropped = extract_cage_sensor_health(signals, validity_mask=validity)
    assert dropped.reliability[1] == 0.0
    gyro_valid_mean = dropped.feature_names.index("gyroscope__valid_fraction_mean")
    assert dropped.features[1, gyro_valid_mean] == 0.0
    assert dropped.features.shape == (5, len(dropped.feature_names))


def test_cage_expert_reliability_depends_only_on_required_modalities() -> None:
    primary, gravity = _sensor_windows()
    signals = np.concatenate((primary, gravity), axis=2)
    validity = np.ones_like(signals, dtype=np.bool_)
    validity[0, :, 3:6] = False
    health = extract_cage_sensor_health(signals, validity_mask=validity)
    gauge = extract_cage_gauge_context(
        primary,
        gravity,
        sampling_rate_hz=50.0,
        validity_mask=validity,
    )
    acceleration_only = cage_expert_reliability(
        health,
        required_modalities=("accelerometer",),
    )
    full = cage_expert_reliability(
        health,
        required_modalities=("accelerometer", "gyroscope", "gravity"),
        gravity_gauge=gauge,
    )
    assert acceleration_only[0] > 0.99
    assert full[0] == 0.0


def test_counterfactual_advantage_and_router_features_are_label_safe_at_inference() -> None:
    base = np.asarray([[0.1, 0.6, 0.3], [0.1, 0.7, 0.2]], dtype=np.float64)
    expert = np.asarray([[0.1, 0.2, 0.7], [0.1, 0.8, 0.1]], dtype=np.float64)
    labels = np.asarray([2, 1], dtype=np.int64)
    advantage = counterfactual_log_loss_advantage(base, expert, labels)
    assert np.all(advantage > 0.0)
    features, names = build_advantage_router_features(
        base,
        expert,
        context_features=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        expert_reliability=np.asarray([0.9, 0.8]),
    )
    assert features.shape == (2, len(names))
    assert "expert_reliability" in names
    assert all("label" not in name or name == "base_expert_label_disagreement" for name in names)


def test_participant_balanced_ridge_and_jackknife_predict_advantage() -> None:
    participants = np.repeat(np.asarray(["a", "b", "c", "d", "e", "f"]), 8)
    x = np.tile(np.linspace(-1.0, 1.0, 8), 6)
    features = np.column_stack((x, x**2))
    target = 0.6 * x - 0.05 * x**2
    model = fit_ridge_advantage_model(
        features,
        target,
        participants,
        ridge_penalty=0.1,
    )
    prediction = model.predict(np.asarray([[-0.8, 0.64], [0.8, 0.64]]))
    assert prediction[0] < 0.0 < prediction[1]
    jackknife = participant_jackknife_advantage_prediction(
        features,
        target,
        participants,
        np.asarray([[-0.8, 0.64], [0.8, 0.64]]),
        ridge_penalty=0.1,
        lower_bound_z=1.0,
    )
    assert jackknife.model_count == 6
    assert jackknife.lower_bound[0] < 0.0 < jackknife.lower_bound[1]
    assert np.all(jackknife.lower_bound <= jackknife.mean)


def test_cage_trust_region_preserves_mobility_and_kl_cap() -> None:
    base = np.asarray(
        [[0.10, 0.75, 0.15], [0.80, 0.10, 0.10], [0.10, 0.75, 0.15]],
        dtype=np.float64,
    )
    posture = np.asarray(
        [[0.30, 0.10, 0.60], [0.10, 0.20, 0.70], [0.30, 0.10, 0.60]],
        dtype=np.float64,
    )
    full = np.asarray(
        [[0.05, 0.20, 0.75], [0.20, 0.30, 0.50], [0.05, 0.20, 0.75]],
        dtype=np.float64,
    )
    result = apply_cage_trust_region(
        base,
        np.stack((posture, full), axis=1),
        np.asarray([[0.5, 0.1], [0.5, 0.1], [0.5, 0.4]]),
        np.asarray([[1.0, 1.0], [0.2, 1.0], [0.2, 1.0]]),
        np.asarray([True, False]),
        minimum_advantage=0.05,
        minimum_reliability=0.5,
        maximum_mix_weight=1.0,
        advantage_scale=0.1,
        maximum_kl=0.02,
    )
    assert result.routed.tolist() == [True, True, True]
    assert result.chosen_expert.tolist() == [0, 1, 1]
    assert result.probabilities[0, 0] == pytest.approx(base[0, 0])
    assert result.probabilities[1, 0] != pytest.approx(base[1, 0])
    assert np.all(result.achieved_kl <= 0.02 + 1e-9)
    assert np.all((result.mix_weight > 0.0) & (result.mix_weight <= 1.0))


def test_cage_trust_region_returns_base_exactly_when_no_expert_is_safe() -> None:
    base = np.asarray([[0.2, 0.4, 0.4]], dtype=np.float64)
    expert = np.asarray([[[0.1, 0.8, 0.1]]], dtype=np.float64)
    result = apply_cage_trust_region(
        base,
        expert,
        np.asarray([[1.0]]),
        np.asarray([[0.1]]),
        np.asarray([True]),
        minimum_advantage=0.0,
        minimum_reliability=0.5,
        maximum_mix_weight=1.0,
        advantage_scale=0.1,
        maximum_kl=0.1,
    )
    np.testing.assert_array_equal(result.probabilities, base)
    assert result.routed.tolist() == [False]
    assert result.chosen_expert.tolist() == [-1]


def test_probability_repair_identity_and_normalization() -> None:
    probability = np.asarray([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]], dtype=np.float64)
    identity = apply_probability_repair(
        probability,
        temperature=1.0,
        posture_logit_offset=0.0,
    )
    np.testing.assert_allclose(identity, probability)
    repaired = apply_probability_repair(
        probability,
        temperature=0.8,
        posture_logit_offset=0.15,
    )
    np.testing.assert_allclose(repaired.sum(axis=1), 1.0)
    assert repaired[0, 1] / repaired[0, 2] > probability[0, 1] / probability[0, 2]


def test_cage_synthetic_cross_fit_exercises_all_ablation_paths() -> None:
    config = _load_config(Path("configs/experiments/cage_har_v1.yaml"))
    result, predictions = run_cage_cross_fitted_arrays(
        synthetic_cage_bundle(),
        config=config,
    )
    assert set(result["reports"]) == {
        "base_uncalibrated",
        "base_decision_repaired",
        "global_trust_blend",
        "confidence_trust_gate",
        "disagreement_trust_gate",
        "cage_har",
    }
    assert result["participant_count"] == 12
    assert result["outer_participant_labels_used_for_router_or_repair"] is False
    assert result["routing_audits"]["cage_har"]["routed_count"] > 0
    assert result["routing_audits"]["cage_har"]["changed_count"] > 0
    assert predictions["cage_har_probabilities"].shape == (1080, 3)


def test_cage_synthetic_result_is_create_only_and_self_hashed(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    result = run_cage_synthetic_smoke(
        config_path=Path("configs/experiments/cage_har_v1.yaml"),
        output_directory=output,
        code_commit="synthetic-test-only",
        seed=20260904,
    )
    assert result["status"] == "pass_synthetic_contract_only"
    assert result["real_har_performance_claim_allowed"] is False
    stored = json.loads((output / "result.json").read_text(encoding="utf-8"))
    claimed = stored.pop("record_sha256")
    assert claimed == canonical_json_sha256(stored)
    with pytest.raises(FileExistsError):
        run_cage_synthetic_smoke(
            config_path=Path("configs/experiments/cage_har_v1.yaml"),
            output_directory=output,
            code_commit="synthetic-test-only",
            seed=20260904,
        )


def test_cage_real_runner_refuses_consumed_participant_ids(tmp_path: Path) -> None:
    bundle = synthetic_cage_bundle()
    participants = bundle.participant_ids.copy()
    participants[participants == "synthetic-00"] = "1"
    bundle_path = tmp_path / "bundle.npz"
    np.savez_compressed(
        bundle_path,
        base_probabilities=bundle.base_probability,
        expert_probabilities=bundle.expert_probabilities,
        labels=bundle.labels,
        participant_ids=participants,
        window_ids=bundle.window_ids,
        context_features=bundle.context_features,
        expert_reliability=bundle.expert_reliability,
        expert_names=np.asarray(bundle.expert_names),
        expert_posture_only=bundle.expert_posture_only,
    )
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "record_kind": "cage_har_cross_fitted_development_bundle",
        "status": "complete_new_development_oof",
        "cohort_role": "new_development",
        "dataset_id": "new-test-cohort",
        "channel_count": 9,
        "participant_ids": sorted(set(participants.tolist())),
        "bundle": {"path": bundle_path.as_posix(), "sha256": sha256_file(bundle_path)},
        "oof_contract": {
            "base_predictions_participant_exclusive": True,
            "expert_predictions_participant_exclusive": True,
            "router_features_label_free": True,
        },
        "evidence_boundary": {
            "includes_inclusivehar_participants_1_through_20": False,
            "includes_consumed_daghar_targets": False,
            "target_or_confirmatory_cohort": False,
        },
    }
    manifest["record_sha256"] = canonical_json_sha256(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PermissionError, match="consumed participant"):
        run_cage_development(
            config_path=Path("configs/experiments/cage_har_v1.yaml"),
            bundle_manifest_path=manifest_path,
            output_directory=tmp_path / "result",
            code_commit="synthetic-test-only",
        )


def test_semantic_gauge_query_is_label_free_and_softly_repairs_posture() -> None:
    base = np.asarray(
        [[0.05, 0.90, 0.05], [0.05, 0.80, 0.15], [0.70, 0.20, 0.10]],
        dtype=np.float64,
    )
    expert = np.asarray(
        [[0.05, 0.05, 0.90], [0.05, 0.15, 0.80], [0.70, 0.10, 0.20]],
        dtype=np.float64,
    )
    windows = np.asarray(["a", "b", "c"])
    queried, scores = select_semantic_gauge_query(
        base,
        expert,
        np.asarray([1.0, 0.9, 1.0]),
        windows,
        minimum_stationary_probability=0.5,
        minimum_suspicion_score=0.1,
    )
    assert queried == 0
    assert scores[0] > scores[1]
    repaired = apply_soft_semantic_gauge(
        base,
        queried_index=queried,
        observed_label=2,
        prior_swap_probability=0.1,
        maximum_swap_weight=1.0,
        evidence_temperature=1.0,
    )
    assert repaired.posterior_swap_probability > 0.5
    assert repaired.probabilities[0, 2] > repaired.probabilities[0, 1]
    np.testing.assert_allclose(repaired.probabilities[:, 0], base[:, 0])
    np.testing.assert_allclose(repaired.probabilities.sum(axis=1), 1.0)
