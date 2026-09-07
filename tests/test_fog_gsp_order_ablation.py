from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from inclusive_shift_har.experiments.fog_gsp_order_ablation import (
    PERMUTATION_NAMESPACE,
    SIX_CHANNEL_NAMES,
    _moment_audit,
    analyse,
    apply_joint_permutations,
    deterministic_joint_permutations,
    validate_config,
)
from inclusive_shift_har.experiments.fog_rf_feature_weight_factorial import method_report
from inclusive_shift_har.models.geometric_spectral_pyramid import (
    extract_geometric_spectral_pyramid_features,
    geometric_spectral_pyramid_feature_names,
)
from inclusive_shift_har.preprocessing.features import extract_engineered_features

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/fog_gsp_order_ablation_v1.yaml"


def _config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _probabilities(predictions: np.ndarray[Any, np.dtype[np.int64]]) -> np.ndarray[Any, Any]:
    return np.asarray(np.eye(3, dtype=np.float64)[predictions], dtype=np.float64)


def test_frozen_config_rejects_compute_estimator_and_permutation_drift() -> None:
    config = _config()
    validate_config(config)

    mutations = (
        ("estimator", "n_estimators", 501),
        ("resource_limits", "maximum_fit_attempts", 16),
        ("permutation", "namespace", "post-outcome-replicate"),
    )
    for section, field, value in mutations:
        changed = copy.deepcopy(config)
        changed[section][field] = value
        with pytest.raises(ValueError, match="changed"):
            validate_config(changed)


def test_joint_permutation_matches_independent_hash_construction() -> None:
    window_ids = np.asarray(["fogstar:001:window-0001", "fogstar:002:window-0001"])
    permutations = deterministic_joint_permutations(window_ids)
    expected_first = np.asarray(
        sorted(
            range(128),
            key=lambda index: (
                hashlib.sha256(
                    f"{PERMUTATION_NAMESPACE}|{window_ids[0]}|{index:03d}".encode()
                ).digest(),
                index,
            ),
        ),
        dtype=np.int16,
    )

    assert permutations.dtype == np.int16
    assert permutations.shape == (2, 128)
    assert np.array_equal(permutations[0], expected_first)
    assert np.array_equal(deterministic_joint_permutations(window_ids.copy()), permutations)
    assert not np.array_equal(permutations[0], permutations[1])


def test_joint_permutation_preserves_tuples_moments_and_window_boundaries() -> None:
    row = np.arange(3, dtype=np.float64)[:, None, None] * 100_000.0
    time = np.arange(128, dtype=np.float64)[None, :, None] * 100.0
    channel = np.arange(6, dtype=np.float64)[None, None, :]
    signals = row + time + channel
    permutations = deterministic_joint_permutations(
        np.asarray(["candidate-a", "candidate-b", "candidate-c"])
    )
    shuffled = apply_joint_permutations(signals, permutations)
    audit = _moment_audit(signals, shuffled)

    for index in range(signals.shape[0]):
        original_tuples = {tuple(item) for item in signals[index].tolist()}
        shuffled_tuples = {tuple(item) for item in shuffled[index].tolist()}
        assert shuffled_tuples == original_tuples
        assert all(int(item[0] // 100_000) == index for item in shuffled[index])
    assert audit["passed"] is True
    assert audit["float64_maximum_absolute_channel_mean_difference"] <= 1e-10
    assert audit["float64_maximum_absolute_global_covariance_difference"] <= 1e-10


def test_order_ablation_changes_gsp_but_not_feature_schema_or_proxies() -> None:
    generator = np.random.default_rng(20260907)
    signals = generator.normal(size=(3, 128, 6)).astype(np.float64)
    permutations = deterministic_joint_permutations(np.asarray(["a", "b", "c"]))
    shuffled = apply_joint_permutations(signals, permutations)
    ordinary = extract_engineered_features(signals, channel_names=SIX_CHANNEL_NAMES)
    ordered_gsp = extract_geometric_spectral_pyramid_features(signals, sampling_rate_hz=50.0)
    shuffled_gsp = extract_geometric_spectral_pyramid_features(shuffled, sampling_rate_hz=50.0)
    names = geometric_spectral_pyramid_feature_names()
    forbidden = ("participant", "subject", "label", "timestamp", "location", "window_id", "fold")

    assert ordinary.values.shape == (3, 80)
    assert ordered_gsp.shape == shuffled_gsp.shape == (3, 1400)
    assert len(names) == len(set(names)) == 1400
    assert np.isfinite(ordered_gsp).all() and np.isfinite(shuffled_gsp).all()
    assert not np.array_equal(ordered_gsp, shuffled_gsp)
    assert all(
        token not in name.lower() for name in (*ordinary.names, *names) for token in forbidden
    )


def test_practical_and_order_gates_require_the_full_fixed_comparison() -> None:
    config = _config()
    roster = [f"fogstar:{index:03d}" for index in range(1, 23)]
    labels = np.tile(np.arange(3, dtype=np.int64), len(roster))
    participants = np.repeat(np.asarray(roster, dtype=np.str_), 3)
    impaired = labels.copy()
    impaired[labels == 1] = 2
    impaired_report = method_report(
        labels=labels,
        probabilities=_probabilities(impaired),
        participant_ids=participants,
        roster=roster,
    )
    perfect_report = method_report(
        labels=labels,
        probabilities=_probabilities(labels),
        participant_ids=participants,
        roster=roster,
    )
    probabilities = np.stack(
        [_probabilities(impaired), _probabilities(labels), _probabilities(impaired)]
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        promoted = analyse(
            {"a": impaired_report, "b": perfect_report, "c": impaired_report},
            probabilities=probabilities,
            labels=labels,
            participant_ids=participants,
            config=config,
        )
        unchanged = analyse(
            {"a": perfect_report, "b": perfect_report, "c": perfect_report},
            probabilities=np.stack([_probabilities(labels)] * 3),
            labels=labels,
            participant_ids=participants,
            config=config,
        )

    assert promoted["practical_promotion_gate"]["status"] == "pass"
    assert promoted["order_mechanism_gate"]["status"] == "pass"
    assert all(promoted["practical_promotion_gate"]["checks"].values())
    assert all(promoted["order_mechanism_gate"]["checks"].values())
    assert promoted["event_topology"]["a_to_b"]["rescues"] == 22
    assert promoted["event_topology"]["a_to_b"]["harms"] == 0
    assert unchanged["practical_promotion_gate"]["status"] == "fail"
    assert unchanged["order_mechanism_gate"]["status"] == "fail"
