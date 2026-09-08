"""Independent synthetic contracts; no training, optimizers, or real-data access."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiments/fog_motion_factorization_v1.yaml"
PROTOCOL_PATH = ROOT / "docs/research/FOG_MOTION_FACTORIZATION_V1_PROTOCOL.md"


def _config() -> dict[str, Any]:
    loaded = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


@pytest.fixture
def motion() -> Any:
    """Allow configuration tests to run before the separately owned implementation lands."""
    return importlib.import_module("inclusive_shift_har.experiments.fog_motion_factorization")


def test_config_and_protocol_have_independently_verified_bidirectional_hash_binding() -> None:
    config = _config()
    assert config["protocol_path"] == PROTOCOL_PATH.relative_to(ROOT).as_posix()
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    assert config["protocol_sha256"] == hashlib.sha256(protocol_bytes).hexdigest()
    identity = {key: value for key, value in config.items() if key != "protocol_sha256"}
    config_hash = hashlib.sha256(
        json.dumps(
            identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    assert config_hash in protocol_bytes.decode("utf-8")


def test_config_locks_source_partitions_common_support_and_twenty_fit_attempts() -> None:
    config = _config()
    assert config["source"]["raw_sha256"] == (
        "888875133fef386affddd0a5864f122d527d8d7bc588a69905cc0871bef53477"
    )
    assert config["source"]["observable_count"] == 1939
    assert config["source"]["scored_count"] == 1213
    assert config["source"]["scored_class_counts"] == [954, 74, 185]
    assert config["source"]["download_allowed"] is False
    assert config["source"]["inclusivehar_p11_p20_access"] is False
    assert config["folds"]["held_out_participants"] == {
        0: ["fogstar:007", "fogstar:012", "fogstar:016", "fogstar:017", "fogstar:018"],
        1: ["fogstar:001", "fogstar:003", "fogstar:014", "fogstar:015", "fogstar:020"],
        2: ["fogstar:006", "fogstar:010", "fogstar:019", "fogstar:021"],
        3: ["fogstar:002", "fogstar:004", "fogstar:009", "fogstar:011"],
        4: ["fogstar:005", "fogstar:008", "fogstar:013", "fogstar:022"],
    }
    held_out = [
        person for fold in config["folds"]["held_out_participants"].values() for person in fold
    ]
    assert sorted(held_out) == [f"fogstar:{index:03d}" for index in range(1, 23)]
    assert config["folds"]["seed_base"] == 11
    assert config["folds"]["seed_rule"] == "11_plus_zero_based_outer_fold_index"
    assert config["folds"]["inner_selection"] is False
    context = config["context"]
    assert (context["long_samples"], context["query_samples"], context["history_samples"]) == (
        500,
        128,
        372,
    )
    assert context["common_training_and_intervention_support"] is True
    assert context["expected_scored_full_history"] == 1098
    assert context["expected_scored_short_history_fallback"] == 56
    assert context["expected_scored_missing_ankle_fallback"] == 59
    assert context["expected_training_rows_by_fold"] == [904, 780, 902, 780, 1026]
    assert context["expected_training_original_class_counts_by_fold"] == [
        [762, 19, 123],
        [641, 28, 111],
        [753, 21, 128],
        [619, 28, 133],
        [849, 32, 145],
    ]
    assert context["activity_labels_may_define_history_boundaries"] is False
    assert context["concatenate_cached_windows"] is False
    assert context["bridge_gaps_or_nonfinite_resets"] is False
    assert config["cells"]["order"] == ["E2", "T128", "T500", "T500-P"]
    assert config["cells"]["logistic_fits"] == 5
    assert config["cells"]["neural_fits"] == 15
    assert config["runtime"]["maximum_fit_attempts"] == 20
    assert config["runtime"]["count_failed_fit_attempts"] is True
    assert config["runtime"]["automatic_retry"] is False
    assert config["runtime"]["additional_seed"] is False
    assert config["runtime"]["automatic_follow_on"] is False


def test_config_preserves_exact_estimator_weighting_and_optimization_contract() -> None:
    config = _config()
    e2 = config["e2"]
    assert e2["feature_names"] == ["accelerometer_norm__rms", "gyroscope_norm__rms"]
    for key, expected in {
        "rms_floor": 1e-8,
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "fit_intercept": True,
        "max_iter": 1000,
        "tol": 1e-6,
        "class_weight": None,
        "warm_start": False,
    }.items():
        assert e2[key] == expected
    weighting = config["weighting"]
    assert weighting["original_class_order"] == ["mobility", "sitting", "standing"]
    assert weighting["raw_weight"] == "1 / (k_i * n_ic)"
    assert weighting["recompute_after_common_support_restriction"] is True
    assert weighting["binary_rebalancing"] is False
    assert weighting["class_weight"] is None
    training = config["training"]
    assert training["epochs"] == 80
    assert training["checkpoint"] == "last_epoch_only"
    assert training["learning_rate"] == 1e-3
    assert training["minimum_learning_rate"] == 1e-5
    assert training["batch_size"] == 32
    assert training["weight_decay"] == 1e-4
    assert training["minibatch_loss"] == "sum_weight_times_BCE_divided_by_sum_weight_in_minibatch"
    assert training["use_evaluation_labels_for_training_or_selection"] is False
    assert training["use_evaluation_probabilities_for_training_or_selection"] is False
    assert training["identical_initial_state_across_temporal_cells"] is True
    assert training["identical_batch_orders_across_temporal_cells"] is True


def test_config_gates_metrics_and_claim_limits_match_prospective_specification() -> None:
    config = _config()
    gates = config["gates"]
    assert gates["primary_candidate"] == "T500"
    assert [claim["comparator"] for claim in gates["sequential_claims"]] == [
        "l9v",
        "E2",
        "T128",
        "T500-P",
    ]
    assert [claim["minimum_mean_gain"] for claim in gates["sequential_claims"]] == [
        0.015,
        0.010,
        0.010,
        0.010,
    ]
    assert gates["every_stage"]["every_leave_one_person_and_fold_mean_greater_than"] == 0.0
    assert gates["every_stage"]["minimum_strict_participant_wins"] == 14
    assert gates["every_stage"]["minimum_paired_individual_difference"] == -0.05
    assert gates["stop_claim_progression_at_first_failure"] is True
    assert gates["no_automatic_alternate_primary_selection"] is True
    metrics = config["metrics"]
    assert metrics["strict_win_epsilon"] == 1e-12
    assert metrics["bottom_count"] == 7
    assert metrics["ece_bins"] == 10
    assert metrics["ece_aggregations"] == ["pooled_windows", "equal_person"]
    assert metrics["binary_threshold"] == 0.5
    assert metrics["binary_threshold_equality"] == "mobility"
    assert metrics["bootstrap"]["resamples"] == 10_000
    assert metrics["bootstrap"]["seed"] == 1729
    assert metrics["bootstrap"]["participants_per_resample"] == 22
    assert not any(config["claims"].values())


def test_composition_preserves_posture_ratio_and_both_baseline_fallbacks(motion: Any) -> None:
    baseline = np.asarray(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.25, 0.25, 0.5],
            [1.0, 0.0, 0.0],
            [0.6, 0.2, 0.2],
            [0.4, 0.3, 0.3],
        ],
        dtype=np.float64,
    )
    baseline_before = baseline.copy()
    b0 = np.tile(np.asarray([0.6, 0.2, 0.2]), (len(baseline), 1))
    motion_mass = np.asarray([0.2, 0.8, 0.9, 0.4, 0.99, 1.0])
    intervention = np.asarray([True, True, False, True, False, True])
    result = motion.compose_motion_probability(motion_mass, baseline, intervention, b0)

    # Full-history rows get only new motion mass; q has no extra epsilon or routing rule.
    np.testing.assert_allclose(result[0], [0.2, 0.8 * (2 / 3), 0.8 * (1 / 3)])
    np.testing.assert_allclose(result[1], [0.8, 0.15, 0.05])
    np.testing.assert_array_equal(result[5], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-15, rtol=0.0)

    # Missing long history retains L9v; missing ankle retains its existing exact B0 bytes.
    assert result[2].tobytes() == baseline[2].tobytes()
    assert result[4].tobytes() == b0[4].tobytes()
    # Undefined q must preserve the whole L9v vector, despite an eligible intervention mask.
    assert result[3].tobytes() == baseline[3].tobytes()
    np.testing.assert_array_equal(baseline, baseline_before)


def test_fixed_posture_ratio_does_not_promise_unchanged_class_decisions(motion: Any) -> None:
    baseline = np.asarray([[0.4, 0.36, 0.24], [0.4, 0.36, 0.24]])
    masses = np.asarray([0.3, 0.7])
    result = motion.compose_motion_probability(
        masses, baseline, np.ones(2, dtype=np.bool_), baseline.copy()
    )
    np.testing.assert_allclose(result[:, 1] / result[:, 1:].sum(axis=1), [0.6, 0.6])
    assert np.argmax(result, axis=1).tolist() == [1, 0]


def test_tiny_positive_stationary_mass_is_not_replaced_by_a_uniform_q(motion: Any) -> None:
    baseline = np.asarray([[1.0 - 4e-12, 1e-12, 3e-12]])
    result = motion.compose_motion_probability(
        np.asarray([0.2]), baseline, np.ones(1, dtype=np.bool_), baseline.copy()
    )
    np.testing.assert_allclose(result[0], [0.2, 0.2, 0.6], atol=1e-15, rtol=0.0)


def test_weights_equal_people_and_original_classes_without_binary_rebalancing(motion: Any) -> None:
    labels = np.asarray([0, 0, 1, 2, 0, 0, 0, 2, 1], dtype=np.int64)
    people = np.asarray(["A", "A", "A", "A", "B", "B", "B", "B", "C"])
    result = motion.participant_class_weights(labels, people)
    # A has 3 represented original classes, B has 2, C has 1. Each person gets total 3.
    np.testing.assert_allclose(result, [0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 0.5, 1.5, 3.0])
    assert result.mean() == pytest.approx(1.0)
    for person in np.unique(people):
        assert result[people == person].sum() == pytest.approx(3.0)
        represented = np.unique(labels[people == person])
        for label in represented:
            cell = (people == person) & (labels == label)
            assert result[cell].sum() == pytest.approx(3.0 / len(represented))
    assert result[labels == 0].sum() == pytest.approx(2.5)
    assert result[labels != 0].sum() == pytest.approx(6.5)


def test_weighting_uses_classes_after_eligibility_and_is_row_order_equivariant(motion: Any) -> None:
    labels = np.asarray([0, 0, 1, 2, 0, 2], dtype=np.int64)
    people = np.asarray(["A", "A", "A", "A", "B", "B"])
    eligible = np.asarray([True, True, False, False, True, True])
    selected_labels, selected_people = labels[eligible], people[eligible]
    # A's omitted classes must not retain any weight mass after the common-support mask.
    np.testing.assert_allclose(
        motion.participant_class_weights(selected_labels, selected_people), np.ones(4)
    )
    order = np.asarray([5, 3, 0, 2, 1, 4])
    expected = motion.participant_class_weights(labels, people)
    reordered = motion.participant_class_weights(labels[order], people[order])
    np.testing.assert_allclose(reordered, expected[order])


@pytest.mark.parametrize("window_id", ["fixture-A", "fixture-B", "fogstar:007:123.45"])
def test_history_permutation_matches_independent_sha256_definition(
    window_id: str, motion: Any
) -> None:
    expected = sorted(
        range(3),
        key=lambda index: (
            hashlib.sha256(f"motion-history-v1|{window_id}|{index}".encode()).digest(),
            index,
        ),
    )
    first = motion.history_block_permutation(window_id)
    second = motion.history_block_permutation(window_id)
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)
    assert sorted(first) == [0, 1, 2]


def test_history_permutation_preserves_joint_sample_multiset_masks_and_current_query(
    motion: Any,
) -> None:
    values = np.arange(500 * 6, dtype=np.float64).reshape(500, 6)
    mask = np.arange(500) % 7 != 0
    original_values, original_mask = values.copy(), mask.copy()
    permutation = np.asarray([2, 0, 1], dtype=np.int64)
    shuffled, shuffled_mask = motion.apply_history_block_permutation(values, mask, permutation)
    expected_order = np.concatenate(
        [np.arange(248, 372), np.arange(0, 124), np.arange(124, 248), np.arange(372, 500)]
    )
    np.testing.assert_array_equal(shuffled, values[expected_order])
    np.testing.assert_array_equal(shuffled_mask, mask[expected_order])
    np.testing.assert_array_equal(shuffled[372:], values[372:])
    np.testing.assert_array_equal(shuffled_mask[372:], mask[372:])
    np.testing.assert_array_equal(np.sort(shuffled[:372], axis=0), values[:372])
    np.testing.assert_array_equal(values, original_values)
    np.testing.assert_array_equal(mask, original_mask)


def test_identity_history_permutation_is_retained_without_retry(motion: Any) -> None:
    values = np.arange(500 * 6, dtype=np.float64).reshape(500, 6)
    mask = np.ones(500, dtype=np.bool_)
    shuffled, shuffled_mask = motion.apply_history_block_permutation(
        values, mask, np.arange(3, dtype=np.int64)
    )
    np.testing.assert_array_equal(shuffled, values)
    np.testing.assert_array_equal(shuffled_mask, mask)


@pytest.fixture
def cpu_torch() -> Iterator[Any]:
    """Bound tiny forward/derivative checks to one CPU worker and restore global state."""
    torch = pytest.importorskip("torch")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(29)
            yield torch
    finally:
        torch.set_num_threads(previous_threads)


def test_temporal_architecture_has_fixed_width_and_505_sample_receptive_field(
    cpu_torch: Any, motion: Any
) -> None:
    model = motion.MotionResidualTCN().cpu().eval()
    convolutions = [layer for layer in model.modules() if isinstance(layer, cpu_torch.nn.Conv1d)]
    stems = [layer for layer in convolutions if layer.kernel_size == (1,)]
    temporal = [layer for layer in convolutions if layer.kernel_size == (5,)]
    assert len(stems) == 1
    assert stems[0].in_channels == 7  # Six signals and their observation-mask channel.
    assert stems[0].out_channels == 32
    assert sum(parameter.numel() for parameter in model.parameters()) == 62_881
    assert len(temporal) == 12
    assert [layer.dilation[0] for layer in temporal] == [
        1,
        1,
        2,
        2,
        4,
        4,
        8,
        8,
        16,
        16,
        32,
        32,
    ]
    assert all(layer.in_channels == layer.out_channels == 32 for layer in temporal)
    assert all(layer.stride == (1,) for layer in temporal)
    assert 1 + sum((layer.kernel_size[0] - 1) * layer.dilation[0] for layer in temporal) == 505
    norms = [layer for layer in model.modules() if isinstance(layer, cpu_torch.nn.LayerNorm)]
    assert norms and all(layer.normalized_shape == (32,) for layer in norms)
    assert not any(isinstance(layer, cpu_torch.nn.BatchNorm1d) for layer in model.modules())


def test_zero_initialized_residual_preserves_frozen_e2_logits(cpu_torch: Any, motion: Any) -> None:
    model = motion.MotionResidualTCN().cpu().eval()
    values = cpu_torch.randn(2, 500, 6)
    mask = cpu_torch.ones(2, 500, dtype=cpu_torch.bool)
    mask[0, :372] = False
    e2_logits = cpu_torch.tensor([-1.5, 2.0])
    with cpu_torch.inference_mode():
        result = model(values, mask, e2_logits)
    assert result.shape == (2,)
    cpu_torch.testing.assert_close(result, e2_logits, atol=0.0, rtol=0.0)


def test_masked_prefix_cannot_influence_observed_encoder_states(
    cpu_torch: Any, motion: Any
) -> None:
    model = motion.MotionResidualTCN().cpu().eval()
    values = cpu_torch.randn(1, 500, 6)
    mask = cpu_torch.ones(1, 500, dtype=cpu_torch.bool)
    mask[:, :372] = False
    changed = values.clone()
    changed[:, :372] = cpu_torch.randn(1, 372, 6) * 10_000
    with cpu_torch.inference_mode():
        original_encoding = model.encode_sequence(values, mask)
        changed_encoding = model.encode_sequence(changed, mask)
    assert original_encoding.shape == (1, 500, 32)
    assert cpu_torch.count_nonzero(original_encoding[:, :372]).item() == 0
    cpu_torch.testing.assert_close(original_encoding, changed_encoding, atol=0.0, rtol=0.0)


def test_encoder_has_no_future_resampled_sample_dependency(cpu_torch: Any, motion: Any) -> None:
    model = motion.MotionResidualTCN().cpu().eval()
    values = cpu_torch.randn(1, 500, 6)
    mask = cpu_torch.ones(1, 500, dtype=cpu_torch.bool)
    changed = values.clone()
    changed[:, 301:] += cpu_torch.randn(1, 199, 6) * 100
    with cpu_torch.inference_mode():
        original_encoding = model.encode_sequence(values, mask)
        changed_encoding = model.encode_sequence(changed, mask)
    cpu_torch.testing.assert_close(
        original_encoding[:, :301], changed_encoding[:, :301], atol=0.0, rtol=0.0
    )
    assert not cpu_torch.equal(original_encoding[:, 301:], changed_encoding[:, 301:])


def test_encoder_can_use_earliest_real_history_but_not_masked_history(
    cpu_torch: Any, motion: Any
) -> None:
    """Autograd proves a dependency; no optimizer or model-fitting operation is used."""
    model = motion.MotionResidualTCN().cpu().eval()
    values = cpu_torch.randn(1, 500, 6, requires_grad=True)
    full_mask = cpu_torch.ones(1, 500, dtype=cpu_torch.bool)
    encoded = model.encode_sequence(values, full_mask)
    # One channel avoids the cancellation that summing a LayerNorm vector would create.
    full_gradient = cpu_torch.autograd.grad(encoded[0, -1, 0], values)[0]
    assert cpu_torch.count_nonzero(full_gradient[0, 0]).item() > 0
    assert cpu_torch.isfinite(full_gradient).all().item()

    short_mask = full_mask.clone()
    short_mask[:, :372] = False
    short_encoded = model.encode_sequence(values, short_mask)
    short_gradient = cpu_torch.autograd.grad(short_encoded[0, -1, 0], values)[0]
    assert cpu_torch.count_nonzero(short_gradient[0, :372]).item() == 0
    assert cpu_torch.count_nonzero(short_gradient[0, 372:]).item() > 0


def test_temporal_readout_pools_only_current_128_outputs(
    cpu_torch: Any, monkeypatch: pytest.MonkeyPatch, motion: Any
) -> None:
    model = motion.MotionResidualTCN().cpu().eval()
    heads = [
        layer
        for layer in model.modules()
        if isinstance(layer, cpu_torch.nn.Linear) and layer.out_features == 1
    ]
    assert len(heads) == 1
    with cpu_torch.no_grad():
        heads[0].weight.zero_()
        heads[0].weight[0, 0] = 2.0
        if heads[0].bias is not None:
            heads[0].bias.zero_()

    def encode_fixture(values: Any, observation_mask: Any) -> Any:
        assert values.shape == (2, 500, 6)
        assert observation_mask.shape == (2, 500)
        hidden = cpu_torch.zeros(2, 500, 32)
        hidden[:, :372, 0] = 100.0
        hidden[:, 372:, 0] = 0.75
        return hidden

    monkeypatch.setattr(model, "encode_sequence", encode_fixture)
    e2_logits = cpu_torch.tensor([-2.0, 3.0])
    with cpu_torch.inference_mode():
        result = model(
            cpu_torch.zeros(2, 500, 6),
            cpu_torch.ones(2, 500, dtype=cpu_torch.bool),
            e2_logits,
        )
    cpu_torch.testing.assert_close(result, e2_logits + 1.5, atol=0.0, rtol=0.0)
