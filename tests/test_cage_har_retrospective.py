from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from inclusive_shift_har.experiments.cage_har import _load_config, synthetic_cage_bundle
from inclusive_shift_har.experiments.cage_har_retrospective import (
    _RETROSPECTIVE_METHODS,
    evaluate_cage_outer_arrays,
    load_cage_retrospective_config,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_retrospective_config_preserves_evidence_boundary() -> None:
    config = load_cage_retrospective_config(
        _root() / "configs/experiments/cage_har_retrospective_v1.yaml"
    )
    assert config["fixed_seeds"] == [11, 23, 47, 89, 131]
    assert config["participant_ids"] == [str(index) for index in range(1, 11)]
    assert config["claim_policy"]["participants_11_through_20_may_be_loaded"] is False
    assert config["claim_policy"]["daghar_may_be_loaded"] is False
    assert config["preliminary_probe_disclosure"]["probe_is_publication_evidence"] is False


def test_retrospective_config_rejects_claim_boundary_change(tmp_path: Path) -> None:
    source = _root() / "configs/experiments/cage_har_retrospective_v1.yaml"
    text = source.read_text(encoding="utf-8").replace(
        "state_of_the_art_claim_allowed: false",
        "state_of_the_art_claim_allowed: true",
    )
    changed = tmp_path / "changed.yaml"
    changed.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="claim boundary"):
        load_cage_retrospective_config(changed)


def test_outer_array_evaluation_has_all_frozen_methods_and_no_eval_labels() -> None:
    bundle = synthetic_cage_bundle(seed=90210)
    cage_config = _load_config(_root() / "configs/experiments/cage_har_v1.yaml")
    training = np.isin(bundle.participant_ids, np.unique(bundle.participant_ids)[:8])
    evaluation = ~training
    methods, diagnostics, arrays = evaluate_cage_outer_arrays(
        training_base_probability=bundle.base_probability[training],
        training_expert_probability=bundle.expert_probabilities[training, 0],
        training_labels=bundle.labels[training],
        training_participants=bundle.participant_ids[training],
        training_context=bundle.context_features[training],
        training_reliability=bundle.expert_reliability[training, 0],
        evaluation_base_probability=bundle.base_probability[evaluation],
        evaluation_expert_probability=bundle.expert_probabilities[evaluation, 0],
        evaluation_ctgr_probability=bundle.base_probability[evaluation],
        evaluation_context=bundle.context_features[evaluation],
        evaluation_reliability=bundle.expert_reliability[evaluation, 0],
        expert_name="synthetic-gravity",
        cage_config=cage_config,
    )
    assert tuple(methods) == _RETROSPECTIVE_METHODS
    assert diagnostics["outer_labels_used_for_training_selection_or_repair"] is False
    assert diagnostics["router_jackknife_model_count"] == 8
    assert set(arrays) >= {
        "cage_har_route_mask",
        "cage_mix_weight",
        "cage_predicted_advantage_lower_bound",
    }
    for probability in methods.values():
        assert probability.shape == (int(evaluation.sum()), 3)
        np.testing.assert_allclose(probability.sum(axis=1), 1.0)


def test_outer_predictions_do_not_depend_on_unprovided_evaluation_labels() -> None:
    bundle = synthetic_cage_bundle(seed=12345)
    cage_config = _load_config(_root() / "configs/experiments/cage_har_v1.yaml")
    participants = np.unique(bundle.participant_ids)
    training = np.isin(bundle.participant_ids, participants[:8])
    evaluation = ~training
    kwargs = {
        "training_base_probability": bundle.base_probability[training],
        "training_expert_probability": bundle.expert_probabilities[training, 0],
        "training_labels": bundle.labels[training],
        "training_participants": bundle.participant_ids[training],
        "training_context": bundle.context_features[training],
        "training_reliability": bundle.expert_reliability[training, 0],
        "evaluation_base_probability": bundle.base_probability[evaluation],
        "evaluation_expert_probability": bundle.expert_probabilities[evaluation, 0],
        "evaluation_ctgr_probability": bundle.base_probability[evaluation],
        "evaluation_context": bundle.context_features[evaluation],
        "evaluation_reliability": bundle.expert_reliability[evaluation, 0],
        "expert_name": "synthetic-gravity",
        "cage_config": deepcopy(cage_config),
    }
    first, _, _ = evaluate_cage_outer_arrays(**kwargs)
    second, _, _ = evaluate_cage_outer_arrays(**kwargs)
    for name in _RETROSPECTIVE_METHODS:
        np.testing.assert_array_equal(first[name], second[name])
