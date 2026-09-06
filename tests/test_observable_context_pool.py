from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inclusive_shift_har.data.external_har import (
    ExternalHARWindows,
    ObservableWindowPool,
    SourceReceipt,
    _WindowAccumulator,
    observable_modelling_pool,
    resample_physical_segment,
)
from inclusive_shift_har.data.participant_partitions import build_participant_partition_plan
from inclusive_shift_har.experiments import cross_dataset_har, cross_dataset_transfer
from inclusive_shift_har.models.hera_ctgr import (
    GravityKinematicContext,
    ResponderSignatures,
    apply_physics_reference,
    build_responder_signatures,
    extract_gravity_kinematic_context,
    fit_physics_reference,
)


def _annotated_accumulator(labels: np.ndarray) -> _WindowAccumulator:
    time = np.arange(640, dtype=np.float64) / 60
    acceleration = np.column_stack((np.sin(time), np.cos(time), np.full(time.size, 9.81)))
    segment = resample_physical_segment(
        timestamps=time,
        acceleration=acceleration,
        gyroscope=acceleration / 20,
        gravity=None,
        source_rate_hz=60,
        target_rate_hz=50,
        gravity_cutoff_hz=0.3,
        window_samples=128,
    )
    accumulator = _WindowAccumulator.empty()
    accumulator.add_annotated_segment(
        segment,
        source_labels=labels,
        label_map={1: 0, 2: 1, 3: 2},
        participant="p",
        session="s",
        trial="t",
        run_index=0,
    )
    return accumulator


@pytest.mark.parametrize("mutation", ["transition", "missing", "removed", "unsupported"])
def test_candidate_pool_and_context_are_annotation_invariant(mutation: str) -> None:
    labels = np.ones(640)
    changed = labels.copy()
    if mutation == "transition":
        changed[170:390] = 2
    elif mutation == "missing":
        changed[190] = np.nan
    elif mutation == "removed":
        changed[:] = np.nan
    else:
        changed[170:390] = -1
    left, right = _annotated_accumulator(labels), _annotated_accumulator(changed)
    a, b = left.observable_pool(), right.observable_pool()
    assert a is not None and b is not None
    assert a.audit() == b.audit()
    assert len(left.labels) != len(right.labels)
    assert "labels" not in {field.name for field in fields(ObservableWindowPool)}
    probability = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.4, 0.3, 0.3]])

    def signature(pool: ObservableWindowPool) -> np.ndarray:
        context = extract_gravity_kinematic_context(pool.signals, pool.gravity, sampling_rate_hz=50)
        return build_responder_signatures(
            probability,
            probability[:, [0, 2, 1]],
            probability,
            probability,
            pool.participant_ids,
            gravity_reliability=np.ones(4),
            kinematic_context=context,
            physics_trusted=np.ones(4, dtype=bool),
            low_confidence_threshold=0.6,
        ).features

    np.testing.assert_array_equal(signature(a), signature(b))


def _data(dataset: str, *, with_pool: bool = True) -> ExternalHARWindows:
    count = 48
    signals = np.random.default_rng(129).normal(size=(count, 128, 6)).astype(np.float32)
    gravity = np.zeros((count, 128, 3), dtype=np.float32)
    gravity[:, :, 2] = 9.81
    persons = np.repeat([f"{dataset}:p{i:02d}" for i in range(12)], 4)
    pool = ObservableWindowPool(
        signals=signals,
        gravity=gravity,
        participant_ids=persons,
        session_ids=np.char.add(persons, ":s"),
        trial_ids=np.char.add(persons, ":t"),
        window_ids=np.array([f"{dataset}:w{i}" for i in range(count)]),
    )
    eligible = np.arange(count) % 4 != 3
    return ExternalHARWindows(
        dataset_id=dataset,
        channel_lane="derived-gravity-9ch",
        sampling_rate_hz=50,
        signals=signals[eligible],
        gravity=gravity[eligible],
        labels=np.tile(np.arange(3), 12),
        participant_ids=persons[eligible],
        session_ids=pool.session_ids[eligible],
        trial_ids=pool.trial_ids[eligible],
        window_ids=pool.window_ids[eligible],
        receipts=(
            SourceReceipt(dataset, "https://example.invalid/synthetic", None, 1, 1, "0" * 64),
        ),
        observable_candidates=pool if with_pool else None,
        participant_partition_plan=build_participant_partition_plan(
            dataset,
            persons,
            roster_basis="synthetic observable participant roster",
        ),
    )


def test_supervised_masks_exclude_placeholder_labels_and_target_has_no_real_labels() -> None:
    data = _data("fog_star_v3")
    data.validate()
    modelling, score, eligible = observable_modelling_pool(data, include_supervised_labels=True)
    assert modelling.labels.size == 48
    assert eligible.sum() == 36
    np.testing.assert_array_equal(modelling.labels[score], data.labels)
    np.testing.assert_array_equal(modelling.signals[score], data.signals)
    target, target_score, _ = observable_modelling_pool(data, include_supervised_labels=False)
    assert np.all(target.labels == 0)
    np.testing.assert_array_equal(target_score, score)


def test_scoring_annotation_changes_cannot_rephase_inference_pool() -> None:
    data = _data("fog_star_v3")
    retained = np.arange(data.labels.size) % 3 != 1
    changed = replace(
        data,
        **{
            name: getattr(data, name)[retained]
            for name in (
                "signals",
                "gravity",
                "labels",
                "participant_ids",
                "session_ids",
                "trial_ids",
                "window_ids",
            )
        },
    )
    left, first_indices, _ = observable_modelling_pool(data, include_supervised_labels=False)
    right, second_indices, _ = observable_modelling_pool(changed, include_supervised_labels=False)
    assert first_indices.size == 36 and second_indices.size == 24
    for name in (
        "signals",
        "gravity",
        "labels",
        "participant_ids",
        "session_ids",
        "trial_ids",
        "window_ids",
    ):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))


def test_zero_scored_participant_remains_in_observable_roster_and_partition_plan() -> None:
    data = _data("fog_star_v3")
    removed_person = "fog_star_v3:p11"
    keep = data.participant_ids != removed_person
    changed = replace(
        data,
        **{
            name: getattr(data, name)[keep]
            for name in (
                "signals",
                "gravity",
                "labels",
                "participant_ids",
                "session_ids",
                "trial_ids",
                "window_ids",
            )
        },
    )
    reference, _, _ = observable_modelling_pool(data, include_supervised_labels=True)
    modelling, score, eligible = observable_modelling_pool(changed, include_supervised_labels=True)
    np.testing.assert_array_equal(reference.window_ids, modelling.window_ids)
    np.testing.assert_array_equal(reference.signals, modelling.signals)
    np.testing.assert_array_equal(reference.gravity, modelling.gravity)
    removed = modelling.participant_ids == removed_person
    assert removed.any()
    assert not eligible[removed].any()
    assert not np.isin(score, np.flatnonzero(removed)).any()
    plan = modelling.participant_partition_plan
    assert plan is not None and plan is data.participant_partition_plan
    assert removed_person in plan.participant_roster


def test_pool_validation_rejects_divergent_scored_values() -> None:
    data = _data("fog_star_v3")
    changed = data.signals.copy()
    changed[0, 0, 0] += 1
    with pytest.raises(ValueError, match="candidate values differ"):
        replace(data, signals=changed).validate()


class _BeforeFit(Exception):
    pass


def test_nested_runner_builds_all_candidate_features_but_never_trains_on_unknowns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _data("fog_star_v3")

    def inspect(**kwargs: Any) -> None:
        assert kwargs["labels"].size == 48
        assert not kwargs["outer_training"][np.arange(48) % 4 == 3].any()
        assert kwargs["outer_training"].sum() == 27
        assert kwargs["outer_training_candidates"].sum() == 36
        assert kwargs["outer_training_candidates"][np.arange(48) % 4 == 3].sum() == 9
        assert np.all(kwargs["outer_training"] <= kwargs["outer_training_candidates"])
        raise _BeforeFit

    monkeypatch.setattr(cross_dataset_har, "_inner_oof_and_selection", inspect)
    with pytest.raises(_BeforeFit):
        cross_dataset_har._evaluate_seed(
            data,
            seed=11,
            outer_fold_count=5,
            inner_fold_count=4,
            n_jobs=1,
            repository_root=Path(__file__).resolve().parents[1],
            include_classical=True,
        )


def test_cage_context_scale_uses_annotation_independent_training_candidates() -> None:
    data = _data("fog_star_v3")
    modelling, _, eligibility = observable_modelling_pool(data, include_supervised_labels=True)
    plan = modelling.participant_partition_plan
    assert plan is not None
    assignment = plan.resolve(plan.participant_roster, fold_count=5, seed=11, role="outer")
    evaluation_ids = {person for person, fold in assignment.items() if fold == 0}
    evaluation = np.isin(modelling.participant_ids, sorted(evaluation_ids))
    training_candidates, annotation_selected_training = cross_dataset_har._outer_population_masks(
        evaluation, eligibility
    )
    changed_eligibility = eligibility.copy()
    changed_eligibility[training_candidates] = ~changed_eligibility[training_candidates]
    changed_candidates, changed_supervision = cross_dataset_har._outer_population_masks(
        evaluation, changed_eligibility
    )
    assert np.any(training_candidates & ~eligibility)
    assert not np.array_equal(training_candidates, annotation_selected_training)
    np.testing.assert_array_equal(training_candidates, changed_candidates)
    assert not np.array_equal(annotation_selected_training, changed_supervision)
    original = cross_dataset_har._outer_context(
        modelling.signals.astype(np.float64),
        modelling.gravity.astype(np.float64),
        training_candidates,
        sampling_rate_hz=modelling.sampling_rate_hz,
    )
    changed = cross_dataset_har._outer_context(
        modelling.signals.astype(np.float64),
        modelling.gravity.astype(np.float64),
        changed_candidates,
        sampling_rate_hz=modelling.sampling_rate_hz,
    )
    for first, second in zip(original, changed, strict=True):
        np.testing.assert_array_equal(first, second)


def test_physics_reference_fit_is_invariant_to_annotation_eligibility() -> None:
    data = _data("fog_star_v3")
    modelling, _, eligibility = observable_modelling_pool(data, include_supervised_labels=True)
    plan = modelling.participant_partition_plan
    assert plan is not None
    assignment = plan.resolve(plan.participant_roster, fold_count=5, seed=11, role="outer")
    evaluation_ids = {person for person, fold in assignment.items() if fold == 0}
    evaluation = np.asarray(
        np.isin(modelling.participant_ids, sorted(evaluation_ids)), dtype=np.bool_
    )
    training_candidates, supervised_training = cross_dataset_har._outer_population_masks(
        evaluation, eligibility
    )
    changed_eligibility = eligibility.copy()
    changed_eligibility[training_candidates] = ~changed_eligibility[training_candidates]
    changed_candidates, changed_supervision = cross_dataset_har._outer_population_masks(
        evaluation, changed_eligibility
    )
    assert not np.array_equal(supervised_training, changed_supervision)
    np.testing.assert_array_equal(training_candidates, changed_candidates)

    index = np.arange(modelling.labels.size, dtype=np.float64)
    normalized_residual = np.where(eligibility, 1.0, 10.0) + index / 10_000.0
    context = GravityKinematicContext(
        features=normalized_residual[:, None],
        normalized_residual_p90=normalized_residual,
        valid_pair_fraction=np.ones(modelling.labels.size, dtype=np.float64),
        feature_names=("synthetic_normalized_residual_p90",),
    )
    parameters = {"veto_quantile": 0.9, "minimum_valid_pair_fraction": 0.5}
    original = cross_dataset_har._fit_observable_physics_reference(
        context, training_candidates, **parameters
    )
    changed = cross_dataset_har._fit_observable_physics_reference(
        context, changed_candidates, **parameters
    )
    assert original == changed
    original_outputs = apply_physics_reference(context, original)
    changed_outputs = apply_physics_reference(context, changed)
    np.testing.assert_array_equal(original_outputs.reliability, changed_outputs.reliability)
    np.testing.assert_array_equal(original_outputs.trusted, changed_outputs.trusted)

    def selected(mask: np.ndarray) -> GravityKinematicContext:
        return replace(
            context,
            features=context.features[mask],
            normalized_residual_p90=context.normalized_residual_p90[mask],
            valid_pair_fraction=context.valid_pair_fraction[mask],
        )

    annotation_selected_original = fit_physics_reference(
        selected(supervised_training), **parameters
    )
    annotation_selected_changed = fit_physics_reference(selected(changed_supervision), **parameters)
    assert (
        annotation_selected_original.normalized_residual_p90_threshold
        != annotation_selected_changed.normalized_residual_p90_threshold
    )


def test_responder_context_keeps_unlabelled_participant_but_utility_fit_excludes_it() -> None:
    signatures = ResponderSignatures(
        participant_ids=np.array(["p1", "p2", "p3"]),
        features=np.arange(12, dtype=np.float64).reshape(3, 4),
        feature_names=("a", "b", "c", "d"),
    )
    features, participant_ids, without_supervision = cross_dataset_har._responder_supervision_rows(
        signatures, np.array(["p1", "p3"])
    )
    np.testing.assert_array_equal(participant_ids, np.array(["p1", "p3"]))
    np.testing.assert_array_equal(features, signatures.features[[0, 2]])
    assert without_supervision == ["p2"]
    assert "p2" in signatures.participant_ids


def test_inner_oof_predicts_every_candidate_but_selects_only_on_eligible_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _data("fog_star_v3")
    modelling, _, eligible = observable_modelling_pool(data, include_supervised_labels=True)
    plan = modelling.participant_partition_plan
    assert plan is not None
    assignment = plan.resolve(plan.participant_roster, fold_count=5, seed=11, role="outer")
    evaluation_ids = {person for person, fold in assignment.items() if fold == 0}
    evaluation = np.isin(modelling.participant_ids, sorted(evaluation_ids))
    training_candidates = ~evaluation
    training = training_candidates & eligible
    feature = np.arange(modelling.labels.size * 4, dtype=np.float64).reshape(-1, 4)
    views = {name: feature for name in cross_dataset_har._VIEW_ORDER}

    class BaseModel:
        classes_ = np.array([0, 1, 2], dtype=np.int64)

        def predict_proba(self, values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            return np.tile(np.array([0.60, 0.25, 0.15]), (values.shape[0], 1))

    class BinaryModel:
        classes_ = np.array([0, 1], dtype=np.int64)

        def predict_proba(self, values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            return np.tile(np.array([0.55, 0.45]), (values.shape[0], 1))

    def fit_base(*_args: Any, **kwargs: Any) -> BaseModel:
        assert not np.any(np.asarray(kwargs.get("training", _args[3])) & ~eligible)
        return BaseModel()

    def fit_expert(*_args: Any, **kwargs: Any) -> BinaryModel:
        assert not np.any(np.asarray(kwargs.get("training", _args[4])) & ~eligible)
        return BinaryModel()

    monkeypatch.setattr(cross_dataset_har, "_fit_base", fit_base)
    monkeypatch.setattr(cross_dataset_har, "_fit_expert", fit_expert)
    config = cross_dataset_har._read_mapping(
        Path(__file__).resolve().parents[1]
        / "configs/experiments/confidence_triggered_gravity_residual_v1.yaml"
    )
    candidates = cross_dataset_har._candidates(config)

    def evaluate(labels: np.ndarray[Any, Any]) -> cross_dataset_har._InnerPredictions:
        return cross_dataset_har._inner_oof_and_selection(
            rmrp=feature,
            views=views,
            dual_features=feature,
            labels=labels,
            participants=modelling.participant_ids,
            outer_training=training,
            outer_training_candidates=training_candidates,
            candidates=candidates,
            outer_index=0,
            seed=11,
            inner_fold_count=4,
            n_jobs=1,
            partition_plan=plan,
            prewindow_training_ids=tuple(sorted(set(assignment) - evaluation_ids)),
            assignment_role="classical_inner_outer_0",
        )

    first = evaluate(modelling.labels)
    changed_labels = modelling.labels.copy()
    changed_labels[~eligible] = np.resize(np.array([2, 1, 0]), int((~eligible).sum()))
    second = evaluate(changed_labels)
    for result in (first, second):
        assert np.isfinite(result.base[training_candidates]).all()
        assert np.isnan(result.base[evaluation]).all()
        assert all(
            record["validation_candidate_window_count"] >= record["validation_scored_window_count"]
            for record in result.folds
        )
        assert sum(
            int(record["validation_candidate_window_count"]) for record in result.folds
        ) == int(training_candidates.sum())
    np.testing.assert_array_equal(first.base[training_candidates], second.base[training_candidates])
    assert first.candidate_ranking == second.candidate_ranking


def test_transfer_runner_materializes_full_target_before_fitting_without_target_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target = _data("imu_har_il_v1", with_pool=False), _data("fog_star_v3")

    def inspect(**kwargs: Any) -> None:
        assert kwargs["labels"].size == 36 + 48
        assert np.all(kwargs["labels"][36:] == 0)
        assert kwargs["outer_training"].sum() == 36
        assert not kwargs["outer_training"][36:].any()
        raise _BeforeFit

    monkeypatch.setattr(cross_dataset_transfer, "_inner_oof_and_selection", inspect)
    with pytest.raises(_BeforeFit):
        cross_dataset_transfer._fit_apply_seed(
            source,
            target,
            seed=11,
            repository_root=Path(__file__).resolve().parents[1],
            n_jobs=1,
            include_classical=True,
        )


@pytest.mark.parametrize("policy", ["complete_requested_core", "available_valid_trials"])
def test_transfer_cli_exposes_cohort_without_changing_seeds(policy: str) -> None:
    args = cross_dataset_transfer._parser().parse_args(
        ["--output-directory", "new-output", "--source-selection-policy", policy]
    )
    assert args.source_selection_policy == policy
    assert args.seeds == [11, 23, 47]
    assert args.source_repetition_limit == 4
