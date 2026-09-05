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
from inclusive_shift_har.experiments import cross_dataset_har, cross_dataset_transfer
from inclusive_shift_har.models.hera_ctgr import (
    build_responder_signatures,
    extract_gravity_kinematic_context,
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
