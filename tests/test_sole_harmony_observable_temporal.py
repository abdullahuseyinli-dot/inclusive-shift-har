from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from shutil import copytree
from typing import Any

import numpy as np
import pytest
import yaml
from numpy.typing import NDArray

from inclusive_shift_har.data.external_har import (
    BOUNDARY_PROVENANCE_PROTOCOL,
    OBSERVABLE_SCORING_ELIGIBILITY_POLICY,
    PHYSICAL_GRID_PROTOCOL,
    ExternalHARWindows,
    ObservableWindowPool,
    SourceReceipt,
    _array_sha256,
    observable_modelling_pool,
)
from inclusive_shift_har.data.participant_partitions import build_participant_partition_plan
from inclusive_shift_har.evaluation.inference_contracts import (
    annotation_selected_context_methods,
    method_inference_contracts,
)
from inclusive_shift_har.experiments import (
    cross_dataset_har,
    external_evidence_validate,
    publication_campaign,
    sole_harmony_temporal,
)
from inclusive_shift_har.experiments.cross_dataset_har import (
    _ObservableClassicalSeedPredictions,
)
from inclusive_shift_har.experiments.external_evidence_validate import (
    _sole_observable_prediction_errors,
    validate_run_directory,
)
from inclusive_shift_har.experiments.publication_split_audit import build_split_audit
from inclusive_shift_har.experiments.sole_harmony_temporal import (
    BASE_METHOD,
    PRIMARY_METHOD,
    _causal_hysteresis_probability,
    _causal_majority_probability,
    _causal_probability,
    _temporal_block_ids,
    evaluate_temporal_lane,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file


def _receipt() -> SourceReceipt:
    return SourceReceipt(
        dataset_id="sole_harmony_v1",
        locator="https://zenodo.org/api/records/19242395/files/C001.zip/content",
        member="C001.zip/session/DataStruct.mat",
        declared_size_bytes=1,
        received_size_bytes=1,
        computed_sha256="0" * 64,
    )


def _sole_session_data() -> ExternalHARWindows:
    participants: list[str] = []
    sessions: list[str] = []
    trials: list[str] = []
    windows: list[str] = []
    full_labels: list[int] = []
    eligible: list[bool] = []
    preprocessing_audit: list[dict[str, Any]] = []
    for participant_index in range(1, 13):
        participant = f"sole:C{participant_index:03d}"
        for session_index in range(2):
            session = f"{participant}:session-day-{session_index + 1}"
            trial = f"{session}:session-recording"
            for window_index, label in enumerate((0, 1, 1, 2)):
                participants.append(participant)
                sessions.append(session)
                trials.append(trial)
                windows.append(
                    f"{participant}/{session}/{trial}/run-0000-finite-000/window-{window_index:06d}"
                )
                full_labels.append(label)
                eligible.append(window_index != 1)
            candidate_starts = np.arange(4, dtype=np.int64) * 128
            preprocessing_audit.append(
                {
                    "protocol_id": PHYSICAL_GRID_PROTOCOL,
                    "participant_id": participant,
                    "session_id": session,
                    "trial_id": trial,
                    "run_index": 0,
                    "resampled_samples": 512,
                    "window_samples": 128,
                    "candidate_start_samples": candidate_starts.tolist(),
                    "candidate_window_count": 4,
                    "candidate_grid_sha256": _array_sha256(candidate_starts),
                    "within_declared_segment_transform_annotation_dependency": False,
                    "segment_boundary_annotation_conditioned": False,
                    "resampling_passes": 1,
                }
            )
    count = len(windows)
    signals = np.zeros((count, 128, 6), dtype=np.float32)
    signals[np.logical_not(eligible), :, :] = 999.0
    gravity = np.ones((count, 128, 3), dtype=np.float32)
    pool = ObservableWindowPool(
        signals=signals,
        gravity=gravity,
        participant_ids=np.asarray(participants, dtype=np.str_),
        session_ids=np.asarray(sessions, dtype=np.str_),
        trial_ids=np.asarray(trials, dtype=np.str_),
        window_ids=np.asarray(windows, dtype=np.str_),
    )
    selected = np.flatnonzero(eligible)
    roster = tuple(f"sole:C{index:03d}" for index in range(1, 13))
    result = ExternalHARWindows(
        dataset_id="sole_harmony_v1",
        channel_lane="derived-gravity-9ch",
        sampling_rate_hz=50.0,
        signals=signals[selected],
        gravity=gravity[selected],
        labels=np.asarray(full_labels, dtype=np.int64)[selected],
        participant_ids=np.asarray(participants, dtype=np.str_)[selected],
        session_ids=np.asarray(sessions, dtype=np.str_)[selected],
        trial_ids=np.asarray(trials, dtype=np.str_)[selected],
        window_ids=np.asarray(windows, dtype=np.str_)[selected],
        receipts=(_receipt(),),
        gravity_source="provider_raw_acceleration_minus_linear_acceleration",
        preprocessing_audit=tuple(preprocessing_audit),
        observable_candidates=pool,
        boundary_provenance={
            "protocol_id": BOUNDARY_PROVENANCE_PROTOCOL,
            "source_boundary_unit": "session, timestamp gap, and finite-sensor run",
            "repository_signal_grid_annotation_independent": True,
            "provider_upstream_annotation_conditioned": False,
            "zero_lookahead_streaming_valid": False,
            "evidence_scope": "synthetic session-observable test",
            "boundary_mode": "session_observable",
            "annotation_application": "camera intervals projected after global signal-grid construction",
        },
        participant_partition_plan=build_participant_partition_plan(
            "sole_harmony_v1",
            roster,
            roster_basis="synthetic requested participant archives fixed before windows",
        ),
    )
    result.validate()
    return result


def _base_probability(count: int) -> NDArray[np.float64]:
    pattern = np.asarray(
        (
            (0.90, 0.05, 0.05),
            (0.05, 0.90, 0.05),
            (0.50, 0.45, 0.05),
            (0.05, 0.10, 0.85),
        ),
        dtype=np.float64,
    )
    return np.tile(pattern, (count // pattern.shape[0], 1))


def _fake_observable_seed(
    data: ExternalHARWindows, *, seed: int, **_kwargs: Any
) -> _ObservableClassicalSeedPredictions:
    modelling, scoring, eligibility = observable_modelling_pool(
        data, include_supervised_labels=True
    )
    plan = data.participant_partition_plan
    assert plan is not None
    plan_hash = plan.audit()["plan_sha256"]
    assignment = plan.resolve(plan.participant_roster, fold_count=5, seed=seed, role="outer")
    folds: list[dict[str, Any]] = []
    for outer_fold in range(5):
        evaluation = {person for person, fold in assignment.items() if fold == outer_fold}
        training = set(assignment) - evaluation
        inner_assignment = plan.resolve(
            sorted(training),
            fold_count=4,
            seed=seed + 10_000 + outer_fold,
            role=f"classical_inner_outer_{outer_fold}",
        )
        inner_folds = []
        for inner_fold in range(4):
            validation = {person for person, fold in inner_assignment.items() if fold == inner_fold}
            inner_folds.append(
                {
                    "inner_fold": inner_fold,
                    "training_participants": sorted(training - validation),
                    "validation_participants": sorted(validation),
                    "participant_partition_plan_sha256": plan_hash,
                }
            )
        folds.append(
            {
                "outer_fold": outer_fold,
                "training_participants": sorted(training),
                "training_participants_with_supervision": sorted(training),
                "evaluation_participants": sorted(evaluation),
                "evaluation_participants_with_candidates": sorted(evaluation),
                "evaluation_participants_with_scoring": sorted(evaluation),
                "participant_partition_plan_sha256": plan_hash,
                "inner_folds": inner_folds,
            }
        )
    return _ObservableClassicalSeedPredictions(
        data=modelling,
        scoring_indices=scoring,
        supervised_eligibility=eligibility,
        probabilities={"XGBoost-6ch": _base_probability(modelling.labels.size)},
        record={
            "seed": seed,
            "participant_partition_plan_sha256": plan_hash,
            "folds": folds,
        },
    )


def test_observable_classical_seed_predicts_full_pool_but_fits_only_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _sole_session_data()
    fitted_training: list[set[str]] = []
    predicted_count = 0

    def fit(
        windows: NDArray[np.float32],
        _labels: NDArray[np.int64],
        participants: list[str],
        **_kwargs: Any,
    ) -> object:
        assert not np.any(windows == 999.0)
        fitted_training.append(set(participants))
        return object()

    def predict(
        _model: object, windows: NDArray[np.float32]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        nonlocal predicted_count
        predicted_count += windows.shape[0]
        probability = np.full((windows.shape[0], 3), 1.0 / 3.0, dtype=np.float64)
        return np.log(probability), probability

    monkeypatch.setattr(cross_dataset_har, "fit_classical_model", fit)
    monkeypatch.setattr(cross_dataset_har, "predict_classical_probabilities", predict)
    result = cross_dataset_har._observable_classical_seed_predictions(data, seed=11)
    plan = data.participant_partition_plan
    assert plan is not None
    assert result.probabilities["XGBoost-6ch"].shape == (96, 3)
    assert result.scoring_indices.size == 72
    assert predicted_count == 96
    assert len(fitted_training) == 5
    assert all(len(participants) >= 9 for participants in fitted_training)
    assert len(result.record["folds"]) == 5
    assert all(len(fold["inner_folds"]) == 4 for fold in result.record["folds"])
    assert all(
        fold["participant_partition_plan_sha256"] == plan.audit()["plan_sha256"]
        for fold in result.record["folds"]
    )
    assert all(
        fold["evaluation_scored_window_count"] < fold["evaluation_candidate_window_count"]
        for fold in result.record["folds"]
    )


def test_observable_temporal_fold_records_pass_publication_split_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _sole_session_data()

    def fit(*_args: Any, **_kwargs: Any) -> object:
        return object()

    def predict(
        _model: object, windows: NDArray[np.float32]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        probability = np.full((windows.shape[0], 3), 1.0 / 3.0, dtype=np.float64)
        return np.log(probability), probability

    monkeypatch.setattr(cross_dataset_har, "fit_classical_model", fit)
    monkeypatch.setattr(cross_dataset_har, "predict_classical_probabilities", predict)
    records = [
        cross_dataset_har._observable_classical_seed_predictions(data, seed=seed).record
        for seed in (11, 23, 47)
    ]
    run = tmp_path / "run"
    run.mkdir()
    prediction = run / "predictions.npz"
    np.savez_compressed(
        prediction,
        labels=data.labels,
        participant_ids=data.participant_ids,
        session_ids=data.session_ids,
        trial_ids=data.trial_ids,
        window_ids=data.window_ids,
        probability__method=np.full((data.labels.size, 3), 1.0 / 3.0),
    )
    result = {
        "experiment_id": "sole-harmony-observable-session-temporal-v1",
        "seeds": [11, 23, 47],
        "dataset": data.summary(),
        "fold_records": records,
        "prediction_artifact": {
            "path": prediction.name,
            "sha256": sha256_file(prediction),
        },
    }
    result["result_payload_sha256_before_serialization"] = canonical_json_sha256(result)
    (run / "result.json").write_text(json.dumps(result), encoding="utf-8")
    audit = build_split_audit(run)
    assert audit["valid"], audit["errors"]
    assert audit["status"] == "PASS_RECORDED_RESULT"
    assert audit["run_kind"] == "classical_nested"
    assert audit["scope"]["pre_window_partition_plan_record_verified"]


def test_temporal_state_uses_ineligible_candidates_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _sole_session_data()
    monkeypatch.setattr(
        sole_harmony_temporal,
        "_observable_classical_seed_predictions",
        _fake_observable_seed,
    )
    root = Path(__file__).resolve().parents[1]
    result, predictions = evaluate_temporal_lane(
        data,
        seeds=(11, 23, 47),
        repository_root=root,
        n_jobs=1,
    )
    method_names = set(result["primary_seed_averaged"]["methods"])
    assert method_names == {
        BASE_METHOD,
        PRIMARY_METHOD,
        "XGBoost-6ch-causal-probability-w3",
        "XGBoost-6ch-causal-probability-w7",
        "XGBoost-6ch-causal-majority-w3",
        "XGBoost-6ch-causal-majority-w5",
        "XGBoost-6ch-causal-majority-w7",
        "XGBoost-6ch-hysteresis-c2",
        "XGBoost-6ch-hysteresis-c3",
    }
    full = predictions["observable-seed-11__XGBoost-6ch-causal-probability-w3"]
    expected = _base_probability(96)[:3].mean(axis=0)
    np.testing.assert_allclose(full[2], expected)
    assert predictions[f"seed-11__{PRIMARY_METHOD}"].shape == (72, 3)
    assert result["prediction_contract"]["observable_candidate_window_count"] == 96
    assert result["prediction_contract"]["scored_window_count"] == 72
    assert result["advancement_gate"]["candidate"] == PRIMARY_METHOD
    assert result["advancement_gate"]["comparator"] == BASE_METHOD
    assert result["advancement_gate"]["applicable"]
    comparison = result["primary_paired_participant_bootstrap"]
    assert comparison["difference_direction"] == "candidate_minus_comparator"
    assert comparison["participant_count"] == 12
    assert set(comparison["participant_difference_values"]) == {
        f"sole:C{index:03d}" for index in range(1, 13)
    }
    supplement = result["participant_statistical_supplement"]
    assert supplement["statistical_unit"] == "participant"
    assert supplement["participants_with_all_declared_classes"] == 12
    assert (
        supplement["methods"][PRIMARY_METHOD][
            "present_true_class_participant_macro_f1_sensitivity"
        ]["participant_n"]
        == 12
    )
    assert set(result["shuffled_time_negative_controls"]) == {
        "width_3",
        "width_5",
        "width_7",
    }
    assert all(
        value["replicate_count"] == 100
        for value in result["shuffled_time_negative_controls"].values()
    )
    assert all(
        record["temporal_state_updated_on_every_observable_candidate"]
        for record in result["fold_records"]
    )

    tampered_shuffle = deepcopy(result["shuffled_time_negative_controls"])
    tampered_shuffle["width_5"]["shuffled_median"] += 0.01
    with pytest.raises(ValueError, match="shuffled-time evidence"):
        sole_harmony_temporal._advancement_gate(
            boundary_mode="session_observable",
            primary=result["primary_seed_averaged"],
            supplement=supplement,
            comparison=comparison,
            shuffled=tampered_shuffle,
        )


def test_temporal_statistics_keep_fixed_and_missing_class_estimands_separate() -> None:
    data = _sole_session_data()
    indices = np.asarray([0, 1, 2, 6], dtype=np.int64)
    sparse = replace(
        data,
        signals=data.signals[indices],
        gravity=data.gravity[indices],
        labels=data.labels[indices],
        participant_ids=data.participant_ids[indices],
        session_ids=data.session_ids[indices],
        trial_ids=data.trial_ids[indices],
        window_ids=data.window_ids[indices],
    )
    probability = np.eye(3, dtype=np.float64)[np.asarray([0, 1, 2, 1], dtype=np.int64)]
    statistics = sole_harmony_temporal._participant_statistical_supplement(
        sparse,
        {11: {"model": probability}, 23: {"model": probability.copy()}},
    )

    assert statistics["perfect_prediction_fixed_class_mean_ceiling"] == pytest.approx(2 / 3)
    assert statistics["participants_with_all_declared_classes"] == 1
    model = statistics["methods"]["model"]
    assert model["fixed_class_participant_macro_f1"]["summary"]["mean"] == pytest.approx(0.5)
    assert model["present_true_class_participant_macro_f1_sensitivity"]["summary"][
        "mean"
    ] == pytest.approx(0.5)
    complete = model["all_declared_classes_supported_participant_sensitivity"]
    assert complete["participant_n"] == 1
    assert complete["eligible_participants"] == ["sole:C001"]
    assert complete["summary"]["mean"] == pytest.approx(1.0)
    sitting = model["per_class_participant_balanced"]["sitting"]
    assert sitting["recall"] == pytest.approx(0.5)
    assert sitting["true_support_eligible_participant_sensitivity"]["participant_n"] == 1
    assert sitting["true_support_eligible_participant_sensitivity"]["recall"] == pytest.approx(1.0)
    assert model["participant_normalized_confusion"][1] == pytest.approx([0.0, 1.0, 0.0])
    assert model["participant_normalized_confusion_eligible_n_by_true_class"] == {
        "mobility": 2,
        "sitting": 1,
        "standing": 1,
    }
    assert "sample-size-sensitive" in model["participant_mean_calibration"]["ece_note"]


def test_primary_pairing_uses_ids_and_declares_lower_tail_estimand() -> None:
    primary = {
        "methods": {
            PRIMARY_METHOD: {"participant_values": {"p2": 0.0, "p1": 1.0}},
            BASE_METHOD: {"participant_values": {"p1": 0.0, "p2": 1.0}},
        }
    }
    comparison = sole_harmony_temporal._paired_primary_comparison(primary)
    assert comparison["participant_difference_values"] == {"p1": 1.0, "p2": -1.0}
    assert comparison["mean_difference"] == pytest.approx(0.0)
    assert comparison[
        "secondary_independently_ranked_bottom_30_percent_distribution_difference"
    ] == pytest.approx(0.0)
    assert comparison["rescue_count"] == comparison["harm_count"] == 1
    assert comparison["tie_count"] == 0
    assert "independently rank" in comparison["secondary_tail_estimand"]


def test_temporal_top_level_report_uses_bottom_ceil_thirty_percent() -> None:
    labels = np.tile(np.arange(3, dtype=np.int64), 4)
    predicted = np.asarray(
        [
            1,
            2,
            0,
            0,
            2,
            0,
            0,
            1,
            0,
            0,
            1,
            2,
        ],
        dtype=np.int64,
    )
    probabilities = np.eye(3, dtype=np.float64)[predicted]
    participants = np.repeat(np.asarray(["p1", "p2", "p3", "p4"]), 3)
    report = sole_harmony_temporal._temporal_report(
        labels,
        probabilities,
        participants,
        ("mobility", "sitting", "standing"),
    )
    participant_values = np.sort(
        np.asarray([row["macro_f1"] for row in report["participants"]], dtype=np.float64)
    )
    assert report["primary"]["bottom_30_percent_participant_macro_f1"] == pytest.approx(
        participant_values[:2].mean()
    )


@pytest.mark.parametrize("mutation", ("outer_folds", "cohort", "base_rationale", "gate"))
def test_protocol_reader_rejects_semantic_drift(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = sole_harmony_temporal._read_protocol(root)
    changed = deepcopy(protocol)
    if mutation == "outer_folds":
        changed["budget"]["outer_folds"] = 4
    elif mutation == "cohort":
        changed["cohort"]["sessions_per_participant"] = "first three ordered sessions"
    elif mutation == "base_rationale":
        changed["base_model"]["rationale"] = "strongest control"
    else:
        changed["advancement_gate"] = changed["advancement_gate"][:-1]
    monkeypatch.setattr(yaml, "safe_load", lambda _text: changed)
    with pytest.raises(ValueError, match="frozen protocol differs"):
        sole_harmony_temporal._read_protocol(root)


def test_sole_evidence_roles_are_explicit_and_lane_scoped() -> None:
    root = Path(__file__).resolve().parents[1]
    ledger = yaml.safe_load(
        (root / "configs/datasets/evidence_roles_20260905_v3.yaml").read_text(encoding="utf-8")
    )
    protocol = sole_harmony_temporal._read_protocol(root)
    provider_audit = yaml.safe_load(
        (root / "configs/protocols/external_har_provider_boundary_audit_v1.yaml").read_text(
            encoding="utf-8"
        )
    )

    sole = ledger["datasets"]["sole_harmony_v1"]
    evidence_role = protocol["evidence_role"]
    provider_role = provider_audit["sole_harmony"]
    assert sole["role"] == evidence_role["dataset_role"] == "development"
    assert provider_role["dataset_role"] == "development"
    assert sole["evidence_status"] == evidence_role["evidence_status"] == "diagnostic"
    assert provider_role["evidence_status"] == "diagnostic"
    assert sole["confirmation_allowed"] is evidence_role["confirmation_allowed"] is False
    assert provider_role["confirmation_allowed"] is False
    assert sole["deployable_claim_allowed"] is False
    assert evidence_role["deployable_claim_allowed"] is False
    assert provider_role["deployable_claim_allowed"] is False
    assert sole["cross_lane_numeric_comparison_allowed"] is False
    assert evidence_role["cross_lane_numeric_comparison_allowed"] is False
    assert provider_role["cross_lane_numeric_comparison_allowed"] is False

    expected_roles = {
        "session_observable": "development",
        "camera_bout_oracle": "oracle_diagnostic",
    }
    assert set(sole["lanes"]) == set(expected_roles)
    assert set(evidence_role["lanes"]) == set(expected_roles)
    assert set(provider_role["lanes"]) == set(expected_roles)
    for lane, expected_role in expected_roles.items():
        ledger_lane = sole["lanes"][lane]
        protocol_lane = evidence_role["lanes"][lane]
        provider_lane = provider_role["lanes"][lane]
        assert ledger_lane["role"] == protocol_lane["role"] == expected_role
        assert provider_lane["role"] == expected_role
        assert ledger_lane["evidence_status"] == "diagnostic"
        assert protocol_lane["evidence_status"] == "diagnostic"
        assert provider_lane["evidence_status"] == "diagnostic"
        assert ledger_lane["confirmation_allowed"] is False
        assert protocol_lane["confirmation_allowed"] is False
        assert ledger_lane["deployable_claim_allowed"] is False
        assert protocol_lane["deployable_claim_allowed"] is False
        assert provider_lane["deployable_claim_allowed"] is False


def test_temporal_lane_rejects_more_than_two_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _sole_session_data()
    modelling, scoring, eligibility = observable_modelling_pool(
        data, include_supervised_labels=True
    )
    session_ids = modelling.session_ids.copy()
    trial_ids = modelling.trial_ids.copy()
    window_ids = modelling.window_ids.copy()
    participant = "sole:C001"
    session = f"{participant}:session-day-3"
    trial = f"{session}:session-recording"
    for local_index, index in enumerate((6, 7)):
        session_ids[index] = session
        trial_ids[index] = trial
        window_ids[index] = (
            f"{participant}/{session}/{trial}/run-0000-finite-000/window-{local_index:06d}"
        )
    three_session_modelling = replace(
        modelling,
        session_ids=session_ids,
        trial_ids=trial_ids,
        window_ids=window_ids,
    )
    monkeypatch.setattr(
        sole_harmony_temporal,
        "observable_modelling_pool",
        lambda *_args, **_kwargs: (three_session_modelling, scoring, eligibility),
    )
    with pytest.raises(ValueError, match="exactly two sessions"):
        sole_harmony_temporal._validate_lane(data, boundary_mode="session_observable")


def test_temporal_run_retains_scored_and_full_observable_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _sole_session_data()
    monkeypatch.setattr(
        sole_harmony_temporal,
        "_observable_classical_seed_predictions",
        _fake_observable_seed,
    )
    monkeypatch.setattr(
        sole_harmony_temporal,
        "_git_state",
        lambda _root: {
            "commit": "0" * 40,
            "worktree_dirty": False,
            "status_entries": [],
        },
    )
    monkeypatch.setattr(sole_harmony_temporal, "_source_manifest_commit_errors", lambda *_args: [])
    root = Path(__file__).resolve().parents[1]
    frozen_manifest = sole_harmony_temporal._source_input_manifest(root)
    monkeypatch.setattr(
        sole_harmony_temporal, "_source_input_manifest", lambda _root: frozen_manifest
    )
    monkeypatch.setattr(external_evidence_validate, "_manifest_commit_errors", lambda *_args: [])
    monkeypatch.setattr(
        external_evidence_validate, "_publication_manifest_errors", lambda *_args: []
    )
    output = tmp_path / "session-observable"
    sole_harmony_temporal.run_and_write(
        data=data,
        output_directory=output,
        repository_root=root,
        seeds=(11, 23, 47),
        n_jobs=1,
    )
    with np.load(output / "predictions.npz", allow_pickle=False) as archive:
        assert archive["labels"].shape == (72,)
        assert archive["observable_window_ids"].shape == (96,)
        assert int(archive["observable_scoring_eligibility"].sum()) == 72
        assert archive[f"probability__seed-11__{PRIMARY_METHOD}"].shape == (72, 3)
        assert archive[f"observable_probability__seed-11__{PRIMARY_METHOD}"].shape == (96, 3)
    result_path = output / "result.json"
    result_text = result_path.read_text(encoding="utf-8")
    payload = json.loads(result_text)
    assert (output / "split_audit.json").is_file()
    assert (output / "validation.json").is_file()
    declared = payload.pop("result_payload_sha256_before_serialization")
    assert declared == canonical_json_sha256(payload)
    assert payload["temporal_contract"]["state_updates_before_scoring_slice"]
    validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    assert validation["integrity_passed"]
    assert validation["scientific_contract_checks"]["metric_reconstruction_errors"] == []

    diagnostic_validation = validation
    assert diagnostic_validation["status"] == "DIAGNOSTIC", diagnostic_validation
    assert diagnostic_validation["publication_evidence_ready"] is False
    assert diagnostic_validation["diagnostic_contract_passed"] is True
    publication_campaign._accept(output, root, diagnostic=True)
    relocated = tmp_path / "renamed-session-observable"
    copytree(output, relocated)
    relocated_validation = external_evidence_validate.validate_and_record_run_directory(
        relocated,
        root,
        diagnostic=True,
    )
    assert relocated_validation["status"] == "DIAGNOSTIC"
    assert relocated_validation["integrity_passed"] is True
    assert relocated_validation["diagnostic_contract_passed"] is True

    mutated = json.loads(result_text)
    mutated["participant_statistical_supplement"]["participant_count"] = 999
    mutated["primary_paired_participant_bootstrap"]["mean_difference"] += 0.01
    mutated["shuffled_time_negative_controls"]["width_5"]["shuffled_mean"] += 0.01
    mutated["advancement_gate"]["decision"] = "FORGED_DECISION"
    mutated.pop("result_payload_sha256_before_serialization")
    mutated["result_payload_sha256_before_serialization"] = canonical_json_sha256(mutated)
    result_path.write_text(json.dumps(mutated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rejected = validate_run_directory(output, root)
    metric_errors = rejected["scientific_contract_checks"]["metric_reconstruction_errors"]
    assert metric_errors
    for field in (
        "participant_statistical_supplement",
        "primary_paired_participant_bootstrap",
        "shuffled_time_negative_controls",
        "advancement_gate",
    ):
        assert field in metric_errors[0]
    result_path.write_text(result_text, encoding="utf-8")


class _MemoryPredictionArchive:
    def __init__(self, values: dict[str, NDArray[Any]]) -> None:
        self.values = values
        self.files = list(values)

    def __getitem__(self, name: str) -> NDArray[Any]:
        return self.values[name]


def _sole_contract_fields(
    boundary_mode: str, methods: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    observable = boundary_mode == "session_observable"
    return {
        "base_model_contract": {
            "method": "XGBoost-6ch",
            "features": "fixed six-channel engineered-feature interface",
            "outer_folds": 5,
            "hyperparameter_trials": 0,
            "inner_fold_assignments": "retained for split audit but unused by fixed control",
            "held_out_predictions_cover_every_observable_candidate": True,
            "outer_training_participant_labels_used_for_supervised_fit": True,
            "outer_evaluation_participant_labels_used_for_fit_selection_or_calibration": False,
        },
        "temporal_contract": {
            "protocol_id": "sole-harmony-observable-session-temporal-v1",
            "boundary_mode": boundary_mode,
            "window_order": (
                "provider participant/session, timestamp-gap or finite-run block, chronological window"
                if observable
                else "provider participant/session, camera-labelled bout, chronological window"
            ),
            "state_reset": (
                "provider session, timestamp discontinuity, or finite-sensor run only"
                if observable
                else "camera-labelled bout or finite sub-run (oracle)"
            ),
            "future_window_access": False,
            "state_updates_before_scoring_slice": True,
            "state_input_population": (
                "all annotation-independent observable candidates"
                if observable
                else "camera-bout-selected oracle candidates"
            ),
            "scoring_eligibility_applied_after_all_temporal_predictions": True,
            "widths_reported_without_outcome_selection": [3, 5, 7],
            "hysteresis_confirmations_reported_without_outcome_selection": [2, 3],
            "primary_candidate": PRIMARY_METHOD,
            "primary_control": BASE_METHOD,
            "offline_symmetric_resampling_precludes_zero_lookahead_claim": True,
        },
        "comparison_contract": {
            "within_lane_matched_methods": sorted(methods),
            "camera_bout_lane_comparable_to_session_observable": False,
            "camera_bout_lane_role": "oracle-boundary upper-bound diagnostic only",
            "cross_lane_before_after_language_allowed": False,
        },
        "claim_policy": {
            "confirmatory_claim_allowed": False,
            "state_of_the_art_claim_allowed": False,
            "clinical_claim_allowed": False,
            "zero_lookahead_streaming_claim_allowed": False,
            "camera_bout_lane_is_deployable": False,
            "alternate_width_selection_after_primary_failure_allowed": False,
            "camera_annotations_define_preprocessing_boundaries": not observable,
            "evaluation_activity_labels_are_model_features_or_fit_targets": False,
        },
    }


def _sole_observable_contract_fixture() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, NDArray[Any]]
]:
    participant = "sole:C001"
    session = f"{participant}:session-1"
    trial = f"{session}:session-recording"
    candidate_identifiers = {
        "participant_ids": np.asarray(["sole:C001"] * 4, dtype=np.str_),
        "session_ids": np.asarray([session] * 4, dtype=np.str_),
        "trial_ids": np.asarray([trial] * 4, dtype=np.str_),
        "window_ids": np.asarray(
            [
                f"{participant}/{session}/{trial}/run-0000-finite-000/window-{index:06d}"
                for index in range(4)
            ],
            dtype=np.str_,
        ),
    }
    indices = np.asarray([0, 2], dtype=np.int64)
    eligibility = np.asarray([True, False, True, False], dtype=np.bool_)
    probability = np.asarray(
        ((0.8, 0.1, 0.1), (0.2, 0.7, 0.1), (0.1, 0.2, 0.7), (0.6, 0.2, 0.2)),
        dtype=np.float64,
    )
    block_ids = _temporal_block_ids(
        type("WindowData", (), {"window_ids": candidate_identifiers["window_ids"]})()
    )
    per_seed = sole_harmony_temporal._temporal_methods(probability, block_ids)
    methods = tuple(per_seed)
    full_probabilities = {
        **{name: values.copy() for name, values in per_seed.items()},
        **{f"seed-11__{name}": values.copy() for name, values in per_seed.items()},
    }
    scored_probabilities = {name: values[indices] for name, values in full_probabilities.items()}
    pool = {
        "protocol_id": "external-har-observable-candidate-context-v1",
        "window_count": 4,
        "participant_window_counts": {"sole:C001": 4},
        "arrays": {name: _array_sha256(values) for name, values in candidate_identifiers.items()},
    }
    candidate_starts = np.arange(4, dtype=np.int64) * 128
    summary = {
        "dataset_id": "sole_harmony_v1",
        "window_count": 2,
        "participant_count": 1,
        "class_window_counts": {"mobility": 1, "sitting": 0, "standing": 1},
        "boundary_provenance": {
            "protocol_id": BOUNDARY_PROVENANCE_PROTOCOL,
            "boundary_mode": "session_observable",
            "repository_signal_grid_annotation_independent": True,
            "provider_upstream_annotation_conditioned": False,
            "zero_lookahead_streaming_valid": False,
            "annotation_application": "camera intervals projected after global signal-grid construction",
        },
        "observable_candidate_pool": pool,
        "preprocessing_audit": [
            {
                "protocol_id": PHYSICAL_GRID_PROTOCOL,
                "participant_id": participant,
                "session_id": session,
                "trial_id": trial,
                "run_index": 0,
                "resampled_samples": 512,
                "window_samples": 128,
                "candidate_start_samples": candidate_starts.tolist(),
                "candidate_window_count": 4,
                "candidate_grid_sha256": _array_sha256(candidate_starts),
                "within_declared_segment_transform_annotation_dependency": False,
                "segment_boundary_annotation_conditioned": False,
                "resampling_passes": 1,
            }
        ],
    }
    arrays: dict[str, NDArray[Any]] = {
        "labels": np.asarray([0, 2], dtype=np.int64),
        "observable_scoring_indices": indices,
        "observable_scoring_eligibility": eligibility,
    }
    arrays.update({name: values[indices] for name, values in candidate_identifiers.items()})
    arrays.update({f"observable_{name}": values for name, values in candidate_identifiers.items()})
    arrays.update({f"probability__{name}": values for name, values in scored_probabilities.items()})
    arrays.update(
        {f"observable_probability__{name}": values for name, values in full_probabilities.items()}
    )
    result = {
        "experiment_id": "sole-harmony-observable-session-temporal-v1",
        "dataset": summary,
        "primary_seed_averaged": {"seeds": [11], "methods": {name: {} for name in methods}},
        "prediction_contract": {
            "scored_window_count": 2,
            "observable_candidate_window_count": 4,
            "scoring_eligibility_policy": OBSERVABLE_SCORING_ELIGIBILITY_POLICY,
            "scoring_indices_sha256": canonical_json_sha256(indices.tolist()),
            "scoring_indices_array_sha256": _array_sha256(indices),
            "observable_scoring_eligibility_sha256": _array_sha256(eligibility),
            "observable_candidate_pool_audit_sha256": canonical_json_sha256(pool),
            "observable_identifier_array_sha256": {
                name: _array_sha256(values) for name, values in candidate_identifiers.items()
            },
            "scored_identifier_array_sha256": {
                name: _array_sha256(values[indices])
                for name, values in candidate_identifiers.items()
            },
            "observable_probability_array_sha256": {
                name: _array_sha256(values) for name, values in full_probabilities.items()
            },
            "scored_probability_array_sha256": {
                name: _array_sha256(values) for name, values in scored_probabilities.items()
            },
            "observable_probabilities_retained_for_every_seed_and_method": True,
            "scored_probabilities_retained_for_every_seed_and_method": True,
        },
        "fold_records": [
            {
                "temporal_state_input_window_count": 4,
                "temporal_scoring_slice_window_count": 2,
                "temporal_state_updated_on_every_observable_candidate": True,
                "temporal_labels_or_eligibility_used_before_state_update": False,
                "boundary_mode": "session_observable",
                "folds": [
                    {
                        "temporal_state_updated_on_every_evaluation_candidate": True,
                        "evaluation_scoring_eligibility_used_before_prediction": False,
                    }
                ],
            }
        ],
        **_sole_contract_fields("session_observable", methods),
    }
    return result, {"dataset": summary}, arrays


def _sole_oracle_contract_fixture() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, NDArray[Any]]
]:
    result, audit, original = _sole_observable_contract_fixture()
    participant = str(original["observable_participant_ids"][0])
    session = str(original["observable_session_ids"][0])
    trial = f"{session}:camera-bout-000001"
    identifiers = {
        "participant_ids": np.asarray([participant] * 4, dtype=np.str_),
        "session_ids": np.asarray([session] * 4, dtype=np.str_),
        "trial_ids": np.asarray([trial] * 4, dtype=np.str_),
        "window_ids": np.asarray(
            [
                f"{participant}/{session}/{trial}/run-0000-finite-000/window-{index:06d}"
                for index in range(4)
            ],
            dtype=np.str_,
        ),
    }
    indices = np.arange(4, dtype=np.int64)
    eligibility = np.ones(4, dtype=np.bool_)
    block_ids = _temporal_block_ids(
        type("WindowData", (), {"window_ids": identifiers["window_ids"]})()
    )
    base = original[f"observable_probability__seed-11__{BASE_METHOD}"].copy()
    temporal = sole_harmony_temporal._temporal_methods(base, block_ids)
    full_probabilities = {
        **{name: values.copy() for name, values in temporal.items()},
        **{f"seed-11__{name}": values.copy() for name, values in temporal.items()},
    }
    arrays: dict[str, NDArray[Any]] = {
        "labels": np.asarray([0, 1, 2, 0], dtype=np.int64),
        "observable_scoring_indices": indices,
        "observable_scoring_eligibility": eligibility,
    }
    arrays.update({name: values.copy() for name, values in identifiers.items()})
    arrays.update({f"observable_{name}": values.copy() for name, values in identifiers.items()})
    arrays.update(
        {f"probability__{name}": values.copy() for name, values in full_probabilities.items()}
    )
    arrays.update(
        {
            f"observable_probability__{name}": values.copy()
            for name, values in full_probabilities.items()
        }
    )
    candidate_starts = np.arange(4, dtype=np.int64) * 128
    summary = result["dataset"]
    summary.pop("observable_candidate_pool")
    summary.update(
        {
            "window_count": 4,
            "class_window_counts": {"mobility": 2, "sitting": 1, "standing": 1},
            "boundary_provenance": {
                "protocol_id": BOUNDARY_PROVENANCE_PROTOCOL,
                "boundary_mode": "camera_bout_oracle",
                "repository_signal_grid_annotation_independent": False,
                "provider_upstream_annotation_conditioned": False,
                "zero_lookahead_streaming_valid": False,
                "annotation_application": "camera interval defines each signal/reset boundary",
            },
            "preprocessing_audit": [
                {
                    "protocol_id": PHYSICAL_GRID_PROTOCOL,
                    "participant_id": participant,
                    "session_id": session,
                    "trial_id": trial,
                    "run_index": 0,
                    "finite_run_index": 0,
                    "resampled_samples": 512,
                    "window_samples": 128,
                    "candidate_start_samples": candidate_starts.tolist(),
                    "candidate_window_count": 4,
                    "candidate_grid_sha256": _array_sha256(candidate_starts),
                    "within_declared_segment_transform_annotation_dependency": False,
                    "segment_boundary_annotation_conditioned": True,
                    "resampling_passes": 1,
                }
            ],
        }
    )
    result["experiment_id"] = "sole-harmony-camera-bout-oracle-temporal-v2"
    result.update(_sole_contract_fields("camera_bout_oracle", tuple(temporal)))
    contract = result["prediction_contract"]
    contract.update(
        {
            "scored_window_count": 4,
            "observable_candidate_window_count": 4,
            "scoring_indices_sha256": canonical_json_sha256(indices.tolist()),
            "scoring_indices_array_sha256": _array_sha256(indices),
            "observable_scoring_eligibility_sha256": _array_sha256(eligibility),
            "observable_candidate_pool_audit_sha256": None,
            "observable_identifier_array_sha256": {
                name: _array_sha256(values) for name, values in identifiers.items()
            },
            "scored_identifier_array_sha256": {
                name: _array_sha256(values) for name, values in identifiers.items()
            },
            "observable_probability_array_sha256": {
                name: _array_sha256(values) for name, values in full_probabilities.items()
            },
            "scored_probability_array_sha256": {
                name: _array_sha256(values) for name, values in full_probabilities.items()
            },
        }
    )
    result["fold_records"][0].update(
        {
            "temporal_state_input_window_count": 4,
            "temporal_scoring_slice_window_count": 4,
            "boundary_mode": "camera_bout_oracle",
        }
    )
    return result, audit, arrays


@pytest.mark.parametrize(
    "mutation",
    ("delete_full", "reorder_full_ids", "change_full_only_row", "change_index", "change_slice"),
)
def test_sole_observable_contract_rejects_full_stream_mutations(mutation: str) -> None:
    result, audit, arrays = _sole_observable_contract_fixture()
    assert not _sole_observable_prediction_errors(result, audit, _MemoryPredictionArchive(arrays))
    changed = {name: values.copy() for name, values in arrays.items()}
    if mutation == "delete_full":
        del changed[f"observable_probability__seed-11__{PRIMARY_METHOD}"]
    elif mutation == "reorder_full_ids":
        changed["observable_window_ids"][[1, 3]] = changed["observable_window_ids"][[3, 1]]
    elif mutation == "change_full_only_row":
        changed[f"observable_probability__seed-11__{PRIMARY_METHOD}"][1] = (0.3, 0.6, 0.1)
    elif mutation == "change_index":
        changed["observable_scoring_indices"] = np.asarray([0, 3], dtype=np.int64)
    else:
        changed[f"probability__seed-11__{PRIMARY_METHOD}"][1] = (0.2, 0.2, 0.6)
    assert _sole_observable_prediction_errors(result, audit, _MemoryPredictionArchive(changed))


def test_sole_observable_contract_rejects_coherently_rehashed_fake_temporal_output() -> None:
    result, audit, arrays = _sole_observable_contract_fixture()
    changed = {name: values.copy() for name, values in arrays.items()}
    indices = changed["observable_scoring_indices"].astype(np.int64)
    fake = changed[f"observable_probability__seed-11__{BASE_METHOD}"].copy()
    changed[f"observable_probability__seed-11__{PRIMARY_METHOD}"] = fake
    changed[f"observable_probability__{PRIMARY_METHOD}"] = fake.copy()
    changed[f"probability__seed-11__{PRIMARY_METHOD}"] = fake[indices]
    changed[f"probability__{PRIMARY_METHOD}"] = fake[indices].copy()
    result["prediction_contract"]["observable_probability_array_sha256"] = {
        name.removeprefix("observable_probability__"): _array_sha256(values)
        for name, values in changed.items()
        if name.startswith("observable_probability__")
    }
    result["prediction_contract"]["scored_probability_array_sha256"] = {
        name.removeprefix("probability__"): _array_sha256(values)
        for name, values in changed.items()
        if name.startswith("probability__")
    }

    errors = _sole_observable_prediction_errors(result, audit, _MemoryPredictionArchive(changed))
    assert any("not independently derived" in error for error in errors)


def test_sole_observable_contract_rejects_coherently_rehashed_fake_reset_boundary() -> None:
    result, audit, arrays = _sole_observable_contract_fixture()
    changed = {name: values.copy() for name, values in arrays.items()}
    participant = str(changed["observable_participant_ids"][0])
    session = str(changed["observable_session_ids"][0])
    trial = str(changed["observable_trial_ids"][0])
    changed_ids = np.asarray(
        [
            f"{participant}/{session}/{trial}/run-{run:04d}-finite-000/window-{index:06d}"
            for run, index in ((0, 0), (0, 1), (1, 0), (1, 1))
        ],
        dtype=np.str_,
    )
    changed["observable_window_ids"] = changed_ids
    indices = changed["observable_scoring_indices"].astype(np.int64)
    changed["window_ids"] = changed_ids[indices]
    block_ids = _temporal_block_ids(type("WindowData", (), {"window_ids": changed_ids})())
    base = changed[f"observable_probability__seed-11__{BASE_METHOD}"]
    temporal = sole_harmony_temporal._temporal_methods(base, block_ids)
    for method, values in temporal.items():
        changed[f"observable_probability__seed-11__{method}"] = values.copy()
        changed[f"observable_probability__{method}"] = values.copy()
        changed[f"probability__seed-11__{method}"] = values[indices]
        changed[f"probability__{method}"] = values[indices].copy()

    pool = result["dataset"]["observable_candidate_pool"]
    pool["arrays"]["window_ids"] = _array_sha256(changed_ids)
    contract = result["prediction_contract"]
    contract["observable_candidate_pool_audit_sha256"] = canonical_json_sha256(pool)
    contract["observable_identifier_array_sha256"]["window_ids"] = _array_sha256(changed_ids)
    contract["scored_identifier_array_sha256"]["window_ids"] = _array_sha256(changed_ids[indices])
    contract["observable_probability_array_sha256"] = {
        name.removeprefix("observable_probability__"): _array_sha256(values)
        for name, values in changed.items()
        if name.startswith("observable_probability__")
    }
    contract["scored_probability_array_sha256"] = {
        name.removeprefix("probability__"): _array_sha256(values)
        for name, values in changed.items()
        if name.startswith("probability__")
    }

    errors = _sole_observable_prediction_errors(result, audit, _MemoryPredictionArchive(changed))
    assert any("audited physical segments" in error for error in errors)


def test_sole_oracle_contract_rejects_coherently_rehashed_fake_reset_boundary() -> None:
    result, audit, arrays = _sole_oracle_contract_fixture()
    assert not _sole_observable_prediction_errors(result, audit, _MemoryPredictionArchive(arrays))
    changed = {name: values.copy() for name, values in arrays.items()}
    participant = str(changed["observable_participant_ids"][0])
    session = str(changed["observable_session_ids"][0])
    trial = str(changed["observable_trial_ids"][0])
    changed_ids = np.asarray(
        [
            f"{participant}/{session}/{trial}/run-{run:04d}-finite-000/window-{index:06d}"
            for run, index in ((0, 0), (0, 1), (1, 0), (1, 1))
        ],
        dtype=np.str_,
    )
    changed["observable_window_ids"] = changed_ids
    changed["window_ids"] = changed_ids.copy()
    block_ids = _temporal_block_ids(type("WindowData", (), {"window_ids": changed_ids})())
    base = changed[f"observable_probability__seed-11__{BASE_METHOD}"]
    temporal = sole_harmony_temporal._temporal_methods(base, block_ids)
    for method, values in temporal.items():
        changed[f"observable_probability__seed-11__{method}"] = values.copy()
        changed[f"observable_probability__{method}"] = values.copy()
        changed[f"probability__seed-11__{method}"] = values.copy()
        changed[f"probability__{method}"] = values.copy()
    contract = result["prediction_contract"]
    contract["observable_identifier_array_sha256"]["window_ids"] = _array_sha256(changed_ids)
    contract["scored_identifier_array_sha256"]["window_ids"] = _array_sha256(changed_ids)
    contract["observable_probability_array_sha256"] = {
        name.removeprefix("observable_probability__"): _array_sha256(values)
        for name, values in changed.items()
        if name.startswith("observable_probability__")
    }
    contract["scored_probability_array_sha256"] = {
        name.removeprefix("probability__"): _array_sha256(values)
        for name, values in changed.items()
        if name.startswith("probability__")
    }

    errors = _sole_observable_prediction_errors(result, audit, _MemoryPredictionArchive(changed))
    assert any("audited physical segments" in error for error in errors)


@pytest.mark.parametrize("mutation", ("experiment", "fold_boundary", "nested_semantics"))
def test_sole_oracle_contract_binds_experiment_and_fold_boundary(mutation: str) -> None:
    result, audit, arrays = _sole_oracle_contract_fixture()
    if mutation == "experiment":
        result["experiment_id"] = "sole-harmony-observable-session-temporal-v1"
    elif mutation == "fold_boundary":
        result["fold_records"][0]["boundary_mode"] = "session_observable"
    else:
        result["fold_records"][0]["folds"][0][
            "evaluation_scoring_eligibility_used_before_prediction"
        ] = True
    errors = _sole_observable_prediction_errors(result, audit, _MemoryPredictionArchive(arrays))
    assert errors


@pytest.mark.parametrize(
    "mutation", ("provider_boundary", "temporal_contract", "comparison_contract", "claim_policy")
)
def test_sole_contract_rejects_coherently_rehashed_semantic_claims(mutation: str) -> None:
    result, audit, arrays = _sole_observable_contract_fixture()
    if mutation == "provider_boundary":
        result["dataset"]["boundary_provenance"]["provider_upstream_annotation_conditioned"] = True
    elif mutation == "temporal_contract":
        result["temporal_contract"]["future_window_access"] = True
    elif mutation == "comparison_contract":
        result["comparison_contract"]["cross_lane_before_after_language_allowed"] = True
    else:
        result["claim_policy"]["zero_lookahead_streaming_claim_allowed"] = True
    errors = _sole_observable_prediction_errors(result, audit, _MemoryPredictionArchive(arrays))
    assert errors


def test_causal_controls_have_fixed_past_only_and_reset_semantics() -> None:
    blocks = np.asarray(["a", "a", "a", "b", "b"], dtype=np.str_)
    probability = np.asarray(
        (
            (0.9, 0.1, 0.0),
            (0.1, 0.9, 0.0),
            (0.1, 0.8, 0.1),
            (0.0, 0.1, 0.9),
            (0.0, 0.8, 0.2),
        ),
        dtype=np.float64,
    )
    changed_future = probability.copy()
    changed_future[2] = (0.0, 0.0, 1.0)
    original = _causal_probability(probability, blocks, width=3)
    changed = _causal_probability(changed_future, blocks, width=3)
    np.testing.assert_allclose(original[:2], changed[:2])
    np.testing.assert_allclose(original[3], probability[3])

    majority = _causal_majority_probability(probability, blocks, width=3)
    assert majority.argmax(axis=1).tolist() == [0, 1, 1, 2, 1]
    hysteresis = _causal_hysteresis_probability(probability, blocks, confirmation_windows=2)
    assert hysteresis.argmax(axis=1).tolist() == [0, 0, 1, 2, 2]


def test_shuffled_time_control_is_reproducible_block_local_and_restores_identity() -> None:
    blocks = np.asarray(["a", "a", "a", "b", "b", "b"], dtype=np.str_)
    probability = np.asarray(
        (
            (0.8, 0.1, 0.1),
            (0.1, 0.8, 0.1),
            (0.2, 0.2, 0.6),
            (0.1, 0.1, 0.8),
            (0.1, 0.7, 0.2),
            (0.6, 0.2, 0.2),
        ),
        dtype=np.float64,
    )
    width_one = sole_harmony_temporal._shuffled_time_probability(
        probability, blocks, width=1, seed=41
    )
    np.testing.assert_allclose(width_one, probability)

    first = sole_harmony_temporal._shuffled_time_probability(probability, blocks, width=3, seed=41)
    repeated = sole_harmony_temporal._shuffled_time_probability(
        probability, blocks, width=3, seed=41
    )
    np.testing.assert_allclose(first, repeated)
    changed_first_block = probability.copy()
    changed_first_block[:3] = changed_first_block[:3, ::-1]
    changed = sole_harmony_temporal._shuffled_time_probability(
        changed_first_block, blocks, width=3, seed=41
    )
    np.testing.assert_allclose(first[3:], changed[3:])


def test_session_temporal_inference_contract_is_not_camera_oracle() -> None:
    data = _sole_session_data()
    result = {
        "dataset": data.summary(),
        "temporal_contract": {"protocol_id": "sole-harmony-observable-session-temporal-v1"},
        "primary_seed_averaged": {"methods": {BASE_METHOD: {}, PRIMARY_METHOD: {}}},
    }
    contracts = method_inference_contracts(result)
    assert annotation_selected_context_methods(result) == []
    assert contracts[BASE_METHOD]["inference_unit"] == "independent_fixed_window"
    assert contracts[PRIMARY_METHOD]["inference_unit"] == ("causal_observable_physical_run_history")
    assert contracts[PRIMARY_METHOD]["input_channel_count"] == 6
    assert contracts[PRIMARY_METHOD]["scientific_role"] == (
        "development_session_observable_offline_resampling"
    )


def test_session_lane_rejects_camera_conditioned_boundary() -> None:
    data = _sole_session_data()
    boundary = dict(data.boundary_provenance or {})
    boundary.update(
        {
            "boundary_mode": "camera_bout_oracle",
            "repository_signal_grid_annotation_independent": False,
        }
    )
    camera = replace(data, boundary_provenance=boundary)
    with pytest.raises(PermissionError, match="annotation-conditioned"):
        sole_harmony_temporal._validate_lane(camera, boundary_mode="camera_bout_oracle")


def test_temporal_block_ids_reject_nonphysical_oracle_free_text() -> None:
    data = _sole_session_data()
    invalid = replace(data, window_ids=np.asarray(["camera-label"] * data.labels.size))
    with pytest.raises(ValueError, match="physical-run provenance"):
        _temporal_block_ids(invalid)


def test_publication_campaign_runs_observable_then_separate_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads: list[str] = []
    runs: list[str] = []
    gates: list[tuple[str, bool]] = []

    monkeypatch.setattr(
        publication_campaign,
        "_git_state",
        lambda _root: {"worktree_dirty": False, "commit": "synthetic", "status_entries": []},
    )
    monkeypatch.setattr(publication_campaign, "_source_input_manifest", lambda _root: {"files": {}})
    monkeypatch.setattr(publication_campaign, "_source_manifest_commit_errors", lambda *_args: [])
    monkeypatch.setattr(publication_campaign, "sha256_file", lambda _path: "a" * 64)

    def load(**kwargs: Any) -> str:
        mode = str(kwargs["boundary_mode"])
        loads.append(mode)
        return mode

    def run_temporal(**kwargs: Any) -> None:
        directory = Path(kwargs["output_directory"])
        directory.mkdir(parents=True)
        result = {"status": "SYNTHETIC_RESULT"}
        result["result_payload_sha256_before_serialization"] = canonical_json_sha256(result)
        (directory / "result.json").write_text(
            json.dumps(result, sort_keys=True) + "\n", encoding="utf-8"
        )
        runs.append(f"{kwargs['data']}:{directory.name}")

    def accept(directory: Path, _root: Path, *, diagnostic: bool = False) -> None:
        (directory / "validation.json").write_text(
            json.dumps(
                {
                    "status": "DIAGNOSTIC",
                    "integrity_passed": True,
                    "publication_evidence_ready": False,
                    "publication_evidence_ready_for_unqualified_methods": False,
                    "diagnostic_contract_passed": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        gates.append((directory.name, diagnostic))

    monkeypatch.setattr(publication_campaign, "load_sole_harmony", load)
    monkeypatch.setattr(sole_harmony_temporal, "run_and_write", run_temporal)
    monkeypatch.setattr(publication_campaign, "_accept", accept)
    completed = publication_campaign.run_campaign(
        dataset="sole-harmony",
        output=tmp_path / "campaign",
        repository_root=tmp_path,
        n_jobs=1,
    )
    assert loads == ["session_observable", "camera_bout_oracle"]
    assert runs == [
        "session_observable:session_observable",
        "camera_bout_oracle:camera_bout_oracle",
    ]
    assert gates == [("session_observable", True), ("camera_bout_oracle", True)]
    assert completed["status"] == "DIAGNOSTIC_COMPLETE"
    assert completed["artifact_role"] == "non_run_create_only_orchestration_index"
    assert completed["root_run_validation_applicable"] is False
    assert all(item["binding_complete"] for item in completed["child_artifact_bindings"])
