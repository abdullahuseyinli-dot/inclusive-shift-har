from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inclusive_shift_har.data.external_har import (
    ExternalHARWindows,
    ObservableWindowPool,
    SourceReceipt,
    _array_sha256,
)
from inclusive_shift_har.data.participant_partitions import build_participant_partition_plan
from inclusive_shift_har.evaluation.external_statistics import (
    ParticipantMetricInputs,
    seed_evidence,
)
from inclusive_shift_har.experiments import hierarchical_posture as runner
from inclusive_shift_har.experiments.external_evidence_validate import (
    _hierarchy_artifact_errors,
    _hierarchy_baseline_reconstruction_errors,
)
from inclusive_shift_har.experiments.hierarchical_posture import (
    _baseline_probability_check,
    _matched_reference_input_errors,
    _reference_run_location,
    posture_advancement_gate,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.classical import ClassicalConfig, _build_estimator
from inclusive_shift_har.models.hierarchical_posture import (
    VARIANTS,
    fit_posture_forest,
    participant_class_weights,
)
from inclusive_shift_har.preprocessing.features import extract_engineered_features


def test_reference_run_location_is_portable_when_governed_by_repository(tmp_path: Path) -> None:
    governed = tmp_path / "results" / "reference"
    governed.mkdir(parents=True)
    assert _reference_run_location(governed, tmp_path) == {
        "path_kind": "repository_relative",
        "path": "results/reference",
    }


def _savez_compressed(path: Path, **arrays: Any) -> None:
    """Write named arrays without exposing NumPy's reserved kwargs to mypy."""
    np.savez_compressed(path, **arrays)


def test_participant_class_mass_is_equal_even_with_missing_classes() -> None:
    labels = np.array([0, 0, 1, 2, 0, 0, 0, 1], dtype=np.int64)
    participants = np.array(["a"] * 4 + ["b"] * 4)
    weights = participant_class_weights(labels, participants)
    assert weights.mean() == pytest.approx(1)
    assert weights[:4].sum() == pytest.approx(weights[4:].sum())
    for person in ("a", "b"):
        masses = [
            weights[(participants == person) & (labels == label)].sum()
            for label in np.unique(labels[participants == person])
        ]
        np.testing.assert_allclose(masses, masses[0])


@pytest.mark.parametrize("variant", VARIANTS)
def test_fixed_tree_budget_probability_order_and_label_free_inference(variant: str) -> None:
    features = np.random.default_rng(11).normal(size=(60, 8))
    labels = np.tile(np.arange(3), 20)
    participants = np.repeat(["a", "b", "c", "d"], 15)
    model = fit_posture_forest(features, labels, participants, variant=variant, seed=11)
    prediction = model.predict(features[:6])
    assert prediction.shape == (6, 3)
    np.testing.assert_allclose(prediction.sum(axis=1), 1)
    assert model.size_summary()["tree_count"] == 500
    assert set(inspect.signature(model.predict).parameters) == {"features"}
    np.testing.assert_array_equal(prediction, model.predict(features[:6].copy()))
    if variant == "RandomForest-6ch":
        base = _build_estimator(ClassicalConfig("random_forest", num_classes=3, seed=11))
        base.fit(features, labels)
        np.testing.assert_array_equal(prediction, base.predict_proba(features[:6]))
    if variant == "PB-HPF-shuffled-posture":
        assert model.shuffled_training_labels > 0


@pytest.mark.parametrize("channels", [6, 9])
def test_cached_window_features_equal_partition_local_features(channels: int) -> None:
    signals = np.random.default_rng(12).normal(size=(20, 128, channels)).astype(np.float32)
    names = tuple(f"channel_{index}" for index in range(channels))
    selected = np.arange(20) % 3 != 0
    full = extract_engineered_features(signals, channel_names=names)
    local = extract_engineered_features(signals[selected], channel_names=names)
    np.testing.assert_array_equal(full.values[selected], local.values)
    assert full.names == local.names


@pytest.mark.parametrize("candidate_correct", [False, True])
def test_advancement_gate_is_not_replaced_by_best_ablation(candidate_correct: bool) -> None:
    labels = np.tile(np.arange(3), 12)
    inputs = ParticipantMetricInputs(
        "fog_star_v3",
        ("mobility", "sitting", "standing"),
        labels,
        np.repeat([f"p{i:02d}" for i in range(12)], 3),
    )
    correct = np.eye(3)[labels] * 0.9 + 0.1 / 3
    wrong = np.eye(3)[(labels + 1) % 3] * 0.9 + 0.1 / 3
    probabilities = {
        seed: {
            name: correct
            if name == "PB-RF-D9" or (candidate_correct and name == "PB-HPF")
            else wrong
            for name in VARIANTS
        }
        for seed in (11, 23, 47)
    }
    statistics, _ = seed_evidence(inputs, probabilities)
    gate = posture_advancement_gate(statistics)
    assert gate["all_advancement_gates_passed"] is candidate_correct
    assert gate["comparisons"]["PB-RF-D9"]["input_tuple_matched"] is False
    if not candidate_correct:
        assert gate["decision"].startswith("STOP_CANDIDATE_EXPANSION")


def test_baseline_binding_rejects_changed_receipt_or_calibration_probabilities() -> None:
    labels = np.tile(np.arange(3), 12)
    people = np.repeat([f"p{i:02d}" for i in range(12)], 3)
    values = np.random.default_rng(92).normal(size=(36, 128, 9)).astype(np.float32)
    data = ExternalHARWindows(
        dataset_id="synthetic_baseline_binding",
        channel_lane="derived-gravity-9ch",
        sampling_rate_hz=50,
        signals=values[:, :, :6],
        gravity=values[:, :, 6:],
        labels=labels,
        participant_ids=people,
        session_ids=people,
        trial_ids=people,
        window_ids=np.array([f"w{i}" for i in range(36)]),
        receipts=(
            SourceReceipt("synthetic_baseline_binding", "synthetic://source", None, 1, 1, "0" * 64),
        ),
    )
    summary = data.summary()
    receipt = data.receipts[0].to_dict()
    result = {"dataset": summary}
    audit = {"dataset": summary, "source_receipts": [receipt]}
    assert _matched_reference_input_errors(data, result, audit) == []
    changed_receipt = {**receipt, "computed_sha256": "1" * 64}
    assert "source receipt identity/version/hash differs" in _matched_reference_input_errors(
        data, result, {**audit, "source_receipts": [changed_receipt]}
    )

    reference = np.array([[0.60, 0.25, 0.15], [0.10, 0.20, 0.70]])
    same_decisions_but_changed_calibration = reference.copy()
    same_decisions_but_changed_calibration[0] = [0.55, 0.30, 0.15]
    check = _baseline_probability_check(reference, same_decisions_but_changed_calibration)
    assert check["class_decisions_exact"] is True
    assert check["probabilities_within_strict_tolerance"] is False


def test_hierarchy_main_preserves_a_pre_writer_acquisition_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "failed-hierarchy"
    events: list[str] = []

    def resolve(**_kwargs: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        events.append("launch")
        return ({"commit": "0" * 40}, {"files": {}}, {"record_sha256": "a" * 64})

    def acquire() -> ExternalHARWindows:
        events.append("dataset_acquisition")
        raise RuntimeError("synthetic provider failure")

    def preserve(**kwargs: Any) -> dict[str, Any]:
        events.append("failure")
        destination = Path(kwargs["output_directory"])
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "failure.json").write_text("{}\n", encoding="utf-8")
        return {}

    monkeypatch.setattr(runner, "_resolve_publication_launch_context", resolve)
    monkeypatch.setattr(runner, "_git_state", lambda _root: {})
    monkeypatch.setattr(runner, "_source_input_manifest", lambda _root: {})
    monkeypatch.setattr(runner, "load_fog_star", acquire)
    monkeypatch.setattr(runner, "_write_launch_failure_envelope", preserve)
    with pytest.raises(RuntimeError, match="synthetic provider failure"):
        runner.main(
            [
                "--repository-root",
                str(tmp_path),
                "--output",
                str(output),
                "--reference-run",
                str(tmp_path / "reference"),
            ]
        )
    assert events == ["launch", "dataset_acquisition", "failure"]
    assert (output / "failure.json").is_file()


class SyntheticForest:
    """Training seam only; runner serialization and statistics stay real."""

    def __init__(self, participants: np.ndarray[Any, Any]) -> None:
        self.training_participants = tuple(np.unique(participants).tolist())

    def predict(self, features: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.tile([0.6, 0.2, 0.2], (len(features), 1))

    def size_summary(self) -> dict[str, Any]:
        return {"synthetic_training_seam": True}

    def mechanism(self, _features: np.ndarray[Any, Any]) -> dict[str, Any]:
        return {"synthetic_training_seam": True}


def test_posture_suite_keeps_every_seed_fold_prediction_and_independent_partitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = np.tile(np.arange(3), 12)
    people = np.repeat([f"p{i:02d}" for i in range(12)], 3)
    values = np.random.default_rng(93).normal(size=(36, 128, 9)).astype(np.float32)
    sessions = people.copy()
    trials = people.copy()
    windows = np.array([f"w{i}" for i in range(36)])
    pool = ObservableWindowPool(
        values[:, :, :6], values[:, :, 6:], people, sessions, trials, windows
    )
    data = ExternalHARWindows(
        dataset_id="synthetic_posture_suite",
        channel_lane="derived-gravity-9ch",
        sampling_rate_hz=50,
        signals=values[:, :, :6],
        gravity=values[:, :, 6:],
        labels=labels,
        participant_ids=people,
        session_ids=sessions,
        trial_ids=trials,
        window_ids=windows,
        receipts=(
            SourceReceipt(
                "synthetic_posture_suite", "synthetic://no-source-data", None, 1, 1, "0" * 64
            ),
        ),
        participant_partition_plan=build_participant_partition_plan(
            "synthetic_posture_suite",
            [*people.tolist(), "p12"],
            roster_basis="synthetic participant inventory before windowing",
        ),
        observable_candidates=pool,
    )
    monkeypatch.setattr(
        runner,
        "fit_posture_forest",
        lambda _features, _labels, participants, **_kwargs: SyntheticForest(participants),
    )
    result, probabilities = runner.evaluate_posture_forests(data, output=tmp_path)
    assert result["seeds"] == [11, 23, 47] and len(result["fold_records"]) == 90
    assert result["advancement_gate"]["all_advancement_gates_passed"] is False
    assert len(list(tmp_path.glob("*__predictions.npz"))) == 15
    for record in result["fold_records"]:
        assert not set(record["training_participants"]) & set(record["evaluation_participants"])
        assert set(record["training_participants"]) | set(record["evaluation_participants"]) == {
            *np.unique(people).tolist(),
            "p12",
        }
        assert "p12" not in record["training_participants_with_supervision"]
        assert "p12" not in record["evaluation_participants_with_scoring"]
    for seed in (11, 23, 47):
        for name in VARIANTS:
            assert probabilities[f"seed-{seed}__{name}"].shape == (36, 3)

    indices = np.arange(36, dtype=np.int64)
    eligibility = np.ones(36, dtype=np.bool_)
    prediction_path = tmp_path / "predictions.npz"
    _savez_compressed(
        prediction_path,
        labels=data.labels,
        participant_ids=data.participant_ids,
        session_ids=data.session_ids,
        trial_ids=data.trial_ids,
        window_ids=data.window_ids,
        candidate_participant_ids=pool.participant_ids,
        candidate_window_ids=pool.window_ids,
        candidate_scoring_indices=indices,
        candidate_scoring_eligibility=eligibility,
        **{f"probability__{name}": value for name, value in probabilities.items()},
    )
    result["prediction_artifact"] = {
        "path": prediction_path.name,
        "sha256": sha256_file(prediction_path),
    }
    result["candidate_prediction_contract"] = {
        "candidate_window_count": 36,
        "scored_window_count": 36,
        "scoring_indices_sha256": canonical_json_sha256(indices.tolist()),
        "scoring_indices_array_sha256": _array_sha256(indices),
        "scoring_eligibility_array_sha256": _array_sha256(eligibility),
        "candidate_identifier_hashes": {
            "participant_ids": _array_sha256(pool.participant_ids),
            "window_ids": _array_sha256(pool.window_ids),
        },
    }
    reference_witness = tmp_path / "baseline_reference_predictions.npz"
    reference_probabilities = {
        seed: probabilities[f"seed-{seed}__RandomForest-6ch"] for seed in (11, 23, 47)
    }
    _savez_compressed(
        reference_witness,
        labels=data.labels,
        participant_ids=data.participant_ids,
        session_ids=data.session_ids,
        trial_ids=data.trial_ids,
        window_ids=data.window_ids,
        **{
            f"probability__seed-{seed}__RandomForest-6ch": values
            for seed, values in reference_probabilities.items()
        },
    )
    result["baseline_reconstruction"] = {
        "reference_run": {"path_kind": "repository_relative", "path": "reference"},
        "reference_result_sha256": "a" * 64,
        "reference_predictions_sha256": "b" * 64,
        "retained_reference_witness": {
            "path": reference_witness.name,
            "sha256": sha256_file(reference_witness),
            "size_bytes": reference_witness.stat().st_size,
        },
        "per_seed": {
            str(seed): _baseline_probability_check(values, values.copy())
            for seed, values in reference_probabilities.items()
        },
        "scientific_input_tuple_exact": True,
        "gate_passed": True,
    }
    audit = {"dataset": data.summary()}
    assert not _hierarchy_artifact_errors(result, audit, tmp_path)

    for invalid_location in (
        str((tmp_path / "reference").resolve()),
        {"path_kind": "repository_relative", "path": "../reference"},
        {"path_kind": "external_absolute", "path": "relative/reference"},
        {"path_kind": "repository_relative", "path": "reference", "extra": True},
    ):
        malformed = deepcopy(result)
        malformed["baseline_reconstruction"]["reference_run"] = invalid_location
        assert _hierarchy_baseline_reconstruction_errors(malformed)

    checkpoint = tmp_path / str(result["fold_records"][0]["checkpoint"]["path"])
    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint.unlink()
    assert _hierarchy_artifact_errors(result, audit, tmp_path)
    checkpoint.write_bytes(checkpoint_bytes)

    swapped = deepcopy(result)
    swapped["fold_records"][0]["checkpoint"]["path"] = swapped["fold_records"][1]["checkpoint"][
        "path"
    ]
    assert _hierarchy_artifact_errors(swapped, audit, tmp_path)
    rehashed = deepcopy(result)
    rehashed["fold_records"][0]["checkpoint"]["sha256"] = "0" * 64
    assert _hierarchy_artifact_errors(rehashed, audit, tmp_path)

    with np.load(prediction_path, allow_pickle=False) as archive:
        main_archive = {name: np.asarray(archive[name]).copy() for name in archive.files}
    main_name = "probability__seed-11__RandomForest-6ch"
    main_archive[main_name][0] = [0.55, 0.25, 0.20]
    _savez_compressed(prediction_path, **main_archive)
    result["prediction_artifact"]["sha256"] = sha256_file(prediction_path)
    assert any(
        "fold probability slice differs" in error or "baseline witness differs" in error
        for error in _hierarchy_artifact_errors(result, audit, tmp_path)
    )
    main_archive[main_name][0] = probabilities["seed-11__RandomForest-6ch"][0]
    _savez_compressed(prediction_path, **main_archive)
    result["prediction_artifact"]["sha256"] = sha256_file(prediction_path)

    with np.load(reference_witness, allow_pickle=False) as archive:
        witness_archive = {name: np.asarray(archive[name]).copy() for name in archive.files}
    witness_archive[main_name][0] = [0.55, 0.25, 0.20]
    _savez_compressed(reference_witness, **witness_archive)
    witness_descriptor = result["baseline_reconstruction"]["retained_reference_witness"]
    witness_descriptor["sha256"] = sha256_file(reference_witness)
    witness_descriptor["size_bytes"] = reference_witness.stat().st_size
    assert any(
        "baseline witness differs" in error
        for error in _hierarchy_artifact_errors(result, audit, tmp_path)
    )
    witness_archive[main_name][0] = reference_probabilities[11][0]
    _savez_compressed(reference_witness, **witness_archive)
    witness_descriptor["sha256"] = sha256_file(reference_witness)
    witness_descriptor["size_bytes"] = reference_witness.stat().st_size

    first_artifact = result["fold_artifacts"][0]
    archive_path = tmp_path / first_artifact["prediction_archive"]["path"]
    with np.load(archive_path, allow_pickle=False) as archive:
        changed_archive = {name: np.asarray(archive[name]).copy() for name in archive.files}
    changed_archive["candidate_window_ids"][0] = "forged-window"
    _savez_compressed(archive_path, **changed_archive)
    first_artifact["prediction_archive"]["sha256"] = sha256_file(archive_path)
    first_artifact["prediction_archive"]["size_bytes"] = archive_path.stat().st_size
    marker_path = tmp_path / first_artifact["completion_marker"]["path"]
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["prediction_archive_sha256"] = sha256_file(archive_path)
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    first_artifact["completion_marker"]["sha256"] = sha256_file(marker_path)
    first_artifact["completion_marker"]["size_bytes"] = marker_path.stat().st_size
    assert any(
        "identity/scoring slice differs" in error
        for error in _hierarchy_artifact_errors(result, audit, tmp_path)
    )


def test_posture_suite_predicts_complete_observable_pool_before_annotation_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_count = 48
    people = np.repeat([f"p{i:02d}" for i in range(12)], 4)
    signals = np.random.default_rng(94).normal(size=(candidate_count, 128, 6)).astype(np.float32)
    gravity = np.zeros((candidate_count, 128, 3), dtype=np.float32)
    gravity[:, :, 2] = 9.81
    sessions = np.char.add(people, "/session")
    trials = np.char.add(people, "/trial")
    windows = np.array([f"candidate-{index:03d}" for index in range(candidate_count)])
    pool = ObservableWindowPool(signals, gravity, people, sessions, trials, windows)
    plan = build_participant_partition_plan(
        "synthetic_posture_pool",
        people,
        roster_basis="synthetic pre-annotation candidate roster",
    )

    def dataset(eligible: np.ndarray[Any, Any]) -> ExternalHARWindows:
        score = np.flatnonzero(eligible)
        # Labels are post hoc scoring metadata. Their values deliberately depend only
        # on scored order, while the complete inference population remains fixed.
        labels = np.arange(score.size, dtype=np.int64) % 3
        return ExternalHARWindows(
            dataset_id="synthetic_posture_pool",
            channel_lane="derived-gravity-9ch",
            sampling_rate_hz=50,
            signals=signals[score],
            gravity=gravity[score],
            labels=labels,
            participant_ids=people[score],
            session_ids=sessions[score],
            trial_ids=trials[score],
            window_ids=windows[score],
            receipts=(
                SourceReceipt("synthetic_posture_pool", "synthetic://pool", None, 1, 1, "0" * 64),
            ),
            observable_candidates=pool,
            participant_partition_plan=plan,
        )

    first_eligible = np.arange(candidate_count) % 4 != 3
    second_eligible = first_eligible.copy()
    second_eligible[np.arange(candidate_count) % 8 == 1] = False
    prediction_rows: list[int] = []
    fit_rows: list[int] = []

    original_predict = SyntheticForest.predict

    def predict(self: SyntheticForest, features: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        prediction_rows.append(len(features))
        return original_predict(self, features)

    def fit(
        features: np.ndarray[Any, Any],
        _labels: np.ndarray[Any, Any],
        participants: np.ndarray[Any, Any],
        **_kwargs: Any,
    ) -> SyntheticForest:
        fit_rows.append(len(features))
        assert len(features) == len(participants)
        return SyntheticForest(participants)

    monkeypatch.setattr(runner, "fit_posture_forest", fit)
    monkeypatch.setattr(SyntheticForest, "predict", predict)
    first, first_probabilities = runner.evaluate_posture_forests(
        dataset(first_eligible), output=tmp_path / "first", seeds=(11,)
    )
    first_prediction_rows = prediction_rows.copy()
    first_fit_rows = fit_rows.copy()
    prediction_rows.clear()
    fit_rows.clear()
    second, second_probabilities = runner.evaluate_posture_forests(
        dataset(second_eligible), output=tmp_path / "second", seeds=(11,)
    )

    assert first["observable_context_protocol"] == "external-har-observable-context-v1"
    assert second["observable_context_protocol"] == "external-har-observable-context-v1"
    assert sum(first_prediction_rows) == candidate_count * len(VARIANTS)
    assert sum(prediction_rows) == candidate_count * len(VARIANTS)
    assert sum(first_fit_rows) == int(first_eligible.sum()) * 4 * len(VARIANTS)
    assert sum(fit_rows) == int(second_eligible.sum()) * 4 * len(VARIANTS)
    first_candidates = [
        (record["method"], record["outer_fold"], record["evaluation_candidate_window_count"])
        for record in first["fold_records"]
    ]
    second_candidates = [
        (record["method"], record["outer_fold"], record["evaluation_candidate_window_count"])
        for record in second["fold_records"]
    ]
    assert first_candidates == second_candidates
    assert (
        sum(
            record["evaluation_candidate_window_count"]
            for record in first["fold_records"]
            if record["method"] == VARIANTS[0]
        )
        == candidate_count
    )
    assert all(
        value.shape == (int(first_eligible.sum()), 3) for value in first_probabilities.values()
    )
    assert all(
        value.shape == (int(second_eligible.sum()), 3) for value in second_probabilities.values()
    )
