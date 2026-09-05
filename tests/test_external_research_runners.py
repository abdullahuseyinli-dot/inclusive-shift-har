"""Synthetic end-to-end runner and inference-isolation regressions.

Training seams are replaced only where the full nested research budget would be
inappropriate for CI; production reports, archive serialization and failures run.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from inclusive_shift_har.data.external_har import ExternalHARWindows, SourceReceipt
from inclusive_shift_har.experiments import (
    cross_dataset_har,
    cross_dataset_neural,
    cross_dataset_transfer,
    external_primary_suite,
    har_pmd_stress,
    publication_campaign,
)
from inclusive_shift_har.experiments.external_evidence_validate import validate_run_directory


def _data(dataset_id: str, classes: int = 3) -> ExternalHARWindows:
    labels = np.tile(np.arange(classes), 12)
    signals = np.random.default_rng(122).normal(size=(len(labels), 128, 6)).astype(np.float32)
    signals[:, :, 0] += labels[:, None] * 3.0
    persons = np.repeat([f"{dataset_id}:p{i:02d}" for i in range(12)], classes)
    return ExternalHARWindows(
        dataset_id=dataset_id,
        channel_lane="native-gravity-9ch" if classes == 5 else "derived-gravity-9ch",
        sampling_rate_hz=50.0,
        signals=signals,
        gravity=np.ones((len(labels), 128, 3), dtype=np.float32),
        labels=labels,
        participant_ids=persons,
        session_ids=np.char.add(persons, ":indoor"),
        trial_ids=np.array([f"{dataset_id}:t{i}" for i in range(len(labels))]),
        window_ids=np.array([f"{dataset_id}:w{i}" for i in range(len(labels))]),
        class_names=("mobility", "sitting", "standing")
        if classes == 3
        else ("stationary", "walking", "crutches", "walker", "manual_wheelchair"),
        receipts=(
            SourceReceipt(dataset_id, "https://example.invalid/synthetic", None, 1, 1, "0" * 64),
        ),
    )


def test_neural_runner_serializes_all_seeds_and_uses_label_free_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _data("synthetic-neural")
    visits: list[set[str]] = []

    def training(
        data: ExternalHARWindows, signals: NDArray[np.float32], **kwargs: Any
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        train = set(data.participant_ids[kwargs["training_indices"]])
        valid = set(data.participant_ids[kwargs["validation_indices"]])
        test = set(data.participant_ids[kwargs["evaluation_indices"]])
        assert not train & valid and not train & test and not valid & test
        visits.append(test)
        values = signals[kwargs["evaluation_indices"]]
        predicted = np.clip(np.rint(values[:, :, 0].mean(axis=1) / 3.0), 0, 2).astype(int)
        return np.eye(3)[predicted], {"synthetic_training_seam": True}

    monkeypatch.setattr(cross_dataset_neural, "_train_fold", training)
    root = Path(__file__).resolve().parents[1]
    result = cross_dataset_neural.run_and_write_neural(
        data=data,
        output_directory=tmp_path / "neural",
        repository_root=root,
        seeds=(11, 23, 47),
        epochs=40,
    )
    assert len(visits) == 30
    assert len(result["fold_records"]) == 30
    with np.load(tmp_path / "neural/predictions.npz") as archive:
        assert "probability__seed-47__TinyHAR-6ch" in archive
    validation = validate_run_directory(tmp_path / "neural", root)
    assert validation["integrity_passed"] is True
    with pytest.raises(FileExistsError):
        cross_dataset_neural.run_and_write_neural(
            data=data,
            output_directory=tmp_path / "neural",
            repository_root=root,
            seeds=(11,),
            epochs=40,
        )


def test_zero_shot_runner_keeps_target_labels_out_of_fit_and_preserves_each_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target = _data("imu_har_il_v1"), _data("fog_star_v3")

    def fit(
        source: ExternalHARWindows, target: ExternalHARWindows, **kwargs: Any
    ) -> tuple[dict[str, NDArray[np.float64]], dict[str, Any]]:
        assert not set(source.participant_ids) & set(target.participant_ids)
        predicted = np.clip(np.rint(target.signals[:, :, 0].mean(axis=1) / 3.0), 0, 2).astype(int)
        probability = np.eye(3)[predicted]
        names = (*cross_dataset_har._INVENTION_METHODS, "XGBoost-6ch")
        return {name: probability for name in names}, {"synthetic_training_seam": True}

    monkeypatch.setattr(cross_dataset_transfer, "_fit_apply_seed", fit)
    result = cross_dataset_transfer.run_and_write_transfer(
        source=source,
        target=target,
        output_directory=tmp_path / "transfer",
        repository_root=Path(__file__).resolve().parents[1],
        seeds=(11, 23, 47),
        n_jobs=1,
        include_classical=True,
    )
    assert result["primary_seed_averaged"]["participant_count"] == 12
    assert (tmp_path / "transfer/result.json").is_file()
    validation = validate_run_directory(tmp_path / "transfer", Path(__file__).resolve().parents[1])
    assert validation["integrity_passed"] is True
    assert (
        validation["publication_evidence_ready"] is False
    )  # Synthetic FoG has no real grid audit.


def test_primary_suite_preserves_acquisition_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(**kwargs: Any) -> ExternalHARWindows:
        raise ConnectionError("synthetic access failure")

    monkeypatch.setattr(external_primary_suite, "load_imu_har_il", unavailable)
    with pytest.raises(ConnectionError):
        external_primary_suite.run_primary_suite(
            output_root=tmp_path / "suite",
            repository_root=Path(__file__).resolve().parents[1],
            seeds=(11, 23, 47),
            epochs=40,
            n_jobs=1,
        )
    failure = json.loads((tmp_path / "suite/suite_failure.json").read_text())
    assert failure["status"] == "FAILED_PRESERVED"
    assert failure["source_input_manifest"]["protocol_id"] == "external-har-session-grid-v3"
    assert not (tmp_path / "suite/suite_complete.json").exists()


@pytest.mark.parametrize("policy", ["complete_requested_core", "available_valid_trials"])
def test_primary_suite_routes_cohort_policy_and_orders_acceptance_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    events: list[str] = []
    source, target = _data("imu_har_il_v1"), _data("fog_star_v3")

    def acquire(**kwargs: Any) -> ExternalHARWindows:
        assert kwargs["selection_policy"] == policy
        assert kwargs["repetition_limit"] == 4
        events.append("source")
        return source

    def acquire_target(**kwargs: Any) -> ExternalHARWindows:
        events.append("target_after_source_gates")
        return target

    def runner(stage: str) -> Any:
        def run(**kwargs: Any) -> dict[str, Any]:
            assert kwargs["seeds"] == (11, 23, 47)
            if stage == "transfer":
                assert kwargs["source"] is source and kwargs["target"] is target
            else:
                assert kwargs["data"] is source
            kwargs["output_directory"].mkdir()
            events.append(stage)
            return {"reports": {}}

        return run

    monkeypatch.setattr(external_primary_suite, "load_imu_har_il", acquire)
    monkeypatch.setattr(external_primary_suite, "load_fog_star", acquire_target)
    monkeypatch.setattr(external_primary_suite, "run_and_write", runner("classical"))
    monkeypatch.setattr(external_primary_suite, "run_and_write_neural", runner("neural"))
    monkeypatch.setattr(external_primary_suite, "run_and_write_transfer", runner("transfer"))
    monkeypatch.setattr(
        external_primary_suite, "_validate_stage", lambda *_args: events.append("gate")
    )
    result = external_primary_suite.run_primary_suite(
        output_root=tmp_path / "suite",
        repository_root=Path(__file__).resolve().parents[1],
        seeds=(11, 23, 47),
        epochs=40,
        n_jobs=1,
        selection_policy=policy,
    )
    assert result["cohort_selection_policy"] == policy
    assert events == [
        "source",
        "classical",
        "gate",
        "neural",
        "gate",
        "target_after_source_gates",
        "transfer",
        "gate",
    ]
    plan = json.loads((tmp_path / "suite/suite_plan.json").read_text())
    assert plan["cohort_selection_policy"] == policy
    tag = "available_trials" if policy == "available_valid_trials" else "all_repetitions"
    assert (tmp_path / f"suite/imu_har_il_{tag}_3seed").is_dir()


def test_available_campaign_cli_dispatches_explicit_selection_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        publication_campaign,
        "_git_state",
        lambda _root: {"worktree_dirty": False, "commit": "synthetic", "status_entries": []},
    )
    monkeypatch.setattr(publication_campaign, "_source_input_manifest", lambda _root: {"files": {}})
    monkeypatch.setattr(publication_campaign, "_manifest_commit_errors", lambda *_args: [])
    monkeypatch.setattr(
        publication_campaign, "run_primary_suite", lambda **kwargs: calls.append(kwargs)
    )
    assert (
        publication_campaign.main(
            [
                "--dataset",
                "imu-har-il-available",
                "--output",
                str(tmp_path / "campaign"),
                "--repository-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert len(calls) == 1 and calls[0]["selection_policy"] == "available_valid_trials"
    assert calls[0]["seeds"] == (11, 23, 47) and calls[0]["epochs"] == 40
    assert (tmp_path / "campaign/campaign_complete.json").is_file()


def test_har_pmd_runner_has_separate_native_and_six_channel_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _data("har_pmd_v1", 5)
    # Two observed environments keep the actual environment report path exercised.
    data = replace(
        data,
        session_ids=np.array(
            [
                person + (":indoor" if index % 2 else ":outdoor")
                for index, person in enumerate(data.participant_ids)
            ]
        ),
    )
    calls: list[int] = []

    def fit(
        windows: NDArray[np.float32],
        labels: NDArray[np.int64],
        participants: list[str],
        **kwargs: Any,
    ) -> int:
        calls.append(windows.shape[2])
        return int(windows.shape[2])

    def predict(
        fitted: int, windows: NDArray[np.float32]
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        assert fitted == windows.shape[2]
        predicted = np.clip(np.rint(windows[:, :, 0].mean(axis=1) / 3.0), 0, 4).astype(np.int64)
        return predicted, np.eye(5)[predicted]

    monkeypatch.setattr(har_pmd_stress, "fit_classical_model", fit)
    monkeypatch.setattr(har_pmd_stress, "predict_classical_probabilities", predict)
    result = har_pmd_stress.run_and_write(
        data=data,
        output_directory=tmp_path / "pmd",
        repository_root=Path(__file__).resolve().parents[1],
        seeds=(11, 23, 47),
    )
    assert calls.count(6) == calls.count(9) == 30
    assert result["dataset"]["participant_count"] == 12
    assert len(result["reports"]) == 4
    assert "primary_seed_averaged" in result
