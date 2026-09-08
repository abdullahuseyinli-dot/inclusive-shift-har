"""Synthetic/mocked contracts only: this module launches no learned model fits."""

from __future__ import annotations

import json
import pickle
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from joblib import parallel_backend  # type: ignore[import-untyped]
from joblib.parallel import get_active_backend  # type: ignore[import-untyped]
from numpy.typing import NDArray

from inclusive_shift_har.experiments import ctgr_source_finalization as final
from inclusive_shift_har.experiments.confidence_triggered_gravity_residual import (
    _candidates,
    _load_config,
    _selection_order,
)
from inclusive_shift_har.experiments.geometric_spectral_pyramid_nested import (
    _build_estimator,
    _participant_class_weights,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class FakeModel:
    """Deterministic fixture; feature zero is an explicit synthetic class code."""

    kind: str
    perfect_base: bool = False
    n_jobs: int = 4

    @property
    def classes_(self) -> NDArray[np.int64]:
        return np.arange(2 if self.kind == "expert" else 3, dtype=np.int64)

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self.n_jobs == 1, "all finalizer predictions must accumulate serially"
        codes = features[:, 0].astype(int)
        if self.kind == "expert":
            sitting = (codes == 1).astype(float)
            return np.column_stack((1.0 - sitting, sitting))
        if self.perfect_base:
            return np.asarray(np.eye(3, dtype=np.float64)[codes], dtype=np.float64)
        return np.tile(np.asarray([0.4, 0.3, 0.3]), (len(features), 1))

    def get_params(self, deep: bool = False) -> dict[str, Any]:
        return {
            "fixture_kind": self.kind,
            "perfect_base": self.perfect_base,
            "n_jobs": self.n_jobs,
        }


def source_fixture() -> final.SourceData:
    labels = np.tile(np.arange(3, dtype=np.int64), 20)
    people = np.repeat(np.asarray(final.PARTICIPANTS), 6)
    base = np.column_stack((labels.astype(float), np.arange(60, dtype=float)))
    physics = base[:, :1].copy()
    views = {
        "physics": physics,
        "physics_plus_rmrp": np.column_stack((physics, base)),
        "total_gsp": base.copy(),
        "dual_gsp": np.column_stack((base, base)),
    }
    return final.SourceData(
        base, views, labels, people, np.asarray([f"fixture-{i}" for i in range(60)])
    )


def candidates() -> list[dict[str, Any]]:
    return _candidates(
        _load_config(ROOT / "configs/experiments/confidence_triggered_gravity_residual_v1.yaml")
    )


def mock_fits(monkeypatch: pytest.MonkeyPatch, *, perfect: bool) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fit_base(features: Any, labels: Any, people: Any, mask: Any, **kwargs: Any) -> FakeModel:
        assert kwargs == {"seed": 11, "n_jobs": 4}
        assert set(people[mask].tolist()).issubset(set(final.PARTICIPANTS))
        calls.append({"kind": "base", "people": set(people[mask].tolist())})
        return FakeModel("base", perfect)

    def fit_expert(
        estimator: str, features: Any, labels: Any, people: Any, mask: Any, **kwargs: Any
    ) -> FakeModel:
        assert estimator in final.ESTIMATORS
        assert kwargs == {"seed": 11, "n_jobs": 4}
        assert set(labels[mask].tolist()) == {0, 1, 2}
        calls.append({"kind": "expert", "people": set(people[mask].tolist())})
        return FakeModel("expert")

    def fit_b9(data: final.SourceData, mask: Any) -> FakeModel:
        calls.append({"kind": "B9", "people": set(data.participants[mask].tolist())})
        return FakeModel("B9", perfect)

    monkeypatch.setattr(final, "_fit_base", fit_base)
    monkeypatch.setattr(final, "_fit_expert", fit_expert)
    monkeypatch.setattr(final, "fit_b9", fit_b9)
    return calls


def test_config_preserves_original_experts_and_finite_scope(tmp_path: Path) -> None:
    path = ROOT / "configs/experiments/ctgr_source_finalization_v1.yaml"
    config = final.load_config(path)
    assert config["expert_estimators"] == ["extra_trees_leaf3", "extra_trees_leaf1"]
    assert config["n_jobs"] == 4 and config["inference_n_jobs"] == 1
    assert len(candidates()) == 121
    config["expert_estimators"] = ["extra_trees_leaf2", "extra_trees_leaf1"]
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="finite contract"):
        final.load_config(changed)


def test_source_partitions_target_rejection_and_feature_order() -> None:
    data = source_fixture()
    final.validate_source(data)
    coverage = np.zeros(len(data.labels), dtype=int)
    for pair in final.PAIRS:
        train, evaluation = final.fold_masks(data, pair)
        assert not np.any(train & evaluation)
        assert len(set(data.participants[train])) == 8
        coverage += evaluation
    assert np.all(coverage == 1)
    data.participants[0] = "11"
    with pytest.raises(PermissionError, match="P1--P10"):
        final.validate_source(data)
    data = source_fixture()
    data.views["physics_plus_rmrp"][:, 0] += 1
    with pytest.raises(ValueError, match="physics-then-RMRP"):
        final.validate_source(data)


def test_b9_contract_differs_from_historical_ctgr_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    labels = np.asarray([0, 0, 1, 1, 2], dtype=np.int64)
    people = np.asarray(["1", "1", "2", "2", "2"])
    weights = final.participant_first_weights(labels, people)
    assert weights.mean() == pytest.approx(1.0)
    assert weights[people == "1"].sum() == pytest.approx(weights[people == "2"].sum())
    assert not np.allclose(weights, _participant_class_weights(labels, people))
    assert _build_estimator("extra_trees_leaf1", seed=11, n_jobs=4).class_weight == "balanced"
    captured: dict[str, Any] = {}

    class ConstructorOnly:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def fit(self, features: Any, labels: Any, **kwargs: Any) -> ConstructorOnly:
            captured["features"] = features
            captured["weights"] = kwargs["sample_weight"]
            return self

    monkeypatch.setattr(final, "ExtraTreesClassifier", ConstructorOnly)
    data = source_fixture()
    final.fit_b9(data, np.ones(len(data.labels), dtype=np.bool_))
    assert captured["class_weight"] is None
    assert captured["n_estimators"] == 500 and captured["min_samples_leaf"] == 1
    assert captured["random_state"] == 11
    assert np.array_equal(captured["features"], data.views["physics_plus_rmrp"])


def test_original_selection_tolerance_tail_trigger_complexity_and_id() -> None:
    grid = [
        {"id": name, "complexity_rank": rank}
        for name, rank in (("best", 5), ("a", 2), ("b", 1), ("z", 1))
    ]
    summaries = [
        {
            "candidate_id": c["id"],
            "mean_participant_macro_f1": 0.8 if c["id"] == "best" else 0.797,
            "lower_30_percent_participant_macro_f1": 0.6 if c["id"] == "best" else 0.7,
            "mean_trigger_fraction": 0.2,
        }
        for c in grid
    ]
    ranking, eligible = _selection_order(summaries, grid, tolerance=0.005)
    assert ranking[0]["candidate_id"] == "b"  # tail, then complexity, then ID
    assert set(eligible) == {"best", "a", "b", "z"}
    summaries[1]["mean_trigger_fraction"] = 0.1
    assert _selection_order(summaries, grid, tolerance=0.005)[0][0]["candidate_id"] == "a"
    summaries[1]["mean_participant_macro_f1"] = 0.79
    assert "a" not in _selection_order(summaries, grid, tolerance=0.005)[1]


@pytest.mark.parametrize("perfect, expected_fits", [(True, 47), (False, 48)])
def test_mocked_recipe_isolation_freeze_base_wins_and_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    perfect: bool,
    expected_fits: int,
) -> None:
    calls = mock_fits(monkeypatch, perfect=perfect)
    output = tmp_path / "run"
    output.mkdir()
    ledger = final.FitLedger(output)
    result = final._fit_recipe(source_fixture(), candidates(), output, ledger)
    assert len(calls) == len(ledger.attempts) == expected_fits
    for fold, pair in enumerate(final.PAIRS):
        for call in calls[fold * 9 : (fold + 1) * 9]:
            assert call["people"] == set(final.PARTICIPANTS) - set(pair)
    assert all(call["people"] == set(final.PARTICIPANTS) for call in calls[45:])
    if perfect:
        assert result["selected_candidate_id"] == "base_no_route"
        assert result["U9_trigger_contrast_status"] == "not_applicable_base_selected"
        with np.load(output / "inference_replay.npz") as replay:
            assert "U9" not in replay.files
            assert np.array_equal(replay["T9"], replay["B6"])
    else:
        assert result["selected_candidate_id"] != "base_no_route"
        assert result["U9_trigger_contrast_status"] == "defined"
    final.write_record(output / "result.json", result)
    final.write_record(output / "worker_shutdown.json", {"remaining_task_threads": []})
    final.seal_artifacts(output)
    validation = final.validate_artifacts(output)
    assert validation["model_fits"] == 0
    assert validation["final_checkpoint_replays"] == expected_fits - 45
    assert validation["final_component_checkpoint_bindings_verified"] is True
    final.write_record(output / "validation.json", validation)
    final.seal_artifacts(output, "completion_manifest.json")
    assert final.validate_artifacts(output)["status"] == "passed"
    with (output / "predictor.pkl").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="integrity"):
        final.validate_artifacts(output)


def test_all_prediction_and_replay_paths_are_serial_without_changing_fit_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit_calls = mock_fits(monkeypatch, perfect=False)
    data = source_fixture()
    prediction_workers = []
    original_predict = FakeModel.predict_proba

    def observe_predict(model: FakeModel, features: NDArray[np.float64]) -> NDArray[np.float64]:
        prediction_workers.append((model.n_jobs, get_active_backend()[1]))
        return original_predict(model, features)

    monkeypatch.setattr(FakeModel, "predict_proba", observe_predict)
    with parallel_backend("threading", n_jobs=4):
        result = final._fit_recipe(data, candidates(), tmp_path, final.FitLedger(tmp_path))
        assert len(prediction_workers) == 51  # 45 OOF + two three-component final replays
        assert get_active_backend()[1] == 4
    final.write_record(tmp_path / "result.json", result)
    final.write_record(tmp_path / "worker_shutdown.json", {"remaining_task_threads": []})
    final.seal_artifacts(tmp_path)
    assert final.validate_artifacts(tmp_path)["status"] == "passed"
    assert len(prediction_workers) == 102  # independent 45 OOF + six final component replays
    with (tmp_path / "predictor.pkl").open("rb") as stream:
        bundle = pickle.load(stream)
    original_bytes = pickle.dumps(bundle)
    monkeypatch.setattr(final, "_feature_views", lambda *args, **kwargs: (data.base, data.views))
    final.predict_native_windows(np.zeros((60, 128, 6)), np.ones((60, 128, 3)), bundle)
    assert len(prediction_workers) == 105
    assert set(prediction_workers) == {(1, 1)}
    assert pickle.dumps(bundle) == original_bytes
    assert len(fit_calls) == 48
    assert all(model.n_jobs == 4 for model in bundle["models"].values())
    for path in (tmp_path / "fits").glob("*-completed.json"):
        assert final.read_record(path)["effective_parameters"]["n_jobs"] == 4


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("final_partition", "checkpoint training partition mismatch"),
        ("substituted_component", "assembled predictor differs from final-fit checkpoints"),
        ("extra_component", "predictor component/config/evidence/selection binding mismatch"),
        ("config", "predictor component/config/evidence/selection binding mismatch"),
        ("evidence_status", "predictor component/config/evidence/selection binding mismatch"),
        ("classes", "predictor component/config/evidence/selection binding mismatch"),
    ],
)
def test_final_checkpoints_independently_bind_predictor_and_full_training_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
    message: str,
) -> None:
    mock_fits(monkeypatch, perfect=False)
    output = tmp_path / "fixture"
    output.mkdir()
    data = source_fixture()
    result = final._fit_recipe(data, candidates(), output, final.FitLedger(output))
    predictor_path = output / "predictor.pkl"
    with predictor_path.open("rb") as stream:
        bundle = pickle.load(stream)
    if defect == "final_partition":
        reference = result["component_checkpoints"]["B6"]
        receipt_path = output / reference["receipt_path"]
        receipt = final.read_record(receipt_path)
        receipt.pop("record_sha256")
        receipt["train_participants"] = list(final.PARTICIPANTS[:-1])
        receipt["record_sha256"] = canonical_json_sha256(receipt)
        # Deliberately reseal only synthetic fixtures: file integrity alone must not
        # hide inconsistent final-training provenance from the independent validator.
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        reference["receipt_record_sha256"] = receipt["record_sha256"]
        bundle["component_checkpoints"] = result["component_checkpoints"]
    elif defect == "substituted_component":
        bundle["models"]["B9"] = FakeModel("B9", True)
        outputs = final.final_probabilities(data, bundle["models"], bundle["selected_candidate"])
        output_arrays: dict[str, Any] = dict(outputs)
        with (output / "inference_replay.npz").open("wb") as stream:
            np.savez(stream, **output_arrays)
    elif defect == "extra_component":
        bundle["models"]["unexpected"] = FakeModel("base")
    elif defect == "config":
        bundle["config"]["seed"] = 23
    elif defect == "evidence_status":
        bundle["evidence_status"] = "confirmation"
    else:
        bundle["class_names"] = tuple(reversed(final.CLASS_NAMES))
    with predictor_path.open("wb") as stream:
        pickle.dump(bundle, stream)
    final.write_record(output / "result.json", result)
    final.write_record(output / "worker_shutdown.json", {"remaining_task_threads": []})
    final.seal_artifacts(output)
    with pytest.raises(ValueError, match=message):
        final.validate_artifacts(output)


def test_attempt_budget_consumes_failure_and_prevents_retry_and_unfrozen_final(
    tmp_path: Path,
) -> None:
    ledger = final.FitLedger(tmp_path)
    with pytest.raises(PermissionError, match="frozen"):
        ledger.fit("final", "premature", lambda: FakeModel("base"))
    assert not ledger.attempts

    def failure() -> None:
        raise RuntimeError("injected fit failure")

    with pytest.raises(RuntimeError, match="injected"):
        ledger.fit("selection", "failed", failure)
    assert len(ledger.attempts) == 1
    assert list((tmp_path / "fits").glob("*-failed.json"))
    ledger.phase_counts["selection"] = 45
    with pytest.raises(RuntimeError, match="fit cap"):
        ledger.fit("selection", "overcap", lambda: FakeModel("base"))
    assert len(ledger.attempts) == 1


def test_create_only_records_and_run_directory(tmp_path: Path) -> None:
    record = tmp_path / "record.json"
    final.write_record(record, {"meaning": "signed difference \u2212 no ASCII substitution"})
    assert final.read_record(record)["meaning"].startswith("signed")
    with pytest.raises(FileExistsError):
        final.write_record(record, {"replacement": True})
    with pytest.raises(FileExistsError):
        final.run_finalization(
            evidence_root=tmp_path,
            config_path=tmp_path / "absent",
            output_directory=tmp_path,
            code_commit="unused",
        )


def test_preflight_failure_is_recorded_without_fit(tmp_path: Path) -> None:
    output = tmp_path / "failure"
    with pytest.raises(FileNotFoundError):
        final.run_finalization(
            evidence_root=tmp_path,
            config_path=tmp_path / "absent",
            output_directory=output,
            code_commit="unused",
        )
    assert final.read_record(output / "INCOMPLETE.json")["fit_attempts"] == 0
    assert final.read_record(output / "worker_shutdown.json")["remaining_task_threads"] == []
    assert (output / "artifact_manifest.json").is_file()


def test_native_inference_interface_has_no_labels_or_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = source_fixture()
    monkeypatch.setattr(final, "_feature_views", lambda *args, **kwargs: (data.base, data.views))
    bundle = {
        "config": final.expected_config(),
        "class_names": final.CLASS_NAMES,
        "models": {"B6": FakeModel("base", True), "B9": FakeModel("B9", True)},
        "selected_candidate": candidates()[0],
    }
    signals = np.zeros((60, 128, 6), dtype=np.float64)
    gravity = np.ones((60, 128, 3), dtype=np.float64)
    probability = final.predict_native_windows(signals, gravity, bundle)
    assert set(probability) == {"B6", "B9", "T9"}
    assert np.array_equal(probability["B6"], probability["T9"])
    gravity[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite aligned"):
        final.predict_native_windows(signals, gravity, bundle)


def test_postrun_validation_failure_never_qualifies_scientific_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_input = tmp_path / "fixture.yaml"
    fake_input.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        final,
        "INPUTS",
        {
            "ctgr_config": (fake_input.name, sha256_file(fake_input)),
        },
    )
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda command, **kwargs: "fixture-commit\n" if "rev-parse" in command else "",
    )
    monkeypatch.setattr(final, "_load_ctgr_config", lambda path: {})
    monkeypatch.setattr(final, "_candidates", lambda config: [])
    monkeypatch.setattr(final, "_load_data", lambda paths: source_fixture())
    monkeypatch.setattr(final, "_fit_recipe", lambda *args: {"status": "complete"})

    def fail_validation(output: Path) -> dict[str, Any]:
        raise ValueError("injected post-run integrity failure")

    monkeypatch.setattr(final, "validate_artifacts", fail_validation)
    output = tmp_path / "validation-failure"
    with pytest.raises(ValueError, match="injected post-run"):
        final.run_finalization(
            evidence_root=tmp_path,
            config_path=ROOT / "configs/experiments/ctgr_source_finalization_v1.yaml",
            output_directory=output,
            code_commit="fixture-commit",
        )
    record = final.read_record(output / "validation.json")
    assert record["status"] == "failed"
    assert record["scientific_result_qualified"] is False
    assert record["model_fits"] == 0
    assert final.read_record(output / "runtime.json")["fit_attempts"] == 0
    assert (output / "completion_manifest.json").is_file()
