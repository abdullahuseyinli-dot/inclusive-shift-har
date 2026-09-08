"""Tiny fixtures exercise independent metrics and fail-visible evidence replay."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from inclusive_shift_har.experiments import portable_evidence_replay as replay_module
from inclusive_shift_har.experiments.portable_evidence_replay import (
    CTGR_METHODS,
    L9V_METHODS,
    metrics,
    relative_input,
    run,
    sealed,
    verify_record,
)
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    load_json_strict,
    sha256_file,
)


def test_metrics_equal_people_fixed_absent_class_and_probability_scores() -> None:
    labels = np.array([0, 0, 0, 1], dtype=np.int64)
    people = np.array(["a", "a", "a", "b"])
    p = np.array([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.1, 0.2, 0.7]])
    report = metrics(labels, p, people)
    assert report["participants"][0]["macro_f1"] == pytest.approx(1 / 3)
    assert report["primary"]["mean_participant_macro_f1"] == pytest.approx(1 / 6)
    assert report["primary"]["bottom_30_percent_participant_macro_f1"] == 0
    assert report["primary"]["mean_participant_nll"] == pytest.approx(
        (-math.log(0.8) - math.log(0.2)) / 2
    )
    assert report["primary"]["mean_participant_multiclass_brier"] == pytest.approx(
        (0.06 + 1.14) / 2
    )
    assert report["pooled"]["accuracy"] == 0.75
    assert report["primary"]["participant_macro_class_recall"] == {
        "mobility": 0.5,
        "sitting": 0.0,
        "standing": 0.0,
    }


def fixtures(root: Path, *, fold_commit: str = "fixture") -> tuple[Path, dict[str, Any]]:
    root.mkdir()
    y = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    people = np.array(["a", "a", "a", "b", "b", "b"])
    windows = np.array([f"w{i}" for i in range(6)])
    p = np.eye(3)[y] * 0.7 + 0.1
    report = metrics(y, p, people)

    def json_file(relative: str, value: dict[str, Any]) -> dict[str, str]:
        path = root / relative
        atomic_write_json_new(sealed(value), path, allowed_root=root)
        return {"path": relative, "sha256": sha256_file(path)}

    ct = root / "ctgr"
    ct.mkdir()
    payload: dict[str, Any] = {f"{name}_probabilities": p for name in CTGR_METHODS}
    np.savez(
        ct / "all_outer_predictions.npz",
        labels=y,
        participant_ids=people,
        window_ids=windows,
        **payload,
    )
    prediction_spec = {
        "path": "ctgr/all_outer_predictions.npz",
        "sha256": sha256_file(ct / "all_outer_predictions.npz"),
    }
    expected_ctgr = {
        **report,
        "window_level_diagnostics": report["pooled"],
        "calibration": {
            "negative_log_likelihood": report["pooled"]["nll"],
            "multiclass_brier_score": report["pooled"]["multiclass_brier"],
        },
    }
    source_spec = json_file(
        "source/source_windows.json",
        {
            "manifest_kind": "source_development_windows",
            "dataset_id": "fixture",
            "target_subject_or_window_records_included": False,
            "target_performance_or_prediction_accessed": False,
            "source_artifact_sha256": "a" * 64,
            "source_window_count": 6,
            "windows": [
                {
                    "window_id": window,
                    "subject_id": person,
                    "canonical_labels": {"functional_core": replay_module.CLASSES[label]},
                }
                for window, person, label in zip(
                    windows.tolist(), people.tolist(), y.tolist(), strict=True
                )
            ],
        },
    )
    dataset_spec = json_file(
        "source/dataset.json",
        {
            "manifest_kind": "dataset",
            "dataset_id": "fixture",
            "artifacts": [
                {
                    "role": "raw_sensor_table",
                    "expected_sha256": "a" * 64,
                    "expected_size_bytes": 12345,
                }
            ],
        },
    )
    folds = []
    for index in range(1, 6):
        fold_id = f"source_cv_{index:02d}"
        model = ct / fold_id / "models.pkl"
        model.parent.mkdir()
        model.write_bytes(f"opaque fixture model {index}".encode())
        fold_record = json_file(
            f"ctgr/{fold_id}/result.json",
            {
                "record_kind": "confidence_triggered_gravity_residual_outer_fold",
                "seed": 11,
                "outer_fold_id": fold_id,
                "code_commit": fold_commit,
                "models": {"path": f"ctgr/{fold_id}/models.pkl", "sha256": sha256_file(model)},
            },
        )
        folds.append(
            {
                **fold_record,
                "outer_fold_id": fold_id,
                "record_sha256": load_json_strict(root / fold_record["path"])["record_sha256"],
            }
        )
    result_spec = json_file(
        "ctgr/result.json",
        {
            "record_kind": "confidence_triggered_gravity_residual_fixed_seed_aggregate",
            "seed": 11,
            "confirmatory_claim_allowed": False,
            "target_performance_or_prediction_accessed": False,
            "predictions": prediction_spec,
            "aggregate_reports": {name: expected_ctgr for name in CTGR_METHODS},
            "evidence_status": "development",
            "code_commit": "fixture",
            "folds": folds,
            "source_manifest": source_spec,
            "dataset_manifest": dataset_spec,
        },
    )
    ct_config = {
        "predictions": prediction_spec,
        "result": result_spec,
        "scored_rows": 6,
        "participant_ids": ["a", "b"],
    }
    lg = root / "l9v"
    lg.mkdir()
    mask = np.array([True, False, True, True, True, True])
    stack = np.stack([p] * 5)
    fold = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    np.savez(
        lg / "predictions.npz",
        method_ids=np.array(L9V_METHODS),
        observable_probabilities=stack,
        scored_probabilities=stack,
        l9v_available_probabilities=p[mask],
        availability_mask=mask,
        available_indices=np.flatnonzero(mask),
        scoring_indices=np.arange(6),
        scored_labels=y,
        observable_window_ids=windows,
        observable_participant_ids=people,
        observable_fold_index=fold,
        scored_participant_ids=people,
        scored_window_ids=windows,
        scored_fold_index=fold,
        observable_decisions=stack.argmax(2),
        scored_decisions=stack.argmax(2),
        fallback_source_method_index=np.array([3]),
        fallback_source_method_id=np.array(["b"]),
    )
    lg_prediction = {"path": "l9v/predictions.npz", "sha256": sha256_file(lg / "predictions.npz")}
    lg_result = json_file(
        "l9v/result.json",
        {
            "InclusiveHAR_P11_P20_loaded": False,
            "completed_fit_count": 5,
            "advancement": {"status": "fail"},
            "evidence_status": "development",
            "code_commit": "fixture",
            "method_summary": {name: report["primary"] for name in L9V_METHODS},
        },
    )
    analysis = json_file("l9v/analysis.json", {"reports": {name: report for name in L9V_METHODS}})
    provenance_specs = [
        json_file(
            "l9v/source_manifest.json",
            {
                "record_kind": "fog_l9v_source_manifest",
                "files": [{"path": "src/fixture.py", "sha256": "b" * 64}],
            },
        ),
        json_file(
            "l9v/input_manifest.json",
            {
                "record_kind": "fog_l9v_input_manifest",
                "raw_source": {"sha256": "c" * 64, "size_bytes": 123456},
            },
        ),
    ]
    model_entries = []
    (lg / "checkpoints").mkdir()
    for index in range(5):
        relative = f"checkpoints/l9v--fold-{index}.pkl"
        model = lg / relative
        model.write_bytes(f"opaque L9v fixture {index}".encode())
        model_entries.append(
            {"path": relative, "sha256": sha256_file(model), "size_bytes": model.stat().st_size}
        )
    completion = json_file(
        "l9v/completion_manifest.json",
        {
            "artifacts": [
                {"path": Path(spec["path"]).name, "sha256": spec["sha256"]}
                for spec in (lg_prediction, lg_result, analysis, *provenance_specs)
            ]
            + model_entries
        },
    )
    erratum = json_file(
        "erratum/ERRATUM.json",
        {
            "predictions_file_sha256": lg_prediction["sha256"],
            "scientific_effect": {"predictions_affected": False},
            "defect": {"stored_unique_values": ["b"]},
            "gate_interpretation": {"complete_redundant_provenance_fields_exact": False},
        },
    )
    config = {
        "experiment_id": "portable-evidence-replay-v1",
        "model_fits": 0,
        "cross_lane_comparison": False,
        "provenance": {
            "model_byte_hashes": True,
            "manifest_byte_hashes": True,
            "model_deserialization": False,
            "raw_bytes_rehashed": False,
        },
        "lanes": {
            "ctgr": ct_config,
            "l9v": {
                "predictions": lg_prediction,
                "result": lg_result,
                "analysis": analysis,
                "completion": completion,
                "erratum": erratum,
                "observable_rows": 6,
                "scored_rows": 6,
                "participant_ids": ["a", "b"],
                "observable_fallback_rows": 1,
                "scored_fallback_rows": 1,
            },
        },
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, config


def test_complete_portable_replay_create_only_and_sealed(tmp_path: Path) -> None:
    config_path, _ = fixtures(tmp_path / "evidence")
    out = tmp_path / "out"
    result = run(tmp_path / "evidence", out, config_path)
    assert result["status"] == "pass"
    assert result["model_fits"] == 0
    assert result["cross_lane_comparison"] is False
    replay = load_json_strict(out / "l9v_replay.json")
    assert replay["fallback_numeric_source_and_probability_exact"] is True
    assert replay["fallback_string_provenance_complete"] is False
    assert "mandatory_erratum" in replay
    assert replay["byte_and_source_provenance"]["model_checkpoint_count"] == 5
    assert replay["byte_and_source_provenance"]["raw_bytes_rehashed"] is False
    ctgr_replay = load_json_strict(out / "ctgr_replay.json")
    assert ctgr_replay["byte_and_source_provenance"]["model_bundle_count"] == 5
    assert ctgr_replay["byte_and_source_provenance"]["models_deserialized"] is False
    manifest = verify_record(load_json_strict(out / "completion_manifest.json"))
    for row in manifest["artifacts"]:
        assert sha256_file(out / row["path"]) == row["sha256"]
        verify_record(load_json_strict(out / row["path"]))
    assert load_json_strict(out / "shutdown_receipt.json")["task_owned_workers_remaining"] == 0
    with pytest.raises(FileExistsError):
        run(tmp_path / "evidence", out, config_path)


def test_corruption_fails_one_lane_and_keeps_other_evidence(tmp_path: Path) -> None:
    config_path, _ = fixtures(tmp_path / "evidence")
    with (tmp_path / "evidence/ctgr/all_outer_predictions.npz").open("ab") as stream:
        stream.write(b"corruption")
    out = tmp_path / "out"
    result = run(tmp_path / "evidence", out, config_path)
    assert result["status"] == "fail"
    assert result["completed_lanes"] == ["l9v"]
    assert "pinned hash mismatch" in result["failures"][0]["message"]
    assert (out / "failure.json").exists()
    assert (out / "shutdown_receipt.json").exists()


@pytest.mark.parametrize(
    ("relative", "failed_lane"),
    [
        ("ctgr/source_cv_01/models.pkl", "ctgr"),
        ("l9v/checkpoints/l9v--fold-0.pkl", "l9v"),
        ("source/source_windows.json", "ctgr"),
        ("source/dataset.json", "ctgr"),
        ("l9v/source_manifest.json", "l9v"),
    ],
)
def test_model_or_source_manifest_corruption_is_visible(
    tmp_path: Path,
    relative: str,
    failed_lane: str,
) -> None:
    config_path, _ = fixtures(tmp_path / "evidence")
    with (tmp_path / "evidence" / relative).open("ab") as stream:
        stream.write(b"corruption")
    result = run(tmp_path / "evidence", tmp_path / "out", config_path)
    assert result["status"] == "fail"
    assert result["failures"][0]["lane"] == failed_lane
    assert "pinned hash mismatch" in result["failures"][0]["message"]
    assert result["model_fits"] == 0


def test_missing_erratum_cannot_be_ignored(tmp_path: Path) -> None:
    config_path, config = fixtures(tmp_path / "evidence")
    config["lanes"]["l9v"].pop("erratum")
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = run(tmp_path / "evidence", tmp_path / "out", config_path)
    assert result["status"] == "fail"
    assert result["completed_lanes"] == ["ctgr"]
    assert result["failures"][0]["lane"] == "l9v"


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("duplicate_id", "duplicate window ID"),
        ("fallback", "fallback probabilities must be byte-exact B0"),
        ("fold", "participant spans evaluation folds"),
    ],
)
def test_array_invariants_independent_of_reported_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
    message: str,
) -> None:
    config_path, _ = fixtures(tmp_path / "evidence")
    original = replay_module.load_arrays

    def altered(path: Path, keys: set[str]) -> dict[str, Any]:
        arrays = original(path, keys)
        if path.name == "predictions.npz":
            if defect == "duplicate_id":
                arrays["observable_window_ids"][1] = arrays["observable_window_ids"][0]
            elif defect == "fold":
                arrays["observable_fold_index"][0] = 1
            else:
                # Change only confidence, retaining hard labels and historical F1.
                arrays["observable_probabilities"][0, 1] = [0.15, 0.75, 0.1]
                arrays["scored_probabilities"][0, 1] = [0.15, 0.75, 0.1]
        return arrays

    monkeypatch.setattr(replay_module, "load_arrays", altered)
    result = run(tmp_path / "evidence", tmp_path / "out", config_path)
    assert result["status"] == "fail"
    assert result["completed_lanes"] == ["ctgr"]
    assert message in result["failures"][0]["message"]


def test_path_escape_and_self_hash_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="relative"):
        relative_input(tmp_path, "../outside.json")
    with pytest.raises(ValueError, match="raw"):
        relative_input(tmp_path, "data/raw/result.json")
    with pytest.raises(ValueError, match="raw"):
        relative_input(tmp_path, "data/RAW/result.json")
    with pytest.raises(ValueError, match="self-hash"):
        verify_record({"record_sha256": "wrong"})


@pytest.mark.parametrize("p", [np.array([[float("nan"), 0.0, 1.0]]), np.array([[0.1, 0.1, 0.1]])])
def test_nonfinite_or_invalid_simplex_rejected(p: Any) -> None:
    with pytest.raises(ValueError):
        metrics(np.array([0], dtype=np.int64), p, np.array(["a"]))


def test_ctgr_fold_commit_must_match_aggregate(tmp_path: Path) -> None:
    config_path, _ = fixtures(tmp_path / "evidence", fold_commit="different-commit")
    result = run(tmp_path / "evidence", tmp_path / "out", config_path)
    assert result["status"] == "fail"
    assert "fold/aggregate source commit binding" in result["failures"][0]["message"]


def test_ctgr_scored_labels_must_match_source_ontology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _ = fixtures(tmp_path / "evidence")
    original = replay_module.load_arrays

    def wrong_label(path: Path, keys: set[str]) -> dict[str, Any]:
        arrays = original(path, keys)
        if path.name == "all_outer_predictions.npz":
            arrays["labels"][0] = 1
        return arrays

    monkeypatch.setattr(replay_module, "load_arrays", wrong_label)
    result = run(tmp_path / "evidence", tmp_path / "out", config_path)
    assert result["status"] == "fail"
    assert "OOF label/source ontology binding" in result["failures"][0]["message"]


@pytest.mark.parametrize("interrupted_lane", ["ctgr", "l9v"])
def test_keyboard_interrupt_preserves_failure_shutdown_and_prior_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted_lane: str,
) -> None:
    config_path, _ = fixtures(tmp_path / "evidence")
    output = tmp_path / "out"

    def interrupt(*args: Any, **kwargs: Any) -> dict[str, Any]:
        (output / f".{interrupted_lane}_replay.json.partial.fixture").write_bytes(b"{partial")
        raise KeyboardInterrupt("injected interruption")

    monkeypatch.setattr(replay_module, f"replay_{interrupted_lane}", interrupt)
    with pytest.raises(KeyboardInterrupt, match="injected interruption"):
        run(tmp_path / "evidence", output, config_path)
    failure = verify_record(load_json_strict(output / "failure.json"))
    assert failure["status"] == "fail"
    assert failure["failures"][0]["lane"] == interrupted_lane
    assert failure["failures"][0]["error_type"] == "KeyboardInterrupt"
    assert (output / f"{interrupted_lane}_failure.json").is_file()
    assert failure["completed_lanes"] == ([] if interrupted_lane == "ctgr" else ["ctgr"])
    assert load_json_strict(output / "shutdown_receipt.json")["task_owned_workers_remaining"] == 0
    manifest = verify_record(load_json_strict(output / "completion_manifest.json"))
    assert manifest["status"] == "fail"
    assert sum(entry["partial_file"] for entry in manifest["artifacts"]) == 1
    for entry in manifest["artifacts"]:
        assert sha256_file(output / entry["path"]) == entry["sha256"]
