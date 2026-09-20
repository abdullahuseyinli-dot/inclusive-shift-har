from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from inclusive_shift_har.experiments import ctgr_native9_confirmation as lane
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file

CONFIG = Path("configs/experiments/ctgr_native9_confirmation_v1.yaml")


def _write_record(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    value = dict(value)
    value["record_sha256"] = canonical_json_sha256(value)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return value


def _config(tmp_path: Path, *, bootstrap_replicates: int = 1_000) -> Path:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    value["gates"]["bootstrap"]["replicates"] = bootstrap_replicates
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _cohort(tmp_path: Path, *, include_labels: bool = False) -> tuple[Path, Path]:
    people = [f"fresh:{index:03d}" for index in range(38)]
    participants = np.repeat(np.asarray(people, dtype=np.str_), 2)
    windows = np.asarray([f"window:{index:03d}" for index in range(76)], dtype=np.str_)
    signals = np.zeros((76, 128, 6), dtype=np.float64)
    signals[:, :, 0] = 0.5 * lane.STANDARD_GRAVITY_M_S2
    signals[:, :, 3] = 0.25
    gravity = np.zeros((76, 128, 3), dtype=np.float64)
    gravity[:, :, 2] = lane.STANDARD_GRAVITY_M_S2
    available = np.ones(76, dtype=np.bool_)
    available[1] = False
    arrays: dict[str, Any] = {
        "signals": signals,
        "native_gravity": gravity,
        "native_gravity_available": available,
        "participant_ids": participants,
        "window_ids": windows,
    }
    if include_labels:
        arrays["labels"] = np.arange(76, dtype=np.int64) % 3
    archive = tmp_path / "windows_label_blind.npz"
    with archive.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    qualification = tmp_path / "qualification.json"
    qualification.write_text("{}\n", encoding="utf-8")
    manifest = _write_record(
        tmp_path / "cohort_manifest.json",
        {
            "schema_version": "1.0.0",
            "record_kind": "ctgr_native9_confirmation_cohort_manifest",
            "cohort_id": "fresh-native-nine-test",
            "evidence_status": "fresh_untouched_confirmation_cohort",
            "freshness_adjudication": {"passed": True},
            "provider_native_gravity": True,
            "ability_relevant": True,
            "participant_partition_before_windowing": True,
            "sampling_rate_hz": 50.0,
            "window_samples": 128,
            "stride_samples": 128,
            "class_names": list(lane.CLASS_NAMES),
            "channel_order": [
                "lin_acc_x",
                "lin_acc_y",
                "lin_acc_z",
                "gyro_x",
                "gyro_y",
                "gyro_z",
                "gravity_x",
                "gravity_y",
                "gravity_z",
            ],
            "units": {
                "linear_acceleration": "m/s^2",
                "angular_velocity": "rad/s",
                "gravity": "m/s^2",
            },
            "participant_roster": people,
            "qualification_record": {
                "path": qualification.name,
                "sha256": sha256_file(qualification),
            },
            "window_archive": {"path": archive.name, "sha256": sha256_file(archive)},
            "labels_separately_custodied": True,
            "labels_available_to_prediction_controller": False,
        },
    )
    assert manifest["record_sha256"]
    return tmp_path / "cohort_manifest.json", archive


def _probabilities(labels: np.ndarray, *, perfect: bool) -> np.ndarray:
    values = np.full((labels.size, 3), 0.05, dtype=np.float64)
    values[np.arange(labels.size), labels] = 0.90
    if not perfect:
        sitting = np.flatnonzero(labels == 1)[::2]
        values[sitting] = np.asarray([0.05, 0.05, 0.90])
    return values


def test_power_analysis_freezes_38_complete_participants() -> None:
    config = lane._read_config(CONFIG)
    result = lane.power_analysis(config)
    assert result["minimum_complete_participants"] == 38
    primary = next(
        item for item in result["sensitivity"] if item["paired_effect"] == pytest.approx(0.02)
    )
    assert primary["attained_power"] == pytest.approx(0.8086594802649786)


def test_prediction_input_rejects_any_label_array(tmp_path: Path) -> None:
    manifest, archive = _cohort(tmp_path, include_labels=True)
    with pytest.raises(PermissionError, match="label-like arrays"):
        lane._cohort_inputs(manifest, archive, lane._read_config(CONFIG))


def test_missing_native_gravity_uses_exact_b6_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, archive = _cohort(tmp_path)
    package = tmp_path / "predictor"
    package.mkdir()
    with (package / "predictor.pkl").open("wb") as stream:
        pickle.dump({}, stream)
    monkeypatch.setattr(lane, "_predictor_binding", lambda *_args, **_kwargs: {"status": "test"})

    def fake_predict(
        signals: np.ndarray, _gravity: np.ndarray, _bundle: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        assert np.allclose(signals[:, :, 0], 0.5)
        assert np.all(signals[:, :, 3] == 0.25)
        assert np.allclose(_gravity[:, :, 2], 1.0) or np.all(_gravity == 0.0)
        count = signals.shape[0]
        base = np.tile(np.asarray([[0.6, 0.3, 0.1]]), (count, 1))
        return {
            "B6": base,
            "B9": np.tile(np.asarray([[0.5, 0.3, 0.2]]), (count, 1)),
            "T9": np.tile(np.asarray([[0.4, 0.3, 0.3]]), (count, 1)),
            "U9": np.tile(np.asarray([[0.3, 0.3, 0.4]]), (count, 1)),
        }

    monkeypatch.setattr(lane, "_predict_native_windows", fake_predict)
    with np.load(archive, allow_pickle=False) as arrays:
        windows = arrays["window_ids"]
    secondary = tmp_path / "secondary.npz"
    secondary_payload: dict[str, Any] = {"window_ids": windows}
    secondary_payload.update(
        {
            method: np.tile(np.asarray([[0.6, 0.3, 0.1]]), (windows.size, 1))
            for method in lane.SECONDARY_METHODS
        }
    )
    with secondary.open("wb") as stream:
        np.savez_compressed(stream, **secondary_payload)
    _write_record(
        secondary.with_suffix(".json"),
        {
            "schema_version": "1.0.0",
            "record_kind": "ctgr_native9_secondary_prediction_seal",
            "status": "sealed_before_target_labels",
            "evidence_status": "source_only_secondary_controls",
            "prediction_archive_sha256": sha256_file(secondary),
            "methods": list(lane.SECONDARY_METHODS),
            "source_only_training": True,
            "target_labels_accessed": False,
            "target_labels_used_for_fit_selection_or_calibration": False,
            "checkpoint_receipts": {
                method: {"sha256": "a" * 64, "method_contract": f"frozen {method}"}
                for method in lane.SECONDARY_METHODS
            },
        },
    )
    output = tmp_path / "prediction-run"
    lane.predict(
        config_path=CONFIG,
        predictor_package=package,
        cohort_manifest=manifest,
        windows_archive=archive,
        secondary_predictions=secondary,
        output=output,
    )
    with np.load(output / "predictions_label_blind.npz", allow_pickle=False) as predictions:
        missing = ~predictions["native_gravity_available"]
        assert missing.sum() == 1
        for method in lane.CORE_METHODS:
            assert predictions[method][missing].tobytes() == predictions["B6"][missing].tobytes()


def test_si_and_g_cohorts_reach_the_identical_physical_model_interface() -> None:
    native = np.zeros((2, 128, 6), dtype=np.float64)
    native[:, :, 0] = 0.5
    native[:, :, 3:] = 0.25
    gravity = np.zeros((2, 128, 3), dtype=np.float64)
    gravity[:, :, 2] = 1.0
    si = native.copy()
    si[:, :, :3] *= lane.STANDARD_GRAVITY_M_S2
    si_before = si.copy()
    converted, converted_g, receipt = lane._to_frozen_model_units(
        si,
        gravity * lane.STANDARD_GRAVITY_M_S2,
        {"linear_acceleration": "m/s^2", "angular_velocity": "rad/s", "gravity": "m/s^2"},
    )
    same, same_g, _ = lane._to_frozen_model_units(native, gravity, lane.FROZEN_MODEL_UNITS)
    assert np.allclose(converted, same, atol=1e-15)
    assert np.allclose(converted_g, same_g, atol=1e-15)
    assert np.array_equal(si, si_before)
    assert np.array_equal(converted[:, :, 3:], si[:, :, 3:])
    assert receipt["model_units"] == lane.FROZEN_MODEL_UNITS
    with pytest.raises(lane.ConfirmationError, match="units"):
        lane._to_frozen_model_units(
            si, gravity, {**lane.FROZEN_MODEL_UNITS, "angular_velocity": "deg/s"}
        )


def test_scoring_uses_sealed_alignment_and_reports_primary_contrast(tmp_path: Path) -> None:
    config = _config(tmp_path)
    prediction_run = tmp_path / "predictions"
    prediction_run.mkdir()
    participants = np.repeat(np.asarray([f"fresh:{index:03d}" for index in range(38)]), 6)
    labels = np.tile(np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64), 38)
    windows = np.asarray([f"window:{index:04d}" for index in range(labels.size)])
    perfect = _probabilities(labels, perfect=True)
    control = _probabilities(labels, perfect=False)
    prediction_path = prediction_run / "predictions_label_blind.npz"
    prediction_payload: dict[str, Any] = {
        "participant_ids": participants,
        "window_ids": windows,
        "native_gravity_available": np.ones(labels.size, dtype=np.bool_),
        "B6": control,
        "B9": control,
        "T9": perfect,
        "U9": control,
    }
    prediction_payload.update({method: control for method in lane.SECONDARY_METHODS})
    with prediction_path.open("wb") as stream:
        np.savez_compressed(stream, **prediction_payload)
    seal = _write_record(
        prediction_run / "prediction_seal.json",
        {
            "schema_version": "1.0.0",
            "record_kind": "ctgr_native9_label_blind_prediction_seal",
            "status": "sealed_before_label_opening",
            "target_labels_accessed": False,
            "prediction_archive": {
                "path": prediction_path.name,
                "sha256": sha256_file(prediction_path),
            },
            "missing_gravity_exact_B6_fallback_verified": True,
        },
    )
    assert seal["record_sha256"]
    labels_path = tmp_path / "labels.npz"
    with labels_path.open("wb") as stream:
        np.savez_compressed(stream, window_ids=windows, labels=labels)
    result = lane.score(
        config_path=config,
        prediction_run=prediction_run,
        labels_archive=labels_path,
        output=tmp_path / "score",
    )
    primary = result["comparisons"]["primary_T9_minus_B9"]
    assert result["participant_count"] == 38
    assert primary["mean_participant_macro_f1_difference"] > 0.15
    assert primary["harms"] == 0
    assert result["primary_promotion_gate"]["passed"] is True
    assert (tmp_path / "score" / "REPORT.md").is_file()
