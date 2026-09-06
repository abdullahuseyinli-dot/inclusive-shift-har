from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from inclusive_shift_har.artifacts.research_provenance import (
    EXECUTED_RESEARCH_MODULE_PATH,
    FOG_STAR_FULL_PARTICIPANT_ROSTER,
    HAR_PMD_FULL_PARTICIPANT_ROSTER,
    IMU_HAR_IL_AVAILABLE_VALID_PARTICIPANT_ROSTER,
    IMU_HAR_IL_COMPLETE_CORE_PARTICIPANT_ROSTER,
    IMU_HAR_IL_PROVIDER_PARTICIPANT_ROSTER,
    PUBLICATION_LAUNCH_CONTEXT_PROTOCOL_ID,
    PUBLICATION_SOURCE_PROTOCOL_ID,
    _external_evidence_status,
    _imu_har_il_expected_cohort_observed,
    _publication_artifact_contract,
    _runtime_environment,
)
from inclusive_shift_har.data.external_har import (
    BOUNDARY_PROVENANCE_PROTOCOL,
    SOURCE_RECEIPT_METADATA,
)
from inclusive_shift_har.data.participant_partitions import build_participant_partition_plan
from inclusive_shift_har.evaluation.external_statistics import (
    ParticipantMetricInputs,
    seed_evidence,
)
from inclusive_shift_har.evaluation.inference_contracts import (
    OBSERVABLE_CONTEXT_TRAINING_PROTOCOL,
)
from inclusive_shift_har.experiments import external_evidence_validate as validator
from inclusive_shift_har.experiments.external_evidence_validate import (
    _environment_errors,
    _fog_physics_reference_record_errors,
    _hierarchy_baseline_reconstruction_errors,
    _imu_receipt_inventory_hashes,
    _method_contract_errors,
    _metric_evidence_errors,
    _neural_runtime_artifact_errors,
    _receipt_contract_errors,
    _safe_run_artifact_path,
    _stable_validation_payload,
    _stored_split_audit_matches,
    _top_level_report,
    validate_and_record_run_directory,
    validate_run_directory,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_external_evidence_validator_accepts_a_complete_create_only_package(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src" / "example.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    probabilities = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]], dtype=np.float64)
    prediction = run / "predictions.npz"
    np.savez_compressed(
        prediction,
        labels=np.array([0, 1], dtype=np.int64),
        participant_ids=np.array(["p1", "p2"]),
        probability__method=probabilities,
    )
    files = {"src/example.py": sha256_file(source)}
    manifest = {
        "protocol_id": "external-har-session-grid-v3",
        "file_count": 1,
        "files": files,
        "manifest_sha256": canonical_json_sha256(files),
        "captured_at": "2026-09-05T00:00:00+00:00",
    }
    _write_json(
        run / "data_audit.json",
        {
            "dataset": {"window_count": 2},
            "source_input_manifest": manifest,
            "source_receipts": [
                {
                    "computed_sha256": "0" * 64,
                    "received_size_bytes": 1,
                    "declared_digest_verified": True,
                    "raw_local_mirror": False,
                }
            ],
        },
    )
    result: dict[str, object] = {
        "dataset": {"window_count": 2},
        "reports": {"method": {"sample_count": 2}},
        "fold_records": [
            {
                "training_participants": ["p1"],
                "evaluation_participants": ["p2"],
                "outer_evaluation_labels_used_before_predictions_fixed": False,
            }
        ],
        "prediction_artifact": {
            "path": prediction.name,
            "sha256": sha256_file(prediction),
        },
        "source_input_manifest": manifest,
        "claim_policy": {
            "confirmatory_claim_allowed": False,
            "state_of_the_art_claim_allowed": False,
        },
    }
    result["result_payload_sha256_before_serialization"] = canonical_json_sha256(result)
    _write_json(run / "result.json", result)
    validation = validate_run_directory(run, tmp_path)
    assert validation["status"] == "SUPERSEDED_PROTOCOL"
    assert validation["integrity_passed"] is True
    assert validation["publication_evidence_ready"] is False
    assert validation["scientific_contract_passed"] is False
    assert (
        validation["scientific_contract_checks"]["source_manifest_protocol_status"]
        == "LEGACY_READABLE_SUPERSEDED"
    )
    assert validation["run_directory"] == "."
    checks = {item["name"]: item for item in validation["checks"]}
    assert checks["data_audit_present"]["detail"] == "data_audit.json"
    assert checks["prediction_artifact_sha256"]["detail"] == "predictions.npz"
    _write_json(run / "validation.json", validation)
    relocated = tmp_path / "renamed-run"
    shutil.copytree(run, relocated)
    reconstructed = validate_run_directory(relocated, tmp_path)
    assert _stable_validation_payload(reconstructed) == _stable_validation_payload(validation)


def test_validator_recomputes_all_primary_metrics_and_requires_every_seed(tmp_path: Path) -> None:
    data = ParticipantMetricInputs(
        dataset_id="fog_star_v3",
        class_names=("mobility", "sitting", "standing"),
        labels=np.array([0, 1, 2, 0, 1, 2]),
        participant_ids=np.array(["p1", "p1", "p1", "p2", "p2", "p2"]),
    )
    probabilities = {
        seed: {"XGBoost-6ch": np.eye(3)[data.labels] * 0.9 + 0.1 / 3} for seed in (11, 23, 47)
    }
    primary, predictions = seed_evidence(data, probabilities)
    path = tmp_path / "predictions.npz"
    np.savez_compressed(
        path,
        labels=data.labels,
        participant_ids=data.participant_ids,
        **{f"probability__{name}": values for name, values in predictions.items()},  # type: ignore[arg-type]
    )
    result = {"dataset": {"dataset_id": data.dataset_id}, "primary_seed_averaged": primary}
    assert not _metric_evidence_errors(result, path)
    primary["methods"]["XGBoost-6ch"]["mean_participant_macro_f1"] = 0.5
    assert "differ" in _metric_evidence_errors(result, path)[0]
    primary["methods"]["XGBoost-6ch"]["mean_participant_macro_f1"] = 1.0
    incomplete = tmp_path / "incomplete.npz"
    np.savez_compressed(incomplete, labels=data.labels, participant_ids=data.participant_ids)
    assert "reconstruction failed" in _metric_evidence_errors(result, incomplete)[0]


def test_validator_uses_ceil_participant_count_for_bottom_thirty_percent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    participant_scores = [0.1, 0.3, 0.7, 0.9]
    monkeypatch.setattr(
        validator,
        "classification_report",
        lambda *_args, **_kwargs: {
            "primary": {},
            "participants": [{"macro_f1": value} for value in participant_scores],
        },
    )
    report = _top_level_report(
        labels=np.zeros(4, dtype=np.int64),
        probabilities=np.ones((4, 1), dtype=np.float64),
        participants=np.asarray(["p1", "p2", "p3", "p4"]),
        class_names=("activity",),
        include_bottom_tail=True,
    )
    assert report["primary"]["bottom_30_percent_participant_macro_f1"] == pytest.approx(0.2)


def test_runtime_environment_binds_executed_module_to_source_manifest() -> None:
    environment = _runtime_environment()
    executed = environment["executed_research_module"]
    manifest = {
        "protocol_id": PUBLICATION_SOURCE_PROTOCOL_ID,
        "files": {EXECUTED_RESEARCH_MODULE_PATH: executed["sha256"]},
    }
    assert _environment_errors(environment, manifest) == []

    wrong_executable = deepcopy(environment)
    wrong_python = wrong_executable["python_executable"]
    assert isinstance(wrong_python, dict)
    wrong_python["sha256"] = "not-a-sha256"
    assert "runtime Python executable identity is incomplete" in _environment_errors(
        wrong_executable, manifest
    )

    wrong_manifest = deepcopy(manifest)
    wrong_manifest_files = wrong_manifest["files"]
    assert isinstance(wrong_manifest_files, dict)
    wrong_manifest_files[EXECUTED_RESEARCH_MODULE_PATH] = "0" * 64
    assert (
        "executed research module hash differs from the publication source manifest"
        in _environment_errors(environment, wrong_manifest)
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("gate_passed", False),
        ("scientific_input_tuple_exact", False),
        ("class_decisions_exact", False),
        ("probabilities_within_strict_tolerance", False),
        ("probability_absolute_tolerance", 1e-9),
        ("maximum_probability_absolute_difference", 2e-12),
    ),
)
def test_hierarchy_validator_requires_exact_baseline_reconstruction(
    field: str, value: object
) -> None:
    check = {
        "class_decisions_exact": True,
        "probabilities_within_strict_tolerance": True,
        "probability_absolute_tolerance": 1e-12,
        "maximum_probability_absolute_difference": 0.0,
    }
    result: dict[str, object] = {
        "experiment_id": "participant-balanced-hierarchical-posture-forest-v3",
        "seeds": [11, 23, 47],
        "baseline_reconstruction": {
            "gate_passed": True,
            "scientific_input_tuple_exact": True,
            "reference_run": {
                "path_kind": "repository_relative",
                "path": "results/reference",
            },
            "per_seed": {str(seed): deepcopy(check) for seed in (11, 23, 47)},
        },
    }
    assert not _hierarchy_baseline_reconstruction_errors(result)
    baseline = result["baseline_reconstruction"]
    assert isinstance(baseline, dict)
    if field in {"gate_passed", "scientific_input_tuple_exact"}:
        baseline[field] = value
    else:
        per_seed = baseline["per_seed"]
        assert isinstance(per_seed, dict)
        seed_check = per_seed["11"]
        assert isinstance(seed_check, dict)
        seed_check[field] = value
    assert _hierarchy_baseline_reconstruction_errors(result)


@pytest.mark.parametrize("mutation", ("cpu", "missing", "swapped", "hash"))
def test_neural_validator_binds_cuda_and_exact_checkpoint_set(
    tmp_path: Path, mutation: str
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    records: list[dict[str, object]] = []
    for seed in (11, 23, 47):
        for fold in range(5):
            for slug, display, disable_cudnn in (
                ("deepconvlstm", "DeepConvLSTM-6ch", True),
                ("tinyhar", "TinyHAR-6ch", False),
            ):
                path = checkpoint_root / f"seed-{seed}__fold-{fold}__{slug}.pt"
                path.write_bytes(f"{seed}:{fold}:{slug}".encode())
                config = {
                    "model_name": slug,
                    "seed": seed + 101 * fold,
                    "epochs": 40,
                    "batch_size": 128,
                    "learning_rate": 3e-4,
                    "weight_decay": 1e-4,
                    "patience": 8,
                    "minimum_epochs": 8,
                    "mixed_precision": "float16",
                    "disable_cudnn": disable_cudnn,
                }
                records.append(
                    {
                        "seed": seed,
                        "outer_fold": fold,
                        "model": display,
                        "training_config": config,
                        "training_config_sha256": canonical_json_sha256(config),
                        "disable_cudnn": disable_cudnn,
                        "mixed_precision": "float16",
                        "device": "cuda",
                        "device_type": "cuda",
                        "cuda_runtime_version": "12.1",
                        "cuda_device_name": "synthetic GPU",
                        "amp_enabled": True,
                        "autocast_device_type": "cuda",
                        "autocast_dtype": "float16",
                        "gradient_scaler_enabled": True,
                        "checkpoint": {
                            "path": path.relative_to(tmp_path).as_posix(),
                            "sha256": sha256_file(path),
                            "size_bytes": path.stat().st_size,
                        },
                    }
                )
    signal_hash = "a" * 64
    summary = {
        "dataset_id": "fog_star_v3",
        "channel_lane": "derived-gravity-9ch",
        "gravity_preprocessing": {"source": "causal_lowpass_acceleration"},
        "retained_array_hashes": {
            "signals": signal_hash,
            "gravity": "b" * 64,
            "nine_channel_signals": "c" * 64,
        },
    }
    backend = {
        "device_type": "cuda",
        "amp_enabled": True,
        "autocast_device_type": "cuda",
        "autocast_dtype": "float16",
        "gradient_scaler_enabled": True,
        "cuda_runtime_version": "12.1",
        "cuda_device_name": "synthetic GPU",
    }
    result = {
        "experiment_id": "cross-dataset-har-neural-controls-v1",
        "runtime_backend_protocol": "external-neural-cuda-nocudnn-v2",
        "runtime_backend": backend,
        "device": "cuda",
        "seeds": [11, 23, 47],
        "method_suffix": "6ch",
        "input_channels": 6,
        "input_representation": {
            "source_field": "signals",
            "method_suffix": "6ch",
            "input_channels": 6,
            "array_sha256": signal_hash,
            "scored_source_array_sha256": signal_hash,
            "gravity_source": None,
        },
        "training_contract": {
            "epochs": 40,
            "batch_size": 128,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "patience": 8,
            "minimum_epochs": 8,
            "mixed_precision": "float16",
            "outer_folds": 5,
            "models": ["deepconvlstm", "tinyhar"],
        },
        "fold_records": records,
    }
    assert not _neural_runtime_artifact_errors(result, {"dataset": summary}, tmp_path)
    if mutation == "cpu":
        records[0]["device"] = "cpu"
    elif mutation == "missing":
        (checkpoint_root / "seed-11__fold-0__deepconvlstm.pt").unlink()
    elif mutation == "swapped":
        checkpoint = records[0]["checkpoint"]
        assert isinstance(checkpoint, dict)
        checkpoint["path"] = "checkpoints/seed-11__fold-0__tinyhar.pt"
    else:
        checkpoint = records[0]["checkpoint"]
        assert isinstance(checkpoint, dict)
        checkpoint["sha256"] = "0" * 64
    assert _neural_runtime_artifact_errors(result, {"dataset": summary}, tmp_path)


@pytest.mark.parametrize(
    "unsafe",
    ["../predictions.npz", "nested/predictions.npz", "C:/predictions.npz", "\\share\\x"],
)
def test_validator_rejects_prediction_paths_outside_the_direct_run_directory(
    tmp_path: Path, unsafe: str
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    assert _safe_run_artifact_path(run, unsafe) is None
    assert _safe_run_artifact_path(run, "predictions.npz") == run / "predictions.npz"


def test_mutating_validator_rejects_wrong_source_root_before_writing(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "result.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="executed research module"):
        validate_and_record_run_directory(run, tmp_path)
    assert not (run / "split_audit.json").exists()
    assert not (run / "validation.json").exists()


@pytest.mark.parametrize("mutation", ["assignment_protocol", "warnings"])
def test_stored_split_audit_binds_all_deterministic_fields_even_after_rehash(
    mutation: str,
) -> None:
    reconstructed: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "PASS_RECORDED_RESULT",
        "valid": True,
        "assignment_protocol": {
            "protocol_id": "external-har-prewindow-partitions-v1",
            "outer_fold_count": 5,
        },
        "warnings": ["scope is participant-exclusive development evidence"],
        "created_at": "2026-09-05T00:00:00+00:00",
    }
    reconstructed["record_sha256"] = canonical_json_sha256(reconstructed)
    stored = deepcopy(reconstructed)
    assert _stored_split_audit_matches(stored, reconstructed)
    if mutation == "assignment_protocol":
        stored["assignment_protocol"] = {
            "protocol_id": "external-har-prewindow-partitions-v1",
            "outer_fold_count": 10,
        }
    else:
        stored["warnings"] = []
    stored.pop("record_sha256")
    stored["record_sha256"] = canonical_json_sha256(stored)
    assert not _stored_split_audit_matches(stored, reconstructed)


@pytest.mark.parametrize("invalid_time", [None, "2026-09-05T00:00:00"])
def test_stored_split_audit_requires_timezone_aware_creation_time(invalid_time: object) -> None:
    reconstructed: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "PASS_RECORDED_RESULT",
        "valid": True,
        "created_at": "2026-09-05T00:00:00+00:00",
    }
    reconstructed["record_sha256"] = canonical_json_sha256(reconstructed)
    stored = deepcopy(reconstructed)
    stored["created_at"] = invalid_time
    stored.pop("record_sha256")
    stored["record_sha256"] = canonical_json_sha256(stored)
    assert not _stored_split_audit_matches(stored, reconstructed)


def test_stored_split_audit_accepts_only_coherent_legacy_absolute_paths() -> None:
    reconstructed: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "PASS_RECORDED_RESULT",
        "valid": True,
        "inputs": {
            "run_directory": ".",
            "result": {"path": "result.json", "sha256": "1" * 64},
            "predictions": {"path": "predictions.npz", "sha256": "2" * 64},
        },
        "warnings": ["scientific scope retained"],
        "created_at": "2026-09-05T00:00:00+00:00",
    }
    reconstructed["record_sha256"] = canonical_json_sha256(reconstructed)
    legacy = deepcopy(reconstructed)
    legacy_inputs = legacy["inputs"]
    assert isinstance(legacy_inputs, dict)
    legacy_inputs["run_directory"] = "C:/archive/original"
    legacy_result = legacy_inputs["result"]
    legacy_predictions = legacy_inputs["predictions"]
    assert isinstance(legacy_result, dict)
    assert isinstance(legacy_predictions, dict)
    legacy_result["path"] = "C:/archive/original/result.json"
    legacy_predictions["path"] = "C:/archive/original/predictions.npz"
    legacy.pop("record_sha256")
    legacy["record_sha256"] = canonical_json_sha256(legacy)
    assert _stored_split_audit_matches(legacy, reconstructed)

    tampered_path = deepcopy(legacy)
    tampered_inputs = tampered_path["inputs"]
    assert isinstance(tampered_inputs, dict)
    tampered_predictions = tampered_inputs["predictions"]
    assert isinstance(tampered_predictions, dict)
    tampered_predictions["path"] = "C:/other/predictions.npz"
    tampered_path.pop("record_sha256")
    tampered_path["record_sha256"] = canonical_json_sha256(tampered_path)
    assert not _stored_split_audit_matches(tampered_path, reconstructed)


def test_stored_validation_normalizes_only_legacy_presentation_paths() -> None:
    current: dict[str, object] = {
        "run_directory": ".",
        "status": "VALIDATED",
        "integrity_passed": True,
        "checks": [
            {"name": "data_audit_present", "passed": True, "detail": "data_audit.json"},
            {
                "name": "prediction_artifact_sha256",
                "passed": True,
                "detail": "predictions.npz",
            },
        ],
        "participant_split_audit": {
            "created_at": "2026-09-05T00:00:00+00:00",
            "record_sha256": "ignored-after-independent-self-hash-check",
            "inputs": {
                "run_directory": ".",
                "result": {"path": "result.json", "sha256": "1" * 64},
                "predictions": {"path": "predictions.npz", "sha256": "2" * 64},
            },
            "warnings": ["scientific limitation retained"],
        },
    }
    legacy = deepcopy(current)
    legacy["run_directory"] = "C:/archive/original"
    legacy_checks = legacy["checks"]
    assert isinstance(legacy_checks, list)
    assert all(isinstance(item, dict) for item in legacy_checks)
    legacy_checks[0]["detail"] = "C:/archive/original/data_audit.json"
    legacy_checks[1]["detail"] = "C:/archive/original/predictions.npz"
    legacy_split = legacy["participant_split_audit"]
    assert isinstance(legacy_split, dict)
    legacy_split_inputs = legacy_split["inputs"]
    assert isinstance(legacy_split_inputs, dict)
    legacy_split_inputs["run_directory"] = "C:/archive/original"
    legacy_split_result = legacy_split_inputs["result"]
    legacy_split_predictions = legacy_split_inputs["predictions"]
    assert isinstance(legacy_split_result, dict)
    assert isinstance(legacy_split_predictions, dict)
    legacy_split_result["path"] = "C:/archive/original/result.json"
    legacy_split_predictions["path"] = "C:/archive/original/predictions.npz"
    assert _stable_validation_payload(legacy) == _stable_validation_payload(current)

    scientifically_tampered = deepcopy(legacy)
    scientifically_tampered["integrity_passed"] = False
    assert _stable_validation_payload(scientifically_tampered) != _stable_validation_payload(
        current
    )

    incoherent_path = deepcopy(legacy)
    incoherent_checks = incoherent_path["checks"]
    assert isinstance(incoherent_checks, list)
    assert isinstance(incoherent_checks[1], dict)
    incoherent_checks[1]["detail"] = "C:/other/predictions.npz"
    assert _stable_validation_payload(incoherent_path) != _stable_validation_payload(current)


@pytest.mark.parametrize("mutation", ["validated_at", "embedded_split_self_hash"])
def test_mutating_validator_rejects_invalid_stored_volatile_integrity_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "result.json").write_text("{}\n", encoding="utf-8")
    (run / "split_audit.json").write_text("{}\n", encoding="utf-8")
    split: dict[str, object] = {
        "created_at": "2026-09-05T00:00:00+00:00",
        "status": "PASS_RECORDED_RESULT",
        "valid": True,
    }
    split["record_sha256"] = canonical_json_sha256(split)
    reconstructed: dict[str, object] = {
        "validated_at": "2026-09-05T00:00:01+00:00",
        "status": "VALIDATED",
        "integrity_passed": True,
        "publication_evidence_ready": True,
        "diagnostic_scope_reasons": [],
        "participant_split_audit": split,
    }
    stored = deepcopy(reconstructed)
    if mutation == "validated_at":
        stored["validated_at"] = "2026-09-05T00:00:01"
    else:
        embedded = stored["participant_split_audit"]
        assert isinstance(embedded, dict)
        embedded["record_sha256"] = "0" * 64
    _write_json(run / "validation.json", stored)
    monkeypatch.setattr(validator, "_assert_executed_repository_root", lambda _root: {})
    monkeypatch.setattr(validator, "validate_run_directory", lambda *_args: reconstructed)
    with pytest.raises(ValueError, match="volatile integrity fields"):
        validate_and_record_run_directory(run, tmp_path)


def test_dangling_terminal_and_audit_entries_are_present_but_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    both = tmp_path / "both-terminals"
    both.mkdir()
    (both / "result.json").write_text("{}\n", encoding="utf-8")
    try:
        (both / "failure.json").symlink_to("missing-failure.json")
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")
    validation = validate_run_directory(both, tmp_path)
    terminal_check = next(
        check for check in validation["checks"] if check["name"] == "exactly_one_terminal_artifact"
    )
    assert terminal_check["passed"] is False
    assert terminal_check["detail"] == {"result": True, "failure": True}
    monkeypatch.setattr(validator, "_assert_executed_repository_root", lambda _root: {})
    with pytest.raises(ValueError, match="exactly one terminal"):
        validate_and_record_run_directory(both, tmp_path)
    assert not (both / "split_audit.json").exists()
    assert not (both / "validation.json").exists()

    dangling_audit = tmp_path / "dangling-audit"
    dangling_audit.mkdir()
    (dangling_audit / "failure.json").write_text("{}\n", encoding="utf-8")
    (dangling_audit / "data_audit.json").symlink_to("missing-audit.json")
    audit_validation = validate_run_directory(dangling_audit, tmp_path)
    assert not any(
        check["name"] == "data_audit_not_applicable_before_dataset_acquisition"
        for check in audit_validation["checks"]
    )
    audit_check = next(
        check for check in audit_validation["checks"] if check["name"] == "data_audit_present"
    )
    assert audit_check["passed"] is False


@pytest.mark.parametrize("mutation", ["locator", "sha256", "accessed_at", "license"])
def test_v4_fog_receipt_is_bound_to_pinned_object_and_frozen_metadata(
    mutation: str,
) -> None:
    receipt: dict[str, object] = {
        "dataset_id": "fog_star_v3",
        "locator": "https://zenodo.org/api/records/17838806/files/sensor_data.csv/content",
        "member": None,
        "declared_size_bytes": 119_629_580,
        "received_size_bytes": 119_629_580,
        "computed_sha256": ("888875133fef386affddd0a5864f122d527d8d7bc588a69905cc0871bef53477"),
        "declared_digest_algorithm": "md5",
        "declared_digest": "952a37ab147da35e6d4e7a1e9bac44cb",
        "computed_declared_digest": "952a37ab147da35e6d4e7a1e9bac44cb",
        "declared_digest_verified": True,
        "raw_local_mirror": False,
        "accessed_at_utc": "2026-09-05T00:00:00+00:00",
        **SOURCE_RECEIPT_METADATA["fog_star_v3"],
    }
    audit = {
        "dataset": {"dataset_id": "fog_star_v3", "receipt_count": 1},
        "source_receipts": [receipt],
    }
    assert _receipt_contract_errors(audit, publication_v4=True) == []
    if mutation == "locator":
        receipt["locator"] = "https://example.invalid/sensor_data.csv"
    elif mutation == "sha256":
        receipt["computed_sha256"] = "f" * 64
    elif mutation == "accessed_at":
        receipt["accessed_at_utc"] = "2026-09-05T00:00:00"
    else:
        receipt["dataset_license"] = "Apache-2.0"
    assert _receipt_contract_errors(audit, publication_v4=True)


def test_malformed_json_and_prediction_archive_fail_closed_without_crashing(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "data_audit.json").write_text(
        '{"source_receipts": [], "source_receipts": []}', encoding="utf-8"
    )
    (malformed / "failure.json").write_text('{"status":"FAILED_PRESERVED"}', encoding="utf-8")
    malformed_validation = validate_run_directory(malformed, tmp_path)
    assert malformed_validation["status"] == "INVALID_EVIDENCE_PACKAGE"
    assert malformed_validation["integrity_passed"] is False

    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    prediction = corrupt / "predictions.npz"
    prediction.write_bytes(b"not a NumPy archive")
    manifest = {
        "protocol_id": "external-har-session-grid-v3",
        "file_count": 1,
        "files": {"src/example.py": "0" * 64},
        "manifest_sha256": canonical_json_sha256({"src/example.py": "0" * 64}),
        "captured_at": "2026-09-05T00:00:00+00:00",
    }
    _write_json(
        corrupt / "data_audit.json",
        {
            "dataset": {"window_count": 1},
            "source_input_manifest": manifest,
            "source_receipts": [
                {
                    "computed_sha256": "0" * 64,
                    "received_size_bytes": 1,
                    "raw_local_mirror": False,
                }
            ],
        },
    )
    result: dict[str, object] = {
        "dataset": {"window_count": 1},
        "reports": {"method": {}},
        "source_input_manifest": manifest,
        "prediction_artifact": {
            "path": "predictions.npz",
            "sha256": sha256_file(prediction),
        },
        "claim_policy": {
            "confirmatory_claim_allowed": False,
            "state_of_the_art_claim_allowed": False,
        },
    }
    result["result_payload_sha256_before_serialization"] = canonical_json_sha256(result)
    _write_json(corrupt / "result.json", result)
    corrupt_validation = validate_run_directory(corrupt, tmp_path)
    assert corrupt_validation["status"] == "INVALID_EVIDENCE_PACKAGE"
    assert corrupt_validation["integrity_passed"] is False


def test_pre_writer_failure_is_validated_without_fabricating_a_dataset_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "failed-launch"
    run.mkdir()
    environment = _runtime_environment()
    manifest = {
        "protocol_id": PUBLICATION_SOURCE_PROTOCOL_ID,
        "files": {
            "tracked.txt": "0" * 64,
            EXECUTED_RESEARCH_MODULE_PATH: environment["executed_research_module"]["sha256"],
        },
    }
    launch = {"commit": "0" * 40, "worktree_dirty": False, "status_entries": []}
    allowed_root = {"kind": "external_absolute", "path": str(run.resolve())}
    context = {
        "schema_version": "1.0.0",
        "protocol_id": PUBLICATION_LAUNCH_CONTEXT_PROTOCOL_ID,
        "allowed_create_only_output_root": allowed_root,
        "git_at_launch": launch,
        "source_input_manifest": manifest,
    }
    failure: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "FAILED_PRESERVED",
        "failure_scope": "pre_writer_configuration_or_dataset_acquisition",
        "stage": "dataset_acquisition",
        "started_at": "2026-09-05T00:00:00+00:00",
        "failed_at": "2026-09-05T00:00:01+00:00",
        "exception_type": "RuntimeError",
        "exception_message": "provider unavailable",
        "traceback": "synthetic traceback",
        "git_at_launch": launch,
        "source_input_manifest": manifest,
        "publication_launch_context": {
            "protocol_id": PUBLICATION_LAUNCH_CONTEXT_PROTOCOL_ID,
            "allowed_create_only_output_root": allowed_root,
            "record_sha256": canonical_json_sha256(context),
        },
        "environment": environment,
    }
    failure["failure_payload_sha256_before_serialization"] = canonical_json_sha256(failure)
    _write_json(run / "failure.json", failure)
    monkeypatch.setattr(validator, "_manifest_commit_errors", lambda *_args: [])
    monkeypatch.setattr(validator, "_publication_manifest_errors", lambda *_args: [])
    monkeypatch.setattr(validator, "_assert_executed_repository_root", lambda _root: {})

    validation = validate_and_record_run_directory(run, tmp_path)
    assert validation["status"] == "FAILED_RUN_PRESERVED"
    assert validation["integrity_passed"] is True
    assert (run / "validation.json").is_file()
    assert not (run / "data_audit.json").exists()
    assert not (run / "split_audit.json").exists()

    relocated = tmp_path / "failed-launch-relocated"
    shutil.copytree(run, relocated)
    relocated_validation = validate_and_record_run_directory(relocated, tmp_path)
    assert relocated_validation["status"] == "FAILED_RUN_PRESERVED"

    binding = failure["publication_launch_context"]
    assert isinstance(binding, dict)
    binding["record_sha256"] = "f" * 64
    failure.pop("failure_payload_sha256_before_serialization")
    failure["failure_payload_sha256_before_serialization"] = canonical_json_sha256(failure)
    _write_json(run / "failure.json", failure)
    assert validate_run_directory(run, tmp_path)["status"] == "INVALID_EVIDENCE_PACKAGE"


def _valid_imu_cohort_summary(selection_policy: str) -> dict[str, object]:
    provider = list(IMU_HAR_IL_PROVIDER_PARTICIPANT_ROSTER)
    incomplete = [
        f"imuharil:P_{participant:02d}"
        for participant in (
            1,
            2,
            4,
            6,
            7,
            9,
            13,
            14,
            15,
            16,
            27,
            28,
            29,
            30,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            43,
            44,
            45,
            46,
            49,
            50,
        )
    ]
    if selection_policy == "complete_requested_core":
        protocol_id = "imu-har-il-complete-requested-core-v1"
        retained = list(IMU_HAR_IL_COMPLETE_CORE_PARTICIPANT_ROSTER)
        trial_count = 228
        selected_files = 264
        inventory_sha256 = "060748891b6869712c7323ab4b0a754b2d57703b6cccb2ef574434676e70c7cc"
        payload_sha256 = "cba6f7bb1dcaea8fc60a319d73e9123aeb478f9f09752c462a6bd94086b4d78c"
        quality = ["imuharil:P_18", "imuharil:P_19", "imuharil:P_22"]
        quarantined = 30
        exclusion_records = 31
        absent = sorted(set(provider) - set(retained))
    else:
        protocol_id = "imu-har-il-available-trials-v1"
        retained = list(IMU_HAR_IL_AVAILABLE_VALID_PARTICIPANT_ROSTER)
        trial_count = 457
        selected_files = 490
        inventory_sha256 = "b7aa481380224e756e39ad28156d94961d49d659d106bf69f8cac358b0271385"
        payload_sha256 = "ca81875cd3bcab424aa3821b7723c8b6f42d170ea8931a507dad47a16e1b1c25"
        quality = [
            "imuharil:P_04",
            "imuharil:P_18",
            "imuharil:P_19",
            "imuharil:P_22",
        ]
        quarantined = 33
        exclusion_records = 32
        absent = ["imuharil:P_18", "imuharil:P_19", "imuharil:P_46"]
    plan = build_participant_partition_plan(
        "imu_har_il_v1",
        provider,
        roster_basis="provider inventory frozen before any window extraction",
    ).audit()
    cohort = {
        "protocol_id": protocol_id,
        "inventory_protocol_id": "imu-har-il-fixed-inventory-v1",
        "selection_policy": selection_policy,
        "provider_reported_participant_count": 50,
        "requested_repetition_limit": 4,
        "requested_repetitions": [
            "Repetition_1",
            "Repetition_2",
            "Repetition_3",
            "Repetition_4",
        ],
        "requested_activities": ["Walk", "Sit", "Stand"],
        "requested_participant_count": 50,
        "requested_participants": provider,
        "requested_trial_count": 600,
        "inventory_selected_source_file_count": selected_files,
        "source_inventory_sha256": inventory_sha256,
        "source_payload_inventory_sha256": payload_sha256,
        "inventory_incomplete_participant_count": 28,
        "inventory_incomplete_participants": incomplete,
        "missing_requested_trial_count": 110,
        "missing_requested_trial_folders_sha256": (
            "651857292b0e4a66a52a7d124016c8871c1a96aa010c71d05d476c16d0f50719"
        ),
        "data_quality_affected_participant_count": len(quality),
        "data_quality_affected_participants": quality,
        "quarantined_trial_count": quarantined,
        "participant_exclusion_record_count": exclusion_records,
        "retained_source_file_count": trial_count,
        "retained_participant_count": len(retained),
        "retained_participants": retained,
        "retained_trial_count": trial_count,
        "participants_with_no_retained_windows": absent,
        "independent_unit": "participant; repetitions do not increase independent N",
        "complete_case_or_repetition1_before_after_comparison_allowed": False,
    }
    return {
        "dataset_id": "imu_har_il_v1",
        "participant_count": len(retained),
        "trial_count": trial_count,
        "receipt_count": selected_files,
        "participant_partition_plan": plan,
        "participant_partition_observation": {
            "planned_participant_count": 50,
            "observed_window_participant_count": len(retained),
            "participants_without_retained_windows": absent,
        },
        "cohort_audit": cohort,
        "boundary_provenance": {
            "protocol_id": BOUNDARY_PROVENANCE_PROTOCOL,
            "repository_signal_grid_annotation_independent": False,
            "provider_upstream_annotation_conditioned": True,
            "zero_lookahead_streaming_valid": False,
        },
    }


def test_imu_inventory_hashes_are_independently_reconstructed_from_receipts() -> None:
    receipt = {
        "dataset_id": "imu_har_il_v1",
        "locator": "https://data.csiro.au/dap/ws/v2/collections/74700/data/123",
        "member": "HAR_IMU_IL/P_01/Repetition_1/Walk/Body-WT.csv",
        "declared_size_bytes": 456,
        "received_size_bytes": 456,
        "computed_sha256": "a" * 64,
    }
    source = [
        {
            "participant": "P_01",
            "repetition": "Repetition_1",
            "activity": "Walk",
            "file_id": 123,
            "filename": "HAR_IMU_IL/P_01/Repetition_1/Walk/Body-WT.csv",
            "file_size": 456,
            "locator": "https://data.csiro.au/dap/ws/v2/collections/74700/data/123",
        }
    ]
    payload = [
        {
            "locator": receipt["locator"],
            "member": receipt["member"],
            "declared_size_bytes": 456,
            "received_size_bytes": 456,
            "computed_sha256": "a" * 64,
        }
    ]
    audit = {
        "dataset": {"dataset_id": "imu_har_il_v1"},
        "source_receipts": [receipt],
    }
    assert _imu_receipt_inventory_hashes(audit) == (
        canonical_json_sha256(source),
        canonical_json_sha256(payload),
    )
    receipt["member"] = "HAR_IMU_IL/P_01/Repetition_1/Stairs/Body-WT.csv"
    assert _imu_receipt_inventory_hashes(audit) is None


@pytest.mark.parametrize(
    ("selection_policy", "participant_count"),
    (("complete_requested_core", 19), ("available_valid_trials", 47)),
)
def test_imu_scientific_contract_requires_exact_lane_specific_inventory(
    selection_policy: str, participant_count: int
) -> None:
    summary = _valid_imu_cohort_summary(selection_policy)
    assert summary["participant_count"] == participant_count
    assert _imu_har_il_expected_cohort_observed(summary, selection_policy=selection_policy)
    plan = summary["participant_partition_plan"]
    assert isinstance(plan, dict)
    result = {
        "seeds": [11, 23, 47],
        "primary_seed_averaged": {},
        "fold_records": [{"participant_partition_plan_sha256": plan["plan_sha256"]}],
        "source_input_manifest": {"protocol_id": PUBLICATION_SOURCE_PROTOCOL_ID},
    }
    message = "exact frozen 50-person provider inventory"
    assert not any(
        message in error for error in _method_contract_errors(result, {"dataset": summary})
    )

    cohort = summary["cohort_audit"]
    assert isinstance(cohort, dict)
    cohort["source_payload_inventory_sha256"] = "0" * 64
    assert not _imu_har_il_expected_cohort_observed(summary, selection_policy=selection_policy)
    assert any(message in error for error in _method_contract_errors(result, {"dataset": summary}))


def test_imu_cohort_guard_rejects_repetition_inflation_and_lane_substitution() -> None:
    summary = _valid_imu_cohort_summary("available_valid_trials")
    cohort = summary["cohort_audit"]
    assert isinstance(cohort, dict)
    cohort["independent_unit"] = "participant-repetition"
    cohort["retained_participant_count"] = 188
    assert not _imu_har_il_expected_cohort_observed(summary)
    assert not _imu_har_il_expected_cohort_observed(
        _valid_imu_cohort_summary("complete_requested_core"),
        selection_policy="available_valid_trials",
    )


def test_post_audit_failure_uses_bound_audit_summary_and_retains_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "failed-after-audit"
    run.mkdir()
    launch = {"commit": "0" * 40, "worktree_dirty": False, "status_entries": []}
    environment = _runtime_environment()
    manifest = {
        "protocol_id": PUBLICATION_SOURCE_PROTOCOL_ID,
        "files": {
            "tracked.txt": "0" * 64,
            EXECUTED_RESEARCH_MODULE_PATH: environment["executed_research_module"]["sha256"],
        },
        "publication_protocol": {"path": "protocol.yaml", "sha256": "1" * 64},
        "evidence_role_ledger": {"path": "roles.json", "sha256": "2" * 64},
        "supersession_ledger": {"path": "supersession.json", "sha256": "3" * 64},
    }
    allowed_root = {"kind": "external_absolute", "path": str(run.resolve())}
    context = {
        "schema_version": "1.0.0",
        "protocol_id": PUBLICATION_LAUNCH_CONTEXT_PROTOCOL_ID,
        "allowed_create_only_output_root": allowed_root,
        "git_at_launch": launch,
        "source_input_manifest": manifest,
    }
    binding = {
        "protocol_id": PUBLICATION_LAUNCH_CONTEXT_PROTOCOL_ID,
        "allowed_create_only_output_root": allowed_root,
        "record_sha256": canonical_json_sha256(context),
    }
    summary = {
        "dataset_id": "fog_star_v3",
        "receipt_count": 1,
    }
    receipt = {
        "dataset_id": "fog_star_v3",
        "locator": "https://zenodo.org/api/records/17838806/files/sensor_data.csv/content",
        "member": None,
        "declared_size_bytes": 119_629_580,
        "received_size_bytes": 119_629_580,
        "computed_sha256": ("888875133fef386affddd0a5864f122d527d8d7bc588a69905cc0871bef53477"),
        "declared_digest_algorithm": "md5",
        "declared_digest": "952a37ab147da35e6d4e7a1e9bac44cb",
        "computed_declared_digest": "952a37ab147da35e6d4e7a1e9bac44cb",
        "declared_digest_verified": True,
        "raw_local_mirror": False,
        "accessed_at_utc": "2026-09-05T00:00:00+00:00",
        **SOURCE_RECEIPT_METADATA["fog_star_v3"],
    }
    evidence_status = _external_evidence_status(summary)
    audit: dict[str, object] = {
        "schema_version": "1.0.0",
        "dataset": summary,
        "source_receipts": [receipt],
        "git_at_launch": launch,
        "source_input_manifest": manifest,
        "publication_launch_context": binding,
        "artifact_evidence_status": evidence_status,
    }
    audit["record_sha256"] = canonical_json_sha256(audit)
    _write_json(run / "data_audit.json", audit)
    audit_reference = {
        "path": "data_audit.json",
        "sha256": sha256_file(run / "data_audit.json"),
        "record_sha256": audit["record_sha256"],
    }
    failure: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "FAILED_PRESERVED",
        "started_at": "2026-09-05T00:00:00+00:00",
        "failed_at": "2026-09-05T00:00:01+00:00",
        "exception_type": "RuntimeError",
        "exception_message": "synthetic post-audit failure",
        "traceback": "synthetic traceback",
        "git_at_launch": launch,
        "source_input_manifest": manifest,
        "publication_launch_context": binding,
        "environment": environment,
        "artifact_evidence_status": evidence_status,
        "data_audit_artifact": audit_reference,
        "artifact_contract": _publication_artifact_contract(manifest, audit_reference),
    }
    # The terminal envelope intentionally does not duplicate ``dataset``: its
    # exact self-hashed audit is the authoritative post-acquisition summary.
    failure["failure_payload_sha256_before_serialization"] = canonical_json_sha256(failure)
    _write_json(run / "failure.json", failure)
    monkeypatch.setattr(validator, "_manifest_commit_errors", lambda *_args: [])
    monkeypatch.setattr(validator, "_publication_manifest_errors", lambda *_args: [])
    monkeypatch.setattr(validator, "_assert_executed_repository_root", lambda _root: {})

    validation = validate_and_record_run_directory(run, tmp_path)
    assert validation["status"] == "FAILED_RUN_PRESERVED"
    assert validation["integrity_passed"] is True
    assert validation["publication_evidence_ready"] is False
    assert (run / "validation.json").is_file()
    assert not (run / "split_audit.json").exists()

    relocated = tmp_path / "failed-after-audit-relocated"
    shutil.copytree(run, relocated)
    relocated_validation = validate_and_record_run_directory(relocated, tmp_path)
    assert relocated_validation["status"] == "FAILED_RUN_PRESERVED"


@pytest.mark.parametrize("participant_count", [12, 120])
def test_har_pmd_scientific_contract_requires_exact_full_planned_cohort(
    participant_count: int,
) -> None:
    roster = list(HAR_PMD_FULL_PARTICIPANT_ROSTER)
    plan = build_participant_partition_plan(
        "har_pmd_v1",
        roster,
        roster_basis="synthetic provider roster frozen before window extraction",
    ).audit()
    summary = {
        "dataset_id": "har_pmd_v1",
        "participant_count": participant_count,
        "participant_partition_plan": plan,
        "participant_partition_observation": {
            "planned_participant_count": 120,
            "observed_window_participant_count": participant_count,
            "participants_without_retained_windows": roster[participant_count:],
        },
        "boundary_provenance": {
            "protocol_id": BOUNDARY_PROVENANCE_PROTOCOL,
            "repository_signal_grid_annotation_independent": True,
            "provider_upstream_annotation_conditioned": False,
            "zero_lookahead_streaming_valid": False,
        },
    }
    result = {
        "seeds": [11, 23, 47],
        "primary_seed_averaged": {},
        "fold_records": [{"participant_partition_plan_sha256": plan["plan_sha256"]}],
        "source_input_manifest": {"protocol_id": PUBLICATION_SOURCE_PROTOCOL_ID},
    }
    errors = _method_contract_errors(result, {"dataset": summary})
    full_cohort_error = any("all 120 planned participants" in error for error in errors)
    assert full_cohort_error is (participant_count == 12)


@pytest.mark.parametrize("participant_count", [21, 22])
def test_fog_scientific_contract_requires_exact_full_scored_cohort(
    participant_count: int,
) -> None:
    roster = list(FOG_STAR_FULL_PARTICIPANT_ROSTER)
    plan = build_participant_partition_plan(
        "fog_star_v3",
        roster,
        roster_basis="pinned provider subjectID roster frozen before window extraction",
    ).audit()
    summary = {
        "dataset_id": "fog_star_v3",
        "participant_count": participant_count,
        "participant_partition_plan": plan,
        "participant_partition_observation": {
            "planned_participant_count": 22,
            "observed_window_participant_count": participant_count,
            "participants_without_retained_windows": roster[participant_count:],
        },
        "boundary_provenance": {
            "protocol_id": BOUNDARY_PROVENANCE_PROTOCOL,
            "repository_signal_grid_annotation_independent": True,
            "provider_upstream_annotation_conditioned": False,
            "zero_lookahead_streaming_valid": False,
        },
    }
    result = {
        "seeds": [11, 23, 47],
        "primary_seed_averaged": {},
        "fold_records": [{"participant_partition_plan_sha256": plan["plan_sha256"]}],
        "source_input_manifest": {"protocol_id": PUBLICATION_SOURCE_PROTOCOL_ID},
    }
    errors = _method_contract_errors(result, {"dataset": summary})
    full_cohort_error = any("exact 22-person provider roster" in error for error in errors)
    assert full_cohort_error is (participant_count == 21)


def test_fog_participant_context_method_requires_frozen_training_population_protocol() -> None:
    result: dict[str, object] = {
        "seeds": [11, 23, 47],
        "primary_seed_averaged": {"methods": {"HERA-DG-full": {}}},
    }
    audit = {"dataset": {"dataset_id": "fog_star_v3"}}
    message = "frozen observable-training population"
    assert any(message in error for error in _method_contract_errors(result, audit))
    result["observable_context_training_protocol"] = OBSERVABLE_CONTEXT_TRAINING_PROTOCOL
    assert not any(message in error for error in _method_contract_errors(result, audit))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "physics_reference_fit_population",
            "annotation_eligible_outer_training_windows",
            "every observable outer-training candidate",
        ),
        ("physics_reference_fit_candidate_window_count", 8, "every observable"),
        ("physics_reference_fit_signal_valid_window_count", 0, "signal-valid fit count"),
        ("physics_reference_fit_signal_valid_window_count", 11, "signal-valid fit count"),
        ("physics_reference_threshold", float("nan"), "threshold is absent or invalid"),
        ("physics_reference_threshold", 0.0, "threshold is absent or invalid"),
    ),
)
def test_fog_physics_reference_evidence_binds_complete_candidate_fit_population(
    field: str, value: object, message: str
) -> None:
    record: dict[str, object] = {
        "physics_reference_fit_population": "all_observable_outer_training_candidates",
        "physics_reference_fit_candidate_window_count": 10,
        "physics_reference_fit_signal_valid_window_count": 9,
        "physics_reference_threshold": 0.75,
    }
    assert _fog_physics_reference_record_errors(record, 10) == []
    record[field] = value
    assert any(message in error for error in _fog_physics_reference_record_errors(record, 10))


@pytest.mark.parametrize(
    "unsafe_name",
    (
        "../outside.json",
        "nested/output.json",
        r"..\outside.json",
        "C:outside.json",
        "..",
        "result.json",
        "failure.json",
        "data_audit.json",
        "split_audit.json",
        "validation-alternate.json",
    ),
)
def test_validator_cli_rejects_escaping_output_name_before_any_run_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_name: str,
) -> None:
    run = tmp_path / "run"
    monkeypatch.setattr(validator, "_assert_executed_repository_root", lambda _root: {})
    with pytest.raises(ValueError, match="direct-child filename"):
        validator.main(
            [
                str(run),
                "--repository-root",
                str(tmp_path),
                "--output-name",
                unsafe_name,
            ]
        )
    assert not run.exists()
    assert not (tmp_path / "outside.json").exists()
