from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import inclusive_shift_har.evaluation.few_person_statistics as statistics_module
import inclusive_shift_har.experiments.few_person as few_person_module
from inclusive_shift_har.evaluation.few_person_statistics import (
    EXPECTED_CELL_COUNT,
    EXPECTED_FOLDS,
    EXPECTED_K_VALUES,
    EXPECTED_MODEL_IDS,
    EXPECTED_SEEDS,
    EXPECTED_TARGET_SUBJECTS,
    CellKey,
    FewPersonStatisticsError,
    _comparisons,
    _Completed,
    _expected_adaptation_seed,
    _load_plan,
    _seed_averaged_curves,
    _validate_completed_record,
    aggregate_few_person_statistics,
    validate_few_person_progress,
)
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    sha256_file,
)
from inclusive_shift_har.protocols.few_person import (
    FewPersonProtocolError,
    build_few_person_manifest,
    write_few_person_manifest_new,
)
from inclusive_shift_har.protocols.few_person_v1_1 import (
    FEW_PERSON_V1_1_EVIDENCE_STATUS,
    FEW_PERSON_V1_1_PROTOCOL_ID,
    FUNCTIONAL_CORE_CLASS_ORDER,
    build_few_person_v1_1_manifest,
    validate_few_person_v1_1_manifest_assignments,
    write_few_person_v1_1_manifest_new,
)
from tests.test_few_person_protocol import materialize_few_person_inputs


def _rewrite_hashed(path: Path, payload: dict[str, Any], field: str) -> None:
    payload.pop(field, None)
    payload[field] = canonical_json_sha256(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _window_ids(subjects: list[str] | tuple[str, ...], *, functional_only: bool) -> list[str]:
    stop = 3 if functional_only else 4
    return sorted(
        f"subject-{subject}-window-{ordinal}" for subject in subjects for ordinal in range(stop)
    )


def _materialize_v1_1_plan(
    tmp_path: Path, repository_root: Path
) -> tuple[Path, Path, dict[str, Any]]:
    v1_config = repository_root / "configs/protocols/few_person_inclusion_curve_v1.yaml"
    v1_1_config = repository_root / "configs/protocols/few_person_inclusion_curve_v1_1.yaml"
    split_path, receipt_path, zero_path, inventory_path = materialize_few_person_inputs(
        tmp_path, v1_config
    )

    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["protocol"] = {"protocol_id": "inclusivehar-released-block-v1.2"}
    functional_schema: dict[str, Any] = {
        "track": "functional_core",
        "class_count": len(FUNCTIONAL_CORE_CLASS_ORDER),
        "class_order": list(FUNCTIONAL_CORE_CLASS_ORDER),
        "index_by_class": {name: index for index, name in enumerate(FUNCTIONAL_CORE_CLASS_ORDER)},
    }
    functional_schema["class_schema_sha256"] = canonical_json_sha256(functional_schema)
    split["ontology"] = {
        "config_sha256": "f" * 64,
        "runnable_track_schemas": {"functional_core": functional_schema},
    }
    raw_path = tmp_path / "data" / "raw" / "synthetic-inclusivehar.csv"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text("synthetic immutable raw fixture\n", encoding="utf-8")
    split["source_artifact_sha256"] = sha256_file(raw_path)
    functional_labels = FUNCTIONAL_CORE_CLASS_ORDER
    split["windows"] = [
        {
            "window_id": f"subject-{subject}-window-{ordinal}",
            "subject_id": subject,
            "partition": "target_sealed",
            "canonical_labels": (
                {"functional_core": functional_labels[ordinal]}
                if ordinal < 3
                else {"inclusive_native": "jogging"}
            ),
        }
        for subject in sorted(EXPECTED_TARGET_SUBJECTS, key=int)
        for ordinal in range(4)
    ]
    for fold in split["few_person_outer_folds"]:
        for scenario in fold["scenarios"]:
            inclusion = scenario["target_inclusion_subjects"]
            evaluation = scenario["evaluation_subjects"]
            scenario["target_inclusion_window_count"] = len(inclusion) * 4
            scenario["target_inclusion_window_ids_sha256"] = canonical_json_sha256(
                _window_ids(inclusion, functional_only=False)
            )
            scenario["evaluation_window_count"] = len(evaluation) * 4
            scenario["evaluation_window_ids_sha256"] = canonical_json_sha256(
                _window_ids(evaluation, functional_only=False)
            )
    _rewrite_hashed(split_path, split, "split_manifest_sha256")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["split_manifest_sha256"] = split["split_manifest_sha256"]
    _rewrite_hashed(receipt_path, receipt, "record_sha256")
    zero = json.loads(zero_path.read_text(encoding="utf-8"))
    zero["split_manifest_sha256"] = split["split_manifest_sha256"]
    zero["opening_receipt_record_sha256"] = receipt["record_sha256"]
    _rewrite_hashed(zero_path, zero, "record_sha256")

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    template = inventory["models"][0]
    models: list[dict[str, Any]] = []
    artifact_directory = tmp_path / "artifacts"
    artifact_directory.mkdir()
    for model_id in EXPECTED_MODEL_IDS:
        for seed in EXPECTED_SEEDS:
            entry = copy.deepcopy(template)
            configuration = entry["training_configuration"]
            configuration["seed"] = seed
            calibrator: dict[str, Any] = {
                "schema_version": "1.0.0",
                "record_kind": "source_temperature_calibrator",
                "status": "frozen_source_validation",
                "evidence_status": "source_validation_calibration_target_sealed",
                "method": "scalar_temperature",
                "fit_partition": "source_validation",
                "checkpoint_sha256": _digest(f"checkpoint:{model_id}:{seed}"),
                "training_configuration_sha256": canonical_json_sha256(configuration),
                "split_manifest_sha256": split["split_manifest_sha256"],
                "validation_window_ids_sha256": _digest(f"validation-windows:{model_id}:{seed}"),
                "class_names": list(FUNCTIONAL_CORE_CLASS_ORDER),
                "class_schema_sha256": canonical_json_sha256(list(FUNCTIONAL_CORE_CLASS_ORDER)),
                "fit_input": {
                    "sample_count": 3,
                    "logits_sha256": _digest(f"logits:{model_id}:{seed}"),
                    "labels_sha256": _digest(f"labels:{model_id}:{seed}"),
                },
                "temperature": 2.0,
                "nll_before": 1.0,
                "nll_after": 0.9,
                "target_subject_or_window_records_used": False,
                "target_labels_or_performance_used": False,
            }
            calibrator["record_sha256"] = canonical_json_sha256(calibrator)
            calibrator_path = artifact_directory / f"{model_id}--seed-{seed}.json"
            calibrator_path.write_text(json.dumps(calibrator), encoding="utf-8")
            entry.update(
                {
                    "model_id": model_id,
                    "seed": seed,
                    "selected_epoch": configuration["epochs"],
                    "training_configuration_sha256": canonical_json_sha256(configuration),
                    "checkpoint": {
                        "path": f"artifacts/{model_id}--seed-{seed}.pt",
                        "sha256": _digest(f"checkpoint:{model_id}:{seed}"),
                    },
                    "calibrator": {
                        "path": calibrator_path.relative_to(tmp_path).as_posix(),
                        "sha256": sha256_file(calibrator_path),
                    },
                    "calibrator_record_sha256": calibrator["record_sha256"],
                }
            )
            models.append(entry)
    inventory["models"] = models
    inventory["split_manifest_sha256"] = split["split_manifest_sha256"]
    _rewrite_hashed(inventory_path, inventory, "inventory_sha256")

    v1 = build_few_person_manifest(
        split_manifest_path=split_path,
        config_path=v1_config,
        opening_receipt_path=receipt_path,
        zero_shot_index_path=zero_path,
        final_freeze_inventory_path=inventory_path,
    )
    v1_path = tmp_path / "few-person-v1.json"
    write_few_person_manifest_new(v1, v1_path, allowed_root=tmp_path)
    v1_1 = build_few_person_v1_1_manifest(
        split_manifest_path=split_path,
        config_path=v1_1_config,
        opening_receipt_path=receipt_path,
        zero_shot_index_path=zero_path,
        final_freeze_inventory_path=inventory_path,
        superseded_v1_manifest_path=v1_path,
    )
    v1_1_path = tmp_path / "few-person-v1-1.json"
    write_few_person_v1_1_manifest_new(v1_1, v1_1_path, allowed_root=tmp_path)
    return v1_path, v1_1_path, v1_1


def test_v1_1_uses_functional_core_metadata_and_preserves_v1(
    tmp_path: Path, repository_root: Path
) -> None:
    v1_path, v1_1_path, v1_1 = _materialize_v1_1_plan(tmp_path, repository_root)
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    old = next(
        value
        for value in v1["scenarios"]
        if value["fold_id"] == "target_outer_01" and value["k"] == 1
    )
    corrected = next(
        value
        for value in v1_1["scenarios"]
        if value["fold_id"] == "target_outer_01" and value["k"] == 1
    )

    assert old["target_inclusion_window_count"] == 4
    assert corrected["target_inclusion_window_count"] == 3
    assert corrected["target_inclusion_window_ids_sha256"] == canonical_json_sha256(
        _window_ids(corrected["target_inclusion_subjects"], functional_only=True)
    )
    assert corrected["evaluation_window_count"] == 6
    assert v1_1["class_names"] == list(FUNCTIONAL_CORE_CLASS_ORDER)
    assert v1_1["supersession"]["prior_outputs_preserved"] is True
    assert v1_path.is_file() and v1_1_path.is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_few_person_v1_1_manifest_new(v1_1, v1_1_path, allowed_root=tmp_path)


def test_read_only_progress_requires_exact_1200_cells(
    tmp_path: Path, repository_root: Path
) -> None:
    _, manifest_path, _ = _materialize_v1_1_plan(tmp_path, repository_root)
    results_root = tmp_path / "results" / "postconfirmatory" / "few_person_v1_1"
    results_root.mkdir(parents=True)

    progress = validate_few_person_progress(
        manifest_path, results_root=results_root, artifact_root=tmp_path
    )

    assert progress["valid_so_far"] is True
    assert progress["aggregation_ready"] is False
    assert progress["observed"]["missing_cell_count"] == EXPECTED_CELL_COUNT
    assert len(progress["missing_cells"]) == 1200
    assert progress["target_raw_values_accessed"] is False
    assert progress["gpu_execution_performed"] is False
    assert progress["record_sha256"] == canonical_json_sha256(
        {key: value for key, value in progress.items() if key != "record_sha256"}
    )


def _write_valid_cell(
    tmp_path: Path,
    manifest_path: Path,
    *,
    wrong_metric: bool = False,
    attempt: int | None = None,
    environment_cudnn_enabled_override: bool | None = None,
    device_cudnn_enabled_override: bool | None = None,
) -> Path:
    plan = _load_plan(manifest_path)
    key = CellKey("compact-erm", 11, "target_outer_01", 1)
    scenario = plan.scenarios[(key.fold_id, key.k)]
    model = plan.models[(key.model_id, key.seed)]
    run_directory_name = key.cell_id if attempt is None else f"{key.cell_id}--attempt-{attempt:03d}"
    run_dir = tmp_path / "results" / run_directory_name
    run_dir.mkdir(parents=True)
    participants: list[str] = []
    window_ids: list[str] = []
    labels: list[int] = []
    for subject in scenario["evaluation_subjects"]:
        for label in range(3):
            participants.append(subject)
            window_ids.append(f"subject-{subject}-window-{label}")
            labels.append(label)
    true_labels = np.asarray(labels, dtype=np.int64)
    logits = np.full((len(labels), 3), -1.0, dtype=np.float64)
    logits[np.arange(len(labels)), true_labels] = 2.0
    shifted = logits - logits.max(axis=1, keepdims=True)
    uncalibrated = np.exp(shifted)
    uncalibrated /= uncalibrated.sum(axis=1, keepdims=True)
    calibrated_shifted = logits / 2.0
    calibrated_shifted -= calibrated_shifted.max(axis=1, keepdims=True)
    probabilities = np.exp(calibrated_shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    prediction_path = run_dir / "predictions.npz"
    with prediction_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            window_ids=np.asarray(window_ids),
            participant_ids=np.asarray(participants),
            true_labels=true_labels,
            logits=logits,
            uncalibrated_probabilities=uncalibrated,
            source_temperature_probabilities=probabilities,
            predicted_labels=true_labels,
        )
    adaptation_seed = _expected_adaptation_seed(key.seed, key.fold_id, key.k)
    configuration = model["training_configuration"]
    commit = "a" * 40
    created_at_utc = "2099-01-01T00:00:00Z"
    environment = {
        "python": "3.11.0",
        "platform": "synthetic",
        "numpy": "2.0.0",
        "torch": "2.12.0",
        "torch_cuda_runtime": "13.2",
        "device_type": "cuda",
        "device_index": 0,
        "device_name": "synthetic",
        "device_total_memory_bytes": 1024,
        "compute_capability": [12, 0],
        "cudnn_enabled_during_run": (
            not configuration["disable_cudnn"]
            if environment_cudnn_enabled_override is None
            else environment_cudnn_enabled_override
        ),
        "cudnn_version": 9000,
        "hostname_recorded": False,
        "process_id_recorded": False,
    }
    split_path = tmp_path / "split.json"
    raw_path = tmp_path / "data" / "raw" / "synthetic-inclusivehar.csv"
    input_base = {
        "schema_version": "1.0.0",
        "split_manifest": {
            "path": split_path.relative_to(tmp_path).as_posix(),
            "record_sha256": plan.record["split_manifest_sha256"],
            "file_sha256": sha256_file(split_path),
        },
        "training": {
            "mode": "immutable_raw_csv_authorized_windows_only",
            "source_path": raw_path.relative_to(tmp_path).as_posix(),
            "source_artifact_sha256": sha256_file(raw_path),
        },
        "evaluation": {
            "mode": "immutable_raw_csv_authorized_windows_only",
            "source_path": raw_path.relative_to(tmp_path).as_posix(),
            "source_artifact_sha256": sha256_file(raw_path),
        },
    }
    checkpoint = {
        "schema_version": "1.0.0",
        "record_kind": "few_person_adapted_neural_checkpoint",
        "created_at_utc": created_at_utc,
        "evidence_status": FEW_PERSON_V1_1_EVIDENCE_STATUS,
        "protocol_id": FEW_PERSON_V1_1_PROTOCOL_ID,
        "model_id": key.model_id,
        "seed": key.seed,
        "adaptation_seed": adaptation_seed,
        "fold_id": key.fold_id,
        "k": key.k,
        "inclusion_subjects": sorted(scenario["target_inclusion_subjects"], key=int),
        "inclusion_window_count": scenario["target_inclusion_window_count"],
        "inclusion_window_ids_sha256": scenario["target_inclusion_window_ids_sha256"],
        "evaluation_subjects_excluded_from_training": sorted(
            scenario["evaluation_subjects"], key=int
        ),
        "normalization_fit_subjects": scenario["normalization_fit_subjects"],
        "target_validation_performed": False,
        "calibration_refit_on_target": False,
        "threshold_selection_performed": False,
        "model_state": {"weight": torch.ones(1)},
        "optimizer_state": {"state": {}},
        "normalization": {"training_participants": scenario["normalization_fit_subjects"]},
        "label_schema": list(FUNCTIONAL_CORE_CLASS_ORDER),
        "base_checkpoint_sha256": model["checkpoint"]["sha256"],
        "base_training_configuration": configuration,
        "base_training_configuration_sha256": model["training_configuration_sha256"],
        "few_person_manifest_sha256": plan.manifest_sha256,
        "split_manifest_sha256": plan.record["split_manifest_sha256"],
        "code_commit": commit,
        "environment": environment,
        "checkpoint_selection_rule": "fixed_last_epoch_no_validation",
        "training_history": [
            {
                "epoch": epoch,
                "training_cross_entropy": 0.5,
                "checkpoint_selected": epoch == configuration["epochs"],
                "validation_accessed": False,
            }
            for epoch in range(1, configuration["epochs"] + 1)
        ],
        "rng_states": {
            "python": (),
            "numpy": (),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": [],
        },
        "parameter_count": 1,
        "input_materialization": {
            **input_base,
            "access_barrier": {
                "evaluation_metadata_preflight_only_before_adaptation": False,
                "evaluation_signals_accessed_before_adapted_checkpoint_fixed": False,
                "adapted_checkpoint_fixed_before_evaluation": False,
            },
        },
    }
    checkpoint_path = run_dir / "adapted.pt"
    torch.save(checkpoint, checkpoint_path)
    report = classification_report(
        true_labels,
        probabilities,
        participants,
        class_names=FUNCTIONAL_CORE_CLASS_ORDER,
    )
    if wrong_metric:
        report["participants"][0]["macro_f1"] = 0.25
    inherited_fields = (
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "mixed_precision",
        "disable_cudnn",
    )
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "few_person_outer_fold_result",
        "created_at_utc": created_at_utc,
        "status": "complete_create_only_postconfirmatory_secondary",
        "evidence_status": FEW_PERSON_V1_1_EVIDENCE_STATUS,
        "protocol_id": FEW_PERSON_V1_1_PROTOCOL_ID,
        "model_id": key.model_id,
        "seed": key.seed,
        "adaptation_seed": adaptation_seed,
        "fold_id": key.fold_id,
        "k": key.k,
        "few_person_manifest_sha256": plan.manifest_sha256,
        "split_manifest_sha256": plan.record["split_manifest_sha256"],
        "opening_receipt_record_sha256": plan.record["opening_receipt_record_sha256"],
        "zero_shot_index_record_sha256": plan.record["zero_shot_index_record_sha256"],
        "base_checkpoint_sha256": model["checkpoint"]["sha256"],
        "base_source_calibrator_sha256": model["calibrator"]["sha256"],
        "base_training_configuration_sha256": model["training_configuration_sha256"],
        "source_hyperparameters_inherited": {
            field: configuration[field] for field in inherited_fields
        },
        "adaptation_objective": "supervised_cross_entropy_only",
        "checkpoint_selection_rule": "fixed_last_epoch_no_validation",
        "normalization_refit_on_target": False,
        "calibration_refit_on_target": False,
        "threshold_selection_performed": False,
        "target_validation_performed": False,
        "inclusion_subjects": sorted(scenario["target_inclusion_subjects"], key=int),
        "evaluation_subjects": sorted(scenario["evaluation_subjects"], key=int),
        "unused_target_subjects": scenario["unused_target_subjects"],
        "adapted_checkpoint": {
            "path": checkpoint_path.relative_to(tmp_path).as_posix(),
            "sha256": sha256_file(checkpoint_path),
        },
        "prediction_artifact": {
            "path": prediction_path.relative_to(tmp_path).as_posix(),
            "sha256": sha256_file(prediction_path),
        },
        "participant_level_report": report,
        "statistical_unit": "participant",
        "target_information_used_for_model_or_hyperparameter_selection": False,
        "input_materialization": {
            **input_base,
            "access_barrier": {
                "evaluation_metadata_preflight_only_before_adaptation": False,
                "evaluation_signals_accessed_before_adapted_checkpoint_fixed": False,
                "adapted_checkpoint_fixed_before_evaluation": True,
                "adapted_checkpoint_sha256": sha256_file(checkpoint_path),
            },
        },
        "device": {
            "type": "cuda",
            "name": "synthetic",
            "peak_vram_bytes": 0,
            "cudnn_enabled": (
                environment["cudnn_enabled_during_run"]
                if device_cudnn_enabled_override is None
                else device_cudnn_enabled_override
            ),
        },
        "code_commit": commit,
        "environment": environment,
        "elapsed_seconds": 1.0,
    }
    result["record_sha256"] = canonical_json_sha256(result)
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return result_path


def test_completed_cell_hashes_and_npz_metrics_are_reconstructed(
    tmp_path: Path, repository_root: Path
) -> None:
    _, manifest_path, _ = _materialize_v1_1_plan(tmp_path, repository_root)
    result_path = _write_valid_cell(tmp_path, manifest_path)
    completed = _validate_completed_record(
        result_path,
        plan=_load_plan(manifest_path),
        artifact_root=tmp_path,
        results_root=tmp_path / "results",
    )
    assert set(completed.participant_macro_f1) == {"12", "18"}


def test_completed_cell_accepts_canonical_attempt_suffixed_directory(
    tmp_path: Path, repository_root: Path
) -> None:
    _, manifest_path, _ = _materialize_v1_1_plan(tmp_path, repository_root)
    result_path = _write_valid_cell(tmp_path, manifest_path, attempt=2)
    assert result_path.parent.name == "compact-erm--seed-11--target_outer_01--k1--attempt-002"

    completed = _validate_completed_record(
        result_path,
        plan=_load_plan(manifest_path),
        artifact_root=tmp_path,
        results_root=tmp_path / "results",
    )

    assert completed.key == CellKey("compact-erm", 11, "target_outer_01", 1)


def test_completed_cell_rejects_actual_cudnn_policy_mismatch_from_frozen_config(
    tmp_path: Path, repository_root: Path
) -> None:
    _, manifest_path, _ = _materialize_v1_1_plan(tmp_path, repository_root)
    plan = _load_plan(manifest_path)
    configuration = plan.models[("compact-erm", 11)]["training_configuration"]
    result_path = _write_valid_cell(
        tmp_path,
        manifest_path,
        device_cudnn_enabled_override=bool(configuration["disable_cudnn"]),
    )

    with pytest.raises(FewPersonStatisticsError, match="cuDNN policy differs"):
        _validate_completed_record(
            result_path,
            plan=plan,
            artifact_root=tmp_path,
            results_root=tmp_path / "results",
        )


def test_completed_cell_rejects_environment_cudnn_policy_mismatch(
    tmp_path: Path, repository_root: Path
) -> None:
    _, manifest_path, _ = _materialize_v1_1_plan(tmp_path, repository_root)
    plan = _load_plan(manifest_path)
    configuration = plan.models[("compact-erm", 11)]["training_configuration"]
    result_path = _write_valid_cell(
        tmp_path,
        manifest_path,
        environment_cudnn_enabled_override=bool(configuration["disable_cudnn"]),
    )

    with pytest.raises(FewPersonStatisticsError, match="cuDNN policy differs"):
        _validate_completed_record(
            result_path,
            plan=plan,
            artifact_root=tmp_path,
            results_root=tmp_path / "results",
        )


def test_few_person_runbook_uses_pinned_cache_and_canonical_attempt_paths(
    repository_root: Path,
) -> None:
    runbook = (repository_root / "docs/EXPERIMENT_RUNBOOK.md").read_text(encoding="utf-8")
    assert (
        '$cacheRecordFileHash = "02a2190e90615a8b9ad4c934a3240aa00de62314de9f0be98d45916d756a00ba"'
        in runbook
    )
    assert (
        '$cellId = "$($model.model_id)--seed-$($model.seed)--$($scenario.fold_id)--k$($scenario.k)"'
        in runbook
    )
    assert '$output = "$outputRoot/$cellId--attempt-$attemptToken"' in runbook


def test_completed_cell_rejects_nonreconstructable_metrics(
    tmp_path: Path, repository_root: Path
) -> None:
    _, manifest_path, _ = _materialize_v1_1_plan(tmp_path, repository_root)
    result_path = _write_valid_cell(tmp_path, manifest_path, wrong_metric=True)
    with pytest.raises(FewPersonStatisticsError, match="metrics do not reconstruct"):
        _validate_completed_record(
            result_path,
            plan=_load_plan(manifest_path),
            artifact_root=tmp_path,
            results_root=tmp_path / "results",
        )


def _rewrite_prediction_result(
    result_path: Path,
    mutate: Any,
    *,
    recompute_report: bool = False,
) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    prediction_path = result_path.parent / "predictions.npz"
    with np.load(prediction_path, allow_pickle=False) as stored:
        arrays: dict[str, Any] = {name: np.asarray(stored[name]) for name in stored.files}
    mutate(arrays)
    with prediction_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    result["prediction_artifact"]["sha256"] = sha256_file(prediction_path)
    if recompute_report:
        result["participant_level_report"] = classification_report(
            np.asarray(arrays["true_labels"], dtype=np.int64),
            np.asarray(arrays["source_temperature_probabilities"], dtype=np.float64),
            [str(value) for value in arrays["participant_ids"].tolist()],
            class_names=FUNCTIONAL_CORE_CLASS_ORDER,
        )
    result.pop("record_sha256")
    result["record_sha256"] = canonical_json_sha256(result)
    result_path.write_text(json.dumps(result), encoding="utf-8")


def test_prediction_rejects_valid_simplex_not_equal_to_softmax_logits(
    tmp_path: Path, repository_root: Path
) -> None:
    _, manifest_path, _ = _materialize_v1_1_plan(tmp_path, repository_root)
    result_path = _write_valid_cell(tmp_path, manifest_path)

    def mutate(arrays: dict[str, Any]) -> None:
        values = arrays["uncalibrated_probabilities"].copy()
        values[:, [0, 1]] = values[:, [1, 0]]
        arrays["uncalibrated_probabilities"] = values

    _rewrite_prediction_result(result_path, mutate)
    with pytest.raises(FewPersonStatisticsError, match=r"softmax\(logits\)"):
        _validate_completed_record(
            result_path,
            plan=_load_plan(manifest_path),
            artifact_root=tmp_path,
            results_root=tmp_path / "results",
        )


def test_prediction_rejects_calibration_not_from_frozen_temperature(
    tmp_path: Path, repository_root: Path
) -> None:
    _, manifest_path, _ = _materialize_v1_1_plan(tmp_path, repository_root)
    result_path = _write_valid_cell(tmp_path, manifest_path)

    def mutate(arrays: dict[str, Any]) -> None:
        values = arrays["source_temperature_probabilities"].copy()
        values[:, [0, 1]] = values[:, [1, 0]]
        arrays["source_temperature_probabilities"] = values
        arrays["predicted_labels"] = values.argmax(axis=1).astype(np.int64)

    _rewrite_prediction_result(result_path, mutate, recompute_report=True)
    with pytest.raises(FewPersonStatisticsError, match="frozen temperature"):
        _validate_completed_record(
            result_path,
            plan=_load_plan(manifest_path),
            artifact_root=tmp_path,
            results_root=tmp_path / "results",
        )


def test_result_rejects_absolute_machine_artifact_path(
    tmp_path: Path, repository_root: Path
) -> None:
    _, manifest_path, _ = _materialize_v1_1_plan(tmp_path, repository_root)
    result_path = _write_valid_cell(tmp_path, manifest_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["adapted_checkpoint"]["path"] = str((result_path.parent / "adapted.pt").resolve())
    result.pop("record_sha256")
    result["record_sha256"] = canonical_json_sha256(result)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(FewPersonStatisticsError, match="repository-relative"):
        _validate_completed_record(
            result_path,
            plan=_load_plan(manifest_path),
            artifact_root=tmp_path,
            results_root=tmp_path / "results",
        )


def test_locked_zero_shot_resolver_rebases_only_frozen_namespace(tmp_path: Path) -> None:
    subtree = tmp_path / "results" / "confirmatory" / "zero_shot_v1"
    subtree.mkdir(parents=True)
    result = subtree / "compact-coral--seed-11.result.json"
    result.write_text("{}\n", encoding="utf-8")
    prediction = subtree / "compact-coral--seed-11.predictions.npz"
    prediction.write_bytes(b"fixture")

    historical = (
        r"C:\Users\researcher\workspace\inclusive-shift-har\results\confirmatory"
        r"\zero_shot_v1\compact-coral--seed-11.result.json"
    )
    assert statistics_module._resolve_locked_zero_shot_file(
        historical, artifact_root=tmp_path, name="zero-shot result"
    ) == result.resolve(strict=True)
    assert statistics_module._resolve_locked_zero_shot_file(
        "confirmatory/zero_shot_v1/compact-coral--seed-11.predictions.npz",
        artifact_root=tmp_path,
        name="zero-shot prediction",
    ) == prediction.resolve(strict=True)


@pytest.mark.parametrize(
    "value",
    [
        r"C:\outside\compact-coral--seed-11.result.json",
        "results/confirmatory/other/compact-coral--seed-11.result.json",
        "results/confirmatory/zero_shot_v1/nested/result.json",
        "results/confirmatory/zero_shot_v1/../result.json",
    ],
)
def test_locked_zero_shot_resolver_rejects_other_paths(tmp_path: Path, value: str) -> None:
    subtree = tmp_path / "results" / "confirmatory" / "zero_shot_v1"
    subtree.mkdir(parents=True)
    with pytest.raises(FewPersonStatisticsError):
        statistics_module._resolve_locked_zero_shot_file(
            value, artifact_root=tmp_path, name="zero-shot result"
        )


def test_checkpoint_rejects_claimed_precheckpoint_evaluation_access(
    tmp_path: Path, repository_root: Path
) -> None:
    _, manifest_path, _ = _materialize_v1_1_plan(tmp_path, repository_root)
    result_path = _write_valid_cell(tmp_path, manifest_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint_path = result_path.parent / "adapted.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["input_materialization"]["access_barrier"][
        "evaluation_signals_accessed_before_adapted_checkpoint_fixed"
    ] = True
    torch.save(checkpoint, checkpoint_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    result["adapted_checkpoint"]["sha256"] = checkpoint_hash
    result["input_materialization"]["access_barrier"]["adapted_checkpoint_sha256"] = checkpoint_hash
    result.pop("record_sha256")
    result["record_sha256"] = canonical_json_sha256(result)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(FewPersonStatisticsError, match="access barrier"):
        _validate_completed_record(
            result_path,
            plan=_load_plan(manifest_path),
            artifact_root=tmp_path,
            results_root=tmp_path / "results",
        )


def test_manifest_assignment_mutation_is_rejected_against_preopening_split(
    tmp_path: Path, repository_root: Path
) -> None:
    _, manifest_path, _ = _materialize_v1_1_plan(tmp_path, repository_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = json.loads((tmp_path / "split.json").read_text(encoding="utf-8"))
    split["few_person_outer_folds"][0]["scenarios"][0]["evaluation_subjects"] = [
        "13",
        "18",
    ]
    with pytest.raises(FewPersonProtocolError, match="pre-opening split assignment changed"):
        validate_few_person_v1_1_manifest_assignments(manifest, split)


def test_deferred_shard_loader_never_opens_signals_without_fixed_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = False

    def forbidden_loader(*_: Any, **__: Any) -> Any:
        nonlocal opened
        opened = True
        raise AssertionError("shard loader must not be reached")

    monkeypatch.setattr(few_person_module, "_load_preflight_shard_array", forbidden_loader)
    preflight = few_person_module._PrimaryCacheMetadataPreflight(
        record_path=tmp_path / "record.json",
        record_file_sha256="a" * 64,
        record_sha256="b" * 64,
        participant_shards={},
    )
    with pytest.raises(few_person_module.FewPersonRunError, match="exact fixed adapted checkpoint"):
        few_person_module._load_cached_participant_shards(
            preflight=preflight,
            participants={"12"},
            records=(),
            class_names=FUNCTIONAL_CORE_CLASS_ORDER,
            artifact_root=tmp_path,
            adapted_checkpoint_path=tmp_path / "missing.pt",
            expected_adapted_checkpoint_sha256="e" * 64,
        )
    assert opened is False


def test_statistics_preflights_every_destination_before_scanning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "exports"
    output.mkdir()
    (output / "blocked.csv").write_text("preserved\n", encoding="utf-8")
    scan_called = False

    def forbidden_scan(*_: Any, **__: Any) -> Any:
        nonlocal scan_called
        scan_called = True
        raise AssertionError("scan must occur only after destination preflight")

    monkeypatch.setattr(
        statistics_module,
        "_require_repository_head",
        lambda _root, *, expected_commit: expected_commit,
    )
    monkeypatch.setattr(statistics_module, "_scan_progress", forbidden_scan)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        aggregate_few_person_statistics(
            tmp_path / "missing-manifest.json",
            results_root=tmp_path / "missing-results",
            artifact_root=tmp_path,
            output_directory=output,
            prefix="blocked",
            created_at_utc="2099-01-01T00:00:00Z",
            aggregation_code_commit="a" * 40,
        )
    assert scan_called is False
    assert not (output / "blocked.json").exists()
    assert not (output / "blocked.md").exists()


def test_statistics_output_directory_is_created_once_inside_allowed_root(tmp_path: Path) -> None:
    parent = tmp_path / "few-person"
    parent.mkdir()
    output = parent / "statistics"
    resolved, json_path, csv_path, markdown_path = (
        statistics_module._preflight_statistics_destinations(output, "safe", allowed_root=tmp_path)
    )
    assert resolved == output.resolve(strict=True)
    assert (json_path.name, csv_path.name, markdown_path.name) == (
        "safe.json",
        "safe.csv",
        "safe.md",
    )
    escaped = tmp_path.parent / f"{tmp_path.name}-escaped-statistics"
    with pytest.raises(FewPersonStatisticsError, match="escapes allowed root"):
        statistics_module._preflight_statistics_destinations(
            escaped, "blocked", allowed_root=tmp_path
        )
    assert not escaped.exists()


def test_statistics_json_completion_marker_is_written_last(
    tmp_path: Path, repository_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, manifest_path, _ = _materialize_v1_1_plan(tmp_path, repository_root)
    plan = _load_plan(manifest_path)
    output = tmp_path / "exports"
    output.mkdir()
    curves = {
        (model_id, k): {participant: 0.5 for participant in EXPECTED_TARGET_SUBJECTS}
        for model_id in EXPECTED_MODEL_IDS
        for k in EXPECTED_K_VALUES
    }
    fake_scan = statistics_module._Scan(
        plan=plan,
        completed={},
        progress={
            "valid_so_far": True,
            "aggregation_ready": True,
            "failed_cells": [],
            "execution_lineage": {
                "code_commits": ["a" * 40],
                "environment_sha256": ["e" * 64],
                "environment_hash_scope": (
                    "machine_and_software_environment_excluding_model_specific_cudnn_policy"
                ),
            },
        },
    )
    monkeypatch.setattr(statistics_module, "_scan_progress", lambda *args, **kwargs: fake_scan)
    monkeypatch.setattr(
        statistics_module,
        "_require_repository_head",
        lambda _root, *, expected_commit: expected_commit,
    )
    monkeypatch.setattr(statistics_module, "_seed_averaged_curves", lambda _: curves)
    monkeypatch.setattr(statistics_module, "_comparisons", lambda _: [])
    original_writer = atomic_write_json_new

    def completion_writer(*args: Any, **kwargs: Any) -> Any:
        assert (output / "ordered.csv").is_file()
        assert (output / "ordered.md").is_file()
        return original_writer(*args, **kwargs)

    monkeypatch.setattr(statistics_module, "atomic_write_json_new", completion_writer)
    aggregate_few_person_statistics(
        manifest_path,
        results_root=tmp_path,
        artifact_root=tmp_path,
        output_directory=output,
        prefix="ordered",
        bootstrap_resamples=100,
        created_at_utc="2099-01-01T00:00:00Z",
        aggregation_code_commit="a" * 40,
    )
    assert (output / "ordered.json").is_file()


def test_unified_few_person_builder_routes_only_to_v1_1() -> None:
    assert not hasattr(few_person_module, "build_few_person_manifest")
    args = few_person_module.build_parser().parse_args(
        [
            "build-manifest",
            "--split-manifest",
            "split.json",
            "--config",
            "v1.1.yaml",
            "--opening-receipt",
            "receipt.json",
            "--zero-shot-index",
            "index.json",
            "--final-freeze-inventory",
            "freeze.json",
            "--superseded-v1-manifest",
            "preserved-v1.json",
            "--output",
            "new-v1.1.json",
            "--allowed-root",
            ".",
        ]
    )
    assert args.superseded_v1_manifest == Path("preserved-v1.json")


def test_seed_averaging_precedes_participant_inference_and_holm() -> None:
    completed: dict[CellKey, _Completed] = {}
    fold_participants = {
        "target_outer_01": ("12", "18"),
        "target_outer_02": ("13", "15"),
        "target_outer_03": ("17", "19"),
        "target_outer_04": ("14", "16"),
        "target_outer_05": ("11", "20"),
    }
    dummy = Path("synthetic")
    for model_index, model_id in enumerate(EXPECTED_MODEL_IDS):
        for seed in EXPECTED_SEEDS:
            for fold in EXPECTED_FOLDS:
                for k in EXPECTED_K_VALUES:
                    key = CellKey(model_id, seed, fold, k)
                    values = {
                        participant: min(
                            0.99,
                            0.2 + model_index / 1000.0 + k / 100.0 + seed / 100_000.0,
                        )
                        for participant in fold_participants[fold]
                    }
                    completed[key] = _Completed(
                        key=key,
                        record_path=dummy,
                        record_sha256="a" * 64,
                        record_file_sha256="b" * 64,
                        checkpoint_path=dummy,
                        checkpoint_sha256="c" * 64,
                        prediction_path=dummy,
                        prediction_sha256="d" * 64,
                        code_commit="a" * 40,
                        environment_sha256="e" * 64,
                        window_ids=(),
                        participant_ids=(),
                        labels=np.asarray([], dtype=np.int64),
                        probabilities=np.empty((0, 3), dtype=np.float64),
                        participant_macro_f1=values,
                    )
    curves = _seed_averaged_curves(completed)
    assert len(curves) == len(EXPECTED_MODEL_IDS) * len(EXPECTED_K_VALUES)
    assert all(set(values) == EXPECTED_TARGET_SUBJECTS for values in curves.values())
    for model_id in EXPECTED_MODEL_IDS:
        curves[(model_id, 0)] = {
            participant: curves[(model_id, 1)][participant] - 0.01
            for participant in EXPECTED_TARGET_SUBJECTS
        }
    comparisons = _comparisons(curves)
    assert len(comparisons) == len(EXPECTED_MODEL_IDS) * 5
    assert all(0.0 <= row["holm_adjusted_permutation_p_value"] <= 1.0 for row in comparisons)
