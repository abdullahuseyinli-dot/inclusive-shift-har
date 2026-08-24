from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from inclusive_shift_har.data.inclusivehar import INCLUSIVEHAR_PRIMARY_CHANNELS
from inclusive_shift_har.data.materialize import MaterializedWindows
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.evaluation.postconfirmatory_inputs import (
    load_consumed_target_context,
)
from inclusive_shift_har.evaluation.within_group_statistics import (
    CellKey,
    WithinGroupStatisticsError,
    _expected_evaluation_rows,
    exact_sign_flip_pvalue,
    expected_cells,
    hierarchical_bootstrap_ci,
    participant_cluster_bootstrap_ci,
    validate_cell,
)
from inclusive_shift_har.evaluation.within_group_statistics import (
    build_parser as build_statistics_parser,
)
from inclusive_shift_har.experiments.postconfirmatory_cache import (
    prepare_postconfirmatory_primary_caches,
)
from inclusive_shift_har.experiments.within_group import (
    WithinGroupRunError,
    _train_fixed_epoch,
    preflight_participant_shard_store,
    require_cuda_device,
)
from inclusive_shift_har.experiments.within_group import (
    build_parser as build_runner_parser,
)
from inclusive_shift_har.manifests.canonical import (
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)
from inclusive_shift_har.preprocessing.normalization import ChannelStandardizer
from inclusive_shift_har.protocols.within_group import (
    TARGET_PARTICIPANTS,
    WithinGroupProtocolError,
    _load_config,
    _validate_preopening_pair_lineage,
    build_within_group_manifest,
)
from inclusive_shift_har.training.engine import (
    TrainingConfig,
    build_model,
    training_config_from_dict,
)
from tests.test_postconfirmatory_cache import _fixture

REPOSITORY_ROOT = Path(__file__).parents[1]


def _actual_manifest() -> dict[str, Any]:
    return build_within_group_manifest(
        config_path=REPOSITORY_ROOT
        / "configs/protocols/inclusivehar_disabled_within_group_v1.yaml",
        split_manifest_path=REPOSITORY_ROOT
        / "results/protocol/splits/inclusivehar_v4_released_block_v1_2.json",
        opening_receipt_path=REPOSITORY_ROOT
        / "results/protocol/confirmatory_target_opening_1.json",
        locked_target_index_path=REPOSITORY_ROOT
        / "results/confirmatory/zero_shot_v1/locked_target_evaluation_index.json",
        final_freeze_inventory_path=REPOSITORY_ROOT
        / "results/protocol/final_source_artifact_freeze_v1.json",
        artifact_root=REPOSITORY_ROOT,
    )


def test_manifest_has_exact_subject_exclusive_complete_outer_matrix() -> None:
    manifest = _actual_manifest()
    checked_in = load_json_strict(
        REPOSITORY_ROOT / "results/protocol/inclusivehar_disabled_within_group_v1.json"
    )

    assert checked_in == manifest
    assert manifest["status"] == "ready_no_cell_run"
    assert manifest["expected_cell_count"] == 75
    assert len(expected_cells(manifest)) == 75
    evaluation_union: set[str] = set()
    validation_counts = {participant: 0 for participant in TARGET_PARTICIPANTS}
    for fold in manifest["folds"]:
        train = set(fold["training"]["participant_ids"])
        validation = set(fold["validation"]["participant_ids"])
        evaluation = set(fold["evaluation"]["participant_ids"])
        assert len(train) == 6
        assert len(validation) == 2
        assert len(evaluation) == 2
        assert not train & validation
        assert not train & evaluation
        assert not validation & evaluation
        assert train | validation | evaluation == TARGET_PARTICIPANTS
        assert not evaluation_union & evaluation
        evaluation_union.update(evaluation)
        for participant in validation:
            validation_counts[participant] += 1
        for role in ("training", "validation", "evaluation"):
            assert set(fold[role]["class_window_counts"]) == {
                "mobility",
                "sitting",
                "standing",
            }
            assert min(fold[role]["class_window_counts"].values()) > 0
    assert evaluation_union == TARGET_PARTICIPANTS
    assert set(validation_counts.values()) == {1}
    assert manifest["target_predictions_or_performance_used_for_fold_or_model_design"] is False
    assert manifest["scientific_role"] == "secondary_descriptive_not_locked_confirmatory"


def test_config_rejects_any_fold_assignment_drift(tmp_path: Path) -> None:
    source = REPOSITORY_ROOT / "configs/protocols/inclusivehar_disabled_within_group_v1.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["folds"][0]["validation_subjects"] = ["11", "20"]
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(WithinGroupProtocolError, match="pre-opening participant pairs"):
        _load_config(mutated)


def test_preopening_pair_claim_is_verified_against_parent_split() -> None:
    split = load_json_strict(
        REPOSITORY_ROOT / "results/protocol/splits/inclusivehar_v4_released_block_v1_2.json"
    )
    split["few_person_outer_folds"][0]["scenarios"][0]["evaluation_subjects"] = [
        "11",
        "20",
    ]

    with pytest.raises(WithinGroupProtocolError, match="pre-opening split metadata"):
        _validate_preopening_pair_lineage(split)


def _synthetic_store(tmp_path: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    paths = _fixture(tmp_path)
    cache_record = prepare_postconfirmatory_primary_caches(
        split_manifest_path=paths["split"],
        opening_receipt_path=paths["receipt"],
        locked_target_index_path=paths["index"],
        raw_csv_path=paths["raw"],
        artifact_root=tmp_path,
        cache_directory="primary-v1",
        cache_root=paths["cache_root"],
        record_output="primary-cache.json",
        record_root=paths["record_root"],
        code_commit="a" * 40,
        created_at_utc="2099-01-01T00:00:00Z",
    )
    split = load_json_strict(paths["split"])
    context = load_consumed_target_context(
        opening_receipt_path=paths["receipt"],
        locked_target_index_path=paths["index"],
        artifact_root=tmp_path,
    )
    plan = {
        "split_manifest": {
            "record_sha256": split["split_manifest_sha256"],
            "file_sha256": sha256_file(paths["split"]),
        },
        "source_artifact_sha256": cache_record["raw_sensor_csv"]["sha256"],
        "opening_1": {
            "receipt_record_sha256": context.receipt["record_sha256"],
            "receipt_file_sha256": context.receipt_file_sha256,
            "index_record_sha256": context.index["record_sha256"],
            "index_file_sha256": context.index_file_sha256,
            "target_seal_id": context.index["target_seal_id"],
        },
    }
    store = preflight_participant_shard_store(
        record_path=paths["record_root"] / "primary-cache.json",
        expected_record_file_sha256=sha256_file(paths["record_root"] / "primary-cache.json"),
        split_manifest_path=paths["split"],
        split=split,
        plan=plan,
        context=context,
        artifact_root=tmp_path,
    )
    return store, cache_record, paths


def test_role_loader_parses_only_selected_participant_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, cache_record, _ = _synthetic_store(tmp_path)
    selected = ["11", "14", "16", "17", "19", "20"]
    loaded_paths: list[Path] = []
    original_load = np.load

    def recording_load(path: Any, *args: Any, **kwargs: Any) -> Any:
        loaded_paths.append(Path(path).resolve())
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", recording_load)
    batch, lineage = store.load_role(selected, role="training")

    expected_paths = {store.shards[participant].array_path for participant in selected}
    whole_target = (tmp_path / cache_record["caches"]["target_sealed"]["array_path"]).resolve()
    assert set(loaded_paths) == expected_paths
    assert whole_target not in loaded_paths
    assert set(batch.participant_ids) == set(selected)
    assert lineage["participant_ids"] == selected
    assert lineage["window_count"] == batch.labels.size


def test_cache_preflight_requires_external_file_hash_pin(tmp_path: Path) -> None:
    store, _, paths = _synthetic_store(tmp_path)
    split = load_json_strict(paths["split"])
    context = load_consumed_target_context(
        opening_receipt_path=paths["receipt"],
        locked_target_index_path=paths["index"],
        artifact_root=tmp_path,
    )
    plan = {
        "split_manifest": {
            "record_sha256": split["split_manifest_sha256"],
            "file_sha256": sha256_file(paths["split"]),
        },
        "source_artifact_sha256": store.source_artifact_sha256,
        "opening_1": {
            "receipt_record_sha256": context.receipt["record_sha256"],
            "receipt_file_sha256": context.receipt_file_sha256,
            "index_record_sha256": context.index["record_sha256"],
            "index_file_sha256": context.index_file_sha256,
            "target_seal_id": context.index["target_seal_id"],
        },
    }
    with pytest.raises(WithinGroupRunError, match="file hash changed"):
        preflight_participant_shard_store(
            record_path=paths["record_root"] / "primary-cache.json",
            expected_record_file_sha256="f" * 64,
            split_manifest_path=paths["split"],
            split=split,
            plan=plan,
            context=context,
            artifact_root=tmp_path,
        )


def test_neural_device_fails_closed_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(WithinGroupRunError, match="CPU fallback is forbidden"):
        require_cuda_device()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only synthetic smoke")
@pytest.mark.parametrize(
    ("model_name", "disable_cudnn"),
    [
        ("compact_residual_96", False),
        ("deepconvlstm", True),
        ("more_har", False),
    ],
)
def test_synthetic_training_smoke_stays_on_cuda(model_name: str, disable_cudnn: bool) -> None:
    rng = np.random.default_rng(7)
    participants = tuple(str(11 + index % 6) for index in range(12))
    signals = rng.normal(size=(12, 128, 6)).astype(np.float32)
    labels = np.asarray([index % 3 for index in range(12)], dtype=np.int64)
    batch = MaterializedWindows(
        signals=signals,
        labels=labels,
        window_ids=tuple(f"synthetic-{index}" for index in range(12)),
        participant_ids=participants,
        released_labels=tuple("synthetic" for _ in range(12)),
        partitions=tuple("target_sealed" for _ in range(12)),
        class_names=("mobility", "sitting", "standing"),
        ontology_track="functional_core",
    )
    normalizer = ChannelStandardizer.fit(
        signals,
        list(participants),
        declared_training_participants=set(participants),
        split_manifest_sha256="a" * 64,
        channel_names=INCLUSIVEHAR_PRIMARY_CHANNELS,
    )
    config = TrainingConfig(
        model_name=model_name,
        num_classes=3,
        seed=7,
        epochs=1,
        batch_size=12,
        patience=1,
        minimum_epochs=1,
        checkpoint_interval=1,
        mixed_precision="float16",
        disable_cudnn=disable_cudnn,
        checkpoint_selection_rule="fixed_last_epoch",
    )

    model, states, history, runtime = _train_fixed_epoch(
        batch,
        normalizer=normalizer,
        config=config,
        device=require_cuda_device(),
    )

    assert all(parameter.device.type == "cuda" for parameter in model.parameters())
    assert history[0]["optimizer_update_count"] == 1
    assert states["optimizer_state"]
    assert runtime["cuda_training_performed"] is True
    assert runtime["cpu_neural_training_performed"] is False


def test_participant_statistics_are_deterministic_and_participant_level() -> None:
    values = np.asarray(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
            [0.3, 0.4, 0.5, 0.6],
        ],
        dtype=np.float64,
    )
    first = participant_cluster_bootstrap_ci(values, replicates=1000, seed=7)
    second = participant_cluster_bootstrap_ci(values, replicates=1000, seed=7)
    hierarchical = hierarchical_bootstrap_ci(values, replicates=1000, seed=8)

    assert first == second
    assert first[0] <= values.mean() <= first[1]
    assert hierarchical[0] <= values.mean() <= hierarchical[1]
    assert exact_sign_flip_pvalue([1.0, 2.0, 3.0]) == 0.25


def _write_self_hashed_json(path: Path, payload: dict[str, Any], field: str) -> None:
    payload[field] = canonical_json_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_completed_cell(tmp_path: Path) -> tuple[dict[str, Any], Any, Path, Path, str]:
    plan = _actual_manifest()
    split = load_json_strict(
        REPOSITORY_ROOT / "results/protocol/splits/inclusivehar_v4_released_block_v1_2.json"
    )
    key = CellKey("disabled_outer_01", "compact-erm", 11)
    fold = next(value for value in plan["folds"] if value["fold_id"] == key.fold_id)
    model = next(
        value
        for value in plan["models"]
        if value["model_id"] == key.model_id and value["seed"] == key.seed
    )
    cell = tmp_path / "results" / key.fold_id / key.model_id / f"seed-{key.seed}"
    cell.mkdir(parents=True)
    cache_path = tmp_path / "primary-cache.json"
    cache_record: dict[str, Any] = {"record_kind": "synthetic_primary_cache"}
    _write_self_hashed_json(cache_path, cache_record, "record_sha256")
    cache_file_sha256 = sha256_file(cache_path)
    training_configuration = model["source_locked_training_configuration"]
    reconstructed = build_model(training_config_from_dict(dict(training_configuration)))
    checkpoint = {
        "record_kind": "within_group_fixed_epoch_checkpoint",
        "status": "frozen_before_validation_or_evaluation_loading",
        "evidence_status": plan["evidence_status"],
        "protocol_id": plan["protocol_id"],
        "fold_id": key.fold_id,
        "model_id": key.model_id,
        "seed": key.seed,
        "epoch": training_configuration["epochs"],
        "selected_epoch": training_configuration["epochs"],
        "checkpoint_selection_rule": "fixed_last_epoch",
        "manifest_sha256": plan["manifest_sha256"],
        "split_manifest_sha256": plan["split_manifest"]["record_sha256"],
        "primary_cache_record_file_sha256": cache_file_sha256,
        "validation_arrays_parsed_before_checkpoint_freeze": False,
        "evaluation_arrays_parsed_before_checkpoint_freeze": False,
        "model_state": reconstructed.state_dict(),
        "optimizer_state": {"state": {}},
        "scheduler_state": {"last_epoch": 1},
        "scaler_state": {},
        "rng_states": {"torch_cpu": torch.get_rng_state()},
        "configuration": training_configuration,
        "configuration_sha256": model["source_locked_training_configuration_sha256"],
        "label_schema": ["mobility", "sitting", "standing"],
        "normalization": {"training_participants": sorted(fold["training"]["participant_ids"])},
    }
    checkpoint_path = cell / "selected.pt"
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    calibrator: dict[str, Any] = {
        "record_kind": "within_group_validation_temperature_calibrator",
        "status": "frozen_before_evaluation_loading",
        "evidence_status": plan["evidence_status"],
        "fold_id": key.fold_id,
        "model_id": key.model_id,
        "seed": key.seed,
        "fit_role": "validation",
        "fit_participants": fold["validation"]["participant_ids"],
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_arrays_parsed_before_calibrator_freeze": False,
        "checkpoint_selection_used_validation": False,
        "target_validation_used_for_calibration": True,
        "temperature": 1.0,
    }
    calibrator_path = cell / "calibrator.json"
    _write_self_hashed_json(calibrator_path, calibrator, "record_sha256")
    evaluation_participants = set(fold["evaluation"]["participant_ids"])
    window_ids, participant_ids, labels = _expected_evaluation_rows(split, evaluation_participants)
    logits = np.full((labels.size, 3), -1.0, dtype=np.float64)
    logits[np.arange(labels.size), labels] = 2.0
    exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = np.asarray(exponent / exponent.sum(axis=1, keepdims=True), dtype=np.float64)
    predictions_path = cell / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        evidence_status=np.asarray([plan["evidence_status"]]),
        window_ids=np.asarray(window_ids),
        participant_ids=np.asarray(participant_ids),
        true_labels=labels,
        logits=logits,
        uncalibrated_probabilities=probabilities,
        calibrated_probabilities=probabilities,
        predicted_labels=probabilities.argmax(axis=1).astype(np.int64),
    )
    report = classification_report(
        labels,
        probabilities,
        participant_ids,
        class_names=("mobility", "sitting", "standing"),
    )
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "within_group_fold_model_seed_result",
        "status": "complete_create_only",
        "evidence_status": plan["evidence_status"],
        "scientific_role": "secondary_descriptive_not_locked_confirmatory",
        "protocol_id": plan["protocol_id"],
        "fold_id": key.fold_id,
        "model_id": key.model_id,
        "seed": key.seed,
        "manifest_sha256": plan["manifest_sha256"],
        "split_manifest_sha256": plan["split_manifest"]["record_sha256"],
        "source_artifact_sha256": plan["source_artifact_sha256"],
        "opening_receipt_record_sha256": plan["opening_1"]["receipt_record_sha256"],
        "opening_index_record_sha256": plan["opening_1"]["index_record_sha256"],
        "class_names": ["mobility", "sitting", "standing"],
        "configuration": model["source_locked_training_configuration"],
        "configuration_sha256": model["source_locked_training_configuration_sha256"],
        "fold_roles": {
            role: {
                field: fold[role][field]
                for field in (
                    "participant_ids",
                    "window_count",
                    "window_ids_sha256",
                    "ordered_window_ids_sha256",
                )
            }
            for role in ("training", "validation", "evaluation")
        },
        "primary_cache": {
            "record_path": cache_path.relative_to(tmp_path).as_posix(),
            "record_sha256": cache_record["record_sha256"],
            "record_file_sha256": cache_file_sha256,
        },
        "checkpoint": {
            "path": checkpoint_path.relative_to(tmp_path).as_posix(),
            "sha256": checkpoint_sha256,
            "selected_epoch": training_configuration["epochs"],
        },
        "calibrator": {
            "path": calibrator_path.relative_to(tmp_path).as_posix(),
            "file_sha256": sha256_file(calibrator_path),
            "record_sha256": calibrator["record_sha256"],
        },
        "prediction_array": {
            "path": predictions_path.relative_to(tmp_path).as_posix(),
            "sha256": sha256_file(predictions_path),
        },
        "participant_level_report": report,
        "participant_assignment_before_array_loading": True,
        "normalization_fit_on_training_only": True,
        "checkpoint_frozen_before_validation_loading": True,
        "calibration_fit_on_validation_only": True,
        "checkpoint_and_calibrator_frozen_before_evaluation_loading": True,
        "evaluation_used_for_training_selection_or_calibration": False,
        "disability_or_assistive_device_metadata_used_as_model_input": False,
        "raw_dataset_file_accessed": False,
        "whole_target_cache_array_accessed": False,
        "opening_1_prediction_array_accessed": False,
        "new_target_opening_created": False,
        "cpu_neural_fallback_used": False,
    }
    result_path = cell / "result.json"
    _write_self_hashed_json(result_path, result, "record_sha256")
    return plan, split, tmp_path / "results", predictions_path, cache_file_sha256


def test_cell_validator_recomputes_metrics_and_detects_prediction_tampering(
    tmp_path: Path,
) -> None:
    plan, split, result_root, predictions_path, cache_hash = _synthetic_completed_cell(tmp_path)
    key = CellKey("disabled_outer_01", "compact-erm", 11)

    validated = validate_cell(
        key=key,
        plan=plan,
        plan_sha256=plan["manifest_sha256"],
        split=split,
        result_root=result_root,
        artifact_root=tmp_path,
        primary_cache_record_file_sha256=cache_hash,
    )
    assert set(validated.participant_macro_f1) == {"12", "18"}

    with predictions_path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(WithinGroupStatisticsError, match="prediction path/hash changed"):
        validate_cell(
            key=key,
            plan=plan,
            plan_sha256=plan["manifest_sha256"],
            split=split,
            result_root=result_root,
            artifact_root=tmp_path,
            primary_cache_record_file_sha256=cache_hash,
        )


def test_run_and_aggregate_parsers_expose_no_raw_or_opening_interface() -> None:
    for parser in (build_runner_parser(), build_statistics_parser()):
        help_text = parser.format_help().casefold()
        assert "--primary-cache-record" in help_text
        assert "--raw-csv" not in help_text
        assert "--unlock-record" not in help_text
        assert "--opening-acknowledgement" not in help_text
        assert "--device" not in help_text
