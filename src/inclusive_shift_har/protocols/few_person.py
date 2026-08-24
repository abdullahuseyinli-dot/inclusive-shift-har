"""Deterministic post-confirmatory few-person inclusion-curve manifests."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml

from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)

FEW_PERSON_SCHEMA_VERSION = "1.0.0"
FEW_PERSON_PROTOCOL_ID = "inclusivehar-v4-few-person-inclusion-curve-v1"
FEW_PERSON_EVIDENCE_STATUS = "post_confirmatory_secondary_few_person_not_zero_shot_confirmatory"


class FewPersonProtocolError(ValueError):
    """Raised when post-confirmatory few-person lineage is not exact."""


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FewPersonProtocolError(f"{name} must be an object")
    return value


def _json_object(path: str | Path, *, name: str) -> dict[str, Any]:
    value = load_json_strict(path)
    return dict(_mapping(value, name=name))


def _self_hash(record: Mapping[str, Any], *, field: str, name: str) -> str:
    claimed = record.get(field)
    body = dict(record)
    body.pop(field, None)
    if not isinstance(claimed, str) or claimed != canonical_json_sha256(body):
        raise FewPersonProtocolError(f"{name} self-hash does not validate")
    return claimed


def _load_config(path: str | Path) -> tuple[dict[str, Any], str]:
    parsed = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    config = dict(_mapping(parsed, name="few-person config"))
    required = {
        "schema_version": FEW_PERSON_SCHEMA_VERSION,
        "protocol_id": FEW_PERSON_PROTOCOL_ID,
        "status": "predeclared_post_confirmatory_secondary",
        "parent_protocol_id": "inclusivehar-released-block-v1.2",
        "dataset_id": "inclusivehar_v4",
        "ontology_track": "functional_core",
        "model_scope": "all_fixed_epoch_neural_entries_in_preconfirmatory_freeze",
        "evidence_status": FEW_PERSON_EVIDENCE_STATUS,
    }
    mismatches = [key for key, expected in required.items() if config.get(key) != expected]
    if mismatches or config.get("k_values") != [1, 2, 4]:
        raise FewPersonProtocolError(f"few-person config contract mismatch: {mismatches}")
    adaptation = _mapping(config.get("adaptation"), name="adaptation policy")
    expected_adaptation = {
        "training_data": "only_the_k_predeclared_target_inclusion_participants",
        "checkpoint_selection": "fixed_last_epoch_no_early_stopping",
        "normalization": "reuse_frozen_source_training_statistics_without_refit",
        "calibration": "reuse_frozen_source_validation_temperature_without_refit",
        "threshold_selection": "none",
        "target_validation": "forbidden",
    }
    if any(adaptation.get(key) != value for key, value in expected_adaptation.items()):
        raise FewPersonProtocolError("few-person leakage-control policy changed")
    execution = _mapping(config.get("execution"), name="execution policy")
    if (
        execution.get("neural_device") != "cuda_only"
        or execution.get("one_scenario_per_process") is not True
    ):
        raise FewPersonProtocolError("few-person execution must remain one-scenario CUDA-only")
    return config, canonical_json_sha256(config)


def _config_folds(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    folds_value = config.get("outer_folds")
    if not isinstance(folds_value, list) or len(folds_value) != 5:
        raise FewPersonProtocolError("few-person config must declare five outer folds")
    folds: list[dict[str, Any]] = []
    evaluation_counts: Counter[str] = Counter()
    for value in folds_value:
        fold = _mapping(value, name="configured outer fold")
        evaluation = fold.get("evaluation_subjects")
        inclusion = fold.get("inclusion_order")
        if (
            not isinstance(evaluation, list)
            or len(evaluation) != 2
            or not isinstance(inclusion, list)
            or len(inclusion) != 8
            or len(set(cast(list[str], evaluation) + cast(list[str], inclusion))) != 10
        ):
            raise FewPersonProtocolError("configured target fold is not a 2/8 partition")
        evaluation_counts.update(cast(list[str], evaluation))
        folds.append(
            {
                "fold_id": str(fold["fold_id"]),
                "evaluation_subjects": list(evaluation),
                "inclusion_order": list(inclusion),
            }
        )
    expected = Counter({str(value): 1 for value in range(11, 21)})
    if evaluation_counts != expected:
        raise FewPersonProtocolError("outer evaluation pairs do not cover every target once")
    return folds


def _validate_postconfirmatory(
    receipt: Mapping[str, Any],
    zero_shot_index: Mapping[str, Any],
) -> None:
    _self_hash(receipt, field="record_sha256", name="opening receipt")
    _self_hash(zero_shot_index, field="record_sha256", name="zero-shot index")
    receipt_required = {
        "record_kind": "confirmatory_target_opening_receipt",
        "status": "unlock_consumed_before_materialization",
        "target_opening_number": 1,
    }
    index_required = {
        "record_kind": "locked_target_evaluation_index",
        "status": "complete_create_only",
        "evidence_status": "locked_confirmatory_target_opening_1",
        "target_opening_number": 1,
        "target_information_used_for_model_selection": False,
    }
    if any(receipt.get(key) != value for key, value in receipt_required.items()):
        raise FewPersonProtocolError("opening receipt is not the consumed confirmatory receipt")
    if any(zero_shot_index.get(key) != value for key, value in index_required.items()):
        raise FewPersonProtocolError("zero-shot index is not complete confirmatory evidence")
    for key in ("split_manifest_sha256", "target_seal_id"):
        if receipt.get(key) != zero_shot_index.get(key):
            raise FewPersonProtocolError(f"zero-shot receipt/index disagree on {key}")
    if zero_shot_index.get("opening_receipt_record_sha256") != receipt.get("record_sha256"):
        raise FewPersonProtocolError("zero-shot index is not bound to the opening receipt")


def _neural_inventory_models(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    _self_hash(inventory, field="inventory_sha256", name="final freeze inventory")
    required = {
        "record_kind": "final_source_artifact_freeze",
        "status": "frozen_before_target_unlock",
        "evidence_status": "source_only_final_models_target_sealed",
    }
    if any(inventory.get(key) != value for key, value in required.items()):
        raise FewPersonProtocolError("final freeze inventory contract changed")
    models_value = inventory.get("models")
    if not isinstance(models_value, list):
        raise FewPersonProtocolError("final freeze inventory lacks models")
    models: list[dict[str, Any]] = []
    for value in models_value:
        entry = _mapping(value, name="freeze model")
        if entry.get("training_regime") != "fixed_epoch_neural":
            continue
        configuration = _mapping(
            entry.get("training_configuration"), name="neural training configuration"
        )
        if configuration.get("checkpoint_selection_rule") != "fixed_last_epoch":
            raise FewPersonProtocolError("few-person base checkpoint is not fixed-last-epoch")
        models.append(
            {
                "model_id": entry["model_id"],
                "seed": entry["seed"],
                "selected_epoch": entry["selected_epoch"],
                "training_configuration": dict(configuration),
                "training_configuration_sha256": entry["training_configuration_sha256"],
                "checkpoint": dict(_mapping(entry.get("checkpoint"), name="checkpoint")),
                "calibrator": dict(_mapping(entry.get("calibrator"), name="calibrator")),
                "calibrator_record_sha256": entry["calibrator_record_sha256"],
            }
        )
    if not models:
        raise FewPersonProtocolError("final freeze inventory has no neural model entries")
    return models


def build_few_person_manifest(
    *,
    split_manifest_path: str | Path,
    config_path: str | Path,
    opening_receipt_path: str | Path,
    zero_shot_index_path: str | Path,
    final_freeze_inventory_path: str | Path,
) -> dict[str, Any]:
    """Build a metadata-only plan; target values and target metrics are never copied."""

    split = _json_object(split_manifest_path, name="parent split manifest")
    split_sha256 = _self_hash(split, field="split_manifest_sha256", name="parent split")
    config, config_sha256 = _load_config(config_path)
    receipt = _json_object(opening_receipt_path, name="opening receipt")
    zero_shot = _json_object(zero_shot_index_path, name="zero-shot index")
    inventory = _json_object(final_freeze_inventory_path, name="final freeze inventory")
    _validate_postconfirmatory(receipt, zero_shot)
    if receipt.get("split_manifest_sha256") != split_sha256:
        raise FewPersonProtocolError("post-confirmatory receipt differs from parent split")
    seal = _mapping(split.get("target_seal"), name="parent target seal")
    if receipt.get("target_seal_id") != seal.get("seal_id"):
        raise FewPersonProtocolError("post-confirmatory receipt differs from parent target seal")
    if inventory.get("split_manifest_sha256") != split_sha256:
        raise FewPersonProtocolError("final freeze inventory differs from parent split")

    configured_folds = _config_folds(config)
    split_folds_value = split.get("few_person_outer_folds")
    if not isinstance(split_folds_value, list):
        raise FewPersonProtocolError("parent split lacks few-person outer folds")
    split_by_id = {
        str(_mapping(value, name="split outer fold")["fold_id"]): _mapping(
            value, name="split outer fold"
        )
        for value in split_folds_value
    }
    scenarios: list[dict[str, Any]] = []
    for configured in configured_folds:
        fold_id = configured["fold_id"]
        split_fold = split_by_id.get(fold_id)
        if split_fold is None or split_fold.get("inclusion_order") != configured["inclusion_order"]:
            raise FewPersonProtocolError(f"parent split differs from configured fold {fold_id}")
        split_scenarios = split_fold.get("scenarios")
        if not isinstance(split_scenarios, list):
            raise FewPersonProtocolError(f"parent split lacks scenarios for {fold_id}")
        for k in (1, 2, 4):
            scenario = next(
                (
                    _mapping(value, name="split scenario")
                    for value in split_scenarios
                    if isinstance(value, Mapping) and value.get("k") == k
                ),
                None,
            )
            if scenario is None:
                raise FewPersonProtocolError(f"parent split lacks {fold_id} k={k}")
            expected_inclusion = configured["inclusion_order"][:k]
            if (
                scenario.get("evaluation_subjects") != configured["evaluation_subjects"]
                or scenario.get("target_inclusion_subjects") != expected_inclusion
            ):
                raise FewPersonProtocolError(f"parent split assignment changed for {fold_id} k={k}")
            scenarios.append(
                {
                    "scenario_id": f"{fold_id}__k{k}",
                    "fold_id": fold_id,
                    "k": k,
                    "target_inclusion_subjects": list(expected_inclusion),
                    "evaluation_subjects": list(configured["evaluation_subjects"]),
                    "unused_target_subjects": list(scenario["unused_target_subjects"]),
                    "target_inclusion_window_count": scenario["target_inclusion_window_count"],
                    "target_inclusion_window_ids_sha256": scenario[
                        "target_inclusion_window_ids_sha256"
                    ],
                    "evaluation_window_count": scenario["evaluation_window_count"],
                    "evaluation_window_ids_sha256": scenario["evaluation_window_ids_sha256"],
                    "normalization_fit_subjects": list(scenario["normalization_fit_subjects"]),
                    "validation_subjects": [],
                    "calibration_subjects": [],
                    "threshold_selection_subjects": [],
                }
            )

    models = _neural_inventory_models(inventory)
    payload: dict[str, Any] = {
        "schema_version": FEW_PERSON_SCHEMA_VERSION,
        "manifest_kind": "postconfirmatory_few_person_inclusion_curve",
        "protocol_id": FEW_PERSON_PROTOCOL_ID,
        "status": "ready_postconfirmatory_metadata_only_no_scenario_run",
        "evidence_status": FEW_PERSON_EVIDENCE_STATUS,
        "config_sha256": config_sha256,
        "split_manifest_sha256": split_sha256,
        "target_seal_id": receipt["target_seal_id"],
        "opening_receipt_record_sha256": receipt["record_sha256"],
        "opening_receipt_file_sha256": sha256_file(opening_receipt_path),
        "zero_shot_index_record_sha256": zero_shot["record_sha256"],
        "zero_shot_index_file_sha256": sha256_file(zero_shot_index_path),
        "final_freeze_inventory_sha256": inventory["inventory_sha256"],
        "final_freeze_inventory_file_sha256": sha256_file(final_freeze_inventory_path),
        "ontology_track": "functional_core",
        "k_values": [1, 2, 4],
        "scenarios": scenarios,
        "models": models,
        "adaptation_policy": dict(_mapping(config["adaptation"], name="adaptation policy")),
        "execution_policy": dict(_mapping(config["execution"], name="execution policy")),
        "target_metrics_or_predictions_used_for_design_or_selection": False,
        "target_raw_values_accessed_during_manifest_build": False,
    }
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    return payload


def write_few_person_manifest_new(
    manifest: Mapping[str, Any], destination: str | Path, *, allowed_root: str | Path
) -> Path:
    """Publish one self-hashed manifest without replacing prior evidence."""

    _self_hash(manifest, field="manifest_sha256", name="few-person manifest")
    return atomic_write_json_new(dict(manifest), destination, allowed_root=allowed_root)
