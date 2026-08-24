"""Functional-core-corrected v1.1 few-person protocol manifest."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml

from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    sha256_file,
)
from inclusive_shift_har.protocols.few_person import (
    FewPersonProtocolError,
    _config_folds,
    _json_object,
    _mapping,
    _neural_inventory_models,
    _self_hash,
    _validate_postconfirmatory,
)

FEW_PERSON_V1_1_SCHEMA_VERSION = "1.0.0"
FEW_PERSON_V1_1_PROTOCOL_ID = "inclusivehar-v4-few-person-inclusion-curve-v1.1"
FEW_PERSON_V1_1_EVIDENCE_STATUS = "post_confirmatory_secondary_few_person_v1_1"
FUNCTIONAL_CORE_CLASS_ORDER = ("mobility", "sitting", "standing")


def _load_config(path: str | Path) -> tuple[dict[str, Any], str]:
    parsed = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    config = dict(_mapping(parsed, name="few-person v1.1 config"))
    required = {
        "schema_version": FEW_PERSON_V1_1_SCHEMA_VERSION,
        "protocol_id": FEW_PERSON_V1_1_PROTOCOL_ID,
        "status": "predeclared_post_confirmatory_secondary_superseding_incompatible_v1",
        "supersedes_protocol_id": "inclusivehar-v4-few-person-inclusion-curve-v1",
        "parent_protocol_id": "inclusivehar-released-block-v1.2",
        "dataset_id": "inclusivehar_v4",
        "ontology_track": "functional_core",
        "class_order": list(FUNCTIONAL_CORE_CLASS_ORDER),
        "scenario_window_lineage": (
            "derive counts and sorted-ID hashes from parent target_sealed windows carrying "
            "the functional_core canonical label"
        ),
        "model_scope": "all_fixed_epoch_neural_entries_in_preconfirmatory_freeze",
        "evidence_status": FEW_PERSON_V1_1_EVIDENCE_STATUS,
    }
    mismatches = [key for key, expected in required.items() if config.get(key) != expected]
    if mismatches or config.get("k_values") != [1, 2, 4]:
        raise FewPersonProtocolError(f"few-person v1.1 config mismatch: {mismatches}")
    execution = _mapping(config.get("execution"), name="v1.1 execution")
    if (
        execution.get("neural_device") != "cuda_only"
        or execution.get("all_metadata_preflight_before_output_creation") is not True
    ):
        raise FewPersonProtocolError("v1.1 preflight/CUDA contract changed")
    return config, canonical_json_sha256(config)


def _functional_records(
    split: Mapping[str, Any], subjects: set[str]
) -> tuple[Mapping[str, Any], ...]:
    windows = split.get("windows")
    if not isinstance(windows, list):
        raise FewPersonProtocolError("parent split lacks windows")
    records = tuple(
        cast(Mapping[str, Any], value)
        for value in windows
        if isinstance(value, Mapping)
        and value.get("partition") == "target_sealed"
        and value.get("subject_id") in subjects
        and isinstance(value.get("canonical_labels"), Mapping)
        and "functional_core" in cast(Mapping[str, Any], value["canonical_labels"])
    )
    if not records:
        raise FewPersonProtocolError("functional-core target record set is empty")
    observed_subjects: set[str] = set()
    observed_ids: set[str] = set()
    for record in records:
        subject = record.get("subject_id")
        window_id = record.get("window_id")
        labels = _mapping(record.get("canonical_labels"), name="window canonical labels")
        if (
            not isinstance(subject, str)
            or subject not in subjects
            or not isinstance(window_id, str)
            or not window_id
            or labels.get("functional_core") not in FUNCTIONAL_CORE_CLASS_ORDER
        ):
            raise FewPersonProtocolError("functional-core target window metadata is invalid")
        if window_id in observed_ids:
            raise FewPersonProtocolError("functional-core target window IDs are duplicated")
        observed_subjects.add(subject)
        observed_ids.add(window_id)
    if observed_subjects != subjects:
        raise FewPersonProtocolError("functional-core coverage is missing for a selected subject")
    return records


def _window_lineage(split: Mapping[str, Any], subjects: set[str]) -> tuple[int, str]:
    records = _functional_records(split, subjects)
    ids = sorted(str(record["window_id"]) for record in records)
    return len(ids), canonical_json_sha256(ids)


def _split_folds_by_id(split: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    value = split.get("few_person_outer_folds")
    if not isinstance(value, list) or len(value) != 5:
        raise FewPersonProtocolError("parent split must contain exactly five few-person folds")
    folds: dict[str, Mapping[str, Any]] = {}
    for item in value:
        fold = _mapping(item, name="parent split few-person fold")
        fold_id = fold.get("fold_id")
        if not isinstance(fold_id, str) or not fold_id or fold_id in folds:
            raise FewPersonProtocolError("parent split few-person fold IDs are invalid")
        folds[fold_id] = fold
    return folds


def _validate_split_fold_assignment(
    *,
    split_fold: Mapping[str, Any],
    fold_id: str,
    evaluation_subjects: list[str],
    inclusion_order: list[str],
) -> None:
    split_inclusion = split_fold.get("inclusion_order")
    if (
        not isinstance(split_inclusion, list)
        or any(not isinstance(value, str) for value in split_inclusion)
        or split_inclusion != inclusion_order
    ):
        raise FewPersonProtocolError(f"pre-opening split fold assignment differs for {fold_id}")
    value = split_fold.get("scenarios")
    if not isinstance(value, list) or len(value) != 3:
        raise FewPersonProtocolError(
            f"pre-opening split must contain exactly three scenarios for {fold_id}"
        )
    scenarios: dict[int, Mapping[str, Any]] = {}
    for item in value:
        scenario = _mapping(item, name="parent split few-person scenario")
        k = scenario.get("k")
        if k not in (1, 2, 4) or int(cast(int, k)) in scenarios:
            raise FewPersonProtocolError(
                f"pre-opening split scenario keys are invalid for {fold_id}"
            )
        scenarios[int(cast(int, k))] = scenario
    if set(scenarios) != {1, 2, 4}:
        raise FewPersonProtocolError(f"pre-opening split scenario coverage changed for {fold_id}")
    target_subjects = {str(value) for value in range(11, 21)}
    for k in (1, 2, 4):
        expected_inclusion = inclusion_order[:k]
        expected_unused = target_subjects - set(evaluation_subjects) - set(expected_inclusion)
        scenario = scenarios[k]
        unused = scenario.get("unused_target_subjects")
        if (
            scenario.get("evaluation_subjects") != evaluation_subjects
            or scenario.get("target_inclusion_subjects") != expected_inclusion
            or not isinstance(unused, list)
            or set(cast(list[str], unused)) != expected_unused
            or len(cast(list[str], unused)) != len(expected_unused)
        ):
            raise FewPersonProtocolError(
                f"pre-opening split assignment changed for {fold_id} k={k}"
            )


def validated_few_person_v1_1_folds(
    config: Mapping[str, Any], split: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Prove every configured fold/scenario matches the immutable parent split."""

    configured = _config_folds(config)
    split_by_id = _split_folds_by_id(split)
    configured_ids = {str(fold["fold_id"]) for fold in configured}
    if set(split_by_id) != configured_ids:
        raise FewPersonProtocolError("pre-opening split/config fold ID sets differ")
    for fold in configured:
        fold_id = str(fold["fold_id"])
        _validate_split_fold_assignment(
            split_fold=split_by_id[fold_id],
            fold_id=fold_id,
            evaluation_subjects=cast(list[str], fold["evaluation_subjects"]),
            inclusion_order=cast(list[str], fold["inclusion_order"]),
        )
    return configured


def validate_few_person_v1_1_manifest_assignments(
    manifest: Mapping[str, Any], split: Mapping[str, Any]
) -> None:
    """Validate all 15 manifest assignments against pre-opening split metadata."""

    scenarios_value = manifest.get("scenarios")
    if not isinstance(scenarios_value, list) or len(scenarios_value) != 15:
        raise FewPersonProtocolError("v1.1 manifest must contain exactly 15 scenarios")
    scenarios: dict[tuple[str, int], Mapping[str, Any]] = {}
    for item in scenarios_value:
        scenario = _mapping(item, name="v1.1 manifest scenario")
        fold_id = scenario.get("fold_id")
        k = scenario.get("k")
        if (
            not isinstance(fold_id, str)
            or k not in (1, 2, 4)
            or (fold_id, int(cast(int, k))) in scenarios
        ):
            raise FewPersonProtocolError("v1.1 manifest scenario identities are invalid")
        scenarios[(fold_id, int(cast(int, k)))] = scenario
    split_by_id = _split_folds_by_id(split)
    expected_keys = {(fold_id, k) for fold_id in split_by_id for k in (1, 2, 4)}
    if set(scenarios) != expected_keys:
        raise FewPersonProtocolError("v1.1 manifest/split scenario sets differ")
    for fold_id, split_fold in split_by_id.items():
        inclusion_value = split_fold.get("inclusion_order")
        split_scenarios = split_fold.get("scenarios")
        if not isinstance(inclusion_value, list) or not isinstance(split_scenarios, list):
            raise FewPersonProtocolError("pre-opening split fold assignments are invalid")
        first_scenario = _mapping(
            split_scenarios[0] if split_scenarios else None,
            name="pre-opening split first scenario",
        )
        evaluation_value = first_scenario.get("evaluation_subjects")
        if not isinstance(evaluation_value, list):
            raise FewPersonProtocolError("pre-opening split evaluation assignment is invalid")
        evaluation = cast(list[str], evaluation_value)
        inclusion_order = cast(list[str], inclusion_value)
        _validate_split_fold_assignment(
            split_fold=split_fold,
            fold_id=fold_id,
            evaluation_subjects=evaluation,
            inclusion_order=inclusion_order,
        )
        for k in (1, 2, 4):
            scenario = scenarios[(fold_id, k)]
            expected_inclusion = inclusion_order[:k]
            expected_unused = (
                {str(value) for value in range(11, 21)} - set(evaluation) - set(expected_inclusion)
            )
            unused = scenario.get("unused_target_subjects")
            if (
                scenario.get("scenario_id") != f"{fold_id}__k{k}"
                or scenario.get("evaluation_subjects") != evaluation
                or scenario.get("target_inclusion_subjects") != expected_inclusion
                or not isinstance(unused, list)
                or set(cast(list[str], unused)) != expected_unused
                or len(cast(list[str], unused)) != len(expected_unused)
            ):
                raise FewPersonProtocolError(
                    f"v1.1 manifest differs from pre-opening split for {fold_id} k={k}"
                )


def build_few_person_v1_1_manifest(
    *,
    split_manifest_path: str | Path,
    config_path: str | Path,
    opening_receipt_path: str | Path,
    zero_shot_index_path: str | Path,
    final_freeze_inventory_path: str | Path,
    superseded_v1_manifest_path: str | Path,
) -> dict[str, Any]:
    """Build corrected scenario hashes from functional-core window metadata only."""

    split = _json_object(split_manifest_path, name="parent split manifest")
    split_sha256 = _self_hash(split, field="split_manifest_sha256", name="parent split")
    config, config_sha256 = _load_config(config_path)
    receipt = _json_object(opening_receipt_path, name="opening receipt")
    zero_shot = _json_object(zero_shot_index_path, name="zero-shot index")
    inventory = _json_object(final_freeze_inventory_path, name="final freeze inventory")
    incompatible_v1 = _json_object(superseded_v1_manifest_path, name="incompatible v1 manifest")
    v1_sha256 = _self_hash(
        incompatible_v1, field="manifest_sha256", name="incompatible v1 manifest"
    )
    if incompatible_v1.get("protocol_id") != "inclusivehar-v4-few-person-inclusion-curve-v1":
        raise FewPersonProtocolError("superseded manifest is not few-person v1")
    _validate_postconfirmatory(receipt, zero_shot)
    seal = _mapping(split.get("target_seal"), name="target seal")
    if (
        receipt.get("split_manifest_sha256") != split_sha256
        or receipt.get("target_seal_id") != seal.get("seal_id")
        or inventory.get("split_manifest_sha256") != split_sha256
    ):
        raise FewPersonProtocolError("v1.1 lineage inputs disagree")
    parent_protocol = _mapping(split.get("protocol"), name="parent split protocol")
    if parent_protocol.get("protocol_id") != config["parent_protocol_id"]:
        raise FewPersonProtocolError("v1.1 parent protocol differs from the split")

    ontology = _mapping(split.get("ontology"), name="parent ontology")
    runnable = _mapping(ontology.get("runnable_track_schemas"), name="runnable schemas")
    functional = _mapping(runnable.get("functional_core"), name="functional-core schema")
    functional_body = dict(functional)
    functional_schema_sha256 = functional_body.pop("class_schema_sha256", None)
    if (
        functional.get("track") != "functional_core"
        or functional.get("class_count") != len(FUNCTIONAL_CORE_CLASS_ORDER)
        or functional.get("class_order") != list(FUNCTIONAL_CORE_CLASS_ORDER)
        or functional.get("index_by_class")
        != {name: index for index, name in enumerate(FUNCTIONAL_CORE_CLASS_ORDER)}
        or functional_schema_sha256 != canonical_json_sha256(functional_body)
    ):
        raise FewPersonProtocolError("parent functional-core class schema changed")
    target_window_count, target_window_hash = _window_lineage(
        split, {str(value) for value in range(11, 21)}
    )

    scenarios: list[dict[str, Any]] = []
    for fold in validated_few_person_v1_1_folds(config, split):
        evaluation = set(cast(list[str], fold["evaluation_subjects"]))
        inclusion_order = cast(list[str], fold["inclusion_order"])
        evaluation_count, evaluation_hash = _window_lineage(split, evaluation)
        for k in (1, 2, 4):
            inclusion = set(inclusion_order[:k])
            unused = {str(value) for value in range(11, 21)} - evaluation - inclusion
            inclusion_count, inclusion_hash = _window_lineage(split, inclusion)
            scenarios.append(
                {
                    "scenario_id": f"{fold['fold_id']}__k{k}",
                    "fold_id": fold["fold_id"],
                    "k": k,
                    "target_inclusion_subjects": inclusion_order[:k],
                    "evaluation_subjects": list(fold["evaluation_subjects"]),
                    "unused_target_subjects": sorted(unused, key=int),
                    "target_inclusion_window_count": inclusion_count,
                    "target_inclusion_window_ids_sha256": inclusion_hash,
                    "evaluation_window_count": evaluation_count,
                    "evaluation_window_ids_sha256": evaluation_hash,
                    "window_lineage_track": "functional_core",
                    "normalization_fit_subjects": [
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "9",
                    ],
                    "validation_subjects": [],
                    "calibration_subjects": [],
                    "threshold_selection_subjects": [],
                }
            )

    assignment_probe = {"scenarios": scenarios}
    validate_few_person_v1_1_manifest_assignments(assignment_probe, split)

    models = _neural_inventory_models(inventory)
    payload: dict[str, Any] = {
        "schema_version": FEW_PERSON_V1_1_SCHEMA_VERSION,
        "manifest_kind": "postconfirmatory_few_person_inclusion_curve",
        "protocol_id": FEW_PERSON_V1_1_PROTOCOL_ID,
        "status": "ready_postconfirmatory_functional_core_v1_1_no_scenario_run",
        "evidence_status": FEW_PERSON_V1_1_EVIDENCE_STATUS,
        "supersession": {
            "superseded_protocol_id": incompatible_v1["protocol_id"],
            "superseded_manifest_sha256": v1_sha256,
            "superseded_manifest_file_sha256": sha256_file(superseded_v1_manifest_path),
            "reason": config["supersession_reason"],
            "prior_outputs_preserved": True,
        },
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
        "ontology_config_sha256": ontology["config_sha256"],
        "ontology_class_schema_sha256": functional_schema_sha256,
        "class_names": list(FUNCTIONAL_CORE_CLASS_ORDER),
        "class_schema_sha256": canonical_json_sha256(list(FUNCTIONAL_CORE_CLASS_ORDER)),
        "functional_core_target_window_count": target_window_count,
        "functional_core_target_window_ids_sha256": target_window_hash,
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


def write_few_person_v1_1_manifest_new(
    manifest: Mapping[str, Any], destination: str | Path, *, allowed_root: str | Path
) -> Path:
    _self_hash(manifest, field="manifest_sha256", name="few-person v1.1 manifest")
    if manifest.get("protocol_id") != FEW_PERSON_V1_1_PROTOCOL_ID:
        raise FewPersonProtocolError("writer accepts only few-person v1.1")
    return atomic_write_json_new(dict(manifest), destination, allowed_root=allowed_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--opening-receipt", type=Path, required=True)
    parser.add_argument("--zero-shot-index", type=Path, required=True)
    parser.add_argument("--final-freeze-inventory", type=Path, required=True)
    parser.add_argument("--superseded-v1-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allowed-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = build_few_person_v1_1_manifest(
            split_manifest_path=args.split_manifest,
            config_path=args.config,
            opening_receipt_path=args.opening_receipt,
            zero_shot_index_path=args.zero_shot_index,
            final_freeze_inventory_path=args.final_freeze_inventory,
            superseded_v1_manifest_path=args.superseded_v1_manifest,
        )
        published = write_few_person_v1_1_manifest_new(
            manifest, args.output, allowed_root=args.allowed_root
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "fail", "message": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest_sha256": manifest["manifest_sha256"],
                "output": str(published),
                "target_metrics_or_predictions_used_for_design_or_selection": False,
                "target_raw_values_accessed_during_manifest_build": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
