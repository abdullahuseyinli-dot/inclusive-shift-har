"""Post-confirmatory disabled-cohort cross-subject protocol construction.

The protocol deliberately reuses the five target participant pairs that were
declared before target opening 1 for the few-person outer folds.  It rotates the
next pair into validation and assigns the remaining six participants to
training.  The resulting evidence is descriptive and post-confirmatory; it is
never promoted to the locked zero-shot endpoint.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml

from inclusive_shift_har.evaluation.postconfirmatory_inputs import (
    load_consumed_target_context,
)
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)

WITHIN_GROUP_SCHEMA_VERSION = "1.0.0"
WITHIN_GROUP_PROTOCOL_ID = "inclusivehar-v4-disabled-within-group-cross-subject-v1"
WITHIN_GROUP_EVIDENCE_STATUS = "post_confirmatory_descriptive_disabled_within_group_cross_subject"
FUNCTIONAL_CORE_CLASS_ORDER = ("mobility", "sitting", "standing")
TARGET_PARTICIPANTS = frozenset(str(value) for value in range(11, 21))
PREOPENING_PAIRS = (
    ("12", "18"),
    ("13", "15"),
    ("17", "19"),
    ("14", "16"),
    ("11", "20"),
)
SUPPORTED_MODEL_IDS = ("compact-erm", "deepconvlstm", "more-har-backbone")
REQUIRED_SEEDS = (11, 23, 47, 89, 131)


class WithinGroupProtocolError(RuntimeError):
    """Raised when descriptive within-group protocol lineage is invalid."""


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WithinGroupProtocolError(f"{name} must be an object")
    return value


def _json_object(path: str | Path, *, name: str) -> dict[str, Any]:
    return dict(_mapping(load_json_strict(path), name=name))


def _artifact_file(value: str | Path, *, root: Path, name: str) -> Path:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    if candidate.is_symlink():
        raise WithinGroupProtocolError(f"{name} may not be a symlink")
    path = candidate.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise WithinGroupProtocolError(f"{name} escapes artifact root") from exc
    if path.is_symlink() or not path.is_file():
        raise WithinGroupProtocolError(f"{name} must be a regular file")
    return path


def _self_hash(record: Mapping[str, Any], *, field: str, name: str) -> str:
    claimed = record.get(field)
    body = dict(record)
    body.pop(field, None)
    if not isinstance(claimed, str) or claimed != canonical_json_sha256(body):
        raise WithinGroupProtocolError(f"{name} self-hash does not validate")
    return claimed


def _string_list(value: Any, *, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise WithinGroupProtocolError(f"{name} must contain unique non-empty strings")
    return cast(list[str], value)


def _integer_list(value: Any, *, name: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or len(set(value)) != len(value)
    ):
        raise WithinGroupProtocolError(f"{name} must contain unique integers")
    return cast(list[int], value)


def _load_config(path: str | Path) -> tuple[dict[str, Any], str, str]:
    config_path = Path(path)
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = dict(_mapping(parsed, name="within-group config"))
    required = {
        "schema_version": WITHIN_GROUP_SCHEMA_VERSION,
        "protocol_id": WITHIN_GROUP_PROTOCOL_ID,
        "status": "predeclared_post_confirmatory_descriptive",
        "dataset_id": "inclusivehar_v4",
        "ontology_track": "functional_core",
        "class_order": list(FUNCTIONAL_CORE_CLASS_ORDER),
        "target_participants": sorted(TARGET_PARTICIPANTS, key=int),
        "source_locked_model_ids": list(SUPPORTED_MODEL_IDS),
        "seeds": list(REQUIRED_SEEDS),
        "evidence_status": WITHIN_GROUP_EVIDENCE_STATUS,
    }
    mismatches = [key for key, expected in required.items() if config.get(key) != expected]
    if mismatches:
        raise WithinGroupProtocolError(f"within-group config contract differs: {mismatches}")
    folds = config.get("folds")
    if not isinstance(folds, list) or len(folds) != len(PREOPENING_PAIRS):
        raise WithinGroupProtocolError("within-group config requires exactly five folds")
    expected_folds: list[dict[str, Any]] = []
    for index, evaluation_pair in enumerate(PREOPENING_PAIRS):
        validation_pair = PREOPENING_PAIRS[(index + 1) % len(PREOPENING_PAIRS)]
        train = sorted(TARGET_PARTICIPANTS - set(evaluation_pair) - set(validation_pair), key=int)
        expected_folds.append(
            {
                "fold_id": f"disabled_outer_{index + 1:02d}",
                "training_subjects": train,
                "validation_subjects": list(validation_pair),
                "evaluation_subjects": list(evaluation_pair),
            }
        )
    if folds != expected_folds:
        raise WithinGroupProtocolError(
            "folds differ from the cyclic rotation of pre-opening participant pairs"
        )
    normalization = _mapping(config.get("normalization"), name="normalization policy")
    calibration = _mapping(config.get("calibration"), name="calibration policy")
    execution = _mapping(config.get("execution"), name="execution policy")
    statistics = _mapping(config.get("statistics"), name="statistics policy")
    if normalization != {
        "fit_scope": "fold_training_participants_only",
        "method": "per_channel_population_standardization",
        "minimum_scale": 1e-8,
    }:
        raise WithinGroupProtocolError("normalization policy changed")
    if calibration != {
        "method": "scalar_temperature",
        "fit_scope": "fold_validation_participants_only",
        "checkpoint_selection": "forbidden_fixed_last_epoch",
        "evaluation_access_before_calibrator_freeze": False,
    }:
        raise WithinGroupProtocolError("calibration/evaluation barrier policy changed")
    required_execution = {
        "neural_device": "cuda_only",
        "cpu_neural_fallback": False,
        "one_cell_per_process": True,
        "sequential_runs_required": True,
        "create_only_artifacts": True,
        "failures_preserved": True,
        "raw_csv_interface": False,
        "new_target_opening_interface": False,
    }
    if execution != required_execution:
        raise WithinGroupProtocolError("execution policy changed")
    if statistics != {
        "participant_is_independent_unit": True,
        "window_as_independent_unit": False,
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 20260824,
        "paired_reference_model_id": "compact-erm",
        "multiple_comparison_correction": "holm",
    }:
        raise WithinGroupProtocolError("statistics policy changed")
    return config, canonical_json_sha256(config), sha256_file(config_path)


def _functional_records(
    split: Mapping[str, Any], participants: set[str]
) -> tuple[Mapping[str, Any], ...]:
    windows = split.get("windows")
    if not isinstance(windows, list):
        raise WithinGroupProtocolError("parent split lacks window records")
    records: list[Mapping[str, Any]] = []
    for value in windows:
        if not isinstance(value, Mapping):
            raise WithinGroupProtocolError("parent split window is not an object")
        labels = value.get("canonical_labels")
        if (
            value.get("partition") == "target_sealed"
            and value.get("subject_id") in participants
            and isinstance(labels, Mapping)
            and labels.get("functional_core") in FUNCTIONAL_CORE_CLASS_ORDER
        ):
            records.append(value)
    records.sort(key=lambda item: int(cast(int, item["start_row_inclusive"])))
    if not records:
        raise WithinGroupProtocolError("fold role contains no functional-core windows")
    observed_participants = {str(record["subject_id"]) for record in records}
    if observed_participants != participants:
        raise WithinGroupProtocolError("fold role lacks functional-core participant coverage")
    ids = [str(record["window_id"]) for record in records]
    if len(set(ids)) != len(ids):
        raise WithinGroupProtocolError("fold role window identities are duplicated")
    return tuple(records)


def _validate_preopening_pair_lineage(split: Mapping[str, Any]) -> None:
    value = split.get("few_person_outer_folds")
    if not isinstance(value, list) or len(value) != len(PREOPENING_PAIRS):
        raise WithinGroupProtocolError("parent split lacks five pre-opening target folds")
    for index, expected_pair in enumerate(PREOPENING_PAIRS):
        fold = _mapping(value[index], name="pre-opening target fold")
        scenarios = fold.get("scenarios")
        if (
            fold.get("fold_id") != f"target_outer_{index + 1:02d}"
            or not isinstance(scenarios, list)
            or len(scenarios) != 3
            or any(
                _mapping(scenario, name="pre-opening scenario").get("evaluation_subjects")
                != list(expected_pair)
                for scenario in scenarios
            )
        ):
            raise WithinGroupProtocolError(
                "configured evaluation pair differs from pre-opening split metadata"
            )


def _role_record(split: Mapping[str, Any], participants: Sequence[str]) -> dict[str, Any]:
    participant_set = set(participants)
    records = _functional_records(split, participant_set)
    ids = [str(record["window_id"]) for record in records]
    counts = Counter(
        str(_mapping(record["canonical_labels"], name="window labels")["functional_core"])
        for record in records
    )
    if set(counts) != set(FUNCTIONAL_CORE_CLASS_ORDER) or min(counts.values()) < 1:
        raise WithinGroupProtocolError("fold role does not contain every functional-core class")
    return {
        "participant_ids": list(participants),
        "participant_count": len(participants),
        "window_count": len(records),
        "window_ids_sha256": canonical_json_sha256(sorted(ids)),
        "ordered_window_ids_sha256": canonical_json_sha256(ids),
        "class_window_counts": {name: counts[name] for name in FUNCTIONAL_CORE_CLASS_ORDER},
    }


def _model_entries(
    inventory: Mapping[str, Any], configured_ids: Sequence[str], seeds: Sequence[int]
) -> list[dict[str, Any]]:
    values = inventory.get("models")
    if not isinstance(values, list):
        raise WithinGroupProtocolError("freeze inventory lacks model records")
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for value in values:
        entry = _mapping(value, name="freeze model entry")
        model_id = entry.get("model_id")
        seed = entry.get("seed")
        if isinstance(model_id, str) and isinstance(seed, int) and not isinstance(seed, bool):
            key = (model_id, seed)
            if key in indexed:
                raise WithinGroupProtocolError("freeze model/seed identities are duplicated")
            indexed[key] = entry
    result: list[dict[str, Any]] = []
    for model_id in configured_ids:
        for seed in seeds:
            try:
                entry = indexed[(model_id, seed)]
            except KeyError as exc:
                raise WithinGroupProtocolError(
                    f"freeze lacks configured model/seed {model_id}/{seed}"
                ) from exc
            configuration = dict(
                _mapping(entry.get("training_configuration"), name="training configuration")
            )
            configuration_hash = entry.get("training_configuration_sha256")
            classification_only = {
                "checkpoint_selection_rule": "fixed_last_epoch",
                "num_classes": len(FUNCTIONAL_CORE_CLASS_ORDER),
                "seed": seed,
                "use_augmentation": False,
                "use_content_objective": False,
                "use_realization_factorization": False,
                "use_group_dro": False,
                "coral_weight": 0.0,
                "dann_domain_loss_weight": 0.0,
                "zero_channel_indices": [],
            }
            mismatch = [
                key
                for key, expected in classification_only.items()
                if configuration.get(key) != expected
            ]
            if (
                entry.get("training_regime") != "fixed_epoch_neural"
                or entry.get("selected_epoch") != configuration.get("epochs")
                or configuration_hash != canonical_json_sha256(configuration)
                or mismatch
            ):
                raise WithinGroupProtocolError(
                    f"{model_id}/{seed} is not a source-locked classification-only model: "
                    f"{mismatch}"
                )
            result.append(
                {
                    "model_id": model_id,
                    "seed": seed,
                    "source_locked_training_configuration": configuration,
                    "source_locked_training_configuration_sha256": configuration_hash,
                    "source_selected_epoch_used_as_fixed_budget": entry["selected_epoch"],
                    "source_checkpoint_or_weights_reused": False,
                    "initialization": "from_scratch_with_declared_seed",
                    "objective": "supervised_cross_entropy_only",
                }
            )
    return result


def build_within_group_manifest(
    *,
    config_path: str | Path,
    split_manifest_path: str | Path,
    opening_receipt_path: str | Path,
    locked_target_index_path: str | Path,
    final_freeze_inventory_path: str | Path,
    artifact_root: str | Path,
) -> dict[str, Any]:
    """Build a metadata-only manifest for the descriptive within-group track."""

    root = Path(artifact_root).resolve(strict=True)
    config_file = _artifact_file(config_path, root=root, name="within-group config")
    split_file = _artifact_file(split_manifest_path, root=root, name="parent split")
    freeze_file = _artifact_file(
        final_freeze_inventory_path, root=root, name="final freeze inventory"
    )
    config, config_sha256, config_file_sha256 = _load_config(config_file)
    split = _json_object(split_file, name="parent split")
    split_sha256 = _self_hash(split, field="split_manifest_sha256", name="parent split")
    protocol = _mapping(split.get("protocol"), name="parent split protocol")
    required_split = {
        "dataset_id": "inclusivehar_v4",
        "subject_assignment_before_windowing": True,
        "window_length_samples": 128,
        "window_stride_samples": 128,
    }
    split_mismatches = [
        key for key, expected in required_split.items() if split.get(key) != expected
    ]
    if split_mismatches or protocol.get("protocol_id") != "inclusivehar-released-block-v1.2":
        raise WithinGroupProtocolError(f"parent split contract differs: {split_mismatches}")
    _validate_preopening_pair_lineage(split)
    context = load_consumed_target_context(
        opening_receipt_path=opening_receipt_path,
        locked_target_index_path=locked_target_index_path,
        artifact_root=root,
    )
    if (
        context.receipt.get("split_manifest_sha256") != split_sha256
        or context.index.get("split_manifest_sha256") != split_sha256
        or context.class_names != FUNCTIONAL_CORE_CLASS_ORDER
    ):
        raise WithinGroupProtocolError("opening-1 lineage differs from parent split/schema")
    inventory = _json_object(freeze_file, name="final freeze inventory")
    inventory_sha256 = _self_hash(
        inventory, field="inventory_sha256", name="final freeze inventory"
    )
    if (
        inventory.get("split_manifest_sha256") != split_sha256
        or context.index.get("final_freeze_inventory_sha256") != inventory_sha256
        or inventory.get("required_seed_order") != list(REQUIRED_SEEDS)
    ):
        raise WithinGroupProtocolError("freeze/opening/split lineage disagrees")
    ontology = _mapping(split.get("ontology"), name="parent ontology")
    schemas = _mapping(ontology.get("runnable_track_schemas"), name="ontology schemas")
    functional = dict(_mapping(schemas.get("functional_core"), name="functional schema"))
    functional_hash = functional.pop("class_schema_sha256", None)
    required_schema = {
        "track": "functional_core",
        "class_count": 3,
        "class_order": list(FUNCTIONAL_CORE_CLASS_ORDER),
        "index_by_class": {name: index for index, name in enumerate(FUNCTIONAL_CORE_CLASS_ORDER)},
    }
    if any(
        functional.get(key) != expected for key, expected in required_schema.items()
    ) or functional_hash != canonical_json_sha256(functional):
        raise WithinGroupProtocolError("functional-core ontology schema changed")

    folds: list[dict[str, Any]] = []
    evaluation_union: set[str] = set()
    for configured in cast(list[dict[str, Any]], config["folds"]):
        train = _string_list(configured["training_subjects"], name="training subjects")
        validation = _string_list(configured["validation_subjects"], name="validation subjects")
        evaluation = _string_list(configured["evaluation_subjects"], name="evaluation subjects")
        sets = (set(train), set(validation), set(evaluation))
        if (
            any(left & right for index, left in enumerate(sets) for right in sets[index + 1 :])
            or set().union(*sets) != TARGET_PARTICIPANTS
        ):
            raise WithinGroupProtocolError("a fold does not partition target participants")
        if evaluation_union & set(evaluation):
            raise WithinGroupProtocolError("evaluation participants repeat across outer folds")
        evaluation_union.update(evaluation)
        folds.append(
            {
                "fold_id": configured["fold_id"],
                "training": _role_record(split, train),
                "validation": _role_record(split, validation),
                "evaluation": _role_record(split, evaluation),
                "participant_sets_are_pairwise_disjoint": True,
                "participant_assignment_precedes_window_loading": True,
                "normalization_fit_role": "training",
                "calibration_fit_role": "validation",
                "checkpoint_selection_rule": "fixed_last_epoch",
                "evaluation_loading_barrier": "after_checkpoint_and_calibrator_are_frozen",
            }
        )
    if evaluation_union != TARGET_PARTICIPANTS:
        raise WithinGroupProtocolError("outer folds do not evaluate every participant once")

    source_evidence = _mapping(split.get("source_evidence"), name="source evidence")
    source_artifact_sha256 = source_evidence.get("sensor_artifact_sha256")
    if not isinstance(source_artifact_sha256, str) or len(source_artifact_sha256) != 64:
        raise WithinGroupProtocolError("split source artifact hash is invalid")
    model_ids = _string_list(config["source_locked_model_ids"], name="model IDs")
    seeds = _integer_list(config["seeds"], name="seeds")
    payload: dict[str, Any] = {
        "schema_version": WITHIN_GROUP_SCHEMA_VERSION,
        "manifest_kind": "postconfirmatory_disabled_within_group_cross_subject",
        "protocol_id": WITHIN_GROUP_PROTOCOL_ID,
        "status": "ready_no_cell_run",
        "evidence_status": WITHIN_GROUP_EVIDENCE_STATUS,
        "scientific_role": "secondary_descriptive_not_locked_confirmatory",
        "config": {
            "path": config_file.relative_to(root).as_posix(),
            "canonical_sha256": config_sha256,
            "file_sha256": config_file_sha256,
        },
        "split_manifest": {
            "path": split_file.relative_to(root).as_posix(),
            "record_sha256": split_sha256,
            "file_sha256": sha256_file(split_file),
        },
        "source_artifact_sha256": source_artifact_sha256,
        "opening_1": {
            "receipt_path": context.receipt_path.relative_to(context.artifact_root).as_posix(),
            "receipt_record_sha256": context.receipt["record_sha256"],
            "receipt_file_sha256": context.receipt_file_sha256,
            "index_path": context.index_path.relative_to(context.artifact_root).as_posix(),
            "index_record_sha256": context.index["record_sha256"],
            "index_file_sha256": context.index_file_sha256,
            "target_seal_id": context.index["target_seal_id"],
            "opening_number": 1,
        },
        "final_freeze_inventory": {
            "path": freeze_file.relative_to(root).as_posix(),
            "record_sha256": inventory_sha256,
            "file_sha256": sha256_file(freeze_file),
        },
        "ontology": {
            "track": "functional_core",
            "config_sha256": ontology["config_sha256"],
            "class_schema_sha256": functional_hash,
            "class_names": list(FUNCTIONAL_CORE_CLASS_ORDER),
        },
        "fold_construction": {
            "basis": "preopening_few_person_outer_evaluation_pairs",
            "preopening_pairs": [list(pair) for pair in PREOPENING_PAIRS],
            "validation_pair_rule": "next_pair_cyclically",
            "training_rule": "remaining_six_target_participants",
            "performance_driven_assignment": False,
            "constructed_after_opening_1": True,
        },
        "folds": folds,
        "models": _model_entries(inventory, model_ids, seeds),
        "model_order": model_ids,
        "seed_order": seeds,
        "normalization_policy": dict(
            _mapping(config["normalization"], name="normalization policy")
        ),
        "calibration_policy": dict(_mapping(config["calibration"], name="calibration policy")),
        "execution_policy": dict(_mapping(config["execution"], name="execution policy")),
        "statistics_policy": dict(_mapping(config["statistics"], name="statistics policy")),
        "expected_cell_count": len(folds) * len(model_ids) * len(seeds),
        "raw_dataset_file_accessed_during_manifest_build": False,
        "target_signals_accessed_during_manifest_build": False,
        "target_predictions_or_performance_used_for_fold_or_model_design": False,
        "target_label_metadata_used_for_postconfirmatory_coverage_audit": True,
        "new_target_opening_created": False,
    }
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    return payload


def write_within_group_manifest_new(
    manifest: Mapping[str, Any], destination: str | Path, *, allowed_root: str | Path
) -> Path:
    """Validate and publish a within-group manifest without replacement."""

    _self_hash(manifest, field="manifest_sha256", name="within-group manifest")
    if manifest.get("protocol_id") != WITHIN_GROUP_PROTOCOL_ID:
        raise WithinGroupProtocolError("writer accepts only the within-group v1 protocol")
    return atomic_write_json_new(dict(manifest), destination, allowed_root=allowed_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--opening-receipt", type=Path, required=True)
    parser.add_argument("--locked-target-index", type=Path, required=True)
    parser.add_argument("--final-freeze-inventory", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = build_within_group_manifest(
            config_path=args.config,
            split_manifest_path=args.split_manifest,
            opening_receipt_path=args.opening_receipt,
            locked_target_index_path=args.locked_target_index,
            final_freeze_inventory_path=args.final_freeze_inventory,
            artifact_root=args.artifact_root,
        )
        path = write_within_group_manifest_new(manifest, args.output, allowed_root=args.output_root)
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "fail", "message": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest_sha256": manifest["manifest_sha256"],
                "output": str(path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
