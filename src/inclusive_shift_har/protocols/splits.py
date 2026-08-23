"""Deterministic construction of participant-exclusive released-block splits."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from inclusive_shift_har.data.windowing import ReleasedBlock, windows_from_released_block
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)
from inclusive_shift_har.protocols.config import ProtocolSpec, load_protocol_spec
from inclusive_shift_har.protocols.ontology import LockedOntology, load_locked_ontology

SENSOR_ARTIFACT_ID = "inclusivehar_v4_sensor_csv"


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _load_yaml_mapping(path: str | Path) -> Mapping[str, Any]:
    parsed = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return _mapping(parsed, name=str(path))


def _verify_authorization(
    path: str | Path,
    *,
    spec: ProtocolSpec,
) -> tuple[Mapping[str, Any], str]:
    record = load_json_strict(path)
    authorization = _mapping(record, name="released-block authorization")
    required = {
        "schema_version": "1.0.0",
        "gate": "released_block_protocol_revision",
        "status": "approved",
        "dataset_id": "inclusivehar_v4",
        "stage3_audit_report_sha256": spec.stage3_audit_report_sha256,
    }
    mismatches = [key for key, value in required.items() if authorization.get(key) != value]
    if mismatches:
        raise PermissionError(f"released-block authorization mismatch: {mismatches}")
    risk = _mapping(authorization.get("accepted_residual_risk"), name="residual risk")
    if risk.get("hidden_trial_boundaries_known") is not False:
        raise PermissionError("authorization must preserve unknown hidden-trial status")
    if risk.get("unconditional_contamination_bound") != 1.0:
        raise PermissionError("authorization must retain the unconditional 100 percent bound")
    if risk.get("trial_safe_claim_authorized") is not False:
        raise PermissionError("authorization may not permit a trial-safe claim")
    return authorization, canonical_json_sha256(authorization)


def _verify_stage3_audit(
    path: str | Path,
    *,
    spec: ProtocolSpec,
) -> tuple[Mapping[str, Any], str]:
    source = Path(path)
    if sha256_file(source) != spec.stage3_audit_file_sha256:
        raise ValueError("Stage 3 audit file hash differs from the locked protocol")
    parsed = load_json_strict(source)
    audit = _mapping(parsed, name="Stage 3 audit")
    claimed_hash = audit.get("report_sha256")
    without_hash = dict(audit)
    without_hash.pop("report_sha256", None)
    observed_hash = canonical_json_sha256(without_hash)
    if claimed_hash != observed_hash or observed_hash != spec.stage3_audit_report_sha256:
        raise ValueError("Stage 3 audit canonical hash differs from the locked protocol")
    if audit.get("status") != "integrity_pass_protocol_quarantine":
        raise ValueError("Stage 3 audit status is not the preserved quarantine status")
    return audit, observed_hash


def _validate_preprocessing_config(
    path: str | Path,
    *,
    spec: ProtocolSpec,
) -> tuple[Mapping[str, Any], str]:
    config = _load_yaml_mapping(path)
    expected = {
        "schema_version": "1.0.0",
        "dataset_id": "inclusivehar_v4",
        "window_length_samples": spec.window_length,
        "window_stride_samples": spec.window_stride,
        "boundary_unit": "released_contiguous_subject_activity_block",
        "remainder_policy": "drop_tail_shorter_than_128_per_released_block",
    }
    mismatches = [key for key, value in expected.items() if config.get(key) != value]
    if mismatches:
        raise ValueError(f"preprocessing config mismatch: {mismatches}")
    normalization = _mapping(config.get("normalization"), name="preprocessing normalization")
    if (
        normalization.get("fit_scope") != "source_training_partition_only"
        or normalization.get("validation_in_fit") is not False
        or normalization.get("target_in_fit") is not False
    ):
        raise ValueError("preprocessing normalization is not source-training-only")
    forbidden = config.get("forbidden_model_inputs")
    if not isinstance(forbidden, list) or not {
        "label",
        "UserID",
        "disabled",
        "global_row_index",
        "file_order",
        "released_block_position",
        "window_offset",
    } <= set(forbidden):
        raise ValueError("preprocessing config lacks required proxy exclusions")
    return config, canonical_json_sha256(config)


def _source_artifact_sha256(audit: Mapping[str, Any]) -> str:
    acquisition = _mapping(audit.get("acquisition"), name="audit acquisition")
    verification = _mapping(acquisition.get("artifact_verification"), name="artifact verification")
    observations = verification.get("observations")
    if not isinstance(observations, list):
        raise ValueError("artifact verification observations must be a list")
    matches = [
        _mapping(item, name="artifact observation")
        for item in observations
        if isinstance(item, Mapping) and item.get("artifact_id") == SENSOR_ARTIFACT_ID
    ]
    if len(matches) != 1:
        raise ValueError("Stage 3 audit does not identify exactly one sensor CSV")
    observed_hash = matches[0].get("observed_sha256")
    if not isinstance(observed_hash, str) or len(observed_hash) != 64:
        raise ValueError("sensor CSV observation lacks SHA-256")
    return observed_hash


def _partition_for_subject(subject_id: str, spec: ProtocolSpec) -> str:
    if subject_id in spec.final_source_train:
        return "source_train"
    if subject_id in spec.final_source_validation:
        return "source_validation"
    if subject_id in spec.target_subjects:
        return "target_sealed"
    raise ValueError(f"subject is outside the locked protocol: {subject_id}")


def _window_ids_for_subjects(windows: Sequence[Mapping[str, Any]], subjects: set[str]) -> list[str]:
    return sorted(
        str(window["window_id"]) for window in windows if str(window["subject_id"]) in subjects
    )


def _partition_summary(
    windows: Sequence[Mapping[str, Any]],
    *,
    subjects: Sequence[str],
    partition: str,
) -> dict[str, Any]:
    window_ids = sorted(
        str(window["window_id"]) for window in windows if window["partition"] == partition
    )
    return {
        "partition": partition,
        "subject_count": len(subjects),
        "subject_ids": list(subjects),
        "window_count": len(window_ids),
        "window_ids_sha256": canonical_json_sha256(window_ids),
    }


def _fold_records(
    windows: Sequence[Mapping[str, Any]], spec: ProtocolSpec
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source = set(spec.source_subjects)
    source_cv: list[dict[str, Any]] = []
    for source_fold in spec.source_cv_folds:
        validation = set(source_fold.validation_subjects)
        train = source - validation
        train_ids = _window_ids_for_subjects(windows, train)
        validation_ids = _window_ids_for_subjects(windows, validation)
        source_cv.append(
            {
                "fold_id": source_fold.fold_id,
                "normalization_fit_subjects": sorted(train, key=int),
                "train_subjects": sorted(train, key=int),
                "train_window_count": len(train_ids),
                "train_window_ids_sha256": canonical_json_sha256(train_ids),
                "validation_subjects": sorted(validation, key=int),
                "validation_window_count": len(validation_ids),
                "validation_window_ids_sha256": canonical_json_sha256(validation_ids),
            }
        )

    source_nested: list[dict[str, Any]] = []
    if spec.nested_source_cv:
        for outer_fold in spec.source_cv_folds:
            outer_test = set(outer_fold.validation_subjects)
            outer_test_ids = _window_ids_for_subjects(windows, outer_test)
            inner_records: list[dict[str, Any]] = []
            for inner_fold in spec.source_cv_folds:
                if inner_fold.fold_id == outer_fold.fold_id:
                    continue
                inner_validation = set(inner_fold.validation_subjects)
                inner_train = source - outer_test - inner_validation
                inner_train_ids = _window_ids_for_subjects(windows, inner_train)
                inner_validation_ids = _window_ids_for_subjects(windows, inner_validation)
                inner_records.append(
                    {
                        "fold_id": f"{outer_fold.fold_id}__inner__{inner_fold.fold_id}",
                        "normalization_fit_subjects": sorted(inner_train, key=int),
                        "train_subjects": sorted(inner_train, key=int),
                        "train_window_count": len(inner_train_ids),
                        "train_window_ids_sha256": canonical_json_sha256(inner_train_ids),
                        "validation_subjects": sorted(inner_validation, key=int),
                        "validation_window_count": len(inner_validation_ids),
                        "validation_window_ids_sha256": canonical_json_sha256(inner_validation_ids),
                    }
                )
            source_nested.append(
                {
                    "inner_folds": inner_records,
                    "outer_fold_id": outer_fold.fold_id,
                    "outer_test_subjects": sorted(outer_test, key=int),
                    "outer_test_window_count": len(outer_test_ids),
                    "outer_test_window_ids_sha256": canonical_json_sha256(outer_test_ids),
                    "outer_test_allowed_for_selection": False,
                }
            )

    target = set(spec.target_subjects)
    few_person: list[dict[str, Any]] = []
    for target_fold in spec.few_person_outer_folds:
        evaluation = set(target_fold.evaluation_subjects)
        scenarios: list[dict[str, Any]] = []
        evaluation_ids = _window_ids_for_subjects(windows, evaluation)
        for k in spec.few_person_k_values:
            inclusion = set(target_fold.inclusion_order[:k])
            unused = target - evaluation - inclusion
            inclusion_ids = _window_ids_for_subjects(windows, inclusion)
            scenarios.append(
                {
                    "evaluation_subjects": sorted(evaluation, key=int),
                    "evaluation_window_count": len(evaluation_ids),
                    "evaluation_window_ids_sha256": canonical_json_sha256(evaluation_ids),
                    "k": k,
                    "normalization_fit_subjects": list(spec.final_source_train),
                    "target_inclusion_subjects": list(target_fold.inclusion_order[:k]),
                    "target_inclusion_window_count": len(inclusion_ids),
                    "target_inclusion_window_ids_sha256": canonical_json_sha256(inclusion_ids),
                    "unused_target_subjects": sorted(unused, key=int),
                }
            )
        few_person.append(
            {
                "fold_id": target_fold.fold_id,
                "inclusion_order": list(target_fold.inclusion_order),
                "scenarios": scenarios,
            }
        )
    return source_cv, source_nested, few_person


def build_released_block_split_manifest(
    *,
    audit_report_path: str | Path,
    protocol_config_path: str | Path,
    ontology_config_path: str | Path,
    preprocessing_config_path: str | Path,
    authorization_path: str | Path,
) -> dict[str, Any]:
    """Build a deterministic manifest without opening raw signals or target performance."""

    spec = load_protocol_spec(protocol_config_path)
    ontology: LockedOntology = load_locked_ontology(ontology_config_path)
    if spec.explicit_class_schema_required and not ontology.explicit_class_order:
        raise ValueError("protocol v1.2 requires an ontology with explicit numeric class order")
    preprocessing, preprocessing_hash = _validate_preprocessing_config(
        preprocessing_config_path, spec=spec
    )
    authorization, authorization_hash = _verify_authorization(authorization_path, spec=spec)
    audit, audit_hash = _verify_stage3_audit(audit_report_path, spec=spec)
    source_hash = _source_artifact_sha256(audit)
    target_seal_id = canonical_json_sha256(
        {
            "dataset_id": "inclusivehar_v4",
            "protocol_sha256": spec.config_sha256,
            "sensor_artifact_sha256": source_hash,
            "target_subjects": list(spec.target_subjects),
        }
    )

    runs_container = _mapping(audit.get("csv_audit"), name="csv audit")
    order = _mapping(runs_container.get("released_order_and_boundaries"), name="released order")
    released_runs = order.get("released_runs")
    if not isinstance(released_runs, list) or not released_runs:
        raise ValueError("Stage 3 audit contains no released runs")

    windows: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    partitioned_subjects = {
        subject: _partition_for_subject(subject, spec)
        for subject in (*spec.source_subjects, *spec.target_subjects)
    }
    for run_value in released_runs:
        run = _mapping(run_value, name="released run")
        subject_id = str(run["subject_id"])
        activity_label = str(run["activity_label"])
        partition = partitioned_subjects[subject_id]
        released_block = ReleasedBlock(
            released_run_id=str(run["released_run_id"]),
            subject_id=subject_id,
            activity_label=activity_label,
            start_row_inclusive=int(run["start_data_row_inclusive"]),
            end_row_inclusive=int(run["end_data_row_inclusive"]),
            row_count=int(run["row_count"]),
        )
        result = windows_from_released_block(
            released_block,
            dataset_id="inclusivehar_v4",
            source_artifact_sha256=source_hash,
            protocol_sha256=spec.config_sha256,
            partition=partition,
            canonical_labels=ontology.labels_for_released_label(activity_label),
            length_samples=spec.window_length,
            stride_samples=spec.window_stride,
            conditional_hidden_joins=spec.conditional_hidden_joins,
        )
        block_summary = result.summary()
        block_summary["partition"] = partition
        blocks.append(block_summary)
        windows.extend(window.to_dict() for window in result.windows)

    windows.sort(key=lambda item: (int(str(item["start_row_inclusive"])), str(item["window_id"])))
    blocks.sort(key=lambda item: int(str(item["start_row_inclusive"])))
    window_ids = [str(window["window_id"]) for window in windows]
    source_cv, source_nested, few_person = _fold_records(windows, spec)
    partition_counts = Counter(str(window["partition"]) for window in windows)
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for window in windows:
        labels = _mapping(window["canonical_labels"], name="window canonical labels")
        for track, label in labels.items():
            label_counts[str(track)][str(label)] += 1

    conditional_crossing_bound = sum(
        int(block["conditional_hidden_join_crossing_window_bound"]) for block in blocks
    )
    total_windows = len(windows)
    participant_risk: dict[str, dict[str, int]] = defaultdict(
        lambda: {"conditional_crossing_window_bound": 0, "window_count": 0}
    )
    partition_risk: dict[str, dict[str, int]] = defaultdict(
        lambda: {"conditional_crossing_window_bound": 0, "window_count": 0}
    )
    for block_summary_record in blocks:
        for key, identifier in (
            ("participant", str(block_summary_record["subject_id"])),
            ("partition", str(block_summary_record["partition"])),
        ):
            target = (
                participant_risk[identifier] if key == "participant" else partition_risk[identifier]
            )
            target["conditional_crossing_window_bound"] += int(
                block_summary_record["conditional_hidden_join_crossing_window_bound"]
            )
            target["window_count"] += int(block_summary_record["window_count"])
    participant_bound_records: list[dict[str, Any]] = [
        {
            **counts,
            "conditional_crossing_window_fraction_bound": counts[
                "conditional_crossing_window_bound"
            ]
            / counts["window_count"],
            "subject_id": subject,
        }
        for subject, counts in sorted(participant_risk.items(), key=lambda item: int(item[0]))
    ]
    partition_bound_records: list[dict[str, Any]] = [
        {
            **counts,
            "conditional_crossing_window_fraction_bound": counts[
                "conditional_crossing_window_bound"
            ]
            / counts["window_count"],
            "partition": partition,
        }
        for partition, counts in sorted(partition_risk.items())
    ]
    worst_participant = max(
        participant_bound_records,
        key=lambda item: (
            float(item["conditional_crossing_window_fraction_bound"]),
            -int(str(item["subject_id"])),
        ),
    )
    worst_block_fraction = max(
        int(block["conditional_hidden_join_crossing_window_bound"]) / int(block["window_count"])
        for block in blocks
    )
    payload: dict[str, Any] = {
        "authorization": {
            "record_path": Path(authorization_path).as_posix(),
            "record_sha256": authorization_hash,
            "status": authorization["status"],
        },
        "block_summaries": blocks,
        "dataset_id": "inclusivehar_v4",
        "few_person_outer_folds": few_person,
        "hidden_join_risk": {
            "conditional_assumption": "exactly three hidden contiguous repetitions and no additional joins per released block",
            "conditional_crossing_window_bound": conditional_crossing_bound,
            "conditional_crossing_window_fraction_bound": conditional_crossing_bound
            / total_windows,
            "conditional_participant_bounds": participant_bound_records,
            "conditional_partition_bounds": partition_bound_records,
            "conditional_worst_block_crossing_window_fraction_bound": worst_block_fraction,
            "conditional_worst_participant": worst_participant,
            "formula": "sum_b min(2, floor(block_rows_b / 128)) / total_windows",
            "unconditional_crossing_window_fraction_bound": 1.0,
        },
        "label_window_counts": {
            track: dict(sorted(counts.items())) for track, counts in sorted(label_counts.items())
        },
        "manifest_kind": "released_block_split",
        "model_input_policy": {
            "allowed_channels_exact_order": list(preprocessing["channels"]),
            "forbidden_inputs": list(preprocessing["forbidden_model_inputs"]),
            "global_rows_and_window_offsets_are_provenance_only": True,
        },
        "normalization_policy": {
            "few_person_reuses_source_only_statistics": True,
            "fit_partition": "source_train",
            "target_excluded": True,
            "validation_excluded": True,
        },
        "ontology": {
            "config_path": Path(ontology_config_path).as_posix(),
            "config_sha256": ontology.config_sha256,
            "ontology_id": ontology.ontology_id,
        },
        "partitions": [
            _partition_summary(
                windows,
                subjects=spec.final_source_train,
                partition="source_train",
            ),
            _partition_summary(
                windows,
                subjects=spec.final_source_validation,
                partition="source_validation",
            ),
            _partition_summary(
                windows,
                subjects=spec.target_subjects,
                partition="target_sealed",
            ),
        ],
        "preprocessing": {
            "config_path": Path(preprocessing_config_path).as_posix(),
            "config_sha256": preprocessing_hash,
            "preprocessing_id": preprocessing["preprocessing_id"],
        },
        "protocol": {
            "config_path": Path(protocol_config_path).as_posix(),
            "config_sha256": spec.config_sha256,
            "protocol_id": spec.protocol_id,
            "status": "locked_conditional_user_authorized_deviation",
        },
        "remainder_accounting": {
            "dropped_tail_rows": sum(int(block["dropped_tail_rows"]) for block in blocks),
            "policy": "drop_tail_shorter_than_128_per_released_block",
            "source_data_rows": sum(int(block["row_count"]) for block in blocks),
            "used_window_rows": sum(int(block["used_rows"]) for block in blocks),
        },
        "schema_version": "1.0.0",
        "source_cv_folds": source_cv,
        "source_evidence": {
            "audit_file_sha256": spec.stage3_audit_file_sha256,
            "audit_path": Path(audit_report_path).as_posix(),
            "audit_report_sha256": audit_hash,
            "sensor_artifact_sha256": source_hash,
        },
        "subject_assignment_before_windowing": True,
        "target_performance_or_prediction_accessed": False,
        "target_seal": {
            "maximum_confirmatory_openings": 1,
            "performance_inspection": "forbidden",
            "seal_id": target_seal_id,
            "status": "sealed",
            "subject_ids": list(spec.target_subjects),
            "unlock_record": None,
        },
        "window_count": total_windows,
        "window_id_set_sha256": canonical_json_sha256(sorted(window_ids)),
        "window_length_samples": spec.window_length,
        "window_stride_samples": spec.window_stride,
        "windows": windows,
        "trial_boundary_status": "unrecoverable",
        "window_duration_interpretation": "2.56 seconds only at the provider-declared, unverified 50 Hz rate",
    }
    if ontology.explicit_class_order:
        ontology_record = _mapping(payload["ontology"], name="ontology record")
        payload["ontology"] = {
            **ontology_record,
            "runnable_track_schemas": {
                track: ontology.runnable_track_schema(track)
                for track in ("functional_core", "inclusive_native")
            },
        }
    if source_nested:
        payload["source_nested_cv"] = source_nested
    if sum(partition_counts.values()) != total_windows:
        raise AssertionError("partition counts do not cover windows")
    payload["split_manifest_sha256"] = canonical_json_sha256(payload)
    return payload


def write_split_manifest_new(
    manifest: Mapping[str, Any],
    destination: str | Path,
    *,
    allowed_root: str | Path,
) -> Path:
    """Publish a split manifest once without replacement."""

    return atomic_write_json_new(manifest, destination, allowed_root=allowed_root)


def build_source_window_manifest(split_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the exact source-only materialization contract from a locked split."""

    claimed_hash = split_manifest.get("split_manifest_sha256")
    without_hash = dict(split_manifest)
    without_hash.pop("split_manifest_sha256", None)
    if claimed_hash != canonical_json_sha256(without_hash):
        raise ValueError("split manifest self-hash does not validate")
    target_seal = _mapping(split_manifest.get("target_seal"), name="target seal")
    if (
        target_seal.get("status") != "sealed"
        or target_seal.get("unlock_record") is not None
        or split_manifest.get("target_performance_or_prediction_accessed") is not False
    ):
        raise PermissionError("source manifest requires an unopened target seal")
    all_windows = _list_mapping_values(split_manifest.get("windows"), name="split windows")
    source_windows = [
        dict(window)
        for window in all_windows
        if window.get("partition") in {"source_train", "source_validation"}
    ]
    allowed_source_subjects = {str(value) for value in range(1, 11)}
    if not source_windows or any(
        str(window.get("subject_id")) not in allowed_source_subjects for window in source_windows
    ):
        raise ValueError("source window selection contains a non-source participant")
    source_partitions = [
        dict(_mapping(partition, name="source partition"))
        for partition in _list_mapping_values(
            split_manifest.get("partitions"), name="split partitions"
        )
        if partition.get("partition") in {"source_train", "source_validation"}
    ]
    window_ids = sorted(str(window["window_id"]) for window in source_windows)
    raw_interval_ids = sorted(str(window["raw_interval_id"]) for window in source_windows)
    payload: dict[str, Any] = {
        "dataset_id": split_manifest.get("dataset_id"),
        "manifest_kind": "source_development_windows",
        "model_input_policy": split_manifest.get("model_input_policy"),
        "normalization_policy": split_manifest.get("normalization_policy"),
        "ontology": split_manifest.get("ontology"),
        "partitions": source_partitions,
        "preprocessing": split_manifest.get("preprocessing"),
        "protocol": split_manifest.get("protocol"),
        "raw_interval_id_set_sha256": canonical_json_sha256(raw_interval_ids),
        "schema_version": "1.0.0",
        "source_artifact_sha256": _mapping(
            split_manifest.get("source_evidence"), name="source evidence"
        ).get("sensor_artifact_sha256"),
        "source_cv_folds": split_manifest.get("source_cv_folds"),
        "source_nested_cv": split_manifest.get("source_nested_cv"),
        "source_split_manifest_sha256": claimed_hash,
        "source_window_count": len(source_windows),
        "target_subject_or_window_records_included": False,
        "target_performance_or_prediction_accessed": False,
        "trial_boundary_status": "unrecoverable",
        "window_id_set_sha256": canonical_json_sha256(window_ids),
        "window_length_samples": split_manifest.get("window_length_samples"),
        "window_stride_samples": split_manifest.get("window_stride_samples"),
        "windows": source_windows,
    }
    payload["source_window_manifest_sha256"] = canonical_json_sha256(payload)
    return payload


def _list_mapping_values(value: Any, *, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return [_mapping(item, name=f"{name} item") for item in value]


def write_source_window_manifest_new(
    manifest: Mapping[str, Any],
    destination: str | Path,
    *,
    allowed_root: str | Path,
) -> Path:
    """Publish a source-only materialization manifest without replacement."""

    return atomic_write_json_new(manifest, destination, allowed_root=allowed_root)
