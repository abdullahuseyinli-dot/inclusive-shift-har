"""Strict loader for the conditionally locked released-block protocol."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from inclusive_shift_har.manifests.canonical import canonical_json_sha256


@dataclass(frozen=True)
class SourceCVFold:
    fold_id: str
    validation_subjects: tuple[str, ...]


@dataclass(frozen=True)
class FewPersonOuterFold:
    fold_id: str
    evaluation_subjects: tuple[str, ...]
    inclusion_order: tuple[str, ...]


@dataclass(frozen=True)
class ProtocolSpec:
    protocol_id: str
    config_path: str
    config_sha256: str
    raw: Mapping[str, Any]
    source_subjects: tuple[str, ...]
    target_subjects: tuple[str, ...]
    final_source_train: tuple[str, ...]
    final_source_validation: tuple[str, ...]
    source_cv_folds: tuple[SourceCVFold, ...]
    few_person_outer_folds: tuple[FewPersonOuterFold, ...]
    few_person_k_values: tuple[int, ...]
    window_length: int
    window_stride: int
    conditional_hidden_joins: int
    stage3_audit_report_sha256: str
    stage3_audit_file_sha256: str
    nested_source_cv: bool
    explicit_class_schema_required: bool


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _string_tuple(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must be a list of non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} contains duplicate subjects")
    return tuple(value)


def _validate_participant_assignments(spec: ProtocolSpec) -> None:
    source = set(spec.source_subjects)
    target = set(spec.target_subjects)
    if source != {str(value) for value in range(1, 11)}:
        raise ValueError("source participants must be the locked released IDs 1-10")
    if target != {str(value) for value in range(11, 21)}:
        raise ValueError("target participants must be the locked released IDs 11-20")
    if source & target:
        raise ValueError("source and target participants overlap")
    final_train = set(spec.final_source_train)
    final_validation = set(spec.final_source_validation)
    if final_train & final_validation or final_train | final_validation != source:
        raise ValueError("final source train/validation must be disjoint and cover source subjects")

    source_validation_counts: Counter[str] = Counter()
    for source_fold in spec.source_cv_folds:
        validation = set(source_fold.validation_subjects)
        if len(validation) != 2 or not validation <= source:
            raise ValueError(f"{source_fold.fold_id} must validate on two source subjects")
        source_validation_counts.update(validation)
    if source_validation_counts != Counter({subject: 1 for subject in source}):
        raise ValueError("source CV validation subjects must cover each source exactly once")

    target_evaluation_counts: Counter[str] = Counter()
    for target_fold in spec.few_person_outer_folds:
        evaluation = set(target_fold.evaluation_subjects)
        inclusion = set(target_fold.inclusion_order)
        if len(evaluation) != 2 or not evaluation <= target:
            raise ValueError(f"{target_fold.fold_id} must evaluate two target subjects")
        if evaluation & inclusion or evaluation | inclusion != target:
            raise ValueError(
                f"{target_fold.fold_id} inclusion order must contain every non-evaluation target once"
            )
        if len(target_fold.inclusion_order) != len(inclusion):
            raise ValueError(f"{target_fold.fold_id} inclusion order contains duplicates")
        target_evaluation_counts.update(evaluation)
    if target_evaluation_counts != Counter({subject: 1 for subject in target}):
        raise ValueError("few-person evaluation folds must cover each target exactly once")


def _sha256_subject_order(subjects: tuple[str, ...], token_template: str) -> tuple[str, ...]:
    if token_template.count("{subject_id}") != 1:
        raise ValueError("fold token_template must contain one {subject_id} placeholder")
    ranked = sorted(
        (
            hashlib.sha256(token_template.format(subject_id=subject).encode("utf-8")).hexdigest(),
            subject,
        )
        for subject in subjects
    )
    return tuple(subject for _, subject in ranked)


def _validate_fold_derivation(spec: ProtocolSpec) -> None:
    source_derivation = _mapping(
        spec.raw.get("source_fold_derivation"), name="source_fold_derivation"
    )
    if source_derivation.get("algorithm_version") != "sha256-adjacent-pairs-v1":
        raise ValueError("source fold derivation version changed")
    source_token = source_derivation.get("token_template")
    if not isinstance(source_token, str):
        raise ValueError("source fold token_template must be a string")
    source_order = _sha256_subject_order(spec.source_subjects, source_token)
    expected_source_pairs = tuple(
        frozenset(source_order[index : index + 2]) for index in range(0, len(source_order), 2)
    )
    observed_source_pairs = tuple(
        frozenset(fold.validation_subjects) for fold in spec.source_cv_folds
    )
    if observed_source_pairs != expected_source_pairs:
        raise ValueError("source folds do not match the locked SHA-256 derivation")
    if frozenset(spec.final_source_validation) != expected_source_pairs[-1]:
        raise ValueError("final source validation is not the locked last source fold")

    target_derivation = _mapping(
        spec.raw.get("target_fold_derivation"), name="target_fold_derivation"
    )
    if target_derivation.get("algorithm_version") != "sha256-adjacent-pairs-with-prefix-order-v1":
        raise ValueError("target fold derivation version changed")
    target_token = target_derivation.get("token_template")
    if not isinstance(target_token, str):
        raise ValueError("target fold token_template must be a string")
    target_order = _sha256_subject_order(spec.target_subjects, target_token)
    expected_target_pairs = tuple(
        frozenset(target_order[index : index + 2]) for index in range(0, len(target_order), 2)
    )
    observed_target_pairs = tuple(
        frozenset(fold.evaluation_subjects) for fold in spec.few_person_outer_folds
    )
    if observed_target_pairs != expected_target_pairs:
        raise ValueError("target outer folds do not match the locked SHA-256 derivation")
    for target_fold in spec.few_person_outer_folds:
        expected_inclusion = tuple(
            subject for subject in target_order if subject not in target_fold.evaluation_subjects
        )
        if target_fold.inclusion_order != expected_inclusion:
            raise ValueError(
                f"{target_fold.fold_id} inclusion order differs from the locked hash order"
            )


def load_protocol_spec(path: str | Path) -> ProtocolSpec:
    source = Path(path)
    parsed = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(parsed, name="protocol root")
    if root.get("schema_version") != "1.0.0":
        raise ValueError("protocol schema_version must equal '1.0.0'")
    if root.get("status") != "locked_conditional_user_authorized_deviation":
        raise ValueError("protocol is not conditionally locked")
    if root.get("dataset_id") != "inclusivehar_v4":
        raise ValueError("protocol dataset_id must equal 'inclusivehar_v4'")
    protocol_id = root.get("protocol_id")
    if protocol_id not in {
        "inclusivehar-released-block-v1",
        "inclusivehar-released-block-v1.1",
        "inclusivehar-released-block-v1.2",
    }:
        raise ValueError("protocol_id is not a supported locked released-block version")
    nested_source_cv = protocol_id in {
        "inclusivehar-released-block-v1.1",
        "inclusivehar-released-block-v1.2",
    }
    explicit_class_schema_required = protocol_id == "inclusivehar-released-block-v1.2"
    if explicit_class_schema_required and root.get("ontology_schema_policy") != (
        "Require an explicit ordered numeric class schema for every runnable track."
    ):
        raise ValueError("protocol v1.2 must require explicit numeric class schemas")
    if (
        _mapping(root.get("selection"), name="selection").get(
            "target_labels_or_performance_allowed"
        )
        is not False
    ):
        raise ValueError("target performance must remain forbidden for selection")
    if nested_source_cv:
        if (
            _mapping(root.get("selection"), name="selection").get(
                "source_nested_outer_test_allowed_for_selection"
            )
            is not False
        ):
            raise ValueError("source nested outer-test results must remain forbidden for selection")
        nested_policy = _mapping(root.get("source_nested_cv_policy"), name="nested policy")
        if nested_policy.get("outer_test_allowed_for_selection") is not False:
            raise ValueError("nested source outer-test folds may not be used for selection")

    windowing = _mapping(root.get("windowing"), name="windowing")
    length = windowing.get("length_samples")
    stride = windowing.get("stride_samples")
    if length != 128 or stride != 128:
        raise ValueError("locked protocol requires length=stride=128")
    if windowing.get("partition_subjects_before_windowing") is not True:
        raise ValueError("participants must be assigned before window generation")
    if windowing.get("remainder_policy") != ("drop_tail_shorter_than_128_per_released_block"):
        raise ValueError("remainder policy changed")
    if windowing.get("raw_overlap_allowed") is not False:
        raise ValueError("raw overlap must remain forbidden")

    participants = _mapping(root.get("participants"), name="participants")
    final_split = _mapping(root.get("final_source_split"), name="final_source_split")
    source_folds_raw = root.get("source_grouped_cv")
    target_folds_raw = root.get("few_person_outer_folds")
    if not isinstance(source_folds_raw, list) or not isinstance(target_folds_raw, list):
        raise ValueError("source and target folds must be lists")
    source_folds = tuple(
        SourceCVFold(
            fold_id=str(_mapping(item, name="source fold")["fold_id"]),
            validation_subjects=_string_tuple(
                _mapping(item, name="source fold").get("validation_subjects"),
                name="source validation subjects",
            ),
        )
        for item in source_folds_raw
    )
    target_folds = tuple(
        FewPersonOuterFold(
            fold_id=str(_mapping(item, name="target fold")["fold_id"]),
            evaluation_subjects=_string_tuple(
                _mapping(item, name="target fold").get("evaluation_subjects"),
                name="target evaluation subjects",
            ),
            inclusion_order=_string_tuple(
                _mapping(item, name="target fold").get("inclusion_order"),
                name="target inclusion order",
            ),
        )
        for item in target_folds_raw
    )
    k_values_raw = root.get("few_person_k_values")
    if k_values_raw != [1, 2, 4]:
        raise ValueError("few-person k values must remain [1, 2, 4]")
    hidden_join = _mapping(root.get("hidden_join_risk"), name="hidden_join_risk")
    if hidden_join.get("unconditional_bound_without_join_metadata") != 1.0:
        raise ValueError("unconditional hidden-join contamination bound must remain 1.0")
    conditional_joins = hidden_join.get("conditional_hidden_joins_per_released_block")
    if conditional_joins != 2:
        raise ValueError("reported three-repeat assumption implies two conditional joins")

    zero_shot = _mapping(root.get("zero_shot_target"), name="zero_shot_target")
    target_subjects = _string_tuple(participants.get("target_subjects"), name="target_subjects")
    if _string_tuple(zero_shot.get("subjects"), name="zero-shot subjects") != target_subjects:
        raise ValueError("zero-shot target must equal the complete target cohort")
    if (
        zero_shot.get("status") != "sealed"
        or zero_shot.get("selection_inputs_allowed") is not False
    ):
        raise ValueError("zero-shot target seal is open or permits selection")

    normalization = _mapping(root.get("normalization"), name="normalization")
    if (
        normalization.get("fit_partition") != "source_train"
        or normalization.get("validation_excluded") is not True
        or normalization.get("target_excluded") is not True
    ):
        raise ValueError("normalization is not source-training-only")

    spec = ProtocolSpec(
        protocol_id=str(protocol_id),
        config_path=source.as_posix(),
        config_sha256=canonical_json_sha256(root),
        raw=root,
        source_subjects=_string_tuple(participants.get("source_subjects"), name="source_subjects"),
        target_subjects=target_subjects,
        final_source_train=_string_tuple(
            final_split.get("train_subjects"), name="final source train"
        ),
        final_source_validation=_string_tuple(
            final_split.get("validation_subjects"), name="final source validation"
        ),
        source_cv_folds=source_folds,
        few_person_outer_folds=target_folds,
        few_person_k_values=tuple(int(value) for value in k_values_raw),
        window_length=int(length),
        window_stride=int(stride),
        conditional_hidden_joins=int(conditional_joins),
        stage3_audit_report_sha256=str(root["stage3_audit_report_sha256"]),
        stage3_audit_file_sha256=str(root["stage3_audit_file_sha256"]),
        nested_source_cv=nested_source_cv,
        explicit_class_schema_required=explicit_class_schema_required,
    )
    _validate_participant_assignments(spec)
    _validate_fold_derivation(spec)
    return spec
