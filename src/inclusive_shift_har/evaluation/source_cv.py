"""Fail-closed aggregation of source-only grouped cross-validation evidence.

This module intentionally consumes only the source-only window manifest and
development summaries.  It never reads prediction arrays, raw signals, the
full split manifest, or any target-cohort record.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np

from inclusive_shift_har.evaluation.statistics import participant_bootstrap_interval
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)


class SourceCVAggregationError(ValueError):
    """Raised when source-development evidence is incomplete or inconsistent."""


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceCVAggregationError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _list(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise SourceCVAggregationError(f"{name} must be a JSON array")
    return value


def _string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SourceCVAggregationError(f"{name} must be a non-empty string")
    return value


def _finite_float(value: Any, *, name: str, lower: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SourceCVAggregationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SourceCVAggregationError(f"{name} must be a finite number")
    if lower is not None and result < lower:
        raise SourceCVAggregationError(f"{name} must be at least {lower}")
    return result


def _probability_metric(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise SourceCVAggregationError(f"{name} must lie in [0, 1]")
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SourceCVAggregationError(f"{name} must be a positive integer")
    return cast(int, value)


def _string_list(value: Any, *, name: str) -> list[str]:
    values = _list(value, name=name)
    parsed = [_string(item, name=f"{name}[{index}]") for index, item in enumerate(values)]
    if len(parsed) != len(set(parsed)):
        raise SourceCVAggregationError(f"{name} contains duplicate identifiers")
    return parsed


def _validate_self_hash(record: Mapping[str, Any], *, path: Path) -> str:
    recorded = _string(record.get("record_sha256"), name=f"{path}.record_sha256")
    unhashed = dict(record)
    unhashed.pop("record_sha256", None)
    actual = canonical_json_sha256(unhashed)
    if recorded != actual:
        raise SourceCVAggregationError(
            f"record_sha256 mismatch for {path}: recorded {recorded}, computed {actual}"
        )
    return actual


def _require_source_only(record: Mapping[str, Any], *, path: Path) -> None:
    if record.get("target_performance_or_prediction_accessed") is not False:
        raise SourceCVAggregationError(
            f"{path} does not explicitly attest that target performance/predictions were unopened"
        )
    if record.get("target_subject_or_window_records_loaded") is not False:
        raise SourceCVAggregationError(
            f"{path} does not explicitly attest that target records were not loaded"
        )
    if record.get("evidence_status") != "source_development_not_confirmatory":
        raise SourceCVAggregationError(
            f"{path} is not labelled source_development_not_confirmatory"
        )


def _safe_source_fold_contract(source_manifest_path: Path) -> dict[str, Any]:
    parsed = _mapping(load_json_strict(source_manifest_path), name="source window manifest")
    if parsed.get("target_subject_or_window_records_included") is not False:
        raise SourceCVAggregationError(
            "source manifest is target-bearing or lacks a false target flag"
        )
    if parsed.get("target_performance_or_prediction_accessed") is not False:
        raise SourceCVAggregationError(
            "source manifest does not attest that target evidence is sealed"
        )

    embedded_hash = _string(
        parsed.get("source_window_manifest_sha256"),
        name="source_window_manifest_sha256",
    )
    unhashed = dict(parsed)
    unhashed.pop("source_window_manifest_sha256", None)
    computed_hash = canonical_json_sha256(unhashed)
    if embedded_hash != computed_hash:
        raise SourceCVAggregationError(
            "source-only manifest embedded hash does not match its canonical content"
        )

    folds: dict[str, dict[str, list[str]]] = {}
    validation_occurrences: list[str] = []
    for index, raw_fold in enumerate(_list(parsed.get("source_cv_folds"), name="source_cv_folds")):
        fold = _mapping(raw_fold, name=f"source_cv_folds[{index}]")
        fold_id = _string(fold.get("fold_id"), name=f"source_cv_folds[{index}].fold_id")
        if fold_id in folds:
            raise SourceCVAggregationError(f"duplicate source fold id: {fold_id}")
        train = _string_list(
            fold.get("train_subjects"), name=f"source_cv_folds[{index}].train_subjects"
        )
        validation = _string_list(
            fold.get("validation_subjects"),
            name=f"source_cv_folds[{index}].validation_subjects",
        )
        if set(train) & set(validation):
            raise SourceCVAggregationError(f"source participant overlap in {fold_id}")
        folds[fold_id] = {"train": train, "validation": validation}
        validation_occurrences.extend(validation)

    if len(folds) < 2:
        raise SourceCVAggregationError("at least two grouped source folds are required")
    counts = Counter(validation_occurrences)
    duplicates = sorted(participant for participant, count in counts.items() if count != 1)
    if duplicates:
        raise SourceCVAggregationError(
            f"source manifest validation participants do not occur exactly once: {duplicates}"
        )
    expected = sorted(
        counts, key=lambda item: (not item.isdecimal(), int(item) if item.isdecimal() else item)
    )
    expected_set = set(expected)
    for fold_id, split in folds.items():
        if set(split["train"]) != expected_set - set(split["validation"]):
            raise SourceCVAggregationError(
                f"{fold_id} training participants are not the complement of its validation group"
            )
    return {
        "embedded_hash": embedded_hash,
        "file_sha256": sha256_file(source_manifest_path),
        "folds": folds,
        "expected_participants": expected,
    }


def derive_model_identity(configuration: Mapping[str, Any]) -> dict[str, str]:
    """Derive semantic and configuration-specific identifiers from a run config."""

    model_name = _string(configuration.get("model_name"), name="configuration.model_name")
    coral_weight = _finite_float(
        configuration.get("coral_weight", 0.0),
        name="configuration.coral_weight",
    )
    coral_token = str(coral_weight).replace("-", "m").replace(".", "p")
    if model_name == "dann_compact_residual_96":
        dann_weight = _finite_float(
            configuration.get("dann_domain_loss_weight"),
            name="configuration.dann_domain_loss_weight",
            lower=0.0,
        )
        grl_max = _finite_float(
            configuration.get("dann_grl_max_strength"),
            name="configuration.dann_grl_max_strength",
            lower=0.0,
        )
        warmup = _positive_int(
            configuration.get("dann_grl_warmup_epochs"),
            name="configuration.dann_grl_warmup_epochs",
        )
        if dann_weight <= 0.0 or not 0.0 < grl_max <= 1.0:
            raise SourceCVAggregationError(
                "DANN loss weight and maximum GRL strength must be positive, with GRL <= 1"
            )
        dann_token = str(dann_weight).replace("-", "m").replace(".", "p")
        grl_token = str(grl_max).replace("-", "m").replace(".", "p")
        variant_name = f"established_lambda_{dann_token}_grl_{grl_token}_warmup_{warmup}"
    elif model_name != "more_har":
        variant_name = f"coral_{coral_token}" if coral_weight > 0.0 else model_name
    else:
        flags = {
            "augmentation": configuration.get("use_augmentation") is True,
            "content": configuration.get("use_content_objective") is True,
            "factorization": configuration.get("use_realization_factorization") is True,
            "groupdro": configuration.get("use_group_dro") is True,
        }
        enabled = [name for name, active in flags.items() if active]
        if coral_weight > 0.0:
            enabled.append(f"coral_{coral_token}")
        if not enabled:
            variant_name = "backbone"
        elif enabled == ["augmentation", "content", "factorization", "groupdro"]:
            variant_name = "full"
        else:
            variant_name = "_plus_".join(enabled)
    zeroed = configuration.get("zero_channel_indices", [])
    if zeroed:
        if not isinstance(zeroed, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in zeroed
        ):
            raise SourceCVAggregationError(
                "configuration.zero_channel_indices must contain non-negative integers"
            )
        suffix = "_zero_channels_" + "_".join(str(value) for value in sorted(set(zeroed)))
        variant_name += suffix

    model_variant_id = model_name if variant_name == model_name else f"{model_name}_{variant_name}"
    configuration_sha256 = canonical_json_sha256(dict(configuration))
    return {
        "model_family_id": model_name,
        "variant_name": variant_name,
        "model_variant_id": model_variant_id,
        "configuration_id": f"{model_variant_id}--{configuration_sha256[:12]}",
        "configuration_sha256": configuration_sha256,
    }


def _validated_development_origins(source_development_dir: Path) -> dict[str, dict[str, str]]:
    matches: dict[str, list[dict[str, str]]] = defaultdict(list)
    for path in sorted(source_development_dir.glob("*.json")):
        record = _mapping(load_json_strict(path), name=str(path))
        record_hash = _validate_self_hash(record, path=path)
        _require_source_only(record, path=path)
        configuration = _mapping(record.get("configuration"), name=f"{path}.configuration")
        computed_config_hash = canonical_json_sha256(dict(configuration))
        recorded_config_hash = _string(
            record.get("configuration_sha256"), name=f"{path}.configuration_sha256"
        )
        if computed_config_hash != recorded_config_hash:
            raise SourceCVAggregationError(f"configuration_sha256 mismatch for {path}")
        matches[recorded_config_hash].append(
            {"path": path.as_posix(), "record_sha256": record_hash}
        )
    duplicates = sorted(key for key, records in matches.items() if len(records) > 1)
    if duplicates:
        raise SourceCVAggregationError(
            f"multiple source-development origins share configuration hashes: {duplicates}"
        )
    return {key: records[0] for key, records in matches.items()}


def _validate_fold_record(
    path: Path,
    *,
    fold_contract: Mapping[str, Mapping[str, list[str]]],
    expected_source_manifest_sha256: str,
) -> dict[str, Any]:
    record = _mapping(load_json_strict(path), name=str(path))
    record_hash = _validate_self_hash(record, path=path)
    _require_source_only(record, path=path)

    source_manifest_hash = _string(
        record.get("source_window_manifest_sha256"),
        name=f"{path}.source_window_manifest_sha256",
    )
    if source_manifest_hash != expected_source_manifest_sha256:
        raise SourceCVAggregationError(f"source manifest lineage mismatch for {path}")
    fold_id = _string(record.get("source_split_id"), name=f"{path}.source_split_id")
    if fold_id not in fold_contract:
        raise SourceCVAggregationError(f"unknown source fold {fold_id!r} in {path}")
    expected_fold = fold_contract[fold_id]
    train = _string_list(record.get("train_participants"), name=f"{path}.train_participants")
    validation = _string_list(
        record.get("validation_participants"), name=f"{path}.validation_participants"
    )
    if set(train) != set(expected_fold["train"]):
        raise SourceCVAggregationError(f"training participant mismatch for {path}")
    if set(validation) != set(expected_fold["validation"]):
        raise SourceCVAggregationError(f"validation participant mismatch for {path}")

    configuration = _mapping(record.get("configuration"), name=f"{path}.configuration")
    identity = derive_model_identity(configuration)
    recorded_config_hash = _string(
        record.get("configuration_sha256"), name=f"{path}.configuration_sha256"
    )
    if recorded_config_hash != identity["configuration_sha256"]:
        raise SourceCVAggregationError(f"configuration_sha256 mismatch for {path}")
    if record.get("model_name") != identity["model_family_id"]:
        raise SourceCVAggregationError(f"model name/configuration mismatch for {path}")
    if record.get("seed") != configuration.get("seed"):
        raise SourceCVAggregationError(f"seed/configuration mismatch for {path}")

    report = _mapping(record.get("validation_report"), name=f"{path}.validation_report")
    participants: list[dict[str, Any]] = []
    participant_ids: list[str] = []
    for index, raw_participant in enumerate(
        _list(report.get("participants"), name=f"{path}.validation_report.participants")
    ):
        row = _mapping(raw_participant, name=f"{path}.participants[{index}]")
        participant_id = _string(
            row.get("participant_id"), name=f"{path}.participants[{index}].participant_id"
        )
        participant_ids.append(participant_id)
        participants.append(
            {
                "participant_id": participant_id,
                "window_count": _positive_int(
                    row.get("window_count"), name=f"{path}.participants[{index}].window_count"
                ),
                "macro_f1": _probability_metric(
                    row.get("macro_f1"), name=f"{path}.participants[{index}].macro_f1"
                ),
                "balanced_accuracy": _probability_metric(
                    row.get("balanced_accuracy"),
                    name=f"{path}.participants[{index}].balanced_accuracy",
                ),
            }
        )
    if len(participant_ids) != len(set(participant_ids)):
        raise SourceCVAggregationError(f"duplicate participant report row in {path}")
    if set(participant_ids) != set(validation):
        raise SourceCVAggregationError(f"participant report/validation mismatch for {path}")
    sample_count = _positive_int(report.get("sample_count"), name=f"{path}.sample_count")
    if sample_count != sum(int(row["window_count"]) for row in participants):
        raise SourceCVAggregationError(f"participant window counts do not sum in {path}")
    if record.get("validation_window_count") != sample_count:
        raise SourceCVAggregationError(f"top-level validation window count mismatch for {path}")

    primary = _mapping(report.get("primary"), name=f"{path}.validation_report.primary")
    macro_values = np.asarray([row["macro_f1"] for row in participants], dtype=np.float64)
    reported_mean = _probability_metric(
        primary.get("mean_participant_macro_f1"), name=f"{path}.primary.mean"
    )
    reported_worst = _probability_metric(
        primary.get("worst_participant_macro_f1"), name=f"{path}.primary.worst"
    )
    reported_lower = _probability_metric(
        primary.get("lower_decile_participant_macro_f1"), name=f"{path}.primary.lower_decile"
    )
    if not math.isclose(reported_mean, float(macro_values.mean()), abs_tol=1e-12):
        raise SourceCVAggregationError(f"participant macro-F1 mean mismatch for {path}")
    if not math.isclose(reported_worst, float(macro_values.min()), abs_tol=1e-12):
        raise SourceCVAggregationError(f"participant macro-F1 worst mismatch for {path}")
    if not math.isclose(
        reported_lower,
        float(np.quantile(macro_values, 0.1, method="linear")),
        abs_tol=1e-12,
    ):
        raise SourceCVAggregationError(f"participant macro-F1 lower decile mismatch for {path}")

    calibration = _mapping(report.get("calibration"), name=f"{path}.validation_report.calibration")
    window_diagnostics = _mapping(
        report.get("window_level_diagnostics"),
        name=f"{path}.validation_report.window_level_diagnostics",
    )
    best_epoch_raw = record.get("best_epoch")
    best_epoch = (
        None if best_epoch_raw is None else _positive_int(best_epoch_raw, name=f"{path}.best_epoch")
    )
    return {
        "path": path.as_posix(),
        "record_sha256": record_hash,
        "fold_id": fold_id,
        "identity": identity,
        "configuration": dict(configuration),
        "code_commit": _string(record.get("code_commit"), name=f"{path}.code_commit"),
        "class_names": _string_list(record.get("class_names"), name=f"{path}.class_names"),
        "seed": record.get("seed"),
        "parameter_count": record.get("parameter_count"),
        "best_epoch": best_epoch,
        "sample_count": sample_count,
        "participants": participants,
        "fold_mean_participant_macro_f1": reported_mean,
        "fold_mean_participant_balanced_accuracy": float(
            np.mean([row["balanced_accuracy"] for row in participants])
        ),
        "fold_window_balanced_accuracy": _probability_metric(
            window_diagnostics.get("balanced_accuracy"),
            name=f"{path}.window_level_diagnostics.balanced_accuracy",
        ),
        "negative_log_likelihood": _finite_float(
            calibration.get("negative_log_likelihood"),
            name=f"{path}.calibration.negative_log_likelihood",
            lower=0.0,
        ),
        "multiclass_brier_score": _finite_float(
            calibration.get("multiclass_brier_score"),
            name=f"{path}.calibration.multiclass_brier_score",
            lower=0.0,
        ),
        "ece": _probability_metric(calibration.get("ece"), name=f"{path}.calibration.ece"),
    }


def _distribution_summary(
    values_by_participant: Mapping[str, float],
    *,
    bootstrap_resamples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    values = np.asarray(
        [values_by_participant[key] for key in sorted(values_by_participant)],
        dtype=np.float64,
    )
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "standard_deviation": float(values.std(ddof=1)),
        "worst": float(values.min()),
        "lower_decile": float(np.quantile(values, 0.1, method="linear")),
        "bootstrap_mean_interval": participant_bootstrap_interval(
            values_by_participant,
            resamples=bootstrap_resamples,
            confidence=bootstrap_confidence,
            seed=bootstrap_seed,
        ),
    }


def _fold_scalar_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def build_source_cv_aggregate(
    *,
    fold_record_dir: str | os.PathLike[str],
    source_development_dir: str | os.PathLike[str],
    source_manifest_path: str | os.PathLike[str],
    bootstrap_resamples: int = 10_000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 1729,
) -> dict[str, Any]:
    """Validate and aggregate complete grouped CV configurations.

    All source participants must be held out exactly once for every aggregated
    configuration.  A configuration is identified by its canonical hash, not a
    filename or user-supplied display label.
    """

    if bootstrap_resamples < 100:
        raise SourceCVAggregationError("bootstrap_resamples must be at least 100")
    if not 0.0 < bootstrap_confidence < 1.0:
        raise SourceCVAggregationError("bootstrap_confidence must lie in (0, 1)")
    records_root = Path(fold_record_dir)
    development_root = Path(source_development_dir)
    manifest_path = Path(source_manifest_path)
    if not records_root.is_dir() or not development_root.is_dir():
        raise SourceCVAggregationError("source CV and source-development roots must exist")

    contract = _safe_source_fold_contract(manifest_path)
    fold_contract = cast(dict[str, Mapping[str, list[str]]], contract["folds"])
    source_origins = _validated_development_origins(development_root)
    record_paths = sorted(records_root.glob("*-source_cv_??.json"))
    if not record_paths:
        raise SourceCVAggregationError("no source CV fold records matched *-source_cv_??.json")
    validated = [
        _validate_fold_record(
            path,
            fold_contract=fold_contract,
            expected_source_manifest_sha256=cast(str, contract["embedded_hash"]),
        )
        for path in record_paths
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in validated:
        identity = cast(Mapping[str, str], record["identity"])
        grouped[identity["configuration_sha256"]].append(record)

    expected_fold_ids = set(fold_contract)
    expected_participants = cast(list[str], contract["expected_participants"])
    models: list[dict[str, Any]] = []
    quarantined_configurations: list[dict[str, Any]] = []
    input_rows: list[dict[str, str]] = [
        {
            "path": cast(str, record["path"]),
            "record_sha256": cast(str, record["record_sha256"]),
            "configuration_id": cast(Mapping[str, str], record["identity"])["configuration_id"],
            "fold_id": cast(str, record["fold_id"]),
        }
        for record in validated
    ]
    for configuration_sha256, records in grouped.items():
        identity = cast(dict[str, str], records[0]["identity"])
        observed_fold_ids = [cast(str, record["fold_id"]) for record in records]
        duplicate_folds = sorted(
            fold for fold, count in Counter(observed_fold_ids).items() if count != 1
        )
        missing_folds = sorted(expected_fold_ids - set(observed_fold_ids))
        extra_folds = sorted(set(observed_fold_ids) - expected_fold_ids)
        quarantine_reasons: list[dict[str, Any]] = []
        if duplicate_folds or set(observed_fold_ids) != expected_fold_ids:
            quarantine_reasons.append(
                {
                    "code": "INCOMPLETE_OR_DUPLICATE_FOLDS",
                    "duplicate_folds": duplicate_folds,
                    "missing_folds": missing_folds,
                    "extra_folds": extra_folds,
                }
            )
        if configuration_sha256 not in source_origins:
            quarantine_reasons.append(
                {
                    "code": "MISSING_EXACT_SOURCE_DEVELOPMENT_ORIGIN",
                    "configuration_sha256": configuration_sha256,
                }
            )
        if any(record["identity"] != identity for record in records):
            quarantine_reasons.append({"code": "INCONSISTENT_MODEL_IDENTITY"})
        for field in ("configuration", "class_names", "seed", "parameter_count"):
            if any(record[field] != records[0][field] for record in records[1:]):
                quarantine_reasons.append({"code": "INCONSISTENT_FOLD_PROVENANCE", "field": field})
        code_commits = sorted({cast(str, record["code_commit"]) for record in records})
        if len(code_commits) != 1:
            quarantine_reasons.append({"code": "MIXED_CODE_COMMITS", "code_commits": code_commits})
        if quarantine_reasons:
            quarantined_configurations.append(
                {
                    **identity,
                    "status": "quarantined_not_aggregated",
                    "reasons": quarantine_reasons,
                    "observed_fold_ids": sorted(observed_fold_ids),
                    "record_count": len(records),
                    "records": [
                        {
                            "path": record["path"],
                            "record_sha256": record["record_sha256"],
                            "fold_id": record["fold_id"],
                            "code_commit": record["code_commit"],
                        }
                        for record in sorted(records, key=lambda item: cast(str, item["fold_id"]))
                    ],
                }
            )
            continue

        records.sort(key=lambda record: cast(str, record["fold_id"]))
        participant_rows: list[dict[str, Any]] = []
        for record in records:
            for participant in cast(list[dict[str, Any]], record["participants"]):
                participant_rows.append({"fold_id": record["fold_id"], **participant})
        participant_counts = Counter(
            cast(str, participant["participant_id"]) for participant in participant_rows
        )
        duplicates = sorted(
            participant for participant, count in participant_counts.items() if count != 1
        )
        missing = sorted(set(expected_participants) - set(participant_counts))
        extra = sorted(set(participant_counts) - set(expected_participants))
        if duplicates or missing or extra:
            quarantined_configurations.append(
                {
                    **identity,
                    "status": "quarantined_not_aggregated",
                    "reasons": [
                        {
                            "code": "PARTICIPANT_COVERAGE_NOT_EXACTLY_ONCE",
                            "duplicate_participants": duplicates,
                            "missing_participants": missing,
                            "extra_participants": extra,
                        }
                    ],
                    "observed_fold_ids": sorted(observed_fold_ids),
                    "record_count": len(records),
                    "records": [
                        {
                            "path": record["path"],
                            "record_sha256": record["record_sha256"],
                            "fold_id": record["fold_id"],
                            "code_commit": record["code_commit"],
                        }
                        for record in records
                    ],
                }
            )
            continue
        participant_rows.sort(
            key=lambda row: (
                not cast(str, row["participant_id"]).isdecimal(),
                int(cast(str, row["participant_id"]))
                if cast(str, row["participant_id"]).isdecimal()
                else cast(str, row["participant_id"]),
            )
        )
        macro_by_participant = {
            cast(str, row["participant_id"]): cast(float, row["macro_f1"])
            for row in participant_rows
        }
        balanced_by_participant = {
            cast(str, row["participant_id"]): cast(float, row["balanced_accuracy"])
            for row in participant_rows
        }
        weights = np.asarray([record["sample_count"] for record in records], dtype=np.float64)
        weights /= weights.sum()
        nll = np.asarray([record["negative_log_likelihood"] for record in records])
        brier = np.asarray([record["multiclass_brier_score"] for record in records])
        ece = np.asarray([record["ece"] for record in records])
        epochs = [record["best_epoch"] for record in records]
        if any(epoch is None for epoch in epochs) and not all(epoch is None for epoch in epochs):
            quarantined_configurations.append(
                {
                    **identity,
                    "status": "quarantined_not_aggregated",
                    "reasons": [{"code": "MIXED_BEST_EPOCH_APPLICABILITY"}],
                    "observed_fold_ids": sorted(observed_fold_ids),
                    "record_count": len(records),
                    "records": [
                        {
                            "path": record["path"],
                            "record_sha256": record["record_sha256"],
                            "fold_id": record["fold_id"],
                            "code_commit": record["code_commit"],
                        }
                        for record in records
                    ],
                }
            )
            continue
        epoch_values = [cast(int, epoch) for epoch in epochs if epoch is not None]

        fold_rows = [
            {
                "fold_id": record["fold_id"],
                "record_path": record["path"],
                "record_sha256": record["record_sha256"],
                "validation_participants": sorted(
                    cast(str, row["participant_id"])
                    for row in cast(list[dict[str, Any]], record["participants"])
                ),
                "validation_window_count": record["sample_count"],
                "mean_participant_macro_f1": record["fold_mean_participant_macro_f1"],
                "mean_participant_balanced_accuracy": record[
                    "fold_mean_participant_balanced_accuracy"
                ],
                "window_balanced_accuracy": record["fold_window_balanced_accuracy"],
                "negative_log_likelihood": record["negative_log_likelihood"],
                "multiclass_brier_score": record["multiclass_brier_score"],
                "ece": record["ece"],
                "best_epoch": record["best_epoch"],
            }
            for record in records
        ]
        model = {
            **identity,
            "configuration": records[0]["configuration"],
            "source_development_origin": source_origins[configuration_sha256],
            "code_commits": code_commits,
            "seed": records[0]["seed"],
            "parameter_count": records[0]["parameter_count"],
            "class_names": records[0]["class_names"],
            "fold_count": len(records),
            "participant_count": len(participant_rows),
            "validation_window_count": int(sum(record["sample_count"] for record in records)),
            "participant_metrics": {
                "macro_f1": _distribution_summary(
                    macro_by_participant,
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_confidence=bootstrap_confidence,
                    bootstrap_seed=bootstrap_seed,
                ),
                "balanced_accuracy": _distribution_summary(
                    balanced_by_participant,
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_confidence=bootstrap_confidence,
                    bootstrap_seed=bootstrap_seed,
                ),
            },
            "fold_metrics": {
                "mean_participant_macro_f1": _fold_scalar_summary(
                    [cast(float, record["fold_mean_participant_macro_f1"]) for record in records]
                ),
                "mean_participant_balanced_accuracy": _fold_scalar_summary(
                    [
                        cast(float, record["fold_mean_participant_balanced_accuracy"])
                        for record in records
                    ]
                ),
                "window_balanced_accuracy": _fold_scalar_summary(
                    [cast(float, record["fold_window_balanced_accuracy"]) for record in records]
                ),
            },
            "calibration": {
                "negative_log_likelihood": {
                    "sample_weighted_fold_aggregate": float(np.sum(weights * nll)),
                    "fold_summary": _fold_scalar_summary(nll.tolist()),
                },
                "multiclass_brier_score": {
                    "sample_weighted_fold_aggregate": float(np.sum(weights * brier)),
                    "fold_summary": _fold_scalar_summary(brier.tolist()),
                },
                "ece": {
                    "sample_weighted_fold_mean": float(np.sum(weights * ece)),
                    "fold_summary": _fold_scalar_summary(ece.tolist()),
                    "status": "descriptive_fold_aggregate_not_pooled_ece",
                },
                "note": "NLL and Brier are additive sample-weighted fold aggregates; ECE is not additive and is reported only as a descriptive fold aggregate.",
            },
            "best_epoch": {
                "applicable": bool(epoch_values),
                "values_by_fold": {
                    cast(str, record["fold_id"]): record["best_epoch"] for record in records
                },
                "median": float(statistics.median(epoch_values)) if epoch_values else None,
                "minimum": min(epoch_values) if epoch_values else None,
                "maximum": max(epoch_values) if epoch_values else None,
            },
            "participants": participant_rows,
            "folds": fold_rows,
        }
        models.append(model)

    models.sort(key=lambda model: cast(str, model["configuration_id"]))
    quarantined_configurations.sort(key=lambda item: cast(str, item["configuration_id"]))
    input_rows.sort(key=lambda row: (row["configuration_id"], row["fold_id"]))
    if not models:
        raise SourceCVAggregationError(
            "no complete, exact-lineage source CV configuration passed aggregation gates"
        )
    aggregate: dict[str, Any] = {
        "schema_version": "1.0.0",
        "artifact_type": "source_only_grouped_cv_aggregate",
        "status": (
            "complete_source_development_target_sealed_with_quarantine"
            if quarantined_configurations
            else "complete_source_development_target_sealed"
        ),
        "evidence_status": "source_development_not_confirmatory",
        "selection_status": "no_model_ranking_selection_or_freeze_performed",
        "source_only_guarantees": {
            "target_performance_or_prediction_accessed": False,
            "target_subject_or_window_records_loaded": False,
            "raw_signals_or_prediction_arrays_read_by_aggregator": False,
            "aggregation_unit": "participant",
            "expected_source_participants": expected_participants,
            "each_participant_held_out_once_per_aggregated_configuration": True,
        },
        "source_manifest": {
            "path": manifest_path.as_posix(),
            "source_window_manifest_sha256": contract["embedded_hash"],
            "file_sha256": contract["file_sha256"],
        },
        "bootstrap": {
            "method": "percentile_participant_cluster_bootstrap",
            "resamples": bootstrap_resamples,
            "confidence": bootstrap_confidence,
            "seed": bootstrap_seed,
            "shared_resample_indices_across_models": True,
        },
        "input_fold_record_count": len(input_rows),
        "input_fold_records": input_rows,
        "configuration_count": len(models),
        "models": models,
        "quarantined_configuration_count": len(quarantined_configurations),
        "quarantined_configurations": quarantined_configurations,
        "interpretation_notes": [
            "All values are source-development evidence and are not confirmatory target results.",
            "Hyperparameters have source_v1 development origins; this aggregate is not presented as a fully nested unbiased model-comparison estimate.",
            "Windows are never treated as independent statistical units for uncertainty.",
            "Models are ordered by stable configuration identifier, not performance.",
        ],
    }
    aggregate["aggregate_record_sha256"] = canonical_json_sha256(aggregate)
    return aggregate


_CSV_FIELDS = (
    "configuration_id",
    "model_variant_id",
    "participant_count",
    "fold_count",
    "mean_participant_macro_f1",
    "macro_f1_ci_lower",
    "macro_f1_ci_upper",
    "worst_participant_macro_f1",
    "lower_decile_participant_macro_f1",
    "mean_participant_balanced_accuracy",
    "balanced_accuracy_ci_lower",
    "balanced_accuracy_ci_upper",
    "worst_participant_balanced_accuracy",
    "lower_decile_participant_balanced_accuracy",
    "fold_mean_macro_f1_mean",
    "fold_mean_macro_f1_standard_deviation",
    "negative_log_likelihood",
    "multiclass_brier_score",
    "ece_descriptive_weighted_fold_mean",
    "best_epoch_median",
)


def render_source_cv_csv(aggregate: Mapping[str, Any]) -> str:
    """Render one compact, deterministic row per validated configuration."""

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for raw_model in _list(aggregate.get("models"), name="aggregate.models"):
        model = _mapping(raw_model, name="aggregate.models[]")
        participant_metrics = _mapping(
            model.get("participant_metrics"), name="model.participant_metrics"
        )
        macro = _mapping(participant_metrics.get("macro_f1"), name="model.macro_f1")
        macro_ci = _mapping(
            _mapping(macro.get("bootstrap_mean_interval"), name="model.macro_f1.bootstrap"),
            name="model.macro_f1.bootstrap",
        )
        balanced = _mapping(
            participant_metrics.get("balanced_accuracy"), name="model.balanced_accuracy"
        )
        balanced_ci = _mapping(
            balanced.get("bootstrap_mean_interval"), name="model.balanced_accuracy.bootstrap"
        )
        fold_metrics = _mapping(model.get("fold_metrics"), name="model.fold_metrics")
        fold_macro = _mapping(
            fold_metrics.get("mean_participant_macro_f1"), name="model.fold_macro"
        )
        calibration = _mapping(model.get("calibration"), name="model.calibration")
        nll = _mapping(calibration.get("negative_log_likelihood"), name="model.nll")
        brier = _mapping(calibration.get("multiclass_brier_score"), name="model.brier")
        ece = _mapping(calibration.get("ece"), name="model.ece")
        best_epoch = _mapping(model.get("best_epoch"), name="model.best_epoch")
        writer.writerow(
            {
                "configuration_id": model["configuration_id"],
                "model_variant_id": model["model_variant_id"],
                "participant_count": model["participant_count"],
                "fold_count": model["fold_count"],
                "mean_participant_macro_f1": macro["mean"],
                "macro_f1_ci_lower": macro_ci["lower"],
                "macro_f1_ci_upper": macro_ci["upper"],
                "worst_participant_macro_f1": macro["worst"],
                "lower_decile_participant_macro_f1": macro["lower_decile"],
                "mean_participant_balanced_accuracy": balanced["mean"],
                "balanced_accuracy_ci_lower": balanced_ci["lower"],
                "balanced_accuracy_ci_upper": balanced_ci["upper"],
                "worst_participant_balanced_accuracy": balanced["worst"],
                "lower_decile_participant_balanced_accuracy": balanced["lower_decile"],
                "fold_mean_macro_f1_mean": fold_macro["mean"],
                "fold_mean_macro_f1_standard_deviation": fold_macro["standard_deviation"],
                "negative_log_likelihood": nll["sample_weighted_fold_aggregate"],
                "multiclass_brier_score": brier["sample_weighted_fold_aggregate"],
                "ece_descriptive_weighted_fold_mean": ece["sample_weighted_fold_mean"],
                "best_epoch_median": best_epoch["median"],
            }
        )
    return output.getvalue()


def render_source_cv_markdown(aggregate: Mapping[str, Any]) -> str:
    """Render an explicitly non-confirmatory compact Markdown evidence table."""

    lines = [
        "# Source-only grouped cross-validation summary",
        "",
        "Status: **source development; not confirmatory**. No target participant record, "
        "prediction, or performance value was read. Rows are sorted by configuration ID and "
        "do not constitute a model selection or freeze.",
        "",
        "| Configuration | Mean participant macro-F1 (95% cluster bootstrap CI) | Worst | "
        "Lower decile | Mean participant balanced accuracy | NLL | Brier | Best epoch median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for raw_model in _list(aggregate.get("models"), name="aggregate.models"):
        model = _mapping(raw_model, name="aggregate.models[]")
        participant_metrics = _mapping(
            model.get("participant_metrics"), name="model.participant_metrics"
        )
        macro = _mapping(participant_metrics.get("macro_f1"), name="model.macro_f1")
        macro_ci = _mapping(macro.get("bootstrap_mean_interval"), name="model.macro_f1.bootstrap")
        balanced = _mapping(
            participant_metrics.get("balanced_accuracy"), name="model.balanced_accuracy"
        )
        calibration = _mapping(model.get("calibration"), name="model.calibration")
        nll = _mapping(calibration.get("negative_log_likelihood"), name="model.nll")
        brier = _mapping(calibration.get("multiclass_brier_score"), name="model.brier")
        best_epoch = _mapping(model.get("best_epoch"), name="model.best_epoch")
        epoch_text = "n/a" if best_epoch["median"] is None else f"{best_epoch['median']:g}"
        lines.append(
            f"| `{model['configuration_id']}` | {macro['mean']:.4f} "
            f"[{macro_ci['lower']:.4f}, {macro_ci['upper']:.4f}] | {macro['worst']:.4f} | "
            f"{macro['lower_decile']:.4f} | {balanced['mean']:.4f} | "
            f"{nll['sample_weighted_fold_aggregate']:.4f} | "
            f"{brier['sample_weighted_fold_aggregate']:.4f} | {epoch_text} |"
        )
    lines.extend(
        [
            "",
            "NLL and Brier are sample-weighted across disjoint grouped folds. ECE remains a "
            "secondary descriptive fold aggregate because ECE is not additive. Uncertainty "
            "resamples participants, never overlapping windows.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_bytes_new(
    payload: bytes,
    destination: Path,
    *,
    allowed_root: Path,
) -> Path:
    root = allowed_root.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = destination.parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(root)
    except ValueError as exc:
        raise SourceCVAggregationError(f"output escapes allowed root: {destination}") from exc
    target = resolved_parent / destination.name
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite existing evidence: {target}")
    temporary = resolved_parent / f".{target.name}.partial.{uuid4().hex}"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    except Exception as exc:
        raise OSError(
            f"publication failed; partial evidence retained at {temporary}: {exc}"
        ) from exc
    else:
        temporary.unlink()
    return target


def write_source_cv_exports_new(
    aggregate: Mapping[str, Any],
    *,
    output_dir: str | os.PathLike[str],
    prefix: str = "source_cv_aggregate",
) -> dict[str, Path]:
    """Publish canonical JSON, CSV, and Markdown without replacing prior evidence."""

    if not prefix or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in prefix
    ):
        raise SourceCVAggregationError(
            "output prefix may contain only lowercase letters, digits, underscore, and hyphen"
        )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    csv_text = render_source_cv_csv(aggregate)
    markdown_text = render_source_cv_markdown(aggregate)
    payload = dict(aggregate)
    payload.pop("aggregate_record_sha256", None)
    payload["exports"] = {
        "csv": {
            "path": f"{prefix}.csv",
            "sha256": hashlib.sha256(csv_text.encode("utf-8")).hexdigest(),
        },
        "markdown": {
            "path": f"{prefix}.md",
            "sha256": hashlib.sha256(markdown_text.encode("utf-8")).hexdigest(),
        },
    }
    payload["aggregate_record_sha256"] = canonical_json_sha256(payload)
    csv_path = _atomic_write_bytes_new(
        csv_text.encode("utf-8"), root / f"{prefix}.csv", allowed_root=root
    )
    markdown_path = _atomic_write_bytes_new(
        markdown_text.encode("utf-8"), root / f"{prefix}.md", allowed_root=root
    )
    json_path = atomic_write_json_new(
        payload,
        root / f"{prefix}.json",
        allowed_root=root,
    )
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}
