"""Create a descriptive source-versus-target participant performance-gap report.

This module deliberately consumes only three strict JSON records: the source
grouped-CV aggregate, the locked target participant statistics, and an explicit
model-to-source-configuration specification.  It never opens prediction arrays
or raw sensor data.  The resulting comparison is descriptive because the source
and target records were produced under different seed and training regimes.
"""

from __future__ import annotations

import csv
import os
import re
from collections.abc import Mapping, Sequence
from io import StringIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)

COHORT_GAP_SCHEMA_VERSION = "1.0.0"
COHORT_GAP_EVIDENCE_STATUS = "descriptive_post_confirmatory_cross_cohort_comparison"
SOURCE_PARTICIPANT_IDS = tuple(str(value) for value in range(1, 11))
TARGET_PARTICIPANT_IDS = tuple(str(value) for value in range(11, 21))
TARGET_SEED_ORDER = (11, 23, 47, 89, 131)
SOURCE_SEED = 11
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 1729
FINAL_MODEL_IDS = (
    "compact-coral",
    "compact-dann",
    "compact-erm",
    "deepconvlstm",
    "legacy-bilstm",
    "legacy-cnn1d",
    "legacy-joint-cnn-bilstm",
    "logistic-regression",
    "more-har-augmentation",
    "more-har-backbone",
    "more-har-content",
    "more-har-factorized",
    "more-har-full",
    "more-har-full-no-accelerometer",
    "more-har-full-no-gyroscope",
    "more-har-groupdro",
    "random-forest",
    "static-dual-branch-matched",
    "svm-rbf",
    "xgboost",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CohortGapError(RuntimeError):
    """Raised when cross-cohort evidence is incomplete, changed, or inconsistent."""


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CohortGapError(f"{name} must be an object")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise CohortGapError(f"{name} must be an array")
    return value


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CohortGapError(f"{name} must be a non-empty string")
    return value


def _sha256(value: Any, *, name: str) -> str:
    digest = _text(value, name=name)
    if _SHA256.fullmatch(digest) is None:
        raise CohortGapError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CohortGapError(f"{name} must be an integer")
    return int(value)


def _number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CohortGapError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise CohortGapError(f"{name} must be a finite number")
    return result


def _strings(value: Any, *, name: str) -> tuple[str, ...]:
    rows = _sequence(value, name=name)
    result = tuple(_text(row, name=f"{name} item") for row in rows)
    if len(set(result)) != len(result):
        raise CohortGapError(f"{name} contains duplicates")
    return result


def _validate_self_hash(record: Mapping[str, Any], *, field: str, role: str) -> str:
    body = dict(record)
    claimed = _sha256(body.pop(field, None), name=f"{role} {field}")
    if claimed != canonical_json_sha256(body):
        raise CohortGapError(f"{role} self-hash does not validate")
    return claimed


def _resolve_existing_json(value: str | Path, *, root: Path, role: str) -> Path:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise CohortGapError(f"{role} escapes the repository root") from exc
    if resolved.is_symlink() or not resolved.is_file() or resolved.suffix.lower() != ".json":
        raise CohortGapError(f"{role} is missing, non-regular, a symlink, or not JSON")
    return resolved


def _resolve_new(value: str | Path, *, root: Path, role: str) -> Path:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    candidate.parent.mkdir(parents=True, exist_ok=True)
    parent = candidate.parent.resolve(strict=True)
    try:
        parent.relative_to(root)
    except ValueError as exc:
        raise CohortGapError(f"{role} escapes the repository root") from exc
    resolved = parent / candidate.name
    if os.path.lexists(resolved):
        raise FileExistsError(f"refusing to overwrite existing {role}: {resolved}")
    return resolved


def _write_text_new(text: str, destination: Path) -> None:
    temporary = destination.parent / f".{destination.name}.partial.{uuid4().hex}"
    try:
        with temporary.open("xb") as handle:
            handle.write(text.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    except Exception as exc:
        raise OSError(
            f"atomic text publication failed; partial retained at {temporary}: {exc}"
        ) from exc
    else:
        temporary.unlink()


def _csv_text(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _assert_close(observed: float, expected: Any, *, name: str) -> None:
    stored = _number(expected, name=name)
    if not np.isclose(observed, stored, atol=1e-12, rtol=1e-12):
        raise CohortGapError(f"{name} differs from participant-level reconstruction")


def _participant_summary(values: Mapping[str, float]) -> dict[str, Any]:
    array = np.asarray(list(values.values()), dtype=np.float64)
    return {
        "participant_values": dict(values),
        "mean_macro_f1": float(np.mean(array)),
        "worst_macro_f1": float(np.min(array)),
        "lower_decile_macro_f1": float(np.quantile(array, 0.1, method="linear")),
    }


def _source_models(
    source: Mapping[str, Any],
    *,
    expected_participants: tuple[str, ...],
) -> dict[str, Mapping[str, Any]]:
    if source.get("artifact_type") != "source_only_grouped_cv_aggregate":
        raise CohortGapError("source artifact type is not the grouped-CV aggregate")
    if source.get("status") != "complete_source_development_target_sealed":
        raise CohortGapError("source aggregate status is not complete with target sealed")
    if source.get("evidence_status") != "source_development_not_confirmatory":
        raise CohortGapError("source aggregate evidence status is not development-only")
    if source.get("selection_status") != "no_model_ranking_selection_or_freeze_performed":
        raise CohortGapError("source aggregate selection-status lineage changed")
    if _integer(source.get("quarantined_configuration_count"), name="source quarantine count") != 0:
        raise CohortGapError("source aggregate contains quarantined configurations")
    guarantees = _mapping(source.get("source_only_guarantees"), name="source guarantees")
    if guarantees.get("aggregation_unit") != "participant":
        raise CohortGapError("source aggregation unit is not participant")
    if guarantees.get("each_participant_held_out_once_per_aggregated_configuration") is not True:
        raise CohortGapError("source participants were not each held out exactly once")
    for field in (
        "raw_signals_or_prediction_arrays_read_by_aggregator",
        "target_performance_or_prediction_accessed",
        "target_subject_or_window_records_loaded",
    ):
        if guarantees.get(field) is not False:
            raise CohortGapError(f"source guarantee {field} is not false")
    guaranteed_participants = _strings(
        guarantees.get("expected_source_participants"), name="source guaranteed participants"
    )
    if guaranteed_participants != expected_participants:
        raise CohortGapError("source aggregate does not declare the exact source participant set")

    model_rows = _sequence(source.get("models"), name="source models")
    if len(model_rows) != _integer(
        source.get("configuration_count"), name="source configuration count"
    ):
        raise CohortGapError("source model count differs from its declaration")
    result: dict[str, Mapping[str, Any]] = {}
    for row_value in model_rows:
        row = _mapping(row_value, name="source model row")
        configuration_id = _text(row.get("configuration_id"), name="source configuration ID")
        if configuration_id in result:
            raise CohortGapError(f"duplicate source configuration ID: {configuration_id}")
        result[configuration_id] = row
    return result


def _source_participants(
    row: Mapping[str, Any],
    *,
    configuration_id: str,
    expected_participants: tuple[str, ...],
) -> dict[str, float]:
    if _integer(row.get("seed"), name=f"{configuration_id} source seed") != SOURCE_SEED:
        raise CohortGapError(f"{configuration_id} does not use source seed {SOURCE_SEED}")
    if _integer(row.get("participant_count"), name=f"{configuration_id} participant count") != len(
        expected_participants
    ):
        raise CohortGapError(f"{configuration_id} source participant count changed")
    participant_rows = _sequence(row.get("participants"), name=f"{configuration_id} participants")
    values: dict[str, float] = {}
    for participant_value in participant_rows:
        participant = _mapping(participant_value, name=f"{configuration_id} participant row")
        participant_id = _text(
            participant.get("participant_id"), name=f"{configuration_id} participant ID"
        )
        if participant_id in values:
            raise CohortGapError(
                f"{configuration_id} duplicates source participant {participant_id}"
            )
        values[participant_id] = _number(
            participant.get("macro_f1"), name=f"{configuration_id} participant macro-F1"
        )
    if set(values) != set(expected_participants) or len(values) != len(expected_participants):
        raise CohortGapError(
            f"{configuration_id} does not contain the exact source participant set"
        )
    ordered = {participant_id: values[participant_id] for participant_id in expected_participants}
    summary = _participant_summary(ordered)
    stored = _mapping(
        _mapping(row.get("participant_metrics"), name=f"{configuration_id} source metrics").get(
            "macro_f1"
        ),
        name=f"{configuration_id} source macro-F1 summary",
    )
    _assert_close(summary["mean_macro_f1"], stored.get("mean"), name=f"{configuration_id} mean")
    _assert_close(summary["worst_macro_f1"], stored.get("worst"), name=f"{configuration_id} worst")
    _assert_close(
        summary["lower_decile_macro_f1"],
        stored.get("lower_decile"),
        name=f"{configuration_id} lower decile",
    )
    return ordered


def _target_models(
    target: Mapping[str, Any],
    *,
    expected_participants: tuple[str, ...],
    expected_seeds: tuple[int, ...],
) -> dict[str, Mapping[str, Any]]:
    if target.get("record_kind") != "locked_target_participant_statistics":
        raise CohortGapError("target record kind is not locked participant statistics")
    if target.get("status") != "complete_create_only":
        raise CohortGapError("target participant statistics are not complete/create-only")
    if target.get("evidence_status") != "locked_confirmatory_target_opening_1":
        raise CohortGapError("target evidence is not locked confirmatory opening 1")
    if target.get("target_information_used_for_model_selection") is not False:
        raise CohortGapError("target information was marked as used for model selection")
    if target.get("statistical_unit") != "participant":
        raise CohortGapError("target statistical unit is not participant")
    if _integer(target.get("participant_count"), name="target participant count") != len(
        expected_participants
    ):
        raise CohortGapError("target participant count changed")
    seeds = tuple(
        _integer(seed, name="target required seed")
        for seed in _sequence(target.get("required_seed_order"), name="target seed order")
    )
    if seeds != expected_seeds:
        raise CohortGapError("target required seed order changed")
    model_rows = _sequence(target.get("models"), name="target models")
    result: dict[str, Mapping[str, Any]] = {}
    for row_value in model_rows:
        row = _mapping(row_value, name="target model row")
        model_id = _text(row.get("model_id"), name="target model ID")
        if model_id in result:
            raise CohortGapError(f"duplicate target model ID: {model_id}")
        result[model_id] = row
    if set(result) != set(FINAL_MODEL_IDS) or len(result) != len(FINAL_MODEL_IDS):
        raise CohortGapError("target statistics do not contain the exact 20 final model IDs")
    return result


def _target_participants(
    row: Mapping[str, Any],
    *,
    model_id: str,
    expected_participants: tuple[str, ...],
    expected_seeds: tuple[int, ...],
) -> dict[str, float]:
    seeds = tuple(
        _integer(seed, name=f"{model_id} seed")
        for seed in _sequence(row.get("seed_order"), name=f"{model_id} seed order")
    )
    if seeds != expected_seeds:
        raise CohortGapError(f"{model_id} target seed order changed")
    seed_rows = _sequence(row.get("seed_level_primary"), name=f"{model_id} seed rows")
    if (
        tuple(
            _integer(_mapping(value, name=f"{model_id} seed row").get("seed"), name="seed")
            for value in seed_rows
        )
        != expected_seeds
    ):
        raise CohortGapError(f"{model_id} target seed rows changed")
    source_result_hashes = _sequence(
        row.get("source_result_record_sha256s"), name=f"{model_id} source result hashes"
    )
    if len(source_result_hashes) != len(expected_seeds):
        raise CohortGapError(f"{model_id} target lineage does not have one source result per seed")
    for index, digest in enumerate(source_result_hashes):
        _sha256(digest, name=f"{model_id} source result hash {index}")

    summary_record = _mapping(
        row.get("participant_seed_averaged"), name=f"{model_id} target participant summary"
    )
    stored_values = _mapping(
        summary_record.get("participant_values"), name=f"{model_id} target participant values"
    )
    values = {
        _text(participant_id, name=f"{model_id} target participant ID"): _number(
            metric, name=f"{model_id} target participant macro-F1"
        )
        for participant_id, metric in stored_values.items()
    }
    if set(values) != set(expected_participants) or len(values) != len(expected_participants):
        raise CohortGapError(f"{model_id} does not contain the exact target participant set")
    ordered = {participant_id: values[participant_id] for participant_id in expected_participants}
    summary = _participant_summary(ordered)
    _assert_close(
        summary["mean_macro_f1"],
        summary_record.get("mean_macro_f1"),
        name=f"{model_id} target mean",
    )
    _assert_close(
        summary["worst_macro_f1"],
        summary_record.get("worst_macro_f1"),
        name=f"{model_id} target worst",
    )
    _assert_close(
        summary["lower_decile_macro_f1"],
        summary_record.get("lower_decile_macro_f1"),
        name=f"{model_id} target lower decile",
    )
    return ordered


def _validate_specification(
    specification: Mapping[str, Any],
    *,
    source_file: Path,
    target_file: Path,
    specification_file: Path,
    root: Path,
) -> tuple[Mapping[str, str], Mapping[str, Any]]:
    if specification.get("record_kind") != "cross_cohort_gap_analysis_specification":
        raise CohortGapError("mapping specification record kind changed")
    if specification.get("status") != "locked_post_confirmatory_descriptive_analysis":
        raise CohortGapError("mapping specification status changed")
    if specification.get("target_information_used_to_choose_mapping") is not False:
        raise CohortGapError("mapping was marked as chosen using target information")
    expected_source = _strings(
        specification.get("source_participant_ids"), name="specified source participants"
    )
    expected_target = _strings(
        specification.get("target_participant_ids"), name="specified target participants"
    )
    if expected_source != SOURCE_PARTICIPANT_IDS or expected_target != TARGET_PARTICIPANT_IDS:
        raise CohortGapError("specification does not lock the exact 10+10 participant sets")
    if _integer(specification.get("source_seed"), name="specified source seed") != SOURCE_SEED:
        raise CohortGapError("specification source seed changed")
    specified_seeds = tuple(
        _integer(seed, name="specified target seed")
        for seed in _sequence(specification.get("target_seed_order"), name="specified target seeds")
    )
    if specified_seeds != TARGET_SEED_ORDER:
        raise CohortGapError("specification target seed order changed")

    bootstrap = _mapping(specification.get("bootstrap"), name="bootstrap specification")
    if bootstrap.get("method") != "independent_participant_cluster_percentile":
        raise CohortGapError("bootstrap method is not the predeclared independent-cohort method")
    if _number(bootstrap.get("confidence"), name="bootstrap confidence") != 0.95:
        raise CohortGapError("bootstrap confidence changed")
    if _integer(bootstrap.get("resamples"), name="bootstrap resamples") != BOOTSTRAP_RESAMPLES:
        raise CohortGapError("bootstrap resample count changed")
    if _integer(bootstrap.get("seed"), name="bootstrap seed") != BOOTSTRAP_SEED:
        raise CohortGapError("bootstrap seed changed")
    if bootstrap.get("independent_cohort_draws") is not True:
        raise CohortGapError("bootstrap source and target draws are not declared independent")
    if bootstrap.get("shared_indices_across_models") is not True:
        raise CohortGapError("bootstrap indices are not declared shared across models")

    inputs = _mapping(specification.get("inputs"), name="specified inputs")
    for role, actual_file in (("source", source_file), ("target", target_file)):
        declared = _mapping(inputs.get(role), name=f"specified {role} input")
        actual_relative = actual_file.relative_to(root).as_posix()
        if declared.get("path") != actual_relative:
            raise CohortGapError(f"{role} input path differs from the locked specification")
        if _sha256(declared.get("file_sha256"), name=f"specified {role} file hash") != sha256_file(
            actual_file
        ):
            raise CohortGapError(f"{role} input file hash differs from the locked specification")

    raw_mapping = _mapping(
        specification.get("model_to_source_configuration"), name="model configuration mapping"
    )
    model_mapping = {
        _text(model_id, name="mapped model ID"): _text(
            configuration_id, name=f"{model_id} source configuration ID"
        )
        for model_id, configuration_id in raw_mapping.items()
    }
    if tuple(sorted(model_mapping)) != FINAL_MODEL_IDS:
        raise CohortGapError("mapping does not contain the exact 20 final model IDs")
    if len(set(model_mapping.values())) != len(FINAL_MODEL_IDS):
        raise CohortGapError("mapping source configuration IDs are not one-to-one")
    if specification_file == source_file or specification_file == target_file:
        raise CohortGapError("mapping specification must be a distinct JSON input")
    return model_mapping, bootstrap


def _validate_input_lineage(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    specification: Mapping[str, Any],
    *,
    root: Path,
    source_file: Path,
    source_self_hash: str,
    target_self_hash: str,
    model_mapping: Mapping[str, str],
    source_models: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    inputs = _mapping(specification.get("inputs"), name="specified inputs")
    source_input = _mapping(inputs.get("source"), name="specified source input")
    target_input = _mapping(inputs.get("target"), name="specified target input")
    if source_input.get("aggregate_record_sha256") != source_self_hash:
        raise CohortGapError("source aggregate self-hash differs from the mapping lineage")
    if target_input.get("record_sha256") != target_self_hash:
        raise CohortGapError("target statistics self-hash differs from the mapping lineage")

    lineage = _mapping(specification.get("lineage"), name="specified lineage")
    source_manifest = _mapping(source.get("source_manifest"), name="source manifest lineage")
    if source_manifest.get("source_window_manifest_sha256") != lineage.get(
        "source_window_manifest_sha256"
    ):
        raise CohortGapError("source window-manifest lineage changed")
    target_checks = {
        "locked_target_index_record_sha256": target.get("locked_target_index_record_sha256"),
        "final_freeze_inventory_sha256": target.get("final_freeze_inventory_sha256"),
        "target_seal_id": target.get("target_seal_id"),
        "analysis_plan_sha256": target.get("analysis_plan_sha256"),
    }
    for field, observed in target_checks.items():
        if _sha256(observed, name=f"target {field}") != _sha256(
            lineage.get(field), name=f"specified {field}"
        ):
            raise CohortGapError(f"target {field} lineage changed")
    mapping_basis = _mapping(lineage.get("mapping_basis"), name="mapping-basis lineage")
    mapping_basis_path = _resolve_existing_json(
        _text(mapping_basis.get("path"), name="mapping-basis path"),
        root=root,
        role="mapping-basis selection plan",
    )
    if sha256_file(mapping_basis_path) != _sha256(
        mapping_basis.get("file_sha256"), name="mapping-basis file hash"
    ):
        raise CohortGapError("mapping-basis selection-plan file hash changed")
    selection_plan = _mapping(
        load_json_strict(mapping_basis_path), name="mapping-basis selection plan"
    )
    selection_plan_hash = _validate_self_hash(
        selection_plan,
        field="selection_plan_sha256",
        role="mapping-basis selection plan",
    )
    if selection_plan_hash != _sha256(
        mapping_basis.get("selection_plan_sha256"), name="mapping-basis record hash"
    ):
        raise CohortGapError("mapping-basis selection-plan record hash changed")
    _validate_mapping_against_selection_plan(
        selection_plan,
        source_file=source_file,
        source_self_hash=source_self_hash,
        model_mapping=model_mapping,
        source_models=source_models,
        root=root,
    )
    return lineage


def _validate_mapping_against_selection_plan(
    selection_plan: Mapping[str, Any],
    *,
    source_file: Path,
    source_self_hash: str,
    model_mapping: Mapping[str, str],
    source_models: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> None:
    """Derive the model mapping from the source-only pre-opening selection plan."""

    required = {
        "record_kind": "final_source_selection_plan",
        "status": "predeclared_source_only_target_sealed",
        "target_subject_or_window_records_used": False,
        "target_predictions_or_performance_accessed": False,
        "final_epoch_rule": "lower integer median of five grouped source-fold best epochs",
    }
    mismatches = [
        field for field, expected in required.items() if selection_plan.get(field) != expected
    ]
    if mismatches:
        raise CohortGapError(f"mapping-basis selection-plan contract changed: {mismatches}")
    declared_source_path = _resolve_existing_json(
        _text(
            selection_plan.get("source_selection_aggregate_path"),
            name="selection-plan source aggregate path",
        ),
        root=root,
        role="selection-plan source aggregate",
    )
    if declared_source_path != source_file:
        raise CohortGapError("selection plan does not bind the supplied source aggregate path")
    if selection_plan.get("source_selection_aggregate_record_sha256") != source_self_hash:
        raise CohortGapError("selection plan does not bind the supplied source aggregate record")
    if tuple(selection_plan.get("required_seed_order", ())) != TARGET_SEED_ORDER:
        raise CohortGapError("mapping-basis required seed order changed")

    common = _mapping(
        selection_plan.get("common_neural_runner_arguments"),
        name="selection-plan common neural arguments",
    )
    plan_rows = _sequence(selection_plan.get("models"), name="selection-plan models")
    plan_by_id: dict[str, Mapping[str, Any]] = {}
    for raw_row in plan_rows:
        row = _mapping(raw_row, name="selection-plan model")
        model_id = _text(row.get("model_id"), name="selection-plan model ID")
        if model_id in plan_by_id:
            raise CohortGapError(f"selection plan duplicates model ID {model_id}")
        plan_by_id[model_id] = row
    if tuple(sorted(plan_by_id)) != FINAL_MODEL_IDS:
        raise CohortGapError("selection plan does not contain the exact 20 final model IDs")

    differentiating_fields = (
        "learning_rate",
        "weight_decay",
        "use_augmentation",
        "use_content_objective",
        "use_realization_factorization",
        "use_group_dro",
        "coral_weight",
        "dann_domain_loss_weight",
        "dann_grl_max_strength",
        "dann_grl_warmup_epochs",
        "zero_channel_indices",
    )
    optional_defaults: dict[str, Any] = {
        "use_augmentation": False,
        "use_content_objective": False,
        "use_realization_factorization": False,
        "use_group_dro": False,
        "coral_weight": 0.0,
        "dann_domain_loss_weight": 0.0,
        "dann_grl_max_strength": 1.0,
        "dann_grl_warmup_epochs": 10,
        "zero_channel_indices": [],
    }
    derived: dict[str, str] = {}
    for model_id in FINAL_MODEL_IDS:
        plan_row = plan_by_id[model_id]
        runner_name = _text(plan_row.get("runner_model_name"), name=f"{model_id} runner model name")
        regime = _text(plan_row.get("training_regime"), name=f"{model_id} training regime")
        runner_arguments = dict(
            _mapping(plan_row.get("runner_arguments"), name=f"{model_id} runner arguments")
        )
        expectations = _mapping(
            plan_row.get("configuration_expectations"),
            name=f"{model_id} configuration expectations",
        )
        if expectations.get("model_name") != runner_name:
            raise CohortGapError(f"{model_id} selection-plan model-name contract changed")
        expected_values: dict[str, Any] = {"model_name": runner_name, "seed": SOURCE_SEED}
        if regime == "fixed_epoch_neural":
            expected_values.update(common)
            expected_values.update(runner_arguments)
        elif regime == "deterministic_classical":
            if "xgboost_device" in expectations:
                expected_values["xgboost_device"] = expectations["xgboost_device"]
        else:
            raise CohortGapError(f"{model_id} has an unsupported selection-plan regime")

        candidates: list[str] = []
        for configuration_id, source_row in source_models.items():
            configuration = _mapping(
                source_row.get("configuration"), name=f"{configuration_id} source configuration"
            )
            identity_fields = ("model_name", "seed", "xgboost_device")
            if any(
                configuration.get(field) != expected
                for field, expected in expected_values.items()
                if field in identity_fields
            ):
                continue
            if regime == "fixed_epoch_neural":
                if any(
                    configuration.get(field, optional_defaults.get(field)) != expected_values[field]
                    for field in differentiating_fields
                    if field in expected_values
                ):
                    continue
                best_epoch = _mapping(
                    source_row.get("best_epoch"), name=f"{configuration_id} best epoch"
                )
                median = _number(
                    best_epoch.get("median"), name=f"{configuration_id} median best epoch"
                )
                planned_epochs = _integer(
                    runner_arguments.get("epochs"), name=f"{model_id} planned final epochs"
                )
                if median != float(planned_epochs):
                    continue
            candidates.append(configuration_id)
        if len(candidates) != 1:
            raise CohortGapError(
                f"selection plan derives {len(candidates)} source configurations for {model_id}"
            )
        derived[model_id] = candidates[0]
    if dict(model_mapping) != derived:
        raise CohortGapError("declared model mapping differs from the source-only selection plan")


def _markdown(models: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Descriptive source-target participant performance gaps",
        "",
        "Status: **descriptive post-confirmatory analysis, not a matched confirmatory or causal "
        "comparison**. Positive gaps mean the one-seed source grouped-CV mean is higher than the "
        "five-seed target participant average.",
        "",
        "| Model | Source mean | Target mean | Mean gap (95% independent-participant bootstrap CI) | Target worst | Source mean - target worst | Target lower decile | Source mean - target lower decile |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in models:
        source = _mapping(row["source_participant_macro_f1"], name="source summary")
        target = _mapping(row["target_participant_macro_f1"], name="target summary")
        gaps = _mapping(row["gaps"], name="gap summary")
        interval = _mapping(gaps["mean_gap_bootstrap_interval"], name="gap interval")
        lines.append(
            "| `{model}` | {source_mean:.4f} | {target_mean:.4f} | {gap:.4f} "
            "[{lower:.4f}, {upper:.4f}] | {target_worst:.4f} | {worst_gap:.4f} | "
            "{target_decile:.4f} | {decile_gap:.4f} |".format(
                model=row["model_id"],
                source_mean=source["mean_macro_f1"],
                target_mean=target["mean_macro_f1"],
                gap=gaps["source_mean_minus_target_mean_macro_f1"],
                lower=interval["lower"],
                upper=interval["upper"],
                target_worst=target["worst_macro_f1"],
                worst_gap=gaps["source_mean_minus_target_worst_macro_f1"],
                target_decile=target["lower_decile_macro_f1"],
                decile_gap=gaps["source_mean_minus_target_lower_decile_macro_f1"],
            )
        )
    lines.extend(
        [
            "",
            "The interval independently resamples the ten source and ten target participants "
            "10,000 times (seed 1729). It describes participant-resampling uncertainty within "
            "these fixed records; it does not make the regimes comparable.",
            "",
            "Limitations: source model/configuration selection is consumed; source values are "
            "from one-seed grouped cross-validation, while target values average five final-fit "
            "seeds; training and evaluation regimes differ; cohorts are unpaired; group labels "
            "support an ability-associated observational description, not a causal disability "
            "effect.",
            "",
        ]
    )
    return "\n".join(lines)


def build_cross_cohort_gap_report(
    source_aggregate_path: str | Path,
    target_statistics_path: str | Path,
    mapping_specification_path: str | Path,
    *,
    output_root: str | Path,
    destination: str | Path,
    csv_destination: str | Path,
    markdown_destination: str | Path,
) -> dict[str, Any]:
    """Validate the locked aggregates and publish a create-only descriptive report."""

    root = Path(output_root).resolve(strict=True)
    if not root.is_dir():
        raise CohortGapError(f"output root is not a directory: {root}")
    source_file = _resolve_existing_json(
        source_aggregate_path, root=root, role="source grouped-CV aggregate"
    )
    target_file = _resolve_existing_json(
        target_statistics_path, root=root, role="target participant statistics"
    )
    specification_file = _resolve_existing_json(
        mapping_specification_path, root=root, role="model mapping specification"
    )
    source = _mapping(load_json_strict(source_file), name="source grouped-CV aggregate")
    target = _mapping(load_json_strict(target_file), name="target participant statistics")
    specification = _mapping(
        load_json_strict(specification_file), name="model mapping specification"
    )
    source_self_hash = _validate_self_hash(
        source, field="aggregate_record_sha256", role="source aggregate"
    )
    target_self_hash = _validate_self_hash(target, field="record_sha256", role="target statistics")
    specification_self_hash = _validate_self_hash(
        specification, field="specification_sha256", role="mapping specification"
    )
    model_mapping, bootstrap = _validate_specification(
        specification,
        source_file=source_file,
        target_file=target_file,
        specification_file=specification_file,
        root=root,
    )
    source_models = _source_models(source, expected_participants=SOURCE_PARTICIPANT_IDS)
    target_models = _target_models(
        target,
        expected_participants=TARGET_PARTICIPANT_IDS,
        expected_seeds=TARGET_SEED_ORDER,
    )
    lineage = _validate_input_lineage(
        source,
        target,
        specification,
        root=root,
        source_file=source_file,
        source_self_hash=source_self_hash,
        target_self_hash=target_self_hash,
        model_mapping=model_mapping,
        source_models=source_models,
    )
    missing_configurations = sorted(set(model_mapping.values()) - set(source_models))
    if missing_configurations:
        raise CohortGapError(
            "mapped source configurations are absent: " + ", ".join(missing_configurations)
        )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    source_indices = rng.integers(
        0,
        len(SOURCE_PARTICIPANT_IDS),
        size=(BOOTSTRAP_RESAMPLES, len(SOURCE_PARTICIPANT_IDS)),
    )
    target_indices = rng.integers(
        0,
        len(TARGET_PARTICIPANT_IDS),
        size=(BOOTSTRAP_RESAMPLES, len(TARGET_PARTICIPANT_IDS)),
    )
    models: list[dict[str, Any]] = []
    for model_id in FINAL_MODEL_IDS:
        configuration_id = model_mapping[model_id]
        source_row = source_models[configuration_id]
        target_row = target_models[model_id]
        source_values = _source_participants(
            source_row,
            configuration_id=configuration_id,
            expected_participants=SOURCE_PARTICIPANT_IDS,
        )
        target_values = _target_participants(
            target_row,
            model_id=model_id,
            expected_participants=TARGET_PARTICIPANT_IDS,
            expected_seeds=TARGET_SEED_ORDER,
        )
        source_summary = _participant_summary(source_values)
        target_summary = _participant_summary(target_values)
        source_array = np.asarray(list(source_values.values()), dtype=np.float64)
        target_array = np.asarray(list(target_values.values()), dtype=np.float64)
        bootstrap_gaps = np.mean(source_array[source_indices], axis=1) - np.mean(
            target_array[target_indices], axis=1
        )
        mean_gap = float(source_summary["mean_macro_f1"] - target_summary["mean_macro_f1"])
        interval = {
            "method": bootstrap["method"],
            "confidence": 0.95,
            "estimate": mean_gap,
            "lower": float(np.quantile(bootstrap_gaps, 0.025, method="linear")),
            "upper": float(np.quantile(bootstrap_gaps, 0.975, method="linear")),
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "cluster_unit": "participant",
            "source_participant_count": len(SOURCE_PARTICIPANT_IDS),
            "target_participant_count": len(TARGET_PARTICIPANT_IDS),
            "cohort_resampling": "independent_unpaired_draws",
            "indices_shared_across_models": True,
        }
        models.append(
            {
                "model_id": model_id,
                "source_configuration_id": configuration_id,
                "source_configuration_sha256": _sha256(
                    source_row.get("configuration_sha256"),
                    name=f"{configuration_id} configuration hash",
                ),
                "source_model_variant_id": _text(
                    source_row.get("model_variant_id"),
                    name=f"{configuration_id} model variant ID",
                ),
                "source_seed": SOURCE_SEED,
                "target_seed_order": list(TARGET_SEED_ORDER),
                "source_participant_macro_f1": source_summary,
                "target_participant_macro_f1": target_summary,
                "gaps": {
                    "sign_convention": "source_minus_target",
                    "source_mean_minus_target_mean_macro_f1": mean_gap,
                    "source_mean_minus_target_worst_macro_f1": float(
                        source_summary["mean_macro_f1"] - target_summary["worst_macro_f1"]
                    ),
                    "source_mean_minus_target_lower_decile_macro_f1": float(
                        source_summary["mean_macro_f1"] - target_summary["lower_decile_macro_f1"]
                    ),
                    "mean_gap_bootstrap_interval": interval,
                },
            }
        )

    destinations = {
        "csv": _resolve_new(csv_destination, root=root, role="cohort-gap CSV"),
        "markdown": _resolve_new(markdown_destination, root=root, role="cohort-gap Markdown"),
        "json": _resolve_new(destination, root=root, role="cohort-gap JSON"),
    }
    if len(set(destinations.values())) != len(destinations):
        raise CohortGapError("cohort-gap output destinations must be distinct")
    csv_fields = [
        "model_id",
        "source_configuration_id",
        "source_seed",
        "target_seed_count",
        "source_mean_macro_f1",
        "source_worst_macro_f1",
        "source_lower_decile_macro_f1",
        "target_mean_macro_f1",
        "target_worst_macro_f1",
        "target_lower_decile_macro_f1",
        "source_mean_minus_target_mean_macro_f1",
        "mean_gap_ci_lower",
        "mean_gap_ci_upper",
        "source_mean_minus_target_worst_macro_f1",
        "source_mean_minus_target_lower_decile_macro_f1",
    ]
    csv_rows: list[dict[str, Any]] = []
    for row in models:
        csv_source = _mapping(row["source_participant_macro_f1"], name="source summary")
        csv_target = _mapping(row["target_participant_macro_f1"], name="target summary")
        csv_gaps = _mapping(row["gaps"], name="gaps")
        csv_interval = _mapping(csv_gaps["mean_gap_bootstrap_interval"], name="gap interval")
        csv_rows.append(
            {
                "model_id": row["model_id"],
                "source_configuration_id": row["source_configuration_id"],
                "source_seed": SOURCE_SEED,
                "target_seed_count": len(TARGET_SEED_ORDER),
                "source_mean_macro_f1": csv_source["mean_macro_f1"],
                "source_worst_macro_f1": csv_source["worst_macro_f1"],
                "source_lower_decile_macro_f1": csv_source["lower_decile_macro_f1"],
                "target_mean_macro_f1": csv_target["mean_macro_f1"],
                "target_worst_macro_f1": csv_target["worst_macro_f1"],
                "target_lower_decile_macro_f1": csv_target["lower_decile_macro_f1"],
                "source_mean_minus_target_mean_macro_f1": csv_gaps[
                    "source_mean_minus_target_mean_macro_f1"
                ],
                "mean_gap_ci_lower": csv_interval["lower"],
                "mean_gap_ci_upper": csv_interval["upper"],
                "source_mean_minus_target_worst_macro_f1": csv_gaps[
                    "source_mean_minus_target_worst_macro_f1"
                ],
                "source_mean_minus_target_lower_decile_macro_f1": csv_gaps[
                    "source_mean_minus_target_lower_decile_macro_f1"
                ],
            }
        )
    _write_text_new(_csv_text(csv_fields, csv_rows), destinations["csv"])
    _write_text_new(_markdown(models), destinations["markdown"])

    output_artifacts = {
        role: {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        }
        for role, path in destinations.items()
        if role != "json"
    }
    payload: dict[str, Any] = {
        "schema_version": COHORT_GAP_SCHEMA_VERSION,
        "record_kind": "cross_cohort_participant_performance_gap_report",
        "status": "complete_create_only",
        "evidence_status": COHORT_GAP_EVIDENCE_STATUS,
        "analysis_id": specification.get("analysis_id"),
        "interpretation_status": "descriptive_only_not_matched_confirmatory_or_causal",
        "input_artifacts": {
            "source_grouped_cv_aggregate": {
                "path": source_file.relative_to(root).as_posix(),
                "file_sha256": sha256_file(source_file),
                "aggregate_record_sha256": source_self_hash,
            },
            "target_participant_statistics": {
                "path": target_file.relative_to(root).as_posix(),
                "file_sha256": sha256_file(target_file),
                "record_sha256": target_self_hash,
            },
            "model_mapping_specification": {
                "path": specification_file.relative_to(root).as_posix(),
                "file_sha256": sha256_file(specification_file),
                "specification_sha256": specification_self_hash,
            },
        },
        "lineage": dict(lineage),
        "participant_sets": {
            "source": list(SOURCE_PARTICIPANT_IDS),
            "target": list(TARGET_PARTICIPANT_IDS),
            "overlap": [],
        },
        "regimes": {
            "source": "one-seed participant-grouped cross-validation development aggregate",
            "target": "five-seed final-fit locked target participant averages",
            "source_seed": SOURCE_SEED,
            "target_seed_order": list(TARGET_SEED_ORDER),
            "statistical_unit": "participant",
            "cohorts_paired": False,
            "source_selection_consumed": True,
            "training_and_evaluation_regimes_identical": False,
        },
        "bootstrap": {
            "method": bootstrap["method"],
            "confidence": 0.95,
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "cluster_unit": "participant",
            "independent_cohort_draws": True,
            "shared_indices_across_models": True,
        },
        "model_count": len(models),
        "model_to_source_configuration": dict(model_mapping),
        "models": models,
        "output_artifacts": output_artifacts,
        "limitations": list(
            _strings(specification.get("limitations"), name="specified limitations")
        ),
        "target_information_used_to_choose_mapping": False,
        "raw_data_or_prediction_arrays_read": False,
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    atomic_write_json_new(payload, destinations["json"], allowed_root=root)
    return payload
