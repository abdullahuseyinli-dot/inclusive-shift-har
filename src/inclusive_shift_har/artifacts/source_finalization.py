"""Fail-closed source-only calibration and final artifact-freeze orchestration."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.artifacts.final_freeze import (
    FrozenModelInput,
    build_final_freeze_inventory,
    validate_final_freeze_inventory_file,
    write_final_freeze_inventory_new,
)
from inclusive_shift_har.evaluation.source_calibration import (
    build_source_temperature_calibrator_record,
    validate_source_temperature_calibrator_record,
    write_source_temperature_calibrator_new,
)
from inclusive_shift_har.manifests.canonical import (
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)
from inclusive_shift_har.models.classical import load_classical_checkpoint
from inclusive_shift_har.training.final_checkpoint import validate_fixed_epoch_checkpoint

FINAL_SELECTION_SCHEMA_VERSION = "1.0.0"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class SourceFinalizationError(RuntimeError):
    """Raised when final source evidence cannot safely be frozen."""


@dataclass(frozen=True)
class _ValidatedRun:
    model_id: str
    seed: int
    training_regime: str
    selected_epoch: int | None
    configuration: Mapping[str, Any]
    checkpoint_path: Path
    checkpoint_sha256: str
    logits: NDArray[np.float64]
    labels: NDArray[np.int64]
    window_ids: tuple[str, ...]
    class_names: tuple[str, ...]


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SourceFinalizationError(f"{name} must be a lowercase full SHA-256")
    return value


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceFinalizationError(f"{name} must be an object")
    return value


def _resolve_regular_file(value: Any, *, artifact_root: Path, role: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SourceFinalizationError(f"{role} path is missing")
    raw = Path(value)
    candidate = raw if raw.is_absolute() else artifact_root / raw
    root = artifact_root.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SourceFinalizationError(f"{role} escapes the artifact root") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise SourceFinalizationError(f"{role} is missing, non-regular, or a symlink")
    return resolved


def load_final_selection_plan(path: str | Path) -> dict[str, Any]:
    """Load and validate a self-hashed source-only final selection plan."""

    payload = load_json_strict(path)
    if not isinstance(payload, dict):
        raise SourceFinalizationError("final selection plan root must be an object")
    required = {
        "schema_version": FINAL_SELECTION_SCHEMA_VERSION,
        "record_kind": "final_source_selection_plan",
        "status": "predeclared_source_only_target_sealed",
        "target_subject_or_window_records_used": False,
        "target_predictions_or_performance_accessed": False,
        "final_source_split_id": "final_source_split",
    }
    mismatches = [key for key, expected in required.items() if payload.get(key) != expected]
    if mismatches:
        raise SourceFinalizationError(f"final selection plan contract mismatch: {mismatches}")
    body = dict(payload)
    claimed_hash = body.pop("selection_plan_sha256", None)
    if claimed_hash != canonical_json_sha256(body):
        raise SourceFinalizationError("final selection plan self-hash does not validate")
    for name in (
        "dataset_manifest_sha256",
        "source_window_manifest_sha256",
        "split_manifest_sha256",
        "protocol_lock_sha256",
        "target_seal_id",
    ):
        _require_sha256(payload.get(name), name=name)
    seeds = payload.get("required_seed_order")
    if (
        not isinstance(seeds, list)
        or not seeds
        or len(set(seeds)) != len(seeds)
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise SourceFinalizationError("final selection plan has invalid required seeds")
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        raise SourceFinalizationError("final selection plan has no models")
    model_ids: set[str] = set()
    for index, value in enumerate(models):
        entry = _require_mapping(value, name=f"models[{index}]")
        model_id = entry.get("model_id")
        if (
            not isinstance(model_id, str)
            or _MODEL_ID_RE.fullmatch(model_id) is None
            or model_id in model_ids
        ):
            raise SourceFinalizationError("selection model IDs must be safe and unique")
        model_ids.add(model_id)
        if entry.get("training_regime") not in {
            "fixed_epoch_neural",
            "deterministic_classical",
        }:
            raise SourceFinalizationError(f"{model_id} has an invalid training regime")
        if not isinstance(entry.get("summary_path_template"), str) or "{seed}" not in cast(
            str, entry.get("summary_path_template")
        ):
            raise SourceFinalizationError(f"{model_id} summary template must contain {{seed}}")
        if not isinstance(entry.get("run_directory_template"), str) or "{seed}" not in cast(
            str, entry.get("run_directory_template")
        ):
            raise SourceFinalizationError(
                f"{model_id} run directory template must contain {{seed}}"
            )
        _require_mapping(entry.get("runner_arguments"), name=f"{model_id} runner arguments")
        _require_mapping(entry.get("configuration_expectations"), name=f"{model_id} expectations")
    return payload


def _validate_summary_hash(summary: Mapping[str, Any], *, role: str) -> None:
    body = dict(summary)
    claimed_hash = body.pop("record_sha256", None)
    if claimed_hash != canonical_json_sha256(body):
        raise SourceFinalizationError(f"{role} self-hash does not validate")


def _load_prediction_arrays(
    path: Path,
    *,
    expected_count: int,
    expected_participants: frozenset[str],
    expected_classes: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64], tuple[str, ...]]:
    with np.load(path, allow_pickle=False) as arrays:
        required = {"logits", "probabilities", "labels", "participant_ids", "window_ids"}
        if set(arrays.files) != required:
            raise SourceFinalizationError("source validation prediction keys differ from contract")
        logits = np.asarray(arrays["logits"], dtype=np.float64)
        probabilities = np.asarray(arrays["probabilities"], dtype=np.float64)
        labels = np.asarray(arrays["labels"], dtype=np.int64)
        participants = tuple(str(value) for value in arrays["participant_ids"].tolist())
        window_ids = tuple(str(value) for value in arrays["window_ids"].tolist())
    if logits.shape != (expected_count, expected_classes):
        raise SourceFinalizationError("source validation logits have the wrong shape")
    if probabilities.shape != logits.shape or labels.shape != (expected_count,):
        raise SourceFinalizationError("source validation arrays are not aligned")
    if len(participants) != expected_count or frozenset(participants) != expected_participants:
        raise SourceFinalizationError("source validation participants differ from the plan")
    if len(window_ids) != expected_count or len(set(window_ids)) != expected_count:
        raise SourceFinalizationError("source validation window IDs are not aligned and unique")
    if (
        not np.isfinite(logits).all()
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise SourceFinalizationError("source validation scores are invalid")
    if np.any(labels < 0) or np.any(labels >= expected_classes):
        raise SourceFinalizationError("source validation labels violate the locked schema")
    return logits, labels, window_ids


def _validate_one_run(
    entry: Mapping[str, Any],
    seed: int,
    *,
    plan: Mapping[str, Any],
    artifact_root: Path,
    expected_code_commit: str,
) -> _ValidatedRun:
    model_id = str(entry["model_id"])
    template = str(entry["summary_path_template"])
    summary_path = _resolve_regular_file(
        template.format(seed=seed), artifact_root=artifact_root, role=f"{model_id} summary"
    )
    summary_value = load_json_strict(summary_path)
    summary = _require_mapping(summary_value, name=f"{model_id} summary")
    _validate_summary_hash(summary, role=f"{model_id} summary")
    required = {
        "schema_version": "1.0.0",
        "status": "source_development_complete_target_sealed",
        "evidence_status": "source_development_not_confirmatory",
        "model_name": entry.get("runner_model_name"),
        "seed": seed,
        "source_window_manifest_sha256": plan["source_window_manifest_sha256"],
        "split_manifest_sha256": plan["split_manifest_sha256"],
        "source_split_id": "final_source_split",
        "class_names": plan["class_names"],
        "train_participants": plan["final_train_participants"],
        "validation_participants": plan["final_validation_participants"],
        "train_window_count": plan["final_train_window_count"],
        "validation_window_count": plan["final_validation_window_count"],
        "code_commit": expected_code_commit,
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
    }
    mismatches = [key for key, expected in required.items() if summary.get(key) != expected]
    if mismatches:
        raise SourceFinalizationError(f"{model_id} seed {seed} summary mismatch: {mismatches}")
    configuration = _require_mapping(summary.get("configuration"), name="training configuration")
    if summary.get("configuration_sha256") != canonical_json_sha256(configuration):
        raise SourceFinalizationError(f"{model_id} seed {seed} configuration hash mismatch")
    common_key = (
        "common_neural_configuration_expectations"
        if entry["training_regime"] == "fixed_epoch_neural"
        else "common_classical_configuration_expectations"
    )
    common_expectations = plan.get(common_key, {})
    if not isinstance(common_expectations, Mapping):
        raise SourceFinalizationError(f"selection plan {common_key} must be an object")
    runner_expectations: Mapping[str, Any] = {}
    if entry["training_regime"] == "fixed_epoch_neural":
        common_runner_value = plan.get("common_neural_runner_arguments", {})
        if not isinstance(common_runner_value, Mapping):
            raise SourceFinalizationError(
                "selection plan common_neural_runner_arguments must be an object"
            )
        runner_value = entry.get("runner_arguments")
        if not isinstance(runner_value, Mapping):
            raise SourceFinalizationError(f"{model_id} runner_arguments must be an object")
        runner_expectations = {**dict(common_runner_value), **dict(runner_value)}
    expectations = {
        **dict(common_expectations),
        **dict(runner_expectations),
        **dict(cast(Mapping[str, Any], entry["configuration_expectations"])),
    }
    mismatched_config = [
        key for key, expected in expectations.items() if configuration.get(key) != expected
    ]
    if configuration.get("seed") != seed or mismatched_config:
        raise SourceFinalizationError(
            f"{model_id} seed {seed} differs from predeclared configuration: {mismatched_config}"
        )
    regime = str(entry["training_regime"])
    selected_epoch: int | None
    checkpoint_record = _require_mapping(summary.get("checkpoint"), name="checkpoint record")
    checkpoint_path = _resolve_regular_file(
        checkpoint_record.get("path"), artifact_root=artifact_root, role=f"{model_id} checkpoint"
    )
    checkpoint_sha256 = _require_sha256(
        checkpoint_record.get("sha256"), name=f"{model_id} checkpoint_sha256"
    )
    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise SourceFinalizationError(f"{model_id} seed {seed} checkpoint hash mismatch")
    if regime == "fixed_epoch_neural":
        if summary.get("requested_device") != "cuda" or summary.get("model_device") != "cuda":
            raise SourceFinalizationError(f"{model_id} final neural run was not executed on CUDA")
        epochs = configuration.get("epochs")
        if (
            isinstance(epochs, bool)
            or not isinstance(epochs, int)
            or summary.get("best_epoch") != epochs
        ):
            raise SourceFinalizationError(f"{model_id} is not the fixed predeclared final epoch")
        selected_epoch = epochs
        validated = validate_fixed_epoch_checkpoint(
            checkpoint_path,
            expected_configuration_sha256=str(summary["configuration_sha256"]),
            expected_split_manifest_sha256=str(plan["split_manifest_sha256"]),
            expected_code_commit=expected_code_commit,
        )
        if validated["checkpoint_sha256"] != checkpoint_sha256:
            raise SourceFinalizationError(f"{model_id} fixed checkpoint validation changed hash")
    else:
        selected_epoch = None
        if summary.get("best_epoch") is not None:
            raise SourceFinalizationError(
                f"{model_id} classical run unexpectedly selected an epoch"
            )
        fitted = load_classical_checkpoint(checkpoint_path)
        if canonical_json_sha256(configuration) != canonical_json_sha256(asdict(fitted.config)):
            raise SourceFinalizationError(f"{model_id} classical checkpoint config differs")
        lineage = fitted.lineage
        required_lineage = {
            "dataset_manifest_sha256": plan["dataset_manifest_sha256"],
            "split_manifest_sha256": plan["split_manifest_sha256"],
            "code_commit": expected_code_commit,
            "label_schema": plan["class_names"],
        }
        lineage_mismatches = [
            key for key, expected in required_lineage.items() if lineage.get(key) != expected
        ]
        if lineage_mismatches:
            raise SourceFinalizationError(
                f"{model_id} classical checkpoint lineage mismatch: {lineage_mismatches}"
            )
    prediction = _require_mapping(summary.get("prediction_artifact"), name="prediction record")
    prediction_path = _resolve_regular_file(
        prediction.get("path"), artifact_root=artifact_root, role=f"{model_id} predictions"
    )
    if sha256_file(prediction_path) != _require_sha256(
        prediction.get("sha256"), name=f"{model_id} prediction_sha256"
    ):
        raise SourceFinalizationError(f"{model_id} seed {seed} prediction hash mismatch")
    logits, labels, window_ids = _load_prediction_arrays(
        prediction_path,
        expected_count=int(plan["final_validation_window_count"]),
        expected_participants=frozenset(
            str(value) for value in plan["final_validation_participants"]
        ),
        expected_classes=len(cast(Sequence[Any], plan["class_names"])),
    )
    return _ValidatedRun(
        model_id=model_id,
        seed=seed,
        training_regime=regime,
        selected_epoch=selected_epoch,
        configuration=configuration,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        logits=logits,
        labels=labels,
        window_ids=window_ids,
        class_names=tuple(str(value) for value in cast(Sequence[Any], plan["class_names"])),
    )


def freeze_final_source_runs(
    selection_plan_path: str | Path,
    *,
    artifact_root: str | Path,
    calibrator_directory: str | Path,
    inventory_path: str | Path,
    created_at_utc: str,
    expected_code_commit: str,
) -> dict[str, Any]:
    """Validate every planned run, fit source-only calibration, and freeze exact bytes."""

    root = Path(artifact_root).resolve(strict=True)
    plan = load_final_selection_plan(selection_plan_path)
    runs = [
        _validate_one_run(
            entry,
            int(seed),
            plan=plan,
            artifact_root=root,
            expected_code_commit=expected_code_commit,
        )
        for entry in cast(list[Mapping[str, Any]], plan["models"])
        for seed in cast(list[int], plan["required_seed_order"])
    ]
    calibrator_root = Path(calibrator_directory)
    if not calibrator_root.is_absolute():
        calibrator_root = root / calibrator_root
    calibrator_root.mkdir(parents=True, exist_ok=True)
    try:
        calibrator_root.resolve(strict=True).relative_to(root)
    except ValueError as exc:
        raise SourceFinalizationError("calibrator directory escapes artifact root") from exc
    frozen: list[FrozenModelInput] = []
    calibrator_records: list[dict[str, Any]] = []
    for run in runs:
        record = build_source_temperature_calibrator_record(
            run.logits,
            run.labels,
            run.window_ids,
            fit_partition="source_validation",
            checkpoint_sha256=run.checkpoint_sha256,
            training_configuration_sha256=canonical_json_sha256(run.configuration),
            split_manifest_sha256=str(plan["split_manifest_sha256"]),
            class_names=run.class_names,
        )
        calibrator_path = calibrator_root / f"{run.model_id}--seed-{run.seed}.json"
        if calibrator_path.exists():
            existing = load_json_strict(calibrator_path)
            if existing != record:
                raise SourceFinalizationError(
                    f"existing calibrator differs from validated source inputs: {calibrator_path}"
                )
            validate_source_temperature_calibrator_record(cast(Mapping[str, Any], existing))
        else:
            write_source_temperature_calibrator_new(
                record, calibrator_path, allowed_root=calibrator_root
            )
        calibrator_records.append(
            {
                "model_id": run.model_id,
                "seed": run.seed,
                "path": calibrator_path.resolve().relative_to(root).as_posix(),
                "record_sha256": record["record_sha256"],
                "temperature": record["temperature"],
                "nll_before": record["nll_before"],
                "nll_after": record["nll_after"],
            }
        )
        frozen.append(
            FrozenModelInput(
                model_id=run.model_id,
                seed=run.seed,
                training_regime=run.training_regime,
                training_configuration=run.configuration,
                checkpoint_path=run.checkpoint_path,
                calibrator_path=calibrator_path,
                selected_epoch=run.selected_epoch,
            )
        )
    inventory = build_final_freeze_inventory(
        frozen,
        artifact_root=root,
        created_at_utc=created_at_utc,
        code_commit=expected_code_commit,
        dataset_manifest_sha256=str(plan["dataset_manifest_sha256"]),
        source_window_manifest_sha256=str(plan["source_window_manifest_sha256"]),
        split_manifest_sha256=str(plan["split_manifest_sha256"]),
        protocol_lock_sha256=str(plan["protocol_lock_sha256"]),
        target_seal_id=str(plan["target_seal_id"]),
        required_seed_order=cast(list[int], plan["required_seed_order"]),
        selection_plan_sha256=str(plan["selection_plan_sha256"]),
    )
    destination = Path(inventory_path)
    if not destination.is_absolute():
        destination = root / destination
    write_final_freeze_inventory_new(inventory, destination, allowed_root=root)
    report = validate_final_freeze_inventory_file(
        destination,
        artifact_root=root,
        expected_split_manifest_sha256=str(plan["split_manifest_sha256"]),
        expected_protocol_lock_sha256=str(plan["protocol_lock_sha256"]),
        expected_target_seal_id=str(plan["target_seal_id"]),
    )
    if not report.valid:
        raise SourceFinalizationError(
            f"published final freeze failed validation: {report.to_dict()}"
        )
    return {
        "status": "frozen_source_only_target_sealed",
        "selection_plan_sha256": plan["selection_plan_sha256"],
        "inventory_path": destination.resolve().relative_to(root).as_posix(),
        "inventory_sha256": inventory["inventory_sha256"],
        "frozen_artifact_set_sha256": inventory["frozen_artifact_set_sha256"],
        "model_count": len(plan["models"]),
        "model_seed_count": len(runs),
        "calibrators": calibrator_records,
        "target_subject_or_window_records_used": False,
        "target_predictions_or_performance_accessed": False,
    }
