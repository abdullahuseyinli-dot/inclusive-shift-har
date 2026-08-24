"""Fail-closed, one-opening evaluation of a frozen confirmatory target.

This module deliberately does not know where target signals live.  A caller must
provide a materializer callback, which is invoked exactly once and only after the
exact source-artifact freeze has validated and the canonical opening receipt has
been published create-only.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np
import torch
from numpy.typing import NDArray

from inclusive_shift_har.artifacts.final_freeze import (
    record_target_opening_once,
    validate_exact_target_unlock,
    validate_final_freeze_inventory_file,
)
from inclusive_shift_har.data.materialize import MaterializedWindows
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.evaluation.source_calibration import (
    load_source_temperature_calibrator_file,
)
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_bytes,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)
from inclusive_shift_har.models.classical import (
    FittedClassicalModel,
    load_classical_checkpoint,
    predict_classical_model,
)
from inclusive_shift_har.preprocessing.normalization import ChannelStandardizer
from inclusive_shift_har.training.engine import (
    TrainingConfig,
    predict_model,
    reconstruct_checkpoint,
    training_config_from_dict,
)
from inclusive_shift_har.training.final_checkpoint import validate_fixed_epoch_checkpoint

LOCKED_TARGET_RESULT_SCHEMA_VERSION = "1.0.0"
CANONICAL_OPENING_RECEIPT_NAME = "confirmatory_target_opening_1.json"
LOCKED_TARGET_EVIDENCE_STATUS = "locked_confirmatory_target_opening_1"
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class LockedTargetEvaluationError(RuntimeError):
    """Raised before or during a fail-closed locked-target evaluation."""


TargetMaterializer = Callable[[Mapping[str, Any]], MaterializedWindows]


@dataclass(frozen=True)
class _PreparedModel:
    model_id: str
    seed: int
    training_regime: str
    configuration: Mapping[str, Any]
    configuration_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    calibrator_path: Path
    calibrator_sha256: str
    calibrator_record_sha256: str
    class_names: tuple[str, ...]
    normalizer: ChannelStandardizer
    neural_config: TrainingConfig | None
    classical_model: FittedClassicalModel | None
    temperature: float


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise LockedTargetEvaluationError(f"{name} must be a lowercase full SHA-256")
    return value


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LockedTargetEvaluationError(f"{name} must be an object")
    return value


def _resolve_frozen_file(
    file_record: Mapping[str, Any],
    *,
    artifact_root: Path,
    role: str,
) -> Path:
    relative = file_record.get("path")
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise LockedTargetEvaluationError(f"{role} has an unsafe frozen path")
    root = artifact_root.resolve(strict=True)
    candidate = root.joinpath(*relative.split("/")).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise LockedTargetEvaluationError(f"{role} escapes the artifact root") from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise LockedTargetEvaluationError(f"{role} is missing, non-regular, or a symlink")
    return candidate


def _validate_normalization(
    payload: Any,
    *,
    split_manifest_sha256: str,
    target_subjects: frozenset[str],
) -> ChannelStandardizer:
    normalization = _require_mapping(payload, name="checkpoint normalization")
    required = {
        "schema_version": "1.0.0",
        "method": "per_channel_population_standardization",
        "fit_scope": "training_partition_only",
        "split_manifest_sha256": split_manifest_sha256,
    }
    mismatches = [key for key, expected in required.items() if normalization.get(key) != expected]
    if mismatches:
        raise LockedTargetEvaluationError(
            f"checkpoint normalization violates the training-only contract: {mismatches}"
        )
    standardizer = ChannelStandardizer.from_dict(dict(normalization))
    if len(standardizer.channel_names) != 6 or len(set(standardizer.channel_names)) != 6:
        raise LockedTargetEvaluationError("checkpoint normalization lacks six ordered channels")
    if set(standardizer.training_participants) & target_subjects:
        raise LockedTargetEvaluationError("checkpoint normalization includes a target participant")
    if not standardizer.training_participants:
        raise LockedTargetEvaluationError("checkpoint normalization lacks source participants")
    return standardizer


def _validate_lineage(
    lineage_value: Any,
    *,
    inventory: Mapping[str, Any],
    target_subjects: frozenset[str],
) -> tuple[Mapping[str, Any], tuple[str, ...], ChannelStandardizer]:
    lineage = _require_mapping(lineage_value, name="checkpoint lineage")
    expected = {
        "dataset_manifest_sha256": inventory["dataset_manifest_sha256"],
        "split_manifest_sha256": inventory["split_manifest_sha256"],
        "code_commit": inventory["code_commit"],
    }
    mismatches = [key for key, value in expected.items() if lineage.get(key) != value]
    if mismatches:
        raise LockedTargetEvaluationError(
            f"checkpoint lineage differs from the freeze: {mismatches}"
        )
    for name in ("preprocessing_config_sha256", "ontology_sha256"):
        _require_sha256(lineage.get(name), name=f"checkpoint lineage {name}")
    evidence_status = lineage.get("evidence_status")
    if not isinstance(evidence_status, str) or "target_sealed" not in evidence_status:
        raise LockedTargetEvaluationError(
            "checkpoint lineage is not source-only target-sealed evidence"
        )
    labels = lineage.get("label_schema")
    if (
        not isinstance(labels, (list, tuple))
        or len(labels) < 2
        or any(not isinstance(label, str) or not label for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise LockedTargetEvaluationError("checkpoint label schema is invalid")
    class_names = tuple(cast(list[str] | tuple[str, ...], labels))
    normalizer = _validate_normalization(
        lineage.get("normalization"),
        split_manifest_sha256=str(inventory["split_manifest_sha256"]),
        target_subjects=target_subjects,
    )
    return lineage, class_names, normalizer


def _prepare_model(
    entry_value: Any,
    *,
    inventory: Mapping[str, Any],
    artifact_root: Path,
    target_subjects: frozenset[str],
) -> _PreparedModel:
    entry = _require_mapping(entry_value, name="frozen model entry")
    model_id = entry.get("model_id")
    seed = entry.get("seed")
    if not isinstance(model_id, str) or _MODEL_ID_RE.fullmatch(model_id) is None:
        raise LockedTargetEvaluationError("frozen model_id is unsafe for create-only artifacts")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise LockedTargetEvaluationError("frozen model seed is invalid")
    configuration = _require_mapping(
        entry.get("training_configuration"), name=f"{model_id} configuration"
    )
    configuration_sha256 = _require_sha256(
        entry.get("training_configuration_sha256"), name=f"{model_id} configuration hash"
    )
    if canonical_json_sha256(configuration) != configuration_sha256:
        raise LockedTargetEvaluationError(f"{model_id} configuration hash no longer validates")
    checkpoint_record = _require_mapping(entry.get("checkpoint"), name=f"{model_id} checkpoint")
    calibrator_record = _require_mapping(entry.get("calibrator"), name=f"{model_id} calibrator")
    checkpoint_path = _resolve_frozen_file(
        checkpoint_record, artifact_root=artifact_root, role=f"{model_id} checkpoint"
    )
    calibrator_path = _resolve_frozen_file(
        calibrator_record, artifact_root=artifact_root, role=f"{model_id} calibrator"
    )
    checkpoint_sha256 = _require_sha256(
        checkpoint_record.get("sha256"), name=f"{model_id} checkpoint hash"
    )
    calibrator_sha256 = _require_sha256(
        calibrator_record.get("sha256"), name=f"{model_id} calibrator hash"
    )
    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise LockedTargetEvaluationError(f"{model_id} checkpoint bytes differ from the freeze")
    if sha256_file(calibrator_path) != calibrator_sha256:
        raise LockedTargetEvaluationError(f"{model_id} calibrator bytes differ from the freeze")
    calibrator_payload, calibrator = load_source_temperature_calibrator_file(
        calibrator_path,
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_training_configuration_sha256=configuration_sha256,
        expected_split_manifest_sha256=str(inventory["split_manifest_sha256"]),
    )
    calibrator_record_sha256 = _require_sha256(
        entry.get("calibrator_record_sha256"), name=f"{model_id} calibrator record hash"
    )
    if calibrator_payload.get("record_sha256") != calibrator_record_sha256:
        raise LockedTargetEvaluationError(f"{model_id} calibrator record differs from the freeze")
    calibrator_classes = tuple(str(value) for value in calibrator_payload["class_names"])

    regime = entry.get("training_regime")
    neural_config: TrainingConfig | None = None
    classical_model: FittedClassicalModel | None = None
    if regime == "fixed_epoch_neural":
        validated = validate_fixed_epoch_checkpoint(
            checkpoint_path,
            expected_configuration_sha256=configuration_sha256,
            expected_split_manifest_sha256=str(inventory["split_manifest_sha256"]),
            expected_code_commit=str(inventory["code_commit"]),
        )
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_mapping = _require_mapping(
            checkpoint_payload, name=f"{model_id} neural checkpoint"
        )
        checkpoint_configuration = _require_mapping(
            checkpoint_mapping.get("configuration"), name=f"{model_id} checkpoint configuration"
        )
        if canonical_json_sha256(checkpoint_configuration) != configuration_sha256:
            raise LockedTargetEvaluationError(
                f"{model_id} checkpoint configuration differs from the inventory"
            )
        neural_config = training_config_from_dict(dict(checkpoint_configuration))
        if neural_config.seed != seed or validated["seed"] != seed:
            raise LockedTargetEvaluationError(f"{model_id} checkpoint seed differs from inventory")
        lineage, class_names, normalizer = _validate_lineage(
            checkpoint_mapping.get("lineage"),
            inventory=inventory,
            target_subjects=target_subjects,
        )
        if checkpoint_mapping.get("normalization") != lineage.get("normalization"):
            raise LockedTargetEvaluationError(
                f"{model_id} checkpoint duplicates inconsistent normalization"
            )
        if tuple(checkpoint_mapping.get("label_schema", ())) != class_names:
            raise LockedTargetEvaluationError(f"{model_id} checkpoint label schema is inconsistent")
        environment = _require_mapping(
            checkpoint_mapping.get("environment"), name=f"{model_id} training environment"
        )
        if environment.get("device_type") != "cuda":
            raise LockedTargetEvaluationError(
                f"{model_id} final neural checkpoint was not produced under the CUDA contract"
            )
    elif regime == "deterministic_classical":
        classical_model = load_classical_checkpoint(checkpoint_path)
        if canonical_json_sha256(asdict(classical_model.config)) != configuration_sha256:
            raise LockedTargetEvaluationError(
                f"{model_id} classical checkpoint configuration differs from inventory"
            )
        if classical_model.config.seed != seed:
            raise LockedTargetEvaluationError(f"{model_id} classical seed differs from inventory")
        lineage, class_names, normalizer = _validate_lineage(
            classical_model.lineage,
            inventory=inventory,
            target_subjects=target_subjects,
        )
        if tuple(classical_model.training_participants) != normalizer.training_participants:
            raise LockedTargetEvaluationError(
                f"{model_id} classical training participants differ from normalization"
            )
    else:
        raise LockedTargetEvaluationError(f"{model_id} has unsupported training regime {regime!r}")

    if class_names != calibrator_classes:
        raise LockedTargetEvaluationError(
            f"{model_id} calibrator class order differs from checkpoint"
        )
    expected_classes = (
        neural_config.num_classes
        if neural_config is not None
        else cast(FittedClassicalModel, classical_model).config.num_classes
    )
    if expected_classes != len(class_names):
        raise LockedTargetEvaluationError(f"{model_id} class count differs from locked labels")
    return _PreparedModel(
        model_id=model_id,
        seed=seed,
        training_regime=str(regime),
        configuration=configuration,
        configuration_sha256=configuration_sha256,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        calibrator_path=calibrator_path,
        calibrator_sha256=calibrator_sha256,
        calibrator_record_sha256=calibrator_record_sha256,
        class_names=class_names,
        normalizer=normalizer,
        neural_config=neural_config,
        classical_model=classical_model,
        temperature=calibrator.temperature,
    )


def _target_subjects(target_manifest: Mapping[str, Any]) -> frozenset[str]:
    seal = _require_mapping(target_manifest.get("target_seal"), name="target seal")
    values = seal.get("subject_ids")
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or not value for value in values)
        or len(set(values)) != len(values)
    ):
        raise LockedTargetEvaluationError("target seal lacks a unique ordered subject list")
    return frozenset(values)


def _prepare_inventory(
    inventory_path: Path,
    *,
    artifact_root: Path,
    target_manifest: Mapping[str, Any],
    device: torch.device,
) -> tuple[Mapping[str, Any], list[_PreparedModel], frozenset[str]]:
    subjects = _target_subjects(target_manifest)
    seal = cast(Mapping[str, Any], target_manifest["target_seal"])
    report = validate_final_freeze_inventory_file(
        inventory_path,
        artifact_root=artifact_root,
        expected_split_manifest_sha256=str(target_manifest.get("split_manifest_sha256")),
        expected_target_seal_id=str(seal.get("seal_id")),
    )
    if not report.valid:
        raise LockedTargetEvaluationError(f"final freeze inventory is invalid: {report.to_dict()}")
    inventory_value = load_json_strict(inventory_path)
    inventory = _require_mapping(inventory_value, name="final freeze inventory")
    models_value = inventory.get("models")
    if not isinstance(models_value, list) or not models_value:
        raise LockedTargetEvaluationError("final freeze inventory has no models")
    has_neural = any(
        isinstance(entry, Mapping) and entry.get("training_regime") == "fixed_epoch_neural"
        for entry in models_value
    )
    if has_neural and device.type != "cuda":
        raise LockedTargetEvaluationError(
            "locked neural target evaluation requires CUDA; CPU evaluation is forbidden"
        )
    if has_neural and not torch.cuda.is_available():
        raise LockedTargetEvaluationError("locked neural target evaluation requires available CUDA")
    prepared = [
        _prepare_model(
            entry,
            inventory=inventory,
            artifact_root=artifact_root,
            target_subjects=subjects,
        )
        for entry in models_value
    ]
    schemas = {item.class_names for item in prepared}
    if len(schemas) != 1:
        raise LockedTargetEvaluationError("frozen models do not share one locked class schema")
    return inventory, prepared, subjects


def _safe_output_directory(directory: Path, *, output_root: Path) -> Path:
    root = output_root.resolve(strict=True)
    directory.mkdir(parents=True, exist_ok=True)
    resolved = directory.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise LockedTargetEvaluationError("target result directory escapes output root") from exc
    if resolved.is_symlink() or not resolved.is_dir():
        raise LockedTargetEvaluationError("target result directory is not a regular directory")
    return resolved


def _result_paths(output_directory: Path, models: list[_PreparedModel]) -> list[tuple[Path, Path]]:
    paths: list[tuple[Path, Path]] = []
    for model in models:
        stem = f"{model.model_id}--seed-{model.seed}"
        paths.append(
            (
                output_directory / f"{stem}.predictions.npz",
                output_directory / f"{stem}.result.json",
            )
        )
    index_path = output_directory / "locked_target_evaluation_index.json"
    for path in [item for pair in paths for item in pair] + [index_path]:
        if os.path.lexists(path):
            raise LockedTargetEvaluationError(f"refusing to overwrite target evidence: {path}")
    return paths


def _validate_materialized_target(
    batch: MaterializedWindows,
    *,
    target_subjects: frozenset[str],
    class_names: tuple[str, ...],
) -> None:
    count = batch.signals.shape[0]
    if batch.signals.shape != (count, 128, 6) or batch.labels.shape != (count,):
        raise LockedTargetEvaluationError("materialized target violates [window,128,6] alignment")
    aligned_lengths = {
        count,
        len(batch.window_ids),
        len(batch.participant_ids),
        len(batch.released_labels),
        len(batch.partitions),
    }
    if count < 1 or len(aligned_lengths) != 1:
        raise LockedTargetEvaluationError(
            "materialized target arrays and identifiers are not aligned"
        )
    if not np.isfinite(batch.signals).all():
        raise LockedTargetEvaluationError("materialized target signals are non-finite")
    if len(set(batch.window_ids)) != count or any(not value for value in batch.window_ids):
        raise LockedTargetEvaluationError("materialized target window IDs are not unique")
    observed_subjects = frozenset(batch.participant_ids)
    if observed_subjects != target_subjects:
        raise LockedTargetEvaluationError(
            "materialized participants differ from the exact sealed target cohort"
        )
    if set(batch.partitions) != {"target_sealed"}:
        raise LockedTargetEvaluationError("materializer returned a non-target partition")
    if batch.class_names != class_names:
        raise LockedTargetEvaluationError("materialized class order differs from the frozen models")
    if batch.labels.min() < 0 or batch.labels.max() >= len(class_names):
        raise LockedTargetEvaluationError("materialized label lies outside the frozen class schema")


def _array_sha256(array: NDArray[Any], *, dtype: np.dtype[Any]) -> str:
    canonical = np.ascontiguousarray(array, dtype=dtype)
    header = canonical_json_bytes(
        {
            "algorithm": "numpy-c-contiguous-v1",
            "dtype": canonical.dtype.str,
            "shape": list(canonical.shape),
        }
    )
    import hashlib

    digest = hashlib.sha256()
    digest.update(header)
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _write_npz_create_only(path: Path, arrays: Mapping[str, NDArray[Any]]) -> str:
    temporary = path.parent / f".{path.name}.partial.{uuid4().hex}"
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **cast(dict[str, Any], dict(arrays)))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except Exception as exc:
        raise OSError(f"NPZ publication failed; partial retained at {temporary}: {exc}") from exc
    else:
        temporary.unlink()
    return sha256_file(path)


def _predict(
    model: _PreparedModel,
    batch: MaterializedWindows,
    *,
    device: torch.device,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
    windows = model.normalizer.transform(batch.signals)
    participant_ids = list(batch.participant_ids)
    if model.training_regime == "fixed_epoch_neural":
        config = cast(TrainingConfig, model.neural_config)
        if device.type != "cuda":
            raise LockedTargetEvaluationError("neural target prediction attempted without CUDA")
        original_cudnn = torch.backends.cudnn.enabled
        try:
            torch.backends.cudnn.enabled = not config.disable_cudnn
            neural, payload = reconstruct_checkpoint(model.checkpoint_path, device=device)
            if canonical_json_sha256(payload["configuration"]) != model.configuration_sha256:
                raise LockedTargetEvaluationError(
                    "neural reconstruction changed frozen configuration"
                )
            logits, uncalibrated, _ = predict_model(
                neural,
                windows,
                batch.labels,
                participant_ids,
                class_names=model.class_names,
                batch_size=config.batch_size,
                device=device,
                mixed_precision=config.mixed_precision,
                zero_channel_indices=config.zero_channel_indices,
            )
        finally:
            torch.backends.cudnn.enabled = original_cudnn
    else:
        fitted = cast(FittedClassicalModel, model.classical_model)
        logits, uncalibrated, _ = predict_classical_model(
            fitted,
            windows,
            batch.labels,
            participant_ids,
            class_names=model.class_names,
        )
    shifted = logits / model.temperature
    shifted -= shifted.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    calibrated = np.asarray(exponential / exponential.sum(axis=1, keepdims=True), dtype=np.float64)
    report = classification_report(
        batch.labels,
        calibrated,
        participant_ids,
        class_names=model.class_names,
    )
    return logits, uncalibrated, calibrated, report


def _write_seed_result(
    model: _PreparedModel,
    batch: MaterializedWindows,
    *,
    logits: NDArray[np.float64],
    uncalibrated: NDArray[np.float64],
    calibrated: NDArray[np.float64],
    report: Mapping[str, Any],
    array_path: Path,
    record_path: Path,
    output_root: Path,
    inventory: Mapping[str, Any],
    receipt: Mapping[str, Any],
    receipt_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    predictions = calibrated.argmax(axis=1).astype(np.int64, copy=False)
    arrays: dict[str, NDArray[Any]] = {
        "evidence_status": np.asarray([LOCKED_TARGET_EVIDENCE_STATUS]),
        "window_ids": np.asarray(batch.window_ids),
        "participant_ids": np.asarray(batch.participant_ids),
        "true_labels": np.asarray(batch.labels, dtype=np.int64),
        "logits": np.asarray(logits, dtype=np.float64),
        "uncalibrated_probabilities": np.asarray(uncalibrated, dtype=np.float64),
        "calibrated_probabilities": np.asarray(calibrated, dtype=np.float64),
        "predicted_labels": predictions,
    }
    array_sha256 = _write_npz_create_only(array_path, arrays)
    root = output_root.resolve(strict=True)
    array_relative = array_path.resolve(strict=True).relative_to(root).as_posix()
    record: dict[str, Any] = {
        "schema_version": LOCKED_TARGET_RESULT_SCHEMA_VERSION,
        "record_kind": "locked_target_per_seed_result",
        "status": "complete_create_only",
        "evidence_status": LOCKED_TARGET_EVIDENCE_STATUS,
        "target_opening_number": 1,
        "model_id": model.model_id,
        "seed": model.seed,
        "training_regime": model.training_regime,
        "evaluation_device": device.type
        if model.neural_config is not None
        else "classical_runtime",
        "split_manifest_sha256": inventory["split_manifest_sha256"],
        "target_seal_id": inventory["target_seal_id"],
        "final_freeze_inventory_sha256": inventory["inventory_sha256"],
        "frozen_artifact_set_sha256": inventory["frozen_artifact_set_sha256"],
        "opening_receipt_record_sha256": receipt["record_sha256"],
        "opening_receipt_file_sha256": sha256_file(receipt_path),
        "training_configuration_sha256": model.configuration_sha256,
        "checkpoint_sha256": model.checkpoint_sha256,
        "calibrator_sha256": model.calibrator_sha256,
        "calibrator_record_sha256": model.calibrator_record_sha256,
        "class_names": list(model.class_names),
        "window_count": int(batch.labels.size),
        "participant_count": len(set(batch.participant_ids)),
        "ordered_alignment": {
            "window_ids_sha256": canonical_json_sha256(list(batch.window_ids)),
            "participant_ids_sha256": canonical_json_sha256(list(batch.participant_ids)),
            "true_labels_sha256": _array_sha256(batch.labels, dtype=np.dtype("<i8")),
            "logits_sha256": _array_sha256(logits, dtype=np.dtype("<f8")),
            "uncalibrated_probabilities_sha256": _array_sha256(uncalibrated, dtype=np.dtype("<f8")),
            "calibrated_probabilities_sha256": _array_sha256(calibrated, dtype=np.dtype("<f8")),
            "predicted_labels_sha256": _array_sha256(predictions, dtype=np.dtype("<i8")),
        },
        "prediction_array": {
            "path": array_relative,
            "sha256": array_sha256,
            "format": "npz",
        },
        "participant_level_report": dict(report),
        "statistical_unit": "participant",
        "target_information_used_for_model_selection": False,
    }
    record["record_sha256"] = canonical_json_sha256(record)
    atomic_write_json_new(record, record_path, allowed_root=output_root)
    return {
        "model_id": model.model_id,
        "seed": model.seed,
        "array_path": str(array_path),
        "array_sha256": array_sha256,
        "record_path": str(record_path),
        "record_file_sha256": sha256_file(record_path),
        "record_sha256": record["record_sha256"],
    }


def run_locked_target_evaluation(
    target_manifest: Mapping[str, Any],
    unlock_record: Mapping[str, Any] | None,
    *,
    final_freeze_inventory_path: str | Path,
    artifact_root: str | Path,
    receipt_root: str | Path,
    materialize_target: TargetMaterializer,
    output_directory: str | Path,
    output_root: str | Path,
    opened_at_utc: str,
    machine_record_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    """Consume one exact unlock, materialize once, and emit immutable per-seed evidence.

    Validation and output preflight occur while the target remains untouched.  The
    materializer callback is unreachable until the canonical receipt exists.
    """

    inventory_path = Path(final_freeze_inventory_path)
    artifacts = Path(artifact_root)
    receipts = Path(receipt_root).resolve(strict=True)
    output_root_path = Path(output_root).resolve(strict=True)
    output = _safe_output_directory(Path(output_directory), output_root=output_root_path)
    canonical_receipt = receipts / CANONICAL_OPENING_RECEIPT_NAME

    inventory, prepared, expected_subjects = _prepare_inventory(
        inventory_path,
        artifact_root=artifacts,
        target_manifest=target_manifest,
        device=device,
    )
    result_paths = _result_paths(output, prepared)
    authorized = validate_exact_target_unlock(
        target_manifest,
        unlock_record,
        final_freeze_inventory_path=inventory_path,
        artifact_root=artifacts,
    )
    receipt = record_target_opening_once(
        target_manifest,
        unlock_record,
        final_freeze_inventory_path=inventory_path,
        artifact_root=artifacts,
        opening_receipt_path=canonical_receipt,
        receipt_root=receipts,
        opened_at_utc=opened_at_utc,
        machine_record_sha256=machine_record_sha256,
    )

    batch = materialize_target(authorized)
    class_names = prepared[0].class_names
    _validate_materialized_target(
        batch,
        target_subjects=expected_subjects,
        class_names=class_names,
    )
    results: list[dict[str, Any]] = []
    for model, (array_path, record_path) in zip(prepared, result_paths, strict=True):
        logits, uncalibrated, calibrated, report = _predict(model, batch, device=device)
        results.append(
            _write_seed_result(
                model,
                batch,
                logits=logits,
                uncalibrated=uncalibrated,
                calibrated=calibrated,
                report=report,
                array_path=array_path,
                record_path=record_path,
                output_root=output_root_path,
                inventory=inventory,
                receipt=receipt,
                receipt_path=canonical_receipt,
                device=device,
            )
        )
    index: dict[str, Any] = {
        "schema_version": LOCKED_TARGET_RESULT_SCHEMA_VERSION,
        "record_kind": "locked_target_evaluation_index",
        "status": "complete_create_only",
        "evidence_status": LOCKED_TARGET_EVIDENCE_STATUS,
        "target_opening_number": 1,
        "split_manifest_sha256": inventory["split_manifest_sha256"],
        "target_seal_id": inventory["target_seal_id"],
        "final_freeze_inventory_sha256": inventory["inventory_sha256"],
        "frozen_artifact_set_sha256": inventory["frozen_artifact_set_sha256"],
        "opening_receipt_record_sha256": receipt["record_sha256"],
        "opening_receipt_file_sha256": sha256_file(canonical_receipt),
        "model_seed_result_count": len(results),
        "class_names": list(class_names),
        "window_count": int(batch.labels.size),
        "participant_count": len(expected_subjects),
        "results": results,
        "target_information_used_for_model_selection": False,
    }
    index["record_sha256"] = canonical_json_sha256(index)
    index_path = output / "locked_target_evaluation_index.json"
    atomic_write_json_new(index, index_path, allowed_root=output_root_path)
    return {
        "opening_receipt_path": str(canonical_receipt),
        "opening_receipt_sha256": sha256_file(canonical_receipt),
        "index_path": str(index_path),
        "index_file_sha256": sha256_file(index_path),
        "index_record_sha256": index["record_sha256"],
        "result_count": len(results),
    }
