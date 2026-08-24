"""CUDA-only secondary sensor stress evaluation on existing source/target caches.

The command has no raw-data, unlock, or opening interface.  It uses the consumed
opening-1 index as the target clean reference and never reruns clean target
inference.  Every corruption is applied in physical space before the frozen
source-training normalizer.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from inclusive_shift_har.artifacts.final_freeze import validate_final_freeze_inventory_file
from inclusive_shift_har.artifacts.sensor_stress import (
    create_secondary_stress_artifacts,
    create_source_clean_reference_artifacts,
)
from inclusive_shift_har.data.materialize import MaterializedWindows
from inclusive_shift_har.evaluation._strict_config import require_utc_timestamp
from inclusive_shift_har.evaluation.locked_target import _prepare_model, _PreparedModel
from inclusive_shift_har.evaluation.postconfirmatory_inputs import (
    ConsumedTargetContext,
    TargetCleanReference,
    load_consumed_target_context,
    load_target_clean_reference,
)
from inclusive_shift_har.evaluation.sensor_reliability import (
    SensorReliabilityConfig,
    apply_sensor_reliability_stress,
    load_sensor_reliability_config,
)
from inclusive_shift_har.experiments.postconfirmatory_cache import (
    PreparedPrimaryCacheEvidence,
    load_prepared_primary_cache,
)
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)
from inclusive_shift_har.training.engine import (
    TrainingConfig,
    predict_model,
    reconstruct_checkpoint,
)


class PostconfirmatoryStressError(RuntimeError):
    """Raised when secondary stress execution would violate frozen lineage."""


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PostconfirmatoryStressError(f"{name} must be an object")
    return value


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise PostconfirmatoryStressError(f"{name} must be a full SHA-256")
    normalized = value.casefold()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise PostconfirmatoryStressError(f"{name} must be a full SHA-256")
    return normalized


def _source_hash_from_pinned_primary_cache_record(
    record_path: str | Path,
    *,
    expected_record_file_sha256: str,
    artifact_root: Path,
) -> str:
    """Read raw lineage only after authenticating the externally pinned file."""

    expected_file_hash = _sha256(
        expected_record_file_sha256, name="expected primary cache record file hash"
    )
    raw = Path(record_path)
    candidate = raw if raw.is_absolute() else artifact_root / raw
    if candidate.is_symlink():
        raise PostconfirmatoryStressError("primary cache record may not be a symlink")
    path = candidate.resolve(strict=True)
    try:
        path.relative_to(artifact_root)
    except ValueError as exc:
        raise PostconfirmatoryStressError("primary cache record escapes artifact_root") from exc
    if path.is_symlink() or not path.is_file():
        raise PostconfirmatoryStressError("primary cache record must be a regular file")
    if sha256_file(path) != expected_file_hash:
        raise PostconfirmatoryStressError("primary cache record file hash changed")
    record = _mapping(load_json_strict(path), name="primary cache record")
    raw_sensor = _mapping(record.get("raw_sensor_csv"), name="primary cache raw lineage")
    return _sha256(raw_sensor.get("sha256"), name="primary cache source artifact hash")


def _load_pinned_primary_stress_caches(
    *,
    primary_cache_record_path: str | Path,
    expected_primary_cache_record_file_sha256: str,
    opening_receipt_path: str | Path,
    locked_target_index_path: str | Path,
    artifact_root: Path,
    expected_split_manifest_sha256: str,
    source_partition: str,
    target_partition: str,
) -> tuple[PreparedPrimaryCacheEvidence, PreparedPrimaryCacheEvidence]:
    pinned_file_hash = _sha256(
        expected_primary_cache_record_file_sha256,
        name="expected primary cache record file hash",
    )
    source_artifact_sha256 = _source_hash_from_pinned_primary_cache_record(
        primary_cache_record_path,
        expected_record_file_sha256=pinned_file_hash,
        artifact_root=artifact_root,
    )
    source = load_prepared_primary_cache(
        record_path=primary_cache_record_path,
        expected_record_file_sha256=pinned_file_hash,
        partition=source_partition,
        opening_receipt_path=opening_receipt_path,
        locked_target_index_path=locked_target_index_path,
        artifact_root=artifact_root,
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        expected_source_artifact_sha256=source_artifact_sha256,
    )
    target = load_prepared_primary_cache(
        record_path=primary_cache_record_path,
        expected_record_file_sha256=pinned_file_hash,
        partition=target_partition,
        opening_receipt_path=opening_receipt_path,
        locked_target_index_path=locked_target_index_path,
        artifact_root=artifact_root,
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        expected_source_artifact_sha256=source_artifact_sha256,
    )
    if (
        source.record_path != target.record_path
        or source.record_file_sha256 != target.record_file_sha256
        or source.record_sha256 != target.record_sha256
    ):
        raise PostconfirmatoryStressError(
            "source and target caches do not share one pinned primary cache record"
        )
    return source, target


def _output_directory(path: Path, *, output_root: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(output_root)
    except ValueError as exc:
        raise PostconfirmatoryStressError("output directory escapes output_root") from exc
    if resolved.is_symlink() or not resolved.is_dir():
        raise PostconfirmatoryStressError("output directory is unsafe")
    return resolved


def _eligible_entries(
    inventory: Mapping[str, Any], config: SensorReliabilityConfig
) -> list[Mapping[str, Any]]:
    values = inventory.get("models")
    if not isinstance(values, list):
        raise PostconfirmatoryStressError("final freeze inventory lacks model entries")
    selected: list[Mapping[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for value in values:
        entry = _mapping(value, name="frozen model entry")
        configuration = _mapping(
            entry.get("training_configuration"), name="frozen training configuration"
        )
        if (
            entry.get("training_regime") != config.required_training_regime
            or configuration.get("model_name") not in config.eligible_training_model_names
        ):
            continue
        model_id = entry.get("model_id")
        seed = entry.get("seed")
        if (
            not isinstance(model_id, str)
            or not model_id
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or (model_id, seed) in seen
        ):
            raise PostconfirmatoryStressError("eligible frozen model/seed identity is invalid")
        seen.add((model_id, seed))
        selected.append(entry)
    selected.sort(key=lambda value: (str(value["model_id"]).casefold(), int(value["seed"])))
    if not selected:
        raise PostconfirmatoryStressError(
            "freeze inventory has no entry in the predeclared compact model family"
        )
    return selected


def _planned_paths(
    entries: list[Mapping[str, Any]],
    config: SensorReliabilityConfig,
    output: Path,
) -> tuple[list[Path], Path]:
    paths: list[Path] = []
    for entry in entries:
        stem = f"{entry['model_id']}--seed-{entry['seed']}"
        paths.extend(
            [
                output / f"{stem}--source--clean.predictions.npz",
                output / f"{stem}--source--clean.result.json",
            ]
        )
        for cohort in ("source", "target"):
            for condition in config.conditions:
                condition_stem = f"{stem}--{cohort}--{condition.condition_id}"
                paths.extend(
                    [
                        output / f"{condition_stem}.predictions.npz",
                        output / f"{condition_stem}.result.json",
                    ]
                )
    index_path = output / "sensor_reliability_stress_index.json"
    for path in [*paths, index_path]:
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to overwrite secondary stress evidence: {path}")
    return paths, index_path


def _calibrated_prediction(
    neural: torch.nn.Module,
    prepared: _PreparedModel,
    batch: MaterializedWindows,
    signals: NDArray[np.float32],
    *,
    device: torch.device,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    configuration = cast(TrainingConfig, prepared.neural_config)
    normalized = prepared.normalizer.transform(signals)
    logits, _, _ = predict_model(
        neural,
        normalized,
        batch.labels,
        list(batch.participant_ids),
        class_names=prepared.class_names,
        batch_size=configuration.batch_size,
        device=device,
        mixed_precision=configuration.mixed_precision,
        zero_channel_indices=configuration.zero_channel_indices,
    )
    shifted = logits / prepared.temperature
    shifted -= shifted.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    probabilities = np.asarray(
        exponential / exponential.sum(axis=1, keepdims=True), dtype=np.float64
    )
    return logits, probabilities


def _reference_index_entry(
    *,
    model_id: str,
    seed: int,
    record_path: Path,
    record: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    prediction = _mapping(record.get("prediction_artifact"), name="prediction artifact")
    return {
        "model_id": model_id,
        "seed": seed,
        "cohort": "source",
        "record_path": record_path.relative_to(output_root).as_posix(),
        "record_file_sha256": sha256_file(record_path),
        "record_sha256": record["record_sha256"],
        "prediction_sha256": prediction["sha256"],
    }


def _target_reference_index_entry(
    reference: TargetCleanReference, *, artifact_root: Path
) -> dict[str, Any]:
    return {
        "model_id": reference.model_id,
        "seed": reference.seed,
        "cohort": "target",
        "record_path": reference.record_path.relative_to(artifact_root).as_posix(),
        "record_file_sha256": reference.record_file_sha256,
        "record_sha256": reference.record_sha256,
        "prediction_sha256": reference.prediction_sha256,
        "source": "locked_target_opening_1_no_clean_rerun",
    }


def _stress_index_entry(
    *,
    model_id: str,
    seed: int,
    cohort: str,
    condition_id: str,
    record_path: Path,
    record: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    prediction = _mapping(record.get("prediction_artifact"), name="stress prediction artifact")
    return {
        "model_id": model_id,
        "seed": seed,
        "cohort": cohort,
        "condition_id": condition_id,
        "record_path": record_path.relative_to(output_root).as_posix(),
        "record_file_sha256": sha256_file(record_path),
        "record_sha256": record["record_sha256"],
        "prediction_sha256": prediction["sha256"],
    }


def run_postconfirmatory_sensor_stress(
    *,
    stress_config_path: str | Path,
    final_freeze_inventory_path: str | Path,
    opening_receipt_path: str | Path,
    locked_target_index_path: str | Path,
    primary_cache_record_path: str | Path,
    expected_primary_cache_record_file_sha256: str,
    artifact_root: str | Path,
    output_directory: str | Path,
    output_root: str | Path,
    created_at_utc: str,
    device: torch.device,
) -> dict[str, Any]:
    """Run all locked stresses on source validation and consumed target caches."""

    if device.type != "cuda" or not torch.cuda.is_available():
        raise PostconfirmatoryStressError(
            "CUDA is required before any cache, target index, or checkpoint is opened"
        )
    timestamp = require_utc_timestamp(created_at_utc, location="created_at_utc")
    config = load_sensor_reliability_config(stress_config_path)
    root = Path(artifact_root).resolve(strict=True)
    outputs_root = Path(output_root).resolve(strict=True)
    try:
        outputs_root.relative_to(root)
    except ValueError as exc:
        raise PostconfirmatoryStressError("output_root must be beneath artifact_root") from exc
    output = _output_directory(Path(output_directory), output_root=outputs_root)
    context: ConsumedTargetContext = load_consumed_target_context(
        opening_receipt_path=opening_receipt_path,
        locked_target_index_path=locked_target_index_path,
        artifact_root=root,
    )
    inventory_candidate = Path(final_freeze_inventory_path)
    if not inventory_candidate.is_absolute():
        inventory_candidate = root / inventory_candidate
    if inventory_candidate.is_symlink():
        raise PostconfirmatoryStressError("final freeze inventory may not be a symlink")
    inventory_path = inventory_candidate.resolve(strict=True)
    try:
        inventory_path.relative_to(root)
    except ValueError as exc:
        raise PostconfirmatoryStressError("final freeze inventory escapes artifact_root") from exc
    inventory_report = validate_final_freeze_inventory_file(inventory_path, artifact_root=root)
    if not inventory_report.valid:
        raise PostconfirmatoryStressError(
            f"final freeze inventory is invalid: {inventory_report.to_dict()}"
        )
    inventory = _mapping(load_json_strict(inventory_path), name="final freeze inventory")
    if inventory.get("inventory_sha256") != context.index.get("final_freeze_inventory_sha256"):
        raise PostconfirmatoryStressError("freeze inventory differs from opening-1 index")
    entries = _eligible_entries(inventory, config)
    _, index_path = _planned_paths(entries, config, output)

    split_hash = str(inventory["split_manifest_sha256"])
    source_cache_evidence, target_cache_evidence = _load_pinned_primary_stress_caches(
        primary_cache_record_path=primary_cache_record_path,
        expected_primary_cache_record_file_sha256=(expected_primary_cache_record_file_sha256),
        opening_receipt_path=opening_receipt_path,
        locked_target_index_path=locked_target_index_path,
        artifact_root=root,
        expected_split_manifest_sha256=split_hash,
        source_partition=config.source_partition,
        target_partition=config.target_partition,
    )
    source_cache = source_cache_evidence.cache
    target_cache = target_cache_evidence.cache
    if set(source_cache.batch.participant_ids) & set(target_cache.batch.participant_ids):
        raise PostconfirmatoryStressError("source and target cache participants overlap")
    if set(source_cache.batch.window_ids) & set(target_cache.batch.window_ids):
        raise PostconfirmatoryStressError("source and target cache window IDs overlap")
    target_subjects = frozenset(target_cache.batch.participant_ids)
    prepared_models = [
        _prepare_model(
            entry,
            inventory=inventory,
            artifact_root=root,
            target_subjects=target_subjects,
        )
        for entry in entries
    ]
    if any(model.neural_config is None for model in prepared_models):
        raise PostconfirmatoryStressError("stress family unexpectedly contains a non-neural model")
    for model in prepared_models:
        if set(source_cache.batch.participant_ids) & set(model.normalizer.training_participants):
            raise PostconfirmatoryStressError(
                f"{model.model_id} source stress cache overlaps normalization-fit participants"
            )
    target_references = {
        (model.model_id, model.seed): load_target_clean_reference(
            context, model_id=model.model_id, seed=model.seed
        )
        for model in prepared_models
    }

    source_references: list[dict[str, Any]] = []
    target_reference_entries = [
        _target_reference_index_entry(reference, artifact_root=root)
        for reference in target_references.values()
    ]
    stress_entries: list[dict[str, Any]] = []
    for prepared in prepared_models:
        configuration = cast(TrainingConfig, prepared.neural_config)
        neural, payload = reconstruct_checkpoint(prepared.checkpoint_path, device=device)
        if canonical_json_sha256(payload["configuration"]) != prepared.configuration_sha256:
            raise PostconfirmatoryStressError(
                f"{prepared.model_id} reconstructed configuration differs from freeze"
            )
        original_cudnn = torch.backends.cudnn.enabled
        try:
            torch.backends.cudnn.enabled = not configuration.disable_cudnn
            source_logits, source_probabilities = _calibrated_prediction(
                neural,
                prepared,
                source_cache.batch,
                source_cache.batch.signals,
                device=device,
            )
            stem = f"{prepared.model_id}--seed-{prepared.seed}"
            source_prediction_path = output / f"{stem}--source--clean.predictions.npz"
            source_record_path = output / f"{stem}--source--clean.result.json"
            normalization_hash = canonical_json_sha256(prepared.normalizer.to_dict())
            source_record = create_source_clean_reference_artifacts(
                prediction_path=source_prediction_path,
                record_path=source_record_path,
                allowed_root=outputs_root,
                logits=source_logits,
                probabilities=source_probabilities,
                labels=source_cache.batch.labels,
                participant_ids=source_cache.batch.participant_ids,
                window_ids=source_cache.batch.window_ids,
                class_names=prepared.class_names,
                model_id=prepared.model_id,
                seed=prepared.seed,
                stress_config_sha256=config.config_sha256,
                final_freeze_inventory_sha256=str(inventory["inventory_sha256"]),
                frozen_checkpoint_sha256=prepared.checkpoint_sha256,
                frozen_normalization_sha256=normalization_hash,
                frozen_calibrator_sha256=prepared.calibrator_sha256,
                locked_target_index_record_sha256=str(context.index["record_sha256"]),
                primary_cache_record_file_sha256=(source_cache_evidence.record_file_sha256),
                primary_cache_record_sha256=source_cache_evidence.record_sha256,
                created_at_utc=timestamp,
            )
            source_references.append(
                _reference_index_entry(
                    model_id=prepared.model_id,
                    seed=prepared.seed,
                    record_path=source_record_path,
                    record=source_record,
                    output_root=root,
                )
            )
            base_hashes = {
                "source": str(
                    _mapping(source_record["prediction_artifact"], name="source prediction")[
                        "sha256"
                    ]
                ),
                "target": target_references[(prepared.model_id, prepared.seed)].prediction_sha256,
            }
            for cohort, cache in (
                ("source", source_cache),
                ("target", target_cache),
            ):
                for condition in config.conditions:
                    corrupted = apply_sensor_reliability_stress(
                        cache.batch.signals,
                        cache.batch.window_ids,
                        config=config,
                        condition_id=condition.condition_id,
                    )
                    logits, probabilities = _calibrated_prediction(
                        neural,
                        prepared,
                        cache.batch,
                        np.asarray(corrupted.windows, dtype=np.float32),
                        device=device,
                    )
                    condition_stem = f"{stem}--{cohort}--{condition.condition_id}"
                    prediction_path = output / f"{condition_stem}.predictions.npz"
                    record_path = output / f"{condition_stem}.result.json"
                    stress_record = create_secondary_stress_artifacts(
                        prediction_path=prediction_path,
                        record_path=record_path,
                        allowed_root=outputs_root,
                        logits=logits,
                        probabilities=probabilities,
                        labels=cache.batch.labels,
                        participant_ids=cache.batch.participant_ids,
                        window_ids=cache.batch.window_ids,
                        class_names=prepared.class_names,
                        config=config,
                        condition_metadata=corrupted.metadata,
                        cohort=cast(Any, cohort),
                        model_id=prepared.model_id,
                        seed=prepared.seed,
                        frozen_checkpoint_sha256=prepared.checkpoint_sha256,
                        frozen_normalization_sha256=normalization_hash,
                        frozen_calibrator_sha256=prepared.calibrator_sha256,
                        base_prediction_sha256=base_hashes[cohort],
                        primary_confirmatory_completion_record_sha256=str(
                            context.index["record_sha256"]
                        ),
                        primary_cache_record_file_sha256=(source_cache_evidence.record_file_sha256),
                        primary_cache_record_sha256=source_cache_evidence.record_sha256,
                        created_at_utc=timestamp,
                    )
                    stress_entries.append(
                        _stress_index_entry(
                            model_id=prepared.model_id,
                            seed=prepared.seed,
                            cohort=cohort,
                            condition_id=condition.condition_id,
                            record_path=record_path,
                            record=stress_record,
                            output_root=root,
                        )
                    )
        finally:
            torch.backends.cudnn.enabled = original_cudnn
            del neural
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    index: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "secondary_sensor_reliability_index",
        "status": "complete_create_only",
        "created_at_utc": timestamp,
        "track_role": "secondary_post_confirmatory",
        "primary_claim_eligible": False,
        "used_for_model_selection": False,
        "target_clean_inference_rerun": False,
        "raw_target_materialization_invoked": False,
        "opening_or_unlock_invoked": False,
        "stress_config_sha256": config.config_sha256,
        "model_family_id": config.model_family_id,
        "eligible_training_model_names": list(config.eligible_training_model_names),
        "final_freeze_inventory_sha256": inventory["inventory_sha256"],
        "frozen_artifact_set_sha256": inventory["frozen_artifact_set_sha256"],
        "opening_receipt_record_sha256": context.receipt["record_sha256"],
        "locked_target_index_record_sha256": context.index["record_sha256"],
        "primary_cache_record_path": source_cache_evidence.record_path.relative_to(root).as_posix(),
        "primary_cache_record_file_sha256": source_cache_evidence.record_file_sha256,
        "primary_cache_record_sha256": source_cache_evidence.record_sha256,
        "class_names": list(context.class_names),
        "cache_evidence": {
            "source": {
                "partition": config.source_partition,
                "array_sha256": source_cache.array_sha256,
                "metadata_sha256": source_cache.metadata_sha256,
            },
            "target": {
                "partition": config.target_partition,
                "array_sha256": target_cache.array_sha256,
                "metadata_sha256": target_cache.metadata_sha256,
                "ordered_identity_validated_against_opening_1": True,
            },
        },
        "model_seed_count": len(prepared_models),
        "condition_count": len(config.conditions),
        "stress_result_count": len(stress_entries),
        "source_clean_references": source_references,
        "target_clean_references": target_reference_entries,
        "stress_results": stress_entries,
        "normalization": "frozen_source_training_statistics_no_refit",
        "calibration": "frozen_source_validation_temperature_no_refit",
        "threshold_selection": "none",
        "statistical_unit": "participant",
    }
    index["record_sha256"] = canonical_json_sha256(index)
    atomic_write_json_new(index, index_path, allowed_root=outputs_root)
    return {
        "index_path": str(index_path),
        "index_file_sha256": sha256_file(index_path),
        "index_record_sha256": index["record_sha256"],
        "stress_result_count": len(stress_entries),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress-config", type=Path, required=True)
    parser.add_argument("--final-freeze-inventory", type=Path, required=True)
    parser.add_argument("--opening-receipt", type=Path, required=True)
    parser.add_argument("--locked-target-index", type=Path, required=True)
    parser.add_argument("--primary-cache-record", type=Path, required=True)
    parser.add_argument("--expected-primary-cache-record-file-sha256", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--created-at-utc", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_postconfirmatory_sensor_stress(
        stress_config_path=args.stress_config,
        final_freeze_inventory_path=args.final_freeze_inventory,
        opening_receipt_path=args.opening_receipt,
        locked_target_index_path=args.locked_target_index,
        primary_cache_record_path=args.primary_cache_record,
        expected_primary_cache_record_file_sha256=(args.expected_primary_cache_record_file_sha256),
        artifact_root=args.artifact_root,
        output_directory=args.output_directory,
        output_root=args.output_root,
        created_at_utc=args.created_at_utc,
        device=torch.device("cuda"),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
