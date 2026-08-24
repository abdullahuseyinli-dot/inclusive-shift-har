"""One-scenario CUDA runner for the post-confirmatory few-person curve."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from inclusive_shift_har.data.inclusivehar import INCLUSIVEHAR_PRIMARY_CHANNELS
from inclusive_shift_har.data.materialize import MaterializedWindows
from inclusive_shift_har.data.windowing import WindowRecord
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.evaluation.source_calibration import (
    load_source_temperature_calibrator_file,
)
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)
from inclusive_shift_har.models.common import HAROutput, trainable_parameter_count
from inclusive_shift_har.preprocessing.normalization import ChannelStandardizer
from inclusive_shift_har.protocols.few_person import (
    FEW_PERSON_EVIDENCE_STATUS,
    FEW_PERSON_PROTOCOL_ID,
    FewPersonProtocolError,
    build_few_person_manifest,
    write_few_person_manifest_new,
)
from inclusive_shift_har.training.engine import (
    TrainingConfig,
    configure_determinism,
    predict_model,
    reconstruct_checkpoint,
    training_config_from_dict,
)


class FewPersonRunError(RuntimeError):
    """Raised when a few-person run violates its frozen boundary."""


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FewPersonRunError(f"{name} must be an object")
    return value


def _json_object(path: str | Path, *, name: str) -> dict[str, Any]:
    return dict(_mapping(load_json_strict(path), name=name))


def _self_hash(record: Mapping[str, Any], *, field: str, name: str) -> str:
    claimed = record.get(field)
    body = dict(record)
    body.pop(field, None)
    if not isinstance(claimed, str) or claimed != canonical_json_sha256(body):
        raise FewPersonRunError(f"{name} self-hash does not validate")
    return claimed


def _full_commit(value: str) -> str:
    normalized = value.casefold()
    if len(normalized) != 40 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise FewPersonRunError("code commit must be a full 40-character Git object ID")
    return normalized


def _confined(path: Path, *, root: Path, name: str, must_exist: bool) -> Path:
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=must_exist)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise FewPersonRunError(f"{name} escapes its declared root") from exc
    return resolved


def _selected(
    plan: Mapping[str, Any], *, fold_id: str, k: int, model_id: str, seed: int
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    scenarios = plan.get("scenarios")
    models = plan.get("models")
    if not isinstance(scenarios, list) or not isinstance(models, list):
        raise FewPersonRunError("few-person manifest lacks scenarios or models")
    scenario = next(
        (
            _mapping(value, name="few-person scenario")
            for value in scenarios
            if isinstance(value, Mapping)
            and value.get("fold_id") == fold_id
            and value.get("k") == k
        ),
        None,
    )
    model = next(
        (
            _mapping(value, name="few-person model")
            for value in models
            if isinstance(value, Mapping)
            and value.get("model_id") == model_id
            and value.get("seed") == seed
        ),
        None,
    )
    if scenario is None:
        raise FewPersonRunError(f"scenario {fold_id} k={k} is not predeclared")
    if model is None:
        raise FewPersonRunError(f"model/seed {model_id}/{seed} is not predeclared")
    inclusion = set(cast(list[str], scenario["target_inclusion_subjects"]))
    evaluation = set(cast(list[str], scenario["evaluation_subjects"]))
    unused = set(cast(list[str], scenario["unused_target_subjects"]))
    if inclusion & evaluation or inclusion & unused or evaluation & unused:
        raise FewPersonRunError("target scenario participant sets overlap")
    if len(inclusion) != k or inclusion | evaluation | unused != {
        str(value) for value in range(11, 21)
    }:
        raise FewPersonRunError("target scenario does not partition the target cohort")
    for forbidden in (
        "validation_subjects",
        "calibration_subjects",
        "threshold_selection_subjects",
    ):
        if scenario.get(forbidden) != []:
            raise FewPersonRunError(f"evaluation isolation changed: {forbidden}")
    return scenario, model


def _window_records(
    split: Mapping[str, Any],
    *,
    subjects: set[str],
    ontology_track: str,
    expected_count: int,
    expected_ids_sha256: str,
) -> tuple[WindowRecord, ...]:
    windows = split.get("windows")
    if not isinstance(windows, list):
        raise FewPersonRunError("parent split lacks window records")
    records = tuple(
        WindowRecord(**dict(value))
        for value in windows
        if isinstance(value, Mapping)
        and value.get("partition") == "target_sealed"
        and value.get("subject_id") in subjects
        and ontology_track in cast(Mapping[str, str], value.get("canonical_labels", {}))
    )
    records = tuple(sorted(records, key=lambda item: item.start_row_inclusive))
    ids = sorted(record.window_id for record in records)
    if len(records) != expected_count or canonical_json_sha256(ids) != expected_ids_sha256:
        raise FewPersonRunError("scenario window count/hash differs from the parent split")
    for previous, current in pairwise(records):
        if current.start_row_inclusive <= previous.end_row_inclusive:
            raise FewPersonRunError("selected target windows overlap")
    return records


def _materialize_authorized_records(
    csv_path: Path,
    records: Sequence[WindowRecord],
    *,
    expected_source_sha256: str,
    class_names: tuple[str, ...],
    ontology_track: str,
) -> MaterializedWindows:
    """Read only records authorized by the post-confirmatory scenario manifest."""

    if sha256_file(csv_path) != expected_source_sha256:
        raise FewPersonRunError("InclusiveHAR source artifact hash mismatch")
    if not records:
        raise FewPersonRunError("authorized target record set is empty")
    class_to_index = {name: index for index, name in enumerate(class_names)}
    signals = np.empty((len(records), 128, 6), dtype=np.float32)
    labels = np.empty(len(records), dtype=np.int64)
    window_ids: list[str] = []
    participant_ids: list[str] = []
    released_labels: list[str] = []
    next_window = 0
    active_values: list[list[float]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise FewPersonRunError("InclusiveHAR CSV is empty") from exc
        header_index = {name: index for index, name in enumerate(header)}
        required = set(INCLUSIVEHAR_PRIMARY_CHANNELS) | {"label", "UserID"}
        missing = sorted(required - set(header_index))
        if missing:
            raise FewPersonRunError(f"InclusiveHAR CSV lacks columns: {missing}")
        channel_indices = [header_index[name] for name in INCLUSIVEHAR_PRIMARY_CHANNELS]
        for row_index, row in enumerate(reader, start=1):
            if next_window >= len(records):
                break
            active = records[next_window]
            if row_index < active.start_row_inclusive:
                continue
            if row_index > active.end_row_inclusive:
                raise FewPersonRunError("materializer skipped an authorized target window")
            if row[header_index["label"]].strip() != active.activity_label:
                raise FewPersonRunError("target window crosses an activity boundary")
            try:
                observed_subject = str(int(row[header_index["UserID"]].strip()))
                active_values.append([float(row[index]) for index in channel_indices])
            except ValueError as exc:
                raise FewPersonRunError("target window contains invalid model input") from exc
            if observed_subject != active.subject_id:
                raise FewPersonRunError("target window crosses a participant boundary")
            if row_index == active.end_row_inclusive:
                values = np.asarray(active_values, dtype=np.float32)
                if values.shape != (128, 6) or not np.isfinite(values).all():
                    raise FewPersonRunError("target window violates finite [128,6]")
                canonical = active.canonical_labels[ontology_track]
                if canonical not in class_to_index:
                    raise FewPersonRunError("target label lies outside the frozen schema")
                signals[next_window] = values
                labels[next_window] = class_to_index[canonical]
                window_ids.append(active.window_id)
                participant_ids.append(active.subject_id)
                released_labels.append(active.activity_label)
                next_window += 1
                active_values = []
    if next_window != len(records):
        raise FewPersonRunError("CSV ended before all authorized windows were read")
    return MaterializedWindows(
        signals=signals,
        labels=labels,
        window_ids=tuple(window_ids),
        participant_ids=tuple(participant_ids),
        released_labels=tuple(released_labels),
        partitions=("target_postconfirmatory",) * len(records),
        class_names=class_names,
        ontology_track=ontology_track,
    )


def _adaptation_seed(base_seed: int, fold_id: str, k: int) -> int:
    token = f"inclusive-shift-har|few-person-v1|{base_seed}|{fold_id}|k={k}"
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "big")


def _ablate(signals: Tensor, indices: tuple[int, ...]) -> Tensor:
    if not indices:
        return signals
    result = signals.clone()
    result[:, :, list(indices)] = 0.0
    return result


def _fine_tune_fixed(
    model: nn.Module,
    windows: NDArray[np.float32],
    labels: NDArray[np.int64],
    *,
    config: TrainingConfig,
    adaptation_seed: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], torch.optim.Optimizer]:
    configure_determinism(adaptation_seed)
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(windows)),
        torch.from_numpy(np.ascontiguousarray(labels)),
    )
    generator = torch.Generator().manual_seed(adaptation_seed + 1)
    loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        cast(Any, dataset),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision == "float16")  # type: ignore[attr-defined]
    amp_dtype = torch.bfloat16 if config.mixed_precision == "bfloat16" else torch.float16
    amp_enabled = config.mixed_precision != "disabled"
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        examples = 0
        for signals, targets in loader:
            signals = _ablate(signals.to(device), config.zero_channel_indices)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                output = cast(HAROutput, model(signals))
                loss = F.cross_entropy(output.logits, targets)
            cast(Any, scaler.scale(loss)).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            count = targets.numel()
            total_loss += float(loss.detach()) * count
            examples += count
        history.append(
            {
                "epoch": epoch,
                "training_cross_entropy": total_loss / examples,
                "checkpoint_selected": epoch == config.epochs,
                "validation_accessed": False,
            }
        )
    return history, optimizer


def _write_npz_new(path: Path, arrays: Mapping[str, NDArray[Any]]) -> str:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **cast(dict[str, Any], dict(arrays)))
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_file(path)


def run_few_person_scenario(
    *,
    manifest_path: Path,
    split_manifest_path: Path,
    opening_receipt_path: Path,
    zero_shot_index_path: Path,
    final_freeze_inventory_path: Path,
    raw_csv_path: Path,
    artifact_root: Path,
    fold_id: str,
    k: int,
    model_id: str,
    seed: int,
    code_commit: str,
    output_directory: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Adapt one frozen neural model and evaluate one untouched outer pair."""

    plan = _json_object(manifest_path, name="few-person manifest")
    plan_sha256 = _self_hash(plan, field="manifest_sha256", name="few-person manifest")
    required = {
        "protocol_id": FEW_PERSON_PROTOCOL_ID,
        "status": "ready_postconfirmatory_metadata_only_no_scenario_run",
        "evidence_status": FEW_PERSON_EVIDENCE_STATUS,
        "target_metrics_or_predictions_used_for_design_or_selection": False,
        "target_raw_values_accessed_during_manifest_build": False,
    }
    if any(plan.get(key) != value for key, value in required.items()):
        raise FewPersonRunError("few-person manifest contract mismatch")
    for path, field in (
        (opening_receipt_path, "opening_receipt_file_sha256"),
        (zero_shot_index_path, "zero_shot_index_file_sha256"),
        (final_freeze_inventory_path, "final_freeze_inventory_file_sha256"),
    ):
        if sha256_file(path) != plan.get(field):
            raise FewPersonRunError(f"post-confirmatory evidence hash changed: {field}")
    split = _json_object(split_manifest_path, name="parent split manifest")
    split_sha256 = _self_hash(split, field="split_manifest_sha256", name="parent split")
    if split_sha256 != plan.get("split_manifest_sha256"):
        raise FewPersonRunError("parent split differs from few-person manifest")
    scenario, model_entry = _selected(plan, fold_id=fold_id, k=k, model_id=model_id, seed=seed)
    commit = _full_commit(code_commit)
    if not torch.cuda.is_available():
        raise FewPersonRunError(
            "few-person neural execution requires CUDA; CPU fallback is forbidden"
        )

    root = artifact_root.resolve(strict=True)
    checkpoint_record = _mapping(model_entry["checkpoint"], name="checkpoint")
    calibrator_record = _mapping(model_entry["calibrator"], name="calibrator")
    checkpoint_path = _confined(
        root / str(checkpoint_record["path"]), root=root, name="checkpoint", must_exist=True
    )
    calibrator_path = _confined(
        root / str(calibrator_record["path"]), root=root, name="calibrator", must_exist=True
    )
    if sha256_file(checkpoint_path) != checkpoint_record.get("sha256"):
        raise FewPersonRunError("base checkpoint hash changed")
    if sha256_file(calibrator_path) != calibrator_record.get("sha256"):
        raise FewPersonRunError("source calibrator hash changed")
    device = torch.device("cuda")
    model, checkpoint = reconstruct_checkpoint(checkpoint_path, device=device)
    base_configuration = _mapping(checkpoint.get("configuration"), name="base configuration")
    if canonical_json_sha256(base_configuration) != model_entry.get(
        "training_configuration_sha256"
    ):
        raise FewPersonRunError("base checkpoint configuration differs from freeze")
    config = training_config_from_dict(dict(base_configuration))
    if config.checkpoint_selection_rule != "fixed_last_epoch" or config.seed != seed:
        raise FewPersonRunError("base checkpoint is not the predeclared fixed source model")
    normalizer = ChannelStandardizer.from_dict(
        dict(_mapping(checkpoint.get("normalization"), name="checkpoint normalization"))
    )
    expected_normalization = set(cast(list[str], scenario["normalization_fit_subjects"]))
    if set(normalizer.training_participants) != expected_normalization or any(
        participant in {str(value) for value in range(11, 21)}
        for participant in normalizer.training_participants
    ):
        raise FewPersonRunError("normalization is not the frozen source-training-only transform")
    class_names = tuple(str(value) for value in checkpoint["label_schema"])
    calibrator_payload, calibrator = load_source_temperature_calibrator_file(
        calibrator_path,
        expected_checkpoint_sha256=str(checkpoint_record["sha256"]),
        expected_training_configuration_sha256=str(model_entry["training_configuration_sha256"]),
        expected_split_manifest_sha256=split_sha256,
    )
    if calibrator_payload.get("target_subject_or_window_records_used") is not False:
        raise FewPersonRunError("calibrator is not source-only")

    output = _confined(
        output_directory, root=output_root, name="output directory", must_exist=False
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite few-person output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    inclusion = set(cast(list[str], scenario["target_inclusion_subjects"]))
    evaluation = set(cast(list[str], scenario["evaluation_subjects"]))
    ontology_track = str(plan["ontology_track"])
    train_records = _window_records(
        split,
        subjects=inclusion,
        ontology_track=ontology_track,
        expected_count=int(scenario["target_inclusion_window_count"]),
        expected_ids_sha256=str(scenario["target_inclusion_window_ids_sha256"]),
    )
    source_sha256 = str(split["source_artifact_sha256"])
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    original_cudnn = torch.backends.cudnn.enabled
    try:
        torch.backends.cudnn.enabled = not config.disable_cudnn
        # Only inclusion values are opened before fixed-epoch adaptation.
        train_batch = _materialize_authorized_records(
            raw_csv_path,
            train_records,
            expected_source_sha256=source_sha256,
            class_names=class_names,
            ontology_track=ontology_track,
        )
        if set(train_batch.participant_ids) != inclusion:
            raise FewPersonRunError("training materialization differs from inclusion participants")
        adaptation_seed = _adaptation_seed(seed, fold_id, k)
        history, optimizer = _fine_tune_fixed(
            model,
            normalizer.transform(train_batch.signals),
            train_batch.labels,
            config=config,
            adaptation_seed=adaptation_seed,
            device=device,
        )
        checkpoint_output = output / "adapted.pt"
        checkpoint_payload = {
            "schema_version": "1.0.0",
            "record_kind": "few_person_adapted_neural_checkpoint",
            "evidence_status": FEW_PERSON_EVIDENCE_STATUS,
            "model_id": model_id,
            "seed": seed,
            "adaptation_seed": adaptation_seed,
            "fold_id": fold_id,
            "k": k,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "normalization": normalizer.to_dict(),
            "label_schema": list(class_names),
            "base_checkpoint_sha256": checkpoint_record["sha256"],
            "base_training_configuration": dict(base_configuration),
            "base_training_configuration_sha256": model_entry["training_configuration_sha256"],
            "few_person_manifest_sha256": plan_sha256,
            "split_manifest_sha256": split_sha256,
            "code_commit": commit,
            "checkpoint_selection_rule": "fixed_last_epoch_no_validation",
            "training_history": history,
            "rng_states": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all(),
            },
            "parameter_count": trainable_parameter_count(model),
        }
        with checkpoint_output.open("xb") as stream:
            torch.save(checkpoint_payload, stream)
        checkpoint_sha256 = sha256_file(checkpoint_output)

        # Evaluation records, labels, and values remain unopened until all
        # adaptation epochs finish; participant disjointness was predeclared.
        evaluation_records = _window_records(
            split,
            subjects=evaluation,
            ontology_track=ontology_track,
            expected_count=int(scenario["evaluation_window_count"]),
            expected_ids_sha256=str(scenario["evaluation_window_ids_sha256"]),
        )
        if {record.window_id for record in train_records} & {
            record.window_id for record in evaluation_records
        }:
            raise FewPersonRunError("inclusion/evaluation window identities overlap")
        evaluation_batch = _materialize_authorized_records(
            raw_csv_path,
            evaluation_records,
            expected_source_sha256=source_sha256,
            class_names=class_names,
            ontology_track=ontology_track,
        )
        if set(evaluation_batch.participant_ids) != evaluation:
            raise FewPersonRunError("evaluation materialization differs from outer participants")
        logits, uncalibrated, _ = predict_model(
            model,
            normalizer.transform(evaluation_batch.signals),
            evaluation_batch.labels,
            list(evaluation_batch.participant_ids),
            class_names=class_names,
            batch_size=config.batch_size,
            device=device,
            mixed_precision=config.mixed_precision,
            zero_channel_indices=config.zero_channel_indices,
        )
        calibrated = calibrator.probabilities(logits)
        report = classification_report(
            evaluation_batch.labels,
            calibrated,
            list(evaluation_batch.participant_ids),
            class_names=class_names,
        )
        prediction_path = output / "predictions.npz"
        prediction_sha256 = _write_npz_new(
            prediction_path,
            {
                "window_ids": np.asarray(evaluation_batch.window_ids),
                "participant_ids": np.asarray(evaluation_batch.participant_ids),
                "true_labels": evaluation_batch.labels,
                "logits": logits,
                "uncalibrated_probabilities": uncalibrated,
                "source_temperature_probabilities": calibrated,
                "predicted_labels": calibrated.argmax(axis=1).astype(np.int64),
            },
        )
        torch.cuda.synchronize(device)
        result: dict[str, Any] = {
            "schema_version": "1.0.0",
            "record_kind": "few_person_outer_fold_result",
            "status": "complete_create_only_postconfirmatory_secondary",
            "evidence_status": FEW_PERSON_EVIDENCE_STATUS,
            "model_id": model_id,
            "seed": seed,
            "adaptation_seed": adaptation_seed,
            "fold_id": fold_id,
            "k": k,
            "few_person_manifest_sha256": plan_sha256,
            "split_manifest_sha256": split_sha256,
            "opening_receipt_record_sha256": plan["opening_receipt_record_sha256"],
            "zero_shot_index_record_sha256": plan["zero_shot_index_record_sha256"],
            "base_checkpoint_sha256": checkpoint_record["sha256"],
            "base_source_calibrator_sha256": calibrator_record["sha256"],
            "base_training_configuration_sha256": model_entry["training_configuration_sha256"],
            "source_hyperparameters_inherited": {
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "gradient_clip_norm": config.gradient_clip_norm,
                "mixed_precision": config.mixed_precision,
                "disable_cudnn": config.disable_cudnn,
            },
            "adaptation_objective": "supervised_cross_entropy_only",
            "checkpoint_selection_rule": "fixed_last_epoch_no_validation",
            "normalization_refit_on_target": False,
            "calibration_refit_on_target": False,
            "threshold_selection_performed": False,
            "target_validation_performed": False,
            "inclusion_subjects": sorted(inclusion, key=int),
            "evaluation_subjects": sorted(evaluation, key=int),
            "unused_target_subjects": scenario["unused_target_subjects"],
            "adapted_checkpoint": {
                "path": checkpoint_output.as_posix(),
                "sha256": checkpoint_sha256,
            },
            "prediction_artifact": {
                "path": prediction_path.as_posix(),
                "sha256": prediction_sha256,
            },
            "participant_level_report": report,
            "statistical_unit": "participant",
            "target_information_used_for_model_or_hyperparameter_selection": False,
            "device": {
                "type": "cuda",
                "name": torch.cuda.get_device_properties(device).name,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cudnn_enabled": torch.backends.cudnn.enabled,
            },
            "code_commit": commit,
            "elapsed_seconds": time.perf_counter() - started,
        }
        result["record_sha256"] = canonical_json_sha256(result)
        atomic_write_json_new(result, output / "result.json", allowed_root=output_root)
        return result
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": "1.0.0",
            "record_kind": "few_person_failed_run",
            "status": "failed_preserved_postconfirmatory_secondary",
            "evidence_status": "failed_run_not_result",
            "fold_id": fold_id,
            "k": k,
            "model_id": model_id,
            "seed": seed,
            "few_person_manifest_sha256": plan_sha256,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "elapsed_seconds": time.perf_counter() - started,
        }
        failure["record_sha256"] = canonical_json_sha256(failure)
        atomic_write_json_new(failure, output / "failure.json", allowed_root=output_root)
        raise FewPersonRunError(
            f"few-person scenario failed; evidence preserved at {output / 'failure.json'}"
        ) from exc
    finally:
        torch.backends.cudnn.enabled = original_cudnn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-manifest")
    build.add_argument("--split-manifest", type=Path, required=True)
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--opening-receipt", type=Path, required=True)
    build.add_argument("--zero-shot-index", type=Path, required=True)
    build.add_argument("--final-freeze-inventory", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--allowed-root", type=Path, required=True)
    run = subparsers.add_parser("run-scenario")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--split-manifest", type=Path, required=True)
    run.add_argument("--opening-receipt", type=Path, required=True)
    run.add_argument("--zero-shot-index", type=Path, required=True)
    run.add_argument("--final-freeze-inventory", type=Path, required=True)
    run.add_argument("--raw-csv", type=Path, required=True)
    run.add_argument("--artifact-root", type=Path, required=True)
    run.add_argument("--fold-id", required=True)
    run.add_argument("--k", type=int, choices=(1, 2, 4), required=True)
    run.add_argument("--model-id", required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--code-commit", required=True)
    run.add_argument("--output-directory", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build-manifest":
            manifest = build_few_person_manifest(
                split_manifest_path=args.split_manifest,
                config_path=args.config,
                opening_receipt_path=args.opening_receipt,
                zero_shot_index_path=args.zero_shot_index,
                final_freeze_inventory_path=args.final_freeze_inventory,
            )
            write_few_person_manifest_new(manifest, args.output, allowed_root=args.allowed_root)
            payload: Mapping[str, Any] = {
                "status": manifest["status"],
                "manifest_sha256": manifest["manifest_sha256"],
                "output": str(args.output),
                "target_metrics_or_predictions_used_for_design_or_selection": False,
                "target_raw_values_accessed_during_manifest_build": False,
            }
        else:
            result = run_few_person_scenario(
                manifest_path=args.manifest,
                split_manifest_path=args.split_manifest,
                opening_receipt_path=args.opening_receipt,
                zero_shot_index_path=args.zero_shot_index,
                final_freeze_inventory_path=args.final_freeze_inventory,
                raw_csv_path=args.raw_csv,
                artifact_root=args.artifact_root,
                fold_id=args.fold_id,
                k=args.k,
                model_id=args.model_id,
                seed=args.seed,
                code_commit=args.code_commit,
                output_directory=args.output_directory,
                output_root=args.output_root,
            )
            payload = {
                "status": result["status"],
                "record_sha256": result["record_sha256"],
                "fold_id": result["fold_id"],
                "k": result["k"],
                "model_id": result["model_id"],
                "seed": result["seed"],
            }
    except (
        FewPersonProtocolError,
        FewPersonRunError,
        FileExistsError,
        OSError,
        ValueError,
    ) as exc:
        print(json.dumps({"status": "fail", "message": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
