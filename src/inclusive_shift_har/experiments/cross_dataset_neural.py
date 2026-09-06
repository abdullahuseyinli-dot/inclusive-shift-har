"""Matched participant-exclusive DeepConvLSTM and TinyHAR external controls."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch as torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from inclusive_shift_har.artifacts.research_provenance import (
    _external_evidence_status,
    _publication_artifact_contract,
    _publication_launch_context_binding,
    _resolve_publication_launch_context,
    _runtime_environment,
    _source_manifest_commit_errors,
    _write_launch_failure_envelope,
    _write_self_hashed_json_create_only,
)
from inclusive_shift_har.data.external_har import (
    ExternalHARWindows,
    _array_sha256,
    load_fog_star,
    load_imu_har_il,
    observable_modelling_pool,
)
from inclusive_shift_har.evaluation.inference_contracts import OBSERVABLE_CONTEXT_PROTOCOL
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.experiments.cross_dataset_har import (
    _git_state,
    _paired_bootstrap,
    _read_mapping,
    _source_input_manifest,
    _write_json_create_only,
)
from inclusive_shift_har.experiments.external_evidence_validate import (
    validate_and_record_run_directory,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models import build_baseline, trainable_parameter_count
from inclusive_shift_har.models.common import HAROutput

FloatArray = NDArray[np.float64]
Float32Array = NDArray[np.float32]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class ExternalNeuralConfig:
    model_name: str
    seed: int
    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 8
    minimum_epochs: int = 8
    mixed_precision: str = "float16"
    disable_cudnn: bool = False

    def validate(self) -> None:
        if self.model_name not in {"deepconvlstm", "tinyhar"}:
            raise ValueError("external neural control is not predeclared")
        if self.seed < 0:
            raise ValueError("external neural epoch/seed contract is invalid")
        expected = {
            "epochs": 40,
            "batch_size": 128,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "patience": 8,
            "minimum_epochs": 8,
            "mixed_precision": "float16",
            "disable_cudnn": self.model_name == "deepconvlstm",
        }
        actual = {name: getattr(self, name) for name in expected}
        if actual != expected:
            raise ValueError("external neural configuration differs from the frozen CUDA protocol")


class _IndexedWindows(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(
        self,
        signals: Float32Array,
        labels: IntArray,
        indices: IntArray,
        weights: Float32Array,
        mean: Float32Array,
        scale: Float32Array,
    ) -> None:
        self.signals = signals
        self.labels = labels
        self.indices = indices
        self.weights = weights
        self.mean = mean
        self.scale = scale

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        selected = int(self.indices[index])
        values = (self.signals[selected] - self.mean) / self.scale
        return (
            torch.from_numpy(np.asarray(values, dtype=np.float32)),
            torch.tensor(int(self.labels[selected]), dtype=torch.long),
            torch.tensor(float(self.weights[index]), dtype=torch.float32),
        )


class _InferenceWindows(Dataset[Tensor]):
    """A label-free view used after a checkpoint has been selected."""

    def __init__(
        self,
        signals: Float32Array,
        indices: IntArray,
        mean: Float32Array,
        scale: Float32Array,
    ) -> None:
        self.signals = signals
        self.indices = indices
        self.mean = mean
        self.scale = scale

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> Tensor:
        selected = int(self.indices[index])
        values = (self.signals[selected] - self.mean) / self.scale
        return torch.from_numpy(np.asarray(values, dtype=np.float32))


def _configure_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _participant_class_weights(
    labels: IntArray, participants: StringArray, *, class_count: int
) -> Float32Array:
    weights = np.empty(labels.size, dtype=np.float64)
    for participant in np.unique(participants):
        participant_rows = participants == participant
        for class_index in range(class_count):
            rows = participant_rows & (labels == class_index)
            if rows.any():
                weights[rows] = 1.0 / float(rows.sum())
    weights *= labels.size / weights.sum()
    return np.asarray(weights, dtype=np.float32)


def _report_any(
    labels: IntArray,
    probability: FloatArray,
    participants: StringArray,
    *,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    report = classification_report(
        labels,
        probability,
        participants.tolist(),
        class_names=class_names,
    )
    values = sorted(float(item["macro_f1"]) for item in report["participants"])
    count = max(1, int(np.ceil(0.30 * len(values))))
    report["primary"]["bottom_30_percent_participant_macro_f1"] = float(np.mean(values[:count]))
    return report


def _normalization(signals: Float32Array, indices: IntArray) -> tuple[Float32Array, Float32Array]:
    training = np.asarray(signals[indices], dtype=np.float64)
    mean = training.mean(axis=(0, 1), keepdims=True)
    scale = training.std(axis=(0, 1), keepdims=True)
    scale = np.maximum(scale, 1e-6)
    return np.asarray(mean[0], dtype=np.float32), np.asarray(scale[0], dtype=np.float32)


def _loader(
    data: ExternalHARWindows,
    signals: Float32Array,
    indices: IntArray,
    *,
    mean: Float32Array,
    scale: Float32Array,
    batch_size: int,
    shuffle: bool,
    seed: int,
    class_count: int,
) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
    weights = _participant_class_weights(
        data.labels[indices], data.participant_ids[indices], class_count=class_count
    )
    dataset = _IndexedWindows(signals, data.labels, indices, weights, mean, scale)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _predict(
    model: nn.Module,
    signals: Float32Array,
    indices: IntArray,
    *,
    mean: Float32Array,
    scale: Float32Array,
    batch_size: int,
    device: torch.device,
    mixed_precision: str,
) -> FloatArray:
    loader = DataLoader(
        _InferenceWindows(signals, indices, mean, scale),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    enabled = device.type == "cuda" and mixed_precision != "disabled"
    dtype = torch.bfloat16 if mixed_precision == "bfloat16" else torch.float16
    batches: list[FloatArray] = []
    model.eval()
    with torch.inference_mode():
        for batch_signals in loader:
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
                output = cast(HAROutput, model(batch_signals.to(device, non_blocking=True)))
            batches.append(
                torch.softmax(output.logits.float(), dim=1)
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
    return np.concatenate(batches, axis=0)


def _train_fold(
    data: ExternalHARWindows,
    signals: Float32Array,
    *,
    training_indices: IntArray,
    validation_indices: IntArray,
    evaluation_indices: IntArray,
    config: ExternalNeuralConfig,
    output_path: Path,
    device: torch.device,
    class_names: tuple[str, ...],
) -> tuple[FloatArray, dict[str, Any]]:
    config.validate()
    with torch.backends.cudnn.flags(enabled=not config.disable_cudnn):
        return _train_fold_impl(
            data,
            signals,
            training_indices=training_indices,
            validation_indices=validation_indices,
            evaluation_indices=evaluation_indices,
            config=config,
            output_path=output_path,
            device=device,
            class_names=class_names,
        )


def _train_fold_impl(
    data: ExternalHARWindows,
    signals: Float32Array,
    *,
    training_indices: IntArray,
    validation_indices: IntArray,
    evaluation_indices: IntArray,
    config: ExternalNeuralConfig,
    output_path: Path,
    device: torch.device,
    class_names: tuple[str, ...],
) -> tuple[FloatArray, dict[str, Any]]:
    config.validate()
    _configure_determinism(config.seed)
    class_count = len(class_names)
    mean, scale = _normalization(signals, training_indices)
    loader = _loader(
        data,
        signals,
        training_indices,
        mean=mean,
        scale=scale,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed + 1,
        class_count=class_count,
    )
    model = build_baseline(
        config.model_name,
        num_classes=class_count,
        input_channels=signals.shape[2],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler(  # type: ignore[attr-defined]
        "cuda", enabled=device.type == "cuda" and config.mixed_precision == "float16"
    )
    amp_enabled = device.type == "cuda" and config.mixed_precision != "disabled"
    amp_dtype = torch.bfloat16 if config.mixed_precision == "bfloat16" else torch.float16
    best_metric = float("-inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        examples = 0
        for batch_signals, batch_labels, batch_weights in loader:
            optimizer.zero_grad(set_to_none=True)
            batch_signals = batch_signals.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_weights = batch_weights.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                output = cast(HAROutput, model(batch_signals))
                per_example = torch.nn.functional.cross_entropy(
                    output.logits, batch_labels, reduction="none"
                )
                loss = torch.sum(per_example * batch_weights) / torch.sum(batch_weights)
            scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            batch = batch_labels.numel()
            total_loss += float(loss.detach()) * batch
            examples += batch
        scheduler.step()
        validation_probability = _predict(
            model,
            signals,
            validation_indices,
            mean=mean,
            scale=scale,
            batch_size=config.batch_size,
            device=device,
            mixed_precision=config.mixed_precision,
        )
        report = classification_report(
            data.labels[validation_indices],
            validation_probability,
            data.participant_ids[validation_indices].tolist(),
            class_names=class_names,
        )
        metric = float(report["primary"]["mean_participant_macro_f1"])
        improved = metric > best_metric and not np.isclose(
            metric, best_metric, atol=1e-12, rtol=0.0
        )
        if improved:
            best_metric = metric
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        history.append(
            {
                "epoch": epoch,
                "training_weighted_loss": total_loss / examples,
                "validation_mean_participant_macro_f1": metric,
                "selected_after_epoch": improved,
            }
        )
        if epoch >= config.minimum_epochs and epoch - best_epoch >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("neural fold completed without a validation-selected state")
    model.load_state_dict(best_state)
    outer_probability = _predict(
        model,
        signals,
        evaluation_indices,
        mean=mean,
        scale=scale,
        batch_size=config.batch_size,
        device=device,
        mixed_precision=config.mixed_precision,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        torch.save(
            {
                "schema_version": "1.0.0",
                "evidence_status": "EXTERNAL_DEVELOPMENT_NOT_CONFIRMATORY",
                "model_name": config.model_name,
                "class_names": class_names,
                "input_channels": int(signals.shape[2]),
                "model_state": best_state,
                "config": asdict(config),
                "config_sha256": canonical_json_sha256(asdict(config)),
                "normalization": {"mean": mean.tolist(), "scale": scale.tolist()},
                "training_participants": sorted(
                    np.unique(data.participant_ids[training_indices]).tolist()
                ),
                "validation_participants": sorted(
                    np.unique(data.participant_ids[validation_indices]).tolist()
                ),
                "selected_epoch": best_epoch,
                "selected_validation_mean_participant_macro_f1": best_metric,
                "history": history,
                "parameter_count": trainable_parameter_count(model),
            },
            stream,
        )
    return outer_probability, {
        "training_config": asdict(config),
        "training_config_sha256": canonical_json_sha256(asdict(config)),
        "disable_cudnn": config.disable_cudnn,
        "mixed_precision": config.mixed_precision,
        "device": str(device),
        "device_type": device.type,
        "cuda_runtime_version": str(torch.version.cuda),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "amp_enabled": amp_enabled,
        "autocast_device_type": device.type,
        "autocast_dtype": str(amp_dtype).removeprefix("torch."),
        "gradient_scaler_enabled": bool(scaler.is_enabled()),
        "selected_epoch": best_epoch,
        "selected_validation_mean_participant_macro_f1": best_metric,
        "epochs_completed": len(history),
        "checkpoint": {
            "path": output_path.name,
            "sha256": sha256_file(output_path),
            "size_bytes": output_path.stat().st_size,
        },
        "parameter_count": trainable_parameter_count(model),
    }


def evaluate_neural_controls(
    data: ExternalHARWindows,
    *,
    seeds: tuple[int, ...],
    output_directory: Path,
    epochs: int,
    signals: Float32Array | None = None,
    class_names: tuple[str, ...] | None = None,
    method_suffix: str = "6ch",
    experiment_id: str = "cross-dataset-har-neural-controls-v1",
) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    data.validate()
    if epochs != 40:
        raise ValueError("external publication neural evidence requires exactly 40 epochs")
    if not torch.cuda.is_available():
        raise RuntimeError("external neural CUDA/FP16 protocol requires an available CUDA device")
    if data.participant_partition_plan is None:
        raise PermissionError("external neural evaluation requires a pre-window participant plan")
    partition_plan = data.participant_partition_plan
    partition_plan.validate()
    scored_data = data
    data, scoring_indices, supervised_eligibility = observable_modelling_pool(
        data, include_supervised_labels=True
    )
    scored_six = scored_data.signals
    scored_nine = scored_data.nine_channel_signals
    supplied = scored_six if signals is None else np.asarray(signals)
    supplied_hash = _array_sha256(supplied)
    if supplied_hash == _array_sha256(scored_six):
        selected_signals = data.signals
        representation_field = "signals"
        expected_suffix = "6ch"
    elif supplied_hash == _array_sha256(scored_nine):
        selected_signals = data.nine_channel_signals
        representation_field = "nine_channel_signals"
        expected_suffix = "N9"
    else:
        raise ValueError("neural signals are not a bit-exact canonical dataset representation")
    expected_experiment = (
        "har-pmd-native-interface-neural-controls-v1"
        if scored_data.dataset_id == "har_pmd_v1"
        else "cross-dataset-har-neural-controls-v1"
    )
    if method_suffix != expected_suffix or experiment_id != expected_experiment:
        raise ValueError(
            "neural lane suffix or experiment identity differs from its representation"
        )
    if representation_field == "nine_channel_signals" and scored_data.dataset_id != "har_pmd_v1":
        raise ValueError("nine-channel neural lane is predeclared only for HAR-PMD")
    representation = {
        "source_field": representation_field,
        "method_suffix": expected_suffix,
        "input_channels": int(selected_signals.shape[2]),
        "array_sha256": _array_sha256(selected_signals),
        "scored_source_array_sha256": supplied_hash,
        "gravity_source": (
            scored_data.gravity_source if representation_field == "nine_channel_signals" else None
        ),
    }
    selected_class_names = data.class_names if class_names is None else class_names
    if selected_class_names != data.class_names:
        raise ValueError("neural class names differ from the dataset ontology")
    if (
        selected_signals.ndim != 3
        or selected_signals.shape[:2] != data.signals.shape[:2]
        or not np.isfinite(selected_signals).all()
    ):
        raise ValueError("neural signal representation is not aligned and finite")
    expected_classes = set(range(len(selected_class_names)))
    if set(scored_data.labels.tolist()) != expected_classes:
        raise ValueError("neural dataset lacks a declared class")
    if len(partition_plan.participant_roster) < 12:
        raise ValueError("neural external controls require at least 12 participants")
    device = torch.device("cuda")
    cuda_runtime_version = torch.version.cuda
    if not isinstance(cuda_runtime_version, str) or not cuda_runtime_version:
        raise RuntimeError("PyTorch does not report a CUDA runtime for the frozen neural protocol")
    cuda_device_name = torch.cuda.get_device_name(device)
    if not cuda_device_name:
        raise RuntimeError("CUDA device identity is unavailable for the frozen neural protocol")
    per_seed: dict[int, dict[str, FloatArray]] = {}
    fold_records: list[dict[str, Any]] = []
    for seed in seeds:
        assignment = partition_plan.resolve(
            partition_plan.participant_roster,
            fold_count=5,
            seed=seed,
            role="outer",
        )
        deep_name = f"DeepConvLSTM-{method_suffix}"
        tiny_name = f"TinyHAR-{method_suffix}"
        seed_probability = {
            deep_name: np.full(
                (data.labels.size, len(selected_class_names)), np.nan, dtype=np.float64
            ),
            tiny_name: np.full(
                (data.labels.size, len(selected_class_names)), np.nan, dtype=np.float64
            ),
        }
        for outer_fold in range(5):
            evaluation_ids = {
                participant for participant, fold in assignment.items() if fold == outer_fold
            }
            outer_training_ids = sorted(set(assignment) - evaluation_ids)
            validation_assignment = partition_plan.resolve(
                outer_training_ids,
                fold_count=4,
                seed=seed + 20_000 + outer_fold,
                role=f"neural_validation_outer_{outer_fold}",
            )
            validation_ids = {
                participant for participant, fold in validation_assignment.items() if fold == 0
            }
            training_ids = set(outer_training_ids) - validation_ids
            training_indices = np.flatnonzero(
                np.isin(data.participant_ids, sorted(training_ids)) & supervised_eligibility
            ).astype(np.int64)
            validation_indices = np.flatnonzero(
                np.isin(data.participant_ids, sorted(validation_ids)) & supervised_eligibility
            ).astype(np.int64)
            evaluation_indices = np.flatnonzero(
                np.isin(data.participant_ids, sorted(evaluation_ids))
            ).astype(np.int64)
            if any(
                set(np.unique(data.labels[indices]).tolist()) != expected_classes
                for indices in (training_indices, validation_indices)
            ):
                raise ValueError("neural training or validation partition lacks a core class")
            for model_name, result_name in (
                ("deepconvlstm", deep_name),
                ("tinyhar", tiny_name),
            ):
                config = ExternalNeuralConfig(
                    model_name=model_name,
                    seed=seed + 101 * outer_fold,
                    epochs=epochs,
                    disable_cudnn=model_name == "deepconvlstm",
                )
                checkpoint_path = (
                    output_directory
                    / "checkpoints"
                    / f"seed-{seed}__fold-{outer_fold}__{model_name}.pt"
                )
                values, record = _train_fold(
                    data,
                    selected_signals,
                    training_indices=training_indices,
                    validation_indices=validation_indices,
                    evaluation_indices=evaluation_indices,
                    config=config,
                    output_path=checkpoint_path,
                    device=device,
                    class_names=selected_class_names,
                )
                checkpoint = cast(dict[str, Any], record.get("checkpoint"))
                checkpoint["path"] = checkpoint_path.relative_to(output_directory).as_posix()
                seed_probability[result_name][evaluation_indices] = values
                fold_records.append(
                    {
                        "seed": seed,
                        "outer_fold": outer_fold,
                        "model": result_name,
                        "training_participants": sorted(training_ids),
                        "training_participants_with_supervision": sorted(
                            np.unique(data.participant_ids[training_indices]).tolist()
                        ),
                        "validation_participants": sorted(validation_ids),
                        "validation_participants_with_supervision": sorted(
                            np.unique(data.participant_ids[validation_indices]).tolist()
                        ),
                        "evaluation_participants": sorted(evaluation_ids),
                        "evaluation_participants_with_candidates": sorted(
                            np.unique(data.participant_ids[evaluation_indices]).tolist()
                        ),
                        "evaluation_participants_with_scoring": sorted(
                            set(scored_data.participant_ids.tolist()) & evaluation_ids
                        ),
                        "evaluation_candidate_window_count": int(evaluation_indices.size),
                        "evaluation_scored_window_count": int(
                            supervised_eligibility[evaluation_indices].sum()
                        ),
                        "participant_partition_plan_sha256": partition_plan.audit()["plan_sha256"],
                        "outer_labels_used_for_training_or_selection": False,
                        **record,
                    }
                )
        if any(not np.isfinite(value).all() for value in seed_probability.values()):
            raise ValueError("neural OOF predictions are incomplete")
        per_seed[seed] = {
            method: probability[scoring_indices] for method, probability in seed_probability.items()
        }
    ensemble = {
        method: np.mean(np.stack([per_seed[seed][method] for seed in seeds], axis=0), axis=0)
        for method in per_seed[seeds[0]]
    }
    reports = {
        method: _report_any(
            scored_data.labels,
            probability,
            scored_data.participant_ids,
            class_names=selected_class_names,
        )
        for method, probability in ensemble.items()
    }
    from inclusive_shift_har.evaluation.external_statistics import seed_evidence

    primary_seed_averaged, seed_predictions = seed_evidence(scored_data, per_seed)
    return (
        {
            "schema_version": "1.0.0",
            "experiment_id": experiment_id,
            "observable_context_protocol": (
                OBSERVABLE_CONTEXT_PROTOCOL
                if scored_data.observable_candidates is not None
                else None
            ),
            "evidence_status": "EXTERNAL_DEVELOPMENT_NOT_CONFIRMATORY",
            "dataset": scored_data.summary(),
            "seeds": list(seeds),
            "device": str(device),
            "runtime_backend_protocol": "external-neural-cuda-nocudnn-v2",
            "runtime_backend": {
                "device_type": "cuda",
                "amp_enabled": True,
                "autocast_device_type": "cuda",
                "autocast_dtype": "float16",
                "gradient_scaler_enabled": True,
                "cuda_runtime_version": cuda_runtime_version,
                "cuda_device_name": cuda_device_name,
            },
            "training_contract": {
                "epochs": 40,
                "batch_size": 128,
                "learning_rate": 3e-4,
                "weight_decay": 1e-4,
                "patience": 8,
                "minimum_epochs": 8,
                "mixed_precision": "float16",
                "outer_folds": 5,
                "models": ["deepconvlstm", "tinyhar"],
            },
            "input_representation": representation,
            "input_channels": int(selected_signals.shape[2]),
            "method_suffix": method_suffix,
            "reports": reports,
            "primary_seed_averaged": primary_seed_averaged,
            "fold_records": fold_records,
            "claim_policy": {
                "proposed_method": False,
                "confirmatory_claim_allowed": False,
                "state_of_the_art_claim_allowed": False,
                "full_observable_candidate_inference_before_scoring": True,
            },
        },
        {**ensemble, **seed_predictions},
    )


def run_and_write_neural(
    *,
    data: ExternalHARWindows,
    output_directory: Path,
    repository_root: Path,
    seeds: tuple[int, ...],
    epochs: int,
    signals: Float32Array | None = None,
    class_names: tuple[str, ...] | None = None,
    method_suffix: str = "6ch",
    experiment_id: str = "cross-dataset-har-neural-controls-v1",
    inherited_launch_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if seeds != (11, 23, 47):
        raise ValueError("external publication evidence requires frozen seeds 11, 23, and 47")
    git_at_launch, source_input_manifest, launch_context = _resolve_publication_launch_context(
        repository_root=repository_root,
        output_directory=output_directory,
        current_git_state=_git_state(repository_root),
        current_source_manifest=_source_input_manifest(repository_root),
        manifest_commit_validator=_source_manifest_commit_errors,
        inherited_launch_context=inherited_launch_context,
    )
    launch_context_binding = _publication_launch_context_binding(launch_context)
    output_directory.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC).isoformat()
    dataset_summary = data.summary()
    evidence_status = _external_evidence_status(dataset_summary)
    audit = {
        "schema_version": "1.0.0",
        "created_at": started,
        "dataset": dataset_summary,
        "artifact_evidence_status": evidence_status,
        "source_receipts": [receipt.to_dict() for receipt in data.receipts],
        "raw_local_mirror": any(receipt.raw_local_mirror for receipt in data.receipts),
        "source_input_manifest": source_input_manifest,
        "git_at_launch": git_at_launch,
        "publication_launch_context": launch_context_binding,
        "model_input": {
            "channel_count": int(data.signals.shape[2] if signals is None else signals.shape[2]),
            "method_suffix": method_suffix,
            "class_names": list(data.class_names if class_names is None else class_names),
        },
    }
    data_audit_artifact = _write_self_hashed_json_create_only(
        output_directory / "data_audit.json", audit
    )
    artifact_contract = _publication_artifact_contract(source_input_manifest, data_audit_artifact)
    try:
        result, predictions = evaluate_neural_controls(
            data,
            seeds=seeds,
            output_directory=output_directory,
            epochs=epochs,
            signals=signals,
            class_names=class_names,
            method_suffix=method_suffix,
            experiment_id=experiment_id,
        )
        path = output_directory / "predictions.npz"
        np.savez_compressed(
            path,
            labels=data.labels,
            participant_ids=data.participant_ids,
            session_ids=data.session_ids,
            trial_ids=data.trial_ids,
            window_ids=data.window_ids,
            **{  # type: ignore[arg-type]
                f"probability__{name}": value for name, value in predictions.items()
            },
        )
        _resolve_publication_launch_context(
            repository_root=repository_root,
            output_directory=output_directory,
            current_git_state=_git_state(repository_root),
            current_source_manifest=_source_input_manifest(repository_root),
            manifest_commit_validator=_source_manifest_commit_errors,
            inherited_launch_context=launch_context,
        )
        result["started_at"] = started
        result["created_at"] = datetime.now(UTC).isoformat()
        result["git"] = _git_state(repository_root)
        result["git_at_launch"] = git_at_launch
        result["source_input_manifest"] = source_input_manifest
        result["publication_launch_context"] = launch_context_binding
        result["environment"] = _runtime_environment()
        result["artifact_evidence_status"] = evidence_status
        result["data_audit_artifact"] = data_audit_artifact
        result["artifact_contract"] = artifact_contract
        result["prediction_artifact"] = {"path": path.name, "sha256": sha256_file(path)}
        result["paired_model_comparison"] = _paired_bootstrap(
            result["reports"][f"TinyHAR-{method_suffix}"],
            result["reports"][f"DeepConvLSTM-{method_suffix}"],
        )
        result["result_payload_sha256_before_serialization"] = canonical_json_sha256(result)
        _write_json_create_only(output_directory / "result.json", result)
    except Exception as exc:
        _write_self_hashed_json_create_only(
            output_directory / "failure.json",
            {
                "schema_version": "1.0.0",
                "status": "FAILED_PRESERVED",
                "started_at": started,
                "failed_at": datetime.now(UTC).isoformat(),
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
                "git": _git_state(repository_root),
                "git_at_launch": git_at_launch,
                "source_input_manifest": source_input_manifest,
                "publication_launch_context": launch_context_binding,
                "environment": _runtime_environment(),
                "artifact_evidence_status": evidence_status,
                "data_audit_artifact": data_audit_artifact,
                "artifact_contract": artifact_contract,
            },
            hash_field="failure_payload_sha256_before_serialization",
        )
        validate_and_record_run_directory(output_directory, repository_root)
        raise
    validate_and_record_run_directory(output_directory, repository_root)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("fog-star", "imu-har-il"), required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--participant-limit", type=int)
    parser.add_argument("--repetition-limit", type=int)
    parser.add_argument(
        "--gravity-cutoff-hz",
        type=float,
        help="Declared derived-gravity sensitivity value; defaults to the primary cutoff.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repository_root.resolve()
    output_directory = args.output_directory.resolve()
    _launch, _manifest, launch_context = _resolve_publication_launch_context(
        repository_root=root,
        output_directory=output_directory,
        current_git_state=_git_state(root),
        current_source_manifest=_source_input_manifest(root),
        manifest_commit_validator=_source_manifest_commit_errors,
    )
    started = datetime.now(UTC).isoformat()
    stage = "configuration"
    try:
        if tuple(args.seeds) != (11, 23, 47):
            raise ValueError("external publication evidence requires frozen seeds 11, 23, and 47")
        if args.epochs != 40:
            raise ValueError("external publication neural evidence requires exactly 40 epochs")
        experiment = _read_mapping(root / "configs/experiments/cross_dataset_har_rnd_v1.yaml")
        preprocessing = cast(dict[str, Any], experiment["preprocessing"])
        derived = cast(dict[str, Any], preprocessing["derived_gravity"])
        declared_cutoffs = tuple(float(value) for value in derived["cutoff_sensitivity_hz"])
        gravity_cutoff_hz = (
            float(derived["cutoff_hz"])
            if args.gravity_cutoff_hz is None
            else float(args.gravity_cutoff_hz)
        )
        if not any(np.isclose(gravity_cutoff_hz, value) for value in declared_cutoffs):
            raise ValueError(
                f"gravity cutoff {gravity_cutoff_hz} is outside the declared sensitivity set "
                f"{declared_cutoffs}"
            )
        common = {
            "participant_limit": args.participant_limit,
            "target_rate_hz": float(preprocessing["target_sampling_rate_hz"]),
            "window_samples": int(preprocessing["window_samples"]),
            "gravity_cutoff_hz": gravity_cutoff_hz,
        }
        stage = "dataset_acquisition"
        data = (
            load_fog_star(**common)
            if args.dataset == "fog-star"
            else load_imu_har_il(**common, repetition_limit=args.repetition_limit)
        )
        stage = "experiment_writer"
        result = run_and_write_neural(
            data=data,
            output_directory=output_directory,
            repository_root=root,
            seeds=tuple(args.seeds),
            epochs=args.epochs,
            inherited_launch_context=launch_context,
        )
    except Exception as error:
        if not output_directory.exists():
            _write_launch_failure_envelope(
                repository_root=root,
                output_directory=output_directory,
                launch_context=launch_context,
                started_at=started,
                stage=stage,
                exception=error,
                traceback_text=traceback.format_exc(),
            )
        raise
    print(
        json.dumps(
            {
                name: report["mean_participant_macro_f1"]
                for name, report in result["primary_seed_averaged"]["methods"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
