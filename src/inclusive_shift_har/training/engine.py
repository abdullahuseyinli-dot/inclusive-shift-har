"""Deterministic source-development engine with participant-aware model selection."""

from __future__ import annotations

import copy
import os
import platform
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models import (
    MoReHAR,
    build_baseline,
    build_exact_legacy_model,
    trainable_parameter_count,
)
from inclusive_shift_har.models.common import HAROutput
from inclusive_shift_har.training.augmentation import (
    physically_plausible_augmentation,
    signal_descriptors,
)
from inclusive_shift_har.training.objectives import (
    GroupDROState,
    MoReObjectiveWeights,
    coral_loss,
    more_har_objective,
)


@dataclass(frozen=True)
class TrainingConfig:
    """Fully serializable neural-training configuration."""

    model_name: str
    num_classes: int
    seed: int
    epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 12
    minimum_epochs: int = 8
    gradient_clip_norm: float = 5.0
    mixed_precision: str = "float16"
    data_loader_workers: int = 0
    checkpoint_interval: int = 10
    use_augmentation: bool = False
    use_content_objective: bool = False
    use_realization_factorization: bool = False
    use_group_dro: bool = False
    coral_weight: float = 0.0
    objective_weights: MoReObjectiveWeights = field(default_factory=MoReObjectiveWeights)
    zero_channel_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.num_classes < 2 or self.seed < 0:
            raise ValueError("training requires at least two classes and a non-negative seed")
        if min(self.epochs, self.batch_size, self.patience, self.minimum_epochs) < 1:
            raise ValueError("epoch, batch, and early-stopping values must be positive")
        if self.minimum_epochs > self.epochs:
            raise ValueError("minimum_epochs cannot exceed epochs")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("optimizer and gradient-clip settings are invalid")
        if self.mixed_precision not in {"disabled", "float16", "bfloat16"}:
            raise ValueError("mixed_precision must be disabled, float16, or bfloat16")
        if self.data_loader_workers != 0:
            raise ValueError("the locked initial protocol requires zero DataLoader workers")
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be positive")
        if self.coral_weight < 0:
            raise ValueError("CORAL weight cannot be negative")
        if any(index < 0 or index >= 6 for index in self.zero_channel_indices):
            raise ValueError("input ablation channel indices must lie in [0, 5]")
        if (
            self.use_augmentation
            or self.use_content_objective
            or self.use_realization_factorization
        ) and self.model_name != "more_har":
            raise ValueError("factorization ablations require model_name='more_har'")


@dataclass(frozen=True)
class TrainingLineage:
    """Immutable hashes and schemas embedded into every checkpoint."""

    dataset_manifest_sha256: str
    split_manifest_sha256: str
    preprocessing_config_sha256: str
    ontology_sha256: str
    code_commit: str
    evidence_status: str
    label_schema: tuple[str, ...]
    normalization: dict[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "dataset_manifest_sha256",
            "split_manifest_sha256",
            "preprocessing_config_sha256",
            "ontology_sha256",
        ):
            if len(str(getattr(self, name))) != 64:
                raise ValueError(f"{name} must be a full SHA-256")
        if len(self.label_schema) < 2 or len(set(self.label_schema)) != len(self.label_schema):
            raise ValueError("checkpoint label schema must contain unique classes")


class WindowTensorDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Aligned in-memory windows, labels, and participant-domain indices."""

    def __init__(
        self,
        windows: NDArray[np.float32],
        labels: NDArray[np.int64],
        domain_indices: NDArray[np.int64],
    ) -> None:
        if windows.ndim != 3 or labels.ndim != 1 or domain_indices.ndim != 1:
            raise ValueError("dataset tensors have invalid ranks")
        if windows.shape[0] != labels.size or labels.size != domain_indices.size:
            raise ValueError("dataset tensors are not aligned")
        if windows.shape[1:] != (128, 6):
            raise ValueError("training engine requires [window,128,6]")
        if not np.isfinite(windows).all():
            raise ValueError("training windows must be finite")
        self.windows = torch.from_numpy(np.ascontiguousarray(windows))
        self.labels = torch.from_numpy(np.ascontiguousarray(labels))
        self.domains = torch.from_numpy(np.ascontiguousarray(domain_indices))

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.windows[index], self.labels[index], self.domains[index]


def configure_determinism(seed: int) -> None:
    """Set process RNGs and deterministic PyTorch behavior before constructing models."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def build_model(config: TrainingConfig) -> nn.Module:
    if config.model_name == "more_har":
        return MoReHAR(num_classes=config.num_classes)
    if config.model_name.startswith("legacy_"):
        return build_exact_legacy_model(config.model_name, num_classes=config.num_classes)
    return build_baseline(config.model_name, num_classes=config.num_classes)


def training_config_from_dict(payload: dict[str, Any]) -> TrainingConfig:
    """Reconstruct nested immutable configuration fields from a checkpoint payload."""

    values = dict(payload)
    objective = values.get("objective_weights")
    if isinstance(objective, dict):
        values["objective_weights"] = MoReObjectiveWeights(**objective)
    indices = values.get("zero_channel_indices")
    if isinstance(indices, list):
        values["zero_channel_indices"] = tuple(int(index) for index in indices)
    return TrainingConfig(**values)


def _device_environment(device: torch.device) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "device_type": device.type,
        "hostname_recorded": False,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        payload.update(
            {
                "device_name": properties.name,
                "device_total_memory_bytes": properties.total_memory,
                "compute_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    return payload


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(payload: dict[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    if torch.cuda.is_available() and payload["torch_cuda"]:
        torch.cuda.set_rng_state_all(payload["torch_cuda"])


def _cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _write_checkpoint_create_only(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(payload, stream)
    return sha256_file(path)


def _prepare_arrays(
    windows: NDArray[np.float32],
    labels: NDArray[np.int64],
    participant_ids: list[str],
    *,
    participant_domain_map: dict[str, int],
) -> WindowTensorDataset:
    if len(participant_ids) != labels.size:
        raise ValueError("participant ids are not aligned with training labels")
    try:
        domains = np.asarray(
            [participant_domain_map[participant] for participant in participant_ids],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError(f"participant lacks a declared source-domain index: {exc}") from exc
    return WindowTensorDataset(windows, labels, domains)


def _apply_input_ablation(signals: Tensor, indices: tuple[int, ...]) -> Tensor:
    if not indices:
        return signals
    transformed = signals.clone()
    transformed[:, :, list(indices)] = 0.0
    return transformed


def _autocast_settings(config: TrainingConfig, device: torch.device) -> tuple[bool, torch.dtype]:
    enabled = device.type == "cuda" and config.mixed_precision != "disabled"
    dtype = torch.bfloat16 if config.mixed_precision == "bfloat16" else torch.float16
    return enabled, dtype


def _train_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    config: TrainingConfig,
    *,
    device: torch.device,
    augmentation_generator: torch.Generator,
    group_dro: GroupDROState | None,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    examples = 0
    amp_enabled, amp_dtype = _autocast_settings(config, device)
    for signals, labels, domains in loader:
        signals = _apply_input_ablation(signals.to(device), config.zero_channel_indices)
        labels = labels.to(device)
        domains = domains.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            clean = cast(HAROutput, model(signals))
            component_losses: dict[str, Tensor]
            if config.model_name == "more_har" and config.use_augmentation:
                augmented_batch = physically_plausible_augmentation(
                    signals,
                    generator=augmentation_generator,
                )
                augmented = cast(HAROutput, model(augmented_batch.signals))
                weights = config.objective_weights
                if not config.use_content_objective:
                    weights = MoReObjectiveWeights(
                        content=0.0,
                        consistency=0.0,
                        realization=weights.realization
                        if config.use_realization_factorization
                        else 0.0,
                        factor=weights.factor if config.use_realization_factorization else 0.0,
                    )
                elif not config.use_realization_factorization:
                    weights = MoReObjectiveWeights(
                        content=weights.content,
                        consistency=weights.consistency,
                        realization=0.0,
                        factor=0.0,
                    )
                component_losses = more_har_objective(
                    clean,
                    augmented,
                    labels,
                    signal_descriptors(augmented_batch.signals),
                    weights=weights,
                )
                total_loss = component_losses["total"]
            else:
                per_example = F.cross_entropy(clean.logits, labels, reduction="none")
                total_loss = per_example.mean()
                component_losses = {"classification": total_loss}

            if config.coral_weight:
                if clean.content is None:
                    raise ValueError("CORAL requires model content features")
                coral = coral_loss(clean.content, domains)
                total_loss = total_loss + config.coral_weight * coral
                component_losses["coral"] = coral
            if group_dro is not None:
                per_example = F.cross_entropy(clean.logits, labels, reduction="none")
                robust = group_dro.aggregate(per_example, domains)
                clean_mean = per_example.mean()
                clean_classification_weight = (
                    0.5 if config.model_name == "more_har" and config.use_augmentation else 1.0
                )
                total_loss = total_loss + clean_classification_weight * (robust - clean_mean)
                component_losses["group_dro"] = robust

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        batch_size = labels.numel()
        examples += batch_size
        totals["total"] = totals.get("total", 0.0) + float(total_loss.detach()) * batch_size
        for name, value in component_losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
    return {name: value / examples for name, value in totals.items()}


def predict_model(
    model: nn.Module,
    windows: NDArray[np.float32],
    labels: NDArray[np.int64],
    participant_ids: list[str],
    *,
    class_names: tuple[str, ...],
    batch_size: int,
    device: torch.device,
    mixed_precision: str,
    zero_channel_indices: tuple[int, ...] = (),
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
    """Predict in stable input order and compute participant-aware metrics."""

    domains = np.zeros(labels.size, dtype=np.int64)
    dataset = WindowTensorDataset(windows, labels, domains)
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    logit_batches: list[NDArray[np.float64]] = []
    probability_batches: list[NDArray[np.float64]] = []
    amp_enabled = device.type == "cuda" and mixed_precision != "disabled"
    amp_dtype = torch.bfloat16 if mixed_precision == "bfloat16" else torch.float16
    with torch.inference_mode():
        for signals, _, _ in loader:
            signals = _apply_input_ablation(signals.to(device), zero_channel_indices)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                output = cast(HAROutput, model(signals))
            logits = output.logits.float()
            logit_batches.append(logits.cpu().numpy().astype(np.float64, copy=False))
            probability_batches.append(
                torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64, copy=False)
            )
    logits_array = np.concatenate(logit_batches, axis=0)
    probability_array = np.concatenate(probability_batches, axis=0)
    if logits_array.shape[0] != labels.size:
        raise AssertionError("prediction order/count no longer aligns with labels")
    report = classification_report(
        labels,
        probability_array,
        participant_ids,
        class_names=class_names,
    )
    return logits_array, probability_array, report


def train_source_model(
    train_windows: NDArray[np.float32],
    train_labels: NDArray[np.int64],
    train_participant_ids: list[str],
    validation_windows: NDArray[np.float32],
    validation_labels: NDArray[np.int64],
    validation_participant_ids: list[str],
    *,
    config: TrainingConfig,
    lineage: TrainingLineage,
    output_directory: Path,
    device: torch.device,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Train/tune on source participants only and preserve immutable resumable checkpoints."""

    if config.num_classes != len(lineage.label_schema):
        raise ValueError("training class count disagrees with locked checkpoint label schema")
    if set(train_participant_ids) & set(validation_participant_ids):
        raise ValueError("source training and validation participants overlap")
    configure_determinism(config.seed)
    participant_domain_map = {
        participant: index for index, participant in enumerate(sorted(set(train_participant_ids)))
    }
    train_dataset = _prepare_arrays(
        train_windows,
        train_labels,
        train_participant_ids,
        participant_domain_map=participant_domain_map,
    )
    loader_generator = torch.Generator().manual_seed(config.seed + 1)
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]] = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.data_loader_workers,
        generator=loader_generator,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler(  # type: ignore[attr-defined]
        "cuda",
        enabled=device.type == "cuda" and config.mixed_precision == "float16",
    )
    augmentation_generator = torch.Generator(device=device).manual_seed(config.seed + 2)
    group_dro = (
        GroupDROState.initialize(
            len(participant_domain_map),
            device=device,
        )
        if config.use_group_dro
        else None
    )
    configuration_sha256 = canonical_json_sha256(asdict(config))
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_epoch = 0
    best_metric = float("-inf")
    best_worst_participant = float("-inf")
    best_state: dict[str, Tensor] | None = None

    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        if checkpoint["configuration_sha256"] != configuration_sha256:
            raise ValueError("resume checkpoint configuration hash mismatch")
        if checkpoint["lineage"]["split_manifest_sha256"] != lineage.split_manifest_sha256:
            raise ValueError("resume checkpoint split hash mismatch")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        _restore_rng_state(checkpoint["rng_states"])
        loader_generator.set_state(checkpoint["rng_states"]["loader_generator"])
        augmentation_generator.set_state(checkpoint["rng_states"]["augmentation_generator"])
        history = list(checkpoint["history"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_metric = float(checkpoint["best_metric"])
        best_worst_participant = float(checkpoint["best_worst_participant"])
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in checkpoint["best_model_state"].items()
        }
        if group_dro is not None and checkpoint["group_dro_weights"] is not None:
            group_dro.weights.copy_(checkpoint["group_dro_weights"].to(device))

    stopped_early = False
    last_checkpoint: dict[str, Any] | None = None
    for epoch in range(start_epoch, config.epochs + 1):
        train_losses = _train_epoch(
            model,
            loader,
            optimizer,
            scaler,
            config,
            device=device,
            augmentation_generator=augmentation_generator,
            group_dro=group_dro,
        )
        scheduler.step()
        _, _, validation_report = predict_model(
            model,
            validation_windows,
            validation_labels,
            validation_participant_ids,
            class_names=lineage.label_schema,
            batch_size=config.batch_size,
            device=device,
            mixed_precision=config.mixed_precision,
            zero_channel_indices=config.zero_channel_indices,
        )
        metric = float(validation_report["primary"]["mean_participant_macro_f1"])
        worst = float(validation_report["primary"]["worst_participant_macro_f1"])
        improved = metric > best_metric and not np.isclose(
            metric, best_metric, atol=1e-12, rtol=0.0
        )
        if improved:
            best_epoch = epoch
            best_metric = metric
            best_worst_participant = worst
            best_state = _cpu_state_dict(model)
        history.append(
            {
                "epoch": epoch,
                "training_losses": train_losses,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "validation_mean_participant_macro_f1": metric,
                "validation_worst_participant_macro_f1": worst,
                "best_epoch_after_epoch": best_epoch,
            }
        )
        checkpoint_payload = {
            "schema_version": "1.0.0",
            "evidence_status": lineage.evidence_status,
            "model_name": config.model_name,
            "model_state": model.state_dict(),
            "best_model_state": best_state,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "normalization": lineage.normalization,
            "label_schema": list(lineage.label_schema),
            "lineage": asdict(lineage),
            "configuration": asdict(config),
            "configuration_sha256": configuration_sha256,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "best_worst_participant": best_worst_participant,
            "history": history,
            "rng_states": {
                **_rng_state(),
                "loader_generator": loader_generator.get_state(),
                "augmentation_generator": augmentation_generator.get_state(),
            },
            "environment": _device_environment(device),
            "parameter_count": trainable_parameter_count(model),
            "group_dro_weights": None if group_dro is None else group_dro.weights.detach().cpu(),
        }
        last_checkpoint = checkpoint_payload
        if epoch % config.checkpoint_interval == 0:
            _write_checkpoint_create_only(
                output_directory / f"epoch_{epoch:03d}.pt",
                checkpoint_payload,
            )
        if epoch >= config.minimum_epochs and epoch - best_epoch >= config.patience:
            stopped_early = True
            break

    if best_state is None or last_checkpoint is None:
        raise RuntimeError("training completed without a valid checkpoint")
    model.load_state_dict(best_state)
    validation_logits, validation_probabilities, validation_report = predict_model(
        model,
        validation_windows,
        validation_labels,
        validation_participant_ids,
        class_names=lineage.label_schema,
        batch_size=config.batch_size,
        device=device,
        mixed_precision=config.mixed_precision,
        zero_channel_indices=config.zero_channel_indices,
    )
    final_payload = copy.copy(last_checkpoint)
    final_payload["model_state"] = best_state
    final_payload["checkpoint_role"] = "source_validation_selected"
    final_payload["stopped_early"] = stopped_early
    final_checkpoint_path = output_directory / "selected.pt"
    final_checkpoint_sha256 = _write_checkpoint_create_only(final_checkpoint_path, final_payload)
    return {
        "schema_version": "1.0.0",
        "status": "source_development_complete",
        "evidence_status": lineage.evidence_status,
        "configuration_sha256": configuration_sha256,
        "best_epoch": best_epoch,
        "best_validation_mean_participant_macro_f1": best_metric,
        "best_validation_worst_participant_macro_f1": best_worst_participant,
        "stopped_early": stopped_early,
        "epochs_completed": len(history),
        "history": history,
        "validation_report": validation_report,
        "validation_logits": validation_logits,
        "validation_probabilities": validation_probabilities,
        "checkpoint_path": str(final_checkpoint_path),
        "checkpoint_sha256": final_checkpoint_sha256,
        "parameter_count": trainable_parameter_count(model),
    }


def reconstruct_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild a trusted local checkpoint and validate its public model/label contract."""

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = training_config_from_dict(payload["configuration"])
    model = build_model(config).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    if config.num_classes != len(payload["label_schema"]):
        raise ValueError("checkpoint model and label schema disagree")
    return model, payload
