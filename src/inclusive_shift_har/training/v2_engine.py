"""Target-sealed training engine for FuSE-ReFrame source and external development."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.fuse_reframe import (
    BackboneMode,
    FuSEReFrameHAR,
    FusionMode,
    NormalizationMode,
)
from inclusive_shift_har.preprocessing.v2 import V2PhysicalPreprocessor
from inclusive_shift_har.training.engine import configure_determinism
from inclusive_shift_har.training.sampling import ParticipantClassSampler
from inclusive_shift_har.training.v2_augmentation import (
    PhysicalAugmentationConfig,
    augment_native_signals,
)
from inclusive_shift_har.training.v2_objectives import (
    masked_reconstruction_loss,
    participant_cvar,
    reliability_gate_target,
    set_valued_nll,
    symmetric_probability_kl,
    vicreg_loss,
)
from inclusive_shift_har.training.weight_averaging import (
    EMATracker,
    SWADTracker,
    UniformAveragingTracker,
    cpu_state_dict,
)

WeightAveraging = Literal["none", "ema", "swa", "swad"]


@dataclass(frozen=True)
class V2TrainingConfig:
    """Serializable configuration for one source-only FuSE-ReFrame trial."""

    seed: int
    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 12
    minimum_epochs: int = 12
    gradient_clip_norm: float = 5.0
    hidden_channels: int = 64
    embedding_dim: int = 96
    dropout: float = 0.1
    backbone: BackboneMode = "multiscale"
    hierarchical: bool = True
    fusion_mode: FusionMode = "gated"
    normalization: NormalizationMode = "group"
    balanced_sampler: bool = True
    use_physical_augmentation: bool = True
    augmentation: PhysicalAugmentationConfig = field(default_factory=PhysicalAugmentationConfig)
    augmented_classification_weight: float = 0.5
    consistency_weight: float = 0.1
    reconstruction_weight: float = 0.05
    reconstruction_mask_probability: float = 0.15
    participant_cvar_weight: float = 0.0
    participant_cvar_fraction: float = 0.3
    branch_supervision_weight: float = 0.0
    gate_supervision_weight: float = 0.0
    gate_oracle_temperature: float = 0.25
    vicreg_weight: float = 0.0
    domain_adversarial_weight: float = 0.0
    domain_grl_max_strength: float = 1.0
    domain_grl_warmup_epochs: int = 10
    weight_averaging: WeightAveraging = "swad"
    ema_decay: float = 0.99
    swa_start_fraction: float = 0.5
    swad_optimum_patience: int = 3
    swad_overfit_patience: int = 6
    swad_tolerance_rate: float = 1.2
    mixed_precision: bool = True

    def __post_init__(self) -> None:
        if self.seed < 0 or min(self.epochs, self.batch_size, self.patience) < 1:
            raise ValueError("seed, epoch, batch, and patience settings are invalid")
        if not 1 <= self.minimum_epochs <= self.epochs:
            raise ValueError("minimum_epochs must lie within the epoch budget")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("optimizer settings are invalid")
        if self.hidden_channels < 8 or self.embedding_dim < 8 or not 0 <= self.dropout < 1:
            raise ValueError("model dimension/dropout settings are invalid")
        if self.backbone not in {"multiscale", "compact_residual", "inception", "tinyhar"}:
            raise ValueError("invalid FuSE-ReFrame backbone")
        if self.fusion_mode not in {"raw", "invariant", "static", "gated"}:
            raise ValueError("invalid FuSE-ReFrame fusion mode")
        if self.normalization not in {"batch", "group", "layer"}:
            raise ValueError("normalization must be batch, group, or layer")
        for name in (
            "augmented_classification_weight",
            "consistency_weight",
            "reconstruction_weight",
            "participant_cvar_weight",
            "branch_supervision_weight",
            "gate_supervision_weight",
            "vicreg_weight",
            "domain_adversarial_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if not 0 < self.reconstruction_mask_probability < 1:
            raise ValueError("reconstruction mask probability must lie in (0, 1)")
        if not 0 < self.participant_cvar_fraction <= 1:
            raise ValueError("participant CVaR fraction must lie in (0, 1]")
        if self.gate_oracle_temperature <= 0:
            raise ValueError("gate-oracle temperature must be positive")
        if self.gate_supervision_weight and self.fusion_mode != "gated":
            raise ValueError("gate supervision requires gated fusion")
        if not 0 <= self.domain_grl_max_strength <= 1:
            raise ValueError("maximum domain GRL strength must lie in [0, 1]")
        if self.domain_grl_warmup_epochs < 1:
            raise ValueError("domain GRL warmup must be positive")
        if self.weight_averaging not in {"none", "ema", "swa", "swad"}:
            raise ValueError("weight_averaging must be none, ema, swa, or swad")
        if not 0 < self.ema_decay < 1:
            raise ValueError("EMA decay must lie in (0, 1)")
        if not 0 < self.swa_start_fraction < 1:
            raise ValueError("SWA start fraction must lie in (0, 1)")


def v2_training_config_from_dict(payload: dict[str, Any]) -> V2TrainingConfig:
    """Restore a serialized v2 config, including its nested augmentation record."""

    values = dict(payload)
    augmentation = values.get("augmentation")
    if isinstance(augmentation, dict):
        values["augmentation"] = PhysicalAugmentationConfig(**augmentation)
    return V2TrainingConfig(**values)


class V2WindowDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Native-unit signals, allowed class sets, and participant indices."""

    def __init__(
        self,
        signals: NDArray[np.float32],
        allowed_classes: NDArray[np.bool_],
        participant_indices: NDArray[np.int64],
    ) -> None:
        if signals.ndim != 3 or signals.shape[1:] != (128, 6):
            raise ValueError("v2 signals must have shape [window,128,6]")
        if allowed_classes.shape != (signals.shape[0], 3):
            raise ValueError("v2 allowed-class masks must have shape [window,3]")
        if participant_indices.shape != (signals.shape[0],):
            raise ValueError("v2 participant indices are not aligned")
        if not np.isfinite(signals).all() or not np.all(allowed_classes.any(axis=1)):
            raise ValueError("v2 dataset contains non-finite input or an empty label set")
        self.signals = torch.from_numpy(np.ascontiguousarray(signals))
        self.allowed_classes = torch.from_numpy(np.ascontiguousarray(allowed_classes))
        self.participants = torch.from_numpy(np.ascontiguousarray(participant_indices))

    def __len__(self) -> int:
        return self.signals.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.signals[index], self.allowed_classes[index], self.participants[index]


def exact_allowed_classes(labels: NDArray[np.int64], *, num_classes: int = 3) -> NDArray[np.bool_]:
    """Convert exact integer labels to an allowed-class mask."""

    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or np.any(values < 0) or np.any(values >= num_classes):
        raise ValueError("exact labels are outside the declared class range")
    result = np.zeros((values.size, num_classes), dtype=np.bool_)
    result[np.arange(values.size), values] = True
    return result


def _participant_indices(participant_ids: list[str]) -> tuple[NDArray[np.int64], dict[str, int]]:
    names = sorted(set(participant_ids))
    mapping = {name: index for index, name in enumerate(names)}
    return np.asarray([mapping[name] for name in participant_ids], dtype=np.int64), mapping


def _allowed_class_set_ids(allowed_classes: NDArray[np.bool_]) -> list[int]:
    """Encode exact or partial label sets without inventing an arbitrary leaf class."""

    if allowed_classes.ndim != 2 or allowed_classes.shape[1] != 3:
        raise ValueError("allowed-class sets must have shape [example,3]")
    bit_weights = np.asarray((1, 2, 4), dtype=np.int64)
    identifiers = np.asarray(allowed_classes, dtype=np.int64) @ bit_weights
    if np.any(identifiers == 0):
        raise ValueError("an allowed-class set cannot be empty")
    return [int(item) for item in identifiers]


def _build_model(
    preprocessor: V2PhysicalPreprocessor,
    config: V2TrainingConfig,
    *,
    num_source_domains: int,
) -> FuSEReFrameHAR:
    return FuSEReFrameHAR(
        raw_mean=torch.from_numpy(preprocessor.raw.mean.copy()),
        raw_scale=torch.from_numpy(preprocessor.raw.scale.copy()),
        invariant_mean=torch.from_numpy(preprocessor.invariant_mean.copy()),
        invariant_scale=torch.from_numpy(preprocessor.invariant_scale.copy()),
        clipping_thresholds=torch.from_numpy(preprocessor.clipping_thresholds.copy()),
        hidden_channels=config.hidden_channels,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
        backbone=config.backbone,
        hierarchical=config.hierarchical,
        fusion_mode=config.fusion_mode,
        normalization=config.normalization,
        num_source_domains=(num_source_domains if config.domain_adversarial_weight else None),
    )


def _autocast(device: torch.device, enabled: bool) -> torch.autocast:
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=enabled and device.type == "cuda",
    )


def _predict(
    model: FuSEReFrameHAR,
    signals: NDArray[np.float32],
    labels: NDArray[np.int64],
    participant_ids: list[str],
    *,
    class_names: tuple[str, str, str],
    batch_size: int,
    device: torch.device,
    mixed_precision: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any], NDArray[np.float64]]:
    model.eval()
    logits: list[NDArray[np.float64]] = []
    gates: list[NDArray[np.float64]] = []
    with torch.no_grad():
        for start in range(0, signals.shape[0], batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(signals[start : start + batch_size])).to(
                device
            )
            with _autocast(device, mixed_precision):
                output = model(batch)
            logits.append(output.logits.detach().float().cpu().numpy().astype(np.float64))
            gates.append(output.gate.detach().float().cpu().numpy().astype(np.float64))
    log_probabilities = np.concatenate(logits, axis=0)
    probabilities = np.exp(log_probabilities)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    report = classification_report(
        labels,
        probabilities,
        participant_ids,
        class_names=class_names,
    )
    return log_probabilities, probabilities, report, np.concatenate(gates, axis=0)


def _mask_native_batch(
    signals: Tensor,
    raw_mean: Tensor,
    *,
    probability: float,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    mask = (
        torch.rand(
            signals.shape,
            device=signals.device,
            generator=generator,
        )
        < probability
    )
    masked = torch.where(mask, raw_mean.reshape(1, 1, 6).to(signals), signals)
    return masked, mask


def _train_epoch(
    model: FuSEReFrameHAR,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    config: V2TrainingConfig,
    preprocessor: V2PhysicalPreprocessor,
    *,
    epoch: int,
    device: torch.device,
    augmentation_generator: torch.Generator,
    mask_generator: torch.Generator,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    example_count = 0
    raw_mean = torch.from_numpy(preprocessor.raw.mean.copy()).to(device=device, dtype=torch.float32)
    raw_scale = torch.from_numpy(preprocessor.raw.scale.copy()).to(
        device=device, dtype=torch.float32
    )
    grl_strength = config.domain_grl_max_strength * min(
        1.0,
        epoch / config.domain_grl_warmup_epochs,
    )
    for signals, allowed_classes, participant_indices in loader:
        signals = signals.to(device)
        allowed_classes = allowed_classes.to(device)
        participant_indices = participant_indices.to(device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, config.mixed_precision):
            clean = model(signals, grl_strength=grl_strength)
            exact_or_partial = set_valued_nll(clean.logits, allowed_classes)
            classification = exact_or_partial.mean()
            total = classification
            components: dict[str, Tensor] = {"classification": classification}
            if config.branch_supervision_weight:
                branch_supervision = 0.5 * (
                    set_valued_nll(clean.raw_logits, allowed_classes).mean()
                    + set_valued_nll(clean.invariant_logits, allowed_classes).mean()
                )
                total = total + config.branch_supervision_weight * branch_supervision
                components["branch_supervision"] = branch_supervision
            if config.domain_adversarial_weight:
                if clean.domain_logits is None:
                    raise RuntimeError("domain-adversarial training requires a domain head")
                domain_adversarial = F.cross_entropy(clean.domain_logits, participant_indices)
                total = total + config.domain_adversarial_weight * domain_adversarial
                components["domain_adversarial"] = domain_adversarial
                components["domain_accuracy"] = (
                    (clean.domain_logits.argmax(dim=1) == participant_indices).float().mean()
                )
                components["grl_strength"] = torch.as_tensor(
                    grl_strength,
                    dtype=signals.dtype,
                    device=signals.device,
                )
            if config.participant_cvar_weight:
                cvar = participant_cvar(
                    exact_or_partial,
                    participant_indices,
                    tail_fraction=config.participant_cvar_fraction,
                )
                total = total + config.participant_cvar_weight * cvar
                components["participant_cvar"] = cvar
            if config.use_physical_augmentation:
                augmented_batch = augment_native_signals(
                    signals,
                    generator=augmentation_generator,
                    config=config.augmentation,
                )
                augmented = model(augmented_batch.signals)
                augmented_classification = set_valued_nll(
                    augmented.logits,
                    allowed_classes,
                ).mean()
                consistency = symmetric_probability_kl(clean.logits, augmented.logits)
                total = (
                    total
                    + config.augmented_classification_weight * augmented_classification
                    + config.consistency_weight * consistency
                )
                components["augmented_classification"] = augmented_classification
                components["consistency"] = consistency
                if config.branch_supervision_weight:
                    augmented_branch_supervision = 0.5 * (
                        set_valued_nll(augmented.raw_logits, allowed_classes).mean()
                        + set_valued_nll(augmented.invariant_logits, allowed_classes).mean()
                    )
                    total = (
                        total
                        + config.branch_supervision_weight
                        * config.augmented_classification_weight
                        * augmented_branch_supervision
                    )
                    components["augmented_branch_supervision"] = augmented_branch_supervision
                if config.gate_supervision_weight:
                    gate_target = reliability_gate_target(
                        augmented.raw_logits,
                        augmented.invariant_logits,
                        allowed_classes,
                        temperature=config.gate_oracle_temperature,
                    )
                    gate_supervision = F.binary_cross_entropy(
                        augmented.gate.squeeze(1),
                        gate_target,
                    )
                    total = total + config.gate_supervision_weight * gate_supervision
                    components["gate_supervision"] = gate_supervision
                    components["gate_oracle_raw_preference"] = gate_target.mean()
                if config.vicreg_weight:
                    vicreg = vicreg_loss(clean.content, augmented.content)
                    total = total + config.vicreg_weight * vicreg
                    components["vicreg"] = vicreg
            if config.reconstruction_weight:
                masked_signals, reconstruction_mask = _mask_native_batch(
                    signals,
                    raw_mean,
                    probability=config.reconstruction_mask_probability,
                    generator=mask_generator,
                )
                masked_output = model(masked_signals)
                standardized_target = (signals - raw_mean.reshape(1, 1, 6)) / raw_scale.reshape(
                    1, 1, 6
                )
                reconstruction = masked_reconstruction_loss(
                    masked_output.reconstruction,
                    standardized_target,
                    reconstruction_mask,
                )
                total = total + config.reconstruction_weight * reconstruction
                components["reconstruction"] = reconstruction
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        batch_size = signals.shape[0]
        example_count += batch_size
        totals["total"] = totals.get("total", 0.0) + float(total.detach()) * batch_size
        for name, value in components.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
    return {name: value / example_count for name, value in totals.items()}


def _write_json_create_only(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
        stream.write(b"\n")
    return sha256_file(path)


def train_fuse_reframe(
    train_signals: NDArray[np.float32],
    train_allowed_classes: NDArray[np.bool_],
    train_participant_ids: list[str],
    validation_signals: NDArray[np.float32],
    validation_labels: NDArray[np.int64],
    validation_participant_ids: list[str],
    *,
    preprocessor: V2PhysicalPreprocessor,
    config: V2TrainingConfig,
    class_names: tuple[str, str, str],
    lineage: dict[str, Any],
    output_directory: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Train one immutable, target-sealed FuSE-ReFrame source-development cell."""

    if set(train_participant_ids) & set(validation_participant_ids):
        raise ValueError("v2 training and validation participants overlap")
    if lineage.get("target_subject_or_window_records_loaded") is not False:
        raise PermissionError("v2 source engine refuses target-bearing lineage")
    if lineage.get("target_performance_or_prediction_accessed") is not False:
        raise PermissionError("v2 source engine refuses target-informed lineage")
    if output_directory.exists():
        raise FileExistsError(f"v2 output directory already exists: {output_directory}")
    output_directory.mkdir(parents=True)
    configure_determinism(config.seed)
    participant_indices, participant_map = _participant_indices(train_participant_ids)
    dataset = V2WindowDataset(train_signals, train_allowed_classes, participant_indices)
    sampler_label_sets = _allowed_class_set_ids(train_allowed_classes)
    sampler = (
        ParticipantClassSampler(
            train_participant_ids,
            sampler_label_sets,
            seed=config.seed + 1,
        )
        if config.balanced_sampler
        else None
    )
    loader_generator = torch.Generator().manual_seed(config.seed + 1)
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]] = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        generator=loader_generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = _build_model(
        preprocessor,
        config,
        num_source_domains=len(participant_map),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler(  # type: ignore[attr-defined]
        "cuda",
        enabled=device.type == "cuda" and config.mixed_precision,
    )
    augmentation_generator = torch.Generator(device=device).manual_seed(config.seed + 2)
    mask_generator = torch.Generator(device=device).manual_seed(config.seed + 3)
    ema = EMATracker(config.ema_decay) if config.weight_averaging == "ema" else None
    swa = UniformAveragingTracker() if config.weight_averaging == "swa" else None
    swa_start_epoch = max(1, math.ceil(config.epochs * config.swa_start_fraction))
    swad = (
        SWADTracker(
            optimum_patience=config.swad_optimum_patience,
            overfit_patience=config.swad_overfit_patience,
            tolerance_rate=config.swad_tolerance_rate,
        )
        if config.weight_averaging == "swad"
        else None
    )
    best_metric = float("-inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_losses = _train_epoch(
            model,
            loader,
            optimizer,
            scaler,
            config,
            preprocessor,
            epoch=epoch,
            device=device,
            augmentation_generator=augmentation_generator,
            mask_generator=mask_generator,
        )
        scheduler.step()
        if ema is not None:
            ema.update(model)
        if swa is not None and epoch >= swa_start_epoch:
            swa.update(model)
        _, _, validation_report, validation_gate = _predict(
            model,
            validation_signals,
            validation_labels,
            validation_participant_ids,
            class_names=class_names,
            batch_size=config.batch_size,
            device=device,
            mixed_precision=config.mixed_precision,
        )
        metric = float(validation_report["primary"]["mean_participant_macro_f1"])
        if metric > best_metric and not math.isclose(metric, best_metric, abs_tol=1e-12):
            best_metric = metric
            best_epoch = epoch
            best_state = cpu_state_dict(model)
        if swad is not None:
            swad.update(epoch, 1.0 - metric, model)
        history.append(
            {
                "epoch": epoch,
                "training_losses": train_losses,
                "validation_mean_participant_macro_f1": metric,
                "validation_worst_participant_macro_f1": float(
                    validation_report["primary"]["worst_participant_macro_f1"]
                ),
                "validation_gate_mean": float(validation_gate.mean()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "best_epoch_after_epoch": best_epoch,
            }
        )
        if epoch >= config.minimum_epochs and epoch - best_epoch >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("v2 training completed without a selectable source checkpoint")
    selection: dict[str, Any] = {"method": "source_validation_best", "epoch": best_epoch}
    selected_state = best_state
    if ema is not None and ema.state is not None:
        selected_state = ema.state
        selection = {"method": "ema", "update_count": ema.update_count, "decay": ema.decay}
    elif swa is not None and swa.state is not None:
        selected_state = swa.state
        selection = {
            "method": "swa",
            "start_epoch": swa_start_epoch,
            "end_epoch": len(history),
            "state_count": swa.update_count,
        }
    elif swa is not None:
        selection = {
            "method": "source_validation_best_swa_fallback",
            "epoch": best_epoch,
            "reason": "training_stopped_before_predeclared_swa_interval",
        }
    elif swad is not None:
        swad_result = swad.result()
        if swad_result is not None:
            selected_state = swad_result.state
            selection = {
                "method": "swad",
                "start_epoch": swad_result.start_epoch,
                "end_epoch": swad_result.end_epoch,
                "state_count": swad_result.state_count,
                "threshold": swad_result.threshold,
            }
        else:
            selection = {
                "method": "source_validation_best_swad_fallback",
                "epoch": best_epoch,
            }
    model.load_state_dict(selected_state)
    validation_logits, validation_probabilities, validation_report, validation_gate = _predict(
        model,
        validation_signals,
        validation_labels,
        validation_participant_ids,
        class_names=class_names,
        batch_size=config.batch_size,
        device=device,
        mixed_precision=config.mixed_precision,
    )
    checkpoint = {
        "schema_version": "1.0.0",
        "record_kind": "fuse_reframe_source_development_checkpoint",
        "evidence_status": "source_development_not_confirmatory_target_sealed",
        "configuration": asdict(config),
        "configuration_sha256": canonical_json_sha256(asdict(config)),
        "preprocessor": preprocessor.to_dict(),
        "lineage": copy.deepcopy(lineage),
        "participant_index_map": participant_map,
        "class_names": list(class_names),
        "selection": selection,
        "best_epoch": best_epoch,
        "best_validation_mean_participant_macro_f1": best_metric,
        "history": history,
        "model_state": selected_state,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "target_information_used_for_selection": False,
    }
    checkpoint_path = output_directory / "selected.pt"
    with checkpoint_path.open("xb") as stream:
        torch.save(checkpoint, stream)
    prediction_path = output_directory / "source_validation_predictions.npz"
    with prediction_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            logits=validation_logits,
            probabilities=validation_probabilities,
            labels=validation_labels,
            participant_ids=np.asarray(validation_participant_ids),
            gate=validation_gate,
        )
    summary: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "fuse_reframe_source_development_result",
        "status": "source_development_complete_target_sealed",
        "evidence_status": "source_development_not_confirmatory",
        "configuration": asdict(config),
        "configuration_sha256": canonical_json_sha256(asdict(config)),
        "preprocessor": preprocessor.to_dict(),
        "lineage": copy.deepcopy(lineage),
        "selection": selection,
        "train_window_count": int(train_signals.shape[0]),
        "validation_window_count": int(validation_signals.shape[0]),
        "train_participants": sorted(set(train_participant_ids)),
        "validation_participants": sorted(set(validation_participant_ids)),
        "validation_report": validation_report,
        "validation_gate_mean": float(validation_gate.mean()),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "checkpoint": {
            "path": checkpoint_path.as_posix(),
            "sha256": sha256_file(checkpoint_path),
        },
        "predictions": {
            "path": prediction_path.as_posix(),
            "sha256": sha256_file(prediction_path),
        },
        "history": history,
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
    }
    summary["record_sha256"] = canonical_json_sha256(summary)
    summary_path = output_directory / "result.json"
    _write_json_create_only(summary_path, summary)
    return summary


def train_fuse_reframe_fixed_evaluation(
    train_signals: NDArray[np.float32],
    train_allowed_classes: NDArray[np.bool_],
    train_participant_ids: list[str],
    evaluation_signals: NDArray[np.float32],
    evaluation_labels: NDArray[np.int64],
    evaluation_participant_ids: list[str],
    *,
    preprocessor: V2PhysicalPreprocessor,
    config: V2TrainingConfig,
    fixed_epochs: int,
    averaging_interval: tuple[int, int] | None,
    class_names: tuple[str, str, str],
    lineage: dict[str, Any],
    output_directory: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Fit without reading the outer fold, then evaluate that fold exactly once."""

    if fixed_epochs < 1:
        raise ValueError("fixed outer-fit epochs must be positive")
    if set(train_participant_ids) & set(evaluation_participant_ids):
        raise ValueError("fixed-fit training and evaluation participants overlap")
    if lineage.get("target_subject_or_window_records_loaded") is not False:
        raise PermissionError("fixed source fit refuses target-bearing lineage")
    if lineage.get("target_performance_or_prediction_accessed") is not False:
        raise PermissionError("fixed source fit refuses target-informed lineage")
    if lineage.get("fixed_schedule_derived_without_outer_evaluation") is not True:
        raise PermissionError("fixed source fit requires an inner-derived schedule declaration")
    if config.weight_averaging == "swad":
        if averaging_interval is None:
            raise ValueError("fixed SWAD refit requires an interval derived from inner folds")
        start_epoch, end_epoch = averaging_interval
        if not 1 <= start_epoch <= end_epoch <= fixed_epochs:
            raise ValueError("inner-derived SWAD interval is outside the fixed epoch schedule")
    elif averaging_interval is not None:
        raise ValueError("an explicit averaging interval is only valid for SWAD refits")
    if output_directory.exists():
        raise FileExistsError(f"v2 output directory already exists: {output_directory}")

    effective_config = replace(
        config,
        epochs=fixed_epochs,
        minimum_epochs=fixed_epochs,
        patience=fixed_epochs,
    )
    output_directory.mkdir(parents=True)
    configure_determinism(effective_config.seed)
    participant_indices, participant_map = _participant_indices(train_participant_ids)
    dataset = V2WindowDataset(train_signals, train_allowed_classes, participant_indices)
    sampler_label_sets = _allowed_class_set_ids(train_allowed_classes)
    sampler = (
        ParticipantClassSampler(
            train_participant_ids,
            sampler_label_sets,
            seed=effective_config.seed + 1,
        )
        if effective_config.balanced_sampler
        else None
    )
    loader_generator = torch.Generator().manual_seed(effective_config.seed + 1)
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]] = DataLoader(
        dataset,
        batch_size=effective_config.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        generator=loader_generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = _build_model(
        preprocessor,
        effective_config,
        num_source_domains=len(participant_map),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=effective_config.learning_rate,
        weight_decay=effective_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=fixed_epochs)
    scaler = torch.amp.GradScaler(  # type: ignore[attr-defined]
        "cuda",
        enabled=device.type == "cuda" and effective_config.mixed_precision,
    )
    augmentation_generator = torch.Generator(device=device).manual_seed(effective_config.seed + 2)
    mask_generator = torch.Generator(device=device).manual_seed(effective_config.seed + 3)
    ema = EMATracker(effective_config.ema_decay) if config.weight_averaging == "ema" else None
    uniform = UniformAveragingTracker() if config.weight_averaging in {"swa", "swad"} else None
    if config.weight_averaging == "swa":
        averaging_start = max(1, math.ceil(fixed_epochs * config.swa_start_fraction))
        averaging_end = fixed_epochs
    elif config.weight_averaging == "swad":
        assert averaging_interval is not None
        averaging_start, averaging_end = averaging_interval
    else:
        averaging_start = fixed_epochs
        averaging_end = fixed_epochs

    history: list[dict[str, Any]] = []
    for epoch in range(1, fixed_epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_losses = _train_epoch(
            model,
            loader,
            optimizer,
            scaler,
            effective_config,
            preprocessor,
            epoch=epoch,
            device=device,
            augmentation_generator=augmentation_generator,
            mask_generator=mask_generator,
        )
        scheduler.step()
        if ema is not None:
            ema.update(model)
        if uniform is not None and averaging_start <= epoch <= averaging_end:
            uniform.update(model)
        history.append(
            {
                "epoch": epoch,
                "training_losses": train_losses,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "outer_evaluation_accessed": False,
            }
        )

    if ema is not None and ema.state is not None:
        selected_state = ema.state
        selection: dict[str, Any] = {
            "method": "fixed_epoch_ema_inner_derived",
            "fixed_epochs": fixed_epochs,
            "update_count": ema.update_count,
            "decay": ema.decay,
        }
    elif uniform is not None and uniform.state is not None:
        selected_state = uniform.state
        selection = {
            "method": f"fixed_epoch_{config.weight_averaging}_inner_derived",
            "fixed_epochs": fixed_epochs,
            "start_epoch": averaging_start,
            "end_epoch": averaging_end,
            "state_count": uniform.update_count,
        }
    else:
        selected_state = cpu_state_dict(model)
        selection = {"method": "fixed_last_epoch_inner_derived", "fixed_epochs": fixed_epochs}
    model.load_state_dict(selected_state)

    evaluation_logits, evaluation_probabilities, evaluation_report, evaluation_gate = _predict(
        model,
        evaluation_signals,
        evaluation_labels,
        evaluation_participant_ids,
        class_names=class_names,
        batch_size=effective_config.batch_size,
        device=device,
        mixed_precision=effective_config.mixed_precision,
    )
    checkpoint = {
        "schema_version": "1.0.0",
        "record_kind": "fuse_reframe_fixed_source_checkpoint",
        "evidence_status": "outer_source_development_not_confirmatory_target_sealed",
        "configuration": asdict(effective_config),
        "configuration_sha256": canonical_json_sha256(asdict(effective_config)),
        "preprocessor": preprocessor.to_dict(),
        "lineage": copy.deepcopy(lineage),
        "participant_index_map": participant_map,
        "class_names": list(class_names),
        "selection": selection,
        "history": history,
        "model_state": selected_state,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "outer_evaluation_access_count_during_training": 0,
        "target_information_used_for_selection": False,
    }
    checkpoint_path = output_directory / "selected.pt"
    with checkpoint_path.open("xb") as stream:
        torch.save(checkpoint, stream)
    prediction_path = output_directory / "outer_source_predictions.npz"
    with prediction_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            logits=evaluation_logits,
            probabilities=evaluation_probabilities,
            labels=evaluation_labels,
            participant_ids=np.asarray(evaluation_participant_ids),
            gate=evaluation_gate,
        )
    summary: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "fuse_reframe_nested_outer_source_result",
        "status": "outer_source_evaluation_complete_target_sealed",
        "evidence_status": "outer_source_development_not_confirmatory",
        "configuration": asdict(effective_config),
        "configuration_sha256": canonical_json_sha256(asdict(effective_config)),
        "preprocessor": preprocessor.to_dict(),
        "lineage": copy.deepcopy(lineage),
        "selection": selection,
        "train_window_count": int(train_signals.shape[0]),
        "evaluation_window_count": int(evaluation_signals.shape[0]),
        "train_participants": sorted(set(train_participant_ids)),
        "evaluation_participants": sorted(set(evaluation_participant_ids)),
        "evaluation_report": evaluation_report,
        "evaluation_gate_mean": float(evaluation_gate.mean()),
        "checkpoint": {"path": checkpoint_path.as_posix(), "sha256": sha256_file(checkpoint_path)},
        "predictions": {
            "path": prediction_path.as_posix(),
            "sha256": sha256_file(prediction_path),
        },
        "history": history,
        "outer_evaluation_access_count_during_training": 0,
        "outer_evaluation_access_count_after_training": 1,
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
    }
    summary["record_sha256"] = canonical_json_sha256(summary)
    _write_json_create_only(output_directory / "result.json", summary)
    return summary
