"""Deterministic source-only sensor-corruption diagnostics for FuSE-ReFrame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.models.common import require_window_tensor
from inclusive_shift_har.models.fuse_reframe import FuSEReFrameHAR
from inclusive_shift_har.training.v2_augmentation import (
    PhysicalAugmentationConfig,
    augment_native_signals,
)

CorruptionName = Literal[
    "clean",
    "rotation",
    "gain",
    "noise",
    "bias",
    "drift",
    "channel_dropout",
    "temporal_gap",
    "clipping",
    "time_warp",
]


@dataclass(frozen=True)
class CorruptionSpec:
    """One named corruption with a normalized, explicitly recorded severity."""

    name: CorruptionName
    severity: float

    def __post_init__(self) -> None:
        if self.name == "clean":
            if self.severity != 0:
                raise ValueError("clean corruption severity must be zero")
        elif not 0 < self.severity <= 1:
            raise ValueError("non-clean corruption severity must lie in (0, 1]")


@dataclass(frozen=True)
class CorruptedSignals:
    """Corrupted tensor plus an auditable affected-value fraction."""

    signals: Tensor
    affected_fraction: float


def default_corruption_suite() -> tuple[CorruptionSpec, ...]:
    """Return the predeclared moderate/severe sensor-reliability grid."""

    return (
        CorruptionSpec("clean", 0.0),
        CorruptionSpec("rotation", 0.25),
        CorruptionSpec("rotation", 0.50),
        CorruptionSpec("gain", 0.20),
        CorruptionSpec("gain", 0.50),
        CorruptionSpec("noise", 0.10),
        CorruptionSpec("noise", 0.30),
        CorruptionSpec("bias", 0.20),
        CorruptionSpec("bias", 0.50),
        CorruptionSpec("drift", 0.20),
        CorruptionSpec("drift", 0.50),
        CorruptionSpec("channel_dropout", 1 / 6),
        CorruptionSpec("channel_dropout", 2 / 6),
        CorruptionSpec("temporal_gap", 0.125),
        CorruptionSpec("temporal_gap", 0.25),
        CorruptionSpec("clipping", 0.25),
        CorruptionSpec("clipping", 0.50),
        CorruptionSpec("time_warp", 0.10),
        CorruptionSpec("time_warp", 0.25),
    )


def _neutral_augmentation(**overrides: float) -> PhysicalAugmentationConfig:
    values: dict[str, float] = {
        "amplitude_low": 1.0,
        "amplitude_high": 1.0,
        "time_scale_low": 1.0,
        "time_scale_high": 1.0,
        "rotation_degrees": 0.0,
        "acceleration_noise_sd_g": 0.0,
        "gyroscope_noise_sd_rad_s": 0.0,
        "acceleration_bias_g": 0.0,
        "gyroscope_bias_rad_s": 0.0,
        "acceleration_drift_g": 0.0,
        "gyroscope_drift_rad_s": 0.0,
        "channel_dropout_probability": 0.0,
    }
    values.update(overrides)
    return PhysicalAugmentationConfig(**values)


def apply_sensor_corruption(
    signals: Tensor,
    spec: CorruptionSpec,
    *,
    channel_scale: Tensor,
    clipping_thresholds: Tensor,
    seed: int,
) -> CorruptedSignals:
    """Apply a replayable corruption in native units without reading labels."""

    require_window_tensor(signals)
    if channel_scale.shape != (6,) or clipping_thresholds.shape != (6,):
        raise ValueError("corruption scales and clipping thresholds must contain six values")
    if torch.any(channel_scale <= 0) or torch.any(clipping_thresholds <= 0):
        raise ValueError("corruption scales and clipping thresholds must be positive")
    generator = torch.Generator(device=signals.device).manual_seed(seed)
    if spec.name == "clean":
        return CorruptedSignals(signals.clone(), 0.0)
    if spec.name == "rotation":
        augmented = augment_native_signals(
            signals,
            generator=generator,
            config=_neutral_augmentation(rotation_degrees=90.0 * spec.severity),
        )
        return CorruptedSignals(augmented.signals, 1.0)
    if spec.name == "gain":
        augmented = augment_native_signals(
            signals,
            generator=generator,
            config=_neutral_augmentation(
                amplitude_low=max(0.05, 1.0 - spec.severity),
                amplitude_high=1.0 + spec.severity,
            ),
        )
        return CorruptedSignals(augmented.signals, 1.0)
    if spec.name == "time_warp":
        augmented = augment_native_signals(
            signals,
            generator=generator,
            config=_neutral_augmentation(
                time_scale_low=max(0.25, 1.0 - spec.severity),
                time_scale_high=1.0 + spec.severity,
            ),
        )
        return CorruptedSignals(augmented.signals, 1.0)

    result = signals.clone()
    scale = channel_scale.to(device=signals.device, dtype=signals.dtype).reshape(1, 1, 6)
    if spec.name == "noise":
        noise = torch.randn(
            signals.shape,
            dtype=signals.dtype,
            device=signals.device,
            generator=generator,
        )
        return CorruptedSignals(result + spec.severity * scale * noise, 1.0)
    if spec.name in {"bias", "drift"}:
        offset = (
            (
                2.0
                * torch.rand(
                    (signals.shape[0], 1, 6),
                    dtype=signals.dtype,
                    device=signals.device,
                    generator=generator,
                )
                - 1.0
            )
            * spec.severity
            * scale
        )
        if spec.name == "bias":
            return CorruptedSignals(result + offset, 1.0)
        ramp = torch.linspace(-1.0, 1.0, signals.shape[1], device=signals.device).to(signals)
        return CorruptedSignals(result + ramp.reshape(1, -1, 1) * offset, 1.0)
    if spec.name == "channel_dropout":
        channel_count = min(6, max(1, round(6 * spec.severity)))
        mask = torch.zeros((signals.shape[0], 6), dtype=torch.bool, device=signals.device)
        for example in range(signals.shape[0]):
            selected = torch.randperm(6, generator=generator, device=signals.device)[:channel_count]
            mask[example, selected] = True
        result.masked_fill_(mask.unsqueeze(1), 0.0)
        return CorruptedSignals(result, channel_count / 6)
    if spec.name == "temporal_gap":
        gap_length = min(signals.shape[1], max(1, round(signals.shape[1] * spec.severity)))
        maximum_start = signals.shape[1] - gap_length
        starts = torch.randint(
            0,
            maximum_start + 1,
            (signals.shape[0],),
            generator=generator,
            device=signals.device,
        )
        for example, start in enumerate(starts.tolist()):
            result[example, start : start + gap_length] = 0.0
        return CorruptedSignals(result, gap_length / signals.shape[1])
    if spec.name == "clipping":
        threshold = (
            clipping_thresholds.to(device=signals.device, dtype=signals.dtype)
            * (1.0 - 0.9 * spec.severity)
        ).reshape(1, 1, 6)
        clipped = torch.maximum(torch.minimum(result, threshold), -threshold)
        affected = float((clipped != result).float().mean().item())
        return CorruptedSignals(clipped, affected)
    raise AssertionError(f"unhandled corruption {spec.name!r}")


def _probabilities(log_probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    probabilities = np.exp(log_probabilities)
    return np.asarray(
        probabilities / probabilities.sum(axis=1, keepdims=True),
        dtype=np.float64,
    )


def evaluate_corruption_suite(
    model: FuSEReFrameHAR,
    signals: NDArray[np.float32],
    labels: NDArray[np.int64],
    participant_ids: list[str],
    *,
    class_names: tuple[str, str, str],
    channel_scale: NDArray[np.float64],
    clipping_thresholds: NDArray[np.float64],
    specs: tuple[CorruptionSpec, ...],
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate combined and branch predictions on a source-only corruption grid."""

    if signals.shape[0] != labels.shape[0] or len(participant_ids) != labels.shape[0]:
        raise ValueError("corruption evaluation arrays are not aligned")
    model.eval()
    scale = torch.from_numpy(np.asarray(channel_scale, dtype=np.float32)).to(device)
    thresholds = torch.from_numpy(np.asarray(clipping_thresholds, dtype=np.float32)).to(device)
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, spec in enumerate(specs):
            combined_rows: list[NDArray[np.float64]] = []
            raw_rows: list[NDArray[np.float64]] = []
            invariant_rows: list[NDArray[np.float64]] = []
            gate_rows: list[NDArray[np.float64]] = []
            affected_mass = 0.0
            for start in range(0, signals.shape[0], batch_size):
                batch = torch.from_numpy(
                    np.ascontiguousarray(signals[start : start + batch_size])
                ).to(device)
                corrupted = apply_sensor_corruption(
                    batch,
                    spec,
                    channel_scale=scale,
                    clipping_thresholds=thresholds,
                    seed=seed + index * 100_003 + start,
                )
                output = model(corrupted.signals)
                combined_rows.append(output.logits.float().cpu().numpy().astype(np.float64))
                raw_rows.append(output.raw_logits.float().cpu().numpy().astype(np.float64))
                invariant_rows.append(
                    output.invariant_logits.float().cpu().numpy().astype(np.float64)
                )
                gate_rows.append(output.gate.float().cpu().numpy().astype(np.float64))
                affected_mass += corrupted.affected_fraction * batch.shape[0]
            combined_log = np.concatenate(combined_rows)
            raw_log = np.concatenate(raw_rows)
            invariant_log = np.concatenate(invariant_rows)
            gate = np.concatenate(gate_rows).reshape(-1)
            label_rows = np.arange(labels.size)
            raw_nll = -raw_log[label_rows, labels]
            invariant_nll = -invariant_log[label_rows, labels]
            combined_nll = -combined_log[label_rows, labels]
            oracle_raw = raw_nll <= invariant_nll
            records.append(
                {
                    "corruption": spec.name,
                    "severity": spec.severity,
                    "affected_fraction": affected_mass / signals.shape[0],
                    "combined_report": classification_report(
                        labels,
                        _probabilities(combined_log),
                        participant_ids,
                        class_names=class_names,
                    ),
                    "raw_report": classification_report(
                        labels,
                        _probabilities(raw_log),
                        participant_ids,
                        class_names=class_names,
                    ),
                    "invariant_report": classification_report(
                        labels,
                        _probabilities(invariant_log),
                        participant_ids,
                        class_names=class_names,
                    ),
                    "gate": {
                        "mean_raw_weight": float(gate.mean()),
                        "standard_deviation": float(gate.std()),
                        "oracle_routing_accuracy": float(np.mean((gate >= 0.5) == oracle_raw)),
                        "combined_nll_regret_vs_best_branch": float(
                            np.mean(combined_nll - np.minimum(raw_nll, invariant_nll))
                        ),
                    },
                }
            )
    clean = next(record for record in records if record["corruption"] == "clean")
    clean_score = float(clean["combined_report"]["primary"]["mean_participant_macro_f1"])
    corrupted_scores = [
        float(record["combined_report"]["primary"]["mean_participant_macro_f1"])
        for record in records
        if record["corruption"] != "clean"
    ]
    return {
        "schema_version": "1.0.0",
        "suite_role": "source_only_diagnostic_not_target_evidence",
        "case_count": len(records),
        "clean_mean_participant_macro_f1": clean_score,
        "mean_corrupted_participant_macro_f1": float(np.mean(corrupted_scores)),
        "worst_corrupted_participant_macro_f1": min(corrupted_scores),
        "records": records,
    }
