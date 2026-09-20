"""CPU replication of source-only DeepConvLSTM and TinyHAR AICOS controls."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from inclusive_shift_har.data.aicos_har import (
    aicos_in_inclusivehar_units,
    load_aicos_har,
    verify_aicos_archive,
)
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.experiments.aicos_external_benchmark import (
    _complete_participant_mask,
    _report,
    _source_windows,
)
from inclusive_shift_har.experiments.cross_dataset_har import _paired_bootstrap
from inclusive_shift_har.experiments.cross_dataset_neural import (
    ExternalNeuralConfig,
    _configure_determinism,
    _loader,
    _normalization,
    _predict,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models import build_baseline, trainable_parameter_count
from inclusive_shift_har.models.common import HAROutput

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Float32Array = NDArray[np.float32]


def _train_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
) -> float:
    model.train()
    total, examples = 0.0, 0
    for signals, labels, weights in loader:
        optimizer.zero_grad(set_to_none=True)
        signals = signals.to(device)
        labels = labels.to(device)
        weights = weights.to(device)
        output = cast(HAROutput, model(signals))
        per_example = torch.nn.functional.cross_entropy(output.logits, labels, reduction="none")
        loss = torch.sum(per_example * weights) / torch.sum(weights)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.detach()) * labels.numel()
        examples += labels.numel()
    return total / examples


def _fit_control(
    *,
    source: Any,
    target: Any,
    model_name: str,
    output_path: Path,
) -> tuple[FloatArray, dict[str, Any]]:
    config = ExternalNeuralConfig(
        model_name=model_name,
        seed=11,
        epochs=40,
        disable_cudnn=model_name == "deepconvlstm",
    )
    config.validate()
    device = torch.device("cpu")
    source_signals = np.asarray(source.signals, dtype=np.float32)
    target_signals = np.asarray(target.signals, dtype=np.float32)
    validation_people = {"inclusivehar:P8", "inclusivehar:P10"}
    validation = np.flatnonzero(np.isin(source.participant_ids, sorted(validation_people))).astype(
        np.int64
    )
    training = np.flatnonzero(~np.isin(source.participant_ids, sorted(validation_people))).astype(
        np.int64
    )
    if any(set(source.labels[index].tolist()) != {0, 1, 2} for index in (training, validation)):
        raise ValueError("fixed neural source training/validation split lacks a class")

    _configure_determinism(config.seed)
    mean, scale = _normalization(source_signals, training)
    loader = _loader(
        source,
        source_signals,
        training,
        mean=mean,
        scale=scale,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed + 1,
        class_count=3,
    )
    model = build_baseline(model_name, num_classes=3, input_channels=6).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    best_metric, best_epoch = float("-inf"), 0
    history: list[dict[str, Any]] = []
    selection_started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        loss = _train_epoch(model, loader, optimizer, device=device)
        scheduler.step()
        probability = _predict(
            model,
            source_signals,
            validation,
            mean=mean,
            scale=scale,
            batch_size=config.batch_size,
            device=device,
            mixed_precision="disabled",
        )
        report = classification_report(
            source.labels[validation],
            probability,
            source.participant_ids[validation].tolist(),
            class_names=source.class_names,
        )
        metric = float(report["primary"]["mean_participant_macro_f1"])
        improved = metric > best_metric and not np.isclose(metric, best_metric, atol=1e-12)
        if improved:
            best_metric, best_epoch = metric, epoch
        history.append(
            {
                "epoch": epoch,
                "training_weighted_loss": loss,
                "validation_mean_participant_macro_f1": metric,
                "selected_after_epoch": improved,
            }
        )
        if epoch >= config.minimum_epochs and epoch - best_epoch >= config.patience:
            break
    if best_epoch == 0:
        raise RuntimeError("neural source validation did not select an epoch")
    selection_seconds = time.perf_counter() - selection_started

    _configure_determinism(config.seed)
    full = np.arange(source.labels.size, dtype=np.int64)
    final_mean, final_scale = _normalization(source_signals, full)
    final_loader = _loader(
        source,
        source_signals,
        full,
        mean=final_mean,
        scale=final_scale,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed + 1,
        class_count=3,
    )
    final_model = build_baseline(model_name, num_classes=3, input_channels=6).to(device)
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    final_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        final_optimizer, T_max=config.epochs
    )
    final_started = time.perf_counter()
    final_losses = []
    for _epoch in range(1, best_epoch + 1):
        final_losses.append(_train_epoch(final_model, final_loader, final_optimizer, device=device))
        final_scheduler.step()
    target_probability = _predict(
        final_model,
        target_signals,
        np.arange(target.labels.size, dtype=np.int64),
        mean=final_mean,
        scale=final_scale,
        batch_size=config.batch_size,
        device=device,
        mixed_precision="disabled",
    )
    final_seconds = time.perf_counter() - final_started
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        torch.save(
            {
                "schema_version": "1.0.0",
                "model_name": model_name,
                "model_state": final_model.state_dict(),
                "normalization": {"mean": final_mean.tolist(), "scale": final_scale.tolist()},
                "selected_epoch": best_epoch,
                "source_participants": sorted(np.unique(source.participant_ids).tolist()),
                "target_labels_used": False,
                "config": asdict(config),
            },
            stream,
        )
    return target_probability, {
        "model_name": model_name,
        "seed": config.seed,
        "training_config": asdict(config),
        "runtime_backend": "cpu-float32-replication",
        "protocol_difference_from_frozen_external_neural_control": (
            "CPU float32 replaces CUDA float16; architecture, optimizer, weighting, "
            "validation and epoch contract are retained"
        ),
        "selection_training_participants": sorted(
            np.unique(source.participant_ids[training]).tolist()
        ),
        "selection_validation_participants": sorted(
            np.unique(source.participant_ids[validation]).tolist()
        ),
        "selected_epoch": best_epoch,
        "selected_validation_mean_participant_macro_f1": best_metric,
        "selection_history": history,
        "final_training_participants": sorted(np.unique(source.participant_ids).tolist()),
        "final_training_losses": final_losses,
        "parameter_count": trainable_parameter_count(final_model),
        "selection_seconds": selection_seconds,
        "final_fit_and_target_inference_seconds": final_seconds,
        "checkpoint": {
            "path": output_path.name,
            "sha256": sha256_file(output_path),
            "size_bytes": output_path.stat().st_size,
        },
    }


def run(
    *,
    repository_root: Path,
    archive_path: Path,
    source_csv: Path,
    output_directory: Path,
) -> dict[str, Any]:
    if output_directory.exists():
        raise FileExistsError(f"create-only output already exists: {output_directory}")
    output_directory.mkdir(parents=True)
    started = time.perf_counter()
    source = _source_windows(repository_root, source_csv)
    verification = verify_aicos_archive(archive_path)
    target = load_aicos_har(archive_path, verification=verification, folds=("test",))
    target = aicos_in_inclusivehar_units(target)
    probabilities: dict[str, FloatArray] = {}
    records = []
    for model_name, display_name in (
        ("deepconvlstm", "DeepConvLSTM-6ch-CPU"),
        ("tinyhar", "TinyHAR-6ch-CPU"),
    ):
        probability, record = _fit_control(
            source=source,
            target=target,
            model_name=model_name,
            output_path=output_directory / f"{model_name}.pt",
        )
        probabilities[display_name] = probability
        records.append(record)
    prediction_path = output_directory / "predictions.npz"
    prediction_payload: dict[str, Any] = {
        "labels": target.labels,
        "participant_ids": target.participant_ids,
        "window_ids": target.window_ids,
        **{f"probability__{name}": value for name, value in probabilities.items()},
    }
    with prediction_path.open("xb") as stream:
        np.savez_compressed(stream, **prediction_payload)
    primary_mask = _complete_participant_mask(target)
    reports = {
        name: _report(target, probability, primary_mask)
        for name, probability in probabilities.items()
    }
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "experiment_id": "inclusivehar-to-aicos-neural-controls-cpu-v1",
        "status": "complete_cpu_replication_control",
        "evidence_status": "EXTERNAL_ZERO_SHOT_DIAGNOSTIC_CPU_REPLICATION",
        "created_at": datetime.now(UTC).isoformat(),
        "source_dataset": source.summary(),
        "target_dataset": target.summary(),
        "primary_participants": sorted(np.unique(target.participant_ids[primary_mask]).tolist()),
        "reports": reports,
        "paired_tinyhar_vs_deepconvlstm": _paired_bootstrap(
            reports["TinyHAR-6ch-CPU"], reports["DeepConvLSTM-6ch-CPU"], seed=2026091902
        ),
        "training_records": records,
        "prediction_artifact": {
            "path": prediction_path.name,
            "sha256": sha256_file(prediction_path),
        },
        "runtime_seconds": time.perf_counter() - started,
        "claim_policy": {
            "target_labels_used_for_fit_selection_or_normalization": False,
            "source_validation_selects_epoch": True,
            "final_refit_uses_all_source_participants": True,
            "cuda_protocol_equivalence_claimed": False,
            "state_of_the_art_claim_allowed": False,
        },
    }
    result["record_sha256"] = canonical_json_sha256(result)
    with (output_directory / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    manifest = {
        "schema_version": "1.0.0",
        "files": [
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(output_directory.iterdir())
            if path.is_file()
        ],
    }
    manifest["record_sha256"] = canonical_json_sha256(manifest)
    with (output_directory / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(
        repository_root=args.repository_root.resolve(),
        archive_path=args.archive.resolve(),
        source_csv=args.source_csv.resolve(),
        output_directory=args.output_directory.resolve(),
    )
    print(
        json.dumps(
            {
                name: {
                    "accuracy": report["window_level_diagnostics"]["accuracy"],
                    "macro_f1": report["primary"]["mean_participant_macro_f1"],
                }
                for name, report in result["reports"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
