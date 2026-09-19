"""Bounded SAGE-X standing experiment on paired HARTH accelerometers.

SAGE-X (Standing-Aware Geometry Exchange) is a small, auditable temporal model
whose back tower is trained from lower-back data only at evaluation time.  The
paired thigh stream is used only inside an outer-training fold as a masked
cross-placement reconstruction target.  This is deliberately a finite screen,
not a replacement for the locked InclusiveHAR/HERA evidence.

The script keeps the complete raw-window materialisation local to the run and
does not retain the downloaded archive.  It uses one seed and the retained
five participant folds.  Four arms are fitted: supervised temporal control,
same-sensor masked reconstruction, paired cross-sensor reconstruction, and the
paired model with a standing-focused contrastive margin.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from zipfile import ZipFile

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.experiments.harth_crsp_back import (
    LABEL_MAP,
    SEED,
    SOURCE_RATE_HZ,
    STANDARD_GRAVITY,
    _fold_assignment,
    _physical_runs,
)

CLASS_NAMES = ("sitting", "standing")
PROTOCOL_PATH = "docs/research/HARTH_SAGE_X_V1_PROTOCOL.md"
SOURCE_RECORD = "https://archive.ics.uci.edu/dataset/779/harth"
MAX_TRAIN_ROWS = 12_000
MAX_EPOCHS = 18
BATCH_SIZE = 256
MASK_PROBABILITY = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_raw_windows(archive: Path, window_samples: int = 250) -> tuple[dict[str, Any], dict[str, Any]]:
    """Materialise paired non-overlapping windows without using labels as features."""

    back_windows: list[np.ndarray] = []
    thigh_windows: list[np.ndarray] = []
    labels: list[int] = []
    participants: list[str] = []
    window_ids: list[str] = []
    stats: dict[str, Any] = {
        "member_count": 0,
        "rows_total": 0,
        "physical_run_count": 0,
        "timestamp_boundary_count": 0,
        "label_run_count": 0,
        "window_label_counts": {name: 0 for name in CLASS_NAMES},
    }
    with ZipFile(archive) as source:
        members = sorted(
            (item for item in source.infolist() if item.filename.endswith(".csv")),
            key=lambda item: item.filename,
        )
        stats["member_count"] = len(members)
        for item in members:
            participant = Path(item.filename).stem
            frame = pd.read_csv(
                source.open(item),
                usecols=[
                    "timestamp",
                    "back_x",
                    "back_y",
                    "back_z",
                    "thigh_x",
                    "thigh_y",
                    "thigh_z",
                    "label",
                ],
            )
            stats["rows_total"] += len(frame)
            parsed = pd.to_datetime(frame["timestamp"], errors="coerce")
            timestamp = parsed.astype("int64").to_numpy(dtype=np.float64) / 1.0e9
            timestamp[parsed.isna().to_numpy()] = np.nan
            back = frame[["back_x", "back_y", "back_z"]].to_numpy(dtype=np.float32) * STANDARD_GRAVITY
            thigh = frame[["thigh_x", "thigh_y", "thigh_z"]].to_numpy(dtype=np.float32) * STANDARD_GRAVITY
            values = np.column_stack((back, thigh))
            physical_runs, gap_count = _physical_runs(timestamp, values)
            stats["physical_run_count"] += len(physical_runs)
            stats["timestamp_boundary_count"] += int(gap_count)
            raw_labels = pd.to_numeric(frame["label"], errors="coerce").to_numpy(dtype=np.float64)
            for physical_start, physical_stop in physical_runs:
                selected = np.isin(raw_labels[physical_start:physical_stop], np.asarray(tuple(LABEL_MAP)))
                for local_start, local_stop in _true_runs(selected):
                    absolute_start = physical_start + local_start
                    absolute_stop = physical_start + local_stop
                    segment_labels = raw_labels[absolute_start:absolute_stop]
                    changes = np.r_[True, segment_labels[1:] != segment_labels[:-1]]
                    change_starts = np.flatnonzero(changes)
                    change_stops = np.concatenate((change_starts[1:], np.array([segment_labels.size])))
                    for label_start, label_stop in zip(change_starts, change_stops, strict=True):
                        stats["label_run_count"] += 1
                        code = int(segment_labels[int(label_start)])
                        if code not in LABEL_MAP:
                            continue
                        start = absolute_start + int(label_start)
                        stop = absolute_start + int(label_stop)
                        count = (stop - start) // window_samples
                        for index in range(count):
                            left = start + index * window_samples
                            right = left + window_samples
                            back_window = back[left:right]
                            thigh_window = thigh[left:right]
                            if back_window.shape != (window_samples, 3) or not np.isfinite(back_window).all():
                                continue
                            if thigh_window.shape != (window_samples, 3) or not np.isfinite(thigh_window).all():
                                continue
                            back_windows.append(back_window)
                            thigh_windows.append(thigh_window)
                            label = LABEL_MAP[code]
                            labels.append(label)
                            participants.append(participant)
                            window_ids.append(f"{participant}:{item.filename}:{left}:{right}")
                            stats["window_label_counts"][CLASS_NAMES[label]] += 1
    if not labels:
        raise RuntimeError("HARTH produced no selected windows")
    arrays = {
        "back": np.asarray(back_windows, dtype=np.float32),
        "thigh": np.asarray(thigh_windows, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "participants": np.asarray(participants, dtype=np.str_),
        "window_ids": np.asarray(window_ids, dtype=np.str_),
    }
    stats["participants"] = sorted(np.unique(cast(np.ndarray, arrays["participants"])).tolist())
    stats["participant_count"] = len(stats["participants"])
    stats["window_count"] = len(labels)
    stats["window_samples"] = window_samples
    stats["sampling_rate_hz"] = SOURCE_RATE_HZ
    if not np.isfinite(cast(np.ndarray, arrays["back"])).all() or not np.isfinite(cast(np.ndarray, arrays["thigh"])).all():
        raise RuntimeError("raw materialisation produced non-finite values")
    return arrays, stats


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate((np.array([False]), mask, np.array([False]))).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def _seed_everything(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class TemporalEncoder(nn.Module):
    def __init__(self, input_channels: int = 3, hidden: int = 32, embedding: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, hidden, kernel_size=5, padding=2),
            nn.GroupNorm(4, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, dilation=2, padding=4),
            nn.GroupNorm(4, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden * 2, kernel_size=5, dilation=4, padding=8),
            nn.GroupNorm(8, hidden * 2),
            nn.GELU(),
            nn.Conv1d(hidden * 2, hidden * 2, kernel_size=5, dilation=8, padding=16),
            nn.GroupNorm(8, hidden * 2),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(nn.Linear(hidden * 2, embedding), nn.LayerNorm(embedding), nn.GELU())

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        sequence = self.net(x)
        embedding = self.projection(self.pool(sequence).squeeze(-1))
        return sequence, embedding


class SageModel(nn.Module):
    def __init__(self, arm: str) -> None:
        super().__init__()
        self.arm = arm
        self.back_encoder = TemporalEncoder()
        self.classifier = nn.Linear(64, 2)
        self.reconstruction = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 75))
        self.thigh_encoder = TemporalEncoder() if arm in {
            "paired_csmr",
            "paired_contrastive",
            "paired_distill",
            "paired_distill_contrastive",
        } else None
        self.thigh_classifier = nn.Linear(64, 2) if self.thigh_encoder is not None else None

    def forward(self, back: Tensor, thigh: Tensor | None = None) -> dict[str, Tensor]:
        back_sequence, back_embedding = self.back_encoder(back)
        output: dict[str, Tensor] = {
            "logits": self.classifier(back_embedding),
            "embedding": back_embedding,
        }
        if self.arm == "same_sensor_ssl":
            pooled = F.adaptive_avg_pool1d(back_sequence, 25).transpose(1, 2).reshape(back.shape[0], -1)
            output["reconstruction"] = self.reconstruction(back_embedding)
            output["target"] = pooled
        elif self.arm in {
            "paired_csmr",
            "paired_contrastive",
            "paired_distill",
            "paired_distill_contrastive",
        }:
            if thigh is None:
                return output
            if self.thigh_encoder is None:
                raise ValueError("paired arm requires a configured thigh tower")
            _, thigh_embedding = self.thigh_encoder(thigh)
            pooled = F.adaptive_avg_pool1d(thigh, 25).transpose(1, 2).reshape(back.shape[0], -1)
            output["reconstruction"] = self.reconstruction(back_embedding)
            output["target"] = pooled
            output["thigh_embedding"] = thigh_embedding.detach()
            if self.thigh_classifier is not None:
                output["thigh_logits"] = self.thigh_classifier(thigh_embedding)
        return output


def _participant_weights(labels: np.ndarray, participants: np.ndarray, train_indices: np.ndarray) -> np.ndarray:
    values = np.zeros(train_indices.size, dtype=np.float32)
    selected_labels = labels[train_indices]
    selected_participants = participants[train_indices]
    for participant in np.unique(selected_participants):
        participant_mask = selected_participants == participant
        present_classes = np.unique(selected_labels[participant_mask])
        for class_index in present_classes:
            mask = participant_mask & (selected_labels == class_index)
            values[mask] = 1.0 / (len(present_classes) * int(mask.sum()))
    values /= max(float(values.mean()), 1.0e-12)
    return values


def _standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=(0, 1), keepdims=True)
    scale = train.std(axis=(0, 1), keepdims=True)
    scale = np.where(scale < 1.0e-4, 1.0, scale)
    return (train - mean) / scale, (other - mean) / scale, mean, scale


def _mask_batch(x: Tensor, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    mask = torch.rand(x.shape, generator=generator, device=x.device) < MASK_PROBABILITY
    # Preserve at least one unmasked sample value per row/channel.
    masked = torch.where(mask, torch.zeros_like(x), x)
    return masked, mask


def _contrastive_loss(embedding: Tensor, labels: Tensor) -> Tensor:
    """Standing-focused supervised margin using the hardest opposite in the batch."""

    normalized = F.normalize(embedding, dim=1)
    similarity = normalized @ normalized.T
    losses: list[Tensor] = []
    for index in range(embedding.shape[0]):
        if int(labels[index]) != 1:
            continue
        positives = similarity[index][labels == 1]
        negatives = similarity[index][labels == 0]
        if positives.numel() <= 1 or negatives.numel() == 0:
            continue
        positive = positives[positives < 0.999].mean() if (positives < 0.999).any() else positives.mean()
        negative = negatives.max()
        losses.append(F.relu(0.20 - positive + negative))
    return torch.stack(losses).mean() if losses else embedding.new_zeros(())


def _fit_fold(
    arrays: dict[str, Any],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    arm: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    _seed_everything(seed)
    back_train, back_test, mean, scale = _standardize(arrays["back"][train_indices], arrays["back"][test_indices])
    thigh_train = (arrays["thigh"][train_indices] - mean) / scale
    labels_train = arrays["labels"][train_indices]
    weights = _participant_weights(arrays["labels"], arrays["participants"], train_indices)
    # A deterministic balanced order avoids a rare-class batch disappearing while retaining all rows.
    rng = np.random.default_rng(seed + 100)
    order = np.arange(train_indices.size)
    order = rng.permutation(order)
    model = SageModel(arm).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    generator = torch.Generator(device=DEVICE).manual_seed(seed + 200)
    back_train_tensor = torch.from_numpy(back_train).to(DEVICE).transpose(1, 2)
    thigh_train_tensor = torch.from_numpy(thigh_train).to(DEVICE).transpose(1, 2)
    labels_tensor = torch.from_numpy(labels_train).to(DEVICE)
    weights_tensor = torch.from_numpy(weights).to(DEVICE)
    model.train()
    loss_trace: list[float] = []
    for epoch in range(MAX_EPOCHS):
        epoch_order = np.roll(order, epoch * 97)
        total = 0.0
        count = 0
        for start in range(0, len(epoch_order), BATCH_SIZE):
            batch = torch.from_numpy(epoch_order[start : start + BATCH_SIZE]).to(DEVICE)
            raw_back = back_train_tensor[batch]
            masked_back, _mask = _mask_batch(raw_back, generator)
            model_input = raw_back if model.arm == "supervised_temporal" else masked_back
            output = model(model_input, thigh_train_tensor[batch] if model.thigh_encoder is not None else None)
            ce = F.cross_entropy(output["logits"], labels_tensor[batch], reduction="none")
            loss = (ce * weights_tensor[batch]).mean()
            if model.thigh_encoder is not None:
                reconstruction = F.mse_loss(output["reconstruction"], output["target"], reduction="none").mean(dim=1)
                loss = loss + 0.25 * reconstruction.mean()
                if "thigh_embedding" in output:
                    loss = loss + 0.10 * (1.0 - F.cosine_similarity(output["embedding"], output["thigh_embedding"], dim=1)).mean()
                if model.arm in {"paired_distill", "paired_distill_contrastive"}:
                    teacher_logits = output["thigh_logits"]
                    teacher_loss = F.cross_entropy(teacher_logits, labels_tensor[batch])
                    temperature = 2.0
                    distillation = F.kl_div(
                        F.log_softmax(output["logits"] / temperature, dim=1),
                        F.softmax(teacher_logits.detach() / temperature, dim=1),
                        reduction="batchmean",
                    ) * temperature**2
                    loss = loss + 0.50 * teacher_loss + 0.25 * distillation
            if model.arm == "same_sensor_ssl":
                target = F.adaptive_avg_pool1d(raw_back, 25).transpose(1, 2).reshape(raw_back.shape[0], -1)
                reconstruction = F.mse_loss(output["reconstruction"], target, reduction="none").mean(dim=1)
                loss = loss + 0.25 * reconstruction.mean()
            if model.arm in {"paired_contrastive", "paired_distill_contrastive"}:
                loss = loss + 0.15 * _contrastive_loss(output["embedding"], labels_tensor[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(batch)
            count += len(batch)
        loss_trace.append(total / max(count, 1))
    model.eval()
    with torch.inference_mode():
        probabilities: list[np.ndarray] = []
        for start in range(0, len(test_indices), BATCH_SIZE):
            batch_back = torch.from_numpy(back_test[start : start + BATCH_SIZE]).to(DEVICE).transpose(1, 2)
            output = model(batch_back, None)
            probabilities.append(torch.softmax(output["logits"], dim=1).cpu().numpy().astype(np.float64))
    report = classification_report(
        arrays["labels"][test_indices],
        np.vstack(probabilities),
        arrays["participants"][test_indices].tolist(),
        class_names=CLASS_NAMES,
    )
    return np.vstack(probabilities), {
        "arm": arm,
        "seed": seed,
        "device": str(DEVICE),
        "epochs": MAX_EPOCHS,
        "train_windows": len(train_indices),
        "test_windows": len(test_indices),
        "feature_channels": 3,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "loss_trace": loss_trace,
        "normalization_mean": mean.reshape(-1).astype(float).tolist(),
        "normalization_scale": scale.reshape(-1).astype(float).tolist(),
        "report": report,
    }


def _aggregate_reports(reports: list[dict[str, Any]], arrays: dict[str, np.ndarray], test_indices: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    report = classification_report(
        arrays["labels"][test_indices],
        probability,
        arrays["participants"][test_indices].tolist(),
        class_names=CLASS_NAMES,
    )
    return {"report": report, "folds": reports}


def _participant_contrast(
    control_report: dict[str, Any], candidate_report: dict[str, Any]
) -> dict[str, Any]:
    control = {str(row["participant_id"]): float(row["macro_f1"]) for row in control_report["participants"]}
    candidate = {str(row["participant_id"]): float(row["macro_f1"]) for row in candidate_report["participants"]}
    common = sorted(set(control) & set(candidate))
    differences = np.asarray([candidate[item] - control[item] for item in common], dtype=np.float64)
    rng = np.random.default_rng(1729)
    draws = rng.choice(differences, size=(10_000, differences.size), replace=True).mean(axis=1)
    return {
        "mean": float(differences.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "wins": int(np.sum(differences > 0.0)),
        "harms": int(np.sum(differences < 0.0)),
        "ties": int(np.sum(differences == 0.0)),
        "worst_difference": float(differences.min()),
        "participants": {item: float(value) for item, value in zip(common, differences, strict=True)},
    }


def _retained_rich_control() -> dict[str, Any] | None:
    root = Path(__file__).resolve().parents[3]
    path = root / ".audit" / "harth_crsp_back_20260918" / "run-005" / "FINAL_RESULT.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for method in payload.get("methods", []):
        if method.get("arm") == "C_rich_250_rf":
            return cast(dict[str, Any], method.get("report"))
    return None


def run(
    archive: Path,
    output: Path,
    arms: tuple[str, ...] = (
        "supervised_temporal",
        "same_sensor_ssl",
        "paired_csmr",
        "paired_contrastive",
    ),
) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    print("SAGE-X: materializing HARTH windows", flush=True)
    arrays, data_stats = _load_raw_windows(archive)
    print(f"SAGE-X: materialized {arrays['labels'].size} windows", flush=True)
    participants = sorted(np.unique(arrays["participants"]).tolist())
    assignment = _fold_assignment(participants)
    all_results: dict[str, Any] = {}
    for arm in arms:
        print(f"SAGE-X: arm={arm}", flush=True)
        probabilities = np.zeros((arrays["labels"].size, 2), dtype=np.float64)
        fold_records: list[dict[str, Any]] = []
        for fold in range(5):
            print(f"SAGE-X: arm={arm} fold={fold}", flush=True)
            test_mask = np.asarray([assignment[item] == fold for item in arrays["participants"]])
            train_indices = np.flatnonzero(~test_mask)
            test_indices = np.flatnonzero(test_mask)
            if train_indices.size > MAX_TRAIN_ROWS:
                # Stratified deterministic cap is a compute bound, never applied to held-out rows.
                rng = np.random.default_rng(SEED + fold)
                selected: list[int] = []
                for class_index in (0, 1):
                    candidates = train_indices[arrays["labels"][train_indices] == class_index]
                    take = min(len(candidates), MAX_TRAIN_ROWS // 2)
                    selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
                train_indices = np.asarray(sorted(selected), dtype=np.int64)
            prediction, record = _fit_fold(arrays, train_indices, test_indices, arm, SEED + fold)
            probabilities[test_indices] = prediction
            record["fold"] = fold
            record["evaluation_participants"] = [item for item in participants if assignment[item] == fold]
            fold_records.append(record)
        aggregate = _aggregate_reports(fold_records, arrays, np.arange(arrays["labels"].size), probabilities)
        np.savez_compressed(
            output / f"{arm}_predictions.npz",
            window_ids=arrays["window_ids"],
            participants=arrays["participants"],
            labels=arrays["labels"],
            probabilities=probabilities,
        )
        all_results[arm] = aggregate
    controls = {
        "compact_rf": {"participant_macro_f1": 0.5576350826727572, "standing_recall": 0.38676740861752096},
        "rich_rf": {"participant_macro_f1": 0.5903696635673075, "standing_recall": 0.3741339491916859},
    }
    retained_control = _retained_rich_control()
    for arm in arms:
        candidate_report = all_results[arm]["report"]
        all_results[arm]["contrast_vs_rich_rf"] = {
            "mean": float(candidate_report["primary"]["mean_participant_macro_f1"] - controls["rich_rf"]["participant_macro_f1"]),
            "standing_recall_gain": float(
                candidate_report["window_level_diagnostics"]["per_class_recall"]["standing"]
                - controls["rich_rf"]["standing_recall"]
            ),
        }
        if retained_control is not None:
            all_results[arm]["contrast_vs_rich_rf"].update(
                _participant_contrast(retained_control, candidate_report)
            )
    elapsed = time.perf_counter() - started
    payload = {
        "record_kind": "harth_sage_x_v1",
        "status": "completed_exploratory_external_development",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol": PROTOCOL_PATH,
        "source_record": SOURCE_RECORD,
        "source_sha256": _sha256(archive),
        "contract": {
            "window_samples": 250,
            "sampling_rate_hz": SOURCE_RATE_HZ,
            "classes": list(CLASS_NAMES),
            "label_map": {str(key): value for key, value in LABEL_MAP.items()},
            "participant_exclusive_folds": 5,
            "fold_seed": SEED,
            "inference_sensor": "lower_back_only",
            "paired_thigh_usage": "outer_training_only_reconstruction_and_alignment",
            "torch_version": torch.__version__,
            "device": str(DEVICE),
        },
        "data_stats": data_stats,
        "fold_assignment": assignment,
        "controls": controls,
        "arms": all_results,
        "runtime_seconds": elapsed,
        "promotion_gate": {
            "candidate_vs_rich_rf_macro_f1_gain_min": 0.05,
            "standing_recall_gain_min": 0.10,
            "standing_f1_gain_min": 0.10,
            "sitting_recall_loss_max": 0.02,
            "participant_wins_min": 16,
            "participant_harm_floor": -0.05,
            "paired_bootstrap": "not recomputed in this initial neural screen; see analysis artifact",
        },
    }
    (output / "RESULT.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    if len(sys.argv) not in {3, 4}:
        raise SystemExit("usage: harth_sage_x.py HARTH_ZIP OUTPUT_DIR [comma-separated-arms]")
    if len(sys.argv) == 3:
        run(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        run(Path(sys.argv[1]), Path(sys.argv[2]), arms=tuple(sys.argv[3].split(",")))
