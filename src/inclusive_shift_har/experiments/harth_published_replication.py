"""Diagnostic reproduction of the published HARTH traditional-model protocol.

The lane is deliberately separate from the locked InclusiveHAR/HERA protocol and
from the recent binary sitting/standing screens.  It reproduces the HARTH paper's
12-label, five-second feature contract with leave-one-subject-out evaluation, then
evaluates the existing fixed rich-feature readout under the same subject folds.
The output is exploratory protocol-parity evidence, not a publication claim.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from zipfile import ZipFile

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import scipy.signal  # type: ignore[import-untyped]
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
from sklearn.metrics import confusion_matrix  # type: ignore[import-untyped]
from sklearn.preprocessing import MinMaxScaler  # type: ignore[import-untyped]
from sklearn.svm import SVC  # type: ignore[import-untyped]

from inclusive_shift_har.experiments.harth_crsp_back import _rich_features

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - runtime dependency is validated by the runner
    XGBClassifier = None  # type: ignore[assignment,misc]


SOURCE_URL = "https://archive.ics.uci.edu/static/public/779/harth.zip"
SOURCE_RECORD = "https://archive.ics.uci.edu/dataset/779/harth"
PAPER_URL = "https://doi.org/10.3390/s21237853"
PROTOCOL_PATH = "docs/research/HARTH_PUBLISHED_REPLICATION_V1_PROTOCOL.md"
FOLD_COUNT = 22
SAMPLE_RATE_HZ = 50.0
WINDOW_SAMPLES = 250
PUBLISHED_LABELS = (1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140)
PUBLISHED_NAMES = (
    "walking",
    "running",
    "shuffling",
    "stairs_ascending",
    "stairs_descending",
    "standing",
    "sitting",
    "lying",
    "cycling_sit",
    "cycling_stand",
    "cycling_sit_inactive",
    "cycling_stand_inactive",
)
RAW_TO_INDEX = {label: index for index, label in enumerate(PUBLISHED_LABELS)}
MERGED_NAMES = (
    "walking",
    "running",
    "standing",
    "stairs_ascending",
    "stairs_descending",
    "sitting",
    "lying",
    "cycling_sit",
    "cycling_stand",
)
MERGED_INDEX = {
    1: 0,
    2: 1,
    3: 2,
    6: 2,
    140: 2,
    4: 3,
    5: 4,
    7: 5,
    130: 5,
    8: 6,
    13: 7,
    14: 8,
}


@dataclass
class SubjectData:
    participant: str
    complete_features: dict[str, np.ndarray]
    complete_labels: np.ndarray
    tail_features: dict[str, np.ndarray]
    raw_labels: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_cv(data: np.ndarray) -> np.ndarray:
    mean = data.mean(axis=1)
    std = data.std(axis=1)
    return np.asarray(
        np.nan_to_num(std / np.where(np.abs(mean) > 1.0e-12, mean, 1.0), nan=0.0),
        dtype=np.float64,
    )


def _official_features(windows: np.ndarray) -> np.ndarray:
    """Compute the official repository's 161-feature traditional view."""

    if windows.ndim != 3 or windows.shape[1:] != (WINDOW_SAMPLES, 6):
        raise ValueError(f"unexpected window shape: {windows.shape}")
    magnitudes = np.stack(
        (
            np.linalg.norm(windows[:, :, :3], axis=2),
            np.linalg.norm(windows[:, :, 3:], axis=2),
        ),
        axis=2,
    )
    raw = np.concatenate((windows, magnitudes), axis=2)
    butter_b, butter_a = scipy.signal.butter(4, 1.0, fs=SAMPLE_RATE_HZ, btype="lowpass")
    gravity = scipy.signal.filtfilt(butter_b, butter_a, raw, axis=1)
    movement = raw - gravity
    gravity_sorted = np.sort(gravity, axis=1)
    blocks: list[np.ndarray] = []

    blocks.extend((gravity.mean(axis=1), _safe_cv(gravity), gravity.std(axis=1)))
    for quantile in (0.0, 0.25, 0.50, 0.75, 1.0):
        position = min(int(quantile * WINDOW_SAMPLES), WINDOW_SAMPLES - 1)
        blocks.append(gravity_sorted[:, position, :])

    centered = movement - movement.mean(axis=1, keepdims=True)
    std = movement.std(axis=1)
    skew = np.mean(
        (centered / np.where(std[:, None, :] > 1.0e-12, std[:, None, :], 1.0)) ** 3, axis=1
    )
    kurtosis = (
        np.mean((centered / np.where(std[:, None, :] > 1.0e-12, std[:, None, :], 1.0)) ** 4, axis=1)
        - 3.0
    )
    blocks.extend((np.nan_to_num(skew), np.nan_to_num(kurtosis), np.sum(movement**2, axis=1)))

    correlation: list[np.ndarray] = []
    for left in range(8):
        for right in range(left + 1, 8):
            if (left >= 6) != (right >= 6):
                continue
            left_values = movement[:, :, left]
            right_values = movement[:, :, right]
            numerator = (left_values * right_values).mean(axis=1) - left_values.mean(
                axis=1
            ) * right_values.mean(axis=1)
            denominator = np.maximum(left_values.std(axis=1) * right_values.std(axis=1), 1.0e-6)
            correlation.append((numerator / denominator)[:, None])
    blocks.append(np.concatenate(correlation, axis=1))

    cross_sensor_means = []
    for left in range(3):
        for right in range(3, 6):
            cross_sensor_means.append(
                ((movement[:, :, left] + movement[:, :, right]) / 2.0).mean(axis=1)[:, None]
            )
    blocks.append(np.concatenate(cross_sensor_means, axis=1))

    fft_amplitude = np.abs(np.fft.rfft(movement, axis=1))
    fft_frequency = np.fft.rfftfreq(WINDOW_SAMPLES)
    total_amplitude = np.maximum(fft_amplitude.sum(axis=1), 1.0e-12)
    blocks.extend(
        (
            fft_amplitude.mean(axis=1),
            fft_frequency[fft_amplitude.argmax(axis=1)],
            fft_amplitude.max(axis=1),
            np.sum(fft_amplitude**2, axis=1),
            fft_amplitude.std(axis=1),
            np.sum(fft_frequency[None, :, None] * fft_amplitude, axis=1) / total_amplitude,
        )
    )
    result = np.concatenate(blocks, axis=1).astype(np.float64, copy=False)
    if result.shape[1] != 161 or not np.isfinite(result).all():
        raise RuntimeError(f"official feature contract failed: {result.shape}")
    return np.asarray(result, dtype=np.float64)


def _filtered_values(values: np.ndarray) -> np.ndarray:
    b, a = scipy.signal.butter(4, 20.0, fs=SAMPLE_RATE_HZ, btype="lowpass")
    return np.asarray(scipy.signal.filtfilt(b, a, values, axis=0), dtype=np.float64)


def _window_majority(labels: np.ndarray) -> np.ndarray:
    labels = labels.reshape(-1, WINDOW_SAMPLES)
    output = np.empty(labels.shape[0], dtype=np.int64)
    for index, row in enumerate(labels):
        counts = np.bincount(row, minlength=len(PUBLISHED_LABELS))
        output[index] = int(np.argmax(counts))
    return output


def _load_subjects(archive: Path) -> tuple[list[SubjectData], dict[str, Any]]:
    subjects: list[SubjectData] = []
    audit: dict[str, Any] = {
        "member_count": 0,
        "rows_total": 0,
        "window_count": 0,
        "label_counts": {name: 0 for name in PUBLISHED_NAMES},
    }
    with ZipFile(archive) as source:
        members = sorted(
            (item for item in source.infolist() if item.filename.endswith(".csv")),
            key=lambda item: item.filename,
        )
        audit["member_count"] = len(members)
        for member in members:
            participant = Path(member.filename).stem
            print(
                json.dumps({"stage": "materializing_participant", "participant": participant}),
                flush=True,
            )
            frame = pd.read_csv(
                source.open(member),
                usecols=["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z", "label"],
            )
            audit["rows_total"] += len(frame)
            raw_codes = pd.to_numeric(frame["label"], errors="coerce").to_numpy(dtype=np.int64)
            if not np.isin(raw_codes, np.asarray(PUBLISHED_LABELS)).all():
                raise ValueError(f"unexpected labels in {participant}")
            values = frame[
                ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]
            ].to_numpy(dtype=np.float64)
            filtered = _filtered_values(values)
            complete_count = filtered.shape[0] // WINDOW_SAMPLES
            complete = filtered[: complete_count * WINDOW_SAMPLES].reshape(
                complete_count, WINDOW_SAMPLES, 6
            )
            complete_labels = _window_majority(
                np.asarray(
                    [
                        RAW_TO_INDEX[int(code)]
                        for code in raw_codes[: complete_count * WINDOW_SAMPLES]
                    ],
                    dtype=np.int64,
                )
            )
            feature_views = {
                "published_161": _official_features(complete),
                "ours_back_rich_161": np.asarray(
                    [_rich_features(window[:, :3]) for window in complete], dtype=np.float64
                ),
                "ours_thigh_rich_161": np.asarray(
                    [_rich_features(window[:, 3:]) for window in complete], dtype=np.float64
                ),
            }
            feature_views["ours_fused_rich_322"] = np.column_stack(
                (feature_views["ours_back_rich_161"], feature_views["ours_thigh_rich_161"])
            )
            tail_features: dict[str, np.ndarray] = {}
            if complete_count == 0:
                raise ValueError(f"participant {participant} has no complete windows")
            tail_length = filtered.shape[0] - complete_count * WINDOW_SAMPLES
            if tail_length:
                tail = np.zeros((WINDOW_SAMPLES, 6), dtype=np.float64)
                tail[: min(WINDOW_SAMPLES, filtered.shape[0])] = filtered[-WINDOW_SAMPLES:]
                tail_features["published_161"] = _official_features(tail[None, ...])
                tail_features["ours_back_rich_161"] = _rich_features(tail[:, :3])[None, :]
                tail_features["ours_thigh_rich_161"] = _rich_features(tail[:, 3:])[None, :]
                tail_features["ours_fused_rich_322"] = np.column_stack(
                    (tail_features["ours_back_rich_161"], tail_features["ours_thigh_rich_161"])
                )
            subjects.append(
                SubjectData(
                    participant,
                    feature_views,
                    complete_labels,
                    tail_features,
                    np.asarray([RAW_TO_INDEX[int(code)] for code in raw_codes], dtype=np.int64),
                )
            )
            audit["window_count"] += int(complete_count)
            counts = np.bincount(complete_labels, minlength=len(PUBLISHED_LABELS))
            for index, name in enumerate(PUBLISHED_NAMES):
                audit["label_counts"][name] += int(counts[index])
            print(
                json.dumps(
                    {
                        "stage": "materialized_participant",
                        "participant": participant,
                        "windows": int(complete_count),
                    }
                ),
                flush=True,
            )
    audit["participant_count"] = len(subjects)
    audit["participants"] = [item.participant for item in subjects]
    return subjects, audit


def _fit_model(name: str, x: np.ndarray, y: np.ndarray) -> Any:
    if name == "published_svm":
        return SVC(
            C=10.0,
            kernel="rbf",
            gamma="scale",
            class_weight=None,
            probability=False,
            cache_size=1024,
        )
    if name == "published_rf":
        return RandomForestClassifier(
            n_estimators=80,
            min_samples_split=10,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            bootstrap=True,
            n_jobs=4,
            random_state=11,
        )
    if name == "published_xgb":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is unavailable")
        return XGBClassifier(
            n_estimators=1024,
            max_depth=3,
            learning_rate=0.1,
            reg_lambda=1.0,
            reg_alpha=0.0,
            gamma=0.0,
            objective="multi:softprob",
            eval_metric="merror",
            tree_method="hist",
            n_jobs=4,
            random_state=11,
            verbosity=0,
        )
    if name == "ours_back_rich_rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            bootstrap=True,
            n_jobs=4,
            random_state=11,
        )
    if name == "ours_thigh_rich_rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            bootstrap=True,
            n_jobs=4,
            random_state=11,
        )
    if name == "ours_fused_rich_rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            bootstrap=True,
            n_jobs=4,
            random_state=11,
        )
    raise ValueError(name)


def _metrics(confusion: np.ndarray, names: tuple[str, ...]) -> dict[str, Any]:
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion).astype(np.float64)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return {
        "accuracy": float(true_positive.sum() / max(confusion.sum(), 1)),
        "macro_f1": float(f1.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(names)
        },
        "confusion_matrix": confusion.astype(int).tolist(),
    }


def _aggregate_predictions(
    subject: SubjectData, predictions: np.ndarray, tail_prediction: int | None
) -> tuple[np.ndarray, np.ndarray]:
    full_windows = np.repeat(predictions.astype(np.int64), WINDOW_SAMPLES)
    complete_length = min(full_windows.size, subject.raw_labels.size)
    output = full_windows[:complete_length]
    if complete_length < subject.raw_labels.size:
        if tail_prediction is None:
            raise ValueError("missing tail prediction")
        output = np.concatenate(
            (
                output,
                np.full(subject.raw_labels.size - complete_length, tail_prediction, dtype=np.int64),
            )
        )
    return subject.raw_labels, output


def _run_model(
    name: str, subjects: list[SubjectData]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    started = time.perf_counter()
    print(json.dumps({"stage": "model_started", "model": name}), flush=True)
    windows_true: list[np.ndarray] = []
    windows_pred: list[np.ndarray] = []
    pooled_confusion = np.zeros((len(PUBLISHED_LABELS), len(PUBLISHED_LABELS)), dtype=np.int64)
    participant_scores: list[float] = []
    view = {
        "published_svm": "published_161",
        "published_rf": "published_161",
        "published_xgb": "published_161",
        "ours_back_rich_rf": "ours_back_rich_161",
        "ours_thigh_rich_rf": "ours_thigh_rich_161",
        "ours_fused_rich_rf": "ours_fused_rich_322",
    }[name]
    sample_confusion = pooled_confusion.copy()
    for test_index, test_subject in enumerate(subjects):
        print(
            json.dumps(
                {
                    "stage": "fold_started",
                    "model": name,
                    "fold": test_index + 1,
                    "test_participant": test_subject.participant,
                }
            ),
            flush=True,
        )
        train_subjects = [subject for index, subject in enumerate(subjects) if index != test_index]
        train_x = np.concatenate(
            [subject.complete_features[view] for subject in train_subjects], axis=0
        )
        train_y = np.concatenate([subject.complete_labels for subject in train_subjects], axis=0)
        test_x = test_subject.complete_features[view]
        scaler = None
        if name == "published_svm":
            scaler = MinMaxScaler().fit(train_x)
            train_x = scaler.transform(train_x)
            test_x = scaler.transform(test_x)
        model = _fit_model(name, train_x, train_y)
        model.fit(train_x, train_y)
        test_prediction = np.asarray(model.predict(test_x), dtype=np.int64)
        tail_prediction: int | None = None
        if test_subject.tail_features:
            tail_x = test_subject.tail_features[view]
            if scaler is not None:
                tail_x = scaler.transform(tail_x)
            tail_prediction = int(model.predict(tail_x)[0])
        windows_true.append(test_subject.complete_labels)
        windows_pred.append(test_prediction)
        pooled_confusion += confusion_matrix(
            test_subject.complete_labels, test_prediction, labels=np.arange(len(PUBLISHED_LABELS))
        )
        raw_true, raw_prediction = _aggregate_predictions(
            test_subject, test_prediction, tail_prediction
        )
        sample_confusion += confusion_matrix(
            raw_true, raw_prediction, labels=np.arange(len(PUBLISHED_LABELS))
        )
        participant_confusion = confusion_matrix(
            raw_true, raw_prediction, labels=np.arange(len(PUBLISHED_LABELS))
        )
        participant_scores.append(_metrics(participant_confusion, PUBLISHED_NAMES)["macro_f1"])
        print(
            json.dumps(
                {
                    "stage": "fold_finished",
                    "model": name,
                    "fold": test_index + 1,
                    "test_participant": test_subject.participant,
                }
            ),
            flush=True,
        )

    window_metrics = _metrics(pooled_confusion, PUBLISHED_NAMES)
    sample_metrics = _metrics(sample_confusion, PUBLISHED_NAMES)
    merge = np.zeros((len(MERGED_NAMES), len(MERGED_NAMES)), dtype=np.int64)
    for true_index, row in enumerate(sample_confusion):
        for predicted_index, value in enumerate(row):
            true_code = PUBLISHED_LABELS[true_index]
            pred_code = PUBLISHED_LABELS[predicted_index]
            merge[MERGED_INDEX[true_code], MERGED_INDEX[pred_code]] += int(value)
    result = {
        "method": name,
        "view": view,
        "published_protocol_sample_metrics_12": sample_metrics,
        "published_protocol_sample_metrics_9_merged": _metrics(merge, MERGED_NAMES),
        "window_metrics_12_secondary": window_metrics,
        "participant_macro_f1_12_secondary": {
            "mean": float(np.mean(participant_scores)),
            "std": float(np.std(participant_scores, ddof=1)),
            "min": float(np.min(participant_scores)),
            "max": float(np.max(participant_scores)),
            "values": [float(value) for value in participant_scores],
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    predictions = {
        "window_true": np.concatenate(windows_true),
        "window_pred": np.concatenate(windows_pred),
    }
    print(
        json.dumps(
            {"stage": "model_finished", "model": name, "runtime_seconds": result["runtime_seconds"]}
        ),
        flush=True,
    )
    return result, predictions


def run(archive: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"create-only output exists: {output}")
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    try:
        subjects, audit = _load_subjects(archive)
        if len(subjects) != FOLD_COUNT:
            raise RuntimeError(
                f"HARTH published LOSO requires 22 participants, observed {len(subjects)}"
            )
        model_names = (
            "published_svm",
            "published_rf",
            "published_xgb",
            "ours_back_rich_rf",
            "ours_thigh_rich_rf",
            "ours_fused_rich_rf",
        )
        results: list[dict[str, Any]] = []
        prediction_arrays: dict[str, np.ndarray] = {}
        for name in model_names:
            result, predictions = _run_model(name, subjects)
            results.append(result)
            prediction_arrays[f"{name}_window_true"] = predictions["window_true"]
            prediction_arrays[f"{name}_window_pred"] = predictions["window_pred"]
            np.savez_compressed(output / f"{name}_PREDICTIONS.npz", **cast(Any, prediction_arrays))
            (output / "PROGRESS.json").write_text(
                json.dumps(
                    {
                        "status": "partial",
                        "completed_models": [item["method"] for item in results],
                        "results": results,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    except Exception as error:
        (output / "FAILED_ATTEMPT.json").write_text(
            json.dumps(
                {
                    "status": "FAILED_PRESERVED",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    payload = {
        "schema_version": "1.0.0",
        "record_kind": "harth_published_protocol_replication",
        "status": "completed_diagnostic_protocol_parity",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "protocol_path": PROTOCOL_PATH,
        "published_reference": {
            "paper_url": PAPER_URL,
            "source_url": SOURCE_URL,
            "source_record": SOURCE_RECORD,
        },
        "contract": {
            "participants": FOLD_COUNT,
            "loso": True,
            "sampling_rate_hz": SAMPLE_RATE_HZ,
            "window_samples": WINDOW_SAMPLES,
            "window_overlap_samples": 0,
            "label_count": len(PUBLISHED_LABELS),
            "feature_count": 161,
            "feature_scale": "train_only_minmax_for_svm; official_rf_xgb_scaling_null",
            "majority_label_windowing": True,
            "twenty_hz_lowpass": True,
            "one_hz_gravity_filter": True,
        },
        "data_audit": audit,
        "results": results,
        "runtime_seconds": float(time.perf_counter() - started),
        "claim_boundary": "Diagnostic reproduction; not a universal state-of-the-art or publication-confirmation claim.",
    }
    payload["source"] = {
        "path": str(archive.resolve()),
        "sha256": _sha256(archive),
        "size_bytes": archive.stat().st_size,
        "raw_archive_retained": False,
    }
    payload["result_payload_sha256_before_serialization"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    (output / "SOURCE_RECEIPT.json").write_text(
        json.dumps(payload["source"], indent=2) + "\n", encoding="utf-8"
    )
    (output / "RESULT.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output / "PREDICTIONS.npz").write_bytes(
        (output / "ours_fused_rich_rf_PREDICTIONS.npz").read_bytes()
    )
    return payload


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: harth_published_replication.py HARTH_ZIP OUTPUT_DIR")
    result = run(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
    print(
        json.dumps(
            {
                "status": result["status"],
                "runtime_seconds": result["runtime_seconds"],
                "output": str(Path(sys.argv[2]).resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
