"""Bounded HARTH CRSP-Back screening experiment.

This module implements the first offline stage of the cross-sensor representation
with selective physics (CRSP-Back) hypothesis.  It keeps the HARTH source contract
separate from the locked InclusiveHAR/HERA experiments and uses the thigh stream
only as a training-time reconstruction target in the paired arm.  Inference for
all reported back arms uses lower-back acceleration only.

The implementation is intentionally CPU-only and deterministic.  In environments
without the optional neural-training dependency, the temporal encoder is represented
by a fixed multi-scale temporal feature map followed by the same participant-aware
forest readout; this keeps the information ablation executable without silently
changing the source contract.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import sklearn  # type: ignore[import-untyped]
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from inclusive_shift_har.evaluation.metrics import classification_report

SOURCE_URL = "https://archive.ics.uci.edu/static/public/779/harth.zip"
SOURCE_RECORD = "https://archive.ics.uci.edu/dataset/779/harth"
SOURCE_DOI = "10.24432/C5NC90"
PROTOCOL_PATH = "docs/research/HARTH_CRSP_BACK_V1_PROTOCOL.md"
STANDARD_GRAVITY = 9.80665
SOURCE_RATE_HZ = 50.0
SEED = 11
FOLD_COUNT = 5
CLASS_NAMES = ("sitting", "standing")
LABEL_MAP = {7: 0, 6: 1}
MAX_GAP_SECONDS = 3.0 / SOURCE_RATE_HZ


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate((np.array([False]), mask, np.array([False]))).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def _physical_runs(timestamps: np.ndarray, values: np.ndarray) -> tuple[list[tuple[int, int]], int]:
    finite = np.isfinite(timestamps) & np.isfinite(values).all(axis=1)
    output: list[tuple[int, int]] = []
    gap_count = 0
    for finite_start, finite_stop in _true_runs(finite):
        local = timestamps[finite_start:finite_stop]
        boundary = np.ones(local.size, dtype=np.bool_)
        if local.size > 1:
            delta = np.diff(local)
            gap_boundary = (delta <= 0.0) | (delta > MAX_GAP_SECONDS)
            gap_count += int(gap_boundary.sum())
            boundary[1:] = gap_boundary
        starts = np.flatnonzero(boundary)
        stops = np.concatenate((starts[1:], np.array([local.size])))
        output.extend(
            (finite_start + int(start), finite_start + int(stop))
            for start, stop in zip(starts, stops, strict=True)
        )
    return output, gap_count


def _safe_skew(values: np.ndarray) -> float:
    centered = values - float(values.mean())
    scale = float(values.std())
    return 0.0 if scale <= 1.0e-12 else float(np.mean((centered / scale) ** 3))


def _safe_kurtosis(values: np.ndarray) -> float:
    centered = values - float(values.mean())
    scale = float(values.std())
    return 0.0 if scale <= 1.0e-12 else float(np.mean((centered / scale) ** 4) - 3.0)


def _base_features(window: np.ndarray) -> np.ndarray:
    """The fixed 24-dimensional view used by the retained run-004 control."""

    mean = window.mean(axis=0)
    std = window.std(axis=0, ddof=0)
    minimum = window.min(axis=0)
    maximum = window.max(axis=0)
    rms = np.sqrt(np.mean(window**2, axis=0))
    mean_abs_difference = np.mean(np.abs(np.diff(window, axis=0)), axis=0)
    norm = np.linalg.norm(window, axis=1)
    mean_norm = float(norm.mean())
    std_norm = float(norm.std(ddof=0))
    dynamic_rms = float(np.sqrt(np.mean(np.sum((window - mean) ** 2, axis=1))))
    direction = mean / max(float(np.linalg.norm(mean)), 1.0e-12)
    result = np.concatenate(
        (
            mean,
            std,
            minimum,
            maximum,
            rms,
            mean_abs_difference,
            np.array([mean_norm, std_norm, dynamic_rms]),
            direction,
        )
    )
    if result.shape != (24,) or not np.isfinite(result).all():
        raise RuntimeError("invalid base feature vector")
    return result  # type: ignore[no-any-return]


def _rich_features(window: np.ndarray) -> np.ndarray:
    """A fixed 161-feature, orientation/temporal feature family.

    It is a transparent reproduction of the established HARTH-style feature
    categories (axis statistics, norms, correlations, covariance/eigenvalues,
    spectrum, subwindows and lag structure), with no labels or participant IDs.
    """

    out: list[float] = []
    n = window.shape[0]
    centered = window - window.mean(axis=0, keepdims=True)
    for axis in range(3):
        values = window[:, axis]
        diff = np.diff(values)
        second = np.diff(values, n=2)
        mean = float(values.mean())
        std = float(values.std())
        out.extend(
            [
                mean,
                std,
                float(values.min()),
                float(values.max()),
                float(np.median(values)),
                float(np.quantile(values, 0.05)),
                float(np.quantile(values, 0.25)),
                float(np.quantile(values, 0.75)),
                float(np.quantile(values, 0.95)),
                float(np.sqrt(np.mean(values**2))),
                float(np.mean(np.abs(values - mean))),
                float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
                float(values.max() - values.min()),
                float(np.mean(np.abs(diff))),
                float(diff.std()),
                float(np.mean(np.abs(second))),
                _safe_skew(values),
                _safe_kurtosis(values),
                float(np.mean(values**2)),
                float(np.count_nonzero(np.diff(np.signbit(values - mean)))),
            ]
        )
    norm = np.linalg.norm(window, axis=1)
    norm_diff = np.diff(norm)
    out.extend(
        [
            float(norm.mean()),
            float(norm.std()),
            float(norm.min()),
            float(norm.max()),
            float(np.median(norm)),
            float(np.quantile(norm, 0.05)),
            float(np.quantile(norm, 0.95)),
            float(np.sqrt(np.mean(norm**2))),
            float(np.mean(np.abs(norm - norm.mean()))),
            float(np.quantile(norm, 0.75) - np.quantile(norm, 0.25)),
            float(np.mean(np.abs(norm_diff))),
            float(norm_diff.std()),
            float(np.mean(norm**2)),
            _safe_skew(norm),
            _safe_kurtosis(norm),
        ]
    )
    axis_stds = window.std(axis=0)
    correlation = np.eye(3, dtype=np.float64)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        if axis_stds[i] > 1.0e-12 and axis_stds[j] > 1.0e-12:
            value = float(np.corrcoef(window[:, i], window[:, j])[0, 1])
            if np.isfinite(value):
                correlation[i, j] = correlation[j, i] = value
    covariance = np.cov(window, rowvar=False, ddof=0)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        out.append(float(correlation[i, j]) if np.isfinite(correlation[i, j]) else 0.0)
    mean = window.mean(axis=0)
    out.extend((mean / max(float(np.linalg.norm(mean)), 1.0e-12)).tolist())
    out.extend(
        [
            float(covariance[0, 0]),
            float(covariance[1, 1]),
            float(covariance[2, 2]),
            float(covariance[0, 1]),
            float(covariance[0, 2]),
            float(covariance[1, 2]),
        ]
    )
    out.extend(np.linalg.eigvalsh(covariance).tolist())
    frequencies = np.fft.rfftfreq(n, d=1.0 / SOURCE_RATE_HZ)
    for axis in range(3):
        power = np.abs(np.fft.rfft(centered[:, axis])) ** 2
        positive = power[1:]
        positive_freq = frequencies[1:]
        total = float(positive.sum())
        normalized = positive / max(total, 1.0e-12)
        entropy = float(-np.sum(normalized * np.log(normalized + 1.0e-12)))
        peak = int(np.argmax(positive)) if positive.size else 0
        low_stop = max(1, int(1.0 * n / SOURCE_RATE_HZ))
        mid_stop = max(low_stop + 1, int(3.0 * n / SOURCE_RATE_HZ))
        out.extend(
            [
                float(positive[:low_stop].sum() / max(total, 1.0e-12)),
                float(positive[low_stop:mid_stop].sum() / max(total, 1.0e-12)),
                entropy,
                float(positive_freq[peak]) if positive.size else 0.0,
                float(np.sum(positive_freq * normalized)) if positive.size else 0.0,
            ]
        )
    segments = np.array_split(window, 4, axis=0)
    for segment in segments:
        segment_mean = segment.mean(axis=0)
        out.extend(segment_mean.tolist())
        out.extend(segment.std(axis=0).tolist())
        segment_norm = np.linalg.norm(segment, axis=1)
        out.append(float(segment_norm.mean()))
        out.append(float(np.sqrt(np.mean(np.sum((segment - segment_mean) ** 2, axis=1)))))
    for lag in (1, 5):
        for axis in range(3):
            if (
                n <= lag
                or np.std(window[:-lag, axis]) <= 1.0e-12
                or np.std(window[lag:, axis]) <= 1.0e-12
            ):
                out.append(0.0)
            else:
                value = float(np.corrcoef(window[:-lag, axis], window[lag:, axis])[0, 1])
                out.append(value if np.isfinite(value) else 0.0)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        difference = window[:, i] - window[:, j]
        out.extend([float(difference.mean()), float(difference.std())])
    for i, j in ((0, 1), (0, 2), (1, 2)):
        difference = window[:, i] - window[:, j]
        out.extend(
            [float(np.sqrt(np.mean(difference**2))), float(np.mean(np.abs(np.diff(difference))))]
        )
    out.extend(window[0].tolist())
    out.extend(window[-1].tolist())
    result = np.asarray(out, dtype=np.float64)
    if result.shape != (161,) or not np.isfinite(result).all():
        raise RuntimeError(f"invalid rich feature vector shape={result.shape}")
    return result


def _temporal_features(window: np.ndarray) -> np.ndarray:
    """Fixed multi-scale temporal map used when torch is unavailable."""

    pieces = [_base_features(window)]
    segments = np.array_split(window, 4, axis=0)
    segment_base = [_base_features(segment) for segment in segments]
    pieces.extend(segment_base)
    pieces.extend([segment_base[i + 1] - segment_base[i] for i in range(3)])
    result = np.concatenate(pieces)
    if result.shape != (192,) or not np.isfinite(result).all():
        raise RuntimeError("invalid temporal feature vector")
    return result  # type: ignore[no-any-return]


def _physics_features(window: np.ndarray) -> np.ndarray:
    mean = window.mean(axis=0)
    norm_mean = max(float(np.linalg.norm(mean)), 1.0e-12)
    gravity = mean / norm_mean
    residual = window - mean
    dynamic_energy = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    gravity_energy = float(norm_mean)
    ratio = dynamic_energy / max(gravity_energy, 1.0e-12)
    direction_stability = float(
        np.mean(
            np.linalg.norm(
                np.diff(
                    window / np.maximum(np.linalg.norm(window, axis=1, keepdims=True), 1.0e-12),
                    axis=0,
                ),
                axis=1,
            )
        )
    )
    covariance = np.cov(window, rowvar=False, ddof=0)
    eigenvalues = np.linalg.eigvalsh(covariance)
    stationary_confidence = 1.0 / (1.0 + 8.0 * ratio + 20.0 * direction_stability)
    gated_gravity = gravity * stationary_confidence
    result = np.concatenate(
        (
            gravity,
            np.array([dynamic_energy, gravity_energy, ratio, direction_stability]),
            eigenvalues,
            np.array([stationary_confidence]),
            gated_gravity,
        )
    )
    if result.shape != (14,) or not np.isfinite(result).all():
        raise RuntimeError("invalid physics feature vector")
    return result  # type: ignore[no-any-return]


def _load_windows(
    archive: Path, window_samples: int, *, retain_raw: bool = False
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    base_features: list[np.ndarray] = []
    rich_features: list[np.ndarray] = []
    temporal_features: list[np.ndarray] = []
    physics_features: list[np.ndarray] = []
    thigh_rich_features: list[np.ndarray] = []
    raw_back_windows: list[np.ndarray] = []
    labels: list[int] = []
    participants: list[str] = []
    window_ids: list[str] = []
    stats: dict[str, Any] = {
        "member_count": 0,
        "rows_total": 0,
        "physical_run_count": 0,
        "timestamp_boundary_count": 0,
        "participants": [],
        "window_count": 0,
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
            back = (
                frame[["back_x", "back_y", "back_z"]].to_numpy(dtype=np.float64) * STANDARD_GRAVITY
            )
            thigh = (
                frame[["thigh_x", "thigh_y", "thigh_z"]].to_numpy(dtype=np.float64)
                * STANDARD_GRAVITY
            )
            values = np.column_stack((back, thigh))
            raw_labels = pd.to_numeric(frame["label"], errors="coerce").to_numpy(dtype=np.float64)
            physical_runs, gap_count = _physical_runs(timestamp, values)
            stats["physical_run_count"] += len(physical_runs)
            stats["timestamp_boundary_count"] += int(gap_count)
            for physical_start, physical_stop in physical_runs:
                selected = np.isin(
                    raw_labels[physical_start:physical_stop], np.asarray(tuple(LABEL_MAP))
                )
                for local_start, local_stop in _true_runs(selected):
                    absolute_start = physical_start + local_start
                    absolute_stop = physical_start + local_stop
                    segment_labels = raw_labels[absolute_start:absolute_stop]
                    changes = np.r_[True, segment_labels[1:] != segment_labels[:-1]]
                    change_starts = np.flatnonzero(changes)
                    change_stops = np.concatenate(
                        (change_starts[1:], np.array([segment_labels.size]))
                    )
                    for label_start, label_stop in zip(change_starts, change_stops, strict=True):
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
                            if (
                                back_window.shape != (window_samples, 3)
                                or thigh_window.shape != (window_samples, 3)
                                or not np.isfinite(back_window).all()
                                or not np.isfinite(thigh_window).all()
                            ):
                                continue
                            base_features.append(_base_features(back_window))
                            rich_features.append(_rich_features(back_window))
                            temporal_features.append(_temporal_features(back_window))
                            physics_features.append(_physics_features(back_window))
                            thigh_rich_features.append(_rich_features(thigh_window))
                            if retain_raw:
                                raw_back_windows.append(np.asarray(back_window, dtype=np.float64))
                            labels.append(LABEL_MAP[code])
                            participants.append(participant)
                            window_ids.append(f"{participant}:{item.filename}:{left}:{right}")
                            stats["window_label_counts"][CLASS_NAMES[LABEL_MAP[code]]] += 1
    if not labels:
        raise RuntimeError("HARTH produced no complete selected windows")
    arrays = {
        "base": np.asarray(base_features, dtype=np.float64),
        "rich": np.asarray(rich_features, dtype=np.float64),
        "temporal": np.asarray(temporal_features, dtype=np.float64),
        "physics": np.asarray(physics_features, dtype=np.float64),
        "thigh_rich": np.asarray(thigh_rich_features, dtype=np.float64),
        "labels": np.asarray(labels, dtype=np.int64),
        "participants": np.asarray(participants, dtype=np.str_),
        "window_ids": np.asarray(window_ids, dtype=np.str_),
    }
    if retain_raw:
        arrays["raw_back"] = np.asarray(raw_back_windows, dtype=np.float64)
    if any(not np.isfinite(value).all() for value in arrays.values() if value.dtype.kind == "f"):
        raise RuntimeError("feature materialization produced non-finite values")
    stats["participants"] = sorted(np.unique(arrays["participants"]).tolist())
    stats["participant_count"] = len(stats["participants"])
    stats["window_count"] = int(arrays["labels"].size)
    stats["window_samples"] = window_samples
    stats["sampling_rate_hz"] = SOURCE_RATE_HZ
    return arrays, stats


def _fold_assignment(participants: list[str]) -> dict[str, int]:
    order = np.random.default_rng(SEED).permutation(len(participants))
    return {
        participant: int(position % FOLD_COUNT)
        for position, participant in enumerate(np.asarray(participants)[order])
    }


def _probability_align(raw: np.ndarray, classes: np.ndarray) -> np.ndarray:
    aligned = np.zeros((raw.shape[0], 2), dtype=np.float64)
    for column, class_index in enumerate(classes.tolist()):
        aligned[:, int(class_index)] = raw[:, column]
    return aligned


def _participant_scores(report: dict[str, Any]) -> dict[str, float]:
    return {str(row["participant_id"]): float(row["macro_f1"]) for row in report["participants"]}


def _paired_bootstrap(control: dict[str, float], candidate: dict[str, float]) -> dict[str, Any]:
    common = sorted(set(control) & set(candidate))
    differences = np.asarray([candidate[item] - control[item] for item in common], dtype=np.float64)
    rng = np.random.default_rng(1729)
    draws = rng.choice(differences, size=(10000, differences.size), replace=True).mean(axis=1)
    return {
        "mean": float(differences.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "wins": int(np.sum(differences > 0.0)),
        "harms": int(np.sum(differences < 0.0)),
        "ties": int(np.sum(differences == 0.0)),
        "worst_difference": float(differences.min()),
        "common_participants": common,
    }


def _train_feature_map(
    train_x: np.ndarray,
    test_x: np.ndarray,
    mode: str,
    train_thigh: np.ndarray | None = None,
    test_thigh: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build an outer-fold-only representation for an arm."""

    if mode == "base":
        return train_x, test_x, {"map": "identity"}
    if mode == "rich":
        return train_x, test_x, {"map": "identity"}
    if mode == "temporal":
        return train_x, test_x, {"map": "identity"}
    if mode == "temporal_physics":
        return train_x, test_x, {"map": "identity_plus_physics"}
    if mode == "paired_reconstruction":
        if train_thigh is None or test_thigh is None:
            raise ValueError("paired reconstruction requires aligned thigh features")
        # The mapping is fitted only on outer-training paired windows.  The test
        # thigh vector is retained solely for the held-out reconstruction diagnostic;
        # inference features use the back-derived prediction, never test_thigh.
        mapper = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        mapper.fit(train_x, train_thigh)
        predicted_train = np.asarray(mapper.predict(train_x), dtype=np.float64)
        predicted_test = np.asarray(mapper.predict(test_x), dtype=np.float64)
        return (
            np.column_stack((train_x, predicted_train)),
            np.column_stack((test_x, predicted_test)),
            {
                "map": "back_to_thigh_ridge",
                "alpha": 10.0,
                "train_reconstruction_rmse": float(
                    np.sqrt(np.mean((predicted_train - train_thigh) ** 2))
                ),
                "test_reconstruction_rmse_diagnostic": float(
                    np.sqrt(np.mean((predicted_test - test_thigh) ** 2))
                ),
            },
        )
    if mode == "self_reconstruction":
        # Generic label-free control: compress and reconstruct the back feature
        # map inside each outer fold, then classify the learned latent coordinates.
        components = min(32, train_x.shape[1], max(2, train_x.shape[0] - 1))
        pca = make_pipeline(StandardScaler(), PCA(n_components=components, random_state=SEED))
        return (
            np.asarray(pca.fit_transform(train_x)),
            np.asarray(pca.transform(test_x)),
            {"map": "back_pca_latent", "components": components},
        )
    raise ValueError(f"unknown representation mode: {mode}")


def _run_arm(
    *,
    arm: str,
    mode: str,
    features: np.ndarray,
    physics: np.ndarray,
    thigh: np.ndarray,
    labels: np.ndarray,
    participants: np.ndarray,
    fold_assignment: dict[str, int],
    window_samples: int,
) -> dict[str, Any]:
    probability_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    participant_parts: list[np.ndarray] = []
    fold_records: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    for fold in range(FOLD_COUNT):
        test_mask = np.asarray(
            [fold_assignment[str(value)] == fold for value in participants], dtype=np.bool_
        )
        train_mask = ~test_mask
        if mode in ("temporal_physics", "paired_reconstruction"):
            train_input = np.column_stack((features[train_mask], physics[train_mask]))
            test_input = np.column_stack((features[test_mask], physics[test_mask]))
        else:
            train_input = features[train_mask]
            test_input = features[test_mask]
        train_input, test_input, map_receipt = _train_feature_map(
            train_input,
            test_input,
            mode,
            train_thigh=thigh[train_mask] if mode == "paired_reconstruction" else None,
            test_thigh=thigh[test_mask] if mode == "paired_reconstruction" else None,
        )
        estimator = RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED + fold,
            n_jobs=4,
        )
        estimator.fit(train_input, labels[train_mask])
        raw_probability = np.asarray(estimator.predict_proba(test_input), dtype=np.float64)
        probability = _probability_align(raw_probability, estimator.classes_)
        probability_parts.append(probability)
        label_parts.append(labels[test_mask])
        participant_parts.append(participants[test_mask])
        fold_records.append(
            {
                "fold": fold,
                "training_participants": sorted(
                    p for p, assigned in fold_assignment.items() if assigned != fold
                ),
                "evaluation_participants": sorted(
                    p for p, assigned in fold_assignment.items() if assigned == fold
                ),
                "training_window_count": int(train_mask.sum()),
                "evaluation_window_count": int(test_mask.sum()),
                "feature_count": int(train_input.shape[1]),
                "map_receipt": map_receipt,
            }
        )
    observed = np.concatenate(label_parts)
    observed_participants = np.concatenate(participant_parts)
    probability = np.concatenate(probability_parts)
    report = classification_report(
        observed, probability, observed_participants.tolist(), class_names=CLASS_NAMES
    )
    return {
        "arm": arm,
        "representation": mode,
        "window_samples": window_samples,
        "feature_count": int(
            features.shape[1]
            + (physics.shape[1] if mode in ("temporal_physics", "paired_reconstruction") else 0)
            + (thigh.shape[1] if mode == "paired_reconstruction" else 0)
        ),
        "fold_count": FOLD_COUNT,
        "fit_count": FOLD_COUNT,
        "folds": fold_records,
        "random_forest": {
            "n_estimators": 300,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
            "seed_base": SEED,
            "n_jobs": 4,
        },
        "report": report,
        "runtime_seconds": time.perf_counter() - start_time,
    }


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: harth_crsp_back.py ARCHIVE OUTPUT_DIRECTORY")
    archive = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC)
    archive_hash = _sha256(archive)
    all_features: dict[int, dict[str, np.ndarray]] = {}
    data_audit: dict[str, Any] = {}
    for window_samples in (128, 250):
        all_features[window_samples], data_audit[str(window_samples)] = _load_windows(
            archive, window_samples
        )
    participant_list = sorted(np.unique(all_features[128]["participants"]).tolist())
    fold_assignment = _fold_assignment(participant_list)
    methods: list[dict[str, Any]] = []
    # Exact control and the fixed context/feature/temporal/physics arms.
    arms = [
        ("A_control_128_base_rf", 128, "base"),
        ("B_context_250_base_rf", 250, "base"),
        ("C_rich_250_rf", 250, "rich"),
        ("D_temporal_250_rf", 250, "temporal"),
        ("E_temporal_physics_250_rf", 250, "temporal_physics"),
        ("F_paired_reconstruction_250_rf", 250, "paired_reconstruction"),
    ]
    for arm, window_samples, mode in arms:
        source = all_features[window_samples]
        feature_key = "base" if mode == "base" else "rich" if mode == "rich" else "temporal"
        methods.append(
            _run_arm(
                arm=arm,
                mode=mode,
                features=source[feature_key],
                physics=source["physics"],
                thigh=source["thigh_rich"],
                labels=source["labels"],
                participants=source["participants"],
                fold_assignment=fold_assignment,
                window_samples=window_samples,
            )
        )
    by_arm = {str(item["arm"]): item for item in methods}
    control_scores = _participant_scores(by_arm["A_control_128_base_rf"]["report"])
    contrasts: dict[str, Any] = {}
    for item in methods[1:]:
        contrasts[str(item["arm"])] = _paired_bootstrap(
            control_scores, _participant_scores(item["report"])
        )
    ended = datetime.now(UTC)
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "harth_crsp_back_screen",
        "status": "completed_exploratory_external_development",
        "created_at_utc": ended.isoformat(),
        "started_at_utc": started.isoformat(),
        "protocol_path": PROTOCOL_PATH,
        "dataset": {
            "id": "HARTH",
            "record_url": SOURCE_RECORD,
            "download_url": SOURCE_URL,
            "doi": SOURCE_DOI,
            "license": "CC BY 4.0",
            "sha256": archive_hash,
            "received_size_bytes": int(archive.stat().st_size),
            "raw_local_mirror_retained": False,
            "source_description": "22 participants, right-thigh and lower-back Axivity accelerometers, nominal 50 Hz",
        },
        "contract": {
            "classes": list(CLASS_NAMES),
            "label_map": {str(k): v for k, v in LABEL_MAP.items()},
            "participant_exclusive_folds": FOLD_COUNT,
            "fold_seed": SEED,
            "window_overlap_samples": 0,
            "maximum_timestamp_gap_seconds": MAX_GAP_SECONDS,
            "gravity": "derived low-pass vector only; no native gravity or gyroscope",
            "inference_sensor": "lower_back_only",
            "paired_thigh_usage": "training_only_reconstruction_target",
            "feature_cache": "in_memory deterministic materialization; no label/identity columns",
            "torch_available": False,
        },
        "data_audit": data_audit,
        "fold_assignment": fold_assignment,
        "methods": methods,
        "contrasts_vs_A_control": contrasts,
        "promotion_gate": {
            "mean_macro_f1_gain_min": 0.05,
            "bootstrap_ci_lower_positive": True,
            "standing_recall_gain_min": 0.10,
            "sitting_recall_loss_max": 0.02,
            "participant_wins_min": 16,
            "participant_harm_floor": -0.05,
        },
        "claim_boundary": {
            "hera_confirmation_eligible": False,
            "inclusivehar_population_claim": False,
            "novelty_status": "hypothesis_only_until_independent_confirmation",
            "interpretation": "Bounded HARTH back-only screen; paired thigh is used only to train a back-derived reconstruction map.",
        },
        "methods_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    payload = json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False)
    result["result_payload_sha256_before_serialization"] = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()
    (output / "result.json").write_text(
        json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "SOURCE_RECEIPT.json").write_text(
        json.dumps(result["dataset"], sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                item["arm"]: item["report"]["primary"]["mean_participant_macro_f1"]
                for item in methods
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
