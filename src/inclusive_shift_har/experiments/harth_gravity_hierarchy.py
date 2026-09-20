"""Bounded HARTH gravity-geometry and state-decoder experiment.

This experiment separates three hypotheses on the existing lower-back stream:
information in gravity geometry, a training-only operating-point correction, and
temporal state regularization.  It does not use thigh data at inference, add seeds,
or synthesize channels.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]

from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.experiments.harth_crsp_back import (
    CLASS_NAMES,
    FOLD_COUNT,
    LABEL_MAP,
    MAX_GAP_SECONDS,
    SEED,
    SOURCE_RATE_HZ,
    _fold_assignment,
    _load_windows,
    _paired_bootstrap,
    _participant_scores,
)

SOURCE_URL = "https://archive.ics.uci.edu/static/public/779/harth.zip"
SOURCE_RECORD = "https://archive.ics.uci.edu/dataset/779/harth"
PROTOCOL_PATH = "docs/research/HARTH_GRAVITY_HIERARCHY_V1_PROTOCOL.md"
WINDOW_SAMPLES = 250
MIN_DWELL_WINDOWS = 2
CALIBRATION_EPSILON = 1.0e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _moving_average(window: np.ndarray, width: int = 25) -> np.ndarray:
    if window.shape != (WINDOW_SAMPLES, 3) or width <= 0 or width > window.shape[0]:
        raise ValueError("invalid gravity window or smoothing width")
    kernel = np.ones(width, dtype=np.float64) / float(width)
    padded = np.pad(window, ((width // 2, width - 1 - width // 2), (0, 0)), mode="edge")
    return np.column_stack(
        [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(3)]
    )


def _gravity_geometry_features(window: np.ndarray) -> np.ndarray:
    """Return a fixed low-pass gravity/dynamic feature vector.

    The feature map uses only the current lower-back window.  It avoids native
    gyroscope channels and leaves all learned scaling/calibration to the outer fold.
    """

    smoothed = _moving_average(window)
    gravity = smoothed.mean(axis=0)
    gravity_norm = max(float(np.linalg.norm(gravity)), 1.0e-12)
    gravity_unit = gravity / gravity_norm
    unit_series = smoothed / np.maximum(np.linalg.norm(smoothed, axis=1, keepdims=True), 1.0e-12)
    unit_delta = np.linalg.norm(np.diff(unit_series, axis=0), axis=1)
    residual = window - smoothed
    residual_norm = np.linalg.norm(residual, axis=1)
    subunits = np.array_split(smoothed, 4, axis=0)
    subunit_units = [
        part.mean(axis=0) / max(float(np.linalg.norm(part.mean(axis=0))), 1.0e-12)
        for part in subunits
    ]
    subunit_delta = np.concatenate(
        [subunit_units[index + 1] - subunit_units[index] for index in range(3)]
    )
    abs_unit = np.abs(gravity_unit)
    sorted_abs = np.sort(abs_unit)
    pairwise = np.asarray(
        [
            abs_unit[0] / max(abs_unit[1], 1.0e-6),
            abs_unit[0] / max(abs_unit[2], 1.0e-6),
            abs_unit[1] / max(abs_unit[2], 1.0e-6),
        ],
        dtype=np.float64,
    )
    features = np.concatenate(
        (
            gravity,
            np.asarray([gravity_norm], dtype=np.float64),
            gravity_unit,
            pairwise,
            np.asarray(
                [
                    float(sorted_abs[-1]),
                    float(sorted_abs[0]),
                    float(abs_unit.sum()),
                    float(np.std(unit_series, axis=0).mean()),
                    float(unit_delta.mean()),
                    float(np.quantile(unit_delta, 0.90)),
                    float(unit_delta.max()),
                    float(residual_norm.mean()),
                    float(residual_norm.std()),
                    float(np.quantile(residual_norm, 0.90)),
                    float(np.sqrt(np.mean(residual_norm**2))),
                ],
                dtype=np.float64,
            ),
            subunit_delta,
        )
    )
    if features.shape != (30,) or not np.isfinite(features).all():
        raise RuntimeError(f"invalid gravity geometry feature shape={features.shape}")
    return np.asarray(features, dtype=np.float64)


def _geometry_matrix(raw_windows: np.ndarray) -> np.ndarray:
    if raw_windows.ndim != 3 or raw_windows.shape[1:] != (WINDOW_SAMPLES, 3):
        raise ValueError("raw lower-back windows have the wrong shape")
    return np.asarray(
        [_gravity_geometry_features(window) for window in raw_windows], dtype=np.float64
    )


def _align_probability(raw: np.ndarray, classes: np.ndarray) -> np.ndarray:
    output = np.zeros((raw.shape[0], 2), dtype=np.float64)
    for column, class_index in enumerate(classes.tolist()):
        output[:, int(class_index)] = raw[:, column]
    return output


def _prior_calibrate(probability: np.ndarray, train_labels: np.ndarray) -> tuple[np.ndarray, float]:
    """Correct a balanced RF's prior using only outer-training labels."""

    prior = float(np.mean(train_labels == 1))
    prior = min(max(prior, CALIBRATION_EPSILON), 1.0 - CALIBRATION_EPSILON)
    delta = math.log(prior / (1.0 - prior))
    clipped = np.clip(probability, CALIBRATION_EPSILON, 1.0 - CALIBRATION_EPSILON)
    logits = np.log(clipped[:, 1]) - np.log(clipped[:, 0])
    adjusted = logits + delta
    standing = 1.0 / (1.0 + np.exp(-adjusted))
    return np.column_stack((1.0 - standing, standing)), delta


def _parse_window_id(value: str) -> tuple[str, str, int, int]:
    parts = value.split(":")
    if len(parts) != 4:
        raise ValueError(f"unexpected window id: {value}")
    return parts[0], parts[1], int(parts[2]), int(parts[3])


def _transition_log_probs(
    labels: np.ndarray,
    participants: np.ndarray,
    window_ids: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    counts = np.full((2, 2), 0.5, dtype=np.float64)
    records = sorted(
        (
            _parse_window_id(str(window_ids[index])),
            index,
        )
        for index in np.flatnonzero(train_mask)
    )
    for (left_key, left_index), (right_key, right_index) in itertools.pairwise(records):
        if left_key[0] != right_key[0] or left_key[1] != right_key[1]:
            continue
        if right_key[2] != left_key[3]:
            continue
        counts[int(labels[left_index]), int(labels[right_index])] += 1.0
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    return np.asarray(np.log(probabilities), dtype=np.float64)


def _decode_sequence(probability: np.ndarray, transition_log: np.ndarray) -> np.ndarray:
    if probability.ndim != 2 or probability.shape[1] != 2 or probability.shape[0] == 0:
        raise ValueError("invalid sequence probability matrix")
    log_emission = np.log(np.clip(probability, CALIBRATION_EPSILON, 1.0))
    states = 2
    dwell = MIN_DWELL_WINDOWS
    score = np.full((probability.shape[0], states, dwell), -np.inf, dtype=np.float64)
    back_state = np.zeros((probability.shape[0], states, dwell), dtype=np.int8)
    back_dwell = np.zeros((probability.shape[0], states, dwell), dtype=np.int8)
    score[0, :, 0] = log_emission[0]
    for time_index in range(1, probability.shape[0]):
        for state in range(states):
            for previous_state in range(states):
                for previous_dwell in range(dwell):
                    if previous_state != state and previous_dwell < dwell - 1:
                        continue
                    next_dwell = (
                        min(dwell - 1, previous_dwell + 1) if previous_state == state else 0
                    )
                    candidate = (
                        score[time_index - 1, previous_state, previous_dwell]
                        + transition_log[previous_state, state]
                    )
                    if candidate > score[time_index, state, next_dwell]:
                        score[time_index, state, next_dwell] = candidate
                        back_state[time_index, state, next_dwell] = previous_state
                        back_dwell[time_index, state, next_dwell] = previous_dwell
            score[time_index, state, :] += log_emission[time_index, state]
    state, dwell_index = (
        int(value) for value in np.unravel_index(np.argmax(score[-1]), score[-1].shape)
    )
    decoded = np.empty(probability.shape[0], dtype=np.int64)
    decoded[-1] = state
    for time_index in range(probability.shape[0] - 1, 0, -1):
        previous_state = int(back_state[time_index, state, dwell_index])
        previous_dwell = int(back_dwell[time_index, state, dwell_index])
        state, dwell_index = previous_state, previous_dwell
        decoded[time_index - 1] = state
    return np.asarray(decoded, dtype=np.int64)


def _decode_sequences(
    probability: np.ndarray,
    participants: np.ndarray,
    window_ids: np.ndarray,
    indices: np.ndarray,
    transition_log: np.ndarray,
) -> np.ndarray:
    decoded = np.argmax(probability, axis=1).astype(np.int64)
    records = sorted(
        (_parse_window_id(str(window_ids[index])), int(index)) for index in indices.tolist()
    )
    groups: list[list[int]] = []
    current: list[int] = []
    previous: tuple[str, str, int, int] | None = None
    for key, index in records:
        if (
            previous is None
            or key[0] != previous[0]
            or key[1] != previous[1]
            or key[2] != previous[3]
        ):
            if current:
                groups.append(current)
            current = []
        current.append(index)
        previous = key
    if current:
        groups.append(current)
    for group in groups:
        order = np.asarray(group, dtype=np.int64)
        predicted = _decode_sequence(probability[order], transition_log)
        decoded[order] = predicted
    return np.asarray(decoded, dtype=np.int64)


def _one_hot(predicted: np.ndarray) -> np.ndarray:
    output = np.full((predicted.size, 2), CALIBRATION_EPSILON, dtype=np.float64)
    output[np.arange(predicted.size), predicted] = 1.0 - CALIBRATION_EPSILON
    return output


def _fit_arm(
    *,
    arm: str,
    features: np.ndarray,
    geometry: np.ndarray | None,
    labels: np.ndarray,
    participants: np.ndarray,
    window_ids: np.ndarray,
    fold_assignment: dict[str, int],
    output_path: Path,
    state_decoder: bool = False,
    calibrate: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    probabilities = np.zeros((labels.size, 2), dtype=np.float64)
    calibrated_probability = np.zeros_like(probabilities)
    decoded = np.zeros(labels.size, dtype=np.int64)
    folds: list[dict[str, Any]] = []
    for fold in range(FOLD_COUNT):
        test_mask = np.asarray(
            [fold_assignment[str(item)] == fold for item in participants], dtype=np.bool_
        )
        train_mask = ~test_mask
        train_x = features[train_mask]
        test_x = features[test_mask]
        if geometry is not None:
            train_x = np.column_stack((train_x, geometry[train_mask]))
            test_x = np.column_stack((test_x, geometry[test_mask]))
        estimator = RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED + fold,
            n_jobs=4,
        )
        estimator.fit(train_x, labels[train_mask])
        raw = _align_probability(estimator.predict_proba(test_x), estimator.classes_)
        probabilities[test_mask] = raw
        calibrated, delta = _prior_calibrate(raw, labels[train_mask])
        calibrated_probability[test_mask] = calibrated
        if state_decoder:
            transition_log = _transition_log_probs(labels, participants, window_ids, train_mask)
            decoded_fold = _decode_sequences(
                probabilities,
                participants,
                window_ids,
                np.flatnonzero(test_mask),
                transition_log,
            )
            decoded[test_mask] = decoded_fold[test_mask]
        else:
            decoded[test_mask] = np.argmax(raw, axis=1)
        folds.append(
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
                "feature_count": int(train_x.shape[1]),
                "prior_correction_logit": delta,
                "transition_log_probabilities": transition_log.tolist() if state_decoder else None,
            }
        )
    report_probability = calibrated_probability if calibrate else probabilities
    if state_decoder:
        report_probability = _one_hot(decoded)
    report = classification_report(
        labels, report_probability, participants.tolist(), class_names=CLASS_NAMES
    )
    raw_report = classification_report(
        labels, probabilities, participants.tolist(), class_names=CLASS_NAMES
    )
    calibrated_report = classification_report(
        labels, calibrated_probability, participants.tolist(), class_names=CLASS_NAMES
    )
    np.savez_compressed(
        output_path / f"{arm}_predictions.npz",
        labels=labels,
        participants=participants,
        window_ids=window_ids,
        raw_probabilities=probabilities,
        calibrated_probabilities=calibrated_probability,
        decoded=decoded,
    )
    return {
        "arm": arm,
        "representation": "rich" if geometry is None else "rich_plus_gravity_geometry",
        "state_decoder": state_decoder,
        "calibration": calibrate,
        "window_samples": WINDOW_SAMPLES,
        "feature_count": int(
            features.shape[1] + (geometry.shape[1] if geometry is not None else 0)
        ),
        "fold_count": FOLD_COUNT,
        "fit_count": FOLD_COUNT,
        "folds": folds,
        "report": report,
        "raw_geometry_report": raw_report if geometry is not None else None,
        "calibrated_report": calibrated_report if geometry is not None else None,
        "runtime_seconds": time.perf_counter() - started,
    }


def run(archive: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"create-only output already exists: {output}")
    output.mkdir(parents=True)
    started = datetime.now(UTC)
    archive_hash = _sha256(archive)
    arrays, data_audit = _load_windows(archive, WINDOW_SAMPLES, retain_raw=True)
    raw_windows = arrays.pop("raw_back")
    geometry = _geometry_matrix(raw_windows)
    participants = sorted(np.unique(arrays["participants"]).tolist())
    assignment = _fold_assignment(participants)
    rich = arrays["rich"]
    labels = arrays["labels"]
    participant_ids = arrays["participants"]
    window_ids = arrays["window_ids"]
    results = []
    for arm, use_geometry, calibrate, state_decoder in (
        ("C_rich_control", False, False, False),
        ("G_gravity_geometry", True, False, False),
        ("G_calibrated", True, True, False),
        ("G_state_decoder", True, False, True),
    ):
        results.append(
            _fit_arm(
                arm=arm,
                features=rich,
                geometry=geometry if use_geometry else None,
                labels=labels,
                participants=participant_ids,
                window_ids=window_ids,
                fold_assignment=assignment,
                output_path=output,
                state_decoder=state_decoder,
                calibrate=calibrate,
            )
        )
    by_arm = {item["arm"]: item for item in results}
    control_people = _participant_scores(by_arm["C_rich_control"]["report"])
    contrasts = {}
    for item in results[1:]:
        contrasts[item["arm"]] = _paired_bootstrap(
            control_people, _participant_scores(item["report"])
        )
    ended = datetime.now(UTC)
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "harth_gravity_hierarchy_screen",
        "status": "completed_exploratory_external_development",
        "created_at_utc": ended.isoformat(),
        "started_at_utc": started.isoformat(),
        "protocol_path": PROTOCOL_PATH,
        "source": {
            "url": SOURCE_URL,
            "record": SOURCE_RECORD,
            "sha256": archive_hash,
            "size_bytes": archive.stat().st_size,
            "raw_archive_retained": False,
        },
        "contract": {
            "classes": list(CLASS_NAMES),
            "label_map": {str(key): value for key, value in LABEL_MAP.items()},
            "participant_exclusive_folds": FOLD_COUNT,
            "fold_seed": SEED,
            "window_samples": WINDOW_SAMPLES,
            "window_overlap_samples": 0,
            "sampling_rate_hz": SOURCE_RATE_HZ,
            "maximum_timestamp_gap_seconds": MAX_GAP_SECONDS,
            "inference_sensor": "lower_back_only",
            "gravity_source": "low_passed_accelerometer_only",
            "calibration": "outer_training empirical class-prior logit correction",
            "state_decoder": {
                "minimum_dwell_windows": MIN_DWELL_WINDOWS,
                "transition_counts": "outer_training only",
            },
        },
        "data_audit": data_audit,
        "geometry_feature_count": int(geometry.shape[1]),
        "fold_assignment": assignment,
        "results": results,
        "contrasts_vs_rich_control": contrasts,
        "promotion_gate": {
            "mean_macro_f1_gain_min": 0.05,
            "bootstrap_ci_lower_positive": True,
            "standing_recall_gain_min": 0.10,
            "standing_f1_gain_min": 0.10,
            "sitting_recall_loss_max": 0.02,
            "participant_wins_min": 16,
            "participant_harm_floor": -0.05,
        },
        "claim_boundary": {
            "hera_confirmation_eligible": False,
            "inclusivehar_population_claim": False,
            "novelty_status": "hypothesis_only_until_independent_confirmation",
            "interpretation": "Bounded HARTH lower-back experiment separating gravity geometry, calibration, and temporal state effects.",
        },
        "runtime_seconds": (ended - started).total_seconds(),
    }
    payload = json.dumps(result, sort_keys=True, indent=2)
    result["result_payload_sha256_before_serialization"] = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()
    (output / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "SOURCE_RECEIPT.json").write_text(
        json.dumps(result["source"], indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: harth_gravity_hierarchy.py HARTH_ZIP OUTPUT_DIR")
    payload = run(Path(sys.argv[1]), Path(sys.argv[2]))
    print(
        json.dumps(
            {
                item["arm"]: item["report"]["primary"]["mean_participant_macro_f1"]
                for item in payload["results"]
            },
            indent=2,
            sort_keys=True,
        )
    )
