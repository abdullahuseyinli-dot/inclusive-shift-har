"""Existing-data standing observability experiment on HARTH.

The experiment compares the retained lower-back control with a thigh expert,
matched back-plus-thigh fusion, and a train-only confidence gate.  The gate lets
the thigh branch override a sitting back decision only when an outer-training
out-of-fold threshold supports the change.  No held-out labels select a threshold.
"""

from __future__ import annotations

import hashlib
import json
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
PROTOCOL_PATH = "docs/research/HARTH_STANDING_OBSERVABILITY_GATE_V1_PROTOCOL.md"
WINDOW_SAMPLES = 250
GATE_LOSS_LIMIT = 0.02
INNER_FOLD_COUNT = 4
GATE_GRID_SIZE = 201


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _align_probability(raw: np.ndarray, classes: np.ndarray) -> np.ndarray:
    output = np.zeros((raw.shape[0], 2), dtype=np.float64)
    for column, class_index in enumerate(classes.tolist()):
        output[:, int(class_index)] = raw[:, column]
    return output


def _fit_probability(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    random_state: int,
) -> np.ndarray:
    estimator = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=4,
    )
    estimator.fit(train_x, train_y)
    return _align_probability(estimator.predict_proba(test_x), estimator.classes_)


def _one_hot(prediction: np.ndarray) -> np.ndarray:
    output = np.full((prediction.size, 2), 1.0e-6, dtype=np.float64)
    output[np.arange(prediction.size), prediction.astype(np.int64)] = 1.0 - 1.0e-6
    return output


def _inner_oof_probabilities(
    train_mask: np.ndarray,
    back: np.ndarray,
    thigh: np.ndarray,
    labels: np.ndarray,
    participants: np.ndarray,
    outer_fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create train-only OOF probabilities for gate threshold selection."""

    train_indices = np.flatnonzero(train_mask)
    people = sorted(np.unique(participants[train_mask]).tolist())
    order = np.random.default_rng(SEED + 1000 + outer_fold).permutation(len(people))
    assignment = {
        person: int(position % INNER_FOLD_COUNT)
        for position, person in enumerate(np.asarray(people)[order])
    }
    back_oof = np.zeros((train_indices.size, 2), dtype=np.float64)
    thigh_oof = np.zeros((train_indices.size, 2), dtype=np.float64)
    for inner_fold in range(INNER_FOLD_COUNT):
        validation = np.asarray(
            [assignment[str(value)] == inner_fold for value in participants[train_indices]],
            dtype=np.bool_,
        )
        inner_train = ~validation
        validation_indices = train_indices[validation]
        inner_train_indices = train_indices[inner_train]
        back_oof[validation] = _fit_probability(
            back[inner_train_indices],
            back[validation_indices],
            labels[inner_train_indices],
            SEED + 1000 + outer_fold * 10 + inner_fold,
        )
        thigh_oof[validation] = _fit_probability(
            thigh[inner_train_indices],
            thigh[validation_indices],
            labels[inner_train_indices],
            SEED + 2000 + outer_fold * 10 + inner_fold,
        )
    return back_oof, thigh_oof


def _recall(labels: np.ndarray, prediction: np.ndarray, class_index: int) -> float:
    support = int(np.sum(labels == class_index))
    return (
        0.0
        if support == 0
        else float(np.sum((labels == class_index) & (prediction == class_index)) / support)
    )


def _gate_prediction(
    back_prediction: np.ndarray, thigh_probability: np.ndarray, threshold: float
) -> np.ndarray:
    """Override only sitting back decisions with high-confidence thigh standing."""

    prediction = np.asarray(back_prediction, dtype=np.int64).copy()
    override = (prediction == 0) & (thigh_probability[:, 1] >= threshold)
    prediction[override] = 1
    return prediction


def _choose_gate_threshold(
    labels: np.ndarray,
    back_probability: np.ndarray,
    thigh_probability: np.ndarray,
) -> dict[str, Any]:
    back_prediction = np.argmax(back_probability, axis=1).astype(np.int64)
    back_sitting_recall = _recall(labels, back_prediction, 0)
    candidates = np.unique(
        np.concatenate((np.linspace(0.0, 1.0, GATE_GRID_SIZE), thigh_probability[:, 1]))
    )
    feasible: list[tuple[float, float, float, np.ndarray]] = []
    for threshold in candidates.tolist():
        prediction = _gate_prediction(back_prediction, thigh_probability, float(threshold))
        sitting_recall = _recall(labels, prediction, 0)
        standing_recall = _recall(labels, prediction, 1)
        loss = back_sitting_recall - sitting_recall
        if loss <= GATE_LOSS_LIMIT + 1.0e-12:
            feasible.append((standing_recall, sitting_recall, float(threshold), prediction))
    if feasible:
        selected = max(feasible, key=lambda item: (item[0], item[1], item[2]))
        fallback = False
    else:
        selected = max(
            (
                (
                    _recall(
                        labels,
                        _gate_prediction(back_prediction, thigh_probability, float(threshold)),
                        1,
                    ),
                    _recall(
                        labels,
                        _gate_prediction(back_prediction, thigh_probability, float(threshold)),
                        0,
                    ),
                    float(threshold),
                    _gate_prediction(back_prediction, thigh_probability, float(threshold)),
                )
                for threshold in candidates.tolist()
            ),
            key=lambda item: (item[1], item[0], item[2]),
        )
        fallback = True
    return {
        "threshold": selected[2],
        "back_sitting_recall": back_sitting_recall,
        "selected_sitting_recall": selected[1],
        "selected_standing_recall": selected[0],
        "sitting_recall_loss": back_sitting_recall - selected[1],
        "fallback_no_feasible_threshold": fallback,
        "candidate_count": int(candidates.size),
    }


def _fit_outer(
    *,
    back: np.ndarray,
    thigh: np.ndarray,
    labels: np.ndarray,
    participants: np.ndarray,
    window_ids: np.ndarray,
    assignment: dict[str, int],
    output_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    started = time.perf_counter()
    probabilities = {
        "B_back_rich": np.zeros((labels.size, 2), dtype=np.float64),
        "T_thigh_rich": np.zeros((labels.size, 2), dtype=np.float64),
        "F_fused_rich": np.zeros((labels.size, 2), dtype=np.float64),
        "G_confidence_gated": np.zeros((labels.size, 2), dtype=np.float64),
    }
    decoded = {name: np.zeros(labels.size, dtype=np.int64) for name in probabilities}
    folds: dict[str, list[dict[str, Any]]] = {name: [] for name in probabilities}
    for fold in range(FOLD_COUNT):
        test_mask = np.asarray(
            [assignment[str(value)] == fold for value in participants], dtype=np.bool_
        )
        train_mask = ~test_mask
        back_test = _fit_probability(
            back[train_mask], back[test_mask], labels[train_mask], SEED + fold
        )
        thigh_test = _fit_probability(
            thigh[train_mask], thigh[test_mask], labels[train_mask], SEED + 100 + fold
        )
        fused_test = _fit_probability(
            np.column_stack((back[train_mask], thigh[train_mask])),
            np.column_stack((back[test_mask], thigh[test_mask])),
            labels[train_mask],
            SEED + 200 + fold,
        )
        inner_back, inner_thigh = _inner_oof_probabilities(
            train_mask, back, thigh, labels, participants, fold
        )
        gate = _choose_gate_threshold(labels[train_mask], inner_back, inner_thigh)
        gate_test_prediction = _gate_prediction(
            np.argmax(back_test, axis=1), thigh_test, float(gate["threshold"])
        )
        probabilities["B_back_rich"][test_mask] = back_test
        probabilities["T_thigh_rich"][test_mask] = thigh_test
        probabilities["F_fused_rich"][test_mask] = fused_test
        probabilities["G_confidence_gated"][test_mask] = _one_hot(gate_test_prediction)
        decoded["B_back_rich"][test_mask] = np.argmax(back_test, axis=1)
        decoded["T_thigh_rich"][test_mask] = np.argmax(thigh_test, axis=1)
        decoded["F_fused_rich"][test_mask] = np.argmax(fused_test, axis=1)
        decoded["G_confidence_gated"][test_mask] = gate_test_prediction
        for name in probabilities:
            folds[name].append(
                {
                    "fold": fold,
                    "training_participants": sorted(
                        person for person, assigned in assignment.items() if assigned != fold
                    ),
                    "evaluation_participants": sorted(
                        person for person, assigned in assignment.items() if assigned == fold
                    ),
                    "training_window_count": int(train_mask.sum()),
                    "evaluation_window_count": int(test_mask.sum()),
                    "feature_count": int(
                        back.shape[1]
                        if name == "B_back_rich"
                        else thigh.shape[1]
                        if name == "T_thigh_rich"
                        else back.shape[1] + thigh.shape[1]
                    ),
                    "gate": gate if name == "G_confidence_gated" else None,
                }
            )
    results: list[dict[str, Any]] = []
    for name in probabilities:
        report = classification_report(
            labels, probabilities[name], participants.tolist(), class_names=CLASS_NAMES
        )
        prediction = decoded[name]
        results.append(
            {
                "arm": name,
                "representation": {
                    "B_back_rich": "lower_back_rich_161",
                    "T_thigh_rich": "right_thigh_rich_161",
                    "F_fused_rich": "lower_back_plus_right_thigh_rich_322",
                    "G_confidence_gated": "back_rich_with_outer_training_thigh_override",
                }[name],
                "postprocess": "train_only_inner_participant_gate"
                if name.startswith("G_")
                else "none",
                "report": report,
                "folds": folds[name],
                "fit_count": FOLD_COUNT,
                "runtime_seconds": time.perf_counter() - started,
                "prediction_positive_rate": float(np.mean(prediction == 1)),
            }
        )
        np.savez_compressed(
            output_path / f"{name}_predictions.npz",
            labels=labels,
            participants=participants,
            window_ids=window_ids,
            probabilities=probabilities[name],
            predictions=prediction,
        )
    return results, probabilities


def run(archive: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"create-only output already exists: {output}")
    output.mkdir(parents=True)
    started = datetime.now(UTC)
    archive_hash = _sha256(archive)
    arrays, data_audit = _load_windows(archive, WINDOW_SAMPLES)
    participants = np.asarray(arrays["participants"], dtype=np.str_)
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    window_ids = np.asarray(arrays["window_ids"], dtype=np.str_)
    back = np.asarray(arrays["rich"], dtype=np.float64)
    thigh = np.asarray(arrays["thigh_rich"], dtype=np.float64)
    people = sorted(np.unique(participants).tolist())
    assignment = _fold_assignment(people)
    results, _probabilities = _fit_outer(
        back=back,
        thigh=thigh,
        labels=labels,
        participants=participants,
        window_ids=window_ids,
        assignment=assignment,
        output_path=output,
    )
    control_people = _participant_scores(
        next(item["report"] for item in results if item["arm"] == "B_back_rich")
    )
    contrasts = {
        item["arm"]: _paired_bootstrap(
            control_people,
            _participant_scores(item["report"]),
        )
        for item in results
        if item["arm"] != "B_back_rich"
    }
    ended = datetime.now(UTC)
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "harth_standing_observability_gate",
        "status": "completed_exploratory_existing_data",
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
            "inference_sensor_by_arm": {
                "B_back_rich": "lower_back_only",
                "T_thigh_rich": "right_thigh_only",
                "F_fused_rich": "lower_back_and_right_thigh",
                "G_confidence_gated": "lower_back_and_right_thigh_gate",
            },
            "estimator": {
                "name": "random_forest",
                "n_estimators": 300,
                "max_features": "sqrt",
                "min_samples_leaf": 2,
                "class_weight": "balanced_subsample",
                "seed_base": SEED,
                "n_jobs": 4,
            },
            "gate": {
                "threshold_selection": "outer_training_inner_participant_oof",
                "override": "sitting_back_decision_only",
                "sitting_recall_loss_limit": GATE_LOSS_LIMIT,
            },
        },
        "data_audit": data_audit,
        "fold_assignment": assignment,
        "results": results,
        "contrasts_vs_back_control": contrasts,
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
            "back_only_inference_claim": False,
            "hera_confirmation_eligible": False,
            "inclusivehar_population_claim": False,
            "interpretation": "Existing-data sensor observability test; thigh and fused arms are information controls, not back-only deployment claims.",
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
        raise SystemExit("usage: harth_standing_observability.py HARTH_ZIP OUTPUT_DIR")
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
