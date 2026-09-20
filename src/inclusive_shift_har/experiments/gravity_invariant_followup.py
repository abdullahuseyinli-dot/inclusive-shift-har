"""Bounded follow-up for the Round-A native-gravity failure.

This experiment is deliberately small and predeclared: it reuses the frozen A2
denoised GSP matrix and appends only scalar gravity-relative dynamics. Coordinate
axes, absolute gravity components, and sorted component features are excluded so
the test asks whether the Round-A loss was caused by orientation-sensitive and
high-dimensional gravity features rather than by denoising or the estimator.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import ExtraTreesClassifier  # type: ignore[import-untyped]

from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]

CLASS_NAMES = ("mobility", "sitting", "standing")
PARTICIPANT_PAIRS = (("1", "4"), ("6", "7"), ("2", "3"), ("5", "9"), ("8", "10"))
SEED = 11
EXPECTED_ROWS = 725
INVARIANT_GROUPS = (
    "user_acceleration_parallel_gravity",
    "user_acceleration_perpendicular_norm",
    "gyroscope_parallel_gravity",
    "gyroscope_perpendicular_norm",
    "total_acceleration_parallel_gravity",
    "total_acceleration_perpendicular_norm",
    "gravity_direction_angular_speed",
    "user_acceleration_norm",
    "gyroscope_norm",
    "normalized_acceleration_gyroscope_dot",
)


def participant_first_weights(labels: IntArray, participants: StringArray) -> FloatArray:
    """Give every present person/class cell equal mass, then normalize to mean one."""

    if labels.ndim != 1 or participants.shape != labels.shape or labels.size == 0:
        raise ValueError("labels and participant ids must be nonempty aligned vectors")
    weights = np.empty(labels.size, dtype=np.float64)
    for participant in np.unique(participants):
        person = participants == participant
        present = np.unique(labels[person])
        for label in present:
            cell = person & (labels == label)
            weights[cell] = 1.0 / (len(present) * int(cell.sum()))
    return np.asarray(weights / weights.mean(), dtype=np.float64)


def select_invariant_features(
    gravity_features: FloatArray, feature_names: NDArray[np.str_]
) -> tuple[FloatArray, list[str], list[int]]:
    """Select the frozen scalar gravity-relative groups; reject accidental axes."""

    if gravity_features.ndim != 2 or gravity_features.shape[1] != feature_names.size:
        raise ValueError("gravity feature matrix and names are not aligned")
    selected = [
        index
        for index, name in enumerate(feature_names.astype(str).tolist())
        if len(name.split("__")) >= 3 and name.split("__")[1] in INVARIANT_GROUPS
    ]
    if len(selected) != len(INVARIANT_GROUPS) * 15:
        raise ValueError("invariant feature contract did not select exactly ten groups")
    names = [str(feature_names[index]) for index in selected]
    if any("gravity_unit_" in name or "gravity_abs_" in name for name in names):
        raise ValueError("orientation-sensitive gravity component leaked into invariant view")
    values = np.asarray(gravity_features[:, selected], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("selected invariant features are not finite")
    return values, names, selected


def _align_probability(model: Any, values: FloatArray) -> FloatArray:
    classes = np.asarray(model.classes_, dtype=np.int64)
    if not np.array_equal(classes, np.arange(3, dtype=np.int64)):
        raise ValueError("estimator class order differs from frozen [0, 1, 2]")
    return np.asarray(values, dtype=np.float64)


def paired_bootstrap(
    candidate: dict[str, float], reference: dict[str, float], *, seed: int = 1729
) -> dict[str, float]:
    people = sorted(candidate)
    if people != sorted(reference) or len(people) != 10:
        raise ValueError("follow-up bootstrap requires the ten fixed source participants")
    a = np.asarray([candidate[p] for p in people], dtype=np.float64)
    b = np.asarray([reference[p] for p in people], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(people), size=(10_000, len(people)))
    differences = (a[indices] - b[indices]).mean(axis=1)
    return {
        "mean_difference": float((a - b).mean()),
        "lower": float(np.quantile(differences, 0.025)),
        "upper": float(np.quantile(differences, 0.975)),
        "resamples": 10_000,
        "seed": seed,
    }


def _load_npz(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def run_followup(
    *,
    baseline_cache: Path,
    gravity_cache: Path,
    baseline_predictions: Path,
    output: Path,
    source_commit: str,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"create-only output already exists: {output}")
    baseline = _load_npz(baseline_cache)
    gravity = _load_npz(gravity_cache)
    cached = _load_npz(baseline_predictions)
    required = {"features", "labels", "participant_ids", "window_ids", "feature_names"}
    if not required.issubset(baseline) or not required.issubset(gravity):
        raise ValueError("feature cache is missing a required aligned array")
    labels = np.asarray(baseline["labels"], dtype=np.int64)
    participants = np.asarray(baseline["participant_ids"], dtype=np.str_)
    windows = np.asarray(baseline["window_ids"], dtype=np.str_)
    gravity_labels = np.asarray(gravity["labels"], dtype=np.int64)
    gravity_participants = np.asarray(gravity["participant_ids"], dtype=np.str_)
    gravity_windows = np.asarray(gravity["window_ids"], dtype=np.str_)
    if labels.size != EXPECTED_ROWS or len(np.unique(windows)) != EXPECTED_ROWS:
        raise ValueError("follow-up requires the frozen 725-window population")
    for left, right, name in (
        (labels, gravity_labels, "labels"),
        (participants, gravity_participants, "participant ids"),
        (windows, gravity_windows, "window ids"),
    ):
        if not np.array_equal(left, right):
            raise ValueError(f"{name} are not aligned between caches")
    if (
        not np.array_equal(cached["labels"], labels)
        or not np.array_equal(cached["participant_ids"], participants)
        or not np.array_equal(cached["window_ids"], windows)
    ):
        raise ValueError("cached A2 predictions are not aligned by authoritative ids")
    if not np.array_equal(np.unique(labels), np.arange(3, dtype=np.int64)):
        raise ValueError("labels do not use fixed three-class ontology")
    invariant, _invariant_names, selected_indices = select_invariant_features(
        np.asarray(gravity["features"], dtype=np.float64),
        np.asarray(gravity["feature_names"], dtype=np.str_),
    )
    a2 = np.asarray(baseline["features"], dtype=np.float64)
    x_followup = np.concatenate((a2, invariant), axis=1)
    if not np.isfinite(x_followup).all():
        raise ValueError("follow-up feature matrix contains nonfinite values")
    if x_followup.shape[1] != 1550:
        raise ValueError("follow-up feature dimension differs from frozen 1400+150")

    started = time.perf_counter()
    probabilities = np.empty((EXPECTED_ROWS, 3), dtype=np.float64)
    fold_records: list[dict[str, Any]] = []
    for fold_index, pair in enumerate(PARTICIPANT_PAIRS):
        test = np.isin(participants, pair)
        train = ~test
        if np.any(train & test) or set(participants[test]) != set(pair):
            raise ValueError("participant-exclusive fold contract failed")
        model = ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight=None,
            random_state=SEED,
            n_jobs=4,
        )
        fit_started = time.perf_counter()
        model.fit(
            x_followup[train],
            labels[train],
            sample_weight=participant_first_weights(labels[train], participants[train]),
        )
        fit_seconds = time.perf_counter() - fit_started
        prediction_started = time.perf_counter()
        probabilities[test] = _align_probability(model, model.predict_proba(x_followup[test]))
        prediction_seconds = time.perf_counter() - prediction_started
        fold_records.append(
            {
                "fold_index": fold_index,
                "held_out_participants": list(pair),
                "training_rows": int(train.sum()),
                "test_rows": int(test.sum()),
                "feature_count": int(x_followup.shape[1]),
                "fit_seconds": fit_seconds,
                "prediction_seconds": prediction_seconds,
                "effective_parameters": model.get_params(deep=False),
            }
        )

    candidate_report = classification_report(
        labels, probabilities, participants.tolist(), class_names=CLASS_NAMES
    )
    reference_report = classification_report(
        labels,
        np.asarray(cached["a2_probabilities"], dtype=np.float64),
        participants.tolist(),
        class_names=CLASS_NAMES,
    )
    candidate_people = {
        str(row["participant_id"]): float(row["macro_f1"])
        for row in candidate_report["participants"]
    }
    reference_people = {
        str(row["participant_id"]): float(row["macro_f1"])
        for row in reference_report["participants"]
    }
    differences = {
        person: candidate_people[person] - reference_people[person] for person in candidate_people
    }
    output.mkdir(parents=True)
    np.savez_compressed(
        output / "predictions.npz",
        labels=labels,
        participant_ids=participants,
        window_ids=windows,
        gravity_invariant_probabilities=probabilities,
        a2_probabilities=np.asarray(cached["a2_probabilities"], dtype=np.float64),
    )
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "gravity_invariant_followup_result",
        "experiment_id": "gravity-invariant-followup-v1",
        "status": "complete_exploratory_source_development",
        "evidence_status": "reused_source_development_not_independent",
        "source_commit": source_commit,
        "target_performance_or_prediction_accessed": False,
        "configuration": {
            "candidate": "A2 denoised GSP plus predeclared scalar gravity-relative dynamics",
            "reference": "A2 cached denoised GSP",
            "estimator": "ExtraTreesClassifier",
            "n_estimators": 500,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "random_state": SEED,
            "class_weight": None,
            "weighting": "1/(present_person_class_count * training_person_class_window_count), normalized mean one",
            "outer_folds": [list(pair) for pair in PARTICIPANT_PAIRS],
            "selected_invariant_groups": list(INVARIANT_GROUPS),
            "selected_feature_count": len(selected_indices),
            "feature_count": int(x_followup.shape[1]),
        },
        "folds": fold_records,
        "candidate": candidate_report,
        "reference_a2": reference_report,
        "paired_participant_macro_f1_difference": differences,
        "paired_bootstrap": paired_bootstrap(candidate_people, reference_people),
        "participant_wins": int(sum(value > 0.0 for value in differences.values())),
        "participant_harms": int(sum(value < 0.0 for value in differences.values())),
        "participant_ties": int(sum(value == 0.0 for value in differences.values())),
        "runtime_seconds": time.perf_counter() - started,
        "artifacts": {"predictions": "predictions.npz"},
        "input_hashes": {
            "baseline_cache": sha256_file(baseline_cache),
            "gravity_cache": sha256_file(gravity_cache),
            "baseline_predictions": sha256_file(baseline_predictions),
        },
        "software": {
            "numpy": importlib.metadata.version("numpy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
        "interpretation": "This is a single predeclared mechanism test. It can diagnose compact rotation-invariant gravity-relative dynamics, but cannot establish independent confirmation or novelty.",
    }
    result["record_sha256"] = canonical_json_sha256(result)
    with (output / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    manifest = {
        "schema_version": "1.0.0",
        "record_kind": "gravity_invariant_followup_manifest",
        "artifacts": [
            {
                "path": "result.json",
                "sha256": sha256_file(output / "result.json"),
                "size_bytes": (output / "result.json").stat().st_size,
            },
            {
                "path": "predictions.npz",
                "sha256": sha256_file(output / "predictions.npz"),
                "size_bytes": (output / "predictions.npz").stat().st_size,
            },
        ],
    }
    manifest["record_sha256"] = canonical_json_sha256(manifest)
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--gravity-cache", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    result = run_followup(
        baseline_cache=args.baseline_cache.resolve(),
        gravity_cache=args.gravity_cache.resolve(),
        baseline_predictions=args.baseline_predictions.resolve(),
        output=args.output.resolve(),
        source_commit=args.source_commit,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "mean_difference": result["paired_bootstrap"]["mean_difference"],
                "runtime_seconds": result["runtime_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
