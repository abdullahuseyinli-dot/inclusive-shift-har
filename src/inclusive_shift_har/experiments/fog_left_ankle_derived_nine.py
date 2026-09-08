"""Execute and independently replay the bounded corrected-FoG L9v closure probe."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import pickle
import platform
import sys
import threading
import time
import traceback
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import sklearn  # type: ignore[import-untyped]
import yaml
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
from threadpoolctl import threadpool_info, threadpool_limits  # type: ignore[import-untyped]

from inclusive_shift_har.experiments.fog_rf_feature_weight_factorial import (
    _array_sha256,
    _git_state,
    _mapping,
    _predict_proba_deterministically,
    _require,
    _sealed,
    _verify_sealed,
    _write_bytes_create_only,
    _write_json_create_only,
    method_report,
    paired_comparison,
    participant_first_weights,
)
from inclusive_shift_har.experiments.fog_spatial_information_probe import (
    ANKLE_FEATURE_SHA256,
    ANKLE_GRAVITY_SHA256,
    ANKLE_SIGNAL_SHA256,
    MASK_SHA256,
    ORDERED_ID_SHA256,
    OUTER_FOLDS,
    SIX_CHANNEL_NAMES,
    _availability_array_sha256,
    _check_cancel,
    _events,
    _leave_one_fold,
    _materialize_ankle_windows,
    _verify_manifest,
    apply_back_fallback,
    availability_mask_from_rows,
    fold_masks,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.preprocessing.features import extract_engineered_features

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
StringArray: TypeAlias = NDArray[np.str_]

EXPERIMENT_ID = "fog-left-ankle-derived-nine-probe-v1"
RUN_DIRECTORY_NAME = "fog-left-ankle-derived-nine-seed11-20260908-003"
EVIDENCE_FAMILY = ".audit/fog_left_ankle_derived_nine"
SPATIAL_RELATIVE = Path(
    ".audit/fog_spatial_information_probe/fog-spatial-information-probe-seed11-20260907-001"
)
PLAN_RELATIVE = Path(".audit/fog_spatial_team_review_20260907-001/NEXT_EXPERIMENT_SPECIFICATION.md")
CONFIG_RELATIVE = Path("configs/experiments/fog_left_ankle_derived_nine_v1.yaml")
PROTOCOL_RELATIVE = Path("docs/research/FOG_LEFT_ANKLE_DERIVED_NINE_V1_PROTOCOL.md")
EXPECTED_CONFIG_RECORD_SHA256 = "4563a64c7f85deb842547ebdab6c4dcf53507246cca2d40fd4c58d340ca2b21e"
PLAN_SHA256 = "4bbcd2388518b514f98ba912af84d5378f7782aa902f781243dcbfe1587a22fb"
RAW_SHA256 = "888875133fef386affddd0a5864f122d527d8d7bc588a69905cc0871bef53477"
RAW_SIZE = 119_629_580
COVERAGE_SHA256 = "16661d1e48b6b202a2295ccb2c258944d16d421024d0a7af506aa291a5ae6f14"
AVAILABILITY_SHA256 = "ba80d1ff36567ed7d9dbac0ebb0a885fddabce22bac0e81968c1fa64d98a533d"
SPATIAL_HASHES = {
    "predictions.npz": "f28ed71abb51d04f5968cfd3f3751e26dcac0f420e3451390b88b27654f0b1ef",
    "feature_cache.npz": "d92f350128655dc70f7bf60a2773577a907d1688146d48e5c35723a426893088",
    "analysis.json": "051cf2d9e40df748378fbd91a7ea2e524234d5f5579e47d993fe2801418cdb1a",
    "validation.json": "2cc1211645a62b88a7a56e68d0df7811cc397ab9f10c6f9bb2779814c5b32f98",
    "completion_manifest.json": (
        "4b014e2a085b2133db5a37e61ef7d204b0ca26cc2ce9fe5613c1ea43663eefd1"
    ),
}
NINE_CHANNEL_NAMES = (*SIX_CHANNEL_NAMES, "gravity_x", "gravity_y", "gravity_z")
EXPECTED_RESTRICTED_TRAINING_ROWS = (940, 825, 947, 829, 1075)
EXPECTED_AVAILABLE_EVALUATION_ROWS = (408, 516, 314, 447, 132)
EXPECTED_WEIGHT_HASHES = (
    "2968090d2f4d7860e6a21b43ff031f4907b7841615ef3bb72f5c5e3d97842b32",
    "22e9149cb58539cfc8add835271324609737098b9a357b0509764f47abbf128f",
    "78b5a3416d3df97e35cdbdda5d18d0508919852bbb734c7a24d9a57ac01b9a1e",
    "a4d71b366138469dc6c040b6416944676e6bc1711bd8336e6264a1dd1cce399b",
    "b0ec80a767e3191057f1b7f935b35de59f3259b8514a135efd05c357ff99fc98",
)
EXPECTED_TRAINING_ID_HASHES = (
    "ce4f6cef8a553e13145a9ecbe8afcc6da8247207b7922cf0b04f1f64985d358f",
    "15303cf535de5abea2672b00b8fb23cf9442e13172f99a8e59deec31e51492b7",
    "6a4d5d09d70d13ed2a6493a02f2c7aefa673cbd6330f6d6f4d032134540f8dce",
    "550a2f7b2e3dd27c255c6895024be8824686307f430197aeb964e87dbb222244",
    "bb97f7981f8a3ff67b47f76599fe45b7fa788535e94329a336914ae05fe7c52b",
)
EXPECTED_EVALUATION_ID_HASHES = (
    "ae6eb9dfb4a67dcfba49c4958f51cf5966a96293db1c1b52720cdd312370206b",
    "750e6d5b48ef5e19c666099f3d0d7bda58aec5d0d14672e4480efcc8889cf606",
    "3caee5f4000f9f4414f7c9cb6ff3eba9ff54e991adae10a10de2cf32970fc568",
    "ef6cc8005833ba9b4ead6cdd81aeccde3495ed7cbf764451b94df36a0b63e5e5",
    "5b0d1c9170272a96c6227d5edd1883fa782d9f34d9554d83196469aa2fc9f266",
)
EXPECTED_GENERIC_MASK_SHA256 = "c003da099d835da78f6d989469ff0dd3731e3bf037f868c4c2577c098ff9e6dd"
EXPECTED_L9V_FEATURE_SHA256 = "b1b38b52e66d10b31c6d70a689f90e917f81e269340534fb422d25d9514fdc70"
EXPECTED_L9V_NAMES_ARRAY_SHA256 = "cef905a4c45655db3a50016e9405ea47f8ae3019db7b967195ddc2d96c59f257"
EXPECTED_L9V_NAMES_CANONICAL_SHA256 = (
    "1cea223cd9fa2df361c3061672e846a81d6ad5b667a1c148961497ffe44ce4ef"
)
CONTROL_ORDER = ("lv", "bv", "b0", "f3")
METHOD_ORDER = ("l9v", *CONTROL_ORDER)
B0_METHOD_INDEX = METHOD_ORDER.index("b0")
MAXIMUM_FIT_ATTEMPTS = 5
COMPUTE_CAP_SECONDS = 900


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _read_yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), str(path))


def _read_npz(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _write_npz_create_only(path: Path, **arrays: NDArray[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]


def _effective_parameters(random_state: int, n_jobs: int = 4) -> dict[str, Any]:
    estimator = RandomForestClassifier(
        n_estimators=500,
        criterion="gini",
        max_features="sqrt",
        min_samples_leaf=2,
        min_samples_split=2,
        bootstrap=True,
        class_weight=None,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return cast(dict[str, Any], estimator.get_params(deep=False))


def validate_config(config: Mapping[str, Any]) -> None:
    _require(
        canonical_json_sha256(dict(config)) == EXPECTED_CONFIG_RECORD_SHA256,
        "frozen L9v config changed",
    )


def _deadline_check(deadline: float, stage: str) -> None:
    _require(time.perf_counter() <= deadline, f"900-second compute cap exceeded at {stage}")


def _source_manifest(repository_root: Path) -> dict[str, Any]:
    paths = (
        "AGENTS.md",
        "src/inclusive_shift_har/data/external_har.py",
        "src/inclusive_shift_har/preprocessing/features.py",
        "src/inclusive_shift_har/experiments/fog_rf_feature_weight_factorial.py",
        "src/inclusive_shift_har/experiments/fog_spatial_information_probe.py",
        "src/inclusive_shift_har/experiments/fog_left_ankle_derived_nine.py",
        CONFIG_RELATIVE.as_posix(),
        PROTOCOL_RELATIVE.as_posix(),
    )
    return {
        "record_kind": "fog_l9v_source_manifest",
        "files": [{"path": path, "sha256": sha256_file(repository_root / path)} for path in paths],
    }


def _snapshot_inputs(
    repository_root: Path,
    evidence_root: Path,
    config_path: Path,
    protocol_path: Path,
    output_directory: Path,
) -> None:
    _write_bytes_create_only(output_directory / "config_snapshot.yaml", config_path.read_bytes())
    _write_json_create_only(
        output_directory / "protocol_snapshot.json",
        _sealed(
            {
                "record_kind": "fog_l9v_protocol_snapshot",
                "source_path": protocol_path.relative_to(repository_root).as_posix(),
                "source_sha256": sha256_file(protocol_path),
                "text": protocol_path.read_text(encoding="utf-8"),
            }
        ),
    )
    plan = evidence_root / PLAN_RELATIVE
    _require(plan.is_file() and sha256_file(plan) == PLAN_SHA256, "frozen L9v plan changed")
    _write_json_create_only(
        output_directory / "plan_snapshot.json",
        _sealed(
            {
                "record_kind": "fog_l9v_plan_snapshot",
                "source_path": str(plan),
                "source_sha256": PLAN_SHA256,
                "text": plan.read_text(encoding="utf-8"),
            }
        ),
    )


def _worker_baseline() -> dict[str, set[int]]:
    return {
        "child_pids": {
            int(child.pid)
            for child in multiprocessing.active_children()
            if child.is_alive() and child.pid is not None
        },
        "thread_object_ids": {id(thread) for thread in threading.enumerate()},
    }


def _stop_task_owned_workers(
    baseline: Mapping[str, set[int]], attempts: int, completed: int, *, terminal: str
) -> dict[str, Any]:
    baseline_pids = baseline["child_pids"]
    baseline_threads = baseline["thread_object_ids"]
    children = [
        child
        for child in multiprocessing.active_children()
        if child.is_alive() and child.pid is not None and int(child.pid) not in baseline_pids
    ]
    terminated: list[int] = []
    for child in children:
        try:
            if child.pid is not None:
                terminated.append(int(child.pid))
            child.terminate()
            child.join(timeout=2.0)
            if child.is_alive():
                child.kill()
                child.join(timeout=2.0)
        except BaseException:
            continue
    threads = [
        thread
        for thread in threading.enumerate()
        if thread.is_alive() and id(thread) not in baseline_threads
    ]
    for thread in threads:
        try:
            thread.join(timeout=2.0)
        except BaseException:
            continue
    remaining_children = [
        child
        for child in multiprocessing.active_children()
        if child.is_alive() and child.pid is not None and int(child.pid) not in baseline_pids
    ]
    remaining_threads = [
        thread
        for thread in threading.enumerate()
        if thread.is_alive() and id(thread) not in baseline_threads
    ]
    return _sealed(
        {
            "record_kind": "fog_l9v_worker_shutdown",
            "observed_at_utc": _now(),
            "terminal_status": terminal,
            "fit_attempt_count": attempts,
            "completed_fit_count": completed,
            "baseline_multiprocessing_child_pids": sorted(baseline_pids),
            "task_owned_multiprocessing_child_pids_observed": sorted(
                int(child.pid) for child in children if child.pid is not None
            ),
            "task_owned_multiprocessing_child_pids_terminated": sorted(terminated),
            "remaining_task_owned_multiprocessing_child_pids": sorted(
                int(child.pid) for child in remaining_children if child.pid is not None
            ),
            "task_owned_nonbaseline_python_threads_observed": sorted(
                thread.name for thread in threads
            ),
            "remaining_task_owned_nonbaseline_python_threads": sorted(
                thread.name for thread in remaining_threads
            ),
            "task_owned_fit_workers_and_monitors_stopped": (
                not remaining_children and not remaining_threads
            ),
        }
    )


def _verify_spatial_reference(evidence_root: Path) -> tuple[Path, dict[str, Any]]:
    run = (evidence_root / SPATIAL_RELATIVE).resolve()
    _require(run.is_dir(), "sealed spatial reference run missing")
    rows: list[dict[str, Any]] = []
    for name, expected in SPATIAL_HASHES.items():
        path = run / name
        _require(
            path.is_file() and sha256_file(path) == expected, f"spatial reference changed: {name}"
        )
        rows.append({"path": str(path), "sha256": expected, "size_bytes": path.stat().st_size})
    _require(
        _read_json(run / "validation.json")["status"] == "validated", "spatial validation absent"
    )
    completion_count = _verify_manifest(run, "completion_manifest.json")
    return run, {
        "record_kind": "fog_l9v_spatial_reference_receipt",
        "checked_files": rows,
        "completion_manifest_entries_verified": completion_count,
        "source_run_status": "complete_and_independently_replayed",
    }


def _input_paths(spatial_run: Path) -> dict[str, Path]:
    manifest = _read_json(spatial_run / "input_manifest.json")
    _verify_sealed(manifest, "spatial input manifest")
    paths: dict[str, Path] = {}
    expected = {
        "raw_source": (RAW_SHA256, RAW_SIZE),
        "coverage": (COVERAGE_SHA256, None),
        "availability": (AVAILABILITY_SHA256, None),
    }
    for name, (expected_hash, expected_size) in expected.items():
        item = _mapping(manifest[name], name)
        path = Path(str(item["path"])).resolve()
        _require(path.is_file() and sha256_file(path) == expected_hash, f"{name} changed")
        if expected_size is not None:
            _require(path.stat().st_size == expected_size, f"{name} size changed")
        paths[name] = path
    return paths


def coverage_rows_from_audit(coverage: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the exact legacy coverage contract without inventing a self-hash field."""
    _require(
        coverage.get("record_kind") == "fog_spatial_information_coverage_audit"
        and coverage.get("status") == "complete_zero_fit_observable_coverage_audit"
        and coverage.get("model_fits") == 0
        and coverage.get("activity_fog_clinical_outcome_columns_accessed") is False
        and coverage.get("raw_sha256") == RAW_SHA256
        and coverage.get("raw_size_bytes") == RAW_SIZE,
        "coverage audit contract changed",
    )
    rows = cast(list[dict[str, Any]], coverage.get("candidate_availability"))
    _require(len(rows) == 1939, "coverage audit candidate count changed")
    return rows


def _quantiles(values: FloatArray) -> dict[str, float]:
    points = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
    result = np.quantile(values, points)
    return {
        f"p{int(point * 100):02d}": float(value)
        for point, value in zip(points, result, strict=True)
    }


def measurement_screen(
    *,
    signal: FloatArray,
    gravity: FloatArray,
    l9v_values: FloatArray,
    l9v_names: Sequence[str],
    ankle80_values: FloatArray,
    ankle80_names: Sequence[str],
    availability: BoolArray,
    observable_people: StringArray,
    segment_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the fixed label-blind measurement description; it has no tuning thresholds."""

    _require(signal.shape == (1817, 128, 6), "qualified signal shape changed")
    _require(gravity.shape == (1817, 128, 3), "qualified gravity shape changed")
    _require(l9v_values.shape == (1817, 120), "L9v feature shape changed")
    _require(len(l9v_names) == 120 and len(set(l9v_names)) == 120, "L9v names changed")
    _require(ankle80_values.shape == (1817, 80), "ankle80 shape changed")
    shared = name_aligned_features(l9v_values, l9v_names, ankle80_names)
    _require(np.array_equal(shared, ankle80_values), "name-aligned ankle80 features changed")
    _require(_array_sha256(shared) == ANKLE_FEATURE_SHA256, "shared ankle80 hash changed")
    norms = np.linalg.vector_norm(gravity, axis=2)
    unit = np.divide(
        gravity,
        norms[:, :, None],
        out=np.zeros_like(gravity),
        where=norms[:, :, None] > 0,
    )
    direction_stability = np.linalg.vector_norm(unit.mean(axis=1), axis=1)
    participants = np.unique(observable_people)
    physical_sessions = {
        (str(row["participant_id"]), str(row["session_id"])) for row in segment_receipts
    }
    per_person = []
    for person in participants:
        person_mask = observable_people == person
        per_person.append(
            {
                "participant_id": str(person),
                "observable_candidate_count": int(person_mask.sum()),
                "available_candidate_count": int((person_mask & availability).sum()),
                "fallback_candidate_count": int((person_mask & ~availability).sum()),
            }
        )
    source_runs = len(segment_receipts)
    finite_rates = {}
    for name, values in (
        ("signal", signal),
        ("gravity", gravity),
        ("l9v_features", l9v_values),
    ):
        finite_count = int(np.isfinite(values).sum())
        total_count = int(values.size)
        finite_rates[name] = {
            "finite_count": finite_count,
            "total_count": total_count,
            "finite_fraction": finite_count / total_count,
        }
    return {
        "record_kind": "fog_l9v_label_blind_measurement_screen",
        "created_before_label_or_outcome_probability_load": True,
        "screen_role": "integrity_and_description_only",
        "selection_or_tuning_permitted": False,
        "all_signal_values_finite": bool(np.isfinite(signal).all()),
        "all_gravity_values_finite": bool(np.isfinite(gravity).all()),
        "all_feature_values_finite": bool(np.isfinite(l9v_values).all()),
        "finite_rates": finite_rates,
        "signal_array_sha256": _array_sha256(signal),
        "gravity_array_sha256": _array_sha256(gravity),
        "availability_contract_sha256": _availability_array_sha256(availability),
        "availability_generic_array_sha256": _array_sha256(availability),
        "l9v_feature_array_sha256": _array_sha256(l9v_values),
        "l9v_feature_names_array_sha256": _array_sha256(np.asarray(l9v_names, dtype=np.str_)),
        "l9v_feature_names_canonical_list_sha256": canonical_json_sha256(list(l9v_names)),
        "name_aligned_shared_ankle80_sha256": _array_sha256(shared),
        "gravity_norm_m_per_s2": {
            "mean": float(norms.mean()),
            "standard_deviation": float(norms.std(ddof=0)),
            "quantiles": _quantiles(norms.reshape(-1)),
        },
        "within_window_gravity_direction_resultant_length": {
            "definition": "norm of mean per-sample unit gravity vector; range zero to one",
            "mean": float(direction_stability.mean()),
            "standard_deviation": float(direction_stability.std(ddof=0)),
            "quantiles": _quantiles(direction_stability),
        },
        "timestamp_and_physical_run_boundaries": {
            "session_count": len(physical_sessions),
            "finite_monotonic_resampled_run_count": source_runs,
            "additional_run_boundaries_beyond_sessions": source_runs - len(physical_sessions),
            "run_receipt_sha256": canonical_json_sha256(list(segment_receipts)),
        },
        "per_participant_availability": per_person,
        "observable_available_count": int(availability.sum()),
        "observable_fallback_count": int((~availability).sum()),
        "integrity_status": "pass",
    }


def name_aligned_features(
    values: FloatArray,
    source_names: Sequence[str],
    requested_names: Sequence[str],
) -> FloatArray:
    """Select shared features by name while retaining the requested schema order."""

    _require(len(source_names) == values.shape[1], "source feature names misalign")
    _require(len(set(source_names)) == len(source_names), "source feature names repeat")
    _require(len(set(requested_names)) == len(requested_names), "requested names repeat")
    lookup = {name: index for index, name in enumerate(source_names)}
    _require(all(name in lookup for name in requested_names), "requested feature missing")
    return np.asarray(values[:, [lookup[name] for name in requested_names]], dtype=np.float64)


def partition_preflight(
    *,
    labels: IntArray,
    people: StringArray,
    folds: IntArray,
    eligibility: BoolArray,
    availability: BoolArray,
) -> dict[str, Any]:
    observable_labels = np.zeros(people.size, dtype=np.int64)
    observable_labels[eligibility] = labels
    rows = []
    for fold in OUTER_FOLDS:
        masks = fold_masks(folds, eligibility, availability, fold)
        training = masks["restricted_training"]
        evaluation = masks["restricted_evaluation"]
        training_ids = np.flatnonzero(training).astype(np.int64)
        evaluation_ids = np.flatnonzero(evaluation).astype(np.int64)
        weights = participant_first_weights(observable_labels[training], people[training])
        _require(
            int(training.sum()) == EXPECTED_RESTRICTED_TRAINING_ROWS[fold], "training rows changed"
        )
        _require(
            int(evaluation.sum()) == EXPECTED_AVAILABLE_EVALUATION_ROWS[fold],
            "available evaluation rows changed",
        )
        _require(_array_sha256(weights) == EXPECTED_WEIGHT_HASHES[fold], "weight hash changed")
        _require(
            _array_sha256(training_ids) == EXPECTED_TRAINING_ID_HASHES[fold],
            "Lv training membership changed",
        )
        _require(
            _array_sha256(evaluation_ids) == EXPECTED_EVALUATION_ID_HASHES[fold],
            "Lv evaluation membership changed",
        )
        _require(
            set(np.unique(people[training])).isdisjoint(np.unique(people[folds == fold])),
            "participant leakage",
        )
        _require(set(np.unique(observable_labels[training]).tolist()) == {0, 1, 2}, "class missing")
        rows.append(
            {
                "outer_fold": fold,
                "training_row_count": int(training.sum()),
                "available_evaluation_candidate_count": int(evaluation.sum()),
                "training_class_counts": np.bincount(
                    observable_labels[training], minlength=3
                ).tolist(),
                "training_participants": sorted(np.unique(people[training]).tolist()),
                "evaluation_participants": sorted(np.unique(people[folds == fold]).tolist()),
                "training_global_indices_sha256": _array_sha256(training_ids),
                "evaluation_global_indices_sha256": _array_sha256(evaluation_ids),
                "sample_weight_sha256": _array_sha256(weights),
                "sample_weight_mean": float(weights.mean()),
            }
        )
    return {
        "record_kind": "fog_l9v_partition_preflight",
        "created_before_model_fits": True,
        "partition_count": 5,
        "fit_partition_count": 5,
        "rows": rows,
    }


def _fit_l9v(
    *,
    training_values: FloatArray,
    training_labels: IntArray,
    training_participants: StringArray,
    evaluation_values: FloatArray,
    evaluation_participants: StringArray,
    training_global_indices: IntArray,
    evaluation_global_indices: IntArray,
    outer_fold: int,
    attempt_number: int,
    feature_names: tuple[str, ...],
    output_directory: Path,
    deadline: float,
) -> tuple[FloatArray, dict[str, Any]]:
    _require(1 <= attempt_number <= MAXIMUM_FIT_ATTEMPTS, "five-attempt budget exhausted")
    _require(attempt_number == outer_fold + 1, "fit order changed")
    _deadline_check(deadline, f"before L9v fold {outer_fold}")
    weights = participant_first_weights(training_labels, training_participants)
    _require(_array_sha256(weights) == EXPECTED_WEIGHT_HASHES[outer_fold], "fit weights changed")
    started = _sealed(
        {
            "record_kind": "fog_l9v_fit_attempt_started",
            "attempt_number": attempt_number,
            "cell_id": "l9v",
            "outer_fold": outer_fold,
            "started_at_utc": _now(),
            "random_state": 11 + outer_fold,
            "training_row_count": int(training_labels.size),
            "evaluation_candidate_count": int(evaluation_values.shape[0]),
        }
    )
    _write_json_create_only(
        output_directory / "fit_attempts" / f"{attempt_number:02d}--started.json", started
    )
    estimator = RandomForestClassifier(
        n_estimators=500,
        criterion="gini",
        max_features="sqrt",
        min_samples_leaf=2,
        min_samples_split=2,
        bootstrap=True,
        class_weight=None,
        random_state=11 + outer_fold,
        n_jobs=4,
    )
    wall_started = time.perf_counter()
    with threadpool_limits(limits=1):
        fit_started = time.perf_counter()
        estimator.fit(training_values, training_labels, sample_weight=weights)
        fit_seconds = time.perf_counter() - fit_started
        predict_started = time.perf_counter()
        probabilities = _predict_proba_deterministically(estimator, evaluation_values)
        prediction_seconds = time.perf_counter() - predict_started
    elapsed = time.perf_counter() - wall_started
    _deadline_check(deadline, f"after L9v fold {outer_fold}")
    _require(np.array_equal(estimator.classes_, np.arange(3)), "class order changed")
    _require(
        probabilities.shape == (evaluation_values.shape[0], 3)
        and bool(np.isfinite(probabilities).all())
        and bool(np.all(probabilities >= 0))
        and np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12),
        "invalid probabilities",
    )
    metadata: dict[str, Any] = {
        "attempt_number": attempt_number,
        "cell_id": "l9v",
        "outer_fold": outer_fold,
        "random_state": 11 + outer_fold,
        "feature_names": list(feature_names),
        "feature_names_sha256": canonical_json_sha256(list(feature_names)),
        "training_participants": sorted(np.unique(training_participants).tolist()),
        "evaluation_participants": sorted(np.unique(evaluation_participants).tolist()),
        "training_row_count": int(training_labels.size),
        "training_class_counts": np.bincount(training_labels, minlength=3).tolist(),
        "evaluation_candidate_count": int(evaluation_values.shape[0]),
        "training_global_indices_sha256": _array_sha256(training_global_indices),
        "evaluation_global_indices_sha256": _array_sha256(evaluation_global_indices),
        "sample_weight_policy": "participant_first_mean_one",
        "sample_weight_sha256": _array_sha256(weights),
        "effective_parameters": estimator.get_params(deep=False),
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "fit_and_predict_seconds": elapsed,
        "prediction_worker_count": 1,
    }
    checkpoint_path = output_directory / "checkpoints" / f"l9v--fold-{outer_fold}.pkl"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("xb") as stream:
        pickle.dump({"metadata": metadata, "estimator": estimator}, stream, protocol=5)
    metadata["checkpoint"] = {
        "path": checkpoint_path.relative_to(output_directory).as_posix(),
        "sha256": sha256_file(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
    }
    completed = _sealed(
        {
            "record_kind": "fog_l9v_fit_attempt_completed",
            "attempt_number": attempt_number,
            "cell_id": "l9v",
            "outer_fold": outer_fold,
            "completed_at_utc": _now(),
            "fit_seconds": fit_seconds,
            "prediction_seconds": prediction_seconds,
            "checkpoint": metadata["checkpoint"],
        }
    )
    _write_json_create_only(
        output_directory / "fit_attempts" / f"{attempt_number:02d}--completed.json", completed
    )
    return probabilities, metadata


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], RandomForestClassifier]:
    with path.open("rb") as stream:
        value = pickle.load(stream)
    _require(isinstance(value, dict), "checkpoint payload changed")
    metadata = _mapping(value.get("metadata"), "checkpoint metadata")
    estimator = value.get("estimator")
    _require(isinstance(estimator, RandomForestClassifier), "checkpoint estimator changed")
    return metadata, estimator


def _gate(
    comparison: Mapping[str, Any],
    leave_fold: Sequence[Mapping[str, Any]],
    *,
    source_reference_exact: bool,
) -> dict[str, Any]:
    recalls = _mapping(comparison["class_recall_differences"], "class recall differences")
    checks = {
        "full_22_person_roster_scored": True,
        "source_reference_and_fallback_exact": source_reference_exact,
        "mean_gain": float(comparison["mean_difference"]) >= 0.015,
        "participant_wins": int(comparison["participant_wins"]) >= 14,
        "bottom_30": float(comparison["bottom_30_percent_difference"]) >= -0.01,
        "worst_participant_minimum": float(comparison["worst_participant_difference"]) >= -0.03,
        "mobility_recall": float(recalls["mobility"]) >= -0.01,
        "sitting_recall": float(recalls["sitting"]) >= -0.02,
        "standing_recall": float(recalls["standing"]) >= -0.02,
        "all_leave_one_participant_out_positive": all(
            float(value) > 1e-12
            for value in cast(list[float], comparison["leave_one_participant_out_mean_differences"])
        ),
        "all_leave_one_fold_means_positive": all(
            float(row["mean_participant_difference"]) > 1e-12 for row in leave_fold
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "passed_check_count": sum(checks.values()),
        "required_check_count": len(checks),
    }


def analyse(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    labels: IntArray,
    probabilities: Mapping[str, FloatArray],
    participants: StringArray,
    participant_folds: IntArray,
    roster: Sequence[str],
    source_reference_exact: bool,
) -> dict[str, Any]:
    _require(set(reports) == set(METHOD_ORDER), "analysis method set changed")
    comparisons: dict[str, dict[str, Any]] = {}
    leave_folds: dict[str, list[dict[str, Any]]] = {}
    gates: dict[str, dict[str, Any]] = {}
    events: dict[str, dict[str, Any]] = {}
    for control in CONTROL_ORDER:
        name = f"l9v_minus_{control}"
        comparisons[name] = paired_comparison(reports["l9v"], reports[control])
        leave_folds[name] = _leave_one_fold(reports["l9v"], reports[control], participant_folds)
        gates[name] = _gate(
            comparisons[name], leave_folds[name], source_reference_exact=source_reference_exact
        )
        events[name] = _events(
            labels, probabilities[control], probabilities["l9v"], participants, roster
        )
    passed = all(gate["status"] == "pass" for gate in gates.values())
    decision = (
        "pass_freeze_l9v_recommend_separately_authorized_seed_stability"
        if passed
        else "fail_close_static_ankle_gravity_prioritize_physical_information_pilot"
    )
    return {
        "status": "complete_analysis",
        "evidence_status": "corrected_fog_adaptive_development_closure_not_confirmation",
        "reports": dict(reports),
        "comparisons": comparisons,
        "leave_one_outer_fold_sensitivity": leave_folds,
        "gates": gates,
        "advancement": {
            "status": "pass" if passed else "fail",
            "required_comparisons": [f"l9v_minus_{name}" for name in CONTROL_ORDER],
            "minimum_absolute_mean_participant_macro_f1": 0.6045943466990131,
        },
        "event_topology": events,
        "decision": decision,
        "fallback_system_requires_back_sensor": True,
        "automatic_follow_on_launched": False,
        "architecture_built": False,
        "architecture_reason": (
            "dual-branch architecture remains conditional on separately collected "
            "physical-information evidence"
        ),
        "bootstrap_scope": (
            "participant resampling conditional on fitted overlapping-fold models and an "
            "adaptively consumed development cohort"
        ),
    }


def _outcome_summary(analysis: Mapping[str, Any]) -> str:
    reports = _mapping(analysis["reports"], "reports")
    lines = [
        "# Corrected FoG left-ankle derived-nine outcome",
        "",
        f"Decision: **{analysis['decision']}**.",
        "",
        "| Method | Mean participant F1 | Pooled accuracy | Bottom seven | Worst | NLL | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        report = _mapping(reports[method], method)
        primary = _mapping(report["primary"], "primary")
        pooled = _mapping(report["pooled"], "pooled")
        lines.append(
            f"| {method.upper()} | {float(primary['mean_participant_macro_f1']):.9f} | "
            f"{float(pooled['accuracy']):.9f} | "
            f"{float(primary['bottom_30_percent_participant_macro_f1']):.9f} | "
            f"{float(primary['worst_participant_macro_f1']):.9f} | "
            f"{float(primary['mean_participant_nll']):.9f} | "
            f"{float(primary['mean_participant_multiclass_brier']):.9f} |"
        )
    lines.extend(
        [
            "",
            "| Required comparison | F1 difference | 95% bootstrap interval | W/H/T | "
            "Maximum harm | Gate |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    comparisons = _mapping(analysis["comparisons"], "comparisons")
    gates = _mapping(analysis["gates"], "gates")
    for name in [f"l9v_minus_{control}" for control in CONTROL_ORDER]:
        comparison = _mapping(comparisons[name], name)
        interval = cast(list[float], comparison["mean_difference_95_percent_bootstrap_interval"])
        lines.append(
            f"| {name} | {float(comparison['mean_difference']):+.9f} | "
            f"[{float(interval[0]):+.9f}, {float(interval[1]):+.9f}] | "
            f"{int(comparison['participant_wins'])}/"
            f"{int(comparison['participant_harms'])}/"
            f"{int(comparison['participant_ties'])} | "
            f"{float(comparison['minimum_paired_participant_difference']):+.9f} | "
            f"{_mapping(gates[name], name)['status']} |"
        )
    lines.extend(
        [
            "",
            "The endpoint retains all 22 people and 1,213 scored rows. L9v uses exact B0",
            "fallback on 122 observable candidates, including 59 scored rows. This is adaptive",
            "development on a consumed cohort. No follow-on, seed expansion, architecture,",
            "publication, or target-cohort access was launched.",
            "",
        ]
    )
    return "\n".join(lines)


def _method_summary(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        method: {
            "mean_participant_macro_f1": reports[method]["primary"]["mean_participant_macro_f1"],
            "pooled_accuracy": reports[method]["pooled"]["accuracy"],
            "bottom_30_percent_participant_macro_f1": reports[method]["primary"][
                "bottom_30_percent_participant_macro_f1"
            ],
            "worst_participant_macro_f1": reports[method]["primary"]["worst_participant_macro_f1"],
            "mean_participant_nll": reports[method]["primary"]["mean_participant_nll"],
            "mean_participant_multiclass_brier": reports[method]["primary"][
                "mean_participant_multiclass_brier"
            ],
        }
        for method in METHOD_ORDER
    }


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "threadpool_info": threadpool_info(),
    }


def _cache_metadata(cache: Mapping[str, NDArray[Any]]) -> dict[str, Any]:
    return {
        "record_kind": "fog_l9v_feature_cache_metadata",
        "created_before_label_or_outcome_probability_load": True,
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _array_sha256(value),
            }
            for name, value in cache.items()
        },
    }


def _verify_fallback_topology(
    availability: BoolArray,
    people: StringArray,
    scoring_indices: IntArray,
) -> dict[str, Any]:
    observable = {
        str(person): int(((people == person) & ~availability).sum())
        for person in np.unique(people)
        if int(((people == person) & ~availability).sum()) > 0
    }
    scored_people = people[scoring_indices]
    scored_available = availability[scoring_indices]
    scored = {
        str(person): int(((scored_people == person) & ~scored_available).sum())
        for person in np.unique(scored_people)
        if int(((scored_people == person) & ~scored_available).sum()) > 0
    }
    _require(
        observable == {"fogstar:004": 65, "fogstar:005": 32, "fogstar:008": 25},
        "observable fallback topology changed",
    )
    _require(
        scored == {"fogstar:004": 42, "fogstar:005": 8, "fogstar:008": 9},
        "scored fallback topology changed",
    )
    return {
        "observable_fallback_by_participant": observable,
        "scored_fallback_by_participant": scored,
        "observable_fallback_count": int((~availability).sum()),
        "scored_fallback_count": int((~scored_available).sum()),
    }


def _artifact_manifest(output_directory: Path) -> dict[str, Any]:
    omitted = {
        "artifact_manifest.json",
        "validation_worker_shutdown.json",
        "validation.json",
        "completion_manifest.json",
        "controller_final_receipt.json",
        "VALIDATION_INCOMPLETE.json",
    }
    files = [
        path for path in output_directory.rglob("*") if path.is_file() and path.name not in omitted
    ]
    return _sealed(
        {
            "record_kind": "fog_l9v_artifact_manifest",
            "artifacts": [
                {
                    "path": path.relative_to(output_directory).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(files)
            ],
        }
    )


def _verify_artifact_manifest(output_directory: Path) -> int:
    manifest = _read_json(output_directory / "artifact_manifest.json")
    _verify_sealed(manifest, "L9v artifact manifest")
    _require(manifest["record_kind"] == "fog_l9v_artifact_manifest", "manifest kind changed")
    rows = cast(list[dict[str, Any]], manifest["artifacts"])
    listed = {str(row["path"]) for row in rows}
    _require(len(listed) == len(rows), "duplicate artifact path")
    for row in rows:
        path = (output_directory / str(row["path"])).resolve()
        _require(path.is_relative_to(output_directory), "artifact path escapes run")
        _require(
            path.is_file()
            and path.stat().st_size == int(row["size_bytes"])
            and sha256_file(path) == row["sha256"],
            f"artifact changed: {path}",
        )
    allowed_unlisted = {
        "artifact_manifest.json",
        "validation_worker_shutdown.json",
        "validation.json",
        "completion_manifest.json",
        "controller_final_receipt.json",
        "VALIDATION_INCOMPLETE.json",
    }
    actual = {
        path.relative_to(output_directory).as_posix()
        for path in output_directory.rglob("*")
        if path.is_file() and path.name not in allowed_unlisted
    }
    _require(actual == listed, "artifact manifest coverage changed")
    return len(rows)


def run_experiment(
    *,
    repository_root: Path,
    evidence_root: Path,
    config_path: Path,
    protocol_path: Path,
    output_directory: Path,
    code_commit: str,
    deadline: float,
    controller_started: float,
    cancel_file: Path | None,
) -> dict[str, Any]:
    baseline = _worker_baseline()
    attempts = 0
    completed = 0
    output_directory = output_directory.resolve()
    try:
        repository_root = repository_root.resolve()
        evidence_root = evidence_root.resolve()
        config_path = config_path.resolve()
        protocol_path = protocol_path.resolve()
        _require(repository_root.is_dir() and evidence_root.is_dir(), "root missing")
        _require(config_path == repository_root / CONFIG_RELATIVE, "config path changed")
        _require(protocol_path == repository_root / PROTOCOL_RELATIVE, "protocol path changed")
        _require(
            output_directory.parent == evidence_root / EVIDENCE_FAMILY
            and output_directory.name == RUN_DIRECTORY_NAME,
            "create-only L9v output location changed",
        )
        _require(not output_directory.exists(), "L9v run directory already exists")
        output_directory.mkdir(parents=True)
        config = _read_yaml(config_path)
        validate_config(config)
        _require(config["experiment_id"] == EXPERIMENT_ID, "experiment ID changed")
        git = _git_state(repository_root)
        _require(git["clean"] is True and git["commit"] == code_commit, "source must be clean")
        _snapshot_inputs(
            repository_root, evidence_root, config_path, protocol_path, output_directory
        )
        source = _sealed(_source_manifest(repository_root))
        _write_json_create_only(output_directory / "source_manifest.json", source)
        spatial_run, reference_body = _verify_spatial_reference(evidence_root)
        reference = _sealed(reference_body)
        _write_json_create_only(output_directory / "reference_receipt.json", reference)
        inputs = _input_paths(spatial_run)
        input_manifest = _sealed(
            {
                "record_kind": "fog_l9v_input_manifest",
                "raw_source": {
                    "path": str(inputs["raw_source"]),
                    "sha256": RAW_SHA256,
                    "size_bytes": RAW_SIZE,
                },
                "coverage": {"path": str(inputs["coverage"]), "sha256": COVERAGE_SHA256},
                "availability": {
                    "path": str(inputs["availability"]),
                    "sha256": AVAILABILITY_SHA256,
                },
                "spatial_reference_run": str(spatial_run),
                "spatial_reference_record_sha256": reference["record_sha256"],
            }
        )
        _write_json_create_only(output_directory / "input_manifest.json", input_manifest)
        _deadline_check(deadline, "after source and reference verification")

        coverage = _read_json(inputs["coverage"])
        coverage_rows = coverage_rows_from_audit(coverage)
        coverage_ids = np.asarray([str(row["window_id"]) for row in coverage_rows], dtype=np.str_)
        _require(
            _availability_array_sha256(coverage_ids) == ORDERED_ID_SHA256,
            "coverage candidate ID hash changed",
        )
        availability = availability_mask_from_rows(coverage_rows)
        availability_freeze = _sealed(
            {
                "record_kind": "fog_l9v_availability_freeze",
                "created_before_label_or_outcome_probability_load": True,
                "mask_contract_sha256": _availability_array_sha256(availability),
                "mask_generic_array_sha256": _array_sha256(availability),
                "ordered_observable_ids_contract_sha256": _availability_array_sha256(coverage_ids),
                "observable_available_count": int(availability.sum()),
                "observable_fallback_count": int((~availability).sum()),
            }
        )
        _require(availability_freeze["mask_contract_sha256"] == MASK_SHA256, "mask changed")
        _require(
            availability_freeze["mask_generic_array_sha256"]
            == "c003da099d835da78f6d989469ff0dd3731e3bf037f868c4c2577c098ff9e6dd",
            "generic mask hash changed",
        )
        _write_json_create_only(output_directory / "availability_freeze.json", availability_freeze)

        with np.load(spatial_run / "feature_cache.npz", allow_pickle=False) as archive:
            reference_availability = np.asarray(archive["availability_mask"], dtype=np.bool_)
            available_indices = np.asarray(archive["available_indices"], dtype=np.int64)
            ankle80_values = np.asarray(archive["ankle_values"], dtype=np.float64)
            ankle80_names = np.asarray(archive["ankle_names"], dtype=np.str_)
            window_ids = np.asarray(archive["observable_window_ids"], dtype=np.str_)
            people = np.asarray(archive["observable_participant_ids"], dtype=np.str_)
            folds = np.asarray(archive["observable_fold_index"], dtype=np.int64)
        _require(np.array_equal(availability, reference_availability), "reference mask differs")
        _require(np.array_equal(coverage_ids, window_ids), "coverage candidate IDs differ")
        _require(np.array_equal(available_indices, np.flatnonzero(availability)), "indices differ")
        _require(
            _array_sha256(availability)
            == "c003da099d835da78f6d989469ff0dd3731e3bf037f868c4c2577c098ff9e6dd",
            "mask array differs",
        )
        _require(_array_sha256(ankle80_values) == ANKLE_FEATURE_SHA256, "ankle80 changed")

        signal, gravity, fresh_indices, alignment_receipts, segment_receipts = (
            _materialize_ankle_windows(inputs["raw_source"], coverage_rows, availability)
        )
        _require(np.array_equal(fresh_indices, available_indices), "fresh indices differ")
        _require(_array_sha256(signal) == ANKLE_SIGNAL_SHA256, "ankle signal changed")
        _require(_array_sha256(gravity) == ANKLE_GRAVITY_SHA256, "ankle gravity changed")
        fresh_ankle80 = extract_engineered_features(signal, channel_names=SIX_CHANNEL_NAMES)
        _require(np.array_equal(fresh_ankle80.values, ankle80_values), "ankle80 replay changed")
        _require(fresh_ankle80.names == tuple(ankle80_names.tolist()), "ankle80 schema changed")
        nine_windows = np.concatenate((signal, gravity), axis=2)
        l9v = extract_engineered_features(nine_windows, channel_names=NINE_CHANNEL_NAMES)
        _require(_array_sha256(l9v.values) == EXPECTED_L9V_FEATURE_SHA256, "L9v matrix changed")
        _require(
            _array_sha256(np.asarray(l9v.names, dtype=np.str_)) == EXPECTED_L9V_NAMES_ARRAY_SHA256,
            "L9v name-array hash changed",
        )
        _require(
            canonical_json_sha256(list(l9v.names)) == EXPECTED_L9V_NAMES_CANONICAL_SHA256,
            "L9v canonical-name hash changed",
        )
        pre_outcome_cache = {
            "l9v_values": np.asarray(l9v.values, dtype=np.float64),
            "l9v_names": np.asarray(l9v.names, dtype=np.str_),
            "available_indices": available_indices,
            "availability_mask": availability,
            "observable_window_ids": window_ids,
            "observable_participant_ids": people,
            "observable_fold_index": folds,
        }
        _write_npz_create_only(output_directory / "feature_cache.npz", **pre_outcome_cache)
        cache_metadata = _sealed(_cache_metadata(pre_outcome_cache))
        _write_json_create_only(output_directory / "feature_cache_metadata.json", cache_metadata)
        screen = _sealed(
            measurement_screen(
                signal=signal,
                gravity=gravity,
                l9v_values=l9v.values,
                l9v_names=l9v.names,
                ankle80_values=ankle80_values,
                ankle80_names=ankle80_names.tolist(),
                availability=availability,
                observable_people=people,
                segment_receipts=segment_receipts,
            )
        )
        _require(
            screen["l9v_feature_array_sha256"] == EXPECTED_L9V_FEATURE_SHA256
            and screen["l9v_feature_names_array_sha256"] == EXPECTED_L9V_NAMES_ARRAY_SHA256
            and screen["l9v_feature_names_canonical_list_sha256"]
            == EXPECTED_L9V_NAMES_CANONICAL_SHA256,
            "measurement feature receipt changed",
        )
        _write_json_create_only(output_directory / "measurement_preflight.json", screen)
        _write_json_create_only(
            output_directory / "alignment_receipts.json",
            _sealed(
                {
                    "record_kind": "fog_l9v_alignment_receipts",
                    "alignment_receipt_count": len(alignment_receipts),
                    "alignment_receipts": alignment_receipts,
                    "segment_receipts": segment_receipts,
                }
            ),
        )
        _deadline_check(deadline, "after label-blind feature freeze")

        spatial_cache = _read_npz(spatial_run / "feature_cache.npz")
        spatial_predictions = _read_npz(spatial_run / "predictions.npz")
        _require(
            spatial_predictions["cell_ids"].tolist() == ["b0", "bv", "lv", "blv", "bbv"],
            "spatial cell order changed",
        )
        for name in (
            "availability_mask",
            "available_indices",
            "observable_window_ids",
            "observable_participant_ids",
            "observable_fold_index",
        ):
            _require(
                np.array_equal(spatial_cache[name], pre_outcome_cache[name]),
                f"post-freeze spatial array differs: {name}",
            )
        scoring_indices = np.asarray(spatial_cache["scoring_indices"], dtype=np.int64)
        eligibility = np.asarray(spatial_cache["scoring_eligibility"], dtype=np.bool_)
        labels = np.asarray(spatial_cache["scored_labels"], dtype=np.int64)
        scored_people = np.asarray(spatial_cache["scored_participant_ids"], dtype=np.str_)
        scored_window_ids = np.asarray(spatial_cache["scored_window_ids"], dtype=np.str_)
        scored_folds = np.asarray(spatial_cache["scored_fold_index"], dtype=np.int64)
        _require(np.array_equal(people[scoring_indices], scored_people), "scored people differ")
        _require(
            np.array_equal(window_ids[scoring_indices], scored_window_ids), "scored IDs differ"
        )
        _require(
            labels.shape == (1213,) and np.bincount(labels, minlength=3).tolist() == [954, 74, 185],
            "label support changed",
        )
        topology = _verify_fallback_topology(availability, people, scoring_indices)
        _require(
            topology["observable_fallback_count"] == 122
            and topology["scored_fallback_count"] == 59,
            "fallback count changed",
        )

        controls = {
            "b0": np.asarray(spatial_predictions["observable_probabilities"][0], dtype=np.float64),
            "bv": np.asarray(spatial_predictions["observable_probabilities"][1], dtype=np.float64),
            "lv": np.asarray(spatial_predictions["observable_probabilities"][2], dtype=np.float64),
            "f3": np.asarray(spatial_predictions["f3_reference_probabilities"], dtype=np.float64),
        }
        for name in ("bv", "lv"):
            _require(
                np.array_equal(controls[name][~availability], controls["b0"][~availability]),
                f"{name} fallback differs",
            )
        partitions = partition_preflight(
            labels=labels,
            people=people,
            folds=folds,
            eligibility=eligibility,
            availability=availability,
        )
        partition_record = _sealed(partitions)
        _write_json_create_only(output_directory / "partition_preflight.json", partition_record)
        environment = _environment()
        reference_environment = _mapping(
            _read_json(spatial_run / "preflight.json")["environment"],
            "spatial reference environment",
        )
        _require(
            str(environment["python"]).startswith("3.11.9 ")
            and environment["numpy"] == "2.3.5"
            and environment["scikit_learn"] == "1.8.0"
            and environment["python"] == reference_environment["python"]
            and Path(str(environment["python_executable"])).resolve()
            == Path(str(reference_environment["python_executable"])).resolve(),
            "runtime differs from sealed spatial reference",
        )
        preflight = _sealed(
            {
                "record_kind": "fog_l9v_preflight",
                "created_before_model_fits": True,
                "repository_root": str(repository_root),
                "evidence_root": str(evidence_root),
                "code_commit": code_commit,
                "git": git,
                "environment": environment,
                "configuration_sha256": sha256_file(config_path),
                "protocol_sha256": sha256_file(protocol_path),
                "plan_sha256": PLAN_SHA256,
                "source_manifest_record_sha256": source["record_sha256"],
                "reference_record_sha256": reference["record_sha256"],
                "measurement_record_sha256": screen["record_sha256"],
                "feature_cache_record_sha256": cache_metadata["record_sha256"],
                "partition_record_sha256": partition_record["record_sha256"],
                "expected_fit_count": 5,
                "effective_parameter_sets": {
                    f"l9v--fold-{fold}": _effective_parameters(11 + fold) for fold in OUTER_FOLDS
                },
                "new_candidate_outcomes_available_when_feature_cache_written": False,
                "scored_labels_loaded_after_availability_and_feature_freeze": True,
                "compute_cap_seconds": COMPUTE_CAP_SECONDS,
            }
        )
        _write_json_create_only(output_directory / "preflight.json", preflight)
        _deadline_check(deadline, "before fits")

        l9v_available = np.full((1817, 3), np.nan, dtype=np.float64)
        compact_lookup = np.full(1939, -1, dtype=np.int64)
        compact_lookup[available_indices] = np.arange(1817, dtype=np.int64)
        observable_labels = np.zeros(1939, dtype=np.int64)
        observable_labels[eligibility] = labels
        fit_rows: list[dict[str, Any]] = []
        for fold in OUTER_FOLDS:
            _check_cancel(cancel_file, f"before_l9v_fold_{fold}")
            masks = fold_masks(folds, eligibility, availability, fold)
            train_global = masks["restricted_training"]
            eval_global = masks["restricted_evaluation"]
            train_compact = compact_lookup[np.flatnonzero(train_global)]
            eval_compact = compact_lookup[np.flatnonzero(eval_global)]
            _require(
                bool(np.all(train_compact >= 0) and np.all(eval_compact >= 0)),
                "compact lookup failed",
            )
            attempts += 1
            probabilities, metadata = _fit_l9v(
                training_values=l9v.values[train_compact],
                training_labels=observable_labels[train_global],
                training_participants=people[train_global],
                evaluation_values=l9v.values[eval_compact],
                evaluation_participants=people[eval_global],
                training_global_indices=np.flatnonzero(train_global).astype(np.int64),
                evaluation_global_indices=np.flatnonzero(eval_global).astype(np.int64),
                outer_fold=fold,
                attempt_number=attempts,
                feature_names=l9v.names,
                output_directory=output_directory,
                deadline=deadline,
            )
            completed += 1
            fit_rows.append(metadata)
            l9v_available[eval_compact] = probabilities
            _check_cancel(cancel_file, f"after_l9v_fold_{fold}")
        _require(
            attempts == completed == 5 and bool(np.isfinite(l9v_available).all()),
            "fit family incomplete",
        )
        l9v_full = apply_back_fallback(controls["b0"], availability, l9v_available)
        _require(
            np.array_equal(l9v_full[~availability], controls["b0"][~availability]),
            "L9v fallback differs",
        )

        scored_probabilities = {
            "l9v": l9v_full[scoring_indices],
            **{name: value[scoring_indices] for name, value in controls.items()},
        }
        roster = cast(list[str], _mapping(config["source"], "source")["participant_roster"])
        reports = {
            name: method_report(
                labels=labels,
                probabilities=values,
                participant_ids=scored_people,
                roster=roster,
            )
            for name, values in scored_probabilities.items()
        }
        participant_folds = np.asarray(
            [int(np.unique(scored_folds[scored_people == person])[0]) for person in roster],
            dtype=np.int64,
        )
        analysis = analyse(
            reports,
            labels=labels,
            probabilities=scored_probabilities,
            participants=scored_people,
            participant_folds=participant_folds,
            roster=roster,
            source_reference_exact=True,
        )
        scored_available = availability[scoring_indices]
        strata = {
            "available": {
                name: method_report(
                    labels=labels[scored_available],
                    probabilities=values[scored_available],
                    participant_ids=scored_people[scored_available],
                    roster=roster,
                )
                for name, values in scored_probabilities.items()
            },
            "fallback": {
                name: method_report(
                    labels=labels[~scored_available],
                    probabilities=values[~scored_available],
                    participant_ids=scored_people[~scored_available],
                    roster=roster,
                )
                for name, values in scored_probabilities.items()
            },
            "support": {
                "available_rows": int(scored_available.sum()),
                "fallback_rows": int((~scored_available).sum()),
                "available_class_support": np.bincount(
                    labels[scored_available], minlength=3
                ).tolist(),
                "fallback_class_support": np.bincount(
                    labels[~scored_available], minlength=3
                ).tolist(),
                **topology,
            },
        }
        analysis_record = _sealed({"record_kind": "fog_l9v_analysis", **analysis, "strata": strata})
        _write_json_create_only(output_directory / "analysis.json", analysis_record)
        _write_json_create_only(
            output_directory / "participant_metrics.json",
            _sealed(
                {"record_kind": "fog_l9v_participant_metrics", "methods": reports, "strata": strata}
            ),
        )
        _write_json_create_only(
            output_directory / "fit_reports.json",
            _sealed(
                {
                    "record_kind": "fog_l9v_fit_reports",
                    "fit_attempt_count": attempts,
                    "completed_fit_count": completed,
                    "rows": fit_rows,
                }
            ),
        )
        observable_stack = np.stack([l9v_full, *[controls[name] for name in CONTROL_ORDER]])
        _write_npz_create_only(
            output_directory / "predictions.npz",
            method_ids=np.asarray(METHOD_ORDER, dtype=np.str_),
            observable_probabilities=observable_stack,
            scored_probabilities=np.stack([scored_probabilities[name] for name in METHOD_ORDER]),
            l9v_available_probabilities=l9v_available,
            availability_mask=availability,
            available_indices=available_indices,
            scoring_indices=scoring_indices,
            scored_labels=labels,
            observable_window_ids=window_ids,
            observable_participant_ids=people,
            observable_fold_index=folds,
            scored_participant_ids=scored_people,
            scored_window_ids=scored_window_ids,
            scored_fold_index=scored_folds,
            observable_decisions=observable_stack.argmax(axis=2).astype(np.int64),
            scored_decisions=np.stack(
                [scored_probabilities[name].argmax(axis=1) for name in METHOD_ORDER]
            ).astype(np.int64),
            fallback_source_method_index=np.full(122, B0_METHOD_INDEX, dtype=np.int64),
            fallback_source_method_id=np.full(122, "b0", dtype=np.str_),
        )
        result = _sealed(
            {
                "record_kind": "fog_l9v_result",
                "status": "complete_awaiting_independent_replay",
                "experiment_id": EXPERIMENT_ID,
                "evidence_status": analysis["evidence_status"],
                "code_commit": code_commit,
                "fit_attempt_count": attempts,
                "completed_fit_count": completed,
                "method_summary": _method_summary(reports),
                "advancement": analysis["advancement"],
                "decision": analysis["decision"],
                "fallback_topology": topology,
                "InclusiveHAR_P11_P20_loaded": False,
                "additional_seeds_launched": False,
                "automatic_follow_on_launched": False,
                "architecture_built": False,
            }
        )
        _write_json_create_only(output_directory / "result.json", result)
        _write_bytes_create_only(
            output_directory / "OUTCOME_SUMMARY.md", _outcome_summary(analysis_record).encode()
        )
        worker = _stop_task_owned_workers(
            baseline, attempts, completed, terminal="fit_phase_complete"
        )
        _require(
            worker["task_owned_fit_workers_and_monitors_stopped"] is True, "fit worker remains"
        )
        _write_json_create_only(output_directory / "worker_shutdown.json", worker)
        fit_phase_seconds = time.perf_counter() - controller_started
        _deadline_check(deadline, "after fit worker shutdown")
        runtime = _sealed(
            {
                "record_kind": "fog_l9v_runtime",
                "compute_cap_seconds": COMPUTE_CAP_SECONDS,
                "controller_seconds_through_fit_phase": fit_phase_seconds,
                "fit_and_prediction_seconds": sum(
                    float(row["fit_and_predict_seconds"]) for row in fit_rows
                ),
                "fit_attempt_count": attempts,
                "completed_fit_count": completed,
                "cap_exceeded": False,
                "execution_parallelism": "sequential_models_single_controller",
                "fit_worker_count": 4,
                "prediction_worker_count": 1,
                "blas_thread_limit": 1,
            }
        )
        _write_json_create_only(output_directory / "runtime.json", runtime)
        _deadline_check(deadline, "after runtime finalization")
        _write_json_create_only(
            output_directory / "artifact_manifest.json", _artifact_manifest(output_directory)
        )
        _deadline_check(deadline, "after prevalidation manifest finalization")
        return result
    except BaseException as exc:
        cleanup = _stop_task_owned_workers(
            baseline, attempts, completed, terminal="incomplete_cleanup"
        )
        if attempts > completed and output_directory.is_dir():
            started_path = output_directory / "fit_attempts" / f"{attempts:02d}--started.json"
            completed_path = output_directory / "fit_attempts" / f"{attempts:02d}--completed.json"
            failed_path = output_directory / "fit_attempts" / f"{attempts:02d}--failed.json"
            if started_path.is_file() and not completed_path.exists() and not failed_path.exists():
                try:
                    _write_json_create_only(
                        failed_path,
                        _sealed(
                            {
                                "record_kind": "fog_l9v_fit_attempt_failed",
                                "attempt_number": attempts,
                                "cell_id": "l9v",
                                "outer_fold": attempts - 1,
                                "failed_at_utc": _now(),
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        ),
                    )
                except BaseException:
                    pass
        if output_directory.is_dir() and not (output_directory / "INCOMPLETE.json").exists():
            failure = _sealed(
                {
                    "record_kind": "fog_l9v_incomplete",
                    "status": "incomplete_blocker_not_scientific_rejection",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "fit_attempt_count": attempts,
                    "completed_fit_count": completed,
                    "worker_cleanup": cleanup,
                    "no_retry_launched": True,
                    "automatic_follow_on_launched": False,
                }
            )
            try:
                _write_json_create_only(output_directory / "INCOMPLETE.json", failure)
            except BaseException:
                pass
        raise


def _validate_run_impl(
    run_directory: Path,
    *,
    deadline: float,
    controller_started: float,
    validation_baseline: Mapping[str, set[int]],
) -> dict[str, Any]:
    validation_started = time.perf_counter()
    _require(run_directory.is_dir(), "L9v run directory missing")
    _require(not (run_directory / "INCOMPLETE.json").exists(), "incomplete run cannot validate")
    _require(
        not (run_directory / "VALIDATION_INCOMPLETE.json").exists(),
        "prior validation blocker exists",
    )
    _require(not (run_directory / "validation.json").exists(), "validation already exists")
    _require(not (run_directory / "completion_manifest.json").exists(), "completion exists")
    _deadline_check(deadline, "validation start")
    artifact_count = _verify_artifact_manifest(run_directory)
    required = {
        "alignment_receipts.json",
        "analysis.json",
        "artifact_manifest.json",
        "availability_freeze.json",
        "config_snapshot.yaml",
        "feature_cache.npz",
        "feature_cache_metadata.json",
        "fit_reports.json",
        "input_manifest.json",
        "measurement_preflight.json",
        "OUTCOME_SUMMARY.md",
        "participant_metrics.json",
        "partition_preflight.json",
        "plan_snapshot.json",
        "predictions.npz",
        "preflight.json",
        "protocol_snapshot.json",
        "reference_receipt.json",
        "result.json",
        "runtime.json",
        "source_manifest.json",
        "worker_shutdown.json",
    }
    _require(all((run_directory / name).is_file() for name in required), "artifact missing")
    config = _read_yaml(run_directory / "config_snapshot.yaml")
    validate_config(config)
    preflight = _read_json(run_directory / "preflight.json")
    _verify_sealed(preflight, "L9v preflight")
    repository_root = Path(str(preflight["repository_root"])).resolve()
    evidence_root = Path(str(preflight["evidence_root"])).resolve()
    _require(
        sha256_file(run_directory / "config_snapshot.yaml") == preflight["configuration_sha256"],
        "config snapshot binding changed",
    )
    protocol = _read_json(run_directory / "protocol_snapshot.json")
    plan = _read_json(run_directory / "plan_snapshot.json")
    _verify_sealed(protocol, "protocol snapshot")
    _verify_sealed(plan, "plan snapshot")
    protocol_source = (repository_root / str(protocol["source_path"])).resolve()
    plan_source = Path(str(plan["source_path"])).resolve()
    _require(
        protocol_source.is_file()
        and sha256_file(protocol_source) == protocol["source_sha256"]
        and protocol_source.read_text(encoding="utf-8") == protocol["text"],
        "protocol source changed",
    )
    _require(
        plan_source.is_file()
        and sha256_file(plan_source) == PLAN_SHA256
        and plan_source.read_text(encoding="utf-8") == plan["text"],
        "plan source changed",
    )
    source = _read_json(run_directory / "source_manifest.json")
    _verify_sealed(source, "source manifest")
    _require(
        source["record_sha256"] == preflight["source_manifest_record_sha256"],
        "source binding changed",
    )
    for item in cast(list[dict[str, Any]], source["files"]):
        path = (repository_root / str(item["path"])).resolve()
        _require(
            path.is_relative_to(repository_root) and sha256_file(path) == item["sha256"],
            f"source changed: {path}",
        )
    source_body = dict(source)
    source_body.pop("record_sha256")
    _require(
        canonical_json_sha256(source_body)
        == canonical_json_sha256(_source_manifest(repository_root)),
        "source manifest membership changed",
    )
    git = _git_state(repository_root)
    _require(
        git["clean"] is True and git["commit"] == preflight["code_commit"],
        "run source commit/worktree changed",
    )

    spatial_run, reference_body = _verify_spatial_reference(evidence_root)
    reference = _read_json(run_directory / "reference_receipt.json")
    _verify_sealed(reference, "reference receipt")
    _require(
        canonical_json_sha256({k: v for k, v in reference.items() if k != "record_sha256"})
        == canonical_json_sha256(reference_body),
        "reference receipt changed",
    )
    _require(
        reference["record_sha256"] == preflight["reference_record_sha256"],
        "preflight/reference binding changed",
    )
    inputs = _input_paths(spatial_run)
    input_record = _read_json(run_directory / "input_manifest.json")
    _verify_sealed(input_record, "input manifest")
    for name in ("raw_source", "coverage", "availability"):
        item = _mapping(input_record[name], name)
        _require(Path(str(item["path"])).resolve() == inputs[name], f"{name} path changed")
        _require(sha256_file(inputs[name]) == item["sha256"], f"{name} hash changed")
    coverage = _read_json(inputs["coverage"])
    coverage_rows = coverage_rows_from_audit(coverage)
    coverage_ids = np.asarray([str(row["window_id"]) for row in coverage_rows], dtype=np.str_)
    _require(
        _availability_array_sha256(coverage_ids) == ORDERED_ID_SHA256,
        "validation coverage candidate ID hash changed",
    )
    availability = availability_mask_from_rows(coverage_rows)
    freeze = _read_json(run_directory / "availability_freeze.json")
    _verify_sealed(freeze, "availability freeze")
    _require(
        freeze["mask_contract_sha256"] == _availability_array_sha256(availability) == MASK_SHA256
        and freeze["mask_generic_array_sha256"] == _array_sha256(availability),
        "availability freeze changed",
    )

    cache = _read_npz(run_directory / "feature_cache.npz")
    expected_cache_keys = {
        "l9v_values",
        "l9v_names",
        "available_indices",
        "availability_mask",
        "observable_window_ids",
        "observable_participant_ids",
        "observable_fold_index",
    }
    _require(set(cache) == expected_cache_keys, "L9v cache schema changed")
    cache_metadata = _read_json(run_directory / "feature_cache_metadata.json")
    _verify_sealed(cache_metadata, "cache metadata")
    metadata_arrays = _mapping(cache_metadata["arrays"], "cache arrays")
    _require(set(metadata_arrays) == expected_cache_keys, "cache metadata schema changed")
    for name, value in cache.items():
        row = _mapping(metadata_arrays[name], name)
        _require(
            list(value.shape) == row["shape"]
            and str(value.dtype) == row["dtype"]
            and _array_sha256(value) == row["sha256"],
            f"cache array changed: {name}",
        )
    _require(
        cache_metadata["record_sha256"] == preflight["feature_cache_record_sha256"],
        "preflight/cache binding changed",
    )
    _require(np.array_equal(cache["availability_mask"], availability), "cache mask differs")
    available_indices = np.asarray(cache["available_indices"], dtype=np.int64)
    _require(
        np.array_equal(available_indices, np.flatnonzero(availability)), "cache indices differ"
    )
    people = np.asarray(cache["observable_participant_ids"], dtype=np.str_)
    folds = np.asarray(cache["observable_fold_index"], dtype=np.int64)
    l9v_values = np.asarray(cache["l9v_values"], dtype=np.float64)
    l9v_names = tuple(np.asarray(cache["l9v_names"], dtype=np.str_).tolist())
    _require(
        np.array_equal(coverage_ids, cache["observable_window_ids"])
        and freeze["ordered_observable_ids_contract_sha256"] == ORDERED_ID_SHA256,
        "validation candidate IDs differ",
    )
    _require(_array_sha256(l9v_values) == EXPECTED_L9V_FEATURE_SHA256, "stored L9v changed")
    _require(
        _array_sha256(np.asarray(l9v_names, dtype=np.str_)) == EXPECTED_L9V_NAMES_ARRAY_SHA256,
        "stored L9v names changed",
    )

    with np.load(spatial_run / "feature_cache.npz", allow_pickle=False) as archive:
        ankle80_values = np.asarray(archive["ankle_values"], dtype=np.float64)
        ankle80_names = np.asarray(archive["ankle_names"], dtype=np.str_)
    signal, gravity, fresh_indices, alignment_receipts, segment_receipts = (
        _materialize_ankle_windows(inputs["raw_source"], coverage_rows, availability)
    )
    fresh80 = extract_engineered_features(signal, channel_names=SIX_CHANNEL_NAMES)
    fresh9 = extract_engineered_features(
        np.concatenate((signal, gravity), axis=2), channel_names=NINE_CHANNEL_NAMES
    )
    _require(np.array_equal(fresh_indices, available_indices), "validation indices differ")
    _require(_array_sha256(signal) == ANKLE_SIGNAL_SHA256, "validation signal differs")
    _require(_array_sha256(gravity) == ANKLE_GRAVITY_SHA256, "validation gravity differs")
    _require(np.array_equal(fresh80.values, ankle80_values), "validation ankle80 differs")
    _require(
        np.array_equal(fresh9.values, l9v_values) and fresh9.names == l9v_names,
        "validation L9v differs",
    )
    screen = _read_json(run_directory / "measurement_preflight.json")
    _verify_sealed(screen, "measurement preflight")
    fresh_screen = measurement_screen(
        signal=signal,
        gravity=gravity,
        l9v_values=fresh9.values,
        l9v_names=fresh9.names,
        ankle80_values=ankle80_values,
        ankle80_names=ankle80_names.tolist(),
        availability=availability,
        observable_people=people,
        segment_receipts=segment_receipts,
    )
    screen_body = dict(screen)
    screen_body.pop("record_sha256")
    _require(
        canonical_json_sha256(screen_body) == canonical_json_sha256(fresh_screen),
        "measurement screen replay differs",
    )
    _require(
        screen["record_sha256"] == preflight["measurement_record_sha256"],
        "preflight/measurement binding changed",
    )
    alignment = _read_json(run_directory / "alignment_receipts.json")
    _verify_sealed(alignment, "alignment receipts")
    _require(
        alignment["alignment_receipt_count"] == 1817
        and canonical_json_sha256(alignment["alignment_receipts"])
        == canonical_json_sha256(alignment_receipts)
        and canonical_json_sha256(alignment["segment_receipts"])
        == canonical_json_sha256(segment_receipts),
        "alignment receipt replay differs",
    )
    _deadline_check(deadline, "after validation feature replay")

    spatial_cache = _read_npz(spatial_run / "feature_cache.npz")
    spatial_predictions = _read_npz(spatial_run / "predictions.npz")
    scoring_indices = np.asarray(spatial_cache["scoring_indices"], dtype=np.int64)
    eligibility = np.asarray(spatial_cache["scoring_eligibility"], dtype=np.bool_)
    labels = np.asarray(spatial_cache["scored_labels"], dtype=np.int64)
    scored_people = np.asarray(spatial_cache["scored_participant_ids"], dtype=np.str_)
    scored_window_ids = np.asarray(spatial_cache["scored_window_ids"], dtype=np.str_)
    scored_folds = np.asarray(spatial_cache["scored_fold_index"], dtype=np.int64)
    topology = _verify_fallback_topology(availability, people, scoring_indices)
    controls = {
        "b0": np.asarray(spatial_predictions["observable_probabilities"][0], dtype=np.float64),
        "bv": np.asarray(spatial_predictions["observable_probabilities"][1], dtype=np.float64),
        "lv": np.asarray(spatial_predictions["observable_probabilities"][2], dtype=np.float64),
        "f3": np.asarray(spatial_predictions["f3_reference_probabilities"], dtype=np.float64),
    }
    partition = _read_json(run_directory / "partition_preflight.json")
    _verify_sealed(partition, "partition preflight")
    fresh_partition = partition_preflight(
        labels=labels,
        people=people,
        folds=folds,
        eligibility=eligibility,
        availability=availability,
    )
    partition_body = dict(partition)
    partition_body.pop("record_sha256")
    _require(
        canonical_json_sha256(partition_body) == canonical_json_sha256(fresh_partition),
        "partition replay differs",
    )
    _require(
        partition["record_sha256"] == preflight["partition_record_sha256"],
        "preflight/partition binding changed",
    )

    fit_record = _read_json(run_directory / "fit_reports.json")
    _verify_sealed(fit_record, "fit reports")
    rows = cast(list[dict[str, Any]], fit_record["rows"])
    _require(
        len(rows) == 5
        and fit_record["fit_attempt_count"] == 5
        and fit_record["completed_fit_count"] == 5,
        "fit ledger count changed",
    )
    _require(
        [(row["cell_id"], row["outer_fold"], row["attempt_number"]) for row in rows]
        == [("l9v", fold, fold + 1) for fold in OUTER_FOLDS],
        "fit ledger order changed",
    )
    expected_receipts = {
        *(f"{attempt:02d}--started.json" for attempt in range(1, 6)),
        *(f"{attempt:02d}--completed.json" for attempt in range(1, 6)),
    }
    _require(
        {path.name for path in (run_directory / "fit_attempts").iterdir()} == expected_receipts,
        "attempt receipt family changed",
    )
    _require(
        {path.name for path in (run_directory / "checkpoints").iterdir()}
        == {f"l9v--fold-{fold}.pkl" for fold in OUTER_FOLDS},
        "checkpoint family changed",
    )
    compact_lookup = np.full(1939, -1, dtype=np.int64)
    compact_lookup[available_indices] = np.arange(1817, dtype=np.int64)
    observable_labels = np.zeros(1939, dtype=np.int64)
    observable_labels[eligibility] = labels
    replay_available = np.full((1817, 3), np.nan, dtype=np.float64)
    for fold, row in enumerate(rows):
        started_receipt = _read_json(
            run_directory / "fit_attempts" / f"{fold + 1:02d}--started.json"
        )
        completed_receipt = _read_json(
            run_directory / "fit_attempts" / f"{fold + 1:02d}--completed.json"
        )
        _verify_sealed(started_receipt, f"started receipt {fold + 1}")
        _verify_sealed(completed_receipt, f"completed receipt {fold + 1}")
        for key in ("attempt_number", "cell_id", "outer_fold"):
            _require(
                started_receipt[key] == row[key] == completed_receipt[key],
                f"receipt differs: {key}",
            )
        _require(
            started_receipt["random_state"] == row["random_state"]
            and started_receipt["training_row_count"] == row["training_row_count"]
            and started_receipt["evaluation_candidate_count"] == row["evaluation_candidate_count"]
            and completed_receipt["fit_seconds"] == row["fit_seconds"]
            and completed_receipt["prediction_seconds"] == row["prediction_seconds"]
            and completed_receipt["checkpoint"] == row["checkpoint"],
            f"attempt receipt metadata differs: {fold + 1}",
        )
        checkpoint = _mapping(row["checkpoint"], "checkpoint")
        path = (run_directory / str(checkpoint["path"])).resolve()
        _require(
            path.is_relative_to(run_directory) and sha256_file(path) == checkpoint["sha256"],
            "checkpoint changed",
        )
        metadata, estimator = _load_checkpoint(path)
        _require(
            metadata == {key: value for key, value in row.items() if key != "checkpoint"},
            "checkpoint metadata differs",
        )
        _require(
            estimator.get_params(deep=False) == _effective_parameters(11 + fold),
            "RF parameters differ",
        )
        masks = fold_masks(folds, eligibility, availability, fold)
        training = masks["restricted_training"]
        evaluation = masks["restricted_evaluation"]
        evaluation_compact = compact_lookup[np.flatnonzero(evaluation)]
        _require(
            row["training_global_indices_sha256"] == EXPECTED_TRAINING_ID_HASHES[fold],
            "training IDs differ",
        )
        _require(
            row["evaluation_global_indices_sha256"] == EXPECTED_EVALUATION_ID_HASHES[fold],
            "evaluation IDs differ",
        )
        _require(row["sample_weight_sha256"] == EXPECTED_WEIGHT_HASHES[fold], "weights differ")
        _require(tuple(row["feature_names"]) == l9v_names, "checkpoint feature schema differs")
        _require(
            row["feature_names_sha256"] == EXPECTED_L9V_NAMES_CANONICAL_SHA256
            and row["training_participants"] == sorted(np.unique(people[training]).tolist())
            and row["evaluation_participants"] == sorted(np.unique(people[evaluation]).tolist()),
            "fit metadata differs from partition",
        )
        expected_weights = participant_first_weights(observable_labels[training], people[training])
        _require(
            _array_sha256(expected_weights) == row["sample_weight_sha256"],
            "replayed weights differ",
        )
        with threadpool_limits(limits=1):
            values = _predict_proba_deterministically(estimator, l9v_values[evaluation_compact])
        replay_available[evaluation_compact] = values
        _deadline_check(deadline, f"after checkpoint replay {fold}")
    _require(bool(np.isfinite(replay_available).all()), "checkpoint replay incomplete")
    replay_l9v = apply_back_fallback(controls["b0"], availability, replay_available)
    stored = _read_npz(run_directory / "predictions.npz")
    expected_prediction_keys = {
        "method_ids",
        "observable_probabilities",
        "scored_probabilities",
        "l9v_available_probabilities",
        "availability_mask",
        "available_indices",
        "scoring_indices",
        "scored_labels",
        "observable_window_ids",
        "observable_participant_ids",
        "observable_fold_index",
        "scored_participant_ids",
        "scored_window_ids",
        "scored_fold_index",
        "observable_decisions",
        "scored_decisions",
        "fallback_source_method_index",
        "fallback_source_method_id",
    }
    _require(set(stored) == expected_prediction_keys, "prediction schema changed")
    _require(stored["method_ids"].tolist() == list(METHOD_ORDER), "method order changed")
    expected_observable = np.stack([replay_l9v, *[controls[name] for name in CONTROL_ORDER]])
    _require(
        np.array_equal(stored["observable_probabilities"], expected_observable),
        "observable predictions differ",
    )
    _require(
        np.array_equal(stored["l9v_available_probabilities"], replay_available),
        "available predictions differ",
    )
    _require(
        np.array_equal(
            stored["fallback_source_method_index"],
            np.full(122, B0_METHOD_INDEX, dtype=np.int64),
        )
        and np.array_equal(stored["fallback_source_method_id"], np.full(122, "b0", dtype=np.str_)),
        "fallback source differs",
    )
    _require(
        np.array_equal(replay_l9v[~availability], controls["b0"][~availability]),
        "fallback probabilities differ",
    )
    _require(
        np.array_equal(stored["scoring_indices"], scoring_indices)
        and np.array_equal(stored["scored_labels"], labels)
        and np.array_equal(stored["observable_window_ids"], cache["observable_window_ids"])
        and np.array_equal(stored["observable_participant_ids"], people)
        and np.array_equal(stored["observable_fold_index"], folds)
        and np.array_equal(stored["scored_participant_ids"], scored_people)
        and np.array_equal(stored["scored_window_ids"], scored_window_ids)
        and np.array_equal(stored["scored_fold_index"], scored_folds),
        "prediction provenance changed",
    )
    expected_scored = expected_observable[:, scoring_indices]
    _require(
        np.array_equal(stored["scored_probabilities"], expected_scored), "scored predictions differ"
    )
    _require(
        np.array_equal(stored["observable_decisions"], expected_observable.argmax(axis=2))
        and np.array_equal(stored["scored_decisions"], expected_scored.argmax(axis=2)),
        "decisions differ",
    )

    probabilities = {method: expected_scored[index] for index, method in enumerate(METHOD_ORDER)}
    roster = cast(list[str], _mapping(config["source"], "source")["participant_roster"])
    reports = {
        method: method_report(
            labels=labels,
            probabilities=value,
            participant_ids=scored_people,
            roster=roster,
        )
        for method, value in probabilities.items()
    }
    participant_folds = np.asarray(
        [int(np.unique(scored_folds[scored_people == person])[0]) for person in roster],
        dtype=np.int64,
    )
    reconstructed = analyse(
        reports,
        labels=labels,
        probabilities=probabilities,
        participants=scored_people,
        participant_folds=participant_folds,
        roster=roster,
        source_reference_exact=True,
    )
    scored_available = availability[scoring_indices]
    strata = {
        "available": {
            method: method_report(
                labels=labels[scored_available],
                probabilities=value[scored_available],
                participant_ids=scored_people[scored_available],
                roster=roster,
            )
            for method, value in probabilities.items()
        },
        "fallback": {
            method: method_report(
                labels=labels[~scored_available],
                probabilities=value[~scored_available],
                participant_ids=scored_people[~scored_available],
                roster=roster,
            )
            for method, value in probabilities.items()
        },
        "support": {
            "available_rows": int(scored_available.sum()),
            "fallback_rows": int((~scored_available).sum()),
            "available_class_support": np.bincount(labels[scored_available], minlength=3).tolist(),
            "fallback_class_support": np.bincount(labels[~scored_available], minlength=3).tolist(),
            **topology,
        },
    }
    recorded = _read_json(run_directory / "analysis.json")
    _verify_sealed(recorded, "analysis")
    for key, value in {**reconstructed, "strata": strata}.items():
        _require(
            canonical_json_sha256(recorded[key]) == canonical_json_sha256(value),
            f"analysis replay differs: {key}",
        )
    participant_record = _read_json(run_directory / "participant_metrics.json")
    _verify_sealed(participant_record, "participant metrics")
    _require(
        canonical_json_sha256(participant_record["methods"]) == canonical_json_sha256(reports)
        and canonical_json_sha256(participant_record["strata"]) == canonical_json_sha256(strata),
        "participant metrics replay differs",
    )
    _require(
        (run_directory / "OUTCOME_SUMMARY.md").read_text(encoding="utf-8")
        == _outcome_summary(recorded),
        "summary replay differs",
    )
    result = _read_json(run_directory / "result.json")
    runtime = _read_json(run_directory / "runtime.json")
    worker = _read_json(run_directory / "worker_shutdown.json")
    for item, name in ((result, "result"), (runtime, "runtime"), (worker, "worker")):
        _verify_sealed(item, name)
    _require(
        result["fit_attempt_count"] == result["completed_fit_count"] == 5,
        "result fit count differs",
    )
    _require(
        result["decision"] == reconstructed["decision"]
        and canonical_json_sha256(result["method_summary"])
        == canonical_json_sha256(_method_summary(reports))
        and canonical_json_sha256(result["advancement"])
        == canonical_json_sha256(reconstructed["advancement"])
        and canonical_json_sha256(result["fallback_topology"]) == canonical_json_sha256(topology)
        and result["code_commit"] == preflight["code_commit"],
        "result replay differs",
    )
    _require(result["architecture_built"] is False, "unauthorized architecture recorded")
    _require(
        runtime["compute_cap_seconds"] == COMPUTE_CAP_SECONDS and runtime["cap_exceeded"] is False,
        "runtime contract differs",
    )
    _require(
        worker["task_owned_fit_workers_and_monitors_stopped"] is True, "fit workers not stopped"
    )
    _deadline_check(deadline, "before validation shutdown")
    validation_worker = _stop_task_owned_workers(
        validation_baseline, 0, 0, terminal="validation_replay_complete"
    )
    _require(
        validation_worker["task_owned_fit_workers_and_monitors_stopped"] is True,
        "validation worker remains",
    )
    _write_json_create_only(run_directory / "validation_worker_shutdown.json", validation_worker)
    validation_seconds = time.perf_counter() - validation_started
    combined_seconds = time.perf_counter() - controller_started
    _deadline_check(deadline, "after validation worker shutdown")
    validation = _sealed(
        {
            "record_kind": "fog_l9v_validation",
            "status": "validated",
            "validated_at_utc": _now(),
            "run_source_commit": preflight["code_commit"],
            "validator_source_commit": git["commit"],
            "artifact_hashes_verified": artifact_count,
            "checkpoint_predictions_replayed": 5,
            "model_fits_during_validation": 0,
            "measurement_source_and_feature_replayed": True,
            "five_fold_ledger_replayed": True,
            "control_probabilities_exact": True,
            "fallback_probabilities_byte_exact": True,
            "fallback_topology_exact": True,
            "participant_reports_recomputed": 5,
            "comparisons_recomputed": 4,
            "advancement_gates_recomputed": 4,
            "full_22_person_roster_retained": True,
            "full_1213_scored_rows_retained": True,
            "InclusiveHAR_P11_P20_loaded": False,
            "source_downloaded_during_validation": False,
            "automatic_follow_on_launched": False,
            "architecture_built": False,
            "validation_task_owned_workers_and_monitors_stopped": True,
            "validation_seconds": validation_seconds,
            "combined_controller_seconds_before_completion_manifest": combined_seconds,
            "compute_cap_seconds": COMPUTE_CAP_SECONDS,
        }
    )
    _write_json_create_only(run_directory / "validation.json", validation)
    _deadline_check(deadline, "after validation record")
    files = [
        path
        for path in run_directory.rglob("*")
        if path.is_file()
        and path.name not in {"completion_manifest.json", "controller_final_receipt.json"}
    ]
    completion = _sealed(
        {
            "record_kind": "fog_l9v_completion_manifest",
            "status": "validation_complete_awaiting_controller_final_receipt",
            "self_excluded": True,
            "controller_final_receipt_excluded_and_binds_this_manifest": True,
            "artifacts": [
                {
                    "path": path.relative_to(run_directory).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(files)
            ],
            "validation_record_sha256": validation["record_sha256"],
            "decision": result["decision"],
            "task_owned_workers_and_monitors_stopped": True,
            "automatic_follow_on_launched": False,
            "architecture_built": False,
        }
    )
    _write_json_create_only(run_directory / "completion_manifest.json", completion)
    _deadline_check(deadline, "after completion manifest")
    for item in cast(list[dict[str, Any]], completion["artifacts"]):
        path = run_directory / str(item["path"])
        _require(
            path.is_file()
            and path.stat().st_size == int(item["size_bytes"])
            and sha256_file(path) == item["sha256"],
            f"completion manifest changed: {path}",
        )
    _deadline_check(deadline, "after completion manifest replay")
    return validation


def validate_run(
    run_directory: Path,
    *,
    deadline: float,
    controller_started: float,
) -> dict[str, Any]:
    baseline = _worker_baseline()
    resolved = run_directory.resolve()
    try:
        return _validate_run_impl(
            resolved,
            deadline=deadline,
            controller_started=controller_started,
            validation_baseline=baseline,
        )
    except BaseException as exc:
        cleanup = _stop_task_owned_workers(
            baseline, attempts=0, completed=0, terminal="validation_incomplete_cleanup"
        )
        if (
            resolved.is_dir()
            and not (resolved / "controller_final_receipt.json").exists()
            and not (resolved / "VALIDATION_INCOMPLETE.json").exists()
        ):
            failure = _sealed(
                {
                    "record_kind": "fog_l9v_validation_incomplete",
                    "status": "incomplete_blocker_not_scientific_rejection",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "model_fits_during_validation": 0,
                    "worker_cleanup": cleanup,
                    "no_retry_launched": True,
                    "automatic_follow_on_launched": False,
                }
            )
            try:
                _write_json_create_only(resolved / "VALIDATION_INCOMPLETE.json", failure)
            except BaseException:
                pass
        raise


def execute(
    *,
    repository_root: Path,
    evidence_root: Path,
    config_path: Path,
    protocol_path: Path,
    output_directory: Path,
    code_commit: str,
    timeout_seconds: int,
    cancel_file: Path | None,
) -> dict[str, Any]:
    _require(timeout_seconds == COMPUTE_CAP_SECONDS, "compute cap must remain 900 seconds")
    controller_started = time.perf_counter()
    deadline = controller_started + timeout_seconds
    run_experiment(
        repository_root=repository_root,
        evidence_root=evidence_root,
        config_path=config_path,
        protocol_path=protocol_path,
        output_directory=output_directory,
        code_commit=code_commit,
        deadline=deadline,
        controller_started=controller_started,
        cancel_file=cancel_file,
    )
    validation = validate_run(
        output_directory,
        deadline=deadline,
        controller_started=controller_started,
    )
    elapsed = time.perf_counter() - controller_started
    _deadline_check(deadline, "controller completion")
    completion_path = output_directory / "completion_manifest.json"
    receipt = _sealed(
        {
            "record_kind": "fog_l9v_controller_final_receipt",
            "status": "complete",
            "observed_at_utc": _now(),
            "complete_compute_seconds": elapsed,
            "compute_cap_seconds": COMPUTE_CAP_SECONDS,
            "cap_exceeded": False,
            "completion_manifest_file_sha256": sha256_file(completion_path),
            "validation_record_sha256": validation["record_sha256"],
            "model_fit_attempts": 5,
            "validation_model_fits": 0,
            "all_task_owned_workers_and_monitors_stopped": True,
            "automatic_follow_on_launched": False,
            "architecture_built": False,
        }
    )
    _write_json_create_only(output_directory / "controller_final_receipt.json", receipt)
    _deadline_check(deadline, "after controller final receipt")
    final_elapsed = time.perf_counter() - controller_started
    return {
        "status": "complete_and_independently_replayed",
        "decision": _read_json(output_directory / "result.json")["decision"],
        "complete_compute_seconds": final_elapsed,
        "completion_manifest_file_sha256": receipt["completion_manifest_file_sha256"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute_parser = commands.add_parser("execute")
    execute_parser.add_argument("--repository-root", type=Path, required=True)
    execute_parser.add_argument("--evidence-root", type=Path, required=True)
    execute_parser.add_argument("--config", type=Path, required=True)
    execute_parser.add_argument("--protocol", type=Path, required=True)
    execute_parser.add_argument("--output-directory", type=Path, required=True)
    execute_parser.add_argument("--code-commit", required=True)
    execute_parser.add_argument("--timeout-seconds", type=int, default=COMPUTE_CAP_SECONDS)
    execute_parser.add_argument("--cancel-file", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = execute(
        repository_root=arguments.repository_root,
        evidence_root=arguments.evidence_root,
        config_path=arguments.config,
        protocol_path=arguments.protocol,
        output_directory=arguments.output_directory,
        code_commit=arguments.code_commit,
        timeout_seconds=arguments.timeout_seconds,
        cancel_file=arguments.cancel_file,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
