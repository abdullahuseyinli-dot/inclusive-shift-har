"""Bounded corrected-FoG spatial-information and availability probe.

Five fixed participant-held-out Random Forest cells test a predeclared left-ankle
stream while preserving the original full-roster endpoint through a back-only
fallback. The validator replays every checkpoint without fitting models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import pickle
import platform
import sys
import threading
import time
import traceback
from collections import defaultdict
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

from inclusive_shift_har.data.external_har import (
    _contiguous_signal_runs,
    resample_physical_segment,
)
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
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.preprocessing.features import extract_engineered_features

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
StringArray: TypeAlias = NDArray[np.str_]

EXPERIMENT_ID = "fog-spatial-information-probe-v1"
CELL_ORDER = ("b0", "bv", "lv", "blv", "bbv")
RESTRICTED_CELLS = CELL_ORDER[1:]
OUTER_FOLDS = tuple(range(5))
CLASS_NAMES = ("mobility", "sitting", "standing")
SIX_CHANNEL_NAMES = (
    "lin_acc_x",
    "lin_acc_y",
    "lin_acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)
MAXIMUM_FIT_ATTEMPTS = 25
B0_REFERENCE_WEIGHT_SHA256 = (
    "834a265c9bbd98d21b51989090aa1cd9fe8155de82129e5fdb0329537b25822c",
    "e5e8101799863d9b8c5707ee843a976d1710abec4fbaea221d8844c69463a613",
    "f5aa392e6b11437177330804b49e6c74b69b3d80305e1b6707d1426c053c79cd",
    "d876008e68886eacad504470916a022fd8052227b35e8a9d8d4d21572a20d57f",
    "1e05f8c4ab2dc5bd586a88c2c57450dcfbae328e90914bb218600a69709d517e",
)
B0_REFERENCE_TRAINING_ROWS = (999, 884, 1006, 846, 1117)
EXPECTED_CONFIG_RECORD_SHA256 = "49471058cc28fb3228a6c07a39a996f2d70244d9a77f67e38110f96c735b9fe0"
RAW_SHA256 = "888875133fef386affddd0a5864f122d527d8d7bc588a69905cc0871bef53477"
RAW_SIZE = 119_629_580
COVERAGE_SHA256 = "16661d1e48b6b202a2295ccb2c258944d16d421024d0a7af506aa291a5ae6f14"
AVAILABILITY_SHA256 = "ba80d1ff36567ed7d9dbac0ebb0a885fddabce22bac0e81968c1fa64d98a533d"
MASK_SHA256 = "27c4d82a970975d93784cce15aa67799b07300eeacf7bb82d1ee62ec7f6b9a3d"
ORDERED_ID_SHA256 = "4c73070017ae3b2b911df6daa5cd20a3db0f6c8712e23c2f7846c5410e956e44"
ANKLE_SIGNAL_SHA256 = "4a4e9b199e78acf728df405fdb7ac9a1c28973a476d4b575c0c6d2903d39f012"
ANKLE_GRAVITY_SHA256 = "15a6cdd15b9bc66b2c6f8c8f571e0ab3fe94aed8a9ed9caed17beb24f9290247"
ANKLE_FEATURE_SHA256 = "4ee0db3c22f6e9f858835acb194d8f39fecd57830b12ac1ff1ff9a3de0079e50"
ANKLE_SEGMENT_RECEIPTS_SHA256 = "4c9a9f97f9ece2b9a4851a2095523b4ca288e186572ca022e0f22d0fce8e1944"
GSP_HASHES = {
    "feature_cache.npz": "6aabaae394bbf38e5ab84e738e9598c1d977e119d965f0f0eb17f7575aa7960a",
    "predictions.npz": "499219539c5105776c7dc396fbf87ee312a6ccf90a3f9b8a52bff8187876e489",
    "validation.json": "f769d2c2c4d7f2ef2e30f8fd8a37de765b3918251265c9d1feca72ca2fbfdf13",
    "completion_manifest.json": "10d844a7ebd77930f2bb19d6ed08fd90433205ff0303c8ccfc659655db417698",
}
FACTORIAL_HASHES = {
    "feature_cache.npz": "a9d6cfb4c3d0ea75623b3152255775f09292c510d5797d1c0b58192db20e4bea",
    "predictions.npz": "c3c7ab375f20bf875a8ff871cf1f1dc7c6b7718d896f37f2d1f6440227ceac6b",
    "participant_metrics.json": "417ee43e8b1ccf875805204b99a48322c5823939d086d053fc54704315e7b322",
    "validation.json": "1946240d9cb6a2f92777dc463c0516df30eeb10182eb8911dbe800d4df4355e5",
    "completion_manifest.json": "e91e6c9fbe61256fd49bc54574ea85728b8c960afe4c366458b5af30a96fcce2",
}
DECISION_HASHES = {
    "predictions.npz": "109fe67e55b9ee590d042db353eb02b17e5f218f6d2aa94b6a56011ce7c89f60",
    "validation.json": "e7815391955a2d00393e3aee49b505344cbe515d49b908020ff8d0825b8a4171",
    "completion_manifest.json": "49927afb88c1f1843460a681867c2e64d84dd854a10af014feea485747cd6e01",
}
REFERENCE_RELATIVE = {
    "gsp": Path(".audit/fog_gsp_order_ablation/fog-gsp-order-ablation-seed11-20260907-001"),
    "factorial": Path(
        ".audit/fog_rf_feature_weight_factorial/fog-rf-feature-weight-factorial-seed11-20260907-001"
    ),
    "decision": Path(".audit/fog_decision_rule_probe/fog-decision-rule-probe-seed11-20260907-001"),
    "review": Path(".audit/fog_decision_team_review_20260907-001"),
}
REQUIRED_PREVALIDATION_FILES = (
    "analysis.json",
    "artifact_manifest.json",
    "availability_receipt.json",
    "availability_freeze.json",
    "config_snapshot.yaml",
    "feature_cache.npz",
    "feature_cache_metadata.json",
    "fit_reports.json",
    "input_manifest.json",
    "OUTCOME_SUMMARY.md",
    "participant_metrics.json",
    "partition_preflight.json",
    "predictions.npz",
    "preflight.json",
    "protocol_snapshot.json",
    "reference_receipts.json",
    "result.json",
    "runtime.json",
    "source_manifest.json",
    "worker_shutdown.json",
)


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


def _availability_array_sha256(values: NDArray[Any]) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str((array.shape, array.dtype.str)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _check_cancel(cancel_file: Path | None, stage: str) -> None:
    if cancel_file is not None and cancel_file.exists():
        raise KeyboardInterrupt(f"cooperative cancellation requested at {stage}: {cancel_file}")


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject any change to the source-frozen YAML record."""

    _require(
        canonical_json_sha256(dict(config)) == EXPECTED_CONFIG_RECORD_SHA256,
        "frozen spatial-information config changed",
    )


def availability_mask_from_rows(rows: Sequence[Mapping[str, Any]]) -> BoolArray:
    """Construct the outcome-blind sensor-availability mask."""

    _require(len(rows) == 1939, "coverage candidate count changed")
    identifiers = [str(row["window_id"]) for row in rows]
    _require(len(set(identifiers)) == len(identifiers), "coverage candidate IDs are not unique")
    mask = np.asarray(
        [
            bool(row["ankle_finite_run_contains_window"])
            and bool(row["ankle_own_continuous_50hz_grid_matches_all128_times"])
            and len(cast(Sequence[Any], row["ankle_matching_grid_run_indices"])) == 1
            for row in rows
        ],
        dtype=np.bool_,
    )
    _require(int(mask.sum()) == 1817, "availability count changed")
    _require(_availability_array_sha256(mask) == MASK_SHA256, "availability mask hash changed")
    return mask


def apply_back_fallback(
    back_probabilities: FloatArray,
    available_mask: BoolArray,
    available_probabilities: FloatArray,
) -> FloatArray:
    """Insert branch probabilities only where the predeclared sensor stream qualifies."""

    back = np.asarray(back_probabilities, dtype=np.float64)
    mask = np.asarray(available_mask, dtype=np.bool_)
    branch = np.asarray(available_probabilities, dtype=np.float64)
    _require(back.ndim == 2 and back.shape[1] == 3, "back probability shape changed")
    _require(mask.shape == (back.shape[0],), "fallback mask shape changed")
    _require(branch.shape == (int(mask.sum()), 3), "available branch shape changed")
    _require(bool(np.isfinite(back).all() and np.isfinite(branch).all()), "nonfinite probability")
    result = back.copy()
    result[mask] = branch
    _require(np.array_equal(result[~mask], back[~mask]), "fallback differs from B0")
    return result


def fold_masks(
    folds: IntArray, eligibility: BoolArray, availability: BoolArray, outer_fold: int
) -> dict[str, BoolArray]:
    """Return the exact full and availability-qualified fold masks."""

    _require(outer_fold in OUTER_FOLDS, "outer fold leaves frozen plan")
    _require(folds.shape == eligibility.shape == availability.shape, "fold mask misalignment")
    evaluation = folds == outer_fold
    return {
        "b0_training": (~evaluation) & eligibility,
        "restricted_training": (~evaluation) & eligibility & availability,
        "b0_evaluation": evaluation,
        "restricted_evaluation": evaluation & availability,
    }


def assert_b0_replay(actual: FloatArray, expected: FloatArray) -> dict[str, Any]:
    """Apply the hard barrier that must pass before fit attempt six."""

    _require(actual.shape == expected.shape == (1939, 3), "B0 replay shape changed")
    _require(np.array_equal(actual, expected), "fresh B0 probability replay mismatch")
    decisions_exact = np.array_equal(actual.argmax(axis=1), expected.argmax(axis=1))
    _require(decisions_exact, "fresh B0 decision replay mismatch")
    return {
        "checked_before_restricted_fits": True,
        "attempt_count_before_decision": 5,
        "all_candidate_probabilities_byte_exact": True,
        "all_candidate_decisions_byte_exact": True,
        "maximum_absolute_probability_difference": 0.0,
    }


def make_feature_blocks(
    back_values: FloatArray,
    ankle_values: FloatArray,
    back_names: Sequence[str],
    ankle_names: Sequence[str],
) -> dict[str, tuple[FloatArray, tuple[str, ...]]]:
    """Create the four matched-support feature blocks without missing-row imputation."""

    back = np.asarray(back_values, dtype=np.float64)
    ankle = np.asarray(ankle_values, dtype=np.float64)
    _require(back.shape == ankle.shape and back.shape[1] == 80, "matched features misalign")
    _require(len(back_names) == len(ankle_names) == 80, "feature names misalign")
    prefixed_back = tuple(f"back::{name}" for name in back_names)
    prefixed_ankle = tuple(f"ankleL::{name}" for name in ankle_names)
    duplicate = tuple(f"back_duplicate::{name}" for name in back_names)
    return {
        "bv": (back.copy(), prefixed_back),
        "lv": (ankle.copy(), prefixed_ankle),
        "blv": (np.column_stack((back, ankle)), (*prefixed_back, *prefixed_ankle)),
        "bbv": (np.column_stack((back, back)), (*prefixed_back, *duplicate)),
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


def _verify_manifest(run_directory: Path, filename: str) -> int:
    payload = _read_json(run_directory / filename)
    _verify_sealed(payload, str(run_directory / filename))
    artifacts = cast(list[dict[str, Any]], payload["artifacts"])
    paths = [str(item["path"]) for item in artifacts]
    _require(len(paths) == len(set(paths)), f"duplicate path in {filename}")
    if filename == "artifact_manifest.json":
        _require(
            payload["record_kind"] == "fog_spatial_artifact_manifest",
            "artifact manifest kind changed",
        )
        allowed_unlisted = {
            "artifact_manifest.json",
            "validation.json",
            "completion_manifest.json",
            "VALIDATION_INCOMPLETE.json",
        }
    elif filename == "completion_manifest.json":
        _require(
            str(payload["record_kind"]).endswith("completion_manifest"),
            "completion manifest kind changed",
        )
        allowed_unlisted = {"completion_manifest.json"}
    else:
        raise ValueError(f"unsupported manifest family: {filename}")
    for item in artifacts:
        path = (run_directory / str(item["path"])).resolve()
        _require(path.is_relative_to(run_directory), "manifest path escapes run")
        _require(
            path.is_file()
            and path.stat().st_size == int(item["size_bytes"])
            and sha256_file(path) == item["sha256"],
            f"manifest mismatch: {path}",
        )
    actual = {
        path.relative_to(run_directory).as_posix()
        for path in run_directory.rglob("*")
        if path.is_file() and path.relative_to(run_directory).as_posix() not in allowed_unlisted
    }
    _require(actual == set(paths), f"manifest filesystem coverage changed: {filename}")
    return len(artifacts)


def _verify_reference_files(run: Path, hashes: Mapping[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, expected in hashes.items():
        path = run / name
        _require(path.is_file(), f"missing reference: {path}")
        actual = sha256_file(path)
        _require(actual == expected, f"reference hash changed: {path}")
        rows.append({"path": str(path), "sha256": actual, "size_bytes": path.stat().st_size})
    _require(
        _read_json(run / "validation.json")["status"] in {"validated", "pass"},
        "reference validation absent",
    )
    _verify_manifest(run, "completion_manifest.json")
    return rows


def _load_reference_arrays(evidence_root: Path) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    gsp = evidence_root / REFERENCE_RELATIVE["gsp"]
    factorial = evidence_root / REFERENCE_RELATIVE["factorial"]
    decision = evidence_root / REFERENCE_RELATIVE["decision"]
    checked = {
        "gsp": _verify_reference_files(gsp, GSP_HASHES),
        "factorial": _verify_reference_files(factorial, FACTORIAL_HASHES),
        "decision": _verify_reference_files(decision, DECISION_HASHES),
    }
    gsp_cache = _read_npz(gsp / "feature_cache.npz")
    factorial_cache = _read_npz(factorial / "feature_cache.npz")
    gsp_predictions = _read_npz(gsp / "predictions.npz")
    factorial_predictions = _read_npz(factorial / "predictions.npz")
    decision_predictions = _read_npz(decision / "predictions.npz")
    prediction_keys = {
        "cell_ids",
        "observable_probabilities",
        "scored_probabilities",
        "observable_window_ids",
        "observable_participant_ids",
        "observable_fold_index",
        "scoring_indices",
        "scored_labels",
        "scored_participant_ids",
        "scored_window_ids",
        "scored_fold_index",
    }
    _require(set(gsp_predictions) == prediction_keys, "GSP prediction schema changed")
    _require(set(factorial_predictions) == prediction_keys, "factorial prediction schema changed")
    _require(gsp_predictions["cell_ids"].tolist() == ["a", "b", "c"], "GSP cell order changed")
    _require(
        factorial_predictions["cell_ids"].tolist() == ["f0", "f1", "f2", "f3"],
        "factorial cell order changed",
    )
    keys = (
        "observable_window_ids",
        "observable_participant_ids",
        "observable_fold_index",
        "scoring_indices",
        "scored_labels",
        "scored_participant_ids",
        "scored_window_ids",
        "scored_fold_index",
    )
    for key in keys:
        _require(
            np.array_equal(gsp_cache[key], factorial_cache[key]),
            f"reference alignment differs: {key}",
        )
        _require(
            np.array_equal(gsp_predictions[key], gsp_cache[key])
            and np.array_equal(factorial_predictions[key], factorial_cache[key])
            and np.array_equal(decision_predictions[key], gsp_cache[key]),
            f"reference prediction provenance differs: {key}",
        )
    scoring_indices = np.asarray(gsp_cache["scoring_indices"], dtype=np.int64)
    _require(
        np.array_equal(
            gsp_predictions["scored_probabilities"],
            gsp_predictions["observable_probabilities"][:, scoring_indices],
        )
        and np.array_equal(
            factorial_predictions["scored_probabilities"],
            factorial_predictions["observable_probabilities"][:, scoring_indices],
        ),
        "reference scored-probability projection changed",
    )
    expected_eligibility = np.zeros(1939, dtype=np.bool_)
    expected_eligibility[scoring_indices] = True
    _require(
        np.array_equal(decision_predictions["scoring_eligibility"], expected_eligibility),
        "decision reference eligibility changed",
    )
    _require(
        np.array_equal(gsp_cache["a_values"], factorial_cache["six_values"]),
        "back80 feature references differ",
    )
    _require(
        np.array_equal(gsp_cache["a_names"], factorial_cache["six_names"]),
        "back80 name references differ",
    )
    b0_gsp = np.asarray(gsp_predictions["observable_probabilities"][0], dtype=np.float64)
    b0_factorial = np.asarray(
        factorial_predictions["observable_probabilities"][1], dtype=np.float64
    )
    b0_decision = np.asarray(decision_predictions["raw_outer_probabilities"], dtype=np.float64)
    _require(
        np.array_equal(b0_gsp, b0_factorial) and np.array_equal(b0_gsp, b0_decision),
        "archived B0 references differ",
    )
    _require(
        np.array_equal(b0_gsp.argmax(axis=1), decision_predictions["d0_decisions"]),
        "archived B0 decisions differ",
    )
    f3_factorial = np.asarray(
        factorial_predictions["observable_probabilities"][3], dtype=np.float64
    )
    _require(
        np.array_equal(f3_factorial, decision_predictions["f3_reference_probabilities"])
        and np.array_equal(
            f3_factorial.argmax(axis=1), decision_predictions["f3_reference_decisions"]
        ),
        "archived F3 reference differs",
    )
    decision_fit_reports = _read_json(decision / "fit_reports.json")
    _verify_sealed(decision_fit_reports, "decision fit reports")
    outer_rows = sorted(
        (
            row
            for row in cast(list[dict[str, Any]], decision_fit_reports["rows"])
            if row["role"] == "outer"
        ),
        key=lambda row: int(row["outer_fold"]),
    )
    _require(len(outer_rows) == 5, "archived B0 outer-fit count changed")
    archived_weight_hashes = tuple(
        str(_mapping(row["sample_weight_summary"], "sample weights")["sha256"])
        for row in outer_rows
    )
    archived_training_rows = tuple(int(row["training_row_count"]) for row in outer_rows)
    _require(
        archived_weight_hashes == B0_REFERENCE_WEIGHT_SHA256
        and archived_training_rows == B0_REFERENCE_TRAINING_ROWS
        and tuple(int(row["outer_fold"]) for row in outer_rows) == OUTER_FOLDS
        and tuple(int(row["attempt_number"]) for row in outer_rows) == (1, 2, 3, 4, 5)
        and tuple(int(row["random_state"]) for row in outer_rows) == (11, 12, 13, 14, 15),
        "archived B0 fit ledger changed",
    )
    arrays = {
        "back_values": np.asarray(gsp_cache["a_values"], dtype=np.float64),
        "back_names": np.asarray(gsp_cache["a_names"], dtype=np.str_),
        "observable_window_ids": np.asarray(gsp_cache["observable_window_ids"], dtype=np.str_),
        "observable_participant_ids": np.asarray(
            gsp_cache["observable_participant_ids"], dtype=np.str_
        ),
        "observable_fold_index": np.asarray(gsp_cache["observable_fold_index"], dtype=np.int64),
        "scoring_indices": np.asarray(gsp_cache["scoring_indices"], dtype=np.int64),
        "scored_labels": np.asarray(gsp_cache["scored_labels"], dtype=np.int64),
        "scored_participant_ids": np.asarray(gsp_cache["scored_participant_ids"], dtype=np.str_),
        "scored_window_ids": np.asarray(gsp_cache["scored_window_ids"], dtype=np.str_),
        "scored_fold_index": np.asarray(gsp_cache["scored_fold_index"], dtype=np.int64),
        "b0_reference_probabilities": b0_gsp,
        "f3_reference_probabilities": f3_factorial,
        "b0_reference_weight_hashes": np.asarray(archived_weight_hashes, dtype=np.str_),
        "b0_reference_training_rows": np.asarray(archived_training_rows, dtype=np.int64),
    }
    arrays["scoring_eligibility"] = np.zeros(1939, dtype=np.bool_)
    arrays["scoring_eligibility"][cast(IntArray, arrays["scoring_indices"])] = True
    receipt = {
        "record_kind": "fog_spatial_reference_receipts",
        "checked_files": checked,
        "completion_entries_verified": {
            name: len(
                _read_json(evidence_root / REFERENCE_RELATIVE[name] / "completion_manifest.json")[
                    "artifacts"
                ]
            )
            for name in ("gsp", "factorial", "decision")
        },
        "back_features_exact_across_references": True,
        "b0_probabilities_exact_across_three_references": True,
        "f3_probabilities_exact_across_factorial_and_decision_references": True,
        "prediction_provenance_arrays_exact_across_references": True,
        "prediction_cell_orders_verified": {
            "gsp": ["a", "b", "c"],
            "factorial": ["f0", "f1", "f2", "f3"],
        },
        "b0_outer_weight_hashes": list(archived_weight_hashes),
        "b0_outer_training_rows": list(archived_training_rows),
        "f3_cell_index": 3,
    }
    return arrays, receipt


def _materialize_ankle_windows(
    raw_path: Path,
    coverage_rows: Sequence[Mapping[str, Any]],
    availability: BoolArray,
) -> tuple[
    FloatArray,
    FloatArray,
    IntArray,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Materialize only prequalified left-ankle windows; no annotation argument exists."""

    columns = ["timestamp", "subjectID", "sessionID"]
    for sensor in ("back", "ankleL"):
        columns.extend(
            f"{sensor}_{modality}_{axis}" for modality in ("acc", "gyro") for axis in "xyz"
        )
    frame = pd.read_csv(raw_path, usecols=columns)
    _require(set(frame.columns) == set(columns), "raw sensor column set changed")
    rows_by_session: dict[tuple[str, str], list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(coverage_rows):
        if availability[index]:
            rows_by_session[(str(row["participant_id"]), str(row["session_id"]))].append(
                (index, row)
            )
    windows: list[FloatArray] = []
    gravity_windows: list[FloatArray] = []
    global_indices: list[int] = []
    receipts: list[dict[str, Any]] = []
    segment_receipts: list[dict[str, Any]] = []
    for (subject, session_value), group in frame.groupby(
        ["subjectID", "sessionID"], sort=True, dropna=False
    ):
        participant = f"fogstar:{int(subject):03d}"
        session = f"{participant}:session-{int(session_value):03d}"
        wanted_rows = rows_by_session.get((participant, session), [])
        timestamps = group["timestamp"].to_numpy(dtype=np.float64)
        signal_names = [
            f"ankleL_{modality}_{axis}" for modality in ("acc", "gyro") for axis in "xyz"
        ]
        values = group[signal_names].to_numpy(dtype=np.float64)
        total = values[:, :3] * 9.80665
        gyro = values[:, 3:] * np.pi / 180.0
        runs = _contiguous_signal_runs(
            timestamps, total, gyro, nominal_rate_hz=60.0, maximum_gap_factor=3.0
        )
        materialized: dict[int, Any] = {}
        for run_index, (left, right) in enumerate(runs):
            if right - left < 3:
                continue
            segment = resample_physical_segment(
                timestamps=timestamps[left:right],
                acceleration=total[left:right],
                gyroscope=gyro[left:right],
                gravity=None,
                source_rate_hz=60.0,
                target_rate_hz=50.0,
                gravity_cutoff_hz=0.30,
                window_samples=128,
            )
            materialized[run_index] = segment
            segment_receipts.append(
                {
                    "participant_id": participant,
                    "session_id": session,
                    "ankle_run_index": run_index,
                    "source_left_index": left,
                    "source_right_index_exclusive": right,
                    "audit": segment.audit(),
                }
            )
        for global_index, row in wanted_rows:
            matches = cast(list[int], row["ankle_matching_grid_run_indices"])
            _require(len(matches) == 1 and matches[0] in materialized, "ankle run match changed")
            segment = materialized[matches[0]]
            start = float(row["start_time"])
            offset = int(np.round((start - float(segment.timestamps[0])) * 50.0))
            _require(
                0 <= offset and offset + 128 <= len(segment.timestamps), "ankle offset invalid"
            )
            expected = start + np.arange(128, dtype=np.float64) / 50.0
            actual = np.asarray(segment.timestamps[offset : offset + 128], dtype=np.float64)
            _require(
                np.allclose(actual, expected, rtol=0.0, atol=1e-10),
                "ankle timestamp correspondence changed",
            )
            signal = np.asarray(segment.signals[offset : offset + 128], dtype=np.float64)
            gravity = np.asarray(segment.gravity[offset : offset + 128], dtype=np.float64)
            _require(signal.shape == (128, 6) and bool(np.isfinite(signal).all()), "ankle invalid")
            _require(
                gravity.shape == (128, 3) and bool(np.isfinite(gravity).all()),
                "ankle gravity invalid",
            )
            global_indices.append(global_index)
            windows.append(signal)
            gravity_windows.append(gravity)
            receipts.append(
                {
                    "global_candidate_index": global_index,
                    "window_id": str(row["window_id"]),
                    "participant_id": participant,
                    "session_id": session,
                    "ankle_run_index": matches[0],
                    "ankle_offset": offset,
                    "timestamp_sha256": _array_sha256(actual),
                    "signal_sha256": _array_sha256(signal),
                    "gravity_sha256": _array_sha256(gravity),
                }
            )
    order = np.argsort(np.asarray(global_indices, dtype=np.int64))
    indices = np.asarray(global_indices, dtype=np.int64)[order]
    stacked = np.stack(windows)[order]
    stacked_gravity = np.stack(gravity_windows)[order]
    ordered_receipts = [receipts[int(index)] for index in order]
    _require(np.array_equal(indices, np.flatnonzero(availability)), "ankle index order changed")
    _require(
        stacked.shape == (int(availability.sum()), 128, 6),
        "ankle window shape changed",
    )
    _require(
        stacked_gravity.shape == (int(availability.sum()), 128, 3),
        "ankle gravity-window shape changed",
    )
    return stacked, stacked_gravity, indices, ordered_receipts, segment_receipts


def _weight_summary(labels: IntArray, participants: StringArray) -> dict[str, Any]:
    weights = participant_first_weights(labels, participants)
    totals = [float(weights[participants == person].sum()) for person in np.unique(participants)]
    return {
        "minimum": float(weights.min()),
        "maximum": float(weights.max()),
        "mean": float(weights.mean()),
        "sum": float(weights.sum()),
        "sha256": _array_sha256(weights),
        "participant_total_minimum": min(totals),
        "participant_total_maximum": max(totals),
    }


def partition_preflight(
    *,
    labels: IntArray,
    observable_participants: StringArray,
    observable_folds: IntArray,
    scoring_eligibility: BoolArray,
    availability: BoolArray,
) -> dict[str, Any]:
    observable_labels = np.zeros(observable_participants.size, dtype=np.int64)
    observable_labels[scoring_eligibility] = labels
    rows: list[dict[str, Any]] = []
    expected_restricted_counts = (940, 825, 947, 829, 1075)
    for fold in OUTER_FOLDS:
        masks = fold_masks(observable_folds, scoring_eligibility, availability, fold)
        evaluation = masks["b0_evaluation"]
        base = masks["b0_training"]
        restricted = masks["restricted_training"]
        _require(
            int(restricted.sum()) == expected_restricted_counts[fold], "training count changed"
        )
        _require(
            set(np.unique(observable_labels[base]).tolist())
            == set(np.unique(observable_labels[restricted]).tolist())
            == {0, 1, 2},
            "training partition lacks a class",
        )
        _require(
            set(np.unique(observable_participants[base])).isdisjoint(
                np.unique(observable_participants[evaluation])
            ),
            "outer participant leakage",
        )
        base_weight = _weight_summary(observable_labels[base], observable_participants[base])
        restricted_weight = _weight_summary(
            observable_labels[restricted], observable_participants[restricted]
        )
        rows.append(
            {
                "outer_fold": fold,
                "evaluation_participants": sorted(
                    np.unique(observable_participants[evaluation]).tolist()
                ),
                "evaluation_candidate_count": int(evaluation.sum()),
                "evaluation_available_candidate_count": int((evaluation & availability).sum()),
                "b0_training_row_count": int(base.sum()),
                "b0_training_class_counts": np.bincount(
                    observable_labels[base], minlength=3
                ).tolist(),
                "b0_weight_summary": base_weight,
                "restricted_training_row_count": int(restricted.sum()),
                "restricted_training_class_counts": np.bincount(
                    observable_labels[restricted], minlength=3
                ).tolist(),
                "restricted_training_participants": sorted(
                    np.unique(observable_participants[restricted]).tolist()
                ),
                "restricted_training_ids_sha256": _array_sha256(
                    np.flatnonzero(restricted).astype(np.int64)
                ),
                "restricted_weight_summary": restricted_weight,
                "restricted_cells_identical_training_ids_and_weights": True,
            }
        )
    return {
        "record_kind": "fog_spatial_partition_preflight",
        "created_before_model_fits": True,
        "partition_count": 10,
        "fit_partition_count": 25,
        "rows": rows,
    }


def _fit_model(
    *,
    training_values: FloatArray,
    training_labels: IntArray,
    training_participants: StringArray,
    evaluation_values: FloatArray,
    evaluation_participants: StringArray,
    training_global_indices: IntArray,
    evaluation_global_indices: IntArray,
    cell_id: str,
    outer_fold: int,
    attempt_number: int,
    feature_names: tuple[str, ...],
    output_directory: Path,
    deadline: float,
) -> tuple[FloatArray, dict[str, Any]]:
    _require(1 <= attempt_number <= MAXIMUM_FIT_ATTEMPTS, "fit-attempt budget exhausted")
    _require(time.perf_counter() < deadline, "compute cap reached before registered fit")
    _require(set(np.unique(training_labels).tolist()) == {0, 1, 2}, "fit lacks class")
    weights = participant_first_weights(training_labels, training_participants)
    started_record = _sealed(
        {
            "record_kind": "fog_spatial_fit_attempt_started",
            "attempt_number": attempt_number,
            "cell_id": cell_id,
            "outer_fold": outer_fold,
            "started_at_utc": _now(),
            "random_state": 11 + outer_fold,
            "training_row_count": int(training_labels.size),
            "evaluation_candidate_count": int(evaluation_values.shape[0]),
        }
    )
    _write_json_create_only(
        output_directory / "fit_attempts" / f"{attempt_number:02d}--started.json",
        started_record,
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
    _require(time.perf_counter() <= deadline, "compute cap reached during fit")
    _require(np.array_equal(estimator.classes_, np.arange(3)), "RF class order changed")
    _require(
        probabilities.shape == (evaluation_values.shape[0], 3)
        and bool(np.isfinite(probabilities).all())
        and bool(np.all(probabilities >= 0))
        and np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12),
        "RF probabilities invalid",
    )
    metadata: dict[str, Any] = {
        "attempt_number": attempt_number,
        "cell_id": cell_id,
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
        "sample_weight_summary": _weight_summary(training_labels, training_participants),
        "effective_parameters": estimator.get_params(deep=False),
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "fit_and_predict_seconds": elapsed,
        "prediction_worker_count": 1,
    }
    checkpoint_path = output_directory / "checkpoints" / f"{cell_id}--fold-{outer_fold}.pkl"
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
            "record_kind": "fog_spatial_fit_attempt_completed",
            "attempt_number": attempt_number,
            "cell_id": cell_id,
            "outer_fold": outer_fold,
            "completed_at_utc": _now(),
            "fit_seconds": fit_seconds,
            "prediction_seconds": prediction_seconds,
            "checkpoint": metadata["checkpoint"],
        }
    )
    _write_json_create_only(
        output_directory / "fit_attempts" / f"{attempt_number:02d}--completed.json",
        completed,
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


def _participant_values(report: Mapping[str, Any]) -> FloatArray:
    rows = cast(list[dict[str, Any]], report["participants"])
    _require(all(row["eligible"] is True for row in rows), "full roster comparison required")
    return np.asarray([float(row["macro_f1"]) for row in rows], dtype=np.float64)


def _leave_one_fold(
    left: Mapping[str, Any], right: Mapping[str, Any], participant_folds: IntArray
) -> list[dict[str, Any]]:
    differences = _participant_values(left) - _participant_values(right)
    _require(differences.shape == participant_folds.shape, "participant fold alignment changed")
    return [
        {
            "omitted_outer_fold": fold,
            "remaining_participant_count": int((participant_folds != fold).sum()),
            "mean_participant_difference": float(differences[participant_folds != fold].mean()),
        }
        for fold in OUTER_FOLDS
    ]


def _gate(
    comparison: Mapping[str, Any], leave_fold: Sequence[Mapping[str, Any]], replay_exact: bool
) -> dict[str, Any]:
    recalls = _mapping(comparison["class_recall_differences"], "class recall differences")
    checks = {
        "full_22_person_roster_scored": True,
        "fresh_b0_exact_replay": replay_exact,
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


def _events(
    labels: IntArray,
    old_probabilities: FloatArray,
    new_probabilities: FloatArray,
    people: StringArray,
    roster: Sequence[str],
) -> dict[str, Any]:
    old = old_probabilities.argmax(axis=1)
    new = new_probabilities.argmax(axis=1)
    rescued = (old != labels) & (new == labels)
    harmed = (old == labels) & (new != labels)
    changed = old != new
    tied = ~changed
    return {
        "rescues": int(rescued.sum()),
        "harms": int(harmed.sum()),
        "ties": int(tied.sum()),
        "changed_decisions": int(changed.sum()),
        "wrong_to_different_wrong": int((changed & ~rescued & ~harmed).sum()),
        "per_participant": [
            {
                "participant_id": person,
                "rescues": int((rescued & (people == person)).sum()),
                "harms": int((harmed & (people == person)).sum()),
                "ties": int((tied & (people == person)).sum()),
                "changed": int((changed & (people == person)).sum()),
            }
            for person in roster
        ],
    }


CONTRAST_PAIRS = {
    "bv_minus_b0": ("bv", "b0"),
    "lv_minus_bv": ("lv", "bv"),
    "blv_minus_bv": ("blv", "bv"),
    "blv_minus_lv": ("blv", "lv"),
    "blv_minus_bbv": ("blv", "bbv"),
    "lv_minus_b0": ("lv", "b0"),
    "lv_minus_f3": ("lv", "f3"),
    "blv_minus_b0": ("blv", "b0"),
    "blv_minus_f3": ("blv", "f3"),
    "bbv_minus_bv": ("bbv", "bv"),
}
LV_GATES = ("lv_minus_bv", "lv_minus_b0", "lv_minus_f3")
BLV_GATES = (
    "blv_minus_bv",
    "blv_minus_lv",
    "blv_minus_bbv",
    "blv_minus_b0",
    "blv_minus_f3",
)


def analyse(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    labels: IntArray,
    probabilities: Mapping[str, FloatArray],
    participants: StringArray,
    participant_folds: IntArray,
    roster: Sequence[str],
    b0_replay_exact: bool,
) -> dict[str, Any]:
    _require(set(reports) == set(CELL_ORDER) | {"f3"}, "analysis report set changed")
    comparisons = {
        name: paired_comparison(reports[left], reports[right])
        for name, (left, right) in CONTRAST_PAIRS.items()
    }
    leave_folds = {
        name: _leave_one_fold(reports[left], reports[right], participant_folds)
        for name, (left, right) in CONTRAST_PAIRS.items()
    }
    gated_names = (*LV_GATES, *BLV_GATES)
    gates = {
        name: _gate(comparisons[name], leave_folds[name], b0_replay_exact) for name in gated_names
    }
    lv_pass = all(gates[name]["status"] == "pass" for name in LV_GATES)
    blv_pass = all(gates[name]["status"] == "pass" for name in BLV_GATES)
    if blv_pass:
        decision = "blv_pass_propose_separately_authorized_replication"
    elif lv_pass:
        decision = "lv_pass_retain_ankle_when_aligned_back_fallback_no_automatic_replication"
    else:
        decision = "close_finite_spatial_information_package_retain_existing_baselines"
    events = {
        name: _events(
            labels,
            probabilities[right],
            probabilities[left],
            participants,
            roster,
        )
        for name, (left, right) in CONTRAST_PAIRS.items()
    }
    return {
        "status": "complete_analysis",
        "evidence_status": "corrected_fog_exploratory_development_not_confirmation",
        "reports": dict(reports),
        "comparisons": comparisons,
        "leave_one_outer_fold_sensitivity": leave_folds,
        "gates": gates,
        "advancement": {
            "lv": {"status": "pass" if lv_pass else "fail", "required": list(LV_GATES)},
            "blv": {"status": "pass" if blv_pass else "fail", "required": list(BLV_GATES)},
        },
        "event_topology": events,
        "decision": decision,
        "fallback_system_requires_back_sensor": True,
        "automatic_follow_on_launched": False,
        "bootstrap_scope": (
            "participant resampling conditional on fitted overlapping-fold models and an "
            "adaptively consumed development cohort"
        ),
    }


def _source_manifest(repository_root: Path) -> dict[str, Any]:
    paths = (
        "AGENTS.md",
        "src/inclusive_shift_har/data/external_har.py",
        "src/inclusive_shift_har/preprocessing/features.py",
        "src/inclusive_shift_har/experiments/fog_rf_feature_weight_factorial.py",
        "src/inclusive_shift_har/experiments/fog_spatial_information_probe.py",
        "configs/experiments/fog_spatial_information_probe_v1.yaml",
        "docs/research/FOG_SPATIAL_INFORMATION_PROBE_V1_PROTOCOL.md",
    )
    return {
        "record_kind": "fog_spatial_source_manifest",
        "files": [{"path": path, "sha256": sha256_file(repository_root / path)} for path in paths],
    }


def _snapshot_inputs(
    repository_root: Path, config_path: Path, protocol_path: Path, output_directory: Path
) -> None:
    _write_bytes_create_only(output_directory / "config_snapshot.yaml", config_path.read_bytes())
    _write_json_create_only(
        output_directory / "protocol_snapshot.json",
        _sealed(
            {
                "record_kind": "fog_spatial_protocol_snapshot",
                "source_path": protocol_path.relative_to(repository_root).as_posix(),
                "source_sha256": sha256_file(protocol_path),
                "text": protocol_path.read_text(encoding="utf-8"),
            }
        ),
    )


def _worker_baseline() -> dict[str, set[int]]:
    """Record workers that predate this controller so cleanup never targets unrelated work."""

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
    """Stop and account for Python workers created after the controller baseline."""

    baseline_pids = baseline["child_pids"]
    baseline_threads = baseline["thread_object_ids"]
    children = [
        child
        for child in multiprocessing.active_children()
        if child.is_alive() and child.pid is not None and int(child.pid) not in baseline_pids
    ]
    terminated_pids: list[int] = []
    for child in children:
        try:
            child_pid = child.pid
            if child_pid is None:
                continue
            terminated_pids.append(child_pid)
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
    stopped = not remaining_children and not remaining_threads
    return _sealed(
        {
            "record_kind": "fog_spatial_worker_shutdown",
            "observed_at_utc": _now(),
            "terminal_status": terminal,
            "fit_attempt_count": attempts,
            "completed_fit_count": completed,
            "baseline_multiprocessing_child_pids": sorted(baseline_pids),
            "task_owned_multiprocessing_child_pids_observed": sorted(
                int(child.pid) for child in children if child.pid is not None
            ),
            "task_owned_multiprocessing_child_pids_terminated": sorted(terminated_pids),
            "remaining_task_owned_multiprocessing_child_pids": sorted(
                int(child.pid) for child in remaining_children if child.pid is not None
            ),
            "task_owned_nonbaseline_python_threads_observed": sorted(
                thread.name for thread in threads
            ),
            "remaining_task_owned_nonbaseline_python_threads": sorted(
                thread.name for thread in remaining_threads
            ),
            "task_owned_fit_workers_and_monitors_stopped": stopped,
        }
    )


def _outcome_summary(analysis: Mapping[str, Any], availability: BoolArray) -> str:
    reports = _mapping(analysis["reports"], "reports")
    lines = [
        "# Corrected FoG spatial-information outcome",
        "",
        f"Decision: **{analysis['decision']}**.",
        "",
        "| Cell | Mean participant F1 | Pooled accuracy | Bottom 30% | Worst |",
        "|---|---:|---:|---:|---:|",
    ]
    for cell in (*CELL_ORDER, "f3"):
        report = _mapping(reports[cell], cell)
        primary = _mapping(report["primary"], "primary")
        pooled = _mapping(report["pooled"], "pooled")
        lines.append(
            f"| {cell.upper()} | {float(primary['mean_participant_macro_f1']):.6f} | "
            f"{float(pooled['accuracy']):.6f} | "
            f"{float(primary['bottom_30_percent_participant_macro_f1']):.6f} | "
            f"{float(primary['worst_participant_macro_f1']):.6f} |"
        )
    lines.extend(
        [
            "",
            "| Contrast | Mean difference | 95% participant bootstrap CI | Wins/harms/ties |",
            "|---|---:|---|---|",
        ]
    )
    comparisons = _mapping(analysis["comparisons"], "comparisons")
    for name in CONTRAST_PAIRS:
        row = _mapping(comparisons[name], name)
        interval = cast(list[float], row["mean_difference_95_percent_bootstrap_interval"])
        lines.append(
            f"| {name} | {float(row['mean_difference']):+.6f} | "
            f"[{interval[0]:+.6f}, {interval[1]:+.6f}] | "
            f"{row['participant_wins']}/{row['participant_harms']}/{row['participant_ties']} |"
        )
    lines.extend(
        [
            "",
            f"Ankle branch availability: {int(availability.sum())}/1939 candidates. "
            "Full-roster metrics use B0 fallback elsewhere.",
            "Lv is an ankle-when-aligned/back-fallback system, not a standalone ankle model.",
            "Evidence remains exploratory on the consumed FoG cohort. No additional seed, "
            "sensor, dataset, target cohort, or publication action was launched.",
            "",
        ]
    )
    return "\n".join(lines)


def _feature_cache_metadata(cache: Mapping[str, NDArray[Any]]) -> dict[str, Any]:
    return _sealed(
        {
            "record_kind": "fog_spatial_feature_cache",
            "features_built_before_outcome_fits": True,
            "unavailable_ankle_features_materialized": False,
            "arrays": {
                name: {
                    "shape": list(values.shape),
                    "dtype": str(values.dtype),
                    "sha256": _array_sha256(values),
                }
                for name, values in cache.items()
            },
        }
    )


def run_experiment(
    *,
    repository_root: Path,
    evidence_root: Path,
    config_path: Path,
    protocol_path: Path,
    output_directory: Path,
    code_commit: str,
    timeout_seconds: int,
    cancel_file: Path | None = None,
) -> dict[str, Any]:
    """Execute the source-frozen 25-fit probe and retain every terminal state."""

    repository_root = repository_root.resolve()
    evidence_root = evidence_root.resolve()
    config_path = config_path.resolve()
    protocol_path = protocol_path.resolve()
    output_directory = output_directory.resolve()
    _require(timeout_seconds == 3600, "compute timeout changed")
    _require(
        config_path == repository_root / "configs/experiments/fog_spatial_information_probe_v1.yaml"
        and protocol_path
        == repository_root / "docs/research/FOG_SPATIAL_INFORMATION_PROBE_V1_PROTOCOL.md",
        "config/protocol source path changed",
    )
    _require(
        output_directory.parent == evidence_root / ".audit" / "fog_spatial_information_probe",
        "wrong evidence family",
    )
    _require(not output_directory.exists(), "output directory exists")
    config = _read_yaml(config_path)
    validate_config(config)
    git_state = _git_state(repository_root)
    _require(git_state["clean"] is True, "source worktree must be clean")
    _require(git_state["commit"] == code_commit, "code commit argument differs from HEAD")
    output_directory.mkdir(parents=True)
    worker_baseline = _worker_baseline()
    started = time.perf_counter()
    deadline = started + timeout_seconds
    attempts = 0
    completed = 0
    fit_rows: list[dict[str, Any]] = []
    stage = "initialization"
    try:
        _check_cancel(cancel_file, "initialization")
        _snapshot_inputs(repository_root, config_path, protocol_path, output_directory)
        source_manifest = _sealed(_source_manifest(repository_root))
        _write_json_create_only(output_directory / "source_manifest.json", source_manifest)
        review = evidence_root / REFERENCE_RELATIVE["review"]
        raw_path = review / "inputs" / "sensor_data.csv"
        coverage_path = review / "COVERAGE_AUDIT.json"
        availability_path = review / "AVAILABILITY_CONTRACT_AUDIT.json"
        _require(
            sha256_file(raw_path) == RAW_SHA256 and raw_path.stat().st_size == RAW_SIZE,
            "raw source changed",
        )
        _require(sha256_file(coverage_path) == COVERAGE_SHA256, "coverage audit changed")
        _require(
            sha256_file(availability_path) == AVAILABILITY_SHA256,
            "availability audit changed",
        )
        coverage = _read_json(coverage_path)
        coverage_rows = cast(list[dict[str, Any]], coverage["candidate_availability"])
        availability = availability_mask_from_rows(coverage_rows)
        coverage_ids = np.asarray([row["window_id"] for row in coverage_rows], dtype=np.str_)
        _require(
            _availability_array_sha256(coverage_ids) == ORDERED_ID_SHA256,
            "coverage ID hash changed",
        )
        frozen_mask_sha = _availability_array_sha256(availability)
        availability_freeze = _sealed(
            {
                "record_kind": "fog_spatial_availability_freeze",
                "created_before_scoring_metadata_load": True,
                "coverage_audit_sha256": COVERAGE_SHA256,
                "raw_source_sha256": RAW_SHA256,
                "mask_definition": (
                    "raw_interval_contained AND independent_50hz_grid_matches_all128_times "
                    "AND exactly_one_matching_run"
                ),
                "mask_sha256": frozen_mask_sha,
                "ordered_observable_ids_sha256": _availability_array_sha256(coverage_ids),
                "observable_available_count": int(availability.sum()),
                "observable_fallback_count": int((~availability).sum()),
                "outcome_metadata_accessed": False,
            }
        )
        _write_json_create_only(output_directory / "availability_freeze.json", availability_freeze)
        _check_cancel(cancel_file, "before_scoring_metadata_load")
        stage = "reference_loading_after_availability_freeze"
        references, reference_receipt = _load_reference_arrays(evidence_root)
        _require(
            np.array_equal(coverage_ids, references["observable_window_ids"]),
            "coverage/reference candidate IDs differ",
        )
        _require(
            _availability_array_sha256(availability) == frozen_mask_sha,
            "availability changed after outcome metadata load",
        )
        support_audit = _read_json(availability_path)
        _require(
            support_audit["mask_sha256"] == frozen_mask_sha
            and support_audit["counts"]["valid_scored_rows"] == 1154
            and support_audit["counts"]["fallback_scored_rows"] == 59
            and support_audit["structural_gates"]["all5_training_partitions_have3_classes"] is True,
            "post-freeze availability support audit changed",
        )
        _write_json_create_only(
            output_directory / "reference_receipts.json", _sealed(reference_receipt)
        )
        source_receipt = _sealed(
            {
                "record_kind": "fog_spatial_input_receipt",
                "raw_path": str(raw_path),
                "raw_sha256": RAW_SHA256,
                "raw_size_bytes": RAW_SIZE,
                "coverage_audit_path": str(coverage_path),
                "coverage_audit_sha256": COVERAGE_SHA256,
                "availability_audit_path": str(availability_path),
                "availability_audit_sha256": AVAILABILITY_SHA256,
                "mask_frozen_before_scoring_metadata_load": True,
                "availability_freeze_record_sha256": availability_freeze["record_sha256"],
                "mask_sha256": frozen_mask_sha,
                "observable_available_count": int(availability.sum()),
                "observable_fallback_count": int((~availability).sum()),
                "fixed_sensor": "ankleL",
                "InclusiveHAR_P11_P20_loaded": False,
            }
        )
        _write_json_create_only(output_directory / "availability_receipt.json", source_receipt)
        stage = "label_free_ankle_materialization"
        _check_cancel(cancel_file, "before_ankle_materialization")
        (
            ankle_windows,
            ankle_gravity,
            available_indices,
            alignment_receipts,
            segment_receipts,
        ) = _materialize_ankle_windows(raw_path, coverage_rows, availability)
        ankle_batch = extract_engineered_features(ankle_windows, channel_names=SIX_CHANNEL_NAMES)
        _require(_array_sha256(ankle_windows) == ANKLE_SIGNAL_SHA256, "ankle signals changed")
        _require(_array_sha256(ankle_gravity) == ANKLE_GRAVITY_SHA256, "ankle gravity changed")
        _require(
            canonical_json_sha256(segment_receipts) == ANKLE_SEGMENT_RECEIPTS_SHA256,
            "ankle segment audit changed",
        )
        _require(
            _array_sha256(ankle_batch.values) == ANKLE_FEATURE_SHA256,
            "ankle features changed",
        )
        back_values = cast(FloatArray, references["back_values"])
        back_names = tuple(cast(StringArray, references["back_names"]).tolist())
        _require(back_values.shape == (1939, 80), "back feature shape changed")
        _require(
            _array_sha256(back_values)
            == "b890eaa901357cb7ef4ab950665e090a92664e1796290f233cf30a93f1505ad4",
            "back features changed",
        )
        _require(tuple(ankle_batch.names) == back_names, "ankle feature schema changed")
        blocks = make_feature_blocks(
            back_values[available_indices],
            ankle_batch.values,
            back_names,
            ankle_batch.names,
        )
        cache = {
            "back_values": back_values,
            "back_names": np.asarray(back_names, dtype=np.str_),
            "ankle_values": ankle_batch.values,
            "ankle_names": np.asarray(ankle_batch.names, dtype=np.str_),
            "ankle_gravity": ankle_gravity,
            "available_indices": available_indices,
            "availability_mask": availability,
            "observable_window_ids": cast(StringArray, references["observable_window_ids"]),
            "observable_participant_ids": cast(
                StringArray, references["observable_participant_ids"]
            ),
            "observable_fold_index": cast(IntArray, references["observable_fold_index"]),
            "scoring_indices": cast(IntArray, references["scoring_indices"]),
            "scoring_eligibility": cast(BoolArray, references["scoring_eligibility"]),
            "scored_labels": cast(IntArray, references["scored_labels"]),
            "scored_participant_ids": cast(StringArray, references["scored_participant_ids"]),
            "scored_window_ids": cast(StringArray, references["scored_window_ids"]),
            "scored_fold_index": cast(IntArray, references["scored_fold_index"]),
        }
        _write_npz_create_only(output_directory / "feature_cache.npz", **cache)
        feature_metadata = _feature_cache_metadata(cache)
        feature_metadata["ankle_alignment_receipts"] = alignment_receipts
        feature_metadata["ankle_segment_receipts"] = segment_receipts
        feature_metadata = _sealed(
            {k: v for k, v in feature_metadata.items() if k != "record_sha256"}
        )
        _write_json_create_only(output_directory / "feature_cache_metadata.json", feature_metadata)
        labels = cast(IntArray, references["scored_labels"])
        eligible = cast(BoolArray, references["scoring_eligibility"])
        people = cast(StringArray, references["observable_participant_ids"])
        folds = cast(IntArray, references["observable_fold_index"])
        observable_labels = np.zeros(1939, dtype=np.int64)
        observable_labels[eligible] = labels
        preflight_partitions = partition_preflight(
            labels=labels,
            observable_participants=people,
            observable_folds=folds,
            scoring_eligibility=eligible,
            availability=availability,
        )
        _write_json_create_only(
            output_directory / "partition_preflight.json", _sealed(preflight_partitions)
        )
        reference_environment = _read_json(
            evidence_root / REFERENCE_RELATIVE["gsp"] / "preflight.json"
        )["environment"]
        environment = _environment()
        _require(
            str(environment["python"]).startswith("3.11.9 ")
            and environment["numpy"] == "2.3.5"
            and environment["scikit_learn"] == "1.8.0"
            and environment["python"] == reference_environment["python"]
            and Path(str(environment["python_executable"])).resolve()
            == Path(str(reference_environment["python_executable"])).resolve(),
            "runtime differs from B0 reference",
        )
        preflight = _sealed(
            {
                "record_kind": "fog_spatial_preflight",
                "created_before_model_fits": True,
                "repository_root": str(repository_root),
                "evidence_root": str(evidence_root),
                "code_commit": code_commit,
                "git": git_state,
                "environment": environment,
                "configuration_sha256": sha256_file(config_path),
                "protocol_sha256": sha256_file(protocol_path),
                "source_manifest_record_sha256": source_manifest["record_sha256"],
                "availability_record_sha256": source_receipt["record_sha256"],
                "feature_cache_record_sha256": feature_metadata["record_sha256"],
                "partition_record_sha256": _read_json(
                    output_directory / "partition_preflight.json"
                )["record_sha256"],
                "expected_fit_count": 25,
                "effective_parameter_sets": {
                    f"{cell}--fold-{fold}": _effective_parameters(11 + fold)
                    for cell in CELL_ORDER
                    for fold in OUTER_FOLDS
                },
                "new_candidate_model_outcomes_available_when_written": False,
                "archived_control_probabilities_loaded": True,
                "scored_labels_loaded_after_availability_freeze": True,
                "compute_cap_seconds": timeout_seconds,
            }
        )
        _write_json_create_only(output_directory / "preflight.json", preflight)
        input_manifest = _sealed(
            {
                "record_kind": "fog_spatial_input_manifest",
                "raw_source": {"path": str(raw_path), "sha256": RAW_SHA256, "size_bytes": RAW_SIZE},
                "coverage": {"path": str(coverage_path), "sha256": COVERAGE_SHA256},
                "availability": {"path": str(availability_path), "sha256": AVAILABILITY_SHA256},
                "references": reference_receipt["checked_files"],
            }
        )
        _write_json_create_only(output_directory / "input_manifest.json", input_manifest)
        _require(time.perf_counter() < deadline, "compute cap reached before fits")
        preflight_seconds = time.perf_counter() - started
        stage = "b0_fits"
        final_probabilities = {
            cell: np.full((1939, 3), np.nan, dtype=np.float64) for cell in CELL_ORDER
        }
        available_branch = {
            cell: np.full((1817, 3), np.nan, dtype=np.float64) for cell in RESTRICTED_CELLS
        }
        compact_lookup = np.full(1939, -1, dtype=np.int64)
        compact_lookup[available_indices] = np.arange(1817, dtype=np.int64)
        for fold in OUTER_FOLDS:
            _check_cancel(cancel_file, f"before_b0_fold_{fold}")
            _require(time.perf_counter() < deadline, "compute cap reached before next fit")
            masks = fold_masks(folds, eligible, availability, fold)
            train = masks["b0_training"]
            evaluate = masks["b0_evaluation"]
            attempts += 1
            probabilities, metadata = _fit_model(
                training_values=back_values[train],
                training_labels=observable_labels[train],
                training_participants=people[train],
                evaluation_values=back_values[evaluate],
                evaluation_participants=people[evaluate],
                training_global_indices=np.flatnonzero(train).astype(np.int64),
                evaluation_global_indices=np.flatnonzero(evaluate).astype(np.int64),
                cell_id="b0",
                outer_fold=fold,
                attempt_number=attempts,
                feature_names=tuple(f"back::{name}" for name in back_names),
                output_directory=output_directory,
                deadline=deadline,
            )
            completed += 1
            _check_cancel(cancel_file, f"after_b0_fold_{fold}")
            fit_rows.append(metadata)
            final_probabilities["b0"][evaluate] = probabilities
        b0_reference = cast(FloatArray, references["b0_reference_probabilities"])
        b0_replay = assert_b0_replay(final_probabilities["b0"], b0_reference)
        b0_replay["back_features_byte_exact_to_pinned_reference"] = True
        _require(
            tuple(
                str(_mapping(row["sample_weight_summary"], "sample weights")["sha256"])
                for row in fit_rows
            )
            == B0_REFERENCE_WEIGHT_SHA256
            and tuple(int(row["training_row_count"]) for row in fit_rows)
            == B0_REFERENCE_TRAINING_ROWS
            and tuple(cast(StringArray, references["b0_reference_weight_hashes"]).tolist())
            == B0_REFERENCE_WEIGHT_SHA256,
            "fresh B0 training weights differ from archived controls",
        )
        _require(attempts == 5, "restricted fits started before B0 barrier")
        stage = "restricted_fits"
        for cell in RESTRICTED_CELLS:
            feature_values, feature_names = blocks[cell]
            final_probabilities[cell] = final_probabilities["b0"].copy()
            for fold in OUTER_FOLDS:
                _check_cancel(cancel_file, f"before_{cell}_fold_{fold}")
                _require(time.perf_counter() < deadline, "compute cap reached before next fit")
                masks = fold_masks(folds, eligible, availability, fold)
                train_global = masks["restricted_training"]
                eval_global = masks["restricted_evaluation"]
                train_compact = compact_lookup[np.flatnonzero(train_global)]
                eval_compact = compact_lookup[np.flatnonzero(eval_global)]
                _require(
                    bool(np.all(train_compact >= 0) and np.all(eval_compact >= 0)),
                    "restricted row lacks features",
                )
                attempts += 1
                probabilities, metadata = _fit_model(
                    training_values=feature_values[train_compact],
                    training_labels=observable_labels[train_global],
                    training_participants=people[train_global],
                    evaluation_values=feature_values[eval_compact],
                    evaluation_participants=people[eval_global],
                    training_global_indices=np.flatnonzero(train_global).astype(np.int64),
                    evaluation_global_indices=np.flatnonzero(eval_global).astype(np.int64),
                    cell_id=cell,
                    outer_fold=fold,
                    attempt_number=attempts,
                    feature_names=feature_names,
                    output_directory=output_directory,
                    deadline=deadline,
                )
                completed += 1
                _check_cancel(cancel_file, f"after_{cell}_fold_{fold}")
                fit_rows.append(metadata)
                final_probabilities[cell][eval_global] = probabilities
                available_branch[cell][eval_compact] = probabilities
            final_probabilities[cell] = apply_back_fallback(
                final_probabilities["b0"], availability, available_branch[cell]
            )
            _require(
                np.array_equal(
                    final_probabilities[cell][~availability],
                    final_probabilities["b0"][~availability],
                ),
                f"{cell} fallback differs from B0",
            )
            _require(bool(np.isfinite(available_branch[cell]).all()), f"{cell} branch incomplete")
        _require(attempts == completed == 25 and len(fit_rows) == 25, "fit count changed")
        stage = "analysis"
        _check_cancel(cancel_file, "before_analysis")
        scored_indices = cast(IntArray, references["scoring_indices"])
        scored_people = cast(StringArray, references["scored_participant_ids"])
        roster = cast(list[str], _mapping(config["source"], "source")["participant_roster"])
        scored_probabilities = {
            cell: final_probabilities[cell][scored_indices] for cell in CELL_ORDER
        }
        scored_probabilities["f3"] = cast(FloatArray, references["f3_reference_probabilities"])[
            scored_indices
        ]
        reports = {
            cell: method_report(
                labels=labels,
                probabilities=probabilities,
                participant_ids=scored_people,
                roster=roster,
            )
            for cell, probabilities in scored_probabilities.items()
        }
        participant_folds = np.asarray(
            [int(np.unique(folds[people == person])[0]) for person in roster], dtype=np.int64
        )
        analysis = analyse(
            reports,
            labels=labels,
            probabilities=scored_probabilities,
            participants=scored_people,
            participant_folds=participant_folds,
            roster=roster,
            b0_replay_exact=True,
        )
        scored_available = availability[scored_indices]
        strata = {
            "available": {
                cell: method_report(
                    labels=labels[scored_available],
                    probabilities=probabilities[scored_available],
                    participant_ids=scored_people[scored_available],
                    roster=roster,
                )
                for cell, probabilities in scored_probabilities.items()
            },
            "fallback": {
                cell: method_report(
                    labels=labels[~scored_available],
                    probabilities=probabilities[~scored_available],
                    participant_ids=scored_people[~scored_available],
                    roster=roster,
                )
                for cell, probabilities in scored_probabilities.items()
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
            },
        }
        analysis_record = _sealed(
            {"record_kind": "fog_spatial_analysis", **analysis, "strata": strata}
        )
        _write_json_create_only(output_directory / "analysis.json", analysis_record)
        _write_json_create_only(
            output_directory / "participant_metrics.json",
            _sealed(
                {
                    "record_kind": "fog_spatial_participant_metrics",
                    "methods": reports,
                    "strata": strata,
                }
            ),
        )
        _write_json_create_only(
            output_directory / "fit_reports.json",
            _sealed(
                {
                    "record_kind": "fog_spatial_fit_reports",
                    "fit_attempt_count": attempts,
                    "completed_fit_count": completed,
                    "rows": fit_rows,
                }
            ),
        )
        prediction_arrays: dict[str, NDArray[Any]] = {
            "cell_ids": np.asarray(CELL_ORDER, dtype=np.str_),
            "observable_probabilities": np.stack(
                [final_probabilities[cell] for cell in CELL_ORDER]
            ),
            "scored_probabilities": np.stack([scored_probabilities[cell] for cell in CELL_ORDER]),
            "available_branch_cell_ids": np.asarray(RESTRICTED_CELLS, dtype=np.str_),
            "available_branch_probabilities": np.stack(
                [available_branch[cell] for cell in RESTRICTED_CELLS]
            ),
            "f3_reference_probabilities": cast(
                FloatArray, references["f3_reference_probabilities"]
            ),
            "availability_mask": availability,
            "available_indices": available_indices,
            "scoring_indices": scored_indices,
            "scored_labels": labels,
            "observable_window_ids": cast(StringArray, references["observable_window_ids"]),
            "observable_participant_ids": people,
            "observable_fold_index": folds,
            "scored_participant_ids": scored_people,
            "scored_window_ids": cast(StringArray, references["scored_window_ids"]),
            "scored_fold_index": cast(IntArray, references["scored_fold_index"]),
            "explicit_decisions": np.stack(
                [final_probabilities[cell].argmax(axis=1) for cell in CELL_ORDER]
            ).astype(np.int64),
            "fallback_source_cell_index": np.zeros((4, 122), dtype=np.int64),
        }
        _write_npz_create_only(output_directory / "predictions.npz", **prediction_arrays)
        result = _sealed(
            {
                "record_kind": "fog_spatial_result",
                "status": "complete_awaiting_independent_replay",
                "experiment_id": EXPERIMENT_ID,
                "evidence_status": analysis["evidence_status"],
                "code_commit": code_commit,
                "fit_attempt_count": attempts,
                "completed_fit_count": completed,
                "observable_candidate_count": 1939,
                "scored_window_count": 1213,
                "observable_available_count": int(availability.sum()),
                "scored_available_count": int(scored_available.sum()),
                "method_summary": {
                    cell: {
                        "mean_participant_macro_f1": reports[cell]["primary"][
                            "mean_participant_macro_f1"
                        ],
                        "pooled_accuracy": reports[cell]["pooled"]["accuracy"],
                        "bottom_30_percent_participant_macro_f1": reports[cell]["primary"][
                            "bottom_30_percent_participant_macro_f1"
                        ],
                        "worst_participant_macro_f1": reports[cell]["primary"][
                            "worst_participant_macro_f1"
                        ],
                    }
                    for cell in (*CELL_ORDER, "f3")
                },
                "advancement": analysis["advancement"],
                "decision": analysis["decision"],
                "b0_replay": b0_replay,
                "InclusiveHAR_P11_P20_loaded": False,
                "additional_seeds_launched": False,
                "automatic_follow_on_launched": False,
            }
        )
        _write_json_create_only(output_directory / "result.json", result)
        _write_bytes_create_only(
            output_directory / "OUTCOME_SUMMARY.md",
            _outcome_summary(analysis, availability).encode(),
        )
        _require(time.perf_counter() - started <= timeout_seconds, "compute cap exceeded")
        worker = _stop_task_owned_workers(
            worker_baseline, attempts, completed, terminal="fit_phase_complete"
        )
        _write_json_create_only(output_directory / "worker_shutdown.json", worker)
        _require(
            worker["task_owned_fit_workers_and_monitors_stopped"] is True,
            "task-owned fit worker remains",
        )
        elapsed = time.perf_counter() - started
        _require(elapsed <= timeout_seconds, "compute cap exceeded during worker shutdown")
        runtime = _sealed(
            {
                "record_kind": "fog_spatial_runtime",
                "compute_cap_seconds": timeout_seconds,
                "total_run_seconds": elapsed,
                "measured_scope": (
                    "initialization through all fits, analysis, prediction/result/summary writes, "
                    "and task-owned worker shutdown"
                ),
                "unmeasured_finalization_operations": [
                    "runtime.json serialization",
                    "artifact_manifest.json hashing and serialization",
                ],
                "fit_attempt_count": attempts,
                "completed_fit_count": completed,
                "feature_and_preflight_seconds": preflight_seconds,
                "fit_and_prediction_seconds": sum(
                    float(row["fit_and_predict_seconds"]) for row in fit_rows
                ),
                "cap_exceeded": False,
                "execution_parallelism": "sequential_models_single_controller",
                "fit_worker_count": 4,
                "prediction_worker_count": 1,
                "blas_thread_limit": 1,
            }
        )
        _write_json_create_only(output_directory / "runtime.json", runtime)
        manifest_files = [
            path
            for path in output_directory.rglob("*")
            if path.is_file()
            and path.name
            not in {
                "artifact_manifest.json",
                "validation.json",
                "completion_manifest.json",
                "VALIDATION_INCOMPLETE.json",
            }
        ]
        manifest = _sealed(
            {
                "record_kind": "fog_spatial_artifact_manifest",
                "artifacts": [
                    {
                        "path": path.relative_to(output_directory).as_posix(),
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in sorted(manifest_files)
                ],
            }
        )
        _write_json_create_only(output_directory / "artifact_manifest.json", manifest)
        return result
    except BaseException as exc:
        cleanup_error: str | None = None
        try:
            shutdown = _stop_task_owned_workers(
                worker_baseline, attempts, completed, terminal="incomplete_cleanup"
            )
            if not (output_directory / "worker_shutdown.json").exists():
                _write_json_create_only(output_directory / "worker_shutdown.json", shutdown)
        except BaseException as cleanup_exc:
            cleanup_error = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
        try:
            incomplete = _sealed(
                {
                    "record_kind": "fog_spatial_incomplete",
                    "status": "incomplete_blocker",
                    "failed_stage": stage,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "fit_attempt_count": attempts,
                    "completed_fit_count": completed,
                    "elapsed_seconds": time.perf_counter() - started,
                    "worker_cleanup_error": cleanup_error,
                    "no_retry_launched": True,
                    "automatic_follow_on_launched": False,
                }
            )
            _write_json_create_only(output_directory / "INCOMPLETE.json", incomplete)
        except BaseException:
            pass
        raise


def _validate_run_impl(
    run_directory: Path, validation_worker_baseline: Mapping[str, set[int]]
) -> dict[str, Any]:
    """Replay all 25 saved checkpoints and seal a completed run without fitting."""

    validation_started = time.perf_counter()
    run_directory = run_directory.resolve()
    _require(run_directory.is_dir(), "run directory missing")
    _require(not (run_directory / "INCOMPLETE.json").exists(), "incomplete run cannot validate")
    _require(
        not (run_directory / "VALIDATION_INCOMPLETE.json").exists(),
        "validation failure already recorded",
    )
    _require(not (run_directory / "validation.json").exists(), "validation already exists")
    _require(not (run_directory / "completion_manifest.json").exists(), "completion exists")
    config = _read_yaml(run_directory / "config_snapshot.yaml")
    validate_config(config)
    artifact_count = _verify_manifest(run_directory, "artifact_manifest.json")
    _require(
        all((run_directory / path).is_file() for path in REQUIRED_PREVALIDATION_FILES),
        "required prevalidation artifact missing",
    )
    cache = _read_npz(run_directory / "feature_cache.npz")
    expected_cache_keys = {
        "back_values",
        "back_names",
        "ankle_values",
        "ankle_names",
        "ankle_gravity",
        "available_indices",
        "availability_mask",
        "observable_window_ids",
        "observable_participant_ids",
        "observable_fold_index",
        "scoring_indices",
        "scoring_eligibility",
        "scored_labels",
        "scored_participant_ids",
        "scored_window_ids",
        "scored_fold_index",
    }
    _require(set(cache) == expected_cache_keys, "feature-cache schema changed")
    metadata = _read_json(run_directory / "feature_cache_metadata.json")
    _verify_sealed(metadata, "feature metadata")
    metadata_arrays = _mapping(metadata["arrays"], "cache arrays")
    _require(set(metadata_arrays) == expected_cache_keys, "feature metadata schema changed")
    for name, item in metadata_arrays.items():
        row = _mapping(item, name)
        _require(
            name in cache
            and list(cache[name].shape) == row["shape"]
            and str(cache[name].dtype) == row["dtype"]
            and _array_sha256(cache[name]) == row["sha256"],
            f"cache array changed: {name}",
        )
    availability = np.asarray(cache["availability_mask"], dtype=np.bool_)
    available_indices = np.asarray(cache["available_indices"], dtype=np.int64)
    _require(_availability_array_sha256(availability) == MASK_SHA256, "mask changed")
    _require(
        _array_sha256(cache["ankle_values"]) == ANKLE_FEATURE_SHA256,
        "ankle feature hash changed",
    )
    preflight = _read_json(run_directory / "preflight.json")
    _verify_sealed(preflight, "preflight")
    repository_root = Path(str(preflight["repository_root"])).resolve()
    evidence_root = Path(str(preflight["evidence_root"])).resolve()
    _require(repository_root.is_dir(), "source repository missing")
    source_manifest = _read_json(run_directory / "source_manifest.json")
    _verify_sealed(source_manifest, "source manifest")
    _require(
        source_manifest["record_sha256"] == preflight["source_manifest_record_sha256"],
        "preflight/source-manifest binding changed",
    )
    for item in cast(list[dict[str, Any]], source_manifest["files"]):
        path = (repository_root / str(item["path"])).resolve()
        _require(path.is_relative_to(repository_root), "source path escapes repository")
        _require(sha256_file(path) == item["sha256"], f"source changed: {path}")
    source_manifest_body = dict(source_manifest)
    source_manifest_body.pop("record_sha256")
    _require(
        canonical_json_sha256(source_manifest_body)
        == canonical_json_sha256(_source_manifest(repository_root)),
        "source manifest membership changed",
    )
    protocol_snapshot = _read_json(run_directory / "protocol_snapshot.json")
    _verify_sealed(protocol_snapshot, "protocol snapshot")
    protocol_source = (repository_root / str(protocol_snapshot["source_path"])).resolve()
    _require(
        protocol_source.is_relative_to(repository_root)
        and sha256_file(protocol_source) == protocol_snapshot["source_sha256"]
        and protocol_snapshot["source_sha256"] == preflight["protocol_sha256"]
        and protocol_source.read_text(encoding="utf-8") == protocol_snapshot["text"],
        "protocol snapshot/source binding changed",
    )
    _require(
        sha256_file(run_directory / "config_snapshot.yaml") == preflight["configuration_sha256"],
        "config snapshot/preflight binding changed",
    )
    current_git = _git_state(repository_root)
    _require(
        current_git["commit"] == preflight["code_commit"] and current_git["clean"] is True,
        "source commit/worktree changed",
    )
    inputs = _read_json(run_directory / "input_manifest.json")
    _verify_sealed(inputs, "input manifest")
    input_paths: dict[str, Path] = {}
    for name in ("raw_source", "coverage", "availability"):
        item = _mapping(inputs[name], name)
        path = Path(str(item["path"])).resolve()
        input_paths[name] = path
        _require(path.is_file() and sha256_file(path) == item["sha256"], f"input changed: {name}")
        if "size_bytes" in item:
            _require(path.stat().st_size == int(item["size_bytes"]), f"input size changed: {name}")
    for rows in _mapping(inputs["references"], "references").values():
        for item in cast(list[dict[str, Any]], rows):
            path = Path(str(item["path"])).resolve()
            _require(path.is_file() and sha256_file(path) == item["sha256"], "reference changed")
    references, fresh_reference_receipt = _load_reference_arrays(evidence_root)
    stored_reference_receipt = _read_json(run_directory / "reference_receipts.json")
    _verify_sealed(stored_reference_receipt, "reference receipt")
    stored_reference_body = dict(stored_reference_receipt)
    stored_reference_body.pop("record_sha256")
    _require(
        canonical_json_sha256(stored_reference_body)
        == canonical_json_sha256(fresh_reference_receipt),
        "reference receipt replay differs",
    )
    availability_freeze = _read_json(run_directory / "availability_freeze.json")
    availability_receipt = _read_json(run_directory / "availability_receipt.json")
    _verify_sealed(availability_freeze, "availability freeze")
    _verify_sealed(availability_receipt, "availability receipt")
    _require(
        availability_freeze["created_before_scoring_metadata_load"] is True
        and availability_freeze["mask_sha256"] == MASK_SHA256
        and availability_receipt["availability_freeze_record_sha256"]
        == availability_freeze["record_sha256"],
        "availability freeze/receipt binding changed",
    )
    _require(
        preflight["availability_record_sha256"] == availability_receipt["record_sha256"]
        and preflight["feature_cache_record_sha256"] == metadata["record_sha256"]
        and preflight["new_candidate_model_outcomes_available_when_written"] is False
        and preflight["archived_control_probabilities_loaded"] is True
        and preflight["scored_labels_loaded_after_availability_freeze"] is True,
        "preflight evidence binding changed",
    )
    coverage_payload = _read_json(input_paths["coverage"])
    coverage_rows = cast(list[dict[str, Any]], coverage_payload["candidate_availability"])
    fresh_availability = availability_mask_from_rows(coverage_rows)
    _require(np.array_equal(fresh_availability, availability), "availability replay differs")
    (
        fresh_ankle_windows,
        fresh_ankle_gravity,
        fresh_available_indices,
        fresh_alignment_receipts,
        fresh_segment_receipts,
    ) = _materialize_ankle_windows(input_paths["raw_source"], coverage_rows, fresh_availability)
    _require(
        _array_sha256(fresh_ankle_windows) == ANKLE_SIGNAL_SHA256,
        "replayed ankle signals changed",
    )
    _require(
        _array_sha256(fresh_ankle_gravity) == ANKLE_GRAVITY_SHA256,
        "replayed ankle gravity changed",
    )
    fresh_ankle_batch = extract_engineered_features(
        fresh_ankle_windows, channel_names=SIX_CHANNEL_NAMES
    )
    _require(
        np.array_equal(fresh_available_indices, available_indices)
        and np.array_equal(fresh_ankle_batch.values, cache["ankle_values"])
        and np.array_equal(fresh_ankle_gravity, cache["ankle_gravity"])
        and tuple(fresh_ankle_batch.names) == tuple(cache["ankle_names"].tolist())
        and canonical_json_sha256(fresh_alignment_receipts)
        == canonical_json_sha256(metadata["ankle_alignment_receipts"]),
        "raw ankle materialization replay differs",
    )
    _require(
        canonical_json_sha256(fresh_segment_receipts)
        == canonical_json_sha256(metadata["ankle_segment_receipts"])
        == ANKLE_SEGMENT_RECEIPTS_SHA256,
        "raw ankle segment audit replay differs",
    )
    del fresh_ankle_windows, fresh_ankle_gravity, fresh_ankle_batch
    back = np.asarray(cache["back_values"], dtype=np.float64)
    ankle = np.asarray(cache["ankle_values"], dtype=np.float64)
    back_names = tuple(cache["back_names"].tolist())
    ankle_names = tuple(cache["ankle_names"].tolist())
    for name in (
        "back_values",
        "back_names",
        "observable_window_ids",
        "observable_participant_ids",
        "observable_fold_index",
        "scoring_indices",
        "scoring_eligibility",
        "scored_labels",
        "scored_participant_ids",
        "scored_window_ids",
        "scored_fold_index",
    ):
        reference_name = name
        _require(
            np.array_equal(cache[name], references[reference_name]), f"reference differs: {name}"
        )
    _require(
        np.array_equal(available_indices, np.flatnonzero(availability)),
        "available indices changed",
    )
    _require(ankle_names == back_names, "ankle/back feature-name schema differs")
    blocks = make_feature_blocks(back[available_indices], ankle, back_names, ankle_names)
    people = np.asarray(cache["observable_participant_ids"], dtype=np.str_)
    folds = np.asarray(cache["observable_fold_index"], dtype=np.int64)
    eligible = np.asarray(cache["scoring_eligibility"], dtype=np.bool_)
    labels = np.asarray(cache["scored_labels"], dtype=np.int64)
    observable_labels = np.zeros(1939, dtype=np.int64)
    observable_labels[eligible] = labels
    stored = _read_npz(run_directory / "predictions.npz")
    expected_prediction_keys = {
        "cell_ids",
        "observable_probabilities",
        "scored_probabilities",
        "available_branch_cell_ids",
        "available_branch_probabilities",
        "f3_reference_probabilities",
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
        "explicit_decisions",
        "fallback_source_cell_index",
    }
    _require(set(stored) == expected_prediction_keys, "prediction archive schema changed")
    _require(stored["cell_ids"].tolist() == list(CELL_ORDER), "cell order changed")
    _require(
        stored["available_branch_cell_ids"].tolist() == list(RESTRICTED_CELLS)
        and np.array_equal(stored["availability_mask"], availability)
        and np.array_equal(stored["available_indices"], available_indices)
        and np.array_equal(stored["scoring_indices"], cache["scoring_indices"])
        and np.array_equal(stored["scored_labels"], labels)
        and np.array_equal(stored["observable_window_ids"], cache["observable_window_ids"])
        and np.array_equal(
            stored["observable_participant_ids"], cache["observable_participant_ids"]
        )
        and np.array_equal(stored["observable_fold_index"], folds)
        and np.array_equal(stored["scored_participant_ids"], cache["scored_participant_ids"])
        and np.array_equal(stored["scored_window_ids"], cache["scored_window_ids"])
        and np.array_equal(stored["scored_fold_index"], cache["scored_fold_index"]),
        "prediction provenance arrays changed",
    )
    _require(
        stored["fallback_source_cell_index"].shape == (4, 122)
        and bool(np.all(stored["fallback_source_cell_index"] == 0)),
        "fallback-source encoding changed",
    )
    replay = {cell: np.full((1939, 3), np.nan, dtype=np.float64) for cell in CELL_ORDER}
    compact_lookup = np.full(1939, -1, dtype=np.int64)
    compact_lookup[available_indices] = np.arange(1817, dtype=np.int64)
    reports_record = _read_json(run_directory / "fit_reports.json")
    _verify_sealed(reports_record, "fit reports")
    rows = cast(list[dict[str, Any]], reports_record["rows"])
    _require(
        len(rows) == 25
        and reports_record["fit_attempt_count"] == 25
        and reports_record["completed_fit_count"] == 25,
        "fit ledger changed",
    )
    expected_order = [(cell, fold) for cell in CELL_ORDER for fold in OUTER_FOLDS]
    _require(
        [(str(row["cell_id"]), int(row["outer_fold"])) for row in rows] == expected_order
        and [int(row["attempt_number"]) for row in rows] == list(range(1, 26)),
        "fit-attempt order changed",
    )
    expected_started = {f"{attempt:02d}--started.json" for attempt in range(1, 26)}
    expected_completed = {f"{attempt:02d}--completed.json" for attempt in range(1, 26)}
    expected_checkpoints = {f"{cell}--fold-{fold}.pkl" for cell, fold in expected_order}
    _require(
        {path.name for path in (run_directory / "fit_attempts").iterdir() if path.is_file()}
        == expected_started | expected_completed
        and {path.name for path in (run_directory / "checkpoints").iterdir() if path.is_file()}
        == expected_checkpoints,
        "attempt/checkpoint artifact family changed",
    )
    for attempt, row in enumerate(rows, start=1):
        started_record = _read_json(run_directory / "fit_attempts" / f"{attempt:02d}--started.json")
        completed_record = _read_json(
            run_directory / "fit_attempts" / f"{attempt:02d}--completed.json"
        )
        _verify_sealed(started_record, f"fit attempt {attempt} started")
        _verify_sealed(completed_record, f"fit attempt {attempt} completed")
        shared = ("attempt_number", "cell_id", "outer_fold")
        _require(
            all(started_record[name] == row[name] for name in shared)
            and all(completed_record[name] == row[name] for name in shared)
            and started_record["random_state"] == row["random_state"]
            and started_record["training_row_count"] == row["training_row_count"]
            and started_record["evaluation_candidate_count"] == row["evaluation_candidate_count"]
            and completed_record["fit_seconds"] == row["fit_seconds"]
            and completed_record["prediction_seconds"] == row["prediction_seconds"]
            and completed_record["checkpoint"] == row["checkpoint"],
            f"fit-attempt receipt differs: {attempt}",
        )
    lookup = {(str(row["cell_id"]), int(row["outer_fold"])): row for row in rows}
    _require(len(lookup) == 25, "fit reports duplicate")
    for cell in CELL_ORDER:
        for fold in OUTER_FOLDS:
            if cell == "b0":
                masks = fold_masks(folds, eligible, availability, fold)
                training = masks["b0_training"]
                evaluation = masks["b0_evaluation"]
                evaluation_values = back[evaluation]
                feature_names = tuple(f"back::{name}" for name in back_names)
            else:
                masks = fold_masks(folds, eligible, availability, fold)
                training = masks["restricted_training"]
                evaluation = masks["restricted_evaluation"]
                block, feature_names = blocks[cell]
                evaluation_values = block[compact_lookup[np.flatnonzero(evaluation)]]
            row = lookup[(cell, fold)]
            checkpoint = _mapping(row["checkpoint"], "checkpoint")
            path = (run_directory / str(checkpoint["path"])).resolve()
            _require(path.is_relative_to(run_directory), "checkpoint escapes run")
            _require(sha256_file(path) == checkpoint["sha256"], "checkpoint hash changed")
            checkpoint_metadata, estimator = _load_checkpoint(path)
            _require(
                checkpoint_metadata == {k: v for k, v in row.items() if k != "checkpoint"},
                "checkpoint metadata differs",
            )
            _require(
                estimator.get_params(deep=False) == _effective_parameters(11 + fold),
                "RF parameters changed",
            )
            _require(tuple(row["feature_names"]) == feature_names, "feature schema changed")
            expected_weights = participant_first_weights(
                observable_labels[training], people[training]
            )
            _require(
                row["sample_weight_summary"]["sha256"] == _array_sha256(expected_weights),
                "weights changed",
            )
            if cell == "b0":
                _require(
                    row["sample_weight_summary"]["sha256"] == B0_REFERENCE_WEIGHT_SHA256[fold]
                    and row["training_row_count"] == B0_REFERENCE_TRAINING_ROWS[fold],
                    "fresh B0 weights differ from archived control",
                )
            _require(
                row["training_global_indices_sha256"]
                == _array_sha256(np.flatnonzero(training).astype(np.int64))
                and row["evaluation_global_indices_sha256"]
                == _array_sha256(np.flatnonzero(evaluation).astype(np.int64)),
                "checkpoint partition changed",
            )
            with threadpool_limits(limits=1):
                values = _predict_proba_deterministically(estimator, evaluation_values)
            if cell == "b0":
                replay[cell][evaluation] = values
            else:
                if not np.isfinite(replay[cell]).any():
                    replay[cell] = replay["b0"].copy()
                replay[cell][evaluation] = values
    for cell_index, cell in enumerate(CELL_ORDER):
        _require(
            np.array_equal(replay[cell], stored["observable_probabilities"][cell_index]),
            f"checkpoint replay differs: {cell}",
        )
        if cell != "b0":
            _require(
                np.array_equal(replay[cell][~availability], replay["b0"][~availability]),
                "fallback changed",
            )
            branch_index = RESTRICTED_CELLS.index(cell)
            _require(
                np.array_equal(
                    replay[cell][availability],
                    stored["available_branch_probabilities"][branch_index],
                ),
                f"stored branch differs: {cell}",
            )
        _require(
            np.array_equal(replay[cell].argmax(axis=1), stored["explicit_decisions"][cell_index]),
            f"explicit decisions differ: {cell}",
        )
    assert_b0_replay(replay["b0"], cast(FloatArray, references["b0_reference_probabilities"]))
    _require(
        np.array_equal(
            stored["f3_reference_probabilities"], references["f3_reference_probabilities"]
        ),
        "F3 reference probabilities changed",
    )
    scored_indices = np.asarray(cache["scoring_indices"], dtype=np.int64)
    _require(
        np.array_equal(
            stored["scored_probabilities"],
            np.stack([replay[cell][scored_indices] for cell in CELL_ORDER]),
        ),
        "stored scored probabilities changed",
    )
    scored_people = np.asarray(cache["scored_participant_ids"], dtype=np.str_)
    scored_folds = np.asarray(cache["scored_fold_index"], dtype=np.int64)
    f3 = np.asarray(stored["f3_reference_probabilities"], dtype=np.float64)
    probabilities = {cell: replay[cell][scored_indices] for cell in CELL_ORDER}
    probabilities["f3"] = f3[scored_indices]
    roster = cast(list[str], _mapping(config["source"], "source")["participant_roster"])
    reports = {
        cell: method_report(
            labels=labels,
            probabilities=value,
            participant_ids=scored_people,
            roster=roster,
        )
        for cell, value in probabilities.items()
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
        b0_replay_exact=True,
    )
    recorded = _read_json(run_directory / "analysis.json")
    _verify_sealed(recorded, "analysis")
    for key in (
        "status",
        "evidence_status",
        "reports",
        "comparisons",
        "leave_one_outer_fold_sensitivity",
        "gates",
        "advancement",
        "event_topology",
        "decision",
        "fallback_system_requires_back_sensor",
        "automatic_follow_on_launched",
        "bootstrap_scope",
    ):
        _require(
            canonical_json_sha256(recorded[key]) == canonical_json_sha256(reconstructed[key]),
            f"analysis replay differs: {key}",
        )
    scored_available = availability[scored_indices]
    strata = {
        "available": {
            cell: method_report(
                labels=labels[scored_available],
                probabilities=value[scored_available],
                participant_ids=scored_people[scored_available],
                roster=roster,
            )
            for cell, value in probabilities.items()
        },
        "fallback": {
            cell: method_report(
                labels=labels[~scored_available],
                probabilities=value[~scored_available],
                participant_ids=scored_people[~scored_available],
                roster=roster,
            )
            for cell, value in probabilities.items()
        },
        "support": {
            "available_rows": int(scored_available.sum()),
            "fallback_rows": int((~scored_available).sum()),
            "available_class_support": np.bincount(labels[scored_available], minlength=3).tolist(),
            "fallback_class_support": np.bincount(labels[~scored_available], minlength=3).tolist(),
        },
    }
    _require(
        canonical_json_sha256(recorded["strata"]) == canonical_json_sha256(strata),
        "stratum report replay differs",
    )
    participant_record = _read_json(run_directory / "participant_metrics.json")
    _verify_sealed(participant_record, "participant metrics")
    _require(
        canonical_json_sha256(participant_record["methods"]) == canonical_json_sha256(reports)
        and canonical_json_sha256(participant_record["strata"]) == canonical_json_sha256(strata),
        "participant metrics replay differs",
    )
    partition_record = _read_json(run_directory / "partition_preflight.json")
    _verify_sealed(partition_record, "partition preflight")
    partition_body = dict(partition_record)
    partition_body.pop("record_sha256")
    expected_partitions = partition_preflight(
        labels=labels,
        observable_participants=people,
        observable_folds=folds,
        scoring_eligibility=eligible,
        availability=availability,
    )
    _require(
        canonical_json_sha256(partition_body) == canonical_json_sha256(expected_partitions),
        "partition preflight replay differs",
    )
    _require(
        preflight["partition_record_sha256"] == partition_record["record_sha256"],
        "preflight/partition binding changed",
    )
    _require(
        (run_directory / "OUTCOME_SUMMARY.md").read_text(encoding="utf-8")
        == _outcome_summary(recorded, availability),
        "outcome summary replay differs",
    )
    result = _read_json(run_directory / "result.json")
    runtime = _read_json(run_directory / "runtime.json")
    _verify_sealed(result, "result")
    _verify_sealed(runtime, "runtime")
    _require(
        result["status"] == "complete_awaiting_independent_replay"
        and result["fit_attempt_count"] == result["completed_fit_count"] == 25,
        "result fit status changed",
    )
    _require(
        canonical_json_sha256(result["method_summary"])
        == canonical_json_sha256(
            {
                cell: {
                    "mean_participant_macro_f1": reports[cell]["primary"][
                        "mean_participant_macro_f1"
                    ],
                    "pooled_accuracy": reports[cell]["pooled"]["accuracy"],
                    "bottom_30_percent_participant_macro_f1": reports[cell]["primary"][
                        "bottom_30_percent_participant_macro_f1"
                    ],
                    "worst_participant_macro_f1": reports[cell]["primary"][
                        "worst_participant_macro_f1"
                    ],
                }
                for cell in (*CELL_ORDER, "f3")
            }
        )
        and canonical_json_sha256(result["advancement"])
        == canonical_json_sha256(recorded["advancement"])
        and result["decision"] == recorded["decision"],
        "result summary replay differs",
    )
    _require(
        runtime["fit_attempt_count"] == runtime["completed_fit_count"] == 25
        and runtime["compute_cap_seconds"] == 3600
        and runtime["cap_exceeded"] is False,
        "runtime contract changed",
    )
    worker = _read_json(run_directory / "worker_shutdown.json")
    _verify_sealed(worker, "worker shutdown")
    _require(
        worker["task_owned_fit_workers_and_monitors_stopped"] is True
        and worker["terminal_status"] == "fit_phase_complete",
        "workers not stopped",
    )
    validation_worker = _stop_task_owned_workers(
        validation_worker_baseline,
        attempts=0,
        completed=0,
        terminal="validation_replay_complete",
    )
    _require(
        validation_worker["task_owned_fit_workers_and_monitors_stopped"] is True,
        "validation worker remains",
    )
    validation_seconds = time.perf_counter() - validation_started
    _require(
        float(runtime["total_run_seconds"]) + validation_seconds <= 3600.0,
        "combined run and validation compute cap exceeded",
    )
    _write_json_create_only(run_directory / "validation_worker_shutdown.json", validation_worker)
    validation = _sealed(
        {
            "record_kind": "fog_spatial_validation",
            "status": "validated",
            "validated_at_utc": _now(),
            "artifact_hashes_verified": artifact_count,
            "checkpoint_predictions_replayed": 25,
            "model_fits_during_validation": 0,
            "all_five_cells_replayed": True,
            "b0_reference_replay_exact": True,
            "fallback_probabilities_byte_exact": True,
            "participant_reports_recomputed": 6,
            "comparisons_recomputed": len(CONTRAST_PAIRS),
            "advancement_gates_recomputed": len(LV_GATES) + len(BLV_GATES),
            "full_22_person_roster_retained": True,
            "full_1213_scored_rows_retained": True,
            "InclusiveHAR_P11_P20_loaded": False,
            "source_downloaded_during_validation": False,
            "automatic_follow_on_launched": False,
            "validation_task_owned_workers_and_monitors_stopped": True,
            "validation_seconds": validation_seconds,
            "combined_run_and_validation_seconds": float(runtime["total_run_seconds"])
            + validation_seconds,
        }
    )
    _write_json_create_only(run_directory / "validation.json", validation)
    files = [
        path
        for path in run_directory.rglob("*")
        if path.is_file() and path.name != "completion_manifest.json"
    ]
    completion = _sealed(
        {
            "record_kind": "fog_spatial_completion_manifest",
            "status": "complete_and_independently_replayed",
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
        }
    )
    _write_json_create_only(run_directory / "completion_manifest.json", completion)
    return validation


def validate_run(run_directory: Path) -> dict[str, Any]:
    """Validate without fitting and retain an explicit terminal blocker on failure."""

    started = time.perf_counter()
    baseline = _worker_baseline()
    resolved = run_directory.resolve()
    try:
        return _validate_run_impl(resolved, baseline)
    except BaseException as exc:
        cleanup = _stop_task_owned_workers(
            baseline, attempts=0, completed=0, terminal="validation_incomplete_cleanup"
        )
        if (
            resolved.is_dir()
            and not (resolved / "completion_manifest.json").exists()
            and not (resolved / "VALIDATION_INCOMPLETE.json").exists()
        ):
            failure = _sealed(
                {
                    "record_kind": "fog_spatial_validation_incomplete",
                    "status": "incomplete_blocker",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "validation_seconds_before_failure": time.perf_counter() - started,
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--repository-root", type=Path, required=True)
    run.add_argument("--evidence-root", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--output-directory", type=Path, required=True)
    run.add_argument("--code-commit", required=True)
    run.add_argument("--timeout-seconds", type=int, default=3600)
    run.add_argument("--cancel-file", type=Path)
    validate = commands.add_parser("validate")
    validate.add_argument("--run-directory", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "run":
        result = run_experiment(
            repository_root=arguments.repository_root,
            evidence_root=arguments.evidence_root,
            config_path=arguments.config,
            protocol_path=arguments.protocol,
            output_directory=arguments.output_directory,
            code_commit=arguments.code_commit,
            timeout_seconds=arguments.timeout_seconds,
            cancel_file=arguments.cancel_file,
        )
    else:
        result = validate_run(arguments.run_directory)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
