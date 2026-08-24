"""Verified reporting for the corrected UCI-HAR source-only reproduction.

The official UCI test split is deliberately outside this module's interface.  Reports are
constructed only from the five participant-grouped out-of-fold partitions of the released
training split.  Every metric is recomputed from immutable prediction arrays before a run is
accepted into the aggregate.
"""

from __future__ import annotations

import csv
import io
import math
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.data.uci_har import UCI_HAR_CHANNELS, UCI_HAR_WINDOW_LENGTH
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.evaluation.statistics import (
    holm_adjust,
    paired_participant_comparison,
    participant_bootstrap_interval,
)
from inclusive_shift_har.experiments.uci_source import (
    UCI_CLASS_NAMES,
    UCI_CORRECTED_MODEL_IDS,
)
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)

DEFAULT_UCI_REPRODUCTION_SEEDS: tuple[int, ...] = (42, 1337, 2025, 31415, 271828)


class UCIReportingError(ValueError):
    """Raised when source-only reproduction evidence is incomplete or inconsistent."""


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise UCIReportingError(f"{name} must be an object")
    return cast(Mapping[str, Any], value)


def _sequence(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise UCIReportingError(f"{name} must be an array")
    return value


def _string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise UCIReportingError(f"{name} must be a non-empty string")
    return value


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UCIReportingError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise UCIReportingError(f"{name} must be finite")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise UCIReportingError(f"{name} must be a positive integer")
    return int(value)


def _sha256(value: Any, *, name: str) -> str:
    result = _string(value, name=name).casefold()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise UCIReportingError(f"{name} must be a full SHA-256")
    return result


def _git_commit(value: Any, *, name: str) -> str:
    result = _string(value, name=name).casefold()
    if len(result) != 40 or any(character not in "0123456789abcdef" for character in result):
        raise UCIReportingError(f"{name} must be a full Git object ID")
    return result


def _self_hash(record: Mapping[str, Any], *, path: Path, field: str) -> str:
    claimed = _string(record.get(field), name=f"{path}.{field}")
    unhashed = dict(record)
    unhashed.pop(field, None)
    observed = canonical_json_sha256(unhashed)
    if claimed != observed:
        raise UCIReportingError(f"{field} mismatch for {path}")
    return observed


def _protocol_contract(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise UCIReportingError("UCI protocol must be a regular, non-symlink file")
    protocol = _mapping(load_json_strict(path), name="UCI protocol")
    protocol_hash = _self_hash(protocol, path=path, field="protocol_sha256")
    if protocol.get("dataset_id") != "uci_har_v1":
        raise UCIReportingError("protocol is not for UCI-HAR v1")
    input_record = _mapping(protocol.get("input"), name="protocol.input")
    if input_record.get("released_split") != "train":
        raise UCIReportingError("only the official training split may be aggregated")
    leakage = _mapping(protocol.get("leakage_controls"), name="protocol.leakage_controls")
    if leakage.get("official_test_status") != "legacy_exploratory_development_consumed":
        raise UCIReportingError("protocol does not preserve the legacy UCI test status")
    policy = _mapping(
        protocol.get("source_pretraining_policy"), name="protocol.source_pretraining_policy"
    )
    if policy.get("official_test_allowed_for_new_claims") is not False:
        raise UCIReportingError("protocol does not forbid official-test claims")
    assignment = _mapping(protocol.get("fold_assignment"), name="protocol.fold_assignment")
    folds: dict[str, dict[str, Any]] = {}
    all_validation_subjects: list[str] = []
    for index, raw in enumerate(_sequence(assignment.get("folds"), name="folds")):
        fold = _mapping(raw, name=f"folds[{index}]")
        fold_id = _string(fold.get("fold_id"), name=f"folds[{index}].fold_id")
        if fold_id in folds:
            raise UCIReportingError(f"duplicate protocol fold {fold_id}")
        validation_subjects = [
            str(value)
            for value in _sequence(
                fold.get("validation_subject_ids"), name=f"{fold_id}.validation_subject_ids"
            )
        ]
        train_subjects = [
            str(value)
            for value in _sequence(
                fold.get("train_subject_ids"), name=f"{fold_id}.train_subject_ids"
            )
        ]
        normalization_subjects = [
            str(value)
            for value in _sequence(
                fold.get("normalization_fit_subjects"),
                name=f"{fold_id}.normalization_fit_subjects",
            )
        ]
        if (
            not train_subjects
            or not validation_subjects
            or len(set(train_subjects)) != len(train_subjects)
            or len(set(validation_subjects)) != len(validation_subjects)
        ):
            raise UCIReportingError(f"invalid or duplicate participants in protocol fold {fold_id}")
        if set(train_subjects) & set(validation_subjects):
            raise UCIReportingError(f"participant overlap in protocol fold {fold_id}")
        if normalization_subjects != train_subjects:
            raise UCIReportingError(f"normalization participants differ in protocol fold {fold_id}")
        folds[fold_id] = {
            "train_subject_ids": train_subjects,
            "validation_subject_ids": validation_subjects,
            "train_window_count": _positive_integer(
                fold.get("train_window_count"), name=f"{fold_id}.train_window_count"
            ),
            "validation_window_count": _positive_integer(
                fold.get("validation_window_count"), name=f"{fold_id}.validation_window_count"
            ),
            "train_window_ids_sha256": _sha256(
                fold.get("train_window_ids_sha256"), name=f"{fold_id}.train_window_ids_sha256"
            ),
            "validation_window_ids_sha256": _sha256(
                fold.get("validation_window_ids_sha256"),
                name=f"{fold_id}.validation_window_ids_sha256",
            ),
        }
        all_validation_subjects.extend(validation_subjects)
    counts = Counter(all_validation_subjects)
    if not folds or any(value != 1 for value in counts.values()):
        raise UCIReportingError("protocol validation subjects must occur exactly once")
    return {
        "protocol": dict(protocol),
        "protocol_sha256": protocol_hash,
        "protocol_file_sha256": sha256_file(path),
        "dataset_manifest_sha256": _sha256(
            protocol.get("dataset_manifest_sha256"),
            name="protocol.dataset_manifest_sha256",
        ),
        "processed_archive_sha256": _sha256(
            input_record.get("archive_sha256"), name="protocol.input.archive_sha256"
        ),
        "folds": folds,
        "participants": sorted(counts, key=int),
        "window_count": _positive_integer(
            input_record.get("window_count"), name="protocol.input.window_count"
        ),
        "window_ids_sha256": _sha256(
            input_record.get("window_ids_sha256"), name="protocol.input.window_ids_sha256"
        ),
    }


def _window_hash(values: Sequence[str]) -> str:
    return canonical_json_sha256(sorted(values))


def _artifact_path(raw: str, *, allowed_root: Path) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        raise UCIReportingError("prediction artifact path must be output-root-relative")
    root = allowed_root.resolve(strict=True)
    candidate = root / candidate
    if candidate.is_symlink():
        raise UCIReportingError(f"prediction artifact must not be a symlink: {candidate}")
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise UCIReportingError(f"prediction artifact escapes the run root: {candidate}") from exc
    if not resolved.exists() or not resolved.is_file():
        raise UCIReportingError(f"prediction artifact is not a regular file: {candidate}")
    return resolved


def _reports_close(observed: Any, expected: Any, *, location: str = "report") -> None:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(observed) != set(expected):
            raise UCIReportingError(f"{location} object keys differ after reconstruction")
        for key in expected:
            _reports_close(observed[key], expected[key], location=f"{location}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise UCIReportingError(f"{location} array shape differs after reconstruction")
        for index, value in enumerate(expected):
            _reports_close(observed[index], value, location=f"{location}[{index}]")
        return
    if isinstance(expected, float):
        if not isinstance(observed, (int, float)) or not math.isclose(
            float(observed), expected, rel_tol=1e-10, abs_tol=1e-12
        ):
            raise UCIReportingError(f"{location} differs after reconstruction")
        return
    if observed != expected:
        raise UCIReportingError(f"{location} differs after reconstruction")


def _load_run(
    path: Path,
    *,
    contract: Mapping[str, Any],
    expected_models: set[str],
    expected_seeds: set[int],
    allowed_artifact_root: Path,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise UCIReportingError(f"run summary must be a regular, non-symlink file: {path}")
    record = _mapping(load_json_strict(path), name=str(path))
    record_hash = _self_hash(record, path=path, field="record_sha256")
    if record.get("status") != "corrected_uci_source_fold_complete_official_test_unopened":
        raise UCIReportingError(f"incomplete or failed record supplied: {path}")
    required_false = (
        "official_test_member_opened",
        "official_test_performance_or_prediction_accessed",
        "inclusivehar_data_or_target_accessed",
    )
    if any(record.get(field) is not False for field in required_false):
        raise UCIReportingError(f"forbidden test/target access is not explicitly false in {path}")
    if record.get("released_split_loaded") != "train":
        raise UCIReportingError(f"non-training UCI split recorded by {path}")
    if record.get("dataset_id") != "uci_har_v1":
        raise UCIReportingError(f"dataset identity mismatch for {path}")
    if record.get("source_protocol_sha256") != contract["protocol_sha256"]:
        raise UCIReportingError(f"protocol lineage mismatch for {path}")
    model = _string(record.get("model_name"), name=f"{path}.model_name")
    if model not in expected_models:
        raise UCIReportingError(f"unexpected model {model!r} in {path}")
    seed_raw = record.get("seed")
    if (
        isinstance(seed_raw, bool)
        or not isinstance(seed_raw, int)
        or seed_raw not in expected_seeds
    ):
        raise UCIReportingError(f"unexpected seed in {path}")
    fold = _mapping(record.get("fold"), name=f"{path}.fold")
    fold_id = _string(fold.get("fold_id"), name=f"{path}.fold.fold_id")
    folds = cast(Mapping[str, Mapping[str, Any]], contract["folds"])
    if fold_id not in folds:
        raise UCIReportingError(f"unknown fold {fold_id!r} in {path}")
    expected_fold = folds[fold_id]
    for key in (
        "train_subject_ids",
        "validation_subject_ids",
        "train_window_count",
        "validation_window_count",
        "train_window_ids_sha256",
        "validation_window_ids_sha256",
    ):
        observed = fold.get(key)
        expected = expected_fold[key]
        if key.endswith("subject_ids"):
            observed = [str(value) for value in cast(list[Any], observed)]
        if observed != expected:
            raise UCIReportingError(f"fold contract mismatch at {path}:{key}")
    device = _mapping(record.get("device"), name=f"{path}.device")
    if device.get("requested") != "cuda" or device.get("actual") != "cuda":
        raise UCIReportingError(f"non-CUDA neural run supplied: {path}")
    if _string(record.get("evidence_status"), name=f"{path}.evidence_status") != (
        "source_grouped_development_not_confirmatory"
    ):
        raise UCIReportingError(f"incorrect evidence status in {path}")
    class_names = _sequence(record.get("class_names"), name=f"{path}.class_names")
    if class_names != list(UCI_CLASS_NAMES):
        raise UCIReportingError(f"class ordering mismatch in {path}")

    code_commit = _git_commit(record.get("code_commit"), name=f"{path}.code_commit")
    dataset_manifest_sha256 = _sha256(
        record.get("dataset_manifest_sha256"), name=f"{path}.dataset_manifest_sha256"
    )
    processed_archive_sha256 = _sha256(
        record.get("processed_archive_sha256"), name=f"{path}.processed_archive_sha256"
    )
    if dataset_manifest_sha256 != contract["dataset_manifest_sha256"]:
        raise UCIReportingError(f"dataset manifest lineage mismatch for {path}")
    if processed_archive_sha256 != contract["processed_archive_sha256"]:
        raise UCIReportingError(f"processed archive lineage mismatch for {path}")
    configuration = _mapping(record.get("configuration"), name=f"{path}.configuration")
    configuration_sha256 = _sha256(
        record.get("configuration_sha256"), name=f"{path}.configuration_sha256"
    )
    if canonical_json_sha256(configuration) != configuration_sha256:
        raise UCIReportingError(f"configuration hash mismatch for {path}")
    if (
        configuration.get("model_name") != model
        or configuration.get("seed") != seed_raw
        or configuration.get("num_classes") != len(UCI_CLASS_NAMES)
    ):
        raise UCIReportingError(f"configuration identity mismatch for {path}")
    normalization = _mapping(record.get("normalization"), name=f"{path}.normalization")
    if (
        normalization.get("method") != "per_channel_population_standardization"
        or normalization.get("fit_scope") != "training_partition_only"
        or normalization.get("training_participants") != expected_fold["train_subject_ids"]
        or normalization.get("split_manifest_sha256") != contract["protocol_sha256"]
        or normalization.get("channel_names") != list(UCI_HAR_CHANNELS)
        or normalization.get("fitted_value_count_per_channel")
        != expected_fold["train_window_count"] * UCI_HAR_WINDOW_LENGTH
    ):
        raise UCIReportingError(f"normalization lineage mismatch for {path}")
    means = np.asarray(normalization.get("mean"), dtype=np.float64)
    scales = np.asarray(normalization.get("scale"), dtype=np.float64)
    if (
        means.shape != (len(UCI_HAR_CHANNELS),)
        or scales.shape != means.shape
        or not np.isfinite(means).all()
        or not np.isfinite(scales).all()
        or np.any(scales <= 0)
    ):
        raise UCIReportingError(f"invalid normalization moments for {path}")

    artifact = _mapping(record.get("prediction_artifact"), name=f"{path}.prediction_artifact")
    if artifact.get("path_base") != "record_directory_parent":
        raise UCIReportingError(f"prediction artifact path base is not portable for {path}")
    prediction_path = _artifact_path(
        _string(artifact.get("path"), name=f"{path}.prediction_artifact.path"),
        allowed_root=allowed_artifact_root,
    )
    if sha256_file(prediction_path) != artifact.get("sha256"):
        raise UCIReportingError(f"prediction artifact hash mismatch for {path}")
    with np.load(prediction_path, allow_pickle=False) as arrays:
        required = {"logits", "probabilities", "labels", "participant_ids", "window_ids"}
        if set(arrays.files) != required:
            raise UCIReportingError(f"prediction array keys differ for {path}")
        if (
            not np.issubdtype(arrays["logits"].dtype, np.floating)
            or not np.issubdtype(arrays["probabilities"].dtype, np.floating)
            or not np.issubdtype(arrays["labels"].dtype, np.integer)
            or not np.issubdtype(arrays["participant_ids"].dtype, np.str_)
            or not np.issubdtype(arrays["window_ids"].dtype, np.str_)
        ):
            raise UCIReportingError(f"prediction array dtypes differ for {path}")
        probabilities = np.asarray(arrays["probabilities"], dtype=np.float64)
        labels = np.asarray(arrays["labels"], dtype=np.int64)
        participants = np.asarray(arrays["participant_ids"], dtype=np.str_)
        window_ids = np.asarray(arrays["window_ids"], dtype=np.str_)
        logits = np.asarray(arrays["logits"], dtype=np.float64)
    if (
        logits.ndim != 2
        or probabilities.ndim != 2
        or labels.ndim != 1
        or participants.ndim != 1
        or window_ids.ndim != 1
        or logits.shape != probabilities.shape
        or probabilities.shape != (labels.size, len(UCI_CLASS_NAMES))
        or participants.shape != labels.shape
        or window_ids.shape != labels.shape
    ):
        raise UCIReportingError(f"prediction array alignment failure for {path}")
    if not np.isfinite(logits).all():
        raise UCIReportingError(f"logits must be finite for {path}")
    shifted = logits - logits.max(axis=1, keepdims=True)
    reconstructed_probabilities = np.exp(shifted)
    reconstructed_probabilities /= reconstructed_probabilities.sum(axis=1, keepdims=True)
    if not np.allclose(reconstructed_probabilities, probabilities, rtol=1e-6, atol=1e-7):
        raise UCIReportingError(f"logit/probability alignment failure for {path}")
    if len(set(window_ids.tolist())) != window_ids.size or np.any(np.char.str_len(window_ids) == 0):
        raise UCIReportingError(f"prediction window identities are invalid for {path}")
    if probabilities.shape[0] != fold.get("validation_window_count"):
        raise UCIReportingError(f"prediction count mismatch for {path}")
    if _window_hash(window_ids.tolist()) != fold.get("validation_window_ids_sha256"):
        raise UCIReportingError(f"prediction window hash mismatch for {path}")
    if set(participants.tolist()) != set(expected_fold["validation_subject_ids"]):
        raise UCIReportingError(f"prediction participant mismatch for {path}")
    reconstructed = classification_report(
        labels,
        probabilities,
        participants.tolist(),
        class_names=UCI_CLASS_NAMES,
    )
    reported = _mapping(record.get("validation_report"), name=f"{path}.validation_report")
    _reports_close(reconstructed, reported)
    return {
        "path": path.relative_to(allowed_artifact_root).as_posix(),
        "record_sha256": record_hash,
        "model": model,
        "seed": seed_raw,
        "fold_id": fold_id,
        "code_commit": code_commit,
        "summary_file_sha256": sha256_file(path),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "processed_archive_sha256": processed_archive_sha256,
        "configuration_sha256": configuration_sha256,
        "prediction_path": prediction_path.relative_to(allowed_artifact_root).as_posix(),
        "prediction_sha256": artifact.get("sha256"),
        "probabilities": probabilities,
        "labels": labels,
        "participants": participants,
        "window_ids": window_ids,
        "parameter_count": _positive_integer(
            _mapping(record.get("training"), name=f"{path}.training").get("parameter_count"),
            name=f"{path}.training.parameter_count",
        ),
        "elapsed_seconds": _finite(record.get("elapsed_seconds"), name=f"{path}.elapsed_seconds"),
        "peak_vram_bytes": _positive_integer(
            device.get("peak_vram_bytes"), name=f"{path}.device.peak_vram_bytes"
        ),
    }


def _mean_participant_values(seed_reports: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for report in seed_reports:
        for row in cast(list[dict[str, Any]], report["participants"]):
            values[str(row["participant_id"])].append(float(row["macro_f1"]))
    expected_seed_count = len(seed_reports)
    if any(len(items) != expected_seed_count for items in values.values()):
        raise UCIReportingError("participant coverage differs across seeds")
    return {participant: float(np.mean(items)) for participant, items in sorted(values.items())}


def build_uci_reproduction_report(
    *,
    record_directory: str | os.PathLike[str],
    protocol_path: str | os.PathLike[str],
    models: Sequence[str] = UCI_CORRECTED_MODEL_IDS,
    seeds: Sequence[int] = DEFAULT_UCI_REPRODUCTION_SEEDS,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 1729,
) -> dict[str, Any]:
    """Validate a complete CUDA matrix and aggregate official-train OOF predictions."""

    if tuple(models) != UCI_CORRECTED_MODEL_IDS:
        raise UCIReportingError(
            "complete reporting requires all corrected legacy models in predeclared order"
        )
    if tuple(seeds) != DEFAULT_UCI_REPRODUCTION_SEEDS:
        raise UCIReportingError("complete reporting requires the five predeclared seeds in order")
    if bootstrap_resamples < 100:
        raise UCIReportingError("bootstrap_resamples must be at least 100")
    root = Path(record_directory)
    if root.is_symlink() or not root.is_dir():
        raise UCIReportingError("record directory must be a non-symlink directory")
    root = root.resolve(strict=True)
    artifact_root = root.parent
    protocol = _protocol_contract(Path(protocol_path))
    paths = sorted(root.glob("*.json"))
    expected_count = len(models) * len(seeds) * len(cast(Mapping[str, Any], protocol["folds"]))
    if len(paths) != expected_count:
        raise UCIReportingError(f"expected exactly {expected_count} summaries, found {len(paths)}")
    runs = [
        _load_run(
            path,
            contract=protocol,
            expected_models=set(models),
            expected_seeds=set(seeds),
            allowed_artifact_root=artifact_root,
        )
        for path in paths
    ]
    keys = [(run["model"], run["seed"], run["fold_id"]) for run in runs]
    duplicates = [key for key, count in Counter(keys).items() if count != 1]
    expected_keys = {
        (model, seed, fold_id)
        for model in models
        for seed in seeds
        for fold_id in cast(Mapping[str, Any], protocol["folds"])
    }
    if duplicates or set(keys) != expected_keys:
        raise UCIReportingError("run matrix has duplicate, missing, or unexpected cells")
    commits = {run["code_commit"] for run in runs}
    if len(commits) != 1:
        raise UCIReportingError("all UCI runs must share one code commit")
    dataset_manifests = {run["dataset_manifest_sha256"] for run in runs}
    processed_archives = {run["processed_archive_sha256"] for run in runs}
    if dataset_manifests != {protocol["dataset_manifest_sha256"]}:
        raise UCIReportingError("dataset manifest lineage differs from the UCI protocol")
    if processed_archives != {protocol["processed_archive_sha256"]}:
        raise UCIReportingError("processed archive lineage differs from the UCI protocol")
    configurations: dict[tuple[str, int], set[str]] = defaultdict(set)
    identity_by_window: dict[str, tuple[str, int]] = {}
    for run in runs:
        configurations[(str(run["model"]), int(run["seed"]))].add(str(run["configuration_sha256"]))
        for window_id, participant, label in zip(
            cast(NDArray[np.str_], run["window_ids"]).tolist(),
            cast(NDArray[np.str_], run["participants"]).tolist(),
            cast(NDArray[np.int64], run["labels"]).tolist(),
            strict=True,
        ):
            identity = (str(participant), int(label))
            previous = identity_by_window.setdefault(str(window_id), identity)
            if previous != identity:
                raise UCIReportingError(
                    f"participant/label identity differs across runs for window {window_id}"
                )
    if any(len(values) != 1 for values in configurations.values()):
        raise UCIReportingError("training configuration differs across folds of a model/seed")

    seed_reports: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for model in models:
        for seed in seeds:
            selected = sorted(
                (run for run in runs if run["model"] == model and run["seed"] == seed),
                key=lambda item: str(item["fold_id"]),
            )
            window_ids = np.concatenate(
                [cast(NDArray[np.str_], run["window_ids"]) for run in selected]
            )
            if len(set(window_ids.tolist())) != window_ids.size:
                raise UCIReportingError(f"duplicate OOF windows for {model}, seed {seed}")
            if window_ids.size != protocol["window_count"] or (
                _window_hash(window_ids.tolist()) != protocol["window_ids_sha256"]
            ):
                raise UCIReportingError(f"incomplete OOF window coverage for {model}, seed {seed}")
            labels = np.concatenate([cast(NDArray[np.int64], run["labels"]) for run in selected])
            probabilities = np.concatenate(
                [cast(NDArray[np.float64], run["probabilities"]) for run in selected]
            )
            participants = np.concatenate(
                [cast(NDArray[np.str_], run["participants"]) for run in selected]
            )
            report = classification_report(
                labels,
                probabilities,
                participants.tolist(),
                class_names=UCI_CLASS_NAMES,
            )
            report["model_name"] = model
            report["seed"] = seed
            seed_reports[model].append(report)

    model_rows: list[dict[str, Any]] = []
    participant_maps: dict[str, dict[str, float]] = {}
    for model in models:
        reports = seed_reports[model]
        participant_values = _mean_participant_values(reports)
        participant_maps[model] = participant_values
        values = np.asarray(list(participant_values.values()), dtype=np.float64)
        primary_seed_means = [
            float(report["primary"]["mean_participant_macro_f1"]) for report in reports
        ]
        calibration = [cast(Mapping[str, Any], report["calibration"]) for report in reports]
        diagnostics = [
            cast(Mapping[str, Any], report["window_level_diagnostics"]) for report in reports
        ]
        run_subset = [run for run in runs if run["model"] == model]
        parameter_counts = {run["parameter_count"] for run in run_subset}
        if len(parameter_counts) != 1:
            raise UCIReportingError(f"parameter count differs across runs for {model}")
        model_rows.append(
            {
                "model_name": model,
                "participant_count": len(participant_values),
                "seed_count": len(reports),
                "mean_participant_macro_f1": float(values.mean()),
                "worst_participant_macro_f1": float(values.min()),
                "lower_decile_participant_macro_f1": float(
                    np.quantile(values, 0.1, method="linear")
                ),
                "participant_bootstrap_mean_ci": participant_bootstrap_interval(
                    participant_values,
                    resamples=bootstrap_resamples,
                    confidence=0.95,
                    seed=bootstrap_seed,
                ),
                "seed_mean_participant_macro_f1": primary_seed_means,
                "seed_mean_standard_deviation": float(np.std(primary_seed_means, ddof=1)),
                "mean_window_balanced_accuracy": float(
                    np.mean([float(item["balanced_accuracy"]) for item in diagnostics])
                ),
                "mean_window_accuracy": float(
                    np.mean([float(item["accuracy"]) for item in diagnostics])
                ),
                "mean_window_macro_f1": float(
                    np.mean([float(item["macro_f1"]) for item in diagnostics])
                ),
                "mean_per_class_recall": {
                    class_name: float(
                        np.mean(
                            [
                                float(cast(Mapping[str, Any], item["per_class_recall"])[class_name])
                                for item in diagnostics
                            ]
                        )
                    )
                    for class_name in UCI_CLASS_NAMES
                },
                "mean_nll": float(
                    np.mean([float(item["negative_log_likelihood"]) for item in calibration])
                ),
                "mean_brier": float(
                    np.mean([float(item["multiclass_brier_score"]) for item in calibration])
                ),
                "mean_ece": float(np.mean([float(item["ece"]) for item in calibration])),
                "mean_aurc": float(
                    np.mean(
                        [
                            float(cast(Mapping[str, Any], report["selective_risk"])["aurc"])
                            for report in reports
                        ]
                    )
                ),
                "parameter_count": parameter_counts.pop(),
                "mean_fold_elapsed_seconds": float(
                    np.mean([float(run["elapsed_seconds"]) for run in run_subset])
                ),
                "maximum_peak_vram_bytes": max(int(run["peak_vram_bytes"]) for run in run_subset),
                "participant_macro_f1": participant_values,
                "reconstructed_seed_reports": reports,
            }
        )
    model_rows.sort(key=lambda row: float(row["mean_participant_macro_f1"]), reverse=True)
    comparisons: list[dict[str, Any]] = []
    for reference, candidate in combinations(models, 2):
        comparison = paired_participant_comparison(
            participant_maps[reference], participant_maps[candidate]
        )
        comparison["reference_model"] = reference
        comparison["candidate_model"] = candidate
        comparisons.append(comparison)
    if comparisons:
        permutation_adjusted = holm_adjust(
            [float(item["permutation"]["two_sided_p_value"]) for item in comparisons]
        )
        wilcoxon_adjusted = holm_adjust(
            [float(item["wilcoxon"]["two_sided_p_value"]) for item in comparisons]
        )
        for item, permutation_p, wilcoxon_p in zip(
            comparisons, permutation_adjusted, wilcoxon_adjusted, strict=True
        ):
            item["permutation"]["holm_adjusted_p_value"] = permutation_p
            item["wilcoxon"]["holm_adjusted_p_value"] = wilcoxon_p

    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "corrected_uci_source_grouped_reproduction_complete",
        "evidence_status": "source_grouped_development_official_test_unopened_not_confirmatory",
        "dataset_id": "uci_har_v1",
        "released_split_aggregated": "train",
        "official_test_member_opened": False,
        "official_test_performance_or_prediction_accessed": False,
        "inclusivehar_data_or_target_accessed": False,
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_file_sha256": protocol["protocol_file_sha256"],
        "code_commit": commits.pop(),
        "dataset_manifest_sha256": protocol["dataset_manifest_sha256"],
        "processed_archive_sha256": protocol["processed_archive_sha256"],
        "models": model_rows,
        "highest_observed_mean_model": model_rows[0]["model_name"],
        "all_pairwise_comparisons": comparisons,
        "inputs": [
            {
                "summary_path": run["path"],
                "record_sha256": run["record_sha256"],
                "summary_file_sha256": run["summary_file_sha256"],
                "prediction_path": run["prediction_path"],
                "prediction_sha256": run["prediction_sha256"],
                "configuration_sha256": run["configuration_sha256"],
            }
            for run in runs
        ],
        "analysis": {
            "primary_unit": "participant",
            "seed_aggregation": "average each participant across five seeds before cohort summary",
            "uncertainty": "participant-clustered percentile bootstrap",
            "multiplicity": "Holm correction within the three corrected legacy models",
            "comparison_family": (
                "all three model pairs are pre-enumerated; no observed winner is selected as "
                "the inferential reference"
            ),
            "window_metrics": "descriptive only",
            "ece": "secondary calibration diagnostic",
        },
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    return payload


def _text_write_new(text: str, destination: Path, *, allowed_root: Path) -> Path:
    root = allowed_root.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent = destination.parent.resolve(strict=True)
    try:
        parent.relative_to(root)
    except ValueError as exc:
        raise UCIReportingError("report destination escapes allowed root") from exc
    target = parent / destination.name
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite existing file: {target}")
    temporary = parent / f".{target.name}.partial.{uuid4().hex}"
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    except Exception as exc:
        raise OSError(f"report publication failed; partial retained at {temporary}: {exc}") from exc
    else:
        temporary.unlink()
    return target


def write_uci_reproduction_exports_new(
    report: Mapping[str, Any],
    *,
    output_directory: str | os.PathLike[str],
) -> dict[str, Path]:
    """Create JSON, CSV, and Markdown exports without replacing prior evidence."""

    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    json_path = root / "uci_source_grouped_report_v1.json"
    csv_path = root / "uci_source_grouped_model_summary_v1.csv"
    markdown_path = root / "uci_source_grouped_model_summary_v1.md"
    destinations = (json_path, csv_path, markdown_path)
    existing = [path for path in destinations if os.path.lexists(path)]
    if existing:
        raise FileExistsError(
            "refusing to create a partial export bundle because destinations exist: "
            + ", ".join(str(path) for path in existing)
        )
    rows = cast(list[Mapping[str, Any]], report.get("models"))
    fields = [
        "model_name",
        "mean_participant_macro_f1",
        "worst_participant_macro_f1",
        "lower_decile_participant_macro_f1",
        "seed_mean_standard_deviation",
        "mean_window_accuracy",
        "mean_window_macro_f1",
        "mean_window_balanced_accuracy",
        "mean_nll",
        "mean_brier",
        "mean_ece",
        "mean_aurc",
        "parameter_count",
        "maximum_peak_vram_bytes",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fields})
    markdown = [
        "# Corrected UCI-HAR source-grouped reproduction",
        "",
        "The official UCI test split was not opened. Values are participant-grouped",
        "out-of-fold development evidence on the released training split.",
        "",
        "| Model | Mean participant macro-F1 | Worst | Lower decile | Balanced accuracy | NLL | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            "| {model_name} | {mean_participant_macro_f1:.4f} | "
            "{worst_participant_macro_f1:.4f} | {lower_decile_participant_macro_f1:.4f} | "
            "{mean_window_balanced_accuracy:.4f} | {mean_nll:.4f} | {mean_brier:.4f} | "
            "{mean_ece:.4f} |".format(**row)
        )
    # Publish the self-hashed JSON last so its presence is the bundle-completion marker.
    _text_write_new(buffer.getvalue(), csv_path, allowed_root=root)
    _text_write_new("\n".join(markdown) + "\n", markdown_path, allowed_root=root)
    atomic_write_json_new(dict(report), json_path, allowed_root=root)
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}
