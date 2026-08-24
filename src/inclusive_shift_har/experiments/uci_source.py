"""CUDA-only, official-train-only corrected UCI-HAR reproduction runner."""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from inclusive_shift_har.data.uci_har import (
    UCI_HAR_ACTIVITY_NAMES,
    UCI_HAR_CHANNELS,
    UCIHARWindows,
    load_uci_har_split,
)
from inclusive_shift_har.evaluation._strict_config import (
    StrictConfigError,
    load_strict_yaml_mapping,
    require_exact_keys,
    require_mapping,
)
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)
from inclusive_shift_har.preprocessing.normalization import ChannelStandardizer
from inclusive_shift_har.protocols.uci_source import (
    UCISourceFold,
    build_grouped_source_folds,
    build_uci_source_protocol_manifest,
)
from inclusive_shift_har.training.engine import (
    TrainingConfig,
    TrainingLineage,
    train_source_model,
)

UCI_CORRECTED_MODEL_IDS: tuple[str, ...] = (
    "legacy_cnn1d_h128",
    "legacy_bilstm_h192",
    "legacy_joint_bilstm256_cnn128",
)
UCI_CLASS_NAMES: tuple[str, ...] = tuple(
    UCI_HAR_ACTIVITY_NAMES[index] for index in sorted(UCI_HAR_ACTIVITY_NAMES)
)
UCI_CORRECTED_FOLD_IDS: tuple[str, ...] = tuple(
    f"uci_source_cv_{index:02d}" for index in range(1, 6)
)
UCI_CORRECTED_SEEDS: tuple[int, ...] = (42, 1337, 2025, 31415, 271828)


class UCISourceRunError(RuntimeError):
    """Raised when a corrected source-only run fails closed."""


@dataclass(frozen=True)
class UCIReproductionConfig:
    """Strict, hash-pinned corrected-reproduction matrix contract."""

    path: Path
    file_sha256: str
    canonical_sha256: str
    experiment_id: str
    dataset_manifest: str
    protocol_manifest: str
    models: tuple[str, ...]
    disable_cudnn_by_model: Mapping[str, bool]
    fold_order: tuple[str, ...]
    seed_order: tuple[int, ...]
    training: Mapping[str, Any]
    outputs: Mapping[str, Any]


def _require_config_value(actual: Any, expected: Any, *, location: str) -> None:
    if actual != expected:
        raise StrictConfigError(f"{location} is {actual!r}; required {expected!r}")


def load_uci_reproduction_config(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> UCIReproductionConfig:
    """Load the exact v1.1 matrix and reject any unrecorded protocol drift."""

    config_path = Path(path).resolve(strict=True)
    observed_file_sha256 = sha256_file(config_path)
    expected_digest = expected_file_sha256.casefold()
    if (
        len(expected_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_digest)
        or observed_file_sha256 != expected_digest
    ):
        raise StrictConfigError("UCI reproduction config file SHA-256 mismatch")
    raw = load_strict_yaml_mapping(config_path)
    require_exact_keys(
        raw,
        {
            "schema_version",
            "experiment_id",
            "status",
            "evidence_status",
            "supersedes",
            "supersession_reason",
            "dataset_manifest",
            "protocol_manifest",
            "allowed_released_split",
            "official_test_member_opening_allowed",
            "input_shape",
            "label_track",
            "models",
            "fold_order",
            "seed_order",
            "training",
            "execution",
            "outputs",
        },
        location="$",
    )
    constants = {
        "schema_version": "1.0.0",
        "experiment_id": "uci_har_corrected_legacy_reproduction_v1.1",
        "status": "configured_not_run_superseding_v1_model_scope",
        "evidence_status": "corrected_source_development_no_results",
        "supersedes": "configs/experiments/uci_har_legacy_reproduction.yaml",
        "supersession_reason": (
            "v1 listed a temporal BiLSTM variant outside the implemented exact three-model "
            "reproduction matrix"
        ),
        "dataset_manifest": "manifests/datasets/uci_har_v1.json",
        "protocol_manifest": "results/protocol/uci_har_source_grouped_v1.json",
        "allowed_released_split": "train",
        "official_test_member_opening_allowed": False,
        "input_shape": [128, 6],
        "label_track": "uci_native_six",
    }
    for key, expected in constants.items():
        _require_config_value(raw.get(key), expected, location=key)

    models_value = raw["models"]
    if not isinstance(models_value, list) or len(models_value) != len(UCI_CORRECTED_MODEL_IDS):
        raise StrictConfigError("models must contain exactly the three corrected legacy models")
    model_ids: list[str] = []
    disable_cudnn: dict[str, bool] = {}
    expected_disable = (False, True, True)
    for index, (value, expected_model, expected_disabled) in enumerate(
        zip(models_value, UCI_CORRECTED_MODEL_IDS, expected_disable, strict=True)
    ):
        model = require_mapping(value, location=f"models[{index}]")
        require_exact_keys(model, {"model_id", "disable_cudnn"}, location=f"models[{index}]")
        _require_config_value(
            model.get("model_id"), expected_model, location=f"models[{index}].model_id"
        )
        _require_config_value(
            model.get("disable_cudnn"),
            expected_disabled,
            location=f"models[{index}].disable_cudnn",
        )
        model_ids.append(expected_model)
        disable_cudnn[expected_model] = expected_disabled

    _require_config_value(raw["fold_order"], list(UCI_CORRECTED_FOLD_IDS), location="fold_order")
    _require_config_value(raw["seed_order"], list(UCI_CORRECTED_SEEDS), location="seed_order")

    training = require_mapping(raw["training"], location="training")
    expected_training = {
        "device": "cuda",
        "cpu_fallback": False,
        "epochs_max": 40,
        "batch_size": 128,
        "optimizer": "adamw",
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "early_stopping_patience": 8,
        "minimum_epochs": 8,
        "checkpoint_selection_rule": "source_validation_best",
        "selection_metric": "validation_participant_macro_f1",
        "normalization": "fold_training_windows_only",
        "mixed_precision": "float16",
    }
    require_exact_keys(training, set(expected_training), location="training")
    _require_config_value(training, expected_training, location="training")

    execution = require_mapping(raw["execution"], location="execution")
    expected_execution = {
        "expected_run_count": 75,
        "one_run_per_process": True,
        "sequential_runs": True,
        "create_only": True,
        "preserve_failures": True,
        "attempt_id_format": "attempt-{positive_integer:03d}",
        "retry_policy": (
            "new_attempt_only_after_every_prior_attempt_has_a_valid_preserved_failure"
        ),
        "aggregation_policy": (
            "exactly_one_complete_attempt_per_matrix_cell_with_all_prior_failures_linked"
        ),
        "official_test_evaluation": "forbidden_consumed_legacy_evidence",
    }
    require_exact_keys(execution, set(expected_execution), location="execution")
    _require_config_value(execution, expected_execution, location="execution")

    outputs = require_mapping(raw["outputs"], location="outputs")
    expected_outputs = {
        "root": "results/legacy_reproduction/uci_har_source_grouped_v1",
        "participant_level_metrics": True,
        "no_inclusivehar_target_evaluation": True,
    }
    require_exact_keys(outputs, set(expected_outputs), location="outputs")
    _require_config_value(outputs, expected_outputs, location="outputs")
    return UCIReproductionConfig(
        path=config_path,
        file_sha256=observed_file_sha256,
        canonical_sha256=canonical_json_sha256(raw),
        experiment_id=str(raw["experiment_id"]),
        dataset_manifest=str(raw["dataset_manifest"]),
        protocol_manifest=str(raw["protocol_manifest"]),
        models=tuple(model_ids),
        disable_cudnn_by_model=disable_cudnn,
        fold_order=UCI_CORRECTED_FOLD_IDS,
        seed_order=UCI_CORRECTED_SEEDS,
        training=training,
        outputs=outputs,
    )


def build_uci_training_config(
    config: UCIReproductionConfig,
    *,
    model_name: str,
    seed: int,
) -> TrainingConfig:
    """Materialize exactly one immutable cell's shared-engine configuration."""

    if model_name not in config.models:
        raise UCISourceRunError(f"model is outside the locked matrix: {model_name}")
    if seed not in config.seed_order:
        raise UCISourceRunError(f"seed is outside the locked matrix: {seed}")
    training = config.training
    epochs = int(training["epochs_max"])
    return TrainingConfig(
        model_name=model_name,
        num_classes=len(UCI_CLASS_NAMES),
        seed=seed,
        epochs=epochs,
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        patience=int(training["early_stopping_patience"]),
        minimum_epochs=int(training["minimum_epochs"]),
        mixed_precision=str(training["mixed_precision"]),
        checkpoint_interval=epochs,
        checkpoint_selection_rule=str(training["checkpoint_selection_rule"]),
        disable_cudnn=config.disable_cudnn_by_model[model_name],
    )


@dataclass(frozen=True)
class PreparedUCISourceFold:
    """Normalized official-train fold ready for the shared training engine."""

    windows: UCIHARWindows
    fold: UCISourceFold
    train_windows: NDArray[np.float32]
    train_labels: NDArray[np.int64]
    train_participant_ids: list[str]
    validation_windows: NDArray[np.float32]
    validation_labels: NDArray[np.int64]
    validation_participant_ids: list[str]
    validation_window_ids: tuple[str, ...]
    standardizer: ChannelStandardizer
    dataset_manifest_sha256: str
    protocol_sha256: str


def _object_from_json(path: Path, *, description: str) -> dict[str, Any]:
    value = load_json_strict(path)
    if not isinstance(value, dict):
        raise UCISourceRunError(f"{description} root must be an object")
    return cast(dict[str, Any], value)


def _validated_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = _object_from_json(path, description="UCI source protocol")
    claimed = protocol.get("protocol_sha256")
    unhashed = dict(protocol)
    unhashed.pop("protocol_sha256", None)
    observed = canonical_json_sha256(unhashed)
    if claimed != observed:
        raise UCISourceRunError("UCI source protocol self-hash does not validate")
    if (
        protocol.get("dataset_id") != "uci_har_v1"
        or protocol.get("input", {}).get("released_split") != "train"
        or protocol.get("source_pretraining_policy", {}).get("official_test_allowed_for_new_claims")
        is not False
    ):
        raise UCISourceRunError(
            "protocol is not the locked official-train-only UCI source protocol"
        )
    return protocol, observed


def _processed_archive_hash(dataset_manifest: dict[str, Any]) -> str:
    try:
        value = dataset_manifest["expected_data"]["official_download_wrapper"][
            "embedded_processed_archive_sha256"
        ]
    except (KeyError, TypeError) as exc:
        raise UCISourceRunError("dataset manifest lacks the processed UCI archive hash") from exc
    digest = str(value).casefold()
    if len(digest) != 64:
        raise UCISourceRunError("processed UCI archive hash is not a full SHA-256")
    return digest


def prepare_uci_source_fold(
    *,
    archive_path: Path,
    dataset_manifest_path: Path,
    protocol_path: Path,
    fold_id: str,
) -> PreparedUCISourceFold:
    """Verify and normalize one fold without reading any official-test member."""

    dataset_manifest = _object_from_json(dataset_manifest_path, description="dataset manifest")
    if dataset_manifest.get("dataset_id") != "uci_har_v1":
        raise UCISourceRunError("dataset manifest is not UCI-HAR v1")
    dataset_manifest_sha256 = canonical_json_sha256(dataset_manifest)
    protocol, protocol_sha256 = _validated_protocol(protocol_path)
    if protocol.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise UCISourceRunError("dataset manifest hash disagrees with the locked protocol")

    expected_archive_sha256 = _processed_archive_hash(dataset_manifest)
    windows = load_uci_har_split(
        archive_path,
        split="train",
        expected_sha256=expected_archive_sha256,
    )
    assignment = protocol.get("fold_assignment")
    if not isinstance(assignment, dict):
        raise UCISourceRunError("protocol fold assignment is missing")
    recorded_folds = assignment.get("folds")
    if not isinstance(recorded_folds, list):
        raise UCISourceRunError("protocol fold records are missing")
    seed = assignment.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise UCISourceRunError("protocol fold seed is invalid")
    rebuilt = build_uci_source_protocol_manifest(
        windows,
        dataset_manifest_sha256=dataset_manifest_sha256,
        n_folds=len(recorded_folds),
        seed=seed,
    )
    if rebuilt["protocol_sha256"] != protocol_sha256:
        raise UCISourceRunError("protocol does not reproduce from the official train windows")
    folds = build_grouped_source_folds(windows, n_folds=len(recorded_folds), seed=seed)
    fold = next((candidate for candidate in folds if candidate.fold_id == fold_id), None)
    if fold is None:
        raise UCISourceRunError(f"unknown UCI source fold: {fold_id}")

    train_participants = [str(value) for value in windows.subject_ids[fold.train_indices]]
    validation_participants = [str(value) for value in windows.subject_ids[fold.validation_indices]]
    standardizer = ChannelStandardizer.fit(
        windows.signals[fold.train_indices],
        train_participants,
        declared_training_participants=set(train_participants),
        split_manifest_sha256=protocol_sha256,
        channel_names=UCI_HAR_CHANNELS,
    )
    train_labels = np.asarray(windows.activity_ids[fold.train_indices] - 1, dtype=np.int64)
    validation_labels = np.asarray(
        windows.activity_ids[fold.validation_indices] - 1, dtype=np.int64
    )
    return PreparedUCISourceFold(
        windows=windows,
        fold=fold,
        train_windows=standardizer.transform(windows.signals[fold.train_indices]),
        train_labels=train_labels,
        train_participant_ids=train_participants,
        validation_windows=standardizer.transform(windows.signals[fold.validation_indices]),
        validation_labels=validation_labels,
        validation_participant_ids=validation_participants,
        validation_window_ids=tuple(
            windows.window_ids[int(index)] for index in fold.validation_indices
        ),
        standardizer=standardizer,
        dataset_manifest_sha256=dataset_manifest_sha256,
        protocol_sha256=protocol_sha256,
    )


def _confined_output(path: Path, *, allowed_root: Path, kind: str) -> Path:
    root = allowed_root.resolve(strict=True)
    if not root.is_dir():
        raise UCISourceRunError(f"allowed output root is not a directory: {root}")
    candidate = path.resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise UCISourceRunError(f"{kind} escapes the allowed output root") from exc
    if candidate == root:
        raise UCISourceRunError(f"{kind} cannot be the allowed output root itself")
    return candidate


def _portable_output_reference(path: Path, *, allowed_root: Path, kind: str) -> str:
    root = allowed_root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise UCISourceRunError(f"{kind} escapes the allowed output root") from exc
    if relative == Path("."):
        raise UCISourceRunError(f"{kind} cannot be the allowed output root itself")
    return relative.as_posix()


def _write_predictions_new(
    path: Path,
    *,
    logits: NDArray[np.float64],
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int64],
    participant_ids: list[str],
    window_ids: tuple[str, ...],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            logits=logits,
            probabilities=probabilities,
            labels=labels,
            participant_ids=np.asarray(participant_ids),
            window_ids=np.asarray(window_ids),
        )
    return sha256_file(path)


def _code_commit(value: str) -> str:
    normalized = value.casefold()
    if len(normalized) != 40 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise UCISourceRunError("code commit must be a full 40-character Git object ID")
    return normalized


def _validate_repository_head(repository_root: Path, *, expected_commit: str) -> str:
    root = repository_root.resolve(strict=True)
    if not root.is_dir():
        raise UCISourceRunError("repository root must be a directory")
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise UCISourceRunError("cannot verify the executing Git HEAD") from exc
    observed = _code_commit(completed.stdout.strip())
    if observed != expected_commit:
        raise UCISourceRunError(
            f"supplied code commit {expected_commit} does not match executing HEAD {observed}"
        )
    return observed


def _validate_matrix_paths(
    *,
    repository_root: Path,
    experiment_config: UCIReproductionConfig,
    archive_path: Path,
    dataset_manifest_path: Path,
    protocol_path: Path,
    model_name: str,
    fold_id: str,
    seed: int,
    attempt: int,
    run_directory: Path,
    summary_path: Path,
    allowed_output_root: Path,
) -> None:
    root = repository_root.resolve(strict=True)
    expected_config_path = (
        root / "configs/experiments/uci_har_corrected_reproduction_v1_1.yaml"
    ).resolve(strict=True)
    expected_dataset_path = (root / experiment_config.dataset_manifest).resolve(strict=True)
    expected_protocol_path = (root / experiment_config.protocol_manifest).resolve(strict=True)
    expected_output_root = (root / str(experiment_config.outputs["root"])).resolve(strict=False)
    expected_allowed_root = (root / "results").resolve(strict=True)
    stem = f"{model_name}--seed-{seed}--{fold_id}"
    attempt_id = f"attempt-{attempt:03d}"
    comparisons = (
        (experiment_config.path, expected_config_path, "experiment config"),
        (dataset_manifest_path.resolve(strict=True), expected_dataset_path, "dataset manifest"),
        (protocol_path.resolve(strict=True), expected_protocol_path, "source protocol"),
        (allowed_output_root.resolve(strict=True), expected_allowed_root, "allowed output root"),
        (
            run_directory.resolve(strict=False),
            expected_output_root / "runs" / stem / attempt_id,
            "run directory",
        ),
        (
            summary_path.resolve(strict=False),
            expected_output_root / "records" / stem / f"{attempt_id}.json",
            "summary path",
        ),
    )
    for observed, expected, role in comparisons:
        if observed != expected:
            raise UCISourceRunError(f"{role} differs from the locked matrix path")
    expected_archive = (root / "data/raw/uci_har/v1/UCI HAR Dataset.zip").resolve(strict=False)
    if archive_path.resolve(strict=True) != expected_archive:
        raise UCISourceRunError("archive path differs from the locked UCI inner archive path")


def _validate_retry_chain(
    *,
    run_path: Path,
    summary_path: Path,
    result_root: Path,
    model_name: str,
    fold_id: str,
    seed: int,
    attempt: int,
    code_commit: str,
    experiment_config: UCIReproductionConfig,
) -> list[dict[str, Any]]:
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1 or attempt > 999:
        raise UCISourceRunError("attempt must be an integer in [1, 999]")
    cell_run_root = run_path.parent
    cell_record_root = summary_path.parent
    expected_prior_names = {f"attempt-{value:03d}" for value in range(1, attempt)}
    for root, suffix in ((cell_run_root, ""), (cell_record_root, ".json")):
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise UCISourceRunError("UCI attempt parent is unsafe")
        observed = {
            path.name.removesuffix(suffix)
            for path in root.iterdir()
            if path.name.startswith("attempt-")
        }
        unexpected = observed - expected_prior_names
        if unexpected:
            raise UCISourceRunError(
                f"unexpected or future UCI attempt artifacts exist: {sorted(unexpected)}"
            )

    prior_failures: list[dict[str, Any]] = []
    for prior_attempt in range(1, attempt):
        attempt_id = f"attempt-{prior_attempt:03d}"
        prior_run = cell_run_root / attempt_id
        prior_summary = cell_record_root / f"{attempt_id}.json"
        failure_path = prior_run / "failure.json"
        if os.path.lexists(prior_summary):
            raise UCISourceRunError("retry is forbidden after a completed UCI cell attempt")
        if (
            prior_run.is_symlink()
            or not prior_run.is_dir()
            or failure_path.is_symlink()
            or not failure_path.is_file()
        ):
            raise UCISourceRunError("every prior UCI attempt must have a preserved failure")
        failure = _object_from_json(failure_path, description="prior UCI attempt failure")
        claimed = failure.get("record_sha256")
        body = dict(failure)
        body.pop("record_sha256", None)
        required = {
            "status": "corrected_uci_source_fold_failed_preserved",
            "model_name": model_name,
            "fold_id": fold_id,
            "seed": seed,
            "attempt": prior_attempt,
            "code_commit": code_commit,
            "official_test_member_opened": False,
            "official_test_performance_or_prediction_accessed": False,
            "inclusivehar_data_or_target_accessed": False,
        }
        if claimed != canonical_json_sha256(body) or any(
            failure.get(key) != expected for key, expected in required.items()
        ):
            raise UCISourceRunError("prior UCI attempt failure lineage is invalid")
        lineage = failure.get("experiment_config")
        if lineage != {
            "experiment_id": experiment_config.experiment_id,
            "file_sha256": experiment_config.file_sha256,
            "canonical_sha256": experiment_config.canonical_sha256,
        }:
            raise UCISourceRunError("prior UCI attempt uses a different experiment config")
        prior_failures.append(
            {
                "attempt": prior_attempt,
                "path": failure_path.relative_to(result_root).as_posix(),
                "file_sha256": sha256_file(failure_path),
                "record_sha256": claimed,
            }
        )
    return prior_failures


def run_uci_source_fold(
    *,
    archive_path: Path,
    dataset_manifest_path: Path,
    protocol_path: Path,
    model_name: str,
    fold_id: str,
    seed: int,
    attempt: int,
    code_commit: str,
    repository_root: Path,
    experiment_config_path: Path,
    expected_experiment_config_file_sha256: str,
    run_directory: Path,
    summary_path: Path,
    allowed_output_root: Path,
) -> dict[str, Any]:
    """Run one immutable grouped source fold on CUDA; official test stays unopened."""

    if model_name not in UCI_CORRECTED_MODEL_IDS:
        raise UCISourceRunError(f"model must be one of {UCI_CORRECTED_MODEL_IDS}")
    commit = _code_commit(code_commit)
    if not torch.cuda.is_available():
        raise UCISourceRunError("CUDA is required for corrected UCI source training")
    _validate_repository_head(repository_root, expected_commit=commit)
    experiment_config = load_uci_reproduction_config(
        experiment_config_path,
        expected_file_sha256=expected_experiment_config_file_sha256,
    )
    if fold_id not in experiment_config.fold_order:
        raise UCISourceRunError(f"fold is outside the locked matrix: {fold_id}")
    _validate_matrix_paths(
        repository_root=repository_root,
        experiment_config=experiment_config,
        archive_path=archive_path,
        dataset_manifest_path=dataset_manifest_path,
        protocol_path=protocol_path,
        model_name=model_name,
        fold_id=fold_id,
        seed=seed,
        attempt=attempt,
        run_directory=run_directory,
        summary_path=summary_path,
        allowed_output_root=allowed_output_root,
    )
    configuration = build_uci_training_config(
        experiment_config,
        model_name=model_name,
        seed=seed,
    )
    run_path = _confined_output(
        run_directory, allowed_root=allowed_output_root, kind="run directory"
    )
    summary_output = _confined_output(
        summary_path, allowed_root=allowed_output_root, kind="summary path"
    )
    if run_path.exists() or summary_output.exists():
        raise FileExistsError("refusing to overwrite an existing UCI run directory or summary")
    result_root = (
        repository_root.resolve(strict=True) / str(experiment_config.outputs["root"])
    ).resolve(strict=False)
    prior_failures = _validate_retry_chain(
        run_path=run_path,
        summary_path=summary_output,
        result_root=result_root,
        model_name=model_name,
        fold_id=fold_id,
        seed=seed,
        attempt=attempt,
        code_commit=commit,
        experiment_config=experiment_config,
    )
    run_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    run_path.mkdir(exist_ok=False)
    device = torch.device("cuda")
    started = time.perf_counter()
    prepared: PreparedUCISourceFold | None = None
    execution_stage = "official_train_fold_preparation"
    try:
        prepared = prepare_uci_source_fold(
            archive_path=archive_path,
            dataset_manifest_path=dataset_manifest_path,
            protocol_path=protocol_path,
            fold_id=fold_id,
        )
        preprocessing_sha256 = canonical_json_sha256(
            {
                "channels": list(UCI_HAR_CHANNELS),
                "normalization": "per_channel_population_standardization_fold_train_only",
                "tensor_shape": [128, 6],
            }
        )
        ontology_sha256 = canonical_json_sha256(
            {str(index): name for index, name in UCI_HAR_ACTIVITY_NAMES.items()}
        )
        lineage = TrainingLineage(
            dataset_manifest_sha256=prepared.dataset_manifest_sha256,
            split_manifest_sha256=prepared.protocol_sha256,
            preprocessing_config_sha256=preprocessing_sha256,
            ontology_sha256=ontology_sha256,
            code_commit=commit,
            evidence_status="corrected_uci_source_grouped_development_official_test_unopened",
            label_schema=UCI_CLASS_NAMES,
            normalization=prepared.standardizer.to_dict(),
        )
        execution_stage = "cuda_peak_memory_reset"
        torch.cuda.reset_peak_memory_stats(device)
        execution_stage = "cuda_source_training"
        trained = train_source_model(
            prepared.train_windows,
            prepared.train_labels,
            prepared.train_participant_ids,
            prepared.validation_windows,
            prepared.validation_labels,
            prepared.validation_participant_ids,
            config=configuration,
            lineage=lineage,
            output_directory=run_path,
            device=device,
        )
        logits = cast(NDArray[np.float64], trained.pop("validation_logits"))
        probabilities = cast(NDArray[np.float64], trained.pop("validation_probabilities"))
        report = cast(dict[str, Any], trained.pop("validation_report"))
        checkpoint_path_value = trained.get("checkpoint_path")
        if not isinstance(checkpoint_path_value, str):
            raise UCISourceRunError("training result lacks its selected checkpoint path")
        checkpoint_path = Path(checkpoint_path_value).resolve(strict=True)
        if checkpoint_path.is_symlink() or sha256_file(checkpoint_path) != trained.get(
            "checkpoint_sha256"
        ):
            raise UCISourceRunError("selected checkpoint hash changed before publication")
        trained["checkpoint_path"] = _portable_output_reference(
            checkpoint_path,
            allowed_root=result_root,
            kind="selected checkpoint",
        )
        trained["checkpoint_path_base"] = "result_root"
        execution_stage = "validation_prediction_publication"
        prediction_path = run_path / "source_validation_predictions.npz"
        prediction_sha256 = _write_predictions_new(
            prediction_path,
            logits=logits,
            probabilities=probabilities,
            labels=prepared.validation_labels,
            participant_ids=prepared.validation_participant_ids,
            window_ids=prepared.validation_window_ids,
        )
        torch.cuda.synchronize(device)
        properties = torch.cuda.get_device_properties(device)
        summary: dict[str, Any] = {
            "schema_version": "1.0.0",
            "status": "corrected_uci_source_fold_complete_official_test_unopened",
            "evidence_status": "source_grouped_development_not_confirmatory",
            "dataset_id": "uci_har_v1",
            "dataset_manifest_sha256": prepared.dataset_manifest_sha256,
            "processed_archive_sha256": prepared.windows.archive_sha256,
            "source_protocol_sha256": prepared.protocol_sha256,
            "released_split_loaded": "train",
            "official_test_member_opened": False,
            "official_test_performance_or_prediction_accessed": False,
            "inclusivehar_data_or_target_accessed": False,
            "model_name": model_name,
            "seed": seed,
            "attempt": attempt,
            "prior_attempt_failures": prior_failures,
            "code_commit": commit,
            "experiment_config": {
                "experiment_id": experiment_config.experiment_id,
                "file_sha256": experiment_config.file_sha256,
                "canonical_sha256": experiment_config.canonical_sha256,
            },
            "fold": {
                "fold_id": prepared.fold.fold_id,
                "train_subject_ids": list(prepared.fold.train_subject_ids),
                "validation_subject_ids": list(prepared.fold.validation_subject_ids),
                "train_window_count": int(prepared.train_labels.size),
                "validation_window_count": int(prepared.validation_labels.size),
                "train_window_ids_sha256": prepared.fold.train_window_ids_sha256,
                "validation_window_ids_sha256": prepared.fold.validation_window_ids_sha256,
            },
            "class_names": list(UCI_CLASS_NAMES),
            "normalization": prepared.standardizer.to_dict(),
            "configuration": asdict(configuration),
            "configuration_sha256": canonical_json_sha256(asdict(configuration)),
            "training": trained,
            "validation_report": report,
            "prediction_artifact": {
                "path": _portable_output_reference(
                    prediction_path,
                    allowed_root=result_root,
                    kind="prediction artifact",
                ),
                "path_base": "result_root",
                "sha256": prediction_sha256,
            },
            "device": {
                "requested": "cuda",
                "actual": "cuda",
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": list(torch.cuda.get_device_capability(device)),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cudnn_enabled": torch.backends.cudnn.enabled,
                "cudnn_version": cast(Any, torch.backends.cudnn).version(),
                "mixed_precision": configuration.mixed_precision,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        summary["record_sha256"] = canonical_json_sha256(summary)
        atomic_write_json_new(summary, summary_output, allowed_root=allowed_output_root)
        return summary
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": "1.0.0",
            "status": "corrected_uci_source_fold_failed_preserved",
            "evidence_status": "failed_source_development_not_model_result",
            "execution_stage": execution_stage,
            "dataset_manifest_sha256": (
                None if prepared is None else prepared.dataset_manifest_sha256
            ),
            "source_protocol_sha256": None if prepared is None else prepared.protocol_sha256,
            "fold_id": fold_id,
            "model_name": model_name,
            "seed": seed,
            "attempt": attempt,
            "prior_attempt_failures": prior_failures,
            "code_commit": commit,
            "experiment_config": {
                "experiment_id": experiment_config.experiment_id,
                "file_sha256": experiment_config.file_sha256,
                "canonical_sha256": experiment_config.canonical_sha256,
            },
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "input_files": {
                "archive_sha256": sha256_file(archive_path),
                "dataset_manifest_file_sha256": sha256_file(dataset_manifest_path),
                "protocol_file_sha256": sha256_file(protocol_path),
            },
            "official_test_member_opened": False,
            "official_test_performance_or_prediction_accessed": False,
            "inclusivehar_data_or_target_accessed": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        failure["record_sha256"] = canonical_json_sha256(failure)
        atomic_write_json_new(failure, run_path / "failure.json", allowed_root=allowed_output_root)
        raise UCISourceRunError(
            f"UCI source run failed; evidence preserved at {run_path / 'failure.json'}"
        ) from exc
