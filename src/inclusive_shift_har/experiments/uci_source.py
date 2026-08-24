"""CUDA-only, official-train-only corrected UCI-HAR reproduction runner."""

from __future__ import annotations

import time
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


class UCISourceRunError(RuntimeError):
    """Raised when a corrected source-only run fails closed."""


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


def run_uci_source_fold(
    *,
    archive_path: Path,
    dataset_manifest_path: Path,
    protocol_path: Path,
    model_name: str,
    fold_id: str,
    seed: int,
    code_commit: str,
    run_directory: Path,
    summary_path: Path,
    allowed_output_root: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    minimum_epochs: int,
    checkpoint_selection_rule: str,
    mixed_precision: str = "float16",
    disable_cudnn: bool = False,
) -> dict[str, Any]:
    """Run one immutable grouped source fold on CUDA; official test stays unopened."""

    if model_name not in UCI_CORRECTED_MODEL_IDS:
        raise UCISourceRunError(f"model must be one of {UCI_CORRECTED_MODEL_IDS}")
    commit = _code_commit(code_commit)
    if not torch.cuda.is_available():
        raise UCISourceRunError("CUDA is required for corrected UCI source training")
    run_path = _confined_output(
        run_directory, allowed_root=allowed_output_root, kind="run directory"
    )
    summary_output = _confined_output(
        summary_path, allowed_root=allowed_output_root, kind="summary path"
    )
    if run_path.exists() or summary_output.exists():
        raise FileExistsError("refusing to overwrite an existing UCI run directory or summary")

    prepared = prepare_uci_source_fold(
        archive_path=archive_path,
        dataset_manifest_path=dataset_manifest_path,
        protocol_path=protocol_path,
        fold_id=fold_id,
    )
    configuration = TrainingConfig(
        model_name=model_name,
        num_classes=len(UCI_CLASS_NAMES),
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        minimum_epochs=minimum_epochs,
        mixed_precision=mixed_precision,
        checkpoint_interval=max(1, epochs),
        checkpoint_selection_rule=checkpoint_selection_rule,
        disable_cudnn=disable_cudnn,
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
    run_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    run_path.mkdir(exist_ok=False)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
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
            "code_commit": commit,
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
                "path": prediction_path.as_posix(),
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
                "mixed_precision": mixed_precision,
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
            "dataset_manifest_sha256": prepared.dataset_manifest_sha256,
            "source_protocol_sha256": prepared.protocol_sha256,
            "fold_id": fold_id,
            "model_name": model_name,
            "seed": seed,
            "code_commit": commit,
            "exception_type": type(exc).__name__,
            "message": str(exc),
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
