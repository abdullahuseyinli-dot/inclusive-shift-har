"""Execute and independently replay the fixed corrected-FoG motion-factorization study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import platform
import random
import subprocess
import sys
import time
import traceback
import warnings
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, TypeAlias, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import psutil  # type: ignore[import-untyped]
import scipy  # type: ignore[import-untyped]
import sklearn  # type: ignore[import-untyped]
import torch
import yaml
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]
from threadpoolctl import threadpool_info, threadpool_limits  # type: ignore[import-untyped]
from torch import Tensor, nn
from torch.nn import functional as F

from inclusive_shift_har.experiments.fog_left_ankle_derived_nine import (
    _input_paths,
    _stop_task_owned_workers,
    _verify_spatial_reference,
    _worker_baseline,
)
from inclusive_shift_har.experiments.fog_motion_factorization import (
    MotionResidualTCN,
    apply_history_block_permutation,
    compose_motion_probability,
    history_block_permutation,
    participant_class_weights,
    preflight_fog_motion_context,
)
from inclusive_shift_har.experiments.fog_rf_feature_weight_factorial import (
    _array_sha256,
    _git_state,
    _mapping,
    _require,
    _sealed,
    _verify_sealed,
    _write_bytes_create_only,
    _write_json_create_only,
    method_report,
    paired_comparison,
)
from inclusive_shift_har.experiments.fog_spatial_information_probe import (
    _events,
    _leave_one_fold,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file

FloatArray: TypeAlias = NDArray[np.float64]
Float32Array: TypeAlias = NDArray[np.float32]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
StringArray: TypeAlias = NDArray[np.str_]

EXPERIMENT_ID = "fog-motion-factorization-v1"
RUN_DIRECTORY_NAME = "fog-motion-factorization-seed11-20260908-001"
EVIDENCE_FAMILY = Path(".audit/fog_motion_factorization")
CONFIG_RELATIVE = Path("configs/experiments/fog_motion_factorization_v1.yaml")
PROTOCOL_RELATIVE = Path("docs/research/FOG_MOTION_FACTORIZATION_V1_PROTOCOL.md")
SPEC_RELATIVE = Path(".audit/research_invention_review_20260908-001/EXACT_NEXT_EXPERIMENT.md")
SPEC_SHA256 = "0310bed0c618031119df8b6df6161c84f777f15b32663ae5fb7f5ce31b7a9224"
CONFIG_SEMANTIC_SHA256 = "4fd758eae61f8e505e64c421eff298b24d3b08cbf45bc5676fffcbb22dc1ec77"
REFERENCE_RELATIVE = Path(
    ".audit/fog_left_ankle_derived_nine/fog-left-ankle-derived-nine-seed11-20260908-003"
)
CELL_ORDER = ("E2", "T128", "T500", "T500-P")
REFERENCE_ORDER = ("l9v", "lv", "bv", "b0", "f3")
METHOD_ORDER = (*CELL_ORDER, *REFERENCE_ORDER)
OUTER_FOLDS = tuple(range(5))
CLASS_NAMES = ("mobility", "sitting", "standing")
MAXIMUM_FIT_ATTEMPTS = 20
EXPECTED_PARAMETER_COUNT = 62_881
FIT_SCHEDULE = tuple((fold, cell) for fold in OUTER_FOLDS for cell in CELL_ORDER)
_THREADPOOL_CONTROLLER: Any = None


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


def _write_pickle_create_only(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        pickle.dump(value, stream, protocol=5)


def _write_torch_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(dict(value), stream)


def _canonical_config_sha256(config: Mapping[str, Any]) -> str:
    value = dict(config)
    value.pop("protocol_sha256", None)
    return canonical_json_sha256(value)


def validate_config(config: Mapping[str, Any], *, protocol_path: Path | None = None) -> None:
    """Validate the complete frozen record and its cross-file binding."""

    _require(config.get("experiment_id") == EXPERIMENT_ID, "experiment ID changed")
    _require(config.get("status") == "frozen_before_fitting", "config is not frozen")
    _require(
        _canonical_config_sha256(config) == CONFIG_SEMANTIC_SHA256,
        "configuration semantic identity changed",
    )
    _require(
        _mapping(config["authority"], "authority")["specification_sha256"] == SPEC_SHA256,
        "specification hash changed",
    )
    _require(_mapping(config["cells"], "cells")["order"] == list(CELL_ORDER), "cell order changed")
    runtime = _mapping(config["runtime"], "runtime")
    _require(runtime["maximum_fit_attempts"] == MAXIMUM_FIT_ATTEMPTS, "fit cap changed")
    _require(runtime["fit_order"] == "outer_fold_then_E2_T128_T500_T500_P", "fit order changed")
    _require(
        runtime["additional_seed"] is False and runtime["automatic_retry"] is False,
        "retry or seed policy changed",
    )
    context = _mapping(config["context"], "context")
    _require(
        context["query_samples"] == 128
        and context["long_samples"] == 500
        and context["expected_observable_full_history"] == 1724
        and context["expected_scored_full_history"] == 1098,
        "context contract changed",
    )
    temporal = _mapping(config["temporal"], "temporal")
    _require(
        temporal["dilations"] == [1, 2, 4, 8, 16, 32]
        and temporal["expected_trainable_parameters"] == EXPECTED_PARAMETER_COUNT,
        "temporal architecture changed",
    )
    _require(_mapping(config["training"], "training")["epochs"] == 80, "epoch count changed")
    if protocol_path is not None:
        _require(protocol_path.is_file(), "protocol missing")
        _require(sha256_file(protocol_path) == config["protocol_sha256"], "protocol hash changed")


def _environment() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    gpu: dict[str, Any] | None = None
    if cuda:
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": int(properties.total_memory),
            "capability": list(torch.cuda.get_device_capability(0)),
            "device_count": torch.cuda.device_count(),
        }
    driver = subprocess.run(
        ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    distributions = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib_metadata.distributions()
        if distribution.metadata["Name"]
    )
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "implementation_file": str(Path(__file__).resolve()),
        "motion_context_implementation_file": str(
            Path(str(sys.modules[MotionResidualTCN.__module__].__file__)).resolve()
        ),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "PyYAML": yaml.__version__,
        "joblib": joblib.__version__,
        "psutil": psutil.__version__,
        "cuda_available": cuda,
        "torch_cuda": torch.version.cuda,
        "gpu": gpu,
        "nvidia_driver": driver.stdout.strip() if driver.returncode == 0 else None,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "dependency_inventory": distributions,
        "dependency_inventory_sha256": canonical_json_sha256(distributions),
        "threadpool_info": threadpool_info(),
    }


def _require_environment(config: Mapping[str, Any]) -> dict[str, Any]:
    observed = _environment()
    expected = _mapping(_mapping(config["runtime"], "runtime")["expected_versions"], "versions")
    _require(
        platform.python_version() == _mapping(config["runtime"], "runtime")["expected_python"],
        "Python version changed",
    )
    for name in ("numpy", "scipy", "scikit-learn", "torch", "PyYAML", "joblib"):
        key = "scikit_learn" if name == "scikit-learn" else name
        _require(observed[key] == expected[name], f"runtime version changed: {name}")
    _require(observed["cuda_available"] is True and observed["gpu"] is not None, "CUDA unavailable")
    _require(
        observed["torch_num_threads"] == observed["torch_num_interop_threads"] == 1
        and all(int(row.get("num_threads", 1)) == 1 for row in observed["threadpool_info"]),
        "single-thread runtime contract changed",
    )
    return observed


def _configure_determinism(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _configure_runtime(seed: int) -> None:
    """Apply the frozen single-controller deterministic runtime before CUDA work."""

    global _THREADPOOL_CONTROLLER
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    if _THREADPOOL_CONTROLLER is None:
        _THREADPOOL_CONTROLLER = threadpool_limits(limits=1)
    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    _configure_determinism(seed)


def _state_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str((value.shape, value.dtype.str)).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _rng_receipt() -> dict[str, Any]:
    cpu = torch.get_rng_state().cpu().numpy()
    cuda = torch.cuda.get_rng_state(0).cpu().numpy()
    return {
        "torch_cpu_rng_state_sha256": _array_sha256(cpu),
        "torch_cuda_rng_state_sha256": _array_sha256(cuda),
    }


def _source_manifest(repository_root: Path) -> dict[str, Any]:
    paths = (
        "AGENTS.md",
        "src/inclusive_shift_har/data/external_har.py",
        "src/inclusive_shift_har/preprocessing/features.py",
        "src/inclusive_shift_har/experiments/fog_rf_feature_weight_factorial.py",
        "src/inclusive_shift_har/experiments/fog_spatial_information_probe.py",
        "src/inclusive_shift_har/experiments/fog_left_ankle_derived_nine.py",
        "src/inclusive_shift_har/experiments/fog_motion_factorization.py",
        "src/inclusive_shift_har/experiments/fog_motion_factorization_run.py",
        "tests/test_fog_motion_factorization_context.py",
        "tests/test_fog_motion_factorization_contract.py",
        "tests/test_fog_motion_factorization_run.py",
        "pyproject.toml",
        "uv.lock",
        CONFIG_RELATIVE.as_posix(),
        PROTOCOL_RELATIVE.as_posix(),
    )
    return {
        "record_kind": "fog_motion_factorization_source_manifest",
        "files": [{"path": name, "sha256": sha256_file(repository_root / name)} for name in paths],
    }


def _fit_binding(
    repository_root: Path,
    *,
    code_commit: str,
    source_manifest_record_sha256: str,
    input_manifest_record_sha256: str,
) -> dict[str, Any]:
    """Recheck immutable implementation inputs and return their receipt binding."""

    git = _git_state(repository_root)
    _require(
        git["clean"] is True and git["commit"] == code_commit, "source changed before fit launch"
    )
    _require(
        canonical_json_sha256(_source_manifest(repository_root)) == source_manifest_record_sha256,
        "source manifest changed before fit launch",
    )
    config_path = repository_root / CONFIG_RELATIVE
    protocol_path = repository_root / PROTOCOL_RELATIVE
    config = _read_yaml(config_path)
    validate_config(config, protocol_path=protocol_path)
    _require(
        (repository_root / SPEC_RELATIVE).is_file()
        and sha256_file(repository_root / SPEC_RELATIVE) == SPEC_SHA256,
        "governing specification changed before fit launch",
    )
    return {
        "code_commit": code_commit,
        "source_manifest_record_sha256": source_manifest_record_sha256,
        "input_manifest_record_sha256": input_manifest_record_sha256,
        "config_sha256": sha256_file(config_path),
        "config_semantic_sha256": CONFIG_SEMANTIC_SHA256,
        "protocol_sha256": sha256_file(protocol_path),
        "specification_sha256": SPEC_SHA256,
    }


def _recheck_fit_binding(repository_root: Path, binding: Mapping[str, Any]) -> None:
    current = _fit_binding(
        repository_root,
        code_commit=str(binding["code_commit"]),
        source_manifest_record_sha256=str(binding["source_manifest_record_sha256"]),
        input_manifest_record_sha256=str(binding["input_manifest_record_sha256"]),
    )
    _require(current == dict(binding), "source binding changed during fit")


def _verify_reference(
    evidence_root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    section = _mapping(config["reference"], "reference")
    run = (evidence_root / str(section["run_path"])).resolve()
    _require(
        run == (evidence_root / REFERENCE_RELATIVE).resolve() and run.is_dir(),
        "L9v reference run changed",
    )
    checked = []
    for name, expected in _mapping(section["files"], "reference files").items():
        path = run / name
        _require(path.is_file() and sha256_file(path) == expected, f"reference changed: {name}")
        checked.append({"path": str(path), "sha256": expected, "size_bytes": path.stat().st_size})
    erratum = (evidence_root / str(section["erratum_path"])).resolve()
    _require(
        erratum.is_file() and sha256_file(erratum) == section["erratum_sha256"],
        "L9v erratum changed",
    )
    predictions = _read_npz(run / "predictions.npz")
    features = _read_npz(run / "feature_cache.npz")
    _require(predictions["method_ids"].tolist() == list(REFERENCE_ORDER), "reference order changed")
    arrays = {
        **features,
        **{f"reference_{name}": value for name, value in predictions.items()},
    }
    return arrays, {
        "record_kind": "fog_motion_factorization_reference_receipt",
        "run": str(run),
        "checked_files": checked,
        "erratum": {
            "path": str(erratum),
            "sha256": section["erratum_sha256"],
            "text": erratum.read_text(encoding="utf-8"),
        },
    }


def current_energy_features(context_signals: FloatArray) -> FloatArray:
    """Return the two frozen log-RMS features from the final 128 samples."""

    values = np.asarray(context_signals, dtype=np.float64)
    _require(values.ndim == 3 and values.shape[1:] == (500, 6), "context shape changed")
    current = values[:, -128:]
    acceleration = np.linalg.vector_norm(current[:, :, :3], axis=2)
    gyroscope = np.linalg.vector_norm(current[:, :, 3:], axis=2)
    rms = np.column_stack(
        (np.sqrt(np.mean(acceleration**2, axis=1)), np.sqrt(np.mean(gyroscope**2, axis=1)))
    )
    result = np.log(np.maximum(rms, 1e-8))
    _require(bool(np.isfinite(result).all()), "energy features are invalid")
    return cast(FloatArray, result)


def _standardize_fit(values: FloatArray) -> tuple[FloatArray, FloatArray]:
    mean = np.asarray(values.mean(axis=0), dtype=np.float64)
    scale = np.asarray(values.std(axis=0, ddof=0), dtype=np.float64)
    scale[scale == 0.0] = 1.0
    return mean, scale


def _standardize(values: FloatArray, mean: FloatArray, scale: FloatArray) -> FloatArray:
    result = (np.asarray(values, dtype=np.float64) - mean) / scale
    _require(bool(np.isfinite(result).all()), "standardized values are invalid")
    return result


def _weight_summary(
    labels: IntArray, participants: StringArray, weights: FloatArray
) -> dict[str, Any]:
    rows = []
    for participant in sorted(np.unique(participants).tolist()):
        selected_person = participants == participant
        class_rows = []
        for label in range(3):
            selected = selected_person & (labels == label)
            class_rows.append(
                {
                    "class_index": label,
                    "row_count": int(selected.sum()),
                    "weight_total": float(weights[selected].sum()),
                }
            )
        rows.append(
            {
                "participant_id": participant,
                "row_count": int(selected_person.sum()),
                "weight_total": float(weights[selected_person].sum()),
                "original_classes": class_rows,
            }
        )
    target = labels == 0
    return {
        "rows": rows,
        "original_class_totals": [float(weights[labels == value].sum()) for value in range(3)],
        "binary_stationary_motion_totals": [
            float(weights[~target].sum()),
            float(weights[target].sum()),
        ],
        "unweighted_mean": float(weights.mean()),
        "total": float(weights.sum()),
        "sha256": _array_sha256(weights),
    }


def _fit_e2(
    *,
    features: FloatArray,
    labels: IntArray,
    participants: StringArray,
    training_indices: IntArray,
    prediction_indices: IntArray,
    fold: int,
    attempt: int,
    output_directory: Path,
    binding: Mapping[str, Any],
    repository_root: Path,
) -> tuple[FloatArray, dict[str, Any], dict[str, Any]]:
    _require(1 <= attempt <= MAXIMUM_FIT_ATTEMPTS, "fit attempt exceeds cap")
    started = time.perf_counter()
    target = np.asarray(labels[training_indices] == 0, dtype=np.int64)
    weights = participant_class_weights(labels[training_indices], participants[training_indices])
    mean, scale = _standardize_fit(features[training_indices])
    estimator = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        fit_intercept=True,
        max_iter=1000,
        tol=1e-6,
        class_weight=None,
        warm_start=False,
    )
    started_record = _sealed(
        {
            "record_kind": "fog_motion_fit_attempt_started",
            "attempt_number": attempt,
            "cell_id": "E2",
            "outer_fold": fold,
            "fold_seed": 11 + fold,
            "training_row_count": int(training_indices.size),
            "prediction_candidate_count": int(prediction_indices.size),
            "training_indices_sha256": _array_sha256(training_indices),
            "prediction_indices_sha256": _array_sha256(prediction_indices),
            "sample_weight_sha256": _array_sha256(weights),
            "started_at_utc": _now(),
            **dict(binding),
        }
    )
    _write_json_create_only(
        output_directory / "fit_attempts" / f"{attempt:02d}--started.json", started_record
    )
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        estimator.fit(
            _standardize(features[training_indices], mean, scale), target, sample_weight=weights
        )
        predicted = np.asarray(
            estimator.predict_proba(_standardize(features[prediction_indices], mean, scale))[:, 1],
            dtype=np.float64,
        )
    convergence_warnings = [
        str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)
    ]
    _require(not convergence_warnings, f"E2 convergence warning: {convergence_warnings}")
    _require(estimator.classes_.tolist() == [0, 1], "E2 class order changed")
    _require(bool(np.all(estimator.n_iter_ < 1000)), "E2 did not converge")
    fit_seconds = time.perf_counter() - started
    train_logits = np.asarray(
        estimator.decision_function(_standardize(features[training_indices], mean, scale)),
        dtype=np.float64,
    )
    data_losses = np.logaddexp(0.0, train_logits) - target * train_logits
    weighted_data_log_loss = float(np.sum(weights * data_losses) / np.sum(weights))
    regularization = float(np.sum(estimator.coef_**2) / (2.0 * np.sum(weights)))
    checkpoint_body = {
        "cell_id": "E2",
        "outer_fold": fold,
        "mean": mean,
        "scale": scale,
        "estimator": estimator,
        "feature_names": ("accelerometer_norm__rms", "gyroscope_norm__rms"),
        "training_indices_sha256": _array_sha256(training_indices),
        "prediction_indices_sha256": _array_sha256(prediction_indices),
        **dict(binding),
    }
    checkpoint_path = output_directory / "checkpoints" / f"E2--fold-{fold}.pkl"
    _write_pickle_create_only(checkpoint_path, checkpoint_body)
    _recheck_fit_binding(repository_root, binding)
    metadata = {
        "attempt_number": attempt,
        "cell_id": "E2",
        "outer_fold": fold,
        "fold_seed": 11 + fold,
        "training_row_count": int(training_indices.size),
        "prediction_candidate_count": int(prediction_indices.size),
        "training_indices_sha256": _array_sha256(training_indices),
        "prediction_indices_sha256": _array_sha256(prediction_indices),
        "training_original_class_counts": np.bincount(
            labels[training_indices], minlength=3
        ).tolist(),
        "training_binary_class_counts": np.bincount(target, minlength=2).tolist(),
        "sample_weight_sha256": _array_sha256(weights),
        "sample_weight_sum": float(weights.sum()),
        "binary_weight_totals": [float(weights[target == value].sum()) for value in (0, 1)],
        "normalizer_mean": mean.tolist(),
        "normalizer_scale": scale.tolist(),
        "coefficient": estimator.coef_.tolist(),
        "intercept": estimator.intercept_.tolist(),
        "iterations": estimator.n_iter_.tolist(),
        "fit_and_predict_seconds": fit_seconds,
        "weighted_training_data_log_loss": weighted_data_log_loss,
        "explicit_regularization_term": regularization,
        "explicit_weighted_objective": weighted_data_log_loss + regularization,
        "convergence_warnings": convergence_warnings,
        "weight_summary": _weight_summary(
            labels[training_indices], participants[training_indices], weights
        ),
        "source_binding_reverified_at_completion": True,
        **dict(binding),
        "checkpoint": {
            "path": checkpoint_path.relative_to(output_directory).as_posix(),
            "sha256": sha256_file(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
        },
    }
    completed = _sealed(
        {"record_kind": "fog_motion_fit_attempt_completed", **metadata, "completed_at_utc": _now()}
    )
    _write_json_create_only(
        output_directory / "fit_attempts" / f"{attempt:02d}--completed.json", completed
    )
    state = {"estimator": estimator, "mean": mean, "scale": scale, "weights": weights}
    return predicted, metadata, state


def _epoch_orders(count: int, seed: int, epochs: int = 80) -> list[IntArray]:
    rng = np.random.default_rng(seed)
    return [np.asarray(rng.permutation(count), dtype=np.int64) for _ in range(epochs)]


def _epoch_learning_rate(epoch: int) -> float:
    return 1e-5 + 0.5 * (1e-3 - 1e-5) * (1.0 + math.cos(math.pi * epoch / 79.0))


def _prepare_temporal_inputs(
    signals: FloatArray,
    observation_mask: BoolArray,
    normalizer_mean: FloatArray,
    normalizer_scale: FloatArray,
    window_ids: StringArray,
    cell_id: str,
) -> tuple[Float32Array, Float32Array]:
    values = np.asarray((signals - normalizer_mean) / normalizer_scale, dtype=np.float32)
    mask = np.asarray(observation_mask, dtype=np.float32).copy()
    _require(mask.shape == values.shape[:2], "observation mask shape changed")
    if cell_id == "T128":
        values[:, :372] = 0.0
        mask[:, :372] = 0.0
    elif cell_id == "T500-P":
        for index, window_id in enumerate(window_ids.tolist()):
            values[index], mask[index] = apply_history_block_permutation(
                values[index], mask[index], history_block_permutation(str(window_id))
            )
    elif cell_id != "T500":
        raise ValueError(f"unsupported temporal cell: {cell_id}")
    values *= mask[:, :, None]
    _require(bool(np.isfinite(values).all()), "temporal input invalid")
    return values, mask


def _predict_temporal(
    model: MotionResidualTCN,
    values: Float32Array,
    mask: Float32Array,
    e2_logits: FloatArray,
    indices: IntArray,
    *,
    device: torch.device,
    batch_size: int = 32,
) -> FloatArray:
    result = np.empty(indices.size, dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for left in range(0, indices.size, batch_size):
            selected = indices[left : left + batch_size]
            output = model(
                torch.from_numpy(values[selected]).to(device),
                torch.from_numpy(mask[selected]).to(device),
                torch.from_numpy(np.asarray(e2_logits[selected], dtype=np.float32)).to(device),
            )
            result[left : left + selected.size] = (
                torch.sigmoid(output).cpu().numpy().astype(np.float64)
            )
    return result


def _fit_temporal(
    *,
    cell_id: str,
    signals: FloatArray,
    observation_mask: BoolArray,
    window_ids: StringArray,
    labels: IntArray,
    participants: StringArray,
    training_indices: IntArray,
    prediction_indices: IntArray,
    e2_logits_all: FloatArray,
    fold: int,
    attempt: int,
    shared_initial_state: Mapping[str, Tensor],
    output_directory: Path,
    device: torch.device,
    binding: Mapping[str, Any],
    e2_checkpoint_sha256: str,
    cancel_file: Path | None,
    repository_root: Path,
) -> tuple[FloatArray, dict[str, Any]]:
    _require(
        cell_id in CELL_ORDER[1:] and 1 <= attempt <= MAXIMUM_FIT_ATTEMPTS,
        "temporal fit contract changed",
    )
    fit_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    current_training = signals[training_indices, -128:].reshape(-1, 6)
    mean, scale = _standardize_fit(current_training)
    values, mask = _prepare_temporal_inputs(
        signals, observation_mask, mean, scale, window_ids, cell_id
    )
    weights64 = participant_class_weights(labels[training_indices], participants[training_indices])
    target = np.asarray(labels[training_indices] == 0, dtype=np.float32)
    orders = _epoch_orders(training_indices.size, 11 + fold)
    order_array = np.stack(orders)
    _configure_determinism(11 + fold)
    expected_training_rng = _rng_receipt()
    started_record = _sealed(
        {
            "record_kind": "fog_motion_fit_attempt_started",
            "attempt_number": attempt,
            "cell_id": cell_id,
            "outer_fold": fold,
            "fold_seed": 11 + fold,
            "training_row_count": int(training_indices.size),
            "prediction_candidate_count": int(prediction_indices.size),
            "training_indices_sha256": _array_sha256(training_indices),
            "prediction_indices_sha256": _array_sha256(prediction_indices),
            "sample_weight_sha256": _array_sha256(weights64),
            "batch_orders_sha256": _array_sha256(order_array),
            "shared_initial_state_sha256": _state_sha256(shared_initial_state),
            "frozen_e2_checkpoint_sha256": e2_checkpoint_sha256,
            "training_rng": expected_training_rng,
            "started_at_utc": _now(),
            **dict(binding),
        }
    )
    _write_json_create_only(
        output_directory / "fit_attempts" / f"{attempt:02d}--started.json", started_record
    )
    _configure_determinism(11 + fold)
    model = MotionResidualTCN().to(device)
    model.load_state_dict(shared_initial_state, strict=True)
    _configure_determinism(11 + fold)
    training_rng = _rng_receipt()
    _require(training_rng == expected_training_rng, "training RNG reset changed")
    _require(
        sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETER_COUNT,
        "parameter count changed",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
        amsgrad=False,
        foreach=False,
        fused=False,
    )
    x = torch.from_numpy(values).to(device)
    observed = torch.from_numpy(mask).to(device)
    e2 = torch.from_numpy(np.asarray(e2_logits_all, dtype=np.float32)).to(device)
    y = torch.from_numpy(target).to(device)
    weights = torch.from_numpy(np.asarray(weights64, dtype=np.float32)).to(device)
    train_global = torch.from_numpy(training_indices).to(device)
    history: list[dict[str, Any]] = []
    peak_allocated = 0
    for epoch, order in enumerate(orders):
        if cancel_file is not None and cancel_file.exists():
            raise KeyboardInterrupt(
                f"cooperative cancellation requested during {cell_id} fold {fold} epoch {epoch}"
            )
        lr = _epoch_learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        loss_numerator = 0.0
        loss_denominator = 0.0
        max_gradient_norm = 0.0
        for left in range(0, order.size, 32):
            local = torch.from_numpy(order[left : left + 32]).to(device)
            selected = train_global[local]
            batch_weight = weights[local]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[selected], observed[selected], e2[selected])
            losses = F.binary_cross_entropy_with_logits(logits, y[local], reduction="none")
            numerator = torch.sum(batch_weight * losses)
            denominator = torch.sum(batch_weight)
            loss = numerator / denominator
            _require(bool(torch.isfinite(loss).item()), "nonfinite training loss")
            loss.backward()  # type: ignore[no-untyped-call]
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0, norm_type=2.0, error_if_nonfinite=True
            )
            optimizer.step()
            loss_numerator += float(numerator.detach().cpu())
            loss_denominator += float(denominator.detach().cpu())
            max_gradient_norm = max(max_gradient_norm, float(gradient_norm.detach().cpu()))
            peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated(device)))
        history.append(
            {
                "epoch": epoch,
                "learning_rate": lr,
                "weighted_training_loss": loss_numerator / loss_denominator,
                "maximum_preclip_gradient_norm": max_gradient_norm,
            }
        )
    model.eval()
    final_numerator = 0.0
    final_denominator = 0.0
    with torch.inference_mode():
        for left in range(0, training_indices.size, 32):
            local = torch.arange(left, min(left + 32, training_indices.size), device=device)
            selected = train_global[local]
            batch_weight = weights[local]
            losses = F.binary_cross_entropy_with_logits(
                model(x[selected], observed[selected], e2[selected]), y[local], reduction="none"
            )
            final_numerator += float(torch.sum(batch_weight * losses).cpu())
            final_denominator += float(torch.sum(batch_weight).cpu())
    probabilities = _predict_temporal(
        model, values, mask, e2_logits_all, prediction_indices, device=device
    )
    fit_seconds = time.perf_counter() - fit_started
    checkpoint_path = output_directory / "checkpoints" / f"{cell_id}--fold-{fold}.pt"
    state_cpu = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    _write_torch_create_only(
        checkpoint_path,
        {
            "cell_id": cell_id,
            "outer_fold": fold,
            "state_dict": state_cpu,
            "normalizer_mean": mean,
            "normalizer_scale": scale,
            "shared_initial_state_sha256": _state_sha256(shared_initial_state),
            "frozen_e2_checkpoint_sha256": e2_checkpoint_sha256,
            "training_indices_sha256": _array_sha256(training_indices),
            "prediction_indices_sha256": _array_sha256(prediction_indices),
            **dict(binding),
        },
    )
    _recheck_fit_binding(repository_root, binding)
    metadata = {
        "attempt_number": attempt,
        "cell_id": cell_id,
        "outer_fold": fold,
        "fold_seed": 11 + fold,
        "training_row_count": int(training_indices.size),
        "prediction_candidate_count": int(prediction_indices.size),
        "training_indices_sha256": _array_sha256(training_indices),
        "prediction_indices_sha256": _array_sha256(prediction_indices),
        "training_original_class_counts": np.bincount(
            labels[training_indices], minlength=3
        ).tolist(),
        "training_binary_class_counts": np.bincount(target.astype(np.int64), minlength=2).tolist(),
        "sample_weight_sha256": _array_sha256(weights64),
        "sample_weight_sum": float(weights64.sum()),
        "binary_weight_totals": [float(weights64[target == value].sum()) for value in (0.0, 1.0)],
        "weight_summary": _weight_summary(
            labels[training_indices], participants[training_indices], weights64
        ),
        "normalizer_mean": mean.tolist(),
        "normalizer_scale": scale.tolist(),
        "batch_orders_sha256": _array_sha256(order_array),
        "shared_initial_state_sha256": _state_sha256(shared_initial_state),
        "frozen_e2_checkpoint_sha256": e2_checkpoint_sha256,
        "training_rng": training_rng,
        "source_binding_reverified_at_completion": True,
        "final_state_sha256": _state_sha256(state_cpu),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": 80,
        "training_history": history,
        "final_whole_training_weighted_loss": final_numerator / final_denominator,
        "final_whole_training_weight_denominator": final_denominator,
        "fit_and_predict_seconds": fit_seconds,
        "peak_cuda_memory_allocated_bytes": peak_allocated,
        "checkpoint": {
            "path": checkpoint_path.relative_to(output_directory).as_posix(),
            "sha256": sha256_file(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
        },
        **dict(binding),
    }
    completed = _sealed(
        {"record_kind": "fog_motion_fit_attempt_completed", **metadata, "completed_at_utc": _now()}
    )
    _write_json_create_only(
        output_directory / "fit_attempts" / f"{attempt:02d}--completed.json", completed
    )
    return probabilities, metadata


def _ece(labels: IntArray, probabilities: FloatArray) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    total = labels.size
    result = 0.0
    for index in range(10):
        low, high = index / 10.0, (index + 1) / 10.0
        selected = (confidence >= low) & (
            (confidence <= high) if index == 9 else (confidence < high)
        )
        if selected.any():
            result += float(selected.sum() / total) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return result


def _binary_ece(target: BoolArray, motion_probability: FloatArray) -> float:
    probability = np.asarray(motion_probability, dtype=np.float64)
    predicted = probability >= 0.5
    confidence = np.maximum(probability, 1.0 - probability)
    correct = predicted == target
    result = 0.0
    for index in range(10):
        low, high = index / 10.0, (index + 1) / 10.0
        selected = (confidence >= low) & (
            (confidence <= high) if index == 9 else (confidence < high)
        )
        if selected.any():
            result += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return result


def extended_method_report(
    *,
    labels: IntArray,
    probabilities: FloatArray,
    participants: StringArray,
    roster: Sequence[str],
    conditional_posture_probabilities: FloatArray | None = None,
) -> dict[str, Any]:
    report = method_report(
        labels=labels, probabilities=probabilities, participant_ids=participants, roster=roster
    )
    target = np.asarray(labels == 0, dtype=np.bool_)
    motion = np.asarray(probabilities[:, 0], dtype=np.float64)
    predicted = motion >= 0.5
    sensitivity = float(predicted[target].mean()) if target.any() else None
    specificity = float((~predicted[~target]).mean()) if (~target).any() else None
    person_ece: list[float] = []
    balanced: list[float] = []
    aucs: list[float] = []
    for person in roster:
        selected = participants == person
        _require(bool(selected.any()), f"participant missing from report: {person}")
        person_ece.append(_ece(labels[selected], probabilities[selected]))
        person_target = target[selected]
        person_motion = motion[selected]
        if person_target.any() and (~person_target).any():
            person_predicted = person_motion >= 0.5
            balanced.append(
                0.5
                * (
                    float(person_predicted[person_target].mean())
                    + float((~person_predicted[~person_target]).mean())
                )
            )
            aucs.append(float(roc_auc_score(person_target, person_motion)))
    clipped = np.clip(motion, 1e-12, 1.0 - 1e-12)
    binary_nll = float(-np.mean(target * np.log(clipped) + (~target) * np.log(1.0 - clipped)))
    posture_probabilities = (
        probabilities
        if conditional_posture_probabilities is None
        else np.asarray(conditional_posture_probabilities, dtype=np.float64)
    )
    _require(
        posture_probabilities.shape == probabilities.shape,
        "conditional posture reference is not aligned",
    )
    stationary_mass = posture_probabilities[:, 1] + posture_probabilities[:, 2]
    posture_rows = (labels != 0) & (stationary_mass > 0.0)
    posture_target = labels[posture_rows] == 1
    posture_q = posture_probabilities[posture_rows, 1] / stationary_mass[posture_rows]
    pooled_posture_auc = (
        float(roc_auc_score(posture_target, posture_q))
        if posture_rows.any() and np.unique(posture_target).size == 2
        else None
    )
    posture_person_aucs: list[float] = []
    for person in roster:
        selected = posture_rows & (participants == person)
        person_target = labels[selected] == 1
        if selected.any() and np.unique(person_target).size == 2:
            person_q = posture_probabilities[selected, 1] / stationary_mass[selected]
            posture_person_aucs.append(float(roc_auc_score(person_target, person_q)))
    report["calibration"] = {
        "pooled_multiclass_ece_10": _ece(labels, probabilities),
        "mean_person_multiclass_ece_10": float(np.mean(person_ece)),
    }
    report["binary_motion"] = {
        "threshold": 0.5,
        "threshold_rule": "greater_than_or_equal_is_mobility",
        "pooled_sensitivity": sensitivity,
        "pooled_specificity": specificity,
        "eligible_person_count": len(balanced),
        "mean_person_balanced_accuracy": float(np.mean(balanced)),
        "pooled_auc": float(roc_auc_score(target, motion)),
        "mean_eligible_person_auc": float(np.mean(aucs)),
        "binary_nll": binary_nll,
        "binary_brier": float(np.mean((motion - target.astype(np.float64)) ** 2)),
        "binary_confidence_ece_10": _binary_ece(target, motion),
    }
    report["conditional_posture"] = {
        "sitting_positive": True,
        "probability_source": (
            "method_probabilities"
            if conditional_posture_probabilities is None
            else "provided_frozen_L9v_probabilities"
        ),
        "eligible_true_stationary_nonzero_q_rows": int(posture_rows.sum()),
        "eligible_person_count": len(posture_person_aucs),
        "pooled_auc": pooled_posture_auc,
        "mean_eligible_person_auc": (
            float(np.mean(posture_person_aucs)) if posture_person_aucs else None
        ),
    }
    return report


def raw_motion_report(
    *, labels: IntArray, motion_probability: FloatArray, participants: StringArray
) -> dict[str, Any]:
    """Report the learned binary head before q composition or fallback."""

    target = np.asarray(labels == 0, dtype=np.bool_)
    motion = np.asarray(motion_probability, dtype=np.float64)
    _require(
        labels.shape == motion.shape == participants.shape
        and bool(np.isfinite(motion).all())
        and bool(np.all((motion >= 0.0) & (motion <= 1.0))),
        "raw motion report arrays changed",
    )
    predicted = motion >= 0.5
    clipped = np.clip(motion, 1e-12, 1.0 - 1e-12)
    people_rows = []
    aucs = []
    balanced = []
    for person in sorted(np.unique(participants).tolist()):
        selected = participants == person
        person_target = target[selected]
        row: dict[str, Any] = {
            "participant_id": person,
            "row_count": int(selected.sum()),
            "mobility_rows": int(person_target.sum()),
            "stationary_rows": int((~person_target).sum()),
        }
        if person_target.any() and (~person_target).any():
            person_motion = motion[selected]
            person_predicted = predicted[selected]
            auc = float(roc_auc_score(person_target, person_motion))
            ba = 0.5 * (
                float(person_predicted[person_target].mean())
                + float((~person_predicted[~person_target]).mean())
            )
            row.update({"auc": auc, "balanced_accuracy": ba})
            aucs.append(auc)
            balanced.append(ba)
        else:
            row.update({"auc": None, "balanced_accuracy": None})
        people_rows.append(row)
    return {
        "row_count": int(labels.size),
        "class_counts_stationary_motion": [int((~target).sum()), int(target.sum())],
        "pooled_auc": float(roc_auc_score(target, motion)),
        "pooled_sensitivity": float(predicted[target].mean()),
        "pooled_specificity": float((~predicted[~target]).mean()),
        "eligible_person_count": len(aucs),
        "mean_eligible_person_auc": float(np.mean(aucs)),
        "mean_eligible_person_balanced_accuracy": float(np.mean(balanced)),
        "binary_nll": float(-np.mean(target * np.log(clipped) + (~target) * np.log(1.0 - clipped))),
        "binary_brier": float(np.mean((motion - target.astype(np.float64)) ** 2)),
        "binary_confidence_ece_10": _binary_ece(target, motion),
        "participants": people_rows,
    }


def _peak_working_set_bytes() -> int:
    memory = psutil.Process(os.getpid()).memory_info()
    return int(getattr(memory, "peak_wset", memory.rss))


def _gate(
    comparison: Mapping[str, Any],
    leave_folds: Sequence[Mapping[str, Any]],
    *,
    minimum_mean_gain: float,
    validation_complete: bool,
) -> dict[str, Any]:
    interval = cast(list[float], comparison["mean_difference_95_percent_bootstrap_interval"])
    recalls = _mapping(comparison["class_recall_differences"], "class recalls")
    checks = {
        "mean_gain": float(comparison["mean_difference"]) >= minimum_mean_gain,
        "positive_bootstrap_lower_bound": float(interval[0]) > 0.0,
        "participant_wins": int(comparison["participant_wins"]) >= 14,
        "bottom_seven": float(comparison["bottom_30_percent_difference"]) >= -0.010,
        "worst_person_minimum": float(comparison["worst_participant_difference"]) >= -0.030,
        "minimum_paired_person": float(comparison["minimum_paired_participant_difference"])
        >= -0.050,
        "mobility_recall": float(recalls["mobility"]) >= -0.010,
        "sitting_recall": float(recalls["sitting"]) >= -0.020,
        "standing_recall": float(recalls["standing"]) >= -0.020,
        "all_leave_one_person_positive": all(
            float(value) > 0.0
            for value in cast(list[float], comparison["leave_one_participant_out_mean_differences"])
        ),
        "all_leave_one_fold_positive": all(
            float(row["mean_participant_difference"]) > 0.0 for row in leave_folds
        ),
        "validation_complete": validation_complete,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "passed_check_count": sum(checks.values()),
        "required_check_count": len(checks),
        "minimum_mean_gain": minimum_mean_gain,
    }


def analyse(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    labels: IntArray,
    probabilities: Mapping[str, FloatArray],
    participants: StringArray,
    participant_folds: IntArray,
    roster: Sequence[str],
    validation_complete: bool,
) -> dict[str, Any]:
    _require(set(reports) == set(METHOD_ORDER), "report method set changed")
    stages = (
        ("practical", "T500", "l9v", 0.015),
        ("learned_signal", "T500", "E2", 0.010),
        ("additional_context", "T500", "T128", 0.010),
        ("coarse_history_order", "T500", "T500-P", 0.010),
    )
    comparisons: dict[str, Any] = {}
    leave_folds: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    for stage, candidate, comparator, threshold in stages:
        name = f"{candidate}_minus_{comparator}"
        comparisons[name] = paired_comparison(reports[candidate], reports[comparator])
        leave_folds[name] = _leave_one_fold(
            reports[candidate], reports[comparator], participant_folds
        )
        gates[stage] = {
            "comparison": name,
            **_gate(
                comparisons[name],
                leave_folds[name],
                minimum_mean_gain=threshold,
                validation_complete=validation_complete,
            ),
        }
    e2_name = "E2_minus_l9v"
    e2_comparison = paired_comparison(reports["E2"], reports["l9v"])
    e2_leave_folds = _leave_one_fold(reports["E2"], reports["l9v"], participant_folds)
    comparisons[e2_name] = e2_comparison
    leave_folds[e2_name] = e2_leave_folds
    diagnostic_gates = {
        "E2_minus_l9v_practical_guards": {
            "comparison": e2_name,
            "claim_sequence_member": False,
            **_gate(
                e2_comparison,
                e2_leave_folds,
                minimum_mean_gain=0.015,
                validation_complete=validation_complete,
            ),
        }
    }
    pairwise: dict[str, Any] = {}
    for left_index, left in enumerate(METHOD_ORDER):
        for right in METHOD_ORDER[left_index + 1 :]:
            name = f"{left}_minus_{right}"
            pairwise[name] = paired_comparison(reports[left], reports[right])
    claim_open = True
    sequence = []
    for stage, *_rest in stages:
        gate_status = gates[stage]["status"]
        eligible = claim_open
        sequence.append(
            {
                "stage": stage,
                "eligible": eligible,
                "gate_status": gate_status,
                "claim_status": gate_status if eligible else "blocked_by_prior_failure",
            }
        )
        if gate_status != "pass":
            claim_open = False
    events = {
        f"{cell}_minus_l9v": _events(
            labels, probabilities["l9v"], probabilities[cell], participants, roster
        )
        for cell in CELL_ORDER
    }
    return {
        "record_kind": "fog_motion_factorization_analysis",
        "status": "complete_analysis" if validation_complete else "analysis_awaiting_replay",
        "evidence_status": "corrected_fog_adaptive_development_not_confirmation",
        "reports": dict(reports),
        "primary_comparisons": comparisons,
        "all_pairwise_comparisons": pairwise,
        "leave_one_outer_fold_sensitivity": leave_folds,
        "gates": gates,
        "diagnostic_gates": diagnostic_gates,
        "sequential_claims": sequence,
        "advancement": {
            "status": "pass" if all(row["claim_status"] == "pass" for row in sequence) else "fail",
            "primary_candidate": "T500",
        },
        "decision": (
            "pass_all_four_claim_stages_freeze_T500_no_automatic_follow_on"
            if all(row["claim_status"] == "pass" for row in sequence)
            else "fail_full_claim_sequence_retain_diagnostic_lessons"
        ),
        "event_topology": events,
        "bootstrap_scope": (
            "descriptive participant resampling conditional on overlapping fitted folds, "
            "adaptive hypothesis choice, and a consumed development cohort"
        ),
        "automatic_follow_on_launched": False,
        "additional_seed_launched": False,
        "confirmation_claimed": False,
        "novelty_claimed": False,
    }


def _method_summary(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in METHOD_ORDER:
        primary = _mapping(reports[method]["primary"], f"{method} primary")
        pooled = _mapping(reports[method]["pooled"], f"{method} pooled")
        binary = _mapping(reports[method]["binary_motion"], f"{method} binary")
        result[method] = {
            "mean_participant_macro_f1": primary["mean_participant_macro_f1"],
            "bottom_seven_participant_macro_f1": primary["bottom_30_percent_participant_macro_f1"],
            "worst_participant_macro_f1": primary["worst_participant_macro_f1"],
            "pooled_accuracy": pooled["accuracy"],
            "mean_participant_nll": primary["mean_participant_nll"],
            "mean_participant_brier": primary["mean_participant_multiclass_brier"],
            "pooled_motion_auc": binary["pooled_auc"],
            "mean_person_motion_balanced_accuracy": binary["mean_person_balanced_accuracy"],
        }
    return result


def _outcome_summary(analysis: Mapping[str, Any]) -> str:
    reports = _mapping(analysis["reports"], "reports")
    lines = [
        "# Corrected FoG motion-factorization outcome",
        "",
        f"Decision: **{analysis['decision']}**.",
        "",
        "| Method | Mean person F1 | Accuracy | Bottom seven | Worst | Motion AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        report = _mapping(reports[method], method)
        primary = _mapping(report["primary"], "primary")
        pooled = _mapping(report["pooled"], "pooled")
        binary = _mapping(report["binary_motion"], "binary")
        lines.append(
            f"| {method} | {float(primary['mean_participant_macro_f1']):.9f} | "
            f"{float(pooled['accuracy']):.9f} | "
            f"{float(primary['bottom_30_percent_participant_macro_f1']):.9f} | "
            f"{float(primary['worst_participant_macro_f1']):.9f} | "
            f"{float(binary['pooled_auc']):.9f} |"
        )
    lines += [
        "",
        "| Claim stage | Comparison | Mean delta | 95% interval | W/H/T | Gate |",
        "|---|---|---:|---:|---:|---|",
    ]
    comparisons = _mapping(analysis["primary_comparisons"], "comparisons")
    gates = _mapping(analysis["gates"], "gates")
    sequence = {
        str(row["stage"]): str(row["claim_status"])
        for row in cast(list[dict[str, Any]], analysis["sequential_claims"])
    }
    for stage in ("practical", "learned_signal", "additional_context", "coarse_history_order"):
        gate = _mapping(gates[stage], stage)
        comparison = _mapping(comparisons[str(gate["comparison"])], "comparison")
        interval = cast(list[float], comparison["mean_difference_95_percent_bootstrap_interval"])
        lines.append(
            f"| {stage} | {gate['comparison']} | {float(comparison['mean_difference']):+.9f} | "
            f"[{float(interval[0]):+.9f}, {float(interval[1]):+.9f}] | "
            f"{comparison['participant_wins']}/{comparison['participant_harms']}/"
            f"{comparison['participant_ties']} | {sequence[stage]} |"
        )
    diagnostic = _mapping(
        _mapping(analysis["diagnostic_gates"], "diagnostic gates")["E2_minus_l9v_practical_guards"],
        "E2 diagnostic",
    )
    diagnostic_comparison = _mapping(comparisons["E2_minus_l9v"], "E2 comparison")
    lines += [
        "",
        "E2-minus-L9v practical-guard diagnostic: "
        f"{diagnostic['status']} (mean delta "
        f"{float(diagnostic_comparison['mean_difference']):+.9f}).",
    ]
    lines += [
        "",
        "This is adaptive development evidence on 22 previously consumed people.",
        "All 1,213 scored rows remain through exact L9v/B0 fallback. No extra seed,",
        "new dataset, automatic follow-on, confirmation, novelty, publication or release",
        "action was launched.",
        "",
    ]
    return "\n".join(lines)


def _artifact_manifest(output_directory: Path) -> dict[str, Any]:
    omitted = {
        "artifact_manifest.json",
        "validation.json",
        "validation_worker_shutdown.json",
        "completion_manifest.json",
        "controller_final_receipt.json",
        "VALIDATION_INCOMPLETE.json",
        "analysis.json",
        "result.json",
        "OUTCOME_SUMMARY.md",
    }
    files = sorted(
        path for path in output_directory.rglob("*") if path.is_file() and path.name not in omitted
    )
    return _sealed(
        {
            "record_kind": "fog_motion_factorization_artifact_manifest",
            "artifacts": [
                {
                    "path": path.relative_to(output_directory).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in files
            ],
        }
    )


def _verify_artifact_manifest(output_directory: Path) -> int:
    manifest = _read_json(output_directory / "artifact_manifest.json")
    _verify_sealed(manifest, "motion artifact manifest")
    rows = cast(list[dict[str, Any]], manifest["artifacts"])
    listed = {str(row["path"]) for row in rows}
    _require(len(rows) == len(listed), "duplicate artifact path")
    for row in rows:
        path = (output_directory / str(row["path"])).resolve()
        _require(path.is_relative_to(output_directory), "artifact path escapes run")
        _require(
            path.is_file()
            and path.stat().st_size == int(row["size_bytes"])
            and sha256_file(path) == row["sha256"],
            f"artifact changed: {path}",
        )
    allowed = {
        "artifact_manifest.json",
        "validation.json",
        "validation_worker_shutdown.json",
        "completion_manifest.json",
        "controller_final_receipt.json",
        "VALIDATION_INCOMPLETE.json",
        "analysis.json",
        "result.json",
        "OUTCOME_SUMMARY.md",
    }
    actual = {
        path.relative_to(output_directory).as_posix()
        for path in output_directory.rglob("*")
        if path.is_file() and path.name not in allowed
    }
    _require(actual == listed, "artifact manifest coverage changed")
    return len(rows)


def _roster(config: Mapping[str, Any]) -> list[str]:
    held_out = _mapping(_mapping(config["folds"], "folds")["held_out_participants"], "held out")
    result = sorted(
        str(person) for people in held_out.values() for person in cast(list[Any], people)
    )
    _require(result == [f"fogstar:{index:03d}" for index in range(1, 23)], "roster changed")
    return result


def _snapshot_inputs(
    repository_root: Path,
    config_path: Path,
    protocol_path: Path,
    output_directory: Path,
) -> None:
    _write_bytes_create_only(output_directory / "config_snapshot.yaml", config_path.read_bytes())
    for name, path in (
        ("protocol_snapshot.json", protocol_path),
        ("specification_snapshot.json", repository_root / SPEC_RELATIVE),
    ):
        _write_json_create_only(
            output_directory / name,
            _sealed(
                {
                    "record_kind": f"fog_motion_{name.removesuffix('.json')}",
                    "source_path": path.relative_to(repository_root).as_posix(),
                    "source_sha256": sha256_file(path),
                    "text": path.read_text(encoding="utf-8"),
                }
            ),
        )


def _cache_metadata(arrays: Mapping[str, NDArray[Any]]) -> dict[str, Any]:
    return _sealed(
        {
            "record_kind": "fog_motion_context_cache_metadata",
            "arrays": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": _array_sha256(value),
                }
                for name, value in arrays.items()
            },
        }
    )


def _record_fit_failure(
    output_directory: Path,
    *,
    attempt: int,
    cell_id: str,
    fold: int,
    error: BaseException,
    binding: Mapping[str, Any],
) -> None:
    path = output_directory / "fit_attempts" / f"{attempt:02d}--failed.json"
    if path.exists():
        return
    _write_json_create_only(
        path,
        _sealed(
            {
                "record_kind": "fog_motion_fit_attempt_failed",
                "attempt_number": attempt,
                "cell_id": cell_id,
                "outer_fold": fold,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "failed_at_utc": _now(),
                "retry_launched": False,
                **dict(binding),
            }
        ),
    )


def run_experiment(
    *,
    repository_root: Path,
    evidence_root: Path,
    config_path: Path,
    protocol_path: Path,
    output_directory: Path,
    code_commit: str,
    cancel_file: Path | None,
) -> dict[str, Any]:
    """Run exactly twenty fixed fits and leave final claims pending independent replay."""

    baseline_workers = _worker_baseline()
    attempts = 0
    completed = 0
    started = time.perf_counter()
    output_directory = output_directory.resolve()
    fit_rows: list[dict[str, Any]] = []
    output_owned = False
    try:
        repository_root = repository_root.resolve()
        evidence_root = evidence_root.resolve()
        config_path = config_path.resolve()
        protocol_path = protocol_path.resolve()
        _require(config_path == repository_root / CONFIG_RELATIVE, "config path changed")
        _require(protocol_path == repository_root / PROTOCOL_RELATIVE, "protocol path changed")
        _require(
            output_directory.parent == evidence_root / EVIDENCE_FAMILY
            and output_directory.name == RUN_DIRECTORY_NAME,
            "output location changed",
        )
        _require(not output_directory.exists(), "create-only run directory exists")
        output_directory.mkdir(parents=True)
        output_owned = True
        config = _read_yaml(config_path)
        validate_config(config, protocol_path=protocol_path)
        _require(
            (repository_root / SPEC_RELATIVE).is_file()
            and sha256_file(repository_root / SPEC_RELATIVE) == SPEC_SHA256,
            "governing specification changed",
        )
        git = _git_state(repository_root)
        _require(git["clean"] is True and git["commit"] == code_commit, "source must be clean")
        _configure_runtime(11)
        environment = _require_environment(config)
        _require(
            Path(str(environment["implementation_file"])).resolve()
            == repository_root
            / "src/inclusive_shift_har/experiments/fog_motion_factorization_run.py"
            and Path(str(environment["motion_context_implementation_file"])).resolve()
            == repository_root / "src/inclusive_shift_har/experiments/fog_motion_factorization.py",
            "imported implementation path does not match source manifest root",
        )
        _snapshot_inputs(repository_root, config_path, protocol_path, output_directory)
        source_manifest = _sealed(_source_manifest(repository_root))
        _write_json_create_only(output_directory / "source_manifest.json", source_manifest)
        _write_json_create_only(
            output_directory / "environment.json",
            _sealed(
                {
                    "record_kind": "fog_motion_environment_receipt",
                    **environment,
                }
            ),
        )
        _write_json_create_only(
            output_directory / "command_receipt.json",
            _sealed(
                {
                    "record_kind": "fog_motion_command_receipt",
                    "argv": list(sys.argv),
                    "working_directory": str(Path.cwd()),
                    "repository_root": str(repository_root),
                    "evidence_root": str(evidence_root),
                    "output_directory": str(output_directory),
                    "controller_process_id": os.getpid(),
                    "controller_count": 1,
                    "launched_at_utc": _now(),
                }
            ),
        )

        spatial_run, spatial_receipt = _verify_spatial_reference(evidence_root)
        inputs = _input_paths(spatial_run)
        _require(
            inputs["raw_source"]
            == (evidence_root / str(_mapping(config["source"], "source")["raw_path"])).resolve(),
            "raw source path changed",
        )
        context_preflight = preflight_fog_motion_context(
            raw_path=inputs["raw_source"],
            coverage_path=inputs["coverage"],
            spatial_cache_path=spatial_run / "feature_cache.npz",
        )
        contexts = context_preflight.contexts
        _require(
            context_preflight.receipt["record_sha256"]
            == ("f6d956ce3f175d5578a82a496c7ef82ccaf1df6acf530553c96ad51b985db920"),
            "context preflight receipt changed",
        )
        context_arrays: dict[str, NDArray[Any]] = {
            "signals": contexts.signals,
            "derived_gravity": contexts.derived_gravity,
            "timestamps": contexts.timestamps,
            "observation_mask": contexts.observation_mask,
            "current_availability_mask": contexts.current_availability_mask,
            "full_context_mask": contexts.full_context_mask,
            "observable_window_ids": contexts.observable_window_ids,
        }
        _write_npz_create_only(output_directory / "context_cache.npz", **context_arrays)
        context_metadata = _cache_metadata(context_arrays)
        _write_json_create_only(output_directory / "context_cache_metadata.json", context_metadata)
        _verify_sealed(context_preflight.receipt, "context preflight")
        _write_json_create_only(
            output_directory / "context_preflight.json", context_preflight.receipt
        )
        _write_json_create_only(
            output_directory / "context_alignment_receipts.json",
            _sealed(
                {
                    "record_kind": "fog_motion_context_alignment_receipts",
                    "alignment_receipts": list(contexts.alignment_receipts),
                    "segment_receipts": list(contexts.segment_receipts),
                }
            ),
        )
        # Only after context support is frozen may labels and OOF outcome probabilities be loaded.
        references, reference_receipt = _verify_reference(evidence_root, config)
        reference_receipt["spatial_reference_receipt"] = spatial_receipt
        _write_json_create_only(
            output_directory / "reference_receipt.json", _sealed(reference_receipt)
        )
        _write_json_create_only(
            output_directory / "mandatory_l9v_fallback_erratum_snapshot.json",
            _sealed(
                {
                    "record_kind": "fog_motion_l9v_erratum_snapshot",
                    "source": _mapping(reference_receipt["erratum"], "erratum"),
                }
            ),
        )
        input_manifest = _sealed(
            {
                "record_kind": "fog_motion_input_manifest",
                "raw_source": {
                    "path": str(inputs["raw_source"]),
                    "sha256": sha256_file(inputs["raw_source"]),
                    "size_bytes": inputs["raw_source"].stat().st_size,
                },
                "coverage": {
                    "path": str(inputs["coverage"]),
                    "sha256": sha256_file(inputs["coverage"]),
                },
                "availability": {
                    "path": str(inputs["availability"]),
                    "sha256": sha256_file(inputs["availability"]),
                },
                "reference_run": str(evidence_root / REFERENCE_RELATIVE),
                "context_record_sha256": context_preflight.receipt["record_sha256"],
            }
        )
        _write_json_create_only(output_directory / "input_manifest.json", input_manifest)

        labels_scored = np.asarray(context_preflight.scored_labels, dtype=np.int64)
        scoring_indices = np.asarray(context_preflight.scoring_indices, dtype=np.int64)
        scoring = np.asarray(context_preflight.scoring_eligibility, dtype=np.bool_)
        people = np.asarray(context_preflight.observable_participant_ids, dtype=np.str_)
        folds = np.asarray(context_preflight.observable_fold_index, dtype=np.int64)
        observable_labels = np.zeros(people.size, dtype=np.int64)
        observable_labels[scoring_indices] = labels_scored
        full_indices = np.flatnonzero(contexts.full_context_mask).astype(np.int64)
        _require(
            full_indices.size == 1724
            and int(contexts.full_context_mask[scoring_indices].sum()) == 1098,
            "common support changed",
        )
        baseline_stack = np.asarray(
            references["reference_observable_probabilities"], dtype=np.float64
        )
        _require(baseline_stack.shape == (5, 1939, 3), "reference probability shape changed")
        controls = {name: baseline_stack[index] for index, name in enumerate(REFERENCE_ORDER)}
        _require(
            np.array_equal(
                references["reference_observable_window_ids"], contexts.observable_window_ids
            )
            and np.array_equal(references["observable_window_ids"], contexts.observable_window_ids)
            and np.array_equal(references["reference_observable_participant_ids"], people)
            and np.array_equal(references["observable_participant_ids"], people)
            and np.array_equal(references["reference_observable_fold_index"], folds)
            and np.array_equal(references["observable_fold_index"], folds)
            and np.array_equal(references["reference_scoring_indices"], scoring_indices)
            and np.array_equal(references["reference_scored_labels"], labels_scored)
            and np.array_equal(
                references["reference_availability_mask"], contexts.current_availability_mask
            )
            and np.array_equal(references["availability_mask"], contexts.current_availability_mask),
            "reference/context provenance alignment changed",
        )
        _require(
            np.array_equal(
                references["reference_scored_probabilities"], baseline_stack[:, scoring_indices]
            )
            and np.array_equal(
                references["reference_scored_participant_ids"], people[scoring_indices]
            )
            and np.array_equal(
                references["reference_scored_window_ids"],
                contexts.observable_window_ids[scoring_indices],
            )
            and np.array_equal(references["reference_scored_fold_index"], folds[scoring_indices]),
            "reference scoring projection changed",
        )
        _require(
            np.array_equal(
                controls["l9v"][~contexts.current_availability_mask],
                controls["b0"][~contexts.current_availability_mask],
            ),
            "L9v/B0 fallback changed",
        )
        energy = current_energy_features(contexts.signals)
        l9v_names = references["l9v_names"].tolist()
        available_indices = np.asarray(references["available_indices"], dtype=np.int64)
        available_lookup = np.full(1939, -1, dtype=np.int64)
        available_lookup[available_indices] = np.arange(available_indices.size)
        for column, name in enumerate(("accelerometer_norm__rms", "gyroscope_norm__rms")):
            cached = np.asarray(references["l9v_values"], dtype=np.float64)[
                available_lookup[full_indices], l9v_names.index(name)
            ]
            _require(
                np.array_equal(energy[full_indices, column], np.log(np.maximum(cached, 1e-8))),
                f"E2 feature replay changed: {name}",
            )
        roster = _roster(config)
        retained_analysis = _read_json(evidence_root / REFERENCE_RELATIVE / "analysis.json")
        retained_l9v = _mapping(
            _mapping(retained_analysis["reports"], "retained reports")["l9v"], "retained L9v report"
        )
        replayed_l9v = method_report(
            labels=labels_scored,
            probabilities=controls["l9v"][scoring_indices],
            participant_ids=people[scoring_indices],
            roster=roster,
        )
        _require(
            canonical_json_sha256(replayed_l9v) == canonical_json_sha256(retained_l9v),
            "inherited L9v metric parity changed",
        )
        _write_json_create_only(
            output_directory / "l9v_metric_parity.json",
            _sealed(
                {
                    "record_kind": "fog_motion_l9v_metric_parity",
                    "status": "pass_before_first_fit",
                    "replayed_report_sha256": canonical_json_sha256(replayed_l9v),
                    "retained_report_sha256": canonical_json_sha256(retained_l9v),
                }
            ),
        )

        permutations = np.stack(
            [history_block_permutation(str(value)) for value in contexts.observable_window_ids]
        )
        _write_npz_create_only(
            output_directory / "history_permutations.npz",
            observable_window_ids=contexts.observable_window_ids,
            permutations=permutations,
            full_context_mask=contexts.full_context_mask,
        )
        identity_count = int(np.all(permutations == np.arange(3), axis=1).sum())
        _write_json_create_only(
            output_directory / "permutation_receipt.json",
            _sealed(
                {
                    "record_kind": "fog_motion_history_permutation_receipt",
                    "payload": "motion-history-v1|{observable_window_id}|{block_index}",
                    "permutation_sha256": _array_sha256(permutations),
                    "identity_count_all_observable": identity_count,
                    "identity_count_full_context": int(
                        np.all(permutations[full_indices] == np.arange(3), axis=1).sum()
                    ),
                    "current_query_unchanged": True,
                }
            ),
        )
        partition_rows = []
        for fold in OUTER_FOLDS:
            train = np.flatnonzero(scoring & contexts.full_context_mask & (folds != fold))
            predict = np.flatnonzero(contexts.full_context_mask & (folds == fold))
            partition_rows.append(
                {
                    "outer_fold": fold,
                    "training_indices": train.tolist(),
                    "prediction_indices": predict.tolist(),
                    "training_indices_sha256": _array_sha256(train),
                    "prediction_indices_sha256": _array_sha256(predict),
                    "training_rows": int(train.size),
                    "prediction_candidates": int(predict.size),
                    "training_original_class_counts": np.bincount(
                        observable_labels[train], minlength=3
                    ).tolist(),
                    "training_participants": sorted(np.unique(people[train]).tolist()),
                    "held_out_participants": sorted(np.unique(people[predict]).tolist()),
                }
            )
        _require(
            [row["training_rows"] for row in partition_rows] == [904, 780, 902, 780, 1026],
            "training support changed",
        )
        partition_record = _sealed(
            {
                "record_kind": "fog_motion_partition_preflight",
                "rows": partition_rows,
                "participant_partition_before_windowing_inherited": True,
            }
        )
        _write_json_create_only(output_directory / "partition_preflight.json", partition_record)
        preflight = _sealed(
            {
                "record_kind": "fog_motion_preflight",
                "status": "pass_before_first_fit",
                "created_before_model_fits": True,
                "repository_root": str(repository_root),
                "evidence_root": str(evidence_root),
                "code_commit": code_commit,
                "git": git,
                "environment": environment,
                "source_manifest_record_sha256": source_manifest["record_sha256"],
                "context_preflight_record_sha256": context_preflight.receipt["record_sha256"],
                "context_cache_record_sha256": context_metadata["record_sha256"],
                "partition_record_sha256": partition_record["record_sha256"],
                "config_sha256": sha256_file(config_path),
                "config_semantic_sha256": _canonical_config_sha256(config),
                "protocol_sha256": sha256_file(protocol_path),
                "specification_sha256": SPEC_SHA256,
                "expected_fit_attempts": 20,
                "context_frozen_before_outcomes_loaded": True,
                "zero_lookahead_streaming_claim": False,
                "automatic_follow_on_allowed": False,
            }
        )
        _write_json_create_only(output_directory / "preflight.json", preflight)

        device = torch.device("cuda:0")
        raw_motion = {cell: np.full(1939, np.nan, dtype=np.float64) for cell in CELL_ORDER}
        batch_order_arrays: dict[str, NDArray[Any]] = {}
        fold_contract_arrays: dict[str, NDArray[Any]] = {}
        initial_state_rows: list[dict[str, Any]] = []
        initial_states: dict[int, dict[str, Tensor]] = {}
        for fold in OUTER_FOLDS:
            train = np.asarray(partition_rows[fold]["training_indices"], dtype=np.int64)
            predict = np.asarray(partition_rows[fold]["prediction_indices"], dtype=np.int64)
            weights = participant_class_weights(observable_labels[train], people[train])
            temporal_mean, temporal_scale = _standardize_fit(
                contexts.signals[train, -128:].reshape(-1, 6)
            )
            e2_mean, e2_scale = _standardize_fit(energy[train])
            orders = np.stack(_epoch_orders(train.size, 11 + fold))
            batch_order_arrays[f"fold_{fold}"] = orders
            fold_contract_arrays[f"training_indices_fold_{fold}"] = train
            fold_contract_arrays[f"prediction_indices_fold_{fold}"] = predict
            fold_contract_arrays[f"weights_fold_{fold}"] = weights
            fold_contract_arrays[f"temporal_mean_fold_{fold}"] = temporal_mean
            fold_contract_arrays[f"temporal_scale_fold_{fold}"] = temporal_scale
            fold_contract_arrays[f"e2_mean_fold_{fold}"] = e2_mean
            fold_contract_arrays[f"e2_scale_fold_{fold}"] = e2_scale
            _configure_determinism(11 + fold)
            template = MotionResidualTCN().cpu()
            state = {
                name: value.detach().cpu().clone() for name, value in template.state_dict().items()
            }
            _require(
                sum(parameter.numel() for parameter in template.parameters())
                == EXPECTED_PARAMETER_COUNT,
                "initial parameter count changed",
            )
            initial_states[fold] = state
            initial_path = output_directory / "initial_states" / f"fold-{fold}.pt"
            _write_torch_create_only(
                initial_path,
                {
                    "outer_fold": fold,
                    "state_dict": state,
                    "config_semantic_sha256": CONFIG_SEMANTIC_SHA256,
                },
            )
            initial_state_rows.append(
                {
                    "outer_fold": fold,
                    "sha256": sha256_file(initial_path),
                    "state_sha256": _state_sha256(state),
                    "size_bytes": initial_path.stat().st_size,
                }
            )
        _write_npz_create_only(output_directory / "batch_orders.npz", **batch_order_arrays)
        _write_npz_create_only(
            output_directory / "fold_training_contract.npz", **fold_contract_arrays
        )
        _write_json_create_only(
            output_directory / "initial_state_receipt.json",
            _sealed(
                {
                    "record_kind": "fog_motion_initial_state_receipt",
                    "created_before_first_fit": True,
                    "rows": initial_state_rows,
                }
            ),
        )
        for fold in OUTER_FOLDS:
            if cancel_file is not None and cancel_file.exists():
                raise KeyboardInterrupt(f"cooperative cancellation requested before fold {fold}")
            train = np.asarray(partition_rows[fold]["training_indices"], dtype=np.int64)
            predict = np.asarray(partition_rows[fold]["prediction_indices"], dtype=np.int64)
            binding = _fit_binding(
                repository_root,
                code_commit=code_commit,
                source_manifest_record_sha256=str(source_manifest["record_sha256"]),
                input_manifest_record_sha256=str(input_manifest["record_sha256"]),
            )
            attempts += 1
            try:
                e2_predicted, e2_metadata, e2_state = _fit_e2(
                    features=energy,
                    labels=observable_labels,
                    participants=people,
                    training_indices=train,
                    prediction_indices=predict,
                    fold=fold,
                    attempt=attempts,
                    output_directory=output_directory,
                    binding=binding,
                    repository_root=repository_root,
                )
            except BaseException as exc:
                _record_fit_failure(
                    output_directory,
                    attempt=attempts,
                    cell_id="E2",
                    fold=fold,
                    error=exc,
                    binding=binding,
                )
                raise
            completed += 1
            fit_rows.append(e2_metadata)
            raw_motion["E2"][predict] = e2_predicted
            estimator = cast(LogisticRegression, e2_state["estimator"])
            with threadpool_limits(limits=1):
                e2_all = np.asarray(
                    estimator.decision_function(
                        _standardize(
                            energy,
                            cast(FloatArray, e2_state["mean"]),
                            cast(FloatArray, e2_state["scale"]),
                        )
                    ),
                    dtype=np.float64,
                )
            initial_state = initial_states[fold]
            e2_checkpoint_sha256 = str(
                _mapping(e2_metadata["checkpoint"], "E2 checkpoint")["sha256"]
            )
            for cell in CELL_ORDER[1:]:
                if cancel_file is not None and cancel_file.exists():
                    raise KeyboardInterrupt(
                        f"cooperative cancellation requested before {cell} fold {fold}"
                    )
                binding = _fit_binding(
                    repository_root,
                    code_commit=code_commit,
                    source_manifest_record_sha256=str(source_manifest["record_sha256"]),
                    input_manifest_record_sha256=str(input_manifest["record_sha256"]),
                )
                attempts += 1
                try:
                    values, metadata = _fit_temporal(
                        cell_id=cell,
                        signals=contexts.signals,
                        observation_mask=contexts.observation_mask,
                        window_ids=contexts.observable_window_ids,
                        labels=observable_labels,
                        participants=people,
                        training_indices=train,
                        prediction_indices=predict,
                        e2_logits_all=e2_all,
                        fold=fold,
                        attempt=attempts,
                        shared_initial_state=initial_state,
                        output_directory=output_directory,
                        device=device,
                        binding=binding,
                        e2_checkpoint_sha256=e2_checkpoint_sha256,
                        cancel_file=cancel_file,
                        repository_root=repository_root,
                    )
                except BaseException as exc:
                    _record_fit_failure(
                        output_directory,
                        attempt=attempts,
                        cell_id=cell,
                        fold=fold,
                        error=exc,
                        binding=binding,
                    )
                    raise
                completed += 1
                fit_rows.append(metadata)
                raw_motion[cell][predict] = values
        _require(attempts == completed == 20, "twenty-fit family incomplete")
        for cell in CELL_ORDER:
            _require(
                bool(np.isfinite(raw_motion[cell][full_indices]).all()),
                f"{cell} full-context prediction incomplete",
            )
            _require(
                bool(np.isnan(raw_motion[cell][~contexts.full_context_mask]).all()),
                f"{cell} predicted outside intervention support",
            )
        probabilities: dict[str, FloatArray] = {}
        effective_motion: dict[str, FloatArray] = {}
        for cell in CELL_ORDER:
            motion = controls["l9v"][:, 0].copy()
            motion[full_indices] = raw_motion[cell][full_indices]
            probabilities[cell] = compose_motion_probability(
                motion, controls["l9v"], contexts.full_context_mask, controls["b0"]
            )
            effective_motion[cell] = probabilities[cell][:, 0].copy()
            _require(
                np.array_equal(
                    probabilities[cell][~contexts.full_context_mask],
                    controls["l9v"][~contexts.full_context_mask],
                ),
                f"{cell} fallback changed",
            )
        probabilities.update(controls)
        scored_people = people[scoring_indices]
        scored_folds = folds[scoring_indices]
        scored_probabilities = {
            method: value[scoring_indices] for method, value in probabilities.items()
        }
        reports = {
            method: extended_method_report(
                labels=labels_scored,
                probabilities=scored_probabilities[method],
                participants=scored_people,
                roster=roster,
                conditional_posture_probabilities=(
                    scored_probabilities["l9v"] if method in CELL_ORDER else None
                ),
            )
            for method in METHOD_ORDER
        }
        participant_folds = np.asarray(
            [int(np.unique(scored_folds[scored_people == person])[0]) for person in roster],
            dtype=np.int64,
        )
        prevalidation_analysis = analyse(
            reports,
            labels=labels_scored,
            probabilities=scored_probabilities,
            participants=scored_people,
            participant_folds=participant_folds,
            roster=roster,
            validation_complete=False,
        )
        _write_json_create_only(
            output_directory / "analysis_prevalidation.json", _sealed(prevalidation_analysis)
        )
        scored_full = contexts.full_context_mask[scoring_indices]
        scored_current = contexts.current_availability_mask[scoring_indices]
        observable_q_zero = contexts.full_context_mask & (controls["l9v"][:, 1:].sum(axis=1) == 0.0)
        scored_q_zero = observable_q_zero[scoring_indices]
        strata = {
            "scored_full_context_qualified": int(scored_full.sum()),
            "scored_full_context_intervention": int((scored_full & ~scored_q_zero).sum()),
            "scored_full_context_q_zero_l9v_fallback": int(scored_q_zero.sum()),
            "current_ankle_short_history_l9v_fallback": int((scored_current & ~scored_full).sum()),
            "missing_ankle_b0_fallback": int((~scored_current).sum()),
            "observable_full_context_qualified": int(contexts.full_context_mask.sum()),
            "observable_full_context_q_zero_l9v_fallback": int(observable_q_zero.sum()),
            "P017_full_context_scored": int((scored_full & (scored_people == "fogstar:017")).sum()),
        }
        full_scored_global = scoring_indices[scored_full]
        raw_diagnostics = {
            cell: raw_motion_report(
                labels=labels_scored[scored_full],
                motion_probability=raw_motion[cell][full_scored_global],
                participants=scored_people[scored_full],
            )
            for cell in CELL_ORDER
        }
        _write_json_create_only(
            output_directory / "raw_motion_diagnostics.json",
            _sealed(
                {
                    "record_kind": "fog_motion_raw_learner_diagnostics",
                    "support": "all_1098_scored_full_context_rows_before_q_zero_fallback",
                    "methods": raw_diagnostics,
                }
            ),
        )
        _write_json_create_only(
            output_directory / "participant_metrics.json",
            _sealed(
                {
                    "record_kind": "fog_motion_participant_metrics",
                    "methods": reports,
                    "strata": strata,
                }
            ),
        )
        _write_json_create_only(
            output_directory / "fit_reports.json",
            _sealed(
                {
                    "record_kind": "fog_motion_fit_reports",
                    "fit_attempt_count": attempts,
                    "completed_fit_count": completed,
                    "rows": fit_rows,
                }
            ),
        )
        probability_stack = np.stack([probabilities[name] for name in METHOD_ORDER])
        _write_npz_create_only(
            output_directory / "predictions.npz",
            method_ids=np.asarray(METHOD_ORDER, dtype=np.str_),
            observable_probabilities=probability_stack,
            scored_probabilities=probability_stack[:, scoring_indices],
            raw_full_context_motion_probabilities=np.stack(
                [raw_motion[cell][full_indices] for cell in CELL_ORDER]
            ),
            effective_observable_motion_probabilities=np.stack(
                [effective_motion[cell] for cell in CELL_ORDER]
            ),
            full_context_indices=full_indices,
            full_context_mask=contexts.full_context_mask,
            current_availability_mask=contexts.current_availability_mask,
            scoring_indices=scoring_indices,
            scored_labels=labels_scored,
            observable_window_ids=contexts.observable_window_ids,
            observable_participant_ids=people,
            observable_fold_index=folds,
            scored_participant_ids=scored_people,
            scored_window_ids=contexts.observable_window_ids[scoring_indices],
            scored_fold_index=scored_folds,
            observable_decisions=probability_stack.argmax(axis=2).astype(np.int64),
            scored_decisions=probability_stack[:, scoring_indices].argmax(axis=2).astype(np.int64),
        )
        _write_json_create_only(
            output_directory / "result_prevalidation.json",
            _sealed(
                {
                    "record_kind": "fog_motion_result_prevalidation",
                    "status": "twenty_fits_complete_awaiting_independent_replay",
                    "experiment_id": EXPERIMENT_ID,
                    "code_commit": code_commit,
                    "fit_attempt_count": attempts,
                    "completed_fit_count": completed,
                    "method_summary": _method_summary(reports),
                    "scientific_checks_before_validation": {
                        stage: {
                            **dict(gate),
                            "checks": {
                                **dict(_mapping(gate["checks"], "checks")),
                                "validation_complete": "pending",
                            },
                        }
                        for stage, gate in _mapping(
                            prevalidation_analysis["gates"], "gates"
                        ).items()
                    },
                    "automatic_follow_on_launched": False,
                    "additional_seed_launched": False,
                    "InclusiveHAR_P11_P20_loaded": False,
                }
            ),
        )
        completion_binding = _fit_binding(
            repository_root,
            code_commit=code_commit,
            source_manifest_record_sha256=str(source_manifest["record_sha256"]),
            input_manifest_record_sha256=str(input_manifest["record_sha256"]),
        )
        _write_json_create_only(
            output_directory / "completion_source_binding.json",
            _sealed(
                {
                    "record_kind": "fog_motion_completion_source_binding",
                    "status": "unchanged_after_twenty_fits",
                    **completion_binding,
                }
            ),
        )
        worker = _stop_task_owned_workers(
            baseline_workers, attempts, completed, terminal="fit_phase_complete"
        )
        _require(
            worker["task_owned_fit_workers_and_monitors_stopped"] is True,
            "task-owned fit worker remains",
        )
        _write_json_create_only(output_directory / "worker_shutdown.json", worker)
        elapsed = time.perf_counter() - started
        _write_json_create_only(
            output_directory / "runtime.json",
            _sealed(
                {
                    "record_kind": "fog_motion_runtime",
                    "total_run_seconds": elapsed,
                    "fit_attempt_count": attempts,
                    "completed_fit_count": completed,
                    "fit_and_predict_seconds": sum(
                        float(row["fit_and_predict_seconds"]) for row in fit_rows
                    ),
                    "peak_cuda_memory_allocated_bytes": max(
                        int(row.get("peak_cuda_memory_allocated_bytes", 0)) for row in fit_rows
                    ),
                    "peak_process_working_set_bytes": _peak_working_set_bytes(),
                    "checkpoint_file_bytes": sum(
                        path.stat().st_size
                        for path in (output_directory / "checkpoints").iterdir()
                        if path.is_file()
                    ),
                    "trainable_parameter_count_per_temporal_model": EXPECTED_PARAMETER_COUNT,
                    "single_controller": True,
                    "automatic_follow_on_launched": False,
                }
            ),
        )
        _write_json_create_only(
            output_directory / "artifact_manifest.json", _artifact_manifest(output_directory)
        )
        return {
            "status": "twenty_fits_complete_awaiting_independent_replay",
            "fit_attempts": attempts,
            "run_seconds": elapsed,
        }
    except BaseException as exc:
        cleanup = _stop_task_owned_workers(
            baseline_workers, attempts, completed, terminal="fit_phase_incomplete_cleanup"
        )
        if (
            output_owned
            and output_directory.is_dir()
            and not (output_directory / "INCOMPLETE.json").exists()
        ):
            failure = _sealed(
                {
                    "record_kind": "fog_motion_incomplete",
                    "status": "incomplete",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "fit_attempt_count": attempts,
                    "completed_fit_count": completed,
                    "unattempted_fit_count": MAXIMUM_FIT_ATTEMPTS - attempts,
                    "fit_schedule": [
                        {
                            "attempt_number": index + 1,
                            "outer_fold": fold,
                            "cell_id": cell,
                            "status": (
                                "completed"
                                if index < completed
                                else "failed_or_interrupted"
                                if index < attempts
                                else "unattempted"
                            ),
                        }
                        for index, (fold, cell) in enumerate(FIT_SCHEDULE)
                    ],
                    "automatic_retry_launched": False,
                    "automatic_follow_on_launched": False,
                    "worker_cleanup": cleanup,
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
    validation_baseline: Mapping[str, Any],
    controller_started: float,
) -> dict[str, Any]:
    """Independently rebuild context and replay all twenty saved checkpoints."""

    validation_started = time.perf_counter()
    _require(run_directory.is_dir(), "run directory is missing")
    _require(not (run_directory / "INCOMPLETE.json").exists(), "fit phase is incomplete")
    _require(not (run_directory / "validation.json").exists(), "run is already validated")
    artifact_count = _verify_artifact_manifest(run_directory)

    preflight = _read_json(run_directory / "preflight.json")
    _verify_sealed(preflight, "preflight")
    repository_root = Path(str(preflight["repository_root"])).resolve()
    evidence_root = Path(str(preflight["evidence_root"])).resolve()
    _require(
        run_directory == (evidence_root / EVIDENCE_FAMILY / RUN_DIRECTORY_NAME).resolve(),
        "validated output location changed",
    )
    command = _read_json(run_directory / "command_receipt.json")
    _verify_sealed(command, "command receipt")
    _require(
        Path(str(command["repository_root"])).resolve() == repository_root
        and Path(str(command["evidence_root"])).resolve() == evidence_root
        and Path(str(command["output_directory"])).resolve() == run_directory
        and command["controller_count"] == 1,
        "command receipt changed",
    )
    config = _read_yaml(run_directory / "config_snapshot.yaml")
    validate_config(config)
    _require(
        sha256_file(run_directory / "config_snapshot.yaml") == preflight["config_sha256"]
        and _canonical_config_sha256(config) == CONFIG_SEMANTIC_SHA256,
        "configuration snapshot changed",
    )
    protocol_snapshot = _read_json(run_directory / "protocol_snapshot.json")
    specification_snapshot = _read_json(run_directory / "specification_snapshot.json")
    _verify_sealed(protocol_snapshot, "protocol snapshot")
    _verify_sealed(specification_snapshot, "specification snapshot")
    _require(
        protocol_snapshot["source_sha256"] == config["protocol_sha256"]
        and specification_snapshot["source_sha256"] == SPEC_SHA256,
        "governing document snapshot changed",
    )
    code_commit = str(preflight["code_commit"])
    git = _git_state(repository_root)
    _require(
        git["clean"] is True and git["commit"] == code_commit,
        "validator source is not the clean run commit",
    )
    _require(
        sha256_file(repository_root / CONFIG_RELATIVE)
        == sha256_file(run_directory / "config_snapshot.yaml")
        and sha256_file(repository_root / PROTOCOL_RELATIVE) == protocol_snapshot["source_sha256"]
        and sha256_file(repository_root / SPEC_RELATIVE) == SPEC_SHA256,
        "current governing inputs changed",
    )
    source_manifest = _read_json(run_directory / "source_manifest.json")
    _verify_sealed(source_manifest, "source manifest")
    _require(
        canonical_json_sha256(_source_manifest(repository_root))
        == source_manifest["record_sha256"],
        "current source inventory changed",
    )
    _require(
        preflight["source_manifest_record_sha256"] == source_manifest["record_sha256"]
        and preflight["config_semantic_sha256"] == CONFIG_SEMANTIC_SHA256
        and preflight["expected_fit_attempts"] == 20
        and preflight["context_frozen_before_outcomes_loaded"] is True,
        "preflight source or lifecycle binding changed",
    )
    environment_record = _read_json(run_directory / "environment.json")
    _verify_sealed(environment_record, "environment")
    _configure_runtime(11)
    observed_environment = _require_environment(config)
    for key in (
        "python",
        "python_executable",
        "implementation_file",
        "motion_context_implementation_file",
        "platform",
        "numpy",
        "pandas",
        "scipy",
        "scikit_learn",
        "torch",
        "PyYAML",
        "joblib",
        "psutil",
        "cuda_available",
        "torch_cuda",
        "gpu",
        "nvidia_driver",
        "dependency_inventory_sha256",
        "torch_num_threads",
        "torch_num_interop_threads",
        "threadpool_info",
    ):
        _require(
            environment_record[key] == observed_environment[key],
            f"validation environment changed: {key}",
        )

    input_manifest = _read_json(run_directory / "input_manifest.json")
    _verify_sealed(input_manifest, "input manifest")
    _require(
        input_manifest["context_record_sha256"] == preflight["context_preflight_record_sha256"],
        "input/context binding changed",
    )
    raw = _mapping(input_manifest["raw_source"], "raw source")
    coverage = _mapping(input_manifest["coverage"], "coverage")
    availability = _mapping(input_manifest["availability"], "availability")
    raw_path = Path(str(raw["path"])).resolve()
    coverage_path = Path(str(coverage["path"])).resolve()
    availability_path = Path(str(availability["path"])).resolve()
    _require(
        raw_path.is_relative_to(evidence_root)
        and coverage_path.is_relative_to(evidence_root)
        and availability_path.is_relative_to(evidence_root),
        "input path escapes evidence root",
    )
    for path, row, name in (
        (raw_path, raw, "raw source"),
        (coverage_path, coverage, "coverage"),
        (availability_path, availability, "availability"),
    ):
        _require(path.is_file() and sha256_file(path) == row["sha256"], f"{name} changed")
        if "size_bytes" in row:
            _require(path.stat().st_size == int(row["size_bytes"]), f"{name} size changed")

    spatial_run, spatial_receipt = _verify_spatial_reference(evidence_root)
    replay_context = preflight_fog_motion_context(
        raw_path=raw_path,
        coverage_path=coverage_path,
        spatial_cache_path=spatial_run / "feature_cache.npz",
    )
    stored_context_receipt = _read_json(run_directory / "context_preflight.json")
    _verify_sealed(stored_context_receipt, "stored context preflight")
    _require(
        canonical_json_sha256(stored_context_receipt)
        == canonical_json_sha256(replay_context.receipt),
        "independent raw context receipt changed",
    )
    contexts = replay_context.contexts
    context_cache = _read_npz(run_directory / "context_cache.npz")
    expected_context = {
        "signals": contexts.signals,
        "derived_gravity": contexts.derived_gravity,
        "timestamps": contexts.timestamps,
        "observation_mask": contexts.observation_mask,
        "current_availability_mask": contexts.current_availability_mask,
        "full_context_mask": contexts.full_context_mask,
        "observable_window_ids": contexts.observable_window_ids,
    }
    _require(set(context_cache) == set(expected_context), "context-cache schema changed")
    for name, value in expected_context.items():
        _require(np.array_equal(context_cache[name], value), f"raw context replay changed: {name}")
    context_metadata = _read_json(run_directory / "context_cache_metadata.json")
    _verify_sealed(context_metadata, "context cache metadata")
    _require(
        canonical_json_sha256(context_metadata)
        == canonical_json_sha256(_cache_metadata(expected_context)),
        "context-cache metadata changed",
    )
    context_alignment = _read_json(run_directory / "context_alignment_receipts.json")
    _verify_sealed(context_alignment, "context alignment receipts")
    _require(
        canonical_json_sha256(context_alignment["alignment_receipts"])
        == canonical_json_sha256(list(contexts.alignment_receipts))
        and canonical_json_sha256(context_alignment["segment_receipts"])
        == canonical_json_sha256(list(contexts.segment_receipts)),
        "context alignment or segment receipts changed",
    )
    expected_permutations = np.stack(
        [history_block_permutation(str(value)) for value in contexts.observable_window_ids]
    )
    permutation_arrays = _read_npz(run_directory / "history_permutations.npz")
    _require(
        set(permutation_arrays) == {"observable_window_ids", "permutations", "full_context_mask"}
        and np.array_equal(
            permutation_arrays["observable_window_ids"], contexts.observable_window_ids
        )
        and np.array_equal(permutation_arrays["permutations"], expected_permutations)
        and np.array_equal(permutation_arrays["full_context_mask"], contexts.full_context_mask),
        "history permutation replay changed",
    )
    permutation_receipt = _read_json(run_directory / "permutation_receipt.json")
    _verify_sealed(permutation_receipt, "permutation receipt")
    _require(
        permutation_receipt["permutation_sha256"] == _array_sha256(expected_permutations)
        and permutation_receipt["identity_count_all_observable"]
        == int(np.all(expected_permutations == np.arange(3), axis=1).sum())
        and permutation_receipt["identity_count_full_context"]
        == int(
            np.all(expected_permutations[contexts.full_context_mask] == np.arange(3), axis=1).sum()
        )
        and permutation_receipt["current_query_unchanged"] is True,
        "permutation receipt changed",
    )

    references, reference_receipt = _verify_reference(evidence_root, config)
    reference_receipt["spatial_reference_receipt"] = spatial_receipt
    stored_reference_receipt = _read_json(run_directory / "reference_receipt.json")
    _verify_sealed(stored_reference_receipt, "reference receipt")
    expected_reference_receipt = _sealed(reference_receipt)
    _require(
        canonical_json_sha256(stored_reference_receipt)
        == canonical_json_sha256(expected_reference_receipt),
        "reference receipt replay changed",
    )
    erratum_snapshot = _read_json(run_directory / "mandatory_l9v_fallback_erratum_snapshot.json")
    _verify_sealed(erratum_snapshot, "mandatory L9v fallback erratum")
    _require(
        canonical_json_sha256(erratum_snapshot["source"])
        == canonical_json_sha256(reference_receipt["erratum"]),
        "mandatory L9v fallback erratum changed",
    )
    baseline_stack = np.asarray(references["reference_observable_probabilities"], dtype=np.float64)
    controls = {name: baseline_stack[index] for index, name in enumerate(REFERENCE_ORDER)}
    scoring_indices = np.asarray(replay_context.scoring_indices, dtype=np.int64)
    scoring = np.asarray(replay_context.scoring_eligibility, dtype=np.bool_)
    labels_scored = np.asarray(replay_context.scored_labels, dtype=np.int64)
    people = np.asarray(replay_context.observable_participant_ids, dtype=np.str_)
    folds = np.asarray(replay_context.observable_fold_index, dtype=np.int64)
    observable_labels = np.zeros(people.size, dtype=np.int64)
    observable_labels[scoring_indices] = labels_scored
    full_indices = np.flatnonzero(contexts.full_context_mask).astype(np.int64)
    _require(
        np.array_equal(
            references["reference_observable_window_ids"], contexts.observable_window_ids
        )
        and np.array_equal(references["observable_window_ids"], contexts.observable_window_ids)
        and np.array_equal(references["reference_observable_participant_ids"], people)
        and np.array_equal(references["observable_participant_ids"], people)
        and np.array_equal(references["reference_observable_fold_index"], folds)
        and np.array_equal(references["observable_fold_index"], folds)
        and np.array_equal(references["reference_scoring_indices"], scoring_indices)
        and np.array_equal(references["reference_scored_labels"], labels_scored)
        and np.array_equal(
            references["reference_availability_mask"], contexts.current_availability_mask
        )
        and np.array_equal(references["availability_mask"], contexts.current_availability_mask)
        and np.array_equal(
            references["available_indices"], np.flatnonzero(contexts.current_availability_mask)
        ),
        "independent reference alignment changed",
    )
    _require(
        np.array_equal(
            references["reference_scored_probabilities"], baseline_stack[:, scoring_indices]
        )
        and np.array_equal(references["reference_scored_participant_ids"], people[scoring_indices])
        and np.array_equal(
            references["reference_scored_window_ids"],
            contexts.observable_window_ids[scoring_indices],
        )
        and np.array_equal(references["reference_scored_fold_index"], folds[scoring_indices]),
        "independent reference scoring projection changed",
    )
    roster = _roster(config)
    retained_analysis = _read_json(evidence_root / REFERENCE_RELATIVE / "analysis.json")
    retained_l9v = _mapping(
        _mapping(retained_analysis["reports"], "retained reports")["l9v"],
        "retained L9v report",
    )
    replayed_l9v = method_report(
        labels=labels_scored,
        probabilities=controls["l9v"][scoring_indices],
        participant_ids=people[scoring_indices],
        roster=roster,
    )
    parity = _read_json(run_directory / "l9v_metric_parity.json")
    _verify_sealed(parity, "L9v metric parity")
    _require(
        canonical_json_sha256(replayed_l9v) == canonical_json_sha256(retained_l9v)
        and parity["status"] == "pass_before_first_fit"
        and parity["replayed_report_sha256"] == canonical_json_sha256(replayed_l9v)
        and parity["retained_report_sha256"] == canonical_json_sha256(retained_l9v),
        "retained L9v metric parity changed",
    )

    partition = _read_json(run_directory / "partition_preflight.json")
    _verify_sealed(partition, "partition preflight")
    partition_rows = cast(list[dict[str, Any]], partition["rows"])
    _require(len(partition_rows) == 5, "partition row count changed")
    fold_contract = _read_npz(run_directory / "fold_training_contract.npz")
    batch_orders = _read_npz(run_directory / "batch_orders.npz")
    expected_fold_contract_keys = {
        f"{name}_fold_{fold}"
        for fold in OUTER_FOLDS
        for name in (
            "training_indices",
            "prediction_indices",
            "weights",
            "temporal_mean",
            "temporal_scale",
            "e2_mean",
            "e2_scale",
        )
    }
    _require(
        set(fold_contract) == expected_fold_contract_keys
        and set(batch_orders) == {f"fold_{fold}" for fold in OUTER_FOLDS},
        "prefit fold-contract schema changed",
    )
    initial_receipt = _read_json(run_directory / "initial_state_receipt.json")
    _verify_sealed(initial_receipt, "initial-state receipt")
    _require(
        initial_receipt["created_before_first_fit"] is True, "initial states were not marked prefit"
    )
    initial_rows = {
        int(row["outer_fold"]): row for row in cast(list[dict[str, Any]], initial_receipt["rows"])
    }
    _require(set(initial_rows) == set(OUTER_FOLDS), "initial-state folds changed")

    fit_report = _read_json(run_directory / "fit_reports.json")
    _verify_sealed(fit_report, "fit reports")
    fit_rows = cast(list[dict[str, Any]], fit_report["rows"])
    _require(
        fit_report["fit_attempt_count"] == fit_report["completed_fit_count"] == 20
        and len(fit_rows) == 20,
        "twenty-fit ledger changed",
    )
    row_lookup: dict[tuple[int, str], dict[str, Any]] = {}
    binding = {
        "code_commit": code_commit,
        "source_manifest_record_sha256": source_manifest["record_sha256"],
        "input_manifest_record_sha256": input_manifest["record_sha256"],
        "config_sha256": preflight["config_sha256"],
        "config_semantic_sha256": CONFIG_SEMANTIC_SHA256,
        "protocol_sha256": preflight["protocol_sha256"],
        "specification_sha256": SPEC_SHA256,
    }
    energy = current_energy_features(contexts.signals)
    l9v_names = references["l9v_names"].tolist()
    available_indices = np.asarray(references["available_indices"], dtype=np.int64)
    available_lookup = np.full(people.size, -1, dtype=np.int64)
    available_lookup[available_indices] = np.arange(available_indices.size)
    for column, name in enumerate(("accelerometer_norm__rms", "gyroscope_norm__rms")):
        cached = np.asarray(references["l9v_values"], dtype=np.float64)[
            available_lookup[full_indices], l9v_names.index(name)
        ]
        _require(
            np.array_equal(energy[full_indices, column], np.log(np.maximum(cached, 1e-8))),
            f"validation E2 energy parity changed: {name}",
        )
    for attempt, (fold, cell) in enumerate(FIT_SCHEDULE, start=1):
        started_receipt = _read_json(
            run_directory / "fit_attempts" / f"{attempt:02d}--started.json"
        )
        completed_receipt = _read_json(
            run_directory / "fit_attempts" / f"{attempt:02d}--completed.json"
        )
        _verify_sealed(started_receipt, f"fit {attempt} start")
        _verify_sealed(completed_receipt, f"fit {attempt} completion")
        _require(
            started_receipt["attempt_number"] == completed_receipt["attempt_number"] == attempt
            and started_receipt["outer_fold"] == completed_receipt["outer_fold"] == fold
            and started_receipt["cell_id"] == completed_receipt["cell_id"] == cell,
            f"fit schedule changed at attempt {attempt}",
        )
        for name, value in binding.items():
            _require(
                started_receipt[name] == completed_receipt[name] == value,
                f"fit binding changed: {attempt} {name}",
            )
        train = np.flatnonzero(scoring & contexts.full_context_mask & (folds != fold)).astype(
            np.int64
        )
        predict = np.flatnonzero(contexts.full_context_mask & (folds == fold)).astype(np.int64)
        labels_train = observable_labels[train]
        people_train = people[train]
        weights = participant_class_weights(labels_train, people_train)
        expected_common = {
            "fold_seed": 11 + fold,
            "training_row_count": int(train.size),
            "prediction_candidate_count": int(predict.size),
            "training_indices_sha256": _array_sha256(train),
            "prediction_indices_sha256": _array_sha256(predict),
            "sample_weight_sha256": _array_sha256(weights),
        }
        for receipt in (started_receipt, completed_receipt):
            for field_name, expected_value in expected_common.items():
                _require(
                    receipt[field_name] == expected_value,
                    f"fit support changed: {attempt} {field_name}",
                )
        expected_binary = np.asarray(labels_train == 0, dtype=np.int64)
        _require(
            completed_receipt["training_original_class_counts"]
            == np.bincount(labels_train, minlength=3).tolist()
            and completed_receipt["training_binary_class_counts"]
            == np.bincount(expected_binary, minlength=2).tolist()
            and completed_receipt["sample_weight_sum"] == float(weights.sum())
            and canonical_json_sha256(completed_receipt["weight_summary"])
            == canonical_json_sha256(_weight_summary(labels_train, people_train, weights))
            and completed_receipt["source_binding_reverified_at_completion"] is True,
            f"fit label/weight report changed: {attempt}",
        )
        partition_row = partition_rows[fold]
        _require(
            partition_row["outer_fold"] == fold
            and partition_row["training_indices_sha256"] == _array_sha256(train)
            and partition_row["prediction_indices_sha256"] == _array_sha256(predict)
            and partition_row["training_rows"] == int(train.size)
            and partition_row["prediction_candidates"] == int(predict.size)
            and partition_row["training_original_class_counts"]
            == np.bincount(labels_train, minlength=3).tolist()
            and partition_row["training_participants"] == sorted(np.unique(people[train]).tolist())
            and partition_row["held_out_participants"]
            == sorted(np.unique(people[predict]).tolist())
            and set(partition_row["training_participants"]).isdisjoint(
                partition_row["held_out_participants"]
            ),
            f"participant partition changed: {fold}",
        )
        row = fit_rows[attempt - 1]
        row_lookup[(fold, cell)] = row
        body = dict(completed_receipt)
        body.pop("record_kind")
        body.pop("record_sha256")
        body.pop("completed_at_utc")
        _require(
            canonical_json_sha256(body) == canonical_json_sha256(row),
            f"fit report differs from completion receipt: {attempt}",
        )
        checkpoint = _mapping(row["checkpoint"], "fit checkpoint")
        expected_suffix = ".pkl" if cell == "E2" else ".pt"
        _require(
            checkpoint["path"] == f"checkpoints/{cell}--fold-{fold}{expected_suffix}"
            and row["fold_seed"] == 11 + fold,
            f"checkpoint identity changed: {attempt}",
        )
        if cell == "E2":
            e2_mean, e2_scale = _standardize_fit(energy[train])
            _require(
                row["normalizer_mean"] == e2_mean.tolist()
                and row["normalizer_scale"] == e2_scale.tolist()
                and row["convergence_warnings"] == []
                and all(int(value) < 1000 for value in cast(list[int], row["iterations"])),
                f"E2 training contract changed: {fold}",
            )
        else:
            temporal_mean, temporal_scale = _standardize_fit(
                contexts.signals[train, -128:].reshape(-1, 6)
            )
            expected_order_hash = _array_sha256(np.stack(_epoch_orders(train.size, 11 + fold)))
            _configure_determinism(11 + fold)
            expected_rng = _rng_receipt()
            _require(
                row["normalizer_mean"] == temporal_mean.tolist()
                and row["normalizer_scale"] == temporal_scale.tolist()
                and started_receipt["batch_orders_sha256"]
                == row["batch_orders_sha256"]
                == expected_order_hash
                and started_receipt["shared_initial_state_sha256"]
                == row["shared_initial_state_sha256"]
                == initial_rows[fold]["state_sha256"]
                and started_receipt["training_rng"] == row["training_rng"] == expected_rng
                and row["parameter_count"] == EXPECTED_PARAMETER_COUNT
                and row["epochs"] == 80
                and len(cast(list[Any], row["training_history"])) == 80
                and cast(list[dict[str, Any]], row["training_history"])[0]["learning_rate"]
                == _epoch_learning_rate(0)
                and cast(list[dict[str, Any]], row["training_history"])[-1]["learning_rate"]
                == _epoch_learning_rate(79),
                f"temporal training contract changed: {cell} fold {fold}",
            )
    _require(len(row_lookup) == 20, "duplicate fit ledger cell")

    predictions = _read_npz(run_directory / "predictions.npz")
    expected_prediction_keys = {
        "method_ids",
        "observable_probabilities",
        "scored_probabilities",
        "raw_full_context_motion_probabilities",
        "effective_observable_motion_probabilities",
        "full_context_indices",
        "full_context_mask",
        "current_availability_mask",
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
    }
    _require(set(predictions) == expected_prediction_keys, "prediction schema changed")
    _require(predictions["method_ids"].tolist() == list(METHOD_ORDER), "method order changed")
    for name, value in (
        ("full_context_indices", full_indices),
        ("full_context_mask", contexts.full_context_mask),
        ("current_availability_mask", contexts.current_availability_mask),
        ("scoring_indices", scoring_indices),
        ("scored_labels", labels_scored),
        ("observable_window_ids", contexts.observable_window_ids),
        ("observable_participant_ids", people),
        ("observable_fold_index", folds),
        ("scored_participant_ids", people[scoring_indices]),
        ("scored_window_ids", contexts.observable_window_ids[scoring_indices]),
        ("scored_fold_index", folds[scoring_indices]),
    ):
        _require(np.array_equal(predictions[name], value), f"prediction provenance changed: {name}")

    raw_replay = {cell: np.full(people.size, np.nan, dtype=np.float64) for cell in CELL_ORDER}
    device = torch.device("cuda:0")
    neural_errors: dict[str, float] = {}
    for fold in OUTER_FOLDS:
        train = np.flatnonzero(scoring & contexts.full_context_mask & (folds != fold)).astype(
            np.int64
        )
        predict = np.flatnonzero(contexts.full_context_mask & (folds == fold)).astype(np.int64)
        _require(
            np.array_equal(train, fold_contract[f"training_indices_fold_{fold}"])
            and np.array_equal(predict, fold_contract[f"prediction_indices_fold_{fold}"])
            and np.array_equal(
                np.stack(_epoch_orders(train.size, 11 + fold)), batch_orders[f"fold_{fold}"]
            ),
            f"saved fold contract changed: {fold}",
        )
        weights = participant_class_weights(observable_labels[train], people[train])
        temporal_mean, temporal_scale = _standardize_fit(
            contexts.signals[train, -128:].reshape(-1, 6)
        )
        e2_mean, e2_scale = _standardize_fit(energy[train])
        _require(
            np.array_equal(weights, fold_contract[f"weights_fold_{fold}"])
            and np.array_equal(temporal_mean, fold_contract[f"temporal_mean_fold_{fold}"])
            and np.array_equal(temporal_scale, fold_contract[f"temporal_scale_fold_{fold}"])
            and np.array_equal(e2_mean, fold_contract[f"e2_mean_fold_{fold}"])
            and np.array_equal(e2_scale, fold_contract[f"e2_scale_fold_{fold}"]),
            f"training-only fold statistics changed: {fold}",
        )

        e2_row = row_lookup[(fold, "E2")]
        e2_checkpoint = _mapping(e2_row["checkpoint"], "E2 checkpoint")
        e2_path = run_directory / str(e2_checkpoint["path"])
        _require(
            e2_path.is_file() and sha256_file(e2_path) == e2_checkpoint["sha256"],
            f"E2 checkpoint changed: {fold}",
        )
        with e2_path.open("rb") as stream:
            e2_saved = cast(dict[str, Any], pickle.load(stream))
        estimator = cast(LogisticRegression, e2_saved["estimator"])
        estimator_parameters = estimator.get_params(deep=False)
        _require(
            e2_saved["cell_id"] == "E2"
            and e2_saved["outer_fold"] == fold
            and e2_saved["training_indices_sha256"] == _array_sha256(train)
            and e2_saved["prediction_indices_sha256"] == _array_sha256(predict)
            and np.array_equal(e2_saved["mean"], e2_mean)
            and np.array_equal(e2_saved["scale"], e2_scale)
            and tuple(e2_saved["feature_names"])
            == ("accelerometer_norm__rms", "gyroscope_norm__rms")
            and estimator.classes_.tolist() == [0, 1]
            and estimator_parameters["penalty"] == "l2"
            and estimator_parameters["C"] == 1.0
            and estimator_parameters["solver"] == "lbfgs"
            and estimator_parameters["fit_intercept"] is True
            and estimator_parameters["max_iter"] == 1000
            and estimator_parameters["tol"] == 1e-6
            and estimator_parameters["class_weight"] is None
            and estimator_parameters["warm_start"] is False,
            f"E2 checkpoint contract changed: {fold}",
        )
        for name, value in binding.items():
            _require(e2_saved[name] == value, f"E2 checkpoint binding changed: {fold} {name}")
        with threadpool_limits(limits=1):
            e2_probability = np.asarray(
                estimator.predict_proba(_standardize(energy[predict], e2_mean, e2_scale))[:, 1],
                dtype=np.float64,
            )
            e2_all_logits = np.asarray(
                estimator.decision_function(_standardize(energy, e2_mean, e2_scale)),
                dtype=np.float64,
            )
            e2_training_logits = np.asarray(
                estimator.decision_function(_standardize(energy[train], e2_mean, e2_scale)),
                dtype=np.float64,
            )
        e2_target = np.asarray(observable_labels[train] == 0, dtype=np.int64)
        expected_data_loss = float(
            np.sum(
                weights * (np.logaddexp(0.0, e2_training_logits) - e2_target * e2_training_logits)
            )
            / np.sum(weights)
        )
        expected_penalty = float(np.sum(estimator.coef_**2) / (2.0 * np.sum(weights)))
        _require(
            e2_row["coefficient"] == estimator.coef_.tolist()
            and e2_row["intercept"] == estimator.intercept_.tolist()
            and e2_row["iterations"] == estimator.n_iter_.tolist()
            and e2_row["weighted_training_data_log_loss"] == expected_data_loss
            and e2_row["explicit_regularization_term"] == expected_penalty
            and e2_row["explicit_weighted_objective"] == expected_data_loss + expected_penalty,
            f"E2 objective replay changed: {fold}",
        )
        raw_replay["E2"][predict] = e2_probability

        initial_path = run_directory / "initial_states" / f"fold-{fold}.pt"
        initial_row = initial_rows[fold]
        _require(
            initial_path.is_file()
            and sha256_file(initial_path) == initial_row["sha256"]
            and initial_path.stat().st_size == int(initial_row["size_bytes"]),
            f"initial state file changed: {fold}",
        )
        initial_saved = cast(
            dict[str, Any], torch.load(initial_path, map_location="cpu", weights_only=False)
        )
        initial_state = cast(dict[str, Tensor], initial_saved["state_dict"])
        _configure_determinism(11 + fold)
        regenerated_model = MotionResidualTCN().cpu()
        regenerated_initial_state = {
            name: value.detach().cpu().clone()
            for name, value in regenerated_model.state_dict().items()
        }
        _require(
            initial_saved["outer_fold"] == fold
            and initial_saved["config_semantic_sha256"] == CONFIG_SEMANTIC_SHA256
            and _state_sha256(initial_state) == initial_row["state_sha256"]
            and _state_sha256(regenerated_initial_state) == initial_row["state_sha256"],
            f"initial state changed: {fold}",
        )
        for cell in CELL_ORDER[1:]:
            row = row_lookup[(fold, cell)]
            checkpoint = _mapping(row["checkpoint"], "temporal checkpoint")
            checkpoint_path = run_directory / str(checkpoint["path"])
            _require(
                checkpoint_path.is_file() and sha256_file(checkpoint_path) == checkpoint["sha256"],
                f"temporal checkpoint changed: {cell} fold {fold}",
            )
            saved = cast(
                dict[str, Any], torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            )
            state = cast(dict[str, Tensor], saved["state_dict"])
            _require(
                saved["cell_id"] == cell
                and saved["outer_fold"] == fold
                and saved["training_indices_sha256"] == _array_sha256(train)
                and saved["prediction_indices_sha256"] == _array_sha256(predict)
                and saved["shared_initial_state_sha256"] == initial_row["state_sha256"]
                and saved["frozen_e2_checkpoint_sha256"] == e2_checkpoint["sha256"]
                and np.array_equal(saved["normalizer_mean"], temporal_mean)
                and np.array_equal(saved["normalizer_scale"], temporal_scale)
                and _state_sha256(state) == row["final_state_sha256"],
                f"temporal checkpoint contract changed: {cell} fold {fold}",
            )
            for name, value in binding.items():
                _require(
                    saved[name] == value,
                    f"temporal checkpoint binding changed: {cell} fold {fold} {name}",
                )
            model = MotionResidualTCN().to(device)
            model.load_state_dict(state, strict=True)
            _require(
                sum(parameter.numel() for parameter in model.parameters())
                == EXPECTED_PARAMETER_COUNT,
                "replay parameter count changed",
            )
            values, mask = _prepare_temporal_inputs(
                contexts.signals,
                contexts.observation_mask,
                temporal_mean,
                temporal_scale,
                contexts.observable_window_ids,
                cell,
            )
            replay_loss_numerator = 0.0
            replay_loss_denominator = 0.0
            model.eval()
            with torch.inference_mode():
                for left in range(0, train.size, 32):
                    selected = train[left : left + 32]
                    logits = model(
                        torch.from_numpy(values[selected]).to(device),
                        torch.from_numpy(mask[selected]).to(device),
                        torch.from_numpy(np.asarray(e2_all_logits[selected], dtype=np.float32)).to(
                            device
                        ),
                    )
                    batch_target = torch.from_numpy(
                        np.asarray(observable_labels[selected] == 0, dtype=np.float32)
                    ).to(device)
                    batch_weight = torch.from_numpy(
                        np.asarray(weights[left : left + selected.size], dtype=np.float32)
                    ).to(device)
                    losses = F.binary_cross_entropy_with_logits(
                        logits, batch_target, reduction="none"
                    )
                    replay_loss_numerator += float(torch.sum(batch_weight * losses).cpu())
                    replay_loss_denominator += float(torch.sum(batch_weight).cpu())
            _require(
                row["final_whole_training_weight_denominator"] == replay_loss_denominator
                and abs(
                    float(row["final_whole_training_weighted_loss"])
                    - replay_loss_numerator / replay_loss_denominator
                )
                <= 1e-7,
                f"final training-loss replay changed: {cell} fold {fold}",
            )
            replayed = _predict_temporal(model, values, mask, e2_all_logits, predict, device=device)
            raw_replay[cell][predict] = replayed
            stored_raw = predictions["raw_full_context_motion_probabilities"][
                CELL_ORDER.index(cell)
            ]
            lookup = np.searchsorted(full_indices, predict)
            neural_errors[f"{cell}--fold-{fold}"] = float(
                np.max(np.abs(replayed - stored_raw[lookup]))
            )

    stored_raw = np.asarray(predictions["raw_full_context_motion_probabilities"], dtype=np.float64)
    _require(stored_raw.shape == (4, 1724), "raw motion prediction shape changed")
    _require(
        np.array_equal(raw_replay["E2"][full_indices], stored_raw[0]),
        "E2 checkpoint replay changed",
    )
    maximum_neural_error = max(neural_errors.values())
    _require(maximum_neural_error <= 1e-7, "neural checkpoint replay exceeds tolerance")
    for cell in CELL_ORDER:
        _require(
            bool(np.isfinite(raw_replay[cell][full_indices]).all()),
            f"checkpoint replay incomplete: {cell}",
        )

    replay_probabilities: dict[str, FloatArray] = {}
    replay_effective: dict[str, FloatArray] = {}
    for cell in CELL_ORDER:
        motion = controls["l9v"][:, 0].copy()
        motion[full_indices] = raw_replay[cell][full_indices]
        replay_probabilities[cell] = compose_motion_probability(
            motion, controls["l9v"], contexts.full_context_mask, controls["b0"]
        )
        replay_effective[cell] = replay_probabilities[cell][:, 0].copy()
    replay_probabilities.update(controls)
    replay_stack = np.stack([replay_probabilities[name] for name in METHOD_ORDER])
    stored_observable = np.asarray(predictions["observable_probabilities"], dtype=np.float64)
    stored_scored = np.asarray(predictions["scored_probabilities"], dtype=np.float64)
    _require(stored_observable.shape == (9, 1939, 3), "observable prediction shape changed")
    _require(
        np.array_equal(stored_observable[4:], baseline_stack)
        and np.array_equal(stored_scored[4:], baseline_stack[:, scoring_indices]),
        "retained controls changed",
    )
    candidate_error = float(np.max(np.abs(replay_stack[:4] - stored_observable[:4])))
    _require(candidate_error <= 1e-7, "composed checkpoint replay exceeds tolerance")
    _require(
        np.array_equal(stored_observable[:, scoring_indices], stored_scored)
        and np.array_equal(predictions["observable_decisions"], replay_stack.argmax(axis=2))
        and np.array_equal(
            predictions["scored_decisions"], replay_stack[:, scoring_indices].argmax(axis=2)
        ),
        "projection or exact decision replay changed",
    )
    q_zero = contexts.full_context_mask & (controls["l9v"][:, 1:].sum(axis=1) == 0.0)
    for index, cell in enumerate(CELL_ORDER):
        _require(
            np.array_equal(
                stored_observable[index, ~contexts.full_context_mask],
                controls["l9v"][~contexts.full_context_mask],
            )
            and np.array_equal(stored_observable[index, q_zero], controls["l9v"][q_zero])
            and np.array_equal(
                stored_observable[index, ~contexts.current_availability_mask],
                controls["b0"][~contexts.current_availability_mask],
            ),
            f"exact fallback replay changed: {cell}",
        )
        effective_error = float(
            np.max(
                np.abs(
                    predictions["effective_observable_motion_probabilities"][index]
                    - replay_effective[cell]
                )
            )
        )
        _require(effective_error <= 1e-7, f"effective motion replay changed: {cell}")

    roster = _roster(config)
    scored_people = people[scoring_indices]
    scored_folds = folds[scoring_indices]
    stored_probabilities = {
        method: stored_scored[index] for index, method in enumerate(METHOD_ORDER)
    }
    reports = {
        method: extended_method_report(
            labels=labels_scored,
            probabilities=stored_probabilities[method],
            participants=scored_people,
            roster=roster,
            conditional_posture_probabilities=(
                stored_probabilities["l9v"] if method in CELL_ORDER else None
            ),
        )
        for method in METHOD_ORDER
    }
    scored_full = contexts.full_context_mask[scoring_indices]
    scored_current = contexts.current_availability_mask[scoring_indices]
    scored_q_zero = q_zero[scoring_indices]
    strata = {
        "scored_full_context_qualified": int(scored_full.sum()),
        "scored_full_context_intervention": int((scored_full & ~scored_q_zero).sum()),
        "scored_full_context_q_zero_l9v_fallback": int(scored_q_zero.sum()),
        "current_ankle_short_history_l9v_fallback": int((scored_current & ~scored_full).sum()),
        "missing_ankle_b0_fallback": int((~scored_current).sum()),
        "observable_full_context_qualified": int(contexts.full_context_mask.sum()),
        "observable_full_context_q_zero_l9v_fallback": int(q_zero.sum()),
        "P017_full_context_scored": int((scored_full & (scored_people == "fogstar:017")).sum()),
    }
    participant_record = _read_json(run_directory / "participant_metrics.json")
    _verify_sealed(participant_record, "participant metrics")
    _require(
        canonical_json_sha256(participant_record["methods"]) == canonical_json_sha256(reports)
        and canonical_json_sha256(participant_record["strata"]) == canonical_json_sha256(strata),
        "participant report replay changed",
    )
    full_scored_global = scoring_indices[scored_full]
    full_scored_lookup = np.searchsorted(full_indices, full_scored_global)
    raw_reports = {
        cell: raw_motion_report(
            labels=labels_scored[scored_full],
            motion_probability=stored_raw[index, full_scored_lookup],
            participants=scored_people[scored_full],
        )
        for index, cell in enumerate(CELL_ORDER)
    }
    raw_record = _read_json(run_directory / "raw_motion_diagnostics.json")
    _verify_sealed(raw_record, "raw motion diagnostics")
    _require(
        full_scored_global.size == 1098
        and canonical_json_sha256(raw_record["methods"]) == canonical_json_sha256(raw_reports),
        "raw learner diagnostic replay changed",
    )
    participant_folds = np.asarray(
        [int(np.unique(scored_folds[scored_people == person])[0]) for person in roster],
        dtype=np.int64,
    )
    prevalidation_expected = analyse(
        reports,
        labels=labels_scored,
        probabilities=stored_probabilities,
        participants=scored_people,
        participant_folds=participant_folds,
        roster=roster,
        validation_complete=False,
    )
    prevalidation_record = _read_json(run_directory / "analysis_prevalidation.json")
    _verify_sealed(prevalidation_record, "prevalidation analysis")
    prevalidation_body = dict(prevalidation_record)
    prevalidation_body.pop("record_sha256")
    _require(
        canonical_json_sha256(prevalidation_body) == canonical_json_sha256(prevalidation_expected),
        "prevalidation analysis replay changed",
    )
    final_analysis = analyse(
        reports,
        labels=labels_scored,
        probabilities=stored_probabilities,
        participants=scored_people,
        participant_folds=participant_folds,
        roster=roster,
        validation_complete=True,
    )
    analysis_record = _sealed(final_analysis)
    _write_json_create_only(run_directory / "analysis.json", analysis_record)
    _write_bytes_create_only(
        run_directory / "OUTCOME_SUMMARY.md", _outcome_summary(analysis_record).encode("utf-8")
    )

    runtime = _read_json(run_directory / "runtime.json")
    worker = _read_json(run_directory / "worker_shutdown.json")
    result_prevalidation = _read_json(run_directory / "result_prevalidation.json")
    for record, name in (
        (runtime, "runtime"),
        (worker, "fit worker shutdown"),
        (result_prevalidation, "prevalidation result"),
    ):
        _verify_sealed(record, name)
    _require(
        runtime["fit_attempt_count"] == runtime["completed_fit_count"] == 20
        and worker["task_owned_fit_workers_and_monitors_stopped"] is True,
        "fit runtime or worker completion changed",
    )
    result = _sealed(
        {
            "record_kind": "fog_motion_factorization_result",
            "status": "complete_and_independently_replayed",
            "experiment_id": EXPERIMENT_ID,
            "code_commit": code_commit,
            "decision": final_analysis["decision"],
            "advancement": final_analysis["advancement"],
            "gates": final_analysis["gates"],
            "diagnostic_gates": final_analysis["diagnostic_gates"],
            "method_summary": _method_summary(reports),
            "fallback_strata": strata,
            "fit_attempt_count": 20,
            "completed_fit_count": 20,
            "attempted_configurations": [
                {
                    "attempt_number": index + 1,
                    "outer_fold": fold,
                    "cell_id": cell,
                    "status": "completed",
                }
                for index, (fold, cell) in enumerate(FIT_SCHEDULE)
            ],
            "checkpoint_replay_maximum_absolute_neural_probability_error": maximum_neural_error,
            "composed_replay_maximum_absolute_probability_error": candidate_error,
            "source_provenance": {
                "raw_path": str(raw_path),
                "raw_sha256": raw["sha256"],
                "reference_run": str(evidence_root / REFERENCE_RELATIVE),
                "source_manifest_record_sha256": source_manifest["record_sha256"],
            },
            "InclusiveHAR_P11_P20_loaded": False,
            "automatic_follow_on_launched": False,
            "additional_seed_launched": False,
            "confirmation_claimed": False,
            "novelty_claimed": False,
            "publication_or_release_action": False,
        }
    )
    _write_json_create_only(run_directory / "result.json", result)

    completion_binding = _fit_binding(
        repository_root,
        code_commit=code_commit,
        source_manifest_record_sha256=str(source_manifest["record_sha256"]),
        input_manifest_record_sha256=str(input_manifest["record_sha256"]),
    )
    stored_completion_binding = _read_json(run_directory / "completion_source_binding.json")
    _verify_sealed(stored_completion_binding, "completion source binding")
    _require(
        stored_completion_binding["status"] == "unchanged_after_twenty_fits"
        and all(
            stored_completion_binding[name] == value for name, value in completion_binding.items()
        ),
        "post-fit source binding changed",
    )
    validation_worker = _stop_task_owned_workers(
        validation_baseline, attempts=0, completed=0, terminal="validation_replay_complete"
    )
    _require(
        validation_worker["task_owned_fit_workers_and_monitors_stopped"] is True,
        "validation worker remains",
    )
    _write_json_create_only(run_directory / "validation_worker_shutdown.json", validation_worker)
    validation_seconds = time.perf_counter() - validation_started
    validation = _sealed(
        {
            "record_kind": "fog_motion_factorization_validation",
            "status": "validated",
            "validated_at_utc": _now(),
            "run_source_commit": code_commit,
            "validator_source_commit": git["commit"],
            "artifact_hashes_verified": artifact_count,
            "independent_raw_context_replay": True,
            "context_arrays_byte_exact": len(expected_context),
            "checkpoint_predictions_replayed": 20,
            "model_fits_during_validation": 0,
            "maximum_absolute_neural_probability_error": maximum_neural_error,
            "maximum_allowed_neural_probability_error": 1e-7,
            "maximum_absolute_composed_probability_error": candidate_error,
            "decisions_and_ids_exact": True,
            "fallback_probabilities_byte_exact": True,
            "retained_reference_probabilities_byte_exact": True,
            "participant_reports_recomputed": len(METHOD_ORDER),
            "primary_comparisons_recomputed": 5,
            "all_pairwise_comparisons_recomputed": 36,
            "advancement_gates_recomputed": 4,
            "diagnostic_gates_recomputed": 1,
            "full_22_person_roster_retained": True,
            "full_1213_scored_rows_retained": True,
            "InclusiveHAR_P11_P20_loaded": False,
            "source_downloaded_during_validation": False,
            "automatic_follow_on_launched": False,
            "validation_task_owned_workers_and_monitors_stopped": True,
            "validation_seconds": validation_seconds,
            "combined_controller_seconds_before_completion_manifest": (
                time.perf_counter() - controller_started
            ),
            "completion_source_binding": completion_binding,
        }
    )
    _write_json_create_only(run_directory / "validation.json", validation)
    files = [
        path
        for path in run_directory.rglob("*")
        if path.is_file()
        and path.name not in {"completion_manifest.json", "controller_final_receipt.json"}
    ]
    completion = _sealed(
        {
            "record_kind": "fog_motion_factorization_completion_manifest",
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
            "analysis_record_sha256": analysis_record["record_sha256"],
            "result_record_sha256": result["record_sha256"],
            "decision": final_analysis["decision"],
            "task_owned_workers_and_monitors_stopped": True,
            "automatic_follow_on_launched": False,
        }
    )
    _write_json_create_only(run_directory / "completion_manifest.json", completion)
    for row in cast(list[dict[str, Any]], completion["artifacts"]):
        path = run_directory / str(row["path"])
        _require(
            path.is_file()
            and path.stat().st_size == int(row["size_bytes"])
            and sha256_file(path) == row["sha256"],
            f"completion artifact changed: {path}",
        )
    return validation


def validate_run(run_directory: Path, *, controller_started: float | None = None) -> dict[str, Any]:
    """Run the zero-fit validator with an explicit failure and worker-shutdown record."""

    baseline = _worker_baseline()
    resolved = run_directory.resolve()
    started = time.perf_counter() if controller_started is None else controller_started
    recognized_run = False
    try:
        ownership = _read_json(resolved / "preflight.json")
        _verify_sealed(ownership, "validation ownership preflight")
        declared_evidence = Path(str(ownership["evidence_root"])).resolve()
        snapshot = _read_yaml(resolved / "config_snapshot.yaml")
        recognized_run = (
            resolved == (declared_evidence / EVIDENCE_FAMILY / RUN_DIRECTORY_NAME).resolve()
            and snapshot.get("experiment_id") == EXPERIMENT_ID
            and ownership.get("code_commit") is not None
        )
    except BaseException:
        recognized_run = False
    try:
        return _validate_run_impl(
            resolved, validation_baseline=baseline, controller_started=started
        )
    except BaseException as exc:
        cleanup = _stop_task_owned_workers(
            baseline, attempts=0, completed=0, terminal="validation_incomplete_cleanup"
        )
        if (
            recognized_run
            and resolved.is_dir()
            and not (resolved / "completion_manifest.json").exists()
            and not (resolved / "VALIDATION_INCOMPLETE.json").exists()
        ):
            try:
                _write_json_create_only(
                    resolved / "VALIDATION_INCOMPLETE.json",
                    _sealed(
                        {
                            "record_kind": "fog_motion_factorization_validation_incomplete",
                            "status": "incomplete_blocker_not_scientific_rejection",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                            "model_fits_during_validation": 0,
                            "worker_cleanup": cleanup,
                            "retry_launched": False,
                            "automatic_follow_on_launched": False,
                        }
                    ),
                )
            except BaseException:
                pass
        raise


def _finalize_controller_receipt(
    output_directory: Path,
    validation: Mapping[str, Any],
    *,
    controller_started: float,
    operation: str,
) -> dict[str, Any]:
    completion_path = output_directory / "completion_manifest.json"
    receipt = _sealed(
        {
            "record_kind": "fog_motion_factorization_controller_final_receipt",
            "status": "complete",
            "operation": operation,
            "observed_at_utc": _now(),
            "complete_controller_seconds": time.perf_counter() - controller_started,
            "completion_manifest_file_sha256": sha256_file(completion_path),
            "validation_record_sha256": validation["record_sha256"],
            "model_fit_attempts_in_sealed_run": 20,
            "model_fits_during_validation": 0,
            "all_task_owned_workers_and_monitors_stopped": True,
            "automatic_follow_on_launched": False,
            "additional_seed_launched": False,
        }
    )
    _write_json_create_only(output_directory / "controller_final_receipt.json", receipt)
    return receipt


def execute(
    *,
    repository_root: Path,
    evidence_root: Path,
    config_path: Path,
    protocol_path: Path,
    output_directory: Path,
    code_commit: str,
    cancel_file: Path | None,
) -> dict[str, Any]:
    """Run the bounded family, validate it without fitting, and seal final evidence."""

    controller_started = time.perf_counter()
    run_experiment(
        repository_root=repository_root,
        evidence_root=evidence_root,
        config_path=config_path,
        protocol_path=protocol_path,
        output_directory=output_directory,
        code_commit=code_commit,
        cancel_file=cancel_file,
    )
    validation = validate_run(output_directory, controller_started=controller_started)
    receipt = _finalize_controller_receipt(
        output_directory,
        validation,
        controller_started=controller_started,
        operation="fit_then_independent_validation",
    )
    return {
        "status": "complete_and_independently_replayed",
        "decision": _read_json(output_directory / "result.json")["decision"],
        "complete_controller_seconds": receipt["complete_controller_seconds"],
        "output_directory": str(output_directory.resolve()),
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
    execute_parser.add_argument("--cancel-file", type=Path)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--run-directory", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "execute":
        result = execute(
            repository_root=arguments.repository_root,
            evidence_root=arguments.evidence_root,
            config_path=arguments.config,
            protocol_path=arguments.protocol,
            output_directory=arguments.output_directory,
            code_commit=str(arguments.code_commit),
            cancel_file=arguments.cancel_file,
        )
    else:
        validation_started = time.perf_counter()
        validation = validate_run(arguments.run_directory, controller_started=validation_started)
        receipt = _finalize_controller_receipt(
            arguments.run_directory,
            validation,
            controller_started=validation_started,
            operation="standalone_independent_validation",
        )
        result = {
            "status": "complete_and_independently_replayed",
            "decision": _read_json(arguments.run_directory / "result.json")["decision"],
            "complete_controller_seconds": receipt["complete_controller_seconds"],
            "output_directory": str(arguments.run_directory.resolve()),
            "completion_manifest_file_sha256": receipt["completion_manifest_file_sha256"],
        }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
