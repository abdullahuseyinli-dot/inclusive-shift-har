"""Execute and replay the frozen corrected-FoG optional pretrained-context study."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
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
from threadpoolctl import threadpool_info  # type: ignore[import-untyped]

from inclusive_shift_har.experiments.fog_left_ankle_derived_nine import (
    _input_paths,
    _stop_task_owned_workers,
    _verify_spatial_reference,
    _worker_baseline,
)
from inclusive_shift_har.experiments.fog_motion_factorization import (
    compose_motion_probability,
    participant_class_weights,
    preflight_fog_motion_context,
)
from inclusive_shift_har.experiments.fog_motion_factorization_run import (
    _configure_runtime,
    _verify_reference,
    current_energy_features,
    extended_method_report,
    raw_motion_report,
)
from inclusive_shift_har.experiments.fog_pretrained_optional_context import (
    HARNET_EMBEDDING_DIMENSION,
    NativeHarnetContexts,
    extract_harnet_embeddings,
    fit_standardizer,
    instantiate_harnet_extractor,
    prospective_motion_headroom,
    reconstruct_native_harnet_contexts,
    standardized_optional_embedding,
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

EXPERIMENT_ID = "fog-pretrained-optional-context-v1"
RUN_DIRECTORY_NAME = "fog-pretrained-optional-context-seed11-20260908-001"
EVIDENCE_FAMILY = Path(".audit/fog_pretrained_optional_context")
CONFIG_RELATIVE = Path("configs/experiments/fog_pretrained_optional_context_v1.yaml")
PROTOCOL_RELATIVE = Path("docs/research/FOG_PRETRAINED_OPTIONAL_CONTEXT_V1_PROTOCOL.md")
SPEC_RELATIVE = Path(".audit/research_breakthrough_review_20260908-002/EXACT_NEXT_EXPERIMENT.md")
OLD_CONFIG_RELATIVE = Path("configs/experiments/fog_motion_factorization_v1.yaml")
REFERENCE_RELATIVE = Path(
    ".audit/fog_left_ankle_derived_nine/fog-left-ankle-derived-nine-seed11-20260908-003"
)
HISTORICAL_MOTION_RELATIVE = Path(
    ".audit/fog_motion_factorization/fog-motion-factorization-seed11-20260908-001"
)
CONFIG_SHA256 = "8e7b959332bfd98482ffb36a7d1d89215ab7e8d440045ba1b69a880bbe85ae69"
SPEC_SHA256 = "716c939d5c6742705b924e7288572294c6654f6bdd88e3f78ff7cefac0e913ca"
PROTOCOL_SHA256 = "27a31156bc89008d56c3d72881993a54189adaf319cded0a39fbed7245b49c2d"
HARNET_COMMIT = "150550ea5d41800229c95e36f88f5bf0d2e7cf04"
HARNET_CHECKPOINT_SHA256 = "c64f9135d99e2dcdfc9ae7cc0672f2bcc438df9ceb8215665882f92cddd162a6"
HARNET_SOURCE_SHA256 = "3721adb0064135c5ec0703af59fccf2afceca2a8b16176c6339875d148e3cda0"
HARNET_LICENSE_SHA256 = "331fd5fc22235c1af71044c100bd46b9a429f74d4f8397a5fe050984871703ea"
CELL_ORDER = ("Q", "M", "R", "P")
HISTORICAL_MOTION_ORDER = ("E2", "T128", "T500", "T500-P")
REFERENCE_ORDER = ("l9v", "lv", "bv", "b0", "f3")
METHOD_ORDER = (*CELL_ORDER, *HISTORICAL_MOTION_ORDER, *REFERENCE_ORDER)
HISTORICAL_MOTION_SOURCE_COMMIT = "4d4e591b9d714c384abdcaac2afc67acf08f87c7"
HISTORICAL_MOTION_FILE_SHA256 = {
    "predictions.npz": "0b5b399fea0159644ca8d63f08215a99d98e60c42d1d927191fdd9ce78e34a73",
    "analysis.json": "17911b6475d1cd9cce7f3ced92fdd389072b5293dcb3a23a16052f17f6fb496d",
    "validation.json": "7e1838b61d8e794850459073b5664c698cdd1a8dd5598e13727b3c22f753512b",
    "completion_manifest.json": (
        "39d83696f0251609c1684c7809e0bc1fa60721a099a322daf3d7e9562ef02311"
    ),
}
OUTER_FOLDS = tuple(range(5))
MAXIMUM_FIT_ATTEMPTS = 20
FIT_SCHEDULE = tuple((fold, arm) for fold in OUTER_FOLDS for arm in CELL_ORDER)
EXPECTED_TRAINING_ROWS = (940, 825, 947, 829, 1_075)
EXPECTED_EVALUATION_ROWS = (214, 329, 207, 325, 79)
EXPECTED_TRAINING_CLASSES = (
    (767, 43, 130),
    (649, 55, 121),
    (761, 53, 133),
    (626, 60, 143),
    (857, 65, 153),
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


def _write_pickle_create_only(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        pickle.dump(value, stream, protocol=5)


def _roster(config: Mapping[str, Any]) -> list[str]:
    held_out = _mapping(_mapping(config["folds"], "folds")["held_out_participants"], "held out")
    people = sorted(str(value) for values in held_out.values() for value in cast(list[Any], values))
    _require(people == [f"fogstar:{index:03d}" for index in range(1, 23)], "roster changed")
    return people


def _participant_fold_vector(
    people: StringArray, folds: IntArray, roster: Sequence[str]
) -> IntArray:
    _require(people.shape == folds.shape, "row fold arrays are misaligned")
    values = []
    for person in roster:
        selected = np.unique(folds[people == person])
        _require(selected.size == 1, f"participant fold assignment changed: {person}")
        values.append(int(selected[0]))
    result = np.asarray(values, dtype=np.int64)
    _require(
        result.shape == (22,) and set(result.tolist()) == set(OUTER_FOLDS),
        "participant fold vector changed",
    )
    return result


def validate_config(config: Mapping[str, Any], config_path: Path, protocol_path: Path) -> None:
    _require(config.get("experiment_id") == EXPERIMENT_ID, "experiment ID changed")
    _require(config.get("status") == "frozen_before_fitting", "configuration is not frozen")
    _require(sha256_file(config_path) == CONFIG_SHA256, "configuration bytes changed")
    _require(
        protocol_path.is_file()
        and sha256_file(protocol_path) == PROTOCOL_SHA256
        and config["protocol_sha256"] == PROTOCOL_SHA256,
        "protocol binding changed",
    )
    authority = _mapping(config["authority"], "authority")
    _require(authority["specification_sha256"] == SPEC_SHA256, "specification binding changed")
    harnet = _mapping(config["harnet"], "harnet")
    _require(
        harnet["source_commit"] == HARNET_COMMIT
        and harnet["checkpoint_sha256"] == HARNET_CHECKPOINT_SHA256
        and harnet["embedding_dimension"] == HARNET_EMBEDDING_DIMENSION,
        "HARNET contract changed",
    )
    _require(
        harnet["repeat_max_absolute_tolerance"] == 0.0
        and harnet["batch_single_max_absolute_tolerance"] == 1e-5,
        "HARNET numeric equivalence tolerance changed",
    )
    reference = _mapping(config["reference"], "reference")
    _require(
        reference["historical_motion_run_path"] == HISTORICAL_MOTION_RELATIVE.as_posix()
        and reference["historical_motion_methods"] == list(HISTORICAL_MOTION_ORDER)
        and reference["historical_motion_source_commit"] == HISTORICAL_MOTION_SOURCE_COMMIT
        and reference["historical_motion_predictions_sha256"]
        == HISTORICAL_MOTION_FILE_SHA256["predictions.npz"]
        and reference["historical_motion_analysis_sha256"]
        == HISTORICAL_MOTION_FILE_SHA256["analysis.json"]
        and reference["historical_motion_validation_sha256"]
        == HISTORICAL_MOTION_FILE_SHA256["validation.json"]
        and reference["historical_motion_completion_sha256"]
        == HISTORICAL_MOTION_FILE_SHA256["completion_manifest.json"],
        "historical motion reference contract changed",
    )
    cells = _mapping(config["cells"], "cells")
    _require(cells["order"] == list(CELL_ORDER) and cells["primary"] == "P", "cell order changed")
    runtime = _mapping(config["runtime"], "runtime")
    _require(
        runtime["maximum_fit_attempts"] == MAXIMUM_FIT_ATTEMPTS
        and runtime["automatic_retry"] is False
        and runtime["additional_seed"] is False
        and runtime["automatic_follow_on"] is False,
        "runtime bound changed",
    )
    oracle = _mapping(
        _mapping(config["gates"], "gates")["prospective_label_oracle_preflight"], "oracle"
    )
    _require(
        oracle["expected_editable_rows"] == 853
        and oracle["expected_maximum_possible_strict_wins"] == 17
        and oracle["never_use_for_fitting_or_prediction"] is True,
        "gate-feasibility contract changed",
    )
    _roster(config)


def _git_output(root: Path, *arguments: str) -> str:
    return subprocess.check_output(("git", "-C", str(root), *arguments), text=True).strip()


def _verify_harnet_root(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    section = _mapping(config["harnet"], "harnet")
    expected = {
        "sslearning/models/accNet.py": str(section["source_file_sha256"]),
        "hubconf.py": str(section["hubconf_sha256"]),
        "data_parsing/README.md": str(section["preprocessing_readme_sha256"]),
        "LICENSE.md": str(section["license_sha256"]),
        "model_check_point/mtl_best.mdl": str(section["checkpoint_sha256"]),
    }
    _require(root.is_dir() and (root / ".git").exists(), "HARNET source checkout is missing")
    _require(_git_output(root, "rev-parse", "HEAD") == HARNET_COMMIT, "HARNET commit changed")
    _require(_git_output(root, "status", "--porcelain=v1") == "", "HARNET checkout is dirty")
    files = []
    for name, digest in expected.items():
        path = root / name
        _require(path.is_file() and sha256_file(path) == digest, f"HARNET input changed: {name}")
        files.append({"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size})
    checkpoint = root / str(section["checkpoint_file"])
    _require(
        checkpoint.stat().st_size == section["checkpoint_size_bytes"], "checkpoint size changed"
    )
    _require(
        _git_output(root, "hash-object", str(section["checkpoint_file"]))
        == section["checkpoint_git_blob_sha1"],
        "checkpoint Git object changed",
    )
    return _sealed(
        {
            "record_kind": "fog_pretrained_harnet_input_receipt",
            "source_repository": section["source_repository"],
            "source_commit": HARNET_COMMIT,
            "checkout_clean": True,
            "files": files,
            "checkpoint_source_url": section["checkpoint_source_url"],
            "checkpoint_git_blob_sha1": section["checkpoint_git_blob_sha1"],
            "strict_feature_state_load_required": True,
            "licence": section["license"],
            "licence_scope_used": "internal academic non-commercial thesis research",
            "upstream_bytes_will_not_be_committed_or_redistributed": True,
        }
    )


def _environment() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    properties = torch.cuda.get_device_properties(0) if cuda else None
    distributions = sorted(
        f"{item.metadata['Name']}=={item.version}"
        for item in importlib_metadata.distributions()
        if item.metadata["Name"]
    )
    driver = subprocess.run(
        ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return {
        "python": sys.version,
        "python_executable": sys.executable,
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
        "gpu": None
        if properties is None
        else {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": int(properties.total_memory),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "nvidia_driver": driver.stdout.strip() if driver.returncode == 0 else None,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "threadpool_info": threadpool_info(),
        "dependency_inventory": distributions,
        "dependency_inventory_sha256": canonical_json_sha256(distributions),
    }


def _require_environment(config: Mapping[str, Any]) -> dict[str, Any]:
    observed = _environment()
    expected = _mapping(_mapping(config["runtime"], "runtime")["expected_versions"], "versions")
    _require(platform.python_version() == config["runtime"]["expected_python"], "Python changed")
    for name in ("numpy", "scipy", "scikit-learn", "torch", "PyYAML", "joblib"):
        key = "scikit_learn" if name == "scikit-learn" else name
        _require(observed[key] == expected[name], f"runtime version changed: {name}")
    _require(observed["cuda_available"] is True, "CUDA is unavailable")
    _require(
        observed["torch_num_threads"] == observed["torch_num_interop_threads"] == 1
        and all(int(row.get("num_threads", 1)) == 1 for row in observed["threadpool_info"]),
        "single-thread runtime contract changed",
    )
    return observed


def _source_manifest(repository_root: Path) -> dict[str, Any]:
    paths = (
        "AGENTS.md",
        "src/inclusive_shift_har/data/external_har.py",
        "src/inclusive_shift_har/experiments/fog_rf_feature_weight_factorial.py",
        "src/inclusive_shift_har/experiments/fog_spatial_information_probe.py",
        "src/inclusive_shift_har/experiments/fog_left_ankle_derived_nine.py",
        "src/inclusive_shift_har/experiments/fog_motion_factorization.py",
        "src/inclusive_shift_har/experiments/fog_motion_factorization_run.py",
        "src/inclusive_shift_har/experiments/fog_pretrained_optional_context.py",
        "src/inclusive_shift_har/experiments/fog_pretrained_optional_context_run.py",
        "tests/test_fog_pretrained_optional_context.py",
        "tests/test_fog_pretrained_optional_context_run.py",
        "configs/experiments/fog_motion_factorization_v1.yaml",
        CONFIG_RELATIVE.as_posix(),
        PROTOCOL_RELATIVE.as_posix(),
        "pyproject.toml",
        "uv.lock",
    )
    return {
        "record_kind": "fog_pretrained_optional_context_source_manifest",
        "files": [{"path": name, "sha256": sha256_file(repository_root / name)} for name in paths],
    }


def _fit_binding(
    repository_root: Path,
    harnet_root: Path,
    config: Mapping[str, Any],
    code_commit: str,
    source_manifest_sha256: str,
    input_manifest_sha256: str,
) -> dict[str, Any]:
    git = _git_state(repository_root)
    _require(git["clean"] is True and git["commit"] == code_commit, "source changed before fit")
    source = _source_manifest(repository_root)
    _require(
        canonical_json_sha256(source) == source_manifest_sha256,
        "source manifest changed before fit",
    )
    harnet = _verify_harnet_root(harnet_root, config)
    return {
        "code_commit": code_commit,
        "source_manifest_record_sha256": source_manifest_sha256,
        "input_manifest_record_sha256": input_manifest_sha256,
        "config_sha256": CONFIG_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "specification_sha256": SPEC_SHA256,
        "harnet_input_record_sha256": harnet["record_sha256"],
    }


def _snapshot_inputs(
    repository_root: Path,
    config_path: Path,
    protocol_path: Path,
    output_directory: Path,
) -> None:
    _write_bytes_create_only(output_directory / "config_snapshot.yaml", config_path.read_bytes())
    for output_name, source in (
        ("protocol_snapshot.json", protocol_path),
        ("specification_snapshot.json", repository_root / SPEC_RELATIVE),
    ):
        _write_json_create_only(
            output_directory / output_name,
            _sealed(
                {
                    "record_kind": output_name.removesuffix(".json"),
                    "source_path": str(source),
                    "source_sha256": sha256_file(source),
                    "text": source.read_text(encoding="utf-8"),
                }
            ),
        )


def _load_reference(
    repository_root: Path, evidence_root: Path
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    old_config = _read_yaml(repository_root / OLD_CONFIG_RELATIVE)
    return _verify_reference(evidence_root, old_config)


def _load_historical_motion_reference(
    evidence_root: Path,
    preflight: Any,
    references: Mapping[str, NDArray[Any]],
) -> tuple[dict[str, FloatArray], dict[str, Mapping[str, Any]], dict[str, Any]]:
    run = evidence_root / HISTORICAL_MOTION_RELATIVE
    _require(run.is_dir(), "historical motion reference run is missing")
    for name, expected in HISTORICAL_MOTION_FILE_SHA256.items():
        path = run / name
        _require(
            path.is_file() and sha256_file(path) == expected,
            f"historical motion reference changed: {name}",
        )
    validation = _read_json(run / "validation.json")
    _require(
        validation["status"] == "validated"
        and validation["run_source_commit"] == HISTORICAL_MOTION_SOURCE_COMMIT
        and validation["model_fits_during_validation"] == 0
        and validation["InclusiveHAR_P11_P20_loaded"] is False,
        "historical motion validation contract changed",
    )
    predictions = _read_npz(run / "predictions.npz")
    expected_methods = (*HISTORICAL_MOTION_ORDER, *REFERENCE_ORDER)
    _require(
        predictions["method_ids"].tolist() == list(expected_methods)
        and np.array_equal(
            predictions["observable_window_ids"], preflight.contexts.observable_window_ids
        )
        and np.array_equal(
            predictions["observable_participant_ids"], preflight.observable_participant_ids
        )
        and np.array_equal(predictions["observable_fold_index"], preflight.observable_fold_index)
        and np.array_equal(predictions["scoring_indices"], preflight.scoring_indices)
        and np.array_equal(predictions["scored_labels"], preflight.scored_labels)
        and np.array_equal(
            predictions["current_availability_mask"],
            preflight.contexts.current_availability_mask,
        )
        and np.array_equal(predictions["full_context_mask"], preflight.contexts.full_context_mask),
        "historical motion reference alignment changed",
    )
    probability_stack = cast(
        FloatArray, np.asarray(predictions["observable_probabilities"], dtype=np.float64)
    )
    _require(
        probability_stack.shape == (len(expected_methods), 1_939, 3)
        and np.array_equal(
            probability_stack[len(HISTORICAL_MOTION_ORDER) :],
            np.asarray(references["reference_observable_probabilities"], dtype=np.float64),
        ),
        "historical motion baseline probabilities changed",
    )
    historical: dict[str, FloatArray] = {
        name: probability_stack[index] for index, name in enumerate(HISTORICAL_MOTION_ORDER)
    }
    analysis = _read_json(run / "analysis.json")
    prior_reports = _mapping(analysis["reports"], "historical motion reports")
    reports: dict[str, Mapping[str, Any]] = {
        name: _mapping(prior_reports[name], name) for name in HISTORICAL_MOTION_ORDER
    }
    receipt = {
        "run_path": str(run),
        "source_commit": HISTORICAL_MOTION_SOURCE_COMMIT,
        "method_ids": list(HISTORICAL_MOTION_ORDER),
        "file_sha256": dict(HISTORICAL_MOTION_FILE_SHA256),
        "validation_record_sha256": validation["record_sha256"],
        "probability_sha256_by_method": {
            name: _array_sha256(probability) for name, probability in historical.items()
        },
        "zero_new_fits": True,
    }
    return historical, reports, receipt


def _prepare_data(
    repository_root: Path,
    evidence_root: Path,
) -> tuple[
    Any,
    NativeHarnetContexts,
    dict[str, NDArray[Any]],
    dict[str, Any],
    dict[str, Path],
    dict[str, FloatArray],
    dict[str, Mapping[str, Any]],
    dict[str, Any],
]:
    spatial_run, spatial_receipt = _verify_spatial_reference(evidence_root)
    inputs = _input_paths(spatial_run)
    preflight = preflight_fog_motion_context(
        raw_path=inputs["raw_source"],
        coverage_path=inputs["coverage"],
        spatial_cache_path=spatial_run / "feature_cache.npz",
    )
    coverage = _read_json(inputs["coverage"])
    rows = cast(list[dict[str, Any]], coverage["candidate_availability"])
    native = reconstruct_native_harnet_contexts(
        raw_path=inputs["raw_source"],
        coverage_rows=rows,
        current_availability_mask=preflight.contexts.current_availability_mask,
        expected_full_context_mask=preflight.contexts.full_context_mask,
    )
    references, reference_receipt = _load_reference(repository_root, evidence_root)
    reference_receipt["spatial_reference_receipt"] = spatial_receipt
    contexts = preflight.contexts
    _require(
        np.array_equal(
            references["reference_observable_window_ids"], contexts.observable_window_ids
        )
        and np.array_equal(
            references["reference_observable_participant_ids"],
            preflight.observable_participant_ids,
        )
        and np.array_equal(
            references["reference_observable_fold_index"], preflight.observable_fold_index
        )
        and np.array_equal(references["reference_scoring_indices"], preflight.scoring_indices)
        and np.array_equal(references["reference_scored_labels"], preflight.scored_labels)
        and np.array_equal(
            references["reference_availability_mask"], contexts.current_availability_mask
        ),
        "reference and context alignment changed",
    )
    historical, historical_reports, historical_receipt = _load_historical_motion_reference(
        evidence_root, preflight, references
    )
    reference_receipt["historical_motion_reference"] = historical_receipt
    return (
        preflight,
        native,
        references,
        reference_receipt,
        inputs,
        historical,
        historical_reports,
        historical_receipt,
    )


def _energy_parity(contexts: Any, references: Mapping[str, NDArray[Any]]) -> FloatArray:
    energy = current_energy_features(contexts.signals)
    available_indices = np.asarray(references["available_indices"], dtype=np.int64)
    l9v_names = np.asarray(references["l9v_names"], dtype=np.str_).tolist()
    l9v_values = np.asarray(references["l9v_values"], dtype=np.float64)
    expected = np.column_stack(
        [
            np.log(np.maximum(l9v_values[:, l9v_names.index(name)], 1e-8))
            for name in ("accelerometer_norm__rms", "gyroscope_norm__rms")
        ]
    )
    _require(
        np.array_equal(energy[available_indices], expected),
        "current-query energy features differ from the qualified L9v cache",
    )
    return energy


def _standardized_query(
    energy: FloatArray, current_mask: BoolArray, training_mask: BoolArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    _require(
        bool(np.all(training_mask <= current_mask)), "query training includes unavailable rows"
    )
    mean, scale = fit_standardizer(energy[training_mask])
    result = (energy - mean) / scale
    _require(bool(np.isfinite(result).all()), "query standardization failed")
    return result, mean, scale


def _arm_features(
    *,
    arm: str,
    query: FloatArray,
    full_context: BoolArray,
    random_embedding: Float32Array,
    pretrained_embedding: Float32Array,
    training_mask: BoolArray,
) -> tuple[FloatArray, dict[str, FloatArray]]:
    mask_column = np.asarray(full_context, dtype=np.float64)[:, None]
    if arm == "Q":
        return query, {}
    if arm == "M":
        return np.column_stack((query, mask_column)), {}
    embeddings = random_embedding if arm == "R" else pretrained_embedding
    standardized, mean, scale = standardized_optional_embedding(
        embeddings, full_context, training_mask
    )
    result = np.column_stack((query, mask_column, standardized))
    _require(result.shape[1] == 1_027, f"{arm} readout dimension changed")
    return result, {"embedding_mean": mean, "embedding_scale": scale}


def _held_out_label_swap_invariance(
    *,
    energy: FloatArray,
    current: BoolArray,
    full: BoolArray,
    scoring: BoolArray,
    folds: IntArray,
    observable_labels: IntArray,
    people: StringArray,
    random_embeddings: Float32Array,
    pretrained_embeddings: Float32Array,
) -> dict[str, Any]:
    rows = []
    for fold in OUTER_FOLDS:
        training = scoring & current & (folds != fold)
        held_out = scoring & current & (folds == fold)
        train_indices = np.flatnonzero(training).astype(np.int64)
        altered = observable_labels.copy()
        altered[held_out] = (altered[held_out] + 1) % 3
        original_weights = participant_class_weights(
            observable_labels[train_indices], people[train_indices]
        )
        altered_weights = participant_class_weights(altered[train_indices], people[train_indices])
        query_a, mean_a, scale_a = _standardized_query(energy, current, training)
        query_b, mean_b, scale_b = _standardized_query(energy, current, training)
        _require(
            np.array_equal(observable_labels[train_indices], altered[train_indices])
            and np.array_equal(original_weights, altered_weights)
            and np.array_equal(query_a, query_b)
            and np.array_equal(mean_a, mean_b)
            and np.array_equal(scale_a, scale_b),
            f"held-out label swap changed common fit inputs: fold {fold}",
        )
        feature_hashes = {}
        for arm in CELL_ORDER:
            matrix_a, extra_a = _arm_features(
                arm=arm,
                query=query_a,
                full_context=full,
                random_embedding=random_embeddings[fold],
                pretrained_embedding=pretrained_embeddings,
                training_mask=training,
            )
            matrix_b, extra_b = _arm_features(
                arm=arm,
                query=query_b,
                full_context=full,
                random_embedding=random_embeddings[fold],
                pretrained_embedding=pretrained_embeddings,
                training_mask=training,
            )
            _require(
                np.array_equal(matrix_a, matrix_b)
                and set(extra_a) == set(extra_b)
                and all(np.array_equal(extra_a[name], extra_b[name]) for name in extra_a),
                f"held-out label swap changed {arm} features: fold {fold}",
            )
            feature_hashes[arm] = _array_sha256(matrix_a)
        rows.append(
            {
                "outer_fold": fold,
                "swapped_held_out_scored_current_labels": int(held_out.sum()),
                "training_labels_sha256": _array_sha256(observable_labels[train_indices]),
                "training_weights_sha256": _array_sha256(original_weights),
                "feature_matrix_sha256": feature_hashes,
                "fit_inputs_exact_after_held_out_label_swap": True,
                "prediction_interface_accepts_no_labels": True,
            }
        )
    return {
        "record_kind": "fog_pretrained_held_out_label_swap_invariance",
        "status": "pass_before_first_fit",
        "rows": rows,
        "features_scalers_weights_and_deterministic_fit_inputs_unchanged": True,
        "held_out_labels_not_available_to_prediction_interface": True,
    }


def _weighted_binary_loss(target: IntArray, probability: FloatArray, weights: FloatArray) -> float:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    loss = -(target * np.log(clipped) + (1 - target) * np.log(1.0 - clipped))
    return float(np.sum(weights * loss) / np.sum(weights))


def _fit_readout(
    features: FloatArray,
    labels: IntArray,
    weights: FloatArray,
    seed: int,
) -> tuple[LogisticRegression, dict[str, Any]]:
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        fit_intercept=True,
        max_iter=1_000,
        tol=1e-6,
        class_weight=None,
        warm_start=False,
        random_state=seed,
    )
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(features, labels, sample_weight=weights)
    elapsed = time.perf_counter() - started
    convergence = [
        str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)
    ]
    _require(not convergence, "logistic readout did not converge")
    _require(model.classes_.tolist() == [0, 1], "binary class order changed")
    probability = model.predict_proba(features)[:, 1]
    coefficient = np.concatenate((model.intercept_.ravel(), model.coef_.ravel()))
    return model, {
        "fit_seconds": elapsed,
        "n_iter": int(model.n_iter_[0]),
        "convergence_warnings": convergence,
        "feature_dimension": int(features.shape[1]),
        "fitted_parameter_count": int(coefficient.size),
        "coefficient_sha256": _array_sha256(np.asarray(coefficient, dtype=np.float64)),
        "whole_training_weighted_binary_nll": _weighted_binary_loss(labels, probability, weights),
    }


def _gate(
    comparison: Mapping[str, Any],
    leave_folds: Sequence[Mapping[str, Any]],
    *,
    minimum_mean_gain: float,
    require_fourteen_wins: bool,
    validation_complete: bool,
) -> dict[str, Any]:
    interval = cast(list[float], comparison["mean_difference_95_percent_bootstrap_interval"])
    recall = _mapping(comparison["class_recall_differences"], "recall differences")
    checks = {
        "mean_gain": float(comparison["mean_difference"]) >= minimum_mean_gain,
        "positive_bootstrap_lower_bound": float(interval[0]) > 0.0,
        "bottom_seven": float(comparison["bottom_30_percent_difference"]) >= -0.010,
        "worst_person_minimum": float(comparison["worst_participant_difference"]) >= -0.030,
        "minimum_paired_person": float(comparison["minimum_paired_participant_difference"])
        >= -0.050,
        "mobility_recall": float(recall["mobility"]) >= -0.010,
        "sitting_recall": float(recall["sitting"]) >= -0.020,
        "standing_recall": float(recall["standing"]) >= -0.020,
        "all_leave_one_person_positive": all(
            float(value) > 0.0
            for value in cast(list[float], comparison["leave_one_participant_out_mean_differences"])
        ),
        "all_leave_one_fold_positive": all(
            float(row["mean_participant_difference"]) > 0.0 for row in leave_folds
        ),
        "validation_complete": validation_complete,
    }
    if require_fourteen_wins:
        checks["participant_wins"] = int(comparison["participant_wins"]) >= 14
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "passed_check_count": sum(checks.values()),
        "required_check_count": len(checks),
        "minimum_mean_gain": minimum_mean_gain,
        "fourteen_win_requirement_applied": require_fourteen_wins,
    }


def analyse(
    *,
    reports: Mapping[str, Mapping[str, Any]],
    labels: IntArray,
    probabilities: Mapping[str, FloatArray],
    participants: StringArray,
    participant_folds: IntArray,
    roster: Sequence[str],
    validation_complete: bool,
) -> dict[str, Any]:
    _require(set(reports) == set(METHOD_ORDER), "report method set changed")
    stages = (
        ("practical", "P", "l9v", 0.015, True),
        ("transferred_weights", "P", "R", 0.010, False),
        ("beyond_query_and_availability", "P", "M", 0.010, False),
    )
    comparisons: dict[str, Any] = {}
    leave_out: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    for stage, candidate, comparator, threshold, wins in stages:
        name = f"{candidate}_minus_{comparator}"
        comparison = paired_comparison(reports[candidate], reports[comparator])
        leave = _leave_one_fold(reports[candidate], reports[comparator], participant_folds)
        comparisons[name] = comparison
        leave_out[name] = leave
        gates[stage] = {
            "comparison": name,
            **_gate(
                comparison,
                leave,
                minimum_mean_gain=threshold,
                require_fourteen_wins=wins,
                validation_complete=validation_complete,
            ),
        }
    diagnostics = ("M", "Q"), ("R", "M"), ("P", "Q"), ("Q", "E2")
    for candidate, comparator in diagnostics:
        name = f"{candidate}_minus_{comparator}"
        comparisons[name] = paired_comparison(reports[candidate], reports[comparator])
        leave_out[name] = _leave_one_fold(
            reports[candidate], reports[comparator], participant_folds
        )
    pairwise = {}
    for left_index, left in enumerate(METHOD_ORDER):
        for right in METHOD_ORDER[left_index + 1 :]:
            pairwise[f"{left}_minus_{right}"] = paired_comparison(reports[left], reports[right])
    sequence = []
    open_claim = True
    for stage, *_rest in stages:
        eligible = open_claim
        status = str(gates[stage]["status"])
        sequence.append(
            {
                "stage": stage,
                "eligible": eligible,
                "gate_status": status,
                "claim_status": status if eligible else "blocked_by_prior_failure",
            }
        )
        if status != "pass":
            open_claim = False
    events = {
        f"{arm}_minus_l9v": _events(
            labels, probabilities["l9v"], probabilities[arm], participants, roster
        )
        for arm in CELL_ORDER
    }
    events["Q_minus_E2"] = _events(
        labels, probabilities["E2"], probabilities["Q"], participants, roster
    )
    passed = all(row["claim_status"] == "pass" for row in sequence)
    return {
        "record_kind": "fog_pretrained_optional_context_analysis",
        "status": "complete_analysis" if validation_complete else "analysis_awaiting_replay",
        "evidence_status": "corrected_fog_adaptive_development_not_confirmation",
        "reports": dict(reports),
        "primary_and_mechanism_comparisons": comparisons,
        "all_pairwise_comparisons": pairwise,
        "leave_one_outer_fold_sensitivity": leave_out,
        "gates": gates,
        "sequential_claims": sequence,
        "event_topology": events,
        "advancement": {"status": "pass" if passed else "fail", "primary_candidate": "P"},
        "decision": (
            "pass_all_three_stages_freeze_P_without_automatic_follow_on"
            if passed
            else "fail_claim_sequence_retain_diagnostic_lessons"
        ),
        "bootstrap_scope": (
            "descriptive participant resampling conditional on overlapping fitted folds, "
            "adaptive development and a repeatedly consumed cohort"
        ),
        "confirmation_claimed": False,
        "novelty_claimed": False,
        "automatic_follow_on_launched": False,
        "additional_seed_launched": False,
        "historical_motion_references_zero_fit": list(HISTORICAL_MOTION_ORDER),
        "Q_minus_E2_cross_support_warning": (
            "Q and E2 differ in current-query support, training rows, weights, and normalization"
        ),
    }


def _outcome_summary(analysis: Mapping[str, Any]) -> str:
    reports = _mapping(analysis["reports"], "reports")
    lines = [
        "# Corrected FoG optional pretrained-context outcome",
        "",
        f"Decision: **{analysis['decision']}**.",
        "",
        "| Method | Mean person F1 | Accuracy | Bottom seven | Worst |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        primary = _mapping(reports[method]["primary"], "primary")
        pooled = _mapping(reports[method]["pooled"], "pooled")
        lines.append(
            f"| {method} | {100 * float(primary['mean_participant_macro_f1']):.3f}% | "
            f"{100 * float(pooled['accuracy']):.3f}% | "
            f"{100 * float(primary['bottom_30_percent_participant_macro_f1']):.3f}% | "
            f"{100 * float(primary['worst_participant_macro_f1']):.3f}% |"
        )
    lines.extend(
        [
            "",
            "P is the sole primary candidate. Gates, paired uncertainty, participant harms, "
            "mechanism contrasts and raw motion diagnostics are in `analysis.json`.",
            "",
            "This is repeatedly consumed FoGSTAR development evidence, not independent "
            "confirmation or a novelty claim. No automatic follow-on was launched.",
            "",
        ]
    )
    return "\n".join(lines)


def _participant_metric_rows(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    _require(set(reports) == set(METHOD_ORDER), "participant report method set changed")
    rows = {}
    for method in METHOD_ORDER:
        report = _mapping(reports[method], f"{method} report")
        participants = cast(list[Any], report["participants"])
        _require(len(participants) == 22, f"{method} participant row count changed")
        rows[method] = participants
    return rows


def _artifact_rows(directory: Path, exclusions: set[str]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(directory).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in exclusions
    ]


def _model_feature_matrix(
    checkpoint: Mapping[str, Any],
    energy: FloatArray,
    full_context: BoolArray,
    random_embeddings: Float32Array,
    pretrained_embeddings: Float32Array,
) -> FloatArray:
    query = np.asarray(
        (energy - np.asarray(checkpoint["query_mean"], dtype=np.float64))
        / np.asarray(checkpoint["query_scale"], dtype=np.float64),
        dtype=np.float64,
    )
    arm = str(checkpoint["arm"])
    if arm == "Q":
        return query
    mask_column = np.asarray(full_context, dtype=np.float64)[:, None]
    if arm == "M":
        return np.column_stack((query, mask_column))
    raw = random_embeddings if arm == "R" else pretrained_embeddings
    standardized = np.zeros_like(raw, dtype=np.float64)
    standardized[full_context] = (
        raw[full_context] - np.asarray(checkpoint["embedding_mean"])
    ) / np.asarray(checkpoint["embedding_scale"])
    return np.column_stack((query, mask_column, standardized))


def run_experiment(
    *,
    repository_root: Path,
    evidence_root: Path,
    harnet_root: Path,
    config_path: Path,
    protocol_path: Path,
    output_directory: Path,
    code_commit: str,
) -> dict[str, Any]:
    """Run exactly four readout arms across five participant folds."""

    repository_root = repository_root.resolve()
    evidence_root = evidence_root.resolve()
    harnet_root = harnet_root.resolve()
    output_directory = output_directory.resolve()
    _require(config_path.resolve() == repository_root / CONFIG_RELATIVE, "config path changed")
    _require(
        protocol_path.resolve() == repository_root / PROTOCOL_RELATIVE, "protocol path changed"
    )
    _require(
        output_directory.parent == evidence_root / EVIDENCE_FAMILY
        and output_directory.name == RUN_DIRECTORY_NAME,
        "output path changed",
    )
    _require(not output_directory.exists(), "create-only run directory exists")
    output_directory.mkdir(parents=True)
    (output_directory / "fit_attempts").mkdir()
    (output_directory / "checkpoints").mkdir()
    started = time.perf_counter()
    attempts = 0
    completed = 0
    config = _read_yaml(config_path)
    try:
        validate_config(config, config_path, protocol_path)
        _require(
            (repository_root / SPEC_RELATIVE).is_file()
            and sha256_file(repository_root / SPEC_RELATIVE) == SPEC_SHA256,
            "governing specification changed",
        )
        git = _git_state(repository_root)
        _require(git["clean"] is True and git["commit"] == code_commit, "source must be clean")
        _configure_runtime(11)
        environment = _require_environment(config)
        harnet_receipt = _verify_harnet_root(harnet_root, config)
        _snapshot_inputs(repository_root, config_path, protocol_path, output_directory)
        source_manifest = _sealed(_source_manifest(repository_root))
        _write_json_create_only(output_directory / "source_manifest.json", source_manifest)
        _write_json_create_only(output_directory / "environment.json", _sealed(environment))
        _write_json_create_only(output_directory / "harnet_input_receipt.json", harnet_receipt)
        _write_json_create_only(
            output_directory / "command_receipt.json",
            _sealed(
                {
                    "record_kind": "fog_pretrained_command_receipt",
                    "argv": list(sys.argv),
                    "working_directory": str(Path.cwd()),
                    "controller_process_id": os.getpid(),
                    "controller_count": 1,
                    "repository_root": str(repository_root),
                    "evidence_root": str(evidence_root),
                    "harnet_root": str(harnet_root),
                    "launched_at_utc": _now(),
                }
            ),
        )

        (
            preflight,
            native,
            references,
            reference_receipt,
            inputs,
            historical_motion,
            historical_motion_reports,
            historical_motion_receipt,
        ) = _prepare_data(repository_root, evidence_root)
        contexts = preflight.contexts
        _write_json_create_only(output_directory / "context_preflight.json", preflight.receipt)
        _write_json_create_only(
            output_directory / "context_alignment_receipts.json",
            _sealed(
                {
                    "record_kind": "fog_pretrained_context_alignment_receipts",
                    "legacy_alignment_receipts": list(contexts.alignment_receipts),
                    "legacy_segment_receipts": list(contexts.segment_receipts),
                    "native_harnet_receipts": list(native.receipts),
                }
            ),
        )
        _write_json_create_only(
            output_directory / "reference_receipt.json", _sealed(reference_receipt)
        )
        energy = _energy_parity(contexts, references)
        _write_npz_create_only(
            output_directory / "context_cache.npz",
            query_energy=energy,
            current_availability_mask=contexts.current_availability_mask,
            full_context_mask=contexts.full_context_mask,
            observable_window_ids=contexts.observable_window_ids,
            native_harnet_histories=native.histories,
            native_harnet_timestamps=native.target_timestamps,
        )
        input_manifest = _sealed(
            {
                "record_kind": "fog_pretrained_input_manifest",
                "raw_source": {
                    "path": str(inputs["raw_source"]),
                    "sha256": sha256_file(inputs["raw_source"]),
                    "size_bytes": inputs["raw_source"].stat().st_size,
                },
                "coverage": {
                    "path": str(inputs["coverage"]),
                    "sha256": sha256_file(inputs["coverage"]),
                },
                "reference_run": str(evidence_root / REFERENCE_RELATIVE),
                "historical_motion_run": str(evidence_root / HISTORICAL_MOTION_RELATIVE),
                "historical_motion_predictions_sha256": historical_motion_receipt["file_sha256"][
                    "predictions.npz"
                ],
                "harnet_input_record_sha256": harnet_receipt["record_sha256"],
                "legacy_context_record_sha256": preflight.receipt["record_sha256"],
                "query_energy_sha256": _array_sha256(energy),
                "native_history_sha256": _array_sha256(native.histories),
                "native_timestamps_sha256": _array_sha256(native.target_timestamps),
                "full_context_mask_sha256": _array_sha256(contexts.full_context_mask),
                "labels_not_used_by_signal_or_encoder_adapter": True,
            }
        )
        _write_json_create_only(output_directory / "input_manifest.json", input_manifest)

        checkpoint_path = harnet_root / str(config["harnet"]["checkpoint_file"])
        device = torch.device("cuda:0")
        pretrained_extractor = instantiate_harnet_extractor(
            source_root=harnet_root, seed=11, checkpoint_path=checkpoint_path
        )
        pretrained_embeddings, pretrained_extract = extract_harnet_embeddings(
            extractor=pretrained_extractor,
            contexts=native,
            device=device,
            batch_size=int(config["harnet"]["batch_size"]),
            repeat_tolerance=float(config["harnet"]["repeat_max_absolute_tolerance"]),
            batch_single_tolerance=float(config["harnet"]["batch_single_max_absolute_tolerance"]),
        )
        random_embeddings = np.zeros(
            (5, energy.shape[0], HARNET_EMBEDDING_DIMENSION), dtype=np.float32
        )
        random_extracts = []
        random_state_hashes = []
        for fold in OUTER_FOLDS:
            extractor = instantiate_harnet_extractor(
                source_root=harnet_root, seed=11 + fold, checkpoint_path=None
            )
            random_state_hashes.append(extractor.state_sha256)
            random_embeddings[fold], receipt = extract_harnet_embeddings(
                extractor=extractor,
                contexts=native,
                device=device,
                batch_size=int(config["harnet"]["batch_size"]),
                repeat_tolerance=float(config["harnet"]["repeat_max_absolute_tolerance"]),
                batch_single_tolerance=float(
                    config["harnet"]["batch_single_max_absolute_tolerance"]
                ),
            )
            random_extracts.append({"outer_fold": fold, **receipt})
        encoder_receipt = _sealed(
            {
                "record_kind": "fog_pretrained_encoder_qualification",
                "pretrained": {
                    "state_sha256": pretrained_extractor.state_sha256,
                    "parameter_count": pretrained_extractor.parameter_count,
                    "state_key_count": pretrained_extractor.state_key_count,
                    "checkpoint_feature_key_count": (
                        pretrained_extractor.checkpoint_feature_key_count
                    ),
                    **pretrained_extract,
                },
                "random": {
                    "seed_rule": "11_plus_outer_fold",
                    "state_sha256_by_fold": random_state_hashes,
                    "extractions": random_extracts,
                },
                "total_extraction_seconds": float(
                    pretrained_extract["extract_seconds"]
                    + sum(float(row["extract_seconds"]) for row in random_extracts)
                ),
                "strict_complete_feature_state_loaded": True,
                "feature_parameters_fitted_on_fogstar": 0,
                "batchnorm_updates": 0,
                "random_and_pretrained_input_history_sha256": _array_sha256(native.histories),
            }
        )
        _write_json_create_only(output_directory / "encoder_qualification.json", encoder_receipt)
        _write_npz_create_only(
            output_directory / "embedding_cache.npz",
            pretrained_embeddings=pretrained_embeddings,
            random_embeddings=random_embeddings,
            full_context_mask=contexts.full_context_mask,
            observable_window_ids=contexts.observable_window_ids,
        )

        scoring = np.asarray(preflight.scoring_eligibility, dtype=np.bool_)
        scoring_indices = np.asarray(preflight.scoring_indices, dtype=np.int64)
        labels_scored = np.asarray(preflight.scored_labels, dtype=np.int64)
        people = np.asarray(preflight.observable_participant_ids, dtype=np.str_)
        folds = np.asarray(preflight.observable_fold_index, dtype=np.int64)
        current = np.asarray(contexts.current_availability_mask, dtype=np.bool_)
        full = np.asarray(contexts.full_context_mask, dtype=np.bool_)
        observable_labels = np.zeros(people.size, dtype=np.int64)
        observable_labels[scoring_indices] = labels_scored
        baseline_stack = np.asarray(
            references["reference_observable_probabilities"], dtype=np.float64
        )
        _require(baseline_stack.shape == (5, 1_939, 3), "reference shape changed")
        controls = {name: baseline_stack[index] for index, name in enumerate(REFERENCE_ORDER)}
        _require(
            np.array_equal(controls["l9v"][~current], controls["b0"][~current]),
            "missing-ankle B0 fallback changed",
        )
        roster = _roster(config)
        participant_folds = _participant_fold_vector(
            people[scoring_indices], folds[scoring_indices], roster
        )
        retained_analysis = _read_json(evidence_root / REFERENCE_RELATIVE / "analysis.json")
        retained_l9v = _mapping(_mapping(retained_analysis["reports"], "reports")["l9v"], "l9v")
        replayed_l9v = method_report(
            labels=labels_scored,
            probabilities=controls["l9v"][scoring_indices],
            participant_ids=people[scoring_indices],
            roster=roster,
        )
        _require(
            canonical_json_sha256(retained_l9v) == canonical_json_sha256(replayed_l9v),
            "retained L9v metric parity failed",
        )
        _write_json_create_only(
            output_directory / "l9v_metric_parity.json",
            _sealed(
                {
                    "record_kind": "fog_pretrained_l9v_metric_parity",
                    "status": "pass_before_first_fit",
                    "report_sha256": canonical_json_sha256(replayed_l9v),
                }
            ),
        )
        scored_current = current[scoring_indices]
        l9v_scored = controls["l9v"][scoring_indices]
        replayed_historical_reports = {
            method: extended_method_report(
                labels=labels_scored,
                probabilities=historical_motion[method][scoring_indices],
                participants=people[scoring_indices],
                roster=roster,
                conditional_posture_probabilities=l9v_scored,
            )
            for method in HISTORICAL_MOTION_ORDER
        }
        historical_report_hashes = {
            method: canonical_json_sha256(report)
            for method, report in replayed_historical_reports.items()
        }
        _require(
            all(
                historical_report_hashes[method]
                == canonical_json_sha256(historical_motion_reports[method])
                for method in HISTORICAL_MOTION_ORDER
            ),
            "historical motion report parity failed",
        )
        _write_json_create_only(
            output_directory / "historical_motion_parity.json",
            _sealed(
                {
                    "record_kind": "fog_pretrained_historical_motion_parity",
                    "status": "pass_before_first_fit",
                    "zero_new_fits": True,
                    "method_report_sha256": historical_report_hashes,
                }
            ),
        )
        q_nonzero = l9v_scored[:, 1:].sum(axis=1) > 0.0
        headroom = prospective_motion_headroom(
            labels=labels_scored,
            participant_ids=people[scoring_indices],
            l9v_probabilities=l9v_scored,
            editable_mask=scored_current & q_nonzero,
            roster=roster,
        )
        _require(
            headroom["editable_rows"] == 853
            and headroom["maximum_possible_strict_participant_wins"] == 17
            and abs(float(headroom["mean_motion_only_ceiling"]) - 0.7518754751866962) < 1e-12,
            "prospective gate-feasibility audit changed",
        )
        _write_json_create_only(
            output_directory / "prospective_gate_feasibility.json", _sealed(headroom)
        )

        partitions = []
        for fold in OUTER_FOLDS:
            training = scoring & current & (folds != fold)
            prediction = current & (folds == fold)
            scored_evaluation = scoring & prediction
            counts = tuple(np.bincount(observable_labels[training], minlength=3).tolist())
            training_people = sorted(np.unique(people[training]).tolist())
            held_out_people = sorted(np.unique(people[prediction]).tolist())
            _require(
                set(training_people).isdisjoint(held_out_people)
                and sorted((*training_people, *held_out_people)) == roster,
                f"participant leakage or omission: fold {fold}",
            )
            partitions.append(
                {
                    "outer_fold": fold,
                    "training_indices": np.flatnonzero(training).tolist(),
                    "prediction_indices": np.flatnonzero(prediction).tolist(),
                    "scored_evaluation_indices": np.flatnonzero(scored_evaluation).tolist(),
                    "training_rows": int(training.sum()),
                    "scored_evaluation_rows": int(scored_evaluation.sum()),
                    "training_original_class_counts": list(counts),
                    "training_participants": training_people,
                    "held_out_participants": held_out_people,
                    "participant_sets_disjoint": True,
                }
            )
        _require(
            tuple(row["training_rows"] for row in partitions) == EXPECTED_TRAINING_ROWS
            and tuple(row["scored_evaluation_rows"] for row in partitions)
            == EXPECTED_EVALUATION_ROWS
            and tuple(tuple(row["training_original_class_counts"]) for row in partitions)
            == EXPECTED_TRAINING_CLASSES,
            "partition support changed",
        )
        _write_json_create_only(
            output_directory / "partition_preflight.json",
            _sealed(
                {
                    "record_kind": "fog_pretrained_partition_preflight",
                    "rows": partitions,
                    "participant_partition_before_windowing_inherited": True,
                }
            ),
        )
        label_swap_invariance = _sealed(
            _held_out_label_swap_invariance(
                energy=energy,
                current=current,
                full=full,
                scoring=scoring,
                folds=folds,
                observable_labels=observable_labels,
                people=people,
                random_embeddings=random_embeddings,
                pretrained_embeddings=pretrained_embeddings,
            )
        )
        _write_json_create_only(
            output_directory / "held_out_label_swap_invariance.json", label_swap_invariance
        )
        prefit = _sealed(
            {
                "record_kind": "fog_pretrained_prefit",
                "status": "pass_before_first_fit",
                "source_commit": code_commit,
                "source_manifest_record_sha256": source_manifest["record_sha256"],
                "input_manifest_record_sha256": input_manifest["record_sha256"],
                "harnet_input_record_sha256": harnet_receipt["record_sha256"],
                "encoder_qualification_record_sha256": encoder_receipt["record_sha256"],
                "held_out_label_swap_record_sha256": label_swap_invariance["record_sha256"],
                "current_ankle_observable": int(current.sum()),
                "current_ankle_scored": int(scored_current.sum()),
                "full_history_observable": int(full.sum()),
                "full_history_scored": int(full[scoring_indices].sum()),
                "fit_schedule": [list(row) for row in FIT_SCHEDULE],
                "maximum_fit_attempts": MAXIMUM_FIT_ATTEMPTS,
                "gate_structurally_reachable": True,
                "oracle_not_used_for_features_or_predictions": True,
                "model_fits_before_this_record": 0,
            }
        )
        _write_json_create_only(output_directory / "preflight.json", prefit)

        raw_motion = {arm: controls["l9v"][:, 0].copy() for arm in CELL_ORDER}
        fit_rows = []
        fold_contracts = []
        for fold in OUTER_FOLDS:
            seed = 11 + fold
            training = scoring & current & (folds != fold)
            prediction = current & (folds == fold)
            train_indices = np.flatnonzero(training).astype(np.int64)
            prediction_indices = np.flatnonzero(prediction).astype(np.int64)
            labels = np.asarray(observable_labels[train_indices] == 0, dtype=np.int64)
            weights = participant_class_weights(
                observable_labels[train_indices], people[train_indices]
            )
            query, query_mean, query_scale = _standardized_query(energy, current, training)
            arm_matrices = {}
            arm_extra = {}
            for arm in CELL_ORDER:
                matrix, extra = _arm_features(
                    arm=arm,
                    query=query,
                    full_context=full,
                    random_embedding=random_embeddings[fold],
                    pretrained_embedding=pretrained_embeddings,
                    training_mask=training,
                )
                arm_matrices[arm] = matrix
                arm_extra[arm] = extra
            fold_contracts.append(
                {
                    "outer_fold": fold,
                    "seed": seed,
                    "training_indices_sha256": _array_sha256(train_indices),
                    "prediction_indices_sha256": _array_sha256(prediction_indices),
                    "training_weight_sha256": _array_sha256(weights),
                    "training_weight_mean": float(weights.mean()),
                    "training_binary_weight_totals": [
                        float(weights[labels == value].sum()) for value in (0, 1)
                    ],
                    "training_original_class_weight_totals": [
                        float(weights[observable_labels[train_indices] == value].sum())
                        for value in range(3)
                    ],
                    "query_mean": query_mean.tolist(),
                    "query_scale": query_scale.tolist(),
                    "random_encoder_state_sha256": random_state_hashes[fold],
                    "pretrained_encoder_state_sha256": pretrained_extractor.state_sha256,
                }
            )
            for arm in CELL_ORDER:
                attempts += 1
                attempt = attempts
                binding = _fit_binding(
                    repository_root,
                    harnet_root,
                    config,
                    code_commit,
                    str(source_manifest["record_sha256"]),
                    str(input_manifest["record_sha256"]),
                )
                try:
                    model, report = _fit_readout(
                        arm_matrices[arm][train_indices], labels, weights, seed
                    )
                    predicted = model.predict_proba(arm_matrices[arm][prediction_indices])[:, 1]
                    _require(
                        bool(np.isfinite(predicted).all())
                        and bool(np.all((predicted >= 0.0) & (predicted <= 1.0))),
                        "readout predictions are invalid",
                    )
                    raw_motion[arm][prediction_indices] = predicted
                    checkpoint = {
                        "record_kind": "fog_pretrained_logistic_checkpoint",
                        "arm": arm,
                        "outer_fold": fold,
                        "seed": seed,
                        "model": model,
                        "query_mean": query_mean,
                        "query_scale": query_scale,
                        "embedding_mean": arm_extra[arm].get("embedding_mean"),
                        "embedding_scale": arm_extra[arm].get("embedding_scale"),
                        "training_indices": train_indices,
                        "prediction_indices": prediction_indices,
                        "training_weights": weights,
                        "encoder_state_sha256": (
                            None
                            if arm in ("Q", "M")
                            else (
                                random_state_hashes[fold]
                                if arm == "R"
                                else pretrained_extractor.state_sha256
                            )
                        ),
                        **binding,
                    }
                    checkpoint_path_out = (
                        output_directory / "checkpoints" / f"fold-{fold}--{arm}.pkl"
                    )
                    _write_pickle_create_only(checkpoint_path_out, checkpoint)
                    row = {
                        "attempt_number": attempt,
                        "status": "complete",
                        "arm": arm,
                        "outer_fold": fold,
                        "seed": seed,
                        "training_rows": int(train_indices.size),
                        "prediction_rows": int(prediction_indices.size),
                        "training_original_class_counts": np.bincount(
                            observable_labels[train_indices], minlength=3
                        ).tolist(),
                        "training_binary_counts_stationary_motion": np.bincount(
                            labels, minlength=2
                        ).tolist(),
                        "checkpoint_path": checkpoint_path_out.relative_to(
                            output_directory
                        ).as_posix(),
                        "checkpoint_sha256": sha256_file(checkpoint_path_out),
                        "prediction_sha256": _array_sha256(predicted),
                        **report,
                        **binding,
                    }
                    _write_json_create_only(
                        output_directory / "fit_attempts" / f"{attempt:02d}--complete.json",
                        _sealed(row),
                    )
                    fit_rows.append(row)
                    completed += 1
                except BaseException as error:
                    _write_json_create_only(
                        output_directory / "fit_attempts" / f"{attempt:02d}--failed.json",
                        _sealed(
                            {
                                "record_kind": "fog_pretrained_fit_failure",
                                "attempt_number": attempt,
                                "arm": arm,
                                "outer_fold": fold,
                                "error_type": type(error).__name__,
                                "error": str(error),
                                "traceback": traceback.format_exc(),
                                "automatic_retry": False,
                                **binding,
                            }
                        ),
                    )
                    raise
        _require(attempts == completed == MAXIMUM_FIT_ATTEMPTS, "fit count changed")
        _write_json_create_only(
            output_directory / "fold_contracts.json",
            _sealed({"record_kind": "fog_pretrained_fold_contracts", "rows": fold_contracts}),
        )
        _write_json_create_only(
            output_directory / "fit_reports.json",
            _sealed({"record_kind": "fog_pretrained_fit_reports", "rows": fit_rows}),
        )

        probabilities = {
            arm: compose_motion_probability(
                raw_motion[arm], controls["l9v"], current, controls["b0"]
            )
            for arm in CELL_ORDER
        }
        probabilities.update(historical_motion)
        probabilities.update(controls)
        for arm in CELL_ORDER:
            q_zero = controls["l9v"][:, 1:].sum(axis=1) == 0.0
            _require(
                np.array_equal(probabilities[arm][~current], controls["b0"][~current])
                and np.array_equal(probabilities[arm][q_zero], controls["l9v"][q_zero]),
                f"{arm} fallback changed",
            )
        probability_stack = np.stack([probabilities[name] for name in METHOD_ORDER])
        raw_stack = np.stack([raw_motion[name] for name in CELL_ORDER])
        _write_npz_create_only(
            output_directory / "predictions.npz",
            method_ids=np.asarray(METHOD_ORDER, dtype=np.str_),
            arm_ids=np.asarray(CELL_ORDER, dtype=np.str_),
            observable_probabilities=probability_stack,
            observable_raw_motion_probabilities=raw_stack,
            scored_probabilities=probability_stack[:, scoring_indices],
            scored_raw_motion_probabilities=raw_stack[:, scoring_indices],
            observable_decisions=probability_stack.argmax(axis=2).astype(np.int64),
            scored_decisions=probability_stack[:, scoring_indices].argmax(axis=2).astype(np.int64),
            observable_window_ids=contexts.observable_window_ids,
            observable_participant_ids=people,
            observable_fold_index=folds,
            scoring_indices=scoring_indices,
            scored_window_ids=contexts.observable_window_ids[scoring_indices],
            scored_participant_ids=people[scoring_indices],
            scored_fold_index=folds[scoring_indices],
            scored_labels=labels_scored,
            current_availability_mask=current,
            full_context_mask=full,
        )
        reports = {
            method: extended_method_report(
                labels=labels_scored,
                probabilities=probabilities[method][scoring_indices],
                participants=people[scoring_indices],
                roster=roster,
                conditional_posture_probabilities=(
                    l9v_scored if method in (*CELL_ORDER, *HISTORICAL_MOTION_ORDER) else None
                ),
            )
            for method in METHOD_ORDER
        }
        raw_reports = {
            arm: raw_motion_report(
                labels=labels_scored[scored_current],
                motion_probability=raw_motion[arm][scoring_indices][scored_current],
                participants=people[scoring_indices][scored_current],
            )
            for arm in CELL_ORDER
        }
        analysis = analyse(
            reports=reports,
            labels=labels_scored,
            probabilities={name: probabilities[name][scoring_indices] for name in METHOD_ORDER},
            participants=people[scoring_indices],
            participant_folds=participant_folds,
            roster=roster,
            validation_complete=False,
        )
        analysis["raw_motion_reports_current_ankle_rows"] = raw_reports
        analysis["coverage"] = {
            "scored_rows": 1_213,
            "current_ankle_scored": int(scored_current.sum()),
            "full_history_scored": int(full[scoring_indices].sum()),
            "short_history_current_scored": int((scored_current & ~full[scoring_indices]).sum()),
            "missing_ankle_scored": int((~scored_current).sum()),
            "editable_q_nonzero_scored": int((scored_current & q_nonzero).sum()),
        }
        _write_json_create_only(output_directory / "analysis_prevalidation.json", _sealed(analysis))
        participant_rows = _participant_metric_rows(reports)
        _write_json_create_only(
            output_directory / "participant_metrics.json",
            _sealed(
                {"record_kind": "fog_pretrained_participant_metrics", "rows": participant_rows}
            ),
        )
        runtime = _sealed(
            {
                "record_kind": "fog_pretrained_runtime",
                "run_stage_seconds": time.perf_counter() - started,
                "fit_seconds": float(sum(float(row["fit_seconds"]) for row in fit_rows)),
                "encoder_extraction_seconds": encoder_receipt["total_extraction_seconds"],
                "fit_attempts": attempts,
                "completed_fits": completed,
                "peak_process_working_set_bytes": int(
                    getattr(psutil.Process(os.getpid()).memory_info(), "peak_wset", 0)
                ),
                "peak_cuda_allocated_bytes": max(
                    [int(pretrained_extract["peak_cuda_allocated_bytes"])]
                    + [int(row["peak_cuda_allocated_bytes"]) for row in random_extracts]
                ),
                "peak_cuda_reserved_bytes": max(
                    [int(pretrained_extract["peak_cuda_reserved_bytes"])]
                    + [int(row["peak_cuda_reserved_bytes"]) for row in random_extracts]
                ),
            }
        )
        _write_json_create_only(output_directory / "runtime_prevalidation.json", runtime)
        _write_json_create_only(
            output_directory / "result_prevalidation.json",
            _sealed(
                {
                    "record_kind": "fog_pretrained_result_prevalidation",
                    "status": "complete_awaiting_independent_replay",
                    "decision": analysis["decision"],
                    "fit_attempts": attempts,
                    "completed_fits": completed,
                    "new_encoder_parameter_fits": 0,
                    "automatic_follow_on_launched": False,
                }
            ),
        )
        return {
            "attempts": attempts,
            "completed": completed,
            "started": started,
            "analysis": analysis,
        }
    except BaseException as error:
        incomplete = output_directory / "INCOMPLETE.json"
        if not incomplete.exists():
            _write_json_create_only(
                incomplete,
                _sealed(
                    {
                        "record_kind": "fog_pretrained_incomplete",
                        "status": "incomplete",
                        "fit_attempts": attempts,
                        "completed_fits": completed,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "automatic_retry": False,
                        "automatic_follow_on_launched": False,
                    }
                ),
            )
        raise


def validate_run(
    *,
    repository_root: Path,
    evidence_root: Path,
    harnet_root: Path,
    run_directory: Path,
    code_commit: str,
) -> dict[str, Any]:
    """Zero-fit reconstruction and replay of every fitted readout."""

    started = time.perf_counter()
    config_path = repository_root / CONFIG_RELATIVE
    protocol_path = repository_root / PROTOCOL_RELATIVE
    config = _read_yaml(config_path)
    validate_config(config, config_path, protocol_path)
    validator_git = _git_state(repository_root)
    _require(validator_git["clean"] is True, "source dirty during validation")
    _verify_harnet_root(harnet_root, config)
    _require(
        sha256_file(run_directory / "config_snapshot.yaml") == CONFIG_SHA256,
        "run config snapshot changed",
    )
    (
        preflight,
        native,
        references,
        _receipt,
        _inputs,
        historical_motion,
        historical_motion_reports,
        _historical_motion_receipt,
    ) = _prepare_data(repository_root, evidence_root)
    contexts = preflight.contexts
    energy = _energy_parity(contexts, references)
    cache = _read_npz(run_directory / "context_cache.npz")
    _require(
        np.array_equal(cache["query_energy"], energy)
        and np.array_equal(cache["native_harnet_histories"], native.histories)
        and np.array_equal(cache["native_harnet_timestamps"], native.target_timestamps),
        "context cache replay changed",
    )
    device = torch.device("cuda:0")
    checkpoint_path = harnet_root / str(config["harnet"]["checkpoint_file"])
    pretrained_extractor = instantiate_harnet_extractor(
        source_root=harnet_root, seed=11, checkpoint_path=checkpoint_path
    )
    pretrained, pretrained_receipt = extract_harnet_embeddings(
        extractor=pretrained_extractor,
        contexts=native,
        device=device,
        batch_size=int(config["harnet"]["batch_size"]),
        repeat_tolerance=float(config["harnet"]["repeat_max_absolute_tolerance"]),
        batch_single_tolerance=float(config["harnet"]["batch_single_max_absolute_tolerance"]),
    )
    random = np.zeros((5, energy.shape[0], HARNET_EMBEDDING_DIMENSION), dtype=np.float32)
    random_receipts = []
    random_hashes = []
    for fold in OUTER_FOLDS:
        extractor = instantiate_harnet_extractor(
            source_root=harnet_root, seed=11 + fold, checkpoint_path=None
        )
        random_hashes.append(extractor.state_sha256)
        random[fold], receipt = extract_harnet_embeddings(
            extractor=extractor,
            contexts=native,
            device=device,
            batch_size=int(config["harnet"]["batch_size"]),
            repeat_tolerance=float(config["harnet"]["repeat_max_absolute_tolerance"]),
            batch_single_tolerance=float(config["harnet"]["batch_single_max_absolute_tolerance"]),
        )
        random_receipts.append(receipt)
    saved_embeddings = _read_npz(run_directory / "embedding_cache.npz")
    _require(
        np.array_equal(saved_embeddings["pretrained_embeddings"], pretrained)
        and np.array_equal(saved_embeddings["random_embeddings"], random),
        "embedding replay changed",
    )
    labels = np.asarray(preflight.scored_labels, dtype=np.int64)
    scoring_indices = np.asarray(preflight.scoring_indices, dtype=np.int64)
    scoring = np.asarray(preflight.scoring_eligibility, dtype=np.bool_)
    people = np.asarray(preflight.observable_participant_ids, dtype=np.str_)
    folds = np.asarray(preflight.observable_fold_index, dtype=np.int64)
    current = np.asarray(contexts.current_availability_mask, dtype=np.bool_)
    full = np.asarray(contexts.full_context_mask, dtype=np.bool_)
    observable_labels = np.zeros(people.size, dtype=np.int64)
    observable_labels[scoring_indices] = labels
    replayed_label_invariance = _sealed(
        _held_out_label_swap_invariance(
            energy=energy,
            current=current,
            full=full,
            scoring=scoring,
            folds=folds,
            observable_labels=observable_labels,
            people=people,
            random_embeddings=random,
            pretrained_embeddings=pretrained,
        )
    )
    _require(
        replayed_label_invariance
        == _read_json(run_directory / "held_out_label_swap_invariance.json"),
        "held-out label-swap invariance replay changed",
    )
    baseline_stack = np.asarray(references["reference_observable_probabilities"], dtype=np.float64)
    controls = {name: baseline_stack[index] for index, name in enumerate(REFERENCE_ORDER)}
    saved_predictions = _read_npz(run_directory / "predictions.npz")
    replay_raw = {arm: controls["l9v"][:, 0].copy() for arm in CELL_ORDER}
    checkpoint_replays = []
    maximum_difference = 0.0
    for fold, arm in FIT_SCHEDULE:
        path = run_directory / "checkpoints" / f"fold-{fold}--{arm}.pkl"
        with path.open("rb") as stream:
            checkpoint = _mapping(pickle.load(stream), str(path))
        _require(
            checkpoint["arm"] == arm
            and checkpoint["outer_fold"] == fold
            and checkpoint["code_commit"] == code_commit,
            "checkpoint identity changed",
        )
        matrix = _model_feature_matrix(checkpoint, energy, full, random[fold], pretrained)
        indices = np.asarray(checkpoint["prediction_indices"], dtype=np.int64)
        expected_indices = np.flatnonzero(current & (folds == fold)).astype(np.int64)
        _require(np.array_equal(indices, expected_indices), "checkpoint prediction support changed")
        model = cast(LogisticRegression, checkpoint["model"])
        predicted = model.predict_proba(matrix[indices])[:, 1]
        saved = np.asarray(
            saved_predictions["observable_raw_motion_probabilities"], dtype=np.float64
        )[CELL_ORDER.index(arm), indices]
        difference = float(np.max(np.abs(predicted - saved)))
        maximum_difference = max(maximum_difference, difference)
        _require(difference == 0.0, "checkpoint probability replay changed")
        replay_raw[arm][indices] = predicted
        checkpoint_replays.append(
            {"outer_fold": fold, "arm": arm, "maximum_absolute_difference": difference}
        )
    probabilities = {
        arm: compose_motion_probability(replay_raw[arm], controls["l9v"], current, controls["b0"])
        for arm in CELL_ORDER
    }
    probabilities.update(historical_motion)
    probabilities.update(controls)
    replay_stack = np.stack([probabilities[name] for name in METHOD_ORDER])
    _require(
        np.array_equal(replay_stack, saved_predictions["observable_probabilities"]),
        "composed probability replay changed",
    )
    roster = _roster(config)
    participant_folds = _participant_fold_vector(
        people[scoring_indices], folds[scoring_indices], roster
    )
    l9v_scored = controls["l9v"][scoring_indices]
    reports = {
        method: extended_method_report(
            labels=labels,
            probabilities=probabilities[method][scoring_indices],
            participants=people[scoring_indices],
            roster=roster,
            conditional_posture_probabilities=(
                l9v_scored if method in (*CELL_ORDER, *HISTORICAL_MOTION_ORDER) else None
            ),
        )
        for method in METHOD_ORDER
    }
    _require(
        all(
            canonical_json_sha256(reports[method])
            == canonical_json_sha256(historical_motion_reports[method])
            for method in HISTORICAL_MOTION_ORDER
        ),
        "historical motion report replay changed during validation",
    )
    scored_current = current[scoring_indices]
    raw_reports = {
        arm: raw_motion_report(
            labels=labels[scored_current],
            motion_probability=replay_raw[arm][scoring_indices][scored_current],
            participants=people[scoring_indices][scored_current],
        )
        for arm in CELL_ORDER
    }
    analysis = analyse(
        reports=reports,
        labels=labels,
        probabilities={name: probabilities[name][scoring_indices] for name in METHOD_ORDER},
        participants=people[scoring_indices],
        participant_folds=participant_folds,
        roster=roster,
        validation_complete=True,
    )
    analysis["raw_motion_reports_current_ankle_rows"] = raw_reports
    q_nonzero = l9v_scored[:, 1:].sum(axis=1) > 0.0
    analysis["coverage"] = {
        "scored_rows": 1_213,
        "current_ankle_scored": int(scored_current.sum()),
        "full_history_scored": int(full[scoring_indices].sum()),
        "short_history_current_scored": int((scored_current & ~full[scoring_indices]).sum()),
        "missing_ankle_scored": int((~scored_current).sum()),
        "editable_q_nonzero_scored": int((scored_current & q_nonzero).sum()),
    }
    _write_json_create_only(run_directory / "analysis.json", _sealed(analysis))
    validation = _sealed(
        {
            "record_kind": "fog_pretrained_optional_context_validation",
            "status": "validated",
            "model_fits_during_validation": 0,
            "contexts_reconstructed": True,
            "native_histories_replayed_exact": True,
            "pretrained_embedding_replayed_exact": True,
            "random_embeddings_replayed_exact": True,
            "historical_motion_probabilities_replayed_exact": True,
            "historical_motion_reports_replayed_exact": True,
            "held_out_label_swap_invariance_replayed_exact": True,
            "pretrained_extraction_receipt": pretrained_receipt,
            "random_extraction_receipts": random_receipts,
            "random_state_sha256_by_fold": random_hashes,
            "checkpoint_predictions_replayed": len(checkpoint_replays),
            "checkpoint_replays": checkpoint_replays,
            "maximum_absolute_probability_difference": maximum_difference,
            "fallback_probabilities_replayed_exact": True,
            "metric_and_gate_replay": True,
            "source_commit": code_commit,
            "run_source_commit": code_commit,
            "validator_source_commit": validator_git["commit"],
            "validation_seconds": time.perf_counter() - started,
            "InclusiveHAR_P11_P20_loaded": False,
        }
    )
    _write_json_create_only(run_directory / "validation.json", validation)
    return {"analysis": analysis, "validation": validation}


def execute(
    *,
    repository_root: Path,
    evidence_root: Path,
    harnet_root: Path,
    config_path: Path,
    protocol_path: Path,
    output_directory: Path,
    code_commit: str,
) -> dict[str, Any]:
    baseline = _worker_baseline()
    controller_started = time.perf_counter()
    attempts = 0
    completed = 0
    try:
        run = run_experiment(
            repository_root=repository_root,
            evidence_root=evidence_root,
            harnet_root=harnet_root,
            config_path=config_path,
            protocol_path=protocol_path,
            output_directory=output_directory,
            code_commit=code_commit,
        )
        attempts = int(run["attempts"])
        completed = int(run["completed"])
        validated = validate_run(
            repository_root=repository_root,
            evidence_root=evidence_root,
            harnet_root=harnet_root,
            run_directory=output_directory,
            code_commit=code_commit,
        )
        shutdown = _stop_task_owned_workers(
            baseline, attempts, completed, terminal="complete_validated"
        )
        _write_json_create_only(output_directory / "worker_shutdown.json", shutdown)
        _require(
            shutdown["task_owned_fit_workers_and_monitors_stopped"] is True,
            "task-owned workers remain",
        )
        runtime_pre = _read_json(output_directory / "runtime_prevalidation.json")
        runtime = _sealed(
            {
                "record_kind": "fog_pretrained_complete_runtime",
                "complete_controller_seconds": time.perf_counter() - controller_started,
                "fit_seconds": runtime_pre["fit_seconds"],
                "initial_encoder_extraction_seconds": runtime_pre["encoder_extraction_seconds"],
                "validation_seconds": validated["validation"]["validation_seconds"],
                "fit_attempts": attempts,
                "completed_fits": completed,
            }
        )
        _write_json_create_only(output_directory / "runtime.json", runtime)
        analysis = _mapping(validated["analysis"], "analysis")
        _write_json_create_only(
            output_directory / "result.json",
            _sealed(
                {
                    "record_kind": "fog_pretrained_optional_context_result",
                    "status": "complete_and_independently_replayed",
                    "decision": analysis["decision"],
                    "advancement": analysis["advancement"],
                    "fit_attempts": attempts,
                    "completed_fits": completed,
                    "encoder_parameter_fits": 0,
                    "confirmation_claimed": False,
                    "novelty_claimed": False,
                    "automatic_follow_on_launched": False,
                    "all_task_owned_workers_and_monitors_stopped": True,
                }
            ),
        )
        with (output_directory / "OUTCOME_SUMMARY.md").open("x", encoding="utf-8") as stream:
            stream.write(_outcome_summary(analysis))
        exclusions = {
            "artifact_manifest.json",
            "completion_manifest.json",
            "controller_final_receipt.json",
        }
        artifact_manifest = _sealed(
            {
                "record_kind": "fog_pretrained_artifact_manifest",
                "artifacts": _artifact_rows(output_directory, exclusions),
            }
        )
        _write_json_create_only(output_directory / "artifact_manifest.json", artifact_manifest)
        completion = _sealed(
            {
                "record_kind": "fog_pretrained_completion_manifest",
                "status": "complete",
                "artifacts": _artifact_rows(
                    output_directory, {"completion_manifest.json", "controller_final_receipt.json"}
                ),
                "fit_attempts": attempts,
                "completed_fits": completed,
                "validation_status": validated["validation"]["status"],
                "all_task_owned_workers_and_monitors_stopped": True,
            }
        )
        _write_json_create_only(output_directory / "completion_manifest.json", completion)
        final = _sealed(
            {
                "record_kind": "fog_pretrained_controller_final_receipt",
                "status": "complete",
                "completion_manifest_file_sha256": sha256_file(
                    output_directory / "completion_manifest.json"
                ),
                "model_fit_attempts": attempts,
                "completed_model_fits": completed,
                "model_fits_during_validation": 0,
                "automatic_follow_on_launched": False,
                "all_task_owned_workers_and_monitors_stopped": True,
                "complete_controller_seconds": runtime["complete_controller_seconds"],
            }
        )
        _write_json_create_only(output_directory / "controller_final_receipt.json", final)
        return final
    except BaseException:
        if output_directory.exists() and not (output_directory / "worker_shutdown.json").exists():
            incomplete_path = output_directory / "INCOMPLETE.json"
            if incomplete_path.is_file():
                incomplete = _read_json(incomplete_path)
                attempts = int(incomplete.get("fit_attempts", attempts))
                completed = int(incomplete.get("completed_fits", completed))
            shutdown = _stop_task_owned_workers(
                baseline, attempts, completed, terminal="incomplete"
            )
            _write_json_create_only(output_directory / "worker_shutdown.json", shutdown)
        raise


def recover_postprocessing(
    *,
    repository_root: Path,
    evidence_root: Path,
    harnet_root: Path,
    config_path: Path,
    protocol_path: Path,
    output_directory: Path,
    run_code_commit: str,
    recovery_code_commit: str,
) -> dict[str, Any]:
    """Finish an exact, zero-fit replay after the preserved participant-export failure."""

    baseline = _worker_baseline()
    started = time.perf_counter()
    repository_root = repository_root.resolve()
    evidence_root = evidence_root.resolve()
    harnet_root = harnet_root.resolve()
    output_directory = output_directory.resolve()
    _require(config_path.resolve() == repository_root / CONFIG_RELATIVE, "config path changed")
    _require(
        protocol_path.resolve() == repository_root / PROTOCOL_RELATIVE, "protocol path changed"
    )
    _require(
        output_directory.parent == evidence_root / EVIDENCE_FAMILY
        and output_directory.name == RUN_DIRECTORY_NAME
        and output_directory.is_dir(),
        "recovery run path changed",
    )
    config = _read_yaml(config_path)
    validate_config(config, config_path, protocol_path)
    git = _git_state(repository_root)
    _require(
        git["clean"] is True and git["commit"] == recovery_code_commit,
        "recovery source must be clean and exactly identified",
    )
    ancestor = subprocess.run(
        (
            "git",
            "-C",
            str(repository_root),
            "merge-base",
            "--is-ancestor",
            run_code_commit,
            recovery_code_commit,
        ),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    _require(ancestor.returncode == 0, "recovery commit does not descend from run commit")
    changed = tuple(
        name
        for name in _git_output(
            repository_root, "diff", "--name-only", f"{run_code_commit}..{recovery_code_commit}"
        ).splitlines()
        if name
    )
    allowed_changes = {
        "src/inclusive_shift_har/experiments/fog_pretrained_optional_context_run.py",
        "tests/test_fog_pretrained_optional_context_run.py",
    }
    _require(
        set(changed) <= allowed_changes
        and "src/inclusive_shift_har/experiments/fog_pretrained_optional_context_run.py" in changed,
        "recovery source change scope is not postprocessing-only",
    )
    required_existing = (
        "INCOMPLETE.json",
        "worker_shutdown.json",
        "analysis_prevalidation.json",
        "predictions.npz",
        "fit_reports.json",
        "encoder_qualification.json",
        "source_manifest.json",
    )
    _require(
        all((output_directory / name).is_file() for name in required_existing),
        "incomplete run is missing required retained evidence",
    )
    required_absent = (
        "analysis.json",
        "validation.json",
        "participant_metrics.json",
        "runtime_prevalidation.json",
        "result_prevalidation.json",
        "runtime.json",
        "result.json",
        "OUTCOME_SUMMARY.md",
        "postprocessing_recovery_intent.json",
        "postprocessing_recovery.json",
        "recovery_worker_shutdown.json",
        "artifact_manifest.json",
        "completion_manifest.json",
        "controller_final_receipt.json",
    )
    _require(
        all(not (output_directory / name).exists() for name in required_absent),
        "recovery destination already contains postprocessing artifacts",
    )
    incomplete = _read_json(output_directory / "INCOMPLETE.json")
    original_shutdown = _read_json(output_directory / "worker_shutdown.json")
    prevalidation = _read_json(output_directory / "analysis_prevalidation.json")
    fit_report = _read_json(output_directory / "fit_reports.json")
    encoder = _read_json(output_directory / "encoder_qualification.json")
    source_manifest = _read_json(output_directory / "source_manifest.json")
    for name, payload in (
        ("incomplete", incomplete),
        ("original worker shutdown", original_shutdown),
        ("prevalidation analysis", prevalidation),
        ("fit report", fit_report),
        ("encoder qualification", encoder),
        ("source manifest", source_manifest),
    ):
        _verify_sealed(payload, name)
    fit_rows = cast(list[dict[str, Any]], fit_report["rows"])
    _require(
        incomplete["status"] == "incomplete"
        and incomplete["error_type"] == "KeyError"
        and incomplete["error"] == "'participants'"
        and incomplete["fit_attempts"] == incomplete["completed_fits"] == 20
        and prevalidation["status"] == "analysis_awaiting_replay"
        and len(fit_rows) == 20
        and [int(row["attempt_number"]) for row in fit_rows] == list(range(1, 21))
        and all(row["status"] == "complete" for row in fit_rows)
        and len(list((output_directory / "checkpoints").glob("*.pkl"))) == 20
        and len(list((output_directory / "fit_attempts").glob("*--complete.json"))) == 20
        and not list((output_directory / "fit_attempts").glob("*--failed.json")),
        "retained fit completion contract changed",
    )
    _require(
        all(
            sha256_file(output_directory / str(row["checkpoint_path"])) == row["checkpoint_sha256"]
            for row in fit_rows
        ),
        "retained checkpoint hash changed",
    )
    _require(
        all(row["code_commit"] == run_code_commit for row in fit_rows),
        "retained fit source commit changed",
    )
    intent = _sealed(
        {
            "record_kind": "fog_pretrained_postprocessing_recovery_intent",
            "status": "zero_fit_replay_only",
            "original_incomplete_record_sha256": incomplete["record_sha256"],
            "run_source_commit": run_code_commit,
            "recovery_source_commit": recovery_code_commit,
            "changed_tracked_files": list(changed),
            "retained_fit_attempts": 20,
            "retained_completed_fits": 20,
            "additional_fit_budget": 0,
            "automatic_retry": False,
            "outcome_selection_or_method_change": False,
        }
    )
    _write_json_create_only(output_directory / "postprocessing_recovery_intent.json", intent)
    try:
        validated = validate_run(
            repository_root=repository_root,
            evidence_root=evidence_root,
            harnet_root=harnet_root,
            run_directory=output_directory,
            code_commit=run_code_commit,
        )
        analysis = _mapping(validated["analysis"], "analysis")
        reports = {
            method: _mapping(_mapping(analysis["reports"], "reports")[method], method)
            for method in METHOD_ORDER
        }
        _write_json_create_only(
            output_directory / "participant_metrics.json",
            _sealed(
                {
                    "record_kind": "fog_pretrained_participant_metrics",
                    "rows": _participant_metric_rows(reports),
                    "recovered_without_model_fitting": True,
                }
            ),
        )
        command = _read_json(output_directory / "command_receipt.json")
        launched = datetime.fromisoformat(str(command["launched_at_utc"]))
        stopped = datetime.fromisoformat(str(original_shutdown["observed_at_utc"]))
        original_wall_seconds = float((stopped - launched).total_seconds())
        pretrained_extract = _mapping(encoder["pretrained"], "pretrained extraction")
        random_section = _mapping(encoder["random"], "random extraction")
        random_extracts = cast(list[dict[str, Any]], random_section["extractions"])
        runtime_pre = _sealed(
            {
                "record_kind": "fog_pretrained_runtime",
                "status": "reconstructed_after_preserved_postprocessing_failure",
                "original_controller_wall_seconds_to_shutdown": original_wall_seconds,
                "fit_seconds": float(sum(float(row["fit_seconds"]) for row in fit_rows)),
                "encoder_extraction_seconds": encoder["total_extraction_seconds"],
                "fit_attempts": 20,
                "completed_fits": 20,
                "peak_process_working_set_bytes": None,
                "peak_process_working_set_limitation": (
                    "not emitted before the preserved postprocessing failure"
                ),
                "peak_cuda_allocated_bytes": max(
                    [int(pretrained_extract["peak_cuda_allocated_bytes"])]
                    + [int(row["peak_cuda_allocated_bytes"]) for row in random_extracts]
                ),
                "peak_cuda_reserved_bytes": max(
                    [int(pretrained_extract["peak_cuda_reserved_bytes"])]
                    + [int(row["peak_cuda_reserved_bytes"]) for row in random_extracts]
                ),
            }
        )
        _write_json_create_only(output_directory / "runtime_prevalidation.json", runtime_pre)
        _write_json_create_only(
            output_directory / "result_prevalidation.json",
            _sealed(
                {
                    "record_kind": "fog_pretrained_result_prevalidation",
                    "status": "retained_complete_outcomes_after_postprocessing_failure",
                    "decision": prevalidation["decision"],
                    "fit_attempts": 20,
                    "completed_fits": 20,
                    "new_encoder_parameter_fits": 0,
                    "automatic_follow_on_launched": False,
                }
            ),
        )
        shutdown = _stop_task_owned_workers(
            baseline, 20, 20, terminal="zero_fit_postprocessing_recovery_complete"
        )
        _write_json_create_only(output_directory / "recovery_worker_shutdown.json", shutdown)
        _require(
            shutdown["task_owned_fit_workers_and_monitors_stopped"] is True,
            "task-owned workers remain after recovery",
        )
        runtime = _sealed(
            {
                "record_kind": "fog_pretrained_complete_runtime",
                "status": "complete_after_zero_fit_postprocessing_recovery",
                "original_controller_wall_seconds_to_shutdown": original_wall_seconds,
                "recovery_controller_seconds": time.perf_counter() - started,
                "fit_seconds": runtime_pre["fit_seconds"],
                "initial_encoder_extraction_seconds": runtime_pre["encoder_extraction_seconds"],
                "validation_seconds": validated["validation"]["validation_seconds"],
                "fit_attempts": 20,
                "completed_fits": 20,
                "model_fits_during_recovery": 0,
            }
        )
        _write_json_create_only(output_directory / "runtime.json", runtime)
        _write_json_create_only(
            output_directory / "result.json",
            _sealed(
                {
                    "record_kind": "fog_pretrained_optional_context_result",
                    "status": "complete_and_independently_replayed_after_zero_fit_recovery",
                    "decision": analysis["decision"],
                    "advancement": analysis["advancement"],
                    "fit_attempts": 20,
                    "completed_fits": 20,
                    "encoder_parameter_fits": 0,
                    "model_fits_during_recovery": 0,
                    "original_incomplete_record_preserved": True,
                    "confirmation_claimed": False,
                    "novelty_claimed": False,
                    "automatic_follow_on_launched": False,
                    "all_task_owned_workers_and_monitors_stopped": True,
                }
            ),
        )
        with (output_directory / "OUTCOME_SUMMARY.md").open("x", encoding="utf-8") as stream:
            stream.write(_outcome_summary(analysis))
        recovery = _sealed(
            {
                "record_kind": "fog_pretrained_postprocessing_recovery",
                "status": "complete_zero_fit_replay",
                "original_incomplete_record_sha256": incomplete["record_sha256"],
                "original_worker_shutdown_record_sha256": original_shutdown["record_sha256"],
                "original_worker_shutdown_count_limitation": (
                    "the outer exception scope reported zero attempts even though the preserved "
                    "fit and incomplete records prove 20 of 20 completed"
                ),
                "run_source_commit": run_code_commit,
                "recovery_source_commit": recovery_code_commit,
                "changed_tracked_files": list(changed),
                "participant_export_schema_fix": "report.participants",
                "model_fits_during_recovery": 0,
                "checkpoint_replays": validated["validation"]["checkpoint_predictions_replayed"],
                "maximum_absolute_probability_difference": validated["validation"][
                    "maximum_absolute_probability_difference"
                ],
                "analysis_file_sha256": sha256_file(output_directory / "analysis.json"),
                "validation_file_sha256": sha256_file(output_directory / "validation.json"),
                "preserved_failure_visible": True,
                "outcome_selection_or_method_change": False,
            }
        )
        _write_json_create_only(output_directory / "postprocessing_recovery.json", recovery)
        exclusions = {
            "artifact_manifest.json",
            "completion_manifest.json",
            "controller_final_receipt.json",
        }
        artifact_manifest = _sealed(
            {
                "record_kind": "fog_pretrained_artifact_manifest",
                "artifacts": _artifact_rows(output_directory, exclusions),
            }
        )
        _write_json_create_only(output_directory / "artifact_manifest.json", artifact_manifest)
        completion = _sealed(
            {
                "record_kind": "fog_pretrained_completion_manifest",
                "status": "complete_after_zero_fit_postprocessing_recovery",
                "artifacts": _artifact_rows(
                    output_directory, {"completion_manifest.json", "controller_final_receipt.json"}
                ),
                "fit_attempts": 20,
                "completed_fits": 20,
                "model_fits_during_recovery": 0,
                "validation_status": validated["validation"]["status"],
                "original_incomplete_record_preserved": True,
                "all_task_owned_workers_and_monitors_stopped": True,
            }
        )
        _write_json_create_only(output_directory / "completion_manifest.json", completion)
        final = _sealed(
            {
                "record_kind": "fog_pretrained_controller_final_receipt",
                "status": "complete_after_zero_fit_postprocessing_recovery",
                "completion_manifest_file_sha256": sha256_file(
                    output_directory / "completion_manifest.json"
                ),
                "model_fit_attempts": 20,
                "completed_model_fits": 20,
                "model_fits_during_validation": 0,
                "model_fits_during_recovery": 0,
                "automatic_follow_on_launched": False,
                "original_incomplete_record_preserved": True,
                "all_task_owned_workers_and_monitors_stopped": True,
                "recovery_controller_seconds": runtime["recovery_controller_seconds"],
            }
        )
        _write_json_create_only(output_directory / "controller_final_receipt.json", final)
        return final
    except BaseException as error:
        if not (output_directory / "recovery_worker_shutdown.json").exists():
            shutdown = _stop_task_owned_workers(
                baseline, 20, 20, terminal="zero_fit_postprocessing_recovery_incomplete"
            )
            _write_json_create_only(output_directory / "recovery_worker_shutdown.json", shutdown)
        if not (output_directory / "RECOVERY_INCOMPLETE.json").exists():
            _write_json_create_only(
                output_directory / "RECOVERY_INCOMPLETE.json",
                _sealed(
                    {
                        "record_kind": "fog_pretrained_postprocessing_recovery_incomplete",
                        "status": "incomplete",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "model_fits_during_recovery": 0,
                        "automatic_retry": False,
                    }
                ),
            )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--harnet-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--recover-postprocessing", action="store_true")
    parser.add_argument("--recovery-commit")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.recover_postprocessing:
        _require(arguments.recovery_commit is not None, "recovery commit is required")
        result = recover_postprocessing(
            repository_root=arguments.repository_root,
            evidence_root=arguments.evidence_root,
            harnet_root=arguments.harnet_root,
            config_path=arguments.config,
            protocol_path=arguments.protocol,
            output_directory=arguments.output,
            run_code_commit=arguments.code_commit,
            recovery_code_commit=arguments.recovery_commit,
        )
    else:
        _require(arguments.recovery_commit is None, "recovery commit requires recovery mode")
        result = execute(
            repository_root=arguments.repository_root,
            evidence_root=arguments.evidence_root,
            harnet_root=arguments.harnet_root,
            config_path=arguments.config,
            protocol_path=arguments.protocol,
            output_directory=arguments.output,
            code_commit=arguments.code_commit,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
