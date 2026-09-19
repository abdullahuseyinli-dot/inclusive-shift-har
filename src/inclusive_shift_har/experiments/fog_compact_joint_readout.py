"""Execute the frozen corrected-FoG compact joint-readout experiment.

The runner consumes hash-bound development caches, fits ten train-only PCA
transforms before fifteen multinomial classifiers, and produces two probability
compositions from each classifier without refitting.  It never accesses the
InclusiveHAR target cohort or trains an encoder.
"""

# ruff: noqa: E402  -- numerical thread limits must precede third-party imports.

from __future__ import annotations

import argparse
import hashlib
import json
import os

# These must be set before NumPy or scikit-learn load a numerical runtime.  The
# effective pools are also inspected immediately before fitting and validation.
for _thread_environment_name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_environment_name] = "1"

import pickle
import platform
import subprocess
import sys
import time
import traceback
import warnings
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import psutil  # type: ignore[import-untyped]
import sklearn  # type: ignore[import-untyped]
import yaml
from numpy.typing import NDArray
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from sklearn.exceptions import ConvergenceWarning  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from threadpoolctl import threadpool_info  # type: ignore[import-untyped]

from inclusive_shift_har.experiments.fog_motion_factorization import (
    participant_class_weights,
)
from inclusive_shift_har.experiments.fog_motion_factorization_run import (
    extended_method_report,
    raw_motion_report,
)
from inclusive_shift_har.experiments.fog_pretrained_optional_context import (
    prospective_motion_headroom,
)
from inclusive_shift_har.experiments.fog_rf_feature_weight_factorial import (
    _array_sha256,
    _require,
    method_report,
    paired_comparison,
)
from inclusive_shift_har.experiments.fog_spatial_information_probe import (
    _events,
    _leave_one_fold,
)
from inclusive_shift_har.manifests.canonical import (
    canonical_json_sha256,
    sha256_file,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
StringArray: TypeAlias = NDArray[np.str_]

EXPERIMENT_ID = "fog-compact-joint-readout-v1"
CONFIG_RELATIVE = Path("configs/experiments/fog_compact_joint_readout_v1.yaml")
PROTOCOL_RELATIVE = Path("docs/research/FOG_COMPACT_JOINT_READOUT_V1_PROTOCOL.md")
SPEC_RELATIVE = Path(".audit/research_synthesis_20260908-002/NEXT_EXPERIMENT_SPEC.md")
EVIDENCE_FAMILY = Path(".audit/fog_compact_joint_readout")
RUN_DIRECTORY_NAME = "fog-compact-joint-readout-seed11-20260909-001"
OPTIONAL_RUN = Path(
    ".audit/fog_pretrained_optional_context/fog-pretrained-optional-context-seed11-20260908-001"
)
L9V_RUN = Path(".audit/fog_left_ankle_derived_nine/fog-left-ankle-derived-nine-seed11-20260908-003")
MOTION_RUN = Path(".audit/fog_motion_factorization/fog-motion-factorization-seed11-20260908-001")
REPRESENTATIONS = ("S", "R16", "P16")
OUTPUTS = ("S-fixed", "S-full", "R16-fixed", "R16-full", "P16-fixed", "P16-full")
REFERENCE_METHODS = ("l9v", "M")
METHOD_ORDER = (*OUTPUTS, *REFERENCE_METHODS)
OUTER_FOLDS = tuple(range(5))
PCA_SCHEDULE = tuple(
    (fold, representation) for fold in OUTER_FOLDS for representation in ("R16", "P16")
)
FIT_SCHEDULE = tuple(
    (fold, representation) for fold in OUTER_FOLDS for representation in REPRESENTATIONS
)
EXPECTED_TRAINING_ROWS = (940, 825, 947, 829, 1075)
EXPECTED_PCA_ROWS = (904, 780, 902, 780, 1026)
SPEC_SHA256 = "0d8817cc413e4d294f9de84583bb5fd32b7a072b7df378dbae7f26705e21f855"
PROTOCOL_SHA256 = "b5256e8268899e7ca9d0855a3c7d76fd511cae3e609dbe96e9580350adc57ec3"
INPUT_FILE_HASHES = {
    "optional_context": "d2887c6db0be5652bb223f7b97ef1e138aa99f5ce6811f89d833040d9d7216f9",
    "embeddings": "8830e4a423661134413bf97a8c7397395f12abfcbc071f042a8a6fdb1494da98",
    "optional_predictions": "670522cd9dfe4608b6f31e94b98f6f52958870dd45e13992970c733569aafe21",
    "l9v_predictions": "18db784799b42212952fc2d6644cd874a83f5e2caf132ab3c1192e15a08b73eb",
    "motion_context": "24a5819734fcbb8a14cb119f27af8b084497c95c51b6c1369c12b824d9e1185a",
}
REFERENCE_FILE_HASHES = {
    "optional_completion": "849e08d916e88594b841cb3ff1cda8dee43ee9300ff1e41d4f42ba5a79524678",
    "optional_validation": "ba6b8f68036bed6941dfba3df3d06618bc5f52c0ff1b0145ed6b1a57d1b51509",
    "optional_recovery": "5f7ed486761711b1755c048a7a79e444c5e9f0a53ab87e68a03331b680eb43f7",
    "optional_analysis": "59892f1fc3b0fcd0440c2f39f8eb8f7c14f827e851f119b2234ff091fcf0d7cb",
    "optional_fold_contracts": "83f82e3351793fbd8a3180e7f443121fc19a9d8509b7a29c96a1de6d6cd2e36a",
    "optional_partitions": "5e7a151702251edf71e0e04a2fdc164e71577bbcee6bfa5674a256b91dc59af1",
    "optional_gate_feasibility": "eb9b2c5cefd62d28b14b8ff1e9a9caa361c756cf6d236b2a752c0678402df0d8",
}
EXPECTED_TRAINING_INDEX_HASHES = (
    "ce4f6cef8a553e13145a9ecbe8afcc6da8247207b7922cf0b04f1f64985d358f",
    "15303cf535de5abea2672b00b8fb23cf9442e13172f99a8e59deec31e51492b7",
    "6a4d5d09d70d13ed2a6493a02f2c7aefa673cbd6330f6d6f4d032134540f8dce",
    "550a2f7b2e3dd27c255c6895024be8824686307f430197aeb964e87dbb222244",
    "bb97f7981f8a3ff67b47f76599fe45b7fa788535e94329a336914ae05fe7c52b",
)
EXPECTED_PREDICTION_INDEX_HASHES = (
    "ae6eb9dfb4a67dcfba49c4958f51cf5966a96293db1c1b52720cdd312370206b",
    "750e6d5b48ef5e19c666099f3d0d7bda58aec5d0d14672e4480efcc8889cf606",
    "3caee5f4000f9f4414f7c9cb6ff3eba9ff54e991adae10a10de2cf32970fc568",
    "ef6cc8005833ba9b4ead6cdd81aeccde3495ed7cbf764451b94df36a0b63e5e5",
    "5b0d1c9170272a96c6227d5edd1883fa782d9f34d9554d83196469aa2fc9f266",
)
EXPECTED_TRAINING_WEIGHT_HASHES = (
    "2968090d2f4d7860e6a21b43ff031f4907b7841615ef3bb72f5c5e3d97842b32",
    "22e9149cb58539cfc8add835271324609737098b9a357b0509764f47abbf128f",
    "78b5a3416d3df97e35cdbdda5d18d0508919852bbb734c7a24d9a57ac01b9a1e",
    "a4d71b366138469dc6c040b6416944676e6bc1711bd8336e6264a1dd1cce399b",
    "b0ec80a767e3191057f1b7f935b35de59f3259b8514a135efd05c357ff99fc98",
)
CURRENT_GRAVITY_SHA256 = "15a6cdd15b9bc66b2c6f8c8f571e0ab3fe94aed8a9ed9caed17beb24f9290247"
MAXIMUM_TOTAL_SECONDS = 1800.0


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return cast(dict[str, Any], value)


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"YAML object required: {path}")
    return cast(dict[str, Any], value)


def _read_npz(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _write_json_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")


def _write_text_create_only(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def _write_npz_create_only(path: Path, **arrays: NDArray[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]


def _write_pickle_create_only(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        pickle.dump(value, stream, protocol=5)


def _sealed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["record_sha256"] = canonical_json_sha256(result)
    return result


def _read_sealed_json(path: Path) -> dict[str, Any]:
    record = _read_json(path)
    expected = record.get("record_sha256")
    _require(isinstance(expected, str), f"sealed record hash missing: {path}")
    unhashed = dict(record)
    del unhashed["record_sha256"]
    _require(canonical_json_sha256(unhashed) == expected, f"sealed record changed: {path}")
    return record


def _threadpool_receipt() -> dict[str, Any]:
    pools = []
    for row in threadpool_info():
        pools.append(
            {
                key: row.get(key)
                for key in (
                    "user_api",
                    "internal_api",
                    "num_threads",
                    "prefix",
                    "filepath",
                    "version",
                    "threading_layer",
                    "architecture",
                )
                if key in row
            }
        )
    _require(bool(pools), "no numerical thread pool was discoverable")
    _require(
        all(int(cast(int, row["num_threads"])) == 1 for row in pools),
        "effective numerical thread count is not one",
    )
    return {
        "environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "effective_pools": pools,
        "all_effective_pool_threads_equal_one": True,
    }


def _git(repository_root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository_root), *arguments],
        text=True,
        encoding="utf-8",
    ).strip()


def _git_receipt(repository_root: Path) -> dict[str, Any]:
    status = _git(repository_root, "status", "--porcelain=v1").splitlines()
    return {
        "commit": _git(repository_root, "rev-parse", "HEAD"),
        "branch": _git(repository_root, "branch", "--show-current"),
        "status": status,
        "dirty": bool(status),
        "diff_sha256": hashlib.sha256(
            _git(repository_root, "diff", "--binary", "--no-ext-diff").encode("utf-8")
        ).hexdigest(),
        "untracked_paths": [line[3:] for line in status if line.startswith("?? ")],
    }


def _source_paths(repository_root: Path) -> tuple[Path, ...]:
    return (
        repository_root / "AGENTS.md",
        repository_root / "pyproject.toml",
        repository_root / "uv.lock",
        repository_root / CONFIG_RELATIVE,
        repository_root / PROTOCOL_RELATIVE,
        repository_root / "src/inclusive_shift_har/experiments/fog_compact_joint_readout.py",
        repository_root / "src/inclusive_shift_har/experiments/fog_motion_factorization.py",
        repository_root / "src/inclusive_shift_har/experiments/fog_motion_factorization_run.py",
        repository_root / "src/inclusive_shift_har/experiments/fog_pretrained_optional_context.py",
        repository_root / "src/inclusive_shift_har/experiments/fog_rf_feature_weight_factorial.py",
        repository_root / "src/inclusive_shift_har/experiments/fog_spatial_information_probe.py",
        repository_root / "src/inclusive_shift_har/manifests/canonical.py",
        repository_root / "tests/test_fog_compact_joint_readout.py",
    )


def _source_manifest(repository_root: Path) -> dict[str, Any]:
    files = []
    texts: dict[str, str] = {}
    for path in _source_paths(repository_root):
        _require(path.is_file(), f"source dependency missing: {path}")
        relative = path.relative_to(repository_root).as_posix()
        files.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
        if path.suffix in {".py", ".md", ".yaml", ".toml"}:
            texts[relative] = path.read_text(encoding="utf-8")
    return _sealed(
        {
            "record_kind": "fog_compact_joint_source_manifest",
            "git": _git_receipt(repository_root),
            "binding_mode": "content_addressed_files_and_text_snapshots",
            "dirty_checkout_allowed_only_with_complete_file_hash_binding": True,
            "files": files,
            "text_snapshots": texts,
        }
    )


def _deterministic_probability_fixture(features: FloatArray) -> FloatArray:
    """Exercise the probability interface without fitting or consulting labels."""

    _require(features.ndim == 2 and features.shape[1] >= 3, "fixture features changed")
    logits = np.column_stack(
        (
            np.tanh(features[:, 0]),
            np.tanh(features[:, 1]),
            np.tanh(features[:, 2]),
        )
    )
    logits -= logits.max(axis=1, keepdims=True)
    exponent = np.exp(logits)
    result = exponent / exponent.sum(axis=1, keepdims=True)
    _require(bool(np.isfinite(result).all()), "probability fixture is not finite")
    return np.asarray(result, dtype=np.float64)


def validate_config(
    config: Mapping[str, Any], config_path: Path, protocol_path: Path, specification_path: Path
) -> None:
    _require(config_path.name == CONFIG_RELATIVE.name, "config name changed")
    _require(protocol_path.name == PROTOCOL_RELATIVE.name, "protocol name changed")
    _require(config["experiment_id"] == EXPERIMENT_ID, "experiment ID changed")
    _require(config["status"] == "frozen_before_fitting", "config is not frozen")
    _require(sha256_file(protocol_path) == PROTOCOL_SHA256, "protocol bytes changed")
    _require(config["protocol_sha256"] == PROTOCOL_SHA256, "protocol binding changed")
    authority = cast(Mapping[str, Any], config["authority"])
    _require(specification_path.is_file(), "governing specification missing")
    _require(sha256_file(specification_path) == SPEC_SHA256, "governing specification changed")
    _require(authority["specification_sha256"] == SPEC_SHA256, "specification binding changed")
    _require(
        list(cast(Mapping[str, Any], config["representations"])["order"]) == list(REPRESENTATIONS),
        "representation order changed",
    )
    _require(
        list(cast(Mapping[str, Any], config["outputs"])["order"]) == list(OUTPUTS),
        "output order changed",
    )
    _require(
        cast(Mapping[str, Any], config["runtime"])["pca_fit_attempts"] == 10,
        "PCA fit count changed",
    )
    _require(
        cast(Mapping[str, Any], config["runtime"])["classifier_fit_attempts"] == 15,
        "classifier fit count changed",
    )


def _input_paths(evidence_root: Path) -> dict[str, Path]:
    return {
        "optional_context": evidence_root / OPTIONAL_RUN / "context_cache.npz",
        "embeddings": evidence_root / OPTIONAL_RUN / "embedding_cache.npz",
        "optional_predictions": evidence_root / OPTIONAL_RUN / "predictions.npz",
        "l9v_predictions": evidence_root / L9V_RUN / "predictions.npz",
        "motion_context": evidence_root / MOTION_RUN / "context_cache.npz",
        "specification": evidence_root / SPEC_RELATIVE,
        "optional_completion": evidence_root / OPTIONAL_RUN / "completion_manifest.json",
        "optional_validation": evidence_root / OPTIONAL_RUN / "validation.json",
        "optional_recovery": evidence_root / OPTIONAL_RUN / "postprocessing_recovery.json",
        "optional_analysis": evidence_root / OPTIONAL_RUN / "analysis.json",
        "optional_fold_contracts": evidence_root / OPTIONAL_RUN / "fold_contracts.json",
        "optional_partitions": evidence_root / OPTIONAL_RUN / "partition_preflight.json",
        "optional_gate_feasibility": (
            evidence_root / OPTIONAL_RUN / "prospective_gate_feasibility.json"
        ),
    }


def _verify_input_files(evidence_root: Path) -> dict[str, Any]:
    paths = _input_paths(evidence_root)
    rows = []
    for name, expected in INPUT_FILE_HASHES.items():
        path = paths[name]
        _require(path.is_file(), f"input missing: {name}")
        actual = sha256_file(path)
        _require(actual == expected, f"input hash changed: {name}")
        rows.append(
            {"name": name, "path": str(path), "sha256": actual, "size_bytes": path.stat().st_size}
        )
    for name, expected in REFERENCE_FILE_HASHES.items():
        path = paths[name]
        _require(path.is_file(), f"reference record missing: {name}")
        actual = sha256_file(path)
        _require(actual == expected, f"reference record hash changed: {name}")
        rows.append(
            {"name": name, "path": str(path), "sha256": actual, "size_bytes": path.stat().st_size}
        )
    _require(sha256_file(paths["specification"]) == SPEC_SHA256, "specification input changed")
    for relative in (
        OPTIONAL_RUN / "validation.json",
        OPTIONAL_RUN / "completion_manifest.json",
        L9V_RUN / "validation.json",
        L9V_RUN / "completion_manifest.json",
        MOTION_RUN / "validation.json",
        MOTION_RUN / "completion_manifest.json",
    ):
        record = _read_json(evidence_root / relative)
        status = str(record.get("status", "")).lower()
        _require(
            status in {"validated", "pass", "complete"}
            or status.startswith("complete_")
            or status.startswith("validation_complete_"),
            f"inherited record is not complete: {relative}",
        )
    return _sealed(
        {
            "record_kind": "fog_compact_joint_input_manifest",
            "files": rows,
            "InclusiveHAR_P11_P20_loaded": False,
        }
    )


def _method_index(method_ids: NDArray[Any], name: str) -> int:
    values = np.asarray(method_ids, dtype=np.str_).tolist()
    _require(name in values, f"missing method: {name}")
    return int(values.index(name))


def _compact_raw(
    energy: FloatArray, full: BoolArray, gravity: FloatArray, current: BoolArray
) -> tuple[FloatArray, dict[str, Any]]:
    _require(energy.shape == (1939, 2), "query energy shape changed")
    _require(full.shape == current.shape == (1939,), "support masks changed")
    _require(gravity.shape == (1939, 128, 3), "current gravity shape changed")
    norms = np.linalg.norm(gravity, axis=2)
    unit = np.divide(
        gravity, norms[:, :, None], out=np.zeros_like(gravity), where=norms[:, :, None] > 0.0
    )
    mean = unit.mean(axis=1)
    centered = unit - mean[:, None, :]
    covariance = np.einsum("nti,ntj->nij", centered, centered) / 128.0
    cov6 = np.column_stack(
        (
            covariance[:, 0, 0],
            covariance[:, 0, 1],
            covariance[:, 0, 2],
            covariance[:, 1, 1],
            covariance[:, 1, 2],
            covariance[:, 2, 2],
        )
    )
    result = np.column_stack((energy, full.astype(np.float64), mean, cov6))
    _require(
        result.shape == (1939, 12) and bool(np.isfinite(result).all()),
        "compact feature construction failed",
    )
    _require(
        bool(np.all((result[:, 2] == 0.0) | (result[:, 2] == 1.0))), "history mask feature changed"
    )
    zero_current = int(np.sum(norms[current] == 0.0))
    return result, {
        "feature_order": [
            "log_rms_linear_acceleration",
            "log_rms_gyroscope",
            "full_history_available",
            "mean_unit_gravity_x",
            "mean_unit_gravity_y",
            "mean_unit_gravity_z",
            "cov_unit_gravity_xx",
            "cov_unit_gravity_xy",
            "cov_unit_gravity_xz",
            "cov_unit_gravity_yy",
            "cov_unit_gravity_yz",
            "cov_unit_gravity_zz",
        ],
        "zero_gravity_norm_samples_current": zero_current,
        "current_gravity_sha256": _array_sha256(gravity[current]),
        "raw_compact_sha256": _array_sha256(result),
    }


def _standardize_compact(
    raw: FloatArray, training: BoolArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    continuous = np.asarray([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)
    mean = raw[training][:, continuous].mean(axis=0)
    scale = raw[training][:, continuous].std(axis=0, ddof=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    result = raw.copy()
    result[:, continuous] = (result[:, continuous] - mean) / scale
    _require(bool(np.isfinite(result).all()), "compact standardization failed")
    _require(np.array_equal(result[:, 2], raw[:, 2]), "history bit was standardized")
    return result, np.asarray(mean, dtype=np.float64), np.asarray(scale, dtype=np.float64)


def _apply_compact_standardizer(raw: FloatArray, mean: FloatArray, scale: FloatArray) -> FloatArray:
    continuous = np.asarray([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)
    _require(mean.shape == scale.shape == (11,), "compact scaler shape changed")
    result = raw.copy()
    result[:, continuous] = (result[:, continuous] - mean) / scale
    _require(bool(np.isfinite(result).all()), "compact scaler replay failed")
    _require(np.array_equal(result[:, 2], raw[:, 2]), "history bit changed in scaler replay")
    return result


def _load_inputs(evidence_root: Path) -> dict[str, Any]:
    paths = _input_paths(evidence_root)
    context = _read_npz(paths["optional_context"])
    embeddings = _read_npz(paths["embeddings"])
    optional_predictions = _read_npz(paths["optional_predictions"])
    l9v_predictions = _read_npz(paths["l9v_predictions"])
    motion = _read_npz(paths["motion_context"])
    ids = np.asarray(optional_predictions["observable_window_ids"], dtype=np.str_)
    people = np.asarray(optional_predictions["observable_participant_ids"], dtype=np.str_)
    folds = np.asarray(optional_predictions["observable_fold_index"], dtype=np.int64)
    scoring_indices = np.asarray(optional_predictions["scoring_indices"], dtype=np.int64)
    labels = np.asarray(optional_predictions["scored_labels"], dtype=np.int64)
    current = np.asarray(context["current_availability_mask"], dtype=np.bool_)
    full = np.asarray(context["full_context_mask"], dtype=np.bool_)
    _require(
        ids.shape == people.shape == folds.shape == current.shape == full.shape == (1939,),
        "observable arrays changed",
    )
    _require(scoring_indices.shape == (1213,) and labels.shape == (1213,), "scoring arrays changed")
    for arrays in (context, embeddings, motion):
        _require(np.array_equal(arrays["observable_window_ids"], ids), "cache ID alignment changed")
    _require(
        np.array_equal(embeddings["full_context_mask"], full)
        and np.array_equal(motion["current_availability_mask"], current)
        and np.array_equal(motion["full_context_mask"], full),
        "cache support alignment changed",
    )
    _require(
        np.array_equal(l9v_predictions["observable_window_ids"], ids)
        and np.array_equal(l9v_predictions["observable_participant_ids"], people)
        and np.array_equal(l9v_predictions["observable_fold_index"], folds)
        and np.array_equal(l9v_predictions["scoring_indices"], scoring_indices)
        and np.array_equal(l9v_predictions["scored_labels"], labels),
        "reference alignment changed",
    )
    optional_stack = np.asarray(optional_predictions["observable_probabilities"], dtype=np.float64)
    l9v_stack = np.asarray(l9v_predictions["observable_probabilities"], dtype=np.float64)
    l9v = l9v_stack[_method_index(l9v_predictions["method_ids"], "l9v")]
    b0 = l9v_stack[_method_index(l9v_predictions["method_ids"], "b0")]
    m_control = optional_stack[_method_index(optional_predictions["method_ids"], "M")]
    _require(
        np.array_equal(
            optional_stack[_method_index(optional_predictions["method_ids"], "l9v")], l9v
        ),
        "optional L9v changed",
    )
    _require(
        np.array_equal(optional_stack[_method_index(optional_predictions["method_ids"], "b0")], b0),
        "optional B0 changed",
    )
    _require(np.array_equal(l9v[~current], b0[~current]), "historical L9v/B0 fallback changed")
    energy = np.asarray(context["query_energy"], dtype=np.float64)
    gravity = np.asarray(motion["derived_gravity"], dtype=np.float64)[:, -128:, :]
    _require(
        _array_sha256(gravity[current]) == CURRENT_GRAVITY_SHA256,
        "qualified current gravity changed",
    )
    scoring = np.zeros(1939, dtype=np.bool_)
    scoring[scoring_indices] = True
    observable_labels = np.zeros(1939, dtype=np.int64)
    observable_labels[scoring_indices] = labels
    q_nonzero = l9v[:, 1:].sum(axis=1) > 0.0
    _require(
        int(scoring.sum()) == 1213
        and tuple(np.bincount(labels, minlength=3).tolist()) == (954, 74, 185)
        and int((scoring & current).sum()) == 1154
        and int((scoring & full).sum()) == 1098
        and int((scoring & current & ~full).sum()) == 56
        and int((scoring & ~current).sum()) == 59
        and int((scoring & current & ~q_nonzero).sum()) == 301
        and int((scoring & current & q_nonzero).sum()) == 853,
        "frozen support counts changed",
    )
    _require(
        np.asarray(embeddings["pretrained_embeddings"]).shape == (1939, 1024)
        and np.asarray(embeddings["random_embeddings"]).shape == (5, 1939, 1024),
        "embedding shapes changed",
    )
    return {
        "ids": ids,
        "people": people,
        "folds": folds,
        "scoring_indices": scoring_indices,
        "labels": labels,
        "scoring": scoring,
        "observable_labels": observable_labels,
        "current": current,
        "full": full,
        "q_nonzero": q_nonzero,
        "energy": energy,
        "gravity": gravity,
        "pretrained": np.asarray(embeddings["pretrained_embeddings"], dtype=np.float64),
        "random": np.asarray(embeddings["random_embeddings"], dtype=np.float64),
        "l9v": l9v,
        "b0": b0,
        "M": m_control,
    }


def _reference_preflight(evidence_root: Path, data: Mapping[str, Any]) -> dict[str, Any]:
    """Replay frozen controls, partitions and the feasibility oracle before fitting."""

    paths = _input_paths(evidence_root)
    prior_analysis = _read_sealed_json(paths["optional_analysis"])
    prior_folds = _read_sealed_json(paths["optional_fold_contracts"])
    prior_partitions = _read_sealed_json(paths["optional_partitions"])
    prior_headroom = _read_sealed_json(paths["optional_gate_feasibility"])
    prior_reports = cast(Mapping[str, Mapping[str, Any]], prior_analysis["reports"])
    scoring_indices = cast(IntArray, data["scoring_indices"])
    scored_labels = cast(IntArray, data["labels"])
    observable_labels = cast(IntArray, data["observable_labels"])
    people = cast(StringArray, data["people"])
    folds = cast(IntArray, data["folds"])
    scoring = cast(BoolArray, data["scoring"])
    current = cast(BoolArray, data["current"])
    q_nonzero = cast(BoolArray, data["q_nonzero"])
    roster = _roster(data)

    report_hashes: dict[str, str] = {}
    for method in REFERENCE_METHODS:
        historical_conditional = (
            cast(FloatArray, data["l9v"])[scoring_indices] if method == "M" else None
        )
        replayed = extended_method_report(
            labels=scored_labels,
            probabilities=cast(FloatArray, data[method])[scoring_indices],
            participants=people[scoring_indices],
            roster=roster,
            conditional_posture_probabilities=historical_conditional,
        )
        _require(
            canonical_json_sha256(replayed) == canonical_json_sha256(prior_reports[method]),
            f"historical {method} report parity failed",
        )
        report_hashes[method] = canonical_json_sha256(replayed)

    inherited_fold_rows = {
        int(row["outer_fold"]): row
        for row in cast(Sequence[Mapping[str, Any]], prior_folds["rows"])
    }
    inherited_partition_rows = {
        int(row["outer_fold"]): row
        for row in cast(Sequence[Mapping[str, Any]], prior_partitions["rows"])
    }
    partition_rows = []
    for fold in OUTER_FOLDS:
        training = scoring & current & (folds != fold)
        prediction = current & (folds == fold)
        scored_evaluation = scoring & prediction
        train_indices = np.flatnonzero(training).astype(np.int64)
        prediction_indices = np.flatnonzero(prediction).astype(np.int64)
        scored_indices = np.flatnonzero(scored_evaluation).astype(np.int64)
        weights = participant_class_weights(observable_labels[train_indices], people[train_indices])
        train_people = sorted(np.unique(people[train_indices]).tolist())
        held_out_people = sorted(np.unique(people[prediction_indices]).tolist())
        partition_row = inherited_partition_rows[fold]
        fold_row = inherited_fold_rows[fold]
        _require(
            np.array_equal(train_indices, np.asarray(partition_row["training_indices"]))
            and np.array_equal(prediction_indices, np.asarray(partition_row["prediction_indices"]))
            and np.array_equal(
                scored_indices, np.asarray(partition_row["scored_evaluation_indices"])
            ),
            f"inherited partition indices changed: fold {fold}",
        )
        _require(
            train_people == list(partition_row["training_participants"])
            and held_out_people == list(partition_row["held_out_participants"])
            and set(train_people).isdisjoint(held_out_people)
            and sorted((*train_people, *held_out_people)) == roster,
            f"participant partition changed: fold {fold}",
        )
        class_counts = np.bincount(observable_labels[train_indices], minlength=3).tolist()
        _require(
            class_counts == list(partition_row["training_original_class_counts"])
            and int(train_indices.size) == EXPECTED_TRAINING_ROWS[fold]
            and int(scored_indices.size) == int(partition_row["scored_evaluation_rows"]),
            f"partition counts changed: fold {fold}",
        )
        _require(
            _array_sha256(train_indices)
            == EXPECTED_TRAINING_INDEX_HASHES[fold]
            == fold_row["training_indices_sha256"]
            and _array_sha256(prediction_indices)
            == EXPECTED_PREDICTION_INDEX_HASHES[fold]
            == fold_row["prediction_indices_sha256"]
            and _array_sha256(weights)
            == EXPECTED_TRAINING_WEIGHT_HASHES[fold]
            == fold_row["training_weight_sha256"],
            f"inherited fold hash changed: fold {fold}",
        )
        partition_rows.append(
            {
                "fold": fold,
                "training_indices_sha256": _array_sha256(train_indices),
                "prediction_indices_sha256": _array_sha256(prediction_indices),
                "training_weight_sha256": _array_sha256(weights),
                "training_rows": int(train_indices.size),
                "scored_evaluation_rows": int(scored_indices.size),
                "training_class_counts": class_counts,
                "training_participants": train_people,
                "held_out_participants": held_out_people,
                "participant_sets_disjoint": True,
            }
        )

    headroom = prospective_motion_headroom(
        labels=scored_labels,
        participant_ids=people[scoring_indices],
        l9v_probabilities=cast(FloatArray, data["l9v"])[scoring_indices],
        editable_mask=(current & q_nonzero)[scoring_indices],
        roster=roster,
    )
    prior_headroom_without_hash = dict(prior_headroom)
    del prior_headroom_without_hash["record_sha256"]
    _require(headroom == prior_headroom_without_hash, "gate-feasibility replay changed")
    _require(
        headroom["editable_rows"] == 853
        and headroom["maximum_possible_strict_participant_wins"] == 17
        and float(headroom["mean_motion_only_ceiling"]) == 0.7518754751866962,
        "gate-feasibility contract changed",
    )
    return {
        "record_kind": "fog_compact_joint_reference_preflight",
        "status": "pass_before_first_estimator_fit",
        "reference_report_sha256": report_hashes,
        "partition_rows": partition_rows,
        "gate_feasibility": headroom,
        "reference_file_sha256": {
            name: REFERENCE_FILE_HASHES[name] for name in REFERENCE_FILE_HASHES
        },
        "zero_new_fits": True,
    }


def _fit_pca(
    raw_embedding: FloatArray,
    full: BoolArray,
    training: BoolArray,
    *,
    seed: int,
) -> tuple[PCA, FloatArray, FloatArray, FloatArray, FloatArray, dict[str, Any]]:
    selected = training & full
    mean = raw_embedding[selected].mean(axis=0)
    scale = raw_embedding[selected].std(axis=0, ddof=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    standardized_training = (raw_embedding[selected] - mean) / scale
    pca = PCA(n_components=16, svd_solver="full", whiten=False, random_state=seed)
    started = time.perf_counter()
    pca.fit(standardized_training)
    fit_seconds = time.perf_counter() - started
    singular_values = np.asarray(pca.singular_values_, dtype=np.float64)
    numerical_rank_tolerance = float(
        singular_values[0]
        * max(standardized_training.shape)
        * np.finfo(standardized_training.dtype).eps
    )
    _require(
        int(pca.n_components_) == 16
        and int(pca.components_.shape[0]) == 16
        and singular_values.shape == (16,)
        and bool(np.isfinite(singular_values).all())
        and float(singular_values[-1]) > numerical_rank_tolerance,
        "PCA training data lacks numerical rank 16",
    )
    coordinates = np.zeros((raw_embedding.shape[0], 16), dtype=np.float64)
    coordinates[full] = pca.transform((raw_embedding[full] - mean) / scale)
    coordinate_scale = coordinates[selected].std(axis=0, ddof=0)
    _require(
        bool(np.isfinite(coordinate_scale).all() and np.all(coordinate_scale > 0.0)),
        "PCA coordinate scale is rank deficient",
    )
    coordinates[full] /= coordinate_scale
    coordinates[~full] = 0.0
    _require(bool(np.isfinite(coordinates).all()), "PCA coordinates invalid")
    _require(bool(np.all(coordinates[~full] == 0.0)), "absent histories are not exact zero")
    return (
        pca,
        np.asarray(mean, dtype=np.float64),
        np.asarray(scale, dtype=np.float64),
        np.asarray(coordinate_scale, dtype=np.float64),
        coordinates,
        {
            "fit_seconds": fit_seconds,
            "training_rows": int(selected.sum()),
            "n_components": 16,
            "retained_numerical_rank": int(np.sum(singular_values > numerical_rank_tolerance)),
            "numerical_rank_tolerance": numerical_rank_tolerance,
            "smallest_retained_singular_value": float(singular_values[-1]),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            "singular_values": pca.singular_values_.tolist(),
            "embedding_mean_sha256": _array_sha256(np.asarray(mean, dtype=np.float64)),
            "embedding_scale_sha256": _array_sha256(np.asarray(scale, dtype=np.float64)),
            "coordinate_scale_sha256": _array_sha256(
                np.asarray(coordinate_scale, dtype=np.float64)
            ),
            "coordinate_sha256": _array_sha256(coordinates),
        },
    )


def _feature_matrix(
    compact_raw: FloatArray,
    training: BoolArray,
    representation: str,
    pca_coordinates: Mapping[tuple[int, str], FloatArray],
    fold: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    compact, mean, scale = _standardize_compact(compact_raw, training)
    if representation == "S":
        result = compact
    else:
        result = np.column_stack((compact, pca_coordinates[(fold, representation)]))
    expected = 12 if representation == "S" else 28
    _require(
        result.shape == (1939, expected) and bool(np.isfinite(result).all()),
        "feature matrix changed",
    )
    return result, mean, scale


def _assemble_feature_matrix(
    compact: FloatArray,
    representation: str,
    pca_coordinates: Mapping[tuple[int, str], FloatArray],
    fold: int,
) -> FloatArray:
    result = (
        compact
        if representation == "S"
        else np.column_stack((compact, pca_coordinates[(fold, representation)]))
    )
    expected = 12 if representation == "S" else 28
    _require(
        result.shape == (1939, expected) and bool(np.isfinite(result).all()),
        "assembled feature matrix changed",
    )
    return np.asarray(result, dtype=np.float64)


def _weighted_multiclass_nll(
    labels: IntArray, probabilities: FloatArray, weights: FloatArray
) -> float:
    selected = np.clip(probabilities[np.arange(labels.size), labels], 1e-12, 1.0)
    return float(np.sum(-np.log(selected) * weights) / np.sum(weights))


def _fit_classifier(
    features: FloatArray, labels: IntArray, weights: FloatArray, seed: int
) -> tuple[LogisticRegression, dict[str, Any]]:
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        fit_intercept=True,
        max_iter=1000,
        tol=1e-6,
        class_weight=None,
        warm_start=False,
        random_state=seed,
    )
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(features, labels, sample_weight=weights)
    seconds = time.perf_counter() - started
    convergence = [
        str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)
    ]
    _require(not convergence, "multinomial classifier did not converge")
    _require(model.classes_.tolist() == [0, 1, 2], "multinomial class order changed")
    training_probability = np.asarray(model.predict_proba(features), dtype=np.float64)
    stored = int(model.coef_.size + model.intercept_.size)
    return model, {
        "fit_seconds": seconds,
        "n_iter": [int(value) for value in model.n_iter_.tolist()],
        "stored_coefficient_count": stored,
        "coefficient_sha256": _array_sha256(np.asarray(model.coef_, dtype=np.float64)),
        "intercept_sha256": _array_sha256(np.asarray(model.intercept_, dtype=np.float64)),
        "training_weighted_multiclass_nll": _weighted_multiclass_nll(
            labels, training_probability, weights
        ),
        "convergence_warnings": convergence,
    }


def _compose(raw: Mapping[str, FloatArray], data: Mapping[str, Any]) -> dict[str, FloatArray]:
    current = cast(BoolArray, data["current"])
    q_nonzero = cast(BoolArray, data["q_nonzero"])
    l9v = cast(FloatArray, data["l9v"])
    b0 = cast(FloatArray, data["b0"])
    editable = current & q_nonzero
    stationary = l9v[:, 1:].sum(axis=1)
    q = np.zeros(1939, dtype=np.float64)
    q[q_nonzero] = l9v[q_nonzero, 1] / stationary[q_nonzero]
    output: dict[str, FloatArray] = {}
    for representation in REPRESENTATIONS:
        learned = np.asarray(raw[representation], dtype=np.float64)
        _require(
            bool(np.isfinite(learned[current]).all()),
            f"missing raw probabilities: {representation}",
        )
        full_probability = l9v.copy()
        fixed_probability = l9v.copy()
        full_probability[~current] = b0[~current]
        fixed_probability[~current] = b0[~current]
        full_probability[editable] = learned[editable]
        m = learned[editable, 0]
        fixed_probability[editable, 0] = m
        fixed_probability[editable, 1] = (1.0 - m) * q[editable]
        fixed_probability[editable, 2] = (1.0 - m) * (1.0 - q[editable])
        for name, probability in (
            (f"{representation}-fixed", fixed_probability),
            (f"{representation}-full", full_probability),
        ):
            _require(
                bool(np.isfinite(probability).all() and np.all(probability >= 0.0)),
                f"invalid {name} probabilities",
            )
            _require(
                np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-12),
                f"{name} probability sum changed",
            )
            _require(
                np.array_equal(probability[~current], b0[~current]),
                f"{name} missing fallback changed",
            )
            _require(
                np.array_equal(probability[current & ~q_nonzero], l9v[current & ~q_nonzero]),
                f"{name} q-zero fallback changed",
            )
            output[name] = probability
        _require(
            np.array_equal(full_probability[editable, 0], fixed_probability[editable, 0]),
            f"{representation} raw mobility differs",
        )
    return output


def _roster(data: Mapping[str, Any]) -> list[str]:
    people = cast(StringArray, data["people"])
    scoring = cast(BoolArray, data["scoring"])
    result = sorted(np.unique(people[scoring]).tolist())
    _require(len(result) == 22, "participant roster changed")
    return [str(value) for value in result]


def _participant_folds(data: Mapping[str, Any], roster: Sequence[str]) -> IntArray:
    people = cast(StringArray, data["people"])
    folds = cast(IntArray, data["folds"])
    scoring = cast(BoolArray, data["scoring"])
    values = []
    for person in roster:
        unique = np.unique(folds[scoring & (people == person)])
        _require(unique.size == 1, f"participant crosses folds: {person}")
        values.append(int(unique[0]))
    return np.asarray(values, dtype=np.int64)


def _same_roster(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    left_ids = [
        row["participant_id"] for row in cast(Sequence[Mapping[str, Any]], left["participants"])
    ]
    right_ids = [
        row["participant_id"] for row in cast(Sequence[Mapping[str, Any]], right["participants"])
    ]
    _require(left_ids == right_ids and len(left_ids) == 22, "paired participant order changed")


def _gate(
    comparison: Mapping[str, Any],
    leave_folds: Sequence[Mapping[str, Any]],
    minimum_gain: float,
    minimum_wins: int | None,
    validation_complete: bool,
) -> dict[str, Any]:
    interval = cast(Sequence[float], comparison["mean_difference_95_percent_bootstrap_interval"])
    recall = cast(Mapping[str, float], comparison["class_recall_differences"])
    checks = {
        "mean_gain": float(comparison["mean_difference"]) >= minimum_gain,
        "positive_bootstrap_lower_bound": float(interval[0]) > 0.0,
        "bottom_seven": float(comparison["bottom_30_percent_difference"]) >= -0.010,
        "worst_score": float(comparison["worst_participant_difference"]) >= -0.030,
        "minimum_paired_person": float(comparison["minimum_paired_participant_difference"])
        >= -0.050,
        "mobility_recall": float(recall["mobility"]) >= -0.010,
        "sitting_recall": float(recall["sitting"]) >= -0.020,
        "standing_recall": float(recall["standing"]) >= -0.020,
        "all_leave_one_person_positive": all(
            float(value) > 0.0
            for value in cast(
                Sequence[float], comparison["leave_one_participant_out_mean_differences"]
            )
        ),
        "all_leave_one_fold_positive": all(
            float(row["mean_participant_difference"]) > 0.0 for row in leave_folds
        ),
        "validation_complete": validation_complete,
    }
    if minimum_wins is not None:
        checks["participant_wins"] = int(comparison["participant_wins"]) >= minimum_wins
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "passed": int(sum(checks.values())),
        "required": len(checks),
        "minimum_gain": minimum_gain,
        "minimum_wins": minimum_wins,
    }


def _gravity_zero_diagnostics(data: Mapping[str, Any]) -> dict[str, Any]:
    gravity = cast(FloatArray, data["gravity"])
    current = cast(BoolArray, data["current"])
    scoring = cast(BoolArray, data["scoring"])
    folds = cast(IntArray, data["folds"])
    people = cast(StringArray, data["people"])
    zero = np.linalg.norm(gravity, axis=2) == 0.0
    rows = []
    for fold in OUTER_FOLDS:
        for person in sorted(np.unique(people[current & (folds == fold)]).tolist()):
            selected = current & (folds == fold) & (people == person)
            rows.append(
                {
                    "fold": fold,
                    "participant_id": person,
                    "current_windows": int(selected.sum()),
                    "scored_current_windows": int((selected & scoring).sum()),
                    "zero_norm_samples": int(zero[selected].sum()),
                    "windows_with_any_zero_norm_sample": int(np.any(zero[selected], axis=1).sum()),
                }
            )
    return {
        "zero_norm_samples_current": int(zero[current].sum()),
        "windows_with_any_zero_norm_sample_current": int(np.any(zero[current], axis=1).sum()),
        "per_person_fold": rows,
    }


def _error_diagnostics(
    all_probabilities: Mapping[str, FloatArray], data: Mapping[str, Any]
) -> dict[str, Any]:
    scoring_indices = cast(IntArray, data["scoring_indices"])
    labels = cast(IntArray, data["labels"])
    people = cast(StringArray, data["people"])[scoring_indices]
    decisions = {
        method: all_probabilities[method][scoring_indices].argmax(axis=1).astype(np.int64)
        for method in METHOD_ORDER
    }
    errors = {method: decision != labels for method, decision in decisions.items()}
    partitions: dict[str, Any] = {}
    for method in METHOD_ORDER:
        decision = decisions[method]
        error = errors[method]
        directions = []
        for true_class in range(3):
            for predicted_class in range(3):
                if true_class == predicted_class:
                    continue
                selected = (labels == true_class) & (decision == predicted_class)
                directions.append(
                    {
                        "true_class": true_class,
                        "predicted_class": predicted_class,
                        "row_count": int(selected.sum()),
                        "participant_count": int(np.unique(people[selected]).size),
                    }
                )
        partitions[method] = {
            "correct_rows": int((~error).sum()),
            "error_rows": int(error.sum()),
            "error_scoring_indices_sha256": _array_sha256(scoring_indices[error]),
            "participants_with_error": int(np.unique(people[error]).size),
            "class_directions": directions,
        }
    overlaps = {}
    for left, right in combinations(METHOD_ORDER, 2):
        left_error = errors[left]
        right_error = errors[right]
        both_error = left_error & right_error
        union_error = left_error | right_error
        overlaps[f"{left}__{right}"] = {
            "both_wrong": int(both_error.sum()),
            "left_only_wrong": int((left_error & ~right_error).sum()),
            "right_only_wrong": int((~left_error & right_error).sum()),
            "both_correct": int((~union_error).sum()),
            "error_jaccard": (
                float(both_error.sum() / union_error.sum()) if bool(union_error.any()) else 1.0
            ),
        }
    transition_pairs = {
        "P16_full_minus_l9v": ("P16-full", "l9v"),
        "P16_full_minus_M": ("P16-full", "M"),
        "P16_full_minus_S_full": ("P16-full", "S-full"),
        "P16_full_minus_R16_full": ("P16-full", "R16-full"),
        "S_full_minus_fixed": ("S-full", "S-fixed"),
        "R16_full_minus_fixed": ("R16-full", "R16-fixed"),
        "P16_full_minus_fixed": ("P16-full", "P16-fixed"),
    }
    transitions: dict[str, Any] = {}
    for name, (candidate, comparator) in transition_pairs.items():
        candidate_decision = decisions[candidate]
        comparator_decision = decisions[comparator]
        rows = []
        for true_class in range(3):
            for comparator_class in range(3):
                for candidate_class in range(3):
                    selected = (
                        (labels == true_class)
                        & (comparator_decision == comparator_class)
                        & (candidate_decision == candidate_class)
                    )
                    if not bool(selected.any()) or comparator_class == candidate_class:
                        continue
                    if comparator_class != true_class and candidate_class == true_class:
                        category = "rescue"
                    elif comparator_class == true_class and candidate_class != true_class:
                        category = "harm"
                    else:
                        category = "changed_while_both_wrong"
                    rows.append(
                        {
                            "category": category,
                            "true_class": true_class,
                            "comparator_prediction": comparator_class,
                            "candidate_prediction": candidate_class,
                            "row_count": int(selected.sum()),
                            "participant_count": int(np.unique(people[selected]).size),
                        }
                    )
        transitions[name] = rows
    return {
        "method_error_partitions": partitions,
        "pairwise_error_overlaps": overlaps,
        "class_direction_changes": transitions,
    }


def _loss_diagnostics(
    probabilities: Mapping[str, FloatArray],
    data: Mapping[str, Any],
    fit_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    scoring = cast(BoolArray, data["scoring"])
    current = cast(BoolArray, data["current"])
    folds = cast(IntArray, data["folds"])
    labels = cast(IntArray, data["observable_labels"])
    people = cast(StringArray, data["people"])
    by_representation: dict[str, Any] = {}
    for representation in REPRESENTATIONS:
        rows = sorted(
            (row for row in fit_rows if row["representation"] == representation),
            key=lambda row: int(row["fold"]),
        )
        _require(len(rows) == 5, f"loss rows changed: {representation}")
        fold_rows = []
        for row in rows:
            fold = int(row["fold"])
            selected = scoring & current & (folds == fold)
            indices = np.flatnonzero(selected).astype(np.int64)
            weights = participant_class_weights(labels[indices], people[indices])
            fold_rows.append(
                {
                    "fold": fold,
                    "training_raw_multiclass_nll": row["training_weighted_multiclass_nll"],
                    "test_raw_multiclass_nll": row["test_weighted_multiclass_nll"],
                    "test_fixed_composed_multiclass_nll": _weighted_multiclass_nll(
                        labels[indices],
                        probabilities[f"{representation}-fixed"][indices],
                        weights,
                    ),
                    "test_full_composed_multiclass_nll": _weighted_multiclass_nll(
                        labels[indices], probabilities[f"{representation}-full"][indices], weights
                    ),
                    "test_rows": int(indices.size),
                    "test_weight_sha256": _array_sha256(weights),
                }
            )
        by_representation[representation] = {
            "folds": fold_rows,
            "equal_fold_mean_training_raw_multiclass_nll": float(
                np.mean([row["training_raw_multiclass_nll"] for row in fold_rows])
            ),
            "equal_fold_mean_test_raw_multiclass_nll": float(
                np.mean([row["test_raw_multiclass_nll"] for row in fold_rows])
            ),
            "equal_fold_mean_test_fixed_composed_multiclass_nll": float(
                np.mean([row["test_fixed_composed_multiclass_nll"] for row in fold_rows])
            ),
            "equal_fold_mean_test_full_composed_multiclass_nll": float(
                np.mean([row["test_full_composed_multiclass_nll"] for row in fold_rows])
            ),
        }
    return by_representation


def _analyse(
    probabilities: Mapping[str, FloatArray],
    raw: Mapping[str, FloatArray],
    data: Mapping[str, Any],
    fit_rows: Sequence[Mapping[str, Any]],
    *,
    validation_complete: bool,
) -> dict[str, Any]:
    scoring_indices = cast(IntArray, data["scoring_indices"])
    labels = cast(IntArray, data["labels"])
    people = cast(StringArray, data["people"])[scoring_indices]
    roster = _roster(data)
    participant_folds = _participant_folds(data, roster)
    reports: dict[str, dict[str, Any]] = {}
    all_probabilities = dict(probabilities)
    all_probabilities["l9v"] = cast(FloatArray, data["l9v"])
    all_probabilities["M"] = cast(FloatArray, data["M"])
    for method in METHOD_ORDER:
        reports[method] = extended_method_report(
            labels=labels,
            probabilities=all_probabilities[method][scoring_indices],
            participants=people,
            roster=roster,
            conditional_posture_probabilities=None,
        )
    comparisons: dict[str, Any] = {}
    pairs = [
        ("P16-full", "l9v"),
        ("P16-full", "M"),
        ("P16-full", "S-full"),
        ("P16-full", "R16-full"),
        ("S-full", "S-fixed"),
        ("R16-full", "R16-fixed"),
        ("P16-full", "P16-fixed"),
    ]
    for left, right in pairs:
        _same_roster(reports[left], reports[right])
        comparisons[f"{left}_minus_{right}"] = paired_comparison(reports[left], reports[right])
    stages = (
        ("practical", "P16-full", "l9v", 0.015, 14),
        ("beyond_M", "P16-full", "M", 0.010, None),
        ("beyond_small", "P16-full", "S-full", 0.010, None),
        ("beyond_random", "P16-full", "R16-full", 0.010, None),
    )
    gates: dict[str, Any] = {}
    sequence = []
    open_claim = True
    for stage, left, right, gain, wins in stages:
        comparison = comparisons[f"{left}_minus_{right}"]
        leave = _leave_one_fold(reports[left], reports[right], participant_folds)
        gate = _gate(comparison, leave, gain, wins, validation_complete)
        gates[stage] = {"comparison": f"{left}_minus_{right}", "leave_one_fold": leave, **gate}
        sequence.append(
            {
                "stage": stage,
                "eligible": open_claim,
                "gate_status": gate["status"],
                "claim_status": gate["status"] if open_claim else "blocked_by_prior_failure",
            }
        )
        if gate["status"] != "pass":
            open_claim = False
    events = {
        name: _events(
            labels,
            all_probabilities[right][scoring_indices],
            all_probabilities[left][scoring_indices],
            people,
            roster,
        )
        for name, (left, right) in {
            "P16_full_minus_l9v": ("P16-full", "l9v"),
            "P16_full_minus_M": ("P16-full", "M"),
            "P16_full_minus_S_full": ("P16-full", "S-full"),
            "P16_full_minus_R16_full": ("P16-full", "R16-full"),
            "S_full_minus_fixed": ("S-full", "S-fixed"),
            "R16_full_minus_fixed": ("R16-full", "R16-fixed"),
            "P16_full_minus_fixed": ("P16-full", "P16-fixed"),
        }.items()
    }
    scored_current = cast(BoolArray, data["current"])[scoring_indices]
    raw_motion = {
        representation: raw_motion_report(
            labels=labels[scored_current],
            motion_probability=raw[representation][scoring_indices][scored_current, 0],
            participants=people[scored_current],
        )
        for representation in REPRESENTATIONS
    }
    within_pair = {}
    editable = (cast(BoolArray, data["current"]) & cast(BoolArray, data["q_nonzero"]))[
        scoring_indices
    ]
    for representation in REPRESENTATIONS:
        fixed = probabilities[f"{representation}-fixed"][scoring_indices]
        full = probabilities[f"{representation}-full"][scoring_indices]
        within_pair[representation] = {
            "editable_rows": int(editable.sum()),
            "maximum_absolute_raw_mobility_difference": float(
                np.max(np.abs(fixed[editable, 0] - full[editable, 0]))
            ),
            "binary_motion_report_from_shared_raw_classifier": raw_motion[representation],
            "binary_motion_metrics_fixed_full_exact": True,
            "hard_three_class_decision_changes": int(
                np.sum(fixed.argmax(axis=1) != full.argmax(axis=1))
            ),
            "mobility_decision_changes": int(
                np.sum((fixed.argmax(axis=1) == 0) != (full.argmax(axis=1) == 0))
            ),
            "conditional_posture_fixed": reports[f"{representation}-fixed"]["conditional_posture"],
            "conditional_posture_full": reports[f"{representation}-full"]["conditional_posture"],
        }
    strata_masks = {
        "current_ankle": scored_current,
        "full_history": cast(BoolArray, data["full"])[scoring_indices],
        "short_history_current": scored_current & ~cast(BoolArray, data["full"])[scoring_indices],
        "missing_ankle": ~scored_current,
        "q_zero_current": scored_current & ~cast(BoolArray, data["q_nonzero"])[scoring_indices],
        "editable": scored_current & cast(BoolArray, data["q_nonzero"])[scoring_indices],
    }
    strata: dict[str, Any] = {}
    for name, selected in strata_masks.items():
        selected_people = sorted(np.unique(people[selected]).tolist())
        strata[name] = {
            "row_count": int(selected.sum()),
            "eligible_person_count": len(selected_people),
            "reports": {
                method: method_report(
                    labels=labels[selected],
                    probabilities=all_probabilities[method][scoring_indices][selected],
                    participant_ids=people[selected],
                    roster=selected_people,
                )
                for method in METHOD_ORDER
            },
        }
    return {
        "record_kind": "fog_compact_joint_analysis",
        "status": "complete" if validation_complete else "awaiting_zero_fit_replay",
        "evidence_status": "corrected_fog_adaptive_development_not_confirmation",
        "reports": reports,
        "comparisons": comparisons,
        "gates": gates,
        "sequential_claims": sequence,
        "advancement": "pass" if all(row["claim_status"] == "pass" for row in sequence) else "fail",
        "primary_candidate": "P16-full",
        "events": events,
        "within_representation_composition": within_pair,
        "strata": strata,
        "raw_motion_reports_current_rows": raw_motion,
        "error_diagnostics": _error_diagnostics(all_probabilities, data),
        "matched_weight_loss_diagnostics": _loss_diagnostics(probabilities, data, fit_rows),
        "confirmation_claimed": False,
        "novelty_claimed": False,
        "automatic_follow_on_launched": False,
    }


def _probability_archive(
    probabilities: Mapping[str, FloatArray], raw: Mapping[str, FloatArray], data: Mapping[str, Any]
) -> dict[str, NDArray[Any]]:
    scoring_indices = cast(IntArray, data["scoring_indices"])
    stack = np.stack([probabilities[name] for name in OUTPUTS])
    raw_stack = np.stack([raw[name] for name in REPRESENTATIONS])
    return {
        "method_ids": np.asarray(OUTPUTS, dtype=np.str_),
        "representation_ids": np.asarray(REPRESENTATIONS, dtype=np.str_),
        "observable_probabilities": stack,
        "observable_raw_multinomial_probabilities": raw_stack,
        "scored_probabilities": stack[:, scoring_indices],
        "scored_raw_multinomial_probabilities": raw_stack[:, scoring_indices],
        "observable_decisions": stack.argmax(axis=2).astype(np.int64),
        "scored_decisions": stack[:, scoring_indices].argmax(axis=2).astype(np.int64),
        "observable_window_ids": cast(StringArray, data["ids"]),
        "observable_participant_ids": cast(StringArray, data["people"]),
        "observable_fold_index": cast(IntArray, data["folds"]),
        "scoring_indices": scoring_indices,
        "scored_window_ids": cast(StringArray, data["ids"])[scoring_indices],
        "scored_participant_ids": cast(StringArray, data["people"])[scoring_indices],
        "scored_fold_index": cast(IntArray, data["folds"])[scoring_indices],
        "scored_labels": cast(IntArray, data["labels"]),
        "current_availability_mask": cast(BoolArray, data["current"]),
        "full_context_mask": cast(BoolArray, data["full"]),
        "q_nonzero_mask": cast(BoolArray, data["q_nonzero"]),
    }


def _attempt_count(directory: Path, subdirectory: str, suffix: str) -> int:
    target = directory / subdirectory
    return len(list(target.glob(f"*{suffix}"))) if target.is_dir() else 0


def _deadline_check(deadline: float, stage: str) -> None:
    _require(time.perf_counter() <= deadline, f"runtime bound exceeded before {stage}")


def _record_incomplete(
    output_directory: Path, error: BaseException, stage: str, started: float
) -> None:
    if not output_directory.exists() or (output_directory / "INCOMPLETE.json").exists():
        return
    _write_json_create_only(
        output_directory / "INCOMPLETE.json",
        _sealed(
            {
                "record_kind": "fog_compact_joint_incomplete",
                "status": "incomplete",
                "stage": stage,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "completed_pca_attempt_receipts": _attempt_count(
                    output_directory, "pca_attempts", "complete.json"
                ),
                "started_pca_attempt_receipts": _attempt_count(
                    output_directory, "pca_attempts", "started.json"
                ),
                "failed_pca_attempt_receipts": _attempt_count(
                    output_directory, "pca_attempts", "failed.json"
                ),
                "completed_classifier_attempt_receipts": _attempt_count(
                    output_directory, "fit_attempts", "complete.json"
                ),
                "started_classifier_attempt_receipts": _attempt_count(
                    output_directory, "fit_attempts", "started.json"
                ),
                "failed_classifier_attempt_receipts": _attempt_count(
                    output_directory, "fit_attempts", "failed.json"
                ),
                "runtime_seconds": time.perf_counter() - started,
                "automatic_retry": False,
                "automatic_follow_on": False,
            }
        ),
    )
    children = psutil.Process(os.getpid()).children(recursive=True)
    for child in children:
        child.terminate()
    _, alive = psutil.wait_procs(children, timeout=5.0)
    for child in alive:
        child.kill()
    _, alive = psutil.wait_procs(alive, timeout=5.0)
    _write_json_create_only(
        output_directory / "incomplete_worker_shutdown.json",
        _sealed(
            {
                "record_kind": "fog_compact_joint_incomplete_worker_shutdown",
                "status": "pass" if not alive else "failed",
                "terminated_child_process_ids": [child.pid for child in children],
                "remaining_child_process_ids": [child.pid for child in alive],
                "task_owned_experiment_workers_remaining": len(alive),
                "task_owned_monitors_remaining": 0,
            }
        ),
    )


def run_experiment(
    repository_root: Path, evidence_root: Path, output_directory: Path
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    evidence_root = evidence_root.resolve()
    output_directory = output_directory.resolve()
    _require(
        output_directory.parent == evidence_root / EVIDENCE_FAMILY
        and output_directory.name == RUN_DIRECTORY_NAME,
        "output path changed",
    )
    _require(not output_directory.exists(), "create-only run directory exists")
    _require(
        Path(__file__).resolve()
        == (repository_root / "src/inclusive_shift_har/experiments/fog_compact_joint_readout.py"),
        "executing compact runner is outside the declared repository",
    )
    started = time.perf_counter()
    deadline = started + MAXIMUM_TOTAL_SECONDS
    stage = "initialization"
    output_directory.mkdir(parents=True)
    for name in ("pca_attempts", "fit_attempts", "transforms", "checkpoints"):
        (output_directory / name).mkdir()
    try:
        config_path = repository_root / CONFIG_RELATIVE
        protocol_path = repository_root / PROTOCOL_RELATIVE
        specification_path = evidence_root / SPEC_RELATIVE
        config = _read_yaml(config_path)
        validate_config(config, config_path, protocol_path, specification_path)
        thread_receipt = _threadpool_receipt()
        input_manifest = _verify_input_files(evidence_root)
        source_manifest = _source_manifest(repository_root)
        _write_json_create_only(output_directory / "source_manifest.json", source_manifest)
        _write_json_create_only(output_directory / "input_manifest.json", input_manifest)
        _write_text_create_only(
            output_directory / "config_snapshot.yaml", config_path.read_text(encoding="utf-8")
        )
        _write_json_create_only(
            output_directory / "protocol_snapshot.json",
            _sealed(
                {
                    "record_kind": "protocol_snapshot",
                    "source_sha256": PROTOCOL_SHA256,
                    "text": protocol_path.read_text(encoding="utf-8"),
                }
            ),
        )
        _write_json_create_only(
            output_directory / "specification_snapshot.json",
            _sealed(
                {
                    "record_kind": "specification_snapshot",
                    "source_sha256": SPEC_SHA256,
                    "text": specification_path.read_text(encoding="utf-8"),
                }
            ),
        )
        _write_json_create_only(
            output_directory / "environment.json",
            _sealed(
                {
                    "record_kind": "fog_compact_joint_environment",
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "scikit_learn": sklearn.__version__,
                    "PyYAML": yaml.__version__,
                    "platform": platform.platform(),
                    "process_id": os.getpid(),
                    "numerical_thread_pools": thread_receipt,
                }
            ),
        )
        _write_json_create_only(
            output_directory / "command_receipt.json",
            _sealed(
                {
                    "record_kind": "fog_compact_joint_command",
                    "argv": list(sys.argv),
                    "working_directory": str(Path.cwd()),
                    "controller_process_id": os.getpid(),
                    "controller_count": 1,
                    "launched_at_utc": _now(),
                }
            ),
        )
        stage = "feature_preflight"
        preparation_started = time.perf_counter()
        data = _load_inputs(evidence_root)
        reference_preflight = _reference_preflight(evidence_root, data)
        _write_json_create_only(
            output_directory / "reference_preflight.json", _sealed(reference_preflight)
        )
        compact_raw, compact_receipt = _compact_raw(
            cast(FloatArray, data["energy"]),
            cast(BoolArray, data["full"]),
            cast(FloatArray, data["gravity"]),
            cast(BoolArray, data["current"]),
        )
        _require(
            compact_receipt["current_gravity_sha256"] == CURRENT_GRAVITY_SHA256,
            "gravity feature input changed",
        )
        _require(
            compact_receipt["zero_gravity_norm_samples_current"] == 0,
            "qualified current gravity contains zero norms",
        )
        gravity_zero_diagnostics = _gravity_zero_diagnostics(data)
        _require(
            gravity_zero_diagnostics["zero_norm_samples_current"] == 0,
            "per-person gravity integrity changed",
        )
        compact_receipt["gravity_zero_diagnostics"] = gravity_zero_diagnostics
        _write_json_create_only(
            output_directory / "runtime_policy.json",
            _sealed(
                {
                    "record_kind": "fog_compact_joint_runtime_policy",
                    "status": "frozen_before_first_estimator_fit",
                    "maximum_total_seconds": MAXIMUM_TOTAL_SECONDS,
                    "input_reference_replay_and_compact_feature_seconds": (
                        time.perf_counter() - preparation_started
                    ),
                    "automatic_extension": False,
                }
            ),
        )
        scoring = cast(BoolArray, data["scoring"])
        current = cast(BoolArray, data["current"])
        full = cast(BoolArray, data["full"])
        folds = cast(IntArray, data["folds"])
        labels = cast(IntArray, data["observable_labels"])
        people = cast(StringArray, data["people"])
        roster = _roster(data)
        participant_folds = _participant_folds(data, roster)
        _require(set(participant_folds.tolist()) == set(OUTER_FOLDS), "outer folds changed")
        pca_coordinates: dict[tuple[int, str], FloatArray] = {}
        pca_reports = []
        stage = "all_pca_before_supervised"
        pca_phase_started = time.perf_counter()
        for attempt, (fold, representation) in enumerate(PCA_SCHEDULE, start=1):
            _deadline_check(deadline, f"PCA attempt {attempt}")
            training = scoring & current & (folds != fold)
            pca_training = training & full
            _require(
                int(training.sum()) == EXPECTED_TRAINING_ROWS[fold],
                f"classifier training rows changed: fold {fold}",
            )
            _require(
                int(pca_training.sum()) == EXPECTED_PCA_ROWS[fold],
                f"PCA training rows changed: fold {fold}",
            )
            raw_embedding = (
                cast(FloatArray, data["random"])[fold]
                if representation == "R16"
                else cast(FloatArray, data["pretrained"])
            )
            pca_train_indices = np.flatnonzero(pca_training).astype(np.int64)
            _write_json_create_only(
                output_directory / "pca_attempts" / f"{attempt:02d}--started.json",
                _sealed(
                    {
                        "record_kind": "fog_compact_joint_pca_attempt_started",
                        "attempt_number": attempt,
                        "status": "started",
                        "started_at_utc": _now(),
                        "fold": fold,
                        "representation": representation,
                        "seed": 11 + fold,
                        "training_indices_sha256": _array_sha256(pca_train_indices),
                        "standardization_input_sha256": _array_sha256(raw_embedding[pca_training]),
                        "source_manifest_record_sha256": source_manifest["record_sha256"],
                        "input_manifest_record_sha256": input_manifest["record_sha256"],
                        "protocol_sha256": PROTOCOL_SHA256,
                        "specification_sha256": SPEC_SHA256,
                        "estimator": {
                            "class": "sklearn.decomposition.PCA",
                            "n_components": 16,
                            "svd_solver": "full",
                            "whiten": False,
                        },
                    }
                ),
            )
            try:
                pca, embedding_mean, embedding_scale, coordinate_scale, coordinates, report = (
                    _fit_pca(raw_embedding, full, training, seed=11 + fold)
                )
                transform = {
                    "record_kind": "fog_compact_joint_pca_transform",
                    "fold": fold,
                    "representation": representation,
                    "pca": pca,
                    "embedding_mean": embedding_mean,
                    "embedding_scale": embedding_scale,
                    "coordinate_scale": coordinate_scale,
                    "training_indices": pca_train_indices,
                    "source_manifest_record_sha256": source_manifest["record_sha256"],
                    "input_manifest_record_sha256": input_manifest["record_sha256"],
                }
                path = output_directory / "transforms" / f"fold-{fold}--{representation}.pkl"
                _write_pickle_create_only(path, transform)
                row = {
                    "record_kind": "fog_compact_joint_pca_attempt",
                    "attempt_number": attempt,
                    "status": "complete",
                    "completed_at_utc": _now(),
                    "fold": fold,
                    "representation": representation,
                    "seed": 11 + fold,
                    "training_indices_sha256": _array_sha256(pca_train_indices),
                    "source_manifest_record_sha256": source_manifest["record_sha256"],
                    "input_manifest_record_sha256": input_manifest["record_sha256"],
                    "checkpoint": {
                        "path": path.relative_to(output_directory).as_posix(),
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    },
                    **report,
                }
                _write_json_create_only(
                    output_directory / "pca_attempts" / f"{attempt:02d}--complete.json",
                    _sealed(row),
                )
                pca_reports.append(row)
                pca_coordinates[(fold, representation)] = coordinates
            except BaseException as error:
                _write_json_create_only(
                    output_directory / "pca_attempts" / f"{attempt:02d}--failed.json",
                    _sealed(
                        {
                            "record_kind": "fog_compact_joint_pca_failure",
                            "attempt_number": attempt,
                            "status": "failed",
                            "failed_at_utc": _now(),
                            "fold": fold,
                            "representation": representation,
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                            "automatic_retry": False,
                        }
                    ),
                )
                raise
        _require(
            len(pca_reports) == 10
            and _attempt_count(output_directory, "pca_attempts", "started.json") == 10
            and _attempt_count(output_directory, "fit_attempts", "started.json") == 0,
            "all PCA transforms must precede supervised fits",
        )
        pca_phase_seconds = time.perf_counter() - pca_phase_started
        post_pca_feature_started = time.perf_counter()
        feature_matrices: dict[tuple[int, str], FloatArray] = {}
        compact_scalers: dict[int, tuple[FloatArray, FloatArray]] = {}
        fold_rows = []
        for fold in OUTER_FOLDS:
            training = scoring & current & (folds != fold)
            prediction = current & (folds == fold)
            compact, mean, scale = _standardize_compact(compact_raw, training)
            compact_scalers[fold] = (mean, scale)
            for representation in REPRESENTATIONS:
                feature_matrices[(fold, representation)] = _assemble_feature_matrix(
                    compact, representation, pca_coordinates, fold
                )
            train_indices = np.flatnonzero(training).astype(np.int64)
            predict_indices = np.flatnonzero(prediction).astype(np.int64)
            weights = participant_class_weights(labels[train_indices], people[train_indices])
            _require(
                _array_sha256(train_indices) == EXPECTED_TRAINING_INDEX_HASHES[fold]
                and _array_sha256(predict_indices) == EXPECTED_PREDICTION_INDEX_HASHES[fold]
                and _array_sha256(weights) == EXPECTED_TRAINING_WEIGHT_HASHES[fold],
                f"sealed training contract changed: fold {fold}",
            )
            participant_class_rows = []
            for person in sorted(np.unique(people[train_indices]).tolist()):
                person_selected = people[train_indices] == person
                for class_index in range(3):
                    selected = person_selected & (labels[train_indices] == class_index)
                    if bool(selected.any()):
                        participant_class_rows.append(
                            {
                                "participant_id": person,
                                "class_index": class_index,
                                "row_count": int(selected.sum()),
                                "weight_total": float(weights[selected].sum()),
                            }
                        )
            fold_rows.append(
                {
                    "fold": fold,
                    "training_rows": int(train_indices.size),
                    "pca_training_rows": int((training & full).sum()),
                    "prediction_rows": int(predict_indices.size),
                    "training_indices_sha256": _array_sha256(train_indices),
                    "prediction_indices_sha256": _array_sha256(predict_indices),
                    "training_weight_sha256": _array_sha256(weights),
                    "training_class_counts": np.bincount(
                        labels[train_indices], minlength=3
                    ).tolist(),
                    "training_weight_mean": float(weights.mean()),
                    "training_weight_sum": float(weights.sum()),
                    "participant_class_weight_rows": participant_class_rows,
                    "training_participants": sorted(np.unique(people[train_indices]).tolist()),
                    "held_out_participants": sorted(np.unique(people[predict_indices]).tolist()),
                    "compact_mean": compact_scalers[fold][0].tolist(),
                    "compact_scale": compact_scalers[fold][1].tolist(),
                    "feature_sha256": {
                        representation: _array_sha256(feature_matrices[(fold, representation)])
                        for representation in REPRESENTATIONS
                    },
                }
            )
        _write_npz_create_only(
            output_directory / "feature_cache.npz",
            compact_raw=compact_raw,
            fold_S=np.stack([feature_matrices[(fold, "S")] for fold in OUTER_FOLDS]),
            fold_R16=np.stack([feature_matrices[(fold, "R16")] for fold in OUTER_FOLDS]),
            fold_P16=np.stack([feature_matrices[(fold, "P16")] for fold in OUTER_FOLDS]),
            observable_window_ids=cast(StringArray, data["ids"]),
            current_availability_mask=current,
            full_context_mask=full,
        )
        _write_json_create_only(
            output_directory / "pca_reports.json",
            _sealed({"record_kind": "fog_compact_joint_pca_reports", "rows": pca_reports}),
        )
        _write_json_create_only(
            output_directory / "fold_contracts.json",
            _sealed({"record_kind": "fog_compact_joint_fold_contracts", "rows": fold_rows}),
        )
        label_swap_rows = []
        fixture_raw = {
            representation: np.full((1939, 3), np.nan, dtype=np.float64)
            for representation in REPRESENTATIONS
        }
        all_altered = labels.copy()
        for fold in OUTER_FOLDS:
            altered = labels.copy()
            held_out = scoring & current & (folds == fold)
            altered[held_out] = (altered[held_out] + 1) % 3
            all_altered[held_out] = altered[held_out]
            training = scoring & current & (folds != fold)
            train_indices = np.flatnonzero(training).astype(np.int64)
            prediction_indices = np.flatnonzero(current & (folds == fold)).astype(np.int64)
            _require(
                np.array_equal(labels[train_indices], altered[train_indices]),
                "held-out label swap changed training targets",
            )
            weights_a = participant_class_weights(labels[train_indices], people[train_indices])
            weights_b = participant_class_weights(altered[train_indices], people[train_indices])
            _require(
                np.array_equal(weights_a, weights_b), "held-out label swap changed training weights"
            )
            replayed_feature_hashes = {}
            fixture_hashes = {}
            transform_hashes = {}
            replayed_compact = _apply_compact_standardizer(
                compact_raw, compact_scalers[fold][0], compact_scalers[fold][1]
            )
            for representation in REPRESENTATIONS:
                replayed = _assemble_feature_matrix(
                    replayed_compact, representation, pca_coordinates, fold
                )
                original = feature_matrices[(fold, representation)]
                _require(
                    np.array_equal(replayed, original),
                    f"held-out labels changed features: {fold}/{representation}",
                )
                fixture = _deterministic_probability_fixture(replayed[prediction_indices])
                fixture_repeat = _deterministic_probability_fixture(replayed[prediction_indices])
                _require(
                    np.array_equal(fixture, fixture_repeat),
                    f"probability fixture changed: {fold}/{representation}",
                )
                fixture_raw[representation][prediction_indices] = fixture
                replayed_feature_hashes[representation] = _array_sha256(replayed)
                fixture_hashes[representation] = _array_sha256(fixture)
                if representation != "S":
                    transform_path = (
                        output_directory / "transforms" / f"fold-{fold}--{representation}.pkl"
                    )
                    transform_hashes[representation] = sha256_file(transform_path)
            label_swap_rows.append(
                {
                    "fold": fold,
                    "swapped_held_out_scored_current_labels": int(held_out.sum()),
                    "training_labels_sha256": _array_sha256(labels[train_indices]),
                    "training_weights_sha256": _array_sha256(weights_a),
                    "feature_matrix_sha256": replayed_feature_hashes,
                    "transform_checkpoint_sha256": transform_hashes,
                    "deterministic_probability_fixture_sha256": fixture_hashes,
                    "fit_inputs_exact_after_held_out_label_swap": True,
                    "prediction_interface_accepts_no_labels": True,
                }
            )
        fixture_data = dict(data)
        fixture_data["observable_labels"] = all_altered
        fixture_data["labels"] = all_altered[cast(IntArray, data["scoring_indices"])]
        fixture_original = _compose(fixture_raw, data)
        fixture_altered = _compose(fixture_raw, fixture_data)
        fixture_output_hashes = {}
        for name in OUTPUTS:
            _require(
                np.array_equal(fixture_original[name], fixture_altered[name]),
                f"held-out labels entered probability composition: {name}",
            )
            fixture_output_hashes[name] = _array_sha256(fixture_original[name])
        _write_json_create_only(
            output_directory / "held_out_label_swap_invariance.json",
            _sealed(
                {
                    "record_kind": "fog_compact_joint_label_swap_invariance",
                    "status": "pass_before_first_supervised_fit",
                    "training_targets_and_weights_unchanged": True,
                    "feature_and_transform_hashes_label_independent": True,
                    "deterministic_probability_path_fixture_passed": True,
                    "rows": label_swap_rows,
                    "fixture_output_sha256": fixture_output_hashes,
                    "extra_supervised_fits": 0,
                }
            ),
        )
        _write_json_create_only(
            output_directory / "preflight.json",
            _sealed(
                {
                    "record_kind": "fog_compact_joint_preflight",
                    "status": "pass_before_first_supervised_fit",
                    "model_fits_before_record": 0,
                    "pca_fits_before_record": 10,
                    "all_pca_transforms_qualified": True,
                    "compact_feature_receipt": compact_receipt,
                    "fold_contract_sha256": canonical_json_sha256(fold_rows),
                    "fit_schedule": [list(row) for row in FIT_SCHEDULE],
                    "maximum_classifier_fit_attempts": 15,
                    "normalizer_accounting": {
                        "compact_outer_training_estimates": 5,
                        "embedding_outer_training_estimates": 10,
                        "pca_coordinate_scale_estimates": 10,
                        "held_out_label_invariance_estimates": 0,
                        "all_unweighted_population_statistics": True,
                    },
                    "gate_reachability": {
                        **cast(Mapping[str, Any], reference_preflight["gate_feasibility"]),
                        "required_strict_participant_wins": 14,
                        "reachable": (
                            cast(Mapping[str, Any], reference_preflight["gate_feasibility"])[
                                "maximum_possible_strict_participant_wins"
                            ]
                            >= 14
                        ),
                        "replayed_before_first_estimator_fit": True,
                        "not_used_for_features_weights_or_predictions": True,
                    },
                    "preparation_seconds": time.perf_counter() - preparation_started,
                }
            ),
        )
        post_pca_feature_seconds = time.perf_counter() - post_pca_feature_started
        stage = "supervised_fitting"
        raw = {
            representation: np.full((1939, 3), np.nan, dtype=np.float64)
            for representation in REPRESENTATIONS
        }
        fit_rows = []
        classifier_phase_started = time.perf_counter()
        for attempt, (fold, representation) in enumerate(FIT_SCHEDULE, start=1):
            _deadline_check(deadline, f"classifier attempt {attempt}")
            training = scoring & current & (folds != fold)
            prediction = current & (folds == fold)
            train_indices = np.flatnonzero(training).astype(np.int64)
            prediction_indices = np.flatnonzero(prediction).astype(np.int64)
            weights = participant_class_weights(labels[train_indices], people[train_indices])
            features = feature_matrices[(fold, representation)]
            _write_json_create_only(
                output_directory / "fit_attempts" / f"{attempt:02d}--started.json",
                _sealed(
                    {
                        "record_kind": "fog_compact_joint_fit_attempt_started",
                        "attempt_number": attempt,
                        "status": "started",
                        "started_at_utc": _now(),
                        "fold": fold,
                        "representation": representation,
                        "seed": 11 + fold,
                        "training_indices_sha256": _array_sha256(train_indices),
                        "prediction_indices_sha256": _array_sha256(prediction_indices),
                        "training_labels_sha256": _array_sha256(labels[train_indices]),
                        "training_weight_sha256": _array_sha256(weights),
                        "feature_sha256": _array_sha256(features),
                        "source_manifest_record_sha256": source_manifest["record_sha256"],
                        "input_manifest_record_sha256": input_manifest["record_sha256"],
                        "protocol_sha256": PROTOCOL_SHA256,
                        "specification_sha256": SPEC_SHA256,
                        "estimator": {
                            "class": "sklearn.linear_model.LogisticRegression",
                            "penalty": "l2",
                            "C": 1.0,
                            "solver": "lbfgs",
                            "fit_intercept": True,
                            "max_iter": 1000,
                            "tol": 1e-6,
                            "class_weight": None,
                            "warm_start": False,
                        },
                    }
                ),
            )
            try:
                model, report = _fit_classifier(
                    features[train_indices], labels[train_indices], weights, 11 + fold
                )
                predicted = np.asarray(
                    model.predict_proba(features[prediction_indices]), dtype=np.float64
                )
                _require(
                    predicted.shape == (prediction_indices.size, 3), "prediction shape changed"
                )
                raw[representation][prediction_indices] = predicted
                test_indices = np.flatnonzero(scoring & current & (folds == fold)).astype(np.int64)
                test_weights = participant_class_weights(labels[test_indices], people[test_indices])
                test_probability = np.asarray(
                    model.predict_proba(features[test_indices]), dtype=np.float64
                )
                checkpoint = {
                    "record_kind": "fog_compact_joint_classifier_checkpoint",
                    "fold": fold,
                    "representation": representation,
                    "seed": 11 + fold,
                    "model": model,
                    "training_indices": train_indices,
                    "prediction_indices": prediction_indices,
                    "training_weights": weights,
                    "compact_mean": compact_scalers[fold][0],
                    "compact_scale": compact_scalers[fold][1],
                    "feature_sha256": _array_sha256(features),
                    "source_manifest_record_sha256": source_manifest["record_sha256"],
                    "input_manifest_record_sha256": input_manifest["record_sha256"],
                }
                path = output_directory / "checkpoints" / f"fold-{fold}--{representation}.pkl"
                _write_pickle_create_only(path, checkpoint)
                row = {
                    "record_kind": "fog_compact_joint_fit_attempt",
                    "attempt_number": attempt,
                    "status": "complete",
                    "completed_at_utc": _now(),
                    "fold": fold,
                    "representation": representation,
                    "seed": 11 + fold,
                    "training_rows": int(train_indices.size),
                    "prediction_rows": int(prediction_indices.size),
                    "training_indices_sha256": _array_sha256(train_indices),
                    "prediction_indices_sha256": _array_sha256(prediction_indices),
                    "training_weight_sha256": _array_sha256(weights),
                    "test_weight_sha256": _array_sha256(test_weights),
                    "test_weighted_multiclass_nll": _weighted_multiclass_nll(
                        labels[test_indices], test_probability, test_weights
                    ),
                    "prediction_sha256": _array_sha256(predicted),
                    "checkpoint": {
                        "path": path.relative_to(output_directory).as_posix(),
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    },
                    **report,
                }
                _write_json_create_only(
                    output_directory / "fit_attempts" / f"{attempt:02d}--complete.json",
                    _sealed(row),
                )
                fit_rows.append(row)
            except BaseException as error:
                _write_json_create_only(
                    output_directory / "fit_attempts" / f"{attempt:02d}--failed.json",
                    _sealed(
                        {
                            "record_kind": "fog_compact_joint_fit_failure",
                            "attempt_number": attempt,
                            "status": "failed",
                            "failed_at_utc": _now(),
                            "fold": fold,
                            "representation": representation,
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                            "automatic_retry": False,
                        }
                    ),
                )
                raise
        _require(len(fit_rows) == 15, "supervised fit count changed")
        _require(
            _attempt_count(output_directory, "fit_attempts", "started.json") == 15
            and _attempt_count(output_directory, "fit_attempts", "complete.json") == 15
            and _attempt_count(output_directory, "fit_attempts", "failed.json") == 0,
            "supervised attempt receipts changed",
        )
        classifier_phase_seconds = time.perf_counter() - classifier_phase_started
        probabilities = _compose(raw, data)
        altered_data = dict(data)
        altered_data["observable_labels"] = all_altered
        altered_data["labels"] = all_altered[cast(IntArray, data["scoring_indices"])]
        altered_probabilities = _compose(raw, altered_data)
        _require(
            all(
                np.array_equal(probabilities[name], altered_probabilities[name]) for name in OUTPUTS
            ),
            "post-fit predictions changed under held-out label replacement",
        )
        _write_json_create_only(
            output_directory / "postfit_label_swap_invariance.json",
            _sealed(
                {
                    "record_kind": "fog_compact_joint_postfit_label_swap_invariance",
                    "status": "pass_without_refitting",
                    "raw_probability_sha256": {
                        name: _array_sha256(raw[name]) for name in REPRESENTATIONS
                    },
                    "output_probability_sha256": {
                        name: _array_sha256(probabilities[name]) for name in OUTPUTS
                    },
                    "model_fits_for_check": 0,
                    "pca_fits_for_check": 0,
                }
            ),
        )
        archive = _probability_archive(probabilities, raw, data)
        _write_npz_create_only(output_directory / "predictions.npz", **archive)
        analysis_started = time.perf_counter()
        analysis = _analyse(probabilities, raw, data, fit_rows, validation_complete=False)
        _write_json_create_only(output_directory / "analysis_prevalidation.json", _sealed(analysis))
        _write_json_create_only(
            output_directory / "fit_reports.json",
            _sealed({"record_kind": "fog_compact_joint_fit_reports", "rows": fit_rows}),
        )
        runtime = {
            "record_kind": "fog_compact_joint_runtime_prevalidation",
            "run_stage_seconds": time.perf_counter() - started,
            "input_reference_and_compact_feature_seconds": float(
                cast(Mapping[str, Any], _read_json(output_directory / "runtime_policy.json"))[
                    "input_reference_replay_and_compact_feature_seconds"
                ]
            ),
            "pca_and_embedding_normalization_wall_seconds": pca_phase_seconds,
            "pca_fit_seconds": float(sum(float(row["fit_seconds"]) for row in pca_reports)),
            "post_pca_feature_and_invariance_seconds": post_pca_feature_seconds,
            "classifier_loop_wall_seconds": classifier_phase_seconds,
            "classifier_fit_seconds": float(sum(float(row["fit_seconds"]) for row in fit_rows)),
            "analysis_and_prevalidation_artifact_seconds": time.perf_counter() - analysis_started,
            "pca_fit_attempts": 10,
            "completed_pca_fits": 10,
            "classifier_fit_attempts": 15,
            "completed_classifier_fits": 15,
            "encoder_fits": 0,
            "peak_process_working_set_bytes": int(
                getattr(psutil.Process(os.getpid()).memory_info(), "peak_wset", 0)
            ),
        }
        _write_json_create_only(output_directory / "runtime_prevalidation.json", _sealed(runtime))
        _write_json_create_only(
            output_directory / "result_prevalidation.json",
            _sealed(
                {
                    "record_kind": "fog_compact_joint_result_prevalidation",
                    "status": "complete_awaiting_zero_fit_replay",
                    "primary_candidate": "P16-full",
                    "advancement_before_validation": analysis["advancement"],
                    "pca_fits": 10,
                    "classifier_fits": 15,
                    "encoder_fits": 0,
                    "automatic_follow_on_launched": False,
                }
            ),
        )
        return {
            "status": "complete_awaiting_zero_fit_replay",
            "output_directory": str(output_directory),
            "pca_fits": 10,
            "classifier_fits": 15,
        }
    except BaseException as error:
        _record_incomplete(output_directory, error, stage, started)
        raise


def _validated_attempt_rows(
    output_directory: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pca_summary = _read_sealed_json(output_directory / "pca_reports.json")
    fit_summary = _read_sealed_json(output_directory / "fit_reports.json")
    pca_rows = [dict(row) for row in cast(Sequence[Mapping[str, Any]], pca_summary["rows"])]
    fit_rows = [dict(row) for row in cast(Sequence[Mapping[str, Any]], fit_summary["rows"])]
    _require(len(pca_rows) == len(PCA_SCHEDULE) == 10, "PCA receipt count changed")
    _require(len(fit_rows) == len(FIT_SCHEDULE) == 15, "fit receipt count changed")
    for subdirectory, schedule, rows in (
        ("pca_attempts", PCA_SCHEDULE, pca_rows),
        ("fit_attempts", FIT_SCHEDULE, fit_rows),
    ):
        _require(
            _attempt_count(output_directory, subdirectory, "started.json") == len(schedule)
            and _attempt_count(output_directory, subdirectory, "complete.json") == len(schedule)
            and _attempt_count(output_directory, subdirectory, "failed.json") == 0,
            f"terminal attempt accounting changed: {subdirectory}",
        )
        for attempt, ((fold, representation), summary_row) in enumerate(
            zip(schedule, rows, strict=True), start=1
        ):
            started = _read_sealed_json(
                output_directory / subdirectory / f"{attempt:02d}--started.json"
            )
            completed = _read_sealed_json(
                output_directory / subdirectory / f"{attempt:02d}--complete.json"
            )
            completed_without_hash = dict(completed)
            del completed_without_hash["record_sha256"]
            _require(completed_without_hash == summary_row, f"summary receipt changed: {attempt}")
            _require(
                started["attempt_number"] == completed["attempt_number"] == attempt
                and started["fold"] == completed["fold"] == fold
                and started["representation"] == completed["representation"] == representation
                and started["status"] == "started"
                and completed["status"] == "complete",
                f"attempt identity changed: {subdirectory}/{attempt}",
            )
            checkpoint = cast(Mapping[str, Any], completed["checkpoint"])
            checkpoint_path = output_directory / str(checkpoint["path"])
            _require(
                checkpoint_path.is_file()
                and checkpoint_path.stat().st_size == checkpoint["size_bytes"]
                and sha256_file(checkpoint_path) == checkpoint["sha256"],
                f"attempt checkpoint changed: {subdirectory}/{attempt}",
            )
    return pca_rows, fit_rows


def _reconstruct_features(
    output_directory: Path,
    data: Mapping[str, Any],
    pca_rows: Sequence[Mapping[str, Any]],
    fit_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], FloatArray]:
    compact_raw, _ = _compact_raw(
        cast(FloatArray, data["energy"]),
        cast(BoolArray, data["full"]),
        cast(FloatArray, data["gravity"]),
        cast(BoolArray, data["current"]),
    )
    full = cast(BoolArray, data["full"])
    result: dict[tuple[int, str], FloatArray] = {}
    pca_by_identity = {(int(row["fold"]), str(row["representation"])): row for row in pca_rows}
    fit_by_identity = {(int(row["fold"]), str(row["representation"])): row for row in fit_rows}
    for fold in OUTER_FOLDS:
        for representation in REPRESENTATIONS:
            checkpoint_path = (
                output_directory / "checkpoints" / f"fold-{fold}--{representation}.pkl"
            )
            fit_receipt = fit_by_identity[(fold, representation)]
            checkpoint_receipt = cast(Mapping[str, Any], fit_receipt["checkpoint"])
            _require(
                str(checkpoint_receipt["path"])
                == checkpoint_path.relative_to(output_directory).as_posix()
                and sha256_file(checkpoint_path) == checkpoint_receipt["sha256"],
                f"classifier receipt changed: {fold}/{representation}",
            )
            with checkpoint_path.open("rb") as stream:
                checkpoint = cast(dict[str, Any], pickle.load(stream))
            mean = np.asarray(checkpoint["compact_mean"], dtype=np.float64)
            scale = np.asarray(checkpoint["compact_scale"], dtype=np.float64)
            continuous = np.asarray([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)
            compact = compact_raw.copy()
            compact[:, continuous] = (compact[:, continuous] - mean) / scale
            if representation == "S":
                matrix = compact
            else:
                transform_path = (
                    output_directory / "transforms" / f"fold-{fold}--{representation}.pkl"
                )
                pca_receipt = pca_by_identity[(fold, representation)]
                transform_receipt = cast(Mapping[str, Any], pca_receipt["checkpoint"])
                _require(
                    str(transform_receipt["path"])
                    == transform_path.relative_to(output_directory).as_posix()
                    and sha256_file(transform_path) == transform_receipt["sha256"],
                    f"transform receipt changed: {fold}/{representation}",
                )
                with transform_path.open("rb") as stream:
                    transform = cast(dict[str, Any], pickle.load(stream))
                training = (
                    cast(BoolArray, data["scoring"])
                    & cast(BoolArray, data["current"])
                    & (cast(IntArray, data["folds"]) != fold)
                    & full
                )
                expected_training = np.flatnonzero(training).astype(np.int64)
                _require(
                    transform["fold"] == fold
                    and transform["representation"] == representation
                    and np.array_equal(transform["training_indices"], expected_training)
                    and _array_sha256(expected_training) == pca_receipt["training_indices_sha256"]
                    and transform["source_manifest_record_sha256"]
                    == pca_receipt["source_manifest_record_sha256"]
                    and transform["input_manifest_record_sha256"]
                    == pca_receipt["input_manifest_record_sha256"],
                    f"transform training contract changed: {fold}/{representation}",
                )
                raw_embedding = (
                    cast(FloatArray, data["random"])[fold]
                    if representation == "R16"
                    else cast(FloatArray, data["pretrained"])
                )
                coordinates = np.zeros((1939, 16), dtype=np.float64)
                standardized = (
                    raw_embedding[full] - np.asarray(transform["embedding_mean"])
                ) / np.asarray(transform["embedding_scale"])
                pca = cast(PCA, transform["pca"])
                singular_values = np.asarray(pca.singular_values_, dtype=np.float64)
                _require(
                    int(pca.n_components_) == 16
                    and pca.svd_solver == "full"
                    and not bool(pca.whiten)
                    and singular_values.shape == (16,)
                    and float(singular_values[-1]) > float(pca_receipt["numerical_rank_tolerance"]),
                    f"saved PCA qualification changed: {fold}/{representation}",
                )
                coordinates[full] = pca.transform(standardized) / np.asarray(
                    transform["coordinate_scale"]
                )
                coordinates[~full] = 0.0
                matrix = np.column_stack((compact, coordinates))
            _require(
                _array_sha256(matrix) == checkpoint["feature_sha256"],
                f"feature replay changed: {fold}/{representation}",
            )
            result[(fold, representation)] = matrix
    return result


def _manifest_files(output_directory: Path) -> list[dict[str, Any]]:
    exclusions = {"artifact_manifest.json", "completion_manifest.json"}
    rows = []
    for path in sorted(output_directory.rglob("*")):
        if path.is_file() and path.relative_to(output_directory).as_posix() not in exclusions:
            rows.append(
                {
                    "path": path.relative_to(output_directory).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return rows


def _summary(analysis: Mapping[str, Any]) -> str:
    reports = cast(Mapping[str, Mapping[str, Any]], analysis["reports"])
    comparisons = cast(Mapping[str, Mapping[str, Any]], analysis["comparisons"])
    lines = [
        "# Compact joint-readout outcome",
        "",
        "Evidence: corrected-FoG adaptive development; not independent confirmation.",
        "",
        "| Method | Equal-person fixed-class macro-F1 | Pooled accuracy |",
        "|---|---:|---:|",
    ]
    for method in METHOD_ORDER:
        report = reports[method]
        primary = cast(Mapping[str, Any], report["primary"])
        pooled = cast(Mapping[str, Any], report["pooled"])
        lines.append(
            f"| {method} | {float(primary['mean_participant_macro_f1']):.6%} | {float(pooled['accuracy']):.6%} |"
        )
    lines += [
        "",
        "## Predeclared comparisons",
        "",
        "| Contrast | Difference | 95% participant bootstrap | Wins / harms / ties |",
        "|---|---:|---:|---:|",
    ]
    for name, comparison in comparisons.items():
        interval = cast(
            Sequence[float], comparison["mean_difference_95_percent_bootstrap_interval"]
        )
        lines.append(
            f"| {name} | {float(comparison['mean_difference']):+.6%} | "
            f"[{float(interval[0]):+.6%}, {float(interval[1]):+.6%}] | "
            f"{comparison['participant_wins']} / {comparison['participant_harms']} / {comparison['participant_ties']} |"
        )
    lines += ["", "## Claim sequence", ""]
    for row in cast(Sequence[Mapping[str, Any]], analysis["sequential_claims"]):
        lines.append(f"- {row['stage']}: {row['claim_status']} (gate {row['gate_status']})")
    lines += [
        "",
        f"Overall advancement: **{analysis['advancement']}**.",
        "",
        "All six outputs, participant harms, fallbacks, strata, transforms, checkpoints, runtime and zero-fit replay are retained. No automatic follow-on was launched.",
        "",
    ]
    return "\n".join(lines)


def _validate_run_once(
    repository_root: Path, evidence_root: Path, output_directory: Path
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    evidence_root = evidence_root.resolve()
    output_directory = output_directory.resolve()
    _require(output_directory.is_dir(), "run directory missing")
    _require(not (output_directory / "validation.json").exists(), "validation already exists")
    started = time.perf_counter()
    pre_runtime = _read_sealed_json(output_directory / "runtime_prevalidation.json")
    remaining_seconds = MAXIMUM_TOTAL_SECONDS - float(pre_runtime["run_stage_seconds"])
    _require(remaining_seconds > 0.0, "runtime bound exhausted before validation")
    deadline = started + remaining_seconds
    thread_receipt = _threadpool_receipt()
    source_manifest = _read_sealed_json(output_directory / "source_manifest.json")
    saved_input_manifest = _read_sealed_json(output_directory / "input_manifest.json")
    fresh_input_manifest = _verify_input_files(evidence_root)
    _require(
        fresh_input_manifest["record_sha256"] == saved_input_manifest["record_sha256"],
        "bound inherited inputs changed before validation",
    )
    fresh_source_manifest = _source_manifest(repository_root)
    _require(
        fresh_source_manifest["record_sha256"] == source_manifest["record_sha256"],
        "executing source manifest changed before validation",
    )
    for row in cast(Sequence[Mapping[str, Any]], source_manifest["files"]):
        path = repository_root / str(row["path"])
        _require(
            path.is_file() and sha256_file(path) == row["sha256"],
            f"bound source changed: {row['path']}",
        )
    _deadline_check(deadline, "attempt receipt authentication")
    pca_rows, fit_rows = _validated_attempt_rows(output_directory)
    data = _load_inputs(evidence_root)
    replayed_reference = _reference_preflight(evidence_root, data)
    saved_reference = _read_sealed_json(output_directory / "reference_preflight.json")
    replayed_reference_sealed = _sealed(replayed_reference)
    _require(
        replayed_reference_sealed["record_sha256"] == saved_reference["record_sha256"],
        "reference preflight replay changed",
    )
    _deadline_check(deadline, "feature and transform replay")
    features = _reconstruct_features(output_directory, data, pca_rows, fit_rows)
    feature_cache = _read_npz(output_directory / "feature_cache.npz")
    compact_replay, compact_replay_receipt = _compact_raw(
        cast(FloatArray, data["energy"]),
        cast(BoolArray, data["full"]),
        cast(FloatArray, data["gravity"]),
        cast(BoolArray, data["current"]),
    )
    _require(
        np.array_equal(feature_cache["compact_raw"], compact_replay)
        and compact_replay_receipt["current_gravity_sha256"] == CURRENT_GRAVITY_SHA256
        and np.array_equal(feature_cache["observable_window_ids"], data["ids"])
        and np.array_equal(feature_cache["current_availability_mask"], data["current"])
        and np.array_equal(feature_cache["full_context_mask"], data["full"]),
        "feature cache identity changed",
    )
    for representation in REPRESENTATIONS:
        _require(
            np.array_equal(
                feature_cache[f"fold_{representation}"],
                np.stack([features[(fold, representation)] for fold in OUTER_FOLDS]),
            ),
            f"feature cache replay changed: {representation}",
        )
    fold_contracts = _read_sealed_json(output_directory / "fold_contracts.json")
    fold_rows = cast(Sequence[Mapping[str, Any]], fold_contracts["rows"])
    _require(len(fold_rows) == 5, "fold contract count changed")
    for fold, row in enumerate(fold_rows):
        _require(
            row["fold"] == fold
            and row["training_indices_sha256"] == EXPECTED_TRAINING_INDEX_HASHES[fold]
            and row["prediction_indices_sha256"] == EXPECTED_PREDICTION_INDEX_HASHES[fold]
            and row["training_weight_sha256"] == EXPECTED_TRAINING_WEIGHT_HASHES[fold]
            and all(
                row["feature_sha256"][representation]
                == _array_sha256(features[(fold, representation)])
                for representation in REPRESENTATIONS
            ),
            f"fold contract replay changed: {fold}",
        )
    label_swap = _read_sealed_json(output_directory / "held_out_label_swap_invariance.json")
    _require(
        label_swap["status"] == "pass_before_first_supervised_fit"
        and label_swap["training_targets_and_weights_unchanged"] is True
        and label_swap["feature_and_transform_hashes_label_independent"] is True
        and label_swap["deterministic_probability_path_fixture_passed"] is True
        and len(cast(Sequence[Any], label_swap["rows"])) == 5,
        "pre-fit label-swap evidence changed",
    )
    folds = cast(IntArray, data["folds"])
    current = cast(BoolArray, data["current"])
    raw = {
        representation: np.full((1939, 3), np.nan, dtype=np.float64)
        for representation in REPRESENTATIONS
    }
    checkpoint_rows = []
    fit_by_identity = {(int(row["fold"]), str(row["representation"])): row for row in fit_rows}
    for fold, representation in FIT_SCHEDULE:
        _deadline_check(deadline, f"checkpoint replay {fold}/{representation}")
        fit_receipt = fit_by_identity[(fold, representation)]
        path = output_directory / "checkpoints" / f"fold-{fold}--{representation}.pkl"
        path_receipt = cast(Mapping[str, Any], fit_receipt["checkpoint"])
        _require(
            sha256_file(path) == path_receipt["sha256"],
            f"checkpoint authentication changed: {fold}/{representation}",
        )
        with path.open("rb") as stream:
            checkpoint = cast(dict[str, Any], pickle.load(stream))
        _require(
            checkpoint["fold"] == fold and checkpoint["representation"] == representation,
            "checkpoint identity changed",
        )
        prediction_indices = np.asarray(checkpoint["prediction_indices"], dtype=np.int64)
        expected = np.flatnonzero(current & (folds == fold)).astype(np.int64)
        _require(
            np.array_equal(prediction_indices, expected), "checkpoint prediction support changed"
        )
        training = cast(BoolArray, data["scoring"]) & current & (folds != fold)
        training_indices = np.flatnonzero(training).astype(np.int64)
        expected_weights = participant_class_weights(
            cast(IntArray, data["observable_labels"])[training_indices],
            cast(StringArray, data["people"])[training_indices],
        )
        _require(
            np.array_equal(checkpoint["training_indices"], training_indices)
            and np.array_equal(checkpoint["training_weights"], expected_weights)
            and _array_sha256(training_indices) == fit_receipt["training_indices_sha256"]
            and _array_sha256(prediction_indices) == fit_receipt["prediction_indices_sha256"]
            and _array_sha256(expected_weights) == fit_receipt["training_weight_sha256"]
            and checkpoint["source_manifest_record_sha256"] == source_manifest["record_sha256"]
            and checkpoint["input_manifest_record_sha256"] == saved_input_manifest["record_sha256"],
            f"checkpoint training contract changed: {fold}/{representation}",
        )
        model = cast(LogisticRegression, checkpoint["model"])
        params = model.get_params(deep=False)
        _require(
            model.classes_.tolist() == [0, 1, 2]
            and params["penalty"] == "l2"
            and params["C"] == 1.0
            and params["solver"] == "lbfgs"
            and params["fit_intercept"] is True
            and params["max_iter"] == 1000
            and params["tol"] == 1e-6
            and params["class_weight"] is None
            and params["warm_start"] is False
            and params["random_state"] == 11 + fold
            and _array_sha256(np.asarray(model.coef_, dtype=np.float64))
            == fit_receipt["coefficient_sha256"]
            and _array_sha256(np.asarray(model.intercept_, dtype=np.float64))
            == fit_receipt["intercept_sha256"],
            f"classifier state changed: {fold}/{representation}",
        )
        expected_stored = 39 if representation == "S" else 87
        _require(
            int(model.coef_.size + model.intercept_.size)
            == expected_stored
            == fit_receipt["stored_coefficient_count"],
            f"stored coefficient count changed: {fold}/{representation}",
        )
        predicted = np.asarray(
            model.predict_proba(features[(fold, representation)][prediction_indices]),
            dtype=np.float64,
        )
        _require(
            _array_sha256(predicted) == fit_receipt["prediction_sha256"],
            f"saved prediction receipt changed: {fold}/{representation}",
        )
        raw[representation][prediction_indices] = predicted
        checkpoint_rows.append(
            {
                "fold": fold,
                "representation": representation,
                "checkpoint_sha256": sha256_file(path),
                "prediction_sha256": _array_sha256(predicted),
            }
        )
    probabilities = _compose(raw, data)
    postfit_swap = _read_sealed_json(output_directory / "postfit_label_swap_invariance.json")
    _require(
        postfit_swap["status"] == "pass_without_refitting"
        and all(
            postfit_swap["raw_probability_sha256"][name] == _array_sha256(raw[name])
            for name in REPRESENTATIONS
        )
        and all(
            postfit_swap["output_probability_sha256"][name] == _array_sha256(probabilities[name])
            for name in OUTPUTS
        ),
        "post-fit label-swap receipt changed",
    )
    archive = _read_npz(output_directory / "predictions.npz")
    expected_archive = _probability_archive(probabilities, raw, data)
    maximum_probability_difference = 0.0
    for name, expected in expected_archive.items():
        actual = archive[name]
        if np.issubdtype(np.asarray(expected).dtype, np.floating):
            difference = float(np.nanmax(np.abs(np.asarray(actual) - np.asarray(expected))))
            maximum_probability_difference = max(maximum_probability_difference, difference)
            _require(
                np.array_equal(actual, expected, equal_nan=True),
                f"prediction replay changed: {name}",
            )
        else:
            _require(np.array_equal(actual, expected), f"prediction replay changed: {name}")
    altered_labels = cast(IntArray, data["observable_labels"]).copy()
    altered_labels[cast(BoolArray, data["scoring"]) & current] = (
        altered_labels[cast(BoolArray, data["scoring"]) & current] + 1
    ) % 3
    altered_data = dict(data)
    altered_data["observable_labels"] = altered_labels
    altered_data["labels"] = altered_labels[cast(IntArray, data["scoring_indices"])]
    altered_probabilities = _compose(raw, altered_data)
    _require(
        all(np.array_equal(probabilities[name], altered_probabilities[name]) for name in OUTPUTS),
        "validation label replacement changed prediction replay",
    )
    _deadline_check(deadline, "metric replay")
    analysis = _analyse(probabilities, raw, data, fit_rows, validation_complete=True)
    _write_json_create_only(output_directory / "analysis.json", _sealed(analysis))
    participant_rows = {
        method: cast(Mapping[str, Any], analysis["reports"])[method]["participants"]
        for method in METHOD_ORDER
    }
    _write_json_create_only(
        output_directory / "participant_metrics.json",
        _sealed({"record_kind": "fog_compact_joint_participant_metrics", "rows": participant_rows}),
    )
    validation = _sealed(
        {
            "record_kind": "fog_compact_joint_validation",
            "status": "validated",
            "validated_at_utc": _now(),
            "model_fits_during_validation": 0,
            "pca_fits_during_validation": 0,
            "encoder_fits_during_validation": 0,
            "checkpoint_replays": 15,
            "transform_replays": 10,
            "pca_attempts_started": 10,
            "pca_attempts_completed": 10,
            "classifier_attempts_started": 15,
            "classifier_attempts_completed": 15,
            "maximum_probability_replay_difference": maximum_probability_difference,
            "probability_replay_exact": True,
            "participant_metric_replay_complete": True,
            "all_outputs_actual_conditional_posture_metrics": True,
            "within_pair_raw_mobility_exact": all(
                cast(Mapping[str, Any], analysis["within_representation_composition"])[
                    representation
                ]["maximum_absolute_raw_mobility_difference"]
                == 0.0
                for representation in REPRESENTATIONS
            ),
            "fallback_replay_exact": True,
            "postfit_held_out_label_swap_prediction_invariance_exact": True,
            "reference_preflight_replayed_exact": True,
            "effective_numerical_thread_pools": thread_receipt,
            "checkpoint_rows": checkpoint_rows,
            "InclusiveHAR_P11_P20_loaded": False,
        }
    )
    _write_json_create_only(output_directory / "validation.json", validation)
    validation_seconds = time.perf_counter() - started
    _require(
        float(pre_runtime["run_stage_seconds"]) + validation_seconds <= MAXIMUM_TOTAL_SECONDS,
        "complete experiment runtime bound exceeded",
    )
    runtime = _sealed(
        {
            "record_kind": "fog_compact_joint_runtime",
            **{
                key: value
                for key, value in pre_runtime.items()
                if key not in {"record_kind", "record_sha256"}
            },
            "validation_seconds": validation_seconds,
            "total_recorded_stage_seconds": float(pre_runtime["run_stage_seconds"])
            + validation_seconds,
        }
    )
    _write_json_create_only(output_directory / "runtime.json", runtime)
    reports = cast(Mapping[str, Mapping[str, Any]], analysis["reports"])
    primary = cast(Mapping[str, Any], reports["P16-full"]["primary"])
    result = _sealed(
        {
            "record_kind": "fog_compact_joint_result",
            "status": "complete",
            "evidence_status": analysis["evidence_status"],
            "primary_candidate": "P16-full",
            "primary_mean_participant_macro_f1": primary["mean_participant_macro_f1"],
            "advancement": analysis["advancement"],
            "sequential_claims": analysis["sequential_claims"],
            "pca_fits": 10,
            "classifier_fits": 15,
            "encoder_fits": 0,
            "validation_model_fits": 0,
            "automatic_follow_on_launched": False,
            "confirmation_claimed": False,
            "novelty_claimed": False,
        }
    )
    _write_json_create_only(output_directory / "result.json", result)
    _write_text_create_only(output_directory / "OUTCOME_SUMMARY.md", _summary(analysis))
    children = psutil.Process(os.getpid()).children(recursive=True)
    _require(not children, "task-owned child process remains")
    shutdown = _sealed(
        {
            "record_kind": "fog_compact_joint_worker_shutdown",
            "status": "pass",
            "controller_process_id": os.getpid(),
            "remaining_child_processes": [],
            "task_owned_experiment_workers_remaining": 0,
            "task_owned_monitors_remaining": 0,
        }
    )
    _write_json_create_only(output_directory / "worker_shutdown.json", shutdown)
    files = _manifest_files(output_directory)
    manifest = _sealed(
        {
            "record_kind": "fog_compact_joint_artifact_manifest",
            "status": "complete",
            "root_relative_exclusions": ["artifact_manifest.json", "completion_manifest.json"],
            "artifacts": files,
            "artifact_count": len(files),
        }
    )
    _write_json_create_only(output_directory / "artifact_manifest.json", manifest)
    completion = _sealed(
        {
            "record_kind": "fog_compact_joint_completion",
            "status": "complete",
            "result_record_sha256": result["record_sha256"],
            "validation_record_sha256": validation["record_sha256"],
            "artifact_manifest_record_sha256": manifest["record_sha256"],
            "artifact_manifest_sha256": sha256_file(output_directory / "artifact_manifest.json"),
            "worker_shutdown_record_sha256": shutdown["record_sha256"],
            "pca_fits": 10,
            "classifier_fits": 15,
            "encoder_fits": 0,
            "automatic_follow_on_launched": False,
        }
    )
    _write_json_create_only(output_directory / "completion_manifest.json", completion)
    return {
        "status": "complete",
        "output_directory": str(output_directory),
        "primary_mean_participant_macro_f1": primary["mean_participant_macro_f1"],
        "advancement": analysis["advancement"],
        "pca_fits": 10,
        "classifier_fits": 15,
        "validation_fits": 0,
    }


def _verify_completed_package(output_directory: Path) -> dict[str, Any]:
    completion = _read_sealed_json(output_directory / "completion_manifest.json")
    manifest = _read_sealed_json(output_directory / "artifact_manifest.json")
    result = _read_sealed_json(output_directory / "result.json")
    validation = _read_sealed_json(output_directory / "validation.json")
    shutdown = _read_sealed_json(output_directory / "worker_shutdown.json")
    _require(
        completion["status"] == "complete"
        and manifest["status"] == "complete"
        and result["status"] == "complete"
        and validation["status"] == "validated"
        and shutdown["status"] == "pass",
        "completed package status changed",
    )
    _require(
        completion["result_record_sha256"] == result["record_sha256"]
        and completion["validation_record_sha256"] == validation["record_sha256"]
        and completion["artifact_manifest_record_sha256"] == manifest["record_sha256"]
        and completion["artifact_manifest_sha256"]
        == sha256_file(output_directory / "artifact_manifest.json")
        and completion["worker_shutdown_record_sha256"] == shutdown["record_sha256"],
        "completion binding changed",
    )
    listed_paths = set()
    for row in cast(Sequence[Mapping[str, Any]], manifest["artifacts"]):
        relative = Path(str(row["path"])).as_posix()
        _require(relative not in listed_paths, f"duplicate manifest path: {relative}")
        path = (output_directory / relative).resolve()
        _require(
            path.is_relative_to(output_directory)
            and path.is_file()
            and path.stat().st_size == row["size_bytes"]
            and sha256_file(path) == row["sha256"],
            f"manifest artifact changed: {relative}",
        )
        listed_paths.add(relative)
    actual_paths = {
        path.relative_to(output_directory).as_posix()
        for path in output_directory.rglob("*")
        if path.is_file()
        and path.relative_to(output_directory).as_posix()
        not in {"artifact_manifest.json", "completion_manifest.json"}
    }
    _require(listed_paths == actual_paths, "completed package file coverage changed")
    return {
        "status": "complete_read_only_revalidation",
        "output_directory": str(output_directory),
        "primary_mean_participant_macro_f1": result["primary_mean_participant_macro_f1"],
        "advancement": result["advancement"],
        "pca_fits": completion["pca_fits"],
        "classifier_fits": completion["classifier_fits"],
        "validation_fits": 0,
    }


def validate_run(
    repository_root: Path, evidence_root: Path, output_directory: Path
) -> dict[str, Any]:
    output_directory = output_directory.resolve()
    if (output_directory / "completion_manifest.json").is_file():
        return _verify_completed_package(output_directory)
    started = time.perf_counter()
    try:
        return _validate_run_once(repository_root, evidence_root, output_directory)
    except BaseException as error:
        children = psutil.Process(os.getpid()).children(recursive=True)
        for child in children:
            child.terminate()
        _, alive = psutil.wait_procs(children, timeout=5.0)
        for child in alive:
            child.kill()
        _, alive = psutil.wait_procs(alive, timeout=5.0)
        if (
            output_directory.is_dir()
            and not (output_directory / "VALIDATION_INCOMPLETE.json").exists()
        ):
            _write_json_create_only(
                output_directory / "VALIDATION_INCOMPLETE.json",
                _sealed(
                    {
                        "record_kind": "fog_compact_joint_validation_incomplete",
                        "status": "incomplete",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "runtime_seconds": time.perf_counter() - started,
                        "model_fits_during_validation": 0,
                        "pca_fits_during_validation": 0,
                        "automatic_refit": False,
                    }
                ),
            )
        if (
            output_directory.is_dir()
            and not (output_directory / "validation_worker_shutdown.json").exists()
        ):
            _write_json_create_only(
                output_directory / "validation_worker_shutdown.json",
                _sealed(
                    {
                        "record_kind": "fog_compact_joint_validation_worker_shutdown",
                        "status": "pass" if not alive else "failed",
                        "terminated_child_process_ids": [child.pid for child in children],
                        "remaining_child_process_ids": [child.pid for child in alive],
                        "task_owned_experiment_workers_remaining": len(alive),
                    }
                ),
            )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "validate"))
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.action == "run":
        result = run_experiment(
            arguments.repository_root, arguments.evidence_root, arguments.output_directory
        )
    else:
        result = validate_run(
            arguments.repository_root, arguments.evidence_root, arguments.output_directory
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
