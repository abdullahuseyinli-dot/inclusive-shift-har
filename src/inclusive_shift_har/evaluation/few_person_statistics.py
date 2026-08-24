"""Fail-closed progress validation and participant-level few-person statistics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from inclusive_shift_har.calibration import TemperatureCalibrator
from inclusive_shift_har.evaluation._strict_config import require_utc_timestamp
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.evaluation.source_calibration import (
    load_source_temperature_calibrator_file,
)
from inclusive_shift_har.evaluation.statistics import (
    holm_adjust,
    paired_participant_comparison,
)
from inclusive_shift_har.experiments.postconfirmatory_cache import (
    PRIMARY_CACHE_EVIDENCE_STATUS,
    PRIMARY_CACHE_RECORD_KIND,
    PRIMARY_CACHE_SCHEMA_VERSION,
)
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)
from inclusive_shift_har.protocols.few_person import FewPersonProtocolError
from inclusive_shift_har.protocols.few_person_v1_1 import (
    FEW_PERSON_V1_1_EVIDENCE_STATUS,
    FEW_PERSON_V1_1_PROTOCOL_ID,
    FUNCTIONAL_CORE_CLASS_ORDER,
    validate_few_person_v1_1_manifest_assignments,
)

FEW_PERSON_STATISTICS_SCHEMA_VERSION = "1.0.0"
EXPECTED_MODEL_IDS: tuple[str, ...] = (
    "compact-coral",
    "compact-dann",
    "compact-erm",
    "deepconvlstm",
    "legacy-bilstm",
    "legacy-cnn1d",
    "legacy-joint-cnn-bilstm",
    "more-har-augmentation",
    "more-har-backbone",
    "more-har-content",
    "more-har-factorized",
    "more-har-full",
    "more-har-full-no-accelerometer",
    "more-har-full-no-gyroscope",
    "more-har-groupdro",
    "static-dual-branch-matched",
)
EXPECTED_SEEDS: tuple[int, ...] = (11, 23, 47, 89, 131)
EXPECTED_FOLDS: tuple[str, ...] = tuple(f"target_outer_{index:02d}" for index in range(1, 6))
EXPECTED_K_VALUES: tuple[int, ...] = (1, 2, 4)
EXPECTED_TARGET_SUBJECTS: frozenset[str] = frozenset(str(value) for value in range(11, 21))
EXPECTED_SOURCE_NORMALIZATION_SUBJECTS: tuple[str, ...] = (
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "9",
)
EXPECTED_CELL_COUNT = 16 * 5 * 5 * 3
_RUN_DIRECTORY_PATTERN = re.compile(
    r".+--seed-(?:11|23|47|89|131)--target_outer_0[1-5]--k(?:1|2|4)(?:--attempt-\d+)?$"
)
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class FewPersonStatisticsError(RuntimeError):
    """Raised when v1.1 evidence is incomplete, changed, or misaligned."""


@dataclass(frozen=True, order=True)
class CellKey:
    model_id: str
    seed: int
    fold_id: str
    k: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "seed": self.seed,
            "fold_id": self.fold_id,
            "k": self.k,
        }

    @property
    def cell_id(self) -> str:
        return f"{self.model_id}--seed-{self.seed}--{self.fold_id}--k{self.k}"


@dataclass(frozen=True)
class _Plan:
    record: Mapping[str, Any]
    manifest_sha256: str
    manifest_file_sha256: str
    class_names: tuple[str, ...]
    scenarios: Mapping[tuple[str, int], Mapping[str, Any]]
    models: Mapping[tuple[str, int], Mapping[str, Any]]
    expected_cells: frozenset[CellKey]


@dataclass(frozen=True)
class _Completed:
    key: CellKey
    record_path: Path
    record_sha256: str
    record_file_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    prediction_path: Path
    prediction_sha256: str
    code_commit: str
    environment_sha256: str
    window_ids: tuple[str, ...]
    participant_ids: tuple[str, ...]
    labels: NDArray[np.int64]
    probabilities: NDArray[np.float64]
    participant_macro_f1: Mapping[str, float]


@dataclass(frozen=True)
class _Scan:
    plan: _Plan
    completed: Mapping[CellKey, _Completed]
    progress: Mapping[str, Any]


@lru_cache(maxsize=512)
def _sha256_file_snapshot(path: str, size: int, modified_ns: int) -> str:
    del size, modified_ns
    return sha256_file(path)


def _cached_sha256_file(path: Path) -> str:
    stat = path.stat()
    return _sha256_file_snapshot(str(path), stat.st_size, stat.st_mtime_ns)


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FewPersonStatisticsError(f"{name} must be an object")
    return value


def _string_tuple(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise FewPersonStatisticsError(f"{name} must be a list of strings")
    return tuple(value)


def _self_hash(record: Mapping[str, Any], *, field: str, name: str) -> str:
    body = dict(record)
    claimed = body.pop(field, None)
    if not isinstance(claimed, str) or claimed != canonical_json_sha256(body):
        raise FewPersonStatisticsError(f"{name} self-hash does not validate")
    return claimed


def _finite_probability(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or not 0.0 <= float(value) <= 1.0
    ):
        raise FewPersonStatisticsError(f"{name} must lie in [0,1]")
    return float(value)


def _validate_execution_metadata(
    record: Mapping[str, Any],
    *,
    name: str,
    expected_cudnn_enabled: bool | None = None,
) -> tuple[str, str, str]:
    timestamp = record.get("created_at_utc")
    if not isinstance(timestamp, str):
        raise FewPersonStatisticsError(f"{name} created_at_utc is missing")
    try:
        timestamp = require_utc_timestamp(timestamp, location=f"{name}.created_at_utc")
    except ValueError as exc:
        raise FewPersonStatisticsError(str(exc)) from exc
    commit = record.get("code_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise FewPersonStatisticsError(f"{name} code commit is invalid")
    environment = _mapping(record.get("environment"), name=f"{name} environment")
    string_fields = (
        "python",
        "platform",
        "numpy",
        "torch",
        "torch_cuda_runtime",
        "device_type",
        "device_name",
    )
    capability = environment.get("compute_capability")
    if (
        any(
            not isinstance(environment.get(field), str) or not environment[field]
            for field in string_fields
        )
        or environment.get("device_type") != "cuda"
        or isinstance(environment.get("device_index"), bool)
        or not isinstance(environment.get("device_index"), int)
        or int(environment["device_index"]) < 0
        or isinstance(environment.get("device_total_memory_bytes"), bool)
        or not isinstance(environment.get("device_total_memory_bytes"), int)
        or int(environment["device_total_memory_bytes"]) <= 0
        or not isinstance(capability, list)
        or len(capability) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in capability)
        or not isinstance(environment.get("cudnn_enabled_during_run"), bool)
        or isinstance(environment.get("cudnn_version"), bool)
        or not isinstance(environment.get("cudnn_version"), int)
        or int(environment["cudnn_version"]) <= 0
        or environment.get("hostname_recorded") is not False
        or environment.get("process_id_recorded") is not False
    ):
        raise FewPersonStatisticsError(f"{name} environment metadata is invalid")
    if (
        expected_cudnn_enabled is not None
        and environment.get("cudnn_enabled_during_run") is not expected_cudnn_enabled
    ):
        raise FewPersonStatisticsError(
            f"{name} recorded cuDNN policy differs from the frozen configuration"
        )
    execution_machine = dict(environment)
    execution_machine.pop("cudnn_enabled_during_run")
    return timestamp, commit, canonical_json_sha256(execution_machine)


def _digest(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FewPersonStatisticsError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_repository_head(root: Path, *, expected_commit: str) -> str:
    """Bind statistics-generation code to the repository's actual HEAD."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FewPersonStatisticsError("could not resolve repository Git HEAD") from exc
    observed = completed.stdout.strip().casefold()
    if _GIT_COMMIT_PATTERN.fullmatch(observed) is None:
        raise FewPersonStatisticsError("repository Git HEAD is not a full object ID")
    if observed != expected_commit:
        raise FewPersonStatisticsError("aggregation code commit differs from repository Git HEAD")
    return observed


def _expected_adaptation_seed(base_seed: int, fold_id: str, k: int) -> int:
    # The seed token is retained from v1 for continuity; v1.1 supersedes only
    # window-lineage construction, not the predeclared adaptation RNG schedule.
    token = f"inclusive-shift-har|few-person-v1|{base_seed}|{fold_id}|k={k}"
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "big")


def _inherited_hyperparameters(configuration: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "mixed_precision",
        "disable_cudnn",
    )
    if any(field not in configuration for field in fields):
        raise FewPersonStatisticsError("frozen training configuration is incomplete")
    return {field: configuration[field] for field in fields}


def _load_plan(path: str | Path) -> _Plan:
    value = load_json_strict(path)
    record = _mapping(value, name="few-person v1.1 manifest")
    manifest_sha256 = _self_hash(record, field="manifest_sha256", name="few-person manifest")
    manifest_file_sha256 = sha256_file(path)
    required = {
        "schema_version": "1.0.0",
        "manifest_kind": "postconfirmatory_few_person_inclusion_curve",
        "protocol_id": FEW_PERSON_V1_1_PROTOCOL_ID,
        "status": "ready_postconfirmatory_functional_core_v1_1_no_scenario_run",
        "evidence_status": FEW_PERSON_V1_1_EVIDENCE_STATUS,
        "ontology_track": "functional_core",
        "class_names": list(FUNCTIONAL_CORE_CLASS_ORDER),
        "class_schema_sha256": canonical_json_sha256(list(FUNCTIONAL_CORE_CLASS_ORDER)),
        "k_values": list(EXPECTED_K_VALUES),
        "target_metrics_or_predictions_used_for_design_or_selection": False,
        "target_raw_values_accessed_during_manifest_build": False,
    }
    mismatches = [key for key, expected in required.items() if record.get(key) != expected]
    if mismatches:
        raise FewPersonStatisticsError(f"few-person v1.1 manifest mismatch: {mismatches}")
    target_window_count = record.get("functional_core_target_window_count")
    if not isinstance(target_window_count, int) or target_window_count <= 0:
        raise FewPersonStatisticsError("few-person v1.1 target window count is invalid")
    _digest(
        record.get("functional_core_target_window_ids_sha256"),
        name="functional-core target window IDs hash",
    )
    for field in (
        "split_manifest_sha256",
        "opening_receipt_record_sha256",
        "opening_receipt_file_sha256",
        "zero_shot_index_record_sha256",
        "zero_shot_index_file_sha256",
        "final_freeze_inventory_sha256",
        "final_freeze_inventory_file_sha256",
        "ontology_config_sha256",
        "ontology_class_schema_sha256",
        "class_schema_sha256",
    ):
        _digest(record.get(field), name=field)
    supersession = _mapping(record.get("supersession"), name="v1.1 supersession")
    if (
        supersession.get("superseded_protocol_id")
        != "inclusivehar-v4-few-person-inclusion-curve-v1"
        or supersession.get("prior_outputs_preserved") is not True
    ):
        raise FewPersonStatisticsError("v1.1 does not preserve/supersede incompatible v1")
    for field in ("superseded_manifest_sha256", "superseded_manifest_file_sha256"):
        _digest(supersession.get(field), name=field)

    models_value = record.get("models")
    if not isinstance(models_value, list):
        raise FewPersonStatisticsError("few-person manifest lacks models")
    models: dict[tuple[str, int], Mapping[str, Any]] = {}
    model_counts: Counter[str] = Counter()
    for value in models_value:
        model = _mapping(value, name="few-person model")
        model_id = str(model.get("model_id"))
        seed = model.get("seed")
        if model_id not in EXPECTED_MODEL_IDS or seed not in EXPECTED_SEEDS:
            raise FewPersonStatisticsError(f"unexpected frozen model/seed: {model_id}/{seed}")
        key = (model_id, int(cast(int, seed)))
        if key in models:
            raise FewPersonStatisticsError(f"duplicate frozen model/seed: {key}")
        configuration = _mapping(
            model.get("training_configuration"), name="frozen training configuration"
        )
        configuration_sha256 = _digest(
            model.get("training_configuration_sha256"),
            name="training configuration hash",
        )
        if canonical_json_sha256(configuration) != configuration_sha256:
            raise FewPersonStatisticsError(f"training configuration hash differs for {key}")
        if (
            configuration.get("seed") != seed
            or configuration.get("checkpoint_selection_rule") != "fixed_last_epoch"
            or model.get("selected_epoch") != configuration.get("epochs")
        ):
            raise FewPersonStatisticsError(f"frozen training contract differs for {key}")
        for role in ("checkpoint", "calibrator"):
            reference = _mapping(model.get(role), name=f"frozen {role}")
            if not isinstance(reference.get("path"), str) or not reference.get("path"):
                raise FewPersonStatisticsError(f"frozen {role} path is invalid for {key}")
            _digest(reference.get("sha256"), name=f"frozen {role} hash")
        models[key] = model
        model_counts[model_id] += 1
    if set(models) != {
        (model_id, seed) for model_id in EXPECTED_MODEL_IDS for seed in EXPECTED_SEEDS
    } or model_counts != Counter({model_id: 5 for model_id in EXPECTED_MODEL_IDS}):
        raise FewPersonStatisticsError("manifest is not the exact 16-model x 5-seed family")

    scenarios_value = record.get("scenarios")
    if not isinstance(scenarios_value, list):
        raise FewPersonStatisticsError("few-person manifest lacks scenarios")
    scenarios: dict[tuple[str, int], Mapping[str, Any]] = {}
    evaluation_counts: dict[int, Counter[str]] = {k: Counter() for k in EXPECTED_K_VALUES}
    for value in scenarios_value:
        scenario = _mapping(value, name="few-person scenario")
        fold_id = str(scenario.get("fold_id"))
        k = scenario.get("k")
        if fold_id not in EXPECTED_FOLDS or k not in EXPECTED_K_VALUES:
            raise FewPersonStatisticsError(f"unexpected scenario: {fold_id}/k={k}")
        key = (fold_id, int(cast(int, k)))
        if key in scenarios:
            raise FewPersonStatisticsError(f"duplicate scenario: {key}")
        inclusion = _string_tuple(
            scenario.get("target_inclusion_subjects"), name=f"{key} inclusion subjects"
        )
        evaluation = _string_tuple(
            scenario.get("evaluation_subjects"), name=f"{key} evaluation subjects"
        )
        unused = _string_tuple(
            scenario.get("unused_target_subjects"), name=f"{key} unused subjects"
        )
        normalization_subjects = _string_tuple(
            scenario.get("normalization_fit_subjects"),
            name=f"{key} normalization-fit subjects",
        )
        if (
            scenario.get("scenario_id") != f"{fold_id}__k{k}"
            or len(inclusion) != k
            or len(evaluation) != 2
            or len(set(inclusion)) != len(inclusion)
            or len(set(evaluation)) != len(evaluation)
            or len(set(unused)) != len(unused)
            or set(inclusion) & set(evaluation)
            or set(inclusion) | set(evaluation) | set(unused) != EXPECTED_TARGET_SUBJECTS
            or len(inclusion) + len(evaluation) + len(unused) != len(EXPECTED_TARGET_SUBJECTS)
            or normalization_subjects != EXPECTED_SOURCE_NORMALIZATION_SUBJECTS
            or scenario.get("window_lineage_track") != "functional_core"
            or scenario.get("validation_subjects") != []
            or scenario.get("calibration_subjects") != []
            or scenario.get("threshold_selection_subjects") != []
        ):
            raise FewPersonStatisticsError(f"participant isolation fails for {key}")
        for field in (
            "target_inclusion_window_ids_sha256",
            "evaluation_window_ids_sha256",
        ):
            _digest(scenario.get(field), name=f"{key} {field}")
        for field in ("target_inclusion_window_count", "evaluation_window_count"):
            count = scenario.get(field)
            if not isinstance(count, int) or count <= 0:
                raise FewPersonStatisticsError(f"{key} lacks a positive {field}")
        evaluation_counts[int(cast(int, k))].update(evaluation)
        scenarios[key] = scenario
    expected_scenarios = {(fold, k) for fold in EXPECTED_FOLDS for k in EXPECTED_K_VALUES}
    if set(scenarios) != expected_scenarios or any(
        counts != Counter({subject: 1 for subject in EXPECTED_TARGET_SUBJECTS})
        for counts in evaluation_counts.values()
    ):
        raise FewPersonStatisticsError("outer folds do not cover each target exactly once per k")
    for k in EXPECTED_K_VALUES:
        if (
            sum(int(scenarios[(fold, k)]["evaluation_window_count"]) for fold in EXPECTED_FOLDS)
            != target_window_count
        ):
            raise FewPersonStatisticsError(
                f"evaluation folds do not partition functional-core target windows for k={k}"
            )
    expected_cells = frozenset(
        CellKey(model_id, seed, fold, k)
        for model_id in EXPECTED_MODEL_IDS
        for seed in EXPECTED_SEEDS
        for fold in EXPECTED_FOLDS
        for k in EXPECTED_K_VALUES
    )
    if len(expected_cells) != EXPECTED_CELL_COUNT:
        raise AssertionError("internal few-person cell count changed")
    return _Plan(
        record=record,
        manifest_sha256=manifest_sha256,
        manifest_file_sha256=manifest_file_sha256,
        class_names=FUNCTIONAL_CORE_CLASS_ORDER,
        scenarios=scenarios,
        models=models,
        expected_cells=expected_cells,
    )


def _cell_from_record(record: Mapping[str, Any]) -> CellKey:
    try:
        return CellKey(
            model_id=str(record["model_id"]),
            seed=int(record["seed"]),
            fold_id=str(record["fold_id"]),
            k=int(record["k"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FewPersonStatisticsError("scenario record lacks a valid cell identity") from exc


def _resolve_artifact(
    value: Any,
    *,
    artifact_root: Path,
    required_subtree: Path,
    name: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise FewPersonStatisticsError(f"{name} path is missing")
    raw = Path(value)
    if raw.is_absolute():
        raise FewPersonStatisticsError(f"{name} path must be repository-relative")
    candidate = artifact_root / raw
    if candidate.is_symlink():
        raise FewPersonStatisticsError(f"{name} must not be a symlink")
    resolved = candidate.resolve(strict=True)
    root = artifact_root.resolve(strict=True)
    subtree = required_subtree.resolve(strict=True)
    for boundary, label in ((root, "artifact root"), (subtree, "few-person result root")):
        try:
            resolved.relative_to(boundary)
        except ValueError as exc:
            raise FewPersonStatisticsError(f"{name} escapes {label}") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise FewPersonStatisticsError(f"{name} is not a regular file")
    return resolved


def _resolve_repository_file(value: Any, *, artifact_root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise FewPersonStatisticsError(f"{name} path is missing")
    raw = Path(value)
    if raw.is_absolute():
        raise FewPersonStatisticsError(f"{name} path must be repository-relative")
    candidate = artifact_root / raw
    if candidate.is_symlink():
        raise FewPersonStatisticsError(f"{name} must not be a symlink")
    root = artifact_root.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise FewPersonStatisticsError(f"{name} escapes artifact root") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise FewPersonStatisticsError(f"{name} is not a regular file")
    return resolved


_LOCKED_ZERO_SHOT_SUBTREE = ("results", "confirmatory", "zero_shot_v1")


def _resolve_locked_zero_shot_file(value: Any, *, artifact_root: Path, name: str) -> Path:
    """Rebase only the immutable opening-1 zero-shot artifact namespace.

    Opening 1 predates repository-relative index paths. Its hash-pinned index
    stores absolute Windows result paths, while the result records store
    prediction paths relative to the historical ``results`` artifact root.
    Accepting arbitrary absolute paths would weaken the current artifact
    contract, so this compatibility resolver recognizes only the exact frozen
    ``results/confirmatory/zero_shot_v1/<file>`` suffix.
    """

    if not isinstance(value, str) or not value:
        raise FewPersonStatisticsError(f"{name} path is missing")
    portable = PurePosixPath(value.replace("\\", "/"))
    parts = tuple(part for part in portable.parts if part not in ("", "/", "//"))
    if any(part in (".", "..") for part in parts):
        raise FewPersonStatisticsError(f"{name} path contains a traversal component")
    lowered = tuple(part.casefold() for part in parts)
    marker = tuple(part.casefold() for part in _LOCKED_ZERO_SHOT_SUBTREE)
    matches = [
        index
        for index in range(len(parts) - len(marker) + 1)
        if lowered[index : index + len(marker)] == marker
    ]
    if len(matches) == 1:
        relative_parts = parts[matches[0] :]
    elif lowered[:2] == marker[1:]:
        relative_parts = (_LOCKED_ZERO_SHOT_SUBTREE[0], *parts)
    else:
        raise FewPersonStatisticsError(
            f"{name} path is outside the frozen zero-shot artifact namespace"
        )
    if len(relative_parts) != len(_LOCKED_ZERO_SHOT_SUBTREE) + 1:
        raise FewPersonStatisticsError(f"{name} path has an unexpected frozen layout")

    root_candidate = Path(artifact_root)
    if root_candidate.is_symlink():
        raise FewPersonStatisticsError("artifact root must not be a symlink")
    root = root_candidate.resolve(strict=True)
    subtree = root.joinpath(*_LOCKED_ZERO_SHOT_SUBTREE).resolve(strict=True)
    candidate = root.joinpath(*relative_parts)
    if candidate.is_symlink():
        raise FewPersonStatisticsError(f"{name} must not be a symlink")
    resolved = candidate.resolve(strict=True)
    for boundary, label in ((root, "artifact root"), (subtree, "zero-shot subtree")):
        try:
            resolved.relative_to(boundary)
        except ValueError as exc:
            raise FewPersonStatisticsError(f"{name} escapes {label}") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise FewPersonStatisticsError(f"{name} is not a regular file")
    return resolved


def _participant_values(report: Mapping[str, Any]) -> dict[str, float]:
    rows = report.get("participants")
    if not isinstance(rows, list) or not rows:
        raise FewPersonStatisticsError("participant report is missing")
    output: dict[str, float] = {}
    for value in rows:
        row = _mapping(value, name="participant metric row")
        participant = row.get("participant_id")
        if not isinstance(participant, str) or participant in output:
            raise FewPersonStatisticsError("participant metric IDs are invalid")
        output[participant] = _finite_probability(row.get("macro_f1"), name="participant macro-F1")
    return output


def _validate_checkpoint(
    path: Path,
    *,
    record: Mapping[str, Any],
    plan: _Plan,
    scenario: Mapping[str, Any],
    model_entry: Mapping[str, Any],
    key: CellKey,
) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint = _mapping(payload, name="adapted checkpoint")
    required = {
        "schema_version": "1.0.0",
        "record_kind": "few_person_adapted_neural_checkpoint",
        "evidence_status": FEW_PERSON_V1_1_EVIDENCE_STATUS,
        "protocol_id": FEW_PERSON_V1_1_PROTOCOL_ID,
        "model_id": key.model_id,
        "seed": key.seed,
        "fold_id": key.fold_id,
        "k": key.k,
        "adaptation_seed": _expected_adaptation_seed(key.seed, key.fold_id, key.k),
        "few_person_manifest_sha256": plan.manifest_sha256,
        "split_manifest_sha256": plan.record["split_manifest_sha256"],
        "base_checkpoint_sha256": model_entry["checkpoint"]["sha256"],
        "base_training_configuration_sha256": model_entry["training_configuration_sha256"],
        "checkpoint_selection_rule": "fixed_last_epoch_no_validation",
        "label_schema": list(plan.class_names),
        "inclusion_subjects": sorted(
            cast(list[str], scenario["target_inclusion_subjects"]), key=int
        ),
        "inclusion_window_count": scenario["target_inclusion_window_count"],
        "inclusion_window_ids_sha256": scenario["target_inclusion_window_ids_sha256"],
        "evaluation_subjects_excluded_from_training": sorted(
            cast(list[str], scenario["evaluation_subjects"]), key=int
        ),
        "normalization_fit_subjects": scenario["normalization_fit_subjects"],
        "target_validation_performed": False,
        "calibration_refit_on_target": False,
        "threshold_selection_performed": False,
    }
    mismatches = [
        field for field, expected in required.items() if checkpoint.get(field) != expected
    ]
    if mismatches:
        raise FewPersonStatisticsError(f"{key.cell_id} checkpoint mismatch: {mismatches}")
    normalizer = _mapping(checkpoint.get("normalization"), name="checkpoint normalization")
    training_participants = normalizer.get("training_participants")
    if not isinstance(training_participants, list) or any(
        not isinstance(value, str) for value in training_participants
    ):
        raise FewPersonStatisticsError(
            f"{key.cell_id} normalization participant metadata is invalid"
        )
    if (
        training_participants != scenario["normalization_fit_subjects"]
        or set(training_participants) & EXPECTED_TARGET_SUBJECTS
    ):
        raise FewPersonStatisticsError(f"{key.cell_id} normalization includes target participants")
    history = checkpoint.get("training_history")
    configuration = _mapping(
        checkpoint.get("base_training_configuration"), name="base training configuration"
    )
    disable_cudnn = configuration.get("disable_cudnn")
    if not isinstance(disable_cudnn, bool):
        raise FewPersonStatisticsError(f"{key.cell_id} frozen disable_cudnn policy is not Boolean")
    expected_cudnn_enabled = not disable_cudnn
    epochs = configuration.get("epochs")
    if (
        canonical_json_sha256(configuration) != model_entry["training_configuration_sha256"]
        or not isinstance(epochs, int)
        or not isinstance(history, list)
        or len(history) != epochs
    ):
        raise FewPersonStatisticsError(f"{key.cell_id} training budget/history changed")
    for index, value in enumerate(history, start=1):
        row = _mapping(value, name="adaptation history row")
        loss = row.get("training_cross_entropy")
        if (
            row.get("epoch") != index
            or row.get("validation_accessed") is not False
            or row.get("checkpoint_selected") is not (index == epochs)
            or isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not np.isfinite(loss)
            or float(loss) < 0.0
        ):
            raise FewPersonStatisticsError(f"{key.cell_id} used validation/early selection")
    if not isinstance(checkpoint.get("model_state"), Mapping) or not isinstance(
        checkpoint.get("optimizer_state"), Mapping
    ):
        raise FewPersonStatisticsError(f"{key.cell_id} checkpoint state is incomplete")
    if record.get("adaptation_seed") != checkpoint.get("adaptation_seed") or record.get(
        "code_commit"
    ) != checkpoint.get("code_commit"):
        raise FewPersonStatisticsError(
            f"{key.cell_id} adaptation seed/commit differs from checkpoint"
        )
    if record.get("created_at_utc") != checkpoint.get("created_at_utc") or record.get(
        "environment"
    ) != checkpoint.get("environment"):
        raise FewPersonStatisticsError(
            f"{key.cell_id} timestamp/environment differs from checkpoint"
        )
    _validate_execution_metadata(
        checkpoint,
        name=f"{key.cell_id} checkpoint",
        expected_cudnn_enabled=expected_cudnn_enabled,
    )
    rng_states = checkpoint.get("rng_states")
    if (
        not isinstance(checkpoint.get("parameter_count"), int)
        or int(checkpoint["parameter_count"]) <= 0
        or not isinstance(rng_states, Mapping)
        or set(rng_states) != {"python", "numpy", "torch_cpu", "torch_cuda"}
    ):
        raise FewPersonStatisticsError(f"{key.cell_id} checkpoint metadata is incomplete")
    return checkpoint


def _load_frozen_calibrator(
    *,
    artifact_root: Path,
    plan: _Plan,
    model_entry: Mapping[str, Any],
    key: CellKey,
) -> TemperatureCalibrator:
    reference = _mapping(model_entry.get("calibrator"), name="frozen calibrator")
    path = _resolve_repository_file(
        reference.get("path"), artifact_root=artifact_root, name="frozen calibrator"
    )
    if _cached_sha256_file(path) != reference.get("sha256"):
        raise FewPersonStatisticsError(f"{key.cell_id} frozen calibrator file hash changed")
    try:
        payload, calibrator = load_source_temperature_calibrator_file(
            path,
            expected_checkpoint_sha256=str(model_entry["checkpoint"]["sha256"]),
            expected_training_configuration_sha256=str(
                model_entry["training_configuration_sha256"]
            ),
            expected_split_manifest_sha256=str(plan.record["split_manifest_sha256"]),
        )
    except (OSError, ValueError) as exc:
        raise FewPersonStatisticsError(
            f"{key.cell_id} frozen calibrator does not validate: {exc}"
        ) from exc
    if payload.get("record_sha256") != model_entry.get("calibrator_record_sha256") or payload.get(
        "class_names"
    ) != list(plan.class_names):
        raise FewPersonStatisticsError(f"{key.cell_id} frozen calibrator lineage differs")
    return calibrator


def _validate_prediction(
    path: Path,
    *,
    record: Mapping[str, Any],
    plan: _Plan,
    scenario: Mapping[str, Any],
    key: CellKey,
    calibrator: TemperatureCalibrator,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    NDArray[np.int64],
    NDArray[np.float64],
    Mapping[str, float],
]:
    with np.load(path, allow_pickle=False) as arrays:
        required_keys = {
            "window_ids",
            "participant_ids",
            "true_labels",
            "logits",
            "uncalibrated_probabilities",
            "source_temperature_probabilities",
            "predicted_labels",
        }
        if set(arrays.files) != required_keys:
            raise FewPersonStatisticsError(f"{key.cell_id} prediction keys differ")
        window_ids = tuple(str(value) for value in arrays["window_ids"].tolist())
        participants = tuple(str(value) for value in arrays["participant_ids"].tolist())
        labels = np.asarray(arrays["true_labels"], dtype=np.int64)
        logits = np.asarray(arrays["logits"], dtype=np.float64)
        uncalibrated = np.asarray(arrays["uncalibrated_probabilities"], dtype=np.float64)
        probabilities = np.asarray(arrays["source_temperature_probabilities"], dtype=np.float64)
        predictions = np.asarray(arrays["predicted_labels"], dtype=np.int64)
    count = labels.size
    if (
        count != scenario["evaluation_window_count"]
        or labels.shape != (count,)
        or len(window_ids) != count
        or len(set(window_ids)) != count
        or canonical_json_sha256(sorted(window_ids)) != scenario["evaluation_window_ids_sha256"]
        or len(participants) != count
        or set(participants) != set(cast(list[str], scenario["evaluation_subjects"]))
        or set(participants) & set(cast(list[str], scenario["target_inclusion_subjects"]))
        or logits.shape != (count, len(plan.class_names))
        or uncalibrated.shape != logits.shape
        or probabilities.shape != logits.shape
        or predictions.shape != (count,)
        or np.any(labels < 0)
        or np.any(labels >= len(plan.class_names))
        or np.any(predictions < 0)
        or np.any(predictions >= len(plan.class_names))
        or not np.isfinite(logits).all()
        or not np.isfinite(uncalibrated).all()
        or not np.isfinite(probabilities).all()
        or np.any(uncalibrated < 0.0)
        or np.any(uncalibrated > 1.0)
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.allclose(uncalibrated.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6)
        or not np.array_equal(predictions, probabilities.argmax(axis=1))
    ):
        raise FewPersonStatisticsError(f"{key.cell_id} prediction alignment fails")
    shifted = logits - logits.max(axis=1, keepdims=True)
    expected_uncalibrated = np.exp(shifted)
    expected_uncalibrated /= expected_uncalibrated.sum(axis=1, keepdims=True)
    expected_calibrated = np.asarray(calibrator.probabilities(logits), dtype=np.float64)
    if not np.allclose(uncalibrated, expected_uncalibrated, atol=1e-6, rtol=1e-6):
        raise FewPersonStatisticsError(
            f"{key.cell_id} uncalibrated probabilities are not softmax(logits)"
        )
    if not np.allclose(probabilities, expected_calibrated, atol=1e-10, rtol=1e-10):
        raise FewPersonStatisticsError(
            f"{key.cell_id} calibrated probabilities do not use the frozen temperature"
        )
    recomputed = classification_report(
        labels, probabilities, participants, class_names=plan.class_names
    )
    stored = _mapping(record.get("participant_level_report"), name="stored result report")
    if canonical_json_sha256(recomputed) != canonical_json_sha256(stored):
        raise FewPersonStatisticsError(f"{key.cell_id} metrics do not reconstruct from NPZ")
    return window_ids, participants, labels, probabilities, _participant_values(recomputed)


def _split_source_sha256(split: Mapping[str, Any]) -> str:
    value = split.get("source_artifact_sha256")
    if value is None:
        value = _mapping(split.get("source_evidence"), name="split source evidence").get(
            "sensor_artifact_sha256"
        )
    return _digest(value, name="split source artifact hash")


def _validate_shard_input_lineage(
    value: Mapping[str, Any],
    *,
    artifact_root: Path,
    plan: _Plan,
    split: Mapping[str, Any],
    expected_participants: set[str],
    expected_window_count: int,
    expected_window_ids_sha256: str,
    source_sha256: str,
    key: CellKey,
) -> None:
    required = {
        "mode": "hash_pinned_target_participant_shards",
        "participant_ids": sorted(expected_participants, key=int),
        "selected_window_count": expected_window_count,
        "selected_window_ids_sha256": expected_window_ids_sha256,
    }
    if any(value.get(field) != expected for field, expected in required.items()):
        raise FewPersonStatisticsError(f"{key.cell_id} participant-shard selection changed")
    record_path = _resolve_repository_file(
        value.get("record_path"), artifact_root=artifact_root, name="primary cache record"
    )
    if _cached_sha256_file(record_path) != value.get("record_file_sha256"):
        raise FewPersonStatisticsError(f"{key.cell_id} primary cache record file hash changed")
    cache_record = _mapping(load_json_strict(record_path), name="primary cache record")
    cache_record_hash = _self_hash(cache_record, field="record_sha256", name="primary cache record")
    cache_required = {
        "schema_version": PRIMARY_CACHE_SCHEMA_VERSION,
        "record_kind": PRIMARY_CACHE_RECORD_KIND,
        "status": "complete_create_only",
        "evidence_status": PRIMARY_CACHE_EVIDENCE_STATUS,
        "ontology_track": "functional_core",
        "class_names": list(plan.class_names),
        "target_cache_ordered_alignment_validated_against_opening_1": True,
        "unlock_api_called": False,
        "new_target_opening_created": False,
    }
    if cache_record_hash != value.get("record_sha256") or any(
        cache_record.get(field) != expected for field, expected in cache_required.items()
    ):
        raise FewPersonStatisticsError(f"{key.cell_id} primary cache record contract changed")
    cache_split = _mapping(cache_record.get("split_manifest"), name="cache split lineage")
    cache_raw = _mapping(cache_record.get("raw_sensor_csv"), name="cache raw lineage")
    opening = _mapping(cache_record.get("opening_1"), name="cache opening lineage")
    if (
        cache_split.get("record_sha256") != plan.record["split_manifest_sha256"]
        or cache_raw.get("sha256") != source_sha256
        or opening.get("receipt_record_sha256") != plan.record["opening_receipt_record_sha256"]
        or opening.get("receipt_file_sha256") != plan.record["opening_receipt_file_sha256"]
        or opening.get("index_record_sha256") != plan.record["zero_shot_index_record_sha256"]
        or opening.get("index_file_sha256") != plan.record["zero_shot_index_file_sha256"]
        or opening.get("target_seal_id") != plan.record["target_seal_id"]
    ):
        raise FewPersonStatisticsError(f"{key.cell_id} primary cache opening/split lineage changed")
    indexed = _mapping(
        cache_record.get("target_participant_shards"), name="cache target shard index"
    )
    if set(indexed) != EXPECTED_TARGET_SUBJECTS:
        raise FewPersonStatisticsError(f"{key.cell_id} cache target shard subject set changed")
    selected = _mapping(value.get("participant_shards"), name="selected participant shards")
    if set(selected) != expected_participants:
        raise FewPersonStatisticsError(f"{key.cell_id} selected participant shard set changed")
    observed_count = 0
    observed_ids: set[str] = set()
    split_windows = split.get("windows")
    if not isinstance(split_windows, list):
        raise FewPersonStatisticsError("parent split lacks windows")
    for participant in sorted(expected_participants, key=int):
        selected_ref = _mapping(selected.get(participant), name=f"selected shard {participant}")
        indexed_ref = _mapping(indexed.get(participant), name=f"indexed shard {participant}")
        expected_ids = sorted(
            str(item["window_id"])
            for item in split_windows
            if isinstance(item, Mapping)
            and item.get("partition") == "target_sealed"
            and item.get("subject_id") == participant
            and isinstance(item.get("canonical_labels"), Mapping)
            and "functional_core" in cast(Mapping[str, Any], item["canonical_labels"])
        )
        expected_hash = canonical_json_sha256(expected_ids)
        selected_required = {
            "participant_id": participant,
            "window_count": len(expected_ids),
            "window_ids_sha256": expected_hash,
            "array_path": indexed_ref.get("array_path"),
            "array_sha256": indexed_ref.get("array_sha256"),
            "metadata_path": indexed_ref.get("metadata_path"),
            "metadata_sha256": indexed_ref.get("metadata_sha256"),
        }
        if not expected_ids or any(
            selected_ref.get(field) != expected for field, expected in selected_required.items()
        ):
            raise FewPersonStatisticsError(f"{key.cell_id} selected shard {participant} changed")
        array_path = _resolve_repository_file(
            selected_ref.get("array_path"),
            artifact_root=artifact_root,
            name=f"target shard {participant} array",
        )
        metadata_path = _resolve_repository_file(
            selected_ref.get("metadata_path"),
            artifact_root=artifact_root,
            name=f"target shard {participant} metadata",
        )
        if _cached_sha256_file(array_path) != selected_ref.get(
            "array_sha256"
        ) or _cached_sha256_file(metadata_path) != selected_ref.get("metadata_sha256"):
            raise FewPersonStatisticsError(f"{key.cell_id} shard {participant} file hash changed")
        metadata = _mapping(
            load_json_strict(metadata_path), name=f"target shard {participant} metadata"
        )
        lineage = _mapping(metadata.get("lineage"), name=f"target shard {participant} lineage")
        metadata_required = {
            "schema_version": "1.0.0",
            "array_sha256": selected_ref["array_sha256"],
            "window_count": len(expected_ids),
            "shape": [len(expected_ids), 128, 6],
            "class_names": list(plan.class_names),
            "ontology_track": "functional_core",
        }
        lineage_required = {
            "split_manifest_sha256": plan.record["split_manifest_sha256"],
            "source_artifact_sha256": source_sha256,
            "partition": "target_sealed",
            "cache_role": "few_person_target_participant_shard",
            "participant_id": participant,
            "window_count": len(expected_ids),
            "window_ids_sha256": expected_hash,
            "opening_receipt_record_sha256": plan.record["opening_receipt_record_sha256"],
            "locked_target_index_record_sha256": plan.record["zero_shot_index_record_sha256"],
            "target_seal_id": plan.record["target_seal_id"],
        }
        if any(
            metadata.get(field) != expected for field, expected in metadata_required.items()
        ) or any(lineage.get(field) != expected for field, expected in lineage_required.items()):
            raise FewPersonStatisticsError(f"{key.cell_id} shard {participant} metadata changed")
        observed_count += len(expected_ids)
        overlap = observed_ids & set(expected_ids)
        if overlap:
            raise FewPersonStatisticsError(f"{key.cell_id} selected shards overlap")
        observed_ids.update(expected_ids)
    if (
        observed_count != expected_window_count
        or canonical_json_sha256(sorted(observed_ids)) != expected_window_ids_sha256
    ):
        raise FewPersonStatisticsError(f"{key.cell_id} selected shard union changed")


def _validate_input_materialization(
    *,
    record: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_sha256: str,
    artifact_root: Path,
    plan: _Plan,
    scenario: Mapping[str, Any],
    key: CellKey,
) -> None:
    value = _mapping(record.get("input_materialization"), name="result input materialization")
    checkpoint_value = _mapping(
        checkpoint.get("input_materialization"), name="checkpoint input materialization"
    )
    if value.get("schema_version") != "1.0.0" or checkpoint_value.get("schema_version") != "1.0.0":
        raise FewPersonStatisticsError(f"{key.cell_id} input-lineage schema changed")
    split_ref = _mapping(value.get("split_manifest"), name="input split lineage")
    checkpoint_split = _mapping(
        checkpoint_value.get("split_manifest"), name="checkpoint split lineage"
    )
    if canonical_json_sha256(split_ref) != canonical_json_sha256(checkpoint_split):
        raise FewPersonStatisticsError(f"{key.cell_id} checkpoint/result split lineage differs")
    split_path = _resolve_repository_file(
        split_ref.get("path"), artifact_root=artifact_root, name="input split manifest"
    )
    split = _mapping(load_json_strict(split_path), name="input split manifest")
    if (
        _cached_sha256_file(split_path) != split_ref.get("file_sha256")
        or _self_hash(split, field="split_manifest_sha256", name="input split manifest")
        != plan.record["split_manifest_sha256"]
        or split_ref.get("record_sha256") != plan.record["split_manifest_sha256"]
    ):
        raise FewPersonStatisticsError(f"{key.cell_id} exact split file lineage changed")
    try:
        validate_few_person_v1_1_manifest_assignments(plan.record, split)
    except FewPersonProtocolError as exc:
        raise FewPersonStatisticsError(f"{key.cell_id} split assignments differ: {exc}") from exc
    source_sha256 = _split_source_sha256(split)
    training = _mapping(value.get("training"), name="training input lineage")
    evaluation = _mapping(value.get("evaluation"), name="evaluation input lineage")
    checkpoint_training = _mapping(
        checkpoint_value.get("training"), name="checkpoint training input lineage"
    )
    checkpoint_evaluation = _mapping(
        checkpoint_value.get("evaluation"), name="checkpoint evaluation input lineage"
    )
    if canonical_json_sha256(training) != canonical_json_sha256(
        checkpoint_training
    ) or canonical_json_sha256(evaluation) != canonical_json_sha256(checkpoint_evaluation):
        raise FewPersonStatisticsError(f"{key.cell_id} checkpoint/result input lineage differs")
    cache_mode = training.get("mode") == "hash_pinned_target_participant_shards"
    if cache_mode != (evaluation.get("mode") == "hash_pinned_target_participant_shards"):
        raise FewPersonStatisticsError(f"{key.cell_id} mixes raw and shard input modes")
    if cache_mode:
        _validate_shard_input_lineage(
            training,
            artifact_root=artifact_root,
            plan=plan,
            split=split,
            expected_participants=set(cast(list[str], scenario["target_inclusion_subjects"])),
            expected_window_count=int(scenario["target_inclusion_window_count"]),
            expected_window_ids_sha256=str(scenario["target_inclusion_window_ids_sha256"]),
            source_sha256=source_sha256,
            key=key,
        )
        _validate_shard_input_lineage(
            evaluation,
            artifact_root=artifact_root,
            plan=plan,
            split=split,
            expected_participants=set(cast(list[str], scenario["evaluation_subjects"])),
            expected_window_count=int(scenario["evaluation_window_count"]),
            expected_window_ids_sha256=str(scenario["evaluation_window_ids_sha256"]),
            source_sha256=source_sha256,
            key=key,
        )
    else:
        raw_required = {
            "mode": "immutable_raw_csv_authorized_windows_only",
            "source_artifact_sha256": source_sha256,
        }
        if any(training.get(field) != expected for field, expected in raw_required.items()) or any(
            evaluation.get(field) != expected for field, expected in raw_required.items()
        ):
            raise FewPersonStatisticsError(f"{key.cell_id} raw input lineage changed")
        train_path = _resolve_repository_file(
            training.get("source_path"), artifact_root=artifact_root, name="training raw CSV"
        )
        eval_path = _resolve_repository_file(
            evaluation.get("source_path"), artifact_root=artifact_root, name="evaluation raw CSV"
        )
        if (
            train_path != eval_path
            or _cached_sha256_file(train_path) != source_sha256
            or _cached_sha256_file(eval_path) != source_sha256
        ):
            raise FewPersonStatisticsError(f"{key.cell_id} raw source file lineage changed")
    result_barrier = _mapping(value.get("access_barrier"), name="result access barrier")
    checkpoint_barrier = _mapping(
        checkpoint_value.get("access_barrier"), name="checkpoint access barrier"
    )
    expected_checkpoint_barrier = {
        "evaluation_metadata_preflight_only_before_adaptation": cache_mode,
        "evaluation_signals_accessed_before_adapted_checkpoint_fixed": False,
        "adapted_checkpoint_fixed_before_evaluation": False,
    }
    expected_result_barrier = {
        "evaluation_metadata_preflight_only_before_adaptation": cache_mode,
        "evaluation_signals_accessed_before_adapted_checkpoint_fixed": False,
        "adapted_checkpoint_fixed_before_evaluation": True,
        "adapted_checkpoint_sha256": checkpoint_sha256,
    }
    if checkpoint_barrier != expected_checkpoint_barrier or result_barrier != (
        expected_result_barrier
    ):
        raise FewPersonStatisticsError(f"{key.cell_id} held-out access barrier changed")


def _validate_completed_record(
    path: Path,
    *,
    plan: _Plan,
    artifact_root: Path,
    results_root: Path,
) -> _Completed:
    value = load_json_strict(path)
    record = _mapping(value, name="few-person result")
    record_sha256 = _self_hash(record, field="record_sha256", name="few-person result")
    key = _cell_from_record(record)
    if key not in plan.expected_cells:
        raise FewPersonStatisticsError(f"unexpected completed cell: {key.cell_id}")
    if (
        path.name != "result.json"
        or re.fullmatch(rf"{re.escape(key.cell_id)}(?:--attempt-\d+)?", path.parent.name) is None
    ):
        raise FewPersonStatisticsError(f"{key.cell_id} result directory identity differs")
    scenario = plan.scenarios[(key.fold_id, key.k)]
    model_entry = plan.models[(key.model_id, key.seed)]
    base_configuration = _mapping(
        model_entry.get("training_configuration"), name="frozen training configuration"
    )
    disable_cudnn = base_configuration.get("disable_cudnn")
    if not isinstance(disable_cudnn, bool):
        raise FewPersonStatisticsError(f"{key.cell_id} frozen disable_cudnn policy is not Boolean")
    expected_cudnn_enabled = not disable_cudnn
    required = {
        "schema_version": "1.0.0",
        "record_kind": "few_person_outer_fold_result",
        "status": "complete_create_only_postconfirmatory_secondary",
        "evidence_status": FEW_PERSON_V1_1_EVIDENCE_STATUS,
        "protocol_id": FEW_PERSON_V1_1_PROTOCOL_ID,
        "model_id": key.model_id,
        "seed": key.seed,
        "fold_id": key.fold_id,
        "k": key.k,
        "adaptation_seed": _expected_adaptation_seed(key.seed, key.fold_id, key.k),
        "few_person_manifest_sha256": plan.manifest_sha256,
        "split_manifest_sha256": plan.record["split_manifest_sha256"],
        "opening_receipt_record_sha256": plan.record["opening_receipt_record_sha256"],
        "zero_shot_index_record_sha256": plan.record["zero_shot_index_record_sha256"],
        "base_checkpoint_sha256": model_entry["checkpoint"]["sha256"],
        "base_source_calibrator_sha256": model_entry["calibrator"]["sha256"],
        "base_training_configuration_sha256": model_entry["training_configuration_sha256"],
        "source_hyperparameters_inherited": _inherited_hyperparameters(base_configuration),
        "adaptation_objective": "supervised_cross_entropy_only",
        "checkpoint_selection_rule": "fixed_last_epoch_no_validation",
        "normalization_refit_on_target": False,
        "calibration_refit_on_target": False,
        "threshold_selection_performed": False,
        "target_validation_performed": False,
        "inclusion_subjects": sorted(
            cast(list[str], scenario["target_inclusion_subjects"]), key=int
        ),
        "evaluation_subjects": sorted(cast(list[str], scenario["evaluation_subjects"]), key=int),
        "unused_target_subjects": scenario["unused_target_subjects"],
        "statistical_unit": "participant",
        "target_information_used_for_model_or_hyperparameter_selection": False,
    }
    mismatches = [field for field, expected in required.items() if record.get(field) != expected]
    if mismatches:
        raise FewPersonStatisticsError(f"{key.cell_id} result mismatch: {mismatches}")
    _, commit, environment_sha256 = _validate_execution_metadata(
        record,
        name=f"{key.cell_id} result",
        expected_cudnn_enabled=expected_cudnn_enabled,
    )
    device = _mapping(record.get("device"), name="few-person result device")
    elapsed = record.get("elapsed_seconds")
    if device.get("cudnn_enabled") is not expected_cudnn_enabled:
        raise FewPersonStatisticsError(
            f"{key.cell_id} recorded actual cuDNN policy differs from the frozen configuration"
        )
    if (
        device.get("type") != "cuda"
        or not isinstance(device.get("peak_vram_bytes"), int)
        or int(device["peak_vram_bytes"]) < 0
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not np.isfinite(elapsed)
        or float(elapsed) < 0.0
    ):
        raise FewPersonStatisticsError(f"{key.cell_id} execution metadata is invalid")
    checkpoint_ref = _mapping(record.get("adapted_checkpoint"), name="adapted checkpoint ref")
    prediction_ref = _mapping(record.get("prediction_artifact"), name="prediction ref")
    checkpoint_path = _resolve_artifact(
        checkpoint_ref.get("path"),
        artifact_root=artifact_root,
        required_subtree=results_root,
        name="adapted checkpoint",
    )
    prediction_path = _resolve_artifact(
        prediction_ref.get("path"),
        artifact_root=artifact_root,
        required_subtree=results_root,
        name="prediction artifact",
    )
    if (
        checkpoint_path.parent != path.parent.resolve(strict=True)
        or checkpoint_path.name != "adapted.pt"
        or prediction_path.parent != path.parent.resolve(strict=True)
        or prediction_path.name != "predictions.npz"
    ):
        raise FewPersonStatisticsError(f"{key.cell_id} artifacts are not local to the run")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    prediction_sha256 = sha256_file(prediction_path)
    if checkpoint_sha256 != checkpoint_ref.get("sha256"):
        raise FewPersonStatisticsError(f"{key.cell_id} adapted checkpoint hash mismatch")
    if prediction_sha256 != prediction_ref.get("sha256"):
        raise FewPersonStatisticsError(f"{key.cell_id} prediction hash mismatch")
    checkpoint = _validate_checkpoint(
        checkpoint_path,
        record=record,
        plan=plan,
        scenario=scenario,
        model_entry=model_entry,
        key=key,
    )
    _validate_input_materialization(
        record=record,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        artifact_root=artifact_root,
        plan=plan,
        scenario=scenario,
        key=key,
    )
    calibrator = _load_frozen_calibrator(
        artifact_root=artifact_root, plan=plan, model_entry=model_entry, key=key
    )
    window_ids, participants, labels, probabilities, participant_values = _validate_prediction(
        prediction_path,
        record=record,
        plan=plan,
        scenario=scenario,
        key=key,
        calibrator=calibrator,
    )
    return _Completed(
        key=key,
        record_path=path,
        record_sha256=record_sha256,
        record_file_sha256=sha256_file(path),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        prediction_path=prediction_path,
        prediction_sha256=prediction_sha256,
        code_commit=commit,
        environment_sha256=environment_sha256,
        window_ids=window_ids,
        participant_ids=participants,
        labels=labels,
        probabilities=probabilities,
        participant_macro_f1=participant_values,
    )


def _validate_failure(path: Path, *, plan: _Plan) -> tuple[CellKey, Mapping[str, Any]]:
    value = load_json_strict(path)
    record = _mapping(value, name="few-person failure")
    _self_hash(record, field="record_sha256", name="few-person failure")
    key = _cell_from_record(record)
    if key not in plan.expected_cells:
        raise FewPersonStatisticsError(f"unexpected failed cell: {key.cell_id}")
    if (
        path.name != "failure.json"
        or re.fullmatch(rf"{re.escape(key.cell_id)}(?:--attempt-\d+)?", path.parent.name) is None
    ):
        raise FewPersonStatisticsError(f"{key.cell_id} failure directory identity differs")
    required = {
        "schema_version": "1.0.0",
        "record_kind": "few_person_failed_run",
        "status": "failed_preserved_postconfirmatory_secondary",
        "evidence_status": "failed_run_not_result",
        "protocol_id": FEW_PERSON_V1_1_PROTOCOL_ID,
        "model_id": key.model_id,
        "seed": key.seed,
        "fold_id": key.fold_id,
        "k": key.k,
        "few_person_manifest_sha256": plan.manifest_sha256,
    }
    mismatches = [field for field, expected in required.items() if record.get(field) != expected]
    if mismatches:
        raise FewPersonStatisticsError(f"{key.cell_id} failure mismatch: {mismatches}")
    _validate_execution_metadata(record, name=f"{key.cell_id} failure")
    elapsed = record.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not np.isfinite(elapsed)
        or float(elapsed) < 0.0
        or not isinstance(record.get("exception_type"), str)
        or not isinstance(record.get("message"), str)
    ):
        raise FewPersonStatisticsError(f"{key.cell_id} failure execution metadata is invalid")
    return key, record


def _scan_progress(
    manifest_path: str | Path,
    *,
    results_root: str | Path,
    artifact_root: str | Path,
) -> _Scan:
    _sha256_file_snapshot.cache_clear()
    plan = _load_plan(manifest_path)
    root = Path(results_root).resolve(strict=True)
    artifacts = Path(artifact_root).resolve(strict=True)
    try:
        root.relative_to(artifacts)
    except ValueError as exc:
        raise FewPersonStatisticsError("few-person results root escapes artifact root") from exc

    def relative(path: Path) -> str:
        return path.resolve(strict=True).relative_to(artifacts).as_posix()

    completed: dict[CellKey, _Completed] = {}
    failures: dict[CellKey, list[dict[str, Any]]] = defaultdict(list)
    errors: list[dict[str, str]] = []
    for path in sorted(root.rglob("result.json")):
        try:
            item = _validate_completed_record(
                path, plan=plan, artifact_root=artifacts, results_root=root
            )
            if item.key in completed:
                raise FewPersonStatisticsError(f"duplicate completed cell: {item.key.cell_id}")
            completed[item.key] = item
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append({"path": relative(path), "message": str(exc)})
    for path in sorted(root.rglob("failure.json")):
        try:
            key, record = _validate_failure(path, plan=plan)
            failures[key].append(
                {
                    "path": relative(path),
                    "record_sha256": record["record_sha256"],
                    "record_file_sha256": sha256_file(path),
                    "exception_type": record.get("exception_type"),
                    "message": record.get("message"),
                }
            )
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append({"path": relative(path), "message": str(exc)})
    fold_alignment: dict[str, tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...]]] = {}
    for key, item in sorted(completed.items()):
        alignment = (
            item.window_ids,
            item.participant_ids,
            tuple(int(value) for value in item.labels.tolist()),
        )
        expected_alignment = fold_alignment.setdefault(key.fold_id, alignment)
        if alignment != expected_alignment:
            errors.append(
                {
                    "path": relative(item.record_path),
                    "message": (
                        f"{key.cell_id} ordered window/participant/label alignment differs "
                        "from its outer fold"
                    ),
                }
            )
    completed_commits = sorted({item.code_commit for item in completed.values()})
    completed_environments = sorted({item.environment_sha256 for item in completed.values()})
    if len(completed_commits) > 1:
        errors.append(
            {
                "path": relative(root),
                "message": "completed cells combine multiple executing Git commits",
            }
        )
    if len(completed_environments) > 1:
        errors.append(
            {
                "path": relative(root),
                "message": "completed cells combine multiple execution environments",
            }
        )
    incomplete_directories: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_dir()):
        if not _RUN_DIRECTORY_PATTERN.fullmatch(path.name):
            continue
        result_exists = (path / "result.json").exists()
        failure_exists = (path / "failure.json").exists()
        if result_exists and failure_exists:
            errors.append(
                {
                    "path": relative(path),
                    "message": "run directory contains both result.json and failure.json",
                }
            )
        elif not result_exists and not failure_exists:
            incomplete_directories.append(relative(path))
    partial_files = sorted(relative(path) for path in root.rglob("*") if ".partial." in path.name)
    missing = sorted(plan.expected_cells - set(completed))
    failure_rows = [
        {**key.to_dict(), "attempts": values} for key, values in sorted(failures.items())
    ]
    preserved_incomplete = bool(incomplete_directories or partial_files)
    if errors:
        status = "invalid"
    elif not missing:
        status = (
            "complete_with_preserved_failures_or_incomplete"
            if failures or preserved_incomplete
            else "complete"
        )
    else:
        status = (
            "in_progress_with_preserved_failures_or_incomplete"
            if failures or preserved_incomplete
            else "in_progress"
        )
    completed_rows = [
        {
            **key.to_dict(),
            "record_path": item.record_path.resolve(strict=True).relative_to(artifacts).as_posix(),
            "record_sha256": item.record_sha256,
            "record_file_sha256": item.record_file_sha256,
            "checkpoint_path": item.checkpoint_path.relative_to(artifacts).as_posix(),
            "checkpoint_sha256": item.checkpoint_sha256,
            "prediction_path": item.prediction_path.relative_to(artifacts).as_posix(),
            "prediction_sha256": item.prediction_sha256,
        }
        for key, item in sorted(completed.items())
    ]
    progress: dict[str, Any] = {
        "schema_version": FEW_PERSON_STATISTICS_SCHEMA_VERSION,
        "record_kind": "few_person_v1_1_progress_validation",
        "status": status,
        "valid_so_far": not errors,
        "aggregation_ready": not errors and not missing,
        "protocol_id": FEW_PERSON_V1_1_PROTOCOL_ID,
        "few_person_manifest_sha256": plan.manifest_sha256,
        "few_person_manifest_file_sha256": plan.manifest_file_sha256,
        "expected": {
            "model_count": len(EXPECTED_MODEL_IDS),
            "seeds_per_model": len(EXPECTED_SEEDS),
            "outer_fold_count": len(EXPECTED_FOLDS),
            "k_values": list(EXPECTED_K_VALUES),
            "cell_count": EXPECTED_CELL_COUNT,
        },
        "observed": {
            "completed_cell_count": len(completed),
            "missing_cell_count": len(missing),
            "cells_with_preserved_failures": len(failures),
            "failure_attempt_count": sum(len(value) for value in failures.values()),
            "invalid_artifact_count": len(errors),
            "incomplete_directory_count": len(incomplete_directories),
            "partial_file_count": len(partial_files),
        },
        "execution_lineage": {
            "code_commits": completed_commits,
            "environment_sha256": completed_environments,
            "environment_hash_scope": (
                "machine_and_software_environment_excluding_model_specific_cudnn_policy"
            ),
        },
        "missing_cells": [key.to_dict() for key in missing],
        "validated_completed_cells": completed_rows,
        "failed_cells": failure_rows,
        "invalid_artifacts": errors,
        "incomplete_directories": incomplete_directories,
        "partial_files": partial_files,
        "resumable_policy": (
            "run each missing cell in a new create-only directory; preserve every prior "
            "failure/incomplete directory; aggregation requires one valid completion per cell"
        ),
        "target_raw_values_accessed": False,
        "gpu_execution_performed": False,
    }
    progress["record_sha256"] = canonical_json_sha256(progress)
    return _Scan(plan=plan, completed=completed, progress=progress)


def validate_few_person_progress(
    manifest_path: str | Path,
    *,
    results_root: str | Path,
    artifact_root: str | Path,
) -> dict[str, Any]:
    """Deeply validate every available artifact without raw data or GPU access."""

    return dict(
        _scan_progress(
            manifest_path, results_root=results_root, artifact_root=artifact_root
        ).progress
    )


def _bootstrap_endpoints(
    participant_values: Mapping[str, float],
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    if set(participant_values) != EXPECTED_TARGET_SUBJECTS or resamples < 100:
        raise FewPersonStatisticsError("bootstrap requires exactly ten target participants")
    if not 0.0 < confidence < 1.0:
        raise FewPersonStatisticsError("bootstrap confidence must lie in (0,1)")
    participants = sorted(participant_values, key=int)
    values = np.asarray([participant_values[value] for value in participants], dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(resamples, values.size))
    draws = values[indices]
    estimates = {
        "mean_participant_macro_f1": draws.mean(axis=1),
        "worst_participant_macro_f1": draws.min(axis=1),
        "lower_decile_participant_macro_f1": np.quantile(draws, 0.1, axis=1, method="linear"),
    }
    point = {
        "mean_participant_macro_f1": float(values.mean()),
        "worst_participant_macro_f1": float(values.min()),
        "lower_decile_participant_macro_f1": float(np.quantile(values, 0.1, method="linear")),
    }
    alpha = 1.0 - confidence
    return {
        "method": "participant_cluster_percentile_after_seed_averaging",
        "participant_count": len(participants),
        "resamples": resamples,
        "confidence": confidence,
        "seed": seed,
        "endpoints": {
            name: {
                "estimate": point[name],
                "lower": float(np.quantile(values_draw, alpha / 2.0)),
                "upper": float(np.quantile(values_draw, 1.0 - alpha / 2.0)),
            }
            for name, values_draw in estimates.items()
        },
    }


def _load_zero_shot(
    index_path: str | Path,
    *,
    artifact_root: Path,
    class_names: tuple[str, ...],
    expected_index_record_sha256: str,
    expected_index_file_sha256: str,
) -> tuple[dict[tuple[str, int], Mapping[str, float]], Mapping[str, Any]]:
    resolved_index_path = _resolve_repository_file(
        str(index_path), artifact_root=artifact_root, name="zero-shot index"
    )
    if _cached_sha256_file(resolved_index_path) != expected_index_file_sha256:
        raise FewPersonStatisticsError("zero-shot index file hash differs from manifest")
    value = load_json_strict(resolved_index_path)
    index = _mapping(value, name="zero-shot index")
    index_record_sha256 = _self_hash(index, field="record_sha256", name="zero-shot index")
    if index_record_sha256 != expected_index_record_sha256:
        raise FewPersonStatisticsError("zero-shot index record hash differs from manifest")
    required = {
        "schema_version": "1.0.0",
        "record_kind": "locked_target_evaluation_index",
        "status": "complete_create_only",
        "evidence_status": "locked_confirmatory_target_opening_1",
        "target_opening_number": 1,
        "class_names": list(class_names),
        "target_information_used_for_model_selection": False,
    }
    mismatches = [key for key, expected in required.items() if index.get(key) != expected]
    if mismatches:
        raise FewPersonStatisticsError(f"zero-shot index mismatch: {mismatches}")
    entries = index.get("results")
    if not isinstance(entries, list):
        raise FewPersonStatisticsError("zero-shot index lacks result entries")
    selected: dict[tuple[str, int], Mapping[str, float]] = {}
    root = artifact_root.resolve(strict=True)
    for value in entries:
        entry = _mapping(value, name="zero-shot index entry")
        model_id = str(entry.get("model_id"))
        seed = entry.get("seed")
        if model_id not in EXPECTED_MODEL_IDS or seed not in EXPECTED_SEEDS:
            continue
        key = (model_id, int(cast(int, seed)))
        if key in selected:
            raise FewPersonStatisticsError(f"duplicate zero-shot model/seed: {key}")
        record_path = _resolve_locked_zero_shot_file(
            entry.get("record_path"),
            artifact_root=root,
            name="zero-shot result",
        )
        if sha256_file(record_path) != entry.get("record_file_sha256"):
            raise FewPersonStatisticsError(f"zero-shot record file hash changed: {key}")
        result = _mapping(load_json_strict(record_path), name="zero-shot result")
        _self_hash(result, field="record_sha256", name="zero-shot result")
        if result.get("record_sha256") != entry.get("record_sha256"):
            raise FewPersonStatisticsError(f"zero-shot record hash differs from index: {key}")
        prediction = _mapping(result.get("prediction_array"), name="zero-shot prediction")
        array_path = _resolve_locked_zero_shot_file(
            prediction.get("path"),
            artifact_root=root,
            name="zero-shot prediction",
        )
        if sha256_file(array_path) != prediction.get("sha256") or prediction.get(
            "sha256"
        ) != entry.get("array_sha256"):
            raise FewPersonStatisticsError(f"zero-shot prediction hash changed: {key}")
        with np.load(array_path, allow_pickle=False) as arrays:
            participant_ids = tuple(str(item) for item in arrays["participant_ids"].tolist())
            labels = np.asarray(arrays["true_labels"], dtype=np.int64)
            probabilities = np.asarray(arrays["calibrated_probabilities"], dtype=np.float64)
            window_ids = tuple(str(item) for item in arrays["window_ids"].tolist())
        if (
            set(participant_ids) != EXPECTED_TARGET_SUBJECTS
            or len(window_ids) != labels.size
            or len(set(window_ids)) != labels.size
            or probabilities.shape != (labels.size, len(class_names))
        ):
            raise FewPersonStatisticsError(f"zero-shot alignment fails: {key}")
        recomputed = classification_report(
            labels, probabilities, participant_ids, class_names=class_names
        )
        stored = _mapping(result.get("participant_level_report"), name="zero-shot report")
        if canonical_json_sha256(recomputed) != canonical_json_sha256(stored):
            raise FewPersonStatisticsError(f"zero-shot metrics do not reconstruct: {key}")
        selected[key] = _participant_values(recomputed)
    return selected, index


def _seed_averaged_curves(
    completed: Mapping[CellKey, _Completed],
) -> dict[tuple[str, int], dict[str, float]]:
    seed_participants: dict[tuple[str, int, int], dict[str, float]] = {}
    for model_id in EXPECTED_MODEL_IDS:
        for k in EXPECTED_K_VALUES:
            for seed in EXPECTED_SEEDS:
                combined: dict[str, float] = {}
                for fold in EXPECTED_FOLDS:
                    item = completed[CellKey(model_id, seed, fold, k)]
                    overlap = set(combined) & set(item.participant_macro_f1)
                    if overlap:
                        raise FewPersonStatisticsError(
                            f"{model_id}/k{k}/seed{seed} repeats held-out participants: {overlap}"
                        )
                    combined.update(item.participant_macro_f1)
                if set(combined) != EXPECTED_TARGET_SUBJECTS:
                    raise FewPersonStatisticsError(
                        f"{model_id}/k{k}/seed{seed} does not combine each target once"
                    )
                seed_participants[(model_id, k, seed)] = combined
    averaged: dict[tuple[str, int], dict[str, float]] = {}
    for model_id in EXPECTED_MODEL_IDS:
        for k in EXPECTED_K_VALUES:
            averaged[(model_id, k)] = {
                participant: float(
                    np.mean(
                        [
                            seed_participants[(model_id, k, seed)][participant]
                            for seed in EXPECTED_SEEDS
                        ]
                    )
                )
                for participant in sorted(EXPECTED_TARGET_SUBJECTS, key=int)
            }
    return averaged


def _zero_seed_average(
    values: Mapping[tuple[str, int], Mapping[str, float]],
) -> dict[tuple[str, int], dict[str, float]]:
    output: dict[tuple[str, int], dict[str, float]] = {}
    for model_id in EXPECTED_MODEL_IDS:
        available = [seed for seed in EXPECTED_SEEDS if (model_id, seed) in values]
        if not available:
            continue
        if tuple(available) != EXPECTED_SEEDS:
            raise FewPersonStatisticsError(f"{model_id} has incomplete k=0 seed coverage")
        if any(set(values[(model_id, seed)]) != EXPECTED_TARGET_SUBJECTS for seed in available):
            raise FewPersonStatisticsError(f"{model_id} k=0 participant coverage differs")
        output[(model_id, 0)] = {
            participant: float(
                np.mean([values[(model_id, seed)][participant] for seed in available])
            )
            for participant in sorted(EXPECTED_TARGET_SUBJECTS, key=int)
        }
    return output


def _summary_row(
    model_id: str,
    k: int,
    values: Mapping[str, float],
    *,
    bootstrap_resamples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
    source: str,
) -> dict[str, Any]:
    bootstrap = _bootstrap_endpoints(
        values,
        resamples=bootstrap_resamples,
        confidence=bootstrap_confidence,
        seed=bootstrap_seed,
    )
    return {
        "model_id": model_id,
        "k": k,
        "source": source,
        "participant_seed_averaged_macro_f1": dict(values),
        "bootstrap": bootstrap,
    }


def _comparisons(curves: Mapping[tuple[str, int], Mapping[str, float]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for model_id in EXPECTED_MODEL_IDS:
        if (model_id, 0) in curves:
            for k in EXPECTED_K_VALUES:
                comparison = paired_participant_comparison(
                    curves[(model_id, 0)], curves[(model_id, k)]
                )
                comparisons.append(
                    {
                        "comparison_id": f"{model_id}:k{k}-vs-k0",
                        "model_id": model_id,
                        "reference_k": 0,
                        "candidate_k": k,
                        **comparison,
                    }
                )
        for reference_k, candidate_k in ((1, 2), (2, 4)):
            comparison = paired_participant_comparison(
                curves[(model_id, reference_k)], curves[(model_id, candidate_k)]
            )
            comparisons.append(
                {
                    "comparison_id": f"{model_id}:k{candidate_k}-vs-k{reference_k}",
                    "model_id": model_id,
                    "reference_k": reference_k,
                    "candidate_k": candidate_k,
                    **comparison,
                }
            )
    permutation = holm_adjust(
        [float(item["permutation"]["two_sided_p_value"]) for item in comparisons]
    )
    wilcoxon = holm_adjust([float(item["wilcoxon"]["two_sided_p_value"]) for item in comparisons])
    for item, permutation_value, wilcoxon_value in zip(
        comparisons, permutation, wilcoxon, strict=True
    ):
        item["multiplicity_family"] = "all_predeclared_models_and_few_person_curve_tests"
        item["holm_adjusted_permutation_p_value"] = permutation_value
        item["holm_adjusted_wilcoxon_p_value"] = wilcoxon_value
    return comparisons


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    fields = [
        "model_id",
        "k",
        "source",
        "participant_count",
        "mean_participant_macro_f1",
        "mean_ci_lower",
        "mean_ci_upper",
        "worst_participant_macro_f1",
        "worst_ci_lower",
        "worst_ci_upper",
        "lower_decile_participant_macro_f1",
        "lower_decile_ci_lower",
        "lower_decile_ci_upper",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        endpoints = cast(Mapping[str, Mapping[str, float]], row["bootstrap"]["endpoints"])
        writer.writerow(
            {
                "model_id": row["model_id"],
                "k": row["k"],
                "source": row["source"],
                "participant_count": row["bootstrap"]["participant_count"],
                "mean_participant_macro_f1": endpoints["mean_participant_macro_f1"]["estimate"],
                "mean_ci_lower": endpoints["mean_participant_macro_f1"]["lower"],
                "mean_ci_upper": endpoints["mean_participant_macro_f1"]["upper"],
                "worst_participant_macro_f1": endpoints["worst_participant_macro_f1"]["estimate"],
                "worst_ci_lower": endpoints["worst_participant_macro_f1"]["lower"],
                "worst_ci_upper": endpoints["worst_participant_macro_f1"]["upper"],
                "lower_decile_participant_macro_f1": endpoints["lower_decile_participant_macro_f1"][
                    "estimate"
                ],
                "lower_decile_ci_lower": endpoints["lower_decile_participant_macro_f1"]["lower"],
                "lower_decile_ci_upper": endpoints["lower_decile_participant_macro_f1"]["upper"],
            }
        )
    return output.getvalue().encode("utf-8")


def _markdown_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    lines = [
        "# Few-person inclusion curve v1.1",
        "",
        "Status: post-confirmatory secondary evidence. Participants, not windows, are the "
        "inferential units; participant macro-F1 is averaged across five seeds first.",
        "",
        "| Model | k | Mean macro-F1 (95% CI) | Worst | Lower decile |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        endpoints = cast(Mapping[str, Mapping[str, float]], row["bootstrap"]["endpoints"])
        mean = endpoints["mean_participant_macro_f1"]
        worst = endpoints["worst_participant_macro_f1"]
        lower = endpoints["lower_decile_participant_macro_f1"]
        lines.append(
            f"| {row['model_id']} | {row['k']} | {mean['estimate']:.4f} "
            f"[{mean['lower']:.4f}, {mean['upper']:.4f}] | "
            f"{worst['estimate']:.4f} | {lower['estimate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Intervals are deterministic participant-cluster percentile bootstrap intervals. "
            "Paired sign-flip and Wilcoxon tests with global Holm correction are in JSON.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _write_bytes_new(path: Path, payload: bytes) -> str:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def _preflight_statistics_destinations(
    output_directory: str | Path, prefix: str, *, allowed_root: str | Path
) -> tuple[Path, Path, Path, Path]:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", prefix) is None:
        raise FewPersonStatisticsError("statistics prefix must be one portable filename stem")
    root_candidate = Path(allowed_root)
    if root_candidate.is_symlink():
        raise FewPersonStatisticsError("statistics allowed root must not be a symlink")
    root = root_candidate.resolve(strict=True)
    requested_output = Path(output_directory)
    candidate = requested_output if requested_output.is_absolute() else root / requested_output
    if os.path.lexists(candidate) and candidate.is_symlink():
        raise FewPersonStatisticsError("statistics output directory must not be a symlink")
    if not os.path.lexists(candidate):
        if candidate.parent.is_symlink():
            raise FewPersonStatisticsError("statistics output parent is unsafe")
        parent = candidate.parent.resolve(strict=True)
        try:
            parent.relative_to(root)
        except ValueError as exc:
            raise FewPersonStatisticsError(
                "statistics output directory escapes allowed root"
            ) from exc
        if not parent.is_dir():
            raise FewPersonStatisticsError("statistics output parent is unsafe")
        candidate.mkdir(exist_ok=False)
    output = candidate.resolve(strict=True)
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise FewPersonStatisticsError("statistics output directory escapes allowed root") from exc
    if not output.is_dir():
        raise FewPersonStatisticsError("statistics output directory must be a regular directory")
    json_path = output / f"{prefix}.json"
    csv_path = output / f"{prefix}.csv"
    markdown_path = output / f"{prefix}.md"
    for path in (csv_path, markdown_path, json_path):
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to overwrite statistics export: {path}")
    return output, json_path, csv_path, markdown_path


def aggregate_few_person_statistics(
    manifest_path: str | Path,
    *,
    results_root: str | Path,
    artifact_root: str | Path,
    output_directory: str | Path,
    prefix: str = "few_person_v1_1_statistics",
    zero_shot_index_path: str | Path | None = None,
    bootstrap_resamples: int = 10_000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 1729,
    created_at_utc: str,
    aggregation_code_commit: str,
) -> dict[str, Any]:
    """Aggregate exact complete coverage after reconstructing every stored metric."""

    root = Path(artifact_root).resolve(strict=True)
    commit = aggregation_code_commit.casefold()
    if _GIT_COMMIT_PATTERN.fullmatch(commit) is None:
        raise FewPersonStatisticsError("aggregation code commit must be a full Git object ID")
    _require_repository_head(root, expected_commit=commit)
    output, json_path, csv_path, markdown_path = _preflight_statistics_destinations(
        output_directory, prefix, allowed_root=artifact_root
    )
    timestamp = require_utc_timestamp(created_at_utc, location="created_at_utc")
    scan = _scan_progress(manifest_path, results_root=results_root, artifact_root=artifact_root)
    if scan.progress["valid_so_far"] is not True or scan.progress["aggregation_ready"] is not True:
        raise FewPersonStatisticsError(
            "few-person aggregation requires all 1,200 valid cells; run validate-progress"
        )
    curves = _seed_averaged_curves(scan.completed)
    zero_index: Mapping[str, Any] | None = None
    if zero_shot_index_path is not None:
        zero_values, zero_index = _load_zero_shot(
            zero_shot_index_path,
            artifact_root=Path(artifact_root),
            class_names=scan.plan.class_names,
            expected_index_record_sha256=str(scan.plan.record["zero_shot_index_record_sha256"]),
            expected_index_file_sha256=str(scan.plan.record["zero_shot_index_file_sha256"]),
        )
        curves.update(_zero_seed_average(zero_values))
    summaries: list[dict[str, Any]] = []
    for model_id in EXPECTED_MODEL_IDS:
        for k in (0, *EXPECTED_K_VALUES):
            if (model_id, k) not in curves:
                continue
            summaries.append(
                _summary_row(
                    model_id,
                    k,
                    curves[(model_id, k)],
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_confidence=bootstrap_confidence,
                    bootstrap_seed=bootstrap_seed,
                    source="locked_zero_shot_k0" if k == 0 else "few_person_v1_1",
                )
            )
    comparisons = _comparisons(curves)
    csv_payload = _csv_bytes(summaries)
    markdown_payload = _markdown_bytes(summaries)
    payload: dict[str, Any] = {
        "schema_version": FEW_PERSON_STATISTICS_SCHEMA_VERSION,
        "record_kind": "few_person_v1_1_participant_statistics",
        "created_at_utc": timestamp,
        "status": "complete_create_only_postconfirmatory_secondary",
        "evidence_status": FEW_PERSON_V1_1_EVIDENCE_STATUS,
        "protocol_id": FEW_PERSON_V1_1_PROTOCOL_ID,
        "few_person_manifest_sha256": scan.plan.manifest_sha256,
        "aggregation_code_commit": commit,
        "validated_scenario_result_count": len(scan.completed),
        "execution_lineage": scan.progress["execution_lineage"],
        "exact_coverage": {
            "model_ids": list(EXPECTED_MODEL_IDS),
            "seeds": list(EXPECTED_SEEDS),
            "outer_folds": list(EXPECTED_FOLDS),
            "k_values": list(EXPECTED_K_VALUES),
            "cell_count": EXPECTED_CELL_COUNT,
        },
        "zero_shot_comparator": None
        if zero_index is None
        else {
            "index_record_sha256": zero_index["record_sha256"],
            "index_file_sha256": sha256_file(cast(str | Path, zero_shot_index_path)),
            "available_model_ids": sorted(
                model_id for model_id in EXPECTED_MODEL_IDS if (model_id, 0) in curves
            ),
        },
        "statistical_unit": "participant",
        "seed_handling": (
            "average each held-out participant macro-F1 over five seeds before inference"
        ),
        "fold_handling": (
            "combine each target participant from its single held-out outer fold exactly once"
        ),
        "bootstrap": {
            "method": "participant_cluster_percentile_after_seed_averaging",
            "resamples": bootstrap_resamples,
            "confidence": bootstrap_confidence,
            "seed": bootstrap_seed,
        },
        "models_and_k": summaries,
        "paired_comparisons": comparisons,
        "multiplicity": {
            "method": "Holm",
            "family": "all_predeclared_models_and_few_person_curve_tests",
            "comparison_count": len(comparisons),
        },
        "preserved_failures": scan.progress["failed_cells"],
        "exports": {
            "csv": {
                "filename": csv_path.name,
                "sha256": hashlib.sha256(csv_payload).hexdigest(),
            },
            "markdown": {
                "filename": markdown_path.name,
                "sha256": hashlib.sha256(markdown_payload).hexdigest(),
            },
        },
        "independence_note": "windows are never treated as independent inferential units",
        "target_information_used_for_model_or_hyperparameter_selection": False,
        "gpu_execution_performed_during_aggregation": False,
        "raw_target_values_accessed_during_aggregation": False,
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    if _write_bytes_new(csv_path, csv_payload) != payload["exports"]["csv"]["sha256"]:
        raise FewPersonStatisticsError("CSV export hash differs after write")
    if (
        _write_bytes_new(markdown_path, markdown_payload)
        != payload["exports"]["markdown"]["sha256"]
    ):
        raise FewPersonStatisticsError("Markdown export hash differs after write")
    # JSON is the completion marker and is intentionally published last.
    atomic_write_json_new(payload, json_path, allowed_root=output)
    return payload


def _write_progress_new(record: Mapping[str, Any], path: Path, *, output_root: Path) -> None:
    _self_hash(record, field="record_sha256", name="progress record")
    atomic_write_json_new(dict(record), path, allowed_root=output_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    progress = subparsers.add_parser("validate-progress")
    progress.add_argument("--manifest", type=Path, required=True)
    progress.add_argument("--results-root", type=Path, required=True)
    progress.add_argument("--artifact-root", type=Path, required=True)
    progress.add_argument("--output", type=Path)
    progress.add_argument("--output-root", type=Path)
    progress.add_argument("--full-json", action="store_true")
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--manifest", type=Path, required=True)
    aggregate.add_argument("--results-root", type=Path, required=True)
    aggregate.add_argument("--artifact-root", type=Path, required=True)
    aggregate.add_argument("--output-directory", type=Path, required=True)
    aggregate.add_argument("--prefix", default="few_person_v1_1_statistics")
    aggregate.add_argument("--zero-shot-index", type=Path)
    aggregate.add_argument("--bootstrap-resamples", type=int, default=10_000)
    aggregate.add_argument("--bootstrap-confidence", type=float, default=0.95)
    aggregate.add_argument("--bootstrap-seed", type=int, default=1729)
    aggregate.add_argument("--created-at-utc", required=True)
    aggregate.add_argument("--aggregation-code-commit", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate-progress":
            record = validate_few_person_progress(
                args.manifest,
                results_root=args.results_root,
                artifact_root=args.artifact_root,
            )
            if args.output is not None:
                if args.output_root is None:
                    raise FewPersonStatisticsError("--output requires --output-root")
                _write_progress_new(record, args.output, output_root=args.output_root)
            payload: Mapping[str, Any] = (
                record
                if args.full_json
                else {
                    "status": record["status"],
                    "valid_so_far": record["valid_so_far"],
                    "aggregation_ready": record["aggregation_ready"],
                    "observed": record["observed"],
                    "record_sha256": record["record_sha256"],
                }
            )
            exit_code = 0 if record["valid_so_far"] is True else 2
        else:
            record = aggregate_few_person_statistics(
                args.manifest,
                results_root=args.results_root,
                artifact_root=args.artifact_root,
                output_directory=args.output_directory,
                prefix=args.prefix,
                zero_shot_index_path=args.zero_shot_index,
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_confidence=args.bootstrap_confidence,
                bootstrap_seed=args.bootstrap_seed,
                created_at_utc=args.created_at_utc,
                aggregation_code_commit=args.aggregation_code_commit,
            )
            payload = {
                "status": record["status"],
                "record_sha256": record["record_sha256"],
                "validated_scenario_result_count": record["validated_scenario_result_count"],
            }
            exit_code = 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "fail", "message": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
