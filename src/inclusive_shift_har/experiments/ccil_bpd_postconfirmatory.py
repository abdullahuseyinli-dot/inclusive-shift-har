"""CUDA-only post-confirmatory CCIL/BPD paper-adaptation extension.

The operation fits only on the frozen source-training participant partition,
uses source validation for candidate selection and temperature calibration, and
loads target signals only after a durable source-stage lock exists.  Target
signals come exclusively from the hash-pinned cache bound to consumed opening
1; this module has no raw-data, unlock, or new-opening interface.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import random
import re
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from inclusive_shift_har.data.inclusivehar import INCLUSIVEHAR_PRIMARY_CHANNELS
from inclusive_shift_har.evaluation._strict_config import (
    StrictConfigError,
    load_strict_yaml_mapping,
    require_exact_keys,
    require_mapping,
    require_utc_timestamp,
)
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.evaluation.postconfirmatory_inputs import (
    ConsumedTargetContext,
    load_consumed_target_context,
    load_target_clean_reference,
)
from inclusive_shift_har.evaluation.source_calibration import (
    build_source_temperature_calibrator_record,
    load_source_temperature_calibrator_file,
    write_source_temperature_calibrator_new,
)
from inclusive_shift_har.experiments.postconfirmatory_cache import (
    PreparedPrimaryCacheEvidence,
    load_prepared_primary_cache,
)
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)
from inclusive_shift_har.models.baselines import CompactResidualHAR
from inclusive_shift_har.models.common import HAROutput, trainable_parameter_count
from inclusive_shift_har.models.paper_adaptations import (
    BPDBoundarySafeCompactAdapter,
    mine_dv_lower_bound,
    redundant_class_confusion_loss,
)
from inclusive_shift_har.preprocessing.normalization import ChannelStandardizer
from inclusive_shift_har.training.ccil_paper import CCILPaperConceptMeanLoss
from inclusive_shift_har.training.engine import (
    WindowTensorDataset,
    configure_determinism,
    predict_model,
)

SCHEMA_VERSION = "1.0.0"
EVIDENCE_STATUS = "post_confirmatory_descriptive_paper_adaptation"
TRACK_ROLE = "post_confirmatory_extension_not_primary_claim_evidence"
METHOD_IDS = (
    "ccil-paper-loss-compact",
    "bpd-boundary-safe-compact-adaptation",
)
COMPARATOR_IDS = ("compact-erm", "more-har-full")
SEED_ORDER = (11, 23, 47, 89, 131)
SOURCE_TRAIN_PARTICIPANTS = ("1", "2", "3", "4", "5", "6", "7", "9")
SOURCE_VALIDATION_PARTICIPANTS = ("8", "10")
TARGET_PARTICIPANTS = tuple(str(value) for value in range(11, 21))
CLASS_NAMES = ("mobility", "sitting", "standing")
EXPECTED_WINDOWS = {"source_train": 582, "source_validation": 143, "target_sealed": 807}
EXPECTED_SPLIT_SHA256 = "ccb6c3d1254c1464c48e412afb6f83e7942113299e853f89c77df1d6cfad131b"
EXPECTED_SOURCE_SHA256 = "0209542286e9031dc64a65abb47e1d84676088b1a77e52f738635887cb9bce34"
EXPECTED_SPLIT_FILE_SHA256 = "af909c7d914b68c417b238b576945c112aa58740859a7d998b6b77219dfb1920"
EXPECTED_DATASET_MANIFEST_FILE_SHA256 = (
    "52de5370682f13fbd9a4e9affe805b4f5ee0f28901d137743884b77471b24d29"
)
EXPECTED_PREPROCESSING_FILE_SHA256 = (
    "bb23327375f491708c81aa2a605b0a24e9a8274a20dee9ef66b8232cc9525145"
)
EXPECTED_PREPROCESSING_CANONICAL_SHA256 = (
    "48515caf9da8e20510913baab0bcf9dd874af26e6ca415663e36be81d7848125"
)
EXPECTED_ONTOLOGY_SHA256 = "85f6d8ed911420f51c971da05539b739bad30c9e1816d376445abd18ade6d03e"
EXPECTED_OPENING_RECEIPT_FILE_SHA256 = (
    "71711626f595c354d00fe3950127516a86cfabae8bef76bf54018c98707ab719"
)
EXPECTED_OPENING_RECEIPT_RECORD_SHA256 = (
    "5704f65efd416e9cd16d6ed24c2735c1d52b7fcaafddb113e481498797cd182a"
)
EXPECTED_TARGET_INDEX_FILE_SHA256 = (
    "ae0daace937bc275bce2edd895f1c05eb540f61d1554eda471c9d2f27bf60e49"
)
EXPECTED_TARGET_INDEX_RECORD_SHA256 = (
    "79434d8fbc136cb55e18fa980490e5aaa94a91fb3c823837cccf6137b026b5b9"
)
EXPECTED_TARGET_SEAL_ID = "aecba05fa4a0fc4e4bbc135ac30944b686be19c0b6a2820c838e6c8b802bb29d"
EXPECTED_FREEZE_FILE_SHA256 = "e62ca1233c55ef086f3d7f2352c9dca8ef5a8eb6ce8a3a568e2e3ee53d5af690"
EXPECTED_FREEZE_INVENTORY_SHA256 = (
    "d867ade3778fd07a1bb1404121d4f84323ddb66fe9bf15705603175fcf74b070"
)
EXPECTED_FROZEN_ARTIFACT_SET_SHA256 = (
    "e759b60f32b965e7ae3e5a994d919a08553c4958f9bdcf10d7497697f685dd51"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class PaperAdaptationRunError(RuntimeError):
    """Raised when the post-confirmatory operation violates its frozen boundary."""


@dataclass(frozen=True, slots=True)
class AdaptationCandidate:
    method_id: str
    candidate_id: str
    parameters: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class PaperAdaptationConfig:
    path: Path
    file_sha256: str
    canonical_sha256: str
    experiment_id: str
    protocol: Mapping[str, Any]
    opening_1: Mapping[str, Any]
    comparator: Mapping[str, Any]
    execution: Mapping[str, Any]
    training: Mapping[str, Any]
    methods: tuple[Mapping[str, Any], ...]
    candidates: Mapping[str, tuple[AdaptationCandidate, ...]]
    selection: Mapping[str, Any]
    statistics: Mapping[str, Any]
    outputs: Mapping[str, Any]
    limitations: tuple[str, ...]
    forbidden_claims: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceArrays:
    normalizer: ChannelStandardizer
    train_windows: NDArray[np.float32]
    train_labels: NDArray[np.int64]
    train_participants: tuple[str, ...]
    train_window_ids: tuple[str, ...]
    validation_windows: NDArray[np.float32]
    validation_labels: NDArray[np.int64]
    validation_participants: tuple[str, ...]
    validation_window_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrainedAdapter:
    method_id: str
    candidate: AdaptationCandidate
    seed: int
    checkpoint_path: Path
    checkpoint_sha256: str
    configuration: Mapping[str, Any]
    configuration_sha256: str
    validation_logits: NDArray[np.float64]
    validation_probabilities: NDArray[np.float64]
    validation_report: Mapping[str, Any]
    history: tuple[Mapping[str, Any], ...]
    parameter_count: int
    peak_vram_bytes: int
    elapsed_seconds: float


def _string(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise StrictConfigError(f"{location} must be a non-empty string")
    return value


def _integer(value: Any, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StrictConfigError(f"{location} must be an integer")
    return int(value)


def _number(value: Any, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StrictConfigError(f"{location} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise StrictConfigError(f"{location} must be finite")
    return result


def _strings(value: Any, *, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise StrictConfigError(f"{location} must be a list of non-empty strings")
    return tuple(value)


def _integers(value: Any, *, location: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise StrictConfigError(f"{location} must be a list of integers")
    return tuple(value)


def _require_value(actual: Any, expected: Any, *, location: str) -> None:
    if actual != expected:
        raise StrictConfigError(f"{location} is {actual!r}; required {expected!r}")


def load_paper_adaptation_config(path: str | Path) -> PaperAdaptationConfig:
    """Load the exact post-confirmatory design and reject silent drift."""

    config_path = Path(path).resolve(strict=True)
    raw = load_strict_yaml_mapping(config_path)
    require_exact_keys(
        raw,
        {
            "schema_version",
            "experiment_id",
            "status",
            "evidence_status",
            "designed_after_target_opening",
            "primary_claim_eligible",
            "protocol",
            "opening_1",
            "comparator",
            "execution",
            "training",
            "methods",
            "selection",
            "statistics",
            "outputs",
            "limitations",
            "forbidden_claims",
        },
        location="$",
    )
    constants = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "ccil-bpd-postconfirmatory-v1",
        "status": "post_confirmatory_extension_predeclared_after_consumed_opening_1",
        "evidence_status": EVIDENCE_STATUS,
        "designed_after_target_opening": True,
        "primary_claim_eligible": False,
    }
    for key, expected in constants.items():
        _require_value(raw.get(key), expected, location=key)

    protocol = require_mapping(raw["protocol"], location="protocol")
    require_exact_keys(
        protocol,
        {
            "description",
            "trial_safe",
            "hidden_join_unconditional_risk_bound",
            "split_manifest_path",
            "split_manifest_sha256",
            "split_manifest_file_sha256",
            "source_artifact_sha256",
            "dataset_manifest_path",
            "dataset_manifest_file_sha256",
            "preprocessing_config_path",
            "preprocessing_config_file_sha256",
            "ontology_config_sha256",
            "ontology_track",
            "class_names",
            "channels",
            "window_shape",
            "window_stride_samples",
            "source_train_participants",
            "source_validation_participants",
            "target_participants",
            "expected_source_train_windows",
            "expected_source_validation_windows",
            "expected_target_windows",
        },
        location="protocol",
    )
    protocol_constants = {
        "description": "participant-exclusive released-block protocol; not trial-safe",
        "trial_safe": False,
        "hidden_join_unconditional_risk_bound": 1.0,
        "split_manifest_path": ("results/protocol/splits/inclusivehar_v4_released_block_v1_2.json"),
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "split_manifest_file_sha256": EXPECTED_SPLIT_FILE_SHA256,
        "source_artifact_sha256": EXPECTED_SOURCE_SHA256,
        "dataset_manifest_path": "manifests/datasets/inclusivehar_v4.json",
        "dataset_manifest_file_sha256": EXPECTED_DATASET_MANIFEST_FILE_SHA256,
        "preprocessing_config_path": "configs/preprocessing/inclusivehar_primary_128.yaml",
        "preprocessing_config_file_sha256": EXPECTED_PREPROCESSING_FILE_SHA256,
        "ontology_config_sha256": EXPECTED_ONTOLOGY_SHA256,
        "ontology_track": "functional_core",
        "class_names": list(CLASS_NAMES),
        "channels": list(INCLUSIVEHAR_PRIMARY_CHANNELS),
        "window_shape": [128, 6],
        "window_stride_samples": 128,
        "source_train_participants": list(SOURCE_TRAIN_PARTICIPANTS),
        "source_validation_participants": list(SOURCE_VALIDATION_PARTICIPANTS),
        "target_participants": list(TARGET_PARTICIPANTS),
        "expected_source_train_windows": EXPECTED_WINDOWS["source_train"],
        "expected_source_validation_windows": EXPECTED_WINDOWS["source_validation"],
        "expected_target_windows": EXPECTED_WINDOWS["target_sealed"],
    }
    for key, expected in protocol_constants.items():
        _require_value(protocol.get(key), expected, location=f"protocol.{key}")
    opening = require_mapping(raw["opening_1"], location="opening_1")
    require_exact_keys(
        opening,
        {
            "opening_receipt_path",
            "opening_receipt_file_sha256",
            "opening_receipt_record_sha256",
            "locked_target_index_path",
            "locked_target_index_file_sha256",
            "locked_target_index_record_sha256",
            "target_seal_id",
            "unlock_or_new_opening_invocation_allowed",
        },
        location="opening_1",
    )
    opening_constants = {
        "opening_receipt_path": "results/protocol/confirmatory_target_opening_1.json",
        "opening_receipt_file_sha256": EXPECTED_OPENING_RECEIPT_FILE_SHA256,
        "opening_receipt_record_sha256": EXPECTED_OPENING_RECEIPT_RECORD_SHA256,
        "locked_target_index_path": (
            "results/confirmatory/zero_shot_v1/locked_target_evaluation_index.json"
        ),
        "locked_target_index_file_sha256": EXPECTED_TARGET_INDEX_FILE_SHA256,
        "locked_target_index_record_sha256": EXPECTED_TARGET_INDEX_RECORD_SHA256,
        "target_seal_id": EXPECTED_TARGET_SEAL_ID,
        "unlock_or_new_opening_invocation_allowed": False,
    }
    for key, expected in opening_constants.items():
        _require_value(opening.get(key), expected, location=f"opening_1.{key}")

    comparator = require_mapping(raw["comparator"], location="comparator")
    require_exact_keys(
        comparator,
        {
            "model_ids",
            "architectures",
            "source_model_names",
            "evidence",
            "final_freeze_path",
            "final_freeze_file_sha256",
            "final_freeze_inventory_sha256",
            "frozen_artifact_set_sha256",
            "source_record_templates",
            "source_calibrator_templates",
        },
        location="comparator",
    )
    _require_value(
        comparator.get("model_ids"), list(COMPARATOR_IDS), location="comparator.model_ids"
    )
    architectures = require_mapping(
        comparator.get("architectures"), location="comparator.architectures"
    )
    source_model_names = require_mapping(
        comparator.get("source_model_names"), location="comparator.source_model_names"
    )
    record_templates = require_mapping(
        comparator.get("source_record_templates"),
        location="comparator.source_record_templates",
    )
    calibrator_templates = require_mapping(
        comparator.get("source_calibrator_templates"),
        location="comparator.source_calibrator_templates",
    )
    for name, mapping in (
        ("architectures", architectures),
        ("source_model_names", source_model_names),
        ("source_record_templates", record_templates),
        ("source_calibrator_templates", calibrator_templates),
    ):
        require_exact_keys(mapping, set(COMPARATOR_IDS), location=f"comparator.{name}")
        for model_id in COMPARATOR_IDS:
            _string(mapping[model_id], location=f"comparator.{name}.{model_id}")
    _require_value(
        dict(architectures),
        {
            "compact-erm": "compact_residual_96",
            "more-har-full": "locked MoRe-HAR full model",
        },
        location="comparator.architectures",
    )
    _require_value(
        dict(source_model_names),
        {"compact-erm": "compact_residual_96", "more-har-full": "more_har"},
        location="comparator.source_model_names",
    )
    _require_value(
        dict(record_templates),
        {
            "compact-erm": ("results/final/source_only_v1/records/compact-erm--seed-{seed}.json"),
            "more-har-full": (
                "results/final/source_only_v1/records/more-har-full--seed-{seed}.json"
            ),
        },
        location="comparator.source_record_templates",
    )
    _require_value(
        dict(calibrator_templates),
        {
            "compact-erm": (
                "results/final/source_only_v1/calibrators/compact-erm--seed-{seed}.json"
            ),
            "more-har-full": (
                "results/final/source_only_v1/calibrators/more-har-full--seed-{seed}.json"
            ),
        },
        location="comparator.source_calibrator_templates",
    )
    _require_value(
        comparator.get("evidence"),
        "frozen source-only artifacts and consumed opening-1 results; neither comparator is rerun",
        location="comparator.evidence",
    )
    comparator_constants = {
        "final_freeze_path": "results/protocol/final_source_artifact_freeze_v1.json",
        "final_freeze_file_sha256": EXPECTED_FREEZE_FILE_SHA256,
        "final_freeze_inventory_sha256": EXPECTED_FREEZE_INVENTORY_SHA256,
        "frozen_artifact_set_sha256": EXPECTED_FROZEN_ARTIFACT_SET_SHA256,
    }
    for key, expected in comparator_constants.items():
        _require_value(comparator.get(key), expected, location=f"comparator.{key}")

    execution = require_mapping(raw["execution"], location="execution")
    require_exact_keys(
        execution,
        {
            "required_device",
            "one_process_one_gpu",
            "sequential_runs",
            "required_seed_order",
            "selection_seed",
            "source_checkpoint_rule",
            "target_stage_after_durable_source_lock",
            "target_tuning_or_refit",
            "mixed_precision",
        },
        location="execution",
    )
    execution_constants = {
        "required_device": "cuda",
        "one_process_one_gpu": True,
        "sequential_runs": True,
        "required_seed_order": list(SEED_ORDER),
        "selection_seed": SEED_ORDER[0],
        "source_checkpoint_rule": "fixed_last_epoch_23_matching_frozen_compact_erm",
        "target_stage_after_durable_source_lock": True,
        "target_tuning_or_refit": False,
        "mixed_precision": "float16",
    }
    for key, expected in execution_constants.items():
        _require_value(execution.get(key), expected, location=f"execution.{key}")

    training = require_mapping(raw["training"], location="training")
    require_exact_keys(
        training,
        {
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "data_loader_workers",
            "optimizer",
            "scheduler",
            "normalization",
            "calibration",
        },
        location="training",
    )
    training_constants = {
        "epochs": 23,
        "batch_size": 256,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 5.0,
        "data_loader_workers": 0,
        "optimizer": "adamw",
        "scheduler": "cosine_annealing_per_completed_epoch",
        "normalization": "per_channel_population_standardization_fit_source_train_only",
        "calibration": "scalar_temperature_fit_source_validation_only_per_final_seed",
    }
    for key, expected in training_constants.items():
        _require_value(training.get(key), expected, location=f"training.{key}")

    methods_value = raw["methods"]
    if not isinstance(methods_value, list) or len(methods_value) != len(METHOD_IDS):
        raise StrictConfigError("methods must contain exactly the CCIL and BPD adaptations")
    methods: list[Mapping[str, Any]] = []
    candidate_map: dict[str, tuple[AdaptationCandidate, ...]] = {}
    for index, value in enumerate(methods_value):
        method = require_mapping(value, location=f"methods[{index}]")
        method_id = _string(method.get("model_id"), location=f"methods[{index}].model_id")
        if method_id != METHOD_IDS[index]:
            raise StrictConfigError("method order or identity changed")
        required_common = {
            "model_id",
            "claim_label",
            "architecture",
            "official_code_used",
            "paper_doi",
            "implementation_module",
            "selection_candidates",
        }
        required = (
            required_common
            if method_id == METHOD_IDS[0]
            else required_common | {"third_party_code_copied", "latent_dim", "mine_steps_per_batch"}
        )
        require_exact_keys(method, required, location=f"methods[{index}]")
        _require_value(
            method.get("official_code_used"), False, location=f"methods[{index}].official_code_used"
        )
        claim_label = _string(
            method.get("claim_label"), location=f"methods[{index}].claim_label"
        ).casefold()
        required_qualifier = (
            "not official code or faithful reproduction"
            if method_id == METHOD_IDS[0]
            else "not official-faithful bpd"
        )
        if required_qualifier not in claim_label:
            raise StrictConfigError(f"{method_id} claim label lost its required qualifier")
        if method_id == METHOD_IDS[1]:
            _require_value(
                method.get("third_party_code_copied"),
                False,
                location=f"methods[{index}].third_party_code_copied",
            )
            _require_value(method.get("latent_dim"), 24, location=f"methods[{index}].latent_dim")
            _require_value(
                method.get("mine_steps_per_batch"),
                1,
                location=f"methods[{index}].mine_steps_per_batch",
            )
        candidates_value = method.get("selection_candidates")
        if not isinstance(candidates_value, list) or len(candidates_value) != 4:
            raise StrictConfigError(f"{method_id} must have exactly four source candidates")
        candidates: list[AdaptationCandidate] = []
        for candidate_index, candidate_value in enumerate(candidates_value):
            candidate = require_mapping(
                candidate_value,
                location=f"methods[{index}].selection_candidates[{candidate_index}]",
            )
            parameter_names = (
                {"alpha", "ema_update_weight"}
                if method_id == METHOD_IDS[0]
                else {
                    "mutual_information_weight",
                    "confusion_weight",
                    "reconstruction_weight",
                }
            )
            require_exact_keys(
                candidate,
                {"candidate_id", *parameter_names},
                location=f"methods[{index}].selection_candidates[{candidate_index}]",
            )
            candidate_id = _string(candidate["candidate_id"], location=f"{method_id} candidate_id")
            parameters = {
                name: _number(candidate[name], location=f"{method_id}.{candidate_id}.{name}")
                for name in sorted(parameter_names)
            }
            if any(number <= 0 for number in parameters.values()):
                raise StrictConfigError("adaptation candidate weights must be positive")
            candidates.append(AdaptationCandidate(method_id, candidate_id, parameters))
        if len({value.candidate_id for value in candidates}) != 4:
            raise StrictConfigError(f"{method_id} candidate IDs must be unique")
        methods.append(method)
        candidate_map[method_id] = tuple(candidates)

    selection = require_mapping(raw["selection"], location="selection")
    require_exact_keys(
        selection,
        {
            "equal_candidate_count_per_method",
            "fit_partition",
            "scoring_partition",
            "primary_metric",
            "tie_breakers",
            "target_signals_labels_predictions_or_metrics_allowed",
        },
        location="selection",
    )
    selection_constants = {
        "equal_candidate_count_per_method": 4,
        "fit_partition": "source_train",
        "scoring_partition": "source_validation",
        "primary_metric": "mean_participant_macro_f1",
        "tie_breakers": ["worst_participant_macro_f1", "candidate_id_lexicographic"],
        "target_signals_labels_predictions_or_metrics_allowed": False,
    }
    for key, expected in selection_constants.items():
        _require_value(selection.get(key), expected, location=f"selection.{key}")

    statistics = require_mapping(raw["statistics"], location="statistics")
    require_exact_keys(
        statistics,
        {
            "statistical_unit",
            "seed_aggregation",
            "comparison_references",
            "participant_bootstrap_resamples",
            "participant_bootstrap_seed",
            "paired_permutation_samples",
            "paired_permutation_seed",
            "multiple_comparison_correction",
        },
        location="statistics",
    )
    statistics_constants = {
        "statistical_unit": "participant",
        "seed_aggregation": "mean_each_participant_across_five_seeds_before_model_comparison",
        "comparison_references": list(COMPARATOR_IDS),
        "participant_bootstrap_resamples": 10_000,
        "participant_bootstrap_seed": 1729,
        "paired_permutation_samples": 100_000,
        "paired_permutation_seed": 2718,
        "multiple_comparison_correction": (
            "holm_across_four_adapter_vs_locked_comparator_target_comparisons"
        ),
    }
    for key, expected in statistics_constants.items():
        _require_value(statistics.get(key), expected, location=f"statistics.{key}")
    outputs = require_mapping(raw["outputs"], location="outputs")
    require_exact_keys(
        outputs,
        {
            "output_directory",
            "selection_lock_filename",
            "source_stage_lock_filename",
            "index_filename",
            "aggregate_filename",
        },
        location="outputs",
    )
    for key, value in outputs.items():
        _string(value, location=f"outputs.{key}")
    limitations = _strings(raw["limitations"], location="limitations")
    forbidden = _strings(raw["forbidden_claims"], location="forbidden_claims")
    if len(limitations) < 5 or len(forbidden) < 5:
        raise StrictConfigError("limitations and forbidden claims are incomplete")
    return PaperAdaptationConfig(
        path=config_path,
        file_sha256=sha256_file(config_path),
        canonical_sha256=canonical_json_sha256(raw),
        experiment_id=str(raw["experiment_id"]),
        protocol=protocol,
        opening_1=opening,
        comparator=comparator,
        execution=execution,
        training=training,
        methods=tuple(methods),
        candidates=candidate_map,
        selection=selection,
        statistics=statistics,
        outputs=outputs,
        limitations=limitations,
        forbidden_claims=forbidden,
    )


def _build_model(method_id: str) -> nn.Module:
    if method_id == METHOD_IDS[0]:
        return CompactResidualHAR(3, width=96, blocks=5)
    if method_id == METHOD_IDS[1]:
        return BPDBoundarySafeCompactAdapter(3, backbone_width=96, backbone_blocks=5)
    raise ValueError(f"unsupported paper adaptation: {method_id}")


def _training_configuration(
    config: PaperAdaptationConfig,
    candidate: AdaptationCandidate,
    *,
    seed: int,
    role: str,
    primary_cache_record_sha256: str,
    primary_cache_record_file_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "role": role,
        "method_id": candidate.method_id,
        "candidate_id": candidate.candidate_id,
        "candidate_parameters": dict(candidate.parameters),
        "seed": seed,
        "epochs": int(config.training["epochs"]),
        "batch_size": int(config.training["batch_size"]),
        "learning_rate": float(config.training["learning_rate"]),
        "weight_decay": float(config.training["weight_decay"]),
        "gradient_clip_norm": float(config.training["gradient_clip_norm"]),
        "mixed_precision": str(config.execution["mixed_precision"]),
        "data_loader_workers": int(config.training["data_loader_workers"]),
        "checkpoint_selection_rule": "fixed_last_epoch",
        "source_train_only": True,
        "source_validation_for_selection_or_calibration_only": True,
        "target_information_used": False,
        "primary_cache_record_sha256": primary_cache_record_sha256,
        "primary_cache_record_file_sha256": primary_cache_record_file_sha256,
        "source_artifact_sha256": EXPECTED_SOURCE_SHA256,
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "dataset_manifest_file_sha256": EXPECTED_DATASET_MANIFEST_FILE_SHA256,
        "preprocessing_config_file_sha256": EXPECTED_PREPROCESSING_FILE_SHA256,
        "ontology_config_sha256": EXPECTED_ONTOLOGY_SHA256,
    }


def _cpu_state_dict(module: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def _optimizer_state_cpu(optimizer: torch.optim.Optimizer) -> Mapping[str, Any]:
    # torch.save safely serializes CUDA optimizer tensors, but moving them to
    # CPU makes reconstruction device-independent and avoids hidden GPU ties.
    payload = optimizer.state_dict()
    for state in cast(dict[Any, dict[str, Any]], payload["state"]).values():
        for name, value in list(state.items()):
            if isinstance(value, Tensor):
                state[name] = value.detach().cpu()
    return payload


def _rng_state(
    loader_generator: torch.Generator,
    mine_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": [value.cpu() for value in torch.cuda.get_rng_state_all()],
        "loader_generator": loader_generator.get_state(),
        "mine_generator": mine_generator.get_state().cpu(),
    }


def _gradient_step(
    loss: Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    parameters: Sequence[nn.Parameter],
    *,
    clip_norm: float,
) -> tuple[bool, float]:
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, clip_norm))
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    return float(scaler.get_scale()) >= scale_before, gradient_norm


def _ccil_epoch(
    model: nn.Module,
    objective: CCILPaperConceptMeanLoss,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    *,
    alpha: float,
    device: torch.device,
    clip_norm: float,
) -> tuple[dict[str, float], int, int]:
    if not isinstance(model, CompactResidualHAR):
        raise TypeError("CCIL adaptation requires CompactResidualHAR")
    model.train()
    objective.train()
    totals = {"classification": 0.0, "ccil": 0.0, "total": 0.0}
    examples = 0
    updates = 0
    skipped = 0
    for signals, labels, _ in loader:
        signals = signals.to(device, non_blocking=False)
        labels = labels.to(device, non_blocking=False)
        objective_state_before_step = {
            name: value.detach().clone() for name, value in objective.state_dict().items()
        }
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            output = cast(HAROutput, model(signals))
            classification = F.cross_entropy(output.logits, labels)
        if output.content is None:
            raise RuntimeError("compact CCIL model did not expose penultimate features")
        with torch.autocast(device_type="cuda", enabled=False):
            ccil = objective(
                output.content.float(),
                labels,
                model.classifier.weight.float(),
                update=True,
            )
            total = classification.float() + alpha * ccil
        successful, _ = _gradient_step(
            total,
            optimizer,
            scaler,
            tuple(model.parameters()),
            clip_norm=clip_norm,
        )
        updates += int(successful)
        skipped += int(not successful)
        if not successful:
            objective.load_state_dict(objective_state_before_step, strict=True)
        count = labels.numel()
        examples += count
        totals["classification"] += float(classification.detach()) * count
        totals["ccil"] += float(ccil.detach()) * count
        totals["total"] += float(total.detach()) * count
    if examples < 1 or not bool(objective.initialized.all()):
        raise RuntimeError("CCIL source epoch did not initialize every activity-class mean")
    return ({name: value / examples for name, value in totals.items()}, updates, skipped)


@contextlib.contextmanager
def _frozen_parameters(*modules: nn.Module) -> Iterator[None]:
    parameters = [parameter for module in modules for parameter in module.parameters()]
    previous = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, state in zip(parameters, previous, strict=True):
            parameter.requires_grad_(state)


def _bpd_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    main_optimizer: torch.optim.Optimizer,
    auxiliary_optimizer: torch.optim.Optimizer,
    main_scaler: Any,
    auxiliary_scaler: Any,
    mine_generator: torch.Generator,
    *,
    parameters: Mapping[str, float],
    device: torch.device,
    clip_norm: float,
) -> tuple[dict[str, float], int, int]:
    if not isinstance(model, BPDBoundarySafeCompactAdapter):
        raise TypeError("BPD adaptation requires BPDBoundarySafeCompactAdapter")
    model.train()
    totals = {
        "activity_classification": 0.0,
        "redundant_classifier": 0.0,
        "mine_critic_lower_bound": 0.0,
        "main_mine_lower_bound": 0.0,
        "class_confusion": 0.0,
        "reconstruction": 0.0,
        "main_total": 0.0,
    }
    examples = 0
    updates = 0
    skipped = 0
    main_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("mine.") and not name.startswith("redundant_activity_classifier.")
    ]
    auxiliary_parameters = [
        *model.mine.parameters(),
        *model.redundant_activity_classifier.parameters(),
    ]
    for signals, labels, _ in loader:
        signals = signals.to(device, non_blocking=False)
        labels = labels.to(device, non_blocking=False)
        batch_size = labels.numel()
        permutation = torch.randperm(batch_size, device=device, generator=mine_generator)
        main_optimizer.zero_grad(set_to_none=True)
        with (
            _frozen_parameters(model.mine, model.redundant_activity_classifier),
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True),
        ):
            output = model.decompose(signals)
            activity_classification = F.cross_entropy(output.activity_logits, labels)
            confusion = redundant_class_confusion_loss(output.redundant_activity_logits)
            reconstruction = F.mse_loss(
                output.reconstructed_backbone_features,
                output.backbone_features.detach(),
            )
            main_bound = mine_dv_lower_bound(
                model.mine,
                output.activity_features,
                output.redundant_features,
                permutation=permutation,
            )
            main_total = (
                activity_classification.float()
                + parameters["confusion_weight"] * confusion.float()
                + parameters["reconstruction_weight"] * reconstruction.float()
                + parameters["mutual_information_weight"] * main_bound.float()
            )
        successful, _ = _gradient_step(
            main_total,
            main_optimizer,
            main_scaler,
            main_parameters,
            clip_norm=clip_norm,
        )
        updates += int(successful)
        skipped += int(not successful)

        # The nuisance classifier and MINE critic see detached features from
        # the same single generator pass. This avoids updating the compact
        # backbone's BatchNorm statistics twice per source batch.
        auxiliary_optimizer.zero_grad(set_to_none=True)
        activity_detached = output.activity_features.detach()
        redundant_detached = output.redundant_features.detach()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            redundant_logits = model.redundant_activity_classifier(redundant_detached)
            redundant_classification = F.cross_entropy(redundant_logits, labels)
            critic_bound = mine_dv_lower_bound(
                model.mine,
                activity_detached,
                redundant_detached,
                permutation=permutation,
            )
            auxiliary_loss = redundant_classification.float() - critic_bound
        successful, _ = _gradient_step(
            auxiliary_loss,
            auxiliary_optimizer,
            auxiliary_scaler,
            list(auxiliary_parameters),
            clip_norm=clip_norm,
        )
        updates += int(successful)
        skipped += int(not successful)
        examples += batch_size
        values = {
            "activity_classification": activity_classification,
            "redundant_classifier": redundant_classification,
            "mine_critic_lower_bound": critic_bound,
            "main_mine_lower_bound": main_bound,
            "class_confusion": confusion,
            "reconstruction": reconstruction,
            "main_total": main_total,
        }
        for name, value in values.items():
            totals[name] += float(value.detach()) * batch_size
    if examples < 1:
        raise RuntimeError("BPD adaptation source epoch was empty")
    return ({name: value / examples for name, value in totals.items()}, updates, skipped)


def reconstruct_paper_adaptation_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str,
    expected_method_id: str,
    expected_seed: int,
    expected_configuration_sha256: str,
    expected_code_commit: str,
    device: torch.device,
) -> tuple[nn.Module, Mapping[str, Any]]:
    """Fail closed when rebuilding one locally generated adapter checkpoint."""

    path = Path(checkpoint_path).resolve(strict=True)
    if sha256_file(path) != expected_sha256:
        raise PaperAdaptationRunError("paper-adaptation checkpoint file hash changed")
    payload_value = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload_value, Mapping):
        raise PaperAdaptationRunError("paper-adaptation checkpoint root is not an object")
    payload = dict(payload_value)
    required = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": "postconfirmatory_paper_adaptation_checkpoint",
        "status": "complete_create_only",
        "evidence_status": EVIDENCE_STATUS,
        "method_id": expected_method_id,
        "seed": expected_seed,
        "configuration_sha256": expected_configuration_sha256,
        "code_commit": expected_code_commit,
        "label_schema": list(CLASS_NAMES),
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "source_artifact_sha256": EXPECTED_SOURCE_SHA256,
        "target_information_used": False,
        "checkpoint_selection_rule": "fixed_last_epoch",
    }
    mismatches = [key for key, expected in required.items() if payload.get(key) != expected]
    configuration_value = payload.get("configuration")
    if (
        not isinstance(configuration_value, Mapping)
        or canonical_json_sha256(configuration_value) != expected_configuration_sha256
    ):
        mismatches.append("configuration")
    if mismatches:
        raise PaperAdaptationRunError(
            f"paper-adaptation checkpoint contract changed: {sorted(set(mismatches))}"
        )
    if not isinstance(configuration_value, Mapping):
        raise PaperAdaptationRunError("paper-adaptation checkpoint configuration is invalid")
    configuration = configuration_value
    if payload.get("primary_cache_record_sha256") != configuration.get(
        "primary_cache_record_sha256"
    ) or payload.get("primary_cache_record_file_sha256") != configuration.get(
        "primary_cache_record_file_sha256"
    ):
        raise PaperAdaptationRunError("paper-adaptation checkpoint cache lineage changed")
    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping):
        raise PaperAdaptationRunError("paper-adaptation checkpoint lacks model state")
    model = _build_model(expected_method_id).to(device)
    model.load_state_dict(cast(Mapping[str, Tensor], model_state), strict=True)
    if expected_method_id == METHOD_IDS[0]:
        ccil_state = payload.get("ccil_state")
        candidate_parameters = configuration.get("candidate_parameters")
        if not isinstance(ccil_state, Mapping) or not isinstance(candidate_parameters, Mapping):
            raise PaperAdaptationRunError("CCIL checkpoint lacks concept-mean state")
        objective = CCILPaperConceptMeanLoss(
            len(CLASS_NAMES),
            96,
            ema_update_weight=float(candidate_parameters["ema_update_weight"]),
            device=device,
        )
        objective.load_state_dict(cast(Mapping[str, Tensor], ccil_state), strict=True)
        if not bool(objective.initialized.all()) or bool((objective.update_counts < 1).any()):
            raise PaperAdaptationRunError("CCIL checkpoint concept means are incomplete")
    model.eval()
    return model, payload


def _train_candidate(
    source: SourceArrays,
    *,
    config: PaperAdaptationConfig,
    candidate: AdaptationCandidate,
    seed: int,
    role: str,
    output_directory: Path,
    repository_root: Path,
    cache: PreparedPrimaryCacheEvidence,
    code_commit: str,
    device: torch.device,
) -> TrainedAdapter:
    """Train one fixed-epoch source-only adapter and reconstruct its checkpoint."""

    configure_determinism(seed)
    configuration = _training_configuration(
        config,
        candidate,
        seed=seed,
        role=role,
        primary_cache_record_sha256=cache.record_sha256,
        primary_cache_record_file_sha256=cache.record_file_sha256,
    )
    configuration_sha = canonical_json_sha256(configuration)
    model = _build_model(candidate.method_id).to(device)
    dataset = WindowTensorDataset(
        source.train_windows,
        source.train_labels,
        np.zeros(source.train_labels.size, dtype=np.int64),
    )
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(seed + 10_003)
    mine_generator = torch.Generator(device=device)
    mine_generator.manual_seed(seed + 20_003)
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]] = DataLoader(
        dataset,
        batch_size=int(config.training["batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
        drop_last=False,
    )
    learning_rate = float(config.training["learning_rate"])
    weight_decay = float(config.training["weight_decay"])
    clip_norm = float(config.training["gradient_clip_norm"])
    epochs = int(config.training["epochs"])
    amp_scalers: dict[str, Any] = {}
    optimizers: dict[str, torch.optim.Optimizer] = {}
    schedulers: dict[str, Any] = {}
    ccil_objective: CCILPaperConceptMeanLoss | None = None
    if candidate.method_id == METHOD_IDS[0]:
        if not isinstance(model, CompactResidualHAR):
            raise AssertionError("CCIL model construction changed")
        ccil_objective = CCILPaperConceptMeanLoss(
            len(CLASS_NAMES),
            96,
            ema_update_weight=float(candidate.parameters["ema_update_weight"]),
            device=device,
        )
        ccil_optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        optimizers["main"] = ccil_optimizer
        schedulers["main"] = torch.optim.lr_scheduler.CosineAnnealingLR(
            ccil_optimizer, T_max=epochs
        )
        amp_scalers["main"] = torch.amp.GradScaler(  # type: ignore[attr-defined]
            "cuda", enabled=True
        )
    else:
        if not isinstance(model, BPDBoundarySafeCompactAdapter):
            raise AssertionError("BPD model construction changed")
        main_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith("mine.")
            and not name.startswith("redundant_activity_classifier.")
        ]
        auxiliary_parameters = [
            *model.mine.parameters(),
            *model.redundant_activity_classifier.parameters(),
        ]
        optimizers["main"] = torch.optim.AdamW(
            main_parameters, lr=learning_rate, weight_decay=weight_decay
        )
        optimizers["auxiliary"] = torch.optim.AdamW(
            auxiliary_parameters, lr=learning_rate, weight_decay=weight_decay
        )
        for name, bpd_optimizer in optimizers.items():
            schedulers[name] = torch.optim.lr_scheduler.CosineAnnealingLR(
                bpd_optimizer, T_max=epochs
            )
            amp_scalers[name] = torch.amp.GradScaler(  # type: ignore[attr-defined]
                "cuda", enabled=True
            )

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    history: list[Mapping[str, Any]] = []
    for epoch in range(1, epochs + 1):
        if candidate.method_id == METHOD_IDS[0]:
            if ccil_objective is None:
                raise AssertionError("CCIL objective was not constructed")
            losses, updates, skipped = _ccil_epoch(
                model,
                ccil_objective,
                loader,
                optimizers["main"],
                amp_scalers["main"],
                alpha=float(candidate.parameters["alpha"]),
                device=device,
                clip_norm=clip_norm,
            )
        else:
            losses, updates, skipped = _bpd_epoch(
                model,
                loader,
                optimizers["main"],
                optimizers["auxiliary"],
                amp_scalers["main"],
                amp_scalers["auxiliary"],
                mine_generator,
                parameters=candidate.parameters,
                device=device,
                clip_norm=clip_norm,
            )
        if updates < 1:
            raise PaperAdaptationRunError("AMP skipped every optimizer update in an epoch")
        for scheduler in schedulers.values():
            scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "training_losses": dict(losses),
                "optimizer_update_count": updates,
                "amp_skipped_step_count": skipped,
                "learning_rates": {
                    name: float(optimizer.param_groups[0]["lr"])
                    for name, optimizer in optimizers.items()
                },
                "source_validation_accessed": False,
                "target_accessed": False,
            }
        )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_vram = int(torch.cuda.max_memory_allocated(device))
    output_directory.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_directory / "selected.pt"
    dependency_lock_path = _resolve_existing(
        "uv.lock", root=repository_root, name="dependency lockfile"
    )
    checkpoint_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": "postconfirmatory_paper_adaptation_checkpoint",
        "status": "complete_create_only",
        "evidence_status": EVIDENCE_STATUS,
        "track_role": TRACK_ROLE,
        "role": role,
        "method_id": candidate.method_id,
        "candidate_id": candidate.candidate_id,
        "seed": seed,
        "epoch": epochs,
        "checkpoint_selection_rule": "fixed_last_epoch",
        "configuration": configuration,
        "configuration_sha256": configuration_sha,
        "model_state": _cpu_state_dict(model),
        "ccil_state": _cpu_state_dict(ccil_objective) if ccil_objective is not None else None,
        "optimizer_states": {
            name: _optimizer_state_cpu(optimizer) for name, optimizer in optimizers.items()
        },
        "scheduler_states": {
            name: scheduler.state_dict() for name, scheduler in schedulers.items()
        },
        "scaler_states": {name: scaler.state_dict() for name, scaler in amp_scalers.items()},
        "history": list(history),
        "normalization": source.normalizer.to_dict(),
        "normalization_sha256": canonical_json_sha256(source.normalizer.to_dict()),
        "label_schema": list(CLASS_NAMES),
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "source_artifact_sha256": EXPECTED_SOURCE_SHA256,
        "primary_cache_record_sha256": cache.record_sha256,
        "primary_cache_record_file_sha256": cache.record_file_sha256,
        "experiment_config_file_sha256": config.file_sha256,
        "experiment_config_canonical_sha256": config.canonical_sha256,
        "code_commit": code_commit,
        "rng_states": _rng_state(loader_generator, mine_generator),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cuda_runtime": torch.version.cuda,
            "cuda_device_name": torch.cuda.get_device_name(device),
            "cuda_compute_capability": list(torch.cuda.get_device_capability(device)),
            "cudnn_enabled": bool(torch.backends.cudnn.enabled),
            "cudnn_version": torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
            "mixed_precision": "float16",
            "dependency_lock": {
                "path": dependency_lock_path.relative_to(repository_root).as_posix(),
                "sha256": sha256_file(dependency_lock_path),
            },
        },
        "source_train_window_count": source.train_labels.size,
        "source_train_participants": list(SOURCE_TRAIN_PARTICIPANTS),
        "source_validation_used_for_checkpoint_selection": False,
        "target_information_used": False,
        "disability_or_assistive_metadata_input": False,
    }
    with checkpoint_path.open("xb") as stream:
        torch.save(checkpoint_payload, stream)
    checkpoint_sha = sha256_file(checkpoint_path)
    rebuilt, _ = reconstruct_paper_adaptation_checkpoint(
        checkpoint_path,
        expected_sha256=checkpoint_sha,
        expected_method_id=candidate.method_id,
        expected_seed=seed,
        expected_configuration_sha256=configuration_sha,
        expected_code_commit=code_commit,
        device=device,
    )
    logits, probabilities, report = predict_model(
        rebuilt,
        source.validation_windows,
        source.validation_labels,
        list(source.validation_participants),
        class_names=CLASS_NAMES,
        batch_size=int(config.training["batch_size"]),
        device=device,
        mixed_precision="float16",
    )
    parameter_count = trainable_parameter_count(model)
    del rebuilt, model
    return TrainedAdapter(
        method_id=candidate.method_id,
        candidate=candidate,
        seed=seed,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        configuration=configuration,
        configuration_sha256=configuration_sha,
        validation_logits=logits,
        validation_probabilities=probabilities,
        validation_report=report,
        history=tuple(history),
        parameter_count=parameter_count,
        peak_vram_bytes=peak_vram,
        elapsed_seconds=elapsed,
    )


def select_source_candidate(
    candidates: Sequence[AdaptationCandidate],
    reports: Mapping[str, Mapping[str, Any]],
) -> AdaptationCandidate:
    """Apply the frozen participant-level source-validation ordering."""

    expected = {candidate.candidate_id for candidate in candidates}
    if not candidates or set(reports) != expected:
        raise PaperAdaptationRunError("source candidate report matrix is incomplete")
    rows: list[tuple[float, float, str, AdaptationCandidate]] = []
    for candidate in candidates:
        report = reports[candidate.candidate_id]
        primary = report.get("primary")
        if not isinstance(primary, Mapping):
            raise PaperAdaptationRunError("source candidate report lacks primary endpoints")
        mean = primary.get("mean_participant_macro_f1")
        worst = primary.get("worst_participant_macro_f1")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
            for value in (mean, worst)
        ):
            raise PaperAdaptationRunError("source candidate endpoint is not finite")
        rows.append(
            (
                -float(cast(float, mean)),
                -float(cast(float, worst)),
                candidate.candidate_id,
                candidate,
            )
        )
    return min(rows)[3]


def _write_prediction_new(
    path: Path,
    *,
    method_id: str,
    seed: int,
    cohort: str,
    window_ids: Sequence[str],
    participant_ids: Sequence[str],
    labels: NDArray[np.int64],
    logits: NDArray[np.float64],
    uncalibrated: NDArray[np.float64],
    calibrated: NDArray[np.float64],
) -> str:
    count = labels.size
    if (
        logits.shape != (count, len(CLASS_NAMES))
        or uncalibrated.shape != logits.shape
        or calibrated.shape != logits.shape
        or len(window_ids) != count
        or len(participant_ids) != count
        or not np.isfinite(logits).all()
        or not np.isfinite(uncalibrated).all()
        or not np.isfinite(calibrated).all()
    ):
        raise PaperAdaptationRunError("prediction arrays are not finite and aligned")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            evidence_status=np.asarray([EVIDENCE_STATUS]),
            method_id=np.asarray([method_id]),
            seed=np.asarray([seed], dtype=np.int64),
            cohort=np.asarray([cohort]),
            window_ids=np.asarray(window_ids),
            participant_ids=np.asarray(participant_ids),
            true_labels=np.asarray(labels, dtype=np.int64),
            logits=np.asarray(logits, dtype=np.float64),
            uncalibrated_probabilities=np.asarray(uncalibrated, dtype=np.float64),
            calibrated_probabilities=np.asarray(calibrated, dtype=np.float64),
            predicted_labels=np.asarray(calibrated.argmax(axis=1), dtype=np.int64),
        )
    return sha256_file(path)


def _entry(
    *,
    record_path: Path,
    record: Mapping[str, Any],
    prediction_path: Path,
    prediction_sha256: str,
    root: Path,
) -> dict[str, Any]:
    return {
        "method_id": record["method_id"],
        "seed": record["seed"],
        "cohort": record["cohort"],
        "record_path": record_path.relative_to(root).as_posix(),
        "record_file_sha256": sha256_file(record_path),
        "record_sha256": record["record_sha256"],
        "prediction_path": prediction_path.relative_to(root).as_posix(),
        "prediction_sha256": prediction_sha256,
    }


def _evaluation_record(
    *,
    trained: TrainedAdapter,
    cohort: str,
    report: Mapping[str, Any],
    prediction_path: Path,
    prediction_sha256: str,
    calibrator_path: Path,
    calibrator_file_sha256: str,
    calibrator_record_sha256: str,
    source_stage_lock_sha256: str | None,
    config: PaperAdaptationConfig,
    root: Path,
    code_commit: str,
    created_at_utc: str,
    cache_record_sha256: str,
    cache_record_file_sha256: str,
    opening_record_sha256: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": "postconfirmatory_paper_adaptation_evaluation",
        "status": "complete_create_only",
        "created_at_utc": created_at_utc,
        "evidence_status": EVIDENCE_STATUS,
        "track_role": TRACK_ROLE,
        "primary_claim_eligible": False,
        "method_id": trained.method_id,
        "claim_label": next(
            str(method["claim_label"])
            for method in config.methods
            if method["model_id"] == trained.method_id
        ),
        "candidate_id": trained.candidate.candidate_id,
        "candidate_parameters": dict(trained.candidate.parameters),
        "seed": trained.seed,
        "cohort": cohort,
        "participant_level_report": dict(report),
        "checkpoint": {
            "path": trained.checkpoint_path.relative_to(root).as_posix(),
            "sha256": trained.checkpoint_sha256,
            "configuration_sha256": trained.configuration_sha256,
        },
        "calibrator": {
            "path": calibrator_path.relative_to(root).as_posix(),
            "file_sha256": calibrator_file_sha256,
            "record_sha256": calibrator_record_sha256,
            "fit_partition": "source_validation",
        },
        "prediction_array": {
            "path": prediction_path.relative_to(root).as_posix(),
            "sha256": prediction_sha256,
        },
        "source_stage_lock_record_sha256": source_stage_lock_sha256,
        "primary_cache_record_sha256": cache_record_sha256,
        "primary_cache_record_file_sha256": cache_record_file_sha256,
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "experiment_config_file_sha256": config.file_sha256,
        "experiment_config_canonical_sha256": config.canonical_sha256,
        "code_commit": code_commit,
        "opening_receipt_record_sha256": opening_record_sha256,
        "target_tuning_or_refit": False,
        "target_information_used_for_source_selection": False,
        "new_opening_or_unlock_invoked": False,
        "trial_safe": False,
        "statistical_unit": "participant",
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    return payload


def _resolve_existing(value: str | Path, *, root: Path, name: str) -> Path:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    if candidate.is_symlink():
        raise PaperAdaptationRunError(f"{name} may not be a symlink")
    path = candidate.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PaperAdaptationRunError(f"{name} escapes repository_root") from exc
    if path.is_symlink() or not path.is_file():
        raise PaperAdaptationRunError(f"{name} must be a regular non-symlink file")
    return path


def _resolve_new_directory(value: str | Path, *, root: Path) -> Path:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    path = candidate.resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PaperAdaptationRunError("output directory escapes repository_root") from exc
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite paper-adaptation output: {path}")
    parent = path.parent.resolve(strict=True)
    if parent.is_symlink():
        raise PaperAdaptationRunError("output parent may not be a symlink")
    return path


def _git_state(root: Path) -> tuple[str, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, status


def _validate_static_provenance(config: PaperAdaptationConfig, *, root: Path) -> None:
    checks = (
        (
            config.protocol["split_manifest_path"],
            config.protocol["split_manifest_file_sha256"],
            "split manifest",
        ),
        (
            config.protocol["dataset_manifest_path"],
            config.protocol["dataset_manifest_file_sha256"],
            "dataset manifest",
        ),
        (
            config.protocol["preprocessing_config_path"],
            config.protocol["preprocessing_config_file_sha256"],
            "preprocessing config",
        ),
        (
            config.opening_1["opening_receipt_path"],
            config.opening_1["opening_receipt_file_sha256"],
            "opening-1 receipt",
        ),
        (
            config.opening_1["locked_target_index_path"],
            config.opening_1["locked_target_index_file_sha256"],
            "locked target index",
        ),
        (
            config.comparator["final_freeze_path"],
            config.comparator["final_freeze_file_sha256"],
            "final source artifact freeze",
        ),
    )
    for value, expected, name in checks:
        path = _resolve_existing(cast(str, value), root=root, name=name)
        if sha256_file(path) != expected:
            raise PaperAdaptationRunError(f"{name} file hash changed")
    split = load_json_strict(
        _resolve_existing(
            cast(str, config.protocol["split_manifest_path"]), root=root, name="split manifest"
        )
    )
    if not isinstance(split, Mapping):
        raise PaperAdaptationRunError("split manifest root changed")
    body = dict(split)
    claimed = body.pop("split_manifest_sha256", None)
    if claimed != EXPECTED_SPLIT_SHA256 or canonical_json_sha256(body) != claimed:
        raise PaperAdaptationRunError("split manifest self-hash changed")
    ontology = split.get("ontology")
    preprocessing = split.get("preprocessing")
    if (
        not isinstance(ontology, Mapping)
        or ontology.get("config_sha256") != config.protocol["ontology_config_sha256"]
        or not isinstance(preprocessing, Mapping)
        or preprocessing.get("config_sha256") != EXPECTED_PREPROCESSING_CANONICAL_SHA256
    ):
        raise PaperAdaptationRunError("split ontology/preprocessing lineage changed")
    freeze = load_json_strict(
        _resolve_existing(
            cast(str, config.comparator["final_freeze_path"]),
            root=root,
            name="final source artifact freeze",
        )
    )
    if not isinstance(freeze, Mapping):
        raise PaperAdaptationRunError("final source artifact freeze root changed")
    freeze_required = {
        "record_kind": "final_source_artifact_freeze",
        "status": "frozen_before_target_unlock",
        "inventory_sha256": config.comparator["final_freeze_inventory_sha256"],
        "frozen_artifact_set_sha256": config.comparator["frozen_artifact_set_sha256"],
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "target_seal_id": config.opening_1["target_seal_id"],
        "required_seed_order": list(SEED_ORDER),
    }
    if any(freeze.get(key) != expected for key, expected in freeze_required.items()):
        raise PaperAdaptationRunError("final source artifact freeze contract changed")
    target_state = freeze.get("target_state")
    if not isinstance(target_state, Mapping) or (
        target_state.get("target_predictions_or_performance_accessed") is not False
        or target_state.get("target_subject_or_window_records_loaded") is not False
    ):
        raise PaperAdaptationRunError("final source artifact freeze was not target-blind")
    models = freeze.get("models")
    if not isinstance(models, list):
        raise PaperAdaptationRunError("final source artifact freeze model inventory changed")
    frozen_identities = {
        (entry.get("model_id"), entry.get("seed")) for entry in models if isinstance(entry, Mapping)
    }
    required_comparators = {(model_id, seed) for model_id in COMPARATOR_IDS for seed in SEED_ORDER}
    if not required_comparators.issubset(frozen_identities):
        raise PaperAdaptationRunError("locked comparator source artifacts are absent from freeze")


def _load_source_arrays(
    *,
    config: PaperAdaptationConfig,
    root: Path,
    cache_record_path: Path,
    expected_cache_record_file_sha256: str,
) -> tuple[SourceArrays, PreparedPrimaryCacheEvidence, PreparedPrimaryCacheEvidence]:
    train = load_prepared_primary_cache(
        record_path=cache_record_path,
        expected_record_file_sha256=expected_cache_record_file_sha256,
        partition="source_train",
        opening_receipt_path=cast(str, config.opening_1["opening_receipt_path"]),
        locked_target_index_path=cast(str, config.opening_1["locked_target_index_path"]),
        artifact_root=root,
        expected_split_manifest_sha256=EXPECTED_SPLIT_SHA256,
        expected_source_artifact_sha256=EXPECTED_SOURCE_SHA256,
    )
    validation = load_prepared_primary_cache(
        record_path=cache_record_path,
        expected_record_file_sha256=expected_cache_record_file_sha256,
        partition="source_validation",
        opening_receipt_path=cast(str, config.opening_1["opening_receipt_path"]),
        locked_target_index_path=cast(str, config.opening_1["locked_target_index_path"]),
        artifact_root=root,
        expected_split_manifest_sha256=EXPECTED_SPLIT_SHA256,
        expected_source_artifact_sha256=EXPECTED_SOURCE_SHA256,
    )
    if train.record_sha256 != validation.record_sha256 or (
        train.record_file_sha256 != validation.record_file_sha256
    ):
        raise PaperAdaptationRunError("source partitions do not come from one pinned cache record")
    train_batch = train.cache.batch
    validation_batch = validation.cache.batch
    if (
        train_batch.signals.shape != (EXPECTED_WINDOWS["source_train"], 128, 6)
        or validation_batch.signals.shape != (EXPECTED_WINDOWS["source_validation"], 128, 6)
        or set(train_batch.participant_ids) != set(SOURCE_TRAIN_PARTICIPANTS)
        or set(validation_batch.participant_ids) != set(SOURCE_VALIDATION_PARTICIPANTS)
        or set(train_batch.window_ids) & set(validation_batch.window_ids)
    ):
        raise PaperAdaptationRunError("source cache partition/count/identity contract changed")
    normalizer = ChannelStandardizer.fit(
        train_batch.signals,
        list(train_batch.participant_ids),
        declared_training_participants=set(SOURCE_TRAIN_PARTICIPANTS),
        split_manifest_sha256=EXPECTED_SPLIT_SHA256,
        channel_names=INCLUSIVEHAR_PRIMARY_CHANNELS,
    )
    source = SourceArrays(
        normalizer=normalizer,
        train_windows=normalizer.transform(train_batch.signals),
        train_labels=np.asarray(train_batch.labels, dtype=np.int64),
        train_participants=train_batch.participant_ids,
        train_window_ids=train_batch.window_ids,
        validation_windows=normalizer.transform(validation_batch.signals),
        validation_labels=np.asarray(validation_batch.labels, dtype=np.int64),
        validation_participants=validation_batch.participant_ids,
        validation_window_ids=validation_batch.window_ids,
    )
    return source, train, validation


def _source_candidate_record(
    trained: TrainedAdapter,
    *,
    prediction_path: Path,
    prediction_sha256: str,
    config: PaperAdaptationConfig,
    root: Path,
    cache: PreparedPrimaryCacheEvidence,
    code_commit: str,
    created_at_utc: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": "postconfirmatory_paper_adaptation_source_candidate",
        "status": "complete_create_only",
        "created_at_utc": created_at_utc,
        "evidence_status": EVIDENCE_STATUS,
        "track_role": TRACK_ROLE,
        "method_id": trained.method_id,
        "candidate_id": trained.candidate.candidate_id,
        "candidate_parameters": dict(trained.candidate.parameters),
        "selection_seed": trained.seed,
        "cohort": "source_validation",
        "participant_level_report": dict(trained.validation_report),
        "checkpoint": {
            "path": trained.checkpoint_path.relative_to(root).as_posix(),
            "sha256": trained.checkpoint_sha256,
            "configuration_sha256": trained.configuration_sha256,
        },
        "prediction_array": {
            "path": prediction_path.relative_to(root).as_posix(),
            "sha256": prediction_sha256,
        },
        "parameter_count": trained.parameter_count,
        "peak_vram_bytes": trained.peak_vram_bytes,
        "elapsed_seconds": trained.elapsed_seconds,
        "primary_cache_record_sha256": cache.record_sha256,
        "primary_cache_record_file_sha256": cache.record_file_sha256,
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "experiment_config_file_sha256": config.file_sha256,
        "experiment_config_canonical_sha256": config.canonical_sha256,
        "code_commit": code_commit,
        "checkpoint_selection_rule": "fixed_last_epoch",
        "source_validation_used_for_candidate_selection": True,
        "source_validation_used_for_early_stopping": False,
        "target_signals_labels_predictions_or_metrics_used": False,
        "new_opening_or_unlock_invoked": False,
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    return payload


def _write_source_selection_lock(
    *,
    selected: Mapping[str, AdaptationCandidate],
    entries: Sequence[Mapping[str, Any]],
    config: PaperAdaptationConfig,
    output: Path,
    root: Path,
    cache: PreparedPrimaryCacheEvidence,
    code_commit: str,
    created_at_utc: str,
) -> dict[str, Any]:
    path = output / str(config.outputs["selection_lock_filename"])
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": "postconfirmatory_ccil_bpd_source_selection_lock",
        "status": "complete_create_only_target_signals_not_accessed",
        "created_at_utc": created_at_utc,
        "evidence_status": EVIDENCE_STATUS,
        "experiment_config_file_sha256": config.file_sha256,
        "experiment_config_canonical_sha256": config.canonical_sha256,
        "code_commit": code_commit,
        "selection_seed": SEED_ORDER[0],
        "required_method_order": list(METHOD_IDS),
        "equal_candidate_count_per_method": 4,
        "candidate_result_count": len(entries),
        "candidate_results": [dict(entry) for entry in entries],
        "selected_candidates": {
            method_id: {
                "candidate_id": selected[method_id].candidate_id,
                "parameters": dict(selected[method_id].parameters),
            }
            for method_id in METHOD_IDS
        },
        "selection_metric": "mean_participant_macro_f1",
        "tie_breakers": ["worst_participant_macro_f1", "candidate_id_lexicographic"],
        "fit_partition": "source_train",
        "scoring_partition": "source_validation",
        "primary_cache_record_sha256": cache.record_sha256,
        "primary_cache_record_file_sha256": cache.record_file_sha256,
        "consumed_opening_metadata_validated_by_cache_loader": True,
        "target_signal_arrays_loaded": False,
        "target_labels_predictions_or_metrics_used": False,
        "new_opening_or_unlock_invoked": False,
    }
    if len(entries) != len(METHOD_IDS) * 4:
        raise PaperAdaptationRunError("selection lock requires the exact eight-candidate matrix")
    payload["record_sha256"] = canonical_json_sha256(payload)
    atomic_write_json_new(payload, path, allowed_root=root)
    return {
        "path": path.relative_to(root).as_posix(),
        "file_sha256": sha256_file(path),
        "record_sha256": payload["record_sha256"],
        "selected_candidates": payload["selected_candidates"],
    }


def _load_frozen_source_comparator(
    *,
    model_id: str,
    seed: int,
    config: PaperAdaptationConfig,
    source: SourceArrays,
    output: Path,
    root: Path,
    created_at_utc: str,
) -> dict[str, Any]:
    if model_id not in COMPARATOR_IDS:
        raise PaperAdaptationRunError(f"unsupported frozen comparator: {model_id}")
    record_templates = cast(Mapping[str, str], config.comparator["source_record_templates"])
    calibrator_templates = cast(Mapping[str, str], config.comparator["source_calibrator_templates"])
    source_model_names = cast(Mapping[str, str], config.comparator["source_model_names"])
    record_path = _resolve_existing(
        record_templates[model_id].format(seed=seed),
        root=root,
        name=f"frozen {model_id} source record",
    )
    record = load_json_strict(record_path)
    if not isinstance(record, Mapping):
        raise PaperAdaptationRunError(f"frozen {model_id} source record root changed")
    body = dict(record)
    claimed = body.pop("record_sha256", None)
    required = {
        "status": "source_development_complete_target_sealed",
        "evidence_status": "source_development_not_confirmatory",
        "model_name": source_model_names[model_id],
        "seed": seed,
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "target_performance_or_prediction_accessed": False,
        "target_subject_or_window_records_loaded": False,
        "class_names": list(CLASS_NAMES),
        "source_split_id": "final_source_split",
        "train_participants": list(SOURCE_TRAIN_PARTICIPANTS),
        "validation_participants": list(SOURCE_VALIDATION_PARTICIPANTS),
        "train_window_count": EXPECTED_WINDOWS["source_train"],
        "validation_window_count": EXPECTED_WINDOWS["source_validation"],
        "requested_device": "cuda",
        "model_device": "cuda",
    }
    if claimed != canonical_json_sha256(body) or any(
        record.get(key) != expected for key, expected in required.items()
    ):
        raise PaperAdaptationRunError(f"frozen {model_id} source record contract changed")
    frozen_normalization = record.get("normalization")
    if not isinstance(frozen_normalization, Mapping) or canonical_json_sha256(
        frozen_normalization
    ) != canonical_json_sha256(source.normalizer.to_dict()):
        raise PaperAdaptationRunError(f"frozen {model_id} normalization differs from cache fit")
    prediction = record.get("prediction_artifact")
    checkpoint = record.get("checkpoint")
    configuration_sha = record.get("configuration_sha256")
    if (
        not isinstance(prediction, Mapping)
        or not isinstance(checkpoint, Mapping)
        or not isinstance(configuration_sha, str)
    ):
        raise PaperAdaptationRunError(f"frozen {model_id} source lineage is incomplete")
    checkpoint_path = _resolve_existing(
        cast(str, checkpoint.get("path")),
        root=root,
        name=f"frozen {model_id} checkpoint",
    )
    if sha256_file(checkpoint_path) != checkpoint.get("sha256"):
        raise PaperAdaptationRunError(f"frozen {model_id} checkpoint hash changed")
    freeze_path = _resolve_existing(
        cast(str, config.comparator["final_freeze_path"]),
        root=root,
        name="final source artifact freeze",
    )
    freeze = load_json_strict(freeze_path)
    freeze_models = freeze.get("models") if isinstance(freeze, Mapping) else None
    if not isinstance(freeze_models, list):
        raise PaperAdaptationRunError("final source artifact freeze model inventory changed")
    matching_freeze_entries = [
        entry
        for entry in freeze_models
        if isinstance(entry, Mapping)
        and entry.get("model_id") == model_id
        and entry.get("seed") == seed
    ]
    if len(matching_freeze_entries) != 1:
        raise PaperAdaptationRunError(f"freeze lacks one exact {model_id} seed {seed} entry")
    freeze_entry = matching_freeze_entries[0]
    freeze_checkpoint = freeze_entry.get("checkpoint")
    if (
        not isinstance(freeze_checkpoint, Mapping)
        or freeze_checkpoint.get("sha256") != checkpoint.get("sha256")
        or freeze_entry.get("training_configuration_sha256") != configuration_sha
    ):
        raise PaperAdaptationRunError(f"frozen {model_id} source record differs from freeze")
    prediction_path = _resolve_existing(
        cast(str, prediction.get("path")), root=root, name="frozen source prediction"
    )
    if sha256_file(prediction_path) != prediction.get("sha256"):
        raise PaperAdaptationRunError("frozen source prediction hash changed")
    with np.load(prediction_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {
            "logits",
            "probabilities",
            "labels",
            "participant_ids",
            "window_ids",
        }:
            raise PaperAdaptationRunError("frozen source prediction schema changed")
        logits = np.asarray(arrays["logits"], dtype=np.float64)
        labels = np.asarray(arrays["labels"], dtype=np.int64)
        participants = tuple(str(value) for value in arrays["participant_ids"].tolist())
        windows = tuple(str(value) for value in arrays["window_ids"].tolist())
    if (
        logits.shape != (EXPECTED_WINDOWS["source_validation"], len(CLASS_NAMES))
        or not np.array_equal(labels, source.validation_labels)
        or participants != source.validation_participants
        or windows != source.validation_window_ids
    ):
        raise PaperAdaptationRunError(f"frozen {model_id} source alignment changed")
    calibrator_path = _resolve_existing(
        calibrator_templates[model_id].format(seed=seed),
        root=root,
        name=f"frozen {model_id} calibrator",
    )
    calibrator_record, calibrator = load_source_temperature_calibrator_file(
        calibrator_path,
        expected_checkpoint_sha256=str(checkpoint["sha256"]),
        expected_training_configuration_sha256=configuration_sha,
        expected_split_manifest_sha256=EXPECTED_SPLIT_SHA256,
    )
    freeze_calibrator = freeze_entry.get("calibrator")
    if (
        not isinstance(freeze_calibrator, Mapping)
        or freeze_calibrator.get("sha256") != sha256_file(calibrator_path)
        or freeze_entry.get("calibrator_record_sha256") != calibrator_record.get("record_sha256")
    ):
        raise PaperAdaptationRunError(f"frozen {model_id} calibrator differs from freeze")
    calibrated = calibrator.probabilities(logits)
    report = classification_report(labels, calibrated, participants, class_names=CLASS_NAMES)
    derived_path = output / "comparator" / f"{model_id}--seed-{seed}--source.json"
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": "postconfirmatory_frozen_source_comparator_reference",
        "status": "complete_create_only",
        "created_at_utc": created_at_utc,
        "evidence_status": "frozen_source_reference_postconfirmatory_derivation",
        "model_id": model_id,
        "seed": seed,
        "cohort": "source_validation",
        "participant_level_report": report,
        "frozen_source_record": {
            "path": record_path.relative_to(root).as_posix(),
            "file_sha256": sha256_file(record_path),
            "record_sha256": claimed,
        },
        "prediction_array": {
            "path": prediction_path.relative_to(root).as_posix(),
            "sha256": prediction["sha256"],
        },
        "calibrator": {
            "path": calibrator_path.relative_to(root).as_posix(),
            "file_sha256": sha256_file(calibrator_path),
            "record_sha256": calibrator_record["record_sha256"],
        },
        "target_information_used": False,
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    atomic_write_json_new(payload, derived_path, allowed_root=root)
    return {
        "model_id": model_id,
        "seed": seed,
        "cohort": "source_validation",
        "record_path": derived_path.relative_to(root).as_posix(),
        "record_file_sha256": sha256_file(derived_path),
        "record_sha256": payload["record_sha256"],
    }


def _write_normalization_record(
    source: SourceArrays,
    *,
    config: PaperAdaptationConfig,
    output: Path,
    root: Path,
    cache: PreparedPrimaryCacheEvidence,
    created_at_utc: str,
) -> dict[str, Any]:
    path = output / "source_training_normalization.json"
    normalization = source.normalizer.to_dict()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": "postconfirmatory_ccil_bpd_source_normalization",
        "status": "complete_create_only",
        "created_at_utc": created_at_utc,
        "evidence_status": EVIDENCE_STATUS,
        "normalization": normalization,
        "normalization_sha256": canonical_json_sha256(normalization),
        "fit_partition": "source_train",
        "fit_participants": list(SOURCE_TRAIN_PARTICIPANTS),
        "fit_window_count": EXPECTED_WINDOWS["source_train"],
        "primary_cache_record_sha256": cache.record_sha256,
        "primary_cache_record_file_sha256": cache.record_file_sha256,
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "target_in_fit": False,
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    atomic_write_json_new(payload, path, allowed_root=root)
    return {
        "path": path.relative_to(root).as_posix(),
        "file_sha256": sha256_file(path),
        "record_sha256": payload["record_sha256"],
        "normalization_sha256": payload["normalization_sha256"],
    }


def _run_source_selection(
    source: SourceArrays,
    *,
    config: PaperAdaptationConfig,
    output: Path,
    root: Path,
    cache: PreparedPrimaryCacheEvidence,
    code_commit: str,
    created_at_utc: str,
    device: torch.device,
) -> tuple[dict[str, AdaptationCandidate], list[dict[str, Any]], dict[str, Any]]:
    selected: dict[str, AdaptationCandidate] = {}
    entries: list[dict[str, Any]] = []
    for method_id in METHOD_IDS:
        reports: dict[str, Mapping[str, Any]] = {}
        for candidate in config.candidates[method_id]:
            run_directory = (
                output
                / "selection_runs"
                / method_id
                / candidate.candidate_id
                / f"seed-{SEED_ORDER[0]}"
            )
            trained = _train_candidate(
                source,
                config=config,
                candidate=candidate,
                seed=SEED_ORDER[0],
                role="source_candidate_selection",
                output_directory=run_directory,
                repository_root=root,
                cache=cache,
                code_commit=code_commit,
                device=device,
            )
            prediction_path = (
                output
                / "selection_predictions"
                / f"{method_id}--{candidate.candidate_id}--seed-{SEED_ORDER[0]}.npz"
            )
            prediction_sha = _write_prediction_new(
                prediction_path,
                method_id=method_id,
                seed=SEED_ORDER[0],
                cohort="source_validation_selection",
                window_ids=source.validation_window_ids,
                participant_ids=source.validation_participants,
                labels=source.validation_labels,
                logits=trained.validation_logits,
                uncalibrated=trained.validation_probabilities,
                calibrated=trained.validation_probabilities,
            )
            record = _source_candidate_record(
                trained,
                prediction_path=prediction_path,
                prediction_sha256=prediction_sha,
                config=config,
                root=root,
                cache=cache,
                code_commit=code_commit,
                created_at_utc=created_at_utc,
            )
            record_path = (
                output
                / "selection_records"
                / f"{method_id}--{candidate.candidate_id}--seed-{SEED_ORDER[0]}.json"
            )
            atomic_write_json_new(record, record_path, allowed_root=root)
            entries.append(
                {
                    "method_id": method_id,
                    "candidate_id": candidate.candidate_id,
                    "seed": SEED_ORDER[0],
                    "record_path": record_path.relative_to(root).as_posix(),
                    "record_file_sha256": sha256_file(record_path),
                    "record_sha256": record["record_sha256"],
                    "prediction_path": prediction_path.relative_to(root).as_posix(),
                    "prediction_sha256": prediction_sha,
                    "checkpoint_path": trained.checkpoint_path.relative_to(root).as_posix(),
                    "checkpoint_sha256": trained.checkpoint_sha256,
                }
            )
            reports[candidate.candidate_id] = trained.validation_report
            del trained
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        selected[method_id] = select_source_candidate(config.candidates[method_id], reports)
    lock = _write_source_selection_lock(
        selected=selected,
        entries=entries,
        config=config,
        output=output,
        root=root,
        cache=cache,
        code_commit=code_commit,
        created_at_utc=created_at_utc,
    )
    return selected, entries, lock


def _run_final_source_models(
    source: SourceArrays,
    *,
    selected: Mapping[str, AdaptationCandidate],
    selection_lock: Mapping[str, Any],
    normalization_entry: Mapping[str, Any],
    config: PaperAdaptationConfig,
    output: Path,
    root: Path,
    cache: PreparedPrimaryCacheEvidence,
    code_commit: str,
    created_at_utc: str,
    device: torch.device,
) -> tuple[list[TrainedAdapter], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    trained_runs: list[TrainedAdapter] = []
    source_entries: list[dict[str, Any]] = []
    comparator_entries: list[dict[str, Any]] = []
    for method_id in METHOD_IDS:
        candidate = selected[method_id]
        for seed in SEED_ORDER:
            trained = _train_candidate(
                source,
                config=config,
                candidate=candidate,
                seed=seed,
                role="final_source_fit_postconfirmatory",
                output_directory=output / "runs" / method_id / f"seed-{seed}",
                repository_root=root,
                cache=cache,
                code_commit=code_commit,
                device=device,
            )
            calibrator_record = build_source_temperature_calibrator_record(
                trained.validation_logits,
                source.validation_labels,
                source.validation_window_ids,
                fit_partition="source_validation",
                checkpoint_sha256=trained.checkpoint_sha256,
                training_configuration_sha256=trained.configuration_sha256,
                split_manifest_sha256=EXPECTED_SPLIT_SHA256,
                class_names=CLASS_NAMES,
            )
            calibrator_path = output / "calibrators" / f"{method_id}--seed-{seed}.json"
            written = write_source_temperature_calibrator_new(
                calibrator_record, calibrator_path, allowed_root=root
            )
            _, calibrator = load_source_temperature_calibrator_file(
                calibrator_path,
                expected_checkpoint_sha256=trained.checkpoint_sha256,
                expected_training_configuration_sha256=trained.configuration_sha256,
                expected_split_manifest_sha256=EXPECTED_SPLIT_SHA256,
            )
            calibrated = calibrator.probabilities(trained.validation_logits)
            report = classification_report(
                source.validation_labels,
                calibrated,
                source.validation_participants,
                class_names=CLASS_NAMES,
            )
            prediction_path = output / "predictions" / f"{method_id}--seed-{seed}--source.npz"
            prediction_sha = _write_prediction_new(
                prediction_path,
                method_id=method_id,
                seed=seed,
                cohort="source_validation",
                window_ids=source.validation_window_ids,
                participant_ids=source.validation_participants,
                labels=source.validation_labels,
                logits=trained.validation_logits,
                uncalibrated=trained.validation_probabilities,
                calibrated=calibrated,
            )
            record = _evaluation_record(
                trained=trained,
                cohort="source_validation",
                report=report,
                prediction_path=prediction_path,
                prediction_sha256=prediction_sha,
                calibrator_path=calibrator_path,
                calibrator_file_sha256=str(written["file_sha256"]),
                calibrator_record_sha256=str(written["record_sha256"]),
                source_stage_lock_sha256=None,
                config=config,
                root=root,
                code_commit=code_commit,
                created_at_utc=created_at_utc,
                cache_record_sha256=cache.record_sha256,
                cache_record_file_sha256=cache.record_file_sha256,
                opening_record_sha256=None,
            )
            record_path = output / "records" / f"{method_id}--seed-{seed}--source.json"
            atomic_write_json_new(record, record_path, allowed_root=root)
            source_entries.append(
                _entry(
                    record_path=record_path,
                    record=record,
                    prediction_path=prediction_path,
                    prediction_sha256=prediction_sha,
                    root=root,
                )
            )
            trained_runs.append(trained)
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
    for model_id in COMPARATOR_IDS:
        for seed in SEED_ORDER:
            comparator_entries.append(
                _load_frozen_source_comparator(
                    model_id=model_id,
                    seed=seed,
                    config=config,
                    source=source,
                    output=output,
                    root=root,
                    created_at_utc=created_at_utc,
                )
            )
    lock_path = output / str(config.outputs["source_stage_lock_filename"])
    if len(comparator_entries) != len(COMPARATOR_IDS) * len(SEED_ORDER):
        raise PaperAdaptationRunError("source stage lacks the locked comparator/seed matrix")
    lock_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": "postconfirmatory_ccil_bpd_source_stage_lock",
        "status": "complete_create_only_target_signals_not_accessed",
        "created_at_utc": created_at_utc,
        "evidence_status": EVIDENCE_STATUS,
        "required_method_order": list(METHOD_IDS),
        "required_comparator_order": list(COMPARATOR_IDS),
        "required_seed_order": list(SEED_ORDER),
        "model_seed_count": len(trained_runs),
        "source_results": source_entries,
        "frozen_source_comparator_references": comparator_entries,
        "selection_lock": dict(selection_lock),
        "normalization": dict(normalization_entry),
        "selected_candidates": {
            method_id: {
                "candidate_id": selected[method_id].candidate_id,
                "parameters": dict(selected[method_id].parameters),
            }
            for method_id in METHOD_IDS
        },
        "source_train_participants": list(SOURCE_TRAIN_PARTICIPANTS),
        "source_validation_participants": list(SOURCE_VALIDATION_PARTICIPANTS),
        "source_train_window_count": EXPECTED_WINDOWS["source_train"],
        "source_validation_window_count": EXPECTED_WINDOWS["source_validation"],
        "calibration_fit_partition": "source_validation",
        "checkpoint_selection_rule": "fixed_last_epoch",
        "primary_cache_record_sha256": cache.record_sha256,
        "primary_cache_record_file_sha256": cache.record_file_sha256,
        "experiment_config_file_sha256": config.file_sha256,
        "experiment_config_canonical_sha256": config.canonical_sha256,
        "code_commit": code_commit,
        "consumed_opening_metadata_validated_by_source_cache_loader": True,
        "target_signal_arrays_loaded": False,
        "target_labels_predictions_or_metrics_used_for_selection": False,
        "target_calibration_or_refit": False,
        "new_opening_or_unlock_invoked": False,
    }
    if len(trained_runs) != len(METHOD_IDS) * len(SEED_ORDER):
        raise PaperAdaptationRunError("source stage lacks the exact method/seed matrix")
    lock_payload["record_sha256"] = canonical_json_sha256(lock_payload)
    atomic_write_json_new(lock_payload, lock_path, allowed_root=root)
    lock = {
        "path": lock_path.relative_to(root).as_posix(),
        "file_sha256": sha256_file(lock_path),
        "record_sha256": lock_payload["record_sha256"],
    }
    return trained_runs, source_entries, comparator_entries, lock


def _load_target_cache_after_source_lock(
    *,
    source_lock: Mapping[str, Any],
    config: PaperAdaptationConfig,
    root: Path,
    cache_record_path: Path,
    expected_cache_record_file_sha256: str,
) -> tuple[PreparedPrimaryCacheEvidence, ConsumedTargetContext]:
    lock_path = _resolve_existing(
        cast(str, source_lock["path"]), root=root, name="source-stage lock"
    )
    lock_record = load_json_strict(lock_path)
    if not isinstance(lock_record, Mapping):
        raise PaperAdaptationRunError("source-stage lock root changed")
    body = dict(lock_record)
    claimed = body.pop("record_sha256", None)
    if (
        sha256_file(lock_path) != source_lock["file_sha256"]
        or claimed != source_lock["record_sha256"]
        or claimed != canonical_json_sha256(body)
        or lock_record.get("target_signal_arrays_loaded") is not False
    ):
        raise PaperAdaptationRunError("source-stage lock does not validate before target access")
    target = load_prepared_primary_cache(
        record_path=cache_record_path,
        expected_record_file_sha256=expected_cache_record_file_sha256,
        partition="target_sealed",
        opening_receipt_path=cast(str, config.opening_1["opening_receipt_path"]),
        locked_target_index_path=cast(str, config.opening_1["locked_target_index_path"]),
        artifact_root=root,
        expected_split_manifest_sha256=EXPECTED_SPLIT_SHA256,
        expected_source_artifact_sha256=EXPECTED_SOURCE_SHA256,
    )
    context = load_consumed_target_context(
        opening_receipt_path=cast(str, config.opening_1["opening_receipt_path"]),
        locked_target_index_path=cast(str, config.opening_1["locked_target_index_path"]),
        artifact_root=root,
    )
    if (
        context.receipt_file_sha256 != config.opening_1["opening_receipt_file_sha256"]
        or context.receipt.get("record_sha256") != config.opening_1["opening_receipt_record_sha256"]
        or context.index_file_sha256 != config.opening_1["locked_target_index_file_sha256"]
        or context.index.get("record_sha256")
        != config.opening_1["locked_target_index_record_sha256"]
        or context.index.get("target_seal_id") != config.opening_1["target_seal_id"]
        or target.cache.batch.signals.shape != (EXPECTED_WINDOWS["target_sealed"], 128, 6)
        or set(target.cache.batch.participant_ids) != set(TARGET_PARTICIPANTS)
    ):
        raise PaperAdaptationRunError("target cache/opening-1 contract changed")
    return target, context


def _run_target_models(
    trained_runs: Sequence[TrainedAdapter],
    *,
    source: SourceArrays,
    source_lock: Mapping[str, Any],
    target: PreparedPrimaryCacheEvidence,
    context: ConsumedTargetContext,
    config: PaperAdaptationConfig,
    output: Path,
    root: Path,
    code_commit: str,
    created_at_utc: str,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    batch = target.cache.batch
    normalized = source.normalizer.transform(batch.signals)
    target_entries: list[dict[str, Any]] = []
    for trained in trained_runs:
        model, _ = reconstruct_paper_adaptation_checkpoint(
            trained.checkpoint_path,
            expected_sha256=trained.checkpoint_sha256,
            expected_method_id=trained.method_id,
            expected_seed=trained.seed,
            expected_configuration_sha256=trained.configuration_sha256,
            expected_code_commit=code_commit,
            device=device,
        )
        logits, uncalibrated, _ = predict_model(
            model,
            normalized,
            batch.labels,
            list(batch.participant_ids),
            class_names=CLASS_NAMES,
            batch_size=int(config.training["batch_size"]),
            device=device,
            mixed_precision="float16",
        )
        calibrator_path = output / "calibrators" / f"{trained.method_id}--seed-{trained.seed}.json"
        calibrator_record, calibrator = load_source_temperature_calibrator_file(
            calibrator_path,
            expected_checkpoint_sha256=trained.checkpoint_sha256,
            expected_training_configuration_sha256=trained.configuration_sha256,
            expected_split_manifest_sha256=EXPECTED_SPLIT_SHA256,
        )
        calibrated = calibrator.probabilities(logits)
        report = classification_report(
            batch.labels,
            calibrated,
            batch.participant_ids,
            class_names=CLASS_NAMES,
        )
        prediction_path = (
            output / "predictions" / f"{trained.method_id}--seed-{trained.seed}--target.npz"
        )
        prediction_sha = _write_prediction_new(
            prediction_path,
            method_id=trained.method_id,
            seed=trained.seed,
            cohort="target_sealed",
            window_ids=batch.window_ids,
            participant_ids=batch.participant_ids,
            labels=batch.labels,
            logits=logits,
            uncalibrated=uncalibrated,
            calibrated=calibrated,
        )
        record = _evaluation_record(
            trained=trained,
            cohort="target_sealed",
            report=report,
            prediction_path=prediction_path,
            prediction_sha256=prediction_sha,
            calibrator_path=calibrator_path,
            calibrator_file_sha256=sha256_file(calibrator_path),
            calibrator_record_sha256=str(calibrator_record["record_sha256"]),
            source_stage_lock_sha256=str(source_lock["record_sha256"]),
            config=config,
            root=root,
            code_commit=code_commit,
            created_at_utc=created_at_utc,
            cache_record_sha256=target.record_sha256,
            cache_record_file_sha256=target.record_file_sha256,
            opening_record_sha256=str(context.receipt["record_sha256"]),
        )
        record_path = output / "records" / f"{trained.method_id}--seed-{trained.seed}--target.json"
        atomic_write_json_new(record, record_path, allowed_root=root)
        target_entries.append(
            _entry(
                record_path=record_path,
                record=record,
                prediction_path=prediction_path,
                prediction_sha256=prediction_sha,
                root=root,
            )
        )
        del model
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    comparator_entries: list[dict[str, Any]] = []
    for model_id in COMPARATOR_IDS:
        for seed in SEED_ORDER:
            reference = load_target_clean_reference(context, model_id=model_id, seed=seed)
            index_entry = context.result_entries[(model_id, seed)]
            comparator_entries.append(
                {
                    "model_id": model_id,
                    "seed": seed,
                    "cohort": "target_sealed",
                    "record_path": reference.record_path.relative_to(root).as_posix(),
                    "record_file_sha256": reference.record_file_sha256,
                    "record_sha256": reference.record_sha256,
                    "prediction_path": index_entry["array_path"],
                    "prediction_sha256": reference.prediction_sha256,
                    "evidence_status": ("locked_confirmatory_target_opening_1_consumed_reference"),
                }
            )
    if len(target_entries) != len(METHOD_IDS) * len(SEED_ORDER):
        raise PaperAdaptationRunError("target stage lacks the exact method/seed matrix")
    if len(comparator_entries) != len(COMPARATOR_IDS) * len(SEED_ORDER):
        raise PaperAdaptationRunError("target stage lacks the consumed comparator/seed matrix")
    return target_entries, comparator_entries


def _write_index(
    *,
    status: str,
    config: PaperAdaptationConfig,
    output: Path,
    root: Path,
    code_commit: str,
    created_at_utc: str,
    cache: PreparedPrimaryCacheEvidence,
    normalization: Mapping[str, Any] | None,
    selection_lock: Mapping[str, Any] | None,
    source_lock: Mapping[str, Any] | None,
    selection_entries: Sequence[Mapping[str, Any]],
    source_entries: Sequence[Mapping[str, Any]],
    target_entries: Sequence[Mapping[str, Any]],
    comparator_source_entries: Sequence[Mapping[str, Any]],
    comparator_target_entries: Sequence[Mapping[str, Any]],
    target_accessed: bool,
    failure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    path = output / str(config.outputs["index_filename"])
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": "postconfirmatory_ccil_bpd_evaluation_index",
        "status": status,
        "created_at_utc": created_at_utc,
        "evidence_status": EVIDENCE_STATUS,
        "track_role": TRACK_ROLE,
        "primary_claim_eligible": False,
        "designed_after_target_opening": True,
        "experiment_id": config.experiment_id,
        "experiment_config": {
            "path": config.path.relative_to(root).as_posix(),
            "file_sha256": config.file_sha256,
            "canonical_sha256": config.canonical_sha256,
        },
        "code_commit": code_commit,
        "required_device": "cuda",
        "execution": "sequential_cuda_one_model_seed_at_a_time",
        "required_method_order": list(METHOD_IDS),
        "required_comparator_order": list(COMPARATOR_IDS),
        "required_seed_order": list(SEED_ORDER),
        "selection_candidate_count": len(selection_entries),
        "source_result_count": len(source_entries),
        "target_result_count": len(target_entries),
        "comparator_source_reference_count": len(comparator_source_entries),
        "comparator_target_reference_count": len(comparator_target_entries),
        "selection_results": [dict(entry) for entry in selection_entries],
        "source_results": [dict(entry) for entry in source_entries],
        "target_results": [dict(entry) for entry in target_entries],
        "frozen_source_comparator_references": [dict(entry) for entry in comparator_source_entries],
        "consumed_opening_1_comparator_target_references": [
            dict(entry) for entry in comparator_target_entries
        ],
        "normalization": dict(normalization) if normalization is not None else None,
        "source_selection_lock": dict(selection_lock) if selection_lock is not None else None,
        "source_stage_lock": dict(source_lock) if source_lock is not None else None,
        "primary_cache_record": {
            "path": cache.record_path.relative_to(root).as_posix(),
            "file_sha256": cache.record_file_sha256,
            "record_sha256": cache.record_sha256,
        },
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "source_artifact_sha256": EXPECTED_SOURCE_SHA256,
        "opening_receipt_record_sha256": config.opening_1["opening_receipt_record_sha256"],
        "locked_target_index_record_sha256": config.opening_1["locked_target_index_record_sha256"],
        "target_seal_id": config.opening_1["target_seal_id"],
        "source_stage_completed_before_target_signal_access": source_lock is not None,
        "target_signals_or_labels_accessed": target_accessed,
        "target_information_used_for_source_selection": False,
        "target_tuning_or_refit": False,
        "new_opening_or_unlock_invoked": False,
        "statistical_unit": "participant",
        "trial_safe": False,
        "interpretation": "post_confirmatory_descriptive_only",
        "limitations": list(config.limitations),
        "forbidden_claims": list(config.forbidden_claims),
        "failure": dict(failure) if failure is not None else None,
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    atomic_write_json_new(payload, path, allowed_root=root)
    return payload


def _preserve_failure(
    *,
    exc: Exception,
    stage: str,
    output: Path,
    root: Path,
    code_commit: str,
    created_at_utc: str,
    target_accessed: bool,
) -> dict[str, Any]:
    path = output / "failures" / f"failure-{stage}.json"
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": "postconfirmatory_ccil_bpd_failure",
        "status": "failed_preserved_create_only",
        "created_at_utc": created_at_utc,
        "evidence_status": "failed_run_not_result_evidence",
        "stage": stage,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "code_commit": code_commit,
        "target_signals_or_labels_accessed_before_failure": target_accessed,
        "target_tuning_or_refit": False,
        "new_opening_or_unlock_invoked": False,
        "partial_outputs_preserved": True,
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    atomic_write_json_new(payload, path, allowed_root=root)
    return {
        "path": path.relative_to(root).as_posix(),
        "file_sha256": sha256_file(path),
        "record_sha256": payload["record_sha256"],
        "stage": stage,
    }


def run_paper_adaptation_extension(
    *,
    config_path: str | Path,
    repository_root: str | Path,
    primary_cache_record_path: str | Path,
    expected_primary_cache_record_file_sha256: str,
    expected_code_commit: str,
    created_at_utc: str,
    device: torch.device,
) -> dict[str, Any]:
    """Execute the full source-locked extension without any new target opening."""

    # This gate intentionally precedes configuration, Git, cache, opening, and
    # output access. Real neural execution is never silently moved to CPU.
    if device.type != "cuda" or not torch.cuda.is_available():
        raise PaperAdaptationRunError(
            "CUDA is required before configuration or cache access; CPU neural fallback is forbidden"
        )
    try:
        probe = torch.ones(1, device=device)
        if float((probe + 1).item()) != 2.0:
            raise RuntimeError("CUDA arithmetic probe returned an invalid value")
        torch.cuda.synchronize(device)
    except Exception as exc:
        raise PaperAdaptationRunError(
            "CUDA runtime probe failed before configuration or cache access"
        ) from exc
    if _SHA256_RE.fullmatch(expected_primary_cache_record_file_sha256) is None:
        raise PaperAdaptationRunError("primary cache record pin must be a lowercase SHA-256")
    commit_expected = expected_code_commit.casefold()
    if _COMMIT_RE.fullmatch(commit_expected) is None:
        raise PaperAdaptationRunError("expected_code_commit must be a full Git commit")
    timestamp = require_utc_timestamp(created_at_utc, location="created_at_utc")
    root_candidate = Path(repository_root)
    if root_candidate.is_symlink():
        raise PaperAdaptationRunError("repository_root may not be a symlink")
    root = root_candidate.resolve(strict=True)
    config = load_paper_adaptation_config(config_path)
    try:
        config.path.relative_to(root)
    except ValueError as exc:
        raise PaperAdaptationRunError("experiment config escapes repository_root") from exc
    observed_commit, status = _git_state(root)
    if observed_commit != commit_expected:
        raise PaperAdaptationRunError("Git HEAD differs from the externally pinned code commit")
    if status:
        raise PaperAdaptationRunError("worktree must be clean before post-confirmatory training")
    _validate_static_provenance(config, root=root)
    cache_record = _resolve_existing(
        primary_cache_record_path, root=root, name="primary cache record"
    )
    if sha256_file(cache_record) != expected_primary_cache_record_file_sha256:
        raise PaperAdaptationRunError("primary cache record file hash changed before cache loading")
    source, train_cache, _ = _load_source_arrays(
        config=config,
        root=root,
        cache_record_path=cache_record,
        expected_cache_record_file_sha256=expected_primary_cache_record_file_sha256,
    )
    output = _resolve_new_directory(cast(str, config.outputs["output_directory"]), root=root)
    output.mkdir(parents=True, exist_ok=False)
    selection_entries: list[dict[str, Any]] = []
    source_entries: list[dict[str, Any]] = []
    target_entries: list[dict[str, Any]] = []
    comparator_source_entries: list[dict[str, Any]] = []
    comparator_target_entries: list[dict[str, Any]] = []
    normalization_entry: dict[str, Any] | None = None
    selection_lock: dict[str, Any] | None = None
    source_lock: dict[str, Any] | None = None
    target_accessed = False
    stage = "source_selection"
    try:
        normalization_entry = _write_normalization_record(
            source,
            config=config,
            output=output,
            root=root,
            cache=train_cache,
            created_at_utc=timestamp,
        )
        selected, selection_entries, selection_lock = _run_source_selection(
            source,
            config=config,
            output=output,
            root=root,
            cache=train_cache,
            code_commit=commit_expected,
            created_at_utc=timestamp,
            device=device,
        )
        stage = "final_source_training_and_calibration"
        trained, source_entries, comparator_source_entries, source_lock = _run_final_source_models(
            source,
            selected=selected,
            selection_lock=selection_lock,
            normalization_entry=normalization_entry,
            config=config,
            output=output,
            root=root,
            cache=train_cache,
            code_commit=commit_expected,
            created_at_utc=timestamp,
            device=device,
        )
        stage = "postconfirmatory_target_evaluation"
        target_accessed = True
        target, context = _load_target_cache_after_source_lock(
            source_lock=source_lock,
            config=config,
            root=root,
            cache_record_path=cache_record,
            expected_cache_record_file_sha256=expected_primary_cache_record_file_sha256,
        )
        target_entries, comparator_target_entries = _run_target_models(
            trained,
            source=source,
            source_lock=source_lock,
            target=target,
            context=context,
            config=config,
            output=output,
            root=root,
            code_commit=commit_expected,
            created_at_utc=timestamp,
            device=device,
        )
        return _write_index(
            status="complete_create_only",
            config=config,
            output=output,
            root=root,
            code_commit=commit_expected,
            created_at_utc=timestamp,
            cache=train_cache,
            normalization=normalization_entry,
            selection_lock=selection_lock,
            source_lock=source_lock,
            selection_entries=selection_entries,
            source_entries=source_entries,
            target_entries=target_entries,
            comparator_source_entries=comparator_source_entries,
            comparator_target_entries=comparator_target_entries,
            target_accessed=target_accessed,
            failure=None,
        )
    except Exception as exc:
        failure = _preserve_failure(
            exc=exc,
            stage=stage,
            output=output,
            root=root,
            code_commit=commit_expected,
            created_at_utc=timestamp,
            target_accessed=target_accessed,
        )
        _write_index(
            status="failed_preserved_create_only",
            config=config,
            output=output,
            root=root,
            code_commit=commit_expected,
            created_at_utc=timestamp,
            cache=train_cache,
            normalization=normalization_entry,
            selection_lock=selection_lock,
            source_lock=source_lock,
            selection_entries=selection_entries,
            source_entries=source_entries,
            target_entries=target_entries,
            comparator_source_entries=comparator_source_entries,
            comparator_target_entries=comparator_target_entries,
            target_accessed=target_accessed,
            failure=failure,
        )
        raise PaperAdaptationRunError(
            f"paper-adaptation extension failed at {stage}; all partial evidence was preserved"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--primary-cache-record", type=Path, required=True)
    parser.add_argument("--expected-primary-cache-record-file-sha256", required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--created-at-utc", required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run_paper_adaptation_extension(
        config_path=arguments.config,
        repository_root=arguments.repository_root,
        primary_cache_record_path=arguments.primary_cache_record,
        expected_primary_cache_record_file_sha256=(
            arguments.expected_primary_cache_record_file_sha256
        ),
        expected_code_commit=arguments.expected_code_commit,
        created_at_utc=arguments.created_at_utc,
        device=torch.device(arguments.device),
    )
    print(json.dumps({"status": result["status"], "record_sha256": result["record_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
