"""Deterministic, participant-exclusive source folds for UCI-HAR v1."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.data.uci_har import (
    UCI_HAR_ACTIVITY_NAMES,
    UCI_HAR_CHANNELS,
    UCI_HAR_DATASET_ID,
    UCIHARWindows,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256

UCI_SOURCE_PROTOCOL_ALGORITHM_VERSION = "1.0.0"
UCI_SOURCE_FOLD_NAMESPACE = "inclusive-shift-har|uci_har_v1|source_grouped_cv_v1"


class UCISourceProtocolError(ValueError):
    """Raised when source-only participant folds violate protocol invariants."""


@dataclass(frozen=True)
class UCISourceFold:
    """One participant-exclusive train/validation assignment."""

    fold_id: str
    train_subject_ids: tuple[int, ...]
    validation_subject_ids: tuple[int, ...]
    train_indices: NDArray[np.int64]
    validation_indices: NDArray[np.int64]
    train_window_ids_sha256: str
    validation_window_ids_sha256: str


def _subject_rank(subject_id: int, *, seed: int) -> bytes:
    token = f"{UCI_SOURCE_FOLD_NAMESPACE}|seed={seed}|subject={subject_id}"
    return hashlib.sha256(token.encode("utf-8")).digest()


def _window_id_hash(window_ids: list[str]) -> str:
    return canonical_json_sha256(sorted(window_ids))


def build_grouped_source_folds(
    windows: UCIHARWindows,
    *,
    n_folds: int = 5,
    seed: int = 5062,
) -> tuple[UCISourceFold, ...]:
    """Build deterministic folds using only released official-train subjects."""

    if windows.released_split != "train":
        raise UCISourceProtocolError("source folds may only be built from official train")
    if isinstance(n_folds, bool) or n_folds < 2:
        raise UCISourceProtocolError("n_folds must be an integer of at least two")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise UCISourceProtocolError("seed must be a non-negative integer")
    subjects = sorted(set(int(value) for value in windows.subject_ids))
    if len(subjects) < n_folds:
        raise UCISourceProtocolError("n_folds cannot exceed the number of source subjects")
    ranked_subjects = sorted(subjects, key=lambda value: (_subject_rank(value, seed=seed), value))
    validation_buckets: list[list[int]] = [[] for _ in range(n_folds)]
    for index, subject_id in enumerate(ranked_subjects):
        validation_buckets[index % n_folds].append(subject_id)

    folds: list[UCISourceFold] = []
    for fold_index, bucket in enumerate(validation_buckets, start=1):
        validation_subjects = tuple(sorted(bucket))
        validation_set = set(validation_subjects)
        train_subjects = tuple(subject for subject in subjects if subject not in validation_set)
        validation_mask = np.isin(windows.subject_ids, validation_subjects)
        validation_indices = np.asarray(np.flatnonzero(validation_mask), dtype=np.int64)
        train_indices = np.asarray(np.flatnonzero(~validation_mask), dtype=np.int64)
        train_indices.setflags(write=False)
        validation_indices.setflags(write=False)
        folds.append(
            UCISourceFold(
                fold_id=f"uci_source_cv_{fold_index:02d}",
                train_subject_ids=train_subjects,
                validation_subject_ids=validation_subjects,
                train_indices=train_indices,
                validation_indices=validation_indices,
                train_window_ids_sha256=_window_id_hash(
                    [windows.window_ids[int(index)] for index in train_indices]
                ),
                validation_window_ids_sha256=_window_id_hash(
                    [windows.window_ids[int(index)] for index in validation_indices]
                ),
            )
        )
    audit_grouped_source_folds(windows, tuple(folds))
    return tuple(folds)


def audit_grouped_source_folds(
    windows: UCIHARWindows,
    folds: tuple[UCISourceFold, ...],
) -> dict[str, Any]:
    """Prove participant/window exclusivity and complete validation coverage."""

    if windows.released_split != "train":
        raise UCISourceProtocolError("fold audit requires the official train split")
    if not folds:
        raise UCISourceProtocolError("at least one fold is required")
    all_subjects = set(int(value) for value in windows.subject_ids)
    all_indices = set(range(len(windows.window_ids)))
    validation_subject_counts: Counter[int] = Counter()
    fold_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    for fold in folds:
        if fold.fold_id in fold_ids:
            raise UCISourceProtocolError(f"duplicate fold ID: {fold.fold_id}")
        fold_ids.add(fold.fold_id)
        train_subjects = set(fold.train_subject_ids)
        validation_subjects = set(fold.validation_subject_ids)
        if train_subjects & validation_subjects:
            raise UCISourceProtocolError(f"subject overlap in {fold.fold_id}")
        if train_subjects | validation_subjects != all_subjects:
            raise UCISourceProtocolError(f"subject coverage failure in {fold.fold_id}")
        train_indices = set(int(value) for value in fold.train_indices)
        validation_indices = set(int(value) for value in fold.validation_indices)
        if train_indices & validation_indices:
            raise UCISourceProtocolError(f"window overlap in {fold.fold_id}")
        if train_indices | validation_indices != all_indices:
            raise UCISourceProtocolError(f"window coverage failure in {fold.fold_id}")
        observed_train_subjects = set(
            int(value) for value in windows.subject_ids[fold.train_indices]
        )
        observed_validation_subjects = set(
            int(value) for value in windows.subject_ids[fold.validation_indices]
        )
        if observed_train_subjects != train_subjects:
            raise UCISourceProtocolError(f"train subject/index mismatch in {fold.fold_id}")
        if observed_validation_subjects != validation_subjects:
            raise UCISourceProtocolError(f"validation subject/index mismatch in {fold.fold_id}")
        observed_train_hash = _window_id_hash(
            [windows.window_ids[int(index)] for index in fold.train_indices]
        )
        observed_validation_hash = _window_id_hash(
            [windows.window_ids[int(index)] for index in fold.validation_indices]
        )
        if observed_train_hash != fold.train_window_ids_sha256:
            raise UCISourceProtocolError(f"train window hash mismatch in {fold.fold_id}")
        if observed_validation_hash != fold.validation_window_ids_sha256:
            raise UCISourceProtocolError(f"validation window hash mismatch in {fold.fold_id}")
        validation_subject_counts.update(validation_subjects)
        validation_labels = sorted(
            set(int(value) for value in windows.activity_ids[fold.validation_indices])
        )
        records.append(
            {
                "fold_id": fold.fold_id,
                "train_subject_count": len(train_subjects),
                "train_window_count": len(train_indices),
                "validation_activity_ids": validation_labels,
                "validation_subject_count": len(validation_subjects),
                "validation_window_count": len(validation_indices),
            }
        )
    expected_counts = Counter({subject: 1 for subject in all_subjects})
    if validation_subject_counts != expected_counts:
        raise UCISourceProtocolError(
            "every source subject must appear in validation exactly once across folds"
        )
    return {
        "all_source_subjects_covered": True,
        "fold_count": len(folds),
        "folds": records,
        "participant_overlap": False,
        "status": "pass",
        "validation_subject_once": True,
        "window_overlap": False,
    }


def build_uci_source_protocol_manifest(
    windows: UCIHARWindows,
    *,
    dataset_manifest_sha256: str,
    n_folds: int = 5,
    seed: int = 5062,
) -> dict[str, Any]:
    """Build the immutable source-only protocol record for corrected reproduction."""

    folds = build_grouped_source_folds(windows, n_folds=n_folds, seed=seed)
    audit = audit_grouped_source_folds(windows, folds)
    fold_records: list[dict[str, Any]] = []
    for fold in folds:
        fold_records.append(
            {
                "fold_id": fold.fold_id,
                "normalization_fit_subjects": list(fold.train_subject_ids),
                "train_subject_ids": list(fold.train_subject_ids),
                "train_window_count": int(fold.train_indices.size),
                "train_window_ids_sha256": fold.train_window_ids_sha256,
                "validation_subject_ids": list(fold.validation_subject_ids),
                "validation_window_count": int(fold.validation_indices.size),
                "validation_window_ids_sha256": fold.validation_window_ids_sha256,
            }
        )
    payload: dict[str, Any] = {
        "audit": audit,
        "class_count_policy": "fixed_locked_uci_native_ontology_six_never_test_derived",
        "dataset_id": UCI_HAR_DATASET_ID,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "evidence_status": "corrected_source_development_protocol_no_model_results",
        "fold_assignment": {
            "algorithm": "sha256_rank_then_round_robin",
            "folds": fold_records,
            "namespace": UCI_SOURCE_FOLD_NAMESPACE,
            "seed": seed,
        },
        "input": {
            "archive_sha256": windows.archive_sha256,
            "channels": list(UCI_HAR_CHANNELS),
            "released_split": "train",
            "subject_count": len(set(int(value) for value in windows.subject_ids)),
            "window_count": len(windows.window_ids),
            "window_ids_sha256": _window_id_hash(list(windows.window_ids)),
        },
        "label_ontology": {
            "activity_ids": UCI_HAR_ACTIVITY_NAMES,
            "source": "locked_release_metadata_not_test_inference",
        },
        "leakage_controls": {
            "normalization": "fit_on_each_fold_training_windows_only",
            "official_test_opened_for_protocol_construction": False,
            "official_test_status": "legacy_exploratory_development_consumed",
            "overlapping_released_windows_cross_partitions": False,
            "participant_partition_before_model_preprocessing": True,
            "raw_sample_overlap_proof_basis": "participant_exclusivity",
        },
        "protocol_algorithm_version": UCI_SOURCE_PROTOCOL_ALGORITHM_VERSION,
        "protocol_id": "uci_har_v1_source_grouped_cv_v1",
        "source_pretraining_policy": {
            "allowed_training_partition": "official_train_only",
            "cross_source_mapping_controlled_by_locked_ontology": True,
            "official_test_allowed_for_new_claims": False,
        },
        "status": "source_protocol_ready_no_training_run",
    }
    payload["protocol_sha256"] = canonical_json_sha256(payload)
    return payload
