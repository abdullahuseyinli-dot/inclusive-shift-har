"""Validate and aggregate the post-confirmatory CCIL/BPD adapter extension."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.evaluation._strict_config import require_utc_timestamp
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.evaluation.postconfirmatory_inputs import (
    ConsumedTargetContext,
    load_consumed_target_context,
    load_target_clean_reference,
)
from inclusive_shift_har.evaluation.source_calibration import (
    load_source_temperature_calibrator_file,
)
from inclusive_shift_har.evaluation.statistics import (
    holm_adjust,
    paired_participant_comparison,
    participant_bootstrap_interval,
)
from inclusive_shift_har.experiments.ccil_bpd_postconfirmatory import (
    CLASS_NAMES,
    COMPARATOR_IDS,
    EVIDENCE_STATUS,
    EXPECTED_WINDOWS,
    METHOD_IDS,
    SCHEMA_VERSION,
    SEED_ORDER,
    SOURCE_VALIDATION_PARTICIPANTS,
    TARGET_PARTICIPANTS,
    TRACK_ROLE,
    load_paper_adaptation_config,
)
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)


class PaperAdaptationAggregationError(ValueError):
    """Raised when an adapter artifact is incomplete, changed, or misaligned."""


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PaperAdaptationAggregationError(f"{name} must be an object")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise PaperAdaptationAggregationError(f"{name} must be a list")
    return value


def _self_hash(record: Mapping[str, Any], *, name: str) -> str:
    claimed = record.get("record_sha256")
    body = dict(record)
    body.pop("record_sha256", None)
    if not isinstance(claimed, str) or claimed != canonical_json_sha256(body):
        raise PaperAdaptationAggregationError(f"{name} self-hash does not validate")
    return claimed


def _resolve_file(value: Any, *, root: Path, name: str, allow_absolute: bool = False) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise PaperAdaptationAggregationError(f"{name} path is missing")
    raw = Path(value)
    if raw.is_absolute() and not allow_absolute:
        raise PaperAdaptationAggregationError(f"{name} path must be repository-relative")
    if not raw.is_absolute() and ".." in raw.parts:
        raise PaperAdaptationAggregationError(f"{name} path escapes repository_root")
    candidate = raw if raw.is_absolute() else root / raw
    if candidate.is_symlink():
        raise PaperAdaptationAggregationError(f"{name} may not be a symlink")
    path = candidate.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PaperAdaptationAggregationError(f"{name} escapes repository_root") from exc
    if path.is_symlink() or not path.is_file():
        raise PaperAdaptationAggregationError(f"{name} must be a regular file")
    return path


def _new_file(value: str | Path, *, root: Path, name: str) -> Path:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    path = candidate.resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PaperAdaptationAggregationError(f"{name} escapes repository_root") from exc
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite {name}: {path}")
    return path


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PaperAdaptationAggregationError(f"{name} must be finite")
    result = float(value)
    if not np.isfinite(result):
        raise PaperAdaptationAggregationError(f"{name} must be finite")
    return result


def _participant_values(
    report_value: Any,
    *,
    expected_participants: frozenset[str],
) -> dict[str, dict[str, float]]:
    report = _mapping(report_value, name="participant report")
    rows = _sequence(report.get("participants"), name="participant rows")
    result: dict[str, dict[str, float]] = {}
    for value in rows:
        row = _mapping(value, name="participant row")
        participant = row.get("participant_id")
        if not isinstance(participant, str) or participant in result:
            raise PaperAdaptationAggregationError("participant identity is invalid or duplicated")
        result[participant] = {
            "macro_f1": _finite(row.get("macro_f1"), name="participant macro-F1"),
            "balanced_accuracy": _finite(
                row.get("balanced_accuracy"), name="participant balanced accuracy"
            ),
        }
    if set(result) != set(expected_participants) or report.get("participant_count") != len(result):
        raise PaperAdaptationAggregationError("participant report cohort coverage changed")
    return result


def _load_extension_entry(
    entry: Mapping[str, Any],
    *,
    root: Path,
    method_id: str,
    seed: int,
    cohort: str,
    expected_alignment: tuple[tuple[str, ...], tuple[str, ...], NDArray[np.int64]] | None = None,
) -> Mapping[str, Any]:
    if (
        entry.get("method_id") != method_id
        or entry.get("seed") != seed
        or entry.get("cohort") != cohort
    ):
        raise PaperAdaptationAggregationError("adapter index entry identity changed")
    record_path = _resolve_file(entry.get("record_path"), root=root, name="adapter record")
    if sha256_file(record_path) != entry.get("record_file_sha256"):
        raise PaperAdaptationAggregationError("adapter record file hash changed")
    record = _mapping(load_json_strict(record_path), name="adapter record")
    record_hash = _self_hash(record, name="adapter record")
    required = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": "postconfirmatory_paper_adaptation_evaluation",
        "status": "complete_create_only",
        "evidence_status": EVIDENCE_STATUS,
        "track_role": TRACK_ROLE,
        "primary_claim_eligible": False,
        "method_id": method_id,
        "seed": seed,
        "cohort": cohort,
        "target_tuning_or_refit": False,
        "target_information_used_for_source_selection": False,
        "new_opening_or_unlock_invoked": False,
        "trial_safe": False,
        "statistical_unit": "participant",
    }
    mismatches = [key for key, expected in required.items() if record.get(key) != expected]
    if record_hash != entry.get("record_sha256") or mismatches:
        raise PaperAdaptationAggregationError(f"adapter record contract changed: {mismatches}")
    prediction = _mapping(record.get("prediction_array"), name="adapter prediction reference")
    prediction_path = _resolve_file(
        prediction.get("path"), root=root, name="adapter prediction array"
    )
    if (
        sha256_file(prediction_path) != prediction.get("sha256")
        or prediction.get("sha256") != entry.get("prediction_sha256")
        or prediction_path.relative_to(root).as_posix() != entry.get("prediction_path")
    ):
        raise PaperAdaptationAggregationError("adapter prediction lineage changed")
    required_keys = {
        "evidence_status",
        "method_id",
        "seed",
        "cohort",
        "window_ids",
        "participant_ids",
        "true_labels",
        "logits",
        "uncalibrated_probabilities",
        "calibrated_probabilities",
        "predicted_labels",
    }
    with np.load(prediction_path, allow_pickle=False) as arrays:
        if set(arrays.files) != required_keys:
            raise PaperAdaptationAggregationError("adapter prediction schema changed")
        labels = np.asarray(arrays["true_labels"], dtype=np.int64)
        logits = np.asarray(arrays["logits"], dtype=np.float64)
        uncalibrated = np.asarray(arrays["uncalibrated_probabilities"], dtype=np.float64)
        calibrated = np.asarray(arrays["calibrated_probabilities"], dtype=np.float64)
        predicted = np.asarray(arrays["predicted_labels"], dtype=np.int64)
        participants = tuple(str(value) for value in arrays["participant_ids"].tolist())
        windows = tuple(str(value) for value in arrays["window_ids"].tolist())
        stored_method = tuple(str(value) for value in arrays["method_id"].tolist())
        stored_seed = np.asarray(arrays["seed"], dtype=np.int64)
        stored_cohort = tuple(str(value) for value in arrays["cohort"].tolist())
        stored_status = tuple(str(value) for value in arrays["evidence_status"].tolist())
    count = labels.size
    if (
        logits.shape != (count, len(CLASS_NAMES))
        or uncalibrated.shape != logits.shape
        or calibrated.shape != logits.shape
        or len(participants) != count
        or len(windows) != count
        or len(set(windows)) != count
        or not np.isfinite(logits).all()
        or not np.allclose(uncalibrated.sum(axis=1), 1.0, atol=1e-8, rtol=1e-8)
        or not np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-8, rtol=1e-8)
        or not np.array_equal(predicted, calibrated.argmax(axis=1))
        or stored_method != (method_id,)
        or stored_seed.tolist() != [seed]
        or stored_cohort != (cohort,)
        or stored_status != (EVIDENCE_STATUS,)
    ):
        raise PaperAdaptationAggregationError("adapter prediction arrays are misaligned")
    expected_participants = (
        frozenset(SOURCE_VALIDATION_PARTICIPANTS)
        if cohort == "source_validation"
        else frozenset(TARGET_PARTICIPANTS)
    )
    if expected_alignment is not None and (
        windows != expected_alignment[0]
        or participants != expected_alignment[1]
        or not np.array_equal(labels, expected_alignment[2])
    ):
        raise PaperAdaptationAggregationError(
            "adapter target alignment differs from consumed opening 1"
        )
    reconstructed = classification_report(labels, calibrated, participants, class_names=CLASS_NAMES)
    _participant_values(reconstructed, expected_participants=expected_participants)
    if canonical_json_sha256(reconstructed) != canonical_json_sha256(
        _mapping(record.get("participant_level_report"), name="stored adapter report")
    ):
        raise PaperAdaptationAggregationError(
            "adapter report does not reconstruct from predictions"
        )
    return reconstructed


def _load_source_reference(
    entry: Mapping[str, Any],
    *,
    root: Path,
    model_id: str,
    seed: int,
) -> Mapping[str, Any]:
    if (
        entry.get("model_id") != model_id
        or entry.get("seed") != seed
        or entry.get("cohort") != "source_validation"
    ):
        raise PaperAdaptationAggregationError("source comparator entry identity changed")
    path = _resolve_file(entry.get("record_path"), root=root, name="source comparator record")
    if sha256_file(path) != entry.get("record_file_sha256"):
        raise PaperAdaptationAggregationError("source comparator record file hash changed")
    record = _mapping(load_json_strict(path), name="source comparator record")
    if (
        _self_hash(record, name="source comparator record") != entry.get("record_sha256")
        or record.get("record_kind") != "postconfirmatory_frozen_source_comparator_reference"
        or record.get("status") != "complete_create_only"
        or record.get("model_id") != model_id
        or record.get("seed") != seed
        or record.get("cohort") != "source_validation"
        or record.get("target_information_used") is not False
    ):
        raise PaperAdaptationAggregationError("source comparator contract changed")
    prediction = _mapping(record.get("prediction_array"), name="source comparator prediction")
    prediction_path = _resolve_file(
        prediction.get("path"), root=root, name="source comparator prediction"
    )
    if sha256_file(prediction_path) != prediction.get("sha256"):
        raise PaperAdaptationAggregationError("source comparator prediction hash changed")
    calibrator_ref = _mapping(record.get("calibrator"), name="source comparator calibrator")
    calibrator_path = _resolve_file(
        calibrator_ref.get("path"), root=root, name="source comparator calibrator"
    )
    if sha256_file(calibrator_path) != calibrator_ref.get("file_sha256"):
        raise PaperAdaptationAggregationError("source comparator calibrator hash changed")
    calibrator_record, calibrator = load_source_temperature_calibrator_file(calibrator_path)
    if calibrator_record.get("record_sha256") != calibrator_ref.get("record_sha256"):
        raise PaperAdaptationAggregationError("source comparator calibrator record changed")
    with np.load(prediction_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {
            "logits",
            "probabilities",
            "labels",
            "participant_ids",
            "window_ids",
        }:
            raise PaperAdaptationAggregationError("source comparator prediction schema changed")
        logits = np.asarray(arrays["logits"], dtype=np.float64)
        probabilities = np.asarray(arrays["probabilities"], dtype=np.float64)
        labels = np.asarray(arrays["labels"], dtype=np.int64)
        participants = tuple(str(value) for value in arrays["participant_ids"].tolist())
        windows = tuple(str(value) for value in arrays["window_ids"].tolist())
    if (
        logits.shape != (EXPECTED_WINDOWS["source_validation"], len(CLASS_NAMES))
        or probabilities.shape != logits.shape
        or labels.shape != (EXPECTED_WINDOWS["source_validation"],)
        or len(participants) != labels.size
        or len(windows) != labels.size
        or len(set(windows)) != labels.size
        or set(participants) != set(SOURCE_VALIDATION_PARTICIPANTS)
        or not np.isfinite(logits).all()
        or not np.isfinite(probabilities).all()
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8, rtol=1e-8)
    ):
        raise PaperAdaptationAggregationError("source comparator predictions are misaligned")
    reconstructed = classification_report(
        labels, calibrator.probabilities(logits), participants, class_names=CLASS_NAMES
    )
    _participant_values(
        reconstructed, expected_participants=frozenset(SOURCE_VALIDATION_PARTICIPANTS)
    )
    if canonical_json_sha256(reconstructed) != canonical_json_sha256(
        _mapping(record.get("participant_level_report"), name="source comparator report")
    ):
        raise PaperAdaptationAggregationError("source comparator report does not reconstruct")
    return reconstructed


def _average_seed_reports(
    reports: Mapping[int, Mapping[str, Any]],
    *,
    expected_participants: frozenset[str],
) -> dict[str, Any]:
    if set(reports) != set(SEED_ORDER):
        raise PaperAdaptationAggregationError("model lacks the exact five-seed report matrix")
    by_seed = {
        seed: _participant_values(report, expected_participants=expected_participants)
        for seed, report in reports.items()
    }
    participant_macro = {
        participant: float(np.mean([by_seed[seed][participant]["macro_f1"] for seed in SEED_ORDER]))
        for participant in sorted(expected_participants, key=int)
    }
    participant_balanced = {
        participant: float(
            np.mean([by_seed[seed][participant]["balanced_accuracy"] for seed in SEED_ORDER])
        )
        for participant in sorted(expected_participants, key=int)
    }
    values = np.asarray(list(participant_macro.values()), dtype=np.float64)
    calibration_fields = ("negative_log_likelihood", "multiclass_brier_score", "ece")
    calibration = {
        field: float(
            np.mean(
                [
                    _finite(
                        _mapping(reports[seed].get("calibration"), name="calibration").get(field),
                        name=field,
                    )
                    for seed in SEED_ORDER
                ]
            )
        )
        for field in calibration_fields
    }
    return {
        "participant_macro_f1": participant_macro,
        "participant_balanced_accuracy": participant_balanced,
        "mean_participant_macro_f1": float(values.mean()),
        "worst_participant_macro_f1": float(values.min()),
        "lower_decile_participant_macro_f1": float(np.quantile(values, 0.1, method="linear")),
        "participant_bootstrap_interval": participant_bootstrap_interval(
            participant_macro, resamples=10_000, seed=1729
        ),
        "calibration_mean_across_seeds": calibration,
        "seed_order": list(SEED_ORDER),
        "seed_aggregation": "participant_metric_mean_across_five_seeds",
    }


def _comparison(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    reference_id: str,
) -> dict[str, Any]:
    reference_values = cast(Mapping[str, float], reference["participant_macro_f1"])
    candidate_values = cast(Mapping[str, float], candidate["participant_macro_f1"])
    differences = {
        participant: candidate_values[participant] - reference_values[participant]
        for participant in sorted(reference_values, key=int)
    }
    paired = paired_participant_comparison(
        reference_values,
        candidate_values,
        permutation_samples=100_000,
        seed=2718,
    )
    return {
        "reference_model_id": reference_id,
        "candidate_minus_reference": paired,
        "paired_difference_bootstrap_interval": participant_bootstrap_interval(
            differences, resamples=10_000, seed=1729
        ),
        "candidate_minus_reference_worst_participant_macro_f1": float(
            candidate["worst_participant_macro_f1"] - reference["worst_participant_macro_f1"]
        ),
        "candidate_minus_reference_lower_decile_participant_macro_f1": float(
            candidate["lower_decile_participant_macro_f1"]
            - reference["lower_decile_participant_macro_f1"]
        ),
        "improves_mean_and_worst_descriptively": bool(
            candidate["mean_participant_macro_f1"] > reference["mean_participant_macro_f1"]
            and candidate["worst_participant_macro_f1"] > reference["worst_participant_macro_f1"]
        ),
    }


def _apply_target_multiplicity(
    comparisons: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    """Apply one declared Holm family to all adapter/comparator target tests."""

    comparison_order = tuple(
        (method_id, reference_id) for method_id in METHOD_IDS for reference_id in COMPARATOR_IDS
    )
    if set(comparisons) != set(METHOD_IDS):
        raise PaperAdaptationAggregationError(
            "target comparison matrix lacks the exact adaptation family"
        )
    entries: list[dict[str, Any]] = []
    for method_id, reference_id in comparison_order:
        method_entries = comparisons.get(method_id)
        if not isinstance(method_entries, Mapping) or set(method_entries) != set(COMPARATOR_IDS):
            raise PaperAdaptationAggregationError(
                "target comparison matrix lacks the exact comparator family"
            )
        entry = method_entries.get(reference_id)
        if not isinstance(entry, dict):
            raise PaperAdaptationAggregationError("target comparison entry is not mutable")
        entries.append(entry)
    permutation_p = [
        _finite(
            _mapping(
                _mapping(entry.get("candidate_minus_reference"), name="paired comparison").get(
                    "permutation"
                ),
                name="permutation comparison",
            ).get("two_sided_p_value"),
            name="permutation p-value",
        )
        for entry in entries
    ]
    wilcoxon_p = [
        _finite(
            _mapping(
                _mapping(entry.get("candidate_minus_reference"), name="paired comparison").get(
                    "wilcoxon"
                ),
                name="Wilcoxon comparison",
            ).get("two_sided_p_value"),
            name="Wilcoxon p-value",
        )
        for entry in entries
    ]
    adjusted_permutation = holm_adjust(permutation_p)
    adjusted_wilcoxon = holm_adjust(wilcoxon_p)
    for index, entry in enumerate(entries):
        entry["holm_family"] = "four_adapter_vs_locked_comparator_target_comparisons"
        entry["holm_family_size"] = len(comparison_order)
        entry["holm_adjusted_permutation_p_value"] = adjusted_permutation[index]
        entry["holm_adjusted_wilcoxon_p_value"] = adjusted_wilcoxon[index]
    return comparison_order


def _target_alignment(
    context: ConsumedTargetContext,
    *,
    root: Path,
) -> tuple[tuple[str, ...], tuple[str, ...], NDArray[np.int64]]:
    reference = load_target_clean_reference(context, model_id="compact-erm", seed=SEED_ORDER[0])
    path = reference.prediction_path
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PaperAdaptationAggregationError(
            "opening-1 target array escapes repository_root"
        ) from exc
    if sha256_file(path) != reference.prediction_sha256:
        raise PaperAdaptationAggregationError("opening-1 target alignment hash changed")
    with np.load(path, allow_pickle=False) as arrays:
        windows = tuple(str(value) for value in arrays["window_ids"].tolist())
        participants = tuple(str(value) for value in arrays["participant_ids"].tolist())
        labels = np.asarray(arrays["true_labels"], dtype=np.int64)
    if (
        set(participants) != set(TARGET_PARTICIPANTS)
        or len(windows) != EXPECTED_WINDOWS["target_sealed"]
        or len(windows) != labels.size
        or len(set(windows)) != len(windows)
    ):
        raise PaperAdaptationAggregationError("opening-1 target alignment coverage changed")
    return windows, participants, labels


def _load_target_reference(
    entry: Mapping[str, Any],
    *,
    context: ConsumedTargetContext,
    root: Path,
    model_id: str,
    seed: int,
    expected_alignment: tuple[tuple[str, ...], tuple[str, ...], NDArray[np.int64]],
) -> Mapping[str, Any]:
    """Reconstruct one consumed comparator report without rerunning inference."""

    if (
        entry.get("model_id") != model_id
        or entry.get("seed") != seed
        or entry.get("cohort") != "target_sealed"
        or entry.get("evidence_status") != "locked_confirmatory_target_opening_1_consumed_reference"
    ):
        raise PaperAdaptationAggregationError("target comparator entry identity changed")
    reference = load_target_clean_reference(context, model_id=model_id, seed=seed)
    index_entry = context.result_entries[(model_id, seed)]
    if (
        entry.get("record_path") != reference.record_path.relative_to(root).as_posix()
        or entry.get("record_sha256") != reference.record_sha256
        or entry.get("record_file_sha256") != reference.record_file_sha256
        or entry.get("prediction_path") != index_entry.get("array_path")
        or entry.get("prediction_sha256") != reference.prediction_sha256
    ):
        raise PaperAdaptationAggregationError("target comparator lineage changed")
    prediction_path = _resolve_file(
        index_entry.get("array_path"),
        root=root,
        name="consumed comparator target prediction",
        allow_absolute=True,
    )
    if sha256_file(prediction_path) != reference.prediction_sha256:
        raise PaperAdaptationAggregationError("target comparator prediction hash changed")
    required_keys = {
        "evidence_status",
        "window_ids",
        "participant_ids",
        "true_labels",
        "logits",
        "uncalibrated_probabilities",
        "calibrated_probabilities",
        "predicted_labels",
    }
    with np.load(prediction_path, allow_pickle=False) as arrays:
        if set(arrays.files) != required_keys:
            raise PaperAdaptationAggregationError("target comparator prediction schema changed")
        windows = tuple(str(value) for value in arrays["window_ids"].tolist())
        participants = tuple(str(value) for value in arrays["participant_ids"].tolist())
        labels = np.asarray(arrays["true_labels"], dtype=np.int64)
        logits = np.asarray(arrays["logits"], dtype=np.float64)
        uncalibrated = np.asarray(arrays["uncalibrated_probabilities"], dtype=np.float64)
        calibrated = np.asarray(arrays["calibrated_probabilities"], dtype=np.float64)
        predictions = np.asarray(arrays["predicted_labels"], dtype=np.int64)
        stored_status = tuple(str(value) for value in arrays["evidence_status"].tolist())
    if (
        windows != expected_alignment[0]
        or participants != expected_alignment[1]
        or not np.array_equal(labels, expected_alignment[2])
        or logits.shape != (labels.size, len(CLASS_NAMES))
        or uncalibrated.shape != logits.shape
        or calibrated.shape != (labels.size, len(CLASS_NAMES))
        or not np.isfinite(logits).all()
        or not np.isfinite(uncalibrated).all()
        or not np.isfinite(calibrated).all()
        or not np.allclose(uncalibrated.sum(axis=1), 1.0, atol=1e-8, rtol=1e-8)
        or not np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-8, rtol=1e-8)
        or not np.array_equal(predictions, calibrated.argmax(axis=1))
        or stored_status != ("locked_confirmatory_target_opening_1",)
    ):
        raise PaperAdaptationAggregationError(
            "target comparator predictions differ from consumed opening-1 alignment"
        )
    reconstructed = classification_report(labels, calibrated, participants, class_names=CLASS_NAMES)
    _participant_values(reconstructed, expected_participants=frozenset(TARGET_PARTICIPANTS))
    if canonical_json_sha256(reconstructed) != canonical_json_sha256(
        reference.participant_level_report
    ):
        raise PaperAdaptationAggregationError("target comparator report does not reconstruct")
    return reconstructed


def _validate_lock_reference(
    value: Any,
    *,
    root: Path,
    name: str,
    expected_kind: str,
) -> Mapping[str, Any]:
    reference = _mapping(value, name=f"{name} reference")
    path = _resolve_file(reference.get("path"), root=root, name=name)
    if sha256_file(path) != reference.get("file_sha256"):
        raise PaperAdaptationAggregationError(f"{name} file hash changed")
    record = _mapping(load_json_strict(path), name=name)
    if (
        _self_hash(record, name=name) != reference.get("record_sha256")
        or record.get("record_kind") != expected_kind
    ):
        raise PaperAdaptationAggregationError(f"{name} contract changed")
    return record


def aggregate_paper_adaptation_extension(
    index_path: str | Path,
    *,
    repository_root: str | Path,
    destination: str | Path,
    csv_destination: str | Path,
    markdown_destination: str | Path,
    created_at_utc: str,
) -> dict[str, Any]:
    """Reconstruct all reports and produce participant-clustered statistics."""

    root_candidate = Path(repository_root)
    if root_candidate.is_symlink():
        raise PaperAdaptationAggregationError("repository_root may not be a symlink")
    root = root_candidate.resolve(strict=True)
    destination_path = _new_file(destination, root=root, name="aggregate JSON")
    csv_path = _new_file(csv_destination, root=root, name="aggregate CSV")
    markdown_path = _new_file(markdown_destination, root=root, name="aggregate Markdown")
    timestamp = require_utc_timestamp(created_at_utc, location="created_at_utc")
    source_index_path = _resolve_file(
        index_path, root=root, name="paper-adaptation index", allow_absolute=True
    )
    index = _mapping(load_json_strict(source_index_path), name="paper-adaptation index")
    index_hash = _self_hash(index, name="paper-adaptation index")
    required = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": "postconfirmatory_ccil_bpd_evaluation_index",
        "status": "complete_create_only",
        "evidence_status": EVIDENCE_STATUS,
        "track_role": TRACK_ROLE,
        "primary_claim_eligible": False,
        "designed_after_target_opening": True,
        "required_device": "cuda",
        "execution": "sequential_cuda_one_model_seed_at_a_time",
        "required_method_order": list(METHOD_IDS),
        "required_comparator_order": list(COMPARATOR_IDS),
        "required_seed_order": list(SEED_ORDER),
        "selection_candidate_count": len(METHOD_IDS) * 4,
        "source_result_count": len(METHOD_IDS) * len(SEED_ORDER),
        "target_result_count": len(METHOD_IDS) * len(SEED_ORDER),
        "comparator_source_reference_count": len(COMPARATOR_IDS) * len(SEED_ORDER),
        "comparator_target_reference_count": len(COMPARATOR_IDS) * len(SEED_ORDER),
        "source_stage_completed_before_target_signal_access": True,
        "target_signals_or_labels_accessed": True,
        "target_information_used_for_source_selection": False,
        "target_tuning_or_refit": False,
        "new_opening_or_unlock_invoked": False,
        "statistical_unit": "participant",
        "trial_safe": False,
        "interpretation": "post_confirmatory_descriptive_only",
        "failure": None,
    }
    mismatches = [key for key, expected in required.items() if index.get(key) != expected]
    if mismatches:
        raise PaperAdaptationAggregationError(f"paper-adaptation index changed: {mismatches}")
    config_ref = _mapping(index.get("experiment_config"), name="experiment config reference")
    config_path = _resolve_file(config_ref.get("path"), root=root, name="experiment config")
    if sha256_file(config_path) != config_ref.get("file_sha256"):
        raise PaperAdaptationAggregationError("experiment config file hash changed")
    config = load_paper_adaptation_config(config_path)
    if config.canonical_sha256 != config_ref.get("canonical_sha256"):
        raise PaperAdaptationAggregationError("experiment config canonical hash changed")
    selection_lock = _validate_lock_reference(
        index.get("source_selection_lock"),
        root=root,
        name="source selection lock",
        expected_kind="postconfirmatory_ccil_bpd_source_selection_lock",
    )
    source_lock = _validate_lock_reference(
        index.get("source_stage_lock"),
        root=root,
        name="source stage lock",
        expected_kind="postconfirmatory_ccil_bpd_source_stage_lock",
    )
    if (
        selection_lock.get("target_signal_arrays_loaded") is not False
        or selection_lock.get("target_labels_predictions_or_metrics_used") is not False
        or source_lock.get("target_signal_arrays_loaded") is not False
        or source_lock.get("target_labels_predictions_or_metrics_used_for_selection") is not False
    ):
        raise PaperAdaptationAggregationError("source locks do not preserve target blindness")
    opening = config.opening_1
    context = load_consumed_target_context(
        opening_receipt_path=cast(str, opening["opening_receipt_path"]),
        locked_target_index_path=cast(str, opening["locked_target_index_path"]),
        artifact_root=root,
    )
    if (
        context.receipt_file_sha256 != opening["opening_receipt_file_sha256"]
        or context.receipt.get("record_sha256") != opening["opening_receipt_record_sha256"]
        or context.index_file_sha256 != opening["locked_target_index_file_sha256"]
        or context.index.get("record_sha256") != opening["locked_target_index_record_sha256"]
    ):
        raise PaperAdaptationAggregationError("consumed opening-1 lineage changed")
    target_alignment = _target_alignment(context, root=root)

    source_values = _sequence(index.get("source_results"), name="source result entries")
    target_values = _sequence(index.get("target_results"), name="target result entries")
    expected_identities = {(method_id, seed) for method_id in METHOD_IDS for seed in SEED_ORDER}
    source_entries: dict[tuple[str, int], Mapping[str, Any]] = {}
    target_entries: dict[tuple[str, int], Mapping[str, Any]] = {}
    for values, destination_map, cohort in (
        (source_values, source_entries, "source_validation"),
        (target_values, target_entries, "target_sealed"),
    ):
        for value in values:
            entry = _mapping(value, name=f"{cohort} entry")
            method_value = entry.get("method_id")
            seed_value = entry.get("seed")
            if (
                not isinstance(method_value, str)
                or isinstance(seed_value, bool)
                or not isinstance(seed_value, int)
            ):
                raise PaperAdaptationAggregationError(f"{cohort} identity has invalid types")
            identity = (method_value, seed_value)
            if identity not in expected_identities or identity in destination_map:
                raise PaperAdaptationAggregationError(
                    f"{cohort} identity is unexpected or duplicated"
                )
            destination_map[identity] = entry
    if set(source_entries) != expected_identities or set(target_entries) != expected_identities:
        raise PaperAdaptationAggregationError("adapter method/seed matrix is incomplete")

    reports: dict[str, dict[str, dict[int, Mapping[str, Any]]]] = {
        "source_validation": {method_id: {} for method_id in (*METHOD_IDS, *COMPARATOR_IDS)},
        "target_sealed": {method_id: {} for method_id in (*METHOD_IDS, *COMPARATOR_IDS)},
    }
    for method_id, seed in sorted(expected_identities, key=lambda item: (item[0], item[1])):
        reports["source_validation"][method_id][seed] = _load_extension_entry(
            source_entries[(method_id, seed)],
            root=root,
            method_id=method_id,
            seed=seed,
            cohort="source_validation",
        )
        reports["target_sealed"][method_id][seed] = _load_extension_entry(
            target_entries[(method_id, seed)],
            root=root,
            method_id=method_id,
            seed=seed,
            cohort="target_sealed",
            expected_alignment=target_alignment,
        )

    source_reference_values = _sequence(
        index.get("frozen_source_comparator_references"),
        name="frozen source comparator references",
    )
    comparator_identities = {(model_id, seed) for model_id in COMPARATOR_IDS for seed in SEED_ORDER}
    source_reference_entries: dict[tuple[str, int], Mapping[str, Any]] = {}
    for value in source_reference_values:
        entry = _mapping(value, name="frozen source comparator reference")
        model_value = entry.get("model_id")
        seed_value = entry.get("seed")
        if (
            not isinstance(model_value, str)
            or isinstance(seed_value, bool)
            or not isinstance(seed_value, int)
            or (model_value, seed_value) not in comparator_identities
            or (model_value, seed_value) in source_reference_entries
        ):
            raise PaperAdaptationAggregationError("source comparator identity is invalid")
        source_reference_entries[(model_value, seed_value)] = entry
    if set(source_reference_entries) != comparator_identities:
        raise PaperAdaptationAggregationError("source comparator references are incomplete")
    for model_id, seed in sorted(comparator_identities):
        reports["source_validation"][model_id][seed] = _load_source_reference(
            source_reference_entries[(model_id, seed)],
            root=root,
            model_id=model_id,
            seed=seed,
        )

    target_reference_values = _sequence(
        index.get("consumed_opening_1_comparator_target_references"),
        name="consumed opening-1 comparator target references",
    )
    target_reference_entries: dict[tuple[str, int], Mapping[str, Any]] = {}
    for value in target_reference_values:
        entry = _mapping(value, name="consumed comparator target reference")
        model_value = entry.get("model_id")
        seed_value = entry.get("seed")
        if (
            not isinstance(model_value, str)
            or isinstance(seed_value, bool)
            or not isinstance(seed_value, int)
            or (model_value, seed_value) not in comparator_identities
            or (model_value, seed_value) in target_reference_entries
        ):
            raise PaperAdaptationAggregationError("target comparator identity is invalid")
        target_reference_entries[(model_value, seed_value)] = entry
    if set(target_reference_entries) != comparator_identities:
        raise PaperAdaptationAggregationError("target comparator references are incomplete")
    for model_id, seed in sorted(comparator_identities):
        reports["target_sealed"][model_id][seed] = _load_target_reference(
            target_reference_entries[(model_id, seed)],
            context=context,
            root=root,
            model_id=model_id,
            seed=seed,
            expected_alignment=target_alignment,
        )

    summaries: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, dict[str, Any]] = {}
    for cohort, participants in (
        ("source_validation", frozenset(SOURCE_VALIDATION_PARTICIPANTS)),
        ("target_sealed", frozenset(TARGET_PARTICIPANTS)),
    ):
        summaries[cohort] = {
            model_id: _average_seed_reports(model_reports, expected_participants=participants)
            for model_id, model_reports in reports[cohort].items()
        }
        comparisons[cohort] = {
            method_id: {
                reference_id: _comparison(
                    summaries[cohort][reference_id],
                    summaries[cohort][method_id],
                    reference_id=reference_id,
                )
                for reference_id in COMPARATOR_IDS
            }
            for method_id in METHOD_IDS
        }

    comparison_order = _apply_target_multiplicity(comparisons["target_sealed"])

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": "postconfirmatory_ccil_bpd_participant_statistics",
        "status": "complete_create_only",
        "created_at_utc": timestamp,
        "evidence_status": EVIDENCE_STATUS,
        "track_role": TRACK_ROLE,
        "primary_claim_eligible": False,
        "designed_after_target_opening": True,
        "input_index": {
            "path": source_index_path.relative_to(root).as_posix(),
            "file_sha256": sha256_file(source_index_path),
            "record_sha256": index_hash,
        },
        "experiment_config": {
            "path": config.path.relative_to(root).as_posix(),
            "file_sha256": config.file_sha256,
            "canonical_sha256": config.canonical_sha256,
        },
        "method_order": [*COMPARATOR_IDS, *METHOD_IDS],
        "seed_order": list(SEED_ORDER),
        "cohort_summaries": summaries,
        "paired_comparisons": comparisons,
        "target_multiplicity_family": {
            "comparison_order": [
                {"candidate": method_id, "reference": reference_id}
                for method_id, reference_id in comparison_order
            ],
            "family_size": len(comparison_order),
            "correction": ("Holm separately for paired permutation and Wilcoxon p-values"),
            "source_validation_comparisons": (
                "descriptive development diagnostics; no adjusted source p-values reported"
            ),
        },
        "statistical_unit": "participant",
        "window_independence_assumed": False,
        "target_tuning_or_refit": False,
        "new_opening_or_unlock_invoked": False,
        "interpretation": (
            "post-confirmatory descriptive comparison; significance values do not restore "
            "confirmatory status"
        ),
        "claim_labels": {
            method_id: next(
                str(method["claim_label"])
                for method in config.methods
                if method["model_id"] == method_id
            )
            for method_id in METHOD_IDS
        },
        "limitations": list(config.limitations),
        "forbidden_claims": list(config.forbidden_claims),
    }
    payload["record_sha256"] = canonical_json_sha256(payload)

    rows: list[dict[str, Any]] = []
    for cohort in ("source_validation", "target_sealed"):
        for model_id in (*COMPARATOR_IDS, *METHOD_IDS):
            summary = summaries[cohort][model_id]
            rows.append(
                {
                    "cohort": cohort,
                    "model_id": model_id,
                    "mean_participant_macro_f1": summary["mean_participant_macro_f1"],
                    "worst_participant_macro_f1": summary["worst_participant_macro_f1"],
                    "lower_decile_participant_macro_f1": summary[
                        "lower_decile_participant_macro_f1"
                    ],
                    "bootstrap_lower": summary["participant_bootstrap_interval"]["lower"],
                    "bootstrap_upper": summary["participant_bootstrap_interval"]["upper"],
                    "nll": summary["calibration_mean_across_seeds"]["negative_log_likelihood"],
                    "brier": summary["calibration_mean_across_seeds"]["multiclass_brier_score"],
                    "ece": summary["calibration_mean_across_seeds"]["ece"],
                    "evidence_status": EVIDENCE_STATUS,
                }
            )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Post-confirmatory CCIL/BPD paper-adaptation summary",
        "",
        "This is descriptive evidence designed after target opening 1; it is not confirmatory.",
        "",
        "| Cohort | Model | Mean participant macro-F1 | Worst | Lower decile | 95% CI |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {cohort} | {model_id} | {mean:.4f} | {worst:.4f} | {lower:.4f} | "
            "[{ci_low:.4f}, {ci_high:.4f}] |".format(
                cohort=row["cohort"],
                model_id=row["model_id"],
                mean=row["mean_participant_macro_f1"],
                worst=row["worst_participant_macro_f1"],
                lower=row["lower_decile_participant_macro_f1"],
                ci_low=row["bootstrap_lower"],
                ci_high=row["bootstrap_upper"],
            )
        )
    lines.extend(
        [
            "",
            "CCIL is a paper-derived loss adaptation, not official CCIL code. The BPD row is a "
            "boundary-safe local protocol adaptation, not an official-faithful BPD reproduction.",
            "Both adaptations are compared with locked compact ERM and locked MoRe-HAR full. "
            "Neither comparator is rerun; target references are consumed opening-1 artifacts. "
            "Holm correction covers all four adapter-versus-comparator target comparisons.",
            "",
        ]
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with markdown_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    # JSON is written last, so its presence certifies both tabular exports.
    atomic_write_json_new(payload, destination_path, allowed_root=root)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--csv-destination", type=Path, required=True)
    parser.add_argument("--markdown-destination", type=Path, required=True)
    parser.add_argument("--created-at-utc", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = aggregate_paper_adaptation_extension(
        arguments.index,
        repository_root=arguments.repository_root,
        destination=arguments.destination,
        csv_destination=arguments.csv_destination,
        markdown_destination=arguments.markdown_destination,
        created_at_utc=arguments.created_at_utc,
    )
    print(json.dumps({"status": result["status"], "record_sha256": result["record_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
