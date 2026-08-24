"""Create-only participant aggregation for raw/total-acceleration sensitivity evidence."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Mapping, Sequence
from io import StringIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from inclusive_shift_har.evaluation._strict_config import require_utc_timestamp
from inclusive_shift_har.experiments.raw_total_acceleration import (
    EVIDENCE_STATUS,
    MODEL_IDS,
    SEED_ORDER,
    SOURCE_VALIDATION_PARTICIPANTS,
    TARGET_PARTICIPANTS,
    TRACK_ROLE,
    load_raw_total_sensitivity_config,
)
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)


class RawTotalAggregationError(RuntimeError):
    """Raised when sensitivity evidence is incomplete, mutated, or misaligned."""


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RawTotalAggregationError(f"{name} must be an object")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise RawTotalAggregationError(f"{name} must be an array")
    return value


def _string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RawTotalAggregationError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RawTotalAggregationError(f"{name} must be an integer")
    return int(value)


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RawTotalAggregationError(f"{name} must be finite")
    result = float(value)
    if not np.isfinite(result):
        raise RawTotalAggregationError(f"{name} must be finite")
    return result


def _self_hash(record: Mapping[str, Any], *, field: str, name: str) -> str:
    claimed = record.get(field)
    body = dict(record)
    body.pop(field, None)
    if not isinstance(claimed, str) or claimed != canonical_json_sha256(body):
        raise RawTotalAggregationError(f"{name} self-hash does not validate")
    return claimed


def _resolve_file(value: Any, *, root: Path, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise RawTotalAggregationError(f"{name} path must be a non-empty string or Path")
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    if candidate.is_symlink():
        raise RawTotalAggregationError(f"{name} may not be a symlink")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RawTotalAggregationError(f"{name} escapes artifact root") from exc
    if not resolved.is_file():
        raise RawTotalAggregationError(f"{name} is not a regular file")
    return resolved


def _resolve_new(value: str | Path, *, root: Path, name: str) -> Path:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    candidate.parent.mkdir(parents=True, exist_ok=True)
    parent = candidate.parent.resolve(strict=True)
    try:
        parent.relative_to(root)
    except ValueError as exc:
        raise RawTotalAggregationError(f"{name} escapes artifact root") from exc
    resolved = parent / candidate.name
    if os.path.lexists(resolved):
        raise FileExistsError(f"refusing to overwrite existing {name}: {resolved}")
    return resolved


def _write_text_new(text: str, destination: Path) -> None:
    temporary = destination.parent / f".{destination.name}.partial.{uuid4().hex}"
    try:
        with temporary.open("xb") as handle:
            handle.write(text.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    except Exception as exc:
        raise OSError(
            f"atomic text publication failed; partial retained at {temporary}: {exc}"
        ) from exc
    else:
        temporary.unlink()


def _csv_text(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _load_record_entry(
    entry: Mapping[str, Any],
    *,
    root: Path,
    expected_kind: str,
) -> Mapping[str, Any]:
    path = _resolve_file(entry.get("record_path"), root=root, name="indexed result record")
    if sha256_file(path) != entry.get("record_file_sha256"):
        raise RawTotalAggregationError("indexed result record file hash changed")
    record = _mapping(load_json_strict(path), name="indexed result record")
    record_hash = _self_hash(record, field="record_sha256", name="indexed result record")
    if record_hash != entry.get("record_sha256") or record.get("record_kind") != expected_kind:
        raise RawTotalAggregationError("indexed result record lineage changed")
    prediction_path_value = entry.get("prediction_path")
    if prediction_path_value is not None:
        prediction_path = _resolve_file(
            prediction_path_value, root=root, name="indexed prediction artifact"
        )
        if sha256_file(prediction_path) != entry.get("prediction_sha256"):
            raise RawTotalAggregationError("indexed prediction artifact hash changed")
    return record


def _participant_values(
    record: Mapping[str, Any],
    *,
    expected_participants: tuple[str, ...],
    name: str,
) -> tuple[dict[str, float], Mapping[str, Any]]:
    report = _mapping(record.get("participant_level_report"), name=f"{name} report")
    rows = _sequence(report.get("participants"), name=f"{name} participant rows")
    values: dict[str, float] = {}
    for value in rows:
        row = _mapping(value, name=f"{name} participant row")
        participant = _string(row.get("participant_id"), name=f"{name} participant ID")
        if participant in values:
            raise RawTotalAggregationError(f"{name} duplicates participant {participant}")
        values[participant] = _finite(row.get("macro_f1"), name=f"{name} macro-F1")
    if set(values) != set(expected_participants) or len(values) != len(expected_participants):
        raise RawTotalAggregationError(f"{name} participant set differs from the fixed cohort")
    ordered = {participant: values[participant] for participant in expected_participants}
    array = np.asarray(list(ordered.values()), dtype=np.float64)
    primary = _mapping(report.get("primary"), name=f"{name} primary summary")
    reconstructed = {
        "mean_participant_macro_f1": float(array.mean()),
        "worst_participant_macro_f1": float(array.min()),
        "lower_decile_participant_macro_f1": float(np.quantile(array, 0.1, method="linear")),
    }
    for field, observed in reconstructed.items():
        stored = _finite(primary.get(field), name=f"{name} {field}")
        if not np.isclose(observed, stored, atol=1e-12, rtol=1e-12):
            raise RawTotalAggregationError(f"{name} {field} does not reconstruct")
    return ordered, report


def _summary(values: Mapping[str, float]) -> dict[str, Any]:
    array = np.asarray(list(values.values()), dtype=np.float64)
    return {
        "participant_values": dict(values),
        "participant_count": len(values),
        "mean_macro_f1": float(array.mean()),
        "worst_macro_f1": float(array.min()),
        "lower_decile_macro_f1": float(np.quantile(array, 0.1, method="linear")),
    }


def _seed_average(
    seed_values: Mapping[tuple[str, int], Mapping[str, float]],
    *,
    model_id: str,
    participants: tuple[str, ...],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for participant in participants:
        result[participant] = float(
            np.mean(
                np.asarray(
                    [seed_values[(model_id, seed)][participant] for seed in SEED_ORDER],
                    dtype=np.float64,
                )
            )
        )
    return result


def _markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Raw/total-acceleration signal-definition sensitivity",
        "",
        "Status: **post-confirmatory exploratory analysis**. Both architectures were fixed "
        "independently of the observed ranking. This is not a new confirmatory opening, an exact "
        "UCI signal match, trial-safe evidence, or a causal disability analysis.",
        "",
        "| Model | Raw source mean | Raw target mean | Primary target mean | Raw - primary mean delta (95% paired-participant bootstrap CI) | Raw target worst | Raw target lower decile |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        interval = _mapping(row["paired_bootstrap_mean_delta"], name="bootstrap interval")
        lines.append(
            "| `{model}` | {source:.4f} | {raw:.4f} | {primary:.4f} | {delta:.4f} "
            "[{lower:.4f}, {upper:.4f}] | {worst:.4f} | {decile:.4f} |".format(
                model=row["model_id"],
                source=row["raw_source_validation"]["mean_macro_f1"],
                raw=row["raw_target"]["mean_macro_f1"],
                primary=row["primary_target"]["mean_macro_f1"],
                delta=row["raw_minus_primary_target"]["mean_macro_f1"],
                lower=interval["lower"],
                upper=interval["upper"],
                worst=row["raw_target"]["worst_macro_f1"],
                decile=row["raw_target"]["lower_decile_macro_f1"],
            )
        )
    lines.extend(
        [
            "",
            "Participants, not windows, are the statistical units. Participant metrics are "
            "averaged over the five fixed seeds before the paired target bootstrap (10,000 "
            "resamples; seed 1729). Source normalization and temperature calibration use only "
            "the source training and source validation partitions, respectively.",
            "",
            "The raw/total accelerometer includes gravity and is not equivalent to UCI body "
            "acceleration. InclusiveHAR trial boundaries remain unrecoverable, so this track is "
            "not trial-safe.",
            "",
        ]
    )
    return "\n".join(lines)


def aggregate_raw_total_acceleration_sensitivity(
    index_path: str | Path,
    *,
    artifact_root: str | Path,
    destination: str | Path,
    csv_destination: str | Path,
    markdown_destination: str | Path,
    created_at_utc: str,
) -> dict[str, Any]:
    """Validate all 30 records and publish participant-level sensitivity tables."""

    root = Path(artifact_root).resolve(strict=True)
    timestamp = require_utc_timestamp(created_at_utc, location="created_at_utc")
    index_file = _resolve_file(index_path, root=root, name="raw/total sensitivity index")
    index = _mapping(load_json_strict(index_file), name="raw/total sensitivity index")
    index_hash = _self_hash(index, field="record_sha256", name="sensitivity index")
    required = {
        "record_kind": "postconfirmatory_raw_total_acceleration_sensitivity_index",
        "status": "complete_create_only",
        "evidence_status": EVIDENCE_STATUS,
        "track_role": TRACK_ROLE,
        "primary_claim_eligible": False,
        "model_selection_use": False,
        "required_model_order": list(MODEL_IDS),
        "required_seed_order": list(SEED_ORDER),
        "expected_model_seed_count": 10,
        "source_result_count": 10,
        "target_result_count": 10,
        "primary_reference_count": 10,
        "failure_count": 0,
        "source_stage_completed_before_target_access": True,
        "target_signals_or_labels_accessed": True,
        "target_tuning": False,
        "target_calibration_or_refit": False,
        "new_opening_or_unlock_invoked": False,
    }
    mismatches = [key for key, expected in required.items() if index.get(key) != expected]
    if mismatches:
        raise RawTotalAggregationError(f"sensitivity index contract mismatch: {mismatches}")
    if _sequence(index.get("failures"), name="failures"):
        raise RawTotalAggregationError("complete sensitivity index contains failures")

    config_entry = _mapping(index.get("experiment_config"), name="experiment config entry")
    config_path = _resolve_file(config_entry.get("path"), root=root, name="experiment config")
    if sha256_file(config_path) != config_entry.get("file_sha256"):
        raise RawTotalAggregationError("experiment config hash changed")
    config = load_raw_total_sensitivity_config(config_path)
    if config.file_sha256 != config_entry.get("file_sha256"):
        raise RawTotalAggregationError("strict experiment configuration differs from index")
    for field, name in (
        ("normalization", "normalization record"),
        ("source_stage_lock", "source stage lock"),
    ):
        entry = _mapping(index.get(field), name=name)
        path = _resolve_file(entry.get("path"), root=root, name=name)
        if sha256_file(path) != entry.get("file_sha256"):
            raise RawTotalAggregationError(f"{name} file hash changed")
        record = _mapping(load_json_strict(path), name=name)
        if _self_hash(record, field="record_sha256", name=name) != entry.get("record_sha256"):
            raise RawTotalAggregationError(f"{name} record hash changed")

    raw_source: dict[tuple[str, int], Mapping[str, float]] = {}
    raw_target: dict[tuple[str, int], Mapping[str, float]] = {}
    primary_target: dict[tuple[str, int], Mapping[str, float]] = {}
    seed_rows: list[dict[str, Any]] = []
    source_entries = _sequence(index.get("source_results"), name="source results")
    target_entries = _sequence(index.get("target_results"), name="target results")
    reference_entries = _sequence(
        index.get("primary_target_references"), name="primary target references"
    )
    entry_sets = (
        (source_entries, "source_validation", raw_source, SOURCE_VALIDATION_PARTICIPANTS),
        (target_entries, "target_consumed_opening_1", raw_target, TARGET_PARTICIPANTS),
    )
    for entries, cohort, destination_map, participants in entry_sets:
        for value in entries:
            entry = _mapping(value, name=f"{cohort} entry")
            model_id = _string(entry.get("model_id"), name="model ID")
            seed = _integer(entry.get("seed"), name="seed")
            identity = (model_id, seed)
            if identity in destination_map:
                raise RawTotalAggregationError(f"duplicate {cohort} model/seed")
            record = _load_record_entry(
                entry,
                root=root,
                expected_kind="postconfirmatory_raw_total_acceleration_evaluation",
            )
            record_required = {
                "evidence_status": EVIDENCE_STATUS,
                "track_role": TRACK_ROLE,
                "primary_claim_eligible": False,
                "model_selection_use": False,
                "model_id": model_id,
                "seed": seed,
                "cohort": cohort,
                "target_tuning": False,
                "target_calibration_or_refit": False,
                "new_opening_or_unlock_invoked": False,
            }
            if any(record.get(key) != expected for key, expected in record_required.items()):
                raise RawTotalAggregationError(f"{cohort} result lineage changed")
            participant_values, report = _participant_values(
                record,
                expected_participants=participants,
                name=f"{model_id} seed {seed} {cohort}",
            )
            destination_map[identity] = participant_values
            calibration = _mapping(report.get("calibration"), name="calibration report")
            seed_rows.append(
                {
                    "model_id": model_id,
                    "seed": seed,
                    "cohort": cohort,
                    "mean_participant_macro_f1": float(
                        np.mean(np.asarray(list(participant_values.values())))
                    ),
                    "negative_log_likelihood": _finite(
                        calibration.get("negative_log_likelihood"), name="NLL"
                    ),
                    "multiclass_brier_score": _finite(
                        calibration.get("multiclass_brier_score"), name="Brier score"
                    ),
                    "ece": _finite(calibration.get("ece"), name="ECE"),
                    "record_sha256": record["record_sha256"],
                }
            )

    for value in reference_entries:
        entry = _mapping(value, name="primary reference entry")
        model_id = _string(entry.get("model_id"), name="primary model ID")
        seed = _integer(entry.get("seed"), name="primary seed")
        identity = (model_id, seed)
        if identity in primary_target:
            raise RawTotalAggregationError("duplicate primary target reference")
        record = _load_record_entry(entry, root=root, expected_kind="locked_target_per_seed_result")
        if (
            record.get("model_id") != model_id
            or record.get("seed") != seed
            or record.get("evidence_status") != "locked_confirmatory_target_opening_1"
            or record.get("target_information_used_for_model_selection") is not False
        ):
            raise RawTotalAggregationError("primary target reference lineage changed")
        values, _ = _participant_values(
            record,
            expected_participants=TARGET_PARTICIPANTS,
            name=f"{model_id} seed {seed} primary target",
        )
        primary_target[identity] = values

    expected_identities = {(model, seed) for model in MODEL_IDS for seed in SEED_ORDER}
    if (
        set(raw_source) != expected_identities
        or set(raw_target) != expected_identities
        or set(primary_target) != expected_identities
    ):
        raise RawTotalAggregationError("model/seed coverage is not exactly 2 x 5")

    rng = np.random.default_rng(1729)
    bootstrap_indices = rng.integers(
        0, len(TARGET_PARTICIPANTS), size=(10000, len(TARGET_PARTICIPANTS))
    )
    model_rows: list[dict[str, Any]] = []
    participant_rows: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        source_values = _seed_average(
            raw_source, model_id=model_id, participants=SOURCE_VALIDATION_PARTICIPANTS
        )
        raw_values = _seed_average(raw_target, model_id=model_id, participants=TARGET_PARTICIPANTS)
        primary_values = _seed_average(
            primary_target, model_id=model_id, participants=TARGET_PARTICIPANTS
        )
        deltas = {
            participant: raw_values[participant] - primary_values[participant]
            for participant in TARGET_PARTICIPANTS
        }
        source_summary = _summary(source_values)
        raw_summary = _summary(raw_values)
        primary_summary = _summary(primary_values)
        delta_array = np.asarray(list(deltas.values()), dtype=np.float64)
        bootstrap = np.mean(delta_array[bootstrap_indices], axis=1)
        interval = {
            "method": "paired_participant_cluster_percentile",
            "confidence": 0.95,
            "estimate": float(delta_array.mean()),
            "lower": float(np.quantile(bootstrap, 0.025, method="linear")),
            "upper": float(np.quantile(bootstrap, 0.975, method="linear")),
            "resamples": 10000,
            "seed": 1729,
            "cluster_unit": "participant",
            "indices_shared_across_models": True,
        }
        model_rows.append(
            {
                "model_id": model_id,
                "seed_order": list(SEED_ORDER),
                "raw_source_validation": source_summary,
                "raw_target": raw_summary,
                "primary_target": primary_summary,
                "raw_minus_primary_target": {
                    "participant_values": deltas,
                    "mean_macro_f1": float(delta_array.mean()),
                    "worst_macro_f1": float(
                        raw_summary["worst_macro_f1"] - primary_summary["worst_macro_f1"]
                    ),
                    "lower_decile_macro_f1": float(
                        raw_summary["lower_decile_macro_f1"]
                        - primary_summary["lower_decile_macro_f1"]
                    ),
                },
                "raw_source_minus_raw_target_mean_macro_f1": float(
                    source_summary["mean_macro_f1"] - raw_summary["mean_macro_f1"]
                ),
                "paired_bootstrap_mean_delta": interval,
            }
        )
        for participant in TARGET_PARTICIPANTS:
            participant_rows.append(
                {
                    "model_id": model_id,
                    "participant_id": participant,
                    "raw_total_target_macro_f1_seed_mean": raw_values[participant],
                    "primary_user_acceleration_target_macro_f1_seed_mean": primary_values[
                        participant
                    ],
                    "raw_minus_primary_macro_f1": deltas[participant],
                }
            )

    destinations = {
        "csv": _resolve_new(csv_destination, root=root, name="raw/total summary CSV"),
        "markdown": _resolve_new(
            markdown_destination, root=root, name="raw/total summary Markdown"
        ),
        "json": _resolve_new(destination, root=root, name="raw/total aggregate JSON"),
    }
    if len(set(destinations.values())) != 3:
        raise RawTotalAggregationError("aggregate output destinations must be distinct")
    csv_fields = [
        "model_id",
        "raw_source_mean_macro_f1",
        "raw_target_mean_macro_f1",
        "primary_target_mean_macro_f1",
        "raw_minus_primary_target_mean_macro_f1",
        "paired_bootstrap_ci_lower",
        "paired_bootstrap_ci_upper",
        "raw_target_worst_macro_f1",
        "raw_target_lower_decile_macro_f1",
        "raw_source_minus_raw_target_mean_macro_f1",
    ]
    csv_rows: list[dict[str, Any]] = []
    for row in model_rows:
        csv_interval = _mapping(row["paired_bootstrap_mean_delta"], name="bootstrap interval")
        csv_rows.append(
            {
                "model_id": row["model_id"],
                "raw_source_mean_macro_f1": row["raw_source_validation"]["mean_macro_f1"],
                "raw_target_mean_macro_f1": row["raw_target"]["mean_macro_f1"],
                "primary_target_mean_macro_f1": row["primary_target"]["mean_macro_f1"],
                "raw_minus_primary_target_mean_macro_f1": row["raw_minus_primary_target"][
                    "mean_macro_f1"
                ],
                "paired_bootstrap_ci_lower": csv_interval["lower"],
                "paired_bootstrap_ci_upper": csv_interval["upper"],
                "raw_target_worst_macro_f1": row["raw_target"]["worst_macro_f1"],
                "raw_target_lower_decile_macro_f1": row["raw_target"]["lower_decile_macro_f1"],
                "raw_source_minus_raw_target_mean_macro_f1": row[
                    "raw_source_minus_raw_target_mean_macro_f1"
                ],
            }
        )
    _write_text_new(_csv_text(csv_fields, csv_rows), destinations["csv"])
    _write_text_new(_markdown(model_rows), destinations["markdown"])
    output_artifacts = {
        role: {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
        for role, path in destinations.items()
        if role != "json"
    }
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "postconfirmatory_raw_total_acceleration_participant_aggregate",
        "status": "complete_create_only",
        "created_at_utc": timestamp,
        "evidence_status": EVIDENCE_STATUS,
        "track_role": TRACK_ROLE,
        "primary_claim_eligible": False,
        "model_selection_use": False,
        "sensitivity_index": {
            "path": index_file.relative_to(root).as_posix(),
            "file_sha256": sha256_file(index_file),
            "record_sha256": index_hash,
        },
        "model_rows": model_rows,
        "participant_target_rows": participant_rows,
        "seed_rows": sorted(
            seed_rows, key=lambda row: (str(row["model_id"]), str(row["cohort"]), int(row["seed"]))
        ),
        "output_artifacts": output_artifacts,
        "statistical_unit": "participant",
        "seed_handling": config.aggregation["seed_handling"],
        "paired_bootstrap": {
            "resamples": 10000,
            "seed": 1729,
            "confidence": 0.95,
            "indices_shared_across_models": True,
        },
        "target_tuning_calibration_or_refit": False,
        "new_opening_or_unlock_invoked": False,
        "uci_body_acceleration_equivalent": False,
        "trial_safe": False,
        "interpretation": "post_confirmatory_exploratory_not_confirmatory_or_causal",
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    atomic_write_json_new(payload, destinations["json"], allowed_root=root)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path.cwd())
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--csv-destination", type=Path, required=True)
    parser.add_argument("--markdown-destination", type=Path, required=True)
    parser.add_argument("--created-at-utc", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = aggregate_raw_total_acceleration_sensitivity(
        args.index,
        artifact_root=args.artifact_root,
        destination=args.destination,
        csv_destination=args.csv_destination,
        markdown_destination=args.markdown_destination,
        created_at_utc=args.created_at_utc,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
