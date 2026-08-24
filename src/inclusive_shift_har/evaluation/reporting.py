"""Create publication tables from immutable locked-target evidence."""

from __future__ import annotations

import csv
import os
from collections.abc import Mapping, Sequence
from io import StringIO
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np

from inclusive_shift_har.evaluation.locked_target import LOCKED_TARGET_EVIDENCE_STATUS
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)

LOCKED_REPORT_SCHEMA_VERSION = "1.0.0"


class LockedReportError(RuntimeError):
    """Raised when publication reporting encounters changed or incomplete evidence."""


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LockedReportError(f"{name} must be an object")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise LockedReportError(f"{name} must be an array")
    return value


def _number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise LockedReportError(f"{name} must be a finite number")
    return float(value)


def _validate_self_hash(record: Mapping[str, Any], *, field: str, role: str) -> None:
    body = dict(record)
    claimed = body.pop(field, None)
    if claimed != canonical_json_sha256(body):
        raise LockedReportError(f"{role} self-hash does not validate")


def _resolve_file(value: Any, *, root: Path, role: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise LockedReportError(f"{role} path is missing")
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise LockedReportError(f"{role} escapes the output root") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise LockedReportError(f"{role} is missing, non-regular, or a symlink")
    return resolved


def _resolve_new(value: str | Path, *, root: Path, role: str) -> Path:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    candidate.parent.mkdir(parents=True, exist_ok=True)
    parent = candidate.parent.resolve(strict=True)
    try:
        parent.relative_to(root)
    except ValueError as exc:
        raise LockedReportError(f"{role} escapes the output root") from exc
    resolved = parent / candidate.name
    if os.path.lexists(resolved):
        raise FileExistsError(f"refusing to overwrite existing {role}: {resolved}")
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


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise LockedReportError("cannot average an empty metric sequence")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _publication_row(
    statistics_row: Mapping[str, Any],
    result_records: Sequence[Mapping[str, Any]],
    *,
    class_names: Sequence[str],
) -> dict[str, Any]:
    model_id = str(statistics_row.get("model_id", ""))
    participant = _mapping(
        statistics_row.get("participant_seed_averaged"), name=f"{model_id} participant summary"
    )
    interval = _mapping(
        participant.get("bootstrap_mean_macro_f1"), name=f"{model_id} bootstrap interval"
    )
    window_rows = [
        _mapping(
            _mapping(record.get("participant_level_report"), name="participant report").get(
                "window_level_diagnostics"
            ),
            name="window diagnostics",
        )
        for record in result_records
    ]
    report_rows = [
        _mapping(record.get("participant_level_report"), name="participant report")
        for record in result_records
    ]
    calibration_rows = [
        _mapping(report.get("calibration"), name="calibration") for report in report_rows
    ]
    selective_rows = [
        _mapping(report.get("selective_risk"), name="selective risk") for report in report_rows
    ]
    per_class = {
        class_name: _mean(
            [
                _number(
                    _mapping(row.get("per_class_recall"), name="per-class recall").get(class_name),
                    name=f"{model_id} {class_name} recall",
                )
                for row in window_rows
            ]
        )
        for class_name in class_names
    }
    row = {
        "model_id": model_id,
        "mean_participant_macro_f1": _number(
            participant.get("mean_macro_f1"), name=f"{model_id} participant mean"
        ),
        "macro_f1_ci_lower": _number(interval.get("lower"), name=f"{model_id} CI lower"),
        "macro_f1_ci_upper": _number(interval.get("upper"), name=f"{model_id} CI upper"),
        "worst_participant_macro_f1": _number(
            participant.get("worst_macro_f1"), name=f"{model_id} participant worst"
        ),
        "lower_decile_participant_macro_f1": _number(
            participant.get("lower_decile_macro_f1"), name=f"{model_id} participant lower decile"
        ),
        "mean_window_balanced_accuracy": _mean(
            [
                _number(row_value.get("balanced_accuracy"), name="balanced accuracy")
                for row_value in window_rows
            ]
        ),
        "mean_window_macro_f1": _mean(
            [
                _number(row_value.get("macro_f1"), name="window macro-F1")
                for row_value in window_rows
            ]
        ),
        "mean_negative_log_likelihood": _mean(
            [
                _number(row_value.get("negative_log_likelihood"), name="NLL")
                for row_value in calibration_rows
            ]
        ),
        "mean_multiclass_brier_score": _mean(
            [
                _number(row_value.get("multiclass_brier_score"), name="Brier score")
                for row_value in calibration_rows
            ]
        ),
        "mean_ece": _mean(
            [_number(row_value.get("ece"), name="ECE") for row_value in calibration_rows]
        ),
        "mean_aurc": _mean(
            [_number(row_value.get("aurc"), name="AURC") for row_value in selective_rows]
        ),
        "mean_per_class_recall": per_class,
        "evaluation_devices": sorted(
            {str(record.get("evaluation_device", "unknown")) for record in result_records}
        ),
        "seed_result_record_sha256s": [str(record["record_sha256"]) for record in result_records],
    }
    calibration_summary = _mapping(
        statistics_row.get("calibration_seed_mean"), name=f"{model_id} calibration summary"
    )
    checks = {
        "negative_log_likelihood": row["mean_negative_log_likelihood"],
        "multiclass_brier_score": row["mean_multiclass_brier_score"],
        "ece": row["mean_ece"],
    }
    for key, observed_value in checks.items():
        observed = cast(float, observed_value)
        expected = _number(calibration_summary.get(key), name=f"{model_id} stored {key}")
        if not np.isclose(observed, expected, atol=1e-12, rtol=1e-12):
            raise LockedReportError(f"{model_id} {key} differs from locked statistics")
    return row


def build_locked_target_publication_report(
    index_path: str | Path,
    statistics_path: str | Path,
    *,
    output_root: str | Path,
    destination: str | Path,
    model_table_csv: str | Path,
    model_table_markdown: str | Path,
    participant_table_csv: str | Path,
    comparison_table_csv: str | Path,
) -> dict[str, Any]:
    """Validate locked evidence and publish create-only machine/human-readable tables."""

    root = Path(output_root).resolve(strict=True)
    index_file = _resolve_file(index_path, root=root, role="locked target index")
    statistics_file = _resolve_file(statistics_path, root=root, role="locked target statistics")
    index = _mapping(load_json_strict(index_file), name="locked target index")
    statistics = _mapping(load_json_strict(statistics_file), name="locked target statistics")
    _validate_self_hash(index, field="record_sha256", role="locked target index")
    _validate_self_hash(statistics, field="record_sha256", role="locked target statistics")
    if statistics.get("locked_target_index_record_sha256") != index.get("record_sha256"):
        raise LockedReportError("statistics and target index lineage differ")
    if statistics.get("evidence_status") != LOCKED_TARGET_EVIDENCE_STATUS:
        raise LockedReportError("statistics evidence status is not locked confirmatory")
    if statistics.get("target_information_used_for_model_selection") is not False:
        raise LockedReportError("target information was marked as used for model selection")
    class_names = tuple(str(value) for value in _sequence(index.get("class_names"), name="classes"))
    seed_order = tuple(
        int(value) for value in _sequence(statistics.get("required_seed_order"), name="seeds")
    )

    records_by_model: dict[str, list[Mapping[str, Any]]] = {}
    index_entries = _sequence(index.get("results"), name="target result index")
    if len(index_entries) != index.get("model_seed_result_count"):
        raise LockedReportError("target result index count differs from its declaration")
    for entry_value in index_entries:
        entry = _mapping(entry_value, name="target index entry")
        record_file = _resolve_file(
            entry.get("record_path"), root=root, role="target result sidecar"
        )
        if sha256_file(record_file) != entry.get("record_file_sha256"):
            raise LockedReportError("target result sidecar file hash mismatch")
        record = _mapping(load_json_strict(record_file), name="target result sidecar")
        _validate_self_hash(record, field="record_sha256", role="target result sidecar")
        if record.get("record_sha256") != entry.get("record_sha256"):
            raise LockedReportError("target result sidecar self-hash differs from index")
        if record.get("evidence_status") != LOCKED_TARGET_EVIDENCE_STATUS:
            raise LockedReportError("target result sidecar evidence status is invalid")
        if record.get("target_information_used_for_model_selection") is not False:
            raise LockedReportError("a target sidecar marks target-based model selection")
        model_id = str(record.get("model_id", ""))
        records_by_model.setdefault(model_id, []).append(record)
    for model_id, records in records_by_model.items():
        records.sort(key=lambda record: seed_order.index(int(record["seed"])))
        if tuple(int(record["seed"]) for record in records) != seed_order:
            raise LockedReportError(f"{model_id} does not contain the exact seed order")

    statistics_rows = [
        _mapping(value, name="statistics model row")
        for value in _sequence(statistics.get("models"), name="statistics models")
    ]
    statistics_model_ids = {str(row.get("model_id", "")) for row in statistics_rows}
    if statistics_model_ids != set(records_by_model):
        raise LockedReportError("statistics and result sidecar model families differ")
    model_rows = [
        _publication_row(
            row,
            records_by_model[str(row["model_id"])],
            class_names=class_names,
        )
        for row in statistics_rows
    ]
    model_rows.sort(
        key=lambda row: (-cast(float, row["mean_participant_macro_f1"]), str(row["model_id"]))
    )

    destinations = {
        "model_table_csv": _resolve_new(model_table_csv, root=root, role="model table CSV"),
        "model_table_markdown": _resolve_new(
            model_table_markdown, root=root, role="model table Markdown"
        ),
        "participant_table_csv": _resolve_new(
            participant_table_csv, root=root, role="participant table CSV"
        ),
        "comparison_table_csv": _resolve_new(
            comparison_table_csv, root=root, role="comparison table CSV"
        ),
        "report": _resolve_new(destination, root=root, role="publication report"),
    }
    scalar_fields = [
        "model_id",
        "mean_participant_macro_f1",
        "macro_f1_ci_lower",
        "macro_f1_ci_upper",
        "worst_participant_macro_f1",
        "lower_decile_participant_macro_f1",
        "mean_window_balanced_accuracy",
        "mean_window_macro_f1",
        "mean_negative_log_likelihood",
        "mean_multiclass_brier_score",
        "mean_ece",
        "mean_aurc",
    ]
    csv_rows = [
        {
            **{field: row[field] for field in scalar_fields},
            **{
                f"mean_recall_{class_name}": cast(
                    Mapping[str, float], row["mean_per_class_recall"]
                )[class_name]
                for class_name in class_names
            },
        }
        for row in model_rows
    ]
    model_fieldnames = [*scalar_fields, *(f"mean_recall_{name}" for name in class_names)]
    _write_text_new(_csv_text(model_fieldnames, csv_rows), destinations["model_table_csv"])

    markdown_lines = [
        "# Locked zero-shot target results",
        "",
        "Status: **locked confirmatory target opening 1**. Participant-level macro-F1 is the "
        "primary endpoint; window metrics, calibration, and AURC are descriptive diagnostics.",
        "",
        "| Model | Mean participant macro-F1 (95% participant bootstrap CI) | Worst | Lower decile | Balanced accuracy | NLL | Brier | ECE | AURC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in model_rows:
        markdown_lines.append(
            "| `{model}` | {mean:.4f} [{lower:.4f}, {upper:.4f}] | {worst:.4f} | "
            "{decile:.4f} | {balanced:.4f} | {nll:.4f} | {brier:.4f} | {ece:.4f} | "
            "{aurc:.4f} |".format(
                model=row["model_id"],
                mean=row["mean_participant_macro_f1"],
                lower=row["macro_f1_ci_lower"],
                upper=row["macro_f1_ci_upper"],
                worst=row["worst_participant_macro_f1"],
                decile=row["lower_decile_participant_macro_f1"],
                balanced=row["mean_window_balanced_accuracy"],
                nll=row["mean_negative_log_likelihood"],
                brier=row["mean_multiclass_brier_score"],
                ece=row["mean_ece"],
                aurc=row["mean_aurc"],
            )
        )
    markdown_lines.extend(
        [
            "",
            "Metrics are averaged over the five predeclared seeds. The confidence interval "
            "resamples the ten target participants after seed averaging; windows are never "
            "treated as inferential replicates.",
            "",
        ]
    )
    _write_text_new("\n".join(markdown_lines), destinations["model_table_markdown"])

    participant_rows: list[dict[str, Any]] = []
    for statistics_row in statistics_rows:
        model_id = str(statistics_row["model_id"])
        participant = _mapping(
            statistics_row.get("participant_seed_averaged"), name="participant summary"
        )
        values = _mapping(participant.get("participant_values"), name="participant values")
        for participant_id in sorted(values, key=lambda value: (len(str(value)), str(value))):
            participant_rows.append(
                {
                    "model_id": model_id,
                    "participant_id": participant_id,
                    "seed_averaged_macro_f1": _number(
                        values[participant_id], name="participant macro-F1"
                    ),
                }
            )
    _write_text_new(
        _csv_text(["model_id", "participant_id", "seed_averaged_macro_f1"], participant_rows),
        destinations["participant_table_csv"],
    )

    comparisons = [
        _mapping(value, name="candidate comparison")
        for value in _sequence(statistics.get("candidate_comparisons"), name="comparisons")
    ]
    comparison_fields = [
        "reference_model_id",
        "participant_count",
        "candidate_minus_reference_mean",
        "candidate_minus_reference_median",
        "paired_standardized_effect",
        "rank_biserial_effect",
        "holm_adjusted_permutation_p_value",
        "holm_adjusted_wilcoxon_p_value",
    ]
    _write_text_new(
        _csv_text(
            comparison_fields,
            [
                {field: comparison.get(field) for field in comparison_fields}
                for comparison in comparisons
            ],
        ),
        destinations["comparison_table_csv"],
    )

    artifact_rows = {
        name: {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        }
        for name, path in destinations.items()
        if name != "report"
    }
    payload: dict[str, Any] = {
        "schema_version": LOCKED_REPORT_SCHEMA_VERSION,
        "record_kind": "locked_target_publication_report",
        "status": "complete_create_only_post_opening_derivation",
        "evidence_status": LOCKED_TARGET_EVIDENCE_STATUS,
        "target_opening_number": 1,
        "locked_target_index": {
            "path": index_file.relative_to(root).as_posix(),
            "file_sha256": sha256_file(index_file),
            "record_sha256": index["record_sha256"],
        },
        "locked_target_statistics": {
            "path": statistics_file.relative_to(root).as_posix(),
            "file_sha256": sha256_file(statistics_file),
            "record_sha256": statistics["record_sha256"],
        },
        "participant_count": statistics["participant_count"],
        "window_count": index.get("window_count"),
        "class_names": list(class_names),
        "required_seed_order": list(seed_order),
        "model_rows": model_rows,
        "candidate_model_id": statistics["candidate_model_id"],
        "candidate_comparisons": comparisons,
        "hypothesis_decision": statistics["hypothesis_decision"],
        "output_artifacts": artifact_rows,
        "metric_scope": {
            "primary": "participant macro-F1 averaged over seeds, then participants",
            "uncertainty": "participant-cluster bootstrap after seed averaging",
            "window_metrics": "descriptive seed means only",
            "ece": "secondary diagnostic",
            "aurc": "descriptive; no target-tuned abstention threshold",
        },
        "target_information_used_for_model_selection": False,
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    atomic_write_json_new(payload, destinations["report"], allowed_root=root)
    return payload
