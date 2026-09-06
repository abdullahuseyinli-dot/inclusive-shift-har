"""Reproducible mechanism, cost and participant-distribution publication diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from inclusive_shift_har.artifacts.research_provenance import (
    _external_evidence_status,
    _git_state,
    _source_input_manifest,
    _typed_path_locator,
    _write_json_create_only,
)
from inclusive_shift_har.evaluation.inference_contracts import method_inference_contracts
from inclusive_shift_har.experiments.external_evidence_validate import validate_run_directory
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file

_VALIDATION_BOOLEAN_FIELDS = (
    "integrity_passed",
    "publication_evidence_ready",
    "publication_evidence_ready_for_unqualified_methods",
    "diagnostic_contract_passed",
)
_ACCEPTED_VALIDATION_MODES = {
    "VALIDATED": (True, True, False, False),
    "PARTIALLY_VALIDATED_METHODS": (True, False, True, False),
    "DIAGNOSTIC": (True, False, False, True),
}


def mechanism_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Count retained mechanisms, never infer activation from a winning score."""
    records = result.get("fold_records", [])
    nested = [fold for record in records for fold in record.get("folds", [])]
    summary: dict[str, Any] = {
        "interpretation": "descriptive mechanism evidence; activation is not an advancement gate",
        "recorded_outer_fold_count": len(nested),
    }
    if nested:
        routes = [fold["hera_v2_route_selection"] for fold in nested]
        summary["hera_v2_routing"] = {
            "enabled_fold_count": sum(route["enabled"] is True for route in routes),
            "recorded_fold_count": len(routes),
            "reason_counts": dict(
                sorted(
                    Counter(str(route.get("reason", "not_recorded")) for route in routes).items()
                )
            ),
            "evaluation_route_fraction": None,
            "fraction_note": "route enabled is recorded; it is not a count of changed evaluation decisions",
        }
        for name in ("selected_ctgr_candidate", "selected_cage_expert"):
            summary[name] = dict(sorted(Counter(str(fold[name]) for fold in nested).items()))
        for name in ("cage_evaluation_route_count", "hera_v1_strict_veto_count"):
            summary[name] = {
                "sum_across_all_seed_folds": sum(int(fold[name]) for fold in nested),
                "per_fold": [int(fold[name]) for fold in nested],
                "unit": "window-seed events; not independent participant N",
            }
        summary["physics_trusted_fraction_per_fold"] = [
            float(fold["physics_evaluation_trusted_fraction"]) for fold in nested
        ]
    methods: dict[str, Any] = {}
    for name in sorted(
        {str(row.get("method", row.get("model"))) for row in records if "folds" not in row}
    ):
        selected = [row for row in records if row.get("method", row.get("model")) == name]
        item: dict[str, Any] = {"recorded_fits": len(selected)}
        for key in (
            "parameter_count",
            "epochs_completed",
            "selected_epoch",
            "disable_cudnn",
            "mixed_precision",
        ):
            values = [row[key] for row in selected if key in row]
            if values:
                item[key] = values
        if any("mechanism" in row for row in selected):
            item["mechanism_by_fit"] = [row.get("mechanism") for row in selected]
            item["hierarchy_active_fit_count"] = sum(
                row.get("mechanism", {}).get("hierarchy_used") is True for row in selected
            )
        for key in ("fit_seconds", "prediction_seconds"):
            if all(key in row for row in selected):
                values = [float(row[key]) for row in selected]
                item[key] = {"sum": float(sum(values)), "per_fit": values}
        item["model_size_by_fit"] = [row["model_size"] for row in selected if "model_size" in row]
        item["checkpoint_bytes_by_fit"] = [
            row["checkpoint"]["size_bytes"]
            for row in selected
            if "size_bytes" in row.get("checkpoint", {})
        ]
        methods[name] = item
    summary["method_fit_records"] = methods
    summary["timing_limitation"] = (
        "Only explicitly measured fields are summarized. Local wall times include concurrent "
        "work and do not establish portable device latency, energy, or equal development compute."
    )
    summary["prospective_posture_advancement_gate"] = result.get("advancement_gate")
    summary["retained_legacy_gate_not_the_seed_averaged_primary"] = result.get(
        "development_advancement_gate"
    )
    summary["seed_averaged_comparisons_vs_xgboost_6ch"] = result.get(
        "primary_seed_averaged", {}
    ).get("comparisons_vs_xgboost_6ch")
    summary["baseline_reconstruction"] = result.get("baseline_reconstruction")
    return summary


def _checked_statistics(result: dict[str, Any]) -> dict[str, Any]:
    primary = result["primary_seed_averaged"]
    if result["seeds"] != [11, 23, 47] or not primary["methods"]:
        raise ValueError("figures require the complete fixed-seed primary evidence")
    for row in primary["methods"].values():
        values = np.asarray(list(row["participant_values"].values()), dtype=float)
        if (
            values.size != primary["participant_count"]
            or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 1))
            or not np.isclose(values.mean(), row["mean_participant_macro_f1"], rtol=0, atol=1e-12)
        ):
            raise ValueError("participant distribution is inconsistent with the primary mean")
    return cast(dict[str, Any], primary)


def _figure_method_label(name: str, contract: dict[str, Any]) -> str:
    count = contract["input_channel_count"]
    lane = "unknown channels" if count is None else f"{count}ch"
    qualifier = (
        "; annotation-selected diagnostic"
        if contract["annotation_selected_evaluation_context"]
        else "; participant batch"
        if contract["inference_unit"] == "noncausal_participant_batch"
        else ""
    )
    return name.replace("TinyHAR-", "TinyHAR-style-") + f" [{lane}{qualifier}]"


def write_distribution_figure(result: dict[str, Any], output: Path, *, status: str) -> list[Path]:
    """Show every frozen method and participant on a common 0--1 scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = _checked_statistics(result)
    names = sorted(primary["methods"])
    contracts = method_inference_contracts(result)
    paths = [output / "participant_distribution.svg", output / "participant_distribution.png"]
    if any(path.exists() for path in paths):
        raise FileExistsError("publication figures are create-only")
    with plt.rc_context({"svg.hashsalt": "external-har-session-grid-v3", "font.size": 10}):
        fig, axis = plt.subplots(figsize=(12, max(4.5, 0.43 * len(names) + 1.8)))
        for position, name in enumerate(names):
            row = primary["methods"][name]
            participants = sorted(row["participant_values"])
            values = [row["participant_values"][participant] for participant in participants]
            offsets = np.linspace(-0.11, 0.11, len(values))
            axis.scatter(values, position + offsets, s=14, color="#92999f", alpha=0.65, zorder=2)
            mean = row["mean_participant_macro_f1"]
            lo, hi = row["participant_bootstrap_95_percent_ci"]
            axis.errorbar(
                mean,
                position,
                xerr=[[mean - lo], [hi - mean]],
                fmt="s",
                color="#185a80",
                markersize=5,
                capsize=3,
                linewidth=1.4,
                zorder=3,
            )
        axis.set_yticks(
            range(len(names)),
            [_figure_method_label(name, contracts[name]) for name in names],
        )
        axis.invert_yaxis()
        axis.set_xlim(-0.02, 1.02)
        axis.set_xticks(np.linspace(0, 1, 6))
        axis.set_xlabel("Participant fixed-class macro-F1 (mean across seeds 11, 23, 47)")
        dataset = result.get("target_dataset", result.get("dataset", {}))
        axis.set_title(
            f"{dataset['dataset_id']} | N = {primary['participant_count']} | {status}",
            fontsize=11,
            pad=13,
        )
        axis.grid(axis="x", color="#d9dee2", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        fig.text(
            0.5,
            0.025,
            "Grey: each participant. Blue: mean and participant-bootstrap 95% interval.\n"
            "Labels state actual channel counts. Different input/context budgets are not matched comparisons.\n"
            "All methods shown; native and derived gravity remain separate. No SOTA or confirmation claim.",
            ha="center",
            fontsize=9,
        )
        # Reserve enough space for the three-line evidence qualification even
        # when a comparison group contains only one or two methods.
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.savefig(
            paths[0],
            metadata={"Date": None, "Creator": "InclusiveShift-HAR publication diagnostics"},
        )
        fig.savefig(
            paths[1], dpi=180, metadata={"Software": "InclusiveShift-HAR publication diagnostics"}
        )
        plt.close(fig)
    return paths


def _derived_evidence_status(result: dict[str, Any]) -> str:
    """Derive the run role from the exact frozen dataset summaries."""

    summaries = [
        value
        for name in ("source_dataset", "target_dataset", "dataset")
        if isinstance(value := result.get(name), dict)
    ]
    if not summaries:
        raise ValueError("diagnostic input lacks frozen dataset summaries")
    return _external_evidence_status(*summaries)


def _accepted_validation_mode(validation: dict[str, Any], evidence_status: str) -> str:
    """Require one exact current-result acceptance tuple and compatible role."""

    mode_value = validation.get("status")
    mode = mode_value if isinstance(mode_value, str) else ""
    expected = _ACCEPTED_VALIDATION_MODES.get(mode)
    if expected is None or any(
        validation.get(name) is not value
        for name, value in zip(_VALIDATION_BOOLEAN_FIELDS, expected, strict=True)
    ):
        raise ValueError(
            "diagnostic input is not accepted current-protocol evidence: "
            f"validator={mode_value!r}, frozen_evidence_status={evidence_status!r}"
        )
    expected_prefix = "diagnostic_" if mode == "DIAGNOSTIC" else "validated_"
    if not evidence_status.startswith(expected_prefix):
        raise ValueError(
            f"validator mode {mode!r} cannot promote evidence status {evidence_status!r}"
        )
    if mode == "DIAGNOSTIC":
        reasons = validation.get("diagnostic_scope_reasons")
        if not (
            isinstance(reasons, list)
            and reasons
            and all(isinstance(reason, str) and reason for reason in reasons)
            and len(reasons) == len(set(reasons))
        ):
            raise ValueError("diagnostic evidence requires explicit unique scope reasons")
    if mode == "PARTIALLY_VALIDATED_METHODS":
        unqualified = validation.get("unqualified_method_names")
        annotation_selected = validation.get("annotation_selected_context_methods")
        if not (
            isinstance(unqualified, list)
            and unqualified
            and all(isinstance(name, str) and name for name in unqualified)
            and len(unqualified) == len(set(unqualified))
            and isinstance(annotation_selected, list)
            and annotation_selected
            and all(isinstance(name, str) and name for name in annotation_selected)
            and len(annotation_selected) == len(set(annotation_selected))
            and set(unqualified).isdisjoint(annotation_selected)
        ):
            raise ValueError(
                "partially validated diagnostics require distinct non-empty unqualified "
                "and annotation-selected method sets"
            )
    return mode


def _bound_evidence_status(result: dict[str, Any], validation: dict[str, Any]) -> tuple[str, str]:
    """Bind canonical run status to both frozen data and independent validation."""

    derived = _derived_evidence_status(result)
    mode = _accepted_validation_mode(validation, derived)
    recorded = result.get("artifact_evidence_status")
    if not isinstance(recorded, str) or recorded != derived:
        raise ValueError(
            "diagnostic input artifact evidence status differs from its frozen dataset role"
        )
    return recorded, mode


def _runtime_qualification_records(directory: Path, repository_root: Path) -> list[dict[str, Any]]:
    """Bind enclosing runtime notices with typed, deterministic path locators."""

    notices: dict[Path, dict[str, Any]] = {}
    for parent in (directory, *directory.parents[:3]):
        for path in sorted(parent.glob("runtime_termination_failure*.json")):
            if path.is_symlink() or not path.is_file():
                raise ValueError(
                    f"runtime qualification must be a regular non-symlink file: {path}"
                )
            resolved = path.resolve(strict=True)
            notices[resolved] = {
                "locator": _typed_path_locator(resolved, repository_root),
                "sha256": sha256_file(resolved),
            }
        if (parent / "campaign_plan.json").is_file() or parent.name in {".audit", "results"}:
            break
    return [notices[path] for path in sorted(notices, key=lambda item: str(item))]


def write_diagnostics(directory: Path, output: Path, repository_root: Path) -> dict[str, Any]:
    validation = validate_run_directory(directory, repository_root)
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("diagnostic result artifact is not an object")
    status, validation_mode = _bound_evidence_status(result, validation)
    runtime_qualifications = _runtime_qualification_records(directory, repository_root)
    output.mkdir(parents=True, exist_ok=False)
    figure_status = status
    if validation_mode == "PARTIALLY_VALIDATED_METHODS":
        figure_status += " | contains annotation-selected diagnostic methods"
    if runtime_qualifications:
        figure_status += " | runtime-qualified"
    record: dict[str, Any] = {
        "schema_version": "2.0.0",
        "record_kind": "external_publication_diagnostics",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "evidence_status": status,
        "validation_mode": validation_mode,
        "evidence_status_binding": {
            "result_field": "artifact_evidence_status",
            "derived_from_frozen_dataset_summaries": True,
            "independent_validation_acceptance_tuple_passed": True,
        },
        "run_directory": _typed_path_locator(directory, repository_root),
        "result_sha256": sha256_file(directory / "result.json"),
        "data_audit_sha256": sha256_file(directory / "data_audit.json"),
        "prediction_sha256": sha256_file(directory / result["prediction_artifact"]["path"]),
        "validation": validation,
        "method_inference_contracts": method_inference_contracts(result),
        "mechanisms_and_recorded_cost": mechanism_summary(result),
        "primary_definition": result["primary_seed_averaged"]["estimand"],
        "run_started_at": result.get("started_at"),
        "run_completed_at": result.get("created_at"),
        "git_at_analysis": _git_state(repository_root),
        "analysis_source_manifest": _source_input_manifest(repository_root),
        "inference_status": "descriptive; no new test, model selection or prospective claim",
        "runtime_qualifications": runtime_qualifications,
        "runtime_qualification_present": bool(runtime_qualifications),
    }
    figures = write_distribution_figure(result, output, status=figure_status)
    record["figure_files"] = {path.name: sha256_file(path) for path in figures}
    record["record_sha256"] = canonical_json_sha256(record)
    _write_json_create_only(output / "diagnostics.json", record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    record = write_diagnostics(
        args.run_directory.resolve(), args.output.resolve(), args.repository_root.resolve()
    )
    print(
        json.dumps({"status": record["evidence_status"], "record_sha256": record["record_sha256"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
