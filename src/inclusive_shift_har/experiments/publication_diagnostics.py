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
    _git_state,
    _source_input_manifest,
    _write_json_create_only,
)
from inclusive_shift_har.evaluation.inference_contracts import method_inference_contracts
from inclusive_shift_har.experiments.external_evidence_validate import validate_run_directory
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file


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
        fig, axis = plt.subplots(figsize=(10, max(3.5, 0.43 * len(names) + 1.8)))
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
            [
                name.replace("TinyHAR-", "TinyHAR-style-")
                + (
                    " [annotation-selected diagnostic]"
                    if contracts[name]["annotation_selected_evaluation_context"]
                    else " [participant batch]"
                    if contracts[name]["inference_unit"] == "noncausal_participant_batch"
                    else ""
                )
                for name in names
            ],
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
            "All methods shown; channel suffixes identify distinct input lanes. No SOTA or confirmation claim.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        fig.savefig(
            paths[0],
            metadata={"Date": None, "Creator": "InclusiveShift-HAR publication diagnostics"},
        )
        fig.savefig(
            paths[1], dpi=180, metadata={"Software": "InclusiveShift-HAR publication diagnostics"}
        )
        plt.close(fig)
    return paths


def write_diagnostics(directory: Path, output: Path, repository_root: Path) -> dict[str, Any]:
    validation = validate_run_directory(directory, repository_root)
    if not validation["integrity_passed"]:
        raise ValueError("diagnostic input failed artifact integrity")
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    dataset = result.get("target_dataset", result.get("dataset", {}))
    oracle = dataset.get("dataset_id") == "sole_harmony_v1"
    if (
        not oracle
        and not validation["publication_evidence_ready"]
        and not validation.get("publication_evidence_ready_for_unqualified_methods", False)
    ):
        raise ValueError("non-oracle input failed the scientific evidence contract")
    output.mkdir(parents=True, exist_ok=False)
    status = (
        "oracle diagnostic"
        if oracle
        else "validated stress test"
        if dataset["dataset_id"] == "har_pmd_v1"
        else "validated development"
    )
    if validation.get("annotation_selected_context_methods"):
        status = "mixed validated and diagnostic methods"
    record: dict[str, Any] = {
        "record_kind": "external_publication_diagnostics",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "evidence_status": status,
        "run_directory": str(directory.resolve()),
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
    }
    notices: dict[str, str] = {}
    for parent in (directory, *directory.parents[:3]):
        notices.update(
            {
                str(path.resolve()): sha256_file(path)
                for path in parent.glob("runtime_termination_failure*.json")
            }
        )
        if (parent / "campaign_plan.json").is_file() or parent.name in {".audit", "results"}:
            break
    record["runtime_qualifications"] = notices
    if notices:
        status += "; runtime-qualified"
        record["evidence_status"] = status
    figures = write_distribution_figure(result, output, status=status)
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
