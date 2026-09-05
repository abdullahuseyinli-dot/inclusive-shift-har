"""Reconstruct a matched external table from validated, immutable predictions."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from inclusive_shift_har.artifacts.research_provenance import _write_json_create_only
from inclusive_shift_har.evaluation.external_statistics import (
    ParticipantMetricInputs,
    seed_evidence,
)
from inclusive_shift_har.evaluation.inference_contracts import method_inference_contracts
from inclusive_shift_har.experiments.external_evidence_validate import validate_run_directory
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file

_IDENTITY_ARRAYS = ("labels", "participant_ids", "session_ids", "trial_ids", "window_ids")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object: {path}")
    return value


def _receipt_contract(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        name: sorted(
            (
                str(item["dataset_id"]),
                str(item["locator"]),
                str(item.get("member")),
                str(item["computed_sha256"]),
            )
            for item in items
        )
        for name, items in audit.items()
        if name.endswith("receipts") and isinstance(items, list)
    }


def _preprocessing_contract(result: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    """Prefer complete observed FoG tensors/grids to an unrelated-loader file hash.

    The entire dataset audit, raw receipts and prediction identities must also
    match. This is numerical equivalence for the observed benchmark only, not a
    claim that two arbitrary preprocessing implementations are interchangeable.
    Transfer and loaders without materialized segment witnesses remain strict.
    """
    segments = dataset.get("preprocessing_audit", [])
    if (
        dataset.get("dataset_id") == "fog_star_v3"
        and "target_dataset" not in result
        and isinstance(segments, list)
        and segments
    ):
        for segment in segments:
            if not isinstance(segment, dict) or any(
                not isinstance(digest := segment.get(name), str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for name in (
                    "signals_sha256",
                    "gravity_sha256",
                    "source_timestamps_sha256",
                    "timestamps_sha256",
                    "candidate_grid_sha256",
                )
            ):
                raise ValueError("non-comparable incomplete FoG materialization witness")
        return {
            "equivalence_basis": "complete_observed_fog_segment_tensors_and_grids",
            "segment_audit_sha256": canonical_json_sha256(segments),
            "scope": "observed benchmark only; not arbitrary unseen inputs",
        }
    return {
        "equivalence_basis": "identical_preprocessing_source_file",
        "source_sha256": result["source_input_manifest"]["files"][
            "src/inclusive_shift_har/data/external_har.py"
        ],
    }


def _strongest_control_comparisons(
    statistics: dict[str, Any], inference_contracts: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    methods = statistics["methods"]
    controls = [
        name
        for name in methods
        if name.startswith(("RandomForest-", "XGBoost-", "DeepConvLSTM-", "TinyHAR-"))
    ]
    if not controls:
        return {"status": "no_applicable_control_in_table"}
    strongest = max(sorted(controls), key=lambda name: methods[name]["mean_participant_macro_f1"])
    base = np.asarray(list(methods[strongest]["participant_values"].values()))
    draws = np.random.default_rng(20260905).integers(0, base.size, size=(10000, base.size))
    pairs = {}
    tail = max(1, int(np.ceil(0.30 * base.size)))
    for name, method in methods.items():
        candidate = np.asarray(list(method["participant_values"].values()))
        delta = candidate - base
        pairs[name] = {
            "input_or_inference_contract_differences": [
                key
                for key in ("input_channel_count", "inference_unit", "context_population")
                if inference_contracts[name][key] != inference_contracts[strongest][key]
            ],
            "annotation_selected_context_diagnostic": inference_contracts[name][
                "annotation_selected_evaluation_context"
            ],
            "mean_difference": float(delta.mean()),
            "paired_participant_bootstrap_95_percent_ci": np.quantile(
                delta[draws].mean(axis=1), [0.025, 0.975]
            ).tolist(),
            "bottom_30_percent_difference_95_percent_ci": np.quantile(
                np.sort(candidate[draws], axis=1)[:, :tail].mean(axis=1)
                - np.sort(base[draws], axis=1)[:, :tail].mean(axis=1),
                [0.025, 0.975],
            ).tolist(),
            "rescue_count": int(np.sum(delta > 1e-12)),
            "harm_count": int(np.sum(delta < -1e-12)),
            "tie_count": int(np.sum(np.abs(delta) <= 1e-12)),
        }
    return {
        "control": strongest,
        "selection": "largest observed primary mean among frozen applicable controls; lexical tie break",
        "inference_status": "descriptive post-selection comparisons; no new primary test or superiority gate",
        "pairs": pairs,
    }


def reconstruct_matched_table(run_directories: list[Path], repository_root: Path) -> dict[str, Any]:
    """Reject mismatched datasets/grids/roles before pooling distinct methods.

    No model is fit and no source signal is opened. Neural/classical supervision
    allocation may differ as frozen by method, but outer participants, raw-source
    hashes, ontology, preprocessing, window identity and seeds must match exactly.
    Native/derived gravity lanes and transfer/within-dataset endpoints never mix.
    """
    if not run_directories:
        raise ValueError("at least one validated run is required")
    common: dict[str, Any] | None = None
    identity: dict[str, Any] = {}
    merged: dict[int, dict[str, Any]] = {seed: {} for seed in (11, 23, 47)}
    sources: list[dict[str, Any]] = []
    qualifications: dict[str, str] = {}
    inference_contracts: dict[str, dict[str, Any]] = {}
    method_statuses: dict[str, str] = {}
    for directory in run_directories:
        validation = validate_run_directory(directory, repository_root)
        if not validation["publication_evidence_ready"] and not validation.get(
            "publication_evidence_ready_for_unqualified_methods", False
        ):
            raise ValueError(f"run is not validated current-protocol evidence: {directory}")
        result, audit = _read(directory / "result.json"), _read(directory / "data_audit.json")
        transfer = "target_dataset" in result
        dataset = result["target_dataset" if transfer else "dataset"]
        methods = result["primary_seed_averaged"]["methods"]
        first_report = next(iter(result["reports"].values()))
        contract = {
            "dataset": {
                key: value for key, value in dataset.items() if key != "observable_candidate_pool"
            },
            "source_dataset": result.get("source_dataset"),
            "source_receipts": _receipt_contract(audit),
            "class_names": first_report["class_names"],
            "seeds": result["seeds"],
            "endpoint": "source_only_transfer"
            if transfer
            else "participant_exclusive_within_dataset",
            "protocol_id": result["source_input_manifest"]["protocol_id"],
            "preprocessing_equivalence": _preprocessing_contract(result, dataset),
            "estimand": result["primary_seed_averaged"]["estimand"],
            "outer_folds": None if transfer else 5,
            "personalization_budget": 0,
            "inference_unit": "explicit_per_method_not_assumed_equal",
            "bootstrap_unit": "participant_cluster_with_all_seeds",
        }
        if contract["seeds"] != [11, 23, 47]:
            raise ValueError("table requires all three frozen seeds in declared order")
        if common is not None and canonical_json_sha256(contract) != canonical_json_sha256(common):
            raise ValueError("non-comparable run contracts; present these as separate tables")
        common = contract
        current_contracts = method_inference_contracts(result)
        inference_contracts.update(current_contracts)
        prediction = directory / result["prediction_artifact"]["path"]
        with np.load(prediction, allow_pickle=False) as archive:
            for name in _IDENTITY_ARRAYS:
                values = archive[name]
                if name in identity and not np.array_equal(identity[name], values):
                    raise ValueError(f"non-comparable prediction identity: {name}")
                identity[name] = values.copy()
            for seed in merged:
                for method in methods:
                    if method in merged[seed]:
                        raise ValueError(
                            f"duplicate method cannot silently replace evidence: {method}"
                        )
                    merged[seed][method] = np.asarray(
                        archive[f"probability__seed-{seed}__{method}"], dtype=np.float64
                    )
        sources.append(
            {
                "run_directory": str(directory.resolve()),
                "result_sha256": sha256_file(directory / "result.json"),
                "predictions_sha256": sha256_file(prediction),
                "data_audit_sha256": sha256_file(directory / "data_audit.json"),
                "launch_commit": result["git_at_launch"]["commit"],
                "clean_git_at_launch": not result["git_at_launch"]["worktree_dirty"],
                "source_manifest_sha256": result["source_input_manifest"]["manifest_sha256"],
                "preprocessing_source_sha256": result["source_input_manifest"]["files"][
                    "src/inclusive_shift_har/data/external_har.py"
                ],
                "runtime_backend_protocol": result.get("runtime_backend_protocol"),
                "observable_candidate_pool": dataset.get("observable_candidate_pool"),
                "validation": validation,
            }
        )
        local_qualifications: dict[str, str] = {}
        for enclosing in (directory, *directory.parents[:3]):
            for notice in enclosing.glob("runtime_termination_failure*.json"):
                local_qualifications[str(notice.resolve())] = sha256_file(notice)
            if (enclosing / "campaign_plan.json").is_file() or enclosing.name in {
                ".audit",
                "results",
            }:
                break
        qualifications.update(local_qualifications)
        for method, inference in current_contracts.items():
            status = (
                "diagnostic_annotation_selected_context"
                if inference["annotation_selected_evaluation_context"]
                else "validated_stress_test"
                if dataset["dataset_id"] == "har_pmd_v1"
                else "validated_development"
            )
            if local_qualifications:
                status += "_with_runtime_qualification"
            method_statuses[method] = status
    assert common is not None
    inputs = ParticipantMetricInputs(
        dataset_id=common["dataset"]["dataset_id"],
        class_names=tuple(common["class_names"]),
        labels=identity["labels"],
        participant_ids=identity["participant_ids"],
    )
    statistics, _ = seed_evidence(
        inputs,
        merged,
        primary_contrast_eligible=common["endpoint"] != "source_only_transfer",
    )
    record: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "matched_external_publication_table",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "evidence_status": "validated_development"
        if inputs.dataset_id != "har_pmd_v1"
        else "validated_stress_test",
        "comparison_contract": common,
        "comparison_contract_sha256": canonical_json_sha256(common),
        "prediction_identity_sha256": canonical_json_sha256(
            {name: values.tolist() for name, values in identity.items()}
        ),
        "statistics": statistics,
        "method_inference_contracts": inference_contracts,
        "method_evidence_statuses": method_statuses,
        "comparison_scope": "Shared dataset/grid/split/seed contract only. Different channel or inference-unit groups are non-comparable for a before/after model-improvement claim.",
        "primary_contrast_qualification": (
            "The retained HERA-full versus XGBoost contrast has different channel/context budgets. "
            "Its original numerical failure is preserved; it is not matched-window superiority evidence."
            if "HERA-DG-full" in inference_contracts
            else None
        ),
        "descriptive_comparisons_vs_strongest_observed_control": _strongest_control_comparisons(
            statistics, inference_contracts
        ),
        "sources": sources,
        "runtime_qualifications": qualifications,
        "reconstruction_only_no_training_or_raw_data_access": True,
        "independent_confirmation_or_sota_claim_allowed": False,
    }
    if qualifications:
        record["evidence_status"] += "_with_runtime_qualification"
    if any(
        value["annotation_selected_evaluation_context"] for value in inference_contracts.values()
    ):
        record["evidence_status"] = "mixed_validated_and_diagnostic_methods"
    record["record_sha256"] = canonical_json_sha256(record)
    return record


def table_markdown(record: dict[str, Any]) -> str:
    statistics = record["statistics"]
    lines = [
        "# External results with explicit comparison groups",
        "",
        f"Evidence status: {record['evidence_status']}. Seeds: 11, 23, 47. "
        f"Independent participants: {statistics['participant_count']}.",
        "",
        "Primary: seed-averaged participant fixed-class macro-F1. Intervals resample "
        "participants, retaining their seeds. Probability ensembles are not the primary.",
        "",
    ]
    contracts = record["method_inference_contracts"]
    groups = sorted(
        {(item["inference_unit"], item["input_channel_count"]) for item in contracts.values()},
        key=lambda item: (item[0], -1 if item[1] is None else item[1]),
    )
    for unit, channels in groups:
        lines.extend(
            [
                f"## {unit.replace('_', ' ')}; {channels if channels is not None else 'unknown'} input channels",
                "",
                "Different inference/channel groups are not a matched before/after comparison.",
                "",
                "| Method | Mean F1 [95% CI] | Worst | Bottom 30% | Present-class sensitivity | Evidence status |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for method, row in sorted(statistics["methods"].items()):
            if (contracts[method]["inference_unit"], contracts[method]["input_channel_count"]) != (
                unit,
                channels,
            ):
                continue
            lo, hi = row["participant_bootstrap_95_percent_ci"]
            name = method.replace("TinyHAR-", "TinyHAR-style-")
            lines.append(
                f"| {name} | {row['mean_participant_macro_f1']:.6f} [{lo:.6f}, {hi:.6f}] "
                f"| {row['worst_participant_macro_f1']:.6f} "
                f"| {row['bottom_30_percent_participant_macro_f1']:.6f} "
                f"| {row['present_class_sensitivity_mean']:.6f} | {record['method_evidence_statuses'][method]} |"
            )
        lines.append("")
    lines.extend(
        [
            "",
            "Native-nine and derived-gravity datasets are never pooled. Within a dataset, "
            "6ch/N9/DG suffixes identify the frozen representation comparison. "
            "Neural architecture names denote repository implementations, not verified "
            "reproductions of the original papers' training recipes.",
            "",
            f"Comparison contract SHA-256: `{record['comparison_contract_sha256']}`.",
            "",
            "Full seed reports, per-class metrics, confusion/calibration, participant "
            "distributions and paired comparisons are in the companion JSON.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, action="append", required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    record = reconstruct_matched_table(args.run_directory, args.repository_root.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    _write_json_create_only(args.output / "table.json", record)
    with (args.output / "table.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(table_markdown(record))
    print(json.dumps({"status": "TABLE_RECONSTRUCTED", "record_sha256": record["record_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
