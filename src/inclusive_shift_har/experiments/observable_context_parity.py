"""Create-only parity gate for the annotation-independent context correction."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from inclusive_shift_har.artifacts.research_provenance import (
    _git_state,
    _source_input_manifest,
    _write_json_create_only,
)
from inclusive_shift_har.evaluation.inference_contracts import (
    OBSERVABLE_CONTEXT_PROTOCOL,
    PARTICIPANT_CONTEXT_METHODS,
)
from inclusive_shift_har.experiments.external_evidence_validate import validate_run_directory
from inclusive_shift_har.experiments.publication_table import _receipt_contract
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file


def compare_context_replacement(reference: Path, replacement: Path, root: Path) -> dict[str, Any]:
    packages = []
    provenance = []
    for directory in (reference, replacement):
        validation = validate_run_directory(directory, root)
        if not validation.get("publication_evidence_ready") and not validation.get(
            "publication_evidence_ready_for_unqualified_methods"
        ):
            raise ValueError("parity requires independently validated window-local evidence")
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        audit = json.loads((directory / "data_audit.json").read_text(encoding="utf-8"))
        packages.append((result, audit))
        provenance.append(
            {
                "directory": str(directory.resolve()),
                "result_sha256": sha256_file(directory / "result.json"),
                "predictions_sha256": sha256_file(directory / "predictions.npz"),
                "validation": validation,
            }
        )
    old, new = packages[0][0], packages[1][0]
    if new.get("observable_context_protocol") != OBSERVABLE_CONTEXT_PROTOCOL:
        raise ValueError("replacement does not declare the observable context correction")
    for key in ("dataset", "source_dataset", "target_dataset"):
        summaries = [
            {
                name: value
                for name, value in result.get(key, {}).items()
                if name != "observable_candidate_pool"
            }
            for result in (old, new)
        ]
        if summaries[0] != summaries[1]:
            raise ValueError(f"scoring cohort, grid or preprocessing differs: {key}")
    if _receipt_contract(packages[0][1]) != _receipt_contract(packages[1][1]):
        raise ValueError("provider source receipts differ")
    if old.get("seeds") != [11, 23, 47] or new.get("seeds") != [11, 23, 47]:
        raise ValueError("parity requires the three frozen seeds")
    old_methods, new_methods = (
        set(result["primary_seed_averaged"]["methods"]) for result in (old, new)
    )
    if old_methods != new_methods:
        raise ValueError("method set changed during a context-only repair")
    comparisons: dict[str, Any] = {}
    with (
        np.load(reference / "predictions.npz", allow_pickle=False) as left,
        np.load(replacement / "predictions.npz", allow_pickle=False) as right,
    ):
        for name in ("labels", "participant_ids", "session_ids", "trial_ids", "window_ids"):
            if not np.array_equal(left[name], right[name]):
                raise ValueError(f"scoring identities changed: {name}")
        for method in sorted(new_methods):
            per_seed = {}
            for seed in (11, 23, 47):
                key = f"probability__seed-{seed}__{method}"
                a, b = left[key], right[key]
                per_seed[str(seed)] = {
                    "decisions_exact": bool(np.array_equal(a.argmax(axis=1), b.argmax(axis=1))),
                    "probabilities_bitwise_equal": bool(np.array_equal(a, b)),
                    "maximum_absolute_probability_difference": float(np.max(np.abs(a - b))),
                    "probabilities_within_predeclared_tolerance": bool(
                        np.allclose(a, b, rtol=0, atol=1e-12)
                    ),
                }
            affected = method in PARTICIPANT_CONTEXT_METHODS
            comparisons[method] = {
                "context_affected": affected,
                "seeds": per_seed,
                "parity_required": not affected,
                "parity_passed": None
                if affected
                else all(
                    row["decisions_exact"] and row["probabilities_within_predeclared_tolerance"]
                    for row in per_seed.values()
                ),
                "old_mean_participant_macro_f1": old["primary_seed_averaged"]["methods"][method][
                    "mean_participant_macro_f1"
                ],
                "replacement_mean_participant_macro_f1": new["primary_seed_averaged"]["methods"][
                    method
                ]["mean_participant_macro_f1"],
            }
    record = {
        "protocol_id": OBSERVABLE_CONTEXT_PROTOCOL,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PARITY_PASSED"
        if all(item["parity_passed"] for item in comparisons.values() if item["parity_required"])
        else "PARITY_DISCREPANCY_PRESERVED",
        "unaffected_probability_absolute_tolerance": 1e-12,
        "tolerance_protocol": "configs/protocols/external_har_observable_context_v1.yaml",
        "methods": comparisons,
        "sources": provenance,
        "analysis_git": _git_state(root),
        "analysis_source_input_manifest": _source_input_manifest(root),
        "affected_context_before_after_is_a_validity_correction_not_a_matched_model_improvement": True,
        "better_score_selection_allowed": False,
    }
    record["record_sha256"] = canonical_json_sha256(record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--replacement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    record = compare_context_replacement(
        args.reference, args.replacement, args.repository_root.resolve()
    )
    _write_json_create_only(args.output, record)
    print(json.dumps({"status": record["status"], "record_sha256": record["record_sha256"]}))
    return 0 if record["status"] == "PARITY_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
