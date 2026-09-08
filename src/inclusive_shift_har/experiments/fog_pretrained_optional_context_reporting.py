"""Zero-fit reporting supplement for the completed optional-context experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.experiments.fog_left_ankle_derived_nine import (
    _stop_task_owned_workers,
    _worker_baseline,
)
from inclusive_shift_har.experiments.fog_pretrained_optional_context_run import METHOD_ORDER
from inclusive_shift_har.experiments.fog_rf_feature_weight_factorial import (
    _mapping,
    _require,
    _sealed,
    _write_json_create_only,
    method_report,
)
from inclusive_shift_har.manifests.canonical import sha256_file

RUN_NAME = "fog-pretrained-optional-context-seed11-20260908-001"
SUPPLEMENT_NAME = f"{RUN_NAME}-reporting-supplement"
ARMS = ("Q", "M", "R", "P")


def _read_json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _read_npz(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def reporting_strata(
    *, current: NDArray[np.bool_], full: NDArray[np.bool_], q_zero: NDArray[np.bool_]
) -> dict[str, NDArray[np.bool_]]:
    _require(current.shape == full.shape == q_zero.shape, "reporting masks are misaligned")
    _require(bool(np.all(full <= current)), "full history exists without current ankle")
    return {
        "all_scored": np.ones(current.size, dtype=np.bool_),
        "current_ankle": current,
        "full_history": full,
        "short_history_current": current & ~full,
        "missing_current_ankle": ~current,
        "q_zero_all_scored": q_zero,
        "current_q_zero": current & q_zero,
        "current_q_nonzero_editable": current & ~q_zero,
        "full_history_q_zero": full & q_zero,
        "full_history_q_nonzero": full & ~q_zero,
    }


def _verify_manifest(run: Path, expected_completion_sha256: str) -> int:
    completion = run / "completion_manifest.json"
    _require(
        completion.is_file() and sha256_file(completion) == expected_completion_sha256,
        "source completion manifest changed",
    )
    manifest = _read_json(completion)
    rows = cast(list[dict[str, Any]], manifest["artifacts"])
    for row in rows:
        path = run / str(row["path"])
        _require(
            path.is_file() and sha256_file(path) == row["sha256"],
            f"source run artifact changed: {row['path']}",
        )
    return len(rows)


def _summary(
    strata: Mapping[str, Mapping[str, Mapping[str, Any]]], counts: Mapping[str, int]
) -> str:
    lines = [
        "# Optional-context reporting supplement",
        "",
        "All values below are zero-fit descriptive strata from the fixed saved predictions.",
        "Means use participants with at least one row in the named stratum; pooled accuracy uses rows.",
        "",
        "| Stratum | Rows | Method | Eligible people | Mean person macro-F1 | Pooled accuracy |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for name, reports in strata.items():
        for method in ("Q", "M", "R", "P", "l9v"):
            report = reports[method]
            people = [
                float(row["macro_f1"])
                for row in cast(list[dict[str, Any]], report["participants"])
                if row["eligible"] is True
            ]
            lines.append(
                f"| {name} | {counts[name]} | {method} | {len(people)} | "
                f"{100 * float(np.mean(people)):.3f}% | "
                f"{100 * float(report['pooled']['accuracy']):.3f}% |"
            )
    lines.extend(["", "This supplement changes no gate, selection, checkpoint, or claim.", ""])
    return "\n".join(lines)


def generate_reporting_supplement(
    *,
    repository_root: Path,
    run: Path,
    output: Path,
    code_commit: str,
    expected_completion_sha256: str,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    run = run.resolve()
    output = output.resolve()
    _require(run.name == RUN_NAME and run.is_dir(), "source run path changed")
    _require(output.name == SUPPLEMENT_NAME and output.parent == run.parent, "output path changed")
    _require(not output.exists(), "create-only reporting supplement exists")
    git_commit = subprocess.check_output(
        ("git", "-C", str(repository_root), "rev-parse", "HEAD"), text=True
    ).strip()
    git_status = subprocess.check_output(
        ("git", "-C", str(repository_root), "status", "--porcelain=v1"), text=True
    ).strip()
    _require(git_commit == code_commit and not git_status, "reporting source is not clean")
    checked = _verify_manifest(run, expected_completion_sha256)
    validation = _read_json(run / "validation.json")
    result = _read_json(run / "result.json")
    _require(
        validation["status"] == "validated"
        and validation["model_fits_during_validation"] == 0
        and result["completed_fits"] == result["fit_attempts"] == 20,
        "source run is not complete and replayed",
    )
    arrays = _read_npz(run / "predictions.npz")
    methods = arrays["method_ids"].tolist()
    _require(methods == list(METHOD_ORDER), "prediction method order changed")
    probabilities = np.asarray(arrays["scored_probabilities"], dtype=np.float64)
    labels = np.asarray(arrays["scored_labels"], dtype=np.int64)
    people = np.asarray(arrays["scored_participant_ids"], dtype=np.str_)
    current = np.asarray(arrays["current_availability_mask"], dtype=np.bool_)[
        np.asarray(arrays["scoring_indices"], dtype=np.int64)
    ]
    full = np.asarray(arrays["full_context_mask"], dtype=np.bool_)[
        np.asarray(arrays["scoring_indices"], dtype=np.int64)
    ]
    l9v = probabilities[methods.index("l9v")]
    q_zero = l9v[:, 1:].sum(axis=1) == 0.0
    masks = reporting_strata(current=current, full=full, q_zero=q_zero)
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    _require(
        counts
        == {
            "all_scored": 1213,
            "current_ankle": 1154,
            "full_history": 1098,
            "short_history_current": 56,
            "missing_current_ankle": 59,
            "q_zero_all_scored": 301,
            "current_q_zero": 301,
            "current_q_nonzero_editable": 853,
            "full_history_q_zero": 298,
            "full_history_q_nonzero": 800,
        },
        "reporting stratum support changed",
    )
    roster = [f"fogstar:{index:03d}" for index in range(1, 23)]
    reports = {
        stratum: {
            method: method_report(
                labels=labels[mask],
                probabilities=probabilities[index, mask],
                participant_ids=people[mask],
                roster=roster,
            )
            for index, method in enumerate(methods)
        }
        for stratum, mask in masks.items()
    }
    observable = np.asarray(arrays["observable_probabilities"], dtype=np.float64)
    observable_current = np.asarray(arrays["current_availability_mask"], dtype=np.bool_)
    observable_l9v = observable[methods.index("l9v")]
    observable_b0 = observable[methods.index("b0")]
    observable_q_zero = observable_l9v[:, 1:].sum(axis=1) == 0.0
    for arm in ARMS:
        candidate = observable[methods.index(arm)]
        _require(
            np.array_equal(candidate[~observable_current], observable_b0[~observable_current])
            and np.array_equal(candidate[observable_q_zero], observable_l9v[observable_q_zero]),
            f"{arm} fallback changed",
        )
    contexts = _read_json(run / "context_alignment_receipts.json")
    native = cast(list[dict[str, Any]], contexts["native_harnet_receipts"])
    complete = [row for row in native if row["status"] == "complete_native_harnet_history"]
    total_values = sum(int(row["unclipped_value_count"]) for row in complete)
    clipped_low = sum(int(row["clipped_low_count"]) for row in complete)
    clipped_high = sum(int(row["clipped_high_count"]) for row in complete)
    output.mkdir(parents=True)
    binding = _sealed(
        {
            "record_kind": "fog_pretrained_reporting_source_binding",
            "source_run": str(run),
            "source_completion_manifest_sha256": expected_completion_sha256,
            "source_artifacts_verified": checked,
            "reporting_code_commit": code_commit,
            "model_fits": 0,
            "outcome_selection": False,
        }
    )
    _write_json_create_only(output / "source_binding.json", binding)
    _write_json_create_only(
        output / "stratified_metrics.json",
        _sealed(
            {
                "record_kind": "fog_pretrained_stratified_metrics",
                "status": "complete_zero_fit_descriptive_reporting",
                "counts": counts,
                "methods": methods,
                "reports": reports,
                "model_fits": 0,
                "changes_primary_analysis_or_gates": False,
            }
        ),
    )
    _write_json_create_only(
        output / "fallback_and_adapter_receipt.json",
        _sealed(
            {
                "record_kind": "fog_pretrained_fallback_and_adapter_reporting",
                "fallbacks_exact_for_arms": list(ARMS),
                "missing_current_scored_rows": counts["missing_current_ankle"],
                "q_zero_scored_rows": counts["q_zero_all_scored"],
                "native_complete_histories": len(complete),
                "native_total_values_before_clipping": total_values,
                "clipped_low_values": clipped_low,
                "clipped_high_values": clipped_high,
                "clipped_value_rate": float((clipped_low + clipped_high) / total_values),
            }
        ),
    )
    with (output / "STRATIFIED_RESULTS.md").open("x", encoding="utf-8") as stream:
        stream.write(_summary(reports, counts))
    shutdown = _stop_task_owned_workers(_worker_baseline(), 0, 0, terminal="reporting_complete")
    _write_json_create_only(output / "worker_shutdown.json", shutdown)
    artifacts = [
        {
            "path": path.relative_to(output).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output.iterdir())
        if path.is_file()
    ]
    completion = _sealed(
        {
            "record_kind": "fog_pretrained_reporting_completion",
            "status": "complete",
            "artifacts": artifacts,
            "model_fits": 0,
            "all_task_owned_workers_and_monitors_stopped": shutdown[
                "task_owned_fit_workers_and_monitors_stopped"
            ],
        }
    )
    _write_json_create_only(output / "completion_manifest.json", completion)
    return completion


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--run-completion-sha256", required=True)
    arguments = parser.parse_args(argv)
    result = generate_reporting_supplement(
        repository_root=arguments.repository_root,
        run=arguments.run,
        output=arguments.output,
        code_commit=arguments.code_commit,
        expected_completion_sha256=arguments.run_completion_sha256,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
