"""Execute a predeclared final source-only suite sequentially with CUDA neural training."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch

from inclusive_shift_har.artifacts.source_finalization import load_final_selection_plan
from inclusive_shift_har.experiments.inclusivehar_source import run_source_development
from inclusive_shift_har.manifests.canonical import atomic_write_json_new, canonical_json_sha256
from inclusive_shift_har.models.classical import ClassicalConfig
from inclusive_shift_har.training.engine import TrainingConfig


class FinalSourceSuiteError(RuntimeError):
    """Raised when the predeclared final source suite cannot proceed safely."""


_RUNNER_ARGUMENTS = frozenset(
    {
        "epochs",
        "learning_rate",
        "weight_decay",
        "use_augmentation",
        "use_content_objective",
        "use_realization_factorization",
        "use_group_dro",
        "coral_weight",
        "dann_domain_loss_weight",
        "dann_grl_max_strength",
        "dann_grl_warmup_epochs",
        "disable_cudnn",
        "zero_channel_indices",
    }
)


def _git_commit(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_status(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _safe_output_path(template: str, *, seed: int, repository_root: Path, role: str) -> Path:
    rendered = template.format(seed=seed)
    raw = Path(rendered)
    candidate = raw if raw.is_absolute() else repository_root / raw
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(repository_root)
    except ValueError as exc:
        raise FinalSourceSuiteError(f"{role} escapes the repository root") from exc
    return resolved


def _runner_kwargs(
    value: Any,
    *,
    common_value: Any,
    model_id: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not isinstance(common_value, Mapping):
        raise FinalSourceSuiteError(f"{model_id} runner_arguments must be an object")
    combined = {**dict(common_value), **dict(value)}
    unknown = sorted(set(combined) - _RUNNER_ARGUMENTS)
    if unknown:
        raise FinalSourceSuiteError(f"{model_id} has unknown runner arguments: {unknown}")
    required = {"epochs", "learning_rate", "weight_decay"}
    missing = sorted(required - set(combined))
    if missing:
        raise FinalSourceSuiteError(f"{model_id} lacks runner arguments: {missing}")
    kwargs = combined
    zeroed = kwargs.get("zero_channel_indices", [])
    if not isinstance(zeroed, list):
        raise FinalSourceSuiteError(f"{model_id} zero_channel_indices must be a list")
    kwargs["zero_channel_indices"] = tuple(int(index) for index in zeroed)
    return kwargs


def _record_failure(
    *,
    failure_root: Path,
    model_id: str,
    seed: int,
    code_commit: str,
    exc: Exception,
) -> Path:
    failure_root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "final_source_suite_failure",
        "status": "failed_preserved",
        "failed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "model_id": model_id,
        "seed": seed,
        "code_commit": code_commit,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "target_subject_or_window_records_loaded": False,
        "target_predictions_or_performance_accessed": False,
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    destination = failure_root / f"{model_id}--seed-{seed}.json"
    return atomic_write_json_new(payload, destination, allowed_root=failure_root)


def validate_plan_configuration_contract(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct every seed-11 configuration before any final training starts."""

    validated: list[dict[str, Any]] = []
    common_runner = plan.get("common_neural_runner_arguments", {})
    for entry in cast(list[Mapping[str, Any]], plan["models"]):
        model_id = str(entry["model_id"])
        regime = str(entry["training_regime"])
        arguments = _runner_kwargs(
            entry.get("runner_arguments"),
            common_value=common_runner if regime == "fixed_epoch_neural" else {},
            model_id=model_id,
        )
        if regime == "fixed_epoch_neural":
            epochs = int(arguments.pop("epochs"))
            learning_rate = float(arguments.pop("learning_rate"))
            weight_decay = float(arguments.pop("weight_decay"))
            config = TrainingConfig(
                model_name=str(entry["runner_model_name"]),
                num_classes=len(cast(list[str], plan["class_names"])),
                seed=11,
                epochs=epochs,
                batch_size=256,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                patience=min(12, max(2, epochs // 3)),
                minimum_epochs=min(8, epochs),
                mixed_precision="float16",
                checkpoint_interval=max(epochs, 1),
                checkpoint_selection_rule="fixed_last_epoch",
                **arguments,
            )
            configuration = json.loads(json.dumps(asdict(config)))
            common_expectations = plan.get("common_neural_configuration_expectations", {})
            runner_expectations = {
                **dict(cast(Mapping[str, Any], common_runner)),
                **dict(cast(Mapping[str, Any], entry["runner_arguments"])),
            }
        else:
            xgboost_device = "cuda" if entry["runner_model_name"] == "xgboost" else "cpu"
            configuration = asdict(
                ClassicalConfig(
                    model_name=str(entry["runner_model_name"]),
                    num_classes=len(cast(list[str], plan["class_names"])),
                    seed=11,
                    xgboost_device=xgboost_device,
                )
            )
            common_expectations = plan.get("common_classical_configuration_expectations", {})
            runner_expectations = {}
        if not isinstance(common_expectations, Mapping):
            raise FinalSourceSuiteError(f"{model_id} common expectations are invalid")
        expectations = {
            **dict(common_expectations),
            **runner_expectations,
            **dict(cast(Mapping[str, Any], entry["configuration_expectations"])),
        }
        expected_keys = set(configuration) - {"seed"}
        if set(expectations) != expected_keys:
            raise FinalSourceSuiteError(
                f"{model_id} expectations do not cover the exact configuration keys"
            )
        mismatches = [
            key for key, expected in expectations.items() if configuration.get(key) != expected
        ]
        if mismatches:
            raise FinalSourceSuiteError(
                f"{model_id} predeclared configuration does not reconstruct: {mismatches}"
            )
        validated.append(
            {
                "model_id": model_id,
                "training_regime": regime,
                "seed_11_configuration_sha256": canonical_json_sha256(configuration),
            }
        )
    return validated


def run_final_source_suite(
    selection_plan_path: str | Path,
    *,
    repository_root: str | Path,
    source_manifest_path: str | Path,
    raw_csv_path: str | Path,
    dataset_manifest_path: str | Path,
    expected_code_commit: str,
    failure_directory: str | Path,
) -> list[dict[str, Any]]:
    """Run the exact plan once; every neural entry is forced through CUDA."""

    root = Path(repository_root).resolve(strict=True)
    plan = load_final_selection_plan(selection_plan_path)
    validate_plan_configuration_contract(plan)
    if _git_commit(root) != expected_code_commit:
        raise FinalSourceSuiteError("current Git commit differs from the final execution commit")
    status = _git_status(root)
    if status:
        raise FinalSourceSuiteError(f"final source suite requires a clean worktree: {status}")
    if not torch.cuda.is_available():
        raise FinalSourceSuiteError("final source suite requires available CUDA")
    if plan.get("neural_training_device") != "cuda":
        raise FinalSourceSuiteError("final selection plan does not lock neural training to CUDA")
    failures = Path(failure_directory)
    if not failures.is_absolute():
        failures = root / failures
    summaries: list[dict[str, Any]] = []
    for entry_value in cast(list[Mapping[str, Any]], plan["models"]):
        model_id = str(entry_value["model_id"])
        runner_model_name = str(entry_value["runner_model_name"])
        arguments = _runner_kwargs(
            entry_value.get("runner_arguments"),
            common_value=plan.get("common_neural_runner_arguments", {})
            if entry_value["training_regime"] == "fixed_epoch_neural"
            else {},
            model_id=model_id,
        )
        for seed in cast(list[int], plan["required_seed_order"]):
            summary_path = _safe_output_path(
                str(entry_value["summary_path_template"]),
                seed=seed,
                repository_root=root,
                role=f"{model_id} summary",
            )
            run_directory = _safe_output_path(
                str(entry_value["run_directory_template"]),
                seed=seed,
                repository_root=root,
                role=f"{model_id} run directory",
            )
            if summary_path.exists() or run_directory.exists():
                raise FinalSourceSuiteError(
                    f"refusing to reuse final source destination for {model_id} seed {seed}"
                )
            try:
                summary = run_source_development(
                    source_manifest_path=Path(source_manifest_path),
                    raw_csv_path=Path(raw_csv_path),
                    dataset_manifest_path=Path(dataset_manifest_path),
                    model_name=runner_model_name,
                    seed=seed,
                    run_directory=run_directory,
                    summary_path=summary_path,
                    code_commit=expected_code_commit,
                    device_name="cuda",
                    fold_id="final_source_split",
                    checkpoint_selection_rule="fixed_last_epoch",
                    **arguments,
                )
            except Exception as exc:
                failure_path = _record_failure(
                    failure_root=failures,
                    model_id=model_id,
                    seed=seed,
                    code_commit=expected_code_commit,
                    exc=exc,
                )
                raise FinalSourceSuiteError(
                    f"final source run failed; evidence preserved at {failure_path}"
                ) from exc
            summaries.append(
                {
                    "model_id": model_id,
                    "seed": seed,
                    "summary_path": summary_path.relative_to(root).as_posix(),
                    "summary_record_sha256": summary["record_sha256"],
                    "model_device": summary["model_device"],
                    "best_epoch": summary["best_epoch"],
                }
            )
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--failure-directory", default="results/failures/final_source_suite_v1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summaries = run_final_source_suite(
        args.plan,
        repository_root=args.repository_root,
        source_manifest_path=args.source_manifest,
        raw_csv_path=args.raw_csv,
        dataset_manifest_path=args.dataset_manifest,
        expected_code_commit=args.code_commit,
        failure_directory=args.failure_directory,
    )
    print(
        json.dumps(
            {
                "status": "complete_source_only_target_sealed",
                "model_seed_count": len(summaries),
                "summaries": summaries,
                "target_subject_or_window_records_loaded": False,
                "target_predictions_or_performance_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
