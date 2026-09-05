"""Run the full IMU-HAR-IL development suite and zero-shot FoG transfer once."""

from __future__ import annotations

import argparse
import gc
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from inclusive_shift_har.data.external_har import load_fog_star, load_imu_har_il
from inclusive_shift_har.experiments.cross_dataset_har import (
    _git_state,
    _read_mapping,
    _source_input_manifest,
    _write_json_create_only,
    run_and_write,
)
from inclusive_shift_har.experiments.cross_dataset_neural import run_and_write_neural
from inclusive_shift_har.experiments.cross_dataset_transfer import run_and_write_transfer


def _method_means(result: dict[str, Any]) -> dict[str, float]:
    reports = cast(dict[str, dict[str, Any]], result["reports"])
    return {
        name: float(report["primary"]["mean_participant_macro_f1"])
        for name, report in reports.items()
    }


def run_primary_suite(
    *,
    output_root: Path,
    repository_root: Path,
    seeds: tuple[int, ...],
    epochs: int,
    n_jobs: int,
) -> dict[str, Any]:
    """Reuse one full source materialization across source and transfer experiments."""

    git_at_launch = _git_state(repository_root)
    source_input_manifest = _source_input_manifest(repository_root)
    output_root.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC).isoformat()
    _write_json_create_only(
        output_root / "suite_plan.json",
        {
            "schema_version": "1.0.0",
            "started_at": started,
            "execution_order": [
                "IMU-HAR-IL all-repetition nested invention/classical evaluation",
                "IMU-HAR-IL all-repetition neural controls",
                "IMU-HAR-IL to FoG-STAR zero-shot transfer",
            ],
            "seeds": list(seeds),
            "neural_epochs": epochs,
            "raw_local_mirror": False,
            "source_materialization_reused_in_memory": True,
            "git_at_launch": git_at_launch,
            "source_input_manifest": source_input_manifest,
        },
    )
    try:
        experiment = _read_mapping(
            repository_root / "configs/experiments/cross_dataset_har_rnd_v1.yaml"
        )
        preprocessing = cast(dict[str, Any], experiment["preprocessing"])
        derived = cast(dict[str, Any], preprocessing["derived_gravity"])
        target_rate_hz = float(preprocessing["target_sampling_rate_hz"])
        window_samples = int(preprocessing["window_samples"])
        gravity_cutoff_hz = float(derived["cutoff_hz"])
        source = load_imu_har_il(
            repetition_limit=4,
            target_rate_hz=target_rate_hz,
            window_samples=window_samples,
            gravity_cutoff_hz=gravity_cutoff_hz,
        )
        classical = run_and_write(
            data=source,
            output_directory=output_root / "imu_har_il_all_repetitions_3seed",
            repository_root=repository_root,
            seeds=seeds,
            n_jobs=n_jobs,
            include_classical=True,
        )
        _write_json_create_only(
            output_root / "stage_01_imu_classical_complete.json",
            {
                "completed_at": datetime.now(UTC).isoformat(),
                "dataset": source.summary(),
                "method_means": _method_means(classical),
            },
        )
        neural = run_and_write_neural(
            data=source,
            output_directory=output_root / "imu_har_il_all_repetitions_neural_3seed",
            repository_root=repository_root,
            seeds=seeds,
            epochs=epochs,
        )
        _write_json_create_only(
            output_root / "stage_02_imu_neural_complete.json",
            {
                "completed_at": datetime.now(UTC).isoformat(),
                "method_means": _method_means(neural),
            },
        )
        gc.collect()
        target = load_fog_star(
            target_rate_hz=target_rate_hz,
            window_samples=window_samples,
            gravity_cutoff_hz=gravity_cutoff_hz,
        )
        transfer = run_and_write_transfer(
            source=source,
            target=target,
            output_directory=output_root / "imu_all_to_fog_zero_shot_3seed",
            repository_root=repository_root,
            seeds=seeds,
            n_jobs=n_jobs,
            include_classical=True,
        )
        completed = {
            "schema_version": "1.0.0",
            "status": "PRIMARY_SUITE_COMPLETE",
            "started_at": started,
            "completed_at": datetime.now(UTC).isoformat(),
            "imu_method_means": _method_means(classical),
            "imu_neural_method_means": _method_means(neural),
            "zero_shot_method_means": _method_means(transfer),
            "raw_local_mirror": False,
        }
        _write_json_create_only(output_root / "suite_complete.json", completed)
        return completed
    except Exception as exc:
        _write_json_create_only(
            output_root / "suite_failure.json",
            {
                "schema_version": "1.0.0",
                "status": "FAILED_PRESERVED",
                "started_at": started,
                "failed_at": datetime.now(UTC).isoformat(),
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
                "git": _git_state(repository_root),
                "git_at_launch": git_at_launch,
                "source_input_manifest": source_input_manifest,
            },
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_primary_suite(
        output_root=args.output_root.resolve(),
        repository_root=args.repository_root.resolve(),
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        n_jobs=args.n_jobs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
