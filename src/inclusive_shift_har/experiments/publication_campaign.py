"""Frozen-source external campaigns with acquisition failures and acceptance gates."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inclusive_shift_har.data.external_har import (
    load_fog_star,
    load_har_pmd_native,
    load_sole_harmony,
)
from inclusive_shift_har.experiments import cross_dataset_har, cross_dataset_neural
from inclusive_shift_har.experiments.cross_dataset_har import (
    _git_state,
    _source_input_manifest,
    _write_json_create_only,
)
from inclusive_shift_har.experiments.external_evidence_validate import (
    _manifest_commit_errors,
    validate_run_directory,
)
from inclusive_shift_har.experiments.external_primary_suite import run_primary_suite
from inclusive_shift_har.manifests.canonical import canonical_json_sha256


def _accept(directory: Path, root: Path, *, diagnostic: bool = False) -> None:
    validation = validate_run_directory(directory, root)
    _write_json_create_only(directory / "validation.json", validation)
    accepted = (
        validation["integrity_passed"] if diagnostic else validation["publication_evidence_ready"]
    )
    if not accepted:
        raise ValueError(f"acceptance gate failed for {directory}")


def run_campaign(
    *, dataset: str, output: Path, repository_root: Path, n_jobs: int = 4
) -> dict[str, Any]:
    launch = _git_state(repository_root)
    manifest = _source_input_manifest(repository_root)
    if launch["worktree_dirty"] or _manifest_commit_errors(
        repository_root, launch["commit"], manifest["files"]
    ):
        raise ValueError(
            "publication campaigns require clean committed source matching the executed package"
        )
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "protocol_id": "external-har-session-grid-v3",
        "dataset": dataset,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "seeds": [11, 23, 47],
        "neural_epochs": 40,
        "n_jobs": n_jobs,
        "git_at_launch": launch,
        "source_input_manifest": manifest,
        "environment": {
            dist.metadata["Name"]: dist.version
            for dist in importlib.metadata.distributions()
            if "Name" in dist.metadata
        },
        "evidence_role": "oracle_diagnostic"
        if dataset == "sole-harmony"
        else "stress_test"
        if dataset == "har-pmd"
        else "development",
    }
    plan["record_sha256"] = canonical_json_sha256(plan)
    _write_json_create_only(output / "campaign_plan.json", plan)
    print(
        json.dumps(
            {"stage": "frozen_campaign_started", "dataset": dataset, "commit": launch["commit"]}
        ),
        flush=True,
    )
    try:
        if dataset == "imu-har-il":
            run_primary_suite(
                output_root=output / "primary_suite",
                repository_root=repository_root,
                seeds=(11, 23, 47),
                epochs=40,
                n_jobs=n_jobs,
            )
        elif dataset == "fog-star":
            data = load_fog_star()
            cross_dataset_har.run_and_write(
                data=data,
                output_directory=output / "classical",
                repository_root=repository_root,
                seeds=(11, 23, 47),
                n_jobs=n_jobs,
                include_classical=True,
            )
            _accept(output / "classical", repository_root)
            print("FoG classical acceptance gate passed; starting neural controls", flush=True)
            cross_dataset_neural.run_and_write_neural(
                data=data,
                output_directory=output / "neural",
                repository_root=repository_root,
                seeds=(11, 23, 47),
                epochs=40,
            )
            _accept(output / "neural", repository_root)
        elif dataset == "har-pmd":
            from inclusive_shift_har.experiments.har_pmd_stress import run_and_write

            data = load_har_pmd_native()
            if data.summary()["participant_count"] != 120:
                raise ValueError("full HAR-PMD campaign requires all 120 participants")
            run_and_write(
                data=data,
                output_directory=output / "classical",
                repository_root=repository_root,
                seeds=(11, 23, 47),
            )
            _accept(output / "classical", repository_root)
            for suffix, signals in (("6ch", data.signals), ("N9", data.nine_channel_signals)):
                cross_dataset_neural.run_and_write_neural(
                    data=data,
                    signals=signals,
                    class_names=data.class_names,
                    method_suffix=suffix,
                    experiment_id="har-pmd-native-interface-neural-controls-v1",
                    output_directory=output / f"neural_{suffix}",
                    repository_root=repository_root,
                    seeds=(11, 23, 47),
                    epochs=40,
                )
                _accept(output / f"neural_{suffix}", repository_root)
        elif dataset == "sole-harmony":
            from inclusive_shift_har.experiments.sole_harmony_temporal import (
                run_and_write as run_temporal,
            )

            data = load_sole_harmony(participant_limit=12, sessions_per_participant=2)
            run_temporal(
                data=data,
                output_directory=output / "oracle_temporal",
                repository_root=repository_root,
                seeds=(11, 23, 47),
                n_jobs=n_jobs,
            )
            _accept(output / "oracle_temporal", repository_root, diagnostic=True)
        else:
            raise ValueError(f"dataset is not in the frozen campaign: {dataset}")
        completed = {
            "status": "DIAGNOSTIC_COMPLETE" if dataset == "sole-harmony" else "CAMPAIGN_COMPLETE",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "plan_sha256": plan["record_sha256"],
        }
        _write_json_create_only(output / "campaign_complete.json", completed)
        return completed
    except Exception as error:
        failure = {
            "status": "FAILED_PRESERVED",
            "failed_at_utc": datetime.now(UTC).isoformat(),
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": traceback.format_exc(),
            "git_at_launch": launch,
            "source_input_manifest": manifest,
            "plan_sha256": plan["record_sha256"],
        }
        _write_json_create_only(output / "campaign_failure.json", failure)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("fog-star", "imu-har-il", "har-pmd", "sole-harmony"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args(argv)
    result = run_campaign(
        dataset=args.dataset,
        output=args.output.resolve(),
        repository_root=args.repository_root.resolve(),
        n_jobs=args.n_jobs,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
