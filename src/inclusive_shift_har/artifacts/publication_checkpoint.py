"""Create-only command receipts, quality gates and historical supersession snapshots."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from inclusive_shift_har.experiments.cross_dataset_har import (
    _git_state,
    _source_input_manifest,
    _write_json_create_only,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file


def record_command(directory: Path, command: list[str], repository_root: Path) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    record: dict[str, Any] = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "command_argv": command,
        "cwd": str(repository_root),
        "git_at_launch": _git_state(repository_root),
        "source_input_manifest": _source_input_manifest(repository_root),
    }
    _write_json_create_only(directory / "launch.json", record)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    with (
        (directory / "stdout.log").open("xb") as stdout,
        (directory / "stderr.log").open("xb") as stderr,
    ):
        try:
            process = subprocess.run(
                command,
                cwd=repository_root,
                stdout=stdout,
                stderr=stderr,
                env=environment,
                check=False,
            )
            exit_code = process.returncode
        except OSError as error:
            exit_code = -1
            stderr.write(str(error).encode("utf-8"))
    record.update(
        {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "exit_code": exit_code,
            "status": "PASS" if exit_code == 0 else "FAILED_PRESERVED",
            "stdout_sha256": sha256_file(directory / "stdout.log"),
            "stderr_sha256": sha256_file(directory / "stderr.log"),
        }
    )
    record["record_sha256"] = canonical_json_sha256(record)
    _write_json_create_only(directory / "completion.json", record)
    print(
        json.dumps({"command": command, "exit_code": exit_code, "record": str(directory)}),
        flush=True,
    )
    return record


def quality_gates(directory: Path, repository_root: Path, uv: Path) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    python = sys.executable
    prefix = [python, "-m", "inclusive_shift_har.cli"]
    commands = {
        "tests": [
            python,
            "-m",
            "inclusive_shift_har.artifacts.parallel_test_gate",
            "--output",
            str(directory / "full_test_shards"),
            "--workers",
            "4",
        ],
        "lint": [python, "-m", "ruff", "check", "src", "tests"],
        "format": [python, "-m", "ruff", "format", "--check", "src", "tests"],
        "types": [python, "-m", "mypy", "src", "tests"],
        "manifests": [*prefix, "validate-manifests", "--json"],
        "splits": [
            *prefix,
            "audit-splits",
            "--split-manifest",
            "results/protocol/splits/inclusivehar_v4_released_block_v1_2.json",
            "--json",
        ],
        "artifacts": [
            *prefix,
            "validate-artifacts",
            "--artifact-root",
            "results",
            "--require-artifacts",
            "--json",
        ],
        "lock": [str(uv), "lock", "--check", "--offline"],
        "git_fsck": ["git", "fsck", "--full"],
    }
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            name: executor.submit(record_command, directory / name, command, repository_root)
            for name, command in commands.items()
        }
        records = {name: future.result() for name, future in futures.items()}
    configuration_errors = []
    configs = sorted((repository_root / "configs").rglob("*.yaml")) + sorted(
        (repository_root / "configs").rglob("*.json")
    )
    for path in configs:
        try:
            with path.open(encoding="utf-8") as stream:
                yaml.safe_load(stream) if path.suffix == ".yaml" else json.load(stream)
        except (ValueError, yaml.YAMLError) as error:
            configuration_errors.append({"file": str(path), "error": str(error)})
    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS"
        if not configuration_errors and all(item["exit_code"] == 0 for item in records.values())
        else "FAILED_PRESERVED",
        "gates": {
            name: {"exit_code": item["exit_code"], "record_sha256": item["record_sha256"]}
            for name, item in records.items()
        },
        "configuration_count": len(configs),
        "configuration_errors": configuration_errors,
        "environment": {
            dist.metadata["Name"]: dist.version
            for dist in importlib.metadata.distributions()
            if "Name" in dist.metadata
        },
        "source_input_manifest": _source_input_manifest(repository_root),
        "security_note": "index/history secret and license checks are separate required command receipts",
    }
    summary["record_sha256"] = canonical_json_sha256(summary)
    _write_json_create_only(directory / "quality_gates.json", summary)
    return summary


def supersession_snapshot(directory: Path, repository_root: Path) -> dict[str, Any]:
    root = repository_root / "results/research/cross_dataset_har_v1"
    entries = []
    for run in sorted(path for path in root.iterdir() if path.is_dir()):
        files = {
            path.relative_to(repository_root).as_posix(): sha256_file(path)
            for path in sorted(run.rglob("*"))
            if path.is_file()
        }
        if "fog" in run.name:
            status, reason = (
                "superseded",
                "ground-truth-dependent FoG resampling/window phase; physical-grid-v2 replacement required",
            )
        elif run.name.startswith("sole"):
            status, reason = (
                "diagnostic",
                "one participant/session and camera-bout oracle boundaries",
            )
        elif (run / "failure.json").exists() or not (run / "result.json").exists():
            status, reason = (
                "failed_preserved",
                "failed or incomplete historical evidence; original files retained",
            )
        else:
            status, reason = (
                "provisional",
                "consumed development/pilot; full-cohort current-contract clean-source reconstruction required",
            )
        entries.append(
            {
                "run_directory": run.relative_to(repository_root).as_posix(),
                "status": status,
                "reason": reason,
                "file_hashes": files,
                "replacement_run": None,
                "replacement_status": "pending_not_claimed_completed",
            }
        )
    missing = root / "imu_rep1_to_fog_trial_continuous_seed11_validation_20260905"
    record = {
        "schema_version": "2.0.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_id": "external-har-physical-grid-v2",
        "entries": entries,
        "prior_ledger": {
            "path": (root / "supersession_ledger_20260905.json")
            .relative_to(repository_root)
            .as_posix(),
            "sha256": sha256_file(root / "supersession_ledger_20260905.json"),
        },
        "missing_previous_replacement": {
            "path": missing.relative_to(repository_root).as_posix(),
            "exists": missing.exists(),
            "status": "not_completed_no_result_claim",
        },
        "preserved_untracked_test_report": {
            "path": (root / "repository_test_gate_20260905.xml")
            .relative_to(repository_root)
            .as_posix(),
            "sha256": sha256_file(root / "repository_test_gate_20260905.xml"),
        },
        "artifacts_deleted_or_modified": False,
    }
    record["record_sha256"] = canonical_json_sha256(record)
    _write_json_create_only(directory, record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("gates", "command", "supersession"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--uv", type=Path)
    parser.add_argument("--command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    root, output = args.repository_root.resolve(), args.output.resolve()
    if args.action == "gates":
        if args.uv is None:
            parser.error("--uv is required for the lock gate")
        result = quality_gates(output, root, args.uv)
        return 0 if result["status"] == "PASS" else 1
    if args.action == "supersession":
        supersession_snapshot(output, root)
        return 0
    if not args.command:
        parser.error("--command is required")
    result = record_command(output, args.command, root)
    return 0 if result["exit_code"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
