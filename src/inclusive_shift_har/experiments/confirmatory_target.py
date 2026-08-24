"""One-time operational boundary for the locked InclusiveHAR target opening."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch

from inclusive_shift_har.data.materialize import materialize_inclusivehar_windows
from inclusive_shift_har.data.windowing import WindowRecord
from inclusive_shift_har.evaluation.locked_target import run_locked_target_evaluation
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, load_json_strict

OPENING_ACKNOWLEDGEMENT = "OPEN-INCLUSIVEHAR-TARGET-ONCE"


class ConfirmatoryTargetError(RuntimeError):
    """Raised before the one-time opening when an operational gate differs."""


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfirmatoryTargetError(f"{name} must be an object")
    return value


def _git_state(root: Path) -> tuple[str, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, status


def _require_unchanged_evaluation_code(
    root: Path,
    *,
    recorded_commit: str,
    current_commit: str,
) -> None:
    if not recorded_commit:
        raise ConfirmatoryTargetError("unlock record lacks evaluation tooling commit")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", recorded_commit, current_commit],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise ConfirmatoryTargetError("evaluation tooling commit is not an ancestor of HEAD")
    changed = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            recorded_commit,
            current_commit,
            "--",
            "src",
            "pyproject.toml",
            "uv.lock",
        ],
        cwd=root,
        check=False,
    )
    if changed.returncode != 0:
        raise ConfirmatoryTargetError("evaluation code changed after the approved tooling commit")


def run_confirmatory_target_once(
    *,
    acknowledgement: str,
    repository_root: str | Path,
    target_manifest_path: str | Path,
    unlock_record_path: str | Path,
    final_freeze_inventory_path: str | Path,
    source_selection_plan_path: str | Path,
    raw_csv_path: str | Path,
    machine_record_path: str | Path,
    receipt_root: str | Path,
    output_directory: str | Path,
    output_root: str | Path,
    opened_at_utc: str,
) -> dict[str, Any]:
    """Consume the exact unlock and perform all frozen inference in one CUDA pass."""

    if acknowledgement != OPENING_ACKNOWLEDGEMENT:
        raise ConfirmatoryTargetError("exact one-time opening acknowledgement is required")
    root = Path(repository_root).resolve(strict=True)
    target_value = load_json_strict(target_manifest_path)
    target_manifest = _mapping(target_value, name="target split manifest")
    unlock_value = load_json_strict(unlock_record_path)
    unlock = _mapping(unlock_value, name="target unlock record")
    selection_value = load_json_strict(source_selection_plan_path)
    selection = _mapping(selection_value, name="source selection plan")
    machine_value = load_json_strict(machine_record_path)
    machine = _mapping(machine_value, name="machine record")
    machine_sha = machine.get("record_sha256")
    if not isinstance(machine_sha, str) or len(machine_sha) != 64:
        raise ConfirmatoryTargetError("machine record lacks a full self-hash")
    commit, status = _git_state(root)
    if status:
        raise ConfirmatoryTargetError(f"confirmatory opening requires a clean worktree: {status}")
    _require_unchanged_evaluation_code(
        root,
        recorded_commit=str(unlock.get("evaluation_tooling_code_commit", "")),
        current_commit=commit,
    )
    if not torch.cuda.is_available():
        raise ConfirmatoryTargetError("locked confirmatory neural inference requires CUDA")
    class_names_value = selection.get("class_names")
    if not isinstance(class_names_value, list) or class_names_value != [
        "mobility",
        "sitting",
        "standing",
    ]:
        raise ConfirmatoryTargetError("source selection plan has an unexpected class schema")
    class_names = tuple(str(value) for value in class_names_value)
    source_window_path_value = selection.get("source_window_manifest_path")
    if not isinstance(source_window_path_value, str):
        raise ConfirmatoryTargetError("source selection plan lacks its source manifest path")
    source_window_value = load_json_strict(root / source_window_path_value)
    source_window = _mapping(source_window_value, name="source window manifest")
    source_window_body = dict(source_window)
    source_window_claimed_hash = source_window_body.pop("source_window_manifest_sha256", None)
    if source_window_claimed_hash != selection.get(
        "source_window_manifest_sha256"
    ) or source_window_claimed_hash != canonical_json_sha256(source_window_body):
        raise ConfirmatoryTargetError("source window manifest lineage changed after selection")
    source_hash = source_window.get("source_artifact_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ConfirmatoryTargetError("source selection plan lacks the raw source hash")

    def materialize_after_receipt(authorized: Mapping[str, Any]) -> Any:
        windows_value = target_manifest.get("windows")
        if not isinstance(windows_value, list):
            raise ConfirmatoryTargetError("target split manifest lacks windows after opening")
        records = tuple(
            WindowRecord(**cast(dict[str, Any], value))
            for value in windows_value
            if isinstance(value, dict) and value.get("partition") == "target_sealed"
        )
        return materialize_inclusivehar_windows(
            Path(raw_csv_path),
            records,
            expected_source_sha256=source_hash,
            ontology_track="functional_core",
            class_names=class_names,
            allowed_partitions={"target_sealed"},
            target_unlock=dict(authorized),
            expected_target_seal_id=str(unlock["target_seal_id"]),
            expected_split_manifest_sha256=str(unlock["split_manifest_sha256"]),
        )

    return run_locked_target_evaluation(
        target_manifest,
        unlock,
        final_freeze_inventory_path=final_freeze_inventory_path,
        artifact_root=root,
        receipt_root=receipt_root,
        materialize_target=materialize_after_receipt,
        output_directory=output_directory,
        output_root=output_root,
        opened_at_utc=opened_at_utc,
        machine_record_sha256=machine_sha,
        device=torch.device("cuda"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acknowledge-one-time-opening", required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--unlock-record", type=Path, required=True)
    parser.add_argument("--final-freeze-inventory", type=Path, required=True)
    parser.add_argument("--source-selection-plan", type=Path, required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--machine-record", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--opened-at-utc", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_confirmatory_target_once(
        acknowledgement=args.acknowledge_one_time_opening,
        repository_root=args.repository_root,
        target_manifest_path=args.target_manifest,
        unlock_record_path=args.unlock_record,
        final_freeze_inventory_path=args.final_freeze_inventory,
        source_selection_plan_path=args.source_selection_plan,
        raw_csv_path=args.raw_csv,
        machine_record_path=args.machine_record,
        receipt_root=args.receipt_root,
        output_directory=args.output_directory,
        output_root=args.output_root,
        opened_at_utc=args.opened_at_utc,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
