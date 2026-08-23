"""Command-line boundary for the gated InclusiveShift-HAR workflow."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from inclusive_shift_har import __version__
from inclusive_shift_har.artifacts.validation import validate_artifact_directory
from inclusive_shift_har.data.audit import DataAuditMode, DataGateError, audit_manifest_data
from inclusive_shift_har.data.inclusivehar import (
    InclusiveHARAuditError,
    audit_inclusivehar_v4_dataset,
)
from inclusive_shift_har.manifests.validation import (
    ManifestValidationError,
    validate_manifest_directory,
    validate_manifest_file,
)

EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_VALIDATION = 2
EXIT_GATE_CLOSED = 3

_GATED_COMMANDS: dict[str, dict[str, str]] = {
    "build-splits": {
        "code": "DATA_AUDIT_GATE_CLOSED",
        "required_gate": "data_audit_pass",
        "message": "split construction is gated and not implemented until the data audit passes",
    },
    "audit-splits": {
        "code": "SPLIT_MANIFEST_GATE_CLOSED",
        "required_gate": "data_audit_pass_and_split_manifest_present",
        "message": "split audit is gated and not implemented until a deterministic split manifest exists",
    },
    "train": {
        "code": "TRAINING_GATE_CLOSED_NOT_IMPLEMENTED",
        "required_gate": "split_audit_pass",
        "message": "training is gated and not implemented in the Stage 2 provenance scaffold",
    },
    "evaluate": {
        "code": "EVALUATION_GATE_CLOSED_NOT_IMPLEMENTED",
        "required_gate": "final_evaluation_unlock",
        "message": "evaluation is gated and not implemented; no confirmatory target was opened",
    },
}


def gated_command_payload(command: str) -> dict[str, Any]:
    """Return the stable machine-readable payload for a gated placeholder."""

    detail = _GATED_COMMANDS[command]
    return {
        "code": detail["code"],
        "command": command,
        "exit_code": EXIT_GATE_CLOSED,
        "message": detail["message"],
        "required_gate": detail["required_gate"],
        "status": "gated_not_implemented",
    }


def _emit(payload: MappingLike, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    status = payload.get("status", "unknown")
    print(f"status: {status}")
    if "message" in payload:
        print(str(payload["message"]))
    if "code" in payload:
        print(f"code: {payload['code']}")
    if "required_gate" in payload:
        print(f"required gate: {payload['required_gate']}")
    if "report" in payload:
        print(json.dumps(payload["report"], ensure_ascii=False, indent=2, sort_keys=True))


MappingLike = dict[str, Any]


def _validate_manifests(args: argparse.Namespace) -> int:
    if args.paths:
        results: list[dict[str, Any]] = []
        top_errors: list[dict[str, str]] = []
        for raw_path in args.paths:
            path = Path(raw_path)
            if path.is_dir():
                directory_report = validate_manifest_directory(path)
                results.extend(item.to_dict() for item in directory_report.results)
                top_errors.extend(item.to_dict() for item in directory_report.errors)
            else:
                results.append(validate_manifest_file(path).to_dict())
        valid = not top_errors and bool(results) and all(item["valid"] for item in results)
        report: MappingLike = {
            "errors": top_errors,
            "manifest_count": len(results),
            "results": results,
            "valid": valid,
        }
    else:
        report = validate_manifest_directory(args.manifest_root).to_dict()
        valid = bool(report["valid"])
    payload: MappingLike = {
        "command": "validate-manifests",
        "report": report,
        "status": "pass" if valid else "fail",
    }
    _emit(payload, as_json=args.json)
    return EXIT_SUCCESS if valid else EXIT_VALIDATION


def _audit_data(args: argparse.Namespace) -> int:
    if args.profile == "inclusivehar-v4" and (
        args.mode != DataAuditMode.READ_ONLY.value
        or args.data_root is None
        or args.gate_record is None
    ):
        gate_payload: MappingLike = {
            "code": "FULL_AUDIT_REQUIRES_READ_ONLY_GATE",
            "command": "audit-data",
            "message": "the InclusiveHAR v4 content profile requires --read-only, --data-root, and --gate-record",
            "required_gate": "raw_data_read_access_approved",
            "status": "gated",
        }
        _emit(gate_payload, as_json=args.json)
        return EXIT_GATE_CLOSED
    try:
        if args.profile == "inclusivehar-v4":
            report_payload = audit_inclusivehar_v4_dataset(
                args.manifest,
                data_root=args.data_root,
                gate_record_path=args.gate_record,
            )
            audit_status = str(report_payload["status"])
            profile_payload: MappingLike = {
                "command": "audit-data",
                "profile": args.profile,
                "report": report_payload,
                "status": audit_status,
            }
            _emit(profile_payload, as_json=args.json)
            return EXIT_VALIDATION if audit_status == "data_audit_fail" else EXIT_SUCCESS
        report = audit_manifest_data(
            args.manifest,
            mode=args.mode,
            data_root=args.data_root,
            gate_record_path=args.gate_record,
        )
    except DataGateError as exc:
        payload: MappingLike = {
            "code": "RAW_DATA_READ_GATE_CLOSED",
            "command": "audit-data",
            "message": str(exc),
            "required_gate": "raw_data_read_access_approved",
            "status": "gated",
        }
        _emit(payload, as_json=args.json)
        return EXIT_GATE_CLOSED
    except (InclusiveHARAuditError, ManifestValidationError, OSError, ValueError) as exc:
        payload = {
            "code": "DATA_AUDIT_VALIDATION_ERROR",
            "command": "audit-data",
            "message": str(exc),
            "status": "fail",
        }
        _emit(payload, as_json=args.json)
        return EXIT_VALIDATION
    if args.mode == DataAuditMode.DRY_RUN.value:
        audit_status = "dry_run_pass" if report.valid else "dry_run_fail"
    else:
        audit_status = "read_only_integrity_pass" if report.valid else "read_only_integrity_fail"
    payload = {
        "command": "audit-data",
        "report": report.to_dict(),
        "status": audit_status,
    }
    _emit(payload, as_json=args.json)
    return EXIT_SUCCESS if report.valid else EXIT_VALIDATION


def _validate_artifacts(args: argparse.Namespace) -> int:
    report = validate_artifact_directory(
        args.artifact_root,
        require_artifacts=args.require_artifacts,
    )
    payload: MappingLike = {
        "command": "validate-artifacts",
        "report": report.to_dict(),
        "status": "pass" if report.valid else "fail",
    }
    _emit(payload, as_json=args.json)
    return EXIT_SUCCESS if report.valid else EXIT_VALIDATION


def _gated(args: argparse.Namespace) -> int:
    payload = gated_command_payload(args.command)
    _emit(payload, as_json=args.json)
    return EXIT_GATE_CLOSED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inclusive-shift-har",
        description="Leakage-safe, evidence-gated InclusiveShift-HAR research commands.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser(
        "validate-manifests",
        help="validate immutable dataset provenance manifests without network access",
    )
    manifest_parser.add_argument("paths", nargs="*", help="optional manifest files or directories")
    manifest_parser.add_argument("--manifest-root", default="manifests/datasets")
    manifest_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    manifest_parser.set_defaults(handler=_validate_manifests)

    audit_parser = subparsers.add_parser(
        "audit-data",
        help="plan a no-access audit or run an explicitly gated read-only integrity check",
    )
    audit_parser.add_argument("--manifest", required=True, help="dataset manifest JSON path")
    mode_group = audit_parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        action="store_const",
        const=DataAuditMode.DRY_RUN.value,
        dest="mode",
        help="perform no raw-data or filesystem access (default)",
    )
    mode_group.add_argument(
        "--read-only",
        action="store_const",
        const=DataAuditMode.READ_ONLY.value,
        dest="mode",
        help="verify whole-file integrity after checking an explicit read gate",
    )
    audit_parser.set_defaults(mode=DataAuditMode.DRY_RUN.value)
    audit_parser.add_argument(
        "--data-root", help="external raw-data root; required for --read-only"
    )
    audit_parser.add_argument("--gate-record", help="matching raw-data read access gate JSON")
    audit_parser.add_argument(
        "--profile",
        choices=("integrity-only", "inclusivehar-v4"),
        default="integrity-only",
        help="select whole-file integrity only or the gated privacy-safe InclusiveHAR v4 content audit",
    )
    audit_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    audit_parser.set_defaults(handler=_audit_data)

    artifact_parser = subparsers.add_parser(
        "validate-artifacts",
        help="validate artifact lineage manifests and referenced hashes without mutation",
    )
    artifact_parser.add_argument("--artifact-root", default="results")
    artifact_parser.add_argument("--require-artifacts", action="store_true")
    artifact_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    artifact_parser.set_defaults(handler=_validate_artifacts)

    for command in _GATED_COMMANDS:
        gated_parser = subparsers.add_parser(command, help=_GATED_COMMANDS[command]["message"])
        gated_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
        gated_parser.add_argument(
            "arguments",
            nargs=argparse.REMAINDER,
            help="reserved for the later validated implementation; currently ignored",
        )
        gated_parser.set_defaults(handler=_gated)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a stable process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except BrokenPipeError:
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
