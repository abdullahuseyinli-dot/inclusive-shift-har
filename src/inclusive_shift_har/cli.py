"""Command-line boundary for the gated InclusiveShift-HAR workflow."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from inclusive_shift_har import __version__
from inclusive_shift_har.artifacts.validation import validate_artifact_directory
from inclusive_shift_har.data.audit import DataAuditMode, DataGateError, audit_manifest_data
from inclusive_shift_har.data.inclusivehar import (
    InclusiveHARAuditError,
    audit_inclusivehar_v4_dataset,
)
from inclusive_shift_har.manifests.canonical import load_json_strict
from inclusive_shift_har.manifests.validation import (
    ManifestValidationError,
    validate_manifest_directory,
    validate_manifest_file,
)
from inclusive_shift_har.protocols.audit import (
    audit_split_manifest_file,
    write_split_audit_new,
)
from inclusive_shift_har.protocols.splits import (
    build_released_block_split_manifest,
    build_source_window_manifest,
    write_source_window_manifest_new,
    write_split_manifest_new,
)

EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_VALIDATION = 2
EXIT_GATE_CLOSED = 3

_GATED_COMMANDS: dict[str, dict[str, str]] = {
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


def _build_splits(args: argparse.Namespace) -> int:
    try:
        manifest = build_released_block_split_manifest(
            audit_report_path=args.audit_report,
            protocol_config_path=args.protocol_config,
            ontology_config_path=args.ontology_config,
            preprocessing_config_path=args.preprocessing_config,
            authorization_path=args.authorization,
        )
        published = write_split_manifest_new(
            manifest,
            args.output,
            allowed_root=args.allowed_root,
        )
    except (OSError, PermissionError, ValueError) as exc:
        payload: MappingLike = {
            "code": "SPLIT_BUILD_VALIDATION_ERROR",
            "command": "build-splits",
            "message": str(exc),
            "status": "fail",
        }
        _emit(payload, as_json=args.json)
        return EXIT_VALIDATION
    payload = {
        "command": "build-splits",
        "output": published.as_posix(),
        "split_manifest_sha256": manifest["split_manifest_sha256"],
        "status": "built_conditional_released_block",
        "target_performance_or_prediction_accessed": False,
        "window_count": manifest["window_count"],
    }
    _emit(payload, as_json=args.json)
    return EXIT_SUCCESS


def _audit_splits(args: argparse.Namespace) -> int:
    try:
        report = audit_split_manifest_file(
            split_manifest_path=args.split_manifest,
            audit_report_path=args.audit_report,
            protocol_config_path=args.protocol_config,
            ontology_config_path=args.ontology_config,
            preprocessing_config_path=args.preprocessing_config,
            authorization_path=args.authorization,
        )
        published: Path | None = None
        if args.output is not None:
            published = write_split_audit_new(
                report,
                args.output,
                allowed_root=args.allowed_root,
            )
    except (OSError, PermissionError, ValueError) as exc:
        payload: MappingLike = {
            "code": "SPLIT_AUDIT_VALIDATION_ERROR",
            "command": "audit-splits",
            "message": str(exc),
            "status": "fail",
        }
        _emit(payload, as_json=args.json)
        return EXIT_VALIDATION
    payload = {
        "command": "audit-splits",
        "output": published.as_posix() if published is not None else None,
        "report": report,
        "status": report["status"],
    }
    _emit(payload, as_json=args.json)
    return EXIT_SUCCESS if report["valid"] else EXIT_VALIDATION


def _build_source_windows(args: argparse.Namespace) -> int:
    try:
        parsed = load_json_strict(args.split_manifest)
        if not isinstance(parsed, Mapping):
            raise ValueError("split manifest root must be an object")
        manifest = build_source_window_manifest(parsed)
        published = write_source_window_manifest_new(
            manifest,
            args.output,
            allowed_root=args.allowed_root,
        )
    except (OSError, PermissionError, ValueError) as exc:
        payload: MappingLike = {
            "code": "SOURCE_WINDOW_MANIFEST_VALIDATION_ERROR",
            "command": "build-source-windows",
            "message": str(exc),
            "status": "fail",
        }
        _emit(payload, as_json=args.json)
        return EXIT_VALIDATION
    payload = {
        "command": "build-source-windows",
        "output": published.as_posix(),
        "source_window_count": manifest["source_window_count"],
        "source_window_manifest_sha256": manifest["source_window_manifest_sha256"],
        "status": "built_source_only_materialization_manifest",
        "target_performance_or_prediction_accessed": False,
        "target_subject_or_window_records_included": False,
    }
    _emit(payload, as_json=args.json)
    return EXIT_SUCCESS


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

    split_defaults = {
        "audit_report": "results/data_audit/inclusivehar_v4.audit.json",
        "authorization": "results/protocol/released_block_deviation_authorization.json",
        "ontology_config": "configs/ontologies/inclusivehar_v1_1.yaml",
        "preprocessing_config": "configs/preprocessing/inclusivehar_primary_128.yaml",
        "protocol_config": "configs/protocols/inclusivehar_released_block_v1_2.yaml",
    }
    build_split_parser = subparsers.add_parser(
        "build-splits",
        help="build the authorized deterministic participant-exclusive released-block split",
    )
    build_split_parser.add_argument("--audit-report", default=split_defaults["audit_report"])
    build_split_parser.add_argument("--protocol-config", default=split_defaults["protocol_config"])
    build_split_parser.add_argument("--ontology-config", default=split_defaults["ontology_config"])
    build_split_parser.add_argument(
        "--preprocessing-config", default=split_defaults["preprocessing_config"]
    )
    build_split_parser.add_argument("--authorization", default=split_defaults["authorization"])
    build_split_parser.add_argument(
        "--output",
        default="results/protocol/splits/inclusivehar_v4_released_block_v1_2.json",
    )
    build_split_parser.add_argument("--allowed-root", default="results/protocol")
    build_split_parser.add_argument("--json", action="store_true")
    build_split_parser.set_defaults(handler=_build_splits)

    audit_split_parser = subparsers.add_parser(
        "audit-splits",
        help="rebuild and independently audit a released-block split without reading signals",
    )
    audit_split_parser.add_argument("--split-manifest", required=True)
    audit_split_parser.add_argument("--audit-report", default=split_defaults["audit_report"])
    audit_split_parser.add_argument("--protocol-config", default=split_defaults["protocol_config"])
    audit_split_parser.add_argument("--ontology-config", default=split_defaults["ontology_config"])
    audit_split_parser.add_argument(
        "--preprocessing-config", default=split_defaults["preprocessing_config"]
    )
    audit_split_parser.add_argument("--authorization", default=split_defaults["authorization"])
    audit_split_parser.add_argument("--output")
    audit_split_parser.add_argument("--allowed-root", default="results/protocol")
    audit_split_parser.add_argument("--json", action="store_true")
    audit_split_parser.set_defaults(handler=_audit_splits)

    source_window_parser = subparsers.add_parser(
        "build-source-windows",
        help="derive a source-only materialization manifest from a sealed audited split",
    )
    source_window_parser.add_argument("--split-manifest", required=True)
    source_window_parser.add_argument("--output", required=True)
    source_window_parser.add_argument("--allowed-root", default="results/protocol")
    source_window_parser.add_argument("--json", action="store_true")
    source_window_parser.set_defaults(handler=_build_source_windows)

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
