"""Command-line boundary for the gated InclusiveShift-HAR workflow."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
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

_TRAIN_TRACKS: dict[str, str] = {
    "inclusivehar-source": "run one participant-exclusive InclusiveHAR source-development job",
    "fuse-reframe-source": "run one target-sealed FuSE-ReFrame v2 source-development job",
    "fuse-reframe-nested": "run one target-sealed nested FuSE-ReFrame outer source fold",
    "fuse-reframe-corruptions": "diagnose one FuSE-ReFrame checkpoint on source corruptions",
    "fuse-reframe-router": "run one target-sealed nested heterogeneous OOF router fold",
    "fuse-reframe-committee": "reanalyze frozen inner checkpoints as an equal-weight committee",
    "multirocket-source": "run the fixed target-sealed MultiRocket grouped source control",
    "spectral-shape-nested": "run deterministic nested SpectralShape source development",
    "crossfit-expert-stack": "cross-fit frozen SpectralShape and neural committee experts",
    "aeon-source-control": "run one fixed target-sealed modern time-series control",
    "geometric-pyramid-nested": "run nested Geometric Spectral Pyramid source development",
    "geometric-pyramid-corruptions": "replay frozen geometric models on source corruptions",
    "geometric-pyramid-seed-ensemble": "evaluate the fixed five-seed geometric probability ensemble",
    "robust-multiscale-nested": "run nested Robust Multiscale Residual Pyramid development",
    "robust-multiscale-corruptions": "replay frozen robust multiscale models on corruptions",
    "gravity-anchored-nested": "run nested gravity sensor-sufficiency development",
    "microstate-posture-select": "freeze MPG-RMRP candidates using nested inner participants",
    "microstate-posture-evaluate": "evaluate frozen MPG-RMRP candidates at one locked seed",
    "microstate-posture-ablations": "run the frozen explanatory MPG-RMRP ablation suite",
    "ctgr-select": "freeze confidence-triggered gravity residual candidates",
    "ctgr-evaluate": "evaluate frozen confidence-triggered gravity residual candidates",
    "cage-har-development": "run CAGE-HAR on a guarded new-development OOF bundle",
    "cage-har-synthetic-smoke": "exercise CAGE-HAR on non-scientific synthetic data",
    "active-semantic-sentinel": "evaluate active labelled posture-semantic personalization",
    "provenance-mask-robustness": "replay RMRP with explicit validity-mask reconstruction",
    "semantic-anchor-reconciliation": "evaluate labelled posture-semantic personalization",
    "fixed-oof-fusion": "evaluate fixed pooling of frozen target-sealed OOF experts",
    "daghar-augmented-nested": "run nested source selection with DAGHAR development support",
    "daghar-external-freeze": "freeze one source candidate before held-out DAGHAR evaluation",
    "daghar-external-evaluate": "consume the one held-out DAGHAR performance opening",
    "final-source-suite": "run the predeclared final source-only model/seed suite sequentially",
    "uci-source-fold": "run one official-train-only grouped UCI-HAR fold",
    "few-person": "build or run a post-confirmatory few-person scenario",
    "within-group": ("run one cache-backed post-confirmatory disabled-cohort grouped-CV cell"),
    "raw-total-sensitivity": (
        "run the predeclared post-confirmatory raw/total-acceleration sensitivity"
    ),
}

_EVALUATE_TRACKS: dict[str, str] = {
    "source-cv": "validate and aggregate source-only grouped-CV evidence",
    "uci-source": "validate and aggregate official-train-only UCI-HAR reproduction evidence",
    "few-person-statistics": "validate progress or aggregate few-person participant statistics",
    "within-group-statistics": (
        "validate and aggregate disabled-cohort grouped-CV participant statistics"
    ),
    "efficiency": "aggregate frozen post-confirmatory efficiency profiles",
    "sensor-stress": "aggregate frozen post-confirmatory sensor-stress evidence",
    "raw-total-statistics": "aggregate raw/total-acceleration sensitivity evidence",
    "microstate-posture-summary": "aggregate the frozen five-seed MPG-RMRP evaluation",
    "ctgr-summary": "aggregate the frozen five-seed CTGR sensor-sufficiency evaluation",
}

_INCLUSIVEHAR_NATIVE_CPU_MODELS = frozenset({"random_forest", "svm_rbf", "logistic_regression"})
_INCLUSIVEHAR_XGBOOST_MODEL = "xgboost"

WorkflowMain = Callable[[list[str] | None], int]


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


def _aggregate_source_cv(args: argparse.Namespace) -> int:
    try:
        from inclusive_shift_har.evaluation.source_cv import (
            SourceCVAggregationError,
            build_source_cv_aggregate,
            write_source_cv_exports_new,
        )

        aggregate = build_source_cv_aggregate(
            fold_record_dir=args.fold_record_dir,
            source_development_dir=args.source_development_dir,
            source_manifest_path=args.source_manifest,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_confidence=args.bootstrap_confidence,
            bootstrap_seed=args.bootstrap_seed,
        )
        outputs = write_source_cv_exports_new(
            aggregate,
            output_dir=args.output_dir,
            prefix=args.prefix,
        )
        published = load_json_strict(outputs["json"])
        if not isinstance(published, Mapping):
            raise SourceCVAggregationError("published aggregate root is not an object")
    except (FileExistsError, OSError, SourceCVAggregationError, ValueError) as exc:
        payload: MappingLike = {
            "code": "SOURCE_CV_AGGREGATION_VALIDATION_ERROR",
            "command": "aggregate-source-cv",
            "message": str(exc),
            "status": "fail",
        }
        _emit(payload, as_json=args.json)
        return EXIT_VALIDATION
    payload = {
        "aggregate_record_sha256": published["aggregate_record_sha256"],
        "command": "aggregate-source-cv",
        "configuration_count": published["configuration_count"],
        "outputs": {key: value.as_posix() for key, value in outputs.items()},
        "quarantined_configuration_count": published["quarantined_configuration_count"],
        "status": published["status"],
        "target_performance_or_prediction_accessed": False,
        "target_subject_or_window_records_loaded": False,
    }
    _emit(payload, as_json=args.json)
    return EXIT_SUCCESS


def _run_uci_source_fold(args: argparse.Namespace) -> int:
    try:
        from inclusive_shift_har.experiments.uci_source import run_uci_source_fold

        summary = run_uci_source_fold(
            archive_path=Path(args.archive),
            dataset_manifest_path=Path(args.dataset_manifest),
            protocol_path=Path(args.protocol),
            model_name=args.model,
            fold_id=args.fold_id,
            seed=args.seed,
            attempt=args.attempt,
            code_commit=args.code_commit,
            repository_root=Path(args.repository_root),
            experiment_config_path=Path(args.experiment_config),
            expected_experiment_config_file_sha256=(args.expected_experiment_config_file_sha256),
            run_directory=Path(args.run_directory),
            summary_path=Path(args.summary),
            allowed_output_root=Path(args.allowed_output_root),
        )
    except (FileExistsError, OSError, PermissionError, RuntimeError, ValueError) as exc:
        payload: MappingLike = {
            "code": "UCI_SOURCE_RUN_VALIDATION_OR_EXECUTION_ERROR",
            "command": "run-uci-source-fold",
            "message": str(exc),
            "status": "fail",
        }
        _emit(payload, as_json=args.json)
        return EXIT_VALIDATION
    payload = {
        "command": "run-uci-source-fold",
        "fold_id": summary["fold"]["fold_id"],
        "model_name": summary["model_name"],
        "record_sha256": summary["record_sha256"],
        "status": summary["status"],
        "summary": Path(args.summary).as_posix(),
        "official_test_performance_or_prediction_accessed": False,
        "inclusivehar_data_or_target_accessed": False,
    }
    _emit(payload, as_json=args.json)
    return EXIT_SUCCESS


def _aggregate_uci_source(args: argparse.Namespace) -> int:
    try:
        from inclusive_shift_har.evaluation.uci_reporting import (
            UCIReportingError,
            build_uci_reproduction_report,
            write_uci_reproduction_exports_new,
        )

        report = build_uci_reproduction_report(
            record_directory=args.record_directory,
            protocol_path=args.protocol,
            experiment_config_path=args.experiment_config,
            expected_experiment_config_file_sha256=(args.expected_experiment_config_file_sha256),
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
        outputs = write_uci_reproduction_exports_new(
            report,
            output_directory=args.output_directory,
        )
    except (FileExistsError, OSError, UCIReportingError, ValueError) as exc:
        payload: MappingLike = {
            "code": "UCI_SOURCE_AGGREGATION_VALIDATION_ERROR",
            "command": "aggregate-uci-source",
            "message": str(exc),
            "status": "fail",
        }
        _emit(payload, as_json=args.json)
        return EXIT_VALIDATION
    payload = {
        "command": "aggregate-uci-source",
        "official_test_member_opened": False,
        "official_test_performance_or_prediction_accessed": False,
        "inclusivehar_data_or_target_accessed": False,
        "outputs": {key: value.as_posix() for key, value in outputs.items()},
        "record_sha256": report["record_sha256"],
        "status": report["status"],
    }
    _emit(payload, as_json=args.json)
    return EXIT_SUCCESS


def _workflow_entrypoint(
    workflow: str,
    track: str,
) -> tuple[WorkflowMain, tuple[str, ...], bool]:
    """Resolve one explicitly allowlisted workflow without importing target-opening code."""

    if workflow == "train":
        if track == "inclusivehar-source":
            from inclusive_shift_har.experiments.inclusivehar_source import main as entrypoint

            return entrypoint, (), False
        if track == "fuse-reframe-source":
            from inclusive_shift_har.experiments.fuse_reframe_source import main as entrypoint

            return entrypoint, (), False
        if track == "fuse-reframe-nested":
            from inclusive_shift_har.experiments.fuse_reframe_nested import main as entrypoint

            return entrypoint, (), False
        if track == "fuse-reframe-corruptions":
            from inclusive_shift_har.experiments.fuse_reframe_corruptions import main as entrypoint

            return entrypoint, (), False
        if track == "fuse-reframe-router":
            from inclusive_shift_har.experiments.fuse_reframe_router import main as entrypoint

            return entrypoint, (), False
        if track == "fuse-reframe-committee":
            from inclusive_shift_har.experiments.fuse_reframe_committee import main as entrypoint

            return entrypoint, (), False
        if track == "multirocket-source":
            from inclusive_shift_har.experiments.multirocket_source import main as entrypoint

            return entrypoint, (), False
        if track == "spectral-shape-nested":
            from inclusive_shift_har.experiments.spectral_shape_nested import main as entrypoint

            return entrypoint, (), False
        if track == "crossfit-expert-stack":
            from inclusive_shift_har.experiments.crossfit_expert_stack import main as entrypoint

            return entrypoint, (), False
        if track == "aeon-source-control":
            from inclusive_shift_har.experiments.aeon_source_controls import main as entrypoint

            return entrypoint, (), False
        if track == "geometric-pyramid-nested":
            from inclusive_shift_har.experiments.geometric_spectral_pyramid_nested import (
                main as entrypoint,
            )

            return entrypoint, (), False
        if track == "geometric-pyramid-corruptions":
            from inclusive_shift_har.experiments.geometric_pyramid_corruptions import (
                main as entrypoint,
            )

            return entrypoint, (), False
        if track == "geometric-pyramid-seed-ensemble":
            from inclusive_shift_har.experiments.geometric_pyramid_seed_ensemble import (
                main as entrypoint,
            )

            return entrypoint, (), False
        if track == "robust-multiscale-nested":
            from inclusive_shift_har.experiments.robust_multiscale_residual_nested import (
                main as entrypoint,
            )

            return entrypoint, (), False
        if track == "robust-multiscale-corruptions":
            from inclusive_shift_har.experiments.robust_multiscale_corruptions import (
                main as entrypoint,
            )

            return entrypoint, (), False
        if track == "gravity-anchored-nested":
            from inclusive_shift_har.experiments.gravity_anchored_geometric_nested import (
                main as entrypoint,
            )

            return entrypoint, (), False
        if track in {"microstate-posture-select", "microstate-posture-evaluate"}:
            from inclusive_shift_har.experiments.microstate_posture_graph_nested import (
                main as entrypoint,
            )

            command = "select" if track.endswith("select") else "evaluate"
            return entrypoint, (command,), False
        if track == "microstate-posture-ablations":
            from inclusive_shift_har.experiments.microstate_posture_graph_ablations import (
                main as entrypoint,
            )

            return entrypoint, (), False
        if track in {"ctgr-select", "ctgr-evaluate"}:
            from inclusive_shift_har.experiments.confidence_triggered_gravity_residual import (
                main as entrypoint,
            )

            command = "select" if track.endswith("select") else "evaluate"
            return entrypoint, (command,), False
        if track in {"cage-har-development", "cage-har-synthetic-smoke"}:
            from inclusive_shift_har.experiments.cage_har import main as entrypoint

            command = "run" if track.endswith("development") else "synthetic-smoke"
            return entrypoint, (command,), False
        if track == "active-semantic-sentinel":
            from inclusive_shift_har.experiments.active_semantic_gauge_sentinel import (
                main as entrypoint,
            )

            return entrypoint, (), False
        if track == "provenance-mask-robustness":
            from inclusive_shift_har.experiments.provenance_mask_robustness import (
                main as entrypoint,
            )

            return entrypoint, (), False
        if track == "semantic-anchor-reconciliation":
            from inclusive_shift_har.experiments.semantic_anchor_reconciliation import (
                main as entrypoint,
            )

            return entrypoint, (), False
        if track == "fixed-oof-fusion":
            from inclusive_shift_har.experiments.fixed_oof_fusion import main as entrypoint

            return entrypoint, (), False
        if track == "daghar-augmented-nested":
            from inclusive_shift_har.experiments.daghar_augmented_geometric_nested import (
                main as entrypoint,
            )

            return entrypoint, (), False
        if track in {"daghar-external-freeze", "daghar-external-evaluate"}:
            from inclusive_shift_har.experiments.daghar_external_evaluation import (
                main as entrypoint,
            )

            command = "freeze" if track.endswith("freeze") else "evaluate"
            return entrypoint, (command,), False
        if track == "final-source-suite":
            from inclusive_shift_har.experiments.final_source_suite import main as entrypoint

            return entrypoint, (), False
        if track == "uci-source-fold":
            return main, ("run-uci-source-fold",), True
        if track == "few-person":
            from inclusive_shift_har.experiments.few_person import main as entrypoint

            return entrypoint, (), False
        if track == "within-group":
            from inclusive_shift_har.experiments.within_group import main as entrypoint

            return entrypoint, (), False
        if track == "raw-total-sensitivity":
            from inclusive_shift_har.experiments.raw_total_acceleration import main as entrypoint

            return entrypoint, (), False
    elif workflow == "evaluate":
        if track == "source-cv":
            return main, ("aggregate-source-cv",), True
        if track == "uci-source":
            return main, ("aggregate-uci-source",), True
        if track == "few-person-statistics":
            from inclusive_shift_har.evaluation.few_person_statistics import main as entrypoint

            return entrypoint, (), False
        if track == "within-group-statistics":
            from inclusive_shift_har.evaluation.within_group_statistics import main as entrypoint

            return entrypoint, (), False
        if track in {"efficiency", "sensor-stress"}:
            from inclusive_shift_har.evaluation.secondary_aggregation import main as entrypoint

            return entrypoint, (track,), False
        if track == "raw-total-statistics":
            from inclusive_shift_har.evaluation.raw_total_reporting import main as entrypoint

            return entrypoint, (), False
        if track == "microstate-posture-summary":
            from inclusive_shift_har.experiments.microstate_posture_graph_nested import (
                main as entrypoint,
            )

            return entrypoint, ("summarize",), False
        if track == "ctgr-summary":
            from inclusive_shift_har.experiments.confidence_triggered_gravity_residual import (
                main as entrypoint,
            )

            return entrypoint, ("summary",), False
    raise KeyError(track)


def _workflow_error(
    *,
    workflow: str,
    track: str | None,
    message: str,
    code: str,
    as_json: bool,
) -> int:
    payload: MappingLike = {
        "code": code,
        "command": workflow,
        "exit_code": EXIT_VALIDATION,
        "message": message,
        "one_time_confirmatory_target_operation_exposed": False,
        "status": "fail",
        "track": track,
    }
    _emit(payload, as_json=as_json)
    return EXIT_VALIDATION


def _forwarded_option(arguments: Sequence[str], option: str) -> str | None:
    """Return argparse's last option value without consuming or rewriting argv."""

    value: str | None = None
    prefix = f"{option}="
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            break
        if argument.startswith(prefix):
            value = argument[len(prefix) :]
        elif argument == option and index + 1 < len(arguments):
            candidate = arguments[index + 1]
            if candidate != "--" and not candidate.startswith("--"):
                value = candidate
                index += 1
        index += 1
    return value


def _inclusivehar_source_device_policy_error(arguments: Sequence[str]) -> str | None:
    """Keep neural and XGBoost routed jobs on CUDA while preserving native CPU methods."""

    if _forwarded_option(arguments, "--device") != "cpu":
        return None
    model = _forwarded_option(arguments, "--model")
    if model in _INCLUSIVEHAR_NATIVE_CPU_MODELS:
        return None
    if model == _INCLUSIVEHAR_XGBOOST_MODEL:
        return "XGBoost must use --device cuda in the unified InclusiveHAR source route"
    return (
        "neural InclusiveHAR source models must use --device cuda; CPU is reserved for the "
        "native random_forest, svm_rbf, and logistic_regression baselines"
    )


def _dispatch_workflow(args: argparse.Namespace) -> int:
    """Forward arguments to a current safe module entry point without shell interpretation."""

    workflow = str(args.workflow)
    track = None if args.track is None else str(args.track)
    tracks = _TRAIN_TRACKS if workflow == "train" else _EVALUATE_TRACKS
    as_json = bool(args.json)
    if track is None:
        return _workflow_error(
            workflow=workflow,
            track=None,
            message=f"{workflow} requires one of these tracks: {', '.join(tracks)}",
            code=f"{workflow.upper()}_TRACK_REQUIRED",
            as_json=as_json,
        )
    if track not in tracks:
        return _workflow_error(
            workflow=workflow,
            track=track,
            message=f"unknown {workflow} track {track!r}; choose one of: {', '.join(tracks)}",
            code=f"UNKNOWN_{workflow.upper()}_TRACK",
            as_json=as_json,
        )
    forwarded = [str(value) for value in args.arguments]
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    if workflow == "train" and track == "inclusivehar-source":
        policy_error = _inclusivehar_source_device_policy_error(forwarded)
        if policy_error is not None:
            return _workflow_error(
                workflow=workflow,
                track=track,
                message=policy_error,
                code="TRAIN_TRACK_DEVICE_POLICY_ERROR",
                as_json=as_json,
            )
    try:
        entrypoint, prefix, accepts_json = _workflow_entrypoint(workflow, track)
        child_arguments = [*prefix, *forwarded]
        if as_json and accepts_json and "--json" not in child_arguments:
            child_arguments.append("--json")
        return int(entrypoint(child_arguments))
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else EXIT_ERROR
        if exit_code != EXIT_SUCCESS and as_json:
            _workflow_error(
                workflow=workflow,
                track=track,
                message="forwarded arguments were rejected; see the track usage on stderr",
                code=f"{workflow.upper()}_TRACK_ARGUMENT_ERROR",
                as_json=True,
            )
        return exit_code
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return _workflow_error(
            workflow=workflow,
            track=track,
            message=str(exc),
            code=f"{workflow.upper()}_TRACK_EXECUTION_ERROR",
            as_json=as_json,
        )


def _workflow_epilog(tracks: Mapping[str, str], *, locked_report_note: bool = False) -> str:
    lines = ["Available tracks:"]
    lines.extend(f"  {name:<24} {description}" for name, description in tracks.items())
    lines.extend(
        [
            "",
            "Arguments after TRACK are forwarded as an argv list without shell execution.",
            "Use `--` before a forwarded `--help` request, for example: TRACK -- --help.",
            "The one-time confirmatory target-opening operation is never routed here.",
        ]
    )
    if locked_report_note:
        lines.extend(
            [
                "",
                "No locked-report track is exposed: the current publication-report API is",
                "create-only and has no public read-only validator. `validate-artifacts` remains",
                "available for repository artifact manifests.",
            ]
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inclusive-shift-har",
        description=(
            "Participant-exclusive, evidence-gated InclusiveShift-HAR research commands; "
            "InclusiveHAR released-block windows are not trial-safe."
        ),
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

    source_cv_parser = subparsers.add_parser(
        "aggregate-source-cv",
        help="validate and aggregate complete source-only grouped CV evidence",
    )
    source_cv_parser.add_argument("--fold-record-dir", default="results/development/source_cv_v1")
    source_cv_parser.add_argument(
        "--source-development-dir", default="results/development/source_v1"
    )
    source_cv_parser.add_argument(
        "--source-manifest",
        default=("results/protocol/source_windows/inclusivehar_v4_source_development_v1_2.json"),
    )
    source_cv_parser.add_argument("--output-dir", default="results/development/source_cv_v1")
    source_cv_parser.add_argument("--prefix", default="source_cv_aggregate")
    source_cv_parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    source_cv_parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    source_cv_parser.add_argument("--bootstrap-seed", type=int, default=1729)
    source_cv_parser.add_argument("--json", action="store_true")
    source_cv_parser.set_defaults(handler=_aggregate_source_cv)

    uci_source_parser = subparsers.add_parser(
        "run-uci-source-fold",
        help="run one corrected official-train-only UCI-HAR grouped fold on CUDA",
    )
    uci_source_parser.add_argument("--archive", required=True)
    uci_source_parser.add_argument(
        "--dataset-manifest", default="manifests/datasets/uci_har_v1.json"
    )
    uci_source_parser.add_argument(
        "--protocol", default="results/protocol/uci_har_source_grouped_v1.json"
    )
    uci_source_parser.add_argument(
        "--model",
        required=True,
        choices=(
            "legacy_cnn1d_h128",
            "legacy_bilstm_h192",
            "legacy_joint_bilstm256_cnn128",
        ),
    )
    uci_source_parser.add_argument(
        "--fold-id",
        required=True,
        choices=tuple(f"uci_source_cv_{index:02d}" for index in range(1, 6)),
    )
    uci_source_parser.add_argument("--seed", type=int, required=True)
    uci_source_parser.add_argument("--attempt", type=int, required=True)
    uci_source_parser.add_argument("--code-commit", required=True)
    uci_source_parser.add_argument("--repository-root", default=".")
    uci_source_parser.add_argument(
        "--experiment-config",
        default="configs/experiments/uci_har_corrected_reproduction_v1_1.yaml",
    )
    uci_source_parser.add_argument(
        "--expected-experiment-config-file-sha256",
        required=True,
    )
    uci_source_parser.add_argument("--run-directory", required=True)
    uci_source_parser.add_argument("--summary", required=True)
    uci_source_parser.add_argument("--allowed-output-root", default="results")
    uci_source_parser.add_argument("--json", action="store_true")
    uci_source_parser.set_defaults(handler=_run_uci_source_fold)

    uci_aggregate_parser = subparsers.add_parser(
        "aggregate-uci-source",
        help="verify and aggregate the complete CUDA UCI official-train grouped-CV matrix",
    )
    uci_aggregate_parser.add_argument(
        "--record-directory",
        default="results/legacy_reproduction/uci_har_source_grouped_v1/records",
    )
    uci_aggregate_parser.add_argument(
        "--protocol", default="results/protocol/uci_har_source_grouped_v1.json"
    )
    uci_aggregate_parser.add_argument(
        "--experiment-config",
        default="configs/experiments/uci_har_corrected_reproduction_v1_1.yaml",
    )
    uci_aggregate_parser.add_argument(
        "--expected-experiment-config-file-sha256",
        required=True,
    )
    uci_aggregate_parser.add_argument(
        "--output-directory",
        default="results/legacy_reproduction/uci_har_source_grouped_v1",
    )
    uci_aggregate_parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    uci_aggregate_parser.add_argument("--bootstrap-seed", type=int, default=1729)
    uci_aggregate_parser.add_argument("--json", action="store_true")
    uci_aggregate_parser.set_defaults(handler=_aggregate_uci_source)

    train_parser = subparsers.add_parser(
        "train",
        help="dispatch an allowlisted source or post-confirmatory training track",
        description=(
            "Dispatch a current training module. This namespace cannot invoke the consumed "
            "one-time confirmatory target-opening operation."
        ),
        epilog=_workflow_epilog(_TRAIN_TRACKS),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    train_parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable dispatcher errors; routed direct commands also receive it",
    )
    train_parser.add_argument("track", nargs="?", metavar="TRACK")
    train_parser.add_argument("arguments", nargs=argparse.REMAINDER, metavar="ARG")
    train_parser.set_defaults(handler=_dispatch_workflow, workflow="train")

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="dispatch an allowlisted aggregation or read-only validation track",
        description=(
            "Dispatch current derivation and validation modules over existing evidence. This "
            "namespace never invokes a target opening or target inference operation."
        ),
        epilog=_workflow_epilog(_EVALUATE_TRACKS, locked_report_note=True),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    evaluate_parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable dispatcher errors; routed direct commands also receive it",
    )
    evaluate_parser.add_argument("track", nargs="?", metavar="TRACK")
    evaluate_parser.add_argument("arguments", nargs=argparse.REMAINDER, metavar="ARG")
    evaluate_parser.set_defaults(handler=_dispatch_workflow, workflow="evaluate")
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
