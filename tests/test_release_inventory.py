from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from inclusive_shift_har.artifacts.github_evidence import (
    COMPONENT_FILES,
    PROVENANCE_FILES,
)
from inclusive_shift_har.artifacts.release_gate import (
    audit_licenses,
    scan_index,
    scan_repository,
    validate_secret_scan,
)
from inclusive_shift_har.artifacts.release_inventory import (
    CANONICAL_REQUIRED_ROLE_PATHS,
    GATE_NAMES,
    POSTCONFIRMATORY_TRACKS,
    REQUIRED_ROLES,
    ReleaseEvidenceError,
    _new_output,
    generate_release_evidence_inventory,
    validate_release_evidence_inventory_file,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256


@dataclass(frozen=True)
class ReleaseWorkspace:
    repository: Path
    spec_path: Path
    output_root: Path
    commit: str
    spec: dict[str, Any]


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _git_bytes(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _self_hashed_record(**values: Any) -> dict[str, Any]:
    record = dict(values)
    record["record_sha256"] = canonical_json_sha256(record)
    return record


def _blob(repository: Path, commit: str, relative: str) -> bytes:
    return _git_bytes(repository, "cat-file", "blob", f"{commit}:{relative}")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory_schema_validator(repository: Path) -> Draft202012Validator:
    schema = json.loads(
        (repository / "configs/schema/release_evidence_inventory.schema.json").read_text(
            encoding="utf-8"
        )
    )
    return Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER)


def _build_workspace(
    tmp_path: Path,
    *,
    requested_status: str = "draft",
    bad_opening_self_hash: bool = False,
    opening_number: int | float | bool = 1,
    unlock_opening_number: int | float | bool = 1,
    maximum_target_openings: int | float | bool = 1,
    bad_opening_kind: bool = False,
    bad_target_cross_hash: bool = False,
    extra_opening_field: bool = False,
    extra_target_field: bool = False,
    wrong_report_parent: bool = False,
    wrong_report_policy: bool = False,
    bad_machine_binding: bool = False,
    bad_freeze_binding: bool = False,
) -> ReleaseWorkspace:
    repository = tmp_path / "repository"
    repository.mkdir(parents=True)
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.email", "synthetic@example.invalid")
    _git(repository, "config", "user.name", "Synthetic Release Test")
    _git(repository, "config", "core.autocrlf", "false")
    (repository / ".gitignore").write_text(".audit/\nrelease/inventory.json\n", encoding="utf-8")
    (repository / ".gitleaks.toml").write_bytes(
        b'title = "synthetic release fixture"\n[extend]\nuseDefault = true\n'
    )
    (repository / ".gitleaksignore").write_bytes(b"# No synthetic suppressions.\n")
    policy_path = repository / "configs/release/release_gate_policy_v1.json"
    _write_json(
        policy_path,
        {
            "schema_version": "1.0.0",
            "policy_kind": "final_release_gate_policy",
            "policy_id": "synthetic-release-policy",
            "staged_index_scan": {
                "required_before_release_candidate_commit": True,
                "scan_scope": "exact_git_index",
            },
            "git_refs": {
                "non_tag_refs_must_target_commits": True,
                "tags_must_be_annotated_direct_to_commits": True,
                "required_for_release_tags": ["protocol-v1.2.0"],
                "pinned_annotated_tags": {},
                "permitted_candidate_tags": {
                    "protocol-v1.2.0": {
                        "message": "synthetic locked protocol",
                        "tagger_name": "Synthetic Release Test",
                        "tagger_email": "synthetic@example.invalid",
                    }
                },
                "historical_notes": {},
            },
            "maximum_blob_size_bytes": 16 * 1024 * 1024,
            "forbidden_path_prefixes": [".audit/", "data/raw/"],
            "forbidden_basenames": [
                ".env",
                "credentials.json",
                "secrets.json",
                "secrets.yaml",
                "secrets.yml",
            ],
            "forbidden_basename_prefixes": [".env."],
            "allowed_sensitive_paths": [".env.example"],
            "forbidden_suffixes": [".zip", ".npz", ".onnx", ".pt"],
            "disguised_binary_signatures": {"zip": "504b0304", "pdf": "25504446"},
            "allowed_signature_suffixes": {"pdf": [".pdf"]},
            "allowed_binary_suffixes": [".pdf"],
            "gitleaks": {
                "version": "8.30.1",
                "config_path": ".gitleaks.toml",
                "config_sha256": hashlib.sha256(
                    b'title = "synthetic release fixture"\n[extend]\nuseDefault = true\n'
                ).hexdigest(),
                "ignore_path": ".gitleaksignore",
                "reviewed_ignore_entries": [],
                "required_scope": "complete_commit_history_with_policy_pinned_tag_metadata",
            },
            "prohibited_license_tokens": ["gpl", "unknown", "proprietary"],
            "license_exceptions": [],
        },
    )
    (repository / "README.md").write_text("# Synthetic release fixture\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "synthetic base")
    base_commit = _git(repository, "rev-parse", "HEAD").strip()

    machine_record = _self_hashed_record(
        schema_version="1.0.0",
        record_kind="confirmatory_machine_environment",
        status="observed_read_only_cuda_validated",
    )
    machine_path = repository / "results/environment/confirmatory_machine_20260824.json"
    _write_json(machine_path, machine_record)
    unlock_record = _self_hashed_record(
        schema_version="1.0.0",
        gate="final_evaluation_unlock",
        status="approved",
        target_opening_number=unlock_opening_number,
        maximum_target_openings=maximum_target_openings,
        target_performance_previously_accessed=False,
        target_subject_or_window_records_loaded_at_approval=False,
        target_predictions_or_performance_accessed_at_approval=False,
        source_models_and_calibrators_frozen=True,
        target_seal_id="a" * 64,
        split_manifest_sha256="b" * 64,
        final_freeze_inventory_sha256="c" * 64,
        frozen_artifact_set_sha256="d" * 64,
    )
    unlock_path = repository / "results/protocol/final_evaluation_unlock_v1.json"
    _write_json(unlock_path, unlock_record)

    paths: dict[str, str] = {}
    opening_record: dict[str, Any] | None = None
    opening_payload: bytes | None = None
    for role in REQUIRED_ROLES:
        relative = CANONICAL_REQUIRED_ROLE_PATHS.get(
            role,
            (
                f"evidence/{role}.json"
                if role in {"opening_receipt", "locked_target_index"}
                else f"evidence/{role}.md"
            ),
        )
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if role == "opening_receipt":
            record = _self_hashed_record(
                schema_version="1.0.0",
                record_kind=(
                    "wrong_opening_receipt"
                    if bad_opening_kind
                    else "confirmatory_target_opening_receipt"
                ),
                status="unlock_consumed_before_materialization",
                target_opening_number=opening_number,
                target_predictions_or_performance_accessed_at_receipt_time=False,
                target_signals_materialized_at_receipt_time=False,
                target_seal_id="a" * 64,
                split_manifest_sha256="b" * 64,
                final_freeze_inventory_sha256="c" * 64,
                frozen_artifact_set_sha256="d" * 64,
                machine_record_sha256=(
                    "0" * 64 if bad_machine_binding else machine_record["record_sha256"]
                ),
                unlock_record_sha256=canonical_json_sha256(unlock_record),
                opened_at_utc="2026-08-24T11:58:00Z",
            )
            if extra_opening_field:
                record.pop("record_sha256")
                record["target_rerun_permitted"] = True
                record["record_sha256"] = canonical_json_sha256(record)
            if bad_opening_self_hash:
                record["record_sha256"] = "0" * 64
            _write_json(path, record)
            opening_record = record
            opening_payload = path.read_bytes()
        elif role == "source_artifact_freeze":
            _write_json(
                path,
                {
                    "schema_version": "1.0.0",
                    "record_kind": "final_source_artifact_freeze",
                    "status": "frozen_before_target_unlock",
                    "target_seal_id": "a" * 64,
                    "split_manifest_sha256": "9" * 64 if bad_freeze_binding else "b" * 64,
                    "inventory_sha256": "c" * 64,
                    "frozen_artifact_set_sha256": "d" * 64,
                    "target_state": {
                        "target_predictions_or_performance_accessed": False,
                        "target_subject_or_window_records_loaded": False,
                        "unlock_record": None,
                    },
                },
            )
        elif role == "split_manifest_and_audit":
            _write_json(
                path,
                {
                    "schema_version": "1.0.0",
                    "status": "pass_conditional_released_block",
                    "valid": True,
                    "target_performance_or_prediction_accessed": False,
                    "split_manifest_sha256": "b" * 64,
                },
            )
        elif role == "locked_target_index":
            assert opening_record is not None and opening_payload is not None
            record = _self_hashed_record(
                schema_version="1.0.0",
                record_kind="locked_target_evaluation_index",
                status="complete_create_only",
                target_opening_number=opening_number,
                target_information_used_for_model_selection=False,
                opening_receipt_record_sha256=opening_record["record_sha256"],
                opening_receipt_file_sha256=hashlib.sha256(opening_payload).hexdigest(),
                target_seal_id=(
                    "f" * 64 if bad_target_cross_hash else opening_record["target_seal_id"]
                ),
                split_manifest_sha256=opening_record["split_manifest_sha256"],
                final_freeze_inventory_sha256=opening_record["final_freeze_inventory_sha256"],
                frozen_artifact_set_sha256=opening_record["frozen_artifact_set_sha256"],
                class_names=["mobility", "sitting", "standing"],
                evidence_status="locked_confirmatory_target_opening_1",
                participant_count=1,
                window_count=1,
                model_seed_result_count=1,
                results=[
                    {
                        "array_path": "synthetic/predictions.npz",
                        "array_sha256": "1" * 64,
                        "model_id": "synthetic-model",
                        "record_file_sha256": "2" * 64,
                        "record_path": "synthetic/result.json",
                        "record_sha256": "3" * 64,
                        "seed": 11,
                    }
                ],
            )
            if extra_target_field:
                record.pop("record_sha256")
                record["target_rerun_permitted"] = True
                record["record_sha256"] = canonical_json_sha256(record)
            _write_json(path, record)
        elif role == "release_gate_report":
            report = _self_hashed_record(
                schema_version="1.0.0",
                record_kind="final_release_gate_report",
                mode="tracked_precommit_report",
                status="pending",
                created_at_utc="2026-08-24T12:00:00Z",
                repository={
                    "parent_commit": "0" * 40 if wrong_report_parent else base_commit,
                    "content_commit": None,
                    "worktree_clean": False,
                },
                policy=(
                    {
                        "path": "README.md",
                        "size_bytes": len((repository / "README.md").read_bytes()),
                        "file_sha256": hashlib.sha256(
                            (repository / "README.md").read_bytes()
                        ).hexdigest(),
                    }
                    if wrong_report_policy
                    else {
                        "path": "configs/release/release_gate_policy_v1.json",
                        "size_bytes": len(policy_path.read_bytes()),
                        "file_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
                    }
                ),
                gates={
                    gate_name: {"status": "not_run", "evidence": None}
                    for gate_name in (
                        "repository_scan",
                        "secret_scan",
                        "license_audit",
                        "tests",
                        "lint",
                        "format",
                        "types",
                        "manifests",
                        "splits",
                        "configuration",
                        "artifacts",
                        "ci",
                    )
                },
                external_candidate_attestation={
                    "status": "pending",
                    "expected_root": ".audit/release-attestations/<candidate_commit>/",
                    "report_path": None,
                },
            )
            _write_json(path, report)
        else:
            path.write_text(f"# Synthetic {role}\n", encoding="utf-8")
        paths[role] = relative
    output_root = repository / "release"
    output_root.mkdir()
    (output_root / ".gitkeep").write_text("", encoding="utf-8")
    spec_root = tmp_path / "attestations"
    spec_root.mkdir(parents=True)
    _git(repository, "add", ".")
    staged_record = scan_index(
        repository_root=repository,
        policy_path=policy_path,
        created_at_utc="2026-08-24T11:59:00Z",
    )
    _git(repository, "commit", "-m", "synthetic candidate")
    commit_value = _git(repository, "rev-parse", "HEAD")
    commit = commit_value.strip()
    _git(repository, "tag", "-a", "protocol-v1.2.0", "-m", "synthetic locked protocol", commit)

    artifacts: list[dict[str, Any]] = []
    for role in REQUIRED_ROLES:
        payload = _blob(repository, commit, paths[role])
        artifacts.append(
            {
                "role": role,
                "path": paths[role],
                "expected_file_sha256": hashlib.sha256(payload).hexdigest(),
                "evidence_status": "synthetic_release_fixture",
                "validation_status": "pass",
                "required_for_release": True,
                "claim_scope": "synthetic_only",
                "embedded_self_hash_field": (
                    "record_sha256"
                    if role in {"opening_receipt", "locked_target_index", "release_gate_report"}
                    else None
                ),
            }
        )

    gates: dict[str, dict[str, Any]] = {}
    remote: dict[str, Any]
    exact_attestation: dict[str, Any]
    if requested_status in {"ready", "released"}:
        repository_name = "example/inclusive-shift-har"
        repository_url = f"https://github.com/{repository_name}"
        repository_api_url = f"https://api.github.com/repos/{repository_name}"
        workflow_url = f"{repository_url}/actions/runs/1"
        artifact_api_url = f"{repository_api_url}/actions/artifacts/2"
        artifact_archive_url = f"{artifact_api_url}/zip"
        artifact_digest = f"sha256:{'0' * 64}"
        api_values: dict[str, Any] = {
            "repository-api.json": {
                "full_name": repository_name,
                "html_url": repository_url,
                "url": repository_api_url,
                "private": True,
                "visibility": "private",
                "default_branch": "main",
            },
            "main-ref-api.json": {
                "ref": "refs/heads/main",
                "url": f"{repository_api_url}/git/refs/heads/main",
                "object": {"type": "commit", "sha": commit},
            },
            "workflow-run-api.json": {
                "id": 1,
                "run_attempt": 1,
                "status": "completed",
                "conclusion": "success",
                "event": "push",
                "head_branch": "main",
                "head_sha": commit,
                "path": ".github/workflows/ci.yml",
                "html_url": workflow_url,
                "repository": {"full_name": repository_name},
            },
            "run-artifacts-api.json": {
                "total_count": 1,
                "artifacts": [
                    {
                        "id": 2,
                        "name": f"release-security-{commit}",
                        "expired": False,
                        "url": artifact_api_url,
                        "archive_download_url": artifact_archive_url,
                        "size_in_bytes": 1024,
                        "digest": artifact_digest,
                        "workflow_run": {"id": 1, "head_sha": commit},
                    }
                ],
            },
        }
        api_payloads = {
            basename: (json.dumps(value, sort_keys=True) + "\n").encode()
            for basename, value in api_values.items()
        }
        for basename, payload in api_payloads.items():
            (spec_root / basename).write_bytes(payload)

        def api_reference(basename: str) -> dict[str, Any]:
            payload = api_payloads[basename]
            return {
                "basename": basename,
                "size_bytes": len(payload),
                "file_sha256": hashlib.sha256(payload).hexdigest(),
            }

        quality_gate_names = [
            gate_name
            for gate_name in GATE_NAMES
            if gate_name
            in {
                "tests",
                "lint",
                "format",
                "types",
                "manifests",
                "splits",
                "configuration",
                "artifacts",
            }
        ]
        quality_gate_names.append("ci_quality_proxy")
        gate_evidence = {
            gate_name: {
                "exit_code": 0,
                "started_at_utc": "2026-08-24T12:00:00Z",
                "completed_at_utc": "2026-08-24T12:00:01Z",
                "log_path": f".audit/release-attestations/{commit}/{gate_name}.log",
                "log_size_bytes": 0,
                "log_sha256": hashlib.sha256(b"").hexdigest(),
            }
            for gate_name in quality_gate_names
            if gate_name != "ci_quality_proxy"
        }
        for provenance in gate_evidence.values():
            log_relative = provenance["log_path"]
            assert isinstance(log_relative, str)
            log_path = spec_root / log_relative
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_bytes(b"")
        quality_path = spec_root / "gates" / "quality.json"
        _write_json(
            quality_path,
            _self_hashed_record(
                schema_version="1.0.0",
                record_kind="ci_release_quality_gate_evidence",
                status="pass",
                candidate_commit=commit,
                created_at_utc="2026-08-24T12:00:02Z",
                gates=quality_gate_names,
                gate_statuses={gate_name: "pass" for gate_name in quality_gate_names},
                gate_evidence=gate_evidence,
                tool_versions={
                    "python": "3.11.9",
                    "uv": "uv 0.11.29",
                    "pytest": "pytest 9.0.2",
                    "ruff": "ruff 0.14.14",
                    "mypy": "mypy 1.19.1",
                },
                machine={"runner_os": "Linux", "architecture": "x86_64", "platform": "Linux"},
                scope="upstream synthetic-validation matrix and preceding release-security steps",
                ci_evidence_scope="within_job_quality_proxy_not_completed_workflow",
                external_completed_ci_required_for_release_inventory=True,
            ),
        )
        for gate_name in GATE_NAMES:
            if gate_name == "ci":
                evidence_path = spec_root / "gates" / f"{gate_name}.json"
                mirror_root = spec_root / ".audit" / "release-attestations" / commit
                mirror_root.mkdir(parents=True, exist_ok=True)
                mirrored_sources = {
                    "staged_index_scan.json": (
                        spec_root / gates["staged_index_scan"]["evidence_path"]
                    ),
                    "repository_scan.json": (
                        spec_root / gates["tracked_file_scan"]["evidence_path"]
                    ),
                    "secret_scan.json": spec_root / gates["secret_scan"]["evidence_path"],
                    "license_audit.json": spec_root / gates["license_audit"]["evidence_path"],
                    "quality_gates.json": quality_path,
                }
                for basename, source in mirrored_sources.items():
                    (mirror_root / basename).write_bytes(source.read_bytes())
                bundle_gate_basenames = {
                    "repository_scan": "repository_scan.json",
                    "secret_scan": "secret_scan.json",
                    "license_audit": "license_audit.json",
                    **{
                        name: "quality_gates.json"
                        for name in (*quality_gate_names, "ci_quality_proxy")
                    },
                }
                bundle_evidence = {
                    name: {
                        "path": f".audit/release-attestations/{commit}/{basename}",
                        "size_bytes": (mirror_root / basename).stat().st_size,
                        "file_sha256": _file_sha256(mirror_root / basename),
                    }
                    for name, basename in bundle_gate_basenames.items()
                }
                bundle_path = mirror_root / "ci_bundle_validation.json"
                _write_json(
                    bundle_path,
                    _self_hashed_record(
                        schema_version="1.0.0",
                        record_kind="ci_release_gate_bundle_validation",
                        status="pass",
                        created_at_utc="2026-08-24T12:04:00Z",
                        candidate_commit=commit,
                        parent_commit=base_commit,
                        ci_evidence_scope="within_job_quality_proxy_not_completed_workflow",
                        external_completed_ci_required_for_release_inventory=True,
                        validated_gates=sorted(bundle_evidence),
                        evidence=bundle_evidence,
                    ),
                )
                release_gate_file_sha256s = {
                    "repository_scan": _file_sha256(mirror_root / "repository_scan.json"),
                    "secret_scan": _file_sha256(mirror_root / "secret_scan.json"),
                    "license_audit": _file_sha256(mirror_root / "license_audit.json"),
                    "quality_gates": _file_sha256(mirror_root / "quality_gates.json"),
                    "ci_bundle_validation": _file_sha256(bundle_path),
                }
                for executed_gate in quality_gate_names:
                    if executed_gate == "ci_quality_proxy":
                        continue
                    provenance_payloads = {
                        f"{executed_gate}.status": b"pass\n",
                        f"{executed_gate}.exit": b"0\n",
                        f"{executed_gate}.started": b"2026-08-24T12:00:00Z\n",
                        f"{executed_gate}.completed": b"2026-08-24T12:00:01Z\n",
                    }
                    for basename, provenance_payload in provenance_payloads.items():
                        (mirror_root / basename).write_bytes(provenance_payload)
                archive_path = spec_root / f"release-security-{commit}.zip"
                with zipfile.ZipFile(
                    archive_path, "w", compression=zipfile.ZIP_DEFLATED
                ) as archive:
                    for basename in (*COMPONENT_FILES, *PROVENANCE_FILES):
                        archive.write(
                            mirror_root / basename,
                            f".audit/release-attestations/{commit}/{basename}",
                        )
                archive_payload = archive_path.read_bytes()
                archive_sha256 = hashlib.sha256(archive_payload).hexdigest()
                artifact_digest = f"sha256:{archive_sha256}"
                artifact_item = api_values["run-artifacts-api.json"]["artifacts"][0]
                artifact_item["size_in_bytes"] = len(archive_payload)
                artifact_item["digest"] = artifact_digest
                api_payloads["run-artifacts-api.json"] = (
                    json.dumps(api_values["run-artifacts-api.json"], sort_keys=True) + "\n"
                ).encode()
                (spec_root / "run-artifacts-api.json").write_bytes(
                    api_payloads["run-artifacts-api.json"]
                )
                api_payloads[archive_path.name] = archive_payload
                artifact_member_sha256s = {
                    basename: _file_sha256(mirror_root / basename)
                    for basename in (*COMPONENT_FILES, *PROVENANCE_FILES)
                }
                evidence = _self_hashed_record(
                    schema_version="1.0.0",
                    record_kind="ci_run_evidence",
                    status="pass",
                    conclusion="success",
                    candidate_commit=commit,
                    workflow_status="completed",
                    ci_evidence_scope="completed_remote_workflow_run",
                    event="push",
                    head_branch="main",
                    head_sha=commit,
                    repository=repository_name,
                    workflow_path=".github/workflows/ci.yml",
                    run_id=1,
                    run_attempt=1,
                    workflow_url=workflow_url,
                    artifact_name=f"release-security-{commit}",
                    artifact_id=2,
                    artifact_expired=False,
                    artifact_workflow_run_id=1,
                    artifact_workflow_run_head_sha=commit,
                    artifact_api_url=artifact_api_url,
                    artifacts_list_api_url=(
                        "https://api.github.com/repos/example/inclusive-shift-har/"
                        f"actions/runs/1/artifacts?name=release-security-{commit}"
                    ),
                    artifact_archive_download_url=artifact_archive_url,
                    artifact_size_in_bytes=len(archive_payload),
                    artifact_digest=artifact_digest,
                    downloaded_artifact_archive_sha256=archive_sha256,
                    downloaded_artifact_archive=api_reference(archive_path.name),
                    artifact_member_sha256s=artifact_member_sha256s,
                    release_gate_file_sha256s=release_gate_file_sha256s,
                    api_response_files={
                        "workflow_run": api_reference("workflow-run-api.json"),
                        "run_artifacts": api_reference("run-artifacts-api.json"),
                    },
                    queried_at_utc="2026-08-24T12:05:00Z",
                )
                _write_json(evidence_path, evidence)
            elif gate_name in quality_gate_names:
                evidence_path = quality_path
            elif gate_name == "staged_index_scan":
                evidence_path = spec_root / "gates" / f"{gate_name}.json"
                _write_json(evidence_path, staged_record)
            elif gate_name == "tracked_file_scan":
                evidence_path = spec_root / "gates" / f"{gate_name}.json"
                evidence = scan_repository(
                    repository_root=repository,
                    candidate_commit=commit,
                    policy_path=policy_path,
                    created_at_utc="2026-08-24T12:00:00Z",
                )
                _write_json(evidence_path, evidence)
            elif gate_name == "secret_scan":
                evidence_path = spec_root / "gates" / f"{gate_name}.json"
                raw_report = spec_root / "gates" / "gitleaks.json"
                _write_json(raw_report, [])
                evidence = validate_secret_scan(
                    repository_root=repository,
                    report_path=raw_report,
                    candidate_commit=commit,
                    policy_path=policy_path,
                    config_path=repository / ".gitleaks.toml",
                    ignore_path=repository / ".gitleaksignore",
                    gitleaks_version="8.30.1",
                    gitleaks_exit_code=0,
                    created_at_utc="2026-08-24T12:00:00Z",
                )
                _write_json(evidence_path, evidence)
            else:
                assert gate_name == "license_audit"
                evidence_path = spec_root / "gates" / f"{gate_name}.json"
                raw_licenses = spec_root / "gates" / "python_licenses.json"
                _write_json(raw_licenses, [{"Name": "safe", "Version": "1", "License": "MIT"}])
                evidence = audit_licenses(
                    repository_root=repository,
                    inventory_path=raw_licenses,
                    candidate_commit=commit,
                    policy_path=policy_path,
                    pip_licenses_version="5.5.5",
                    created_at_utc="2026-08-24T12:00:00Z",
                )
                _write_json(evidence_path, evidence)
            gates[gate_name] = {
                "status": "pass",
                "evidence_source": "spec_root",
                "evidence_path": evidence_path.relative_to(spec_root).as_posix(),
                "expected_sha256": _file_sha256(evidence_path),
                "tool_version": (
                    {
                        "tests": "pytest 9.0.2",
                        "lint": "ruff 0.14.14",
                        "format": "ruff 0.14.14",
                        "types": "mypy 1.19.1",
                        "configuration": "pytest 9.0.2",
                        "secret_scan": "gitleaks 8.30.1",
                        "license_audit": "pip-licenses 5.5.5",
                        "ci": "github-actions run 1 attempt 1",
                    }.get(gate_name, f"inclusive-shift-har@{commit}")
                ),
                "note": "synthetic gate evidence",
            }
        exact_gate_basenames = {
            "staged_index_scan": "staged_index_scan.json",
            "repository_scan": "repository_scan.json",
            "secret_scan": "secret_scan.json",
            "license_audit": "license_audit.json",
            **{name: "quality_gates.json" for name in quality_gate_names},
        }
        exact_gate_evidence = {
            gate_name: {
                "status": "pass",
                "evidence": {
                    "path": f".audit/release-attestations/{commit}/{basename}",
                    "size_bytes": (mirror_root / basename).stat().st_size,
                    "file_sha256": _file_sha256(mirror_root / basename),
                },
            }
            for gate_name, basename in exact_gate_basenames.items()
        }
        exact_report_path = mirror_root / "final_release_gate_report.json"
        _write_json(
            exact_report_path,
            _self_hashed_record(
                schema_version="1.0.0",
                record_kind="final_release_gate_report",
                mode="exact_candidate_attestation",
                status="pass",
                created_at_utc="2026-08-24T12:06:00Z",
                repository={
                    "parent_commit": base_commit,
                    "content_commit": commit,
                    "worktree_clean": True,
                },
                policy={
                    "path": "configs/release/release_gate_policy_v1.json",
                    "size_bytes": len(policy_path.read_bytes()),
                    "file_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
                },
                gates=exact_gate_evidence,
                external_candidate_attestation={
                    "status": "complete",
                    "expected_root": ".audit/release-attestations/<candidate_commit>/",
                    "report_path": (
                        f".audit/release-attestations/{commit}/final_release_gate_report.json"
                    ),
                },
            ),
        )
        exact_attestation = {
            "status": "pass",
            "evidence_source": "spec_root",
            "evidence_path": (
                f".audit/release-attestations/{commit}/final_release_gate_report.json"
            ),
            "expected_sha256": _file_sha256(exact_report_path),
        }
        remote_path = spec_root / "remote.json"
        remote_url = repository_url
        _write_json(
            remote_path,
            _self_hashed_record(
                schema_version="1.0.0",
                record_kind="github_remote_evidence",
                status="pass",
                candidate_commit=commit,
                visibility="private",
                remote_url=remote_url,
                repository=repository_name,
                default_branch="main",
                ref_name="refs/heads/main",
                ref_commit=commit,
                repository_api_url=repository_api_url,
                ref_api_url=f"{repository_api_url}/git/refs/heads/main",
                api_response_files={
                    "repository": api_reference("repository-api.json"),
                    "main_ref": api_reference("main-ref-api.json"),
                },
                queried_at_utc="2026-08-24T12:05:00Z",
            ),
        )
        remote = {
            "visibility": "private",
            "url": remote_url,
            "evidence_source": "spec_root",
            "evidence_path": "remote.json",
            "expected_sha256": _file_sha256(remote_path),
        }
    else:
        gates = {
            gate_name: {
                "status": "not_run",
                "evidence_source": None,
                "evidence_path": None,
                "expected_sha256": None,
                "tool_version": None,
                "note": "not run in synthetic draft",
            }
            for gate_name in GATE_NAMES
        }
        remote = {
            "visibility": "absent",
            "url": None,
            "evidence_source": None,
            "evidence_path": None,
            "expected_sha256": None,
        }
        exact_attestation = {
            "status": "not_run",
            "evidence_source": None,
            "evidence_path": None,
            "expected_sha256": None,
        }

    postconfirmatory = [
        {
            "track": track,
            "status": "incomplete" if index == 0 else "not_run",
            "evidence_role": None,
            "note": "explicit synthetic incomplete state",
            "primary_claim_eligible": False,
        }
        for index, track in enumerate(POSTCONFIRMATORY_TRACKS)
    ]
    spec = {
        "schema_version": "1.0.0",
        "spec_kind": "release_evidence_inventory_spec",
        "requested_status": requested_status,
        "created_at_utc": "2026-08-24T12:00:00Z",
        "protocol_tag": "protocol-v1.2.0",
        "remote": remote,
        "exact_candidate_attestation": exact_attestation,
        "confirmatory_state": {
            "opening_count": 1,
            "target_rerun_permitted": False,
        },
        "artifacts": artifacts,
        "gates": gates,
        "postconfirmatory_statuses": postconfirmatory,
        "blockers": [],
    }
    spec_path = spec_root / "release-spec.json"
    _write_json(spec_path, spec)
    assert not str(_git(repository, "status", "--porcelain=v1", "--untracked-files=all")).strip()
    return ReleaseWorkspace(repository, spec_path, output_root, commit, spec)


def _generate(
    workspace: ReleaseWorkspace, *, destination: str = "inventory.json"
) -> dict[str, Any]:
    return generate_release_evidence_inventory(
        spec_path=workspace.spec_path,
        repository_root=workspace.repository,
        candidate_commit=workspace.commit,
        require_clean_worktree=True,
        destination=destination,
        allowed_output_root=workspace.output_root,
    )


def _mutate_gate_evidence(
    workspace: ReleaseWorkspace,
    gate_name: str,
    mutate: Callable[[dict[str, Any]], Any],
    *,
    repair_self_hash: bool = True,
) -> None:
    gate = workspace.spec["gates"][gate_name]
    relative = gate["evidence_path"]
    path = workspace.spec_path.parent / relative
    record = json.loads(path.read_text(encoding="utf-8"))
    mutate(record)
    if repair_self_hash:
        record.pop("record_sha256", None)
        record["record_sha256"] = canonical_json_sha256(record)
    _write_json(path, record)
    observed = _file_sha256(path)
    for item in workspace.spec["gates"].values():
        if item["evidence_path"] == relative:
            item["expected_sha256"] = observed
    _write_json(workspace.spec_path, workspace.spec)


def _mutate_ci_api_response(
    workspace: ReleaseWorkspace,
    *,
    basename: str,
    reference_name: str,
    mutate: Callable[[dict[str, Any]], Any],
) -> None:
    spec_root = workspace.spec_path.parent
    raw_path = spec_root / basename
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    mutate(raw)
    _write_json(raw_path, raw)
    gate = workspace.spec["gates"]["ci"]
    evidence_path = spec_root / gate["evidence_path"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["api_response_files"][reference_name].update(
        {
            "size_bytes": raw_path.stat().st_size,
            "file_sha256": _file_sha256(raw_path),
        }
    )
    evidence.pop("record_sha256")
    evidence["record_sha256"] = canonical_json_sha256(evidence)
    _write_json(evidence_path, evidence)
    gate["expected_sha256"] = _file_sha256(evidence_path)
    _write_json(workspace.spec_path, workspace.spec)


def test_generate_and_validate_draft_with_explicit_not_run_tracks(
    tmp_path: Path, repository_root: Path
) -> None:
    workspace = _build_workspace(tmp_path)
    inventory = _generate(workspace)

    assert inventory["schema_version"] == "1.1.0"
    assert inventory["delivery_mode"] == "external_release_asset"
    assert inventory["tracked"] is False
    assert inventory["status"] == "draft"
    assert inventory["repository"]["worktree_clean"] is True
    assert len(inventory["repository"]["protocol_tag_object"]) == 40
    assert inventory["repository"]["protocol_tag_target"] == workspace.commit
    assert inventory["repository"]["remote_visibility"] == "absent"
    assert inventory["gates"]["ci"]["status"] == "not_run"
    assert inventory["confirmatory_state"]["opening_count"] == 1
    assert inventory["confirmatory_state"]["target_rerun_permitted"] is False
    assert len({entry["role"] for entry in inventory["artifacts"]}) == len(REQUIRED_ROLES)
    assert {entry["status"] for entry in inventory["postconfirmatory_statuses"]} == {
        "incomplete",
        "not_run",
    }
    serialized = json.dumps(inventory)
    assert str(tmp_path) not in serialized
    _inventory_schema_validator(repository_root).validate(inventory)
    report = validate_release_evidence_inventory_file(
        workspace.output_root / "inventory.json",
        spec_path=workspace.spec_path,
        repository_root=workspace.repository,
        expected_candidate_commit=workspace.commit,
    )
    assert report["valid"] is True, report


def test_ready_requires_and_validates_external_ci_and_remote_attestations(
    tmp_path: Path, repository_root: Path
) -> None:
    workspace = _build_workspace(tmp_path, requested_status="ready")
    inventory = _generate(workspace)

    assert inventory["status"] == "ready"
    assert inventory["repository"]["remote_visibility"] == "private"
    assert inventory["repository"]["remote_evidence_scope"] == "spec_root"
    assert inventory["gates"]["ci"]["status"] == "pass"
    assert inventory["gates"]["ci"]["evidence_scope"] == "spec_root"
    assert inventory["exact_candidate_attestation"]["status"] == "pass"
    _inventory_schema_validator(repository_root).validate(inventory)


def test_ready_requires_exact_candidate_attestation(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path, requested_status="ready")
    workspace.spec["exact_candidate_attestation"] = {
        "status": "not_run",
        "evidence_source": None,
        "evidence_path": None,
        "expected_sha256": None,
    }
    _write_json(workspace.spec_path, workspace.spec)

    with pytest.raises(ReleaseEvidenceError, match="requires exact candidate attestation"):
        _generate(workspace)


def test_ready_rejects_rehashed_exact_attestation_extra_field(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path, requested_status="ready")
    item = workspace.spec["exact_candidate_attestation"]
    path = workspace.spec_path.parent / item["evidence_path"]
    record = json.loads(path.read_text(encoding="utf-8"))
    record["unexpected"] = True
    record.pop("record_sha256")
    record["record_sha256"] = canonical_json_sha256(record)
    _write_json(path, record)
    item["expected_sha256"] = _file_sha256(path)
    _write_json(workspace.spec_path, workspace.spec)

    with pytest.raises(ReleaseEvidenceError, match="exact candidate attestation keys differ"):
        _generate(workspace)


def test_ready_requires_retained_raw_api_responses(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path, requested_status="ready")
    (workspace.spec_path.parent / "workflow-run-api.json").unlink()

    with pytest.raises(ReleaseEvidenceError, match="not retained at spec_root"):
        _generate(workspace)


def test_ready_requires_retained_mirrored_ci_bundle(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path, requested_status="ready")
    bundle = (
        workspace.spec_path.parent
        / ".audit"
        / "release-attestations"
        / workspace.commit
        / "ci_bundle_validation.json"
    )
    bundle.unlink()

    with pytest.raises(ReleaseEvidenceError, match="missing from spec_root"):
        _generate(workspace)


def test_ready_requires_retained_ci_archive(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path, requested_status="ready")
    (workspace.spec_path.parent / f"release-security-{workspace.commit}.zip").unlink()

    with pytest.raises(ReleaseEvidenceError, match="not retained at spec_root"):
        _generate(workspace)


def test_ready_rejects_rehashed_wrong_workflow_api_response(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path, requested_status="ready")
    _mutate_ci_api_response(
        workspace,
        basename="workflow-run-api.json",
        reference_name="workflow_run",
        mutate=lambda raw: raw.update({"head_sha": "f" * 40}),
    )

    with pytest.raises(ReleaseEvidenceError, match="workflow-run API response differs"):
        _generate(workspace)


def test_ready_rejects_boolean_ci_artifact_workflow_id(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path, requested_status="ready")
    _mutate_gate_evidence(
        workspace,
        "ci",
        lambda record: record.update({"artifact_workflow_run_id": True}),
    )

    with pytest.raises(ReleaseEvidenceError, match="CI run identity is inconsistent"):
        _generate(workspace)


def test_ready_rejects_integer_expired_flag_in_raw_artifact_response(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path, requested_status="ready")
    _mutate_ci_api_response(
        workspace,
        basename="run-artifacts-api.json",
        reference_name="run_artifacts",
        mutate=lambda raw: raw["artifacts"][0].update({"expired": 0}),
    )

    with pytest.raises(ReleaseEvidenceError, match="run-artifacts API response differs"):
        _generate(workspace)


@pytest.mark.parametrize(
    "basename,reference_name,mutate",
    [
        (
            "workflow-run-api.json",
            "workflow_run",
            lambda raw: raw.update({"id": 1.0}),
        ),
        (
            "workflow-run-api.json",
            "workflow_run",
            lambda raw: raw.update({"run_attempt": 1.0}),
        ),
        (
            "run-artifacts-api.json",
            "run_artifacts",
            lambda raw: raw.update({"total_count": 1.0}),
        ),
        (
            "run-artifacts-api.json",
            "run_artifacts",
            lambda raw: raw["artifacts"][0].update({"id": 2.0}),
        ),
        (
            "run-artifacts-api.json",
            "run_artifacts",
            lambda raw: raw["artifacts"][0].update({"size_in_bytes": 1.0}),
        ),
        (
            "run-artifacts-api.json",
            "run_artifacts",
            lambda raw: raw["artifacts"][0]["workflow_run"].update({"id": 1.0}),
        ),
    ],
)
def test_ready_rejects_float_github_integer_fields(
    tmp_path: Path,
    basename: str,
    reference_name: str,
    mutate: Callable[[dict[str, Any]], Any],
) -> None:
    workspace = _build_workspace(tmp_path, requested_status="ready")
    _mutate_ci_api_response(
        workspace,
        basename=basename,
        reference_name=reference_name,
        mutate=mutate,
    )

    with pytest.raises(ReleaseEvidenceError, match="exact JSON integer"):
        _generate(workspace)


def test_ready_rejects_api_response_basename_alias_across_roles(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path, requested_status="ready")
    spec_root = workspace.spec_path.parent
    repository_path = spec_root / "repository-api.json"
    repository_response = json.loads(repository_path.read_text(encoding="utf-8"))
    artifact_response = json.loads(
        (spec_root / "run-artifacts-api.json").read_text(encoding="utf-8")
    )
    repository_response.update(artifact_response)
    _write_json(repository_path, repository_response)
    shared_reference = {
        "basename": repository_path.name,
        "size_bytes": repository_path.stat().st_size,
        "file_sha256": _file_sha256(repository_path),
    }

    ci_gate = workspace.spec["gates"]["ci"]
    ci_path = spec_root / ci_gate["evidence_path"]
    ci_record = json.loads(ci_path.read_text(encoding="utf-8"))
    ci_record["api_response_files"]["run_artifacts"] = shared_reference
    ci_record.pop("record_sha256")
    ci_record["record_sha256"] = canonical_json_sha256(ci_record)
    _write_json(ci_path, ci_record)
    ci_gate["expected_sha256"] = _file_sha256(ci_path)

    remote_item = workspace.spec["remote"]
    remote_path = spec_root / remote_item["evidence_path"]
    remote_record = json.loads(remote_path.read_text(encoding="utf-8"))
    remote_record["api_response_files"]["repository"] = shared_reference
    remote_record.pop("record_sha256")
    remote_record["record_sha256"] = canonical_json_sha256(remote_record)
    _write_json(remote_path, remote_record)
    remote_item["expected_sha256"] = _file_sha256(remote_path)
    _write_json(workspace.spec_path, workspace.spec)

    with pytest.raises(ReleaseEvidenceError, match="distinct basenames"):
        _generate(workspace)


def test_ready_rejects_internal_candidate_mirror_ancestor_symlink(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path, requested_status="ready")
    spec_root = workspace.spec_path.parent
    mirror = spec_root / ".audit" / "release-attestations" / workspace.commit
    alias = spec_root / "candidate-mirror-alias"
    mirror.rename(alias)
    try:
        mirror.symlink_to(alias, target_is_directory=True)
    except OSError as exc:
        alias.rename(mirror)
        pytest.skip(f"directory symlinks unavailable: {exc}")

    with pytest.raises(ReleaseEvidenceError, match="may not traverse a symlink"):
        _generate(workspace)


def test_ready_rejects_rehashed_wrong_repository_api_response(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path, requested_status="ready")
    spec_root = workspace.spec_path.parent
    raw_path = spec_root / "repository-api.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["default_branch"] = "release"
    _write_json(raw_path, raw)

    remote_item = workspace.spec["remote"]
    evidence_path = spec_root / remote_item["evidence_path"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["api_response_files"]["repository"].update(
        {
            "size_bytes": raw_path.stat().st_size,
            "file_sha256": _file_sha256(raw_path),
        }
    )
    evidence.pop("record_sha256")
    evidence["record_sha256"] = canonical_json_sha256(evidence)
    _write_json(evidence_path, evidence)
    remote_item["expected_sha256"] = _file_sha256(evidence_path)
    _write_json(workspace.spec_path, workspace.spec)

    with pytest.raises(ReleaseEvidenceError, match="repository/main-ref API responses differ"):
        _generate(workspace)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["gates"]["tests"].update({"status": "fail"}),
        lambda value: value["repository"].update({"remote_visibility": "absent"}),
        lambda value: value["artifacts"][0].update({"validation_status": "failed"}),
        lambda value: value["blockers"].append(
            {
                "blocker_id": "open-release-blocker",
                "status": "open",
                "description": "synthetic open blocker",
                "evidence_role": None,
            }
        ),
    ],
)
def test_ready_inventory_schema_rejects_semantic_overclaims(
    tmp_path: Path,
    repository_root: Path,
    mutate: Callable[[dict[str, Any]], Any],
) -> None:
    workspace = _build_workspace(tmp_path, requested_status="ready")
    inventory = deepcopy(_generate(workspace))
    mutate(inventory)

    with pytest.raises(ValidationError):
        _inventory_schema_validator(repository_root).validate(inventory)


def test_inventory_v1_1_refuses_unattested_released_status(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path, requested_status="released")

    with pytest.raises(ReleaseEvidenceError, match="requested_status is invalid"):
        _generate(workspace)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda record: record.update({"record_kind": "unreviewed_quality_claim"}),
            "record kind is incompatible",
        ),
        (
            lambda record: record.update({"status": "fail"}),
            "status differs from the generation spec",
        ),
        (
            lambda record: record.update({"candidate_commit": "f" * 40}),
            "another candidate commit",
        ),
        (
            lambda record: (
                record["gates"].remove("lint"),
                record["gate_statuses"].pop("lint"),
            ),
            "does not cover the cited gate and status",
        ),
        (
            lambda record: record["gate_statuses"].update({"lint": "fail"}),
            "overall status is inconsistent",
        ),
        (
            lambda record: record["gates"].append("lint"),
            "duplicates gate coverage",
        ),
        (
            lambda record: record.update({"ci_evidence_scope": "completed_remote_workflow_run"}),
            "does not cover the cited gate and status",
        ),
    ],
)
def test_ready_rejects_semantically_invalid_quality_gate_evidence(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], Any],
    match: str,
) -> None:
    workspace = _build_workspace(tmp_path, requested_status="ready")
    _mutate_gate_evidence(workspace, "lint", mutate)

    with pytest.raises(ReleaseEvidenceError, match=match):
        _generate(workspace)


@pytest.mark.parametrize(
    "gate_name,mutate,match",
    [
        (
            "tracked_file_scan",
            lambda record: record["violations"].append({"code": "forbidden_path"}),
            "violations are inconsistent",
        ),
        (
            "secret_scan",
            lambda record: record.update({"finding_count": 1}),
            "finding count or scope is inconsistent",
        ),
        (
            "secret_scan",
            lambda record: record.update({"finding_count": False}),
            "finding count or scope is inconsistent",
        ),
        (
            "license_audit",
            lambda record: record["violations"].append({"package": "unsafe"}),
            "violations are inconsistent",
        ),
        (
            "staged_index_scan",
            lambda record: record.update({"replace_ref_count": False}),
            "exact JSON integer",
        ),
        (
            "tracked_file_scan",
            lambda record: record.update({"replace_ref_count": 0.0}),
            "exact JSON integer",
        ),
        (
            "secret_scan",
            lambda record: record.update({"gitleaks_exit_code": 0.0}),
            "exact JSON integer",
        ),
    ],
)
def test_ready_rejects_semantically_invalid_special_gate_evidence(
    tmp_path: Path,
    gate_name: str,
    mutate: Callable[[dict[str, Any]], Any],
    match: str,
) -> None:
    workspace = _build_workspace(tmp_path, requested_status="ready")
    _mutate_gate_evidence(workspace, gate_name, mutate)

    with pytest.raises(ReleaseEvidenceError, match=match):
        _generate(workspace)


@pytest.mark.parametrize(
    "gate_name,mutate",
    [
        ("lint", lambda record: record.update({"unexpected": True})),
        ("staged_index_scan", lambda record: record.pop("created_at_utc")),
        ("tracked_file_scan", lambda record: record.update({"unexpected": True})),
        ("secret_scan", lambda record: record.pop("raw_report")),
        ("license_audit", lambda record: record.pop("inventory")),
    ],
)
def test_ready_rejects_rehashed_gate_records_with_missing_or_extra_keys(
    tmp_path: Path,
    gate_name: str,
    mutate: Callable[[dict[str, Any]], Any],
) -> None:
    workspace = _build_workspace(tmp_path, requested_status="ready")
    _mutate_gate_evidence(workspace, gate_name, mutate)

    with pytest.raises(ReleaseEvidenceError, match="keys differ"):
        _generate(workspace)


def test_ready_rejects_rehashed_gate_evidence_with_invalid_embedded_hash(
    tmp_path: Path,
) -> None:
    workspace = _build_workspace(tmp_path, requested_status="ready")
    _mutate_gate_evidence(
        workspace,
        "lint",
        lambda record: record.update({"record_sha256": "0" * 64}),
        repair_self_hash=False,
    )

    with pytest.raises(ReleaseEvidenceError, match="self-hash does not reconstruct"):
        _generate(workspace)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda record: record.update(
                {"ci_evidence_scope": "within_job_quality_proxy_not_completed_workflow"}
            ),
            "incompatible status or commit",
        ),
        (
            lambda record: record.update({"workflow_url": "https://example.com/run/1"}),
            "canonical completed workflow URL",
        ),
    ],
)
def test_ready_rejects_quality_proxy_or_noncanonical_url_as_completed_ci(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], Any],
    match: str,
) -> None:
    workspace = _build_workspace(tmp_path, requested_status="ready")
    _mutate_gate_evidence(workspace, "ci", mutate)

    with pytest.raises(ReleaseEvidenceError, match=match):
        _generate(workspace)


def test_ready_requires_external_generation_spec(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path, requested_status="ready")
    exclude_path = workspace.repository / ".git" / "info" / "exclude"
    exclude_path.write_text(".audit/\n", encoding="utf-8")
    ignored_root = workspace.repository / ".audit"
    ignored_root.mkdir()
    internal_spec = ignored_root / "release-spec.json"
    _write_json(internal_spec, workspace.spec)

    with pytest.raises(ReleaseEvidenceError, match="protected raw/cache evidence"):
        generate_release_evidence_inventory(
            spec_path=internal_spec,
            repository_root=workspace.repository,
            candidate_commit=workspace.commit,
            require_clean_worktree=True,
            destination="inventory.json",
            allowed_output_root=workspace.output_root,
        )


def test_validator_rechecks_mutated_external_gate_evidence(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path, requested_status="ready")
    _generate(workspace)
    quality_path = workspace.spec_path.parent / "gates" / "quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["status"] = "fail"
    quality.pop("record_sha256")
    quality["record_sha256"] = canonical_json_sha256(quality)
    _write_json(quality_path, quality)

    report = validate_release_evidence_inventory_file(
        workspace.output_root / "inventory.json",
        spec_path=workspace.spec_path,
        repository_root=workspace.repository,
        expected_candidate_commit=workspace.commit,
    )
    assert report["valid"] is False
    assert "reviewed pin" in report["errors"][0]


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda spec: spec["artifacts"].pop(), "required release roles are missing"),
        (
            lambda spec: spec["artifacts"].append(dict(spec["artifacts"][0])),
            "unsupported or duplicated",
        ),
        (
            lambda spec: spec["confirmatory_state"].update({"opening_count": 2}),
            "must equal 1",
        ),
        (
            lambda spec: spec["confirmatory_state"].update({"target_rerun_permitted": True}),
            "forbid rerun",
        ),
        (
            lambda spec: spec["postconfirmatory_statuses"].pop(),
            "status entries are missing",
        ),
    ],
)
def test_generator_rejects_coverage_and_confirmatory_mutations(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], Any],
    match: str,
) -> None:
    workspace = _build_workspace(tmp_path)
    mutate(workspace.spec)
    _write_json(workspace.spec_path, workspace.spec)

    with pytest.raises(ReleaseEvidenceError, match=match):
        _generate(workspace)


@pytest.mark.parametrize("opening_count", [True, 1.0])
def test_generator_rejects_noninteger_confirmatory_opening_count(
    tmp_path: Path, opening_count: bool | float
) -> None:
    workspace = _build_workspace(tmp_path)
    workspace.spec["confirmatory_state"]["opening_count"] = opening_count
    _write_json(workspace.spec_path, workspace.spec)

    with pytest.raises(ReleaseEvidenceError, match="exact JSON integer"):
        _generate(workspace)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"opening_number": True},
        {"opening_number": 1.0},
        {"unlock_opening_number": True},
        {"unlock_opening_number": 1.0},
        {"maximum_target_openings": True},
        {"maximum_target_openings": 1.0},
    ],
)
def test_generator_rejects_noninteger_confirmatory_lineage_counts(
    tmp_path: Path, kwargs: dict[str, Any]
) -> None:
    workspace = _build_workspace(tmp_path, **kwargs)

    with pytest.raises(ReleaseEvidenceError, match="exact JSON integer"):
        _generate(workspace)


def test_generator_rejects_bad_declared_embedded_self_hash(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path, bad_opening_self_hash=True)

    with pytest.raises(ReleaseEvidenceError, match="embedded self-hash"):
        _generate(workspace)


@pytest.mark.parametrize(
    ("workspace_kwargs", "match"),
    [
        ({"opening_number": 2}, "must equal 1"),
        ({"bad_opening_kind": True}, "opening_receipt confirmatory semantics"),
        ({"bad_target_cross_hash": True}, "does not cross-bind"),
        ({"extra_opening_field": True}, "opening_receipt keys differ"),
        ({"extra_target_field": True}, "locked_target_index keys differ"),
        ({"wrong_report_parent": True}, "differs from candidate first parent"),
        ({"wrong_report_policy": True}, "differs from the candidate policy blob"),
        ({"bad_machine_binding": True}, "machine provenance does not cross-bind"),
        ({"bad_freeze_binding": True}, "freeze or split provenance does not cross-bind"),
    ],
)
def test_generator_rejects_rehashed_incompatible_confirmatory_records(
    tmp_path: Path, workspace_kwargs: dict[str, Any], match: str
) -> None:
    workspace = _build_workspace(tmp_path, **workspace_kwargs)

    with pytest.raises(ReleaseEvidenceError, match=match):
        _generate(workspace)


def test_generator_rejects_onnx_payload_reference(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path)
    workspace.spec["artifacts"][0]["path"] = "evidence/model.onnx"
    _write_json(workspace.spec_path, workspace.spec)

    with pytest.raises(ReleaseEvidenceError, match="prohibited raw/checkpoint extension"):
        _generate(workspace)


@pytest.mark.parametrize("role", ["literature_matrix", "legacy_audit", "dataset_manifest"])
def test_generator_rejects_required_role_path_substitution(tmp_path: Path, role: str) -> None:
    workspace = _build_workspace(tmp_path)
    artifact = next(item for item in workspace.spec["artifacts"] if item["role"] == role)
    artifact["path"] = "README.md"
    _write_json(workspace.spec_path, workspace.spec)

    with pytest.raises(ReleaseEvidenceError, match="must use canonical path"):
        _generate(workspace)


def test_every_required_release_role_has_one_canonical_path() -> None:
    assert set(CANONICAL_REQUIRED_ROLE_PATHS) == set(REQUIRED_ROLES)
    assert len(set(CANONICAL_REQUIRED_ROLE_PATHS.values())) == len(REQUIRED_ROLES)


def test_inventory_output_rejects_lexical_symlink_components(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    allowed = repository / "release"
    real = allowed / "real"
    real.mkdir(parents=True)
    alias = allowed / "alias"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    with pytest.raises(ReleaseEvidenceError, match="traverse a symlink"):
        _new_output(
            "alias/inventory.json",
            allowed_root=allowed,
            repository_root=repository.resolve(),
        )


def test_generator_rejects_ci_pass_without_pinned_evidence(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path)
    workspace.spec["gates"]["ci"]["status"] = "pass"
    _write_json(workspace.spec_path, workspace.spec)

    with pytest.raises(ReleaseEvidenceError, match="requires pinned evidence"):
        _generate(workspace)


def test_generator_rejects_decorative_tool_version_mismatch(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path, requested_status="ready")
    workspace.spec["gates"]["secret_scan"]["tool_version"] = "gitleaks 1.0.0"
    _write_json(workspace.spec_path, workspace.spec)

    with pytest.raises(ReleaseEvidenceError, match="tool_version differs from parsed evidence"):
        _generate(workspace)


def test_generator_rejects_remote_claim_without_validation_evidence(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path)
    workspace.spec["remote"].update(
        {
            "visibility": "private",
            "url": "https://github.com/example/inclusive-shift-har",
        }
    )
    _write_json(workspace.spec_path, workspace.spec)

    with pytest.raises(ReleaseEvidenceError, match="requires pinned validation evidence"):
        _generate(workspace)


def test_generator_rejects_dirty_tree_wrong_commit_and_overwrite(tmp_path: Path) -> None:
    dirty = _build_workspace(tmp_path / "dirty")
    (dirty.repository / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ReleaseEvidenceError, match="not clean"):
        _generate(dirty)

    wrong = _build_workspace(tmp_path / "wrong")
    with pytest.raises(ReleaseEvidenceError, match="differs from repository HEAD"):
        generate_release_evidence_inventory(
            spec_path=wrong.spec_path,
            repository_root=wrong.repository,
            candidate_commit="0" * 40,
            require_clean_worktree=True,
            destination="inventory.json",
            allowed_output_root=wrong.output_root,
        )

    create_only = _build_workspace(tmp_path / "create-only")
    _generate(create_only)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _generate(create_only)


def test_generator_requires_annotated_protocol_tag_on_candidate_lineage(tmp_path: Path) -> None:
    absent = _build_workspace(tmp_path / "absent")
    _git(absent.repository, "tag", "-d", "protocol-v1.2.0")
    with pytest.raises(ReleaseEvidenceError, match="does not exist"):
        _generate(absent)

    lightweight = _build_workspace(tmp_path / "lightweight")
    _git(lightweight.repository, "tag", "-d", "protocol-v1.2.0")
    _git(lightweight.repository, "tag", "protocol-v1.2.0", lightweight.commit)
    with pytest.raises(ReleaseEvidenceError, match="annotated tag"):
        _generate(lightweight)

    wrong = _build_workspace(tmp_path / "wrong-lineage")
    _git(wrong.repository, "tag", "-d", "protocol-v1.2.0")
    tree = _git(wrong.repository, "rev-parse", f"{wrong.commit}^{{tree}}").strip()
    unrelated = _git(wrong.repository, "commit-tree", tree, "-m", "unrelated root").strip()
    _git(
        wrong.repository,
        "tag",
        "-a",
        "protocol-v1.2.0",
        "-m",
        "wrong protocol lineage",
        unrelated,
    )
    with pytest.raises(ReleaseEvidenceError, match="not an ancestor"):
        _generate(wrong)


def test_generator_rejects_protocol_tag_role_substitution(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path)
    workspace.spec["protocol_tag"] = "legacy-audit-v0.1.0"
    _write_json(workspace.spec_path, workspace.spec)

    with pytest.raises(ReleaseEvidenceError, match="active protocol anchor"):
        _generate(workspace)


def test_generator_rejects_legacy_git_grafts(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path)
    grafts = workspace.repository / ".git" / "info" / "grafts"
    grafts.write_text(f"{workspace.commit}\n", encoding="utf-8")

    with pytest.raises(ReleaseEvidenceError, match="legacy Git grafts"):
        _generate(workspace)


def test_relative_output_root_is_repository_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _build_workspace(tmp_path)
    monkeypatch.chdir(tmp_path)

    generate_release_evidence_inventory(
        spec_path=workspace.spec_path,
        repository_root=workspace.repository,
        candidate_commit=workspace.commit,
        require_clean_worktree=True,
        destination="nested/inventory.json",
        allowed_output_root="release",
    )

    assert (workspace.output_root / "nested/inventory.json").is_file()
    assert not (tmp_path / "release/nested/inventory.json").exists()


def test_generator_rejects_nested_repository_root_external_spec_bypass(
    tmp_path: Path,
) -> None:
    workspace = _build_workspace(tmp_path)
    nested = workspace.repository / "release"

    with pytest.raises(ReleaseEvidenceError, match="Git worktree top-level"):
        generate_release_evidence_inventory(
            spec_path=workspace.spec_path,
            repository_root=nested,
            candidate_commit=workspace.commit,
            require_clean_worktree=True,
            destination="inventory.json",
            allowed_output_root=nested,
        )


def test_validator_rejects_self_rehashed_inventory_mutation(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path)
    _generate(workspace)
    path = workspace.output_root / "inventory.json"
    inventory = json.loads(path.read_text(encoding="utf-8"))
    inventory["artifacts"][0]["file_sha256"] = "0" * 64
    inventory.pop("record_sha256")
    inventory["record_sha256"] = canonical_json_sha256(inventory)
    _write_json(path, inventory)

    report = validate_release_evidence_inventory_file(
        path,
        spec_path=workspace.spec_path,
        repository_root=workspace.repository,
        expected_candidate_commit=workspace.commit,
    )
    assert report["valid"] is False
    assert "deterministic reconstruction" in report["errors"][0]
