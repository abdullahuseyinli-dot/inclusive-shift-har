from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from inclusive_shift_har.artifacts.release_gate import (
    LICENSE_KIND,
    SCAN_KIND,
    SECRET_KIND,
    ReleaseGateError,
    assemble_report,
    audit_licenses,
    scan_repository,
    validate_report,
    validate_secret_scan,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _policy(path: Path) -> Path:
    value = {
        "schema_version": "1.0.0",
        "policy_kind": "final_release_gate_policy",
        "maximum_blob_size_bytes": 1024,
        "forbidden_path_prefixes": ["data/raw/", ".audit/"],
        "forbidden_suffixes": [".zip", ".npz", ".onnx", ".pt"],
        "disguised_binary_signatures": {"zip": "504b0304", "pdf": "25504446"},
        "allowed_signature_suffixes": {"pdf": [".pdf"]},
        "gitleaks": {"version": "8.30.1"},
        "prohibited_license_tokens": ["gpl", "unknown", "proprietary"],
        "license_exceptions": [],
    }
    _write_json(path, value)
    return path


def _repository(path: Path) -> tuple[Path, str]:
    path.mkdir()
    _git(path, "init")
    _git(path, "config", "user.email", "release@example.invalid")
    _git(path, "config", "user.name", "Release Gate Test")
    (path / ".gitignore").write_text(".audit/\n", encoding="utf-8")
    (path / "README.md").write_text("# safe\n", encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "-m", "safe")
    return path, _git(path, "rev-parse", "HEAD")


def test_git_object_scan_passes_safe_tree_and_binds_complete_history(tmp_path: Path) -> None:
    repository, commit = _repository(tmp_path / "repository")
    policy = _policy(tmp_path / "policy.json")

    report = scan_repository(
        repository_root=repository,
        candidate_commit=commit,
        policy_path=policy,
        created_at_utc="2026-08-24T12:00:00Z",
    )

    assert report["record_kind"] == SCAN_KIND
    assert report["status"] == "pass"
    assert report["commit_count"] == 1
    assert report["unique_blob_count"] == 2
    assert report["raw_worktree_paths_opened"] is False
    body = dict(report)
    assert body.pop("record_sha256") == canonical_json_sha256(body)


def test_git_object_scan_rejects_forbidden_payload_even_after_deletion(tmp_path: Path) -> None:
    repository, _commit_value = _repository(tmp_path / "repository")
    policy = _policy(tmp_path / "policy.json")
    archive = repository / "data/raw/source.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"PK\x03\x04historical")
    _git(repository, "add", "-f", "data/raw/source.zip")
    _git(repository, "commit", "-m", "unsafe historical payload")
    archive.unlink()
    _git(repository, "add", "-u")
    _git(repository, "commit", "-m", "remove payload")
    candidate = _git(repository, "rev-parse", "HEAD")

    report = scan_repository(
        repository_root=repository,
        candidate_commit=candidate,
        policy_path=policy,
        created_at_utc="2026-08-24T12:00:00Z",
    )

    assert report["status"] == "fail"
    assert {item["code"] for item in report["violations"]} >= {
        "forbidden_path",
        "forbidden_extension",
    }


def test_git_object_scan_rejects_disguised_archive(tmp_path: Path) -> None:
    repository, _commit_value = _repository(tmp_path / "repository")
    policy = _policy(tmp_path / "policy.json")
    (repository / "innocent.md").write_bytes(b"PK\x03\x04not markdown")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "disguised")

    report = scan_repository(
        repository_root=repository,
        candidate_commit=_git(repository, "rev-parse", "HEAD"),
        policy_path=policy,
        created_at_utc="2026-08-24T12:00:00Z",
    )
    assert report["status"] == "fail"
    assert any(item["code"] == "disguised_binary_signature" for item in report["violations"])


def test_secret_and_license_attestations_are_version_and_candidate_bound(tmp_path: Path) -> None:
    policy = _policy(tmp_path / "policy.json")
    gitleaks = tmp_path / "gitleaks.json"
    _write_json(gitleaks, [])
    commit = "a" * 40
    secret = validate_secret_scan(
        report_path=gitleaks,
        candidate_commit=commit,
        policy_path=policy,
        gitleaks_version="8.30.1",
        created_at_utc="2026-08-24T12:00:00Z",
    )
    assert secret["record_kind"] == SECRET_KIND
    assert secret["status"] == "pass"
    with pytest.raises(ReleaseGateError, match="version differs"):
        validate_secret_scan(
            report_path=gitleaks,
            candidate_commit=commit,
            policy_path=policy,
            gitleaks_version="8.29.0",
            created_at_utc="2026-08-24T12:00:00Z",
        )

    licenses = tmp_path / "licenses.json"
    _write_json(licenses, [{"Name": "safe", "Version": "1", "License": "MIT"}])
    audit = audit_licenses(
        inventory_path=licenses,
        candidate_commit=commit,
        policy_path=policy,
        pip_licenses_version="5.5.5",
        created_at_utc="2026-08-24T12:00:00Z",
    )
    assert audit["record_kind"] == LICENSE_KIND
    assert audit["status"] == "pass"
    _write_json(licenses, [{"Name": "unsafe", "Version": "1", "License": "UNKNOWN"}])
    assert (
        audit_licenses(
            inventory_path=licenses,
            candidate_commit=commit,
            policy_path=policy,
            pip_licenses_version="5.5.5",
            created_at_utc="2026-08-24T12:00:00Z",
        )["status"]
        == "fail"
    )


def test_tracked_report_is_truthfully_pending_and_rejects_self_rehashed_overclaim(
    tmp_path: Path,
) -> None:
    repository, commit = _repository(tmp_path / "repository")
    policy = _policy(repository / "configs/release/release_gate_policy_v1.json")
    report = assemble_report(
        repository_root=repository,
        policy_path=policy,
        mode="tracked_precommit_report",
        created_at_utc="2026-08-24T12:00:00Z",
    )
    assert report["repository"] == {
        "parent_commit": commit,
        "content_commit": None,
        "worktree_clean": False,
    }
    assert report["status"] == "pending"
    path = repository / "report.json"
    _write_json(path, report)
    assert validate_report(path, repository_root=repository)["valid"] is True

    report["status"] = "pass"
    report.pop("record_sha256")
    report["record_sha256"] = canonical_json_sha256(report)
    _write_json(path, report)
    validation = validate_report(path, repository_root=repository)
    assert validation["valid"] is False
    assert "overstates" in validation["errors"][0]


def test_exact_candidate_attestation_revalidates_every_pinned_gate(tmp_path: Path) -> None:
    repository, parent = _repository(tmp_path / "repository")
    policy = _policy(repository / "configs/release/release_gate_policy_v1.json")
    _git(repository, "add", "configs/release/release_gate_policy_v1.json")
    _git(repository, "commit", "-m", "release policy")
    candidate = _git(repository, "rev-parse", "HEAD")
    evidence_root = repository / ".audit/release-attestations" / candidate
    policy_payload = policy.read_bytes()
    policy_reference = {
        "path": "configs/release/release_gate_policy_v1.json",
        "size_bytes": len(policy_payload),
        "file_sha256": hashlib.sha256(policy_payload).hexdigest(),
    }
    quality = {
        "schema_version": "1.0.0",
        "record_kind": "synthetic_release_gate_evidence",
        "status": "pass",
        "candidate_commit": candidate,
        "policy": policy_reference,
        "gates": list(
            (
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
        ),
    }
    quality["record_sha256"] = canonical_json_sha256(quality)
    quality_path = evidence_root / "quality.json"
    _write_json(quality_path, quality)
    relative = quality_path.relative_to(repository).as_posix()
    reference = {
        "status": "pass",
        "path": relative,
        "expected_sha256": hashlib.sha256(quality_path.read_bytes()).hexdigest(),
    }
    spec = {
        "schema_version": "1.0.0",
        "spec_kind": "final_release_gate_attestation_spec",
        "created_at_utc": "2026-08-24T12:00:00Z",
        "candidate_commit": candidate,
        "parent_commit": parent,
        "worktree_clean": True,
        "gates": {name: dict(reference) for name in quality["gates"]},
    }
    spec_path = evidence_root / "spec.json"
    _write_json(spec_path, spec)

    report = assemble_report(
        repository_root=repository,
        policy_path=policy,
        mode="exact_candidate_attestation",
        created_at_utc="2026-08-24T12:00:00Z",
        spec_path=spec_path,
    )
    report_path = evidence_root / "final_release_gate_report.json"
    _write_json(report_path, report)
    assert validate_report(report_path, repository_root=repository)["valid"] is True

    quality["candidate_commit"] = "f" * 40
    quality.pop("record_sha256")
    quality["record_sha256"] = canonical_json_sha256(quality)
    _write_json(quality_path, quality)
    failed = validate_report(report_path, repository_root=repository)
    assert failed["valid"] is False
    assert "SHA-256 differs" in failed["errors"][0]


def test_release_policy_pins_supply_chain_and_forbids_model_payloads(
    repository_root: Path,
) -> None:
    policy_path = repository_root / "configs/release/release_gate_policy_v1.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    assert policy["gitleaks"]["version"] == "8.30.1"
    assert policy["gitleaks"]["linux_x64_archive_sha256"] == (
        "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb"
    )
    assert ".onnx" in policy["forbidden_suffixes"]
    assert policy["maximum_blob_size_bytes"] == 16 * 1024 * 1024
    assert policy["ci_action_pins"] == {
        "actions/checkout": "11d5960a326750d5838078e36cf38b85af677262",
        "actions/setup-python": "a26af69be951a213d495a4c3e4e4022e16d87065",
        "actions/upload-artifact": "ea165f8d65b6e75b540449e92b4886f43607fa02",
        "astral-sh/setup-uv": "d0cc045d04ccac9d8b7881df0226f9e82c39688e",
    }


def test_global_ignore_policy_covers_release_forbidden_payloads(repository_root: Path) -> None:
    lines = set((repository_root / ".gitignore").read_text(encoding="utf-8").splitlines())
    assert {"*.zip", "*.npy", "*.npz", "*.docx", "*.onnx"} <= lines
    assert hashlib.sha256(
        (repository_root / "configs/release/release_gate_policy_v1.json").read_bytes()
    ).hexdigest()


def test_ci_uses_immutable_actions_and_preserves_failed_security_evidence(
    repository_root: Path,
) -> None:
    text = (repository_root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    for expected in (
        "actions/checkout@11d5960a326750d5838078e36cf38b85af677262",
        "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065",
        "astral-sh/setup-uv@d0cc045d04ccac9d8b7881df0226f9e82c39688e",
        "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02",
        "fetch-depth: 0",
        "persist-credentials: false",
        "if: always()",
        "exact_candidate_attestation",
        "final_release_gate_report.json",
        "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb",
    ):
        assert expected in text
    assert "actions/checkout@v" not in text
    assert "actions/setup-python@v" not in text
    assert "astral-sh/setup-uv@v" not in text
    assert "actions/upload-artifact@v" not in text
