from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from inclusive_shift_har.artifacts.github_evidence import (
    COMPONENT_FILES,
    PROVENANCE_FILES,
    GitHubEvidenceError,
    capture_ci_evidence,
    capture_remote_evidence,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256

CANDIDATE = "a" * 40
REPOSITORY = "example/inclusive-shift-har"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _record_bytes(**values: Any) -> bytes:
    record = dict(values)
    record["record_sha256"] = canonical_json_sha256(record)
    return (json.dumps(record, sort_keys=True) + "\n").encode()


def test_capture_remote_evidence_binds_private_main_ref_create_only(tmp_path: Path) -> None:
    repository_response = tmp_path / "repository.json"
    ref_response = tmp_path / "main-ref.json"
    output_root = tmp_path / "output"
    output_root.mkdir()
    _write_json(
        repository_response,
        {
            "full_name": REPOSITORY,
            "html_url": f"https://github.com/{REPOSITORY}",
            "url": f"https://api.github.com/repos/{REPOSITORY}",
            "private": True,
            "visibility": "private",
            "default_branch": "main",
        },
    )
    _write_json(
        ref_response,
        {
            "ref": "refs/heads/main",
            "url": f"https://api.github.com/repos/{REPOSITORY}/git/refs/heads/main",
            "object": {"type": "commit", "sha": CANDIDATE},
        },
    )

    record = capture_remote_evidence(
        repository_response_path=repository_response,
        main_ref_response_path=ref_response,
        candidate_commit=CANDIDATE,
        queried_at_utc="2026-08-25T12:00:00Z",
        output_path="remote.json",
        allowed_output_root=output_root,
    )

    assert record["visibility"] == "private"
    assert record["ref_commit"] == CANDIDATE
    assert (
        record["api_response_files"]["repository"]["file_sha256"]
        == hashlib.sha256(repository_response.read_bytes()).hexdigest()
    )
    body = dict(record)
    assert body.pop("record_sha256") == canonical_json_sha256(body)
    with pytest.raises(FileExistsError):
        capture_remote_evidence(
            repository_response_path=repository_response,
            main_ref_response_path=ref_response,
            candidate_commit=CANDIDATE,
            queried_at_utc="2026-08-25T12:00:00Z",
            output_path="remote.json",
            allowed_output_root=output_root,
        )


def test_capture_remote_evidence_rejects_noncanonical_repository(tmp_path: Path) -> None:
    repository_response = tmp_path / "repository.json"
    ref_response = tmp_path / "main-ref.json"
    output_root = tmp_path / "output"
    output_root.mkdir()
    _write_json(
        repository_response,
        {
            "full_name": "example/.",
            "html_url": "https://github.com/example/.",
            "url": "https://api.github.com/repos/example/.",
            "private": True,
            "visibility": "private",
            "default_branch": "main",
        },
    )
    _write_json(
        ref_response,
        {
            "ref": "refs/heads/main",
            "url": "https://api.github.com/repos/example/./git/refs/heads/main",
            "object": {"type": "commit", "sha": CANDIDATE},
        },
    )

    with pytest.raises(GitHubEvidenceError, match="canonical owner/repository"):
        capture_remote_evidence(
            repository_response_path=repository_response,
            main_ref_response_path=ref_response,
            candidate_commit=CANDIDATE,
            queried_at_utc="2026-08-25T12:00:00Z",
            output_path="remote.json",
            allowed_output_root=output_root,
        )


def test_capture_remote_evidence_rejects_temp_clone_token_field(tmp_path: Path) -> None:
    repository_response = tmp_path / "repository.json"
    ref_response = tmp_path / "main-ref.json"
    output_root = tmp_path / "output"
    output_root.mkdir()
    _write_json(
        repository_response,
        {
            "full_name": REPOSITORY,
            "html_url": f"https://github.com/{REPOSITORY}",
            "url": f"https://api.github.com/repos/{REPOSITORY}",
            "private": True,
            "visibility": "private",
            "default_branch": "main",
            "temp_clone_token": None,
        },
    )
    _write_json(
        ref_response,
        {
            "ref": "refs/heads/main",
            "url": f"https://api.github.com/repos/{REPOSITORY}/git/refs/heads/main",
            "object": {"type": "commit", "sha": CANDIDATE},
        },
    )

    with pytest.raises(GitHubEvidenceError, match="forbidden temp_clone_token field"):
        capture_remote_evidence(
            repository_response_path=repository_response,
            main_ref_response_path=ref_response,
            candidate_commit=CANDIDATE,
            queried_at_utc="2026-08-25T12:00:00Z",
            output_path="remote.json",
            allowed_output_root=output_root,
        )
    assert not (output_root / "remote.json").exists()


def _ci_inputs(tmp_path: Path, *, unsafe_member: str | None = None) -> tuple[Path, Path, Path]:
    run_response = tmp_path / "run.json"
    artifact_response = tmp_path / "artifact.json"
    archive = tmp_path / "artifact.zip"
    _write_json(
        run_response,
        {
            "id": 123,
            "run_attempt": 1,
            "status": "completed",
            "conclusion": "success",
            "event": "push",
            "head_branch": "main",
            "head_sha": CANDIDATE,
            "path": ".github/workflows/ci.yml",
            "html_url": f"https://github.com/{REPOSITORY}/actions/runs/123",
            "repository": {"full_name": REPOSITORY},
        },
    )
    created_at = "2026-08-25T11:59:00Z"
    kinds = {
        "repository_scan.json": "git_object_release_scan",
        "secret_scan.json": "gitleaks_secret_scan_attestation",
        "license_audit.json": "python_license_audit",
        "quality_gates.json": "ci_release_quality_gate_evidence",
    }
    files = {
        name: _record_bytes(
            schema_version="1.0.0",
            record_kind=kind,
            status="pass",
            candidate_commit=CANDIDATE,
            created_at_utc=created_at,
        )
        for name, kind in kinds.items()
    }
    prefix = f".audit/release-attestations/{CANDIDATE}/"
    evidence = {
        gate_name: {
            "path": f"{prefix}{basename}",
            "size_bytes": len(files[basename]),
            "file_sha256": hashlib.sha256(files[basename]).hexdigest(),
        }
        for gate_name, basename in {
            "repository_scan": "repository_scan.json",
            "secret_scan": "secret_scan.json",
            "license_audit": "license_audit.json",
            **{
                gate_name: "quality_gates.json"
                for gate_name in (
                    "tests",
                    "lint",
                    "format",
                    "types",
                    "manifests",
                    "splits",
                    "configuration",
                    "artifacts",
                    "ci_quality_proxy",
                )
            },
        }.items()
    }
    files["ci_bundle_validation.json"] = _record_bytes(
        schema_version="1.0.0",
        record_kind="ci_release_gate_bundle_validation",
        status="pass",
        created_at_utc=created_at,
        candidate_commit=CANDIDATE,
        parent_commit="b" * 40,
        ci_evidence_scope="within_job_quality_proxy_not_completed_workflow",
        external_completed_ci_required_for_release_inventory=True,
        validated_gates=sorted(evidence),
        evidence=evidence,
    )
    files.update({name: f"{name}\n".encode() for name in PROVENANCE_FILES})
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in (*COMPONENT_FILES, *PROVENANCE_FILES):
            member = (
                unsafe_member
                if unsafe_member is not None and name == COMPONENT_FILES[0]
                else (f"{prefix}{name}")
            )
            bundle.writestr(member, files[name])
    archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive_size = archive.stat().st_size
    artifact = {
        "id": 456,
        "name": f"release-security-{CANDIDATE}",
        "expired": False,
        "url": f"https://api.github.com/repos/{REPOSITORY}/actions/artifacts/456",
        "archive_download_url": (
            f"https://api.github.com/repos/{REPOSITORY}/actions/artifacts/456/zip"
        ),
        "size_in_bytes": archive_size,
        "digest": f"sha256:{archive_sha256}",
        "workflow_run": {"id": 123, "head_sha": CANDIDATE},
    }
    _write_json(
        artifact_response,
        {
            "total_count": 1,
            "artifacts": [artifact],
        },
    )
    return run_response, artifact_response, archive


def test_capture_ci_evidence_validates_archive_and_mirrors_sanitized_files(
    tmp_path: Path,
) -> None:
    run_response, artifact_response, archive = _ci_inputs(tmp_path)
    output_root = tmp_path / "external"
    output_root.mkdir()

    record = capture_ci_evidence(
        workflow_run_response_path=run_response,
        artifact_response_path=artifact_response,
        downloaded_archive_path=archive,
        candidate_commit=CANDIDATE,
        queried_at_utc="2026-08-25T12:00:00Z",
        extraction_root=output_root,
        output_path="ci.json",
        allowed_output_root=output_root,
    )

    extracted = output_root / ".audit" / "release-attestations" / CANDIDATE
    assert {path.name for path in extracted.iterdir()} == set((*COMPONENT_FILES, *PROVENANCE_FILES))
    assert record["artifact_digest"] == f"sha256:{hashlib.sha256(archive.read_bytes()).hexdigest()}"
    assert (
        record["release_gate_file_sha256s"]["repository_scan"]
        == hashlib.sha256((extracted / "repository_scan.json").read_bytes()).hexdigest()
    )
    assert "ci_bundle_validation" in record["release_gate_file_sha256s"]
    assert (
        record["downloaded_artifact_archive"]["file_sha256"]
        == hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    assert set(record["artifact_member_sha256s"]) == set((*COMPONENT_FILES, *PROVENANCE_FILES))
    bundle = json.loads((extracted / "ci_bundle_validation.json").read_text(encoding="utf-8"))
    assert "ci_quality_proxy" in bundle["validated_gates"]
    assert bundle["evidence"]["ci_quality_proxy"]["path"].endswith("/quality_gates.json")
    assert (
        record["api_response_files"]["workflow_run"]["file_sha256"]
        == hashlib.sha256(run_response.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "unsafe_member",
    [
        "../outside.txt",
        f".audit//release-attestations/{CANDIDATE}/repository_scan.json",
        f".audit/./release-attestations/{CANDIDATE}/repository_scan.json",
    ],
)
def test_capture_ci_evidence_rejects_archive_path_injection(
    tmp_path: Path, unsafe_member: str
) -> None:
    run_response, artifact_response, archive = _ci_inputs(tmp_path, unsafe_member=unsafe_member)
    output_root = tmp_path / "external"
    output_root.mkdir()

    with pytest.raises(GitHubEvidenceError, match="noncanonical member path"):
        capture_ci_evidence(
            workflow_run_response_path=run_response,
            artifact_response_path=artifact_response,
            downloaded_archive_path=archive,
            candidate_commit=CANDIDATE,
            queried_at_utc="2026-08-25T12:00:00Z",
            extraction_root=output_root,
            output_path="ci.json",
            allowed_output_root=output_root,
        )


def test_capture_ci_rejects_digest_before_publishing(tmp_path: Path) -> None:
    run_response, artifact_response, archive = _ci_inputs(tmp_path)
    response = json.loads(artifact_response.read_text(encoding="utf-8"))
    response["artifacts"][0]["digest"] = f"sha256:{'0' * 64}"
    _write_json(artifact_response, response)
    output_root = tmp_path / "external"
    output_root.mkdir()

    with pytest.raises(GitHubEvidenceError, match="digest or size"):
        capture_ci_evidence(
            workflow_run_response_path=run_response,
            artifact_response_path=artifact_response,
            downloaded_archive_path=archive,
            candidate_commit=CANDIDATE,
            queried_at_utc="2026-08-25T12:00:00Z",
            extraction_root=output_root,
            output_path="ci.json",
            allowed_output_root=output_root,
        )

    assert not (output_root / ".audit").exists()
    assert not (output_root / "ci.json").exists()


def test_capture_ci_rejects_boolean_artifact_workflow_id(tmp_path: Path) -> None:
    run_response, artifact_response, archive = _ci_inputs(tmp_path)
    response = json.loads(artifact_response.read_text(encoding="utf-8"))
    response["artifacts"][0]["workflow_run"]["id"] = True
    _write_json(artifact_response, response)
    output_root = tmp_path / "external"
    output_root.mkdir()

    with pytest.raises(GitHubEvidenceError, match="positive integer"):
        capture_ci_evidence(
            workflow_run_response_path=run_response,
            artifact_response_path=artifact_response,
            downloaded_archive_path=archive,
            candidate_commit=CANDIDATE,
            queried_at_utc="2026-08-25T12:00:00Z",
            extraction_root=output_root,
            output_path="ci.json",
            allowed_output_root=output_root,
        )

    assert not (output_root / ".audit").exists()


def test_capture_ci_rejects_float_artifact_total_count(tmp_path: Path) -> None:
    run_response, artifact_response, archive = _ci_inputs(tmp_path)
    response = json.loads(artifact_response.read_text(encoding="utf-8"))
    response["total_count"] = 1.0
    _write_json(artifact_response, response)
    output_root = tmp_path / "external"
    output_root.mkdir()

    with pytest.raises(GitHubEvidenceError, match="positive integer"):
        capture_ci_evidence(
            workflow_run_response_path=run_response,
            artifact_response_path=artifact_response,
            downloaded_archive_path=archive,
            candidate_commit=CANDIDATE,
            queried_at_utc="2026-08-25T12:00:00Z",
            extraction_root=output_root,
            output_path="ci.json",
            allowed_output_root=output_root,
        )

    assert not (output_root / ".audit").exists()


def test_capture_ci_rejects_extraction_ancestor_symlink(tmp_path: Path) -> None:
    run_response, artifact_response, archive = _ci_inputs(tmp_path)
    output_root = tmp_path / "external"
    output_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (output_root / ".audit").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    with pytest.raises(GitHubEvidenceError, match="traverse a symlink"):
        capture_ci_evidence(
            workflow_run_response_path=run_response,
            artifact_response_path=artifact_response,
            downloaded_archive_path=archive,
            candidate_commit=CANDIDATE,
            queried_at_utc="2026-08-25T12:00:00Z",
            extraction_root=output_root,
            output_path="ci.json",
            allowed_output_root=output_root,
        )

    assert not list(outside.iterdir())


def test_capture_ci_preflights_existing_candidate_directory(tmp_path: Path) -> None:
    run_response, artifact_response, archive = _ci_inputs(tmp_path)
    output_root = tmp_path / "external"
    destination = output_root / ".audit" / "release-attestations" / CANDIDATE
    destination.mkdir(parents=True)
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        capture_ci_evidence(
            workflow_run_response_path=run_response,
            artifact_response_path=artifact_response,
            downloaded_archive_path=archive,
            candidate_commit=CANDIDATE,
            queried_at_utc="2026-08-25T12:00:00Z",
            extraction_root=output_root,
            output_path="ci.json",
            allowed_output_root=output_root,
        )

    assert {path.name for path in destination.iterdir()} == {"sentinel.txt"}


def test_capture_ci_preflights_output_collision_before_mirroring(tmp_path: Path) -> None:
    run_response, artifact_response, archive = _ci_inputs(tmp_path)
    output_root = tmp_path / "external"
    output_root.mkdir()
    output = output_root / "ci.json"
    output.write_text("preserve\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        capture_ci_evidence(
            workflow_run_response_path=run_response,
            artifact_response_path=artifact_response,
            downloaded_archive_path=archive,
            candidate_commit=CANDIDATE,
            queried_at_utc="2026-08-25T12:00:00Z",
            extraction_root=output_root,
            output_path="ci.json",
            allowed_output_root=output_root,
        )

    assert output.read_text(encoding="utf-8") == "preserve\n"
    assert not (output_root / ".audit").exists()
