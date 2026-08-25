from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pytest

from inclusive_shift_har.artifacts import release_bundle
from inclusive_shift_har.artifacts.github_evidence import EXPECTED_ARTIFACT_FILES
from inclusive_shift_har.artifacts.release_bundle import (
    LOCAL_GATE_COMMANDS,
    MANIFEST_ARCHIVE_PATH,
    SPEC_ARCHIVE_PATH,
    ReleaseBundleError,
    _create_new_directory,
    _validate_member_payload,
    assemble_release_bundle_manifest,
    build_release_bundle,
    create_standard_release_bundle_spec,
    validate_release_bundle_archive,
    write_release_notes,
)
from inclusive_shift_har.artifacts.release_gate import scan_index
from inclusive_shift_har.manifests.canonical import canonical_json_sha256

from .test_release_gate import _git, _policy, _repository


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_json(path: Path, value: Any) -> None:
    _write(path, (json.dumps(value, sort_keys=True) + "\n").encode())


def _self_hashed(**values: Any) -> dict[str, Any]:
    values["record_sha256"] = canonical_json_sha256(values)
    return values


def _bundle_workspace(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    candidate = "a" * 40
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    members: list[dict[str, str]] = []

    def member(role: str, path: str, payload: bytes = b"{}\n") -> None:
        _write(source / Path(*path.split("/")), payload)
        members.append({"role": role, "path": path})

    inventory_path = "final_release_evidence_inventory.json"
    release_spec_path = "release-spec.json"
    member("release_inventory", inventory_path)
    member("release_inventory_spec", release_spec_path)
    member("github_remote_evidence", "remote.json")
    member("github_completed_ci_evidence", "ci.json")
    for index in range(4):
        member("github_api_response", f"github/api-{index}.json")
    actions_path = f"release-security-{candidate}.zip"
    actions_file = source / actions_path
    with zipfile.ZipFile(actions_file, "w") as archive:
        archive.writestr("reviewed.txt", b"reviewed\n")
    members.append({"role": "github_actions_archive", "path": actions_path})
    member(
        "candidate_attestation",
        f".audit/release-attestations/{candidate}/final_release_gate_report.json",
    )
    member(
        "candidate_attestation_spec",
        f".audit/release-attestations/{candidate}/attestation_spec.json",
    )
    member(
        "staged_index_scan",
        f".audit/release-attestations/{candidate}/staged_index_scan.json",
    )
    member("staged_secret_evidence", "local/staged_secret_scan.json")
    member("local_cuda_evidence", "local/cuda_gate.json")
    member("local_license_evidence", "local/license_audit.json")
    member("local_quality_log", "local/quality.log", b"all local gates passed\n")
    for index in range(45):
        member(
            "ci_mirror",
            f".audit/release-attestations/{candidate}/ci-member-{index:02d}.txt",
            f"ci member {index}\n".encode(),
        )
    bundle_spec = {
        "schema_version": "1.0.0",
        "spec_kind": "release_evidence_bundle_spec",
        "created_at_utc": "2026-08-25T12:00:00Z",
        "candidate_commit": candidate,
        "inventory_path": inventory_path,
        "release_spec_path": release_spec_path,
        "members": members,
    }
    spec_path = source / "bundle-spec.json"
    _write_json(spec_path, bundle_spec)
    return source, output, spec_path, candidate


def _build(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    source, output, spec_path, candidate = _bundle_workspace(tmp_path)
    manifest_path = output / "bundle-manifest.json"
    assemble_release_bundle_manifest(
        bundle_spec_path=spec_path,
        source_root=source,
        created_at_utc="2026-08-25T12:01:00Z",
        output_path=manifest_path,
        allowed_output_root=output,
    )
    archive_path = output / "bundle.zip"
    build_release_bundle(
        bundle_spec_path=spec_path,
        manifest_path=manifest_path,
        source_root=source,
        output_path=archive_path,
        allowed_output_root=output,
    )
    return spec_path, manifest_path, archive_path, candidate


def _standard_workspace(tmp_path: Path) -> tuple[Path, Path, str]:
    repository, _initial = _repository(tmp_path / "repository")
    fake_executable = tmp_path / "gitleaks.exe"
    fake_executable.write_bytes(b"synthetic-gitleaks")
    policy_source = _policy(tmp_path / "policy.json")
    policy = json.loads(policy_source.read_text(encoding="utf-8"))
    policy["gitleaks"].update(
        {
            "windows_x64_executable": "gitleaks.exe",
            "windows_x64_executable_size_bytes": fake_executable.stat().st_size,
            "windows_x64_executable_sha256": hashlib.sha256(
                fake_executable.read_bytes()
            ).hexdigest(),
        }
    )
    tracked_policy = repository / "configs" / "release" / "release_gate_policy_v1.json"
    _write_json(tracked_policy, policy)
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "add release policy")
    parent = _git(repository, "rev-parse", "HEAD")
    (repository / "README.md").write_text("# candidate\n", encoding="utf-8")
    _git(repository, "add", "README.md")
    staged = scan_index(
        repository_root=repository,
        policy_path=tracked_policy,
        created_at_utc="2026-08-25T12:00:00Z",
    )
    _git(repository, "commit", "-m", "candidate")
    candidate = _git(repository, "rev-parse", "HEAD")

    source = tmp_path / "source"
    source.mkdir()
    mirror = source / ".audit" / "release-attestations" / candidate
    for basename in EXPECTED_ARTIFACT_FILES:
        _write(mirror / basename, f"{basename}\n".encode())
    staged_path = mirror / "staged_index_scan.json"
    _write_json(staged_path, staged)
    _write(mirror / "attestation_spec.json", b"{}\n")
    _write(mirror / "final_release_gate_report.json", b"{}\n")
    for basename in (
        "final_release_evidence_inventory.json",
        "release-spec.json",
        "remote.json",
        "ci.json",
        "repository-api.json",
        "main-ref-api.json",
        "workflow-run-api.json",
        "run-artifacts-api.json",
    ):
        _write(source / basename, b"{}\n")
    with zipfile.ZipFile(source / f"release-security-{candidate}.zip", "w") as archive:
        archive.writestr("reviewed.txt", b"reviewed\n")

    policy_payload = tracked_policy.read_bytes()
    policy_reference = {
        "path": "configs/release/release_gate_policy_v1.json",
        "size_bytes": len(policy_payload),
        "file_sha256": hashlib.sha256(policy_payload).hexdigest(),
    }

    def tracked_reference(path: str) -> dict[str, Any]:
        payload = (repository / path).read_bytes()
        return {
            "path": path,
            "size_bytes": len(payload),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
        }

    local = source / "local"
    cuda_payload = b'{"status":"pass"}\n'
    _write(local / "cuda_allocation.log", cuda_payload)
    gates: dict[str, Any] = {}
    for gate_name, command in LOCAL_GATE_COMMANDS.items():
        log_payload = f"{gate_name} pass\n".encode()
        _write(local / f"{gate_name}.log", log_payload)
        gates[gate_name] = {
            "status": "pass",
            "command": command,
            "started_at_utc": "2026-08-25T12:00:00Z",
            "completed_at_utc": "2026-08-25T12:00:01Z",
            "exit_code": 0,
            "raw_log": {
                "basename": f"{gate_name}.raw.log",
                "size_bytes": len(log_payload),
                "file_sha256": hashlib.sha256(log_payload).hexdigest(),
                "retained_locally_only": True,
            },
            "sanitized_log": {
                "basename": f"{gate_name}.log",
                "size_bytes": len(log_payload),
                "file_sha256": hashlib.sha256(log_payload).hexdigest(),
                "path": f"local/{gate_name}.log",
                "replacements": [],
            },
        }
    _write_json(
        local / "local_cuda_gate.json",
        _self_hashed(
            schema_version="1.0.0",
            record_kind="local_cuda_release_gate_evidence",
            status="pass",
            created_at_utc="2026-08-25T12:00:02Z",
            candidate_commit=candidate,
            parent_commit=parent,
            scope="candidate_bound_local_release_gates_without_target_evaluation",
            confirmatory_evaluator_invoked=False,
            cuda_required=True,
            cuda={
                "status": "pass",
                "started_at_utc": "2026-08-25T12:00:00Z",
                "completed_at_utc": "2026-08-25T12:00:01Z",
                "torch_version": "2.12.0+cu132",
                "cuda_build": "13.2",
                "device": "Synthetic CUDA GPU",
                "device_count": 1,
                "nvidia_smi": "Synthetic CUDA GPU, 12288, 596.72",
                "peak_allocated_bytes": 4096,
                "allocation_checksum": 1.25,
            },
            cuda_log={
                "basename": "cuda_allocation.log",
                "size_bytes": len(cuda_payload),
                "file_sha256": hashlib.sha256(cuda_payload).hexdigest(),
                "path": "local/cuda_allocation.log",
            },
            machine={"platform": "test", "architecture": "x86_64", "python": "3.11.9"},
            tool_versions={
                "uv": "uv 0.11.29",
                "pytest": "pytest 9.0.2",
                "ruff": "ruff 0.15.4",
                "mypy": "mypy 1.19.1",
            },
            gates=gates,
        ),
    )
    staged_payload = staged_path.read_bytes()
    staged_report_payload = b"[]\n"
    _write_json(
        local / "staged_secret_scan.json",
        _self_hashed(
            schema_version="1.0.0",
            record_kind="staged_gitleaks_evidence",
            status="pass",
            created_at_utc="2026-08-25T12:00:03Z",
            candidate_commit=candidate,
            parent_commit=parent,
            scan_scope="exact_precommit_git_index",
            gitleaks_version="8.30.1",
            gitleaks_exit_code=0,
            finding_count=0,
            gitleaks_executable={
                "basename": "gitleaks.exe",
                "size_bytes": fake_executable.stat().st_size,
                "file_sha256": hashlib.sha256(fake_executable.read_bytes()).hexdigest(),
            },
            raw_report={
                "basename": "staged-gitleaks.json",
                "size_bytes": len(staged_report_payload),
                "file_sha256": hashlib.sha256(staged_report_payload).hexdigest(),
                "included_in_release_bundle": False,
            },
            staged_index_scan={
                "basename": "staged_index_scan.json",
                "size_bytes": len(staged_payload),
                "file_sha256": hashlib.sha256(staged_payload).hexdigest(),
                "record_sha256": staged["record_sha256"],
                "index_entry_index_sha256": staged["index_entry_index_sha256"],
            },
            policy=policy_reference,
            config=tracked_reference(".gitleaks.toml"),
            ignore=tracked_reference(".gitleaksignore"),
        ),
    )
    inventory_payload = b'[{"Name":"safe","Version":"1","License":"MIT"}]\n'
    _write_json(
        local / "license_audit.json",
        _self_hashed(
            schema_version="1.0.0",
            record_kind="python_license_audit",
            status="pass",
            created_at_utc="2026-08-25T12:00:04Z",
            candidate_commit=candidate,
            pip_licenses_version="5.5.5",
            policy=policy_reference,
            inventory={
                "basename": "python_licenses.json",
                "size_bytes": len(inventory_payload),
                "file_sha256": hashlib.sha256(inventory_payload).hexdigest(),
            },
            package_count=1,
            normalized_inventory_sha256=hashlib.sha256(b"safe\t1\tMIT\n").hexdigest(),
            violations=[],
        ),
    )
    return source, repository, candidate


def test_release_bundle_build_and_exact_structural_validation(tmp_path: Path) -> None:
    spec, manifest, archive, candidate = _build(tmp_path)

    report = validate_release_bundle_archive(
        archive_path=archive,
        bundle_spec_path=spec,
        manifest_path=manifest,
        candidate_commit=candidate,
        created_at_utc="2026-08-25T12:02:00Z",
    )

    assert report["status"] == "pass"
    assert report["extra_member_count"] == 0
    assert report["offline_inventory_reconstructed"] is False
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
    assert SPEC_ARCHIVE_PATH in names
    assert MANIFEST_ARCHIVE_PATH in names
    assert len(names) == report["member_count"]


def test_release_bundle_rejects_unmanifested_outer_member(tmp_path: Path) -> None:
    spec, manifest, archive, candidate = _build(tmp_path)
    tampered = archive.with_name("bundle-with-extra.zip")
    shutil.copyfile(archive, tampered)
    with zipfile.ZipFile(tampered, "a") as bundle:
        bundle.writestr("unreviewed.txt", b"must not be accepted\n")

    with pytest.raises(ReleaseBundleError, match="member set differs"):
        validate_release_bundle_archive(
            archive_path=tampered,
            bundle_spec_path=spec,
            manifest_path=manifest,
            candidate_commit=candidate,
            created_at_utc="2026-08-25T12:02:00Z",
        )


@pytest.mark.parametrize("mutation", ["leading-prefix", "trailing-overlay", "archive-comment"])
def test_release_bundle_rejects_noncanonical_container_bytes(tmp_path: Path, mutation: str) -> None:
    spec, manifest, archive, candidate = _build(tmp_path)
    tampered = archive.with_name(f"bundle-{mutation}.zip")
    shutil.copyfile(archive, tampered)
    if mutation == "leading-prefix":
        tampered.write_bytes(b"SECRET-LEADING-PREFIX" + tampered.read_bytes())
    elif mutation == "trailing-overlay":
        with tampered.open("ab") as handle:
            handle.write(b"SECRET-TRAILING-OVERLAY")
    else:
        with zipfile.ZipFile(tampered, "a") as bundle:
            bundle.comment = b"SECRET-IN-ARCHIVE-COMMENT"

    with pytest.raises(ReleaseBundleError, match="canonical deterministic ZIP bytes"):
        validate_release_bundle_archive(
            archive_path=tampered,
            bundle_spec_path=spec,
            manifest_path=manifest,
            candidate_commit=candidate,
            created_at_utc="2026-08-25T12:02:00Z",
        )


@pytest.mark.parametrize("metadata", ["member-comment", "member-extra-field"])
def test_release_bundle_rejects_noncanonical_member_metadata(tmp_path: Path, metadata: str) -> None:
    spec, manifest, archive, candidate = _build(tmp_path)
    tampered = archive.with_name(f"bundle-{metadata}.zip")
    with zipfile.ZipFile(archive) as original:
        entries = [(info, original.read(info)) for info in original.infolist()]
    with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_STORED) as rewritten:
        for index, (original_info, payload) in enumerate(entries):
            info = zipfile.ZipInfo(original_info.filename, date_time=original_info.date_time)
            info.compress_type = original_info.compress_type
            info.create_system = original_info.create_system
            info.external_attr = original_info.external_attr
            if index == 0 and metadata == "member-comment":
                info.comment = b"SECRET-MEMBER-COMMENT"
            if index == 0 and metadata == "member-extra-field":
                info.extra = b"\xfe\xca\x04\x00TEST"
            rewritten.writestr(info, payload)

    with pytest.raises(ReleaseBundleError, match="canonical deterministic ZIP bytes"):
        validate_release_bundle_archive(
            archive_path=tampered,
            bundle_spec_path=spec,
            manifest_path=manifest,
            candidate_commit=candidate,
            created_at_utc="2026-08-25T12:02:00Z",
        )


def test_release_bundle_rejects_binary_and_escaped_absolute_path_payloads() -> None:
    with pytest.raises(ReleaseBundleError, match="nested ZIP payload"):
        _validate_member_payload(
            role="local_quality_log",
            path="local/quality.log",
            payload=b"PK\x05\x06\x00\x00binary",
        )
    with pytest.raises(ReleaseBundleError, match="absolute user path"):
        _validate_member_payload(
            role="local_quality_log",
            path="local/quality.log",
            payload=b'{"path":"C:\\\\Users\\\\DELL\\\\secret.txt"}\n',
        )


def test_release_bundle_offline_validation_requires_ready_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, repository, candidate = _standard_workspace(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    spec = source / "bundle-spec.json"
    create_standard_release_bundle_spec(
        source_root=source,
        repository_root=repository,
        candidate_commit=candidate,
        created_at_utc="2026-08-25T12:01:00Z",
        output_path=spec,
        allowed_output_root=source,
    )
    manifest = output / "bundle-manifest.json"
    assemble_release_bundle_manifest(
        bundle_spec_path=spec,
        source_root=source,
        created_at_utc="2026-08-25T12:02:00Z",
        output_path=manifest,
        allowed_output_root=output,
    )
    archive = output / "bundle.zip"
    build_release_bundle(
        bundle_spec_path=spec,
        manifest_path=manifest,
        source_root=source,
        output_path=archive,
        allowed_output_root=output,
    )
    monkeypatch.setattr(
        release_bundle,
        "validate_release_evidence_inventory_file",
        lambda *args, **kwargs: {"valid": True, "status": "draft", "errors": []},
    )

    with pytest.raises(ReleaseBundleError, match="ready status"):
        validate_release_bundle_archive(
            archive_path=archive,
            bundle_spec_path=spec,
            manifest_path=manifest,
            candidate_commit=candidate,
            repository_root=repository,
            extraction_root=tmp_path / "extracted",
            created_at_utc="2026-08-25T12:02:00Z",
        )


def test_release_bundle_rejects_source_change_after_manifest_review(tmp_path: Path) -> None:
    source, output, spec, _candidate = _bundle_workspace(tmp_path)
    manifest = output / "bundle-manifest.json"
    assemble_release_bundle_manifest(
        bundle_spec_path=spec,
        source_root=source,
        created_at_utc="2026-08-25T12:01:00Z",
        output_path=manifest,
        allowed_output_root=output,
    )
    (source / "remote.json").write_text('{"changed":true}\n', encoding="utf-8")

    with pytest.raises(ReleaseBundleError, match="changed after manifest review"):
        build_release_bundle(
            bundle_spec_path=spec,
            manifest_path=manifest,
            source_root=source,
            output_path=output / "bundle.zip",
            allowed_output_root=output,
        )


def test_release_bundle_build_is_create_only(tmp_path: Path) -> None:
    spec, manifest, archive, _candidate = _build(tmp_path)

    with pytest.raises(FileExistsError, match="overwrite release bundle"):
        build_release_bundle(
            bundle_spec_path=spec,
            manifest_path=manifest,
            source_root=spec.parent,
            output_path=archive,
            allowed_output_root=archive.parent,
        )


def test_release_bundle_extraction_directory_is_create_only(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="overwrite bundle extraction root"):
        _create_new_directory(existing, name="bundle extraction root")


def test_release_bundle_extraction_directory_rejects_symlink_parent(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    with pytest.raises(ReleaseBundleError, match="may not traverse a symlink"):
        _create_new_directory(linked / "extract", name="bundle extraction root")


def test_release_bundle_rejects_symlinked_control_input(tmp_path: Path) -> None:
    spec, manifest, archive, candidate = _build(tmp_path)
    linked = tmp_path / "bundle-spec-link.json"
    try:
        linked.symlink_to(spec)
    except OSError as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")

    with pytest.raises(ReleaseBundleError, match="may not traverse a symlink"):
        validate_release_bundle_archive(
            archive_path=archive,
            bundle_spec_path=linked,
            manifest_path=manifest,
            candidate_commit=candidate,
            created_at_utc="2026-08-25T12:02:00Z",
        )


def test_release_bundle_spec_rejects_raw_and_secret_members(tmp_path: Path) -> None:
    source, output, spec, _candidate = _bundle_workspace(tmp_path)
    value = json.loads(spec.read_text(encoding="utf-8"))
    value["members"].append({"role": "local_quality_log", "path": "data/raw/secret.env"})
    _write(source / "data" / "raw" / "secret.env", b"token=unsafe\n")
    _write_json(spec, value)

    with pytest.raises(ReleaseBundleError, match="forbidden raw/checkpoint path"):
        assemble_release_bundle_manifest(
            bundle_spec_path=spec,
            source_root=source,
            created_at_utc="2026-08-25T12:01:00Z",
            output_path=output / "bundle-manifest.json",
            allowed_output_root=output,
        )


def test_standard_bundle_spec_enumerates_only_expected_candidate_evidence(
    tmp_path: Path,
) -> None:
    source, repository, candidate = _standard_workspace(tmp_path)

    spec_path = source / "bundle-spec.json"
    spec = create_standard_release_bundle_spec(
        source_root=source,
        repository_root=repository,
        candidate_commit=candidate,
        created_at_utc="2026-08-25T12:00:00Z",
        output_path=spec_path,
        allowed_output_root=source,
    )

    assert len(spec["members"]) == 71
    assert sum(item["role"] == "ci_mirror" for item in spec["members"]) == 45
    assert {item["path"] for item in spec["members"] if item["role"] == "local_quality_log"} == {
        "local/cuda_allocation.log",
        *(f"local/{gate_name}.log" for gate_name in LOCAL_GATE_COMMANDS),
    }


def test_standard_bundle_spec_rejects_incomplete_local_gate_record(tmp_path: Path) -> None:
    source, repository, candidate = _standard_workspace(tmp_path)
    record_path = source / "local" / "local_cuda_gate.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["gates"].pop("tests")
    record.pop("record_sha256")
    record["record_sha256"] = canonical_json_sha256(record)
    _write_json(record_path, record)

    with pytest.raises(ReleaseBundleError, match="local CUDA gates keys differ"):
        create_standard_release_bundle_spec(
            source_root=source,
            repository_root=repository,
            candidate_commit=candidate,
            created_at_utc="2026-08-25T12:00:00Z",
            output_path=source / "bundle-spec.json",
            allowed_output_root=source,
        )


def test_release_notes_are_candidate_bound_and_create_only(tmp_path: Path) -> None:
    _spec, manifest, archive, candidate = _build(tmp_path)
    inventory = tmp_path / "final_release_evidence_inventory.json"
    _write_json(
        inventory,
        _self_hashed(
            schema_version="1.1.0",
            inventory_kind="release_evidence_inventory",
            status="ready",
            repository={"code_commit": candidate},
            confirmatory_state={"opening_count": 1, "target_rerun_permitted": False},
        ),
    )
    notes = tmp_path / "release-notes.md"

    result = write_release_notes(
        inventory_path=inventory,
        manifest_path=manifest,
        archive_path=archive,
        candidate_commit=candidate,
        output_path=notes,
        allowed_output_root=tmp_path,
    )

    assert result["status"] == "pass"
    text = notes.read_text(encoding="utf-8")
    assert candidate in text
    assert "opened once" in text
    assert "No DOI was minted" in text
    with pytest.raises(FileExistsError, match="overwrite release notes"):
        write_release_notes(
            inventory_path=inventory,
            manifest_path=manifest,
            archive_path=archive,
            candidate_commit=candidate,
            output_path=notes,
            allowed_output_root=tmp_path,
        )
