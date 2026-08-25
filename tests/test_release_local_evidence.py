from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from inclusive_shift_har.artifacts.release_gate import scan_index
from inclusive_shift_har.artifacts.release_local_evidence import (
    LocalReleaseEvidenceError,
    _resolve_input,
    _sanitize,
    attest_staged_gitleaks,
    capture_local_cuda_release_gates,
)

from .test_release_gate import _git, _policy, _repository, _write_json


def test_local_log_sanitization_removes_repository_and_user_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    payload = f"root={repository}\nprofile={tmp_path}\n".encode()

    sanitized, replacements = _sanitize(payload, repository_root=repository)

    assert str(repository).encode() not in sanitized
    assert str(tmp_path).encode() not in sanitized
    assert replacements == ["repository_root", "user_profile"]


def test_staged_gitleaks_attestation_binds_empty_report_to_candidate_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _initial = _repository(tmp_path / "repository")
    policy_path = _policy(tmp_path / "policy.json")
    fake_executable = tmp_path / "gitleaks.exe"
    fake_executable.write_bytes(b"synthetic-gitleaks")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["gitleaks"].update(
        {
            "windows_x64_executable_size_bytes": fake_executable.stat().st_size,
            "windows_x64_executable_sha256": hashlib.sha256(
                fake_executable.read_bytes()
            ).hexdigest(),
        }
    )
    _write_json(policy_path, policy)
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
    staged_path = tmp_path / "staged_index_scan.json"
    _write_json(staged_path, staged)
    _git(repository, "commit", "-m", "candidate")
    candidate = _git(repository, "rev-parse", "HEAD")
    assert staged["head_commit"] == parent
    report_path = tmp_path / "staged-gitleaks.json"
    _write_json(report_path, [])
    output_root = tmp_path / "output"
    output_root.mkdir()
    output = output_root / "staged-secret.json"
    original_run = subprocess.run

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        command = args[0]
        if command == [str(fake_executable.resolve()), "version"]:
            return subprocess.CompletedProcess(command, 0, stdout="8.30.1\n", stderr="")
        return original_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)

    record = attest_staged_gitleaks(
        repository_root=repository,
        candidate_commit=candidate,
        policy_path=tracked_policy,
        staged_index_path=staged_path,
        raw_report_path=report_path,
        gitleaks_executable=fake_executable,
        gitleaks_exit_code=0,
        created_at_utc="2026-08-25T12:01:00Z",
        output_path=output,
        allowed_output_root=output_root,
    )

    assert record["status"] == "pass"
    assert record["candidate_commit"] == candidate
    assert record["parent_commit"] == parent
    assert record["finding_count"] == 0
    assert record["raw_report"]["included_in_release_bundle"] is False
    assert record["policy"]["path"] == "configs/release/release_gate_policy_v1.json"
    assert record["config"]["path"] == ".gitleaks.toml"
    assert record["ignore"]["path"] == ".gitleaksignore"

    with pytest.raises(LocalReleaseEvidenceError, match="empty finding set"):
        attest_staged_gitleaks(
            repository_root=repository,
            candidate_commit=candidate,
            policy_path=tracked_policy,
            staged_index_path=staged_path,
            raw_report_path=report_path,
            gitleaks_executable=fake_executable,
            gitleaks_exit_code=False,
            created_at_utc="2026-08-25T12:02:00Z",
            output_path=output_root / "bool-exit.json",
            allowed_output_root=output_root,
        )


def test_local_evidence_input_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    linked = tmp_path / "linked.json"
    try:
        linked.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")

    with pytest.raises(LocalReleaseEvidenceError, match="may not traverse a symlink"):
        _resolve_input(linked, name="test input", directory=False)


def test_candidate_local_gate_capture_records_exact_cuda_and_gate_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _parent = _repository(tmp_path / "repository")
    (repository / "README.md").write_text("# candidate\n", encoding="utf-8")
    _git(repository, "add", "README.md")
    _git(repository, "commit", "-m", "candidate")
    candidate = _git(repository, "rev-parse", "HEAD")
    (repository / ".audit").mkdir()

    class FakeTensor:
        def __matmul__(self, other: object) -> FakeTensor:
            return self

        def __getitem__(self, key: object) -> FakeTensor:
            return self

        def item(self) -> float:
            return 1.25

    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: None,
        get_device_name=lambda index: "Synthetic CUDA GPU",
        device_count=lambda: 1,
        max_memory_allocated=lambda: 4096,
        empty_cache=lambda: None,
    )
    fake_torch = SimpleNamespace(
        __version__="2.12.0+cu132",
        version=SimpleNamespace(cuda="13.2"),
        cuda=fake_cuda,
        manual_seed=lambda seed: None,
        randn=lambda shape, device: FakeTensor(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_local_evidence.platform.platform",
        lambda: "Synthetic Windows",
    )
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_local_evidence.platform.machine",
        lambda: "x86_64",
    )
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_local_evidence.platform.python_version",
        lambda: "3.11.9",
    )
    original_run = subprocess.run

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        command = args[0]
        if command[0] == "git":
            return original_run(*args, **kwargs)
        if command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="Synthetic CUDA GPU, 12288, 596.72\n",
                stderr="",
            )
        if "--version" in command:
            return subprocess.CompletedProcess(command, 0, stdout="synthetic 1.0\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout=b"gate pass\n", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    output = repository / ".audit" / "local-candidate" / candidate

    record = capture_local_cuda_release_gates(
        repository_root=repository,
        candidate_commit=candidate,
        output_root=output,
    )

    assert record["status"] == "pass"
    assert record["cuda"]["status"] == "pass"
    assert record["cuda"]["device"] == "Synthetic CUDA GPU"
    assert set(record["gates"]) == {
        "lock",
        "tests",
        "lint",
        "format",
        "types",
        "manifests",
        "splits",
        "configuration",
        "artifacts",
        "git_integrity",
    }
    assert (output / "local_cuda_gate.json").is_file()
    assert (output / "cuda_allocation.raw.log").is_file()
