"""Capture candidate-bound local CUDA gates and staged secret-scan evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
)

SCHEMA_VERSION = "1.0.0"
LOCAL_KIND = "local_cuda_release_gate_evidence"
STAGED_SECRET_KIND = "staged_gitleaks_evidence"
COMMIT_LENGTH = 40
SHA256_LENGTH = 64
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
STAGED_INDEX_RECORD_KEYS = {
    "schema_version",
    "record_kind",
    "status",
    "created_at_utc",
    "head_commit",
    "scan_scope",
    "policy",
    "index_entry_count",
    "staged_change_count",
    "unique_blob_count",
    "replace_ref_count",
    "grafts_file_present",
    "index_entry_index_sha256",
    "violations",
    "raw_worktree_paths_opened",
    "git_objects_written",
    "record_sha256",
}


class LocalReleaseEvidenceError(RuntimeError):
    """Raised when local release evidence cannot be captured safely."""


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise LocalReleaseEvidenceError(f"{name} must be a string-keyed object")
    return value


def _commit(value: str, *, name: str) -> str:
    if len(value) != COMMIT_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise LocalReleaseEvidenceError(f"{name} must be a lowercase Git commit")
    return value


def _sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LocalReleaseEvidenceError(f"{name} must be a lowercase SHA-256")
    return value


def _timestamp(value: str) -> str:
    if TIMESTAMP_RE.fullmatch(value) is None:
        raise LocalReleaseEvidenceError("created_at_utc must be canonical UTC ISO-8601")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise LocalReleaseEvidenceError("created_at_utc is invalid") from exc
    if parsed.tzinfo != UTC:
        raise LocalReleaseEvidenceError("created_at_utc must be UTC")
    return value


def _file_reference(path: Path, *, include_path: str | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LocalReleaseEvidenceError(f"evidence file must be regular and non-symlink: {path}")
    payload = path.read_bytes()
    result: dict[str, Any] = {
        "basename": path.name,
        "size_bytes": len(payload),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
    }
    if include_path is not None:
        result["path"] = include_path
    return result


def _resolve_input(path: str | Path, *, name: str, directory: bool) -> Path:
    raw = Path(path)
    if any(part in {".", ".."} for part in raw.parts):
        raise LocalReleaseEvidenceError(f"{name} path must not contain dot traversal")
    absolute = raw if raw.is_absolute() else Path.cwd() / raw
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise LocalReleaseEvidenceError(f"{name} may not traverse a symlink")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise LocalReleaseEvidenceError(f"{name} does not exist") from exc
    if directory and not resolved.is_dir():
        raise LocalReleaseEvidenceError(f"{name} must be an existing directory")
    if not directory and not resolved.is_file():
        raise LocalReleaseEvidenceError(f"{name} must be a regular file")
    return resolved


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "--no-replace-objects", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "--no-replace-objects", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _tracked_reference(root: Path, candidate: str, path: str) -> dict[str, Any]:
    relative = Path(*path.split("/"))
    source = root / relative
    resolved = _resolve_input(source, name=f"candidate file {path}", directory=False)
    if resolved != source.resolve(strict=True):
        raise LocalReleaseEvidenceError(f"candidate file path differs: {path}")
    payload = resolved.read_bytes()
    if _git_bytes(root, "show", f"{candidate}:{path}") != payload:
        raise LocalReleaseEvidenceError(f"working file differs from candidate blob: {path}")
    return {
        "path": path,
        "size_bytes": len(payload),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _candidate_tree_index(root: Path, candidate: str) -> tuple[int, str]:
    payload = _git_bytes(root, "ls-tree", "-r", "-z", "-l", candidate)
    rows: list[str] = []
    for raw in (item for item in payload.split(b"\x00") if item):
        try:
            metadata, raw_path = raw.split(b"\t", 1)
            mode, kind, object_id, raw_size = metadata.decode("ascii").split(" ", 3)
            path = raw_path.decode("utf-8")
            if "\\" in path or path.startswith("/") or ".." in Path(path).parts:
                raise ValueError("unsafe path")
            size: int | None = None if raw_size == "-" else int(raw_size)
        except (UnicodeError, ValueError) as exc:
            raise LocalReleaseEvidenceError("candidate tree metadata is invalid") from exc
        rows.append(f"{mode} {object_id} 0 {kind} {size} {path}")
    return len(rows), hashlib.sha256(("\n".join(sorted(rows)) + "\n").encode()).hexdigest()


def _repository(path: str | Path, *, candidate_commit: str) -> tuple[Path, str]:
    source = _resolve_input(path, name="repository_root", directory=True)
    root = Path(_git(source, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if source.resolve(strict=True) != root:
        raise LocalReleaseEvidenceError("repository_root must be the Git worktree top-level")
    candidate = _commit(candidate_commit, name="candidate_commit")
    if _git(root, "rev-parse", "HEAD") != candidate:
        raise LocalReleaseEvidenceError("candidate_commit differs from repository HEAD")
    if _git(root, "branch", "--show-current") != "main":
        raise LocalReleaseEvidenceError("candidate must be checked out on local main")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise LocalReleaseEvidenceError("candidate worktree must be clean")
    parent = _commit(_git(root, "rev-parse", f"{candidate}^"), name="candidate parent")
    return root, parent


def _new_candidate_root(root: Path, output_root: str | Path, *, candidate: str) -> Path:
    audit_root = (root / ".audit").resolve(strict=True)
    raw = Path(output_root)
    prospective = raw if raw.is_absolute() else root / raw
    try:
        relative = prospective.relative_to(audit_root)
    except ValueError as exc:
        raise LocalReleaseEvidenceError("local evidence output must remain under .audit") from exc
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise LocalReleaseEvidenceError("local evidence output must be a canonical path")
    current = audit_root
    for part in relative.parent.parts:
        current = current / part
        if current.is_symlink():
            raise LocalReleaseEvidenceError("local evidence output may not traverse a symlink")
        current.mkdir(exist_ok=True)
    resolved_parent = prospective.parent.resolve(strict=True)
    output = resolved_parent / prospective.name
    if output.name != candidate:
        raise LocalReleaseEvidenceError(
            "local evidence directory must be named for candidate_commit"
        )
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite local candidate evidence: {output}")
    output.mkdir()
    return output


def _write_bytes_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _sanitize(payload: bytes, *, repository_root: Path) -> tuple[bytes, list[str]]:
    replacements: list[tuple[bytes, bytes, str]] = []
    root_values = {
        str(repository_root).encode(),
        repository_root.as_posix().encode(),
    }
    for value in root_values:
        replacements.append((value, b"<REPOSITORY_ROOT>", "repository_root"))
    profile = os.environ.get("USERPROFILE")
    if profile:
        profile_path = Path(profile)
        for value in {str(profile_path).encode(), profile_path.as_posix().encode()}:
            replacements.append((value, b"<USER_PROFILE>", "user_profile"))
    sanitized = payload
    applied: list[str] = []
    for needle, replacement, label in sorted(
        replacements, key=lambda item: len(item[0]), reverse=True
    ):
        if needle and needle in sanitized:
            sanitized = sanitized.replace(needle, replacement)
            applied.append(label)
    return sanitized, sorted(set(applied))


def _tool_version(command: Sequence[str], *, root: Path) -> str:
    result = subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    return (result.stdout or result.stderr).strip().splitlines()[0]


def capture_local_cuda_release_gates(
    *, repository_root: str | Path, candidate_commit: str, output_root: str | Path
) -> dict[str, Any]:
    """Run the non-target local release gates and retain raw plus sanitized logs."""

    candidate = _commit(candidate_commit, name="candidate_commit")
    root, parent = _repository(repository_root, candidate_commit=candidate)
    output = _new_candidate_root(root, output_root, candidate=candidate)
    environment = dict(os.environ)
    environment.pop("VIRTUAL_ENV", None)
    gate_commands: list[tuple[str, list[str]]] = [
        ("lock", ["uv", "lock", "--check"]),
        ("tests", [sys.executable, "-m", "pytest", "-q"]),
        ("lint", [sys.executable, "-m", "ruff", "check", "src", "tests"]),
        ("format", [sys.executable, "-m", "ruff", "format", "--check", "src", "tests"]),
        ("types", [sys.executable, "-m", "mypy", "src", "tests"]),
        (
            "manifests",
            [sys.executable, "-m", "inclusive_shift_har.cli", "validate-manifests", "--json"],
        ),
        (
            "splits",
            [
                sys.executable,
                "-m",
                "inclusive_shift_har.cli",
                "audit-splits",
                "--split-manifest",
                "results/protocol/splits/inclusivehar_v4_released_block_v1_2.json",
                "--json",
            ],
        ),
        (
            "configuration",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_few_person_protocol.py",
                "tests/test_protocol_splits.py",
                "tests/test_schema_documents.py",
                "tests/test_uci_source_protocol.py",
            ],
        ),
        (
            "artifacts",
            [
                sys.executable,
                "-m",
                "inclusive_shift_har.cli",
                "validate-artifacts",
                "--artifact-root",
                "results",
                "--require-artifacts",
                "--json",
            ],
        ),
        ("git_integrity", ["git", "fsck", "--full"]),
    ]
    cuda_pass = True
    cuda_started = _utc_now()
    try:
        import torch

        if not torch.cuda.is_available():
            raise LocalReleaseEvidenceError("CUDA is unavailable in the locked environment")
        torch.manual_seed(5062)
        left = torch.randn((1024, 1024), device="cuda")
        right = left @ left
        torch.cuda.synchronize()
        nvidia_smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        cuda = {
            "status": "pass",
            "started_at_utc": cuda_started,
            "completed_at_utc": _utc_now(),
            "torch_version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "nvidia_smi": nvidia_smi,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "allocation_checksum": float(right[0, 0].item()),
        }
        cuda_payload = (json.dumps(cuda, sort_keys=True) + "\n").encode()
        _write_bytes_new(output / "cuda_allocation.raw.log", cuda_payload)
        _write_bytes_new(output / "cuda_allocation.log", cuda_payload)
        del left, right
        torch.cuda.empty_cache()
    except Exception as exc:
        cuda_pass = False
        cuda = {
            "status": "fail",
            "started_at_utc": cuda_started,
            "completed_at_utc": _utc_now(),
            "error": str(exc),
        }
        failure = f"CUDA allocation gate failed: {exc}\n".encode()
        _write_bytes_new(output / "cuda_allocation.raw.log", failure)
        sanitized, _ = _sanitize(failure, repository_root=root)
        _write_bytes_new(output / "cuda_allocation.log", sanitized)

    gates: dict[str, Any] = {}
    overall_pass = cuda_pass
    for name, command in gate_commands:
        started = _utc_now()
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        finished = _utc_now()
        raw_log = output / f"{name}.raw.log"
        sanitized_log = output / f"{name}.log"
        _write_bytes_new(raw_log, completed.stdout)
        sanitized, replacements = _sanitize(completed.stdout, repository_root=root)
        _write_bytes_new(sanitized_log, sanitized)
        gate_status = "pass" if completed.returncode == 0 else "fail"
        overall_pass &= completed.returncode == 0
        recorded_command = ["python" if token == sys.executable else token for token in command]
        gates[name] = {
            "status": gate_status,
            "command": recorded_command,
            "started_at_utc": started,
            "completed_at_utc": finished,
            "exit_code": completed.returncode,
            "raw_log": {**_file_reference(raw_log), "retained_locally_only": True},
            "sanitized_log": {
                **_file_reference(sanitized_log, include_path=f"local/{sanitized_log.name}"),
                "replacements": replacements,
            },
        }
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": LOCAL_KIND,
        "status": "pass" if overall_pass else "fail",
        "created_at_utc": _utc_now(),
        "candidate_commit": candidate,
        "parent_commit": parent,
        "scope": "candidate_bound_local_release_gates_without_target_evaluation",
        "confirmatory_evaluator_invoked": False,
        "cuda_required": True,
        "cuda": cuda,
        "cuda_log": _file_reference(
            output / "cuda_allocation.log", include_path="local/cuda_allocation.log"
        ),
        "machine": {
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
        },
        "tool_versions": {
            "uv": _tool_version(["uv", "--version"], root=root),
            "pytest": _tool_version([sys.executable, "-m", "pytest", "--version"], root=root),
            "ruff": _tool_version([sys.executable, "-m", "ruff", "--version"], root=root),
            "mypy": _tool_version([sys.executable, "-m", "mypy", "--version"], root=root),
        },
        "gates": gates,
    }
    record["record_sha256"] = canonical_json_sha256(record)
    atomic_write_json_new(record, output / "local_cuda_gate.json", allowed_root=output)
    if not overall_pass:
        raise LocalReleaseEvidenceError("one or more local release gates failed")
    return record


def attest_staged_gitleaks(
    *,
    repository_root: str | Path,
    candidate_commit: str,
    policy_path: str | Path,
    staged_index_path: str | Path,
    raw_report_path: str | Path,
    gitleaks_executable: str | Path,
    gitleaks_exit_code: int,
    created_at_utc: str,
    output_path: str | Path,
    allowed_output_root: str | Path,
) -> dict[str, Any]:
    """Bind an empty staged Gitleaks report to the reviewed candidate index."""

    candidate = _commit(candidate_commit, name="candidate_commit")
    root, parent = _repository(repository_root, candidate_commit=candidate)
    policy_source = _resolve_input(policy_path, name="release policy", directory=False)
    policy = _mapping(load_json_strict(policy_source), name="release policy")
    if (
        policy.get("schema_version") != SCHEMA_VERSION
        or policy.get("policy_kind") != "final_release_gate_policy"
    ):
        raise LocalReleaseEvidenceError("release policy schema or kind differs")
    policy_reference = _tracked_reference(
        root, candidate, "configs/release/release_gate_policy_v1.json"
    )
    if policy_source != (root / str(policy_reference["path"])).resolve(strict=True):
        raise LocalReleaseEvidenceError("release policy must be the canonical candidate file")
    gitleaks_policy = _mapping(policy.get("gitleaks"), name="release policy gitleaks")
    config_path = str(gitleaks_policy.get("config_path"))
    ignore_path = str(gitleaks_policy.get("ignore_path"))
    if config_path != ".gitleaks.toml" or ignore_path != ".gitleaksignore":
        raise LocalReleaseEvidenceError("Gitleaks candidate paths differ from the policy")
    config_reference = _tracked_reference(root, candidate, config_path)
    ignore_reference = _tracked_reference(root, candidate, ignore_path)
    if config_reference["file_sha256"] != gitleaks_policy.get("config_sha256"):
        raise LocalReleaseEvidenceError("Gitleaks configuration differs from the policy pin")
    staged_source = _resolve_input(staged_index_path, name="staged index scan", directory=False)
    staged = _mapping(load_json_strict(staged_source), name="staged index scan")
    if set(staged) != STAGED_INDEX_RECORD_KEYS:
        raise LocalReleaseEvidenceError("staged index record keys differ from the exact contract")
    staged_body = dict(staged)
    staged_hash = _sha256(staged_body.pop("record_sha256", None), name="staged record_sha256")
    if canonical_json_sha256(staged_body) != staged_hash:
        raise LocalReleaseEvidenceError("staged index record self-hash does not reconstruct")
    entry_count, tree_digest = _candidate_tree_index(root, candidate)
    if (
        staged.get("record_kind") != "git_index_release_scan"
        or staged.get("schema_version") != SCHEMA_VERSION
        or staged.get("status") != "pass"
        or staged.get("head_commit") != parent
        or staged.get("scan_scope") != "exact_git_index"
        or staged.get("policy") != policy_reference
        or staged.get("index_entry_count") != entry_count
        or not isinstance(staged.get("staged_change_count"), int)
        or isinstance(staged.get("staged_change_count"), bool)
        or staged.get("staged_change_count", 0) < 1
        or not isinstance(staged.get("unique_blob_count"), int)
        or isinstance(staged.get("unique_blob_count"), bool)
        or not 1 <= staged.get("unique_blob_count", 0) <= entry_count
        or staged.get("replace_ref_count") != 0
        or isinstance(staged.get("replace_ref_count"), bool)
        or staged.get("grafts_file_present") is not False
        or staged.get("index_entry_index_sha256") != tree_digest
        or staged.get("violations") != []
        or staged.get("raw_worktree_paths_opened") is not False
        or staged.get("git_objects_written") is not False
    ):
        raise LocalReleaseEvidenceError(
            "staged index record is not the passing candidate parent scan"
        )
    report_source = _resolve_input(raw_report_path, name="raw Gitleaks report", directory=False)
    if report_source.name != "staged-gitleaks.json":
        raise LocalReleaseEvidenceError("raw staged Gitleaks report basename differs")
    report = load_json_strict(report_source)
    if (
        not isinstance(gitleaks_exit_code, int)
        or isinstance(gitleaks_exit_code, bool)
        or gitleaks_exit_code != 0
        or report != []
    ):
        raise LocalReleaseEvidenceError(
            "staged Gitleaks did not complete with an empty finding set"
        )
    executable = _resolve_input(gitleaks_executable, name="Gitleaks executable", directory=False)
    executable_reference = _file_reference(executable)
    if (
        gitleaks_policy.get("version") != "8.30.1"
        or executable_reference["size_bytes"]
        != gitleaks_policy.get("windows_x64_executable_size_bytes")
        or executable_reference["file_sha256"]
        != gitleaks_policy.get("windows_x64_executable_sha256")
    ):
        raise LocalReleaseEvidenceError("Gitleaks executable differs from the release policy pin")
    version = subprocess.run(
        [str(executable), "version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if version != gitleaks_policy.get("version"):
        raise LocalReleaseEvidenceError("Gitleaks executable version differs from policy")
    if staged_source.name != "staged_index_scan.json":
        raise LocalReleaseEvidenceError("staged index evidence basename differs")
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": STAGED_SECRET_KIND,
        "status": "pass",
        "created_at_utc": _timestamp(created_at_utc),
        "candidate_commit": candidate,
        "parent_commit": parent,
        "scan_scope": "exact_precommit_git_index",
        "gitleaks_version": version,
        "gitleaks_exit_code": gitleaks_exit_code,
        "finding_count": 0,
        "gitleaks_executable": executable_reference,
        "raw_report": {
            **_file_reference(report_source),
            "included_in_release_bundle": False,
        },
        "staged_index_scan": {
            **_file_reference(staged_source),
            "record_sha256": staged_hash,
            "index_entry_index_sha256": _sha256(
                staged.get("index_entry_index_sha256"),
                name="staged index_entry_index_sha256",
            ),
        },
        "policy": policy_reference,
        "config": config_reference,
        "ignore": ignore_reference,
    }
    record["record_sha256"] = canonical_json_sha256(record)
    atomic_write_json_new(record, output_path, allowed_root=allowed_output_root)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture-local-gates")
    capture.add_argument("--repository-root", type=Path, required=True)
    capture.add_argument("--candidate-commit", required=True)
    capture.add_argument("--output-root", type=Path, required=True)
    staged = commands.add_parser("attest-staged-gitleaks")
    staged.add_argument("--repository-root", type=Path, required=True)
    staged.add_argument("--candidate-commit", required=True)
    staged.add_argument("--policy", type=Path, required=True)
    staged.add_argument("--staged-index", type=Path, required=True)
    staged.add_argument("--raw-report", type=Path, required=True)
    staged.add_argument("--gitleaks-executable", type=Path, required=True)
    staged.add_argument("--gitleaks-exit-code", type=int, required=True)
    staged.add_argument("--created-at-utc", required=True)
    staged.add_argument("--output", type=Path, required=True)
    staged.add_argument("--allowed-output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "capture-local-gates":
            result = capture_local_cuda_release_gates(
                repository_root=args.repository_root,
                candidate_commit=args.candidate_commit,
                output_root=args.output_root,
            )
        else:
            result = attest_staged_gitleaks(
                repository_root=args.repository_root,
                candidate_commit=args.candidate_commit,
                policy_path=args.policy,
                staged_index_path=args.staged_index,
                raw_report_path=args.raw_report,
                gitleaks_executable=args.gitleaks_executable,
                gitleaks_exit_code=args.gitleaks_exit_code,
                created_at_utc=args.created_at_utc,
                output_path=args.output,
                allowed_output_root=args.allowed_output_root,
            )
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "record_kind": result["record_kind"],
                "record_sha256": result["record_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
