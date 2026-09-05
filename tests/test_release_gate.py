from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tomllib
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from inclusive_shift_har.artifacts.release_gate import (
    EXACT_GATE_NAMES,
    INDEX_SCAN_KIND,
    LICENSE_KIND,
    SCAN_KIND,
    SECRET_KIND,
    ReleaseGateError,
    _blob_content_violations,
    _blob_payload,
    _blob_payloads,
    _commit,
    _safe_relative,
    _sha256,
    _write_new,
    assemble_report,
    audit_licenses,
    scan_index,
    scan_repository,
    validate_ci_gate_bundle,
    validate_license_audit_semantics,
    validate_report,
    validate_secret_scan,
)
from inclusive_shift_har.artifacts.release_inventory import FORBIDDEN_ARTIFACT_SUFFIXES
from inclusive_shift_har.manifests.canonical import canonical_json_sha256


@pytest.mark.parametrize("suffix", [".in", ".xml"])
def test_text_evidence_extensions_do_not_disable_binary_detection(suffix: str) -> None:
    def violations(payload: bytes) -> list[tuple[str, str]]:
        return _blob_content_violations(
            payload,
            path=f"evidence{suffix}",
            signatures={"zip": b"PK\x03\x04"},
            allowed_signatures={},
            allowed_binary_suffixes={".pdf"},
        )

    assert not violations(b"<testsuite tests='1'/>\n")
    assert any(code == "disguised_binary_nul" for code, _ in violations(b"data\x00"))
    assert any(code == "disguised_binary_signature" for code, _ in violations(b"PK\x03\x04"))


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
        "staged_index_scan": {
            "required_before_release_candidate_commit": True,
            "scan_scope": "exact_git_index",
        },
        "git_refs": {
            "non_tag_refs_must_target_commits": True,
            "tags_must_be_annotated_direct_to_commits": True,
            "required_for_release_tags": [],
            "pinned_annotated_tags": {},
            "permitted_candidate_tags": {},
            "historical_notes": {},
        },
        "maximum_blob_size_bytes": 2048,
        "forbidden_path_prefixes": ["data/raw/", ".audit/"],
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
                b'title = "synthetic"\n[extend]\nuseDefault = true\n'
            ).hexdigest(),
            "ignore_path": ".gitleaksignore",
            "reviewed_ignore_entries": [],
            "required_scope": "complete_commit_history_with_policy_pinned_tag_metadata",
        },
        "prohibited_license_tokens": [
            "affero general public license",
            "agpl",
            "commercial",
            "general public license",
            "gpl",
            "proprietary",
            "unknown",
        ],
        "license_exceptions": [],
    }
    _write_json(path, value)
    return path


def _repository(path: Path) -> tuple[Path, str]:
    path.mkdir()
    _git(path, "init", "-b", "main")
    # Byte-bound release evidence must not depend on the caller's global Git
    # setting. In particular, Windows installations commonly enable autocrlf.
    _git(path, "config", "core.autocrlf", "false")
    _git(path, "config", "user.email", "release@example.invalid")
    _git(path, "config", "user.name", "Release Gate Test")
    (path / ".gitignore").write_text(".audit/\n", encoding="utf-8")
    (path / ".gitleaks.toml").write_bytes(b'title = "synthetic"\n[extend]\nuseDefault = true\n')
    (path / ".gitleaksignore").write_bytes(b"# No synthetic suppressions.\n")
    (path / "README.md").write_text("# safe\n", encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "-m", "safe")
    return path, _git(path, "rev-parse", "HEAD")


EXCEPTION_MARKER = (
    "sys_platform == 'linux' or (extra == 'extra-19-inclusive-shift-har-training-cpu' "
    "and extra == 'extra-19-inclusive-shift-har-training-cuda')"
)


def test_batched_git_payloads_match_single_reads_and_reject_mismatched_objects(
    tmp_path: Path,
) -> None:
    repository, commit = _repository(tmp_path / "batch-repository")
    expected = {}
    for name in ("README.md", ".gitignore", ".gitleaks.toml"):
        object_id = _git(repository, "rev-parse", f"{commit}:{name}")
        expected[object_id] = (repository / name).stat().st_size
    payloads = dict(_blob_payloads(repository, expected))
    assert set(payloads) == set(expected)
    for object_id, size in expected.items():
        assert payloads[object_id] == _blob_payload(repository, object_id, expected_size=size)
    object_id = next(iter(expected))
    with pytest.raises(ReleaseGateError, match="identity/type/size"):
        list(_blob_payloads(repository, {object_id: expected[object_id] + 1}))
    with pytest.raises(ReleaseGateError, match="identity/type/size"):
        list(_blob_payloads(repository, {"f" * 40: 1}))


def _exception_repository(
    path: Path,
    *,
    lock_mutator: Callable[[str], str] | None = None,
    review_mutator: Callable[[dict[str, Any]], None] | None = None,
    policy_mutator: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path, str, Path]:
    repository, _initial = _repository(path)
    source_review = (
        Path(__file__).resolve().parents[1] / "docs/release/NVIDIA_NCCL_CU12_2_31_2_REVIEW.json"
    )
    approved_review = json.loads(source_review.read_text(encoding="utf-8"))
    review = deepcopy(approved_review)
    if review_mutator is not None:
        review_mutator(review)
    review.pop("record_sha256", None)
    review["record_sha256"] = canonical_json_sha256(review)
    review_path = repository / "docs/release/NVIDIA_NCCL_CU12_2_31_2_REVIEW.json"
    _write_json(review_path, review)

    wheel = approved_review["distribution"]["wheel"]
    lock_text = f'''version = 1
revision = 3
requires-python = ">=3.11,<3.12"

[[package]]
name = "nvidia-nccl-cu12"
version = "2.31.2"
source = {{ registry = "https://pypi.org/simple" }}
wheels = [
    {{ url = "https://files.pythonhosted.org/packages/37/85/b073e54c993cd9f79faa955d7c9bd7356da408935483aebc3cbb53a922ec/nvidia_nccl_cu12-2.31.2-py3-none-manylinux_2_18_aarch64.whl", hash = "sha256:f208de397e431631eab0eca946444404a495d43a007535baa333d7de9e510ca2", size = 342026203, upload-time = "2026-08-11T23:23:40.509Z" }},
    {{ url = "{wheel["url"]}", hash = "sha256:{wheel["sha256"]}", size = {wheel["size_bytes"]}, upload-time = "2026-08-11T23:24:24.167Z" }},
]

[[package]]
name = "xgboost"
version = "3.2.0"
source = {{ registry = "https://pypi.org/simple" }}
dependencies = [
    {{ name = "nvidia-nccl-cu12", marker = "{EXCEPTION_MARKER}" }},
]
'''
    if lock_mutator is not None:
        lock_text = lock_mutator(lock_text)
    lock_path = repository / "uv.lock"
    lock_path.write_text(lock_text, encoding="utf-8", newline="\n")

    policy_path = _policy(repository / "configs/release/release_gate_policy_v1.json")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["maximum_blob_size_bytes"] = 8192
    metadata = approved_review["distribution"]["core_metadata"]
    embedded = approved_review["distribution"]["embedded_license"]
    sla = approved_review["official_terms"]["nvidia_software_license_agreement"]
    bsd = approved_review["official_terms"]["nvidia_bsd_license"]
    policy["license_exceptions"] = ["nvidia-nccl-cu12@2.31.2"]
    policy["license_exception_records"] = {
        "nvidia-nccl-cu12@2.31.2": {
            "license_expression": "LicenseRef-NVIDIA-Proprietary",
            "required_by": "xgboost@3.2.0",
            "dependency_marker": EXCEPTION_MARKER,
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "scope": ("transitive_linux_xgboost_gpu_runtime_installed_from_pypi_not_redistributed"),
            "lock_path": "uv.lock",
            "lock_sha256": hashlib.sha256(lock_text.encode()).hexdigest(),
            "review_path": "docs/release/NVIDIA_NCCL_CU12_2_31_2_REVIEW.json",
            "review_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
            "review_record_sha256": review["record_sha256"],
            "wheel_filename": wheel["filename"],
            "wheel_url": wheel["url"],
            "wheel_size_bytes": wheel["size_bytes"],
            "wheel_sha256": wheel["sha256"],
            "metadata_url": metadata["url"],
            "metadata_size_bytes": metadata["size_bytes"],
            "metadata_sha256": metadata["sha256"],
            "embedded_license_member": embedded["member"],
            "embedded_license_size_bytes": embedded["size_bytes"],
            "embedded_license_sha256": embedded["sha256"],
            "terms_sla_url": sla["url"],
            "terms_sla_size_bytes": sla["size_bytes"],
            "terms_sla_sha256": sla["sha256"],
            "terms_bsd_url": bsd["url"],
            "terms_bsd_size_bytes": bsd["size_bytes"],
            "terms_bsd_sha256": bsd["sha256"],
        }
    }
    if policy_mutator is not None:
        policy_mutator(policy)
    _write_json(policy_path, policy)
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "candidate exception evidence")
    return repository, _git(repository, "rev-parse", "HEAD"), policy_path


def _exception_inventory(
    path: Path, *, nccl_license: str = "LicenseRef-NVIDIA-Proprietary"
) -> Path:
    _write_json(
        path,
        [
            {"Name": "xgboost", "Version": "3.2.0", "License": "Apache-2.0"},
            {"Name": "nvidia-nccl-cu12", "Version": "2.31.2", "License": nccl_license},
        ],
    )
    return path


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
    assert report["unique_blob_count"] == 4
    assert report["raw_worktree_paths_opened"] is False
    assert report["shallow_repository"] is False
    assert report["replace_ref_count"] == 0
    assert report["grafts_file_present"] is False
    body = dict(report)
    assert body.pop("record_sha256") == canonical_json_sha256(body)


def test_repository_scan_allows_lagging_remote_tracking_ref_before_push(
    tmp_path: Path,
) -> None:
    repository, previous = _repository(tmp_path / "repository")
    policy = _policy(tmp_path / "policy.json")
    (repository / "README.md").write_text("# candidate\n", encoding="utf-8")
    _git(repository, "add", "README.md")
    _git(repository, "commit", "-m", "candidate")
    candidate = _git(repository, "rev-parse", "HEAD")
    _git(repository, "update-ref", "refs/remotes/origin/main", previous)

    report = scan_repository(
        repository_root=repository,
        candidate_commit=candidate,
        policy_path=policy,
        created_at_utc="2026-08-25T12:00:00Z",
    )

    assert report["status"] == "pass"
    assert not {
        "release_main_ref_mismatch",
        "release_main_ref_missing",
    } & {item["code"] for item in report["violations"]}

    _git(repository, "checkout", "--detach", candidate)
    _git(repository, "branch", "-D", "main")
    _git(repository, "update-ref", "refs/remotes/origin/main", candidate)
    remote_only_report = scan_repository(
        repository_root=repository,
        candidate_commit=candidate,
        policy_path=policy,
        created_at_utc="2026-08-25T12:00:01Z",
    )

    assert remote_only_report["status"] == "pass"
    assert remote_only_report["ref_metadata_index_sha256"] == report["ref_metadata_index_sha256"]


def test_repository_scan_prefers_local_main_over_remote_tracking_ref(
    tmp_path: Path,
) -> None:
    repository, candidate = _repository(tmp_path / "repository")
    policy = _policy(tmp_path / "policy.json")
    _git(repository, "update-ref", "refs/remotes/origin/main", candidate)
    (repository / "README.md").write_text("# later local main\n", encoding="utf-8")
    _git(repository, "add", "README.md")
    _git(repository, "commit", "-m", "later local main")
    _git(repository, "checkout", "--detach", candidate)

    report = scan_repository(
        repository_root=repository,
        candidate_commit=candidate,
        policy_path=policy,
        created_at_utc="2026-08-25T12:00:00Z",
    )

    assert report["status"] == "fail"
    mismatches = [
        item for item in report["violations"] if item["code"] == "release_main_ref_mismatch"
    ]
    assert [item["path"] for item in mismatches] == ["refs/heads/main"]


def test_release_gate_rejects_nested_repository_root(tmp_path: Path) -> None:
    repository, commit = _repository(tmp_path / "repository")
    nested = repository / "nested"
    nested.mkdir()
    policy = _policy(tmp_path / "policy.json")

    with pytest.raises(ReleaseGateError, match="Git worktree top-level"):
        scan_repository(
            repository_root=nested,
            candidate_commit=commit,
            policy_path=policy,
            created_at_utc="2026-08-24T12:00:00Z",
        )


@pytest.mark.parametrize("value", ["bad\npath.json", "bad\rpath.json", "bad\x7fpath.json"])
def test_release_paths_reject_control_characters(value: str) -> None:
    with pytest.raises(ReleaseGateError, match="portable POSIX relative path"):
        _safe_relative(value, name="test path")


def test_release_hash_and_commit_contracts_reject_uppercase() -> None:
    with pytest.raises(ReleaseGateError, match="lowercase Git commit"):
        _commit("A" * 40)
    with pytest.raises(ReleaseGateError, match="lowercase SHA-256"):
        _sha256("A" * 64, name="test hash")


def test_release_output_rejects_lexical_symlink_components(tmp_path: Path) -> None:
    root = tmp_path / "root"
    real = root / "real"
    real.mkdir(parents=True)
    alias = root / "alias"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    with pytest.raises(ReleaseGateError, match="traverse a symlink"):
        _write_new({"status": "pass"}, "alias/report.json", root=root)


def test_repository_scan_rejects_objects_reachable_only_from_unsafe_refs(tmp_path: Path) -> None:
    repository, commit = _repository(tmp_path / "repository")
    policy = _policy(tmp_path / "policy.json")
    orphan = repository / "orphan.txt"
    orphan.write_text("uncommitted tag-only payload\n", encoding="utf-8")
    object_id = _git(repository, "hash-object", "-w", "orphan.txt")
    _git(repository, "update-ref", "refs/tags/blob-only", object_id)

    report = scan_repository(
        repository_root=repository,
        candidate_commit=commit,
        policy_path=policy,
        created_at_utc="2026-08-24T12:00:00Z",
    )

    assert report["status"] == "fail"
    assert "lightweight_or_non_tag_ref" in {item["code"] for item in report["violations"]}


def test_repository_scan_rejects_unreviewed_annotated_tag_payload(tmp_path: Path) -> None:
    repository, commit = _repository(tmp_path / "repository")
    policy = _policy(tmp_path / "policy.json")
    _git(repository, "tag", "-a", "unreviewed-v1", "-m", "token in unreviewed tag", commit)

    report = scan_repository(
        repository_root=repository,
        candidate_commit=commit,
        policy_path=policy,
        created_at_utc="2026-08-24T12:00:00Z",
    )

    assert report["status"] == "fail"
    assert "unreviewed_annotated_tag" in {item["code"] for item in report["violations"]}


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


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        (b"a" * 5000 + b"\x00late", "disguised_binary_nul"),
        (b"plain prefix\n" + b"\xff\xfe", "invalid_text_encoding"),
    ],
)
def test_text_blob_validation_covers_entire_payload(
    tmp_path: Path, payload: bytes, expected_code: str
) -> None:
    repository, _commit_value = _repository(tmp_path / "repository")
    policy = _policy(tmp_path / "policy.json")
    policy_value = json.loads(policy.read_text(encoding="utf-8"))
    policy_value["maximum_blob_size_bytes"] = 8192
    _write_json(policy, policy_value)
    (repository / "disguised.md").write_bytes(payload)
    _git(repository, "add", "disguised.md")

    report = scan_index(
        repository_root=repository,
        policy_path=policy,
        created_at_utc="2026-08-24T12:00:00Z",
    )
    assert report["status"] == "fail"
    assert expected_code in {item["code"] for item in report["violations"]}


def test_scans_reject_git_lfs_pointer_and_filter(tmp_path: Path) -> None:
    repository, _commit_value = _repository(tmp_path / "repository")
    policy = _policy(tmp_path / "policy.json")
    (repository / "data.csv").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        "size 123\n",
        encoding="utf-8",
    )
    (repository / ".gitattributes").write_text(
        "*.csv filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8"
    )
    _git(repository, "add", "data.csv", ".gitattributes")

    report = scan_index(
        repository_root=repository,
        policy_path=policy,
        created_at_utc="2026-08-24T12:00:00Z",
    )
    codes = {item["code"] for item in report["violations"]}
    assert {"git_lfs_pointer", "git_lfs_filter"} <= codes


def test_scans_reject_blob_at_exact_protected_root(tmp_path: Path) -> None:
    repository, _commit_value = _repository(tmp_path / "repository")
    policy = _policy(tmp_path / "policy.json")
    protected_root_blob = repository / "data" / "raw"
    protected_root_blob.parent.mkdir()
    protected_root_blob.write_text("must not be released\n", encoding="utf-8")
    _git(repository, "add", "data/raw")

    index_report = scan_index(
        repository_root=repository,
        policy_path=policy,
        created_at_utc="2026-08-24T12:00:00Z",
    )
    assert index_report["status"] == "fail"
    assert "forbidden_path" in {violation["code"] for violation in index_report["violations"]}

    _git(repository, "commit", "-m", "exact protected root blob")
    candidate = _git(repository, "rev-parse", "HEAD")
    history_report = scan_repository(
        repository_root=repository,
        candidate_commit=candidate,
        policy_path=policy,
        created_at_utc="2026-08-24T12:00:00Z",
    )
    assert history_report["status"] == "fail"
    assert "forbidden_path" in {violation["code"] for violation in history_report["violations"]}


def test_scans_reject_credential_basenames_but_allow_env_example(tmp_path: Path) -> None:
    repository, _commit_value = _repository(tmp_path / "repository")
    policy = _policy(tmp_path / "policy.json")
    (repository / ".env.local").write_text("MODE=example\n", encoding="utf-8")
    (repository / "credentials.json").write_text('{"mode": "example"}\n', encoding="utf-8")
    (repository / ".env.example").write_text("MODE=example\n", encoding="utf-8")
    _git(repository, "add", "-f", ".env.local", "credentials.json", ".env.example")

    report = scan_index(
        repository_root=repository,
        policy_path=policy,
        created_at_utc="2026-08-24T12:00:00Z",
    )
    forbidden_paths = {
        violation["path"]
        for violation in report["violations"]
        if violation["code"] == "forbidden_path"
    }
    assert forbidden_paths == {".env.local", "credentials.json"}


def test_scans_inspect_each_path_when_one_blob_has_safe_and_unsafe_suffixes(
    tmp_path: Path,
) -> None:
    repository, _head = _repository(tmp_path / "repository")
    policy = _policy(tmp_path / "policy.json")
    payload = b"%PDF-synthetic"
    (repository / "allowed.pdf").write_bytes(payload)
    (repository / "disguised.md").write_bytes(payload)
    _git(repository, "add", "allowed.pdf", "disguised.md")
    staged = scan_index(
        repository_root=repository,
        policy_path=policy,
        created_at_utc="2026-08-24T12:00:00Z",
    )
    assert staged["status"] == "fail"
    assert any(
        item["code"] == "disguised_binary_signature" and item["path"] == "disguised.md"
        for item in staged["violations"]
    )
    _git(repository, "commit", "-m", "same blob under two suffixes")
    history = scan_repository(
        repository_root=repository,
        candidate_commit=_git(repository, "rev-parse", "HEAD"),
        policy_path=policy,
        created_at_utc="2026-08-24T12:00:00Z",
    )
    assert history["status"] == "fail"
    assert any(
        item["code"] == "disguised_binary_signature" and item["path"] == "disguised.md"
        for item in history["violations"]
    )


def test_repository_scan_rejects_replace_refs_grafts_and_shallow_history(tmp_path: Path) -> None:
    policy = _policy(tmp_path / "policy.json")
    repository, _head = _repository(tmp_path / "repository")
    (repository / "unsafe.md").write_bytes(b"PK\x03\x04hidden")
    _git(repository, "add", "unsafe.md")
    _git(repository, "commit", "-m", "unsafe")
    candidate = _git(repository, "rev-parse", "HEAD")
    unsafe_object = _git(repository, "rev-parse", f"{candidate}:unsafe.md")
    safe_blob = (
        subprocess.run(
            ["git", "hash-object", "-w", "--stdin"],
            cwd=repository,
            input=b"safe replacement\n",
            check=True,
            capture_output=True,
        )
        .stdout.decode("ascii")
        .strip()
    )
    _git(repository, "replace", unsafe_object, safe_blob)
    replaced = scan_repository(
        repository_root=repository,
        candidate_commit=candidate,
        policy_path=policy,
        created_at_utc="2026-08-24T12:00:00Z",
    )
    assert replaced["status"] == "fail"
    assert {item["code"] for item in replaced["violations"]} >= {
        "git_replace_ref",
        "disguised_binary_signature",
    }
    _git(repository, "replace", "-d", unsafe_object)

    grafts = repository / ".git/info/grafts"
    grafts.write_text(candidate + "\n", encoding="utf-8")
    grafted = scan_repository(
        repository_root=repository,
        candidate_commit=candidate,
        policy_path=policy,
        created_at_utc="2026-08-24T12:00:00Z",
    )
    assert grafted["status"] == "fail"
    assert grafted["scan_scope"] == "candidate_tree_and_observed_incomplete_history"
    assert any(item["code"] == "git_grafts_file" for item in grafted["violations"])
    grafts.unlink()

    shallow = tmp_path / "shallow"
    subprocess.run(
        ["git", "clone", "--depth", "1", repository.as_uri(), str(shallow)],
        check=True,
        capture_output=True,
    )
    shallow_candidate = _git(shallow, "rev-parse", "HEAD")
    shallow_report = scan_repository(
        repository_root=shallow,
        candidate_commit=shallow_candidate,
        policy_path=policy,
        created_at_utc="2026-08-24T12:00:00Z",
    )
    assert shallow_report["status"] == "fail"
    assert shallow_report["shallow_repository"] is True
    assert any(item["code"] == "shallow_repository" for item in shallow_report["violations"])


def test_git_index_scan_reads_staged_blobs_not_worktree_and_mutates_no_history(
    tmp_path: Path,
) -> None:
    repository, head = _repository(tmp_path / "repository")
    policy = _policy(tmp_path / "policy.json")
    staged = repository / "staged.md"
    staged.write_text("reviewed staged content\n", encoding="utf-8")
    _git(repository, "add", "staged.md")
    index_before = subprocess.run(
        ["git", "ls-files", "--stage", "-z"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    objects_before = _git(repository, "count-objects", "-v")
    history_before = _git(repository, "rev-list", "--all")
    staged.write_bytes(b"PK\x03\x04unsafe worktree content that is not staged")

    report = scan_index(
        repository_root=repository,
        policy_path=policy,
        created_at_utc="2026-08-24T12:00:00Z",
    )

    assert report["record_kind"] == INDEX_SCAN_KIND
    assert report["status"] == "pass"
    assert report["head_commit"] == head
    assert report["staged_change_count"] == 1
    assert report["raw_worktree_paths_opened"] is False
    assert report["git_objects_written"] is False
    assert _git(repository, "rev-list", "--all") == history_before
    assert _git(repository, "count-objects", "-v") == objects_before
    assert (
        subprocess.run(
            ["git", "ls-files", "--stage", "-z"],
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
        == index_before
    )


def test_git_index_scan_rejects_forbidden_oversized_and_disguised_staged_blobs(
    tmp_path: Path,
) -> None:
    repository, _head = _repository(tmp_path / "repository")
    policy = _policy(tmp_path / "policy.json")
    forbidden = repository / "data" / "raw" / "source.txt"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_text("raw\n", encoding="utf-8")
    oversized = repository / "oversized.md"
    oversized.write_bytes(b"x" * 2049)
    disguised = repository / "disguised.md"
    disguised.write_bytes(b"PK\x03\x04archive")
    _git(repository, "add", "data/raw/source.txt", "oversized.md", "disguised.md")

    report = scan_index(
        repository_root=repository,
        policy_path=policy,
        created_at_utc="2026-08-24T12:00:00Z",
    )

    assert report["status"] == "fail"
    codes = {item["code"] for item in report["violations"]}
    assert {"forbidden_path", "oversized_blob", "disguised_binary_signature"} <= codes


def test_secret_and_license_attestations_are_version_and_candidate_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, commit = _repository(tmp_path / "repository")
    policy = _policy(repository / "configs/release/release_gate_policy_v1.json")
    gitleaks = tmp_path / "gitleaks.json"
    _write_json(gitleaks, [])
    secret = validate_secret_scan(
        repository_root=repository,
        report_path=gitleaks,
        candidate_commit=commit,
        policy_path=policy,
        config_path=repository / ".gitleaks.toml",
        ignore_path=repository / ".gitleaksignore",
        gitleaks_version="8.30.1",
        gitleaks_exit_code=0,
        created_at_utc="2026-08-24T12:00:00Z",
    )
    assert secret["record_kind"] == SECRET_KIND
    assert secret["status"] == "pass"
    assert secret["ignore"]["path"] == ".gitleaksignore"
    operational_error = validate_secret_scan(
        repository_root=repository,
        report_path=gitleaks,
        candidate_commit=commit,
        policy_path=policy,
        config_path=repository / ".gitleaks.toml",
        ignore_path=repository / ".gitleaksignore",
        gitleaks_version="8.30.1",
        gitleaks_exit_code=2,
        created_at_utc="2026-08-24T12:00:00Z",
    )
    assert operational_error["status"] == "fail"
    assert operational_error["execution_status"] == "operational_error"
    (repository / ".gitleaksignore").write_text("broad-suppression\n", encoding="utf-8")
    with pytest.raises(ReleaseGateError, match="reviewed policy"):
        validate_secret_scan(
            repository_root=repository,
            report_path=gitleaks,
            candidate_commit=commit,
            policy_path=policy,
            config_path=repository / ".gitleaks.toml",
            ignore_path=repository / ".gitleaksignore",
            gitleaks_version="8.30.1",
            gitleaks_exit_code=0,
            created_at_utc="2026-08-24T12:00:00Z",
        )
    (repository / ".gitleaksignore").write_bytes(b"# No synthetic suppressions.\n")
    (repository / ".gitleaks.toml").write_bytes(
        b'title = "synthetic"\n[extend]\nuseDefault = true\n[allowlist]\npaths = [".*"]\n'
    )
    with pytest.raises(ReleaseGateError, match="SHA-256 differs from the policy pin"):
        validate_secret_scan(
            repository_root=repository,
            report_path=gitleaks,
            candidate_commit=commit,
            policy_path=policy,
            config_path=repository / ".gitleaks.toml",
            ignore_path=repository / ".gitleaksignore",
            gitleaks_version="8.30.1",
            gitleaks_exit_code=0,
            created_at_utc="2026-08-24T12:00:00Z",
        )
    (repository / ".gitleaks.toml").write_bytes(
        b'title = "synthetic"\n[extend]\nuseDefault = true\n'
    )
    with pytest.raises(ReleaseGateError, match="version differs"):
        validate_secret_scan(
            repository_root=repository,
            report_path=gitleaks,
            candidate_commit=commit,
            policy_path=policy,
            config_path=repository / ".gitleaks.toml",
            ignore_path=repository / ".gitleaksignore",
            gitleaks_version="8.29.0",
            gitleaks_exit_code=0,
            created_at_utc="2026-08-24T12:00:00Z",
        )

    license_repository, license_commit, license_policy = _exception_repository(
        tmp_path / "license-repository"
    )
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.system", lambda: "Windows"
    )
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.machine", lambda: "AMD64"
    )
    licenses = tmp_path / "licenses.json"
    _write_json(licenses, [{"Name": "safe", "Version": "1", "License": "MIT"}])
    audit = audit_licenses(
        repository_root=license_repository,
        inventory_path=licenses,
        candidate_commit=license_commit,
        policy_path=license_policy,
        pip_licenses_version="5.5.5",
        created_at_utc="2026-08-24T12:00:00Z",
    )
    assert audit["record_kind"] == LICENSE_KIND
    assert audit["status"] == "pass"
    assert audit["exceptions_applied"] == []
    _write_json(licenses, [{"Name": "unsafe", "Version": "1", "License": "UNKNOWN"}])
    assert (
        audit_licenses(
            repository_root=license_repository,
            inventory_path=licenses,
            candidate_commit=license_commit,
            policy_path=license_policy,
            pip_licenses_version="5.5.5",
            created_at_utc="2026-08-24T12:00:00Z",
        )["status"]
        == "fail"
    )


def test_nvidia_nccl_exception_is_lock_bound_and_fully_reconstructable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, candidate, policy = _exception_repository(tmp_path / "repository")
    inventory = _exception_inventory(tmp_path / "licenses.json")
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.machine", lambda: "x86_64"
    )

    audit = audit_licenses(
        repository_root=repository,
        inventory_path=inventory,
        candidate_commit=candidate,
        policy_path=policy,
        pip_licenses_version="5.5.5",
        created_at_utc="2026-08-25T12:00:00Z",
    )

    assert audit["status"] == "pass"
    assert audit["package_count"] == 2
    assert [item["identity"] for item in audit["packages"]] == [
        "nvidia-nccl-cu12@2.31.2",
        "xgboost@3.2.0",
    ]
    assert audit["violations"] == []
    assert len(audit["exceptions_applied"]) == 1
    application = audit["exceptions_applied"][0]
    assert application["required_by"] == "xgboost@3.2.0"
    assert application["dependency_marker"] == EXCEPTION_MARKER
    assert application["platform_system"] == "Linux"
    assert application["platform_machine"] == "x86_64"
    assert application["wheel_sha256"] == (
        "f9b1dc3c2a7e20176054144ebb3b32fea83b40402ee5d7ac7045cd11ecc956c0"
    )
    assert application["metadata_sha256"] == (
        "ac64882e612ff2e1c4674386c2a662e0e0b6cae12e85ebe789b1608fa7c3a5fd"
    )
    validate_license_audit_semantics(
        audit,
        root=repository,
        candidate=candidate,
        policy=json.loads(policy.read_text(encoding="utf-8")),
    )

    for applications in ([], [application, application]):
        fabricated = deepcopy(audit)
        fabricated["exceptions_applied"] = applications
        fabricated.pop("record_sha256")
        fabricated["record_sha256"] = canonical_json_sha256(fabricated)
        with pytest.raises(ReleaseGateError):
            validate_license_audit_semantics(
                fabricated,
                root=repository,
                candidate=candidate,
                policy=json.loads(policy.read_text(encoding="utf-8")),
            )

    dropped = deepcopy(audit)
    dropped["packages"] = [
        {"identity": "safe@1", "package": "safe", "version": "1", "license": "MIT"}
    ]
    dropped["package_count"] = 1
    dropped["normalized_inventory_sha256"] = hashlib.sha256(b"safe\t1\tMIT\n").hexdigest()
    dropped["exceptions_applied"] = []
    dropped["violations"] = []
    dropped["status"] = "pass"
    dropped.pop("record_sha256")
    dropped["record_sha256"] = canonical_json_sha256(dropped)
    with pytest.raises(ReleaseGateError, match="both be inventoried"):
        validate_license_audit_semantics(
            dropped,
            root=repository,
            candidate=candidate,
            policy=json.loads(policy.read_text(encoding="utf-8")),
        )


def test_license_inventory_preserves_multiline_license_words(tmp_path: Path) -> None:
    repository, candidate, policy = _exception_repository(tmp_path / "repository")
    inventory = tmp_path / "inventory.json"
    _write_json(
        inventory,
        [{"Name": "example", "Version": "1.0", "License": "BSD-3-Clause\n Copyright holder"}],
    )
    result = audit_licenses(
        repository_root=repository,
        inventory_path=inventory,
        candidate_commit=candidate,
        policy_path=policy,
        pip_licenses_version="5.5.5",
        created_at_utc="2026-09-05T02:15:00Z",
    )
    assert result["status"] == "pass"
    assert result["packages"][0]["license"] == "BSD-3-Clause Copyright holder"
    assert result["inventory"]["file_sha256"] == hashlib.sha256(inventory.read_bytes()).hexdigest()


def test_nvidia_nccl_exception_rejects_duplicate_missing_parent_and_wrong_license(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, candidate, policy = _exception_repository(tmp_path / "repository")
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.machine", lambda: "x86_64"
    )
    duplicate = tmp_path / "duplicate.json"
    package = {
        "Name": "nvidia-nccl-cu12",
        "Version": "2.31.2",
        "License": "LicenseRef-NVIDIA-Proprietary",
    }
    _write_json(
        duplicate,
        [
            {"Name": "xgboost", "Version": "3.2.0", "License": "Apache-2.0"},
            package,
            package,
        ],
    )
    with pytest.raises(ReleaseGateError, match="duplicated"):
        audit_licenses(
            repository_root=repository,
            inventory_path=duplicate,
            candidate_commit=candidate,
            policy_path=policy,
            pip_licenses_version="5.5.5",
            created_at_utc="2026-08-25T12:00:00Z",
        )

    missing_parent = tmp_path / "missing-parent.json"
    _write_json(missing_parent, [package])
    with pytest.raises(ReleaseGateError, match="both be inventoried"):
        audit_licenses(
            repository_root=repository,
            inventory_path=missing_parent,
            candidate_commit=candidate,
            policy_path=policy,
            pip_licenses_version="5.5.5",
            created_at_utc="2026-08-25T12:00:00Z",
        )

    wrong_license = _exception_inventory(
        tmp_path / "wrong-license.json", nccl_license="LicenseRef-Other-Proprietary"
    )
    with pytest.raises(ReleaseGateError, match="licence differs"):
        audit_licenses(
            repository_root=repository,
            inventory_path=wrong_license,
            candidate_commit=candidate,
            policy_path=policy,
            pip_licenses_version="5.5.5",
            created_at_utc="2026-08-25T12:00:00Z",
        )


def test_nvidia_nccl_exception_does_not_apply_on_another_architecture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, candidate, policy = _exception_repository(tmp_path / "repository")
    inventory = _exception_inventory(tmp_path / "licenses.json")
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.machine", lambda: "aarch64"
    )

    audit = audit_licenses(
        repository_root=repository,
        inventory_path=inventory,
        candidate_commit=candidate,
        policy_path=policy,
        pip_licenses_version="5.5.5",
        created_at_utc="2026-08-25T12:00:00Z",
    )

    assert audit["status"] == "fail"
    assert audit["exceptions_applied"] == []
    assert audit["violations"] == [
        {
            "package": "nvidia-nccl-cu12",
            "version": "2.31.2",
            "license": "LicenseRef-NVIDIA-Proprietary",
        }
    ]


@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        ("sys_platform == 'linux'", "sys_platform == 'darwin'"),
        (
            "f9b1dc3c2a7e20176054144ebb3b32fea83b40402ee5d7ac7045cd11ecc956c0",
            "0" * 64,
        ),
        ("size = 342105414", "size = 342105413"),
        ("size = 342105414", "size = 342105414.0"),
        ("size = 342026203", "size = 342026203.0"),
        ("https://files.pythonhosted.org/packages/", "https://example.invalid/packages/"),
        ('version = "3.2.0"', 'version = "3.2.1"'),
        (
            'name = "xgboost"\nversion = "3.2.0"\nsource = { registry = '
            '"https://pypi.org/simple" }',
            'name = "xgboost"\nversion = "3.2.0"\nsource = { registry = '
            '"https://example.invalid/simple" }',
        ),
    ],
)
def test_nvidia_nccl_exception_rejects_mutated_candidate_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    needle: str,
    replacement: str,
) -> None:
    repository, candidate, policy = _exception_repository(
        tmp_path / "repository",
        lock_mutator=lambda text: text.replace(needle, replacement, 1),
    )
    inventory = _exception_inventory(tmp_path / "licenses.json")
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.machine", lambda: "x86_64"
    )
    with pytest.raises(ReleaseGateError):
        audit_licenses(
            repository_root=repository,
            inventory_path=inventory,
            candidate_commit=candidate,
            policy_path=policy,
            pip_licenses_version="5.5.5",
            created_at_utc="2026-08-25T12:00:00Z",
        )


def test_nvidia_nccl_exception_rejects_additional_compatible_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extra_wheel = (
        '    { url = "https://files.pythonhosted.org/packages/00/00/'
        'nvidia_nccl_cu12-2.31.2-py3-none-manylinux_2_18_x86_64.whl", '
        'hash = "sha256:' + "0" * 64 + '", size = 342105415, '
        'upload-time = "2026-08-12T00:00:00Z" },\n'
    )

    def mutate_lock(text: str) -> str:
        boundary = ']\n\n[[package]]\nname = "xgboost"'
        return text.replace(boundary, f"{extra_wheel}{boundary}", 1)

    repository, candidate, policy = _exception_repository(
        tmp_path / "repository", lock_mutator=mutate_lock
    )
    inventory = _exception_inventory(tmp_path / "licenses.json")
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.machine", lambda: "x86_64"
    )
    with pytest.raises(ReleaseGateError, match="exact reviewed package stanza"):
        audit_licenses(
            repository_root=repository,
            inventory_path=inventory,
            candidate_commit=candidate,
            policy_path=policy,
            pip_licenses_version="5.5.5",
            created_at_utc="2026-08-25T12:00:00Z",
        )


@pytest.mark.parametrize(
    "additional_package",
    [
        """
[[package]]
name = "inclusive-shift-har"
version = "0.1.6a0"
source = { editable = "." }
dependencies = [
    { name = "nvidia_nccl.cu12" },
]
""",
        """
[[package]]
name = "other-parent"
version = "1"
source = { registry = "https://pypi.org/simple" }
dependencies = [
    { name = "nvidia-nccl-cu12" },
]
""",
    ],
)
def test_nvidia_nccl_exception_rejects_additional_inbound_edge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    additional_package: str,
) -> None:
    repository, candidate, policy = _exception_repository(
        tmp_path / "repository", lock_mutator=lambda text: text + additional_package
    )
    inventory = tmp_path / "licenses.json"
    _write_json(
        inventory,
        [
            {"Name": "xgboost", "Version": "3.2.0", "License": "Apache-2.0"},
            {
                "Name": "nvidia-nccl-cu12",
                "Version": "2.31.2",
                "License": "LicenseRef-NVIDIA-Proprietary",
            },
            {"Name": "other-parent", "Version": "1", "License": "MIT"},
        ],
    )
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.machine", lambda: "x86_64"
    )
    with pytest.raises(ReleaseGateError, match="reviewed exception edge"):
        audit_licenses(
            repository_root=repository,
            inventory_path=inventory,
            candidate_commit=candidate,
            policy_path=policy,
            pip_licenses_version="5.5.5",
            created_at_utc="2026-08-25T12:00:00Z",
        )


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("distribution", "core_metadata", "sha256"), "1" * 64),
        (("distribution", "embedded_license", "sha256"), "2" * 64),
        (
            ("official_terms", "nvidia_software_license_agreement", "url"),
            "https://example.invalid/nccl/sla.html",
        ),
        (("distribution", "wheel", "filename"), "unreviewed.whl"),
        (("controls", "repository_redistributes_wheel"), 0),
    ],
)
def test_nvidia_nccl_exception_rejects_rehashed_review_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str, ...],
    replacement: Any,
) -> None:
    def mutate_review(review: dict[str, Any]) -> None:
        target: dict[str, Any] = review
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement

    repository, candidate, policy = _exception_repository(
        tmp_path / "repository", review_mutator=mutate_review
    )
    inventory = _exception_inventory(tmp_path / "licenses.json")
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.machine", lambda: "x86_64"
    )
    with pytest.raises(ReleaseGateError, match="provenance differs"):
        audit_licenses(
            repository_root=repository,
            inventory_path=inventory,
            candidate_commit=candidate,
            policy_path=policy,
            pip_licenses_version="5.5.5",
            created_at_utc="2026-08-25T12:00:00Z",
        )


def test_nvidia_nccl_exception_rejects_rehashed_review_identity_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mutate_review(review: dict[str, Any]) -> None:
        review["identity"] = "other-package@9"
        review["package"]["name"] = "other-package"
        review["package"]["version"] = "9"

    repository, candidate, policy = _exception_repository(
        tmp_path / "repository", review_mutator=mutate_review
    )
    inventory = _exception_inventory(tmp_path / "licenses.json")
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.machine", lambda: "x86_64"
    )
    with pytest.raises(ReleaseGateError, match="provenance differs"):
        audit_licenses(
            repository_root=repository,
            inventory_path=inventory,
            candidate_commit=candidate,
            policy_path=policy,
            pip_licenses_version="5.5.5",
            created_at_utc="2026-08-25T12:00:00Z",
        )


def test_nvidia_nccl_exception_rejects_coordinated_identity_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mutate_review(review: dict[str, Any]) -> None:
        review["identity"] = "other-package@9"
        review["package"]["name"] = "other-package"
        review["package"]["version"] = "9"

    def mutate_policy(policy: dict[str, Any]) -> None:
        records = policy["license_exception_records"]
        records["other-package@9"] = records.pop("nvidia-nccl-cu12@2.31.2")
        policy["license_exceptions"] = ["other-package@9"]

    repository, candidate, policy = _exception_repository(
        tmp_path / "repository",
        lock_mutator=lambda text: text.replace("nvidia-nccl-cu12", "other-package"),
        review_mutator=mutate_review,
        policy_mutator=mutate_policy,
    )
    inventory = tmp_path / "licenses.json"
    _write_json(
        inventory,
        [
            {"Name": "xgboost", "Version": "3.2.0", "License": "Apache-2.0"},
            {
                "Name": "other-package",
                "Version": "9",
                "License": "LicenseRef-NVIDIA-Proprietary",
            },
        ],
    )
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.machine", lambda: "x86_64"
    )
    with pytest.raises(ReleaseGateError, match="exact reviewed NCCL exception"):
        audit_licenses(
            repository_root=repository,
            inventory_path=inventory,
            candidate_commit=candidate,
            policy_path=policy,
            pip_licenses_version="5.5.5",
            created_at_utc="2026-08-25T12:00:00Z",
        )


def test_nvidia_nccl_exception_rejects_coordinated_parent_and_marker_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replacement_marker = "sys_platform == 'linux'"

    def mutate_review(review: dict[str, Any]) -> None:
        review["dependency"]["required_by"] = "other-parent@1"
        review["dependency"]["marker"] = replacement_marker

    def mutate_policy(policy: dict[str, Any]) -> None:
        record = policy["license_exception_records"]["nvidia-nccl-cu12@2.31.2"]
        record["required_by"] = "other-parent@1"
        record["dependency_marker"] = replacement_marker

    def mutate_lock(text: str) -> str:
        return (
            text.replace('name = "xgboost"', 'name = "other-parent"')
            .replace('version = "3.2.0"', 'version = "1"')
            .replace(EXCEPTION_MARKER, replacement_marker)
        )

    repository, candidate, policy = _exception_repository(
        tmp_path / "repository",
        lock_mutator=mutate_lock,
        review_mutator=mutate_review,
        policy_mutator=mutate_policy,
    )
    inventory = tmp_path / "licenses.json"
    _write_json(
        inventory,
        [
            {"Name": "other-parent", "Version": "1", "License": "Apache-2.0"},
            {
                "Name": "nvidia-nccl-cu12",
                "Version": "2.31.2",
                "License": "LicenseRef-NVIDIA-Proprietary",
            },
        ],
    )
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.machine", lambda: "x86_64"
    )
    with pytest.raises(ReleaseGateError, match="exact reviewed NCCL exception"):
        audit_licenses(
            repository_root=repository,
            inventory_path=inventory,
            candidate_commit=candidate,
            policy_path=policy,
            pip_licenses_version="5.5.5",
            created_at_utc="2026-08-25T12:00:00Z",
        )


def test_nvidia_nccl_exception_rejects_removal_with_weakened_license_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mutate_policy(policy: dict[str, Any]) -> None:
        policy["license_exceptions"] = []
        policy["license_exception_records"] = {}
        policy["prohibited_license_tokens"].remove("proprietary")

    repository, candidate, policy = _exception_repository(
        tmp_path / "repository", policy_mutator=mutate_policy
    )
    inventory = _exception_inventory(tmp_path / "licenses.json")
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.machine", lambda: "x86_64"
    )
    with pytest.raises(ReleaseGateError, match="exact reviewed NCCL exception"):
        audit_licenses(
            repository_root=repository,
            inventory_path=inventory,
            candidate_commit=candidate,
            policy_path=policy,
            pip_licenses_version="5.5.5",
            created_at_utc="2026-08-25T12:00:00Z",
        )


def test_nvidia_nccl_exception_rejects_weakened_license_tokens_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mutate_policy(policy: dict[str, Any]) -> None:
        policy["prohibited_license_tokens"].remove("proprietary")

    repository, candidate, policy = _exception_repository(
        tmp_path / "repository", policy_mutator=mutate_policy
    )
    inventory = _exception_inventory(tmp_path / "licenses.json")
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.machine", lambda: "x86_64"
    )
    with pytest.raises(ReleaseGateError, match="exact reviewed set"):
        audit_licenses(
            repository_root=repository,
            inventory_path=inventory,
            candidate_commit=candidate,
            policy_path=policy,
            pip_licenses_version="5.5.5",
            created_at_utc="2026-08-25T12:00:00Z",
        )


@pytest.mark.parametrize(
    "policy_mutator",
    [
        lambda value: value["license_exception_records"]["nvidia-nccl-cu12@2.31.2"].pop(
            "review_path"
        ),
        lambda value: value["license_exception_records"]["nvidia-nccl-cu12@2.31.2"].update(
            {"platform_machine": "aarch64"}
        ),
        lambda value: value["license_exception_records"]["nvidia-nccl-cu12@2.31.2"].update(
            {"terms_bsd_url": "https://example.invalid/bsd.html"}
        ),
    ],
)
def test_nvidia_nccl_exception_rejects_broadened_or_incomplete_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy_mutator: Callable[[dict[str, Any]], None],
) -> None:
    repository, candidate, policy = _exception_repository(
        tmp_path / "repository", policy_mutator=policy_mutator
    )
    inventory = _exception_inventory(tmp_path / "licenses.json")
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.machine", lambda: "x86_64"
    )
    with pytest.raises(ReleaseGateError):
        audit_licenses(
            repository_root=repository,
            inventory_path=inventory,
            candidate_commit=candidate,
            policy_path=policy,
            pip_licenses_version="5.5.5",
            created_at_utc="2026-08-25T12:00:00Z",
        )


def test_tracked_report_is_truthfully_pending_and_rejects_self_rehashed_overclaim(
    tmp_path: Path,
) -> None:
    repository, parent = _repository(tmp_path / "repository")
    policy = _policy(repository / "configs/release/release_gate_policy_v1.json")
    report = assemble_report(
        repository_root=repository,
        policy_path=policy,
        mode="tracked_precommit_report",
        created_at_utc="2026-08-24T12:00:00Z",
    )
    assert report["repository"] == {
        "parent_commit": parent,
        "content_commit": None,
        "worktree_clean": False,
    }
    assert report["status"] == "pending"
    path = repository / "report.json"
    _write_json(path, report)
    _git(repository, "add", "configs/release/release_gate_policy_v1.json", "report.json")
    _git(repository, "commit", "-m", "tracked pending release report")
    later = repository / "later.md"
    later.write_text("later candidate work\n", encoding="utf-8")
    _git(repository, "add", "later.md")
    _git(repository, "commit", "-m", "later candidate work")
    assert validate_report(path, repository_root=repository)["valid"] is True
    policy.write_text("{}\n", encoding="utf-8")
    assert validate_report(path, repository_root=repository)["valid"] is True

    not_run_with_evidence = deepcopy(report)
    not_run_with_evidence["gates"]["tests"]["evidence"] = {
        "path": "report.json",
        "size_bytes": 1,
        "file_sha256": "a" * 64,
    }
    not_run_with_evidence.pop("record_sha256")
    not_run_with_evidence["record_sha256"] = canonical_json_sha256(not_run_with_evidence)
    _write_json(path, not_run_with_evidence)
    validation = validate_report(path, repository_root=repository)
    assert validation["valid"] is False
    assert "null evidence" in validation["errors"][0]

    report["status"] = "pass"
    report.pop("record_sha256")
    report["record_sha256"] = canonical_json_sha256(report)
    _write_json(path, report)
    validation = validate_report(path, repository_root=repository)
    assert validation["valid"] is False
    assert "overstates" in validation["errors"][0]


def test_exact_candidate_attestation_revalidates_every_pinned_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "inclusive_shift_har.artifacts.release_gate.platform.machine", lambda: "x86_64"
    )
    repository, parent, policy = _exception_repository(tmp_path / "repository")
    (repository / "README.md").write_text("# release candidate\n", encoding="utf-8")
    _git(repository, "add", "README.md")
    staged = scan_index(
        repository_root=repository,
        policy_path=policy,
        created_at_utc="2026-08-24T11:59:00Z",
    )
    _git(repository, "commit", "-m", "release candidate")
    candidate = _git(repository, "rev-parse", "HEAD")
    evidence_root = repository / ".audit/release-attestations" / candidate
    evidence_root.mkdir(parents=True)
    quality_gate_names = [
        name
        for name in EXACT_GATE_NAMES
        if name not in {"staged_index_scan", "repository_scan", "secret_scan", "license_audit"}
    ]
    executed_gates = [name for name in quality_gate_names if name != "ci_quality_proxy"]
    gate_evidence: dict[str, dict[str, Any]] = {}
    for name in executed_gates:
        log_path = evidence_root / f"{name}.log"
        log_path.write_text(f"{name} passed\n", encoding="utf-8")
        gate_evidence[name] = {
            "exit_code": 0,
            "started_at_utc": "2026-08-24T12:00:00Z",
            "completed_at_utc": "2026-08-24T12:00:01Z",
            "log_path": log_path.relative_to(repository).as_posix(),
            "log_size_bytes": len(log_path.read_bytes()),
            "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
        }
    quality = {
        "schema_version": "1.0.0",
        "record_kind": "ci_release_quality_gate_evidence",
        "status": "pass",
        "candidate_commit": candidate,
        "created_at_utc": "2026-08-24T12:00:02Z",
        "gates": quality_gate_names,
        "gate_statuses": {name: "pass" for name in quality_gate_names},
        "gate_evidence": gate_evidence,
        "tool_versions": {
            "python": "3.11.9",
            "uv": "uv 0.11.29",
            "pytest": "pytest 9.0.2",
            "ruff": "ruff 0.14.14",
            "mypy": "mypy 1.19.1",
        },
        "machine": {"runner_os": "Linux", "architecture": "x86_64", "platform": "Linux"},
        "scope": "upstream synthetic-validation matrix and preceding release-security steps",
        "ci_evidence_scope": "within_job_quality_proxy_not_completed_workflow",
        "external_completed_ci_required_for_release_inventory": True,
    }
    quality["record_sha256"] = canonical_json_sha256(quality)
    quality_path = evidence_root / "quality_gates.json"
    _write_json(quality_path, quality)
    staged_path = evidence_root / "staged_index_scan.json"
    _write_json(staged_path, staged)
    repository_path = evidence_root / "repository_scan.json"
    _write_json(
        repository_path,
        scan_repository(
            repository_root=repository,
            candidate_commit=candidate,
            policy_path=policy,
            created_at_utc="2026-08-24T12:00:00Z",
        ),
    )
    raw_secret = evidence_root / "gitleaks.json"
    _write_json(raw_secret, [])
    secret_path = evidence_root / "secret_scan.json"
    _write_json(
        secret_path,
        validate_secret_scan(
            repository_root=repository,
            report_path=raw_secret,
            candidate_commit=candidate,
            policy_path=policy,
            config_path=repository / ".gitleaks.toml",
            ignore_path=repository / ".gitleaksignore",
            gitleaks_version="8.30.1",
            gitleaks_exit_code=0,
            created_at_utc="2026-08-24T12:00:00Z",
        ),
    )
    raw_licenses = evidence_root / "licenses.json"
    _exception_inventory(raw_licenses)
    license_path = evidence_root / "license_audit.json"
    _write_json(
        license_path,
        audit_licenses(
            repository_root=repository,
            inventory_path=raw_licenses,
            candidate_commit=candidate,
            policy_path=policy,
            pip_licenses_version="5.5.5",
            created_at_utc="2026-08-24T12:00:00Z",
        ),
    )
    bundle = validate_ci_gate_bundle(
        repository_root=repository,
        candidate_commit=candidate,
        policy_path=policy,
        evidence_root=evidence_root,
        created_at_utc="2026-08-24T12:00:03Z",
    )
    assert bundle["status"] == "pass"
    (repository / "README.md").write_text("# quality gate mutated the checkout\n", encoding="utf-8")
    with pytest.raises(ReleaseGateError, match="worktree is not clean"):
        validate_ci_gate_bundle(
            repository_root=repository,
            candidate_commit=candidate,
            policy_path=policy,
            evidence_root=evidence_root,
            created_at_utc="2026-08-24T12:00:03Z",
        )
    (repository / "README.md").write_text("# release candidate\n", encoding="utf-8")

    evidence_paths = {
        "staged_index_scan": staged_path,
        "repository_scan": repository_path,
        "secret_scan": secret_path,
        "license_audit": license_path,
        **{name: quality_path for name in quality_gate_names},
    }
    references = {
        name: {
            "status": "pass",
            "path": path.relative_to(repository).as_posix(),
            "expected_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in evidence_paths.items()
    }
    spec: dict[str, Any] = {
        "schema_version": "1.0.0",
        "spec_kind": "final_release_gate_attestation_spec",
        "created_at_utc": "2026-08-24T12:00:00Z",
        "candidate_commit": candidate,
        "parent_commit": parent,
        "worktree_clean": True,
        "gates": references,
    }
    spec_path = evidence_root / "spec.json"
    _write_json(spec_path, spec)

    bad_parent_spec = dict(spec)
    bad_parent_spec["parent_commit"] = "f" * 40
    bad_parent_path = evidence_root / "bad-parent-spec.json"
    _write_json(bad_parent_path, bad_parent_spec)
    with pytest.raises(ReleaseGateError, match="candidate first parent"):
        assemble_report(
            repository_root=repository,
            policy_path=policy,
            mode="exact_candidate_attestation",
            created_at_utc="2026-08-24T12:00:00Z",
            spec_path=bad_parent_path,
        )

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
    assert validate_report(report_path)["valid"] is False
    assert "requires repository_root" in validate_report(report_path)["errors"][0]

    original_repository_evidence = repository_path.read_bytes()
    malformed_repository_evidence = json.loads(original_repository_evidence)
    malformed_repository_evidence["unexpected"] = True
    malformed_repository_evidence.pop("record_sha256")
    malformed_repository_evidence["record_sha256"] = canonical_json_sha256(
        malformed_repository_evidence
    )
    _write_json(repository_path, malformed_repository_evidence)
    malformed_evidence_spec = deepcopy(spec)
    malformed_evidence_spec["gates"]["repository_scan"]["expected_sha256"] = hashlib.sha256(
        repository_path.read_bytes()
    ).hexdigest()
    malformed_evidence_spec_path = evidence_root / "malformed-evidence-spec.json"
    _write_json(malformed_evidence_spec_path, malformed_evidence_spec)
    with pytest.raises(ReleaseGateError, match="keys differ"):
        assemble_report(
            repository_root=repository,
            policy_path=policy,
            mode="exact_candidate_attestation",
            created_at_utc="2026-08-24T12:00:00Z",
            spec_path=malformed_evidence_spec_path,
        )
    repository_path.write_bytes(original_repository_evidence)

    for gate_name, evidence_path, field, value in (
        ("staged_index_scan", staged_path, "replace_ref_count", False),
        ("repository_scan", repository_path, "replace_ref_count", 0.0),
        ("secret_scan", secret_path, "gitleaks_exit_code", 0.0),
    ):
        original_evidence = evidence_path.read_bytes()
        malformed_numeric = json.loads(original_evidence)
        malformed_numeric[field] = value
        malformed_numeric.pop("record_sha256")
        malformed_numeric["record_sha256"] = canonical_json_sha256(malformed_numeric)
        _write_json(evidence_path, malformed_numeric)
        numeric_spec = deepcopy(spec)
        numeric_spec["gates"][gate_name]["expected_sha256"] = hashlib.sha256(
            evidence_path.read_bytes()
        ).hexdigest()
        numeric_spec_path = evidence_root / f"bad-numeric-{gate_name}.json"
        _write_json(numeric_spec_path, numeric_spec)
        with pytest.raises(ReleaseGateError, match="exact JSON integer"):
            assemble_report(
                repository_root=repository,
                policy_path=policy,
                mode="exact_candidate_attestation",
                created_at_utc="2026-08-24T12:00:00Z",
                spec_path=numeric_spec_path,
            )
        evidence_path.write_bytes(original_evidence)

    mutations: tuple[tuple[str, Callable[[dict[str, Any]], None], str], ...] = (
        (
            "extra-policy",
            lambda value: value["policy"].update({"unexpected": True}),
            "policy keys differ",
        ),
        (
            "extra-gate",
            lambda value: value["gates"]["tests"].update({"unexpected": True}),
            "gates.tests keys differ",
        ),
        (
            "extra-external",
            lambda value: value["external_candidate_attestation"].update({"unexpected": True}),
            "external_candidate_attestation keys differ",
        ),
    )
    for label, mutate, expected_error in mutations:
        malformed = deepcopy(report)
        mutate(malformed)
        malformed.pop("record_sha256")
        malformed["record_sha256"] = canonical_json_sha256(malformed)
        malformed_path = evidence_root / f"{label}.json"
        _write_json(malformed_path, malformed)
        malformed_validation = validate_report(malformed_path, repository_root=repository)
        assert malformed_validation["valid"] is False
        assert expected_error in malformed_validation["errors"][0]

    wrong_parent_report = dict(report)
    wrong_parent_report["repository"] = {
        **report["repository"],
        "parent_commit": "f" * 40,
    }
    wrong_parent_report.pop("record_sha256")
    wrong_parent_report["record_sha256"] = canonical_json_sha256(wrong_parent_report)
    wrong_parent_report_path = evidence_root / "wrong-parent-report.json"
    _write_json(wrong_parent_report_path, wrong_parent_report)
    wrong_parent_validation = validate_report(wrong_parent_report_path, repository_root=repository)
    assert wrong_parent_validation["valid"] is False
    assert "candidate first parent" in wrong_parent_validation["errors"][0]

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
    assert policy["gitleaks"]["config_path"] == ".gitleaks.toml"
    assert policy["gitleaks"]["ignore_path"] == ".gitleaksignore"
    assert len(policy["gitleaks"]["reviewed_ignore_entries"]) == 3
    assert policy["gitleaks"]["linux_x64_archive_sha256"] == (
        "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb"
    )
    assert {".onnx", ".safetensors", ".parquet", ".hdf5", ".7z"} <= set(
        policy["forbidden_suffixes"]
    )
    assert set(policy["forbidden_suffixes"]) == set(FORBIDDEN_ARTIFACT_SUFFIXES)
    assert policy["maximum_blob_size_bytes"] == 16 * 1024 * 1024
    assert policy["staged_index_scan"] == {
        "required_before_release_candidate_commit": True,
        "scan_scope": "exact_git_index",
    }
    assert policy["ci_action_pins"] == {
        "actions/checkout": "3d3c42e5aac5ba805825da76410c181273ba90b1",
        "actions/setup-python": "a309ff8b426b58ec0e2a45f0f869d46889d02405",
        "actions/upload-artifact": "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
        "astral-sh/setup-uv": "20cfd1bf945f4377ade1205e4dbc17946fc9a30d",
    }


def test_gitleaks_allowlist_is_rule_path_and_exact_field_scoped(repository_root: Path) -> None:
    config_path = repository_root / ".gitleaks.toml"
    config_payload = config_path.read_bytes()
    policy = json.loads(
        (repository_root / "configs/release/release_gate_policy_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert policy["gitleaks"]["config_sha256"] == hashlib.sha256(config_payload).hexdigest()
    config = tomllib.loads(config_payload.decode("utf-8"))
    assert set(config) == {"title", "extend", "rules"}
    assert config["extend"] == {"useDefault": True}
    rules = config.get("rules")
    assert isinstance(rules, list)
    assert len(rules) == 1
    rule = rules[0]
    assert isinstance(rule, dict)
    assert set(rule) == {"id", "allowlists"}
    assert rule["id"] == "generic-api-key"
    allowlists = rule["allowlists"]
    assert isinstance(allowlists, list)
    assert len(allowlists) == 2
    allowlist = allowlists[0]
    assert isinstance(allowlist, dict)
    assert set(allowlist) == {
        "description",
        "condition",
        "regexTarget",
        "paths",
        "regexes",
    }
    assert allowlist["condition"] == "AND"
    assert allowlist["regexTarget"] == "match"
    assert allowlist["paths"] == [r"(?:^|[/\\])ci\.json$"]
    assert allowlist["regexes"] == [
        r'^secret_scan":"[0-9a-f]{64}"$',
        r'^secret_scan\.json":"[0-9a-f]{64}"$',
    ]

    source_hash_rule = allowlists[1]
    assert source_hash_rule["condition"] == "AND"
    assert source_hash_rule["regexTarget"] == "match"
    public_digest = hashlib.sha256(
        (repository_root / "docs/research/CONTROLLED_DATA_ACCESS_CHECKLIST.md").read_bytes()
    ).hexdigest()
    source_pattern = re.compile(source_hash_rule["regexes"][0])
    source_path = re.compile(source_hash_rule["paths"][0])
    assert source_path.search("results/research/cross_dataset_har_v3/run/result.json")
    assert not source_path.search("credentials.json")
    assert source_pattern.fullmatch(f'CONTROLLED_DATA_ACCESS_CHECKLIST.md": "{public_digest}"')
    assert not source_pattern.fullmatch(f'CONTROLLED_DATA_ACCESS_CHECKLIST.md": "{"a" * 64}"')
    assert not source_pattern.fullmatch(f'api_key": "{public_digest}"')

    path_pattern = re.compile(allowlist["paths"][0])
    assert path_pattern.search("ci.json")
    assert path_pattern.search("C:/validated-release/ci.json")
    assert not path_pattern.search("repository-api.json")
    assert not path_pattern.search("ci.json.backup")

    digest = hashlib.sha256(b"synthetic release evidence").hexdigest()
    content_patterns = [re.compile(value) for value in allowlist["regexes"]]
    allowed_matches = {
        f'secret_scan":"{digest}"',
        f'secret_scan.json":"{digest}"',
    }
    assert all(
        any(pattern.fullmatch(value) for pattern in content_patterns) for value in allowed_matches
    )
    for rejected in (
        f'other_secret":"{digest}"',
        f'temp_clone_token":"{digest}"',
        f'secret_scan":"{digest.upper()}"',
        'secret_scan":"not-a-sha256"',
    ):
        assert not any(pattern.fullmatch(rejected) for pattern in content_patterns)


def test_global_ignore_policy_covers_release_forbidden_payloads(repository_root: Path) -> None:
    lines = set((repository_root / ".gitignore").read_text(encoding="utf-8").splitlines())
    assert {"*.zip", "*.npy", "*.npz", "*.docx", "*.onnx"} <= lines
    assert hashlib.sha256(
        (repository_root / "configs/release/release_gate_policy_v1.json").read_bytes()
    ).hexdigest()


def test_ci_uses_immutable_actions_and_preserves_failed_security_evidence(
    repository_root: Path,
) -> None:
    workflow_path = repository_root / ".github/workflows/ci.yml"
    text = workflow_path.read_text(encoding="utf-8")
    for expected in (
        "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
        "actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405",
        "astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d",
        "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
        "fetch-depth: 0",
        "persist-credentials: false",
        "if: always()",
        "validate-ci-bundle",
        "ci_quality_proxy",
        "within_job_quality_proxy_not_completed_workflow",
        "branches:",
        "- main",
        "ci_bundle_validation.json",
        "include-hidden-files: true",
        'version: "0.11.29"',
        'GIT_NO_REPLACE_OBJECTS: "1"',
        "--gitleaks-exit-code",
        "--config .gitleaks.toml",
        "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb",
    ):
        assert expected in text
    assert text.count('version: "0.11.29"') == 2

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(
        loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
    ) -> dict[Any, Any]:
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise AssertionError(f"duplicate YAML key: {key}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    workflow = yaml.load(text, Loader=UniqueKeyLoader)

    uses: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "uses":
                    assert isinstance(child, str)
                    uses.append(child)
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(workflow)
    pins = json.loads(
        (repository_root / "configs/release/release_gate_policy_v1.json").read_text(
            encoding="utf-8"
        )
    )["ci_action_pins"]
    assert uses
    for use in uses:
        if use.startswith("./"):
            continue
        match = re.fullmatch(r"([^@]+)@([0-9a-f]{40})", use)
        assert match is not None, f"action is not pinned to a full commit: {use}"
        assert pins.get(match.group(1)) == match.group(2), f"action is not policy-approved: {use}"
