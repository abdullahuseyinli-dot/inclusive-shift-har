"""Git-object release security scans and exact-candidate attestations.

The scanners in this module never walk raw-data directories. Repository content
is read through Git object IDs, and only blobs whose paths already pass the
forbidden-path/extension policy are inspected for disguised binary payloads.
Every generated record is create-only and canonically self-hashed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
)

SCHEMA_VERSION = "1.0.0"
REPORT_KIND = "final_release_gate_report"
SCAN_KIND = "git_object_release_scan"
INDEX_SCAN_KIND = "git_index_release_scan"
SECRET_KIND = "gitleaks_secret_scan_attestation"
LICENSE_KIND = "python_license_audit"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
TAG_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
MAX_TAG_OBJECT_BYTES = 64 * 1024
TRACKED_GATE_NAMES = (
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
EXACT_GATE_NAMES = ("staged_index_scan", *TRACKED_GATE_NAMES[:-1], "ci_quality_proxy")
# Backwards-compatible public alias for the tracked precommit record contract.
GATE_NAMES = TRACKED_GATE_NAMES
TEXT_SUFFIXES = {
    "",
    ".bib",
    ".cff",
    ".csv",
    ".gitignore",
    ".gitleaksignore",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".typed",
    ".yaml",
    ".yml",
}
QUALITY_GATE_NAMES = frozenset(
    {
        "tests",
        "lint",
        "format",
        "types",
        "manifests",
        "splits",
        "configuration",
        "artifacts",
        "ci_quality_proxy",
    }
)
QUALITY_EVIDENCE_KIND = "ci_release_quality_gate_evidence"
QUALITY_EVIDENCE_SCOPE = "within_job_quality_proxy_not_completed_workflow"
SECRET_SCAN_SCOPE = "complete_commit_history_with_policy_pinned_tag_metadata"


class ReleaseGateError(RuntimeError):
    """Raised when release evidence is unsafe, incomplete, or inconsistent."""


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ReleaseGateError(f"{name} must be a string-keyed object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ReleaseGateError(
            f"{name} keys differ; missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _array(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReleaseGateError(f"{name} must be an array")
    return value


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseGateError(f"{name} must be a non-empty string")
    return value


def _commit(value: Any, *, name: str = "commit") -> str:
    result = _text(value, name=name)
    if COMMIT_RE.fullmatch(result) is None:
        raise ReleaseGateError(f"{name} must be a full lowercase Git commit")
    return result


def _sha256(value: Any, *, name: str) -> str:
    result = _text(value, name=name)
    if SHA256_RE.fullmatch(result) is None:
        raise ReleaseGateError(f"{name} must be a lowercase SHA-256")
    return result


def _exact_integer(
    value: Any,
    *,
    name: str,
    expected: int | None = None,
    minimum: int | None = None,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ReleaseGateError(f"{name} must be an exact JSON integer")
    if expected is not None and value != expected:
        raise ReleaseGateError(f"{name} must equal {expected}")
    if minimum is not None and value < minimum:
        raise ReleaseGateError(f"{name} must be at least {minimum}")
    return value


def _timestamp(value: Any) -> str:
    result = _text(value, name="created_at_utc")
    if TIMESTAMP_RE.fullmatch(result) is None:
        raise ReleaseGateError("created_at_utc must use canonical ISO-8601 UTC Z form")
    try:
        parsed = datetime.fromisoformat(result[:-1] + "+00:00")
    except ValueError as exc:
        raise ReleaseGateError("created_at_utc must be valid ISO-8601") from exc
    if parsed.tzinfo != UTC:
        raise ReleaseGateError("created_at_utc must be UTC")
    return result


def _safe_relative(value: Any, *, name: str) -> str:
    result = _text(value, name=name)
    path = PurePosixPath(result)
    if (
        "\\" in result
        or ":" in result
        or any(ord(character) < 32 or ord(character) == 127 for character in result)
        or path.is_absolute()
        or path.as_posix() != result
        or any(part in {".", ".."} for part in path.parts)
    ):
        raise ReleaseGateError(f"{name} must be a portable POSIX relative path")
    return result


def _git(
    root: Path, arguments: Sequence[str], *, binary: bool = False, check: bool = True
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", "--no-replace-objects", *arguments],
            cwd=root,
            check=check,
            capture_output=True,
            text=not binary,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReleaseGateError(f"Git command failed: {' '.join(arguments)}") from exc


def _repository(path: str | Path) -> Path:
    source = Path(path)
    if source.is_symlink():
        raise ReleaseGateError("repository_root may not be a symlink")
    root = source.resolve(strict=True)
    if not root.is_dir():
        raise ReleaseGateError("repository_root must be a directory")
    inside = _git(root, ("rev-parse", "--is-inside-work-tree"))
    assert isinstance(inside.stdout, str)
    if inside.stdout.strip() != "true":
        raise ReleaseGateError("repository_root is not a Git worktree")
    top_level = _git(root, ("rev-parse", "--show-toplevel"))
    assert isinstance(top_level.stdout, str)
    try:
        actual_root = Path(top_level.stdout.strip()).resolve(strict=True)
    except OSError as exc:
        raise ReleaseGateError("Git worktree root cannot be resolved") from exc
    if not root.samefile(actual_root):
        raise ReleaseGateError("repository_root must equal the Git worktree top-level")
    return root


def _require_candidate_head(root: Path, candidate: str, *, operation: str) -> None:
    head_result = _git(root, ("rev-parse", "HEAD"))
    assert isinstance(head_result.stdout, str)
    head = _commit(head_result.stdout.strip(), name="repository HEAD")
    if candidate != head:
        raise ReleaseGateError(f"{operation} candidate_commit differs from repository HEAD")


def _strict_json(path: str | Path, *, name: str) -> Mapping[str, Any]:
    try:
        return _mapping(load_json_strict(path), name=name)
    except (OSError, ValueError) as exc:
        raise ReleaseGateError(f"cannot load {name}: {exc}") from exc


def _strict_json_bytes(payload: bytes, *, name: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReleaseGateError(f"duplicate JSON key in {name}: {key!r}")
            result[key] = value
        return result

    def reject_non_finite(token: str) -> None:
        raise ReleaseGateError(f"non-finite JSON number in {name}: {token}")

    try:
        value = json.loads(
            payload.decode("utf-8-sig"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_non_finite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseGateError(f"cannot load {name}: {exc}") from exc
    return _mapping(value, name=name)


def _policy(path: str | Path) -> tuple[Mapping[str, Any], dict[str, Any]]:
    source = Path(path)
    if source.is_symlink():
        raise ReleaseGateError("policy may not be a symlink")
    payload = source.resolve(strict=True).read_bytes()
    value = _strict_json_bytes(payload, name="release gate policy")
    if value.get("schema_version") != "1.0.0" or value.get("policy_kind") != (
        "final_release_gate_policy"
    ):
        raise ReleaseGateError("release gate policy version or kind differs")
    reference = {
        "path": "configs/release/release_gate_policy_v1.json",
        "size_bytes": len(payload),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
    }
    return value, reference


def _write_new(record: dict[str, Any], destination: str | Path, *, root: Path) -> None:
    raw = Path(destination)
    if any(part == ".." for part in raw.parts):
        raise ReleaseGateError("destination may not contain parent traversal")
    output = raw if raw.is_absolute() else root / raw
    try:
        relative = output.relative_to(root)
    except ValueError as exc:
        raise ReleaseGateError("destination escapes repository_root") from exc
    for component_count in range(1, len(relative.parts) + 1):
        component = root.joinpath(*relative.parts[:component_count])
        if component.is_symlink():
            raise ReleaseGateError("destination may not traverse a symlink")
    prospective = output.resolve(strict=False)
    try:
        prospective.relative_to(root)
    except ValueError as exc:
        raise ReleaseGateError("destination escapes repository_root") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    for component_count in range(1, len(relative.parts) + 1):
        component = root.joinpath(*relative.parts[:component_count])
        if component.is_symlink():
            raise ReleaseGateError("destination may not traverse a symlink")
    atomic_write_json_new(record, output, allowed_root=root)


def _self_hash(body: dict[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["record_sha256"] = canonical_json_sha256(result)
    return result


def _verify_self_hash(value: Mapping[str, Any], *, name: str) -> str:
    body = dict(value)
    observed = _sha256(body.pop("record_sha256", None), name=f"{name}.record_sha256")
    if canonical_json_sha256(body) != observed:
        raise ReleaseGateError(f"{name} self-hash does not reconstruct")
    return observed


def _tree_entries(root: Path, commit: str) -> list[tuple[str, str, str, int | None, str]]:
    result = _git(root, ("ls-tree", "-r", "-z", "-l", commit), binary=True)
    assert isinstance(result.stdout, bytes)
    entries: list[tuple[str, str, str, int | None, str]] = []
    for raw in (item for item in result.stdout.split(b"\x00") if item):
        try:
            metadata, raw_path = raw.split(b"\t", 1)
            mode, kind, object_id, size_text = metadata.decode("ascii").split(" ", 3)
            path = _safe_relative(raw_path.decode("utf-8"), name="Git tree path")
        except (UnicodeError, ValueError) as exc:
            raise ReleaseGateError("Git tree contains undecodable metadata or path") from exc
        size = None if size_text == "-" else int(size_text)
        entries.append((mode, kind, object_id, size, path))
    return entries


def _entry_index_row(
    *, mode: str, object_id: str, stage: int, kind: str, size: int | None, path: str
) -> str:
    return f"{mode} {object_id} {stage} {kind} {size} {path}"


def _candidate_tree_index(root: Path, commit: str) -> tuple[int, str]:
    rows = [
        _entry_index_row(
            mode=mode,
            object_id=object_id,
            stage=0,
            kind=kind,
            size=size,
            path=path,
        )
        for mode, kind, object_id, size, path in _tree_entries(root, commit)
    ]
    return len(rows), hashlib.sha256(("\n".join(sorted(rows)) + "\n").encode()).hexdigest()


def _content_scan_policy(
    policy: Mapping[str, Any],
) -> tuple[
    int,
    tuple[str, ...],
    set[str],
    dict[str, bytes],
    dict[str, set[str]],
    set[str],
]:
    maximum = policy.get("maximum_blob_size_bytes")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
        raise ReleaseGateError("maximum_blob_size_bytes must be a positive integer")
    prefixes = tuple(
        _safe_relative(str(value).rstrip("/"), name="forbidden_path_prefix")
        for value in _array(policy.get("forbidden_path_prefixes"), name="forbidden prefixes")
    )
    suffixes = {
        _text(value, name="forbidden_suffix").casefold()
        for value in _array(policy.get("forbidden_suffixes"), name="forbidden suffixes")
    }
    signatures = {
        _text(name, name="signature name"): bytes.fromhex(_text(signature, name="signature"))
        for name, signature in _mapping(
            policy.get("disguised_binary_signatures"), name="binary signatures"
        ).items()
    }
    allowed_signatures = {
        str(name): {str(item).casefold() for item in _array(values, name="allowed suffixes")}
        for name, values in _mapping(
            policy.get("allowed_signature_suffixes"), name="allowed signatures"
        ).items()
    }
    allowed_binary_suffixes = {
        _text(value, name="allowed_binary_suffix").casefold()
        for value in _array(policy.get("allowed_binary_suffixes"), name="allowed binary suffixes")
    }
    if any(
        suffix
        not in {
            allowed_suffix
            for suffixes_for_signature in allowed_signatures.values()
            for allowed_suffix in suffixes_for_signature
        }
        for suffix in allowed_binary_suffixes
    ):
        raise ReleaseGateError("every allowed binary suffix must have an allowed magic signature")
    return maximum, prefixes, suffixes, signatures, allowed_signatures, allowed_binary_suffixes


def _is_forbidden_path(path: str, prefixes: Sequence[str]) -> bool:
    folded = path.casefold()
    return any(
        folded == prefix.casefold() or folded.startswith(f"{prefix.casefold()}/")
        for prefix in prefixes
    )


def _credential_path_policy(
    policy: Mapping[str, Any],
) -> tuple[set[str], tuple[str, ...], set[str]]:
    basenames = {
        _text(value, name="forbidden_basename").casefold()
        for value in _array(policy.get("forbidden_basenames"), name="forbidden basenames")
    }
    basename_prefixes = tuple(
        _text(value, name="forbidden_basename_prefix").casefold()
        for value in _array(
            policy.get("forbidden_basename_prefixes"), name="forbidden basename prefixes"
        )
    )
    allowed_paths = {
        _safe_relative(value, name="allowed_sensitive_path").casefold()
        for value in _array(policy.get("allowed_sensitive_paths"), name="allowed sensitive paths")
    }
    return basenames, basename_prefixes, allowed_paths


def _is_forbidden_credential_path(
    path: str,
    *,
    basenames: set[str],
    basename_prefixes: Sequence[str],
    allowed_paths: set[str],
) -> bool:
    folded = path.casefold()
    if folded in allowed_paths:
        return False
    basename = PurePosixPath(path).name.casefold()
    return basename in basenames or any(basename.startswith(prefix) for prefix in basename_prefixes)


def _tag_policy(
    policy: Mapping[str, Any],
) -> tuple[
    set[str],
    dict[str, tuple[str, str, str, str, str]],
    dict[str, tuple[str, str, str]],
]:
    value = _mapping(policy.get("git_refs"), name="git_refs policy")
    _exact_keys(
        value,
        {
            "non_tag_refs_must_target_commits",
            "tags_must_be_annotated_direct_to_commits",
            "required_for_release_tags",
            "pinned_annotated_tags",
            "permitted_candidate_tags",
            "historical_notes",
        },
        name="git_refs policy",
    )
    if (
        value.get("non_tag_refs_must_target_commits") is not True
        or value.get("tags_must_be_annotated_direct_to_commits") is not True
    ):
        raise ReleaseGateError("git_refs policy must fail closed on every ref and tag target")
    required = {
        _text(item, name="required release tag")
        for item in _array(value.get("required_for_release_tags"), name="required_for_release_tags")
    }
    pinned: dict[str, tuple[str, str, str, str, str]] = {}
    for raw_name, raw_item in _mapping(
        value.get("pinned_annotated_tags"), name="pinned_annotated_tags"
    ).items():
        name = _text(raw_name, name="pinned tag name")
        if TAG_NAME_RE.fullmatch(name) is None:
            raise ReleaseGateError("pinned tag name is not release-portable")
        item = _mapping(raw_item, name=f"pinned tag {name}")
        _exact_keys(
            item,
            {"object_id", "target_commit", "message", "tagger_name", "tagger_email"},
            name=f"pinned tag {name}",
        )
        pinned[name] = (
            _commit(item.get("object_id"), name=f"pinned tag {name} object_id"),
            _commit(item.get("target_commit"), name=f"pinned tag {name} target_commit"),
            _text(item.get("message"), name=f"pinned tag {name} message"),
            _text(item.get("tagger_name"), name=f"pinned tag {name} tagger_name"),
            _text(item.get("tagger_email"), name=f"pinned tag {name} tagger_email"),
        )
    permitted: dict[str, tuple[str, str, str]] = {}
    for raw_name, raw_item in _mapping(
        value.get("permitted_candidate_tags"), name="permitted_candidate_tags"
    ).items():
        name = _text(raw_name, name="permitted candidate tag name")
        if TAG_NAME_RE.fullmatch(name) is None:
            raise ReleaseGateError("permitted candidate tag name is not release-portable")
        item = _mapping(raw_item, name=f"permitted candidate tag {name}")
        _exact_keys(
            item,
            {"message", "tagger_name", "tagger_email"},
            name=f"permitted candidate tag {name}",
        )
        message = _text(item.get("message"), name=f"permitted candidate tag {name} message")
        tagger_name = _text(
            item.get("tagger_name"), name=f"permitted candidate tag {name} tagger_name"
        )
        tagger_email = _text(
            item.get("tagger_email"), name=f"permitted candidate tag {name} tagger_email"
        )
        if any(
            ord(character) < 32 or ord(character) == 127
            for text in (message, tagger_name, tagger_email)
            for character in text
        ) or any(character in tagger_email for character in "<> "):
            raise ReleaseGateError("permitted candidate tag metadata contains unsafe characters")
        permitted[name] = (message, tagger_name, tagger_email)
    notes = _mapping(value.get("historical_notes"), name="git_refs historical_notes")
    if any(name not in pinned for name in notes) or any(
        not isinstance(note, str) or not note for note in notes.values()
    ):
        raise ReleaseGateError("historical tag notes must describe pinned tags")
    if set(pinned) & set(permitted) or required != set(pinned) | set(permitted):
        raise ReleaseGateError("required release tags must exactly match pinned and permitted tags")
    return required, pinned, permitted


def inspect_git_ref_state(
    root: Path, policy: Mapping[str, Any], *, candidate: str
) -> tuple[dict[str, int | str], list[dict[str, str]]]:
    required, pinned, permitted = _tag_policy(policy)
    listing = _git(
        root,
        ("for-each-ref", "--format=%(refname)%09%(objectname)%09%(objecttype)"),
    )
    assert isinstance(listing.stdout, str)
    violations: list[dict[str, str]] = []
    ref_rows: list[str] = []
    release_main_observed = False
    observed_tags: set[str] = set()
    tag_ref_count = 0
    validated_tag_count = 0
    invalid_ref_target_count = 0
    for line in listing.stdout.splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise ReleaseGateError("Git ref inventory returned malformed metadata")
        refname, raw_object_id, object_type = parts
        object_id = _commit(raw_object_id, name=f"{refname} object ID")
        peeled_result = _git(root, ("rev-parse", "--verify", f"{refname}^{{commit}}"), check=False)
        peeled: str | None = None
        if peeled_result.returncode == 0:
            assert isinstance(peeled_result.stdout, str)
            peeled = _commit(peeled_result.stdout.strip(), name=f"{refname} commit target")
        is_tag_ref = refname.startswith("refs/tags/")
        if not is_tag_ref:
            if refname in {"refs/heads/main", "refs/remotes/origin/main"}:
                if object_type == "commit" and object_id == candidate and peeled == candidate:
                    release_main_observed = True
                else:
                    violations.append(
                        {
                            "code": "release_main_ref_mismatch",
                            "commit": candidate,
                            "path": refname,
                            "detail": f"{object_type}:{object_id}",
                        }
                    )
            else:
                ref_rows.append(
                    f"ref-target\t{refname}\t{object_type}\t{object_id}\t{peeled or '-'}"
                )
            if object_type != "commit" or peeled != object_id:
                invalid_ref_target_count += 1
                violations.append(
                    {
                        "code": "non_commit_ref_target",
                        "commit": candidate,
                        "path": refname,
                        "detail": f"{object_type}:{object_id}",
                    }
                )
            if object_type == "tag":
                violations.append(
                    {
                        "code": "tag_object_outside_tag_namespace",
                        "commit": candidate,
                        "path": refname,
                        "detail": object_id,
                    }
                )
            continue

        tag_ref_count += 1
        tag_name = refname.removeprefix("refs/tags/")
        observed_tags.add(tag_name)
        ref_rows.append(f"tag\t{tag_name}\t{object_id}\t{object_type}\t{peeled or '-'}")
        if TAG_NAME_RE.fullmatch(tag_name) is None:
            violations.append(
                {
                    "code": "unsafe_tag_name",
                    "commit": candidate,
                    "path": refname,
                    "detail": tag_name,
                }
            )
            continue
        if object_type != "tag":
            invalid_ref_target_count += 1
            violations.append(
                {
                    "code": "lightweight_or_non_tag_ref",
                    "commit": candidate,
                    "path": refname,
                    "detail": f"{object_type}:{object_id}",
                }
            )
            continue
        tag_object = _git(root, ("cat-file", "tag", object_id), binary=True)
        assert isinstance(tag_object.stdout, bytes)
        payload = tag_object.stdout
        if len(payload) > MAX_TAG_OBJECT_BYTES:
            violations.append(
                {
                    "code": "oversized_tag_object",
                    "commit": candidate,
                    "path": refname,
                    "detail": str(len(payload)),
                }
            )
            continue
        try:
            header_bytes, message_bytes = payload.split(b"\n\n", 1)
            header_lines = header_bytes.decode("utf-8").splitlines()
            message = message_bytes.decode("utf-8")
        except (UnicodeError, ValueError) as exc:
            raise ReleaseGateError(f"tag object {tag_name} is not canonical UTF-8") from exc
        if len(header_lines) != 4:
            violations.append(
                {
                    "code": "noncanonical_tag_headers",
                    "commit": candidate,
                    "path": refname,
                    "detail": str(len(header_lines)),
                }
            )
            continue
        object_line, type_line, tag_line, tagger_line = header_lines
        direct_target = object_line.removeprefix("object ")
        tagger_match = re.fullmatch(r"tagger (.+) <([^<>]+)> ([0-9]+) ([+-][0-9]{4})", tagger_line)
        if (
            not object_line.startswith("object ")
            or COMMIT_RE.fullmatch(direct_target) is None
            or type_line != "type commit"
            or tag_line != f"tag {tag_name}"
            or tagger_match is None
            or peeled != direct_target
        ):
            invalid_ref_target_count += 1
            violations.append(
                {
                    "code": "noncanonical_tag_target",
                    "commit": candidate,
                    "path": refname,
                    "detail": object_id,
                }
            )
            continue
        if tag_name in pinned:
            (
                expected_object,
                expected_target,
                expected_message,
                expected_name,
                expected_email,
            ) = pinned[tag_name]
            assert tagger_match is not None
            if (
                object_id != expected_object
                or direct_target != expected_target
                or message != f"{expected_message}\n"
                or tagger_match.group(1) != expected_name
                or tagger_match.group(2) != expected_email
            ):
                violations.append(
                    {
                        "code": "pinned_tag_mismatch",
                        "commit": candidate,
                        "path": refname,
                        "detail": object_id,
                    }
                )
                continue
        elif tag_name in permitted:
            expected_message, expected_name, expected_email = permitted[tag_name]
            assert tagger_match is not None
            if (
                direct_target != candidate
                or message != f"{expected_message}\n"
                or tagger_match.group(1) != expected_name
                or tagger_match.group(2) != expected_email
            ):
                violations.append(
                    {
                        "code": "candidate_tag_metadata_mismatch",
                        "commit": candidate,
                        "path": refname,
                        "detail": object_id,
                    }
                )
                continue
        else:
            violations.append(
                {
                    "code": "unreviewed_annotated_tag",
                    "commit": candidate,
                    "path": refname,
                    "detail": object_id,
                }
            )
            continue
        validated_tag_count += 1
    for missing in sorted(required - observed_tags):
        violations.append(
            {
                "code": "required_release_tag_missing",
                "commit": candidate,
                "path": f"refs/tags/{missing}",
                "detail": "required by git_refs policy",
            }
        )
    if not release_main_observed:
        violations.append(
            {
                "code": "release_main_ref_missing",
                "commit": candidate,
                "path": "refs/heads/main|refs/remotes/origin/main",
                "detail": "no canonical main ref targets candidate_commit",
            }
        )
    else:
        ref_rows.append(f"release-main\tcommit\t{candidate}\t{candidate}")
    metrics: dict[str, int | str] = {
        "ref_count": len(ref_rows),
        "tag_ref_count": tag_ref_count,
        "validated_annotated_tag_count": validated_tag_count,
        "invalid_ref_target_count": invalid_ref_target_count,
        "ref_metadata_index_sha256": hashlib.sha256(
            ("\n".join(sorted(ref_rows)) + "\n").encode()
        ).hexdigest(),
    }
    return metrics, violations


def _blob_payload(root: Path, object_id: str, *, expected_size: int) -> bytes:
    result = _git(root, ("cat-file", "blob", object_id), binary=True)
    assert isinstance(result.stdout, bytes)
    if len(result.stdout) != expected_size:
        raise ReleaseGateError("Git blob size differs from tree/index metadata")
    return result.stdout


def _blob_content_violations(
    payload: bytes,
    *,
    path: str,
    signatures: Mapping[str, bytes],
    allowed_signatures: Mapping[str, set[str]],
    allowed_binary_suffixes: set[str],
) -> list[tuple[str, str]]:
    suffix = PurePosixPath(path).suffix.casefold()
    violations: list[tuple[str, str]] = []
    if payload.startswith(b"version https://git-lfs.github.com/spec/v1"):
        violations.append(("git_lfs_pointer", "Git LFS pointers are forbidden"))
    if suffix in TEXT_SUFFIXES:
        if b"\x00" in payload:
            violations.append(("disguised_binary_nul", "NUL byte in text-declared blob"))
        try:
            decoded = payload.decode("utf-8")
        except UnicodeDecodeError:
            violations.append(("invalid_text_encoding", "text-declared blob is not UTF-8"))
        else:
            if PurePosixPath(path).name.casefold() == ".gitattributes" and re.search(
                r"\bfilter\s*=\s*lfs\b", decoded, flags=re.IGNORECASE
            ):
                violations.append(
                    ("git_lfs_filter", "Git LFS filters are forbidden in .gitattributes")
                )
    elif suffix not in allowed_binary_suffixes:
        violations.append(("unsupported_binary_extension", suffix or "<none>"))
    elif not any(
        payload.startswith(signatures[name])
        for name, suffixes_for_signature in allowed_signatures.items()
        if suffix in suffixes_for_signature and name in signatures
    ):
        violations.append(
            ("invalid_allowed_binary_signature", f"{suffix} lacks its required magic")
        )
    for signature_name, signature in signatures.items():
        if payload.startswith(signature) and suffix not in allowed_signatures.get(
            signature_name, set()
        ):
            violations.append(("disguised_binary_signature", signature_name))
            break
    return violations


def _index_entries(root: Path) -> list[tuple[str, str, int, str, str, int | None]]:
    listing = _git(root, ("ls-files", "--stage", "-z"), binary=True)
    assert isinstance(listing.stdout, bytes)
    parsed: list[tuple[str, str, int, str]] = []
    for raw in (item for item in listing.stdout.split(b"\x00") if item):
        try:
            metadata, raw_path = raw.split(b"\t", 1)
            mode, object_id, stage_text = metadata.decode("ascii").split(" ", 2)
            path = _safe_relative(raw_path.decode("utf-8"), name="Git index path")
            stage = int(stage_text)
        except (UnicodeError, ValueError) as exc:
            raise ReleaseGateError("Git index contains undecodable metadata or path") from exc
        parsed.append((mode, object_id, stage, path))

    object_ids = sorted({object_id for _mode, object_id, _stage, _path in parsed})
    metadata_by_object: dict[str, tuple[str, int | None]] = {}
    if object_ids:
        request = ("\n".join(object_ids) + "\n").encode("ascii")
        try:
            checked = subprocess.run(
                [
                    "git",
                    "--no-replace-objects",
                    "cat-file",
                    "--batch-check=%(objectname) %(objecttype) %(objectsize)",
                ],
                cwd=root,
                input=request,
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ReleaseGateError("Git index object inspection failed") from exc
        lines = checked.stdout.decode("ascii").splitlines()
        if len(lines) != len(object_ids):
            raise ReleaseGateError("Git index object inspection returned an incomplete response")
        for requested, line in zip(object_ids, lines, strict=True):
            parts = line.split(" ")
            if len(parts) == 2 and parts == [requested, "missing"]:
                metadata_by_object[requested] = ("missing", None)
                continue
            if len(parts) != 3 or parts[0] != requested:
                raise ReleaseGateError("Git index object inspection returned unexpected metadata")
            try:
                size = int(parts[2])
            except ValueError as exc:
                raise ReleaseGateError("Git index object size is invalid") from exc
            metadata_by_object[requested] = (parts[1], size)
    return [(*entry, *metadata_by_object.get(entry[1], ("missing", None))) for entry in parsed]


def scan_index(
    *, repository_root: str | Path, policy_path: str | Path, created_at_utc: str
) -> dict[str, Any]:
    """Inspect the exact Git index tree without reading worktree payloads or writing Git objects."""

    root = _repository(repository_root)
    policy, policy_reference = _policy(policy_path)
    staged_policy = _mapping(policy.get("staged_index_scan"), name="staged_index_scan")
    if staged_policy != {
        "required_before_release_candidate_commit": True,
        "scan_scope": "exact_git_index",
    }:
        raise ReleaseGateError("staged_index_scan policy is missing or incompatible")
    maximum, prefixes, suffixes, signatures, allowed_signatures, allowed_binary_suffixes = (
        _content_scan_policy(policy)
    )
    credential_basenames, credential_prefixes, allowed_sensitive_paths = _credential_path_policy(
        policy
    )
    head_result = _git(root, ("rev-parse", "HEAD"))
    assert isinstance(head_result.stdout, str)
    head = _commit(head_result.stdout.strip(), name="HEAD")
    changed_result = _git(root, ("diff", "--cached", "--name-only", "-z"), binary=True)
    assert isinstance(changed_result.stdout, bytes)
    changed_paths = [value for value in changed_result.stdout.split(b"\x00") if value]

    violations: list[dict[str, str]] = []
    replace_result = _git(root, ("for-each-ref", "--format=%(refname)", "refs/replace/"))
    assert isinstance(replace_result.stdout, str)
    replace_refs = sorted(value for value in replace_result.stdout.splitlines() if value)
    graft_result = _git(root, ("rev-parse", "--git-path", "info/grafts"))
    assert isinstance(graft_result.stdout, str)
    graft_path = Path(graft_result.stdout.strip())
    if not graft_path.is_absolute():
        graft_path = root / graft_path
    grafts_file_present = graft_path.is_symlink() or (
        graft_path.exists() and graft_path.stat().st_size > 0
    )
    for reference in replace_refs:
        violations.append(
            {
                "code": "git_replace_ref",
                "path": reference,
                "detail": "replace refs are forbidden for release evidence",
            }
        )
    if grafts_file_present:
        violations.append(
            {
                "code": "git_grafts_file",
                "path": ".git/info/grafts",
                "detail": "legacy grafts can hide reachable history",
            }
        )
    folded_paths: dict[str, str] = {}
    index_rows: list[str] = []
    inspection_targets: set[tuple[str, str]] = set()
    unique_blobs: set[str] = set()
    entries = _index_entries(root)
    for mode, object_id, stage, path, kind, size in entries:
        index_rows.append(
            _entry_index_row(
                mode=mode,
                object_id=object_id,
                stage=stage,
                kind=kind,
                size=size,
                path=path,
            )
        )
        folded = path.casefold()
        previous = folded_paths.get(folded)
        if previous is not None and previous != path:
            violations.append(
                {"code": "case_collision", "path": path, "detail": f"collides with {previous}"}
            )
        folded_paths[folded] = path
        if stage != 0:
            violations.append(
                {"code": "unmerged_index_entry", "path": path, "detail": f"stage {stage}"}
            )
            continue
        if mode == "120000":
            violations.append({"code": "symlink", "path": path, "detail": mode})
            continue
        if mode == "160000" or kind == "commit":
            violations.append({"code": "submodule", "path": path, "detail": mode})
            continue
        if mode not in {"100644", "100755"} or kind != "blob" or size is None:
            violations.append(
                {
                    "code": "unsupported_index_entry",
                    "path": path,
                    "detail": f"{mode} {kind}",
                }
            )
            continue
        unique_blobs.add(object_id)
        prohibited_path = _is_forbidden_path(path, prefixes) or _is_forbidden_credential_path(
            path,
            basenames=credential_basenames,
            basename_prefixes=credential_prefixes,
            allowed_paths=allowed_sensitive_paths,
        )
        suffix = PurePosixPath(path).suffix.casefold()
        prohibited_suffix = suffix in suffixes
        if prohibited_path:
            violations.append(
                {
                    "code": "forbidden_path",
                    "path": path,
                    "detail": "protected/raw/transient prefix",
                }
            )
        if prohibited_suffix:
            violations.append({"code": "forbidden_extension", "path": path, "detail": suffix})
        if size > maximum:
            violations.append({"code": "oversized_blob", "path": path, "detail": str(size)})
        if not prohibited_path and not prohibited_suffix and size <= maximum:
            inspection_targets.add((object_id, path))

    for object_id, path in sorted(inspection_targets):
        size = next(
            entry_size
            for _mode, entry_object_id, stage, entry_path, _kind, entry_size in entries
            if entry_object_id == object_id and stage == 0 and entry_path == path
        )
        assert size is not None
        payload = _blob_payload(root, object_id, expected_size=size)
        for code, detail in _blob_content_violations(
            payload,
            path=path,
            signatures=signatures,
            allowed_signatures=allowed_signatures,
            allowed_binary_suffixes=allowed_binary_suffixes,
        ):
            violations.append(
                {
                    "code": code,
                    "path": path,
                    "detail": detail,
                }
            )

    unique_violations = sorted(
        {(item["code"], item["path"], item["detail"]): item for item in violations}.values(),
        key=lambda item: (item["code"], item["path"], item["detail"]),
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": INDEX_SCAN_KIND,
        "status": "pass" if not unique_violations else "fail",
        "created_at_utc": _timestamp(created_at_utc),
        "head_commit": head,
        "scan_scope": "exact_git_index",
        "policy": policy_reference,
        "index_entry_count": len(entries),
        "staged_change_count": len(changed_paths),
        "unique_blob_count": len(unique_blobs),
        "replace_ref_count": len(replace_refs),
        "grafts_file_present": grafts_file_present,
        "index_entry_index_sha256": hashlib.sha256(
            ("\n".join(sorted(index_rows)) + "\n").encode()
        ).hexdigest(),
        "violations": unique_violations,
        "raw_worktree_paths_opened": False,
        "git_objects_written": False,
    }
    return _self_hash(body)


def scan_repository(
    *,
    repository_root: str | Path,
    candidate_commit: str,
    policy_path: str | Path,
    created_at_utc: str,
) -> dict[str, Any]:
    """Scan the candidate tree and every reachable historical Git tree."""

    root = _repository(repository_root)
    candidate = _commit(candidate_commit, name="candidate_commit")
    _require_candidate_head(root, candidate, operation="repository scan")
    exists = _git(root, ("cat-file", "-e", f"{candidate}^{{commit}}"), check=False)
    if exists.returncode != 0:
        raise ReleaseGateError("candidate_commit is not present in the repository")
    policy, policy_reference = _policy(policy_path)
    maximum, prefixes, suffixes, signatures, allowed_signatures, allowed_binary_suffixes = (
        _content_scan_policy(policy)
    )
    credential_basenames, credential_prefixes, allowed_sensitive_paths = _credential_path_policy(
        policy
    )
    revisions = _git(root, ("rev-list", "--all", candidate))
    assert isinstance(revisions.stdout, str)
    commits = sorted(set(revisions.stdout.splitlines()) | {candidate})
    ref_metrics, ref_violations = inspect_git_ref_state(root, policy, candidate=candidate)
    violations: list[dict[str, Any]] = list(ref_violations)
    objects: dict[str, tuple[int, str]] = {}
    path_observations = 0
    inspection_targets: dict[tuple[str, str], str] = {}
    shallow_result = _git(root, ("rev-parse", "--is-shallow-repository"))
    assert isinstance(shallow_result.stdout, str)
    shallow = shallow_result.stdout.strip() == "true"
    replace_result = _git(root, ("for-each-ref", "--format=%(refname)", "refs/replace/"))
    assert isinstance(replace_result.stdout, str)
    replace_refs = sorted(value for value in replace_result.stdout.splitlines() if value)
    graft_result = _git(root, ("rev-parse", "--git-path", "info/grafts"))
    assert isinstance(graft_result.stdout, str)
    graft_path = Path(graft_result.stdout.strip())
    if not graft_path.is_absolute():
        graft_path = root / graft_path
    grafts_file_present = graft_path.is_symlink() or (
        graft_path.exists() and graft_path.stat().st_size > 0
    )
    if shallow:
        violations.append(
            {
                "code": "shallow_repository",
                "commit": candidate,
                "path": ".git",
                "detail": "complete reachable history is unavailable",
            }
        )
    for reference in replace_refs:
        violations.append(
            {
                "code": "git_replace_ref",
                "commit": candidate,
                "path": reference,
                "detail": "replace refs are forbidden for release evidence",
            }
        )
    if grafts_file_present:
        violations.append(
            {
                "code": "git_grafts_file",
                "commit": candidate,
                "path": ".git/info/grafts",
                "detail": "legacy grafts can hide reachable history",
            }
        )
    for revision in commits:
        folded_paths: dict[str, str] = {}
        for mode, kind, object_id, size, path in _tree_entries(root, revision):
            path_observations += 1
            folded = path.casefold()
            previous = folded_paths.get(folded)
            if previous is not None and previous != path:
                violations.append(
                    {
                        "code": "case_collision",
                        "commit": revision,
                        "path": path,
                        "detail": f"collides with {previous}",
                    }
                )
            folded_paths[folded] = path
            if mode == "120000":
                violations.append(
                    {"code": "symlink", "commit": revision, "path": path, "detail": mode}
                )
                continue
            if mode == "160000" or kind == "commit":
                violations.append(
                    {"code": "submodule", "commit": revision, "path": path, "detail": mode}
                )
                continue
            if kind != "blob" or size is None:
                violations.append(
                    {
                        "code": "unsupported_tree_entry",
                        "commit": revision,
                        "path": path,
                        "detail": f"{mode} {kind}",
                    }
                )
                continue
            objects.setdefault(object_id, (size, path))
            prohibited_path = _is_forbidden_path(path, prefixes) or _is_forbidden_credential_path(
                path,
                basenames=credential_basenames,
                basename_prefixes=credential_prefixes,
                allowed_paths=allowed_sensitive_paths,
            )
            prohibited_suffix = PurePosixPath(path).suffix.casefold() in suffixes
            if prohibited_path:
                violations.append(
                    {
                        "code": "forbidden_path",
                        "commit": revision,
                        "path": path,
                        "detail": "protected/raw/transient prefix",
                    }
                )
            if prohibited_suffix:
                violations.append(
                    {
                        "code": "forbidden_extension",
                        "commit": revision,
                        "path": path,
                        "detail": PurePosixPath(path).suffix.casefold(),
                    }
                )
            if size > maximum:
                violations.append(
                    {
                        "code": "oversized_blob",
                        "commit": revision,
                        "path": path,
                        "detail": str(size),
                    }
                )
            if prohibited_path or prohibited_suffix or size > maximum:
                continue
            inspection_targets.setdefault((object_id, path), revision)
    for (object_id, path), revision in sorted(inspection_targets.items()):
        size = objects[object_id][0]
        payload = _blob_payload(root, object_id, expected_size=size)
        for code, detail in _blob_content_violations(
            payload,
            path=path,
            signatures=signatures,
            allowed_signatures=allowed_signatures,
            allowed_binary_suffixes=allowed_binary_suffixes,
        ):
            violations.append(
                {
                    "code": code,
                    "commit": revision,
                    "path": path,
                    "detail": detail,
                }
            )
    # Deduplicate violations caused by the same path/object recurring unchanged in history.
    unique_violations = sorted(
        {(item["code"], item["path"], item["detail"]): item for item in violations}.values(),
        key=lambda item: (str(item["code"]), str(item["path"]), str(item["commit"])),
    )
    object_digest_rows = [f"{oid} {size} {path}" for oid, (size, path) in sorted(objects.items())]
    body = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": SCAN_KIND,
        "status": "pass" if not unique_violations else "fail",
        "created_at_utc": _timestamp(created_at_utc),
        "candidate_commit": candidate,
        "scan_scope": (
            "candidate_tree_and_complete_reachable_git_history"
            if not shallow and not grafts_file_present
            else "candidate_tree_and_observed_incomplete_history"
        ),
        "policy": policy_reference,
        "commit_count": len(commits),
        "unique_blob_count": len(objects),
        "tree_entry_observation_count": path_observations,
        **ref_metrics,
        "shallow_repository": shallow,
        "replace_ref_count": len(replace_refs),
        "grafts_file_present": grafts_file_present,
        "scanned_object_index_sha256": hashlib.sha256(
            ("\n".join(object_digest_rows) + "\n").encode()
        ).hexdigest(),
        "violations": unique_violations,
        "raw_worktree_paths_opened": False,
    }
    return _self_hash(body)


def validate_secret_scan(
    *,
    repository_root: str | Path,
    report_path: str | Path,
    candidate_commit: str,
    policy_path: str | Path,
    config_path: str | Path,
    ignore_path: str | Path,
    gitleaks_version: str,
    gitleaks_exit_code: int,
    created_at_utc: str,
) -> dict[str, Any]:
    """Validate a Gitleaks JSON report and bind it to an exact candidate."""

    root = _repository(repository_root)
    policy, policy_reference = _policy(policy_path)
    gitleaks_policy = _mapping(policy.get("gitleaks"), name="gitleaks policy")
    expected_version = _text(gitleaks_policy.get("version"), name="gitleaks.version")
    if gitleaks_version != expected_version:
        raise ReleaseGateError("Gitleaks version differs from the policy pin")
    if gitleaks_policy.get("required_scope") != SECRET_SCAN_SCOPE:
        raise ReleaseGateError("Gitleaks policy does not declare the required commit/tag scope")
    candidate = _commit(candidate_commit, name="candidate_commit")
    _require_candidate_head(root, candidate, operation="secret scan")
    ref_metrics, ref_violations = inspect_git_ref_state(root, policy, candidate=candidate)
    if ref_violations:
        raise ReleaseGateError("secret scan cannot attest unreviewed or unsafe Git ref metadata")
    expected_config_path = _safe_relative(
        gitleaks_policy.get("config_path"), name="gitleaks.config_path"
    )
    expected_config_sha256 = _sha256(
        gitleaks_policy.get("config_sha256"), name="gitleaks.config_sha256"
    )
    config_source = Path(config_path)
    if config_source.is_symlink():
        raise ReleaseGateError("Gitleaks configuration may not be a symlink")
    config_resolved = config_source.resolve(strict=True)
    try:
        config_relative = _safe_relative(
            config_resolved.relative_to(root).as_posix(), name="Gitleaks configuration"
        )
    except ValueError as exc:
        raise ReleaseGateError("Gitleaks configuration escapes repository_root") from exc
    if config_relative != expected_config_path or not config_resolved.is_file():
        raise ReleaseGateError("Gitleaks configuration differs from the policy path")
    config_payload = config_resolved.read_bytes()
    config_reference = {
        "path": config_relative,
        "size_bytes": len(config_payload),
        "file_sha256": hashlib.sha256(config_payload).hexdigest(),
    }
    if config_reference["file_sha256"] != expected_config_sha256:
        raise ReleaseGateError("Gitleaks configuration SHA-256 differs from the policy pin")
    if _git_blob_reference(root, candidate, config_reference, name="Gitleaks configuration") != (
        config_reference
    ):
        raise ReleaseGateError("Gitleaks configuration differs from the candidate blob")
    expected_ignore_path = _safe_relative(
        gitleaks_policy.get("ignore_path"), name="gitleaks.ignore_path"
    )
    ignore_source = Path(ignore_path)
    if ignore_source.is_symlink():
        raise ReleaseGateError("Gitleaks ignore file may not be a symlink")
    ignore_resolved = ignore_source.resolve(strict=True)
    try:
        ignore_relative = _safe_relative(
            ignore_resolved.relative_to(root).as_posix(), name="Gitleaks ignore file"
        )
    except ValueError as exc:
        raise ReleaseGateError("Gitleaks ignore file escapes repository_root") from exc
    if ignore_relative != expected_ignore_path or not ignore_resolved.is_file():
        raise ReleaseGateError("Gitleaks ignore file differs from the policy path")
    ignore_payload = ignore_resolved.read_bytes()
    try:
        ignore_text = ignore_payload.decode("utf-8-sig")
    except UnicodeError as exc:
        raise ReleaseGateError("Gitleaks ignore file must be UTF-8") from exc
    _validate_gitleaks_ignore_entries(policy, ignore_text)
    ignore_reference = {
        "path": ignore_relative,
        "size_bytes": len(ignore_payload),
        "file_sha256": hashlib.sha256(ignore_payload).hexdigest(),
    }
    if _git_blob_reference(root, candidate, ignore_reference, name="Gitleaks ignore file") != (
        ignore_reference
    ):
        raise ReleaseGateError("Gitleaks ignore file differs from the candidate blob")
    source = Path(report_path)
    if source.is_symlink():
        raise ReleaseGateError("Gitleaks report may not be a symlink")
    report_resolved = source.resolve(strict=True)
    if not report_resolved.is_file() or report_resolved.stat().st_size > MAX_EVIDENCE_BYTES:
        raise ReleaseGateError("Gitleaks report exceeds the evidence size cap")
    payload = report_resolved.read_bytes()
    if len(payload) > MAX_EVIDENCE_BYTES:
        raise ReleaseGateError("Gitleaks report exceeds the evidence size cap")
    try:
        findings = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseGateError("Gitleaks report is not UTF-8 JSON") from exc
    if not isinstance(findings, list):
        raise ReleaseGateError("Gitleaks JSON report must be an array")
    if not isinstance(gitleaks_exit_code, int) or isinstance(gitleaks_exit_code, bool):
        raise ReleaseGateError("Gitleaks exit code must be an integer")
    execution_complete = (gitleaks_exit_code == 0 and not findings) or (
        gitleaks_exit_code == 1 and bool(findings)
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": SECRET_KIND,
        "status": "pass" if execution_complete and not findings else "fail",
        "created_at_utc": _timestamp(created_at_utc),
        "candidate_commit": candidate,
        "scan_scope": SECRET_SCAN_SCOPE,
        "ref_metadata_index_sha256": ref_metrics["ref_metadata_index_sha256"],
        "validated_annotated_tag_count": ref_metrics["validated_annotated_tag_count"],
        "gitleaks_version": gitleaks_version,
        "gitleaks_exit_code": gitleaks_exit_code,
        "execution_status": "completed" if execution_complete else "operational_error",
        "policy": policy_reference,
        "config": config_reference,
        "ignore": ignore_reference,
        "raw_report": {
            "basename": source.name,
            "size_bytes": len(payload),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
        },
        "finding_count": len(findings),
    }
    return _self_hash(body)


def audit_licenses(
    *,
    repository_root: str | Path,
    inventory_path: str | Path,
    candidate_commit: str,
    policy_path: str | Path,
    pip_licenses_version: str,
    created_at_utc: str,
) -> dict[str, Any]:
    """Validate pip-licenses JSON against the versioned release policy."""

    root = _repository(repository_root)
    candidate = _commit(candidate_commit, name="candidate_commit")
    _require_candidate_head(root, candidate, operation="license audit")
    if pip_licenses_version != "5.5.5":
        raise ReleaseGateError("pip-licenses version must equal the locked 5.5.5 release tool")
    policy, policy_reference = _policy(policy_path)
    source = Path(inventory_path)
    if source.is_symlink():
        raise ReleaseGateError("license inventory may not be a symlink")
    inventory_resolved = source.resolve(strict=True)
    if not inventory_resolved.is_file() or inventory_resolved.stat().st_size > MAX_EVIDENCE_BYTES:
        raise ReleaseGateError("license inventory exceeds the evidence size cap")
    payload = inventory_resolved.read_bytes()
    if len(payload) > MAX_EVIDENCE_BYTES:
        raise ReleaseGateError("license inventory exceeds the evidence size cap")
    try:
        packages = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseGateError("license inventory is not UTF-8 JSON") from exc
    values = _array(packages, name="license inventory")
    prohibited = tuple(
        _text(value, name="prohibited license token").casefold()
        for value in _array(policy.get("prohibited_license_tokens"), name="license tokens")
    )
    exceptions = {
        _text(value, name="license exception").casefold()
        for value in _array(policy.get("license_exceptions"), name="license exceptions")
    }
    violations: list[dict[str, str]] = []
    normalized: list[str] = []
    for index, raw_package in enumerate(values):
        package = _mapping(raw_package, name=f"license inventory[{index}]")
        name = _text(package.get("Name"), name=f"license inventory[{index}].Name")
        version = _text(package.get("Version"), name=f"license inventory[{index}].Version")
        license_name = _text(package.get("License"), name=f"license inventory[{index}].License")
        identity = f"{name}@{version}".casefold()
        normalized.append(f"{name}\t{version}\t{license_name}")
        if identity not in exceptions and any(
            token in license_name.casefold() for token in prohibited
        ):
            violations.append({"package": name, "version": version, "license": license_name})
    body = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": LICENSE_KIND,
        "status": "pass" if not violations else "fail",
        "created_at_utc": _timestamp(created_at_utc),
        "candidate_commit": candidate,
        "pip_licenses_version": pip_licenses_version,
        "policy": policy_reference,
        "inventory": {
            "basename": source.name,
            "size_bytes": len(payload),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
        },
        "package_count": len(values),
        "normalized_inventory_sha256": hashlib.sha256(
            ("\n".join(sorted(normalized, key=str.casefold)) + "\n").encode()
        ).hexdigest(),
        "violations": sorted(violations, key=lambda item: item["package"].casefold()),
    }
    return _self_hash(body)


def _evidence_snapshot(
    root: Path, value: Mapping[str, Any], *, name: str
) -> tuple[dict[str, Any], bytes]:
    path = _safe_relative(value.get("path"), name=f"{name}.path")
    expected = _sha256(value.get("expected_sha256"), name=f"{name}.expected_sha256")
    source = root / Path(path)
    current = root
    for part in PurePosixPath(path).parts:
        current = current / part
        if current.is_symlink():
            raise ReleaseGateError(f"{name} may not traverse a symlink")
    resolved = source.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ReleaseGateError(f"{name} escapes repository_root") from exc
    if not resolved.is_file() or resolved.stat().st_size > MAX_EVIDENCE_BYTES:
        raise ReleaseGateError(f"{name} must be a regular file within the evidence size cap")
    payload = resolved.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise ReleaseGateError(f"{name} SHA-256 differs")
    return {"path": path, "size_bytes": len(payload), "file_sha256": observed}, payload


def _evidence_reference(root: Path, value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    reference, _payload = _evidence_snapshot(root, value, name=name)
    return reference


def _file_reference_shape(value: Any, *, name: str) -> dict[str, Any]:
    reference = _mapping(value, name=name)
    if set(reference) != {"path", "size_bytes", "file_sha256"}:
        raise ReleaseGateError(f"{name} keys differ")
    path = _safe_relative(reference.get("path"), name=f"{name}.path")
    size = reference.get("size_bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ReleaseGateError(f"{name}.size_bytes must be a non-negative integer")
    digest = _sha256(reference.get("file_sha256"), name=f"{name}.file_sha256")
    return {"path": path, "size_bytes": size, "file_sha256": digest}


def _git_blob_reference(
    root: Path, commit: str, value: Mapping[str, Any], *, name: str
) -> dict[str, Any]:
    path = _safe_relative(value.get("path"), name=f"{name}.path")
    expected = _sha256(value.get("file_sha256"), name=f"{name}.file_sha256")
    tree = _git(root, ("ls-tree", "-z", commit, "--", path), binary=True)
    assert isinstance(tree.stdout, bytes)
    entries = [entry for entry in tree.stdout.split(b"\x00") if entry]
    if len(entries) != 1:
        raise ReleaseGateError(f"{name} is not one tracked candidate file")
    try:
        metadata, observed_path = entries[0].split(b"\t", 1)
        mode, kind, _object_id = metadata.decode("ascii").split(" ", 2)
        decoded_path = observed_path.decode("utf-8")
    except (UnicodeError, ValueError) as exc:
        raise ReleaseGateError(f"{name} has invalid Git tree metadata") from exc
    if decoded_path != path or kind != "blob" or mode not in {"100644", "100755"}:
        raise ReleaseGateError(f"{name} must be a regular tracked file")
    blob = _git(root, ("cat-file", "blob", f"{commit}:{path}"), binary=True)
    assert isinstance(blob.stdout, bytes)
    observed = hashlib.sha256(blob.stdout).hexdigest()
    if observed != expected:
        raise ReleaseGateError(f"{name} SHA-256 differs from its bound Git blob")
    return {"path": path, "size_bytes": len(blob.stdout), "file_sha256": observed}


def _git_blob_payload(root: Path, commit: str, value: Mapping[str, Any], *, name: str) -> bytes:
    _git_blob_reference(root, commit, value, name=name)
    path = _safe_relative(value.get("path"), name=f"{name}.path")
    blob = _git(root, ("cat-file", "blob", f"{commit}:{path}"), binary=True)
    assert isinstance(blob.stdout, bytes)
    return blob.stdout


def _validate_gitleaks_ignore_entries(policy: Mapping[str, Any], ignore_text: str) -> None:
    gitleaks_policy = _mapping(policy.get("gitleaks"), name="gitleaks policy")
    reviewed = [
        _text(value, name="gitleaks.reviewed_ignore_entries entry")
        for value in _array(
            gitleaks_policy.get("reviewed_ignore_entries"),
            name="gitleaks.reviewed_ignore_entries",
        )
    ]
    observed = [
        line.strip()
        for line in ignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if observed != reviewed:
        raise ReleaseGateError("Gitleaks ignore entries differ from the reviewed policy")


def _tracked_report_policy_reference(
    root: Path,
    report_path: str | Path,
    parent_commit: str,
    policy_reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve a pending report's policy without trusting mutable worktree bytes."""

    try:
        return _git_blob_reference(root, parent_commit, policy_reference, name="policy")
    except ReleaseGateError as parent_error:
        source = Path(report_path).resolve(strict=True)
        try:
            relative = _safe_relative(
                source.relative_to(root).as_posix(), name="tracked report path"
            )
        except ValueError as exc:
            raise ReleaseGateError(
                "tracked precommit report policy is not bound to repository history"
            ) from exc
        payload = source.read_bytes()
        revisions = _git(root, ("rev-list", "--all", "--", relative))
        assert isinstance(revisions.stdout, str)
        for candidate in revisions.stdout.splitlines():
            candidate = _commit(candidate, name="report-containing commit")
            observed_parent = _git(root, ("rev-parse", f"{candidate}^"), check=False)
            if observed_parent.returncode != 0:
                continue
            assert isinstance(observed_parent.stdout, str)
            if observed_parent.stdout.strip().casefold() != parent_commit:
                continue
            report_blob = _git(root, ("cat-file", "blob", f"{candidate}:{relative}"), binary=True)
            assert isinstance(report_blob.stdout, bytes)
            if report_blob.stdout != payload:
                continue
            try:
                return _git_blob_reference(root, candidate, policy_reference, name="policy")
            except ReleaseGateError:
                continue
        raise ReleaseGateError(
            "tracked precommit report policy is not bound to its parent or direct content commit"
        ) from parent_error


def validate_ci_gate_bundle(
    *,
    repository_root: str | Path,
    candidate_commit: str,
    policy_path: str | Path,
    evidence_root: str | Path,
    created_at_utc: str,
) -> dict[str, Any]:
    """Validate the candidate-bound CI gate bundle without claiming completed CI."""

    root = _repository(repository_root)
    candidate = _commit(candidate_commit, name="candidate_commit")
    head_result = _git(root, ("rev-parse", "HEAD"))
    assert isinstance(head_result.stdout, str)
    if head_result.stdout.strip().casefold() != candidate:
        raise ReleaseGateError("CI bundle candidate differs from repository HEAD")
    status_result = _git(root, ("status", "--porcelain=v1", "--untracked-files=all"))
    assert isinstance(status_result.stdout, str)
    if status_result.stdout.strip():
        raise ReleaseGateError("CI bundle repository worktree is not clean")
    parent_result = _git(root, ("rev-parse", f"{candidate}^"), check=False)
    if parent_result.returncode != 0:
        raise ReleaseGateError("CI bundle candidate has no resolvable first parent")
    assert isinstance(parent_result.stdout, str)
    parent = _commit(parent_result.stdout.strip(), name="candidate first parent")
    _policy_value, policy_reference = _policy(policy_path)
    bundle_root = Path(evidence_root)
    if bundle_root.is_symlink():
        raise ReleaseGateError("CI evidence root may not be a symlink")
    resolved_root = bundle_root.resolve(strict=True)
    try:
        resolved_root.relative_to(root)
    except ValueError as exc:
        raise ReleaseGateError("CI evidence root escapes repository_root") from exc
    if not resolved_root.is_dir():
        raise ReleaseGateError("CI evidence root must be a directory")

    paths = {
        "repository_scan": resolved_root / "repository_scan.json",
        "secret_scan": resolved_root / "secret_scan.json",
        "license_audit": resolved_root / "license_audit.json",
        **{name: resolved_root / "quality_gates.json" for name in QUALITY_GATE_NAMES},
    }
    references: dict[str, dict[str, Any]] = {}
    for gate_name, path in paths.items():
        if path.is_symlink():
            raise ReleaseGateError(f"CI gate {gate_name} evidence may not be a symlink")
        resolved_path = path.resolve(strict=True)
        if not resolved_path.is_file() or resolved_path.stat().st_size > MAX_EVIDENCE_BYTES:
            raise ReleaseGateError(f"CI gate {gate_name} evidence exceeds the evidence size cap")
        payload = resolved_path.read_bytes()
        relative = _safe_relative(
            path.resolve().relative_to(root).as_posix(), name=f"CI gate {gate_name} evidence"
        )
        reference = {
            "path": relative,
            "size_bytes": len(payload),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
        }
        evidence = _strict_json_bytes(payload, name=f"CI gate {gate_name} evidence")
        _validate_exact_gate_evidence(
            root,
            gate_name=gate_name,
            evidence=evidence,
            candidate=candidate,
            parent=parent,
            policy_reference=policy_reference,
        )
        references[gate_name] = reference
    return _self_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "record_kind": "ci_release_gate_bundle_validation",
            "status": "pass",
            "created_at_utc": _timestamp(created_at_utc),
            "candidate_commit": candidate,
            "parent_commit": parent,
            "ci_evidence_scope": QUALITY_EVIDENCE_SCOPE,
            "external_completed_ci_required_for_release_inventory": True,
            "validated_gates": sorted(references),
            "evidence": references,
        }
    )


def _validate_exact_gate_evidence(
    root: Path,
    *,
    gate_name: str,
    evidence: Mapping[str, Any],
    candidate: str,
    parent: str,
    policy_reference: Mapping[str, Any],
) -> None:
    """Fail closed on the semantics of one exact-candidate gate record."""

    _verify_self_hash(evidence, name=f"gates.{gate_name} evidence")
    if evidence.get("schema_version") != SCHEMA_VERSION or evidence.get("status") != "pass":
        raise ReleaseGateError(f"gates.{gate_name} evidence version or status differs")

    if gate_name == "staged_index_scan":
        _exact_keys(
            evidence,
            {
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
            },
            name="gates.staged_index_scan evidence",
        )
        _timestamp(evidence.get("created_at_utc"))
        _exact_integer(
            evidence.get("replace_ref_count"),
            name="gates.staged_index_scan evidence.replace_ref_count",
            expected=0,
        )
        if (
            evidence.get("record_kind") != INDEX_SCAN_KIND
            or evidence.get("head_commit") != parent
            or evidence.get("scan_scope") != "exact_git_index"
            or evidence.get("policy") != policy_reference
            or evidence.get("raw_worktree_paths_opened") is not False
            or evidence.get("git_objects_written") is not False
            or evidence.get("grafts_file_present") is not False
        ):
            raise ReleaseGateError("staged_index_scan evidence is incompatible or unsafe")
        violations = _array(
            evidence.get("violations"), name="gates.staged_index_scan evidence.violations"
        )
        staged_changes = evidence.get("staged_change_count")
        entry_count = evidence.get("index_entry_count")
        unique_blob_count = evidence.get("unique_blob_count")
        digest = evidence.get("index_entry_index_sha256")
        if (
            violations
            or not isinstance(staged_changes, int)
            or isinstance(staged_changes, bool)
            or staged_changes < 1
            or not isinstance(entry_count, int)
            or isinstance(entry_count, bool)
            or entry_count < 1
            or not isinstance(unique_blob_count, int)
            or isinstance(unique_blob_count, bool)
            or unique_blob_count < 1
            or unique_blob_count > entry_count
        ):
            raise ReleaseGateError("staged_index_scan evidence coverage is incomplete")
        _sha256(digest, name="gates.staged_index_scan evidence.index_entry_index_sha256")
        candidate_count, candidate_digest = _candidate_tree_index(root, candidate)
        if entry_count != candidate_count or digest != candidate_digest:
            raise ReleaseGateError("staged_index_scan does not reconstruct the candidate tree")
        return

    if evidence.get("candidate_commit") != candidate:
        raise ReleaseGateError(f"gates.{gate_name} evidence is bound to another commit")

    if gate_name in QUALITY_GATE_NAMES:
        _exact_keys(
            evidence,
            {
                "schema_version",
                "record_kind",
                "status",
                "candidate_commit",
                "created_at_utc",
                "gates",
                "gate_statuses",
                "scope",
                "ci_evidence_scope",
                "external_completed_ci_required_for_release_inventory",
                "gate_evidence",
                "tool_versions",
                "machine",
                "record_sha256",
            },
            name=f"gates.{gate_name} quality evidence",
        )
        if (
            evidence.get("record_kind") != QUALITY_EVIDENCE_KIND
            or evidence.get("ci_evidence_scope") != QUALITY_EVIDENCE_SCOPE
            or evidence.get("external_completed_ci_required_for_release_inventory") is not True
        ):
            raise ReleaseGateError(f"gates.{gate_name} quality evidence scope differs")
        declared_gates = [
            _text(value, name=f"gates.{gate_name} evidence gate")
            for value in _array(evidence.get("gates"), name=f"gates.{gate_name} evidence.gates")
        ]
        covered = set(declared_gates)
        expected_gates = set(QUALITY_GATE_NAMES)
        statuses = _mapping(
            evidence.get("gate_statuses"), name=f"gates.{gate_name} evidence.gate_statuses"
        )
        gate_evidence = _mapping(
            evidence.get("gate_evidence"), name=f"gates.{gate_name} evidence.gate_evidence"
        )
        executed_gates = expected_gates - {"ci_quality_proxy"}
        tools = _mapping(
            evidence.get("tool_versions"), name=f"gates.{gate_name} evidence.tool_versions"
        )
        machine = _mapping(evidence.get("machine"), name=f"gates.{gate_name} evidence.machine")
        if (
            len(covered) != len(declared_gates)
            or covered != expected_gates
            or set(statuses) != covered
            or any(value not in {"pass", "fail"} for value in statuses.values())
            or ("pass" if all(value == "pass" for value in statuses.values()) else "fail")
            != evidence.get("status")
            or gate_name not in covered
            or statuses.get(gate_name) != "pass"
            or set(gate_evidence) != executed_gates
            or set(tools) != {"python", "uv", "pytest", "ruff", "mypy"}
            or any(not isinstance(value, str) or not value for value in tools.values())
            or not str(tools.get("uv", "")).startswith("uv 0.11.29")
            or set(machine) != {"runner_os", "architecture", "platform"}
            or machine.get("runner_os") != "Linux"
            or any(not isinstance(machine.get(key), str) or not machine.get(key) for key in machine)
            or evidence.get("scope")
            != "upstream synthetic-validation matrix and preceding release-security steps"
        ):
            raise ReleaseGateError(f"gates.{gate_name} quality coverage is inconsistent")
        _timestamp(evidence.get("created_at_utc"))
        for executed_gate in sorted(executed_gates):
            item = _mapping(
                gate_evidence[executed_gate],
                name=f"gates.{gate_name} evidence.gate_evidence.{executed_gate}",
            )
            if set(item) != {
                "exit_code",
                "started_at_utc",
                "completed_at_utc",
                "log_path",
                "log_size_bytes",
                "log_sha256",
            }:
                raise ReleaseGateError(f"quality gate {executed_gate} provenance keys differ")
            exit_code = item.get("exit_code")
            started = _timestamp(item.get("started_at_utc"))
            completed = _timestamp(item.get("completed_at_utc"))
            if datetime.fromisoformat(completed[:-1] + "+00:00") < datetime.fromisoformat(
                started[:-1] + "+00:00"
            ):
                raise ReleaseGateError(f"quality gate {executed_gate} timestamps are reversed")
            if (
                not isinstance(exit_code, int)
                or isinstance(exit_code, bool)
                or (statuses[executed_gate] == "pass" and exit_code != 0)
                or (statuses[executed_gate] == "fail" and exit_code == 0)
            ):
                raise ReleaseGateError(f"quality gate {executed_gate} exit code is inconsistent")
            log_reference = _evidence_reference(
                root,
                {
                    "path": item.get("log_path"),
                    "expected_sha256": item.get("log_sha256"),
                },
                name=f"quality gate {executed_gate} log",
            )
            if item.get("log_size_bytes") != log_reference["size_bytes"]:
                raise ReleaseGateError(f"quality gate {executed_gate} log size differs")
        return

    if gate_name == "repository_scan":
        _exact_keys(
            evidence,
            {
                "schema_version",
                "record_kind",
                "status",
                "created_at_utc",
                "candidate_commit",
                "scan_scope",
                "policy",
                "commit_count",
                "unique_blob_count",
                "tree_entry_observation_count",
                "ref_count",
                "tag_ref_count",
                "validated_annotated_tag_count",
                "invalid_ref_target_count",
                "ref_metadata_index_sha256",
                "shallow_repository",
                "replace_ref_count",
                "grafts_file_present",
                "scanned_object_index_sha256",
                "violations",
                "raw_worktree_paths_opened",
                "record_sha256",
            },
            name="gates.repository_scan evidence",
        )
        _timestamp(evidence.get("created_at_utc"))
        commit_count = evidence.get("commit_count")
        unique_blob_count = evidence.get("unique_blob_count")
        observation_count = evidence.get("tree_entry_observation_count")
        candidate_policy_payload = _git_blob_payload(
            root, candidate, policy_reference, name="release gate policy"
        )
        candidate_policy = _strict_json_bytes(
            candidate_policy_payload, name="candidate release gate policy"
        )
        ref_metrics, ref_violations = inspect_git_ref_state(
            root, candidate_policy, candidate=candidate
        )
        if ref_violations:
            raise ReleaseGateError("repository_scan current Git ref state is unsafe")
        for name, value in {
            "commit_count": commit_count,
            "unique_blob_count": unique_blob_count,
            "tree_entry_observation_count": observation_count,
            "ref_count": evidence.get("ref_count"),
            "tag_ref_count": evidence.get("tag_ref_count"),
            "validated_annotated_tag_count": evidence.get("validated_annotated_tag_count"),
            "invalid_ref_target_count": evidence.get("invalid_ref_target_count"),
        }.items():
            minimum = (
                1
                if name
                in {
                    "commit_count",
                    "unique_blob_count",
                    "tree_entry_observation_count",
                    "ref_count",
                }
                else 0
            )
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ReleaseGateError(f"repository_scan {name} is incomplete")
        assert isinstance(unique_blob_count, int)
        assert isinstance(observation_count, int)
        if observation_count < unique_blob_count:
            raise ReleaseGateError("repository_scan observation coverage is incomplete")
        _sha256(
            evidence.get("scanned_object_index_sha256"),
            name="repository_scan.scanned_object_index_sha256",
        )
        _sha256(
            evidence.get("ref_metadata_index_sha256"),
            name="repository_scan.ref_metadata_index_sha256",
        )
        for metric in (
            "ref_count",
            "tag_ref_count",
            "validated_annotated_tag_count",
            "invalid_ref_target_count",
            "ref_metadata_index_sha256",
        ):
            if evidence.get(metric) != ref_metrics[metric]:
                raise ReleaseGateError(f"repository_scan {metric} differs from current Git refs")
        _exact_integer(
            evidence.get("replace_ref_count"),
            name="repository_scan.replace_ref_count",
            expected=0,
        )
        if (
            evidence.get("record_kind") != SCAN_KIND
            or evidence.get("policy") != policy_reference
            or evidence.get("scan_scope") != "candidate_tree_and_complete_reachable_git_history"
            or evidence.get("raw_worktree_paths_opened") is not False
            or evidence.get("shallow_repository") is not False
            or evidence.get("grafts_file_present") is not False
            or _array(evidence.get("violations"), name="repository_scan violations")
        ):
            raise ReleaseGateError("repository_scan evidence is incomplete or unsafe")
        return

    if gate_name == "secret_scan":
        _exact_keys(
            evidence,
            {
                "schema_version",
                "record_kind",
                "status",
                "created_at_utc",
                "candidate_commit",
                "scan_scope",
                "ref_metadata_index_sha256",
                "validated_annotated_tag_count",
                "gitleaks_version",
                "gitleaks_exit_code",
                "execution_status",
                "policy",
                "config",
                "ignore",
                "raw_report",
                "finding_count",
                "record_sha256",
            },
            name="gates.secret_scan evidence",
        )
        _timestamp(evidence.get("created_at_utc"))
        finding_count = evidence.get("finding_count")
        config = _mapping(evidence.get("config"), name="secret_scan config")
        ignore = _mapping(evidence.get("ignore"), name="secret_scan ignore")
        raw_report = _mapping(evidence.get("raw_report"), name="secret_scan raw_report")
        for name, reference in {"config": config, "ignore": ignore}.items():
            _exact_keys(
                reference,
                {"path", "size_bytes", "file_sha256"},
                name=f"secret_scan {name}",
            )
        _exact_keys(
            raw_report,
            {"basename", "size_bytes", "file_sha256"},
            name="secret_scan raw_report",
        )
        raw_basename = _text(raw_report.get("basename"), name="secret_scan raw report basename")
        raw_size = raw_report.get("size_bytes")
        if (
            PurePosixPath(raw_basename).name != raw_basename
            or "\\" in raw_basename
            or ":" in raw_basename
            or any(ord(character) < 32 or ord(character) == 127 for character in raw_basename)
            or not isinstance(raw_size, int)
            or isinstance(raw_size, bool)
            or raw_size < 0
        ):
            raise ReleaseGateError("secret_scan raw report reference is invalid")
        _sha256(raw_report.get("file_sha256"), name="secret_scan raw report file_sha256")
        observed_config = _git_blob_reference(
            root, candidate, config, name="Gitleaks configuration"
        )
        observed_ignore = _git_blob_reference(root, candidate, ignore, name="Gitleaks ignore file")
        policy_payload = _git_blob_payload(
            root, candidate, policy_reference, name="release gate policy"
        )
        candidate_policy = _strict_json_bytes(policy_payload, name="candidate release gate policy")
        ref_metrics, ref_violations = inspect_git_ref_state(
            root, candidate_policy, candidate=candidate
        )
        if ref_violations:
            raise ReleaseGateError("secret_scan current Git ref state is unsafe")
        gitleaks_policy = _mapping(
            candidate_policy.get("gitleaks"), name="candidate Gitleaks policy"
        )
        ignore_payload = _git_blob_payload(root, candidate, ignore, name="Gitleaks ignore file")
        try:
            ignore_text = ignore_payload.decode("utf-8-sig")
        except UnicodeError as exc:
            raise ReleaseGateError("Gitleaks ignore file must be UTF-8") from exc
        _validate_gitleaks_ignore_entries(candidate_policy, ignore_text)
        _exact_integer(
            evidence.get("gitleaks_exit_code"),
            name="secret_scan.gitleaks_exit_code",
            expected=0,
        )
        if (
            evidence.get("record_kind") != SECRET_KIND
            or evidence.get("policy") != policy_reference
            or evidence.get("scan_scope") != SECRET_SCAN_SCOPE
            or evidence.get("ref_metadata_index_sha256") != ref_metrics["ref_metadata_index_sha256"]
            or evidence.get("validated_annotated_tag_count")
            != ref_metrics["validated_annotated_tag_count"]
            or evidence.get("gitleaks_version") != gitleaks_policy.get("version")
            or gitleaks_policy.get("required_scope") != SECRET_SCAN_SCOPE
            or config.get("path") != gitleaks_policy.get("config_path")
            or config.get("file_sha256") != gitleaks_policy.get("config_sha256")
            or ignore.get("path") != gitleaks_policy.get("ignore_path")
            or evidence.get("execution_status") != "completed"
            or observed_config != config
            or observed_ignore != ignore
            or not isinstance(finding_count, int)
            or isinstance(finding_count, bool)
            or finding_count != 0
        ):
            raise ReleaseGateError("secret_scan evidence is incomplete or inconsistent")
        return

    if gate_name == "license_audit":
        _exact_keys(
            evidence,
            {
                "schema_version",
                "record_kind",
                "status",
                "created_at_utc",
                "candidate_commit",
                "pip_licenses_version",
                "policy",
                "inventory",
                "package_count",
                "normalized_inventory_sha256",
                "violations",
                "record_sha256",
            },
            name="gates.license_audit evidence",
        )
        _timestamp(evidence.get("created_at_utc"))
        package_count = evidence.get("package_count")
        inventory = _mapping(evidence.get("inventory"), name="license_audit inventory")
        _exact_keys(
            inventory,
            {"basename", "size_bytes", "file_sha256"},
            name="license_audit inventory",
        )
        inventory_basename = _text(
            inventory.get("basename"), name="license_audit inventory basename"
        )
        inventory_size = inventory.get("size_bytes")
        if (
            PurePosixPath(inventory_basename).name != inventory_basename
            or "\\" in inventory_basename
            or ":" in inventory_basename
            or any(ord(character) < 32 or ord(character) == 127 for character in inventory_basename)
            or not isinstance(inventory_size, int)
            or isinstance(inventory_size, bool)
            or inventory_size < 0
        ):
            raise ReleaseGateError("license_audit inventory reference is invalid")
        _sha256(inventory.get("file_sha256"), name="license_audit inventory file_sha256")
        _sha256(
            evidence.get("normalized_inventory_sha256"),
            name="license_audit normalized_inventory_sha256",
        )
        if (
            evidence.get("record_kind") != LICENSE_KIND
            or evidence.get("policy") != policy_reference
            or evidence.get("pip_licenses_version") != "5.5.5"
            or not isinstance(package_count, int)
            or isinstance(package_count, bool)
            or package_count < 1
            or _array(evidence.get("violations"), name="license_audit violations")
        ):
            raise ReleaseGateError("license_audit evidence is incomplete or inconsistent")
        return

    raise ReleaseGateError(f"no semantic validator exists for gate {gate_name}")


def assemble_report(
    *,
    repository_root: str | Path,
    policy_path: str | Path,
    mode: str,
    created_at_utc: str,
    spec_path: str | Path | None = None,
) -> dict[str, Any]:
    """Assemble a truthful tracked precommit report or exact-candidate attestation."""

    root = _repository(repository_root)
    _policy_value, policy_reference = _policy(policy_path)
    head_result = _git(root, ("rev-parse", "HEAD"))
    assert isinstance(head_result.stdout, str)
    head = _commit(head_result.stdout.strip(), name="HEAD")
    if mode == "tracked_precommit_report":
        if spec_path is not None:
            raise ReleaseGateError("tracked_precommit_report does not accept an attestation spec")
        return _self_hash(
            {
                "schema_version": SCHEMA_VERSION,
                "record_kind": REPORT_KIND,
                "mode": mode,
                "status": "pending",
                "created_at_utc": _timestamp(created_at_utc),
                "repository": {
                    "parent_commit": head,
                    "content_commit": None,
                    "worktree_clean": False,
                },
                "policy": policy_reference,
                "gates": {
                    name: {"status": "not_run", "evidence": None} for name in TRACKED_GATE_NAMES
                },
                "external_candidate_attestation": {
                    "status": "pending",
                    "expected_root": ".audit/release-attestations/<candidate_commit>/",
                    "report_path": None,
                },
            }
        )
    if mode != "exact_candidate_attestation" or spec_path is None:
        raise ReleaseGateError("exact_candidate_attestation requires --spec")
    spec = _strict_json(spec_path, name="exact candidate attestation spec")
    expected_keys = {
        "schema_version",
        "spec_kind",
        "created_at_utc",
        "candidate_commit",
        "parent_commit",
        "worktree_clean",
        "gates",
    }
    if set(spec) != expected_keys:
        raise ReleaseGateError("exact candidate attestation spec keys differ")
    if spec.get("schema_version") != "1.0.0" or spec.get("spec_kind") != (
        "final_release_gate_attestation_spec"
    ):
        raise ReleaseGateError("exact candidate attestation spec version or kind differs")
    candidate = _commit(spec.get("candidate_commit"), name="candidate_commit")
    parent = _commit(spec.get("parent_commit"), name="parent_commit")
    if spec.get("worktree_clean") is not True:
        raise ReleaseGateError("exact candidate attestation must record a clean worktree")
    if candidate != head:
        raise ReleaseGateError("candidate_commit differs from repository HEAD")
    status_result = _git(root, ("status", "--porcelain=v1", "--untracked-files=all"))
    assert isinstance(status_result.stdout, str)
    if status_result.stdout.strip():
        raise ReleaseGateError("exact candidate worktree is not clean")
    parent_result = _git(root, ("rev-parse", f"{candidate}^"), check=False)
    if parent_result.returncode != 0:
        raise ReleaseGateError("exact candidate must have a resolvable first parent")
    assert isinstance(parent_result.stdout, str)
    actual_parent = _commit(parent_result.stdout.strip(), name="candidate first parent")
    if parent != actual_parent:
        raise ReleaseGateError("declared parent_commit differs from candidate first parent")
    gates_spec = _mapping(spec.get("gates"), name="gates")
    if set(gates_spec) != set(EXACT_GATE_NAMES):
        raise ReleaseGateError("exact candidate attestation must include every release gate")
    gates: dict[str, Any] = {}
    for name in EXACT_GATE_NAMES:
        gate = _mapping(gates_spec[name], name=f"gates.{name}")
        if set(gate) != {"status", "path", "expected_sha256"} or gate.get("status") != "pass":
            raise ReleaseGateError(f"gates.{name} must be a pinned pass")
        reference, payload = _evidence_snapshot(root, gate, name=f"gates.{name}")
        evidence = _strict_json_bytes(payload, name=f"gates.{name} evidence")
        _validate_exact_gate_evidence(
            root,
            gate_name=name,
            evidence=evidence,
            candidate=candidate,
            parent=parent,
            policy_reference=policy_reference,
        )
        gates[name] = {"status": "pass", "evidence": reference}
    report_path = f".audit/release-attestations/{candidate}/final_release_gate_report.json"
    return _self_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "record_kind": REPORT_KIND,
            "mode": mode,
            "status": "pass",
            "created_at_utc": _timestamp(spec.get("created_at_utc")),
            "repository": {
                "parent_commit": parent,
                "content_commit": candidate,
                "worktree_clean": True,
            },
            "policy": policy_reference,
            "gates": gates,
            "external_candidate_attestation": {
                "status": "complete",
                "expected_root": ".audit/release-attestations/<candidate_commit>/",
                "report_path": report_path,
            },
        }
    )


def validate_report(
    report_path: str | Path, *, repository_root: str | Path | None = None
) -> dict[str, Any]:
    """Validate self-hash, exact keys, and mode-specific release semantics."""

    try:
        value = _strict_json(report_path, name="final release gate report")
        _verify_self_hash(value, name="final release gate report")
        expected = {
            "schema_version",
            "record_kind",
            "mode",
            "status",
            "created_at_utc",
            "repository",
            "policy",
            "gates",
            "external_candidate_attestation",
            "record_sha256",
        }
        if set(value) != expected:
            raise ReleaseGateError("final release gate report keys differ")
        if value.get("schema_version") != SCHEMA_VERSION or value.get("record_kind") != REPORT_KIND:
            raise ReleaseGateError("final release gate report version or kind differs")
        _timestamp(value.get("created_at_utc"))
        repository = _mapping(value.get("repository"), name="repository")
        if set(repository) != {"parent_commit", "content_commit", "worktree_clean"}:
            raise ReleaseGateError("repository binding keys differ")
        _commit(repository.get("parent_commit"), name="repository.parent_commit")
        content = repository.get("content_commit")
        if content is not None:
            _commit(content, name="repository.content_commit")
        if not isinstance(repository.get("worktree_clean"), bool):
            raise ReleaseGateError("repository.worktree_clean must be boolean")
        policy_ref = _file_reference_shape(value.get("policy"), name="policy")
        gates = _mapping(value.get("gates"), name="gates")
        mode = value.get("mode")
        external = _mapping(
            value.get("external_candidate_attestation"), name="external_candidate_attestation"
        )
        if set(external) != {"status", "expected_root", "report_path"}:
            raise ReleaseGateError("external_candidate_attestation keys differ")
        if external.get("expected_root") != ".audit/release-attestations/<candidate_commit>/":
            raise ReleaseGateError("external candidate attestation root differs")
        if mode == "tracked_precommit_report":
            if set(gates) != set(TRACKED_GATE_NAMES):
                raise ReleaseGateError("tracked precommit release gate set differs")
            for name, raw_gate in gates.items():
                gate = _mapping(raw_gate, name=f"gates.{name}")
                if set(gate) != {"status", "evidence"}:
                    raise ReleaseGateError(f"gates.{name} keys differ")
                if gate.get("status") != "not_run" or gate.get("evidence") is not None:
                    raise ReleaseGateError(f"gates.{name} must be not_run with null evidence")
            if (
                value.get("status") != "pending"
                or content is not None
                or repository.get("worktree_clean") is not False
                or external.get("status") != "pending"
                or external.get("report_path") is not None
            ):
                raise ReleaseGateError("tracked precommit report overstates release completion")
        elif mode == "exact_candidate_attestation":
            if set(gates) != set(EXACT_GATE_NAMES):
                raise ReleaseGateError("exact candidate release gate set differs")
            candidate = _commit(content, name="repository.content_commit")
            for name, raw_gate in gates.items():
                gate = _mapping(raw_gate, name=f"gates.{name}")
                if set(gate) != {"status", "evidence"}:
                    raise ReleaseGateError(f"gates.{name} keys differ")
                if gate.get("status") != "pass":
                    raise ReleaseGateError(f"gates.{name} must pass")
                _file_reference_shape(gate.get("evidence"), name=f"gates.{name}.evidence")
            expected_path = (
                f".audit/release-attestations/{candidate}/final_release_gate_report.json"
            )
            if (
                value.get("status") != "pass"
                or repository.get("worktree_clean") is not True
                or external.get("status") != "complete"
                or external.get("report_path") != expected_path
            ):
                raise ReleaseGateError("exact candidate attestation is incomplete")
        else:
            raise ReleaseGateError("release gate mode is invalid")
        if mode == "exact_candidate_attestation" and repository_root is None:
            raise ReleaseGateError(
                "exact candidate attestation validation requires repository_root"
            )
        if repository_root is not None:
            root = _repository(repository_root)
            if mode == "tracked_precommit_report":
                observed_ref = _tracked_report_policy_reference(
                    root,
                    report_path,
                    _commit(
                        repository.get("parent_commit"),
                        name="repository.parent_commit",
                    ),
                    policy_ref,
                )
            else:
                observed_ref = _git_blob_reference(
                    root,
                    _commit(content, name="repository.content_commit"),
                    policy_ref,
                    name="policy",
                )
            if observed_ref != policy_ref:
                raise ReleaseGateError("policy reference size differs")
            if mode == "exact_candidate_attestation":
                assert content is not None
                parent_result = _git(root, ("rev-parse", f"{content}^"), check=False)
                if parent_result.returncode != 0:
                    raise ReleaseGateError("exact candidate first parent cannot be resolved")
                assert isinstance(parent_result.stdout, str)
                actual_parent = _commit(parent_result.stdout.strip(), name="candidate first parent")
                if repository.get("parent_commit") != actual_parent:
                    raise ReleaseGateError(
                        "exact candidate report parent differs from candidate first parent"
                    )
                for name, raw_gate in gates.items():
                    gate = _mapping(raw_gate, name=f"gates.{name}")
                    reference = _mapping(gate.get("evidence"), name=f"gates.{name}.evidence")
                    observed, payload = _evidence_snapshot(
                        root,
                        {
                            "path": reference.get("path"),
                            "expected_sha256": reference.get("file_sha256"),
                        },
                        name=f"gates.{name}",
                    )
                    if observed != reference:
                        raise ReleaseGateError(f"gates.{name} evidence size differs")
                    evidence = _strict_json_bytes(payload, name=f"gates.{name} evidence")
                    _validate_exact_gate_evidence(
                        root,
                        gate_name=name,
                        evidence=evidence,
                        candidate=content,
                        parent=actual_parent,
                        policy_reference=policy_ref,
                    )
        return {
            "valid": True,
            "mode": mode,
            "status": value["status"],
            "record_sha256": value["record_sha256"],
            "errors": [],
        }
    except Exception as exc:
        return {"valid": False, "errors": [str(exc)]}


def _write_and_status(record: dict[str, Any], args: argparse.Namespace) -> int:
    root = _repository(args.repository_root)
    _write_new(record, args.output, root=root)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] in {"pass", "pending"} else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    index_scan = subparsers.add_parser("scan-index")
    index_scan.add_argument("--repository-root", type=Path, required=True)
    index_scan.add_argument("--policy", type=Path, required=True)
    index_scan.add_argument("--created-at-utc", required=True)
    index_scan.add_argument("--output", type=Path, required=True)
    scan = subparsers.add_parser("scan-repository")
    scan.add_argument("--repository-root", type=Path, required=True)
    scan.add_argument("--candidate-commit", required=True)
    scan.add_argument("--policy", type=Path, required=True)
    scan.add_argument("--created-at-utc", required=True)
    scan.add_argument("--output", type=Path, required=True)
    secret = subparsers.add_parser("validate-secret-scan")
    secret.add_argument("--repository-root", type=Path, required=True)
    secret.add_argument("--report", type=Path, required=True)
    secret.add_argument("--candidate-commit", required=True)
    secret.add_argument("--policy", type=Path, required=True)
    secret.add_argument("--config", type=Path, required=True)
    secret.add_argument("--ignore", type=Path, required=True)
    secret.add_argument("--gitleaks-version", required=True)
    secret.add_argument("--gitleaks-exit-code", type=int, required=True)
    secret.add_argument("--created-at-utc", required=True)
    secret.add_argument("--output", type=Path, required=True)
    licenses = subparsers.add_parser("audit-licenses")
    licenses.add_argument("--repository-root", type=Path, required=True)
    licenses.add_argument("--inventory", type=Path, required=True)
    licenses.add_argument("--candidate-commit", required=True)
    licenses.add_argument("--policy", type=Path, required=True)
    licenses.add_argument("--pip-licenses-version", required=True)
    licenses.add_argument("--created-at-utc", required=True)
    licenses.add_argument("--output", type=Path, required=True)
    ci_bundle = subparsers.add_parser("validate-ci-bundle")
    ci_bundle.add_argument("--repository-root", type=Path, required=True)
    ci_bundle.add_argument("--candidate-commit", required=True)
    ci_bundle.add_argument("--policy", type=Path, required=True)
    ci_bundle.add_argument("--evidence-root", type=Path, required=True)
    ci_bundle.add_argument("--created-at-utc", required=True)
    ci_bundle.add_argument("--output", type=Path, required=True)
    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--repository-root", type=Path, required=True)
    assemble.add_argument("--policy", type=Path, required=True)
    assemble.add_argument(
        "--mode",
        choices=("tracked_precommit_report", "exact_candidate_attestation"),
        required=True,
    )
    assemble.add_argument("--created-at-utc", required=True)
    assemble.add_argument("--spec", type=Path)
    assemble.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--report", type=Path, required=True)
    validate.add_argument("--repository-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "scan-index":
            record = scan_index(
                repository_root=args.repository_root,
                policy_path=args.policy,
                created_at_utc=args.created_at_utc,
            )
            return _write_and_status(record, args)
        if args.command == "scan-repository":
            record = scan_repository(
                repository_root=args.repository_root,
                candidate_commit=args.candidate_commit,
                policy_path=args.policy,
                created_at_utc=args.created_at_utc,
            )
            return _write_and_status(record, args)
        if args.command == "validate-secret-scan":
            record = validate_secret_scan(
                repository_root=args.repository_root,
                report_path=args.report,
                candidate_commit=args.candidate_commit,
                policy_path=args.policy,
                config_path=args.config,
                ignore_path=args.ignore,
                gitleaks_version=args.gitleaks_version,
                gitleaks_exit_code=args.gitleaks_exit_code,
                created_at_utc=args.created_at_utc,
            )
            return _write_and_status(record, args)
        if args.command == "audit-licenses":
            record = audit_licenses(
                repository_root=args.repository_root,
                inventory_path=args.inventory,
                candidate_commit=args.candidate_commit,
                policy_path=args.policy,
                pip_licenses_version=args.pip_licenses_version,
                created_at_utc=args.created_at_utc,
            )
            return _write_and_status(record, args)
        if args.command == "validate-ci-bundle":
            record = validate_ci_gate_bundle(
                repository_root=args.repository_root,
                candidate_commit=args.candidate_commit,
                policy_path=args.policy,
                evidence_root=args.evidence_root,
                created_at_utc=args.created_at_utc,
            )
            return _write_and_status(record, args)
        if args.command == "assemble":
            record = assemble_report(
                repository_root=args.repository_root,
                policy_path=args.policy,
                mode=args.mode,
                created_at_utc=args.created_at_utc,
                spec_path=args.spec,
            )
            return _write_and_status(record, args)
        report = validate_report(args.report, repository_root=args.repository_root)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["valid"] is True else 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
