from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from inclusive_shift_har.artifacts.release_inventory import (
    GATE_NAMES,
    POSTCONFIRMATORY_TRACKS,
    REQUIRED_ROLES,
    ReleaseEvidenceError,
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


def _write_json(path: Path, value: dict[str, Any]) -> None:
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


def _build_workspace(
    tmp_path: Path,
    *,
    requested_status: str = "draft",
    bad_opening_self_hash: bool = False,
) -> ReleaseWorkspace:
    repository = tmp_path / "repository"
    repository.mkdir(parents=True)
    _git(repository, "init")
    _git(repository, "config", "user.email", "synthetic@example.invalid")
    _git(repository, "config", "user.name", "Synthetic Release Test")
    _git(repository, "config", "core.autocrlf", "false")

    paths: dict[str, str] = {}
    for role in REQUIRED_ROLES:
        relative = (
            f"evidence/{role}.json"
            if role
            in {
                "opening_receipt",
                "locked_target_index",
            }
            else f"evidence/{role}.md"
        )
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if role in {"opening_receipt", "locked_target_index"}:
            record = _self_hashed_record(
                schema_version="1.0.0",
                record_kind=f"synthetic_{role}",
                status="complete",
            )
            if bad_opening_self_hash and role == "opening_receipt":
                record["record_sha256"] = "0" * 64
            _write_json(path, record)
        else:
            path.write_text(f"# Synthetic {role}\n", encoding="utf-8")
        paths[role] = relative
    output_root = repository / "release"
    output_root.mkdir()
    (output_root / ".gitkeep").write_text("", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "synthetic candidate")
    commit_value = _git(repository, "rev-parse", "HEAD")
    commit = commit_value.strip()

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
                    "record_sha256" if role in {"opening_receipt", "locked_target_index"} else None
                ),
            }
        )

    spec_root = tmp_path / "attestations"
    spec_root.mkdir(parents=True)
    gates: dict[str, dict[str, Any]] = {}
    remote: dict[str, Any]
    if requested_status in {"ready", "released"}:
        for gate_name in GATE_NAMES:
            evidence_path = spec_root / "gates" / f"{gate_name}.json"
            if gate_name == "ci":
                evidence = _self_hashed_record(
                    record_kind="ci_run_evidence",
                    status="pass",
                    conclusion="success",
                    candidate_commit=commit,
                    workflow_url="https://github.com/example/inclusive-shift-har/actions/runs/1",
                )
            else:
                evidence = {
                    "record_kind": "synthetic_release_gate",
                    "gate": gate_name,
                    "status": "pass",
                    "candidate_commit": commit,
                }
            _write_json(evidence_path, evidence)
            gates[gate_name] = {
                "status": "pass",
                "evidence_source": "spec_root",
                "evidence_path": f"gates/{gate_name}.json",
                "expected_sha256": _file_sha256(evidence_path),
                "tool_version": "synthetic-1",
                "note": "synthetic gate evidence",
            }
        remote_path = spec_root / "remote.json"
        remote_url = "https://github.com/example/inclusive-shift-har"
        _write_json(
            remote_path,
            _self_hashed_record(
                record_kind="github_remote_evidence",
                status="pass",
                candidate_commit=commit,
                visibility="private",
                remote_url=remote_url,
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


def test_generate_and_validate_draft_with_explicit_not_run_tracks(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path)
    inventory = _generate(workspace)

    assert inventory["schema_version"] == "1.1.0"
    assert inventory["delivery_mode"] == "external_release_asset"
    assert inventory["tracked"] is False
    assert inventory["status"] == "draft"
    assert inventory["repository"]["worktree_clean"] is True
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
    report = validate_release_evidence_inventory_file(
        workspace.output_root / "inventory.json",
        spec_path=workspace.spec_path,
        repository_root=workspace.repository,
        expected_candidate_commit=workspace.commit,
    )
    assert report["valid"] is True, report


def test_ready_requires_and_validates_external_ci_and_remote_attestations(
    tmp_path: Path,
) -> None:
    workspace = _build_workspace(tmp_path, requested_status="ready")
    inventory = _generate(workspace)

    assert inventory["status"] == "ready"
    assert inventory["repository"]["remote_visibility"] == "private"
    assert inventory["repository"]["remote_evidence_scope"] == "spec_root"
    assert inventory["gates"]["ci"]["status"] == "pass"
    assert inventory["gates"]["ci"]["evidence_scope"] == "spec_root"


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
            "preserve one opening",
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


def test_generator_rejects_bad_declared_embedded_self_hash(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path, bad_opening_self_hash=True)

    with pytest.raises(ReleaseEvidenceError, match="embedded self-hash"):
        _generate(workspace)


def test_generator_rejects_onnx_payload_reference(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path)
    workspace.spec["artifacts"][0]["path"] = "evidence/model.onnx"
    _write_json(workspace.spec_path, workspace.spec)

    with pytest.raises(ReleaseEvidenceError, match="prohibited raw/checkpoint extension"):
        _generate(workspace)


def test_generator_rejects_ci_pass_without_pinned_evidence(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path)
    workspace.spec["gates"]["ci"]["status"] = "pass"
    _write_json(workspace.spec_path, workspace.spec)

    with pytest.raises(ReleaseEvidenceError, match="requires pinned evidence"):
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
