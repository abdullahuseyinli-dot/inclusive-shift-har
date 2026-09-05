"""Lightweight create-only research provenance without importing model runtimes."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file


def _write_json_create_only(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _git_state(repository_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "worktree_dirty": bool(status), "status_entries": status}


def _source_input_manifest(repository_root: Path) -> dict[str, Any]:
    """Hash the executed package and the unchanged declared source-input scope."""
    if Path(__file__).resolve().parents[3] != repository_root.resolve():
        raise ValueError(
            "executed research package does not belong to the declared repository root"
        )
    candidates = list((repository_root / "src").rglob("*.py"))
    candidates.extend((repository_root / "configs").rglob("*.yaml"))
    candidates.extend((repository_root / "configs").rglob("*.json"))
    candidates.extend((repository_root / "docs/research").glob("*.md"))
    candidates.extend(
        repository_root / relative
        for relative in (
            "configs/datasets/external_har_portfolio_v1.yaml",
            "configs/experiments/cross_dataset_har_rnd_v1.yaml",
            "docs/research/CROSS_DATASET_HAR_RND_V1_PROTOCOL.md",
            "docs/research/EXTERNAL_HAR_PHYSICAL_GRID_V2_CORRECTION.md",
            "configs/protocols/external_har_physical_grid_v2.yaml",
            "configs/datasets/evidence_roles_20260905_v2.yaml",
            "pyproject.toml",
            "uv.lock",
            "requirements/external-har-research.in",
            "requirements/external-har-research.lock",
        )
    )
    files = {
        path.relative_to(repository_root).as_posix(): sha256_file(path)
        for path in sorted(set(candidates))
        if path.is_file()
    }
    return {
        "protocol_id": "external-har-session-grid-v3",
        "file_count": len(files),
        "files": files,
        "manifest_sha256": canonical_json_sha256(files),
        "captured_at": datetime.now(UTC).isoformat(),
    }
