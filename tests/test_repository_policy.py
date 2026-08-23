"""Repository-level evidence and CI safety policies."""

from __future__ import annotations

import fnmatch
from pathlib import Path


def _pattern_matches(pattern: str, relative_path: str) -> bool:
    pattern = pattern.strip().replace("\\", "/")
    relative_path = relative_path.replace("\\", "/")
    if pattern.startswith("/"):
        pattern = pattern[1:]
    if pattern.endswith("/"):
        return relative_path.startswith(pattern)
    return fnmatch.fnmatchcase(relative_path, pattern)


def _gitignore_decision(gitignore: Path, relative_path: str) -> bool:
    """Evaluate the small subset of gitignore syntax used by this repository.

    Processing in order and supporting negation matters: a later exception must
    not be able to silently re-include raw sensor evidence.
    """

    ignored = False
    for raw_line in gitignore.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        pattern = line[1:] if negated else line
        if _pattern_matches(pattern, relative_path):
            ignored = not negated
    return ignored


def test_raw_data_paths_are_ignored(repository_root: Path) -> None:
    gitignore = repository_root / ".gitignore"
    assert gitignore.is_file(), "Stage 2 requires an explicit raw-data ignore policy"

    sentinels = (
        "data/raw/inclusivehar-v4/archive.zip",
        "data/raw/uci-har/inertial_signals.csv",
        "data/raw/nested/Subject01/Trial01.csv",
    )
    for sentinel in sentinels:
        assert _gitignore_decision(gitignore, sentinel), (
            f"raw evidence path is not ignored: {sentinel}"
        )


def test_raw_data_policy_does_not_ignore_small_manifest_metadata(
    repository_root: Path,
) -> None:
    gitignore = repository_root / ".gitignore"
    assert not _gitignore_decision(gitignore, "manifests/datasets/inclusivehar-v4.json"), (
        "lawful checksum/provenance manifests must remain committable"
    )


def test_ci_declares_dataset_offline_mode(repository_root: Path) -> None:
    workflow = repository_root / ".github" / "workflows" / "ci.yml"
    text = workflow.read_text(encoding="utf-8-sig")

    required_assignments = (
        'INCLUSIVE_SHIFT_HAR_OFFLINE: "1"',
        'HF_DATASETS_OFFLINE: "1"',
        'TRANSFORMERS_OFFLINE: "1"',
        "WANDB_MODE: offline",
    )
    for assignment in required_assignments:
        assert assignment in text

    lowered = text.lower()
    forbidden_dataset_download_commands = (
        "wget ",
        "curl ",
        "inclusive-shift-har download",
        "python -m inclusive_shift_har.data.acquisition",
    )
    for command in forbidden_dataset_download_commands:
        assert command not in lowered


def test_ci_has_read_only_default_permissions(repository_root: Path) -> None:
    text = (repository_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8-sig")
    assert "permissions:" in text
    assert "contents: read" in text


def test_synthetic_validation_files_remain_tiny(repository_root: Path) -> None:
    validation_roots = (
        repository_root / "tests",
        repository_root / "configs" / "schema",
    )
    files = tuple(
        path
        for root in validation_roots
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )

    assert files
    assert all(path.stat().st_size < 256 * 1024 for path in files)
    assert sum(path.stat().st_size for path in files) < 2 * 1024 * 1024


def test_raw_dataset_payload_location_is_explicitly_git_ignored(
    repository_root: Path,
) -> None:
    gitignore_lines = {
        line.strip()
        for line in (repository_root / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "/data/raw/" in gitignore_lines

    raw_root = repository_root / "data" / "raw"
    if not raw_root.exists():
        return

    raw_files = [path for path in raw_root.rglob("*") if path.is_file()]
    assert all(path.resolve().is_relative_to(raw_root.resolve()) for path in raw_files)
