from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from inclusive_shift_har.artifacts import publication_checkpoint
from inclusive_shift_har.artifacts.research_provenance import (
    _source_input_manifest,
    _write_json_create_only,
)


def test_command_recorder_does_not_import_torch_or_sklearn() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import inclusive_shift_har.artifacts.publication_checkpoint; "
            "assert 'torch' not in sys.modules; assert 'sklearn' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_readonly_research_audits_do_not_import_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from inclusive_shift_har.experiments import external_evidence_validate, "
            "publication_table, publication_diagnostics, observable_context_parity; "
            "assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_lightweight_manifest_keeps_executed_package_binding(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not belong"):
        _source_input_manifest(tmp_path)
    root = Path(__file__).resolve().parents[1]
    manifest = _source_input_manifest(root)
    assert manifest["protocol_id"] == "external-har-session-grid-v3"
    assert "src/inclusive_shift_har/artifacts/research_provenance.py" in manifest["files"]
    assert manifest["file_count"] == len(manifest["files"])


def test_create_only_writer_cannot_replace_prior_evidence(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    _write_json_create_only(path, {"value": 1})
    with pytest.raises(FileExistsError):
        _write_json_create_only(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}


@pytest.mark.parametrize("drift", [False, True])
def test_quality_gate_cannot_pass_with_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: bool
) -> None:
    manifests = iter(
        [
            {"manifest_sha256": "a" * 64},
            {"manifest_sha256": ("b" if drift else "a") * 64},
        ]
    )
    monkeypatch.setattr(
        publication_checkpoint, "_source_input_manifest", lambda _root: next(manifests)
    )

    def command(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "exit_code": 0,
            "record_sha256": "c" * 64,
            "source_input_manifest": {"manifest_sha256": "a" * 64},
        }

    monkeypatch.setattr(publication_checkpoint, "record_command", command)
    result = publication_checkpoint.quality_gates(
        tmp_path / "gates", tmp_path, Path("uv"), workers=1
    )
    assert (result["status"] == "PASS") is not drift
    assert result["source_unchanged_across_all_gate_launches_and_completion"] is not drift
