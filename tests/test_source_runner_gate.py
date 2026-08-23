from __future__ import annotations

import json
from pathlib import Path

import pytest

from inclusive_shift_har.experiments.inclusivehar_source import _load_source_manifest
from inclusive_shift_har.manifests.canonical import canonical_json_sha256


def test_source_runner_rejects_target_bearing_manifest(tmp_path: Path) -> None:
    payload = {
        "source_window_manifest_sha256": "placeholder",
        "target_subject_or_window_records_included": True,
        "target_performance_or_prediction_accessed": False,
        "windows": [],
    }
    without_hash = dict(payload)
    without_hash.pop("source_window_manifest_sha256")
    payload["source_window_manifest_sha256"] = canonical_json_sha256(without_hash)
    path = tmp_path / "source.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PermissionError, match="target-bearing"):
        _load_source_manifest(path)
