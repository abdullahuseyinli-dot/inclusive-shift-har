from __future__ import annotations

import json
from pathlib import Path

import pytest

from inclusive_shift_har.experiments.inclusivehar_source import (
    _load_source_manifest,
    build_parser,
)
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


def test_source_runner_parses_fixed_epoch_channel_ablation() -> None:
    args = build_parser().parse_args(
        [
            "--source-manifest",
            "source.json",
            "--raw-csv",
            "raw.csv",
            "--dataset-manifest",
            "dataset.json",
            "--model",
            "more_har",
            "--seed",
            "11",
            "--run-directory",
            "run",
            "--summary",
            "summary.json",
            "--code-commit",
            "a" * 40,
            "--checkpoint-selection-rule",
            "fixed_last_epoch",
            "--zero-channel-indices",
            "0",
            "1",
            "2",
        ]
    )

    assert args.checkpoint_selection_rule == "fixed_last_epoch"
    assert args.zero_channel_indices == [0, 1, 2]
