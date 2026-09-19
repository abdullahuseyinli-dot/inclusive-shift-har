from __future__ import annotations

import json
from pathlib import Path

import pytest

from inclusive_shift_har.experiments.same_attachment_reference_qualification import (
    run_synthetic_qualification,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file


def test_full_synthetic_schedule_is_counted_and_cannot_unlock_physical_fit(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "qualification"
    result = run_synthetic_qualification(root, output)
    assert result["status"] == "pass"
    assert len(result["fit_ledger"]) == 72
    assert len(result["arm_fold_outputs"]) == 54
    assert result["accounting"]["synthetic_optimizer_fits_executed"] == 72
    assert result["accounting"]["human_model_fits_executed"] == 0
    assert result["physical_admission"]["fit_authorized"] is False
    assert all(not row["held_out_labels_used_by_prediction_api"] for row in result["fold_receipts"])
    stored = json.loads((output / "result.json").read_text(encoding="utf-8"))
    expected_hash = stored.pop("record_sha256")
    assert canonical_json_sha256(stored) == expected_hash
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"][0]["sha256"] == sha256_file(output / "result.json")
    assert manifest["record_sha256"] == canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "record_sha256"}
    )


def test_synthetic_qualification_is_create_only(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "qualification"
    output.mkdir()
    marker = output / "preserve.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="create-only"):
        run_synthetic_qualification(root, output)
    assert marker.read_text(encoding="utf-8") == "preserve"
