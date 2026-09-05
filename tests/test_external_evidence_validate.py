from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from inclusive_shift_har.experiments.external_evidence_validate import (
    validate_run_directory,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_external_evidence_validator_accepts_a_complete_create_only_package(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src" / "example.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    probabilities = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]], dtype=np.float64)
    prediction = run / "predictions.npz"
    np.savez_compressed(
        prediction,
        labels=np.array([0, 1], dtype=np.int64),
        participant_ids=np.array(["p1", "p2"]),
        probability__method=probabilities,
    )
    files = {"src/example.py": sha256_file(source)}
    manifest = {
        "file_count": 1,
        "files": files,
        "manifest_sha256": canonical_json_sha256(files),
        "captured_at": "2026-09-05T00:00:00+00:00",
    }
    _write_json(
        run / "data_audit.json",
        {
            "dataset": {"window_count": 2},
            "source_input_manifest": manifest,
            "source_receipts": [
                {
                    "computed_sha256": "0" * 64,
                    "received_size_bytes": 1,
                    "declared_digest_verified": True,
                    "raw_local_mirror": False,
                }
            ],
        },
    )
    result: dict[str, object] = {
        "dataset": {"window_count": 2},
        "reports": {"method": {"sample_count": 2}},
        "fold_records": [
            {
                "training_participants": ["p1"],
                "evaluation_participants": ["p2"],
                "outer_evaluation_labels_used_before_predictions_fixed": False,
            }
        ],
        "prediction_artifact": {
            "path": prediction.name,
            "sha256": sha256_file(prediction),
        },
        "source_input_manifest": manifest,
        "claim_policy": {
            "confirmatory_claim_allowed": False,
            "state_of_the_art_claim_allowed": False,
        },
    }
    result["result_payload_sha256_before_serialization"] = canonical_json_sha256(result)
    _write_json(run / "result.json", result)
    validation = validate_run_directory(run, tmp_path)
    assert validation["status"] == "VALIDATED"
    assert validation["integrity_passed"] is True
    assert validation["publication_evidence_ready"] is True
