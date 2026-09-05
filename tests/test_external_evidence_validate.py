from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from inclusive_shift_har.evaluation.external_statistics import (
    ParticipantMetricInputs,
    seed_evidence,
)
from inclusive_shift_har.experiments.external_evidence_validate import (
    _metric_evidence_errors,
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
    assert validation["status"] == "PROVISIONAL"
    assert validation["integrity_passed"] is True
    assert validation["publication_evidence_ready"] is False
    assert validation["scientific_contract_passed"] is False


def test_validator_recomputes_all_primary_metrics_and_requires_every_seed(tmp_path: Path) -> None:
    data = ParticipantMetricInputs(
        dataset_id="fog_star_v3",
        class_names=("mobility", "sitting", "standing"),
        labels=np.array([0, 1, 2, 0, 1, 2]),
        participant_ids=np.array(["p1", "p1", "p1", "p2", "p2", "p2"]),
    )
    probabilities = {
        seed: {"XGBoost-6ch": np.eye(3)[data.labels] * 0.9 + 0.1 / 3} for seed in (11, 23, 47)
    }
    primary, predictions = seed_evidence(data, probabilities)
    path = tmp_path / "predictions.npz"
    np.savez_compressed(
        path,
        labels=data.labels,
        participant_ids=data.participant_ids,
        **{f"probability__{name}": values for name, values in predictions.items()},  # type: ignore[arg-type]
    )
    result = {"dataset": {"dataset_id": data.dataset_id}, "primary_seed_averaged": primary}
    assert not _metric_evidence_errors(result, path)
    primary["methods"]["XGBoost-6ch"]["mean_participant_macro_f1"] = 0.5
    assert "differ" in _metric_evidence_errors(result, path)[0]
    primary["methods"]["XGBoost-6ch"]["mean_participant_macro_f1"] = 1.0
    incomplete = tmp_path / "incomplete.npz"
    np.savez_compressed(incomplete, labels=data.labels, participant_ids=data.participant_ids)
    assert "reconstruction failed" in _metric_evidence_errors(result, incomplete)[0]
