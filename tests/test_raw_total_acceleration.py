from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import inclusive_shift_har.experiments.raw_total_acceleration as raw_total_module
from inclusive_shift_har.data.windowing import WindowRecord
from inclusive_shift_har.evaluation._strict_config import StrictConfigError
from inclusive_shift_har.evaluation.raw_total_reporting import (
    RawTotalAggregationError,
    aggregate_raw_total_acceleration_sensitivity,
)
from inclusive_shift_har.experiments.raw_total_acceleration import (
    EVIDENCE_STATUS,
    MODEL_IDS,
    RAW_TOTAL_CHANNELS,
    SEED_ORDER,
    SOURCE_VALIDATION_PARTICIPANTS,
    TARGET_PARTICIPANTS,
    TRACK_ROLE,
    RawTotalSensitivityError,
    load_raw_total_sensitivity_config,
    materialize_signal_definition_windows,
    run_raw_total_acceleration_sensitivity,
)
from inclusive_shift_har.manifests.canonical import (
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)

CONFIG_PATH = Path("configs/experiments/raw_total_acceleration_sensitivity_v1.yaml")


def _write_hashed(path: Path, payload: dict[str, Any]) -> None:
    payload["record_sha256"] = canonical_json_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _window(*, partition: str = "source_train", label: str = "Walking") -> WindowRecord:
    return WindowRecord(
        window_id="w_test_raw_total",
        raw_interval_id="test:rows:1-128",
        released_run_id="released_run_test",
        subject_id="1",
        activity_label=label,
        canonical_labels={"functional_core": "mobility"},
        partition=partition,
        window_ordinal=0,
        start_row_inclusive=1,
        end_row_inclusive=128,
        length_samples=128,
        stride_samples=128,
        trial_id=None,
        trial_status="unrecoverable",
    )


def _raw_csv(path: Path, *, row_label: str = "Walking") -> None:
    header = [*RAW_TOTAL_CHANNELS, "label", "UserID"]
    rows = [",".join(header)]
    for row in range(128):
        values = [str(row + channel / 10.0) for channel in range(6)]
        rows.append(",".join([*values, row_label, "1"]))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_strict_raw_total_config_locks_two_architectures_and_five_seeds(tmp_path: Path) -> None:
    config = load_raw_total_sensitivity_config(CONFIG_PATH)

    assert tuple(model.model_id for model in config.models) == MODEL_IDS
    assert tuple(config.execution["required_seed_order"]) == SEED_ORDER
    assert config.signal_interface["uci_body_acceleration_equivalent"] is False
    assert config.signal_interface["trial_safe"] is False
    assert config.opening_context["unlock_or_new_opening_invocation_allowed"] is False

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        CONFIG_PATH.read_text(encoding="utf-8") + '\nstatus: "duplicate"\n',
        encoding="utf-8",
    )
    with pytest.raises(StrictConfigError, match="duplicate key"):
        load_raw_total_sensitivity_config(duplicate)


def test_raw_total_materializer_reads_only_declared_channels_and_preserves_identity(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "synthetic.csv"
    _raw_csv(csv_path)

    batch = materialize_signal_definition_windows(
        csv_path,
        (_window(),),
        expected_source_sha256=sha256_file(csv_path),
        channels=RAW_TOTAL_CHANNELS,
        ontology_track="functional_core",
        class_names=("mobility", "sitting", "standing"),
        allowed_partitions=frozenset({"source_train"}),
    )

    assert batch.signals.shape == (1, 128, 6)
    assert batch.signals[0, 1].tolist() == pytest.approx([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    assert batch.window_ids == ("w_test_raw_total",)
    assert batch.participant_ids == ("1",)
    assert batch.labels.tolist() == [0]

    with pytest.raises(PermissionError, match="consumed opening 1"):
        materialize_signal_definition_windows(
            csv_path,
            (_window(partition="target_sealed"),),
            expected_source_sha256=sha256_file(csv_path),
            channels=RAW_TOTAL_CHANNELS,
            ontology_track="functional_core",
            class_names=("mobility", "sitting", "standing"),
            allowed_partitions=frozenset({"target_sealed"}),
        )


def test_cuda_gate_precedes_configuration_raw_and_target_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RawTotalSensitivityError, match="before configuration"):
        run_raw_total_acceleration_sensitivity(
            config_path=tmp_path / "missing.yaml",
            repository_root=tmp_path,
            raw_csv_path=tmp_path / "missing.csv",
            expected_code_commit="0" * 40,
            created_at_utc="2099-01-01T00:00:00Z",
            device=torch.device("cuda"),
        )
    assert list(tmp_path.iterdir()) == []


def _participant_report(values: dict[str, float]) -> dict[str, Any]:
    array = np.asarray(list(values.values()), dtype=np.float64)
    return {
        "schema_version": "1.0.0",
        "sample_count": len(values),
        "participant_count": len(values),
        "participants": [
            {
                "participant_id": participant,
                "window_count": 1,
                "macro_f1": metric,
                "balanced_accuracy": metric,
                "cohort": None,
            }
            for participant, metric in values.items()
        ],
        "primary": {
            "mean_participant_macro_f1": float(array.mean()),
            "worst_participant_macro_f1": float(array.min()),
            "lower_decile_participant_macro_f1": float(np.quantile(array, 0.1, method="linear")),
        },
        "calibration": {
            "negative_log_likelihood": 0.5,
            "multiclass_brier_score": 0.4,
            "ece": 0.1,
        },
    }


def _result_entry(
    tmp_path: Path,
    *,
    model_id: str,
    seed: int,
    cohort: str,
    values: dict[str, float],
) -> dict[str, Any]:
    stem = f"{model_id}--{seed}--{cohort}"
    prediction = tmp_path / "predictions" / f"{stem}.npz"
    prediction.parent.mkdir(parents=True, exist_ok=True)
    prediction.write_bytes(stem.encode("utf-8"))
    record: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "postconfirmatory_raw_total_acceleration_evaluation",
        "status": "complete_create_only",
        "evidence_status": EVIDENCE_STATUS,
        "track_role": TRACK_ROLE,
        "primary_claim_eligible": False,
        "model_selection_use": False,
        "model_id": model_id,
        "seed": seed,
        "cohort": cohort,
        "target_tuning": False,
        "target_calibration_or_refit": False,
        "new_opening_or_unlock_invoked": False,
        "participant_level_report": _participant_report(values),
    }
    record_path = tmp_path / "records" / f"{stem}.json"
    _write_hashed(record_path, record)
    return {
        "model_id": model_id,
        "seed": seed,
        "record_path": record_path.relative_to(tmp_path).as_posix(),
        "record_file_sha256": sha256_file(record_path),
        "record_sha256": record["record_sha256"],
        "prediction_path": prediction.relative_to(tmp_path).as_posix(),
        "prediction_sha256": sha256_file(prediction),
    }


def _primary_entry(
    tmp_path: Path,
    *,
    model_id: str,
    seed: int,
    values: dict[str, float],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "locked_target_per_seed_result",
        "status": "complete_create_only",
        "evidence_status": "locked_confirmatory_target_opening_1",
        "model_id": model_id,
        "seed": seed,
        "target_information_used_for_model_selection": False,
        "participant_level_report": _participant_report(values),
    }
    path = tmp_path / "primary" / f"{model_id}--{seed}.json"
    _write_hashed(path, record)
    return {
        "model_id": model_id,
        "seed": seed,
        "record_path": path.relative_to(tmp_path).as_posix(),
        "record_file_sha256": sha256_file(path),
        "record_sha256": record["record_sha256"],
        "prediction_sha256": "a" * 64,
    }


def _aggregation_fixture(tmp_path: Path) -> Path:
    config_path = tmp_path / "raw-total-config.yaml"
    config_path.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    source_results: list[dict[str, Any]] = []
    target_results: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for model_index, model_id in enumerate(MODEL_IDS):
        source_values = {
            participant: 0.70 + index * 0.02 + model_index * 0.01
            for index, participant in enumerate(SOURCE_VALIDATION_PARTICIPANTS)
        }
        raw_values = {
            participant: 0.60 + index * 0.01 + model_index * 0.01
            for index, participant in enumerate(TARGET_PARTICIPANTS)
        }
        primary_values = {participant: value - 0.10 for participant, value in raw_values.items()}
        for seed in SEED_ORDER:
            source_results.append(
                _result_entry(
                    tmp_path,
                    model_id=model_id,
                    seed=seed,
                    cohort="source_validation",
                    values=source_values,
                )
            )
            target_results.append(
                _result_entry(
                    tmp_path,
                    model_id=model_id,
                    seed=seed,
                    cohort="target_consumed_opening_1",
                    values=raw_values,
                )
            )
            references.append(
                _primary_entry(
                    tmp_path,
                    model_id=model_id,
                    seed=seed,
                    values=primary_values,
                )
            )
    normalization: dict[str, Any] = {"record_kind": "normalization"}
    normalization_path = tmp_path / "normalization.json"
    _write_hashed(normalization_path, normalization)
    source_lock: dict[str, Any] = {"record_kind": "source-lock"}
    source_lock_path = tmp_path / "source-lock.json"
    _write_hashed(source_lock_path, source_lock)
    index: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "postconfirmatory_raw_total_acceleration_sensitivity_index",
        "status": "complete_create_only",
        "evidence_status": EVIDENCE_STATUS,
        "track_role": TRACK_ROLE,
        "primary_claim_eligible": False,
        "model_selection_use": False,
        "required_model_order": list(MODEL_IDS),
        "required_seed_order": list(SEED_ORDER),
        "expected_model_seed_count": 10,
        "source_result_count": 10,
        "target_result_count": 10,
        "primary_reference_count": 10,
        "failure_count": 0,
        "source_stage_completed_before_target_access": True,
        "target_signals_or_labels_accessed": True,
        "target_tuning": False,
        "target_calibration_or_refit": False,
        "new_opening_or_unlock_invoked": False,
        "experiment_config": {
            "path": config_path.relative_to(tmp_path).as_posix(),
            "file_sha256": sha256_file(config_path),
        },
        "normalization": {
            "path": normalization_path.relative_to(tmp_path).as_posix(),
            "file_sha256": sha256_file(normalization_path),
            "record_sha256": normalization["record_sha256"],
        },
        "source_stage_lock": {
            "path": source_lock_path.relative_to(tmp_path).as_posix(),
            "file_sha256": sha256_file(source_lock_path),
            "record_sha256": source_lock["record_sha256"],
        },
        "source_results": source_results,
        "target_results": target_results,
        "primary_target_references": references,
        "failures": [],
    }
    index_path = tmp_path / "sensitivity-index.json"
    _write_hashed(index_path, index)
    return index_path


def _aggregate(tmp_path: Path, index_path: Path, *, prefix: str = "aggregate") -> dict[str, Any]:
    return aggregate_raw_total_acceleration_sensitivity(
        index_path,
        artifact_root=tmp_path,
        destination=f"outputs/{prefix}.json",
        csv_destination=f"outputs/{prefix}.csv",
        markdown_destination=f"outputs/{prefix}.md",
        created_at_utc="2099-01-01T00:00:00Z",
    )


def test_raw_total_aggregator_builds_paired_participant_sensitivity(tmp_path: Path) -> None:
    index_path = _aggregation_fixture(tmp_path)

    result = _aggregate(tmp_path, index_path)

    assert len(result["model_rows"]) == 2
    for row in result["model_rows"]:
        assert row["raw_minus_primary_target"]["mean_macro_f1"] == pytest.approx(0.1)
        assert row["paired_bootstrap_mean_delta"]["lower"] == pytest.approx(0.1)
        assert row["paired_bootstrap_mean_delta"]["upper"] == pytest.approx(0.1)
    assert len(result["participant_target_rows"]) == 20
    assert result["primary_claim_eligible"] is False
    assert result["uci_body_acceleration_equivalent"] is False
    assert result["trial_safe"] is False
    published = load_json_strict(tmp_path / "outputs" / "aggregate.json")
    assert isinstance(published, dict)
    body = dict(published)
    claimed = body.pop("record_sha256")
    assert claimed == canonical_json_sha256(body)
    markdown = (tmp_path / "outputs" / "aggregate.md").read_text(encoding="utf-8")
    assert "post-confirmatory exploratory" in markdown
    assert "not trial-safe" in markdown

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _aggregate(tmp_path, index_path)


def test_raw_total_aggregator_rejects_mutated_indexed_record(tmp_path: Path) -> None:
    index_path = _aggregation_fixture(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    record_path = tmp_path / index["target_results"][0]["record_path"]
    record_path.write_bytes(record_path.read_bytes() + b"mutation")

    with pytest.raises(RawTotalAggregationError, match="file hash changed"):
        _aggregate(tmp_path, index_path, prefix="mutated")


@pytest.mark.parametrize(
    ("status", "expected_exit"),
    [("complete_create_only", 0), ("target_stage_failed_preserved_partial", 2)],
)
def test_raw_total_cli_exit_status_tracks_completion(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    expected_exit: int,
) -> None:
    monkeypatch.setattr(
        raw_total_module,
        "run_raw_total_acceleration_sensitivity",
        lambda **_kwargs: {"status": status},
    )
    exit_code = raw_total_module.main(
        [
            "--config",
            "config.yaml",
            "--repository-root",
            ".",
            "--raw-csv",
            "raw.csv",
            "--code-commit",
            "a" * 40,
            "--created-at-utc",
            "2099-01-01T00:00:00Z",
        ]
    )
    assert exit_code == expected_exit
