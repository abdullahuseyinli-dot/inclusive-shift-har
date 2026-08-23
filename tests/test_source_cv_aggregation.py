from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inclusive_shift_har.evaluation.source_cv import (
    SourceCVAggregationError,
    build_source_cv_aggregate,
    derive_model_identity,
    write_source_cv_exports_new,
)
from inclusive_shift_har.manifests.canonical import (
    canonical_json_sha256,
    load_json_strict,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _source_manifest(path: Path) -> str:
    payload: dict[str, Any] = {
        "schema_version": "test",
        "target_subject_or_window_records_included": False,
        "target_performance_or_prediction_accessed": False,
        "source_cv_folds": [
            {
                "fold_id": "source_cv_01",
                "train_subjects": ["p3", "p4"],
                "validation_subjects": ["p1", "p2"],
            },
            {
                "fold_id": "source_cv_02",
                "train_subjects": ["p1", "p2"],
                "validation_subjects": ["p3", "p4"],
            },
        ],
    }
    payload["source_window_manifest_sha256"] = canonical_json_sha256(payload)
    _write_json(path, payload)
    return str(payload["source_window_manifest_sha256"])


def _configuration(model_name: str = "random_forest") -> dict[str, Any]:
    if model_name != "more_har":
        return {
            "model_name": model_name,
            "num_classes": 3,
            "seed": 11,
            "xgboost_device": "cpu",
        }
    return {
        "model_name": "more_har",
        "num_classes": 3,
        "seed": 11,
        "coral_weight": 0.0,
        "use_augmentation": True,
        "use_content_objective": True,
        "use_realization_factorization": True,
        "use_group_dro": True,
        "zero_channel_indices": [],
    }


def _development_origin(path: Path, configuration: dict[str, Any]) -> None:
    payload: dict[str, Any] = {
        "configuration": configuration,
        "configuration_sha256": canonical_json_sha256(configuration),
        "evidence_status": "source_development_not_confirmatory",
        "target_performance_or_prediction_accessed": False,
        "target_subject_or_window_records_loaded": False,
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    _write_json(path, payload)


def _fold_record(
    path: Path,
    *,
    configuration: dict[str, Any],
    source_manifest_sha256: str,
    fold_id: str,
    train: list[str],
    participant_values: list[tuple[str, int, float, float]],
    nll: float,
    brier: float,
    ece: float,
    tamper_hash: bool = False,
    target_loaded: bool = False,
) -> None:
    macro_values = np.asarray([row[2] for row in participant_values])
    sample_count = sum(row[1] for row in participant_values)
    participants = [
        {
            "participant_id": participant,
            "window_count": windows,
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced_accuracy,
        }
        for participant, windows, macro_f1, balanced_accuracy in participant_values
    ]
    payload: dict[str, Any] = {
        "best_epoch": None,
        "class_names": ["mobility", "sitting", "standing"],
        "code_commit": "a" * 40,
        "configuration": configuration,
        "configuration_sha256": canonical_json_sha256(configuration),
        "evidence_status": "source_development_not_confirmatory",
        "model_name": configuration["model_name"],
        "parameter_count": None,
        "seed": configuration["seed"],
        "source_split_id": fold_id,
        "source_window_manifest_sha256": source_manifest_sha256,
        "target_performance_or_prediction_accessed": False,
        "target_subject_or_window_records_loaded": target_loaded,
        "train_participants": train,
        "validation_participants": [row[0] for row in participant_values],
        "validation_report": {
            "sample_count": sample_count,
            "primary": {
                "mean_participant_macro_f1": float(macro_values.mean()),
                "worst_participant_macro_f1": float(macro_values.min()),
                "lower_decile_participant_macro_f1": float(
                    np.quantile(macro_values, 0.1, method="linear")
                ),
            },
            "participants": participants,
            "calibration": {
                "negative_log_likelihood": nll,
                "multiclass_brier_score": brier,
                "ece": ece,
            },
            "window_level_diagnostics": {"balanced_accuracy": 0.7},
        },
        "validation_window_count": sample_count,
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    if tamper_hash:
        payload["record_sha256"] = "0" * 64
    _write_json(path, payload)


def _complete_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    cv_root = tmp_path / "cv"
    development_root = tmp_path / "development"
    manifest_path = tmp_path / "source-manifest.json"
    manifest_hash = _source_manifest(manifest_path)
    configuration = _configuration()
    _development_origin(development_root / "rf.json", configuration)
    _fold_record(
        cv_root / "rf-source_cv_01.json",
        configuration=configuration,
        source_manifest_sha256=manifest_hash,
        fold_id="source_cv_01",
        train=["p3", "p4"],
        participant_values=[("p1", 2, 0.4, 0.5), ("p2", 3, 0.6, 0.7)],
        nll=0.2,
        brier=0.3,
        ece=0.1,
    )
    _fold_record(
        cv_root / "rf-source_cv_02.json",
        configuration=configuration,
        source_manifest_sha256=manifest_hash,
        fold_id="source_cv_02",
        train=["p1", "p2"],
        participant_values=[("p3", 4, 0.8, 0.9), ("p4", 1, 1.0, 1.0)],
        nll=0.4,
        brier=0.5,
        ece=0.2,
    )
    return cv_root, development_root, manifest_path


def test_more_har_identity_is_derived_from_flags_not_filename() -> None:
    full = derive_model_identity(_configuration("more_har"))
    backbone_configuration = _configuration("more_har")
    for key in (
        "use_augmentation",
        "use_content_objective",
        "use_realization_factorization",
        "use_group_dro",
    ):
        backbone_configuration[key] = False
    backbone = derive_model_identity(backbone_configuration)

    assert full["model_variant_id"] == "more_har_full"
    assert backbone["model_variant_id"] == "more_har_backbone"
    assert full["configuration_id"].startswith("more_har_full--")
    assert full == derive_model_identity(_configuration("more_har"))


def test_source_cv_aggregate_validates_exact_participant_coverage_and_is_deterministic(
    tmp_path: Path,
) -> None:
    cv_root, development_root, manifest_path = _complete_fixture(tmp_path)

    def build() -> dict[str, Any]:
        return build_source_cv_aggregate(
            fold_record_dir=cv_root,
            source_development_dir=development_root,
            source_manifest_path=manifest_path,
            bootstrap_resamples=500,
            bootstrap_seed=9,
        )

    first = build()
    second = build()

    assert first == second
    assert first["configuration_count"] == 1
    assert first["input_fold_record_count"] == 2
    assert first["source_only_guarantees"][
        "each_participant_held_out_once_per_aggregated_configuration"
    ]
    model = first["models"][0]
    assert model["participant_count"] == 4
    assert model["participant_metrics"]["macro_f1"]["mean"] == pytest.approx(0.7)
    assert model["participant_metrics"]["macro_f1"]["worst"] == 0.4
    assert model["calibration"]["negative_log_likelihood"][
        "sample_weighted_fold_aggregate"
    ] == pytest.approx(0.3)
    assert [row["participant_id"] for row in model["participants"]] == [
        "p1",
        "p2",
        "p3",
        "p4",
    ]
    unhashed = dict(first)
    recorded = unhashed.pop("aggregate_record_sha256")
    assert recorded == canonical_json_sha256(unhashed)


def test_bad_record_hash_and_target_access_fail_closed(tmp_path: Path) -> None:
    cv_root, development_root, manifest_path = _complete_fixture(tmp_path)
    bad_path = cv_root / "rf-source_cv_01.json"
    parsed = json.loads(bad_path.read_text(encoding="utf-8"))
    parsed["target_subject_or_window_records_loaded"] = True
    _write_json(bad_path, parsed)

    with pytest.raises(SourceCVAggregationError, match="record_sha256 mismatch"):
        build_source_cv_aggregate(
            fold_record_dir=cv_root,
            source_development_dir=development_root,
            source_manifest_path=manifest_path,
            bootstrap_resamples=100,
        )

    parsed["record_sha256"] = canonical_json_sha256(
        {key: value for key, value in parsed.items() if key != "record_sha256"}
    )
    _write_json(bad_path, parsed)
    with pytest.raises(SourceCVAggregationError, match="target records were not loaded"):
        build_source_cv_aggregate(
            fold_record_dir=cv_root,
            source_development_dir=development_root,
            source_manifest_path=manifest_path,
            bootstrap_resamples=100,
        )


def test_incomplete_configuration_is_quarantined_without_poisoning_complete_evidence(
    tmp_path: Path,
) -> None:
    cv_root, development_root, manifest_path = _complete_fixture(tmp_path)
    manifest = load_json_strict(manifest_path)
    assert isinstance(manifest, dict)
    more_configuration = _configuration("more_har")
    _development_origin(development_root / "more.json", more_configuration)
    _fold_record(
        cv_root / "more-source_cv_01.json",
        configuration=more_configuration,
        source_manifest_sha256=str(manifest["source_window_manifest_sha256"]),
        fold_id="source_cv_01",
        train=["p3", "p4"],
        participant_values=[("p1", 2, 0.5, 0.5), ("p2", 3, 0.5, 0.5)],
        nll=0.5,
        brier=0.5,
        ece=0.2,
    )

    aggregate = build_source_cv_aggregate(
        fold_record_dir=cv_root,
        source_development_dir=development_root,
        source_manifest_path=manifest_path,
        bootstrap_resamples=100,
    )

    assert aggregate["configuration_count"] == 1
    assert aggregate["quarantined_configuration_count"] == 1
    quarantined = aggregate["quarantined_configurations"][0]
    assert quarantined["model_variant_id"] == "more_har_full"
    assert quarantined["reasons"][0] == {
        "code": "INCOMPLETE_OR_DUPLICATE_FOLDS",
        "duplicate_folds": [],
        "missing_folds": ["source_cv_02"],
        "extra_folds": [],
    }


def test_exports_are_hashed_and_never_overwritten(tmp_path: Path) -> None:
    cv_root, development_root, manifest_path = _complete_fixture(tmp_path)
    aggregate = build_source_cv_aggregate(
        fold_record_dir=cv_root,
        source_development_dir=development_root,
        source_manifest_path=manifest_path,
        bootstrap_resamples=100,
    )
    output_root = tmp_path / "exports"
    outputs = write_source_cv_exports_new(aggregate, output_dir=output_root, prefix="summary")
    published = load_json_strict(outputs["json"])
    assert isinstance(published, dict)
    for kind in ("csv", "markdown"):
        assert (
            hashlib.sha256(outputs[kind].read_bytes()).hexdigest()
            == published["exports"][kind]["sha256"]
        )
    unhashed = dict(published)
    recorded = unhashed.pop("aggregate_record_sha256")
    assert recorded == canonical_json_sha256(unhashed)
    assert "not confirmatory" in outputs["markdown"].read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_source_cv_exports_new(aggregate, output_dir=output_root, prefix="summary")
