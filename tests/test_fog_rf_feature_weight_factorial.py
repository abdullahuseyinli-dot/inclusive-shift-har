from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import inclusive_shift_har.experiments.fog_rf_feature_weight_factorial as factorial
from inclusive_shift_har.data.external_har import (
    ExternalHARWindows,
    ObservableWindowPool,
    SourceReceipt,
)
from inclusive_shift_har.data.participant_partitions import (
    build_participant_partition_plan,
)
from inclusive_shift_har.experiments.fog_rf_feature_weight_factorial import (
    NINE_CHANNEL_NAMES,
    SIX_CHANNEL_NAMES,
    _array_sha256,
    _feature_cache_key,
    _fit_cell,
    _write_bytes_create_only,
    analyse,
    method_report,
    participant_first_weights,
    validate_config,
)
from inclusive_shift_har.preprocessing.features import extract_engineered_features

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/fog_rf_feature_weight_factorial_v1.yaml"


def _config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _probabilities(
    predictions: np.ndarray[Any, np.dtype[np.int64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.asarray(np.eye(3, dtype=np.float64)[predictions], dtype=np.float64)


def _complete_reports() -> dict[str, dict[str, Any]]:
    roster = [f"fogstar:{index:03d}" for index in range(1, 23)]
    labels = np.tile(np.arange(3, dtype=np.int64), len(roster))
    participants = np.repeat(np.asarray(roster, dtype=np.str_), 3)
    control_prediction = labels.copy()
    control_prediction[labels == 1] = 2
    control = method_report(
        labels=labels,
        probabilities=_probabilities(control_prediction),
        participant_ids=participants,
        roster=roster,
    )
    candidate = method_report(
        labels=labels,
        probabilities=_probabilities(labels),
        participant_ids=participants,
        roster=roster,
    )
    return {
        "f0": control,
        "f1": copy.deepcopy(control),
        "f2": copy.deepcopy(control),
        "f3": candidate,
    }


def test_frozen_configuration_rejects_parameter_drift() -> None:
    config = _config()
    validate_config(config)
    changed = copy.deepcopy(config)
    changed["estimator"]["n_estimators"] = 499
    with pytest.raises(ValueError, match="n_estimators"):
        validate_config(changed)
    changed = copy.deepcopy(config)
    changed["weighting"]["participant_first"]["raw_sample_weight"] = "1 / n_ic"
    with pytest.raises(ValueError, match="formula"):
        validate_config(changed)


def test_participant_first_weights_equalize_people_and_present_classes() -> None:
    labels = np.asarray([0, 0, 1, 0, 2, 2, 2], dtype=np.int64)
    participants = np.asarray(["p1", "p1", "p1", "p2", "p2", "p2", "p2"], dtype=np.str_)
    weights = participant_first_weights(labels, participants)

    assert weights.mean() == pytest.approx(1.0)
    participant_totals = [
        float(weights[participants == participant].sum()) for participant in ("p1", "p2")
    ]
    assert participant_totals[0] == pytest.approx(participant_totals[1])
    for participant in ("p1", "p2"):
        selected = participants == participant
        class_totals = [
            float(weights[selected & (labels == label)].sum())
            for label in np.unique(labels[selected])
        ]
        assert class_totals == pytest.approx([class_totals[0]] * len(class_totals))


def test_report_retains_zero_window_people_and_fixed_missing_classes() -> None:
    labels = np.asarray([0, 0], dtype=np.int64)
    report = method_report(
        labels=labels,
        probabilities=_probabilities(labels),
        participant_ids=np.asarray(["p1", "p1"], dtype=np.str_),
        roster=["p1", "p2"],
    )

    assert report["status"] == "incomplete_zero_scored_roster_participant"
    assert report["primary"] is None
    assert report["participants_without_scored_rows"] == ["p2"]
    first = report["participants"][0]
    assert first["macro_f1"] == pytest.approx(1.0 / 3.0)
    assert first["present_class_macro_f1"] == pytest.approx(1.0)
    assert report["participants"][1]["eligible"] is False


def test_analysis_uses_declared_control_tie_order_and_all_gate_checks() -> None:
    result = analyse(_complete_reports(), config=_config())

    assert result["strongest_control"] == "f0"
    assert result["promotion_gate"]["status"] == "pass"
    assert all(result["promotion_gate"]["checks"].values())
    comparison = result["comparisons"]["f3_minus_f0"]
    assert comparison["participant_wins"] == 22
    assert comparison["participant_harms"] == 0
    assert comparison["leave_one_out_minimum"] > 0.0
    assert set(result["factorial_contrasts"]) == {
        "derived_nine_minus_six_ordinary",
        "derived_nine_minus_six_participant_first",
        "participant_first_minus_ordinary_six",
        "participant_first_minus_ordinary_derived_nine",
        "difference_in_differences",
    }


def test_feature_cache_key_binds_every_invalidation_dimension() -> None:
    arguments = {
        "source_sha256": "a" * 64,
        "participant_plan_sha256": "b" * 64,
        "observable_window_ids_sha256": "c" * 64,
        "config_sha256": "d" * 64,
        "feature_code_sha256": "e" * 64,
    }
    reference = _feature_cache_key(**arguments)
    assert reference == _feature_cache_key(**arguments)
    for name in arguments:
        changed = dict(arguments)
        changed[name] = "f" * 64
        assert _feature_cache_key(**changed)["cache_key_sha256"] != reference["cache_key_sha256"]


def test_feature_schemas_contain_only_signal_derived_columns() -> None:
    values = np.random.default_rng(7).normal(size=(2, 128, 9))
    six = extract_engineered_features(values[:, :, :6], channel_names=SIX_CHANNEL_NAMES)
    nine = extract_engineered_features(values, channel_names=NINE_CHANNEL_NAMES)
    forbidden = ("participant", "subject", "label", "timestamp", "location", "window_id")

    assert six.values.shape[0] == nine.values.shape[0] == 2
    assert nine.values.shape[1] > six.values.shape[1]
    assert all(
        token not in name.lower() for name in (*six.names, *nine.names) for token in forbidden
    )


def test_registered_fit_excludes_evaluation_person_and_records_weights(tmp_path: Path) -> None:
    participants = np.repeat(np.asarray([f"p{index}" for index in range(1, 7)], dtype=np.str_), 3)
    labels = np.tile(np.arange(3, dtype=np.int64), 6)
    features = np.random.default_rng(11).normal(size=(labels.size, 4))
    evaluation = participants == "p1"
    training = ~evaluation

    probabilities, metadata = _fit_cell(
        feature_values=features,
        labels=labels,
        participants=participants,
        training=training,
        evaluation=evaluation,
        cell_id="f1",
        fold_index=0,
        n_jobs=1,
        output_directory=tmp_path,
        feature_names=("a", "b", "c", "d"),
        deadline=time.perf_counter() + 60.0,
    )

    assert probabilities.shape == (3, 3)
    assert "p1" not in metadata["training_participants"]
    assert metadata["evaluation_participants"] == ["p1"]
    assert metadata["effective_parameters"]["n_estimators"] == 500
    expected = participant_first_weights(labels[training], participants[training])
    assert metadata["sample_weight_summary"]["sha256"] == _array_sha256(expected)
    checkpoint = tmp_path / metadata["checkpoint"]["path"]
    assert checkpoint.is_file()


def test_timeout_and_create_only_guards_write_no_model(tmp_path: Path) -> None:
    target = tmp_path / "once.bin"
    _write_bytes_create_only(target, b"first")
    with pytest.raises(FileExistsError):
        _write_bytes_create_only(target, b"second")
    with pytest.raises(TimeoutError, match="before starting"):
        _fit_cell(
            feature_values=np.zeros((6, 2), dtype=np.float64),
            labels=np.tile(np.arange(3, dtype=np.int64), 2),
            participants=np.asarray(["p1"] * 3 + ["p2"] * 3, dtype=np.str_),
            training=np.asarray([False] * 3 + [True] * 3),
            evaluation=np.asarray([True] * 3 + [False] * 3),
            cell_id="f0",
            fold_index=0,
            n_jobs=1,
            output_directory=tmp_path,
            feature_names=("a", "b"),
            deadline=time.perf_counter() - 1.0,
        )
    assert not (tmp_path / "checkpoints").exists()


def test_synthetic_full_run_and_checkpoint_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roster = tuple(f"fogstar:{index:03d}" for index in range(1, 23))
    observable_people = np.repeat(np.asarray(roster, dtype=np.str_), 4)
    within_person = np.tile(np.arange(4, dtype=np.int64), len(roster))
    observable_windows = np.asarray(
        [
            f"{participant}:candidate-{candidate}"
            for participant, candidate in zip(
                observable_people.tolist(), within_person.tolist(), strict=True
            )
        ],
        dtype=np.str_,
    )
    session_ids = np.asarray(
        [f"{participant}:session-001" for participant in observable_people],
        dtype=np.str_,
    )
    trial_ids = np.asarray(
        [f"{participant}:session-001:recording" for participant in observable_people],
        dtype=np.str_,
    )
    generator = np.random.default_rng(19)
    signals = generator.normal(size=(observable_people.size, 128, 6)).astype(np.float32)
    gravity = generator.normal(size=(observable_people.size, 128, 3)).astype(np.float32)
    scored = within_person < 3
    labels = within_person[scored].astype(np.int64)
    receipt = SourceReceipt(
        dataset_id="fog_star_v3",
        locator="https://zenodo.org/api/records/17838806/files/sensor_data.csv/content",
        member=None,
        declared_size_bytes=119_629_580,
        received_size_bytes=119_629_580,
        computed_sha256="888875133fef386affddd0a5864f122d527d8d7bc588a69905cc0871bef53477",
        declared_digest_algorithm="md5",
        declared_digest="952a37ab147da35e6d4e7a1e9bac44cb",
        computed_declared_digest="952a37ab147da35e6d4e7a1e9bac44cb",
        declared_digest_verified=True,
    )
    pool = ObservableWindowPool(
        signals=signals,
        gravity=gravity,
        participant_ids=observable_people,
        session_ids=session_ids,
        trial_ids=trial_ids,
        window_ids=observable_windows,
    )
    plan = build_participant_partition_plan(
        "fog_star_v3",
        roster,
        roster_basis="synthetic pre-window roster",
        seeds=(11,),
    )
    data = ExternalHARWindows(
        dataset_id="fog_star_v3",
        channel_lane="derived-gravity-9ch",
        sampling_rate_hz=50.0,
        signals=signals[scored],
        gravity=gravity[scored],
        labels=labels,
        participant_ids=observable_people[scored],
        session_ids=session_ids[scored],
        trial_ids=trial_ids[scored],
        window_ids=observable_windows[scored],
        receipts=(receipt,),
        gravity_source="causal low-pass derived from total acceleration",
        gravity_cutoff_hz=0.30,
        observable_candidates=pool,
        participant_partition_plan=plan,
    )
    data.validate()

    def fake_loader(**_: object) -> ExternalHARWindows:
        return data

    code_commit = "a" * 40

    def fake_git_state(_: Path) -> dict[str, Any]:
        return {
            "branch": "test",
            "commit": code_commit,
            "clean": True,
            "porcelain": [],
        }

    monkeypatch.setattr(factorial, "load_fog_star", fake_loader)
    monkeypatch.setattr(factorial, "_git_state", fake_git_state)
    output = tmp_path / ".audit" / "fog_rf_feature_weight_factorial" / "synthetic-replay"
    result = factorial.run_experiment(
        repository_root=ROOT,
        evidence_root=tmp_path,
        config_path=CONFIG,
        protocol_path=ROOT / "docs/research/FOG_RF_FEATURE_WEIGHT_FACTORIAL_V1_PROTOCOL.md",
        output_directory=output,
        code_commit=code_commit,
        timeout_seconds=7200,
    )
    validation = factorial.validate_run(output)

    assert result["fit_count"] == 20
    assert result["status"] == "complete_awaiting_independent_replay"
    assert validation["status"] == "validated"
    assert validation["checkpoint_predictions_replayed"] == 20
    assert (output / "completion_manifest.json").is_file()
