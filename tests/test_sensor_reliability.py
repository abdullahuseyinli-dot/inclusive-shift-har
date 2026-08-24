from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from inclusive_shift_har.artifacts.sensor_stress import (
    SensorStressArtifactError,
    create_secondary_stress_artifacts,
    create_source_clean_reference_artifacts,
)
from inclusive_shift_har.evaluation.sensor_reliability import (
    SensorReliabilityError,
    apply_sensor_reliability_stress,
    load_sensor_reliability_config,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file


def _config_path(repository_root: Path) -> Path:
    return repository_root / "configs/experiments/sensor_reliability_stress_v1.yaml"


def _windows() -> tuple[np.ndarray, tuple[str, ...]]:
    generator = np.random.default_rng(17)
    windows = generator.normal(size=(3, 128, 6)).astype(np.float32)
    return windows, ("window-a", "window-b", "window-c")


def test_locked_stress_config_is_self_hashed_and_secondary(repository_root: Path) -> None:
    config = load_sensor_reliability_config(_config_path(repository_root))
    assert config.track_role == "secondary_post_confirmatory"
    assert config.input_space == "physical_pre_normalization"
    assert len(config.conditions) == 6
    assert {condition.kind for condition in config.conditions} == {
        "gaussian_noise",
        "contiguous_dropout",
        "missing_axis",
        "missing_modality",
        "rate_reduction",
        "linear_drift",
    }


def test_stress_config_rejects_policy_or_duplicate_key_drift(
    repository_root: Path, tmp_path: Path
) -> None:
    parsed = yaml.safe_load(_config_path(repository_root).read_text(encoding="utf-8"))
    parsed["policy"]["used_for_model_selection"] = True
    parsed.pop("config_sha256")
    parsed["config_sha256"] = canonical_json_sha256(parsed)
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(parsed, sort_keys=False), encoding="utf-8")
    with pytest.raises(SensorReliabilityError, match="policy contract"):
        load_sensor_reliability_config(changed)

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("schema_version: '1.0.0'\nschema_version: '1.0.0'\n", encoding="utf-8")
    with pytest.raises(SensorReliabilityError, match="duplicate key"):
        load_sensor_reliability_config(duplicate)


def test_every_stress_is_deterministic_order_invariant_and_non_mutating(
    repository_root: Path,
) -> None:
    config = load_sensor_reliability_config(_config_path(repository_root))
    windows, ids = _windows()
    original = windows.copy()
    order = np.asarray([2, 0, 1], dtype=np.int64)
    inverse = np.argsort(order)
    for condition in config.conditions:
        first = apply_sensor_reliability_stress(
            windows, ids, config=config, condition_id=condition.condition_id
        )
        second = apply_sensor_reliability_stress(
            windows[order],
            tuple(ids[index] for index in order),
            config=config,
            condition_id=condition.condition_id,
        )
        assert first.windows.shape == windows.shape
        assert first.windows.dtype == windows.dtype
        assert np.isfinite(first.windows).all()
        np.testing.assert_array_equal(first.windows, second.windows[inverse])
        assert first.metadata["track_role"] == "secondary_post_confirmatory"
        assert first.metadata["primary_claim_eligible"] is False
        assert first.metadata["used_for_model_selection"] is False
        metadata = dict(first.metadata)
        claimed = metadata.pop("record_sha256")
        assert claimed == canonical_json_sha256(metadata)
    np.testing.assert_array_equal(windows, original)


def test_missing_axis_modality_dropout_rate_and_drift_have_exact_effects(
    repository_root: Path,
) -> None:
    config = load_sensor_reliability_config(_config_path(repository_root))
    ones = np.ones((2, 128, 6), dtype=np.float64)
    ids = ("one", "two")

    axis = apply_sensor_reliability_stress(
        ones, ids, config=config, condition_id="missing-accelerometer-z"
    ).windows
    assert np.count_nonzero(axis[:, :, 2]) == 0
    np.testing.assert_array_equal(axis[:, :, [0, 1, 3, 4, 5]], ones[:, :, [0, 1, 3, 4, 5]])

    modality = apply_sensor_reliability_stress(
        ones, ids, config=config, condition_id="missing-gyroscope"
    ).windows
    np.testing.assert_array_equal(modality[:, :, :3], ones[:, :, :3])
    assert np.count_nonzero(modality[:, :, 3:]) == 0

    dropout = apply_sensor_reliability_stress(
        ones, ids, config=config, condition_id="contiguous-dropout-32-all-channels"
    ).windows
    for row in range(dropout.shape[0]):
        missing = np.flatnonzero(np.all(dropout[row] == 0.0, axis=1))
        assert missing.size == 32
        np.testing.assert_array_equal(missing, np.arange(missing[0], missing[0] + 32))

    samples = np.arange(128, dtype=np.float64) ** 2
    quadratic = np.repeat(samples[None, :, None], 6, axis=2)
    reduced = apply_sensor_reliability_stress(
        quadratic, ("quadratic",), config=config, condition_id="rate-reduction-factor-2"
    ).windows
    np.testing.assert_array_equal(reduced[0, 0::2], quadratic[0, 0::2])
    assert reduced[0, 1, 0] == pytest.approx((samples[0] + samples[2]) / 2.0)

    drifted = apply_sensor_reliability_stress(
        ones, ids, config=config, condition_id="linear-drift-level-1"
    ).windows
    np.testing.assert_allclose(drifted[:, 0], ones[:, 0])
    np.testing.assert_allclose(drifted[:, -1], ones[:, -1] + 0.05)


def test_gaussian_noise_changes_with_stable_window_identity(repository_root: Path) -> None:
    config = load_sensor_reliability_config(_config_path(repository_root))
    windows = np.zeros((2, 128, 6), dtype=np.float32)
    first = apply_sensor_reliability_stress(
        windows,
        ("identity-a", "identity-b"),
        config=config,
        condition_id="gaussian-noise-level-1",
    ).windows
    second = apply_sensor_reliability_stress(
        windows,
        ("identity-a", "identity-c"),
        config=config,
        condition_id="gaussian-noise-level-1",
    ).windows
    np.testing.assert_array_equal(first[0], second[0])
    assert not np.array_equal(first[1], second[1])


def test_secondary_artifacts_reuse_prediction_contract_and_participant_metrics(
    repository_root: Path, tmp_path: Path
) -> None:
    config = load_sensor_reliability_config(_config_path(repository_root))
    windows = np.zeros((6, 128, 6), dtype=np.float32)
    window_ids = tuple(f"window-{index}" for index in range(6))
    corruption = apply_sensor_reliability_stress(
        windows,
        window_ids,
        config=config,
        condition_id="missing-gyroscope",
    )
    labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    probabilities = np.full((6, 3), 0.05, dtype=np.float64)
    probabilities[np.arange(6), labels] = 0.9
    logits = np.log(probabilities)
    participants = ("participant-a",) * 3 + ("participant-b",) * 3
    record = create_secondary_stress_artifacts(
        prediction_path=tmp_path / "predictions.npz",
        record_path=tmp_path / "record.json",
        allowed_root=tmp_path,
        logits=logits,
        probabilities=probabilities,
        labels=labels,
        participant_ids=participants,
        window_ids=window_ids,
        class_names=("mobility", "sitting", "standing"),
        config=config,
        condition_metadata=corruption.metadata,
        cohort="target",
        model_id="synthetic-model",
        seed=7,
        frozen_checkpoint_sha256="a" * 64,
        frozen_normalization_sha256="b" * 64,
        frozen_calibrator_sha256="e" * 64,
        base_prediction_sha256="c" * 64,
        primary_confirmatory_completion_record_sha256="d" * 64,
        primary_cache_record_file_sha256="f" * 64,
        primary_cache_record_sha256="1" * 64,
        created_at_utc="2099-01-01T00:00:00Z",
    )
    assert record["track_role"] == "secondary_post_confirmatory"
    assert record["primary_claim_eligible"] is False
    assert record["participant_level_report"]["participant_count"] == 2
    assert record["primary_cache_record_file_sha256"] == "f" * 64
    assert record["primary_cache_record_sha256"] == "1" * 64
    stored = json.loads((tmp_path / "record.json").read_text(encoding="utf-8"))
    claimed = stored.pop("record_sha256")
    assert claimed == canonical_json_sha256(stored)
    assert record["prediction_artifact"]["sha256"] == sha256_file(tmp_path / "predictions.npz")
    with np.load(tmp_path / "predictions.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {
            "logits",
            "probabilities",
            "labels",
            "participant_ids",
            "window_ids",
        }
        np.testing.assert_array_equal(archive["window_ids"], np.asarray(window_ids))

    with pytest.raises(FileExistsError, match="overwrite"):
        create_secondary_stress_artifacts(
            prediction_path=tmp_path / "predictions.npz",
            record_path=tmp_path / "second-record.json",
            allowed_root=tmp_path,
            logits=logits,
            probabilities=probabilities,
            labels=labels,
            participant_ids=participants,
            window_ids=window_ids,
            class_names=("mobility", "sitting", "standing"),
            config=config,
            condition_metadata=corruption.metadata,
            cohort="target",
            model_id="synthetic-model",
            seed=7,
            frozen_checkpoint_sha256="a" * 64,
            frozen_normalization_sha256="b" * 64,
            frozen_calibrator_sha256="e" * 64,
            base_prediction_sha256="c" * 64,
            primary_confirmatory_completion_record_sha256="d" * 64,
            primary_cache_record_file_sha256="f" * 64,
            primary_cache_record_sha256="1" * 64,
            created_at_utc="2099-01-01T00:00:00Z",
        )


def test_source_clean_result_binds_primary_cache_record_hashes(tmp_path: Path) -> None:
    labels = np.asarray([0, 1, 2], dtype=np.int64)
    probabilities = np.eye(3, dtype=np.float64) * 0.8 + 0.2 / 3.0
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    record = create_source_clean_reference_artifacts(
        prediction_path=tmp_path / "source-clean.npz",
        record_path=tmp_path / "source-clean.json",
        allowed_root=tmp_path,
        logits=np.log(probabilities),
        probabilities=probabilities,
        labels=labels,
        participant_ids=("p1", "p1", "p2"),
        window_ids=("w1", "w2", "w3"),
        class_names=("mobility", "sitting", "standing"),
        model_id="synthetic-model",
        seed=7,
        stress_config_sha256="a" * 64,
        final_freeze_inventory_sha256="b" * 64,
        frozen_checkpoint_sha256="c" * 64,
        frozen_normalization_sha256="d" * 64,
        frozen_calibrator_sha256="e" * 64,
        locked_target_index_record_sha256="f" * 64,
        primary_cache_record_file_sha256="1" * 64,
        primary_cache_record_sha256="2" * 64,
        created_at_utc="2099-01-01T00:00:00Z",
    )

    assert record["primary_cache_record_file_sha256"] == "1" * 64
    assert record["primary_cache_record_sha256"] == "2" * 64


def test_secondary_artifact_rejects_changed_condition_metadata(repository_root: Path) -> None:
    config = load_sensor_reliability_config(_config_path(repository_root))
    corruption = apply_sensor_reliability_stress(
        np.zeros((1, 128, 6), dtype=np.float32),
        ("window",),
        config=config,
        condition_id="missing-gyroscope",
    )
    changed = dict(corruption.metadata)
    changed["used_for_model_selection"] = True
    with pytest.raises(SensorStressArtifactError, match="self-hash"):
        create_secondary_stress_artifacts(
            prediction_path=Path("unused.npz"),
            record_path=Path("unused.json"),
            allowed_root=repository_root,
            logits=np.zeros((1, 2)),
            probabilities=np.asarray([[0.5, 0.5]]),
            labels=np.asarray([0]),
            participant_ids=("p",),
            window_ids=("window",),
            class_names=("a", "b"),
            config=config,
            condition_metadata=changed,
            cohort="target",
            model_id="synthetic",
            seed=0,
            frozen_checkpoint_sha256="a" * 64,
            frozen_normalization_sha256="b" * 64,
            frozen_calibrator_sha256="e" * 64,
            base_prediction_sha256="c" * 64,
            primary_confirmatory_completion_record_sha256="d" * 64,
            primary_cache_record_file_sha256="f" * 64,
            primary_cache_record_sha256="1" * 64,
            created_at_utc="2099-01-01T00:00:00Z",
        )
