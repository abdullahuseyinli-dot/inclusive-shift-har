from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from inclusive_shift_har.experiments import fog_compact_joint_readout as joint_readout
from inclusive_shift_har.experiments.fog_compact_joint_readout import (
    CONFIG_RELATIVE,
    CURRENT_GRAVITY_SHA256,
    FIT_SCHEDULE,
    OUTPUTS,
    PCA_SCHEDULE,
    PROTOCOL_RELATIVE,
    PROTOCOL_SHA256,
    REPRESENTATIONS,
    _compact_raw,
    _compose,
    _gate,
    _standardize_compact,
    run_experiment,
    validate_config,
)


def test_frozen_configuration_and_attempt_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / CONFIG_RELATIVE).read_text(encoding="utf-8"))
    assert config["authority"]["specification_sha256"] == joint_readout.SPEC_SHA256

    # Exercise the binding contract without requiring the private execution archive.
    # The production specification hash and tracked protocol remain unchanged.
    specification = tmp_path / "specification.md"
    specification_bytes = b"Synthetic governing specification for contract verification.\n"
    specification.write_bytes(specification_bytes)
    fixture_hash = hashlib.sha256(specification_bytes).hexdigest()
    monkeypatch.setattr(joint_readout, "SPEC_SHA256", fixture_hash)
    config["authority"]["specification_sha256"] = fixture_hash
    validate_config(
        config,
        root / CONFIG_RELATIVE,
        root / PROTOCOL_RELATIVE,
        specification,
    )
    assert REPRESENTATIONS == ("S", "R16", "P16")
    assert OUTPUTS[-1] == "P16-full"
    assert len(PCA_SCHEDULE) == 10
    assert len(FIT_SCHEDULE) == 15
    assert PCA_SCHEDULE[0] == (0, "R16")
    assert FIT_SCHEDULE[-1] == (4, "P16")
    assert config["protocol_sha256"] == PROTOCOL_SHA256
    assert len(CURRENT_GRAVITY_SHA256) == 64

    specification.write_bytes(specification_bytes + b"Tampered.\n")
    with pytest.raises(ValueError, match="governing specification changed"):
        validate_config(config, root / CONFIG_RELATIVE, root / PROTOCOL_RELATIVE, specification)
    with pytest.raises(ValueError, match="governing specification missing"):
        validate_config(
            config, root / CONFIG_RELATIVE, root / PROTOCOL_RELATIVE, tmp_path / "missing.md"
        )


def test_compact_feature_definition_and_train_only_standardization() -> None:
    energy = np.zeros((1939, 2), dtype=np.float64)
    full = np.zeros(1939, dtype=np.bool_)
    current = np.zeros(1939, dtype=np.bool_)
    current[:2] = True
    full[0] = True
    gravity = np.zeros((1939, 128, 3), dtype=np.float64)
    gravity[0, :, 0] = 1.0
    gravity[1, :64, 1] = 1.0
    gravity[1, 64:, 1] = -1.0
    raw, receipt = _compact_raw(energy, full, gravity, current)
    assert raw.shape == (1939, 12)
    assert np.array_equal(raw[:, 2], full.astype(np.float64))
    assert np.allclose(raw[0, 3:6], [1.0, 0.0, 0.0])
    assert np.allclose(raw[0, 6:], 0.0)
    assert np.allclose(raw[1, 3:6], 0.0)
    assert raw[1, 9] == pytest.approx(1.0)
    assert receipt["zero_gravity_norm_samples_current"] == 0

    training = np.zeros(1939, dtype=np.bool_)
    training[:2] = True
    standardized, mean, scale = _standardize_compact(raw, training)
    assert standardized.shape == raw.shape
    assert mean.shape == scale.shape == (11,)
    assert np.array_equal(standardized[:, 2], raw[:, 2])
    assert np.isfinite(standardized).all()


def test_composition_preserves_exact_fallbacks_and_raw_mobility() -> None:
    current = np.ones(1939, dtype=np.bool_)
    current[0] = False
    q_nonzero = np.ones(1939, dtype=np.bool_)
    q_nonzero[1] = False
    l9v = np.tile(np.array([0.2, 0.3, 0.5]), (1939, 1))
    l9v[0] = [0.7, 0.1, 0.2]
    l9v[1] = [1.0, 0.0, 0.0]
    b0 = l9v.copy()
    b0[0] = [0.6, 0.2, 0.2]
    learned = np.tile(np.array([0.4, 0.5, 0.1]), (1939, 1))
    raw = {name: learned.copy() for name in REPRESENTATIONS}
    outputs = _compose(raw, {"current": current, "q_nonzero": q_nonzero, "l9v": l9v, "b0": b0})
    for representation in REPRESENTATIONS:
        fixed = outputs[f"{representation}-fixed"]
        full = outputs[f"{representation}-full"]
        assert np.array_equal(fixed[0], b0[0])
        assert np.array_equal(full[0], b0[0])
        assert np.array_equal(fixed[1], l9v[1])
        assert np.array_equal(full[1], l9v[1])
        assert np.array_equal(fixed[2:, 0], full[2:, 0])
        assert np.allclose(fixed.sum(axis=1), 1.0)
        assert np.allclose(full.sum(axis=1), 1.0)


def test_gate_applies_win_rule_only_when_requested() -> None:
    comparison = {
        "mean_difference": 0.02,
        "mean_difference_95_percent_bootstrap_interval": [0.001, 0.04],
        "participant_wins": 13,
        "bottom_30_percent_difference": 0.0,
        "worst_participant_difference": 0.0,
        "minimum_paired_participant_difference": 0.0,
        "class_recall_differences": {"mobility": 0.0, "sitting": 0.0, "standing": 0.0},
        "leave_one_participant_out_mean_differences": [0.01] * 22,
    }
    leave = [{"mean_participant_difference": 0.01} for _ in range(5)]
    without_wins = _gate(comparison, leave, 0.01, None, True)
    with_wins = _gate(comparison, leave, 0.01, 14, True)
    assert without_wins["status"] == "pass"
    assert "participant_wins" not in without_wins["checks"]
    assert with_wins["status"] == "fail"
    assert with_wins["checks"]["participant_wins"] is False


def test_create_only_run_rejects_existing_directory(tmp_path: Path) -> None:
    output = (
        tmp_path
        / ".audit"
        / "fog_compact_joint_readout"
        / "fog-compact-joint-readout-seed11-20260909-001"
    )
    output.mkdir(parents=True)
    marker = output / "preserve.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="create-only run directory exists"):
        run_experiment(tmp_path, tmp_path, output)
    assert marker.read_text(encoding="utf-8") == "preserve"
