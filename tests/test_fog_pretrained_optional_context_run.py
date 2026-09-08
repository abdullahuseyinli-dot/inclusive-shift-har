from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from inclusive_shift_har.experiments.fog_pretrained_optional_context_run import (
    CELL_ORDER,
    CONFIG_RELATIVE,
    FIT_SCHEDULE,
    HISTORICAL_MOTION_ORDER,
    MAXIMUM_FIT_ATTEMPTS,
    METHOD_ORDER,
    PROTOCOL_RELATIVE,
    _gate,
    _participant_fold_vector,
    run_experiment,
    validate_config,
)


def test_frozen_configuration_and_twenty_attempt_schedule() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / CONFIG_RELATIVE
    protocol_path = root / PROTOCOL_RELATIVE
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config, config_path, protocol_path)
    assert CELL_ORDER == ("Q", "M", "R", "P")
    assert len(FIT_SCHEDULE) == MAXIMUM_FIT_ATTEMPTS == 20
    assert FIT_SCHEDULE[:4] == ((0, "Q"), (0, "M"), (0, "R"), (0, "P"))
    assert FIT_SCHEDULE[-1] == (4, "P")
    assert HISTORICAL_MOTION_ORDER == ("E2", "T128", "T500", "T500-P")
    assert METHOD_ORDER[4:8] == HISTORICAL_MOTION_ORDER


def test_leave_one_fold_contract_uses_one_fold_per_participant() -> None:
    roster = [f"fogstar:{index:03d}" for index in range(1, 23)]
    expected = np.asarray([0] * 5 + [1] * 5 + [2] * 4 + [3] * 4 + [4] * 4, dtype=np.int64)
    people = np.repeat(np.asarray(roster, dtype=np.str_), 3)
    row_folds = np.repeat(expected, 3)
    assert np.array_equal(_participant_fold_vector(people, row_folds, roster), expected)

    row_folds[1] = 4
    with pytest.raises(ValueError, match="participant fold assignment changed"):
        _participant_fold_vector(people, row_folds, roster)


def test_mechanism_gate_does_not_inherit_practical_fourteen_win_requirement() -> None:
    comparison = {
        "mean_difference": 0.02,
        "mean_difference_95_percent_bootstrap_interval": [0.001, 0.03],
        "participant_wins": 1,
        "bottom_30_percent_difference": 0.0,
        "worst_participant_difference": 0.0,
        "minimum_paired_participant_difference": 0.0,
        "class_recall_differences": {"mobility": 0.0, "sitting": 0.0, "standing": 0.0},
        "leave_one_participant_out_mean_differences": [0.01, 0.02],
    }
    leave_folds = [{"mean_participant_difference": 0.01} for _ in range(5)]
    mechanism = _gate(
        comparison,
        leave_folds,
        minimum_mean_gain=0.01,
        require_fourteen_wins=False,
        validation_complete=True,
    )
    practical = _gate(
        comparison,
        leave_folds,
        minimum_mean_gain=0.01,
        require_fourteen_wins=True,
        validation_complete=True,
    )
    assert mechanism["status"] == "pass"
    assert "participant_wins" not in mechanism["checks"]
    assert practical["status"] == "fail"
    assert practical["checks"]["participant_wins"] is False


def test_existing_run_rejection_preserves_create_only_evidence(tmp_path: Path) -> None:
    output = (
        tmp_path
        / ".audit"
        / "fog_pretrained_optional_context"
        / "fog-pretrained-optional-context-seed11-20260908-001"
    )
    output.mkdir(parents=True)
    marker = output / "preserve.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="create-only run directory exists"):
        run_experiment(
            repository_root=tmp_path,
            evidence_root=tmp_path,
            harnet_root=tmp_path,
            config_path=tmp_path / CONFIG_RELATIVE,
            protocol_path=tmp_path / PROTOCOL_RELATIVE,
            output_directory=output,
            code_commit="unused",
        )
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert sorted(path.name for path in output.iterdir()) == ["preserve.txt"]
