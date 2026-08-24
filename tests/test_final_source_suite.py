from __future__ import annotations

from pathlib import Path

import pytest

from inclusive_shift_har.artifacts.source_finalization import load_final_selection_plan
from inclusive_shift_har.experiments.final_source_suite import (
    FinalSourceSuiteError,
    _runner_kwargs,
    _safe_output_path,
    validate_plan_configuration_contract,
)


def test_repository_final_plan_is_self_hashed_and_complete(repository_root: Path) -> None:
    plan = load_final_selection_plan(
        repository_root / "results/protocol/final_source_selection_plan_v1.json"
    )
    assert len(plan["models"]) == 20
    assert plan["required_seed_order"] == [11, 23, 47, 89, 131]
    assert plan["neural_training_device"] == "cuda"
    assert plan["target_subject_or_window_records_used"] is False
    assert plan["target_predictions_or_performance_accessed"] is False
    validated = validate_plan_configuration_contract(plan)
    assert len(validated) == 20


def test_final_runner_arguments_merge_predeclared_common_values() -> None:
    arguments = _runner_kwargs(
        {"epochs": 7, "learning_rate": 0.001, "disable_cudnn": True},
        common_value={
            "weight_decay": 0.0001,
            "disable_cudnn": False,
            "zero_channel_indices": [],
        },
        model_id="synthetic",
    )
    assert arguments["disable_cudnn"] is True
    assert arguments["zero_channel_indices"] == ()
    assert arguments["weight_decay"] == 0.0001


def test_final_runner_rejects_unknown_argument_and_path_escape(tmp_path: Path) -> None:
    with pytest.raises(FinalSourceSuiteError, match="unknown runner arguments"):
        _runner_kwargs(
            {
                "epochs": 7,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "target_subjects": ["forbidden"],
            },
            common_value={},
            model_id="synthetic",
        )
    with pytest.raises(FinalSourceSuiteError, match="escapes"):
        _safe_output_path(
            "../outside/seed-{seed}.json",
            seed=11,
            repository_root=tmp_path.resolve(),
            role="synthetic output",
        )
