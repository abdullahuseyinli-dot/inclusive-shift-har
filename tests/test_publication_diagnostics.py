from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inclusive_shift_har.experiments.publication_diagnostics import (
    _checked_statistics,
    mechanism_summary,
    write_distribution_figure,
)


def test_mechanism_counts_do_not_infer_activation_or_participant_n() -> None:
    fold = {
        "hera_v2_route_selection": {"enabled": False, "reason": "insufficient_training_support"},
        "selected_ctgr_candidate": "candidate-a",
        "selected_cage_expert": "expert-b",
        "cage_evaluation_route_count": 8,
        "hera_v1_strict_veto_count": 3,
        "physics_evaluation_trusted_fraction": 0.75,
    }
    summary = mechanism_summary({"fold_records": [{"seed": 11, "folds": [fold, fold]}]})
    assert summary["recorded_outer_fold_count"] == 2
    assert summary["hera_v2_routing"]["enabled_fold_count"] == 0
    assert summary["hera_v2_routing"]["evaluation_route_fraction"] is None
    assert summary["selected_ctgr_candidate"] == {"candidate-a": 2}
    assert summary["cage_evaluation_route_count"]["sum_across_all_seed_folds"] == 16
    assert "not independent participant N" in summary["cage_evaluation_route_count"]["unit"]


def test_cost_summary_preserves_missing_measurements_and_separates_gate_estimands() -> None:
    result = {
        "fold_records": [{"model": "DeepConvLSTM-6ch", "parameter_count": 200867}],
        "development_advancement_gate": {"historical": True},
        "primary_seed_averaged": {"comparisons_vs_xgboost_6ch": {"primary": "failed"}},
    }
    summary = mechanism_summary(result)
    model = summary["method_fit_records"]["DeepConvLSTM-6ch"]
    assert "fit_seconds" not in model
    assert model["checkpoint_bytes_by_fit"] == []
    assert summary["prospective_posture_advancement_gate"] is None
    assert summary["retained_legacy_gate_not_the_seed_averaged_primary"] == {"historical": True}
    assert summary["seed_averaged_comparisons_vs_xgboost_6ch"] == {"primary": "failed"}


def _result() -> dict[str, Any]:
    return {
        "dataset": {"dataset_id": "synthetic_test_only"},
        "seeds": [11, 23, 47],
        "primary_seed_averaged": {
            "participant_count": 3,
            "methods": {
                name: {
                    "participant_values": {"p1": 0.2, "p2": 0.5, "p3": 0.8},
                    "mean_participant_macro_f1": 0.5,
                    "participant_bootstrap_95_percent_ci": [0.2, 0.8],
                }
                for name in ("RandomForest-6ch", "TinyHAR-6ch")
            },
        },
    }


@pytest.mark.parametrize("mutation", ["seeds", "mean", "count", "nonfinite", "range"])
def test_distribution_checks_reject_invalid_primary_evidence(mutation: str) -> None:
    result = _result()
    primary = result["primary_seed_averaged"]
    row = primary["methods"]["RandomForest-6ch"]
    if mutation == "seeds":
        result["seeds"] = [11]
    elif mutation == "mean":
        row["mean_participant_macro_f1"] = 0.6
    elif mutation == "count":
        primary["participant_count"] = 4
    else:
        row["participant_values"]["p1"] = float("nan") if mutation == "nonfinite" else -0.2
    with pytest.raises(ValueError):
        _checked_statistics(result)


def test_figure_contains_all_methods_status_and_preserves_existing_output(tmp_path: Path) -> None:
    paths = write_distribution_figure(_result(), tmp_path, status="synthetic diagnostic")
    assert paths[1].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    svg = paths[0].read_text(encoding="utf-8")
    assert "RandomForest-6ch" in svg
    assert "TinyHAR-style-6ch" in svg
    assert "synthetic diagnostic" in svg
    assert len(_checked_statistics(_result())["methods"]) == 2
    with pytest.raises(FileExistsError, match="create-only"):
        write_distribution_figure(_result(), tmp_path, status="changed")
