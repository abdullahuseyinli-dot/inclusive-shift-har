from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from inclusive_shift_har.artifacts.research_provenance import (
    FOG_STAR_FULL_PARTICIPANT_ROSTER,
    HAR_PMD_FULL_PARTICIPANT_ROSTER,
    _typed_path_locator,
)
from inclusive_shift_har.experiments import publication_diagnostics as diagnostics
from inclusive_shift_har.experiments.publication_diagnostics import (
    _bound_evidence_status,
    _checked_statistics,
    _figure_method_label,
    _runtime_qualification_records,
    mechanism_summary,
    write_distribution_figure,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file


@pytest.mark.parametrize(
    ("name", "count", "batch", "diagnostic", "expected"),
    [
        ("RMRP-DG", 6, False, False, "RMRP-DG [6ch]"),
        ("CTGR-DG", 9, False, False, "CTGR-DG [9ch]"),
        ("HERA-DG-full", 9, True, False, "HERA-DG-full [9ch; participant batch]"),
        ("unrecognized", None, False, False, "unrecognized [unknown channels]"),
        (
            "HERA-DG-full",
            9,
            True,
            True,
            "HERA-DG-full [9ch; annotation-selected diagnostic]",
        ),
    ],
)
def test_figure_labels_state_actual_channels_not_misleading_method_suffixes(
    name: str, count: int | None, batch: bool, diagnostic: bool, expected: str
) -> None:
    assert (
        _figure_method_label(
            name,
            {
                "input_channel_count": count,
                "inference_unit": "noncausal_participant_batch" if batch else "independent_window",
                "annotation_selected_evaluation_context": diagnostic,
            },
        )
        == expected
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


def _validation(mode: str) -> dict[str, Any]:
    value: dict[str, Any] = {
        "status": mode,
        "integrity_passed": True,
        "publication_evidence_ready": False,
        "publication_evidence_ready_for_unqualified_methods": False,
        "diagnostic_contract_passed": False,
        "unqualified_method_names": [],
        "annotation_selected_context_methods": [],
        "diagnostic_scope_reasons": [],
    }
    if mode == "VALIDATED":
        value["publication_evidence_ready"] = True
    elif mode == "PARTIALLY_VALIDATED_METHODS":
        value["publication_evidence_ready_for_unqualified_methods"] = True
        value["unqualified_method_names"] = ["XGBoost-6ch"]
        value["annotation_selected_context_methods"] = ["HERA-DG-full"]
    elif mode == "DIAGNOSTIC":
        value["diagnostic_contract_passed"] = True
        value["diagnostic_scope_reasons"] = ["synthetic diagnostic scope"]
    return value


def _boundary_provenance(*, mode: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "protocol_id": "external-har-boundary-provenance-v1",
        "repository_signal_grid_annotation_independent": mode != "camera_bout_oracle",
        "provider_upstream_annotation_conditioned": False,
    }
    if mode is not None:
        value["boundary_mode"] = mode
    return value


def _full_cohort_summary(dataset_id: str) -> dict[str, Any]:
    roster = (
        list(HAR_PMD_FULL_PARTICIPANT_ROSTER)
        if dataset_id == "har_pmd_v1"
        else list(FOG_STAR_FULL_PARTICIPANT_ROSTER)
    )
    return {
        "dataset_id": dataset_id,
        "boundary_provenance": _boundary_provenance(),
        "participant_count": len(roster),
        "participant_partition_plan": {"participant_roster": roster},
        "participant_partition_observation": {
            "planned_participant_count": len(roster),
            "observed_window_participant_count": len(roster),
            "participants_without_retained_windows": [],
        },
    }


@pytest.mark.parametrize(
    ("result", "mode", "expected"),
    [
        (
            {
                "dataset": {
                    "dataset_id": "imu_har_il_v1",
                    "boundary_provenance": {"provider_upstream_annotation_conditioned": False},
                },
                "artifact_evidence_status": "diagnostic_provider_presegmented_development",
            },
            "DIAGNOSTIC",
            "diagnostic_provider_presegmented_development",
        ),
        (
            {
                "source_dataset": {"dataset_id": "imu_har_il_v1"},
                "target_dataset": _full_cohort_summary("fog_star_v3"),
                "artifact_evidence_status": "diagnostic_provider_presegmented_source_transfer",
            },
            "DIAGNOSTIC",
            "diagnostic_provider_presegmented_source_transfer",
        ),
        (
            {
                "dataset": {
                    "dataset_id": "sole_harmony_v1",
                    "boundary_provenance": _boundary_provenance(mode="camera_bout_oracle"),
                },
                "artifact_evidence_status": "diagnostic_oracle_boundary",
            },
            "DIAGNOSTIC",
            "diagnostic_oracle_boundary",
        ),
        (
            {
                "dataset": {
                    "dataset_id": "sole_harmony_v1",
                    "boundary_provenance": _boundary_provenance(mode="session_observable"),
                },
                "artifact_evidence_status": ("diagnostic_session_observable_temporal_development"),
            },
            "DIAGNOSTIC",
            "diagnostic_session_observable_temporal_development",
        ),
        (
            {
                "dataset": _full_cohort_summary("har_pmd_v1"),
                "artifact_evidence_status": "validated_stress_test",
            },
            "VALIDATED",
            "validated_stress_test",
        ),
        (
            {
                "dataset": _full_cohort_summary("fog_star_v3"),
                "artifact_evidence_status": "validated_development",
            },
            "VALIDATED",
            "validated_development",
        ),
    ],
)
def test_diagnostics_status_is_bound_to_frozen_dataset_role_and_validator(
    result: dict[str, Any], mode: str, expected: str
) -> None:
    assert _bound_evidence_status(result, _validation(mode)) == (expected, mode)


def test_har_pmd_pilot_cannot_be_upgraded_by_forged_validated_mode() -> None:
    pilot = _full_cohort_summary("har_pmd_v1")
    pilot["participant_count"] = 12
    pilot["participant_partition_observation"] = {
        "planned_participant_count": 120,
        "observed_window_participant_count": 12,
        "participants_without_retained_windows": list(HAR_PMD_FULL_PARTICIPANT_ROSTER[12:]),
    }
    result = {"dataset": pilot, "artifact_evidence_status": "provisional_pilot_stress"}
    with pytest.raises(ValueError, match="cannot promote"):
        _bound_evidence_status(result, _validation("VALIDATED"))


@pytest.mark.parametrize(
    "validation",
    [
        _validation("SUPERSEDED_PROTOCOL"),
        _validation("PROVISIONAL"),
        {**_validation("DIAGNOSTIC"), "diagnostic_contract_passed": False},
        {**_validation("VALIDATED"), "publication_evidence_ready": False},
    ],
)
def test_superseded_provisional_or_contradictory_validation_is_rejected(
    validation: dict[str, Any],
) -> None:
    result = {
        "dataset": _full_cohort_summary("fog_star_v3"),
        "artifact_evidence_status": "validated_development",
    }
    with pytest.raises(ValueError, match="not accepted current-protocol evidence"):
        _bound_evidence_status(result, validation)


def test_diagnostic_mode_requires_explicit_scope_reasons() -> None:
    result = {
        "dataset": {"dataset_id": "imu_har_il_v1"},
        "artifact_evidence_status": "diagnostic_provider_presegmented_development",
    }
    validation = {**_validation("DIAGNOSTIC"), "diagnostic_scope_reasons": []}
    with pytest.raises(ValueError, match="explicit unique scope reasons"):
        _bound_evidence_status(result, validation)


def test_partially_validated_mode_requires_disjoint_explicit_method_sets() -> None:
    result = {
        "dataset": _full_cohort_summary("fog_star_v3"),
        "artifact_evidence_status": "validated_development",
    }
    assert _bound_evidence_status(result, _validation("PARTIALLY_VALIDATED_METHODS")) == (
        "validated_development",
        "PARTIALLY_VALIDATED_METHODS",
    )
    mutations: tuple[dict[str, list[str]], ...] = (
        {"unqualified_method_names": []},
        {"annotation_selected_context_methods": []},
        {
            "unqualified_method_names": ["same"],
            "annotation_selected_context_methods": ["same"],
        },
    )
    for mutation in mutations:
        validation = {**_validation("PARTIALLY_VALIDATED_METHODS"), **mutation}
        with pytest.raises(ValueError, match="distinct non-empty"):
            _bound_evidence_status(result, validation)


def test_forged_artifact_status_cannot_override_frozen_dataset_role() -> None:
    result = {
        "dataset": {
            "dataset_id": "sole_harmony_v1",
            "boundary_provenance": _boundary_provenance(mode="session_observable"),
        },
        "artifact_evidence_status": "diagnostic_oracle_boundary",
    }
    with pytest.raises(ValueError, match="differs from its frozen dataset role"):
        _bound_evidence_status(result, _validation("DIAGNOSTIC"))


def test_typed_path_locator_distinguishes_repository_and_external_paths(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    inside = repository / "results" / "run"
    outside = tmp_path / "external" / "run"
    inside.mkdir(parents=True)
    outside.mkdir(parents=True)
    assert _typed_path_locator(inside, repository) == {
        "path_kind": "repository_relative",
        "path": "results/run",
    }
    assert _typed_path_locator(outside, repository) == {
        "path_kind": "external_absolute",
        "path": str(outside.resolve()),
    }


def test_runtime_notice_locators_are_typed_and_sorted(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    campaign = repository / "results" / "campaign"
    run = campaign / "run"
    run.mkdir(parents=True)
    (campaign / "campaign_plan.json").write_text("{}\n", encoding="utf-8")
    second = campaign / "runtime_termination_failure_b.json"
    first = run / "runtime_termination_failure_a.json"
    first.write_text('{"status":"failed"}\n', encoding="utf-8")
    second.write_text('{"status":"failed"}\n', encoding="utf-8")
    records = _runtime_qualification_records(run, repository)
    assert records == [
        {
            "locator": _typed_path_locator(first, repository),
            "sha256": sha256_file(first),
        },
        {
            "locator": _typed_path_locator(second, repository),
            "sha256": sha256_file(second),
        },
    ]


def test_runtime_notice_symlinks_are_rejected(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    campaign = repository / "results" / "campaign"
    run = campaign / "run"
    run.mkdir(parents=True)
    (campaign / "campaign_plan.json").write_text("{}\n", encoding="utf-8")
    physical = campaign / "physical.json"
    physical.write_text("{}\n", encoding="utf-8")
    linked = campaign / "runtime_termination_failure_link.json"
    try:
        linked.symlink_to(physical)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")
    with pytest.raises(ValueError, match="regular non-symlink"):
        _runtime_qualification_records(run, repository)


def _write_minimal_diagnostic_run(run: Path, result: dict[str, Any]) -> None:
    run.mkdir(parents=True)
    (run / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (run / "data_audit.json").write_text("{}\n", encoding="utf-8")
    (run / "predictions.npz").write_bytes(b"synthetic predictions")


def test_write_diagnostics_preserves_canonical_status_and_typed_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    campaign = repository / "results" / "campaign"
    run = campaign / "imu"
    output = repository / "reports" / "diagnostics"
    result = {
        "dataset": {"dataset_id": "imu_har_il_v1"},
        "artifact_evidence_status": "diagnostic_provider_presegmented_development",
        "prediction_artifact": {"path": "predictions.npz"},
        "primary_seed_averaged": {"estimand": "participant fixed-class macro-F1"},
    }
    _write_minimal_diagnostic_run(run, result)
    (campaign / "campaign_plan.json").write_text("{}\n", encoding="utf-8")
    notice = campaign / "runtime_termination_failure_example.json"
    notice.write_text('{"status":"failed"}\n', encoding="utf-8")
    monkeypatch.setattr(
        diagnostics, "validate_run_directory", lambda *_args: _validation("DIAGNOSTIC")
    )
    monkeypatch.setattr(diagnostics, "method_inference_contracts", lambda _result: {})
    monkeypatch.setattr(diagnostics, "_git_state", lambda _root: {"commit": "a" * 40})
    monkeypatch.setattr(
        diagnostics,
        "_source_input_manifest",
        lambda _root: {"protocol_id": "synthetic-test-only"},
    )
    rendered_statuses: list[str] = []

    def fake_figure(_result: dict[str, Any], figure_output: Path, *, status: str) -> list[Path]:
        rendered_statuses.append(status)
        paths = [figure_output / "participant_distribution.svg", figure_output / "figure.png"]
        for path in paths:
            path.write_bytes(b"figure")
        return paths

    monkeypatch.setattr(diagnostics, "write_distribution_figure", fake_figure)
    record = diagnostics.write_diagnostics(run, output, repository)
    assert record["schema_version"] == "2.0.0"
    assert record["evidence_status"] == "diagnostic_provider_presegmented_development"
    assert record["validation_mode"] == "DIAGNOSTIC"
    assert record["runtime_qualification_present"] is True
    assert record["run_directory"] == {
        "path_kind": "repository_relative",
        "path": "results/campaign/imu",
    }
    assert record["runtime_qualifications"] == [
        {
            "locator": _typed_path_locator(notice, repository),
            "sha256": sha256_file(notice),
        }
    ]
    assert rendered_statuses == ["diagnostic_provider_presegmented_development | runtime-qualified"]
    unhashed = dict(record)
    declared_hash = unhashed.pop("record_sha256")
    assert declared_hash == canonical_json_sha256(unhashed)


def test_unqualified_sole_diagnostic_is_rejected_before_output_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    run = repository / "results" / "sole"
    output = repository / "reports" / "diagnostics"
    result = {
        "dataset": {
            "dataset_id": "sole_harmony_v1",
            "boundary_provenance": {"boundary_mode": "camera_bout_oracle"},
        },
        "artifact_evidence_status": "diagnostic_oracle_boundary",
        "prediction_artifact": {"path": "predictions.npz"},
        "primary_seed_averaged": {"estimand": "participant fixed-class macro-F1"},
    }
    _write_minimal_diagnostic_run(run, result)
    monkeypatch.setattr(
        diagnostics,
        "validate_run_directory",
        lambda *_args: {
            **_validation("DIAGNOSTIC"),
            "diagnostic_contract_passed": False,
        },
    )
    with pytest.raises(ValueError, match="not accepted current-protocol evidence"):
        diagnostics.write_diagnostics(run, output, repository)
    assert not output.exists()


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


def test_small_comparison_figure_xlabel_does_not_overlap_evidence_footer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from matplotlib.figure import Figure

    original = Figure.savefig
    checked = []

    def checked_savefig(figure: Figure, *args: Any, **kwargs: Any) -> None:
        figure.canvas.draw()
        xlabel = figure.axes[0].xaxis.label.get_window_extent()
        footer = figure.texts[0].get_window_extent()
        assert xlabel.y0 > footer.y1 + 2
        checked.append(True)
        original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", checked_savefig)
    write_distribution_figure(_result(), tmp_path, status="synthetic diagnostic")
    assert checked == [True, True]
