from __future__ import annotations

import inspect
import math
from pathlib import Path

import pytest
import yaml

import inclusive_shift_har.experiments.same_attachment_directional_reference_geometry as runner
from inclusive_shift_har.experiments.same_attachment_directional_reference_geometry import (
    FIXTURE_NAMES,
    GeometryQualificationError,
    _load_config,
    build_geometry_report,
    run_geometry_qualification,
    validate_geometry_qualification,
)
from inclusive_shift_har.manifests.canonical import load_json_strict
from inclusive_shift_har.models.same_attachment_directional_reference import (
    compose_same_attachment_probabilities,
    directional_reference_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/same_attachment_directional_reference_geometry_v1.yaml"
PROTOCOL = ROOT / "docs/research/SAME_ATTACHMENT_DIRECTIONAL_REFERENCE_GEOMETRY_V1_PROTOCOL.md"


def test_geometry_fixture_schedule_passes_without_data_or_fits() -> None:
    report = build_geometry_report(_load_config(CONFIG))
    assert report["status"] == "pass"
    assert report["fixture_count"] == report["passed_fixture_count"] == len(FIXTURE_NAMES)
    assert report["failed_fixture_count"] == 0
    assert report["model_fit_count"] == report["encoder_fit_count"] == 0
    assert report["target_cohort_loaded"] is False
    assert report["external_dataset_loaded"] is False
    assert report["physical_measurement_loaded"] is False
    assert report["automatic_follow_on_launched"] is False
    assert report["claims"] == {
        "software_algebra_qualified": True,
        "physical_identifiability_demonstrated": False,
        "reattachment_robustness_demonstrated": False,
        "model_performance_demonstrated": False,
        "architecture_promoted": False,
    }


def test_public_inference_api_accepts_no_query_labels_or_motion_stratum() -> None:
    forbidden = {
        "label",
        "labels",
        "activity",
        "true_activity",
        "motion_stratum",
        "participant_id",
    }
    for function in (directional_reference_evidence, compose_same_attachment_probabilities):
        assert forbidden.isdisjoint(inspect.signature(function).parameters)
    assert (
        inspect.signature(compose_same_attachment_probabilities).parameters["query_valid"].default
        is inspect.Parameter.empty
    )


def test_config_rejects_nonfinite_numerics(tmp_path: Path) -> None:
    with CONFIG.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    value["numerics"]["norm_epsilon"] = math.nan
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(GeometryQualificationError, match="must be positive"):
        _load_config(invalid)


def test_create_only_package_and_read_only_replay(tmp_path: Path) -> None:
    output = tmp_path / "geometry-run"
    result = run_geometry_qualification(
        repository_root=ROOT,
        config_path=CONFIG,
        protocol_path=PROTOCOL,
        governing_spec_path=PROTOCOL,
        synthesis_path=PROTOCOL,
        output_directory=output,
    )
    assert result["status"] == "complete_read_only_revalidation"
    assert result["fixture_count"] == result["passed_fixture_count"] == len(FIXTURE_NAMES)
    assert result["model_fit_count"] == result["validation_fit_count"] == 0
    completion = load_json_strict(output / "completion_manifest.json")
    assert completion["status"] == "complete"
    assert completion["automatic_follow_on_launched"] is False
    assert not (output / "INCOMPLETE.json").exists()
    assert validate_geometry_qualification(output) == result
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_geometry_qualification(
            repository_root=ROOT,
            config_path=CONFIG,
            protocol_path=PROTOCOL,
            governing_spec_path=PROTOCOL,
            synthesis_path=PROTOCOL,
            output_directory=output,
        )


def test_validator_rejects_unmanifested_file(tmp_path: Path) -> None:
    output = tmp_path / "geometry-run"
    run_geometry_qualification(
        repository_root=ROOT,
        config_path=CONFIG,
        protocol_path=PROTOCOL,
        governing_spec_path=PROTOCOL,
        synthesis_path=PROTOCOL,
        output_directory=output,
    )
    (output / "unmanifested.txt").write_text("must fail closed\n", encoding="utf-8")
    with pytest.raises(GeometryQualificationError, match="inventory"):
        validate_geometry_qualification(output)


def test_final_replay_failure_is_retained_as_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "geometry-run"

    def fail_replay(run_directory: Path) -> dict[str, object]:
        del run_directory
        raise GeometryQualificationError("forced final replay failure")

    monkeypatch.setattr(runner, "validate_geometry_qualification", fail_replay)
    with pytest.raises(GeometryQualificationError, match="forced final replay failure"):
        runner.run_geometry_qualification(
            repository_root=ROOT,
            config_path=CONFIG,
            protocol_path=PROTOCOL,
            governing_spec_path=PROTOCOL,
            synthesis_path=PROTOCOL,
            output_directory=output,
        )
    incomplete = load_json_strict(output / "INCOMPLETE.json")
    assert incomplete["status"] == "incomplete"
    assert incomplete["model_fit_count"] == incomplete["encoder_fit_count"] == 0
    assert incomplete["workers_remaining"] == incomplete["monitors_remaining"] == 0
