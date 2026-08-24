from __future__ import annotations

from pathlib import Path

from inclusive_shift_har.evaluation.reporting import build_locked_target_publication_report
from inclusive_shift_har.manifests.canonical import load_json_strict
from tests.test_locked_statistics import SEEDS, _fixture


def test_locked_report_builds_create_only_publication_tables(tmp_path: Path) -> None:
    from inclusive_shift_har.evaluation.locked_statistics import (
        aggregate_locked_target_statistics,
    )

    index_path = _fixture(tmp_path)
    aggregate_locked_target_statistics(
        index_path,
        output_root=tmp_path,
        destination="statistics.json",
        required_seed_order=SEEDS,
        candidate_model_id="candidate",
        eligible_reference_model_ids=("reference-a", "reference-b"),
        source_noninferiority_gate_passed=True,
        analysis_plan_sha256="a" * 64,
        bootstrap_resamples=100,
        bootstrap_seed=7,
    )
    report = build_locked_target_publication_report(
        index_path,
        "statistics.json",
        output_root=tmp_path,
        destination="report.json",
        model_table_csv="model.csv",
        model_table_markdown="model.md",
        participant_table_csv="participants.csv",
        comparison_table_csv="comparisons.csv",
    )
    assert report["hypothesis_decision"]["supported"] is True
    assert len(report["model_rows"]) == 3
    assert report["model_rows"][0]["model_id"] == "candidate"
    assert (
        (tmp_path / "model.md")
        .read_text(encoding="utf-8")
        .startswith("# Locked zero-shot target results")
    )
    assert load_json_strict(tmp_path / "report.json")["record_sha256"] == report["record_sha256"]


def test_locked_report_refuses_existing_outputs(tmp_path: Path) -> None:
    from inclusive_shift_har.evaluation.locked_statistics import (
        aggregate_locked_target_statistics,
    )

    index_path = _fixture(tmp_path)
    aggregate_locked_target_statistics(
        index_path,
        output_root=tmp_path,
        destination="statistics.json",
        required_seed_order=SEEDS,
        candidate_model_id="candidate",
        eligible_reference_model_ids=("reference-a", "reference-b"),
        source_noninferiority_gate_passed=True,
        analysis_plan_sha256="a" * 64,
        bootstrap_resamples=100,
    )
    (tmp_path / "model.csv").write_text("preserve\n", encoding="utf-8")
    try:
        build_locked_target_publication_report(
            index_path,
            "statistics.json",
            output_root=tmp_path,
            destination="report.json",
            model_table_csv="model.csv",
            model_table_markdown="model.md",
            participant_table_csv="participants.csv",
            comparison_table_csv="comparisons.csv",
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing output was overwritten")
    assert (tmp_path / "model.csv").read_text(encoding="utf-8") == "preserve\n"
