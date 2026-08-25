from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inclusive_shift_har.artifacts.release_inventory import (
    ARTIFACT_ROLES,
    POSTCONFIRMATORY_TRACKS,
    REQUIRED_ROLES,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_release_inventory_schema_keeps_consumed_target_and_required_roles(
    repository_root: Path,
) -> None:
    schema = _load(repository_root / "configs/schema/release_evidence_inventory.schema.json")
    assert schema["properties"]["schema_version"]["const"] == "1.1.0"
    assert schema["properties"]["delivery_mode"]["const"] == "external_release_asset"
    assert schema["properties"]["tracked"]["const"] is False
    confirmatory = schema["properties"]["confirmatory_state"]["properties"]
    assert confirmatory["opening_count"]["const"] == 1
    assert confirmatory["target_rerun_permitted"]["const"] is False
    required_roles = schema["properties"]["required_roles"]["const"]
    assert required_roles == list(REQUIRED_ROLES)
    assert len(required_roles) == len(set(required_roles))
    assert {
        "legacy_audit",
        "literature_matrix",
        "dataset_manifest",
        "locked_protocol",
        "opening_receipt",
        "locked_target_index",
        "participant_statistics",
        "baseline_coverage",
        "release_gate_report",
    } <= set(required_roles)
    artifact_roles = set(schema["$defs"]["artifact"]["properties"]["role"]["enum"])
    assert artifact_roles == ARTIFACT_ROLES
    tracks = schema["$defs"]["postconfirmatory_status"]["properties"]["track"]["enum"]
    assert tracks == list(POSTCONFIRMATORY_TRACKS)
    postconfirmatory = schema["properties"]["postconfirmatory_statuses"]
    assert postconfirmatory["minItems"] == len(POSTCONFIRMATORY_TRACKS)
    assert postconfirmatory["maxItems"] == len(POSTCONFIRMATORY_TRACKS)


def test_release_inventory_runbook_uses_create_only_generator(repository_root: Path) -> None:
    text = (repository_root / "docs/RELEASE_EVIDENCE_GATE.md").read_text(encoding="utf-8")
    assert "inclusive_shift_har.artifacts.release_inventory generate" in text
    assert "inclusive_shift_har.artifacts.release_inventory validate" in text
    assert "--require-clean-worktree" in text
    assert "--candidate-commit $candidate" in text
    assert "Neither operation loads sensor arrays or raw target recordings" in text


def test_release_runbook_fail_closes_outer_bundle_and_draft_publication(
    repository_root: Path,
) -> None:
    text = (repository_root / "docs/RELEASE_EVIDENCE_GATE.md").read_text(encoding="utf-8")
    for expected in (
        "inclusive_shift_har.artifacts.release_local_evidence",
        "capture-local-gates",
        "attest-staged-gitleaks",
        "inclusive_shift_har.artifacts.release_bundle",
        "standard-spec",
        "candidate-bound release-notes assembly",
        "outer-bundle manifest assembly",
        "outer-bundle build",
        "outer-bundle offline reconstruction",
        "Outer-bundle Gitleaks scan failed",
        "assets.Count -ne 2",
        "gh release edit benchmark-v0.1.3",
        "--draft=false --prerelease --latest=false",
        "published-release-api.json",
        "Published downloaded asset differs",
        "Repository is no longer private",
        "Remote annotated tag identity differs",
    ):
        assert expected in text
    assert 'Save-GhApiResponse "repos/$repository/releases?per_page=100" $draftApi' in text
    assert (
        'Save-GhApiResponse "repos/$repository/releases/tags/benchmark-v0.1.3" $publishedApi'
        in text
    )
    assert "Set-Content -LiteralPath $draftApi" not in text
    assert "Set-Content -LiteralPath $publishedApi" not in text


def test_data_acquisition_runbook_pins_official_layout(repository_root: Path) -> None:
    text = (repository_root / "docs/DATA_ACQUISITION_RUNBOOK.md").read_text(encoding="utf-8")
    for expected in (
        "10.17632/r78dn3f6nc.4",
        "0209542286e9031dc64a65abb47e1d84676088b1a77e52f738635887cb9bce34",
        "b8c033dd630412103b8dd463c8a171aa4727fc7222ba6b7fdc1e2487315e0dfb",
        "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031",
        "2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0",
        "eaee53f45825f349681a1a1b92a997a0082648f9b5bf93ddb3290caeede4b3a8",
        "do_not_redistribute_raw",
        "failure",
    ):
        assert expected in text


def test_experiment_runbook_does_not_offer_target_rerun(repository_root: Path) -> None:
    text = (repository_root / "docs/EXPERIMENT_RUNBOOK.md").read_text(encoding="utf-8")
    assert "Do not invoke, reconstruct,\nor rerun target opening 1" in text
    assert "primary_channels_opening1_v1" in text
    assert 'ToString("yyyy-MM-ddTHH:mm:ssZ")' in text
    assert "3 models" in text
    assert "75 first-attempt CUDA" in text
    assert "completed, self-hashed v1.1 report" in text
    assert "1,200 sequential" in text


def test_cnn_har_is_explicitly_an_omitted_non_proxy_baseline(repository_root: Path) -> None:
    text = (repository_root / "docs/baselines/BASELINE_COVERAGE_AND_OMISSIONS.md").read_text(
        encoding="utf-8"
    )
    assert "No faithful CNN-HAR implementation exists" in text
    assert "is **not CNN-HAR**" in text
    assert "Missing baselines" in text
    assert "not zero-valued results" in text
