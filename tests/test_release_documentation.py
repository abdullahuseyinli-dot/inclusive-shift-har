from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from inclusive_shift_har.artifacts.release_inventory import (
    ARTIFACT_ROLES,
    POSTCONFIRMATORY_TRACKS,
    REQUIRED_ROLES,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _powershell() -> str | None:
    return shutil.which("powershell.exe") or shutil.which("powershell")


def _powershell_blocks(text: str) -> list[str]:
    return re.findall(r"```powershell\n(.*?)```", text, flags=re.DOTALL)


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
        "bundle-gitleaks-default-only.toml",
        "bundle-gitleaks-empty.ignore",
        "bundle-gitleaks-unreviewed.json",
        "--config $bundleBaselineConfig",
        "@($bundleFindings).Count -ne 2",
        "$expectedCiMatches.SetEquals($observedCiMatches)",
        "--gitleaks-ignore-path $bundleBaselineIgnore",
        "no dynamic fingerprint ignore is used",
        "Outer-bundle Gitleaks scan failed",
        "@($remainingBundleFindings).Count -ne 0",
        "assets.Count -ne 2",
        "git tag -a benchmark-v0.1.7 $candidate",
        "gh release create benchmark-v0.1.7",
        "gh release edit benchmark-v0.1.7",
        "--draft=false --prerelease --latest=false",
        "published-release-api.json",
        "Published downloaded asset differs",
        "Repository is no longer private",
        "Remote annotated tag identity differs",
        "32836567358",
        "df8e681ced4cee4cdcdae67eb18f1aaa835410cb",
        "eaa30d18ca60b3c123d1ccf9b095d8d78a03469d",
        "32846091138",
        "e1d1de6aea68234def75eb8d57fb262ef6539db0",
        "968a6e72d68d624e5287f8b8c7f9accc84cf26de",
        "all eleven tag objects/peeled targets",
        "Read-StrictUtf8Json",
        "[System.Text.UTF8Encoding]::new($false, $true)",
    ):
        assert expected in text
    assert 'Save-GhApiResponse "repos/$repository/releases?per_page=100" $draftApi' in text
    assert (
        'Save-GhApiResponse "repos/$repository/releases/tags/benchmark-v0.1.7" $publishedApi'
        in text
    )
    assert "-ArgumentList @('api', $Endpoint, '--jq', 'del(.temp_clone_token)')" in text
    assert text.count('Save-SanitizedRepositoryApiResponse "repos/$repository"') == 2
    assert 'Save-GhApiResponse "repos/$repository" "$external/repository-api.json"' not in text
    assert 'Save-GhApiResponse "repos/$repository" $finalRepositoryApi' not in text
    assert "[string]$finding.Secret -cne $expectedSecretHash" in text
    assert "$bundleFindings = @(" not in text
    assert "$remainingBundleFindings = @(" not in text
    assert "rule-local `AND` allowance" in text
    assert "'benchmark-v0.1.4', 'benchmark-v0.1.5', 'benchmark-v0.1.6'," in text
    assert "'benchmark-v0.1.7'" in text
    final_tag_block = text.split("$requiredTags = @(", 1)[1].split("\n)", 1)[0]
    assert re.findall(r"'([^']+)'", final_tag_block) == [
        "benchmark-v0.1.0",
        "benchmark-v0.1.1",
        "benchmark-v0.1.2",
        "benchmark-v0.1.3",
        "benchmark-v0.1.4",
        "benchmark-v0.1.5",
        "benchmark-v0.1.6",
        "benchmark-v0.1.7",
        "legacy-audit-v0.1.0",
        "protocol-v1.0.0",
        "protocol-v1.2.0",
    ]
    for obsolete_candidate_operation in (
        "git show-ref --verify --quiet refs/tags/benchmark-v0.1.6",
        "git tag -a benchmark-v0.1.6 $candidate",
        "$candidateTagProbe = @(git ls-remote origin refs/tags/benchmark-v0.1.6",
        "gh release create benchmark-v0.1.6",
        "gh release download benchmark-v0.1.6",
        "gh release edit benchmark-v0.1.6",
        "releases/tags/benchmark-v0.1.6",
        "$tag -ceq 'benchmark-v0.1.6' -and $localTarget -cne $candidate",
    ):
        assert obsolete_candidate_operation not in text
    assert '$toolRoot = ".audit/tools/$review/gitleaks-8.30.1-windows-x64"' in text
    assert "final eleven-tag policy scan" in text
    assert "Set-Content -LiteralPath $draftApi" not in text
    assert "Set-Content -LiteralPath $publishedApi" not in text
    for unsafe_saved_api_parse in (
        "Get-Content -Raw -LiteralPath $draftApi",
        "Get-Content -Raw -LiteralPath $publishedApi",
        "Get-Content -Raw -LiteralPath $finalRepositoryApi",
        "Get-Content -Raw -LiteralPath $finalMainApi",
        'Get-Content -Raw "$external/run-artifacts-api.json"',
    ):
        assert unsafe_saved_api_parse not in text
    for strict_saved_api_parse in (
        '$artifactList = Read-StrictUtf8Json "$external/run-artifacts-api.json"',
        "$draftMatches = @((Read-StrictUtf8Json $draftApi) |",
        "$published = Read-StrictUtf8Json $publishedApi",
        "$finalRepository = Read-StrictUtf8Json $finalRepositoryApi",
        "$finalMain = Read-StrictUtf8Json $finalMainApi",
    ):
        assert strict_saved_api_parse in text
    helper_block = next(
        block for block in _powershell_blocks(text) if "Read-StrictUtf8Json" in block
    )
    helper = helper_block[
        helper_block.index("function Read-StrictUtf8Json") : helper_block.index(
            "function Save-SanitizedRepositoryApiResponse"
        )
    ]
    assert "[System.Text.UTF8Encoding]::new($false, $true)" in helper


def test_release_runbook_powershell_blocks_parse_when_powershell_is_available(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    powershell = _powershell()
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    text = (repository_root / "docs/RELEASE_EVIDENCE_GATE.md").read_text(encoding="utf-8")
    blocks = _powershell_blocks(text)
    assert len(blocks) == 15
    parser = tmp_path / "parse.ps1"
    parser.write_text(
        "$tokens = $null\n"
        "$errors = $null\n"
        "[System.Management.Automation.Language.Parser]::ParseFile(\n"
        "  $args[0], [ref]$tokens, [ref]$errors\n"
        ") | Out-Null\n"
        "if ($errors.Count -ne 0) {\n"
        "  $errors | ForEach-Object { [Console]::Error.WriteLine($_.Message) }\n"
        "  exit 1\n"
        "}\n",
        encoding="ascii",
    )
    for index, block in enumerate(blocks):
        source = tmp_path / f"block-{index:02d}.ps1"
        source.write_text(block, encoding="utf-8")
        completed = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-File", str(parser), str(source)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr


def test_release_runbook_strict_utf8_reader_preserves_em_dash_on_windows(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    powershell = _powershell()
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    text = (repository_root / "docs/RELEASE_EVIDENCE_GATE.md").read_text(encoding="utf-8")
    helper_block = next(
        block for block in _powershell_blocks(text) if "Read-StrictUtf8Json" in block
    )
    helper = helper_block[
        helper_block.index("function Read-StrictUtf8Json") : helper_block.index(
            "function Save-SanitizedRepositoryApiResponse"
        )
    ]
    fixture = tmp_path / "utf8-release.json"
    script = tmp_path / "strict-utf8-regression.ps1"
    script.write_text(
        helper
        + "\n$expected = 'before ' + [char]0x2014 + ' after'\n"
        + "$payload = '{\"body\":\"' + $expected + '\"}'\n"
        + "[System.IO.File]::WriteAllText(\n"
        + "  $args[0], $payload, [System.Text.UTF8Encoding]::new($false, $true)\n"
        + ")\n"
        + "$decoded = Read-StrictUtf8Json $args[0]\n"
        + "if ($decoded.body -cne $expected) { throw 'Strict UTF-8 round trip failed' }\n",
        encoding="ascii",
    )
    completed = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-File", str(script), str(fixture)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_nccl_dependency_review_is_exact_date_only_and_policy_bound(repository_root: Path) -> None:
    review_path = repository_root / "docs/release/NVIDIA_NCCL_CU12_2_31_2_REVIEW.json"
    review = _load(review_path)
    record_hash = review.pop("record_sha256")
    assert record_hash == canonical_json_sha256(review)
    assert review["review_date"] == "2026-08-25"
    assert "reviewed_at_utc" not in review
    assert all(type(value) is bool for value in review["controls"].values())
    assert {item["rendered_content_version"] for item in review["official_terms"].values()} == {
        "2.29.2"
    }
    assert any("supplementary terms evidence" in item for item in review["review_findings"])

    policy = _load(repository_root / "configs/release/release_gate_policy_v1.json")
    refs = policy["git_refs"]
    assert refs["required_for_release_tags"] == [
        "benchmark-v0.1.0",
        "benchmark-v0.1.1",
        "benchmark-v0.1.2",
        "benchmark-v0.1.3",
        "benchmark-v0.1.4",
        "benchmark-v0.1.5",
        "benchmark-v0.1.6",
        "benchmark-v0.1.7",
        "legacy-audit-v0.1.0",
        "protocol-v1.0.0",
        "protocol-v1.2.0",
    ]
    assert refs["pinned_annotated_tags"]["benchmark-v0.1.5"] == {
        "object_id": "df8e681ced4cee4cdcdae67eb18f1aaa835410cb",
        "target_commit": "eaa30d18ca60b3c123d1ccf9b095d8d78a03469d",
        "message": "InclusiveShift-HAR benchmark v0.1.5",
        "tagger_name": "Abdulla Huseyinli",
        "tagger_email": "abdullahuseyinli@gmail.com",
    }
    assert refs["pinned_annotated_tags"]["benchmark-v0.1.6"] == {
        "object_id": "e1d1de6aea68234def75eb8d57fb262ef6539db0",
        "target_commit": "968a6e72d68d624e5287f8b8c7f9accc84cf26de",
        "message": "InclusiveShift-HAR benchmark v0.1.6",
        "tagger_name": "Abdulla Huseyinli",
        "tagger_email": "abdullahuseyinli@gmail.com",
    }
    assert refs["permitted_candidate_tags"] == {
        "benchmark-v0.1.7": {
            "message": "InclusiveShift-HAR benchmark v0.1.7",
            "tagger_name": "Abdulla Huseyinli",
            "tagger_email": "abdullahuseyinli@gmail.com",
        }
    }
    assert "32836567358" in refs["historical_notes"]["benchmark-v0.1.5"]
    assert "32846091138" in refs["historical_notes"]["benchmark-v0.1.6"]
    exception = policy["license_exception_records"]["nvidia-nccl-cu12@2.31.2"]
    assert exception["review_record_sha256"] == record_hash
    assert exception["review_sha256"] == hashlib.sha256(review_path.read_bytes()).hexdigest()
    assert (
        exception["lock_sha256"]
        == hashlib.sha256((repository_root / "uv.lock").read_bytes()).hexdigest()
    )


def test_prominent_documents_use_evidence_status_instead_of_generic_disclaimers(
    repository_root: Path,
) -> None:
    generic_heading = "## Claim " + "limits"
    for relative in (
        "README.md",
        "docs/PROJECT_STATUS.md",
        "docs/BENCHMARK_CARD.md",
        "docs/MODEL_CARD_MORE_HAR.md",
        "paper/OUTLINE.md",
    ):
        text = (repository_root / relative).read_text(encoding="utf-8")
        assert generic_heading not in text


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
