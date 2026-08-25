from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inclusive_shift_har.manifests.canonical import canonical_json_sha256


def test_v015_bundle_scan_failure_is_preserved_without_secret_material(
    repository_root: Path,
) -> None:
    path = repository_root / "results/release/failures/benchmark-v0.1.5-bundle-gitleaks.json"
    payload = path.read_text(encoding="utf-8")
    record: dict[str, Any] = json.loads(payload)
    record_sha256 = record.pop("record_sha256")

    assert record_sha256 == canonical_json_sha256(record)
    assert record["status"] == "failed_preserved_quarantined_unreleased"
    assert record["candidate"]["commit"] == ("eaa30d18ca60b3c123d1ccf9b095d8d78a03469d")
    assert record["candidate"]["tag"]["object_id"] == ("df8e681ced4cee4cdcdae67eb18f1aaa835410cb")
    assert record["ci"] == {
        "conclusion": "success",
        "head_commit": "eaa30d18ca60b3c123d1ccf9b095d8d78a03469d",
        "run_id": 32836567358,
    }
    assert record["failure"]["finding_count"] == 3
    assert sum(item["count"] for item in record["failure"]["findings"]) == 3
    assert all(item["secret_value_recorded"] is False for item in record["failure"]["findings"])
    assert record["release_state"] == {
        "assets_uploaded": False,
        "github_release_created": False,
        "released": False,
    }
    assert record["research_state"] == {
        "confirmatory_target_reopened": False,
        "research_results_changed": False,
    }
    assert "C:\\Users" not in payload and "C:/Users" not in payload
    assert '"secret":' not in payload and '"match":' not in payload


def test_v016_draft_utf8_failure_is_preserved_without_external_paths(
    repository_root: Path,
) -> None:
    path = repository_root / "results/release/failures/benchmark-v0.1.6-draft-utf8-validation.json"
    payload = path.read_text(encoding="utf-8")
    record: dict[str, Any] = json.loads(payload)
    record_sha256 = record.pop("record_sha256")

    assert record_sha256 == canonical_json_sha256(record)
    assert record["status"] == "failed_preserved_draft_unpublished"
    assert record["candidate"] == {
        "commit": "968a6e72d68d624e5287f8b8c7f9accc84cf26de",
        "tag": {
            "name": "benchmark-v0.1.6",
            "object_id": "e1d1de6aea68234def75eb8d57fb262ef6539db0",
            "target_commit": "968a6e72d68d624e5287f8b8c7f9accc84cf26de",
        },
    }
    assert record["ci"]["run_id"] == 32846091138
    assert record["ci"]["conclusion"] == "success"
    assert record["failure"] == {
        "classification": "powershell_5_default_utf8_decoding_false_negative",
        "default_get_content_body_equal": False,
        "first_expected_code_point": "U+2014",
        "first_misdecoded_code_point": "U+00E2",
        "proper_utf8_body_equal": True,
        "stage": "draft_release_metadata_validation",
        "tracked_runbook_affected": True,
        "utf8_body_sha256": ("709be90be1ac885d76415b9c1f851e4f322e41392eeb472b483056efdcfbaad0"),
    }
    assert record["release_state"] == {
        "asset_count": 2,
        "draft": True,
        "github_release_created": True,
        "prerelease": True,
        "published": False,
        "release_id": 376392743,
    }
    assert record["security_gate"] == {
        "authoritative_bundle_finding_count": 0,
        "default_only_validated_hash_finding_count": 2,
        "prohibited_repository_field_present": False,
    }
    assert record["research_state"] == {
        "confirmatory_target_reopened": False,
        "research_results_changed": False,
    }
    assert "C:\\Users" not in payload and "C:/Users" not in payload
