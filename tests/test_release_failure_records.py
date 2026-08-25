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
