from __future__ import annotations

import json
from pathlib import Path

import yaml

from inclusive_shift_har.manifests.canonical import canonical_json_sha256


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _assert_self_hash(path: Path, field: str) -> None:
    payload = _load_json(path)
    recorded = payload.pop(field)
    assert recorded == canonical_json_sha256(payload)


def test_uci_machine_records_have_valid_canonical_hashes(repository_root: Path) -> None:
    _assert_self_hash(repository_root / "results/data_audit/uci_har_v1.audit.json", "report_sha256")
    _assert_self_hash(
        repository_root / "results/protocol/uci_har_source_grouped_v1.json",
        "protocol_sha256",
    )
    _assert_self_hash(
        repository_root / "results/acquisition/uci_har_v1.receipt.json", "receipt_sha256"
    )
    _assert_self_hash(
        repository_root / "results/gates/raw_data_read_access_uci_har_v1.json",
        "record_sha256",
    )
    _assert_self_hash(repository_root / "legacy/corrected_architecture_specs.json", "report_sha256")
    _assert_self_hash(
        repository_root / "results/legacy_reproduction/uci_har_v1.status.json",
        "report_sha256",
    )


def test_current_uci_manifest_matches_hash_named_history_copy(repository_root: Path) -> None:
    current_path = repository_root / "manifests/datasets/uci_har_v1.json"
    current = _load_json(current_path)
    manifest_hash = canonical_json_sha256(current)
    history_path = repository_root / f"manifests/history/uci_har_v1.{manifest_hash}.json"

    assert history_path.is_file()
    assert _load_json(history_path) == current
    config = yaml.safe_load(
        (repository_root / "configs/datasets/uci_har_v1.yaml").read_text(encoding="utf-8")
    )
    assert config["manifest_sha256"] == manifest_hash
