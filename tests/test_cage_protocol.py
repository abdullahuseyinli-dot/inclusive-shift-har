from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inclusive_shift_har.experiments.cage_har import _load_config
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file


def test_cage_protocol_is_self_hashed_and_all_lineage_matches() -> None:
    path = Path("results/protocol/cage_har_v1.json")
    record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    claimed = record.pop("record_sha256")
    assert claimed == canonical_json_sha256(record)

    def validate_references(value: object) -> None:
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                referenced = Path(str(value["path"]))
                assert referenced.is_file(), referenced
                assert sha256_file(referenced) == value["sha256"]
            for child in value.values():
                validate_references(child)
        elif isinstance(value, list):
            for child in value:
                validate_references(child)

    validate_references(record)
    boundary = record["evidence_boundary"]
    assert boundary["real_cage_har_performance_generated"] is False
    assert boundary["confirmatory_claim_allowed"] is False
    assert boundary["new_development_cohort_required"] is True


def test_cage_config_and_protocol_have_matching_locked_gates() -> None:
    config = _load_config(Path("configs/experiments/cage_har_v1.yaml"))
    gate = config["development_advancement_gate"]
    assert gate["minimum_mean_participant_macro_f1_difference"] == 0.020
    assert gate["minimum_intervention_precision"] == 0.80
    assert gate["maximum_harmful_changed_fraction"] == 0.15
    assert gate["targets_are_promises"] is False
    assert config["claim_policy"]["locked_inclusivehar_reuse_allowed"] is False


def test_cage_implementation_summary_is_self_hashed_and_claim_safe() -> None:
    path = Path("results/development/cage_har_v1_implementation_summary.json")
    record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    claimed = record.pop("record_sha256")
    assert claimed == canonical_json_sha256(record)
    protocol = record["protocol"]
    assert sha256_file(Path(protocol["path"])) == protocol["sha256"]
    smoke = record["synthetic_contract_smoke"]
    smoke_path = Path(smoke["path"])
    if smoke_path.exists():
        assert sha256_file(smoke_path) == smoke["sha256"]
    assert smoke["real_har_performance_claim_allowed"] is False
    assert record["human_data_result_status"]["new_cage_har_score_available"] is False
    assert record["claim_boundary"]["state_of_the_art_claim_allowed"] is False
