"""The target remains inaccessible without a complete one-time unlock record."""

from __future__ import annotations

import pytest

from inclusive_shift_har.protocols.seal import TargetSealError, validate_target_unlock_record


def test_target_seal_fails_closed_without_unlock() -> None:
    with pytest.raises(TargetSealError, match="remains sealed"):
        validate_target_unlock_record(
            None,
            split_manifest_sha256="a" * 64,
            target_seal_id="b" * 64,
        )


def test_target_unlock_requires_every_predeclared_gate() -> None:
    record = {
        "approved_at_utc": "2099-01-01T00:00:00Z",
        "code_commit": "synthetic-test-commit",
        "final_gates": {
            "artifact_validation_passed": True,
            "configuration_validation_passed": True,
            "manifest_validation_passed": True,
            "protocol_lock_present": True,
            "split_audit_passed": False,
            "tests_passed": True,
            "type_checks_passed": True,
            "working_tree_clean_or_documented": True,
        },
        "gate": "final_evaluation_unlock",
        "protocol_lock_sha256": "c" * 64,
        "reason": "locked_confirmatory_evaluation",
        "schema_version": "1.0.0",
        "split_manifest_sha256": "a" * 64,
        "status": "approved",
        "target_opening_number": 1,
        "target_performance_previously_accessed": False,
        "target_seal_id": "b" * 64,
    }
    with pytest.raises(TargetSealError, match="split_audit_passed"):
        validate_target_unlock_record(
            record,
            split_manifest_sha256="a" * 64,
            target_seal_id="b" * 64,
        )
