"""Fail-closed validation for the zero-shot target seal and future unlock record."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class TargetSealError(PermissionError):
    """Raised when a target access request lacks a valid one-time unlock."""


REQUIRED_FINAL_GATE_FIELDS = frozenset(
    {
        "artifact_validation_passed",
        "configuration_validation_passed",
        "manifest_validation_passed",
        "protocol_lock_present",
        "split_audit_passed",
        "tests_passed",
        "type_checks_passed",
        "working_tree_clean_or_documented",
    }
)


def validate_target_unlock_record(
    unlock_record: Mapping[str, Any] | None,
    *,
    split_manifest_sha256: str,
    target_seal_id: str,
) -> Mapping[str, Any]:
    """Validate authorization only; this function never reads target data or predictions."""

    if unlock_record is None:
        raise TargetSealError("zero-shot target remains sealed: no unlock record supplied")
    required = {
        "schema_version": "1.0.0",
        "gate": "final_evaluation_unlock",
        "status": "approved",
        "split_manifest_sha256": split_manifest_sha256,
        "target_seal_id": target_seal_id,
        "target_opening_number": 1,
        "target_performance_previously_accessed": False,
        "reason": "locked_confirmatory_evaluation",
    }
    mismatches = [key for key, value in required.items() if unlock_record.get(key) != value]
    if mismatches:
        raise TargetSealError(f"target unlock record mismatch: {mismatches}")
    if not isinstance(unlock_record.get("approved_at_utc"), str):
        raise TargetSealError("target unlock record lacks approved_at_utc")
    if not isinstance(unlock_record.get("code_commit"), str) or not unlock_record["code_commit"]:
        raise TargetSealError("target unlock record lacks code_commit")
    protocol_lock_sha256 = unlock_record.get("protocol_lock_sha256")
    if not isinstance(protocol_lock_sha256, str) or len(protocol_lock_sha256) != 64:
        raise TargetSealError("target unlock record lacks protocol_lock_sha256")
    gates = unlock_record.get("final_gates")
    if not isinstance(gates, Mapping) or set(gates) != REQUIRED_FINAL_GATE_FIELDS:
        raise TargetSealError("target unlock final_gates fields differ from the locked contract")
    failing = sorted(key for key, value in gates.items() if value is not True)
    if failing:
        raise TargetSealError(f"target unlock has unpassed final gates: {failing}")
    return unlock_record
