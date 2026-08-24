from __future__ import annotations

from pathlib import Path

import pytest

from inclusive_shift_har.experiments.confirmatory_target import (
    OPENING_ACKNOWLEDGEMENT,
    ConfirmatoryTargetError,
    build_parser,
    run_confirmatory_target_once,
)


def test_confirmatory_operation_requires_exact_ack_before_reading_any_path(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "must-not-be-read.json"
    with pytest.raises(ConfirmatoryTargetError, match="acknowledgement"):
        run_confirmatory_target_once(
            acknowledgement="wrong",
            repository_root=tmp_path,
            target_manifest_path=missing,
            unlock_record_path=missing,
            final_freeze_inventory_path=missing,
            source_selection_plan_path=missing,
            raw_csv_path=missing,
            machine_record_path=missing,
            receipt_root=tmp_path,
            output_directory=tmp_path / "target",
            output_root=tmp_path,
            opened_at_utc="2099-01-01T00:00:00Z",
        )


def test_confirmatory_parser_preserves_explicit_one_time_acknowledgement() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--acknowledge-one-time-opening",
            OPENING_ACKNOWLEDGEMENT,
            "--target-manifest",
            "target.json",
            "--unlock-record",
            "unlock.json",
            "--final-freeze-inventory",
            "freeze.json",
            "--source-selection-plan",
            "selection.json",
            "--raw-csv",
            "raw.csv",
            "--machine-record",
            "machine.json",
            "--receipt-root",
            "receipts",
            "--output-directory",
            "target-results",
            "--output-root",
            "results",
            "--opened-at-utc",
            "2099-01-01T00:00:00Z",
        ]
    )
    assert args.acknowledge_one_time_opening == OPENING_ACKNOWLEDGEMENT
