from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inclusive_shift_har.experiments.fog_motion_factorization_run import (
    EVIDENCE_FAMILY,
    RUN_DIRECTORY_NAME,
    extended_method_report,
    run_experiment,
    validate_run,
)


def test_conditional_posture_uses_frozen_l9v_when_motion_saturates() -> None:
    labels = np.asarray([0, 1, 2], dtype=np.int64)
    participants = np.asarray(["person", "person", "person"], dtype=np.str_)
    saturated_motion = np.asarray(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64
    )
    frozen_l9v = np.asarray([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], dtype=np.float64)

    report = extended_method_report(
        labels=labels,
        probabilities=saturated_motion,
        participants=participants,
        roster=["person"],
        conditional_posture_probabilities=frozen_l9v,
    )

    assert report["conditional_posture"] == {
        "sitting_positive": True,
        "probability_source": "provided_frozen_L9v_probabilities",
        "eligible_true_stationary_nonzero_q_rows": 2,
        "eligible_person_count": 1,
        "pooled_auc": 1.0,
        "mean_eligible_person_auc": 1.0,
    }


def test_existing_run_rejection_never_mutates_preserved_evidence(tmp_path: Path) -> None:
    evidence_root = tmp_path / "evidence"
    output = evidence_root / EVIDENCE_FAMILY / RUN_DIRECTORY_NAME
    output.mkdir(parents=True)
    marker = output / "preserved.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(ValueError, match="create-only run directory exists"):
        run_experiment(
            repository_root=tmp_path / "repository",
            evidence_root=evidence_root,
            config_path=tmp_path
            / "repository"
            / "configs"
            / "experiments"
            / "fog_motion_factorization_v1.yaml",
            protocol_path=tmp_path
            / "repository"
            / "docs"
            / "research"
            / "FOG_MOTION_FACTORIZATION_V1_PROTOCOL.md",
            output_directory=output,
            code_commit="unused",
            cancel_file=None,
        )

    assert marker.read_text(encoding="utf-8") == "preserve"
    assert sorted(path.name for path in output.iterdir()) == ["preserved.txt"]


def test_validation_rejection_never_mutates_unrecognized_directory(tmp_path: Path) -> None:
    marker = tmp_path / "preserved.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        validate_run(tmp_path)

    assert marker.read_text(encoding="utf-8") == "preserve"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["preserved.txt"]
