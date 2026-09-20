from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from inclusive_shift_har.data import aicos_har


def _sensor(norm: float, *, samples: int = 200) -> bytes:
    timestamps = np.arange(samples, dtype=np.float64) * 20_000_000.0
    values = np.zeros((samples, 4), dtype=np.float64)
    values[:, 0] = timestamps
    values[:, 3] = norm
    return "\n".join(",".join(f"{item:.9f}" for item in row) for row in values).encode()


def _archive(path: Path) -> None:
    records = [
        ("S1/Sitting_1/PhoneA_Leg", 9.80665),
        ("S2/Standing_1/PhoneB_Hand", 1.0),
        ("S3/Walking_1/PhoneC_Hip", 9.80665),
        ("S4/Walking_1/PhoneD_Arm", 3.0),
    ]
    metadata = pd.DataFrame(
        [
            {
                "AcquisitionID": acquisition,
                "Age": "30-39",
                "Gender": "Female",
                "Position_FullName": acquisition.rsplit("_", 1)[-1],
                "Fold": "test",
            }
            for acquisition, _norm in records
        ]
    )
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("AICOS-HAR/metadata.csv", metadata.to_csv(index=False))
        for acquisition, norm in records:
            archive.writestr(f"AICOS-HAR/{acquisition}/Accelerometer.txt", _sensor(norm))
            archive.writestr(f"AICOS-HAR/{acquisition}/Gyroscope.txt", _sensor(0.0))


def test_aicos_adapter_harmonizes_units_and_quarantines_ambiguity(
    tmp_path: Path, monkeypatch: object
) -> None:
    archive_path = tmp_path / "AICOS-HAR.zip"
    _archive(archive_path)
    payload = archive_path.read_bytes()
    monkeypatch.setattr(aicos_har, "AICOS_ARCHIVE_SIZE_BYTES", len(payload))  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        aicos_har, "AICOS_ARCHIVE_MD5", hashlib.md5(payload).hexdigest()
    )

    verification = aicos_har.verify_aicos_archive(archive_path)
    audit = aicos_har.audit_aicos_archive(archive_path, folds=("test",))
    data = aicos_har.load_aicos_har(
        archive_path,
        verification=verification,
        folds=("test",),
    )

    assert audit["class_scores_calculated"] is False
    assert audit["status_counts"] == {"qualified": 3, "quarantined": 1}
    assert audit["unit_rule_counts"] == {
        "physical_g_band_0.75_to_1.25": 1,
        "physical_m_s2_band_6_to_16": 2,
    }
    assert data.signals.shape == (3, 128, 6)
    assert data.gravity.shape == (3, 128, 3)
    assert set(data.labels.tolist()) == {0, 1, 2}
    assert set(data.participant_ids.tolist()) == {"aicos:S1", "aicos:S2", "aicos:S3"}
    assert np.allclose(np.linalg.norm(data.gravity[:, 0], axis=1), 9.80665)
    assert any(item["reason"] == "signal_qualification_quarantine" for item in data.exclusions)
    assert data.source_storage_audit is not None
    assert data.source_storage_audit["archive_extracted"] is False

    # The adapter's SI output cannot go directly to the g-trained predictor.
    # Exercise both accepted raw representations and the observable pool too.
    from inclusive_shift_har.experiments.aicos_external_benchmark import (
        _frozen_predictor_probabilities,
    )
    from inclusive_shift_har.models.gravity_posture_reference import gravity_reference_time_series

    with pytest.raises(ValueError, match="requires acceleration/gravity in g"):
        _frozen_predictor_probabilities(data, tmp_path / "must-not-be-opened.pkl")
    assert data.observable_candidates is not None
    dynamic = data.signals.copy()
    dynamic[:, :, 0] = 0.5 * aicos_har.STANDARD_GRAVITY_M_S2
    dynamic[:, :, 3] = 0.25  # radians/second must not be scaled with acceleration
    si_data = replace(
        data,
        signals=dynamic,
        observable_candidates=replace(data.observable_candidates, signals=dynamic.copy()),
    )
    converted = aicos_har.aicos_in_inclusivehar_units(si_data)
    assert np.allclose(converted.signals[:, :, 0], 0.5)
    assert np.allclose(np.linalg.norm(converted.gravity, axis=2), 1.0)
    assert np.array_equal(converted.signals[:, :, 3:], si_data.signals[:, :, 3:])
    assert converted.observable_candidates is not None
    assert np.array_equal(converted.observable_candidates.signals, converted.signals)
    assert np.array_equal(converted.observable_candidates.gravity, converted.gravity)
    assert np.allclose(si_data.signals[:, :, 0], 0.5 * aicos_har.STANDARD_GRAVITY_M_S2)
    assert np.allclose(np.linalg.norm(si_data.gravity, axis=2), 9.80665)
    assert converted.window_ids is si_data.window_ids
    assert converted.labels is si_data.labels
    assert converted.participant_ids is si_data.participant_ids

    # One g along z plus 0.5 g horizontal motion must yield one g parallel
    # total acceleration, 0.5 g perpendicular, and 0.25 rad/s gyro norm.
    physics = gravity_reference_time_series(
        converted.signals, converted.gravity, sampling_rate_hz=50.0
    )
    assert np.allclose(physics[:, :, 13], 1.0)
    assert np.allclose(physics[:, :, 14], 0.5)
    assert np.allclose(physics[:, :, 17], 0.25)
    with pytest.raises(ValueError, match="declared SI"):
        aicos_har.aicos_in_inclusivehar_units(converted)
    with pytest.raises(ValueError, match="declared SI"):
        aicos_har.aicos_in_inclusivehar_units(replace(si_data, cohort_audit={}))
