from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from inclusive_shift_har.data import external_har as data


def _sources() -> tuple[data._IMUSourceFile, ...]:
    return tuple(
        data._IMUSourceFile(
            participant=person,
            repetition="Repetition_1",
            activity=activity,
            file_id=index,
            filename=f"{person}/Repetition_1/{activity}/Body-WT.csv",
            file_size=0,
            locator=f"https://example.invalid/{person}/{activity}",
        )
        for index, (person, activity) in enumerate(
            (person, activity)
            for person in ("P_01", "P_02")
            for activity in ("Walk", "Sit", "Stand")
        )
    )


def test_inventory_retains_existing_trials_without_inventing_missing_repetitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _sources()
    specs = [
        (s.participant, s.repetition, s.activity, s.filename.rsplit("/", 1)[0]) for s in sources
    ]
    monkeypatch.setattr(data, "_request_json", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(data, "_imu_folder_paths", lambda *_args, **_kwargs: specs)
    lookup = dict(zip(specs, sources, strict=True))
    monkeypatch.setattr(
        data,
        "_resolve_imu_file_optional",
        lambda spec: (spec, None if spec == specs[-1] else lookup[spec]),
    )
    common: dict[str, Any] = dict(
        participant_limit=None, repetition_limit=4, activities=("Walk", "Sit", "Stand"), workers=1
    )
    complete, omitted = data._imu_har_il_inventory(**common)
    available, missing = data._imu_har_il_inventory(**common, require_complete_core=False)
    assert len(complete) == 3 and len(available) == 5
    assert all(s.participant == "P_01" for s in complete)
    assert missing[0]["missing_folders"] == omitted[0]["missing_folders"]
    assert "other available trials retained" in missing[0]["action"]


@pytest.mark.parametrize("weakness", ["nonfinite_signal", "conflicting_label", "missing_label"])
def test_available_policy_localizes_quarantine_and_preserves_retained_signal_grids(
    monkeypatch: pytest.MonkeyPatch, weakness: str
) -> None:
    sources = _sources()
    monkeypatch.setattr(data, "_imu_har_il_inventory", lambda **_kwargs: (sources, ()))
    mode = "clean"

    def download(source: data._IMUSourceFile) -> tuple[data._IMUSourceFile, bytes]:
        rows = 320
        values = np.sin(np.arange(rows) / 20.0) + source.file_id
        frame = pd.DataFrame(
            {name: values.copy() for name in ("Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z")}
        )
        frame["Activity_label"] = {"Walk": 5, "Sit": 1, "Stand": 2}[source.activity]
        if source.participant == "P_02" and source.activity == "Walk":
            if mode == "nonfinite_signal":
                frame["Acc_X"] = np.nan
            elif mode == "conflicting_label":
                frame["Activity_label"] = 1
            elif mode == "missing_label":
                frame = frame.drop(columns=["Activity_label"])
        payload = frame.to_csv(index=False).encode()
        return replace(source, file_size=len(payload)), payload

    monkeypatch.setattr(data, "_download_imu_payload", download)
    reference = data.load_imu_har_il(
        selection_policy="available_valid_trials", download_workers=1, repetition_limit=4
    )
    mode = weakness
    available = data.load_imu_har_il(
        selection_policy="available_valid_trials", download_workers=1, repetition_limit=4
    )
    complete = data.load_imu_har_il(download_workers=1, repetition_limit=4)
    assert set(available.participant_ids) == {"imuharil:P_01", "imuharil:P_02"}
    audit = available.summary()["cohort_audit"]
    assert audit["requested_participant_count"] == audit["retained_participant_count"] == 2
    assert audit["protocol_id"] == "imu-har-il-available-trials-v1"
    if weakness == "missing_label":
        assert audit["quarantined_trial_count"] == 0
        np.testing.assert_array_equal(reference.window_ids, available.window_ids)
        np.testing.assert_array_equal(reference.labels, available.labels)
    else:
        assert audit["quarantined_trial_count"] == 1
        assert set(complete.participant_ids) == {"imuharil:P_01"}
        assert available.summary()["participants_with_all_classes"] == 1
        assert available.summary()[
            "perfect_prediction_mean_participant_macro_f1_ceiling"
        ] == pytest.approx(5 / 6)
    reference_index = {name: index for index, name in enumerate(reference.window_ids)}
    for index, name in enumerate(available.window_ids):
        old = reference_index[name]
        np.testing.assert_array_equal(available.signals[index], reference.signals[old])
        np.testing.assert_array_equal(available.gravity[index], reference.gravity[old])
    assert complete.cohort_audit is None


def test_unknown_selection_policy_fails_before_provider_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("invalid policy must not access the provider")

    monkeypatch.setattr(data, "_imu_har_il_inventory", forbidden)
    with pytest.raises(ValueError, match="selection policy"):
        data.load_imu_har_il(selection_policy="choose_highest_score")
