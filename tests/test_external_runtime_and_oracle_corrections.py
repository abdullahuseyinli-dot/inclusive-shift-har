from __future__ import annotations

import json
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from inclusive_shift_har.data import external_har
from inclusive_shift_har.experiments import cross_dataset_neural as neural


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_neural_backend_setting_is_explicit_and_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, disabled: bool, fail: bool
) -> None:
    before = torch.backends.cudnn.enabled

    def implementation(*_args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        assert torch.backends.cudnn.enabled is not disabled
        assert kwargs["config"].mixed_precision == "float16"
        if fail:
            raise RuntimeError("synthetic training failure")
        return np.eye(3), {"disable_cudnn": disabled}

    monkeypatch.setattr(neural, "_train_fold_impl", implementation)
    kwargs: dict[str, Any] = {
        "training_indices": np.array([0]),
        "validation_indices": np.array([1]),
        "evaluation_indices": np.array([2]),
        "config": neural.ExternalNeuralConfig("deepconvlstm", seed=11, disable_cudnn=disabled),
        "output_path": tmp_path / "unused.pt",
        "device": torch.device("cpu"),
        "class_names": ("mobility", "sitting", "standing"),
    }
    if not disabled:
        with pytest.raises(ValueError, match="frozen CUDA protocol"):
            neural._train_fold(None, np.zeros((3, 128, 6)), **kwargs)  # type: ignore[arg-type]
    elif fail:
        with pytest.raises(RuntimeError, match="synthetic training failure"):
            neural._train_fold(None, np.zeros((3, 128, 6)), **kwargs)  # type: ignore[arg-type]
    else:
        _, record = neural._train_fold(None, np.zeros((3, 128, 6)), **kwargs)  # type: ignore[arg-type]
        assert record["disable_cudnn"] == disabled
    assert torch.backends.cudnn.enabled is before


def _mock_sole(monkeypatch: pytest.MonkeyPatch, labels: np.ndarray[Any, Any]) -> None:
    prefix = "C001/session"
    metadata = json.dumps(
        {
            "recording": {"sampling_hz": 270},
            "units": {"acc": "m/s^2", "gyr": "rad/s"},
        }
    ).encode()
    payloads = {f"{prefix}/meta.json": metadata, f"{prefix}/DataStruct.mat": b"synthetic-mat"}

    class Archive:
        def __init__(self, _url: str) -> None:
            pass

        def __enter__(self) -> Archive:
            return self

        def __exit__(self, *_args: Any) -> None:
            pass

        def namelist(self) -> list[str]:
            return list(payloads)

        def getinfo(self, name: str) -> SimpleNamespace:
            return SimpleNamespace(file_size=len(payloads[name]), CRC=zlib.crc32(payloads[name]))

        def read(self, name: str) -> bytes:
            return payloads[name]

    monkeypatch.setattr(external_har, "RemoteZip", Archive)
    monkeypatch.setattr(
        external_har,
        "_request_json",
        lambda *_args: {
            "files": [
                {"key": "C001.zip", "links": {"content": "https://example.invalid/C001.zip"}}
            ],
        },
    )
    timestamps = np.arange(8100, dtype=np.float64) * (1000.0 / 270)
    values = np.column_stack(
        (np.sin(timestamps / 100), np.cos(timestamps / 100), timestamps / 10000)
    )
    monkeypatch.setattr(
        external_har,
        "_sole_session_arrays",
        lambda _payload: (
            timestamps,
            values,
            values / 100,
            values + np.array([0, 0, 9.8]),
            labels,
        ),
    )


def test_zero_duration_unknown_omission_preserves_positive_bout_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = np.array([[0, 0, 10000], [1, 10000, 20000], [2, 20000, 30000]], dtype=float)
    _mock_sole(monkeypatch, labels)
    before = external_har.load_sole_harmony(participant_limit=1, sessions_per_participant=1)
    _mock_sole(monkeypatch, np.vstack((labels, [-1, 5000, 5000])))
    after = external_har.load_sole_harmony(participant_limit=1, sessions_per_participant=1)
    np.testing.assert_array_equal(before.signals, after.signals)
    np.testing.assert_array_equal(before.gravity, after.gravity)
    np.testing.assert_array_equal(before.labels, after.labels)
    assert len(after.source_issues) == 1
    issue = after.source_issues[0]
    assert issue["excluded_duration_ms"] == 0 and issue["source_label"] == -1
    assert issue["stable_sorted_interval_index"] == 1
    assert after.summary()["window_count"] == before.summary()["window_count"]


@pytest.mark.parametrize(
    "bad",
    [
        [0, 5000, 5000],
        [-1, 5000, 4999],
        [-1, np.nan, 5000],
        [0.5, 5000, 5001],
        [1, 5000, 11000],
    ],
)
def test_oracle_correction_does_not_relax_material_annotation_ambiguity(
    monkeypatch: pytest.MonkeyPatch, bad: list[float]
) -> None:
    labels = np.array([[0, 0, 10000], [1, 10000, 20000], [2, 20000, 30000], bad], dtype=float)
    _mock_sole(monkeypatch, labels)
    with pytest.raises(ValueError, match="camera"):
        external_har.load_sole_harmony(participant_limit=1, sessions_per_participant=1)
