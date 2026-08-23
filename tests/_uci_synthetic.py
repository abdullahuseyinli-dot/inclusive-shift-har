"""Synthetic UCI-HAR ZIP builder; no third-party bytes or network access."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.data.uci_har import UCI_HAR_ACTIVITY_NAMES, UCI_HAR_CHANNELS


def _matrix_text(values: NDArray[np.float32]) -> str:
    output = io.StringIO()
    np.savetxt(output, values, fmt="%.7f")
    return output.getvalue()


def _vector_text(values: NDArray[np.int64]) -> str:
    output = io.StringIO()
    np.savetxt(output, values, fmt="%d")
    return output.getvalue()


def materialize_synthetic_uci_archive(
    path: Path,
    *,
    train_subjects: tuple[int, ...] = (1, 2, 3, 4, 5),
    test_subjects: tuple[int, ...] = (6, 7),
) -> Path:
    """Write a tiny schema-faithful processed UCI-HAR archive."""

    path.parent.mkdir(parents=True, exist_ok=True)
    root = "UCI HAR Dataset"
    label_table = "".join(
        f"{label_id} {name}\n" for label_id, name in UCI_HAR_ACTIVITY_NAMES.items()
    )
    readme = (
        "Synthetic fixture only.\n"
        "License fixture: publication acknowledgement; commercial-use restriction.\n"
    )
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{root}/activity_labels.txt", label_table)
        archive.writestr(f"{root}/README.txt", readme)
        for split, subjects in (("train", train_subjects), ("test", test_subjects)):
            activity_ids = np.tile(np.arange(1, 7, dtype=np.int64), len(subjects))
            subject_ids = np.repeat(np.asarray(subjects, dtype=np.int64), 6)
            window_count = activity_ids.size
            base = np.arange(window_count * 128, dtype=np.float32).reshape(window_count, 128)
            for channel_index, channel in enumerate(UCI_HAR_CHANNELS):
                values = base * np.float32(0.001) + np.float32(channel_index)
                archive.writestr(
                    f"{root}/{split}/Inertial Signals/{channel}_{split}.txt",
                    _matrix_text(values),
                )
            archive.writestr(f"{root}/{split}/y_{split}.txt", _vector_text(activity_ids))
            archive.writestr(f"{root}/{split}/subject_{split}.txt", _vector_text(subject_ids))
    return path
