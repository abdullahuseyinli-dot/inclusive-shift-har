"""Shared, synthetic-only test policy.

The test suite must remain runnable without downloading any research dataset.
Network access is denied after test collection, and common third-party clients are
put into offline mode for both in-process checks and CLI subprocesses.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from typing import NoReturn

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"

if SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(SOURCE_ROOT))


OFFLINE_ENVIRONMENT = {
    "INCLUSIVE_SHIFT_HAR_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "WANDB_MODE": "offline",
    "WANDB_DISABLED": "true",
}

for _name, _value in OFFLINE_ENVIRONMENT.items():
    os.environ.setdefault(_name, _value)


def _network_denied(*_args: object, **_kwargs: object) -> NoReturn:
    raise RuntimeError("network access is forbidden in synthetic CI tests")


@pytest.fixture(autouse=True)
def deny_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any accidental in-process network access immediately."""

    monkeypatch.setattr(socket, "create_connection", _network_denied)
    monkeypatch.setattr(socket.socket, "connect", _network_denied)


@pytest.fixture
def repository_root() -> Path:
    return REPOSITORY_ROOT


@pytest.fixture
def offline_subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(OFFLINE_ENVIRONMENT)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(SOURCE_ROOT), existing_pythonpath) if part
    )
    return environment
