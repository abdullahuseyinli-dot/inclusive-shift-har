"""InclusiveShift-HAR research-software package."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("inclusive-shift-har")
except PackageNotFoundError:  # Source tree used without an installed distribution.
    __version__ = "0.1.5a0"

__all__ = ["__version__", "main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Delegate to the evidence-gated command-line interface."""

    from inclusive_shift_har.cli import main as cli_main

    return cli_main(argv)
