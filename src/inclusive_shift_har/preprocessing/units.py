"""Explicit inertial unit conversions used by declared sensitivity tracks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

STANDARD_GRAVITY_METRES_PER_SECOND_SQUARED = 9.80665


def acceleration_g_to_metres_per_second_squared(
    values: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Convert acceleration from standard gravity units to SI without mutating input data."""

    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("acceleration values must be finite")
    return array * STANDARD_GRAVITY_METRES_PER_SECOND_SQUARED


def acceleration_metres_per_second_squared_to_g(
    values: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Convert SI acceleration to standard gravity units without mutating input data."""

    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("acceleration values must be finite")
    return array / STANDARD_GRAVITY_METRES_PER_SECOND_SQUARED
