from __future__ import annotations

import numpy as np
import pytest

from inclusive_shift_har.preprocessing.units import (
    STANDARD_GRAVITY_METRES_PER_SECOND_SQUARED,
    acceleration_g_to_metres_per_second_squared,
    acceleration_metres_per_second_squared_to_g,
)


def test_standard_gravity_conversion_is_exact_and_round_trips() -> None:
    acceleration_g = np.asarray([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)

    acceleration_si = acceleration_g_to_metres_per_second_squared(acceleration_g)

    assert acceleration_si.dtype == np.float64
    assert acceleration_si[4] == STANDARD_GRAVITY_METRES_PER_SECOND_SQUARED
    assert np.allclose(
        acceleration_metres_per_second_squared_to_g(acceleration_si),
        acceleration_g.astype(np.float64),
        rtol=0.0,
        atol=1e-15,
    )
    assert np.array_equal(
        acceleration_g,
        np.asarray([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float32),
    )


def test_conversion_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        acceleration_g_to_metres_per_second_squared(np.asarray([0.0, np.inf], dtype=np.float64))
    with pytest.raises(ValueError, match="finite"):
        acceleration_metres_per_second_squared_to_g(np.asarray([0.0, np.nan], dtype=np.float64))
