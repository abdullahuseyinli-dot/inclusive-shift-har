from __future__ import annotations

import numpy as np

from inclusive_shift_har.experiments.hera_compact_evidence_integration import (
    ALPHA,
    _validate_probability,
)


def test_fixed_alpha_is_predeclared() -> None:
    assert ALPHA == 0.25


def test_probability_contract() -> None:
    values = _validate_probability(np.full((2, 3), 1.0 / 3.0), "fixture")
    assert values.shape == (2, 3)
