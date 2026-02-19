from __future__ import annotations

import numpy as np
import pytest

from loreley.core.map_elites.vector_math import (
    divide_vector,
    l2_normalize,
    mean_and_maybe_l2_normalize_from_sum,
)


def test_l2_normalize_returns_expected_unit_vector() -> None:
    assert l2_normalize((3.0, 4.0)) == pytest.approx((0.6, 0.8))


def test_l2_normalize_preserves_zero_vector_when_tuple() -> None:
    zero = (0.0, 0.0)
    assert l2_normalize(zero) == zero


def test_mean_and_maybe_l2_normalize_from_sum_matches_reference() -> None:
    sum_vector = (3.0, 4.0)

    assert mean_and_maybe_l2_normalize_from_sum(sum_vector, 2, normalize=False) == pytest.approx(
        (1.5, 2.0)
    )
    assert mean_and_maybe_l2_normalize_from_sum(sum_vector, 2, normalize=True) == pytest.approx(
        (0.6, 0.8)
    )

    mean_only = mean_and_maybe_l2_normalize_from_sum(sum_vector, 2, normalize=False)
    mag = float(np.linalg.norm(np.asarray(mean_only, dtype=np.float64)))
    assert mag == pytest.approx(2.5)


def test_divide_vector_returns_empty_on_non_positive_divisor() -> None:
    assert divide_vector((1.0, 2.0), 0.0) == ()
    assert divide_vector((1.0, 2.0), -1.0) == ()

