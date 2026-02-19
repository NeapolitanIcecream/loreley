"""Shared vector math helpers for MAP-Elites.

The MAP-Elites pipeline frequently performs simple linear algebra operations
over moderately large embedding vectors (e.g. 256-3072 dims). Using NumPy
reduces Python-level loop overhead on hot paths.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

Vector = tuple[float, ...]


def l2_normalize(vector: Sequence[float] | np.ndarray) -> Vector:
    """Return L2-normalized copy of a 1-D vector.

    - Returns the input tuple unchanged for zero vectors when possible.
    - Returns an empty tuple for empty inputs.
    """

    arr = np.asarray(vector, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("L2 normalization expects a 1-D vector.")
    if arr.size <= 0:
        return ()

    magnitude = float(np.linalg.norm(arr))
    if magnitude == 0.0:
        if isinstance(vector, tuple):
            return vector
        return tuple(float(v) for v in arr.tolist())
    return tuple((arr / magnitude).tolist())


def mean_and_maybe_l2_normalize_from_sum(
    sum_vector: Sequence[float] | np.ndarray,
    count: int,
    *,
    normalize: bool,
) -> Vector:
    """Return mean(sum_vector/count) and optionally L2-normalize it.

    Returns an empty tuple when count <= 0 or when the input vector is empty.
    """

    effective_count = int(count)
    if effective_count <= 0:
        return ()

    arr = np.asarray(sum_vector, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Vector mean expects a 1-D vector.")
    if arr.size <= 0:
        return ()

    mean = arr / float(effective_count)
    if not normalize:
        return tuple(mean.tolist())

    magnitude = float(np.linalg.norm(mean))
    if magnitude > 0.0:
        mean = mean / magnitude
    return tuple(mean.tolist())


def divide_vector(vector: Sequence[float] | np.ndarray, divisor: float) -> Vector:
    """Return vector/divisor for a 1-D vector.

    Returns an empty tuple when divisor is not positive or when the input is empty.
    """

    denom = float(divisor)
    if denom <= 0.0:
        return ()

    arr = np.asarray(vector, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Vector division expects a 1-D vector.")
    if arr.size <= 0:
        return ()

    return tuple((arr / denom).tolist())


def accumulate_in_place(
    target: np.ndarray,
    delta: Sequence[float] | np.ndarray,
    *,
    subtract: bool = False,
) -> None:
    """Accumulate delta into target (target += delta) in place."""

    if not isinstance(target, np.ndarray):
        raise TypeError("Target must be a NumPy array.")
    if target.ndim != 1:
        raise ValueError("Target must be a 1-D vector.")

    delta_array = np.asarray(delta, dtype=np.float64).reshape(-1)
    if delta_array.shape != target.shape:
        raise ValueError("Embedding dimension mismatch during incremental aggregation.")
    if subtract:
        target -= delta_array
    else:
        target += delta_array


def apply_replacement_delta_in_place(
    target: np.ndarray,
    new: Sequence[float] | np.ndarray,
    old: Sequence[float] | np.ndarray,
) -> None:
    """Apply replacement delta (target += new - old) in place."""

    if not isinstance(target, np.ndarray):
        raise TypeError("Target must be a NumPy array.")
    if target.ndim != 1:
        raise ValueError("Target must be a 1-D vector.")

    new_array = np.asarray(new, dtype=np.float64).reshape(-1)
    old_array = np.asarray(old, dtype=np.float64).reshape(-1)
    if new_array.shape != old_array.shape or new_array.shape != target.shape:
        raise ValueError("Embedding dimension mismatch during incremental aggregation.")

    target += new_array - old_array

