from __future__ import annotations

import numpy as np
import pytest

from loreley.config import Settings
from loreley.core.map_elites.archive_ops import add_batch, build_archive, build_feature_bounds
from loreley.core.map_elites.types import IslandState


@pytest.mark.benchmark(group="archive")
def test_archive_add_batch(benchmark, settings: Settings) -> None:
    target_dims = 2
    cells_per_dim = 64
    batch_size = 256

    settings.mapelites_dimensionality_target_dims = target_dims
    settings.mapelites_archive_cells_per_dim = cells_per_dim

    lower, upper = build_feature_bounds(target_dims=target_dims)

    rng = np.random.default_rng(0)
    measures = rng.random((batch_size, target_dims), dtype=np.float64)
    objectives = rng.standard_normal((batch_size, 1)).astype(np.float64)
    timestamps = np.linspace(0.0, 1.0, batch_size, dtype=np.float64).tolist()
    commit_hashes = tuple(f"c{i:04d}" for i in range(batch_size))

    def _setup():  # type: ignore[no-untyped-def]
        archive = build_archive(
            settings=settings,
            target_dims=target_dims,
            cells_per_dim=cells_per_dim,
            lower_template=lower,
            upper_template=upper,
            lower_bounds=None,
            upper_bounds=None,
        )
        state = IslandState(
            archive=archive,
            lower_bounds=lower.copy(),
            upper_bounds=upper.copy(),
        )
        return (), {
            "state": state,
            "island_id": "main",
            "commit_hashes": commit_hashes,
            "objective_values": objectives,
            "objective_scores": objectives,
            "measures": measures,
            "timestamps": timestamps,
        }

    statuses, values = benchmark.pedantic(add_batch, setup=_setup, rounds=5, iterations=1)
    assert statuses.shape == (batch_size,)
    assert values.shape == (batch_size,)
