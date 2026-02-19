from __future__ import annotations

import numpy as np

from loreley.config import Settings
from loreley.core.map_elites.archive_ops import add_batch, build_archive, build_feature_bounds
from loreley.core.map_elites.types import IslandState


def test_add_batch_accepts_ndarray_measures(settings: Settings) -> None:
    settings.mapelites_dimensionality_target_dims = 2
    settings.mapelites_archive_cells_per_dim = 4

    lower, upper = build_feature_bounds(target_dims=2)
    archive = build_archive(
        settings=settings,
        target_dims=2,
        cells_per_dim=4,
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

    commit_to_island: dict[str, str] = {}
    measures = np.asarray([[0.1, 0.1], [0.9, 0.9]], dtype=np.float64)
    statuses, values = add_batch(
        state=state,
        island_id="main",
        commit_hashes=("c1", "c2"),
        objectives=(1.0, 2.0),
        measures=measures,
        timestamps=(1.0, 2.0),
        commit_to_island=commit_to_island,
    )

    assert statuses.shape == (2,)
    assert values.shape == (2,)
    assert int(np.count_nonzero(statuses > 0)) == 2
    assert "c1" in commit_to_island
    assert "c2" in commit_to_island

