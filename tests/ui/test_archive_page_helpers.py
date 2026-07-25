from __future__ import annotations

import numpy as np

from loreley.ui.pages.archive import _build_primary_grid


def test_primary_grid_aggregates_each_pareto_cell_by_direction() -> None:
    records = [
        {"cell_index": 1, "primary_metric_value": 3.0},
        {"cell_index": 1, "primary_metric_value": 7.0},
        {"cell_index": 2, "primary_metric_value": 5.0},
        {"cell_index": 2, "primary_metric_value": None},
    ]

    maximums = _build_primary_grid(
        records,
        cells_per_dim=2,
        higher_is_better=True,
    )
    minimums = _build_primary_grid(
        records,
        cells_per_dim=2,
        higher_is_better=False,
    )

    assert maximums[np.unravel_index(1, maximums.shape)] == 7.0
    assert minimums[np.unravel_index(1, minimums.shape)] == 3.0
    assert maximums[np.unravel_index(2, maximums.shape)] == 5.0
