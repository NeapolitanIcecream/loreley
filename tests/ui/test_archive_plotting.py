from __future__ import annotations

from loreley.ui.archive_plotting import build_scatter_points


def test_build_scatter_points_falls_back_to_fitness_when_metric_value_missing() -> None:
    points = build_scatter_points(
        [
            {
                "measures": [0.1, 0.2],
                "metric_value": None,
                "fitness": 1.5,
                "commit_hash": "a",
                "cell_index": 1,
            },
            {
                "measures": [0.3, 0.4],
                "metric_value": 2.5,
                "fitness": 2.5,
                "commit_hash": "b",
                "cell_index": 2,
            },
        ],
        dim_x=0,
        dim_y=1,
        value_key="metric_value",
    )

    assert points == [
        {"x": 0.1, "y": 0.2, "value": 1.5, "commit_hash": "a", "cell_index": 1},
        {"x": 0.3, "y": 0.4, "value": 2.5, "commit_hash": "b", "cell_index": 2},
    ]
