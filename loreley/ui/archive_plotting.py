"""Pure helpers for archive visualisation data preparation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def build_scatter_points(
    records: Sequence[Mapping[str, Any]],
    *,
    dim_x: int,
    dim_y: int,
    value_key: str,
) -> list[dict[str, Any]]:
    """Return Plotly-ready scatter points, skipping invalid rows."""

    points: list[dict[str, Any]] = []
    required_dim = max(int(dim_x), int(dim_y))
    for record in records:
        vec = record.get("measures")
        if not isinstance(vec, list) or len(vec) <= required_dim:
            continue

        raw_value = record.get(value_key)
        if raw_value is None:
            continue

        try:
            value = float(raw_value)
            x = float(vec[dim_x])
            y = float(vec[dim_y])
        except (TypeError, ValueError):
            continue

        points.append(
            {
                "x": x,
                "y": y,
                "value": value,
                "commit_hash": record.get("commit_hash"),
                "cell_index": record.get("cell_index"),
                "candidate_fate_label": record.get("candidate_fate_label"),
            }
        )
    return points
