"""MAP-Elites archive schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from loreley.api.schemas import OrmOutModel


class IslandStatsOut(OrmOutModel):
    island_id: str
    occupied: int
    cells: int
    qd_score: float
    best_fitness: float


class ArchiveRecordOut(OrmOutModel):
    commit_hash: str
    island_id: str
    cell_index: int
    fitness: float
    measures: list[float] = Field(default_factory=list)
    solution: list[float] = Field(default_factory=list)
    timestamp: float


class ArchiveSnapshotMetaOut(OrmOutModel):
    island_id: str
    entry_count: int
    dims: int
    cells_per_dim: int
    lower_bounds: list[float] = Field(default_factory=list)
    upper_bounds: list[float] = Field(default_factory=list)
    has_projection: bool
    history_length: int
    updated_at: datetime | None = None


