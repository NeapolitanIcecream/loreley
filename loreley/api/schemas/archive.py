"""MAP-Elites archive schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel


class IslandStatsOut(BaseModel):
    island_id: str
    occupied: int
    cells: int
    qd_score: float
    best_fitness: float


class ArchiveRecordOut(BaseModel):
    commit_hash: str
    island_id: str
    cell_index: int
    fitness: float
    measures: list[float]
    solution: list[float]
    timestamp: float


class ArchiveSnapshotMetaOut(BaseModel):
    experiment_id: UUID
    island_id: str
    entry_count: int
    dims: int
    cells_per_dim: int
    lower_bounds: list[float]
    upper_bounds: list[float]
    has_projection: bool
    history_length: int
    updated_at: datetime | None = None


