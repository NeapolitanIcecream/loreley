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
    coverage: float
    qd_score: float
    norm_qd_score: float
    best_fitness: float
    best_objective: float
    metric_name: str | None = None
    higher_is_better: bool


class ArchiveRecordOut(OrmOutModel):
    commit_hash: str
    island_id: str
    cell_index: int
    fitness: float
    objective: float | None = None
    metric_value: float | None = None
    metric_name: str | None = None
    higher_is_better: bool | None = None
    measures: list[float] = Field(default_factory=list)
    solution: list[float] = Field(default_factory=list)
    timestamp: float
    has_evaluation_evidence: bool = False
    agent_visible_evidence_count: int = 0
    top_evaluation_diagnosis: str | None = None


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


class ArchiveRecordPageOut(OrmOutModel):
    items: list[ArchiveRecordOut] = Field(default_factory=list)
    next_cursor: str | None = None
