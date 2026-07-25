"""MAP-Elites archive schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from loreley.api.schemas import OrmOutModel


class IslandStatsOut(OrmOutModel):
    island_id: str
    occupied: int
    elites: int
    cells: int
    coverage: float
    objective_count: int
    front_max_size: int
    best_primary_value: float | None = None
    primary_metric_name: str
    primary_metric_higher_is_better: bool


class ArchiveRecordOut(OrmOutModel):
    commit_hash: str
    island_id: str
    cell_index: int
    objective_values: list[float] = Field(default_factory=list)
    objective_scores: list[float] = Field(default_factory=list)
    primary_metric_value: float | None = None
    primary_metric_name: str | None = None
    primary_metric_higher_is_better: bool | None = None
    campaign_baseline_id: str | None = None
    baseline_key_hash: str | None = None
    baseline_status: str | None = None
    delta_from_root_baseline: float | None = None
    measures: list[float] = Field(default_factory=list)
    timestamp: float
    has_evaluation_evidence: bool = False
    agent_visible_evidence_count: int = 0
    top_evaluation_diagnosis: str | None = None
    candidate_fate_label: str | None = None
    candidate_fate_reason: str | None = None


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
