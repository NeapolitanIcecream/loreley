"""Graph schemas for visualizing commit lineage."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field, field_validator

from loreley.api.schemas import OrmOutModel


class CommitGraphNodeOut(OrmOutModel):
    commit_hash: str
    parent_commit_hash: str | None = None
    island_id: str | None = None
    created_at: datetime | None = None
    author: str | None = None
    message: str | None = None
    primary_metric_name: str | None = None
    primary_metric_value: float | None = None
    is_elite: bool = False
    has_evaluation_evidence: bool = False
    agent_visible_evidence_count: int = 0
    top_evaluation_diagnosis: str | None = None
    candidate_fate_label: str | None = None
    candidate_fate_reason: str | None = None
    seed_portfolio_hash: str | None = None
    seed_direction_id: str | None = None
    seed_admission_lane: str | None = None
    seed_admission_reason: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("extra", mode="before")
    @classmethod
    def _extra_default(cls, v: object) -> dict[str, Any]:
        if v is None:
            return {}
        return dict(v)  # type: ignore[arg-type]


class CommitGraphEdgeOut(OrmOutModel):
    source: str
    target: str
    kind: str = "parent"


class CommitGraphOut(OrmOutModel):
    primary_metric_name: str
    primary_metric_higher_is_better: bool
    mode: str
    max_nodes: int
    truncated: bool
    nodes: list[CommitGraphNodeOut]
    edges: list[CommitGraphEdgeOut]
