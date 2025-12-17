"""Graph schemas for visualizing commit lineage."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class CommitGraphNodeOut(BaseModel):
    commit_hash: str
    parent_commit_hash: str | None = None
    island_id: str | None = None
    experiment_id: UUID | None = None
    created_at: datetime | None = None
    author: str | None = None
    message: str | None = None
    metric_name: str | None = None
    metric_value: float | None = None
    fitness: float | None = None
    is_elite: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)


class CommitGraphEdgeOut(BaseModel):
    source: str
    target: str
    kind: str = "parent"


class CommitGraphOut(BaseModel):
    experiment_id: UUID
    metric_name: str | None
    mode: str
    max_nodes: int
    truncated: bool
    nodes: list[CommitGraphNodeOut]
    edges: list[CommitGraphEdgeOut]


