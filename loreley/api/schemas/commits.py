"""Commit and metric schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel


class MetricOut(BaseModel):
    id: UUID
    name: str
    value: float
    unit: str | None
    higher_is_better: bool
    details: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class CommitOut(BaseModel):
    commit_hash: str
    parent_commit_hash: str | None
    island_id: str | None
    experiment_id: UUID | None
    author: str | None
    message: str | None
    evaluation_summary: str | None
    tags: list[str]
    created_at: datetime
    updated_at: datetime


class CommitDetailOut(CommitOut):
    extra_context: dict[str, Any]
    metrics: list[MetricOut]


