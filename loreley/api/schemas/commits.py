"""Commit and metric schemas."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class MetricOut(BaseModel):
    id: UUID
    name: str
    value: float
    unit: str | None
    higher_is_better: bool
    details: dict[str, object]
    created_at: datetime
    updated_at: datetime


class CommitOut(BaseModel):
    commit_hash: str
    parent_commit_hash: str | None
    island_id: str | None
    experiment_id: UUID | None
    job_id: UUID | None = None
    author: str | None
    subject: str
    change_summary: str
    evaluation_summary: str | None
    tags: list[str]
    key_files: list[str] = []
    highlights: list[str] = []
    created_at: datetime
    updated_at: datetime


class CommitDetailOut(CommitOut):
    metrics: list[MetricOut]
    artifacts: "CommitArtifactsOut | None" = None


class CommitArtifactsOut(BaseModel):
    planning_prompt_url: str | None = None
    planning_raw_output_url: str | None = None
    planning_plan_json_url: str | None = None

    coding_prompt_url: str | None = None
    coding_raw_output_url: str | None = None
    coding_execution_json_url: str | None = None

    evaluation_json_url: str | None = None
    evaluation_logs_url: str | None = None


