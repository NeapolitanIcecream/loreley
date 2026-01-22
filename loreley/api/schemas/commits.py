"""Commit and metric schemas."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from loreley.api.schemas import OrmOutModel


class MetricOut(OrmOutModel):
    id: UUID
    name: str
    value: float
    unit: str | None
    higher_is_better: bool
    details: dict[str, object] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    @field_validator("details", mode="before")
    @classmethod
    def _details_default(cls, v: object) -> dict[str, object]:
        if v is None:
            return {}
        return dict(v)  # type: ignore[arg-type]


class CommitOut(OrmOutModel):
    commit_hash: str
    parent_commit_hash: str | None
    island_id: str | None
    job_id: UUID | None = None
    author: str | None
    subject: str
    change_summary: str
    evaluation_summary: str | None
    tags: list[str] = Field(default_factory=list)
    key_files: list[str] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    @field_validator("tags", "key_files", "highlights", mode="before")
    @classmethod
    def _list_defaults(cls, v: object) -> list[str]:
        if v is None:
            return []
        return list(v)  # type: ignore[arg-type]

    @model_validator(mode="after")
    def _fill_display_defaults(self) -> "CommitOut":
        # Keep router logic centralized: empty subject/change_summary get fallbacks.
        if not (self.subject or "").strip():
            self.subject = f"Commit {self.commit_hash}"
        if not (self.change_summary or "").strip():
            self.change_summary = "N/A"
        return self


class CommitDetailOut(CommitOut):
    metrics: list[MetricOut] = Field(default_factory=list)
    artifacts: "CommitArtifactsOut | None" = None


class CommitArtifactsOut(OrmOutModel):
    planning_prompt_url: str | None = None
    planning_raw_output_url: str | None = None
    planning_plan_json_url: str | None = None

    coding_prompt_url: str | None = None
    coding_raw_output_url: str | None = None
    coding_execution_json_url: str | None = None

    evaluation_json_url: str | None = None
    evaluation_logs_url: str | None = None


