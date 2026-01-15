"""Evolution job schemas."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field, field_validator

from loreley.api.schemas import OrmOutModel


class JobOut(OrmOutModel):
    id: UUID
    status: str
    priority: int
    island_id: str | None
    experiment_id: UUID | None
    base_commit_hash: str | None
    scheduled_at: datetime | None
    started_at: datetime | None
    completed_at: datetime | None
    last_error: str | None

    is_seed_job: bool = False
    result_commit_hash: str | None = None
    ingestion_status: str | None = None

    @field_validator("status", mode="before")
    @classmethod
    def _status_to_str(cls, v: object) -> str:
        if v is None:
            return ""
        value = getattr(v, "value", None)
        if value is not None:
            return str(value)
        return str(v)


class JobDetailOut(JobOut):
    inspiration_commit_hashes: list[str] = Field(default_factory=list)
    goal: str | None = None
    constraints: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    iteration_hint: str | None = None

    sampling_strategy: str | None = None
    sampling_initial_radius: int | None = None
    sampling_radius_used: int | None = None
    sampling_fallback_inspirations: int | None = None

    ingestion_attempts: int = 0
    ingestion_delta: float | None = None
    ingestion_status_code: int | None = None
    ingestion_message: str | None = None
    ingestion_cell_index: int | None = None
    ingestion_last_attempt_at: datetime | None = None
    ingestion_reason: str | None = None

    artifacts: "JobArtifactsOut | None" = None


class JobArtifactsOut(OrmOutModel):
    planning_prompt_url: str | None = None
    planning_raw_output_url: str | None = None
    planning_plan_json_url: str | None = None

    coding_prompt_url: str | None = None
    coding_raw_output_url: str | None = None
    coding_execution_json_url: str | None = None

    evaluation_json_url: str | None = None
    evaluation_logs_url: str | None = None


