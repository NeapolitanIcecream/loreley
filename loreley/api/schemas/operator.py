"""Operator console API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field, field_validator

from loreley.api.schemas import OrmOutModel


class CampaignProgramFileOut(OrmOutModel):
    found: bool = False
    source_path: str | None = None
    hash: str | None = None
    normalized_hash: str | None = None
    title: str | None = None
    recognized_sections: list[str] = Field(default_factory=list)
    parse_warnings: list[dict[str, Any]] = Field(default_factory=list)
    sections: dict[str, Any] = Field(default_factory=dict)
    error_summary: str | None = None


class CampaignProgramSchedulerOut(OrmOutModel):
    active_hash: str | None = None
    active_source: str | None = None
    persisted_hash: str | None = None
    persisted_source: str | None = None
    current_hash: str | None = None
    current_matches_active: bool | None = None
    change_policy: str | None = None


class BaselineStatusOut(OrmOutModel):
    campaign_baseline_id: str | None = None
    baseline_key_hash: str | None = None
    root_baseline_commit: str | None = None
    root_baseline_metric: str | None = None
    root_baseline_value: float | None = None
    root_baseline_direction: str | None = None
    root_baseline_status: str | None = None
    baseline_campaign_program_hash: str | None = None
    failure_kind: str | None = None
    failure_summary: str | None = None


class RepairPoolSummaryOut(OrmOutModel):
    total_failed_candidates: int = 0
    active_repair_jobs: int = 0
    by_repair_state: dict[str, int] = Field(default_factory=dict)
    by_lifecycle_status: dict[str, int] = Field(default_factory=dict)
    by_failure_kind: dict[str, int] = Field(default_factory=dict)


class JobHealthOut(OrmOutModel):
    jobs: dict[str, int] = Field(default_factory=dict)
    job_leases: dict[str, int] = Field(default_factory=dict)
    by_status: dict[str, int] = Field(default_factory=dict)
    by_job_kind: dict[str, int] = Field(default_factory=dict)
    progress: dict[str, Any] = Field(default_factory=dict)


class OperatorStatusOut(OrmOutModel):
    campaign_program: dict[str, Any] = Field(default_factory=dict)
    baseline: BaselineStatusOut | None = None
    repair_pool: RepairPoolSummaryOut
    job_health: JobHealthOut
    generated_at: datetime | None = None


class OperatorTaskOut(OrmOutModel):
    id: UUID
    kind: str
    status: str
    request_payload: dict[str, Any] = Field(default_factory=dict)
    result_payload: dict[str, Any] = Field(default_factory=dict)
    error_summary: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    @field_validator("request_payload", "result_payload", mode="before")
    @classmethod
    def _payload_default(cls, value: object) -> dict[str, Any]:
        if value is None:
            return {}
        return dict(value)  # type: ignore[arg-type]


class OperatorTaskPageOut(OrmOutModel):
    items: list[OperatorTaskOut] = Field(default_factory=list)
