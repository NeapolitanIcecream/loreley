"""Repair pool API schemas."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from loreley.api.schemas import OrmOutModel
from loreley.api.schemas.operator import RepairPoolSummaryOut


class RepairPoolCandidateOut(OrmOutModel):
    id: UUID
    commit_hash: str
    git_parent_commit_hash: str
    nearest_viable_ancestor_hash: str | None = None
    island_id: str | None = None
    produced_by_job_id: UUID | None = None
    job_kind: str = "evolution"
    repair_source_candidate_id: UUID | None = None
    campaign_program_hash: str | None = None
    publication_status: str
    evaluation_status: str
    archive_status: str
    lifecycle_status: str
    failure_stage: str | None = None
    failure_kind: str | None = None
    failure_summary: str | None = None
    repair_state: str
    failed_depth: int
    repair_attempts: int
    last_repair_job_id: UUID | None = None
    last_repair_job_status: str | None = None
    active_repair_job_id: UUID | None = None
    active_repair_job_status: str | None = None
    diagnostic_policy_passed: bool | None = None
    diagnostic_summary: str | None = None
    diagnostic_omitted_reasons: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class RepairPoolPageOut(OrmOutModel):
    items: list[RepairPoolCandidateOut] = Field(default_factory=list)
    next_cursor: str | None = None
    summary: RepairPoolSummaryOut | None = None


class RepairScheduleOut(OrmOutModel):
    scheduled: bool
    job_id: UUID | None = None
    repair_source_candidate_id: UUID | None = None
    base_commit_hash: str | None = None
    message: str


class RepairCandidateActionOut(OrmOutModel):
    candidate: RepairPoolCandidateOut
    action: str


class RepairActionRequest(BaseModel):
    reason: str | None = None
