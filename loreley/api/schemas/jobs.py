"""Evolution job schemas."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field, field_validator
from pydantic import BaseModel

from loreley.api.schemas import OrmOutModel
from loreley.api.schemas.evidence import (
    EvaluationAgentFeedbackOut,
    EvaluationArtifactOut,
)


class JobOut(OrmOutModel):
    id: UUID
    status: str
    priority: int
    island_id: str | None
    base_commit_hash: str | None
    scheduled_at: datetime | None
    started_at: datetime | None
    completed_at: datetime | None
    last_error: str | None

    is_seed_job: bool = False
    job_kind: str = "evolution"
    execution_mode: str = "agent"
    input_candidate_commit_hash: str | None = None
    archive_ingestion_enabled: bool = True
    repair_source_candidate_id: UUID | None = None
    repair_mode: str | None = None
    result_commit_hash: str | None = None
    ingestion_status: str | None = None
    has_evaluation_evidence: bool = False
    agent_visible_evidence_count: int = 0
    top_evaluation_diagnosis: str | None = None
    candidate_fate_label: str | None = None
    candidate_fate_reason: str | None = None

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
    input_candidate_summary: str | None = None
    external_submission_key: str = ""
    input_provenance: dict[str, object] = Field(default_factory=dict)
    candidate_commit_hash: str | None = None
    candidate_branch_name: str | None = None
    candidate_published_at: datetime | None = None
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
    evaluation_artifacts: list[EvaluationArtifactOut] = Field(default_factory=list)
    evaluation_agent_feedback: EvaluationAgentFeedbackOut | None = None
    latest_evaluation_attempt: "EvaluationAttemptOut | None" = None


class MeasurementEvidenceOut(OrmOutModel):
    key: str
    sha256: str
    size_bytes: int | None = None


class EvaluationAttemptOut(OrmOutModel):
    id: UUID
    attempt_ordinal: int | None = None
    protocol: str
    outcome_kind: str
    evaluator_name: str | None = None
    evaluator_version: str | None = None
    campaign_program_hash: str | None = None
    candidate_identity: str | None = None
    evaluation_identity_key: str | None = None
    measurement_cache_key: str | None = None
    measurement_contract_fingerprint: str | None = None
    measurement_id: UUID | None = None
    measurement_reused: bool = False
    measurement_executed: bool = False
    reuse_kind: str = "none"
    reused_from_attempt_id: UUID | None = None
    measurement_payload_sha256: str | None = None
    measurement_evidence: list[MeasurementEvidenceOut] = Field(default_factory=list)
    evaluator_slot: int | None = None
    evaluator_slot_scope: str | None = None
    evaluator_slot_wait_seconds: float | None = None
    evaluator_slot_acquired_at: datetime | None = None
    evaluator_slot_released_at: datetime | None = None
    evaluator_slot_release_reason: str | None = None
    failure_stage: str | None = None
    failure_kind: str | None = None
    safe_failure_summary: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class JobArtifactsOut(OrmOutModel):
    planning_prompt_url: str | None = None
    planning_raw_output_url: str | None = None
    planning_plan_json_url: str | None = None

    coding_prompt_url: str | None = None
    coding_raw_output_url: str | None = None
    coding_execution_json_url: str | None = None

    evaluation_json_url: str | None = None
    evaluation_logs_url: str | None = None


class JobPageOut(OrmOutModel):
    items: list[JobOut] = Field(default_factory=list)
    next_cursor: str | None = None


class JobRetryRequest(BaseModel):
    reason: str | None = None


class JobRetryOut(OrmOutModel):
    job_id: str
    previous_status: str
    new_status: str
    recovery_count_reset_from: int
    reason: str


class JobsRetryFailedStaleRequest(BaseModel):
    all: bool = False
    limit: int | None = Field(default=None, ge=1)
    reason: str | None = None


class JobsRetryFailedStaleOut(OrmOutModel):
    filters: dict[str, object] = Field(default_factory=dict)
    count: int
    retried_jobs: list[JobRetryOut] = Field(default_factory=list)
