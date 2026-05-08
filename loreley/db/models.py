from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    BigInteger,
    CheckConstraint,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.orm import Mapped, mapped_column, relationship

from loreley.db.base import Base


class TimestampMixin:
    """Shared timestamp columns."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class JobStatus(str, enum.Enum):
    """Possible job lifecycle states."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OperatorTaskStatus(str, enum.Enum):
    """Background operator task lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class OperatorTaskKind(str, enum.Enum):
    """Background operator task kinds."""

    BASELINE_ENSURE = "baseline_ensure"


class InstanceMetadata(TimestampMixin, Base):
    """Single-row instance metadata marker for single-tenant databases."""

    __tablename__ = "instance_metadata"
    __table_args__ = (
        CheckConstraint("id = 1", name="ck_instance_metadata_single_row"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    experiment_id_raw: Mapped[str] = mapped_column(String(128), nullable=False)
    experiment_uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    root_commit_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    repository_slug: Mapped[str | None] = mapped_column(String(255))
    repository_canonical_origin: Mapped[str | None] = mapped_column(String(1024))

    def __repr__(self) -> str:  # pragma: no cover - repr helper
        return (
            "<InstanceMetadata "
            f"experiment_id_raw={self.experiment_id_raw!r} "
            f"root_commit_hash={self.root_commit_hash!r}>"
        )


class OperatorTask(TimestampMixin, Base):
    """UI API background operator task state."""

    __tablename__ = "operator_tasks"
    __table_args__ = (
        Index("ix_operator_tasks_kind_status_created", "kind", "status", "created_at"),
        Index("ix_operator_tasks_status_started", "status", "started_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32),
        default=OperatorTaskStatus.PENDING.value,
        nullable=False,
    )
    request_payload: Mapped[dict[str, Any]] = mapped_column(
        MutableDict.as_mutable(JSONB),
        default=dict,
        nullable=False,
    )
    result_payload: Mapped[dict[str, Any]] = mapped_column(
        MutableDict.as_mutable(JSONB),
        default=dict,
        nullable=False,
    )
    error_summary: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    def __repr__(self) -> str:  # pragma: no cover - repr helper
        return f"<OperatorTask id={self.id!r} kind={self.kind!r} status={self.status!r}>"


class CommitCard(TimestampMixin, Base):
    """Lightweight commit representation used for inspiration and UI."""

    __tablename__ = "commit_cards"
    __table_args__ = (
        UniqueConstraint("commit_hash", name="uq_commit_cards_commit_hash"),
        Index("ix_commit_cards_island_id", "island_id"),
        Index("ix_commit_cards_parent_hash", "parent_commit_hash"),
        Index("ix_commit_cards_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    commit_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    parent_commit_hash: Mapped[str | None] = mapped_column(String(64))
    island_id: Mapped[str | None] = mapped_column(String(64))
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evolution_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    author: Mapped[str | None] = mapped_column(String(128))
    subject: Mapped[str] = mapped_column(String(72), nullable=False)
    change_summary: Mapped[str] = mapped_column(String(512), nullable=False)
    evaluation_summary: Mapped[str | None] = mapped_column(String(512))
    tags: Mapped[list[str]] = mapped_column(
        MutableList.as_mutable(ARRAY(String(64))),
        default=list,
        nullable=False,
    )
    key_files: Mapped[list[str]] = mapped_column(
        MutableList.as_mutable(ARRAY(String(256))),
        default=list,
        nullable=False,
    )
    highlights: Mapped[list[str]] = mapped_column(
        MutableList.as_mutable(ARRAY(String(200))),
        default=list,
        nullable=False,
    )

    metrics: Mapped[list["Metric"]] = relationship(
        back_populates="commit",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    def __repr__(self) -> str:  # pragma: no cover - repr helper
        return (
            f"<CommitCard id={self.id!r} commit_hash={self.commit_hash!r} "
            f"island={self.island_id!r}>"
        )


class CommitChunkSummary(TimestampMixin, Base):
    """Cached LLM summary for a fixed-size block of commit-to-parent steps."""

    __tablename__ = "commit_chunk_summaries"
    __table_args__ = (
        Index("ix_commit_chunk_summaries_end_hash", "end_commit_hash"),
    )

    # Cache key is stable for root-aligned full chunks on the CommitCard parent chain.
    start_commit_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    end_commit_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    block_size: Mapped[int] = mapped_column(Integer, primary_key=True)
    model: Mapped[str] = mapped_column(String(255), default="", nullable=False)

    step_count: Mapped[int] = mapped_column(Integer, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)

    def __repr__(self) -> str:  # pragma: no cover - repr helper
        return (
            "<CommitChunkSummary "
            f"start={self.start_commit_hash[:12]!r} end={self.end_commit_hash[:12]!r} "
            f"block={self.block_size!r} model={self.model!r}>"
        )


class Metric(TimestampMixin, Base):
    """Metric captured from evaluation step."""

    __tablename__ = "metrics"
    __table_args__ = (
        UniqueConstraint("commit_card_id", "name", name="uq_metric_commit_card_name"),
        Index("ix_metrics_commit_card_id", "commit_card_id"),
        Index("ix_metrics_name_value", "name", "value"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    commit_card_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("commit_cards.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str | None] = mapped_column(String(32))
    higher_is_better: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column(
        MutableDict.as_mutable(JSONB),
        default=dict,
        nullable=False,
    )

    commit: Mapped["CommitCard"] = relationship(
        back_populates="metrics",
        primaryjoin="CommitCard.id == Metric.commit_card_id",
        foreign_keys=[commit_card_id],
    )

    def __repr__(self) -> str:  # pragma: no cover - repr helper
        return (
            f"<Metric commit_card_id={self.commit_card_id!r} "
            f"name={self.name!r} value={self.value!r}>"
        )


class CampaignProgram(TimestampMixin, Base):
    """Content-addressed campaign program snapshot."""

    __tablename__ = "campaign_programs"

    hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    source_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    title: Mapped[str | None] = mapped_column(String(256))
    raw_markdown: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_snapshot: Mapped[dict[str, Any]] = mapped_column(
        MutableDict.as_mutable(JSONB),
        default=dict,
        nullable=False,
    )
    recognized_sections: Mapped[list[str]] = mapped_column(
        MutableList.as_mutable(ARRAY(String(64))),
        default=list,
        nullable=False,
    )
    parse_warnings: Mapped[list[dict[str, Any]]] = mapped_column(
        MutableList.as_mutable(JSONB),
        default=list,
        nullable=False,
    )


class CampaignBaseline(TimestampMixin, Base):
    """Source-of-truth root baseline for a comparable campaign contract."""

    __tablename__ = "campaign_baselines"
    __table_args__ = (
        UniqueConstraint("baseline_key_hash", name="uq_campaign_baselines_key_hash"),
        Index("ix_campaign_baselines_root_commit", "root_commit_hash"),
        Index("ix_campaign_baselines_campaign_program_hash", "campaign_program_hash"),
        Index("ix_campaign_baselines_status", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    baseline_key_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    root_commit_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    campaign_program_hash: Mapped[str | None] = mapped_column(String(64))
    evaluator_name: Mapped[str | None] = mapped_column(String(128))
    evaluator_version: Mapped[str | None] = mapped_column(String(128))
    primary_metric_name: Mapped[str] = mapped_column(String(128), nullable=False)
    primary_metric_higher_is_better: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    runtime_profile: Mapped[str | None] = mapped_column(String(128))
    effective_settings_fingerprint: Mapped[str | None] = mapped_column(String(64))

    status: Mapped[str] = mapped_column(String(32), nullable=False)

    metric_value: Mapped[float | None] = mapped_column(Float)
    metric_unit: Mapped[str | None] = mapped_column(String(32))
    evaluation_summary: Mapped[str | None] = mapped_column(Text)
    failure_kind: Mapped[str | None] = mapped_column(String(64))
    failure_summary: Mapped[str | None] = mapped_column(Text)

    commit_card_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("commit_cards.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    metric_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("metrics.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class CandidateCommit(TimestampMixin, Base):
    """Durable ledger row for a worker-produced candidate commit."""

    __tablename__ = "candidate_commits"
    __table_args__ = (
        UniqueConstraint("commit_hash", name="uq_candidate_commits_commit_hash"),
        Index("ix_candidate_commits_produced_by_job_id", "produced_by_job_id"),
        Index(
            "ix_candidate_commits_repair_pool",
            "island_id",
            "repair_state",
            "evaluation_status",
            "updated_at",
        ),
        Index("ix_candidate_commits_repair_source", "repair_source_candidate_id"),
        Index("ix_candidate_commits_git_parent", "git_parent_commit_hash"),
        Index("ix_candidate_commits_nearest_viable_ancestor", "nearest_viable_ancestor_hash"),
        Index("ix_candidate_commits_campaign_program_hash", "campaign_program_hash"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    commit_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    git_parent_commit_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    nearest_viable_ancestor_hash: Mapped[str | None] = mapped_column(String(64))
    island_id: Mapped[str | None] = mapped_column(String(64))

    produced_by_job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evolution_jobs.id", ondelete="SET NULL"),
        nullable=True,
    )
    run_token: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    job_kind: Mapped[str] = mapped_column(String(32), default="evolution", nullable=False)
    repair_source_candidate_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("candidate_commits.id", ondelete="SET NULL"),
        nullable=True,
    )
    repair_mode: Mapped[str | None] = mapped_column(String(32))
    campaign_program_hash: Mapped[str | None] = mapped_column(String(64))

    candidate_branch_name: Mapped[str | None] = mapped_column(String(255))
    candidate_published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    publication_status: Mapped[str] = mapped_column(String(32), default="created", nullable=False)

    evaluation_status: Mapped[str] = mapped_column(
        String(32),
        default="not_evaluated",
        nullable=False,
    )
    latest_evaluation_attempt_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "evaluation_attempts.id",
            name="fk_candidate_commits_latest_evaluation_attempt_id",
            use_alter=True,
        ),
        nullable=True,
    )

    archive_status: Mapped[str] = mapped_column(
        String(32),
        default="not_considered",
        nullable=False,
    )
    lifecycle_status: Mapped[str] = mapped_column(String(32), default="active", nullable=False)

    failure_stage: Mapped[str | None] = mapped_column(String(32))
    failure_kind: Mapped[str | None] = mapped_column(String(64))
    failure_summary: Mapped[str | None] = mapped_column(Text)
    failure_evidence_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("diagnostic_capsules.id", ondelete="SET NULL"),
        nullable=True,
    )

    repair_state: Mapped[str] = mapped_column(String(32), default="audit_only", nullable=False)
    failed_depth: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    repair_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_repair_job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evolution_jobs.id", ondelete="SET NULL"),
        nullable=True,
    )

    repo_state_aggregate_status: Mapped[str] = mapped_column(
        String(32),
        default="not_required",
        nullable=False,
    )
    repo_state_aggregate_error: Mapped[str | None] = mapped_column(Text)

    commit_card_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("commit_cards.id", ondelete="SET NULL"),
        nullable=True,
    )
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    def __repr__(self) -> str:  # pragma: no cover - repr helper
        return (
            f"<CandidateCommit id={self.id!r} commit_hash={self.commit_hash!r} "
            f"evaluation_status={self.evaluation_status!r}>"
        )


class DiagnosticCapsule(TimestampMixin, Base):
    """Sanitized evaluator failure evidence safe for repair prompts."""

    __tablename__ = "diagnostic_capsules"
    __table_args__ = (
        Index("ix_diagnostic_capsules_candidate_commit_id", "candidate_commit_id"),
        Index("ix_diagnostic_capsules_job_id", "job_id"),
        Index("ix_diagnostic_capsules_policy", "policy_version", "policy_passed"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    candidate_commit_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("candidate_commits.id", ondelete="CASCADE"),
        nullable=True,
    )
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evolution_jobs.id", ondelete="CASCADE"),
        nullable=True,
    )
    evaluation_attempt_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evaluation_attempts.id", ondelete="SET NULL"),
        nullable=True,
    )
    schema_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    policy_version: Mapped[str] = mapped_column(String(64), nullable=False)
    policy_passed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(
        MutableDict.as_mutable(JSONB),
        default=dict,
        nullable=False,
    )
    omitted_reasons: Mapped[list[str]] = mapped_column(
        MutableList.as_mutable(ARRAY(String(64))),
        default=list,
        nullable=False,
    )


class EvaluationAttempt(TimestampMixin, Base):
    """One evaluator outcome observed for a candidate commit."""

    __tablename__ = "evaluation_attempts"
    __table_args__ = (
        Index("ix_evaluation_attempts_candidate_started", "candidate_commit_id", "started_at"),
        Index("ix_evaluation_attempts_job_id", "job_id"),
        Index("ix_evaluation_attempts_outcome_kind", "outcome_kind"),
        Index("ix_evaluation_attempts_campaign_program_hash", "campaign_program_hash"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    candidate_commit_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("candidate_commits.id", ondelete="CASCADE"),
        nullable=True,
    )
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evolution_jobs.id", ondelete="CASCADE"),
        nullable=True,
    )
    evaluator_name: Mapped[str | None] = mapped_column(String(128))
    evaluator_version: Mapped[str | None] = mapped_column(String(128))
    campaign_program_hash: Mapped[str | None] = mapped_column(String(64))
    outcome_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    failure_kind: Mapped[str | None] = mapped_column(String(64))
    failure_stage: Mapped[str | None] = mapped_column(String(32))
    repairability: Mapped[str | None] = mapped_column(String(32))
    safe_failure_summary: Mapped[str | None] = mapped_column(Text)
    diagnostic_capsule_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("diagnostic_capsules.id", ondelete="SET NULL"),
        nullable=True,
    )
    artifact_policy_version: Mapped[str | None] = mapped_column(String(64))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EvolutionJob(TimestampMixin, Base):
    """Job produced by the scheduler that drives one evolution iteration."""

    __tablename__ = "evolution_jobs"
    __table_args__ = (
        Index("ix_evolution_jobs_status", "status"),
        Index("ix_evolution_jobs_base_commit", "base_commit_hash"),
        Index("ix_evolution_jobs_running_lease", "status", "lease_expires_at"),
        Index(
            "ix_evolution_jobs_pending_priority",
            "status",
            "priority",
            "scheduled_at",
            "created_at",
        ),
        Index(
            "ix_evolution_jobs_ingestion_scan",
            "status",
            "ingestion_status",
            "completed_at",
            "result_commit_hash",
        ),
        Index("ix_evolution_jobs_kind_status_scheduled", "job_kind", "status", "scheduled_at"),
        Index("ix_evolution_jobs_repair_source_status", "repair_source_candidate_id", "status"),
        Index("ix_evolution_jobs_campaign_program_hash", "campaign_program_hash"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    status: Mapped[JobStatus] = mapped_column(
        SAEnum(JobStatus, name="job_status"),
        default=JobStatus.PENDING,
        nullable=False,
    )
    base_commit_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    island_id: Mapped[str | None] = mapped_column(String(64))
    inspiration_commit_hashes: Mapped[list[str]] = mapped_column(
        MutableList.as_mutable(ARRAY(String(64))),
        default=list,
        nullable=False,
    )
    plan_summary: Mapped[str | None] = mapped_column(Text)
    goal: Mapped[str | None] = mapped_column(String(512))
    constraints: Mapped[list[str]] = mapped_column(
        MutableList.as_mutable(ARRAY(String(200))),
        default=list,
        nullable=False,
    )
    acceptance_criteria: Mapped[list[str]] = mapped_column(
        MutableList.as_mutable(ARRAY(String(200))),
        default=list,
        nullable=False,
    )
    notes: Mapped[list[str]] = mapped_column(
        MutableList.as_mutable(ARRAY(String(200))),
        default=list,
        nullable=False,
    )
    tags: Mapped[list[str]] = mapped_column(
        MutableList.as_mutable(ARRAY(String(64))),
        default=list,
        nullable=False,
    )
    iteration_hint: Mapped[str | None] = mapped_column(String(256))
    sampling_strategy: Mapped[str | None] = mapped_column(String(64))
    sampling_initial_radius: Mapped[int | None] = mapped_column(Integer)
    sampling_radius_used: Mapped[int | None] = mapped_column(Integer)
    sampling_fallback_inspirations: Mapped[int | None] = mapped_column(Integer)
    is_seed_job: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    job_kind: Mapped[str] = mapped_column(String(32), default="evolution", nullable=False)
    repair_source_candidate_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("candidate_commits.id", ondelete="SET NULL"),
        nullable=True,
    )
    repair_mode: Mapped[str | None] = mapped_column(String(32))
    campaign_program_hash: Mapped[str | None] = mapped_column(String(64))
    candidate_commit_hash: Mapped[str | None] = mapped_column(String(64))
    candidate_branch_name: Mapped[str | None] = mapped_column(String(255))
    candidate_published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    result_commit_hash: Mapped[str | None] = mapped_column(String(64))
    ingestion_status: Mapped[str | None] = mapped_column(String(32))
    ingestion_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    ingestion_delta: Mapped[float | None] = mapped_column(Float)
    ingestion_status_code: Mapped[int | None] = mapped_column(Integer)
    ingestion_message: Mapped[str | None] = mapped_column(Text)
    ingestion_cell_index: Mapped[int | None] = mapped_column(Integer)
    ingestion_last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ingestion_reason: Mapped[str | None] = mapped_column(Text)
    priority: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    scheduled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    run_token: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    worker_id: Mapped[str | None] = mapped_column(String(128))
    recovery_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)

    def __repr__(self) -> str:  # pragma: no cover - repr helper
        return f"<EvolutionJob id={self.id} status={self.status}>"


class JobArtifacts(TimestampMixin, Base):
    """Filesystem paths for cold-path artifacts produced by a job."""

    __tablename__ = "job_artifacts"

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evolution_jobs.id", ondelete="CASCADE"),
        primary_key=True,
    )

    planning_prompt_path: Mapped[str | None] = mapped_column(String(1024))
    planning_raw_output_path: Mapped[str | None] = mapped_column(String(1024))
    planning_plan_json_path: Mapped[str | None] = mapped_column(String(1024))

    coding_prompt_path: Mapped[str | None] = mapped_column(String(1024))
    coding_raw_output_path: Mapped[str | None] = mapped_column(String(1024))
    coding_execution_json_path: Mapped[str | None] = mapped_column(String(1024))

    evaluation_json_path: Mapped[str | None] = mapped_column(String(1024))
    evaluation_logs_path: Mapped[str | None] = mapped_column(String(1024))


class EvaluationArtifactRecord(TimestampMixin, Base):
    """Evaluator-declared diagnostic artifact metadata materialized by a job."""

    __tablename__ = "evaluation_artifacts"
    __table_args__ = (
        UniqueConstraint("job_id", "key", name="uq_evaluation_artifacts_job_key"),
        Index("ix_evaluation_artifacts_job_id", "job_id"),
        Index("ix_evaluation_artifacts_commit_hash", "commit_hash"),
        Index("ix_evaluation_artifacts_commit_card_id", "commit_card_id"),
        Index(
            "ix_evaluation_artifacts_visibility_projection",
            "visibility",
            "agent_projection",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evolution_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    commit_card_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("commit_cards.id", ondelete="SET NULL"),
        nullable=True,
    )
    commit_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    key: Mapped[str] = mapped_column(String(128), nullable=False)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(128), nullable=False)
    label: Mapped[str | None] = mapped_column(String(128))
    summary: Mapped[str | None] = mapped_column(String(1024))
    visibility: Mapped[str] = mapped_column(String(32), nullable=False)
    agent_projection: Mapped[str] = mapped_column(String(32), nullable=False)
    storage_path: Mapped[str | None] = mapped_column(String(1024))
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    sha256: Mapped[str | None] = mapped_column(String(64))
    diagnostics: Mapped[list[dict[str, Any]]] = mapped_column(
        MutableList.as_mutable(JSONB),
        default=list,
        nullable=False,
    )
    artifact_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSONB),
        default=dict,
        nullable=False,
    )


class MapElitesState(TimestampMixin, Base):
    """Persisted MAP-Elites archive snapshot per island."""

    __tablename__ = "map_elites_states"

    island_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    snapshot: Mapped[dict[str, Any]] = mapped_column(
        MutableDict.as_mutable(JSONB),
        default=dict,
        nullable=False,
    )

    def __repr__(self) -> str:  # pragma: no cover - repr helper
        return (
            f"<MapElitesState island_id={self.island_id!r}>"
        )


class MapElitesArchiveCell(TimestampMixin, Base):
    """Single occupied MAP-Elites archive cell stored incrementally.

    This table replaces embedding the full archive inside `MapElitesState.snapshot`.
    Each occupied cell is stored as one row so inserts can be persisted via upserts.
    """

    __tablename__ = "map_elites_archive_cells"
    __table_args__ = (
        Index(
            "ix_map_elites_archive_cells_commit_hash",
            "commit_hash",
        ),
    )

    island_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    cell_index: Mapped[int] = mapped_column(Integer, primary_key=True)

    commit_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    objective: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    measures: Mapped[list[float]] = mapped_column(
        MutableList.as_mutable(ARRAY(Float)),
        default=list,
        nullable=False,
    )
    solution: Mapped[list[float]] = mapped_column(
        MutableList.as_mutable(ARRAY(Float)),
        default=list,
        nullable=False,
    )
    # Epoch seconds used by the archive extra field.
    timestamp: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    def __repr__(self) -> str:  # pragma: no cover - repr helper
        return (
            "<MapElitesArchiveCell "
            f"island_id={self.island_id!r} cell_index={self.cell_index!r} "
            f"commit={self.commit_hash!r}>"
        )


class MapElitesPcaHistory(TimestampMixin, Base):
    """Commit embedding history entries persisted for PCA reconstruction.

    Rows are keyed by commit hash so updates are idempotent and `last_seen_at`
    can be used to load the most recent history window after restarts.
    """

    __tablename__ = "map_elites_pca_history"
    __table_args__ = (
        Index(
            "ix_map_elites_pca_history_last_seen",
            "island_id",
            "last_seen_at",
        ),
    )

    island_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    commit_hash: Mapped[str] = mapped_column(String(64), primary_key=True)

    vector: Mapped[list[float]] = mapped_column(
        MutableList.as_mutable(ARRAY(Float)),
        default=list,
        nullable=False,
    )
    embedding_model: Mapped[str] = mapped_column(String(255), default="", nullable=False)

    # Epoch seconds used to restore ordered, bounded history windows.
    last_seen_at: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    def __repr__(self) -> str:  # pragma: no cover - repr helper
        return (
            "<MapElitesPcaHistory "
            f"island_id={self.island_id!r} commit={self.commit_hash!r} "
            f"last_seen_at={self.last_seen_at!r}>"
        )


class MapElitesFileEmbeddingCache(TimestampMixin, Base):
    """Persistent file-level embedding cache for the repo-state pipeline.

    The cache is designed for single-tenant databases:
    - Keyed by blob SHA.
    - Stores the final file-level embedding vector (list of floats).
    - Stores `embedding_model` and `dimensions` for validation and debugging.
    """

    __tablename__ = "map_elites_file_embedding_cache"

    blob_sha: Mapped[str] = mapped_column(String(64), primary_key=True)
    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    vector: Mapped[list[float]] = mapped_column(
        MutableList.as_mutable(ARRAY(Float)),
        default=list,
        nullable=False,
    )

    def __repr__(self) -> str:  # pragma: no cover - repr helper
        return (
            "<MapElitesFileEmbeddingCache "
            f"blob_sha={self.blob_sha!r} model={self.embedding_model!r} "
            f"dims={self.dimensions!r}>"
        )


class MapElitesRepoStateAggregate(TimestampMixin, Base):
    """Persisted repo-state aggregate per commit.

    Stores the sum of file embedding vectors and the number of files contributing
    to the sum so the commit vector can be derived as sum/count.
    """

    __tablename__ = "map_elites_repo_state_aggregates"

    commit_hash: Mapped[str] = mapped_column(String(64), primary_key=True)

    file_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    sum_vector: Mapped[list[float]] = mapped_column(
        MutableList.as_mutable(ARRAY(Float)),
        default=list,
        nullable=False,
    )

    def __repr__(self) -> str:  # pragma: no cover - repr helper
        return (
            "<MapElitesRepoStateAggregate "
            f"commit={self.commit_hash[:12]!r} files={self.file_count!r}>"
        )


Index(
    "ix_evolution_jobs_ingestion_sort_expr",
    EvolutionJob.status,
    EvolutionJob.ingestion_status,
    func.coalesce(EvolutionJob.completed_at, EvolutionJob.created_at),
    EvolutionJob.id,
    postgresql_where=(
        EvolutionJob.result_commit_hash.is_not(None)
        & (EvolutionJob.result_commit_hash != "")
    ),
)
Index(
    "ix_evolution_jobs_ui_sort_expr",
    func.coalesce(EvolutionJob.completed_at, EvolutionJob.created_at).desc(),
    EvolutionJob.id.desc(),
)
Index(
    "ix_map_elites_archive_cells_island_commit",
    MapElitesArchiveCell.island_id,
    MapElitesArchiveCell.commit_hash,
)
