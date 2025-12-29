from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
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


class Repository(TimestampMixin, Base):
    """Source code repository tracked by Loreley."""

    __tablename__ = "repositories"
    __table_args__ = (
        UniqueConstraint("slug", name="uq_repositories_slug"),
        Index("ix_repositories_slug", "slug"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    slug: Mapped[str] = mapped_column(String(255), nullable=False)
    remote_url: Mapped[str | None] = mapped_column(String(1024))
    root_path: Mapped[str | None] = mapped_column(String(1024))
    extra: Mapped[dict[str, Any]] = mapped_column(
        MutableDict.as_mutable(JSONB),
        default=dict,
        nullable=False,
    )

    experiments: Mapped[list["Experiment"]] = relationship(
        back_populates="repository",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:  # pragma: no cover - repr helper
        return f"<Repository slug={self.slug!r}>"


class Experiment(TimestampMixin, Base):
    """Single experiment run configuration within a repository."""

    __tablename__ = "experiments"
    __table_args__ = (
        UniqueConstraint("repository_id", "config_hash", name="uq_experiments_repo_config"),
        Index("ix_experiments_repository_id", "repository_id"),
        Index("ix_experiments_config_hash", "config_hash"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    repository_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
    )
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str | None] = mapped_column(String(255))
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(
        MutableDict.as_mutable(JSONB),
        default=dict,
        nullable=False,
    )
    status: Mapped[str | None] = mapped_column(String(32))

    repository: Mapped["Repository"] = relationship(
        back_populates="experiments",
    )
    jobs: Mapped[list["EvolutionJob"]] = relationship(
        back_populates="experiment",
        passive_deletes=True,
    )
    commits: Mapped[list["CommitCard"]] = relationship(
        back_populates="experiment",
        passive_deletes=True,
    )
    map_elites_states: Mapped[list["MapElitesState"]] = relationship(
        back_populates="experiment",
        passive_deletes=True,
    )

    def __repr__(self) -> str:  # pragma: no cover - repr helper
        return f"<Experiment id={self.id} repository_id={self.repository_id} hash={self.config_hash[:8]!r}>"


class CommitCard(TimestampMixin, Base):
    """Lightweight commit representation used for inspiration and UI."""

    __tablename__ = "commit_cards"
    __table_args__ = (
        Index("ix_commit_cards_island_id", "island_id"),
        Index("ix_commit_cards_parent_hash", "parent_commit_hash"),
    )

    commit_hash: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        nullable=False,
        index=True,
    )
    parent_commit_hash: Mapped[str | None] = mapped_column(String(64))
    island_id: Mapped[str | None] = mapped_column(String(64))
    experiment_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiments.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
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
    jobs_as_base: Mapped[list["EvolutionJob"]] = relationship(
        back_populates="base_commit",
        foreign_keys="EvolutionJob.base_commit_hash",
        passive_deletes=True,
    )
    experiment: Mapped["Experiment | None"] = relationship(
        back_populates="commits",
        foreign_keys=[experiment_id],
    )

    def __repr__(self) -> str:  # pragma: no cover - repr helper
        return (
            f"<CommitCard commit_hash={self.commit_hash!r} "
            f"island={self.island_id!r} experiment_id={self.experiment_id!r}>"
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
    model: Mapped[str] = mapped_column(String(255), primary_key=True)
    prompt_signature: Mapped[str] = mapped_column(String(64), primary_key=True)

    step_count: Mapped[int] = mapped_column(Integer, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)

    def __repr__(self) -> str:  # pragma: no cover - repr helper
        return (
            "<CommitChunkSummary "
            f"start={self.start_commit_hash[:12]!r} end={self.end_commit_hash[:12]!r} "
            f"block={self.block_size!r} model={self.model!r} sig={self.prompt_signature!r}>"
        )


class Metric(TimestampMixin, Base):
    """Metric captured from evaluation step."""

    __tablename__ = "metrics"
    __table_args__ = (
        UniqueConstraint("commit_hash", "name", name="uq_metric_commit_name"),
        Index("ix_metrics_commit_hash", "commit_hash"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    commit_hash: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("commit_cards.commit_hash", ondelete="CASCADE"),
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
        primaryjoin="CommitCard.commit_hash == Metric.commit_hash",
        foreign_keys=[commit_hash],
    )

    def __repr__(self) -> str:  # pragma: no cover - repr helper
        return f"<Metric commit={self.commit_hash!r} name={self.name!r} value={self.value!r}>"


class EvolutionJob(TimestampMixin, Base):
    """Job produced by the scheduler that drives one evolution iteration."""

    __tablename__ = "evolution_jobs"
    __table_args__ = (
        Index("ix_evolution_jobs_status", "status"),
        Index("ix_evolution_jobs_base_commit", "base_commit_hash"),
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
        ForeignKey("commit_cards.commit_hash", ondelete="SET NULL"),
        nullable=True,
    )
    island_id: Mapped[str | None] = mapped_column(String(64))
    experiment_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiments.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
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
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)

    base_commit: Mapped["CommitCard"] = relationship(
        back_populates="jobs_as_base",
        foreign_keys=[base_commit_hash],
    )
    experiment: Mapped["Experiment | None"] = relationship(
        back_populates="jobs",
        foreign_keys=[experiment_id],
    )

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


class MapElitesState(TimestampMixin, Base):
    """Persisted MAP-Elites archive snapshot per island."""

    __tablename__ = "map_elites_states"

    experiment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        primary_key=True,
    )
    island_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    snapshot: Mapped[dict[str, Any]] = mapped_column(
        MutableDict.as_mutable(JSONB),
        default=dict,
        nullable=False,
    )

    experiment: Mapped["Experiment"] = relationship(
        back_populates="map_elites_states",
    )

    def __repr__(self) -> str:  # pragma: no cover - repr helper
        return (
            f"<MapElitesState experiment_id={self.experiment_id!r} "
            f"island_id={self.island_id!r}>"
        )


class MapElitesArchiveCell(TimestampMixin, Base):
    """Single occupied MAP-Elites archive cell stored incrementally.

    This table replaces embedding the full archive inside `MapElitesState.snapshot`.
    Each occupied cell is stored as one row so inserts can be persisted via upserts.
    """

    __tablename__ = "map_elites_archive_cells"
    __table_args__ = (
        Index(
            "ix_map_elites_archive_cells_island",
            "experiment_id",
            "island_id",
        ),
        Index(
            "ix_map_elites_archive_cells_commit_hash",
            "commit_hash",
        ),
    )

    experiment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        primary_key=True,
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
            f"experiment_id={self.experiment_id!r} island_id={self.island_id!r} "
            f"cell_index={self.cell_index!r} commit={self.commit_hash!r}>"
        )


class MapElitesPcaHistory(TimestampMixin, Base):
    """Penultimate embedding history entries persisted for PCA reconstruction.

    Rows are keyed by commit hash so updates are idempotent and `last_seen_at`
    can be used to load the most recent history window after restarts.
    """

    __tablename__ = "map_elites_pca_history"
    __table_args__ = (
        Index(
            "ix_map_elites_pca_history_last_seen",
            "experiment_id",
            "island_id",
            "last_seen_at",
        ),
    )

    experiment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        primary_key=True,
    )
    island_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    commit_hash: Mapped[str] = mapped_column(String(64), primary_key=True)

    vector: Mapped[list[float]] = mapped_column(
        MutableList.as_mutable(ARRAY(Float)),
        default=list,
        nullable=False,
    )
    code_dimensions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    summary_dimensions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    code_model: Mapped[str | None] = mapped_column(String(255))
    summary_model: Mapped[str | None] = mapped_column(String(255))
    summary_embedding_model: Mapped[str | None] = mapped_column(String(255))

    # Epoch seconds used to restore ordered, bounded history windows.
    last_seen_at: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    def __repr__(self) -> str:  # pragma: no cover - repr helper
        return (
            "<MapElitesPcaHistory "
            f"experiment_id={self.experiment_id!r} island_id={self.island_id!r} "
            f"commit={self.commit_hash!r} last_seen_at={self.last_seen_at!r}>"
        )


class MapElitesFileEmbeddingCache(TimestampMixin, Base):
    """Persistent file-level embedding cache keyed by git blob SHA.

    The cache is designed for the repo-state embedding pipeline:
    - Keyed by (blob_sha, embedding_model, dimensions, pipeline_signature).
    - Stores the final file-level embedding vector (list of floats).

    `pipeline_signature` captures preprocessing/chunking settings so that
    embeddings can be invalidated when the pipeline changes while the blob
    content remains identical.
    """

    __tablename__ = "map_elites_file_embedding_cache"
    __table_args__ = (
        Index("ix_mapelites_file_embedding_cache_blob_sha", "blob_sha"),
        Index("ix_mapelites_file_embedding_cache_signature", "pipeline_signature"),
    )

    blob_sha: Mapped[str] = mapped_column(String(64), primary_key=True)
    embedding_model: Mapped[str] = mapped_column(String(255), primary_key=True)
    dimensions: Mapped[int] = mapped_column(Integer, primary_key=True)
    pipeline_signature: Mapped[str] = mapped_column(String(128), primary_key=True)
    vector: Mapped[list[float]] = mapped_column(
        MutableList.as_mutable(ARRAY(Float)),
        default=list,
        nullable=False,
    )

    def __repr__(self) -> str:  # pragma: no cover - repr helper
        return (
            "<MapElitesFileEmbeddingCache "
            f"blob_sha={self.blob_sha!r} model={self.embedding_model!r} "
            f"dims={self.dimensions!r} sig={self.pipeline_signature[:8]!r}>"
        )
