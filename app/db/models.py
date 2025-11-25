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

from app.db.base import Base


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


class CommitMetadata(TimestampMixin, Base):
    """Git commit metadata captured during evolution."""

    __tablename__ = "commits"
    __table_args__ = (
        Index("ix_commits_island_id", "island_id"),
        Index("ix_commits_parent_hash", "parent_commit_hash"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    commit_hash: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
    )
    parent_commit_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    island_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    author: Mapped[str | None] = mapped_column(String(128))
    message: Mapped[str | None] = mapped_column(Text)
    evaluation_summary: Mapped[str | None] = mapped_column(Text)
    tags: Mapped[list[str]] = mapped_column(
        MutableList.as_mutable(ARRAY(String(64))),
        default=list,
        nullable=False,
    )
    extra_context: Mapped[dict[str, Any]] = mapped_column(
        MutableDict.as_mutable(JSONB),
        default=dict,
        nullable=False,
    )

    metrics: Mapped[list["Metric"]] = relationship(
        back_populates="commit",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    jobs_as_base: Mapped[list["EvolutionJob"]] = relationship(
        back_populates="base_commit",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:  # pragma: no cover - repr helper
        return f"<CommitMetadata commit_hash={self.commit_hash!r} island={self.island_id!r}>"


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
        ForeignKey("commits.commit_hash", ondelete="CASCADE"),
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

    commit: Mapped["CommitMetadata"] = relationship(
        back_populates="metrics",
        primaryjoin="CommitMetadata.commit_hash == Metric.commit_hash",
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
        ForeignKey("commits.commit_hash", ondelete="SET NULL"),
        nullable=True,
    )
    island_id: Mapped[str | None] = mapped_column(String(64))
    inspiration_commit_hashes: Mapped[list[str]] = mapped_column(
        MutableList.as_mutable(ARRAY(String(64))),
        default=list,
        nullable=False,
    )
    payload: Mapped[dict[str, Any]] = mapped_column(
        MutableDict.as_mutable(JSONB),
        default=dict,
        nullable=False,
    )
    plan_summary: Mapped[str | None] = mapped_column(Text)
    priority: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    scheduled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)

    base_commit: Mapped["CommitMetadata"] = relationship(
        back_populates="jobs_as_base",
        foreign_keys=[base_commit_hash],
    )

    def __repr__(self) -> str:  # pragma: no cover - repr helper
        return f"<EvolutionJob id={self.id} status={self.status}>"
