"""Commit/metric queries for the UI API."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select

from loreley.db.base import session_scope
from loreley.db.models import CommitCard, Metric


def list_commits(
    *,
    experiment_id: UUID | None = None,
    island_id: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> list[CommitCard]:
    """Return commits ordered by creation time descending."""

    limit = max(1, min(int(limit), 2000))
    offset = max(0, int(offset))

    with session_scope() as session:
        stmt = select(CommitCard)
        if experiment_id is not None:
            stmt = stmt.where(CommitCard.experiment_id == experiment_id)
        if island_id:
            stmt = stmt.where(CommitCard.island_id == island_id)
        stmt = stmt.order_by(CommitCard.created_at.desc())
        stmt = stmt.limit(limit).offset(offset)
        return list(session.execute(stmt).scalars())


def get_commit(*, commit_hash: str) -> CommitCard | None:
    """Return a commit metadata row by hash."""

    with session_scope() as session:
        stmt = select(CommitCard).where(CommitCard.commit_hash == commit_hash)
        return session.execute(stmt).scalar_one_or_none()


def list_metrics(*, commit_hash: str) -> list[Metric]:
    """Return metrics for a commit ordered by name."""

    with session_scope() as session:
        stmt = select(Metric).where(Metric.commit_hash == commit_hash).order_by(Metric.name.asc())
        return list(session.execute(stmt).scalars())


