"""Commit/metric queries for the UI API."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select

from loreley.api.pagination import normalize_pagination
from loreley.db.base import session_scope
from loreley.db.models import CommitCard, Metric


def list_commits(
    *,
    island_id: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> list[CommitCard]:
    """Return commits ordered by creation time descending."""

    limit, offset = normalize_pagination(limit, offset)

    with session_scope() as session:
        stmt = select(CommitCard)
        if island_id:
            stmt = stmt.where(CommitCard.island_id == island_id)
        stmt = stmt.order_by(CommitCard.created_at.desc())
        stmt = stmt.limit(limit).offset(offset)
        return list(session.execute(stmt).scalars())


def get_commit(*, commit_hash: str) -> CommitCard | None:
    """Return a commit metadata row by commit_hash."""

    with session_scope() as session:
        stmt = select(CommitCard).where(CommitCard.commit_hash == commit_hash)
        return session.execute(stmt).scalar_one_or_none()


def list_metrics(*, commit_card_id: UUID) -> list[Metric]:
    """Return metrics for a commit card ordered by name."""

    with session_scope() as session:
        stmt = (
            select(Metric)
            .where(Metric.commit_card_id == commit_card_id)
            .order_by(Metric.name.asc())
        )
        return list(session.execute(stmt).scalars())


