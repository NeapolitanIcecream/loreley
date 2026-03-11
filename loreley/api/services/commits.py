"""Commit/metric queries for the UI API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import and_, or_, select

from loreley.api.pagination import PaginationCursorError, decode_cursor, encode_cursor, normalize_pagination
from loreley.db.base import session_scope
from loreley.db.models import CommitCard, Metric


@dataclass(frozen=True, slots=True)
class CommitPage:
    items: list[CommitCard]
    next_cursor: str | None


def _normalize_cursor_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise PaginationCursorError("Commits cursor is missing created_at.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise PaginationCursorError("Commits cursor has an invalid timestamp.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _encode_commit_cursor(commit: CommitCard) -> str:
    if not isinstance(commit.created_at, datetime):
        raise ValueError("commits cursor requires created_at")
    return encode_cursor(
        {
            "created_at": commit.created_at.isoformat(),
            "commit_id": str(commit.id),
        }
    )


def list_commits_page(
    *,
    island_id: str | None = None,
    query: str | None = None,
    limit: int = 200,
    cursor: str | None = None,
) -> CommitPage:
    """Return a cursor-paginated page of commits ordered newest-first."""

    limit, _ = normalize_pagination(limit, 0)

    with session_scope() as session:
        stmt = _apply_commit_filters(
            select(CommitCard),
            island_id=island_id,
            query=query,
        )
        if cursor:
            try:
                payload = decode_cursor(cursor)
                cursor_ts = _normalize_cursor_datetime(payload.get("created_at"))
                cursor_commit_id = UUID(str(payload.get("commit_id")))
            except (PaginationCursorError, ValueError) as exc:
                raise PaginationCursorError("Commits cursor is invalid.") from exc
            stmt = stmt.where(
                or_(
                    CommitCard.created_at < cursor_ts,
                    and_(
                        CommitCard.created_at == cursor_ts,
                        CommitCard.id < cursor_commit_id,
                    ),
                )
            )
        stmt = stmt.order_by(CommitCard.created_at.desc(), CommitCard.id.desc())
        stmt = stmt.limit(limit + 1)
        rows = list(session.execute(stmt).scalars())

    items = rows[:limit]
    next_cursor = _encode_commit_cursor(items[-1]) if len(rows) > limit and items else None
    return CommitPage(items=items, next_cursor=next_cursor)


def list_commits(
    *,
    island_id: str | None = None,
    query: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> list[CommitCard]:
    """Return commits ordered by creation time descending."""

    limit, offset = normalize_pagination(limit, offset)

    with session_scope() as session:
        stmt = _apply_commit_filters(
            select(CommitCard),
            island_id=island_id,
            query=query,
        )
        stmt = stmt.order_by(CommitCard.created_at.desc(), CommitCard.id.desc())
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


def _apply_commit_filters(
    stmt,
    *,
    island_id: str | None,
    query: str | None,
):
    if island_id:
        stmt = stmt.where(CommitCard.island_id == island_id)

    text_query = str(query or "").strip()
    if not text_query:
        return stmt

    pattern = f"%{text_query}%"
    return stmt.where(
        or_(
            CommitCard.commit_hash.ilike(pattern),
            CommitCard.author.ilike(pattern),
            CommitCard.subject.ilike(pattern),
            CommitCard.change_summary.ilike(pattern),
        )
    )
