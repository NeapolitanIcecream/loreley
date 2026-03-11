"""Job queries for the UI API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import and_, func, or_, select

from loreley.api.pagination import PaginationCursorError, decode_cursor, encode_cursor, normalize_pagination
from loreley.db.base import session_scope
from loreley.db.models import EvolutionJob, JobArtifacts, JobStatus


@dataclass(frozen=True, slots=True)
class JobPage:
    items: list[EvolutionJob]
    next_cursor: str | None


def _normalize_cursor_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise PaginationCursorError("Jobs cursor is missing sort_ts.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise PaginationCursorError("Jobs cursor has an invalid timestamp.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _encode_job_cursor(job: EvolutionJob) -> str:
    sort_ts = getattr(job, "completed_at", None) or getattr(job, "created_at", None)
    if not isinstance(sort_ts, datetime):
        raise ValueError("jobs cursor requires a timestamp")
    return encode_cursor(
        {
            "sort_ts": sort_ts.isoformat(),
            "job_id": str(job.id),
        }
    )


def list_jobs_page(
    *,
    status: JobStatus | None = None,
    limit: int = 200,
    cursor: str | None = None,
) -> JobPage:
    """Return a cursor-paginated page of jobs ordered newest-first."""

    limit, _ = normalize_pagination(limit, 0)
    sort_ts = func.coalesce(EvolutionJob.completed_at, EvolutionJob.created_at)

    with session_scope() as session:
        stmt = select(EvolutionJob)
        if status is not None:
            stmt = stmt.where(EvolutionJob.status == status)
        if cursor:
            try:
                payload = decode_cursor(cursor)
                cursor_ts = _normalize_cursor_datetime(payload.get("sort_ts"))
                cursor_job_id = UUID(str(payload.get("job_id")))
            except (PaginationCursorError, ValueError) as exc:
                raise PaginationCursorError("Jobs cursor is invalid.") from exc
            stmt = stmt.where(
                or_(
                    sort_ts < cursor_ts,
                    and_(
                        sort_ts == cursor_ts,
                        EvolutionJob.id < cursor_job_id,
                    ),
                )
            )
        stmt = stmt.order_by(sort_ts.desc(), EvolutionJob.id.desc())
        stmt = stmt.limit(limit + 1)
        rows = list(session.execute(stmt).scalars())

    items = rows[:limit]
    next_cursor = _encode_job_cursor(items[-1]) if len(rows) > limit and items else None
    return JobPage(items=items, next_cursor=next_cursor)


def list_jobs(
    *,
    status: JobStatus | None = None,
    limit: int = 200,
    offset: int = 0,
) -> list[EvolutionJob]:
    """Return jobs ordered by completion time (or creation time) descending."""

    limit, offset = normalize_pagination(limit, offset)

    with session_scope() as session:
        stmt = select(EvolutionJob)
        if status is not None:
            stmt = stmt.where(EvolutionJob.status == status)
        stmt = stmt.order_by(
            EvolutionJob.completed_at.desc().nullslast(),
            EvolutionJob.created_at.desc(),
            EvolutionJob.id.desc(),
        )
        stmt = stmt.limit(limit).offset(offset)
        return list(session.execute(stmt).scalars())


def get_job(*, job_id: UUID) -> EvolutionJob | None:
    """Return a single job or None."""

    with session_scope() as session:
        return session.get(EvolutionJob, job_id)


def get_job_artifacts(*, job_id: UUID) -> JobArtifacts | None:
    """Return JobArtifacts row for a job."""

    with session_scope() as session:
        return session.get(JobArtifacts, job_id)
