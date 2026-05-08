"""Job queries for the UI API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import and_, func, or_, select

from loreley.api.pagination import PaginationCursorError, decode_cursor, encode_cursor, normalize_pagination
from loreley.config import Settings, get_settings
from loreley.core.job_retry import (
    db_utc_now,
    job_retry_state,
    retry_failed_stale_jobs_payload,
    retry_job_row,
)
from loreley.db.base import session_scope
from loreley.db.models import EvolutionJob, JobArtifacts, JobStatus


@dataclass(frozen=True, slots=True)
class JobPage:
    items: list[EvolutionJob]
    next_cursor: str | None


class JobNotFoundError(RuntimeError):
    """Raised when a job cannot be found."""


class JobRetryConflictError(RuntimeError):
    """Raised when a job is not retryable."""


class JobRetryValidationError(RuntimeError):
    """Raised when a bulk retry request is invalid."""


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
    job_kind: str | None = None,
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
        job_kind_filter = str(job_kind or "").strip()
        if job_kind_filter:
            stmt = stmt.where(EvolutionJob.job_kind == job_kind_filter)
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
    job_kind: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> list[EvolutionJob]:
    """Return jobs ordered by completion time (or creation time) descending."""

    limit, offset = normalize_pagination(limit, offset)

    with session_scope() as session:
        stmt = select(EvolutionJob)
        if status is not None:
            stmt = stmt.where(EvolutionJob.status == status)
        job_kind_filter = str(job_kind or "").strip()
        if job_kind_filter:
            stmt = stmt.where(EvolutionJob.job_kind == job_kind_filter)
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


def retry_job_by_id(
    *,
    job_id: UUID,
    reason: str | None = None,
) -> dict[str, object]:
    """Retry one FAILED or stale/missing-lease RUNNING job."""

    with session_scope() as session:
        job = session.get(EvolutionJob, job_id)
        if job is None:
            raise JobNotFoundError("Job not found.")
        now = db_utc_now(session)
        retryable, lease_state = job_retry_state(job=job, now=now)
        if not retryable:
            status = getattr(getattr(job, "status", None), "value", getattr(job, "status", None))
            raise JobRetryConflictError(
                f"Only failed or stuck RUNNING jobs can be retried "
                f"(status={status}, lease_state={lease_state or 'n/a'})."
            )
        return retry_job_row(
            job=job,
            reason=str(reason or "").strip() or "manual retry requested via UI API",
            now=now,
        )


def retry_failed_stale_jobs(
    *,
    retry_all: bool,
    limit: int | None,
    reason: str | None = None,
    settings: Settings | None = None,
) -> dict[str, object]:
    """Retry FAILED jobs that exhausted stale-lease recovery."""

    if bool(retry_all) and limit is not None:
        raise JobRetryValidationError("Use either all=true or limit, not both.")
    if not bool(retry_all) and limit is None:
        raise JobRetryValidationError("Use either all=true or limit.")
    active_settings = settings or get_settings()
    with session_scope() as session:
        now = db_utc_now(session)
        return retry_failed_stale_jobs_payload(
            session=session,
            max_recovery_attempts=int(active_settings.scheduler_stale_running_max_recovery_attempts),
            retry_all=bool(retry_all),
            limit=limit,
            reason=str(reason or "").strip() or "manual retry requested via UI API",
            now=now,
        )
