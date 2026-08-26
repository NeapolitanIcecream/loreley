"""Shared job retry and lease-state helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def iso_or_none(value: object) -> str | None:
    """Return an ISO timestamp string for datetime values."""

    if isinstance(value, datetime):
        return value.isoformat()
    return None


def job_status_value(value: object) -> str:
    """Normalize an ORM/job status enum or string for operator payloads."""

    status_value = getattr(value, "value", value)
    return str(status_value or "").strip().lower()


def db_utc_now(session: Any) -> datetime:
    """Return the current database timestamp normalized to UTC."""

    from sqlalchemy import func, select

    value = session.execute(select(func.now())).scalar_one()
    if not isinstance(value, datetime):
        raise RuntimeError(f"Database current timestamp returned unsupported value: {value!r}")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def job_lease_payload(*, job: Any, now: datetime) -> dict[str, object]:
    """Return the lease state payload used by CLI and API retry paths."""

    lease_expires_at = getattr(job, "lease_expires_at", None)
    run_token = getattr(job, "run_token", None)
    worker_id = getattr(job, "worker_id", None)
    status = job_status_value(getattr(job, "status", None))

    if status == "running":
        if run_token is None or worker_id is None or lease_expires_at is None:
            state = "missing"
        elif isinstance(lease_expires_at, datetime) and lease_expires_at < now:
            state = "stale"
        else:
            state = "active"
    else:
        state = "none"

    return {
        "state": state,
        "heartbeat_at": iso_or_none(getattr(job, "heartbeat_at", None)),
        "lease_expires_at": iso_or_none(lease_expires_at),
        "run_token": str(run_token) if run_token is not None else None,
        "worker_id": str(worker_id) if worker_id is not None else None,
    }


def job_retry_state(*, job: Any, now: datetime) -> tuple[bool, str | None]:
    """Return whether a job is manually retryable and its lease state."""

    from loreley.db.models import JobStatus

    status = getattr(job, "status", None)
    if status == JobStatus.FAILED:
        return True, None

    lease_state = str(job_lease_payload(job=job, now=now)["state"])
    if status == JobStatus.RUNNING and lease_state in {"missing", "stale"}:
        return True, lease_state
    return False, lease_state


def failed_stale_job_conditions(
    *,
    EvolutionJob: Any,
    JobStatus: Any,
    func: Any,
    max_recovery_attempts: int,
) -> tuple[Any, ...]:
    """Return the shared FAILED-stale retry filter used by CLI and API."""

    from sqlalchemy import or_

    lease_error_norm = func.lower(func.trim(func.coalesce(EvolutionJob.last_error, "")))
    return (
        EvolutionJob.status == JobStatus.FAILED,
        EvolutionJob.recovery_count > int(max_recovery_attempts),
        or_(
            lease_error_norm.like("lease expired after missing heartbeat;%"),
            lease_error_norm.like("lease metadata missing for running job;%"),
        ),
    )


def retry_job_row(
    *,
    job: Any,
    reason: str,
    now: datetime,
    session: Any | None = None,
) -> dict[str, object]:
    """Reset a retryable job row to PENDING and return the mutation payload."""

    previous_status = job_status_value(getattr(job, "status", None))
    previous_recovery_count = int(getattr(job, "recovery_count", 0) or 0)
    previous_run_token = getattr(job, "run_token", None)
    previous_candidate_commit = getattr(job, "candidate_commit_hash", None)
    from loreley.db.models import JobStatus

    job.status = JobStatus.PENDING
    job.scheduled_at = now
    job.started_at = None
    job.completed_at = None
    job.heartbeat_at = None
    job.lease_expires_at = None
    job.run_token = None
    job.worker_id = None
    job.recovery_count = 0
    job.candidate_commit_hash = None
    job.candidate_branch_name = None
    job.candidate_published_at = None
    job.result_commit_hash = None
    job.last_error = str(reason or "").strip() or "manual retry requested"
    job.failure_stage = None
    job.failure_kind = None
    if session is not None:
        from loreley.core.evolution_events import (
            JOB_REQUEUED,
            next_event_ordinal,
            record_evolution_event,
        )

        ordinal = next_event_ordinal(
            session,
            event_type=JOB_REQUEUED,
            job_id=getattr(job, "id", None),
        )
        record_evolution_event(
            session,
            event_type=JOB_REQUEUED,
            job_id=getattr(job, "id", None),
            run_token=previous_run_token,
            island_id=getattr(job, "island_id", None),
            commit_hash=previous_candidate_commit,
            occurred_at=now,
            ordinal=ordinal,
            payload={
                "reason": "manual_retry",
                "previous_status": previous_status,
                "recovery_count": previous_recovery_count,
                "manual": True,
            },
            key_parts=("manual_retry",),
        )
    return {
        "job_id": str(getattr(job, "id", "")),
        "previous_status": previous_status,
        "new_status": job_status_value(getattr(job, "status", None)),
        "recovery_count_reset_from": previous_recovery_count,
        "reason": job.last_error,
    }


def retry_failed_stale_jobs_payload(
    *,
    session: Any,
    max_recovery_attempts: int,
    retry_all: bool,
    limit: int | None,
    reason: str,
    now: datetime,
) -> dict[str, object]:
    """Retry FAILED jobs that exhausted stale-lease recovery."""

    rows = load_failed_stale_retry_rows(
        session=session,
        max_recovery_attempts=int(max_recovery_attempts),
        retry_all=retry_all,
        limit=limit,
    )
    retried_jobs = [
        retry_job_row(job=row, reason=reason, now=now, session=session)
        for row in rows
    ]
    return {
        "filters": {
            "failed_stale": True,
            "all": bool(retry_all),
            "limit": None if retry_all else int(limit or 0),
        },
        "count": len(retried_jobs),
        "retried_jobs": retried_jobs,
    }


def load_failed_stale_retry_rows(
    *,
    session: Any,
    max_recovery_attempts: int,
    retry_all: bool,
    limit: int | None,
) -> list[Any]:
    """Load rows eligible for the FAILED-stale bulk retry operation."""

    from sqlalchemy import func, select

    from loreley.db.models import EvolutionJob, JobStatus

    stmt = (
        select(EvolutionJob)
        .where(
            *failed_stale_job_conditions(
                EvolutionJob=EvolutionJob,
                JobStatus=JobStatus,
                func=func,
                max_recovery_attempts=max_recovery_attempts,
            )
        )
        .order_by(
            EvolutionJob.completed_at.desc().nullslast(),
            EvolutionJob.created_at.desc(),
        )
    )
    if not retry_all and limit is not None:
        stmt = stmt.limit(int(limit))
    return list(session.execute(stmt).scalars())
