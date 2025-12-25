"""Job queries for the UI API."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select

from loreley.db.base import session_scope
from loreley.db.models import EvolutionJob, JobArtifacts, JobStatus


def list_jobs(
    *,
    experiment_id: UUID | None = None,
    status: JobStatus | None = None,
    limit: int = 200,
    offset: int = 0,
) -> list[EvolutionJob]:
    """Return jobs ordered by completion time (or creation time) descending."""

    limit = max(1, min(int(limit), 2000))
    offset = max(0, int(offset))

    with session_scope() as session:
        stmt = select(EvolutionJob)
        if experiment_id is not None:
            stmt = stmt.where(EvolutionJob.experiment_id == experiment_id)
        if status is not None:
            stmt = stmt.where(EvolutionJob.status == status)
        stmt = stmt.order_by(EvolutionJob.completed_at.desc().nullslast(), EvolutionJob.created_at.desc())
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


