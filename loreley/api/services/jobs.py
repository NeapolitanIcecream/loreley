"""Job queries for the UI API."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import select

from loreley.db.base import session_scope
from loreley.db.models import EvolutionJob, JobStatus


def _payload_result_commit_hash(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    result = payload.get("result")
    if not isinstance(result, dict):
        return None
    value = result.get("commit_hash")
    return str(value).strip() if value else None


def _payload_ingestion_status(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    ingestion = payload.get("ingestion")
    if not isinstance(ingestion, dict):
        return None
    map_elites = ingestion.get("map_elites")
    if not isinstance(map_elites, dict):
        return None
    value = map_elites.get("status")
    return str(value).strip() if value else None


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


def job_extracted_fields(job: EvolutionJob) -> tuple[str | None, str | None]:
    """Extract (result_commit_hash, ingestion_status) from the job payload."""

    payload = getattr(job, "payload", None)
    return _payload_result_commit_hash(payload), _payload_ingestion_status(payload)


