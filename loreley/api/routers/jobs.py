"""Evolution job endpoints."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from loreley.api.schemas.jobs import JobDetailOut, JobOut
from loreley.api.services.jobs import get_job, job_extracted_fields, list_jobs
from loreley.db.models import JobStatus

router = APIRouter()


@router.get("/jobs", response_model=list[JobOut])
def get_jobs(
    experiment_id: UUID | None = None,
    status: JobStatus | None = None,
    limit: int = Query(default=200, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
) -> list[JobOut]:
    jobs = list_jobs(experiment_id=experiment_id, status=status, limit=limit, offset=offset)
    out: list[JobOut] = []
    for job in jobs:
        result_commit, ingestion_status = job_extracted_fields(job)
        out.append(
            JobOut(
                id=job.id,
                status=str(job.status.value if hasattr(job.status, "value") else job.status),
                priority=int(job.priority),
                island_id=job.island_id,
                experiment_id=job.experiment_id,
                base_commit_hash=job.base_commit_hash,
                scheduled_at=job.scheduled_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                last_error=job.last_error,
                result_commit_hash=result_commit,
                ingestion_status=ingestion_status,
            )
        )
    return out


@router.get("/jobs/{job_id}", response_model=JobDetailOut)
def get_job_detail(job_id: UUID) -> JobDetailOut:
    job = get_job(job_id=job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    result_commit, ingestion_status = job_extracted_fields(job)
    return JobDetailOut(
        id=job.id,
        status=str(job.status.value if hasattr(job.status, "value") else job.status),
        priority=int(job.priority),
        island_id=job.island_id,
        experiment_id=job.experiment_id,
        base_commit_hash=job.base_commit_hash,
        scheduled_at=job.scheduled_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        last_error=job.last_error,
        result_commit_hash=result_commit,
        ingestion_status=ingestion_status,
        payload=dict(job.payload or {}),
    )


