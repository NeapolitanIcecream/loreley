"""Evolution job endpoints."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from loreley.api.artifacts import (
    ARTIFACT_KEYS,
    artifact_filename,
    artifact_media_type,
    artifact_path_column,
    build_artifact_urls,
)
from loreley.api.pagination import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT
from loreley.api.schemas.jobs import JobArtifactsOut, JobDetailOut, JobOut
from loreley.api.services.jobs import get_job, get_job_artifacts, list_jobs
from loreley.db.models import JobStatus

router = APIRouter()


@router.get("/jobs", response_model=list[JobOut])
def get_jobs(
    experiment_id: UUID | None = None,
    status: JobStatus | None = None,
    limit: int = Query(default=DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    offset: int = Query(default=0, ge=0),
) -> list[JobOut]:
    return list_jobs(experiment_id=experiment_id, status=status, limit=limit, offset=offset)


@router.get("/jobs/{job_id}", response_model=JobDetailOut)
def get_job_detail(job_id: UUID) -> JobDetailOut:
    job = get_job(job_id=job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    artifacts = get_job_artifacts(job_id=job_id)
    base = JobDetailOut.model_validate(job)
    artifacts_out = None
    if artifacts is not None:
        artifacts_out = JobArtifactsOut(**build_artifact_urls(job_id=job.id, row=artifacts))
    return base.model_copy(update={"artifacts": artifacts_out})


@router.get("/jobs/{job_id}/artifacts", response_model=JobArtifactsOut)
def get_job_artifacts_index(job_id: UUID) -> JobArtifactsOut:
    row = get_job_artifacts(job_id=job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Artifacts not found.")
    return JobArtifactsOut(**build_artifact_urls(job_id=job_id, row=row))


@router.get("/jobs/{job_id}/artifacts/{artifact_key}")
def download_job_artifact(job_id: UUID, artifact_key: str):
    row = get_job_artifacts(job_id=job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Artifacts not found.")
    if artifact_key not in ARTIFACT_KEYS:
        raise HTTPException(status_code=404, detail="Unknown artifact key.")
    column = artifact_path_column(artifact_key)
    raw_path = getattr(row, column, None)
    if not raw_path:
        raise HTTPException(status_code=404, detail="Artifact missing.")
    path = Path(str(raw_path))
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found.")
    return FileResponse(
        path,
        media_type=artifact_media_type(artifact_key),
        filename=artifact_filename(artifact_key),
    )


