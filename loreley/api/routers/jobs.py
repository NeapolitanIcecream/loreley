"""Evolution job endpoints."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from loreley.api.schemas.jobs import JobArtifactsOut, JobDetailOut, JobOut
from loreley.api.services.jobs import get_job, get_job_artifacts, list_jobs
from loreley.db.models import JobStatus

router = APIRouter()

_ARTIFACT_KEY_MAP: dict[str, tuple[str, str, str]] = {
    "planning_prompt": ("planning_prompt_path", "planning_prompt.txt", "text/plain"),
    "planning_raw_output": ("planning_raw_output_path", "planning_raw_output.txt", "text/plain"),
    "planning_plan_json": ("planning_plan_json_path", "planning_plan.json", "application/json"),
    "coding_prompt": ("coding_prompt_path", "coding_prompt.txt", "text/plain"),
    "coding_raw_output": ("coding_raw_output_path", "coding_raw_output.txt", "text/plain"),
    "coding_execution_json": ("coding_execution_json_path", "coding_execution.json", "application/json"),
    "evaluation_json": ("evaluation_json_path", "evaluation.json", "application/json"),
    "evaluation_logs": ("evaluation_logs_path", "evaluation_logs.txt", "text/plain"),
}


def _artifact_urls(job_id: UUID, row) -> JobArtifactsOut | None:
    if row is None:
        return None
    base = f"/api/v1/jobs/{job_id}/artifacts"
    return JobArtifactsOut(
        planning_prompt_url=f"{base}/planning_prompt" if getattr(row, "planning_prompt_path", None) else None,
        planning_raw_output_url=f"{base}/planning_raw_output" if getattr(row, "planning_raw_output_path", None) else None,
        planning_plan_json_url=f"{base}/planning_plan_json" if getattr(row, "planning_plan_json_path", None) else None,
        coding_prompt_url=f"{base}/coding_prompt" if getattr(row, "coding_prompt_path", None) else None,
        coding_raw_output_url=f"{base}/coding_raw_output" if getattr(row, "coding_raw_output_path", None) else None,
        coding_execution_json_url=f"{base}/coding_execution_json" if getattr(row, "coding_execution_json_path", None) else None,
        evaluation_json_url=f"{base}/evaluation_json" if getattr(row, "evaluation_json_path", None) else None,
        evaluation_logs_url=f"{base}/evaluation_logs" if getattr(row, "evaluation_logs_path", None) else None,
    )


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
                is_seed_job=bool(getattr(job, "is_seed_job", False)),
                result_commit_hash=getattr(job, "result_commit_hash", None),
                ingestion_status=getattr(job, "ingestion_status", None),
            )
        )
    return out


@router.get("/jobs/{job_id}", response_model=JobDetailOut)
def get_job_detail(job_id: UUID) -> JobDetailOut:
    job = get_job(job_id=job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    artifacts = get_job_artifacts(job_id=job_id)
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
        is_seed_job=bool(getattr(job, "is_seed_job", False)),
        result_commit_hash=getattr(job, "result_commit_hash", None),
        ingestion_status=getattr(job, "ingestion_status", None),
        inspiration_commit_hashes=list(getattr(job, "inspiration_commit_hashes", []) or []),
        goal=getattr(job, "goal", None),
        constraints=list(getattr(job, "constraints", []) or []),
        acceptance_criteria=list(getattr(job, "acceptance_criteria", []) or []),
        notes=list(getattr(job, "notes", []) or []),
        tags=list(getattr(job, "tags", []) or []),
        iteration_hint=getattr(job, "iteration_hint", None),
        sampling_strategy=getattr(job, "sampling_strategy", None),
        sampling_initial_radius=getattr(job, "sampling_initial_radius", None),
        sampling_radius_used=getattr(job, "sampling_radius_used", None),
        sampling_fallback_inspirations=getattr(job, "sampling_fallback_inspirations", None),
        ingestion_attempts=int(getattr(job, "ingestion_attempts", 0) or 0),
        ingestion_delta=getattr(job, "ingestion_delta", None),
        ingestion_status_code=getattr(job, "ingestion_status_code", None),
        ingestion_message=getattr(job, "ingestion_message", None),
        ingestion_cell_index=getattr(job, "ingestion_cell_index", None),
        ingestion_last_attempt_at=getattr(job, "ingestion_last_attempt_at", None),
        ingestion_reason=getattr(job, "ingestion_reason", None),
        artifacts=_artifact_urls(job.id, artifacts),
    )


@router.get("/jobs/{job_id}/artifacts", response_model=JobArtifactsOut)
def get_job_artifacts_index(job_id: UUID) -> JobArtifactsOut:
    row = get_job_artifacts(job_id=job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Artifacts not found.")
    out = _artifact_urls(job_id, row)
    return out or JobArtifactsOut()


@router.get("/jobs/{job_id}/artifacts/{artifact_key}")
def download_job_artifact(job_id: UUID, artifact_key: str):
    row = get_job_artifacts(job_id=job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Artifacts not found.")
    spec = _ARTIFACT_KEY_MAP.get(artifact_key)
    if spec is None:
        raise HTTPException(status_code=404, detail="Unknown artifact key.")
    column, filename, media_type = spec
    raw_path = getattr(row, column, None)
    if not raw_path:
        raise HTTPException(status_code=404, detail="Artifact missing.")
    path = Path(str(raw_path))
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found.")
    return FileResponse(path, media_type=media_type, filename=filename)


