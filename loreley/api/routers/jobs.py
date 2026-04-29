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
from loreley.api.pagination import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT, PaginationCursorError
from loreley.api.schemas.evidence import EvaluationArtifactOut
from loreley.api.schemas.jobs import JobArtifactsOut, JobDetailOut, JobOut, JobPageOut
from loreley.api.services.evidence import (
    artifact_file_path,
    build_agent_feedback_payload,
    build_evaluation_artifact_payload,
    evaluation_artifact_filename,
    get_downloadable_evaluation_artifact,
    list_evaluation_artifacts_for_job,
    load_evidence_indicators_by_commit_hash,
)
from loreley.api.services.jobs import get_job, get_job_artifacts, list_jobs, list_jobs_page
from loreley.db.models import JobStatus

router = APIRouter()


@router.get("/jobs", response_model=list[JobOut])
def get_jobs(
    status: JobStatus | None = None,
    limit: int = Query(default=DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    offset: int = Query(default=0, ge=0),
) -> list[JobOut]:
    rows = list_jobs(status=status, limit=limit, offset=offset)
    indicators = _job_evidence_indicators(rows)
    return [_job_out(row, indicators) for row in rows]


@router.get("/jobs/page", response_model=JobPageOut)
def get_jobs_page(
    status: JobStatus | None = None,
    limit: int = Query(default=DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    cursor: str | None = Query(default=None, description="Opaque pagination cursor."),
) -> JobPageOut:
    try:
        page = list_jobs_page(status=status, limit=limit, cursor=cursor)
    except PaginationCursorError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    indicators = _job_evidence_indicators(page.items)
    return JobPageOut(
        items=[_job_out(row, indicators) for row in page.items],
        next_cursor=page.next_cursor,
    )


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
    evidence_rows = list_evaluation_artifacts_for_job(job_id=job_id)
    commit_hash = _job_commit_hash(job)
    indicators = load_evidence_indicators_by_commit_hash([commit_hash]) if commit_hash else {}
    indicator = indicators.get(commit_hash)
    update = {
        "artifacts": artifacts_out,
        "evaluation_artifacts": [
            EvaluationArtifactOut.model_validate(build_evaluation_artifact_payload(row))
            for row in evidence_rows
        ],
        "evaluation_agent_feedback": build_agent_feedback_payload(evidence_rows),
    }
    if indicator is not None:
        update.update(indicator.as_dict())
    return base.model_copy(update=update)


@router.get("/jobs/{job_id}/artifacts", response_model=JobArtifactsOut)
def get_job_artifacts_index(job_id: UUID) -> JobArtifactsOut:
    row = get_job_artifacts(job_id=job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Artifacts not found.")
    return JobArtifactsOut(**build_artifact_urls(job_id=job_id, row=row))


@router.get("/jobs/{job_id}/evaluation-artifacts", response_model=list[EvaluationArtifactOut])
def get_job_evaluation_artifacts(job_id: UUID) -> list[EvaluationArtifactOut]:
    rows = list_evaluation_artifacts_for_job(job_id=job_id)
    return [
        EvaluationArtifactOut.model_validate(build_evaluation_artifact_payload(row))
        for row in rows
    ]


@router.get("/jobs/{job_id}/evaluation-artifacts/{artifact_key}")
def download_job_evaluation_artifact(job_id: UUID, artifact_key: str):
    row = get_downloadable_evaluation_artifact(job_id=job_id, artifact_key=artifact_key)
    if row is None:
        raise HTTPException(status_code=404, detail="Evaluation artifact not found.")
    path = artifact_file_path(row)
    if path is None:
        raise HTTPException(status_code=404, detail="Evaluation artifact file not found.")
    return FileResponse(
        path,
        media_type=row.mime_type,
        filename=evaluation_artifact_filename(row),
    )


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


def _job_evidence_indicators(rows: list[object]) -> dict[str, object]:
    commit_hashes = [_job_commit_hash(row) for row in rows]
    return load_evidence_indicators_by_commit_hash(commit_hashes)


def _job_commit_hash(row: object) -> str:
    return str(
        getattr(row, "result_commit_hash", None)
        or getattr(row, "candidate_commit_hash", None)
        or ""
    ).strip()


def _job_out(row: object, indicators: dict[str, object]) -> JobOut:
    commit_hash = _job_commit_hash(row)
    out = JobOut.model_validate(row)
    indicator = indicators.get(commit_hash)
    if indicator is None:
        return out
    return out.model_copy(update=indicator.as_dict())
