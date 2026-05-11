"""Evolution job endpoints."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from loreley.api.auth import require_write_auth
from loreley.api.artifacts import (
    ARTIFACT_KEYS,
    artifact_filename,
    artifact_media_type,
    artifact_path_column,
    build_artifact_urls,
)
from loreley.api.pagination import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT, PaginationCursorError
from loreley.api.schemas.evidence import EvaluationArtifactOut
from loreley.api.schemas.jobs import (
    JobArtifactsOut,
    JobDetailOut,
    JobOut,
    JobPageOut,
    JobRetryOut,
    JobRetryRequest,
    JobsRetryFailedStaleOut,
    JobsRetryFailedStaleRequest,
)
from loreley.api.services.candidate_fates import (
    job_candidate_commit_hash,
    load_candidate_fates_for_jobs,
)
from loreley.api.services.evidence import (
    artifact_file_path,
    build_agent_feedback_payload,
    build_evaluation_artifact_payload,
    evaluation_artifact_filename,
    get_downloadable_evaluation_artifact,
    list_evaluation_artifacts_for_job,
    load_evidence_indicators_by_commit_hash,
)
from loreley.api.services.jobs import (
    JOB_EVIDENCE_FILTERS,
    JobNotFoundError,
    JobRetryConflictError,
    JobRetryValidationError,
    get_job,
    get_job_artifacts,
    list_jobs,
    list_jobs_page,
    retry_failed_stale_jobs,
    retry_job_by_id,
)
from loreley.core.candidate_fate import CANDIDATE_FATE_LABELS, CandidateFate
from loreley.db.models import JobStatus

router = APIRouter()


JobCandidateFateFilter = Enum(
    "JobCandidateFateFilter",
    {label.upper(): label for label in sorted(CANDIDATE_FATE_LABELS)},
    type=str,
)
JobEvidenceFilter = Enum(
    "JobEvidenceFilter",
    {label.upper(): label for label in sorted(JOB_EVIDENCE_FILTERS)},
    type=str,
)


@router.get("/jobs", response_model=list[JobOut])
def get_jobs(
    status: JobStatus | None = None,
    job_kind: str | None = Query(default=None, description="Optional job kind filter."),
    candidate_fate: JobCandidateFateFilter | None = Query(
        default=None,
        description="Optional canonical candidate fate label filter.",
    ),
    evidence: JobEvidenceFilter | None = Query(
        default=None,
        description="Optional evidence filter: has_evidence, agent_visible, or none.",
    ),
    limit: int = Query(default=DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    offset: int = Query(default=0, ge=0),
) -> list[JobOut]:
    rows = list_jobs(
        status=status,
        job_kind=job_kind,
        candidate_fate=_enum_value(candidate_fate),
        evidence=_enum_value(evidence),
        limit=limit,
        offset=offset,
    )
    indicators = _job_evidence_indicators(rows)
    fates = load_candidate_fates_for_jobs(rows)
    return [_job_out(row, indicators, fates) for row in rows]


@router.get("/jobs/page", response_model=JobPageOut)
def get_jobs_page(
    status: JobStatus | None = None,
    job_kind: str | None = Query(default=None, description="Optional job kind filter."),
    candidate_fate: JobCandidateFateFilter | None = Query(
        default=None,
        description="Optional canonical candidate fate label filter.",
    ),
    evidence: JobEvidenceFilter | None = Query(
        default=None,
        description="Optional evidence filter: has_evidence, agent_visible, or none.",
    ),
    limit: int = Query(default=DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    cursor: str | None = Query(default=None, description="Opaque pagination cursor."),
) -> JobPageOut:
    try:
        page = list_jobs_page(
            status=status,
            job_kind=job_kind,
            candidate_fate=_enum_value(candidate_fate),
            evidence=_enum_value(evidence),
            limit=limit,
            cursor=cursor,
        )
    except PaginationCursorError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    indicators = _job_evidence_indicators(page.items)
    fates = load_candidate_fates_for_jobs(page.items)
    return JobPageOut(
        items=[_job_out(row, indicators, fates) for row in page.items],
        next_cursor=page.next_cursor,
    )


@router.post("/jobs/retry-failed-stale", response_model=JobsRetryFailedStaleOut)
def post_retry_failed_stale_jobs(
    body: JobsRetryFailedStaleRequest,
    _actor: str = Depends(require_write_auth),
) -> JobsRetryFailedStaleOut:
    try:
        payload = retry_failed_stale_jobs(
            retry_all=bool(body.all),
            limit=body.limit,
            reason=body.reason,
        )
    except JobRetryValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JobsRetryFailedStaleOut.model_validate(payload)


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
    fates = load_candidate_fates_for_jobs([job])
    fate = fates.get(str(job.id))
    update = {
        "artifacts": artifacts_out,
        "evaluation_artifacts": [
            EvaluationArtifactOut.model_validate(build_evaluation_artifact_payload(row))
            for row in evidence_rows
        ],
        "evaluation_agent_feedback": build_agent_feedback_payload(evidence_rows),
    }
    if fate is not None:
        update.update(fate.as_dict())
    if indicator is not None:
        update.update(indicator.as_dict())
    return base.model_copy(update=update)


@router.post("/jobs/{job_id}/retry", response_model=JobRetryOut)
def post_retry_job(
    job_id: UUID,
    body: JobRetryRequest | None = None,
    _actor: str = Depends(require_write_auth),
) -> JobRetryOut:
    try:
        payload = retry_job_by_id(
            job_id=job_id,
            reason=body.reason if body is not None else None,
        )
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except JobRetryConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JobRetryOut.model_validate(payload)


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
    return job_candidate_commit_hash(row)


def _job_out(row: object, indicators: dict[str, object], fates: dict[str, CandidateFate]) -> JobOut:
    commit_hash = _job_commit_hash(row)
    out = JobOut.model_validate(row)
    fate = fates.get(str(getattr(row, "id", "") or ""))
    if fate is not None:
        out = out.model_copy(update=fate.as_dict())
    indicator = indicators.get(commit_hash)
    if indicator is None:
        return out
    return out.model_copy(update=indicator.as_dict())


def _enum_value(value: object) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))
