"""Commit endpoints."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from loreley.api.artifacts import build_artifact_urls
from loreley.api.pagination import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT
from loreley.api.schemas.commits import CommitArtifactsOut, CommitDetailOut, CommitOut
from loreley.api.services.commits import get_commit, list_commits, list_metrics
from loreley.api.services.jobs import get_job_artifacts

router = APIRouter()


@router.get("/commits", response_model=list[CommitOut])
def get_commits(
    experiment_id: UUID | None = None,
    island_id: str | None = None,
    limit: int = Query(default=DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    offset: int = Query(default=0, ge=0),
) -> list[CommitOut]:
    return list_commits(experiment_id=experiment_id, island_id=island_id, limit=limit, offset=offset)


@router.get("/commits/{commit_hash}", response_model=CommitDetailOut)
def get_commit_detail(
    commit_hash: str,
    experiment_id: UUID | None = Query(default=None),
) -> CommitDetailOut:
    if experiment_id is None:
        raise HTTPException(status_code=400, detail="experiment_id is required.")
    commit = get_commit(experiment_id=experiment_id, commit_hash=commit_hash)
    if commit is None:
        raise HTTPException(status_code=404, detail="Commit not found.")
    metrics = list_metrics(commit_card_id=commit.id)
    artifacts = None
    job_id = commit.job_id
    if isinstance(job_id, UUID):
        artifacts_row = get_job_artifacts(job_id=job_id)
        if artifacts_row is not None:
            artifacts = CommitArtifactsOut(**build_artifact_urls(job_id=job_id, row=artifacts_row))
    base = CommitOut.model_validate(commit)
    return CommitDetailOut(**base.model_dump(), metrics=metrics, artifacts=artifacts)


