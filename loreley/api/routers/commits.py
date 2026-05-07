"""Commit endpoints."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from loreley.api.artifacts import build_artifact_urls
from loreley.api.pagination import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT, PaginationCursorError
from loreley.api.schemas.evidence import EvaluationArtifactOut
from loreley.api.schemas.commits import CommitArtifactsOut, CommitDetailOut, CommitOut, CommitPageOut, MetricOut
from loreley.api.services.candidate_fates import load_candidate_fates_for_commits
from loreley.api.services.evidence import (
    build_agent_feedback_payload,
    build_evaluation_artifact_payload,
    list_evaluation_artifacts_for_commit,
    load_evidence_indicators_by_commit_hash,
)
from loreley.api.services.commits import get_commit, list_commits, list_commits_page, list_metrics
from loreley.api.services.jobs import get_job_artifacts
from loreley.core.candidate_fate import CandidateFate

router = APIRouter()


@router.get("/commits", response_model=list[CommitOut])
def get_commits(
    island_id: str | None = None,
    query: str | None = None,
    limit: int = Query(default=DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    offset: int = Query(default=0, ge=0),
) -> list[CommitOut]:
    rows = list_commits(island_id=island_id, query=query, limit=limit, offset=offset)
    indicators = load_evidence_indicators_by_commit_hash([row.commit_hash for row in rows])
    fates = load_candidate_fates_for_commits(rows)
    return [_commit_out(row, indicators, fates) for row in rows]


@router.get("/commits/page", response_model=CommitPageOut)
def get_commits_page(
    island_id: str | None = None,
    query: str | None = Query(default=None, description="Server-side text filter."),
    limit: int = Query(default=DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    cursor: str | None = Query(default=None, description="Opaque pagination cursor."),
) -> CommitPageOut:
    try:
        page = list_commits_page(island_id=island_id, query=query, limit=limit, cursor=cursor)
    except PaginationCursorError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    indicators = load_evidence_indicators_by_commit_hash([row.commit_hash for row in page.items])
    fates = load_candidate_fates_for_commits(page.items)
    return CommitPageOut(
        items=[_commit_out(row, indicators, fates) for row in page.items],
        next_cursor=page.next_cursor,
    )


@router.get("/commits/{commit_hash}", response_model=CommitDetailOut)
def get_commit_detail(
    commit_hash: str,
) -> CommitDetailOut:
    commit = get_commit(commit_hash=commit_hash)
    if commit is None:
        raise HTTPException(status_code=404, detail="Commit not found.")
    metrics = [MetricOut.model_validate(row) for row in list_metrics(commit_card_id=commit.id)]
    artifacts = None
    job_id = commit.job_id
    if isinstance(job_id, UUID):
        artifacts_row = get_job_artifacts(job_id=job_id)
        if artifacts_row is not None:
            artifacts = CommitArtifactsOut(**build_artifact_urls(job_id=job_id, row=artifacts_row))
    base = CommitOut.model_validate(commit)
    evidence_rows = list_evaluation_artifacts_for_commit(commit_hash=commit_hash)
    indicators = load_evidence_indicators_by_commit_hash([commit_hash])
    indicator = indicators.get(commit_hash)
    update = indicator.as_dict() if indicator is not None else {}
    fate = load_candidate_fates_for_commits([commit]).get(commit_hash)
    if fate is not None:
        update.update(fate.as_dict())
    base = base.model_copy(update=update)
    return CommitDetailOut(
        **base.model_dump(),
        metrics=metrics,
        artifacts=artifacts,
        evaluation_artifacts=[
            EvaluationArtifactOut.model_validate(build_evaluation_artifact_payload(row))
            for row in evidence_rows
        ],
        evaluation_agent_feedback=build_agent_feedback_payload(evidence_rows),
    )


@router.get("/commits/{commit_hash}/evaluation-artifacts", response_model=list[EvaluationArtifactOut])
def get_commit_evaluation_artifacts(commit_hash: str) -> list[EvaluationArtifactOut]:
    rows = list_evaluation_artifacts_for_commit(commit_hash=commit_hash)
    return [
        EvaluationArtifactOut.model_validate(build_evaluation_artifact_payload(row))
        for row in rows
    ]


def _commit_out(row: object, indicators: dict[str, object], fates: dict[str, CandidateFate]) -> CommitOut:
    out = CommitOut.model_validate(row)
    fate = fates.get(str(getattr(row, "commit_hash", "") or ""))
    if fate is not None:
        out = out.model_copy(update=fate.as_dict())
    indicator = indicators.get(str(getattr(row, "commit_hash", "") or ""))
    if indicator is None:
        return out
    return out.model_copy(update=indicator.as_dict())
