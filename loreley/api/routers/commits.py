"""Commit endpoints."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from loreley.api.schemas.commits import CommitDetailOut, CommitOut, MetricOut
from loreley.api.services.commits import get_commit, list_commits, list_metrics

router = APIRouter()


@router.get("/commits", response_model=list[CommitOut])
def get_commits(
    experiment_id: UUID | None = None,
    island_id: str | None = None,
    limit: int = Query(default=200, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
) -> list[CommitOut]:
    commits = list_commits(experiment_id=experiment_id, island_id=island_id, limit=limit, offset=offset)
    return [
        CommitOut(
            commit_hash=c.commit_hash,
            parent_commit_hash=c.parent_commit_hash,
            island_id=c.island_id,
            experiment_id=c.experiment_id,
            author=c.author,
            message=c.message,
            evaluation_summary=c.evaluation_summary,
            tags=list(c.tags or []),
            created_at=c.created_at,
            updated_at=c.updated_at,
        )
        for c in commits
    ]


@router.get("/commits/{commit_hash}", response_model=CommitDetailOut)
def get_commit_detail(commit_hash: str) -> CommitDetailOut:
    commit = get_commit(commit_hash=commit_hash)
    if commit is None:
        raise HTTPException(status_code=404, detail="Commit not found.")
    metrics = list_metrics(commit_hash=commit_hash)
    return CommitDetailOut(
        commit_hash=commit.commit_hash,
        parent_commit_hash=commit.parent_commit_hash,
        island_id=commit.island_id,
        experiment_id=commit.experiment_id,
        author=commit.author,
        message=commit.message,
        evaluation_summary=commit.evaluation_summary,
        tags=list(commit.tags or []),
        created_at=commit.created_at,
        updated_at=commit.updated_at,
        extra_context=dict(commit.extra_context or {}),
        metrics=[
            MetricOut(
                id=m.id,
                name=m.name,
                value=float(m.value),
                unit=m.unit,
                higher_is_better=bool(m.higher_is_better),
                details=dict(m.details or {}),
                created_at=m.created_at,
                updated_at=m.updated_at,
            )
            for m in metrics
        ],
    )


