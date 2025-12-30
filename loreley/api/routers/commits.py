"""Commit endpoints."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from loreley.api.schemas.commits import CommitArtifactsOut, CommitDetailOut, CommitOut, MetricOut
from loreley.api.services.commits import get_commit, list_commits, list_metrics
from loreley.api.services.jobs import get_job_artifacts

router = APIRouter()


def _artifact_urls(job_id: UUID, row) -> CommitArtifactsOut | None:
    if row is None:
        return None
    base = f"/api/v1/jobs/{job_id}/artifacts"
    return CommitArtifactsOut(
        planning_prompt_url=f"{base}/planning_prompt" if getattr(row, "planning_prompt_path", None) else None,
        planning_raw_output_url=f"{base}/planning_raw_output" if getattr(row, "planning_raw_output_path", None) else None,
        planning_plan_json_url=f"{base}/planning_plan_json" if getattr(row, "planning_plan_json_path", None) else None,
        coding_prompt_url=f"{base}/coding_prompt" if getattr(row, "coding_prompt_path", None) else None,
        coding_raw_output_url=f"{base}/coding_raw_output" if getattr(row, "coding_raw_output_path", None) else None,
        coding_execution_json_url=f"{base}/coding_execution_json" if getattr(row, "coding_execution_json_path", None) else None,
        evaluation_json_url=f"{base}/evaluation_json" if getattr(row, "evaluation_json_path", None) else None,
        evaluation_logs_url=f"{base}/evaluation_logs" if getattr(row, "evaluation_logs_path", None) else None,
    )


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
            job_id=getattr(c, "job_id", None),
            author=c.author,
            subject=getattr(c, "subject", "") or f"Commit {c.commit_hash}",
            change_summary=getattr(c, "change_summary", "") or "N/A",
            evaluation_summary=c.evaluation_summary,
            tags=list(c.tags or []),
            key_files=list(getattr(c, "key_files", []) or []),
            highlights=list(getattr(c, "highlights", []) or []),
            created_at=c.created_at,
            updated_at=c.updated_at,
        )
        for c in commits
    ]


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
    job_id = getattr(commit, "job_id", None)
    if isinstance(job_id, UUID):
        artifacts_row = get_job_artifacts(job_id=job_id)
        artifacts = _artifact_urls(job_id, artifacts_row)
    return CommitDetailOut(
        commit_hash=commit.commit_hash,
        parent_commit_hash=commit.parent_commit_hash,
        island_id=commit.island_id,
        experiment_id=commit.experiment_id,
        job_id=getattr(commit, "job_id", None),
        author=commit.author,
        subject=getattr(commit, "subject", "") or f"Commit {commit.commit_hash}",
        change_summary=getattr(commit, "change_summary", "") or "N/A",
        evaluation_summary=commit.evaluation_summary,
        tags=list(commit.tags or []),
        created_at=commit.created_at,
        updated_at=commit.updated_at,
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
        key_files=list(getattr(commit, "key_files", []) or []),
        highlights=list(getattr(commit, "highlights", []) or []),
        artifacts=artifacts,
    )


