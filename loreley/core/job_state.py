"""Shared evolution-job lifecycle query helpers."""

from __future__ import annotations

from typing import Any


def pending_ingestion_job_conditions(
    *,
    EvolutionJob: Any,
    JobStatus: Any,
    func: Any,
) -> tuple[Any, ...]:
    """Return the shared filter for succeeded jobs awaiting result ingestion."""

    status_norm = func.lower(func.trim(func.coalesce(EvolutionJob.ingestion_status, "")))
    commit_norm = func.trim(func.coalesce(EvolutionJob.result_commit_hash, ""))
    return (
        EvolutionJob.status == JobStatus.SUCCEEDED,
        status_norm.not_in(("succeeded", "skipped")),
        commit_norm != "",
    )
