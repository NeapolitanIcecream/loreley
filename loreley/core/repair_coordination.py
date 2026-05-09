"""Shared coordination helpers for failed-candidate repair mutations."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from sqlalchemy import func, select

from loreley.db.base import session_scope
from loreley.db.models import EvolutionJob, InstanceMetadata, JobStatus

_T = TypeVar("_T")


def with_repair_scheduling_lock(*, callback: Callable[[Any], _T]) -> _T:
    """Run a repair scheduling mutation while holding the instance row lock."""

    with session_scope() as session:
        session.execute(
            select(InstanceMetadata)
            .where(InstanceMetadata.id == 1)
            .with_for_update()
        ).scalar_one()
        return callback(session)


def repair_tokens_available(*, settings: object, session: Any | None = None) -> int:
    """Return repair tokens available using persisted job history."""

    max_tokens = max(0, int(getattr(settings, "failed_candidate_repair_max_tokens", 0)))
    if max_tokens <= 0:
        return 0
    normal_jobs_per_token = max(
        1,
        int(getattr(settings, "failed_candidate_repair_normal_jobs_per_token", 1)),
    )
    if session is not None:
        return _repair_tokens_available_from_session(
            session=session,
            max_tokens=max_tokens,
            normal_jobs_per_token=normal_jobs_per_token,
        )
    with session_scope() as session:
        return _repair_tokens_available_from_session(
            session=session,
            max_tokens=max_tokens,
            normal_jobs_per_token=normal_jobs_per_token,
        )


def _repair_tokens_available_from_session(
    *,
    session: Any,
    max_tokens: int,
    normal_jobs_per_token: int,
) -> int:
    completed_normal_jobs = int(
        session.execute(
            select(func.count(EvolutionJob.id)).where(
                EvolutionJob.status == JobStatus.SUCCEEDED,
                EvolutionJob.job_kind != "repair",
            )
        ).scalar_one()
    )
    scheduled_repair_jobs = int(
        session.execute(
            select(func.count(EvolutionJob.id)).where(
                EvolutionJob.job_kind == "repair",
            )
        ).scalar_one()
    )
    earned = completed_normal_jobs // normal_jobs_per_token
    return min(max_tokens, max(0, earned - scheduled_repair_jobs))
