"""Experiment queries for the UI API."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select

from loreley.db.base import session_scope
from loreley.db.models import Experiment


def list_experiments(*, repository_id: UUID) -> list[Experiment]:
    """Return experiments for a repository ordered by newest first."""

    with session_scope() as session:
        stmt = (
            select(Experiment)
            .where(Experiment.repository_id == repository_id)
            .order_by(Experiment.created_at.desc())
        )
        return list(session.execute(stmt).scalars())


def get_experiment(*, experiment_id: UUID) -> Experiment | None:
    """Return a single experiment or None."""

    with session_scope() as session:
        return session.get(Experiment, experiment_id)


