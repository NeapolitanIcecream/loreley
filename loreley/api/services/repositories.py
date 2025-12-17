"""Repository queries for the UI API."""

from __future__ import annotations

from sqlalchemy import select

from loreley.db.base import session_scope
from loreley.db.models import Repository


def list_repositories() -> list[Repository]:
    """Return repositories ordered by most recently created first."""

    with session_scope() as session:
        stmt = select(Repository).order_by(Repository.created_at.desc())
        return list(session.execute(stmt).scalars())


