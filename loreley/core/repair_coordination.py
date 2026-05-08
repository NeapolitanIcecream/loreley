"""Shared coordination helpers for failed-candidate repair mutations."""

from __future__ import annotations

from typing import Callable, TypeVar

from sqlalchemy import select

from loreley.db.base import session_scope
from loreley.db.models import InstanceMetadata

_T = TypeVar("_T")


def with_repair_scheduling_lock(*, callback: Callable[[], _T]) -> _T:
    """Run a repair scheduling mutation while holding the instance row lock."""

    with session_scope() as session:
        session.execute(
            select(InstanceMetadata)
            .where(InstanceMetadata.id == 1)
            .with_for_update()
        ).scalar_one()
        return callback()
