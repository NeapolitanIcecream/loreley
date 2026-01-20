"""Experiment schemas."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from loreley.api.schemas import OrmOutModel


class ExperimentOut(OrmOutModel):
    id: UUID
    repository_id: UUID
    config_hash: str
    name: str | None
    status: str | None
    created_at: datetime
    updated_at: datetime


class ExperimentDetailOut(ExperimentOut):
    """Detailed experiment view."""


