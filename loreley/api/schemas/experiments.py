"""Experiment schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field

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
    config_snapshot: dict[str, Any] = Field(default_factory=dict)


