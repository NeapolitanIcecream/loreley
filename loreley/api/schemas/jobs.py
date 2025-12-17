"""Evolution job schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel


class JobOut(BaseModel):
    id: UUID
    status: str
    priority: int
    island_id: str | None
    experiment_id: UUID | None
    base_commit_hash: str | None
    scheduled_at: datetime | None
    started_at: datetime | None
    completed_at: datetime | None
    last_error: str | None

    # Extracted from payload for convenient UI filtering.
    result_commit_hash: str | None = None
    ingestion_status: str | None = None


class JobDetailOut(JobOut):
    payload: dict[str, Any]


