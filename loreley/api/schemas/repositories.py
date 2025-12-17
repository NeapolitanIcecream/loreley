"""Repository schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel


class RepositoryOut(BaseModel):
    id: UUID
    slug: str
    remote_url: str | None
    root_path: str | None
    extra: dict[str, Any]
    created_at: datetime
    updated_at: datetime


