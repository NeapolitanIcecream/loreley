"""Repository schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field, field_validator

from loreley.api.schemas import OrmOutModel


class RepositoryOut(OrmOutModel):
    id: UUID
    slug: str
    remote_url: str | None
    root_path: str | None
    extra: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    @field_validator("extra", mode="before")
    @classmethod
    def _extra_default(cls, v: object) -> dict[str, Any]:
        if v is None:
            return {}
        return dict(v)  # type: ignore[arg-type]


