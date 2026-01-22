"""Instance metadata schemas."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from loreley.api.schemas import OrmOutModel


class InstanceOut(OrmOutModel):
    schema_version: int
    experiment_id_raw: str
    experiment_uuid: UUID
    root_commit_hash: str
    repository_slug: str | None
    repository_canonical_origin: str | None
    created_at: datetime
    updated_at: datetime

