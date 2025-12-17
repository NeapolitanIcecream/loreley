"""Repository endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from loreley.api.schemas.repositories import RepositoryOut
from loreley.api.services.repositories import list_repositories

router = APIRouter()


@router.get("/repositories", response_model=list[RepositoryOut])
def get_repositories() -> list[RepositoryOut]:
    repos = list_repositories()
    return [
        RepositoryOut(
            id=r.id,
            slug=r.slug,
            remote_url=r.remote_url,
            root_path=r.root_path,
            extra=dict(r.extra or {}),
            created_at=r.created_at,
            updated_at=r.updated_at,
        )
        for r in repos
    ]


