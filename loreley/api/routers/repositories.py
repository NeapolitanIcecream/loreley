"""Repository endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from loreley.api.schemas.repositories import RepositoryOut
from loreley.api.services.repositories import list_repositories

router = APIRouter()


@router.get("/repositories", response_model=list[RepositoryOut])
def get_repositories() -> list[RepositoryOut]:
    return list_repositories()


