"""Instance metadata endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from loreley.api.schemas.instance import InstanceOut
from loreley.api.services.instance import get_instance_metadata

router = APIRouter()


@router.get("/instance", response_model=InstanceOut)
def get_instance() -> InstanceOut:
    meta = get_instance_metadata()
    return InstanceOut.model_validate(meta)

