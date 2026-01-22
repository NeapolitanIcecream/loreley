"""Instance metadata endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from loreley.api.schemas.instance import InstanceOut
from loreley.api.services.instance import get_instance_metadata

router = APIRouter()


@router.get("/instance", response_model=InstanceOut)
def get_instance() -> InstanceOut:
    try:
        meta = get_instance_metadata()
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return InstanceOut.model_validate(meta)

