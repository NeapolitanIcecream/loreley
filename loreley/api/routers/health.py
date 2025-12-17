"""Health endpoints for the UI API."""

from __future__ import annotations

from fastapi import APIRouter

from loreley.api.deps import now_utc_iso, resolve_logs_root, safe_settings
from loreley.api.schemas.health import DBHealth, HealthResponse, LogsHealth
from loreley.api.services.health import check_db
from loreley.config import get_settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return basic health information for monitoring and UI bootstrapping."""

    settings = get_settings()

    db_result = check_db()
    logs_root = resolve_logs_root(settings)

    return HealthResponse(
        ok=bool(db_result.ok),
        time_utc=now_utc_iso(),
        db=DBHealth(ok=db_result.ok, error=db_result.error),
        logs=LogsHealth(ok=logs_root.exists(), path=str(logs_root), exists=logs_root.exists()),
        settings=safe_settings(settings),
    )


