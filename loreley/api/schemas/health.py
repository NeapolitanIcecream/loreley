"""Schemas for the health endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DBHealth(BaseModel):
    """Database connectivity status."""

    ok: bool
    error: str | None = None


class LogsHealth(BaseModel):
    """Filesystem status for logs directory."""

    ok: bool
    path: str
    exists: bool = Field(description="True when the directory exists on disk.")


class HealthResponse(BaseModel):
    """Response model for `GET /api/v1/health`."""

    ok: bool
    time_utc: str
    db: DBHealth
    logs: LogsHealth
    settings: dict[str, Any]


