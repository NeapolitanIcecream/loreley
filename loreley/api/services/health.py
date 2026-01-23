"""Health checks for the UI API."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text

from loreley.db.base import get_engine


@dataclass(frozen=True, slots=True)
class DBCheckResult:
    ok: bool
    error: str | None = None


def check_db() -> DBCheckResult:
    """Attempt a trivial DB query to validate connectivity."""

    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return DBCheckResult(ok=True, error=None)
    except Exception as exc:  # pragma: no cover - environment dependent
        return DBCheckResult(ok=False, error=str(exc))


