"""Common helpers for the UI API.

The UI API is read-only by design. Keep helpers in this module side-effect free
to avoid impacting the scheduler/worker runtime.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.engine.url import make_url

from loreley.config import Settings, get_settings


def now_utc_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


def resolve_logs_root(settings: Settings) -> Path:
    """Resolve the logs root directory (``<base>/logs``) for the current settings."""

    if settings.logs_base_dir:
        base_dir = Path(settings.logs_base_dir).expanduser()
    else:
        base_dir = Path.cwd()
    return (base_dir / "logs").resolve()


def sanitize_dsn(raw_dsn: str) -> str:
    """Hide sensitive parts of a DSN when returning it to clients."""

    url = make_url(raw_dsn)
    if url.password:
        url = url.set(password="***")
    return str(url)


def safe_settings(settings: Settings | None = None) -> dict[str, Any]:
    """Return non-sensitive Settings for debugging and UI display."""

    effective = settings or get_settings()
    return dict(effective.export_safe())


