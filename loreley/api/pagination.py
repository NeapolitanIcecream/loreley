"""Pagination helpers for the UI API."""

from __future__ import annotations

import base64
import json
from typing import Final

MIN_PAGE_LIMIT: Final[int] = 1
DEFAULT_PAGE_LIMIT: Final[int] = 200
MAX_PAGE_LIMIT: Final[int] = 2000

DEFAULT_PAGE_OFFSET: Final[int] = 0
MIN_PAGE_OFFSET: Final[int] = 0


class PaginationCursorError(ValueError):
    """Raised when an opaque pagination cursor cannot be decoded."""


def normalize_pagination(limit: int, offset: int, *, max_limit: int = MAX_PAGE_LIMIT) -> tuple[int, int]:
    """Normalize limit/offset values for SQL queries."""

    try:
        limit_i = int(limit)
    except Exception:
        limit_i = DEFAULT_PAGE_LIMIT
    try:
        offset_i = int(offset)
    except Exception:
        offset_i = DEFAULT_PAGE_OFFSET

    try:
        max_limit_i = int(max_limit)
    except Exception:
        max_limit_i = MAX_PAGE_LIMIT
    if max_limit_i <= 0:
        max_limit_i = MAX_PAGE_LIMIT

    limit_i = max(MIN_PAGE_LIMIT, min(limit_i, max_limit_i))
    offset_i = max(MIN_PAGE_OFFSET, offset_i)
    return limit_i, offset_i


def encode_cursor(payload: dict[str, object]) -> str:
    """Encode a small JSON payload as an opaque pagination cursor."""

    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> dict[str, object]:
    """Decode an opaque pagination cursor into its JSON payload."""

    value = str(cursor or "").strip()
    if not value:
        raise PaginationCursorError("Pagination cursor is empty.")

    padding = "=" * (-len(value) % 4)
    try:
        decoded = base64.urlsafe_b64decode((value + padding).encode("ascii"))
        payload = json.loads(decoded.decode("utf-8"))
    except Exception as exc:
        raise PaginationCursorError("Pagination cursor is invalid.") from exc
    if not isinstance(payload, dict):
        raise PaginationCursorError("Pagination cursor payload is invalid.")
    return dict(payload)
