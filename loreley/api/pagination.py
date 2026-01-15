"""Pagination helpers for the UI API."""

from __future__ import annotations

from typing import Final

MIN_PAGE_LIMIT: Final[int] = 1
DEFAULT_PAGE_LIMIT: Final[int] = 200
MAX_PAGE_LIMIT: Final[int] = 2000

DEFAULT_PAGE_OFFSET: Final[int] = 0
MIN_PAGE_OFFSET: Final[int] = 0


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

