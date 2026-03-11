"""Pure helpers for Streamlit cursor pagination state."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Sequence


@dataclass(frozen=True, slots=True)
class CursorPagerState:
    signature: str
    cursors: tuple[str | None, ...]
    index: int


def pager_signature(parts: Sequence[object]) -> str:
    """Return a stable signature for cursor state invalidation."""

    return json.dumps([str(part) for part in parts], separators=(",", ":"))


def normalize_cursor_pager(
    *,
    signature: str,
    stored_signature: str | None,
    cursors: Sequence[str | None] | None,
    index: int | None,
) -> CursorPagerState:
    """Normalize persisted cursor state and reset when the inputs changed."""

    if stored_signature != signature or not cursors:
        return CursorPagerState(signature=signature, cursors=(None,), index=0)

    normalized_cursors = tuple(cursors)
    if not normalized_cursors:
        normalized_cursors = (None,)
    normalized_index = max(0, min(int(index or 0), len(normalized_cursors) - 1))
    return CursorPagerState(
        signature=signature,
        cursors=normalized_cursors,
        index=normalized_index,
    )


def current_cursor(state: CursorPagerState) -> str | None:
    """Return the active cursor for the current page."""

    return state.cursors[state.index]


def advance_cursor_pager(
    state: CursorPagerState,
    *,
    next_cursor: str | None,
) -> CursorPagerState:
    """Append the next cursor and move to the next page when available."""

    value = str(next_cursor or "").strip()
    if not value:
        return state

    retained = state.cursors[: state.index + 1]
    return CursorPagerState(
        signature=state.signature,
        cursors=retained + (value,),
        index=state.index + 1,
    )


def retreat_cursor_pager(state: CursorPagerState) -> CursorPagerState:
    """Move back one page without discarding forward cursors."""

    if state.index <= 0:
        return state
    return CursorPagerState(
        signature=state.signature,
        cursors=state.cursors,
        index=state.index - 1,
    )
