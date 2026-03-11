from __future__ import annotations

from loreley.ui.paging import (
    CursorPagerState,
    advance_cursor_pager,
    current_cursor,
    normalize_cursor_pager,
    retreat_cursor_pager,
)


def test_normalize_cursor_pager_resets_when_signature_changes() -> None:
    state = normalize_cursor_pager(
        signature="jobs:running",
        stored_signature="jobs:failed",
        cursors=("start", "next"),
        index=1,
    )

    assert state == CursorPagerState(signature="jobs:running", cursors=(None,), index=0)


def test_advance_cursor_pager_appends_next_cursor() -> None:
    state = CursorPagerState(signature="jobs:running", cursors=(None,), index=0)

    advanced = advance_cursor_pager(state, next_cursor="cursor-2")

    assert advanced.cursors == (None, "cursor-2")
    assert advanced.index == 1
    assert current_cursor(advanced) == "cursor-2"


def test_retreat_cursor_pager_moves_back_one_page() -> None:
    state = CursorPagerState(signature="jobs:running", cursors=(None, "cursor-2"), index=1)

    previous = retreat_cursor_pager(state)

    assert previous.cursors == (None, "cursor-2")
    assert previous.index == 0
    assert current_cursor(previous) is None
