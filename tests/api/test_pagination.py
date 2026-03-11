from __future__ import annotations

import pytest

from loreley.api.pagination import PaginationCursorError, decode_cursor, encode_cursor


def test_cursor_roundtrip_preserves_payload() -> None:
    payload = {"created_at": "2026-03-11T10:00:00+00:00", "commit_id": "abc"}

    encoded = encode_cursor(payload)

    assert decode_cursor(encoded) == payload


def test_decode_cursor_rejects_invalid_input() -> None:
    with pytest.raises(PaginationCursorError):
        decode_cursor("not-a-valid-cursor")
