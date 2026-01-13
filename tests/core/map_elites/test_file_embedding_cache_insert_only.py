from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from sqlalchemy.dialects import postgresql

import loreley.core.map_elites.file_embedding_cache as fec


class _FakeSession:
    def __init__(self, calls: list[object]) -> None:
        self._calls = calls

    def execute(self, stmt):  # type: ignore[no-untyped-def]
        self._calls.append(stmt)
        return SimpleNamespace()


def test_db_file_cache_put_many_is_insert_only(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = fec.DatabaseFileEmbeddingCache(
        experiment_id="00000000-0000-0000-0000-000000000000",
        embedding_model="stub",
        requested_dimensions=2,
    )

    calls: list[object] = []

    @contextmanager
    def _fake_session_scope():  # type: ignore[no-untyped-def]
        yield _FakeSession(calls)

    monkeypatch.setattr(fec, "session_scope", _fake_session_scope)

    cache.put_many({"sha": (1.0, 2.0)})

    assert len(calls) == 1
    stmt = calls[0]
    compiled = str(stmt.compile(dialect=postgresql.dialect())).upper()
    assert "ON CONFLICT" in compiled
    assert "DO NOTHING" in compiled
    assert "DO UPDATE" not in compiled


