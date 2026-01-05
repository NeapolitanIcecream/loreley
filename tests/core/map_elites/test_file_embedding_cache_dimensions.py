from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import loreley.core.map_elites.file_embedding_cache as fec


class _FakeScalarResult:
    def __init__(self, rows):  # type: ignore[no-untyped-def]
        self._rows = rows

    def scalars(self):  # type: ignore[no-untyped-def]
        return iter(self._rows)


class _FakeSession:
    def __init__(self, calls, rows_by_call):  # type: ignore[no-untyped-def]
        self._calls = calls
        self._rows_by_call = rows_by_call
        self._idx = 0

    def execute(self, stmt):  # type: ignore[no-untyped-def]
        self._calls.append(stmt)
        rows = self._rows_by_call[min(self._idx, len(self._rows_by_call) - 1)]
        self._idx += 1
        return _FakeScalarResult(rows)


def _extract_dimensions_predicates(stmt) -> list[int]:  # type: ignore[no-untyped-def]
    values: list[int] = []
    for crit in list(getattr(stmt, "_where_criteria", ())):
        left = getattr(crit, "left", None)
        right = getattr(crit, "right", None)
        if left is None or right is None:
            continue
        if getattr(left, "key", None) != "dimensions":
            continue
        bind_value = getattr(right, "value", None)
        if bind_value is None:
            continue
        values.append(int(bind_value))
    return values


def test_db_file_cache_always_filters_by_requested_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = fec.DatabaseFileEmbeddingCache(
        embedding_model="stub",
        requested_dimensions=2,
        pipeline_signature="sig",
    )

    calls: list[object] = []
    rows_by_call = [
        [
            SimpleNamespace(blob_sha="sha0", dimensions=2, vector=[1.0, 2.0]),
            # Defensive: even if a wrong-dimension row leaks in, it must not be returned.
            SimpleNamespace(blob_sha="sha1", dimensions=3, vector=[0.0, 0.0, 0.0]),
        ],
        [
            SimpleNamespace(blob_sha="sha500", dimensions=2, vector=[3.0, 4.0]),
        ],
    ]

    @contextmanager
    def _fake_session_scope():  # type: ignore[no-untyped-def]
        yield _FakeSession(calls, rows_by_call)

    monkeypatch.setattr(fec, "session_scope", _fake_session_scope)

    blob_shas = [f"sha{i}" for i in range(501)]
    found = cache.get_many(blob_shas)

    # Only vectors matching the requested dimensions are returned.
    assert set(found.keys()) == {"sha0", "sha500"}
    assert all(len(vec) == 2 for vec in found.values())

    # Ensure every batch query includes an explicit dimensions predicate.
    assert len(calls) == 2
    for stmt in calls:
        dims = _extract_dimensions_predicates(stmt)
        assert dims == [2]


