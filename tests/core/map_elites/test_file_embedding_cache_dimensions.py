from __future__ import annotations

from contextlib import contextmanager
import uuid
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


def _extract_experiment_id_predicates(stmt) -> list[str]:  # type: ignore[no-untyped-def]
    values: list[str] = []
    for crit in list(getattr(stmt, "_where_criteria", ())):
        left = getattr(crit, "left", None)
        right = getattr(crit, "right", None)
        if left is None or right is None:
            continue
        if getattr(left, "key", None) != "experiment_id":
            continue
        bind_value = getattr(right, "value", None)
        if bind_value is None:
            continue
        values.append(str(bind_value))
    return values


def test_db_file_cache_scopes_by_experiment_id(monkeypatch: pytest.MonkeyPatch) -> None:
    exp_id = "00000000-0000-0000-0000-000000000000"
    cache = fec.DatabaseFileEmbeddingCache(
        experiment_id=exp_id,
        embedding_model="stub",
        requested_dimensions=2,
    )

    calls: list[object] = []
    rows_by_call = [
        [
            SimpleNamespace(blob_sha="sha0", embedding_model="stub", dimensions=2, vector=[1.0, 2.0]),
        ],
        [
            SimpleNamespace(blob_sha="sha500", embedding_model="stub", dimensions=2, vector=[3.0, 4.0]),
        ],
    ]

    @contextmanager
    def _fake_session_scope():  # type: ignore[no-untyped-def]
        yield _FakeSession(calls, rows_by_call)

    monkeypatch.setattr(fec, "session_scope", _fake_session_scope)

    blob_shas = [f"sha{i}" for i in range(501)]
    found = cache.get_many(blob_shas)

    assert set(found.keys()) == {"sha0", "sha500"}
    assert all(len(vec) == 2 for vec in found.values())

    # Ensure every batch query scopes by experiment id.
    assert len(calls) == 2
    for stmt in calls:
        exp_ids = _extract_experiment_id_predicates(stmt)
        assert exp_ids == [str(uuid.UUID(exp_id))]


def test_db_file_cache_raises_on_dimension_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = fec.DatabaseFileEmbeddingCache(
        experiment_id="00000000-0000-0000-0000-000000000000",
        embedding_model="stub",
        requested_dimensions=2,
    )

    calls: list[object] = []
    rows_by_call = [
        [
            SimpleNamespace(
                blob_sha="sha0",
                embedding_model="stub",
                dimensions=3,
                vector=[0.0, 0.0, 0.0],
            ),
        ],
    ]

    @contextmanager
    def _fake_session_scope():  # type: ignore[no-untyped-def]
        yield _FakeSession(calls, rows_by_call)

    monkeypatch.setattr(fec, "session_scope", _fake_session_scope)

    with pytest.raises(ValueError, match="unexpected dimensions"):
        _ = cache.get_many(["sha0"])


