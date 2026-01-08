from __future__ import annotations

from uuid import UUID

import pytest

from loreley.db.locks import AdvisoryLock, release_pg_advisory_lock, try_acquire_pg_advisory_lock, uuid_to_pg_bigint_lock_key


def test_uuid_to_pg_bigint_lock_key_is_stable_and_in_range() -> None:
    experiment_id = UUID("00000000-0000-0000-0000-000000000123")
    key1 = uuid_to_pg_bigint_lock_key(experiment_id)
    key2 = uuid_to_pg_bigint_lock_key(experiment_id)
    assert key1 == key2

    # Signed int64 range.
    assert -(2**63) <= key1 <= (2**63 - 1)

    other = UUID("00000000-0000-0000-0000-000000000124")
    assert uuid_to_pg_bigint_lock_key(other) != key1


class _FakeResult:
    def __init__(self, value: object) -> None:
        self._value = value

    def scalar_one(self) -> object:
        return self._value


class _FakeConnection:
    def __init__(self, locked: bool) -> None:
        self.locked = locked
        self.closed = False
        self.exec_calls: list[tuple[object, object]] = []

    def execution_options(self, **_kwargs):  # type: ignore[no-untyped-def]
        return self

    def execute(self, stmt, params):  # type: ignore[no-untyped-def]
        self.exec_calls.append((stmt, params))
        return _FakeResult(self.locked)

    def close(self) -> None:
        self.closed = True


class _FakeEngine:
    def __init__(self, conn: _FakeConnection) -> None:
        self._conn = conn

    def connect(self):  # type: ignore[no-untyped-def]
        return self._conn


def test_try_acquire_pg_advisory_lock_returns_none_when_unavailable() -> None:
    conn = _FakeConnection(locked=False)
    engine = _FakeEngine(conn)

    lock = try_acquire_pg_advisory_lock(engine=engine, key=123)
    assert lock is None
    assert conn.closed is True


def test_try_acquire_pg_advisory_lock_holds_connection_until_released() -> None:
    conn = _FakeConnection(locked=True)
    engine = _FakeEngine(conn)

    lock = try_acquire_pg_advisory_lock(engine=engine, key=123)
    assert isinstance(lock, AdvisoryLock)
    assert lock.key == 123
    assert conn.closed is False

    release_pg_advisory_lock(lock)
    assert conn.closed is True


