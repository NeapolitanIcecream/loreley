from __future__ import annotations

import pytest

import loreley.db.base as db_base
import loreley.db.migrations.runner as migration_runner
from loreley.db.migrations.runner import MigrationRequiredError, ensure_schema_current
from tests.support import TestSettings


class _FakeConnection:
    pass


class _FakeBegin:
    def __init__(self, conn: _FakeConnection) -> None:
        self._conn = conn

    def __enter__(self) -> _FakeConnection:
        return self._conn

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeEngine:
    def __init__(self, conn: _FakeConnection) -> None:
        self._conn = conn

    def begin(self) -> _FakeBegin:
        return _FakeBegin(self._conn)


def test_fresh_database_requires_explicit_migrate_when_auto_migrate_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: DB_AUTO_MIGRATE=false must not initialize a fresh database."""

    conn = _FakeConnection()
    engine = _FakeEngine(conn)
    settings = TestSettings(
        DB_AUTO_MIGRATE=False,
        EXPERIMENT_ID="fresh-disabled",
        MAPELITES_EXPERIMENT_ROOT_COMMIT="deadbeef",
    )
    audit_calls: list[object] = []

    monkeypatch.setattr(
        migration_runner,
        "_acquire_migration_lock",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(migration_runner, "_table_exists", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(migration_runner, "_has_loreley_tables_without_marker", lambda *_args: False)
    monkeypatch.setattr(
        migration_runner,
        "_create_audit_table",
        lambda call_conn: audit_calls.append(call_conn),
    )
    monkeypatch.setattr(
        db_base.Base.metadata,
        "create_all",
        lambda *_args, **_kwargs: pytest.fail(
            "fresh startup should not create ORM tables",
        ),
    )
    monkeypatch.setattr(
        migration_runner,
        "_seed_marker",
        lambda **_kwargs: pytest.fail("fresh startup should not seed instance metadata"),
    )
    monkeypatch.setattr(
        migration_runner,
        "_record_audit",
        lambda **_kwargs: pytest.fail("fresh startup should not record migration audit"),
    )

    with pytest.raises(MigrationRequiredError, match="uv run loreley db migrate"):
        ensure_schema_current(
            engine=engine,
            settings=settings,
            target_version=db_base.INSTANCE_SCHEMA_VERSION,
            auto_migrate=False,
        )

    assert audit_calls == []
