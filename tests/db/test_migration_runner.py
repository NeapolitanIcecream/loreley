from __future__ import annotations

import pytest

import loreley.db.base as db_base
import loreley.db.migrations.runner as migration_runner
from loreley.db.migrations.runner import (
    MigrationRequiredError,
    SchemaStatus,
    UnsupportedMigrationError,
    ensure_schema_current,
    validate_database_schema,
)
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


class _FakeConnect(_FakeBegin):
    pass


class _FakeEngine:
    def __init__(self, conn: _FakeConnection) -> None:
        self._conn = conn

    def begin(self) -> _FakeBegin:
        return _FakeBegin(self._conn)

    def connect(self) -> _FakeConnect:
        return _FakeConnect(self._conn)


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


def test_current_marker_validates_current_schema_before_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: marker-current databases must still fail fast on schema drift."""

    conn = _FakeConnection()
    engine = _FakeEngine(conn)
    settings = TestSettings(
        EXPERIMENT_ID="current-schema",
        MAPELITES_EXPERIMENT_ROOT_COMMIT="deadbeef",
    )
    target_version = db_base.INSTANCE_SCHEMA_VERSION
    audit_calls: list[object] = []
    validation_calls: list[tuple[object, int]] = []

    monkeypatch.setattr(
        migration_runner,
        "_acquire_migration_lock",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(migration_runner, "_table_exists", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        migration_runner,
        "_require_instance_marker",
        lambda _conn: {"schema_version": target_version},
    )
    monkeypatch.setattr(
        migration_runner,
        "_create_audit_table",
        lambda call_conn: audit_calls.append(call_conn),
    )
    monkeypatch.setattr(
        migration_runner,
        "_validate_current_schema",
        lambda call_conn, *, target_version: validation_calls.append(
            (call_conn, int(target_version))
        ),
    )

    result = ensure_schema_current(
        engine=engine,
        settings=settings,
        target_version=target_version,
        auto_migrate=False,
    )

    assert audit_calls == [conn]
    assert validation_calls == [(conn, target_version)]
    assert result.from_version == target_version
    assert result.to_version == target_version
    assert result.applied_versions == ()


def test_validate_database_schema_preserves_unsupported_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: unsupported old schemas must not suggest a migration that cannot run."""

    conn = _FakeConnection()
    engine = _FakeEngine(conn)
    settings = TestSettings(
        EXPERIMENT_ID="unsupported-schema",
        MAPELITES_EXPERIMENT_ROOT_COMMIT="deadbeef",
    )

    monkeypatch.setattr(
        migration_runner,
        "_describe_schema",
        lambda *_args, **_kwargs: SchemaStatus(
            schema_version=4,
            target_version=db_base.INSTANCE_SCHEMA_VERSION,
            state="unsupported",
            needs_migration=True,
            detail=(
                "No Loreley native migration path from schema_version=4 "
                f"to {db_base.INSTANCE_SCHEMA_VERSION}."
            ),
        ),
    )

    with pytest.raises(UnsupportedMigrationError, match="No Loreley native migration path"):
        validate_database_schema(
            engine=engine,
            settings=settings,
            target_version=db_base.INSTANCE_SCHEMA_VERSION,
        )
