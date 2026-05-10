from __future__ import annotations

import pytest

import loreley.db.base as db_base
import loreley.db.migrations.runner as migration_runner


class _FakeConnection:
    def __init__(self) -> None:
        self.statements: list[object] = []

    def execute(self, stmt) -> None:
        self.statements.append(stmt)


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


def test_ensure_database_schema_creates_job_sort_expression_indexes(
    monkeypatch,
    settings,
) -> None:
    conn = _FakeConnection()
    engine = _FakeEngine(conn)
    migration_calls: list[dict[str, object]] = []

    monkeypatch.setattr(db_base, "get_settings", lambda: settings)
    monkeypatch.setattr(db_base, "get_engine", lambda: engine)
    monkeypatch.setattr(
        migration_runner,
        "ensure_schema_current",
        lambda **kwargs: migration_calls.append(kwargs),
    )

    db_base.ensure_database_schema(validate_marker=False, settings=settings)

    assert migration_calls == [
        {
            "engine": engine,
            "settings": settings,
            "target_version": db_base.INSTANCE_SCHEMA_VERSION,
            "auto_migrate": settings.db_auto_migrate,
        }
    ]
    sql_texts = [getattr(stmt, "text", str(stmt)) for stmt in conn.statements]
    assert any("ix_evolution_jobs_ingestion_sort_expr" in text for text in sql_texts)
    assert any("ix_evolution_jobs_ui_sort_expr" in text for text in sql_texts)
    assert any("COALESCE(completed_at, created_at)" in text for text in sql_texts)


def test_reset_database_schema_recreates_native_current_schema(
    monkeypatch,
    settings,
) -> None:
    """Regression: reset-db must leave a schema that db validate considers current."""

    conn = _FakeConnection()
    engine = _FakeEngine(conn)
    migration_calls: list[dict[str, object]] = []
    managed_ddl_calls: list[object] = []

    monkeypatch.setattr(db_base, "get_settings", lambda: settings)
    monkeypatch.setattr(db_base, "get_engine", lambda: engine)
    monkeypatch.setattr(
        db_base.Base.metadata,
        "create_all",
        lambda *_args, **_kwargs: pytest.fail(
            "reset should use the native migration initializer",
        ),
    )
    monkeypatch.setattr(
        migration_runner,
        "ensure_schema_current",
        lambda **kwargs: migration_calls.append(kwargs),
    )
    monkeypatch.setattr(
        db_base,
        "_run_managed_post_schema_ddl",
        lambda call_engine: managed_ddl_calls.append(call_engine),
    )

    db_base.reset_database_schema(include_console_log=False)

    assert migration_calls == [
        {
            "engine": engine,
            "settings": settings,
            "target_version": db_base.INSTANCE_SCHEMA_VERSION,
            "auto_migrate": True,
        }
    ]
    assert managed_ddl_calls == [engine]
    sql_texts = [getattr(stmt, "text", str(stmt)) for stmt in conn.statements]
    assert any(
        'DROP TABLE IF EXISTS "loreley_schema_migrations" CASCADE' in text
        for text in sql_texts
    )
