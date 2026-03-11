from __future__ import annotations

import loreley.db.base as db_base


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
    create_all_calls: list[object] = []

    monkeypatch.setattr(db_base, "get_settings", lambda: settings)
    monkeypatch.setattr(db_base, "get_engine", lambda: engine)
    monkeypatch.setattr(
        db_base.Base.metadata,
        "create_all",
        lambda bind: create_all_calls.append(bind),
    )

    db_base.ensure_database_schema(validate_marker=False, settings=settings)

    assert create_all_calls == [engine]
    sql_texts = [getattr(stmt, "text", str(stmt)) for stmt in conn.statements]
    assert any("ix_evolution_jobs_ingestion_sort_expr" in text for text in sql_texts)
    assert any("ix_evolution_jobs_ui_sort_expr" in text for text in sql_texts)
    assert any("COALESCE(completed_at, created_at)" in text for text in sql_texts)
