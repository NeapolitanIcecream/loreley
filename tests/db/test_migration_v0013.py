from __future__ import annotations

from typing import Any

from loreley.db.migrations.versions import v0013_llm_usage_events as migration


class _FakeConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, statement, params: dict[str, Any] | None = None):  # noqa: ANN001, ARG002
        self.statements.append(str(statement))


def test_v0013_creates_llm_usage_table_and_external_usage_dedupe_index() -> None:
    conn = _FakeConnection()

    migration.upgrade(conn, _settings=None)  # type: ignore[arg-type]

    ddl = "\n".join(conn.statements)
    assert "CREATE TABLE IF NOT EXISTS llm_usage_events" in ddl
    assert "job_id UUID NULL REFERENCES evolution_jobs(id) ON DELETE SET NULL" in ddl
    assert "cached_input_tokens BIGINT NOT NULL DEFAULT 0" in ddl
    assert "cost_usd NUMERIC(18, 8)" in ddl
    assert "raw_usage JSONB NOT NULL DEFAULT '{}'::jsonb" in ddl
    assert "CREATE UNIQUE INDEX IF NOT EXISTS uq_llm_usage_events_external_usage_id" in ddl
    assert "WHERE external_usage_id <> ''" in ddl
