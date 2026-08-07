from __future__ import annotations

from typing import Any

from loreley.db.migrations.versions import v0018_full_change_summaries as migration
from tests.support import TestSettings


class _FakeConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, statement: Any, _params: dict[str, Any] | None = None) -> None:
        self.statements.append(str(statement))


def test_v0018_preserves_complete_bounded_change_summaries() -> None:
    conn = _FakeConnection()

    migration.upgrade(conn, TestSettings())

    ddl = "\n".join(conn.statements)
    assert "ALTER TABLE commit_cards" in ddl
    assert "change_summary TYPE VARCHAR(800)" in ddl
