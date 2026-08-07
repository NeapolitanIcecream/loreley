from __future__ import annotations

from typing import Any

from loreley.db.migrations.versions import v0016_candidate_identities as migration
from tests.support import TestSettings


class _FakeConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, statement: Any, _params: dict[str, Any] | None = None) -> None:
        self.statements.append(str(statement))


def test_v0016_adds_candidate_identity_ledger_columns_and_indexes() -> None:
    conn = _FakeConnection()

    migration.upgrade(conn, TestSettings())

    ddl = "\n".join(conn.statements)
    assert "ALTER TABLE candidate_commits" in ddl
    assert "ALTER TABLE evaluation_attempts" in ddl
    assert "candidate_identity VARCHAR(512)" in ddl
    assert "evaluation_identity_key VARCHAR(64)" in ddl
    assert "ix_candidate_commits_evaluation_identity_key" in ddl
    assert "ix_evaluation_attempts_identity_key" in ddl
