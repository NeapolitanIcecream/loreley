from __future__ import annotations

from typing import Any

from loreley.db.migrations.versions import (
    v0021_append_only_evaluation_evidence as migration,
)
from tests.support import TestSettings


class _FakeConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, statement: Any, _params: dict[str, Any] | None = None) -> None:
        self.statements.append(str(statement))


def test_v0021_makes_attempt_evidence_append_only() -> None:
    connection = _FakeConnection()

    migration.upgrade(connection, TestSettings())

    ddl = "\n".join(connection.statements)
    assert "ADD COLUMN IF NOT EXISTS run_token UUID" in ddl
    assert "ADD COLUMN IF NOT EXISTS attempt_ordinal INTEGER" in ddl
    assert "artifact_paths JSONB NOT NULL" in ddl
    assert "row_number() OVER" in ddl
    assert "uq_evaluation_attempts_job_ordinal" in ddl
    assert "DROP CONSTRAINT IF EXISTS uq_evaluation_artifacts_job_key" in ddl
    assert "uq_evaluation_artifacts_attempt_key" in ddl
    assert "uq_evaluation_artifacts_legacy_job_key" in ddl
