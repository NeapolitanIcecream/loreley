from __future__ import annotations

from typing import Any

from loreley.db.migrations.versions import v0019_evaluation_runtime as migration
from tests.support import TestSettings


class _FakeConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, statement: Any, _params: dict[str, Any] | None = None) -> None:
        self.statements.append(str(statement))


def test_v0019_adds_phased_measurement_and_evaluator_runtime_contracts() -> None:
    conn = _FakeConnection()

    migration.upgrade(conn, TestSettings())

    ddl = "\n".join(conn.statements)
    assert "CREATE TABLE IF NOT EXISTS evaluation_measurements" in ddl
    assert "CONSTRAINT uq_evaluation_measurements_cache_key UNIQUE (cache_key)" in ddl
    assert "CREATE TABLE IF NOT EXISTS evaluation_concurrency_contracts" in ddl
    assert "experiment_id VARCHAR(128) NOT NULL" in ddl
    assert "CREATE TABLE IF NOT EXISTS evaluation_resource_leases" in ddl
    assert "measurement_executed BOOLEAN NOT NULL DEFAULT FALSE" in ddl
    assert "reuse_kind VARCHAR(32) NOT NULL DEFAULT 'none'" in ddl
    assert "evaluator_slot_acquired_at TIMESTAMP WITH TIME ZONE" in ddl
    assert "evaluator_slot_released_at TIMESTAMP WITH TIME ZONE" in ddl
    assert "failure_kind VARCHAR(64)" in ddl
