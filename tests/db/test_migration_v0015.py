from __future__ import annotations

from typing import Any

from loreley.db.migrations.versions import v0015_multiobjective_islands as migration
from tests.support import TestSettings


class _FakeConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []
        self.params: list[dict[str, Any] | None] = []

    def execute(self, statement, params: dict[str, Any] | None = None):
        self.statements.append(str(statement))
        self.params.append(params)


def test_v0015_replaces_scalar_archive_and_marks_successes_for_reingestion() -> None:
    settings = TestSettings(
        MAPELITES_OBJECTIVES=[
            {"name": "quality", "direction": "max"},
            {"name": "latency_ms", "direction": "min"},
        ]
    )
    conn = _FakeConnection()

    migration.upgrade(conn, settings)

    ddl = "\n".join(conn.statements)
    assert "DROP TABLE map_elites_archive_cells" in ddl
    assert "objective_values DOUBLE PRECISION[] NOT NULL" in ddl
    assert "PRIMARY KEY (island_id, cell_index, commit_hash)" in ddl
    assert "UNIQUE (island_id, commit_hash)" in ddl
    assert "migration_source_island_id VARCHAR(64)" in ddl
    assert "migration_commit_hash VARCHAR(64)" in ddl
    assert "SET result_commit_hash = candidate_commit_hash" in ddl
    assert "ingestion_status = NULL" in ddl
    assert "status = 'SUCCEEDED'" in ddl
    assert "result_commit_hash IS NOT NULL" in ddl
    assert "objective_contract_fingerprint" in ddl
    assert "CAST(:objective_contract_fingerprint AS TEXT)" in ddl
    assert any(
        params and params.get("objective_contract_fingerprint")
        for params in conn.params
    )
