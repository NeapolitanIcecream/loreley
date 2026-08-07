from __future__ import annotations

from typing import Any

from loreley.db.migrations.versions import v0017_sampling_recipe_and_source_tree as migration
from tests.support import TestSettings


class _FakeConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, statement: Any, _params: dict[str, Any] | None = None) -> None:
        self.statements.append(str(statement))


def test_v0017_adds_sampling_recipe_and_source_tree_provenance() -> None:
    conn = _FakeConnection()

    migration.upgrade(conn, TestSettings())

    ddl = "\n".join(conn.statements)
    assert "sampling_ordinal INTEGER" in ddl
    assert "sampling_recipe_hash VARCHAR(64)" in ddl
    assert "sampling_recipe_reused BOOLEAN" in ddl
    assert "source_tree_hash VARCHAR(64)" in ddl
    assert "ix_evolution_jobs_island_recipe_created" in ddl
    assert "uq_evolution_jobs_island_sampling_ordinal" in ddl
    assert "ix_candidate_commits_source_tree_contract" in ddl
