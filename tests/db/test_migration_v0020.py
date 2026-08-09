from __future__ import annotations

from typing import Any

from loreley.db.migrations.versions import v0020_supplied_candidates as migration
from loreley.db.models import EvolutionJob, MapElitesPcaHistory
from tests.support import TestSettings


class _FakeConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, statement: Any, _params: dict[str, Any] | None = None) -> None:
        self.statements.append(str(statement))


def test_v0020_adds_staged_supplied_candidate_contract() -> None:
    connection = _FakeConnection()

    migration.upgrade(connection, TestSettings())

    ddl = "\n".join(connection.statements)
    assert "ALTER TYPE job_status ADD VALUE IF NOT EXISTS 'STAGED'" in ddl
    assert "execution_mode VARCHAR(32) NOT NULL DEFAULT 'agent'" in ddl
    assert "input_candidate_commit_hash VARCHAR(64)" in ddl
    assert "input_provenance JSONB NOT NULL" in ddl
    assert "archive_ingestion_enabled BOOLEAN NOT NULL DEFAULT TRUE" in ddl
    assert "uq_evolution_jobs_external_submission_key" in ddl
    assert "uq_evolution_jobs_manual_seed_commit" in ddl
    assert "ck_evolution_jobs_execution_input" in ddl
    assert "ck_evolution_jobs_manual_seed_contract" in ddl


def test_supplied_candidate_constraints_belong_to_evolution_jobs() -> None:
    job_constraints = {constraint.name for constraint in EvolutionJob.__table__.constraints}
    history_constraints = {
        constraint.name for constraint in MapElitesPcaHistory.__table__.constraints
    }

    assert "ck_evolution_jobs_execution_input" in job_constraints
    assert "ck_evolution_jobs_manual_seed_contract" in job_constraints
    assert "ck_evolution_jobs_execution_input" not in history_constraints
    assert "ck_evolution_jobs_manual_seed_contract" not in history_constraints
