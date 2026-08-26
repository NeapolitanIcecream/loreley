from __future__ import annotations

from typing import Any

from loreley.db.migrations.versions import v0022_seed_portfolios as migration
from loreley.db.models import (
    CandidateCommit,
    CommitCard,
    EvaluationAttempt,
    EvolutionJob,
    SeedDirection,
    SeedPortfolio,
)
from tests.support import TestSettings


class _FakeConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, statement: Any, _params: dict[str, Any] | None = None) -> None:
        self.statements.append(str(statement))


def test_v0022_adds_seed_portfolio_contract_and_lineage_provenance() -> None:
    connection = _FakeConnection()

    migration.upgrade(connection, TestSettings())

    ddl = "\n".join(connection.statements)
    assert "CREATE TABLE IF NOT EXISTS seed_portfolios" in ddl
    assert "CREATE TABLE IF NOT EXISTS seed_directions" in ddl
    assert "uq_seed_portfolios_request_fingerprint" in ddl
    assert "uq_seed_directions_portfolio_direction" in ddl
    assert "seed_direction_payload JSONB" in ddl
    assert "ALTER TABLE commit_cards" in ddl
    assert "ALTER TABLE candidate_commits" in ddl
    assert "ALTER TABLE evaluation_attempts" in ddl
    assert "ix_evolution_jobs_seed_direction" in ddl
    assert "ix_evaluation_attempts_seed_direction" in ddl


def test_seed_portfolio_models_expose_required_provenance_columns() -> None:
    assert {
        "request_fingerprint",
        "portfolio_hash",
        "root_commit_hash",
        "campaign_program_hash",
        "objective_contract_fingerprint",
        "input_evidence_fingerprints",
        "model_name",
        "reasoning_effort",
        "payload",
    }.issubset(SeedPortfolio.__table__.c.keys())
    assert {
        "portfolio_id",
        "direction_id",
        "content_hash",
        "admission_intent",
        "payload",
    }.issubset(SeedDirection.__table__.c.keys())
    for model in (EvolutionJob, CommitCard, CandidateCommit):
        assert {
            "seed_portfolio_hash",
            "seed_direction_id",
            "seed_admission_lane",
            "seed_admission_reason",
        }.issubset(model.__table__.c.keys())
    assert {
        "seed_portfolio_hash",
        "seed_direction_id",
    }.issubset(EvaluationAttempt.__table__.c.keys())
