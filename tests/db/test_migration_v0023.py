from __future__ import annotations

from typing import Any

from loreley.db.migrations.versions import v0023_evolution_events as migration
from loreley.db.models import EvolutionEvent
from tests.support import TestSettings


class _FakeConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, statement: Any, _params: dict[str, Any] | None = None) -> None:
        self.statements.append(str(statement))


def test_v0023_adds_append_only_evolution_events_and_boundary() -> None:
    connection = _FakeConnection()

    migration.upgrade(connection, TestSettings())

    ddl = "\n".join(connection.statements)
    assert "CREATE TABLE IF NOT EXISTS evolution_events" in ddl
    assert "uq_evolution_events_event_key" in ddl
    assert "ix_evolution_events_job_timeline" in ddl
    assert "ix_evolution_events_type_timeline" in ddl
    assert "timeline.history_boundary" in ddl
    assert "archive.member.initial_state" in ddl
    assert "FROM map_elites_archive_cells" in ddl


def test_evolution_event_model_exposes_required_timeline_columns() -> None:
    assert {
        "id",
        "event_key",
        "event_type",
        "job_id",
        "run_token",
        "island_id",
        "commit_hash",
        "occurred_at",
        "ordinal",
        "duration_seconds",
        "payload",
    } == set(EvolutionEvent.__table__.c.keys())
