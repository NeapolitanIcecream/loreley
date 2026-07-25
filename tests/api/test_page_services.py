from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

import loreley.api.services.archive as archive_service
import loreley.api.services.commits as commits_service
import loreley.api.services.jobs as jobs_service
from loreley.api.pagination import decode_cursor
from loreley.config import Settings
from loreley.core.map_elites.objectives import ObjectiveSpec
from loreley.db.models import JobStatus


class _ScalarResult:
    def __init__(self, rows):
        self._rows = list(rows)

    def __iter__(self):
        return iter(self._rows)

    def all(self):
        return list(self._rows)


class _ExecResult:
    def __init__(self, rows):
        self._rows = list(rows)

    def scalars(self):
        return _ScalarResult(self._rows)

    def all(self):
        return list(self._rows)


def test_list_jobs_page_returns_next_cursor_without_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2026, 3, 11, tzinfo=timezone.utc)
    rows = [
        SimpleNamespace(id=uuid4(), completed_at=now, created_at=now, status=JobStatus.SUCCEEDED),
        SimpleNamespace(id=uuid4(), completed_at=now, created_at=now, status=JobStatus.SUCCEEDED),
        SimpleNamespace(id=uuid4(), completed_at=now, created_at=now, status=JobStatus.SUCCEEDED),
    ]
    statements: list[object] = []

    class _Session:
        def execute(self, stmt):
            statements.append(stmt)
            return _ExecResult(rows)

    @contextmanager
    def _fake_scope():
        yield _Session()

    monkeypatch.setattr(jobs_service, "session_scope", _fake_scope)

    page = jobs_service.list_jobs_page(limit=2)

    assert len(page.items) == 2
    assert page.next_cursor is not None
    assert decode_cursor(page.next_cursor)["job_id"] == str(rows[1].id)
    assert getattr(statements[0], "_offset_clause", None) is None


def test_list_jobs_page_applies_job_kind_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2026, 3, 11, tzinfo=timezone.utc)
    rows = [
        SimpleNamespace(id=uuid4(), completed_at=now, created_at=now, status=JobStatus.SUCCEEDED),
    ]
    statements: list[object] = []

    class _Session:
        def execute(self, stmt):
            statements.append(stmt)
            return _ExecResult(rows)

    @contextmanager
    def _fake_scope():
        yield _Session()

    monkeypatch.setattr(jobs_service, "session_scope", _fake_scope)

    page = jobs_service.list_jobs_page(job_kind="repair", limit=2)

    assert len(page.items) == 1
    sql = str(statements[0].compile())
    params = {str(key): value for key, value in statements[0].compile().params.items()}
    assert "evolution_jobs.job_kind" in sql
    assert "repair" in params.values()


def test_list_jobs_page_pushes_candidate_fate_filter_into_sql(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: sparse candidate_fate filters must not scan unbounded Python batches."""

    now = datetime(2026, 3, 11, tzinfo=timezone.utc)
    rows = [
        SimpleNamespace(id=uuid4(), completed_at=now, created_at=now, status=JobStatus.SUCCEEDED),
    ]
    statements: list[object] = []

    class _Session:
        def execute(self, stmt):
            statements.append(stmt)
            return _ExecResult(rows)

    @contextmanager
    def _fake_scope():
        yield _Session()

    monkeypatch.setattr(jobs_service, "session_scope", _fake_scope)
    page = jobs_service.list_jobs_page(candidate_fate="elite_inserted", limit=1)

    assert not hasattr(jobs_service, "load_candidate_fates_for_jobs")
    assert page.items == rows
    assert page.next_cursor is None
    assert len(statements) == 1
    compiled = statements[0].compile()
    sql = str(compiled)
    params = {str(key): value for key, value in compiled.params.items()}
    assert "candidate_commits" in sql
    assert "map_elites_archive_cells" in sql
    assert "elite_inserted" in params.values()


def test_list_jobs_pushes_evidence_filter_into_sql(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: sparse evidence filters must not scan unbounded Python batches."""

    now = datetime(2026, 3, 11, tzinfo=timezone.utc)
    rows = [
        SimpleNamespace(
            id=uuid4(),
            completed_at=now,
            created_at=now,
            status=JobStatus.SUCCEEDED,
            result_commit_hash="c3",
        ),
    ]
    statements: list[object] = []

    class _Session:
        def execute(self, stmt):
            statements.append(stmt)
            return _ExecResult(rows)

    @contextmanager
    def _fake_scope():
        yield _Session()

    monkeypatch.setattr(jobs_service, "session_scope", _fake_scope)
    result = jobs_service.list_jobs(evidence="agent_visible", limit=1, offset=1)

    assert not hasattr(jobs_service, "load_evidence_indicators_by_commit_hash")
    assert result == rows
    assert len(statements) == 1
    compiled = statements[0].compile()
    sql = str(compiled)
    params = {str(key): value for key, value in compiled.params.items()}
    assert "evaluation_artifacts" in sql
    assert "agent_visible" in params.values()
    assert 1 in params.values()


def test_list_jobs_projection_filter_uses_one_sql_query_for_large_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: offset pagination with sparse projection filters must not batch-scan."""

    statements: list[object] = []

    class _Session:
        def execute(self, stmt):
            statements.append(stmt)
            return _ExecResult([])

    @contextmanager
    def _fake_scope():
        yield _Session()

    monkeypatch.setattr(jobs_service, "session_scope", _fake_scope)
    result = jobs_service.list_jobs(evidence="has_evidence", limit=1, offset=200_000)

    assert not hasattr(jobs_service, "load_evidence_indicators_by_commit_hash")
    assert result == []
    assert len(statements) == 1
    compiled = statements[0].compile()
    sql = str(compiled)
    params = {str(key): value for key, value in compiled.params.items()}
    assert "evaluation_artifacts" in sql
    assert 200_000 in params.values()


def test_list_commits_page_applies_cursor_without_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2026, 3, 11, tzinfo=timezone.utc)
    rows = [
        SimpleNamespace(id=uuid4(), created_at=now, island_id="main"),
        SimpleNamespace(id=uuid4(), created_at=now, island_id="main"),
        SimpleNamespace(id=uuid4(), created_at=now, island_id="main"),
    ]
    statements: list[object] = []

    class _Session:
        def execute(self, stmt):
            statements.append(stmt)
            return _ExecResult(rows)

    @contextmanager
    def _fake_scope():
        yield _Session()

    monkeypatch.setattr(commits_service, "session_scope", _fake_scope)

    first_page = commits_service.list_commits_page(island_id="main", limit=2)
    second_page = commits_service.list_commits_page(
        island_id="main",
        limit=2,
        cursor=first_page.next_cursor,
    )

    assert len(first_page.items) == 2
    assert second_page.next_cursor is not None
    assert all(getattr(stmt, "_offset_clause", None) is None for stmt in statements)


def test_list_commits_page_applies_server_side_query_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2026, 3, 11, tzinfo=timezone.utc)
    rows = [
        SimpleNamespace(id=uuid4(), created_at=now, island_id="main"),
        SimpleNamespace(id=uuid4(), created_at=now, island_id="main"),
    ]
    statements: list[object] = []

    class _Session:
        def execute(self, stmt):
            statements.append(stmt)
            return _ExecResult(rows)

    @contextmanager
    def _fake_scope():
        yield _Session()

    monkeypatch.setattr(commits_service, "session_scope", _fake_scope)

    page = commits_service.list_commits_page(island_id="main", query="bugfix", limit=2)

    assert len(page.items) == 2
    assert len(statements) == 1
    compiled = statements[0].compile()
    params = {str(key): value for key, value in compiled.params.items()}
    assert any(value == "%bugfix%" for value in params.values())
    sql = str(compiled)
    assert "commit_cards.commit_hash" in sql
    assert "commit_cards.author" in sql
    assert "commit_cards.subject" in sql
    assert "commit_cards.change_summary" in sql
    assert getattr(statements[0], "_offset_clause", None) is None


def test_list_records_page_returns_cursor_and_metric_value(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    settings.mapelites_objectives = (
        ObjectiveSpec(name="latency_ms", direction="min"),
    )
    rows = [
        SimpleNamespace(
            commit_hash="c1",
            island_id="main",
            cell_index=1,
            objective_values=[12.5],
            measures=[0.1, 0.2],
            timestamp=10.0,
        ),
        SimpleNamespace(
            commit_hash="c2",
            island_id="main",
            cell_index=2,
            objective_values=[10.0],
            measures=[0.3, 0.4],
            timestamp=11.0,
        ),
    ]

    class _Session:
        def __init__(self) -> None:
            self.calls = 0
            self.statements: list[object] = []

        def execute(self, stmt):
            self.statements.append(stmt)
            self.calls += 1
            return _ExecResult(rows)

    session = _Session()

    @contextmanager
    def _fake_scope():
        yield session

    monkeypatch.setattr(archive_service, "session_scope", _fake_scope)
    monkeypatch.setattr(
        archive_service,
        "_validate_persisted_objective_contract",
        lambda **_kwargs: None,
    )

    page = archive_service.list_records_page(
        island_id="main",
        settings=settings,
        limit=1,
    )

    assert len(page.items) == 1
    assert page.items[0].primary_metric_value == pytest.approx(12.5)
    assert page.items[0].objective_scores == pytest.approx((-12.5,))
    assert page.next_cursor is not None
    assert decode_cursor(page.next_cursor)["cell_index"] == 1
    assert decode_cursor(page.next_cursor)["commit_hash"] == "c1"
    assert getattr(session.statements[0], "_offset_clause", None) is None
