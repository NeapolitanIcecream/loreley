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
    settings.mapelites_fitness_metric = "latency_ms"
    settings.mapelites_fitness_higher_is_better = False
    rows = [
        SimpleNamespace(
            commit_hash="c1",
            island_id="main",
            cell_index=1,
            objective=-12.5,
            measures=[0.1, 0.2],
            solution=[],
            timestamp=10.0,
        ),
        SimpleNamespace(
            commit_hash="c2",
            island_id="main",
            cell_index=2,
            objective=-10.0,
            measures=[0.3, 0.4],
            solution=[],
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
            if self.calls == 1:
                return _ExecResult(rows)
            return _ExecResult([("c1", 12.5)])

    session = _Session()

    @contextmanager
    def _fake_scope():
        yield session

    monkeypatch.setattr(archive_service, "session_scope", _fake_scope)

    page = archive_service.list_records_page(
        island_id="main",
        settings=settings,
        limit=1,
    )

    assert len(page.items) == 1
    assert page.items[0].fitness == pytest.approx(12.5)
    assert page.items[0].objective == pytest.approx(-12.5)
    assert page.next_cursor is not None
    assert decode_cursor(page.next_cursor)["cell_index"] == 1
    assert getattr(session.statements[0], "_offset_clause", None) is None
