from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import loreley.api.services.archive as archive_service
from loreley.config import Settings


class _ScalarRows:
    def __init__(self, rows):
        self._rows = list(rows)

    def all(self):
        return list(self._rows)


class _ExecResult:
    def __init__(self, *, row=None, rows=None):
        self._row = row
        self._rows = list(rows or [])

    def one(self):
        return self._row

    def scalars(self):
        return _ScalarRows(self._rows)


class _FakeSession:
    def __init__(self, result: _ExecResult) -> None:
        self._result = result
        self.statements = []

    def execute(self, _stmt):
        self.statements.append(_stmt)
        return self._result


def test_describe_island_reads_stats_from_archive_rows(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    settings.mapelites_dimensionality_target_dims = 2
    settings.mapelites_archive_cells_per_dim = 4

    @contextmanager
    def _fake_scope():
        yield _FakeSession(_ExecResult(row=(3, 5.5, 3.0)))

    monkeypatch.setattr(archive_service, "session_scope", _fake_scope)

    stats = archive_service.describe_island(island_id="main", settings=settings)

    assert stats == {
        "island_id": "main",
        "occupied": 3,
        "cells": 16,
        "coverage": pytest.approx(3 / 16),
        "qd_score": pytest.approx(5.5),
        "norm_qd_score": pytest.approx(5.5 / 16),
        "best_fitness": pytest.approx(3.0),
    }


def test_list_records_reads_archive_cells_without_manager(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    rows = [
        SimpleNamespace(
            commit_hash="c1",
            island_id="main",
            cell_index=1,
            objective=1.2,
            measures=[0.1, 0.2],
            solution=[],
            timestamp=10.0,
        ),
        SimpleNamespace(
            commit_hash="c2",
            island_id="main",
            cell_index=2,
            objective=2.4,
            measures=[0.5, 0.6],
            solution=[0.7, 0.8],
            timestamp=11.0,
        ),
    ]
    session = _FakeSession(_ExecResult(rows=rows))

    @contextmanager
    def _fake_scope():
        yield session

    monkeypatch.setattr(archive_service, "session_scope", _fake_scope)

    records = archive_service.list_records(
        island_id="main",
        settings=settings,
        limit=1,
        offset=1,
    )

    assert [record.commit_hash for record in records] == ["c1", "c2"]
    assert records[0].measures == pytest.approx((0.1, 0.2))
    assert records[0].solution == pytest.approx((0.1, 0.2))
    assert records[1].solution == pytest.approx((0.7, 0.8))
    stmt = session.statements[0]
    assert int(stmt._limit_clause.value) == 1
    assert int(stmt._offset_clause.value) == 1
