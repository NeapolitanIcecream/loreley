from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
import uuid

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

    def all(self):
        return list(self._rows)

    def scalar_one_or_none(self):
        row = self._row
        if isinstance(row, tuple):
            if len(row) != 1:
                raise AssertionError("scalar_one_or_none() expected a single-column row")
            return row[0]
        return row

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

    class _SequencedSession:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, _stmt):
            self.calls += 1
            if self.calls == 1:
                return _ExecResult(row=(3, 5.5, 3.0))
            if self.calls == 2:
                return _ExecResult(row=(3.0,))
            raise AssertionError("unexpected extra query")

    @contextmanager
    def _fake_scope():
        yield _SequencedSession()

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
        "best_objective": pytest.approx(3.0),
        "metric_name": settings.mapelites_fitness_metric,
        "higher_is_better": settings.mapelites_fitness_higher_is_better,
    }


def test_list_records_reads_archive_cells_without_manager(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    settings.mapelites_fitness_metric = ""
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


def test_describe_island_uses_raw_metric_value_for_best_fitness_when_lower_is_better(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    settings.mapelites_dimensionality_target_dims = 2
    settings.mapelites_archive_cells_per_dim = 4
    settings.mapelites_fitness_metric = "latency_ms"
    settings.mapelites_fitness_higher_is_better = False

    class _SequencedSession:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, _stmt):
            self.calls += 1
            if self.calls == 1:
                return _ExecResult(row=(3, -5.5, -1.0))
            if self.calls == 2:
                return _ExecResult(row=(12.5,))
            raise AssertionError("unexpected extra query")

    @contextmanager
    def _fake_scope():
        yield _SequencedSession()

    monkeypatch.setattr(archive_service, "session_scope", _fake_scope)

    stats = archive_service.describe_island(island_id="main", settings=settings)

    assert stats["best_fitness"] == pytest.approx(12.5)
    assert stats["best_objective"] == pytest.approx(-1.0)
    assert stats["metric_name"] == "latency_ms"
    assert stats["higher_is_better"] is False


def test_list_records_exposes_metric_value_separately_from_objective(
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
    ]

    class _SequencedSession:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, _stmt):
            self.calls += 1
            if self.calls == 1:
                return _ExecResult(rows=rows)
            if self.calls == 2:
                return _ExecResult(rows=[("c1", 12.5)])
            raise AssertionError("unexpected extra query")

    @contextmanager
    def _fake_scope():
        yield _SequencedSession()

    monkeypatch.setattr(archive_service, "session_scope", _fake_scope)

    records = archive_service.list_records(
        island_id="main",
        settings=settings,
        limit=10,
        offset=0,
    )

    assert len(records) == 1
    assert records[0].fitness == pytest.approx(12.5)
    assert records[0].metric_value == pytest.approx(12.5)
    assert records[0].objective == pytest.approx(-12.5)


def test_list_records_scopes_baseline_to_each_candidate_campaign_program(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    """Regression: archive deltas must not reuse the latest baseline from another campaign."""

    settings.mapelites_experiment_root_commit = "root123"
    settings.mapelites_fitness_metric = "score"
    settings.mapelites_fitness_higher_is_better = True
    rows = [
        SimpleNamespace(
            commit_hash="commit-a",
            island_id="main",
            cell_index=1,
            objective=20.0,
            measures=[0.1],
            solution=[],
            timestamp=10.0,
        ),
        SimpleNamespace(
            commit_hash="commit-b",
            island_id="main",
            cell_index=2,
            objective=7.0,
            measures=[0.2],
            solution=[],
            timestamp=11.0,
        ),
    ]
    baseline_a_id = uuid.UUID("aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa")
    baseline_b_id = uuid.UUID("bbbbbbbb-bbbb-4bbb-bbbb-bbbbbbbbbbbb")
    baseline_a = SimpleNamespace(
        id=baseline_a_id,
        baseline_key_hash="a" * 64,
        status="valid",
        metric_value=10.0,
        primary_metric_higher_is_better=True,
    )
    baseline_b = SimpleNamespace(
        id=baseline_b_id,
        baseline_key_hash="b" * 64,
        status="valid",
        metric_value=5.0,
        primary_metric_higher_is_better=True,
    )
    baseline_calls: list[tuple[str | None, bool]] = []

    def _load_baseline(**kwargs):
        campaign_program_hash = kwargs.get("campaign_program_hash")
        valid_only = bool(kwargs.get("valid_only", False))
        baseline_calls.append((campaign_program_hash, valid_only))
        if campaign_program_hash == "program-a":
            return baseline_a
        if campaign_program_hash == "program-b":
            return baseline_b
        return baseline_b

    class _SequencedSession:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, _stmt):
            self.calls += 1
            if self.calls == 1:
                return _ExecResult(rows=rows)
            if self.calls == 2:
                return _ExecResult(rows=[("commit-a", 20.0), ("commit-b", 7.0)])
            if self.calls == 3:
                return _ExecResult(rows=[("commit-a", "program-a"), ("commit-b", "program-b")])
            raise AssertionError("unexpected extra query")

    @contextmanager
    def _fake_scope():
        yield _SequencedSession()

    monkeypatch.setattr(archive_service, "session_scope", _fake_scope)
    monkeypatch.setattr(archive_service, "load_latest_matching_baseline", _load_baseline)

    records = archive_service.list_records(
        island_id="main",
        settings=settings,
        limit=10,
        offset=0,
    )

    assert [record.campaign_baseline_id for record in records] == [
        str(baseline_a_id),
        str(baseline_b_id),
    ]
    assert [record.delta_from_root_baseline for record in records] == pytest.approx([10.0, 2.0])
    assert set(baseline_calls) == {("program-a", False), ("program-b", False)}


def test_list_records_surfaces_degraded_baseline_without_delta(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    """Regression: warn-policy degraded baselines stay visible while deltas remain unavailable."""

    settings.mapelites_experiment_root_commit = "root123"
    settings.mapelites_fitness_metric = "score"
    settings.mapelites_fitness_higher_is_better = True
    rows = [
        SimpleNamespace(
            commit_hash="commit-a",
            island_id="main",
            cell_index=1,
            objective=20.0,
            measures=[0.1],
            solution=[],
            timestamp=10.0,
        ),
    ]
    degraded_id = uuid.UUID("dddddddd-dddd-4ddd-dddd-dddddddddddd")
    degraded = SimpleNamespace(
        id=degraded_id,
        baseline_key_hash="d" * 64,
        status="degraded",
        metric_value=None,
        primary_metric_higher_is_better=True,
    )
    baseline_calls: list[tuple[str | None, bool]] = []

    def _load_baseline(**kwargs):
        baseline_calls.append((kwargs.get("campaign_program_hash"), bool(kwargs.get("valid_only", False))))
        if kwargs.get("valid_only"):
            return None
        return degraded

    class _SequencedSession:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, _stmt):
            self.calls += 1
            if self.calls == 1:
                return _ExecResult(rows=rows)
            if self.calls == 2:
                return _ExecResult(rows=[("commit-a", 20.0)])
            if self.calls == 3:
                return _ExecResult(rows=[("commit-a", "program-a")])
            raise AssertionError("unexpected extra query")

    @contextmanager
    def _fake_scope():
        yield _SequencedSession()

    monkeypatch.setattr(archive_service, "session_scope", _fake_scope)
    monkeypatch.setattr(archive_service, "load_latest_matching_baseline", _load_baseline)

    records = archive_service.list_records(
        island_id="main",
        settings=settings,
        limit=10,
        offset=0,
    )

    assert len(records) == 1
    assert records[0].campaign_baseline_id == str(degraded_id)
    assert records[0].baseline_status == "degraded"
    assert records[0].delta_from_root_baseline is None
    assert baseline_calls == [("program-a", False)]
