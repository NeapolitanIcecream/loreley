from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

import loreley.api.services.graphs as graph_service
from loreley.config import Settings


class _ExecResult:
    def __init__(self, *, rows=None):
        self._rows = list(rows or [])

    def scalars(self):
        return iter(self._rows)

    def all(self):
        return list(self._rows)


def _patch_no_graph_enrichment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(graph_service, "load_evidence_indicators_by_commit_hash", lambda _hashes: {})
    monkeypatch.setattr(graph_service, "load_candidate_fates_for_commits", lambda _commits: {})


def test_build_commit_lineage_graph_exposes_raw_metric_and_objective(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    settings.mapelites_fitness_metric = "latency_ms"
    settings.mapelites_fitness_higher_is_better = False
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    commit = SimpleNamespace(
        id=uuid4(),
        commit_hash="c1",
        parent_commit_hash=None,
        island_id="main",
        created_at=created_at,
        author="bot",
        subject="Improve latency",
    )

    class _SequencedSession:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, _stmt):
            self.calls += 1
            if self.calls == 1:
                return _ExecResult(rows=[commit])
            if self.calls == 2:
                return _ExecResult(rows=[(commit.id, 12.5)])
            raise AssertionError("unexpected extra query")

    @contextmanager
    def _fake_scope():
        yield _SequencedSession()

    monkeypatch.setattr(graph_service, "session_scope", _fake_scope)
    _patch_no_graph_enrichment(monkeypatch)

    graph = graph_service.build_commit_lineage_graph(max_nodes=1, settings=settings)

    assert graph.metric_name == "latency_ms"
    assert graph.higher_is_better is False
    assert graph.truncated is False
    assert len(graph.nodes) == 1
    assert graph.nodes[0].metric_value == pytest.approx(12.5)
    assert graph.nodes[0].fitness == pytest.approx(12.5)
    assert graph.nodes[0].objective == pytest.approx(-12.5)
    assert graph.nodes[0].agent_visible_evidence_count == 0


def test_build_commit_lineage_graph_marks_truncated_only_when_more_rows_exist(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    commits = [
        SimpleNamespace(
            id=uuid4(),
            commit_hash=f"c{idx}",
            parent_commit_hash=None,
            island_id="main",
            created_at=created_at,
            author="bot",
            subject=f"Commit {idx}",
        )
        for idx in range(2)
    ]

    class _SequencedSession:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, _stmt):
            self.calls += 1
            if self.calls == 1:
                return _ExecResult(rows=commits)
            if self.calls == 2:
                return _ExecResult(rows=[])
            raise AssertionError("unexpected extra query")

    @contextmanager
    def _fake_scope():
        yield _SequencedSession()

    monkeypatch.setattr(graph_service, "session_scope", _fake_scope)
    _patch_no_graph_enrichment(monkeypatch)

    graph = graph_service.build_commit_lineage_graph(max_nodes=2, settings=settings)

    assert len(graph.nodes) == 2
    assert graph.truncated is False
