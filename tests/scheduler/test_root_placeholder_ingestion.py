from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast
from types import SimpleNamespace

from sqlalchemy.exc import MultipleResultsFound

from loreley.core.worker.evaluator import EvaluationArtifact, EvaluationMetric, EvaluationResult
from loreley.db.models import CommitCard, EvaluationArtifactRecord, Metric
from loreley.scheduler import ingestion as ingestion_mod
from loreley.scheduler.ingestion import MapElitesIngestion
from tests.support import TestSettings


class DummyManager:
    """Lightweight stub that records ingest calls for assertions."""

    def __init__(self) -> None:
        self.ingest_calls: list[dict[str, Any]] = []

    def get_records(self, island_id: str | None = None) -> tuple[Any, ...]:
        # Root archives start empty for this test.
        return ()

    def ingest(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - defensive
        self.ingest_calls.append({"args": args, "kwargs": kwargs})


def test_root_initialisation_evaluates_without_ingesting_into_archive(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Root initialisation should register and evaluate the root commit only.

    In particular, it must not attempt to ingest the root into any MAP-Elites
    archive or rely on placeholder files.
    """

    settings = TestSettings(MAPELITES_CODE_EMBEDDING_DIMENSIONS=8)
    manager = DummyManager()
    ingestion = MapElitesIngestion(
        settings=settings,
        console=ingestion_mod.Console(),
        repo_root=tmp_path,
        repo=cast(Any, object()),
        manager=manager,  # type: ignore[arg-type]
    )

    calls: dict[str, int] = {"available": 0, "metadata": 0, "repo_state_bootstrap": 0, "evaluated": 0}

    def _fake_ensure_available(self: MapElitesIngestion, commit_hash: str) -> None:
        calls["available"] += 1

    def _fake_ensure_metadata(self: MapElitesIngestion, commit_hash: str) -> None:
        calls["metadata"] += 1

    def _fake_repo_state_bootstrap(self: MapElitesIngestion, commit_hash: str) -> None:
        calls["repo_state_bootstrap"] += 1

    def _fake_ensure_evaluated(self: MapElitesIngestion, commit_hash: str) -> None:
        calls["evaluated"] += 1

    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_ensure_commit_available",
        _fake_ensure_available,
    )
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_ensure_root_commit_metadata",
        _fake_ensure_metadata,
    )
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_ensure_root_commit_repo_state_bootstrap",
        _fake_repo_state_bootstrap,
    )
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_ensure_root_commit_evaluated",
        _fake_ensure_evaluated,
    )

    root_hash = "root123"
    ingestion.initialise_root_commit(root_hash)

    assert calls["available"] == 1
    assert calls["metadata"] == 1
    assert calls["repo_state_bootstrap"] == 1
    assert calls["evaluated"] == 1
    # Root initialisation should not attempt to ingest the root commit into any
    # MAP-Elites archive.
    assert manager.ingest_calls == []


def test_root_commit_evaluation_skips_when_metrics_already_exist_even_if_multiple(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Regression: root evaluation should skip when any existing metric row is present."""

    settings = TestSettings(MAPELITES_CODE_EMBEDDING_DIMENSIONS=8)
    ingestion = MapElitesIngestion(
        settings=settings,
        console=ingestion_mod.Console(),
        repo_root=tmp_path,
        repo=cast(Any, object()),
        manager=DummyManager(),  # type: ignore[arg-type]
    )

    commit_row = SimpleNamespace(id="card-1")

    class _CommitResult:
        def scalar_one_or_none(self) -> object:
            return commit_row

    class _MetricExistsResult:
        def scalar_one_or_none(self) -> object:
            raise MultipleResultsFound("multiple metrics exist")

        def scalar(self) -> object:
            return "metric-1"

        def first(self) -> tuple[str]:
            return ("metric-1",)

    class _Session:
        def __init__(self) -> None:
            self._calls = 0

        def execute(self, _stmt: object) -> object:
            self._calls += 1
            if self._calls == 1:
                return _CommitResult()
            return _MetricExistsResult()

    @contextmanager
    def _fake_scope():
        yield _Session()

    monkeypatch.setattr(ingestion_mod, "session_scope", _fake_scope)

    worker_repo_calls: list[object] = []

    class _ForbiddenWorkerRepository:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            worker_repo_calls.append(object())

    monkeypatch.setattr(ingestion_mod, "WorkerRepository", _ForbiddenWorkerRepository)

    ingestion._ensure_root_commit_evaluated("abc123")

    assert worker_repo_calls == []


def test_root_commit_evaluation_ignores_evaluator_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = TestSettings(MAPELITES_CODE_EMBEDDING_DIMENSIONS=8)

    class _Repo:
        def commit(self, _commit_hash: str) -> object:
            return SimpleNamespace(
                parents=[],
                author=SimpleNamespace(name="author"),
                message="Root subject\n\nbody",
            )

    ingestion = MapElitesIngestion(
        settings=settings,
        console=ingestion_mod.Console(),
        repo_root=tmp_path,
        repo=cast(Any, _Repo()),
        manager=DummyManager(),  # type: ignore[arg-type]
    )

    class _ScalarNone:
        def scalar_one_or_none(self) -> object:
            return None

        def first(self) -> object:
            return None

    added: list[object] = []
    sessions: list[object] = []

    class _Session:
        def __init__(self) -> None:
            self.calls = 0
            sessions.append(self)

        def execute(self, _stmt: object) -> _ScalarNone:
            self.calls += 1
            return _ScalarNone()

        def add(self, obj: object) -> None:
            added.append(obj)

    @contextmanager
    def _fake_scope():
        yield _Session()

    class _WorkerRepository:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        @contextmanager
        def checkout_lease_for_job(self, **_kwargs: object):
            yield SimpleNamespace(worktree=tmp_path)

    class _Evaluator:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def evaluate(self, context: object) -> EvaluationResult:
            assert getattr(context, "job_id") is None
            return EvaluationResult(
                summary="baseline summary",
                metrics=(EvaluationMetric(name="score", value=1.0),),
                artifacts=(
                    EvaluationArtifact(
                        key="baseline_report",
                        kind="benchmark_json",
                        mime_type="application/json",
                        inline_payload={"score": 1.0},
                        summary="should not be persisted",
                        visibility="agent_visible",
                    ),
                ),
            )

    monkeypatch.setattr(ingestion_mod, "session_scope", _fake_scope)
    monkeypatch.setattr(ingestion_mod, "WorkerRepository", _WorkerRepository)
    monkeypatch.setattr(ingestion_mod, "Evaluator", _Evaluator)

    ingestion._ensure_root_commit_evaluated("abc123")

    assert any(isinstance(obj, CommitCard) for obj in added)
    assert any(isinstance(obj, Metric) for obj in added)
    assert not any(isinstance(obj, EvaluationArtifactRecord) for obj in added)
