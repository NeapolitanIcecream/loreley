from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import MultipleResultsFound

import loreley.core.map_elites.repository_state_embedding as repo_state_mod
from loreley.core.worker.evaluator import EvaluationArtifact, EvaluationMetric, EvaluationResult
from loreley.db.models import CommitCard, EvaluationArtifactRecord, Metric
from loreley.scheduler import ingestion as ingestion_mod
from loreley.scheduler.ingestion import IngestionError, MapElitesIngestion
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


def test_root_initialisation_bootstraps_repo_state_without_ingesting_into_archive(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Root initialisation should register root metadata and repo-state only.

    In particular, it must not attempt to ingest the root into any MAP-Elites
    archive or run the campaign evaluator baseline service.
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
    assert calls["evaluated"] == 0
    # Root initialisation should not attempt to ingest the root commit into any
    # MAP-Elites archive.
    assert manager.ingest_calls == []


def test_root_initialisation_stops_when_root_commit_is_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = TestSettings(MAPELITES_CODE_EMBEDDING_DIMENSIONS=8)
    ingestion = MapElitesIngestion(
        settings=settings,
        console=ingestion_mod.Console(),
        repo_root=tmp_path,
        repo=cast(Any, object()),
        manager=DummyManager(),  # type: ignore[arg-type]
    )

    calls: list[str] = []

    def _fail_available(self: MapElitesIngestion, commit_hash: str) -> str:
        calls.append("available")
        raise IngestionError("missing root")

    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_ensure_commit_available",
        _fail_available,
    )
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_ensure_root_commit_metadata",
        lambda *_args, **_kwargs: calls.append("metadata"),
    )
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_ensure_root_commit_repo_state_bootstrap",
        lambda *_args, **_kwargs: calls.append("repo_state_bootstrap"),
    )
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_ensure_root_commit_evaluated",
        lambda *_args, **_kwargs: calls.append("evaluated"),
    )

    with pytest.raises(IngestionError, match="missing root"):
        ingestion.initialise_root_commit("bad-root")

    assert calls == ["available"]


def test_root_repo_state_bootstrap_fails_when_no_embedding_is_created(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = TestSettings(MAPELITES_CODE_EMBEDDING_DIMENSIONS=8)
    ingestion = MapElitesIngestion(
        settings=settings,
        console=ingestion_mod.Console(),
        repo_root=tmp_path,
        repo=cast(Any, object()),
        manager=DummyManager(),  # type: ignore[arg-type]
    )

    stats = SimpleNamespace(
        eligible_files=2,
        files_aggregated=0,
        skipped_failed_embedding=2,
    )
    monkeypatch.setattr(
        repo_state_mod,
        "bootstrap_repository_state_aggregate",
        lambda **_kwargs: (None, stats),
    )

    with pytest.raises(IngestionError, match="Repo-state bootstrap produced no embedding"):
        ingestion._ensure_root_commit_repo_state_bootstrap("root123")


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
    ingestion = MapElitesIngestion(
        settings=settings,
        console=ingestion_mod.Console(),
        repo_root=tmp_path,
        repo=cast(Any, _RootEvalRepo()),
        manager=DummyManager(),  # type: ignore[arg-type]
    )
    added: list[object] = []
    monkeypatch.setattr(ingestion_mod, "session_scope", _root_eval_scope_factory(added))
    monkeypatch.setattr(ingestion_mod, "WorkerRepository", _root_eval_worker_repository(tmp_path))
    monkeypatch.setattr(ingestion_mod, "Evaluator", _RootArtifactEvaluator)

    ingestion._ensure_root_commit_evaluated("abc123")

    assert any(isinstance(obj, CommitCard) for obj in added)
    assert any(isinstance(obj, Metric) for obj in added)
    assert not any(isinstance(obj, EvaluationArtifactRecord) for obj in added)


class _RootEvalRepo:
    def commit(self, _commit_hash: str) -> object:
        return SimpleNamespace(
            parents=[],
            author=SimpleNamespace(name="author"),
            message="Root subject\n\nbody",
        )


class _RootEvalScalarNone:
    def scalar_one_or_none(self) -> object:
        return None

    def first(self) -> object:
        return None


class _RootEvalSession:
    def __init__(self, added: list[object]) -> None:
        self.calls = 0
        self.added = added

    def execute(self, _stmt: object) -> _RootEvalScalarNone:
        self.calls += 1
        return _RootEvalScalarNone()

    def add(self, obj: object) -> None:
        self.added.append(obj)


def _root_eval_scope_factory(added: list[object]):
    @contextmanager
    def _fake_scope():
        yield _RootEvalSession(added)

    return _fake_scope


def _root_eval_worker_repository(tmp_path: Path):
    class _WorkerRepository:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        @contextmanager
        def checkout_lease_for_job(self, **_kwargs: object):
            yield SimpleNamespace(worktree=tmp_path)

    return _WorkerRepository


class _RootArtifactEvaluator:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def evaluate(self, context: object) -> EvaluationResult:
        assert getattr(context, "job_id") is None
        return EvaluationResult(
            summary="baseline summary",
            metrics=(EvaluationMetric(name="score", value=1.0),),
            artifacts=(_root_eval_artifact(),),
        )


def _root_eval_artifact() -> EvaluationArtifact:
    return EvaluationArtifact(
        key="baseline_report",
        kind="benchmark_json",
        mime_type="application/json",
        inline_payload={"score": 1.0},
        summary="should not be persisted",
        visibility="agent_visible",
    )
