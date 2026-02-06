from __future__ import annotations

from contextlib import contextmanager
from typing import Any, cast

from loreley.scheduler import ingestion as ingestion_mod
from loreley.scheduler.ingestion import MapElitesIngestion
from tests.support import TestSettings


def test_jobs_requiring_ingestion_does_not_require_experiment_filter(
    monkeypatch,
    tmp_path,
) -> None:
    """Ensure ingestion does not require experiment scoping."""

    settings = TestSettings(MAPELITES_CODE_EMBEDDING_DIMENSIONS=8)
    ingestion = MapElitesIngestion(
        settings=settings,
        console=ingestion_mod.Console(),
        repo_root=tmp_path,
        repo=cast(Any, object()),
        manager=cast(Any, object()),  # not used by _jobs_requiring_ingestion
    )

    class DummyResult:
        def scalars(self):  # pragma: no cover - trivial
            return []

    class DummySession:
        def execute(self, stmt: Any) -> DummyResult:
            try:
                params = stmt.compile().params
            except Exception:  # pragma: no cover - defensive
                params = {}
            assert "experiment_id" not in params
            return DummyResult()

    @contextmanager
    def fake_scope():
        yield DummySession()

    monkeypatch.setattr(ingestion_mod, "session_scope", fake_scope)

    snapshots = ingestion._jobs_requiring_ingestion(limit=5)
    assert snapshots == []

