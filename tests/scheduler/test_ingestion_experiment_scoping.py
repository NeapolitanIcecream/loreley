from __future__ import annotations

import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

from loreley.config import Settings
from loreley.scheduler import ingestion as ingestion_mod
from loreley.scheduler.ingestion import MapElitesIngestion


def test_jobs_requiring_ingestion_filters_by_experiment_id(
    monkeypatch,
    tmp_path,
) -> None:
    """Ensure ingestion never scans SUCCEEDED jobs across experiments."""

    settings = Settings()
    experiment_id = uuid.uuid4()
    experiment = SimpleNamespace(id=experiment_id)
    repository = SimpleNamespace(id=uuid.uuid4())

    ingestion = MapElitesIngestion(
        settings=settings,
        console=ingestion_mod.Console(),
        repo_root=tmp_path,
        repo=object(),
        manager=object(),  # not used by _jobs_requiring_ingestion
        experiment=experiment,
        repository=repository,
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
            # We don't parse the full SQL; it's enough to ensure the expected
            # experiment id is bound into the statement params.
            assert any(str(v) == str(experiment_id) for v in params.values())
            return DummyResult()

    @contextmanager
    def fake_scope():
        yield DummySession()

    monkeypatch.setattr(ingestion_mod, "session_scope", fake_scope)

    snapshots = ingestion._jobs_requiring_ingestion(limit=5)
    assert snapshots == []

