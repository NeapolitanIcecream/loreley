from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from loreley.config import Settings
from loreley.scheduler import ingestion as ingestion_mod
from loreley.scheduler.ingestion import MapElitesIngestion


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

    settings = Settings()
    manager = DummyManager()
    experiment = SimpleNamespace(id="exp-123")
    repository = SimpleNamespace(id="repo-456")

    ingestion = MapElitesIngestion(
        settings=settings,
        console=ingestion_mod.Console(),
        repo_root=tmp_path,
        repo=object(),
        manager=manager,  # type: ignore[arg-type]
        experiment=experiment,
        repository=repository,
    )

    calls: dict[str, int] = {"available": 0, "metadata": 0, "evaluated": 0}

    def _fake_ensure_available(self: MapElitesIngestion, commit_hash: str) -> None:
        calls["available"] += 1

    def _fake_ensure_metadata(self: MapElitesIngestion, commit_hash: str) -> None:
        calls["metadata"] += 1

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
        "_ensure_root_commit_evaluated",
        _fake_ensure_evaluated,
    )

    root_hash = "root123"
    ingestion.initialise_root_commit(root_hash)

    assert calls["available"] == 1
    assert calls["metadata"] == 1
    assert calls["evaluated"] == 1
    # Root initialisation should not attempt to ingest the root commit into any
    # MAP-Elites archive.
    assert manager.ingest_calls == []

