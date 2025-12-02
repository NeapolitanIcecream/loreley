from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from loreley.config import Settings
from loreley.core.map_elites.map_elites import (
    CommitEmbeddingArtifacts,
    MapElitesInsertionResult,
    MapElitesRecord,
)
from loreley.core.map_elites.preprocess import ChangedFile
from loreley.scheduler import ingestion as ingestion_mod
from loreley.scheduler.ingestion import MapElitesIngestion, ROOT_PLACEHOLDER_FILENAME


class DummyManager:
    """Lightweight stub that records ingest calls for assertions."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get_records(self, island_id: str | None = None) -> tuple[Any, ...]:
        # Root archives start empty for this test.
        return ()

    def ingest(
        self,
        *,
        commit_hash: str,
        changed_files: list[ChangedFile],
        metrics: Mapping[str, Any] | None = None,
        island_id: str | None = None,
        repo_root: Path | None = None,
        treeish: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MapElitesInsertionResult:
        record = MapElitesRecord(
            commit_hash=commit_hash,
            island_id=island_id or "main",
            cell_index=0,
            fitness=1.0,
            measures=(0.0,),
            solution=(0.0,),
            metadata=metadata or {},
            timestamp=0.0,
        )
        artifacts = CommitEmbeddingArtifacts(
            preprocessed_files=(),
            chunked_files=(),
            code_embedding=None,
            summary_embedding=None,
            final_embedding=None,
        )
        call = {
            "commit_hash": commit_hash,
            "changed_files": changed_files,
            "metrics": metrics,
            "island_id": island_id,
            "repo_root": repo_root,
            "treeish": treeish,
            "metadata": dict(metadata or {}),
        }
        self.calls.append(call)
        return MapElitesInsertionResult(
            status=1,
            delta=1.0,
            record=record,
            artifacts=artifacts,
            message=None,
        )


@contextmanager
def _dummy_session_scope() -> Any:
    """Stub for session_scope that avoids real database access."""

    class DummyResult:
        def all(self) -> list[Any]:
            return []

    class DummySession:
        def scalars(self, _stmt: Any) -> DummyResult:
            return DummyResult()

    yield DummySession()


def test_root_placeholder_ingestion_uses_synthetic_changed_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = Settings()
    default_island = settings.mapelites_default_island_id or "main"

    # Avoid touching the real database.
    monkeypatch.setattr(ingestion_mod, "session_scope", _dummy_session_scope)

    manager = DummyManager()
    experiment = SimpleNamespace(id="exp-123")
    repository = SimpleNamespace(id="repo-456")

    ingestion = MapElitesIngestion(
        settings=settings,
        console=ingestion_mod.Console(),
        repo_root=tmp_path,
        repo=object(),  # Not used by _ensure_root_commit_ingested.
        manager=manager,  # type: ignore[arg-type]
        experiment=experiment,
        repository=repository,
    )

    root_hash = "root123"
    ingestion._ensure_root_commit_ingested(root_hash)  # type: ignore[attr-defined]

    assert len(manager.calls) == 1
    call = manager.calls[0]

    changed_files = call["changed_files"]
    assert isinstance(changed_files, list)
    assert len(changed_files) == 1

    placeholder = changed_files[0]
    assert isinstance(placeholder, ChangedFile)
    assert placeholder.path.name == ROOT_PLACEHOLDER_FILENAME.name
    assert placeholder.content is not None
    assert root_hash in placeholder.content
    assert default_island in placeholder.content or "island_id" in placeholder.content

    metadata = call["metadata"]
    assert metadata.get("root_commit") is True
    assert metadata.get("root_placeholder") is True
    assert metadata.get("root_placeholder_file") == str(ROOT_PLACEHOLDER_FILENAME)
    assert metadata.get("island_id") == default_island


