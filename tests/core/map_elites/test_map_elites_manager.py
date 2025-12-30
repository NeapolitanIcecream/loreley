from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pytest

import loreley.core.map_elites.map_elites as map_elites_module
from loreley.config import Settings
from loreley.core.map_elites.code_embedding import CommitCodeEmbedding
from loreley.core.map_elites.dimension_reduction import FinalEmbedding, PenultimateEmbedding
from loreley.core.map_elites.map_elites import MapElitesManager, MapElitesRecord
from loreley.core.map_elites.repository_state_embedding import RepoStateEmbeddingStats


def test_manager_lazy_loads_persisted_snapshot_for_stats_and_records(settings: Settings) -> None:
    settings.mapelites_dimensionality_target_dims = 2
    settings.mapelites_archive_cells_per_dim = 4

    snapshot = {
        "island_id": "main",
        "lower_bounds": [0.0, 0.0],
        "upper_bounds": [1.0, 1.0],
        "history": [],
        "projection": None,
        "archive": [
            {
                "index": 0,
                "objective": 1.23,
                "measures": [0.1, 0.1],
                "solution": [0.1, 0.1],
                "commit_hash": "c1",
                "timestamp": 42.0,
            }
        ],
    }

    class DummySnapshotBackend:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def load(self, island_id: str) -> dict[str, object] | None:
            if island_id != "main":
                return None
            return dict(self._payload)

        def save(self, island_id: str, payload: object) -> None:
            return None

    manager = MapElitesManager(settings=settings, repo_root=Path("."), experiment_id="00000000-0000-0000-0000-000000000000")
    manager._snapshot_backend = DummySnapshotBackend(snapshot)  # type: ignore[attr-defined]

    stats = manager.describe_island("main")
    assert stats["cells"] == 16
    assert stats["occupied"] == 1
    assert stats["best_fitness"] == pytest.approx(1.23)

    records = manager.get_records("main")
    assert len(records) == 1
    assert records[0].commit_hash == "c1"


def test_manager_adopts_snapshot_dimensionality_when_settings_mismatch(settings: Settings) -> None:
    settings.mapelites_dimensionality_target_dims = 4
    settings.mapelites_archive_cells_per_dim = 4

    snapshot = {
        "island_id": "main",
        "lower_bounds": [0.0, 0.0],
        "upper_bounds": [1.0, 1.0],
        "history": [],
        "projection": None,
        "archive": [
            {
                "index": 0,
                "objective": 1.23,
                "measures": [0.1, 0.1],
                "solution": [0.1, 0.1],
                "commit_hash": "c1",
                "timestamp": 42.0,
            }
        ],
    }

    class DummySnapshotBackend:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def load(self, island_id: str) -> dict[str, object] | None:
            if island_id != "main":
                return None
            return dict(self._payload)

        def save(self, island_id: str, payload: object) -> None:
            return None

    manager = MapElitesManager(
        settings=settings,
        repo_root=Path("."),
        experiment_id="00000000-0000-0000-0000-000000000000",
    )
    manager._snapshot_backend = DummySnapshotBackend(snapshot)  # type: ignore[attr-defined]

    stats = manager.describe_island("main")
    assert stats["cells"] == 16
    assert stats["occupied"] == 1
    assert manager._target_dims == 2  # Snapshot dimensionality is authoritative for persisted experiments.

    records = manager.get_records("main")
    assert len(records) == 1
    assert len(records[0].measures) == 2


def test_ingest_short_circuits_when_no_repo_state_embedding(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    stats = RepoStateEmbeddingStats(
        treeish="abc",
        eligible_files=0,
        files_embedded=0,
        files_aggregated=0,
        unique_blobs=0,
        cache_hits=0,
        cache_misses=0,
        skipped_empty_after_preprocess=0,
        skipped_failed_embedding=0,
    )
    monkeypatch.setattr(
        map_elites_module,
        "embed_repository_state",
        lambda *args, **kwargs: (None, stats),
    )

    manager = MapElitesManager(settings=settings, repo_root=Path("."))
    result = manager.ingest(
        commit_hash="abc",
        changed_files=[{"path": "a.py", "change_count": 1}],
    )

    assert result.status == 0
    assert result.record is None
    assert result.artifacts.preprocessed_files == ()
    assert "No eligible repository files" in (result.message or "")


def test_ingest_builds_record_with_stubbed_dependencies(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    settings.mapelites_dimensionality_target_dims = 2
    settings.mapelites_feature_clip = True
    settings.mapelites_feature_truncation_k = 1.0
    settings.mapelites_fitness_metric = "score"

    code_embedding = CommitCodeEmbedding(
        files=(),
        vector=(0.5, -0.5),
        model="code",
        dimensions=2,
    )
    stats = RepoStateEmbeddingStats(
        treeish="abc",
        eligible_files=2,
        files_embedded=1,
        files_aggregated=2,
        unique_blobs=2,
        cache_hits=1,
        cache_misses=1,
        skipped_empty_after_preprocess=0,
        skipped_failed_embedding=0,
    )
    penultimate = PenultimateEmbedding(
        commit_hash="abc",
        vector=(0.5, -0.5),
        code_dimensions=2,
        summary_dimensions=0,
        code_model="code",
        summary_model=None,
        summary_embedding_model=None,
    )
    final_embedding = FinalEmbedding(
        commit_hash="abc",
        vector=(0.2, 0.8),
        dimensions=2,
        penultimate=penultimate,
        projection=None,
    )

    monkeypatch.setattr(
        map_elites_module,
        "embed_repository_state",
        lambda *args, **kwargs: (code_embedding, stats),
    )
    monkeypatch.setattr(
        map_elites_module,
        "reduce_commit_embeddings",
        lambda **kwargs: (final_embedding, (penultimate,), None),
    )

    manager = MapElitesManager(settings=settings, repo_root=Path("."))
    monkeypatch.setattr(manager, "_persist_island_state", lambda *args, **kwargs: None)

    captured: dict[str, object] = {}

    def _fake_add_to_archive(
        *,
        state: object,
        island_id: str,
        commit_hash: str,
        fitness: float,
        measures: np.ndarray,
    ) -> tuple[int, float, MapElitesRecord]:
        captured["measures"] = measures
        captured["fitness"] = fitness
        record = MapElitesRecord(
            commit_hash=commit_hash,
            island_id=island_id,
            cell_index=0,
            fitness=fitness,
            measures=tuple(measures.tolist()),
            solution=tuple(measures.tolist()),
            timestamp=123.0,
        )
        return 1, 0.1, record

    monkeypatch.setattr(manager, "_add_to_archive", _fake_add_to_archive)

    result = manager.ingest(
        commit_hash="abc",
        changed_files=[{"path": "a.py", "change_count": 2}],
        metrics={"score": 1.2},
    )

    assert result.inserted
    assert captured["fitness"] == 1.2
    assert captured["measures"] is not None
    assert tuple(captured["measures"].tolist()) == pytest.approx((0.6, 0.9))  # type: ignore[index]
    assert result.record is not None
    assert result.record.commit_hash == "abc"
    assert result.artifacts.code_embedding is code_embedding
    assert result.artifacts.summary_embedding is None
    assert result.artifacts.final_embedding is final_embedding
