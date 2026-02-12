from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pytest

import loreley.core.map_elites.map_elites as map_elites_module
from loreley.config import Settings
from loreley.core.map_elites.code_embedding import CommitCodeEmbedding
from loreley.core.map_elites.dimension_reduction import FinalEmbedding, PCAProjection, PcaHistoryEntry
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

        def load(self, island_id: str, *, history_limit: int | None = None) -> dict[str, object] | None:
            if island_id != "main":
                return None
            return dict(self._payload)

    manager = MapElitesManager(settings=settings, repo_root=Path("."))
    manager._snapshot_store = DummySnapshotBackend(snapshot)  # type: ignore[attr-defined]

    stats = manager.describe_island("main")
    assert stats["cells"] == 16
    assert stats["occupied"] == 1
    assert stats["best_fitness"] == pytest.approx(1.23)

    records = manager.get_records("main")
    assert len(records) == 1
    assert records[0].commit_hash == "c1"
    assert manager.get_cell_commits("main") == {0: "c1"}


def test_get_cell_commits_fails_when_bookkeeping_is_incomplete(settings: Settings) -> None:
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
            },
            {
                "index": 1,
                "objective": 0.5,
                "measures": [0.1, 0.3],
                "solution": [0.1, 0.3],
                "commit_hash": "",
                "timestamp": 43.0,
            },
        ],
    }

    class DummySnapshotBackend:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def load(self, island_id: str, *, history_limit: int | None = None) -> dict[str, object] | None:
            if island_id != "main":
                return None
            return dict(self._payload)

    manager = MapElitesManager(settings=settings, repo_root=Path("."))
    manager._snapshot_store = DummySnapshotBackend(snapshot)  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="bookkeeping mismatch"):
        _ = manager.get_cell_commits("main")


def test_manager_rejects_snapshot_dimensionality_when_settings_mismatch(settings: Settings) -> None:
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

        def load(self, island_id: str, *, history_limit: int | None = None) -> dict[str, object] | None:
            if island_id != "main":
                return None
            return dict(self._payload)

    manager = MapElitesManager(
        settings=settings,
        repo_root=Path("."),
    )
    manager._snapshot_store = DummySnapshotBackend(snapshot)  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="Snapshot dimensionality mismatch"):
        _ = manager.describe_island("main")


def test_ingest_short_circuits_when_no_repo_state_embedding(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    stats = RepoStateEmbeddingStats(
        commit_hash="abc",
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
        "embed_repository_state_incremental",
        lambda *args, **kwargs: (None, stats),
    )

    manager = MapElitesManager(
        settings=settings,
        repo_root=Path("."),
    )
    class NullSnapshotStore:
        def load(self, island_id: str, *, history_limit: int | None = None) -> dict[str, object] | None:
            return None

        def apply_update(self, island_id: str, *, update: object, session: object | None = None) -> None:
            return None

    manager._snapshot_store = NullSnapshotStore()  # type: ignore[attr-defined]
    result = manager.ingest(
        commit_hash="abc",
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
        commit_hash="abc",
        eligible_files=2,
        files_embedded=1,
        files_aggregated=2,
        unique_blobs=2,
        cache_hits=1,
        cache_misses=1,
        skipped_empty_after_preprocess=0,
        skipped_failed_embedding=0,
    )
    entry = PcaHistoryEntry(
        commit_hash="abc",
        vector=(0.5, -0.5),
        embedding_model="code",
    )
    final_embedding = FinalEmbedding(
        commit_hash="abc",
        vector=(0.2, 0.8),
        dimensions=2,
        history_entry=entry,
        projection=None,
    )

    monkeypatch.setattr(
        map_elites_module,
        "embed_repository_state_incremental",
        lambda *args, **kwargs: (code_embedding, stats),
    )
    monkeypatch.setattr(
        map_elites_module,
        "reduce_commit_embeddings",
        lambda **kwargs: (final_embedding, (entry,), None, 0),
    )

    manager = MapElitesManager(
        settings=settings,
        repo_root=Path("."),
    )
    class NullSnapshotStore:
        def load(self, island_id: str, *, history_limit: int | None = None) -> dict[str, object] | None:
            return None

        def apply_update(self, island_id: str, *, update: object, session: object | None = None) -> None:
            return None

    manager._snapshot_store = NullSnapshotStore()  # type: ignore[attr-defined]
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
        metrics={"score": 1.2},
    )

    assert result.inserted
    assert captured["fitness"] == 1.2
    assert captured["measures"] is not None
    assert tuple(captured["measures"].tolist()) == pytest.approx((0.6, 0.9))  # type: ignore[index]
    assert result.record is not None
    assert result.record.commit_hash == "abc"
    assert result.artifacts.code_embedding is code_embedding
    assert result.artifacts.final_embedding is final_embedding


def test_ingest_passes_external_snapshot_session(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    settings.mapelites_dimensionality_target_dims = 2
    settings.mapelites_fitness_metric = "score"
    code_embedding = CommitCodeEmbedding(
        files=(),
        vector=(0.5, -0.5),
        model="code",
        dimensions=2,
    )
    stats = RepoStateEmbeddingStats(
        commit_hash="abc",
        eligible_files=2,
        files_embedded=1,
        files_aggregated=2,
        unique_blobs=2,
        cache_hits=1,
        cache_misses=1,
        skipped_empty_after_preprocess=0,
        skipped_failed_embedding=0,
    )
    entry = PcaHistoryEntry(
        commit_hash="abc",
        vector=(0.5, -0.5),
        embedding_model="code",
    )
    final_embedding = FinalEmbedding(
        commit_hash="abc",
        vector=(0.2, 0.8),
        dimensions=2,
        history_entry=entry,
        projection=None,
    )

    monkeypatch.setattr(
        map_elites_module,
        "embed_repository_state_incremental",
        lambda *args, **kwargs: (code_embedding, stats),
    )
    monkeypatch.setattr(
        map_elites_module,
        "reduce_commit_embeddings",
        lambda **kwargs: (final_embedding, (entry,), None, 0),
    )

    class SnapshotStoreRecorder:
        def __init__(self) -> None:
            self.last_session: object | None = None

        def load(self, island_id: str, *, history_limit: int | None = None) -> dict[str, object] | None:
            return None

        def apply_update(self, island_id: str, *, update: object, session: object | None = None) -> None:
            self.last_session = session

    manager = MapElitesManager(
        settings=settings,
        repo_root=Path("."),
    )
    recorder = SnapshotStoreRecorder()
    manager._snapshot_store = recorder  # type: ignore[attr-defined]

    monkeypatch.setattr(
        manager,
        "_add_to_archive",
        lambda **kwargs: (
            1,
            0.1,
            MapElitesRecord(
                commit_hash="abc",
                island_id="default",
                cell_index=0,
                fitness=1.0,
                measures=(0.2, 0.8),
                solution=(0.2, 0.8),
                timestamp=123.0,
            ),
        ),
    )

    session_marker = object()
    _ = manager.ingest(
        commit_hash="abc",
        metrics={"score": 1.0},
        snapshot_session=session_marker,  # type: ignore[arg-type]
    )

    assert recorder.last_session is session_marker


def test_ingest_rebuilds_archive_when_pca_projection_refits(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """Regression: PCA refits must rebuild existing archive cells."""

    settings.mapelites_default_island_id = "main"
    settings.mapelites_dimensionality_target_dims = 2
    settings.mapelites_archive_cells_per_dim = 4
    settings.mapelites_feature_clip = True
    settings.mapelites_feature_truncation_k = 1.0
    settings.mapelites_fitness_metric = "score"

    # Two fake projections over 3D vectors. The refit changes the 2nd component
    # from dimension-2 to dimension-3, so existing commits must be reprojected.
    old_projection = PCAProjection(
        feature_count=3,
        components=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        mean=(0.0, 0.0, 0.0),
        explained_variance=(1.0, 1.0),
        explained_variance_ratio=(0.5, 0.5),
        sample_count=2,
        epoch=0,
        fitted_at=10.0,
        whiten=False,
        rotation=None,
    )
    new_projection = PCAProjection(
        feature_count=3,
        components=((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        mean=(0.0, 0.0, 0.0),
        explained_variance=(1.0, 1.0),
        explained_variance_ratio=(0.5, 0.5),
        sample_count=3,
        epoch=1,
        fitted_at=20.0,
        whiten=False,
        rotation=None,
    )

    entry_c1 = PcaHistoryEntry(
        commit_hash="c1",
        vector=(0.0, 1.0, 0.0),
        embedding_model="code",
    )
    entry_c2 = PcaHistoryEntry(
        commit_hash="c2",
        vector=(1.0, 0.0, 1.0),
        embedding_model="code",
    )

    final_c1 = FinalEmbedding(
        commit_hash="c1",
        vector=(0.0, 1.0),
        dimensions=2,
        history_entry=entry_c1,
        projection=old_projection,
    )
    final_c2 = FinalEmbedding(
        commit_hash="c2",
        vector=(0.0, 0.0),
        dimensions=2,
        history_entry=entry_c2,
        projection=new_projection,
    )

    code_embedding = CommitCodeEmbedding(
        files=(),
        vector=(0.0, 0.0, 0.0),
        model="code",
        dimensions=3,
    )
    stats = RepoStateEmbeddingStats(
        commit_hash="c1",
        eligible_files=1,
        files_embedded=1,
        files_aggregated=1,
        unique_blobs=1,
        cache_hits=0,
        cache_misses=1,
        skipped_empty_after_preprocess=0,
        skipped_failed_embedding=0,
    )
    monkeypatch.setattr(
        map_elites_module,
        "embed_repository_state_incremental",
        lambda *args, **kwargs: (code_embedding, stats),
    )

    def _fake_reduce_commit_embeddings(**kwargs: object):
        commit = kwargs.get("commit_hash")
        if commit == "c1":
            return final_c1, (entry_c1,), old_projection, 0
        if commit == "c2":
            # Return both vectors so the rebuild can load c1 without a DB query.
            return final_c2, (entry_c1, entry_c2), new_projection, 0
        raise AssertionError(f"Unexpected commit {commit!r}")

    monkeypatch.setattr(map_elites_module, "reduce_commit_embeddings", _fake_reduce_commit_embeddings)

    manager = MapElitesManager(settings=settings, repo_root=Path("."))

    class NullSnapshotStore:
        def load(self, island_id: str, *, history_limit: int | None = None) -> dict[str, object] | None:
            return None

        def apply_update(self, island_id: str, *, update: object, session: object | None = None) -> None:
            return None

    manager._snapshot_store = NullSnapshotStore()  # type: ignore[attr-defined]
    monkeypatch.setattr(manager, "_persist_island_state", lambda *args, **kwargs: None)

    _ = manager.ingest(commit_hash="c1", metrics={"score": 1.0})
    before = {rec.commit_hash: rec.measures for rec in manager.get_records("main")}
    assert before["c1"] == pytest.approx((0.5, 1.0))

    _ = manager.ingest(commit_hash="c2", metrics={"score": 0.5})
    after = {rec.commit_hash: rec.measures for rec in manager.get_records("main")}

    # Under the refit projection, c1 loses its second component (it becomes 0),
    # so the y measure must move from 1.0 to 0.5 after rebuild.
    assert after["c1"] == pytest.approx((0.5, 0.5))
