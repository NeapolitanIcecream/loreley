from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
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
    projection = PCAProjection(
        feature_count=2,
        components=((1.0, 0.0), (0.0, 1.0)),
        mean=(0.0, 0.0),
        explained_variance=(1.0, 1.0),
        explained_variance_ratio=(0.5, 0.5),
        sample_count=1,
        epoch=0,
        fitted_at=0.0,
        whiten=False,
        rotation=None,
    )
    final_embedding = FinalEmbedding(
        commit_hash="abc",
        vector=(0.2, 0.8),
        dimensions=2,
        history_entry=entry,
        projection=projection,
    )

    monkeypatch.setattr(
        map_elites_module,
        "embed_repository_state_incremental",
        lambda *args, **kwargs: (code_embedding, stats),
    )
    monkeypatch.setattr(
        map_elites_module,
        "reduce_commit_embeddings",
        lambda **kwargs: (final_embedding, (entry,), projection, 0),
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


def test_ingest_seeds_archive_after_initial_pca_fit(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """Regression: warmup commits must populate the archive once PCA is fitted."""

    settings.mapelites_default_island_id = "main"
    settings.mapelites_dimensionality_target_dims = 2
    settings.mapelites_archive_cells_per_dim = 4
    settings.mapelites_feature_clip = True
    settings.mapelites_feature_truncation_k = 1.0
    settings.mapelites_fitness_metric = "score"

    projection = PCAProjection(
        feature_count=3,
        components=((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        mean=(0.0, 0.0, 0.0),
        explained_variance=(1.0, 1.0),
        explained_variance_ratio=(0.5, 0.5),
        sample_count=2,
        epoch=0,
        fitted_at=10.0,
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
        vector=(0.9, 0.1),  # Warmup fallback that must not be stored in the archive.
        dimensions=2,
        history_entry=entry_c1,
        projection=None,
    )
    final_c2 = FinalEmbedding(
        commit_hash="c2",
        vector=(1.0, 1.0),
        dimensions=2,
        history_entry=entry_c2,
        projection=projection,
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
            return final_c1, (entry_c1,), None, 0
        if commit == "c2":
            return final_c2, (entry_c1, entry_c2), projection, 0
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
    monkeypatch.setattr(
        manager,
        "_load_commit_fitnesses",
        lambda *, commit_hashes, snapshot_session: {"c1": 1.0, "c2": 0.5},
    )

    first = manager.ingest(commit_hash="c1", metrics={"score": 1.0})
    assert first.inserted is False
    assert manager.get_records("main") == ()

    _ = manager.ingest(commit_hash="c2", metrics={"score": 0.5})
    records = {rec.commit_hash: rec.measures for rec in manager.get_records("main")}

    assert records["c1"] == pytest.approx((0.5, 0.5))
    assert records["c2"] == pytest.approx((1.0, 1.0))


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


def test_add_batch_to_archive_uses_status_value_to_update_mappings(settings: Settings) -> None:
    settings.mapelites_dimensionality_target_dims = 2
    manager = MapElitesManager(settings=settings, repo_root=Path("."))

    class DummyArchive:
        def __init__(self) -> None:
            self.add_calls: list[dict[str, np.ndarray]] = []

        def index_of(self, measures: np.ndarray) -> np.ndarray:
            batch = np.asarray(measures, dtype=np.float64)
            assert batch.shape[0] == 3
            return np.asarray([0, 0, 1], dtype=np.int64)

        def add(
            self,
            solution: np.ndarray,
            objective: np.ndarray,
            measures: np.ndarray,
            *,
            commit_hash: np.ndarray,
            timestamp: np.ndarray,
        ) -> Mapping[str, np.ndarray]:
            self.add_calls.append(
                {
                    "solution": np.asarray(solution),
                    "objective": np.asarray(objective),
                    "measures": np.asarray(measures),
                    "commit_hash": np.asarray(commit_hash),
                    "timestamp": np.asarray(timestamp),
                }
            )
            # Same cell for c1/c2 with equal status: value decides winner (c2).
            return {
                "status": np.asarray([1, 1, 0], dtype=np.int64),
                "value": np.asarray([0.1, 0.4, -0.2], dtype=np.float64),
            }

    state = map_elites_module.IslandState(
        archive=DummyArchive(),  # type: ignore[arg-type]
        lower_bounds=np.asarray([0.0, 0.0], dtype=np.float64),
        upper_bounds=np.asarray([1.0, 1.0], dtype=np.float64),
    )
    state.index_to_commit[0] = "old"
    state.commit_to_index["old"] = 0
    manager._commit_to_island["old"] = "main"

    statuses, values = manager._add_batch_to_archive(
        state=state,
        island_id="main",
        commit_hashes=["c1", "c2", "c3"],
        objectives=[1.0, 2.0, 3.0],
        measures=[
            np.asarray([0.1, 0.1], dtype=np.float64),
            np.asarray([0.1, 0.1], dtype=np.float64),
            np.asarray([0.9, 0.9], dtype=np.float64),
        ],
        timestamps=[10.0, 11.0, 12.0],
    )

    assert statuses.tolist() == [1, 1, 0]
    assert values.tolist() == pytest.approx([0.1, 0.4, -0.2])
    assert state.index_to_commit == {0: "c2"}
    assert state.commit_to_index == {"c2": 0}
    assert manager._commit_to_island == {"c2": "main"}


def test_load_commit_vectors_batches_long_in_queries(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: long vector lookups must batch IN clauses for both lookup stages."""

    settings.mapelites_dimensionality_penultimate_normalize = False
    manager = MapElitesManager(settings=settings, repo_root=Path("."))
    monkeypatch.setattr(map_elites_module, "_IN_QUERY_BATCH_SIZE", 2)

    state = map_elites_module.IslandState(
        archive=object(),  # type: ignore[arg-type]
        lower_bounds=np.asarray([0.0], dtype=np.float64),
        upper_bounds=np.asarray([1.0], dtype=np.float64),
        history=(),
    )
    commits = [f"c{i}" for i in range(5)]

    class _ScalarResult:
        def __init__(self, rows: list[SimpleNamespace]) -> None:
            self._rows = rows

        def all(self) -> list[SimpleNamespace]:
            return list(self._rows)

    class _ExecResult:
        def __init__(self, rows: list[SimpleNamespace]) -> None:
            self._rows = rows

        def scalars(self) -> _ScalarResult:
            return _ScalarResult(self._rows)

    class _FakeSession:
        def __init__(self) -> None:
            self.pca_batches: list[int] = []
            self.aggregate_batches: list[int] = []

        def execute(self, stmt: object) -> _ExecResult:
            compiled = stmt.compile()  # type: ignore[attr-defined]
            batch: list[str] | None = None
            for value in compiled.params.values():
                if isinstance(value, list):
                    batch = [str(v) for v in value]
                    break
            assert batch is not None
            sql = str(stmt)
            if "map_elites_pca_history" in sql:
                self.pca_batches.append(len(batch))
                return _ExecResult([])
            if "map_elites_repo_state_aggregate" in sql:
                self.aggregate_batches.append(len(batch))
                rows = [
                    SimpleNamespace(commit_hash=commit, file_count=2, sum_vector=[2.0, 4.0])
                    for commit in batch
                ]
                return _ExecResult(rows)
            raise AssertionError(f"Unexpected statement: {sql}")

    fake_session = _FakeSession()
    vectors = manager._load_commit_vectors(
        island_id="main",
        commit_hashes=commits,
        state=state,
        snapshot_session=fake_session,  # type: ignore[arg-type]
    )

    assert fake_session.pca_batches == [2, 2, 1]
    assert fake_session.aggregate_batches == [2, 2, 1]
    assert set(vectors.keys()) == set(commits)
    for vector in vectors.values():
        assert vector == pytest.approx((1.0, 2.0))
