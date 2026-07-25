from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import loreley.core.map_elites.db_ops as map_elites_db_ops
import loreley.core.map_elites.manager as map_elites_module
from loreley.config import Settings, resolve_objective_contract
from loreley.core.map_elites.code_embedding import CommitCodeEmbedding
from loreley.core.map_elites.dimension_reduction import FinalEmbedding, PCAProjection, PcaHistoryEntry
from loreley.core.map_elites.manager import MapElitesManager
from loreley.core.map_elites.types import MapElitesRecord
from loreley.core.map_elites.objectives import ObjectiveSpec, ResolvedObjectives
from loreley.core.map_elites.repository_state_embedding import RepoStateEmbeddingStats, RepositoryStateEmbedder


class _RecordingSnapshotStore:
    def __init__(self) -> None:
        self.updates: list[tuple[str, object, object | None]] = []

    def load(self, island_id: str, *, history_limit: int | None = None) -> dict[str, object] | None:
        return None

    def apply_update(self, island_id: str, *, update: object, session: object | None = None) -> None:
        self.updates.append((island_id, update, session))


def _repo_state_stats(
    *,
    commit_hash: str = "abc",
    cache_misses: int = 1,
    source: str = "unknown",
) -> RepoStateEmbeddingStats:
    return RepoStateEmbeddingStats(
        commit_hash=commit_hash,
        eligible_files=1,
        files_embedded=1,
        files_aggregated=1,
        unique_blobs=1,
        cache_hits=0,
        cache_misses=cache_misses,
        skipped_empty_after_preprocess=0,
        skipped_failed_embedding=0,
        source=source,
    )


def _code_embedding(vector: tuple[float, ...] = (0.5, -0.5)) -> CommitCodeEmbedding:
    return CommitCodeEmbedding(
        files=(),
        vector=vector,
        model="code",
        dimensions=len(vector),
    )


def _history_entry(
    *,
    commit_hash: str = "abc",
    vector: tuple[float, ...] = (0.5, -0.5),
) -> PcaHistoryEntry:
    return PcaHistoryEntry(
        commit_hash=commit_hash,
        vector=vector,
        embedding_model="code",
    )


def _identity_projection(*, dims: int = 2, epoch: int = 0) -> PCAProjection:
    components = tuple(
        tuple(1.0 if row == column else 0.0 for column in range(dims))
        for row in range(dims)
    )
    return PCAProjection(
        feature_count=dims,
        components=components,
        mean=tuple(0.0 for _ in range(dims)),
        explained_variance=tuple(1.0 for _ in range(dims)),
        explained_variance_ratio=tuple(1.0 / dims for _ in range(dims)),
        sample_count=1,
        epoch=epoch,
        fitted_at=0.0,
        whiten=False,
        rotation=None,
    )


def _configure_score_objective(settings: Settings) -> None:
    settings.mapelites_islands = ("main",)
    settings.mapelites_objectives = (
        ObjectiveSpec(name="score", direction="max"),
    )


def _score_metrics(value: float) -> tuple[dict[str, object], ...]:
    return (
        {
            "name": "score",
            "value": value,
            "higher_is_better": True,
        },
    )


def _with_contract(
    settings: Settings,
    snapshot: dict[str, object],
) -> dict[str, object]:
    contract = resolve_objective_contract(settings)
    return {
        **snapshot,
        "objective_contract": contract.as_payload(),
        "objective_contract_fingerprint": contract.fingerprint,
    }


def test_manager_lazy_loads_persisted_snapshot_for_stats_and_records(settings: Settings) -> None:
    _configure_score_objective(settings)
    settings.mapelites_dimensionality_target_dims = 2
    settings.mapelites_archive_cells_per_dim = 4

    snapshot = _with_contract(settings, {
        "island_id": "main",
        "lower_bounds": [0.0, 0.0],
        "upper_bounds": [1.0, 1.0],
        "history": [],
        "projection": None,
        "archive": [
            {
                "index": 0,
                "objective_values": [1.23],
                "measures": [0.1, 0.1],
                "commit_hash": "c1",
                "timestamp": 42.0,
            }
        ],
    })

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
    assert stats["elites"] == 1
    assert stats["coverage"] == pytest.approx(1 / 16)
    assert stats["objective_count"] == 1
    assert stats["front_max_size"] == settings.mapelites_pareto_front_max_size
    assert stats["primary_metric_name"] == "score"
    assert stats["primary_metric_direction"] == "max"
    assert stats["best_primary_value"] == pytest.approx(1.23)

    records = manager.get_records("main")
    assert len(records) == 1
    assert records[0].commit_hash == "c1"
    assert records[0].objective_values == pytest.approx((1.23,))
    assert manager.get_cell_fronts("main") == {0: ("c1",)}


def test_manager_validates_every_configured_island_eagerly(settings: Settings) -> None:
    settings.mapelites_islands = ("alpha", "beta")
    manager = MapElitesManager(settings=settings, repo_root=Path("."))
    loaded: list[str] = []

    class DummySnapshotBackend:
        def load(
            self,
            island_id: str,
            *,
            history_limit: int | None = None,
        ) -> None:
            loaded.append(island_id)
            return None

    manager._snapshot_store = DummySnapshotBackend()  # type: ignore[assignment]

    manager.validate_configured_islands()

    assert loaded == ["alpha", "beta"]
    assert set(manager._archives) == {"alpha", "beta"}


def test_count_pca_history_samples_counts_non_empty_snapshot_entries(settings: Settings) -> None:
    _configure_score_objective(settings)
    settings.mapelites_dimensionality_target_dims = 2
    settings.mapelites_archive_cells_per_dim = 4

    snapshot = _with_contract(settings, {
        "island_id": "main",
        "lower_bounds": [0.0, 0.0],
        "upper_bounds": [1.0, 1.0],
        "history": [
            {
                "commit_hash": "c1",
                "vector": [0.1, 0.2],
                "embedding_model": "code",
            },
            {
                "commit_hash": "empty",
                "vector": [],
                "embedding_model": "code",
            },
            {
                "commit_hash": "c2",
                "vector": [0.3, 0.4],
                "embedding_model": "code",
            },
        ],
        "projection": None,
        "archive": [],
    })

    class DummySnapshotBackend:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def load(self, island_id: str, *, history_limit: int | None = None) -> dict[str, object] | None:
            if island_id != "main":
                return None
            return dict(self._payload)

    manager = MapElitesManager(settings=settings, repo_root=Path("."))
    manager._snapshot_store = DummySnapshotBackend(snapshot)  # type: ignore[attr-defined]

    assert manager.count_pca_history_samples("main") == 2


def test_get_cell_fronts_fails_when_bookkeeping_is_incomplete(settings: Settings) -> None:
    _configure_score_objective(settings)
    settings.mapelites_dimensionality_target_dims = 2
    settings.mapelites_archive_cells_per_dim = 4

    snapshot = _with_contract(settings, {
        "island_id": "main",
        "lower_bounds": [0.0, 0.0],
        "upper_bounds": [1.0, 1.0],
        "history": [],
        "projection": None,
        "archive": [
            {
                "index": 0,
                "objective_values": [1.23],
                "measures": [0.1, 0.1],
                "commit_hash": "c1",
                "timestamp": 42.0,
            },
        ],
    })

    class DummySnapshotBackend:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def load(self, island_id: str, *, history_limit: int | None = None) -> dict[str, object] | None:
            if island_id != "main":
                return None
            return dict(self._payload)

    manager = MapElitesManager(settings=settings, repo_root=Path("."))
    manager._snapshot_store = DummySnapshotBackend(snapshot)  # type: ignore[attr-defined]

    state = manager._ensure_island("main")  # noqa: SLF001
    state.index_to_commits.clear()
    with pytest.raises(RuntimeError, match="bookkeeping mismatch"):
        _ = manager.get_cell_fronts("main")


def test_manager_rejects_snapshot_dimensionality_when_settings_mismatch(settings: Settings) -> None:
    _configure_score_objective(settings)
    settings.mapelites_dimensionality_target_dims = 4
    settings.mapelites_archive_cells_per_dim = 4

    snapshot = _with_contract(settings, {
        "island_id": "main",
        "lower_bounds": [0.0, 0.0],
        "upper_bounds": [1.0, 1.0],
        "history": [],
        "projection": None,
        "archive": [
            {
                "index": 0,
                "objective_values": [1.23],
                "measures": [0.1, 0.1],
                "commit_hash": "c1",
                "timestamp": 42.0,
            }
        ],
    })

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
    monkeypatch: pytest.MonkeyPatch, settings: Settings, captured_logs: list[dict[str, object]]
) -> None:
    settings.mapelites_ingest_info_log_every = 1
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
    stage_logs = [
        record
        for record in captured_logs
        if record.get("module") == "map_elites.manager"
        and record.get("message") == "MAP-Elites ingest stage metrics"
    ]
    assert stage_logs
    stage_extra = stage_logs[-1].get("extra")
    assert isinstance(stage_extra, dict)
    assert stage_extra.get("aggregate_hit_count") == 0
    assert stage_extra.get("incremental_count") == 0
    assert stage_extra.get("embedding_cache_miss_count") == 0
    assert stage_extra.get("status_code") == 0


def test_ingest_info_logs_are_sampled(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    captured_logs: list[dict[str, object]],
) -> None:
    settings.mapelites_ingest_info_log_every = 3
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

    for idx in range(4):
        _ = manager.ingest(commit_hash=f"c{idx}")

    ingest_logs = [
        record
        for record in captured_logs
        if record.get("module") == "map_elites.manager"
        and str(record.get("message", "")).startswith("Ingesting commit ")
    ]
    assert len(ingest_logs) == 2
    assert "c0" in str(ingest_logs[0].get("message"))
    assert "c3" in str(ingest_logs[1].get("message"))

    stage_logs = [
        record
        for record in captured_logs
        if record.get("module") == "map_elites.manager"
        and record.get("message") == "MAP-Elites ingest stage metrics"
    ]
    assert len(stage_logs) == 2


def test_manager_reuses_repo_state_embedder_across_ingests(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    settings.mapelites_ingest_info_log_every = 10_000
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
    seen_embedders: list[object] = []

    def _fake_embed_repository_state_incremental(*args, **kwargs):  # type: ignore[no-untyped-def]
        seen_embedders.append(kwargs.get("embedder"))
        return None, stats

    monkeypatch.setattr(
        map_elites_module,
        "embed_repository_state_incremental",
        _fake_embed_repository_state_incremental,
    )

    manager = MapElitesManager(settings=settings, repo_root=Path("."))

    class NullSnapshotStore:
        def load(self, island_id: str, *, history_limit: int | None = None) -> dict[str, object] | None:
            return None

        def apply_update(self, island_id: str, *, update: object, session: object | None = None) -> None:
            return None

    manager._snapshot_store = NullSnapshotStore()  # type: ignore[attr-defined]
    _ = manager.ingest(commit_hash="c1")
    _ = manager.ingest(commit_hash="c2")

    assert len(seen_embedders) == 2
    assert isinstance(seen_embedders[0], RepositoryStateEmbedder)
    assert seen_embedders[0] is seen_embedders[1]


def test_manager_passes_snapshot_session_into_repo_state_incremental(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    settings.mapelites_ingest_info_log_every = 10_000
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
    seen_sessions: list[object | None] = []

    def _fake_embed_repository_state_incremental(*args, **kwargs):  # type: ignore[no-untyped-def]
        seen_sessions.append(kwargs.get("session"))
        return None, stats

    monkeypatch.setattr(
        map_elites_module,
        "embed_repository_state_incremental",
        _fake_embed_repository_state_incremental,
    )

    manager = MapElitesManager(settings=settings, repo_root=Path("."))
    snapshot_session = object()

    class NullSnapshotStore:
        def load(self, island_id: str, *, history_limit: int | None = None) -> dict[str, object] | None:
            return None

        def apply_update(self, island_id: str, *, update: object, session: object | None = None) -> None:
            return None

    manager._snapshot_store = NullSnapshotStore()  # type: ignore[attr-defined]
    _ = manager.ingest(commit_hash="c1", snapshot_session=snapshot_session)

    assert seen_sessions == [snapshot_session]


@pytest.mark.parametrize(
    ("source", "expected_aggregate", "expected_incremental"),
    [
        ("aggregate_hit", 1, 0),
        ("incremental", 0, 1),
    ],
)
def test_ingest_stage_metrics_count_repo_state_sources(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    captured_logs: list[dict[str, object]],
    source: str,
    expected_aggregate: int,
    expected_incremental: int,
) -> None:
    _configure_score_objective(settings)
    settings.mapelites_dimensionality_target_dims = 2
    settings.mapelites_ingest_info_log_every = 1

    projection = _identity_projection()
    entry = _history_entry()
    final_embedding = FinalEmbedding(
        commit_hash="abc",
        vector=(0.2, 0.8),
        dimensions=2,
        history_entry=entry,
        projection=projection,
    )
    stats = _repo_state_stats(cache_misses=2, source=source)
    monkeypatch.setattr(
        map_elites_module,
        "embed_repository_state_incremental",
        lambda *args, **kwargs: (_code_embedding(), stats),
    )
    monkeypatch.setattr(
        map_elites_module,
        "reduce_commit_embeddings",
        lambda **kwargs: (final_embedding, (entry,), projection, 0),
    )

    manager = MapElitesManager(settings=settings, repo_root=Path("."))
    manager._snapshot_store = _RecordingSnapshotStore()  # type: ignore[attr-defined]
    manager._ensure_island("main").projection = projection  # type: ignore[attr-defined]
    monkeypatch.setattr(manager, "_add_to_archive", lambda **kwargs: (0, 0.0, None))

    result = manager.ingest(commit_hash="abc", metrics=_score_metrics(1.0))

    assert result.status == 0
    stage_logs = [
        record
        for record in captured_logs
        if record.get("module") == "map_elites.manager"
        and record.get("message") == "MAP-Elites ingest stage metrics"
    ]
    assert stage_logs
    stage_extra = stage_logs[-1].get("extra")
    assert isinstance(stage_extra, dict)
    assert stage_extra["aggregate_hit_count"] == expected_aggregate
    assert stage_extra["incremental_count"] == expected_incremental
    assert stage_extra["embedding_cache_miss_count"] == 2


def test_ingest_warmup_persists_history_without_archive_update(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    _configure_score_objective(settings)
    settings.mapelites_dimensionality_target_dims = 2
    settings.mapelites_ingest_info_log_every = 1

    entry = _history_entry(commit_hash="warm")
    final_embedding = FinalEmbedding(
        commit_hash="warm",
        vector=(0.2, 0.8),
        dimensions=2,
        history_entry=entry,
        projection=None,
    )
    monkeypatch.setattr(
        map_elites_module,
        "embed_repository_state_incremental",
        lambda *args, **kwargs: (
            _code_embedding(),
            _repo_state_stats(commit_hash="warm", source="incremental"),
        ),
    )
    monkeypatch.setattr(
        map_elites_module,
        "reduce_commit_embeddings",
        lambda **kwargs: (final_embedding, (entry,), None, 1),
    )

    manager = MapElitesManager(settings=settings, repo_root=Path("."))
    recorder = _RecordingSnapshotStore()
    manager._snapshot_store = recorder  # type: ignore[attr-defined]

    def _fail_add_to_archive(**kwargs: object) -> tuple[int, float, None]:
        raise AssertionError("warmup commits must not touch the archive")

    monkeypatch.setattr(manager, "_add_to_archive", _fail_add_to_archive)

    result = manager.ingest(commit_hash="warm", metrics=_score_metrics(1.0))

    assert result.status == 0
    assert result.record is None
    assert result.message == "PCA warmup: projection is not ready; skipping archive update."
    assert recorder.updates
    island_id, update, session = recorder.updates[-1]
    assert island_id == "main"
    assert session is None
    assert isinstance(update, map_elites_module.SnapshotUpdate)
    assert update.history_upsert is entry
    assert update.projection is None
    assert update.front_replace is None
    assert update.archive_replace is None


def test_ingest_skips_archive_when_fitness_is_not_finite(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    _configure_score_objective(settings)
    settings.mapelites_dimensionality_target_dims = 2

    projection = _identity_projection()
    entry = _history_entry()
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
        lambda *args, **kwargs: (_code_embedding(), _repo_state_stats()),
    )
    monkeypatch.setattr(
        map_elites_module,
        "reduce_commit_embeddings",
        lambda **kwargs: (final_embedding, (entry,), projection, 0),
    )

    manager = MapElitesManager(settings=settings, repo_root=Path("."))
    recorder = _RecordingSnapshotStore()
    manager._snapshot_store = recorder  # type: ignore[attr-defined]
    manager._ensure_island("main").projection = projection  # type: ignore[attr-defined]

    def _fail_add_to_archive(**kwargs: object) -> tuple[int, float, None]:
        raise AssertionError("invalid fitness must not touch the archive")

    monkeypatch.setattr(manager, "_add_to_archive", _fail_add_to_archive)

    result = manager.ingest(
        commit_hash="abc",
        metrics=_score_metrics(float("nan")),
    )

    assert result.status == 0
    assert result.record is None
    assert "finite" in (result.message or "")
    assert recorder.updates
    update = recorder.updates[-1][1]
    assert isinstance(update, map_elites_module.SnapshotUpdate)
    assert update.front_replace is None


def test_ingest_builds_record_with_stubbed_dependencies(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    _configure_score_objective(settings)
    settings.mapelites_dimensionality_target_dims = 2
    settings.mapelites_feature_clip = True
    settings.mapelites_feature_truncation_k = 1.0

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
        objective_values: tuple[float, ...],
        objective_scores: tuple[float, ...],
        measures: np.ndarray,
    ) -> tuple[int, float, MapElitesRecord]:
        captured["measures"] = measures
        captured["objective_values"] = objective_values
        captured["objective_scores"] = objective_scores
        record = MapElitesRecord(
            commit_hash=commit_hash,
            island_id=island_id,
            cell_index=0,
            objective_values=objective_values,
            objective_scores=objective_scores,
            measures=tuple(measures.tolist()),
            timestamp=123.0,
        )
        return 1, 0.1, record

    monkeypatch.setattr(manager, "_add_to_archive", _fake_add_to_archive)

    result = manager.ingest(
        commit_hash="abc",
        metrics=_score_metrics(1.2),
    )

    assert result.inserted
    assert captured["objective_values"] == (1.2,)
    assert captured["objective_scores"] == (1.2,)
    assert captured["measures"] is not None
    assert tuple(captured["measures"].tolist()) == pytest.approx((0.6, 0.9))  # type: ignore[index]
    assert result.record is not None
    assert result.record.commit_hash == "abc"
    assert result.artifacts.code_embedding is code_embedding
    assert result.artifacts.final_embedding is final_embedding


def test_ingest_sets_message_when_archive_rejects_commit(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    _configure_score_objective(settings)
    settings.mapelites_dimensionality_target_dims = 2
    settings.mapelites_feature_clip = True
    settings.mapelites_feature_truncation_k = 1.0

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
        def load(
            self, island_id: str, *, history_limit: int | None = None
        ) -> dict[str, object] | None:
            return None

        def apply_update(
            self, island_id: str, *, update: object, session: object | None = None
        ) -> None:
            return None

    manager._snapshot_store = NullSnapshotStore()  # type: ignore[attr-defined]
    monkeypatch.setattr(manager, "_persist_island_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(manager, "_add_to_archive", lambda **kwargs: (-1, -0.2, None))

    result = manager.ingest(commit_hash="abc", metrics=_score_metrics(1.2))

    assert result.inserted is False
    assert result.status == -1
    assert result.record is None
    assert result.message is not None
    assert "not retained" in result.message


def test_reingesting_an_existing_commit_persists_the_complete_archive(
    settings: Settings,
) -> None:
    settings.mapelites_objectives = (
        ObjectiveSpec(name="quality", direction="max"),
        ObjectiveSpec(name="novelty", direction="max"),
    )
    settings.mapelites_dimensionality_target_dims = 2
    manager = MapElitesManager(settings=settings, repo_root=Path("."))
    lower, upper = manager._build_feature_bounds()  # noqa: SLF001
    state = map_elites_module.IslandState(
        archive=manager._build_archive(),  # noqa: SLF001
        lower_bounds=lower,
        upper_bounds=upper,
    )
    measures = np.asarray([0.2, 0.2], dtype=np.float64)
    manager._add_to_archive(  # noqa: SLF001
        state=state,
        island_id="main",
        commit_hash="repeat",
        objective_values=(5.0, 1.0),
        objective_scores=(5.0, 1.0),
        measures=measures,
    )
    manager._add_to_archive(  # noqa: SLF001
        state=state,
        island_id="main",
        commit_hash="other",
        objective_values=(1.0, 5.0),
        objective_scores=(1.0, 5.0),
        measures=measures,
    )
    assert "repeat" in state.commit_to_index
    update = map_elites_module.SnapshotUpdate(
        objective_contract=resolve_objective_contract(settings)
    )

    result = manager._insert_archive_candidate_for_ingest(  # noqa: SLF001
        state=state,
        island_id="main",
        commit_hash="repeat",
        candidate=map_elites_module._ArchiveCandidate(  # noqa: SLF001
            objectives=ResolvedObjectives(
                values=(0.0, 0.0),
                scores=(0.0, 0.0),
            ),
            vector=measures,
        ),
        update=update,
        archive_replace_needed=False,
        artifacts=map_elites_module.CommitEmbeddingArtifacts(
            repo_state_stats=None,
            preprocessed_files=(),
            code_embedding=None,
            final_embedding=None,
        ),
        emit_sampled_info=False,
        stage_metrics=map_elites_module._IngestStageMetrics(  # noqa: SLF001
            started_at=0.0
        ),
    )

    assert result.record is None
    assert "repeat" not in state.commit_to_index
    assert update.front_replace is None
    assert update.archive_replace is not None
    assert [elite.commit_hash for elite in update.archive_replace] == ["other"]


def test_ingest_passes_external_snapshot_session(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    _configure_score_objective(settings)
    settings.mapelites_dimensionality_target_dims = 2
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
                objective_values=(1.0,),
                objective_scores=(1.0,),
                measures=(0.2, 0.8),
                timestamp=123.0,
            ),
        ),
    )

    session_marker = object()
    _ = manager.ingest(
        commit_hash="abc",
        metrics=_score_metrics(1.0),
        snapshot_session=session_marker,  # type: ignore[arg-type]
    )

    assert recorder.last_session is session_marker


def test_ingest_seeds_archive_after_initial_pca_fit(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """Regression: warmup commits must populate the archive once PCA is fitted."""

    _configure_score_objective(settings)
    settings.mapelites_dimensionality_target_dims = 2
    settings.mapelites_archive_cells_per_dim = 4
    settings.mapelites_feature_clip = True
    settings.mapelites_feature_truncation_k = 1.0

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
        "_load_commit_objectives",
        lambda *, commit_hashes, snapshot_session: {
            "c1": ResolvedObjectives(values=(1.0,), scores=(1.0,)),
            "c2": ResolvedObjectives(values=(0.5,), scores=(0.5,)),
        },
    )

    first = manager.ingest(commit_hash="c1", metrics=_score_metrics(1.0))
    assert first.inserted is False
    assert manager.get_records("main") == ()

    _ = manager.ingest(commit_hash="c2", metrics=_score_metrics(0.5))
    records = {rec.commit_hash: rec.measures for rec in manager.get_records("main")}

    assert records["c1"] == pytest.approx((0.5, 0.5))
    assert records["c2"] == pytest.approx((1.0, 1.0))


def test_ingest_rebuilds_archive_when_pca_projection_refits(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """Regression: PCA refits must rebuild existing archive cells."""

    _configure_score_objective(settings)
    settings.mapelites_dimensionality_target_dims = 2
    settings.mapelites_archive_cells_per_dim = 4
    settings.mapelites_feature_clip = True
    settings.mapelites_feature_truncation_k = 1.0

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

    _ = manager.ingest(commit_hash="c1", metrics=_score_metrics(1.0))
    before = {rec.commit_hash: rec.measures for rec in manager.get_records("main")}
    assert before["c1"] == pytest.approx((0.5, 1.0))

    _ = manager.ingest(commit_hash="c2", metrics=_score_metrics(0.5))
    after = {rec.commit_hash: rec.measures for rec in manager.get_records("main")}

    # Under the refit projection, c1 loses its second component (it becomes 0),
    # so the y measure must move from 1.0 to 0.5 after rebuild.
    assert after["c1"] == pytest.approx((0.5, 0.5))


def test_refit_missing_an_elite_vector_preserves_the_previous_in_memory_state(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    _configure_score_objective(settings)
    settings.mapelites_dimensionality_target_dims = 2
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
    old_entry = PcaHistoryEntry(
        commit_hash="elite",
        vector=(0.0, 1.0, 0.0),
        embedding_model="code",
    )
    current_entry = PcaHistoryEntry(
        commit_hash="current",
        vector=(1.0, 0.0, 1.0),
        embedding_model="code",
    )
    current = FinalEmbedding(
        commit_hash="current",
        vector=(1.0, 1.0),
        dimensions=2,
        history_entry=current_entry,
        projection=new_projection,
    )
    manager = MapElitesManager(settings=settings, repo_root=Path("."))
    lower, upper = manager._build_feature_bounds()  # noqa: SLF001
    state = map_elites_module.IslandState(
        archive=manager._build_archive(),  # noqa: SLF001
        lower_bounds=lower,
        upper_bounds=upper,
        history=(old_entry,),
        projection=old_projection,
        samples_since_fit=3,
    )
    manager._archives["main"] = state  # noqa: SLF001
    manager._add_to_archive(  # noqa: SLF001
        state=state,
        island_id="main",
        commit_hash="elite",
        objective_values=(1.0,),
        objective_scores=(1.0,),
        measures=np.asarray([0.5, 0.5], dtype=np.float64),
    )
    before = manager.get_records("main")
    monkeypatch.setattr(
        map_elites_module,
        "reduce_commit_embeddings",
        lambda **_kwargs: (
            current,
            (old_entry, current_entry),
            new_projection,
            0,
        ),
    )
    monkeypatch.setattr(manager, "_load_commit_vectors", lambda **_kwargs: {})

    with pytest.raises(ValueError, match="stored vectors are missing"):
        manager._update_projection_for_ingest(  # noqa: SLF001
            state=state,
            island_id="main",
            commit_hash="current",
            code_embedding=CommitCodeEmbedding(
                files=(),
                vector=current_entry.vector,
                model="code",
                dimensions=3,
            ),
            snapshot_session=None,
            stage_metrics=map_elites_module._IngestStageMetrics(  # noqa: SLF001
                started_at=0.0
            ),
        )

    assert manager.get_records("main") == before
    assert state.history == (old_entry,)
    assert state.projection == old_projection
    assert state.samples_since_fit == 3
    assert manager._reducers["main"].projection == old_projection  # noqa: SLF001


def test_load_commit_vectors_batches_long_in_queries(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: long vector lookups must batch IN clauses for both lookup stages."""

    settings.mapelites_dimensionality_penultimate_normalize = False
    manager = MapElitesManager(settings=settings, repo_root=Path("."))
    monkeypatch.setattr(map_elites_db_ops, "_IN_QUERY_BATCH_SIZE", 2)

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
