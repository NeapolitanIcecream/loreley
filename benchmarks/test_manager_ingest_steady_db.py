from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.db_benchmark_support import (
    blob_sha,
    clear_commit_aggregates,
    commit_all,
    init_repo,
    prepare_postgres_benchmark,
    sum_vectors,
    vector_for_sha,
)
from loreley.config import Settings
from loreley.db.base import session_scope
from loreley.core.map_elites.dimension_reduction import DimensionReducer, PCAProjection, PcaHistoryEntry
from loreley.core.map_elites.file_embedding_cache import DatabaseFileEmbeddingCache
from loreley.core.map_elites.manager import MapElitesManager
from loreley.core.map_elites.repository_state_embedding import RepositoryStateEmbedder


def _configure_manager_settings(settings: Settings) -> None:
    settings.mapelites_code_embedding_model = "stub"
    settings.mapelites_preprocess_allowed_extensions = [".py"]
    settings.mapelites_preprocess_allowed_filenames = []
    settings.mapelites_preprocess_excluded_globs = []
    settings.mapelites_preprocess_max_file_size_kb = 64
    settings.mapelites_repo_state_ignore_text = ""
    settings.mapelites_islands = ("main",)
    settings.mapelites_dimensionality_target_dims = 2
    settings.mapelites_dimensionality_min_fit_samples = 2
    settings.mapelites_dimensionality_history_size = 8
    settings.mapelites_dimensionality_refit_interval = 10_000
    settings.mapelites_archive_cells_per_dim = 8
    settings.mapelites_ingest_info_log_every = 10_000


class _NullSnapshotStore:
    def load(self, island_id: str, *, history_limit: int | None = None) -> dict[str, object] | None:
        del island_id, history_limit
        return None

    def apply_update(self, island_id: str, *, update: object, session: object | None = None) -> None:
        del island_id, update, session
        return None


def _warm_history_entry(*, commit_hash: str, base: float, dims: int, model: str) -> PcaHistoryEntry:
    vector = tuple(base + (idx * 0.1) for idx in range(dims))
    return PcaHistoryEntry(
        commit_hash=commit_hash,
        vector=vector,
        embedding_model=model,
    )


@pytest.mark.benchmark(group="manager-ingest-db")
def test_manager_ingest_steady_db(
    benchmark,
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_manager_settings(settings)
    repo = init_repo(tmp_path)

    file_count = 16
    modified = 4
    added = 2
    child_steps = 5

    for idx in range(file_count):
        (tmp_path / f"f{idx:03d}.py").write_text(f"print({idx})\n", encoding="utf-8")
    parent = commit_all(repo, "c1")

    children: list[str] = []
    for step in range(1, child_steps + 1):
        for idx in range(modified):
            (tmp_path / f"f{idx:03d}.py").write_text(f"print({idx} + {step})\n", encoding="utf-8")
        for idx in range(added):
            (tmp_path / f"new{idx:03d}.py").write_text(
                f"print('new-{idx}-{step}')\n",
                encoding="utf-8",
            )
        children.append(commit_all(repo, f"c{step + 1}"))

    prepare_postgres_benchmark(monkeypatch, root_commit=parent)

    cache = DatabaseFileEmbeddingCache(
        embedding_model=str(settings.mapelites_code_embedding_model),
        requested_dimensions=int(settings.mapelites_code_embedding_dimensions or 0),
    )
    embedder = RepositoryStateEmbedder(settings=settings, cache=cache, repo=repo)

    parent_blobs = [blob_sha(repo, parent, f"f{idx:03d}.py") for idx in range(file_count)]
    all_blobs = list(parent_blobs)
    for commit_hash in children:
        commit_blobs = [blob_sha(repo, commit_hash, f"f{idx:03d}.py") for idx in range(file_count)] + [
            blob_sha(repo, commit_hash, f"new{idx:03d}.py") for idx in range(added)
        ]
        all_blobs.extend(commit_blobs)
    cache.put_many(
        {
            sha: vector_for_sha(sha, dims=cache.requested_dimensions)
            for sha in sorted(set(all_blobs))
        }
    )

    dims = cache.requested_dimensions
    projection = PCAProjection(
        feature_count=dims,
        components=(
            tuple(1.0 if idx == 0 else 0.0 for idx in range(dims)),
            tuple(1.0 if idx == 1 else 0.0 for idx in range(dims)),
        ),
        mean=tuple(0.0 for _ in range(dims)),
        explained_variance=(1.0, 1.0),
        explained_variance_ratio=(0.5, 0.5),
        sample_count=2,
        epoch=0,
        fitted_at=0.0,
        whiten=False,
        rotation=None,
    )
    history = (
        _warm_history_entry(commit_hash="warm-1", base=0.1, dims=dims, model=cache.embedding_model),
        _warm_history_entry(commit_hash="warm-2", base=0.2, dims=dims, model=cache.embedding_model),
    )

    def _setup():  # type: ignore[no-untyped-def]
        clear_commit_aggregates(children)
        embedder._persist_aggregate(
            commit_hash=parent,
            repo_root=tmp_path,
            sum_vector=sum_vectors(parent_blobs, dims=cache.requested_dimensions),
            file_count=len(parent_blobs),
        )
        manager = MapElitesManager(settings=settings, repo_root=tmp_path)
        manager._snapshot_store = _NullSnapshotStore()  # type: ignore[attr-defined]
        state = manager._ensure_island("main")
        state.history = history
        state.projection = projection
        state.samples_since_fit = 0
        manager._reducers["main"] = DimensionReducer(
            settings=settings,
            history=state.history,
            projection=state.projection,
            samples_since_fit=state.samples_since_fit,
        )
        return (manager, children), {}

    def _run(manager: MapElitesManager, child_commits: list[str]):  # type: ignore[no-untyped-def]
        last = None
        with session_scope() as batch_session:
            for idx, commit_hash in enumerate(child_commits, start=1):
                last = manager.ingest(
                    commit_hash=commit_hash,
                    metrics={"composite_score": float(idx)},
                    repo_root=tmp_path,
                    snapshot_session=batch_session,
                )
        assert last is not None
        return last

    result = benchmark.pedantic(_run, setup=_setup, rounds=8, iterations=1)
    assert result.inserted
    assert result.status > 0
