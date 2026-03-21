from __future__ import annotations

from pathlib import Path
import inspect

import pytest

from benchmarks.db_benchmark_support import (
    blob_sha,
    clear_commit_aggregates,
    commit_all,
    commit_empty,
    init_repo,
    prepare_postgres_benchmark,
    sum_vectors,
    vector_for_sha,
)
from loreley.config import Settings
from loreley.db.base import session_scope
from loreley.core.map_elites.file_embedding_cache import DatabaseFileEmbeddingCache
from loreley.core.map_elites.repository_state_embedding import RepositoryStateEmbedder


def _configure_repo_state_settings(settings: Settings) -> None:
    settings.mapelites_code_embedding_model = "stub"
    settings.mapelites_preprocess_allowed_extensions = [".py"]
    settings.mapelites_preprocess_allowed_filenames = []
    settings.mapelites_preprocess_excluded_globs = []
    settings.mapelites_preprocess_max_file_size_kb = 64
    settings.mapelites_repo_state_ignore_text = ""


def _embed_incremental_with_optional_session(
    *,
    embedder: RepositoryStateEmbedder,
    commit_hash: str,
    repo_root: Path,
    batch_session: object,
):
    if "session" in inspect.signature(embedder.embed_incremental).parameters:
        return embedder.embed_incremental(
            commit_hash=commit_hash,
            repo_root=repo_root,
            session=batch_session,
        )
    return embedder.embed_incremental(
        commit_hash=commit_hash,
        repo_root=repo_root,
    )


@pytest.mark.benchmark(group="repo-state-db")
def test_repo_state_nodiff_db(
    benchmark,
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_repo_state_settings(settings)
    repo = init_repo(tmp_path)
    (tmp_path / "a.py").write_text("print('a1')\n", encoding="utf-8")
    parent = commit_all(repo, "c1")
    children = [commit_empty(repo, f"c{idx}") for idx in range(2, 8)]
    requested_dims = int(settings.mapelites_code_embedding_dimensions or 0)

    prepare_postgres_benchmark(monkeypatch, root_commit=parent)

    parent_blob = blob_sha(repo, parent, "a.py")

    def _setup():  # type: ignore[no-untyped-def]
        clear_commit_aggregates(children)
        cache = DatabaseFileEmbeddingCache(
            embedding_model=str(settings.mapelites_code_embedding_model),
            requested_dimensions=int(settings.mapelites_code_embedding_dimensions or 0),
        )
        embedder = RepositoryStateEmbedder(settings=settings, cache=cache, repo=repo)
        embedder._persist_aggregate(
            commit_hash=parent,
            repo_root=tmp_path,
            sum_vector=sum_vectors([parent_blob], dims=cache.requested_dimensions),
            file_count=1,
        )
        return (embedder, children), {}

    def _run(embedder: RepositoryStateEmbedder, child_commits: list[str]):  # type: ignore[no-untyped-def]
        last = None
        with session_scope() as batch_session:
            for commit_hash in child_commits:
                last = _embed_incremental_with_optional_session(
                    embedder=embedder,
                    commit_hash=commit_hash,
                    repo_root=tmp_path,
                    batch_session=batch_session,
                )
        assert last is not None
        return last

    embedding, stats = benchmark.pedantic(_run, setup=_setup, rounds=8, iterations=1)
    assert embedding is not None
    assert embedding.vector == pytest.approx(vector_for_sha(parent_blob, dims=requested_dims))
    assert stats.source == "incremental"
    assert stats.cache_misses == 0


@pytest.mark.benchmark(group="repo-state-db")
def test_repo_state_diff_zero_cache_miss_db(
    benchmark,
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_repo_state_settings(settings)
    repo = init_repo(tmp_path)

    file_count = 64
    modified = 16

    for idx in range(file_count):
        (tmp_path / f"f{idx:03d}.py").write_text(f"print({idx})\n", encoding="utf-8")
    parent = commit_all(repo, "c1")
    children: list[str] = []
    for step in range(1, 6):
        for idx in range(modified):
            (tmp_path / f"f{idx:03d}.py").write_text(
                f"print({idx} + {step})\n",
                encoding="utf-8",
            )
        children.append(commit_all(repo, f"c{step + 1}"))

    prepare_postgres_benchmark(monkeypatch, root_commit=parent)

    cache = DatabaseFileEmbeddingCache(
        embedding_model=str(settings.mapelites_code_embedding_model),
        requested_dimensions=int(settings.mapelites_code_embedding_dimensions or 0),
    )
    parent_blobs = [blob_sha(repo, parent, f"f{idx:03d}.py") for idx in range(file_count)]
    all_blobs = list(parent_blobs)
    changed_new_blobs: list[str] = []
    for commit_hash in children:
        all_blobs.extend(blob_sha(repo, commit_hash, f"f{idx:03d}.py") for idx in range(file_count))
        changed_new_blobs.extend(blob_sha(repo, commit_hash, f"f{idx:03d}.py") for idx in range(modified))
    cache.put_many(
        {
            sha: vector_for_sha(sha, dims=cache.requested_dimensions)
            for sha in sorted(set(all_blobs))
        }
    )

    def _setup():  # type: ignore[no-untyped-def]
        clear_commit_aggregates(children)
        embedder = RepositoryStateEmbedder(settings=settings, cache=cache, repo=repo)
        embedder._persist_aggregate(
            commit_hash=parent,
            repo_root=tmp_path,
            sum_vector=sum_vectors(parent_blobs, dims=cache.requested_dimensions),
            file_count=len(parent_blobs),
        )
        return (embedder, children), {}

    def _run(embedder: RepositoryStateEmbedder, child_commits: list[str]):  # type: ignore[no-untyped-def]
        last = None
        with session_scope() as batch_session:
            for commit_hash in child_commits:
                last = _embed_incremental_with_optional_session(
                    embedder=embedder,
                    commit_hash=commit_hash,
                    repo_root=tmp_path,
                    batch_session=batch_session,
                )
        assert last is not None
        return last

    embedding, stats = benchmark.pedantic(_run, setup=_setup, rounds=8, iterations=1)
    assert embedding is not None
    assert stats.source == "incremental"
    assert stats.cache_misses == 0
    assert stats.cache_hits == len(set(changed_new_blobs[-modified:]))
