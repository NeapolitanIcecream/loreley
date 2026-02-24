from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from git import Repo

from loreley.config import Settings
from loreley.core.map_elites.file_embedding_cache import DatabaseFileEmbeddingCache
from loreley.core.map_elites.repository_state_embedding import RepositoryStateEmbedder


def _init_repo(tmp_path: Path) -> Repo:
    repo = Repo.init(tmp_path)
    with repo.config_writer() as cfg:
        cfg.set_value("user", "name", "Benchmark User")
        cfg.set_value("user", "email", "benchmark@example.com")
    return repo


def _commit_all(repo: Repo, message: str) -> str:
    repo.git.add(A=True)
    commit = repo.index.commit(message)
    return commit.hexsha


def _blob_sha(repo: Repo, commit_hash: str, path: str) -> str:
    return repo.git.rev_parse(f"{commit_hash}:{path}").strip()


def _vec_for_sha(sha: str) -> tuple[float, float]:
    # Stable, deterministic pseudo-vector for benchmarks.
    digest = bytes.fromhex(sha[:40])
    return (digest[0] / 255.0, digest[1] / 255.0)


@pytest.mark.benchmark(group="repo-state")
def test_repo_state_incremental_embed(
    benchmark,
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_repo(tmp_path)
    settings.mapelites_preprocess_allowed_extensions = [".py"]
    settings.mapelites_preprocess_allowed_filenames = []
    settings.mapelites_preprocess_excluded_globs = []
    settings.mapelites_preprocess_max_file_size_kb = 64
    settings.mapelites_repo_state_ignore_text = ""

    file_count = 64
    modified = 16
    added = 8

    for i in range(file_count):
        (tmp_path / f"f{i}.py").write_text(f"print({i})\n", encoding="utf-8")
    c1 = _commit_all(repo, "c1")

    for i in range(modified):
        (tmp_path / f"f{i}.py").write_text(f"print({i} + 1)\n", encoding="utf-8")
    for i in range(added):
        (tmp_path / f"new{i}.py").write_text("print('new')\n", encoding="utf-8")
    c2 = _commit_all(repo, "c2")

    parent_blob_shas = [_blob_sha(repo, c1, f"f{i}.py") for i in range(file_count)]
    parent_sha_set = set(parent_blob_shas)
    sum0 = 0.0
    sum1 = 0.0
    for sha in parent_blob_shas:
        v0, v1 = _vec_for_sha(sha)
        sum0 += v0
        sum1 += v1

    parent_agg = SimpleNamespace(
        file_count=file_count,
        sum_vector=[sum0, sum1],
    )

    persisted: dict[str, object] = {}

    def _fake_load_aggregate(*, commit_hash: str, repo_root: Path):  # type: ignore[no-untyped-def]
        if commit_hash == c1:
            return parent_agg
        return persisted.get(commit_hash)

    def _fake_persist_aggregate(  # type: ignore[no-untyped-def]
        *,
        commit_hash: str,
        repo_root: Path,
        sum_vector,
        file_count: int,
    ) -> None:
        persisted[commit_hash] = SimpleNamespace(
            file_count=int(file_count),
            sum_vector=list(sum_vector),
        )

    def _fake_load_file_cache_metadata(*, blob_shas, dimensions: int):  # type: ignore[no-untyped-def]
        dims = int(dimensions)
        if dims != 2:
            raise AssertionError(f"Unexpected dimensions: {dims}")
        return {
            sha: RepositoryStateEmbedder._VectorMeta(vector=_vec_for_sha(sha))
            for sha in blob_shas
            if sha in parent_sha_set
        }

    def _fake_embed_cache_misses(  # type: ignore[no-untyped-def]
        *,
        root: Path,
        commit_hash: str,
        repo_files,
        missing_blob_shas,
    ):
        vectors = {sha: _vec_for_sha(sha) for sha in missing_blob_shas}
        return vectors, len(vectors), 0

    cache = DatabaseFileEmbeddingCache(
        embedding_model="stub",
        requested_dimensions=2,
    )
    monkeypatch.setattr(DatabaseFileEmbeddingCache, "put_many", lambda *_args, **_kwargs: None)

    embedder = RepositoryStateEmbedder(
        settings=settings,
        cache=cache,
        repo=repo,
    )
    monkeypatch.setattr(embedder, "_load_aggregate", _fake_load_aggregate)
    monkeypatch.setattr(embedder, "_persist_aggregate", _fake_persist_aggregate)
    monkeypatch.setattr(embedder, "_load_file_cache_metadata", _fake_load_file_cache_metadata)
    monkeypatch.setattr(embedder, "_embed_cache_misses", _fake_embed_cache_misses)

    def _run():  # type: ignore[no-untyped-def]
        persisted.pop(c2, None)
        return embedder.embed_incremental(commit_hash=c2, repo_root=tmp_path)

    embedding, stats = benchmark(_run)
    assert embedding is not None
    assert stats.source == "incremental"

