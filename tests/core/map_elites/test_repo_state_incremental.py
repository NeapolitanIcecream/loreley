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
        cfg.set_value("user", "name", "Test User")
        cfg.set_value("user", "email", "test@example.com")
    return repo


def _commit_all(repo: Repo, message: str) -> str:
    repo.git.add(A=True)
    commit = repo.index.commit(message)
    return commit.hexsha


def _blob_sha(repo: Repo, commit_hash: str, path: str) -> str:
    return repo.git.rev_parse(f"{commit_hash}:{path}").strip()


def _vec_for_sha(sha: str) -> tuple[float, float]:
    # Stable, deterministic pseudo-vector for tests.
    digest = bytes.fromhex(sha[:40])
    return (digest[0] / 255.0, digest[1] / 255.0)


def test_repo_state_incremental_aggregate_add_and_modify(
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_repo(tmp_path)
    settings.mapelites_preprocess_allowed_extensions = [".py"]
    settings.mapelites_preprocess_allowed_filenames = []
    settings.mapelites_preprocess_excluded_globs = []
    settings.mapelites_preprocess_max_file_size_kb = 64

    (tmp_path / "a.py").write_text("print('a1')\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("print('b1')\n", encoding="utf-8")
    c1 = _commit_all(repo, "c1")

    (tmp_path / "a.py").write_text("print('a2')\n", encoding="utf-8")
    (tmp_path / "c.py").write_text("print('c1')\n", encoding="utf-8")
    c2 = _commit_all(repo, "c2")

    sha_a1 = _blob_sha(repo, c1, "a.py")
    sha_b1 = _blob_sha(repo, c1, "b.py")
    sha_a2 = _blob_sha(repo, c2, "a.py")
    sha_c2 = _blob_sha(repo, c2, "c.py")

    parent_sum = _vec_for_sha(sha_a1)
    parent_sum = (parent_sum[0] + _vec_for_sha(sha_b1)[0], parent_sum[1] + _vec_for_sha(sha_b1)[1])

    parent_agg = SimpleNamespace(
        file_count=2,
        sum_vector=[parent_sum[0], parent_sum[1]],
    )

    persisted: dict[str, object] = {}

    # Fake DB aggregate store
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

    # Fake DB file-cache metadata
    def _fake_load_file_cache_metadata(*, blob_shas, dimensions: int):  # type: ignore[no-untyped-def]
        dims = int(dimensions)
        assert dims == 2
        meta = {}
        # Parent blobs exist; new blobs are treated as cache misses so embed_cache_misses is exercised.
        meta[sha_a1] = RepositoryStateEmbedder._VectorMeta(vector=_vec_for_sha(sha_a1))
        meta[sha_b1] = RepositoryStateEmbedder._VectorMeta(vector=_vec_for_sha(sha_b1))
        # Treat new blobs as cache misses so embed_cache_misses is exercised.
        return meta

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
        experiment_id="00000000-0000-0000-0000-000000000000",
        embedding_model="stub",
        requested_dimensions=2,
    )
    monkeypatch.setattr(DatabaseFileEmbeddingCache, "put_many", lambda *_args, **_kwargs: None)

    embedder = RepositoryStateEmbedder(
        settings=settings,
        cache=cache,
        repo=repo,
        experiment_id="00000000-0000-0000-0000-000000000000",
    )

    monkeypatch.setattr(embedder, "_load_aggregate", _fake_load_aggregate)
    monkeypatch.setattr(embedder, "_persist_aggregate", _fake_persist_aggregate)
    monkeypatch.setattr(embedder, "_load_file_cache_metadata", _fake_load_file_cache_metadata)
    monkeypatch.setattr(embedder, "_embed_cache_misses", _fake_embed_cache_misses)

    embedding, stats = embedder.run(commit_hash=c2, repo_root=tmp_path)
    assert embedding is not None
    assert stats.files_aggregated == 3

    expected_sum = (
        _vec_for_sha(sha_a2)[0] + _vec_for_sha(sha_b1)[0] + _vec_for_sha(sha_c2)[0],
        _vec_for_sha(sha_a2)[1] + _vec_for_sha(sha_b1)[1] + _vec_for_sha(sha_c2)[1],
    )
    expected_mean = (expected_sum[0] / 3.0, expected_sum[1] / 3.0)
    assert embedding.vector == pytest.approx(expected_mean)


