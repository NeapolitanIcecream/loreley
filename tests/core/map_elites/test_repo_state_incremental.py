from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import pytest
from git import Repo

import loreley.core.map_elites.repository_state_embedding as repo_state_mod
from loreley.config import Settings
from loreley.core.map_elites.file_embedding_cache import DatabaseFileEmbeddingCache
from loreley.core.map_elites.repository_state_embedding import RepoStateEmbeddingError, RepositoryStateEmbedder
from loreley.db.models import MapElitesRepoStateAggregate


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


def _commit_empty(repo: Repo, message: str) -> str:
    repo.git.commit("--allow-empty", "-m", message)
    return repo.head.commit.hexsha


def _blob_sha(repo: Repo, commit_hash: str, path: str) -> str:
    return repo.git.rev_parse(f"{commit_hash}:{path}").strip()


def _vec_for_sha(sha: str) -> tuple[float, float]:
    # Stable, deterministic pseudo-vector for tests.
    digest = bytes.fromhex(sha[:40])
    return (digest[0] / 255.0, digest[1] / 255.0)


def _aggregate_row(
    *,
    commit_hash: str,
    file_count: int,
    sum_vector: Sequence[float],
) -> MapElitesRepoStateAggregate:
    return MapElitesRepoStateAggregate(
        commit_hash=commit_hash,
        file_count=int(file_count),
        sum_vector=[float(value) for value in sum_vector],
    )


class _FakeExecuteResult:
    def __init__(
        self,
        *,
        scalar: Any | None = None,
        rows: Sequence[Any] = (),
    ) -> None:
        self._scalar = scalar
        self._rows = list(rows)

    def scalar_one_or_none(self) -> Any | None:
        return self._scalar

    def scalar_one(self) -> Any:
        if self._scalar is None:
            raise AssertionError("scalar_one() expected a row")
        return self._scalar

    def all(self) -> list[Any]:
        return list(self._rows)


class _RepoStateSessionRecorder:
    def __init__(
        self,
        *,
        embedding_model: str,
        dimensions: int,
        aggregates: dict[str, MapElitesRepoStateAggregate] | None = None,
        file_cache: dict[str, tuple[float, ...]] | None = None,
        fail_persist: Exception | None = None,
    ) -> None:
        self.embedding_model = embedding_model
        self.dimensions = int(dimensions)
        self.aggregates = dict(aggregates or {})
        self.file_cache = dict(file_cache or {})
        self.info: dict[object, object] = {}
        self.fail_persist = fail_persist
        self.scope_entries = 0
        self.execute_calls = 0
        self.aggregate_selects = 0
        self.file_cache_selects = 0
        self.aggregate_write_calls = 0
        self.loaded_commits: list[str] = []
        self.loaded_blob_batches: list[list[str]] = []
        self.persisted_commits: list[str] = []

    @contextmanager
    def scope(self):  # type: ignore[no-untyped-def]
        self.scope_entries += 1
        yield self

    def execute(self, stmt: Any) -> _FakeExecuteResult:
        self.execute_calls += 1

        table_name = getattr(getattr(stmt, "table", None), "name", None)
        if table_name == "map_elites_repo_state_aggregates":
            if self.fail_persist is not None:
                raise self.fail_persist
            params = stmt.compile().params
            row = _aggregate_row(
                commit_hash=str(params["commit_hash"]),
                file_count=int(params["file_count"]),
                sum_vector=tuple(float(value) for value in params["sum_vector"]),
            )
            self.aggregate_write_calls += 1
            self.persisted_commits.append(row.commit_hash)
            self.aggregates[row.commit_hash] = row
            return _FakeExecuteResult(scalar=row)

        froms = [from_clause.name for from_clause in stmt.get_final_froms()]
        compiled = stmt.compile()
        if "map_elites_repo_state_aggregates" in froms:
            commit_hash = str(next(iter(compiled.params.values())))
            self.aggregate_selects += 1
            self.loaded_commits.append(commit_hash)
            return _FakeExecuteResult(scalar=self.aggregates.get(commit_hash))
        if "map_elites_file_embedding_cache" in froms:
            batch: list[str] = []
            for value in compiled.params.values():
                if isinstance(value, list):
                    batch = [str(item) for item in value]
                    break
            self.file_cache_selects += 1
            self.loaded_blob_batches.append(batch)
            rows = [
                (sha, list(vector), self.embedding_model, self.dimensions)
                for sha, vector in self.file_cache.items()
                if sha in batch
            ]
            return _FakeExecuteResult(rows=rows)

        raise AssertionError(f"Unexpected statement: {stmt}")

    def merge(self, row: MapElitesRepoStateAggregate) -> None:
        if self.fail_persist is not None:
            raise self.fail_persist
        persisted = _aggregate_row(
            commit_hash=str(row.commit_hash),
            file_count=int(row.file_count),
            sum_vector=tuple(float(value) for value in row.sum_vector),
        )
        self.aggregate_write_calls += 1
        self.persisted_commits.append(persisted.commit_hash)
        self.aggregates[persisted.commit_hash] = persisted


def _build_test_cache(*, dimensions: int = 2) -> DatabaseFileEmbeddingCache:
    return DatabaseFileEmbeddingCache(
        embedding_model="stub",
        requested_dimensions=dimensions,
    )


def test_repo_state_db_helpers_reuse_provided_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: Settings,
) -> None:
    cache = _build_test_cache()
    session = _RepoStateSessionRecorder(
        embedding_model=cache.embedding_model,
        dimensions=cache.requested_dimensions,
        aggregates={
            "parent": _aggregate_row(
                commit_hash="parent",
                file_count=1,
                sum_vector=(0.1, 0.2),
            ),
        },
        file_cache={"sha-parent": (0.1, 0.2)},
    )

    @contextmanager
    def _unexpected_session_scope():  # type: ignore[no-untyped-def]
        raise AssertionError("session_scope() should not be used when a session is provided")
        yield session

    monkeypatch.setattr(repo_state_mod, "session_scope", _unexpected_session_scope)

    embedder = RepositoryStateEmbedder(settings=settings, cache=cache)
    loaded = embedder._load_aggregate(
        commit_hash="parent",
        repo_root=tmp_path,
        session=session,
    )
    metadata = embedder._load_file_cache_metadata(
        blob_shas=["sha-parent"],
        dimensions=2,
        session=session,
    )
    persisted = embedder._persist_aggregate(
        commit_hash="child",
        repo_root=tmp_path,
        sum_vector=(0.3, 0.4),
        file_count=1,
        session=session,
    )

    assert loaded is not None
    assert metadata["sha-parent"].vector == pytest.approx((0.1, 0.2))
    assert persisted.commit_hash == "child"
    assert session.scope_entries == 0
    assert session.aggregate_selects == 1
    assert session.file_cache_selects == 1
    assert session.aggregate_write_calls == 1


def test_repo_state_incremental_nodiff_uses_one_session_and_skips_reload(
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

    (tmp_path / "a.py").write_text("print('a1')\n", encoding="utf-8")
    c1 = _commit_all(repo, "c1")
    c2 = _commit_empty(repo, "c2")

    sha_a1 = _blob_sha(repo, c1, "a.py")
    parent_vector = _vec_for_sha(sha_a1)
    cache = _build_test_cache()
    session = _RepoStateSessionRecorder(
        embedding_model=cache.embedding_model,
        dimensions=cache.requested_dimensions,
        aggregates={
            c1: _aggregate_row(
                commit_hash=c1,
                file_count=1,
                sum_vector=parent_vector,
            ),
        },
    )
    monkeypatch.setattr(repo_state_mod, "session_scope", session.scope)

    embedder = RepositoryStateEmbedder(settings=settings, cache=cache, repo=repo)
    embedding, stats = embedder.embed_incremental(commit_hash=c2, repo_root=tmp_path)

    assert embedding is not None
    assert embedding.vector == pytest.approx(parent_vector)
    assert stats.source == "incremental"
    assert session.scope_entries == 1
    assert session.aggregate_selects == 2
    assert session.file_cache_selects == 0
    assert session.aggregate_write_calls == 1
    assert session.loaded_commits == [c2, c1]
    assert session.persisted_commits == [c2]


def test_repo_state_incremental_zero_cache_miss_uses_one_session(
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

    parent_sum = (
        _vec_for_sha(sha_a1)[0] + _vec_for_sha(sha_b1)[0],
        _vec_for_sha(sha_a1)[1] + _vec_for_sha(sha_b1)[1],
    )
    cache = _build_test_cache()
    session = _RepoStateSessionRecorder(
        embedding_model=cache.embedding_model,
        dimensions=cache.requested_dimensions,
        aggregates={
            c1: _aggregate_row(
                commit_hash=c1,
                file_count=2,
                sum_vector=parent_sum,
            ),
        },
        file_cache={
            sha_a1: _vec_for_sha(sha_a1),
            sha_a2: _vec_for_sha(sha_a2),
            sha_c2: _vec_for_sha(sha_c2),
        },
    )
    monkeypatch.setattr(repo_state_mod, "session_scope", session.scope)

    embedder = RepositoryStateEmbedder(settings=settings, cache=cache, repo=repo)

    def _unexpected_embed_cache_misses(**_kwargs: Any) -> tuple[dict[str, tuple[float, ...]], int, int]:
        raise AssertionError("_embed_cache_misses() should not run when file cache metadata is warm")

    monkeypatch.setattr(embedder, "_embed_cache_misses", _unexpected_embed_cache_misses)

    embedding, stats = embedder.embed_incremental(commit_hash=c2, repo_root=tmp_path)

    assert embedding is not None
    assert embedding.vector == pytest.approx(
        (
            (_vec_for_sha(sha_a2)[0] + _vec_for_sha(sha_b1)[0] + _vec_for_sha(sha_c2)[0]) / 3.0,
            (_vec_for_sha(sha_a2)[1] + _vec_for_sha(sha_b1)[1] + _vec_for_sha(sha_c2)[1]) / 3.0,
        )
    )
    assert stats.source == "incremental"
    assert stats.cache_hits == 2
    assert stats.cache_misses == 0
    assert session.scope_entries == 1
    assert session.aggregate_selects == 2
    assert session.file_cache_selects == 1
    assert session.aggregate_write_calls == 1
    assert session.loaded_commits == [c2, c1]


def test_repo_state_incremental_reuses_cached_parent_aggregate_across_ingests(
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

    (tmp_path / "a.py").write_text("print('a1')\n", encoding="utf-8")
    c1 = _commit_all(repo, "c1")
    c2 = _commit_empty(repo, "c2")
    c3 = _commit_empty(repo, "c3")

    sha_a1 = _blob_sha(repo, c1, "a.py")
    parent_vector = _vec_for_sha(sha_a1)
    cache = _build_test_cache()
    session = _RepoStateSessionRecorder(
        embedding_model=cache.embedding_model,
        dimensions=cache.requested_dimensions,
        aggregates={
            c1: _aggregate_row(
                commit_hash=c1,
                file_count=1,
                sum_vector=parent_vector,
            ),
        },
    )
    monkeypatch.setattr(repo_state_mod, "session_scope", session.scope)

    embedder = RepositoryStateEmbedder(settings=settings, cache=cache, repo=repo)
    first_embedding, _first_stats = embedder.embed_incremental(commit_hash=c2, repo_root=tmp_path)
    second_embedding, second_stats = embedder.embed_incremental(commit_hash=c3, repo_root=tmp_path)

    assert first_embedding is not None
    assert second_embedding is not None
    assert second_embedding.vector == pytest.approx(first_embedding.vector)
    assert second_stats.source == "incremental"
    assert session.scope_entries == 2
    assert session.aggregate_selects == 3
    assert session.loaded_commits == [c2, c1, c3]
    assert session.persisted_commits == [c2, c3]


def test_file_cache_metadata_reuses_memory_cache_for_repeated_blob_queries(
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _build_test_cache()
    session = _RepoStateSessionRecorder(
        embedding_model=cache.embedding_model,
        dimensions=cache.requested_dimensions,
        file_cache={
            "sha-a": (0.1, 0.2),
            "sha-b": (0.3, 0.4),
        },
    )
    monkeypatch.setattr(repo_state_mod, "session_scope", session.scope)

    embedder = RepositoryStateEmbedder(settings=settings, cache=cache)
    first = embedder._load_file_cache_metadata(blob_shas=["sha-a"], dimensions=2)
    second = embedder._load_file_cache_metadata(blob_shas=["sha-a", "sha-b"], dimensions=2)

    assert first["sha-a"].vector == pytest.approx((0.1, 0.2))
    assert second["sha-a"].vector == pytest.approx((0.1, 0.2))
    assert second["sha-b"].vector == pytest.approx((0.3, 0.4))
    assert session.scope_entries == 2
    assert session.file_cache_selects == 2
    assert session.loaded_blob_batches == [["sha-a"], ["sha-b"]]


def test_session_persisted_aggregate_stays_transaction_local_until_commit(
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _build_test_cache()
    writer_session = _RepoStateSessionRecorder(
        embedding_model=cache.embedding_model,
        dimensions=cache.requested_dimensions,
    )
    reader_session = _RepoStateSessionRecorder(
        embedding_model=cache.embedding_model,
        dimensions=cache.requested_dimensions,
    )

    @contextmanager
    def _reader_scope():  # type: ignore[no-untyped-def]
        yield reader_session

    monkeypatch.setattr(repo_state_mod, "session_scope", _reader_scope)

    embedder = RepositoryStateEmbedder(settings=settings, cache=cache)
    persisted = embedder._persist_aggregate(
        commit_hash="child",
        repo_root=tmp_path,
        sum_vector=(0.3, 0.4),
        file_count=1,
        session=writer_session,
    )

    in_tx = embedder._load_aggregate(
        commit_hash="child",
        repo_root=tmp_path,
        session=writer_session,
    )
    outside_tx = embedder._load_aggregate(
        commit_hash="child",
        repo_root=tmp_path,
    )

    assert persisted.commit_hash == "child"
    assert in_tx is not None
    assert in_tx.commit_hash == "child"
    assert outside_tx is None
    assert embedder._cached_aggregate("child") is None
    assert writer_session.aggregate_selects == 0
    assert reader_session.aggregate_selects == 1


def test_load_aggregate_keeps_dimension_mismatch_error_semantics(
    tmp_path: Path,
    settings: Settings,
) -> None:
    cache = _build_test_cache(dimensions=2)
    session = _RepoStateSessionRecorder(
        embedding_model=cache.embedding_model,
        dimensions=cache.requested_dimensions,
        aggregates={
            "bad": _aggregate_row(
                commit_hash="bad",
                file_count=1,
                sum_vector=(1.0, 2.0, 3.0),
            ),
        },
    )
    embedder = RepositoryStateEmbedder(settings=settings, cache=cache)

    with pytest.raises(RepoStateEmbeddingError, match="unexpected dimensions"):
        embedder._load_aggregate(
            commit_hash="bad",
            repo_root=tmp_path,
            session=session,
        )


def test_load_aggregate_keeps_empty_sum_vector_error_semantics(
    tmp_path: Path,
    settings: Settings,
) -> None:
    cache = _build_test_cache(dimensions=2)
    session = _RepoStateSessionRecorder(
        embedding_model=cache.embedding_model,
        dimensions=cache.requested_dimensions,
        aggregates={
            "bad": _aggregate_row(
                commit_hash="bad",
                file_count=1,
                sum_vector=(),
            ),
        },
    )
    embedder = RepositoryStateEmbedder(settings=settings, cache=cache)

    with pytest.raises(RepoStateEmbeddingError, match="sum vector is missing"):
        embedder._load_aggregate(
            commit_hash="bad",
            repo_root=tmp_path,
            session=session,
        )


def test_persist_aggregate_keeps_persist_failure_semantics(
    tmp_path: Path,
    settings: Settings,
) -> None:
    cache = _build_test_cache(dimensions=2)
    session = _RepoStateSessionRecorder(
        embedding_model=cache.embedding_model,
        dimensions=cache.requested_dimensions,
        fail_persist=RuntimeError("boom"),
    )
    embedder = RepositoryStateEmbedder(settings=settings, cache=cache)

    with pytest.raises(RepoStateEmbeddingError, match="persist failed"):
        embedder._persist_aggregate(
            commit_hash="bad",
            repo_root=tmp_path,
            sum_vector=(1.0, 2.0),
            file_count=1,
            session=session,
        )


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
    def _fake_load_aggregate(
        *,
        commit_hash: str,
        repo_root: Path,
        session: object | None = None,
    ):  # type: ignore[no-untyped-def]
        del repo_root, session
        if commit_hash == c1:
            return parent_agg
        return persisted.get(commit_hash)

    def _fake_persist_aggregate(  # type: ignore[no-untyped-def]
        *,
        commit_hash: str,
        repo_root: Path,
        sum_vector,
        file_count: int,
        session: object | None = None,
    ) -> SimpleNamespace:
        del repo_root, session
        persisted_row = SimpleNamespace(
            file_count=int(file_count),
            sum_vector=list(sum_vector),
        )
        persisted[commit_hash] = persisted_row
        return persisted_row

    # Fake DB file-cache metadata
    def _fake_load_file_cache_metadata(
        *,
        blob_shas,
        dimensions: int,
        session: object | None = None,
    ):  # type: ignore[no-untyped-def]
        del session
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

    embedding, stats = embedder.embed_incremental(commit_hash=c2, repo_root=tmp_path)
    assert embedding is not None
    assert stats.files_aggregated == 3

    expected_sum = (
        _vec_for_sha(sha_a2)[0] + _vec_for_sha(sha_b1)[0] + _vec_for_sha(sha_c2)[0],
        _vec_for_sha(sha_a2)[1] + _vec_for_sha(sha_b1)[1] + _vec_for_sha(sha_c2)[1],
    )
    expected_mean = (expected_sum[0] / 3.0, expected_sum[1] / 3.0)
    assert embedding.vector == pytest.approx(expected_mean)


def test_repo_state_incremental_delete_last_eligible_file_returns_empty_embedding(
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: deleting the final eligible file should not raise incremental embedding errors."""

    repo = _init_repo(tmp_path)
    settings.mapelites_preprocess_allowed_extensions = [".py"]
    settings.mapelites_preprocess_allowed_filenames = []
    settings.mapelites_preprocess_excluded_globs = []
    settings.mapelites_preprocess_max_file_size_kb = 64
    settings.mapelites_repo_state_ignore_text = ""

    (tmp_path / "a.py").write_text("print('a1')\n", encoding="utf-8")
    c1 = _commit_all(repo, "c1")

    (tmp_path / "a.py").unlink()
    c2 = _commit_all(repo, "c2")

    sha_a1 = _blob_sha(repo, c1, "a.py")
    parent_agg = SimpleNamespace(
        file_count=1,
        sum_vector=list(_vec_for_sha(sha_a1)),
    )

    persisted: dict[str, object] = {}

    def _fake_load_aggregate(
        *,
        commit_hash: str,
        repo_root: Path,
        session: object | None = None,
    ):  # type: ignore[no-untyped-def]
        del repo_root, session
        if commit_hash == c1:
            return parent_agg
        return persisted.get(commit_hash)

    def _fake_persist_aggregate(  # type: ignore[no-untyped-def]
        *,
        commit_hash: str,
        repo_root: Path,
        sum_vector,
        file_count: int,
        session: object | None = None,
    ) -> SimpleNamespace:
        del repo_root, session
        persisted_row = SimpleNamespace(
            file_count=int(file_count),
            sum_vector=list(sum_vector),
        )
        persisted[commit_hash] = persisted_row
        return persisted_row

    def _fake_load_file_cache_metadata(
        *,
        blob_shas,
        dimensions: int,
        session: object | None = None,
    ):  # type: ignore[no-untyped-def]
        del session
        dims = int(dimensions)
        assert dims == 2
        return {
            sha_a1: RepositoryStateEmbedder._VectorMeta(vector=_vec_for_sha(sha_a1)),
        }

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

    embedding, stats = embedder.embed_incremental(commit_hash=c2, repo_root=tmp_path)

    assert embedding is None
    assert stats.source == "incremental"
    assert stats.files_aggregated == 0
    persisted_agg = persisted[c2]
    assert persisted_agg.file_count == 0
    assert tuple(persisted_agg.sum_vector) == pytest.approx((0.0, 0.0))


def test_repo_state_incremental_reuses_blob_size_checks_across_selection_passes(
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Incremental selection should not repeat blob-size git calls for the same diff entry."""

    repo = _init_repo(tmp_path)
    settings.mapelites_preprocess_allowed_extensions = [".py"]
    settings.mapelites_preprocess_allowed_filenames = []
    settings.mapelites_preprocess_excluded_globs = []
    settings.mapelites_preprocess_max_file_size_kb = 64
    settings.mapelites_repo_state_ignore_text = ""

    (tmp_path / "a.py").write_text("print('a1')\n", encoding="utf-8")
    c1 = _commit_all(repo, "c1")

    (tmp_path / "a.py").write_text("print('a2')\n", encoding="utf-8")
    c2 = _commit_all(repo, "c2")

    sha_a1 = _blob_sha(repo, c1, "a.py")
    sha_a2 = _blob_sha(repo, c2, "a.py")

    parent_agg = SimpleNamespace(
        file_count=1,
        sum_vector=list(_vec_for_sha(sha_a1)),
    )

    persisted: dict[str, object] = {}

    def _fake_load_aggregate(
        *,
        commit_hash: str,
        repo_root: Path,
        session: object | None = None,
    ):  # type: ignore[no-untyped-def]
        del repo_root, session
        if commit_hash == c1:
            return parent_agg
        return persisted.get(commit_hash)

    def _fake_persist_aggregate(  # type: ignore[no-untyped-def]
        *,
        commit_hash: str,
        repo_root: Path,
        sum_vector,
        file_count: int,
        session: object | None = None,
    ) -> SimpleNamespace:
        del repo_root, session
        persisted_row = SimpleNamespace(
            file_count=int(file_count),
            sum_vector=list(sum_vector),
        )
        persisted[commit_hash] = persisted_row
        return persisted_row

    def _fake_load_file_cache_metadata(
        *,
        blob_shas,
        dimensions: int,
        session: object | None = None,
    ):  # type: ignore[no-untyped-def]
        del session
        dims = int(dimensions)
        assert dims == 2
        return {
            sha_a1: RepositoryStateEmbedder._VectorMeta(vector=_vec_for_sha(sha_a1)),
            sha_a2: RepositoryStateEmbedder._VectorMeta(vector=_vec_for_sha(sha_a2)),
        }

    seen_blob_size_batches: list[list[str]] = []

    def _fake_load_blob_sizes(repo_obj, blob_shas):  # type: ignore[no-untyped-def]
        assert repo_obj is repo
        batch = list(blob_shas)
        seen_blob_size_batches.append(batch)
        return {sha: 16 for sha in batch}

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
    monkeypatch.setattr(repo_state_mod, "_load_blob_sizes", _fake_load_blob_sizes)

    embedding, stats = embedder.embed_incremental(commit_hash=c2, repo_root=tmp_path)

    assert embedding is not None
    assert stats.files_aggregated == 1
    assert len(seen_blob_size_batches) == 1
    assert sorted(seen_blob_size_batches[0]) == sorted([sha_a1, sha_a2])


def test_repo_state_incremental_can_rebuild_from_zero_file_parent(
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a child commit must recover from a persisted zero-file parent aggregate."""

    repo = _init_repo(tmp_path)
    settings.mapelites_preprocess_allowed_extensions = [".py"]
    settings.mapelites_preprocess_allowed_filenames = []
    settings.mapelites_preprocess_excluded_globs = []
    settings.mapelites_preprocess_max_file_size_kb = 64
    settings.mapelites_repo_state_ignore_text = ""

    (tmp_path / "a.py").write_text("print('a1')\n", encoding="utf-8")
    c1 = _commit_all(repo, "c1")

    (tmp_path / "a.py").unlink()
    c2 = _commit_all(repo, "c2")

    (tmp_path / "b.py").write_text("print('b1')\n", encoding="utf-8")
    c3 = _commit_all(repo, "c3")

    sha_a1 = _blob_sha(repo, c1, "a.py")
    sha_b3 = _blob_sha(repo, c3, "b.py")
    parent_agg = SimpleNamespace(
        file_count=1,
        sum_vector=list(_vec_for_sha(sha_a1)),
    )

    persisted: dict[str, object] = {}

    def _fake_load_aggregate(
        *,
        commit_hash: str,
        repo_root: Path,
        session: object | None = None,
    ):  # type: ignore[no-untyped-def]
        del repo_root, session
        if commit_hash == c1:
            return parent_agg
        return persisted.get(commit_hash)

    def _fake_persist_aggregate(  # type: ignore[no-untyped-def]
        *,
        commit_hash: str,
        repo_root: Path,
        sum_vector,
        file_count: int,
        session: object | None = None,
    ) -> SimpleNamespace:
        del repo_root, session
        persisted_row = SimpleNamespace(
            file_count=int(file_count),
            sum_vector=list(sum_vector),
        )
        persisted[commit_hash] = persisted_row
        return persisted_row

    def _fake_load_file_cache_metadata(
        *,
        blob_shas,
        dimensions: int,
        session: object | None = None,
    ):  # type: ignore[no-untyped-def]
        del session
        dims = int(dimensions)
        assert dims == 2
        found = {}
        if sha_a1 in blob_shas:
            found[sha_a1] = RepositoryStateEmbedder._VectorMeta(vector=_vec_for_sha(sha_a1))
        return found

    def _fake_embed_cache_misses(  # type: ignore[no-untyped-def]
        *,
        root: Path,
        commit_hash: str,
        repo_files,
        missing_blob_shas,
    ):
        del root, commit_hash, repo_files
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

    empty_embedding, empty_stats = embedder.embed_incremental(commit_hash=c2, repo_root=tmp_path)
    restored_embedding, restored_stats = embedder.embed_incremental(commit_hash=c3, repo_root=tmp_path)

    assert empty_embedding is None
    assert empty_stats.files_aggregated == 0
    assert persisted[c2].file_count == 0
    assert restored_embedding is not None
    assert restored_embedding.vector == pytest.approx(_vec_for_sha(sha_b3))
    assert restored_stats.files_aggregated == 1
