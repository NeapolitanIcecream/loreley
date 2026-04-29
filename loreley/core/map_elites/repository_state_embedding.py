"""Repo-state commit embeddings built from file-level embeddings.

This module implements the repo-state embedding design:

- Enumerate eligible files for a given commit hash using pinned ignore rules
  from `Settings` (pinned at scheduler startup from the configured root commit)
  plus basic MAP-Elites preprocessing filters.
- Reuse a file-level embedding cache keyed by git blob SHA.
- Only embed cache misses (new/modified files).
- Aggregate all file embeddings into a single commit vector via **uniform mean**
  (each eligible file contributes weight 1).
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
from typing import Iterable, Sequence, TypeVar, cast

from git import Repo
from git.exc import GitCommandError
from loguru import logger
import numpy as np
from sqlalchemy import event, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from loreley.config import Settings, get_settings
from loreley.db.base import session_scope
from loreley.db.models import MapElitesFileEmbeddingCache, MapElitesRepoStateAggregate
from .chunk import PreprocessedArtifact, chunk_preprocessed_files
from .code_embedding import CommitCodeEmbedding, embed_chunked_files
from .file_embedding_cache import FileEmbeddingCache, build_file_embedding_cache
from .preprocess import CodePreprocessor, PreprocessedFile
from .repository_files import RepositoryFile, build_pinned_ignore_spec, is_ignored_path, list_repository_files
from .vector_math import (
    apply_replacement_delta_in_place,
    accumulate_in_place,
    divide_vector,
)

log = logger.bind(module="map_elites.repository_state_embedding")

Vector = tuple[float, ...]
T = TypeVar("T")
_AGGREGATE_CACHE_LIMIT = 256
_FILE_METADATA_CACHE_LIMIT = 4096
_SESSION_AGGREGATE_CACHE_KEY = "map_elites.repo_state.aggregate_cache"


class RepoStateEmbeddingError(RuntimeError):
    """Raised when repo-state embedding cannot proceed."""

__all__ = [
    "RepoStateEmbeddingError",
    "RepoStateEmbeddingStats",
    "RepositoryStateEmbedder",
    "bootstrap_repository_state_aggregate",
    "embed_repository_state_incremental",
]


@dataclass(frozen=True, slots=True)
class RepoStateEmbeddingStats:
    commit_hash: str | None
    eligible_files: int
    files_embedded: int
    files_aggregated: int
    unique_blobs: int
    cache_hits: int
    cache_misses: int
    skipped_empty_after_preprocess: int
    skipped_failed_embedding: int
    source: str = "unknown"


@dataclass(frozen=True, slots=True)
class _IncrementalAggregateResult:
    aggregate: MapElitesRepoStateAggregate
    vector: Vector
    unique_blobs: int
    cache_hits: int
    cache_misses: int
    files_embedded: int
    skipped_empty_after_preprocess: int
    skipped_failed_embedding: int


@dataclass(frozen=True, slots=True)
class _IncrementalParent:
    repo: Repo
    parent_hash: str


@dataclass(frozen=True, slots=True)
class _ParentAggregateState:
    commit_hash: str
    sum_vector: np.ndarray
    file_count: int
    dimensions: int


@dataclass(frozen=True, slots=True)
class _SelectedDiffBlob:
    selected: bool
    repo_rel: Path | None
    blob_sha: str | None


@dataclass(frozen=True, slots=True)
class _IncrementalDiffSelection:
    entries: tuple["_DiffEntry", ...]
    old_shas: tuple[str, ...]
    new_shas: tuple[str, ...]
    repo_files_for_new_misses: tuple[RepositoryFile, ...]
    selector: "_DiffSelector"


@dataclass(frozen=True, slots=True)
class _IncrementalCacheMissResult:
    unique_new_shas: tuple[str, ...]
    cache_hits: int
    cache_misses: int
    files_embedded: int
    skipped_empty_after_preprocess: int
    skipped_failed_embedding: int


@dataclass(slots=True)
class _DiffSelector:
    repo_prefix: str | None
    ignore_spec: object | None
    preprocess_filter: CodePreprocessor
    blob_sizes: dict[str, int]
    max_bytes: int
    _cache: dict[tuple[str | None, str | None], _SelectedDiffBlob] = field(default_factory=dict)

    def select(self, path_str: str | None, sha: str | None) -> _SelectedDiffBlob:
        cache_key = (path_str, sha)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        if not path_str or _is_null_sha(sha):
            return self._remember(cache_key, _SelectedDiffBlob(False, None, None))

        git_path = path_str.strip().lstrip("/")
        if not git_path:
            return self._remember(cache_key, _SelectedDiffBlob(False, None, None))
        if self.ignore_spec and is_ignored_path(self.ignore_spec, git_path):
            return self._remember(cache_key, _SelectedDiffBlob(False, None, None))

        repo_rel = _git_path_to_repo_rel(git_path, repo_prefix=self.repo_prefix)
        if repo_rel is None:
            return self._remember(cache_key, _SelectedDiffBlob(False, None, None))
        if self.preprocess_filter.is_excluded(repo_rel):
            return self._remember(cache_key, _SelectedDiffBlob(False, None, None))
        if not self.preprocess_filter.is_code_file(repo_rel):
            return self._remember(cache_key, _SelectedDiffBlob(False, None, None))

        size = self.blob_sizes.get(str(sha))
        if size is None or size <= 0 or size > self.max_bytes:
            return self._remember(cache_key, _SelectedDiffBlob(False, None, None))
        return self._remember(cache_key, _SelectedDiffBlob(True, repo_rel, str(sha)))

    def _remember(
        self,
        cache_key: tuple[str | None, str | None],
        selection: _SelectedDiffBlob,
    ) -> _SelectedDiffBlob:
        self._cache[cache_key] = selection
        return selection


class RepositoryStateEmbedder:
    """Compute repo-state commit embeddings with a file-level cache."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        cache: FileEmbeddingCache | None = None,
        repo: Repo | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._repo = repo
        self.cache = cache or build_file_embedding_cache(
            settings=self.settings,
        )
        self._aggregate_cache: OrderedDict[str, MapElitesRepoStateAggregate] = OrderedDict()
        self._file_metadata_cache: OrderedDict[str, RepositoryStateEmbedder._VectorMeta] = OrderedDict()

    def embed_incremental(
        self,
        *,
        commit_hash: str,
        repo_root: Path | None = None,
        session: Session | None = None,
    ) -> tuple[CommitCodeEmbedding | None, RepoStateEmbeddingStats]:
        """Return a commit embedding derived from an existing aggregate or incremental diff."""

        canonical, _repo, root = self._resolve_repo_and_commit(commit_hash, repo_root=repo_root)

        session_cm = nullcontext(session) if session is not None else session_scope()
        with session_cm as active_session:
            aggregate = self._load_aggregate(
                commit_hash=canonical,
                repo_root=root,
                session=active_session,
            )
            if aggregate is not None:
                return self._embedding_from_aggregate(commit_hash=canonical, aggregate=aggregate)

            incremental = self._try_incremental_aggregate(
                commit_hash=canonical,
                repo_root=root,
                session=active_session,
            )
            if incremental is None:
                raise RepoStateEmbeddingError(
                    "Repo-state embedding requires an aggregate cache hit or a valid incremental derivation; "
                    f"no aggregate and no incremental path for commit {canonical}."
                )

        agg_row = incremental.aggregate
        vector = incremental.vector
        stats = RepoStateEmbeddingStats(
            commit_hash=canonical,
            eligible_files=int(agg_row.file_count),
            files_embedded=int(incremental.files_embedded),
            files_aggregated=int(agg_row.file_count),
            unique_blobs=int(incremental.unique_blobs),
            cache_hits=int(incremental.cache_hits),
            cache_misses=int(incremental.cache_misses),
            skipped_empty_after_preprocess=int(incremental.skipped_empty_after_preprocess),
            skipped_failed_embedding=int(incremental.skipped_failed_embedding),
            source="incremental",
        )
        if not vector:
            return None, stats
        embedding = CommitCodeEmbedding(
            files=(),
            vector=vector,
            model=self.cache.embedding_model,
            dimensions=len(vector),
        )
        log.info(
            "Repo-state aggregate incrementally updated for commit {} (files={} dims={})",
            canonical,
            agg_row.file_count,
            len(vector),
        )
        return embedding, stats

    def bootstrap_aggregate(
        self,
        *,
        commit_hash: str,
        repo_root: Path | None = None,
        session: Session | None = None,
    ) -> tuple[CommitCodeEmbedding | None, RepoStateEmbeddingStats]:
        """Ensure a commit aggregate exists, recomputing fully when missing."""

        canonical, repo, root = self._resolve_repo_and_commit(commit_hash, repo_root=repo_root)

        session_cm = nullcontext(session) if session is not None else session_scope()
        with session_cm as active_session:
            aggregate = self._load_aggregate(
                commit_hash=canonical,
                repo_root=root,
                session=active_session,
            )
            if aggregate is not None:
                return self._embedding_from_aggregate(commit_hash=canonical, aggregate=aggregate)

            return self._full_recompute_and_persist(
                commit_hash=canonical,
                repo_root=root,
                repo=repo,
                session=active_session,
            )

    def _resolve_repo_and_commit(
        self,
        commit_hash: str,
        *,
        repo_root: Path | None = None,
    ) -> tuple[str, Repo, Path]:
        requested_commit = str(commit_hash).strip()
        if not requested_commit:
            raise ValueError("Commit hash must be provided.")

        root = Path(repo_root or Path.cwd()).resolve()
        repo = self._repo
        if repo is None:
            try:
                repo = Repo(root, search_parent_directories=True)
            except Exception as exc:
                raise RepoStateEmbeddingError(
                    f"Failed to open git repository at {root}."
                ) from exc

        try:
            canonical = str(getattr(repo.commit(requested_commit), "hexsha", "") or "").strip()
        except Exception as exc:
            raise ValueError(f"Unknown commit {requested_commit!r}") from exc
        if not canonical:
            raise ValueError(f"Unknown commit {requested_commit!r}")

        self._repo = repo
        return canonical, repo, root

    def _embedding_from_aggregate(
        self,
        *,
        commit_hash: str,
        aggregate: MapElitesRepoStateAggregate,
    ) -> tuple[CommitCodeEmbedding | None, RepoStateEmbeddingStats]:
        file_count = int(aggregate.file_count or 0)
        if file_count < 0:
            raise RepoStateEmbeddingError(
                f"Repo-state aggregate has an invalid file count (commit={commit_hash})."
            )
        vector = divide_vector(
            aggregate.sum_vector or (),
            float(file_count),
        )
        stats = RepoStateEmbeddingStats(
            commit_hash=commit_hash,
            eligible_files=file_count,
            files_embedded=0,
            files_aggregated=file_count,
            unique_blobs=0,
            cache_hits=0,
            cache_misses=0,
            skipped_empty_after_preprocess=0,
            skipped_failed_embedding=0,
            source="aggregate_hit",
        )
        if not vector:
            return None, stats
        embedding = CommitCodeEmbedding(
            files=(),
            vector=vector,
            model=self.cache.embedding_model,
            dimensions=len(vector),
        )
        log.info(
            "Repo-state aggregate cache hit for commit {} (files={} dims={})",
            commit_hash,
            aggregate.file_count,
            len(vector),
        )
        return embedding, stats

    def _full_recompute_and_persist(
        self,
        *,
        commit_hash: str,
        repo_root: Path,
        repo: Repo,
        session: Session | None = None,
    ) -> tuple[CommitCodeEmbedding | None, RepoStateEmbeddingStats]:
        repo_files = list_repository_files(
            repo_root=repo_root,
            commit_hash=commit_hash,
            settings=self.settings,
            repo=repo,
        )
        if not repo_files:
            stats = RepoStateEmbeddingStats(
                commit_hash=commit_hash,
                eligible_files=0,
                files_embedded=0,
                files_aggregated=0,
                unique_blobs=0,
                cache_hits=0,
                cache_misses=0,
                skipped_empty_after_preprocess=0,
                skipped_failed_embedding=0,
                source="full_recompute",
            )
            return None, stats

        blob_shas = [entry.blob_sha for entry in repo_files if entry.blob_sha]
        blob_weights = Counter(blob_shas)
        unique_blob_shas = sorted(blob_weights.keys())
        cached = self.cache.get_many(unique_blob_shas)

        misses = [sha for sha in unique_blob_shas if sha not in cached]
        cache_hits = len(unique_blob_shas) - len(misses)

        vectors_for_misses, embedded_count, skipped_empty = self._embed_cache_misses(
            root=repo_root,
            commit_hash=commit_hash,
            repo_files=repo_files,
            missing_blob_shas=misses,
        )
        if vectors_for_misses:
            self.cache.put_many(vectors_for_misses)
            cached.update(vectors_for_misses)

        sum_array = np.zeros(0, dtype=np.float64)
        aggregated_count = 0
        skipped_failed = len(repo_files) - len(blob_shas)
        dims = 0
        for blob_sha in unique_blob_shas:
            weight = int(blob_weights.get(blob_sha, 0) or 0)
            if weight <= 0:
                continue

            vec = cached.get(blob_sha)
            if not vec:
                skipped_failed += weight
                continue

            vector_array = np.asarray(vec, dtype=np.float64)
            if vector_array.ndim != 1:
                raise ValueError("Embedding dimension mismatch during repo-state aggregation.")
            if vector_array.size <= 0:
                skipped_failed += weight
                continue

            if dims == 0:
                dims = int(vector_array.size)
                sum_array = np.zeros(dims, dtype=np.float64)
            elif int(vector_array.size) != dims:
                raise ValueError("Embedding dimension mismatch during repo-state aggregation.")

            sum_array += vector_array * float(weight)
            aggregated_count += weight

        if aggregated_count <= 0 or sum_array.size <= 0:
            commit_vector, sum_vector = (), ()
        else:
            sum_vector = tuple(float(value) for value in sum_array.tolist())
            commit_vector = divide_vector(sum_array, float(aggregated_count))
        stats = RepoStateEmbeddingStats(
            commit_hash=commit_hash,
            eligible_files=len(repo_files),
            files_embedded=embedded_count,
            files_aggregated=aggregated_count,
            unique_blobs=len(unique_blob_shas),
            cache_hits=cache_hits,
            cache_misses=len(misses),
            skipped_empty_after_preprocess=skipped_empty,
            skipped_failed_embedding=skipped_failed,
            source="full_recompute",
        )

        if not commit_vector:
            return None, stats

        embedding = CommitCodeEmbedding(
            files=(),  # keep lightweight; file-level vectors are cached externally
            vector=commit_vector,
            model=self.cache.embedding_model,
            dimensions=len(commit_vector),
        )
        log.info(
            "Repo-state embedding for commit {}: files={} blobs={} hits={} misses={} agg_files={} dims={}",
            commit_hash,
            stats.eligible_files,
            stats.unique_blobs,
            stats.cache_hits,
            stats.cache_misses,
            stats.files_aggregated,
            embedding.dimensions,
        )

        self._persist_aggregate(
            commit_hash=commit_hash,
            repo_root=repo_root,
            sum_vector=sum_vector,
            file_count=aggregated_count,
            session=session,
        )
        return embedding, stats

    def load_aggregate(
        self,
        *,
        commit_hash: str,
        repo_root: Path,
    ) -> MapElitesRepoStateAggregate | None:
        """Load a persisted repo-state aggregate when available.

        This is a thin wrapper around the internal aggregate lookup. It is
        exposed so callers and tests can access the persisted aggregate without
        relying on private methods.
        """

        return self._load_aggregate(commit_hash=commit_hash, repo_root=repo_root)

    def _load_aggregate(
        self,
        *,
        commit_hash: str,
        repo_root: Path,
        session: Session | None = None,
    ) -> MapElitesRepoStateAggregate | None:
        requested_dims = int(getattr(self.cache, "requested_dimensions", 0))
        if requested_dims <= 0:
            raise RepoStateEmbeddingError(
                "Repo-state embedding requires a positive embedding dimension."
            )
        try:
            if session is not None:
                cached = self._cached_session_aggregate(session, commit_hash)
                if cached is not None:
                    return self._validate_aggregate_row(
                        row=cached,
                        commit_hash=commit_hash,
                        requested_dims=requested_dims,
                    )
            else:
                cached = self._cached_aggregate(commit_hash)
                if cached is not None:
                    return self._validate_aggregate_row(
                        row=cached,
                        commit_hash=commit_hash,
                        requested_dims=requested_dims,
                    )
            if session is None:
                with session_scope() as owned_session:
                    stmt = select(MapElitesRepoStateAggregate).where(
                        MapElitesRepoStateAggregate.commit_hash == str(commit_hash),
                    )
                    row = owned_session.execute(stmt).scalar_one_or_none()
            else:
                stmt = select(MapElitesRepoStateAggregate).where(
                    MapElitesRepoStateAggregate.commit_hash == str(commit_hash),
                )
                row = session.execute(stmt).scalar_one_or_none()
        except Exception as exc:  # pragma: no cover - DB failure handling
            raise RepoStateEmbeddingError(
                f"Repo-state aggregate read failed for {commit_hash}: {exc}"
            ) from exc

        validated = self._validate_aggregate_row(
            row=row,
            commit_hash=commit_hash,
            requested_dims=requested_dims,
        )
        if validated is not None:
            if session is not None:
                self._remember_session_aggregate(session, validated)
            else:
                self._remember_aggregate(validated)
        return validated

    def _persist_aggregate(
        self,
        *,
        commit_hash: str,
        repo_root: Path,
        sum_vector: Vector,
        file_count: int,
        session: Session | None = None,
    ) -> MapElitesRepoStateAggregate:
        requested_dims = int(getattr(self.cache, "requested_dimensions", 0) or 0)
        if requested_dims <= 0:
            raise RepoStateEmbeddingError(
                "Repo-state embedding requires a positive embedding dimension."
            )
        normalized_sum = tuple(float(v) for v in sum_vector)
        if file_count < 0:
            raise RepoStateEmbeddingError(
                f"Repo-state aggregate persist requires file_count >= 0 (commit={commit_hash})."
            )
        if not normalized_sum and file_count == 0:
            normalized_sum = tuple(0.0 for _ in range(requested_dims))
        if len(normalized_sum) != requested_dims:
            raise RepoStateEmbeddingError(
                "Repo-state aggregate has unexpected dimensions; "
                f"expected {requested_dims} got {len(normalized_sum)} (commit={commit_hash})."
            )
        values = {
            "commit_hash": str(commit_hash),
            "file_count": int(file_count),
            "sum_vector": [float(v) for v in normalized_sum],
        }
        insert_stmt = pg_insert(MapElitesRepoStateAggregate).values(values)
        stmt = insert_stmt.on_conflict_do_update(
            index_elements=["commit_hash"],
            set_={
                "file_count": insert_stmt.excluded.file_count,
                "sum_vector": insert_stmt.excluded.sum_vector,
            },
        )
        stmt = stmt.returning(MapElitesRepoStateAggregate)

        try:
            if session is None:
                with session_scope() as owned_session:
                    row = owned_session.execute(stmt).scalar_one()
            else:
                row = session.execute(stmt).scalar_one()
        except Exception as exc:  # pragma: no cover - DB failure handling
            raise RepoStateEmbeddingError(
                f"Repo-state aggregate persist failed for {commit_hash}: {exc}"
            ) from exc
        validated = self._validate_aggregate_row(
            row=row,
            commit_hash=commit_hash,
            requested_dims=requested_dims,
        )
        if session is not None:
            self._remember_session_aggregate(session, validated)
        else:
            self._remember_aggregate(validated)
        return validated

    @dataclass(frozen=True, slots=True)
    class _VectorMeta:
        vector: Vector

    def _load_file_cache_metadata(
        self,
        *,
        blob_shas: Sequence[str],
        dimensions: int,
        session: Session | None = None,
    ) -> dict[str, "RepositoryStateEmbedder._VectorMeta"]:
        if not blob_shas:
            return {}
        dims = int(dimensions or 0)
        if dims <= 0:
            raise RepoStateEmbeddingError(
                "Repo-state file cache metadata requires positive dimensions."
            )
        cleaned = [str(sha).strip() for sha in blob_shas if str(sha).strip()]
        unique = sorted(set(cleaned))
        if not unique:
            return {}

        found: dict[str, RepositoryStateEmbedder._VectorMeta] = {}
        missing = self._load_missing_blob_metadata_from_memory(unique, found=found)
        if not missing:
            return found
        try:
            session_cm = nullcontext(session) if session is not None else session_scope()
            with session_cm as active_session:
                self._load_file_cache_metadata_from_db(
                    blob_shas=missing,
                    dimensions=dims,
                    session=active_session,
                    found=found,
                )
        except RepoStateEmbeddingError:
            raise
        except Exception as exc:  # pragma: no cover - DB failure handling
            raise RepoStateEmbeddingError(f"Repo-state file cache metadata read failed: {exc}") from exc

        return found

    def _load_file_cache_metadata_from_db(
        self,
        *,
        blob_shas: Sequence[str],
        dimensions: int,
        session: Session,
        found: dict[str, "RepositoryStateEmbedder._VectorMeta"],
    ) -> None:
        for batch in _batched(blob_shas, 500):
            stmt = self._file_cache_metadata_query(batch)
            for sha, vec, model, stored_dims in session.execute(stmt).all():
                meta = self._validate_file_cache_metadata_row(
                    blob_sha=sha,
                    vector=vec,
                    embedding_model=model,
                    stored_dimensions=stored_dims,
                    expected_dimensions=dimensions,
                )
                key = str(sha)
                found[key] = meta
                self._remember_file_metadata(key, meta)

    @staticmethod
    def _file_cache_metadata_query(blob_shas: Sequence[str]):
        return (
            select(
                MapElitesFileEmbeddingCache.blob_sha,
                MapElitesFileEmbeddingCache.vector,
                MapElitesFileEmbeddingCache.embedding_model,
                MapElitesFileEmbeddingCache.dimensions,
            )
            .where(
                MapElitesFileEmbeddingCache.blob_sha.in_(blob_shas),
            )
        )

    def _validate_file_cache_metadata_row(
        self,
        *,
        blob_sha: object,
        vector: object,
        embedding_model: object,
        stored_dimensions: object,
        expected_dimensions: int,
    ) -> "RepositoryStateEmbedder._VectorMeta":
        if str(embedding_model or "") != str(self.cache.embedding_model):
            raise RepoStateEmbeddingError(
                "File embedding cache entry has an unexpected embedding model; "
                "reset the DB (dev). "
                f"(blob_sha={blob_sha} expected_model={self.cache.embedding_model!r} "
                f"got_model={embedding_model!r})"
            )
        if int(stored_dimensions or 0) != expected_dimensions:
            raise RepoStateEmbeddingError(
                "File embedding cache entry has unexpected dimensions; reset the DB (dev). "
                f"(blob_sha={blob_sha} expected_dims={expected_dimensions} got_dims={stored_dimensions!r})"
            )
        normalized = tuple(float(v) for v in (vector or ()))
        if not normalized:
            raise RepoStateEmbeddingError(
                "File embedding cache contains an empty vector; reset the DB (dev). "
                f"(blob_sha={blob_sha} dims={expected_dimensions})"
            )
        if len(normalized) != expected_dimensions:
            raise RepoStateEmbeddingError(
                "File embedding cache vector has unexpected dimensions; reset the DB (dev). "
                f"(blob_sha={blob_sha} expected_dims={expected_dimensions} got_dims={len(normalized)})"
            )
        return RepositoryStateEmbedder._VectorMeta(vector=normalized)

    def _try_incremental_aggregate(
        self,
        *,
        commit_hash: str,
        repo_root: Path,
        session: Session | None = None,
    ) -> _IncrementalAggregateResult | None:
        parent = self._resolve_incremental_parent(commit_hash=commit_hash, repo_root=repo_root)
        if parent is None:
            return None

        parent_state = self._load_parent_aggregate_state(
            parent_hash=parent.parent_hash,
            repo_root=repo_root,
            session=session,
        )
        if parent_state is None:
            return None

        diffs = self._load_incremental_diff_entries(
            repo=parent.repo,
            parent_hash=parent.parent_hash,
            commit_hash=commit_hash,
        )
        if diffs is None:
            return None
        if not diffs:
            return self._persist_incremental_result(
                commit_hash=commit_hash,
                repo_root=repo_root,
                sum_vector=parent_state.sum_vector,
                file_count=parent_state.file_count,
                session=session,
                unique_blobs=0,
                cache_hits=0,
                cache_misses=0,
                files_embedded=0,
                skipped_empty_after_preprocess=0,
                skipped_failed_embedding=0,
            )

        diff_selection = self._select_incremental_diff_candidates(
            repo=parent.repo,
            repo_root=repo_root,
            diffs=diffs,
        )
        metadata = self._load_file_cache_metadata(
            blob_shas=sorted(set(diff_selection.old_shas + diff_selection.new_shas)),
            dimensions=parent_state.dimensions,
            session=session,
        )
        miss_result = self._embed_incremental_cache_misses(
            commit_hash=commit_hash,
            repo_root=repo_root,
            diff_selection=diff_selection,
            metadata=metadata,
            dimensions=parent_state.dimensions,
        )
        file_count = self._apply_incremental_delta(
            diff_selection=diff_selection,
            metadata=metadata,
            sum_vector=parent_state.sum_vector,
            file_count=parent_state.file_count,
        )
        return self._persist_incremental_result(
            commit_hash=commit_hash,
            repo_root=repo_root,
            sum_vector=parent_state.sum_vector,
            file_count=file_count,
            session=session,
            unique_blobs=len(miss_result.unique_new_shas),
            cache_hits=miss_result.cache_hits,
            cache_misses=miss_result.cache_misses,
            files_embedded=miss_result.files_embedded,
            skipped_empty_after_preprocess=miss_result.skipped_empty_after_preprocess,
            skipped_failed_embedding=miss_result.skipped_failed_embedding,
        )

    def _resolve_incremental_parent(
        self,
        *,
        commit_hash: str,
        repo_root: Path,
    ) -> _IncrementalParent | None:
        repo = self._repo
        if repo is None:
            try:
                repo = Repo(repo_root, search_parent_directories=True)
            except Exception:
                return None

        try:
            commit = repo.commit(commit_hash)
        except Exception:
            return None

        parents = list(getattr(commit, "parents", []) or [])
        if len(parents) != 1:
            return None
        parent = parents[0]
        parent_hash = str(getattr(parent, "hexsha", "") or "").strip()
        if not parent_hash:
            return None
        return _IncrementalParent(repo=repo, parent_hash=parent_hash)

    def _load_parent_aggregate_state(
        self,
        *,
        parent_hash: str,
        repo_root: Path,
        session: Session | None,
    ) -> _ParentAggregateState | None:
        parent_agg = self._load_aggregate(
            commit_hash=parent_hash,
            repo_root=repo_root,
            session=session,
        )
        if parent_agg is None:
            return None
        if int(parent_agg.file_count or 0) < 0:
            raise RepoStateEmbeddingError(
                f"Parent aggregate has an invalid file count (commit={parent_hash})."
            )

        sum_vec = np.asarray(parent_agg.sum_vector or (), dtype=np.float64)
        if sum_vec.ndim != 1:
            raise RepoStateEmbeddingError(
                f"Parent aggregate has an invalid vector shape (commit={parent_hash})."
            )
        dims = int(sum_vec.size)
        if dims <= 0:
            raise RepoStateEmbeddingError(
                f"Parent aggregate has no vector data (commit={parent_hash})."
            )
        requested_dims = getattr(self.cache, "requested_dimensions", 0)
        if dims != int(requested_dims or 0):
            raise RepoStateEmbeddingError(
                "Parent aggregate dimensions do not match the embedding cache; "
                f"expected {requested_dims} got {dims} "
                f"(commit={parent_hash})."
            )
        return _ParentAggregateState(
            commit_hash=parent_hash,
            sum_vector=sum_vec,
            file_count=int(parent_agg.file_count),
            dimensions=dims,
        )

    @staticmethod
    def _load_incremental_diff_entries(
        *,
        repo: Repo,
        parent_hash: str,
        commit_hash: str,
    ) -> tuple["_DiffEntry", ...] | None:
        try:
            raw = repo.git.diff_tree(
                "-r",
                "--no-commit-id",
                "--raw",
                "-z",
                "-M",
                parent_hash,
                commit_hash,
            )
        except GitCommandError:
            return None
        return tuple(_parse_diff_tree_raw_z(raw))

    def _select_incremental_diff_candidates(
        self,
        *,
        repo: Repo,
        repo_root: Path,
        diffs: Sequence["_DiffEntry"],
    ) -> _IncrementalDiffSelection:
        pinned_ignore = str(getattr(self.settings, "mapelites_repo_state_ignore_text", "") or "").strip()
        repo_prefix = _resolve_git_prefix(repo, repo_root)
        ignore_spec = build_pinned_ignore_spec(pinned_ignore)
        preprocess_filter = CodePreprocessor(
            repo_root=repo_root,
            settings=self.settings,
            commit_hash=None,
        )
        max_bytes = max(int(self.settings.mapelites_preprocess_max_file_size_kb), 1) * 1024

        candidate_blob_shas = sorted(
            {
                str(sha).strip()
                for entry in diffs
                for sha in (entry.old_sha, entry.new_sha)
                if str(sha or "").strip() and not _is_null_sha(sha)
            }
        )
        blob_sizes = _load_blob_sizes(repo, candidate_blob_shas)
        selector = _DiffSelector(
            repo_prefix=repo_prefix,
            ignore_spec=ignore_spec,
            preprocess_filter=preprocess_filter,
            blob_sizes=blob_sizes,
            max_bytes=max_bytes,
        )

        old_shas: list[str] = []
        new_shas: list[str] = []
        repo_files_for_new_misses: list[RepositoryFile] = []

        for entry in diffs:
            old_blob = selector.select(entry.old_path, entry.old_sha)
            new_blob = selector.select(entry.new_path, entry.new_sha)
            if old_blob.selected and old_blob.blob_sha:
                old_shas.append(old_blob.blob_sha)
            if new_blob.selected and new_blob.blob_sha:
                new_shas.append(new_blob.blob_sha)
            if new_blob.selected and new_blob.blob_sha and new_blob.repo_rel:
                repo_files_for_new_misses.append(
                    RepositoryFile(path=new_blob.repo_rel, blob_sha=new_blob.blob_sha, size_bytes=0)
                )
        return _IncrementalDiffSelection(
            entries=tuple(diffs),
            old_shas=tuple(old_shas),
            new_shas=tuple(new_shas),
            repo_files_for_new_misses=tuple(repo_files_for_new_misses),
            selector=selector,
        )

    def _embed_incremental_cache_misses(
        self,
        *,
        commit_hash: str,
        repo_root: Path,
        diff_selection: _IncrementalDiffSelection,
        metadata: dict[str, "RepositoryStateEmbedder._VectorMeta"],
        dimensions: int,
    ) -> _IncrementalCacheMissResult:
        unique_new_shas = tuple(sorted(set(diff_selection.new_shas)))
        missing_new = [sha for sha in unique_new_shas if sha not in metadata]
        files_embedded = 0
        skipped_empty_after_preprocess = 0
        if missing_new:
            vectors_for_misses, embedded_count, skipped_empty = self._embed_cache_misses(
                root=repo_root,
                commit_hash=commit_hash,
                repo_files=diff_selection.repo_files_for_new_misses,
                missing_blob_shas=missing_new,
            )
            files_embedded = int(embedded_count)
            skipped_empty_after_preprocess = int(skipped_empty)
            if vectors_for_misses:
                self.cache.put_many(vectors_for_misses)
                for sha, vec in vectors_for_misses.items():
                    if vec and len(vec) == dimensions:
                        meta = RepositoryStateEmbedder._VectorMeta(vector=vec)
                        metadata[sha] = meta
                        self._remember_file_metadata(sha, meta)

        cache_misses = len(missing_new)
        cache_hits = max(len(unique_new_shas) - cache_misses, 0)
        skipped_failed_embedding = max(
            cache_misses - files_embedded - skipped_empty_after_preprocess,
            0,
        )
        return _IncrementalCacheMissResult(
            unique_new_shas=unique_new_shas,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            files_embedded=files_embedded,
            skipped_empty_after_preprocess=skipped_empty_after_preprocess,
            skipped_failed_embedding=skipped_failed_embedding,
        )

    @staticmethod
    def _apply_incremental_delta(
        *,
        diff_selection: _IncrementalDiffSelection,
        metadata: dict[str, "RepositoryStateEmbedder._VectorMeta"],
        sum_vector: np.ndarray,
        file_count: int,
    ) -> int:
        for entry in diff_selection.entries:
            old_blob = diff_selection.selector.select(entry.old_path, entry.old_sha)
            new_blob = diff_selection.selector.select(entry.new_path, entry.new_sha)
            old_meta: RepositoryStateEmbedder._VectorMeta | None = None
            if old_blob.selected and old_blob.blob_sha:
                old_meta = metadata.get(old_blob.blob_sha)
            new_meta: RepositoryStateEmbedder._VectorMeta | None = None
            if new_blob.selected and new_blob.blob_sha:
                new_meta = metadata.get(new_blob.blob_sha)

            old_included = bool(old_meta is not None and old_meta.vector)
            new_included = bool(new_meta is not None and new_meta.vector)

            if old_included and not new_included:
                if old_meta is None:  # pragma: no cover - type narrowing guard
                    continue
                accumulate_in_place(sum_vector, old_meta.vector, subtract=True)
                file_count -= 1
                continue
            if not old_included and new_included:
                if new_meta is None:  # pragma: no cover - type narrowing guard
                    continue
                accumulate_in_place(sum_vector, new_meta.vector)
                file_count += 1
                continue
            if old_included and new_included:
                if old_meta is None or new_meta is None:  # pragma: no cover - type narrowing guard
                    continue
                apply_replacement_delta_in_place(sum_vector, new_meta.vector, old_meta.vector)
        return file_count

    def _persist_incremental_result(
        self,
        *,
        commit_hash: str,
        repo_root: Path,
        sum_vector: np.ndarray,
        file_count: int,
        session: Session | None,
        unique_blobs: int,
        cache_hits: int,
        cache_misses: int,
        files_embedded: int,
        skipped_empty_after_preprocess: int,
        skipped_failed_embedding: int,
    ) -> _IncrementalAggregateResult:
        immutable_sum = tuple(float(v) for v in sum_vector.tolist())
        if file_count < 0:
            raise RepoStateEmbeddingError(
                f"Incremental repo-state aggregate underflowed file count (commit={commit_hash})."
            )
        vector = divide_vector(sum_vector, float(file_count))
        persisted = self._persist_aggregate(
            commit_hash=commit_hash,
            repo_root=repo_root,
            sum_vector=immutable_sum,
            file_count=file_count,
            session=session,
        )
        return _IncrementalAggregateResult(
            aggregate=persisted,
            vector=vector,
            unique_blobs=unique_blobs,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            files_embedded=files_embedded,
            skipped_empty_after_preprocess=skipped_empty_after_preprocess,
            skipped_failed_embedding=skipped_failed_embedding,
        )

    def _validate_aggregate_row(
        self,
        *,
        row: MapElitesRepoStateAggregate | None,
        commit_hash: str,
        requested_dims: int,
    ) -> MapElitesRepoStateAggregate | None:
        if row is None:
            return None
        if int(row.file_count or 0) < 0:
            raise RepoStateEmbeddingError(
                f"Repo-state aggregate has an invalid file count (commit={commit_hash})."
            )
        if not row.sum_vector:
            raise RepoStateEmbeddingError(
                f"Repo-state aggregate sum vector is missing (commit={commit_hash})."
            )
        if len(row.sum_vector) != requested_dims:
            raise RepoStateEmbeddingError(
                "Repo-state aggregate has unexpected dimensions; "
                f"expected {requested_dims} got {len(row.sum_vector)} (commit={commit_hash})."
            )
        return row

    def _cached_aggregate(self, commit_hash: str) -> MapElitesRepoStateAggregate | None:
        key = str(commit_hash).strip()
        if not key:
            return None
        row = self._aggregate_cache.get(key)
        if row is None:
            return None
        self._aggregate_cache.move_to_end(key)
        return row

    def _remember_aggregate(self, row: MapElitesRepoStateAggregate) -> None:
        key = str(getattr(row, "commit_hash", "") or "").strip()
        if not key:
            return
        self._aggregate_cache[key] = row
        self._aggregate_cache.move_to_end(key)
        while len(self._aggregate_cache) > _AGGREGATE_CACHE_LIMIT:
            self._aggregate_cache.popitem(last=False)

    def _session_aggregate_bucket(
        self,
        session: Session,
        *,
        create: bool,
    ) -> OrderedDict[str, MapElitesRepoStateAggregate] | None:
        info = getattr(session, "info", None)
        if not isinstance(info, dict):
            return None
        per_session = info.get(_SESSION_AGGREGATE_CACHE_KEY)
        if per_session is None:
            if not create:
                return None
            per_session = {}
            info[_SESSION_AGGREGATE_CACHE_KEY] = per_session
        if not isinstance(per_session, dict):
            return None
        bucket = per_session.get(self)
        if bucket is None:
            if not create:
                return None
            bucket = OrderedDict()
            per_session[self] = bucket
        if not isinstance(bucket, OrderedDict):
            return None
        return bucket

    def _cached_session_aggregate(
        self,
        session: Session,
        commit_hash: str,
    ) -> MapElitesRepoStateAggregate | None:
        bucket = self._session_aggregate_bucket(session, create=False)
        if bucket is None:
            return None
        key = str(commit_hash).strip()
        if not key:
            return None
        row = bucket.get(key)
        if row is None:
            return None
        bucket.move_to_end(key)
        return row

    def _remember_session_aggregate(
        self,
        session: Session,
        row: MapElitesRepoStateAggregate,
    ) -> None:
        bucket = self._session_aggregate_bucket(session, create=True)
        if bucket is None:
            return
        key = str(getattr(row, "commit_hash", "") or "").strip()
        if not key:
            return
        bucket[key] = row
        bucket.move_to_end(key)
        while len(bucket) > _AGGREGATE_CACHE_LIMIT:
            bucket.popitem(last=False)

    def _load_missing_blob_metadata_from_memory(
        self,
        blob_shas: Sequence[str],
        *,
        found: dict[str, "RepositoryStateEmbedder._VectorMeta"],
    ) -> list[str]:
        missing: list[str] = []
        for sha in blob_shas:
            key = str(sha).strip()
            if not key:
                continue
            cached = self._file_metadata_cache.get(key)
            if cached is None:
                missing.append(key)
                continue
            self._file_metadata_cache.move_to_end(key)
            found[key] = cached
        return missing

    def _remember_file_metadata(
        self,
        blob_sha: str,
        meta: "RepositoryStateEmbedder._VectorMeta",
    ) -> None:
        key = str(blob_sha).strip()
        if not key:
            return
        self._file_metadata_cache[key] = meta
        self._file_metadata_cache.move_to_end(key)
        while len(self._file_metadata_cache) > _FILE_METADATA_CACHE_LIMIT:
            self._file_metadata_cache.popitem(last=False)

    def _embed_cache_misses(
        self,
        *,
        root: Path,
        commit_hash: str,
        repo_files: Sequence[RepositoryFile],
        missing_blob_shas: Sequence[str],
    ) -> tuple[dict[str, Vector], int, int]:
        """Embed missing blobs and return (blob_sha->vector, embedded_count, skipped_empty)."""

        if not missing_blob_shas:
            return {}, 0, 0

        missing_set = set(str(sha).strip() for sha in missing_blob_shas if str(sha).strip())

        # Pick one representative path per missing blob.
        wanted: dict[str, RepositoryFile] = {}
        for entry in repo_files:
            if entry.blob_sha in missing_set and entry.blob_sha not in wanted:
                wanted[entry.blob_sha] = entry
        if not wanted:
            return {}, 0, 0

        preprocessor = CodePreprocessor(
            repo_root=root,
            settings=self.settings,
            commit_hash=commit_hash,
            repo=self._repo,
        )

        wanted_items = sorted(
            wanted.items(),
            key=lambda item: item[1].path.as_posix(),
        )

        vectors: dict[str, Vector] = {}
        skipped_empty_total = 0

        # Embed in batches to keep memory and request payloads bounded.
        batch_size = 64
        for batch in _batched(wanted_items, batch_size):
            preprocessed: list[PreprocessedFile] = []
            blob_for_path: dict[Path, str] = {}
            skipped_empty = 0
            blob_texts: dict[str, str] = {}

            if self._repo is not None:
                blob_texts = _load_blob_texts(
                    self._repo,
                    [blob_sha for blob_sha, _entry in batch],
                )

            for blob_sha, entry in batch:
                raw = blob_texts.get(blob_sha)
                if raw is None:
                    raw = preprocessor.load_text(entry.path)
                if raw is None:
                    continue
                cleaned = preprocessor.cleanup_text(raw)
                if not cleaned.strip():
                    skipped_empty += 1
                    continue
                preprocessed.append(
                    PreprocessedFile(
                        path=entry.path,
                        change_count=1,  # uniform weighting per file
                        content=cleaned,
                    )
                )
                blob_for_path[entry.path] = blob_sha

            skipped_empty_total += skipped_empty
            if not preprocessed:
                continue

            chunked = chunk_preprocessed_files(
                cast(Sequence[PreprocessedArtifact], preprocessed),
                settings=self.settings,
            )
            log.debug(
                "Embedding {} cache-miss files for commit {}",
                len(chunked),
                commit_hash,
            )
            commit_embedding = embed_chunked_files(chunked, settings=self.settings)
            if not commit_embedding:
                continue

            for file_embedding in commit_embedding.files:
                blob_sha = blob_for_path.get(file_embedding.file.path)
                if not blob_sha:
                    continue
                vec = tuple(float(v) for v in file_embedding.vector)
                if vec:
                    vectors[blob_sha] = vec

        return vectors, len(vectors), skipped_empty_total


def embed_repository_state_incremental(
    *,
    commit_hash: str,
    repo_root: Path | None = None,
    settings: Settings | None = None,
    cache: FileEmbeddingCache | None = None,
    repo: Repo | None = None,
    embedder: RepositoryStateEmbedder | None = None,
    session: Session | None = None,
) -> tuple[CommitCodeEmbedding | None, RepoStateEmbeddingStats]:
    """Return an incremental-only repo-state embedding for a commit."""

    effective_embedder = embedder or RepositoryStateEmbedder(
        settings=settings,
        cache=cache,
        repo=repo,
    )
    return effective_embedder.embed_incremental(
        commit_hash=commit_hash,
        repo_root=repo_root,
        session=session,
    )


def bootstrap_repository_state_aggregate(
    *,
    commit_hash: str,
    repo_root: Path | None = None,
    settings: Settings | None = None,
    cache: FileEmbeddingCache | None = None,
    repo: Repo | None = None,
    session: Session | None = None,
) -> tuple[CommitCodeEmbedding | None, RepoStateEmbeddingStats]:
    """Ensure the repo-state aggregate exists, recomputing fully when missing."""

    embedder = RepositoryStateEmbedder(
        settings=settings,
        cache=cache,
        repo=repo,
    )
    return embedder.bootstrap_aggregate(
        commit_hash=commit_hash,
        repo_root=repo_root,
        session=session,
    )


@event.listens_for(Session, "after_commit")
def _promote_session_aggregate_cache(session: Session) -> None:
    info = getattr(session, "info", None)
    if not isinstance(info, dict):
        return
    cached = info.pop(_SESSION_AGGREGATE_CACHE_KEY, None)
    if not isinstance(cached, dict):
        return
    for embedder, bucket in cached.items():
        if not isinstance(embedder, RepositoryStateEmbedder):
            continue
        if not isinstance(bucket, OrderedDict):
            continue
        for row in bucket.values():
            embedder._remember_aggregate(row)


@event.listens_for(Session, "after_soft_rollback")
def _clear_session_aggregate_cache(session: Session, _previous_transaction: object) -> None:
    info = getattr(session, "info", None)
    if isinstance(info, dict):
        info.pop(_SESSION_AGGREGATE_CACHE_KEY, None)


@dataclass(frozen=True, slots=True)
class _DiffEntry:
    status: str
    old_path: str | None
    new_path: str | None
    old_sha: str | None
    new_sha: str | None


def _parse_diff_tree_raw_z(output: str) -> list[_DiffEntry]:
    parts = output.split("\0")
    entries: list[_DiffEntry] = []
    i = 0
    while i < len(parts):
        header = parts[i]
        if not header:
            i += 1
            continue
        header = header.strip("\n")
        if not header.startswith(":"):
            i += 1
            continue
        tokens = header[1:].split()
        if len(tokens) < 5:
            i += 1
            continue
        old_sha = tokens[2].strip() or None
        new_sha = tokens[3].strip() or None
        status = tokens[4].strip()
        status_letter = status[:1]
        i += 1
        old_path = None
        new_path = None
        if i < len(parts):
            old_path = parts[i] or None
            i += 1
        if status_letter in {"R", "C"}:
            if i < len(parts):
                new_path = parts[i] or None
                i += 1
        else:
            new_path = old_path
        entries.append(
            _DiffEntry(
                status=status_letter,
                old_path=old_path,
                new_path=new_path,
                old_sha=old_sha,
                new_sha=new_sha,
            )
        )
    return entries


def _is_null_sha(sha: str | None) -> bool:
    if not sha:
        return True
    value = sha.strip()
    return not value or set(value) == {"0"}


def _resolve_git_prefix(repo: Repo, repo_root: Path) -> str | None:
    root_dir = getattr(repo, "working_tree_dir", None)
    if not root_dir:
        return None
    git_root = Path(root_dir).resolve()
    try:
        prefix = Path(repo_root).resolve().relative_to(git_root)
    except ValueError:
        return None
    if str(prefix) == ".":
        return None
    return prefix.as_posix().strip("/") or None


def _git_path_to_repo_rel(git_path: str, *, repo_prefix: str | None) -> Path | None:
    raw = (git_path or "").strip().lstrip("/")
    if not raw:
        return None
    if repo_prefix:
        prefix = repo_prefix.rstrip("/") + "/"
        if not raw.startswith(prefix):
            return None
        raw = raw[len(prefix) :]
    if not raw:
        return None
    return Path(raw)


def _blob_size_bytes(repo: Repo, blob_sha: str) -> int | None:
    if _is_null_sha(blob_sha):
        return None
    try:
        size_str = repo.git.cat_file("-s", str(blob_sha))
        return int(size_str.strip())
    except Exception:
        return None


def _load_blob_sizes(repo: Repo, blob_shas: Sequence[str]) -> dict[str, int]:
    cleaned = sorted(
        {
            str(sha).strip()
            for sha in blob_shas
            if str(sha or "").strip() and not _is_null_sha(sha)
        }
    )
    if not cleaned:
        return {}

    git_root = getattr(repo, "working_tree_dir", None)
    if not git_root:
        return {sha: size for sha in cleaned if (size := _blob_size_bytes(repo, sha)) is not None}

    proc = subprocess.run(
        [
            "git",
            "cat-file",
            "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        ],
        cwd=str(git_root),
        input="".join(f"{sha}\n" for sha in cleaned).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return {sha: size for sha in cleaned if (size := _blob_size_bytes(repo, sha)) is not None}

    sizes: dict[str, int] = {}
    for raw_line in (proc.stdout or b"").splitlines():
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            continue
        sha, obj_type, size_text = parts
        if obj_type != "blob":
            continue
        try:
            sizes[sha] = int(size_text)
        except ValueError:
            continue
    return sizes


def _load_blob_texts(repo: Repo, blob_shas: Sequence[str]) -> dict[str, str]:
    cleaned = [
        str(sha).strip()
        for sha in blob_shas
        if str(sha or "").strip() and not _is_null_sha(sha)
    ]
    if not cleaned:
        return {}

    git_root = getattr(repo, "working_tree_dir", None)
    if not git_root:
        return {}

    proc = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=str(git_root),
        input="".join(f"{sha}\n" for sha in cleaned).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return {}

    payload = proc.stdout or b""
    cursor = 0
    texts: dict[str, str] = {}
    while cursor < len(payload):
        line_end = payload.find(b"\n", cursor)
        if line_end < 0:
            break
        header = payload[cursor:line_end].decode("utf-8", errors="replace").strip()
        cursor = line_end + 1
        if not header:
            continue
        if header.endswith(" missing"):
            continue
        parts = header.split()
        if len(parts) != 3:
            break
        sha, obj_type, size_text = parts
        try:
            size = int(size_text)
        except ValueError:
            break
        content_end = cursor + size
        content = payload[cursor:content_end]
        cursor = min(content_end + 1, len(payload))
        if obj_type != "blob":
            continue
        texts[sha] = content.decode("utf-8", errors="ignore")
    return texts


def _batched(items: Sequence[T], batch_size: int) -> Iterable[Sequence[T]]:
    step = max(1, int(batch_size))
    for start in range(0, len(items), step):
        yield items[start : start + step]
