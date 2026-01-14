"""Repo-state commit embeddings built from file-level embeddings.

This module implements the repo-state embedding design:

- Enumerate eligible files for a given commit hash using pinned ignore rules
  from the experiment snapshot plus basic MAP-Elites preprocessing filters.
- Reuse a file-level embedding cache keyed by git blob SHA.
- Only embed cache misses (new/modified files).
- Aggregate all file embeddings into a single commit vector via **uniform mean**
  (each eligible file contributes weight 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence, cast
from uuid import UUID

from git import Repo
from git.exc import GitCommandError
from loguru import logger
from sqlalchemy import select

from loreley.config import Settings, get_settings
from loreley.db.base import session_scope
from loreley.db.models import MapElitesFileEmbeddingCache, MapElitesRepoStateAggregate
from .chunk import PreprocessedArtifact, chunk_preprocessed_files
from .code_embedding import CommitCodeEmbedding, embed_chunked_files
from .file_embedding_cache import DatabaseFileEmbeddingCache, FileEmbeddingCache, build_file_embedding_cache
from .preprocess import CodePreprocessor, PreprocessedFile
from .repository_files import (
    ROOT_IGNORE_FILES,
    RepositoryFile,
    build_pinned_ignore_spec,
    is_ignored_path,
    list_repository_files,
)

log = logger.bind(module="map_elites.repository_state_embedding")

Vector = tuple[float, ...]
RepoStateEmbeddingMode = Literal["auto", "incremental_only"]


class RepoStateEmbeddingError(RuntimeError):
    """Raised when repo-state embedding cannot proceed under the requested mode."""

__all__ = [
    "RepoStateEmbeddingError",
    "RepoStateEmbeddingStats",
    "RepositoryStateEmbedder",
    "embed_repository_state",
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


class RepositoryStateEmbedder:
    """Compute repo-state commit embeddings with a file-level cache."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        cache: FileEmbeddingCache | None = None,
        cache_backend: str | None = None,
        repo: Repo | None = None,
        experiment_id: UUID | str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._repo = repo
        self._experiment_id: UUID | None = _coerce_uuid(experiment_id)
        self.cache = cache or build_file_embedding_cache(
            settings=self.settings,
            backend=cache_backend,
            experiment_id=self._experiment_id,
        )

    def run(
        self,
        *,
        commit_hash: str,
        repo_root: Path | None = None,
        mode: RepoStateEmbeddingMode = "auto",
    ) -> tuple[CommitCodeEmbedding | None, RepoStateEmbeddingStats]:
        """Return a commit-level embedding representing the repo state."""

        requested_commit = str(commit_hash).strip() or None
        if not requested_commit:
            stats = RepoStateEmbeddingStats(
                commit_hash=requested_commit,
                eligible_files=0,
                files_embedded=0,
                files_aggregated=0,
                unique_blobs=0,
                cache_hits=0,
                cache_misses=0,
                skipped_empty_after_preprocess=0,
                skipped_failed_embedding=0,
            )
            return None, stats

        root = Path(repo_root or Path.cwd()).resolve()

        repo = self._repo
        if repo is None:
            try:
                repo = Repo(root, search_parent_directories=True)
            except Exception:
                stats = RepoStateEmbeddingStats(
                    commit_hash=requested_commit,
                    eligible_files=0,
                    files_embedded=0,
                    files_aggregated=0,
                    unique_blobs=0,
                    cache_hits=0,
                    cache_misses=0,
                    skipped_empty_after_preprocess=0,
                    skipped_failed_embedding=0,
                )
                return None, stats

        try:
            canonical = str(getattr(repo.commit(requested_commit), "hexsha", "") or "").strip()
        except Exception as exc:
            raise ValueError(f"Unknown commit {requested_commit!r}") from exc
        if not canonical:
            raise ValueError(f"Unknown commit {requested_commit!r}")

        # Fast path: use persisted aggregate when available (requires experiment_id + DB cache backend).
        aggregate = self._load_aggregate(commit_hash=canonical, repo_root=root)
        if aggregate is not None:
            vector = _divide_vector(tuple(float(v) for v in aggregate.sum_vector), int(aggregate.file_count))
            stats = RepoStateEmbeddingStats(
                commit_hash=canonical,
                eligible_files=int(aggregate.file_count),
                files_embedded=0,
                files_aggregated=int(aggregate.file_count),
                unique_blobs=0,
                cache_hits=0,
                cache_misses=0,
                skipped_empty_after_preprocess=0,
                skipped_failed_embedding=0,
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
                canonical,
                aggregate.file_count,
                len(vector),
            )
            return embedding, stats

        # Attempt diff-based incremental aggregation when possible.
        try:
            incremental = self._try_incremental_aggregate(commit_hash=canonical, repo_root=root)
        except RepoStateEmbeddingError as exc:
            if mode == "incremental_only":
                raise
            log.warning(
                "Repo-state incremental aggregation failed for commit {} (falling back to full recompute): {}",
                canonical[:12],
                exc,
            )
            incremental = None
        if incremental is not None:
            agg_row, vector = incremental
            stats = RepoStateEmbeddingStats(
                commit_hash=canonical,
                eligible_files=int(agg_row.file_count),
                files_embedded=0,
                files_aggregated=int(agg_row.file_count),
                unique_blobs=0,
                cache_hits=0,
                cache_misses=0,
                skipped_empty_after_preprocess=0,
                skipped_failed_embedding=0,
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

        if mode == "incremental_only":
            raise RepoStateEmbeddingError(
                "Repo-state embedding is running in incremental-only mode; "
                f"no aggregate cache hit and no valid incremental derivation for commit {canonical}."
            )

        repo_files = list_repository_files(
            repo_root=root,
            commit_hash=canonical,
            settings=self.settings,
            repo=repo,
        )
        if not repo_files:
            stats = RepoStateEmbeddingStats(
                commit_hash=canonical,
                eligible_files=0,
                files_embedded=0,
                files_aggregated=0,
                unique_blobs=0,
                cache_hits=0,
                cache_misses=0,
                skipped_empty_after_preprocess=0,
                skipped_failed_embedding=0,
            )
            return None, stats

        # Cache lookups are keyed by blob SHA; de-duplicate upfront.
        blob_shas = [entry.blob_sha for entry in repo_files if entry.blob_sha]
        unique_blob_shas = sorted(set(blob_shas))
        cached = self.cache.get_many(unique_blob_shas)

        misses = [sha for sha in unique_blob_shas if sha not in cached]
        cache_hits = len(unique_blob_shas) - len(misses)

        vectors_for_misses, embedded_count, skipped_empty = self._embed_cache_misses(
            root=root,
            commit_hash=canonical,
            repo_files=repo_files,
            missing_blob_shas=misses,
        )
        if vectors_for_misses:
            self.cache.put_many(vectors_for_misses)
            cached.update(vectors_for_misses)

        # Aggregate per-file vectors (uniform weight per file path).
        file_vectors: list[Vector] = []
        skipped_failed = 0
        for file in repo_files:
            vec = cached.get(file.blob_sha)
            if not vec:
                skipped_failed += 1
                continue
            file_vectors.append(vec)

        commit_vector, sum_vector, aggregated_count = _mean_and_sum_vectors(file_vectors)
        stats = RepoStateEmbeddingStats(
            commit_hash=canonical,
            eligible_files=len(repo_files),
            files_embedded=embedded_count,
            files_aggregated=aggregated_count,
            unique_blobs=len(unique_blob_shas),
            cache_hits=cache_hits,
            cache_misses=len(misses),
            skipped_empty_after_preprocess=skipped_empty,
            skipped_failed_embedding=skipped_failed,
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
            canonical,
            stats.eligible_files,
            stats.unique_blobs,
            stats.cache_hits,
            stats.cache_misses,
            stats.files_aggregated,
            embedding.dimensions,
        )

        # Persist aggregate for reuse when enabled.
        self._persist_aggregate(
            commit_hash=canonical,
            repo_root=root,
            sum_vector=sum_vector,
            file_count=aggregated_count,
        )
        return embedding, stats

    def load_aggregate(
        self,
        *,
        commit_hash: str,
        repo_root: Path,
    ) -> MapElitesRepoStateAggregate | None:
        """Load a persisted repo-state aggregate when available.

        This is a thin wrapper around the internal aggregate lookup, exposed so
        other modules (e.g. scheduler bootstrap) can verify persistence without
        reaching into private methods.
        """

        return self._load_aggregate(commit_hash=commit_hash, repo_root=repo_root)

    def _load_aggregate(
        self,
        *,
        commit_hash: str,
        repo_root: Path,
    ) -> MapElitesRepoStateAggregate | None:
        if self._experiment_id is None:
            return None
        if not isinstance(self.cache, DatabaseFileEmbeddingCache):
            return None

        requested_dims = int(getattr(self.cache, "requested_dimensions", 0))
        if requested_dims <= 0:
            return None
        try:
            with session_scope() as session:
                stmt = select(MapElitesRepoStateAggregate).where(
                    MapElitesRepoStateAggregate.experiment_id == self._experiment_id,
                    MapElitesRepoStateAggregate.commit_hash == str(commit_hash),
                )
                row = session.execute(stmt).scalar_one_or_none()
        except Exception as exc:  # pragma: no cover - DB failure handling
            log.warning("Repo-state aggregate read failed for {}: {}", commit_hash, exc)
            return None

        if not row:
            return None
        if int(row.file_count or 0) <= 0:
            return None
        if not row.sum_vector:
            return None
        if len(row.sum_vector) != requested_dims:
            return None
        return row

    def _persist_aggregate(
        self,
        *,
        commit_hash: str,
        repo_root: Path,
        sum_vector: Vector,
        file_count: int,
    ) -> None:
        if self._experiment_id is None:
            return
        if not isinstance(self.cache, DatabaseFileEmbeddingCache):
            return
        if file_count <= 0 or not sum_vector:
            return
        if len(sum_vector) != int(getattr(self.cache, "requested_dimensions", 0) or 0):
            return
        row = MapElitesRepoStateAggregate(
            experiment_id=self._experiment_id,
            commit_hash=str(commit_hash),
            file_count=int(file_count),
            sum_vector=[float(v) for v in sum_vector],
        )

        try:
            with session_scope() as session:
                session.merge(row)
        except Exception as exc:  # pragma: no cover - DB failure handling
            log.warning("Repo-state aggregate persist failed for {}: {}", commit_hash, exc)

    @dataclass(frozen=True, slots=True)
    class _VectorMeta:
        vector: Vector

    def _load_file_cache_metadata(
        self,
        *,
        blob_shas: Sequence[str],
        dimensions: int,
    ) -> dict[str, "RepositoryStateEmbedder._VectorMeta"]:
        if not blob_shas:
            return {}
        if self._experiment_id is None:
            return {}
        if not isinstance(self.cache, DatabaseFileEmbeddingCache):
            return {}
        dims = int(dimensions or 0)
        if dims <= 0:
            return {}
        cleaned = [str(sha).strip() for sha in blob_shas if str(sha).strip()]
        unique = sorted(set(cleaned))
        if not unique:
            return {}

        found: dict[str, RepositoryStateEmbedder._VectorMeta] = {}
        try:
            with session_scope() as session:
                for batch in _batched(unique, 500):
                    stmt = (
                        select(
                            MapElitesFileEmbeddingCache.blob_sha,
                            MapElitesFileEmbeddingCache.vector,
                            MapElitesFileEmbeddingCache.embedding_model,
                            MapElitesFileEmbeddingCache.dimensions,
                        )
                        .where(
                            MapElitesFileEmbeddingCache.experiment_id == self._experiment_id,
                            MapElitesFileEmbeddingCache.blob_sha.in_(batch),
                        )
                    )
                    for sha, vec, model, stored_dims in session.execute(stmt).all():
                        if str(model or "") != str(self.cache.embedding_model):
                            raise RepoStateEmbeddingError(
                                "File embedding cache entry has an unexpected embedding model; "
                                "reset the DB (dev). "
                                f"(experiment_id={self._experiment_id} blob_sha={sha} "
                                f"expected_model={self.cache.embedding_model!r} got_model={model!r})"
                            )
                        if int(stored_dims or 0) != dims:
                            raise RepoStateEmbeddingError(
                                "File embedding cache entry has unexpected dimensions; reset the DB (dev). "
                                f"(experiment_id={self._experiment_id} blob_sha={sha} "
                                f"expected_dims={dims} got_dims={stored_dims!r})"
                            )
                        vector = tuple(float(v) for v in (vec or ()))
                        if not vector:
                            raise RepoStateEmbeddingError(
                                "File embedding cache contains an empty vector; reset the DB (dev). "
                                f"(experiment_id={self._experiment_id} blob_sha={sha} dims={dims})"
                            )
                        if len(vector) != dims:
                            raise RepoStateEmbeddingError(
                                "File embedding cache vector has unexpected dimensions; reset the DB (dev). "
                                f"(experiment_id={self._experiment_id} blob_sha={sha} "
                                f"expected_dims={dims} got_dims={len(vector)})"
                            )
                        found[str(sha)] = RepositoryStateEmbedder._VectorMeta(vector=vector)
        except RepoStateEmbeddingError:
            raise
        except Exception as exc:  # pragma: no cover - DB failure handling
            raise RepoStateEmbeddingError(f"Repo-state file cache metadata read failed: {exc}") from exc

        return found

    def _try_incremental_aggregate(
        self,
        *,
        commit_hash: str,
        repo_root: Path,
    ) -> tuple[MapElitesRepoStateAggregate, Vector] | None:
        if self._experiment_id is None:
            return None
        if not isinstance(self.cache, DatabaseFileEmbeddingCache):
            return None

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

        parent_agg = self._load_aggregate(commit_hash=parent_hash, repo_root=repo_root)
        if parent_agg is None:
            return None
        if int(parent_agg.file_count or 0) <= 0:
            return None

        pinned_ignore = str(getattr(self.settings, "mapelites_repo_state_ignore_text", "") or "").strip()
        # Ignore rules are pinned for the experiment lifecycle, so root ignore file changes in the
        # evolved history do not affect eligibility. We log a warning for observability.
        ignore_changed = False
        for filename in ROOT_IGNORE_FILES:
            if _root_file_blob_sha(repo, parent_hash, filename) != _root_file_blob_sha(repo, commit_hash, filename):
                ignore_changed = True
        if ignore_changed and pinned_ignore:
            log.warning(
                "Root ignore files changed for commit {} (parent={}) but ignore rules are pinned; "
                "the changes will not affect repo-state eligibility.",
                commit_hash[:12],
                parent_hash[:12],
            )

        raw_sum = tuple(float(v) for v in (parent_agg.sum_vector or ()))
        dims = len(raw_sum)
        if dims <= 0:
            return None
        if dims != int(getattr(self.cache, "requested_dimensions", 0) or 0):
            return None
        sum_vec: Vector = raw_sum
        file_count = int(parent_agg.file_count)

        repo_prefix = _resolve_git_prefix(repo, repo_root)
        ignore_spec = build_pinned_ignore_spec(pinned_ignore)
        preprocess_filter = CodePreprocessor(
            repo_root=repo_root,
            settings=self.settings,
            commit_hash=None,
        )
        max_bytes = max(int(self.settings.mapelites_preprocess_max_file_size_kb), 1) * 1024

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

        diffs = _parse_diff_tree_raw_z(raw)
        if not diffs:
            # No changes relative to parent: derive directly from parent aggregate.
            vector = _divide_vector(sum_vec, file_count)
            self._persist_aggregate(
                commit_hash=commit_hash,
                repo_root=repo_root,
                sum_vector=sum_vec,
                file_count=file_count,
            )
            persisted = self._load_aggregate(commit_hash=commit_hash, repo_root=repo_root)
            if persisted is None:
                return None
            return persisted, vector

        def _selected(path_str: str | None, sha: str | None) -> tuple[bool, Path | None, str | None]:
            if not path_str or _is_null_sha(sha):
                return False, None, None
            git_path = path_str.strip().lstrip("/")
            if not git_path:
                return False, None, None
            if ignore_spec and is_ignored_path(ignore_spec, git_path):
                return False, None, None
            repo_rel = _git_path_to_repo_rel(git_path, repo_prefix=repo_prefix)
            if repo_rel is None:
                return False, None, None
            if preprocess_filter.is_excluded(repo_rel):
                return False, None, None
            if not preprocess_filter.is_code_file(repo_rel):
                return False, None, None
            size = _blob_size_bytes(repo, str(sha))
            if size is None or size <= 0 or size > max_bytes:
                return False, None, None
            return True, repo_rel, str(sha)

        old_shas: list[str] = []
        new_shas: list[str] = []
        repo_files_for_new_misses: list[RepositoryFile] = []

        # First pass: determine which blob SHAs we may need.
        for entry in diffs:
            old_ok, old_repo_rel, old_sha = _selected(entry.old_path, entry.old_sha)
            new_ok, new_repo_rel, new_sha = _selected(entry.new_path, entry.new_sha)
            if old_ok and old_sha:
                old_shas.append(old_sha)
            if new_ok and new_sha:
                new_shas.append(new_sha)
            if new_ok and new_sha and new_repo_rel:
                repo_files_for_new_misses.append(
                    RepositoryFile(path=new_repo_rel, blob_sha=new_sha, size_bytes=0)
                )

        candidates = sorted(set(old_shas + new_shas))
        metadata = self._load_file_cache_metadata(blob_shas=candidates, dimensions=dims)

        # Embed cache misses for the *new* side only.
        missing_new = [sha for sha in sorted(set(new_shas)) if sha not in metadata]
        if missing_new:
            vectors_for_misses, _embedded_count, _skipped_empty = self._embed_cache_misses(
                root=repo_root,
                commit_hash=commit_hash,
                repo_files=repo_files_for_new_misses,
                missing_blob_shas=missing_new,
            )
            if vectors_for_misses:
                self.cache.put_many(vectors_for_misses)
                for sha, vec in vectors_for_misses.items():
                    if vec and len(vec) == dims:
                        metadata[sha] = RepositoryStateEmbedder._VectorMeta(vector=vec)

        # Second pass: apply delta updates using "included" semantics.
        for entry in diffs:
            old_ok, _old_repo_rel, old_sha = _selected(entry.old_path, entry.old_sha)
            new_ok, _new_repo_rel, new_sha = _selected(entry.new_path, entry.new_sha)

            old_meta: RepositoryStateEmbedder._VectorMeta | None = None
            if old_ok and old_sha:
                old_meta = metadata.get(old_sha)
            new_meta: RepositoryStateEmbedder._VectorMeta | None = None
            if new_ok and new_sha:
                new_meta = metadata.get(new_sha)

            old_included = bool(old_meta is not None and old_meta.vector)
            new_included = bool(new_meta is not None and new_meta.vector)

            if old_included and not new_included:
                if old_meta is None:  # pragma: no cover - type narrowing guard
                    continue
                sum_vec = _vector_sub(sum_vec, old_meta.vector)
                file_count -= 1
                continue
            if not old_included and new_included:
                if new_meta is None:  # pragma: no cover - type narrowing guard
                    continue
                sum_vec = _vector_add(sum_vec, new_meta.vector)
                file_count += 1
                continue
            if old_included and new_included:
                if old_meta is None or new_meta is None:  # pragma: no cover - type narrowing guard
                    continue
                sum_vec = _vector_add(sum_vec, _vector_sub(new_meta.vector, old_meta.vector))

        if file_count <= 0 or not sum_vec:
            return None

        vector = _divide_vector(sum_vec, file_count)
        if not vector:
            return None

        self._persist_aggregate(
            commit_hash=commit_hash,
            repo_root=repo_root,
            sum_vector=sum_vec,
            file_count=file_count,
        )
        persisted = self._load_aggregate(commit_hash=commit_hash, repo_root=repo_root)
        if persisted is None:
            return None
        return persisted, vector

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

        preprocessed: list[PreprocessedFile] = []
        skipped_empty = 0
        blob_for_path: dict[Path, str] = {}
        for blob_sha, entry in wanted.items():
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

        if not preprocessed:
            return {}, 0, skipped_empty

        chunked = chunk_preprocessed_files(
            cast(Sequence[PreprocessedArtifact], preprocessed),
            settings=self.settings,
        )
        commit_embedding = embed_chunked_files(chunked, settings=self.settings)
        if not commit_embedding:
            return {}, 0, skipped_empty

        vectors: dict[str, Vector] = {}
        for file_embedding in commit_embedding.files:
            blob_sha = blob_for_path.get(file_embedding.file.path)
            if not blob_sha:
                continue
            vectors[blob_sha] = tuple(float(v) for v in file_embedding.vector)

        return vectors, len(vectors), skipped_empty


def embed_repository_state(
    *,
    commit_hash: str,
    repo_root: Path | None = None,
    settings: Settings | None = None,
    cache: FileEmbeddingCache | None = None,
    cache_backend: str | None = None,
    repo: Repo | None = None,
    experiment_id: UUID | str | None = None,
    mode: RepoStateEmbeddingMode = "auto",
) -> tuple[CommitCodeEmbedding | None, RepoStateEmbeddingStats]:
    """Functional wrapper around `RepositoryStateEmbedder`."""

    embedder = RepositoryStateEmbedder(
        settings=settings,
        cache=cache,
        cache_backend=cache_backend,
        repo=repo,
        experiment_id=experiment_id,
    )
    return embedder.run(
        commit_hash=commit_hash,
        repo_root=repo_root,
        mode=mode,
    )


def _mean_and_sum_vectors(vectors: Sequence[Vector]) -> tuple[Vector, Vector, int]:
    """Return (mean_vector, sum_vector, count)."""
    if not vectors:
        return (), (), 0
    dims = len(vectors[0])
    if dims == 0:
        return (), (), 0
    totals = [0.0] * dims
    count = 0
    for vec in vectors:
        if len(vec) != dims:
            raise ValueError("Embedding dimension mismatch during repo-state aggregation.")
        for i in range(dims):
            totals[i] += float(vec[i])
        count += 1
    if count <= 0:
        return (), (), 0
    mean = tuple(value / float(count) for value in totals)
    return mean, tuple(totals), count


def _divide_vector(vector: Vector, count: int) -> Vector:
    if not vector or count <= 0:
        return ()
    return tuple(float(v) / float(count) for v in vector)


def _coerce_uuid(value: UUID | str | None) -> UUID | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return UUID(text)
    except Exception:
        return None


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


def _root_file_blob_sha(repo: Repo, commit_hash: str, path: str) -> str | None:
    """Return the blob SHA for a root-level file at a commit.

    Args:
        repo: GitPython repository.
        commit_hash: Commit-ish to resolve against.
        path: Root-relative path (e.g. ".gitignore", ".loreleyignore").
    """
    raw = str(path or "").strip().lstrip("/")
    if not raw:
        return None
    try:
        value = repo.git.rev_parse(f"{commit_hash}:{raw}").strip()
        return value or None
    except GitCommandError:
        return None


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


def _vector_add(a: Vector, b: Vector) -> Vector:
    if not a:
        return b
    if not b:
        return a
    if len(a) != len(b):
        raise ValueError("Embedding dimension mismatch during vector addition.")
    return tuple(float(x) + float(y) for x, y in zip(a, b))


def _vector_sub(a: Vector, b: Vector) -> Vector:
    if not b:
        return a
    if not a:
        raise ValueError("Cannot subtract from an empty vector.")
    if len(a) != len(b):
        raise ValueError("Embedding dimension mismatch during vector subtraction.")
    return tuple(float(x) - float(y) for x, y in zip(a, b))


def _batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    step = max(1, int(batch_size))
    for start in range(0, len(items), step):
        yield items[start : start + step]


