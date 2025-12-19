"""Repo-state commit embeddings built from file-level embeddings.

This module implements the repo-state embedding design:

- Enumerate eligible files for a given commit (`treeish`) respecting `.gitignore`
  and basic MAP-Elites preprocessing filters.
- Reuse a file-level embedding cache keyed by git blob SHA.
- Only embed cache misses (new/modified files).
- Aggregate all file embeddings into a single commit vector via **uniform mean**
  (each eligible file contributes weight 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, cast

from git import Repo
from loguru import logger

from loreley.config import Settings, get_settings
from .chunk import PreprocessedArtifact, chunk_preprocessed_files
from .code_embedding import CommitCodeEmbedding, embed_chunked_files
from .file_embedding_cache import FileEmbeddingCache, build_file_embedding_cache
from .preprocess import CodePreprocessor, PreprocessedFile
from .repository_files import RepositoryFile, list_repository_files

log = logger.bind(module="map_elites.repository_state_embedding")

Vector = tuple[float, ...]

__all__ = [
    "RepoStateEmbeddingStats",
    "RepositoryStateEmbedder",
    "embed_repository_state",
]


@dataclass(frozen=True, slots=True)
class RepoStateEmbeddingStats:
    treeish: str | None
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
    ) -> None:
        self.settings = settings or get_settings()
        self.cache = cache or build_file_embedding_cache(
            settings=self.settings,
            backend=cache_backend,
        )
        self._repo = repo

    def run(
        self,
        *,
        commit_hash: str,
        repo_root: Path | None = None,
        treeish: str | None = None,
    ) -> tuple[CommitCodeEmbedding | None, RepoStateEmbeddingStats]:
        """Return a commit-level embedding representing the repo state."""

        effective_treeish = (treeish or commit_hash).strip() or None
        if not effective_treeish:
            stats = RepoStateEmbeddingStats(
                treeish=effective_treeish,
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
        repo_files = list_repository_files(
            repo_root=root,
            treeish=effective_treeish,
            settings=self.settings,
            repo=self._repo,
        )
        if not repo_files:
            stats = RepoStateEmbeddingStats(
                treeish=effective_treeish,
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
            treeish=effective_treeish,
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

        commit_vector = _mean_vector(file_vectors)
        stats = RepoStateEmbeddingStats(
            treeish=effective_treeish,
            eligible_files=len(repo_files),
            files_embedded=embedded_count,
            files_aggregated=len(file_vectors),
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
            "Repo-state embedding for commit {} (treeish={}): files={} blobs={} hits={} misses={} agg_files={} dims={}",
            commit_hash,
            effective_treeish,
            stats.eligible_files,
            stats.unique_blobs,
            stats.cache_hits,
            stats.cache_misses,
            stats.files_aggregated,
            embedding.dimensions,
        )
        return embedding, stats

    def _embed_cache_misses(
        self,
        *,
        root: Path,
        treeish: str,
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
            treeish=treeish,
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
    treeish: str | None = None,
    settings: Settings | None = None,
    cache: FileEmbeddingCache | None = None,
    cache_backend: str | None = None,
    repo: Repo | None = None,
) -> tuple[CommitCodeEmbedding | None, RepoStateEmbeddingStats]:
    """Functional wrapper around `RepositoryStateEmbedder`."""

    embedder = RepositoryStateEmbedder(
        settings=settings,
        cache=cache,
        cache_backend=cache_backend,
        repo=repo,
    )
    return embedder.run(
        commit_hash=commit_hash,
        repo_root=repo_root,
        treeish=treeish,
    )


def _mean_vector(vectors: Sequence[Vector]) -> Vector:
    if not vectors:
        return ()
    dims = len(vectors[0])
    if dims == 0:
        return ()
    totals = [0.0] * dims
    count = 0
    for vec in vectors:
        if len(vec) != dims:
            raise ValueError("Embedding dimension mismatch during repo-state aggregation.")
        for i in range(dims):
            totals[i] += float(vec[i])
        count += 1
    if count <= 0:
        return ()
    return tuple(value / float(count) for value in totals)


