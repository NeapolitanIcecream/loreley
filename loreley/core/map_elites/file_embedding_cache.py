"""File-level embedding cache for repo-state embeddings.

The repo-state pipeline embeds *files* (keyed by content fingerprint) and then
aggregates them into a commit-level vector. This module provides a cache so that
unchanged files can reuse prior embeddings across commits.

Cache key:
- `blob_sha`: git blob SHA (preferred content fingerprint).
- `embedding_model`: OpenAI embedding model name.
- `dimensions`: actual output vector length (guard).
- `pipeline_signature`: hash of preprocessing+chunking+embedding knobs that
  affect the produced vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Mapping, Protocol, Sequence, TypeVar

from loguru import logger
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError

from loreley.config import Settings, get_settings
from loreley.db.base import session_scope
from loreley.db.models import MapElitesFileEmbeddingCache

log = logger.bind(module="map_elites.file_embedding_cache")

Vector = tuple[float, ...]
T = TypeVar("T")

__all__ = [
    "Vector",
    "FileEmbeddingCache",
    "InMemoryFileEmbeddingCache",
    "DatabaseFileEmbeddingCache",
    "build_pipeline_signature",
    "build_file_embedding_cache",
]


class FileEmbeddingCache(Protocol):
    """Abstract cache interface keyed by blob sha."""

    embedding_model: str
    requested_dimensions: int | None
    pipeline_signature: str

    def get_many(self, blob_shas: Sequence[str]) -> dict[str, Vector]:
        """Return vectors for any known blob SHAs (missing keys omitted)."""
        ...

    def put_many(self, vectors: Mapping[str, Vector]) -> None:
        """Persist vectors for blob SHAs."""
        ...


def build_pipeline_signature(*, settings: Settings | None = None) -> str:
    """Hash all knobs that affect file-level embeddings.

    This signature is intentionally conservative: any change that could alter
    preprocessing/chunking/embedding output should produce a new signature so
    that cached vectors are not reused incorrectly.
    """

    s = settings or get_settings()
    payload = {
        "version": 1,
        "preprocess": {
            "strip_comments": bool(s.mapelites_preprocess_strip_comments),
            "strip_block_comments": bool(s.mapelites_preprocess_strip_block_comments),
            "max_blank_lines": int(s.mapelites_preprocess_max_blank_lines),
            "tab_width": int(s.mapelites_preprocess_tab_width),
        },
        "chunk": {
            "target_lines": int(s.mapelites_chunk_target_lines),
            "min_lines": int(s.mapelites_chunk_min_lines),
            "overlap_lines": int(s.mapelites_chunk_overlap_lines),
            "max_chunks_per_file": int(s.mapelites_chunk_max_chunks_per_file),
            "boundary_keywords": list(s.mapelites_chunk_boundary_keywords or []),
        },
        "embedding": {
            "model": str(s.mapelites_code_embedding_model),
            "requested_dimensions": (
                int(s.mapelites_code_embedding_dimensions)
                if s.mapelites_code_embedding_dimensions is not None
                else None
            ),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


@dataclass(slots=True)
class InMemoryFileEmbeddingCache:
    """Simple in-memory cache used for tests/local runs."""

    embedding_model: str
    requested_dimensions: int | None
    pipeline_signature: str

    _store: dict[str, Vector]

    def __init__(
        self,
        *,
        embedding_model: str,
        requested_dimensions: int | None,
        pipeline_signature: str,
    ) -> None:
        self.embedding_model = embedding_model
        self.requested_dimensions = requested_dimensions
        self.pipeline_signature = pipeline_signature
        self._store = {}

    def get_many(self, blob_shas: Sequence[str]) -> dict[str, Vector]:
        found: dict[str, Vector] = {}
        for sha in blob_shas:
            key = str(sha).strip()
            if not key:
                continue
            vector = self._store.get(key)
            if vector:
                found[key] = vector
        return found

    def put_many(self, vectors: Mapping[str, Vector]) -> None:
        for sha, vector in vectors.items():
            key = str(sha).strip()
            if not key:
                continue
            self._validate_vector(vector)
            self._store[key] = tuple(float(v) for v in vector)

    def _validate_vector(self, vector: Vector) -> None:
        if not vector:
            raise ValueError("Cannot cache an empty embedding vector.")
        if self.requested_dimensions is not None and len(vector) != self.requested_dimensions:
            raise ValueError(
                "Embedding dimension mismatch for cache insert "
                f"(expected {self.requested_dimensions} got {len(vector)})"
            )


@dataclass(slots=True)
class DatabaseFileEmbeddingCache:
    """Postgres-backed cache using `MapElitesFileEmbeddingCache` table."""

    embedding_model: str
    requested_dimensions: int | None
    pipeline_signature: str

    def get_many(self, blob_shas: Sequence[str]) -> dict[str, Vector]:
        cleaned = _unique_clean_blob_shas(blob_shas)
        if not cleaned:
            return {}

        found: dict[str, Vector] = {}
        try:
            with session_scope() as session:
                for batch in _batched(cleaned, 500):
                    base_conditions = [
                        MapElitesFileEmbeddingCache.blob_sha.in_(batch),
                        MapElitesFileEmbeddingCache.embedding_model == self.embedding_model,
                        MapElitesFileEmbeddingCache.pipeline_signature == self.pipeline_signature,
                    ]
                    if self.requested_dimensions is not None:
                        base_conditions.append(
                            MapElitesFileEmbeddingCache.dimensions
                            == int(self.requested_dimensions)
                        )

                    stmt = select(MapElitesFileEmbeddingCache).where(*base_conditions)
                    rows = list(session.execute(stmt).scalars())
                    if not rows:
                        continue

                    # When no explicit dimensions are requested, we may observe multiple
                    # rows per blob (same signature) with different dimensions.
                    # Select a single, deterministic dimension for this read batch and
                    # only return vectors matching it.
                    if self.requested_dimensions is None:
                        candidates: dict[int, dict[str, Vector]] = {}
                        for row in rows:
                            vector = tuple(float(v) for v in (row.vector or []))
                            if not vector:
                                continue
                            dims = int(row.dimensions or len(vector))
                            if dims != len(vector):
                                continue
                            candidates.setdefault(dims, {})[str(row.blob_sha)] = vector

                        if not candidates:
                            continue
                        if len(candidates) == 1:
                            found.update(next(iter(candidates.values())))
                            continue

                        chosen_dims = max(
                            candidates.keys(),
                            key=lambda d: (len(candidates[d]), d),
                        )
                        log.warning(
                            "Multiple embedding dimensions found for model={} signature={} (dims={} chosen={} batch_size={})",
                            self.embedding_model,
                            self.pipeline_signature[:12],
                            sorted(candidates.keys()),
                            chosen_dims,
                            len(batch),
                        )
                        found.update(candidates[chosen_dims])
                        continue

                    # Explicit dimensions: accept rows directly (already filtered).
                    for row in rows:
                        vector = tuple(float(v) for v in (row.vector or []))
                        if not vector:
                            continue
                        if row.dimensions and len(vector) != int(row.dimensions):
                            continue
                        if (
                            self.requested_dimensions is not None
                            and len(vector) != self.requested_dimensions
                        ):
                            continue
                        found[str(row.blob_sha)] = vector
        except SQLAlchemyError as exc:
            log.error("Failed to read file embedding cache: {}", exc)
            return {}

        return found

    def put_many(self, vectors: Mapping[str, Vector]) -> None:
        if not vectors:
            return

        values: list[dict[str, object]] = []
        for sha, vector in vectors.items():
            key = str(sha).strip()
            if not key:
                continue
            vec = tuple(float(v) for v in vector)
            if not vec:
                continue
            if self.requested_dimensions is not None and len(vec) != self.requested_dimensions:
                raise ValueError(
                    "Embedding dimension mismatch for cache insert "
                    f"(expected {self.requested_dimensions} got {len(vec)})"
                )
            values.append(
                {
                    "blob_sha": key,
                    "embedding_model": self.embedding_model,
                    "dimensions": len(vec),
                    "pipeline_signature": self.pipeline_signature,
                    "vector": list(vec),
                }
            )

        if not values:
            return

        try:
            with session_scope() as session:
                for batch in _batched(values, 500):
                    stmt = pg_insert(MapElitesFileEmbeddingCache).values(batch)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=[
                            "blob_sha",
                            "embedding_model",
                            "dimensions",
                            "pipeline_signature",
                        ],
                        set_={
                            "vector": stmt.excluded.vector,
                        },
                    )
                    session.execute(stmt)
        except SQLAlchemyError as exc:
            log.error("Failed to persist file embedding cache: {}", exc)


def build_file_embedding_cache(
    *,
    settings: Settings | None = None,
    backend: str | None = None,
) -> FileEmbeddingCache:
    """Factory for selecting an embedding cache backend.

    - If `backend` is provided, it wins.
    - Else consult `settings.mapelites_file_embedding_cache_backend` when set.
    - Else default to `db`.
    """

    s = settings or get_settings()
    chosen = (backend or getattr(s, "mapelites_file_embedding_cache_backend", None) or "").strip()
    if not chosen:
        chosen = "db"

    pipeline_signature = build_pipeline_signature(settings=s)
    embedding_model = str(s.mapelites_code_embedding_model)
    requested_dimensions = (
        int(s.mapelites_code_embedding_dimensions)
        if s.mapelites_code_embedding_dimensions is not None
        else None
    )

    if chosen == "memory":
        return InMemoryFileEmbeddingCache(
            embedding_model=embedding_model,
            requested_dimensions=requested_dimensions,
            pipeline_signature=pipeline_signature,
        )
    if chosen == "db":
        return DatabaseFileEmbeddingCache(
            embedding_model=embedding_model,
            requested_dimensions=requested_dimensions,
            pipeline_signature=pipeline_signature,
        )

    raise ValueError(f"Unknown file embedding cache backend: {chosen!r}")


def _unique_clean_blob_shas(blob_shas: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for sha in blob_shas:
        value = str(sha).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return cleaned


def _batched(items: Sequence[T], batch_size: int) -> Iterable[Sequence[T]]:
    step = max(1, int(batch_size))
    for start in range(0, len(items), step):
        yield items[start : start + step]


