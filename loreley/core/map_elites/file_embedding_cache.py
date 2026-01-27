"""File-level embedding cache for repo-state embeddings.

The repo-state pipeline embeds *files* (keyed by content fingerprint) and then
aggregates them into a commit-level vector. This module provides a cache so that
unchanged files can reuse prior embeddings across commits.

Cache key:
- `blob_sha`: git blob SHA (preferred content fingerprint).

The cache stores the embedding model name and output dimensionality alongside
vectors for validation and debugging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol, Sequence, TypeVar

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from loreley.config import Settings, get_settings
from loreley.db.base import session_scope
from loreley.db.models import MapElitesFileEmbeddingCache

Vector = tuple[float, ...]
T = TypeVar("T")

__all__ = [
    "Vector",
    "FileEmbeddingCache",
    "DatabaseFileEmbeddingCache",
    "build_file_embedding_cache",
]


class FileEmbeddingCache(Protocol):
    """Abstract cache interface keyed by blob sha."""

    embedding_model: str
    requested_dimensions: int

    def get_many(self, blob_shas: Sequence[str]) -> dict[str, Vector]:
        """Return vectors for any known blob SHAs (missing keys omitted)."""
        ...

    def put_many(self, vectors: Mapping[str, Vector]) -> None:
        """Persist vectors for blob SHAs."""
        ...


def _resolve_requested_dimensions(settings: Settings) -> int:
    """Return configured embedding dimensionality or raise a helpful error.

    In the env-only settings model, this value must be provided via environment
    variables and kept consistent across long-running processes.
    """

    raw = getattr(settings, "mapelites_code_embedding_dimensions", None)
    if raw is None:
        raise ValueError(
            "MAPELITES_CODE_EMBEDDING_DIMENSIONS is not configured. "
            "Set it in the environment for scheduler/worker/UI processes.",
        )
    dims = int(raw)
    if dims <= 0:
        raise ValueError("MAPELITES_CODE_EMBEDDING_DIMENSIONS must be a positive integer.")
    return dims


@dataclass(slots=True)
class DatabaseFileEmbeddingCache:
    """Postgres-backed cache using `MapElitesFileEmbeddingCache` table."""

    embedding_model: str
    requested_dimensions: int

    def get_many(self, blob_shas: Sequence[str]) -> dict[str, Vector]:
        cleaned = _unique_clean_blob_shas(blob_shas)
        if not cleaned:
            return {}

        dims = int(self.requested_dimensions)
        if dims <= 0:
            raise ValueError("Requested embedding dimensions must be a positive integer.")

        found: dict[str, Vector] = {}
        with session_scope() as session:
            for batch in _batched(cleaned, 500):
                base_conditions = [
                    MapElitesFileEmbeddingCache.blob_sha.in_(batch),
                ]

                stmt = select(MapElitesFileEmbeddingCache).where(*base_conditions)
                rows = list(session.execute(stmt).scalars())
                if not rows:
                    continue

                # Validate cached rows against cache invariants.
                for row in rows:
                    if str(getattr(row, "embedding_model", "") or "") != str(self.embedding_model):
                        raise ValueError(
                            "File embedding cache entry has an unexpected embedding model; "
                            "reset the DB (dev). "
                            f"(blob_sha={row.blob_sha} expected_model={self.embedding_model!r} "
                            f"got_model={row.embedding_model!r})"
                        )
                    if int(getattr(row, "dimensions", 0) or 0) != dims:
                        raise ValueError(
                            "File embedding cache entry has unexpected dimensions; "
                            "reset the DB (dev). "
                            f"(blob_sha={row.blob_sha} expected_dims={dims} got_dims={row.dimensions!r})"
                        )
                    vector = tuple(float(v) for v in (row.vector or []))
                    if not vector:
                        raise ValueError(
                            "File embedding cache contains an empty vector; reset the DB (dev). "
                            f"(blob_sha={row.blob_sha} dims={dims})"
                        )
                    if len(vector) != dims:
                        raise ValueError(
                            "File embedding cache vector has unexpected dimensions; "
                            "reset the DB (dev). "
                            f"(blob_sha={row.blob_sha} expected_dims={dims} got_dims={len(vector)})"
                        )
                    found[str(row.blob_sha)] = vector

        return found

    def put_many(self, vectors: Mapping[str, Vector]) -> None:
        if not vectors:
            return

        dims = int(self.requested_dimensions)
        if dims <= 0:
            raise ValueError("Requested embedding dimensions must be a positive integer.")

        values: list[dict[str, object]] = []
        for sha, vector in vectors.items():
            key = str(sha).strip()
            if not key:
                continue
            vec = tuple(float(v) for v in vector)
            if not vec:
                continue
            if len(vec) != dims:
                raise ValueError(
                    "Embedding dimension mismatch for cache insert "
                    f"(expected {dims} got {len(vec)})"
                )
            values.append(
                {
                    "blob_sha": key,
                    "embedding_model": self.embedding_model,
                    "dimensions": len(vec),
                    "vector": list(vec),
                }
            )

        if not values:
            return

        with session_scope() as session:
            for batch in _batched(values, 500):
                stmt = pg_insert(MapElitesFileEmbeddingCache).values(batch)
                stmt = stmt.on_conflict_do_nothing(
                    index_elements=[
                        "blob_sha",
                    ],
                )
                session.execute(stmt)


def build_file_embedding_cache(
    *,
    settings: Settings | None = None,
) -> FileEmbeddingCache:
    """Factory for constructing the DB-backed embedding cache."""

    s = settings or get_settings()
    embedding_model = str(s.mapelites_code_embedding_model)
    requested_dimensions = _resolve_requested_dimensions(s)

    return DatabaseFileEmbeddingCache(
        embedding_model=embedding_model,
        requested_dimensions=requested_dimensions,
    )


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


