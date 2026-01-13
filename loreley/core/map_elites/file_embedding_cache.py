"""File-level embedding cache for repo-state embeddings.

The repo-state pipeline embeds *files* (keyed by content fingerprint) and then
aggregates them into a commit-level vector. This module provides a cache so that
unchanged files can reuse prior embeddings across commits.

Cache key:
- `experiment_id`: experiment scope for stable, locked behaviour settings.
- `blob_sha`: git blob SHA (preferred content fingerprint).

The embedding model name and output dimensionality are experiment-scoped
invariants persisted in `Experiment.config_snapshot`. The database cache stores
them alongside vectors for validation and debugging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol, Sequence, TypeVar
from uuid import UUID

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

    The embedding dimensionality is an experiment-scoped invariant persisted in
    `Experiment.config_snapshot`. Long-running services should load it from the
    DB snapshot (via `resolve_experiment_settings`) instead of relying on local
    environment variables.
    """

    raw = getattr(settings, "mapelites_code_embedding_dimensions", None)
    if raw is None:
        raise ValueError(
            "MAPELITES_CODE_EMBEDDING_DIMENSIONS is not configured. "
            "This value is experiment-scoped and must be provided by the scheduler "
            "when deriving an experiment, then loaded from the DB snapshot by other services.",
        )
    dims = int(raw)
    if dims <= 0:
        raise ValueError("MAPELITES_CODE_EMBEDDING_DIMENSIONS must be a positive integer.")
    return dims


@dataclass(slots=True)
class InMemoryFileEmbeddingCache:
    """Simple in-memory cache used for tests/local runs."""

    embedding_model: str
    requested_dimensions: int

    _store: dict[str, Vector]

    def __init__(
        self,
        *,
        embedding_model: str,
        requested_dimensions: int,
    ) -> None:
        self.embedding_model = embedding_model
        self.requested_dimensions = requested_dimensions
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
        if len(vector) != int(self.requested_dimensions):
            raise ValueError(
                "Embedding dimension mismatch for cache insert "
                f"(expected {self.requested_dimensions} got {len(vector)})"
            )


@dataclass(slots=True)
class DatabaseFileEmbeddingCache:
    """Postgres-backed cache using `MapElitesFileEmbeddingCache` table."""

    experiment_id: UUID
    embedding_model: str
    requested_dimensions: int

    def __post_init__(self) -> None:
        if isinstance(self.experiment_id, UUID):
            return
        self.experiment_id = UUID(str(self.experiment_id))

    def get_many(self, blob_shas: Sequence[str]) -> dict[str, Vector]:
        cleaned = _unique_clean_blob_shas(blob_shas)
        if not cleaned:
            return {}

        dims = int(self.requested_dimensions)
        if dims <= 0:
            raise ValueError("Requested embedding dimensions must be a positive integer.")

        found: dict[str, Vector] = {}
        try:
            with session_scope() as session:
                for batch in _batched(cleaned, 500):
                    base_conditions = [
                        MapElitesFileEmbeddingCache.experiment_id == self.experiment_id,
                        MapElitesFileEmbeddingCache.blob_sha.in_(batch),
                    ]

                    stmt = select(MapElitesFileEmbeddingCache).where(*base_conditions)
                    rows = list(session.execute(stmt).scalars())
                    if not rows:
                        continue

                    # Validate cached rows against experiment-scoped invariants.
                    for row in rows:
                        if str(getattr(row, "embedding_model", "") or "") != str(self.embedding_model):
                            raise ValueError(
                                "File embedding cache entry has an unexpected embedding model; "
                                "reset the DB (dev). "
                                f"(experiment_id={self.experiment_id} blob_sha={row.blob_sha} "
                                f"expected_model={self.embedding_model!r} got_model={row.embedding_model!r})"
                            )
                        if int(getattr(row, "dimensions", 0) or 0) != dims:
                            raise ValueError(
                                "File embedding cache entry has unexpected dimensions; "
                                "reset the DB (dev). "
                                f"(experiment_id={self.experiment_id} blob_sha={row.blob_sha} "
                                f"expected_dims={dims} got_dims={row.dimensions!r})"
                            )
                        vector = tuple(float(v) for v in (row.vector or []))
                        if not vector:
                            raise ValueError(
                                "File embedding cache contains an empty vector; reset the DB (dev). "
                                f"(experiment_id={self.experiment_id} blob_sha={row.blob_sha} dims={dims})"
                            )
                        if len(vector) != dims:
                            raise ValueError(
                                "File embedding cache vector has unexpected dimensions; "
                                "reset the DB (dev). "
                                f"(experiment_id={self.experiment_id} blob_sha={row.blob_sha} "
                                f"expected_dims={dims} got_dims={len(vector)})"
                            )
                        found[str(row.blob_sha)] = vector
        except SQLAlchemyError as exc:
            log.error("Failed to read file embedding cache: {}", exc)
            return {}

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
                    "experiment_id": self.experiment_id,
                    "blob_sha": key,
                    "embedding_model": self.embedding_model,
                    "dimensions": len(vec),
                    "vector": list(vec),
                }
            )

        if not values:
            return

        try:
            with session_scope() as session:
                for batch in _batched(values, 500):
                    stmt = pg_insert(MapElitesFileEmbeddingCache).values(batch)
                    stmt = stmt.on_conflict_do_nothing(
                        index_elements=[
                            "experiment_id",
                            "blob_sha",
                        ],
                    )
                    session.execute(stmt)
        except SQLAlchemyError as exc:
            log.error("Failed to persist file embedding cache: {}", exc)


def build_file_embedding_cache(
    *,
    settings: Settings | None = None,
    backend: str | None = None,
    experiment_id: UUID | str | None = None,
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

    embedding_model = str(s.mapelites_code_embedding_model)
    requested_dimensions = _resolve_requested_dimensions(s)

    if chosen == "memory":
        return InMemoryFileEmbeddingCache(
            embedding_model=embedding_model,
            requested_dimensions=requested_dimensions,
        )
    if chosen == "db":
        if experiment_id is None:
            raise ValueError(
                "Experiment id is required for MAPELITES_FILE_EMBEDDING_CACHE_BACKEND=db. "
                "Pass experiment_id to embed_repository_state / RepositoryStateEmbedder, "
                "or use MAPELITES_FILE_EMBEDDING_CACHE_BACKEND=memory for local runs.",
            )
        return DatabaseFileEmbeddingCache(
            experiment_id=UUID(str(experiment_id)),
            embedding_model=embedding_model,
            requested_dimensions=requested_dimensions,
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


