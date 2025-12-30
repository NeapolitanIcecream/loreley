"""Snapshot serialisation and persistence helpers for MAP-Elites archives.

This module focuses on **how** MAP-Elites snapshots are represented,
serialised and stored, while callers such as ``MapElitesManager`` decide
**when** a snapshot should be loaded or persisted.

The design keeps the surface area small and decoupled:

- Pure helper functions handle conversion between in-memory structures
  (PCA history/projection, ``GridArchive`` contents) and JSON-compatible
  payloads.
- A pluggable ``SnapshotBackend`` abstraction encapsulates the storage
  mechanism (database, no-op, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Any, Mapping, Sequence

import numpy as np
from loguru import logger
from ribs.archives import GridArchive
from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from loreley.config import get_settings
from loreley.db.base import session_scope
from loreley.db.models import MapElitesArchiveCell, MapElitesPcaHistory, MapElitesState
from .dimension_reduction import PCAProjection, PenultimateEmbedding

log = logger.bind(module="map_elites.snapshot")

Vector = tuple[float, ...]

__all__ = [
    "SnapshotCellUpsert",
    "SnapshotUpdate",
    "SnapshotBackend",
    "NullSnapshotBackend",
    "DatabaseSnapshotBackend",
    "build_snapshot_backend",
    "build_snapshot",
    "apply_snapshot",
    "serialize_history",
    "deserialize_history",
    "serialize_projection",
    "deserialize_projection",
    "serialize_archive",
    "restore_archive_entries",
    "purge_island_commit_mappings",
    "array_to_list",
    "to_list",
]


@dataclass(slots=True, frozen=True)
class SnapshotCellUpsert:
    """Incremental upsert payload for a single archive cell."""

    cell_index: int
    objective: float
    measures: Vector
    solution: Vector
    commit_hash: str
    timestamp: float


@dataclass(slots=True)
class SnapshotUpdate:
    """Incremental snapshot update applied to a persisted island state."""

    lower_bounds: Sequence[float] | None = None
    upper_bounds: Sequence[float] | None = None
    projection: PCAProjection | None = None

    history_upsert: PenultimateEmbedding | None = None
    history_seen_at: float | None = None

    cell_upsert: SnapshotCellUpsert | None = None
    clear: bool = False

    # Optional knob to keep history restoration bounded without relying on global settings.
    history_limit: int | None = None


class SnapshotBackend(ABC):
    """Abstract storage backend for island snapshots.

    Callers provide the *decision* of when to save/load snapshots, while
    backends encapsulate how those snapshots are persisted.
    """

    @abstractmethod
    def load(self, island_id: str) -> dict[str, Any] | None:  # pragma: no cover - interface
        """Load a snapshot for the given island or return ``None``."""

    @abstractmethod
    def save(self, island_id: str, snapshot: Mapping[str, Any]) -> None:  # pragma: no cover - interface
        """Persist a snapshot for the given island."""

    def apply_update(  # pragma: no cover - thin default
        self,
        island_id: str,
        *,
        state: Any,
        update: SnapshotUpdate | None = None,
    ) -> None:
        """Persist an incremental update.

        Default implementation falls back to serialising the full snapshot and
        delegating to ``save``. Storage backends can override this method to
        implement incremental persistence without rewriting large JSON blobs.
        """

        snapshot = build_snapshot(island_id, state)
        self.save(island_id, snapshot)


@dataclass(slots=True)
class NullSnapshotBackend(SnapshotBackend):
    """No-op backend used when snapshot persistence is disabled."""

    def load(self, island_id: str) -> dict[str, Any] | None:
        return None

    def save(self, island_id: str, snapshot: Mapping[str, Any]) -> None:
        # Intentionally ignore all writes.
        return None

    def apply_update(
        self,
        island_id: str,
        *,
        state: Any,
        update: SnapshotUpdate | None = None,
    ) -> None:
        # Intentionally ignore all writes, including full-snapshot fallbacks.
        return None


@dataclass(slots=True)
class DatabaseSnapshotBackend(SnapshotBackend):
    """Database-backed snapshot storage using the ``MapElitesState`` table."""

    experiment_id: Any

    def load(self, island_id: str) -> dict[str, Any] | None:
        try:
            with session_scope() as session:
                stmt = select(MapElitesState).where(
                    MapElitesState.experiment_id == self.experiment_id,
                    MapElitesState.island_id == island_id,
                )
                state = session.execute(stmt).scalar_one_or_none()
                if not state:
                    return None

                meta = dict(state.snapshot or {})
                meta = self._maybe_migrate_legacy_snapshot(session, state, meta, island_id=island_id)

                schema_version = _coerce_int(meta.get("schema_version"), default=1)
                if schema_version < 2:
                    # Legacy mode: return the stored payload as-is.
                    return meta or None

                # Assemble a snapshot payload compatible with `apply_snapshot()`.
                lower = meta.get("lower_bounds")
                upper = meta.get("upper_bounds")
                projection_payload = meta.get("projection")
                history_limit = _coerce_int(meta.get("history_limit"), default=0) or None

                archive_entries = self._load_archive_entries(session, island_id=island_id)
                history_entries = self._load_history_entries(
                    session,
                    island_id=island_id,
                    limit=history_limit,
                )

                payload: dict[str, Any] = {
                    **meta,
                    "island_id": island_id,
                    "lower_bounds": lower if isinstance(lower, Sequence) else None,
                    "upper_bounds": upper if isinstance(upper, Sequence) else None,
                    "projection": projection_payload,
                    "history": history_entries,
                    "archive": archive_entries,
                }
                return payload
        except SQLAlchemyError as exc:
            log.error(
                "Failed to load MAP-Elites snapshot for experiment {} island {}: {}",
                self.experiment_id,
                island_id,
                exc,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.error(
                "Unexpected error while loading snapshot for experiment {} island {}: {}",
                self.experiment_id,
                island_id,
                exc,
            )
        return None

    def save(self, island_id: str, snapshot: Mapping[str, Any]) -> None:
        try:
            with session_scope() as session:
                stmt = select(MapElitesState).where(
                    MapElitesState.experiment_id == self.experiment_id,
                    MapElitesState.island_id == island_id,
                )
                existing = session.execute(stmt).scalar_one_or_none()
                if existing:
                    existing.snapshot = dict(snapshot)
                else:
                    session.add(
                        MapElitesState(
                            experiment_id=self.experiment_id,
                            island_id=island_id,
                            snapshot=dict(snapshot),
                        )
                    )
        except SQLAlchemyError as exc:
            log.error(
                "Failed to persist MAP-Elites snapshot for experiment {} island {}: {}",
                self.experiment_id,
                island_id,
                exc,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.error(
                "Unexpected error while persisting snapshot for experiment {} island {}: {}",
                self.experiment_id,
                island_id,
                exc,
            )

    def apply_update(
        self,
        island_id: str,
        *,
        state: Any,
        update: SnapshotUpdate | None = None,
    ) -> None:
        """Persist an incremental update into per-cell/history tables + lightweight metadata."""

        if update is None:
            # Backwards-compatible fallback.
            return super().apply_update(island_id, state=state, update=update)

        now = float(update.history_seen_at) if update.history_seen_at is not None else time.time()

        try:
            with session_scope() as session:
                # Upsert lightweight metadata row.
                stmt = select(MapElitesState).where(
                    MapElitesState.experiment_id == self.experiment_id,
                    MapElitesState.island_id == island_id,
                )
                existing = session.execute(stmt).scalar_one_or_none()
                meta: dict[str, Any] = dict(existing.snapshot or {}) if existing else {}

                # Ensure we do not keep legacy large fields around.
                meta.pop("archive", None)
                meta.pop("history", None)

                meta["schema_version"] = 2
                meta["storage_backend"] = "cells_history_v1"
                meta["last_update_at"] = now

                if update.history_limit is not None:
                    meta["history_limit"] = int(update.history_limit)

                if update.lower_bounds is not None:
                    meta["lower_bounds"] = [float(v) for v in update.lower_bounds]
                if update.upper_bounds is not None:
                    meta["upper_bounds"] = [float(v) for v in update.upper_bounds]

                # Projection updates are frequent but small; keep them in metadata JSON.
                meta["projection"] = serialize_projection(update.projection)

                if existing:
                    existing.snapshot = meta
                else:
                    session.add(
                        MapElitesState(
                            experiment_id=self.experiment_id,
                            island_id=island_id,
                            snapshot=meta,
                        )
                    )

                if update.clear:
                    session.execute(
                        delete(MapElitesArchiveCell).where(
                            MapElitesArchiveCell.experiment_id == self.experiment_id,
                            MapElitesArchiveCell.island_id == island_id,
                        )
                    )
                    session.execute(
                        delete(MapElitesPcaHistory).where(
                            MapElitesPcaHistory.experiment_id == self.experiment_id,
                            MapElitesPcaHistory.island_id == island_id,
                        )
                    )
                    return

                # Incremental cell upsert (only when a commit improved a cell).
                if update.cell_upsert is not None:
                    cell = update.cell_upsert
                    values = {
                        "experiment_id": self.experiment_id,
                        "island_id": island_id,
                        "cell_index": int(cell.cell_index),
                        "commit_hash": str(cell.commit_hash),
                        "objective": float(cell.objective),
                        "measures": [float(v) for v in cell.measures],
                        "solution": [float(v) for v in cell.solution],
                        "timestamp": float(cell.timestamp),
                    }
                    stmt = pg_insert(MapElitesArchiveCell).values(**values)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=[
                            MapElitesArchiveCell.__table__.c.experiment_id,
                            MapElitesArchiveCell.__table__.c.island_id,
                            MapElitesArchiveCell.__table__.c.cell_index,
                        ],
                        set_={
                            "commit_hash": stmt.excluded.commit_hash,
                            "objective": stmt.excluded.objective,
                            "measures": stmt.excluded.measures,
                            "solution": stmt.excluded.solution,
                            "timestamp": stmt.excluded.timestamp,
                        },
                    )
                    session.execute(stmt)

                # Incremental history upsert (idempotent per commit hash).
                if update.history_upsert is not None:
                    entry = update.history_upsert
                    values = {
                        "experiment_id": self.experiment_id,
                        "island_id": island_id,
                        "commit_hash": str(entry.commit_hash),
                        "vector": [float(v) for v in entry.vector],
                        "code_dimensions": int(entry.code_dimensions),
                        "summary_dimensions": int(entry.summary_dimensions),
                        "code_model": entry.code_model,
                        "summary_model": entry.summary_model,
                        "summary_embedding_model": entry.summary_embedding_model,
                        "last_seen_at": float(now),
                    }
                    stmt = pg_insert(MapElitesPcaHistory).values(**values)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=[
                            MapElitesPcaHistory.experiment_id,
                            MapElitesPcaHistory.island_id,
                            MapElitesPcaHistory.commit_hash,
                        ],
                        set_={
                            "vector": stmt.excluded.vector,
                            "code_dimensions": stmt.excluded.code_dimensions,
                            "summary_dimensions": stmt.excluded.summary_dimensions,
                            "code_model": stmt.excluded.code_model,
                            "summary_model": stmt.excluded.summary_model,
                            "summary_embedding_model": stmt.excluded.summary_embedding_model,
                            "last_seen_at": stmt.excluded.last_seen_at,
                        },
                    )
                    session.execute(stmt)
        except SQLAlchemyError as exc:
            log.error(
                "Failed to persist incremental MAP-Elites snapshot for experiment {} island {}: {}",
                self.experiment_id,
                island_id,
                exc,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.error(
                "Unexpected error while persisting incremental snapshot for experiment {} island {}: {}",
                self.experiment_id,
                island_id,
                exc,
            )

    def _load_archive_entries(self, session, *, island_id: str) -> list[dict[str, Any]]:
        rows = list(
            session.execute(
                select(MapElitesArchiveCell).where(
                    MapElitesArchiveCell.experiment_id == self.experiment_id,
                    MapElitesArchiveCell.island_id == island_id,
                )
            )
            .scalars()
            .all()
        )
        entries: list[dict[str, Any]] = []
        for row in rows:
            entries.append(
                {
                    "index": int(row.cell_index),
                    "objective": float(row.objective or 0.0),
                    "measures": [float(v) for v in (row.measures or [])],
                    "solution": [float(v) for v in (row.solution or [])],
                    "commit_hash": str(row.commit_hash or ""),
                    "timestamp": float(row.timestamp or 0.0),
                }
            )
        return entries

    def _load_history_entries(
        self,
        session,
        *,
        island_id: str,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        effective_limit = limit
        if effective_limit is None:
            settings = get_settings()
            min_fit = max(
                2,
                int(settings.mapelites_dimensionality_min_fit_samples),
                int(settings.mapelites_feature_normalization_warmup_samples),
            )
            effective_limit = max(
                min_fit,
                int(settings.mapelites_dimensionality_history_size),
            )
        effective_limit = max(0, int(effective_limit or 0))

        stmt = (
            select(MapElitesPcaHistory)
            .where(
                MapElitesPcaHistory.experiment_id == self.experiment_id,
                MapElitesPcaHistory.island_id == island_id,
            )
            .order_by(MapElitesPcaHistory.last_seen_at.desc())
        )
        if effective_limit:
            stmt = stmt.limit(effective_limit)
        rows = list(session.execute(stmt).scalars().all())
        # Reverse to restore oldest->newest ordering expected by `DimensionReducer`.
        rows.reverse()
        payload: list[dict[str, Any]] = []
        for row in rows:
            payload.append(
                {
                    "commit_hash": str(row.commit_hash or ""),
                    "vector": [float(v) for v in (row.vector or [])],
                    "code_dimensions": int(row.code_dimensions or 0),
                    "summary_dimensions": int(row.summary_dimensions or 0),
                    "code_model": row.code_model,
                    "summary_model": row.summary_model,
                    "summary_embedding_model": row.summary_embedding_model,
                }
            )
        return payload

    def _maybe_migrate_legacy_snapshot(
        self,
        session,
        state_row: MapElitesState,
        meta: dict[str, Any],
        *,
        island_id: str,
    ) -> dict[str, Any]:
        schema_version = _coerce_int(meta.get("schema_version"), default=1)
        if schema_version >= 2:
            return meta

        archive_payload = meta.get("archive")
        history_payload = meta.get("history")

        has_archive = isinstance(archive_payload, list) and bool(archive_payload)
        has_history = isinstance(history_payload, list) and bool(history_payload)
        if not has_archive and not has_history:
            return meta

        now = time.time()
        migrated_cells = 0
        migrated_history = 0

        if isinstance(archive_payload, list):
            for entry in archive_payload:
                if not isinstance(entry, Mapping):
                    continue
                stored_index = entry.get("index")
                if stored_index is None:
                    continue
                try:
                    cell_index = int(stored_index)
                except (TypeError, ValueError):
                    continue

                measures = array_to_list(entry.get("measures"))
                solution = array_to_list(entry.get("solution"))
                if not measures or not solution:
                    # Skip malformed rows; restore logic similarly expects both.
                    continue

                values = {
                    "experiment_id": self.experiment_id,
                    "island_id": island_id,
                    "cell_index": cell_index,
                    "commit_hash": str(entry.get("commit_hash", "")),
                    "objective": float(entry.get("objective", 0.0)),
                    "measures": [float(v) for v in measures],
                    "solution": [float(v) for v in solution],
                    "timestamp": float(entry.get("timestamp", 0.0)),
                }
                stmt = pg_insert(MapElitesArchiveCell).values(**values)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[
                        MapElitesArchiveCell.__table__.c.experiment_id,
                        MapElitesArchiveCell.__table__.c.island_id,
                        MapElitesArchiveCell.__table__.c.cell_index,
                    ],
                    set_={
                        "commit_hash": stmt.excluded.commit_hash,
                        "objective": stmt.excluded.objective,
                        "measures": stmt.excluded.measures,
                        "solution": stmt.excluded.solution,
                        "timestamp": stmt.excluded.timestamp,
                    },
                )
                session.execute(stmt)
                migrated_cells += 1

        if isinstance(history_payload, list):
            n = len(history_payload)
            for idx, entry in enumerate(history_payload):
                if not isinstance(entry, Mapping):
                    continue
                commit_hash = str(entry.get("commit_hash", "")).strip()
                if not commit_hash:
                    continue
                vec_values = entry.get("vector") or []
                if not isinstance(vec_values, (list, tuple)):
                    vec_values = []
                last_seen_at = now - float(max(0, (n - 1) - idx))
                values = {
                    "experiment_id": self.experiment_id,
                    "island_id": island_id,
                    "commit_hash": commit_hash,
                    "vector": [float(v) for v in vec_values],
                    "code_dimensions": int(entry.get("code_dimensions", 0)),
                    "summary_dimensions": int(entry.get("summary_dimensions", 0)),
                    "code_model": entry.get("code_model"),
                    "summary_model": entry.get("summary_model"),
                    "summary_embedding_model": entry.get("summary_embedding_model"),
                    "last_seen_at": float(last_seen_at),
                }
                stmt = pg_insert(MapElitesPcaHistory).values(**values)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[
                        MapElitesPcaHistory.experiment_id,
                        MapElitesPcaHistory.island_id,
                        MapElitesPcaHistory.commit_hash,
                    ],
                    set_={
                        "vector": stmt.excluded.vector,
                        "code_dimensions": stmt.excluded.code_dimensions,
                        "summary_dimensions": stmt.excluded.summary_dimensions,
                        "code_model": stmt.excluded.code_model,
                        "summary_model": stmt.excluded.summary_model,
                        "summary_embedding_model": stmt.excluded.summary_embedding_model,
                        "last_seen_at": stmt.excluded.last_seen_at,
                    },
                )
                session.execute(stmt)
                migrated_history += 1

        cleaned = dict(meta)
        cleaned.pop("archive", None)
        cleaned.pop("history", None)
        cleaned["schema_version"] = 2
        cleaned["storage_backend"] = "cells_history_v1"
        cleaned["last_update_at"] = now
        cleaned.setdefault("last_migrated_at", now)
        state_row.snapshot = cleaned

        log.info(
            "Migrated legacy MAP-Elites snapshot to incremental tables (experiment={} island={} cells={} history={})",
            self.experiment_id,
            island_id,
            migrated_cells,
            migrated_history,
        )
        return cleaned


def build_snapshot_backend(experiment_id: Any | None) -> SnapshotBackend:
    """Factory that picks the appropriate snapshot backend.

    - When ``experiment_id`` is ``None``, snapshot persistence is disabled and
      a ``NullSnapshotBackend`` is returned.
    - Otherwise, snapshots are stored in the database via ``MapElitesState``.
    """

    if experiment_id is None:
        return NullSnapshotBackend()
    return DatabaseSnapshotBackend(experiment_id=experiment_id)


def build_snapshot(island_id: str, state: Any) -> dict[str, Any]:
    """Serialise an island state into a JSON-compatible snapshot payload.

    The ``state`` object is expected to expose the attributes used here
    (``lower_bounds``, ``upper_bounds``, ``history``, ``projection``,
    and ``archive``) but does not need to be a specific class; this makes
    the function reusable in tests and alternative implementations.
    """

    return {
        "island_id": island_id,
        "lower_bounds": np.asarray(getattr(state, "lower_bounds")).tolist(),
        "upper_bounds": np.asarray(getattr(state, "upper_bounds")).tolist(),
        "history": serialize_history(getattr(state, "history")),
        "projection": serialize_projection(getattr(state, "projection")),
        "archive": serialize_archive(getattr(state, "archive")),
    }


def apply_snapshot(
    *,
    state: Any,
    snapshot: Mapping[str, Any],
    island_id: str,
    commit_to_island: dict[str, str],
) -> None:
    """Apply a previously serialised snapshot onto an island state.

    This updates feature bounds, PCA history/projection, restores archive
    entries, and rebuilds commit-to-cell mappings.
    """

    lower_bounds = snapshot.get("lower_bounds")
    upper_bounds = snapshot.get("upper_bounds")
    if isinstance(lower_bounds, Sequence):
        state.lower_bounds = np.asarray(lower_bounds, dtype=np.float64)
    if isinstance(upper_bounds, Sequence):
        state.upper_bounds = np.asarray(upper_bounds, dtype=np.float64)

    history_payload = snapshot.get("history") or []
    if history_payload:
        state.history = deserialize_history(history_payload)

    projection_payload = snapshot.get("projection")
    if projection_payload:
        state.projection = deserialize_projection(projection_payload)

    state.index_to_commit.clear()
    state.commit_to_index.clear()
    purge_island_commit_mappings(commit_to_island, island_id)

    archive_entries = snapshot.get("archive") or []
    if archive_entries:
        restore_archive_entries(state, archive_entries, island_id, commit_to_island)


def serialize_history(history: Sequence[PenultimateEmbedding]) -> list[dict[str, Any]]:
    """Convert PCA history into a JSON-compatible list of dicts."""

    payload: list[dict[str, Any]] = []
    for entry in history:
        payload.append(
            {
                "commit_hash": entry.commit_hash,
                "vector": [float(value) for value in entry.vector],
                "code_dimensions": entry.code_dimensions,
                "summary_dimensions": entry.summary_dimensions,
                "code_model": entry.code_model,
                "summary_model": entry.summary_model,
                "summary_embedding_model": entry.summary_embedding_model,
            }
        )
    return payload


def deserialize_history(
    payload: Sequence[Mapping[str, Any]],
) -> tuple[PenultimateEmbedding, ...]:
    """Rebuild PCA history from a JSON-compatible payload."""

    history: list[PenultimateEmbedding] = []
    for item in payload:
        vector_values = item.get("vector") or []
        vector = tuple(float(value) for value in vector_values)
        history.append(
            PenultimateEmbedding(
                commit_hash=str(item.get("commit_hash", "")),
                vector=vector,
                code_dimensions=int(item.get("code_dimensions", 0)),
                summary_dimensions=int(item.get("summary_dimensions", 0)),
                code_model=item.get("code_model"),
                summary_model=item.get("summary_model"),
                summary_embedding_model=item.get("summary_embedding_model"),
            )
        )
    return tuple(history)


def serialize_projection(projection: PCAProjection | None) -> dict[str, Any] | None:
    """Convert a ``PCAProjection`` into a JSON-compatible dict."""

    if not projection:
        return None
    return {
        "feature_count": projection.feature_count,
        "components": [[float(value) for value in row] for row in projection.components],
        "mean": [float(value) for value in projection.mean],
        "explained_variance": [float(value) for value in projection.explained_variance],
        "explained_variance_ratio": [
            float(value) for value in projection.explained_variance_ratio
        ],
        "sample_count": projection.sample_count,
        "fitted_at": projection.fitted_at,
        "whiten": projection.whiten,
    }


def deserialize_projection(payload: Mapping[str, Any] | None) -> PCAProjection | None:
    """Rebuild a ``PCAProjection`` instance from JSON-compatible data."""

    if not payload:
        return None
    components_payload = payload.get("components") or []
    components = tuple(tuple(float(value) for value in row) for row in components_payload)
    mean_raw = payload.get("mean") or []
    mean = tuple(float(value) for value in mean_raw)
    explained_variance_raw = payload.get("explained_variance") or []
    explained_variance = tuple(float(value) for value in explained_variance_raw)
    explained_raw = payload.get("explained_variance_ratio") or []
    explained = tuple(float(value) for value in explained_raw)
    return PCAProjection(
        feature_count=int(payload.get("feature_count", len(mean))),
        components=components,
        mean=mean,
        explained_variance=explained_variance,
        explained_variance_ratio=explained,
        sample_count=int(payload.get("sample_count", 0)),
        fitted_at=float(payload.get("fitted_at", 0.0)),
        whiten=bool(payload.get("whiten", False)),
    )


def serialize_archive(archive: GridArchive) -> list[dict[str, Any]]:
    """Serialise a ``GridArchive`` into a list of JSON-compatible entries."""

    data = archive.data()
    if archive.empty or not isinstance(data, dict):
        return []

    indices = to_list(data.get("index"))
    if not indices:
        return []

    objectives = to_list(data.get("objective"))
    measures = to_list(data.get("measures"))
    solutions = to_list(data.get("solution"))
    commit_hashes = to_list(data.get("commit_hash"))
    timestamps = to_list(data.get("timestamp"))

    entries: list[dict[str, Any]] = []
    for idx, cell_index in enumerate(indices):
        entry = {
            "index": int(cell_index),
            "objective": float(objectives[idx]) if idx < len(objectives) else 0.0,
            "measures": array_to_list(measures[idx]) if idx < len(measures) else [],
            "solution": array_to_list(solutions[idx]) if idx < len(solutions) else [],
            "commit_hash": str(commit_hashes[idx]) if idx < len(commit_hashes) else "",
            "timestamp": float(timestamps[idx]) if idx < len(timestamps) else 0.0,
        }
        entries.append(entry)
    return entries


def restore_archive_entries(
    state: Any,
    entries: Sequence[Mapping[str, Any]],
    island_id: str,
    commit_to_island: dict[str, str],
) -> None:
    """Restore archive entries and commit mappings from snapshot data."""

    archive: GridArchive = getattr(state, "archive")
    expected_solution_dim = getattr(archive, "solution_dim", None)
    expected_measures_dim = len(getattr(archive, "dims", ())) or None
    expected_cell_count = None
    if expected_measures_dim is not None:
        try:
            expected_cell_count = int(np.prod(getattr(archive, "dims", ())))
        except Exception:  # pragma: no cover - defensive
            expected_cell_count = None
    for entry in entries:
        solution_values = array_to_list(entry.get("solution"))
        measures_values = array_to_list(entry.get("measures"))
        if not solution_values or not measures_values:
            continue

        solution = np.asarray(solution_values, dtype=np.float64)
        measures = np.asarray(measures_values, dtype=np.float64)
        solution_batch = solution.reshape(1, -1)
        measures_batch = measures.reshape(1, -1)

        if expected_solution_dim is not None and solution_batch.shape[1] != int(expected_solution_dim):
            log.warning(
                "Skipping snapshot archive entry due to incompatible solution_dim "
                "(experiment={} island={} expected={} got={})",
                getattr(getattr(archive, "experiment_id", None), "hex", None) or "?",
                island_id,
                int(expected_solution_dim),
                int(solution_batch.shape[1]),
            )
            continue
        if expected_measures_dim is not None and measures_batch.shape[1] != int(expected_measures_dim):
            log.warning(
                "Skipping snapshot archive entry due to incompatible measures dimensionality "
                "(island={} expected={} got={})",
                island_id,
                int(expected_measures_dim),
                int(measures_batch.shape[1]),
            )
            continue

        objective = np.asarray(
            [float(entry.get("objective", 0.0))],
            dtype=np.float64,
        )
        commit_hash = str(entry.get("commit_hash", ""))
        timestamp_value = float(entry.get("timestamp", 0.0))

        try:
            archive.add(
                solution_batch,
                objective,
                measures_batch,
                commit_hash=np.asarray([commit_hash], dtype=object),
                timestamp=np.asarray([timestamp_value], dtype=np.float64),
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "Failed to restore snapshot entry into GridArchive (island={} commit_hash={}): {}",
                island_id,
                commit_hash,
                exc,
            )
            continue

        stored_index = entry.get("index")
        if stored_index is not None:
            try:
                cell_index = int(stored_index)
            except (TypeError, ValueError):
                cell_index = int(np.asarray(archive.index_of(measures_batch)).item())
        else:
            cell_index = int(np.asarray(archive.index_of(measures_batch)).item())
        if expected_cell_count is not None and (cell_index < 0 or cell_index >= expected_cell_count):
            cell_index = int(np.asarray(archive.index_of(measures_batch)).item())

        state.index_to_commit[cell_index] = commit_hash
        if commit_hash:
            state.commit_to_index[commit_hash] = cell_index
            commit_to_island[commit_hash] = island_id


def purge_island_commit_mappings(commit_to_island: dict[str, str], island_id: str) -> None:
    """Remove any commit-to-island mappings that point at ``island_id``."""

    for commit, mapped_island in tuple(commit_to_island.items()):
        if mapped_island == island_id:
            commit_to_island.pop(commit, None)


def array_to_list(values: Any) -> list[float]:
    """Convert numpy arrays or scalar-like values into plain float lists."""

    if values is None:
        return []
    if isinstance(values, np.ndarray):
        return values.astype(float).tolist()
    if isinstance(values, (list, tuple)):
        return [float(value) for value in values]
    return [float(values)]


def to_list(values: Any) -> list[Any]:
    """Normalise numpy arrays and scalars into Python lists."""

    if values is None:
        return []
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    return [values]


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


