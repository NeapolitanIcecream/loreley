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
from typing import Any, Mapping, Sequence

import numpy as np
from loguru import logger
from ribs.archives import GridArchive
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from loreley.db.base import session_scope
from loreley.db.models import MapElitesState
from .dimension_reduction import PCAProjection, PenultimateEmbedding

log = logger.bind(module="map_elites.snapshot")

Vector = tuple[float, ...]

__all__ = [
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


@dataclass(slots=True)
class NullSnapshotBackend(SnapshotBackend):
    """No-op backend used when snapshot persistence is disabled."""

    def load(self, island_id: str) -> dict[str, Any] | None:
        return None

    def save(self, island_id: str, snapshot: Mapping[str, Any]) -> None:
        # Intentionally ignore all writes.
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
                if not state or not state.snapshot:
                    return None
                return dict(state.snapshot)
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
    metadata_entries = to_list(data.get("metadata"))
    timestamps = to_list(data.get("timestamp"))

    entries: list[dict[str, Any]] = []
    for idx, cell_index in enumerate(indices):
        entry = {
            "index": int(cell_index),
            "objective": float(objectives[idx]) if idx < len(objectives) else 0.0,
            "measures": array_to_list(measures[idx]) if idx < len(measures) else [],
            "solution": array_to_list(solutions[idx]) if idx < len(solutions) else [],
            "commit_hash": str(commit_hashes[idx]) if idx < len(commit_hashes) else "",
            "metadata": (
                _coerce_metadata(metadata_entries[idx])
                if idx < len(metadata_entries)
                else {}
            ),
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
    for entry in entries:
        solution_values = array_to_list(entry.get("solution"))
        measures_values = array_to_list(entry.get("measures"))
        if not solution_values or not measures_values:
            continue

        solution = np.asarray(solution_values, dtype=np.float64)
        measures = np.asarray(measures_values, dtype=np.float64)
        solution_batch = solution.reshape(1, -1)
        measures_batch = measures.reshape(1, -1)
        objective = np.asarray(
            [float(entry.get("objective", 0.0))],
            dtype=np.float64,
        )
        commit_hash = str(entry.get("commit_hash", ""))
        metadata = _coerce_metadata(entry.get("metadata"))
        timestamp_value = float(entry.get("timestamp", 0.0))

        archive.add(
            solution_batch,
            objective,
            measures_batch,
            commit_hash=np.asarray([commit_hash], dtype=object),
            metadata=np.asarray([metadata], dtype=object),
            timestamp=np.asarray([timestamp_value], dtype=np.float64),
        )

        stored_index = entry.get("index")
        if stored_index is not None:
            cell_index = int(stored_index)
        else:
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


def _coerce_metadata(payload: Any) -> dict[str, Any]:
    """Ensure metadata payloads are plain dicts."""

    if isinstance(payload, Mapping):
        return dict(payload)
    return {}


