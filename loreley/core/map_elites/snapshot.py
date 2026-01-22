"""Snapshot persistence helpers for MAP-Elites archives.

Loreley stores MAP-Elites state in Postgres using:
- `map_elites_states`: lightweight per-island metadata (bounds, projection, knobs).
- `map_elites_archive_cells`: occupied archive cells (incremental upserts).
- `map_elites_pca_history`: PCA history entries (incremental upserts).

Only the incremental Postgres storage model is supported.
If the database contains unsupported snapshot payloads, reset it with
`uv run loreley reset-db --yes`.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping, Sequence

import numpy as np
from loguru import logger
from ribs.archives import GridArchive
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError

from loreley.db.base import session_scope
from loreley.db.models import MapElitesArchiveCell, MapElitesPcaHistory, MapElitesState
from .dimension_reduction import PCAProjection, PcaHistoryEntry

log = logger.bind(module="map_elites.snapshot")

Vector = tuple[float, ...]

__all__ = [
    "DatabaseSnapshotStore",
    "SnapshotCellUpsert",
    "SnapshotUpdate",
    "apply_snapshot",
    "array_to_list",
    "deserialize_history",
    "deserialize_projection",
    "ensure_supported_snapshot_meta",
    "purge_island_commit_mappings",
    "restore_archive_entries",
    "serialize_projection",
    "to_list",
]

UNSUPPORTED_META_KEYS = ("archive", "history")


def ensure_supported_snapshot_meta(
    meta: Mapping[str, Any] | None,
    *,
    island_id: str,
) -> None:
    """Fail fast when a stored snapshot payload contains unsupported fields."""

    if not meta:
        return
    for key in UNSUPPORTED_META_KEYS:
        if key in meta:
            raise ValueError(
                "Unsupported MAP-Elites snapshot payload detected; reset the database schema with "
                "`uv run loreley reset-db --yes`. "
                f"(island_id={island_id})"
            )



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

    history_upsert: PcaHistoryEntry | None = None
    history_seen_at: float | None = None

    cell_upsert: SnapshotCellUpsert | None = None
    clear: bool = False

    # Optional knob to keep history restoration bounded without relying on global settings.
    history_limit: int | None = None


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


class DatabaseSnapshotStore:
    """Postgres-backed snapshot store using the incremental MAP-Elites tables."""

    def load(self, island_id: str) -> dict[str, Any] | None:
        """Load a snapshot payload compatible with `apply_snapshot()`."""

        try:
            with session_scope() as session:
                stmt = select(MapElitesState).where(
                    MapElitesState.island_id == island_id,
                )
                state = session.execute(stmt).scalar_one_or_none()
                if not state:
                    return None

                meta = dict(state.snapshot or {})
                ensure_supported_snapshot_meta(meta, island_id=island_id)

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

                return {
                    **meta,
                    "island_id": island_id,
                    "lower_bounds": lower if isinstance(lower, Sequence) else None,
                    "upper_bounds": upper if isinstance(upper, Sequence) else None,
                    "projection": projection_payload,
                    "history": history_entries,
                    "archive": archive_entries,
                }
        except ValueError:
            raise
        except SQLAlchemyError as exc:
            log.error(
                "Failed to load MAP-Elites snapshot for island {}: {}",
                island_id,
                exc,
            )
            return None
        except Exception as exc:  # pragma: no cover - defensive
            log.error(
                "Unexpected error while loading snapshot for island {}: {}",
                island_id,
                exc,
            )
            return None

    def apply_update(
        self,
        island_id: str,
        *,
        update: SnapshotUpdate,
    ) -> None:
        """Persist an incremental update into per-cell/history tables + lightweight metadata."""

        now = float(update.history_seen_at) if update.history_seen_at is not None else time.time()

        try:
            with session_scope() as session:
                stmt = select(MapElitesState).where(
                    MapElitesState.island_id == island_id,
                )
                existing = session.execute(stmt).scalar_one_or_none()
                meta: dict[str, Any] = dict(existing.snapshot or {}) if existing else {}

                ensure_supported_snapshot_meta(meta, island_id=island_id)

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
                            island_id=island_id,
                            snapshot=meta,
                        )
                    )

                if update.clear:
                    session.execute(
                        delete(MapElitesArchiveCell).where(
                            MapElitesArchiveCell.island_id == island_id,
                        )
                    )
                    session.execute(
                        delete(MapElitesPcaHistory).where(
                            MapElitesPcaHistory.island_id == island_id,
                        )
                    )
                    return

                if update.cell_upsert is not None:
                    cell = update.cell_upsert
                    values = {
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

                if update.history_upsert is not None:
                    entry = update.history_upsert
                    values = {
                        "island_id": island_id,
                        "commit_hash": str(entry.commit_hash),
                        "vector": [float(v) for v in entry.vector],
                        "embedding_model": str(entry.embedding_model),
                        "last_seen_at": float(now),
                    }
                    stmt = pg_insert(MapElitesPcaHistory).values(**values)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=[
                            MapElitesPcaHistory.island_id,
                            MapElitesPcaHistory.commit_hash,
                        ],
                        set_={
                            "vector": stmt.excluded.vector,
                            "embedding_model": stmt.excluded.embedding_model,
                            "last_seen_at": stmt.excluded.last_seen_at,
                        },
                    )
                    session.execute(stmt)
        except ValueError:
            raise
        except SQLAlchemyError as exc:
            log.error(
                "Failed to persist MAP-Elites snapshot for island {}: {}",
                island_id,
                exc,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.error(
                "Unexpected error while persisting snapshot for island {}: {}",
                island_id,
                exc,
            )

    def _load_archive_entries(self, session, *, island_id: str) -> list[dict[str, Any]]:
        rows = list(
            session.execute(
                select(MapElitesArchiveCell)
                .where(
                    MapElitesArchiveCell.island_id == island_id,
                )
                .order_by(MapElitesArchiveCell.cell_index.asc())
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
        effective_limit = max(0, int(limit or 0))
        stmt = (
            select(MapElitesPcaHistory)
            .where(
                MapElitesPcaHistory.island_id == island_id,
            )
            .order_by(
                MapElitesPcaHistory.last_seen_at.desc(),
                MapElitesPcaHistory.commit_hash.asc(),
            )
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
                    "embedding_model": str(getattr(row, "embedding_model", "") or ""),
                }
            )
        return payload


def apply_snapshot(
    *,
    state: Any,
    snapshot: Mapping[str, Any],
    island_id: str,
    commit_to_island: dict[str, str],
) -> None:
    """Apply a previously serialised snapshot onto an island state."""

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


def deserialize_history(payload: Sequence[Mapping[str, Any]]) -> tuple[PcaHistoryEntry, ...]:
    """Rebuild PCA history from a JSON-compatible payload."""

    history: list[PcaHistoryEntry] = []
    for item in payload:
        vector_values = item.get("vector") or []
        vector = tuple(float(value) for value in vector_values)
        history.append(
            PcaHistoryEntry(
                commit_hash=str(item.get("commit_hash", "")),
                vector=vector,
                embedding_model=str(item.get("embedding_model", "") or ""),
            )
        )
    return tuple(history)


def serialize_projection(projection: PCAProjection | None) -> dict[str, Any] | None:
    """Convert a `PCAProjection` into a JSON-compatible dict."""

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
    """Rebuild a `PCAProjection` instance from JSON-compatible data."""

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
                "Skipping snapshot archive entry due to incompatible solution_dim (island={} expected={} got={})",
                island_id,
                int(expected_solution_dim),
                int(solution_batch.shape[1]),
            )
            continue
        if expected_measures_dim is not None and measures_batch.shape[1] != int(expected_measures_dim):
            log.warning(
                "Skipping snapshot archive entry due to incompatible measures dimensionality (island={} expected={} got={})",
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
    """Remove any commit-to-island mappings that point at `island_id`."""

    for commit, mapped_island in tuple(commit_to_island.items()):
        if mapped_island == island_id:
            commit_to_island.pop(commit, None)


def array_to_list(values: Any) -> list[float]:
    """Convert numpy arrays or scalar-like values into plain float lists."""

    if values is None:
        return []
    if isinstance(values, np.ndarray):
        try:
            return [float(v) for v in values.tolist()]
        except Exception:
            return []
    if isinstance(values, (list, tuple)):
        return [float(v) for v in values if v is not None]
    try:
        return [float(values)]
    except Exception:
        return []


def to_list(values: Any) -> list[Any]:
    """Convert numpy arrays or scalar-like values into a Python list."""

    if values is None:
        return []
    if isinstance(values, np.ndarray):
        try:
            return list(values.tolist())
        except Exception:
            return []
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values]

