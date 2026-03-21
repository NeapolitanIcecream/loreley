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
from sqlalchemy.orm import Session

from loreley.db.base import session_scope
from loreley.db.models import MapElitesArchiveCell, MapElitesPcaHistory, MapElitesState
from .dimension_reduction import PCAProjection, PcaHistoryEntry
from .types import compact_solution, materialize_solution

log = logger.bind(module="map_elites.snapshot")

Vector = tuple[float, ...]

__all__ = [
    "DatabaseSnapshotStore",
    "SnapshotStoreError",
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


class SnapshotStoreError(RuntimeError):
    """Raised when MAP-Elites snapshot state cannot be loaded or persisted."""


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
    samples_since_fit: int | None = None
    history_limit: int | None = None

    history_upsert: PcaHistoryEntry | None = None
    history_seen_at: float | None = None

    cell_upsert: SnapshotCellUpsert | None = None
    archive_replace: Sequence[SnapshotCellUpsert] | None = None
    clear: bool = False


class DatabaseSnapshotStore:
    """Postgres-backed snapshot store using the incremental MAP-Elites tables."""

    @staticmethod
    def _raise_store_error(*, action: str, island_id: str, exc: Exception) -> None:
        if isinstance(exc, SQLAlchemyError):
            message = f"Failed to {action} MAP-Elites snapshot for island {island_id}: {exc}"
        else:
            message = f"Unexpected error while {action} snapshot for island {island_id}: {exc}"
        log.exception(message)
        raise SnapshotStoreError(message) from exc

    def load(self, island_id: str, *, history_limit: int | None = None) -> dict[str, Any] | None:
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
            self._raise_store_error(action="load", island_id=island_id, exc=exc)
        except Exception as exc:  # pragma: no cover - defensive
            self._raise_store_error(action="loading", island_id=island_id, exc=exc)

    def apply_update(
        self,
        island_id: str,
        *,
        update: SnapshotUpdate,
        session: Session | None = None,
    ) -> None:
        """Persist an incremental update into per-cell/history tables + lightweight metadata."""

        now = float(update.history_seen_at) if update.history_seen_at is not None else time.time()

        if session is not None:
            in_nested = bool(getattr(session, "in_nested_transaction", lambda: False)())
            if in_nested:
                try:
                    self._apply_update_in_session(
                        session,
                        island_id=island_id,
                        update=update,
                        now=now,
                    )
                except ValueError:
                    raise
                except SQLAlchemyError as exc:
                    self._raise_store_error(action="persist", island_id=island_id, exc=exc)
                except Exception as exc:  # pragma: no cover - defensive
                    self._raise_store_error(action="persisting", island_id=island_id, exc=exc)
            else:
                self._apply_update_within_savepoint(
                    session,
                    island_id=island_id,
                    update=update,
                    now=now,
                )
            return

        try:
            with session_scope() as owned_session:
                self._apply_update_in_session(
                    owned_session,
                    island_id=island_id,
                    update=update,
                    now=now,
                )
        except ValueError:
            raise
        except SQLAlchemyError as exc:
            self._raise_store_error(action="persist", island_id=island_id, exc=exc)
        except Exception as exc:  # pragma: no cover - defensive
            self._raise_store_error(action="persisting", island_id=island_id, exc=exc)

    def _apply_update_within_savepoint(
        self,
        session: Session,
        *,
        island_id: str,
        update: SnapshotUpdate,
        now: float,
    ) -> None:
        """Apply one update inside a savepoint so batch ingestion can continue."""

        try:
            with session.begin_nested():
                self._apply_update_in_session(
                    session,
                    island_id=island_id,
                    update=update,
                    now=now,
                )
        except ValueError:
            raise
        except SQLAlchemyError as exc:
            self._raise_store_error(action="persist", island_id=island_id, exc=exc)
        except Exception as exc:  # pragma: no cover - defensive
            self._raise_store_error(action="persisting", island_id=island_id, exc=exc)

    def _apply_update_in_session(
        self,
        session: Session,
        *,
        island_id: str,
        update: SnapshotUpdate,
        now: float,
    ) -> None:
        stmt = select(MapElitesState).where(
            MapElitesState.island_id == island_id,
        )
        existing = session.execute(stmt).scalar_one_or_none()
        meta: dict[str, Any] = dict(existing.snapshot or {}) if existing else {}

        ensure_supported_snapshot_meta(meta, island_id=island_id)

        meta["last_update_at"] = now

        if update.lower_bounds is not None:
            meta["lower_bounds"] = [float(v) for v in update.lower_bounds]
        if update.upper_bounds is not None:
            meta["upper_bounds"] = [float(v) for v in update.upper_bounds]
        if update.samples_since_fit is not None:
            meta["samples_since_fit"] = max(0, int(update.samples_since_fit))

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

        if update.archive_replace is not None:
            self._apply_archive_replace(
                session,
                island_id=island_id,
                cells=update.archive_replace,
            )
        elif update.cell_upsert is not None:
            cell = update.cell_upsert
            self._upsert_archive_cells(
                session,
                values=[self._archive_cell_values(island_id=island_id, cell=cell)],
            )

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
            self._prune_history_entries(
                session,
                island_id=island_id,
                history_limit=update.history_limit,
            )

    def _archive_cell_values(
        self,
        *,
        island_id: str,
        cell: SnapshotCellUpsert,
    ) -> dict[str, object]:
        return {
            "island_id": island_id,
            "cell_index": int(cell.cell_index),
            "commit_hash": str(cell.commit_hash),
            "objective": float(cell.objective),
            "measures": [float(v) for v in cell.measures],
            "solution": [
                float(v)
                for v in compact_solution(
                    measures=cell.measures,
                    solution=cell.solution,
                )
            ],
            "timestamp": float(cell.timestamp),
        }

    def _apply_archive_replace(
        self,
        session: Session,
        *,
        island_id: str,
        cells: Sequence[SnapshotCellUpsert],
    ) -> None:
        desired_indices = sorted({int(cell.cell_index) for cell in cells})
        if desired_indices:
            session.execute(
                delete(MapElitesArchiveCell).where(
                    MapElitesArchiveCell.island_id == island_id,
                    MapElitesArchiveCell.cell_index.not_in(desired_indices),
                )
            )
        else:
            session.execute(
                delete(MapElitesArchiveCell).where(
                    MapElitesArchiveCell.island_id == island_id,
                )
            )
        self._upsert_archive_cells(
            session,
            values=[
                self._archive_cell_values(island_id=island_id, cell=cell)
                for cell in cells
            ],
        )

    def _diff_archive_replace(
        self,
        *,
        island_id: str,
        existing_rows: Sequence[MapElitesArchiveCell],
        cells: Sequence[SnapshotCellUpsert],
    ) -> tuple[list[int], list[dict[str, object]]]:
        desired_by_cell = {
            int(cell.cell_index): self._archive_cell_values(island_id=island_id, cell=cell)
            for cell in cells
        }
        existing_by_cell = {int(getattr(row, "cell_index")): row for row in existing_rows}

        stale_indices = sorted(
            cell_index
            for cell_index in existing_by_cell
            if cell_index not in desired_by_cell
        )
        upsert_values: list[dict[str, object]] = []
        for cell_index in sorted(desired_by_cell):
            values = desired_by_cell[cell_index]
            row = existing_by_cell.get(cell_index)
            if row is not None and self._archive_row_matches_values(row, values):
                continue
            upsert_values.append(values)
        return stale_indices, upsert_values

    def _archive_row_matches_values(
        self,
        row: MapElitesArchiveCell,
        values: Mapping[str, object],
    ) -> bool:
        return (
            str(getattr(row, "commit_hash", "") or "") == str(values.get("commit_hash", "") or "")
            and float(getattr(row, "objective", 0.0) or 0.0) == float(values.get("objective", 0.0) or 0.0)
            and [float(v) for v in (getattr(row, "measures", ()) or ())] == list(values.get("measures", ()))
            and [float(v) for v in (getattr(row, "solution", ()) or ())] == list(values.get("solution", ()))
            and float(getattr(row, "timestamp", 0.0) or 0.0) == float(values.get("timestamp", 0.0) or 0.0)
        )

    def _upsert_archive_cells(
        self,
        session: Session,
        *,
        values: Sequence[Mapping[str, object]],
    ) -> None:
        payload = [dict(value) for value in values if value]
        if not payload:
            return
        stmt = pg_insert(MapElitesArchiveCell).values(payload)
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

    def _prune_history_entries(
        self,
        session: Session,
        *,
        island_id: str,
        history_limit: int | None,
    ) -> None:
        effective_limit = int(history_limit or 0)
        if effective_limit <= 0:
            return

        stale_commit_hashes = list(
            session.execute(
                select(MapElitesPcaHistory.commit_hash)
                .where(MapElitesPcaHistory.island_id == island_id)
                .order_by(
                    MapElitesPcaHistory.last_seen_at.desc(),
                    MapElitesPcaHistory.commit_hash.asc(),
                )
                .offset(effective_limit)
            )
            .scalars()
            .all()
        )
        if not stale_commit_hashes:
            return

        session.execute(
            delete(MapElitesPcaHistory).where(
                MapElitesPcaHistory.island_id == island_id,
                MapElitesPcaHistory.commit_hash.in_(stale_commit_hashes),
            )
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
                    "solution": [
                        float(v)
                        for v in materialize_solution(
                            measures=row.measures or [],
                            solution=row.solution or [],
                        )
                    ],
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

    samples_since_fit = snapshot.get("samples_since_fit")
    if samples_since_fit is not None and hasattr(state, "samples_since_fit"):
        try:
            state.samples_since_fit = max(0, int(samples_since_fit))
        except (TypeError, ValueError):
            state.samples_since_fit = 0

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
    rotation = None
    if projection.rotation:
        rotation = [[float(value) for value in row] for row in projection.rotation]
    return {
        "feature_count": projection.feature_count,
        "components": [[float(value) for value in row] for row in projection.components],
        "mean": [float(value) for value in projection.mean],
        "explained_variance": [float(value) for value in projection.explained_variance],
        "explained_variance_ratio": [
            float(value) for value in projection.explained_variance_ratio
        ],
        "sample_count": projection.sample_count,
        "epoch": int(getattr(projection, "epoch", 0) or 0),
        "fitted_at": projection.fitted_at,
        "whiten": projection.whiten,
        "rotation": rotation,
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
    epoch = int(payload.get("epoch", 0) or 0)
    rotation_payload = payload.get("rotation")
    rotation: tuple[Vector, ...] | None = None
    if isinstance(rotation_payload, (list, tuple)):
        try:
            rotation_rows = []
            for row in rotation_payload:
                if not isinstance(row, (list, tuple)):
                    rotation_rows = []
                    break
                rotation_rows.append(tuple(float(value) for value in row))
            if rotation_rows:
                rotation = tuple(rotation_rows)
        except Exception:
            rotation = None
    return PCAProjection(
        feature_count=int(payload.get("feature_count", len(mean))),
        components=components,
        mean=mean,
        explained_variance=explained_variance,
        explained_variance_ratio=explained,
        sample_count=int(payload.get("sample_count", 0)),
        epoch=max(0, epoch),
        fitted_at=float(payload.get("fitted_at", 0.0)),
        whiten=bool(payload.get("whiten", False)),
        rotation=rotation,
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

    solution_rows: list[np.ndarray] = []
    objective_rows: list[float] = []
    measures_rows: list[np.ndarray] = []
    commit_rows: list[str] = []
    timestamp_rows: list[float] = []
    stored_indices: list[int | None] = []

    for entry in entries:
        measures_values = array_to_list(entry.get("measures"))
        if not measures_values:
            continue
        solution_values = array_to_list(entry.get("solution"))
        if not solution_values:
            solution_values = list(measures_values)

        solution = np.asarray(solution_values, dtype=np.float64)
        measures = np.asarray(measures_values, dtype=np.float64)
        if expected_solution_dim is not None and solution.shape[0] != int(expected_solution_dim):
            log.warning(
                "Skipping snapshot archive entry due to incompatible solution_dim (island={} expected={} got={})",
                island_id,
                int(expected_solution_dim),
                int(solution.shape[0]),
            )
            continue
        if expected_measures_dim is not None and measures.shape[0] != int(expected_measures_dim):
            log.warning(
                "Skipping snapshot archive entry due to incompatible measures dimensionality (island={} expected={} got={})",
                island_id,
                int(expected_measures_dim),
                int(measures.shape[0]),
            )
            continue

        objective = float(entry.get("objective", 0.0))
        commit_hash = str(entry.get("commit_hash", ""))
        timestamp_value = float(entry.get("timestamp", 0.0))

        stored_index = entry.get("index")
        parsed_index: int | None = None
        if stored_index is not None:
            try:
                parsed_index = int(stored_index)
            except (TypeError, ValueError):
                parsed_index = None

        solution_rows.append(solution)
        objective_rows.append(objective)
        measures_rows.append(measures)
        commit_rows.append(commit_hash)
        timestamp_rows.append(timestamp_value)
        stored_indices.append(parsed_index)

    if not solution_rows:
        return

    solution_batch = np.asarray(solution_rows, dtype=np.float64)
    objective_batch = np.asarray(objective_rows, dtype=np.float64)
    measures_batch = np.asarray(measures_rows, dtype=np.float64)
    commit_batch = np.asarray(commit_rows, dtype=object)
    timestamp_batch = np.asarray(timestamp_rows, dtype=np.float64)

    try:
        add_info = archive.add(
            solution_batch,
            objective_batch,
            measures_batch,
            commit_hash=commit_batch,
            timestamp=timestamp_batch,
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.warning(
            "Failed to restore snapshot batch into GridArchive (island={} entries={}): {}",
            island_id,
            len(solution_rows),
            exc,
        )
        return

    statuses = np.asarray(add_info.get("status", ()), dtype=np.int64).reshape(-1)
    values = np.asarray(add_info.get("value", ()), dtype=np.float64).reshape(-1)
    if statuses.shape[0] != len(solution_rows) or values.shape[0] != len(solution_rows):
        raise RuntimeError(
            "GridArchive.add() returned unexpected add_info shape while restoring snapshot "
            f"(island={island_id} expected={len(solution_rows)} got_status={statuses.shape[0]} got_value={values.shape[0]})."
        )

    archive_indices = np.asarray(archive.index_of(measures_batch), dtype=np.int64).reshape(-1)
    if archive_indices.shape[0] != len(solution_rows):
        raise RuntimeError(
            "GridArchive.index_of() returned unexpected index shape while restoring snapshot "
            f"(island={island_id} expected={len(solution_rows)} got={archive_indices.shape[0]})."
        )

    winners_by_cell: dict[int, tuple[int, float, int]] = {}
    for idx, status_raw in enumerate(statuses):
        status = int(status_raw)
        if status <= 0:
            continue

        cell_index = int(archive_indices[idx])
        stored_index = stored_indices[idx]
        if stored_index is not None:
            cell_index = int(stored_index)
        if expected_cell_count is not None and (cell_index < 0 or cell_index >= expected_cell_count):
            cell_index = int(archive_indices[idx])

        rank = (status, float(values[idx]), idx)
        previous = winners_by_cell.get(cell_index)
        if previous is None or (rank[0], rank[1]) > (previous[0], previous[1]):
            winners_by_cell[cell_index] = rank

    for cell_index, (_status, _value, idx) in winners_by_cell.items():
        commit_hash = commit_rows[idx]
        previous_commit = state.index_to_commit.get(cell_index)
        state.index_to_commit[cell_index] = commit_hash
        if commit_hash:
            state.commit_to_index[commit_hash] = cell_index
            commit_to_island[commit_hash] = island_id
        if previous_commit and previous_commit != commit_hash:
            state.commit_to_index.pop(previous_commit, None)
            commit_to_island.pop(previous_commit, None)

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
