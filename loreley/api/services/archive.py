"""MAP-Elites archive access for the UI API (read-only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select

from loreley.config import Settings, get_settings
from loreley.core.map_elites.types import MapElitesRecord
from loreley.core.map_elites.snapshot import ensure_supported_snapshot_meta
from loreley.db.base import session_scope
from loreley.db.models import MapElitesArchiveCell, MapElitesPcaHistory, MapElitesState


@dataclass(frozen=True, slots=True)
class SnapshotMeta:
    entry_count: int
    lower_bounds: list[float]
    upper_bounds: list[float]
    has_projection: bool
    history_length: int


def list_islands() -> list[str]:
    """Return known island IDs for the instance."""

    with session_scope() as session:
        stmt = select(MapElitesState.island_id)
        values = [str(v) for v in session.execute(stmt).scalars().all() if v]
    # Deterministic order for UI.
    values = sorted(set(values))
    if values:
        return values

    base_settings = get_settings()
    default_island = (base_settings.mapelites_default_island_id or "main").strip() or "main"
    return [default_island]


def describe_island(
    *,
    island_id: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Return MAP-Elites stats for an island directly from persisted archive rows."""

    base_settings = settings or get_settings()
    dims = max(1, int(base_settings.mapelites_dimensionality_target_dims))
    cells_per_dim = max(2, int(base_settings.mapelites_archive_cells_per_dim))
    cells = int(cells_per_dim**dims)

    with session_scope() as session:
        occupied, qd_score, best_fitness = session.execute(
            select(
                func.count(MapElitesArchiveCell.cell_index),
                func.coalesce(func.sum(MapElitesArchiveCell.objective), 0.0),
                func.max(MapElitesArchiveCell.objective),
            ).where(
                MapElitesArchiveCell.island_id == island_id,
            )
        ).one()

    occupied_value = int(occupied or 0)
    qd_score_value = float(qd_score or 0.0)
    best_value = float(best_fitness) if best_fitness is not None else 0.0
    coverage = (occupied_value / cells) if cells else 0.0
    norm_qd_score = (qd_score_value / cells) if cells else 0.0
    return {
        "island_id": island_id,
        "occupied": occupied_value,
        "cells": cells,
        "coverage": coverage,
        "qd_score": qd_score_value,
        "norm_qd_score": norm_qd_score,
        "best_fitness": best_value,
    }


def list_records(
    *,
    island_id: str,
    settings: Settings | None = None,
) -> list[MapElitesRecord]:
    """Return all elite records for an island from persisted archive cells."""

    _ = settings or get_settings()

    with session_scope() as session:
        rows = list(
            session.execute(
                select(MapElitesArchiveCell)
                .where(MapElitesArchiveCell.island_id == island_id)
                .order_by(MapElitesArchiveCell.cell_index.asc())
            ).scalars().all()
        )

    records: list[MapElitesRecord] = []
    for row in rows:
        records.append(
            MapElitesRecord(
                commit_hash=str(row.commit_hash or ""),
                island_id=str(row.island_id or island_id),
                cell_index=int(row.cell_index),
                fitness=float(row.objective or 0.0),
                measures=tuple(float(v) for v in (row.measures or ())),
                solution=tuple(float(v) for v in (row.solution or ())),
                timestamp=float(row.timestamp or 0.0),
            )
        )
    return records


def snapshot_meta(
    *,
    island_id: str,
    settings: Settings | None = None,
) -> SnapshotMeta:
    """Return lightweight metadata about the stored snapshot (without reconstructing the archive)."""

    base_settings = settings or get_settings()
    dims = max(1, int(base_settings.mapelites_dimensionality_target_dims))

    with session_scope() as session:
        stmt = select(MapElitesState).where(
            MapElitesState.island_id == island_id,
        )
        row = session.execute(stmt).scalar_one_or_none()
        snapshot = dict(row.snapshot or {}) if row and row.snapshot else {}
        ensure_supported_snapshot_meta(snapshot, island_id=island_id)
        entry_count = int(
            session.execute(
                select(func.count())
                .select_from(MapElitesArchiveCell)
                .where(
                    MapElitesArchiveCell.island_id == island_id,
                )
            ).scalar_one()
            or 0
        )

        history_length = int(
            session.execute(
                select(func.count())
                .select_from(MapElitesPcaHistory)
                .where(
                    MapElitesPcaHistory.island_id == island_id,
                )
            ).scalar_one()
            or 0
        )

    lower = snapshot.get("lower_bounds") or [0.0] * dims
    upper = snapshot.get("upper_bounds") or [1.0] * dims
    has_projection = bool(snapshot.get("projection"))

    return SnapshotMeta(
        entry_count=entry_count,
        lower_bounds=[float(v) for v in lower] if isinstance(lower, list) else [0.0] * dims,
        upper_bounds=[float(v) for v in upper] if isinstance(upper, list) else [1.0] * dims,
        has_projection=has_projection,
        history_length=history_length,
    )


def snapshot_updated_at(*, island_id: str) -> Any:
    """Return updated_at timestamp for the stored snapshot row (if any)."""

    with session_scope() as session:
        stmt = select(MapElitesState.updated_at).where(
            MapElitesState.island_id == island_id,
        )
        return session.execute(stmt).scalar_one_or_none()

