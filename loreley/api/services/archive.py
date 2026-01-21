"""MAP-Elites archive access for the UI API (read-only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import func, select

from loreley.config import Settings, get_settings
from loreley.core.map_elites.map_elites import MapElitesManager
from loreley.db.base import session_scope
from loreley.db.models import MapElitesArchiveCell, MapElitesPcaHistory, MapElitesState


@dataclass(frozen=True, slots=True)
class SnapshotMeta:
    entry_count: int
    lower_bounds: list[float]
    upper_bounds: list[float]
    has_projection: bool
    history_length: int


def list_islands(*, experiment_id: UUID) -> list[str]:
    """Return known island IDs for an experiment."""

    with session_scope() as session:
        stmt = select(MapElitesState.island_id).where(MapElitesState.experiment_id == experiment_id)
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
    experiment_id: UUID,
    island_id: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Return MAP-Elites stats for an island using MapElitesManager."""

    base_settings = settings or get_settings()
    manager = MapElitesManager(settings=base_settings, experiment_id=experiment_id)
    return dict(manager.describe_island(island_id))


def list_records(
    *,
    experiment_id: UUID,
    island_id: str,
    settings: Settings | None = None,
) -> list[Any]:
    """Return all elite records for an island."""

    base_settings = settings or get_settings()
    manager = MapElitesManager(settings=base_settings, experiment_id=experiment_id)
    return list(manager.get_records(island_id))


def snapshot_meta(
    *,
    experiment_id: UUID,
    island_id: str,
    settings: Settings | None = None,
) -> SnapshotMeta:
    """Return lightweight metadata about the stored snapshot (without reconstructing the archive)."""

    base_settings = settings or get_settings()
    dims = max(1, int(base_settings.mapelites_dimensionality_target_dims))

    with session_scope() as session:
        stmt = select(MapElitesState).where(
            MapElitesState.experiment_id == experiment_id,
            MapElitesState.island_id == island_id,
        )
        row = session.execute(stmt).scalar_one_or_none()
        snapshot = dict(row.snapshot or {}) if row and row.snapshot else {}
        if "archive" in snapshot or "history" in snapshot:
            raise ValueError(
                "Legacy MAP-Elites snapshot detected; reset the database schema (dev). "
                f"(experiment_id={experiment_id} island_id={island_id})"
            )
        entry_count = int(
            session.execute(
                select(func.count())
                .select_from(MapElitesArchiveCell)
                .where(
                    MapElitesArchiveCell.experiment_id == experiment_id,
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
                    MapElitesPcaHistory.experiment_id == experiment_id,
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


def snapshot_updated_at(*, experiment_id: UUID, island_id: str) -> Any:
    """Return updated_at timestamp for the stored snapshot row (if any)."""

    with session_scope() as session:
        stmt = select(MapElitesState.updated_at).where(
            MapElitesState.experiment_id == experiment_id,
            MapElitesState.island_id == island_id,
        )
        return session.execute(stmt).scalar_one_or_none()


