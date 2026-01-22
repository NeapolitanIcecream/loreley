"""MAP-Elites archive endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from loreley.api.schemas.archive import ArchiveRecordOut, ArchiveSnapshotMetaOut, IslandStatsOut
from loreley.api.services.archive import (
    describe_island,
    list_islands,
    list_records,
    snapshot_meta,
    snapshot_updated_at,
)
from loreley.config import get_settings

router = APIRouter()


@router.get("/archive/islands", response_model=list[IslandStatsOut])
def get_islands() -> list[IslandStatsOut]:
    settings = get_settings()
    islands = list_islands()
    out: list[IslandStatsOut] = []
    for island_id in islands:
        try:
            stats = describe_island(
                island_id=island_id,
                settings=settings,
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        out.append(IslandStatsOut.model_validate(stats))
    return out


@router.get("/archive/records", response_model=list[ArchiveRecordOut])
def get_records(
    island_id: str = Query(default="", description="Island ID; empty means default island."),
) -> list[ArchiveRecordOut]:
    settings = get_settings()
    effective_island = island_id.strip() or (settings.mapelites_default_island_id or "main")

    try:
        records = list_records(island_id=effective_island, settings=settings)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return records


@router.get("/archive/snapshot_meta", response_model=ArchiveSnapshotMetaOut)
def get_snapshot_meta(
    island_id: str,
) -> ArchiveSnapshotMetaOut:
    settings = get_settings()
    effective_island = island_id.strip() or (settings.mapelites_default_island_id or "main")
    cells_per_dim = max(2, int(settings.mapelites_archive_cells_per_dim))

    meta = snapshot_meta(island_id=effective_island, settings=settings)
    dims = max(1, len(meta.lower_bounds))
    updated_at = snapshot_updated_at(island_id=effective_island)

    return ArchiveSnapshotMetaOut(
        island_id=effective_island,
        entry_count=int(meta.entry_count),
        dims=int(dims),
        cells_per_dim=int(cells_per_dim),
        lower_bounds=list(meta.lower_bounds),
        upper_bounds=list(meta.upper_bounds),
        has_projection=bool(meta.has_projection),
        history_length=int(meta.history_length),
        updated_at=updated_at,
    )


