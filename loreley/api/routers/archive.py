"""MAP-Elites archive endpoints."""

from __future__ import annotations

from uuid import UUID

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
def get_islands(experiment_id: UUID) -> list[IslandStatsOut]:
    settings = get_settings()
    islands = list_islands(experiment_id=experiment_id)
    out: list[IslandStatsOut] = []
    for island_id in islands:
        try:
            stats = describe_island(
                experiment_id=experiment_id,
                island_id=island_id,
                settings=settings,
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        out.append(
            IslandStatsOut(
                island_id=str(stats.get("island_id", island_id)),
                occupied=int(stats.get("occupied", 0)),
                cells=int(stats.get("cells", 0)),
                qd_score=float(stats.get("qd_score", 0.0)),
                best_fitness=float(stats.get("best_fitness", 0.0)),
            )
        )
    return out


@router.get("/archive/records", response_model=list[ArchiveRecordOut])
def get_records(
    experiment_id: UUID,
    island_id: str = Query(default="", description="Island ID; empty means default island."),
) -> list[ArchiveRecordOut]:
    settings = get_settings()
    effective_island = island_id.strip() or (settings.mapelites_default_island_id or "main")

    try:
        records = list_records(experiment_id=experiment_id, island_id=effective_island, settings=settings)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    out: list[ArchiveRecordOut] = []
    for r in records:
        out.append(
            ArchiveRecordOut(
                commit_hash=str(getattr(r, "commit_hash", "")),
                island_id=str(getattr(r, "island_id", effective_island)),
                cell_index=int(getattr(r, "cell_index", -1)),
                fitness=float(getattr(r, "fitness", 0.0)),
                measures=[float(v) for v in getattr(r, "measures", ())],
                solution=[float(v) for v in getattr(r, "solution", ())],
                timestamp=float(getattr(r, "timestamp", 0.0)),
            )
        )
    return out


@router.get("/archive/snapshot_meta", response_model=ArchiveSnapshotMetaOut)
def get_snapshot_meta(
    experiment_id: UUID,
    island_id: str,
) -> ArchiveSnapshotMetaOut:
    settings = get_settings()
    effective_island = island_id.strip() or (settings.mapelites_default_island_id or "main")
    cells_per_dim = max(2, int(settings.mapelites_archive_cells_per_dim))

    meta = snapshot_meta(experiment_id=experiment_id, island_id=effective_island, settings=settings)
    dims = max(1, len(meta.lower_bounds))
    updated_at = snapshot_updated_at(experiment_id=experiment_id, island_id=effective_island)

    return ArchiveSnapshotMetaOut(
        experiment_id=experiment_id,
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


