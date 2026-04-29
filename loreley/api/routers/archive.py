"""MAP-Elites archive endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from fastapi import HTTPException

from loreley.api.pagination import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT, PaginationCursorError
from loreley.api.schemas.archive import ArchiveRecordOut, ArchiveRecordPageOut, ArchiveSnapshotMetaOut, IslandStatsOut
from loreley.api.services.evidence import load_evidence_indicators_by_commit_hash
from loreley.api.services.archive import (
    describe_island,
    list_islands,
    list_records,
    list_records_page,
    snapshot_meta,
    snapshot_updated_at,
)
from loreley.config import get_settings, resolve_default_island_id

router = APIRouter()


@router.get("/archive/islands", response_model=list[IslandStatsOut])
def get_islands() -> list[IslandStatsOut]:
    settings = get_settings()
    islands = list_islands()
    out: list[IslandStatsOut] = []
    for island_id in islands:
        stats = describe_island(
            island_id=island_id,
            settings=settings,
        )
        out.append(IslandStatsOut.model_validate(stats))
    return out


@router.get("/archive/records", response_model=list[ArchiveRecordOut])
def get_records(
    island_id: str = Query(default="", description="Island ID; empty means default island."),
    limit: int = Query(default=DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    offset: int = Query(default=0, ge=0),
) -> list[ArchiveRecordOut]:
    settings = get_settings()
    effective_island = island_id.strip() or resolve_default_island_id(settings)

    records = list_records(
        island_id=effective_island,
        settings=settings,
        limit=limit,
        offset=offset,
    )
    return _records_with_evidence(records)


@router.get("/archive/records/page", response_model=ArchiveRecordPageOut)
def get_records_page(
    island_id: str = Query(default="", description="Island ID; empty means default island."),
    limit: int = Query(default=DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    cursor: str | None = Query(default=None, description="Opaque pagination cursor."),
) -> ArchiveRecordPageOut:
    settings = get_settings()
    effective_island = island_id.strip() or resolve_default_island_id(settings)
    try:
        page = list_records_page(
            island_id=effective_island,
            settings=settings,
            limit=limit,
            cursor=cursor,
        )
    except PaginationCursorError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ArchiveRecordPageOut(
        items=_records_with_evidence(page.items),
        next_cursor=page.next_cursor,
    )


@router.get("/archive/snapshot_meta", response_model=ArchiveSnapshotMetaOut)
def get_snapshot_meta(
    island_id: str,
) -> ArchiveSnapshotMetaOut:
    settings = get_settings()
    effective_island = island_id.strip() or resolve_default_island_id(settings)
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


def _records_with_evidence(records: list[object]) -> list[ArchiveRecordOut]:
    indicators = load_evidence_indicators_by_commit_hash(
        [str(getattr(record, "commit_hash", "") or "") for record in records]
    )
    out: list[ArchiveRecordOut] = []
    for record in records:
        item = ArchiveRecordOut.model_validate(record)
        indicator = indicators.get(item.commit_hash)
        if indicator is not None:
            item = item.model_copy(update=indicator.as_dict())
        out.append(item)
    return out
