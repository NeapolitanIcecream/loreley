"""MAP-Elites archive access for the UI API (read-only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import and_, func, or_, select

from loreley.api.pagination import PaginationCursorError, decode_cursor, encode_cursor, normalize_pagination
from loreley.config import Settings, get_settings, resolve_objective_contract
from loreley.core.candidate_fate import derive_candidate_fate
from loreley.core.map_elites.objectives import ObjectiveContract
from loreley.core.map_elites.types import MapElitesRecord
from loreley.core.map_elites.snapshot import (
    ensure_supported_snapshot_meta,
    validate_snapshot_contract,
)
from loreley.db.base import session_scope
from loreley.db.models import (
    CandidateCommit,
    CommitCard,
    EvolutionJob,
    MapElitesArchiveCell,
    MapElitesPcaHistory,
    MapElitesState,
)
from loreley.scheduler.baselines import improvement_from_baseline, load_latest_matching_baseline


@dataclass(frozen=True, slots=True)
class SnapshotMeta:
    entry_count: int
    lower_bounds: list[float]
    upper_bounds: list[float]
    has_projection: bool
    history_length: int


@dataclass(frozen=True, slots=True)
class ArchiveRecordPage:
    items: list[MapElitesRecord]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class ArchiveRecordBuildContext:
    island_id: str
    objective_contract: ObjectiveContract
    baseline_by_commit: dict[str, Any | None] | None = None
    baseline: Any | None = None


def list_islands(*, settings: Settings | None = None) -> list[str]:
    """Return the configured island contract in scheduling order."""

    return list((settings or get_settings()).mapelites_islands)


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
    contract = resolve_objective_contract(base_settings)
    primary = contract.primary
    primary_value_column = MapElitesArchiveCell.objective_values[1]
    best_primary = (
        func.max(primary_value_column)
        if primary.higher_is_better
        else func.min(primary_value_column)
    )

    with session_scope() as session:
        _validate_persisted_objective_contract(
            session=session,
            island_id=island_id,
            objective_contract=contract,
        )
        occupied, elites, best_primary_value = session.execute(
            select(
                func.count(func.distinct(MapElitesArchiveCell.cell_index)),
                func.count(),
                best_primary,
            ).where(
                MapElitesArchiveCell.island_id == island_id,
            )
        ).one()

    occupied_value = int(occupied or 0)
    elites_value = int(elites or 0)
    coverage = (occupied_value / cells) if cells else 0.0
    return {
        "island_id": island_id,
        "occupied": occupied_value,
        "elites": elites_value,
        "cells": cells,
        "coverage": coverage,
        "objective_count": len(contract.specs),
        "front_max_size": int(base_settings.mapelites_pareto_front_max_size),
        "best_primary_value": (
            float(best_primary_value) if best_primary_value is not None else None
        ),
        "primary_metric_name": primary.name,
        "primary_metric_higher_is_better": primary.higher_is_better,
    }


def list_records(
    *,
    island_id: str,
    settings: Settings | None = None,
    limit: int = 200,
    offset: int = 0,
) -> list[MapElitesRecord]:
    """Return all elite records for an island from persisted archive cells."""
    base_settings = settings or get_settings()
    contract = resolve_objective_contract(base_settings)
    limit, offset = normalize_pagination(limit, offset)

    with session_scope() as session:
        _validate_persisted_objective_contract(
            session=session,
            island_id=island_id,
            objective_contract=contract,
        )
        rows = list(
            session.execute(
                select(MapElitesArchiveCell)
                .where(MapElitesArchiveCell.island_id == island_id)
                .order_by(
                    MapElitesArchiveCell.cell_index.asc(),
                    MapElitesArchiveCell.commit_hash.asc(),
                )
                .limit(limit)
                .offset(offset)
            ).scalars().all()
        )
        baseline_by_commit: dict[str, Any | None] = {}
        if rows:
            commit_hashes = _commit_hashes_from_rows(rows)
            if _baseline_lookup_configured(settings=base_settings):
                baseline_by_commit = _load_baselines_by_commit(
                    session=session,
                    settings=base_settings,
                    commit_hashes=commit_hashes,
                )

    return _build_records_from_rows(
        rows=rows,
        context=ArchiveRecordBuildContext(
            island_id=island_id,
            objective_contract=contract,
            baseline_by_commit=baseline_by_commit,
        ),
    )


def list_records_page(
    *,
    island_id: str,
    settings: Settings | None = None,
    limit: int = 200,
    cursor: str | None = None,
) -> ArchiveRecordPage:
    """Return a cursor-paginated page of archive records for an island."""

    base_settings = settings or get_settings()
    contract = resolve_objective_contract(base_settings)
    limit, _ = normalize_pagination(limit, 0)

    with session_scope() as session:
        _validate_persisted_objective_contract(
            session=session,
            island_id=island_id,
            objective_contract=contract,
        )
        stmt = (
            select(MapElitesArchiveCell)
            .where(MapElitesArchiveCell.island_id == island_id)
            .order_by(
                MapElitesArchiveCell.cell_index.asc(),
                MapElitesArchiveCell.commit_hash.asc(),
            )
        )
        if cursor:
            try:
                payload = decode_cursor(cursor)
                last_cell_index = int(payload.get("cell_index"))
                last_commit_hash = str(payload.get("commit_hash") or "")
                if not last_commit_hash:
                    raise ValueError("missing commit hash")
            except (PaginationCursorError, TypeError, ValueError) as exc:
                raise PaginationCursorError("Archive records cursor is invalid.") from exc
            stmt = stmt.where(
                or_(
                    MapElitesArchiveCell.cell_index > last_cell_index,
                    and_(
                        MapElitesArchiveCell.cell_index == last_cell_index,
                        MapElitesArchiveCell.commit_hash > last_commit_hash,
                    ),
                )
            )
        stmt = stmt.limit(limit + 1)
        rows = list(session.execute(stmt).scalars().all())
        items_rows = rows[:limit]
        baseline_by_commit: dict[str, Any | None] = {}
        if items_rows:
            commit_hashes = _commit_hashes_from_rows(items_rows)
            if _baseline_lookup_configured(settings=base_settings):
                baseline_by_commit = _load_baselines_by_commit(
                    session=session,
                    settings=base_settings,
                    commit_hashes=commit_hashes,
                )

    records = _build_records_from_rows(
        rows=items_rows,
        context=ArchiveRecordBuildContext(
            island_id=island_id,
            objective_contract=contract,
            baseline_by_commit=baseline_by_commit,
        ),
    )
    next_cursor = (
        encode_cursor(
            {
                "cell_index": records[-1].cell_index,
                "commit_hash": records[-1].commit_hash,
            }
        )
        if len(rows) > limit and records
        else None
    )
    return ArchiveRecordPage(items=records, next_cursor=next_cursor)


def _build_records_from_rows(
    *,
    rows: list[MapElitesArchiveCell],
    context: ArchiveRecordBuildContext,
) -> list[MapElitesRecord]:
    return [
        _record_from_archive_row(
            row=row,
            island_id=context.island_id,
            objective_contract=context.objective_contract,
            baseline=_baseline_for_archive_row(
                row=row,
                baseline_by_commit=context.baseline_by_commit,
                fallback=context.baseline,
            ),
        )
        for row in rows
    ]


def _record_from_archive_row(
    *,
    row: MapElitesArchiveCell,
    island_id: str,
    objective_contract: ObjectiveContract,
    baseline: Any | None,
) -> MapElitesRecord:
    commit_hash = str(row.commit_hash or "")
    objectives = objective_contract.resolve_values(row.objective_values or ())
    primary = objective_contract.primary
    primary_value = objectives.values[0]
    fate = derive_candidate_fate(
        current_archive_cell_index=int(row.cell_index),
        current_archive_member=True,
    )
    return MapElitesRecord(
        commit_hash=commit_hash,
        island_id=str(row.island_id or island_id),
        cell_index=int(row.cell_index),
        objective_values=objectives.values,
        objective_scores=objectives.scores,
        primary_metric_value=primary_value,
        primary_metric_name=primary.name,
        primary_metric_higher_is_better=primary.higher_is_better,
        campaign_baseline_id=_baseline_id(baseline),
        baseline_key_hash=getattr(baseline, "baseline_key_hash", None),
        baseline_status=getattr(baseline, "status", None),
        delta_from_root_baseline=improvement_from_baseline(
            candidate_value=primary_value,
            baseline=baseline,
        ),
        candidate_fate_label=fate.label,
        candidate_fate_reason=fate.reason,
        measures=tuple(float(v) for v in (row.measures or ())),
        timestamp=float(row.timestamp or 0.0),
    )


def _baseline_for_archive_row(
    *,
    row: MapElitesArchiveCell,
    baseline_by_commit: dict[str, Any | None] | None,
    fallback: Any | None,
) -> Any | None:
    if baseline_by_commit is None:
        return fallback
    return baseline_by_commit.get(str(row.commit_hash or ""))


def _baseline_id(baseline: Any | None) -> str | None:
    baseline_id = getattr(baseline, "id", None)
    return str(baseline_id) if baseline_id else None


def _validate_persisted_objective_contract(
    *,
    session: Any,
    island_id: str,
    objective_contract: ObjectiveContract,
) -> None:
    snapshot = session.execute(
        select(MapElitesState.snapshot).where(
            MapElitesState.island_id == island_id,
        )
    ).scalar_one_or_none()
    if snapshot is None:
        return
    meta = dict(snapshot or {})
    ensure_supported_snapshot_meta(meta, island_id=island_id)
    validate_snapshot_contract(
        meta,
        island_id=island_id,
        objective_contract=objective_contract,
    )


def _commit_hashes_from_rows(rows: list[Any]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        commit_hash = str(row.commit_hash or "").strip()
        if not commit_hash or commit_hash in seen:
            continue
        seen.add(commit_hash)
        values.append(commit_hash)
    return values


def _baseline_lookup_configured(*, settings: Settings) -> bool:
    root_commit = str(getattr(settings, "mapelites_experiment_root_commit", "") or "").strip()
    return bool(root_commit)


def _load_baselines_by_commit(
    *,
    session: Any,
    settings: Settings,
    commit_hashes: list[str],
) -> dict[str, Any | None]:
    campaign_hash_by_commit = _load_campaign_program_hashes_by_commit(
        session=session,
        commit_hashes=commit_hashes,
    )
    if not campaign_hash_by_commit:
        return {}
    baselines_by_campaign_hash: dict[str | None, Any | None] = {}
    for campaign_program_hash in set(campaign_hash_by_commit.values()):
        baselines_by_campaign_hash[campaign_program_hash] = load_latest_matching_baseline(
            session=session,
            settings=settings,
            campaign_program_hash=campaign_program_hash,
        )
    return {
        commit_hash: baselines_by_campaign_hash.get(campaign_program_hash)
        for commit_hash, campaign_program_hash in campaign_hash_by_commit.items()
    }


def _load_campaign_program_hashes_by_commit(
    *,
    session: Any,
    commit_hashes: list[str],
) -> dict[str, str | None]:
    if not commit_hashes:
        return {}
    campaign_hash_by_commit = _load_candidate_campaign_program_hashes(
        session=session,
        commit_hashes=commit_hashes,
    )
    missing_commits = _campaign_hash_missing_commits(
        commit_hashes=commit_hashes,
        campaign_hash_by_commit=campaign_hash_by_commit,
    )
    if missing_commits:
        resolved_commits = {
            commit_hash
            for commit_hash, campaign_program_hash in campaign_hash_by_commit.items()
            if campaign_program_hash is not None
        }
        campaign_hash_by_commit.update(
            _load_job_campaign_program_hashes(
                session=session,
                commit_hashes=missing_commits,
                existing_commits=resolved_commits,
            )
        )
    return campaign_hash_by_commit


def _load_candidate_campaign_program_hashes(
    *,
    session: Any,
    commit_hashes: list[str],
) -> dict[str, str | None]:
    rows = session.execute(
        select(CandidateCommit.commit_hash, CandidateCommit.campaign_program_hash).where(
            CandidateCommit.commit_hash.in_(commit_hashes),
        )
    ).all()
    return _campaign_hash_rows_to_mapping(rows=rows)


def _campaign_hash_missing_commits(
    *,
    commit_hashes: list[str],
    campaign_hash_by_commit: dict[str, str | None],
) -> list[str]:
    return [
        commit_hash
        for commit_hash in commit_hashes
        if campaign_hash_by_commit.get(commit_hash) is None
    ]


def _load_job_campaign_program_hashes(
    *,
    session: Any,
    commit_hashes: list[str],
    existing_commits: set[str],
) -> dict[str, str | None]:
    job_rows = session.execute(
        select(CommitCard.commit_hash, EvolutionJob.campaign_program_hash)
        .join(EvolutionJob, EvolutionJob.id == CommitCard.job_id)
        .where(CommitCard.commit_hash.in_(commit_hashes))
    ).all()
    return _campaign_hash_rows_to_mapping(rows=job_rows, existing_commits=existing_commits)


def _campaign_hash_rows_to_mapping(
    *,
    rows: list[tuple[Any, Any]],
    existing_commits: set[str] | None = None,
) -> dict[str, str | None]:
    existing_commits = existing_commits or set()
    values: dict[str, str | None] = {}
    for commit_hash, campaign_program_hash in rows:
        normalized_commit = str(commit_hash or "").strip()
        if not normalized_commit or normalized_commit in existing_commits or normalized_commit in values:
            continue
        values[normalized_commit] = (
            str(campaign_program_hash or "").strip() or None
        )
    return values


def snapshot_meta(
    *,
    island_id: str,
    settings: Settings | None = None,
) -> SnapshotMeta:
    """Return lightweight metadata about the stored snapshot (without reconstructing the archive)."""

    base_settings = settings or get_settings()
    dims = max(1, int(base_settings.mapelites_dimensionality_target_dims))
    contract = resolve_objective_contract(base_settings)

    with session_scope() as session:
        stmt = select(MapElitesState).where(
            MapElitesState.island_id == island_id,
        )
        row = session.execute(stmt).scalar_one_or_none()
        snapshot = dict(row.snapshot or {}) if row and row.snapshot else {}
        ensure_supported_snapshot_meta(snapshot, island_id=island_id)
        if row is not None:
            validate_snapshot_contract(
                snapshot,
                island_id=island_id,
                objective_contract=contract,
            )
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
