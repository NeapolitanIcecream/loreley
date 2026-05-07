"""MAP-Elites archive access for the UI API (read-only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select

from loreley.api.pagination import PaginationCursorError, decode_cursor, encode_cursor, normalize_pagination
from loreley.config import Settings, get_settings, resolve_default_island_id
from loreley.core.map_elites.types import MapElitesRecord, materialize_solution
from loreley.core.map_elites.snapshot import ensure_supported_snapshot_meta
from loreley.db.base import session_scope
from loreley.db.models import (
    CandidateCommit,
    CommitCard,
    EvolutionJob,
    MapElitesArchiveCell,
    MapElitesPcaHistory,
    MapElitesState,
    Metric,
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


def _fitness_metric_name(settings: Settings) -> str | None:
    metric_name = str(getattr(settings, "mapelites_fitness_metric", "") or "").strip()
    return metric_name or None


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
    return [resolve_default_island_id(base_settings)]


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
    metric_name = _fitness_metric_name(base_settings)
    higher_is_better = bool(base_settings.mapelites_fitness_higher_is_better)

    with session_scope() as session:
        occupied, qd_score, best_objective = session.execute(
            select(
                func.count(MapElitesArchiveCell.cell_index),
                func.coalesce(func.sum(MapElitesArchiveCell.objective), 0.0),
                func.max(MapElitesArchiveCell.objective),
            ).where(
                MapElitesArchiveCell.island_id == island_id,
            )
        ).one()
        best_metric_value = None
        if metric_name:
            order_column = Metric.value.desc() if higher_is_better else Metric.value.asc()
            best_metric_value = session.execute(
                select(Metric.value)
                .join(CommitCard, CommitCard.id == Metric.commit_card_id)
                .join(
                    MapElitesArchiveCell,
                    MapElitesArchiveCell.commit_hash == CommitCard.commit_hash,
                )
                .where(
                    MapElitesArchiveCell.island_id == island_id,
                    Metric.name == metric_name,
                )
                .order_by(order_column)
                .limit(1)
            ).scalar_one_or_none()

    occupied_value = int(occupied or 0)
    qd_score_value = float(qd_score or 0.0)
    best_objective_value = float(best_objective) if best_objective is not None else 0.0
    best_value = float(best_metric_value) if best_metric_value is not None else best_objective_value
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
        "best_objective": best_objective_value,
        "metric_name": metric_name,
        "higher_is_better": higher_is_better,
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
    metric_name = _fitness_metric_name(base_settings)
    higher_is_better = bool(base_settings.mapelites_fitness_higher_is_better)
    limit, offset = normalize_pagination(limit, offset)

    with session_scope() as session:
        rows = list(
            session.execute(
                select(MapElitesArchiveCell)
                .where(MapElitesArchiveCell.island_id == island_id)
                .order_by(MapElitesArchiveCell.cell_index.asc())
                .limit(limit)
                .offset(offset)
            ).scalars().all()
        )
        metric_values_by_commit: dict[str, float] = {}
        baseline_by_commit: dict[str, Any | None] = {}
        if metric_name and rows:
            commit_hashes = _commit_hashes_from_rows(rows)
            metric_stmt = (
                select(CommitCard.commit_hash, Metric.value)
                .join(Metric, Metric.commit_card_id == CommitCard.id)
                .where(
                    CommitCard.commit_hash.in_(commit_hashes),
                    Metric.name == metric_name,
                )
            )
            metric_values_by_commit = {
                str(commit_hash): float(value)
                for commit_hash, value in session.execute(metric_stmt).all()
                if commit_hash and value is not None
            }
            if _baseline_lookup_configured(settings=base_settings, metric_name=metric_name):
                baseline_by_commit = _load_baselines_by_commit(
                    session=session,
                    settings=base_settings,
                    commit_hashes=commit_hashes,
                )

    return _build_records_from_rows(
        rows=rows,
        island_id=island_id,
        metric_values_by_commit=metric_values_by_commit,
        metric_name=metric_name,
        higher_is_better=higher_is_better,
        baseline_by_commit=baseline_by_commit,
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
    metric_name = _fitness_metric_name(base_settings)
    higher_is_better = bool(base_settings.mapelites_fitness_higher_is_better)
    limit, _ = normalize_pagination(limit, 0)

    with session_scope() as session:
        stmt = (
            select(MapElitesArchiveCell)
            .where(MapElitesArchiveCell.island_id == island_id)
            .order_by(MapElitesArchiveCell.cell_index.asc())
        )
        if cursor:
            try:
                payload = decode_cursor(cursor)
                last_cell_index = int(payload.get("cell_index"))
            except (PaginationCursorError, TypeError, ValueError) as exc:
                raise PaginationCursorError("Archive records cursor is invalid.") from exc
            stmt = stmt.where(MapElitesArchiveCell.cell_index > last_cell_index)
        stmt = stmt.limit(limit + 1)
        rows = list(session.execute(stmt).scalars().all())
        items_rows = rows[:limit]
        metric_values_by_commit: dict[str, float] = {}
        baseline_by_commit: dict[str, Any | None] = {}
        if metric_name and items_rows:
            commit_hashes = _commit_hashes_from_rows(items_rows)
            metric_stmt = (
                select(CommitCard.commit_hash, Metric.value)
                .join(Metric, Metric.commit_card_id == CommitCard.id)
                .where(
                    CommitCard.commit_hash.in_(commit_hashes),
                    Metric.name == metric_name,
                )
            )
            metric_values_by_commit = {
                str(commit_hash): float(value)
                for commit_hash, value in session.execute(metric_stmt).all()
                if commit_hash and value is not None
            }
            if _baseline_lookup_configured(settings=base_settings, metric_name=metric_name):
                baseline_by_commit = _load_baselines_by_commit(
                    session=session,
                    settings=base_settings,
                    commit_hashes=commit_hashes,
                )

    records = _build_records_from_rows(
        rows=items_rows,
        island_id=island_id,
        metric_values_by_commit=metric_values_by_commit,
        metric_name=metric_name,
        higher_is_better=higher_is_better,
        baseline_by_commit=baseline_by_commit,
    )
    next_cursor = encode_cursor({"cell_index": records[-1].cell_index}) if len(rows) > limit and records else None
    return ArchiveRecordPage(items=records, next_cursor=next_cursor)


def _build_records_from_rows(
    *,
    rows: list[MapElitesArchiveCell],
    island_id: str,
    metric_values_by_commit: dict[str, float],
    metric_name: str | None,
    higher_is_better: bool,
    baseline_by_commit: dict[str, Any | None] | None = None,
    baseline: Any | None = None,
) -> list[MapElitesRecord]:
    records: list[MapElitesRecord] = []
    for row in rows:
        commit_hash = str(row.commit_hash or "")
        objective = float(row.objective or 0.0)
        metric_value = metric_values_by_commit.get(commit_hash)
        row_baseline = (
            baseline_by_commit.get(commit_hash)
            if baseline_by_commit is not None
            else baseline
        )
        delta_from_baseline = improvement_from_baseline(
            candidate_value=metric_value,
            baseline=row_baseline,
        )
        records.append(
            MapElitesRecord(
                commit_hash=commit_hash,
                island_id=str(row.island_id or island_id),
                cell_index=int(row.cell_index),
                fitness=float(metric_value) if metric_value is not None else objective,
                objective=objective,
                metric_value=float(metric_value) if metric_value is not None else None,
                metric_name=metric_name,
                higher_is_better=higher_is_better,
                campaign_baseline_id=(
                    str(getattr(row_baseline, "id", ""))
                    if getattr(row_baseline, "id", None)
                    else None
                ),
                baseline_key_hash=getattr(row_baseline, "baseline_key_hash", None),
                baseline_status=getattr(row_baseline, "status", None),
                delta_from_root_baseline=delta_from_baseline,
                measures=tuple(float(v) for v in (row.measures or ())),
                solution=materialize_solution(
                    measures=row.measures or (),
                    solution=row.solution or (),
                ),
                timestamp=float(row.timestamp or 0.0),
            )
        )
    return records


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


def _baseline_lookup_configured(*, settings: Settings, metric_name: str | None) -> bool:
    root_commit = str(getattr(settings, "mapelites_experiment_root_commit", "") or "").strip()
    return bool(root_commit and metric_name)


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
    campaign_hash_by_commit: dict[str, str | None] = {}
    candidate_rows = session.execute(
        select(CandidateCommit.commit_hash, CandidateCommit.campaign_program_hash).where(
            CandidateCommit.commit_hash.in_(commit_hashes),
        )
    ).all()
    for commit_hash, campaign_program_hash in candidate_rows:
        normalized_commit = str(commit_hash or "").strip()
        if not normalized_commit or normalized_commit in campaign_hash_by_commit:
            continue
        campaign_hash_by_commit[normalized_commit] = (
            str(campaign_program_hash or "").strip() or None
        )

    missing_commits = [
        commit_hash
        for commit_hash in commit_hashes
        if commit_hash not in campaign_hash_by_commit
    ]
    if not missing_commits:
        return campaign_hash_by_commit

    job_rows = session.execute(
        select(CommitCard.commit_hash, EvolutionJob.campaign_program_hash)
        .join(EvolutionJob, EvolutionJob.id == CommitCard.job_id)
        .where(CommitCard.commit_hash.in_(missing_commits))
    ).all()
    for commit_hash, campaign_program_hash in job_rows:
        normalized_commit = str(commit_hash or "").strip()
        if not normalized_commit or normalized_commit in campaign_hash_by_commit:
            continue
        campaign_hash_by_commit[normalized_commit] = (
            str(campaign_program_hash or "").strip() or None
        )
    return campaign_hash_by_commit


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
