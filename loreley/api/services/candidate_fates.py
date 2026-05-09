"""Candidate fate lookup helpers for read-only operator surfaces."""

from __future__ import annotations

from typing import Any, Iterable, Sequence
from uuid import UUID

from sqlalchemy import select

from loreley.core.candidate_fate import CandidateFate, derive_candidate_fate
from loreley.db.base import session_scope
from loreley.db.models import CandidateCommit, EvolutionJob, MapElitesArchiveCell

__all__ = [
    "job_candidate_commit_hash",
    "load_candidate_fates_for_commits",
    "load_candidate_fates_for_jobs",
]


def job_candidate_commit_hash(row: object) -> str:
    return str(
        getattr(row, "result_commit_hash", None)
        or getattr(row, "candidate_commit_hash", None)
        or ""
    ).strip()


ArchiveCellsByCommitIsland = dict[tuple[str, str], int]


def load_candidate_fates_for_jobs(jobs: Sequence[object]) -> dict[str, CandidateFate]:
    """Return candidate fates keyed by job id for already-loaded job rows."""

    rows = list(jobs)
    commit_hashes = _unique(job_candidate_commit_hash(row) for row in rows)
    candidates_by_commit, jobs_by_id, archive_cells_by_commit_island = _load_fate_context(
        commit_hashes=commit_hashes,
        fallback_job_ids=[],
        island_ids=_unique(str(getattr(row, "island_id", "") or "").strip() for row in rows),
    )

    fates: dict[str, CandidateFate] = {}
    for job in rows:
        _add_job_fate(
            fates,
            job=job,
            candidates_by_commit=candidates_by_commit,
            jobs_by_id=jobs_by_id,
            archive_cells_by_commit_island=archive_cells_by_commit_island,
        )
    return fates


def _add_job_fate(
    fates: dict[str, CandidateFate],
    *,
    job: object,
    candidates_by_commit: dict[str, CandidateCommit],
    jobs_by_id: dict[str, EvolutionJob],
    archive_cells_by_commit_island: ArchiveCellsByCommitIsland,
) -> None:
    job_id = str(getattr(job, "id", "") or "")
    if not job_id:
        return

    commit_hash = job_candidate_commit_hash(job)
    candidate = candidates_by_commit.get(commit_hash)
    fates[job_id] = _derive_job_candidate_fate(
        job=job,
        candidate=candidate,
        commit_hash=commit_hash,
        archive_cells_by_commit_island=archive_cells_by_commit_island,
    )
    fallback_fate = _producer_job_fate(
        job=job,
        job_id=job_id,
        candidate=candidate,
        jobs_by_id=jobs_by_id,
        commit_hash=commit_hash,
        archive_cells_by_commit_island=archive_cells_by_commit_island,
    )
    if fallback_fate is not None:
        fates[job_id] = fallback_fate


def _producer_job_fate(
    *,
    job: object,
    job_id: str,
    candidate: CandidateCommit | None,
    jobs_by_id: dict[str, EvolutionJob],
    commit_hash: str,
    archive_cells_by_commit_island: ArchiveCellsByCommitIsland,
) -> CandidateFate | None:
    if candidate is None:
        return None
    produced_by_job_id = str(getattr(candidate, "produced_by_job_id", "") or "")
    if not produced_by_job_id or produced_by_job_id == job_id:
        return None
    fallback_job = jobs_by_id.get(produced_by_job_id)
    if fallback_job is None:
        return None
    return _derive_job_candidate_fate(
        job=fallback_job,
        candidate=candidate,
        commit_hash=commit_hash,
        archive_cells_by_commit_island=archive_cells_by_commit_island,
        fallback_row=job,
    )


def _derive_job_candidate_fate(
    *,
    job: object,
    candidate: CandidateCommit | None,
    commit_hash: str,
    archive_cells_by_commit_island: ArchiveCellsByCommitIsland,
    fallback_row: object | None = None,
) -> CandidateFate:
    island_id = _first_attr_text(job, candidate, fallback_row, attr="island_id")
    current_cell = _archive_cell_for(
        archive_cells_by_commit_island,
        commit_hash=commit_hash,
        island_id=island_id,
    )
    return derive_candidate_fate(
        job=job,
        candidate=candidate,
        current_archive_cell_index=current_cell,
        current_archive_member=current_cell is not None,
    )


def load_candidate_fates_for_commits(commits: Sequence[object]) -> dict[str, CandidateFate]:
    """Return candidate fates keyed by commit hash for already-loaded commit rows."""

    rows = list(commits)
    commit_hashes = _unique(str(getattr(row, "commit_hash", "") or "").strip() for row in rows)
    fallback_job_ids = _unique(str(getattr(row, "job_id", "") or "").strip() for row in rows)
    candidates_by_commit, jobs_by_id, archive_cells_by_commit_island = _load_fate_context(
        commit_hashes=commit_hashes,
        fallback_job_ids=fallback_job_ids,
        island_ids=_unique(str(getattr(row, "island_id", "") or "").strip() for row in rows),
    )

    fates: dict[str, CandidateFate] = {}
    for row in rows:
        commit_hash = str(getattr(row, "commit_hash", "") or "").strip()
        if not commit_hash:
            continue
        candidate = candidates_by_commit.get(commit_hash)
        job = _job_for_commit_row(row=row, candidate=candidate, jobs_by_id=jobs_by_id)
        island_id = _first_attr_text(row, candidate, job, attr="island_id")
        current_cell = _archive_cell_for(
            archive_cells_by_commit_island,
            commit_hash=commit_hash,
            island_id=island_id,
        )
        fates[commit_hash] = derive_candidate_fate(
            job=job,
            candidate=candidate,
            current_archive_cell_index=current_cell,
            current_archive_member=current_cell is not None,
        )
    return fates


def _load_fate_context(
    *,
    commit_hashes: list[str],
    fallback_job_ids: list[str],
    island_ids: list[str],
) -> tuple[dict[str, CandidateCommit], dict[str, EvolutionJob], ArchiveCellsByCommitIsland]:
    if not commit_hashes and not fallback_job_ids:
        return {}, {}, {}

    with session_scope() as session:
        candidates_by_commit = _load_candidates_by_commit(session, commit_hashes)
        job_ids = _job_ids_for_context(fallback_job_ids, candidates_by_commit.values())
        jobs_by_id = _load_jobs_by_id(session, job_ids)
        archive_island_ids = _archive_island_ids(
            island_ids=island_ids,
            candidates=candidates_by_commit.values(),
            jobs=jobs_by_id.values(),
        )
        archive_cells_by_commit_island = _load_archive_cells_by_commit_island(
            session,
            commit_hashes=commit_hashes,
            island_ids=archive_island_ids,
        )

    return candidates_by_commit, jobs_by_id, archive_cells_by_commit_island


def _load_candidates_by_commit(session: Any, commit_hashes: list[str]) -> dict[str, CandidateCommit]:
    if not commit_hashes:
        return {}
    candidate_rows = list(
        session.execute(
            select(CandidateCommit).where(CandidateCommit.commit_hash.in_(commit_hashes))
        ).scalars()
    )
    return {
        str(row.commit_hash): row
        for row in candidate_rows
        if str(getattr(row, "commit_hash", "") or "").strip()
    }


def _job_ids_for_context(
    fallback_job_ids: list[str],
    candidates: Iterable[CandidateCommit],
) -> set[str]:
    job_ids = set(fallback_job_ids)
    for candidate in candidates:
        produced_by_job_id = str(getattr(candidate, "produced_by_job_id", "") or "").strip()
        if produced_by_job_id:
            job_ids.add(produced_by_job_id)
    return job_ids


def _load_jobs_by_id(session: Any, job_ids: set[str]) -> dict[str, EvolutionJob]:
    job_uuid_values = _uuid_values(job_ids)
    if not job_uuid_values:
        return {}
    job_rows = list(
        session.execute(
            select(EvolutionJob).where(EvolutionJob.id.in_(job_uuid_values))
        ).scalars()
    )
    return {
        str(row.id): row
        for row in job_rows
        if str(getattr(row, "id", "") or "").strip()
    }


def _archive_island_ids(
    *,
    island_ids: list[str],
    candidates: Iterable[CandidateCommit],
    jobs: Iterable[EvolutionJob],
) -> set[str]:
    result = set(island_ids)
    for candidate in candidates:
        _add_attr_text(result, candidate, "island_id")
    for job in jobs:
        _add_attr_text(result, job, "island_id")
    return result


def _load_archive_cells_by_commit_island(
    session: Any,
    *,
    commit_hashes: list[str],
    island_ids: set[str],
) -> ArchiveCellsByCommitIsland:
    if not commit_hashes or not island_ids:
        return {}
    archive_rows = session.execute(
        select(
            MapElitesArchiveCell.commit_hash,
            MapElitesArchiveCell.island_id,
            MapElitesArchiveCell.cell_index,
        ).where(
            MapElitesArchiveCell.commit_hash.in_(commit_hashes),
            MapElitesArchiveCell.island_id.in_(list(island_ids)),
        )
    ).all()
    archive_cells_by_commit_island: ArchiveCellsByCommitIsland = {}
    for commit_hash, island_id, cell_index in archive_rows:
        _set_archive_cell(archive_cells_by_commit_island, commit_hash, island_id, cell_index)
    return archive_cells_by_commit_island


def _set_archive_cell(
    archive_cells_by_commit_island: ArchiveCellsByCommitIsland,
    commit_hash: object,
    island_id: object,
    cell_index: object,
) -> None:
    normalized_commit = str(commit_hash or "").strip()
    normalized_island = str(island_id or "").strip()
    key = (normalized_commit, normalized_island)
    if not normalized_commit or not normalized_island or key in archive_cells_by_commit_island:
        return
    try:
        archive_cells_by_commit_island[key] = int(cell_index)
    except (TypeError, ValueError):
        return


def _job_for_commit_row(
    *,
    row: object,
    candidate: CandidateCommit | None,
    jobs_by_id: dict[str, EvolutionJob],
) -> EvolutionJob | None:
    candidate_job_id = str(getattr(candidate, "produced_by_job_id", "") or "").strip()
    if candidate_job_id and candidate_job_id in jobs_by_id:
        return jobs_by_id[candidate_job_id]
    row_job_id = str(getattr(row, "job_id", "") or "").strip()
    if row_job_id:
        return jobs_by_id.get(row_job_id)
    return None


def _unique(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _archive_cell_for(
    archive_cells_by_commit_island: ArchiveCellsByCommitIsland,
    *,
    commit_hash: str,
    island_id: str,
) -> int | None:
    normalized_commit = str(commit_hash or "").strip()
    normalized_island = str(island_id or "").strip()
    if not normalized_commit or not normalized_island:
        return None
    return archive_cells_by_commit_island.get((normalized_commit, normalized_island))


def _first_attr_text(*rows: object | None, attr: str) -> str:
    for row in rows:
        value = str(getattr(row, attr, "") or "").strip()
        if value:
            return value
    return ""


def _add_attr_text(values: set[str], row: object | None, attr: str) -> None:
    value = str(getattr(row, attr, "") or "").strip()
    if value:
        values.add(value)


def _uuid_values(values: set[str]) -> list[UUID]:
    result: list[UUID] = []
    for value in values:
        try:
            result.append(UUID(str(value)))
        except (TypeError, ValueError):
            continue
    return result
