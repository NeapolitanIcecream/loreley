"""Candidate fate lookup helpers for read-only operator surfaces."""

from __future__ import annotations

from typing import Any, Sequence
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


def load_candidate_fates_for_jobs(jobs: Sequence[object]) -> dict[str, CandidateFate]:
    """Return candidate fates keyed by job id for already-loaded job rows."""

    rows = list(jobs)
    commit_hashes = _unique(
        job_candidate_commit_hash(row)
        for row in rows
    )
    candidates_by_commit, jobs_by_id, archive_cells_by_commit = _load_fate_context(
        commit_hashes=commit_hashes,
        fallback_job_ids=[],
    )

    fates: dict[str, CandidateFate] = {}
    for job in rows:
        job_id = str(getattr(job, "id", "") or "")
        if not job_id:
            continue
        commit_hash = job_candidate_commit_hash(job)
        candidate = candidates_by_commit.get(commit_hash)
        current_cell = archive_cells_by_commit.get(commit_hash)
        fates[job_id] = derive_candidate_fate(
            job=job,
            candidate=candidate,
            current_archive_cell_index=current_cell,
            current_archive_member=current_cell is not None,
        )
        if candidate is not None:
            produced_by_job_id = str(getattr(candidate, "produced_by_job_id", "") or "")
            if produced_by_job_id and produced_by_job_id != job_id:
                fallback_job = jobs_by_id.get(produced_by_job_id)
                if fallback_job is not None:
                    fates[job_id] = derive_candidate_fate(
                        job=fallback_job,
                        candidate=candidate,
                        current_archive_cell_index=current_cell,
                        current_archive_member=current_cell is not None,
                    )
    return fates


def load_candidate_fates_for_commits(commits: Sequence[object]) -> dict[str, CandidateFate]:
    """Return candidate fates keyed by commit hash for already-loaded commit rows."""

    rows = list(commits)
    commit_hashes = _unique(str(getattr(row, "commit_hash", "") or "").strip() for row in rows)
    fallback_job_ids = _unique(str(getattr(row, "job_id", "") or "").strip() for row in rows)
    candidates_by_commit, jobs_by_id, archive_cells_by_commit = _load_fate_context(
        commit_hashes=commit_hashes,
        fallback_job_ids=fallback_job_ids,
    )

    fates: dict[str, CandidateFate] = {}
    for row in rows:
        commit_hash = str(getattr(row, "commit_hash", "") or "").strip()
        if not commit_hash:
            continue
        candidate = candidates_by_commit.get(commit_hash)
        current_cell = archive_cells_by_commit.get(commit_hash)
        job = _job_for_commit_row(row=row, candidate=candidate, jobs_by_id=jobs_by_id)
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
) -> tuple[dict[str, CandidateCommit], dict[str, EvolutionJob], dict[str, int]]:
    if not commit_hashes and not fallback_job_ids:
        return {}, {}, {}

    with session_scope() as session:
        candidates_by_commit: dict[str, CandidateCommit] = {}
        if commit_hashes:
            candidate_rows = list(
                session.execute(
                    select(CandidateCommit).where(CandidateCommit.commit_hash.in_(commit_hashes))
                ).scalars()
            )
            candidates_by_commit = {
                str(row.commit_hash): row
                for row in candidate_rows
                if str(getattr(row, "commit_hash", "") or "").strip()
            }

        job_ids = set(fallback_job_ids)
        for candidate in candidates_by_commit.values():
            produced_by_job_id = str(getattr(candidate, "produced_by_job_id", "") or "").strip()
            if produced_by_job_id:
                job_ids.add(produced_by_job_id)

        job_uuid_values = _uuid_values(job_ids)
        jobs_by_id: dict[str, EvolutionJob] = {}
        if job_uuid_values:
            job_rows = list(
                session.execute(
                    select(EvolutionJob).where(EvolutionJob.id.in_(job_uuid_values))
                ).scalars()
            )
            jobs_by_id = {
                str(row.id): row
                for row in job_rows
                if str(getattr(row, "id", "") or "").strip()
            }

        archive_cells_by_commit: dict[str, int] = {}
        if commit_hashes:
            archive_rows = session.execute(
                select(
                    MapElitesArchiveCell.commit_hash,
                    MapElitesArchiveCell.cell_index,
                ).where(MapElitesArchiveCell.commit_hash.in_(commit_hashes))
            ).all()
            for commit_hash, cell_index in archive_rows:
                normalized = str(commit_hash or "").strip()
                if not normalized or normalized in archive_cells_by_commit:
                    continue
                try:
                    archive_cells_by_commit[normalized] = int(cell_index)
                except (TypeError, ValueError):
                    continue

    return candidates_by_commit, jobs_by_id, archive_cells_by_commit


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


def _uuid_values(values: set[str]) -> list[UUID]:
    result: list[UUID] = []
    for value in values:
        try:
            result.append(UUID(str(value)))
        except (TypeError, ValueError):
            continue
    return result
