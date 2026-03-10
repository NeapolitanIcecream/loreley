from __future__ import annotations

"""Result ingestion and MAP-Elites maintenance for the evolution scheduler.

The public API here is intentionally small so that ``loreley.scheduler.main``
can delegate all ingestion responsibilities to this module.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import UUID

from git import Repo
from loguru import logger
from rich.console import Console
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from loreley.config import Settings, resolve_default_island_id
from loreley.core.contracts import clamp_text, normalize_single_line
from loreley.core.git import RepositoryError as GitRepositoryError, require_commit
from loreley.core.map_elites.map_elites import MapElitesManager
from loreley.core.worker.evaluator import EvaluationContext, EvaluationError, Evaluator
from loreley.core.worker.repository import RepositoryError, WorkerRepository
from loreley.db.base import session_scope
from loreley.db.models import CommitCard, EvolutionJob, JobStatus, Metric

log = logger.bind(module="scheduler.ingestion")

_INGESTION_REASON_MAX_CHARS = 4096
_INGESTION_MESSAGE_MAX_CHARS = 4096


class IngestionError(RuntimeError):
    """Raised when result ingestion cannot proceed for a commit."""


@dataclass(slots=True, frozen=True)
class JobSnapshot:
    """Immutable view of a job that completed and awaits ingestion."""

    job_id: UUID
    base_commit_hash: str | None
    island_id: str | None
    result_commit_hash: str
    completed_at: datetime | None


@dataclass(slots=True)
class MapElitesIngestion:
    """Handle result ingestion and root‑commit initialisation for MAP‑Elites."""

    settings: Settings
    console: Console
    repo_root: Path
    repo: Repo
    manager: MapElitesManager
    _prefetched_metrics_payload_by_commit: dict[str, list[dict[str, Any]]] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _prefetched_metrics_errors_by_commit: dict[str, str] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    # Public API ------------------------------------------------------------

    def ingest_completed_jobs(self) -> int:
        """Ingest a batch of newly succeeded jobs into MAP-Elites.

        This loop is best-effort per job: failures are recorded onto the job row
        and do not abort the scheduler tick.
        """

        batch = max(0, int(self.settings.scheduler_ingest_batch_size))
        if batch == 0:
            return 0
        snapshots = self._jobs_requiring_ingestion(limit=batch)
        if not snapshots:
            return 0
        ingested = 0

        self._prefetched_metrics_payload_by_commit = None
        self._prefetched_metrics_errors_by_commit = None
        try:
            with session_scope() as prefetch_session:
                (
                    self._prefetched_metrics_payload_by_commit,
                    self._prefetched_metrics_errors_by_commit,
                ) = self._load_metrics_payload_batch(
                    self._canonicalize_prefetch_commit_hashes(snapshots),
                    session=prefetch_session,
                )

            for snapshot in snapshots:
                try:
                    if self._ingest_snapshot(snapshot):
                        ingested += 1
                except Exception as exc:
                    raw_reason = normalize_single_line(str(exc))
                    reason = clamp_text(raw_reason, _INGESTION_REASON_MAX_CHARS) or None
                    reason_display = reason or raw_reason or exc.__class__.__name__
                    self.console.log(
                        f"[bold red]Unhandled ingestion error[/] job={snapshot.job_id} commit={snapshot.result_commit_hash} reason={reason_display}",
                    )
                    log.exception(
                        "Unhandled ingestion error for job {} commit {}: {}",
                        snapshot.job_id,
                        snapshot.result_commit_hash,
                        reason_display,
                    )
                    self._record_ingestion_state(snapshot, status="failed", reason=reason_display)
        finally:
            self._prefetched_metrics_payload_by_commit = None
            self._prefetched_metrics_errors_by_commit = None
        return ingested

    def _canonicalize_prefetch_commit_hashes(self, snapshots: Sequence[JobSnapshot]) -> list[str]:
        """Best-effort canonicalization to improve metrics prefetch hit ratio."""

        resolved_by_raw: dict[str, str] = {}
        canonicalized: list[str] = []
        for snapshot in snapshots:
            raw_hash = str(snapshot.result_commit_hash or "").strip()
            if not raw_hash:
                continue
            resolved = resolved_by_raw.get(raw_hash)
            if resolved is None:
                resolved = self._canonicalize_commit_hash_local(raw_hash)
                resolved_by_raw[raw_hash] = resolved
            canonicalized.append(resolved)
        return canonicalized

    def _canonicalize_commit_hash_local(self, commit_hash: str) -> str:
        """Resolve a commit hash locally without network operations."""

        raw = str(commit_hash or "").strip()
        if not raw:
            return raw

        try:
            resolved = self.repo.commit(raw)
        except Exception:
            return raw

        canonical = str(getattr(resolved, "hexsha", "") or "").strip()
        return canonical or raw

    def initialise_root_commit(self, commit_hash: str) -> None:
        """Ensure the configured root commit is present in DB and evaluated.

        This helper is idempotent and safe to call on every scheduler startup.
        Repo-state bootstrap failures are fatal because runtime ingestion is incremental-only.
        """

        try:
            commit_hash = self._ensure_commit_available(commit_hash)
        except IngestionError as exc:
            self.console.log(
                f"[bold red]Failed to initialise root commit[/] commit={commit_hash} reason={exc}",
            )
            log.error("Failed to initialise root commit {}: {}", commit_hash, exc)
            raise

        # Commit metadata is required for downstream observability / UI.
        self._ensure_root_commit_metadata(commit_hash)

        # Bootstrap repo-state aggregates so runtime ingestion can stay incremental-only.
        self._ensure_root_commit_repo_state_bootstrap(commit_hash)

        # Root commit evaluation is best-effort: failures do not prevent the scheduler loop.
        try:
            self._ensure_root_commit_evaluated(commit_hash)
        except Exception as exc:
            self.console.log(
                f"[bold red]Root commit evaluation failed[/] commit={commit_hash} reason={exc}",
            )
            log.exception("Root commit evaluation failed for {}: {}", commit_hash, exc)

    # Job result ingestion --------------------------------------------------

    def _jobs_requiring_ingestion(self, *, limit: int) -> list[JobSnapshot]:
        batch_limit = max(limit * 4, 32)
        snapshots: list[JobSnapshot] = []
        now = datetime.now(timezone.utc)
        with session_scope() as session:
            status_norm = func.lower(func.trim(func.coalesce(EvolutionJob.ingestion_status, "")))
            commit_norm = func.trim(func.coalesce(EvolutionJob.result_commit_hash, ""))
            # Always prioritise fresh jobs so a backlog of failed ingestions
            # cannot block the scheduler from ingesting new results.
            failed_last = case((status_norm == "failed", 1), else_=0)
            stmt = (
                select(EvolutionJob)
                .where(
                    EvolutionJob.status == JobStatus.SUCCEEDED,
                )
                .where(status_norm.not_in(("succeeded", "skipped")))
                .where(commit_norm != "")
                .order_by(failed_last.asc(), EvolutionJob.completed_at.asc())
                .limit(batch_limit)
            )
            rows = list(session.execute(stmt).scalars())
            for job in rows:
                status = (job.ingestion_status or "").strip().lower()
                if status in {"succeeded", "skipped"}:
                    continue
                if status == "failed" and self._should_backoff_failed_job(job, now=now):
                    continue
                commit_hash = (job.result_commit_hash or "").strip()
                if not commit_hash:
                    continue

                snapshots.append(
                    JobSnapshot(
                        job_id=job.id,
                        base_commit_hash=job.base_commit_hash,
                        island_id=job.island_id,
                        result_commit_hash=commit_hash,
                        completed_at=job.completed_at,
                    )
                )
                if len(snapshots) >= limit:
                    break
        return snapshots

    def _should_backoff_failed_job(self, job: EvolutionJob, *, now: datetime) -> bool:
        last_attempt = getattr(job, "ingestion_last_attempt_at", None)
        if not isinstance(last_attempt, datetime):
            return False
        attempts = int(getattr(job, "ingestion_attempts", 0) or 0)
        attempts = max(1, attempts)
        backoff_seconds = self._retry_backoff_seconds(attempts=attempts)

        last_utc = last_attempt if last_attempt.tzinfo else last_attempt.replace(tzinfo=timezone.utc)
        try:
            last_utc = last_utc.astimezone(timezone.utc)
        except Exception:
            last_utc = last_utc.replace(tzinfo=timezone.utc)
        next_attempt_at = last_utc + timedelta(seconds=backoff_seconds)
        return next_attempt_at > now

    def _retry_backoff_seconds(self, *, attempts: int) -> float:
        poll = float(getattr(self.settings, "scheduler_poll_interval_seconds", 30.0))
        base = max(30.0, poll)
        cap = max(base, 3600.0)
        exponent = max(0, int(attempts) - 1)
        return float(min(cap, base * (2.0**exponent)))

    def _ingest_snapshot(
        self,
        snapshot: JobSnapshot,
        *,
        snapshot_session: Session | None = None,
    ) -> bool:
        raw_commit_hash = (snapshot.result_commit_hash or "").strip()
        commit_hash = raw_commit_hash
        if not commit_hash:
            return False
        try:
            commit_hash = self._ensure_commit_available(commit_hash)
        except IngestionError as exc:
            raw_reason = normalize_single_line(str(exc))
            reason = clamp_text(raw_reason, _INGESTION_REASON_MAX_CHARS) or None
            reason_display = reason or raw_reason or exc.__class__.__name__
            self.console.log(
                f"[bold red]Commit unavailable for ingestion[/] job={snapshot.job_id} commit={commit_hash} reason={reason_display}",
            )
            log.warning(
                "Commit unavailable for ingestion job={} commit={} reason={}",
                snapshot.job_id,
                commit_hash,
                reason_display,
            )
            self._record_ingestion_state(
                snapshot,
                status="failed",
                reason=reason_display,
            )
            return False

        metrics_payload: list[dict[str, Any]] = []
        metrics_error = None
        metrics_errors_by_commit = self._prefetched_metrics_errors_by_commit
        if metrics_errors_by_commit:
            metrics_error = metrics_errors_by_commit.get(commit_hash) or metrics_errors_by_commit.get(raw_commit_hash)
        if metrics_error:
            raise ValueError(metrics_error)

        metrics_payload_by_commit = self._prefetched_metrics_payload_by_commit
        if metrics_payload_by_commit:
            if commit_hash in metrics_payload_by_commit:
                metrics_payload = list(metrics_payload_by_commit.get(commit_hash) or [])
            elif raw_commit_hash in metrics_payload_by_commit:
                metrics_payload = list(metrics_payload_by_commit.get(raw_commit_hash) or [])

            # If the commit hash was canonicalised (e.g. short -> full hash),
            # ensure we don't silently miss metrics due to a cache-key mismatch.
            if not metrics_payload and raw_commit_hash and raw_commit_hash != commit_hash:
                metrics_payload = self._load_metrics_payload_for_commit(commit_hash)
        else:
            metrics_payload = self._load_metrics_payload_for_commit(commit_hash)
        try:
            insertion = self.manager.ingest(
                commit_hash=commit_hash,
                metrics=metrics_payload,
                island_id=snapshot.island_id,
                repo_root=self.repo_root,
                snapshot_session=snapshot_session,
            )
        except Exception as exc:
            raw_reason = normalize_single_line(str(exc))
            reason = clamp_text(raw_reason, _INGESTION_REASON_MAX_CHARS) or None
            reason_display = reason or raw_reason or exc.__class__.__name__
            self.console.log(
                f"[bold red]MAP-Elites ingest failed[/] job={snapshot.job_id} commit={commit_hash} reason={reason_display}",
            )
            log.exception(
                "Failed to ingest commit {} for job {}: {}",
                commit_hash,
                snapshot.job_id,
                reason_display,
            )
            self._record_ingestion_state(
                snapshot,
                status="failed",
                reason=reason_display,
            )
            return False
        if insertion.record:
            self.console.log(
                f"[green]Updated archive[/] job={snapshot.job_id} commit={commit_hash} "
                f"cell={insertion.record.cell_index} Δ={insertion.delta:.4f}",
            )
        else:
            self.console.log(
                f"[yellow]Archive unchanged[/] job={snapshot.job_id} commit={commit_hash} status={insertion.status}",
            )
        self._record_ingestion_state(
            snapshot,
            status="succeeded" if insertion.inserted else "skipped",
            delta=insertion.delta,
            status_code=insertion.status,
            message=insertion.message,
            record=insertion.record,
        )
        return bool(insertion.record)

    def _load_metrics_payload_batch(
        self,
        commit_hashes: Sequence[str],
        *,
        session: Session,
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
        """Load commit metrics payload in a single DB round-trip.

        Returns payloads grouped by commit_hash and a per-commit error map.
        Any commit that fails payload building is reported in the error map and
        gets an empty metrics payload so ingestion can fail per job.
        """

        requested: list[str] = []
        for raw in commit_hashes:
            value = str(raw or "").strip()
            if value:
                requested.append(value)

        unique = list(dict.fromkeys(requested))
        payload_by_commit: dict[str, list[dict[str, Any]]] = {commit_hash: [] for commit_hash in unique}
        errors_by_commit: dict[str, str] = {}
        if not unique:
            return payload_by_commit, errors_by_commit

        stmt = (
            select(
                CommitCard.commit_hash,
                Metric.name,
                Metric.value,
                Metric.unit,
                Metric.higher_is_better,
            )
            .join(Metric, Metric.commit_card_id == CommitCard.id)
            .where(CommitCard.commit_hash.in_(unique))
            .order_by(CommitCard.commit_hash.asc(), Metric.name.asc())
        )
        for commit_hash, name, value, unit, higher_is_better in session.execute(stmt):
            commit_key = str(commit_hash or "").strip()
            if not commit_key:
                continue
            if commit_key in errors_by_commit:
                continue
            try:
                payload_by_commit.setdefault(commit_key, []).append(
                    {
                        "name": name,
                        "value": float(value),
                        "unit": unit,
                        "higher_is_better": bool(higher_is_better),
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive
                errors_by_commit[commit_key] = (
                    "Failed to build metrics payload "
                    f"(commit={commit_key} metric={name!r} reason={exc})."
                )
                payload_by_commit[commit_key] = []

        return payload_by_commit, errors_by_commit

    def _load_metrics_payload_for_commit(self, commit_hash: str) -> list[dict[str, Any]]:
        """Load metrics payload for a single commit (join-based query)."""

        value = str(commit_hash or "").strip()
        if not value:
            return []

        with session_scope() as session:
            stmt = (
                select(Metric)
                .join(CommitCard, CommitCard.id == Metric.commit_card_id)
                .where(CommitCard.commit_hash == value)
                .order_by(Metric.name.asc())
            )
            rows = list(session.scalars(stmt).all())

        payload: list[dict[str, Any]] = []
        for row in rows:
            payload.append(
                {
                    "name": row.name,
                    "value": float(row.value),
                    "unit": row.unit,
                    "higher_is_better": bool(row.higher_is_better),
                }
            )
        return payload

    def _record_ingestion_state(
        self,
        snapshot: JobSnapshot,
        *,
        status: str,
        reason: str | None = None,
        delta: float | None = None,
        status_code: int | None = None,
        message: str | None = None,
        record: Any | None = None,
    ) -> None:
        clamped_reason = self._clamp_ingestion_text(reason, max_chars=_INGESTION_REASON_MAX_CHARS)
        clamped_message = self._clamp_ingestion_text(message, max_chars=_INGESTION_MESSAGE_MAX_CHARS)
        with session_scope() as session:
            job = session.get(EvolutionJob, snapshot.job_id)
            if not job:
                return
            job.ingestion_attempts = int(getattr(job, "ingestion_attempts", 0) or 0) + 1
            job.ingestion_status = status
            job.ingestion_last_attempt_at = datetime.now(timezone.utc)
            job.ingestion_reason = clamped_reason
            job.ingestion_delta = delta
            job.ingestion_status_code = status_code
            job.ingestion_message = clamped_message
            if record is not None and hasattr(record, "cell_index"):
                job.ingestion_cell_index = int(getattr(record, "cell_index"))
            else:
                job.ingestion_cell_index = None

    @staticmethod
    def _clamp_ingestion_text(value: str | None, *, max_chars: int) -> str | None:
        if value is None:
            return None
        normalized = normalize_single_line(value)
        if not normalized:
            return None
        return clamp_text(normalized, max_chars) or None

    # Git helpers -----------------------------------------------------------

    def _ensure_commit_available(self, commit_hash: str) -> str:
        try:
            return require_commit(self.repo, commit_hash, console=self.console)
        except GitRepositoryError as exc:
            raise IngestionError(str(exc)) from exc

    # Root commit initialisation --------------------------------------------

    def _ensure_root_commit_evaluated(self, commit_hash: str) -> None:
        """Run a one-off evaluation for the root commit to populate metrics.

        This helper is idempotent: if any Metric rows already exist for the
        commit, the evaluation step is skipped. Failures are logged but do not
        prevent the scheduler from running.
        """

        # Skip evaluation when metrics already exist for this commit.
        with session_scope() as session:
            commit_row = session.execute(
                select(CommitCard).where(CommitCard.commit_hash == commit_hash)
            ).scalar_one_or_none()
            if commit_row is not None:
                existing = session.execute(
                    select(Metric.id)
                    .where(Metric.commit_card_id == commit_row.id)
                    .limit(1)
                ).first()
                if existing is not None:
                    return

        # Prepare a detached checkout of the root commit using the worker repo.
        try:
            worker_repo = WorkerRepository(self.settings)
        except RepositoryError as exc:
            self.console.log(
                "[yellow]Skipping root commit evaluation; worker repository is not configured[/] "
                f"commit={commit_hash} reason={exc}",
            )
            log.warning(
                "Skipping root commit evaluation for {} because worker repository "
                "could not be initialised: {}",
                commit_hash,
                exc,
            )
            return

        try:
            with worker_repo.checkout_lease_for_job(
                job_id=None,
                base_commit=commit_hash,
                create_branch=False,
            ) as checkout:
                goal = f"Baseline evaluation for root commit {commit_hash}"
                default_island = resolve_default_island_id(self.settings)
                payload: dict[str, Any] = {
                    "job": {
                        "id": None,
                        "island_id": default_island,
                        "goal": goal,
                        "constraints": [],
                        "acceptance_criteria": [],
                        "notes": [],
                    },
                    "plan": {
                        "summary": goal,
                    },
                }

                evaluator = Evaluator(self.settings)
                context = EvaluationContext(
                    worktree=checkout.worktree,
                    base_commit_hash=None,
                    candidate_commit_hash=commit_hash,
                    job_id=None,
                    goal=goal,
                    payload=payload,
                    plan_summary=goal,
                    metadata={
                        "root_commit": True,
                    },
                )  # type: ignore[call-arg]

                try:
                    result = evaluator.evaluate(context)
                except EvaluationError as exc:
                    self.console.log(
                        f"[bold red]Root commit evaluation failed[/] commit={commit_hash} reason={exc}",
                    )
                    log.error("Root commit evaluation failed for {}: {}", commit_hash, exc)
                    return
        except RepositoryError as exc:
            self.console.log(
                "[yellow]Skipping root commit evaluation; checkout failed[/] "
                f"commit={commit_hash} reason={exc}",
            )
            log.warning(
                "Skipping root commit evaluation for {} because checkout failed: {}",
                commit_hash,
                exc,
            )
            return

        metrics_payload = [metric.as_dict() for metric in result.metrics]

        with session_scope() as session:
            commit_row = session.execute(
                select(CommitCard).where(CommitCard.commit_hash == commit_hash)
            ).scalar_one_or_none()
            if commit_row is None:
                # Ensure the commit record exists before writing metrics so that
                # FK constraints are satisfied even if metadata initialisation
                # was skipped or the DB was manually reset.
                git_commit = self.repo.commit(commit_hash)
                parent_hash = git_commit.parents[0].hexsha if git_commit.parents else None
                author = getattr(getattr(git_commit, "author", None), "name", None)
                message = getattr(git_commit, "message", None)
                subject = (
                    str(message or "").splitlines()[0].strip()
                    if message
                    else f"Commit {commit_hash}"
                )
                subject = subject[:72].strip() or f"Commit {commit_hash}"
                default_island = resolve_default_island_id(self.settings)

                commit_row = CommitCard(
                    commit_hash=commit_hash,
                    parent_commit_hash=parent_hash,
                    island_id=default_island,
                    author=author,
                    subject=subject,
                    change_summary="Root baseline commit.",
                    evaluation_summary=result.summary,
                    tags=[],
                    key_files=[],
                    highlights=["Root baseline commit."],
                    job_id=None,
                )
                session.add(commit_row)
            else:
                commit_row.evaluation_summary = result.summary

            for metric in result.metrics:
                existing_metric = session.execute(
                    select(Metric).where(
                        Metric.commit_card_id == commit_row.id,
                        Metric.name == metric.name,
                    )
                ).scalar_one_or_none()
                if existing_metric:
                    existing_metric.value = float(metric.value)
                    existing_metric.unit = metric.unit
                    existing_metric.higher_is_better = bool(metric.higher_is_better)
                    existing_metric.details = dict(metric.details or {})
                else:
                    session.add(
                        Metric(
                            commit=commit_row,
                            name=metric.name,
                            value=metric.value,
                            unit=metric.unit,
                            higher_is_better=metric.higher_is_better,
                            details=dict(metric.details or {}),
                        )
                    )

        self.console.log(
            "[green]Evaluated root commit[/] commit={} metrics={}".format(
                commit_hash,
                len(metrics_payload),
            ),
        )
        log.info(
            "Root commit evaluation completed for {} with {} metrics",
            commit_hash,
            len(metrics_payload),
        )

    def _ensure_root_commit_repo_state_bootstrap(self, commit_hash: str) -> None:
        """Bootstrap the repo-state aggregate for the experiment baseline commit."""
        from loreley.core.map_elites.repository_state_embedding import (
            bootstrap_repository_state_aggregate,
        )

        embedding, stats = bootstrap_repository_state_aggregate(
            commit_hash=commit_hash,
            repo_root=self.repo_root,
            settings=self.settings,
            repo=self.repo,
        )

        if not embedding or not embedding.vector or stats.files_aggregated <= 0:
            raise IngestionError(
                "Repo-state bootstrap produced no embedding; "
                f"eligible_files={stats.eligible_files} files_aggregated={stats.files_aggregated} "
                f"skipped_failed_embedding={stats.skipped_failed_embedding} commit={commit_hash}."
            )
        canonical = str(getattr(self.repo.commit(commit_hash), "hexsha", "") or "").strip()

        self.console.log(
            "[green]Bootstrapped repo-state baseline aggregate[/] commit={} eligible_files={} files_aggregated={} dims={}".format(
                canonical,
                stats.eligible_files,
                stats.files_aggregated,
                embedding.dimensions,
            )
        )
        log.info(
            "Bootstrapped repo-state baseline aggregate commit={} eligible_files={} files_aggregated={} dims={}",
            canonical,
            stats.eligible_files,
            stats.files_aggregated,
            embedding.dimensions,
        )

    def _ensure_root_commit_metadata(self, commit_hash: str) -> None:
        """Create or update CommitCard for the root commit."""

        git_commit = self.repo.commit(commit_hash)
        parent_hash = git_commit.parents[0].hexsha if git_commit.parents else None
        author = getattr(getattr(git_commit, "author", None), "name", None)
        message = getattr(git_commit, "message", None)

        with session_scope() as session:
            stmt = select(CommitCard).where(
                CommitCard.commit_hash == commit_hash,
            )
            existing = session.execute(stmt).scalar_one_or_none()
            default_island = resolve_default_island_id(self.settings)

            if existing:
                updated = False
                if existing.island_id is None:
                    existing.island_id = default_island
                    updated = True
                if not getattr(existing, "highlights", None):
                    existing.highlights = ["Root baseline commit."]
                    updated = True
                if not getattr(existing, "subject", None):
                    subject = str(message or "").splitlines()[0].strip() if message else f"Commit {commit_hash}"
                    existing.subject = subject[:72].strip() or f"Commit {commit_hash}"
                    updated = True
                if not getattr(existing, "change_summary", None):
                    existing.change_summary = "Root baseline commit."
                    updated = True
                if updated:
                    self.console.log(
                        "[cyan]Updated root commit metadata[/] commit={} island={}".format(
                            commit_hash,
                            existing.island_id,
                        ),
                    )
                return

            subject = str(message or "").splitlines()[0].strip() if message else f"Commit {commit_hash}"
            subject = subject[:72].strip() or f"Commit {commit_hash}"
            metadata = CommitCard(
                commit_hash=commit_hash,
                parent_commit_hash=parent_hash,
                island_id=default_island,
                author=author,
                subject=subject,
                change_summary="Root baseline commit.",
                evaluation_summary=None,
                tags=[],
                key_files=[],
                highlights=["Root baseline commit."],
                job_id=None,
            )
            session.add(metadata)
            self.console.log(
                "[bold green]Registered root commit[/] commit={} island={}".format(
                    commit_hash,
                    default_island,
                ),
            )
            log.info(
                "Registered root commit {} on island {}",
                commit_hash,
                default_island,
            )

    # Misc helpers ----------------------------------------------------------

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

