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
from sqlalchemy import and_, case, func, or_, select
from sqlalchemy.orm import Session

from loreley.config import Settings, resolve_default_island_id
from loreley.core.contracts import clamp_text, normalize_single_line
from loreley.core.git import RepositoryError as GitRepositoryError, require_commit
from loreley.core.map_elites.map_elites import MapElitesManager
from loreley.core.repo_lock import repo_lock
from loreley.core.worker.evaluator import EvaluationContext, EvaluationError, EvaluationResult, Evaluator
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


@dataclass(slots=True, frozen=True)
class _JobPageCursor:
    completed_at: datetime
    job_id: UUID


@dataclass(slots=True, frozen=True)
class _IngestionStatePayload:
    status: str
    reason: str | None
    delta: float | None
    status_code: int | None
    message: str | None
    record: Any | None


@dataclass(slots=True, frozen=True)
class _RootRepoStateBootstrapReport:
    canonical_commit_hash: str
    eligible_files: int
    files_aggregated: int
    dimensions: int


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

            with session_scope() as batch_session:
                for snapshot in snapshots:
                    try:
                        if self._ingest_snapshot(snapshot, snapshot_session=batch_session):
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
                        self._record_ingestion_state(
                            snapshot,
                            status="failed",
                            reason=reason_display,
                            session=batch_session,
                        )
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

        commit_hash = self._resolve_root_commit_for_initialisation(commit_hash)

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

    def _resolve_root_commit_for_initialisation(self, commit_hash: str) -> str:
        try:
            return self._ensure_commit_available(commit_hash)
        except IngestionError as exc:
            self.console.log(
                f"[bold red]Failed to initialise root commit[/] commit={commit_hash} reason={exc}",
            )
            log.error("Failed to initialise root commit {}: {}", commit_hash, exc)
            raise

    # Job result ingestion --------------------------------------------------

    def _jobs_requiring_ingestion(self, *, limit: int) -> list[JobSnapshot]:
        page_size = max(limit * 4, 32)
        snapshots: list[JobSnapshot] = []
        seen_job_ids: set[UUID] = set()
        now = datetime.now(timezone.utc)
        with session_scope() as session:
            snapshots.extend(
                self._collect_jobs_requiring_ingestion(
                    session=session,
                    limit=limit,
                    page_size=page_size,
                    now=now,
                    include_exact_failed=False,
                    seen_job_ids=seen_job_ids,
                )
            )
            remaining = max(0, limit - len(snapshots))
            if remaining > 0:
                snapshots.extend(
                    self._collect_jobs_requiring_ingestion(
                        session=session,
                        limit=remaining,
                        page_size=page_size,
                        now=now,
                        include_exact_failed=True,
                        seen_job_ids=seen_job_ids,
                    )
                )
        return snapshots

    def _collect_jobs_requiring_ingestion(
        self,
        *,
        session: Session,
        limit: int,
        page_size: int,
        now: datetime,
        include_exact_failed: bool,
        seen_job_ids: set[UUID],
    ) -> list[JobSnapshot]:
        snapshots: list[JobSnapshot] = []
        cursor: _JobPageCursor | None = None
        while len(snapshots) < limit:
            rows = self._load_jobs_requiring_ingestion_page(
                session=session,
                limit=page_size,
                after=cursor,
                include_exact_failed=include_exact_failed,
            )
            if not rows:
                break
            cursor = self._page_cursor(rows[-1])
            for job in rows:
                snapshot = self._coerce_job_snapshot_for_ingestion(
                    job,
                    now=now,
                    seen_job_ids=seen_job_ids,
                )
                if snapshot is None:
                    continue
                snapshots.append(snapshot)
                if len(snapshots) >= limit:
                    break
            if len(rows) < page_size:
                break
        return snapshots

    def _load_jobs_requiring_ingestion_page(
        self,
        *,
        session: Session,
        limit: int,
        after: _JobPageCursor | None,
        include_exact_failed: bool,
    ) -> list[EvolutionJob]:
        sort_completed_at = func.coalesce(EvolutionJob.completed_at, EvolutionJob.created_at)
        stmt = (
            select(EvolutionJob)
            .where(EvolutionJob.status == JobStatus.SUCCEEDED)
            .where(EvolutionJob.result_commit_hash.is_not(None))
            .where(EvolutionJob.result_commit_hash != "")
        )
        if include_exact_failed:
            stmt = stmt.where(EvolutionJob.ingestion_status == "failed")
        else:
            stmt = stmt.where(
                or_(
                    EvolutionJob.ingestion_status.is_(None),
                    EvolutionJob.ingestion_status == "",
                    EvolutionJob.ingestion_status.not_in(("failed", "succeeded", "skipped")),
                )
            )
        if after is not None:
            stmt = stmt.where(
                or_(
                    sort_completed_at > after.completed_at,
                    and_(
                        sort_completed_at == after.completed_at,
                        EvolutionJob.id > after.job_id,
                    ),
                )
            )
        stmt = (
            stmt.order_by(
                sort_completed_at.asc(),
                EvolutionJob.id.asc(),
            )
            .limit(limit)
        )
        return list(session.execute(stmt).scalars())

    @staticmethod
    def _page_cursor(job: EvolutionJob) -> _JobPageCursor:
        completed_at = getattr(job, "completed_at", None)
        if not isinstance(completed_at, datetime):
            created_at = getattr(job, "created_at", None)
            if isinstance(created_at, datetime):
                completed_at = created_at
            else:  # pragma: no cover - succeeded jobs should always have at least one timestamp
                completed_at = datetime.min.replace(tzinfo=timezone.utc)
        return _JobPageCursor(
            completed_at=completed_at,
            job_id=job.id,
        )

    def _coerce_job_snapshot_for_ingestion(
        self,
        job: EvolutionJob,
        *,
        now: datetime,
        seen_job_ids: set[UUID],
    ) -> JobSnapshot | None:
        job_id = getattr(job, "id", None)
        if job_id is None or job_id in seen_job_ids:
            return None
        status = (job.ingestion_status or "").strip().lower()
        if status in {"succeeded", "skipped"}:
            return None
        if status == "failed" and self._should_backoff_failed_job(job, now=now):
            return None
        commit_hash = (job.result_commit_hash or "").strip()
        if not commit_hash:
            return None
        seen_job_ids.add(job_id)
        return JobSnapshot(
            job_id=job_id,
            base_commit_hash=job.base_commit_hash,
            island_id=job.island_id,
            result_commit_hash=commit_hash,
            completed_at=job.completed_at,
        )

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
        commit_hashes = self._resolve_snapshot_commit(snapshot)
        if commit_hashes is None:
            return False

        raw_commit_hash, commit_hash = commit_hashes
        metrics_payload = self._metrics_payload_for_ingestion(
            commit_hash=commit_hash,
            raw_commit_hash=raw_commit_hash,
        )
        try:
            insertion = self._ingest_with_manager(
                snapshot,
                commit_hash=commit_hash,
                metrics_payload=metrics_payload,
                snapshot_session=snapshot_session,
            )
        except Exception as exc:
            self._handle_manager_ingest_error(
                snapshot,
                commit_hash=commit_hash,
                exc=exc,
                snapshot_session=snapshot_session,
            )
            return False

        self._log_ingestion_result(snapshot, commit_hash=commit_hash, insertion=insertion)
        if snapshot_session is None:
            self._record_successful_ingestion(snapshot, insertion=insertion)
        return bool(insertion.record)

    def _resolve_snapshot_commit(self, snapshot: JobSnapshot) -> tuple[str, str] | None:
        raw_commit_hash = (snapshot.result_commit_hash or "").strip()
        if not raw_commit_hash:
            return None
        try:
            return raw_commit_hash, self._ensure_commit_available(raw_commit_hash)
        except IngestionError as exc:
            self._handle_commit_unavailable(
                snapshot,
                commit_hash=raw_commit_hash,
                exc=exc,
            )
            return None

    def _handle_commit_unavailable(
        self,
        snapshot: JobSnapshot,
        *,
        commit_hash: str,
        exc: IngestionError,
    ) -> None:
        reason_display = self._ingestion_reason_display(exc)
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

    def _metrics_payload_for_ingestion(
        self,
        *,
        commit_hash: str,
        raw_commit_hash: str,
    ) -> list[dict[str, Any]]:
        metrics_error = self._prefetched_metrics_error(
            commit_hash=commit_hash,
            raw_commit_hash=raw_commit_hash,
        )
        if metrics_error:
            raise ValueError(metrics_error)

        metrics_payload = self._prefetched_metrics_payload(
            commit_hash=commit_hash,
            raw_commit_hash=raw_commit_hash,
        )
        if metrics_payload is not None:
            return metrics_payload
        return self._load_metrics_payload_for_commit(commit_hash)

    def _prefetched_metrics_error(
        self,
        *,
        commit_hash: str,
        raw_commit_hash: str,
    ) -> str | None:
        metrics_errors_by_commit = self._prefetched_metrics_errors_by_commit
        if not metrics_errors_by_commit:
            return None
        return metrics_errors_by_commit.get(commit_hash) or metrics_errors_by_commit.get(
            raw_commit_hash
        )

    def _prefetched_metrics_payload(
        self,
        *,
        commit_hash: str,
        raw_commit_hash: str,
    ) -> list[dict[str, Any]] | None:
        metrics_payload_by_commit = self._prefetched_metrics_payload_by_commit
        if not metrics_payload_by_commit:
            return None

        metrics_payload: list[dict[str, Any]] = []
        if commit_hash in metrics_payload_by_commit:
            metrics_payload = list(metrics_payload_by_commit.get(commit_hash) or [])
        elif raw_commit_hash in metrics_payload_by_commit:
            metrics_payload = list(metrics_payload_by_commit.get(raw_commit_hash) or [])

        # If the commit hash was canonicalised (e.g. short -> full hash),
        # ensure we don't silently miss metrics due to a cache-key mismatch.
        if not metrics_payload and raw_commit_hash and raw_commit_hash != commit_hash:
            return None
        return metrics_payload

    def _ingest_with_manager(
        self,
        snapshot: JobSnapshot,
        *,
        commit_hash: str,
        metrics_payload: list[dict[str, Any]],
        snapshot_session: Session | None,
    ) -> Any:
        if snapshot_session is None:
            return self._invoke_manager_ingest(
                snapshot,
                commit_hash=commit_hash,
                metrics_payload=metrics_payload,
                snapshot_session=None,
            )

        with snapshot_session.begin_nested():
            insertion = self._invoke_manager_ingest(
                snapshot,
                commit_hash=commit_hash,
                metrics_payload=metrics_payload,
                snapshot_session=snapshot_session,
            )
            self._record_successful_ingestion(
                snapshot,
                insertion=insertion,
                session=snapshot_session,
            )
            return insertion

    def _invoke_manager_ingest(
        self,
        snapshot: JobSnapshot,
        *,
        commit_hash: str,
        metrics_payload: list[dict[str, Any]],
        snapshot_session: Session | None,
    ) -> Any:
        return self.manager.ingest(
            commit_hash=commit_hash,
            metrics=metrics_payload,
            island_id=snapshot.island_id,
            repo_root=self.repo_root,
            snapshot_session=snapshot_session,
        )

    def _handle_manager_ingest_error(
        self,
        snapshot: JobSnapshot,
        *,
        commit_hash: str,
        exc: Exception,
        snapshot_session: Session | None,
    ) -> None:
        reason_display = self._ingestion_reason_display(exc)
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
            session=snapshot_session,
        )

    def _log_ingestion_result(
        self,
        snapshot: JobSnapshot,
        *,
        commit_hash: str,
        insertion: Any,
    ) -> None:
        if insertion.record:
            self.console.log(
                f"[green]Updated archive[/] job={snapshot.job_id} commit={commit_hash} "
                f"cell={insertion.record.cell_index} Δ={insertion.delta:.4f}",
            )
        else:
            self.console.log(
                f"[yellow]Archive unchanged[/] job={snapshot.job_id} commit={commit_hash} status={insertion.status}",
            )

    def _record_successful_ingestion(
        self,
        snapshot: JobSnapshot,
        *,
        insertion: Any,
        session: Session | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "status": "succeeded" if insertion.inserted else "skipped",
            "delta": insertion.delta,
            "status_code": insertion.status,
            "message": insertion.message,
            "record": insertion.record,
        }
        if session is not None:
            kwargs["session"] = session
        self._record_ingestion_state(snapshot, **kwargs)

    def _ingestion_reason_display(self, exc: Exception) -> str:
        raw_reason = normalize_single_line(str(exc))
        reason = clamp_text(raw_reason, _INGESTION_REASON_MAX_CHARS) or None
        return reason or raw_reason or exc.__class__.__name__

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
        session: Session | None = None,
    ) -> None:
        payload = self._build_ingestion_state_payload(
            status=status,
            reason=reason,
            delta=delta,
            status_code=status_code,
            message=message,
            record=record,
        )
        if session is not None:
            self._record_ingestion_state_with_session(
                snapshot,
                payload=payload,
                session=session,
            )
            return

        with session_scope() as owned_session:
            self._apply_ingestion_state(snapshot, payload=payload, session=owned_session)

    def _build_ingestion_state_payload(
        self,
        *,
        status: str,
        reason: str | None,
        delta: float | None,
        status_code: int | None,
        message: str | None,
        record: Any | None,
    ) -> _IngestionStatePayload:
        return _IngestionStatePayload(
            status=status,
            reason=self._clamp_ingestion_text(reason, max_chars=_INGESTION_REASON_MAX_CHARS),
            delta=delta,
            status_code=status_code,
            message=self._clamp_ingestion_text(message, max_chars=_INGESTION_MESSAGE_MAX_CHARS),
            record=record,
        )

    @staticmethod
    def _ingestion_record_cell_index(record: Any | None) -> int | None:
        if record is not None and hasattr(record, "cell_index"):
            return int(getattr(record, "cell_index"))
        return None

    def _record_ingestion_state_with_session(
        self,
        snapshot: JobSnapshot,
        *,
        payload: _IngestionStatePayload,
        session: Session,
    ) -> None:
        in_nested = bool(getattr(session, "in_nested_transaction", lambda: False)())
        if in_nested or not hasattr(session, "begin_nested"):
            self._apply_ingestion_state(snapshot, payload=payload, session=session)
            return

        with session.begin_nested():
            self._apply_ingestion_state(snapshot, payload=payload, session=session)

    def _apply_ingestion_state(
        self,
        snapshot: JobSnapshot,
        *,
        payload: _IngestionStatePayload,
        session: Session,
    ) -> None:
        job = session.get(EvolutionJob, snapshot.job_id)
        if not job:
            return
        job.ingestion_attempts = int(getattr(job, "ingestion_attempts", 0) or 0) + 1
        job.ingestion_status = payload.status
        job.ingestion_last_attempt_at = datetime.now(timezone.utc)
        job.ingestion_reason = payload.reason
        job.ingestion_delta = payload.delta
        job.ingestion_status_code = payload.status_code
        job.ingestion_message = payload.message
        job.ingestion_cell_index = self._ingestion_record_cell_index(payload.record)

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
            with repo_lock(self.repo_root):
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

        if self._root_commit_has_metrics(commit_hash):
            return

        worker_repo = self._root_evaluation_worker_repository(commit_hash)
        if worker_repo is None:
            return

        result = self._evaluate_root_commit_with_worker_repo(
            commit_hash=commit_hash,
            worker_repo=worker_repo,
        )
        if result is None:
            return

        metrics_count = self._persist_root_commit_evaluation(
            commit_hash=commit_hash,
            result=result,
        )
        self._log_root_commit_evaluation_completed(
            commit_hash=commit_hash,
            metrics_count=metrics_count,
        )

    def _root_commit_has_metrics(self, commit_hash: str) -> bool:
        with session_scope() as session:
            commit_row = session.execute(
                select(CommitCard).where(CommitCard.commit_hash == commit_hash)
            ).scalar_one_or_none()
            if commit_row is None:
                return False
            existing = session.execute(
                select(Metric.id)
                .where(Metric.commit_card_id == commit_row.id)
                .limit(1)
            ).first()
            return existing is not None

    def _root_evaluation_worker_repository(self, commit_hash: str) -> WorkerRepository | None:
        try:
            return WorkerRepository(self.settings)
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
            return None

    def _evaluate_root_commit_with_worker_repo(
        self,
        *,
        commit_hash: str,
        worker_repo: WorkerRepository,
    ) -> EvaluationResult | None:
        try:
            with worker_repo.checkout_lease_for_job(
                job_id=None,
                base_commit=commit_hash,
                create_branch=False,
            ) as checkout:
                return self._evaluate_root_checkout(
                    commit_hash=commit_hash,
                    worktree=checkout.worktree,
                )
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
            return None

    def _evaluate_root_checkout(self, *, commit_hash: str, worktree: Path) -> EvaluationResult | None:
        evaluator = Evaluator(self.settings)
        context = self._root_evaluation_context(
            commit_hash=commit_hash,
            worktree=worktree,
        )
        try:
            return evaluator.evaluate(context)
        except EvaluationError as exc:
            self.console.log(
                f"[bold red]Root commit evaluation failed[/] commit={commit_hash} reason={exc}",
            )
            log.error("Root commit evaluation failed for {}: {}", commit_hash, exc)
            return None

    def _root_evaluation_context(self, *, commit_hash: str, worktree: Path) -> EvaluationContext:
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
        return EvaluationContext(
            worktree=worktree,
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

    def _persist_root_commit_evaluation(
        self,
        *,
        commit_hash: str,
        result: Any,
    ) -> int:
        metrics_payload = [metric.as_dict() for metric in result.metrics]

        with session_scope() as session:
            commit_row = session.execute(
                select(CommitCard).where(CommitCard.commit_hash == commit_hash)
            ).scalar_one_or_none()
            if commit_row is None:
                commit_row = self._build_root_commit_card(
                    commit_hash,
                    evaluation_summary=result.summary,
                )
                session.add(commit_row)
            else:
                commit_row.evaluation_summary = result.summary

            for metric in result.metrics:
                self._upsert_root_commit_metric(
                    session=session,
                    commit_row=commit_row,
                    metric=metric,
                )

        return len(metrics_payload)

    @staticmethod
    def _upsert_root_commit_metric(
        *,
        session: Session,
        commit_row: CommitCard,
        metric: Any,
    ) -> None:
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
            return

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

    def _log_root_commit_evaluation_completed(
        self,
        *,
        commit_hash: str,
        metrics_count: int,
    ) -> None:
        self.console.log(
            "[green]Evaluated root commit[/] commit={} metrics={}".format(
                commit_hash,
                metrics_count,
            ),
        )
        log.info(
            "Root commit evaluation completed for {} with {} metrics",
            commit_hash,
            metrics_count,
        )

    def _ensure_root_commit_repo_state_bootstrap(self, commit_hash: str) -> None:
        """Bootstrap the repo-state aggregate for the experiment baseline commit."""
        report = self._bootstrap_root_repo_state_aggregate(commit_hash)
        self._log_root_repo_state_bootstrap(report)

    def _bootstrap_root_repo_state_aggregate(self, commit_hash: str) -> _RootRepoStateBootstrapReport:
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
        return _RootRepoStateBootstrapReport(
            canonical_commit_hash=canonical,
            eligible_files=int(stats.eligible_files),
            files_aggregated=int(stats.files_aggregated),
            dimensions=int(embedding.dimensions),
        )

    def _log_root_repo_state_bootstrap(self, report: _RootRepoStateBootstrapReport) -> None:
        self.console.log(
            "[green]Bootstrapped repo-state baseline aggregate[/] commit={} eligible_files={} files_aggregated={} dims={}".format(
                report.canonical_commit_hash,
                report.eligible_files,
                report.files_aggregated,
                report.dimensions,
            )
        )
        log.info(
            "Bootstrapped repo-state baseline aggregate commit={} eligible_files={} files_aggregated={} dims={}",
            report.canonical_commit_hash,
            report.eligible_files,
            report.files_aggregated,
            report.dimensions,
        )

    def _ensure_root_commit_metadata(self, commit_hash: str) -> None:
        """Create or update CommitCard for the root commit."""

        metadata = self._root_commit_metadata_fields(commit_hash)

        with session_scope() as session:
            stmt = select(CommitCard).where(
                CommitCard.commit_hash == commit_hash,
            )
            existing = session.execute(stmt).scalar_one_or_none()

            if existing:
                updated = self._apply_missing_root_commit_metadata(existing, metadata)
                if updated:
                    self.console.log(
                        "[cyan]Updated root commit metadata[/] commit={} island={}".format(
                            commit_hash,
                            existing.island_id,
                        ),
                    )
                return

            commit_row = self._build_root_commit_card(commit_hash, evaluation_summary=None)
            session.add(commit_row)
            self.console.log(
                "[bold green]Registered root commit[/] commit={} island={}".format(
                    commit_hash,
                    metadata["island_id"],
                ),
            )
            log.info(
                "Registered root commit {} on island {}",
                commit_hash,
                metadata["island_id"],
            )

    def _root_commit_metadata_fields(self, commit_hash: str) -> dict[str, Any]:
        git_commit = self.repo.commit(commit_hash)
        parent_hash = git_commit.parents[0].hexsha if git_commit.parents else None
        author = getattr(getattr(git_commit, "author", None), "name", None)
        message = getattr(git_commit, "message", None)
        subject = str(message or "").splitlines()[0].strip() if message else f"Commit {commit_hash}"
        subject = subject[:72].strip() or f"Commit {commit_hash}"
        return {
            "commit_hash": commit_hash,
            "parent_commit_hash": parent_hash,
            "island_id": resolve_default_island_id(self.settings),
            "author": author,
            "subject": subject,
            "change_summary": "Root baseline commit.",
            "highlights": ["Root baseline commit."],
        }

    def _build_root_commit_card(
        self,
        commit_hash: str,
        *,
        evaluation_summary: str | None,
    ) -> CommitCard:
        fields = self._root_commit_metadata_fields(commit_hash)
        return CommitCard(
            commit_hash=str(fields["commit_hash"]),
            parent_commit_hash=fields["parent_commit_hash"],
            island_id=str(fields["island_id"]),
            author=fields["author"],
            subject=str(fields["subject"]),
            change_summary=str(fields["change_summary"]),
            evaluation_summary=evaluation_summary,
            tags=[],
            key_files=[],
            highlights=list(fields["highlights"]),
            job_id=None,
        )

    @staticmethod
    def _apply_missing_root_commit_metadata(
        existing: CommitCard,
        metadata: dict[str, Any],
    ) -> bool:
        updated = False
        if existing.island_id is None:
            existing.island_id = str(metadata["island_id"])
            updated = True
        if not getattr(existing, "highlights", None):
            existing.highlights = list(metadata["highlights"])
            updated = True
        if not getattr(existing, "subject", None):
            existing.subject = str(metadata["subject"])
            updated = True
        if not getattr(existing, "change_summary", None):
            existing.change_summary = str(metadata["change_summary"])
            updated = True
        return updated

    # Misc helpers ----------------------------------------------------------

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
