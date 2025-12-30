from __future__ import annotations

"""Result ingestion and MAP-Elites maintenance for the evolution scheduler.

The public API here is intentionally small so that ``loreley.scheduler.main``
can delegate all ingestion responsibilities to this module.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import UUID

from git import Repo
from git.exc import BadName, GitCommandError
from loguru import logger
from rich.console import Console
from sqlalchemy import select

from loreley.config import Settings
from loreley.core.map_elites.map_elites import MapElitesManager
from loreley.core.worker.evaluator import EvaluationContext, EvaluationError, Evaluator
from loreley.core.worker.repository import RepositoryError, WorkerRepository
from loreley.db.base import session_scope
from loreley.db.models import CommitCard, EvolutionJob, JobStatus, MapElitesState, Metric

log = logger.bind(module="scheduler.ingestion")


class IngestionError(RuntimeError):
    """Raised when result ingestion cannot proceed for a commit."""


@dataclass(slots=True, frozen=True)
class JobSnapshot:
    """Immutable view of a job that completed and awaits ingestion."""

    job_id: UUID
    base_commit_hash: str | None
    island_id: str | None
    experiment_id: UUID | None
    repository_id: UUID | None
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
    experiment: Any
    repository: Any | None

    # Public API ------------------------------------------------------------

    def ingest_completed_jobs(self) -> int:
        """Ingest a batch of newly succeeded jobs into MAP-Elites."""

        batch = max(0, int(self.settings.scheduler_ingest_batch_size))
        if batch == 0:
            return 0
        snapshots = self._jobs_requiring_ingestion(limit=batch)
        ingested = 0
        for snapshot in snapshots:
            if self._ingest_snapshot(snapshot):
                ingested += 1
        return ingested

    def initialise_root_commit(self, commit_hash: str) -> None:
        """Ensure the configured root commit is present in DB and evaluated.

        This helper is idempotent and safe to call on every scheduler startup.
        Failures are logged but do not prevent the scheduler from running.
        """

        try:
            self._ensure_commit_available(commit_hash)
        except IngestionError as exc:
            self.console.log(
                f"[bold red]Failed to initialise root commit[/] commit={commit_hash} reason={exc}",
            )
            log.error("Failed to initialise root commit {}: {}", commit_hash, exc)
            return

        try:
            self._ensure_root_commit_metadata(commit_hash)
            self._ensure_root_commit_evaluated(commit_hash)
        except Exception as exc:  # pragma: no cover - defensive
            self.console.log(
                f"[bold red]Root commit initialisation failed[/] commit={commit_hash} reason={exc}",
            )
            log.exception("Root commit initialisation failed for {}: {}", commit_hash, exc)

    # Job result ingestion --------------------------------------------------

    def _jobs_requiring_ingestion(self, *, limit: int) -> list[JobSnapshot]:
        batch_limit = max(limit * 4, 32)
        snapshots: list[JobSnapshot] = []
        with session_scope() as session:
            stmt = (
                select(EvolutionJob)
                .where(
                    EvolutionJob.status == JobStatus.SUCCEEDED,
                    EvolutionJob.experiment_id == getattr(self.experiment, "id", None),
                )
                .order_by(EvolutionJob.completed_at.asc())
                .limit(batch_limit)
            )
            rows = list(session.execute(stmt).scalars())
            for job in rows:
                if (job.ingestion_status or "").strip() in {"succeeded", "skipped"}:
                    continue
                commit_hash = (job.result_commit_hash or "").strip()
                if not commit_hash:
                    continue

                experiment_id = getattr(job, "experiment_id", None)
                repository_id = None
                experiment = getattr(job, "experiment", None)
                if experiment is not None:
                    repository_id = getattr(experiment, "repository_id", None)

                snapshots.append(
                    JobSnapshot(
                        job_id=job.id,
                        base_commit_hash=job.base_commit_hash,
                        island_id=job.island_id,
                        experiment_id=experiment_id,
                        repository_id=repository_id,
                        result_commit_hash=commit_hash,
                        completed_at=job.completed_at,
                    )
                )
                if len(snapshots) >= limit:
                    break
        return snapshots

    def _ingest_snapshot(self, snapshot: JobSnapshot) -> bool:
        commit_hash = (snapshot.result_commit_hash or "").strip()
        if not commit_hash:
            return False
        try:
            self._ensure_commit_available(commit_hash)
        except IngestionError as exc:
            self._record_ingestion_state(
                snapshot,
                status="failed",
                reason=str(exc),
            )
            return False
        metrics_payload: list[dict[str, Any]] = []
        with session_scope() as session:
            commit_row = None
            if snapshot.experiment_id is not None:
                commit_row = session.execute(
                    select(CommitCard).where(
                        CommitCard.experiment_id == snapshot.experiment_id,
                        CommitCard.commit_hash == commit_hash,
                    )
                ).scalar_one_or_none()
            if commit_row is not None:
                rows = session.scalars(
                    select(Metric).where(Metric.commit_card_id == commit_row.id)
                ).all()
                for row in rows:
                    metrics_payload.append(
                        {
                            "name": row.name,
                            "value": float(row.value),
                            "unit": row.unit,
                            "higher_is_better": bool(row.higher_is_better),
                        }
                    )
        try:
            insertion = self.manager.ingest(
                commit_hash=commit_hash,
                metrics=metrics_payload,
                island_id=snapshot.island_id,
                repo_root=self.repo_root,
                treeish=commit_hash,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._record_ingestion_state(
                snapshot,
                status="failed",
                reason=str(exc),
            )
            self.console.log(
                f"[bold red]MAP-Elites ingest failed[/] job={snapshot.job_id} reason={exc}",
            )
            log.exception("Failed to ingest commit {} for job {}: {}", commit_hash, snapshot.job_id, exc)
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
        with session_scope() as session:
            job = session.get(EvolutionJob, snapshot.job_id)
            if not job:
                return
            job.ingestion_attempts = int(getattr(job, "ingestion_attempts", 0) or 0) + 1
            job.ingestion_status = status
            job.ingestion_last_attempt_at = datetime.now(timezone.utc)
            job.ingestion_reason = reason
            job.ingestion_delta = delta
            job.ingestion_status_code = status_code
            job.ingestion_message = message
            if record is not None and hasattr(record, "cell_index"):
                job.ingestion_cell_index = int(getattr(record, "cell_index"))
            else:
                job.ingestion_cell_index = None

    # Git helpers -----------------------------------------------------------

    def _ensure_commit_available(self, commit_hash: str) -> None:
        try:
            self.repo.commit(commit_hash)
            return
        except BadName:
            pass
        self.console.log(f"[yellow]Fetching missing commit[/] {commit_hash}")
        try:
            self.repo.git.fetch("--all", "--tags")
            self.repo.commit(commit_hash)
        except GitCommandError as exc:
            raise IngestionError(f"Cannot fetch commit {commit_hash}: {exc}") from exc
        except BadName as exc:
            raise IngestionError(f"Commit {commit_hash} not found after fetch.") from exc

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
                select(CommitCard).where(
                    CommitCard.experiment_id == getattr(self.experiment, "id", None),
                    CommitCard.commit_hash == commit_hash,
                )
            ).scalar_one_or_none()
            if commit_row is not None:
                existing = session.execute(
                    select(Metric.id).where(Metric.commit_card_id == commit_row.id)
                ).scalar_one_or_none()
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
                default_island = self.settings.mapelites_default_island_id or "main"
                payload: dict[str, Any] = {
                    "job": {
                        "id": None,
                        "island_id": default_island,
                        "experiment_id": str(self.experiment.id),
                        "repository_id": str(self.repository.id) if self.repository is not None else None,
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
                        "experiment_id": str(self.experiment.id),
                        "repository_id": str(self.repository.id) if self.repository is not None else None,
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
                select(CommitCard).where(
                    CommitCard.experiment_id == getattr(self.experiment, "id", None),
                    CommitCard.commit_hash == commit_hash,
                )
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
                default_island = self.settings.mapelites_default_island_id or "main"

                commit_row = CommitCard(
                    commit_hash=commit_hash,
                    parent_commit_hash=parent_hash,
                    island_id=default_island,
                    experiment_id=getattr(self.experiment, "id", None),
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

    def _ensure_root_commit_metadata(self, commit_hash: str) -> None:
        """Create or update CommitCard for the root commit."""

        git_commit = self.repo.commit(commit_hash)
        parent_hash = git_commit.parents[0].hexsha if git_commit.parents else None
        author = getattr(getattr(git_commit, "author", None), "name", None)
        message = getattr(git_commit, "message", None)

        with session_scope() as session:
            stmt = select(CommitCard).where(
                CommitCard.commit_hash == commit_hash,
                CommitCard.experiment_id == self.experiment.id,
            )
            existing = session.execute(stmt).scalar_one_or_none()
            default_island = self.settings.mapelites_default_island_id or "main"

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
                        "[cyan]Updated root commit metadata[/] commit={} experiment={} island={}".format(
                            commit_hash,
                            existing.experiment_id,
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
                experiment_id=self.experiment.id,
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
                "[bold green]Registered root commit[/] commit={} experiment={} island={}".format(
                    commit_hash,
                    self.experiment.id,
                    default_island,
                ),
            )
            log.info(
                "Registered root commit {} for experiment {} on island {}",
                commit_hash,
                self.experiment.id,
                default_island,
            )

    # Misc helpers ----------------------------------------------------------

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


