from __future__ import annotations

"""Result ingestion and MAP-Elites maintenance for the evolution scheduler.

The public API here is intentionally small so that ``loreley.scheduler.main``
can delegate all ingestion responsibilities to this module.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
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
from loreley.db.models import CommitMetadata, EvolutionJob, JobStatus, MapElitesState, Metric

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
    payload: dict[str, Any]
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
                .where(EvolutionJob.status == JobStatus.SUCCEEDED)
                .order_by(EvolutionJob.completed_at.asc())
                .limit(batch_limit)
            )
            rows = list(session.execute(stmt).scalars())
            for job in rows:
                payload = self._coerce_payload(job.payload)
                state = self._current_ingestion_state(payload)
                status = state.get("status")
                if status in {"succeeded", "skipped"}:
                    continue
                result = self._extract_result_block(payload)
                commit_hash = (result.get("commit_hash") or "").strip()
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
                        payload=payload,
                        completed_at=job.completed_at,
                    )
                )
                if len(snapshots) >= limit:
                    break
        return snapshots

    def _ingest_snapshot(self, snapshot: JobSnapshot) -> bool:
        result = self._extract_result_block(snapshot.payload)
        commit_hash = (result.get("commit_hash") or "").strip()
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
        metadata = self._build_ingestion_metadata(snapshot, result)
        metrics = result.get("metrics") or []
        try:
            insertion = self.manager.ingest(
                commit_hash=commit_hash,
                metrics=metrics,
                island_id=snapshot.island_id,
                repo_root=self.repo_root,
                treeish=commit_hash,
                metadata=metadata,
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
        state_payload = {
            "status": "succeeded" if insertion.inserted else "skipped",
            "delta": insertion.delta,
            "status_code": insertion.status,
            "message": insertion.message,
            "record": self._record_to_payload(insertion.record),
        }
        self._record_ingestion_state(snapshot, **state_payload)
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
        record: Mapping[str, Any] | None = None,
    ) -> None:
        payload = snapshot.payload
        state = self._current_ingestion_state(payload)
        state["attempts"] = int(state.get("attempts", 0)) + 1
        state["status"] = status
        state["last_attempt_at"] = self._now_iso()
        state["reason"] = reason
        state["delta"] = delta
        state["status_code"] = status_code
        state["message"] = message
        if record is not None:
            state["record"] = dict(record)
        payload.setdefault("ingestion", {})["map_elites"] = {k: v for k, v in state.items() if v is not None}
        self._persist_payload(snapshot.job_id, payload)

    def _persist_payload(self, job_id: UUID, payload: Mapping[str, Any]) -> None:
        with session_scope() as session:
            job = session.get(EvolutionJob, job_id)
            if not job:
                return
            job.payload = dict(payload)

    def _build_ingestion_metadata(
        self,
        snapshot: JobSnapshot,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        context = {
            "goal": snapshot.payload.get("goal"),
            "constraints": snapshot.payload.get("constraints"),
            "acceptance_criteria": snapshot.payload.get("acceptance_criteria"),
            "tags": snapshot.payload.get("tags"),
            "notes": snapshot.payload.get("notes"),
        }
        return {
            "job": {
                "id": str(snapshot.job_id),
                "base_commit": snapshot.base_commit_hash,
                "island_id": snapshot.island_id,
                "experiment_id": str(snapshot.experiment_id) if snapshot.experiment_id else None,
                "repository_id": str(snapshot.repository_id) if snapshot.repository_id else None,
                "completed_at": snapshot.completed_at.isoformat() if snapshot.completed_at else None,
            },
            "context": context,
            "result": {
                "plan_summary": result.get("plan_summary"),
                "evaluation_summary": result.get("evaluation_summary"),
                "tests_executed": result.get("tests_executed"),
                "tests_recommended": result.get("tests_recommended"),
            },
        }

    def _record_to_payload(self, record: Any) -> dict[str, Any] | None:
        if record is None:
            return None
        return {
            "commit_hash": record.commit_hash,
            "cell_index": record.cell_index,
            "fitness": record.fitness,
            "island_id": record.island_id,
            "timestamp": record.timestamp,
        }

    def _current_ingestion_state(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        ingestion = payload.get("ingestion") or {}
        state = ingestion.get("map_elites") or {}
        return dict(state)

    def _extract_result_block(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        result = payload.get("result")
        if isinstance(result, Mapping):
            return dict(result)
        return {}

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
            existing = session.execute(
                select(Metric.id).where(Metric.commit_hash == commit_hash)
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
            checkout = worker_repo.checkout_for_job(
                job_id=None,
                base_commit=commit_hash,
                create_branch=False,
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
            return

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
        )

        try:
            result = evaluator.evaluate(context)
        except EvaluationError as exc:
            self.console.log(
                f"[bold red]Root commit evaluation failed[/] commit={commit_hash} reason={exc}",
            )
            log.error("Root commit evaluation failed for {}: {}", commit_hash, exc)
            return

        metrics_payload = [metric.as_dict() for metric in result.metrics]

        with session_scope() as session:
            commit_row = session.execute(
                select(CommitMetadata).where(CommitMetadata.commit_hash == commit_hash)
            ).scalar_one_or_none()
            if commit_row is not None:
                commit_row.evaluation_summary = result.summary
                extra = dict(commit_row.extra_context or {})
                root_eval = dict(extra.get("root_evaluation") or {})
                root_eval.update(
                    {
                        "summary": result.summary,
                        "metrics": metrics_payload,
                    }
                )
                extra["root_evaluation"] = root_eval
                commit_row.extra_context = extra

            for metric in result.metrics:
                existing_metric = session.execute(
                    select(Metric).where(
                        Metric.commit_hash == commit_hash,
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
                            commit_hash=commit_hash,
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
        """Create or update CommitMetadata for the root commit."""

        git_commit = self.repo.commit(commit_hash)
        parent_hash = git_commit.parents[0].hexsha if git_commit.parents else None
        author = getattr(getattr(git_commit, "author", None), "name", None)
        message = getattr(git_commit, "message", None)

        with session_scope() as session:
            stmt = select(CommitMetadata).where(CommitMetadata.commit_hash == commit_hash)
            existing = session.execute(stmt).scalar_one_or_none()
            default_island = self.settings.mapelites_default_island_id or "main"

            if existing:
                updated = False
                if existing.experiment_id is None and self.experiment is not None:
                    existing.experiment_id = self.experiment.id
                    updated = True
                if existing.island_id is None:
                    existing.island_id = default_island
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

            extra_context: dict[str, Any] = {
                "root_commit": True,
                "experiment": {
                    "id": str(self.experiment.id),
                    "repository_id": str(self.repository.id) if self.repository is not None else None,
                    "name": getattr(self.experiment, "name", None),
                    "config_hash": getattr(self.experiment, "config_hash", None),
                    "repository_slug": getattr(self.repository, "slug", None) if self.repository is not None else None,
                },
            }

            metadata = CommitMetadata(
                commit_hash=commit_hash,
                parent_commit_hash=parent_hash,
                island_id=default_island,
                experiment_id=self.experiment.id,
                author=author,
                message=message,
                evaluation_summary=None,
                tags=[],
                extra_context=extra_context,
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
    def _coerce_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
        if not payload:
            return {}
        return dict(payload)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


