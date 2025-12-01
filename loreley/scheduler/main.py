"""Central scheduler that coordinates evolution jobs, workers, and MAP-Elites."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import UUID

from git import Repo
from git.exc import BadName, GitCommandError, InvalidGitRepositoryError, NoSuchPathError
from loguru import logger
from rich.console import Console
from sqlalchemy import func, select

from loreley.config import Settings, get_settings
from loreley.core.experiments import ExperimentError, get_or_create_experiment
from loreley.core.map_elites.map_elites import MapElitesManager
from loreley.core.map_elites.preprocess import ChangedFile
from loreley.core.map_elites.sampler import MapElitesSampler, ScheduledSamplerJob
from loreley.db.base import session_scope
from loreley.db.models import CommitMetadata, EvolutionJob, JobStatus, Metric
from loreley.tasks.workers import run_evolution_job

console = Console()
log = logger.bind(module="scheduler.main")

__all__ = ["EvolutionScheduler", "main"]

_UNFINISHED_STATUSES = (
    JobStatus.PENDING,
    JobStatus.QUEUED,
    JobStatus.RUNNING,
)


class SchedulerError(RuntimeError):
    """Raised when the scheduler cannot continue."""


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


class EvolutionScheduler:
    """Orchestrate job sampling, dispatching, and MAP-Elites maintenance."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.console = console
        self.repo_root = self._resolve_repo_root()
        self._repo = self._init_repo()
        try:
            self.repository, self.experiment = get_or_create_experiment(
                settings=self.settings,
                repo_root=self.repo_root,
            )
        except ExperimentError as exc:
            raise SchedulerError(str(exc)) from exc
        self.manager = MapElitesManager(
            settings=self.settings,
            repo_root=self.repo_root,
            experiment_id=self.experiment.id,
        )
        self.sampler = MapElitesSampler(manager=self.manager, settings=self.settings)
        self._stop_requested = False
        self._total_scheduled_jobs = 0

    # Public API ------------------------------------------------------------

    def run_forever(self) -> None:
        """Start the scheduler loop until interrupted."""

        interval = max(1.0, float(self.settings.scheduler_poll_interval_seconds))
        self.console.log(
            "[bold green]Scheduler online[/] repo={} experiment={} interval={}s max_unfinished={}".format(
                self.repo_root,
                getattr(self.experiment, "id", None),
                interval,
                self.settings.scheduler_max_unfinished_jobs,
            ),
        )
        self._install_signal_handlers()
        while not self._stop_requested:
            start = time.perf_counter()
            try:
                self.tick()
            except Exception as exc:  # pragma: no cover - defensive
                self.console.log(
                    f"[bold red]Scheduler tick crashed[/] reason={exc}",
                )
                log.exception("Scheduler tick crashed: {}", exc)
            elapsed = time.perf_counter() - start
            sleep_for = max(0.0, interval - elapsed)
            if self._stop_requested:
                break
            time.sleep(sleep_for)
        self.console.log("[bold yellow]Scheduler stopped[/]")

    def tick(self) -> dict[str, int]:
        """Execute a full scheduler cycle."""

        stats: dict[str, int] = {}
        stats["ingested"] = self._run_stage("ingest", self._ingest_completed_jobs)
        stats["dispatched"] = self._run_stage("dispatch", self._dispatch_pending_jobs)
        unfinished = self._run_stage("measure", self._count_unfinished_jobs)
        stats["scheduled"] = self._run_stage(
            "schedule",
            lambda: self._schedule_jobs(unfinished),
        )
        self._total_scheduled_jobs += stats["scheduled"]
        stats["unfinished"] = unfinished + stats["scheduled"]
        self.console.log(
            "[bold magenta]Scheduler tick[/] ingested={ingested} dispatched={dispatched} "
            "scheduled={scheduled} unfinished={unfinished}".format(**stats),
        )

        max_total = getattr(self.settings, "scheduler_max_total_jobs", None)
        if max_total is not None and max_total > 0:
            if self._total_scheduled_jobs >= max_total and stats["unfinished"] == 0:
                self._create_best_fitness_branch_if_possible()
                self.console.log(
                    "[bold yellow]Scheduler reached max total jobs and all jobs finished; shutting down[/] "
                    f"limit={max_total}",
                )
                log.info(
                    "Scheduler stopping after reaching max_total_jobs={} (total_scheduled={})",
                    max_total,
                    self._total_scheduled_jobs,
                )
                self.stop()
        return stats

    def stop(self) -> None:
        """Signal the scheduler loop to exit."""

        if self._stop_requested:
            return
        self._stop_requested = True

    # Internal helpers ------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self._handle_signal)
        terminate = getattr(signal, "SIGTERM", None)
        if terminate is not None:
            signal.signal(terminate, self._handle_signal)

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        self.console.log(f"[yellow]Received signal[/] signum={signum}; shutting down.")
        log.info("Scheduler received signal {}; stopping", signum)
        self.stop()

    def _run_stage(self, label: str, func: Callable[[], int]) -> int:
        try:
            return func()
        except Exception as exc:  # pragma: no cover - defensive
            self.console.log(f"[bold red]Stage {label} failed[/] reason={exc}")
            log.exception("Scheduler stage {} failed: {}", label, exc)
            return 0

    # Scheduling ------------------------------------------------------------

    def _count_unfinished_jobs(self) -> int:
        with session_scope() as session:
            stmt = (
                select(func.count(EvolutionJob.id))
                .where(EvolutionJob.status.in_(_UNFINISHED_STATUSES))
            )
            return int(session.execute(stmt).scalar_one())

    def _schedule_jobs(self, unfinished_jobs: int) -> int:
        max_jobs = max(0, int(self.settings.scheduler_max_unfinished_jobs))
        if max_jobs == 0:
            return 0
        capacity = max(0, max_jobs - unfinished_jobs)
        if capacity <= 0:
            return 0
        batch = max(1, int(self.settings.scheduler_schedule_batch_size))
        target = min(capacity, batch)

        max_total = getattr(self.settings, "scheduler_max_total_jobs", None)
        if max_total is not None and max_total > 0:
            remaining_total = max_total - self._total_scheduled_jobs
            if remaining_total <= 0:
                self.console.log(
                    "[yellow]Scheduler global job limit reached; no new jobs will be scheduled[/] "
                    f"limit={max_total}",
                )
                log.info(
                    "Global scheduler job limit reached: max_total_jobs={} (total_scheduled={})",
                    max_total,
                    self._total_scheduled_jobs,
                )
                return 0
            target = min(target, remaining_total)

        scheduled_ids: list[UUID] = []
        for _ in range(target):
            job = self._schedule_single_job()
            if not job:
                break
            scheduled_ids.append(job.job_id)
        if scheduled_ids:
            self._enqueue_jobs(scheduled_ids)
        return len(scheduled_ids)

    def _schedule_single_job(self) -> ScheduledSamplerJob | None:
        try:
            scheduled = self.sampler.schedule_job(experiment_id=self.experiment.id)
        except Exception as exc:  # pragma: no cover - defensive
            self.console.log(f"[bold red]Sampler failed[/] reason={exc}")
            log.exception("Sampler failed to create a job: {}", exc)
            return None
        if not scheduled:
            self.console.log("[yellow]Sampler returned no job[/]")
            return None
        self.console.log(
            f"[green]Scheduled job[/] id={scheduled.job_id} island={scheduled.island_id} "
            f"base={scheduled.base_commit_hash}",
        )
        return scheduled

    def _dispatch_pending_jobs(self) -> int:
        batch = max(0, int(self.settings.scheduler_dispatch_batch_size))
        if batch == 0:
            return 0
        pending = self._fetch_pending_job_ids(limit=batch)
        if not pending:
            return 0
        ready = self._mark_jobs_queued(pending)
        dispatched = 0
        for job_id in ready:
            try:
                run_evolution_job.send(str(job_id))
                dispatched += 1
            except Exception as exc:  # pragma: no cover - defensive
                self.console.log(
                    f"[bold red]Failed to enqueue job[/] id={job_id} reason={exc}",
                )
                log.exception("Failed to enqueue job {}: {}", job_id, exc)
        if dispatched:
            self.console.log(f"[cyan]Dispatched {dispatched} job(s) to Dramatiq[/]")
        return dispatched

    def _fetch_pending_job_ids(self, *, limit: int) -> list[UUID]:
        with session_scope() as session:
            stmt = (
                select(EvolutionJob.id)
                .where(EvolutionJob.status == JobStatus.PENDING)
                .order_by(
                    EvolutionJob.priority.desc(),
                    EvolutionJob.scheduled_at.asc(),
                    EvolutionJob.created_at.asc(),
                )
                .limit(limit)
            )
            return list(session.execute(stmt).scalars())

    def _mark_jobs_queued(self, job_ids: Sequence[UUID]) -> list[UUID]:
        ready: list[UUID] = []
        if not job_ids:
            return ready
        now = datetime.now(timezone.utc)
        with session_scope() as session:
            stmt = (
                select(EvolutionJob)
                .where(EvolutionJob.id.in_(job_ids))
                .with_for_update()
            )
            for job in session.execute(stmt).scalars():
                if job.status != JobStatus.PENDING:
                    continue
                job.status = JobStatus.QUEUED
                job.scheduled_at = job.scheduled_at or now
                ready.append(job.id)
        return ready

    def _enqueue_jobs(self, job_ids: Sequence[UUID]) -> None:
        if not job_ids:
            return
        queued = self._mark_jobs_queued(job_ids)
        for job_id in queued:
            try:
                run_evolution_job.send(str(job_id))
                self.console.log(
                    f"[bold green]Queued job[/] id={job_id}",
                )
            except Exception as exc:  # pragma: no cover - defensive
                self.console.log(
                    f"[bold red]Failed to enqueue job[/] id={job_id} reason={exc}",
                )
                log.exception("Failed to enqueue scheduled job {}: {}", job_id, exc)

    # Ingestion -------------------------------------------------------------

    def _ingest_completed_jobs(self) -> int:
        batch = max(0, int(self.settings.scheduler_ingest_batch_size))
        if batch == 0:
            return 0
        snapshots = self._jobs_requiring_ingestion(limit=batch)
        ingested = 0
        for snapshot in snapshots:
            if self._ingest_snapshot(snapshot):
                ingested += 1
        return ingested

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
        except SchedulerError as exc:
            self._record_ingestion_state(
                snapshot,
                status="failed",
                reason=str(exc),
            )
            return False
        changed_files = self._collect_changed_files(commit_hash)
        if not changed_files:
            self._record_ingestion_state(
                snapshot,
                status="skipped",
                reason="No changed files detected for commit.",
            )
            return False
        metadata = self._build_ingestion_metadata(snapshot, result)
        metrics = result.get("metrics") or []
        try:
            insertion = self.manager.ingest(
                commit_hash=commit_hash,
                changed_files=changed_files,
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

    def _resolve_repo_root(self) -> Path:
        candidate = self.settings.scheduler_repo_root or self.settings.worker_repo_worktree
        if candidate:
            return Path(candidate).expanduser().resolve()
        return Path.cwd()

    def _init_repo(self) -> Repo:
        try:
            return Repo(self.repo_root)
        except (NoSuchPathError, InvalidGitRepositoryError) as exc:  # pragma: no cover - filesystem
            raise SchedulerError(f"Scheduler repo {self.repo_root} is not a git repository.") from exc

    def _ensure_commit_available(self, commit_hash: str) -> None:
        try:
            self._repo.commit(commit_hash)
            return
        except BadName:
            pass
        self.console.log(f"[yellow]Fetching missing commit[/] {commit_hash}")
        try:
            self._repo.git.fetch("--all", "--tags")
            self._repo.commit(commit_hash)
        except GitCommandError as exc:
            raise SchedulerError(f"Cannot fetch commit {commit_hash}: {exc}") from exc
        except BadName as exc:
            raise SchedulerError(f"Commit {commit_hash} not found after fetch.") from exc

    def _collect_changed_files(self, commit_hash: str) -> list[ChangedFile]:
        commit = self._repo.commit(commit_hash)
        stats = commit.stats
        files = stats.files or {}
        changed: list[ChangedFile] = []
        for path, info in files.items():
            change_count = int(info.get("lines") or (info.get("insertions", 0) + info.get("deletions", 0)) or 1)
            changed.append(ChangedFile(path=Path(path), change_count=change_count))
        return changed

    # Best-branch helpers ----------------------------------------------------

    def _create_best_fitness_branch_if_possible(self) -> None:
        """Create a git branch for the best-fitness commit when evolution ends.

        The branch is created only when a best commit can be resolved for the
        current experiment. Failures are logged but do not prevent shutdown.
        """

        try:
            best_commit, meta = self._resolve_best_fitness_commit()
        except Exception as exc:  # pragma: no cover - defensive
            self.console.log(
                f"[bold red]Failed to resolve best-fitness commit[/] reason={exc}",
            )
            log.exception("Failed to resolve best-fitness commit: {}", exc)
            return

        if not best_commit:
            self.console.log(
                "[yellow]No best-fitness commit found for experiment; skipping branch creation[/]",
            )
            return

        root_commit = meta.get("root_commit_hash")
        metric_name = meta.get("fitness_metric") or self.settings.mapelites_fitness_metric
        fitness_value = meta.get("fitness_value")
        island_id = meta.get("island_id")

        try:
            branch_name = self._create_best_fitness_branch(
                best_commit_hash=best_commit,
                root_commit_hash=root_commit,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.console.log(
                f"[bold red]Failed to create best-fitness branch[/] commit={best_commit} reason={exc}",
            )
            log.exception("Failed to create best-fitness branch for commit {}: {}", best_commit, exc)
            return

        # Log a concise, human-friendly summary with the key attributes.
        fitness_str: str
        try:
            fitness_str = f"{float(fitness_value):.6f}" if fitness_value is not None else "n/a"
        except (TypeError, ValueError):
            fitness_str = str(fitness_value)

        self.console.log(
            "[bold green]Best-fitness branch created[/] "
            "branch={} commit={} root_commit={} island={} metric={} fitness={}".format(
                branch_name,
                best_commit,
                root_commit or "n/a",
                island_id or "n/a",
                metric_name or "n/a",
                fitness_str,
            ),
        )
        log.info(
            "Best-fitness branch created "
            "(branch={} commit={} root_commit={} island_id={} metric={} fitness={})",
            branch_name,
            best_commit,
            root_commit,
            island_id,
            metric_name,
            fitness_value,
        )

    def _resolve_best_fitness_commit(self) -> tuple[str | None, dict[str, Any]]:
        """Return the commit hash with the best fitness for the current experiment.

        The search is scoped to the scheduler's experiment and uses the
        configured MAP-Elites fitness metric and direction.
        """

        metric_name = self.settings.mapelites_fitness_metric
        if not metric_name:
            return None, {}

        is_higher_better = bool(self.settings.mapelites_fitness_higher_is_better)

        with session_scope() as session:
            order_column = Metric.value.desc() if is_higher_better else Metric.value.asc()

            stmt = (
                select(
                    CommitMetadata.commit_hash,
                    CommitMetadata.island_id,
                    Metric.value,
                )
                .join(Metric, Metric.commit_hash == CommitMetadata.commit_hash)
                .where(
                    CommitMetadata.experiment_id == self.experiment.id,
                    Metric.name == metric_name,
                )
                .order_by(order_column)
                .limit(1)
            )

            row = session.execute(stmt).first()
            if not row:
                return None, {}

            best_commit_hash: str = row[0]
            island_id: str | None = row[1]
            fitness_value: float | None = float(row[2]) if row[2] is not None else None

            root_commit_hash = self._find_root_commit_for_experiment_chain(
                session=session,
                start_commit_hash=best_commit_hash,
            )

        meta: dict[str, Any] = {
            "island_id": island_id,
            "fitness_metric": metric_name,
            "fitness_value": fitness_value,
            "root_commit_hash": root_commit_hash,
        }
        return best_commit_hash, meta

    def _find_root_commit_for_experiment_chain(
        self,
        *,
        session,
        start_commit_hash: str,
    ) -> str | None:
        """Walk the CommitMetadata parent chain to find the experiment root commit.

        The root commit is defined as the earliest known parent in the evolution
        chain for this experiment. This may be a commit that does not itself have
        a CommitMetadata row (for example, the original repository commit used as
        the starting point for the first evolution job).
        """

        current = start_commit_hash
        root: str | None = None
        visited: set[str] = set()

        while current and current not in visited:
            visited.add(current)
            parent_hash = session.execute(
                select(CommitMetadata.parent_commit_hash).where(
                    CommitMetadata.commit_hash == current,
                    CommitMetadata.experiment_id == self.experiment.id,
                )
            ).scalar_one_or_none()
            if not parent_hash:
                break
            root = parent_hash

            # Continue walking only while the parent itself belongs to this experiment.
            exists = session.execute(
                select(CommitMetadata.commit_hash).where(
                    CommitMetadata.commit_hash == parent_hash,
                    CommitMetadata.experiment_id == self.experiment.id,
                )
            ).scalar_one_or_none()
            if not exists:
                break

            current = parent_hash

        return root

    def _create_best_fitness_branch(
        self,
        *,
        best_commit_hash: str,
        root_commit_hash: str | None,
    ) -> str:
        """Create a git branch pointing at the best-fitness commit.

        The branch name is derived from the experiment identifier and is chosen
        to avoid clashing with existing local branches.
        """

        # Ensure the target commit is present locally; this may trigger a fetch.
        self._ensure_commit_available(best_commit_hash)

        try:
            existing_names = {head.name for head in getattr(self._repo, "heads", [])}
        except Exception:  # pragma: no cover - defensive
            existing_names = set()

        prefix = "evolution/best"
        experiment_suffix = str(self.experiment.id).split("-")[0]
        base_name = f"{prefix}/{experiment_suffix}"
        branch_name = base_name
        counter = 1

        while branch_name in existing_names:
            counter += 1
            branch_name = f"{base_name}-{counter}"

        self._repo.create_head(branch_name, best_commit_hash)

        log.info(
            "Created git branch {} for best-fitness commit {} (root_commit={})",
            branch_name,
            best_commit_hash,
            root_commit_hash,
        )

        return branch_name

    # Misc helpers ----------------------------------------------------------

    @staticmethod
    def _coerce_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
        if not payload:
            return {}
        return dict(payload)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Run the Loreley evolution scheduler.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Execute a single scheduling tick and exit.",
    )
    args = parser.parse_args(argv)

    scheduler = EvolutionScheduler()
    if args.once:
        scheduler.tick()
        return 0
    scheduler.run_forever()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

