"""Central scheduler that coordinates evolution jobs, workers, and MAP-Elites."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from git import Repo
from git.exc import BadName, GitCommandError, InvalidGitRepositoryError, NoSuchPathError
from loguru import logger
from rich.console import Console
from sqlalchemy import select

from loreley.config import Settings, get_settings
from loreley.core.experiments import ExperimentError, get_or_create_experiment
from loreley.core.map_elites.map_elites import MapElitesManager
from loreley.core.map_elites.sampler import MapElitesSampler
from loreley.db.base import session_scope
from loreley.db.models import CommitMetadata, Metric
from loreley.scheduler.ingestion import MapElitesIngestion
from loreley.scheduler.job_scheduler import JobScheduler

console = Console()
log = logger.bind(module="scheduler.main")

__all__ = ["EvolutionScheduler", "main"]


class SchedulerError(RuntimeError):
    """Raised when the scheduler cannot continue."""


class EvolutionScheduler:
    """Orchestrate job sampling, dispatching, and MAP-Elites maintenance."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.console = console
        self.repo_root = self._resolve_repo_root()
        self._repo = self._init_repo()
        self._root_commit_hash = (self.settings.mapelites_experiment_root_commit or "").strip() or None
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
        self.job_scheduler = JobScheduler(
            settings=self.settings,
            console=self.console,
            sampler=self.sampler,
            experiment_id=self.experiment.id,
        )
        self.ingestion = MapElitesIngestion(
            settings=self.settings,
            console=self.console,
            repo_root=self.repo_root,
            repo=self._repo,
            manager=self.manager,
            experiment=self.experiment,
            repository=self.repository,
        )
        self._stop_requested = False
        self._total_scheduled_jobs = 0

        # Optionally initialise an explicit experiment root commit so that the
        # archive and database both contain a stable starting point before any
        # evolution jobs run.
        if self._root_commit_hash:
            self.ingestion.initialise_root_commit(self._root_commit_hash)

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
        stats["ingested"] = self._run_stage("ingest", self.ingestion.ingest_completed_jobs)
        stats["dispatched"] = self._run_stage("dispatch", self.job_scheduler.dispatch_pending_jobs)
        unfinished = self._run_stage("measure", self.job_scheduler.count_unfinished_jobs)
        stats["scheduled"] = self._run_stage(
            "schedule",
            lambda: self.job_scheduler.schedule_jobs(
                unfinished_jobs=unfinished,
                total_scheduled_jobs=self._total_scheduled_jobs,
            ),
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

