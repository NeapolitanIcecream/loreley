from __future__ import annotations

import dramatiq
from loguru import logger
from rich.console import Console

from loreley.config import Settings
from loreley.core.worker.evolution import EvolutionWorker, EvolutionWorkerResult
from loreley.core.worker.job_store import EvolutionWorkerError, JobLockConflict, JobPreconditionError
from loreley.naming import resolve_experiment_uuid, tasks_queue_name

# Ensure the broker is configured before registering any actors.
from loreley.tasks.broker import broker  # noqa: F401

console = Console()
log = logger.bind(module="tasks.workers")

__all__ = [
    "build_evolution_job_sender_actor",
    "build_evolution_job_worker_actor",
]


def _time_limit_ms(settings: Settings) -> int:
    return max(int(settings.tasks_worker_time_limit_seconds * 1000), 0)


def build_evolution_job_sender_actor(
    *,
    settings: Settings,
) -> dramatiq.Actor:
    """Build a scheduler-side actor used only for enqueueing messages.

    The callable body must not be executed in the scheduler process; it exists so
    `.send(...)` can produce correctly-formed Dramatiq messages.
    """

    queue = tasks_queue_name(getattr(settings, "experiment_id", None))
    time_limit = _time_limit_ms(settings)

    @dramatiq.actor(
        queue_name=queue,
        max_retries=settings.tasks_worker_max_retries,
        time_limit=time_limit or None,
    )
    def run_evolution_job(job_id: str) -> None:  # pragma: no cover - sender stub
        raise RuntimeError(
            "run_evolution_job sender actor must not be executed. "
            "Start a Loreley worker process to consume evolution jobs.",
        )

    return run_evolution_job


def build_evolution_job_worker_actor(
    *,
    settings: Settings,
) -> dramatiq.Actor:
    """Build an experiment-attached worker actor.

    The returned actor is bound to the experiment-scoped queue name and reuses a
    single `EvolutionWorker` instance for the full process lifetime.
    """

    exp_id = resolve_experiment_uuid(getattr(settings, "experiment_id", None))
    queue = tasks_queue_name(getattr(settings, "experiment_id", None))
    time_limit = _time_limit_ms(settings)

    evolution_worker = EvolutionWorker(settings=settings, attached_experiment_id=exp_id)

    def _log_job_start(job_id: str) -> None:
        console.log(f"[bold cyan]Evolution job started[/] id={job_id} queue={queue}")
        log.info("Starting evolution job {} (queue={})", job_id, queue)

    def _log_job_success(result: EvolutionWorkerResult) -> None:
        console.log(
            "[bold green]Evolution job complete[/] job={} commit={}".format(
                result.job_id,
                result.candidate_commit_hash,
            ),
        )
        log.info(
            "Evolution job {} produced commit {}",
            result.job_id,
            result.candidate_commit_hash,
        )

    @dramatiq.actor(
        queue_name=queue,
        max_retries=settings.tasks_worker_max_retries,
        time_limit=time_limit or None,
    )
    def run_evolution_job(job_id: str) -> None:
        """Dramatiq actor entry point dispatching the evolution worker."""

        job_id_str = str(job_id).strip()
        if not job_id_str:
            raise ValueError("job_id must be provided.")

        _log_job_start(job_id_str)

        try:
            result = evolution_worker.run(job_id_str)
        except JobLockConflict:
            console.log(
                f"[yellow]Evolution job skipped[/] id={job_id_str} reason=lock-conflict",
            )
            log.info("Job {} skipped due to lock conflict", job_id_str)
            return
        except JobPreconditionError as exc:
            console.log(
                f"[yellow]Evolution job skipped[/] id={job_id_str} reason={exc}",
            )
            log.warning("Job {} skipped: {}", job_id_str, exc)
            return
        except EvolutionWorkerError as exc:
            console.log(
                f"[bold red]Evolution job failed[/] id={job_id_str} reason={exc}",
            )
            log.error("Evolution worker failed for job {}: {}", job_id_str, exc)
            raise
        except Exception as exc:  # pragma: no cover - defensive
            console.log(
                f"[bold red]Evolution job crashed[/] id={job_id_str} reason={exc}",
            )
            log.exception("Unexpected failure for job {}", job_id_str)
            raise

        _log_job_success(result)

    return run_evolution_job

