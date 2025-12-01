from __future__ import annotations

import dramatiq
from loguru import logger
from rich.console import Console

from loreley.config import Settings, get_settings
from loreley.core.worker.evolution import EvolutionWorker, EvolutionWorkerResult
from loreley.core.worker.job_store import EvolutionWorkerError, JobLockConflict, JobPreconditionError
from loreley.db.base import ensure_database_schema
from loreley.tasks.broker import broker  # noqa: F401 - ensure broker is initialised

console = Console()
log = logger.bind(module="tasks.workers")
settings = get_settings()

# Ensure that the core Loreley tables exist before any jobs are processed.
# This integrates the schema-initialisation flow into the core worker/scheduler
# pipeline instead of relying on example scripts.
ensure_database_schema()

__all__ = ["run_evolution_job"]

_TIME_LIMIT_MS = max(int(settings.tasks_worker_time_limit_seconds * 1000), 0)


def _run_evolution(job_id: str, *, settings: Settings) -> EvolutionWorkerResult:
    """Create an EvolutionWorker instance and run the requested job."""

    worker = EvolutionWorker(settings=settings)
    return worker.run(job_id)


def _log_job_start(job_id: str) -> None:
    console.log(f"[bold cyan]Evolution job started[/] id={job_id} queue={settings.tasks_queue_name}")
    log.info("Starting evolution job {}", job_id)


def _log_job_success(result: EvolutionWorkerResult) -> None:
    console.log(
        "[bold green]Evolution job complete[/] job={}"
        " commit={}".format(result.job_id, result.candidate_commit_hash),
    )
    log.info(
        "Evolution job {} produced commit {}",
        result.job_id,
        result.candidate_commit_hash,
    )


@dramatiq.actor(
    queue_name=settings.tasks_queue_name,
    max_retries=settings.tasks_worker_max_retries,
    time_limit=_TIME_LIMIT_MS or None,
)
def run_evolution_job(job_id: str) -> None:
    """Dramatiq actor entry point dispatching the evolution worker."""

    job_id_str = str(job_id).strip()
    if not job_id_str:
        raise ValueError("job_id must be provided.")

    _log_job_start(job_id_str)

    try:
        result = _run_evolution(job_id_str, settings=settings)
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

