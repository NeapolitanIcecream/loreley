from __future__ import annotations

"""
Convenience launcher for running Loreley scheduler/worker on the circle-packing example.

This script:
  - Hard-codes a minimal set of environment variables needed for DB, Redis, worker repo,
    evaluator plugin, and MAP-Elites fitness metric.
  - Exposes two subcommands:
      * scheduler  – run the evolution scheduler loop (or a single tick with --once).
      * worker     – run a single-threaded Dramatiq worker.

Usage (from the Loreley repository root, ideally via uv):

    uv run python examples/evol_circle_packing.py scheduler
    uv run python examples/evol_circle_packing.py scheduler --once
    uv run python examples/evol_circle_packing.py worker

Edit the configuration block below to match your local PostgreSQL, Redis,
and git remote setup. OPENAI_API_KEY is always read from the environment.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console


# ============================================================================
# User configuration (edit this block to match your environment)
# ============================================================================

REPO_ROOT: Path = Path(__file__).resolve().parents[1] / "examples" / "circle-packing"
# Evaluation environment lives alongside the candidate repository so that
# evaluation logic is isolated from the evolved code.
EVAL_ENV_ROOT: Path = REPO_ROOT.parent / "circle_packing_env"

# --- Application metadata ---------------------------------------------------

APP_NAME: str = "loreley-circle-packing"
APP_ENV: str = "development"
LOG_LEVEL: str = "INFO"

# --- PostgreSQL database DSN -----------------------------------------------
# Loreley requires PostgreSQL because the ORM models use Postgres-specific
# types (JSONB, ARRAY, UUID). Adjust credentials/host/db name as needed.

DATABASE_URL: str = "postgresql+psycopg://loreley:loreley@localhost:5432/circle_packing"

# --- Redis / Dramatiq broker -----------------------------------------------

# Single Redis URL is usually the simplest way to configure the broker.
TASKS_REDIS_URL: str = "redis://localhost:6379/0"
TASKS_REDIS_NAMESPACE: str = "loreley"

# Queue name used by the Dramatiq actor in loreley.tasks.workers.
TASKS_QUEUE_NAME: str = "loreley.evolution.circle_packing"

# --- Worker repository configuration ----------------------------------------
# The worker will clone this git remote into WORKER_REPO_WORKTREE and push
# evolution branches back to it. Set this to the upstream repository you want
# to evolve (typically the same repo this script lives in, or a fork).

# Example for an SSH remote:
#   WORKER_REPO_REMOTE_URL = "git@github.com:YOUR_USER/YOUR_REPO.git"
# Example for a local path remote:
#   WORKER_REPO_REMOTE_URL = str(REPO_ROOT)
WORKER_REPO_REMOTE_URL: str = str(REPO_ROOT)

# Git branch to track on the remote when syncing the worker clone.
WORKER_REPO_BRANCH: str = "main"

# Local worktree used exclusively by the worker process.
WORKER_REPO_WORKTREE: Path = REPO_ROOT / ".cache" / "loreley" / "worker-repo"

# --- Scheduler configuration ------------------------------------------------

# Use the circle-packing example repository itself as the scheduler repo root
# so that it always sees a valid git worktree, independent of the worker clone.
SCHEDULER_REPO_ROOT: Path | None = REPO_ROOT

# Poll interval (seconds) between scheduler ticks in continuous mode.
SCHEDULER_POLL_INTERVAL_SECONDS: float = 30.0

# Maximum number of unfinished jobs (pending/queued/running) allowed at once.
SCHEDULER_MAX_UNFINISHED_JOBS: int = 1

# Optional global limit on total jobs scheduled by this process.
# Set to None for no global cap.
SCHEDULER_MAX_TOTAL_JOBS: int | None = 1

# --- Circle-packing evaluator configuration --------------------------------

# Additional Python search paths for evaluator plugins. For this example we
# point at the dedicated evaluation environment directory so that the worker
# can import ``evaluate:plugin`` independently of the candidate repo.
WORKER_EVALUATOR_PYTHON_PATHS: list[str] = [str(EVAL_ENV_ROOT)]

# Dotted reference to the evaluation plugin callable.
WORKER_EVALUATOR_PLUGIN: str = "evaluate:plugin"

# --- MAP-Elites tuning ------------------------------------------------------

# Use packing_density (primary objective from the circle-packing evaluator)
# as the fitness metric for MAP-Elites instead of the generic composite_score.
MAPELITES_FITNESS_METRIC: str = "packing_density"

# Give this experiment a dedicated island ID.
MAPELITES_DEFAULT_ISLAND_ID: str = "circle_packing"
MAPELITES_EXPERIMENT_ROOT_COMMIT: str | None = "64884c2c"


# --- Model / LLM configuration (see loreley.config.Settings) ----------------

# Evolution commit message model.
WORKER_EVOLUTION_COMMIT_MODEL: str = "openai/gpt-5.1"
WORKER_EVOLUTION_COMMIT_TEMPERATURE: float = 0.2
WORKER_EVOLUTION_COMMIT_MAX_OUTPUT_TOKENS: int = 128
WORKER_EVOLUTION_COMMIT_MAX_RETRIES: int = 3
WORKER_EVOLUTION_COMMIT_RETRY_BACKOFF_SECONDS: float = 2.0

# Global evolution objective shared across planning and coding prompts.
WORKER_EVOLUTION_GLOBAL_GOAL: str = (
    "Evolve the circle-packing solution so that pack_circles() returns a valid, "
    "non-overlapping set of circles inside the unit square with as high "
    "packing_density as possible, while keeping the code simple, deterministic, "
    "and fast enough for the evaluator."
)

# Code embedding model used for MAP-Elites preprocessing.
MAPELITES_CODE_EMBEDDING_MODEL: str = "text-embedding-3-large"
MAPELITES_CODE_EMBEDDING_DIMENSIONS: int | None = None
MAPELITES_CODE_EMBEDDING_BATCH_SIZE: int = 12
MAPELITES_CODE_EMBEDDING_MAX_CHUNKS_PER_COMMIT: int = 512
MAPELITES_CODE_EMBEDDING_MAX_RETRIES: int = 3
MAPELITES_CODE_EMBEDDING_RETRY_BACKOFF_SECONDS: float = 2.0

# Natural-language summary model used for MAP-Elites.
MAPELITES_SUMMARY_MODEL: str = "openai/gpt-5.1"
MAPELITES_SUMMARY_TEMPERATURE: float = 0.2
MAPELITES_SUMMARY_MAX_OUTPUT_TOKENS: int = 512
MAPELITES_SUMMARY_SOURCE_CHAR_LIMIT: int = 6000
MAPELITES_SUMMARY_MAX_RETRIES: int = 3
MAPELITES_SUMMARY_RETRY_BACKOFF_SECONDS: float = 2.0
MAPELITES_SUMMARY_EMBEDDING_MODEL: str = "text-embedding-3-large"
MAPELITES_SUMMARY_EMBEDDING_DIMENSIONS: int | None = None
MAPELITES_SUMMARY_EMBEDDING_BATCH_SIZE: int = 16


# --- Optional OpenAI-compatible API base URL -------------------------------
# OPENAI_API_KEY is intentionally NOT hard-coded; it is always read from the
# environment at runtime. If you need a custom base URL (e.g. Azure or a
# compatible proxy), set OPENAI_BASE_URL here.

OPENAI_BASE_URL: str | None = None


# ============================================================================
# Internal helpers
# ============================================================================

console = Console()
log = logger.bind(module="examples.evol_circle_packing")


def _set_env_if_unset(name: str, value: Any | None) -> None:
    """Set an environment variable only when it is not already defined."""

    if value is None:
        return
    if name in os.environ and os.environ[name]:
        return
    os.environ[name] = str(value)


def _apply_base_env() -> None:
    """Populate os.environ with the configuration defined above."""

    # Basic app metadata and logging.
    _set_env_if_unset("APP_NAME", APP_NAME)
    _set_env_if_unset("APP_ENV", APP_ENV)
    _set_env_if_unset("LOG_LEVEL", LOG_LEVEL)

    # Database (PostgreSQL).
    _set_env_if_unset("DATABASE_URL", DATABASE_URL)

    # Redis / Dramatiq broker.
    _set_env_if_unset("TASKS_REDIS_URL", TASKS_REDIS_URL)
    _set_env_if_unset("TASKS_REDIS_NAMESPACE", TASKS_REDIS_NAMESPACE)
    _set_env_if_unset("TASKS_QUEUE_NAME", TASKS_QUEUE_NAME)

    # Worker repository.
    _set_env_if_unset("WORKER_REPO_REMOTE_URL", WORKER_REPO_REMOTE_URL)
    _set_env_if_unset("WORKER_REPO_BRANCH", WORKER_REPO_BRANCH)
    _set_env_if_unset("WORKER_REPO_WORKTREE", WORKER_REPO_WORKTREE)

    # Scheduler.
    if SCHEDULER_REPO_ROOT is not None:
        _set_env_if_unset("SCHEDULER_REPO_ROOT", SCHEDULER_REPO_ROOT)
    _set_env_if_unset(
        "SCHEDULER_POLL_INTERVAL_SECONDS",
        SCHEDULER_POLL_INTERVAL_SECONDS,
    )
    _set_env_if_unset(
        "SCHEDULER_MAX_UNFINISHED_JOBS",
        SCHEDULER_MAX_UNFINISHED_JOBS,
    )
    if SCHEDULER_MAX_TOTAL_JOBS is not None:
        _set_env_if_unset(
            "SCHEDULER_MAX_TOTAL_JOBS",
            SCHEDULER_MAX_TOTAL_JOBS,
        )

    # Evaluator for circle-packing.
    if WORKER_EVALUATOR_PYTHON_PATHS:
        # Encode as JSON so that pydantic's list[str] env parsing can consume it directly.
        paths_payload = json.dumps(WORKER_EVALUATOR_PYTHON_PATHS)
        _set_env_if_unset("WORKER_EVALUATOR_PYTHON_PATHS", paths_payload)
    _set_env_if_unset("WORKER_EVALUATOR_PLUGIN", WORKER_EVALUATOR_PLUGIN)

    # MAP-Elites.
    _set_env_if_unset("MAPELITES_FITNESS_METRIC", MAPELITES_FITNESS_METRIC)
    _set_env_if_unset("MAPELITES_DEFAULT_ISLAND_ID", MAPELITES_DEFAULT_ISLAND_ID)
    _set_env_if_unset(
        "MAPELITES_EXPERIMENT_ROOT_COMMIT",
        MAPELITES_EXPERIMENT_ROOT_COMMIT,
    )

    # Model / LLM configuration.
    _set_env_if_unset("WORKER_EVOLUTION_COMMIT_MODEL", WORKER_EVOLUTION_COMMIT_MODEL)
    _set_env_if_unset(
        "WORKER_EVOLUTION_COMMIT_TEMPERATURE",
        WORKER_EVOLUTION_COMMIT_TEMPERATURE,
    )
    _set_env_if_unset(
        "WORKER_EVOLUTION_COMMIT_MAX_OUTPUT_TOKENS",
        WORKER_EVOLUTION_COMMIT_MAX_OUTPUT_TOKENS,
    )
    _set_env_if_unset(
        "WORKER_EVOLUTION_COMMIT_MAX_RETRIES",
        WORKER_EVOLUTION_COMMIT_MAX_RETRIES,
    )
    _set_env_if_unset(
        "WORKER_EVOLUTION_COMMIT_RETRY_BACKOFF_SECONDS",
        WORKER_EVOLUTION_COMMIT_RETRY_BACKOFF_SECONDS,
    )
    _set_env_if_unset(
        "WORKER_EVOLUTION_GLOBAL_GOAL",
        WORKER_EVOLUTION_GLOBAL_GOAL,
    )

    _set_env_if_unset("MAPELITES_CODE_EMBEDDING_MODEL", MAPELITES_CODE_EMBEDDING_MODEL)
    if MAPELITES_CODE_EMBEDDING_DIMENSIONS is not None:
        _set_env_if_unset(
            "MAPELITES_CODE_EMBEDDING_DIMENSIONS",
            MAPELITES_CODE_EMBEDDING_DIMENSIONS,
        )
    _set_env_if_unset(
        "MAPELITES_CODE_EMBEDDING_BATCH_SIZE",
        MAPELITES_CODE_EMBEDDING_BATCH_SIZE,
    )
    _set_env_if_unset(
        "MAPELITES_CODE_EMBEDDING_MAX_CHUNKS_PER_COMMIT",
        MAPELITES_CODE_EMBEDDING_MAX_CHUNKS_PER_COMMIT,
    )
    _set_env_if_unset(
        "MAPELITES_CODE_EMBEDDING_MAX_RETRIES",
        MAPELITES_CODE_EMBEDDING_MAX_RETRIES,
    )
    _set_env_if_unset(
        "MAPELITES_CODE_EMBEDDING_RETRY_BACKOFF_SECONDS",
        MAPELITES_CODE_EMBEDDING_RETRY_BACKOFF_SECONDS,
    )

    _set_env_if_unset("MAPELITES_SUMMARY_MODEL", MAPELITES_SUMMARY_MODEL)
    _set_env_if_unset(
        "MAPELITES_SUMMARY_TEMPERATURE",
        MAPELITES_SUMMARY_TEMPERATURE,
    )
    _set_env_if_unset(
        "MAPELITES_SUMMARY_MAX_OUTPUT_TOKENS",
        MAPELITES_SUMMARY_MAX_OUTPUT_TOKENS,
    )
    _set_env_if_unset(
        "MAPELITES_SUMMARY_SOURCE_CHAR_LIMIT",
        MAPELITES_SUMMARY_SOURCE_CHAR_LIMIT,
    )
    _set_env_if_unset(
        "MAPELITES_SUMMARY_MAX_RETRIES",
        MAPELITES_SUMMARY_MAX_RETRIES,
    )
    _set_env_if_unset(
        "MAPELITES_SUMMARY_RETRY_BACKOFF_SECONDS",
        MAPELITES_SUMMARY_RETRY_BACKOFF_SECONDS,
    )
    _set_env_if_unset(
        "MAPELITES_SUMMARY_EMBEDDING_MODEL",
        MAPELITES_SUMMARY_EMBEDDING_MODEL,
    )
    if MAPELITES_SUMMARY_EMBEDDING_DIMENSIONS is not None:
        _set_env_if_unset(
            "MAPELITES_SUMMARY_EMBEDDING_DIMENSIONS",
            MAPELITES_SUMMARY_EMBEDDING_DIMENSIONS,
        )
    _set_env_if_unset(
        "MAPELITES_SUMMARY_EMBEDDING_BATCH_SIZE",
        MAPELITES_SUMMARY_EMBEDDING_BATCH_SIZE,
    )

    # OpenAI-compatible model endpoint (API key remains external).
    if OPENAI_BASE_URL is not None:
        _set_env_if_unset("OPENAI_BASE_URL", OPENAI_BASE_URL)


def _ensure_repo_on_sys_path() -> None:
    """Ensure the Loreley project root and example repo are importable."""

    # Project root (contains the ``loreley`` package and ``script`` entrypoints).
    project_root = Path(__file__).resolve().parents[1]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    # Example repo root (circle-packing directory, used as the worker repo).
    example_root_str = str(REPO_ROOT)
    if example_root_str not in sys.path:
        sys.path.insert(0, example_root_str)

    # Evaluation environment root (contains the ``evaluate`` plugin).
    eval_env_root_str = str(EVAL_ENV_ROOT)
    if eval_env_root_str not in sys.path:
        sys.path.insert(0, eval_env_root_str)


def _print_environment_summary() -> None:
    """Print a short summary of the effective runtime configuration."""

    console.log(
        "[bold cyan]Circle-packing evolution launcher[/] "
        f"repo_root={REPO_ROOT} worker_worktree={WORKER_REPO_WORKTREE}",
    )
    console.log(
        "[green]DB[/] DATABASE_URL={}".format(os.getenv("DATABASE_URL", "<unset>")),
    )
    console.log(
        "[green]Redis[/] TASKS_REDIS_URL={} namespace={}".format(
            os.getenv("TASKS_REDIS_URL", "<unset>"),
            os.getenv("TASKS_REDIS_NAMESPACE", "<unset>"),
        ),
    )
    console.log(
        "[green]Worker repo[/] remote={} branch={} worktree={}".format(
            os.getenv("WORKER_REPO_REMOTE_URL", "<unset>"),
            os.getenv("WORKER_REPO_BRANCH", "<unset>"),
            os.getenv("WORKER_REPO_WORKTREE", "<unset>"),
        ),
    )
    console.log(
        "[green]Evaluator[/] paths={} plugin={}".format(
            os.getenv("WORKER_EVALUATOR_PYTHON_PATHS", "<unset>"),
            os.getenv("WORKER_EVALUATOR_PLUGIN", "<unset>"),
        ),
    )
    console.log(
        "[green]MAP-Elites[/] fitness_metric={} island_id={} root_commit={}".format(
            os.getenv("MAPELITES_FITNESS_METRIC", "<unset>"),
            os.getenv("MAPELITES_DEFAULT_ISLAND_ID", "<unset>"),
            os.getenv("MAPELITES_EXPERIMENT_ROOT_COMMIT", "<unset>"),
        ),
    )


def _reset_database() -> None:
    """Initialise the Loreley database by clearing all existing records.

    This will TRUNCATE all ORM-managed tables on the configured ``DATABASE_URL``
    while preserving the schema itself.
    """

    _apply_base_env()
    _ensure_repo_on_sys_path()

    console.log("[bold yellow]Resetting Loreley database (TRUNCATE ALL TABLES)…[/]")

    try:
        # Import after environment is configured so that the engine is initialised
        # with the correct DATABASE_URL.
        from sqlalchemy import text

        from loreley.db.base import Base, engine, ensure_database_schema

        # Ensure schema exists so that all metadata tables are present.
        ensure_database_schema()

        # Collect all ORM table names and truncate them in one CASCADE statement.
        table_names = [table.name for table in Base.metadata.sorted_tables]
        if not table_names:
            console.log(
                "[bold yellow]No ORM tables found to truncate – schema appears empty.[/]",
            )
            return

        truncate_sql = "TRUNCATE TABLE {} RESTART IDENTITY CASCADE;".format(
            ", ".join(f'"{name}"' for name in table_names),
        )

        with engine.begin() as connection:
            connection.execute(text(truncate_sql))

        console.log(
            "[bold green]Database reset complete[/] truncated_tables={}".format(
                ", ".join(table_names),
            ),
        )
        log.info("Database reset complete for tables: {}", ", ".join(table_names))
    except Exception as exc:  # pragma: no cover - defensive
        console.log(
            "[bold red]Database reset failed[/] reason={}".format(exc),
        )
        log.exception("Database reset failed: {}", exc)
        raise


def _run_scheduler(once: bool, init_db: bool) -> int:
    """Run the Loreley evolution scheduler."""

    _apply_base_env()
    _ensure_repo_on_sys_path()
    if init_db:
        _reset_database()
    _print_environment_summary()
    # Import after environment is configured so that Settings and DB are
    # initialised correctly. The core worker/scheduler pipeline is responsible
    # for ensuring the database schema exists.
    from script.run_scheduler import main as scheduler_main

    argv: list[str] = []
    if once:
        argv.append("--once")

    console.log(
        "[bold green]Starting scheduler[/] once={} …".format("yes" if once else "no"),
    )
    return int(scheduler_main(argv))


def _run_worker() -> int:
    """Run a single-threaded Loreley evolution worker."""

    _apply_base_env()
    _ensure_repo_on_sys_path()
    _print_environment_summary()
    # Import after environment is configured so that Settings, Redis broker, and
    # DB are initialised with the values defined above. The core worker module
    # takes care of schema initialisation.
    from script.run_worker import main as worker_main

    console.log("[bold green]Starting worker[/] …")
    return int(worker_main(None))


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for this helper script."""

    parser = argparse.ArgumentParser(
        description="Run Loreley scheduler/worker configured for the circle-packing example.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scheduler_parser = subparsers.add_parser(
        "scheduler",
        help="Run the evolution scheduler loop (use --once for a single tick).",
    )
    scheduler_parser.add_argument(
        "--once",
        action="store_true",
        help="Execute a single scheduling tick and exit.",
    )
    scheduler_parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialise the DATABASE_URL by clearing all existing Loreley tables "
        "before running the scheduler.",
    )

    subparsers.add_parser(
        "worker",
        help="Run a single-threaded evolution worker.",
    )

    args = parser.parse_args(argv)

    if args.command == "scheduler":
        return _run_scheduler(once=bool(args.once), init_db=bool(args.init_db))
    if args.command == "worker":
        return _run_worker()

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

