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

# --- Application metadata ---------------------------------------------------

APP_NAME: str = "loreley-circle-packing"
APP_ENV: str = "development"
LOG_LEVEL: str = "INFO"

# --- PostgreSQL database DSN -----------------------------------------------
# Loreley requires PostgreSQL because the ORM models use Postgres-specific
# types (JSONB, ARRAY, UUID). Adjust credentials/host/db name as needed.

DATABASE_URL: str = "postgresql+psycopg://postgres:postgres@localhost:5432/loreley"

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

# If None, the scheduler will default to WORKER_REPO_WORKTREE as its repo root.
SCHEDULER_REPO_ROOT: Path | None = None

# Poll interval (seconds) between scheduler ticks in continuous mode.
SCHEDULER_POLL_INTERVAL_SECONDS: float = 30.0

# Maximum number of unfinished jobs (pending/queued/running) allowed at once.
SCHEDULER_MAX_UNFINISHED_JOBS: int = 1

# Optional global limit on total jobs scheduled by this process.
# Set to None for no global cap.
SCHEDULER_MAX_TOTAL_JOBS: int | None = 1

# --- Circle-packing evaluator configuration --------------------------------

# Additional Python search paths for evaluator plugins. For this example we
# only need the circle-packing directory.
WORKER_EVALUATOR_PYTHON_PATHS: list[str] = [str(REPO_ROOT)]

# Dotted reference to the evaluation plugin callable.
WORKER_EVALUATOR_PLUGIN: str = "evaluate:plugin"

# --- MAP-Elites tuning ------------------------------------------------------

# Use packing_density (primary objective from the circle-packing evaluator)
# as the fitness metric for MAP-Elites instead of the generic composite_score.
MAPELITES_FITNESS_METRIC: str = "packing_density"

# Give this experiment a dedicated island ID.
MAPELITES_DEFAULT_ISLAND_ID: str = "circle_packing"


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
        # Join with the OS-specific path separator; a single element will remain unchanged.
        joined_paths = os.pathsep.join(WORKER_EVALUATOR_PYTHON_PATHS)
        _set_env_if_unset("WORKER_EVALUATOR_PYTHON_PATHS", joined_paths)
    _set_env_if_unset("WORKER_EVALUATOR_PLUGIN", WORKER_EVALUATOR_PLUGIN)

    # MAP-Elites.
    _set_env_if_unset("MAPELITES_FITNESS_METRIC", MAPELITES_FITNESS_METRIC)
    _set_env_if_unset("MAPELITES_DEFAULT_ISLAND_ID", MAPELITES_DEFAULT_ISLAND_ID)

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

    # Example repo root (circle-packing directory, for evaluator plugins, etc.).
    example_root_str = str(REPO_ROOT)
    if example_root_str not in sys.path:
        sys.path.insert(0, example_root_str)


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
        "[green]MAP-Elites[/] fitness_metric={} island_id={}".format(
            os.getenv("MAPELITES_FITNESS_METRIC", "<unset>"),
            os.getenv("MAPELITES_DEFAULT_ISLAND_ID", "<unset>"),
        ),
    )


def _run_scheduler(once: bool) -> int:
    """Run the Loreley evolution scheduler."""

    _apply_base_env()
    _ensure_repo_on_sys_path()
    _print_environment_summary()

    # Import after environment is configured so that Settings and DB are initialised correctly.
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

    # Import after environment is configured so that Settings, Redis broker, and DB
    # are initialised with the values defined above.
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

    subparsers.add_parser(
        "worker",
        help="Run a single-threaded evolution worker.",
    )

    args = parser.parse_args(argv)

    if args.command == "scheduler":
        return _run_scheduler(once=bool(args.once))
    if args.command == "worker":
        return _run_worker()

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

