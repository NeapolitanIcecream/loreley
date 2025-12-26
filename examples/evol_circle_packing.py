from __future__ import annotations

"""
Convenience launcher for running Loreley scheduler/worker on the circle-packing example.

This script:
  - Hard-codes a minimal set of environment variables needed for DB, Redis, worker repo,
    evaluator plugin, and MAP-Elites fitness metric.
  - Exposes four subcommands:
      * scheduler  – run the evolution scheduler loop (or a single tick with --once).
      * worker     – run a single-threaded Dramatiq worker.
      * api        – run the read-only UI API (FastAPI via uvicorn).
      * ui         – run the Streamlit UI.

Usage (from the Loreley repository root, ideally via uv):

    uv run python examples/evol_circle_packing.py scheduler
    uv run python examples/evol_circle_packing.py scheduler --once
    uv run python examples/evol_circle_packing.py worker
    uv run python examples/evol_circle_packing.py api --reload
    uv run python examples/evol_circle_packing.py ui

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

DB_SCHEME: str = "postgresql+psycopg"
DB_HOST: str = "localhost"
DB_PORT: int = 5432
DB_USERNAME: str = "loreley"
DB_PASSWORD: str = "loreley"
DB_NAME: str = "circle_packing"
DB_POOL_SIZE: int = 10
DB_MAX_OVERFLOW: int = 20
DB_POOL_TIMEOUT: int = 30
DB_ECHO: bool = False

DATABASE_URL: str = (
    f"{DB_SCHEME}://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

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

# Local worktree used exclusively by the worker process. Randomisation is now
# handled inside loreley.config via WORKER_REPO_WORKTREE_RANDOMIZE.
WORKER_REPO_WORKTREE: Path = REPO_ROOT / ".cache" / "loreley" / "worker-repo"
WORKER_REPO_WORKTREE_RANDOMIZE: bool = True

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
SCHEDULER_MAX_TOTAL_JOBS: int | None = 2

# --- UI API / Streamlit UI configuration ------------------------------------

UI_API_HOST: str = "127.0.0.1"
UI_API_PORT: int = 8000
UI_API_BASE_URL: str = f"http://{UI_API_HOST}:{UI_API_PORT}"

UI_HOST: str = "127.0.0.1"
UI_PORT: int = 8501

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
MAPELITES_FITNESS_METRIC: str = "sum_radii"

# Give this experiment a dedicated island ID.
MAPELITES_DEFAULT_ISLAND_ID: str = "circle_packing"
MAPELITES_EXPERIMENT_ROOT_COMMIT: str | None = "6dab191"

# Size of the cold-start seed population created from the root commit. The
# scheduler still respects SCHEDULER_MAX_UNFINISHED_JOBS and
# SCHEDULER_MAX_TOTAL_JOBS when creating these jobs.
MAPELITES_SEED_POPULATION_SIZE: int = 1

# Behaviour space sized for <= ~200 total jobs.
MAPELITES_DIMENSION_REDUCTION_TARGET_DIMS: int = 2
MAPELITES_DIMENSION_REDUCTION_MIN_FIT_SAMPLES: int = 8
MAPELITES_FEATURE_TRUNCATION_K: float = 3.0
MAPELITES_FEATURE_NORMALIZATION_WARMUP_SAMPLES: int = (
    MAPELITES_DIMENSION_REDUCTION_MIN_FIT_SAMPLES
)
MAPELITES_ARCHIVE_CELLS_PER_DIM: int = 12
MAPELITES_FEATURE_CLIP: bool = True

# --- Agent backends ---------------------------------------------------------

# WORKER_PLANNING_BACKEND: str | None = None
# WORKER_CODING_BACKEND: str | None = None
WORKER_PLANNING_BACKEND: str = "loreley.core.worker.agent_backend:CursorCliBackend"
WORKER_CODING_BACKEND: str = "loreley.core.worker.agent_backend:CursorCliBackend"

# --- Model / LLM configuration (see loreley.config.Settings) ----------------

# Evolution commit message model.
WORKER_EVOLUTION_COMMIT_MODEL: str = "openai/gpt-5.2"
WORKER_EVOLUTION_COMMIT_TEMPERATURE: float = 0.2
WORKER_EVOLUTION_COMMIT_MAX_OUTPUT_TOKENS: int = 128
WORKER_EVOLUTION_COMMIT_MAX_RETRIES: int = 3
WORKER_EVOLUTION_COMMIT_RETRY_BACKOFF_SECONDS: float = 2.0

# Global evolution objective shared across planning and coding prompts.
WORKER_EVOLUTION_GLOBAL_GOAL: str = (
    "Evolve the circle-packing solution so that pack_circles(n=26) returns a valid, "
    "non-overlapping set of 26 circles inside the unit square with as high sum of "
    "radii as possible (equivalently, large radii at fixed circle count), while "
    "keeping the code simple, deterministic, and fast enough for the evaluator."
)

# Code embedding model used for MAP-Elites preprocessing.
MAPELITES_CODE_EMBEDDING_MODEL: str = "text-embedding-3-large"
MAPELITES_CODE_EMBEDDING_DIMENSIONS: int | None = None
MAPELITES_CODE_EMBEDDING_BATCH_SIZE: int = 12
MAPELITES_CODE_EMBEDDING_MAX_CHUNKS_PER_COMMIT: int = 512
MAPELITES_CODE_EMBEDDING_MAX_RETRIES: int = 3
MAPELITES_CODE_EMBEDDING_RETRY_BACKOFF_SECONDS: float = 2.0

# Natural-language summary model used for MAP-Elites.
MAPELITES_SUMMARY_MODEL: str = "openai/gpt-5.2"
MAPELITES_SUMMARY_TEMPERATURE: float = 0.2
MAPELITES_SUMMARY_MAX_OUTPUT_TOKENS: int = 512
MAPELITES_SUMMARY_SOURCE_CHAR_LIMIT: int = 6000
MAPELITES_SUMMARY_MAX_RETRIES: int = 3
MAPELITES_SUMMARY_RETRY_BACKOFF_SECONDS: float = 2.0
MAPELITES_SUMMARY_EMBEDDING_MODEL: str = "text-embedding-3-large"
MAPELITES_SUMMARY_EMBEDDING_DIMENSIONS: int | None = None
MAPELITES_SUMMARY_EMBEDDING_BATCH_SIZE: int = 16


# --- Optional OpenAI-compatible API configuration -------------------------
# OPENAI_API_KEY is intentionally NOT hard-coded; it is always read from the
# environment at runtime. If you need a custom base URL (e.g. Azure or a
# compatible proxy), set OPENAI_BASE_URL here. OPENAI_API_SPEC selects the API
# surface (responses, chat_completions, etc.) exposed to the Loreley runtime.

OPENAI_BASE_URL: str | None = None
OPENAI_API_SPEC: str = "chat_completions"


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


def _apply_base_env(*, include_worker_repo: bool = False) -> None:
    """Populate os.environ with the configuration defined above."""

    # Basic app metadata and logging.
    _set_env_if_unset("APP_NAME", APP_NAME)
    _set_env_if_unset("APP_ENV", APP_ENV)
    _set_env_if_unset("LOG_LEVEL", LOG_LEVEL)

    # Database (PostgreSQL).
    _set_env_if_unset("DATABASE_URL", DATABASE_URL)
    _set_env_if_unset("DB_SCHEME", DB_SCHEME)
    _set_env_if_unset("DB_HOST", DB_HOST)
    _set_env_if_unset("DB_PORT", DB_PORT)
    _set_env_if_unset("DB_USER", DB_USERNAME)
    _set_env_if_unset("DB_PASSWORD", DB_PASSWORD)
    _set_env_if_unset("DB_NAME", DB_NAME)
    _set_env_if_unset("DB_POOL_SIZE", DB_POOL_SIZE)
    _set_env_if_unset("DB_MAX_OVERFLOW", DB_MAX_OVERFLOW)
    _set_env_if_unset("DB_POOL_TIMEOUT", DB_POOL_TIMEOUT)
    _set_env_if_unset("DB_ECHO", DB_ECHO)

    # Redis / Dramatiq broker.
    _set_env_if_unset("TASKS_REDIS_URL", TASKS_REDIS_URL)
    _set_env_if_unset("TASKS_REDIS_NAMESPACE", TASKS_REDIS_NAMESPACE)
    _set_env_if_unset("TASKS_QUEUE_NAME", TASKS_QUEUE_NAME)

    # Worker repository.
    _set_env_if_unset("WORKER_REPO_REMOTE_URL", WORKER_REPO_REMOTE_URL)
    _set_env_if_unset("WORKER_REPO_BRANCH", WORKER_REPO_BRANCH)
    if include_worker_repo:
        _set_env_if_unset("WORKER_REPO_WORKTREE", WORKER_REPO_WORKTREE)
        _set_env_if_unset(
            "WORKER_REPO_WORKTREE_RANDOMIZE",
            WORKER_REPO_WORKTREE_RANDOMIZE,
        )

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
    _set_env_if_unset(
        "MAPELITES_SEED_POPULATION_SIZE",
        MAPELITES_SEED_POPULATION_SIZE,
    )
    _set_env_if_unset(
        "MAPELITES_DIMENSION_REDUCTION_TARGET_DIMS",
        MAPELITES_DIMENSION_REDUCTION_TARGET_DIMS,
    )
    _set_env_if_unset(
        "MAPELITES_DIMENSION_REDUCTION_MIN_FIT_SAMPLES",
        MAPELITES_DIMENSION_REDUCTION_MIN_FIT_SAMPLES,
    )
    _set_env_if_unset(
        "MAPELITES_ARCHIVE_CELLS_PER_DIM",
        MAPELITES_ARCHIVE_CELLS_PER_DIM,
    )
    _set_env_if_unset(
        "MAPELITES_FEATURE_TRUNCATION_K",
        MAPELITES_FEATURE_TRUNCATION_K,
    )
    _set_env_if_unset(
        "MAPELITES_FEATURE_NORMALIZATION_WARMUP_SAMPLES",
        MAPELITES_FEATURE_NORMALIZATION_WARMUP_SAMPLES,
    )
    _set_env_if_unset("MAPELITES_FEATURE_CLIP", MAPELITES_FEATURE_CLIP)

    # Agent backends
    _set_env_if_unset(
        "WORKER_PLANNING_BACKEND",
        WORKER_PLANNING_BACKEND,
    )
    _set_env_if_unset(
        "WORKER_CODING_BACKEND",
        WORKER_CODING_BACKEND,
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
    _set_env_if_unset("OPENAI_API_SPEC", OPENAI_API_SPEC)
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

    worker_worktree = os.getenv("WORKER_REPO_WORKTREE", "<unset>")

    console.log(
        "[bold cyan]Circle-packing evolution launcher[/] "
        f"repo_root={REPO_ROOT} worker_worktree={worker_worktree}",
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
        "[green]MAP-Elites[/] fitness_metric={} island_id={} root_commit={} seed_population={}".format(
            os.getenv("MAPELITES_FITNESS_METRIC", "<unset>"),
            os.getenv("MAPELITES_DEFAULT_ISLAND_ID", "<unset>"),
            os.getenv("MAPELITES_EXPERIMENT_ROOT_COMMIT", "<unset>"),
            os.getenv("MAPELITES_SEED_POPULATION_SIZE", "<unset>"),
        ),
    )


def _reset_database() -> None:
    """Initialise the Loreley database schema from the current ORM models.

    Loreley intentionally does not ship migrations. The most reliable way to
    align the DB schema with the current code is to drop all ORM-managed tables
    and recreate them. This helper also clears all Dramatiq queues in the
    configured Redis namespace.
    """

    _apply_base_env()
    _ensure_repo_on_sys_path()

    console.log("[bold yellow]Resetting Loreley database schema (DROP + CREATE)…[/]")

    try:
        # Import after environment is configured so that the engine is initialised
        # with the correct DATABASE_URL.
        from loreley.db.base import Base, reset_database_schema
        from loreley.tasks.broker import build_redis_broker

        reset_database_schema(include_console_log=False)

        console.log(
            "[bold green]Database schema reset complete[/] tables={}".format(
                ", ".join(sorted(Base.metadata.tables.keys())),
            ),
        )
        log.info(
            "Database schema reset complete for tables: {}",
            ", ".join(sorted(Base.metadata.tables.keys())),
        )

        # Clear all Dramatiq message queues in the configured Redis namespace so
        # that no jobs from a previous run survive a fresh database initialisation.
        redis_broker = build_redis_broker()
        redis_broker.flush_all()
        console.log(
            "[bold green]Redis broker reset complete[/] namespace={}".format(
                os.getenv("TASKS_REDIS_NAMESPACE", "<unset>"),
            ),
        )
        log.info(
            "Redis broker reset complete for namespace {}",
            os.getenv("TASKS_REDIS_NAMESPACE", "<unset>"),
        )
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

    _apply_base_env(include_worker_repo=True)
    _ensure_repo_on_sys_path()
    _print_environment_summary()
    # Import after environment is configured so that Settings, Redis broker, and
    # DB are initialised with the values defined above. The core worker module
    # takes care of schema initialisation.
    from script.run_worker import main as worker_main

    console.log("[bold green]Starting worker[/] …")
    return int(worker_main([]))


def _run_api(*, host: str, port: int, log_level: str | None, reload: bool) -> int:
    """Run the Loreley read-only UI API (FastAPI via uvicorn)."""

    _apply_base_env()
    _ensure_repo_on_sys_path()
    _print_environment_summary()

    # Import after environment is configured so that Settings is initialised with
    # the values defined above.
    from script.run_api import main as api_main

    argv: list[str] = ["--host", str(host), "--port", str(int(port))]
    if log_level:
        argv += ["--log-level", str(log_level)]
    if reload:
        argv.append("--reload")

    console.log(
        "[bold green]Starting UI API[/] host={} port={} reload={} …".format(
            host,
            port,
            "yes" if reload else "no",
        )
    )
    return int(api_main(argv))


def _run_ui(*, host: str, port: int, api_base_url: str, headless: bool) -> int:
    """Run the Loreley Streamlit UI."""

    _apply_base_env()
    _ensure_repo_on_sys_path()
    _print_environment_summary()

    # Streamlit is invoked as a subprocess by script.run_ui and expects the
    # "loreley/ui/app.py" path to be resolved from the project root.
    project_root = Path(__file__).resolve().parents[1]
    previous_cwd = Path.cwd()
    os.chdir(project_root)
    try:
        from script.run_ui import main as ui_main

        argv: list[str] = [
            "--api-base-url",
            str(api_base_url),
            "--host",
            str(host),
            "--port",
            str(int(port)),
        ]
        if headless:
            argv.append("--headless")

        console.log(
            "[bold green]Starting UI[/] host={} port={} headless={} api_base_url={} …".format(
                host,
                port,
                "yes" if headless else "no",
                api_base_url,
            )
        )
        return int(ui_main(argv))
    finally:
        os.chdir(previous_cwd)


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
        help="Initialise the DATABASE_URL by dropping and recreating all ORM-managed "
        "tables and clearing all Dramatiq Redis task queues before running the scheduler.",
    )

    subparsers.add_parser(
        "worker",
        help="Run a single-threaded evolution worker.",
    )

    api_parser = subparsers.add_parser(
        "api",
        help="Run the read-only UI API (FastAPI via uvicorn).",
    )
    api_parser.add_argument("--host", default=UI_API_HOST, help="Bind host.")
    api_parser.add_argument("--port", type=int, default=UI_API_PORT, help="Bind port.")
    api_parser.add_argument(
        "--log-level",
        dest="log_level",
        help="Override Settings.log_level.",
    )
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only).")

    ui_parser = subparsers.add_parser(
        "ui",
        help="Run the Loreley Streamlit UI.",
    )
    ui_parser.add_argument(
        "--api-base-url",
        default=UI_API_BASE_URL,
        help="Base URL of the Loreley UI API.",
    )
    ui_parser.add_argument("--host", default=UI_HOST, help="Streamlit bind host.")
    ui_parser.add_argument("--port", type=int, default=UI_PORT, help="Streamlit bind port.")
    ui_parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening a browser.",
    )

    args = parser.parse_args(argv)

    if args.command == "scheduler":
        return _run_scheduler(once=bool(args.once), init_db=bool(args.init_db))
    if args.command == "worker":
        return _run_worker()
    if args.command == "api":
        return _run_api(
            host=str(args.host),
            port=int(args.port),
            log_level=(str(args.log_level) if args.log_level else None),
            reload=bool(args.reload),
        )
    if args.command == "ui":
        return _run_ui(
            host=str(args.host),
            port=int(args.port),
            api_base_url=str(args.api_base_url),
            headless=bool(args.headless),
        )

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        console.log("[yellow]Keyboard interrupt received[/]; exiting...")
        raise SystemExit(130)


