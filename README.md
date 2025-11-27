## Loreley

Loreley is an automated code evolution system built around a MAP-Elites search over git commits.  
It continuously samples promising changes from a codebase, asks external agents to implement and
evaluate them, and maintains an archive of high-quality, behaviorally diverse solutions.

The project is designed for long-running, production-style deployments: configuration is
centralised and workers communicate via a Redis-backed Dramatiq queue while persisting state in
PostgreSQL.

---

## Relation to AlphaEvolve and Open-Source Replications

Loreley is conceptually related to systems such as DeepMind's **AlphaEvolve** and open-source
implementations like **OpenEvolve** ([GitHub](https://github.com/algorithmicsuperintelligence/openevolve))
and **ShinkaEvolve** ([GitHub](https://github.com/SakanaAI/ShinkaEvolve)), but makes several
deliberate design choices:

- **Commit-level individuals, whole-repo evolution**  
  Instead of treating a single file or function as the unit of evolution, Loreley uses **git
  commits** as individuals and evolves the **entire repository**. Each job operates on a real
  worktree, producing self-contained commits that can be inspected, tested, and integrated using
  standard git tooling.

- **Feature discovery from embeddings, not hand-crafted descriptors**  
  Rather than relying on manually designed behavior descriptors, Loreley derives behavioural
  features from **code and summary embeddings** (with optional dimensionality reduction such as
  PCA). This allows the MAP-Elites search space to be learned from data, maximising generality
  across languages, domains, and repository layouts.

- **Tight integration with production infra**  
  Loreley is structured as a long-running service with a scheduler, Dramatiq workers, and a
  PostgreSQL-backed archive and job store, making it suitable for continuous evolution of
  real-world codebases rather than purely benchmark-style tasks.

---

## Features

- **MAP-Elites–driven exploration**:  
  - Extracts features from code changes via preprocessing, chunking, code/summary embeddings, and
    dimensionality reduction.  
  - Maintains per-island archives of elite commits, optimising a configurable fitness metric while
    encouraging behavioural diversity.

- **End-to-end evolution pipeline**:  
  - Schedules evolution jobs from the MAP-Elites archive.  
  - Delegates planning, coding, and evaluation to external tools and plugins.  
  - Ingests evaluated commits back into the archive with full provenance.

- **Robust orchestration and workers**:  
  - A central scheduler orchestrates ingest → dispatch → measure → schedule loops.  
  - Dramatiq workers consume jobs from Redis and run evolution iterations in a controlled,
    single-threaded environment.

- **Observability and safety**:  
  - Structured logging with Loguru and Rich, with careful redaction of secrets (database URLs,
    Redis credentials, etc.).  
  - Detailed metrics and job metadata stored in PostgreSQL for inspection and analysis.

---

## High-level Architecture

- **Configuration (`app.config`)**  
  Central `Settings` class (backed by `pydantic-settings` and environment variables) controlling:
  - Application environment and logging.  
  - PostgreSQL database connection.  
  - Redis / Dramatiq task queue (URL or host/port/db/password, namespace, queue name, retry and
    time-limit policy).  
  - Scheduler behaviour (poll interval, batch sizes, capacity limits, target repo root).  
  - Worker repository layout (remote URL, branch, worktree, Git LFS, job branch TTL, etc.).  
  - External planning / coding agents, evaluator plugin, and MAP-Elites pipeline settings
    (preprocessing, chunking, embedding, dimensionality reduction, feature bounds, fitness,
    archive/grid configuration, and sampler options).

- **Database layer (`app.db`)**  
  - `base`: SQLAlchemy engine and session factory using `Settings.database_dsn`, plus helpers for
    safe DSN logging and a scoped session context manager.  
  - `models`: ORM models for commits, metrics, evolution jobs, and MAP-Elites state.

- **MAP-Elites core (`app.core.map_elites`)**  
  - Preprocessing and chunking of changed code files.  
  - Code and summary embeddings, optional dimensionality reduction (e.g. via PCA).  
  - `MapElitesManager` to maintain per-island archives in the database, ingest new commits, and
    expose query helpers.

- **Worker core (`app.core.worker`)**  
  - Git worktree management for the worker repository.  
  - Planning, coding, evaluation, and evolution orchestration logic used by the Dramatiq actor.

- **Task queue (`app.tasks`)**  
  - `broker`: configuration of the global Dramatiq Redis broker, with sanitised connection logging.  
  - `workers`: Dramatiq actors, in particular `run_evolution_job(job_id: str)`, which runs a single
    evolution job via `EvolutionWorker` with robust error handling and logging.

- **Scheduler (`app.scheduler.main`)**  
  - `EvolutionScheduler` couples the MAP-Elites archive, PostgreSQL job store, and Dramatiq queue.  
  - Periodically: ingests newly succeeded jobs, dispatches pending jobs, measures capacity, and
    schedules new jobs from MAP-Elites when allowed.

- **Scripts (`script/`)**  
  - `run_scheduler.py`: thin CLI wrapper that configures logging and delegates to
    `app.scheduler.main.main`.  
  - `run_worker.py`: CLI wrapper that configures logging, initialises the Redis broker, imports
    `app.tasks.workers`, and runs a single-threaded Dramatiq worker.

For module-level details, see the documentation under `docs/app` and `docs/script`.

---

## Requirements

- Python 3.11+ (recommended)  
- `uv` for environment and dependency management  
- PostgreSQL database  
- Redis instance for Dramatiq  
- Git installed and accessible on `PATH`  
- Access to the external planning, coding, and evaluation tools configured via environment
  variables (see `docs/app/config.md`).

---

## Installation

Clone the repository and create the environment using `uv`:

```bash
git clone <YOUR_FORK_OR_ORIGIN_URL> loreley
cd loreley

# Create a virtual environment and install dependencies from pyproject.toml / uv.lock
uv sync
```

If you prefer to use an existing environment:

```bash
uv sync --no-workspace
```

> `uv` will honour the pinned dependencies in `uv.lock` when available.

---

## Configuration

Loreley is configured entirely via environment variables, loaded into `app.config.Settings`
using `pydantic-settings`. Typical configuration is done via a `.env` file or your process
manager.

- **Core settings (examples)**:
  - `APP_NAME`, `ENVIRONMENT`, `LOG_LEVEL`  
  - `DATABASE_URL` *or* individual `DB_*` fields  
  - `TASKS_REDIS_URL` *or* (`TASKS_REDIS_HOST`, `TASKS_REDIS_PORT`, `TASKS_REDIS_DB`,
    `TASKS_REDIS_PASSWORD`), plus `TASKS_REDIS_NAMESPACE`, `TASKS_QUEUE_NAME`  
  - `SCHEDULER_REPO_ROOT`, `SCHEDULER_POLL_INTERVAL_SECONDS`,
    `SCHEDULER_MAX_UNFINISHED_JOBS`, `SCHEDULER_SCHEDULE_BATCH_SIZE`,
    `SCHEDULER_DISPATCH_BATCH_SIZE`, `SCHEDULER_INGEST_BATCH_SIZE`  
  - `WORKER_REPO_REMOTE_URL`, `WORKER_REPO_BRANCH`, `WORKER_REPO_WORKTREE`, `WORKER_REPO_ENABLE_LFS`,
    and other `WORKER_REPO_*` options  
  - `MAPELITES_*` settings to control preprocessing, embeddings, dimensionality reduction,
    feature bounds, archive resolution, fitness metric, and sampler behaviour.

Refer to `docs/app/config.md` for the full list and semantics of all configuration fields.

---

## Running the System

### Run the evolution scheduler

Run the continuous scheduler loop:

```bash
uv run python script/run_scheduler.py
```

Run a single scheduler tick (useful for cron jobs or smoke tests):

```bash
uv run python script/run_scheduler.py --once
```

Alternatively, you can invoke the module directly:

```bash
uv run python -m app.scheduler.main
uv run python -m app.scheduler.main --once
```

See `docs/script/run_scheduler.md` for more details on behaviour and logging.

### Run a worker process

Start a single-process, single-threaded Dramatiq worker:

```bash
uv run python script/run_worker.py
```

The worker will:

- Configure Loguru based on `LOG_LEVEL`.  
- Initialise the Redis broker from `app.tasks.broker`.  
- Import `app.tasks.workers` so that `run_evolution_job` is registered.  
- Start a Dramatiq `Worker` bound to the configured queue (`TASKS_QUEUE_NAME`).

See `docs/script/run_worker.md` for configuration details and operational notes.

---

## Project Layout (Overview)

- `app/` – application code
  - `config.py` – central configuration (`Settings`, `get_settings()`).  
  - `core/map_elites/` – MAP-Elites implementation and related utilities.  
  - `core/worker/` – worker-side repository, planning, coding, evaluation, and evolution logic.  
  - `db/` – SQLAlchemy engine, session helpers, and ORM models.  
  - `scheduler/` – evolution scheduler orchestration loop and CLI entrypoint.  
  - `tasks/` – Dramatiq broker setup and worker actors.
- `script/` – CLI wrapper scripts for scheduler and worker.  
- `docs/` – module-level documentation and usage guides.  
- `pyproject.toml` / `uv.lock` – dependency and environment definitions.

---

## License

This project is licensed under the terms of the license specified in `LICENSE`.
