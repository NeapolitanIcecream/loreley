## Loreley

Loreley is an automated MAP-Elites system that evolves entire git repositories. It continuously samples promising commits, asks external agents to implement and evaluate them, and stores the best-performing variants for later reuse.

---

## Why Loreley?

- **Whole-repo evolution** – each individual is a real git commit, so outputs stay debuggable and compatible with normal tooling.
- **Learned behaviour space** – behaviour descriptors come from embeddings (optionally reduced with PCA) instead of hand-picked heuristics.
- **Production-grade loop** – a Dramatiq/Redis worker fleet, PostgreSQL archive, and central scheduler keep the system running indefinitely.

Related systems: [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) style pipelines and open-source efforts such as [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) and [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve).

---

## Architecture at a Glance

- **Configuration (`loreley.config`)** – a single `Settings` object (pydantic-settings) that centralises environment-driven configuration for logging, database, Redis/Dramatiq, scheduler, worker repositories, and MAP-Elites knobs.
- **Database (`loreley.db`)** – SQLAlchemy engine/session helpers plus ORM models for repositories, experiments, commits, metrics, evolution jobs, and archive state.
- **MAP-Elites core (`loreley.core.map_elites`)** – preprocessing, chunking, code and summary embeddings, dimensionality reduction, archive management, sampling, and snapshot persistence via `MapElitesManager`.
- **Worker pipeline (`loreley.core.worker`)** – worktree lifecycle, planning, coding, evaluation, evolution commits, commit summaries, and job persistence used by Dramatiq actors.
- **Tasks (`loreley.tasks`)** – Redis broker helpers and the `run_evolution_job(job_id)` Dramatiq actor that drives the worker.
- **Scheduler (`loreley.scheduler`)** – `EvolutionScheduler` that ingests completed jobs into MAP-Elites, dispatches pending jobs, maintains a seed population (when a root commit is configured), and can create a best-fitness branch when an experiment reaches its job cap.
- **Operational scripts (`script/run_scheduler.py`, `script/run_worker.py`)** – CLI shims that wire up Loguru/Rich logging, settings, the Redis broker, and the scheduler/worker entrypoints.
- **Docs (`docs/`)** – focused guides for configuration, scheduler behaviour, worker operations, and the MAP-Elites pipeline.

Module-level documentation lives under `docs/loreley/**` and `docs/script/**`. The rendered site is built with MkDocs into `site/`.

---

## Requirements & Tooling

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) for dependency management
- PostgreSQL and Redis (Dramatiq broker)
- Git (worktrees, LFS optional)
- Access to configured external planning/coding/evaluation agents

---

## Quick Start

### 1. Clone and install

```bash
git clone <YOUR_FORK_OR_ORIGIN_URL> loreley
cd loreley
uv sync          # installs according to pyproject.toml / uv.lock
```

If you already have an environment, pin dependencies without creating a workspace:

```bash
uv sync --no-workspace
```

### 2. Configure

All runtime settings come from environment variables consumed by `loreley.config.Settings`. Common examples:

- **Core app**
  - `APP_NAME`
  - `APP_ENV` (environment name, for example `development` / `staging` / `production`)
  - `LOG_LEVEL` (Loguru and Dramatiq log level)
- **Database**
  - `DATABASE_URL`
  - or individual `DB_SCHEME`, `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, `DB_POOL_SIZE`, `DB_MAX_OVERFLOW`, `DB_POOL_TIMEOUT`, `DB_ECHO`
- **Task queue / Dramatiq**
  - `TASKS_REDIS_URL` or (`TASKS_REDIS_HOST`, `TASKS_REDIS_PORT`, `TASKS_REDIS_DB`, `TASKS_REDIS_PASSWORD`)
  - `TASKS_REDIS_NAMESPACE`, `TASKS_QUEUE_NAME`
  - `TASKS_WORKER_MAX_RETRIES`, `TASKS_WORKER_TIME_LIMIT_SECONDS`
- **Scheduler**
  - `SCHEDULER_REPO_ROOT`
  - `SCHEDULER_POLL_INTERVAL_SECONDS`
  - `SCHEDULER_MAX_UNFINISHED_JOBS`, `SCHEDULER_MAX_TOTAL_JOBS`
  - `SCHEDULER_SCHEDULE_BATCH_SIZE`, `SCHEDULER_DISPATCH_BATCH_SIZE`, `SCHEDULER_INGEST_BATCH_SIZE`
- **Worker repository**
  - `WORKER_REPO_REMOTE_URL`, `WORKER_REPO_BRANCH`, `WORKER_REPO_WORKTREE`
  - `WORKER_REPO_ENABLE_LFS`, `WORKER_REPO_FETCH_DEPTH`, `WORKER_REPO_JOB_BRANCH_PREFIX`, `WORKER_REPO_JOB_BRANCH_TTL_HOURS`
- **Worker planning / coding**
  - `WORKER_PLANNING_CODEX_*` and `WORKER_CODING_CODEX_*` options for external planning/coding agents
- **Evaluator**
  - `WORKER_EVALUATOR_PLUGIN`, `WORKER_EVALUATOR_PYTHON_PATHS`
  - `WORKER_EVALUATOR_TIMEOUT_SECONDS`, `WORKER_EVALUATOR_MAX_METRICS`
- **Evolution objective**
  - `WORKER_EVOLUTION_GLOBAL_GOAL` – plain-language global objective shared across jobs
- **MAP-Elites**
  - `MAPELITES_EXPERIMENT_ROOT_COMMIT` – optional experiment root commit used for seeding
  - `MAPELITES_PREPROCESS_*`, `MAPELITES_CHUNK_*`, `MAPELITES_CODE_EMBEDDING_*`, `MAPELITES_SUMMARY_*`, `MAPELITES_SUMMARY_EMBEDDING_*`
  - `MAPELITES_DIMENSION_REDUCTION_*`, `MAPELITES_FEATURE_*`, `MAPELITES_ARCHIVE_*`, `MAPELITES_FITNESS_*`, `MAPELITES_SAMPLER_*`, `MAPELITES_SEED_POPULATION_SIZE`

See `docs/loreley/config.md` for the exhaustive list.

---

## Running Loreley

### Scheduler loop

The scheduler drives ingestion, scheduling, dispatch, seeding, and archive maintenance:

```bash
uv run python script/run_scheduler.py        # continuous loop
uv run python script/run_scheduler.py --once # single tick

# or invoke the module directly
uv run python -m loreley.scheduler.main        # continuous loop
uv run python -m loreley.scheduler.main --once # single tick
```

See `docs/script/run_scheduler.md` and `docs/loreley/scheduler/main.md` for details on scheduler behaviour and configuration.

### Worker process

A worker process consumes jobs from Dramatiq, applies planning/coding/evaluation, and pushes results back into the database:

```bash
uv run python script/run_worker.py
```

The worker configures Loguru/Rich logging, initialises the Redis broker defined in `loreley.tasks.broker`, imports `loreley.tasks.workers`, and launches a single-threaded Dramatiq worker bound to `TASKS_QUEUE_NAME`.

See `docs/script/run_worker.md`, `docs/loreley/core/worker/evolution.md`, and the other worker module docs under `docs/loreley/core/worker/` for deeper operational guidance.

---

## Documentation Map

The full documentation lives under `docs/` and is rendered into `site/` via MkDocs. Useful entry points:

- `docs/index.md` – high-level overview and navigation.
- `docs/loreley/config.md` – global settings and environment variables.
- `docs/loreley/db/` – database engine/sessions and ORM models.
- `docs/loreley/core/map-elites/` – MAP-Elites pipeline (preprocessing, chunking, embeddings, dimensionality reduction, sampler, snapshots, and summary embeddings).
- `docs/loreley/core/worker/` – planning, coding, evaluator, evolution loop, commit summaries, job store, and worker repository.
- `docs/loreley/scheduler/main.md` – scheduler internals and configuration.
- `docs/loreley/tasks/` – Redis broker and Dramatiq actors.
- `docs/script/` – CLI wrappers for the scheduler and worker.

---

## Project Layout

- `loreley/` – core services (`config`, `db`, `core/map_elites`, `core/worker`, `scheduler`, `tasks`).
- `script/` – CLI shims (`run_scheduler.py`, `run_worker.py`).
- `docs/` – module-level docs under `docs/loreley` and `docs/script`, rendered into `site/`.
- `pyproject.toml`, `uv.lock` – dependency definitions for `uv`.
- `examples/` – self-contained optimisation examples used for testing and demos.

---

## Examples

- **`examples/circle-packing`** – a geometric optimisation benchmark based on the classical
  [circle packing](https://en.wikipedia.org/wiki/Circle_packing) problem. The example defines:
  - a simple solution interface in `solution.py` that returns a set of equal-radius circles
    inside the unit square;
  - an evaluator plugin in `evaluate.py` that checks geometric validity (no overlap, inside
    bounds) and reports packing density and related metrics.

  You can wire this into the worker by pointing
  `WORKER_EVALUATOR_PYTHON_PATHS` at `examples/circle-packing` and setting
  `WORKER_EVALUATOR_PLUGIN=evaluate:plugin`, then letting MAP-Elites evolve better packings.

---

## License

See `LICENSE` for details.
